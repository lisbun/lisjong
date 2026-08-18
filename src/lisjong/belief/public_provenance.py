"""公開済み牌情報のcanonical exact-count provenance feature。

Issue #61を実装する。既存semantic state
（`PlayerPublicState.discards` / `PlayerPublicState.melds`、
`RoundState.dora_indicators`）を唯一の正本とし、`PolicyInput`全体から
毎回full recomputationするpure / deterministicなencoderである。
incremental update、mutable feature cache、dirty flag、event-driven
synchronizationは実装しない。将来incremental実装を追加する場合も、
本moduleのfull recomputation結果と完全一致することを要求する。

Issue #59の`HandBelief`（`expected_count` / `red_five_probability`という
推定値）とはsemanticが異なる。本moduleが表すのはbeliefではなく、実際に
観測された牌についてのexact observed integer countであり、tileの
current physical locationではなくtile provenance（どのplayerの打牌・
手牌由来か）を表す。

Issue #59のWind axis・34牌種axis・赤5 axisをそのまま再利用し、mapping自体は
複製しない。flattened representationのWind-major / row-major layoutも#59と
同一である。

```text
tile:     offset = wind_index * 34 + tile_type_index
red-five: offset = wind_index * 3  + red_five_index
```

exact countを#59のfixed-point domainへ変換する場合は、
`fixed_point_mass = exact_count * SCALE`でlosslessに変換できる。
"""

from dataclasses import dataclass

from lisjong.belief.canonical_axes import (
    red_five_index,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

BASE_TILE_COUNT_MAX = 4
RED_FIVE_COUNT_MAX = 1

_SUITED_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)


def _normalize_count_tuple(
    values: object, expected_length: int, max_count: int, field_name: str
) -> tuple[int, ...]:
    try:
        counts = tuple(values)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable of int") from None
    if len(counts) != expected_length:
        raise ValueError(f"{field_name} must contain exactly {expected_length} values")
    for count in counts:
        if type(count) is not int:
            raise TypeError(f"{field_name} must contain only int values")
        if not 0 <= count <= max_count:
            raise ValueError(f"{field_name} values must be between 0 and {max_count}")
    return counts


@dataclass(frozen=True, slots=True)
class TileProvenanceCounts:
    """34基本牌種のexact observed countと、赤5 companion count。

    ```text
    TileProvenanceCounts
    ├── tile_counts     (length 34, canonical tile_type_index順, 0..4)
    └── red_five_counts (length 3, canonical red_five_index順, 0..1)
    ```

    `tile_counts`側は通常5と赤5を合算した値であり、`red_five_counts`は
    そのうち赤5だった分だけを補足する。各色について
    `red_five_counts <= 対応する5のtile_counts`を局所的に検証する。これは
    このfeature単体で完結する不変条件であり、複数feature/stateを横断する
    牌保存則（discard + meld + dora + concealed <= 4等）は対象外とする。
    """

    tile_counts: tuple[int, ...]
    red_five_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        tile_counts = _normalize_count_tuple(
            self.tile_counts, 34, BASE_TILE_COUNT_MAX, "tile_counts"
        )
        red_five_counts = _normalize_count_tuple(
            self.red_five_counts, 3, RED_FIVE_COUNT_MAX, "red_five_counts"
        )

        for category in _SUITED_CATEGORIES:
            five_count = tile_counts[tile_type_index(TileType(category, 5))]
            red_count = red_five_counts[red_five_index(category)]
            if red_count > five_count:
                raise ValueError(
                    "red_five_counts must not exceed the corresponding "
                    "five's tile_counts"
                )

        object.__setattr__(self, "tile_counts", tile_counts)
        object.__setattr__(self, "red_five_counts", red_five_counts)

    def tile_count(self, tile_type: TileType) -> int:
        """基本牌種`tile_type`のexact observed countを返す。"""
        return self.tile_counts[tile_type_index(tile_type)]

    def red_five_count(self, category: TileCategory) -> int:
        """数牌`category`の赤5 exact observed countを返す。"""
        return self.red_five_counts[red_five_index(category)]


@dataclass(frozen=True, slots=True)
class WindTileProvenanceCounts:
    """4 wind分の`TileProvenanceCounts`をcanonical wind_index順に束ねる。

    `flattened_tile_counts` / `flattened_red_five_counts`は、Issue #59と
    同じWind-major / row-major flattened layoutであり、offsetは
    `canonical_axes.concealed_hand_offset()` / `canonical_axes.red_five_offset()`
    と一致する。
    """

    winds: tuple[
        TileProvenanceCounts,
        TileProvenanceCounts,
        TileProvenanceCounts,
        TileProvenanceCounts,
    ]

    def __post_init__(self) -> None:
        try:
            winds = tuple(self.winds)
        except TypeError:
            raise TypeError(
                "winds must be an iterable of TileProvenanceCounts"
            ) from None
        if len(winds) != 4:
            raise ValueError("winds must contain exactly 4 TileProvenanceCounts")
        if any(
            not isinstance(wind_counts, TileProvenanceCounts) for wind_counts in winds
        ):
            raise TypeError("winds must contain only TileProvenanceCounts instances")
        object.__setattr__(self, "winds", winds)

    def counts(self, wind: Wind) -> TileProvenanceCounts:
        """`wind`のTileProvenanceCountsを返す。"""
        return self.winds[wind_index(wind)]

    def tile_count(self, wind: Wind, tile_type: TileType) -> int:
        """`wind`について、`tile_type`のexact observed countを返す。"""
        return self.counts(wind).tile_count(tile_type)

    def red_five_count(self, wind: Wind, category: TileCategory) -> int:
        """`wind`について、`category`の赤5 exact observed countを返す。"""
        return self.counts(wind).red_five_count(category)

    @property
    def flattened_tile_counts(self) -> tuple[int, ...]:
        """Wind-major / row-majorのflattened count buffer（length 136）。"""
        return tuple(
            count for wind_counts in self.winds for count in wind_counts.tile_counts
        )

    @property
    def flattened_red_five_counts(self) -> tuple[int, ...]:
        """Wind-major / row-majorのflattened count buffer（length 12）。"""
        return tuple(
            count for wind_counts in self.winds for count in wind_counts.red_five_counts
        )


@dataclass(frozen=True, slots=True)
class PublicTileProvenance:
    """公開済み牌情報から導出したcanonical exact-count provenance feature。

    ```text
    PublicTileProvenance
    ├── discards           (WindTileProvenanceCounts)
    ├── meld_hand_origin    (WindTileProvenanceCounts)
    └── dora_indicators     (TileProvenanceCounts)
    ```

    `discards`は各windが実際に捨てた牌のexact count（鳴かれた牌も含む）、
    `meld_hand_origin`は各windの副露・槓のうち、そのwind自身の手牌に由来すると
    確定している構成牌だけのexact countである。`dora_indicators`は現在
    公開されているドラ表示牌そのもののexact countであり、ドラ牌への変換結果
    ではない。

    `PlayerPublicState.discards` / `PublicMeld` / `RoundState.dora_indicators`
    が持つ順序・巡目・手出しツモ切り・鳴き種別等のsemantic structureは
    置き換えない。
    """

    discards: WindTileProvenanceCounts
    meld_hand_origin: WindTileProvenanceCounts
    dora_indicators: TileProvenanceCounts

    def __post_init__(self) -> None:
        if not isinstance(self.discards, WindTileProvenanceCounts):
            raise TypeError("discards must be a WindTileProvenanceCounts")
        if not isinstance(self.meld_hand_origin, WindTileProvenanceCounts):
            raise TypeError("meld_hand_origin must be a WindTileProvenanceCounts")
        if not isinstance(self.dora_indicators, TileProvenanceCounts):
            raise TypeError("dora_indicators must be a TileProvenanceCounts")


def _add_tile(tile_counts: list[int], red_five_counts: list[int], tile: Tile) -> None:
    tile_counts[tile_type_index(tile.tile_type)] += 1
    if tile.is_red:
        red_five_counts[red_five_index(tile.tile_type.category)] += 1


def _meld_hand_origin_tiles(meld: PublicMeld) -> tuple[Tile, ...]:
    """meld ownerの手牌に由来すると確定している構成牌だけを返す。

    ANKANは4枚すべてがhand-originである。それ以外のkindは、`meld.tiles`の
    semantic multisetから`meld.called_tile`をexactly one occurrenceだけ
    除外する。`Tile`はphysical copy identityを持たないため、
    `tile != meld.called_tile`のようなvalue filterは同値牌をすべて除外して
    しまい、通常7pのPON等で3枚全部が消える。`list.remove()`は最初に一致した
    1件だけを取り除くため、この契約を満たす。
    """
    if meld.kind is MeldKind.ANKAN:
        return meld.tiles

    tiles = list(meld.tiles)
    tiles.remove(meld.called_tile)
    return tuple(tiles)


def encode_public_tile_provenance(policy_input: PolicyInput) -> PublicTileProvenance:
    """`PolicyInput`の公開済み牌情報から`PublicTileProvenance`をfull recomputationする。

    `policy_input.players`のindexをそのままcanonical Wind orderとしては
    扱わない。`policy_input.round.dealer_seat`と`wind_for_seat()`で各seatの
    自風を明示的に解決してから集計する。この関数はhidden information、
    HandBelief estimator、RiichiEnv / RiichiLab固有型へ依存しない、
    既存semantic snapshotからのpure projectionである。
    """
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("policy_input must be a PolicyInput")

    dealer_seat = policy_input.round.dealer_seat

    discard_tile_counts = [[0] * 34 for _ in range(4)]
    discard_red_five_counts = [[0, 0, 0] for _ in range(4)]
    meld_tile_counts = [[0] * 34 for _ in range(4)]
    meld_red_five_counts = [[0, 0, 0] for _ in range(4)]

    for seat_number, player in enumerate(policy_input.players):
        wind_number = wind_index(wind_for_seat(Seat(seat_number), dealer_seat))

        for discard in player.discards:
            _add_tile(
                discard_tile_counts[wind_number],
                discard_red_five_counts[wind_number],
                discard.tile,
            )

        for meld in player.melds:
            for tile in _meld_hand_origin_tiles(meld):
                _add_tile(
                    meld_tile_counts[wind_number],
                    meld_red_five_counts[wind_number],
                    tile,
                )

    dora_tile_counts = [0] * 34
    dora_red_five_counts = [0, 0, 0]
    for indicator in policy_input.round.dora_indicators:
        _add_tile(dora_tile_counts, dora_red_five_counts, indicator)

    discards = WindTileProvenanceCounts(
        winds=tuple(
            TileProvenanceCounts(
                tile_counts=tuple(discard_tile_counts[wind_number]),
                red_five_counts=tuple(discard_red_five_counts[wind_number]),
            )
            for wind_number in range(4)
        )
    )
    meld_hand_origin = WindTileProvenanceCounts(
        winds=tuple(
            TileProvenanceCounts(
                tile_counts=tuple(meld_tile_counts[wind_number]),
                red_five_counts=tuple(meld_red_five_counts[wind_number]),
            )
            for wind_number in range(4)
        )
    )
    dora_indicators = TileProvenanceCounts(
        tile_counts=tuple(dora_tile_counts), red_five_counts=tuple(dora_red_five_counts)
    )

    return PublicTileProvenance(
        discards=discards,
        meld_hand_origin=meld_hand_origin,
        dora_indicators=dora_indicators,
    )


__all__ = [
    "BASE_TILE_COUNT_MAX",
    "RED_FIVE_COUNT_MAX",
    "PublicTileProvenance",
    "TileProvenanceCounts",
    "WindTileProvenanceCounts",
    "encode_public_tile_provenance",
]
