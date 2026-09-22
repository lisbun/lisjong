"""lisjong所有のplayer-safe model-facing feature representation。

Issue #184のL0bに対応する。learned Policyがmodelへ渡す固定長の実数表現を、
`PolicyInput`から観測可能な情報だけから決定的に構築する。

```text
player-safe PolicyInput
    -> fixed-size deterministic feature vector
```

identityはhistorical Arena experimentのものと別である。

```text
arena-policy-input-feature-v1   historical Arena experiment identity (immutable)
lisjong-offense-l0-player-safe-feature-v1   このmoduleが所有するidentity
```

historical identity、historical dimension（8204）、historical layoutは再利用・
改名・再定義しない。本moduleはoffense L0 use caseに必要な最小のblockだけを
持ち、将来のblock追加・layout変更は新しいidentityとfingerprintで扱う。

固定する原則。

- 入力は`PolicyInput`だけである。`DecisionContext`、legal actions、engine
  state、HandBelief、shanten / ukeire等のPolicy-internal analysis、opponentの
  非公開手牌、wall / dead wallの実配列、future event、training-only ground
  truthは受け取らない。legal actionsはmodel出力側のlegal maskが扱う
  （`lisjong.action_vocabulary`）
- 決定性: 同じ意味内容の`PolicyInput`は常に同じvectorへmaterializeされる
- 順序は明示的なaxis定義だけから決まる。Enum定義順、dict iteration order、
  `list(Enum).index(...)`等の偶然の順序へ依存しない
- seat axisはactor相対（self / 下家 / 対面 / 上家）である。absolute seat
  identityをmodel入力へ持ち込まない
- 物理的に成立しない入力はclipせずfail closedする。牌種ごとの可視枚数が4枚を
  超える、副露が5つある、といったtile conservation違反をfeature側で丸めない
"""

from dataclasses import dataclass

from lisjong.belief.canonical_axes import red_five_index, tile_type_index, wind_for_seat
from lisjong.learning._canonical import value_digest
from lisjong.learning.errors import FeatureError
from lisjong.policy_contract import (
    MeldKind,
    PlayerPublicState,
    PolicyInput,
    RiichiState,
    Tile,
    Wind,
)

FEATURE_IDENTITY = "lisjong-offense-l0-player-safe-feature-v1"
"""このfeature representationのidentity。layoutや意味を変える場合は新identityにする。"""

TILE_TYPE_AXIS_SIZE = 34
RED_FIVE_AXIS_SIZE = 3

RELATIVE_SEAT_AXIS = ("self", "shimocha", "toimen", "kamicha")
"""actor相対seat axis。index rはabsolute seat `(self_seat + r) % 4`を表す。"""

WIND_AXIS = (Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH)
RIICHI_AXIS = (RiichiState.NONE, RiichiState.DECLARED, RiichiState.ACCEPTED)
MELD_KIND_AXIS = (
    MeldKind.CHI,
    MeldKind.PON,
    MeldKind.DAIMINKAN,
    MeldKind.ANKAN,
    MeldKind.KAKAN,
)

TILE_COUNT_SCALE = 4.0
DISCARD_COUNT_SCALE = 24.0
MELD_COUNT_SCALE = 4.0
HAND_NUMBER_SCALE = 4.0
HONBA_SCALE = 8.0
RIICHI_STICK_SCALE = 8.0
LIVE_WALL_SCALE = 70.0
SCORE_SCALE = 25000.0

MAX_TILES_PER_TYPE = 4
MAX_CONCEALED_TILES = 14
MAX_MELDS_PER_PLAYER = 4
MAX_DORA_INDICATORS = 5
MAX_LIVE_WALL_TILES = 136

_RIICHI_AXIS_INDEX = {state: index for index, state in enumerate(RIICHI_AXIS)}
_MELD_KIND_AXIS_INDEX = {kind: index for index, kind in enumerate(MELD_KIND_AXIS)}
_WIND_AXIS_INDEX = {wind: index for index, wind in enumerate(WIND_AXIS)}


@dataclass(frozen=True, slots=True)
class FeatureBlock:
    """feature vector内の1 blockのlayout記述。identityではない。"""

    name: str
    offset: int
    size: int
    axis: str
    scale: float | None


def _build_layout() -> tuple[FeatureBlock, ...]:
    """block定義をoffset付きのlayoutへ展開する。"""
    definitions: list[tuple[str, int, str, float | None]] = [
        ("own_concealed_counts", TILE_TYPE_AXIS_SIZE, "tile_type", TILE_COUNT_SCALE),
        ("own_concealed_red_fives", RED_FIVE_AXIS_SIZE, "red_five", TILE_COUNT_SCALE),
        ("own_drawn_tile_type", TILE_TYPE_AXIS_SIZE, "tile_type", None),
        ("own_drawn_tile_red_five", RED_FIVE_AXIS_SIZE, "red_five", None),
        ("own_has_drawn_tile", 1, "scalar", None),
    ]
    for relative, seat_name in enumerate(RELATIVE_SEAT_AXIS):
        prefix = f"player[{relative}:{seat_name}]"
        definitions.extend(
            [
                (
                    f"{prefix}.discard_counts",
                    TILE_TYPE_AXIS_SIZE,
                    "tile_type",
                    TILE_COUNT_SCALE,
                ),
                (
                    f"{prefix}.discard_red_fives",
                    RED_FIVE_AXIS_SIZE,
                    "red_five",
                    TILE_COUNT_SCALE,
                ),
                (f"{prefix}.riichi_state", len(RIICHI_AXIS), "riichi_state", None),
                (
                    f"{prefix}.meld_kind_counts",
                    len(MELD_KIND_AXIS),
                    "meld_kind",
                    MELD_COUNT_SCALE,
                ),
                (f"{prefix}.discard_count", 1, "scalar", DISCARD_COUNT_SCALE),
                (f"{prefix}.score", 1, "scalar", SCORE_SCALE),
            ]
        )
    definitions.extend(
        [
            ("round_wind", len(WIND_AXIS), "wind", None),
            ("own_seat_wind", len(WIND_AXIS), "wind", None),
            ("own_is_dealer", 1, "scalar", None),
            ("hand_number", 1, "scalar", HAND_NUMBER_SCALE),
            ("honba", 1, "scalar", HONBA_SCALE),
            ("riichi_sticks", 1, "scalar", RIICHI_STICK_SCALE),
            ("live_wall_tiles_remaining", 1, "scalar", LIVE_WALL_SCALE),
            (
                "dora_indicator_counts",
                TILE_TYPE_AXIS_SIZE,
                "tile_type",
                TILE_COUNT_SCALE,
            ),
            (
                "dora_indicator_red_fives",
                RED_FIVE_AXIS_SIZE,
                "red_five",
                TILE_COUNT_SCALE,
            ),
            ("visible_tile_counts", TILE_TYPE_AXIS_SIZE, "tile_type", TILE_COUNT_SCALE),
            ("visible_red_fives", RED_FIVE_AXIS_SIZE, "red_five", TILE_COUNT_SCALE),
        ]
    )

    blocks: list[FeatureBlock] = []
    offset = 0
    for name, size, axis, scale in definitions:
        blocks.append(
            FeatureBlock(name=name, offset=offset, size=size, axis=axis, scale=scale)
        )
        offset += size
    return tuple(blocks)


FEATURE_LAYOUT = _build_layout()
"""feature vectorのblock layout。read-onlyな記述であり、identityではない。"""

FEATURE_DIMENSION = sum(block.size for block in FEATURE_LAYOUT)
"""固定長feature vectorの次元数。decisionによって変化しない。"""

FEATURE_BLOCK_OFFSETS = {block.name: block.offset for block in FEATURE_LAYOUT}


def feature_specification() -> dict[str, object]:
    """fingerprint対象となるfeature contractのcanonical記述を返す。

    identity、dimension、block layout、axis定義、normalization、物理boundを
    すべて含める。いずれかを変更するとfingerprintが変わる。
    """
    return {
        "axes": {
            "meld_kind": [kind.value for kind in MELD_KIND_AXIS],
            "red_five": ["5m", "5p", "5s"],
            "relative_seat": list(RELATIVE_SEAT_AXIS),
            "riichi_state": [state.value for state in RIICHI_AXIS],
            "tile_type": "lisjong-canonical-34-tile-type-index",
            "wind": [wind.value for wind in WIND_AXIS],
        },
        "blocks": [
            {
                "axis": block.axis,
                "name": block.name,
                "offset": block.offset,
                "scale": block.scale,
                "size": block.size,
            }
            for block in FEATURE_LAYOUT
        ],
        "bounds": {
            "max_concealed_tiles": MAX_CONCEALED_TILES,
            "max_dora_indicators": MAX_DORA_INDICATORS,
            "max_live_wall_tiles": MAX_LIVE_WALL_TILES,
            "max_melds_per_player": MAX_MELDS_PER_PLAYER,
            "max_tiles_per_type": MAX_TILES_PER_TYPE,
        },
        "dimension": FEATURE_DIMENSION,
        "identity": FEATURE_IDENTITY,
    }


def feature_fingerprint() -> str:
    """feature contractのfingerprintを返す。artifactはこの値をbindする。"""
    return value_digest(feature_specification())


def _add_tile(
    counts: list[int], red_counts: list[int], tile: Tile, context: str
) -> None:
    index = tile_type_index(tile.tile_type)
    counts[index] += 1
    if counts[index] > MAX_TILES_PER_TYPE:
        raise FeatureError(
            f"{context} exceeds the physical count of {MAX_TILES_PER_TYPE} "
            "tiles for one tile type"
        )
    if tile.is_red:
        red_counts[red_five_index(tile.tile_type.category)] += 1


def _tile_counts(tiles: tuple[Tile, ...], context: str) -> tuple[list[int], list[int]]:
    counts = [0] * TILE_TYPE_AXIS_SIZE
    red_counts = [0] * RED_FIVE_AXIS_SIZE
    for tile in tiles:
        _add_tile(counts, red_counts, tile, context)
    return counts, red_counts


def _write_counts(
    values: list[float],
    block: FeatureBlock,
    counts: list[int],
) -> None:
    if block.scale is None:
        raise FeatureError(f"{block.name} is not a normalized count block")
    for index, count in enumerate(counts):
        values[block.offset + index] = count / block.scale


def _require_player(player: object, context: str) -> PlayerPublicState:
    if not isinstance(player, PlayerPublicState):
        raise FeatureError(f"{context} must be a PlayerPublicState")
    if len(player.melds) > MAX_MELDS_PER_PLAYER:
        raise FeatureError(f"{context} has more than {MAX_MELDS_PER_PLAYER} melds")
    return player


def build_player_safe_feature(policy_input: object) -> tuple[float, ...]:
    """`PolicyInput`から固定長のplayer-safe feature vectorを構築する。

    返り値は長さ`FEATURE_DIMENSION`のfloat tupleである。ML runtimeへは依存
    しない。tensor化はconsumer（trainer / learned inference）の責務とする。
    """
    if not isinstance(policy_input, PolicyInput):
        raise FeatureError("policy_input must be a PolicyInput")

    round_state = policy_input.round
    own_hand = policy_input.own_hand
    if len(own_hand.concealed_tiles) > MAX_CONCEALED_TILES:
        raise FeatureError(
            f"own_hand has more than {MAX_CONCEALED_TILES} concealed tiles"
        )
    if len(round_state.dora_indicators) > MAX_DORA_INDICATORS:
        raise FeatureError(f"round has more than {MAX_DORA_INDICATORS} dora indicators")
    if round_state.live_wall_tiles_remaining > MAX_LIVE_WALL_TILES:
        raise FeatureError(f"live_wall_tiles_remaining exceeds {MAX_LIVE_WALL_TILES}")

    values = [0.0] * FEATURE_DIMENSION
    blocks = {block.name: block for block in FEATURE_LAYOUT}

    # 可視牌はtile conservation上の1つのmultisetとして数える。副露へ吸収された
    # discard（`called_by`が設定されたdiscard）を二重に数えないことで、牌種ごとの
    # 可視枚数が物理的な4枚制約を超えないことを検査できる。
    visible_counts = [0] * TILE_TYPE_AXIS_SIZE
    visible_red_counts = [0] * RED_FIVE_AXIS_SIZE

    own_counts, own_red_counts = _tile_counts(
        own_hand.concealed_tiles, "own_hand.concealed_tiles"
    )
    _write_counts(values, blocks["own_concealed_counts"], own_counts)
    _write_counts(values, blocks["own_concealed_red_fives"], own_red_counts)
    for tile in own_hand.concealed_tiles:
        _add_tile(visible_counts, visible_red_counts, tile, "visible tiles")

    if own_hand.drawn_tile is not None:
        drawn = own_hand.drawn_tile
        values[
            blocks["own_drawn_tile_type"].offset + tile_type_index(drawn.tile_type)
        ] = 1.0
        if drawn.is_red:
            values[
                blocks["own_drawn_tile_red_five"].offset
                + red_five_index(drawn.tile_type.category)
            ] = 1.0
        values[blocks["own_has_drawn_tile"].offset] = 1.0

    for relative, seat_name in enumerate(RELATIVE_SEAT_AXIS):
        seat = (int(policy_input.self_seat) + relative) % 4
        prefix = f"player[{relative}:{seat_name}]"
        player = _require_player(policy_input.players[seat], prefix)

        discard_tiles = tuple(discard.tile for discard in player.discards)
        discard_counts, discard_red_counts = _tile_counts(
            discard_tiles, f"{prefix}.discards"
        )
        _write_counts(values, blocks[f"{prefix}.discard_counts"], discard_counts)
        _write_counts(values, blocks[f"{prefix}.discard_red_fives"], discard_red_counts)
        values[
            blocks[f"{prefix}.riichi_state"].offset + _RIICHI_AXIS_INDEX[player.riichi]
        ] = 1.0

        meld_counts = [0] * len(MELD_KIND_AXIS)
        for meld in player.melds:
            meld_counts[_MELD_KIND_AXIS_INDEX[meld.kind]] += 1
            for tile in meld.tiles:
                _add_tile(visible_counts, visible_red_counts, tile, "visible tiles")
        _write_counts(values, blocks[f"{prefix}.meld_kind_counts"], meld_counts)

        for discard in player.discards:
            if discard.called_by is None:
                _add_tile(
                    visible_counts, visible_red_counts, discard.tile, "visible tiles"
                )

        values[blocks[f"{prefix}.discard_count"].offset] = (
            len(player.discards) / DISCARD_COUNT_SCALE
        )
        values[blocks[f"{prefix}.score"].offset] = player.score / SCORE_SCALE

    values[blocks["round_wind"].offset + _WIND_AXIS_INDEX[round_state.round_wind]] = 1.0
    own_wind = wind_for_seat(policy_input.self_seat, round_state.dealer_seat)
    values[blocks["own_seat_wind"].offset + _WIND_AXIS_INDEX[own_wind]] = 1.0
    if policy_input.self_seat == round_state.dealer_seat:
        values[blocks["own_is_dealer"].offset] = 1.0
    values[blocks["hand_number"].offset] = round_state.hand_number / HAND_NUMBER_SCALE
    values[blocks["honba"].offset] = round_state.honba / HONBA_SCALE
    values[blocks["riichi_sticks"].offset] = (
        round_state.riichi_sticks / RIICHI_STICK_SCALE
    )
    values[blocks["live_wall_tiles_remaining"].offset] = (
        round_state.live_wall_tiles_remaining / LIVE_WALL_SCALE
    )

    dora_counts, dora_red_counts = _tile_counts(
        round_state.dora_indicators, "round.dora_indicators"
    )
    _write_counts(values, blocks["dora_indicator_counts"], dora_counts)
    _write_counts(values, blocks["dora_indicator_red_fives"], dora_red_counts)
    for tile in round_state.dora_indicators:
        _add_tile(visible_counts, visible_red_counts, tile, "visible tiles")

    _write_counts(values, blocks["visible_tile_counts"], visible_counts)
    _write_counts(values, blocks["visible_red_fives"], visible_red_counts)

    return tuple(values)


__all__ = [
    "FEATURE_BLOCK_OFFSETS",
    "FEATURE_DIMENSION",
    "FEATURE_IDENTITY",
    "FEATURE_LAYOUT",
    "MELD_KIND_AXIS",
    "RELATIVE_SEAT_AXIS",
    "RIICHI_AXIS",
    "WIND_AXIS",
    "FeatureBlock",
    "build_player_safe_feature",
    "feature_fingerprint",
    "feature_specification",
]
