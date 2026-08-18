"""観測済みprovenanceから牌保存則を検証し、remaining tile inventoryを導出する。

Issue #63を実装する。self concealed hand（`OwnHandState.concealed_tiles`を
直接exact countする）と、Issue #61の`encode_public_tile_provenance()`が
導出するdiscard / meld hand-origin / dora indicator provenanceを合算した
`exact accounted counts`を、Issue #63の`tile_inventory`が定義するstandard
physical tile inventoryから差し引き、`remaining tile counts`をfail-closedに
導出するpure / deterministicなencoderである。

`remaining tile inventory`は山（live wall）ではない。他家のconcealed hand、
live wall、dead wall、未開示の裏ドラ表示牌等をまとめた残余inventoryであり、
Wind axisやowner / location情報を持たない。不確実性分布でもなく、exact
accounting後に残ったphysical tile countである。

self exact countは`HandBelief` / `exact_self_belief()`を経由しない。
`HandBelief`のred-five companionはprobabilityであり、`OwnHandState`内に
不正な同色赤5重複が存在した場合にmultiplicityを保存則検証から隠して
しまうおそれがあるため、牌保存則の入力としては使用しない。

incremental update、mutable remaining-inventory cache、dirty flag、
event-driven synchronizationは実装しない。将来incremental実装を追加する
場合も、本moduleのfull recomputation結果と完全一致することを要求する。
"""

from dataclasses import dataclass

from lisjong.belief.canonical_axes import red_five_index, tile_type_index
from lisjong.belief.public_provenance import encode_public_tile_provenance
from lisjong.belief.tile_inventory import (
    RED_FIVE_AXIS_COUNT,
    STANDARD_RED_FIVE_COUNTS,
    STANDARD_TILE_COUNTS,
    TILE_TYPE_COUNT,
)
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import TileCategory, TileType
from lisjong.policy_contract.wind import Wind

_SUITED_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)


def _normalize_non_negative_tuple(
    values: object, expected_length: int, field_name: str
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
        if count < 0:
            raise ValueError(f"{field_name} must not contain negative values")
    return counts


@dataclass(frozen=True, slots=True)
class TileConservationResult:
    """牌保存則検証結果。exact accounted provenanceとremaining inventoryを
    exact integerで束ねる。

    ```text
    TileConservationResult
    ├── exact_accounted_counts           (length 34)
    ├── exact_accounted_red_five_counts  (length 3)
    ├── remaining_tile_counts            (length 34)
    └── remaining_red_five_counts        (length 3)
    ```

    いずれもnon-negative exact integerであり、beliefやprobabilityではない。
    `remaining_tile_counts` / `remaining_red_five_counts`にWind axisや
    owner / location情報は持たせない。

    生成時に、`tile_inventory`のstandard physical inventoryを基準として
    以下をfail-closedに検証する。

    - 各基本牌種について`accounted + remaining == inventory`をexactに満たす
      （accountedがinventoryを超える、remainingが負になる状態を拒否する）
    - 各色について`accounted_red + remaining_red == inventory_red`をexactに
      満たす
    - 各色について`accounted_red <= 対応するaccounted_five`
    - 各色について`remaining_red <= 対応するremaining_five`

    最後の条件により、例えば`accounted 5m = 4, accounted red5m = 0`から
    導かれる`remaining 5m = 0, remaining red5m = 1`のような、standard
    physical inventoryと矛盾するstateを拒否する
    （`remaining_red5m(1) <= remaining_5m(0)`を満たさない）。
    """

    exact_accounted_counts: tuple[int, ...]
    exact_accounted_red_five_counts: tuple[int, ...]
    remaining_tile_counts: tuple[int, ...]
    remaining_red_five_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        accounted = _normalize_non_negative_tuple(
            self.exact_accounted_counts, TILE_TYPE_COUNT, "exact_accounted_counts"
        )
        accounted_red = _normalize_non_negative_tuple(
            self.exact_accounted_red_five_counts,
            RED_FIVE_AXIS_COUNT,
            "exact_accounted_red_five_counts",
        )
        remaining = _normalize_non_negative_tuple(
            self.remaining_tile_counts, TILE_TYPE_COUNT, "remaining_tile_counts"
        )
        remaining_red = _normalize_non_negative_tuple(
            self.remaining_red_five_counts,
            RED_FIVE_AXIS_COUNT,
            "remaining_red_five_counts",
        )

        for index in range(TILE_TYPE_COUNT):
            if accounted[index] + remaining[index] != STANDARD_TILE_COUNTS[index]:
                raise ValueError(
                    "exact_accounted_counts and remaining_tile_counts must sum to "
                    "the standard physical inventory for every tile type"
                )

        for color in range(RED_FIVE_AXIS_COUNT):
            if (
                accounted_red[color] + remaining_red[color]
                != STANDARD_RED_FIVE_COUNTS[color]
            ):
                raise ValueError(
                    "exact_accounted_red_five_counts and remaining_red_five_counts "
                    "must sum to the standard red-five inventory for every color"
                )

        for category in _SUITED_CATEGORIES:
            five_index = tile_type_index(TileType(category, 5))
            color = red_five_index(category)
            if accounted_red[color] > accounted[five_index]:
                raise ValueError(
                    "exact_accounted_red_five_counts must not exceed the "
                    "corresponding accounted five count"
                )
            if remaining_red[color] > remaining[five_index]:
                raise ValueError(
                    "remaining_red_five_counts must not exceed the corresponding "
                    "remaining five count"
                )

        object.__setattr__(self, "exact_accounted_counts", accounted)
        object.__setattr__(self, "exact_accounted_red_five_counts", accounted_red)
        object.__setattr__(self, "remaining_tile_counts", remaining)
        object.__setattr__(self, "remaining_red_five_counts", remaining_red)


def _accumulate(target: list[int], source: tuple[int, ...]) -> None:
    for index, value in enumerate(source):
        target[index] += value


def derive_remaining_tile_inventory(
    policy_input: PolicyInput,
) -> TileConservationResult:
    """`PolicyInput`からexact accounted provenanceを合算し、standard physical
    tile inventoryから差し引いた`TileConservationResult`をfull recomputationする。

    self concealed handは`OwnHandState.concealed_tiles`を直接exact countし、
    `HandBelief` / `exact_self_belief()`は経由しない。`drawn_tile`は
    `concealed_tiles`内のmetadataなので別途加算しない。discard / meld
    hand-origin / dora indicatorはIssue #61の`encode_public_tile_provenance()`
    が持つcalled-tile二重count防止契約をそのまま利用する。

    `policy_input`が保持する完全game state以外の情報（他家の実手牌、live
    wall / dead wallの実配列、未開示裏ドラ表示牌等）は参照しない。
    """
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("policy_input must be a PolicyInput")

    accounted_tile_counts = [0] * TILE_TYPE_COUNT
    accounted_red_five_counts = [0] * RED_FIVE_AXIS_COUNT

    for tile in policy_input.own_hand.concealed_tiles:
        accounted_tile_counts[tile_type_index(tile.tile_type)] += 1
        if tile.is_red:
            accounted_red_five_counts[red_five_index(tile.tile_type.category)] += 1

    provenance = encode_public_tile_provenance(policy_input)

    for wind in Wind:
        discard_counts = provenance.discards.counts(wind)
        _accumulate(accounted_tile_counts, discard_counts.tile_counts)
        _accumulate(accounted_red_five_counts, discard_counts.red_five_counts)

        meld_counts = provenance.meld_hand_origin.counts(wind)
        _accumulate(accounted_tile_counts, meld_counts.tile_counts)
        _accumulate(accounted_red_five_counts, meld_counts.red_five_counts)

    _accumulate(accounted_tile_counts, provenance.dora_indicators.tile_counts)
    _accumulate(accounted_red_five_counts, provenance.dora_indicators.red_five_counts)

    remaining_tile_counts = tuple(
        STANDARD_TILE_COUNTS[index] - accounted_tile_counts[index]
        for index in range(TILE_TYPE_COUNT)
    )
    remaining_red_five_counts = tuple(
        STANDARD_RED_FIVE_COUNTS[color] - accounted_red_five_counts[color]
        for color in range(RED_FIVE_AXIS_COUNT)
    )

    return TileConservationResult(
        exact_accounted_counts=tuple(accounted_tile_counts),
        exact_accounted_red_five_counts=tuple(accounted_red_five_counts),
        remaining_tile_counts=remaining_tile_counts,
        remaining_red_five_counts=remaining_red_five_counts,
    )


__all__ = [
    "TileConservationResult",
    "derive_remaining_tile_inventory",
]
