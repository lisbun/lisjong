"""remaining inventoryからnon-player hidden beliefを導出する。

Issue #67を実装する。`NonPlayerHiddenBelief`は、remaining physical tile
inventoryのうちopponent `ConcealedHandBelief`へ割り当てられていない残余massを
表す。live wall / dead wall / 未開示裏ドラ等のlocation別分解は持たない。

導出はcanonical fixed-point raw domainのstrict integer subtractionで行う。
upstream estimatorの不整合をtoleranceやclampで補正せず、basic tile、赤5、
非赤5のいずれかがremaining physical massを1 raw unitでも超えればfail closedする。
"""

from dataclasses import dataclass

from lisjong.belief.canonical_axes import red_five_index, tile_type_index
from lisjong.belief.concealed_hand_belief import ConcealedHandBelief
from lisjong.belief.fixed_point import (
    EXPECTED_COUNT_MAX_RAW,
    RED_FIVE_PROBABILITY_MAX_RAW,
    SCALE,
    raw_to_semantic,
)
from lisjong.belief.tile_conservation import TileConservationResult
from lisjong.belief.tile_inventory import RED_FIVE_AXIS_COUNT, TILE_TYPE_COUNT
from lisjong.policy_contract.tile import TileCategory, TileType
from lisjong.policy_contract.wind import Wind

_SUITED_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)
_CANONICAL_WINDS = (Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH)


def _normalize_raw_tuple(
    values: object, expected_length: int, max_raw: int, field_name: str
) -> tuple[int, ...]:
    try:
        raw_values = tuple(values)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable of int") from None
    if len(raw_values) != expected_length:
        raise ValueError(f"{field_name} must contain exactly {expected_length} values")
    for raw in raw_values:
        if type(raw) is not int:
            raise TypeError(f"{field_name} must contain only int values")
        if not 0 <= raw <= max_raw:
            raise ValueError(
                f"{field_name} values must be within their fixed-point range"
            )
    return raw_values


@dataclass(frozen=True, slots=True)
class NonPlayerHiddenBelief:
    """Wind axisを持たないnon-player hidden領域のcanonical belief。"""

    expected_count_raw: tuple[int, ...]
    red_five_probability_raw: tuple[int, ...]

    def __post_init__(self) -> None:
        expected_count_raw = _normalize_raw_tuple(
            self.expected_count_raw,
            TILE_TYPE_COUNT,
            EXPECTED_COUNT_MAX_RAW,
            "expected_count_raw",
        )
        red_five_probability_raw = _normalize_raw_tuple(
            self.red_five_probability_raw,
            RED_FIVE_AXIS_COUNT,
            RED_FIVE_PROBABILITY_MAX_RAW,
            "red_five_probability_raw",
        )

        for category in _SUITED_CATEGORIES:
            five_raw = expected_count_raw[tile_type_index(TileType(category, 5))]
            red_five_raw = red_five_probability_raw[red_five_index(category)]
            if red_five_raw > five_raw:
                raise ValueError(
                    "red_five_probability must not exceed the corresponding "
                    "five's expected_count"
                )

        object.__setattr__(self, "expected_count_raw", expected_count_raw)
        object.__setattr__(self, "red_five_probability_raw", red_five_probability_raw)

    def expected_count(self, tile_type: TileType) -> float:
        """基本牌種`tile_type`のnon-player hidden expected countを返す。"""
        return raw_to_semantic(self.expected_count_raw[tile_type_index(tile_type)])

    def red_five_probability(self, category: TileCategory) -> float:
        """数牌`category`のnon-player hidden赤5 probability massを返す。"""
        return raw_to_semantic(self.red_five_probability_raw[red_five_index(category)])


def derive_non_player_hidden_belief(
    conservation: TileConservationResult,
    concealed_hand_belief: ConcealedHandBelief,
    self_wind: Wind,
) -> NonPlayerHiddenBelief:
    """remaining massから3 opponentsのconcealed-hand beliefを差し引く。

    basic tile、赤5、非赤5のglobal conservationをraw integerでstrictに検証する。
    self concealed handは`conservation`ですでにaccount済みなので差し引かない。
    """
    if not isinstance(conservation, TileConservationResult):
        raise TypeError("conservation must be a TileConservationResult")
    if not isinstance(concealed_hand_belief, ConcealedHandBelief):
        raise TypeError("concealed_hand_belief must be a ConcealedHandBelief")
    if not isinstance(self_wind, Wind):
        raise TypeError("self_wind must be a Wind")

    opponent_hands = tuple(
        concealed_hand_belief.hand(wind)
        for wind in _CANONICAL_WINDS
        if wind is not self_wind
    )

    expected_count_raw: list[int] = []
    for tile_index, remaining_count in enumerate(conservation.remaining_tile_counts):
        remaining_raw = remaining_count * SCALE
        opponent_raw = sum(
            hand.expected_count_raw[tile_index] for hand in opponent_hands
        )
        if opponent_raw > remaining_raw:
            raise ValueError(
                "opponent expected_count_raw exceeds remaining physical mass "
                f"at tile index {tile_index}"
            )
        expected_count_raw.append(remaining_raw - opponent_raw)

    red_five_probability_raw: list[int] = []
    for color_index, remaining_red_count in enumerate(
        conservation.remaining_red_five_counts
    ):
        remaining_red_raw = remaining_red_count * SCALE
        opponent_red_raw = sum(
            hand.red_five_probability_raw[color_index] for hand in opponent_hands
        )
        if opponent_red_raw > remaining_red_raw:
            raise ValueError(
                "opponent red_five_probability_raw exceeds remaining red-five "
                f"mass at color index {color_index}"
            )
        red_five_probability_raw.append(remaining_red_raw - opponent_red_raw)

    for category in _SUITED_CATEGORIES:
        five_index = tile_type_index(TileType(category, 5))
        color_index = red_five_index(category)
        remaining_normal_raw = (
            conservation.remaining_tile_counts[five_index]
            - conservation.remaining_red_five_counts[color_index]
        ) * SCALE
        opponent_normal_raw = sum(
            hand.expected_count_raw[five_index]
            - hand.red_five_probability_raw[color_index]
            for hand in opponent_hands
        )
        if opponent_normal_raw > remaining_normal_raw:
            raise ValueError(
                "opponent normal-five raw exceeds remaining normal-five mass "
                f"at color index {color_index}"
            )

    return NonPlayerHiddenBelief(
        expected_count_raw=tuple(expected_count_raw),
        red_five_probability_raw=tuple(red_five_probability_raw),
    )


__all__ = ["NonPlayerHiddenBelief", "derive_non_player_hidden_belief"]
