"""4 wind全体の非公開手牌beliefのcanonical container。

Issue #59が固定したWind-major / row-major flattened layoutを実装する。
`ConcealedHandBelief`はwind別の`HandBelief`を束ねるだけで、他家手牌を
実際に推定するalgorithm自体は持たない。`PolicyInput`や`DecisionContext`へは
このIssueでは統合しない。
"""

from dataclasses import dataclass

from lisjong.belief.canonical_axes import wind_index
from lisjong.belief.hand_belief import HandBelief
from lisjong.policy_contract.tile import TileCategory, TileType
from lisjong.policy_contract.wind import Wind


@dataclass(frozen=True, slots=True)
class ConcealedHandBelief:
    """4 wind分の非公開手牌beliefをcanonical Wind axis順で束ねたcontainer。

    ```text
    ConcealedHandBelief
    └── hands  (length 4, canonical wind_index順 = EAST, SOUTH, WEST, NORTH)
    ```

    `flattened_expected_count_raw` / `flattened_red_five_probability_raw`は、
    Issue #59が固定したWind-major / row-major flattened layout
    （`concealed_hand_belief: [4, 34]`, `concealed_red_five_belief: [4, 3]`）
    そのものであり、offsetは`canonical_axes.concealed_hand_offset()` /
    `canonical_axes.red_five_offset()`と一致する。
    """

    hands: tuple[HandBelief, HandBelief, HandBelief, HandBelief]

    def __post_init__(self) -> None:
        try:
            hands = tuple(self.hands)
        except TypeError:
            raise TypeError("hands must be an iterable of HandBelief") from None
        if len(hands) != 4:
            raise ValueError("hands must contain exactly 4 HandBelief")
        if any(not isinstance(hand, HandBelief) for hand in hands):
            raise TypeError("hands must contain only HandBelief instances")
        object.__setattr__(self, "hands", hands)

    def hand(self, wind: Wind) -> HandBelief:
        """`wind`のHandBeliefを返す。"""
        return self.hands[wind_index(wind)]

    def expected_count(self, wind: Wind, tile_type: TileType) -> float:
        """`wind`の手牌における`tile_type`のexpected countを返す。"""
        return self.hand(wind).expected_count(tile_type)

    def red_five_probability(self, wind: Wind, category: TileCategory) -> float:
        """`wind`の手牌における`category`の赤5 probabilityを返す。"""
        return self.hand(wind).red_five_probability(category)

    @property
    def flattened_expected_count_raw(self) -> tuple[int, ...]:
        """Wind-major / row-majorのflattened raw buffer（length 136）。"""
        return tuple(raw for hand in self.hands for raw in hand.expected_count_raw)

    @property
    def flattened_red_five_probability_raw(self) -> tuple[int, ...]:
        """Wind-major / row-majorのflattened raw buffer（length 12）。"""
        return tuple(
            raw for hand in self.hands for raw in hand.red_five_probability_raw
        )
