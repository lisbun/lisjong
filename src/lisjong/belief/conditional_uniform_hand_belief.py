"""remaining tile inventoryから条件付き一様baseline HandBeliefを導出する。

Issue #65のbaseline estimatorとIssue #68のglobal conservation-preserving
quantizationを実装する。Issue #63の`derive_remaining_tile_inventory()`が導出する
`remaining_tile_counts` / `remaining_red_five_counts`が、AIから区別できない
remaining hidden slots（他家concealed hand、live wall、dead wall等）へ
一様かつexchangeableに配置されていると仮定するbaseline estimatorである。

```text
E[count(p, t)]
= remaining_tile_counts[t] * opponent_concealed_slot_counts_by_wind[p]
  / total_hidden_slot_count

total_hidden_slot_count = sum(remaining_tile_counts)
```

赤5も同じslot比率を用いる。河・副露・立直・手出しツモ切り・巡目・筋・壁・
人読み等の追加推論は行わない。#61 / #63を通じてすでにremaining inventoryから
除外された情報以上のsemantic inferenceを持ち込まない。random sampling /
Monte Carloは使わず、期待値を解析的に導出するpure / deterministicな
estimatorである。

selfについてはbaseline推定を行わず、既存`exact_self_belief()`を使う。
`opponent_concealed_slot_counts_by_wind`は各playerの実concealed hand size
ではなく、conditional uniform estimatorがremaining inventoryを配分する対象
となるhidden concealed slot countである。self wind entryは必ず0とする。

fixed-point quantizationでは、各physical tile poolについてopponent全体の
canonical target massをround-half-to-evenで決め、floor allocation後のraw
unitをfractional remainder降順、同値ならcanonical Wind順で配分する。
5牌は赤5と非赤5の排他的poolを別々に配分してから合成する。これにより
column単位のphysical conservationを保つ一方、playerごとのrow massまで
exact保存するbalanced matrix quantizationは実装しない。

同一`PolicyInput`から`derive_remaining_tile_inventory()` /
`exact_self_belief()` / `wind_for_seat()`をすべて導出するため、remaining
inventoryとself exact beliefのsnapshot不整合は起きない。state / cacheを
持たないpure functionであり、estimatorが複数必要になるまで
class / Protocol / ABC等の抽象化framework化は行わない。
"""

from lisjong.belief.canonical_axes import (
    red_five_index,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
from lisjong.belief.concealed_hand_belief import ConcealedHandBelief
from lisjong.belief.fixed_point import SCALE, round_half_to_even_ratio
from lisjong.belief.hand_belief import HandBelief
from lisjong.belief.self_belief import exact_self_belief
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import TileCategory, TileType

_SUITED_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)


def _normalize_slot_counts_by_wind(values: object) -> tuple[int, int, int, int]:
    try:
        counts = tuple(values)
    except TypeError:
        raise TypeError(
            "opponent_concealed_slot_counts_by_wind must be an iterable of int"
        ) from None
    if len(counts) != 4:
        raise ValueError(
            "opponent_concealed_slot_counts_by_wind must contain exactly 4 values"
        )
    for count in counts:
        if type(count) is not int:
            raise TypeError(
                "opponent_concealed_slot_counts_by_wind must contain only int values"
            )
        if count < 0:
            raise ValueError(
                "opponent_concealed_slot_counts_by_wind must not contain negative "
                "values"
            )
    return counts


def _allocate_fixed_point_pool(
    remaining_count: int,
    slot_counts_by_wind: tuple[int, int, int, int],
    total_hidden_slots: int,
) -> tuple[int, int, int, int]:
    """1つのphysical tile poolをcanonical Wind順で同時配分する。"""
    if total_hidden_slots == 0:
        return (0, 0, 0, 0)

    numerators = tuple(
        remaining_count * player_slots * SCALE for player_slots in slot_counts_by_wind
    )
    quotient_remainders = tuple(
        divmod(numerator, total_hidden_slots) for numerator in numerators
    )
    allocated = [quotient for quotient, _remainder in quotient_remainders]

    opponent_slots = sum(slot_counts_by_wind)
    target_total_raw = round_half_to_even_ratio(
        remaining_count * opponent_slots * SCALE, total_hidden_slots
    )
    units_to_distribute = target_total_raw - sum(allocated)

    wind_numbers = sorted(
        range(4),
        key=lambda wind_number: (-quotient_remainders[wind_number][1], wind_number),
    )
    for wind_number in wind_numbers[:units_to_distribute]:
        allocated[wind_number] += 1

    return tuple(allocated)


def _allocate_opponent_hands(
    remaining_tile_counts: tuple[int, ...],
    remaining_red_five_counts: tuple[int, ...],
    slot_counts_by_wind: tuple[int, int, int, int],
    total_hidden_slots: int,
) -> tuple[HandBelief, HandBelief, HandBelief, HandBelief]:
    expected_by_wind = [[0] * 34 for _ in range(4)]
    red_five_by_wind = [[0] * 3 for _ in range(4)]
    five_indices = {
        tile_type_index(TileType(category, 5)) for category in _SUITED_CATEGORIES
    }

    for tile_index, remaining_count in enumerate(remaining_tile_counts):
        if tile_index in five_indices:
            continue
        allocated = _allocate_fixed_point_pool(
            remaining_count, slot_counts_by_wind, total_hidden_slots
        )
        for wind_number, raw in enumerate(allocated):
            expected_by_wind[wind_number][tile_index] = raw

    for category in _SUITED_CATEGORIES:
        five_index = tile_type_index(TileType(category, 5))
        color_index = red_five_index(category)
        remaining_five_count = remaining_tile_counts[five_index]
        remaining_red_count = remaining_red_five_counts[color_index]
        remaining_normal_count = remaining_five_count - remaining_red_count

        normal_allocated = _allocate_fixed_point_pool(
            remaining_normal_count, slot_counts_by_wind, total_hidden_slots
        )
        red_allocated = _allocate_fixed_point_pool(
            remaining_red_count, slot_counts_by_wind, total_hidden_slots
        )
        for wind_number in range(4):
            red_raw = red_allocated[wind_number]
            expected_by_wind[wind_number][five_index] = (
                normal_allocated[wind_number] + red_raw
            )
            red_five_by_wind[wind_number][color_index] = red_raw

    return tuple(
        HandBelief(
            expected_count_raw=tuple(expected_by_wind[wind_number]),
            red_five_probability_raw=tuple(red_five_by_wind[wind_number]),
        )
        for wind_number in range(4)
    )


def estimate_conditional_uniform_hand_belief(
    policy_input: PolicyInput,
    opponent_concealed_slot_counts_by_wind: tuple[int, int, int, int],
) -> ConcealedHandBelief:
    """`policy_input`のremaining tile inventoryから条件付き一様baseline
    `ConcealedHandBelief`を導出する。

    `opponent_concealed_slot_counts_by_wind`はcanonical Wind order
    （EAST/SOUTH/WEST/NORTH）のexact hidden concealed slot countである。
    self windのentryは必ず0でなければならず、各entryはnon-negative int、
    合計は`total_hidden_slot_count = sum(remaining_tile_counts)`を超えては
    ならない。selfのHandBeliefは`exact_self_belief()`をそのまま使い、
    baseline推定の対象にしない。
    """
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("policy_input must be a PolicyInput")

    slot_counts = _normalize_slot_counts_by_wind(opponent_concealed_slot_counts_by_wind)

    self_wind_number = wind_index(
        wind_for_seat(policy_input.self_seat, policy_input.round.dealer_seat)
    )
    if slot_counts[self_wind_number] != 0:
        raise ValueError("opponent_concealed_slot_counts_by_wind[self] must be 0")

    conservation = derive_remaining_tile_inventory(policy_input)
    total_hidden_slots = sum(conservation.remaining_tile_counts)

    if sum(slot_counts) > total_hidden_slots:
        raise ValueError(
            "sum of opponent_concealed_slot_counts_by_wind must not exceed "
            "total_hidden_slot_count"
        )

    opponent_hands = _allocate_opponent_hands(
        conservation.remaining_tile_counts,
        conservation.remaining_red_five_counts,
        slot_counts,
        total_hidden_slots,
    )
    self_hand_belief = exact_self_belief(policy_input.own_hand)
    hands = tuple(
        self_hand_belief
        if wind_number == self_wind_number
        else opponent_hands[wind_number]
        for wind_number in range(4)
    )

    return ConcealedHandBelief(hands=hands)


__all__ = ["estimate_conditional_uniform_hand_belief"]
