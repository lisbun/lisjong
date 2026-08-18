"""remaining tile inventoryから条件付き一様baseline HandBeliefを導出する。

Issue #65を実装する。Issue #63の`derive_remaining_tile_inventory()`が導出する
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

fixed-point quantizationは、Issue #59のround-half-to-even canonical
rounding ruleを`fixed_point.round_half_to_even_ratio()`でbinary floatを
経由せず整数算術のまま再現する。playerごとのrow massと牌種ごとのcolumn
massを同時にexact保存するbalanced matrix quantizationは実装せず、
quantization driftはtestでbound確認する。

同一`PolicyInput`から`derive_remaining_tile_inventory()` /
`exact_self_belief()` / `wind_for_seat()`をすべて導出するため、remaining
inventoryとself exact beliefのsnapshot不整合は起きない。state / cacheを
持たないpure functionであり、estimatorが複数必要になるまで
class / Protocol / ABC等の抽象化framework化は行わない。
"""

from lisjong.belief.canonical_axes import wind_for_seat, wind_index
from lisjong.belief.concealed_hand_belief import ConcealedHandBelief
from lisjong.belief.fixed_point import SCALE, round_half_to_even_ratio
from lisjong.belief.hand_belief import HandBelief
from lisjong.belief.self_belief import exact_self_belief
from lisjong.belief.tile_conservation import (
    TileConservationResult,
    derive_remaining_tile_inventory,
)
from lisjong.policy_contract.policy_input import PolicyInput

_ZERO_EXPECTED_COUNT_RAW = tuple([0] * 34)
_ZERO_RED_FIVE_PROBABILITY_RAW = (0, 0, 0)


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


def _baseline_hand_belief(
    conservation: TileConservationResult, player_slots: int, total_hidden_slots: int
) -> HandBelief:
    if player_slots == 0:
        return HandBelief(
            expected_count_raw=_ZERO_EXPECTED_COUNT_RAW,
            red_five_probability_raw=_ZERO_RED_FIVE_PROBABILITY_RAW,
        )

    expected_count_raw = tuple(
        round_half_to_even_ratio(
            remaining_count * player_slots * SCALE, total_hidden_slots
        )
        for remaining_count in conservation.remaining_tile_counts
    )
    red_five_probability_raw = tuple(
        round_half_to_even_ratio(
            remaining_red_count * player_slots * SCALE, total_hidden_slots
        )
        for remaining_red_count in conservation.remaining_red_five_counts
    )
    return HandBelief(
        expected_count_raw=expected_count_raw,
        red_five_probability_raw=red_five_probability_raw,
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

    self_hand_belief = exact_self_belief(policy_input.own_hand)

    hands = tuple(
        self_hand_belief
        if wind_number == self_wind_number
        else _baseline_hand_belief(
            conservation, slot_counts[wind_number], total_hidden_slots
        )
        for wind_number in range(4)
    )

    return ConcealedHandBelief(hands=hands)


__all__ = ["estimate_conditional_uniform_hand_belief"]
