"""非公開手牌beliefのcanonical representation package。

Issue #59「風別の非公開手牌beliefを固定小数点canonical representationと
して導入する」を実装する。他家手牌を実際に推定するalgorithm（baseline /
uniform estimator、河・副露・手出しツモ切りを使う推定、Policyへの統合、
neural network、training dataset等）はこのpackageの責務ではなく、
canonical representationだけを提供する。

`lisjong.policy_contract`の`Tile` / `TileType` / `TileCategory` / `Seat` /
`Wind` / `OwnHandState`を再利用し、RiichiEnv、RiichiLab、mjai、WebSocket等の
外部library固有型へは依存しない。
"""

from lisjong.belief.canonical_axes import (
    concealed_hand_offset,
    red_five_index,
    red_five_offset,
    seat_for_wind,
    tile_type_from_index,
    tile_type_index,
    wind_for_seat,
    wind_from_index,
    wind_index,
)
from lisjong.belief.concealed_hand_belief import ConcealedHandBelief
from lisjong.belief.fixed_point import (
    EXPECTED_COUNT_MAX_RAW,
    RED_FIVE_PROBABILITY_MAX_RAW,
    SCALE,
    expected_count_to_raw,
    raw_to_semantic,
    red_five_probability_to_raw,
)
from lisjong.belief.hand_belief import HandBelief
from lisjong.belief.self_belief import exact_self_belief

__all__ = [
    "EXPECTED_COUNT_MAX_RAW",
    "RED_FIVE_PROBABILITY_MAX_RAW",
    "SCALE",
    "ConcealedHandBelief",
    "HandBelief",
    "concealed_hand_offset",
    "exact_self_belief",
    "expected_count_to_raw",
    "raw_to_semantic",
    "red_five_index",
    "red_five_offset",
    "red_five_probability_to_raw",
    "seat_for_wind",
    "tile_type_from_index",
    "tile_type_index",
    "wind_for_seat",
    "wind_from_index",
    "wind_index",
]
