"""非公開手牌belief・公開済み牌provenanceのcanonical representation package。

Issue #59「風別の非公開手牌beliefを固定小数点canonical representationと
して導入する」で、非公開手牌の推定値（`HandBelief`の`expected_count` /
`red_five_probability`）を扱うcanonical representationを導入した。
Issue #61「公開済み牌情報を34牌種canonical provenance featureとして導出する」
で、既存semantic state（discard / meld / dora indicator）から導出する
exact observed integer count（`PublicTileProvenance`）を追加した。
Issue #63「観測済みprovenanceから牌保存則を検証しremaining tile inventoryを
導出する」で、self concealed handと#61 provenanceを合算したexact accounted
countをstandard physical tile inventory（`tile_inventory.py`）から差し引き、
牌保存則をfail-closedに検証しながらWind axisを持たないremaining tile
inventory（`TileConservationResult`）を導出する処理を追加した。
Issue #65「remaining tile inventoryから条件付き一様baseline HandBeliefを
導出する」で、`estimate_conditional_uniform_hand_belief()`が
`derive_remaining_tile_inventory()`の結果を、AIから区別できないremaining
hidden slots（他家concealed hand・live wall・dead wall等）へ一様かつ
exchangeableに配置されていると仮定して他家`HandBelief`へ配分する、
lisjong初のbaseline estimatorを追加した。selfは`exact_self_belief()`の
exact beliefのままとし、baseline推定の対象にしない。
Issue #68でphysical tile poolごとのglobal conservation-preserving allocationを
追加し、Issue #67でremaining inventoryから3 opponentsのbeliefをstrictに
差し引く`NonPlayerHiddenBelief`とglobal conservation validationを追加した。

`HandBelief`はbelief（推定値）、`PublicTileProvenance` /
`TileConservationResult`はprovenance / conservation結果（実際に観測された
exact count）であり、同じ34牌種 / red-five axisを共有するがsemanticは
異なる。`estimate_conditional_uniform_hand_belief()`は#59 / #61 / #63の
exact informationだけを条件として使うconditional uniform baselineであり、
河・副露・立直・手出しツモ切り・巡目・筋・壁・人読み等を使う追加推論、
学習済みestimator、Policyへの統合はこのpackageの責務ではない。

`lisjong.policy_contract`の`Tile` / `TileType` / `TileCategory` / `Seat` /
`Wind` / `OwnHandState` / `PolicyInput`等を再利用し、RiichiEnv、RiichiLab、
mjai、WebSocket等の外部library固有型へは依存しない。
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
from lisjong.belief.conditional_uniform_hand_belief import (
    estimate_conditional_uniform_hand_belief,
)
from lisjong.belief.fixed_point import (
    EXPECTED_COUNT_MAX_RAW,
    RED_FIVE_PROBABILITY_MAX_RAW,
    SCALE,
    expected_count_to_raw,
    raw_to_semantic,
    red_five_probability_to_raw,
    round_half_to_even_ratio,
)
from lisjong.belief.hand_belief import HandBelief
from lisjong.belief.non_player_hidden_belief import (
    NonPlayerHiddenBelief,
    derive_non_player_hidden_belief,
)
from lisjong.belief.public_provenance import (
    BASE_TILE_COUNT_MAX,
    RED_FIVE_COUNT_MAX,
    PublicTileProvenance,
    TileProvenanceCounts,
    WindTileProvenanceCounts,
    encode_public_tile_provenance,
)
from lisjong.belief.self_belief import exact_self_belief
from lisjong.belief.tile_conservation import (
    TileConservationResult,
    derive_remaining_tile_inventory,
)
from lisjong.belief.tile_inventory import (
    RED_FIVE_AXIS_COUNT,
    STANDARD_RED_FIVE_COUNTS,
    STANDARD_TILE_COUNTS,
    TILE_TYPE_COUNT,
    TOTAL_PHYSICAL_TILE_COUNT,
)

__all__ = [
    "BASE_TILE_COUNT_MAX",
    "EXPECTED_COUNT_MAX_RAW",
    "RED_FIVE_AXIS_COUNT",
    "RED_FIVE_COUNT_MAX",
    "RED_FIVE_PROBABILITY_MAX_RAW",
    "SCALE",
    "STANDARD_RED_FIVE_COUNTS",
    "STANDARD_TILE_COUNTS",
    "TILE_TYPE_COUNT",
    "TOTAL_PHYSICAL_TILE_COUNT",
    "ConcealedHandBelief",
    "HandBelief",
    "NonPlayerHiddenBelief",
    "PublicTileProvenance",
    "TileConservationResult",
    "TileProvenanceCounts",
    "WindTileProvenanceCounts",
    "concealed_hand_offset",
    "derive_remaining_tile_inventory",
    "derive_non_player_hidden_belief",
    "encode_public_tile_provenance",
    "estimate_conditional_uniform_hand_belief",
    "exact_self_belief",
    "expected_count_to_raw",
    "raw_to_semantic",
    "red_five_index",
    "red_five_offset",
    "red_five_probability_to_raw",
    "round_half_to_even_ratio",
    "seat_for_wind",
    "tile_type_from_index",
    "tile_type_index",
    "wind_for_seat",
    "wind_from_index",
    "wind_index",
]
