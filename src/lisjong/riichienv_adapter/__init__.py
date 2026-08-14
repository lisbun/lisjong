"""RiichiEnv 0.4.8とlisjong内部型を変換するRiichiEnv Adapter境界。

docs/architecture.md「RiichiEnv Adapter」の責務のうち、Issue #29の対象範囲
（RiichiEnv Action変換とdecision-local mapping）を実装する。PolicyInput生成、
materialized state、`DecisionContext`の最終組み立ては含まない
（それぞれIssue #28、#23の責務）。
"""

from lisjong.riichienv_adapter.action_mapping import (
    ActionAdapterError,
    ActorMismatchError,
    ContextResolutionError,
    EmptyLegalActionsError,
    RepresentativeSelectionError,
    RiichiEnvActionMapping,
    StaleActionMappingError,
    UnmappedActionError,
    UnsupportedActionError,
    build_action_mapping,
)
from lisjong.riichienv_adapter.conversions import (
    seat_from_player_index,
    tile_from_physical_id,
)

__all__ = [
    "ActionAdapterError",
    "ActorMismatchError",
    "ContextResolutionError",
    "EmptyLegalActionsError",
    "RepresentativeSelectionError",
    "RiichiEnvActionMapping",
    "StaleActionMappingError",
    "UnmappedActionError",
    "UnsupportedActionError",
    "build_action_mapping",
    "seat_from_player_index",
    "tile_from_physical_id",
]
