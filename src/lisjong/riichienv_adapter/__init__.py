"""RiichiEnv 0.4.8とlisjong内部型を変換するAdapter境界。

`docs/architecture.md`の「RiichiEnv Adapter」責務のうち、Issue #28の
seat-visible materialized state同期と`PolicyInput`生成、Issue #29のlegal Action
変換とdecision-local mapping、およびIssue #23の`DecisionContext`最終組み立てを
実装する。Policy呼び出しとLocal game runnerは対象外である。

`lisjong.policy_contract`とは異なり、このpackageは`riichienv`へ依存する。
`policy_contract` / `policies`側からの依存は逆流させない。
"""

from lisjong.riichienv_adapter.action_mapping import (
    ActionAdapterError,
    ActorMismatchError,
    ContextResolutionError,
    EmptyLegalActionsError,
    RepresentativeSelectionError,
    RiichiEnvActionMapping,
    RiichiEnvActionMappingSession,
    StaleActionMappingError,
    UnmappedActionError,
    UnsupportedActionError,
)
from lisjong.riichienv_adapter.decision import RiichiEnvDecision, build_decision
from lisjong.riichienv_adapter.errors import AdapterSyncError
from lisjong.riichienv_adapter.materialized_state import (
    KyokuIdentity,
    SeatMaterializedState,
)
from lisjong.riichienv_adapter.policy_input import build_policy_input
from lisjong.riichienv_adapter.seat_conversion import seat_from_player_index
from lisjong.riichienv_adapter.tile_conversion import (
    tile_from_mjai,
    tile_from_physical_id,
    tile_to_mjai,
)

__all__ = [
    "ActionAdapterError",
    "ActorMismatchError",
    "AdapterSyncError",
    "ContextResolutionError",
    "EmptyLegalActionsError",
    "KyokuIdentity",
    "RepresentativeSelectionError",
    "RiichiEnvActionMapping",
    "RiichiEnvActionMappingSession",
    "RiichiEnvDecision",
    "SeatMaterializedState",
    "StaleActionMappingError",
    "UnmappedActionError",
    "UnsupportedActionError",
    "build_decision",
    "build_policy_input",
    "seat_from_player_index",
    "tile_from_mjai",
    "tile_from_physical_id",
    "tile_to_mjai",
]
