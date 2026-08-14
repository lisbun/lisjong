"""RiichiEnv seat-visible状態から不変`PolicyInput`を構築するAdapter境界。

`docs/architecture.md`の「RiichiEnv Adapter」責務のうち、seat-visible
`Observation` / eventからのmaterialized state同期と`PolicyInput`生成を
実装する。RiichiEnv legal Actionから`InternalAction`への変換、Policy呼び出し、
Local game runnerは対象外である(別Issueのスコープ)。

`lisjong.policy_contract`とは異なり、このpackageは`riichienv`へ依存する。
`policy_contract` / `policies`側からの依存は逆流させない。
"""

from lisjong.riichienv_adapter.errors import AdapterSyncError
from lisjong.riichienv_adapter.materialized_state import (
    KyokuIdentity,
    SeatMaterializedState,
)
from lisjong.riichienv_adapter.policy_input import build_policy_input
from lisjong.riichienv_adapter.tile_conversion import (
    tile_from_mjai,
    tile_from_physical_id,
)

__all__ = [
    "AdapterSyncError",
    "KyokuIdentity",
    "SeatMaterializedState",
    "build_policy_input",
    "tile_from_mjai",
    "tile_from_physical_id",
]
