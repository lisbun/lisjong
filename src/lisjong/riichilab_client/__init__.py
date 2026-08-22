"""RiichiLab validation/ranked WebSocket Client lower-level runtime。

`docs/riichilab-client.md`「責務境界」を実装する。RiichiLab
`/ws/validate` / `/ws/ranked`とのtransport lifecycle(接続、
`start_game` / `request_action` / `action_ack` / `validation_result` /
`end_game`、`request_id`のgame内lifecycle管理)だけを担当し、Policy判断・
Observation変換・Action mapping・`possible_actions` semantic validationは#38
`lisjong.riichilab_adapter`をconsumerとして再利用する。

`websockets`への依存はこのpackage内だけで使用し、`policy_contract` /
`policies` / `riichienv_adapter`へは逆流させない。

validation result/runnerはpackage rootから公開するが、`python -m ...validation`で
対象moduleを事前importしないようにlazy exportする。ranked one-game orchestration /
CLIはlisjong-arenaへcanonical移管済みであり、このpackageはlower-level
`RankedSession` / transport / trace等だけをtemporaryに提供する。
"""

from typing import TYPE_CHECKING

from lisjong.riichilab_client.errors import (
    ProtocolError,
    RiichiLabClientError,
    TransportError,
    UnexpectedDisconnectError,
)
from lisjong.riichilab_client.session import (
    RankedSession,
    SessionStatus,
    ValidationSession,
)
from lisjong.riichilab_client.trace import (
    JsonlProtocolTraceWriter,
    ProtocolTraceError,
)
from lisjong.riichilab_client.transport import (
    DEFAULT_RANKED_URL,
    DEFAULT_VALIDATION_URL,
    Transport,
    connect_ranked_transport,
    connect_transport,
    connect_validation_transport,
    drive_ranked_session,
    drive_session,
    drive_validation_session,
)

if TYPE_CHECKING:
    from lisjong.riichilab_client.validation import ValidationResult, run_validation


def __getattr__(name: str) -> object:
    """validation実行moduleの公開名を、package access時にだけimportする。"""
    if name in {"ValidationResult", "run_validation"}:
        from lisjong.riichilab_client.validation import ValidationResult, run_validation

        exports = {
            "ValidationResult": ValidationResult,
            "run_validation": run_validation,
        }
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals().update(exports)
    return exports[name]


__all__ = [
    "DEFAULT_RANKED_URL",
    "DEFAULT_VALIDATION_URL",
    "JsonlProtocolTraceWriter",
    "ProtocolError",
    "ProtocolTraceError",
    "RankedSession",
    "RiichiLabClientError",
    "SessionStatus",
    "Transport",
    "TransportError",
    "UnexpectedDisconnectError",
    "ValidationResult",
    "ValidationSession",
    "connect_ranked_transport",
    "connect_transport",
    "connect_validation_transport",
    "drive_ranked_session",
    "drive_session",
    "drive_validation_session",
    "run_validation",
]
