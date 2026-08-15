"""RiichiLab validation WebSocket Client。

`docs/riichilab-client.md`「責務境界」を実装する。RiichiLab `/ws/validate`
とのtransport lifecycle(接続、`start_game` / `request_action` /
`action_ack` / `validation_result` / `end_game`、`request_id`のgame内
lifecycle管理)だけを担当し、Policy判断・Observation変換・Action
mapping・`possible_actions` semantic validationは#38
`lisjong.riichilab_adapter`をconsumerとして再利用する。

`websockets`への依存はこのpackage内だけで使用し、`policy_contract` /
`policies` / `riichienv_adapter`へは逆流させない。

`ValidationResult`と`run_validation`はpackage rootの公開APIとして維持するが、
`python -m lisjong.riichilab_client.validation`で対象moduleを事前importしない
ようにlazy exportする。
"""

from typing import TYPE_CHECKING

from lisjong.riichilab_client.errors import (
    ProtocolError,
    RiichiLabClientError,
    TransportError,
    UnexpectedDisconnectError,
)
from lisjong.riichilab_client.session import SessionStatus, ValidationSession
from lisjong.riichilab_client.transport import (
    DEFAULT_VALIDATION_URL,
    Transport,
    connect_validation_transport,
    drive_validation_session,
)

if TYPE_CHECKING:
    from lisjong.riichilab_client.validation import ValidationResult, run_validation


def __getattr__(name: str) -> object:
    """`validation`の公開名を、package access時にだけimportする。"""
    if name not in {"ValidationResult", "run_validation"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from lisjong.riichilab_client.validation import ValidationResult, run_validation

    exports = {
        "ValidationResult": ValidationResult,
        "run_validation": run_validation,
    }
    globals().update(exports)
    return exports[name]


__all__ = [
    "DEFAULT_VALIDATION_URL",
    "ProtocolError",
    "RiichiLabClientError",
    "SessionStatus",
    "Transport",
    "TransportError",
    "UnexpectedDisconnectError",
    "ValidationResult",
    "ValidationSession",
    "connect_validation_transport",
    "drive_validation_session",
    "run_validation",
]
