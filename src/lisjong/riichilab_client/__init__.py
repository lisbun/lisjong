"""RiichiLab validation/ranked WebSocket Client lower-level runtime。

`docs/riichilab-client.md`「責務境界」を実装する。RiichiLab
`/ws/validate` / `/ws/ranked`とのtransport lifecycle(接続、
`start_game` / `request_action` / `action_ack` / `validation_result` /
`end_game`、`request_id`のgame内lifecycle管理)だけを担当し、Policy判断・
Observation変換・Action mapping・`possible_actions` semantic validationは#38
`lisjong.riichilab_adapter`をconsumerとして再利用する。

`websockets`への依存はこのpackage内だけで使用し、`policy_contract` /
`policies` / `riichienv_adapter`へは逆流させない。

validation one-game orchestration(`ValidationResult` / `run_validation()`) /
CLI、execution profile / credential / common CLI compositionはいずれも
`lisjong-arena`へcanonical移管済み(Issue #19、cleanup lisjong#89)であり、
このpackageは`ValidationSession` / `RankedSession` / transport / protocol
trace writer等のlower-level runtimeだけをtemporaryに提供する。
"""

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
    "ValidationSession",
    "connect_ranked_transport",
    "connect_transport",
    "connect_validation_transport",
    "drive_ranked_session",
    "drive_session",
    "drive_validation_session",
]
