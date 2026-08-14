"""RiichiLab `request_action` Adapterと送信前possible_actions semantic validation。

`docs/riichilab-adapter.md`「責務境界」を実装する。WebSocket接続、token、
`start_game` / `action_ack` / `validation_result` / `end_game`、
`request_id`のgame内lifecycle管理、timeout schedulerはこのpackageの
責務ではない(Issue #39)。

`riichienv_adapter`と同様、このpackageは`riichienv`へ依存する。
`policy_contract` / `policies`側からの依存は逆流させない。
"""

from lisjong.riichilab_adapter.adapter import RiichiLabSeatAdapter, SendReadyResponse
from lisjong.riichilab_adapter.errors import (
    MalformedRequestActionError,
    ObservationDeserializeError,
    PossibleActionsValidationError,
    ProtocolConversionError,
    RiichiLabAdapterError,
    SeatMismatchError,
)
from lisjong.riichilab_adapter.request_action import ParsedRequestAction

__all__ = [
    "MalformedRequestActionError",
    "ObservationDeserializeError",
    "ParsedRequestAction",
    "PossibleActionsValidationError",
    "ProtocolConversionError",
    "RiichiLabAdapterError",
    "RiichiLabSeatAdapter",
    "SeatMismatchError",
    "SendReadyResponse",
]
