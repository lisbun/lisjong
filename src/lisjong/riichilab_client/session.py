"""RiichiLab validation game 1回分のtransport lifecycle state。

`docs/riichilab-client.md`「責務境界」を実装する。実WebSocket接続とは
独立した純粋なstate machineとして実装しており、fake/local test(parsed
JSON eventのdict)だけでlifecycle全体を確認できる。実際のWebSocket
送受信は`lisjong.riichilab_client.transport`が担当する。

`ValidationSession`は#38 `RiichiLabSeatAdapter`をconsumerとして1回だけ
生成し、Policy判断・Observation変換・Action mapping・
`possible_actions` semantic validationを再実装しない。
"""

from __future__ import annotations

from collections.abc import Mapping

from lisjong.policy_contract.policy import Policy
from lisjong.policy_contract.seat import Seat
from lisjong.riichilab_adapter.adapter import RiichiLabSeatAdapter
from lisjong.riichilab_client.errors import ProtocolError

_VALIDATION_SEAT = Seat.SEAT_0

_EVENT_TYPE_START_GAME = "start_game"
_EVENT_TYPE_REQUEST_ACTION = "request_action"
_EVENT_TYPE_ACTION_ACK = "action_ack"
_EVENT_TYPE_VALIDATION_RESULT = "validation_result"
_EVENT_TYPE_END_GAME = "end_game"

# 現在の公式protocolが定義するaction_ack status(Issue #39最新コメント)。
# accepted以外はfatal/non-fatalを問わずack historyへ記録する。
_KNOWN_ACK_STATUSES = frozenset(
    {"accepted", "rejected", "unparseable", "stale", "defaulted"}
)
# rejected/unparseableはchomboにつながり得る重大statusのため、観測した
# 時点でfail closedする。stale/defaultedはnon-fatalとして記録するだけ。
_FATAL_ACK_STATUSES = frozenset({"rejected", "unparseable"})

# server提供time budgetのうち、型だけ検証するfield名(値そのものはClientの
# deadline enforcementへ使わない。Issue #39最新コメント「client-side
# deadline cancellationはMVPでは実装しない」)。
_TIME_BUDGET_FIELDS = ("grace_ms", "bank_ms", "deadline_ms")


class SessionStatus:
    """呼び出し側が確認できる、validation lifecycleの現在状態のsnapshot。

    tokenやraw Observation、raw request_action全文等のsecretを含み得る
    transport dataは保持しない。
    """

    __slots__ = (
        "passed",
        "validation_result_received",
        "end_game_received",
        "failure_reason",
        "requests_received",
        "responses_sent",
        "ack_history",
    )

    def __init__(
        self,
        *,
        passed: bool | None,
        validation_result_received: bool,
        end_game_received: bool,
        failure_reason: str | None,
        requests_received: int,
        responses_sent: int,
        ack_history: Mapping[int, tuple[str, ...]],
    ) -> None:
        self.passed = passed
        self.validation_result_received = validation_result_received
        self.end_game_received = end_game_received
        self.failure_reason = failure_reason
        self.requests_received = requests_received
        self.responses_sent = responses_sent
        self.ack_history = ack_history


def _validate_time_metadata(time_value: object) -> None:
    """`request_action.time`をtransport metadataとして最小限型検証する。

    値そのものをdeadline enforcementへは使わない。存在しなくても許容する
    (公式field表とIssue本文の記述差を、この境界では過剰に前提しない)。
    """
    if time_value is None:
        return
    if not isinstance(time_value, Mapping):
        raise ProtocolError("request_action time metadata must be a mapping")
    for field_name in _TIME_BUDGET_FIELDS:
        if field_name not in time_value:
            continue
        value = time_value[field_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError(f"request_action time.{field_name} must be numeric")


class ValidationSession:
    """1 validation game分のtransport lifecycle stateを所有する。

    parsed済みJSON event(mapping)を受け取り、送信すべきpayloadがあれば
    そのdictを返す。WebSocket API自体・非同期I/Oからは独立している。
    """

    __slots__ = (
        "_policy",
        "_adapter",
        "_accepted_request_ids",
        "_last_accepted_request_id",
        "_sent_request_ids",
        "_ack_history",
        "_requests_received",
        "_responses_sent",
        "_end_game_received",
        "_validation_result_received",
        "_passed",
        "_failure_reason",
    )

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self._adapter: RiichiLabSeatAdapter | None = None
        self._accepted_request_ids: set[int] = set()
        self._last_accepted_request_id: int | None = None
        self._sent_request_ids: set[int] = set()
        self._ack_history: dict[int, list[str]] = {}
        self._requests_received = 0
        self._responses_sent = 0
        self._end_game_received = False
        self._validation_result_received = False
        self._passed: bool | None = None
        self._failure_reason: str | None = None

    @property
    def validation_result_received(self) -> bool:
        return self._validation_result_received

    def status(self) -> SessionStatus:
        return SessionStatus(
            passed=self._passed,
            validation_result_received=self._validation_result_received,
            end_game_received=self._end_game_received,
            failure_reason=self._failure_reason,
            requests_received=self._requests_received,
            responses_sent=self._responses_sent,
            ack_history={
                request_id: tuple(statuses)
                for request_id, statuses in self._ack_history.items()
            },
        )

    def handle_event(self, event: Mapping) -> dict | None:
        """1件のparsed済みJSON eventをdispatchする。

        送信すべきpayloadがある場合だけdictを返す(`request_action`の
        処理結果)。それ以外は`None`を返す。未知event type(`type`欠落を
        含む)はforward compatibilityのためignoreする。既知eventの必須
        fieldが欠落・型不正な場合はfail closedする(`ProtocolError`)。
        """
        if not isinstance(event, Mapping):
            raise ProtocolError("event must be a JSON object")

        event_type = event.get("type")

        if event_type == _EVENT_TYPE_START_GAME:
            self._handle_start_game(event)
            return None
        if event_type == _EVENT_TYPE_REQUEST_ACTION:
            return self._handle_request_action(event)
        if event_type == _EVENT_TYPE_ACTION_ACK:
            self._handle_action_ack(event)
            return None
        if event_type == _EVENT_TYPE_VALIDATION_RESULT:
            self._handle_validation_result(event)
            return None
        if event_type == _EVENT_TYPE_END_GAME:
            # validation modeではend_game受信だけでは切断せず、
            # validation_resultを待つ(Issue #39最新コメント)。
            self._end_game_received = True
            return None

        # standard informational MJAI eventを含む未知event typeは、Policy
        # stateの正本が#38のdeserialize済みObservationであるため、
        # Client側でPolicy stateへ二重適用せずignoreする。
        return None

    def _handle_start_game(self, event: Mapping) -> None:
        # 公式RiichiLab Protocolでは、bot seat indexは`seat`ではなく`id`
        # fieldである(`{"type": "start_game", "id": 0}`、Issue #39初回
        # review blocking finding)。`seat`をfallbackとして併用しない。
        seat_value = event.get("id")
        if isinstance(seat_value, bool) or not isinstance(seat_value, int):
            raise ProtocolError("start_game is missing a valid integer id")
        if seat_value not in (0, 1, 2, 3):
            raise ProtocolError(f"start_game id out of range: {seat_value!r}")
        seat = Seat(seat_value)

        if self._adapter is not None:
            # duplicate start_gameは安全側で扱う: 同一seatなら既存Adapter
            # runtimeをそのまま維持し、作り直さない。seatが食い違う場合は
            # silent補正せずfail closedする。
            if seat != self._adapter.self_seat:
                raise ProtocolError(
                    "duplicate start_game reported a different seat than the "
                    "already-bound adapter"
                )
            return

        if seat != _VALIDATION_SEAT:
            raise ProtocolError(
                f"validation requires seat {int(_VALIDATION_SEAT)}, got {seat_value!r}"
            )

        self._adapter = RiichiLabSeatAdapter(
            self_seat=_VALIDATION_SEAT, policy=self._policy
        )

    def _handle_request_action(self, event: Mapping) -> dict:
        if self._adapter is None:
            raise ProtocolError("request_action received before start_game")

        request_id = event.get("request_id")
        if isinstance(request_id, bool) or not isinstance(request_id, int):
            raise ProtocolError("request_action is missing a valid integer request_id")

        if request_id in self._accepted_request_ids:
            raise ProtocolError(f"duplicate request_id: {request_id!r}")
        if (
            self._last_accepted_request_id is not None
            and request_id <= self._last_accepted_request_id
        ):
            raise ProtocolError(
                f"request_id did not increase monotonically: {request_id!r} "
                f"<= {self._last_accepted_request_id!r}"
            )

        _validate_time_metadata(event.get("time"))

        self._accepted_request_ids.add(request_id)
        self._last_accepted_request_id = request_id
        self._requests_received += 1

        # possible_actions validation、Observation deserialize、Policy
        # 呼び出し、Action mappingはすべて#38が所有する。ここでは
        # consumerとして呼び出すだけで再実装しない。
        response = self._adapter.process_request_action(event)

        if response.request_id != request_id:
            raise ProtocolError(
                "adapter response request_id does not match the current request"
            )
        # send直前にも、current requestへのbindを再確認する(cross-request
        # payload再利用の禁止、Issue #39本文セクション15)。
        if request_id != self._last_accepted_request_id:
            raise ProtocolError("response is no longer bound to the current request")
        if request_id in self._sent_request_ids:
            raise ProtocolError(f"request_id already sent a response: {request_id!r}")

        self._sent_request_ids.add(request_id)
        self._responses_sent += 1

        outgoing = dict(response.action)
        outgoing["request_id"] = response.request_id
        return outgoing

    def _handle_action_ack(self, event: Mapping) -> None:
        request_id = event.get("request_id")
        if isinstance(request_id, bool) or not isinstance(request_id, int):
            raise ProtocolError("action_ack is missing a valid integer request_id")

        status = event.get("status")
        if not isinstance(status, str) or status not in _KNOWN_ACK_STATUSES:
            raise ProtocolError(
                f"action_ack has an unknown or invalid status: {status!r}"
            )

        if request_id not in self._accepted_request_ids:
            raise ProtocolError(
                f"action_ack references an unknown or future request_id: {request_id!r}"
            )

        self._ack_history.setdefault(request_id, []).append(status)

        if status in _FATAL_ACK_STATUSES:
            raise ProtocolError(f"action_ack reported a fatal status: {status!r}")

    def _handle_validation_result(self, event: Mapping) -> None:
        passed = event.get("passed")
        if not isinstance(passed, bool):
            raise ProtocolError("validation_result is missing a valid boolean passed")

        reason = event.get("reason")
        if reason is None:
            reason = event.get("message")
        if reason is not None and not isinstance(reason, str):
            raise ProtocolError("validation_result reason/message must be a string")

        self._validation_result_received = True
        self._passed = passed
        self._failure_reason = reason


__all__ = ["SessionStatus", "ValidationSession"]
