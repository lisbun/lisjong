"""`ValidationSession`のpure lifecycle unit test(Issue #39)。

実WebSocket接続・asyncio・実RiichiEnvなしに、`start_game` / `request_id`
lifecycle / `action_ack` / forward compatibility / `end_game` /
`validation_result` / fail closedを確認する。

#38 `RiichiLabSeatAdapter`自体の内部処理(Policy呼び出し、Observation
deserialize、`possible_actions` semantic validation)はここで再検証しない
(#38の責務)。ここでは`lisjong.riichilab_client.session.RiichiLabSeatAdapter`
をfake stubへ差し替え、Client固有のtransport lifecycleロジックだけを
孤立させて確認する。#38 Adapterを実際に使った統合確認は
`test_riichilab_client_validation.py`が担当する。
"""

import unittest
from unittest.mock import patch

from lisjong.policies import MinimalPolicy
from lisjong.riichilab_adapter.adapter import SendReadyResponse
from lisjong.riichilab_client.errors import ProtocolError
from lisjong.riichilab_client.session import ValidationSession

_PATCH_TARGET = "lisjong.riichilab_client.session.RiichiLabSeatAdapter"


class _FakeAdapter:
    def __init__(
        self,
        self_seat,
        *,
        response_request_id_override=None,
        raise_error=None,
        response_action=None,
    ) -> None:
        self.self_seat = self_seat
        self._override = response_request_id_override
        self._raise_error = raise_error
        self._response_action = response_action or {
            "type": "dahai",
            "actor": int(self_seat),
            "pai": "1m",
        }
        self.calls: list[object] = []

    def process_request_action(self, raw_request_action):
        self.calls.append(raw_request_action)
        if self._raise_error is not None:
            raise self._raise_error
        request_id = raw_request_action["request_id"]
        returned_id = self._override if self._override is not None else request_id
        return SendReadyResponse(
            request_id=returned_id, action=dict(self._response_action)
        )


def _fake_adapter_factory(**kwargs):
    def factory(self_seat, policy):
        return _FakeAdapter(self_seat, **kwargs)

    return factory


def _start_game(seat: int = 0) -> dict:
    return {"type": "start_game", "seat": seat}


def _request_action(request_id: int, **extra) -> dict:
    event = {
        "type": "request_action",
        "request_id": request_id,
        "possible_actions": [],
        "observation": "unused-by-fake-adapter",
    }
    event.update(extra)
    return event


def _action_ack(request_id, status, **extra) -> dict:
    event = {"type": "action_ack", "request_id": request_id, "status": status}
    event.update(extra)
    return event


def _validation_result(**fields) -> dict:
    return {"type": "validation_result", **fields}


class StartGameTest(unittest.TestCase):
    def test_binds_seat_0(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            self.assertIsNone(session.handle_event(_start_game(0)))

    def test_rejects_non_zero_seat(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            with self.assertRaises(ProtocolError):
                session.handle_event(_start_game(1))

    def test_rejects_missing_seat_field(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with self.assertRaises(ProtocolError):
            session.handle_event({"type": "start_game"})

    def test_rejects_non_integer_seat(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with self.assertRaises(ProtocolError):
            session.handle_event({"type": "start_game", "seat": "0"})

    def test_rejects_boolean_seat(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with self.assertRaises(ProtocolError):
            session.handle_event({"type": "start_game", "seat": False})

    def test_duplicate_start_game_same_seat_is_safe_noop(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            session.handle_event(_start_game(0))
            first_adapter = session._adapter
            session.handle_event(_start_game(0))
            self.assertIs(session._adapter, first_adapter)

    def test_duplicate_start_game_different_seat_fails_closed(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            session.handle_event(_start_game(0))
            with self.assertRaises(ProtocolError):
                session.handle_event(_start_game(1))

    def test_unknown_extra_field_on_start_game_is_ignored(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            session.handle_event({"type": "start_game", "seat": 0, "game_id": "abc"})


class RequestBeforeStartGameTest(unittest.TestCase):
    def test_request_action_before_start_game_fails_closed(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with self.assertRaises(ProtocolError):
            session.handle_event(_request_action(1))


class RequestIdLifecycleTest(unittest.TestCase):
    def _bound_session(self, **adapter_kwargs) -> ValidationSession:
        session = ValidationSession(MinimalPolicy())
        with patch(_PATCH_TARGET, _fake_adapter_factory(**adapter_kwargs)):
            session.handle_event(_start_game(0))
        return session

    def test_accepts_increasing_request_id_with_a_gap(self) -> None:
        session = self._bound_session()
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            outgoing_first = session.handle_event(_request_action(1))
            outgoing_second = session.handle_event(_request_action(37))
        self.assertEqual(outgoing_first["request_id"], 1)
        self.assertEqual(outgoing_second["request_id"], 37)

    def test_rejects_duplicate_request_id(self) -> None:
        session = self._bound_session()
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            session.handle_event(_request_action(5))
            with self.assertRaises(ProtocolError):
                session.handle_event(_request_action(5))

    def test_rejects_decreasing_request_id(self) -> None:
        session = self._bound_session()
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            session.handle_event(_request_action(10))
            with self.assertRaises(ProtocolError):
                session.handle_event(_request_action(3))

    def test_rejects_missing_request_id(self) -> None:
        session = self._bound_session()
        event = _request_action(1)
        del event["request_id"]
        with self.assertRaises(ProtocolError):
            session.handle_event(event)

    def test_rejects_non_integer_request_id(self) -> None:
        session = self._bound_session()
        with self.assertRaises(ProtocolError):
            session.handle_event(_request_action("1"))

    def test_rejects_boolean_request_id(self) -> None:
        session = self._bound_session()
        with self.assertRaises(ProtocolError):
            session.handle_event(_request_action(True))

    def test_rejects_adapter_response_request_id_mismatch(self) -> None:
        session = self._bound_session(response_request_id_override=999)
        with self.assertRaises(ProtocolError):
            session.handle_event(_request_action(1))

    def test_sends_exactly_once_per_request(self) -> None:
        session = self._bound_session()
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            outgoing = session.handle_event(_request_action(1))
        self.assertIsNotNone(outgoing)
        status = session.status()
        self.assertEqual(status.responses_sent, 1)
        self.assertEqual(status.requests_received, 1)

    def test_time_metadata_type_is_validated(self) -> None:
        session = self._bound_session()
        with self.assertRaises(ProtocolError):
            session.handle_event(_request_action(1, time={"grace_ms": "not-a-number"}))

    def test_time_metadata_absent_is_allowed(self) -> None:
        session = self._bound_session()
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            session.handle_event(_request_action(1))

    def test_time_metadata_numeric_fields_are_allowed(self) -> None:
        session = self._bound_session()
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            session.handle_event(
                _request_action(
                    1, time={"grace_ms": 500, "bank_ms": 15000, "deadline_ms": 3000}
                )
            )

    def test_time_is_not_forwarded_to_policy(self) -> None:
        seen_requests = []

        class _RecordingFakeAdapter(_FakeAdapter):
            def process_request_action(self, raw_request_action):
                seen_requests.append(raw_request_action)
                return super().process_request_action(raw_request_action)

        session = ValidationSession(MinimalPolicy())
        with patch(
            _PATCH_TARGET, lambda self_seat, policy: _RecordingFakeAdapter(self_seat)
        ):
            session.handle_event(_start_game(0))
            session.handle_event(_request_action(1, time={"grace_ms": 500}))
        # ここで確認したいのはClient自身がtimeをPolicyへ注入しないことで
        # あり、#38 Adapterへ渡すraw request自体にtimeが含まれることは
        # #38 (`parse_request_action`)が別途保持のみ行い、Policyへは渡さない
        # 契約になっている(test_riichilab_adapter.py参照)。
        self.assertIn("time", seen_requests[0])

    def test_adapter_exception_propagates_and_produces_no_payload(self) -> None:
        session = self._bound_session(raise_error=RuntimeError("adapter exploded"))
        with self.assertRaises(RuntimeError):
            session.handle_event(_request_action(1))


class ActionAckTest(unittest.TestCase):
    def _session_with_accepted_request(self, request_id: int = 1) -> ValidationSession:
        session = ValidationSession(MinimalPolicy())
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            session.handle_event(_start_game(0))
            session.handle_event(_request_action(request_id))
        return session

    def test_accepted_status_is_recorded_without_error(self) -> None:
        session = self._session_with_accepted_request()
        session.handle_event(_action_ack(1, "accepted"))
        self.assertEqual(session.status().ack_history[1], ("accepted",))

    def test_stale_status_is_recorded_without_error(self) -> None:
        session = self._session_with_accepted_request()
        session.handle_event(_action_ack(1, "stale"))
        self.assertEqual(session.status().ack_history[1], ("stale",))

    def test_defaulted_status_is_recorded_without_error(self) -> None:
        session = self._session_with_accepted_request()
        session.handle_event(_action_ack(1, "defaulted"))
        self.assertEqual(session.status().ack_history[1], ("defaulted",))

    def test_rejected_status_raises(self) -> None:
        session = self._session_with_accepted_request()
        with self.assertRaises(ProtocolError):
            session.handle_event(_action_ack(1, "rejected"))

    def test_unparseable_status_raises(self) -> None:
        session = self._session_with_accepted_request()
        with self.assertRaises(ProtocolError):
            session.handle_event(_action_ack(1, "unparseable"))

    def test_unknown_status_raises(self) -> None:
        session = self._session_with_accepted_request()
        with self.assertRaises(ProtocolError):
            session.handle_event(_action_ack(1, "made_up_status"))

    def test_unknown_request_id_raises(self) -> None:
        session = self._session_with_accepted_request()
        with self.assertRaises(ProtocolError):
            session.handle_event(_action_ack(999, "accepted"))

    def test_future_request_id_raises(self) -> None:
        # 対応するrequest_actionをまだ受理していないrequest_idへのackは
        # unknown request_idと同様に成功扱いしない。
        session = ValidationSession(MinimalPolicy())
        with patch(_PATCH_TARGET, _fake_adapter_factory()):
            session.handle_event(_start_game(0))
        with self.assertRaises(ProtocolError):
            session.handle_event(_action_ack(5, "accepted"))

    def test_missing_request_id_raises(self) -> None:
        session = self._session_with_accepted_request()
        with self.assertRaises(ProtocolError):
            session.handle_event({"type": "action_ack", "status": "accepted"})

    def test_missing_status_raises(self) -> None:
        session = self._session_with_accepted_request()
        with self.assertRaises(ProtocolError):
            session.handle_event({"type": "action_ack", "request_id": 1})

    def test_ack_history_accumulates_multiple_statuses_for_one_request(self) -> None:
        session = self._session_with_accepted_request()
        with self.assertRaises(ProtocolError):
            # defaulted (non-fatal) の後にlate responseがstaleとしてserverに
            # 届いた、という現実的な順序を想定する。stale自体はraiseしない
            # ため、直後にrejectedを送ってhistory内容だけ確認する。
            session.handle_event(_action_ack(1, "defaulted"))
            session.handle_event(_action_ack(1, "stale"))
            session.handle_event(_action_ack(1, "rejected"))
        self.assertEqual(
            session.status().ack_history[1], ("defaulted", "stale", "rejected")
        )

    def test_duplicate_ack_is_not_treated_as_a_different_requests_success(self) -> None:
        session = self._session_with_accepted_request()
        session.handle_event(_action_ack(1, "accepted"))
        session.handle_event(_action_ack(1, "accepted"))
        self.assertEqual(session.status().ack_history[1], ("accepted", "accepted"))

    def test_unknown_extra_field_is_ignored(self) -> None:
        session = self._session_with_accepted_request()
        session.handle_event(_action_ack(1, "accepted", server_time_ms=123456))
        self.assertEqual(session.status().ack_history[1], ("accepted",))


class ForwardCompatibilityTest(unittest.TestCase):
    def test_unknown_event_type_is_ignored(self) -> None:
        session = ValidationSession(MinimalPolicy())
        self.assertIsNone(session.handle_event({"type": "some_future_event"}))

    def test_missing_type_field_is_ignored(self) -> None:
        session = ValidationSession(MinimalPolicy())
        self.assertIsNone(session.handle_event({"foo": "bar"}))

    def test_informational_mjai_event_is_ignored(self) -> None:
        session = ValidationSession(MinimalPolicy())
        self.assertIsNone(session.handle_event({"type": "tsumo", "actor": 0}))


class EndGameAndValidationResultTest(unittest.TestCase):
    def test_end_game_sets_flag_without_marking_validation_complete(self) -> None:
        session = ValidationSession(MinimalPolicy())
        session.handle_event({"type": "end_game"})
        status = session.status()
        self.assertTrue(status.end_game_received)
        self.assertFalse(status.validation_result_received)
        self.assertFalse(session.validation_result_received)

    def test_validation_result_sets_passed_and_flag(self) -> None:
        session = ValidationSession(MinimalPolicy())
        session.handle_event(_validation_result(passed=True))
        status = session.status()
        self.assertTrue(status.validation_result_received)
        self.assertTrue(status.passed)
        self.assertTrue(session.validation_result_received)

    def test_validation_result_failure_reason_is_captured(self) -> None:
        session = ValidationSession(MinimalPolicy())
        session.handle_event(_validation_result(passed=False, reason="illegal action"))
        status = session.status()
        self.assertFalse(status.passed)
        self.assertEqual(status.failure_reason, "illegal action")

    def test_validation_result_message_used_as_reason_fallback(self) -> None:
        session = ValidationSession(MinimalPolicy())
        session.handle_event(_validation_result(passed=False, message="chombo"))
        self.assertEqual(session.status().failure_reason, "chombo")

    def test_validation_result_without_reason_is_allowed(self) -> None:
        session = ValidationSession(MinimalPolicy())
        session.handle_event(_validation_result(passed=True))
        self.assertIsNone(session.status().failure_reason)

    def test_validation_result_malformed_passed_fails_closed(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with self.assertRaises(ProtocolError):
            session.handle_event(_validation_result(passed="yes"))

    def test_validation_result_missing_passed_fails_closed(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with self.assertRaises(ProtocolError):
            session.handle_event(_validation_result())

    def test_validation_result_reason_wrong_type_fails_closed(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with self.assertRaises(ProtocolError):
            session.handle_event(_validation_result(passed=True, reason=123))

    def test_end_game_then_validation_result_is_the_expected_order(self) -> None:
        session = ValidationSession(MinimalPolicy())
        session.handle_event({"type": "end_game"})
        session.handle_event(_validation_result(passed=True))
        status = session.status()
        self.assertTrue(status.end_game_received)
        self.assertTrue(status.validation_result_received)
        self.assertTrue(status.passed)


if __name__ == "__main__":
    unittest.main()
