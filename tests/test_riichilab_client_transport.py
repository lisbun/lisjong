"""`drive_validation_session()`のfake/local transport test(Issue #39)。

実WebSocket接続なしに、`Transport` protocolへ準拠するfake objectで
JSON text送受信、binary frame無視、送信失敗・受信失敗・予期しない
切断、serialization失敗を確認する。`ValidationSession`のlifecycle detail
自体は`test_riichilab_client_session.py`が担当するため、ここでは
transport層の責務(frame種別判定、JSON parse、send/recv failureの
`ProtocolError`/`TransportError`への変換)に絞る。
"""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from lisjong.policies import MinimalPolicy
from lisjong.riichilab_adapter.adapter import SendReadyResponse
from lisjong.riichilab_client.errors import (
    ProtocolError,
    RiichiLabClientError,
    TransportError,
    UnexpectedDisconnectError,
)
from lisjong.riichilab_client.session import ValidationSession
from lisjong.riichilab_client.trace import JsonlProtocolTraceWriter, ProtocolTraceError
from lisjong.riichilab_client.transport import TransportClosed, drive_validation_session

_PATCH_TARGET = "lisjong.riichilab_client.session.RiichiLabSeatAdapter"


class _FakeAdapter:
    def __init__(self, self_seat) -> None:
        self.self_seat = self_seat

    def process_request_action(self, raw_request_action):
        return SendReadyResponse(
            request_id=raw_request_action["request_id"],
            action={"type": "dahai", "actor": int(self.self_seat), "pai": "1m"},
        )


def _fake_adapter_factory(self_seat, policy):
    return _FakeAdapter(self_seat)


class FakeTransport:
    """`Transport` protocolのtest double。tokenやAuthorization headerは持たない。"""

    def __init__(self, incoming: list) -> None:
        self._incoming = list(incoming)
        self.sent: list[str] = []
        self.closed = False
        self._send_should_fail = False
        self._recv_should_raise: Exception | None = None

    async def recv(self):
        if self._recv_should_raise is not None:
            raise self._recv_should_raise
        if not self._incoming:
            raise TransportClosed("no more fake messages queued")
        return self._incoming.pop(0)

    async def send(self, message: str) -> None:
        if self._send_should_fail:
            raise TransportClosed("fake send failure")
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True


def _run(coro):
    return asyncio.run(coro)


def _event_text(event: dict) -> str:
    return json.dumps(event)


class TextJsonRoundTripTest(unittest.TestCase):
    def test_receives_and_dispatches_json_text_frames(self) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [
                _event_text({"type": "start_game", "id": 0}),
                _event_text(
                    {
                        "type": "request_action",
                        "request_id": 1,
                        "possible_actions": [],
                        "observation": "unused",
                    }
                ),
                _event_text({"type": "validation_result", "passed": True}),
            ]
        )
        with patch(_PATCH_TARGET, _fake_adapter_factory):
            _run(drive_validation_session(session, transport))

        self.assertEqual(len(transport.sent), 1)
        sent_payload = json.loads(transport.sent[0])
        self.assertEqual(sent_payload["request_id"], 1)
        self.assertEqual(sent_payload["type"], "dahai")
        self.assertTrue(session.status().passed)

    def test_binary_frame_is_ignored_without_failing_the_client(self) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [
                b"\x00\x01binary-noise",
                _event_text({"type": "validation_result", "passed": True}),
            ]
        )
        _run(drive_validation_session(session, transport))
        self.assertTrue(session.status().validation_result_received)

    def test_json_syntax_error_fails_closed(self) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(["{not valid json"])
        with self.assertRaises(ProtocolError):
            _run(drive_validation_session(session, transport))

    def test_non_object_top_level_json_fails_closed(self) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(["[1, 2, 3]"])
        with self.assertRaises(ProtocolError):
            _run(drive_validation_session(session, transport))

    def test_known_event_malformed_field_fails_closed(self) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [_event_text({"type": "start_game", "id": "not-an-int"})]
        )
        with self.assertRaises(ProtocolError):
            _run(drive_validation_session(session, transport))

    def test_unknown_event_type_does_not_stop_the_drive_loop(self) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [
                _event_text({"type": "some_future_event", "payload": [1, 2, 3]}),
                _event_text({"type": "validation_result", "passed": True}),
            ]
        )
        _run(drive_validation_session(session, transport))
        self.assertTrue(session.status().validation_result_received)


class DisconnectTest(unittest.TestCase):
    def test_unexpected_disconnect_before_validation_result_is_not_success(
        self,
    ) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport([_event_text({"type": "end_game"})])
        with self.assertRaises(UnexpectedDisconnectError):
            _run(drive_validation_session(session, transport))
        self.assertFalse(session.status().validation_result_received)

    def test_receive_failure_is_reported_as_unexpected_disconnect(self) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport([])
        transport._recv_should_raise = TransportClosed("boom")
        with self.assertRaises(UnexpectedDisconnectError):
            _run(drive_validation_session(session, transport))


class SendFailureTest(unittest.TestCase):
    def test_send_failure_is_reported_as_transport_error(self) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [
                _event_text({"type": "start_game", "id": 0}),
                _event_text(
                    {
                        "type": "request_action",
                        "request_id": 1,
                        "possible_actions": [],
                        "observation": "unused",
                    }
                ),
            ]
        )
        transport._send_should_fail = True
        with patch(_PATCH_TARGET, _fake_adapter_factory):
            with self.assertRaises(TransportError):
                _run(drive_validation_session(session, transport))

    def test_serialization_failure_does_not_send_anything(self) -> None:
        class _UnserializableAdapter(_FakeAdapter):
            def process_request_action(self, raw_request_action):
                return SendReadyResponse(
                    request_id=raw_request_action["request_id"],
                    action={"type": "dahai", "actor": 0, "bad": object()},
                )

        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [
                _event_text({"type": "start_game", "id": 0}),
                _event_text(
                    {
                        "type": "request_action",
                        "request_id": 1,
                        "possible_actions": [],
                        "observation": "unused",
                    }
                ),
            ]
        )
        with patch(
            _PATCH_TARGET, lambda self_seat, policy: _UnserializableAdapter(self_seat)
        ):
            with self.assertRaises(ProtocolError):
                _run(drive_validation_session(session, transport))
        self.assertEqual(transport.sent, [])


class SecretHandlingTest(unittest.TestCase):
    def test_fake_transport_never_carries_a_token(self) -> None:
        # FakeTransportはtoken/Authorization headerを一切保持しない
        # protocolであることを、その属性から確認する(secret保存禁止)。
        transport = FakeTransport([])
        self.assertFalse(hasattr(transport, "token"))
        self.assertFalse(hasattr(transport, "authorization"))


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as trace_file:
        return [json.loads(line) for line in trace_file if line.strip()]


class _RecordingFakeTraceWriter:
    """`drive_session()`が呼ぶ`record()`だけを記録するtest double。

    実fileを持たず、呼び出し順序・引数だけを確認する。tokenや
    Authorization headerを受け取れないことを、そのconstructor
    signatureが`path`しか持たない`JsonlProtocolTraceWriter`と
    同じ形であることで示す。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, dict]] = []

    def record(self, direction: str, event_type: object, payload) -> None:
        self.calls.append((direction, event_type, dict(payload)))


class _FailingTraceWriter:
    """trace書き込み失敗をsilentに無視しないことを確認するためのtest double。"""

    def record(self, direction: str, event_type: object, payload) -> None:
        raise ProtocolTraceError("simulated trace write failure")


class ProtocolTraceIntegrationTest(unittest.TestCase):
    """`drive_session()`のprotocol trace(Issue #45)を確認する。

    trace引数は`None`が既定であり、既存test(このfile内の他test class)は
    すべて`trace`を渡さずに動作することで、tracing OFF時の既存挙動が
    変わらないことを既に固定している。ここではtracing ONの場合の
    record内容・順序・failure伝播だけを確認する。
    """

    def test_recv_event_is_traced_before_session_handles_it(self) -> None:
        # start_game.idが不正でProtocolErrorになる直前でも、recv traceは
        # 既に記録されている(Issue #45セクション3の要求)。
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [_event_text({"type": "start_game", "id": "not-an-int"})]
        )
        trace = _RecordingFakeTraceWriter()

        with self.assertRaises(ProtocolError):
            _run(drive_validation_session(session, transport, trace=trace))

        self.assertEqual(len(trace.calls), 1)
        direction, event_type, payload = trace.calls[0]
        self.assertEqual(direction, "recv")
        self.assertEqual(event_type, "start_game")
        self.assertEqual(payload, {"type": "start_game", "id": "not-an-int"})

    def test_unknown_event_is_traced_and_forward_compatible_flow_continues(
        self,
    ) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [
                _event_text({"type": "some_future_event", "payload": [1, 2, 3]}),
                _event_text({"type": "validation_result", "passed": True}),
            ]
        )
        trace = _RecordingFakeTraceWriter()

        _run(drive_validation_session(session, transport, trace=trace))

        self.assertTrue(session.status().validation_result_received)
        traced_event_types = [event_type for _, event_type, _ in trace.calls]
        self.assertIn("some_future_event", traced_event_types)
        self.assertIn("validation_result", traced_event_types)

    def test_recv_send_order_is_preserved_in_trace(self) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [
                _event_text({"type": "start_game", "id": 0}),
                _event_text(
                    {
                        "type": "request_action",
                        "request_id": 1,
                        "possible_actions": [],
                        "observation": "unused",
                    }
                ),
                _event_text({"type": "validation_result", "passed": True}),
            ]
        )
        trace = _RecordingFakeTraceWriter()

        with patch(_PATCH_TARGET, _fake_adapter_factory):
            _run(drive_validation_session(session, transport, trace=trace))

        directions = [direction for direction, _, _ in trace.calls]
        event_types = [event_type for _, event_type, _ in trace.calls]
        self.assertEqual(directions, ["recv", "recv", "send", "recv"])
        self.assertEqual(
            event_types,
            ["start_game", "request_action", "dahai", "validation_result"],
        )

    def test_binary_frame_is_not_traced(self) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [
                b"\x00\x01binary-noise",
                _event_text({"type": "validation_result", "passed": True}),
            ]
        )
        trace = _RecordingFakeTraceWriter()

        _run(drive_validation_session(session, transport, trace=trace))

        self.assertEqual(len(trace.calls), 1)
        self.assertEqual(trace.calls[0][1], "validation_result")

    def test_serialization_failure_is_not_traced_as_sent(self) -> None:
        class _UnserializableAdapter(_FakeAdapter):
            def process_request_action(self, raw_request_action):
                return SendReadyResponse(
                    request_id=raw_request_action["request_id"],
                    action={"type": "dahai", "actor": 0, "bad": object()},
                )

        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [
                _event_text({"type": "start_game", "id": 0}),
                _event_text(
                    {
                        "type": "request_action",
                        "request_id": 1,
                        "possible_actions": [],
                        "observation": "unused",
                    }
                ),
            ]
        )
        trace = _RecordingFakeTraceWriter()

        with patch(
            _PATCH_TARGET, lambda self_seat, policy: _UnserializableAdapter(self_seat)
        ):
            with self.assertRaises(ProtocolError):
                _run(drive_validation_session(session, transport, trace=trace))

        self.assertEqual(transport.sent, [])
        self.assertNotIn("send", [direction for direction, _, _ in trace.calls])

    def test_send_is_traced_even_if_the_actual_transport_send_fails(self) -> None:
        # trace recordは「送信を試みた」ことを表し、「相手へ届いた」ことを
        # 保証しない(docs/riichilab-client.md「protocol trace」参照)。
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport(
            [
                _event_text({"type": "start_game", "id": 0}),
                _event_text(
                    {
                        "type": "request_action",
                        "request_id": 1,
                        "possible_actions": [],
                        "observation": "unused",
                    }
                ),
            ]
        )
        transport._send_should_fail = True
        trace = _RecordingFakeTraceWriter()

        with patch(_PATCH_TARGET, _fake_adapter_factory):
            with self.assertRaises(TransportError):
                _run(drive_validation_session(session, transport, trace=trace))

        self.assertIn("send", [direction for direction, _, _ in trace.calls])

    def test_trace_writer_failure_is_not_silently_ignored(self) -> None:
        session = ValidationSession(MinimalPolicy())
        transport = FakeTransport([_event_text({"type": "start_game", "id": 0})])
        with self.assertRaises(RiichiLabClientError):
            _run(
                drive_validation_session(
                    session, transport, trace=_FailingTraceWriter()
                )
            )

    def test_recv_and_send_records_read_back_from_real_jsonl_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "trace.jsonl")
            trace = JsonlProtocolTraceWriter(path)
            session = ValidationSession(MinimalPolicy())
            transport = FakeTransport(
                [
                    _event_text({"type": "start_game", "id": 0}),
                    _event_text(
                        {
                            "type": "request_action",
                            "request_id": 1,
                            "possible_actions": [],
                            "observation": "unused",
                        }
                    ),
                    _event_text({"type": "validation_result", "passed": True}),
                ]
            )

            try:
                with patch(_PATCH_TARGET, _fake_adapter_factory):
                    _run(drive_validation_session(session, transport, trace=trace))
            finally:
                trace.close()

            records = _read_jsonl(path)
            self.assertEqual(len(records), 4)
            for record in records:
                self.assertIn("timestamp", record)
                self.assertIn("direction", record)
                self.assertIn("event_type", record)
                self.assertIn("payload", record)
            self.assertEqual(
                [record["direction"] for record in records],
                ["recv", "recv", "send", "recv"],
            )

    def test_no_token_or_authorization_key_ever_reaches_the_trace_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "trace.jsonl")
            trace = JsonlProtocolTraceWriter(path)
            session = ValidationSession(MinimalPolicy())
            transport = FakeTransport(
                [
                    _event_text({"type": "start_game", "id": 0}),
                    _event_text({"type": "validation_result", "passed": True}),
                ]
            )

            try:
                _run(drive_validation_session(session, transport, trace=trace))
            finally:
                trace.close()

            with open(path, encoding="utf-8") as trace_file:
                raw_text = trace_file.read()
            self.assertNotIn("token", raw_text.lower())
            self.assertNotIn("authorization", raw_text.lower())
            self.assertNotIn("bearer", raw_text.lower())


if __name__ == "__main__":
    unittest.main()
