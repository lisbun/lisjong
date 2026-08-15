"""RiichiLab ranked 1半荘lifecycleのunit / fake transport / integration test。"""

import asyncio
import json
import os
import subprocess
import sys
import unittest
from contextlib import asynccontextmanager
from dataclasses import fields
from unittest.mock import patch

from _riichilab_client_test_helpers import server_style_request_action
from riichienv import RiichiEnv

from lisjong.policies import MinimalPolicy
from lisjong.policy_contract.seat import Seat
from lisjong.riichilab_adapter.adapter import SendReadyResponse
from lisjong.riichilab_client.errors import ProtocolError, UnexpectedDisconnectError
from lisjong.riichilab_client.ranked import run_ranked_game
from lisjong.riichilab_client.session import RankedSession, ValidationSession
from lisjong.riichilab_client.transport import (
    TransportClosed,
    drive_ranked_session,
)

_PATCH_TARGET = "lisjong.riichilab_client.session.RiichiLabSeatAdapter"
_FINAL_SCORES = [30000, 25000, 20000, 25000]


class _FakeAdapter:
    def __init__(self, self_seat) -> None:
        self.self_seat = self_seat

    def process_request_action(self, raw_request_action):
        return SendReadyResponse(
            request_id=raw_request_action["request_id"],
            action={
                "type": "dahai",
                "actor": int(self.self_seat),
                "pai": "1m",
            },
        )


class _FakeTransport:
    def __init__(self, incoming: list) -> None:
        self._incoming = list(incoming)
        self.sent: list[str] = []
        self.recv_count = 0

    async def recv(self):
        self.recv_count += 1
        if not self._incoming:
            raise TransportClosed("no more fake messages queued")
        return self._incoming.pop(0)

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        return None


def _event_text(event: dict) -> str:
    return json.dumps(event)


def _request_action(request_id: int) -> dict:
    return {
        "type": "request_action",
        "request_id": request_id,
        "possible_actions": [],
        "observation": "unused-by-fake-adapter",
    }


def _make_fake_connect(
    transport: _FakeTransport, captured_connections: list[tuple[str, str]]
):
    @asynccontextmanager
    async def _connect(url: str, token: str):
        # 接続自体がqueue参加であり、connector entry時点でsendは0件。
        if transport.sent:
            raise AssertionError("ranked connector sent data before server event")
        captured_connections.append((url, token))
        yield transport

    return _connect


class RankedSeatBindTest(unittest.TestCase):
    def test_accepts_all_four_seats(self) -> None:
        for seat_id in range(4):
            with self.subTest(seat_id=seat_id):
                session = RankedSession(MinimalPolicy())
                with patch(
                    _PATCH_TARGET,
                    lambda self_seat, policy: _FakeAdapter(self_seat),
                ):
                    session.handle_event({"type": "start_game", "id": seat_id})
                self.assertEqual(session.status().seat, Seat(seat_id))

    def test_rejects_boolean_and_out_of_range_id(self) -> None:
        for seat_id in (False, "1", None, -1, 4):
            with self.subTest(seat_id=seat_id):
                session = RankedSession(MinimalPolicy())
                with self.assertRaises(ProtocolError):
                    session.handle_event({"type": "start_game", "id": seat_id})

    def test_does_not_fallback_to_seat_field(self) -> None:
        session = RankedSession(MinimalPolicy())
        with self.assertRaises(ProtocolError):
            session.handle_event({"type": "start_game", "seat": 2})

    def test_duplicate_same_seat_keeps_one_adapter(self) -> None:
        created = []

        def factory(self_seat, policy):
            adapter = _FakeAdapter(self_seat)
            created.append(adapter)
            return adapter

        session = RankedSession(MinimalPolicy())
        with patch(_PATCH_TARGET, factory):
            session.handle_event({"type": "start_game", "id": 3})
            first_adapter = session._adapter
            session.handle_event({"type": "start_game", "id": 3})

        self.assertEqual(len(created), 1)
        self.assertIs(session._adapter, first_adapter)

    def test_duplicate_different_seat_fails_closed(self) -> None:
        session = RankedSession(MinimalPolicy())
        with patch(_PATCH_TARGET, lambda self_seat, policy: _FakeAdapter(self_seat)):
            session.handle_event({"type": "start_game", "id": 1})
            with self.assertRaises(ProtocolError):
                session.handle_event({"type": "start_game", "id": 2})

    def test_validation_still_rejects_non_zero_seat(self) -> None:
        session = ValidationSession(MinimalPolicy())
        with self.assertRaises(ProtocolError):
            session.handle_event({"type": "start_game", "id": 1})

    def test_ranked_uses_the_common_monotonic_request_id_contract(self) -> None:
        session = RankedSession(MinimalPolicy())
        with patch(_PATCH_TARGET, lambda self_seat, policy: _FakeAdapter(self_seat)):
            session.handle_event({"type": "start_game", "id": 0})
            session.handle_event(_request_action(1))
            session.handle_event(_request_action(37))
            with self.assertRaises(ProtocolError):
                session.handle_event(_request_action(37))


class RankedTerminalTest(unittest.TestCase):
    def _started_session(self) -> RankedSession:
        session = RankedSession(MinimalPolicy())
        with patch(_PATCH_TARGET, lambda self_seat, policy: _FakeAdapter(self_seat)):
            session.handle_event({"type": "start_game", "id": 0})
        return session

    def test_end_game_is_ranked_terminal_and_captures_scores(self) -> None:
        session = self._started_session()
        session.handle_event({"type": "end_game", "scores": _FINAL_SCORES})
        self.assertTrue(session.is_complete)
        self.assertTrue(session.status().end_game_received)
        self.assertEqual(session.status().scores, tuple(_FINAL_SCORES))

    def test_validation_end_game_is_not_terminal(self) -> None:
        session = ValidationSession(MinimalPolicy())
        session.handle_event({"type": "end_game", "scores": _FINAL_SCORES})
        self.assertFalse(session.is_complete)
        self.assertFalse(session.validation_result_received)

    def test_end_game_before_start_game_fails_closed(self) -> None:
        session = RankedSession(MinimalPolicy())
        with self.assertRaises(ProtocolError):
            session.handle_event({"type": "end_game", "scores": _FINAL_SCORES})

    def test_malformed_final_scores_fail_closed(self) -> None:
        for scores in (None, [25000] * 3, [25000, 25000, 25000, True]):
            with self.subTest(scores=scores):
                session = self._started_session()
                with self.assertRaises(ProtocolError):
                    session.handle_event({"type": "end_game", "scores": scores})

    def test_missing_scores_reports_only_event_shape(self) -> None:
        session = self._started_session()
        sentinel = "do-not-leak-this-value"
        event = {
            "type": "end_game",
            "final_scores": {"nested": sentinel},
            "metadata": sentinel,
        }

        with self.assertRaises(ProtocolError) as caught:
            session.handle_event(event)

        message = str(caught.exception)
        self.assertIn(
            "event_keys=['final_scores', 'metadata', 'type']",
            message,
        )
        self.assertIn("scores_type=NoneType", message)
        self.assertIn("scores_length=None", message)
        self.assertNotIn(sentinel, message)

    def test_non_list_scores_reports_type_without_values(self) -> None:
        session = self._started_session()
        sentinel = "do-not-leak-this-value"

        with self.assertRaises(ProtocolError) as caught:
            session.handle_event({"type": "end_game", "scores": {"nested": sentinel}})

        message = str(caught.exception)
        self.assertIn("event_keys=['scores', 'type']", message)
        self.assertIn("scores_type=dict", message)
        self.assertIn("scores_length=None", message)
        self.assertNotIn(sentinel, message)

    def test_list_scores_reports_length_without_values(self) -> None:
        session = self._started_session()
        sentinel = "do-not-leak-this-value"

        with self.assertRaises(ProtocolError) as caught:
            session.handle_event({"type": "end_game", "scores": [1, sentinel, 3]})

        message = str(caught.exception)
        self.assertIn("event_keys=['scores', 'type']", message)
        self.assertIn("scores_type=list", message)
        self.assertIn("scores_length=3", message)
        self.assertNotIn(sentinel, message)

    def test_invalid_score_element_reports_shape_without_values(self) -> None:
        session = self._started_session()
        sentinel = "do-not-leak-this-value"

        with self.assertRaises(ProtocolError) as caught:
            session.handle_event(
                {"type": "end_game", "scores": [30000, 25000, 20000, sentinel]}
            )

        message = str(caught.exception)
        self.assertIn("ranked end_game scores must be integers", message)
        self.assertIn("event_keys=['scores', 'type']", message)
        self.assertIn("scores_type=list", message)
        self.assertIn("scores_length=4", message)
        self.assertNotIn(sentinel, message)


class RankedFakeTransportTest(unittest.TestCase):
    def test_no_join_payload_and_exactly_one_game(self) -> None:
        session = RankedSession(MinimalPolicy())
        transport = _FakeTransport(
            [
                _event_text({"type": "start_game", "id": 2}),
                _event_text(_request_action(10)),
                _event_text(
                    {"type": "action_ack", "request_id": 10, "status": "accepted"}
                ),
                _event_text({"type": "end_game", "scores": _FINAL_SCORES}),
                _event_text({"type": "start_game", "id": 1}),
            ]
        )
        with patch(_PATCH_TARGET, lambda self_seat, policy: _FakeAdapter(self_seat)):
            asyncio.run(drive_ranked_session(session, transport))

        sent = [json.loads(message) for message in transport.sent]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["request_id"], 10)
        self.assertNotEqual(sent[0].get("type"), "join")
        self.assertEqual(transport.recv_count, 4)
        self.assertEqual(len(transport._incoming), 1)

    def test_binary_and_unknown_event_are_ignored(self) -> None:
        session = RankedSession(MinimalPolicy())
        transport = _FakeTransport(
            [
                b"binary",
                _event_text({"type": "future_queue_event"}),
                _event_text({"type": "start_game", "id": 0}),
                _event_text({"type": "end_game", "scores": _FINAL_SCORES}),
            ]
        )
        with patch(_PATCH_TARGET, lambda self_seat, policy: _FakeAdapter(self_seat)):
            asyncio.run(drive_ranked_session(session, transport))
        self.assertTrue(session.is_complete)

    def test_disconnect_before_end_game_is_failure(self) -> None:
        session = RankedSession(MinimalPolicy())
        transport = _FakeTransport([_event_text({"type": "start_game", "id": 0})])
        with patch(_PATCH_TARGET, lambda self_seat, policy: _FakeAdapter(self_seat)):
            with self.assertRaises(UnexpectedDisconnectError):
                asyncio.run(drive_ranked_session(session, transport))


class RankedMinimalPolicyIntegrationTest(unittest.TestCase):
    def test_request_uses_existing_adapter_policy_and_validation_boundary(self) -> None:
        env = RiichiEnv(seed=42, game_mode="4p-red-east")
        observations = env.reset()
        player_id, observation = next(iter(observations.items()))
        seen_decisions = []

        class _RecordingPolicy:
            def choose_action(self, decision):
                seen_decisions.append(decision)
                return MinimalPolicy().choose_action(decision)

        session = RankedSession(_RecordingPolicy())
        session.handle_event({"type": "start_game", "id": player_id})
        outgoing = session.handle_event(
            server_style_request_action(observation, request_id=37)
        )

        self.assertEqual(outgoing["request_id"], 37)
        self.assertIn("type", outgoing)
        self.assertEqual(len(seen_decisions), 1)
        decision = seen_decisions[0]
        for leaked_attr in (
            "request_id",
            "time",
            "possible_actions",
            "ack",
            "transport",
        ):
            self.assertFalse(hasattr(decision, leaked_attr))


class RunRankedGameTest(unittest.TestCase):
    def test_completes_one_hanchan_with_secret_safe_result(self) -> None:
        env = RiichiEnv(seed=7, game_mode="4p-red-east")
        observations = env.reset()
        player_id, observation = next(iter(observations.items()))
        request = server_style_request_action(observation, request_id=5)
        transport = _FakeTransport(
            [
                _event_text({"type": "start_game", "id": player_id}),
                _event_text(request),
                _event_text(
                    {"type": "action_ack", "request_id": 5, "status": "accepted"}
                ),
                _event_text({"type": "end_game", "scores": _FINAL_SCORES}),
            ]
        )
        captured_connections: list[tuple[str, str]] = []

        async def _run():
            with patch(
                "lisjong.riichilab_client.ranked.connect_ranked_transport",
                _make_fake_connect(transport, captured_connections),
            ):
                return await run_ranked_game(
                    MinimalPolicy(), "fake-token", url="wss://example.invalid/ranked"
                )

        result = asyncio.run(_run())
        self.assertTrue(result.end_game_received)
        self.assertEqual(result.seat, Seat(player_id))
        self.assertEqual(result.requests_received, 1)
        self.assertEqual(result.responses_sent, 1)
        self.assertEqual(result.ack_history[5], ("accepted",))
        self.assertEqual(result.scores, tuple(_FINAL_SCORES))
        self.assertEqual(
            captured_connections,
            [("wss://example.invalid/ranked", "fake-token")],
        )
        for field in fields(result):
            self.assertNotEqual(getattr(result, field.name), "fake-token")

    def test_rejects_empty_token(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(run_ranked_game(MinimalPolicy(), ""))


class RankedModuleCliTest(unittest.TestCase):
    def test_module_cli_does_not_emit_runtime_warning(self) -> None:
        environment = os.environ.copy()
        environment.pop("BOT_TOKEN", None)
        completed = subprocess.run(
            [sys.executable, "-m", "lisjong.riichilab_client.ranked"],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("BOT_TOKEN environment variable is not set", completed.stderr)
        self.assertNotIn("RuntimeWarning", completed.stderr)

    def test_package_root_lazy_exports_ranked_api(self) -> None:
        from lisjong import riichilab_client
        from lisjong.riichilab_client.ranked import RankedGameResult

        self.assertIs(riichilab_client.run_ranked_game, run_ranked_game)
        self.assertIs(riichilab_client.RankedGameResult, RankedGameResult)


if __name__ == "__main__":
    unittest.main()
