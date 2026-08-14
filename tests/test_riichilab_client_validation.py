"""#38 `RiichiLabSeatAdapter` + `MinimalPolicy`統合test、および`run_validation()`
end-to-end test(Issue #39)。

実RiichiEnv 0.4.8が生成する`Observation`を使い、fake WebSocket transportの
`start_game` / `request_action`から`RiichiLabSeatAdapter` /
`MinimalPolicy`を経て送信前validation済みMJAI responseまで届くことを
確認する。あわせて、Policyへ`request_id` / `time` / `ack` / WebSocket
objectが一切漏れないこと、`possible_actions`が#38外部validation専用の
ままであることを確認する。

実RiichiLab tokenやnetworkは使用しない。`run_validation()`の
`connect_validation_transport`だけをfake transportへ差し替える。
"""

import asyncio
import json
import unittest
from contextlib import asynccontextmanager
from dataclasses import fields
from unittest.mock import patch

from _riichilab_client_test_helpers import resolve_for_env, server_style_request_action
from riichienv import RiichiEnv

from lisjong.policies import MinimalPolicy
from lisjong.policy_contract.seat import Seat
from lisjong.riichilab_client.errors import UnexpectedDisconnectError
from lisjong.riichilab_client.transport import TransportClosed
from lisjong.riichilab_client.validation import run_validation


class _FakeTransport:
    def __init__(self, incoming: list) -> None:
        self._incoming = list(incoming)
        self.sent: list[str] = []
        self.closed = False

    async def recv(self):
        if not self._incoming:
            raise TransportClosed("no more fake messages queued")
        return self._incoming.pop(0)

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True


def _event_text(event: dict) -> str:
    return json.dumps(event)


def _make_fake_connect(transport: _FakeTransport, captured_tokens: list):
    """`connect_validation_transport(url, token)`と同じ形の呼び出し規約を
    持つfake connectorを作る。tokenはcapture用listへ記録するだけで、
    fake transport自体はtokenを一切保持しない。
    """

    @asynccontextmanager
    async def _connect(url: str, token: str):
        captured_tokens.append(token)
        yield transport

    return _connect


class SeatAdapterMinimalPolicyIntegrationTest(unittest.TestCase):
    """fake serverの`request_action`が、#38 Adapter経由でsend-ready MJAI
    responseまで届くことを確認する(pure transport test、asyncioなし)。
    """

    def test_request_action_round_trips_through_adapter_and_policy(self) -> None:
        from lisjong.riichilab_client.session import ValidationSession

        env = RiichiEnv(seed=42, game_mode="4p-red-east")
        observations = env.reset()
        player_id, observation = next(iter(observations.items()))
        seat = Seat(player_id)

        seen_decisions = []

        class _RecordingPolicy:
            def choose_action(self, decision):
                seen_decisions.append(decision)
                return MinimalPolicy().choose_action(decision)

        session = ValidationSession(_RecordingPolicy())
        session.handle_event({"type": "start_game", "seat": int(seat)})

        request = server_style_request_action(observation, request_id=1)
        outgoing = session.handle_event(request)

        self.assertIsNotNone(outgoing)
        self.assertEqual(outgoing["request_id"], 1)
        self.assertIn("type", outgoing)

        # Policyへ渡るのはDecisionContextのみで、request_id/time/ack/
        # WebSocket固有情報は一切漏れない。
        self.assertEqual(len(seen_decisions), 1)
        decision = seen_decisions[0]
        self.assertTrue(hasattr(decision, "input"))
        self.assertTrue(hasattr(decision, "legal_actions"))
        for leaked_attr in (
            "request_id",
            "time",
            "possible_actions",
            "ack",
            "transport",
        ):
            self.assertFalse(hasattr(decision, leaked_attr))


class RunValidationEndToEndTest(unittest.TestCase):
    """`run_validation()`をfake transportで駆動するend-to-end test。

    複数requestにわたる`start_game` -> `request_action`(x2) ->
    `action_ack` -> `end_game` -> `validation_result`の順序で、
    実RiichiEnv Observationを使ったfake serverの完走を確認する。
    """

    def test_run_validation_completes_and_reports_passed(self) -> None:
        env = RiichiEnv(seed=7, game_mode="4p-red-east")
        observations = env.reset()
        seat0_player_id = next(
            player_id for player_id in observations if Seat(player_id) == Seat.SEAT_0
        )
        observation = observations[seat0_player_id]

        request_1 = server_style_request_action(observation, request_id=1)
        incoming = [
            _event_text({"type": "start_game", "seat": 0}),
            _event_text(request_1),
        ]

        transport = _FakeTransport(incoming)
        captured_tokens: list[str] = []

        async def _drive_and_finish():
            # request_action送信後にaction_ackとend_game/validation_resultを
            # 積む(実際のfake serverはbotのsendを見てからackを返すが、この
            # testはdrive loop自体をfake serverとして扱わず、event列を
            # 事前に固定して`run_validation()`の公開契約だけを検証する)。
            transport._incoming.append(
                _event_text(
                    {"type": "action_ack", "request_id": 1, "status": "accepted"}
                )
            )
            transport._incoming.append(_event_text({"type": "end_game"}))
            transport._incoming.append(
                _event_text({"type": "validation_result", "passed": True})
            )

        async def _run():
            await _drive_and_finish()
            with patch(
                "lisjong.riichilab_client.validation.connect_validation_transport",
                _make_fake_connect(transport, captured_tokens),
            ):
                return await run_validation(MinimalPolicy(), "fake-token")

        result = asyncio.run(_run())

        self.assertTrue(result.passed)
        self.assertTrue(result.validation_result_received)
        self.assertTrue(result.end_game_received)
        self.assertEqual(result.requests_received, 1)
        self.assertEqual(result.responses_sent, 1)
        self.assertEqual(result.ack_history[1], ("accepted",))
        self.assertEqual(len(transport.sent), 1)

        # tokenはconnect時にだけ使われ、結果側には一切含まれない。
        self.assertEqual(captured_tokens, ["fake-token"])
        for field in fields(result):
            self.assertNotEqual(getattr(result, field.name), "fake-token")

    def test_run_validation_rejects_empty_token(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(run_validation(MinimalPolicy(), ""))

    def test_run_validation_raises_on_unexpected_disconnect(self) -> None:
        transport = _FakeTransport([_event_text({"type": "start_game", "seat": 0})])
        captured_tokens: list[str] = []

        async def _run():
            with patch(
                "lisjong.riichilab_client.validation.connect_validation_transport",
                _make_fake_connect(transport, captured_tokens),
            ):
                return await run_validation(MinimalPolicy(), "fake-token")

        with self.assertRaises(UnexpectedDisconnectError):
            asyncio.run(_run())


class MultiRequestFullGameFakeServerTest(unittest.TestCase):
    """複数seatの完全な半荘進行から、自seat分だけを`ValidationSession`で
    処理し続けられることを確認する(request_idのgap・#38 stateful runtimeの
    継続を、fake serverシナリオに近い形で固定する)。
    """

    def test_processes_many_requests_for_self_seat_across_a_partial_game(self) -> None:
        from lisjong.riichilab_client.session import ValidationSession

        env = RiichiEnv(seed=99, game_mode="4p-red-half")
        observations = env.reset()

        session = ValidationSession(MinimalPolicy())
        session.handle_event({"type": "start_game", "seat": 0})

        request_id = 0
        self_requests_processed = 0
        steps = 0
        while not env.done() and steps < 200 and self_requests_processed < 5:
            actions = {}
            for player_id, observation in observations.items():
                seat = Seat(player_id)
                if seat == Seat.SEAT_0:
                    request_id += 3  # gapを許容することを併せて確認する
                    request = server_style_request_action(
                        observation, request_id=request_id
                    )
                    outgoing = session.handle_event(request)
                    self_requests_processed += 1
                    actions[player_id] = resolve_for_env(observation, outgoing)
                else:
                    # 他seatはこのtestの対象外(RiichiLab server側が処理する
                    # ため、ValidationSessionは他seatのrequest_actionを
                    # 受け取らない)。testの進行のためだけに最初の合法手を選ぶ。
                    actions[player_id] = observation.legal_actions()[0]
            observations = env.step(actions)
            steps += 1

        self.assertGreaterEqual(self_requests_processed, 1)
        status = session.status()
        self.assertEqual(status.requests_received, self_requests_processed)
        self.assertEqual(status.responses_sent, self_requests_processed)


if __name__ == "__main__":
    unittest.main()
