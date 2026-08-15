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
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager, redirect_stdout
from dataclasses import fields
from io import StringIO
from unittest.mock import patch

from _riichilab_client_test_helpers import resolve_for_env, server_style_request_action
from riichienv import RiichiEnv

from lisjong.policies import MinimalPolicy, UkeirePolicy
from lisjong.policy_contract.seat import Seat
from lisjong.riichilab_client.errors import UnexpectedDisconnectError
from lisjong.riichilab_client.transport import TransportClosed
from lisjong.riichilab_client.validation import (
    ValidationResult,
    _run_cli,
    run_validation,
)


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


class ValidationModuleCliTest(unittest.TestCase):
    def test_module_cli_without_profile_fails_closed_without_runtime_warning(
        self,
    ) -> None:
        environment = os.environ.copy()

        completed = subprocess.run(
            [sys.executable, "-m", "lisjong.riichilab_client.validation"],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("--profile", completed.stderr)
        self.assertNotIn("RuntimeWarning", completed.stderr)

    def test_module_cli_rejects_unknown_profile(self) -> None:
        environment = os.environ.copy()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "lisjong.riichilab_client.validation",
                "--profile",
                "lisjong-production",
            ],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("invalid choice", completed.stderr)
        self.assertNotIn("RuntimeWarning", completed.stderr)

    def test_module_cli_missing_profile_credential_fails_closed(self) -> None:
        environment = os.environ.copy()
        environment.pop("LISJONG_BASELINE_BOT_TOKEN", None)

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "lisjong.riichilab_client.validation",
                "--profile",
                "lisjong-baseline",
            ],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("LISJONG_BASELINE_BOT_TOKEN", completed.stderr)
        self.assertNotIn("RuntimeWarning", completed.stderr)

    def test_package_root_keeps_lazy_validation_exports(self) -> None:
        from lisjong import riichilab_client
        from lisjong.riichilab_client.validation import ValidationResult

        self.assertIs(riichilab_client.run_validation, run_validation)
        self.assertIs(riichilab_client.ValidationResult, ValidationResult)


class ValidationModuleCliRuntimeTest(unittest.TestCase):
    """`_run_cli()`のprofile解決・runtime summary表示・secret非露出を、
    実transportなしに確認する(Issue #44、rankedと同じ整合性を維持する)。
    """

    def test_cli_prints_secret_free_summary_and_uses_profile_policy(self) -> None:
        result = ValidationResult(
            passed=True,
            validation_result_received=True,
            end_game_received=True,
            failure_reason=None,
            requests_received=1,
            responses_sent=1,
            ack_history={},
        )
        captured_policies: list = []

        async def _fake_run_validation(policy, token, **kwargs):
            captured_policies.append(policy)
            return result

        output = StringIO()
        with (
            patch.dict(os.environ, {"LISJONG_DEV_BOT_TOKEN": "fake-token"}),
            patch(
                "lisjong.riichilab_client.validation.run_validation",
                _fake_run_validation,
            ),
            redirect_stdout(output),
        ):
            return_code = _run_cli(["--profile", "lisjong-dev"])

        self.assertEqual(return_code, 0)
        self.assertEqual(len(captured_policies), 1)
        self.assertIsInstance(captured_policies[0], UkeirePolicy)
        self.assertIn("profile: lisjong-dev", output.getvalue())
        self.assertIn("policy: UkeirePolicy", output.getvalue())
        self.assertIn("mode: validation", output.getvalue())
        self.assertIn("trace: off", output.getvalue())
        self.assertIn("RiichiLab validation passed", output.getvalue())
        self.assertNotIn("fake-token", output.getvalue())


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
        session.handle_event({"type": "start_game", "id": int(seat)})

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
            _event_text({"type": "start_game", "id": 0}),
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
        transport = _FakeTransport([_event_text({"type": "start_game", "id": 0})])
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
        session.handle_event({"type": "start_game", "id": 0})

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


class RunValidationTraceOptInTest(unittest.TestCase):
    """`run_validation(..., trace_path=...)`のopt-in protocol trace(Issue #45)。"""

    def _run(self, *, trace_path):
        env = RiichiEnv(seed=7, game_mode="4p-red-east")
        observations = env.reset()
        seat0_player_id = next(
            player_id for player_id in observations if Seat(player_id) == Seat.SEAT_0
        )
        observation = observations[seat0_player_id]
        request_1 = server_style_request_action(observation, request_id=1)
        incoming = [
            _event_text({"type": "start_game", "id": 0}),
            _event_text(request_1),
            _event_text({"type": "action_ack", "request_id": 1, "status": "accepted"}),
            _event_text({"type": "end_game"}),
            _event_text({"type": "validation_result", "passed": True}),
        ]
        transport = _FakeTransport(incoming)
        captured_tokens: list[str] = []

        async def _run_inner():
            with patch(
                "lisjong.riichilab_client.validation.connect_validation_transport",
                _make_fake_connect(transport, captured_tokens),
            ):
                return await run_validation(
                    MinimalPolicy(), "fake-token", trace_path=trace_path
                )

        return asyncio.run(_run_inner())

    def test_tracing_off_by_default_creates_no_trace_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = os.path.join(tmp_dir, "trace.jsonl")
            result = self._run(trace_path=None)
            self.assertTrue(result.passed)
            self.assertFalse(os.path.exists(trace_path))

    def test_trace_path_opt_in_writes_jsonl_records_without_the_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = os.path.join(tmp_dir, "trace.jsonl")
            result = self._run(trace_path=trace_path)
            self.assertTrue(result.passed)

            with open(trace_path, encoding="utf-8") as trace_file:
                lines = [line for line in trace_file if line.strip()]

            self.assertGreater(len(lines), 0)
            records = [json.loads(line) for line in lines]
            for record in records:
                self.assertIn("timestamp", record)
                self.assertIn("direction", record)
                self.assertIn("event_type", record)
                self.assertIn("payload", record)
            raw_text = "".join(lines)
            self.assertNotIn("fake-token", raw_text)
            self.assertNotIn("Authorization", raw_text)


if __name__ == "__main__":
    unittest.main()
