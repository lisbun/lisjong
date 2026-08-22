"""#38 `RiichiLabSeatAdapter` + `MinimalPolicy`統合test(Issue #39)。

実RiichiEnv 0.4.8が生成する`Observation`を使い、fake WebSocket transportの
`start_game` / `request_action`から`RiichiLabSeatAdapter` /
`MinimalPolicy`を経て送信前validation済みMJAI responseまで届くことを
確認する。あわせて、Policyへ`request_id` / `time` / `ack` / WebSocket
objectが一切漏れないこと、`possible_actions`が#38外部validation専用の
ままであることを確認する。

`ValidationResult` / `run_validation()` / validation CLIのorchestration-level
coverageは、Issue #19でのArenaへのcanonical移管に伴いlisbun/lisjong#89で
`lisjong-arena`側のtestへ移した。ここに残るのは`lisjong.riichilab_client`
package自身が所有するlower-level `ValidationSession` lifecycleと#38
Adapter統合のcoverageだけである。
"""

import unittest

from _riichilab_client_test_helpers import resolve_for_env, server_style_request_action
from riichienv import RiichiEnv

from lisjong.policies import MinimalPolicy
from lisjong.policy_contract.seat import Seat
from lisjong.riichilab_client.session import ValidationSession


class SeatAdapterMinimalPolicyIntegrationTest(unittest.TestCase):
    """fake serverの`request_action`が、#38 Adapter経由でsend-ready MJAI
    responseまで届くことを確認する(pure transport test、asyncioなし)。
    """

    def test_request_action_round_trips_through_adapter_and_policy(self) -> None:
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


class MultiRequestFullGameFakeServerTest(unittest.TestCase):
    """複数seatの完全な半荘進行から、自seat分だけを`ValidationSession`で
    処理し続けられることを確認する(request_idのgap・#38 stateful runtimeの
    継続を、fake serverシナリオに近い形で固定する)。
    """

    def test_processes_many_requests_for_self_seat_across_a_partial_game(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
