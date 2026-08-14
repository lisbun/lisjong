import unittest
from unittest.mock import patch

from riichienv import ActionType, RiichiEnv

from lisjong.policies import MinimalPolicy
from lisjong.policy_contract.action import PassAction
from lisjong.policy_contract.policy_execution import PolicyActionValidationError
from lisjong.policy_contract.seat import Seat
from lisjong.riichienv_adapter.errors import AdapterSyncError
from lisjong.riichienv_adapter.tile_conversion import (
    tile_from_physical_id,
    tile_to_mjai,
)
from lisjong.riichilab_adapter.adapter import RiichiLabSeatAdapter, SendReadyResponse
from lisjong.riichilab_adapter.errors import (
    PossibleActionsValidationError,
    ProtocolConversionError,
    SeatMismatchError,
)

_CALL_TYPES = {"chi", "pon", "daiminkan"}


def _server_style_request_action(observation, request_id):
    """testが期待するcandidate表現(tsumogiri/target付き)をobservationから作る。

    RiichiEnv `Action.to_mjai()`はhoraの`pai`、chi/pon/daiminkan/ronの
    `target`を含まない(Issue #38実測、`build_mjai_response`と同じ欠落)ため、
    本物のRiichiLab serverが提示するであろう完全な候補に近づけて補う。

    また、RiichiEnvの生の`legal_actions()`は、同じsemantic Actionに対応する
    複数のphysical候補(例: 手牌中の2枚の同一牌)を別々のAction objectとして
    返す(docs/action-identity.mdの「semantic aggregation」が既存
    `RiichiEnvActionMappingSession`側で吸収している事実と同じ)。実際の
    RiichiLab serverは合法選択肢を意味単位で提示すると想定されるため、この
    test helperでも同一semantic表現の重複candidateを1件へ集約する。
    """
    import json

    legal = observation.legal_actions()
    seen = set()
    possible_actions = []
    for action in legal:
        candidate = json.loads(action.to_mjai())
        candidate["actor"] = action.actor

        if candidate["type"] == "dahai":
            candidate["tsumogiri"] = action.tile == observation.drawn_tile
        elif candidate["type"] in _CALL_TYPES:
            candidate["target"] = observation.last_discard
            candidate["pai"] = tile_to_mjai(tile_from_physical_id(action.tile))
        elif candidate["type"] == "hora":
            candidate["pai"] = tile_to_mjai(tile_from_physical_id(action.tile))
            candidate["target"] = (
                action.actor
                if action.action_type == ActionType.TSUMO
                else observation.last_discard
            )

        dedupe_key = json.dumps(candidate, sort_keys=True)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        possible_actions.append(candidate)

    return {
        "type": "request_action",
        "request_id": request_id,
        "possible_actions": possible_actions,
        "observation": observation.serialize_to_base64(),
    }


def _reset_observations(seed=1, game_mode="4p-red-east"):
    env = RiichiEnv(seed=seed, game_mode=game_mode)
    return env, env.reset()


class RiichiLabSeatAdapterConstructionTest(unittest.TestCase):
    def test_rejects_non_seat_self_seat(self) -> None:
        with self.assertRaises(TypeError):
            RiichiLabSeatAdapter(0, MinimalPolicy())


class RiichiLabSeatAdapterRoundTripTest(unittest.TestCase):
    def test_processes_a_single_request_action_end_to_end(self) -> None:
        _env, observations = _reset_observations()
        player_id, observation = next(iter(observations.items()))
        seat = Seat(player_id)
        adapter = RiichiLabSeatAdapter(seat, MinimalPolicy())

        request = _server_style_request_action(observation, request_id=1)
        response = adapter.process_request_action(request)

        self.assertIsInstance(response, SendReadyResponse)
        self.assertEqual(response.request_id, 1)
        self.assertIn("type", response.action)
        self.assertEqual(response.action["actor"], player_id)

    def test_current_request_id_is_echoed_without_being_generated(self) -> None:
        _env, observations = _reset_observations(seed=2)
        player_id, observation = next(iter(observations.items()))
        seat = Seat(player_id)
        adapter = RiichiLabSeatAdapter(seat, MinimalPolicy())

        request = _server_style_request_action(observation, request_id="server-req-9")
        response = adapter.process_request_action(request)

        self.assertEqual(response.request_id, "server-req-9")

    def test_request_id_and_transport_metadata_do_not_reach_the_policy(self) -> None:
        seen_decisions = []

        class _RecordingPolicy:
            def choose_action(self, decision):
                seen_decisions.append(decision)
                return MinimalPolicy().choose_action(decision)

        _env, observations = _reset_observations(seed=3)
        player_id, observation = next(iter(observations.items()))
        seat = Seat(player_id)
        adapter = RiichiLabSeatAdapter(seat, _RecordingPolicy())

        request = _server_style_request_action(observation, request_id=77)
        request["time"] = {"grace_ms": 500}
        adapter.process_request_action(request)

        self.assertEqual(len(seen_decisions), 1)
        decision = seen_decisions[0]
        # DecisionContextはinput(PolicyInput)とlegal_actionsだけを持つ。
        self.assertTrue(hasattr(decision, "input"))
        self.assertTrue(hasattr(decision, "legal_actions"))
        self.assertFalse(hasattr(decision, "request_id"))
        self.assertFalse(hasattr(decision, "time"))
        self.assertFalse(hasattr(decision, "possible_actions"))


class RiichiLabSeatAdapterStatefulRuntimeTest(unittest.TestCase):
    def test_reuses_the_same_tracker_and_mapping_session_across_requests(self) -> None:
        env, observations = _reset_observations(seed=12345, game_mode="4p-red-half")
        adapters = {seat: RiichiLabSeatAdapter(seat, MinimalPolicy()) for seat in Seat}
        trackers = {seat: adapters[seat]._tracker for seat in Seat}
        mapping_sessions = {seat: adapters[seat]._mapping_session for seat in Seat}

        steps = 0
        while not env.done() and steps < 500:
            actions = {}
            for player_id, observation in observations.items():
                seat = Seat(player_id)
                request = _server_style_request_action(
                    observation, request_id=steps * 10 + player_id
                )
                response = adapters[seat].process_request_action(request)
                actions[player_id] = _resolve_for_env(observation, response)
            observations = env.step(actions)
            steps += 1

        self.assertGreater(steps, 0)
        for seat in Seat:
            # runtimeが同一instanceのまま継続していることを確認する。
            self.assertIs(adapters[seat]._tracker, trackers[seat])
            self.assertIs(adapters[seat]._mapping_session, mapping_sessions[seat])

    def test_cross_seat_observation_is_rejected(self) -> None:
        _env, observations = _reset_observations(seed=4)
        player_id, observation = next(iter(observations.items()))
        other_seat = Seat((player_id + 1) % 4)
        adapter = RiichiLabSeatAdapter(other_seat, MinimalPolicy())

        request = _server_style_request_action(observation, request_id=1)

        with self.assertRaises(SeatMismatchError):
            adapter.process_request_action(request)


def _resolve_for_env(observation, response):
    """testのgame進行用に、送信済みresponseと同じ選択をRiichiEnv Actionへ戻す。

    productionのAdapterは`resolve()`をすでに内部で1回消費しているため、この
    helperはtest用に`observation.legal_actions()`から同じtype/actor/牌の
    Actionを再取得するだけであり、Adapterの公開契約には含まれない。
    """
    import json

    response_type = response.action.get("type")
    response_actor = response.action.get("actor")
    response_pai = response.action.get("pai")

    for action in observation.legal_actions():
        candidate_type = json.loads(action.to_mjai()).get("type")
        if candidate_type != response_type or action.actor != response_actor:
            continue
        if response_pai is None:
            return action
        if tile_to_mjai(tile_from_physical_id(action.tile)) == response_pai:
            return action
    # 一致しない場合はテスト側の不整合であり、明示的に失敗させる。
    raise AssertionError("could not resolve a matching RiichiEnv action for test")


class RiichiLabSeatAdapterFailClosedTest(unittest.TestCase):
    def test_observation_seat_mismatch_produces_no_payload(self) -> None:
        _env, observations = _reset_observations(seed=5)
        player_id, observation = next(iter(observations.items()))
        adapter = RiichiLabSeatAdapter(Seat((player_id + 1) % 4), MinimalPolicy())
        request = _server_style_request_action(observation, request_id=1)

        with self.assertRaises(SeatMismatchError):
            adapter.process_request_action(request)

    def test_build_decision_failure_produces_no_payload(self) -> None:
        _env, observations = _reset_observations(seed=6)
        player_id, observation = next(iter(observations.items()))
        seat = Seat(player_id)
        adapter = RiichiLabSeatAdapter(seat, MinimalPolicy())
        request = _server_style_request_action(observation, request_id=1)

        with patch(
            "lisjong.riichilab_adapter.adapter.build_decision",
            side_effect=AdapterSyncError("boom"),
        ):
            with self.assertRaises(AdapterSyncError):
                adapter.process_request_action(request)

    def test_policy_exception_propagates_without_fallback(self) -> None:
        class _RaisingPolicy:
            def choose_action(self, decision):
                raise RuntimeError("policy exploded")

        _env, observations = _reset_observations(seed=8)
        player_id, observation = next(iter(observations.items()))
        seat = Seat(player_id)
        adapter = RiichiLabSeatAdapter(seat, _RaisingPolicy())
        request = _server_style_request_action(observation, request_id=1)

        with self.assertRaises(RuntimeError):
            adapter.process_request_action(request)

    def test_execute_policy_validation_failure_produces_no_payload(self) -> None:
        class _IllegalPolicy:
            def choose_action(self, decision):
                return PassAction(actor=decision.input.self_seat)

        _env, observations = _reset_observations(seed=9)
        player_id, observation = next(iter(observations.items()))
        seat = Seat(player_id)
        adapter = RiichiLabSeatAdapter(seat, _IllegalPolicy())
        request = _server_style_request_action(observation, request_id=1)

        # 初期打牌局面ではPassActionは合法候補に含まれないため、
        # execute_policy()がPolicyActionValidationErrorで拒否するはずである。
        with self.assertRaises(PolicyActionValidationError):
            adapter.process_request_action(request)

    def test_mjai_conversion_failure_produces_no_payload(self) -> None:
        _env, observations = _reset_observations(seed=10)
        player_id, observation = next(iter(observations.items()))
        seat = Seat(player_id)
        adapter = RiichiLabSeatAdapter(seat, MinimalPolicy())
        request = _server_style_request_action(observation, request_id=1)

        with patch(
            "lisjong.riichilab_adapter.adapter.build_mjai_response",
            side_effect=ProtocolConversionError("boom"),
        ):
            with self.assertRaises(ProtocolConversionError):
                adapter.process_request_action(request)

    def test_possible_actions_mismatch_produces_no_payload(self) -> None:
        _env, observations = _reset_observations(seed=11)
        player_id, observation = next(iter(observations.items()))
        seat = Seat(player_id)
        adapter = RiichiLabSeatAdapter(seat, MinimalPolicy())
        request = _server_style_request_action(observation, request_id=1)
        # possible_actionsを、選択され得ない候補だけへ差し替える。
        request["possible_actions"] = [{"type": "ryukyoku", "actor": player_id}]

        with self.assertRaises(PossibleActionsValidationError):
            adapter.process_request_action(request)


if __name__ == "__main__":
    unittest.main()
