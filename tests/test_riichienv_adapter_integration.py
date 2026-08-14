"""実際のriichienvを使うRiichiEnv Adapterの結合test。

`tests/test_riichienv_adapter_materialized_state.py`と
`tests/test_riichienv_adapter_policy_input.py`はfake Observationで各規則を
個別に確認する。本fileは実際の`RiichiEnv`を使い、複数kyoku・複数game modeに
わたって`build_policy_input()`が例外なく`PolicyInput`を生成できることを
確認する結合testである。

CI実行時間を抑えるため、seed数とstep上限を絞る。より広い範囲の実測は
Issue #28の調査段階で個別に実施済みであり(docs/riichienv-investigation.md)、
ここでは実装の回帰を検出できる最小限の再現に絞る。
"""

import unittest

from riichienv import RiichiEnv

from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.seat import Seat
from lisjong.riichienv_adapter import SeatMaterializedState, build_policy_input

_PREFERRED_ACTION_TYPES = (
    "ActionType.RIICHI",
    "ActionType.DAIMINKAN",
    "ActionType.ANKAN",
    "ActionType.KAKAN",
    "ActionType.RON",
    "ActionType.TSUMO",
    "ActionType.CHI",
    "ActionType.PON",
)


def _choose_action(observation):
    """riichi/kan/call/和了を優先し、実測範囲を広く踏むaction選択方針。"""
    legal_actions = observation.legal_actions()
    preferred = [
        action
        for action in legal_actions
        if str(action.action_type) in _PREFERRED_ACTION_TYPES
    ]
    return preferred[0] if preferred else legal_actions[0]


def _module_is_leaked_from_riichienv(value: object, seen: set[int]) -> bool:
    """valueから再帰的に到達可能なobjectにriichienv由来のものがないか調べる。"""
    if id(value) in seen:
        return False
    seen.add(id(value))

    module_name = type(value).__module__
    if module_name.startswith("riichienv"):
        return True

    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return False
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_module_is_leaked_from_riichienv(item, seen) for item in value)
    if isinstance(value, dict):
        return any(
            _module_is_leaked_from_riichienv(key, seen)
            or _module_is_leaked_from_riichienv(item, seen)
            for key, item in value.items()
        )
    if hasattr(value, "__dict__"):
        return any(
            _module_is_leaked_from_riichienv(item, seen)
            for item in vars(value).values()
        )
    if hasattr(value, "__slots__"):
        return any(
            _module_is_leaked_from_riichienv(getattr(value, slot), seen)
            for slot in value.__slots__
            if hasattr(value, slot)
        )
    return False


class RiichiEnvAdapterIntegrationTest(unittest.TestCase):
    def test_builds_policy_input_across_multiple_kyoku_without_failure(self) -> None:
        for seed in range(1, 6):
            with self.subTest(seed=seed):
                env = RiichiEnv(seed=seed, game_mode="4p-red-east")
                observations = env.reset()
                trackers = {}
                steps = 0
                decisions = 0

                while not env.done() and steps < 500:
                    actions = {}
                    for player_id, observation in observations.items():
                        seat = Seat(player_id)
                        tracker = trackers.setdefault(seat, SeatMaterializedState(seat))

                        policy_input = build_policy_input(tracker, observation)
                        decisions += 1

                        self.assertIsInstance(policy_input, PolicyInput)
                        self.assertEqual(policy_input.self_seat, seat)
                        self.assertFalse(
                            _module_is_leaked_from_riichienv(policy_input, set())
                        )

                        actions[player_id] = _choose_action(observation)
                    observations = env.step(actions)
                    steps += 1

                self.assertGreater(decisions, 0)

    def test_own_hand_does_not_reveal_other_seats_hidden_tiles(self) -> None:
        env = RiichiEnv(seed=1, game_mode="4p-red-single")
        observations = env.reset()
        tracker = SeatMaterializedState(Seat.SEAT_0)
        observation = observations[0]

        policy_input = build_policy_input(tracker, observation)

        # 自席が握れるのは配牌13枚+ツモ1枚の範囲であり、他家の非公開牌14枚*3が
        # own_handへ混入していないことを枚数からも確認する。
        self.assertLessEqual(len(policy_input.own_hand.concealed_tiles), 14)


if __name__ == "__main__":
    unittest.main()
