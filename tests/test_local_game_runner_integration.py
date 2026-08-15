import unittest

from lisjong.local_game_runner import LocalGameRunner
from lisjong.policies import MinimalPolicy, ShantenPolicy, UkeirePolicy
from lisjong.policy_contract import DecisionContext, InternalAction, Seat


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


class _RecordingMinimalPolicy:
    def __init__(self) -> None:
        self.decisions: list[DecisionContext] = []
        self._delegate = MinimalPolicy()

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        if not isinstance(decision, DecisionContext):
            raise AssertionError("Policy must receive only DecisionContext")
        if _module_is_leaked_from_riichienv(decision, set()):
            raise AssertionError("RiichiEnv object leaked into DecisionContext")
        self.decisions.append(decision)
        return self._delegate.choose_action(decision)


class LocalGameRunnerIntegrationTest(unittest.TestCase):
    def test_fixed_seed_half_game_completes_and_is_reproducible(self) -> None:
        seed = 12345
        policies = {seat: _RecordingMinimalPolicy() for seat in Seat}
        first_runner = LocalGameRunner(
            policies,
            seed=seed,
            game_mode="4p-red-half",
            max_steps=10_000,
        )

        first = first_runner.run()
        second = LocalGameRunner(
            {seat: MinimalPolicy() for seat in Seat},
            seed=seed,
            game_mode="4p-red-half",
            max_steps=10_000,
        ).run()

        self.assertTrue(first_runner._env.done())
        self.assertEqual(first, second)
        self.assertEqual(first.seed, seed)
        self.assertEqual(first.game_mode, "4p-red-half")
        self.assertEqual(first.scores, (24000, 34400, 24000, 17600))
        self.assertEqual(first.ranks, (2, 1, 3, 4))
        self.assertGreater(first.steps, 1)
        self.assertGreater(first.decisions, first.steps)

        recorded = [
            decision for policy in policies.values() for decision in policy.decisions
        ]
        self.assertEqual(len(recorded), first.decisions)
        self.assertTrue(all(policy.decisions for policy in policies.values()))

        kyoku = {
            (
                decision.input.round.round_wind,
                decision.input.round.hand_number,
                decision.input.round.honba,
            )
            for decision in recorded
        }
        self.assertGreater(len(kyoku), 1)


class LocalGameRunnerShantenPolicyIntegrationTest(unittest.TestCase):
    """Issue #51完了条件: `ShantenPolicy`でRiichiEnv固定seed対局を完走できる。

    ここではPolicyの強さや得点は評価しない。`ShantenPolicy` /
    Policy実行境界 / RiichiEnv Adapter / Local game runnerの一連の
    integrationが、既存のLocal game runner testと同じ構造・同じ固定seedで
    正常完走することだけを確認する。
    """

    def test_fixed_seed_half_game_completes_with_shanten_policy(self) -> None:
        seed = 12345
        runner = LocalGameRunner(
            {seat: ShantenPolicy() for seat in Seat},
            seed=seed,
            game_mode="4p-red-half",
            max_steps=10_000,
        )

        result = runner.run()

        self.assertTrue(runner._env.done())
        self.assertEqual(result.seed, seed)
        self.assertEqual(result.game_mode, "4p-red-half")
        self.assertGreater(result.steps, 1)
        self.assertGreater(result.decisions, result.steps)


class LocalGameRunnerUkeirePolicyIntegrationTest(unittest.TestCase):
    """Issue #52完了条件: `UkeirePolicy`でRiichiEnv固定seed対局を完走できる。

    #51の`ShantenPolicy` integration testと同じ構造・同じ固定seed・同じ
    game modeを再利用する。ここでもPolicyの強さやscore・順位は評価せず、
    `UkeirePolicy` / Policy実行境界 / RiichiEnv Adapter / Local game runnerの
    組み合わせが最後まで処理できることだけを確認する。受け入れ計算は
    discard候補ごとに最大34基礎牌種の`calculate_shanten()`を伴うため、
    この1局だけで既存のPolicy integration testより実行時間が長い。同じseedの
    2局目を追加して再現性まで見ると実行時間が倍になるので、決定性は
    `tests/test_ukeire_policy.py`のorder independence testで固定し、ここへは
    重ねない。環境差でflakyになる厳密なwall-clock thresholdも入れない。
    """

    def test_fixed_seed_half_game_completes_with_ukeire_policy(self) -> None:
        seed = 12345
        runner = LocalGameRunner(
            {seat: UkeirePolicy() for seat in Seat},
            seed=seed,
            game_mode="4p-red-half",
            max_steps=10_000,
        )

        result = runner.run()

        self.assertTrue(runner._env.done())
        self.assertEqual(result.seed, seed)
        self.assertEqual(result.game_mode, "4p-red-half")
        self.assertGreater(result.steps, 1)
        self.assertGreater(result.decisions, result.steps)


if __name__ == "__main__":
    unittest.main()
