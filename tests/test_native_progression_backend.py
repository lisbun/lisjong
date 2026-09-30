"""Issue #232: R5 progression探索のnative化とPython oracleの同等性を固定する。

- factory選択testはnative拡張の有無に依存せず常に実行する。
- native同値性testは`_lisjong_native`がimportできる環境だけで実行し、
  `LISJONG_REQUIRE_NATIVE=1`（native CI job）ではskipせずfailさせる。
- Python oracle（`_TerminalShantenProgressionEvaluator`）は独立した型として
  残っており、ここでは明示的に生成して比較する。本番factoryの選択は変えない。
"""

import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import lisjong.policies.mechanism_riichi_defense_offensive_efficiency_diagnostic as diagnostic
import lisjong.policies.targeted_honor_release_terminal_progression as targeted
import lisjong.policies.terminal_shanten_progression_mechanism_riichi_defense as progression
from lisjong.hand_evaluation import _shanten_backend
from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    _evaluate_completion_masses,
    _falling_factorial,
    _FiniteHorizonEvaluator,
    _root_remaining_counts,
)
from lisjong.policy_contract.own_hand_state import OwnHandState
from tests.test_mechanism_riichi_defense_offensive_efficiency_diagnostic import (
    _all_zero_progression_fixture,
    _all_zero_progression_with_excluded_better_fixture,
)
from tests.test_native_shanten_backend import (
    _lisjong_native,
    _require_native,
    _run_python,
)
from tests.test_terminal_shanten_progression_mechanism_riichi_defense_policy import (
    _CLOSED_FAR_HAND,
    _FOUR_MELD_HAND,
    _THREE_MELD_HAND,
    _TWO_MELD_HAND,
    _counts,
    _distinct_discard_actions,
    _reference_distribution,
    _restricted_input,
)
from tests.test_terminal_shanten_progression_mechanism_riichi_defense_policy import (
    _hand as _progression_hand,
)

_POLICY_ERROR = progression.TerminalShantenProgressionPolicyError
_COUNTER_NAMES = ("visited_states", "cache_hits", "cache_misses", "shanten_evaluations")


def _native_evaluator() -> progression._NativeTerminalShantenProgressionEvaluator:
    """backend選択と独立にnative evaluatorを明示生成する（test専用）。"""
    table = _shanten_backend.build_native_table(_lisjong_native)
    return progression._NativeTerminalShantenProgressionEvaluator(
        table.evaluate_progression
    )


def _native_table():
    return _shanten_backend.build_native_table(_lisjong_native)


def _counters(evaluator) -> tuple[int, ...]:
    return tuple(getattr(evaluator, name) for name in _COUNTER_NAMES)


class FactorySelectionTest(unittest.TestCase):
    """本番factoryはprocess単位のbackend選択に従う（native拡張なしでも実行）。"""

    def test_factory_follows_the_selected_backend(self) -> None:
        evaluator = progression._new_progression_evaluator()
        if _shanten_backend.BACKEND_NAME == _shanten_backend.RUST_BACKEND:
            self.assertIsInstance(
                evaluator, progression._NativeTerminalShantenProgressionEvaluator
            )
        else:
            self.assertIs(
                type(evaluator), progression._TerminalShantenProgressionEvaluator
            )

    def test_factory_returns_a_fresh_evaluator_every_call(self) -> None:
        first = progression._new_progression_evaluator()
        second = progression._new_progression_evaluator()
        self.assertIsNot(first, second)

    def test_python_oracle_and_native_evaluator_are_independent_types(self) -> None:
        self.assertFalse(
            issubclass(
                progression._NativeTerminalShantenProgressionEvaluator,
                progression._TerminalShantenProgressionEvaluator,
            )
        )

    def test_all_three_production_callers_use_the_shared_factory(self) -> None:
        self.assertIs(
            targeted._new_progression_evaluator,
            progression._new_progression_evaluator,
        )
        self.assertIs(
            diagnostic._new_progression_evaluator,
            progression._new_progression_evaluator,
        )

    def test_python_selection_never_exposes_a_native_entry(self) -> None:
        if _shanten_backend.BACKEND_NAME == _shanten_backend.PYTHON_BACKEND:
            self.assertIsNone(_shanten_backend.native_evaluate_progression)
        else:
            self.assertIsNotNone(_shanten_backend.native_evaluate_progression)


class NativeProgressionEquivalenceTest(unittest.TestCase):
    """nativeがPython oracleとdistribution・root shanten・counterまで一致する。"""

    def setUp(self) -> None:
        _require_native(self)

    def _assert_same_as_oracle(self, roots, remaining, horizon) -> None:
        oracle = progression._TerminalShantenProgressionEvaluator()
        native = _native_evaluator()
        expected = oracle.evaluate_roots(roots, remaining, horizon)
        actual = native.evaluate_roots(roots, remaining, horizon)
        self.assertEqual(actual, expected)
        # 子の評価順・cache共有範囲が同じなので、counterも完全に一致する。
        self.assertEqual(_counters(native), _counters(oracle))

    def test_horizons_one_to_three_match_the_oracle(self) -> None:
        remaining = _counts(m2=3, m6=2, s9=2, z3=1)
        roots = (_THREE_MELD_HAND, _TWO_MELD_HAND, _FOUR_MELD_HAND, _CLOSED_FAR_HAND)
        for horizon in (1, 2, 3):
            with self.subTest(horizon=horizon):
                self._assert_same_as_oracle(roots, remaining, horizon)

    def test_matches_the_unpruned_reference_for_small_remaining(self) -> None:
        for horizon, remaining, hands in (
            (1, _counts(m2=3, m5=2, p4=4, z3=1), (_THREE_MELD_HAND, _CLOSED_FAR_HAND)),
            (2, _counts(m2=3, m6=2, s9=2, z3=1), (_THREE_MELD_HAND, _TWO_MELD_HAND)),
            (3, _counts(m2=3, m5=2, s9=2, z3=1), (_FOUR_MELD_HAND, _THREE_MELD_HAND)),
            (3, _counts(m2=3, m6=2), (_CLOSED_FAR_HAND,)),
        ):
            results = _native_evaluator().evaluate_roots(hands, remaining, horizon)
            for hand, (root_shanten, distribution) in zip(hands, results, strict=True):
                with self.subTest(horizon=horizon, hand=hand):
                    self.assertEqual(
                        distribution, _reference_distribution(hand, remaining, horizon)
                    )
                    self.assertEqual(
                        root_shanten,
                        progression._TerminalShantenProgressionEvaluator().shanten(
                            hand
                        ),
                    )
                    self.assertEqual(
                        sum(distribution),
                        _falling_factorial(sum(remaining), horizon),
                    )

    def test_input_root_order_and_duplicates_are_preserved(self) -> None:
        remaining = _counts(m2=3, m6=2, s9=2, z3=1)
        roots = (_TWO_MELD_HAND, _THREE_MELD_HAND, _TWO_MELD_HAND)
        self._assert_same_as_oracle(roots, remaining, 2)
        reversed_roots = tuple(reversed(roots))
        forward = _native_evaluator().evaluate_roots(roots, remaining, 2)
        backward = _native_evaluator().evaluate_roots(reversed_roots, remaining, 2)
        self.assertEqual(forward, tuple(reversed(backward)))

    def test_mass_equal_children_keep_the_same_diagnostic_distribution(self) -> None:
        # canonical順の先頭採用が変わると、massが同じでもdistributionが変わる。
        remaining = _counts(m2=3, m6=2, s9=2, z3=1, p5=2)
        roots = (_THREE_MELD_HAND, _TWO_MELD_HAND)
        self._assert_same_as_oracle(roots, remaining, 3)

    def test_counters_are_measured_not_zero_filled(self) -> None:
        native = _native_evaluator()
        self.assertTrue(all(value is None for value in _counters(native)))
        native.evaluate_roots((_THREE_MELD_HAND,), _counts(m2=3, m6=2, s9=2), 2)
        visited, hits, misses, shanten_evaluations = _counters(native)
        self.assertGreater(visited, 0)
        self.assertGreater(misses, 0)
        self.assertGreater(shanten_evaluations, 0)
        self.assertGreaterEqual(hits, 0)

    def test_one_evaluator_evaluates_one_batch_so_cache_never_carries_over(
        self,
    ) -> None:
        native = _native_evaluator()
        native.evaluate_roots((_THREE_MELD_HAND,), _counts(m2=3, m6=2, s9=2), 2)
        with self.assertRaises(RuntimeError):
            native.evaluate_roots((_THREE_MELD_HAND,), _counts(m2=3, m6=2, s9=2), 2)

    def test_repeated_calls_share_no_state(self) -> None:
        remaining = _counts(m2=3, m6=2, s9=2, z3=1)
        first = _native_evaluator()
        second = _native_evaluator()
        first_result = first.evaluate_roots((_THREE_MELD_HAND,), remaining, 3)
        second_result = second.evaluate_roots((_THREE_MELD_HAND,), remaining, 3)
        self.assertEqual(first_result, second_result)
        self.assertEqual(_counters(first), _counters(second))

    def test_entry_counts_successful_calls_only(self) -> None:
        before = _lisjong_native.progression_evaluation_call_count()
        _native_evaluator().evaluate_roots(
            (_THREE_MELD_HAND,), _counts(m2=3, m6=2, s9=2), 1
        )
        self.assertEqual(
            _lisjong_native.progression_evaluation_call_count(), before + 1
        )
        with self.assertRaises(ValueError):
            _native_evaluator().evaluate_roots(
                (_THREE_MELD_HAND,), _counts(m2=3, m6=2, s9=2), 0
            )
        self.assertEqual(
            _lisjong_native.progression_evaluation_call_count(), before + 1
        )


class NativeProgressionInputBoundaryTest(unittest.TestCase):
    """入力境界は変換・探索より前にTypeError / ValueErrorで拒否される。"""

    def setUp(self) -> None:
        _require_native(self)
        self.table = _native_table()
        self.remaining = _counts(m2=3, m6=2, s9=2, z3=1)

    def _call(self, roots, remaining, horizon):
        return self.table.evaluate_progression(roots, remaining, horizon, _POLICY_ERROR)

    def test_horizon_must_be_an_int_from_one_to_three(self) -> None:
        for horizon in (0, 4, -1, 10**30):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                self._call((_THREE_MELD_HAND,), self.remaining, horizon)
        for horizon in (True, False, 3.0, "3", None):
            with self.subTest(horizon=horizon), self.assertRaises(TypeError):
                self._call((_THREE_MELD_HAND,), self.remaining, horizon)

    def test_remaining_total_must_cover_the_horizon(self) -> None:
        with self.assertRaises(ValueError):
            self._call((_THREE_MELD_HAND,), _counts(m2=2), 3)
        self._call((_THREE_MELD_HAND,), _counts(m2=3), 3)

    def test_count_arrays_are_validated(self) -> None:
        bad_remaining = list(self.remaining)
        for bad in (
            tuple(self.remaining[:33]),
            tuple(self.remaining) + (0,),
            (*self.remaining[:-1], 5),
            (*self.remaining[:-1], -1),
        ):
            with self.subTest(remaining=bad), self.assertRaises(ValueError):
                self._call((_THREE_MELD_HAND,), bad, 2)
        with self.assertRaises(OverflowError):
            self._call((_THREE_MELD_HAND,), (*bad_remaining[:-1], 2**70), 2)
        for bad_hand in (
            _THREE_MELD_HAND[:33],
            (*_THREE_MELD_HAND[:-1], 5),
            (*_THREE_MELD_HAND[:-1], -1),
        ):
            with self.subTest(hand=bad_hand), self.assertRaises(ValueError):
                self._call((bad_hand,), self.remaining, 2)
        with self.assertRaises(TypeError):
            self._call((_THREE_MELD_HAND,), None, 2)
        with self.assertRaises(TypeError):
            self._call(5, self.remaining, 2)

    def test_root_hand_size_must_be_a_valid_concealed_size(self) -> None:
        twelve = [0] * 34
        twelve[0] = twelve[1] = twelve[2] = twelve[3] = 3
        with self.assertRaises(ValueError):
            self._call((tuple(twelve),), self.remaining, 2)
        with self.assertRaises(ValueError):
            self._call((tuple([0] * 34),), self.remaining, 2)

    def test_more_than_four_copies_reached_by_a_draw_is_rejected(self) -> None:
        hand = [0] * 34
        hand[0] = 4
        hand[13] = hand[22] = hand[27] = 1
        remaining = [0] * 34
        remaining[0] = 3  # inconsistent: hand 4 + remaining 3 of the same tile
        remaining[18] = 3
        with self.assertRaises(ValueError):
            self._call((tuple(hand),), tuple(remaining), 2)

    def test_empty_root_batch_returns_empty_results(self) -> None:
        roots, counters = self._call((), self.remaining, 2)
        self.assertEqual((roots, counters), ((), (0, 0, 0, 0)))

    def test_inputs_are_not_modified(self) -> None:
        roots = [list(_THREE_MELD_HAND)]
        remaining = list(self.remaining)
        self._call(roots, remaining, 2)
        self.assertEqual(roots, [list(_THREE_MELD_HAND)])
        self.assertEqual(remaining, list(self.remaining))


class NativeProgressionReleaseBuildAbortTest(unittest.TestCase):
    """異常入力でもprocessがabortせず、Python例外になる（release wheel前提）。

    `native/Cargo.toml`のreleaseは`panic = "abort"`で、overflow検査も無効である。
    入力由来の違反がabort・wrap・truncateにならないことをsubprocessで確認する。
    """

    def setUp(self) -> None:
        _require_native(self)

    def test_abnormal_inputs_raise_instead_of_aborting(self) -> None:
        result = _run_python(
            "import _lisjong_native\n"
            "from lisjong.hand_evaluation import _shanten_backend\n"
            "table = _shanten_backend.build_native_table(_lisjong_native)\n"
            "class PolicyError(Exception): pass\n"
            "good = [0] * 34\n"
            "good[0] = 3\n"
            "good[9] = 1\n"
            "remaining = [1] * 34\n"
            "cases = [\n"
            "    ((good,), remaining, 0),\n"
            "    ((good,), remaining, 4),\n"
            "    ((good,), remaining, True),\n"
            "    (([5] * 34,), remaining, 2),\n"
            "    (([255] * 34,), remaining, 2),\n"
            "    (([2 ** 64] * 34,), remaining, 2),\n"
            "    (([-1] * 34,), remaining, 2),\n"
            "    ((good,), [0] * 34, 1),\n"
            "    ((good[:33],), remaining, 2),\n"
            "    (([4] * 34,), remaining, 2),\n"
            "]\n"
            "for roots, rem, horizon in cases:\n"
            "    try:\n"
            "        table.evaluate_progression(roots, rem, horizon, PolicyError)\n"
            "    except (TypeError, ValueError, OverflowError, PolicyError):\n"
            "        continue\n"
            "    raise SystemExit('accepted abnormal input: %r' % ((horizon,),))\n"
            "print('ok')\n",
            backend="rust",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


class NativeProgressionGilTest(unittest.TestCase):
    """native探索中、別Python threadが進行できる（GILを解放している）。"""

    def setUp(self) -> None:
        _require_native(self)

    def test_another_thread_runs_during_a_single_native_search(self) -> None:
        # 閉じた遠い手 + 全牌種が残る盤面: 単独のnative探索が長く続く入力。
        remaining = tuple(4 - count for count in _CLOSED_FAR_HAND)
        roots = (_CLOSED_FAR_HAND,)
        table = _native_table()

        ticks: list[float] = []
        stop = threading.Event()
        started = threading.Event()

        def worker() -> None:
            started.set()
            while not stop.is_set():
                ticks.append(time.perf_counter())

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            started.wait()
            begin = time.perf_counter()
            table.evaluate_progression(roots, remaining, 3, _POLICY_ERROR)
            end = time.perf_counter()
        finally:
            stop.set()
            thread.join()

        duration = end - begin
        if duration < 0.1:
            self.skipTest(
                f"native search finished in {duration:.3f}s, too short to observe "
                "thread progress"
            )
        margin = duration / 4
        inside = [tick for tick in ticks if begin + margin < tick < end - margin]
        self.assertGreater(
            len(inside), 0, "no other thread progress during the native search"
        )


class ThreeCallersEquivalenceTest(unittest.TestCase):
    """R5の3呼出し元すべてで、Python oracleとnativeの結果が一致する。"""

    # R5発動が確認済みのclosed 14枚手（手牌、drawable牌種、期待activation stage）。
    _TARGETED_CASES = (
        (
            "9m5m8p4m7p7s6p3z3s4p7m4z9s1m",
            ("7s", "1z", "1m", "2z"),
            "R5_PARENT_BEST",
        ),
        (
            "1p3m2p6p5s8p4s5p9p8m3s3z2z4z",
            ("6p", "7z", "2m", "8s"),
            "R5_NON_HONOR_ONLY_BEST",
        ),
        (
            "8p5m6m1z3m8s4z7m3s5z7z5p1m1s",
            ("6s", "5z", "1s", "6p"),
            "R5_HONOR_ONLY_SWITCH",
        ),
    )

    def setUp(self) -> None:
        _require_native(self)
        table = _native_table()
        self._oracle_factory = progression._TerminalShantenProgressionEvaluator
        self._native_factory = lambda: (
            progression._NativeTerminalShantenProgressionEvaluator(
                table.evaluate_progression
            )
        )

    def _run_with(self, module, factory, call):
        with patch.object(module, "_new_progression_evaluator", factory):
            return call()

    def _drawn_input(self, notation: str, drawable):
        tiles = tuple(_progression_hand(notation))
        policy_input = _restricted_input(tiles, drawable)
        policy_input = replace(
            policy_input,
            own_hand=OwnHandState(concealed_tiles=tiles, drawn_tile=tiles[-1]),
        )
        return policy_input, _distinct_discard_actions(tiles)

    def test_progression_policy_matches_for_all_zero_decisions(self) -> None:
        concealed = _progression_hand("147m258p369s13577z")
        actions = _distinct_discard_actions(concealed)
        for drawable in (
            tuple(f"{rank}m" for rank in range(1, 6)),
            tuple(f"{rank}m" for rank in range(1, 10)),
        ):
            policy_input = _restricted_input(concealed, drawable)
            expected = self._run_with(
                progression,
                self._oracle_factory,
                lambda: progression._evaluate_and_choose_discard(policy_input, actions),
            )
            actual = self._run_with(
                progression,
                self._native_factory,
                lambda: progression._evaluate_and_choose_discard(policy_input, actions),
            )
            self.assertTrue(expected[1].progression_activated)
            self.assertEqual(actual, expected)

    def test_targeted_gate_matches_action_stage_and_progression_values(self) -> None:
        seen_stages = set()
        for notation, drawable, stage in self._TARGETED_CASES:
            with self.subTest(stage=stage):
                policy_input, actions = self._drawn_input(notation, drawable)
                expected = self._run_with(
                    targeted,
                    self._oracle_factory,
                    lambda: targeted._evaluate_and_choose_discard(
                        policy_input, actions
                    ),
                )
                actual = self._run_with(
                    targeted,
                    self._native_factory,
                    lambda: targeted._evaluate_and_choose_discard(
                        policy_input, actions
                    ),
                )
                self.assertEqual(expected[1].activation_stage.name, stage)
                seen_stages.add(expected[1].activation_stage.name)
                self.assertEqual(actual, expected)
        self.assertEqual(len(seen_stages), len(self._TARGETED_CASES))

    def test_diagnostic_matches_both_candidate_sets(self) -> None:
        for fixture in (
            _all_zero_progression_fixture,
            _all_zero_progression_with_excluded_better_fixture,
        ):
            policy_input, actions = fixture()

            def analyze():
                return diagnostic.analyze_mechanism_riichi_defense_offensive_efficiency(
                    policy_input, actions, include_terminal_progression=True
                )

            expected = self._run_with(diagnostic, self._oracle_factory, analyze)
            actual = self._run_with(diagnostic, self._native_factory, analyze)
            self.assertIsNotNone(expected.full_legal_terminal_progression_summary)
            self.assertIsNotNone(
                expected.baseline_eligible_terminal_progression_summary
            )
            self.assertEqual(actual, expected)

    def test_candidate_helper_matches_with_both_evaluators(self) -> None:
        concealed = _progression_hand("147m258p369s13577z")
        actions = _distinct_discard_actions(concealed)
        policy_input = _restricted_input(
            concealed, tuple(f"{rank}m" for rank in range(1, 6))
        )
        remaining = _root_remaining_counts(policy_input)
        completions = _evaluate_completion_masses(
            policy_input, actions, remaining, DEFAULT_HORIZON, _FiniteHorizonEvaluator()
        )
        expected = progression._evaluate_progression_candidates(
            policy_input,
            completions,
            remaining,
            DEFAULT_HORIZON,
            self._oracle_factory(),
        )
        actual = progression._evaluate_progression_candidates(
            policy_input,
            completions,
            remaining,
            DEFAULT_HORIZON,
            self._native_factory(),
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            tuple(candidate.action for candidate in actual),
            tuple(evaluation.action for evaluation in completions),
        )


class RustSelectedProductionWiringTest(unittest.TestCase):
    """rust指定のprocessでは3呼出し元がnative R5を実行する（subprocess）。"""

    def setUp(self) -> None:
        _require_native(self)

    def _run(self, body: str):
        return _run_python(
            "import _lisjong_native\n"
            "from dataclasses import replace\n"
            "from lisjong.hand_evaluation import _shanten_backend\n"
            "import lisjong.policies.terminal_shanten_progression_mechanism_riichi_defense as progression\n"
            "import lisjong.policies.targeted_honor_release_terminal_progression as targeted\n"
            "import lisjong.policies.mechanism_riichi_defense_offensive_efficiency_diagnostic as diagnostic\n"
            "from lisjong.policy_contract.own_hand_state import OwnHandState\n"
            "from tests.test_native_progression_backend import ThreeCallersEquivalenceTest\n"
            "from tests.test_mechanism_riichi_defense_offensive_efficiency_diagnostic import _all_zero_progression_fixture\n"
            "from tests.test_terminal_shanten_progression_mechanism_riichi_defense_policy import _hand, _distinct_discard_actions, _restricted_input\n"
            "assert _shanten_backend.BACKEND_NAME == 'rust'\n"
            "def calls():\n"
            "    return _lisjong_native.progression_evaluation_call_count()\n"
            f"{body}",
            backend="rust",
        )

    def test_progression_policy_runs_the_native_search(self) -> None:
        result = self._run(
            "concealed = _hand('147m258p369s13577z')\n"
            "actions = _distinct_discard_actions(concealed)\n"
            "policy_input = _restricted_input(concealed, tuple(f'{r}m' for r in range(1, 6)))\n"
            "before = calls()\n"
            "_, analysis = progression._evaluate_and_choose_discard(policy_input, actions)\n"
            "assert analysis.progression_activated\n"
            "assert calls() == before + 1, (before, calls())\n"
            "print('ok')\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_targeted_gate_runs_the_native_search(self) -> None:
        notation, drawable, _ = ThreeCallersEquivalenceTest._TARGETED_CASES[2]
        result = self._run(
            f"tiles = tuple(_hand({notation!r}))\n"
            f"policy_input = _restricted_input(tiles, {drawable!r})\n"
            "policy_input = replace(policy_input, own_hand=OwnHandState(concealed_tiles=tiles, drawn_tile=tiles[-1]))\n"
            "actions = _distinct_discard_actions(tiles)\n"
            "before = calls()\n"
            "_, analysis = targeted._evaluate_and_choose_discard(policy_input, actions)\n"
            "assert analysis.activation_stage.name == 'R5_HONOR_ONLY_SWITCH'\n"
            "assert calls() == before + 1, (before, calls())\n"
            "print('ok')\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_diagnostic_runs_the_native_search_once_per_candidate_set(self) -> None:
        result = self._run(
            "policy_input, actions = _all_zero_progression_fixture()\n"
            "before = calls()\n"
            "analysis = diagnostic.analyze_mechanism_riichi_defense_offensive_efficiency("
            "policy_input, actions, include_terminal_progression=True)\n"
            "assert analysis.full_legal_terminal_progression_summary is not None\n"
            "assert calls() == before + 1, (before, calls())\n"
            "print('ok')\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
