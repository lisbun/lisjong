"""Issue #209 qualification replayのgame単位process並列化のtest。

`evaluate_semantic_envelope_policy_by_game(..., workers=N)`が、runtime以外は
`workers`によらず既存の`evaluate_semantic_envelope_policy()`と同一の結果を返し、
global decision indexでstride / residual sampleを決め、Policyをworker内で
生成し、失敗時にfail closedすることを固定する。
"""

import importlib.util
import multiprocessing
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import candidate_fixtures as cf
import outcome_fixtures as of

from lisjong.learning import (
    ConstantResidualRuntime,
    SemanticEnvelopeOffensePolicy,
    evaluate_semantic_envelope_policy,
    evaluate_semantic_envelope_policy_by_game,
    read_outcome_source,
    read_source_record,
)
from lisjong.learning import envelope_diagnostics as diagnostics_module
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy

HAS_TORCH = importlib.util.find_spec("torch") is not None

WORKERS = (1, 2, 4)

GAMES = (("TRAIN", 1001), ("TRAIN", 1002), ("TRAIN", 1003), ("SELECT", 2001))


@dataclass(frozen=True, slots=True)
class _Decision:
    """test用のread-only adapter（#206と同じ4 fieldだけを写す）。"""

    split: str
    policy_input: object
    legal_actions: tuple
    selected_action: object


class _LastSurvivorScorer:
    """candidate位置が後ろほど高いscore（canonical先頭以外を選ばせる）。"""

    def score(self, context, candidates):
        return tuple(float(index) for index in range(len(candidates)))


def _last_survivor_policy():
    return SemanticEnvelopeOffensePolicy(_LastSurvivorScorer())


def _worker_only_policy():
    """親processで呼ばれたらfail closedするfactory（worker内生成の確認用）。"""
    if multiprocessing.parent_process() is None:
        raise RuntimeError("policy factory was called in the parent process")
    return _last_survivor_policy()


def _failing_factory():
    raise ValueError("policy factory failed")


class _FailingOnGamePolicy:
    """指定game_ordinalのdecisionで失敗するpolicy。"""

    def __init__(self, seat) -> None:
        self.inner = _last_survivor_policy()
        self.seat = seat

    def decide(self, decision):
        if decision.input.self_seat == self.seat:
            raise ValueError("policy failed on a decision")
        return self.inner.decide(decision)


def _fail_on_seat_two():
    return _FailingOnGamePolicy(of.Seat(2))


class _ThreadCheckingPolicy:
    """decideのたびにtorch intra-op thread数が1であることを要求する。"""

    def __init__(self) -> None:
        self.inner = _last_survivor_policy()

    def decide(self, decision):
        import torch

        if torch.get_num_threads() != 1:
            raise AssertionError(f"torch threads = {torch.get_num_threads()}")
        return self.inner.decide(decision)


def _torch_loading_policy():
    """torchをloadし、あえてthread数を増やしてから返すfactory。"""
    import torch

    torch.set_num_threads(2)
    return _ThreadCheckingPolicy()


def _stable(evaluation):
    """volatileなruntime以外のevaluation。"""
    return {
        key: value for key, value in evaluation.items() if key != "runtime_by_branch"
    }


class _SequenceSource:
    """flattenしたgame列を既存APIの`source.decisions()`として渡す。"""

    def __init__(self, games) -> None:
        self.games = games

    def decisions(self):
        for decisions in self.games:
            yield from decisions


def _outcome_games(source):
    return [
        tuple(
            _Decision(
                split=game.split,
                policy_input=item.decision.input,
                legal_actions=item.decision.legal_actions,
                selected_action=item.selected_action,
            )
            for item in game.decisions
        )
        for game in source.games
    ]


class _Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        record = read_source_record(cf.write_candidate_source_record(root / "record"))
        cls.record_games = [game.decisions for game in record.games]
        source = read_outcome_source(
            of.write_engine_outcome_source(root / "outcome", GAMES)
        )
        cls.outcome_games = _outcome_games(source)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()


class EquivalenceTests(_Case):
    def assertMatchesSequential(self, games, splits, **options) -> dict:
        reference = options.pop("reference", True)
        expected = evaluate_semantic_envelope_policy(
            _last_survivor_policy(),
            _SequenceSource(games),
            splits,
            reference=TwoStepUkeirePolicy() if reference else None,
            **options,
        )
        for workers in WORKERS:
            with self.subTest(workers=workers, splits=splits, **options):
                evaluation = evaluate_semantic_envelope_policy_by_game(
                    _last_survivor_policy,
                    games,
                    splits,
                    workers=workers,
                    reference_factory=TwoStepUkeirePolicy if reference else None,
                    **options,
                )
                self.assertEqual(_stable(evaluation), _stable(expected))
                self.assertEqual(
                    set(evaluation["runtime_by_branch"]),
                    set(expected["runtime_by_branch"]),
                )
        return expected

    def test_matches_the_sequential_replay_for_every_worker_count(self) -> None:
        for games, splits in (
            (self.record_games, ["TRAIN", "SELECT"]),
            (self.outcome_games, ["TRAIN", "SELECT"]),
            (self.outcome_games, ["TRAIN"]),
        ):
            expected = self.assertMatchesSequential(games, splits)
            self.assertGreater(expected["support"]["decisions"], 0)
        self.assertMatchesSequential(
            self.outcome_games, ["TRAIN", "SELECT"], reference=False
        )

    def test_sample_every_uses_the_global_decision_index(self) -> None:
        counts = [
            sum(1 for decision in game if decision.split in {"TRAIN", "SELECT"})
            for game in self.outcome_games
        ]
        for sample_every in (2, 3, 4, 5, 7):
            per_game_restart = sum(
                (count + sample_every - 1) // sample_every for count in counts
            )
            global_stride = (sum(counts) + sample_every - 1) // sample_every
            if sample_every in (4, 5):
                # gameごとに数え直すとsupportまで変わる構成であることを確認する
                self.assertNotEqual(per_game_restart, global_stride)
            expected = self.assertMatchesSequential(
                self.outcome_games, ["TRAIN", "SELECT"], sample_every=sample_every
            )
            self.assertEqual(expected["support"]["decisions"], global_stride)

    def test_residual_samples_are_merged_in_global_decision_order(self) -> None:
        full = self.assertMatchesSequential(
            self.outcome_games, ["TRAIN", "SELECT"], residual_sample_limit=1000
        )
        samples = full["residual_choice"]["samples"]
        self.assertGreater(len(samples), 3)
        for limit in (1, 2, 3):
            limited = self.assertMatchesSequential(
                self.outcome_games, ["TRAIN", "SELECT"], residual_sample_limit=limit
            )
            self.assertEqual(limited["residual_choice"]["samples"], samples[:limit])

    def test_no_selected_decision_fails_like_the_sequential_replay(self) -> None:
        with self.assertRaises(ValueError) as sequential:
            evaluate_semantic_envelope_policy(
                _last_survivor_policy(), _SequenceSource(self.outcome_games), ["X"]
            )
        for workers in WORKERS:
            with self.subTest(workers=workers):
                with self.assertRaises(ValueError) as parallel:
                    evaluate_semantic_envelope_policy_by_game(
                        _last_survivor_policy,
                        self.outcome_games,
                        ["X"],
                        workers=workers,
                    )
                self.assertEqual(str(parallel.exception), str(sequential.exception))


class WorkerBoundaryTests(_Case):
    def test_workers_create_policies_inside_the_worker(self) -> None:
        evaluation = evaluate_semantic_envelope_policy_by_game(
            _worker_only_policy,
            self.outcome_games,
            ["TRAIN", "SELECT"],
            workers=2,
            reference_factory=TwoStepUkeirePolicy,
        )
        self.assertEqual(evaluation["invariants"]["legality"], 1.0)
        with self.assertRaises(RuntimeError):
            evaluate_semantic_envelope_policy_by_game(
                _worker_only_policy, self.outcome_games, ["TRAIN"], workers=1
            )

    def test_workers_one_does_not_start_a_worker_pool(self) -> None:
        with patch.object(
            diagnostics_module,
            "ProcessPoolExecutor",
            side_effect=AssertionError("worker pool must not start"),
        ):
            evaluate_semantic_envelope_policy_by_game(
                ConstantResidualRuntime(), self.outcome_games, ["TRAIN"]
            )

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_worker_torch_threads_are_pinned_to_one(self) -> None:
        import torch

        previous = torch.get_num_threads()
        try:
            for workers in (1, 2):
                with self.subTest(workers=workers):
                    evaluation = evaluate_semantic_envelope_policy_by_game(
                        _torch_loading_policy,
                        self.outcome_games,
                        ["TRAIN", "SELECT"],
                        workers=workers,
                    )
                    self.assertEqual(evaluation["invariants"]["legality"], 1.0)
        finally:
            torch.set_num_threads(previous)

    def test_invalid_arguments_are_rejected_before_replay(self) -> None:
        for workers in (0, -1, True, 2.0, "2", None):
            with self.subTest(workers=workers):
                with self.assertRaises(ValueError):
                    evaluate_semantic_envelope_policy_by_game(
                        _failing_factory, self.outcome_games, ["TRAIN"], workers=workers
                    )
        for splits, sample_every in (([], 1), (["TRAIN"], 0)):
            with self.subTest(splits=splits, sample_every=sample_every):
                with self.assertRaises(ValueError):
                    evaluate_semantic_envelope_policy_by_game(
                        _failing_factory,
                        self.outcome_games,
                        splits,
                        workers=2,
                        sample_every=sample_every,
                    )


class FailClosedTests(_Case):
    def test_factory_failure_propagates_for_every_worker_count(self) -> None:
        for workers in WORKERS:
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "policy factory failed"):
                    evaluate_semantic_envelope_policy_by_game(
                        _failing_factory, self.outcome_games, ["TRAIN"], workers=workers
                    )

    def test_decision_failure_returns_no_partial_evaluation(self) -> None:
        before = {process.pid for process in multiprocessing.active_children()}
        for workers in WORKERS:
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "policy failed on a decision"):
                    evaluate_semantic_envelope_policy_by_game(
                        _fail_on_seat_two,
                        self.outcome_games,
                        ["TRAIN", "SELECT"],
                        workers=workers,
                    )
        after = {process.pid for process in multiprocessing.active_children()}
        self.assertEqual(after - before, set())


if __name__ == "__main__":
    unittest.main()
