"""Issue #209 strict source readのgame単位process並列化のtest。

`read_outcome_source(path, workers=N)`が`workers`によらず同一の検証済み
source・同一のerrorを返し、cross-game検証を親processに残し、失敗時に
workerを残さないことを固定する。
"""

import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import outcome_fixtures as of

from lisjong.learning import OutcomeSourceError, read_outcome_source
from lisjong.learning import outcome_source as outcome_module
from lisjong.learning.outcome_source import (
    DECISION_PAYLOAD_FILENAME,
    KYOKU_PAYLOAD_FILENAME,
)

WORKERS = (1, 2, 4)

GAMES = (("TRAIN", 1001), ("TRAIN", 1002), ("TRAIN", 1003), ("SELECT", 2001))
"""4 hanchan。workers=4で全gameが別workerへ渡りうる。"""


def _selected_identity_indices(source):
    """decisionごとの「selected_actionと`is`で一致するlegal action index」列。"""
    return [
        tuple(
            index
            for index, action in enumerate(decision.decision.legal_actions)
            if action is decision.selected_action
        )
        for _, decision in source.decisions()
    ]


def _read_error(path, workers):
    try:
        read_outcome_source(path, workers=workers)
    except Exception as exc:  # noqa: BLE001 - errorのtype / messageを比較する
        return type(exc), str(exc)
    raise AssertionError(f"workers={workers} accepted a broken source")


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)


class EquivalenceTests(_Case):
    def test_workers_return_the_same_source_for_both_lineages(self) -> None:
        cases = {
            "riichienv-scientific": of.write_outcome_source(
                self.root / "riichienv", GAMES
            ),
            "engine-scientific": of.write_engine_outcome_source(
                self.root / "engine", GAMES
            ),
            "engine-diagnostic": of.write_engine_outcome_source(
                self.root / "diagnostic", of.DIAGNOSTIC_GAMES, role="DIAGNOSTIC"
            ),
        }
        for name, path in cases.items():
            baseline = read_outcome_source(path)
            for workers in WORKERS:
                with self.subTest(source=name, workers=workers):
                    source = read_outcome_source(path, workers=workers)
                    self.assertEqual(source, baseline)
                    self.assertEqual(source.identity, baseline.identity)
                    self.assertEqual(
                        [game.game_ordinal for game in source.games],
                        list(range(len(baseline.games))),
                    )
                    self.assertEqual(
                        [
                            (game.game_ordinal, decision.focal_decision_ordinal)
                            for game, decision in source.decisions()
                        ],
                        [
                            (game.game_ordinal, decision.focal_decision_ordinal)
                            for game, decision in baseline.decisions()
                        ],
                    )
                    self.assertEqual(
                        _selected_identity_indices(source),
                        _selected_identity_indices(baseline),
                    )

    def test_workers_above_game_count_are_accepted(self) -> None:
        path = of.write_outcome_source(self.root / "source")
        self.assertEqual(
            read_outcome_source(path, workers=16), read_outcome_source(path)
        )


class ErrorSelectionTests(_Case):
    def test_reported_error_is_the_lowest_failed_game_for_every_worker_count(
        self,
    ) -> None:
        path = of.write_outcome_source(self.root / "source", GAMES)
        # game 1: kyoku rowを壊す（digestも再計算してrow検証まで到達させる）
        of.mutate_rows(
            path, 1, KYOKU_PAYLOAD_FILENAME, lambda rows: rows[0].update(honba=99)
        )
        # game 3: payloadを書き換えるがsummary digestは古いまま
        payload = path / "game-003" / DECISION_PAYLOAD_FILENAME
        payload.write_text(payload.read_text(encoding="utf-8") + "\n", "utf-8")

        errors = {workers: _read_error(path, workers) for workers in WORKERS}

        self.assertEqual(errors[1][0], OutcomeSourceError)
        self.assertIn("game-001", errors[1][1])
        for workers in WORKERS:
            with self.subTest(workers=workers):
                self.assertEqual(errors[workers], errors[1])

    def test_single_broken_game_fails_closed_for_every_worker_count(self) -> None:
        for broken in range(len(GAMES)):
            path = of.write_outcome_source(self.root / f"broken-{broken}", GAMES)
            payload = path / f"game-{broken:03d}" / KYOKU_PAYLOAD_FILENAME
            payload.write_text(payload.read_text(encoding="utf-8") + "\n", "utf-8")
            baseline = _read_error(path, 1)
            self.assertIn(f"game-{broken:03d}", baseline[1])
            for workers in WORKERS[1:]:
                with self.subTest(broken=broken, workers=workers):
                    self.assertEqual(_read_error(path, workers), baseline)

    def test_failure_leaves_no_worker_process(self) -> None:
        path = of.write_outcome_source(self.root / "source", GAMES)
        payload = path / "game-000" / DECISION_PAYLOAD_FILENAME
        payload.write_text(payload.read_text(encoding="utf-8") + "\n", "utf-8")
        before = {process.pid for process in multiprocessing.active_children()}

        with self.assertRaises(OutcomeSourceError):
            read_outcome_source(path, workers=4)

        after = {process.pid for process in multiprocessing.active_children()}
        self.assertEqual(after - before, set())


class ParentValidationTests(_Case):
    def test_cross_game_validation_runs_in_the_parent_before_any_worker(
        self,
    ) -> None:
        cases = {
            "seed-reuse": lambda path: of.mutate_game_summary(
                path, 1, lambda summary: summary.update(seed=1001)
            ),
            "extra-directory": lambda path: (path / "game-099").mkdir(),
            "allocation-binding": lambda path: of.mutate_manifest(
                path, lambda body: body["allocation_bindings"].pop("SELECT")
            ),
        }
        for name, mutate in cases.items():
            path = of.write_outcome_source(self.root / name, GAMES)
            mutate(path)
            with self.subTest(case=name):
                expected = _read_error(path, 1)
                with patch.object(
                    outcome_module,
                    "ProcessPoolExecutor",
                    side_effect=AssertionError("worker pool must not start"),
                ):
                    self.assertEqual(_read_error(path, 4), expected)
                self.assertIs(expected[0], OutcomeSourceError)

    def test_workers_one_does_not_start_a_worker_pool(self) -> None:
        path = of.write_outcome_source(self.root / "source", GAMES)
        with patch.object(
            outcome_module,
            "ProcessPoolExecutor",
            side_effect=AssertionError("worker pool must not start"),
        ):
            read_outcome_source(path)
            read_outcome_source(path, workers=1)

    def test_invalid_workers_are_rejected_before_reading(self) -> None:
        missing = self.root / "missing"
        for workers in (0, -1, True, 2.0, "2", None):
            with self.subTest(workers=workers):
                with self.assertRaises(ValueError):
                    read_outcome_source(missing, workers=workers)


if __name__ == "__main__":
    unittest.main()
