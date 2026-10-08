import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_ron_legal_source as ron_fixture
from test_learning_hand_belief_source import (
    A_OWN,
    A_PLAYERS,
    make_decision,
)

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.fixed_point import SCALE, probability_to_raw
from lisjong.hand_evaluation._shanten_backend import BACKEND_ENVIRONMENT_VARIABLE
from lisjong.learning import ron_legal_accuracy as accuracy
from lisjong.learning.open_wait_estimator import OpenWaitModel
from lisjong.learning.riichi_wait_estimator import LogisticWaitModel
from lisjong.learning.ron_legal_baseline import (
    RonRateModel,
    StratumRates,
    public_stratum,
)
from lisjong.learning.ron_legal_estimator import (
    estimate_open_ron_legal_belief,
    estimate_riichi_ron_legal_belief,
)
from lisjong.policy_contract import RiichiState, Seat


def row(seed=1, episode="1", seat=1, wait=True, ron=True, public=None):
    public = public or make_decision(0, A_OWN, A_PLAYERS).policy_input
    return accuracy.Row(
        seed,
        (seed, episode, seat),
        public,
        Seat(seat),
        (wait,) + (False,) * 33,
        (ron,) + (False,) * 33,
        frozenset((0, 1)),
    )


class BaselineTest(unittest.TestCase):
    def test_raw_validation_and_public_strata(self):
        public = row().policy_input
        self.assertEqual(
            [public_stratum(public, Seat(s)) for s in (1, 2, 3)],
            ["riichi", "open", "closed_non_riichi"],
        )
        players = list(public.players)
        players[1] = replace(players[1], riichi=RiichiState.DECLARED)
        self.assertEqual(
            public_stratum(replace(public, players=tuple(players)), Seat(1)), "riichi"
        )
        for bad in ((0,) * 33, (True,) * 34, (-1,) * 34, (SCALE + 1,) * 34):
            with self.assertRaises(ValueError):
                StratumRates(bad, (0,) * 34)
        with self.assertRaises(ValueError):
            StratumRates((0,) * 34, (1,) * 34)
        with self.assertRaises(ValueError):
            RonRateModel((None,) * 2)
        with self.assertRaises(ValueError):
            RonRateModel((None,) * 3, "unknown")
        with self.assertRaises(ValueError):
            public_stratum(public, Seat(0))

    def test_public_river_mask_quantized_bounds_and_missing_train(self):
        public = row().policy_input
        rates = StratumRates((SCALE // 2,) * 34, (SCALE // 4,) * 34)
        model = RonRateModel((rates, None, rates))
        rate = model.predict(public, Seat(1), "ron_rate")
        mask = model.predict(public, Seat(1), "wait_genbutsu")
        river = {tile_type_index(d.tile.tile_type) for d in public.players[1].discards}
        self.assertTrue(river)
        self.assertEqual(rate.ron_legal_probability_raw, rates.ron_raw)
        self.assertEqual(mask.wait_probability_raw, rate.wait_probability_raw)
        for i in range(34):
            self.assertEqual(
                mask.ron_legal_probability_raw[i], 0 if i in river else SCALE // 2
            )
        self.assertIsNone(model.predict(public, Seat(2), "ron_rate"))
        with self.assertRaises(ValueError):
            model.predict(public, Seat(1), "unknown")

    def test_inference_without_truth_source_or_native_imports(self):
        code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.startswith(("lisjong.learning.ron_legal_accuracy", "lisjong.learning.ron_legal_source", "lisjong.belief.ron_legal_ground_truth", "_lisjong_native")):
            raise AssertionError("privileged import: " + name)
sys.meta_path.insert(0, Block())
from test_learning_hand_belief_source import A_OWN,A_PLAYERS,make_decision
from lisjong.learning.ron_legal_baseline import RonRateModel,StratumRates
from lisjong.policy_contract import Seat
m=RonRateModel((StratumRates((100,)*34,(50,)*34),)*3)
p=make_decision(0,A_OWN,A_PLAYERS).policy_input
assert m.predict(p,Seat(1),"ron_rate").ron_legal_probability_raw==(50,)*34
assert m.predict(p,Seat(1),"wait_genbutsu") is not None
"""
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0,'tests');\n" + code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            # The Rust shanten backend legitimately loads the native module;
            # the baseline itself must work on the default Python backend.
            env={
                k: v for k, v in os.environ.items() if k != BACKEND_ENVIRONMENT_VARIABLE
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class AggregationTest(unittest.TestCase):
    def test_fit_weights_each_stratum_episode_once(self):
        fit = accuracy.RateFit()
        # A long positive episode has the same weight as a short negative one.
        fit.add([row(episode="1")] * 9 + [row(episode="2", wait=False, ron=False)])
        rates = fit.model().tables[0]
        self.assertEqual(rates.wait_raw[0], probability_to_raw(0.5))
        self.assertEqual(rates.ron_raw[0], probability_to_raw(0.5))
        self.assertEqual(rates.wait_raw[1], probability_to_raw(0.5 / 3))
        self.assertIsNone(fit.model().tables[1])
        # A changing public stratum is independently reweighted within its rows.
        fit.add([row(episode="3", seat=2, wait=True, ron=False)])
        self.assertEqual(fit.model().tables[1].wait_raw[0], probability_to_raw(0.75))
        self.assertEqual(fit.model().tables[1].ron_raw[0], probability_to_raw(0.25))

    def test_final_raw_metrics_and_candidate_subset(self):
        values = accuracy.metrics((0,) * 34, (True,) + (False,) * 33, frozenset((0,)))
        self.assertAlmostEqual(values["candidate_log_loss"], -math.log(1e-6))
        self.assertAlmostEqual(values["brier"], 1 / 34)
        self.assertAlmostEqual(
            values["log_loss"], (-math.log(1e-6) - 33 * math.log(1 - 1e-6)) / 34
        )

    def test_episode_macro_global_and_stratum_are_not_stratum_means(self):
        per_seed = {1: {"m": [1.0, 2]}, 2: {"m": [2.0, 1]}}
        self.assertEqual(accuracy.episode_mean(per_seed, [1, 2], "m"), 1.0)
        self.assertEqual(accuracy.episode_mean(per_seed, [1, 1, 2], "m"), 0.8)
        self.assertIsNone(accuracy.episode_mean({}, [1], "m"))

    def test_unprovided_is_not_zero_and_calibration_is_episode_weighted(self):
        model = RonRateModel((StratumRates((100,) * 34, (50,) * 34), None, None))
        evaluation = accuracy.Evaluation()
        for item in [row()] * 3 + [row(episode="2", ron=False), row(seat=2)]:
            evaluation.add(item, model)
        self.assertEqual(evaluation.coverage["all"]["target_rows"], 5)
        self.assertEqual(evaluation.coverage["all"]["provided_rows"], 4)
        self.assertEqual(evaluation.coverage["all"]["unprovided_rows"], 1)
        self.assertNotIn("open", evaluation.per_seed())
        buckets = evaluation.calibration()["all"]["ron_rate"]
        self.assertAlmostEqual(sum(b["episode_weight"] for b in buckets), 2.0)
        self.assertAlmostEqual(buckets[0]["observed_rate"], 1 / 68)
        intervals = accuracy.confidence_intervals(evaluation, [1, 2])
        self.assertIsNone(intervals["open"]["ron_rate.log_loss"]["point"])
        self.assertEqual(intervals, accuracy.confidence_intervals(evaluation, [1, 2]))

    def test_support_positive_slots_rows_episodes_and_zero_groups(self):
        support = accuracy.Support()
        for item in [row()] * 3 + [row(2, ron=False), row(3, episode="2")]:
            support.add(item)
        counts = support.value()
        self.assertEqual(counts["all"]["positive_rows"], 4)
        self.assertEqual(counts["all"]["positive_slots"], 4)
        self.assertEqual(counts["all"]["positive_episodes"], 2)
        self.assertEqual(counts["all"]["positive_hanchan"], 2)
        self.assertEqual(counts["all"]["support_status"], "held")
        self.assertEqual(counts["open"]["support_status"], "not_estimable")
        with self.assertRaises(accuracy.RonLegalAccuracyError):
            row(wait=False, ron=True)


WAIT_MODEL = LogisticWaitModel(weights=(("bias", -1.0),))
RATES = RonRateModel((StratumRates((100,) * 34, (50,) * 34),) * 3)


def wait_zero(name="riichi_wait_zero"):
    return accuracy.Estimator(
        name,
        {"transform": "test"},
        lambda policy_input, seat: estimate_riichi_ron_legal_belief(
            policy_input, seat, WAIT_MODEL
        ),
    )


def selection_document():
    return {
        "schema": "lisjong-riichi-wait-selection-v1",
        "feature_set": "riichi-wait-features-v1",
        "clip_epsilon": 1e-6,
        "bootstrap": {"resamples": 2000, "seed": 245},
        "l2_grid": [{"l2": l2} for l2 in (0.01, 0.1, 1.0, 10.0, 100.0)],
        "chosen_l2": 1.0,
        "used_seeds": {},
        "source_manifest": {},
        "models": {
            "baseline1_prevalence": {"probabilities": [0.1] * 34},
            "baseline2_classical_platt": {"intercept": 0.0, "slope": 0.0},
            "estimator_logistic": {
                "feature_set": "riichi-wait-features-v1",
                "weights": dict(WAIT_MODEL.weights),
            },
        },
    }


class EstimatorTest(unittest.TestCase):
    def test_estimator_is_scored_only_on_its_rows_paired_with_baselines(self):
        plain, extended = accuracy.Evaluation(), accuracy.Evaluation([wait_zero()])
        # seat 1 is the single riichi seat (provided); seat 3 is out of scope.
        items = [row(), row(episode="2", ron=False), row(seat=3), row(2, seat=3)]
        for item in items:
            plain.add(item, RATES)
            extended.add(item, RATES)
        # Baseline cells, coverage and intervals are those of the plain run.
        self.assertEqual(extended.cells, plain.cells)
        self.assertEqual(extended.coverage, plain.coverage)
        self.assertEqual(extended.coverage["all"]["provided_rows"], 4)
        seeds = [1, 2]
        self.assertEqual(
            accuracy.confidence_intervals(extended, seeds),
            accuracy.confidence_intervals(plain, seeds),
        )
        value = extended.estimator_value(extended.estimators[0], seeds)
        self.assertEqual(
            value["coverage"]["all"],
            {"target_rows": 4, "provided_rows": 2, "unprovided_rows": 2},
        )
        self.assertEqual(value["coverage"]["closed_non_riichi"]["provided_rows"], 0)
        metrics = value["metrics"]
        self.assertIsNone(
            metrics["closed_non_riichi"]["riichi_wait_zero.log_loss"]["point"]
        )
        # "all" holds only the provided riichi rows, so it equals the stratum.
        self.assertEqual(metrics["all"], metrics["riichi"])
        belief = estimate_riichi_ron_legal_belief(
            row().policy_input, Seat(1), WAIT_MODEL
        )
        losses = [
            accuracy.metrics(raw, (y,) + (False,) * 33, frozenset((0, 1)))["log_loss"]
            for raw in (belief.ron_legal_probability_raw, (50,) * 34)
            for y in (True, False)
        ]
        point = {k: v["point"] for k, v in metrics["riichi"].items()}
        self.assertAlmostEqual(
            point["riichi_wait_zero.log_loss"], (losses[0] + losses[1]) / 2
        )
        self.assertAlmostEqual(point["ron_rate.log_loss"], (losses[2] + losses[3]) / 2)
        self.assertAlmostEqual(
            point["riichi_wait_zero_minus_wait_genbutsu.log_loss"],
            point["riichi_wait_zero.log_loss"] - point["wait_genbutsu.log_loss"],
        )
        self.assertIn("riichi_wait_zero_minus_ron_rate.log_loss", point)
        self.assertAlmostEqual(point["riichi_wait_zero.actual_ron"], 0.5)
        self.assertAlmostEqual(point["riichi_wait_zero.actual_wait_not_zeroed"], 1.0)
        self.assertEqual(point["riichi_wait_zero.actual_ron_zeroed"], 0)
        self.assertAlmostEqual(
            point["riichi_wait_zero.predicted_wait"],
            sum(belief.wait_probability_raw) / SCALE,
        )
        self.assertLess(
            point["riichi_wait_zero.predicted_ron"],
            point["riichi_wait_zero.predicted_wait"],
        )
        self.assertEqual(
            sum(b["slots"] for b in value["calibration"]["all"]["riichi_wait_zero"]), 68
        )

    def test_zeroed_ron_truth_is_counted_and_bad_estimators_fail_closed(self):
        belief = estimate_riichi_ron_legal_belief(
            row().policy_input, Seat(1), WAIT_MODEL
        )
        zeroed = replace(belief, ron_legal_probability_raw=(0,) * 34)
        evaluation = accuracy.Evaluation(
            [accuracy.Estimator("zero", {}, lambda policy_input, seat: zeroed)]
        )
        evaluation.add(row(), RATES)
        point = evaluation.estimator_value(evaluation.estimators[0], [1])["metrics"]
        self.assertEqual(point["all"]["zero.actual_ron_zeroed"]["point"], 1)
        self.assertEqual(point["all"]["zero.actual_wait_not_zeroed"]["point"], 0)
        for bad in (replace(belief, ron_legal_probability_raw=None), (0,) * 34):
            broken = accuracy.Evaluation(
                [accuracy.Estimator("bad", {}, lambda policy_input, seat: bad)]
            )
            with self.assertRaises(accuracy.RonLegalAccuracyError):
                broken.add(row(), RATES)
        # A provided row must have both baselines to be paired with.
        unpaired = accuracy.Evaluation([wait_zero()])
        with self.assertRaises(accuracy.RonLegalAccuracyError):
            unpaired.add(row(), RonRateModel((None,) * 3))
        for name in ("ron_rate", "wait_genbutsu", "not a name"):
            with self.assertRaises(accuracy.RonLegalAccuracyError):
                accuracy.Estimator(name, {}, lambda policy_input, seat: None)
        with self.assertRaises(accuracy.RonLegalAccuracyError):
            accuracy.Evaluation([wait_zero(), wait_zero()])

    def test_population_result_keeps_the_baseline_document_unchanged(self):
        from lisjong.learning.hand_belief_source import HandBeliefManifest

        manifest = HandBeliefManifest(
            {}, {"train": (1,), "valid": (2,), "test": (3,)}, {}
        )
        rows = [row(1), row(2), row(3), row(3, seat=3, wait=False, ron=False)]
        expected = {"train": [1], "valid": [2], "eval": [3]}
        with (
            patch.object(accuracy, "check_population", return_value={}),
            patch.object(accuracy, "backend_identity", return_value={}),
            patch.object(accuracy, "read_manifest", return_value=manifest),
            patch.object(accuracy, "rows_from_source", return_value=(manifest, rows)),
        ):
            plain = accuracy.evaluate_population([Path("p")], expected, {})
            extended = accuracy.evaluate_population(
                [Path("p")], expected, {}, estimators=[wait_zero()]
            )
            with self.assertRaises(accuracy.RonLegalAccuracyError):
                accuracy.evaluate_population(
                    [Path("p")],
                    expected,
                    {},
                    support_only=True,
                    estimators=[wait_zero()],
                )
        self.assertNotIn("estimators", plain)
        self.assertNotIn("scored_split", plain)
        self.assertEqual(extended.pop("scored_split"), "eval")
        estimators = extended.pop("estimators")
        self.assertEqual(extended, plain)
        value = estimators["riichi_wait_zero"]
        self.assertEqual(value["identity"], {"transform": "test"})
        self.assertEqual(
            value["coverage"]["all"],
            {"target_rows": 2, "provided_rows": 1, "unprovided_rows": 1},
        )
        json.dumps(estimators, allow_nan=False)

    def test_frozen_wait_selection_identity_fails_closed(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(selection_document()))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            estimator = accuracy.riichi_wait_zero_estimator(path, digest)
            self.assertEqual(estimator.name, "riichi_wait_zero")
            self.assertEqual(estimator.identity["selection_sha256"], digest)
            self.assertEqual(
                estimator.identity["feature_set"], "riichi-wait-features-v1"
            )
            self.assertEqual(estimator.identity["chosen_l2"], 1.0)
            self.assertEqual(
                estimator.identity["transform"], "certain-ron-illegal-zero-v1"
            )
            self.assertEqual(len(estimator.identity["weights_sha256"]), 64)
            self.assertEqual(
                estimator.predict(row().policy_input, Seat(1)),
                estimate_riichi_ron_legal_belief(
                    row().policy_input, Seat(1), WAIT_MODEL
                ),
            )
            self.assertIsNone(estimator.predict(row().policy_input, Seat(2)))
            with self.assertRaises(accuracy.RonLegalAccuracyError):
                accuracy.riichi_wait_zero_estimator(path, "0" * 64)
            changed = selection_document()
            changed["models"]["estimator_logistic"]["feature_set"] = "other"
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                accuracy.riichi_wait_zero_estimator(
                    path, hashlib.sha256(path.read_bytes()).hexdigest()
                )
            # The CLI refuses a selection without its registered digest.
            with self.assertRaises(accuracy.RonLegalAccuracyError):
                accuracy.main(
                    [
                        "--train",
                        "1",
                        "--valid",
                        "2",
                        "--eval",
                        "3",
                        "--producer",
                        "missing",
                        "--output",
                        str(Path(directory) / "result.json"),
                        "--riichi-wait-selection",
                        str(path),
                        "missing",
                    ]
                )


@unittest.skipIf(sys.platform == "win32", "open_wait_evaluation is POSIX-only")
class OpenEstimatorTest(unittest.TestCase):
    def document(self):
        return {
            "schema": "lisjong-open-wait-selection-v1",
            "feature_set": "open-wait-features-v1",
            "selected": {"l2_tenpai": 0.01, "l2_wait": 0.01},
            "models": {
                "estimator": {
                    "feature_set": "open-wait-features-v1",
                    "tenpai_weights": [["bias", 0.5]],
                    "wait_weights": [["bias", -1.0]],
                },
                "rate": [0.1] * 34,
            },
        }

    def test_selected_open_wait_identity_fails_closed(self):
        import hashlib

        model = OpenWaitModel((("bias", 0.5),), (("bias", -1.0),))
        public = row().policy_input
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(self.document()))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            estimator = accuracy.open_wait_zero_estimator(path, digest)
            self.assertEqual(estimator.name, "open_wait_zero")
            self.assertEqual(estimator.identity["selection_sha256"], digest)
            self.assertEqual(estimator.identity["scope"], "open-non-riichi-opponent.v1")
            self.assertEqual(
                estimator.identity["selected_l2"], {"l2_tenpai": 0.01, "l2_wait": 0.01}
            )
            self.assertEqual(
                estimator.identity["transform"], "certain-ron-illegal-zero-v1"
            )
            # Seat 2 is the open seat; the riichi and closed seats stay unprovided.
            self.assertEqual(
                estimator.predict(public, Seat(2)),
                estimate_open_ron_legal_belief(public, Seat(2), model),
            )
            self.assertIsNone(estimator.predict(public, Seat(1)))
            self.assertIsNone(estimator.predict(public, Seat(3)))
            with self.assertRaises(accuracy.RonLegalAccuracyError):
                accuracy.open_wait_zero_estimator(path, "0" * 64)
            # A riichi selection is not accepted as an open one, and vice versa.
            with self.assertRaises(ValueError):
                accuracy.riichi_wait_zero_estimator(path, digest)
            changed = self.document()
            changed["feature_set"] = "other"
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                accuracy.open_wait_zero_estimator(
                    path, hashlib.sha256(path.read_bytes()).hexdigest()
                )
            with self.assertRaises(accuracy.RonLegalAccuracyError):
                accuracy.main(
                    [
                        "--train",
                        "1",
                        "--valid",
                        "2",
                        "--eval",
                        "3",
                        "--producer",
                        "missing",
                        "--output",
                        str(Path(directory) / "result.json"),
                        "--open-wait-selection-sha256",
                        digest,
                        "missing",
                    ]
                )

    def test_each_estimator_is_scored_on_its_own_stratum(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(self.document()))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            evaluation = accuracy.Evaluation(
                [wait_zero(), accuracy.open_wait_zero_estimator(path, digest)]
            )
        plain = accuracy.Evaluation()
        for item in (row(), row(seat=2, ron=False), row(seat=3, wait=False, ron=False)):
            evaluation.add(item, RATES)
            plain.add(item, RATES)
        self.assertEqual(evaluation.cells, plain.cells)
        riichi, opened = (
            evaluation.estimator_value(e, [1]) for e in evaluation.estimators
        )
        self.assertEqual(riichi["coverage"]["riichi"]["provided_rows"], 1)
        self.assertEqual(riichi["coverage"]["open"]["provided_rows"], 0)
        self.assertEqual(
            opened["coverage"]["all"],
            {"target_rows": 3, "provided_rows": 1, "unprovided_rows": 2},
        )
        self.assertEqual(opened["coverage"]["open"]["provided_rows"], 1)
        self.assertEqual(opened["coverage"]["riichi"]["provided_rows"], 0)
        self.assertEqual(opened["metrics"]["all"], opened["metrics"]["open"])
        self.assertIsNotNone(
            opened["metrics"]["open"]["open_wait_zero_minus_wait_genbutsu.log_loss"][
                "point"
            ]
        )
        self.assertEqual(
            opened["metrics"]["open"]["open_wait_zero.actual_wait"]["point"], 1
        )


class PopulationTest(unittest.TestCase):
    def test_seed_cli_rejects_invalid_and_reversed_ranges(self):
        self.assertEqual(accuracy.seed_range("1..3"), [1, 2, 3])
        self.assertEqual(accuracy.seed_range("none"), [])
        for bad in ("-1", "3..1", "1..2..3", "1.0", "", "True"):
            with self.assertRaises(Exception):
                accuracy.seed_range(bad)

    def test_cli_refuses_overwrite_before_reading_source(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            target.write_text("existing")
            with self.assertRaises(accuracy.RonLegalAccuracyError):
                accuracy.main(
                    [
                        "--train",
                        "1",
                        "--valid",
                        "2",
                        "--eval",
                        "3",
                        "--producer",
                        "missing",
                        "--output",
                        str(target),
                        "missing",
                    ]
                )
            self.assertEqual(target.read_text(), "existing")

    def test_population_checks_identity_split_and_duplicates_before_labels(self):
        from lisjong.learning.hand_belief_source import HandBeliefManifest

        producer = {
            "arena_revision": "a" * 40,
            "lisjong_revision": "b" * 40,
            "lisjong_engine_revision": "c" * 40,
            "policy": "p",
        }
        manifest = HandBeliefManifest(
            producer, {"train": (1,), "valid": (2,), "test": (3,)}, {}
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base").mkdir()
            (root / "ron").mkdir()
            (root / "base/manifest.json").write_text("{}")
            (root / "ron/manifest.json").write_text(
                json.dumps(
                    {
                        "context_protocol": accuracy.CONTEXT_PROTOCOL,
                        "rules": accuracy.RULES,
                    }
                )
            )
            expected = {"train": [1], "valid": [2], "eval": [3]}
            with patch.object(accuracy, "read_manifest", return_value=manifest):
                accuracy.check_population([root], expected, producer)
                for sources, seeds, identity in (
                    ([root, root], expected, producer),
                    ([root], {**expected, "eval": [4]}, producer),
                    ([root], {**expected, "eval": [2]}, producer),
                    ([root], expected, {**producer, "arena_revision": "d" * 40}),
                ):
                    with self.assertRaises(accuracy.RonLegalAccuracyError):
                        accuracy.check_population(sources, seeds, identity)

    def test_measurement_fits_train_only_and_scores_only_eval_rows(self):
        from lisjong.learning.hand_belief_source import HandBeliefManifest

        manifest = HandBeliefManifest(
            {}, {"train": (1,), "valid": (2,), "test": (3,)}, {}
        )
        rows = [row(1), row(2, wait=False, ron=False), row(3, wait=False, ron=False)]
        expected = {"train": [1], "valid": [2], "eval": [3]}
        with (
            patch.object(accuracy, "check_population", return_value={}),
            patch.object(accuracy, "backend_identity", return_value={}),
            patch.object(accuracy, "read_manifest", return_value=manifest),
            patch.object(accuracy, "rows_from_source", return_value=(manifest, rows)),
        ):
            result = accuracy.evaluate_population([Path("p")], expected, {})
            support = accuracy.evaluate_population(
                [Path("p")], expected, {}, support_only=True
            )
        # Only the positive train row is fitted: (1 + 0.5) / (1 + 1).
        rate = result["model"]["strata"]["riichi"]["ron_raw"][0]
        self.assertEqual(rate, probability_to_raw(0.75))
        self.assertEqual(
            [result["support"][s]["all"]["rows"] for s in expected], [1, 1, 1]
        )
        self.assertEqual(result["coverage"]["all"]["provided_rows"], 1)
        point = result["metrics"]["riichi"]["ron_rate.log_loss"]["point"]
        self.assertAlmostEqual(
            point,
            accuracy.metrics(
                result["model"]["strata"]["riichi"]["ron_raw"],
                (False,) * 34,
                frozenset((0,)),
            )["log_loss"],
        )
        json.dumps(result, allow_nan=False)
        self.assertEqual(support["mode"], "support_only")
        self.assertEqual(support["metrics"], {})
        self.assertEqual(support["model_sha256"], result["model_sha256"])

    def test_score_valid_scores_valid_rows_and_never_reads_eval_sources(self):
        from lisjong.learning.hand_belief_source import HandBeliefManifest

        manifests = {
            "tv": HandBeliefManifest(
                {}, {"train": (1,), "valid": (2,), "test": ()}, {}
            ),
            "e": HandBeliefManifest({}, {"train": (), "valid": (), "test": (3,)}, {}),
        }
        read = []

        def rows_from(root):
            read.append(root.name)
            return manifests[root.name], [row(1), row(2, wait=False, ron=False)]

        expected = {"train": [1], "valid": [2], "eval": [3]}
        with (
            patch.object(accuracy, "check_population", return_value={}),
            patch.object(accuracy, "backend_identity", return_value={}),
            patch.object(
                accuracy,
                "read_manifest",
                side_effect=lambda p: manifests[p.parent.name],
            ),
            patch.object(accuracy, "rows_from_source", side_effect=rows_from),
        ):
            result = accuracy.evaluate_population(
                [Path("tv"), Path("e")], expected, {}, score="valid"
            )
            for bad in ({"score": "test"}, {"score": "valid", "support_only": True}):
                with self.assertRaises(accuracy.RonLegalAccuracyError):
                    accuracy.evaluate_population([Path("tv")], expected, {}, **bad)
        self.assertEqual(set(read), {"tv"})
        self.assertEqual(result["scored_split"], "valid")
        self.assertEqual(result["estimators"], {})
        self.assertEqual(result["coverage"]["all"]["provided_rows"], 1)
        self.assertEqual(result["support"]["eval"]["all"]["rows"], 0)
        self.assertEqual(
            result["metrics"]["riichi"]["ron_rate.log_loss"]["resamples_with_rows"],
            accuracy.BOOTSTRAP_RESAMPLES,
        )

    def test_source_rows_keep_round_episode_and_replayed_diagnostics(self):
        # Stubbed yaku/backend, like SourceTest: this fixes the join, not scoring.
        fixture = ron_fixture.SourceTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.extra = fixture.root / "ron"
        fixture.extra.mkdir()
        fixture.write()
        with patch("lisjong.learning.ron_legal_source.require_scoring_backend"):
            manifest, rows = accuracy.rows_from_source(fixture.root)
        seed = ron_fixture.SEED
        self.assertEqual(manifest.split_of(seed), "train")
        self.assertEqual(len(rows), 6)
        before, after = (
            next(r for r in rows[start : start + 3] if r.seat is Seat.SEAT_2)
            for start in (0, 3)
        )
        self.assertEqual(before.episode, (seed, "round-0", 2))
        self.assertEqual(before.diagnostics, ())
        self.assertTrue(any(before.ron))
        # The legal ron passed between the two decisions blocks every wait.
        self.assertEqual(after.diagnostics, ("temporary_furiten",))
        self.assertEqual(after.wait, before.wait)
        self.assertFalse(any(after.ron))


if __name__ == "__main__":
    unittest.main()
