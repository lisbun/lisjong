import json
import math
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
from lisjong.learning import ron_legal_accuracy as accuracy
from lisjong.learning.ron_legal_baseline import (
    RonRateModel,
    StratumRates,
    public_stratum,
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
