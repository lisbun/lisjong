import ast
import contextlib
import hashlib
import inspect
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from test_learning_hand_belief_source import (
    SEED,
    all_decisions,
    all_facts,
    tt,
    write_source,
)

from lisjong.belief.canonical_axes import tile_type_index, wind_for_seat, wind_index
from lisjong.belief.conditional_uniform_hand_belief import (
    estimate_conditional_uniform_hand_belief,
)
from lisjong.belief.non_player_hidden_belief import derive_non_player_hidden_belief
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.learning import hand_count_estimator as estimator
from lisjong.learning import hand_count_evaluation as evaluation
from lisjong.learning._canonical import canonical_json_text
from lisjong.learning.hand_belief_source import (
    HandBeliefSourceError,
    label_decisions,
)
from lisjong.policy_contract.seat import Seat


def with_seed(items, seed):
    return [replace(item, key=replace(item.key, seed=seed)) for item in items]


def pi_of(sequence):
    return {d.key.sequence: d for d in all_decisions()}[sequence].policy_input


class AllocationTest(unittest.TestCase):
    def test_zero_weights_reproduce_the_conditional_uniform_baseline(self):
        pi = pi_of(20)
        model = estimator.HandCountModel(weights=())
        predicted = model.predict(pi)
        context = estimator.count_context(pi)
        dealer = pi.round.dealer_seat
        slots = [0, 0, 0, 0]
        for view in context.opponents:
            slots[wind_index(wind_for_seat(Seat(view.seat), dealer))] = view.slots
        uniform = estimate_conditional_uniform_hand_belief(pi, tuple(slots))
        for view in context.opponents:
            wind = wind_for_seat(Seat(view.seat), dealer)
            for a, b in zip(
                predicted.hand(wind).expected_count_raw,
                uniform.hand(wind).expected_count_raw,
            ):
                # 切り捨て（5は赤と非赤の2回）と一様側の丸めの差だけ
                self.assertLessEqual(abs(a - b), 2)

    def test_weights_shift_mass_but_keep_conservation(self):
        pi = pi_of(20)
        model = estimator.HandCountModel(
            weights=(("river_1", -3.0), ("class_honor_guest", 1.5), ("dora", 2.0))
        )
        belief = model.predict(pi)
        conservation = derive_remaining_tile_inventory(pi)
        self_wind = wind_for_seat(pi.self_seat, pi.round.dealer_seat)
        # raises if the opponents hold more than the remaining physical mass
        derive_non_player_hidden_belief(conservation, belief, self_wind)
        context = estimator.count_context(pi, conservation)
        expected = model.expected_from(context)
        for index, remaining in enumerate(context.remaining):
            self.assertLessEqual(sum(row[index] for row in expected), remaining + 1e-9)
        for view, row in zip(context.opponents, expected):
            self.assertAlmostEqual(sum(row), view.slots, places=6)

    def test_red_five_follows_the_share_of_its_five(self):
        pi = pi_of(20)
        context = estimator.count_context(pi)
        counts = [0.0] * 34
        five = tile_type_index(tt("5p"))
        counts[five] = 0.5
        reds = estimator.red_five_probabilities(context, tuple(counts))
        remaining = context.remaining[five]
        self.assertAlmostEqual(reds[1], 0.5 * context.remaining_red[1] / remaining)


class FeatureTest(unittest.TestCase):
    def test_river_and_meld_features(self):
        pi = pi_of(20)  # seat1 riichi discarded 9p and 5z; seat2 pon 5z
        context = estimator.count_context(pi)
        riichi, pon = context.opponents[0], context.opponents[1]
        self.assertEqual(riichi.stratum, "riichi")
        self.assertEqual(pon.stratum, "open")
        nine_p = tile_type_index(tt("9p"))
        names = estimator.cell_features(context, riichi, nine_p)
        self.assertIn("river_1", names)
        self.assertIn("discarded_early", names)
        self.assertIn("riichi_x_terminal", names)
        self.assertIn(
            "near1_in_river", estimator.cell_features(context, riichi, nine_p - 1)
        )
        self.assertIn(
            "in_meld",
            estimator.cell_features(context, pon, tile_type_index(tt("5z"))),
        )


class BoundaryTest(unittest.TestCase):
    def test_inference_does_not_import_ground_truth(self):
        tree = ast.parse(Path(inspect.getfile(estimator)).read_text())
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        forbidden = (
            "hand_belief_source",
            "hand_count_evaluation",
            "hand_belief_accuracy",
            "exact_wait_ground_truth",
            "riichi_ron_label",
            "lisjong_arena",
        )
        self.assertFalse(any(word in name for name in imported for word in forbidden))

    def test_predict_takes_only_the_player_safe_input(self):
        self.assertEqual(
            list(inspect.signature(estimator.HandCountModel.predict).parameters),
            ["self", "policy_input"],
        )


class FitTest(unittest.TestCase):
    def test_poisson_fit_recovers_a_ratio(self):
        # feature "a": observed twice the uniform expectation; bias cell: equal
        patterns = {("a", "bias"): [10.0, 20.0], ("bias",): [10.0, 10.0]}
        weights = dict(evaluation.fit_poisson(patterns, 1e-6))
        self.assertAlmostEqual(weights["bias"], 0.0, places=4)
        self.assertAlmostEqual(weights["a"], 0.6931, places=3)

    def test_verdicts_are_separate(self):
        def interval(high):
            return {"high_97.5": high}

        document = {
            "red_support": "reported",
            "intervals": {
                "all/delta.count_mse": interval(-0.01),
                "all/delta.red_log_loss": interval(0.01),
            },
        }
        self.assertEqual(
            evaluation.verdicts(document),
            {"expected_count": "pass", "red_five": "not_confirmed"},
        )
        document["red_support"] = "held"
        self.assertEqual(evaluation.verdicts(document)["red_five"], "held")


class SelectAndTestTest(unittest.TestCase):
    def write(self, root, seed, splits):
        root.mkdir()
        write_source(
            root,
            with_seed(all_decisions(), seed),
            with_seed(all_facts(), seed),
            splits=splits,
        )

    def test_rows_cover_every_opponent(self):
        labelled = label_decisions(all_decisions(), all_facts())
        decisions = list(evaluation.decisions_of_seed(labelled))
        rows = [row for _, _, opponents in decisions for row in opponents]
        self.assertEqual(len(rows), 9)
        self.assertEqual(
            [row.stratum for row in rows[:3]], ["riichi", "open", "closed_non_riichi"]
        )

    def test_select_then_one_test_on_new_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            for offset, split in enumerate(("train", "valid", "test")):
                self.write(
                    d / split,
                    SEED + offset,
                    {
                        name: [SEED + offset] if name == split else []
                        for name in ("train", "valid", "test")
                    },
                )
            sources = [d / "train", d / "valid", d / "test"]
            chosen = evaluation.select(
                sources, {"train": [SEED], "valid": [SEED + 1], "eval": [SEED + 2]}
            )
            self.assertEqual(len(chosen["valid_grid"]), 5)
            self.assertEqual(chosen["dev_eval"]["counts"]["all"]["rows"], 9)
            selection = json.loads(json.dumps(chosen))
            self.write(
                d / "new", SEED + 3, {"train": [], "valid": [], "test": [SEED + 3]}
            )
            result = evaluation.run_test([d / "new"], [SEED + 3], selection)
            self.assertEqual(result["verdicts"]["red_five"], "held")
            self.assertIn("all/delta.count_mse", result["intervals"])
            with self.assertRaises(evaluation.HandCountEvaluationError):
                evaluation.run_test([d / "test"], [SEED + 2], selection)

            output = d / "selection.json"
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(
                    [
                        "select",
                        "--train",
                        str(SEED),
                        "--valid",
                        str(SEED + 1),
                        "--dev-eval",
                        str(SEED + 2),
                        "--output",
                        str(output),
                        *map(str, sources),
                    ]
                )
                digest = hashlib.sha256(output.read_bytes()).hexdigest()
                evaluation.main(
                    [
                        "test",
                        "--test",
                        str(SEED + 3),
                        "--selection",
                        str(output),
                        "--selection-sha256",
                        digest,
                        "--output",
                        str(d / "result.json"),
                        str(d / "new"),
                    ]
                )
            self.assertIn("verdicts", json.loads((d / "result.json").read_text()))


class ProducerTest(unittest.TestCase):
    """formal testは、selectionを作ったproducerと全fieldで一致するsourceだけを受け付ける。"""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for offset, split in enumerate(("train", "valid", "test")):
            (self.root / split).mkdir()
            write_source(
                self.root / split,
                with_seed(all_decisions(), SEED + offset),
                with_seed(all_facts(), SEED + offset),
                splits={
                    name: [SEED + offset] if name == split else []
                    for name in ("train", "valid", "test")
                },
            )
        self.selection = json.loads(
            json.dumps(
                evaluation.select(
                    [self.root / name for name in ("train", "valid", "test")],
                    {"train": [SEED], "valid": [SEED + 1], "eval": [SEED + 2]},
                )
            )
        )

    def new_source(self, name, change):
        root = self.root / name
        root.mkdir()
        write_source(
            root,
            with_seed(all_decisions(), SEED + 3),
            with_seed(all_facts(), SEED + 3),
            splits={"train": [], "valid": [], "test": [SEED + 3]},
        )
        manifest = json.loads((root / "manifest.json").read_text())
        change(manifest["producer"])
        (root / "manifest.json").write_text(canonical_json_text(manifest))
        return root

    def test_every_producer_field_must_match_the_selection(self):
        registered = self.selection["population"]["producer"]
        self.assertEqual(
            set(registered),
            {"arena_revision", "lisjong_revision", "lisjong_engine_revision", "policy"},
        )
        for field_name in sorted(registered):

            def change(producer, field_name=field_name):
                producer[field_name] = "f" * 40

            source = self.new_source(field_name, change)
            with (
                self.subTest(field_name),
                mock.patch.object(
                    evaluation, "read_labelled_source", side_effect=AssertionError
                ),
                self.assertRaisesRegex(evaluation.HandBeliefAccuracyError, field_name),
            ):
                evaluation.run_test([source], [SEED + 3], self.selection)

    def test_an_extra_producer_field_is_rejected(self):
        source = self.new_source(
            "extra", lambda producer: producer.update(wrapper_revision="d" * 40)
        )
        # producerのfieldはmanifestの読み込みで固定されている（未知のfieldは拒否）
        with self.assertRaisesRegex(HandBeliefSourceError, "unexpected fields"):
            evaluation.run_test([source], [SEED + 3], self.selection)

    def test_the_same_producer_is_accepted(self):
        source = self.new_source("same", lambda producer: None)
        result = evaluation.run_test([source], [SEED + 3], self.selection)
        self.assertEqual(
            result["population"]["producer"], self.selection["population"]["producer"]
        )

    def test_a_registered_test_producer_may_differ_only_in_revisions(self):
        # lisbun/lisjong#279: 新seedのsourceは、事前登録した実行revisionで生成される
        revisions = {
            "arena_revision": "1" * 40,
            "lisjong_revision": "2" * 40,
            "lisjong_engine_revision": "3" * 40,
        }
        selected = self.selection["population"]["producer"]
        registered = selected | revisions
        source = self.new_source("revised", lambda producer: producer.update(revisions))
        with self.assertRaises(evaluation.HandBeliefAccuracyError):
            evaluation.run_test([source], [SEED + 3], self.selection)
        result = evaluation.run_test([source], [SEED + 3], self.selection, registered)
        self.assertEqual(result["population"]["producer"], registered)
        self.assertEqual(
            result["producer_registration"],
            {
                "selection_producer": selected,
                "registered_test_producer": registered,
                "differing_revision_fields": sorted(revisions),
            },
        )
        # 登録と違うproducerのsourceは、登録があってもラベルを読む前に拒否する
        unregistered = self.new_source("unregistered", lambda producer: None)
        with (
            mock.patch.object(
                evaluation, "read_labelled_source", side_effect=AssertionError
            ),
            self.assertRaisesRegex(
                evaluation.HandBeliefAccuracyError, "arena_revision"
            ),
        ):
            evaluation.run_test([unregistered], [SEED + 3], self.selection, registered)
        plain = evaluation.run_test([unregistered], [SEED + 3], self.selection)
        self.assertNotIn("producer_registration", plain)

    def test_a_registration_cannot_change_the_policy_or_the_fields(self):
        selected = self.selection["population"]["producer"]
        other_policy = {"policy": "OtherPolicy"}
        source = self.new_source(
            "policy", lambda producer: producer.update(other_policy)
        )
        for registered in (
            selected | other_policy,
            {name: value for name, value in selected.items() if name != "policy"},
            selected | {"wrapper_revision": "d" * 40},
            selected | {"arena_revision": ""},
            selected | {"arena_revision": None},
        ):
            with (
                self.subTest(registered),
                mock.patch.object(
                    evaluation, "read_labelled_source", side_effect=AssertionError
                ),
                self.assertRaises(evaluation.HandBeliefAccuracyError),
            ):
                evaluation.run_test([source], [SEED + 3], self.selection, registered)


if __name__ == "__main__":
    unittest.main()
