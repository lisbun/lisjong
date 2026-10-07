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

from test_learning_hand_belief_source import (
    SEED,
    all_decisions,
    all_facts,
    chi,
    tiles,
    tt,
    write_source,
)

from lisjong.learning import open_wait_estimator as estimator
from lisjong.learning import open_wait_evaluation as evaluation
from lisjong.learning.hand_belief_source import label_decisions
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import TileCategory


def labelled():
    return label_decisions(all_decisions(), all_facts())


def with_seed(items, seed):
    return [replace(item, key=replace(item.key, seed=seed)) for item in items]


def pi_of(sequence):
    return {d.key.sequence: d for d in all_decisions()}[sequence].policy_input


class ScopeTest(unittest.TestCase):
    def test_only_non_riichi_opponents_with_a_call_are_in_scope(self):
        a = pi_of(20)  # seat1 riichi, seat2 pon, seat3 ankan only
        self.assertEqual(
            [estimator.is_open_opponent(a, seat) for seat in range(4)],
            [False, False, True, False],
        )
        for seat in (0, 1, 3):
            with self.assertRaises(ValueError):
                estimator.open_view(a, seat)

    def test_rows_are_the_open_opponents_with_their_structural_wait(self):
        rows = list(evaluation.build_open_rows(labelled()))
        self.assertEqual([row.episode[1] for row in rows], [2, 2])
        self.assertFalse(rows[0].tenpai)  # A: 1357m2468p99s + pon 5z
        waits = [i for i, label in enumerate(rows[1].labels) if label]
        self.assertEqual(waits, [2])  # B: 12m penchan -> 3m
        # A and B share a kyoku, so seat2 is one episode of two rows
        self.assertEqual(evaluation.episode_weights(rows), [0.5, 0.5])


class FeatureTest(unittest.TestCase):
    def test_meld_dora_counts_repeated_indicators_and_red_tiles(self):
        for meld_spec, indicators, expected in (
            ("456p", "4p", 1),
            ("456p", "44p", 2),
            ("406p", "44p", 3),
        ):
            with self.subTest(meld=meld_spec, indicators=indicators):
                policy_input = pi_of(30)
                players = list(policy_input.players)
                players[2] = replace(
                    players[2], melds=(chi(meld_spec, "4p", Seat.SEAT_1),)
                )
                policy_input = replace(
                    policy_input,
                    players=tuple(players),
                    round=replace(
                        policy_input.round, dora_indicators=tiles(indicators)
                    ),
                )
                view = estimator.open_view(policy_input, 2)
                self.assertEqual(view.dora_in_melds, expected)
                self.assertEqual(view.dora, frozenset({tt("5p")}))
                names = {name for name, _ in estimator.tenpai_features(view)}
                self.assertIn(f"dora_in_melds_{min(expected, 2)}", names)

    def test_meld_signals(self):
        pon_view = estimator.open_view(pi_of(20), 2)
        self.assertTrue(pon_view.yakuhai_meld)
        self.assertTrue(pon_view.all_triplets)
        self.assertIsNone(pon_view.flush_suit)
        self.assertEqual(pon_view.discards_since_call, 0)
        names = {name for name, _ in estimator.tenpai_features(pon_view)}
        self.assertIn("melds_yakuhai", names)
        self.assertNotIn("melds_no_visible_yaku", names)

        chi_view = estimator.open_view(pi_of(30), 2)
        self.assertIs(chi_view.flush_suit, TileCategory.SOUZU)
        self.assertTrue(chi_view.all_simples)
        self.assertFalse(chi_view.yakuhai_meld)
        tile_names = {
            tile: {name for name, _ in features}
            for tile, features in zip(
                estimator.TILE_TYPES, estimator.wait_tile_feature_table(chi_view)
            )
        }
        self.assertIn("flush_in_suit", tile_names[tt("5s")])
        self.assertIn("flush_off_suit", tile_names[tt("5m")])
        self.assertIn("simples_terminal_or_honor", tile_names[tt("1z")])
        self.assertIn("in_meld", tile_names[tt("3s")])

    def test_river_features_never_fix_a_probability_at_zero(self):
        # seat1 discarded 2s, so for the riichi-less seat2 nothing is genbutsu yet;
        # weights that ignore everything give the same value to every tile
        model = estimator.OpenWaitModel(tenpai_weights=(), wait_weights=())
        predicted = model.predict(pi_of(30), 2)
        self.assertEqual(set(predicted.values()), {0.25})
        heavy = estimator.OpenWaitModel(
            tenpai_weights=(), wait_weights=(("genbutsu", -50.0),)
        )
        self.assertTrue(all(p > 0.0 for p in heavy.predict(pi_of(30), 2).values()))

    def test_the_belief_replaces_only_the_wait_table(self):
        model = estimator.OpenWaitModel(tenpai_weights=(), wait_weights=())
        belief = estimator.estimate_open_wait_belief(pi_of(30), 2, model)
        self.assertAlmostEqual(belief.wait_probability(tt("3m")), 0.25, places=6)
        self.assertIsNone(belief.tanki_wait_probability_raw)


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
            "open_wait_evaluation",
            "hand_belief_accuracy",
            "exact_wait_ground_truth",
            "riichi_ron_label",
            "lisjong_arena",
        )
        self.assertFalse(any(word in name for name in imported for word in forbidden))

    def test_predict_takes_only_the_player_safe_input(self):
        self.assertEqual(
            list(inspect.signature(estimator.OpenWaitModel.predict).parameters),
            ["self", "policy_input", "seat"],
        )


class FitTest(unittest.TestCase):
    def test_rate_weights_each_episode_once(self):
        rows = list(evaluation.build_open_rows(labelled()))
        rate = evaluation.fit_rate(rows)
        # one episode (weight 1) whose rows are half tenpai on 3m
        self.assertAlmostEqual(rate[2], (0.5 + 0.5) / (1 + 1))
        self.assertAlmostEqual(rate[0], 0.5 / 2)

    def test_the_wait_stage_learns_from_tenpai_rows_only(self):
        rows = list(evaluation.build_open_rows(labelled()))
        samples = list(evaluation.wait_samples(rows))
        self.assertEqual(len(samples), 34)
        self.assertEqual(sum(label for _, label, _ in samples), 1)
        fits = evaluation.fit_stage(samples, (1.0,))
        weights = dict(fits[1.0])
        self.assertLess(weights[estimator.BIAS], 0.0)

    def test_within_row_auc(self):
        self.assertEqual(evaluation.within_row_auc([0.9, 0.1, 0.1], [1, 0, 0]), 1.0)
        self.assertEqual(evaluation.within_row_auc([0.1, 0.1], [1, 0]), 0.5)
        self.assertIsNone(evaluation.within_row_auc([0.1, 0.1], [0, 0]))

    def test_verdict(self):
        def document(high, support="reported"):
            return {
                "support": support,
                "intervals": {"open/delta.log_loss": {"high_97.5": high}},
            }

        self.assertEqual(evaluation.verdict(document(-0.001)), "pass")
        self.assertEqual(evaluation.verdict(document(0.0)), "not_confirmed")
        self.assertEqual(evaluation.verdict(document(-1.0, "held")), "held")


class SelectAndTestTest(unittest.TestCase):
    def write(self, root, seed, splits):
        root.mkdir()
        write_source(
            root,
            with_seed(all_decisions(), seed),
            with_seed(all_facts(), seed),
            splits=splits,
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
            self.assertEqual(len(chosen["valid_grid"]), 25)
            self.assertEqual(chosen["dev_eval"]["counts"]["rows"], 2)
            self.assertEqual(chosen["dev_eval"]["support"], "held")
            selection = json.loads(json.dumps(chosen))

            self.write(
                d / "new", SEED + 3, {"train": [], "valid": [], "test": [SEED + 3]}
            )
            result = evaluation.run_test([d / "new"], [SEED + 3], selection)
            self.assertEqual(result["verdict"], "held")
            self.assertIn("open/delta.log_loss", result["intervals"])
            with self.assertRaises(evaluation.OpenWaitEvaluationError):
                evaluation.run_test([d / "test"], [SEED + 2], selection)

            output = d / "selection.json"
            quiet = contextlib.redirect_stdout(io.StringIO())
            quiet.__enter__()
            self.addCleanup(quiet.__exit__, None, None, None)
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
            result_path = d / "result.json"
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
                    str(result_path),
                    str(d / "new"),
                ]
            )
            self.assertEqual(json.loads(result_path.read_text())["verdict"], "held")
            with (
                self.assertRaises(SystemExit),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                evaluation.main(
                    [
                        "test",
                        "--test",
                        str(SEED + 3),
                        "--selection",
                        str(output),
                        "--selection-sha256",
                        "0" * 64,
                        "--output",
                        str(d / "other.json"),
                        str(d / "new"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
