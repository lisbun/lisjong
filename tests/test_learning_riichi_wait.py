"""#245 単独リーチ者の構造的待ち確率の推定器・ベースライン・比較手順のtest。"""

import ast
import inspect
import json
import tempfile
import unittest
from dataclasses import replace
from math import log
from pathlib import Path

import test_learning_riichi_deal_in_source as source_fixtures
from test_learning_riichi_deal_in_estimator import TANKI_5Z, policy_input
from test_learning_riichi_deal_in_source import (
    KEY,
    make_decision,
    make_facts,
    tiles,
    tt,
    write_source,
)

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.conditional_uniform_hand_belief import (
    estimate_conditional_uniform_hand_belief,
)
from lisjong.learning import riichi_wait_estimator as estimator_module
from lisjong.learning import riichi_wait_evaluation as evaluation_module
from lisjong.learning.riichi_deal_in_estimator import riichi_view
from lisjong.learning.riichi_deal_in_source import (
    DecisionKey,
    RiichiDealInSourceError,
    RiichiEpisodeKey,
    label_waits,
    read_decisions,
    read_wait_labelled_source,
)
from lisjong.learning.riichi_wait_estimator import (
    CLIP_EPSILON,
    TILE_TYPES,
    ClassicalScoreWaitModel,
    LogisticWaitModel,
    PrevalenceWaitModel,
    classical_wait_score,
    estimate_riichi_wait_belief,
    wait_feature_table,
)
from lisjong.learning.riichi_wait_evaluation import (
    L2_GRID,
    bootstrap_difference,
    compare,
    evaluate,
    fit_classical,
    fit_estimator,
    fit_prevalence,
    main,
    row_weights,
)
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    _classical_riichi_danger_score,
)


def features_of(pi, tile):
    return dict(wait_feature_table(pi)[tile_type_index(tt(tile))])


class WaitLabelTest(unittest.TestCase):
    def test_labels_are_the_structural_waits_of_the_attached_hand(self):
        (labelled,) = label_waits([make_decision()], [make_facts()])
        self.assertEqual(labelled.wait_tile_types, frozenset({tt("1m"), tt("4m")}))
        self.assertEqual(
            labelled.episode,
            RiichiEpisodeKey(seed=KEY.seed, riichi_seat=1, declared_sequence=5),
        )

    def test_furiten_does_not_change_the_structural_wait(self):
        # 4mが河にあるフリテンでも、構造的な待ちは変わらない
        (labelled,) = label_waits([make_decision(r_discards="9p4m")], [make_facts()])
        self.assertEqual(labelled.wait_tile_types, frozenset({tt("1m"), tt("4m")}))

    def test_each_row_uses_its_own_attached_hand(self):
        other = DecisionKey(seed=KEY.seed, sequence=21, seat=0)
        rows = label_waits(
            [make_decision(), make_decision(key=other)],
            [make_facts(), make_facts(key=other, concealed=TANKI_5Z)],
        )
        self.assertEqual(rows[0].wait_tile_types, frozenset({tt("1m"), tt("4m")}))
        self.assertEqual(rows[1].wait_tile_types, frozenset({tt("5z")}))

    def test_timing_and_key_checks_are_shared_with_label_a(self):
        for decisions, facts in (
            ([make_decision()], [make_facts(hand_sequence=KEY.sequence)]),
            ([make_decision()], [make_facts(declared=6, hand_sequence=5)]),
            ([make_decision()], [make_facts(riichi_seat=2)]),
            ([make_decision()], []),
        ):
            with self.assertRaises(RiichiDealInSourceError):
                label_waits(decisions, facts)

    def test_a_hand_without_waits_fails_closed(self):
        with self.assertRaises(RiichiDealInSourceError):
            label_waits(
                [make_decision()], [make_facts(concealed=tiles("1469m258p147s12z5z"))]
            )

    def test_ankan_hand_is_labelled_from_concealed_tiles_and_meld(self):
        from lisjong.policy_contract.meld import MeldKind, PublicMeld

        meld = PublicMeld(
            kind=MeldKind.ANKAN, tiles=tiles("7777s"), from_seat=None, called_tile=None
        )
        (labelled,) = label_waits(
            [make_decision()],
            [make_facts(concealed=tiles("123m456p11z23m"), melds=(meld,))],
        )
        self.assertEqual(labelled.wait_tile_types, frozenset({tt("1m"), tt("4m")}))


class SourceFilesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reading_path_round_trip(self):
        write_source(self.root, [make_decision()], [make_facts()])
        manifest, labelled = read_wait_labelled_source(self.root)
        self.assertEqual(manifest.split_of(KEY.seed), "train")
        self.assertEqual(len(labelled), 1)
        (self.root / "label_facts.jsonl").unlink()
        _, decisions = read_decisions(self.root)  # 推論入力の経路はラベルを読まない
        self.assertEqual(decisions[0].key, KEY)

    def test_tampered_label_facts_fail_closed(self):
        write_source(self.root, [make_decision()], [make_facts()])
        path = self.root / "label_facts.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("false", "true", 1))
        with self.assertRaises(RiichiDealInSourceError):
            read_wait_labelled_source(self.root)


class FeatureTest(unittest.TestCase):
    def test_support_follows_the_remaining_counts(self):
        pi = policy_input({1: [(1, "9p")]})
        # 自分の手牌 1m 4m 5z が見えている。1mの残りは3枚
        features = features_of(pi, "5m")
        self.assertIn("support_kanchan", features)  # 4m 6m
        self.assertIn("support_ryanmen_low", features)  # 6m 7m
        self.assertIn("support_tanki", features)
        self.assertIn("support_shanpon", features)
        self.assertIn("remaining_self_4", features)

    def test_river_features_do_not_remove_the_support(self):
        pi = policy_input({1: [(1, "9p"), (2, "5m")], 2: [(3, "2m")]})
        features = features_of(pi, "5m")
        self.assertIn("genbutsu", features)
        self.assertIn(
            "support_kanchan", features
        )  # 河にあっても形の成立可能性は変えない
        passed = features_of(policy_input({1: [(1, "9p")], 2: [(2, "2m")]}), "2m")
        self.assertIn("passed_since_riichi_discard", passed)

    def test_suji_closes_only_the_open_flag_of_the_ryanmen(self):
        # 5mが河にあると、2mを低い側とする両面(3m4m→2m,5m)は反対側が河にある
        pi = policy_input({1: [(1, "5m")]})
        features = features_of(pi, "2m")
        self.assertIn("support_ryanmen_low", features)
        self.assertNotIn("ryanmen_low_open", features)
        self.assertIn("suji_full", features)  # 2mの筋は5mだけ
        self.assertIn(
            "ryanmen_low_open", features_of(policy_input({1: [(1, "9p")]}), "2m")
        )

    def test_honor_tiles_have_no_number_features(self):
        features = features_of(policy_input({1: [(1, "9p")]}), "3z")
        self.assertIn("class_honor_guest", features)
        self.assertIn(
            "class_honor_yakuhai", features_of(policy_input({1: [(1, "9p")]}), "5z")
        )
        self.assertFalse(any(name.startswith("suji_") for name in features))
        self.assertFalse(any(name.startswith("combo_") for name in features))

    def test_table_covers_all_34_tiles_in_canonical_order(self):
        table = wait_feature_table(policy_input({1: [(1, "9p")]}))
        self.assertEqual(len(table), 34)
        self.assertEqual(len(TILE_TYPES), 34)
        for features in table:
            self.assertEqual(features, tuple(sorted(features)))

    def test_out_of_scope_inputs_fail_closed(self):
        for kwargs in ({"riichi": ()}, {"riichi": (1, 2)}, {"riichi": (0, 1)}):
            with self.assertRaises(ValueError):
                wait_feature_table(
                    policy_input({1: [(1, "9p")], 2: [(2, "1p")]}, **kwargs)
                )

    def test_features_do_not_depend_on_anything_but_the_policy_input(self):
        pi = policy_input({1: [(1, "9p")]})
        self.assertEqual(wait_feature_table(pi), wait_feature_table(pi))
        self.assertEqual(
            list(inspect.signature(wait_feature_table).parameters), ["policy_input"]
        )


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.pi = policy_input({1: [(1, "9p"), (5, "2s")], 2: [(6, "4m")]})

    def test_all_models_predict_34_clipped_probabilities(self):
        models = (
            PrevalenceWaitModel((0.0,) * 34),
            ClassicalScoreWaitModel(intercept=100.0, slope=100.0),
            LogisticWaitModel(weights=(("bias", 0.0),)),
        )
        for model in models:
            predictions = model.predict(self.pi)
            self.assertEqual(set(predictions), set(TILE_TYPES))
            for value in predictions.values():
                self.assertTrue(CLIP_EPSILON <= value <= 1 - CLIP_EPSILON)

    def test_genbutsu_and_passed_tiles_are_not_forced_to_zero(self):
        predictions = LogisticWaitModel(weights=(("bias", 0.0),)).predict(self.pi)
        self.assertEqual(predictions[tt("2s")], 0.5)  # 現物
        self.assertEqual(predictions[tt("4m")], 0.5)  # リーチ後に他家が切った牌
        flagged = LogisticWaitModel(weights=(("genbutsu", -2.0),)).predict(self.pi)
        self.assertLess(flagged[tt("2s")], 0.5)
        self.assertEqual(flagged[tt("1m")], 0.5)

    def test_classical_score_matches_the_champion_where_the_champion_is_defined(self):
        view = riichi_view(self.pi)
        player = self.pi.players[view.riichi_seat]
        for tile in TILE_TYPES:
            if view.remaining_counts[tile_type_index(tile)] > 3:
                continue
            self.assertEqual(
                classical_wait_score(view, player, tile),
                _classical_riichi_danger_score(
                    tile, player, view.remaining_counts
                ).total,
            )

    def test_classical_score_extends_to_tiles_with_four_remaining(self):
        view = riichi_view(self.pi)
        player = self.pi.players[view.riichi_seat]
        tile = tt("6p")
        self.assertEqual(view.remaining_counts[tile_type_index(tile)], 4)
        with self.assertRaises(ValueError):
            _classical_riichi_danger_score(tile, player, view.remaining_counts)
        capped = list(view.remaining_counts)
        capped[tile_type_index(tile)] = 3
        self.assertEqual(
            classical_wait_score(view, player, tile),
            _classical_riichi_danger_score(tile, player, tuple(capped)).total,
        )

    def test_unknown_feature_set_and_bad_prevalence_are_rejected(self):
        with self.assertRaises(ValueError):
            LogisticWaitModel(weights=(), feature_set="other")
        with self.assertRaises(ValueError):
            PrevalenceWaitModel((0.1,) * 33)
        with self.assertRaises(ValueError):
            PrevalenceWaitModel((1.5,) + (0.1,) * 33)


class WaitBeliefTest(unittest.TestCase):
    def test_level_1_belief_keeps_the_conditional_uniform_marginals(self):
        pi = policy_input({1: [(1, "9p")]})
        model = LogisticWaitModel(weights=(("bias", -1.0),))
        belief = estimate_riichi_wait_belief(pi, model)
        self.assertTrue(belief.has_wait_belief)
        self.assertFalse(belief.has_wait_mechanism_belief)
        prediction = model.predict(pi)
        for tile in TILE_TYPES:
            self.assertAlmostEqual(
                belief.wait_probability(tile), prediction[tile], delta=1 / 8192
            )
        slots = (13, 13, 13, 13)
        slots = tuple(0 if index == 0 else count for index, count in enumerate(slots))
        uniform = estimate_conditional_uniform_hand_belief(pi, slots)
        riichi_hand = uniform.hands[1]  # dealerはSEAT_0、リーチ者はSEAT_1=SOUTH
        self.assertEqual(belief.expected_count_raw, riichi_hand.expected_count_raw)
        self.assertEqual(
            belief,
            replace(riichi_hand, wait_probability_raw=belief.wait_probability_raw),
        )


class InferenceBoundaryTest(unittest.TestCase):
    def test_estimator_module_never_reads_label_facts_or_ground_truth(self):
        tree = ast.parse(inspect.getsource(estimator_module))
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
        self.assertNotIn("lisjong.learning.riichi_deal_in_source", imported)
        self.assertNotIn("lisjong.learning.riichi_wait_evaluation", imported)
        self.assertFalse(any(name.startswith("lisjong_arena") for name in imported))
        self.assertFalse(
            any(
                "exact_wait_ground_truth" in name or "riichi_ron_label" in name
                for name in imported
            )
        )

    def test_predict_takes_only_the_player_safe_input(self):
        for model in (PrevalenceWaitModel, ClassicalScoreWaitModel, LogisticWaitModel):
            self.assertEqual(
                list(inspect.signature(model.predict).parameters),
                ["self", "policy_input"],
            )


def episode_decisions(*, seeds, observers=1, r_discards="9p", hand=None):
    """seed(半荘)ごとに1リーチ。観測者が複数なら、同じリーチを見る行を並べる。"""
    decisions, facts = [], []
    for seed in seeds:
        for observer in range(observers):
            key = DecisionKey(seed=seed, sequence=20 + observer, seat=0)
            decisions.append(make_decision(key=key, r_discards=r_discards))
            facts.append(make_facts(key=key, concealed=hand or source_fixtures.R_HAND))
    return decisions, facts


class WeightTest(unittest.TestCase):
    def test_rows_of_one_riichi_share_a_unit_weight(self):
        decisions, facts = episode_decisions(seeds=(1, 2), observers=3)
        # 2リーチ x 3行: 各行 1/3
        rows = label_waits(decisions, facts)
        self.assertEqual(row_weights(rows), [1 / 3] * 6)
        single = label_waits(*episode_decisions(seeds=(1, 2)))
        self.assertEqual(row_weights(single), [1.0, 1.0])


class EvaluationTest(unittest.TestCase):
    def test_loss_is_averaged_over_tiles_then_rows_then_riichi_then_games(self):
        # 半荘1は1リーチ(3行)、半荘2は1リーチ(1行)。リーチ間の平均なので両方の重みが等しい
        d1, f1 = episode_decisions(seeds=(1,), observers=3)
        d2, f2 = episode_decisions(seeds=(2,), hand=TANKI_5Z)
        rows = label_waits(d1 + d2, f1 + f2)
        model = PrevalenceWaitModel((0.25,) * 34)
        result = evaluate(model, rows)
        # 半荘1の待ちは2種、半荘2は1種
        loss_1 = (-(2 * log(0.25) + 32 * log(0.75))) / 34
        loss_2 = (-(1 * log(0.25) + 33 * log(0.75))) / 34
        self.assertEqual(result["riichi_episodes"], 2)
        self.assertEqual(result["decision_rows"], 4)
        self.assertAlmostEqual(result["log_loss"], (loss_1 + loss_2) / 2)
        self.assertAlmostEqual(result["expected_wait_kinds"], 34 * 0.25)
        self.assertAlmostEqual(result["actual_wait_kinds"], 1.5)
        self.assertEqual(result["per_game"]["1"]["episodes"], 1)
        self.assertAlmostEqual(result["per_game"]["1"]["log_loss_sum"], loss_1)

    def test_auc_excludes_rows_without_both_classes_and_counts_them(self):
        rows = label_waits(*episode_decisions(seeds=(1, 2)))
        result = evaluate(PrevalenceWaitModel((0.25,) * 34), rows)
        self.assertEqual(result["auc"]["decision_rows_without_both_classes"], 0)
        self.assertAlmostEqual(result["auc"]["mean_over_episodes"], 0.5)  # 同点
        informative = LogisticWaitModel(
            weights=(("bias", -3.0), ("support_tanki", 0.0))
        )
        self.assertIsNotNone(evaluate(informative, rows)["auc"]["mean_over_episodes"])

    def test_model_must_predict_every_tile_type(self):
        rows = label_waits(*episode_decisions(seeds=(1,)))

        class Partial:
            def predict(self, policy_input):
                return {TILE_TYPES[0]: 0.5}

        with self.assertRaises(ValueError):
            evaluate(Partial(), rows)

    def test_bootstrap_is_deterministic_paired_and_requires_same_games(self):
        rows = label_waits(*episode_decisions(seeds=(1, 2, 3, 4)))
        worse = evaluate(PrevalenceWaitModel((0.25,) * 34), rows)
        better = evaluate(
            PrevalenceWaitModel(
                tuple(0.9 if t in (tt("1m"), tt("4m")) else 0.01 for t in TILE_TYPES)
            ),
            rows,
        )
        first = bootstrap_difference(better, worse, "log_loss")
        self.assertEqual(first, bootstrap_difference(better, worse, "log_loss"))
        self.assertLess(first["point"], 0.0)
        self.assertLess(first["high_97.5"], 0.0)
        self.assertEqual(first["games"], 4)
        other = evaluate(
            PrevalenceWaitModel((0.25,) * 34),
            label_waits(*episode_decisions(seeds=(1, 2, 3))),
        )
        with self.assertRaises(ValueError):
            bootstrap_difference(better, other, "log_loss")

    def test_pass_rule_needs_both_baselines_with_an_upper_end_below_zero(self):
        rows = label_waits(*episode_decisions(seeds=(1, 2, 3, 4)))
        good = PrevalenceWaitModel(
            tuple(0.9 if t in (tt("1m"), tt("4m")) else 0.01 for t in TILE_TYPES)
        )
        base = PrevalenceWaitModel((0.25,) * 34)
        evaluations = {
            "baseline1_prevalence": evaluate(base, rows),
            "baseline2_classical_platt": evaluate(base, rows),
            "estimator_logistic": evaluate(good, rows),
        }
        self.assertTrue(compare(evaluations)["passed"])
        evaluations["baseline2_classical_platt"] = evaluate(good, rows)
        self.assertFalse(compare(evaluations)["passed"])  # 差が0で区間が0をまたぐ


class FittingTest(unittest.TestCase):
    def setUp(self):
        decisions = []
        facts = []
        for seed, hand in ((1, source_fixtures.R_HAND), (2, TANKI_5Z)) * 3:
            d, f = episode_decisions(seeds=(seed,), hand=hand)
            decisions += d
            facts += f
        # 同じ(seed, 宣言)のリーチが重ならないよう、seedを振り直す
        self.rows = []
        for index, (decision, fact) in enumerate(zip(decisions, facts)):
            key = DecisionKey(seed=100 + index, sequence=20, seat=0)
            self.rows += label_waits(
                [replace(decision, key=key)], [replace(fact, key=key)]
            )

    def test_prevalence_is_weighted_and_smoothed(self):
        model = fit_prevalence(self.rows)
        index = tile_type_index
        self.assertAlmostEqual(model.probabilities[index(tt("1m"))], 3.5 / 7.0)
        self.assertAlmostEqual(model.probabilities[index(tt("5z"))], 3.5 / 7.0)
        self.assertAlmostEqual(model.probabilities[index(tt("9s"))], 0.5 / 7.0)

    def test_estimator_beats_the_baselines_on_separable_training_data(self):
        evaluations = {
            name: evaluate(model, self.rows)["log_loss"]
            for name, model in (
                ("prevalence", fit_prevalence(self.rows)),
                ("classical", fit_classical(self.rows)),
                ("estimator", fit_estimator(self.rows, 0.01)),
            )
        }
        self.assertLess(evaluations["estimator"], evaluations["prevalence"])
        self.assertLess(evaluations["estimator"], evaluations["classical"])

    def test_fitted_models_reproduce_the_training_base_rate(self):
        # 切片は正則化しないので、最適解では予測の合計が正例の合計と一致する
        for model in (
            fit_classical(self.rows),
            fit_estimator(self.rows, 1.0),
            fit_estimator(self.rows, 100.0),
        ):
            result = evaluate(model, self.rows)
            self.assertAlmostEqual(
                result["expected_wait_kinds"], result["actual_wait_kinds"], places=3
            )

    def test_fixed_search_space(self):
        self.assertEqual(L2_GRID, (0.01, 0.1, 1.0, 10.0, 100.0))
        self.assertEqual(evaluation_module.BOOTSTRAP_RESAMPLES, 2000)


class PhasesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = self.root / "s1"
        self.source.mkdir()
        decisions, facts = [], []
        for seed, hand in (
            (931000, source_fixtures.R_HAND),
            (931001, TANKI_5Z),
            (931002, source_fixtures.R_HAND),
            (931003, TANKI_5Z),
        ):
            d, f = episode_decisions(seeds=(seed,), hand=hand)
            decisions += d
            facts += f
        write_source(
            self.source,
            decisions,
            facts,
            splits={
                "train": [931000, 931001],
                "valid": [931002],
                "test": [931003],
            },
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _new_test_source(self, seeds, name="s2"):
        path = self.root / name
        path.mkdir()
        decisions, facts = episode_decisions(seeds=seeds)
        write_source(
            path,
            decisions,
            facts,
            splits={"train": [], "valid": [], "test": list(seeds)},
        )
        return path

    def test_select_then_test_on_new_seeds_once(self):
        selection = self.root / "selection.json"
        self.assertEqual(main(["select", str(self.source), str(selection)]), 0)
        chosen = json.loads(selection.read_text(encoding="utf-8"))
        self.assertEqual(chosen["fitted_on"]["baselines"], "train")
        self.assertEqual(chosen["split_sizes"], {"train": 2, "valid": 1, "test": 1})
        self.assertNotIn("test", chosen)
        self.assertEqual(
            sorted(seed for seeds in chosen["used_seeds"].values() for seed in seeds),
            [931000, 931001, 931002, 931003],
        )
        new_source = self._new_test_source([940000, 940001])
        result = self.root / "result.json"
        self.assertEqual(
            main(["test", str(new_source), str(selection), str(result)]), 0
        )
        document = json.loads(result.read_text(encoding="utf-8"))
        self.assertEqual(document["test_seeds"], [940000, 940001])
        self.assertEqual(document["test"]["estimator_logistic"]["riichi_episodes"], 2)
        self.assertIn("passed", document["test_comparison"])
        with self.assertRaises(FileExistsError):
            main(["test", str(new_source), str(selection), str(result)])
        with self.assertRaises(FileExistsError):
            main(["select", str(self.source), str(selection)])

    def test_test_refuses_the_selection_source_and_overlapping_seeds(self):
        selection = self.root / "selection.json"
        main(["select", str(self.source), str(selection)])
        with self.assertRaises(ValueError):
            main(["test", str(self.source), str(selection), str(self.root / "a.json")])
        overlapping = self._new_test_source([931003], "s3")  # S1のtest seed
        with self.assertRaises(ValueError):
            main(["test", str(overlapping), str(selection), str(self.root / "b.json")])
        self.assertFalse((self.root / "b.json").exists())

    def test_test_refuses_a_non_selection_document(self):
        other = self.root / "other.json"
        other.write_text(json.dumps({"schema": "x"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            main(
                [
                    "test",
                    str(self._new_test_source([940000], "s4")),
                    str(other),
                    str(self.root / "c.json"),
                ]
            )


if __name__ == "__main__":
    unittest.main()
