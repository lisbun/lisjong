"""#237 S1 放銃確率推定器・ベースライン・比較手順のtest。"""

import ast
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import test_learning_riichi_deal_in_source as source_fixtures
from test_learning_riichi_deal_in_source import (
    KEY,
    make_decision,
    make_facts,
    tiles,
    tt,
    write_source,
)

from lisjong.learning import riichi_deal_in_estimator as estimator_module
from lisjong.learning.riichi_deal_in_estimator import (
    ClassicalScoreModel,
    ConstantModel,
    LogisticModel,
    candidate_features,
    riichi_view,
)
from lisjong.learning.riichi_deal_in_evaluation import (
    bootstrap_difference,
    evaluate,
    fit_constant,
    fit_logistic,
    main,
)
from lisjong.learning.riichi_deal_in_source import DecisionKey, label_decisions
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.wind import Wind

TANKI_5Z = tiles("123m456p789s111z5z")  # 5z単騎


def policy_input(rivers, *, riichi=(1,), concealed="14m5z"):
    """riversは席ごとの[(order, 牌)]。orderは局内で全席共通の通し番号。"""
    players = tuple(
        PlayerPublicState(
            score=25000,
            discards=tuple(
                Discard(
                    tile=tiles(spec)[0], tsumogiri=False, order=order, called_by=None
                )
                for order, spec in rivers.get(seat, ())
            ),
            melds=(),
            riichi=RiichiState.ACCEPTED if seat in riichi else RiichiState.NONE,
        )
        for seat in range(4)
    )
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=1,
            dora_indicators=(),
            live_wall_tiles_remaining=40,
        ),
        players=players,
        own_hand=OwnHandState(concealed_tiles=tiles(concealed), drawn_tile=None),
    )


class StructuralSafetyTest(unittest.TestCase):
    def test_genbutsu_and_tiles_passed_after_the_last_riichi_discard_are_zero(self):
        pi = policy_input(
            {
                1: [(1, "9p"), (5, "2s")],
                2: [(2, "1m"), (6, "4m")],  # 1mは最後のリーチ者打牌(5)より前
                3: [(3, "5z")],
            }
        )
        view = riichi_view(pi)
        self.assertEqual(view.genbutsu, frozenset({tt("9p"), tt("2s")}))
        self.assertEqual(view.passed_since_last_riichi_discard, frozenset({tt("4m")}))
        model = LogisticModel(weights=(("bias", 0.0),))
        predictions = model.predict(pi, (tt("1m"), tt("4m"), tt("5z")))
        self.assertEqual(predictions[tt("4m")], 0.0)
        self.assertEqual(predictions[tt("1m")], 0.5)
        self.assertEqual(predictions[tt("5z")], 0.5)
        # ベースライン1は定義どおり現物だけを0にする
        constant = ConstantModel(probability=0.1).predict(pi, (tt("4m"), tt("2s")))
        self.assertEqual(constant, {tt("4m"): 0.1, tt("2s"): 0.0})

    def test_out_of_scope_inputs_fail_closed(self):
        for kwargs in ({"riichi": ()}, {"riichi": (1, 2)}, {"riichi": (0, 1)}):
            with self.assertRaises(ValueError):
                riichi_view(policy_input({1: [(1, "9p")], 2: [(2, "1p")]}, **kwargs))

    def test_unknown_feature_set_is_rejected(self):
        with self.assertRaises(ValueError):
            LogisticModel(weights=(), feature_set="other")


class InferenceBoundaryTest(unittest.TestCase):
    """推論はplayer-safeな`PolicyInput`だけを入力にする。"""

    def test_estimator_module_never_reads_label_facts(self):
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
        self.assertFalse(any(name.startswith("lisjong_arena") for name in imported))
        self.assertFalse(
            any(
                "exact_wait_ground_truth" in name or "riichi_ron_label" in name
                for name in imported
            )
        )

    def test_predict_takes_only_the_player_safe_input(self):
        for model in (ConstantModel, ClassicalScoreModel, LogisticModel):
            parameters = list(inspect.signature(model.predict).parameters)
            self.assertEqual(parameters, ["self", "policy_input", "candidates"])

    def test_label_facts_do_not_change_the_prediction(self):
        decision = make_decision()
        (a,) = label_decisions([decision], [make_facts()])
        (b,) = label_decisions(
            [decision], [make_facts(concealed=TANKI_5Z, ron_offered=True)]
        )
        self.assertNotEqual(a.candidates, b.candidates)
        model = LogisticModel(weights=(("bias", -1.0), ("dora", 2.0)))
        candidates = tuple(c.tile_type for c in a.candidates)
        self.assertEqual(
            model.predict(a.decision.policy_input, candidates),
            model.predict(b.decision.policy_input, candidates),
        )

    def test_features_are_named_values(self):
        pi = policy_input({1: [(1, "1m"), (5, "7m")]}, concealed="14m4z5z")
        view = riichi_view(pi)
        features = dict(candidate_features(pi, view, tt("4m")))
        self.assertIn("number_456_suji_full", features)
        self.assertEqual(features["bias"], 1.0)
        # 4m: 1m・7mが河にあり両面は両側ともなし。same-tile 3 + kanchan 3 = 6
        self.assertAlmostEqual(features["classical_score"], 0.6)
        self.assertIn(
            "honor_guest_remaining_3", dict(candidate_features(pi, view, tt("4z")))
        )
        self.assertIn(
            "honor_yakuhai_remaining_3", dict(candidate_features(pi, view, tt("5z")))
        )


class FittingTest(unittest.TestCase):
    def test_logistic_recovers_aggregated_rates(self):
        # bias=logit(0.2)、特徴1あり=logit(0.5)
        weights = fit_logistic(
            {((0, 1.0),): (100, 20), ((0, 1.0), (1, 1.0)): (50, 25)}, 2, 0.0
        )
        self.assertAlmostEqual(weights[0], -1.3862943611, places=6)
        self.assertAlmostEqual(weights[0] + weights[1], 0.0, places=6)

    def test_constant_baseline_excludes_genbutsu(self):
        labelled = label_decisions([make_decision(r_discards="9p5z")], [make_facts()])
        # 1m・4mが待ち、5zは現物
        self.assertEqual(fit_constant(labelled).probability, 1.0)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.labelled = label_decisions([make_decision()], [make_facts()])

    def test_ranking_and_reliability(self):
        # 1m・4mが待ち、5zは安全
        class Fixed:
            def __init__(self, values):
                self.values = values

            def predict(self, policy_input, candidates):
                return {tile: self.values[tile] for tile in candidates}

        perfect = evaluate(
            Fixed({tt("1m"): 0.9, tt("4m"): 0.8, tt("5z"): 0.1}), self.labelled
        )
        self.assertEqual(perfect["ranking"]["within_decision_auc"], 1.0)
        self.assertEqual(perfect["ranking"]["lowest_risk_pick_ron_rate"], 0.0)
        tied = evaluate(
            Fixed(dict.fromkeys(map(tt, ("1m", "4m", "5z")), 0.3)), self.labelled
        )
        self.assertEqual(tied["ranking"]["within_decision_auc"], 0.5)
        self.assertAlmostEqual(tied["ranking"]["lowest_risk_pick_ron_rate"], 2 / 3)
        self.assertEqual(sum(b["candidates"] for b in tied["reliability"]), 3)
        with self.assertRaises(ValueError):
            evaluate(
                Fixed({tt("1m"): 0.0, tt("4m"): 0.5, tt("5z"): 0.5}), self.labelled
            )

    def test_bootstrap_is_by_game_and_deterministic(self):
        a = {
            "per_game": {
                "1": {"candidates": 2, "log_loss_sum": 1.0},
                "2": {"candidates": 2, "log_loss_sum": 3.0},
            }
        }
        b = {
            "per_game": {
                "1": {"candidates": 2, "log_loss_sum": 1.0},
                "2": {"candidates": 2, "log_loss_sum": 1.0},
            }
        }
        first = bootstrap_difference(a, b, "log_loss")
        self.assertEqual(first, bootstrap_difference(a, b, "log_loss"))
        self.assertAlmostEqual(first["point"], 0.5)
        self.assertEqual(first["games"], 2)
        self.assertLessEqual(first["low_2.5"], first["point"])
        with self.assertRaises(ValueError):
            bootstrap_difference(a, {"per_game": {"1": b["per_game"]["1"]}}, "log_loss")


class PhasesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        decisions, facts = [], []
        for seed in (931000, 931001, 931002):
            for sequence, hand in ((20, source_fixtures.R_HAND), (21, TANKI_5Z)):
                key = DecisionKey(seed=seed, sequence=sequence, seat=KEY.seat)
                decisions.append(make_decision(key=key))
                facts.append(
                    make_facts(key=key, concealed=hand, ron_offered=hand is TANKI_5Z)
                )
        write_source(
            self.source,
            decisions,
            facts,
            splits={"train": [931000], "valid": [931001], "test": [931002]},
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_select_then_test_once(self):
        selection = self.root / "selection.json"
        result = self.root / "result.json"
        self.assertEqual(main(["select", str(self.source), str(selection)]), 0)
        chosen = json.loads(selection.read_text(encoding="utf-8"))
        self.assertEqual(chosen["split_sizes"], {"train": 2, "valid": 2, "test": 2})
        self.assertEqual(chosen["fitted_on"]["baselines"], "valid")
        self.assertNotIn("test", chosen)
        self.assertEqual(
            main(["test", str(self.source), str(selection), str(result)]), 0
        )
        document = json.loads(result.read_text(encoding="utf-8"))
        self.assertEqual(document["test"]["estimator_logistic"]["decisions"], 2)
        with self.assertRaises(FileExistsError):
            main(["test", str(self.source), str(selection), str(result)])
        with self.assertRaises(FileExistsError):
            main(["select", str(self.source), str(selection)])

    def test_selection_must_match_the_source(self):
        selection = self.root / "selection.json"
        main(["select", str(self.source), str(selection)])
        other = self.root / "other"
        other.mkdir()
        write_source(
            other,
            [make_decision()],
            [make_facts()],
            splits={"train": [], "valid": [], "test": [931000]},
        )
        with self.assertRaises(ValueError):
            main(["test", str(other), str(selection), str(self.root / "r.json")])


if __name__ == "__main__":
    unittest.main()
