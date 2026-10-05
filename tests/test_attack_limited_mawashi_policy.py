"""#249の攻撃制限付き回し打ち候補（古典score版・待ち推定版）の発動境界と選択を固定する。"""

import ast
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_placement_aware_speed_call_policy import (
    ALL_LAST_TOP,
    _discards,
    _input,
    _player,
    _pon_meld,
    _tile,
)

import lisjong.learning.riichi_wait_mawashi_policy as wait_module
import lisjong.policies.attack_limited_mawashi_placement_aware_speed_call as mawashi
from lisjong.learning.riichi_deal_in_estimator import riichi_view
from lisjong.learning.riichi_wait_estimator import (
    CLIP_EPSILON,
    FEATURE_SET,
    LogisticWaitModel,
)
from lisjong.learning.riichi_wait_mawashi_policy import (
    SELECTED_WAIT_MODEL_SHA256,
    RiichiWaitAttackLimitedMawashiPolicy,
    RiichiWaitModelError,
    load_selected_wait_model,
)
from lisjong.policies import (
    ClassicalAttackLimitedMawashiPolicy,
    PlacementAwareSpeedCallPolicy,
)
from lisjong.policies.attack_limited_mawashi_placement_aware_speed_call import (
    AttackLimitedMawashiAnalysis,
    mawashi_target_riichi_seat,
    ron_safe_tile_types,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.riichi import RiichiState

# 34m 678p 68p 234s 55s + 浮き牌9s・1m: 1向聴。1mと9sのどちらを切っても1向聴・受け入れ11枚。
HAND = "34m678p68p234s55s9s1m"
# 上の雀頭5sを5z対子に替えた形。5z切りは向聴が戻る。
HAND_WITH_HONOR_PAIR = "34m678p68p234s55z9s1m"


def _decision(context) -> DecisionContext:
    return DecisionContext(input=context, legal_actions=_discards(context))


def _wait_policy(**weights: float) -> RiichiWaitAttackLimitedMawashiPolicy:
    return RiichiWaitAttackLimitedMawashiPolicy(
        LogisticWaitModel(weights=tuple(weights.items()))
    )


def _policies():
    return (ClassicalAttackLimitedMawashiPolicy(), _wait_policy(class_rank_1=3.0))


class GateTest(unittest.TestCase):
    def test_target_is_single_riichi_closed_one_shanten_without_genbutsu(self):
        context = _input(HAND, riichi_discards="3z4z")
        self.assertEqual(mawashi_target_riichi_seat(context, _discards(context)), 1)

    def test_non_target_decisions_return_the_champion_decision_unchanged(self):
        two_riichi = _input(HAND, riichi_discards="3z4z")
        players = list(two_riichi.players)
        players[2] = _player(riichi=RiichiState.ACCEPTED, discards="2z")
        two_riichi = replace(two_riichi, players=tuple(players))
        cases = {
            "no riichi": _input(HAND),
            "common genbutsu": _input(HAND, riichi_discards="9s3z"),
            "two riichi": two_riichi,
            "open hand": _input(
                "34m678p68p9s1m55s", own_melds=(_pon_meld("3s"),), riichi_discards="3z"
            ),
            "all last top": _input(HAND, riichi_discards="3z4z", **ALL_LAST_TOP),
            "tenpai": _input("123m123p123s789s11z", riichi_discards="3z4z"),
            "two shanten": _input("345m56679s333517z", riichi_discards="9p"),
        }
        for name, context in cases.items():
            actions = _discards(context)
            self.assertIsNone(mawashi_target_riichi_seat(context, actions), name)
            expected = PolicyDecision(action=actions[0])
            for policy in _policies():
                with (
                    self.subTest(case=name, policy=type(policy).__name__),
                    patch.object(
                        PlacementAwareSpeedCallPolicy,
                        "_decide_discard",
                        return_value=expected,
                    ) as parent,
                ):
                    self.assertIs(policy._decide_discard(context, actions), expected)
                    parent.assert_called_once_with(context, actions)

    def test_non_target_choices_match_the_champion(self):
        contexts = (
            _input(HAND),
            _input(HAND, riichi_discards="9s3z"),
            _input(HAND, riichi_discards="3z4z", **ALL_LAST_TOP),
            _input("345m56679s333517z", riichi_discards="9p"),
        )
        for context in contexts:
            decision = _decision(context)
            expected = PlacementAwareSpeedCallPolicy().choose_action(decision)
            for policy in _policies():
                with self.subTest(context=context, policy=type(policy).__name__):
                    self.assertEqual(policy.choose_action(decision), expected)


class ClassicalSelectionTest(unittest.TestCase):
    def _run(self, context):
        decision = _decision(context)
        c0 = PlacementAwareSpeedCallPolicy().choose_action(decision)
        result = ClassicalAttackLimitedMawashiPolicy().choose_action_with_analysis(
            decision
        )
        self.assertIsInstance(result.analysis, AttackLimitedMawashiAnalysis)
        self.assertEqual(result.analysis.c0_action, c0)
        self.assertEqual(result.analysis.selected_action, result.action)
        return c0, result

    def test_switches_to_a_strictly_safer_discard_within_the_attack_limit(self):
        # 6sが河にあるので9sは両面待ちに当たらず、1mより古典scoreが小さい
        c0, result = self._run(_input(HAND, riichi_discards="3z4z6s"))
        self.assertEqual(c0.tile, _tile("1m"))
        self.assertEqual(result.action.tile, _tile("9s"))

    def test_equal_danger_keeps_the_champion_discard(self):
        c0, result = self._run(_input(HAND, riichi_discards="3z4z"))
        self.assertEqual(result.action, c0)

    def test_safer_discard_that_loses_shanten_is_outside_the_limit(self):
        c0, result = self._run(_input(HAND_WITH_HONOR_PAIR, riichi_discards="3z4z"))
        by_tile = {
            evaluation.action.tile: evaluation
            for evaluation in result.analysis.candidate_evaluations
        }
        honor = by_tile[_tile("5z")]
        self.assertLess(honor.danger, by_tile[c0.tile].danger)
        self.assertFalse(honor.within_attack_limit)
        self.assertTrue(by_tile[c0.tile].within_attack_limit)
        self.assertEqual(result.action, c0)

    def test_attack_limit_compares_ukeire_mass_and_value_with_the_champion(self):
        context = _input(HAND, riichi_discards="3z4z")
        actions = _discards(context)
        c0 = next(action for action in actions if action.tile == _tile("1m"))
        other = next(action for action in actions if action.tile == _tile("9s"))
        dangers = {action.tile.tile_type: 1.0 for action in actions}

        def within(**changed) -> bool:
            base = {"ukeire": 12, "mass": 400, "value": 1}
            rows = {c0: base, other: base | changed}
            with (
                patch.object(
                    mawashi,
                    "ukeire_count",
                    side_effect=lambda hand, *_: rows[
                        c0 if _tile("1m") not in hand else other
                    ]["ukeire"],
                ),
                patch.object(
                    mawashi,
                    "_retained_real_value",
                    side_effect=lambda hand, _: rows[
                        c0 if _tile("1m") not in hand else other
                    ]["value"],
                ),
                patch.object(
                    mawashi,
                    "_evaluate_completion_masses",
                    side_effect=lambda _, candidates, *__: tuple(
                        SimpleNamespace(action=a, completion_mass=rows[a]["mass"])
                        for a in candidates
                    ),
                ),
            ):
                evaluations = mawashi._evaluate_candidates(
                    context, (c0, other), c0, dangers
                )
            return evaluations[1].within_attack_limit

        self.assertTrue(within())
        self.assertTrue(within(ukeire=9, mass=300))
        self.assertFalse(within(ukeire=8))
        self.assertFalse(within(mass=299))
        self.assertFalse(within(value=0))
        self.assertTrue(within(value=2))

    def test_tile_passed_after_the_riichi_discard_is_treated_as_ron_safe(self):
        context = _input(HAND, riichi_discards="3z4z")
        players = list(context.players)
        players[2] = replace(
            players[2],
            discards=(
                Discard(tile=_tile("9s"), tsumogiri=False, order=9, called_by=None),
            ),
        )
        context = replace(context, players=tuple(players))
        safe = ron_safe_tile_types(context, 1)
        self.assertIn(_tile("9s").tile_type, safe)
        self.assertEqual(safe, riichi_view(context).structurally_safe)
        _, result = self._run(context)
        self.assertEqual(result.action.tile, _tile("9s"))

    def test_tile_discarded_before_the_riichi_discard_is_not_ron_safe(self):
        context = _input(HAND, riichi_discards="3z4z")
        players = list(context.players)
        players[2] = replace(
            players[2],
            discards=(
                Discard(tile=_tile("9s"), tsumogiri=False, order=0, called_by=None),
            ),
        )
        context = replace(context, players=tuple(players))
        self.assertNotIn(_tile("9s").tile_type, ron_safe_tile_types(context, 1))

    def test_selected_action_keeps_the_legal_action_identity(self):
        context = _input(HAND, riichi_discards="3z4z6s")
        decision = _decision(context)
        policy = ClassicalAttackLimitedMawashiPolicy()
        selected = policy.choose_action(decision)
        self.assertTrue(any(selected is action for action in decision.legal_actions))
        self.assertEqual(policy.choose_action_with_analysis(decision).action, selected)


class WaitSelectionTest(unittest.TestCase):
    def test_wait_estimate_ranks_the_candidates(self):
        decision = _decision(_input(HAND, riichi_discards="3z4z"))
        self.assertEqual(
            _wait_policy(class_rank_1=3.0).choose_action(decision).tile, _tile("9s")
        )
        self.assertEqual(
            _wait_policy(class_rank_9=3.0).choose_action(decision).tile, _tile("1m")
        )

    def test_raw_wait_probability_is_used_unless_the_tile_is_ron_safe(self):
        # スジ（6sが河）でも生の値は上書きしない。9sの生の値が大きければC0の1mを維持する
        policy = _wait_policy(class_rank_9=3.0)
        context = _input(HAND, riichi_discards="3z4z6s")
        result = policy.choose_action_with_analysis(_decision(context))
        raw = policy._model.predict(context)
        by_tile = {
            evaluation.action.tile.tile_type: evaluation.danger
            for evaluation in result.analysis.candidate_evaluations
        }
        self.assertEqual(by_tile, {tile: raw[tile] for tile in by_tile})
        self.assertEqual(result.action.tile, _tile("1m"))

    def test_model_receives_only_the_policy_input(self):
        policy = _wait_policy(class_rank_1=3.0)
        context = _input(HAND, riichi_discards="3z4z")
        with patch.object(
            LogisticWaitModel,
            "predict",
            autospec=True,
            return_value={action.tile.tile_type: 0.5 for action in _discards(context)},
        ) as predict:
            policy.choose_action(_decision(context))
        predict.assert_called_once_with(policy._model, context)

    def test_requires_a_logistic_wait_model(self):
        with self.assertRaises(TypeError):
            RiichiWaitAttackLimitedMawashiPolicy(object())

    def test_inference_modules_do_not_import_label_paths(self):
        for module in (wait_module, mawashi):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
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
            for forbidden in (
                "riichi_deal_in_source",
                "riichi_wait_evaluation",
                "exact_wait_ground_truth",
                "riichi_ron_label",
                "lisjong_arena",
            ):
                self.assertFalse(
                    any(forbidden in name for name in imported), (module, forbidden)
                )
            if module is mawashi:
                self.assertFalse(
                    any(name.startswith("lisjong.learning") for name in imported)
                )


class SelectedModelLoadTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "selection.json"

    def _write(self, **changed) -> str:
        document = {
            "schema": "lisjong-riichi-wait-selection-v1",
            "feature_set": FEATURE_SET,
            "clip_epsilon": CLIP_EPSILON,
            "models": {
                "estimator_logistic": {
                    "feature_set": FEATURE_SET,
                    "weights": {"bias": -2.0, "class_rank_1": 3.0},
                }
            },
        } | changed
        content = json.dumps(document).encode("utf-8")
        self.path.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    def test_pinned_hash_is_the_formal_selection(self):
        self.assertEqual(
            SELECTED_WAIT_MODEL_SHA256,
            "14475264d7fe4137a9ac8a23ee1d27b420bff2f4434c6e93ca72ecfcf24ccc38",
        )

    def test_loads_the_estimator_when_the_hash_matches(self):
        digest = self._write()
        model = load_selected_wait_model(self.path, expected_sha256=digest)
        self.assertEqual(dict(model.weights), {"bias": -2.0, "class_rank_1": 3.0})
        policy = RiichiWaitAttackLimitedMawashiPolicy.from_selection(
            self.path, expected_sha256=digest
        )
        self.assertEqual(policy._model, model)

    def test_refuses_a_file_whose_hash_differs_from_the_pin(self):
        self._write()
        with self.assertRaises(RiichiWaitModelError):
            load_selected_wait_model(self.path)
        with self.assertRaises(RiichiWaitModelError):
            RiichiWaitAttackLimitedMawashiPolicy.from_selection(self.path)

    def test_refuses_other_schema_feature_set_clip_or_malformed_documents(self):
        for changed in (
            {"schema": "other"},
            {"feature_set": "riichi-wait-features-v0"},
            {"clip_epsilon": 1e-3},
            {"models": {}},
            {"models": {"estimator_logistic": {"feature_set": "other", "weights": {}}}},
        ):
            with self.subTest(changed=changed):
                digest = self._write(**changed)
                with self.assertRaises(RiichiWaitModelError):
                    load_selected_wait_model(self.path, expected_sha256=digest)


if __name__ == "__main__":
    unittest.main()
