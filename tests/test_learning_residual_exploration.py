"""Issue #193 constant-zero baseline runtimeとfocal residual exploration selectorのtest。"""

import ast
import inspect
import subprocess
import sys
import unittest

import candidate_fixtures as cf

from lisjong.learning import (
    CONSTANT_RESIDUAL_RUNTIME_IDENTITY,
    CONSTANT_RESIDUAL_SCORER_IDENTITY,
    RESIDUAL_EXPLORATION_BEHAVIOR_IDENTITY,
    RESIDUAL_EXPLORATION_RUNTIME_IDENTITY,
    SEMANTIC_ENVELOPE_IDENTITY,
    ConstantResidualRuntime,
    LearnedPolicyError,
    SemanticEnvelopeOffensePolicy,
    build_scorer_candidates,
    select_residual_exploration,
    semantic_envelope_runtime_identity,
    semantic_envelope_survivors,
)
from lisjong.learning import residual_baseline as baseline_module
from lisjong.learning import residual_exploration as exploration_module
from lisjong.learning._canonical import value_digest
from lisjong.learning._o0 import O0DecisionKind
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import TileCategory

SIX_SURVIVOR_HAND = "11m456p789s123z567z"
"""孤立字牌6枚がS1 / S2 / S3で同値になり、survivorが6件の打牌decision。"""

SINGLE_SURVIVOR_HAND = "123m456p789s11z23s9m"
"""打牌後聴牌でsurvivorが1件の打牌decision。"""

ZERO_TOKEN = "0" * 64
GOLDEN_TOKEN = "5054494f38b691797c94075c81a0f30b6ba94ad205f19eaedb6b0fc1dc6508cf"
"""sha256(b"lisjong-193-golden")。uint256 % 6 == 3、% 2 == 1。"""

_ORACLE_SPECS = (
    cf.NEAR_HAND,
    cf.SECOND_NEAR_HAND,
    cf.TENPAI_REACHABLE_HAND,
    SINGLE_SURVIVOR_HAND,
    SIX_SURVIVOR_HAND,
    "8m0p8p9p8p2s5m3z2s4z9s8s8m9p",
)


def _guard_decisions():
    return (
        (cf.tsumo_decision(), O0DecisionKind.WIN),
        (cf.ron_response_decision(), O0DecisionKind.WIN),
        (cf.riichi_decision(), O0DecisionKind.RIICHI),
        (cf.response_decision(), O0DecisionKind.RESPONSE),
    )


def _is_legal_object(action, decision) -> bool:
    return any(action is legal for legal in decision.legal_actions)


class ConstantResidualRuntimeTests(unittest.TestCase):
    def test_identity_is_the_frozen_digest(self) -> None:
        self.assertEqual(
            CONSTANT_RESIDUAL_SCORER_IDENTITY,
            "lisjong-offense-l0.3-constant-zero-residual-scorer-v1",
        )
        self.assertEqual(
            CONSTANT_RESIDUAL_RUNTIME_IDENTITY,
            value_digest(
                {
                    "residual_scorer": CONSTANT_RESIDUAL_SCORER_IDENTITY,
                    "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
                }
            ),
        )
        self.assertEqual(
            ConstantResidualRuntime().identity, CONSTANT_RESIDUAL_RUNTIME_IDENTITY
        )

    def test_identity_does_not_collide_with_artifact_backed_runtimes(self) -> None:
        for artifact_identity in (
            CONSTANT_RESIDUAL_SCORER_IDENTITY,
            CONSTANT_RESIDUAL_RUNTIME_IDENTITY,
            "a" * 64,
        ):
            with self.subTest(artifact_identity=artifact_identity):
                self.assertNotEqual(
                    CONSTANT_RESIDUAL_RUNTIME_IDENTITY,
                    semantic_envelope_runtime_identity(artifact_identity),
                )
        self.assertNotEqual(
            CONSTANT_RESIDUAL_RUNTIME_IDENTITY, RESIDUAL_EXPLORATION_RUNTIME_IDENTITY
        )

    def test_scores_are_zero_for_the_full_candidate_tuple(self) -> None:
        candidates = tuple((float(index),) * 3 for index in range(5))
        scores = ConstantResidualRuntime().score((1.0, 2.0), candidates)

        self.assertEqual(scores, (0.0,) * 5)
        self.assertTrue(all(type(score) is float for score in scores))

    def test_factory_returns_the_existing_envelope_policy(self) -> None:
        runtime = ConstantResidualRuntime()
        policy = runtime()

        self.assertIs(type(policy), SemanticEnvelopeOffensePolicy)
        self.assertIs(type(runtime.create_policy()), SemanticEnvelopeOffensePolicy)
        self.assertIs(policy.runtime, runtime)
        self.assertIsNot(runtime(), runtime())

    def test_actions_match_two_step_ukeire(self) -> None:
        policy = ConstantResidualRuntime()()
        teacher = TwoStepUkeirePolicy()
        decisions = [cf.discard_decision(spec) for spec in _ORACLE_SPECS]
        decisions += list(cf.mixed_decisions())
        for index, decision in enumerate(decisions):
            with self.subTest(index=index):
                self.assertIs(
                    policy.choose_action(decision), teacher.choose_action(decision)
                )

    def test_runs_without_the_ml_runtime(self) -> None:
        body = (
            "import sys\n"
            "class _Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('blocked')\n"
            "sys.meta_path.insert(0, _Blocker())\n"
            f"sys.path.insert(0, {str(cf.__file__).rsplit('candidate_fixtures', 1)[0]!r})\n"
            "import candidate_fixtures as cf\n"
            "from lisjong.learning import ConstantResidualRuntime\n"
            "from lisjong.learning import select_residual_exploration\n"
            "decision = cf.discard_decision(cf.NEAR_HAND)\n"
            "ConstantResidualRuntime()().choose_action(decision)\n"
            "select_residual_exploration(decision, '0' * 64)\n"
            "assert 'torch' not in sys.modules\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", body], capture_output=True, text=True, check=False
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


class ResidualExplorationSelectorTests(unittest.TestCase):
    def test_identities(self) -> None:
        self.assertEqual(
            RESIDUAL_EXPLORATION_BEHAVIOR_IDENTITY,
            "lisjong-offense-l0.3-focal-uniform-residual-exploration-v1",
        )
        self.assertEqual(
            RESIDUAL_EXPLORATION_RUNTIME_IDENTITY,
            value_digest(
                {
                    "bucket_rule": exploration_module.RESIDUAL_EXPLORATION_BUCKET_RULE,
                    "exploration_behavior": RESIDUAL_EXPLORATION_BEHAVIOR_IDENTITY,
                    "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
                }
            ),
        )

    def test_guard_decisions_match_the_baseline_policy(self) -> None:
        baseline = ConstantResidualRuntime()()
        for decision, kind in _guard_decisions():
            for token in (ZERO_TOKEN, GOLDEN_TOKEN, "f" * 64):
                result = select_residual_exploration(decision, token)
                with self.subTest(kind=kind, token=token):
                    self.assertIs(result.kind, kind)
                    self.assertIs(result.action, baseline.choose_action(decision))
                    self.assertIsNone(result.candidates)
                    self.assertIsNone(result.survivors)
                    self.assertIsNone(result.survivor_actions)
                    self.assertIsNone(result.selected_candidate_index)
                    self.assertIsNone(result.bucket)

    def test_single_survivor_matches_the_baseline_policy(self) -> None:
        decision = cf.discard_decision(SINGLE_SURVIVOR_HAND)
        expected = ConstantResidualRuntime()().choose_action(decision)
        for token in (ZERO_TOKEN, GOLDEN_TOKEN, "f" * 64):
            result = select_residual_exploration(decision, token)
            with self.subTest(token=token):
                self.assertIs(result.kind, O0DecisionKind.DISCARD)
                self.assertIs(result.action, expected)
                self.assertEqual(len(result.survivors), 1)
                self.assertEqual(result.selected_candidate_index, result.survivors[0])
                self.assertIsNone(result.bucket)

    def test_golden_vectors(self) -> None:
        decision = cf.discard_decision(SIX_SURVIVOR_HAND)
        cases = (
            (ZERO_TOKEN, 0, (1, False)),
            ("0" * 63 + "5", 5, (7, True)),
            ("0" * 62 + "0a", 4, (6, False)),
            ("f" * 64, 3, (5, False)),
            (GOLDEN_TOKEN, 3, (5, False)),
        )
        for token, bucket, (rank, tsumogiri) in cases:
            result = select_residual_exploration(decision, token)
            with self.subTest(token=token):
                self.assertEqual(len(result.survivors), 6)
                self.assertEqual(result.bucket, bucket)
                self.assertEqual(
                    result.selected_candidate_index, result.survivors[bucket]
                )
                self.assertIs(result.action.tile.tile_type.category, TileCategory.HONOR)
                self.assertEqual(result.action.tile.tile_type.rank, rank)
                self.assertEqual(result.action.tsumogiri, tsumogiri)

        near = cf.discard_decision(cf.NEAR_HAND)
        for token, bucket in ((ZERO_TOKEN, 0), (GOLDEN_TOKEN, 1), ("f" * 64, 1)):
            with self.subTest(hand="near", token=token):
                self.assertEqual(
                    select_residual_exploration(near, token).bucket, bucket
                )

    def test_selection_is_inside_the_canonical_survivors(self) -> None:
        tokens = (ZERO_TOKEN, "0" * 63 + "1", GOLDEN_TOKEN, "f" * 64)
        for spec in _ORACLE_SPECS:
            decision = cf.discard_decision(spec)
            candidates = build_scorer_candidates(decision)
            survivors = semantic_envelope_survivors(candidates)
            for token in tokens:
                result = select_residual_exploration(decision, token)
                with self.subTest(spec=spec, token=token):
                    self.assertEqual(result.candidates, candidates)
                    self.assertEqual(result.survivors, survivors)
                    self.assertEqual(result.survivors, tuple(sorted(result.survivors)))
                    self.assertIn(result.selected_candidate_index, result.survivors)
                    self.assertIs(
                        result.action,
                        candidates[result.selected_candidate_index].action,
                    )
                    self.assertTrue(_is_legal_object(result.action, decision))
                    self.assertEqual(
                        result.survivor_actions,
                        tuple(candidates[index].action for index in survivors),
                    )

    def test_selection_is_deterministic(self) -> None:
        decision = cf.discard_decision(SIX_SURVIVOR_HAND)
        first = select_residual_exploration(decision, GOLDEN_TOKEN)
        second = select_residual_exploration(decision, GOLDEN_TOKEN)

        self.assertEqual(first, second)
        self.assertIs(first.action, second.action)

    def test_invalid_tokens_fail_closed(self) -> None:
        decision = cf.discard_decision(cf.NEAR_HAND)
        for token in (
            "A" * 64,
            GOLDEN_TOKEN.upper(),
            "0" * 63,
            "0" * 65,
            "g" * 64,
            "",
            1,
            None,
        ):
            with self.subTest(token=token):
                with self.assertRaises(LearnedPolicyError):
                    select_residual_exploration(decision, token)
        with self.assertRaises(LearnedPolicyError):
            select_residual_exploration(cf.tsumo_decision(), "0" * 63)

    def test_non_decision_context_input_rejected(self) -> None:
        with self.assertRaises(LearnedPolicyError):
            select_residual_exploration(object(), ZERO_TOKEN)


class ModuleBoundaryTests(unittest.TestCase):
    def test_modules_have_no_randomness_or_forbidden_dependency(self) -> None:
        for module in (baseline_module, exploration_module):
            tree = ast.parse(inspect.getsource(module))
            imported = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            } | {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            }
            with self.subTest(module=module.__name__):
                self.assertNotIn("random", imported)
                self.assertNotIn("secrets", imported)
                self.assertFalse(any(name.startswith("torch") for name in imported))
                self.assertFalse(
                    any(
                        name.startswith(
                            (
                                "lisjong.policies",
                                "lisjong_arena",
                                "riichienv",
                                "lisjong.riichi",
                            )
                        )
                        for name in imported
                    )
                )

    def test_selector_is_not_a_policy(self) -> None:
        self.assertFalse(hasattr(exploration_module, "choose_action"))
        self.assertEqual(
            list(inspect.signature(select_residual_exploration).parameters),
            ["decision", "exploration_token"],
        )


if __name__ == "__main__":
    unittest.main()
