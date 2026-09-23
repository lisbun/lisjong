"""Issue #191 semantic-envelope Policyとbounded replay diagnosticsのtest。

scorer runtimeはfakeで差し替え、S1 / S2 / S3 envelope、residual choice、
TwoStep oracle、identity分離、fail closedをML runtimeなしで固定する。
"""

import ast
import inspect
import random
import tempfile
import unittest
from pathlib import Path

import candidate_fixtures as cf

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.learning import (
    CANDIDATE_ENCODING_IDENTITY,
    FEATURE_DIMENSION,
    SECOND_STEP_REQUEST_POLICY,
    SEMANTIC_ENVELOPE_IDENTITY,
    CandidateFeatureError,
    DiscardCandidateFeatures,
    LearnedCandidateOffensePolicy,
    LearnedPolicyError,
    SecondStepStatus,
    SemanticEnvelopeOffensePolicy,
    SemanticEnvelopeRuntime,
    build_player_safe_feature,
    build_scorer_candidates,
    classify_semantic_envelope_result,
    evaluate_semantic_envelope_policy,
    read_source_record,
    semantic_envelope_runtime_identity,
    semantic_envelope_survivors,
)
from lisjong.learning import envelope_diagnostics as diagnostics_module
from lisjong.learning import envelope_policy as envelope_module
from lisjong.learning._o0 import O0DecisionKind
from lisjong.learning.candidate_encoding import CANDIDATE_BLOCK_OFFSETS
from lisjong.learning.envelope_diagnostics import ENVELOPE_INVALID, ENVELOPE_READY
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import (
    DecisionContext,
    DiscardAction,
    InternalAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
    Seat,
    TileCategory,
    TileType,
    TsumoAction,
    execute_policy,
)

_TILE = CANDIDATE_BLOCK_OFFSETS["discard_tile_type"]
_RED = CANDIDATE_BLOCK_OFFSETS["discard_red"]
_TSUMOGIRI = CANDIDATE_BLOCK_OFFSETS["tsumogiri"]

TENPAI_SINGLE_HAND = "123m456p789s11z23s9m"
"""打牌後聴牌で、S2 survivorが1件（9mツモ切り、NOT_APPLICABLE）の打牌decision。"""

SINGLE_NOT_MATERIALIZED_HAND = "8m0p8p9p8p2s5m3z2s4z9s8s8m9p"
"""最小向聴2で、S2 survivorが1件（NOT_MATERIALIZED）の打牌decision。"""

S3_NARROWS_TO_ONE_HAND = "6m5z9p3m1m1p4z7s3s2m9p6z4p1z"
"""S2 finalistが複数で、S3が1件へ絞る打牌decision。"""

_WALL = tuple(
    [
        f"{rank}{suit}"
        for suit in "mps"
        for rank in range(1, 10)
        if rank != 5
        for _ in "1234"
    ]
    + [f"5{suit}" for suit in "mps" for _ in "123"]
    + [f"0{suit}" for suit in "mps"]
    + [f"{rank}z" for rank in range(1, 8) for _ in "1234"]
)
_ORACLE_SEEDS = (0, 1, 2, 3, 4, 5, 8, 11, 12, 14, 15, 18)


def _random_hand(seed):
    return "".join(random.Random(seed).sample(_WALL, 14))


class RecordingRuntime:
    """呼び出しを記録し、candidate vectorからscoreを決めるfake runtime。"""

    def __init__(self, scorer=None) -> None:
        self.calls = []
        self.scorer = scorer or (lambda vector: 0.0)

    def score(self, context, candidates):
        self.calls.append((context, candidates))
        return tuple(self.scorer(vector) for vector in candidates)


def _prefer(*, tile_index=None, red=None, tsumogiri=None):
    def scorer(vector):
        score = 0.0
        if tile_index is not None and vector[_TILE + tile_index] == 1.0:
            score += 1.0
        if red is not None and vector[_RED] == float(red):
            score += 1.0
        if tsumogiri is not None and vector[_TSUMOGIRI] == float(tsumogiri):
            score += 1.0
        return score

    return scorer


def _honor(rank):
    return tile_type_index(TileType(TileCategory.HONOR, rank))


def _policy(scorer=None):
    runtime = RecordingRuntime(scorer)
    return SemanticEnvelopeOffensePolicy(runtime), runtime


def _is_legal_object(action, decision) -> bool:
    return any(action is legal for legal in decision.legal_actions)


def _synthetic(tile_spec, shanten, ukeire, status, score=None):
    return DiscardCandidateFeatures(
        action=DiscardAction(
            actor=Seat.SEAT_0, tile=cf.hand(tile_spec)[0], tsumogiri=False
        ),
        post_discard_shanten=shanten,
        current_ukeire_count=ukeire,
        second_step_ukeire_score=score,
        second_step_status=status,
    )


_E = SecondStepStatus.EVALUATED
_NM = SecondStepStatus.NOT_MATERIALIZED
_NA = SecondStepStatus.NOT_APPLICABLE


class SurvivorStageTests(unittest.TestCase):
    def test_s1_keeps_only_the_minimum_shanten(self) -> None:
        candidates = (
            _synthetic("1m", 2, 60, _NM),
            _synthetic("2m", 1, 20, _NM),
            _synthetic("3m", 3, 90, _NM),
        )

        self.assertEqual(semantic_envelope_survivors(candidates), (1,))

    def test_s2_keeps_only_the_maximum_ukeire_within_s1(self) -> None:
        candidates = (
            _synthetic("1m", 2, 80, _NM),
            _synthetic("2m", 1, 20, _E, 300),
            _synthetic("3m", 1, 24, _E, 100),
            _synthetic("4m", 1, 24, _E, 100),
            _synthetic("5m", 1, 22, _NM),
        )

        self.assertEqual(semantic_envelope_survivors(candidates), (2, 3))

    def test_s3_keeps_only_the_maximum_second_step(self) -> None:
        candidates = (
            _synthetic("1m", 1, 24, _E, 100),
            _synthetic("2m", 1, 24, _E, 140),
            _synthetic("3m", 1, 24, _E, 140),
            _synthetic("4m", 1, 20, _NM),
        )

        self.assertEqual(semantic_envelope_survivors(candidates), (1, 2))

    def test_evaluated_zero_is_a_real_score(self) -> None:
        zero = (_synthetic("1m", 1, 4, _E, 0), _synthetic("2m", 1, 4, _E, 0))
        mixed = (_synthetic("1m", 1, 4, _E, 0), _synthetic("2m", 1, 4, _E, 3))

        self.assertEqual(semantic_envelope_survivors(zero), (0, 1))
        self.assertEqual(semantic_envelope_survivors(mixed), (1,))

    def test_s3_rejects_unevaluated_finalists_instead_of_scoring_them_zero(
        self,
    ) -> None:
        for statuses in ((_E, _NM), (_NM, _NM), (_E, _NA)):
            candidates = tuple(
                _synthetic(f"{rank}m", 1, 24, status, 50 if status is _E else None)
                for rank, status in zip((1, 2), statuses, strict=True)
            )
            with self.subTest(statuses=statuses):
                with self.assertRaisesRegex(CandidateFeatureError, "evaluated"):
                    semantic_envelope_survivors(candidates)

    def test_tenpai_survivors_must_be_not_applicable(self) -> None:
        single = (_synthetic("1m", 0, 8, _NA), _synthetic("2m", 1, 30, _NM))
        multiple = (_synthetic("1m", 0, 8, _NA), _synthetic("2m", 0, 8, _NA))

        self.assertEqual(semantic_envelope_survivors(single), (0,))
        self.assertEqual(semantic_envelope_survivors(multiple), (0, 1))
        for status in (_NM, _E):
            candidates = (_synthetic("1m", 0, 8, status, 5 if status is _E else None),)
            with self.subTest(status=status):
                with self.assertRaisesRegex(CandidateFeatureError, "not_applicable"):
                    semantic_envelope_survivors(candidates)

    def test_single_non_tenpai_survivor_must_be_not_materialized(self) -> None:
        accepted = (_synthetic("1m", 1, 24, _NM), _synthetic("2m", 1, 20, _NM))
        requested = (_synthetic("1m", 1, 24, _E, 70), _synthetic("2m", 1, 20, _NM))

        self.assertEqual(semantic_envelope_survivors(accepted), (0,))
        with self.assertRaisesRegex(CandidateFeatureError, "not_materialized"):
            semantic_envelope_survivors(requested)

    def test_empty_candidates_fail_closed(self) -> None:
        with self.assertRaises(CandidateFeatureError):
            semantic_envelope_survivors(())

    def test_real_decisions_follow_the_two_pass_request_policy(self) -> None:
        cases = {
            TENPAI_SINGLE_HAND: (0, 1, _NA),
            cf.TENPAI_REACHABLE_HAND: (0, 2, _NA),
            SINGLE_NOT_MATERIALIZED_HAND: (2, 1, _NM),
            S3_NARROWS_TO_ONE_HAND: (5, 1, _E),
            cf.NEAR_HAND: (1, 2, _E),
        }
        for spec, (shanten, count, status) in cases.items():
            candidates = build_scorer_candidates(cf.discard_decision(spec))
            survivors = semantic_envelope_survivors(candidates)
            with self.subTest(spec=spec):
                self.assertEqual(
                    min(item.post_discard_shanten for item in candidates), shanten
                )
                self.assertEqual(len(survivors), count)
                self.assertIs(candidates[survivors[0]].second_step_status, status)


class O0GuardTests(unittest.TestCase):
    """win / riichi / no-callは#189 policyと同じdeterministic guardである。"""

    def test_guard_branches_match_the_189_policy_without_scoring(self) -> None:
        cases = (
            (cf.tsumo_decision(), O0DecisionKind.WIN, TsumoAction),
            (cf.ron_response_decision(), O0DecisionKind.WIN, RonAction),
            (cf.riichi_decision(), O0DecisionKind.RIICHI, RiichiAction),
            (cf.response_decision(), O0DecisionKind.RESPONSE, PassAction),
        )
        for decision, kind, action_type in cases:
            policy, runtime = _policy()
            result = policy.decide(decision)
            historical = LearnedCandidateOffensePolicy(RecordingRuntime()).decide(
                decision
            )
            with self.subTest(kind=kind):
                self.assertIs(result.kind, kind)
                self.assertIsInstance(result.action, action_type)
                self.assertIs(result.action, historical.action)
                self.assertIsNone(result.candidates)
                self.assertIsNone(result.survivors)
                self.assertEqual(runtime.calls, [])

    def test_response_without_pass_fails_closed(self) -> None:
        response = cf.response_decision()
        pon = response.legal_actions[1]
        self.assertIsInstance(pon, PonAction)
        decision = DecisionContext(input=response.input, legal_actions=(pon,))

        with self.assertRaises(LearnedPolicyError):
            _policy()[0].choose_action(decision)


class ResidualSelectionTests(unittest.TestCase):
    def test_single_survivor_cannot_be_overridden_by_the_scorer(self) -> None:
        for spec in (TENPAI_SINGLE_HAND, SINGLE_NOT_MATERIALIZED_HAND):
            decision = cf.discard_decision(spec)
            candidates = build_scorer_candidates(decision)
            (survivor,) = semantic_envelope_survivors(candidates)
            for candidate in candidates:
                tile_index = tile_type_index(candidate.action.tile.tile_type)
                policy, runtime = _policy(_prefer(tile_index=tile_index))
                result = policy.decide(decision)
                with self.subTest(spec=spec, tile_index=tile_index):
                    self.assertIs(result.action, candidates[survivor].action)
                    self.assertTrue(_is_legal_object(result.action, decision))
                    self.assertFalse(result.scorer_invoked)
                    self.assertEqual(runtime.calls, [])

    def test_multiple_survivors_let_the_scorer_choose_within_the_set(self) -> None:
        decision = cf.discard_decision(cf.NEAR_HAND)
        for rank in (2, 3):
            policy, runtime = _policy(_prefer(tile_index=_honor(rank)))
            result = policy.decide(decision)
            with self.subTest(rank=rank):
                self.assertEqual(result.action.tile.tile_type.rank, rank)
                self.assertIs(result.action.tile.tile_type.category, TileCategory.HONOR)
                self.assertTrue(_is_legal_object(result.action, decision))
                self.assertTrue(result.scorer_invoked)
                self.assertEqual(len(runtime.calls), 1)

    def test_excluded_candidate_with_maximum_score_is_never_selected(self) -> None:
        decision = cf.discard_decision(cf.NEAR_HAND)
        candidates = build_scorer_candidates(decision)
        survivors = set(semantic_envelope_survivors(candidates))
        excluded = [i for i in range(len(candidates)) if i not in survivors]
        self.assertTrue(excluded)
        for index in excluded:
            target = candidates[index].action

            def scorer(vector, target=target):
                tile = target.tile
                return (
                    100.0
                    if vector[_TILE + tile_type_index(tile.tile_type)] == 1.0
                    and vector[_RED] == float(tile.is_red)
                    and vector[_TSUMOGIRI] == float(target.tsumogiri)
                    else 0.0
                )

            policy, _runtime = _policy(scorer)
            historical = LearnedCandidateOffensePolicy(RecordingRuntime(scorer))
            result = policy.decide(decision)
            with self.subTest(excluded=index):
                self.assertIs(historical.choose_action(decision), target)
                self.assertIsNot(result.action, target)
                self.assertIn(result.action, [candidates[i].action for i in survivors])
                self.assertEqual(max(result.scores), 100.0)

    def test_equal_scores_use_the_canonical_first_survivor(self) -> None:
        decision = cf.discard_decision(cf.NEAR_HAND)
        result = _policy()[0].decide(decision)

        self.assertIs(result.action, result.candidates[result.survivors[0]].action)

    def test_scorer_sees_the_full_candidate_tuple_and_player_safe_context(
        self,
    ) -> None:
        decision = cf.discard_with_history(cf.NEAR_HAND)
        policy, runtime = _policy()

        result = policy.decide(decision)

        context, encoded = runtime.calls[0]
        self.assertEqual(context, build_player_safe_feature(decision.input))
        self.assertEqual(len(context), FEATURE_DIMENSION)
        self.assertEqual(len(encoded), len(result.candidates))
        self.assertEqual(len(result.scores), len(result.candidates))

    def test_red_five_and_normal_five_identity_is_preserved(self) -> None:
        decision = cf.discard_decision(cf.TENPAI_REACHABLE_HAND)
        for red in (True, False):
            action = _policy(_prefer(red=red))[0].choose_action(decision)
            with self.subTest(red=red):
                self.assertEqual(action.tile.tile_type.rank, 5)
                self.assertIs(action.tile.is_red, red)
                self.assertTrue(_is_legal_object(action, decision))

    def test_tsumogiri_identity_is_preserved(self) -> None:
        decision = cf.discard_decision(cf.SECOND_NEAR_HAND)
        drawn = decision.input.own_hand.drawn_tile
        for tsumogiri in (True, False):
            action = _policy(_prefer(tsumogiri=tsumogiri))[0].choose_action(decision)
            with self.subTest(tsumogiri=tsumogiri):
                self.assertEqual(action.tile, drawn)
                self.assertIs(action.tsumogiri, tsumogiri)
                self.assertTrue(_is_legal_object(action, decision))

    def test_selection_is_independent_of_legal_action_input_order(self) -> None:
        for spec, scorer in (
            (cf.NEAR_HAND, _prefer(tile_index=_honor(3))),
            (cf.SECOND_NEAR_HAND, _prefer(tsumogiri=True)),
            (cf.TENPAI_REACHABLE_HAND, _prefer(red=True)),
        ):
            decision = cf.discard_decision(spec)
            expected = _policy(scorer)[0].choose_action(decision)
            for seed in range(4):
                legal = list(decision.legal_actions)
                random.Random(seed).shuffle(legal)
                shuffled = DecisionContext(
                    input=decision.input, legal_actions=tuple(legal)
                )
                action = _policy(scorer)[0].choose_action(shuffled)
                with self.subTest(spec=spec, seed=seed):
                    self.assertEqual(action, expected)
                    self.assertTrue(_is_legal_object(action, shuffled))

    def test_non_finite_score_fails_closed_when_the_scorer_runs(self) -> None:
        decision = cf.discard_decision(cf.NEAR_HAND)
        for value in (float("nan"), float("inf"), float("-inf")):
            policy, _runtime = _policy(lambda vector, value=value: value)
            with self.subTest(value=value):
                with self.assertRaisesRegex(LearnedPolicyError, "non-finite"):
                    policy.choose_action(decision)

    def test_skipped_scorer_is_not_checked_for_non_finite_scores(self) -> None:
        decision = cf.discard_decision(TENPAI_SINGLE_HAND)
        policy, runtime = _policy(lambda vector: float("nan"))

        action = policy.choose_action(decision)

        self.assertTrue(_is_legal_object(action, decision))
        self.assertEqual(runtime.calls, [])

    def test_wrong_score_count_fails_closed(self) -> None:
        class ShortRuntime:
            def score(self, context, candidates):
                return (0.0,)

        with self.assertRaisesRegex(LearnedPolicyError, "score count"):
            SemanticEnvelopeOffensePolicy(ShortRuntime()).choose_action(
                cf.discard_decision(cf.NEAR_HAND)
            )

    def test_non_decision_context_input_rejected(self) -> None:
        with self.assertRaises(LearnedPolicyError):
            _policy()[0].choose_action(cf.discard_decision().input)

    def test_policy_passes_the_normal_execution_boundary(self) -> None:
        for decision in cf.mixed_decisions():
            action = execute_policy(_policy()[0], decision)
            with self.subTest(action=action):
                self.assertTrue(_is_legal_object(action, decision))


class TwoStepOracleTests(unittest.TestCase):
    """constant scorerのenvelope policyはTwoStepと同じaction objectを返す。

    #187 / #189 candidate semantic pathとTwoStep staged selectionという独立
    実装の一致を固定する。
    """

    def test_constant_scorer_matches_two_step_ukeire(self) -> None:
        teacher = TwoStepUkeirePolicy()
        decisions = (
            [("mixed", decision) for decision in cf.mixed_decisions()]
            + [
                (spec, cf.discard_decision(spec))
                for spec in (
                    cf.NEAR_HAND,
                    cf.TENPAI_REACHABLE_HAND,
                    TENPAI_SINGLE_HAND,
                    SINGLE_NOT_MATERIALIZED_HAND,
                    S3_NARROWS_TO_ONE_HAND,
                )
            ]
            + [
                (f"seed={seed}", cf.discard_decision(_random_hand(seed)))
                for seed in _ORACLE_SEEDS
            ]
        )
        for label, decision in decisions:
            with self.subTest(decision=label):
                self.assertIs(
                    _policy()[0].choose_action(decision),
                    teacher.choose_action(decision),
                )


class IdentityTests(unittest.TestCase):
    class _Scorer:
        identity = "a" * 64

        def score(self, context, candidates):
            return tuple(0.0 for _ in candidates)

    def test_runtime_identity_is_composed_and_does_not_collide(self) -> None:
        runtime = SemanticEnvelopeRuntime(self._Scorer())

        self.assertEqual(runtime.artifact_identity, "a" * 64)
        self.assertNotEqual(runtime.identity, runtime.artifact_identity)
        self.assertEqual(runtime.identity, semantic_envelope_runtime_identity("a" * 64))
        self.assertNotEqual(
            runtime.identity, semantic_envelope_runtime_identity("b" * 64)
        )
        self.assertIsInstance(runtime(), SemanticEnvelopeOffensePolicy)
        self.assertIsNot(runtime(), runtime())

    def test_existing_identities_are_unchanged(self) -> None:
        self.assertEqual(
            SEMANTIC_ENVELOPE_IDENTITY, "lisjong-offense-l0.2-semantic-envelope-v1"
        )
        self.assertEqual(
            SECOND_STEP_REQUEST_POLICY,
            "lisjong-offense-l0.2-two-pass-finalist-second-step-v1",
        )
        self.assertEqual(
            CANDIDATE_ENCODING_IDENTITY,
            "lisjong-offense-l0.2-discard-candidate-encoding-v1",
        )


class ReplayDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        path = cf.write_candidate_source_record(Path(self._directory.name) / "source")
        self.source = read_source_record(path)

    def test_constant_scorer_replay_is_ready(self) -> None:
        evaluation = evaluate_semantic_envelope_policy(
            _policy()[0],
            self.source,
            ["TRAIN", "SELECT"],
            reference=TwoStepUkeirePolicy(),
        )
        classification = classify_semantic_envelope_result(evaluation)

        self.assertEqual(classification["outcome"], ENVELOPE_READY)
        self.assertEqual(classification["failures"], [])
        invariants = evaluation["invariants"]
        self.assertEqual(invariants["legality"], 1.0)
        self.assertEqual(
            invariants["max_regret"],
            {"shanten": 0, "current_ukeire": 0, "second_step": 0},
        )
        self.assertGreater(invariants["regret_support"]["second_step"], 0)
        self.assertEqual(evaluation["support"]["scorer_decisions"], 6)
        reference = evaluation["reference"]
        self.assertEqual(reference["constant_scorer_oracle_agreement"], 1.0)
        self.assertEqual(reference["policy_agreement"], 1.0)
        residual = evaluation["residual_choice"]
        self.assertEqual(residual["survivor_count_distribution"], {"2": 6})
        self.assertEqual(residual["fraction_scorer_invoked"], 1.0)
        self.assertEqual(residual["fraction_non_canonical_first_given_residual"], 0.0)

    def test_residual_scorer_choice_is_recorded_but_not_a_failure(self) -> None:
        evaluation = evaluate_semantic_envelope_policy(
            _policy(_prefer(tsumogiri=True, red=True))[0],
            self.source,
            ["SELECT"],
            reference=TwoStepUkeirePolicy(),
        )

        residual = evaluation["residual_choice"]
        self.assertGreater(residual["fraction_non_canonical_first_given_residual"], 0)
        self.assertLess(evaluation["reference"]["policy_agreement"], 1.0)
        self.assertEqual(evaluation["reference"]["disagreement_outside_survivors"], 0)
        self.assertEqual(
            classify_semantic_envelope_result(evaluation)["outcome"], ENVELOPE_READY
        )

    def test_selection_outside_the_envelope_is_invalid(self) -> None:
        class OutsidePolicy:
            """envelope外のcandidateを返す壊れたpolicy（検出力の確認用）。"""

            def decide(self, decision):
                result = _policy()[0].decide(decision)
                if result.candidates is None:
                    return result
                worst = max(
                    result.candidates, key=lambda item: item.post_discard_shanten
                )
                return envelope_module.SemanticEnvelopeDecision(
                    kind=result.kind,
                    action=worst.action,
                    candidates=result.candidates,
                    survivors=result.survivors,
                    scores=result.scores,
                )

        evaluation = evaluate_semantic_envelope_policy(
            OutsidePolicy(), self.source, ["TRAIN"], reference=TwoStepUkeirePolicy()
        )
        classification = classify_semantic_envelope_result(evaluation)

        self.assertEqual(classification["outcome"], ENVELOPE_INVALID)
        names = {failure["invariant"] for failure in classification["failures"]}
        self.assertIn("shanten_regret", names)
        self.assertIn("reference_disagreement_outside_survivors", names)

    def test_sample_every_strides_within_the_selected_splits(self) -> None:
        full = evaluate_semantic_envelope_policy(_policy()[0], self.source, ["TRAIN"])
        strided = evaluate_semantic_envelope_policy(
            _policy()[0], self.source, ["TRAIN"], sample_every=2
        )

        self.assertEqual(full["support"]["decisions"], 7)
        self.assertEqual(strided["support"]["decisions"], 4)
        self.assertNotIn("reference", full)
        with self.assertRaises(ValueError):
            evaluate_semantic_envelope_policy(
                _policy()[0], self.source, ["TRAIN"], sample_every=0
            )


class SourceContractTests(unittest.TestCase):
    def test_modules_have_no_fallback_randomness_or_concrete_policy_dependency(
        self,
    ) -> None:
        for module in (envelope_module, diagnostics_module):
            tree = ast.parse(inspect.getsource(module))
            nodes = list(ast.walk(tree))
            imported = {
                alias.name
                for node in nodes
                if isinstance(node, ast.Import)
                for alias in node.names
            } | {
                node.module
                for node in nodes
                if isinstance(node, ast.ImportFrom) and node.module
            }
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    [node for node in nodes if isinstance(node, ast.ExceptHandler)], []
                )
                self.assertNotIn("random", imported)
                self.assertFalse(any(name.startswith("torch") for name in imported))
                self.assertFalse(
                    any(name.startswith("lisjong.policies") for name in imported)
                )
                self.assertFalse(
                    any(
                        name.startswith(
                            ("riichienv", "lisjong_arena", "lisjong.riichi")
                        )
                        for name in imported
                    )
                )

    def test_policy_is_separate_from_the_historical_189_policy(self) -> None:
        self.assertNotIn(
            LearnedCandidateOffensePolicy, SemanticEnvelopeOffensePolicy.__mro__
        )
        self.assertNotIn(TwoStepUkeirePolicy, SemanticEnvelopeOffensePolicy.__mro__)

    def test_policy_contract_signature(self) -> None:
        signature = inspect.signature(SemanticEnvelopeOffensePolicy.choose_action)

        self.assertEqual(list(signature.parameters), ["self", "decision"])
        self.assertIs(signature.return_annotation, InternalAction)


if __name__ == "__main__":
    unittest.main()
