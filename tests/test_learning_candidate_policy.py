"""Issue #189 O0 decomposed decision seamとcandidate scorer safetyのtest。

scorer runtimeはfakeで差し替え、O0 precedence、legal DiscardAction以外を
返せない構造、canonical tie-break、fail closedをML runtimeなしで固定する。
"""

import ast
import inspect
import random
import unittest

import candidate_fixtures as cf

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.learning import (
    CANDIDATE_ENCODING_DIMENSION,
    FEATURE_DIMENSION,
    LearnedCandidateOffensePolicy,
    LearnedPolicyError,
)
from lisjong.learning import candidate_policy as candidate_policy_module
from lisjong.learning._o0 import O0DecisionKind
from lisjong.learning.candidate_encoding import CANDIDATE_BLOCK_OFFSETS
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import (
    AnkanAction,
    DecisionContext,
    DiscardAction,
    InternalAction,
    KyuushuKyuuhaiAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
    Seat,
    TsumoAction,
    execute_policy,
)
from lisjong.structural_efficiency import discard_action_sort_key

_TILE = CANDIDATE_BLOCK_OFFSETS["discard_tile_type"]
_RED = CANDIDATE_BLOCK_OFFSETS["discard_red"]
_TSUMOGIRI = CANDIDATE_BLOCK_OFFSETS["tsumogiri"]


class RecordingRuntime:
    """呼び出しを記録し、candidate vectorからscoreを決めるfake runtime。"""

    def __init__(self, scorer=None) -> None:
        self.calls = []
        self.scorer = scorer or (lambda vector: 0.0)

    def score(self, context, candidates):
        self.calls.append((context, candidates))
        assert len(context) == FEATURE_DIMENSION
        assert all(len(vector) == CANDIDATE_ENCODING_DIMENSION for vector in candidates)
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


def _is_legal_object(action, decision) -> bool:
    return any(action is legal for legal in decision.legal_actions)


class O0GuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = RecordingRuntime()
        self.policy = LearnedCandidateOffensePolicy(self.runtime)

    def test_legal_win_takes_precedence_over_riichi_and_discard(self) -> None:
        decision = cf.tsumo_decision()

        result = self.policy.decide(decision)

        self.assertIs(result.kind, O0DecisionKind.WIN)
        self.assertIsInstance(result.action, TsumoAction)
        self.assertTrue(_is_legal_object(result.action, decision))
        self.assertEqual(result.action, TwoStepUkeirePolicy().choose_action(decision))
        self.assertEqual(self.runtime.calls, [])

    def test_ron_in_a_response_decision_is_chosen_over_pass(self) -> None:
        decision = cf.ron_response_decision()

        action = self.policy.choose_action(decision)

        self.assertIsInstance(action, RonAction)
        self.assertTrue(_is_legal_object(action, decision))

    def test_multiple_winning_actions_use_the_canonical_tie_break(self) -> None:
        value = cf.policy_input(cf.NEAR_HAND)
        seat = value.self_seat
        tile = value.own_hand.drawn_tile
        wins = (
            TsumoAction(actor=seat, winning_tile=tile),
            RonAction(actor=seat, target=Seat.SEAT_3, winning_tile=tile),
            RonAction(actor=seat, target=Seat.SEAT_1, winning_tile=tile),
        )
        teacher = TwoStepUkeirePolicy()
        for seed in range(4):
            legal = list(cf.discard_actions(value) + wins)
            random.Random(seed).shuffle(legal)
            decision = DecisionContext(input=value, legal_actions=tuple(legal))
            with self.subTest(seed=seed):
                action = self.policy.choose_action(decision)
                self.assertEqual(action, teacher.choose_action(decision))
                self.assertEqual(
                    action, RonAction(actor=seat, target=Seat.SEAT_1, winning_tile=tile)
                )
                self.assertTrue(_is_legal_object(action, decision))

    def test_immediate_riichi_matches_the_always_riichi_semantics(self) -> None:
        decision = cf.riichi_decision()

        result = self.policy.decide(decision)

        self.assertIs(result.kind, O0DecisionKind.RIICHI)
        self.assertIsInstance(result.action, RiichiAction)
        self.assertTrue(_is_legal_object(result.action, decision))
        self.assertEqual(result.action, TwoStepUkeirePolicy().choose_action(decision))
        self.assertEqual(self.runtime.calls, [])

    def test_response_decision_returns_the_canonical_pass(self) -> None:
        decision = cf.response_decision()

        result = self.policy.decide(decision)

        self.assertIs(result.kind, O0DecisionKind.RESPONSE)
        self.assertIsInstance(result.action, PassAction)
        self.assertTrue(_is_legal_object(result.action, decision))
        self.assertEqual(self.runtime.calls, [])

    def test_response_decision_without_pass_fails_closed(self) -> None:
        pon = cf.response_decision().legal_actions[1]
        self.assertIsInstance(pon, PonAction)
        decision = DecisionContext(
            input=cf.response_decision().input, legal_actions=(pon,)
        )

        with self.assertRaises(LearnedPolicyError):
            self.policy.choose_action(decision)

    def test_own_turn_voluntary_actions_are_never_selected(self) -> None:
        ankan = cf.ankan_decision()
        decision = DecisionContext(
            input=ankan.input,
            legal_actions=ankan.legal_actions
            + (KyuushuKyuuhaiAction(actor=ankan.input.self_seat),),
        )
        self.assertTrue(
            any(isinstance(action, AnkanAction) for action in decision.legal_actions)
        )

        result = self.policy.decide(decision)

        self.assertIs(result.kind, O0DecisionKind.DISCARD)
        self.assertIsInstance(result.action, DiscardAction)
        self.assertTrue(_is_legal_object(result.action, decision))
        self.assertEqual(len(self.runtime.calls), 1)

    def test_riichi_declaration_discard_goes_through_the_scorer(self) -> None:
        """宣言後の宣言牌decisionはRiichiActionを含まないdiscard decisionである。"""
        value = cf.policy_input(cf.NEAR_HAND)
        restricted = tuple(
            action
            for action in cf.discard_actions(value)
            if action.tile.tile_type.category.value == "honor"
        )
        decision = DecisionContext(input=value, legal_actions=restricted)

        action = self.policy.choose_action(decision)

        self.assertIn(action, restricted)
        self.assertTrue(_is_legal_object(action, decision))

    def test_forced_tsumogiri_during_riichi_returns_that_object(self) -> None:
        value = cf.policy_input(cf.NEAR_HAND, riichi=True)
        legal = cf.discard_actions(value, tsumogiri_only=True)
        decision = DecisionContext(input=value, legal_actions=legal)

        action = self.policy.choose_action(decision)

        self.assertIs(action, legal[0])
        self.assertTrue(action.tsumogiri)


class ScorerSafetyTests(unittest.TestCase):
    def test_returned_action_is_the_canonical_legal_object(self) -> None:
        decision = cf.discard_decision(cf.NEAR_HAND)
        tile_indices = sorted(
            {
                tile_type_index(action.tile.tile_type)
                for action in decision.legal_actions
            }
        )
        for tile_index in tile_indices:
            policy = LearnedCandidateOffensePolicy(
                RecordingRuntime(_prefer(tile_index=tile_index))
            )
            action = policy.choose_action(decision)
            with self.subTest(tile_index=tile_index):
                self.assertIsInstance(action, DiscardAction)
                self.assertTrue(_is_legal_object(action, decision))

    def test_red_five_and_normal_five_identity_is_preserved(self) -> None:
        decision = cf.discard_decision(cf.TENPAI_REACHABLE_HAND)
        fives = [
            action
            for action in decision.legal_actions
            if action.tile.tile_type.rank == 5
            and action.tile.tile_type.category.value == "manzu"
        ]
        self.assertEqual({action.tile.is_red for action in fives}, {True, False})
        manzu_five = 4

        red = LearnedCandidateOffensePolicy(
            RecordingRuntime(_prefer(tile_index=manzu_five, red=True))
        ).choose_action(decision)
        normal = LearnedCandidateOffensePolicy(
            RecordingRuntime(_prefer(tile_index=manzu_five, red=False))
        ).choose_action(decision)

        self.assertTrue(red.tile.is_red)
        self.assertFalse(normal.tile.is_red)
        self.assertTrue(_is_legal_object(red, decision))
        self.assertTrue(_is_legal_object(normal, decision))

    def test_tsumogiri_identity_is_preserved(self) -> None:
        decision = cf.discard_decision(cf.TENPAI_REACHABLE_HAND)
        drawn = decision.input.own_hand.drawn_tile
        same_tile = [
            action for action in decision.legal_actions if action.tile == drawn
        ]
        self.assertEqual({action.tsumogiri for action in same_tile}, {True, False})

        for tsumogiri in (True, False):
            action = LearnedCandidateOffensePolicy(
                RecordingRuntime(_prefer(tile_index=28, tsumogiri=tsumogiri))
            ).choose_action(decision)
            with self.subTest(tsumogiri=tsumogiri):
                self.assertEqual(action.tile, drawn)
                self.assertIs(action.tsumogiri, tsumogiri)
                self.assertTrue(_is_legal_object(action, decision))

    def test_selection_is_independent_of_legal_action_input_order(self) -> None:
        decision = cf.discard_decision(cf.FAR_HAND)
        scorer = _prefer(tile_index=9, tsumogiri=False)
        expected = LearnedCandidateOffensePolicy(
            RecordingRuntime(scorer)
        ).choose_action(decision)
        for seed in range(5):
            legal = list(decision.legal_actions)
            random.Random(seed).shuffle(legal)
            shuffled = DecisionContext(input=decision.input, legal_actions=tuple(legal))
            action = LearnedCandidateOffensePolicy(
                RecordingRuntime(scorer)
            ).choose_action(shuffled)
            with self.subTest(seed=seed):
                self.assertEqual(action, expected)
                self.assertTrue(_is_legal_object(action, shuffled))

    def test_equal_scores_use_the_canonical_discard_tie_break(self) -> None:
        decision = cf.discard_decision(cf.FAR_HAND)
        policy = LearnedCandidateOffensePolicy(RecordingRuntime())

        action = policy.choose_action(decision)

        self.assertEqual(
            action, min(decision.legal_actions, key=discard_action_sort_key)
        )

    def test_scorer_sees_only_legal_discard_candidates(self) -> None:
        decision = cf.discard_decision(
            cf.FAR_HAND, extra=(KyuushuKyuuhaiAction(actor=Seat.SEAT_0),)
        )
        runtime = RecordingRuntime()

        LearnedCandidateOffensePolicy(runtime).choose_action(decision)

        discards = [a for a in decision.legal_actions if isinstance(a, DiscardAction)]
        self.assertEqual(len(runtime.calls[0][1]), len(discards))

    def test_non_finite_score_fails_closed(self) -> None:
        decision = cf.discard_decision(cf.FAR_HAND)
        for value in (float("nan"), float("inf"), float("-inf")):
            policy = LearnedCandidateOffensePolicy(
                RecordingRuntime(lambda vector, value=value: value)
            )
            with self.subTest(value=value):
                with self.assertRaisesRegex(LearnedPolicyError, "non-finite"):
                    policy.choose_action(decision)

    def test_wrong_score_count_fails_closed(self) -> None:
        class ShortRuntime:
            def score(self, context, candidates):
                return (0.0,)

        with self.assertRaisesRegex(LearnedPolicyError, "score count"):
            LearnedCandidateOffensePolicy(ShortRuntime()).choose_action(
                cf.discard_decision(cf.FAR_HAND)
            )

    def test_non_decision_context_input_rejected(self) -> None:
        with self.assertRaises(LearnedPolicyError):
            LearnedCandidateOffensePolicy(RecordingRuntime()).choose_action(
                cf.discard_decision().input
            )

    def test_policy_passes_the_normal_execution_boundary(self) -> None:
        for decision in cf.mixed_decisions():
            action = execute_policy(
                LearnedCandidateOffensePolicy(RecordingRuntime()), decision
            )
            with self.subTest(action=action):
                self.assertTrue(_is_legal_object(action, decision))

    def test_shared_context_reflects_only_player_visible_input(self) -> None:
        """scorerへ渡るcontextは#184 build_player_safe_feature(PolicyInput)である。"""
        from lisjong.learning import build_player_safe_feature

        runtime = RecordingRuntime()
        decision = cf.discard_with_history()
        LearnedCandidateOffensePolicy(runtime).choose_action(decision)

        self.assertEqual(runtime.calls[0][0], build_player_safe_feature(decision.input))


class SourceContractTests(unittest.TestCase):
    def test_policy_module_has_no_exception_fallback_or_randomness(self) -> None:
        tree = ast.parse(inspect.getsource(candidate_policy_module))
        nodes = list(ast.walk(tree))

        self.assertEqual(
            [node for node in nodes if isinstance(node, ast.ExceptHandler)], []
        )
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
        self.assertNotIn("random", imported)
        self.assertFalse(any(name.startswith("torch") for name in imported))
        self.assertFalse(
            any(
                name.startswith(("riichienv", "lisjong_arena", "lisjong.riichi"))
                for name in imported
            )
        )

    def test_policy_does_not_inherit_two_step_ukeire(self) -> None:
        self.assertNotIn(TwoStepUkeirePolicy, LearnedCandidateOffensePolicy.__mro__)

    def test_policy_contract_signature(self) -> None:
        signature = inspect.signature(LearnedCandidateOffensePolicy.choose_action)

        self.assertEqual(list(signature.parameters), ["self", "decision"])
        self.assertIs(signature.return_annotation, InternalAction)


if __name__ == "__main__":
    unittest.main()
