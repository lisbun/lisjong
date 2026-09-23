"""Issue #187 `lisjong.learning.candidate_features`のunit test。

purpose-specific discard candidate viewが、`lisjong.structural_efficiency`
（Issue #177）のcanonical semanticをsingle sourceとして再利用することと、
second-step materialization contractのavailability semanticsを固定する。

structural semanticそのものをtest側で別実装しない。期待値は常に
`lisjong.structural_efficiency`のsupported APIから導く。
"""

import ast
import inspect
import unittest
from unittest.mock import patch

import lisjong.learning.candidate_features as candidate_features
from lisjong.learning import (
    CANDIDATE_FEATURE_IDENTITY,
    DATASET_SCHEMA,
    FEATURE_IDENTITY,
    MODEL_ARTIFACT_SCHEMA,
    TEACHER_LABEL_SEMANTICS,
    CandidateFeatureError,
    DiscardCandidateFeatures,
    SecondStepStatus,
    build_discard_candidate_features,
    legal_discard_candidates,
)
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import (
    DecisionContext,
    Discard,
    DiscardAction,
    InternalAction,
    OwnHandState,
    PassAction,
    PlayerPublicState,
    PolicyInput,
    RiichiAction,
    RiichiState,
    RoundState,
    Seat,
    Tile,
    TileCategory,
    TileType,
    Wind,
)
from lisjong.structural_efficiency import (
    StructuralShantenEvaluator,
    discard_action_sort_key,
    known_tile_counts,
    post_discard_concealed_hand,
    second_step_ukeire_score,
    ukeire_count,
)

_CATEGORIES = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}


def _hand(spec: str) -> tuple[Tile, ...]:
    """`123m456p`形式のhand specをTile tupleへ変換する。`0`は赤5である。"""
    tiles: list[Tile] = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        category = _CATEGORIES[character]
        for rank_character in ranks:
            rank = int(rank_character)
            tiles.append(
                Tile(TileType(category, 5 if rank == 0 else rank), is_red=rank == 0)
            )
        ranks = ""
    if ranks:
        raise ValueError(f"hand spec has trailing ranks: {spec!r}")
    return tuple(tiles)


def _policy_input(
    concealed: tuple[Tile, ...],
    opponent_discards: tuple[str, str, str] = ("", "", ""),
) -> PolicyInput:
    players = [
        PlayerPublicState(score=25000, discards=(), melds=(), riichi=RiichiState.NONE)
    ]
    for spec in opponent_discards:
        players.append(
            PlayerPublicState(
                score=25000,
                discards=tuple(
                    Discard(tile=tile, tsumogiri=False, order=order, called_by=None)
                    for order, tile in enumerate(_hand(spec))
                ),
                melds=(),
                riichi=RiichiState.NONE,
            )
        )
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(Tile(TileType(TileCategory.HONOR, 6)),),
            live_wall_tiles_remaining=50,
        ),
        players=tuple(players),
        own_hand=OwnHandState(concealed_tiles=concealed, drawn_tile=concealed[-1]),
    )


def _discard(tile: Tile, *, tsumogiri: bool = False) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=tsumogiri)


def _decision(
    hand_spec: str,
    *,
    opponent_discards: tuple[str, str, str] = ("", "", ""),
    extra_actions: tuple[InternalAction, ...] = (),
    discard_actions: tuple[DiscardAction, ...] | None = None,
) -> DecisionContext:
    """純手牌specから通常打牌decisionを作る。

    `discard_actions`未指定時は、手牌に実在する牌identityごとに手出し
    `DiscardAction`を1つずつ作る。
    """
    concealed = _hand(hand_spec)
    if discard_actions is None:
        discard_actions = tuple(_discard(tile) for tile in dict.fromkeys(concealed))
    return DecisionContext(
        input=_policy_input(concealed, opponent_discards),
        legal_actions=discard_actions + extra_actions,
    )


_TWO_STEP_HAND = "345m56679s333577z"
"""#177 testが使う、2段目だけが異なる1向聴の14枚。"""

_WIDE_HAND = "234m5689p345s1177z"
"""打牌候補ごとにpost-discard向聴数が1と2へ割れる14枚。"""

_ZERO_SECOND_STEP_HAND = "1133557799m123p4p"
"""4p切りで1向聴かつ有効牌が全て場に尽き、2段階受け入れが真に0になる14枚。"""

_ZERO_SECOND_STEP_DISCARDS = ("123p", "123p", "123p")
"""`_ZERO_SECOND_STEP_HAND`の有効牌1p/2p/3pを残り0枚にするPolicy-visibleな捨て牌。"""

_TENPAI_HAND = "123456789m111p23p"
"""全打牌候補が打牌後聴牌になる14枚。"""


def _expected(decision: DecisionContext, action: DiscardAction) -> tuple[int, int, int]:
    """canonical semanticから期待値（shanten / current ukeire / 2段目）を導く。"""
    policy_input = decision.input
    known_counts = known_tile_counts(policy_input)
    evaluator = StructuralShantenEvaluator()
    post_discard_hand = post_discard_concealed_hand(
        policy_input.own_hand.concealed_tiles, action.tile
    )
    shanten = evaluator.calculate(post_discard_hand)
    return (
        shanten,
        ukeire_count(post_discard_hand, known_counts, shanten, evaluator),
        second_step_ukeire_score(post_discard_hand, known_counts, shanten, evaluator),
    )


class LegalDiscardCandidateEnumerationTest(unittest.TestCase):
    """legal discard candidateの抽出と並びを固定する。"""

    def test_every_legal_discard_appears_exactly_once(self) -> None:
        decision = _decision(_TWO_STEP_HAND)
        expected = [
            action
            for action in decision.legal_actions
            if isinstance(action, DiscardAction)
        ]

        features = build_discard_candidate_features(decision)

        self.assertEqual(len(features), len(expected))
        self.assertCountEqual([feature.action for feature in features], expected)

    def test_non_discard_legal_actions_are_excluded(self) -> None:
        decision = _decision(
            _TWO_STEP_HAND,
            extra_actions=(
                RiichiAction(actor=Seat.SEAT_0),
                PassAction(actor=Seat.SEAT_0),
            ),
        )

        features = build_discard_candidate_features(decision)

        self.assertTrue(
            all(isinstance(feature.action, DiscardAction) for feature in features)
        )
        self.assertEqual(
            len(features),
            sum(
                1
                for action in decision.legal_actions
                if isinstance(action, DiscardAction)
            ),
        )

    def test_result_is_independent_of_legal_action_input_order(self) -> None:
        concealed = _hand(_TWO_STEP_HAND)
        policy_input = _policy_input(concealed)
        discards = tuple(_discard(tile) for tile in dict.fromkeys(concealed))

        forward = build_discard_candidate_features(
            DecisionContext(input=policy_input, legal_actions=discards)
        )
        reversed_order = build_discard_candidate_features(
            DecisionContext(input=policy_input, legal_actions=tuple(reversed(discards)))
        )

        self.assertEqual(forward, reversed_order)

    def test_candidates_are_in_canonical_order(self) -> None:
        decision = _decision(_TWO_STEP_HAND)

        features = build_discard_candidate_features(decision)

        actions = [feature.action for feature in features]
        self.assertEqual(actions, sorted(actions, key=discard_action_sort_key))

    def test_canonical_order_tie_breaks_red_five_and_tsumogiri(self) -> None:
        concealed = _hand("055m")
        normal_five = Tile(TileType(TileCategory.MANZU, 5))
        red_five = Tile(TileType(TileCategory.MANZU, 5), is_red=True)
        discards = (
            _discard(normal_five, tsumogiri=True),
            _discard(red_five),
            _discard(normal_five),
        )

        candidates = legal_discard_candidates(
            DecisionContext(input=_policy_input(concealed), legal_actions=discards)
        )

        self.assertEqual(
            candidates,
            (
                _discard(normal_five),
                _discard(normal_five, tsumogiri=True),
                _discard(red_five),
            ),
        )

    def test_decision_without_discard_candidates_fails_closed(self) -> None:
        decision = DecisionContext(
            input=_policy_input(_hand(_TWO_STEP_HAND)),
            legal_actions=(PassAction(actor=Seat.SEAT_0),),
        )

        with self.assertRaises(CandidateFeatureError):
            build_discard_candidate_features(decision)


class CandidateActionIdentityTest(unittest.TestCase):
    """candidateがcanonical legal action identityを維持することを固定する。"""

    def test_returned_action_is_the_original_legal_action_object(self) -> None:
        decision = _decision(_TWO_STEP_HAND)

        features = build_discard_candidate_features(decision)

        for feature in features:
            self.assertTrue(
                any(action is feature.action for action in decision.legal_actions),
                feature.action,
            )

    def test_red_five_and_normal_five_stay_distinct_candidates(self) -> None:
        concealed = _hand("055678m234p345s77z")
        normal_five = Tile(TileType(TileCategory.MANZU, 5))
        red_five = Tile(TileType(TileCategory.MANZU, 5), is_red=True)
        self.assertIn(normal_five, concealed)
        self.assertIn(red_five, concealed)
        decision = DecisionContext(
            input=_policy_input(concealed),
            legal_actions=tuple(_discard(tile) for tile in dict.fromkeys(concealed)),
        )

        features = build_discard_candidate_features(decision)

        five_candidates = [
            feature.action.tile
            for feature in features
            if feature.action.tile.tile_type == normal_five.tile_type
        ]
        self.assertEqual(five_candidates, [normal_five, red_five])

    def test_tsumogiri_identity_is_preserved(self) -> None:
        concealed = _hand(_TWO_STEP_HAND)
        drawn = concealed[-1]
        discards = tuple(_discard(tile) for tile in dict.fromkeys(concealed)) + (
            _discard(drawn, tsumogiri=True),
        )
        decision = DecisionContext(
            input=_policy_input(concealed), legal_actions=discards
        )

        features = build_discard_candidate_features(decision)

        tsumogiri = [feature.action for feature in features if feature.action.tsumogiri]
        self.assertEqual(tsumogiri, [_discard(drawn, tsumogiri=True)])


class CanonicalStructuralSemanticsTest(unittest.TestCase):
    """candidate featureが#177のcanonical semanticと一致することを固定する。"""

    def test_shanten_and_ukeire_match_the_canonical_component(self) -> None:
        decision = _decision(_WIDE_HAND)

        features = build_discard_candidate_features(decision)

        for feature in features:
            expected_shanten, expected_ukeire, _ = _expected(decision, feature.action)
            self.assertEqual(feature.post_discard_shanten, expected_shanten)
            self.assertEqual(feature.current_ukeire_count, expected_ukeire)

    def test_current_ukeire_is_materialized_for_every_candidate(self) -> None:
        decision = _decision(_WIDE_HAND)

        features = build_discard_candidate_features(decision)

        self.assertTrue(
            all(type(feature.current_ukeire_count) is int for feature in features)
        )

    def test_second_step_score_matches_the_canonical_component(self) -> None:
        decision = _decision(_TWO_STEP_HAND)
        candidates = legal_discard_candidates(decision)

        features = build_discard_candidate_features(
            decision, second_step_actions=candidates
        )

        evaluated = [
            feature
            for feature in features
            if feature.second_step_status is SecondStepStatus.EVALUATED
        ]
        self.assertTrue(evaluated)
        for feature in evaluated:
            _, _, expected_second_step = _expected(decision, feature.action)
            self.assertEqual(feature.second_step_ukeire_score, expected_second_step)


class SecondStepAvailabilityTest(unittest.TestCase):
    """evaluated zero / not materialized / not applicableの区別を固定する。"""

    def test_unrequested_candidates_are_not_materialized(self) -> None:
        decision = _decision(_TWO_STEP_HAND)

        features = build_discard_candidate_features(decision)

        self.assertTrue(
            all(
                feature.second_step_status is SecondStepStatus.NOT_MATERIALIZED
                and feature.second_step_ukeire_score is None
                for feature in features
            )
        )

    def test_evaluated_zero_is_distinct_from_not_materialized(self) -> None:
        decision = _decision(
            _ZERO_SECOND_STEP_HAND, opponent_discards=_ZERO_SECOND_STEP_DISCARDS
        )
        requested = _discard(Tile(TileType(TileCategory.PINZU, 4)))
        expected_shanten, expected_ukeire, expected_second_step = _expected(
            decision, requested
        )
        self.assertEqual(
            (expected_shanten, expected_ukeire, expected_second_step), (1, 0, 0)
        )

        features = build_discard_candidate_features(
            decision, second_step_actions=(requested,)
        )

        evaluated = next(feature for feature in features if feature.action == requested)
        self.assertEqual(evaluated.second_step_status, SecondStepStatus.EVALUATED)
        self.assertEqual(evaluated.second_step_ukeire_score, 0)
        others = [feature for feature in features if feature.action != requested]
        self.assertTrue(
            all(
                feature.second_step_status is SecondStepStatus.NOT_MATERIALIZED
                for feature in others
            )
        )

    def test_post_discard_tenpai_is_not_applicable(self) -> None:
        decision = _decision(_TENPAI_HAND)
        candidates = legal_discard_candidates(decision)

        features = build_discard_candidate_features(
            decision, second_step_actions=candidates
        )

        tenpai = [feature for feature in features if feature.post_discard_shanten <= 0]
        self.assertTrue(tenpai)
        for feature in tenpai:
            self.assertEqual(
                feature.second_step_status, SecondStepStatus.NOT_APPLICABLE
            )
            self.assertIsNone(feature.second_step_ukeire_score)

    def test_not_applicable_candidates_are_never_evaluated(self) -> None:
        decision = _decision(_TENPAI_HAND)
        candidates = legal_discard_candidates(decision)

        with patch.object(candidate_features, "second_step_ukeire_score") as evaluator:
            features = build_discard_candidate_features(
                decision, second_step_actions=candidates
            )

        evaluator.assert_not_called()
        self.assertTrue(
            all(
                feature.second_step_status is SecondStepStatus.NOT_APPLICABLE
                for feature in features
            )
        )

    def test_score_must_be_absent_unless_evaluated(self) -> None:
        action = _discard(Tile(TileType(TileCategory.MANZU, 1)))

        with self.assertRaises(ValueError):
            DiscardCandidateFeatures(
                action=action,
                post_discard_shanten=1,
                current_ukeire_count=4,
                second_step_ukeire_score=0,
                second_step_status=SecondStepStatus.NOT_MATERIALIZED,
            )

    def test_evaluated_status_requires_a_score(self) -> None:
        action = _discard(Tile(TileType(TileCategory.MANZU, 1)))

        with self.assertRaises(TypeError):
            DiscardCandidateFeatures(
                action=action,
                post_discard_shanten=1,
                current_ukeire_count=4,
                second_step_ukeire_score=None,
                second_step_status=SecondStepStatus.EVALUATED,
            )


class SelectiveSecondStepMaterializationTest(unittest.TestCase):
    """selective second-step materialization contractを固定する。"""

    def test_only_requested_candidates_call_the_second_step_evaluator(self) -> None:
        decision = _decision(_TWO_STEP_HAND)
        candidates = legal_discard_candidates(decision)
        requested = (candidates[0], candidates[-1])
        applicable = {
            feature.action
            for feature in build_discard_candidate_features(decision)
            if feature.post_discard_shanten >= 1
        }
        self.assertTrue(set(requested) <= applicable)

        with patch.object(
            candidate_features,
            "second_step_ukeire_score",
            wraps=second_step_ukeire_score,
        ) as evaluator:
            features = build_discard_candidate_features(
                decision, second_step_actions=requested
            )

        self.assertEqual(evaluator.call_count, len(requested))
        evaluated = {
            feature.action
            for feature in features
            if feature.second_step_status is SecondStepStatus.EVALUATED
        }
        self.assertEqual(evaluated, set(requested))

    def test_no_request_evaluates_no_second_step(self) -> None:
        decision = _decision(_TWO_STEP_HAND)

        with patch.object(candidate_features, "second_step_ukeire_score") as evaluator:
            build_discard_candidate_features(decision)

        evaluator.assert_not_called()

    def test_unknown_action_request_fails_closed(self) -> None:
        decision = _decision(_TWO_STEP_HAND)
        foreign = _discard(Tile(TileType(TileCategory.PINZU, 1)))
        self.assertNotIn(foreign, decision.legal_actions)

        with self.assertRaises(CandidateFeatureError):
            build_discard_candidate_features(decision, second_step_actions=(foreign,))

    def test_request_for_another_actor_fails_closed(self) -> None:
        decision = _decision(_TWO_STEP_HAND)
        candidate = legal_discard_candidates(decision)[0]
        other_actor = DiscardAction(
            actor=Seat.SEAT_1, tile=candidate.tile, tsumogiri=candidate.tsumogiri
        )

        with self.assertRaises(CandidateFeatureError):
            build_discard_candidate_features(
                decision, second_step_actions=(other_actor,)
            )

    def test_non_discard_action_request_is_rejected(self) -> None:
        decision = _decision(_TWO_STEP_HAND)

        with self.assertRaises(TypeError):
            build_discard_candidate_features(
                decision, second_step_actions=(RiichiAction(actor=Seat.SEAT_0),)
            )

    def test_selection_resolves_to_the_canonical_action_object(self) -> None:
        decision = _decision(_TWO_STEP_HAND)
        candidate = next(
            feature.action
            for feature in build_discard_candidate_features(decision)
            if feature.post_discard_shanten >= 1
        )
        equivalent = DiscardAction(
            actor=candidate.actor,
            tile=candidate.tile,
            tsumogiri=candidate.tsumogiri,
        )
        self.assertIsNot(equivalent, candidate)

        features = build_discard_candidate_features(
            decision, second_step_actions=(equivalent,)
        )

        evaluated = [
            feature
            for feature in features
            if feature.second_step_status is SecondStepStatus.EVALUATED
        ]
        self.assertEqual(len(evaluated), 1)
        self.assertTrue(
            any(action is evaluated[0].action for action in decision.legal_actions)
        )


class DecisionLocalReuseTest(unittest.TestCase):
    """decision-localなsemantic reuseを固定する。"""

    def test_known_tile_counts_is_computed_once_per_build(self) -> None:
        decision = _decision(_TWO_STEP_HAND)

        with patch.object(
            candidate_features, "known_tile_counts", wraps=known_tile_counts
        ) as counter:
            build_discard_candidate_features(decision)

        self.assertEqual(counter.call_count, 1)

    def test_one_shanten_evaluator_is_shared_by_every_candidate(self) -> None:
        decision = _decision(_TWO_STEP_HAND)

        with patch.object(
            candidate_features,
            "StructuralShantenEvaluator",
            wraps=StructuralShantenEvaluator,
        ) as evaluator:
            build_discard_candidate_features(decision)

        self.assertEqual(evaluator.call_count, 1)


class InformationBoundaryTest(unittest.TestCase):
    """candidate builderがPolicy-visible情報だけへ依存することを固定する。"""

    def test_module_does_not_import_environments_or_arena(self) -> None:
        tree = ast.parse(inspect.getsource(candidate_features))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        self.assertFalse(
            any(
                module == prefix or module.startswith(f"{prefix}.")
                for module in imported
                for prefix in (
                    "lisjong_arena",
                    "lisjong_engine",
                    "mahjong",
                    "riichienv",
                    "websockets",
                )
            ),
            imported,
        )

    def test_module_does_not_reference_hidden_or_privileged_state(self) -> None:
        tree = ast.parse(inspect.getsource(candidate_features))
        referenced = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

        for forbidden in (
            "live_wall_tiles_remaining",
            "dead_wall",
            "wall",
            "GameTrace",
            "RiichiEnv",
            "RiichiLab",
            "hands",
            "belief",
        ):
            self.assertNotIn(forbidden, referenced)

    def test_module_only_reads_the_decision_context_contract(self) -> None:
        tree = ast.parse(inspect.getsource(candidate_features))
        decision_reads = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "decision"
        }

        self.assertEqual(decision_reads, {"input", "legal_actions"})


class ExistingContractPreservationTest(unittest.TestCase):
    """既存のLearning L0 / TwoStep contractを変えないことを固定する。"""

    def test_candidate_feature_identity_is_separate_from_existing_l0_identities(
        self,
    ) -> None:
        self.assertEqual(
            CANDIDATE_FEATURE_IDENTITY,
            "lisjong-offense-l0.1-discard-candidate-feature-v1",
        )
        self.assertEqual(FEATURE_IDENTITY, "lisjong-offense-l0-player-safe-feature-v1")
        self.assertEqual(DATASET_SCHEMA, "lisjong-offense-l0-bc-dataset-v1")
        self.assertEqual(
            MODEL_ARTIFACT_SCHEMA, "lisjong-offense-l0-bc-model-artifact-v1"
        )
        self.assertEqual(
            TEACHER_LABEL_SEMANTICS, "lisjong-offense-l0-teacher-selected-action-v1"
        )

    def test_two_step_selection_is_unaffected_by_candidate_feature_building(
        self,
    ) -> None:
        policy = TwoStepUkeirePolicy()
        for hand_spec in (_TWO_STEP_HAND, _WIDE_HAND, _TENPAI_HAND):
            with self.subTest(hand_spec=hand_spec):
                decision = _decision(hand_spec)

                before = policy.choose_action(decision)
                build_discard_candidate_features(
                    decision,
                    second_step_actions=legal_discard_candidates(decision),
                )
                after = policy.choose_action(decision)

                self.assertIs(before, after)
                self.assertIn(before, decision.legal_actions)

    def test_two_step_staged_snapshot_is_not_the_candidate_feature_view(self) -> None:
        decision = _decision(_WIDE_HAND)

        analysis = TwoStepUkeirePolicy().choose_action_with_analysis(decision).analysis
        features = build_discard_candidate_features(decision)

        self.assertTrue(
            any(
                evaluation.current_ukeire_count is None
                for evaluation in analysis.candidate_evaluations
            ),
            "TwoStepはminimum-shanten候補だけcurrent ukeireを評価する",
        )
        self.assertTrue(
            all(type(feature.current_ukeire_count) is int for feature in features),
            "candidate viewは全candidateでcurrent ukeireをmaterializeする",
        )
        self.assertEqual(
            [evaluation.action for evaluation in analysis.candidate_evaluations],
            [feature.action for feature in features],
        )


if __name__ == "__main__":
    unittest.main()
