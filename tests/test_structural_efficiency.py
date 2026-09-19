"""Issue #177 `lisjong.structural_efficiency`のunit test。

reusable structural-efficiency semanticを、concrete Policy classを経由せずに
直接固定する。Policy世代ごとのselection behaviorは各Policyのtestが所有する。
"""

import ast
import inspect
import itertools
import pathlib
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import lisjong.structural_efficiency as structural
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind
from lisjong.structural_efficiency import (
    PostDiscardStructuralEvaluation,
    StructuralEfficiencyError,
    StructuralShantenEvaluator,
    discard_action_sort_key,
    effective_tile_types,
    evaluate_post_discard_hands,
    known_tile_counts,
    post_discard_concealed_hand,
    second_step_ukeire_score,
    ukeire_count,
)


def _tile(category: TileCategory, rank: int, *, red: bool = False) -> Tile:
    return Tile(TileType(category, rank), is_red=red)


def _hand(spec: str) -> tuple[Tile, ...]:
    categories = {
        "m": TileCategory.MANZU,
        "p": TileCategory.PINZU,
        "s": TileCategory.SOUZU,
        "z": TileCategory.HONOR,
    }
    tiles: list[Tile] = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        category = categories[character]
        for rank_character in ranks:
            rank = int(rank_character)
            tiles.append(_tile(category, 5 if rank == 0 else rank, red=rank == 0))
        ranks = ""
    if ranks:
        raise ValueError(f"hand spec has trailing ranks: {spec!r}")
    return tuple(tiles)


MANZU_1 = _tile(TileCategory.MANZU, 1)
MANZU_2 = _tile(TileCategory.MANZU, 2)
MANZU_5 = _tile(TileCategory.MANZU, 5)
MANZU_5_RED = _tile(TileCategory.MANZU, 5, red=True)
MANZU_6 = _tile(TileCategory.MANZU, 6)
MANZU_7 = _tile(TileCategory.MANZU, 7)
PINZU_1 = _tile(TileCategory.PINZU, 1)
PINZU_5 = _tile(TileCategory.PINZU, 5)
SOUZU_9 = _tile(TileCategory.SOUZU, 9)
WHITE_DRAGON = _tile(TileCategory.HONOR, 5)

MANZU_1_TYPE = MANZU_1.tile_type
MANZU_2_TYPE = MANZU_2.tile_type
MANZU_5_TYPE = MANZU_5.tile_type

_TWO_STEP_HAND = _hand("345m56679s333577z")
"""9s/5z切りが同じ1向聴・受け入れ21で、2段目だけが異なる14枚。"""

_NINE_MANZU_HAND = _hand("123456789m111p23p")
"""2m/4m切りが同向聴で、4m切りのcurrent受け入れが多い14枚。"""


def _player(
    discards: tuple[Discard, ...] = (), melds: tuple[PublicMeld, ...] = ()
) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=discards,
        melds=melds,
        riichi=RiichiState.NONE,
    )


def _discard_history(tiles: tuple[Tile, ...]) -> tuple[Discard, ...]:
    return tuple(
        Discard(tile=tile, tsumogiri=False, order=order, called_by=None)
        for order, tile in enumerate(tiles)
    )


def _make_input(
    concealed_tiles: tuple[Tile, ...],
    *,
    players: tuple[PlayerPublicState, ...] | None = None,
    dora_indicators: tuple[Tile, ...] = (),
) -> PolicyInput:
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=dora_indicators,
            live_wall_tiles_remaining=70,
        ),
        players=players if players is not None else (_player(),) * 4,
        own_hand=OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=None),
    )


def _discard(tile: Tile, *, tsumogiri: bool = False) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=tsumogiri)


class PostDiscardConcealedHandTest(unittest.TestCase):
    """actual discard identityから打牌後純手牌を導出するsemanticを固定する。"""

    def test_removes_exactly_one_matching_tile(self) -> None:
        concealed = (MANZU_1, MANZU_1, MANZU_1, MANZU_2)

        remaining = post_discard_concealed_hand(concealed, MANZU_1)

        self.assertEqual(remaining, (MANZU_1, MANZU_1, MANZU_2))

    def test_result_is_an_immutable_tuple(self) -> None:
        remaining = post_discard_concealed_hand((MANZU_1, MANZU_2), MANZU_1)

        self.assertIsInstance(remaining, tuple)

    def test_normal_five_discard_keeps_the_red_five(self) -> None:
        concealed = (MANZU_5, MANZU_5_RED, MANZU_2)

        remaining = post_discard_concealed_hand(concealed, MANZU_5)

        self.assertEqual(remaining, (MANZU_5_RED, MANZU_2))

    def test_red_five_discard_keeps_the_normal_five(self) -> None:
        concealed = (MANZU_5, MANZU_5_RED, MANZU_2)

        remaining = post_discard_concealed_hand(concealed, MANZU_5_RED)

        self.assertEqual(remaining, (MANZU_5, MANZU_2))

    def test_missing_discard_identity_fails_closed(self) -> None:
        with self.assertRaises(StructuralEfficiencyError):
            post_discard_concealed_hand((MANZU_5, MANZU_2), MANZU_6)


class PolicyVisibleKnownTileCountTest(unittest.TestCase):
    """Policy-visibleな既知牌countingのsemanticを固定する。"""

    def test_known_counts_cover_every_policy_visible_source_once(self) -> None:
        called_discard = Discard(
            tile=PINZU_5,
            tsumogiri=False,
            order=0,
            called_by=Seat.SEAT_2,
        )
        uncalled_discard = Discard(
            tile=MANZU_5,
            tsumogiri=False,
            order=1,
            called_by=None,
        )
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(PINZU_5, PINZU_5, PINZU_5),
            from_seat=Seat.SEAT_1,
            called_tile=PINZU_5,
        )
        players = (
            _player(),
            _player((called_discard, uncalled_discard)),
            _player(melds=(pon,)),
            _player(),
        )
        policy_input = _make_input(
            (MANZU_5,),
            players=players,
            dora_indicators=(SOUZU_9,),
        )

        counts = known_tile_counts(policy_input)

        self.assertEqual(counts[MANZU_5_TYPE], 2)
        self.assertEqual(counts[PINZU_5.tile_type], 3)
        self.assertEqual(counts[SOUZU_9.tile_type], 1)

    def test_called_discard_is_not_counted_twice_with_its_meld(self) -> None:
        """同一物理牌はmeld側だけで数え、called discardでは数えない。"""
        called_discard = Discard(
            tile=PINZU_5,
            tsumogiri=False,
            order=0,
            called_by=Seat.SEAT_2,
        )
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(PINZU_5, PINZU_5, PINZU_5),
            from_seat=Seat.SEAT_1,
            called_tile=PINZU_5,
        )
        with_call = _make_input(
            (MANZU_2,),
            players=(
                _player(),
                _player((called_discard,)),
                _player(melds=(pon,)),
                _player(),
            ),
        )
        without_call = _make_input(
            (MANZU_2,),
            players=(_player(), _player(), _player(melds=(pon,)), _player()),
        )

        self.assertEqual(
            known_tile_counts(with_call)[PINZU_5.tile_type],
            known_tile_counts(without_call)[PINZU_5.tile_type],
        )

    def test_more_than_four_physical_copies_fails_closed(self) -> None:
        players = (
            _player(),
            _player(_discard_history((MANZU_5, MANZU_5, MANZU_5))),
            _player(),
            _player(),
        )
        policy_input = _make_input((MANZU_5, MANZU_5), players=players)

        with self.assertRaises(StructuralEfficiencyError):
            known_tile_counts(policy_input)

    def test_virtual_draw_adds_one_without_mutating_original_counts(self) -> None:
        original = {MANZU_1_TYPE: 2}

        updated = structural._known_counts_after_draw(original, MANZU_1_TYPE)

        self.assertEqual(original[MANZU_1_TYPE], 2)
        self.assertEqual(updated[MANZU_1_TYPE], 3)
        self.assertIsNot(updated, original)

    def test_drawing_a_fifth_visible_copy_fails_closed(self) -> None:
        with self.assertRaises(StructuralEfficiencyError):
            structural._known_counts_after_draw({MANZU_1_TYPE: 4}, MANZU_1_TYPE)


class PostDiscardStructuralEvaluationTest(unittest.TestCase):
    """post-discard structural evaluationがlow-level immutable valueであることを固定する。"""

    def test_evaluation_is_an_immutable_low_level_value(self) -> None:
        evaluation = PostDiscardStructuralEvaluation(
            action=_discard(SOUZU_9),
            post_discard_hand=(MANZU_1, MANZU_2),
            post_discard_shanten=1,
        )

        with self.assertRaises(FrozenInstanceError):
            evaluation.post_discard_shanten = 0

        self.assertIsInstance(evaluation.post_discard_hand, tuple)

    def test_evaluation_normalizes_its_hand_to_a_detached_tuple(self) -> None:
        hand = [MANZU_1, MANZU_2]

        evaluation = PostDiscardStructuralEvaluation(
            action=_discard(SOUZU_9),
            post_discard_hand=hand,
            post_discard_shanten=1,
        )
        hand.append(MANZU_5)

        self.assertEqual(evaluation.post_discard_hand, (MANZU_1, MANZU_2))

    def test_evaluation_rejects_wrong_types(self) -> None:
        action = _discard(SOUZU_9)
        with self.assertRaises(TypeError):
            PostDiscardStructuralEvaluation(SOUZU_9, (MANZU_1,), 1)
        with self.assertRaises(TypeError):
            PostDiscardStructuralEvaluation(action, (MANZU_1,), True)
        with self.assertRaises(TypeError):
            PostDiscardStructuralEvaluation(action, (MANZU_1_TYPE,), 1)

    def test_every_candidate_keeps_its_canonical_action_and_hand(self) -> None:
        policy_input = _make_input(_NINE_MANZU_HAND)
        discard_2m = _discard(MANZU_2)
        discard_1p = _discard(_tile(TileCategory.PINZU, 1))
        actions = (discard_2m, discard_1p)

        evaluations = evaluate_post_discard_hands(
            policy_input, actions, StructuralShantenEvaluator()
        )

        self.assertEqual(
            tuple(evaluation.action for evaluation in evaluations), actions
        )
        for evaluation, action in zip(evaluations, actions, strict=True):
            self.assertIs(evaluation.action, action)
            self.assertEqual(
                evaluation.post_discard_hand,
                post_discard_concealed_hand(_NINE_MANZU_HAND, action.tile),
            )
            self.assertEqual(
                evaluation.post_discard_shanten,
                calculate_shanten(evaluation.post_discard_hand),
            )

    def test_shanten_is_the_public_calculate_shanten_semantic(self) -> None:
        policy_input = _make_input(_NINE_MANZU_HAND)

        evaluations = evaluate_post_discard_hands(
            policy_input, (_discard(MANZU_2),), StructuralShantenEvaluator()
        )

        self.assertEqual(
            evaluations[0].post_discard_shanten,
            calculate_shanten(_hand("13456789m11123p")),
        )

    def test_red_and_normal_five_keep_distinct_action_identity(self) -> None:
        policy_input = _make_input((MANZU_5, MANZU_5_RED))
        actions = (_discard(MANZU_5), _discard(MANZU_5_RED))

        evaluations = evaluate_post_discard_hands(
            policy_input, actions, StructuralShantenEvaluator()
        )

        self.assertEqual(evaluations[0].post_discard_hand, (MANZU_5_RED,))
        self.assertEqual(evaluations[1].post_discard_hand, (MANZU_5,))
        self.assertEqual(
            evaluations[0].post_discard_shanten, evaluations[1].post_discard_shanten
        )

    def test_missing_discard_identity_fails_closed(self) -> None:
        policy_input = _make_input((MANZU_1, MANZU_2))

        with self.assertRaises(StructuralEfficiencyError):
            evaluate_post_discard_hands(
                policy_input, (_discard(MANZU_7),), StructuralShantenEvaluator()
            )


class EffectiveTileTypesAndUkeireTest(unittest.TestCase):
    """effective tile typesとcurrent ukeireのsemanticを固定する。"""

    def test_effective_tile_types_are_the_types_that_lower_shanten(self) -> None:
        hand = _hand("12356789m11123p")

        effective = effective_tile_types(hand)

        self.assertEqual(
            effective,
            (
                _tile(TileCategory.MANZU, 4).tile_type,
                MANZU_7.tile_type,
            ),
        )
        for tile_type in effective:
            self.assertLess(
                calculate_shanten([*hand, Tile(tile_type)]), calculate_shanten(hand)
            )

    def test_effective_tile_types_are_returned_in_canonical_order(self) -> None:
        hand = _hand("12356789m11123p")

        self.assertEqual(
            effective_tile_types(hand),
            tuple(
                tile_type
                for tile_type in structural._ALL_TILE_TYPES
                if tile_type in effective_tile_types(hand)
            ),
        )

    def test_ukeire_counts_policy_visible_remaining_copies(self) -> None:
        hand = _hand("12356789m11123p")
        known = known_tile_counts(_make_input(_NINE_MANZU_HAND))

        # 4m: 4 - 手牌1 = 3枚、7m: 4 - 手牌1 = 3枚。
        self.assertEqual(ukeire_count(hand, known), 6)

    def test_known_tiles_reduce_the_remaining_count(self) -> None:
        hand = _hand("12356789m11123p")
        players = (
            _player(),
            _player(_discard_history((_tile(TileCategory.MANZU, 4),) * 2 + (MANZU_7,))),
            _player(),
            _player(),
        )
        known = known_tile_counts(_make_input(_NINE_MANZU_HAND, players=players))

        self.assertEqual(ukeire_count(hand, known), 3)

    def test_effective_tile_type_with_no_remaining_copy_is_evaluated_zero(self) -> None:
        hand = _hand("13456789m11123p")
        players = (
            _player(),
            _player(_discard_history((MANZU_2, MANZU_2, MANZU_2))),
            _player(),
            _player(),
        )
        known = known_tile_counts(_make_input(_NINE_MANZU_HAND, players=players))

        self.assertEqual(effective_tile_types(hand), (MANZU_2_TYPE,))
        self.assertEqual(ukeire_count(hand, known), 0)

    def test_red_and_normal_five_share_the_same_structural_ukeire(self) -> None:
        normal = _hand("345m5667s333577z")
        red = _hand("340m5667s333577z")

        self.assertEqual(effective_tile_types(normal), effective_tile_types(red))
        self.assertEqual(ukeire_count(normal, {}), ukeire_count(red, {}))


class SecondStepUkeireScoreTest(unittest.TestCase):
    """`Σ remaining(t) * best_next_ukeire(t)`のexact integer semanticを固定する。"""

    def test_first_draw_remaining_count_is_the_integer_weight(self) -> None:
        known_counts = {MANZU_1_TYPE: 3, MANZU_2_TYPE: 1}

        def next_ukeire(_hand, tile_type, after_draw, _evaluator):
            expected_count = 4 if tile_type == MANZU_1_TYPE else 2
            self.assertEqual(after_draw[tile_type], expected_count)
            return 10 if tile_type == MANZU_1_TYPE else 2

        with (
            patch.object(
                structural,
                "effective_tile_types",
                return_value=(MANZU_1_TYPE, MANZU_2_TYPE),
            ),
            patch.object(structural, "_best_next_ukeire", side_effect=next_ukeire),
        ):
            score = second_step_ukeire_score(
                (PINZU_1,), known_counts, current_shanten=2
            )

        self.assertEqual(score, 1 * 10 + 3 * 2)

    def test_score_matches_the_exact_integer_sum_over_first_branches(self) -> None:
        hand = _hand("345m5667s333577z")
        known_counts = structural._count_tile_types(hand)

        expected = 0
        for tile_type in effective_tile_types(hand):
            remaining = 4 - known_counts.get(tile_type, 0)
            if remaining <= 0:
                continue
            expected += remaining * structural._best_next_ukeire(
                hand,
                tile_type,
                structural._known_counts_after_draw(known_counts, tile_type),
            )

        self.assertEqual(second_step_ukeire_score(hand, known_counts), expected)
        self.assertIsInstance(second_step_ukeire_score(hand, known_counts), int)

    def test_branch_minimizes_shanten_before_comparing_next_ukeire(self) -> None:
        def branch_shanten(hand):
            return 0 if hand[0].tile_type == MANZU_2_TYPE else 1

        with (
            patch.object(structural, "calculate_shanten", side_effect=branch_shanten),
            patch.object(structural, "ukeire_count", return_value=7) as ukeire,
        ):
            result = structural._best_next_ukeire(
                (MANZU_1,), MANZU_2_TYPE, {MANZU_2_TYPE: 1}
            )

        self.assertEqual(result, 7)
        self.assertEqual(ukeire.call_count, 1)
        self.assertEqual(ukeire.call_args.args[0], [MANZU_2])

    def test_same_branch_shanten_uses_maximum_next_ukeire(self) -> None:
        visited: list[tuple[Tile, ...]] = []

        def branch_ukeire(hand, _known_counts, _shanten, _evaluator):
            snapshot = tuple(hand)
            visited.append(snapshot)
            return 9 if snapshot == (MANZU_1,) else 3

        with (
            patch.object(structural, "calculate_shanten", return_value=0),
            patch.object(structural, "ukeire_count", side_effect=branch_ukeire),
        ):
            result = structural._best_next_ukeire(
                (MANZU_1,), MANZU_2_TYPE, {MANZU_2_TYPE: 1}
            )

        self.assertEqual(result, 9)
        self.assertEqual(set(visited), {(MANZU_1,), (MANZU_2,)})

    def test_virtual_discard_does_not_decrement_known_count(self) -> None:
        seen_counts: list[int] = []

        def capture_known(_hand, known_counts, _shanten, _evaluator):
            seen_counts.append(known_counts[MANZU_2_TYPE])
            return 1

        with (
            patch.object(structural, "calculate_shanten", return_value=0),
            patch.object(structural, "ukeire_count", side_effect=capture_known),
        ):
            structural._best_next_ukeire((MANZU_1,), MANZU_2_TYPE, {MANZU_2_TYPE: 2})

        self.assertEqual(seen_counts, [2, 2])

    def test_zero_remaining_effective_tile_is_not_expanded(self) -> None:
        with (
            patch.object(
                structural,
                "effective_tile_types",
                return_value=(MANZU_1_TYPE,),
            ),
            patch.object(structural, "_best_next_ukeire") as next_ukeire,
        ):
            score = second_step_ukeire_score(
                (PINZU_1,), {MANZU_1_TYPE: 4}, current_shanten=1
            )

        self.assertEqual(score, 0)
        next_ukeire.assert_not_called()

    def test_no_positive_first_branch_has_zero_score(self) -> None:
        with patch.object(structural, "effective_tile_types", return_value=()):
            self.assertEqual(
                second_step_ukeire_score((PINZU_1,), {}, current_shanten=1), 0
            )

    def test_closed_and_open_hands_use_n_to_n_plus_one_to_n(self) -> None:
        cases = (
            _hand("123456789m1p24s7z"),
            _hand("19m19p19s1234z"),
            _hand("1111p666z"),
            _hand("1111m"),
        )
        for hand in cases:
            known_counts = structural._count_tile_types(hand)
            with (
                self.subTest(size=len(hand)),
                patch.object(
                    structural, "calculate_shanten", wraps=calculate_shanten
                ) as shanten,
            ):
                score = second_step_ukeire_score(hand, known_counts)

                observed_sizes = {len(call.args[0]) for call in shanten.call_args_list}
                self.assertIsInstance(score, int)
                self.assertIn(len(hand), observed_sizes)
                self.assertIn(len(hand) + 1, observed_sizes)

    def test_red_and_normal_five_have_the_same_structural_score(self) -> None:
        normal = _hand("345m5667s333577z")
        red = _hand("340m5667s333577z")
        normal_known = structural._count_tile_types(normal)
        red_known = structural._count_tile_types(red)

        self.assertEqual(normal_known, red_known)
        self.assertEqual(
            second_step_ukeire_score(normal, normal_known),
            second_step_ukeire_score(red, red_known),
        )


class CanonicalDiscardOrderingTest(unittest.TestCase):
    """canonical `DiscardAction` orderingのdeterminismを固定する。"""

    def test_ordering_is_tile_order_then_tsumogiri(self) -> None:
        actions = (
            _discard(MANZU_1),
            _discard(MANZU_1, tsumogiri=True),
            _discard(MANZU_5),
            _discard(MANZU_5_RED),
            _discard(SOUZU_9),
            _discard(WHITE_DRAGON),
        )

        for permutation in itertools.permutations(actions):
            with self.subTest(permutation=permutation):
                self.assertEqual(
                    tuple(sorted(permutation, key=discard_action_sort_key)), actions
                )

    def test_ordering_does_not_depend_on_legal_action_input_order(self) -> None:
        actions = (_discard(SOUZU_9), _discard(WHITE_DRAGON), _discard(MANZU_2))

        results = {
            tuple(sorted(permutation, key=discard_action_sort_key))
            for permutation in itertools.permutations(actions)
        }

        self.assertEqual(len(results), 1)

    def test_evaluated_candidates_can_be_ordered_canonically(self) -> None:
        policy_input = _make_input(_TWO_STEP_HAND)
        actions = (_discard(WHITE_DRAGON), _discard(SOUZU_9))

        canonical = tuple(
            evaluation.action
            for evaluation in sorted(
                evaluate_post_discard_hands(
                    policy_input, actions, StructuralShantenEvaluator()
                ),
                key=lambda evaluation: discard_action_sort_key(evaluation.action),
            )
        )

        self.assertEqual(canonical, (actions[1], actions[0]))


class StructuralShantenEvaluatorTest(unittest.TestCase):
    """decision-local shanten memoizationがownership移動後も残ることを固定する。"""

    def test_decision_cache_reuses_red_and_normal_structural_hand(self) -> None:
        evaluator = StructuralShantenEvaluator()

        with patch.object(
            structural, "calculate_shanten", wraps=calculate_shanten
        ) as shanten:
            normal_result = evaluator.calculate((MANZU_1, MANZU_5))
            red_result = evaluator.calculate((MANZU_1, MANZU_5_RED))

        self.assertEqual(normal_result, red_result)
        self.assertEqual(shanten.call_count, 1)

    def test_cache_reuses_tile_order_permutations(self) -> None:
        evaluator = StructuralShantenEvaluator()

        with patch.object(
            structural, "calculate_shanten", wraps=calculate_shanten
        ) as shanten:
            first = evaluator.calculate((MANZU_1, MANZU_2, MANZU_5, MANZU_6))
            second = evaluator.calculate((MANZU_6, MANZU_5, MANZU_2, MANZU_1))

        self.assertEqual(first, second)
        self.assertEqual(shanten.call_count, 1)

    def test_post_discard_evaluation_shares_one_decision_local_cache(self) -> None:
        policy_input = _make_input((MANZU_5, MANZU_5_RED, MANZU_2))
        actions = (_discard(MANZU_5), _discard(MANZU_5_RED))
        evaluator = StructuralShantenEvaluator()

        with patch.object(
            structural, "calculate_shanten", wraps=calculate_shanten
        ) as shanten:
            evaluations = evaluate_post_discard_hands(policy_input, actions, evaluator)

        self.assertEqual(
            evaluations[0].post_discard_shanten, evaluations[1].post_discard_shanten
        )
        self.assertEqual(shanten.call_count, 1)

    def test_evaluator_keeps_no_cross_instance_state(self) -> None:
        first = StructuralShantenEvaluator()
        second = StructuralShantenEvaluator()
        first.calculate((MANZU_1, MANZU_5))

        with patch.object(
            structural, "calculate_shanten", wraps=calculate_shanten
        ) as shanten:
            second.calculate((MANZU_1, MANZU_5))

        self.assertEqual(shanten.call_count, 1)

    def test_internal_cache_is_not_part_of_the_supported_surface(self) -> None:
        self.assertFalse(
            any(
                name.startswith("cache") or name == "cache"
                for name in dir(StructuralShantenEvaluator)
                if not name.startswith("_")
            )
        )


class OwnershipBoundaryTest(unittest.TestCase):
    """componentがconcrete Policyやhidden informationへ依存しないことを固定する。"""

    def test_component_does_not_import_concrete_policies_or_environments(self) -> None:
        tree = ast.parse(inspect.getsource(structural))
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
                    "lisjong.policies",
                    "lisjong.belief",
                    "lisjong_arena",
                    "lisjong_engine",
                    "mahjong",
                    "riichienv",
                    "websockets",
                )
            ),
            imported,
        )

    def test_component_only_reads_policy_input_visible_state(self) -> None:
        tree = ast.parse(inspect.getsource(structural))
        referenced = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

        for forbidden in (
            "live_wall_tiles_remaining",
            "dead_wall",
            "wall",
            "GameTrace",
            "RiichiEnv",
            "hands",
            "belief",
        ):
            self.assertNotIn(forbidden, referenced)

        policy_input_reads = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "policy_input"
        }
        self.assertEqual(policy_input_reads, {"own_hand", "players", "round"})

    def test_supported_surface_is_not_re_exported_at_the_top_level(self) -> None:
        import lisjong
        import lisjong.policies

        package_source = inspect.getsource(lisjong)
        self.assertNotIn("structural_efficiency", package_source)

        for name in (
            "StructuralShantenEvaluator",
            "ukeire_count",
            "second_step_ukeire_score",
            "known_tile_counts",
        ):
            self.assertFalse(hasattr(lisjong, name), name)
            self.assertNotIn(name, lisjong.policies.__all__)


_PROMOTED_TWO_STEP_PRIVATE_NAMES = frozenset(
    {
        "_DecisionShantenEvaluator",
        "_discard_action_sort_key",
        "_known_tile_counts",
        "_remove_one_matching_tile",
        "_effective_tile_types",
        "_ukeire_count",
        "_second_step_score",
        "_evaluate_post_discard_hands",
        "_best_next_ukeire",
        "_known_counts_after_draw",
        "_count_tile_types",
    }
)
"""Issue #177でreusable componentへ昇格したsemanticの旧private名。"""


def _policy_module_sources() -> tuple[tuple[str, ast.Module], ...]:
    import lisjong.policies

    directory = pathlib.Path(lisjong.policies.__file__).parent
    return tuple(
        (path.name, ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.py"))
    )


class CrossPolicyPrivateImportTest(unittest.TestCase):
    """昇格semanticについてcross-Policy private importが復活しないことを固定する。"""

    def test_no_policy_module_imports_the_promoted_two_step_privates(self) -> None:
        offenders: list[str] = []
        for name, tree in _policy_module_sources():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module != "lisjong.policies.two_step_ukeire":
                    continue
                for alias in node.names:
                    if alias.name in _PROMOTED_TWO_STEP_PRIVATE_NAMES:
                        offenders.append(f"{name}: {alias.name}")

        self.assertEqual(offenders, [])

    def test_two_step_ukeire_no_longer_defines_the_promoted_semantics(self) -> None:
        import lisjong.policies.two_step_ukeire as two_step

        for name in _PROMOTED_TWO_STEP_PRIVATE_NAMES:
            self.assertFalse(hasattr(two_step, name), name)

    def test_promoted_semantics_are_imported_from_the_reusable_component(self) -> None:
        """旧private helperを使っていたconsumerが新componentへ依存する。"""
        expected_consumers = {
            "experimental_hand_belief_sensitivity.py",
            "finite_horizon_completion.py",
            "genbutsu_defense_finite_horizon_hand_value_aware.py",
            "genbutsu_defense_finite_horizon_value_aware.py",
            "genbutsu_defense_two_step_ukeire.py",
            "hand_value_aware_two_step_ukeire.py",
            "mechanism_riichi_defense_offensive_efficiency_diagnostic.py",
            "mechanism_riichi_defense_yakuhai_call.py",
            "targeted_honor_release_terminal_progression.py",
            "terminal_shanten_progression_mechanism_riichi_defense.py",
            "two_step_ukeire.py",
            "value_aware_two_step_ukeire.py",
        }
        observed = {
            name
            for name, tree in _policy_module_sources()
            if any(
                isinstance(node, ast.ImportFrom)
                and node.module == "lisjong.structural_efficiency"
                for node in ast.walk(tree)
            )
        }

        self.assertEqual(observed, expected_consumers)

    def test_reusable_component_is_not_imported_by_legacy_policies(self) -> None:
        """legacy Ukeire / Shanten Policyは世代独立性のため移行対象外である。"""
        legacy = {"shanten.py", "ukeire.py"}
        for name, tree in _policy_module_sources():
            if name not in legacy:
                continue
            self.assertFalse(
                any(
                    isinstance(node, ast.ImportFrom)
                    and node.module == "lisjong.structural_efficiency"
                    for node in ast.walk(tree)
                ),
                name,
            )


class TwoStepUkeireErrorCompatibilityTest(unittest.TestCase):
    """既存`TwoStepUkeirePolicyError`のcaller behaviorが変わらないことを固定する。"""

    def test_two_step_error_name_still_catches_structural_failures(self) -> None:
        from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicyError

        with self.assertRaises(TwoStepUkeirePolicyError):
            post_discard_concealed_hand((MANZU_1,), MANZU_7)

        self.assertIs(TwoStepUkeirePolicyError, StructuralEfficiencyError)

    def test_structural_error_stays_independent_of_legacy_policy_errors(self) -> None:
        from lisjong.policies.shanten import ShantenPolicyError
        from lisjong.policies.ukeire import UkeirePolicyError

        for legacy in (UkeirePolicyError, ShantenPolicyError):
            self.assertFalse(issubclass(StructuralEfficiencyError, legacy))
            self.assertFalse(issubclass(legacy, StructuralEfficiencyError))


if __name__ == "__main__":
    unittest.main()
