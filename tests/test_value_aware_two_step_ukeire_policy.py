"""Issue #107 `ValueAwareTwoStepUkeirePolicy`のunit test。"""

import ast
import inspect
import itertools
import pickle
import unittest
from dataclasses import FrozenInstanceError, fields
from unittest.mock import patch

import lisjong.policies.value_aware_two_step_ukeire as value_aware
from lisjong.policies import TwoStepUkeirePolicy, ValueAwareTwoStepUkeirePolicy
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicyError
from lisjong.policies.value_aware_two_step_ukeire import (
    ValueAwareTwoStepUkeireAnalysis,
    ValueAwareTwoStepUkeireCandidateEvaluation,
    _dora_tile_type,
    _evaluate_and_choose_discard,
    _retained_concealed_dora_count,
)
from lisjong.policy_contract.action import (
    DiscardAction,
    KyuushuKyuuhaiAction,
    PassAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.decision_trace import DecisionTraceRecorder
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_execution import (
    execute_policy,
    execute_policy_with_trace,
)
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind


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


EAST = _tile(TileCategory.HONOR, 1)
SOUTH = _tile(TileCategory.HONOR, 2)
WEST = _tile(TileCategory.HONOR, 3)
NORTH = _tile(TileCategory.HONOR, 4)
WHITE_DRAGON = _tile(TileCategory.HONOR, 5)
GREEN_DRAGON = _tile(TileCategory.HONOR, 6)
RED_DRAGON = _tile(TileCategory.HONOR, 7)
SOUZU_5 = _tile(TileCategory.SOUZU, 5)
SOUZU_5_RED = _tile(TileCategory.SOUZU, 5, red=True)
SOUZU_9 = _tile(TileCategory.SOUZU, 9)
PINZU_8 = _tile(TileCategory.PINZU, 8)
MANZU_2 = _tile(TileCategory.MANZU, 2)
MANZU_4 = _tile(TileCategory.MANZU, 4)

# discard candidates tie at post_discard_shanten=1, current_ukeire_count=28. The
# only structural difference is which tile is discarded: keeping the isolated 8p
# retains the red 5s, discarding the red 5s directly loses it.
_RED_FIVE_HAND = _hand("345m678m123p33z0s8p1s")

# Issue #58's TwoStepUkeire fixture, reused unmodified: 9s/5z discards tie at
# shanten=1, and (without dora indicators) second-step scores 122/126 decide.
_TWO_STEP_HAND = _hand("345m56679s333577z")

# 4 complete sets + 2 floating singles; discarding either floater keeps tenpai
# (tanki wait on the other) with identical ukeire.
_TENPAI_HAND = _hand("234m567m234p567p5s1z")

# 4 complete sets + a floating single (7z) vs. breaking a set (discarding 4m):
# the tenpai-preserving discard always wins regardless of dora.
_TANKI_VERSUS_ONE_SHANTEN_HAND = _hand("234m567m234p567p5s7z")

_NINE_MANZU_HAND = _hand("123456789m111p23p")
"""discard 4m keeps a strictly higher current ukeire than discard 2m."""

# East/South/North float alongside 3 complete sets + a pair; all three tie at
# shanten=1, current_ukeire_count=7.
_TRIPLE_HONOR_FLOAT_HAND = _hand("345m678m123p33z") + (EAST, SOUTH, NORTH)


def _player(
    discards: tuple[object, ...] = (), melds: tuple[object, ...] = ()
) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=discards,
        melds=melds,
        riichi=RiichiState.NONE,
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


def _decision(
    concealed_tiles: tuple[Tile, ...],
    actions: tuple[object, ...],
    *,
    dora_indicators: tuple[Tile, ...] = (),
) -> DecisionContext:
    return DecisionContext(
        input=_make_input(concealed_tiles, dora_indicators=dora_indicators),
        legal_actions=actions,
    )


def _discard(tile: Tile, *, tsumogiri: bool = False) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=tsumogiri)


class DoraTileTypeCycleTest(unittest.TestCase):
    """公開indicator -> actual doraのfull cycleをすべて固定する。"""

    def test_suited_number_cycle_wraps_nine_to_one(self) -> None:
        for category in (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU):
            with self.subTest(category=category):
                for rank in range(1, 9):
                    self.assertEqual(
                        _dora_tile_type(TileType(category, rank)),
                        TileType(category, rank + 1),
                    )
                self.assertEqual(
                    _dora_tile_type(TileType(category, 9)), TileType(category, 1)
                )

    def test_wind_cycle_is_east_south_west_north_east(self) -> None:
        expected = {
            EAST.tile_type: SOUTH.tile_type,
            SOUTH.tile_type: WEST.tile_type,
            WEST.tile_type: NORTH.tile_type,
            NORTH.tile_type: EAST.tile_type,
        }
        for indicator, dora in expected.items():
            with self.subTest(indicator=indicator):
                self.assertEqual(_dora_tile_type(indicator), dora)

    def test_dragon_cycle_is_white_green_red_white(self) -> None:
        expected = {
            WHITE_DRAGON.tile_type: GREEN_DRAGON.tile_type,
            GREEN_DRAGON.tile_type: RED_DRAGON.tile_type,
            RED_DRAGON.tile_type: WHITE_DRAGON.tile_type,
        }
        for indicator, dora in expected.items():
            with self.subTest(indicator=indicator):
                self.assertEqual(_dora_tile_type(indicator), dora)

    def test_red_five_indicator_derives_the_same_dora_as_normal_five(self) -> None:
        for category in (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU):
            with self.subTest(category=category):
                normal = TileType(category, 5)
                self.assertEqual(_dora_tile_type(normal), TileType(category, 6))
                # `_dora_tile_type()`はTileType(赤区分を含まない)だけを受け取るため、
                # 赤5 indicatorも通常5と全く同じ入力になる。

    def test_red_five_indicator_tile_derives_the_same_dora_end_to_end(self) -> None:
        # `_dora_tile_type()`単体ではなく、赤5の`Tile` indicatorを実際に
        # `_retained_concealed_dora_count()`へ通し、通常5 indicatorと同じ
        # actual dora(この場合pinzu6)を導出することをend-to-endで固定する。
        held_pinzu_6 = _tile(TileCategory.PINZU, 6)
        normal_five_indicator = _tile(TileCategory.PINZU, 5)
        red_five_indicator = _tile(TileCategory.PINZU, 5, red=True)

        self.assertEqual(
            _retained_concealed_dora_count((held_pinzu_6,), (normal_five_indicator,)),
            _retained_concealed_dora_count((held_pinzu_6,), (red_five_indicator,)),
        )
        self.assertEqual(
            _retained_concealed_dora_count((held_pinzu_6,), (red_five_indicator,)), 1
        )


class RetainedConcealedDoraCountTest(unittest.TestCase):
    def test_no_indicators_and_no_red_tiles_is_zero(self) -> None:
        self.assertEqual(_retained_concealed_dora_count((SOUZU_9, WHITE_DRAGON), ()), 0)

    def test_each_red_tile_counts_once(self) -> None:
        hand = (SOUZU_5_RED, MANZU_4, SOUZU_9)
        self.assertEqual(_retained_concealed_dora_count(hand, ()), 1)

        hand_with_two_reds = (SOUZU_5_RED, _tile(TileCategory.PINZU, 5, red=True))
        self.assertEqual(_retained_concealed_dora_count(hand_with_two_reds, ()), 2)

    def test_indicator_derived_dora_counts_per_held_tile(self) -> None:
        # indicator = red dragon -> actual dora = white dragon.
        hand = (WHITE_DRAGON, WHITE_DRAGON, SOUZU_9)
        self.assertEqual(_retained_concealed_dora_count(hand, (RED_DRAGON,)), 2)

    def test_multiple_indicators_for_the_same_dora_preserve_multiplicity(self) -> None:
        hand = (WHITE_DRAGON,)
        self.assertEqual(
            _retained_concealed_dora_count(hand, (RED_DRAGON, RED_DRAGON)), 2
        )
        hand_with_two_copies = (WHITE_DRAGON, WHITE_DRAGON)
        self.assertEqual(
            _retained_concealed_dora_count(
                hand_with_two_copies, (RED_DRAGON, RED_DRAGON)
            ),
            4,
        )

    def test_distinct_indicators_are_summed_independently(self) -> None:
        # indicator north(4z) -> dora east(1z); indicator east(1z) -> dora south(2z).
        hand = (EAST, SOUTH)
        self.assertEqual(_retained_concealed_dora_count(hand, (NORTH, EAST)), 2)

    def test_red_five_that_is_also_indicator_derived_dora_counts_as_two(self) -> None:
        # indicator manzu4 -> dora manzu5; the held tile is both red and that dora.
        hand = (_tile(TileCategory.MANZU, 5, red=True),)
        self.assertEqual(
            _retained_concealed_dora_count(hand, (_tile(TileCategory.MANZU, 4),)), 2
        )

    def test_unrelated_indicator_does_not_count_a_red_tile_twice(self) -> None:
        hand = (SOUZU_5_RED,)
        self.assertEqual(_retained_concealed_dora_count(hand, (WEST,)), 1)


class CandidateEvaluationValueTest(unittest.TestCase):
    def test_value_is_immutable_and_keeps_canonical_action(self) -> None:
        action = _discard(SOUZU_9)
        evaluation = ValueAwareTwoStepUkeireCandidateEvaluation(action, 1, 0, 0, None)

        self.assertIs(evaluation.action, action)
        self.assertEqual(
            tuple(field.name for field in fields(evaluation)),
            (
                "action",
                "post_discard_shanten",
                "current_ukeire_count",
                "retained_concealed_dora_count",
                "second_step_ukeire_score",
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            evaluation.retained_concealed_dora_count = 1

    def test_none_and_evaluated_zero_are_distinct(self) -> None:
        action = _discard(SOUZU_9)

        self.assertNotEqual(
            ValueAwareTwoStepUkeireCandidateEvaluation(action, 1, 0, None, None),
            ValueAwareTwoStepUkeireCandidateEvaluation(action, 1, 0, 0, None),
        )
        self.assertNotEqual(
            ValueAwareTwoStepUkeireCandidateEvaluation(action, 1, 0, 0, None),
            ValueAwareTwoStepUkeireCandidateEvaluation(action, 1, 0, 0, 0),
        )

    def test_value_rejects_wrong_types_without_accepting_bool_as_int(self) -> None:
        action = _discard(SOUZU_9)
        invalid_values = (
            ((object(), 1, None, None, None), "action"),
            ((action, True, None, None, None), "post_discard_shanten"),
            ((action, 1, False, None, None), "current_ukeire_count"),
            ((action, 1, None, False, None), "retained_concealed_dora_count"),
            ((action, 1, None, None, True), "second_step_ukeire_score"),
        )

        for arguments, field_name in invalid_values:
            with self.subTest(field_name=field_name), self.assertRaises(TypeError):
                ValueAwareTwoStepUkeireCandidateEvaluation(*arguments)


class CandidateEvaluationPipelineTest(unittest.TestCase):
    def test_shanten_stage_evaluates_all_candidates_but_not_later_loser_stages(
        self,
    ) -> None:
        break_tenpai = _discard(MANZU_4)
        keep_tenpai = _discard(RED_DRAGON)

        with (
            patch.object(
                value_aware,
                "_retained_concealed_dora_count",
                side_effect=AssertionError("shanten loser must not reach dora stage"),
            ),
            patch.object(
                value_aware,
                "_second_step_score",
                side_effect=AssertionError("shanten loser must not reach second step"),
            ),
        ):
            selected, evaluations = _evaluate_and_choose_discard(
                _make_input(_TANKI_VERSUS_ONE_SHANTEN_HAND),
                (keep_tenpai, break_tenpai),
            )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, keep_tenpai)
        self.assertEqual(by_action[break_tenpai].post_discard_shanten, 1)
        self.assertIsNone(by_action[break_tenpai].current_ukeire_count)
        self.assertIsNone(by_action[break_tenpai].retained_concealed_dora_count)
        self.assertIsNone(by_action[break_tenpai].second_step_ukeire_score)
        self.assertEqual(by_action[keep_tenpai].post_discard_shanten, 0)

    def test_current_ukeire_stage_selects_without_evaluating_dora(self) -> None:
        lower_current_ukeire = _discard(MANZU_2)
        higher_current_ukeire = _discard(MANZU_4)

        with (
            patch.object(
                value_aware,
                "_retained_concealed_dora_count",
                side_effect=AssertionError("unique current ukeire must end evaluation"),
            ),
            patch.object(
                value_aware,
                "_second_step_score",
                side_effect=AssertionError("unique current ukeire must end evaluation"),
            ),
        ):
            selected, evaluations = _evaluate_and_choose_discard(
                _make_input(_NINE_MANZU_HAND),
                (lower_current_ukeire, higher_current_ukeire),
            )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, higher_current_ukeire)
        self.assertGreater(
            by_action[higher_current_ukeire].current_ukeire_count,
            by_action[lower_current_ukeire].current_ukeire_count,
        )
        self.assertIsNone(by_action[lower_current_ukeire].retained_concealed_dora_count)
        self.assertIsNone(
            by_action[higher_current_ukeire].retained_concealed_dora_count
        )

    def test_red_five_retention_wins_a_tied_shanten_and_ukeire_candidate(self) -> None:
        discard_isolated = _discard(PINZU_8)
        discard_red_five = _discard(SOUZU_5_RED)

        selected, evaluations = _evaluate_and_choose_discard(
            _make_input(_RED_FIVE_HAND), (discard_isolated, discard_red_five)
        )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, discard_isolated)
        self.assertEqual(
            by_action[discard_isolated].current_ukeire_count,
            by_action[discard_red_five].current_ukeire_count,
        )
        self.assertEqual(by_action[discard_isolated].retained_concealed_dora_count, 1)
        self.assertEqual(by_action[discard_red_five].retained_concealed_dora_count, 0)
        # dora count alone resolves selection; second step is never reached.
        self.assertIsNone(by_action[discard_isolated].second_step_ukeire_score)
        self.assertIsNone(by_action[discard_red_five].second_step_ukeire_score)

    def test_indicator_derived_dora_retention_wins_a_tied_candidate(self) -> None:
        discard_keep_dora = _discard(SOUZU_9)
        discard_lose_dora = _discard(WHITE_DRAGON)

        selected, evaluations = _evaluate_and_choose_discard(
            _make_input(_TWO_STEP_HAND, dora_indicators=(RED_DRAGON,)),
            (discard_keep_dora, discard_lose_dora),
        )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, discard_keep_dora)
        self.assertEqual(by_action[discard_keep_dora].retained_concealed_dora_count, 1)
        self.assertEqual(by_action[discard_lose_dora].retained_concealed_dora_count, 0)

    def test_dora_count_beats_a_higher_second_step_score(self) -> None:
        # Without a dora indicator, plain TwoStep semantics prefer discard_white
        # (second-step 126 > 122). With the indicator, discard_9s has the lower
        # second-step score but the higher dora count, and wins instead.
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)

        no_indicator_selected, _ = _evaluate_and_choose_discard(
            _make_input(_TWO_STEP_HAND), (discard_9s, discard_white)
        )
        self.assertIs(no_indicator_selected, discard_white)

        selected, evaluations = _evaluate_and_choose_discard(
            _make_input(_TWO_STEP_HAND, dora_indicators=(RED_DRAGON,)),
            (discard_9s, discard_white),
        )
        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, discard_9s)
        self.assertGreater(
            by_action[discard_9s].retained_concealed_dora_count,
            by_action[discard_white].retained_concealed_dora_count,
        )
        self.assertIsNone(by_action[discard_9s].second_step_ukeire_score)
        self.assertIsNone(by_action[discard_white].second_step_ukeire_score)

    def test_dora_count_loser_does_not_reach_second_step(self) -> None:
        discard_east = _discard(EAST)
        discard_south = _discard(SOUTH)
        discard_north = _discard(NORTH)

        selected, evaluations = _evaluate_and_choose_discard(
            _make_input(_TRIPLE_HONOR_FLOAT_HAND, dora_indicators=(WEST,)),
            (discard_east, discard_south, discard_north),
        )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIn(selected, (discard_east, discard_south))
        self.assertEqual(by_action[discard_east].retained_concealed_dora_count, 1)
        self.assertEqual(by_action[discard_south].retained_concealed_dora_count, 1)
        self.assertEqual(by_action[discard_north].retained_concealed_dora_count, 0)
        self.assertIsNotNone(by_action[discard_east].second_step_ukeire_score)
        self.assertIsNotNone(by_action[discard_south].second_step_ukeire_score)
        self.assertIsNone(by_action[discard_north].second_step_ukeire_score)

    def test_equal_dora_count_uses_existing_second_step_semantics(self) -> None:
        discard_east = _discard(EAST)
        discard_south = _discard(SOUTH)
        discard_north = _discard(NORTH)

        selected, _ = _evaluate_and_choose_discard(
            _make_input(_TRIPLE_HONOR_FLOAT_HAND, dora_indicators=(WEST,)),
            (discard_east, discard_south, discard_north),
        )

        self.assertIs(selected, discard_east)

    def test_tenpai_compares_dora_count_but_never_evaluates_second_step(self) -> None:
        discard_keep_dora = _discard(SOUZU_5)
        discard_lose_dora = _discard(EAST)

        with patch.object(
            value_aware,
            "_second_step_score",
            side_effect=AssertionError("tenpai must not enter second step"),
        ):
            selected, evaluations = _evaluate_and_choose_discard(
                _make_input(_TENPAI_HAND, dora_indicators=(NORTH,)),
                (discard_keep_dora, discard_lose_dora),
            )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, discard_keep_dora)
        self.assertEqual(by_action[discard_keep_dora].post_discard_shanten, 0)
        self.assertEqual(by_action[discard_keep_dora].retained_concealed_dora_count, 1)
        self.assertEqual(by_action[discard_lose_dora].retained_concealed_dora_count, 0)

    def test_all_input_permutations_have_identical_canonical_snapshot(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        results = tuple(
            _evaluate_and_choose_discard(
                _make_input(_TWO_STEP_HAND, dora_indicators=(RED_DRAGON,)), actions
            )
            for actions in itertools.permutations((discard_9s, discard_white))
        )

        self.assertTrue(all(selected is discard_9s for selected, _ in results))
        self.assertTrue(all(snapshot == results[0][1] for _, snapshot in results))


class PolicyPriorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ValueAwareTwoStepUkeirePolicy()

    def test_winning_action_has_priority(self) -> None:
        discard = _discard(SOUZU_9)
        ron = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=EAST)
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=EAST)

        for actions in itertools.permutations((discard, ron, tsumo)):
            with self.subTest(actions=actions):
                self.assertEqual(
                    self.policy.choose_action(_decision(_TWO_STEP_HAND, actions)), ron
                )

    def test_riichi_has_priority_over_discard_evaluation(self) -> None:
        riichi = RiichiAction(actor=Seat.SEAT_0)
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)

        for actions in itertools.permutations((riichi, discard_9s, discard_white)):
            with self.subTest(actions=actions):
                self.assertEqual(
                    self.policy.choose_action(_decision(_TWO_STEP_HAND, actions)),
                    riichi,
                )

    def test_red_five_retention_at_the_policy_level(self) -> None:
        discard_isolated = _discard(PINZU_8)
        discard_red_five = _discard(SOUZU_5_RED)
        decision = _decision(_RED_FIVE_HAND, (discard_red_five, discard_isolated))

        self.assertEqual(self.policy.choose_action(decision), discard_isolated)

    def test_indicator_derived_dora_retention_at_the_policy_level(self) -> None:
        discard_keep_dora = _discard(SOUZU_9)
        discard_lose_dora = _discard(WHITE_DRAGON)
        decision = _decision(
            _TWO_STEP_HAND,
            (discard_lose_dora, discard_keep_dora),
            dora_indicators=(RED_DRAGON,),
        )

        self.assertEqual(self.policy.choose_action(decision), discard_keep_dora)

    def test_tenpai_dora_retention_at_the_policy_level(self) -> None:
        discard_keep_dora = _discard(SOUZU_5)
        discard_lose_dora = _discard(EAST)
        decision = _decision(
            _TENPAI_HAND,
            (discard_lose_dora, discard_keep_dora),
            dora_indicators=(NORTH,),
        )

        self.assertEqual(self.policy.choose_action(decision), discard_keep_dora)

    def test_pass_and_riichi_orchestration_matches_the_base_policy(self) -> None:
        pass_action = PassAction(actor=Seat.SEAT_0)
        kyuushu = KyuushuKyuuhaiAction(actor=Seat.SEAT_0)

        self.assertEqual(
            self.policy.choose_action(_decision((), (pass_action, kyuushu))),
            pass_action,
        )


class BaselinePreservationTest(unittest.TestCase):
    """`TwoStepUkeirePolicy`のselectionはValueAware追加後も変化しない。"""

    def test_two_step_policy_is_unaffected_by_the_new_generation(self) -> None:
        # Fixed expected actions, not a self-comparison: this pins TwoStep's
        # own pre-existing selection (already covered by
        # tests/test_two_step_ukeire_policy.py) so a future edit that changes
        # `TwoStepUkeirePolicy`'s behavior fails here too, alongside the
        # unmodified TwoStep test suite.
        scenarios = (
            (
                _TANKI_VERSUS_ONE_SHANTEN_HAND,
                (_discard(MANZU_4), _discard(RED_DRAGON)),
                _discard(RED_DRAGON),
            ),
            (
                _NINE_MANZU_HAND,
                (_discard(MANZU_2), _discard(MANZU_4)),
                _discard(MANZU_4),
            ),
            (
                _TWO_STEP_HAND,
                (_discard(SOUZU_9), _discard(WHITE_DRAGON)),
                _discard(WHITE_DRAGON),
            ),
        )
        for hand, actions, expected in scenarios:
            with self.subTest(actions=actions):
                self.assertEqual(
                    TwoStepUkeirePolicy().choose_action(_decision(hand, actions)),
                    expected,
                )

    def test_value_aware_degenerates_to_two_step_without_dora_information(
        self,
    ) -> None:
        scenarios = (
            (_TANKI_VERSUS_ONE_SHANTEN_HAND, (_discard(MANZU_4), _discard(RED_DRAGON))),
            (_NINE_MANZU_HAND, (_discard(MANZU_2), _discard(MANZU_4))),
            (_TWO_STEP_HAND, (_discard(SOUZU_9), _discard(WHITE_DRAGON))),
        )
        for hand, actions in scenarios:
            with self.subTest(actions=actions):
                self.assertEqual(
                    ValueAwareTwoStepUkeirePolicy().choose_action(
                        _decision(hand, actions)
                    ),
                    TwoStepUkeirePolicy().choose_action(_decision(hand, actions)),
                )


class SubclassStructureTest(unittest.TestCase):
    """`_decide_discard()`だけをoverrideし、上位orchestrationを複製しない。"""

    def test_only_decide_discard_is_overridden(self) -> None:
        self.assertIn("_decide_discard", vars(ValueAwareTwoStepUkeirePolicy))
        for method_name in (
            "_decide",
            "choose_action",
            "choose_action_with_analysis",
        ):
            with self.subTest(method_name=method_name):
                self.assertNotIn(method_name, vars(ValueAwareTwoStepUkeirePolicy))

    def test_value_aware_policy_is_a_two_step_subclass(self) -> None:
        self.assertTrue(issubclass(ValueAwareTwoStepUkeirePolicy, TwoStepUkeirePolicy))


class ValueAwareAnalysisValueTest(unittest.TestCase):
    def test_analysis_is_an_immutable_typed_payload(self) -> None:
        evaluations = (
            ValueAwareTwoStepUkeireCandidateEvaluation(
                _discard(SOUZU_9), 1, 0, 0, None
            ),
        )
        analysis = ValueAwareTwoStepUkeireAnalysis(candidate_evaluations=evaluations)

        self.assertIsInstance(analysis, AnalysisTrace)
        self.assertEqual(
            tuple(field.name for field in fields(analysis)), ("candidate_evaluations",)
        )
        self.assertIs(analysis.candidate_evaluations[0], evaluations[0])
        self.assertFalse(hasattr(analysis, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            analysis.candidate_evaluations = ()

    def test_analysis_normalizes_to_a_detached_tuple(self) -> None:
        evaluations = [
            ValueAwareTwoStepUkeireCandidateEvaluation(_discard(SOUZU_9), 1, 0, 0, None)
        ]

        analysis = ValueAwareTwoStepUkeireAnalysis(candidate_evaluations=evaluations)
        evaluations.clear()

        self.assertIsInstance(analysis.candidate_evaluations, tuple)
        self.assertEqual(len(analysis.candidate_evaluations), 1)

    def test_analysis_rejects_free_form_and_empty_payloads(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be an iterable"):
            ValueAwareTwoStepUkeireAnalysis(candidate_evaluations=7)
        with self.assertRaisesRegex(
            TypeError, "ValueAwareTwoStepUkeireCandidateEvaluation"
        ):
            ValueAwareTwoStepUkeireAnalysis(candidate_evaluations=({"dora": 1},))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            ValueAwareTwoStepUkeireAnalysis(candidate_evaluations=())


class ValueAwareDecisionAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ValueAwareTwoStepUkeirePolicy()

    def _traced(self, decision: DecisionContext):
        recorder = DecisionTraceRecorder()
        selected = execute_policy_with_trace(self.policy, decision, recorder)
        (trace,) = recorder.snapshot()
        return selected, trace

    def test_discard_branch_reuses_the_existing_candidate_evaluations(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        decision = _decision(
            _TWO_STEP_HAND, (discard_9s, discard_white), dora_indicators=(RED_DRAGON,)
        )
        expected_selected, expected_evaluations = _evaluate_and_choose_discard(
            _make_input(_TWO_STEP_HAND, dora_indicators=(RED_DRAGON,)),
            (discard_9s, discard_white),
        )

        selected, trace = self._traced(decision)

        self.assertIs(selected, expected_selected)
        self.assertIsInstance(trace.analysis, ValueAwareTwoStepUkeireAnalysis)
        self.assertEqual(trace.analysis.candidate_evaluations, expected_evaluations)

    def test_discard_evaluation_runs_exactly_once_for_a_traced_decision(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        decision = _decision(_TWO_STEP_HAND, (discard_9s, discard_white))

        with patch.object(
            value_aware,
            "_evaluate_and_choose_discard",
            wraps=value_aware._evaluate_and_choose_discard,
        ) as evaluate:
            self._traced(decision)

        self.assertEqual(evaluate.call_count, 1)

    def test_winning_branch_runs_no_discard_evaluation_and_reports_no_analysis(
        self,
    ) -> None:
        ron = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=EAST)
        decision = _decision(_TWO_STEP_HAND, (_discard(SOUZU_9), ron))

        with patch.object(
            value_aware,
            "_evaluate_and_choose_discard",
            side_effect=AssertionError("winning branch must not evaluate discards"),
        ):
            selected, trace = self._traced(decision)

        self.assertIs(selected, ron)
        self.assertIsNone(trace.analysis)

    def test_riichi_branch_runs_no_discard_evaluation_and_reports_no_analysis(
        self,
    ) -> None:
        riichi = RiichiAction(actor=Seat.SEAT_0)
        decision = _decision(_TWO_STEP_HAND, (_discard(SOUZU_9), riichi))

        with patch.object(
            value_aware,
            "_evaluate_and_choose_discard",
            side_effect=AssertionError("Always Riichi must not evaluate discards"),
        ):
            selected, trace = self._traced(decision)

        self.assertIs(selected, riichi)
        self.assertIsNone(trace.analysis)

    def test_pass_and_fallback_branches_report_no_analysis(self) -> None:
        pass_action = PassAction(actor=Seat.SEAT_0)
        kyuushu = KyuushuKyuuhaiAction(actor=Seat.SEAT_0)

        for actions, expected in (
            ((pass_action, kyuushu), pass_action),
            ((kyuushu,), kyuushu),
        ):
            with self.subTest(actions=actions):
                selected, trace = self._traced(_decision((), actions))

                self.assertIs(selected, expected)
                self.assertIsNone(trace.analysis)

    def test_none_and_evaluated_zero_stage_semantics_survive_the_analysis(self) -> None:
        discard_east = _discard(EAST)
        discard_south = _discard(SOUTH)
        discard_north = _discard(NORTH)
        decision = _decision(
            _TRIPLE_HONOR_FLOAT_HAND,
            (discard_east, discard_south, discard_north),
            dora_indicators=(WEST,),
        )

        selected, trace = self._traced(decision)

        by_action = {
            evaluation.action: evaluation
            for evaluation in trace.analysis.candidate_evaluations
        }
        self.assertIs(selected, discard_east)
        self.assertEqual(by_action[discard_north].retained_concealed_dora_count, 0)
        self.assertIsNone(by_action[discard_north].second_step_ukeire_score)
        self.assertIsNotNone(by_action[discard_east].second_step_ukeire_score)

    def test_analysis_holds_no_mutable_working_state(self) -> None:
        decision = _decision(
            _TWO_STEP_HAND, (_discard(SOUZU_9), _discard(WHITE_DRAGON))
        )

        _, trace = self._traced(decision)

        for evaluation in trace.analysis.candidate_evaluations:
            self.assertEqual(
                tuple(field.name for field in fields(evaluation)),
                (
                    "action",
                    "post_discard_shanten",
                    "current_ukeire_count",
                    "retained_concealed_dora_count",
                    "second_step_ukeire_score",
                ),
            )
            self.assertNotIsInstance(
                evaluation, value_aware._ValueAwareDiscardCandidateWork
            )


class TraceNonInterferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ValueAwareTwoStepUkeirePolicy()

    def _assert_same_selection(self, decision: DecisionContext) -> None:
        untraced = self.policy.choose_action(decision)
        recorder = DecisionTraceRecorder()
        traced = execute_policy_with_trace(self.policy, decision, recorder)

        self.assertEqual(untraced, traced)
        self.assertIs(traced, recorder.snapshot()[0].selected_action)

    def test_every_staged_scenario_selects_the_same_action_with_and_without_trace(
        self,
    ) -> None:
        scenarios = {
            "shanten stage": (
                _TANKI_VERSUS_ONE_SHANTEN_HAND,
                (_discard(MANZU_4), _discard(RED_DRAGON)),
                (),
            ),
            "current ukeire stage": (
                _NINE_MANZU_HAND,
                (_discard(MANZU_2), _discard(MANZU_4)),
                (),
            ),
            "dora stage": (
                _RED_FIVE_HAND,
                (_discard(PINZU_8), _discard(SOUZU_5_RED)),
                (),
            ),
            "second step stage": (
                _TWO_STEP_HAND,
                (_discard(SOUZU_9), _discard(WHITE_DRAGON)),
                (RED_DRAGON,),
            ),
            "tenpai": (
                _TENPAI_HAND,
                (_discard(SOUZU_5), _discard(EAST)),
                (NORTH,),
            ),
        }

        for name, (hand, actions, dora_indicators) in scenarios.items():
            for ordered_actions in itertools.permutations(actions):
                with self.subTest(scenario=name, actions=ordered_actions):
                    self._assert_same_selection(
                        _decision(
                            hand, ordered_actions, dora_indicators=dora_indicators
                        )
                    )

    def test_analysis_capability_shares_the_single_decision_algorithm(self) -> None:
        decision = _decision(
            _TWO_STEP_HAND, (_discard(SOUZU_9), _discard(WHITE_DRAGON))
        )

        proposed = self.policy.choose_action_with_analysis(decision)

        self.assertIsInstance(proposed, PolicyDecision)
        self.assertEqual(proposed.action, self.policy.choose_action(decision))

    def test_policy_keeps_no_cross_decision_analysis_state(self) -> None:
        decision = _decision(
            _TWO_STEP_HAND, (_discard(SOUZU_9), _discard(WHITE_DRAGON))
        )

        execute_policy_with_trace(self.policy, decision, DecisionTraceRecorder())

        self.assertFalse(hasattr(self.policy, "last_analysis"))
        self.assertEqual(vars(self.policy), {})

    def test_untraced_execution_matches_traced_execution_through_the_boundary(
        self,
    ) -> None:
        decision = _decision(
            _TWO_STEP_HAND, (_discard(SOUZU_9), _discard(WHITE_DRAGON))
        )

        untraced = execute_policy(self.policy, decision)
        traced = execute_policy_with_trace(
            self.policy, decision, DecisionTraceRecorder()
        )

        self.assertIs(untraced, traced)


class PackageExportAndSpawnCompatibilityTest(unittest.TestCase):
    def test_value_aware_policy_is_importable_from_the_package(self) -> None:
        from lisjong.policies import ValueAwareTwoStepUkeirePolicy as imported

        self.assertIs(imported, value_aware.ValueAwareTwoStepUkeirePolicy)

    def test_value_aware_policy_is_defined_at_module_level(self) -> None:
        self.assertEqual(
            ValueAwareTwoStepUkeirePolicy.__module__,
            "lisjong.policies.value_aware_two_step_ukeire",
        )
        self.assertIs(
            getattr(value_aware, "ValueAwareTwoStepUkeirePolicy"),
            ValueAwareTwoStepUkeirePolicy,
        )

    def test_value_aware_policy_class_is_picklable_for_spawn(self) -> None:
        # Windows `spawn` + `ProcessPoolExecutor`同様、classをpickleで再構築できる
        # ことを確認する(instanceのpickle可能性は要求されない)。
        roundtrip = pickle.loads(pickle.dumps(ValueAwareTwoStepUkeirePolicy))
        self.assertIs(roundtrip, ValueAwareTwoStepUkeirePolicy)

    def test_value_aware_module_has_no_hidden_or_engine_dependency(self) -> None:
        tree = ast.parse(inspect.getsource(value_aware))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }

        self.assertFalse(
            any(
                module.startswith(prefix)
                for module in imported_modules
                for prefix in (
                    "lisjong.belief",
                    "lisjong_engine",
                    "mahjong",
                    "riichienv",
                    "websockets",
                )
            )
        )

    def test_value_aware_policy_has_an_independent_error_boundary_reuse(self) -> None:
        # ValueAwareは新しいerror型を追加せず、既存TwoStepのfail-closed境界を
        # そのまま再利用する（helperのimportを通じて）。
        with self.assertRaises(TwoStepUkeirePolicyError):
            ValueAwareTwoStepUkeirePolicy().choose_action(
                _decision((), (_discard(SOUZU_9),))
            )


if __name__ == "__main__":
    unittest.main()
