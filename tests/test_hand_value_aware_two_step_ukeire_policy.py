"""Issue #125 `HandValueAwareTwoStepUkeirePolicy`のunit test。"""

import ast
import inspect
import itertools
import pickle
import unittest
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, fields
from unittest.mock import patch

import lisjong.policies.hand_value_aware_two_step_ukeire as hand_value
from lisjong.policies import (
    HandValueAwareTwoStepUkeirePolicy,
    TwoStepUkeirePolicy,
    ValueAwareTwoStepUkeirePolicy,
)
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    HandValueAwareTwoStepUkeireAnalysis,
    HandValueCandidateEvaluation,
    _completed_yakuhai_value,
    _evaluate_and_choose_discard,
    _retained_real_value,
    _yaku_route_value,
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
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
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
WHITE = _tile(TileCategory.HONOR, 5)
GREEN = _tile(TileCategory.HONOR, 6)
RED = _tile(TileCategory.HONOR, 7)
MANZU_2 = _tile(TileCategory.MANZU, 2)
MANZU_4 = _tile(TileCategory.MANZU, 4)
PINZU_2 = _tile(TileCategory.PINZU, 2)
PINZU_8 = _tile(TileCategory.PINZU, 8)
SOUZU_5 = _tile(TileCategory.SOUZU, 5)
SOUZU_5_RED = _tile(TileCategory.SOUZU, 5, red=True)
SOUZU_9 = _tile(TileCategory.SOUZU, 9)

_TWO_STEP_HAND = _hand("345m56679s333577z")
_RED_FIVE_HAND = _hand("345m678m123p33z0s8p1s")
_TANKI_VERSUS_ONE_SHANTEN_HAND = _hand("234m567m234p567p5s7z")
_NINE_MANZU_HAND = _hand("123456789m111p23p")


def _player(*, melds: tuple[PublicMeld, ...] = ()) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=(),
        melds=melds,
        riichi=RiichiState.NONE,
    )


def _meld(kind: MeldKind, tiles: tuple[Tile, ...]) -> PublicMeld:
    if kind is MeldKind.ANKAN:
        return PublicMeld(kind=kind, tiles=tiles, from_seat=None, called_tile=None)
    return PublicMeld(
        kind=kind,
        tiles=tiles,
        from_seat=Seat.SEAT_1,
        called_tile=tiles[0],
    )


def _make_input(
    concealed_tiles: tuple[Tile, ...],
    *,
    self_seat: Seat = Seat.SEAT_0,
    dealer_seat: Seat = Seat.SEAT_0,
    round_wind: Wind = Wind.EAST,
    own_melds: tuple[PublicMeld, ...] = (),
    dora_indicators: tuple[Tile, ...] = (),
) -> PolicyInput:
    players = [_player() for _ in range(4)]
    players[int(self_seat)] = _player(melds=own_melds)
    return PolicyInput(
        self_seat=self_seat,
        round=RoundState(
            round_wind=round_wind,
            hand_number=1,
            dealer_seat=dealer_seat,
            honba=0,
            riichi_sticks=0,
            dora_indicators=dora_indicators,
            live_wall_tiles_remaining=70,
        ),
        players=tuple(players),
        own_hand=OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=None),
    )


def _decision(
    concealed_tiles: tuple[Tile, ...],
    actions: tuple[object, ...],
    **input_kwargs,
) -> DecisionContext:
    return DecisionContext(
        input=_make_input(concealed_tiles, **input_kwargs),
        legal_actions=actions,
    )


def _discard(tile: Tile) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=False)


@contextmanager
def _equal_structural_stages():
    with (
        patch.object(
            hand_value._DecisionShantenEvaluator,
            "calculate",
            return_value=1,
        ),
        patch.object(hand_value, "_ukeire_count", return_value=10),
        patch.object(hand_value, "_second_step_score", return_value=0),
    ):
        yield


class RetainedRealValueTest(unittest.TestCase):
    def test_indicator_dora_and_red_dora_reuse_value_aware_semantics(self) -> None:
        red_dora = _tile(TileCategory.MANZU, 5, red=True)
        policy_input = _make_input((red_dora,), dora_indicators=(MANZU_4,))

        self.assertEqual(_retained_real_value((red_dora,), policy_input), 2)

    def test_each_dragon_triplet_is_one(self) -> None:
        for dragon in (WHITE, GREEN, RED):
            with self.subTest(dragon=dragon):
                self.assertEqual(
                    _completed_yakuhai_value(
                        (dragon,) * 3,
                        (),
                        seat_wind_rank=2,
                        round_wind_rank=1,
                    ),
                    1,
                )

    def test_seat_wind_and_round_wind_are_each_one(self) -> None:
        self.assertEqual(
            _completed_yakuhai_value(
                (SOUTH,) * 3, (), seat_wind_rank=2, round_wind_rank=1
            ),
            1,
        )
        self.assertEqual(
            _completed_yakuhai_value(
                (EAST,) * 3, (), seat_wind_rank=2, round_wind_rank=1
            ),
            1,
        )

    def test_double_wind_is_two(self) -> None:
        self.assertEqual(
            _completed_yakuhai_value(
                (EAST,) * 3, (), seat_wind_rank=1, round_wind_rank=1
            ),
            2,
        )

    def test_yakuhai_pair_is_not_completed_value(self) -> None:
        self.assertEqual(
            _completed_yakuhai_value(
                (WHITE,) * 2, (), seat_wind_rank=2, round_wind_rank=1
            ),
            0,
        )

    def test_open_yakuhai_pon_and_kan_are_completed(self) -> None:
        pon = _meld(MeldKind.PON, (WHITE,) * 3)
        kan = _meld(MeldKind.ANKAN, (GREEN,) * 4)
        self.assertEqual(
            _completed_yakuhai_value(
                (), (pon, kan), seat_wind_rank=2, round_wind_rank=1
            ),
            2,
        )

    def test_self_wind_is_derived_from_dealer_and_self_seat(self) -> None:
        policy_input = _make_input(
            (SOUTH,) * 3,
            self_seat=Seat.SEAT_2,
            dealer_seat=Seat.SEAT_1,
            round_wind=Wind.WEST,
        )
        self.assertEqual(_retained_real_value((SOUTH,) * 3, policy_input), 1)

    def test_yakuhai_stage_changes_selection(self) -> None:
        discard_other = _discard(SOUZU_9)
        break_white = _discard(WHITE)
        hand = (WHITE,) * 3 + (SOUZU_9, MANZU_2)

        with _equal_structural_stages():
            selected, evaluations = _evaluate_and_choose_discard(
                _make_input(hand), (break_white, discard_other)
            )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, discard_other)
        self.assertEqual(by_action[discard_other].retained_real_value, 1)
        self.assertEqual(by_action[break_white].retained_real_value, 0)


class YakuRouteValueTest(unittest.TestCase):
    def test_tanyao_compatibility_is_one(self) -> None:
        self.assertEqual(_yaku_route_value(_hand("234m456p678s"), ()), 1)

    def test_honitsu_compatibility_is_two(self) -> None:
        self.assertEqual(_yaku_route_value(_hand("123456789p11z"), ()), 2)

    def test_chinitsu_compatibility_is_three_without_honitsu_double_count(self) -> None:
        self.assertEqual(_yaku_route_value(_hand("123456789p"), ()), 3)

    def test_tanyao_can_coexist_with_chinitsu_route(self) -> None:
        self.assertEqual(_yaku_route_value(_hand("234456678p"), ()), 4)

    def test_open_meld_participates_in_route_compatibility(self) -> None:
        compatible = _meld(MeldKind.CHI, _hand("234p"))
        incompatible = _meld(MeldKind.CHI, _hand("234m"))
        concealed = _hand("56789p11z")

        self.assertEqual(_yaku_route_value(concealed, (compatible,)), 2)
        self.assertEqual(_yaku_route_value(concealed, (incompatible,)), 0)

    def test_open_hand_route_stage_changes_candidate_selection(self) -> None:
        open_chi = _meld(MeldKind.CHI, _hand("234p"))
        discard_off_suit = _discard(SOUZU_5)
        discard_core_suit = _discard(PINZU_2)
        concealed = _hand("56789p11z5s2p")

        with _equal_structural_stages():
            selected, evaluations = _evaluate_and_choose_discard(
                _make_input(concealed, own_melds=(open_chi,)),
                (discard_core_suit, discard_off_suit),
            )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, discard_off_suit)
        self.assertEqual(by_action[discard_off_suit].yaku_route_value, 2)
        self.assertEqual(by_action[discard_core_suit].yaku_route_value, 0)

    def test_tanyao_stage_changes_selection(self) -> None:
        discard_honor = _discard(EAST)
        discard_middle = _discard(MANZU_2)
        hand = _hand("234567m234p567s1z2m")

        with _equal_structural_stages():
            selected, evaluations = _evaluate_and_choose_discard(
                _make_input(hand), (discard_middle, discard_honor)
            )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, discard_honor)
        self.assertEqual(by_action[discard_honor].yaku_route_value, 1)
        self.assertEqual(by_action[discard_middle].yaku_route_value, 0)

    def test_honitsu_stage_changes_selection_against_stable_tie_break(self) -> None:
        discard_off_suit = _discard(SOUZU_5)
        discard_core_suit = _discard(PINZU_2)
        hand = _hand("123456789p11z5s2p")

        with _equal_structural_stages():
            selected, evaluations = _evaluate_and_choose_discard(
                _make_input(hand), (discard_core_suit, discard_off_suit)
            )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, discard_off_suit)
        self.assertEqual(by_action[discard_off_suit].yaku_route_value, 2)
        self.assertEqual(by_action[discard_core_suit].yaku_route_value, 0)

    def test_chinitsu_stage_changes_selection_against_stable_tie_break(self) -> None:
        discard_off_suit = _discard(SOUZU_5)
        discard_core_suit = _discard(PINZU_2)
        hand = _hand("123456789p55p2p5s")

        with _equal_structural_stages():
            selected, evaluations = _evaluate_and_choose_discard(
                _make_input(hand), (discard_core_suit, discard_off_suit)
            )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, discard_off_suit)
        self.assertEqual(by_action[discard_off_suit].yaku_route_value, 3)
        self.assertEqual(by_action[discard_core_suit].yaku_route_value, 0)


class SelectionPriorityTest(unittest.TestCase):
    def test_shanten_is_a_hard_priority_over_value(self) -> None:
        keep_tenpai = _discard(RED)
        break_tenpai = _discard(MANZU_4)
        decision = _decision(
            _TANKI_VERSUS_ONE_SHANTEN_HAND,
            (break_tenpai, keep_tenpai),
            dora_indicators=(_tile(TileCategory.HONOR, 6),),
        )

        self.assertIs(
            HandValueAwareTwoStepUkeirePolicy().choose_action(decision), keep_tenpai
        )

    def test_current_ukeire_is_a_hard_priority_over_value(self) -> None:
        lower_ukeire = _discard(MANZU_2)
        higher_ukeire = _discard(MANZU_4)
        decision = _decision(
            _NINE_MANZU_HAND,
            (lower_ukeire, higher_ukeire),
            dora_indicators=(_tile(TileCategory.MANZU, 1),),
        )

        selected, evaluations = _evaluate_and_choose_discard(
            decision.input, (lower_ukeire, higher_ukeire)
        )
        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, higher_ukeire)
        self.assertGreater(
            by_action[higher_ukeire].current_ukeire_count,
            by_action[lower_ukeire].current_ukeire_count,
        )
        self.assertIsNone(by_action[lower_ukeire].retained_real_value)
        self.assertIsNone(by_action[higher_ukeire].retained_real_value)

    def test_red_dora_retention_changes_a_tied_selection(self) -> None:
        keep_red = _discard(PINZU_8)
        lose_red = _discard(SOUZU_5_RED)

        selected, evaluations = _evaluate_and_choose_discard(
            _make_input(_RED_FIVE_HAND), (lose_red, keep_red)
        )
        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, keep_red)
        self.assertEqual(by_action[keep_red].retained_real_value, 1)
        self.assertEqual(by_action[lose_red].retained_real_value, 0)

    def test_indicator_dora_retention_changes_existing_two_step_choice(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE)
        policy_input = _make_input(_TWO_STEP_HAND, dora_indicators=(RED,))

        self.assertIs(
            TwoStepUkeirePolicy().choose_action(
                DecisionContext(policy_input, (discard_9s, discard_white))
            ),
            discard_white,
        )
        selected, evaluations = _evaluate_and_choose_discard(
            policy_input, (discard_white, discard_9s)
        )
        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, discard_9s)
        self.assertEqual(by_action[discard_9s].retained_real_value, 1)
        self.assertEqual(by_action[discard_white].retained_real_value, 0)

    def test_retained_real_value_beats_higher_route_value(self) -> None:
        real_favorite = _discard(SOUZU_9)
        route_favorite = _discard(WHITE)

        def real_value(post_hand, _policy_input):
            return 1 if WHITE in post_hand else 0

        def route_value(post_hand, _melds):
            return 3 if WHITE not in post_hand else 0

        with (
            _equal_structural_stages(),
            patch.object(hand_value, "_retained_real_value", side_effect=real_value),
            patch.object(hand_value, "_yaku_route_value", side_effect=route_value),
        ):
            selected, evaluations = _evaluate_and_choose_discard(
                _make_input(_TWO_STEP_HAND), (route_favorite, real_favorite)
            )

        by_action = {evaluation.action: evaluation for evaluation in evaluations}
        self.assertIs(selected, real_favorite)
        self.assertEqual(by_action[real_favorite].retained_real_value, 1)
        self.assertIsNone(by_action[route_favorite].yaku_route_value)

    def test_value_beats_second_step_without_evaluating_loser_branch(self) -> None:
        keep_dora = _discard(SOUZU_9)
        lose_dora = _discard(WHITE)
        with patch.object(
            hand_value,
            "_second_step_score",
            side_effect=AssertionError("value winner must end selection"),
        ):
            selected, evaluations = _evaluate_and_choose_discard(
                _make_input(_TWO_STEP_HAND, dora_indicators=(RED,)),
                (lose_dora, keep_dora),
            )

        self.assertIs(selected, keep_dora)
        self.assertTrue(
            all(
                evaluation.second_step_ukeire_score is None
                for evaluation in evaluations
            )
        )

    def test_equal_value_falls_back_to_existing_two_step_semantics(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE)
        decision = _decision(_TWO_STEP_HAND, (discard_9s, discard_white))

        self.assertIs(TwoStepUkeirePolicy().choose_action(decision), discard_white)
        self.assertIs(
            HandValueAwareTwoStepUkeirePolicy().choose_action(decision), discard_white
        )

    def test_existing_two_step_and_value_aware_behaviors_are_preserved(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE)
        decision = _decision(
            _TWO_STEP_HAND,
            (discard_9s, discard_white),
            dora_indicators=(RED,),
        )

        self.assertIs(TwoStepUkeirePolicy().choose_action(decision), discard_white)
        self.assertIs(
            ValueAwareTwoStepUkeirePolicy().choose_action(decision), discard_9s
        )


class DeterminismAndBoundaryTest(unittest.TestCase):
    def test_legal_action_order_does_not_change_selection(self) -> None:
        actions = (_discard(SOUZU_9), _discard(WHITE))
        results = {
            HandValueAwareTwoStepUkeirePolicy().choose_action(
                _decision(_TWO_STEP_HAND, permutation, dora_indicators=(RED,))
            )
            for permutation in itertools.permutations(actions)
        }
        self.assertEqual(results, {_discard(SOUZU_9)})

    def test_concealed_tile_input_order_does_not_change_selection(self) -> None:
        actions = (_discard(SOUZU_9), _discard(WHITE))
        expected = _discard(SOUZU_9)
        for concealed in (
            _TWO_STEP_HAND,
            tuple(reversed(_TWO_STEP_HAND)),
            _TWO_STEP_HAND[4:] + _TWO_STEP_HAND[:4],
        ):
            with self.subTest(concealed=concealed):
                selected = HandValueAwareTwoStepUkeirePolicy().choose_action(
                    _decision(concealed, actions, dora_indicators=(RED,))
                )
                self.assertEqual(selected, expected)

    def test_module_uses_only_policy_visible_dependencies(self) -> None:
        tree = ast.parse(inspect.getsource(hand_value))
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
                    "lisjong_arena",
                    "riichienv",
                    "mahjong",
                )
            )
        )


class AnalysisAndOrchestrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = HandValueAwareTwoStepUkeirePolicy()

    def test_candidate_analysis_is_typed_immutable_and_distinguishes_none_zero(
        self,
    ) -> None:
        action = _discard(SOUZU_9)
        unevaluated = HandValueCandidateEvaluation(action, 1, 0, None, None, None)
        evaluated_zero = HandValueCandidateEvaluation(action, 1, 0, 0, 0, 0)
        analysis = HandValueAwareTwoStepUkeireAnalysis((evaluated_zero,))

        self.assertIsInstance(analysis, AnalysisTrace)
        self.assertNotEqual(unevaluated, evaluated_zero)
        self.assertEqual(
            tuple(field.name for field in fields(evaluated_zero)),
            (
                "action",
                "post_discard_shanten",
                "current_ukeire_count",
                "retained_real_value",
                "yaku_route_value",
                "second_step_ukeire_score",
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            analysis.candidate_evaluations = ()

    def test_analysis_rejects_free_form_empty_and_wrong_scalar_types(self) -> None:
        with self.assertRaises(TypeError):
            HandValueAwareTwoStepUkeireAnalysis(7)
        with self.assertRaises(TypeError):
            HandValueAwareTwoStepUkeireAnalysis(({"value": 1},))
        with self.assertRaises(ValueError):
            HandValueAwareTwoStepUkeireAnalysis(())
        with self.assertRaises(TypeError):
            HandValueCandidateEvaluation(
                _discard(SOUZU_9), True, None, None, None, None
            )

    def test_traced_and_untraced_execution_select_the_same_action(self) -> None:
        decision = _decision(
            _TWO_STEP_HAND,
            (_discard(WHITE), _discard(SOUZU_9)),
            dora_indicators=(RED,),
        )
        recorder = DecisionTraceRecorder()

        untraced = execute_policy(self.policy, decision)
        traced = execute_policy_with_trace(self.policy, decision, recorder)
        (trace,) = recorder.snapshot()

        self.assertIs(untraced, traced)
        self.assertIs(trace.selected_action, traced)
        self.assertIsInstance(trace.analysis, HandValueAwareTwoStepUkeireAnalysis)

    def test_selection_and_analysis_share_one_value_calculation(self) -> None:
        decision = _decision(
            _TWO_STEP_HAND,
            (_discard(WHITE), _discard(SOUZU_9)),
            dora_indicators=(RED,),
        )
        recorder = DecisionTraceRecorder()
        with patch.object(
            hand_value,
            "_retained_real_value",
            wraps=hand_value._retained_real_value,
        ) as retained:
            execute_policy_with_trace(self.policy, decision, recorder)

        self.assertEqual(retained.call_count, 2)

    def test_only_discard_extension_point_is_overridden(self) -> None:
        self.assertTrue(
            issubclass(HandValueAwareTwoStepUkeirePolicy, TwoStepUkeirePolicy)
        )
        self.assertIn("_decide_discard", vars(HandValueAwareTwoStepUkeirePolicy))
        for method_name in ("_decide", "choose_action", "choose_action_with_analysis"):
            with self.subTest(method_name=method_name):
                self.assertNotIn(method_name, vars(HandValueAwareTwoStepUkeirePolicy))

    def test_winning_riichi_pass_and_fallback_are_inherited(self) -> None:
        discard = _discard(SOUZU_9)
        ron = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_2)
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_2)
        riichi = RiichiAction(actor=Seat.SEAT_0)
        pass_action = PassAction(actor=Seat.SEAT_0)
        fallback = KyuushuKyuuhaiAction(actor=Seat.SEAT_0)

        self.assertIs(self.policy.choose_action(_decision((), (tsumo, ron))), ron)
        self.assertIs(
            self.policy.choose_action(_decision((SOUZU_9,), (discard, riichi))),
            riichi,
        )
        self.assertIs(
            self.policy.choose_action(_decision((), (pass_action,))), pass_action
        )
        self.assertIs(self.policy.choose_action(_decision((), (fallback,))), fallback)


class PublicAndSpawnCompatibilityTest(unittest.TestCase):
    def test_policy_is_public_and_module_level(self) -> None:
        from lisjong.policies import HandValueAwareTwoStepUkeirePolicy as imported

        self.assertIs(imported, hand_value.HandValueAwareTwoStepUkeirePolicy)
        self.assertEqual(
            HandValueAwareTwoStepUkeirePolicy.__module__,
            "lisjong.policies.hand_value_aware_two_step_ukeire",
        )

    def test_policy_class_is_picklable_for_windows_spawn(self) -> None:
        roundtrip = pickle.loads(pickle.dumps(HandValueAwareTwoStepUkeirePolicy))
        self.assertIs(roundtrip, HandValueAwareTwoStepUkeirePolicy)


if __name__ == "__main__":
    unittest.main()
