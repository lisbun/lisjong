"""Issue #143 `GenbutsuDefenseFiniteHorizonHandValueAwarePolicy`のunit test。"""

import ast
import inspect
import itertools
import pickle
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware as new_policy
import lisjong.policies.genbutsu_defense_finite_horizon_value_aware as combined
import lisjong.policies.hand_value_aware_two_step_ukeire as hand_value
from lisjong.policies import (
    FiniteHorizonCompletionPolicy,
    GenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
    GenbutsuDefenseFiniteHorizonValueAwarePolicy,
    GenbutsuDefenseTwoStepUkeirePolicy,
    HandValueAwareTwoStepUkeirePolicy,
    TwoStepUkeirePolicy,
    ValueAwareTwoStepUkeirePolicy,
)
from lisjong.policies.finite_horizon_completion import FiniteHorizonCandidateEvaluation
from lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware import (
    _decide_push_fold,
    _defense_eligible_actions,
    _discard_safety,
    _DiscardSafety,
    _PushFoldDecision,
)
from lisjong.policy_contract.action import (
    DiscardAction,
    KyuushuKyuuhaiAction,
    PassAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.decision_trace import DecisionTraceRecorder
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_execution import execute_policy_with_trace
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
MANZU_2 = _tile(TileCategory.MANZU, 2)
MANZU_3 = _tile(TileCategory.MANZU, 3)
MANZU_4 = _tile(TileCategory.MANZU, 4)
MANZU_5 = _tile(TileCategory.MANZU, 5)
PINZU_2 = _tile(TileCategory.PINZU, 2)
PINZU_8 = _tile(TileCategory.PINZU, 8)
SOUZU_5 = _tile(TileCategory.SOUZU, 5)
SOUZU_5_RED = _tile(TileCategory.SOUZU, 5, red=True)
SOUZU_9 = _tile(TileCategory.SOUZU, 9)
WHITE_DRAGON = _tile(TileCategory.HONOR, 5)
RED_DRAGON = _tile(TileCategory.HONOR, 7)

_TWO_STEP_HAND = _hand("345m56679s333577z")
_RED_FIVE_HAND = _hand("345m678m123p33z0s8p1s")
_TANKI_VERSUS_ONE_SHANTEN_HAND = _hand("234m567m234p567p5s7z")
_YAKUHAI_HAND = (WHITE_DRAGON,) * 3 + (SOUZU_9, MANZU_2)
_TANYAO_HAND = _hand("234567m234p567s1z2m")
_HONITSU_HAND = _hand("123456789p11z5s2p")
_CHINITSU_HAND = _hand("123456789p55p2p5s")


def _player(
    *,
    riichi: RiichiState = RiichiState.NONE,
    discards: tuple[Discard, ...] = (),
) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=discards,
        melds=(),
        riichi=riichi,
    )


def _history(tile: Tile, *, order: int = 0) -> Discard:
    return Discard(tile=tile, tsumogiri=False, order=order, called_by=None)


def _players_with_threat(*safe_tiles: Tile) -> tuple[PlayerPublicState, ...]:
    return (
        _player(),
        _player(
            riichi=RiichiState.ACCEPTED,
            discards=tuple(
                _history(tile, order=order) for order, tile in enumerate(safe_tiles)
            ),
        ),
        _player(),
        _player(),
    )


def _players_with_multiple_threats(
    first_discards: tuple[Tile, ...], second_discards: tuple[Tile, ...]
) -> tuple[PlayerPublicState, ...]:
    return (
        _player(),
        _player(
            riichi=RiichiState.DECLARED,
            discards=tuple(
                _history(tile, order=order) for order, tile in enumerate(first_discards)
            ),
        ),
        _player(
            riichi=RiichiState.ACCEPTED,
            discards=tuple(
                _history(tile, order=order)
                for order, tile in enumerate(second_discards)
            ),
        ),
        _player(),
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
    players: tuple[PlayerPublicState, ...] | None = None,
    dora_indicators: tuple[Tile, ...] = (),
) -> DecisionContext:
    return DecisionContext(
        input=_make_input(
            concealed_tiles,
            players=players,
            dora_indicators=dora_indicators,
        ),
        legal_actions=actions,
    )


def _discard(tile: Tile) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=False)


def _mass_patch(
    masses: dict[DiscardAction, int],
    observed: list[tuple[DiscardAction, ...]] | None = None,
):
    def fake_evaluate(
        policy_input, discard_actions, remaining_counts, horizon, evaluator
    ):
        if observed is not None:
            observed.append(discard_actions)
        return tuple(
            FiniteHorizonCandidateEvaluation(
                action=action,
                completion_mass=masses[action],
            )
            for action in discard_actions
        )

    return patch.object(new_policy, "_evaluate_completion_masses", fake_evaluate)


@contextmanager
def _equal_hand_value_structural_stages():
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


class PushFoldDecisionTest(unittest.TestCase):
    """Issue #143: no riichi / tenpai-preserving / all-non-tenpaiのPush-Fold固定。"""

    def test_no_opponent_riichi_is_push(self) -> None:
        actions = (_discard(SOUZU_9), _discard(WHITE_DRAGON))
        policy_input = _make_input(_TWO_STEP_HAND)

        self.assertIs(_decide_push_fold(policy_input, actions), _PushFoldDecision.PUSH)

    def test_riichi_with_a_tenpai_preserving_discard_is_push(self) -> None:
        safe = _discard(MANZU_4)
        tenpai_preserving = _discard(RED_DRAGON)
        policy_input = _make_input(
            _TANKI_VERSUS_ONE_SHANTEN_HAND, players=_players_with_threat(MANZU_4)
        )

        self.assertIs(
            _decide_push_fold(policy_input, (safe, tenpai_preserving)),
            _PushFoldDecision.PUSH,
        )

    def test_riichi_with_every_discard_non_tenpai_is_fold(self) -> None:
        safe = _discard(MANZU_4)
        efficient = _discard(MANZU_5)
        policy_input = _make_input(
            _TWO_STEP_HAND, players=_players_with_threat(MANZU_4)
        )

        self.assertIs(
            _decide_push_fold(policy_input, (efficient, safe)),
            _PushFoldDecision.FOLD,
        )


class DiscardSafetyTest(unittest.TestCase):
    """Issue #143: single/multiple-riichi common genbutsuとUNKNOWNの固定。"""

    def test_tile_in_common_genbutsu_types_is_common_genbutsu(self) -> None:
        self.assertIs(
            _discard_safety(_discard(MANZU_4), frozenset({MANZU_4.tile_type})),
            _DiscardSafety.COMMON_GENBUTSU,
        )

    def test_tile_outside_common_genbutsu_types_is_unknown(self) -> None:
        self.assertIs(
            _discard_safety(_discard(MANZU_5), frozenset({MANZU_4.tile_type})),
            _DiscardSafety.UNKNOWN,
        )

    def test_single_riichi_genbutsu_constrains_the_eligible_set(self) -> None:
        safe = _discard(MANZU_4)
        efficient = _discard(MANZU_5)
        policy_input = _make_input(
            _TWO_STEP_HAND, players=_players_with_threat(MANZU_4)
        )

        self.assertEqual(
            _defense_eligible_actions(policy_input, (efficient, safe)), (safe,)
        )

    def test_multiple_riichi_use_the_common_genbutsu_intersection(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        policy_input = _make_input(
            _TWO_STEP_HAND,
            players=_players_with_multiple_threats((SOUZU_9, WHITE_DRAGON), (SOUZU_9,)),
        )

        self.assertEqual(
            _defense_eligible_actions(policy_input, (discard_white, discard_9s)),
            (discard_9s,),
        )

    def test_fold_without_common_genbutsu_falls_back_to_every_discard(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        policy_input = _make_input(
            _TWO_STEP_HAND, players=_players_with_threat(MANZU_3)
        )

        self.assertEqual(
            _defense_eligible_actions(policy_input, (discard_white, discard_9s)),
            (discard_white, discard_9s),
        )


class DefenseEquivalenceTest(unittest.TestCase):
    """defense decompositionがcurrent Combinedと同一のeligible setを生成することを固定する。"""

    def _scenarios(
        self,
    ) -> dict[str, tuple[PolicyInput, tuple[DiscardAction, ...]]]:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        safe = _discard(MANZU_4)
        efficient = _discard(MANZU_5)
        tenpai_preserving = _discard(RED_DRAGON)
        return {
            "no opponent riichi": (
                _make_input(_TWO_STEP_HAND),
                (discard_9s, discard_white),
            ),
            "riichi + tenpai-preserving discard": (
                _make_input(
                    _TANKI_VERSUS_ONE_SHANTEN_HAND,
                    players=_players_with_threat(MANZU_4),
                ),
                (safe, tenpai_preserving),
            ),
            "riichi + single-riichi common genbutsu": (
                _make_input(_TWO_STEP_HAND, players=_players_with_threat(MANZU_4)),
                (efficient, safe),
            ),
            "riichi + multiple-riichi common genbutsu intersection": (
                _make_input(
                    _TWO_STEP_HAND,
                    players=_players_with_multiple_threats(
                        (SOUZU_9, WHITE_DRAGON), (SOUZU_9,)
                    ),
                ),
                (discard_white, discard_9s),
            ),
            "riichi + no common genbutsu fallback": (
                _make_input(_TWO_STEP_HAND, players=_players_with_threat(MANZU_3)),
                (discard_white, discard_9s),
            ),
        }

    def test_defense_decomposition_matches_current_combined_eligible_set(
        self,
    ) -> None:
        for name, (policy_input, discard_actions) in self._scenarios().items():
            with self.subTest(scenario=name):
                expected = combined._genbutsu_eligible_actions(
                    policy_input, discard_actions
                )
                actual = _defense_eligible_actions(policy_input, discard_actions)
                self.assertEqual(set(actual), set(expected))


class FiniteHorizonPriorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GenbutsuDefenseFiniteHorizonHandValueAwarePolicy()

    def test_defense_constrains_the_finite_horizon_eligible_set(self) -> None:
        safe = _discard(MANZU_4)
        unsafe = _discard(MANZU_5)
        observed: list[tuple[DiscardAction, ...]] = []
        decision = _decision(
            _TWO_STEP_HAND,
            (unsafe, safe),
            players=_players_with_threat(MANZU_4),
        )

        with _mass_patch({safe: 1, unsafe: 999}, observed):
            selected = self.policy.choose_action(decision)

        self.assertIs(selected, safe)
        self.assertEqual(observed, [(safe,)])

    def test_tenpai_disables_the_defense_constraint(self) -> None:
        safe = _discard(MANZU_4)
        tenpai_preserving = _discard(RED_DRAGON)
        observed: list[tuple[DiscardAction, ...]] = []
        decision = _decision(
            _TANKI_VERSUS_ONE_SHANTEN_HAND,
            (safe, tenpai_preserving),
            players=_players_with_threat(MANZU_4),
        )

        with _mass_patch({safe: 1, tenpai_preserving: 2}, observed):
            selected = self.policy.choose_action(decision)

        self.assertIs(selected, tenpai_preserving)
        self.assertEqual(observed, [(safe, tenpai_preserving)])

    def test_no_common_genbutsu_leaves_every_discard_eligible(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        observed: list[tuple[DiscardAction, ...]] = []
        decision = _decision(
            _TWO_STEP_HAND,
            (discard_white, discard_9s),
            players=_players_with_threat(MANZU_3),
        )

        with _mass_patch({discard_9s: 2, discard_white: 1}, observed):
            selected = self.policy.choose_action(decision)

        self.assertIs(selected, discard_9s)
        self.assertEqual(observed, [(discard_white, discard_9s)])

    def test_unique_positive_mass_skips_hand_value_aware_even_when_it_loses_value(
        self,
    ) -> None:
        keep_dora = _discard(SOUZU_9)
        lose_dora = _discard(WHITE_DRAGON)
        decision = _decision(
            _TWO_STEP_HAND,
            (keep_dora, lose_dora),
            dora_indicators=(RED_DRAGON,),
        )

        with (
            _mass_patch({keep_dora: 3, lose_dora: 9}),
            patch.object(
                new_policy,
                "_hand_value_aware_evaluate_and_choose_discard",
                side_effect=AssertionError(
                    "unique positive maximum must not execute HandValueAware"
                ),
            ),
        ):
            selected = self.policy.choose_action(decision)

        self.assertIs(selected, lose_dora)

    def test_completion_mass_loser_is_not_revived_by_hand_value_aware(self) -> None:
        hand_value_favorite = _discard(SOUZU_9)
        maximum_white = _discard(WHITE_DRAGON)
        maximum_5m = _discard(MANZU_5)
        decision = _decision(
            _TWO_STEP_HAND,
            (hand_value_favorite, maximum_white, maximum_5m),
            dora_indicators=(RED_DRAGON,),
        )

        with (
            _mass_patch(
                {
                    hand_value_favorite: 1,
                    maximum_white: 8,
                    maximum_5m: 8,
                }
            ),
            patch.object(
                new_policy,
                "_hand_value_aware_evaluate_and_choose_discard",
                wraps=new_policy._hand_value_aware_evaluate_and_choose_discard,
            ) as hand_value_aware,
        ):
            selected = self.policy.choose_action(decision)

        fallback_actions = hand_value_aware.call_args.args[1]
        self.assertEqual(set(fallback_actions), {maximum_white, maximum_5m})
        self.assertNotIn(hand_value_favorite, fallback_actions)
        self.assertIsNot(selected, hand_value_favorite)

    def test_defense_hard_priority_blocks_a_non_genbutsu_hand_value_favorite(
        self,
    ) -> None:
        """FOLD + COMMON_GENBUTSUありでは、非現物のHandValueAware favoriteを復活させない。"""
        genbutsu = _discard(MANZU_4)
        non_genbutsu_hand_value_favorite = _discard(WHITE_DRAGON)
        decision = _decision(
            _TWO_STEP_HAND,
            (non_genbutsu_hand_value_favorite, genbutsu),
            players=_players_with_threat(MANZU_4),
            dora_indicators=(RED_DRAGON,),
        )

        with _mass_patch({genbutsu: 0}):
            selected = self.policy.choose_action(decision)

        self.assertIs(selected, genbutsu)

    def test_selection_is_independent_of_legal_discard_order(self) -> None:
        unsafe = _discard(MANZU_5)
        safe_keep_dora = _discard(SOUZU_9)
        safe_lose_dora = _discard(WHITE_DRAGON)
        actions = (unsafe, safe_keep_dora, safe_lose_dora)
        masses = {action: 0 for action in actions}

        selections = set()
        for ordered_actions in itertools.permutations(actions):
            with _mass_patch(masses):
                selections.add(
                    self.policy.choose_action(
                        _decision(
                            _TWO_STEP_HAND,
                            ordered_actions,
                            players=_players_with_threat(SOUZU_9, WHITE_DRAGON),
                            dora_indicators=(RED_DRAGON,),
                        )
                    )
                )

        self.assertEqual(selections, {safe_keep_dora})


class HandValueAwareIntegrationTest(unittest.TestCase):
    """Issue #143: HandValueAware固有の役・route差がselectionへ伝播することを固定する。"""

    def setUp(self) -> None:
        self.policy = GenbutsuDefenseFiniteHorizonHandValueAwarePolicy()

    def test_dora_only_case_matches_current_combined(self) -> None:
        """HandValueAware固有featureに差がなければcurrent Combinedと同じ選択になる。"""
        keep_dora = _discard(SOUZU_9)
        lose_dora = _discard(WHITE_DRAGON)
        decision = _decision(
            _TWO_STEP_HAND,
            (lose_dora, keep_dora),
            dora_indicators=(RED_DRAGON,),
        )

        with _mass_patch({keep_dora: 0, lose_dora: 0}):
            new_selected = self.policy.choose_action(decision)
        with patch.object(
            combined,
            "_evaluate_completion_masses",
            lambda *args: (
                FiniteHorizonCandidateEvaluation(action=lose_dora, completion_mass=0),
                FiniteHorizonCandidateEvaluation(action=keep_dora, completion_mass=0),
            ),
        ):
            combined_selected = (
                GenbutsuDefenseFiniteHorizonValueAwarePolicy().choose_action(decision)
            )

        self.assertIs(new_selected, keep_dora)
        self.assertIs(combined_selected, keep_dora)

    def test_completed_yakuhai_difference_changes_selection(self) -> None:
        discard_other = _discard(SOUZU_9)
        break_white = _discard(WHITE_DRAGON)
        decision = _decision(_YAKUHAI_HAND, (break_white, discard_other))

        with (
            _equal_hand_value_structural_stages(),
            _mass_patch({break_white: 5, discard_other: 5}),
        ):
            selected = self.policy.choose_action(decision)

        self.assertIs(selected, discard_other)

    def test_tanyao_route_difference_changes_selection(self) -> None:
        discard_honor = _discard(EAST)
        discard_middle = _discard(MANZU_2)
        decision = _decision(_TANYAO_HAND, (discard_middle, discard_honor))

        with (
            _equal_hand_value_structural_stages(),
            _mass_patch({discard_middle: 5, discard_honor: 5}),
        ):
            selected = self.policy.choose_action(decision)

        self.assertIs(selected, discard_honor)

    def test_honitsu_route_difference_changes_selection(self) -> None:
        discard_off_suit = _discard(SOUZU_5)
        discard_core_suit = _discard(PINZU_2)
        decision = _decision(_HONITSU_HAND, (discard_core_suit, discard_off_suit))

        with (
            _equal_hand_value_structural_stages(),
            _mass_patch({discard_core_suit: 5, discard_off_suit: 5}),
        ):
            selected = self.policy.choose_action(decision)

        self.assertIs(selected, discard_off_suit)

    def test_chinitsu_route_difference_changes_selection(self) -> None:
        discard_off_suit = _discard(SOUZU_5)
        discard_core_suit = _discard(PINZU_2)
        decision = _decision(_CHINITSU_HAND, (discard_core_suit, discard_off_suit))

        with (
            _equal_hand_value_structural_stages(),
            _mass_patch({discard_core_suit: 5, discard_off_suit: 5}),
        ):
            selected = self.policy.choose_action(decision)

        self.assertIs(selected, discard_off_suit)


class ExistingPolicyBaselinePreservationTest(unittest.TestCase):
    def test_existing_generations_keep_their_own_selection_semantics(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        actions = (discard_9s, discard_white)

        self.assertIs(
            TwoStepUkeirePolicy().choose_action(_decision(_TWO_STEP_HAND, actions)),
            discard_white,
        )
        self.assertIs(
            GenbutsuDefenseTwoStepUkeirePolicy().choose_action(
                _decision(
                    _TWO_STEP_HAND,
                    actions,
                    players=_players_with_threat(SOUZU_9),
                )
            ),
            discard_9s,
        )
        self.assertIs(
            ValueAwareTwoStepUkeirePolicy().choose_action(
                _decision(_TWO_STEP_HAND, actions, dora_indicators=(RED_DRAGON,))
            ),
            discard_9s,
        )
        self.assertIs(
            HandValueAwareTwoStepUkeirePolicy().choose_action(
                _decision(_TWO_STEP_HAND, actions, dora_indicators=(RED_DRAGON,))
            ),
            discard_9s,
        )

    def test_current_combined_selection_is_unaffected_by_the_new_module(self) -> None:
        safe = _discard(MANZU_4)
        unsafe = _discard(MANZU_5)
        decision = _decision(
            _TWO_STEP_HAND,
            (unsafe, safe),
            players=_players_with_threat(MANZU_4),
        )

        with patch.object(
            combined,
            "_evaluate_completion_masses",
            lambda *args: (
                FiniteHorizonCandidateEvaluation(action=safe, completion_mass=1),
            ),
        ):
            selected = GenbutsuDefenseFiniteHorizonValueAwarePolicy().choose_action(
                decision
            )

        self.assertIs(selected, safe)


class InheritedOrchestrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GenbutsuDefenseFiniteHorizonHandValueAwarePolicy()

    def test_only_decide_discard_is_overridden(self) -> None:
        self.assertTrue(
            issubclass(
                GenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
                TwoStepUkeirePolicy,
            )
        )
        self.assertIn(
            "_decide_discard",
            vars(GenbutsuDefenseFiniteHorizonHandValueAwarePolicy),
        )
        for method_name in (
            "_decide",
            "choose_action",
            "choose_action_with_analysis",
        ):
            with self.subTest(method_name=method_name):
                self.assertNotIn(
                    method_name,
                    vars(GenbutsuDefenseFiniteHorizonHandValueAwarePolicy),
                )

    def test_winning_action_and_riichi_skip_discard_composition(self) -> None:
        discard = _discard(SOUZU_9)
        ron = RonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            winning_tile=MANZU_3,
        )
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_3)
        riichi = RiichiAction(actor=Seat.SEAT_0)

        with patch.object(
            new_policy,
            "_evaluate_and_choose_discard",
            side_effect=AssertionError("orchestration must skip discard composition"),
        ):
            self.assertIs(
                self.policy.choose_action(
                    _decision(_TWO_STEP_HAND, (discard, tsumo, ron))
                ),
                ron,
            )
            self.assertIs(
                self.policy.choose_action(_decision(_TWO_STEP_HAND, (discard, riichi))),
                riichi,
            )

    def test_pass_and_single_action_fallback_are_inherited(self) -> None:
        pass_action = PassAction(actor=Seat.SEAT_0)
        fallback = KyuushuKyuuhaiAction(actor=Seat.SEAT_0)

        self.assertIs(
            self.policy.choose_action(_decision((), (fallback, pass_action))),
            pass_action,
        )
        self.assertIs(
            self.policy.choose_action(_decision((), (fallback,))),
            fallback,
        )

    def test_discard_trace_runs_composition_once_and_reports_no_analysis(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        decision = _decision(_TWO_STEP_HAND, (discard_9s, discard_white))
        recorder = DecisionTraceRecorder()

        with (
            _mass_patch({discard_9s: 1, discard_white: 2}),
            patch.object(
                new_policy,
                "_evaluate_and_choose_discard",
                wraps=new_policy._evaluate_and_choose_discard,
            ) as evaluate,
        ):
            selected = execute_policy_with_trace(self.policy, decision, recorder)

        (trace,) = recorder.snapshot()
        self.assertIs(selected, discard_white)
        self.assertIs(trace.selected_action, selected)
        self.assertIsNone(trace.analysis)
        self.assertEqual(evaluate.call_count, 1)


class PackageExportAndSpawnCompatibilityTest(unittest.TestCase):
    def test_new_policy_is_public_and_module_level(self) -> None:
        from lisjong.policies import (
            GenbutsuDefenseFiniteHorizonHandValueAwarePolicy as imported,
        )

        self.assertIs(
            imported, new_policy.GenbutsuDefenseFiniteHorizonHandValueAwarePolicy
        )
        self.assertEqual(
            GenbutsuDefenseFiniteHorizonHandValueAwarePolicy.__module__,
            "lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware",
        )

    def test_new_policy_class_is_picklable_for_windows_spawn(self) -> None:
        roundtrip = pickle.loads(
            pickle.dumps(GenbutsuDefenseFiniteHorizonHandValueAwarePolicy)
        )

        self.assertIs(roundtrip, GenbutsuDefenseFiniteHorizonHandValueAwarePolicy)

    def test_module_uses_only_policy_visible_dependencies(self) -> None:
        tree = ast.parse(inspect.getsource(new_policy))
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
                    "lisjong_engine",
                    "lisjong_arena",
                    "riichienv",
                    "mahjong",
                )
            )
        )


class RegressionSmokeTest(unittest.TestCase):
    """FiniteHorizonCompletionPolicyの既存selection semanticも変更していないことを固定する。"""

    def test_existing_finite_horizon_unique_maximum_semantic_is_unchanged(self) -> None:
        import lisjong.policies.finite_horizon_completion as finite_horizon

        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)

        def fake_evaluate(
            policy_input, discard_actions, remaining_counts, horizon, evaluator
        ):
            masses = {discard_9s: 1, discard_white: 9}
            return tuple(
                FiniteHorizonCandidateEvaluation(action, masses[action])
                for action in discard_actions
            )

        with patch.object(finite_horizon, "_evaluate_completion_masses", fake_evaluate):
            selected = FiniteHorizonCompletionPolicy().choose_action(
                _decision(_TWO_STEP_HAND, (discard_9s, discard_white))
            )

        self.assertIs(selected, discard_white)


if __name__ == "__main__":
    unittest.main()
