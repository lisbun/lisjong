import itertools
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch


import lisjong.policies.hand_value_tradeoff_mechanism_riichi_defense as v2
import lisjong.policies.hand_value_tradeoff_targeted_honor_release as combined
import lisjong.policies.targeted_honor_release_terminal_progression as targeted
from lisjong.policies.finite_horizon_completion import (
    FiniteHorizonCandidateEvaluation,
)
from lisjong.policies.targeted_honor_release_terminal_progression import (
    TargetedHonorReleaseTerminalProgressionPolicy,
)
from lisjong.policy import DecisionContext
from lisjong.policy_contract.action import DiscardAction, RiichiAction, RonAction
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind


def _tile(category: TileCategory, rank: int) -> Tile:
    return Tile(TileType(category, rank), is_red=False)


M3 = _tile(TileCategory.MANZU, 3)
M4 = _tile(TileCategory.MANZU, 4)
EAST = _tile(TileCategory.HONOR, 1)
WHITE = _tile(TileCategory.HONOR, 5)
A_M3 = DiscardAction(actor=Seat.SEAT_0, tile=M3, tsumogiri=False)
A_M4 = DiscardAction(actor=Seat.SEAT_0, tile=M4, tsumogiri=False)
A_EAST = DiscardAction(actor=Seat.SEAT_0, tile=EAST, tsumogiri=False)
A_WHITE = DiscardAction(actor=Seat.SEAT_0, tile=WHITE, tsumogiri=False)


def _player() -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=(),
        melds=(),
        riichi=RiichiState.NONE,
    )


def _input(*, normal_turn: bool = True) -> PolicyInput:
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(),
            live_wall_tiles_remaining=70,
        ),
        players=tuple(_player() for _ in range(4)),
        own_hand=OwnHandState(
            concealed_tiles=(M3, M4, EAST),
            drawn_tile=M3 if normal_turn else None,
        ),
    )


def _decision(
    policy_input: PolicyInput,
    actions: tuple[object, ...],
) -> DecisionContext:
    return DecisionContext(policy_input, actions)


def _targeted_analysis(
    *,
    stage: targeted.TargetedHonorReleaseActivationStage,
    parent_action: DiscardAction,
    selected_action: DiscardAction,
) -> targeted.TargetedHonorReleaseAnalysis:
    return targeted._analysis(
        stage=stage,
        parent_action=parent_action,
        selected_action=selected_action,
        branch=targeted.TargetedHonorReleaseBranch.PUSH,
        closed_hand=True,
        parent_snapshot=None,
        eligible_candidate_count=2,
    )


def _completion(
    actions: tuple[DiscardAction, ...],
    masses: dict[DiscardAction, int],
) -> tuple[FiniteHorizonCandidateEvaluation, ...]:
    return tuple(
        FiniteHorizonCandidateEvaluation(action, masses[action]) for action in actions
    )


class CompositionSelectionTest(unittest.TestCase):
    def _base_patches(
        self,
        actions: tuple[DiscardAction, ...],
        masses: dict[DiscardAction, int],
        *,
        branch: targeted.TargetedHonorReleaseBranch = (
            targeted.TargetedHonorReleaseBranch.PUSH
        ),
        closed_hand: bool = True,
    ):
        return (
            patch.object(
                combined,
                "_classify_branch",
                return_value=(branch, actions),
            ),
            patch.object(combined, "_is_closed_hand", return_value=closed_hand),
            patch.object(combined, "_root_remaining_counts", return_value=(4,) * 34),
            patch.object(
                combined,
                "_evaluate_completion_masses",
                return_value=_completion(actions, masses),
            ),
        )

    def test_confirmed_r5_switch_has_precedence_and_skips_v2(self) -> None:
        actions = (A_M3, A_EAST)
        switch = _targeted_analysis(
            stage=targeted.TargetedHonorReleaseActivationStage.R5_HONOR_ONLY_SWITCH,
            parent_action=A_M3,
            selected_action=A_EAST,
        )
        patches = self._base_patches(actions, {A_M3: 0, A_EAST: 0})

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3] as completion,
            patch.object(
                combined,
                "_evaluate_all_zero_targeted_honor_release",
                return_value=(A_EAST, switch),
            ),
            patch.object(
                combined,
                "_select_finite_horizon_hand_value_v2",
                side_effect=AssertionError(
                    "HandValue v2 must not run after confirmed Champion switch"
                ),
            ),
        ):
            selected, analysis = combined._evaluate_and_choose_discard(
                _input(), actions
            )

        self.assertIs(selected, A_EAST)
        self.assertIs(
            analysis.selection_source,
            combined.HandValueTradeoffTargetedHonorReleaseSelectionSource.CHAMPION_TARGETED_HONOR_RELEASE,
        )
        self.assertIsNone(analysis.hand_value_v2_action)
        self.assertFalse(analysis.action_changed_vs_champion)
        self.assertTrue(analysis.action_changed_vs_former_parent)
        completion.assert_called_once()

    def test_inactive_targeted_gate_delegates_to_exact_v2(self) -> None:
        actions = (A_M3, A_EAST)
        inactive = _targeted_analysis(
            stage=(
                targeted.TargetedHonorReleaseActivationStage.PARENT_SHANTEN_TOO_CLOSE
            ),
            parent_action=A_M3,
            selected_action=A_M3,
        )
        patches = self._base_patches(actions, {A_M3: 0, A_EAST: 0})

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch.object(
                combined,
                "_evaluate_all_zero_targeted_honor_release",
                return_value=(A_M3, inactive),
            ),
            patch.object(
                combined,
                "_select_finite_horizon_hand_value_v2",
                return_value=A_EAST,
            ),
        ):
            selected, analysis = combined._evaluate_and_choose_discard(
                _input(), actions
            )

        self.assertIs(selected, A_EAST)
        self.assertIs(
            analysis.selection_source,
            combined.HandValueTradeoffTargetedHonorReleaseSelectionSource.HAND_VALUE_V2,
        )
        self.assertTrue(analysis.action_changed_vs_champion)
        self.assertTrue(analysis.action_changed_vs_former_parent)

    def test_shared_action_is_reported_without_false_divergence(self) -> None:
        actions = (A_M3, A_EAST)
        inactive = _targeted_analysis(
            stage=(
                targeted.TargetedHonorReleaseActivationStage.PARENT_SHANTEN_TOO_CLOSE
            ),
            parent_action=A_M3,
            selected_action=A_M3,
        )
        patches = self._base_patches(actions, {A_M3: 0, A_EAST: 0})

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch.object(
                combined,
                "_evaluate_all_zero_targeted_honor_release",
                return_value=(A_M3, inactive),
            ),
            patch.object(
                combined,
                "_select_finite_horizon_hand_value_v2",
                return_value=A_M3,
            ),
        ):
            selected, analysis = combined._evaluate_and_choose_discard(
                _input(), actions
            )

        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.selection_source,
            combined.HandValueTradeoffTargetedHonorReleaseSelectionSource.SHARED_ACTION,
        )
        self.assertFalse(analysis.action_changed_vs_champion)
        self.assertFalse(analysis.action_changed_vs_former_parent)

    def test_double_wind_pair_v2_adds_incremental_action(self) -> None:
        keep_east = A_WHITE
        keep_white = A_EAST
        actions = (keep_white, keep_east)
        inactive = _targeted_analysis(
            stage=targeted.TargetedHonorReleaseActivationStage.PARENT_SHANTEN_TOO_CLOSE,
            parent_action=keep_white,
            selected_action=keep_white,
        )
        post_hands = {
            keep_east: (EAST, EAST, M3),
            keep_white: (WHITE, WHITE, M3),
        }
        by_hand = {hand: action for action, hand in post_hands.items()}
        structural = tuple(
            SimpleNamespace(
                action=action,
                post_discard_hand=post_hands[action],
                post_discard_shanten=1,
            )
            for action in actions
        )
        patches = self._base_patches(
            actions,
            {keep_white: 0, keep_east: 0},
        )

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch.object(
                combined,
                "_evaluate_all_zero_targeted_honor_release",
                return_value=(keep_white, inactive),
            ),
            patch.object(v2, "evaluate_post_discard_hands", return_value=structural),
            patch.object(v2, "known_tile_counts", return_value={}),
            patch.object(v2, "ukeire_count", return_value=10),
            patch.object(v2, "_retained_real_value", return_value=0),
            patch.object(v2, "_yaku_route_value", return_value=0),
            patch.object(v2, "second_step_ukeire_score", return_value=0),
        ):
            selected, analysis = combined._evaluate_and_choose_discard(
                _input(), actions
            )

        self.assertIs(selected, keep_east)
        self.assertIs(
            analysis.selection_source,
            combined.HandValueTradeoffTargetedHonorReleaseSelectionSource.HAND_VALUE_V2,
        )
        self.assertEqual(
            v2._yakuhai_pair_potential(
                post_hands[keep_east],
                _input(),
                (4,) * 34,
            ),
            2,
        )
        self.assertEqual(
            v2._yakuhai_pair_potential(
                post_hands[keep_white],
                _input(),
                (4,) * 34,
            ),
            1,
        )
        self.assertIs(by_hand[post_hands[keep_east]], keep_east)

    def test_unique_positive_finite_horizon_remains_shared(self) -> None:
        actions = (A_M3, A_EAST)
        patches = self._base_patches(actions, {A_M3: 9, A_EAST: 3})

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3] as completion,
            patch.object(
                combined,
                "_evaluate_all_zero_targeted_honor_release",
                side_effect=AssertionError("positive completion must skip #174 R5"),
            ),
        ):
            selected, analysis = combined._evaluate_and_choose_discard(
                _input(), actions
            )

        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.selection_source,
            combined.HandValueTradeoffTargetedHonorReleaseSelectionSource.SHARED_ACTION,
        )
        completion.assert_called_once()

    def test_positive_completion_never_enters_targeted_r5_gate(self) -> None:
        actions = (A_M3, A_EAST)
        patches = self._base_patches(actions, {A_M3: 9, A_EAST: 9})

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3] as completion,
            patch.object(
                combined,
                "_evaluate_all_zero_targeted_honor_release",
                side_effect=AssertionError("positive completion must skip #174 R5"),
            ),
            patch.object(
                combined,
                "_current_parent_from_completion",
                return_value=A_M3,
            ),
            patch.object(
                combined,
                "_select_finite_horizon_hand_value_v2",
                return_value=A_EAST,
            ),
        ):
            selected, analysis = combined._evaluate_and_choose_discard(
                _input(), actions
            )

        self.assertIs(selected, A_EAST)
        self.assertIsNone(analysis.champion_analysis)
        completion.assert_called_once()

    def test_post_call_phase_does_not_enable_targeted_r5(self) -> None:
        actions = (A_M3, A_EAST)
        patches = self._base_patches(actions, {A_M3: 0, A_EAST: 0})

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch.object(
                combined,
                "_evaluate_all_zero_targeted_honor_release",
                side_effect=AssertionError("post-call phase must skip #174 R5"),
            ),
            patch.object(
                combined,
                "_current_parent_from_completion",
                return_value=A_M3,
            ),
            patch.object(
                combined,
                "_select_finite_horizon_hand_value_v2",
                return_value=A_EAST,
            ),
        ):
            selected, _ = combined._evaluate_and_choose_discard(
                _input(normal_turn=False), actions
            )

        self.assertIs(selected, A_EAST)

    def test_mechanism_filtered_branch_does_not_enable_targeted_r5(self) -> None:
        actions = (A_M3, A_EAST)
        patches = self._base_patches(
            actions,
            {A_M3: 0, A_EAST: 0},
            branch=targeted.TargetedHonorReleaseBranch.MECHANISM_DEFENSE_FILTERED,
        )

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch.object(
                combined,
                "_evaluate_all_zero_targeted_honor_release",
                side_effect=AssertionError("non-PUSH branch must skip #174 R5"),
            ),
            patch.object(
                combined,
                "_current_parent_from_completion",
                return_value=A_M3,
            ),
            patch.object(
                combined,
                "_select_finite_horizon_hand_value_v2",
                return_value=A_EAST,
            ),
        ):
            selected, _ = combined._evaluate_and_choose_discard(_input(), actions)

        self.assertIs(selected, A_EAST)

    def test_composition_is_independent_of_legal_action_order(self) -> None:
        selections = set()
        for actions in itertools.permutations((A_M3, A_EAST)):
            actions = tuple(actions)
            patches = self._base_patches(actions, {A_M3: 0, A_EAST: 0})
            inactive = _targeted_analysis(
                stage=(
                    targeted.TargetedHonorReleaseActivationStage.PARENT_SHANTEN_TOO_CLOSE
                ),
                parent_action=A_M3,
                selected_action=A_M3,
            )
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patch.object(
                    combined,
                    "_evaluate_all_zero_targeted_honor_release",
                    return_value=(A_M3, inactive),
                ),
                patch.object(
                    combined,
                    "_select_finite_horizon_hand_value_v2",
                    return_value=A_EAST,
                ),
            ):
                selected, _ = combined._evaluate_and_choose_discard(_input(), actions)
            selections.add(selected)
        self.assertEqual(selections, {A_EAST})


class PolicyBoundaryTest(unittest.TestCase):
    def test_non_discard_orchestration_is_inherited_from_champion(self) -> None:
        policy = combined.HandValueTradeoffTargetedHonorReleasePolicy()
        discard = A_M3
        ron = RonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            winning_tile=M4,
        )
        riichi = RiichiAction(actor=Seat.SEAT_0)

        with patch.object(
            combined,
            "_evaluate_and_choose_discard",
            side_effect=AssertionError("non-discard action must skip composition"),
        ):
            self.assertIs(
                policy.choose_action(_decision(_input(), (discard, ron))),
                ron,
            )
            self.assertIs(
                policy.choose_action(_decision(_input(), (discard, riichi))),
                riichi,
            )

    def test_only_discard_extension_point_is_overridden(self) -> None:
        policy_type = combined.HandValueTradeoffTargetedHonorReleasePolicy
        self.assertIn("_decide_discard", vars(policy_type))
        for method_name in ("_decide", "choose_action", "choose_action_with_analysis"):
            with self.subTest(method_name=method_name):
                self.assertNotIn(method_name, vars(policy_type))

    def test_policy_is_stateless_public_and_spawn_picklable(self) -> None:
        policy = combined.HandValueTradeoffTargetedHonorReleasePolicy()
        self.assertIsInstance(policy, TargetedHonorReleaseTerminalProgressionPolicy)
        self.assertEqual(vars(policy), {})
        self.assertIs(
            pickle.loads(
                pickle.dumps(combined.HandValueTradeoffTargetedHonorReleasePolicy)
            ),
            combined.HandValueTradeoffTargetedHonorReleasePolicy,
        )
        self.assertEqual(
            combined.HandValueTradeoffTargetedHonorReleasePolicy.__module__,
            "lisjong.policies.hand_value_tradeoff_targeted_honor_release",
        )


if __name__ == "__main__":
    unittest.main()
