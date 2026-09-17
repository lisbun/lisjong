"""Issue #174 targeted honor-release terminal progression Policy tests."""

import pickle
import unittest
from unittest.mock import patch

import lisjong.policies.targeted_honor_release_terminal_progression as targeted
import lisjong.policies.terminal_shanten_progression_mechanism_riichi_defense as progression
from lisjong.policies.finite_horizon_completion import FiniteHorizonCandidateEvaluation
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    HandValueCandidateEvaluation,
    _retained_real_value,
)
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    MechanismRiichiDefenseYakuhaiCallPolicy,
)
from lisjong.policies.terminal_shanten_progression_mechanism_riichi_defense import (
    ProgressionCandidateEvaluation,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind


def _tile(category: TileCategory, rank: int, *, red: bool = False) -> Tile:
    return Tile(TileType(category, rank), is_red=red)


M3 = _tile(TileCategory.MANZU, 3)
M4 = _tile(TileCategory.MANZU, 4)
M5 = _tile(TileCategory.MANZU, 5)
M5_RED = _tile(TileCategory.MANZU, 5, red=True)
EAST = _tile(TileCategory.HONOR, 1)
SOUTH = _tile(TileCategory.HONOR, 2)


def _action(tile: Tile) -> DiscardAction:
    return DiscardAction(Seat.SEAT_0, tile, False)


A_M3 = _action(M3)
A_M4 = _action(M4)
A_EAST = _action(EAST)
A_SOUTH = _action(SOUTH)


def _player(*, melds: tuple[PublicMeld, ...] = ()) -> PlayerPublicState:
    return PlayerPublicState(25000, (), melds, RiichiState.NONE)


def _input(
    *,
    drawn_tile: Tile | None = M5,
    own_melds: tuple[PublicMeld, ...] = (),
    dora_indicators: tuple[Tile, ...] = (),
) -> PolicyInput:
    players = [_player() for _ in range(4)]
    players[0] = _player(melds=own_melds)
    concealed = (M3, M4, M5, EAST, SOUTH)
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
        players=tuple(players),
        own_hand=OwnHandState(concealed, drawn_tile),
    )


def _hva(
    action: DiscardAction,
    *,
    shanten: int = 3,
    ukeire: int | None = 8,
    retained: int | None = 0,
    route: int | None = 0,
    second: int | None = 20,
) -> HandValueCandidateEvaluation:
    return HandValueCandidateEvaluation(
        action=action,
        post_discard_shanten=shanten,
        current_ukeire_count=ukeire,
        retained_real_value=retained,
        yaku_route_value=route,
        second_step_ukeire_score=second,
    )


def _completion(
    action: DiscardAction, mass: int = 0
) -> FiniteHorizonCandidateEvaluation:
    return FiniteHorizonCandidateEvaluation(action, mass)


def _progression(action: DiscardAction, mass: int) -> ProgressionCandidateEvaluation:
    counts = (0, mass, 0, 0, 0, 0, 0, 0, 0)
    return ProgressionCandidateEvaluation(
        action=action,
        completion_mass=0,
        root_post_discard_shanten=3,
        terminal_shanten_mass=mass,
        terminal_shanten_counts=counts,
    )


class PolicyBoundaryTest(unittest.TestCase):
    def test_exact_parent_public_generation_is_stateless_and_picklable(self) -> None:
        from lisjong.policies import TargetedHonorReleaseTerminalProgressionPolicy

        self.assertIs(
            TargetedHonorReleaseTerminalProgressionPolicy,
            targeted.TargetedHonorReleaseTerminalProgressionPolicy,
        )
        policy = TargetedHonorReleaseTerminalProgressionPolicy()
        self.assertIsInstance(policy, MechanismRiichiDefenseYakuhaiCallPolicy)
        self.assertEqual(vars(policy), {})
        self.assertIs(
            pickle.loads(pickle.dumps(TargetedHonorReleaseTerminalProgressionPolicy)),
            TargetedHonorReleaseTerminalProgressionPolicy,
        )

    def test_non_discard_orchestration_is_inherited_from_exact_parent(self) -> None:
        policy_type = targeted.TargetedHonorReleaseTerminalProgressionPolicy
        self.assertIs(
            policy_type.choose_action,
            MechanismRiichiDefenseYakuhaiCallPolicy.choose_action,
        )
        self.assertIs(
            policy_type._decide, MechanismRiichiDefenseYakuhaiCallPolicy._decide
        )
        self.assertIs(
            policy_type.choose_action_with_analysis,
            MechanismRiichiDefenseYakuhaiCallPolicy.choose_action_with_analysis,
        )

    def test_exact_r5_implementation_is_single_source_reuse(self) -> None:
        self.assertIs(
            targeted._evaluate_progression_candidates,
            progression._evaluate_progression_candidates,
        )
        self.assertIs(
            targeted._TerminalShantenProgressionEvaluator,
            progression._TerminalShantenProgressionEvaluator,
        )

    def test_discard_analysis_uses_standard_trace_contract(self) -> None:
        analysis = targeted._analysis(
            stage=targeted.TargetedHonorReleaseActivationStage.NO_QUALIFYING_HONOR_PEER,
            parent_action=A_M3,
            selected_action=A_M3,
            branch=targeted.TargetedHonorReleaseBranch.PUSH,
            closed_hand=True,
            parent_snapshot=None,
            eligible_candidate_count=1,
        )
        with patch.object(
            targeted,
            "_evaluate_and_choose_discard",
            return_value=(A_M3, analysis),
        ) as evaluate:
            decision = targeted.TargetedHonorReleaseTerminalProgressionPolicy()._decide_discard(
                _input(), (A_M3,)
            )
        self.assertIs(decision.action, A_M3)
        self.assertIs(decision.analysis, analysis)
        self.assertIsInstance(decision.analysis, targeted.AnalysisTrace)
        evaluate.assert_called_once()

    def test_ankan_only_is_closed_but_open_meld_is_not(self) -> None:
        ankan = PublicMeld(MeldKind.ANKAN, (EAST,) * 4, None, None)
        pon = PublicMeld(MeldKind.PON, (EAST,) * 3, Seat.SEAT_1, EAST)
        self.assertTrue(targeted._is_closed_hand(_input(own_melds=(ankan,))))
        self.assertFalse(targeted._is_closed_hand(_input(own_melds=(pon,))))


class TargetUniverseTest(unittest.TestCase):
    def test_requires_exact_shanten_ukeire_and_retained_real_value_equality(
        self,
    ) -> None:
        snapshots = (
            _hva(A_M3, shanten=3, ukeire=8, retained=2),
            _hva(A_EAST, shanten=3, ukeire=8, retained=2),
            _hva(A_SOUTH, shanten=4, ukeire=8, retained=2),
            _hva(A_M4, shanten=3, ukeire=7, retained=2),
            _hva(_action(M5), shanten=3, ukeire=8, retained=1),
        )
        self.assertEqual(
            tuple(
                snapshot.action
                for snapshot in targeted._target_candidates(A_M3, snapshots)
            ),
            (A_M3, A_EAST),
        )

    def test_retained_real_value_loss_cannot_enter_target_universe(self) -> None:
        snapshots = (
            _hva(A_M3, retained=2),
            _hva(A_EAST, retained=1),
        )
        self.assertEqual(
            tuple(
                snapshot.action
                for snapshot in targeted._target_candidates(A_M3, snapshots)
            ),
            (A_M3,),
        )

    def test_concrete_dora_red_dora_and_yakuhai_losses_are_excluded(self) -> None:
        cases = (
            (
                "indicator dora",
                _input(dora_indicators=(M4,)),
                (M5,),
                (M3,),
            ),
            (
                "red dora",
                _input(),
                (M5_RED,),
                (M3,),
            ),
            (
                "completed double-wind yakuhai",
                _input(),
                (EAST, EAST, EAST),
                (EAST, EAST, M3),
            ),
        )
        for label, policy_input, parent_hand, peer_hand in cases:
            with self.subTest(label=label):
                parent_value = _retained_real_value(parent_hand, policy_input)
                peer_value = _retained_real_value(peer_hand, policy_input)
                self.assertGreater(parent_value, peer_value)
                snapshots = (
                    _hva(A_M3, retained=parent_value),
                    _hva(A_EAST, retained=peer_value),
                )
                self.assertEqual(
                    tuple(
                        snapshot.action
                        for snapshot in targeted._target_candidates(A_M3, snapshots)
                    ),
                    (A_M3,),
                )


class DecisiveStageTest(unittest.TestCase):
    def test_retained_real_value_is_classified_without_guessing(self) -> None:
        snapshots = (_hva(A_M3, retained=2), _hva(A_EAST, retained=1))
        self.assertIs(
            targeted._classify_hva_decisive_stage(A_M3, snapshots),
            targeted.HandValueDecisiveStage.RETAINED_REAL_VALUE,
        )

    def test_yaku_route_is_classified_without_guessing(self) -> None:
        snapshots = (_hva(A_M3, route=2), _hva(A_EAST, route=1))
        self.assertIs(
            targeted._classify_hva_decisive_stage(A_M3, snapshots),
            targeted.HandValueDecisiveStage.YAKU_ROUTE,
        )

    def test_second_step_and_stable_tie_are_distinguished(self) -> None:
        second = (_hva(A_M3, second=30), _hva(A_EAST, second=20))
        tied = (_hva(A_M3, second=20), _hva(A_EAST, second=20))
        self.assertIs(
            targeted._classify_hva_decisive_stage(A_M3, second),
            targeted.HandValueDecisiveStage.SECOND_STEP,
        )
        expected_stable = min((A_M3, A_EAST), key=targeted._discard_action_sort_key)
        self.assertIs(
            targeted._classify_hva_decisive_stage(expected_stable, tied),
            targeted.HandValueDecisiveStage.STABLE_TIE,
        )

    def test_incomplete_snapshot_is_explicitly_unresolved(self) -> None:
        snapshots = (
            _hva(A_M3, retained=None, route=None, second=None),
            _hva(A_EAST, retained=None, route=None, second=None),
        )
        self.assertIs(
            targeted._classify_hva_decisive_stage(A_M3, snapshots),
            targeted.HandValueDecisiveStage.UNRESOLVED,
        )


class ActivationAndSwitchTest(unittest.TestCase):
    def _active_patches(self, snapshots, progression_results, *, parent_action=A_M3):
        actions = tuple(snapshot.action for snapshot in snapshots)
        return (
            actions,
            patch.object(
                targeted,
                "_classify_branch",
                return_value=(targeted.TargetedHonorReleaseBranch.PUSH, actions),
            ),
            patch.object(targeted, "_root_remaining_counts", return_value=(1,) * 34),
            patch.object(
                targeted,
                "_evaluate_completion_masses",
                return_value=tuple(_completion(action) for action in actions),
            ),
            patch.object(
                targeted,
                "_hand_value_aware_evaluate_and_choose_discard",
                return_value=(parent_action, snapshots),
            ),
            patch.object(
                targeted,
                "_evaluate_progression_candidates",
                return_value=progression_results,
            ),
        )

    def test_post_call_or_unknown_phase_never_runs_r5(self) -> None:
        with (
            patch.object(
                targeted,
                "_classify_branch",
                return_value=(
                    targeted.TargetedHonorReleaseBranch.PUSH,
                    (A_M3, A_EAST),
                ),
            ),
            patch.object(targeted, "_parent_action", return_value=A_M3),
            patch.object(
                targeted,
                "_evaluate_progression_candidates",
                side_effect=AssertionError("R5 must not run"),
            ),
        ):
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(drawn_tile=None), (A_M3, A_EAST)
            )
        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.NOT_NORMAL_TURN,
        )

    def test_open_hand_never_runs_r5(self) -> None:
        pon = PublicMeld(MeldKind.PON, (EAST,) * 3, Seat.SEAT_1, EAST)
        with (
            patch.object(
                targeted,
                "_classify_branch",
                return_value=(
                    targeted.TargetedHonorReleaseBranch.PUSH,
                    (A_M3, A_EAST),
                ),
            ),
            patch.object(targeted, "_parent_action", return_value=A_M3),
            patch.object(
                targeted,
                "_evaluate_progression_candidates",
                side_effect=AssertionError("R5 must not run"),
            ),
        ):
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(own_melds=(pon,)), (A_M3, A_EAST)
            )
        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.OPEN_HAND,
        )

    def test_non_push_branch_never_runs_r5(self) -> None:
        with (
            patch.object(
                targeted,
                "_classify_branch",
                return_value=(
                    targeted.TargetedHonorReleaseBranch.MECHANISM_DEFENSE_FILTERED,
                    (A_M3,),
                ),
            ),
            patch.object(targeted, "_parent_action", return_value=A_M3),
            patch.object(
                targeted,
                "_evaluate_progression_candidates",
                side_effect=AssertionError("R5 must not run"),
            ),
        ):
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), (A_M3, A_EAST)
            )
        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.NON_PUSH_BRANCH,
        )

    def test_positive_completion_preserves_parent_without_r5(self) -> None:
        actions = (A_M3, A_EAST)
        with (
            patch.object(
                targeted,
                "_classify_branch",
                return_value=(targeted.TargetedHonorReleaseBranch.PUSH, actions),
            ),
            patch.object(targeted, "_root_remaining_counts", return_value=(1,) * 34),
            patch.object(
                targeted,
                "_evaluate_completion_masses",
                return_value=(_completion(A_M3, 3), _completion(A_EAST, 1)),
            ),
            patch.object(targeted, "_select_from_completion_masses", return_value=A_M3),
            patch.object(
                targeted,
                "_evaluate_progression_candidates",
                side_effect=AssertionError("R5 must not run"),
            ),
        ):
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), actions
            )
        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.POSITIVE_COMPLETION,
        )

    def test_no_equal_value_honor_peer_never_runs_r5(self) -> None:
        actions = (A_M3, A_EAST)
        snapshots = (_hva(A_M3, retained=2), _hva(A_EAST, retained=1))
        with (
            patch.object(
                targeted,
                "_classify_branch",
                return_value=(targeted.TargetedHonorReleaseBranch.PUSH, actions),
            ),
            patch.object(targeted, "_root_remaining_counts", return_value=(1,) * 34),
            patch.object(
                targeted,
                "_evaluate_completion_masses",
                return_value=tuple(_completion(action) for action in actions),
            ),
            patch.object(
                targeted,
                "_hand_value_aware_evaluate_and_choose_discard",
                return_value=(A_M3, snapshots),
            ),
            patch.object(
                targeted,
                "_evaluate_progression_candidates",
                side_effect=AssertionError("R5 must not run"),
            ),
        ):
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), actions
            )
        self.assertIs(selected, A_M3)
        self.assertEqual(analysis.target_candidate_count, 1)
        self.assertEqual(analysis.honor_target_candidate_count, 0)

    def test_shanten_two_parent_never_runs_r5(self) -> None:
        snapshots = (_hva(A_M3, shanten=2), _hva(A_EAST, shanten=2))
        actions, *patches = self._active_patches(snapshots, ())
        with patches[0], patches[1], patches[2], patches[3], patches[4] as run_r5:
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), actions
            )
        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.PARENT_SHANTEN_TOO_CLOSE,
        )
        run_r5.assert_not_called()

    def test_parent_not_minimum_shanten_never_runs_r5(self) -> None:
        snapshots = (_hva(A_M3, shanten=4), _hva(A_EAST, shanten=3))
        actions, *patches = self._active_patches(snapshots, ())
        with patches[0], patches[1], patches[2], patches[3], patches[4] as run_r5:
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), actions
            )
        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.PARENT_NOT_MINIMUM_SHANTEN,
        )
        run_r5.assert_not_called()

    def test_parent_not_maximum_ukeire_never_runs_r5(self) -> None:
        snapshots = (_hva(A_M3, ukeire=7), _hva(A_EAST, ukeire=8))
        actions, *patches = self._active_patches(snapshots, ())
        with patches[0], patches[1], patches[2], patches[3], patches[4] as run_r5:
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), actions
            )
        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.PARENT_NOT_MAXIMUM_UKEIRE,
        )
        run_r5.assert_not_called()

    def test_parent_already_honor_never_runs_r5(self) -> None:
        snapshots = (_hva(A_EAST, route=2), _hva(A_M3, route=1))
        actions, *patches = self._active_patches(snapshots, (), parent_action=A_EAST)
        with patches[0], patches[1], patches[2], patches[3], patches[4] as run_r5:
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), actions
            )
        self.assertIs(selected, A_EAST)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.PARENT_ALREADY_HONOR,
        )
        run_r5.assert_not_called()

    def test_honor_only_unique_r5_best_switches_once(self) -> None:
        snapshots = (
            _hva(A_M3, route=2),
            _hva(A_EAST, route=1),
            _hva(A_M4, ukeire=7),
        )
        progression_results = (_progression(A_M3, 30), _progression(A_EAST, 10))
        actions, *patches = self._active_patches(snapshots, progression_results)
        with patches[0], patches[1], patches[2], patches[3], patches[4] as run_r5:
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), actions
            )
        self.assertIs(selected, A_EAST)
        self.assertTrue(analysis.action_changed)
        self.assertEqual(analysis.target_candidate_count, 2)
        self.assertEqual(analysis.honor_target_candidate_count, 1)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.R5_HONOR_ONLY_SWITCH,
        )
        run_r5.assert_called_once()
        evaluated = run_r5.call_args.args[1]
        self.assertEqual(tuple(value.action for value in evaluated), (A_M3, A_EAST))

    def test_r5_tie_containing_parent_preserves_parent(self) -> None:
        snapshots = (_hva(A_M3, route=2), _hva(A_EAST, route=1))
        progression_results = (_progression(A_M3, 10), _progression(A_EAST, 10))
        actions, *patches = self._active_patches(snapshots, progression_results)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), actions
            )
        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.R5_PARENT_BEST,
        )

    def test_mixed_honor_suited_r5_best_preserves_parent(self) -> None:
        snapshots = (
            _hva(A_M3, route=2),
            _hva(A_EAST, route=1),
            _hva(A_M4, route=1),
        )
        progression_results = (
            _progression(A_M3, 30),
            _progression(A_EAST, 10),
            _progression(A_M4, 10),
        )
        actions, *patches = self._active_patches(snapshots, progression_results)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), actions
            )
        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.R5_NON_HONOR_ONLY_BEST,
        )

    def test_suited_only_r5_best_preserves_parent(self) -> None:
        snapshots = (
            _hva(A_M3, route=2),
            _hva(A_EAST, route=1),
            _hva(A_M4, route=1),
        )
        progression_results = (
            _progression(A_M3, 30),
            _progression(A_EAST, 20),
            _progression(A_M4, 10),
        )
        actions, *patches = self._active_patches(snapshots, progression_results)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), actions
            )
        self.assertIs(selected, A_M3)
        self.assertIs(
            analysis.activation_stage,
            targeted.TargetedHonorReleaseActivationStage.R5_NON_HONOR_ONLY_BEST,
        )

    def test_active_selection_is_independent_of_legal_action_order(self) -> None:
        results = set()
        for snapshots in (
            (_hva(A_M3, route=2), _hva(A_EAST, route=1)),
            (_hva(A_EAST, route=1), _hva(A_M3, route=2)),
        ):
            progression_results = tuple(
                _progression(snapshot.action, 30 if snapshot.action == A_M3 else 10)
                for snapshot in snapshots
            )
            actions, *patches = self._active_patches(snapshots, progression_results)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                selected, _ = targeted._evaluate_and_choose_discard(_input(), actions)
            results.add(selected)
        self.assertEqual(results, {A_EAST})

    def test_multiple_honor_only_best_reuses_hva_tiebreak(self) -> None:
        snapshots = (
            _hva(A_M3, route=2),
            _hva(A_EAST, route=1),
            _hva(A_SOUTH, route=1),
        )
        progression_results = (
            _progression(A_M3, 30),
            _progression(A_EAST, 10),
            _progression(A_SOUTH, 10),
        )
        actions = tuple(snapshot.action for snapshot in snapshots)
        hva_tie_snapshots = (_hva(A_EAST), _hva(A_SOUTH))
        with (
            patch.object(
                targeted,
                "_classify_branch",
                return_value=(targeted.TargetedHonorReleaseBranch.PUSH, actions),
            ),
            patch.object(targeted, "_root_remaining_counts", return_value=(1,) * 34),
            patch.object(
                targeted,
                "_evaluate_completion_masses",
                return_value=tuple(_completion(action) for action in actions),
            ),
            patch.object(
                targeted,
                "_hand_value_aware_evaluate_and_choose_discard",
                side_effect=((A_M3, snapshots), (A_SOUTH, hva_tie_snapshots)),
            ) as hva,
            patch.object(
                targeted,
                "_evaluate_progression_candidates",
                return_value=progression_results,
            ) as run_r5,
        ):
            selected, analysis = targeted._evaluate_and_choose_discard(
                _input(), actions
            )
        self.assertIs(selected, A_SOUTH)
        self.assertEqual(hva.call_count, 2)
        self.assertEqual(
            hva.call_args_list[1].args[1],
            (A_EAST, A_SOUTH),
        )
        run_r5.assert_called_once()
        self.assertTrue(analysis.action_changed)


if __name__ == "__main__":
    unittest.main()
