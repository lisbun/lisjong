"""Issue #172 offensive-efficiency diagnostic seam tests。"""

import ast
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import lisjong.policies.mechanism_riichi_defense_offensive_efficiency_diagnostic as diagnostic
import lisjong.policies.mechanism_riichi_defense_yakuhai_call as mechanism
import lisjong.policies.terminal_shanten_progression_mechanism_riichi_defense as progression
from lisjong.policies import MechanismRiichiDefenseYakuhaiCallPolicy
from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    _evaluate_completion_masses,
    _FiniteHorizonEvaluator,
    _root_remaining_counts,
)
from lisjong.policies.mechanism_riichi_defense_offensive_efficiency_diagnostic import (
    MechanismRiichiDefenseOffensiveEfficiencyAnalysis,
    OffensiveEfficiencyBranch,
    OffensiveEfficiencyCandidateEvaluation,
    UniverseEfficiencySummary,
    analyze_mechanism_riichi_defense_offensive_efficiency,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

_CATEGORIES = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}
_CATEGORY_SIZES = {
    TileCategory.MANZU: 9,
    TileCategory.PINZU: 9,
    TileCategory.SOUZU: 9,
    TileCategory.HONOR: 7,
}
_MAX_COPIES_PER_TILE_TYPE = 4


def _type(category: TileCategory, rank: int) -> TileType:
    return TileType(category, rank)


def _tile(category: TileCategory, rank: int) -> Tile:
    return Tile(_type(category, rank))


def _hand(spec: str) -> tuple[Tile, ...]:
    result: list[Tile] = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        for raw_rank in ranks:
            rank = int(raw_rank)
            result.append(_tile(_CATEGORIES[character], 5 if rank == 0 else rank))
        ranks = ""
    if ranks:
        raise ValueError("trailing ranks")
    return tuple(result)


def _all_tile_types() -> tuple[TileType, ...]:
    return tuple(
        _type(category, rank)
        for category, size in _CATEGORY_SIZES.items()
        for rank in range(1, size + 1)
    )


def _discard_history(tile: Tile, order: int = 0) -> Discard:
    return Discard(tile=tile, tsumogiri=False, order=order, called_by=None)


def _player(
    *,
    riichi: RiichiState = RiichiState.NONE,
    discards: tuple[Tile, ...] = (),
) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=tuple(
            _discard_history(tile, order) for order, tile in enumerate(discards)
        ),
        melds=(),
        riichi=riichi,
    )


def _input(
    concealed: tuple[Tile, ...],
    *,
    threats: tuple[tuple[Tile, ...], ...] = (),
) -> PolicyInput:
    players = [_player() for _ in range(4)]
    for offset, discards in enumerate(threats, start=1):
        players[offset] = _player(riichi=RiichiState.ACCEPTED, discards=discards)
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
        players=tuple(players),
        own_hand=OwnHandState(concealed_tiles=concealed, drawn_tile=None),
    )


def _custom_input(
    concealed: tuple[Tile, ...], players: tuple[PlayerPublicState, ...]
) -> PolicyInput:
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(),
            live_wall_tiles_remaining=20,
        ),
        players=players,
        own_hand=OwnHandState(concealed_tiles=concealed, drawn_tile=None),
    )


def _restricted_input(
    concealed: tuple[Tile, ...], drawable_specs: tuple[str, ...]
) -> PolicyInput:
    """remaining inventoryをdrawable牌種だけへ絞ったPolicyInputを組み立てる。

    #169 testの`_restricted_input()`と同じPolicy-visibleな公開河構成方法。
    """
    drawable = {_hand(spec)[0].tile_type for spec in drawable_specs}
    accounted: dict[TileType, int] = {}
    for tile in concealed:
        accounted[tile.tile_type] = accounted.get(tile.tile_type, 0) + 1
    red_seen = {tile.tile_type.category for tile in concealed if tile.is_red}

    visible: list[Tile] = []
    for tile_type in _all_tile_types():
        if tile_type in drawable:
            continue
        copies = _MAX_COPIES_PER_TILE_TYPE - accounted.get(tile_type, 0)
        needs_red = (
            tile_type.rank == 5
            and tile_type.category is not TileCategory.HONOR
            and tile_type.category not in red_seen
        )
        for index in range(copies):
            visible.append(Tile(tile_type, is_red=needs_red and index == 0))

    rivers: list[list[Tile]] = [[], [], [], []]
    for index, tile in enumerate(visible):
        rivers[index % 4].append(tile)
    players = tuple(_player(discards=tuple(river)) for river in rivers)
    return _custom_input(concealed, players)


def _action(tile: Tile) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=False)


def _actions(*specs: str) -> tuple[DiscardAction, ...]:
    return tuple(_action(_hand(spec)[0]) for spec in specs)


def _distinct_discard_actions(
    concealed: tuple[Tile, ...],
) -> tuple[DiscardAction, ...]:
    seen: set[Tile] = set()
    actions: list[DiscardAction] = []
    for tile in concealed:
        if tile in seen:
            continue
        seen.add(tile)
        actions.append(_action(tile))
    return tuple(actions)


# ---------------------------------------------------------------------------
# real-hand fixtures reused across multiple assertions
# ---------------------------------------------------------------------------


def _fold_common_genbutsu_fixture() -> tuple[PolicyInput, tuple[DiscardAction, ...]]:
    """R1 / R2 applicabilityがuniverse間で異なるFOLD_COMMON_GENBUTSU fixture。

    discard(3m) -> shanten3、discard(5z) -> shanten2。opponent riverが3mだけを
    含むため、eligible universeは{3m}だけになり、full legal universeのbest
    (5z, shanten2)を失う。
    """
    concealed = _hand("345m56679s333517z")
    policy_input = _input(concealed, threats=(_hand("3m"),))
    return policy_input, _actions("3m", "5z")


def _second_step_applicability_fixture() -> tuple[
    PolicyInput, tuple[DiscardAction, ...]
]:
    """R3 applicabilityがuniverse間で異なるFOLD_COMMON_GENBUTSU fixture。

    discard(4m)がfull legal universeの現在受け入れを単独で最大化するため、
    discard(2m) / discard(3m)はfull legalではsecond-step stageへ到達しない。
    eligible universe（common genbutsuの2m・3mだけ）ではこの2candidateが
    tieし、second-step評価が発火する。
    """
    concealed = _hand("123m456m789p11s35p1z")
    policy_input = _input(concealed, threats=(_hand("2m3m"),))
    return policy_input, _actions("4m", "2m", "3m")


def _other_current_fallback_fixture() -> tuple[PolicyInput, tuple[DiscardAction, ...]]:
    """Issue #172が明示する代表例: riichi + best shanten==1 + no common genbutsu。"""
    concealed = _hand("345m56679s333577z")
    policy_input = _input(concealed, threats=(_hand("1m"),))
    return policy_input, _actions("5s", "9s")


def _push_fixture() -> tuple[PolicyInput, tuple[DiscardAction, ...]]:
    concealed = _hand("123m456m789p11s35p1z")
    policy_input = _input(concealed)
    return policy_input, _actions("4m", "2m", "3m")


def _mechanism_activated_fixture() -> tuple[PolicyInput, tuple[DiscardAction, ...]]:
    concealed = _hand("345m56679s333517z")
    policy_input = _input(concealed, threats=(_hand("9p"),))
    return policy_input, _actions("3m", "5z")


def _all_zero_progression_fixture() -> tuple[PolicyInput, tuple[DiscardAction, ...]]:
    concealed = _hand("147m258p369s13577z")
    actions = _distinct_discard_actions(concealed)
    policy_input = _restricted_input(
        concealed, tuple(f"{rank}m" for rank in range(1, 10))
    )
    return policy_input, actions


class BranchClassificationTest(unittest.TestCase):
    """Issue #172が要求する4branch classificationを固定する。"""

    def test_push_branch_when_no_defensive_restriction_applies(self) -> None:
        policy_input, actions = _push_fixture()
        analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions
        )
        self.assertIs(analysis.branch, OffensiveEfficiencyBranch.PUSH)
        self.assertEqual(analysis.baseline_eligible_actions, actions)

    def test_fold_common_genbutsu_branch(self) -> None:
        policy_input, actions = _fold_common_genbutsu_fixture()
        analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions
        )
        self.assertIs(analysis.branch, OffensiveEfficiencyBranch.FOLD_COMMON_GENBUTSU)
        self.assertEqual(analysis.baseline_eligible_actions, _actions("3m"))

    def test_mechanism_defense_filtered_branch(self) -> None:
        policy_input, actions = _mechanism_activated_fixture()
        analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions
        )
        self.assertIs(
            analysis.branch, OffensiveEfficiencyBranch.MECHANISM_DEFENSE_FILTERED
        )

    def test_other_current_fallback_branch(self) -> None:
        """Issue本文の代表例: riichi + best shanten==1 + no common genbutsu。"""
        policy_input, actions = _other_current_fallback_fixture()
        analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions
        )
        self.assertIs(analysis.branch, OffensiveEfficiencyBranch.OTHER_CURRENT_FALLBACK)
        self.assertEqual(analysis.baseline_eligible_actions, actions)

    def test_mechanism_defense_filtered_survives_a_danger_tie(self) -> None:
        """danger tieでcandidate setが縮まらないactivationでも正しく分類する。

        `mechanism_eligible_actions != input_actions`のようなset size比較では
        このケースを`ACTIVATED`と判定できない。
        """
        one = _tile(TileCategory.MANZU, 1)
        honor = _tile(TileCategory.HONOR, 5)
        actions = (_action(one), _action(honor))
        policy_input = _input((one, honor), threats=(_hand("9p"),))

        with (
            patch.object(
                mechanism,
                "_evaluate_post_discard_hands",
                return_value=tuple(
                    SimpleNamespace(action=action, post_discard_shanten=2)
                    for action in actions
                ),
            ),
            patch.object(
                mechanism,
                "_classical_riichi_danger_score",
                return_value=mechanism._ClassicalRiichiDangerBreakdown(4, 0, 0, 0, 0),
            ),
        ):
            branch, eligible_actions = diagnostic._classify_branch(
                policy_input, actions
            )
        self.assertIs(branch, OffensiveEfficiencyBranch.MECHANISM_DEFENSE_FILTERED)
        self.assertEqual(eligible_actions, actions)


class UniverseSeparationTest(unittest.TestCase):
    """full legal universeとbaseline eligible universeの分離を固定する。"""

    def test_full_legal_and_eligible_disagree_on_best_shanten(self) -> None:
        policy_input, actions = _fold_common_genbutsu_fixture()
        analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions
        )
        self.assertEqual(analysis.full_legal_summary.best_post_discard_shanten, 2)
        self.assertEqual(
            analysis.baseline_eligible_summary.best_post_discard_shanten, 3
        )
        self.assertNotEqual(
            analysis.full_legal_summary.best_post_discard_shanten,
            analysis.baseline_eligible_summary.best_post_discard_shanten,
        )

    def test_r2_applicability_differs_between_universes(self) -> None:
        """full legalではminimum-shanten tier未到達、eligibleでは唯一の候補として到達する。"""
        policy_input, actions = _fold_common_genbutsu_fixture()
        analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions
        )
        manzu_three = next(
            candidate
            for candidate in analysis.candidate_evaluations
            if candidate.action.tile.tile_type == _type(TileCategory.MANZU, 3)
        )
        self.assertIsNone(manzu_three.full_legal_current_ukeire_count)
        self.assertIsNotNone(manzu_three.baseline_eligible_current_ukeire_count)

    def test_r3_applicability_differs_between_universes(self) -> None:
        """full legalでは第3候補が現在受け入れを独占し、second-stepが発火しない。

        eligible universeでは共通現物の2候補が受け入れでもtieし、second-step
        評価が発火する。
        """
        policy_input, actions = _second_step_applicability_fixture()
        analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions
        )
        genbutsu_candidates = tuple(
            candidate
            for candidate in analysis.candidate_evaluations
            if candidate.baseline_eligible
        )
        self.assertEqual(len(genbutsu_candidates), 2)
        for candidate in genbutsu_candidates:
            self.assertIsNone(candidate.full_legal_second_step_ukeire_score)
            self.assertIsNotNone(candidate.baseline_eligible_second_step_ukeire_score)
        self.assertIsNone(analysis.full_legal_summary.second_step_regret)
        self.assertIsNotNone(analysis.baseline_eligible_summary.second_step_regret)

    def test_selected_action_is_always_a_member_of_the_eligible_set(self) -> None:
        for policy_input, actions in (
            _push_fixture(),
            _fold_common_genbutsu_fixture(),
            _mechanism_activated_fixture(),
            _other_current_fallback_fixture(),
        ):
            with self.subTest(actions=actions):
                analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
                    policy_input, actions
                )
                self.assertIn(
                    analysis.baseline_selected_action,
                    analysis.baseline_eligible_actions,
                )


class CompletionMassReuseTest(unittest.TestCase):
    """R4 FiniteHorizon completion massの exact reuse / no-duplicate-work を固定する。"""

    def test_completion_mass_is_computed_once_for_the_full_legal_universe(
        self,
    ) -> None:
        policy_input, actions = _fold_common_genbutsu_fixture()
        with patch.object(
            diagnostic,
            "_evaluate_completion_masses",
            wraps=diagnostic._evaluate_completion_masses,
        ) as spy:
            analyze_mechanism_riichi_defense_offensive_efficiency(policy_input, actions)
        self.assertEqual(spy.call_count, 1)

    def test_completion_mass_matches_independent_evaluation_for_both_universes(
        self,
    ) -> None:
        policy_input, actions = _fold_common_genbutsu_fixture()
        analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions
        )
        remaining_counts = _root_remaining_counts(policy_input)
        reference = {
            evaluation.action: evaluation.completion_mass
            for evaluation in _evaluate_completion_masses(
                policy_input,
                actions,
                remaining_counts,
                DEFAULT_HORIZON,
                _FiniteHorizonEvaluator(),
            )
        }
        for candidate in analysis.candidate_evaluations:
            self.assertEqual(candidate.completion_mass, reference[candidate.action])
        self.assertEqual(
            analysis.full_legal_summary.best_completion_mass,
            max(reference.values()),
        )
        eligible_values = tuple(
            mass
            for action, mass in reference.items()
            if action in analysis.baseline_eligible_actions
        )
        self.assertEqual(
            analysis.baseline_eligible_summary.best_completion_mass,
            max(eligible_values),
        )


class TerminalProgressionOptInTest(unittest.TestCase):
    """R5 explicit opt-in / #169 exact reuse / default起動禁止を固定する。"""

    def test_default_call_does_not_invoke_the_progression_evaluator(self) -> None:
        policy_input, actions = _all_zero_progression_fixture()
        with patch.object(
            diagnostic,
            "_evaluate_progression_candidates",
            side_effect=AssertionError("progression must not run by default"),
        ):
            analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
                policy_input, actions
            )
        self.assertFalse(analysis.include_terminal_progression)
        self.assertIsNone(analysis.terminal_progression_summary)
        self.assertTrue(
            all(
                candidate.terminal_shanten_mass is None
                for candidate in analysis.candidate_evaluations
            )
        )

    def test_opt_in_all_zero_matches_the_exact_169_evaluator(self) -> None:
        policy_input, actions = _all_zero_progression_fixture()
        analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions, include_terminal_progression=True
        )
        self.assertTrue(analysis.baseline_eligible_summary.completion_all_zero)
        self.assertIsNotNone(analysis.terminal_progression_summary)

        remaining_counts = _root_remaining_counts(policy_input)
        completion_evaluations = _evaluate_completion_masses(
            policy_input,
            actions,
            remaining_counts,
            DEFAULT_HORIZON,
            _FiniteHorizonEvaluator(),
        )
        expected = {
            candidate.action: candidate.terminal_shanten_mass
            for candidate in progression._evaluate_progression_candidates(
                policy_input,
                completion_evaluations,
                remaining_counts,
                DEFAULT_HORIZON,
                progression._TerminalShantenProgressionEvaluator(),
            )
        }
        actual = {
            candidate.action: candidate.terminal_shanten_mass
            for candidate in analysis.candidate_evaluations
            if candidate.baseline_eligible
        }
        self.assertEqual(actual, expected)
        best = min(expected.values())
        self.assertEqual(
            analysis.terminal_progression_summary.best_terminal_shanten_mass, best
        )

    def test_opt_in_without_all_zero_stays_inactive(self) -> None:
        policy_input, actions = _second_step_applicability_fixture()
        with patch.object(
            diagnostic,
            "_evaluate_progression_candidates",
            side_effect=AssertionError("progression must not run when not all-zero"),
        ):
            analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
                policy_input, actions, include_terminal_progression=True
            )
        self.assertIsNone(analysis.terminal_progression_summary)


class ProductionSelectionParityTest(unittest.TestCase):
    """baseline_selected_actionがproduction selectionとexact一致することを固定する。"""

    def test_matches_production_selection_across_representative_branches(
        self,
    ) -> None:
        fixtures = (
            ("push", _push_fixture()),
            ("fold_common_genbutsu", _fold_common_genbutsu_fixture()),
            ("mechanism_defense_filtered", _mechanism_activated_fixture()),
            ("other_current_fallback", _other_current_fallback_fixture()),
        )
        for label, (policy_input, actions) in fixtures:
            with self.subTest(branch=label):
                analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
                    policy_input, actions
                )
                decision = DecisionContext(input=policy_input, legal_actions=actions)
                production = MechanismRiichiDefenseYakuhaiCallPolicy().choose_action(
                    decision
                )
                self.assertEqual(analysis.baseline_selected_action, production)


class LegalActionOrderIndependenceTest(unittest.TestCase):
    def test_permuted_legal_actions_produce_the_same_analysis(self) -> None:
        policy_input, actions = _fold_common_genbutsu_fixture()
        reversed_actions = tuple(reversed(actions))

        forward = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions
        )
        backward = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, reversed_actions
        )

        self.assertIs(forward.branch, backward.branch)
        self.assertEqual(
            frozenset(forward.baseline_eligible_actions),
            frozenset(backward.baseline_eligible_actions),
        )
        self.assertEqual(
            forward.baseline_selected_action, backward.baseline_selected_action
        )
        self.assertEqual(forward.candidate_evaluations, backward.candidate_evaluations)
        self.assertEqual(forward.full_legal_summary, backward.full_legal_summary)
        self.assertEqual(
            forward.baseline_eligible_summary, backward.baseline_eligible_summary
        )


class ProductionPathBoundaryTest(unittest.TestCase):
    """diagnostic seamがproduction decision pathから自動実行されないことを固定する。"""

    def test_production_policy_module_does_not_import_the_diagnostic_module(
        self,
    ) -> None:
        tree = ast.parse(inspect.getsource(mechanism))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertFalse(
            any(
                module.startswith(
                    "lisjong.policies.mechanism_riichi_defense_offensive_efficiency_diagnostic"
                )
                for module in imported
            )
        )

    def test_choose_action_does_not_trigger_the_diagnostic_completion_helper(
        self,
    ) -> None:
        """通常productionの`choose_action()`はdiagnostic側のR4 helperを呼ばない。"""
        policy_input, actions = _fold_common_genbutsu_fixture()
        decision = DecisionContext(input=policy_input, legal_actions=actions)
        with patch.object(
            diagnostic,
            "_evaluate_completion_masses",
            side_effect=AssertionError("diagnostic helper must not run in production"),
        ):
            MechanismRiichiDefenseYakuhaiCallPolicy().choose_action(decision)


class TypedResultValueTest(unittest.TestCase):
    """not-applicable(None)とevaluated zeroの区別、および基本的な不変条件を固定する。"""

    def test_not_applicable_and_evaluated_zero_are_distinct(self) -> None:
        policy_input, actions = _fold_common_genbutsu_fixture()
        analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions
        )
        manzu_three = next(
            candidate
            for candidate in analysis.candidate_evaluations
            if candidate.action.tile.tile_type == _type(TileCategory.MANZU, 3)
        )
        self.assertIsNone(manzu_three.full_legal_current_ukeire_count)
        self.assertIsNotNone(manzu_three.baseline_eligible_current_ukeire_count)
        self.assertNotEqual(manzu_three.baseline_eligible_current_ukeire_count, 0)

    def test_analysis_rejects_a_selected_action_outside_the_eligible_set(self) -> None:
        policy_input, actions = _push_fixture()
        analysis = analyze_mechanism_riichi_defense_offensive_efficiency(
            policy_input, actions
        )
        summary_kwargs = {
            "action_count": 1,
            "best_post_discard_shanten": 0,
            "selected_post_discard_shanten": 0,
            "shanten_regret": 0,
            "best_current_ukeire": 0,
            "selected_current_ukeire": None,
            "ukeire_regret": None,
            "best_second_step_score": None,
            "selected_second_step_score": None,
            "second_step_regret": None,
            "best_completion_mass": 0,
            "selected_completion_mass": 0,
            "completion_regret": 0,
            "completion_all_zero": True,
        }
        with self.assertRaises(ValueError):
            MechanismRiichiDefenseOffensiveEfficiencyAnalysis(
                legal_discard_actions=analysis.legal_discard_actions,
                baseline_eligible_actions=(analysis.baseline_eligible_actions[0],),
                baseline_selected_action=next(
                    action
                    for action in analysis.legal_discard_actions
                    if action != analysis.baseline_eligible_actions[0]
                ),
                branch=OffensiveEfficiencyBranch.PUSH,
                horizon=DEFAULT_HORIZON,
                hidden_tile_count=100,
                sequence_denominator=1,
                include_terminal_progression=False,
                candidate_evaluations=(
                    OffensiveEfficiencyCandidateEvaluation(
                        action=analysis.baseline_eligible_actions[0],
                        baseline_eligible=True,
                        post_discard_shanten=0,
                        full_legal_current_ukeire_count=0,
                        full_legal_second_step_ukeire_score=None,
                        baseline_eligible_current_ukeire_count=0,
                        baseline_eligible_second_step_ukeire_score=None,
                        completion_mass=0,
                        terminal_shanten_mass=None,
                        terminal_shanten_counts=None,
                    ),
                ),
                full_legal_summary=UniverseEfficiencySummary(**summary_kwargs),
                baseline_eligible_summary=UniverseEfficiencySummary(**summary_kwargs),
                terminal_progression_summary=None,
            )


if __name__ == "__main__":
    unittest.main()
