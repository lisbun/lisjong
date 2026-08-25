"""Issue #122 combined experimental Policyのunit test。"""

import ast
import inspect
import itertools
import pickle
import unittest
from unittest.mock import patch

import lisjong.policies.finite_horizon_completion as finite_horizon
import lisjong.policies.genbutsu_defense_finite_horizon_value_aware as combined
from lisjong.policies import (
    FiniteHorizonCompletionPolicy,
    GenbutsuDefenseFiniteHorizonValueAwarePolicy,
    GenbutsuDefenseTwoStepUkeirePolicy,
    TwoStepUkeirePolicy,
    ValueAwareTwoStepUkeirePolicy,
)
from lisjong.policies.finite_horizon_completion import (
    FiniteHorizonCandidateEvaluation,
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


MANZU_3 = _tile(TileCategory.MANZU, 3)
MANZU_4 = _tile(TileCategory.MANZU, 4)
MANZU_5 = _tile(TileCategory.MANZU, 5)
PINZU_8 = _tile(TileCategory.PINZU, 8)
SOUZU_5_RED = _tile(TileCategory.SOUZU, 5, red=True)
SOUZU_9 = _tile(TileCategory.SOUZU, 9)
WHITE_DRAGON = _tile(TileCategory.HONOR, 5)
RED_DRAGON = _tile(TileCategory.HONOR, 7)

_TWO_STEP_HAND = _hand("345m56679s333577z")
_RED_FIVE_HAND = _hand("345m678m123p33z0s8p1s")
_TANKI_VERSUS_ONE_SHANTEN_HAND = _hand("234m567m234p567p5s7z")


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

    return patch.object(combined, "_evaluate_completion_masses", fake_evaluate)


class CompositionPriorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GenbutsuDefenseFiniteHorizonValueAwarePolicy()

    def test_genbutsu_constrains_the_finite_horizon_eligible_set(self) -> None:
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

    def test_tenpai_disables_the_genbutsu_constraint(self) -> None:
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

    def test_value_aware_dora_retention_decides_all_zero_and_positive_tie(self) -> None:
        keep_dora = _discard(SOUZU_9)
        lose_dora = _discard(WHITE_DRAGON)
        decision = _decision(
            _TWO_STEP_HAND,
            (lose_dora, keep_dora),
            dora_indicators=(RED_DRAGON,),
        )

        for maximum_mass in (0, 7):
            with self.subTest(maximum_mass=maximum_mass):
                with _mass_patch({keep_dora: maximum_mass, lose_dora: maximum_mass}):
                    selected = self.policy.choose_action(decision)

                self.assertIs(selected, keep_dora)

    def test_value_aware_aka_dora_retention_decides_an_all_zero_fallback(self) -> None:
        keep_red_five = _discard(PINZU_8)
        lose_red_five = _discard(SOUZU_5_RED)
        decision = _decision(
            _RED_FIVE_HAND,
            (lose_red_five, keep_red_five),
        )

        with _mass_patch({keep_red_five: 0, lose_red_five: 0}):
            selected = self.policy.choose_action(decision)

        self.assertIs(selected, keep_red_five)

    def test_unique_positive_mass_skips_value_aware_even_when_it_loses_dora(
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
                combined,
                "_value_aware_evaluate_and_choose_discard",
                side_effect=AssertionError(
                    "unique positive maximum must not execute ValueAware"
                ),
            ),
        ):
            selected = self.policy.choose_action(decision)

        self.assertIs(selected, lose_dora)

    def test_completion_mass_loser_is_not_revived_in_a_positive_tie(self) -> None:
        value_aware_favorite = _discard(SOUZU_9)
        maximum_white = _discard(WHITE_DRAGON)
        maximum_5m = _discard(MANZU_5)
        decision = _decision(
            _TWO_STEP_HAND,
            (value_aware_favorite, maximum_white, maximum_5m),
            dora_indicators=(RED_DRAGON,),
        )

        with (
            _mass_patch(
                {
                    value_aware_favorite: 1,
                    maximum_white: 8,
                    maximum_5m: 8,
                }
            ),
            patch.object(
                combined,
                "_value_aware_evaluate_and_choose_discard",
                wraps=combined._value_aware_evaluate_and_choose_discard,
            ) as value_aware,
        ):
            selected = self.policy.choose_action(decision)

        fallback_actions = value_aware.call_args.args[1]
        self.assertEqual(set(fallback_actions), {maximum_white, maximum_5m})
        self.assertNotIn(value_aware_favorite, fallback_actions)
        self.assertIsNot(selected, value_aware_favorite)

    def test_interaction_is_genbutsu_then_mass_tie_then_dora_retention(self) -> None:
        unsafe = _discard(MANZU_5)
        safe_keep_dora = _discard(SOUZU_9)
        safe_lose_dora = _discard(WHITE_DRAGON)
        observed: list[tuple[DiscardAction, ...]] = []
        decision = _decision(
            _TWO_STEP_HAND,
            (unsafe, safe_lose_dora, safe_keep_dora),
            players=_players_with_threat(SOUZU_9, WHITE_DRAGON),
            dora_indicators=(RED_DRAGON,),
        )

        with (
            _mass_patch(
                {
                    unsafe: 999,
                    safe_keep_dora: 5,
                    safe_lose_dora: 5,
                },
                observed,
            ),
            patch.object(
                combined,
                "_value_aware_evaluate_and_choose_discard",
                wraps=combined._value_aware_evaluate_and_choose_discard,
            ) as value_aware,
        ):
            selected = self.policy.choose_action(decision)

        self.assertEqual(observed, [(safe_lose_dora, safe_keep_dora)])
        self.assertEqual(
            set(value_aware.call_args.args[1]),
            {safe_keep_dora, safe_lose_dora},
        )
        self.assertIs(selected, safe_keep_dora)

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


class ExistingPolicyBaselinePreservationTest(unittest.TestCase):
    def test_existing_two_step_genbutsu_and_value_aware_choices_are_unchanged(
        self,
    ) -> None:
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
                _decision(
                    _TWO_STEP_HAND,
                    actions,
                    dora_indicators=(RED_DRAGON,),
                )
            ),
            discard_9s,
        )

    def test_existing_finite_horizon_unique_maximum_semantic_is_unchanged(self) -> None:
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


class InheritedOrchestrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GenbutsuDefenseFiniteHorizonValueAwarePolicy()

    def test_only_decide_discard_is_overridden(self) -> None:
        self.assertTrue(
            issubclass(
                GenbutsuDefenseFiniteHorizonValueAwarePolicy,
                TwoStepUkeirePolicy,
            )
        )
        self.assertIn(
            "_decide_discard",
            vars(GenbutsuDefenseFiniteHorizonValueAwarePolicy),
        )
        for method_name in (
            "_decide",
            "choose_action",
            "choose_action_with_analysis",
        ):
            with self.subTest(method_name=method_name):
                self.assertNotIn(
                    method_name,
                    vars(GenbutsuDefenseFiniteHorizonValueAwarePolicy),
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
            combined,
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
                combined,
                "_evaluate_and_choose_discard",
                wraps=combined._evaluate_and_choose_discard,
            ) as evaluate,
        ):
            selected = execute_policy_with_trace(self.policy, decision, recorder)

        (trace,) = recorder.snapshot()
        self.assertIs(selected, discard_white)
        self.assertIs(trace.selected_action, selected)
        self.assertIsNone(trace.analysis)
        self.assertEqual(evaluate.call_count, 1)


class PackageExportAndSpawnCompatibilityTest(unittest.TestCase):
    def test_combined_policy_is_public_and_module_level(self) -> None:
        from lisjong.policies import (
            GenbutsuDefenseFiniteHorizonValueAwarePolicy as imported,
        )

        self.assertIs(imported, combined.GenbutsuDefenseFiniteHorizonValueAwarePolicy)
        self.assertEqual(
            GenbutsuDefenseFiniteHorizonValueAwarePolicy.__module__,
            "lisjong.policies.genbutsu_defense_finite_horizon_value_aware",
        )

    def test_combined_policy_class_is_picklable_for_windows_spawn(self) -> None:
        roundtrip = pickle.loads(
            pickle.dumps(GenbutsuDefenseFiniteHorizonValueAwarePolicy)
        )

        self.assertIs(roundtrip, GenbutsuDefenseFiniteHorizonValueAwarePolicy)

    def test_module_uses_only_policy_visible_dependencies(self) -> None:
        tree = ast.parse(inspect.getsource(combined))
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


if __name__ == "__main__":
    unittest.main()
