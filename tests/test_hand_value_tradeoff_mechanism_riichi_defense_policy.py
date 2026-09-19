"""Issue #175 HandValueTradeoffMechanismRiichiDefensePolicy tests."""

import itertools
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import lisjong.policies.hand_value_tradeoff_mechanism_riichi_defense as v2
from lisjong.belief.canonical_axes import tile_type_index
from lisjong.policies import (
    HandValueTradeoffMechanismRiichiDefensePolicy,
    MechanismRiichiDefenseYakuhaiCallPolicy,
)
from lisjong.policies.finite_horizon_completion import (
    FiniteHorizonCandidateEvaluation,
)
from lisjong.policy_contract.action import (
    DiscardAction,
    RiichiAction,
    RonAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
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
        tiles.extend(_tile(category, int(rank)) for rank in ranks)
        ranks = ""
    if ranks:
        raise ValueError(f"trailing ranks: {spec!r}")
    return tuple(tiles)


EAST = _tile(TileCategory.HONOR, 1)
WHITE = _tile(TileCategory.HONOR, 5)
MANZU_1 = _tile(TileCategory.MANZU, 1)
MANZU_2 = _tile(TileCategory.MANZU, 2)
MANZU_4 = _tile(TileCategory.MANZU, 4)
MANZU_5 = _tile(TileCategory.MANZU, 5)
SOUZU_9 = _tile(TileCategory.SOUZU, 9)


def _player() -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=(),
        melds=(),
        riichi=RiichiState.NONE,
    )


def _make_input(
    concealed_tiles: tuple[Tile, ...],
    *,
    self_seat: Seat = Seat.SEAT_0,
    dealer_seat: Seat = Seat.SEAT_0,
    round_wind: Wind = Wind.EAST,
) -> PolicyInput:
    return PolicyInput(
        self_seat=self_seat,
        round=RoundState(
            round_wind=round_wind,
            hand_number=1,
            dealer_seat=dealer_seat,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(),
            live_wall_tiles_remaining=70,
        ),
        players=tuple(_player() for _ in range(4)),
        own_hand=OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=None),
    )


def _decision(
    concealed_tiles: tuple[Tile, ...],
    actions: tuple[object, ...],
) -> DecisionContext:
    return DecisionContext(_make_input(concealed_tiles), actions)


def _discard(tile: Tile) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=False)


def _fallback_case(
    policy_input: PolicyInput,
    actions: tuple[DiscardAction, ...],
    *,
    post_hands: dict[DiscardAction, tuple[Tile, ...]],
    shanten: dict[DiscardAction, int],
    ukeire: dict[DiscardAction, int],
    retained: dict[DiscardAction, int] | None = None,
    potential: dict[DiscardAction, int] | None = None,
) -> DiscardAction:
    retained = retained or {action: 0 for action in actions}
    potential = potential or {action: 0 for action in actions}
    by_hand = {post_hands[action]: action for action in actions}
    structural = tuple(
        SimpleNamespace(
            action=action,
            post_discard_hand=post_hands[action],
            post_discard_shanten=shanten[action],
        )
        for action in actions
    )

    with (
        patch.object(v2, "evaluate_post_discard_hands", return_value=structural),
        patch.object(v2, "known_tile_counts", return_value={}),
        patch.object(v2, "_root_remaining_counts", return_value=(4,) * 34),
        patch.object(
            v2,
            "ukeire_count",
            side_effect=lambda hand, *_args: ukeire[by_hand[tuple(hand)]],
        ),
        patch.object(
            v2,
            "_retained_real_value",
            side_effect=lambda hand, _input: retained[by_hand[tuple(hand)]],
        ),
        patch.object(
            v2,
            "_supported_yaku_han_potential",
            side_effect=lambda hand, _input, _remaining: potential[
                by_hand[tuple(hand)]
            ],
        ),
        patch.object(v2, "_yaku_route_value", return_value=0),
        patch.object(v2, "second_step_ukeire_score", return_value=0),
    ):
        return v2._hand_value_v2_fallback(policy_input, actions)


class SupportedPotentialTest(unittest.TestCase):
    def setUp(self) -> None:
        self.input = _make_input((EAST, EAST, WHITE, WHITE, MANZU_2))

    def test_double_wind_pair_is_two_and_dragon_pair_is_one(self) -> None:
        remaining = [4] * 34
        remaining[tile_type_index(EAST.tile_type)] = 1
        remaining[tile_type_index(WHITE.tile_type)] = 1

        self.assertEqual(
            v2._yakuhai_pair_potential(
                (EAST, EAST, WHITE, WHITE),
                self.input,
                tuple(remaining),
            ),
            2,
        )
        self.assertEqual(
            v2._yakuhai_pair_potential(
                (WHITE, WHITE, MANZU_2),
                self.input,
                tuple(remaining),
            ),
            1,
        )

    def test_non_double_wind_is_one(self) -> None:
        policy_input = _make_input(
            (EAST, EAST, MANZU_2),
            self_seat=Seat.SEAT_1,
            dealer_seat=Seat.SEAT_0,
            round_wind=Wind.EAST,
        )
        remaining = [4] * 34
        remaining[tile_type_index(EAST.tile_type)] = 1
        self.assertEqual(
            v2._yakuhai_pair_potential(
                (EAST, EAST, MANZU_2),
                policy_input,
                tuple(remaining),
            ),
            1,
        )

    def test_dead_yakuhai_pair_has_zero_potential(self) -> None:
        remaining = [4] * 34
        remaining[tile_type_index(EAST.tile_type)] = 0
        self.assertEqual(
            v2._yakuhai_pair_potential(
                (EAST, EAST, MANZU_2),
                self.input,
                tuple(remaining),
            ),
            0,
        )

    def test_tanyao_potential_reuses_existing_compatibility_semantic(self) -> None:
        self.assertEqual(v2._tanyao_potential(_hand("234m456p678s"), self.input), 1)
        self.assertEqual(v2._tanyao_potential(_hand("123m456p678s"), self.input), 0)


class BoundedTradeoffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fast = _discard(MANZU_4)
        self.value = _discard(MANZU_5)
        self.input = _make_input((MANZU_4, MANZU_5, MANZU_2, SOUZU_9))
        self.post_hands = {
            self.fast: (MANZU_2, SOUZU_9),
            self.value: (MANZU_2, MANZU_5),
        }

    def test_ukeire_loss_four_can_trade_for_visible_value(self) -> None:
        selected = _fallback_case(
            self.input,
            (self.fast, self.value),
            post_hands=self.post_hands,
            shanten={self.fast: 1, self.value: 1},
            ukeire={self.fast: 10, self.value: 6},
            potential={self.fast: 0, self.value: 1},
        )
        self.assertIs(selected, self.value)

    def test_ukeire_loss_five_is_outside_the_tradeoff_band(self) -> None:
        selected = _fallback_case(
            self.input,
            (self.fast, self.value),
            post_hands=self.post_hands,
            shanten={self.fast: 1, self.value: 1},
            ukeire={self.fast: 10, self.value: 5},
            potential={self.fast: 0, self.value: 99},
        )
        self.assertIs(selected, self.fast)

    def test_zero_ukeire_candidate_is_not_rescued_when_max_is_positive(self) -> None:
        selected = _fallback_case(
            self.input,
            (self.fast, self.value),
            post_hands=self.post_hands,
            shanten={self.fast: 1, self.value: 1},
            ukeire={self.fast: 4, self.value: 0},
            potential={self.fast: 0, self.value: 99},
        )
        self.assertIs(selected, self.fast)

    def test_worse_shanten_cannot_be_rescued_by_value(self) -> None:
        selected = _fallback_case(
            self.input,
            (self.fast, self.value),
            post_hands=self.post_hands,
            shanten={self.fast: 1, self.value: 2},
            ukeire={self.fast: 1, self.value: 100},
            potential={self.fast: 0, self.value: 99},
        )
        self.assertIs(selected, self.fast)

    def test_equal_visible_value_restores_current_ukeire_priority(self) -> None:
        selected = _fallback_case(
            self.input,
            (self.fast, self.value),
            post_hands=self.post_hands,
            shanten={self.fast: 1, self.value: 1},
            ukeire={self.fast: 10, self.value: 8},
            potential={self.fast: 1, self.value: 1},
        )
        self.assertIs(selected, self.fast)

    def test_selection_is_independent_of_legal_action_order(self) -> None:
        selections = set()
        for actions in itertools.permutations((self.fast, self.value)):
            selections.add(
                _fallback_case(
                    self.input,
                    actions,
                    post_hands=self.post_hands,
                    shanten={self.fast: 1, self.value: 1},
                    ukeire={self.fast: 10, self.value: 6},
                    potential={self.fast: 0, self.value: 1},
                )
            )
        self.assertEqual(selections, {self.value})


class MotivatingPatternTest(unittest.TestCase):
    def test_double_wind_pair_beats_dragon_pair_inside_speed_band(self) -> None:
        keep_east = _discard(WHITE)
        keep_white = _discard(EAST)
        policy_input = _make_input((EAST, EAST, WHITE, WHITE, MANZU_2))
        post_hands = {
            keep_east: (EAST, EAST, MANZU_2),
            keep_white: (WHITE, WHITE, MANZU_2),
        }
        structural = tuple(
            SimpleNamespace(
                action=action,
                post_discard_hand=post_hands[action],
                post_discard_shanten=1,
            )
            for action in (keep_white, keep_east)
        )
        remaining = [4] * 34
        remaining[tile_type_index(EAST.tile_type)] = 1
        remaining[tile_type_index(WHITE.tile_type)] = 1

        with (
            patch.object(v2, "evaluate_post_discard_hands", return_value=structural),
            patch.object(v2, "known_tile_counts", return_value={}),
            patch.object(v2, "_root_remaining_counts", return_value=tuple(remaining)),
            patch.object(v2, "ukeire_count", return_value=10),
            patch.object(v2, "_retained_real_value", return_value=0),
            patch.object(v2, "_yaku_route_value", return_value=0),
            patch.object(v2, "second_step_ukeire_score", return_value=0),
        ):
            selected = v2._hand_value_v2_fallback(
                policy_input, (keep_white, keep_east)
            )

        self.assertIs(selected, keep_east)
        self.assertEqual(
            v2._yakuhai_pair_potential(
                post_hands[keep_east], policy_input, tuple(remaining)
            ),
            2,
        )
        self.assertEqual(
            v2._yakuhai_pair_potential(
                post_hands[keep_white], policy_input, tuple(remaining)
            ),
            1,
        )

    def test_tanyao_route_can_beat_small_speed_loss(self) -> None:
        tanyao = _discard(MANZU_1)
        non_tanyao = _discard(MANZU_4)
        policy_input = _make_input(_hand("1234m456p678s22p"))
        post_hands = {
            tanyao: _hand("234m456p678s22p"),
            non_tanyao: _hand("123m456p678s22p"),
        }
        by_hand = {hand: action for action, hand in post_hands.items()}
        structural = tuple(
            SimpleNamespace(
                action=action,
                post_discard_hand=post_hands[action],
                post_discard_shanten=1,
            )
            for action in (non_tanyao, tanyao)
        )

        with (
            patch.object(v2, "evaluate_post_discard_hands", return_value=structural),
            patch.object(v2, "known_tile_counts", return_value={}),
            patch.object(v2, "_root_remaining_counts", return_value=(4,) * 34),
            patch.object(
                v2,
                "ukeire_count",
                side_effect=lambda hand, *_args: 8
                if by_hand[tuple(hand)] is tanyao
                else 10,
            ),
            patch.object(v2, "_retained_real_value", return_value=0),
            patch.object(v2, "_yaku_route_value", return_value=0),
            patch.object(v2, "second_step_ukeire_score", return_value=0),
        ):
            selected = v2._hand_value_v2_fallback(
                policy_input, (non_tanyao, tanyao)
            )

        self.assertIs(selected, tanyao)
        self.assertEqual(v2._tanyao_potential(post_hands[tanyao], policy_input), 1)
        self.assertEqual(
            v2._tanyao_potential(post_hands[non_tanyao], policy_input), 0
        )


class FiniteHorizonBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.a = _discard(MANZU_4)
        self.b = _discard(MANZU_5)
        self.c = _discard(SOUZU_9)
        self.input = _make_input((MANZU_4, MANZU_5, SOUZU_9, MANZU_2))

    def _mass_patch(self, masses: dict[DiscardAction, int]):
        return patch.object(
            v2,
            "_evaluate_completion_masses",
            side_effect=lambda _input, actions, *_args: tuple(
                FiniteHorizonCandidateEvaluation(action, masses[action])
                for action in actions
            ),
        )

    def test_unique_positive_finite_horizon_winner_skips_v2(self) -> None:
        with (
            patch.object(v2, "_defense_eligible_actions", side_effect=lambda _i, a: a),
            patch.object(v2, "_root_remaining_counts", return_value=(4,) * 34),
            self._mass_patch({self.a: 9, self.b: 3}),
            patch.object(
                v2,
                "_hand_value_v2_from_finite_horizon",
                side_effect=AssertionError("unique positive winner must be exact parent"),
            ),
        ):
            selected = v2._evaluate_finite_horizon_hand_value_v2(
                self.input, (self.a, self.b)
            )
        self.assertIs(selected, self.a)

    def test_positive_tie_passes_only_maximum_subset_to_v2(self) -> None:
        with (
            patch.object(v2, "_defense_eligible_actions", side_effect=lambda _i, a: a),
            patch.object(v2, "_root_remaining_counts", return_value=(4,) * 34),
            self._mass_patch({self.a: 9, self.b: 9, self.c: 8}),
            patch.object(
                v2,
                "_hand_value_v2_from_finite_horizon",
                return_value=self.b,
            ) as fallback,
        ):
            selected = v2._evaluate_finite_horizon_hand_value_v2(
                self.input, (self.a, self.b, self.c)
            )

        self.assertIs(selected, self.b)
        evaluations = fallback.call_args.args[1]
        self.assertEqual(
            tuple(evaluation.action for evaluation in evaluations),
            (self.a, self.b),
        )

    def test_all_zero_passes_every_candidate_to_v2(self) -> None:
        with (
            patch.object(v2, "_defense_eligible_actions", side_effect=lambda _i, a: a),
            patch.object(v2, "_root_remaining_counts", return_value=(4,) * 34),
            self._mass_patch({self.a: 0, self.b: 0}),
            patch.object(
                v2,
                "_hand_value_v2_from_finite_horizon",
                return_value=self.a,
            ) as fallback,
        ):
            v2._evaluate_finite_horizon_hand_value_v2(
                self.input, (self.a, self.b)
            )

        self.assertEqual(
            tuple(evaluation.action for evaluation in fallback.call_args.args[1]),
            (self.a, self.b),
        )


class ParentBoundaryTest(unittest.TestCase):
    def test_mechanism_filter_runs_before_the_new_offensive_fallback(self) -> None:
        unsafe = _discard(MANZU_4)
        eligible = _discard(MANZU_5)
        decision = _decision(
            (MANZU_4, MANZU_5, MANZU_2),
            (unsafe, eligible),
        )
        policy = HandValueTradeoffMechanismRiichiDefensePolicy()

        with (
            patch.object(
                v2,
                "_mechanism_defense_eligible_actions",
                return_value=(eligible,),
            ),
            patch.object(
                v2,
                "_evaluate_finite_horizon_hand_value_v2",
                return_value=eligible,
            ) as evaluate,
        ):
            selected = policy.choose_action(decision)

        self.assertIs(selected, eligible)
        self.assertEqual(evaluate.call_args.args[1], (eligible,))

    def test_winning_and_riichi_orchestration_remain_inherited(self) -> None:
        policy = HandValueTradeoffMechanismRiichiDefensePolicy()
        discard = _discard(MANZU_4)
        ron = RonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            winning_tile=MANZU_5,
        )
        riichi = RiichiAction(actor=Seat.SEAT_0)

        with patch.object(
            v2,
            "_evaluate_finite_horizon_hand_value_v2",
            side_effect=AssertionError("non-discard branch must skip v2"),
        ):
            self.assertIs(policy.choose_action(_decision((MANZU_4,), (discard, ron))), ron)
            self.assertIs(
                policy.choose_action(_decision((MANZU_4,), (discard, riichi))),
                riichi,
            )

    def test_policy_is_stateless_public_and_spawn_picklable(self) -> None:
        policy = HandValueTradeoffMechanismRiichiDefensePolicy()
        self.assertIsInstance(policy, MechanismRiichiDefenseYakuhaiCallPolicy)
        self.assertEqual(vars(policy), {})
        self.assertIs(
            pickle.loads(pickle.dumps(HandValueTradeoffMechanismRiichiDefensePolicy)),
            HandValueTradeoffMechanismRiichiDefensePolicy,
        )
        self.assertEqual(
            HandValueTradeoffMechanismRiichiDefensePolicy.__module__,
            "lisjong.policies.hand_value_tradeoff_mechanism_riichi_defense",
        )


if __name__ == "__main__":
    unittest.main()
