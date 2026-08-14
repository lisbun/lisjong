import inspect
import itertools
import unittest

from lisjong.policies import MinimalPolicy
from lisjong.policy_contract.action import (
    ChiAction,
    DiscardAction,
    PassAction,
    RonAction,
    TsumoAction,
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

MANZU_3 = Tile(TileType(TileCategory.MANZU, 3))
MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))
MANZU_5_RED = Tile(TileType(TileCategory.MANZU, 5), is_red=True)
MANZU_6 = Tile(TileType(TileCategory.MANZU, 6))
PINZU_5 = Tile(TileType(TileCategory.PINZU, 5))


def _make_player(score: int = 25000) -> PlayerPublicState:
    return PlayerPublicState(
        score=score,
        discards=(),
        melds=(),
        riichi=RiichiState.NONE,
    )


def _make_input(self_seat: Seat = Seat.SEAT_0) -> PolicyInput:
    return PolicyInput(
        self_seat=self_seat,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(MANZU_3,),
            live_wall_tiles_remaining=70,
        ),
        players=(_make_player(), _make_player(), _make_player(), _make_player()),
        own_hand=OwnHandState(
            concealed_tiles=(MANZU_4, MANZU_5, MANZU_5_RED, MANZU_6, PINZU_5),
            drawn_tile=MANZU_5,
        ),
    )


def _decision(*actions: object) -> DecisionContext:
    return DecisionContext(input=_make_input(), legal_actions=actions)


class MinimalPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = MinimalPolicy()

    def test_chooses_tsumo_over_non_winning_action(self) -> None:
        discard = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_3, tsumogiri=False)
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_5)

        chosen = self.policy.choose_action(_decision(discard, tsumo))

        self.assertEqual(chosen, tsumo)

    def test_chooses_ron_over_non_winning_action(self) -> None:
        pass_action = PassAction(actor=Seat.SEAT_0)
        ron = RonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            winning_tile=MANZU_5,
        )

        chosen = self.policy.choose_action(_decision(pass_action, ron))

        self.assertEqual(chosen, ron)

    def test_multiple_winning_actions_are_order_independent(self) -> None:
        ron = RonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            winning_tile=MANZU_5,
        )
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_5)

        results = {
            self.policy.choose_action(_decision(*permutation))
            for permutation in itertools.permutations((ron, tsumo))
        }

        self.assertEqual(results, {ron})

    def test_non_winning_choice_is_independent_of_legal_action_order(self) -> None:
        actions = (
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_6, tsumogiri=False),
            PassAction(actor=Seat.SEAT_0),
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
        )

        results = {
            self.policy.choose_action(_decision(*permutation))
            for permutation in itertools.permutations(actions)
        }

        self.assertEqual(
            results,
            {DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)},
        )

    def test_repeated_and_interleaved_calls_do_not_change_choice(self) -> None:
        first_decision = _decision(
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_6, tsumogiri=False),
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
        )
        unrelated_decision = _decision(PassAction(actor=Seat.SEAT_0))

        first_choice = self.policy.choose_action(first_decision)
        self.policy.choose_action(unrelated_decision)
        second_choice = self.policy.choose_action(first_decision)

        self.assertEqual(second_choice, first_choice)

    def test_distinguishes_normal_and_red_tile_identity(self) -> None:
        normal = DiscardAction(
            actor=Seat.SEAT_0,
            tile=MANZU_5,
            tsumogiri=False,
        )
        red = DiscardAction(
            actor=Seat.SEAT_0,
            tile=MANZU_5_RED,
            tsumogiri=False,
        )

        results = {
            self.policy.choose_action(_decision(*permutation))
            for permutation in itertools.permutations((normal, red))
        }

        self.assertEqual(results, {normal})

    def test_distinguishes_tsumogiri_identity(self) -> None:
        tedashi = DiscardAction(
            actor=Seat.SEAT_0,
            tile=MANZU_5,
            tsumogiri=False,
        )
        tsumogiri = DiscardAction(
            actor=Seat.SEAT_0,
            tile=MANZU_5,
            tsumogiri=True,
        )

        results = {
            self.policy.choose_action(_decision(*permutation))
            for permutation in itertools.permutations((tedashi, tsumogiri))
        }

        self.assertEqual(results, {tedashi})

    def test_distinguishes_meld_consumed_composition(self) -> None:
        lower = ChiAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_3,
            called_tile=MANZU_5,
            consumed_tiles=(MANZU_3, MANZU_4),
        )
        upper = ChiAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_3,
            called_tile=MANZU_5,
            consumed_tiles=(MANZU_4, MANZU_6),
        )

        results = {
            self.policy.choose_action(_decision(*permutation))
            for permutation in itertools.permutations((lower, upper))
        }

        self.assertEqual(results, {lower})

    def test_explicit_pass_can_be_selected(self) -> None:
        pass_action = PassAction(actor=Seat.SEAT_0)

        chosen = self.policy.choose_action(_decision(pass_action))

        self.assertIs(chosen, pass_action)

    def test_returns_an_original_legal_action_without_reconstruction(self) -> None:
        actions = (
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_6, tsumogiri=False),
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
        )
        decision = _decision(*actions)

        chosen = self.policy.choose_action(decision)

        self.assertTrue(any(chosen is action for action in decision.legal_actions))

    def test_contract_violation_is_rejected_before_policy_evaluation(self) -> None:
        with self.assertRaises(ValueError):
            _decision()

        duplicate = DiscardAction(
            actor=Seat.SEAT_0,
            tile=MANZU_4,
            tsumogiri=False,
        )
        with self.assertRaises(ValueError):
            _decision(duplicate, duplicate)

        with self.assertRaises(ValueError):
            DecisionContext(
                input=_make_input(self_seat=Seat.SEAT_1),
                legal_actions=(duplicate,),
            )

    def test_public_api_requires_only_decision_context(self) -> None:
        self.assertEqual(tuple(inspect.signature(MinimalPolicy).parameters), ())
        self.assertEqual(
            tuple(inspect.signature(MinimalPolicy.choose_action).parameters),
            ("self", "decision"),
        )


if __name__ == "__main__":
    unittest.main()
