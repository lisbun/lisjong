"""lisjong-project #20 disposable HandBelief sensitivity consumer tests。"""

import unittest
from unittest.mock import patch

import lisjong.policies.experimental_hand_belief_sensitivity as sensitivity
from lisjong.belief import ConcealedHandBelief, HandBelief
from lisjong.belief.fixed_point import SCALE
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind


def _tile(rank: int) -> Tile:
    return Tile(TileType(TileCategory.MANZU, rank))


def _player() -> PlayerPublicState:
    return PlayerPublicState(score=25000, discards=(), melds=(), riichi=RiichiState.NONE)


def _policy_input() -> PolicyInput:
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
        own_hand=OwnHandState(concealed_tiles=(_tile(1), _tile(2)), drawn_tile=None),
    )


def _hand_belief(counts: dict[int, int] | None = None) -> HandBelief:
    raw = [0] * 34
    for rank, count in (counts or {}).items():
        raw[rank - 1] = count * SCALE
    return HandBelief(
        expected_count_raw=tuple(raw),
        red_five_probability_raw=(0, 0, 0),
    )


def _belief(
    *,
    east: dict[int, int] | None = None,
    south: dict[int, int] | None = None,
    west: dict[int, int] | None = None,
    north: dict[int, int] | None = None,
) -> ConcealedHandBelief:
    return ConcealedHandBelief(
        hands=(
            _hand_belief(east),
            _hand_belief(south),
            _hand_belief(west),
            _hand_belief(north),
        )
    )


def _discard(rank: int) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=_tile(rank), tsumogiri=False)


def _effective_tiles(post_discard_hand, *_args):
    ranks = {tile.tile_type.rank for tile in post_discard_hand}
    if 2 in ranks:
        return (TileType(TileCategory.MANZU, 3),)
    if 1 in ranks:
        return (TileType(TileCategory.MANZU, 4),)
    raise AssertionError("unexpected post-discard hand")


class HandBeliefSensitivityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy_input = _policy_input()
        self.actions = (_discard(1), _discard(2))

    def test_belief_only_ranks_candidates_after_structural_tie(self) -> None:
        with (
            patch.object(
                sensitivity._DecisionShantenEvaluator,
                "calculate",
                return_value=1,
            ),
            patch.object(sensitivity, "_ukeire_count", return_value=10),
            patch.object(
                sensitivity,
                "_effective_tile_types",
                side_effect=_effective_tiles,
            ),
        ):
            mostly_three_man_in_opponents = sensitivity.evaluate_hand_belief_sensitive_discard(
                self.policy_input,
                self.actions,
                _belief(south={3: 3}),
            )
            mostly_four_man_in_opponents = sensitivity.evaluate_hand_belief_sensitive_discard(
                self.policy_input,
                self.actions,
                _belief(south={4: 3}),
            )

        self.assertTrue(mostly_three_man_in_opponents.consumer_active)
        self.assertTrue(mostly_four_man_in_opponents.consumer_active)
        self.assertEqual(mostly_three_man_in_opponents.action, _discard(2))
        self.assertEqual(mostly_four_man_in_opponents.action, _discard(1))

    def test_unique_public_ukeire_winner_does_not_activate_consumer(self) -> None:
        with (
            patch.object(
                sensitivity._DecisionShantenEvaluator,
                "calculate",
                return_value=1,
            ),
            patch.object(sensitivity, "_ukeire_count", side_effect=(10, 8)),
        ):
            decision = sensitivity.evaluate_hand_belief_sensitive_discard(
                self.policy_input,
                self.actions,
                _belief(),
            )

        self.assertFalse(decision.consumer_active)
        self.assertEqual(decision.action, _discard(1))
        self.assertTrue(
            all(
                evaluation.non_opponent_effective_tile_mass is None
                for evaluation in decision.candidate_evaluations
            )
        )

    def test_self_hand_belief_is_not_subtracted_as_opponent_mass(self) -> None:
        with (
            patch.object(
                sensitivity._DecisionShantenEvaluator,
                "calculate",
                return_value=1,
            ),
            patch.object(sensitivity, "_ukeire_count", return_value=10),
            patch.object(
                sensitivity,
                "_effective_tile_types",
                side_effect=_effective_tiles,
            ),
        ):
            decision = sensitivity.evaluate_hand_belief_sensitive_discard(
                self.policy_input,
                self.actions,
                _belief(east={3: 4}),
            )

        by_action = {
            evaluation.action: evaluation for evaluation in decision.candidate_evaluations
        }
        self.assertEqual(by_action[_discard(1)].non_opponent_effective_tile_mass, 4.0)

    def test_opponent_mass_cannot_exceed_publicly_unseen_copies(self) -> None:
        with (
            patch.object(
                sensitivity._DecisionShantenEvaluator,
                "calculate",
                return_value=1,
            ),
            patch.object(sensitivity, "_ukeire_count", return_value=10),
            patch.object(
                sensitivity,
                "_effective_tile_types",
                side_effect=_effective_tiles,
            ),
        ):
            with self.assertRaises(sensitivity.HandBeliefSensitivityError):
                sensitivity.evaluate_hand_belief_sensitive_discard(
                    self.policy_input,
                    self.actions,
                    _belief(south={3: 3}, west={3: 2}),
                )


if __name__ == "__main__":
    unittest.main()
