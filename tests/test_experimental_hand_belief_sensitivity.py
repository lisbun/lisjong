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
    return PlayerPublicState(
        score=25000,
        discards=(),
        melds=(),
        riichi=RiichiState.NONE,
    )


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


def _shanten_by_remaining_tile(post_discard_hand) -> int:
    ranks = {tile.tile_type.rank for tile in post_discard_hand}
    if 2 in ranks:
        return 0
    if 1 in ranks:
        return 1
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
            mostly_three_man_in_opponents = (
                sensitivity.evaluate_hand_belief_sensitive_discard(
                    self.policy_input,
                    self.actions,
                    _belief(south={3: 3}),
                )
            )
            mostly_four_man_in_opponents = (
                sensitivity.evaluate_hand_belief_sensitive_discard(
                    self.policy_input,
                    self.actions,
                    _belief(south={4: 3}),
                )
            )

        self.assertTrue(mostly_three_man_in_opponents.consumer_active)
        self.assertTrue(mostly_four_man_in_opponents.consumer_active)
        self.assertEqual(mostly_three_man_in_opponents.action, _discard(2))
        self.assertEqual(mostly_four_man_in_opponents.action, _discard(1))

    def test_worse_shanten_candidate_cannot_be_resurrected_by_belief(self) -> None:
        with (
            patch.object(
                sensitivity._DecisionShantenEvaluator,
                "calculate",
                side_effect=_shanten_by_remaining_tile,
            ),
            patch.object(sensitivity, "_ukeire_count", return_value=10),
        ):
            decision = sensitivity.evaluate_hand_belief_sensitive_discard(
                self.policy_input,
                self.actions,
                _belief(south={3: 4}),
            )

        self.assertFalse(decision.consumer_active)
        self.assertEqual(decision.action, _discard(1))

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
            evaluation.action: evaluation
            for evaluation in decision.candidate_evaluations
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


class OpponentExpectedCountsTest(unittest.TestCase):
    def test_belief_reduction_sums_only_opponent_rows(self) -> None:
        counts = sensitivity.opponent_expected_counts_from_belief(
            _policy_input(),
            _belief(east={3: 4}, south={3: 3}, west={3: 1}, north={4: 2}),
        )

        self.assertEqual(counts.total(TileType(TileCategory.MANZU, 3)), 4.0)
        self.assertEqual(counts.total(TileType(TileCategory.MANZU, 4)), 2.0)

    def test_counts_must_have_canonical_length(self) -> None:
        with self.assertRaises(ValueError):
            sensitivity.OpponentExpectedCounts(counts=(0.0,) * 33)

    def test_counts_reject_values_outside_structural_range(self) -> None:
        with self.assertRaises(ValueError):
            sensitivity.OpponentExpectedCounts(counts=(12.5,) + (0.0,) * 33)
        with self.assertRaises(ValueError):
            sensitivity.OpponentExpectedCounts(counts=(-0.5,) + (0.0,) * 33)

    def test_counts_reject_non_numeric_values(self) -> None:
        with self.assertRaises(TypeError):
            sensitivity.OpponentExpectedCounts(counts=(True,) + (0.0,) * 33)

    def test_structural_upper_bound_is_not_a_conservation_check(self) -> None:
        """positionのunseen枚数を超える値はconstruction時には拒否しない。"""
        counts = sensitivity.OpponentExpectedCounts(counts=(12.0,) + (0.0,) * 33)

        self.assertEqual(counts.total(TileType(TileCategory.MANZU, 1)), 12.0)


class ExpectedCountSeamEquivalenceTest(unittest.TestCase):
    """expected-count-only seamと既存ConcealedHandBelief pathのequivalence。"""

    def setUp(self) -> None:
        self.policy_input = _policy_input()
        self.actions = (_discard(1), _discard(2))

    def _both_paths(self, belief):
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
            from_belief = sensitivity.evaluate_hand_belief_sensitive_discard(
                self.policy_input,
                self.actions,
                belief,
            )
            from_counts = sensitivity.evaluate_expected_count_sensitive_discard(
                self.policy_input,
                self.actions,
                sensitivity.opponent_expected_counts_from_belief(
                    self.policy_input,
                    belief,
                ),
            )
        return from_belief, from_counts

    def test_expected_count_seam_matches_belief_path(self) -> None:
        for belief in (
            _belief(south={3: 3}),
            _belief(south={4: 3}),
            _belief(east={3: 4}),
            _belief(south={3: 2}, west={4: 1}),
            _belief(),
        ):
            with self.subTest(belief=belief):
                from_belief, from_counts = self._both_paths(belief)
                self.assertEqual(from_belief, from_counts)

    def test_expected_count_seam_preserves_conservation_failure(self) -> None:
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
                sensitivity.evaluate_expected_count_sensitive_discard(
                    self.policy_input,
                    self.actions,
                    sensitivity.OpponentExpectedCounts(
                        counts=tuple(5.0 if index == 2 else 0.0 for index in range(34))
                    ),
                )

    def test_expected_count_seam_requires_expected_count_table(self) -> None:
        with self.assertRaises(TypeError):
            sensitivity.evaluate_expected_count_sensitive_discard(
                self.policy_input,
                self.actions,
                _belief(),
            )


if __name__ == "__main__":
    unittest.main()
