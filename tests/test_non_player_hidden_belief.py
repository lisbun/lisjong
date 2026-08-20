import unittest
from dataclasses import FrozenInstanceError

from lisjong.belief.canonical_axes import red_five_index, tile_type_index
from lisjong.belief.concealed_hand_belief import ConcealedHandBelief
from lisjong.belief.conditional_uniform_hand_belief import (
    estimate_conditional_uniform_hand_belief,
)
from lisjong.belief.fixed_point import (
    EXPECTED_COUNT_MAX_RAW,
    RED_FIVE_PROBABILITY_MAX_RAW,
    SCALE,
)
from lisjong.belief.hand_belief import HandBelief
from lisjong.belief.non_player_hidden_belief import (
    NonPlayerHiddenBelief,
    derive_non_player_hidden_belief,
)
from lisjong.belief.tile_conservation import (
    TileConservationResult,
    derive_remaining_tile_inventory,
)
from lisjong.belief.tile_inventory import (
    STANDARD_RED_FIVE_COUNTS,
    STANDARD_TILE_COUNTS,
)
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import TileCategory, TileType
from lisjong.policy_contract.wind import Wind

MANZU_1 = TileType(TileCategory.MANZU, 1)
MANZU_5 = TileType(TileCategory.MANZU, 5)
_CANONICAL_WINDS = (Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH)


def _empty_hand() -> HandBelief:
    return HandBelief(
        expected_count_raw=(0,) * 34,
        red_five_probability_raw=(0, 0, 0),
    )


def _hand(
    expected: dict[TileType, int] | None = None,
    red: dict[TileCategory, int] | None = None,
) -> HandBelief:
    expected_raw = [0] * 34
    red_raw = [0] * 3
    for tile_type, raw in (expected or {}).items():
        expected_raw[tile_type_index(tile_type)] = raw
    for category, raw in (red or {}).items():
        red_raw[red_five_index(category)] = raw
    return HandBelief(
        expected_count_raw=tuple(expected_raw),
        red_five_probability_raw=tuple(red_raw),
    )


def _concealed_hands(
    hands: dict[Wind, HandBelief] | None = None,
) -> ConcealedHandBelief:
    hands = hands or {}
    return ConcealedHandBelief(
        hands=tuple(hands.get(wind, _empty_hand()) for wind in _CANONICAL_WINDS)
    )


def _conservation(
    remaining: dict[TileType, int] | None = None,
    remaining_red: dict[TileCategory, int] | None = None,
) -> TileConservationResult:
    remaining_counts = list(STANDARD_TILE_COUNTS)
    remaining_red_counts = list(STANDARD_RED_FIVE_COUNTS)
    for tile_type, count in (remaining or {}).items():
        remaining_counts[tile_type_index(tile_type)] = count
    for category, count in (remaining_red or {}).items():
        remaining_red_counts[red_five_index(category)] = count
    return TileConservationResult(
        exact_accounted_counts=tuple(
            inventory - count
            for inventory, count in zip(
                STANDARD_TILE_COUNTS, remaining_counts, strict=True
            )
        ),
        exact_accounted_red_five_counts=tuple(
            inventory - count
            for inventory, count in zip(
                STANDARD_RED_FIVE_COUNTS, remaining_red_counts, strict=True
            )
        ),
        remaining_tile_counts=tuple(remaining_counts),
        remaining_red_five_counts=tuple(remaining_red_counts),
    )


def _policy_input() -> PolicyInput:
    player = PlayerPublicState(
        score=25000, discards=(), melds=(), riichi=RiichiState.NONE
    )
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
        players=(player, player, player, player),
        own_hand=OwnHandState(concealed_tiles=(), drawn_tile=None),
    )


class NonPlayerHiddenBeliefTest(unittest.TestCase):
    def test_semantic_accessors_and_immutability(self) -> None:
        expected_raw = [0] * 34
        expected_raw[tile_type_index(MANZU_5)] = SCALE
        belief = NonPlayerHiddenBelief(
            expected_count_raw=tuple(expected_raw),
            red_five_probability_raw=(SCALE // 2, 0, 0),
        )

        self.assertEqual(belief.expected_count(MANZU_5), 1.0)
        self.assertEqual(belief.red_five_probability(TileCategory.MANZU), 0.5)
        with self.assertRaises(FrozenInstanceError):
            belief.expected_count_raw = (0,) * 34

    def test_rejects_malformed_local_representation(self) -> None:
        valid_expected = (0,) * 34
        valid_red = (0, 0, 0)
        invalid_cases = (
            ((0,) * 33, valid_red, ValueError),
            (valid_expected, (0, 0), ValueError),
            ((-1,) + (0,) * 33, valid_red, ValueError),
            ((True,) + (0,) * 33, valid_red, TypeError),
            (
                (EXPECTED_COUNT_MAX_RAW + 1,) + (0,) * 33,
                valid_red,
                ValueError,
            ),
            (
                valid_expected,
                (RED_FIVE_PROBABILITY_MAX_RAW + 1, 0, 0),
                ValueError,
            ),
            (valid_expected, (1, 0, 0), ValueError),
        )
        for expected_raw, red_raw, exception in invalid_cases:
            with self.subTest(expected_raw=expected_raw, red_raw=red_raw):
                with self.assertRaises(exception):
                    NonPlayerHiddenBelief(
                        expected_count_raw=expected_raw,
                        red_five_probability_raw=red_raw,
                    )


class DerivationTest(unittest.TestCase):
    def test_derives_basic_tile_residual_and_ignores_self(self) -> None:
        hands = _concealed_hands(
            {
                Wind.EAST: _hand({MANZU_1: 4 * SCALE}),
                Wind.SOUTH: _hand({MANZU_1: SCALE}),
                Wind.WEST: _hand({MANZU_1: SCALE // 2}),
                Wind.NORTH: _hand({MANZU_1: SCALE // 4}),
            }
        )
        result = derive_non_player_hidden_belief(_conservation(), hands, Wind.EAST)

        self.assertEqual(
            result.expected_count_raw[tile_type_index(MANZU_1)],
            4 * SCALE - SCALE - SCALE // 2 - SCALE // 4,
        )

    def test_derives_red_and_normal_five_residuals(self) -> None:
        hands = _concealed_hands(
            {
                Wind.SOUTH: _hand(
                    {MANZU_5: SCALE + SCALE // 2},
                    {TileCategory.MANZU: SCALE // 2},
                ),
                Wind.WEST: _hand(
                    {MANZU_5: SCALE},
                    {TileCategory.MANZU: SCALE // 4},
                ),
            }
        )
        result = derive_non_player_hidden_belief(_conservation(), hands, Wind.EAST)
        five_raw = result.expected_count_raw[tile_type_index(MANZU_5)]
        red_raw = result.red_five_probability_raw[red_five_index(TileCategory.MANZU)]
        normal_raw = five_raw - red_raw

        self.assertEqual(five_raw, SCALE + SCALE // 2)
        self.assertEqual(red_raw, SCALE // 4)
        self.assertEqual(normal_raw, SCALE + SCALE // 4)
        self.assertEqual(five_raw, red_raw + normal_raw)

    def test_all_zero_residual_is_legal(self) -> None:
        full_inventory_hand = HandBelief(
            expected_count_raw=tuple(count * SCALE for count in STANDARD_TILE_COUNTS),
            red_five_probability_raw=tuple(
                count * SCALE for count in STANDARD_RED_FIVE_COUNTS
            ),
        )
        result = derive_non_player_hidden_belief(
            _conservation(),
            _concealed_hands({Wind.SOUTH: full_inventory_hand}),
            Wind.EAST,
        )

        self.assertEqual(result.expected_count_raw, (0,) * 34)
        self.assertEqual(result.red_five_probability_raw, (0, 0, 0))

    def test_self_wind_is_excluded_for_every_wind(self) -> None:
        same_hand = _hand({MANZU_1: SCALE})
        belief = ConcealedHandBelief(hands=(same_hand,) * 4)
        for self_wind in _CANONICAL_WINDS:
            with self.subTest(self_wind=self_wind):
                result = derive_non_player_hidden_belief(
                    _conservation(), belief, self_wind
                )
                self.assertEqual(
                    result.expected_count_raw[tile_type_index(MANZU_1)], SCALE
                )

    def test_same_input_yields_same_result(self) -> None:
        conservation = _conservation()
        belief = _concealed_hands({Wind.SOUTH: _hand({MANZU_1: SCALE})})
        first = derive_non_player_hidden_belief(conservation, belief, Wind.EAST)
        second = derive_non_player_hidden_belief(conservation, belief, Wind.EAST)
        self.assertEqual(first, second)

    def test_rejects_invalid_input_types(self) -> None:
        conservation = _conservation()
        belief = _concealed_hands()
        invalid_calls = (
            (None, belief, Wind.EAST),
            (conservation, None, Wind.EAST),
            (conservation, belief, None),
        )
        for args in invalid_calls:
            with self.subTest(args=args):
                with self.assertRaises(TypeError):
                    derive_non_player_hidden_belief(*args)


class FailClosedConservationTest(unittest.TestCase):
    def test_rejects_basic_tile_overage_by_one_raw_unit(self) -> None:
        belief = _concealed_hands(
            {
                Wind.SOUTH: _hand({MANZU_1: 2 * SCALE}),
                Wind.WEST: _hand({MANZU_1: 2 * SCALE}),
                Wind.NORTH: _hand({MANZU_1: 1}),
            }
        )
        with self.assertRaises(ValueError):
            derive_non_player_hidden_belief(_conservation(), belief, Wind.EAST)

    def test_rejects_positive_mass_when_remaining_is_zero(self) -> None:
        belief = _concealed_hands({Wind.SOUTH: _hand({MANZU_1: 1})})
        with self.assertRaises(ValueError):
            derive_non_player_hidden_belief(
                _conservation({MANZU_1: 0}), belief, Wind.EAST
            )

    def test_rejects_red_five_overage_by_one_raw_unit(self) -> None:
        belief = _concealed_hands(
            {
                Wind.SOUTH: _hand({MANZU_5: SCALE}, {TileCategory.MANZU: SCALE}),
                Wind.WEST: _hand({MANZU_5: 1}, {TileCategory.MANZU: 1}),
            }
        )
        with self.assertRaises(ValueError):
            derive_non_player_hidden_belief(_conservation(), belief, Wind.EAST)

    def test_rejects_only_normal_five_overage_by_one_raw_unit(self) -> None:
        belief = _concealed_hands(
            {
                Wind.SOUTH: _hand(
                    {MANZU_5: 4 * SCALE},
                    {TileCategory.MANZU: SCALE - 1},
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "normal-five"):
            derive_non_player_hidden_belief(_conservation(), belief, Wind.EAST)


class ConditionalUniformIntegrationTest(unittest.TestCase):
    def test_partial_allocation_derives_positive_non_player_mass(self) -> None:
        policy_input = _policy_input()
        conservation = derive_remaining_tile_inventory(policy_input)
        concealed = estimate_conditional_uniform_hand_belief(
            policy_input, (0, 13, 7, 5)
        )
        result = derive_non_player_hidden_belief(conservation, concealed, Wind.EAST)

        self.assertGreater(sum(result.expected_count_raw), 0)
        self.assertGreater(sum(result.red_five_probability_raw), 0)

    def test_full_allocation_derives_zero_non_player_mass(self) -> None:
        policy_input = _policy_input()
        conservation = derive_remaining_tile_inventory(policy_input)
        concealed = estimate_conditional_uniform_hand_belief(
            policy_input, (0, 46, 45, 45)
        )
        result = derive_non_player_hidden_belief(conservation, concealed, Wind.EAST)

        self.assertEqual(result.expected_count_raw, (0,) * 34)
        self.assertEqual(result.red_five_probability_raw, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
