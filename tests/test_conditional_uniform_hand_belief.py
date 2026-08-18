import unittest

from lisjong.belief.canonical_axes import tile_type_index, wind_for_seat
from lisjong.belief.conditional_uniform_hand_belief import (
    _baseline_hand_belief,
    estimate_conditional_uniform_hand_belief,
)
from lisjong.belief.fixed_point import SCALE, round_half_to_even_ratio
from lisjong.belief.self_belief import exact_self_belief
from lisjong.belief.tile_conservation import (
    TileConservationResult,
    derive_remaining_tile_inventory,
)
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import (
    Tile,
    TileCategory,
    TileType,
)
from lisjong.policy_contract.wind import Wind

MANZU_5 = TileType(TileCategory.MANZU, 5)


def _tile(tile_type: TileType, *, is_red: bool = False) -> Tile:
    return Tile(tile_type, is_red=is_red)


def _player(discards: tuple[Discard, ...] = ()) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000, discards=discards, melds=(), riichi=RiichiState.NONE
    )


def _empty_players() -> tuple[
    PlayerPublicState, PlayerPublicState, PlayerPublicState, PlayerPublicState
]:
    return (_player(), _player(), _player(), _player())


def _round(dealer_seat: Seat = Seat.SEAT_0) -> RoundState:
    return RoundState(
        round_wind=Wind.EAST,
        hand_number=1,
        dealer_seat=dealer_seat,
        honba=0,
        riichi_sticks=0,
        dora_indicators=(),
        live_wall_tiles_remaining=70,
    )


def _policy_input(
    self_seat: Seat = Seat.SEAT_0,
    dealer_seat: Seat = Seat.SEAT_0,
    concealed_tiles: tuple[Tile, ...] = (),
    players: tuple[
        PlayerPublicState, PlayerPublicState, PlayerPublicState, PlayerPublicState
    ]
    | None = None,
) -> PolicyInput:
    return PolicyInput(
        self_seat=self_seat,
        round=_round(dealer_seat=dealer_seat),
        players=players if players is not None else _empty_players(),
        own_hand=OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=None),
    )


def _full_standard_tile_list() -> tuple[Tile, ...]:
    """standard physical inventoryのすべて（136枚）を1人のconcealed_tilesへ
    詰め込み、remaining tile inventoryを全牌種0にするためのfixture。
    `OwnHandState`は枚数制約を持たないため合法である。
    """
    tiles: list[Tile] = []
    for category in (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU):
        for rank in range(1, 10):
            tile_type = TileType(category, rank)
            if rank == 5:
                tiles.append(_tile(tile_type, is_red=True))
                tiles.extend(_tile(tile_type) for _ in range(3))
            else:
                tiles.extend(_tile(tile_type) for _ in range(4))
    for rank in range(1, 8):
        tile_type = TileType(TileCategory.HONOR, rank)
        tiles.extend(_tile(tile_type) for _ in range(4))
    return tuple(tiles)


class SelfSlotValidationTest(unittest.TestCase):
    def test_rejects_non_zero_self_slot(self) -> None:
        policy_input = _policy_input(self_seat=Seat.SEAT_0, dealer_seat=Seat.SEAT_0)
        with self.assertRaises(ValueError):
            estimate_conditional_uniform_hand_belief(policy_input, (1, 0, 0, 0))

    def test_rejects_negative_slot(self) -> None:
        policy_input = _policy_input()
        with self.assertRaises(ValueError):
            estimate_conditional_uniform_hand_belief(policy_input, (0, -1, 0, 0))

    def test_rejects_non_int_slot(self) -> None:
        policy_input = _policy_input()
        with self.assertRaises(TypeError):
            estimate_conditional_uniform_hand_belief(policy_input, (0, 1.5, 0, 0))

    def test_rejects_wrong_length(self) -> None:
        policy_input = _policy_input()
        with self.assertRaises(ValueError):
            estimate_conditional_uniform_hand_belief(policy_input, (0, 0, 0))

    def test_rejects_non_policy_input(self) -> None:
        with self.assertRaises(TypeError):
            estimate_conditional_uniform_hand_belief(None, (0, 0, 0, 0))

    def test_rejects_opponent_slot_sum_exceeding_total_hidden_slots(self) -> None:
        # accountedはself手牌の2枚だけなのでtotal_hidden_slotsは134。
        policy_input = _policy_input(
            concealed_tiles=(_tile(TileType(TileCategory.MANZU, 1)),) * 2
        )
        with self.assertRaises(ValueError):
            estimate_conditional_uniform_hand_belief(policy_input, (0, 200, 0, 0))


class SelfWindMappingTest(unittest.TestCase):
    def _assert_self_belief_matches_exact(
        self, self_seat: Seat, dealer_seat: Seat
    ) -> None:
        concealed_tiles = (_tile(TileType(TileCategory.MANZU, 1)),)
        policy_input = _policy_input(
            self_seat=self_seat,
            dealer_seat=dealer_seat,
            concealed_tiles=concealed_tiles,
        )
        slots = [0, 0, 0, 0]
        result = estimate_conditional_uniform_hand_belief(policy_input, tuple(slots))

        expected_self_belief = exact_self_belief(policy_input.own_hand)
        self_wind = wind_for_seat(self_seat, dealer_seat)
        self.assertEqual(result.hand(self_wind), expected_self_belief)

    def test_self_is_east(self) -> None:
        self._assert_self_belief_matches_exact(Seat.SEAT_0, Seat.SEAT_0)

    def test_self_is_south(self) -> None:
        self._assert_self_belief_matches_exact(Seat.SEAT_0, Seat.SEAT_3)

    def test_self_is_west(self) -> None:
        self._assert_self_belief_matches_exact(Seat.SEAT_0, Seat.SEAT_2)

    def test_self_is_north(self) -> None:
        self._assert_self_belief_matches_exact(Seat.SEAT_0, Seat.SEAT_1)


class UnequalOpponentSlotsTest(unittest.TestCase):
    def test_expected_count_scales_with_slot_count(self) -> None:
        # dealer=SEAT_0 -> self(SEAT_0)=EAST, SEAT_1=SOUTH, SEAT_2=WEST,
        # SEAT_3=NORTH。selfのconcealed手牌は1枚だけ(3m)。
        policy_input = _policy_input(
            concealed_tiles=(_tile(TileType(TileCategory.MANZU, 3)),)
        )
        south_slots = 13
        west_slots = 0
        north_slots = 5
        result = estimate_conditional_uniform_hand_belief(
            policy_input, (0, south_slots, west_slots, north_slots)
        )

        total_hidden_slots = 136 - 1  # self概算の1枚だけaccounted
        manzu_5 = TileType(TileCategory.MANZU, 5)
        remaining_5m = 4  # 5mはaccountされていない

        expected_south_raw = round_half_to_even_ratio(
            remaining_5m * south_slots * SCALE, total_hidden_slots
        )
        expected_north_raw = round_half_to_even_ratio(
            remaining_5m * north_slots * SCALE, total_hidden_slots
        )

        self.assertEqual(
            result.expected_count(Wind.SOUTH, manzu_5), expected_south_raw / SCALE
        )
        self.assertEqual(
            result.expected_count(Wind.NORTH, manzu_5), expected_north_raw / SCALE
        )
        # west_slots == 0 -> belief is exactly zero.
        self.assertEqual(result.expected_count(Wind.WEST, manzu_5), 0.0)
        self.assertGreater(expected_north_raw, 0)
        # north has more slots than west(0) so north > west.
        self.assertGreater(
            result.expected_count(Wind.NORTH, manzu_5),
            result.expected_count(Wind.WEST, manzu_5),
        )


class RedFivePresenceTest(unittest.TestCase):
    def test_remaining_red_five_is_distributed(self) -> None:
        policy_input = _policy_input()  # 完全にempty -> remaining red5m = 1
        result = estimate_conditional_uniform_hand_belief(policy_input, (0, 10, 0, 0))
        self.assertGreater(
            result.red_five_probability(Wind.SOUTH, TileCategory.MANZU), 0.0
        )

    def test_no_remaining_red_five_yields_zero_probability(self) -> None:
        # selfがすでに赤5mを保持している -> remaining red5m = 0
        policy_input = _policy_input(concealed_tiles=(_tile(MANZU_5, is_red=True),))
        result = estimate_conditional_uniform_hand_belief(policy_input, (0, 10, 0, 0))
        self.assertEqual(
            result.red_five_probability(Wind.SOUTH, TileCategory.MANZU), 0.0
        )


class ZeroHiddenSlotTest(unittest.TestCase):
    def test_all_zero_opponent_slots_with_zero_hidden_slots_is_valid(self) -> None:
        policy_input = _policy_input(concealed_tiles=_full_standard_tile_list())
        result = estimate_conditional_uniform_hand_belief(policy_input, (0, 0, 0, 0))
        for wind in (Wind.SOUTH, Wind.WEST, Wind.NORTH):
            self.assertEqual(sum(result.hand(wind).expected_count_raw), 0)
            self.assertEqual(sum(result.hand(wind).red_five_probability_raw), 0)

    def test_positive_opponent_slot_with_zero_hidden_slots_fails_closed(self) -> None:
        policy_input = _policy_input(concealed_tiles=_full_standard_tile_list())
        with self.assertRaises(ValueError):
            estimate_conditional_uniform_hand_belief(policy_input, (0, 1, 0, 0))


class DeterminismTest(unittest.TestCase):
    def test_same_input_yields_same_result(self) -> None:
        policy_input = _policy_input(
            concealed_tiles=(_tile(TileType(TileCategory.MANZU, 3)),)
        )
        slots = (0, 10, 5, 3)
        first = estimate_conditional_uniform_hand_belief(policy_input, slots)
        second = estimate_conditional_uniform_hand_belief(policy_input, slots)
        self.assertEqual(first, second)


class NonPlayerMassNotFullyAllocatedTest(unittest.TestCase):
    def test_wall_like_mass_is_not_assigned_to_the_single_opponent(self) -> None:
        # SOUTHだけがconcealed slotを持ち、それ以外のremaining massは
        # wall/dead wall相当としてどのplayerにも配分されない。
        policy_input = _policy_input(
            concealed_tiles=(_tile(TileType(TileCategory.MANZU, 3)),)
        )
        result = estimate_conditional_uniform_hand_belief(policy_input, (0, 1, 0, 0))

        souzu_9 = TileType(TileCategory.SOUZU, 9)
        # remaining souzu_9 = 4, total_hidden_slots = 135, south_slots = 1
        # -> real expectation は 4/135 ≈ 0.0296、4.0(全量配分)よりずっと小さい。
        self.assertLess(result.expected_count(Wind.SOUTH, souzu_9), 0.5)
        self.assertGreater(result.expected_count(Wind.SOUTH, souzu_9), 0.0)


class ExactHalfToEvenBoundaryTest(unittest.TestCase):
    def _conservation_with_single_type_remaining(
        self, remaining_index: int, remaining_count: int
    ) -> TileConservationResult:
        accounted = [4] * 34
        remaining = [0] * 34
        accounted[remaining_index] = 4 - remaining_count
        remaining[remaining_index] = remaining_count
        return TileConservationResult(
            exact_accounted_counts=tuple(accounted),
            exact_accounted_red_five_counts=(1, 1, 1),
            remaining_tile_counts=tuple(remaining),
            remaining_red_five_counts=(0, 0, 0),
        )

    def test_exact_half_rounds_to_even(self) -> None:
        index = tile_type_index(TileType(TileCategory.MANZU, 1))
        conservation = self._conservation_with_single_type_remaining(index, 1)
        # numerator = 1(remaining) * 1(player_slots) * SCALE(8192)
        # denominator = 16384 -> ratio == 0.5 exactly -> round-half-to-even -> 0
        belief = _baseline_hand_belief(
            conservation, player_slots=1, total_hidden_slots=16384
        )
        self.assertEqual(belief.expected_count_raw[index], 0)

    def test_exact_one_and_a_half_rounds_to_even(self) -> None:
        index = tile_type_index(TileType(TileCategory.MANZU, 1))
        conservation = self._conservation_with_single_type_remaining(index, 3)
        # numerator = 3 * 1 * SCALE, denominator = 16384 -> ratio == 1.5 -> 2
        belief = _baseline_hand_belief(
            conservation, player_slots=1, total_hidden_slots=16384
        )
        self.assertEqual(belief.expected_count_raw[index], 2)


class QuantizationErrorBoundTest(unittest.TestCase):
    def test_per_cell_error_bound(self) -> None:
        policy_input = _policy_input(
            concealed_tiles=(_tile(TileType(TileCategory.MANZU, 3)),)
        )
        south_slots = 13
        result = estimate_conditional_uniform_hand_belief(
            policy_input, (0, south_slots, 0, 0)
        )
        total_hidden_slots = 135

        conservation = derive_remaining_tile_inventory(policy_input)
        for index, remaining_count in enumerate(conservation.remaining_tile_counts):
            raw = result.hand(Wind.SOUTH).expected_count_raw[index]
            error = abs(
                raw * total_hidden_slots - remaining_count * south_slots * SCALE
            )
            self.assertLessEqual(2 * error, total_hidden_slots)

    def test_row_mass_drift_bound(self) -> None:
        policy_input = _policy_input(
            concealed_tiles=(_tile(TileType(TileCategory.MANZU, 3)),)
        )
        south_slots = 13
        result = estimate_conditional_uniform_hand_belief(
            policy_input, (0, south_slots, 0, 0)
        )
        row_mass = sum(result.hand(Wind.SOUTH).expected_count_raw)
        self.assertLessEqual(abs(row_mass - south_slots * SCALE), 17)


if __name__ == "__main__":
    unittest.main()
