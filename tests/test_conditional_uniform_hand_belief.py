import unittest

from lisjong.belief.canonical_axes import (
    red_five_index,
    tile_type_index,
    wind_for_seat,
)
from lisjong.belief.conditional_uniform_hand_belief import (
    _allocate_fixed_point_pool,
    estimate_conditional_uniform_hand_belief,
)
from lisjong.belief.fixed_point import SCALE, round_half_to_even_ratio
from lisjong.belief.self_belief import exact_self_belief
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
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
        souzu_9 = TileType(TileCategory.SOUZU, 9)
        tile_index = tile_type_index(souzu_9)
        remaining_count = 4
        south_raw = result.hand(Wind.SOUTH).expected_count_raw[tile_index]
        north_raw = result.hand(Wind.NORTH).expected_count_raw[tile_index]

        self.assertLess(
            abs(south_raw * total_hidden_slots - remaining_count * south_slots * SCALE),
            total_hidden_slots,
        )
        self.assertLess(
            abs(north_raw * total_hidden_slots - remaining_count * north_slots * SCALE),
            total_hidden_slots,
        )
        # west_slots == 0 -> belief is exactly zero.
        self.assertEqual(result.expected_count(Wind.WEST, souzu_9), 0.0)
        self.assertGreater(north_raw, 0)
        # north has more slots than west(0) so north > west.
        self.assertGreater(
            result.expected_count(Wind.NORTH, souzu_9),
            result.expected_count(Wind.WEST, souzu_9),
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


class WaitBeliefNotProvidedTest(unittest.TestCase):
    def test_baseline_estimator_leaves_wait_belief_unprovided(self) -> None:
        # Issue #82のwait beliefは、意味的に誤ったall-zeroを既存estimatorへ
        # 自動付与せず、未提供（None）のままとする。
        policy_input = _policy_input(
            concealed_tiles=(_tile(TileType(TileCategory.MANZU, 3)),)
        )
        result = estimate_conditional_uniform_hand_belief(policy_input, (0, 10, 5, 3))

        for wind in Wind:
            hand = result.hand(wind)
            self.assertFalse(hand.has_wait_belief)
            self.assertFalse(hand.has_wait_mechanism_belief)
            self.assertIsNone(hand.wait_probability(MANZU_5))


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


class FixedPointPoolAllocationTest(unittest.TestCase):
    def test_known_8193_raw_counterexample_is_conserved(self) -> None:
        allocated = _allocate_fixed_point_pool(1, (0, 1, 1, 1), 3)
        self.assertEqual(allocated, (0, 2731, 2731, 2730))
        self.assertEqual(sum(allocated), SCALE)

    def test_aggregate_target_uses_round_half_to_even(self) -> None:
        self.assertEqual(
            _allocate_fixed_point_pool(1, (1, 0, 0, 0), 16384),
            (0, 0, 0, 0),
        )
        self.assertEqual(
            _allocate_fixed_point_pool(3, (1, 0, 0, 0), 16384),
            (2, 0, 0, 0),
        )

    def test_asymmetric_allocations_are_floor_or_ceil(self) -> None:
        remaining_count = 4
        slot_counts = (0, 1, 2, 4)
        total_hidden_slots = 11
        allocated = _allocate_fixed_point_pool(
            remaining_count, slot_counts, total_hidden_slots
        )

        expected_total = round_half_to_even_ratio(
            remaining_count * sum(slot_counts) * SCALE, total_hidden_slots
        )
        self.assertEqual(sum(allocated), expected_total)
        for raw, player_slots in zip(allocated, slot_counts, strict=True):
            numerator = remaining_count * player_slots * SCALE
            floor_raw, remainder = divmod(numerator, total_hidden_slots)
            self.assertIn(raw, {floor_raw, floor_raw + bool(remainder)})

    def test_zero_remaining_and_zero_hidden_slots(self) -> None:
        self.assertEqual(_allocate_fixed_point_pool(0, (0, 0, 0, 0), 7), (0, 0, 0, 0))
        self.assertEqual(_allocate_fixed_point_pool(0, (0, 0, 0, 0), 0), (0, 0, 0, 0))


class GlobalConservationTest(unittest.TestCase):
    def _opponent_winds(self) -> tuple[Wind, Wind, Wind]:
        return (Wind.SOUTH, Wind.WEST, Wind.NORTH)

    def test_known_counterexample_is_conserved_end_to_end(self) -> None:
        concealed_tiles = list(_full_standard_tile_list())
        for rank in (1, 2, 3):
            concealed_tiles.remove(_tile(TileType(TileCategory.MANZU, rank)))
        policy_input = _policy_input(concealed_tiles=tuple(concealed_tiles))
        result = estimate_conditional_uniform_hand_belief(policy_input, (0, 1, 1, 1))
        tile_index = tile_type_index(TileType(TileCategory.MANZU, 1))
        allocated = tuple(
            result.hand(wind).expected_count_raw[tile_index]
            for wind in self._opponent_winds()
        )

        self.assertEqual(allocated, (2731, 2731, 2730))
        self.assertEqual(sum(allocated), SCALE)

    def test_partial_allocation_preserves_each_physical_pool_target(self) -> None:
        policy_input = _policy_input()
        slots = (0, 13, 7, 5)
        opponent_slots = sum(slots)
        total_hidden_slots = 136
        result = estimate_conditional_uniform_hand_belief(policy_input, slots)
        conservation = derive_remaining_tile_inventory(policy_input)
        five_indices = {
            tile_type_index(TileType(category, 5))
            for category in (
                TileCategory.MANZU,
                TileCategory.PINZU,
                TileCategory.SOUZU,
            )
        }

        for tile_index, remaining_count in enumerate(
            conservation.remaining_tile_counts
        ):
            allocated_total = sum(
                result.hand(wind).expected_count_raw[tile_index]
                for wind in self._opponent_winds()
            )
            self.assertLessEqual(allocated_total, remaining_count * SCALE)
            if tile_index not in five_indices:
                self.assertEqual(
                    allocated_total,
                    round_half_to_even_ratio(
                        remaining_count * opponent_slots * SCALE,
                        total_hidden_slots,
                    ),
                )

        for category in (
            TileCategory.MANZU,
            TileCategory.PINZU,
            TileCategory.SOUZU,
        ):
            five_index = tile_type_index(TileType(category, 5))
            color_index = red_five_index(category)
            remaining_five = conservation.remaining_tile_counts[five_index]
            remaining_red = conservation.remaining_red_five_counts[color_index]
            remaining_normal = remaining_five - remaining_red
            normal_allocated = _allocate_fixed_point_pool(
                remaining_normal, slots, total_hidden_slots
            )
            red_allocated = _allocate_fixed_point_pool(
                remaining_red, slots, total_hidden_slots
            )
            for wind_number, wind in enumerate(
                (Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH)
            ):
                five_raw = result.hand(wind).expected_count_raw[five_index]
                red_raw = result.hand(wind).red_five_probability_raw[color_index]
                self.assertEqual(red_raw, red_allocated[wind_number])
                self.assertEqual(
                    five_raw,
                    normal_allocated[wind_number] + red_allocated[wind_number],
                )
            red_total = sum(
                result.hand(wind).red_five_probability_raw[color_index]
                for wind in self._opponent_winds()
            )
            normal_total = sum(
                result.hand(wind).expected_count_raw[five_index]
                - result.hand(wind).red_five_probability_raw[color_index]
                for wind in self._opponent_winds()
            )
            self.assertEqual(
                red_total,
                round_half_to_even_ratio(
                    remaining_red * opponent_slots * SCALE, total_hidden_slots
                ),
            )
            self.assertEqual(
                normal_total,
                round_half_to_even_ratio(
                    remaining_normal * opponent_slots * SCALE, total_hidden_slots
                ),
            )

    def test_full_allocation_is_exact_for_basic_red_and_normal_pools(self) -> None:
        policy_input = _policy_input()
        result = estimate_conditional_uniform_hand_belief(policy_input, (0, 46, 45, 45))
        conservation = derive_remaining_tile_inventory(policy_input)

        for tile_index, remaining_count in enumerate(
            conservation.remaining_tile_counts
        ):
            allocated_total = sum(
                result.hand(wind).expected_count_raw[tile_index]
                for wind in self._opponent_winds()
            )
            self.assertEqual(allocated_total, remaining_count * SCALE)

        for category in (
            TileCategory.MANZU,
            TileCategory.PINZU,
            TileCategory.SOUZU,
        ):
            five_index = tile_type_index(TileType(category, 5))
            color_index = red_five_index(category)
            red_total = 0
            normal_total = 0
            for wind in self._opponent_winds():
                five_raw = result.hand(wind).expected_count_raw[five_index]
                red_raw = result.hand(wind).red_five_probability_raw[color_index]
                self.assertLessEqual(red_raw, five_raw)
                red_total += red_raw
                normal_total += five_raw - red_raw
            remaining_red = conservation.remaining_red_five_counts[color_index]
            remaining_five = conservation.remaining_tile_counts[five_index]
            self.assertEqual(red_total, remaining_red * SCALE)
            self.assertEqual(normal_total, (remaining_five - remaining_red) * SCALE)

    def test_all_zero_opponent_slots_allocate_no_mass(self) -> None:
        result = estimate_conditional_uniform_hand_belief(_policy_input(), (0, 0, 0, 0))
        for wind in self._opponent_winds():
            self.assertEqual(sum(result.hand(wind).expected_count_raw), 0)
            self.assertEqual(sum(result.hand(wind).red_five_probability_raw), 0)


class QuantizationErrorBoundTest(unittest.TestCase):
    def test_per_cell_error_bound(self) -> None:
        policy_input = _policy_input(
            concealed_tiles=(_tile(TileType(TileCategory.MANZU, 3)),)
        )
        slots = (0, 13, 7, 5)
        result = estimate_conditional_uniform_hand_belief(policy_input, slots)
        total_hidden_slots = 135

        conservation = derive_remaining_tile_inventory(policy_input)
        five_indices = {
            tile_type_index(TileType(category, 5))
            for category in (
                TileCategory.MANZU,
                TileCategory.PINZU,
                TileCategory.SOUZU,
            )
        }
        for wind, player_slots in zip(Wind, slots, strict=True):
            if wind is Wind.EAST:
                continue
            for index, remaining_count in enumerate(conservation.remaining_tile_counts):
                raw = result.hand(wind).expected_count_raw[index]
                error = abs(
                    raw * total_hidden_slots - remaining_count * player_slots * SCALE
                )
                error_bound = (
                    2 * total_hidden_slots
                    if index in five_indices
                    else total_hidden_slots
                )
                self.assertLess(error, error_bound)

    def test_row_mass_drift_bound(self) -> None:
        policy_input = _policy_input(
            concealed_tiles=(_tile(TileType(TileCategory.MANZU, 3)),)
        )
        slots = (0, 13, 7, 5)
        result = estimate_conditional_uniform_hand_belief(policy_input, slots)
        for wind, player_slots in zip(Wind, slots, strict=True):
            if wind is Wind.EAST:
                continue
            row_mass = sum(result.hand(wind).expected_count_raw)
            self.assertLess(abs(row_mass - player_slots * SCALE), 37)


if __name__ == "__main__":
    unittest.main()
