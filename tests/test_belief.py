import unittest

from lisjong.belief.canonical_axes import (
    concealed_hand_offset,
    red_five_index,
    red_five_offset,
    seat_for_wind,
    tile_type_from_index,
    tile_type_index,
    wind_for_seat,
    wind_from_index,
    wind_index,
)
from lisjong.belief.concealed_hand_belief import ConcealedHandBelief
from lisjong.belief.fixed_point import (
    EXPECTED_COUNT_MAX_RAW,
    PROBABILITY_MAX_RAW,
    RED_FIVE_PROBABILITY_MAX_RAW,
    SCALE,
    expected_count_to_raw,
    probability_to_raw,
    raw_to_semantic,
    red_five_probability_to_raw,
    round_half_to_even_ratio,
)
from lisjong.belief.hand_belief import HandBelief
from lisjong.belief.self_belief import exact_self_belief
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import (
    EAST_WIND,
    GREEN_DRAGON,
    NORTH_WIND,
    RED_DRAGON,
    SOUTH_WIND,
    WEST_WIND,
    WHITE_DRAGON,
    Tile,
    TileCategory,
    TileType,
)
from lisjong.policy_contract.wind import Wind

ALL_TILE_TYPES = (
    [TileType(TileCategory.MANZU, rank) for rank in range(1, 10)]
    + [TileType(TileCategory.PINZU, rank) for rank in range(1, 10)]
    + [TileType(TileCategory.SOUZU, rank) for rank in range(1, 10)]
    + [
        EAST_WIND,
        SOUTH_WIND,
        WEST_WIND,
        NORTH_WIND,
        WHITE_DRAGON,
        GREEN_DRAGON,
        RED_DRAGON,
    ]
)


def _empty_expected_count_raw() -> tuple[int, ...]:
    return tuple(0 for _ in range(34))


def _empty_red_five_probability_raw() -> tuple[int, ...]:
    return (0, 0, 0)


class TileTypeIndexTest(unittest.TestCase):
    def test_all_34_tile_types_round_trip(self) -> None:
        seen_indexes = set()
        for tile_type in ALL_TILE_TYPES:
            index = tile_type_index(tile_type)
            self.assertNotIn(index, seen_indexes)
            seen_indexes.add(index)
            self.assertEqual(tile_type_from_index(index), tile_type)
        self.assertEqual(seen_indexes, set(range(34)))

    def test_manzu_1_is_index_0(self) -> None:
        self.assertEqual(tile_type_index(TileType(TileCategory.MANZU, 1)), 0)

    def test_red_dragon_is_index_33(self) -> None:
        self.assertEqual(tile_type_index(RED_DRAGON), 33)

    def test_rejects_out_of_range_index(self) -> None:
        with self.assertRaises(ValueError):
            tile_type_from_index(34)
        with self.assertRaises(ValueError):
            tile_type_from_index(-1)

    def test_rejects_non_tile_type(self) -> None:
        with self.assertRaises(TypeError):
            tile_type_index("1m")


class WindIndexTest(unittest.TestCase):
    def test_canonical_indexes_are_fixed(self) -> None:
        self.assertEqual(wind_index(Wind.EAST), 0)
        self.assertEqual(wind_index(Wind.SOUTH), 1)
        self.assertEqual(wind_index(Wind.WEST), 2)
        self.assertEqual(wind_index(Wind.NORTH), 3)

    def test_round_trips(self) -> None:
        for wind in Wind:
            self.assertEqual(wind_from_index(wind_index(wind)), wind)

    def test_rejects_out_of_range_index(self) -> None:
        with self.assertRaises(ValueError):
            wind_from_index(4)


class RedFiveIndexTest(unittest.TestCase):
    def test_canonical_indexes_are_fixed(self) -> None:
        self.assertEqual(red_five_index(TileCategory.MANZU), 0)
        self.assertEqual(red_five_index(TileCategory.PINZU), 1)
        self.assertEqual(red_five_index(TileCategory.SOUZU), 2)

    def test_rejects_honor_category(self) -> None:
        with self.assertRaises(ValueError):
            red_five_index(TileCategory.HONOR)


class SeatWindTest(unittest.TestCase):
    def test_dealer_is_always_east(self) -> None:
        for dealer_seat in Seat:
            self.assertEqual(wind_for_seat(dealer_seat, dealer_seat), Wind.EAST)

    def test_round_trips_for_every_dealer_seat(self) -> None:
        for dealer_seat in Seat:
            for seat in Seat:
                wind = wind_for_seat(seat, dealer_seat)
                self.assertEqual(seat_for_wind(wind, dealer_seat), seat)

    def test_fixed_dealer_seat_2_mapping(self) -> None:
        dealer_seat = Seat.SEAT_2
        self.assertEqual(wind_for_seat(Seat.SEAT_2, dealer_seat), Wind.EAST)
        self.assertEqual(wind_for_seat(Seat.SEAT_3, dealer_seat), Wind.SOUTH)
        self.assertEqual(wind_for_seat(Seat.SEAT_0, dealer_seat), Wind.WEST)
        self.assertEqual(wind_for_seat(Seat.SEAT_1, dealer_seat), Wind.NORTH)


class FlattenedOffsetTest(unittest.TestCase):
    def test_boundaries(self) -> None:
        manzu_1 = TileType(TileCategory.MANZU, 1)
        self.assertEqual(concealed_hand_offset(Wind.EAST, manzu_1), 0)
        self.assertEqual(concealed_hand_offset(Wind.EAST, RED_DRAGON), 33)
        self.assertEqual(concealed_hand_offset(Wind.SOUTH, manzu_1), 34)
        self.assertEqual(concealed_hand_offset(Wind.NORTH, RED_DRAGON), 135)

    def test_red_five_boundaries(self) -> None:
        self.assertEqual(red_five_offset(Wind.EAST, TileCategory.MANZU), 0)
        self.assertEqual(red_five_offset(Wind.EAST, TileCategory.SOUZU), 2)
        self.assertEqual(red_five_offset(Wind.SOUTH, TileCategory.MANZU), 3)
        self.assertEqual(red_five_offset(Wind.NORTH, TileCategory.SOUZU), 11)


class FixedPointTest(unittest.TestCase):
    def test_exact_integer_counts(self) -> None:
        for count, raw in ((0, 0), (1, 8192), (2, 16384), (3, 24576), (4, 32768)):
            self.assertEqual(raw, count * SCALE)
            self.assertEqual(raw_to_semantic(raw), float(count))

    def test_rejects_raw_above_16_bit_range(self) -> None:
        with self.assertRaises(ValueError):
            raw_to_semantic(0x10000)

    def test_rejects_non_int_raw(self) -> None:
        with self.assertRaises(TypeError):
            raw_to_semantic(1.0)


class SemanticRangeValidationTest(unittest.TestCase):
    def test_expected_count_accepts_lower_and_upper_bound(self) -> None:
        self.assertEqual(expected_count_to_raw(0.0), 0)
        self.assertEqual(expected_count_to_raw(4.0), EXPECTED_COUNT_MAX_RAW)

    def test_expected_count_rejects_below_zero(self) -> None:
        with self.assertRaises(ValueError):
            expected_count_to_raw(-1e-9)

    def test_expected_count_rejects_above_four(self) -> None:
        with self.assertRaises(ValueError):
            expected_count_to_raw(4.0 + 1e-9)

    def test_expected_count_rejects_negative_value_that_would_round_to_zero(
        self,
    ) -> None:
        # roundすればraw 0(範囲内)になるが、quantize前のsemantic valueが
        # 0.0未満であるため、round結果に関係なく拒否する。
        tiny_negative = -0.5 / SCALE / 2
        self.assertLess(tiny_negative, 0.0)
        self.assertEqual(round(tiny_negative * SCALE), 0)
        with self.assertRaises(ValueError):
            expected_count_to_raw(tiny_negative)

    def test_expected_count_rejects_value_above_four_that_would_round_to_max(
        self,
    ) -> None:
        tiny_excess = 4.0 + 0.5 / SCALE / 2
        self.assertGreater(tiny_excess, 4.0)
        self.assertEqual(round(tiny_excess * SCALE), EXPECTED_COUNT_MAX_RAW)
        with self.assertRaises(ValueError):
            expected_count_to_raw(tiny_excess)

    def test_red_five_probability_accepts_lower_and_upper_bound(self) -> None:
        self.assertEqual(red_five_probability_to_raw(0.0), 0)
        self.assertEqual(red_five_probability_to_raw(1.0), RED_FIVE_PROBABILITY_MAX_RAW)

    def test_red_five_probability_rejects_below_zero(self) -> None:
        with self.assertRaises(ValueError):
            red_five_probability_to_raw(-1e-9)

    def test_red_five_probability_rejects_above_one(self) -> None:
        with self.assertRaises(ValueError):
            red_five_probability_to_raw(1.0 + 1e-9)

    def test_probability_accepts_lower_and_upper_bound(self) -> None:
        self.assertEqual(probability_to_raw(0.0), 0)
        self.assertEqual(probability_to_raw(1.0), PROBABILITY_MAX_RAW)

    def test_probability_rejects_out_of_range_value(self) -> None:
        with self.assertRaises(ValueError):
            probability_to_raw(-1e-9)
        with self.assertRaises(ValueError):
            probability_to_raw(1.0 + 1e-9)

    def test_probability_rejects_bool(self) -> None:
        with self.assertRaises(TypeError):
            probability_to_raw(True)

    def test_probability_shares_the_red_five_probability_scale(self) -> None:
        self.assertEqual(PROBABILITY_MAX_RAW, RED_FIVE_PROBABILITY_MAX_RAW)
        for value in (0.0, 0.25, 0.5, 1.0):
            self.assertEqual(
                probability_to_raw(value), red_five_probability_to_raw(value)
            )


class RoundingRuleTest(unittest.TestCase):
    def test_half_way_value_rounds_to_even_raw(self) -> None:
        # 1 / 16384 * SCALE(8192) == 0.5 exactly; round-half-to-evenでは
        # 0(偶数)へ丸める。
        half_way_to_zero = 1 / 16384
        self.assertEqual(expected_count_to_raw(half_way_to_zero), 0)

        # 3 / 16384 * SCALE(8192) == 1.5 exactly; round-half-to-evenでは
        # 2(偶数)へ丸める(1ではない)。
        half_way_to_two = 3 / 16384
        self.assertEqual(expected_count_to_raw(half_way_to_two), 2)


class RoundHalfToEvenRatioTest(unittest.TestCase):
    def test_matches_python_round_for_exact_binary_fractions(self) -> None:
        # 1/16384 * SCALE(8192) == 0.5 exactly -> round-half-to-evenは0。
        self.assertEqual(round_half_to_even_ratio(1 * SCALE, 16384), 0)
        # 3/16384 * SCALE(8192) == 1.5 exactly -> round-half-to-evenは2。
        self.assertEqual(round_half_to_even_ratio(3 * SCALE, 16384), 2)

    def test_rounds_down_below_half(self) -> None:
        self.assertEqual(round_half_to_even_ratio(1, 3), 0)

    def test_rounds_up_above_half(self) -> None:
        self.assertEqual(round_half_to_even_ratio(2, 3), 1)

    def test_exact_division_has_no_error(self) -> None:
        self.assertEqual(round_half_to_even_ratio(12, 4), 3)

    def test_zero_numerator_is_zero(self) -> None:
        self.assertEqual(round_half_to_even_ratio(0, 5), 0)

    def test_rejects_non_positive_denominator(self) -> None:
        with self.assertRaises(ValueError):
            round_half_to_even_ratio(1, 0)
        with self.assertRaises(ValueError):
            round_half_to_even_ratio(1, -1)

    def test_rejects_negative_numerator(self) -> None:
        with self.assertRaises(ValueError):
            round_half_to_even_ratio(-1, 5)

    def test_rejects_non_int_arguments(self) -> None:
        with self.assertRaises(TypeError):
            round_half_to_even_ratio(1.0, 5)
        with self.assertRaises(TypeError):
            round_half_to_even_ratio(1, 5.0)


class HandBeliefTest(unittest.TestCase):
    def _expected_count_raw(self, overrides: dict) -> tuple[int, ...]:
        values = list(_empty_expected_count_raw())
        for tile_type, raw in overrides.items():
            values[tile_type_index(tile_type)] = raw
        return tuple(values)

    def test_accepts_boundary_expected_count_raw(self) -> None:
        manzu_5 = TileType(TileCategory.MANZU, 5)
        for raw in (0, 8192, 16384, 24576, EXPECTED_COUNT_MAX_RAW):
            belief = HandBelief(
                expected_count_raw=self._expected_count_raw({manzu_5: raw}),
                red_five_probability_raw=_empty_red_five_probability_raw(),
            )
            self.assertEqual(belief.expected_count(manzu_5), raw / SCALE)

    def test_rejects_expected_count_raw_above_max(self) -> None:
        manzu_5 = TileType(TileCategory.MANZU, 5)
        with self.assertRaises(ValueError):
            HandBelief(
                expected_count_raw=self._expected_count_raw(
                    {manzu_5: EXPECTED_COUNT_MAX_RAW + 1}
                ),
                red_five_probability_raw=_empty_red_five_probability_raw(),
            )

    def test_rejects_red_five_probability_raw_above_max(self) -> None:
        manzu_5 = TileType(TileCategory.MANZU, 5)
        with self.assertRaises(ValueError):
            HandBelief(
                expected_count_raw=self._expected_count_raw({manzu_5: SCALE}),
                red_five_probability_raw=(RED_FIVE_PROBABILITY_MAX_RAW + 1, 0, 0),
            )

    def test_rejects_red_five_probability_exceeding_expected_count(self) -> None:
        manzu_5 = TileType(TileCategory.MANZU, 5)
        expected_count_raw = self._expected_count_raw({manzu_5: 100})
        with self.assertRaises(ValueError):
            HandBelief(
                expected_count_raw=expected_count_raw,
                red_five_probability_raw=(101, 0, 0),
            )

    def test_allows_red_five_probability_equal_to_expected_count(self) -> None:
        manzu_5 = TileType(TileCategory.MANZU, 5)
        expected_count_raw = self._expected_count_raw({manzu_5: 100})
        belief = HandBelief(
            expected_count_raw=expected_count_raw,
            red_five_probability_raw=(100, 0, 0),
        )
        self.assertEqual(belief.red_five_probability_raw[0], 100)

    def test_allows_red_five_probability_one_raw_unit_below(self) -> None:
        manzu_5 = TileType(TileCategory.MANZU, 5)
        expected_count_raw = self._expected_count_raw({manzu_5: 100})
        belief = HandBelief(
            expected_count_raw=expected_count_raw,
            red_five_probability_raw=(99, 0, 0),
        )
        self.assertEqual(belief.red_five_probability_raw[0], 99)

    def test_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            HandBelief(
                expected_count_raw=tuple(range(33)),
                red_five_probability_raw=_empty_red_five_probability_raw(),
            )


class ConcealedHandBeliefTest(unittest.TestCase):
    def test_flattened_layout_matches_offsets(self) -> None:
        hands = tuple(
            HandBelief(
                expected_count_raw=tuple(
                    SCALE if index == wind_number else 0 for index in range(34)
                ),
                red_five_probability_raw=_empty_red_five_probability_raw(),
            )
            for wind_number in range(4)
        )
        belief = ConcealedHandBelief(hands=hands)

        for wind in Wind:
            tile_type = tile_type_from_index(wind_index(wind))
            offset = concealed_hand_offset(wind, tile_type)
            self.assertEqual(belief.flattened_expected_count_raw[offset], SCALE)
            self.assertEqual(belief.expected_count(wind, tile_type), 1.0)

        self.assertEqual(len(belief.flattened_expected_count_raw), 136)
        self.assertEqual(len(belief.flattened_red_five_probability_raw), 12)

    def test_rejects_wrong_number_of_hands(self) -> None:
        empty_hand = HandBelief(
            expected_count_raw=_empty_expected_count_raw(),
            red_five_probability_raw=_empty_red_five_probability_raw(),
        )
        with self.assertRaises(ValueError):
            ConcealedHandBelief(hands=(empty_hand, empty_hand, empty_hand))


class ExactSelfBeliefTest(unittest.TestCase):
    def test_mixed_normal_and_red_five(self) -> None:
        manzu_5 = Tile(TileType(TileCategory.MANZU, 5))
        manzu_5_red = Tile(TileType(TileCategory.MANZU, 5), is_red=True)
        pinzu_1 = Tile(TileType(TileCategory.PINZU, 1))

        own_hand = OwnHandState(
            concealed_tiles=(manzu_5, manzu_5_red, pinzu_1), drawn_tile=pinzu_1
        )
        belief = exact_self_belief(own_hand)

        self.assertEqual(belief.expected_count(manzu_5.tile_type), 2.0)
        self.assertEqual(belief.red_five_probability(TileCategory.MANZU), 1.0)
        self.assertEqual(belief.red_five_probability(TileCategory.PINZU), 0.0)
        self.assertEqual(belief.expected_count(pinzu_1.tile_type), 1.0)

    def test_drawn_tile_is_not_double_counted(self) -> None:
        pinzu_1 = Tile(TileType(TileCategory.PINZU, 1))
        own_hand = OwnHandState(concealed_tiles=(pinzu_1,), drawn_tile=pinzu_1)
        belief = exact_self_belief(own_hand)
        self.assertEqual(belief.expected_count(pinzu_1.tile_type), 1.0)

    def test_four_of_a_kind(self) -> None:
        souzu_9 = TileType(TileCategory.SOUZU, 9)
        tiles = tuple(Tile(souzu_9) for _ in range(4))
        own_hand = OwnHandState(concealed_tiles=tiles, drawn_tile=None)
        belief = exact_self_belief(own_hand)
        self.assertEqual(belief.expected_count(souzu_9), 4.0)

    def test_empty_hand_is_not_rejected(self) -> None:
        own_hand = OwnHandState(concealed_tiles=(), drawn_tile=None)
        belief = exact_self_belief(own_hand)
        self.assertEqual(sum(belief.expected_count_raw), 0)
        self.assertEqual(sum(belief.red_five_probability_raw), 0)

    def test_expected_count_sums_to_concealed_tile_count(self) -> None:
        tiles = (
            Tile(TileType(TileCategory.MANZU, 1)),
            Tile(TileType(TileCategory.MANZU, 5)),
            Tile(TileType(TileCategory.MANZU, 5), is_red=True),
            Tile(TileType(TileCategory.PINZU, 9)),
            Tile(EAST_WIND),
        )
        own_hand = OwnHandState(concealed_tiles=tiles, drawn_tile=None)
        belief = exact_self_belief(own_hand)
        self.assertEqual(sum(belief.expected_count_raw), len(tiles) * SCALE)

    def test_rejects_non_own_hand_state(self) -> None:
        with self.assertRaises(TypeError):
            exact_self_belief(None)


if __name__ == "__main__":
    unittest.main()
