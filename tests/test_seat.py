import unittest

from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.wind import Wind


class SeatTest(unittest.TestCase):
    def test_has_exactly_four_seats_numbered_0_to_3(self) -> None:
        self.assertEqual([int(seat) for seat in Seat], [0, 1, 2, 3])

    def test_int_equality_is_intentional_for_seat_arithmetic(self) -> None:
        # IntEnumによりSeat.SEAT_0 == 0が成立するのは、(seat + 1) mod 4という
        # docsの契約をそのまま算術で表現するための意図的な性質である。
        self.assertEqual(Seat.SEAT_0, 0)
        self.assertEqual(Seat.SEAT_3, 3)
        self.assertEqual(int(Seat.SEAT_2), 2)

    def test_rejects_out_of_range_value(self) -> None:
        for value in (-1, 4):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Seat(value)

    def test_shimocha_relationship_uses_mod_four(self) -> None:
        # (seat + 1) mod 4 = 下家
        expected_shimocha = {
            Seat.SEAT_0: Seat.SEAT_1,
            Seat.SEAT_1: Seat.SEAT_2,
            Seat.SEAT_2: Seat.SEAT_3,
            Seat.SEAT_3: Seat.SEAT_0,
        }
        for seat, shimocha in expected_shimocha.items():
            with self.subTest(seat=seat):
                self.assertEqual(Seat((int(seat) + 1) % 4), shimocha)

    def test_does_not_equal_wind_with_same_underlying_number(self) -> None:
        self.assertNotEqual(Seat.SEAT_0, Wind.EAST)
        self.assertNotIn(Seat.SEAT_0, set(Wind))

    def test_seat_and_wind_do_not_collide_as_dict_keys(self) -> None:
        mapping = {Seat.SEAT_0: "seat", Wind.EAST: "wind"}
        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping[Seat.SEAT_0], "seat")
        self.assertEqual(mapping[Wind.EAST], "wind")


if __name__ == "__main__":
    unittest.main()
