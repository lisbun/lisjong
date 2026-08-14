import unittest

from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.wind import Wind


class WindTest(unittest.TestCase):
    def test_has_four_winds(self) -> None:
        self.assertEqual(
            {wind.value for wind in Wind},
            {"east", "south", "west", "north"},
        )

    def test_is_not_int_convertible(self) -> None:
        # WindはSeatと異なり、mod-4算術（seat rotation）へ直接使ってはならない。
        with self.assertRaises(TypeError):
            int(Wind.EAST)

    def test_does_not_equal_seat_with_same_underlying_number(self) -> None:
        self.assertNotEqual(Wind.EAST, Seat.SEAT_0)


if __name__ == "__main__":
    unittest.main()
