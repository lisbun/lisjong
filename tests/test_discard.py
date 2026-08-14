import unittest
from dataclasses import FrozenInstanceError, fields

from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))


class DiscardTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        discard = Discard(tile=MANZU_5, tsumogiri=True, order=0, called_by=Seat.SEAT_2)
        self.assertEqual(discard.tile, MANZU_5)
        self.assertTrue(discard.tsumogiri)
        self.assertEqual(discard.order, 0)
        self.assertEqual(discard.called_by, Seat.SEAT_2)

    def test_allows_called_by_none(self) -> None:
        discard = Discard(tile=MANZU_5, tsumogiri=False, order=3, called_by=None)
        self.assertIsNone(discard.called_by)

    def test_is_immutable(self) -> None:
        discard = Discard(tile=MANZU_5, tsumogiri=False, order=0, called_by=None)
        with self.assertRaises(FrozenInstanceError):
            discard.order = 1

    def test_equal_values_compare_equal_and_are_hashable(self) -> None:
        first = Discard(tile=MANZU_5, tsumogiri=False, order=2, called_by=None)
        second = Discard(tile=MANZU_5, tsumogiri=False, order=2, called_by=None)
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)

    def test_rejects_non_tile(self) -> None:
        with self.assertRaises(TypeError):
            Discard(tile="5m", tsumogiri=False, order=0, called_by=None)

    def test_rejects_non_boolean_tsumogiri(self) -> None:
        with self.assertRaises(TypeError):
            Discard(tile=MANZU_5, tsumogiri=1, order=0, called_by=None)

    def test_rejects_boolean_order(self) -> None:
        with self.assertRaises(TypeError):
            Discard(tile=MANZU_5, tsumogiri=False, order=True, called_by=None)

    def test_rejects_non_integer_order(self) -> None:
        with self.assertRaises(TypeError):
            Discard(tile=MANZU_5, tsumogiri=False, order=1.0, called_by=None)

    def test_rejects_negative_order(self) -> None:
        with self.assertRaises(ValueError):
            Discard(tile=MANZU_5, tsumogiri=False, order=-1, called_by=None)

    def test_rejects_non_seat_called_by(self) -> None:
        with self.assertRaises(TypeError):
            Discard(tile=MANZU_5, tsumogiri=False, order=0, called_by=1)

    def test_has_no_ron_specific_field(self) -> None:
        field_names = {field.name for field in fields(Discard)}
        self.assertEqual(field_names, {"tile", "tsumogiri", "order", "called_by"})
        for forbidden in ("is_ron", "ron_by", "won_by"):
            self.assertNotIn(forbidden, field_names)


if __name__ == "__main__":
    unittest.main()
