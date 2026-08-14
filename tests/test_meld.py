import unittest
from dataclasses import FrozenInstanceError, fields

from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import EAST_WIND, Tile, TileCategory, TileType

MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))
MANZU_5_RED = Tile(TileType(TileCategory.MANZU, 5), is_red=True)
MANZU_6 = Tile(TileType(TileCategory.MANZU, 6))
MANZU_7 = Tile(TileType(TileCategory.MANZU, 7))
PINZU_5 = Tile(TileType(TileCategory.PINZU, 5))
PINZU_5_RED = Tile(TileType(TileCategory.PINZU, 5), is_red=True)
PINZU_6 = Tile(TileType(TileCategory.PINZU, 6))
EAST_TILE = Tile(EAST_WIND)


class MeldKindTest(unittest.TestCase):
    def test_has_five_kinds(self) -> None:
        self.assertEqual(
            {kind.value for kind in MeldKind},
            {"chi", "pon", "daiminkan", "ankan", "kakan"},
        )


class PublicMeldCommonTest(unittest.TestCase):
    def test_is_immutable(self) -> None:
        meld = PublicMeld(
            kind=MeldKind.PON,
            tiles=(PINZU_5, PINZU_5, PINZU_5),
            from_seat=Seat.SEAT_1,
            called_tile=PINZU_5,
        )
        with self.assertRaises(FrozenInstanceError):
            meld.kind = MeldKind.CHI

    def test_rejects_non_meld_kind(self) -> None:
        with self.assertRaises(TypeError):
            PublicMeld(
                kind="pon",
                tiles=(PINZU_5, PINZU_5, PINZU_5),
                from_seat=Seat.SEAT_1,
                called_tile=PINZU_5,
            )

    def test_rejects_non_tile_in_tiles(self) -> None:
        with self.assertRaises(TypeError):
            PublicMeld(
                kind=MeldKind.PON,
                tiles=(PINZU_5, PINZU_5, "5p"),
                from_seat=Seat.SEAT_1,
                called_tile=PINZU_5,
            )

    def test_input_order_does_not_affect_identity(self) -> None:
        first = PublicMeld(
            kind=MeldKind.PON,
            tiles=(PINZU_5, PINZU_5, PINZU_5_RED),
            from_seat=Seat.SEAT_1,
            called_tile=PINZU_5,
        )
        second = PublicMeld(
            kind=MeldKind.PON,
            tiles=(PINZU_5_RED, PINZU_5, PINZU_5),
            from_seat=Seat.SEAT_1,
            called_tile=PINZU_5,
        )
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)

    def test_red_construction_and_multiplicity_are_preserved(self) -> None:
        with_one_red = PublicMeld(
            kind=MeldKind.ANKAN,
            tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5_RED),
            from_seat=None,
            called_tile=None,
        )
        all_normal = PublicMeld(
            kind=MeldKind.ANKAN,
            tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5),
            from_seat=None,
            called_tile=None,
        )
        self.assertNotEqual(with_one_red, all_normal)

    def test_has_no_physical_copy_or_source_meld_reference_fields(self) -> None:
        field_names = {field.name for field in fields(PublicMeld)}
        self.assertEqual(field_names, {"kind", "tiles", "from_seat", "called_tile"})
        for forbidden in ("source_meld_id", "source_meld_index", "pon", "owner"):
            self.assertNotIn(forbidden, field_names)


class PublicMeldChiTest(unittest.TestCase):
    def _make(self, tiles=(MANZU_4, MANZU_5, MANZU_6), called_tile=MANZU_5):
        return PublicMeld(
            kind=MeldKind.CHI,
            tiles=tiles,
            from_seat=Seat.SEAT_0,
            called_tile=called_tile,
        )

    def test_creates_with_valid_values(self) -> None:
        meld = self._make()
        self.assertEqual(meld.tiles, (MANZU_4, MANZU_5, MANZU_6))

    def test_rejects_wrong_tile_count(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(MANZU_4, MANZU_5))

    def test_rejects_honor_tiles(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(EAST_TILE, EAST_TILE, EAST_TILE), called_tile=EAST_TILE)

    def test_rejects_non_consecutive_ranks(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(MANZU_4, MANZU_5, MANZU_7), called_tile=MANZU_5)

    def test_rejects_mixed_suits(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(MANZU_4, MANZU_5, PINZU_6), called_tile=MANZU_5)

    def test_requires_from_seat(self) -> None:
        with self.assertRaises(ValueError):
            PublicMeld(
                kind=MeldKind.CHI,
                tiles=(MANZU_4, MANZU_5, MANZU_6),
                from_seat=None,
                called_tile=MANZU_5,
            )

    def test_requires_called_tile(self) -> None:
        with self.assertRaises(ValueError):
            PublicMeld(
                kind=MeldKind.CHI,
                tiles=(MANZU_4, MANZU_5, MANZU_6),
                from_seat=Seat.SEAT_0,
                called_tile=None,
            )

    def test_rejects_called_tile_not_in_tiles(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(MANZU_4, MANZU_5, MANZU_6), called_tile=MANZU_7)

    def test_rejects_called_tile_with_mismatched_red_distinction(self) -> None:
        # called_tile=5mrだが、tilesは通常5mしか含まない場合は不整合として拒否する。
        with self.assertRaises(ValueError):
            self._make(tiles=(MANZU_4, MANZU_5, MANZU_6), called_tile=MANZU_5_RED)


class PublicMeldPonTest(unittest.TestCase):
    def _make(self, tiles=(PINZU_5, PINZU_5, PINZU_5), called_tile=PINZU_5):
        return PublicMeld(
            kind=MeldKind.PON,
            tiles=tiles,
            from_seat=Seat.SEAT_1,
            called_tile=called_tile,
        )

    def test_creates_with_valid_values(self) -> None:
        meld = self._make(tiles=(PINZU_5, PINZU_5, PINZU_5_RED))
        self.assertEqual(len(meld.tiles), 3)

    def test_rejects_wrong_tile_count(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(PINZU_5, PINZU_5))

    def test_rejects_different_base_tile_kind(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(PINZU_5, PINZU_5, PINZU_6))

    def test_requires_from_seat(self) -> None:
        with self.assertRaises(ValueError):
            PublicMeld(
                kind=MeldKind.PON,
                tiles=(PINZU_5, PINZU_5, PINZU_5),
                from_seat=None,
                called_tile=PINZU_5,
            )

    def test_requires_called_tile(self) -> None:
        with self.assertRaises(ValueError):
            PublicMeld(
                kind=MeldKind.PON,
                tiles=(PINZU_5, PINZU_5, PINZU_5),
                from_seat=Seat.SEAT_1,
                called_tile=None,
            )

    def test_rejects_called_tile_with_mismatched_red_distinction(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(PINZU_5, PINZU_5, PINZU_5), called_tile=PINZU_5_RED)


class PublicMeldDaiminkanTest(unittest.TestCase):
    def _make(
        self,
        tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5_RED),
        called_tile=PINZU_5,
    ):
        return PublicMeld(
            kind=MeldKind.DAIMINKAN,
            tiles=tiles,
            from_seat=Seat.SEAT_1,
            called_tile=called_tile,
        )

    def test_creates_with_valid_values(self) -> None:
        meld = self._make()
        self.assertEqual(len(meld.tiles), 4)

    def test_rejects_wrong_tile_count(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(PINZU_5, PINZU_5, PINZU_5))

    def test_rejects_different_base_tile_kind(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_6))

    def test_requires_from_seat(self) -> None:
        with self.assertRaises(ValueError):
            PublicMeld(
                kind=MeldKind.DAIMINKAN,
                tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5),
                from_seat=None,
                called_tile=PINZU_5,
            )

    def test_requires_called_tile(self) -> None:
        with self.assertRaises(ValueError):
            PublicMeld(
                kind=MeldKind.DAIMINKAN,
                tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5),
                from_seat=Seat.SEAT_1,
                called_tile=None,
            )


class PublicMeldAnkanTest(unittest.TestCase):
    def _make(self, tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5_RED)):
        return PublicMeld(
            kind=MeldKind.ANKAN,
            tiles=tiles,
            from_seat=None,
            called_tile=None,
        )

    def test_creates_with_valid_values(self) -> None:
        meld = self._make()
        self.assertIsNone(meld.from_seat)
        self.assertIsNone(meld.called_tile)

    def test_rejects_wrong_tile_count(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(PINZU_5, PINZU_5, PINZU_5))

    def test_rejects_different_base_tile_kind(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_6))

    def test_rejects_non_none_from_seat(self) -> None:
        with self.assertRaises(ValueError):
            PublicMeld(
                kind=MeldKind.ANKAN,
                tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5),
                from_seat=Seat.SEAT_0,
                called_tile=None,
            )

    def test_rejects_non_none_called_tile(self) -> None:
        with self.assertRaises(ValueError):
            PublicMeld(
                kind=MeldKind.ANKAN,
                tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5),
                from_seat=None,
                called_tile=PINZU_5,
            )


class PublicMeldKakanTest(unittest.TestCase):
    def _make(
        self,
        tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5_RED),
        from_seat=Seat.SEAT_1,
        called_tile=PINZU_5,
    ):
        return PublicMeld(
            kind=MeldKind.KAKAN,
            tiles=tiles,
            from_seat=from_seat,
            called_tile=called_tile,
        )

    def test_creates_with_valid_values(self) -> None:
        meld = self._make()
        self.assertEqual(len(meld.tiles), 4)

    def test_rejects_wrong_tile_count(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(PINZU_5, PINZU_5, PINZU_5))

    def test_rejects_different_base_tile_kind(self) -> None:
        with self.assertRaises(ValueError):
            self._make(tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_6))

    def test_keeps_original_pons_from_seat_and_called_tile(self) -> None:
        # kakanだからといってfrom_seat/called_tileをNoneにしない。元Ponの値を
        # そのまま維持する。
        meld = self._make(from_seat=Seat.SEAT_2, called_tile=PINZU_5)
        self.assertEqual(meld.from_seat, Seat.SEAT_2)
        self.assertEqual(meld.called_tile, PINZU_5)

    def test_requires_from_seat(self) -> None:
        with self.assertRaises(ValueError):
            self._make(from_seat=None)

    def test_requires_called_tile(self) -> None:
        with self.assertRaises(ValueError):
            self._make(called_tile=None)

    def test_has_no_source_meld_reference_fields(self) -> None:
        field_names = {field.name for field in fields(PublicMeld)}
        for forbidden in ("source_meld_id", "source_meld_index", "pon"):
            self.assertNotIn(forbidden, field_names)


if __name__ == "__main__":
    unittest.main()
