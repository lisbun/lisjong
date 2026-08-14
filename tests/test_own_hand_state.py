import unittest
from dataclasses import FrozenInstanceError, fields

from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))
MANZU_5_RED = Tile(TileType(TileCategory.MANZU, 5), is_red=True)
PINZU_5 = Tile(TileType(TileCategory.PINZU, 5))


class OwnHandStateTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        state = OwnHandState(
            concealed_tiles=(MANZU_4, MANZU_5, PINZU_5), drawn_tile=MANZU_5
        )
        self.assertEqual(state.concealed_tiles, (MANZU_4, MANZU_5, PINZU_5))
        self.assertEqual(state.drawn_tile, MANZU_5)

    def test_is_immutable(self) -> None:
        state = OwnHandState(concealed_tiles=(MANZU_4,), drawn_tile=None)
        with self.assertRaises(FrozenInstanceError):
            state.drawn_tile = MANZU_4

    def test_equal_values_compare_equal_and_are_hashable(self) -> None:
        first = OwnHandState(
            concealed_tiles=(MANZU_5_RED, MANZU_4), drawn_tile=MANZU_5_RED
        )
        second = OwnHandState(
            concealed_tiles=(MANZU_4, MANZU_5_RED), drawn_tile=MANZU_5_RED
        )
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)

    def test_normalizes_list_input_into_tuple(self) -> None:
        state = OwnHandState(concealed_tiles=[MANZU_4, MANZU_5], drawn_tile=None)
        self.assertIsInstance(state.concealed_tiles, tuple)

    def test_input_order_does_not_affect_identity(self) -> None:
        first = OwnHandState(
            concealed_tiles=(PINZU_5, MANZU_4, MANZU_5), drawn_tile=MANZU_5
        )
        second = OwnHandState(
            concealed_tiles=(MANZU_5, MANZU_4, PINZU_5), drawn_tile=MANZU_5
        )
        self.assertEqual(first, second)
        self.assertEqual(first.concealed_tiles, second.concealed_tiles)

    def test_preserves_duplicate_semantic_tiles(self) -> None:
        state = OwnHandState(
            concealed_tiles=(MANZU_5, MANZU_5, MANZU_5), drawn_tile=MANZU_5
        )
        self.assertEqual(state.concealed_tiles.count(MANZU_5), 3)

    def test_preserves_red_distinction(self) -> None:
        state = OwnHandState(
            concealed_tiles=(MANZU_5, MANZU_5_RED), drawn_tile=MANZU_5_RED
        )
        self.assertIn(MANZU_5, state.concealed_tiles)
        self.assertIn(MANZU_5_RED, state.concealed_tiles)
        self.assertNotEqual(MANZU_5, MANZU_5_RED)

    def test_allows_drawn_tile_none(self) -> None:
        state = OwnHandState(concealed_tiles=(MANZU_4, MANZU_5), drawn_tile=None)
        self.assertIsNone(state.drawn_tile)

    def test_allows_empty_concealed_tiles_with_no_drawn_tile(self) -> None:
        # 非空はOwnHandState単一値の構造的不変条件ではなく、
        # Policy decisionを生成する環境・タイミング側の条件である。
        state = OwnHandState(concealed_tiles=(), drawn_tile=None)
        self.assertEqual(state.concealed_tiles, ())
        self.assertIsNone(state.drawn_tile)

    def test_allows_large_concealed_tiles_without_upper_bound(self) -> None:
        # 13/14枚等の固定枚数制約は課さない。
        tiles = (
            tuple(Tile(TileType(TileCategory.MANZU, rank)) for rank in (1, 2, 3, 4, 5))
            * 3
        )
        state = OwnHandState(concealed_tiles=tiles, drawn_tile=None)
        self.assertEqual(len(state.concealed_tiles), 15)

    def test_rejects_non_iterable_concealed_tiles(self) -> None:
        with self.assertRaises(TypeError):
            OwnHandState(concealed_tiles=123, drawn_tile=None)

    def test_rejects_non_tile_in_concealed_tiles(self) -> None:
        with self.assertRaises(TypeError):
            OwnHandState(concealed_tiles=(MANZU_4, "5m"), drawn_tile=None)

    def test_rejects_non_tile_non_none_drawn_tile(self) -> None:
        with self.assertRaises(TypeError):
            OwnHandState(concealed_tiles=(MANZU_4,), drawn_tile="4m")

    def test_rejects_drawn_tile_not_in_concealed_tiles(self) -> None:
        with self.assertRaises(ValueError):
            OwnHandState(concealed_tiles=(MANZU_4, PINZU_5), drawn_tile=MANZU_5)

    def test_rejects_drawn_tile_when_concealed_tiles_is_empty(self) -> None:
        with self.assertRaises(ValueError):
            OwnHandState(concealed_tiles=(), drawn_tile=MANZU_5)

    def test_rejects_mismatched_red_distinction(self) -> None:
        # concealed_tilesに通常5mしかない場合、drawn_tile=赤5mは不整合。
        with self.assertRaises(ValueError):
            OwnHandState(concealed_tiles=(MANZU_5, PINZU_5), drawn_tile=MANZU_5_RED)

    def test_accepts_matching_red_distinction(self) -> None:
        state = OwnHandState(
            concealed_tiles=(MANZU_5_RED, PINZU_5), drawn_tile=MANZU_5_RED
        )
        self.assertEqual(state.drawn_tile, MANZU_5_RED)

    def test_has_no_physical_copy_identity_fields(self) -> None:
        field_names = {field.name for field in fields(OwnHandState)}
        self.assertEqual(field_names, {"concealed_tiles", "drawn_tile"})
        for forbidden in (
            "copy_index",
            "physical_tile_id",
            "drawn_tile_index",
            "tile_index",
            "source_id",
        ):
            self.assertNotIn(forbidden, field_names)


if __name__ == "__main__":
    unittest.main()
