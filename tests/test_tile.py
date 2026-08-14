import unittest
from dataclasses import FrozenInstanceError, fields

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
    tile_sort_key,
)


class TileCategoryTest(unittest.TestCase):
    def test_has_four_categories(self) -> None:
        self.assertEqual(
            {category.value for category in TileCategory},
            {"manzu", "pinzu", "souzu", "honor"},
        )


class TileTypeTest(unittest.TestCase):
    def test_accepts_boundary_ranks(self) -> None:
        cases = (
            (TileCategory.MANZU, 1),
            (TileCategory.MANZU, 9),
            (TileCategory.PINZU, 1),
            (TileCategory.PINZU, 9),
            (TileCategory.SOUZU, 1),
            (TileCategory.SOUZU, 9),
            (TileCategory.HONOR, 1),
            (TileCategory.HONOR, 7),
        )
        for category, rank in cases:
            with self.subTest(category=category, rank=rank):
                self.assertEqual(TileType(category, rank).rank, rank)

    def test_rejects_numbered_tile_rank_out_of_range(self) -> None:
        for rank in (0, 10):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                TileType(TileCategory.MANZU, rank)

    def test_rejects_honor_rank_out_of_range(self) -> None:
        for rank in (0, 8):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                TileType(TileCategory.HONOR, rank)

    def test_rejects_non_integer_rank(self) -> None:
        for rank in ("1", 1.0, True):
            with self.subTest(rank=rank), self.assertRaises(TypeError):
                TileType(TileCategory.MANZU, rank)

    def test_rejects_non_category(self) -> None:
        with self.assertRaises(TypeError):
            TileType("manzu", 1)

    def test_is_immutable(self) -> None:
        tile_type = TileType(TileCategory.MANZU, 1)
        with self.assertRaises(FrozenInstanceError):
            tile_type.rank = 2

    def test_equal_values_compare_equal_and_are_hashable(self) -> None:
        first = TileType(TileCategory.PINZU, 5)
        second = TileType(TileCategory.PINZU, 5)
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)

    def test_has_no_physical_copy_field(self) -> None:
        field_names = {field.name for field in fields(TileType)}
        self.assertEqual(field_names, {"category", "rank"})


class HonorTileTypeMeaningTest(unittest.TestCase):
    """rank 1..7が指す具体的な字牌を、lisjongの明示的な設計判断として固定する。"""

    def test_named_honor_constants_have_expected_ranks(self) -> None:
        expected_rank_by_constant = {
            EAST_WIND: 1,
            SOUTH_WIND: 2,
            WEST_WIND: 3,
            NORTH_WIND: 4,
            WHITE_DRAGON: 5,
            GREEN_DRAGON: 6,
            RED_DRAGON: 7,
        }
        for constant, expected_rank in expected_rank_by_constant.items():
            with self.subTest(constant=constant):
                self.assertEqual(constant.category, TileCategory.HONOR)
                self.assertEqual(constant.rank, expected_rank)

    def test_named_honor_constants_are_seven_distinct_values(self) -> None:
        constants = (
            EAST_WIND,
            SOUTH_WIND,
            WEST_WIND,
            NORTH_WIND,
            WHITE_DRAGON,
            GREEN_DRAGON,
            RED_DRAGON,
        )
        self.assertEqual(len(set(constants)), 7)
        self.assertEqual(
            {rank for rank in range(1, 8)},
            {constant.rank for constant in constants},
        )


class TileTest(unittest.TestCase):
    def test_rejects_non_tile_type(self) -> None:
        with self.assertRaises(TypeError):
            Tile("1m")

    def test_rejects_non_boolean_red_flag(self) -> None:
        tile_type = TileType(TileCategory.MANZU, 5)
        with self.assertRaises(TypeError):
            Tile(tile_type, is_red=1)

    def test_allows_only_suited_fives_to_be_red(self) -> None:
        for category in (
            TileCategory.MANZU,
            TileCategory.PINZU,
            TileCategory.SOUZU,
        ):
            with self.subTest(category=category):
                self.assertTrue(Tile(TileType(category, 5), is_red=True).is_red)

        invalid_types = (
            TileType(TileCategory.MANZU, 4),
            TileType(TileCategory.HONOR, 5),
        )
        for tile_type in invalid_types:
            with (
                self.subTest(tile_type=tile_type),
                self.assertRaises(ValueError),
            ):
                Tile(tile_type, is_red=True)

    def test_is_immutable(self) -> None:
        tile = Tile(TileType(TileCategory.MANZU, 1))
        with self.assertRaises(FrozenInstanceError):
            tile.is_red = True

    def test_normal_and_red_five_are_different_values(self) -> None:
        tile_type = TileType(TileCategory.PINZU, 5)
        normal_five = Tile(tile_type)
        red_five = Tile(tile_type, is_red=True)
        self.assertNotEqual(normal_five, red_five)
        self.assertEqual(len({normal_five, red_five}), 2)

    def test_tiles_with_same_meaning_are_equal(self) -> None:
        first = Tile(TileType(TileCategory.SOUZU, 3))
        second = Tile(TileType(TileCategory.SOUZU, 3))
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)

    def test_default_is_red_is_false(self) -> None:
        self.assertFalse(Tile(TileType(TileCategory.MANZU, 1)).is_red)

    def test_has_no_physical_copy_field(self) -> None:
        field_names = {field.name for field in fields(Tile)}
        self.assertEqual(field_names, {"tile_type", "is_red"})
        self.assertNotIn("copy_index", field_names)


class TileSortKeyTest(unittest.TestCase):
    def test_rejects_non_tile(self) -> None:
        with self.assertRaises(TypeError):
            tile_sort_key(TileType(TileCategory.MANZU, 1))

    def test_orders_by_category_then_rank_then_red(self) -> None:
        tiles = [
            Tile(TileType(TileCategory.HONOR, 1)),
            Tile(TileType(TileCategory.SOUZU, 9)),
            Tile(TileType(TileCategory.PINZU, 5), is_red=True),
            Tile(TileType(TileCategory.PINZU, 5)),
            Tile(TileType(TileCategory.MANZU, 1)),
        ]
        expected_order = [
            Tile(TileType(TileCategory.MANZU, 1)),
            Tile(TileType(TileCategory.PINZU, 5)),
            Tile(TileType(TileCategory.PINZU, 5), is_red=True),
            Tile(TileType(TileCategory.SOUZU, 9)),
            Tile(TileType(TileCategory.HONOR, 1)),
        ]
        self.assertEqual(sorted(tiles, key=tile_sort_key), expected_order)

    def test_does_not_depend_on_category_enum_string_order(self) -> None:
        # TileCategoryのEnum定義順 (MANZU, PINZU, SOUZU, HONOR) は
        # アルファベット順 (HONOR, MANZU, PINZU, SOUZU) とは異なる。
        # tile_sort_keyが後者に偶然依存していないことを確認する。
        alphabetical_by_value = sorted(TileCategory, key=lambda c: c.value)
        self.assertNotEqual(
            alphabetical_by_value,
            [
                TileCategory.MANZU,
                TileCategory.PINZU,
                TileCategory.SOUZU,
                TileCategory.HONOR,
            ],
        )

        manzu_tile = Tile(TileType(TileCategory.MANZU, 1))
        honor_tile = Tile(TileType(TileCategory.HONOR, 1))
        self.assertLess(tile_sort_key(manzu_tile), tile_sort_key(honor_tile))


if __name__ == "__main__":
    unittest.main()
