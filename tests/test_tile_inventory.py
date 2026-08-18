import unittest

from lisjong.belief.tile_inventory import (
    BASE_TILE_COUNT_MAX,
    RED_FIVE_AXIS_COUNT,
    RED_FIVE_COUNT_MAX,
    STANDARD_RED_FIVE_COUNTS,
    STANDARD_TILE_COUNTS,
    TILE_TYPE_COUNT,
    TOTAL_PHYSICAL_TILE_COUNT,
)


class TileInventoryTest(unittest.TestCase):
    def test_standard_tile_counts_are_four_for_all_34_types(self) -> None:
        self.assertEqual(len(STANDARD_TILE_COUNTS), TILE_TYPE_COUNT)
        self.assertEqual(TILE_TYPE_COUNT, 34)
        self.assertTrue(all(count == 4 for count in STANDARD_TILE_COUNTS))

    def test_standard_red_five_counts_are_one_per_suit(self) -> None:
        self.assertEqual(len(STANDARD_RED_FIVE_COUNTS), RED_FIVE_AXIS_COUNT)
        self.assertEqual(RED_FIVE_AXIS_COUNT, 3)
        self.assertEqual(STANDARD_RED_FIVE_COUNTS, (1, 1, 1))

    def test_total_physical_tile_count_is_136(self) -> None:
        self.assertEqual(TOTAL_PHYSICAL_TILE_COUNT, 136)
        self.assertEqual(sum(STANDARD_TILE_COUNTS), TOTAL_PHYSICAL_TILE_COUNT)

    def test_max_constants_match_standard_inventory(self) -> None:
        self.assertEqual(BASE_TILE_COUNT_MAX, 4)
        self.assertEqual(RED_FIVE_COUNT_MAX, 1)


if __name__ == "__main__":
    unittest.main()
