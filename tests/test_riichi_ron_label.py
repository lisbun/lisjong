import unittest

from lisjong.belief.riichi_ron_label import (
    RiichiFuritenReason,
    RiichiRonLabel,
    riichi_ron_label,
)
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

MANZU = TileCategory.MANZU
PINZU = TileCategory.PINZU
SOUZU = TileCategory.SOUZU
HONOR = TileCategory.HONOR


def t(category, rank: int, red: bool = False) -> Tile:
    return Tile(TileType(category, rank), is_red=red)


def tiles(category, *ranks: int) -> tuple[Tile, ...]:
    return tuple(t(category, rank) for rank in ranks)


# 123m 456p 789s 11z + 23m: ryanmen on 1m / 4m
RYANMEN_HAND = (
    tiles(MANZU, 1, 2, 3)
    + tiles(PINZU, 4, 5, 6)
    + tiles(SOUZU, 7, 8, 9)
    + tiles(HONOR, 1, 1)
    + tiles(MANZU, 2, 3)
)
M1 = TileType(MANZU, 1)
M4 = TileType(MANZU, 4)
P5 = TileType(PINZU, 5)


class RiichiRonLabelTest(unittest.TestCase):
    def test_structural_waits_without_furiten(self):
        label = riichi_ron_label(RYANMEN_HAND)
        self.assertEqual(label.wait_tile_types, frozenset({M1, M4}))
        self.assertFalse(label.furiten)
        self.assertTrue(label.can_ron(M1))
        self.assertTrue(label.can_ron(M4))
        self.assertFalse(label.can_ron(TileType(MANZU, 5)))

    def test_own_discard_of_any_wait_blocks_every_wait(self):
        label = riichi_ron_label(RYANMEN_HAND, own_discards=(t(MANZU, 4),))
        self.assertEqual(label.furiten_reasons, {RiichiFuritenReason.OWN_DISCARD})
        self.assertEqual(label.ron_tile_types, frozenset())
        self.assertFalse(label.can_ron(M1))
        self.assertEqual(label.wait_tile_types, frozenset({M1, M4}))

    def test_passed_wait_after_riichi_blocks_every_wait(self):
        label = riichi_ron_label(RYANMEN_HAND, passed_tile_types=(M1,))
        self.assertEqual(
            label.furiten_reasons, {RiichiFuritenReason.PASSED_AFTER_RIICHI}
        )
        self.assertFalse(label.can_ron(M4))

    def test_both_reasons_are_recorded(self):
        label = riichi_ron_label(
            RYANMEN_HAND, own_discards=(M4,), passed_tile_types=(M1,)
        )
        self.assertEqual(
            label.furiten_reasons,
            {
                RiichiFuritenReason.OWN_DISCARD,
                RiichiFuritenReason.PASSED_AFTER_RIICHI,
            },
        )

    def test_non_wait_discards_and_passes_do_not_cause_furiten(self):
        label = riichi_ron_label(
            RYANMEN_HAND,
            own_discards=tiles(SOUZU, 1, 2) + (t(HONOR, 7),),
            passed_tile_types=(TileType(PINZU, 9),),
        )
        self.assertFalse(label.furiten)
        self.assertEqual(label.ron_tile_types, frozenset({M1, M4}))

    def test_red_five_discard_counts_as_its_base_tile_type(self):
        # 123m 456p 789s 11z + 46p: kanchan on 5p
        hand = (
            tiles(MANZU, 1, 2, 3)
            + tiles(PINZU, 4, 5, 6)
            + tiles(SOUZU, 7, 8, 9)
            + tiles(HONOR, 1, 1)
            + tiles(PINZU, 4, 6)
        )
        self.assertTrue(riichi_ron_label(hand).can_ron(P5))
        label = riichi_ron_label(hand, own_discards=(t(PINZU, 5, red=True),))
        self.assertEqual(label.furiten_reasons, {RiichiFuritenReason.OWN_DISCARD})

    def test_chiitoitsu_tanki_wait(self):
        hand = (
            tiles(MANZU, 1, 1, 2, 2)
            + tiles(PINZU, 3, 3, 4, 4)
            + tiles(SOUZU, 5, 5, 6, 6)
            + tiles(HONOR, 7)
        )
        label = riichi_ron_label(hand)
        self.assertTrue(label.can_ron(TileType(HONOR, 7)))

    def test_non_tenpai_hand_fails_closed(self):
        hand = (
            tiles(MANZU, 1, 3, 5, 7, 9)
            + tiles(PINZU, 1, 3, 5, 7, 9)
            + tiles(SOUZU, 1, 3, 5)
        )
        with self.assertRaises(ValueError):
            riichi_ron_label(hand)

    def test_fourteen_tile_hand_fails_closed(self):
        with self.assertRaises(ValueError):
            riichi_ron_label(RYANMEN_HAND + (t(MANZU, 9),))

    def test_type_checks(self):
        with self.assertRaises(TypeError):
            riichi_ron_label(RYANMEN_HAND, own_discards=("1m",))
        with self.assertRaises(TypeError):
            riichi_ron_label(RYANMEN_HAND).can_ron("1m")
        with self.assertRaises(ValueError):
            RiichiRonLabel(wait_tile_types=frozenset(), furiten_reasons=frozenset())


if __name__ == "__main__":
    unittest.main()
