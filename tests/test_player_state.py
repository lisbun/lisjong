import unittest
from dataclasses import FrozenInstanceError, fields

from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))
MANZU_6 = Tile(TileType(TileCategory.MANZU, 6))
PINZU_5 = Tile(TileType(TileCategory.PINZU, 5))

DISCARD_0 = Discard(tile=MANZU_4, tsumogiri=False, order=0, called_by=None)
DISCARD_1 = Discard(tile=MANZU_5, tsumogiri=True, order=1, called_by=None)
DISCARD_2 = Discard(tile=MANZU_6, tsumogiri=False, order=2, called_by=Seat.SEAT_1)

CHI_MELD = PublicMeld(
    kind=MeldKind.CHI,
    tiles=(MANZU_4, MANZU_5, MANZU_6),
    from_seat=Seat.SEAT_0,
    called_tile=MANZU_5,
)
ANKAN_MELD = PublicMeld(
    kind=MeldKind.ANKAN,
    tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5),
    from_seat=None,
    called_tile=None,
)


class PlayerPublicStateTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        state = PlayerPublicState(
            score=25000,
            discards=(DISCARD_0, DISCARD_1),
            melds=(CHI_MELD,),
            riichi=RiichiState.NONE,
        )
        self.assertEqual(state.score, 25000)
        self.assertEqual(state.discards, (DISCARD_0, DISCARD_1))
        self.assertEqual(state.melds, (CHI_MELD,))
        self.assertEqual(state.riichi, RiichiState.NONE)

    def test_allows_empty_discards_and_melds(self) -> None:
        state = PlayerPublicState(
            score=25000, discards=(), melds=(), riichi=RiichiState.NONE
        )
        self.assertEqual(state.discards, ())
        self.assertEqual(state.melds, ())

    def test_is_immutable(self) -> None:
        state = PlayerPublicState(
            score=25000, discards=(), melds=(), riichi=RiichiState.NONE
        )
        with self.assertRaises(FrozenInstanceError):
            state.score = 24000

    def test_equal_values_compare_equal_and_are_hashable(self) -> None:
        first = PlayerPublicState(
            score=25000,
            discards=(DISCARD_0,),
            melds=(CHI_MELD,),
            riichi=RiichiState.DECLARED,
        )
        second = PlayerPublicState(
            score=25000,
            discards=(DISCARD_0,),
            melds=(CHI_MELD,),
            riichi=RiichiState.DECLARED,
        )
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)

    def test_rejects_non_integer_score(self) -> None:
        with self.assertRaises(TypeError):
            PlayerPublicState(
                score="25000", discards=(), melds=(), riichi=RiichiState.NONE
            )

    def test_rejects_boolean_score(self) -> None:
        with self.assertRaises(TypeError):
            PlayerPublicState(
                score=True, discards=(), melds=(), riichi=RiichiState.NONE
            )

    def test_allows_negative_score(self) -> None:
        state = PlayerPublicState(
            score=-3000, discards=(), melds=(), riichi=RiichiState.NONE
        )
        self.assertEqual(state.score, -3000)

    def test_rejects_non_discard_in_discards(self) -> None:
        with self.assertRaises(TypeError):
            PlayerPublicState(
                score=25000,
                discards=(DISCARD_0, "not a discard"),
                melds=(),
                riichi=RiichiState.NONE,
            )

    def test_rejects_non_public_meld_in_melds(self) -> None:
        with self.assertRaises(TypeError):
            PlayerPublicState(
                score=25000,
                discards=(),
                melds=(CHI_MELD, "not a meld"),
                riichi=RiichiState.NONE,
            )

    def test_rejects_non_riichi_state(self) -> None:
        with self.assertRaises(TypeError):
            PlayerPublicState(score=25000, discards=(), melds=(), riichi=True)

    def test_normalizes_list_input_into_tuples(self) -> None:
        state = PlayerPublicState(
            score=25000,
            discards=[DISCARD_0, DISCARD_1],
            melds=[CHI_MELD],
            riichi=RiichiState.NONE,
        )
        self.assertIsInstance(state.discards, tuple)
        self.assertIsInstance(state.melds, tuple)
        self.assertEqual(state.discards, (DISCARD_0, DISCARD_1))
        self.assertEqual(state.melds, (CHI_MELD,))

    def test_does_not_reorder_discards_by_order_field(self) -> None:
        # orderと逆順・不連続な順で渡しても、勝手にsortしない。
        state = PlayerPublicState(
            score=25000,
            discards=(DISCARD_2, DISCARD_0, DISCARD_1),
            melds=(),
            riichi=RiichiState.NONE,
        )
        self.assertEqual(state.discards, (DISCARD_2, DISCARD_0, DISCARD_1))

    def test_does_not_reorder_melds(self) -> None:
        state = PlayerPublicState(
            score=25000,
            discards=(),
            melds=(ANKAN_MELD, CHI_MELD),
            riichi=RiichiState.NONE,
        )
        self.assertEqual(state.melds, (ANKAN_MELD, CHI_MELD))

    def test_has_no_hidden_or_owner_fields(self) -> None:
        field_names = {field.name for field in fields(PlayerPublicState)}
        self.assertEqual(field_names, {"score", "discards", "melds", "riichi"})
        for forbidden in (
            "seat",
            "concealed_tiles",
            "concealed_hand",
            "drawn_tile",
            "waits",
            "shanten",
            "is_tenpai",
            "furiten",
            "ura_dora",
            "wall",
        ):
            self.assertNotIn(forbidden, field_names)


if __name__ == "__main__":
    unittest.main()
