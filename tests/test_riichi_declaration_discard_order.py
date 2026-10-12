"""Issue #301: PlayerPublicStateのリーチ宣言牌位置の契約test。"""

import unittest
from dataclasses import replace

from learning_fixtures import policy_input as fixture_policy_input

from lisjong.learning._typed_values import parse_policy_input, policy_input_to_value
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))

# 局内の通し番号は3と7で、このseatの0打目・1打目とは一致しない。
FIRST = Discard(tile=MANZU_4, tsumogiri=False, order=3, called_by=None)
SECOND = Discard(tile=MANZU_5, tsumogiri=True, order=7, called_by=None)


def _state(riichi, order, discards=(FIRST, SECOND)):
    return PlayerPublicState(
        score=25000,
        discards=discards,
        melds=(),
        riichi=riichi,
        riichi_declaration_discard_order=order,
    )


class _ValueError(Exception):
    pass


class RiichiDeclarationDiscardOrderTest(unittest.TestCase):
    def test_defaults_to_none(self) -> None:
        state = PlayerPublicState(
            score=25000, discards=(FIRST,), melds=(), riichi=RiichiState.NONE
        )
        self.assertIsNone(state.riichi_declaration_discard_order)

    def test_allowed_combinations(self) -> None:
        for riichi, order in (
            (RiichiState.NONE, None),
            (RiichiState.DECLARED, None),
            (RiichiState.DECLARED, 7),
            (RiichiState.ACCEPTED, 3),
            (RiichiState.ACCEPTED, None),
        ):
            with self.subTest(riichi=riichi, order=order):
                state = _state(riichi, order)
                self.assertEqual(state.riichi_declaration_discard_order, order)

    def test_value_refers_to_the_round_wide_discard_order(self) -> None:
        # 1はこのseatの2打目の位置だが、そのorderを持つ打牌はない。
        with self.assertRaises(ValueError):
            _state(RiichiState.ACCEPTED, 1)
        self.assertEqual(
            _state(RiichiState.ACCEPTED, 7).riichi_declaration_discard_order, 7
        )

    def test_rejects_value_when_riichi_is_none(self) -> None:
        with self.assertRaises(ValueError):
            _state(RiichiState.NONE, 3)

    def test_rejects_value_without_a_matching_discard(self) -> None:
        with self.assertRaises(ValueError):
            _state(RiichiState.ACCEPTED, 5)
        with self.assertRaises(ValueError):
            _state(RiichiState.DECLARED, 0, discards=())

    def test_rejects_non_integer_values(self) -> None:
        for value in (True, 3.0, "3"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    _state(RiichiState.ACCEPTED, value)

    def test_does_not_validate_tsumogiri_after_the_declaration(self) -> None:
        # 宣言牌より後の手出しは、Adapterの整合検証の責務である。
        later_hand_discard = Discard(
            tile=MANZU_5, tsumogiri=False, order=7, called_by=None
        )
        state = _state(RiichiState.ACCEPTED, 3, discards=(FIRST, later_hand_discard))
        self.assertEqual(state.riichi_declaration_discard_order, 3)

    def test_value_takes_part_in_equality_and_hash(self) -> None:
        with_order = _state(RiichiState.ACCEPTED, 3)
        self.assertEqual(with_order, _state(RiichiState.ACCEPTED, 3))
        self.assertEqual(hash(with_order), hash(_state(RiichiState.ACCEPTED, 3)))
        self.assertNotEqual(with_order, _state(RiichiState.ACCEPTED, None))
        self.assertNotEqual(with_order, _state(RiichiState.ACCEPTED, 7))


class SourceRecordWireShapeTest(unittest.TestCase):
    """v1のwire shapeは値を持たない。既存versionのbyte列を変えない。"""

    def _inputs(self):
        policy_input = fixture_policy_input()
        seat = (int(policy_input.self_seat) + 1) % 4
        players = list(policy_input.players)
        discard = Discard(tile=MANZU_4, tsumogiri=False, order=0, called_by=None)
        base = replace(players[seat], discards=(discard,), riichi=RiichiState.ACCEPTED)
        players[seat] = base
        without = replace(policy_input, players=tuple(players))
        players[seat] = replace(base, riichi_declaration_discard_order=0)
        return without, replace(policy_input, players=tuple(players)), seat

    def test_projection_is_unchanged_by_the_value(self) -> None:
        without, with_order, seat = self._inputs()

        projected = policy_input_to_value(with_order, _ValueError, "input")

        self.assertEqual(
            projected, policy_input_to_value(without, _ValueError, "input")
        )
        self.assertEqual(
            set(projected["players"][seat]), {"discards", "melds", "riichi", "score"}
        )

    def test_parsed_policy_input_has_no_declaration_order(self) -> None:
        without, with_order, seat = self._inputs()

        parsed = parse_policy_input(
            policy_input_to_value(with_order, _ValueError, "input"),
            _ValueError,
            "input",
        )

        self.assertIsNone(parsed.players[seat].riichi_declaration_discard_order)
        self.assertEqual(parsed, without)


if __name__ == "__main__":
    unittest.main()
