"""Issue #305: リーチ宣言牌の位置を保持するPolicyInputの保存形式（v2）のtest。"""

import json
import unittest
from dataclasses import replace

from learning_fixtures import policy_input as fixture_policy_input

from lisjong.learning._typed_values import (
    parse_policy_input,
    parse_policy_input_v2,
    policy_input_to_value,
    policy_input_to_value_v2,
)
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))
FIELD = "riichi_declaration_discard_order"


class _ValueError(Exception):
    pass


def _inputs():
    """(値のない入力, seat 1が宣言牌位置7を持つ入力)。"""
    base = fixture_policy_input()
    players = list(base.players)
    discards = (
        Discard(tile=MANZU_4, tsumogiri=False, order=3, called_by=None),
        Discard(tile=MANZU_5, tsumogiri=True, order=7, called_by=None),
    )
    players[1] = replace(players[1], discards=discards, riichi=RiichiState.ACCEPTED)
    without = replace(base, players=tuple(players))
    players[1] = replace(players[1], riichi_declaration_discard_order=7)
    return without, replace(base, players=tuple(players))


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class PolicyInputValueV2Test(unittest.TestCase):
    def test_round_trip_keeps_the_declaration_order(self) -> None:
        _without, with_order = _inputs()

        value = policy_input_to_value_v2(with_order, _ValueError, "input")
        parsed = parse_policy_input_v2(value, _ValueError, "input")

        self.assertEqual(value["players"][1][FIELD], 7)
        self.assertEqual(parsed, with_order)
        self.assertEqual(parsed.players[1].riichi_declaration_discard_order, 7)
        self.assertEqual(
            _bytes(policy_input_to_value_v2(parsed, _ValueError, "input")),
            _bytes(value),
        )

    def test_every_player_carries_the_field_even_when_none(self) -> None:
        without, _with_order = _inputs()

        value = policy_input_to_value_v2(without, _ValueError, "input")

        for player in value["players"]:
            self.assertEqual(
                set(player), {"discards", "melds", "riichi", "score", FIELD}
            )
            self.assertIsNone(player[FIELD])
        self.assertEqual(parse_policy_input_v2(value, _ValueError, "input"), without)

    def test_other_fields_equal_the_v1_projection(self) -> None:
        without, with_order = _inputs()
        v1 = policy_input_to_value(without, _ValueError, "input")

        v2 = policy_input_to_value_v2(with_order, _ValueError, "input")

        for player in v2["players"]:
            del player[FIELD]
        self.assertEqual(v2, v1)

    def test_v2_parser_rejects_a_v1_value(self) -> None:
        without, _with_order = _inputs()
        v1 = policy_input_to_value(without, _ValueError, "input")

        with self.assertRaises(_ValueError):
            parse_policy_input_v2(v1, _ValueError, "input")

    def test_v1_parser_rejects_a_v2_value(self) -> None:
        without, _with_order = _inputs()
        v2 = policy_input_to_value_v2(without, _ValueError, "input")

        with self.assertRaises(_ValueError):
            parse_policy_input(v2, _ValueError, "input")

    def test_v1_projection_still_rejects_an_input_with_the_value(self) -> None:
        _without, with_order = _inputs()

        with self.assertRaises(_ValueError):
            policy_input_to_value(with_order, _ValueError, "input")

    def test_parser_rejects_non_integer_orders(self) -> None:
        _without, with_order = _inputs()
        for bad in (True, 7.0, "7"):
            with self.subTest(bad=bad):
                value = policy_input_to_value_v2(with_order, _ValueError, "input")
                value["players"][1][FIELD] = bad
                with self.assertRaises(_ValueError):
                    parse_policy_input_v2(value, _ValueError, "input")

    def test_parser_rejects_values_the_contract_rejects(self) -> None:
        _without, with_order = _inputs()

        missing_discard = policy_input_to_value_v2(with_order, _ValueError, "input")
        missing_discard["players"][1][FIELD] = 5
        with self.assertRaises(_ValueError):
            parse_policy_input_v2(missing_discard, _ValueError, "input")

        not_in_riichi = policy_input_to_value_v2(with_order, _ValueError, "input")
        not_in_riichi["players"][1]["riichi"] = "none"
        with self.assertRaises(_ValueError):
            parse_policy_input_v2(not_in_riichi, _ValueError, "input")

    def test_parser_rejects_unknown_player_fields(self) -> None:
        _without, with_order = _inputs()
        value = policy_input_to_value_v2(with_order, _ValueError, "input")
        value["players"][0]["extra"] = 1

        with self.assertRaises(_ValueError):
            parse_policy_input_v2(value, _ValueError, "input")

    def test_projection_rejects_non_policy_input(self) -> None:
        with self.assertRaises(_ValueError):
            policy_input_to_value_v2(object(), _ValueError, "input")

    def test_projection_does_not_mutate_the_input(self) -> None:
        _without, with_order = _inputs()
        before = with_order

        policy_input_to_value_v2(with_order, _ValueError, "input")

        self.assertEqual(with_order, before)
        self.assertEqual(with_order.players[1].riichi_declaration_discard_order, 7)


if __name__ == "__main__":
    unittest.main()
