"""Issue #302: リーチ者ごとの確定安全牌の基盤test。"""

import unittest
from dataclasses import replace

from learning_fixtures import HONOR, MANZU, PINZU, SOUZU, discard, hand, tile
from learning_fixtures import policy_input as fixture_policy_input

from lisjong.belief.riichi_safe_tiles import (
    RIICHI_SAFE_TILE_RULE_ASSUMPTION,
    derive_riichi_safe_tiles,
)
from lisjong.candidate_boundary import check_candidate_imports
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat

# 自席（seat 0）の打牌判断。手牌13枚 + ツモ1枚。
DRAWN = tile(HONOR, 7)
DISCARD_HAND = (*hand(0), DRAWN)

BEFORE = tile(SOUZU, 1)
DECLARATION = tile(SOUZU, 2)
AFTER_OWN_RIVER = tile(SOUZU, 3)
SEAT2_BEFORE = tile(PINZU, 7)
SEAT2_AFTER = tile(PINZU, 8)
SEAT3_AFTER_CALLED = tile(PINZU, 9)
SELF_BEFORE = tile(HONOR, 5)
SELF_AFTER = tile(HONOR, 6)


def _player(discards=(), *, riichi=RiichiState.NONE, declaration=None, melds=()):
    return PlayerPublicState(
        score=25000,
        discards=tuple(discards),
        melds=tuple(melds),
        riichi=riichi,
        riichi_declaration_discard_order=declaration,
    )


def _input(players, *, concealed=DISCARD_HAND, drawn=DRAWN):
    return fixture_policy_input(
        self_seat=Seat.SEAT_0, concealed=concealed, drawn=drawn, players=players
    )


def _types(*tiles):
    return frozenset(item.tile_type for item in tiles)


def _single_riichi_players(*, riichi=RiichiState.ACCEPTED, declaration=4):
    """seat 1がorder 4でリーチを宣言した局面。"""
    return (
        _player((discard(SELF_BEFORE, 0), discard(SELF_AFTER, 7))),
        _player(
            (
                discard(BEFORE, 1),
                discard(DECLARATION, 4),
                discard(AFTER_OWN_RIVER, 8, tsumogiri=True),
            ),
            riichi=riichi,
            declaration=declaration,
        ),
        _player((discard(SEAT2_BEFORE, 2), discard(SEAT2_AFTER, 5))),
        _player(
            (
                discard(tile(HONOR, 1), 3),
                discard(SEAT3_AFTER_CALLED, 6, called_by=Seat.SEAT_0),
            )
        ),
    )


class SingleRiichiTest(unittest.TestCase):
    def test_no_riichi_player_yields_no_safe_tiles(self) -> None:
        players = tuple(_player((discard(tile(HONOR, 1 + i), i),)) for i in range(4))

        result = derive_riichi_safe_tiles(_input(players))

        self.assertEqual(result.by_riichi_player, ())
        self.assertEqual(result.safe_against_all_riichi_players, frozenset())
        self.assertTrue(result.all_declarations_known)

    def test_river_and_tiles_passed_after_the_declaration(self) -> None:
        result = derive_riichi_safe_tiles(_input(_single_riichi_players()))

        (riichi,) = result.by_riichi_player
        self.assertEqual(riichi.riichi_seat, Seat.SEAT_1)
        self.assertTrue(riichi.declaration_known)
        # 宣言牌と、宣言の前後の自分の打牌はすべて河として入る。
        self.assertEqual(
            riichi.river_tile_types, _types(BEFORE, DECLARATION, AFTER_OWN_RIVER)
        )
        # 宣言牌より後の他席の打牌（鳴かれた牌・自席の打牌を含む）だけが入る。
        self.assertEqual(
            riichi.passed_tile_types,
            _types(SEAT2_AFTER, SEAT3_AFTER_CALLED, SELF_AFTER),
        )
        self.assertEqual(
            riichi.safe_tile_types,
            riichi.river_tile_types | riichi.passed_tile_types,
        )
        self.assertEqual(result.safe_against_all_riichi_players, riichi.safe_tile_types)

    def test_tiles_discarded_before_the_declaration_are_not_passed(self) -> None:
        result = derive_riichi_safe_tiles(_input(_single_riichi_players()))

        safe = result.safe_against_all_riichi_players
        for item in (SEAT2_BEFORE, SELF_BEFORE, tile(HONOR, 1)):
            self.assertNotIn(item.tile_type, safe)

    def test_order_alone_decides_even_when_a_turn_was_skipped_by_a_call(self) -> None:
        # seat 3の打牌をseat 0が鳴き、seat 1・seat 2の手番が飛んだ並び。
        players = list(_single_riichi_players())
        players[2] = _player((discard(SEAT2_BEFORE, 2),))
        result = derive_riichi_safe_tiles(_input(tuple(players)))

        (riichi,) = result.by_riichi_player
        self.assertEqual(
            riichi.passed_tile_types, _types(SEAT3_AFTER_CALLED, SELF_AFTER)
        )

    def test_red_five_and_plain_five_are_the_same_tile_type(self) -> None:
        players = list(_single_riichi_players())
        players[2] = _player((discard(tile(MANZU, 5, red=True), 5),))
        result = derive_riichi_safe_tiles(_input(tuple(players)))

        self.assertIn(tile(MANZU, 5).tile_type, result.safe_against_all_riichi_players)

    def test_kan_tiles_are_not_passed(self) -> None:
        kan_tile = tile(PINZU, 1)
        ankan = PublicMeld(
            kind=MeldKind.ANKAN,
            tiles=(kan_tile, kan_tile, kan_tile, kan_tile),
            from_seat=None,
            called_tile=None,
        )
        players = list(_single_riichi_players())
        players[2] = replace(players[2], melds=(ankan,))
        result = derive_riichi_safe_tiles(_input(tuple(players)))

        self.assertNotIn(kan_tile.tile_type, result.safe_against_all_riichi_players)

    def test_declared_riichi_right_after_the_declaration_has_nothing_passed(
        self,
    ) -> None:
        players = (
            _player((discard(SELF_BEFORE, 0),)),
            _player(
                (discard(BEFORE, 1), discard(DECLARATION, 3)),
                riichi=RiichiState.DECLARED,
                declaration=3,
            ),
            _player((discard(SEAT2_BEFORE, 2),)),
            _player(()),
        )
        result = derive_riichi_safe_tiles(_input(players))

        (riichi,) = result.by_riichi_player
        self.assertTrue(riichi.declaration_known)
        self.assertEqual(riichi.passed_tile_types, frozenset())
        self.assertEqual(riichi.safe_tile_types, _types(BEFORE, DECLARATION))


class MissingDeclarationTest(unittest.TestCase):
    def test_unknown_declaration_uses_the_river_only(self) -> None:
        for riichi in (RiichiState.ACCEPTED, RiichiState.DECLARED):
            with self.subTest(riichi=riichi):
                players = _single_riichi_players(riichi=riichi, declaration=None)
                result = derive_riichi_safe_tiles(_input(players))

                (player,) = result.by_riichi_player
                self.assertFalse(player.declaration_known)
                self.assertFalse(result.all_declarations_known)
                self.assertEqual(player.passed_tile_types, frozenset())
                # 単独リーチで位置が不明なら、現行の共通現物（河）と同じ集合になる。
                self.assertEqual(
                    result.safe_against_all_riichi_players,
                    _types(BEFORE, DECLARATION, AFTER_OWN_RIVER),
                )


class MultipleRiichiTest(unittest.TestCase):
    def _players(self, *, seat3_declaration=9):
        shared_late = tile(HONOR, 2)
        return (
            _player((discard(SELF_BEFORE, 0), discard(SELF_AFTER, 7))),
            _player(
                (discard(BEFORE, 1), discard(DECLARATION, 4)),
                riichi=RiichiState.ACCEPTED,
                declaration=4,
            ),
            _player((discard(SEAT2_AFTER, 5), discard(shared_late, 10))),
            _player(
                (discard(tile(HONOR, 1), 3), discard(SEAT3_AFTER_CALLED, 9)),
                riichi=RiichiState.ACCEPTED,
                declaration=seat3_declaration,
            ),
        ), shared_late

    def test_sets_are_derived_per_riichi_player(self) -> None:
        players, shared_late = self._players()
        result = derive_riichi_safe_tiles(_input(players))

        first, second = result.by_riichi_player
        self.assertEqual(
            (first.riichi_seat, second.riichi_seat), (Seat.SEAT_1, Seat.SEAT_3)
        )
        # seat 1の宣言（order 4）より後：5, 7, 9, 10
        self.assertEqual(
            first.passed_tile_types,
            _types(SEAT2_AFTER, SELF_AFTER, SEAT3_AFTER_CALLED, shared_late),
        )
        # seat 3の宣言（order 9）より後：10だけ
        self.assertEqual(second.passed_tile_types, _types(shared_late))

    def test_tile_passed_against_only_one_player_is_not_safe_against_all(self) -> None:
        players, shared_late = self._players()
        result = derive_riichi_safe_tiles(_input(players))

        safe = result.safe_against_all_riichi_players
        self.assertIn(shared_late.tile_type, safe)
        # seat 2の8pはseat 1にだけ通っている。
        self.assertNotIn(SEAT2_AFTER.tile_type, safe)
        # seat 3の宣言牌は、seat 3の河であり、seat 1には宣言後に通っている。
        self.assertIn(SEAT3_AFTER_CALLED.tile_type, safe)
        self.assertNotIn(DECLARATION.tile_type, safe)

    def test_partly_unknown_declarations_can_still_widen_the_common_set(self) -> None:
        players, shared_late = self._players(seat3_declaration=None)
        result = derive_riichi_safe_tiles(_input(players))

        first, second = result.by_riichi_player
        self.assertTrue(first.declaration_known)
        self.assertFalse(second.declaration_known)
        self.assertEqual(second.passed_tile_types, frozenset())
        river_only_common = first.river_tile_types & second.river_tile_types
        self.assertEqual(river_only_common, frozenset())
        # 位置が分かっているseat 1に通り、位置が不明なseat 3の河にある牌種は入る。
        self.assertEqual(
            result.safe_against_all_riichi_players, _types(SEAT3_AFTER_CALLED)
        )
        self.assertNotIn(shared_late.tile_type, result.safe_against_all_riichi_players)


class ContractTest(unittest.TestCase):
    def test_response_decision_hand_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            derive_riichi_safe_tiles(
                _input(_single_riichi_players(), concealed=hand(0), drawn=False)
            )

    def test_discard_decision_after_an_own_call_is_accepted(self) -> None:
        # 鳴いた後の打牌判断：純手牌11枚（3n+2）、ツモ牌なし。
        concealed = DISCARD_HAND[:11]
        result = derive_riichi_safe_tiles(
            _input(_single_riichi_players(), concealed=concealed, drawn=False)
        )
        # 自席が鳴いたseat 3の打牌（order 6）も、宣言後に通った牌として入る。
        self.assertIn(
            SEAT3_AFTER_CALLED.tile_type, result.safe_against_all_riichi_players
        )

    def test_own_riichi_state_is_not_a_riichi_opponent(self) -> None:
        players = list(_single_riichi_players())
        players[0] = replace(
            players[0], riichi=RiichiState.ACCEPTED, riichi_declaration_discard_order=0
        )
        result = derive_riichi_safe_tiles(_input(tuple(players)))

        self.assertEqual(
            [item.riichi_seat for item in result.by_riichi_player], [Seat.SEAT_1]
        )

    def test_rejects_non_policy_input(self) -> None:
        with self.assertRaises(TypeError):
            derive_riichi_safe_tiles(object())

    def test_depends_only_on_the_policy_input(self) -> None:
        # 自席の手牌の中身が違っても、公開情報が同じなら結果は同じ。
        other_hand = (*hand(0)[:12], tile(HONOR, 3), DRAWN)
        first = derive_riichi_safe_tiles(_input(_single_riichi_players()))
        second = derive_riichi_safe_tiles(
            _input(_single_riichi_players(), concealed=other_hand)
        )
        self.assertEqual(first, second)

    def test_rule_assumption_identity_is_fixed(self) -> None:
        self.assertEqual(
            RIICHI_SAFE_TILE_RULE_ASSUMPTION,
            "riichi-missed-win-furiten-and-fixed-waits-v1",
        )

    def test_module_passes_the_candidate_import_boundary(self) -> None:
        report = check_candidate_imports("lisjong.belief.riichi_safe_tiles")
        self.assertTrue(report.passed, report.violations)


if __name__ == "__main__":
    unittest.main()
