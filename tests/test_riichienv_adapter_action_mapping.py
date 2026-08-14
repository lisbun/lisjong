"""lisjong.riichienv_adapter.action_mappingのtest。

Issue #29本文および最新コメントを正本として、次を固定する。

- 11 InternalAction variantすべての変換（一部はIssue #27実測をregression
  fixtureとしてRiichiEnv 0.4.8実行から再現する）
- semantic identity（赤牌、tsumogiri、consumed multiset、target、called
  tile、Kakan、winning tile）の保持
- physical copy差だけのsemantic aggregationと、入力順序に依存しない
  deterministic representative
- InternalAction -> decision-local mapping -> 元RiichiEnv legal Actionの
  round-trip
- fail closed（空、変換不能、actor/seat不一致、target不明、Kakan元Pon
  0件/複数件、unmapped、stale、cross-seat、cross-decision、representative
  不整合）
- physical RiichiEnv tile IDやraw RiichiEnv objectがInternalAction object
  graphへ漏れていないこと
"""

import dataclasses
import unittest

from riichienv import Action as RiichiEnvAction
from riichienv import ActionType, Meld, MeldType, Observation, RiichiEnv

from lisjong.policy_contract import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DiscardAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
    Seat,
    Tile,
    TileCategory,
    TileType,
    TsumoAction,
)
from lisjong.riichienv_adapter.action_mapping import (
    ActorMismatchError,
    ContextResolutionError,
    EmptyLegalActionsError,
    RepresentativeSelectionError,
    RiichiEnvActionMapping,
    StaleActionMappingError,
    UnmappedActionError,
    UnsupportedActionError,
    _SemanticGroup,
    build_action_mapping,
)
from lisjong.riichienv_adapter.conversions import (
    seat_from_player_index,
    tile_from_physical_id,
)

# ---------------------------------------------------------------------------
# fixture helpers
#
# `riichienv.Action` / `riichienv.Observation` / `riichienv.Meld`は、Rust実装の
# public typeとして直接構成できる（コンストラクタが公開されている）。実際の
# RiichiEnv 0.4.8局面をsearchせずとも、決定的なfail closedやaggregation規則を
# 検証するための最小Observationを直接組み立てられる。
# ---------------------------------------------------------------------------

_SUIT_BASE_INDEX = {
    TileCategory.MANZU: 0,
    TileCategory.PINZU: 9,
    TileCategory.SOUZU: 18,
}


def physical_id(
    category: TileCategory, rank: int, copy: int = 1, *, red: bool = False
) -> int:
    """testで使う、lisjongのTile意味からRiichiEnv物理牌IDを逆算するhelper。

    conversions.tile_from_physical_idと対になる、test専用の逆変換である。
    """
    if category is TileCategory.HONOR:
        kind_index = 27 + (rank - 1)
    else:
        kind_index = _SUIT_BASE_INDEX[category] + (rank - 1)
    copy_index = 0 if red else copy
    return kind_index * 4 + copy_index


def make_action(
    action_type: ActionType,
    *,
    tile: int | None = None,
    consume_tiles=(),
    actor: int = 0,
) -> RiichiEnvAction:
    return RiichiEnvAction(
        type=action_type, tile=tile, consume_tiles=list(consume_tiles), actor=actor
    )


def make_meld(
    meld_type: MeldType, tiles, *, from_who: int, called_tile: int, opened: bool = True
) -> Meld:
    return Meld(
        meld_type=meld_type,
        tiles=list(tiles),
        opened=opened,
        from_who=from_who,
        called_tile=called_tile,
    )


def make_observation(
    *,
    player_id: int = 0,
    legal_actions,
    melds=None,
    discards=None,
    last_discard=None,
    drawn_tile=None,
    hands=None,
) -> Observation:
    if melds is None:
        melds = [[], [], [], []]
    if discards is None:
        discards = [[], [], [], []]
    if hands is None:
        hands = [[], [], [], []]
    return Observation(
        player_id=player_id,
        hands=hands,
        melds=melds,
        discards=discards,
        dora_indicators=[],
        scores=[25000, 25000, 25000, 25000],
        riichi_declared=[False, False, False, False],
        legal_actions=legal_actions,
        events=[],
        honba=0,
        riichi_sticks=0,
        round_wind=0,
        oya=0,
        kyoku_index=0,
        waits=b"",
        is_tenpai=False,
        riichi_sutehais=[None, None, None, None],
        last_tedashis=[None, None, None, None],
        last_discard=last_discard,
        drawn_tile=drawn_tile,
    )


MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))
MANZU_5_RED = Tile(TileType(TileCategory.MANZU, 5), is_red=True)
MANZU_6 = Tile(TileType(TileCategory.MANZU, 6))
PINZU_1 = Tile(TileType(TileCategory.PINZU, 1))
PINZU_5 = Tile(TileType(TileCategory.PINZU, 5))
SOUZU_8 = Tile(TileType(TileCategory.SOUZU, 8))
WEST_TILE = Tile(TileType(TileCategory.HONOR, 3))


# ---------------------------------------------------------------------------
# conversions.py
# ---------------------------------------------------------------------------


class TileFromPhysicalIdTest(unittest.TestCase):
    def test_matches_riichienv_0_4_8_full_scan(self) -> None:
        # Issue #29着手前の実測（CPython 3.14.0rc2、RiichiEnv 0.4.8）で
        # 136物理牌ID全件を実際にto_mjai()と突き合わせて確認した対応の一部を
        # 固定する。
        self.assertEqual(
            tile_from_physical_id(12), Tile(TileType(TileCategory.MANZU, 4))
        )
        self.assertEqual(
            tile_from_physical_id(16),
            Tile(TileType(TileCategory.MANZU, 5), is_red=True),
        )
        self.assertEqual(
            tile_from_physical_id(17), Tile(TileType(TileCategory.MANZU, 5))
        )
        self.assertEqual(
            tile_from_physical_id(52),
            Tile(TileType(TileCategory.PINZU, 5), is_red=True),
        )
        self.assertEqual(
            tile_from_physical_id(88),
            Tile(TileType(TileCategory.SOUZU, 5), is_red=True),
        )
        self.assertEqual(
            tile_from_physical_id(108), Tile(TileType(TileCategory.HONOR, 1))
        )
        self.assertEqual(
            tile_from_physical_id(135), Tile(TileType(TileCategory.HONOR, 7))
        )

    def test_round_trips_with_physical_id_helper(self) -> None:
        self.assertEqual(
            tile_from_physical_id(physical_id(TileCategory.SOUZU, 8, 2)), SOUZU_8
        )
        self.assertEqual(
            tile_from_physical_id(physical_id(TileCategory.MANZU, 5, red=True)),
            MANZU_5_RED,
        )

    def test_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            tile_from_physical_id(136)
        with self.assertRaises(ValueError):
            tile_from_physical_id(-1)

    def test_rejects_non_int(self) -> None:
        with self.assertRaises(TypeError):
            tile_from_physical_id("12")


class SeatFromPlayerIndexTest(unittest.TestCase):
    def test_valid_indices(self) -> None:
        for index in range(4):
            self.assertEqual(seat_from_player_index(index), Seat(index))

    def test_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            seat_from_player_index(4)

    def test_rejects_non_int(self) -> None:
        with self.assertRaises(TypeError):
            seat_from_player_index(None)


# ---------------------------------------------------------------------------
# 11 variantの変換（synthetic fixture）
# ---------------------------------------------------------------------------


class DiscardConversionTest(unittest.TestCase):
    def test_tedashi_when_tile_differs_from_drawn_tile(self) -> None:
        tile_id = physical_id(TileCategory.SOUZU, 8, 0)
        drawn_id = physical_id(TileCategory.SOUZU, 8, 1)
        action = make_action(ActionType.DISCARD, tile=tile_id, actor=0)
        obs = make_observation(legal_actions=[action], drawn_tile=drawn_id)

        mapping = build_action_mapping(obs)

        self.assertEqual(
            mapping.candidates,
            (DiscardAction(actor=Seat.SEAT_0, tile=SOUZU_8, tsumogiri=False),),
        )

    def test_tsumogiri_when_tile_matches_drawn_tile(self) -> None:
        tile_id = physical_id(TileCategory.SOUZU, 8, 3)
        action = make_action(ActionType.DISCARD, tile=tile_id, actor=0)
        obs = make_observation(legal_actions=[action], drawn_tile=tile_id)

        mapping = build_action_mapping(obs)

        self.assertEqual(
            mapping.candidates,
            (DiscardAction(actor=Seat.SEAT_0, tile=SOUZU_8, tsumogiri=True),),
        )

    def test_red_five_preserved(self) -> None:
        tile_id = physical_id(TileCategory.MANZU, 5, red=True)
        action = make_action(ActionType.DISCARD, tile=tile_id, actor=0)
        obs = make_observation(legal_actions=[action], drawn_tile=tile_id)

        mapping = build_action_mapping(obs)

        self.assertEqual(
            mapping.candidates,
            (DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5_RED, tsumogiri=True),),
        )

    def test_no_drawn_tile_is_always_tedashi(self) -> None:
        # RiichiEnv 0.4.8実測: chi/pon後の打牌decisionではdrawn_tileがNoneになり、
        # 全discardがtedashiとなる。
        tile_id = physical_id(TileCategory.MANZU, 4)
        action = make_action(ActionType.DISCARD, tile=tile_id, actor=0)
        obs = make_observation(legal_actions=[action], drawn_tile=None)

        mapping = build_action_mapping(obs)

        self.assertEqual(mapping.candidates[0].tsumogiri, False)


class RiichiConversionTest(unittest.TestCase):
    def test_translates_to_riichi_action(self) -> None:
        action = make_action(ActionType.RIICHI, actor=2)
        obs = make_observation(player_id=2, legal_actions=[action])

        mapping = build_action_mapping(obs)

        self.assertEqual(mapping.candidates, (RiichiAction(actor=Seat.SEAT_2),))


class ChiConversionTest(unittest.TestCase):
    def test_target_resolved_from_last_discard_and_validated_against_discards(
        self,
    ) -> None:
        called_id = physical_id(TileCategory.MANZU, 5)
        consumed_ids = (
            physical_id(TileCategory.MANZU, 4),
            physical_id(TileCategory.MANZU, 6),
        )
        action = make_action(
            ActionType.CHI, tile=called_id, consume_tiles=consumed_ids, actor=1
        )
        discards = [[called_id], [], [], []]
        obs = make_observation(
            player_id=1, legal_actions=[action], discards=discards, last_discard=0
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(
            mapping.candidates,
            (
                ChiAction(
                    actor=Seat.SEAT_1,
                    target=Seat.SEAT_0,
                    called_tile=MANZU_5,
                    consumed_tiles=(MANZU_4, MANZU_6),
                ),
            ),
        )

    def test_target_mismatch_fails_closed(self) -> None:
        called_id = physical_id(TileCategory.MANZU, 5)
        consumed_ids = (
            physical_id(TileCategory.MANZU, 4),
            physical_id(TileCategory.MANZU, 6),
        )
        action = make_action(
            ActionType.CHI, tile=called_id, consume_tiles=consumed_ids, actor=1
        )
        # discardsの最後がcalled_tileと一致しない
        discards = [[physical_id(TileCategory.PINZU, 1)], [], [], []]
        obs = make_observation(
            player_id=1, legal_actions=[action], discards=discards, last_discard=0
        )

        with self.assertRaises(ContextResolutionError):
            build_action_mapping(obs)

    def test_last_discard_none_fails_closed(self) -> None:
        called_id = physical_id(TileCategory.MANZU, 5)
        consumed_ids = (
            physical_id(TileCategory.MANZU, 4),
            physical_id(TileCategory.MANZU, 6),
        )
        action = make_action(
            ActionType.CHI, tile=called_id, consume_tiles=consumed_ids, actor=1
        )
        obs = make_observation(player_id=1, legal_actions=[action], last_discard=None)

        with self.assertRaises(ContextResolutionError):
            build_action_mapping(obs)


class PonConversionTest(unittest.TestCase):
    def test_target_and_consumed_tiles(self) -> None:
        called_id = physical_id(TileCategory.PINZU, 5)
        consumed_ids = (
            physical_id(TileCategory.PINZU, 5, 2),
            physical_id(TileCategory.PINZU, 5, 3),
        )
        action = make_action(
            ActionType.PON, tile=called_id, consume_tiles=consumed_ids, actor=2
        )
        discards = [[], [], [], [called_id]]
        obs = make_observation(
            player_id=2, legal_actions=[action], discards=discards, last_discard=3
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(
            mapping.candidates,
            (
                PonAction(
                    actor=Seat.SEAT_2,
                    target=Seat.SEAT_3,
                    called_tile=PINZU_5,
                    consumed_tiles=(PINZU_5, PINZU_5),
                ),
            ),
        )


class DaiminkanConversionTest(unittest.TestCase):
    def test_target_and_consumed_tiles(self) -> None:
        called_id = physical_id(TileCategory.SOUZU, 8, 0)
        consumed_ids = (
            physical_id(TileCategory.SOUZU, 8, 1),
            physical_id(TileCategory.SOUZU, 8, 2),
            physical_id(TileCategory.SOUZU, 8, 3),
        )
        action = make_action(
            ActionType.DAIMINKAN, tile=called_id, consume_tiles=consumed_ids, actor=2
        )
        discards = [[], [], [], [called_id]]
        obs = make_observation(
            player_id=2, legal_actions=[action], discards=discards, last_discard=3
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(
            mapping.candidates,
            (
                DaiminkanAction(
                    actor=Seat.SEAT_2,
                    target=Seat.SEAT_3,
                    called_tile=SOUZU_8,
                    consumed_tiles=(SOUZU_8, SOUZU_8, SOUZU_8),
                ),
            ),
        )


class AnkanConversionTest(unittest.TestCase):
    def test_tiles_from_consume_tiles(self) -> None:
        tile_ids = [physical_id(TileCategory.HONOR, 3, copy) for copy in range(4)]
        action = make_action(
            ActionType.ANKAN, tile=tile_ids[0], consume_tiles=tile_ids, actor=3
        )
        obs = make_observation(player_id=3, legal_actions=[action])

        mapping = build_action_mapping(obs)

        self.assertEqual(
            mapping.candidates,
            (
                AnkanAction(
                    actor=Seat.SEAT_3,
                    tiles=(WEST_TILE, WEST_TILE, WEST_TILE, WEST_TILE),
                ),
            ),
        )


class KakanConversionTest(unittest.TestCase):
    def test_resolves_unique_source_pon(self) -> None:
        pon_tiles = [physical_id(TileCategory.PINZU, 1, c) for c in range(3)]
        added_id = physical_id(TileCategory.PINZU, 1, 3)
        action = make_action(
            ActionType.KAKAN, tile=added_id, consume_tiles=pon_tiles, actor=0
        )
        source_meld = make_meld(
            MeldType.Pon, pon_tiles, from_who=1, called_tile=pon_tiles[1]
        )
        melds = [[source_meld], [], [], []]
        obs = make_observation(player_id=0, legal_actions=[action], melds=melds)

        mapping = build_action_mapping(obs)

        self.assertEqual(
            mapping.candidates,
            (
                KakanAction(
                    actor=Seat.SEAT_0,
                    added_tile=PINZU_1,
                    from_seat=Seat.SEAT_1,
                    called_tile=PINZU_1,
                ),
            ),
        )

    def test_zero_source_pon_fails_closed(self) -> None:
        pon_tiles = [physical_id(TileCategory.PINZU, 1, c) for c in range(3)]
        added_id = physical_id(TileCategory.PINZU, 1, 3)
        action = make_action(
            ActionType.KAKAN, tile=added_id, consume_tiles=pon_tiles, actor=0
        )
        obs = make_observation(
            player_id=0, legal_actions=[action], melds=[[], [], [], []]
        )

        with self.assertRaises(ContextResolutionError):
            build_action_mapping(obs)

    def test_multiple_source_pon_fails_closed(self) -> None:
        pon_tiles = [physical_id(TileCategory.PINZU, 1, c) for c in range(3)]
        added_id = physical_id(TileCategory.PINZU, 1, 3)
        action = make_action(
            ActionType.KAKAN, tile=added_id, consume_tiles=pon_tiles, actor=0
        )
        duplicated_melds = [
            make_meld(MeldType.Pon, pon_tiles, from_who=1, called_tile=pon_tiles[0]),
            make_meld(MeldType.Pon, pon_tiles, from_who=2, called_tile=pon_tiles[1]),
        ]
        obs = make_observation(
            player_id=0, legal_actions=[action], melds=[duplicated_melds, [], [], []]
        )

        with self.assertRaises(ContextResolutionError):
            build_action_mapping(obs)

    def test_non_pon_melds_are_not_source_candidates(self) -> None:
        pon_tiles = [physical_id(TileCategory.PINZU, 1, c) for c in range(3)]
        added_id = physical_id(TileCategory.PINZU, 1, 3)
        action = make_action(
            ActionType.KAKAN, tile=added_id, consume_tiles=pon_tiles, actor=0
        )
        already_kakan_meld = make_meld(
            MeldType.Kakan, [*pon_tiles, added_id], from_who=1, called_tile=pon_tiles[0]
        )
        obs = make_observation(
            player_id=0,
            legal_actions=[action],
            melds=[[already_kakan_meld], [], [], []],
        )

        with self.assertRaises(ContextResolutionError):
            build_action_mapping(obs)


class RonConversionTest(unittest.TestCase):
    def test_target_from_last_discard_when_tile_matches_discard(self) -> None:
        winning_id = physical_id(TileCategory.SOUZU, 9)
        action = make_action(ActionType.RON, tile=winning_id, actor=2)
        discards = [[], [], [], [winning_id]]
        obs = make_observation(
            player_id=2, legal_actions=[action], discards=discards, last_discard=3
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(
            mapping.candidates,
            (
                RonAction(
                    actor=Seat.SEAT_2,
                    target=Seat.SEAT_3,
                    winning_tile=Tile(TileType(TileCategory.SOUZU, 9)),
                ),
            ),
        )

    def test_chankan_target_from_last_discard_and_kakan_meld(self) -> None:
        # Issue #27の[AI-REVIEW]対応実測（RiichiEnv v0.4.8ソース確認 + 実機再現、
        # seed=677）: kakanはlast_discard機構を転用してchankan targetを示す。
        winning_id = physical_id(TileCategory.SOUZU, 9, 3)
        action = make_action(ActionType.RON, tile=winning_id, actor=1)
        kakan_meld = make_meld(
            MeldType.Kakan,
            [
                physical_id(TileCategory.SOUZU, 9, 0),
                physical_id(TileCategory.SOUZU, 9, 1),
                physical_id(TileCategory.SOUZU, 9, 2),
                winning_id,
            ],
            from_who=0,
            called_tile=physical_id(TileCategory.SOUZU, 9, 0),
        )
        melds = [[], [], [], [kakan_meld]]
        obs = make_observation(
            player_id=1,
            legal_actions=[action],
            melds=melds,
            discards=[[], [], [], []],
            last_discard=3,
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(mapping.candidates[0].target, Seat.SEAT_3)

    def test_unresolvable_target_fails_closed(self) -> None:
        winning_id = physical_id(TileCategory.SOUZU, 9)
        action = make_action(ActionType.RON, tile=winning_id, actor=2)
        obs = make_observation(
            player_id=2,
            legal_actions=[action],
            discards=[[], [], [], []],
            melds=[[], [], [], []],
            last_discard=3,
        )

        with self.assertRaises(ContextResolutionError):
            build_action_mapping(obs)


class TsumoConversionTest(unittest.TestCase):
    def test_winning_tile_from_action_tile(self) -> None:
        winning_id = physical_id(TileCategory.HONOR, 4)
        action = make_action(ActionType.TSUMO, tile=winning_id, actor=1)
        obs = make_observation(
            player_id=1, legal_actions=[action], drawn_tile=winning_id
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(
            mapping.candidates,
            (
                TsumoAction(
                    actor=Seat.SEAT_1,
                    winning_tile=Tile(TileType(TileCategory.HONOR, 4)),
                ),
            ),
        )


class PassConversionTest(unittest.TestCase):
    def test_translates_to_pass_action(self) -> None:
        action = make_action(ActionType.PASS, actor=0)
        obs = make_observation(legal_actions=[action])

        mapping = build_action_mapping(obs)

        self.assertEqual(mapping.candidates, (PassAction(actor=Seat.SEAT_0),))


class KyuushuKyuuhaiConversionTest(unittest.TestCase):
    def test_translates_to_kyuushu_kyuuhai_action(self) -> None:
        action = make_action(ActionType.KYUSHU_KYUHAI, actor=3)
        obs = make_observation(player_id=3, legal_actions=[action])

        mapping = build_action_mapping(obs)

        self.assertEqual(mapping.candidates, (KyuushuKyuuhaiAction(actor=Seat.SEAT_3),))


# ---------------------------------------------------------------------------
# semantic aggregation / deterministic representative
# ---------------------------------------------------------------------------


class SemanticAggregationTest(unittest.TestCase):
    def test_discard_physical_duplicates_aggregate(self) -> None:
        # drawn_tile=Noneとし（chi/pon後の打牌decision相当）、両candidateとも
        # tedashiで揃える。physical copyだけが異なる純粋な重複にするため。
        tile_a = physical_id(TileCategory.PINZU, 6, 1)
        tile_b = physical_id(TileCategory.PINZU, 6, 3)
        actions = [
            make_action(ActionType.DISCARD, tile=tile_a, actor=0),
            make_action(ActionType.DISCARD, tile=tile_b, actor=0),
        ]
        obs = make_observation(legal_actions=actions, drawn_tile=None)

        mapping = build_action_mapping(obs)

        self.assertEqual(len(mapping.candidates), 1)

    def test_chi_physical_duplicates_aggregate(self) -> None:
        called_id = physical_id(TileCategory.SOUZU, 3)
        discards = [[called_id], [], [], []]
        candidates_ids = [
            (
                physical_id(TileCategory.SOUZU, 4, 1),
                physical_id(TileCategory.SOUZU, 5, 2),
            ),
            (
                physical_id(TileCategory.SOUZU, 4, 2),
                physical_id(TileCategory.SOUZU, 5, 3),
            ),
        ]
        actions = [
            make_action(ActionType.CHI, tile=called_id, consume_tiles=ids, actor=1)
            for ids in candidates_ids
        ]
        obs = make_observation(
            player_id=1, legal_actions=actions, discards=discards, last_discard=0
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(len(mapping.candidates), 1)

    def test_pon_physical_duplicates_aggregate(self) -> None:
        called_id = physical_id(TileCategory.SOUZU, 8, 0)
        discards = [[], [], [], [called_id]]
        candidates_ids = [
            (
                physical_id(TileCategory.SOUZU, 8, 1),
                physical_id(TileCategory.SOUZU, 8, 2),
            ),
            (
                physical_id(TileCategory.SOUZU, 8, 1),
                physical_id(TileCategory.SOUZU, 8, 3),
            ),
            (
                physical_id(TileCategory.SOUZU, 8, 2),
                physical_id(TileCategory.SOUZU, 8, 3),
            ),
        ]
        actions = [
            make_action(ActionType.PON, tile=called_id, consume_tiles=ids, actor=2)
            for ids in candidates_ids
        ]
        obs = make_observation(
            player_id=2, legal_actions=actions, discards=discards, last_discard=3
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(len(mapping.candidates), 1)

    def _duplicate_discard_actions(self) -> tuple[RiichiEnvAction, ...]:
        # drawn_tile=Noneとし、3候補すべてをtedashiで揃える
        # （純粋なphysical copy差だけの重複にするため）。
        tile_a = physical_id(TileCategory.PINZU, 6, 1)
        tile_b = physical_id(TileCategory.PINZU, 6, 3)
        tile_c = physical_id(TileCategory.PINZU, 6, 2)
        return (
            make_action(ActionType.DISCARD, tile=tile_a, actor=0),
            make_action(ActionType.DISCARD, tile=tile_b, actor=0),
            make_action(ActionType.DISCARD, tile=tile_c, actor=0),
        )

    def test_representative_stable_under_reversed_order(self) -> None:
        actions = self._duplicate_discard_actions()
        forward = make_observation(legal_actions=list(actions), drawn_tile=None)
        reversed_obs = make_observation(
            legal_actions=list(reversed(actions)), drawn_tile=None
        )

        forward_mapping = build_action_mapping(forward)
        reversed_mapping = build_action_mapping(reversed_obs)

        forward_repr = forward_mapping.resolve(forward_mapping.candidates[0])
        reversed_repr = reversed_mapping.resolve(reversed_mapping.candidates[0])

        self.assertEqual(forward_repr.to_dict(), reversed_repr.to_dict())

    def test_representative_stable_under_shuffled_order(self) -> None:
        actions = self._duplicate_discard_actions()
        shuffled = (actions[2], actions[0], actions[1])
        forward = make_observation(legal_actions=list(actions), drawn_tile=None)
        shuffled_obs = make_observation(legal_actions=list(shuffled), drawn_tile=None)

        forward_mapping = build_action_mapping(forward)
        shuffled_mapping = build_action_mapping(shuffled_obs)

        forward_repr = forward_mapping.resolve(forward_mapping.candidates[0])
        shuffled_repr = shuffled_mapping.resolve(shuffled_mapping.candidates[0])

        self.assertEqual(forward_repr.to_dict(), shuffled_repr.to_dict())

    def test_ankan_synthetic_duplicates_use_same_representative_rule(self) -> None:
        # RiichiEnv 0.4.8実測ではANKANの重複candidateは出現しなかった（Issue #27）
        # が、Adapterはlist順ではなくphysical fieldに基づく同じ規則で処理できる
        # 必要がある。synthetic候補で確認する。
        tile_ids = [physical_id(TileCategory.HONOR, 5, c) for c in range(4)]
        action_a = make_action(
            ActionType.ANKAN, tile=tile_ids[0], consume_tiles=tile_ids, actor=0
        )
        action_b = make_action(
            ActionType.ANKAN,
            tile=tile_ids[1],
            consume_tiles=list(reversed(tile_ids)),
            actor=0,
        )
        obs = make_observation(legal_actions=[action_a, action_b])

        mapping = build_action_mapping(obs)

        self.assertEqual(len(mapping.candidates), 1)

    def test_daiminkan_synthetic_duplicates_use_same_representative_rule(self) -> None:
        called_id = physical_id(TileCategory.SOUZU, 8, 0)
        consumed = [
            physical_id(TileCategory.SOUZU, 8, 1),
            physical_id(TileCategory.SOUZU, 8, 2),
            physical_id(TileCategory.SOUZU, 8, 3),
        ]
        action_a = make_action(
            ActionType.DAIMINKAN, tile=called_id, consume_tiles=consumed, actor=2
        )
        action_b = make_action(
            ActionType.DAIMINKAN,
            tile=called_id,
            consume_tiles=list(reversed(consumed)),
            actor=2,
        )
        discards = [[], [], [], [called_id]]
        obs = make_observation(
            player_id=2,
            legal_actions=[action_a, action_b],
            discards=discards,
            last_discard=3,
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(len(mapping.candidates), 1)

    def test_kakan_synthetic_duplicates_use_same_representative_rule(self) -> None:
        pon_tiles = [physical_id(TileCategory.PINZU, 1, c) for c in range(3)]
        added_id = physical_id(TileCategory.PINZU, 1, 3)
        action_a = make_action(
            ActionType.KAKAN, tile=added_id, consume_tiles=pon_tiles, actor=0
        )
        action_b = make_action(
            ActionType.KAKAN,
            tile=added_id,
            consume_tiles=list(reversed(pon_tiles)),
            actor=0,
        )
        source_meld = make_meld(
            MeldType.Pon, pon_tiles, from_who=1, called_tile=pon_tiles[0]
        )
        obs = make_observation(
            player_id=0,
            legal_actions=[action_a, action_b],
            melds=[[source_meld], [], [], []],
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(len(mapping.candidates), 1)

    def test_does_not_aggregate_tsumogiri_difference(self) -> None:
        same_kind_a = physical_id(TileCategory.SOUZU, 2, 0)
        drawn_id = physical_id(TileCategory.SOUZU, 2, 1)
        actions = [
            make_action(ActionType.DISCARD, tile=same_kind_a, actor=0),
            make_action(ActionType.DISCARD, tile=drawn_id, actor=0),
        ]
        obs = make_observation(legal_actions=actions, drawn_tile=drawn_id)

        mapping = build_action_mapping(obs)

        self.assertEqual(len(mapping.candidates), 2)
        self.assertEqual({c.tsumogiri for c in mapping.candidates}, {True, False})

    def test_does_not_aggregate_red_difference(self) -> None:
        normal_id = physical_id(TileCategory.MANZU, 5, 1)
        red_id = physical_id(TileCategory.MANZU, 5, red=True)
        actions = [
            make_action(ActionType.DISCARD, tile=normal_id, actor=0),
            make_action(ActionType.DISCARD, tile=red_id, actor=0),
        ]
        obs = make_observation(legal_actions=actions, drawn_tile=None)

        mapping = build_action_mapping(obs)

        self.assertEqual(len(mapping.candidates), 2)
        self.assertEqual({c.tile.is_red for c in mapping.candidates}, {True, False})

    def test_does_not_aggregate_consumed_multiset_difference(self) -> None:
        called_id = physical_id(TileCategory.SOUZU, 3)
        discards = [[called_id], [], [], []]
        action_a = make_action(
            ActionType.CHI,
            tile=called_id,
            consume_tiles=(
                physical_id(TileCategory.SOUZU, 4),
                physical_id(TileCategory.SOUZU, 5),
            ),
            actor=1,
        )
        action_b = make_action(
            ActionType.CHI,
            tile=called_id,
            consume_tiles=(
                physical_id(TileCategory.SOUZU, 2),
                physical_id(TileCategory.SOUZU, 1),
            ),
            actor=1,
        )
        obs = make_observation(
            player_id=1,
            legal_actions=[action_a, action_b],
            discards=discards,
            last_discard=0,
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(len(mapping.candidates), 2)

    def test_does_not_aggregate_target_difference(self) -> None:
        called_id = physical_id(TileCategory.PINZU, 5)
        consumed_a = (
            physical_id(TileCategory.PINZU, 5, 1),
            physical_id(TileCategory.PINZU, 5, 2),
        )
        actions = [
            make_action(
                ActionType.PON, tile=called_id, consume_tiles=consumed_a, actor=2
            )
        ]
        discards_a = [[], [], [], [called_id]]
        obs_a = make_observation(
            player_id=2, legal_actions=actions, discards=discards_a, last_discard=3
        )
        discards_b = [[called_id], [], [], []]
        obs_b = make_observation(
            player_id=2, legal_actions=actions, discards=discards_b, last_discard=0
        )

        mapping_a = build_action_mapping(obs_a)
        mapping_b = build_action_mapping(obs_b)

        self.assertNotEqual(
            mapping_a.candidates[0].target, mapping_b.candidates[0].target
        )

    def test_does_not_aggregate_kakan_difference(self) -> None:
        pon_tiles_1 = [physical_id(TileCategory.PINZU, 1, c) for c in range(3)]
        pon_tiles_2 = [physical_id(TileCategory.PINZU, 2, c) for c in range(3)]
        added_1 = physical_id(TileCategory.PINZU, 1, 3)
        added_2 = physical_id(TileCategory.PINZU, 2, 3)
        action_a = make_action(
            ActionType.KAKAN, tile=added_1, consume_tiles=pon_tiles_1, actor=0
        )
        action_b = make_action(
            ActionType.KAKAN, tile=added_2, consume_tiles=pon_tiles_2, actor=0
        )
        melds = [
            [
                make_meld(
                    MeldType.Pon, pon_tiles_1, from_who=1, called_tile=pon_tiles_1[0]
                ),
                make_meld(
                    MeldType.Pon, pon_tiles_2, from_who=1, called_tile=pon_tiles_2[0]
                ),
            ],
            [],
            [],
            [],
        ]
        obs = make_observation(
            player_id=0, legal_actions=[action_a, action_b], melds=melds
        )

        mapping = build_action_mapping(obs)

        self.assertEqual(len(mapping.candidates), 2)

    def test_does_not_aggregate_winning_tile_difference(self) -> None:
        # last_discardは1decisionにつき1つしか表現できないため、2候補を
        # 別々のObservationとして構成し、winning_tile差がsemantic identityを
        # 分けることを確認する。
        winning_a = physical_id(TileCategory.SOUZU, 9)
        winning_b = physical_id(TileCategory.PINZU, 9)
        obs_a = make_observation(
            player_id=2,
            legal_actions=[make_action(ActionType.RON, tile=winning_a, actor=2)],
            discards=[[], [], [], [winning_a]],
            last_discard=3,
        )
        obs_b = make_observation(
            player_id=2,
            legal_actions=[make_action(ActionType.RON, tile=winning_b, actor=2)],
            discards=[[], [], [], [winning_b]],
            last_discard=3,
        )

        mapping_a = build_action_mapping(obs_a)
        mapping_b = build_action_mapping(obs_b)

        self.assertNotEqual(mapping_a.candidates[0], mapping_b.candidates[0])

    def test_does_not_aggregate_called_tile_difference(self) -> None:
        # 赤牌区分が異なるcalled_tileは別semantic identityとして保持する
        # （通常牌のphysical copy差だけの場合はaggregateされてよいが、赤牌
        # 区分差は麻雀上の意味差である）。
        called_normal = physical_id(TileCategory.PINZU, 5, 1)
        called_red = physical_id(TileCategory.PINZU, 5, red=True)
        consumed = (
            physical_id(TileCategory.PINZU, 5, 2),
            physical_id(TileCategory.PINZU, 5, 3),
        )
        obs_normal = make_observation(
            player_id=2,
            legal_actions=[
                make_action(
                    ActionType.PON, tile=called_normal, consume_tiles=consumed, actor=2
                )
            ],
            discards=[[], [], [], [called_normal]],
            last_discard=3,
        )
        obs_red = make_observation(
            player_id=2,
            legal_actions=[
                make_action(
                    ActionType.PON, tile=called_red, consume_tiles=consumed, actor=2
                )
            ],
            discards=[[], [], [], [called_red]],
            last_discard=3,
        )

        mapping_normal = build_action_mapping(obs_normal)
        mapping_red = build_action_mapping(obs_red)

        self.assertNotEqual(mapping_normal.candidates[0], mapping_red.candidates[0])


# ---------------------------------------------------------------------------
# round-trip / fail closed
# ---------------------------------------------------------------------------


class RoundTripTest(unittest.TestCase):
    def test_resolve_returns_representative_present_in_original_legal_set(self) -> None:
        # RiichiEnv 0.4.8実測: Observation.legal_actions()は呼び出しごとに
        # value相当だが別objectを返し、Action自体もobject identityでしか
        # __eq__判定しない。そのため、生成時に渡したActionそのものへ
        # 戻ることの確認はobject identityではなくvalue（to_dict()）で行う。
        tile_id = physical_id(TileCategory.PINZU, 6, 1)
        action = make_action(ActionType.DISCARD, tile=tile_id, actor=0)
        obs = make_observation(legal_actions=[action], drawn_tile=None)

        mapping = build_action_mapping(obs)
        resolved = mapping.resolve(mapping.candidates[0])

        self.assertEqual(resolved.to_dict(), action.to_dict())


class FailClosedTest(unittest.TestCase):
    def test_empty_legal_actions(self) -> None:
        obs = make_observation(legal_actions=[])

        with self.assertRaises(EmptyLegalActionsError):
            build_action_mapping(obs)

    def test_unsupported_action_type(self) -> None:
        # ActionType.KITAは3人麻雀固有のoperationであり、初期4人麻雀用11
        # variantに含まれない（docs/internal-action-model.md）。
        action = make_action(ActionType.KITA, actor=0)
        obs = make_observation(legal_actions=[action])

        with self.assertRaises(UnsupportedActionError):
            build_action_mapping(obs)

    def test_actor_mismatch_during_conversion(self) -> None:
        action = make_action(ActionType.DISCARD, tile=0, actor=1)
        obs = make_observation(player_id=0, legal_actions=[action])

        with self.assertRaises(ActorMismatchError):
            build_action_mapping(obs)

    def test_unmapped_internal_action(self) -> None:
        action = make_action(ActionType.PASS, actor=0)
        obs = make_observation(legal_actions=[action])
        mapping = build_action_mapping(obs)

        with self.assertRaises(UnmappedActionError):
            mapping.resolve(RiichiAction(actor=Seat.SEAT_0))

    def test_resolve_rejects_non_internal_action(self) -> None:
        action = make_action(ActionType.PASS, actor=0)
        obs = make_observation(legal_actions=[action])
        mapping = build_action_mapping(obs)

        with self.assertRaises(TypeError):
            mapping.resolve("not-an-internal-action")

    def test_stale_mapping_after_first_resolve(self) -> None:
        action = make_action(ActionType.PASS, actor=0)
        obs = make_observation(legal_actions=[action])
        mapping = build_action_mapping(obs)

        mapping.resolve(mapping.candidates[0])

        with self.assertRaises(StaleActionMappingError):
            mapping.resolve(mapping.candidates[0])

    def test_cross_decision_reuse_of_same_mapping_fails_closed(self) -> None:
        # RiichiEnvにはdecision識別用の公式IDが存在しない（Issue #27実測）ため、
        # mapping instance自体を1decisionの使い捨てtokenとして扱う。同じ
        # instanceを「別のdecisionのつもりで」再利用する行為はstaleとして
        # fail closedする。
        action = make_action(ActionType.PASS, actor=0)
        obs = make_observation(legal_actions=[action])
        mapping = build_action_mapping(obs)
        mapping.resolve(mapping.candidates[0])

        later_action = make_action(ActionType.KYUSHU_KYUHAI, actor=0)
        with self.assertRaises(StaleActionMappingError):
            mapping.resolve(KyuushuKyuuhaiAction(actor=Seat.SEAT_0))
        del later_action

    def test_cross_seat_resolve_fails_closed(self) -> None:
        action = make_action(ActionType.PASS, actor=0)
        obs = make_observation(player_id=0, legal_actions=[action])
        mapping = build_action_mapping(obs)

        with self.assertRaises(ActorMismatchError):
            mapping.resolve(PassAction(actor=Seat.SEAT_1))

    def test_representative_missing_from_captured_legal_set_fails_closed(self) -> None:
        # RiichiEnvActionMapping()の公開constructorを直接使い、意図的に
        # representativeを生成時external legal setに含めない矛盾状態を作って、
        # resolve()直前の再検証が機能することを確認する。
        internal_action = PassAction(actor=Seat.SEAT_0)
        included_action = make_action(ActionType.PASS, actor=0)
        stray_representative = make_action(ActionType.PASS, actor=0)
        group = _SemanticGroup(
            internal_action=internal_action,
            external_candidates=(stray_representative,),
            representative=stray_representative,
        )
        mapping = RiichiEnvActionMapping(
            self_seat=Seat.SEAT_0,
            groups={internal_action: group},
            external_legal_actions=(included_action,),
        )

        with self.assertRaises(RepresentativeSelectionError):
            mapping.resolve(internal_action)


# ---------------------------------------------------------------------------
# 情報境界
# ---------------------------------------------------------------------------


class InformationBoundaryTest(unittest.TestCase):
    def test_internal_action_fields_contain_no_riichienv_types(self) -> None:
        called_id = physical_id(TileCategory.MANZU, 5)
        consumed = (
            physical_id(TileCategory.MANZU, 4),
            physical_id(TileCategory.MANZU, 6),
        )
        action = make_action(
            ActionType.CHI, tile=called_id, consume_tiles=consumed, actor=1
        )
        discards = [[called_id], [], [], []]
        obs = make_observation(
            player_id=1, legal_actions=[action], discards=discards, last_discard=0
        )

        mapping = build_action_mapping(obs)
        internal_action = mapping.candidates[0]

        for field in dataclasses.fields(internal_action):
            value = getattr(internal_action, field.name)
            self._assert_no_riichienv_leak(value)

    def _assert_no_riichienv_leak(self, value: object) -> None:
        if isinstance(value, (tuple, list)):
            for item in value:
                self._assert_no_riichienv_leak(item)
            return
        module_name = type(value).__module__
        self.assertNotIn("riichienv", module_name)
        self.assertFalse(hasattr(value, "consume_tiles"))
        self.assertFalse(hasattr(value, "action_type"))

    def test_mapping_candidates_are_plain_internal_actions(self) -> None:
        action = make_action(ActionType.PASS, actor=0)
        obs = make_observation(legal_actions=[action])
        mapping = build_action_mapping(obs)

        for candidate in mapping.candidates:
            self.assertNotIn("riichienv", type(candidate).__module__)


# ---------------------------------------------------------------------------
# Issue #27実測の再現によるregression test（実際のRiichiEnv 0.4.8を実行する）
# ---------------------------------------------------------------------------


def _advance_preferring(env: RiichiEnv, action_type: ActionType, max_steps: int):
    """各decisionでaction_typeを優先しつつ、最初にaction_typeを選べたstepで停止する。"""
    obs_map = env.reset()
    for step in range(max_steps):
        if not obs_map or env.done():
            break
        actions = {}
        matched = None
        for pid, obs in obs_map.items():
            chosen = obs.legal_actions()[0]
            for candidate in obs.legal_actions():
                if candidate.action_type == action_type:
                    chosen = candidate
                    matched = (pid, obs)
            actions[pid] = chosen
        if matched is not None:
            return matched, step
        obs_map = env.step(actions)
    raise AssertionError(f"{action_type} did not appear within {max_steps} steps")


_INTERESTING_ACTION_TYPES = (
    ActionType.RIICHI,
    ActionType.CHI,
    ActionType.PON,
    ActionType.DAIMINKAN,
    ActionType.ANKAN,
    ActionType.KAKAN,
    ActionType.RON,
    ActionType.TSUMO,
    ActionType.KYUSHU_KYUHAI,
)


def _advance_preferring_unseen(env: RiichiEnv, action_type: ActionType, max_steps: int):
    """1局内で、まだ出現していない対象ActionTypeのいずれかを優先しつつ進める。

    riichi等を優先して早めに成立させることで、ron等の後続機会が現れやすい
    局面へ誘導する（docs/riichienv-investigation.mdのIssue #27探索方針と
    同じ考え方を、単一seed内で完結させたもの）。
    """
    seen: set[ActionType] = set()
    obs_map = env.reset()
    for step in range(max_steps):
        if not obs_map or env.done():
            break
        actions = {}
        matched = None
        for pid, obs in obs_map.items():
            chosen = obs.legal_actions()[0]
            for candidate in obs.legal_actions():
                if candidate.action_type in _INTERESTING_ACTION_TYPES and (
                    candidate.action_type not in seen
                ):
                    chosen = candidate
                    break
            seen.add(chosen.action_type)
            actions[pid] = chosen
            if chosen.action_type == action_type:
                matched = (pid, obs)
        if matched is not None:
            return matched, step
        obs_map = env.step(actions)
    raise AssertionError(f"{action_type} did not appear within {max_steps} steps")


def _find_across_seeds(seeds, action_type: ActionType, max_steps: int):
    """複数seedにわたり、action_typeが最初に出現するdecisionを探す。"""
    for seed in seeds:
        env = RiichiEnv(seed=seed)
        try:
            (pid, obs), step = _advance_preferring_unseen(env, action_type, max_steps)
        except AssertionError, IndexError:
            continue
        return seed, (pid, obs), step
    raise AssertionError(f"{action_type} did not appear across seeds {seeds!r}")


class LiveRiichiEnvRegressionTest(unittest.TestCase):
    """#27で実測したRiichiEnv 0.4.8局面をseed固定で再現し、変換をregression化する。"""

    def test_chi_seed1_step1(self) -> None:
        env = RiichiEnv(seed=1)
        (pid, obs), _ = _advance_preferring(env, ActionType.CHI, 5)
        mapping = build_action_mapping(obs)

        self.assertTrue(any(isinstance(c, ChiAction) for c in mapping.candidates))
        chi = next(c for c in mapping.candidates if isinstance(c, ChiAction))
        resolved = mapping.resolve(chi)
        self.assertEqual(resolved.action_type, ActionType.CHI)

    def test_pon_seed1_step19(self) -> None:
        env = RiichiEnv(seed=1)
        (pid, obs), _ = _advance_preferring(env, ActionType.PON, 25)
        mapping = build_action_mapping(obs)

        pon = next(c for c in mapping.candidates if isinstance(c, PonAction))
        resolved = mapping.resolve(pon)
        self.assertEqual(resolved.action_type, ActionType.PON)

    def test_kakan_seed2_step46(self) -> None:
        env = RiichiEnv(seed=2)
        (pid, obs), _ = _advance_preferring(env, ActionType.KAKAN, 60)
        mapping = build_action_mapping(obs)

        kakan = next(c for c in mapping.candidates if isinstance(c, KakanAction))
        resolved = mapping.resolve(kakan)
        self.assertEqual(resolved.action_type, ActionType.KAKAN)

    def test_daiminkan_seed5_step48(self) -> None:
        env = RiichiEnv(seed=5)
        (pid, obs), _ = _advance_preferring(env, ActionType.DAIMINKAN, 60)
        mapping = build_action_mapping(obs)

        daiminkan = next(
            c for c in mapping.candidates if isinstance(c, DaiminkanAction)
        )
        resolved = mapping.resolve(daiminkan)
        self.assertEqual(resolved.action_type, ActionType.DAIMINKAN)

    def test_ankan_seed18_step4(self) -> None:
        env = RiichiEnv(seed=18)
        (pid, obs), _ = _advance_preferring(env, ActionType.ANKAN, 10)
        mapping = build_action_mapping(obs)

        ankan = next(c for c in mapping.candidates if isinstance(c, AnkanAction))
        resolved = mapping.resolve(ankan)
        self.assertEqual(resolved.action_type, ActionType.ANKAN)

    def test_riichi_seed19_step19(self) -> None:
        env = RiichiEnv(seed=19)
        (pid, obs), _ = _advance_preferring(env, ActionType.RIICHI, 25)
        mapping = build_action_mapping(obs)

        riichi = next(c for c in mapping.candidates if isinstance(c, RiichiAction))
        resolved = mapping.resolve(riichi)
        self.assertEqual(resolved.action_type, ActionType.RIICHI)

    def test_ron_appears_in_a_real_game(self) -> None:
        # Issue #27はseed=19/step=44でronの実例を確認しているが、単一target
        # 優先の単純な進行方針では同じ多target累積探索の軌道を再現できない
        # ため、複数seedにわたって同じ単純方針でronを再現できる例を探す。
        _, (pid, obs), _ = _find_across_seeds(range(1, 60), ActionType.RON, 120)
        mapping = build_action_mapping(obs)

        ron = next(c for c in mapping.candidates if isinstance(c, RonAction))
        resolved = mapping.resolve(ron)
        self.assertEqual(resolved.action_type, ActionType.RON)

    def test_tsumo_seed14_step82(self) -> None:
        env = RiichiEnv(seed=14)
        (pid, obs), _ = _advance_preferring(env, ActionType.TSUMO, 90)
        mapping = build_action_mapping(obs)

        tsumo = next(c for c in mapping.candidates if isinstance(c, TsumoAction))
        resolved = mapping.resolve(tsumo)
        self.assertEqual(resolved.action_type, ActionType.TSUMO)

    def test_kyuushu_kyuuhai_seed228_step3(self) -> None:
        env = RiichiEnv(seed=228)
        (pid, obs), _ = _advance_preferring(env, ActionType.KYUSHU_KYUHAI, 10)
        mapping = build_action_mapping(obs)

        kyuushu = next(
            c for c in mapping.candidates if isinstance(c, KyuushuKyuuhaiAction)
        )
        resolved = mapping.resolve(kyuushu)
        self.assertEqual(resolved.action_type, ActionType.KYUSHU_KYUHAI)

    def test_chankan_ron_seed677(self) -> None:
        # Issue #27の[AI-REVIEW]対応実測で確認したkakan chankan局面(seed=677)を
        # 再現し、RonAction.targetがkakan行為者へ解決されることを固定する。
        env = RiichiEnv(seed=677)
        obs_map = env.reset()
        kakan_actor = None
        chankan_observation = None
        for _ in range(200):
            if not obs_map or env.done():
                break
            actions = {}
            for pid, obs in obs_map.items():
                chosen = obs.legal_actions()[0]
                for candidate in obs.legal_actions():
                    if candidate.action_type == ActionType.KAKAN:
                        chosen = candidate
                actions[pid] = chosen
                if chosen.action_type == ActionType.KAKAN:
                    kakan_actor = chosen.actor
            obs_map = env.step(actions)
            if kakan_actor is not None:
                for pid, obs in obs_map.items():
                    if any(
                        a.action_type == ActionType.RON for a in obs.legal_actions()
                    ):
                        chankan_observation = obs
                        break
                kakan_actor_seen = kakan_actor
                kakan_actor = None
                if chankan_observation is not None:
                    break
        self.assertIsNotNone(chankan_observation, "chankan scenario did not reproduce")

        mapping = build_action_mapping(chankan_observation)
        ron_candidates = [c for c in mapping.candidates if isinstance(c, RonAction)]
        self.assertEqual(len(ron_candidates), 1)
        self.assertEqual(ron_candidates[0].target, Seat(kakan_actor_seen))

        resolved = mapping.resolve(ron_candidates[0])
        self.assertEqual(resolved.action_type, ActionType.RON)

    def test_many_real_decisions_never_raise_adapter_error(self) -> None:
        """広範なseedにわたり、adapterがfail closed以外の未処理例外を出さないことを確認する。"""
        for seed in range(1, 30):
            env = RiichiEnv(seed=seed)
            obs_map = env.reset()
            step = 0
            while obs_map and not env.done() and step < 150:
                actions = {}
                for pid, obs in obs_map.items():
                    mapping = build_action_mapping(obs)
                    self.assertGreaterEqual(len(mapping.candidates), 1)
                    chosen_external = obs.legal_actions()[0]
                    actions[pid] = chosen_external
                obs_map = env.step(actions)
                step += 1


if __name__ == "__main__":
    unittest.main()
