"""Issue #211 `Kobalab0004ReferencePolicy`のsource-semantic unit test。

期待値はkobalab/majiang-ai legacy 0004（commit
e75a9720a12b84c03e6c61c3960c1844b8982eb4）の`player-0004.js` /
`suanpai-0004.js`の式を手で適用して求めたmanual golden valueである。
Policy実装を呼んで期待値を作る自己検証は避ける。upstream実行結果との差分検証は
`test_kobalab_0004_upstream_differential.py`が担当する。
"""

import ast
import itertools
import random
import unittest
from pathlib import Path
from unittest import mock

from lisjong.belief import (
    SCALE,
    ConcealedHandBelief,
    HandBelief,
    derive_remaining_tile_inventory,
    exact_self_belief,
    sum_tile_type_values,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
from lisjong.belief import conditional_uniform_hand_belief as uniform_module
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies import Kobalab0004BeliefPaijiaPolicy, Kobalab0004ReferencePolicy
from lisjong.policies import kobalab_0004_reference as reference_module
from lisjong.policies.kobalab_0004_reference import (
    KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR,
    KOBALAB_0004_BELIEF_PAIJIA_IDENTITY,
    KOBALAB_0004_REFERENCE_IDENTITY,
    Kobalab0004ReferencePolicyError,
    _allows_riichi_discard,
    _DiscardStructures,
    _improving_tile_types,
    _opponent_concealed_slot_counts_by_wind,
    _paijia_input_from_belief,
    _PaijiaInput,
    _PublicCounts,
    evaluation_order,
)
from lisjong.policy_contract.action import (
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
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_execution import execute_policy
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

_CATEGORIES = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}


def _hand(spec: str) -> tuple[Tile, ...]:
    """`"123m0s"`形式。`0`は赤5、字牌rankは東南西北白發中=1..7。"""
    tiles: list[Tile] = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        category = _CATEGORIES[character]
        for rank in ranks:
            number = int(rank)
            tiles.append(Tile(TileType(category, number or 5), is_red=number == 0))
        ranks = ""
    assert not ranks, spec
    return tuple(tiles)


def _t(spec: str) -> Tile:
    (tile,) = _hand(spec)
    return tile


def _player(
    discards: tuple[Discard, ...] = (),
    melds: tuple[PublicMeld, ...] = (),
    riichi: RiichiState = RiichiState.NONE,
) -> PlayerPublicState:
    return PlayerPublicState(score=25000, discards=discards, melds=melds, riichi=riichi)


def _discards(spec: str, *, called_by: Seat | None = None) -> tuple[Discard, ...]:
    return tuple(
        Discard(tile=tile, tsumogiri=False, order=index, called_by=called_by)
        for index, tile in enumerate(_hand(spec))
    )


def _input(
    concealed: str,
    drawn: str | None = None,
    *,
    self_seat: Seat = Seat.SEAT_0,
    dealer_seat: Seat = Seat.SEAT_0,
    round_wind: Wind = Wind.EAST,
    players: tuple[PlayerPublicState, ...] | None = None,
    dora_indicators: str = "2z",
) -> PolicyInput:
    """既定dora表示牌は南（ドラ西）。"""
    return PolicyInput(
        self_seat=self_seat,
        round=RoundState(
            round_wind=round_wind,
            hand_number=1,
            dealer_seat=dealer_seat,
            honba=0,
            riichi_sticks=0,
            dora_indicators=_hand(dora_indicators),
            live_wall_tiles_remaining=60,
        ),
        players=players if players is not None else (_player(),) * 4,
        own_hand=OwnHandState(
            concealed_tiles=_hand(concealed),
            drawn_tile=_t(drawn) if drawn is not None else None,
        ),
    )


def _discard(spec: str, *, tsumogiri: bool = False) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=_t(spec), tsumogiri=tsumogiri)


def _all_discards(
    concealed: str, drawn: str | None = None
) -> tuple[DiscardAction, ...]:
    """手出し候補（exact Tile単位で一意）と、drawnがあればツモ切りを列挙する。"""
    tiles = _hand(concealed)
    drawn_tile = _t(drawn) if drawn is not None else None
    actions: list[DiscardAction] = []
    seen: set[Tile] = set()
    for tile in tiles:
        if tile in seen:
            continue
        seen.add(tile)
        copies = tiles.count(tile)
        if tile == drawn_tile and copies == 1:
            continue
        actions.append(DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=False))
    if drawn_tile is not None:
        actions.append(
            DiscardAction(actor=Seat.SEAT_0, tile=drawn_tile, tsumogiri=True)
        )
    return tuple(actions)


def _choose(policy_input: PolicyInput, actions: tuple[object, ...]) -> object:
    decision = DecisionContext(input=policy_input, legal_actions=actions)
    return execute_policy(Kobalab0004ReferencePolicy(), decision)


class IdentityTest(unittest.TestCase):
    def test_stable_identity_is_explicit(self) -> None:
        self.assertEqual(
            KOBALAB_0004_REFERENCE_IDENTITY,
            "kobalab-0004-tile-efficiency-reference-v1",
        )
        self.assertEqual(
            Kobalab0004ReferencePolicy.identity, KOBALAB_0004_REFERENCE_IDENTITY
        )
        self.assertIn(
            "kobalab/majiang-ai legacy 0004",
            Kobalab0004ReferencePolicy.reference_source,
        )
        self.assertIn("MIT", Kobalab0004ReferencePolicy.reference_source)


class PaijiaTest(unittest.TestCase):
    """`SuanPai.paijia()`の式をmanualに適用したgolden value。"""

    def test_honor_round_seat_dragon_and_dora(self) -> None:
        # 各字牌1枚ずつ所持、dora表示牌南(2z) -> ドラ西(3z)。南は表示牌分も既知。
        counts = _PublicCounts(_input("1234567z123m456p"))
        self.assertEqual(counts.paijia(_t("1z")), 3 * 2 * 2)  # 場風・自風
        self.assertEqual(counts.paijia(_t("2z")), 2)
        self.assertEqual(counts.paijia(_t("3z")), 3 * 2 * 2)  # dora weight二重
        self.assertEqual(counts.paijia(_t("4z")), 3)
        self.assertEqual(counts.paijia(_t("5z")), 3 * 2)  # 三元牌

    def test_honor_with_other_round_and_seat_wind(self) -> None:
        counts = _PublicCounts(
            _input(
                "1234567z123m456p",
                self_seat=Seat.SEAT_2,
                round_wind=Wind.SOUTH,
            )
        )
        self.assertEqual(counts.paijia(_t("1z")), 3)
        self.assertEqual(counts.paijia(_t("2z")), 2 * 2)  # 場風南
        self.assertEqual(counts.paijia(_t("3z")), 3 * 2 * 2 * 2)  # dora + 自風西

    def test_suited_edges(self) -> None:
        counts = _PublicCounts(_input("19m1234567z"))
        # 1m: [0, 0, 3, 4, 4] with weights [0, 0, 1, 1, 1]
        self.assertEqual(counts.paijia(_t("1m")), 11)
        # 9m: [4, 4, 3, 0, 0]
        self.assertEqual(counts.paijia(_t("9m")), 11)

    def test_red_five_bonus_and_red_doubling(self) -> None:
        counts = _PublicCounts(_input("4s1234567z123m"))
        # 4s: [4, 4, 3, 4, 4] + min(red=1, n_pai[3]=4)
        self.assertEqual(counts.paijia(_t("4s")), 20)

        counts = _PublicCounts(_input("0555s1p123456789m"))
        # 5s: 残り0。[4, 4, 0, 4, 4]、赤5は自手なのでbonusなし。
        self.assertEqual(counts.paijia(_t("5s")), 16)
        self.assertEqual(counts.paijia(_t("0s")), 32)

    def test_dora_neighbour_weights(self) -> None:
        counts = _PublicCounts(_input("23s1234567z123m", dora_indicators="1s"))
        # 残り: 1s=3（表示牌）, 2s=3, 3s=3（自手）, 4s以降=4。
        # 3s: n_pai=[3, 3, 3, 4, 4], weights=[1, 2, 1, 1, 1] -> 20, red bonus +1
        self.assertEqual(counts.paijia(_t("3s")), 21)
        # 2s: n_pai=[0, 3, 3, 3, 3], weights=[0, 1, 2, 1, 1] -> 15, x dora 2
        self.assertEqual(counts.paijia(_t("2s")), 30)


class RemainingCountTest(unittest.TestCase):
    """実残り枚数: 判断時点の公開情報、鳴かれた牌の重複排除、槓、ドラ表示牌。"""

    def test_called_discard_is_not_double_counted(self) -> None:
        players = (
            _player(),
            _player(discards=_discards("3m", called_by=Seat.SEAT_2)),
            _player(
                melds=(
                    PublicMeld(
                        kind=MeldKind.PON,
                        tiles=_hand("333m"),
                        from_seat=Seat.SEAT_1,
                        called_tile=_t("3m"),
                    ),
                )
            ),
            _player(),
        )
        counts = _PublicCounts(_input("1234567z123p456s", players=players))
        self.assertEqual(counts.remaining(TileType(TileCategory.MANZU, 3)), 1)

    def test_kakan_ankan_and_new_dora_indicator(self) -> None:
        players = (
            _player(),
            _player(discards=_discards("7p", called_by=Seat.SEAT_2)),
            _player(
                melds=(
                    PublicMeld(
                        kind=MeldKind.KAKAN,
                        tiles=_hand("7777p"),
                        from_seat=Seat.SEAT_1,
                        called_tile=_t("7p"),
                    ),
                )
            ),
            _player(
                melds=(
                    PublicMeld(
                        kind=MeldKind.ANKAN,
                        tiles=_hand("9999s"),
                        from_seat=None,
                        called_tile=None,
                    ),
                )
            ),
        )
        counts = _PublicCounts(
            _input("1234567z123m456p", players=players, dora_indicators="2z1s")
        )
        self.assertEqual(counts.remaining(TileType(TileCategory.PINZU, 7)), 0)
        self.assertEqual(counts.remaining(TileType(TileCategory.SOUZU, 9)), 0)
        self.assertEqual(counts.remaining(TileType(TileCategory.SOUZU, 1)), 3)
        self.assertEqual(counts.remaining(TileType(TileCategory.HONOR, 2)), 2)

    def test_red_five_counts_in_five_total_once(self) -> None:
        counts = _PublicCounts(_input("05m1234567z123p45s"))
        self.assertEqual(counts.remaining(TileType(TileCategory.MANZU, 5)), 2)

    def test_own_discards_are_known(self) -> None:
        players = (_player(discards=_discards("9m9m")),) + (_player(),) * 3
        counts = _PublicCounts(_input("9m1234567z123p45s", players=players))
        self.assertEqual(counts.remaining(TileType(TileCategory.MANZU, 9)), 1)

    def test_inconsistent_visible_count_fails_closed(self) -> None:
        players = (_player(discards=_discards("1z1z1z")),) + (_player(),) * 3
        with self.assertRaises(ValueError):
            _PublicCounts(_input("11z23m456p789s123m", players=players))


class ShantenReuseTest(unittest.TestCase):
    def test_chiitoitsu_and_kokushi_improving_tiles(self) -> None:
        self.assertEqual(
            _improving_tile_types(_hand("1133557799m11p3p")),
            (TileType(TileCategory.PINZU, 3),),
        )
        self.assertEqual(
            set(_improving_tile_types(_hand("19m19p19s1234567z"))),
            {tile.tile_type for tile in _hand("19m19p19s1234567z")},
        )

    def test_tile_held_four_times_is_not_a_wait(self) -> None:
        # 123p456p789p1111s: majiang-coreの式では聴牌(0)だが待ち1sは自手4枚で
        # tingpaiなし。lisjong exact shantenは5枚目を要する分解を認めず1向聴と
        # する（docsに記録した既知の向聴定義差）。どちらでも立直不可は一致する。
        self.assertEqual(calculate_shanten(_hand("123456789p1111s")), 1)
        policy_input = _input("123456789p1111s5z", "5z")
        self.assertFalse(_allows_riichi_discard(policy_input, _discard("5z")))
        self.assertTrue(_allows_riichi_discard(policy_input, _discard("1s")))

    def test_post_kan_hand_uses_standard_form(self) -> None:
        # 暗槓後10枚: 23m 456p 789s 5z6z -> 確定1面子込みで1向聴。
        self.assertEqual(len(_improving_tile_types(_hand("23m456p789s56z"))) > 0, True)


def _physical_tiles() -> list[Tile]:
    tiles: list[Tile] = []
    for category in _CATEGORIES.values():
        for rank in range(1, (7 if category is TileCategory.HONOR else 9) + 1):
            tile_type = TileType(category, rank)
            red = 1 if rank == 5 and category is not TileCategory.HONOR else 0
            if red:
                tiles.append(Tile(tile_type, is_red=True))
            tiles += [Tile(tile_type)] * (4 - red)
    return tiles


class DiscardStructuresEquivalenceTest(unittest.TestCase):
    """#218の共有構造評価がTile単位の定義（打牌後`calculate_shanten` /
    `_improving_tile_types`）と中間値で一致することを固定する。"""

    def _assert_equivalent(self, hand: tuple[Tile, ...]) -> None:
        structures = _DiscardStructures(hand)
        self.assertEqual(structures.shanten, calculate_shanten(hand))
        for tile in set(hand):
            after = list(hand)
            after.remove(tile)
            with self.subTest(hand=hand, tile=tile):
                self.assertEqual(
                    structures.shanten_after(tile), calculate_shanten(after)
                )
                self.assertEqual(
                    structures.improving_after(tile),
                    tuple(tile_type_index(t) for t in _improving_tile_types(after)),
                )

    def test_random_hands_of_every_meld_count(self) -> None:
        generator = random.Random(218)
        physical = _physical_tiles()
        for size in (14, 11, 8, 5, 2):
            for _ in range(60):
                self._assert_equivalent(tuple(generator.sample(physical, size)))

    def test_four_copies_red_fives_and_special_forms(self) -> None:
        for spec in (
            "123456789p1111s5z",
            "0555s1p123456789m",
            "1133557799m11p3p5z",
            "19m19p19s1234567z1m",
            "05m1234567z123p45s",
            "1111m2222p3333s44z",
        ):
            self._assert_equivalent(_hand(spec))

    def test_red_and_normal_five_share_structure(self) -> None:
        structures = _DiscardStructures(_hand("05m1234567z123p45s"))
        self.assertEqual(
            structures.improving_after(_t("0m")), structures.improving_after(_t("5m"))
        )

    def test_tile_not_in_hand_fails_closed(self) -> None:
        structures = _DiscardStructures(_hand("123m456p789s1234z5z"))
        with self.assertRaises(Kobalab0004ReferencePolicyError):
            structures.shanten_after(_t("9m"))
        with self.assertRaises(Kobalab0004ReferencePolicyError):
            _DiscardStructures(_hand("123m456p789s1235z5z")).improving_after(_t("0m"))

    def test_wait_with_no_unseen_copy_stays_improving_and_allows_riichi(self) -> None:
        # Issue #221: 待ち5z（単騎）の残り3枚が河に見えていて未見0でも、
        # 構造上の改善牌には残り、立直可否を狭めない。受入合計は0になる。
        players = (
            (_player(),) + (_player(discards=_discards("5z5z5z")),) + (_player(),) * 2
        )
        policy_input = _input("123456789m123p5z9s", "9s", players=players)
        structures = _DiscardStructures(policy_input.own_hand.concealed_tiles)
        improving = structures.improving_after(_t("9s"))
        waits = (tile_type_index(_t("5z").tile_type),)
        self.assertEqual(improving, waits)
        counts = _PublicCounts(policy_input)
        self.assertEqual(sum_tile_type_values(counts.remaining_tile_counts, waits), 0)
        self.assertTrue(
            _allows_riichi_discard(policy_input, _discard("9s"), structures)
        )
        self.assertFalse(
            _allows_riichi_discard(policy_input, _discard("1m"), structures)
        )

    def test_riichi_check_reuses_structures_with_same_result(self) -> None:
        policy_input = _input("123456789p1111s5z", "5z")
        structures = _DiscardStructures(policy_input.own_hand.concealed_tiles)
        for spec in ("5z", "1s", "1p"):
            self.assertEqual(
                _allows_riichi_discard(policy_input, _discard(spec), structures),
                _allows_riichi_discard(policy_input, _discard(spec)),
            )


class DiscardSelectionTest(unittest.TestCase):
    def test_maximum_actual_ukeire_among_non_worsening(self) -> None:
        concealed = "123456789m11p34s6s"
        # 6s切り: 両面25s(8枚) / 3s切り: 嵌張5s(4枚) / 他は向聴悪化。
        self.assertEqual(
            _choose(_input(concealed), _all_discards(concealed)), _discard("6s")
        )

    def test_visible_tiles_flip_tanki_choice(self) -> None:
        concealed = "123456789m234p56z"
        # 公開情報なし: 5z/6zとも単騎3枚、paijia同点 -> source逆順で6zが先。
        self.assertEqual(
            _choose(_input(concealed), _all_discards(concealed)), _discard("6z")
        )
        # 5zが2枚見えると、5z切り(6z単騎3枚) > 6z切り(5z単騎1枚)。
        players = (_player(), _player(discards=_discards("55z")), _player(), _player())
        self.assertEqual(
            _choose(_input(concealed, players=players), _all_discards(concealed)),
            _discard("5z"),
        )

    def test_red_five_is_kept_on_complete_ukeire_tie(self) -> None:
        concealed = "0555s1p123456789m"
        # 1p切り: 5s単騎(自手4枚) -> 0。5s / 0s切り: 1p単騎3枚で同点。
        # paijia(5s)=16 < paijia(0s)=32 -> 通常5sを切り赤5を残す。
        self.assertEqual(
            _choose(_input(concealed), _all_discards(concealed)), _discard("5s")
        )

    def test_tsumogiri_precedes_identical_tedashi(self) -> None:
        concealed = "123456789m23p999s"
        self.assertEqual(
            _choose(_input(concealed, "9s"), _all_discards(concealed, "9s")),
            _discard("9s", tsumogiri=True),
        )

    def test_evaluation_order_matches_source_reverse_then_stable_sort(self) -> None:
        concealed = "0555s1p123456789m"
        order = evaluation_order(_input(concealed), _all_discards(concealed))
        # paijia: 1m=9m=9, 1p=11, 2m=8m=12, 5s=16, 3m..7m=15+赤5m bonus 1=16, 0s=32。
        # 同値はsource逆順（ツモ切り -> 字牌 -> 索子 -> 筒子 -> 萬子、各9..1）。
        self.assertEqual(
            [action.tile for action in order],
            [
                _t(spec)
                for spec in (
                    "9m",
                    "1m",
                    "1p",
                    "8m",
                    "2m",
                    "5s",
                    "7m",
                    "6m",
                    "5m",
                    "4m",
                    "3m",
                    "0s",
                )
            ],
        )

    def test_legal_action_order_does_not_matter(self) -> None:
        concealed = "0555s1p123456789m"
        actions = _all_discards(concealed)
        expected = _choose(_input(concealed), actions)
        for permutation in itertools.islice(itertools.permutations(actions), 0, 200, 7):
            self.assertEqual(_choose(_input(concealed), permutation), expected)
        self.assertEqual(_choose(_input(concealed), tuple(reversed(actions))), expected)


class TopLevelTest(unittest.TestCase):
    def test_tsumo_is_taken(self) -> None:
        concealed = "123456789m11p234s"
        actions = (TsumoAction(actor=Seat.SEAT_0, winning_tile=_t("4s")),) + (
            _all_discards(concealed, "4s")
        )
        self.assertIsInstance(_choose(_input(concealed, "4s"), actions), TsumoAction)

    def test_ron_on_discard_is_taken(self) -> None:
        ron = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=_t("4s"))
        actions = (ron, PassAction(actor=Seat.SEAT_0))
        self.assertEqual(_choose(_input("123456789m11p23s"), actions), ron)

    def test_ron_on_ankan_is_declined_but_kakan_is_taken(self) -> None:
        ankan = PublicMeld(
            kind=MeldKind.ANKAN, tiles=_hand("1111z"), from_seat=None, called_tile=None
        )
        players = (_player(), _player(melds=(ankan,)), _player(), _player())
        ron = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=_t("1z"))
        actions = (ron, PassAction(actor=Seat.SEAT_0))
        self.assertEqual(
            _choose(_input("19m19p19s234567z", players=players), actions),
            PassAction(actor=Seat.SEAT_0),
        )

        kakan = PublicMeld(
            kind=MeldKind.KAKAN,
            tiles=_hand("5555s"),
            from_seat=Seat.SEAT_2,
            called_tile=_t("5s"),
        )
        players = (
            _player(),
            _player(melds=(kakan,)),
            _player(discards=_discards("5s", called_by=Seat.SEAT_1)),
            _player(),
        )
        ron = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=_t("5s"))
        self.assertEqual(
            _choose(
                _input("123456789m11p46s", players=players),
                (ron, PassAction(actor=Seat.SEAT_0)),
            ),
            ron,
        )

    def test_no_voluntary_chi_pon_daiminkan(self) -> None:
        actor = Seat.SEAT_0
        actions = (
            ChiAction(
                actor=actor,
                target=Seat.SEAT_3,
                called_tile=_t("3m"),
                consumed_tiles=_hand("12m"),
            ),
            PonAction(
                actor=actor,
                target=Seat.SEAT_3,
                called_tile=_t("3m"),
                consumed_tiles=_hand("33m"),
            ),
            DaiminkanAction(
                actor=actor,
                target=Seat.SEAT_3,
                called_tile=_t("3m"),
                consumed_tiles=_hand("333m"),
            ),
            PassAction(actor=actor),
        )
        self.assertEqual(
            _choose(_input("12333m456p789s11z"), actions), PassAction(actor=actor)
        )

    def test_kyuushu_kyuuhai_requires_shanten_at_least_four(self) -> None:
        concealed = "19m2468m19p3p1s1234z"
        actions = (KyuushuKyuuhaiAction(actor=Seat.SEAT_0),) + _all_discards(concealed)
        self.assertEqual(
            _choose(_input(concealed), actions), KyuushuKyuuhaiAction(actor=Seat.SEAT_0)
        )

        concealed = "19m258m19p3p19s1234z"  # 国士3向聴
        actions = (KyuushuKyuuhaiAction(actor=Seat.SEAT_0),) + _all_discards(concealed)
        self.assertIsInstance(_choose(_input(concealed), actions), DiscardAction)

    def test_kan_requires_equal_shanten(self) -> None:
        concealed = "1111m123p789s11z56p"
        ankan = AnkanAction(actor=Seat.SEAT_0, tiles=_hand("1111m"))
        self.assertEqual(
            _choose(_input(concealed), (ankan,) + _all_discards(concealed)), ankan
        )

        concealed = "1111m23m456p789s56z"  # 暗槓で123mが崩れ0 -> 1向聴
        ankan = AnkanAction(actor=Seat.SEAT_0, tiles=_hand("1111m"))
        self.assertIsInstance(
            _choose(_input(concealed), (ankan,) + _all_discards(concealed)),
            DiscardAction,
        )

    def test_kan_candidates_follow_source_order(self) -> None:
        concealed = "1111m5555p234s112z"
        m_kan = AnkanAction(actor=Seat.SEAT_0, tiles=_hand("1111m"))
        p_kan = AnkanAction(actor=Seat.SEAT_0, tiles=_hand("5555p"))
        discards = _all_discards(concealed)
        self.assertEqual(_choose(_input(concealed), (p_kan, m_kan) + discards), m_kan)

        concealed = "1111m23m5555p11z7z9s"
        m_kan = AnkanAction(actor=Seat.SEAT_0, tiles=_hand("1111m"))
        p_kan = AnkanAction(actor=Seat.SEAT_0, tiles=_hand("5555p"))
        discards = _all_discards(concealed)
        self.assertEqual(_choose(_input(concealed), (m_kan, p_kan) + discards), p_kan)

    def test_kakan_uses_added_tile(self) -> None:
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=_hand("777z"),
            from_seat=Seat.SEAT_1,
            called_tile=_t("7z"),
        )
        players = (
            _player(melds=(pon,)),
            _player(discards=_discards("7z", called_by=Seat.SEAT_0)),
            _player(),
            _player(),
        )
        concealed = "123m456p789s1z7z"  # Pon 777z + 11枚
        kakan = KakanAction(
            actor=Seat.SEAT_0,
            added_tile=_t("7z"),
            from_seat=Seat.SEAT_1,
            called_tile=_t("7z"),
        )
        self.assertEqual(
            _choose(
                _input(concealed, players=players), (kakan,) + _all_discards(concealed)
            ),
            kakan,
        )

    def test_immediate_riichi_then_same_declaration_discard(self) -> None:
        concealed = "123456789m11p24s1z"
        actions = (RiichiAction(actor=Seat.SEAT_0),) + _all_discards(concealed)
        self.assertEqual(
            _choose(_input(concealed), actions), RiichiAction(actor=Seat.SEAT_0)
        )

        declared = (_player(riichi=RiichiState.DECLARED),) + (_player(),) * 3
        self.assertEqual(
            _choose(_input(concealed, players=declared), (_discard("1z"),)),
            _discard("1z"),
        )
        # 立直が合法でなければ打牌だけを返す。
        self.assertEqual(
            _choose(_input(concealed), _all_discards(concealed)), _discard("1z")
        )

    def test_opponent_riichi_does_not_add_fold_behavior(self) -> None:
        concealed = "0555s1p123456789m"
        riichi = (_player(),) + (_player(riichi=RiichiState.ACCEPTED),) * 3
        self.assertEqual(
            _choose(_input(concealed, players=riichi), _all_discards(concealed)),
            _choose(_input(concealed), _all_discards(concealed)),
        )

    def test_same_input_same_action(self) -> None:
        concealed = "123456789m234p56z"
        policy_input = _input(concealed)
        actions = _all_discards(concealed)
        results = {_choose(policy_input, actions) for _ in range(5)}
        self.assertEqual(len(results), 1)

    def test_undefined_decision_fails_closed(self) -> None:
        decision = DecisionContext(
            input=_input("123456789m11p23s"),
            legal_actions=(RiichiAction(actor=Seat.SEAT_0),),
        )
        with self.assertRaises(Kobalab0004ReferencePolicyError):
            Kobalab0004ReferencePolicy().choose_action(decision)


def _choose_with(policy, policy_input: PolicyInput, actions: tuple) -> object:
    decision = DecisionContext(input=policy_input, legal_actions=actions)
    return execute_policy(policy, decision)


def _belief_with_opponent_mass(
    policy_input: PolicyInput, mass: dict[tuple[Seat, str], int]
) -> ConcealedHandBelief:
    """selfはexact、他家は`mass`（(seat, 牌) -> raw）だけを持つbelief。"""
    rows = [[0] * 34 for _ in range(4)]
    for (seat, spec), raw in mass.items():
        wind = wind_for_seat(seat, policy_input.round.dealer_seat)
        rows[wind_index(wind)][tile_type_index(_t(spec).tile_type)] += raw
    self_wind = wind_for_seat(policy_input.self_seat, policy_input.round.dealer_seat)
    return ConcealedHandBelief(
        hands=tuple(
            exact_self_belief(policy_input.own_hand)
            if number == wind_index(self_wind)
            else HandBelief(
                expected_count_raw=tuple(rows[number]),
                red_five_probability_raw=(0,) * 3,
            )
            for number in range(4)
        )
    )


class BeliefPaijiaIdentityTest(unittest.TestCase):
    def test_belief_variant_has_its_own_identity_and_estimator_record(self) -> None:
        self.assertEqual(
            KOBALAB_0004_BELIEF_PAIJIA_IDENTITY,
            "kobalab-0004-tile-efficiency-belief-paijia-v1",
        )
        self.assertEqual(
            Kobalab0004BeliefPaijiaPolicy.identity, KOBALAB_0004_BELIEF_PAIJIA_IDENTITY
        )
        self.assertNotEqual(
            Kobalab0004BeliefPaijiaPolicy.identity, Kobalab0004ReferencePolicy.identity
        )
        self.assertEqual(
            Kobalab0004BeliefPaijiaPolicy.belief_estimator,
            KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR,
        )
        self.assertIn("conditional-uniform", KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR)
        self.assertEqual(
            Kobalab0004ReferencePolicy.identity, KOBALAB_0004_REFERENCE_IDENTITY
        )


class PaijiaScaleTest(unittest.TestCase):
    """paijiaの式は入力の共通倍率に対して同じ倍率になる（丸めなし）。"""

    def test_common_scale_multiplies_every_paijia(self) -> None:
        generator = random.Random(218)
        policy_input = _input("123m456p789s1234z5z", dora_indicators="4m7z")
        tiles = sorted(set(_physical_tiles()), key=repr)
        fives = tuple(
            TileType(category, 5) for category in list(_CATEGORIES.values())[:3]
        )
        for _ in range(50):
            counts = [generator.randint(0, 4) for _ in range(34)]
            red = [
                generator.randint(0, min(1, counts[tile_type_index(five)]))
                for five in fives
            ]
            base = _PaijiaInput(counts, red, policy_input)
            scaled = _PaijiaInput(
                [c * SCALE for c in counts], [r * SCALE for r in red], policy_input
            )
            for tile in tiles:
                self.assertEqual(scaled.paijia(tile), base.paijia(tile) * SCALE)

    def test_zero_mass_is_all_zero_without_division(self) -> None:
        zero = _PaijiaInput((0,) * 34, (0,) * 3, _input("123m456p789s1234z5z"))
        self.assertEqual({zero.paijia(tile) for tile in _physical_tiles()}, {0})


class OpponentSlotCountTest(unittest.TestCase):
    def test_slots_follow_public_melds_and_seat_wind(self) -> None:
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=_hand("555z"),
            from_seat=Seat.SEAT_0,
            called_tile=_t("5z"),
        )
        ankan = PublicMeld(
            kind=MeldKind.ANKAN, tiles=_hand("9999m"), from_seat=None, called_tile=None
        )
        players = (
            _player(),
            _player(melds=(pon,)),
            _player(melds=(pon, ankan)),
            _player(),
        )
        # dealer SEAT_2 = 東、SEAT_3 = 南、SEAT_0（self）= 西、SEAT_1 = 北。
        policy_input = _input(
            "123m456p789s1234z5z", players=players, dealer_seat=Seat.SEAT_2
        )
        self.assertEqual(
            _opponent_concealed_slot_counts_by_wind(policy_input), (7, 13, 0, 10)
        )

    def test_more_melds_than_a_hand_can_hold_fails_closed(self) -> None:
        ankan = PublicMeld(
            kind=MeldKind.ANKAN, tiles=_hand("9999m"), from_seat=None, called_tile=None
        )
        players = (_player(), _player(melds=(ankan,) * 5), _player(), _player())
        with self.assertRaises(Kobalab0004ReferencePolicyError):
            _opponent_concealed_slot_counts_by_wind(_input("123m", players=players))


class BeliefPaijiaInputTest(unittest.TestCase):
    """Belief由来のpaijia入力 = 未見枚数 − 他家3人の手牌内期待枚数。"""

    def test_uniform_residual_does_not_subtract_self_or_double_count_red(
        self,
    ) -> None:
        policy_input = _input("05m1234567z123p45s", dora_indicators="3p")
        counts = _PublicCounts(policy_input)
        belief = reference_module._estimate_concealed_hand_belief(
            policy_input, counts.conservation
        )
        residual = _paijia_input_from_belief(policy_input, counts)
        self_wind = wind_for_seat(
            policy_input.self_seat, policy_input.round.dealer_seat
        )
        opponents = [
            hand
            for number, hand in enumerate(belief.hands)
            if number != wind_index(self_wind)
        ]
        conservation = counts.conservation
        for index in range(34):
            self.assertEqual(
                residual._counts[index],
                conservation.remaining_tile_counts[index] * SCALE
                - sum(hand.expected_count_raw[index] for hand in opponents),
            )
        for color in range(3):
            self.assertEqual(
                residual._red[color],
                conservation.remaining_red_five_counts[color] * SCALE
                - sum(hand.red_five_probability_raw[color] for hand in opponents),
            )
        # 他家3人のslotは39枚。残余は136 - 14（自手）- 1（ドラ表示牌）- 39 = 82枚分で、
        # 推定器はphysical tile pool（34牌種 + 赤5 3色）ごとに丸めるため合計は
        # pool数以内のraw unitだけずれ得る。
        self.assertLessEqual(abs(sum(residual._counts) - 82 * SCALE), 37)

    def test_without_opponent_mass_belief_paijia_is_reference_times_scale(
        self,
    ) -> None:
        concealed = "1239m456p789s1167z"
        policy_input = _input(concealed, "7z")
        with mock.patch.object(
            reference_module,
            "_estimate_concealed_hand_belief",
            lambda pi, conservation: _belief_with_opponent_mass(pi, {}),
        ):
            counts = _PublicCounts(policy_input)
            residual = _paijia_input_from_belief(policy_input, counts)
            for tile in _hand(concealed):
                self.assertEqual(residual.paijia(tile), counts.paijia(tile) * SCALE)
            actions = _all_discards(concealed, "7z")
            self.assertEqual(
                _choose_with(Kobalab0004BeliefPaijiaPolicy(), policy_input, actions),
                _choose(policy_input, actions),
            )

    def test_non_uniform_belief_changes_paijia_order_but_not_ukeire(self) -> None:
        # 6z / 7zはどちらを切ってもukeireが同じ。参照版はpaijia同点
        # （白・中とも未見3枚）でsource順の7zを切る。他家が6zを2枚持つbeliefでは
        # 6zの残余が1枚分になりpaijiaが下がるため、6zを先に評価して切る。
        concealed = "1239m456p789s1167z"
        policy_input = _input(concealed)
        actions = _all_discards(concealed)
        self.assertEqual(_choose(policy_input, actions), _discard("7z"))
        belief = _belief_with_opponent_mass(
            policy_input, {(Seat.SEAT_2, "6z"): 2 * SCALE}
        )
        with mock.patch.object(
            reference_module,
            "_estimate_concealed_hand_belief",
            lambda pi, conservation: belief,
        ):
            residual = _paijia_input_from_belief(
                policy_input, _PublicCounts(policy_input)
            )
            self.assertEqual(
                residual._counts[tile_type_index(_t("6z").tile_type)], SCALE
            )
            self.assertLess(residual.paijia(_t("6z")), residual.paijia(_t("7z")))
            self.assertEqual(
                _choose_with(Kobalab0004BeliefPaijiaPolicy(), policy_input, actions),
                _discard("6z"),
            )

    def test_belief_exceeding_remaining_mass_fails_closed(self) -> None:
        concealed = "1239m456p789s1167z"
        policy_input = _input(concealed)
        belief = _belief_with_opponent_mass(
            policy_input, {(Seat.SEAT_1, "6z"): 4 * SCALE}
        )
        with mock.patch.object(
            reference_module,
            "_estimate_concealed_hand_belief",
            lambda pi, conservation: belief,
        ):
            with self.assertRaises(Kobalab0004ReferencePolicyError):
                _choose_with(
                    Kobalab0004BeliefPaijiaPolicy(),
                    policy_input,
                    _all_discards(concealed),
                )

    def test_non_discard_rules_are_unchanged(self) -> None:
        concealed = "123456789m11p24s1z"
        actions = (RiichiAction(actor=Seat.SEAT_0),) + _all_discards(concealed)
        self.assertEqual(
            _choose_with(Kobalab0004BeliefPaijiaPolicy(), _input(concealed), actions),
            RiichiAction(actor=Seat.SEAT_0),
        )
        self.assertEqual(
            _choose_with(
                Kobalab0004BeliefPaijiaPolicy(),
                _input("123m"),
                (PassAction(actor=Seat.SEAT_0),),
            ),
            PassAction(actor=Seat.SEAT_0),
        )


class BeliefInventorySharingTest(unittest.TestCase):
    """Issue #220: 未見枚数はdecisionごとに1回だけ導出し、推定器と残余で共有する。"""

    def test_inventory_is_derived_once_per_belief_discard_decision(self) -> None:
        concealed = "05m1234567z123p45s"
        policy_input = _input(concealed, dora_indicators="3p")
        with (
            mock.patch.object(
                reference_module,
                "derive_remaining_tile_inventory",
                wraps=reference_module.derive_remaining_tile_inventory,
            ) as policy_side,
            mock.patch.object(
                uniform_module,
                "derive_remaining_tile_inventory",
                wraps=uniform_module.derive_remaining_tile_inventory,
            ) as estimator_side,
        ):
            _choose_with(
                Kobalab0004BeliefPaijiaPolicy(),
                policy_input,
                _all_discards(concealed),
            )
        self.assertEqual(policy_side.call_count + estimator_side.call_count, 1)

    def test_consecutive_decisions_use_their_own_snapshot(self) -> None:
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=_hand("555z"),
            from_seat=Seat.SEAT_0,
            called_tile=_t("5z"),
        )
        first = _input("1239m456p789s1167z")
        second = _input(
            "1239m456p789s1167z",
            players=(
                _player(discards=_discards("5z", called_by=Seat.SEAT_2)),
                _player(discards=_discards("6z7z")),
                _player(melds=(pon,)),
                _player(),
            ),
            dealer_seat=Seat.SEAT_1,
        )
        actions = _all_discards("1239m456p789s1167z")
        policy = Kobalab0004BeliefPaijiaPolicy()
        with mock.patch.object(
            reference_module,
            "_estimate_from_conservation",
            wraps=reference_module._estimate_from_conservation,
        ) as estimator:
            for policy_input in (first, second, first):
                self.assertEqual(
                    _choose_with(policy, policy_input, actions),
                    _choose_with(
                        Kobalab0004BeliefPaijiaPolicy(), policy_input, actions
                    ),
                )
        received = [call.args[:2] for call in estimator.call_args_list]
        self.assertEqual(len(received), 6)
        for (policy_input, conservation), expected_input in zip(
            received, (first, first, second, second, first, first), strict=True
        ):
            self.assertIs(policy_input, expected_input)
            self.assertEqual(
                conservation, derive_remaining_tile_inventory(expected_input)
            )
        self.assertNotEqual(received[0][1], received[2][1])


class InformationBoundaryTest(unittest.TestCase):
    def test_module_imports_only_lisjong_and_stdlib(self) -> None:
        import lisjong.policies.kobalab_0004_reference as module

        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertLessEqual(imported, {"lisjong", "collections"})


if __name__ == "__main__":
    unittest.main()
