"""Issue #211 `Kobalab0004ReferencePolicy`のsource-semantic unit test。

期待値はkobalab/majiang-ai legacy 0004（commit
e75a9720a12b84c03e6c61c3960c1844b8982eb4）の`player-0004.js` /
`suanpai-0004.js`の式を手で適用して求めたmanual golden valueである。
Policy実装を呼んで期待値を作る自己検証は避ける。upstream実行結果との差分検証は
`test_kobalab_0004_upstream_differential.py`が担当する。
"""

import ast
import itertools
import unittest
from pathlib import Path

from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies import Kobalab0004ReferencePolicy
from lisjong.policies.kobalab_0004_reference import (
    KOBALAB_0004_REFERENCE_IDENTITY,
    Kobalab0004ReferencePolicyError,
    _allows_riichi_discard,
    _improving_tile_types,
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
