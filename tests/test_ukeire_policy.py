"""Issue #52 `UkeirePolicy`のunit test。

`calculate_shanten()`が返す既知の値と、麻雀の牌姿として説明できる待ちの形を
根拠として、discard候補ごとの打牌後向聴数・有効牌・受け入れ枚数を人が読める形で
固定する。Policyの選択ロジック自体をtest側へ複製した自己検証は避ける。

固定牌姿の中心は次の2つである。

`_NINE_MANZU_HAND`
    123456789m + 111p23p の14枚。111p23p は「11p雀頭 + 123p面子」として
    使えるので、萬子側から1枚切っても3面子 + 雀頭 + 塔子1つの聴牌が残る。
    どの萬子を切るかで残る塔子の形（嵌張・両面）が変わるため、向聴数が同じ
    ままで受け入れ枚数だけが変わる。

`_TANKI_VERSUS_ONE_SHANTEN_HAND`
    234m567m234p567p + 7z + 5s の14枚。4面子が既にあり、浮き牌2枚のうち
    片方を切れば単騎聴牌になる。面子から1枚切ると1向聴へ落ちる代わりに
    受け入れ枚数は大きくなるので、「向聴数を受け入れ枚数で逆転させない」
    優先順位をそのまま固定できる。
"""

import itertools
import unittest

from lisjong.policies import ShantenPolicy, UkeirePolicy
from lisjong.policies.ukeire import (
    UkeirePolicyError,
    _effective_tile_types,
    _known_tile_counts,
    _remove_one_matching_tile,
    _ukeire_count,
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


def _tile(category: TileCategory, rank: int, *, red: bool = False) -> Tile:
    return Tile(TileType(category, rank), is_red=red)


def _hand(spec: str) -> tuple[Tile, ...]:
    """`"123m11p"`のような簡潔な表記を`Tile` tupleへ展開するtest helper。

    `m` / `p` / `s` / `z`はそれぞれ萬子・筒子・索子・字牌を表し、字牌rankは
    lisjong内部契約どおり東=1、南=2、西=3、北=4、白=5、發=6、中=7である。
    赤5は表記に含めず、必要なtestが個別のTileを直接組み立てる。
    """
    categories = {
        "m": TileCategory.MANZU,
        "p": TileCategory.PINZU,
        "s": TileCategory.SOUZU,
        "z": TileCategory.HONOR,
    }
    tiles: list[Tile] = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        category = categories[character]
        tiles.extend(_tile(category, int(rank)) for rank in ranks)
        ranks = ""
    if ranks:
        raise ValueError(f"hand spec has trailing ranks without a suit: {spec!r}")
    return tuple(tiles)


MANZU_1 = _tile(TileCategory.MANZU, 1)
MANZU_2 = _tile(TileCategory.MANZU, 2)
MANZU_3 = _tile(TileCategory.MANZU, 3)
MANZU_4 = _tile(TileCategory.MANZU, 4)
MANZU_5 = _tile(TileCategory.MANZU, 5)
MANZU_5_RED = _tile(TileCategory.MANZU, 5, red=True)
MANZU_6 = _tile(TileCategory.MANZU, 6)
MANZU_7 = _tile(TileCategory.MANZU, 7)
PINZU_1 = _tile(TileCategory.PINZU, 1)
PINZU_5 = _tile(TileCategory.PINZU, 5)
PINZU_5_RED = _tile(TileCategory.PINZU, 5, red=True)
SOUZU_5 = _tile(TileCategory.SOUZU, 5)
EAST = _tile(TileCategory.HONOR, 1)
SOUTH = _tile(TileCategory.HONOR, 2)
RED_DRAGON = _tile(TileCategory.HONOR, 7)

MANZU_5_TYPE = TileType(TileCategory.MANZU, 5)

_NINE_MANZU_HAND = _hand("123456789m111p23p")
"""123456789m + 111p23p の14枚。萬子側の切り方で受け入れ枚数だけが変わる。"""

_TANKI_VERSUS_ONE_SHANTEN_HAND = _hand("234m567m234p567p5s7z")
"""4面子 + 浮き牌2枚の14枚。聴牌維持と、受け入れの広い1向聴を比較できる。"""

_SEVEN_PAIRS_HAND = _hand("1133557799m11p3p")
"""六対子 + 3p の13枚。通常形・国士では説明できず、七対子でだけ聴牌になる。"""

_THIRTEEN_ORPHANS_HAND = _hand("19m19p19s1234567z")
"""国士無双13面待ちの13枚。"""


def _player(
    discards: tuple[Discard, ...] = (),
    melds: tuple[PublicMeld, ...] = (),
    score: int = 25000,
) -> PlayerPublicState:
    return PlayerPublicState(
        score=score, discards=discards, melds=melds, riichi=RiichiState.NONE
    )


def _discard_history(
    tiles: tuple[Tile, ...],
    *,
    called_by: Seat | None = None,
    first_order: int = 0,
) -> tuple[Discard, ...]:
    return tuple(
        Discard(
            tile=tile, tsumogiri=False, order=first_order + index, called_by=called_by
        )
        for index, tile in enumerate(tiles)
    )


def _make_input(
    concealed_tiles: tuple[Tile, ...] = (),
    drawn_tile: Tile | None = None,
    *,
    self_seat: Seat = Seat.SEAT_0,
    players: tuple[PlayerPublicState, ...] | None = None,
    dora_indicators: tuple[Tile, ...] = (SOUTH,),
) -> PolicyInput:
    """既定では、公開情報が受け入れ計算へ影響しないPolicyInputを作る。

    既定のdora indicatorは南（2z）で、固定牌姿のどれとも重ならないため、
    既知牌countを検証しないtestが表示牌の存在を意識しなくてよい。
    """
    return PolicyInput(
        self_seat=self_seat,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=dora_indicators,
            live_wall_tiles_remaining=70,
        ),
        players=players if players is not None else (_player(),) * 4,
        own_hand=OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=drawn_tile),
    )


def _decision(
    concealed_tiles: tuple[Tile, ...],
    actions: tuple[object, ...],
    drawn_tile: Tile | None = None,
    *,
    players: tuple[PlayerPublicState, ...] | None = None,
    dora_indicators: tuple[Tile, ...] = (SOUTH,),
) -> DecisionContext:
    return DecisionContext(
        input=_make_input(
            concealed_tiles,
            drawn_tile,
            players=players,
            dora_indicators=dora_indicators,
        ),
        legal_actions=actions,
    )


def _discard(tile: Tile, *, tsumogiri: bool = False) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=tsumogiri)


class DiscardSimulationTest(unittest.TestCase):
    """`_remove_one_matching_tile()`が、同一semantic Tileでも1枚だけ除くことを確認する。"""

    def test_removes_only_one_copy_from_multiple_identical_tiles(self) -> None:
        concealed = (MANZU_1, MANZU_1, MANZU_1, MANZU_2)

        remaining = _remove_one_matching_tile(concealed, MANZU_1)

        self.assertEqual(len(remaining), 3)
        self.assertEqual(remaining.count(MANZU_1), 2)
        self.assertEqual(remaining.count(MANZU_2), 1)

    def test_discarding_normal_five_removes_only_the_normal_copy(self) -> None:
        concealed = (MANZU_5, MANZU_5, MANZU_5, MANZU_5_RED, MANZU_2)

        remaining = _remove_one_matching_tile(concealed, MANZU_5)

        self.assertEqual(len(remaining), 4)
        self.assertEqual(remaining.count(MANZU_5), 2)
        self.assertEqual(remaining.count(MANZU_5_RED), 1)

    def test_discarding_red_five_removes_only_the_red_copy(self) -> None:
        concealed = (MANZU_5, MANZU_5, MANZU_5, MANZU_5_RED, MANZU_2)

        remaining = _remove_one_matching_tile(concealed, MANZU_5_RED)

        self.assertEqual(len(remaining), 4)
        self.assertEqual(remaining.count(MANZU_5), 3)
        self.assertEqual(remaining.count(MANZU_5_RED), 0)

    def test_missing_tile_fails_closed(self) -> None:
        concealed = (MANZU_1, MANZU_2)

        with self.assertRaises(UkeirePolicyError):
            _remove_one_matching_tile(concealed, MANZU_6)


class EffectiveTileTypeTest(unittest.TestCase):
    """有効牌判定が34基礎牌種単位で、`calculate_shanten()`だけを根拠にすることを確認する。"""

    def test_ryanmen_accepts_both_ends(self) -> None:
        # 123m + 56m + 789m + 11p + 123p。両面56mが4m/7mを受ける。
        hand = _hand("12356789m11123p")

        self.assertEqual(
            _effective_tile_types(hand),
            (TileType(TileCategory.MANZU, 4), TileType(TileCategory.MANZU, 7)),
        )

    def test_kanchan_accepts_only_the_middle_tile(self) -> None:
        # 13m + 456m + 789m + 11p + 123p。嵌張13mは2mだけを受ける。
        hand = _hand("13456789m11123p")

        self.assertEqual(
            _effective_tile_types(hand), (TileType(TileCategory.MANZU, 2),)
        )

    def test_seven_pairs_shape_is_detected_through_calculate_shanten(self) -> None:
        """七対子由来の有効牌。通常形・国士では説明できない固定ケース。

        1133557799m + 11p + 3pは対子6つ + 浮き牌1枚で、七対子として0向聴、
        通常形では3向聴、国士では9向聴になる。有効牌が3pだけになるのは
        七対子の向聴数が改善する場合だけであり、Policy側へ七対子専用の
        待ち判定を持たなくても`calculate_shanten()`だけで判定できる。
        """
        self.assertEqual(
            _effective_tile_types(_SEVEN_PAIRS_HAND),
            (TileType(TileCategory.PINZU, 3),),
        )

    def test_thirteen_orphans_shape_is_detected_through_calculate_shanten(self) -> None:
        """国士無双由来の有効牌。13面待ちの么九牌13種すべてを受ける。"""
        expected = tuple(
            TileType(category, rank)
            for category in (
                TileCategory.MANZU,
                TileCategory.PINZU,
                TileCategory.SOUZU,
            )
            for rank in (1, 9)
        ) + tuple(TileType(TileCategory.HONOR, rank) for rank in range(1, 8))

        self.assertEqual(
            sorted(_effective_tile_types(_THIRTEEN_ORPHANS_HAND), key=repr),
            sorted(expected, key=repr),
        )

    def test_tile_type_already_held_four_times_is_excluded(self) -> None:
        """打牌後手牌だけで4枚ある基礎牌種は、5枚目を試さず候補から外す。

        `calculate_shanten()`は同一基礎牌種5枚以上をinvalid inputとして
        fail closedするため、除外しなければValueErrorになる。赤5と通常5は
        同じ基礎牌種として合算するので、通常5m3枚 + 赤5m1枚でも4枚使いである。
        """
        hand = (MANZU_5, MANZU_5, MANZU_5, MANZU_5_RED)

        effective = _effective_tile_types(hand)

        self.assertNotIn(MANZU_5_TYPE, effective)

    def test_red_five_is_not_a_separate_effective_tile_type(self) -> None:
        """赤5は独立した有効牌種にならず、通常5と同じ基礎牌種へ集約される。

        123m456m789m + 55p + 67pは5p/8p待ちの聴牌である。雀頭の5pを1枚
        赤5へ差し替えても、有効牌種は同じ`5p` / `8p`の2種のままで、赤5と通常5が
        別々の有効牌種として2重に数えられることはない。
        """
        normal_hand = _hand("123456789m55p67p")
        red_hand = _hand("123456789m5p67p") + (PINZU_5_RED,)
        expected = (
            TileType(TileCategory.PINZU, 5),
            TileType(TileCategory.PINZU, 8),
        )

        self.assertEqual(_effective_tile_types(normal_hand), expected)
        self.assertEqual(_effective_tile_types(red_hand), expected)


class KnownTileCountTest(unittest.TestCase):
    """既知牌countが、自手・河・meld・dora indicatorをすべて反映することを確認する。"""

    def test_own_concealed_tiles_are_counted(self) -> None:
        policy_input = _make_input((MANZU_5, MANZU_5), dora_indicators=())

        counts = _known_tile_counts(policy_input)

        self.assertEqual(counts[MANZU_5_TYPE], 2)

    def test_drawn_tile_is_not_counted_twice(self) -> None:
        """`drawn_tile`は`concealed_tiles`に含まれる契約なので別枚として数えない。"""
        policy_input = _make_input(
            (MANZU_5, MANZU_5), drawn_tile=MANZU_5, dora_indicators=()
        )

        counts = _known_tile_counts(policy_input)

        self.assertEqual(counts[MANZU_5_TYPE], 2)

    def test_discards_of_every_seat_are_counted(self) -> None:
        players = (
            _player(),
            _player(discards=_discard_history((MANZU_5,))),
            _player(discards=_discard_history((MANZU_5,), first_order=1)),
            _player(),
        )
        policy_input = _make_input((), players=players, dora_indicators=())

        counts = _known_tile_counts(policy_input)

        self.assertEqual(counts[MANZU_5_TYPE], 2)

    def test_melds_of_every_seat_are_counted(self) -> None:
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(MANZU_5, MANZU_5, MANZU_5),
            from_seat=Seat.SEAT_0,
            called_tile=MANZU_5,
        )
        players = (_player(), _player(melds=(pon,)), _player(), _player())
        policy_input = _make_input((), players=players, dora_indicators=())

        counts = _known_tile_counts(policy_input)

        self.assertEqual(counts[MANZU_5_TYPE], 3)

    def test_dora_indicators_are_counted(self) -> None:
        policy_input = _make_input((), dora_indicators=(MANZU_5, MANZU_5))

        counts = _known_tile_counts(policy_input)

        self.assertEqual(counts[MANZU_5_TYPE], 2)

    def test_red_five_and_normal_five_are_aggregated_into_one_tile_type(self) -> None:
        """赤5と通常5は、既知牌countでも同じ基礎牌種として合算する。"""
        players = (
            _player(),
            _player(discards=_discard_history((MANZU_5_RED,))),
            _player(),
            _player(),
        )
        policy_input = _make_input(
            (MANZU_5,), players=players, dora_indicators=(MANZU_5_RED,)
        )

        counts = _known_tile_counts(policy_input)

        self.assertEqual(counts[MANZU_5_TYPE], 3)

    def test_called_discard_is_counted_once_on_the_meld_side_only(self) -> None:
        """鳴かれた捨て牌を`discard + meld`として二重計上しない。

        seat 1が5mを捨て、seat 2がponした状態を表す。`Discard`履歴は鳴かれた
        牌も削除せず`called_by`を保持し、同じ物理牌相当が`PublicMeld.tiles`
        にも現れるため、単純加算すると5mを4枚として数えてしまう。
        """
        called = _discard_history((MANZU_5,), called_by=Seat.SEAT_2)
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(MANZU_5, MANZU_5, MANZU_5),
            from_seat=Seat.SEAT_1,
            called_tile=MANZU_5,
        )
        players = (
            _player(),
            _player(discards=called),
            _player(melds=(pon,)),
            _player(),
        )
        policy_input = _make_input((), players=players, dora_indicators=())

        counts = _known_tile_counts(policy_input)

        self.assertEqual(counts[MANZU_5_TYPE], 3)

    def test_uncalled_discard_of_the_same_tile_type_is_still_counted(self) -> None:
        """`called_by`を持たない捨て牌は、meldと無関係に既知牌として数える。"""
        history = _discard_history(
            (MANZU_5,), called_by=Seat.SEAT_2
        ) + _discard_history((MANZU_5,), first_order=1)
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(MANZU_5, MANZU_5, MANZU_5),
            from_seat=Seat.SEAT_1,
            called_tile=MANZU_5,
        )
        players = (
            _player(),
            _player(discards=history),
            _player(melds=(pon,)),
            _player(),
        )
        policy_input = _make_input((), players=players, dora_indicators=())

        counts = _known_tile_counts(policy_input)

        self.assertEqual(counts[MANZU_5_TYPE], 4)

    def test_known_count_above_four_fails_closed(self) -> None:
        """4へ丸めず、負の未見枚数も作らず、推測で修復もせずfail closedする。"""
        players = (
            _player(),
            _player(discards=_discard_history((MANZU_5, MANZU_5, MANZU_5))),
            _player(),
            _player(),
        )
        policy_input = _make_input(
            (MANZU_5, MANZU_5), players=players, dora_indicators=()
        )

        with self.assertRaises(UkeirePolicyError):
            _known_tile_counts(policy_input)

    def test_known_count_above_four_across_red_and_normal_five_fails_closed(
        self,
    ) -> None:
        """4枚上限判定も、赤5と通常5を合算した基礎牌種単位で行う。"""
        players = (
            _player(),
            _player(discards=_discard_history((MANZU_5_RED, MANZU_5, MANZU_5))),
            _player(),
            _player(),
        )
        policy_input = _make_input(
            (MANZU_5, MANZU_5), players=players, dora_indicators=()
        )

        with self.assertRaises(UkeirePolicyError):
            _known_tile_counts(policy_input)

    def test_known_count_is_independent_of_meld_and_discard_ordering(self) -> None:
        chi = PublicMeld(
            kind=MeldKind.CHI,
            tiles=(MANZU_4, MANZU_5, MANZU_6),
            from_seat=Seat.SEAT_3,
            called_tile=MANZU_5,
        )
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(PINZU_5, PINZU_5, PINZU_5),
            from_seat=Seat.SEAT_1,
            called_tile=PINZU_5,
        )
        history = _discard_history((MANZU_1, MANZU_2, MANZU_3))

        baseline = _known_tile_counts(
            _make_input(
                (),
                players=(
                    _player(),
                    _player(discards=history, melds=(chi, pon)),
                    _player(),
                    _player(),
                ),
                dora_indicators=(),
            )
        )
        reordered = _known_tile_counts(
            _make_input(
                (),
                players=(
                    _player(),
                    _player(discards=tuple(reversed(history)), melds=(pon, chi)),
                    _player(),
                    _player(),
                ),
                dora_indicators=(),
            )
        )

        self.assertEqual(baseline, reordered)


class UkeireCountTest(unittest.TestCase):
    """受け入れ枚数が「初期4枚 - 既知牌枚数」の合計であることを確認する。"""

    def test_counts_unseen_copies_of_every_effective_tile_type(self) -> None:
        # 両面56mが4m/7mを受ける。手牌に4mと7mが1枚ずつあるので、
        # 未見枚数はそれぞれ3枚、合計6枚になる。
        hand = _hand("12356789m11123p")
        known = _known_tile_counts(_make_input(_NINE_MANZU_HAND, dora_indicators=()))

        self.assertEqual(_ukeire_count(hand, known), 6)

    def test_known_tiles_reduce_the_unseen_count(self) -> None:
        hand = _hand("12356789m11123p")
        players = (
            _player(),
            _player(discards=_discard_history((MANZU_4, MANZU_4, MANZU_7))),
            _player(),
            _player(),
        )
        known = _known_tile_counts(
            _make_input(_NINE_MANZU_HAND, players=players, dora_indicators=())
        )

        # 4m: 4 - (手牌1 + 河2) = 1枚、7m: 4 - (手牌1 + 河1) = 2枚。
        self.assertEqual(_ukeire_count(hand, known), 3)

    def test_effective_tile_type_with_no_unseen_copy_contributes_zero(self) -> None:
        """既知4枚の有効牌種は、有効牌のまま未見枚数0として数える。

        「打牌後手牌だけで4枚あるので構造上5枚目を試せない」場合と、
        「構造上は有効牌だが場に見えていない枚数が0」の場合を混同しない。
        """
        hand = _hand("13456789m11123p")
        players = (
            _player(),
            _player(discards=_discard_history((MANZU_2, MANZU_2, MANZU_2))),
            _player(),
            _player(),
        )
        known = _known_tile_counts(
            _make_input(_NINE_MANZU_HAND, players=players, dora_indicators=())
        )

        self.assertEqual(
            _effective_tile_types(hand), (TileType(TileCategory.MANZU, 2),)
        )
        self.assertEqual(_ukeire_count(hand, known), 0)


class UkeirePolicyDiscardSelectionTest(unittest.TestCase):
    """向聴数 > 受け入れ枚数 > tie-breakの優先順位を固定する。"""

    def setUp(self) -> None:
        self.policy = UkeirePolicy()

    def test_prefers_the_discard_with_more_ukeire_at_the_same_shanten(self) -> None:
        """同じ0向聴でも、嵌張(2m受け)より両面(4m/7m受け)を残す。

        2m切り: 13m + 456m + 789m + 11p + 123p -> 嵌張2m待ち、未見3枚
        4m切り: 123m + 56m + 789m + 11p + 123p -> 両面4m/7m待ち、未見6枚
        """
        discard_2m = _discard(MANZU_2)
        discard_4m = _discard(MANZU_4)

        for permutation in itertools.permutations((discard_2m, discard_4m)):
            with self.subTest(order=[action.tile for action in permutation]):
                decision = _decision(_NINE_MANZU_HAND, permutation)
                self.assertEqual(self.policy.choose_action(decision), discard_4m)

    def test_shanten_policy_keeps_choosing_the_smaller_tile_sort_key(self) -> None:
        """同じ入力で`ShantenPolicy`が#51時点の選択を保つことを確認する。

        `ShantenPolicy`は同向聴を`tile_sort_key()`だけでtie-breakするので
        2m切りを選び、`UkeirePolicy`は受け入れ枚数で4m切りを選ぶ。両者が
        別世代のPolicyとして併存していることが、この差で確認できる。
        """
        discard_2m = _discard(MANZU_2)
        discard_4m = _discard(MANZU_4)
        decision = _decision(_NINE_MANZU_HAND, (discard_2m, discard_4m))

        self.assertEqual(ShantenPolicy().choose_action(decision), discard_2m)
        self.assertEqual(UkeirePolicy().choose_action(decision), discard_4m)

    def test_does_not_prefer_a_worse_shanten_candidate_with_larger_ukeire(self) -> None:
        """向聴数が悪い候補を、受け入れ枚数が大きいだけで逆転させない。

        5s切りは4面子 + 7z単騎の聴牌(受け入れ3枚)、2m切りは1向聴だが
        受け入れ16枚である。受け入れが5倍以上でも、最小向聴の5s切りを選ぶ。
        """
        tenpai_discard = _discard(SOUZU_5)
        wide_one_shanten_discard = _discard(MANZU_2)
        known = _known_tile_counts(
            _make_input(_TANKI_VERSUS_ONE_SHANTEN_HAND, dora_indicators=())
        )
        concealed = _TANKI_VERSUS_ONE_SHANTEN_HAND

        tenpai_ukeire = _ukeire_count(
            _remove_one_matching_tile(concealed, SOUZU_5), known
        )
        one_shanten_ukeire = _ukeire_count(
            _remove_one_matching_tile(concealed, MANZU_2), known
        )
        self.assertGreater(one_shanten_ukeire, tenpai_ukeire)

        for permutation in itertools.permutations(
            (tenpai_discard, wide_one_shanten_discard)
        ):
            with self.subTest(order=[action.tile for action in permutation]):
                decision = _decision(concealed, permutation)
                self.assertEqual(self.policy.choose_action(decision), tenpai_discard)

    def test_tie_break_prefers_the_smaller_tile_sort_key_when_ukeire_ties(self) -> None:
        """向聴数・受け入れ枚数まで同値なら`tile_sort_key()`で決定的に選ぶ。

        2m切り(嵌張2m待ち)と3m切り(嵌張3m待ち)は、どちらも0向聴・未見3枚で
        完全にtieする。
        """
        discard_2m = _discard(MANZU_2)
        discard_3m = _discard(MANZU_3)

        for permutation in itertools.permutations((discard_2m, discard_3m)):
            with self.subTest(order=[action.tile for action in permutation]):
                decision = _decision(_NINE_MANZU_HAND, permutation)
                self.assertEqual(self.policy.choose_action(decision), discard_2m)

    def test_tie_break_prefers_tedashi_over_tsumogiri_for_the_same_tile(self) -> None:
        tedashi = _discard(MANZU_2, tsumogiri=False)
        tsumogiri = _discard(MANZU_2, tsumogiri=True)

        for permutation in itertools.permutations((tedashi, tsumogiri)):
            decision = _decision(_NINE_MANZU_HAND, permutation, drawn_tile=MANZU_2)
            self.assertEqual(self.policy.choose_action(decision), tedashi)

    def test_tie_break_prefers_normal_five_over_red_five(self) -> None:
        """Action identityでは赤5と通常5を区別し続ける。

        555m + 赤5m + 2mでは、通常5切りと赤5切りで残る牌姿が同じなので
        向聴数・受け入れともtieし、`tile_sort_key()`のis_red比較で通常5切りが
        選ばれる。両者が同じActionへ潰れないことも同時に確認する。
        """
        concealed = (MANZU_5, MANZU_5, MANZU_5, MANZU_5_RED, MANZU_2)
        discard_normal = _discard(MANZU_5)
        discard_red = _discard(MANZU_5_RED)

        self.assertNotEqual(discard_normal, discard_red)

        for permutation in itertools.permutations((discard_normal, discard_red)):
            decision = _decision(concealed, permutation, drawn_tile=MANZU_2)
            self.assertEqual(self.policy.choose_action(decision), discard_normal)

    def test_single_discard_candidate_is_returned(self) -> None:
        only_discard = _discard(MANZU_2)
        decision = _decision(_NINE_MANZU_HAND, (only_discard,))

        self.assertEqual(self.policy.choose_action(decision), only_discard)

    def test_discard_tile_missing_from_concealed_tiles_fails_closed(self) -> None:
        inconsistent = _discard(RED_DRAGON)
        decision = _decision(_NINE_MANZU_HAND, (inconsistent,))

        with self.assertRaises(UkeirePolicyError):
            self.policy.choose_action(decision)

    def test_inconsistent_known_tile_count_fails_closed_on_a_discard_decision(
        self,
    ) -> None:
        """受け入れ計算の境界で、意味契約と整合しない既知牌countを検出する。

        自手に1pが3枚あり、他家の河にさらに2枚ある状態は基礎牌種5枚であり、
        `PolicyInput`として成立しない。
        """
        players = (
            _player(),
            _player(discards=_discard_history((PINZU_1, PINZU_1))),
            _player(),
            _player(),
        )
        decision = _decision(
            _NINE_MANZU_HAND,
            (_discard(MANZU_2), _discard(MANZU_4)),
            players=players,
            dora_indicators=(),
        )

        with self.assertRaises(UkeirePolicyError):
            self.policy.choose_action(decision)


class UkeirePolicyVisibleInformationTest(unittest.TestCase):
    """自手・河・meld・dora indicatorが、実際に選択結果へ効くことを確認する。

    どのtestも「両面4m/7m待ちの4m切り(既定では未見6枚)」と
    「嵌張2m待ちの2m切り(未見3枚)」を比較し、公開情報で4m/7mの未見枚数が
    減ると選択が2m切りへ入れ替わることを固定する。
    """

    def setUp(self) -> None:
        self.policy = UkeirePolicy()
        self.discard_2m = _discard(MANZU_2)
        self.discard_4m = _discard(MANZU_4)
        self.actions = (self.discard_2m, self.discard_4m)

    def test_baseline_prefers_the_ryanmen_discard(self) -> None:
        decision = _decision(_NINE_MANZU_HAND, self.actions, dora_indicators=())

        self.assertEqual(self.policy.choose_action(decision), self.discard_4m)

    def test_discards_reduce_the_unseen_count_and_flip_the_choice(self) -> None:
        players = (
            _player(),
            _player(discards=_discard_history((MANZU_4, MANZU_4, MANZU_7))),
            _player(),
            _player(),
        )
        decision = _decision(
            _NINE_MANZU_HAND, self.actions, players=players, dora_indicators=()
        )

        # 4m: 未見1枚、7m: 未見2枚 -> 合計3枚で、嵌張2m待ちの3枚とtieし、
        # tile_sort_keyにより2m切りが選ばれる。
        self.assertEqual(self.policy.choose_action(decision), self.discard_2m)

    def test_melds_reduce_the_unseen_count_and_flip_the_choice(self) -> None:
        pon_4m = PublicMeld(
            kind=MeldKind.PON,
            tiles=(MANZU_4, MANZU_4, MANZU_4),
            from_seat=Seat.SEAT_1,
            called_tile=MANZU_4,
        )
        players = (_player(), _player(), _player(melds=(pon_4m,)), _player())
        decision = _decision(
            _NINE_MANZU_HAND, self.actions, players=players, dora_indicators=()
        )

        # 4m: 未見0枚、7m: 未見3枚 -> 合計3枚。
        self.assertEqual(self.policy.choose_action(decision), self.discard_2m)

    def test_dora_indicators_reduce_the_unseen_count_and_flip_the_choice(self) -> None:
        decision = _decision(
            _NINE_MANZU_HAND,
            self.actions,
            dora_indicators=(MANZU_4, MANZU_4, MANZU_7, MANZU_7),
        )

        # 4m: 未見1枚、7m: 未見1枚 -> 合計2枚で、嵌張2m待ちの3枚を下回る。
        self.assertEqual(self.policy.choose_action(decision), self.discard_2m)

    def test_own_concealed_tiles_reduce_the_unseen_count(self) -> None:
        """自手の牌が、その牌種の未見枚数を減らす。

        4m切り後の両面待ちが受ける4m / 7mは、自手にも1枚ずつ残っている。
        既知牌を一切数えなければ受け入れは4 + 4 = 8枚だが、自手のぶんを
        差し引くと3 + 3 = 6枚になる。
        """
        hand_after_discard = _hand("12356789m11123p")
        known = _known_tile_counts(_make_input(_NINE_MANZU_HAND, dora_indicators=()))

        self.assertEqual(known[TileType(TileCategory.MANZU, 4)], 1)
        self.assertEqual(known[TileType(TileCategory.MANZU, 7)], 1)
        self.assertEqual(_ukeire_count(hand_after_discard, {}), 8)
        self.assertEqual(_ukeire_count(hand_after_discard, known), 6)

    def test_called_discard_is_not_counted_on_both_the_discard_and_the_meld(
        self,
    ) -> None:
        """鳴かれた4mを二重計上すると、未見枚数が1枚少なくなってしまう。

        seat 1が4mを1枚捨て、seat 2がそれを含めてponした状態を作る。
        4mの既知枚数は「自手1 + meld 3」の4枚であり、鳴かれたdiscardを
        重ねて数えると5枚になってPolicyInput契約と矛盾する。
        """
        called = _discard_history((MANZU_4,), called_by=Seat.SEAT_2)
        pon_4m = PublicMeld(
            kind=MeldKind.PON,
            tiles=(MANZU_4, MANZU_4, MANZU_4),
            from_seat=Seat.SEAT_1,
            called_tile=MANZU_4,
        )
        players = (
            _player(),
            _player(discards=called),
            _player(melds=(pon_4m,)),
            _player(),
        )
        policy_input = _make_input(
            _NINE_MANZU_HAND, players=players, dora_indicators=()
        )

        known = _known_tile_counts(policy_input)

        self.assertEqual(known[TileType(TileCategory.MANZU, 4)], 4)
        decision = _decision(
            _NINE_MANZU_HAND, self.actions, players=players, dora_indicators=()
        )
        self.assertEqual(self.policy.choose_action(decision), self.discard_2m)


class UkeirePolicyDeterminismTest(unittest.TestCase):
    """入力順序を変えても意味的な結果が変わらないことを確認する。"""

    def setUp(self) -> None:
        self.policy = UkeirePolicy()

    def test_choice_is_independent_of_legal_action_order(self) -> None:
        actions = (
            _discard(MANZU_2),
            _discard(MANZU_4),
            _discard(MANZU_6),
            _discard(MANZU_1),
        )

        results = {
            self.policy.choose_action(_decision(_NINE_MANZU_HAND, permutation))
            for permutation in itertools.permutations(actions)
        }

        self.assertEqual(len(results), 1)

    def test_choice_is_independent_of_meld_and_discard_collection_order(self) -> None:
        pon_4m = PublicMeld(
            kind=MeldKind.PON,
            tiles=(MANZU_4, MANZU_4, MANZU_4),
            from_seat=Seat.SEAT_1,
            called_tile=MANZU_4,
        )
        pon_1p = PublicMeld(
            kind=MeldKind.PON,
            tiles=(PINZU_5, PINZU_5, PINZU_5),
            from_seat=Seat.SEAT_1,
            called_tile=PINZU_5,
        )
        history = _discard_history((MANZU_7, EAST, SOUTH))
        actions = (_discard(MANZU_2), _discard(MANZU_4))

        def choose(melds: tuple[PublicMeld, ...], discards: tuple[Discard, ...]):
            players = (
                _player(),
                _player(),
                _player(melds=melds, discards=discards),
                _player(),
            )
            return self.policy.choose_action(
                _decision(
                    _NINE_MANZU_HAND, actions, players=players, dora_indicators=()
                )
            )

        baseline = choose((pon_4m, pon_1p), history)
        reordered = choose((pon_1p, pon_4m), tuple(reversed(history)))

        self.assertEqual(baseline, reordered)

    def test_choice_is_independent_of_concealed_tile_order(self) -> None:
        actions = (_discard(MANZU_2), _discard(MANZU_4))
        shuffled = tuple(reversed(_NINE_MANZU_HAND))

        self.assertEqual(
            self.policy.choose_action(_decision(_NINE_MANZU_HAND, actions)),
            self.policy.choose_action(_decision(shuffled, actions)),
        )


class UkeirePolicyWinningAndNonDiscardTest(unittest.TestCase):
    """#51で確定した和了優先・保守的な非打牌decisionを維持することを確認する。"""

    def setUp(self) -> None:
        self.policy = UkeirePolicy()

    def test_prefers_tsumo_over_discard_and_pass(self) -> None:
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_2)
        decision = _decision(
            _NINE_MANZU_HAND,
            (_discard(MANZU_2), PassAction(actor=Seat.SEAT_0), tsumo),
        )

        self.assertEqual(self.policy.choose_action(decision), tsumo)

    def test_prefers_ron_over_discard_and_pass(self) -> None:
        ron = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_2)
        decision = _decision(
            _NINE_MANZU_HAND,
            (_discard(MANZU_2), PassAction(actor=Seat.SEAT_0), ron),
        )

        self.assertEqual(self.policy.choose_action(decision), ron)

    def test_multiple_winning_actions_are_chosen_order_independently(self) -> None:
        ron = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_2)
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_2)

        results = {
            self.policy.choose_action(_decision(_NINE_MANZU_HAND, permutation))
            for permutation in itertools.permutations((ron, tsumo))
        }

        self.assertEqual(results, {ron})

    def test_chi_and_pass_choose_pass(self) -> None:
        chi = ChiAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_3,
            called_tile=MANZU_5,
            consumed_tiles=(MANZU_4, MANZU_6),
        )
        pass_action = PassAction(actor=Seat.SEAT_0)
        decision = _decision((), (chi, pass_action))

        self.assertEqual(self.policy.choose_action(decision), pass_action)

    def test_pon_and_pass_choose_pass(self) -> None:
        pon = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5, PINZU_5),
        )
        pass_action = PassAction(actor=Seat.SEAT_0)
        decision = _decision((), (pon, pass_action))

        self.assertEqual(self.policy.choose_action(decision), pass_action)

    def test_daiminkan_and_pass_choose_pass(self) -> None:
        daiminkan = DaiminkanAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5, PINZU_5, PINZU_5),
        )
        pass_action = PassAction(actor=Seat.SEAT_0)
        decision = _decision((), (daiminkan, pass_action))

        self.assertEqual(self.policy.choose_action(decision), pass_action)

    def test_discard_with_optional_riichi_ankan_kakan_or_kyuushu_chooses_discard(
        self,
    ) -> None:
        optional_actions: tuple[object, ...] = (
            RiichiAction(actor=Seat.SEAT_0),
            AnkanAction(actor=Seat.SEAT_0, tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5)),
            KakanAction(
                actor=Seat.SEAT_0,
                added_tile=PINZU_5,
                from_seat=Seat.SEAT_1,
                called_tile=PINZU_5,
            ),
            KyuushuKyuuhaiAction(actor=Seat.SEAT_0),
        )
        discard_2m = _discard(MANZU_2)
        discard_4m = _discard(MANZU_4)

        for optional_action in optional_actions:
            with self.subTest(optional_action=type(optional_action).__name__):
                decision = _decision(
                    _NINE_MANZU_HAND, (discard_2m, discard_4m, optional_action)
                )
                self.assertEqual(self.policy.choose_action(decision), discard_4m)

    def test_single_non_discard_candidate_is_returned_as_a_forced_action(self) -> None:
        riichi = RiichiAction(actor=Seat.SEAT_0)
        decision = _decision((), (riichi,))

        self.assertEqual(self.policy.choose_action(decision), riichi)

    def test_ambiguous_non_discard_decision_fails_closed(self) -> None:
        riichi = RiichiAction(actor=Seat.SEAT_0)
        kyuushu = KyuushuKyuuhaiAction(actor=Seat.SEAT_0)
        decision = _decision((), (riichi, kyuushu))

        with self.assertRaises(UkeirePolicyError):
            self.policy.choose_action(decision)


class UkeirePolicySpecialHandTest(unittest.TestCase):
    """七対子・国士無双の局面でも、同じ仕組みで受け入れを判定できることを確認する。"""

    def setUp(self) -> None:
        self.policy = UkeirePolicy()

    def test_seven_pairs_tenpai_is_kept_over_a_worse_shanten_discard(self) -> None:
        """七対子0向聴を保つ打牌を選ぶ。

        1133557799m + 11p + 3p + 5pの14枚で、5p切りは七対子0向聴のまま、
        1m切りは対子を1つ崩して1向聴になる。
        """
        concealed = _hand("1133557799m1135p")
        keep_seven_pairs = _discard(PINZU_5)
        break_pair = _discard(MANZU_1)

        for permutation in itertools.permutations((keep_seven_pairs, break_pair)):
            with self.subTest(order=[action.tile for action in permutation]):
                decision = _decision(concealed, permutation, dora_indicators=())
                self.assertEqual(self.policy.choose_action(decision), keep_seven_pairs)

    def test_thirteen_orphans_tenpai_is_kept_over_a_worse_shanten_discard(self) -> None:
        """国士無双13面待ちを保つ打牌を選ぶ。

        19m19p19s + 字牌7種 + 5mの14枚で、5m切りは国士0向聴、1m切りは
        么九牌を1種失って1向聴になる。受け入れ枚数はどちらも么九牌13種分
        だが、最小向聴の5m切りが選ばれる。
        """
        concealed = _hand("159m19p19s1234567z")
        keep_orphans = _discard(MANZU_5)
        break_orphans = _discard(MANZU_1)

        for permutation in itertools.permutations((keep_orphans, break_orphans)):
            with self.subTest(order=[action.tile for action in permutation]):
                decision = _decision(concealed, permutation, dora_indicators=())
                self.assertEqual(self.policy.choose_action(decision), keep_orphans)


class PolicyGenerationTest(unittest.TestCase):
    """Policy世代が、比較可能な公開Policyとして併存することを確認する。"""

    def test_every_policy_generation_stays_publicly_available(self) -> None:
        import lisjong.policies as policies

        self.assertEqual(
            set(policies.__all__),
            {
                "FiniteHorizonCompletionPolicy",
                "GenbutsuDefenseFiniteHorizonHandValueAwarePolicy",
                "GenbutsuDefenseFiniteHorizonValueAwarePolicy",
                "GenbutsuDefenseTwoStepUkeirePolicy",
                "HandValueAwareTwoStepUkeirePolicy",
                "MinimalPolicy",
                "ShantenPolicy",
                "UkeirePolicy",
                "TwoStepUkeirePolicy",
                "ValueAwareTwoStepUkeirePolicy",
                "YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy",
            },
        )

    def test_ukeire_policy_does_not_reuse_the_shanten_policy_error_type(self) -> None:
        """世代ごとに独立したfail closed例外を持ち、`ShantenPolicy`へ相乗りしない。"""
        from lisjong.policies.shanten import ShantenPolicyError

        self.assertFalse(issubclass(UkeirePolicyError, ShantenPolicyError))
        self.assertFalse(issubclass(ShantenPolicyError, UkeirePolicyError))


class UkeirePolicyExecutionBoundaryTest(unittest.TestCase):
    """共通のPolicy実行境界を通過することを確認する。"""

    def test_selected_discard_passes_the_execution_boundary(self) -> None:
        discard_2m = _discard(MANZU_2)
        discard_4m = _discard(MANZU_4)
        decision = _decision(_NINE_MANZU_HAND, (discard_2m, discard_4m))

        selected = execute_policy(UkeirePolicy(), decision)

        self.assertEqual(selected, discard_4m)


if __name__ == "__main__":
    unittest.main()
