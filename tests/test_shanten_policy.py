"""Issue #51 `ShantenPolicy`のunit test。

`calculate_shanten()`が返す既知の値（`tests/test_shanten.py`と同じ牌姿の
性質）を根拠として、discard候補ごとの打牌後向聴数を人が読める形で説明する。
Policyの選択ロジック自体をtest側へ複製した自己検証は避ける。
"""

import itertools
import unittest

from lisjong.policies import ShantenPolicy
from lisjong.policies.shanten import ShantenPolicyError, _remove_one_matching_tile
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
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_execution import execute_policy
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

MANZU_1 = Tile(TileType(TileCategory.MANZU, 1))
MANZU_2 = Tile(TileType(TileCategory.MANZU, 2))
MANZU_3 = Tile(TileType(TileCategory.MANZU, 3))
MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))
MANZU_5_RED = Tile(TileType(TileCategory.MANZU, 5), is_red=True)
MANZU_6 = Tile(TileType(TileCategory.MANZU, 6))
PINZU_5 = Tile(TileType(TileCategory.PINZU, 5))
EAST = Tile(TileType(TileCategory.HONOR, 1))
SOUTH = Tile(TileType(TileCategory.HONOR, 2))


def _make_player(score: int = 25000) -> PlayerPublicState:
    return PlayerPublicState(
        score=score, discards=(), melds=(), riichi=RiichiState.NONE
    )


def _make_input(
    concealed_tiles: tuple[Tile, ...],
    drawn_tile: Tile | None,
    self_seat: Seat = Seat.SEAT_0,
) -> PolicyInput:
    return PolicyInput(
        self_seat=self_seat,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(MANZU_3,),
            live_wall_tiles_remaining=70,
        ),
        players=(_make_player(), _make_player(), _make_player(), _make_player()),
        own_hand=OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=drawn_tile),
    )


def _decision(
    concealed_tiles: tuple[Tile, ...],
    actions: tuple[object, ...],
    drawn_tile: Tile | None = None,
    self_seat: Seat = Seat.SEAT_0,
) -> DecisionContext:
    return DecisionContext(
        input=_make_input(concealed_tiles, drawn_tile, self_seat),
        legal_actions=actions,
    )


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

        with self.assertRaises(ShantenPolicyError):
            _remove_one_matching_tile(concealed, MANZU_6)


class ShantenPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ShantenPolicy()

    # -- 打牌後向聴数の比較 -------------------------------------------------

    def test_chooses_the_discard_with_the_smallest_post_discard_shanten(self) -> None:
        # 手牌 1112m (四人打ちでは通常起こり得ないが、Policy契約上の
        # OwnHandStateは固定枚数を要求しない)。1mを切れば111m+2mで
        # タンキ聴牌(0向聴、tests/test_shanten.pyの"111m2m"と同じ形)。
        # 2mを切れば1111mだけが残り、5枚目を引けない4枚使いなので1向聴に
        # 悪化する(同じく"1111m"のcaseと同じ形)。
        concealed = (MANZU_1, MANZU_1, MANZU_1, MANZU_1, MANZU_2)
        keep_shape = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_1, tsumogiri=False)
        worsen_shape = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_2, tsumogiri=True)

        for permutation in itertools.permutations((keep_shape, worsen_shape)):
            with self.subTest(order=[type(a).__name__ for a in permutation]):
                decision = _decision(concealed, permutation, drawn_tile=MANZU_2)
                chosen = self.policy.choose_action(decision)
                self.assertEqual(chosen, keep_shape)

    def test_only_one_copy_is_removed_when_simulating_the_chosen_discard(
        self,
    ) -> None:
        """最小向聴の選択が、同一牌の1枚だけ除くsimulationに基づくことを確認する。

        `worsen_shape`が選ばれてしまうのは、除去simulationが誤って複数枚
        除いてしまう(または0枚しか残さない)実装上のbugがある場合である。
        """
        concealed = (MANZU_1, MANZU_1, MANZU_1, MANZU_1, MANZU_2)
        keep_shape = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_1, tsumogiri=False)
        worsen_shape = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_2, tsumogiri=True)
        decision = _decision(concealed, (worsen_shape, keep_shape), drawn_tile=MANZU_2)

        chosen = self.policy.choose_action(decision)

        self.assertEqual(chosen, keep_shape)

    # -- tie-break -----------------------------------------------------------

    def test_tie_break_prefers_the_smaller_tile_sort_key_and_is_order_independent(
        self,
    ) -> None:
        # 111m + EAST + SOUTH。EAST・SOUTHはどちらも他牌と無関係な浮き牌
        # なので、どちらを切っても残り4枚は「111m + 浮き牌1枚」というタンキ
        # 聴牌の形になり、向聴数は同じ0になる(tie)。tile_sort_keyでは
        # EAST(rank1) < SOUTH(rank2)なので、EASTを切る側が選ばれる。
        concealed = (MANZU_1, MANZU_1, MANZU_1, EAST, SOUTH)
        discard_east = DiscardAction(actor=Seat.SEAT_0, tile=EAST, tsumogiri=False)
        discard_south = DiscardAction(actor=Seat.SEAT_0, tile=SOUTH, tsumogiri=False)

        for permutation in itertools.permutations((discard_east, discard_south)):
            with self.subTest(order=[a.tile for a in permutation]):
                decision = _decision(concealed, permutation)
                chosen = self.policy.choose_action(decision)
                self.assertEqual(chosen, discard_east)

    def test_tie_break_prefers_tedashi_over_tsumogiri_for_the_same_tile(self) -> None:
        # 同じ牌(2m)を切る2候補がtsumogiriだけで区別される場合、
        # 打牌後手牌はどちらも同一(1111m、1向聴)でtieになる。
        concealed = (MANZU_1, MANZU_1, MANZU_1, MANZU_1, MANZU_2)
        tedashi = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_2, tsumogiri=False)
        tsumogiri = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_2, tsumogiri=True)

        for permutation in itertools.permutations((tedashi, tsumogiri)):
            decision = _decision(concealed, permutation, drawn_tile=MANZU_2)
            chosen = self.policy.choose_action(decision)
            self.assertEqual(chosen, tedashi)

    def test_tie_break_prefers_normal_five_over_red_five(self) -> None:
        # 555m(通常3枚)+赤5m+2mで、通常5切りと赤5切りはどちらも「5m系3枚+
        # 2m」という同じ形が残るのでtieになる。tile_sort_keyはis_redを
        # 最後の比較要素にしており、Falseが先になるため通常5切りが選ばれる。
        concealed = (MANZU_5, MANZU_5, MANZU_5, MANZU_5_RED, MANZU_2)
        discard_normal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5, tsumogiri=False)
        discard_red = DiscardAction(
            actor=Seat.SEAT_0, tile=MANZU_5_RED, tsumogiri=False
        )

        for permutation in itertools.permutations((discard_normal, discard_red)):
            decision = _decision(concealed, permutation, drawn_tile=MANZU_2)
            chosen = self.policy.choose_action(decision)
            self.assertEqual(chosen, discard_normal)

    # -- 和了の優先 -----------------------------------------------------------

    def test_prefers_tsumo_over_discard_and_pass(self) -> None:
        concealed = (MANZU_1, MANZU_1, MANZU_1, MANZU_1, MANZU_2)
        discard = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_1, tsumogiri=False)
        pass_action = PassAction(actor=Seat.SEAT_0)
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_2)
        decision = _decision(
            concealed, (discard, pass_action, tsumo), drawn_tile=MANZU_2
        )

        chosen = self.policy.choose_action(decision)

        self.assertEqual(chosen, tsumo)

    def test_prefers_ron_over_discard_and_pass(self) -> None:
        concealed = (MANZU_1, MANZU_1, MANZU_1, MANZU_1, MANZU_2)
        discard = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_1, tsumogiri=False)
        pass_action = PassAction(actor=Seat.SEAT_0)
        ron = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_2)
        decision = _decision(concealed, (discard, pass_action, ron))

        chosen = self.policy.choose_action(decision)

        self.assertEqual(chosen, ron)

    def test_multiple_winning_actions_are_chosen_deterministically_and_order_independently(
        self,
    ) -> None:
        ron = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_2)
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_2)
        concealed = (MANZU_1, MANZU_1, MANZU_1, MANZU_1, MANZU_2)

        results = {
            self.policy.choose_action(_decision(concealed, permutation))
            for permutation in itertools.permutations((ron, tsumo))
        }

        self.assertEqual(results, {ron})

    # -- 非打牌decision --------------------------------------------------------

    def test_chi_and_pass_choose_pass(self) -> None:
        chi = ChiAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_3,
            called_tile=MANZU_5,
            consumed_tiles=(MANZU_4, MANZU_6),
        )
        pass_action = PassAction(actor=Seat.SEAT_0)
        decision = _decision((), (chi, pass_action))

        chosen = self.policy.choose_action(decision)

        self.assertEqual(chosen, pass_action)

    def test_pon_and_pass_choose_pass(self) -> None:
        pon = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5, PINZU_5),
        )
        pass_action = PassAction(actor=Seat.SEAT_0)
        decision = _decision((), (pon, pass_action))

        chosen = self.policy.choose_action(decision)

        self.assertEqual(chosen, pass_action)

    def test_daiminkan_and_pass_choose_pass(self) -> None:
        daiminkan = DaiminkanAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5, PINZU_5, PINZU_5),
        )
        pass_action = PassAction(actor=Seat.SEAT_0)
        decision = _decision((), (daiminkan, pass_action))

        chosen = self.policy.choose_action(decision)

        self.assertEqual(chosen, pass_action)

    def test_discard_with_optional_riichi_ankan_kakan_or_kyuushu_chooses_discard(
        self,
    ) -> None:
        concealed = (MANZU_1, MANZU_1, MANZU_1, MANZU_1, MANZU_2)
        keep_shape = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_1, tsumogiri=False)
        worsen_shape = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_2, tsumogiri=True)

        optional_actions: tuple[object, ...] = (
            RiichiAction(actor=Seat.SEAT_0),
            AnkanAction(
                actor=Seat.SEAT_0,
                tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5),
            ),
            KakanAction(
                actor=Seat.SEAT_0,
                added_tile=PINZU_5,
                from_seat=Seat.SEAT_1,
                called_tile=PINZU_5,
            ),
            KyuushuKyuuhaiAction(actor=Seat.SEAT_0),
        )

        for optional_action in optional_actions:
            with self.subTest(optional_action=type(optional_action).__name__):
                decision = _decision(
                    concealed,
                    (keep_shape, worsen_shape, optional_action),
                    drawn_tile=MANZU_2,
                )
                chosen = self.policy.choose_action(decision)
                self.assertEqual(chosen, keep_shape)

    def test_single_non_discard_candidate_is_returned_as_a_forced_action(
        self,
    ) -> None:
        riichi = RiichiAction(actor=Seat.SEAT_0)
        decision = _decision((), (riichi,))

        chosen = self.policy.choose_action(decision)

        self.assertEqual(chosen, riichi)

    def test_ambiguous_non_discard_decision_fails_closed(self) -> None:
        riichi = RiichiAction(actor=Seat.SEAT_0)
        kyuushu = KyuushuKyuuhaiAction(actor=Seat.SEAT_0)
        decision = _decision((), (riichi, kyuushu))

        with self.assertRaises(ShantenPolicyError):
            self.policy.choose_action(decision)

    # -- 不整合な入力 -----------------------------------------------------------

    def test_discard_tile_missing_from_concealed_tiles_fails_closed(self) -> None:
        concealed = (MANZU_1, MANZU_2)
        inconsistent = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_6, tsumogiri=False)
        decision = _decision(concealed, (inconsistent,))

        with self.assertRaises(ShantenPolicyError):
            self.policy.choose_action(decision)

    # -- Policy実行境界との統合 ---------------------------------------------

    def test_selected_discard_passes_the_execution_boundary(self) -> None:
        concealed = (MANZU_1, MANZU_1, MANZU_1, MANZU_1, MANZU_2)
        keep_shape = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_1, tsumogiri=False)
        worsen_shape = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_2, tsumogiri=True)
        decision = _decision(concealed, (keep_shape, worsen_shape), drawn_tile=MANZU_2)

        selected = execute_policy(self.policy, decision)

        self.assertEqual(selected, keep_shape)


if __name__ == "__main__":
    unittest.main()
