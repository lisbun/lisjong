import unittest

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
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.riichilab_adapter.errors import PossibleActionsValidationError
from lisjong.riichilab_adapter.possible_action_validation import (
    validate_against_possible_actions,
)

MANZU_2 = Tile(TileType(TileCategory.MANZU, 2))
MANZU_3 = Tile(TileType(TileCategory.MANZU, 3))
MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))
MANZU_5_RED = Tile(TileType(TileCategory.MANZU, 5), is_red=True)
PINZU_2 = Tile(TileType(TileCategory.PINZU, 2))


class ValidateAgainstPossibleActionsTest(unittest.TestCase):
    # --- 公式candidate schemaに基づく正常系(回帰防止) -------------------
    #
    # RiichiLab公式`possible_actions` candidateは、Bot-to-Server response
    # よりも小さい最小表現である(Issue #38 review、
    # comment-5298618558)。以下のtestは、candidate schemaとBot response
    # schemaを再び混同しないための回帰防止を目的とする。

    def test_accepts_the_official_minimal_dahai_candidate_shape(self) -> None:
        """公式形`{"type": "dahai", "pai": "1m"}`のように、actorもtsumogiriも
        持たないcandidateを合法として受理できること。"""
        selected = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_3, tsumogiri=False)
        candidates = [{"type": "dahai", "pai": "3m"}]

        validate_against_possible_actions(selected, candidates)

    def test_tsumogiri_selection_still_matches_the_minimal_dahai_candidate(
        self,
    ) -> None:
        """selected側が`tsumogiri=True`でも、candidate側にtsumogiriが
        存在しない公式形candidateへ一致できること。tsumogiriはcandidate
        identityではなくBot response生成側の情報である。"""
        selected = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=True)
        candidates = [{"type": "dahai", "pai": "4m"}]

        validate_against_possible_actions(selected, candidates)

    def test_candidate_without_actor_field_matches_normally(self) -> None:
        """candidateへ`actor`が無くても、公式candidate schemaとして正常に
        照合できること(candidateへ一律actorを要求しない)。"""
        selected = RiichiAction(actor=Seat.SEAT_2)
        candidates = [{"type": "reach"}]

        validate_against_possible_actions(selected, candidates)

    def test_call_candidate_without_bot_response_target_field_matches(self) -> None:
        """chi/pon/daiminkanのcandidateへ、Bot response専用の`target`が
        無くても、公式candidateの`pai` + `consumed`だけで正常に照合
        できること。"""
        selected = ChiAction(
            actor=Seat.SEAT_1,
            target=Seat.SEAT_0,
            called_tile=MANZU_3,
            consumed_tiles=(MANZU_2, MANZU_4),
        )
        candidates = [{"type": "chi", "pai": "3m", "consumed": ["2m", "4m"]}]

        validate_against_possible_actions(selected, candidates)

    def test_hora_candidate_without_actor_or_target_matches(self) -> None:
        """horaのcandidateへ`actor`/`target`が無くても、公式candidateの
        `pai`(和了牌)だけで正常に照合できること。"""
        selected = RonAction(
            actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_5
        )
        candidates = [{"type": "hora", "pai": "5m"}]

        validate_against_possible_actions(selected, candidates)

    # --- 従来からのsemantic matching / fail closed契約 -------------------

    def test_accepts_a_single_exact_dahai_match(self) -> None:
        selected = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_3, tsumogiri=False)
        candidates = [
            {"type": "dahai", "pai": "3m"},
            {"type": "dahai", "pai": "4m"},
        ]

        validate_against_possible_actions(selected, candidates)

    def test_is_independent_of_candidate_order(self) -> None:
        selected = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=True)
        candidates = [
            {"type": "dahai", "pai": "3m"},
            {"type": "reach"},
            {"type": "dahai", "pai": "4m"},
        ]

        validate_against_possible_actions(selected, candidates)

    def test_ignores_semantically_irrelevant_extra_fields_on_candidates(self) -> None:
        selected = RiichiAction(actor=Seat.SEAT_2)
        candidates = [
            {
                "type": "reach",
                "actor": 2,
                # 意味を持たない付加field。誤って拒否理由にしない。
                "display_name": "declare riichi",
                "score_delta": -1000,
            }
        ]

        validate_against_possible_actions(selected, candidates)

    def test_distinguishes_red_five_from_normal_five(self) -> None:
        selected = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5_RED, tsumogiri=False)
        candidates = [
            {"type": "dahai", "pai": "5m"},
        ]

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, candidates)

    def test_chi_requires_matching_consumed_composition(self) -> None:
        selected = ChiAction(
            actor=Seat.SEAT_1,
            target=Seat.SEAT_0,
            called_tile=MANZU_3,
            consumed_tiles=(MANZU_2, MANZU_4),
        )
        wrong_composition = {
            "type": "chi",
            "pai": "3m",
            "consumed": ["4m", "5m"],
        }
        matching = {
            "type": "chi",
            "pai": "3m",
            "consumed": ["4m", "2m"],  # 順序が違っても一致すること
        }

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, [wrong_composition])

        validate_against_possible_actions(selected, [wrong_composition, matching])

    def test_pon_candidate_without_target_still_matches_correctly(self) -> None:
        """公式candidateに`target`が無いため、candidate側は`pai` +
        `consumed`だけで識別する。`target`はBot response側でのみ使う情報
        であり、candidate validationへ混入させない。"""
        selected = PonAction(
            actor=Seat.SEAT_2,
            target=Seat.SEAT_1,
            called_tile=PINZU_2,
            consumed_tiles=(PINZU_2, PINZU_2),
        )
        candidate = {"type": "pon", "pai": "2p", "consumed": ["2p", "2p"]}

        validate_against_possible_actions(selected, [candidate])

    def test_daiminkan_matches_on_full_semantic_key(self) -> None:
        selected = DaiminkanAction(
            actor=Seat.SEAT_3,
            target=Seat.SEAT_2,
            called_tile=PINZU_2,
            consumed_tiles=(PINZU_2, PINZU_2, PINZU_2),
        )
        candidate = {
            "type": "daiminkan",
            "pai": "2p",
            "consumed": ["2p", "2p", "2p"],
        }

        validate_against_possible_actions(selected, [candidate])

    def test_ankan_matches_on_tile_multiset(self) -> None:
        selected = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(PINZU_2, PINZU_2, PINZU_2, PINZU_2),
        )
        candidate = {
            "type": "ankan",
            "consumed": ["2p", "2p", "2p", "2p"],
        }

        validate_against_possible_actions(selected, [candidate])

    def test_kakan_matches_on_added_tile(self) -> None:
        selected = KakanAction(
            actor=Seat.SEAT_1,
            added_tile=PINZU_2,
            from_seat=Seat.SEAT_0,
            called_tile=PINZU_2,
        )
        candidate = {"type": "kakan", "pai": "2p"}

        validate_against_possible_actions(selected, [candidate])

    def test_ron_matches_on_winning_tile_regardless_of_target_field(self) -> None:
        selected = RonAction(
            actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_5
        )
        wrong_pai = {"type": "hora", "pai": "4m"}
        matching = {"type": "hora", "pai": "5m"}

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, [wrong_pai])

        validate_against_possible_actions(selected, [wrong_pai, matching])

    def test_tsumo_matches_the_same_hora_candidate_shape_as_ron(self) -> None:
        selected = TsumoAction(actor=Seat.SEAT_2, winning_tile=MANZU_5)
        candidate = {"type": "hora", "pai": "5m"}

        validate_against_possible_actions(selected, [candidate])

    def test_pass_matches_none_type(self) -> None:
        selected = PassAction(actor=Seat.SEAT_1)
        candidate = {"type": "none"}

        validate_against_possible_actions(selected, [candidate])

    def test_kyuushu_kyuuhai_matches_ryukyoku_type(self) -> None:
        selected = KyuushuKyuuhaiAction(actor=Seat.SEAT_3)
        candidate = {"type": "ryukyoku"}

        validate_against_possible_actions(selected, [candidate])

    def test_rejects_zero_matches(self) -> None:
        selected = PassAction(actor=Seat.SEAT_0)

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, [])

    def test_rejects_ambiguous_multiple_matches(self) -> None:
        # 重複candidateの扱いはIssue #38の判断どおり変更しない: 同一
        # semantic Actionへ複数candidateが一致する場合は安全側でfail
        # closedする(#39のlive接続で実データの重複有無を確認する)。
        selected = PassAction(actor=Seat.SEAT_0)
        candidates = [
            {"type": "none"},
            {"type": "none"},
        ]

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, candidates)

    def test_does_not_fall_back_to_first_or_last_or_arbitrary_candidate(self) -> None:
        # 送信予定Actionにまったく対応しない候補群であっても、既知typeが
        # 存在するというだけで代替受理しない。
        selected = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_3, tsumogiri=False)
        candidates = [
            {"type": "dahai", "pai": "4m"},
            {"type": "reach"},
        ]

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, candidates)

    def test_malformed_candidate_is_skipped_not_fatal(self) -> None:
        selected = PassAction(actor=Seat.SEAT_1)
        candidates = [
            "not-a-mapping",
            {"type": "dahai"},  # pai欠落でmalformed
            {"type": "none"},
        ]

        validate_against_possible_actions(selected, candidates)

    def test_unknown_action_type_candidate_is_ignored(self) -> None:
        selected = PassAction(actor=Seat.SEAT_0)
        candidates = [
            {"type": "future_action_type", "extra": True},
            {"type": "none"},
        ]

        validate_against_possible_actions(selected, candidates)

    def test_dahai_without_pai_field_is_treated_as_malformed(self) -> None:
        selected = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_3, tsumogiri=False)
        candidate = {"type": "dahai"}

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, [candidate])


if __name__ == "__main__":
    unittest.main()
