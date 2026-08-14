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
    def test_accepts_a_single_exact_dahai_match(self) -> None:
        selected = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_3, tsumogiri=False)
        candidates = [
            {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
            {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False},
        ]

        validate_against_possible_actions(selected, candidates)

    def test_is_independent_of_candidate_order(self) -> None:
        selected = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=True)
        candidates = [
            {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": True},
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
            {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
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
            "actor": 1,
            "target": 0,
            "pai": "3m",
            "consumed": ["4m", "5m"],
        }
        matching = {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "3m",
            "consumed": ["4m", "2m"],  # 順序が違っても一致すること
        }

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, [wrong_composition])

        validate_against_possible_actions(selected, [wrong_composition, matching])

    def test_pon_requires_matching_target(self) -> None:
        selected = PonAction(
            actor=Seat.SEAT_2,
            target=Seat.SEAT_1,
            called_tile=PINZU_2,
            consumed_tiles=(PINZU_2, PINZU_2),
        )
        wrong_target = {
            "type": "pon",
            "actor": 2,
            "target": 3,
            "pai": "2p",
            "consumed": ["2p", "2p"],
        }

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, [wrong_target])

    def test_daiminkan_matches_on_full_semantic_key(self) -> None:
        selected = DaiminkanAction(
            actor=Seat.SEAT_3,
            target=Seat.SEAT_2,
            called_tile=PINZU_2,
            consumed_tiles=(PINZU_2, PINZU_2, PINZU_2),
        )
        candidate = {
            "type": "daiminkan",
            "actor": 3,
            "target": 2,
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
            "actor": 0,
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
        candidate = {"type": "kakan", "actor": 1, "pai": "2p"}

        validate_against_possible_actions(selected, [candidate])

    def test_ron_requires_matching_target_and_winning_tile(self) -> None:
        selected = RonAction(
            actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_5
        )
        wrong_target = {"type": "hora", "actor": 0, "target": 2, "pai": "5m"}
        matching = {"type": "hora", "actor": 0, "target": 1, "pai": "5m"}

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, [wrong_target])

        validate_against_possible_actions(selected, [wrong_target, matching])

    def test_tsumo_target_is_the_actor_itself(self) -> None:
        selected = TsumoAction(actor=Seat.SEAT_2, winning_tile=MANZU_5)
        candidate = {"type": "hora", "actor": 2, "target": 2, "pai": "5m"}

        validate_against_possible_actions(selected, [candidate])

    def test_pass_matches_none_type(self) -> None:
        selected = PassAction(actor=Seat.SEAT_1)
        candidate = {"type": "none", "actor": 1}

        validate_against_possible_actions(selected, [candidate])

    def test_kyuushu_kyuuhai_matches_ryukyoku_type(self) -> None:
        selected = KyuushuKyuuhaiAction(actor=Seat.SEAT_3)
        candidate = {"type": "ryukyoku", "actor": 3}

        validate_against_possible_actions(selected, [candidate])

    def test_rejects_zero_matches(self) -> None:
        selected = PassAction(actor=Seat.SEAT_0)

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, [])

    def test_rejects_ambiguous_multiple_matches(self) -> None:
        selected = PassAction(actor=Seat.SEAT_0)
        candidates = [
            {"type": "none", "actor": 0},
            {"type": "none", "actor": 0},
        ]

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, candidates)

    def test_does_not_fall_back_to_first_or_last_or_arbitrary_candidate(self) -> None:
        # 送信予定Actionにまったく対応しない候補群であっても、既知typeが
        # 存在するというだけで代替受理しない。
        selected = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_3, tsumogiri=False)
        candidates = [
            {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False},
            {"type": "reach", "actor": 0},
        ]

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, candidates)

    def test_malformed_candidate_is_skipped_not_fatal(self) -> None:
        selected = PassAction(actor=Seat.SEAT_1)
        candidates = [
            "not-a-mapping",
            {"type": "dahai", "actor": 1},  # pai/tsumogiri欠落でmalformed
            {"type": "none", "actor": 1},
        ]

        validate_against_possible_actions(selected, candidates)

    def test_unknown_action_type_candidate_is_ignored(self) -> None:
        selected = PassAction(actor=Seat.SEAT_0)
        candidates = [
            {"type": "future_action_type", "actor": 0, "extra": True},
            {"type": "none", "actor": 0},
        ]

        validate_against_possible_actions(selected, candidates)

    def test_dahai_without_tsumogiri_field_is_treated_as_malformed(self) -> None:
        selected = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_3, tsumogiri=False)
        candidate = {"type": "dahai", "actor": 0, "pai": "3m"}

        with self.assertRaises(PossibleActionsValidationError):
            validate_against_possible_actions(selected, [candidate])


if __name__ == "__main__":
    unittest.main()
