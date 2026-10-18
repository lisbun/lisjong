"""L2aの判断の分類・教師規則・support計数（Issue #315）のtest。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import candidate_fixtures as cf
import learning_fixtures as fixtures

from lisjong.learning.__main__ import main
from lisjong.learning.errors import LearnedPolicyError
from lisjong.learning.l2a_decision import (
    L2A_DECISION_CLASSIFICATION_IDENTITY,
    L2aDecisionKind,
    classify_l2a_decision,
    guard_action,
    is_single_candidate,
    scorer_candidates,
    teacher_rule_exception,
)
from lisjong.learning.l2a_support import (
    L2A_SUPPORT_REPORT_SCHEMA,
    count_source_support,
    is_all_last_equivalent,
)
from lisjong.learning.source_record import open_source_record, read_source_record
from lisjong.policy_contract import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DecisionContext,
    DiscardAction,
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
from lisjong.policy_contract.wind import Wind

_EAST = Tile(TileType(TileCategory.HONOR, 1))
_ONE_MAN = Tile(TileType(TileCategory.MANZU, 1))


def _response(spec, *actions):
    value = cf.policy_input(spec, drawn=False)
    return DecisionContext(
        input=value, legal_actions=(PassAction(actor=value.self_seat), *actions)
    )


def _pon(tile=_EAST):
    return PonAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_3,
        called_tile=tile,
        consumed_tiles=(tile, tile),
    )


def _daiminkan(tile=_EAST):
    return DaiminkanAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_3,
        called_tile=tile,
        consumed_tiles=(tile, tile, tile),
    )


def _chi():
    def man(rank):
        return Tile(TileType(TileCategory.MANZU, rank))

    return ChiAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_3,
        called_tile=man(3),
        consumed_tiles=(man(1), man(2)),
    )


def _single_candidate_discard():
    """リーチ中を想定した、ツモ切り1つと暗槓だけが合法な判断。"""
    value = cf.policy_input("1111m456p789s122z3z")
    return DecisionContext(
        input=value,
        legal_actions=(
            *cf.discard_actions(value, tsumogiri_only=True),
            AnkanAction(actor=value.self_seat, tiles=(_ONE_MAN,) * 4),
        ),
    )


def _south_four(decision):
    """同じ判断を南4局（オーラス相当）へ置き換える。"""
    value = decision.input
    round_state = replace(value.round, round_wind=Wind.SOUTH, hand_number=4)
    return replace(decision, input=replace(value, round=round_state))


class ClassificationTests(unittest.TestCase):
    def test_each_kind(self) -> None:
        cases = {
            L2aDecisionKind.FORCED: cf.discard_decision(
                cf.NEAR_HAND, tsumogiri_only=True
            ),
            L2aDecisionKind.WIN: cf.tsumo_decision(),
            L2aDecisionKind.RIICHI: cf.riichi_decision(),
            L2aDecisionKind.DISCARD: cf.discard_decision(cf.NEAR_HAND),
            L2aDecisionKind.CALL: cf.response_decision(),
            L2aDecisionKind.PASS_RESPONSE: _response("123m456p789s1112z", _daiminkan()),
        }
        for expected, decision in cases.items():
            with self.subTest(kind=expected):
                self.assertIs(classify_l2a_decision(decision), expected)

    def test_ron_response_is_win(self) -> None:
        self.assertIs(
            classify_l2a_decision(cf.ron_response_decision()), L2aDecisionKind.WIN
        )

    def test_forced_uses_all_legal_actions_not_discard_count(self) -> None:
        decision = _single_candidate_discard()

        self.assertIs(classify_l2a_decision(decision), L2aDecisionKind.DISCARD)
        self.assertTrue(is_single_candidate(decision, L2aDecisionKind.DISCARD))
        self.assertIsInstance(
            guard_action(decision, L2aDecisionKind.DISCARD), DiscardAction
        )

    def test_unclassifiable_decision_fails_closed(self) -> None:
        value = cf.policy_input("123m456p789s1112z", drawn=False)
        for legal in ((_pon(), _daiminkan()), (_chi(), _pon())):
            with self.subTest(legal=[type(action).__name__ for action in legal]):
                decision = DecisionContext(input=value, legal_actions=legal)
                with self.assertRaises(LearnedPolicyError):
                    classify_l2a_decision(decision)

    def test_identity(self) -> None:
        self.assertEqual(
            L2A_DECISION_CLASSIFICATION_IDENTITY,
            "lisjong-l2a-decision-classification-v1",
        )


class CandidateAndGuardTests(unittest.TestCase):
    def test_discard_candidates_exclude_ankan(self) -> None:
        decision = cf.ankan_decision()
        candidates = scorer_candidates(decision, L2aDecisionKind.DISCARD)

        self.assertTrue(candidates)
        self.assertTrue(all(isinstance(action, DiscardAction) for action in candidates))
        self.assertFalse(is_single_candidate(decision, L2aDecisionKind.DISCARD))
        self.assertIsNone(guard_action(decision, L2aDecisionKind.DISCARD))

    def test_call_candidates_exclude_daiminkan(self) -> None:
        decision = _response("123m456p789s1112z", _pon(), _daiminkan())
        candidates = scorer_candidates(decision, L2aDecisionKind.CALL)

        self.assertEqual(
            [type(action) for action in candidates], [PassAction, PonAction]
        )
        self.assertIsNone(guard_action(decision, L2aDecisionKind.CALL))

    def test_guard_actions_are_legal_objects(self) -> None:
        cases = (
            (cf.tsumo_decision(), TsumoAction),
            (cf.ron_response_decision(), RonAction),
            (cf.riichi_decision(), RiichiAction),
            (_response("123m456p789s1112z", _daiminkan()), PassAction),
        )
        for decision, expected_type in cases:
            with self.subTest(expected=expected_type.__name__):
                kind = classify_l2a_decision(decision)
                action = guard_action(decision, kind)
                self.assertIsInstance(action, expected_type)
                self.assertTrue(
                    any(action is legal for legal in decision.legal_actions)
                )

    def test_non_scorer_kind_has_no_candidates(self) -> None:
        with self.assertRaises(LearnedPolicyError):
            scorer_candidates(cf.riichi_decision(), L2aDecisionKind.RIICHI)


class TeacherRuleTests(unittest.TestCase):
    def test_teacher_following_rules_has_no_exception(self) -> None:
        cases = (
            (cf.tsumo_decision(), guard_action),
            (cf.riichi_decision(), guard_action),
            (_single_candidate_discard(), guard_action),
        )
        for decision, choose in cases:
            kind = classify_l2a_decision(decision)
            self.assertIsNone(
                teacher_rule_exception(decision, kind, choose(decision, kind))
            )
        decision = cf.response_decision()
        self.assertIsNone(
            teacher_rule_exception(
                decision, L2aDecisionKind.CALL, decision.legal_actions[1]
            )
        )

    def test_exceptions_are_reported(self) -> None:
        riichi = cf.riichi_decision()
        ankan = cf.ankan_decision()
        ankan_action = next(
            a for a in ankan.legal_actions if isinstance(a, AnkanAction)
        )
        single = _single_candidate_discard()
        single_ankan = next(
            a for a in single.legal_actions if isinstance(a, AnkanAction)
        )
        call = _response("123m456p789s1112z", _pon(), _daiminkan())
        cases = (
            (riichi, riichi.legal_actions[0]),
            (ankan, ankan_action),
            (single, single_ankan),
            (call, call.legal_actions[2]),
        )
        for decision, selected in cases:
            with self.subTest(selected=type(selected).__name__):
                kind = classify_l2a_decision(decision)
                self.assertIsNotNone(teacher_rule_exception(decision, kind, selected))


class AllLastTests(unittest.TestCase):
    def test_all_last_equivalent(self) -> None:
        cases = (
            (Wind.EAST, 4, False),
            (Wind.SOUTH, 3, False),
            (Wind.SOUTH, 4, True),
            (Wind.WEST, 1, True),
        )
        for wind, hand_number, expected in cases:
            with self.subTest(wind=wind, hand_number=hand_number):
                value = fixtures.policy_input(round_wind=wind, hand_number=hand_number)
                self.assertIs(is_all_last_equivalent(value), expected)


def _teacher(decision):
    """規則どおりの教師。DISCARDは最後の合法打牌、CALLはPonを選ぶ。"""
    kind = classify_l2a_decision(decision)
    action = guard_action(decision, kind)
    if action is not None:
        return action
    candidates = scorer_candidates(decision, kind)
    if kind is L2aDecisionKind.CALL:
        return next(a for a in candidates if isinstance(a, PonAction))
    return candidates[-1]


def _first_discard(decision):
    """C0固有層の判定に使う基準。最初の合法打牌を選ぶ。"""
    return next(a for a in decision.legal_actions if isinstance(a, DiscardAction))


class SupportCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def record(self, *, teacher=_teacher, games=None) -> Path:
        if games is None:
            game0 = (
                cf.discard_decision(cf.NEAR_HAND),
                cf.discard_decision(cf.NEAR_HAND, tsumogiri_only=True),
                cf.riichi_decision(),
                cf.response_decision(),
                _single_candidate_discard(),
            )
            game1 = (cf.discard_decision(cf.SECOND_NEAR_HAND), cf.tsumo_decision())
            games = (("TRAIN", game0), ("SELECT", game1))
        written = []
        for ordinal, (split, decisions) in enumerate(games):
            seed = 100 * (ordinal + 1)
            rows = cf.source_rows(
                decisions, game_ordinal=ordinal, seed=seed, split=split, teacher=teacher
            )
            written.append((split, seed, rows))
        return fixtures.write_policy_source_record(self.root / "source-record", written)

    def test_counts_rows_and_hanchan_support(self) -> None:
        report = count_source_support(
            read_source_record(self.record()), reference_policy=_first_discard
        )

        self.assertEqual(report["schema"], L2A_SUPPORT_REPORT_SCHEMA)
        self.assertEqual(
            report["classification_identity"], L2A_DECISION_CLASSIFICATION_IDENTITY
        )
        self.assertEqual(report["source"]["purpose"], "DEVELOPMENT")
        total = report["total"]
        self.assertEqual(total["hanchan"], 2)
        self.assertEqual(total["decisions"], 7)
        self.assertEqual(total["kinds"]["discard"], {"rows": 3, "hanchan": 2})
        self.assertEqual(total["kinds"]["forced"], {"rows": 1, "hanchan": 1})
        self.assertEqual(total["kinds"]["call"], {"rows": 1, "hanchan": 1})
        self.assertEqual(total["kinds"]["win"], {"rows": 1, "hanchan": 1})
        self.assertEqual(total["single_candidate"], {"rows": 1, "hanchan": 1})
        self.assertEqual(total["discard"]["scored"], {"rows": 2, "hanchan": 2})
        self.assertEqual(total["discard"]["c0_specific"], {"rows": 2, "hanchan": 2})
        self.assertEqual(total["call"]["pon_legal"], {"rows": 1, "hanchan": 1})
        self.assertEqual(total["call"]["pon_selected"], {"rows": 1, "hanchan": 1})
        self.assertEqual(total["call"]["chi_legal"], {"rows": 0, "hanchan": 0})
        self.assertEqual(total["selected_actions"]["pon"], {"rows": 1, "hanchan": 1})
        self.assertEqual(total["selected_actions"]["ankan"], {"rows": 0, "hanchan": 0})
        self.assertEqual(total["teacher_rule_exceptions"], 0)
        self.assertEqual(report["teacher_rule_exception_examples"], [])
        self.assertEqual(report["splits"]["SELECT"]["kinds"]["discard"]["rows"], 1)
        self.assertEqual(
            total["all_last_equivalent"]["discard"], {"rows": 0, "hanchan": 0}
        )
        self.assertEqual(total["discard"]["all_last_scored"], {"rows": 0, "hanchan": 0})
        self.assertNotIn("agreement", json.dumps(report))

    def test_all_last_scored_discard_excludes_single_candidate_rows(self) -> None:
        # 南4局で単一候補行しかない半荘と、評価対象の打牌が1つある半荘。
        only_single = (_south_four(_single_candidate_discard()),)
        with_scored = (
            _south_four(_single_candidate_discard()),
            _south_four(cf.discard_decision(cf.NEAR_HAND)),
        )
        report = count_source_support(
            read_source_record(
                self.record(games=(("TRAIN", only_single), ("SELECT", with_scored)))
            ),
            reference_policy=_first_discard,
        )

        only_single_split = report["splits"]["TRAIN"]
        self.assertEqual(
            only_single_split["all_last_equivalent"]["discard"],
            {"rows": 1, "hanchan": 1},
        )
        self.assertEqual(
            only_single_split["discard"]["all_last_scored"], {"rows": 0, "hanchan": 0}
        )
        total = report["total"]
        self.assertEqual(
            total["all_last_equivalent"]["discard"], {"rows": 3, "hanchan": 2}
        )
        self.assertEqual(total["discard"]["all_last_scored"], {"rows": 1, "hanchan": 1})

    def test_streaming_record_gives_same_report(self) -> None:
        path = self.record()
        self.assertEqual(
            count_source_support(
                read_source_record(path), reference_policy=_first_discard
            ),
            count_source_support(
                open_source_record(path), reference_policy=_first_discard
            ),
        )

    def test_one_teacher_rule_exception_is_reported(self) -> None:
        def teacher(decision):
            if classify_l2a_decision(decision) is L2aDecisionKind.RIICHI:
                return next(
                    a for a in decision.legal_actions if isinstance(a, DiscardAction)
                )
            return _teacher(decision)

        report = count_source_support(
            read_source_record(self.record(teacher=teacher)),
            reference_policy=_first_discard,
        )

        self.assertEqual(report["total"]["teacher_rule_exceptions"], 1)
        (example,) = report["teacher_rule_exception_examples"]
        self.assertEqual(example["split"], "TRAIN")
        self.assertEqual(example["decision_ordinal"], 2)
        self.assertIn("RIICHI", example["reason"])

    def test_cli_emits_report(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["l2a-source-support", "--source-record", str(self.record())])

        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["total"]["decisions"], 7)


if __name__ == "__main__":
    unittest.main()
