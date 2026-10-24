"""lisbun/lisjong#343 `SingleOpenGenbutsuPlacementAwareSpeedCallPolicy`のfocused test。

gateの内外ではChampion `PlacementAwareSpeedCallPolicy`と同じactionになること、発動した
判断ではChampionの打牌（C0）と比べて向聴数が同じで、現在受入・retained real value・
yaku route valueが下がらない単独副露者の河の現物だけを選ぶことを固定する。
局面ごとの値は実際の計算結果で、改善の証拠ではない。
"""

import ast
import dataclasses
import inspect
import pickle
import unittest
from unittest import mock

from test_placement_aware_speed_call_policy import (
    ALL_LAST_TOP,
    PASS,
    _chi_meld,
    _discards,
    _hand,
    _input,
    _player,
    _pon_meld,
    _tile,
)

import lisjong.policies.single_open_genbutsu_placement_aware_speed_call as module
from lisjong.candidate_boundary import check_candidate_imports, check_candidate_purity
from lisjong.learning.open_wait_estimator import is_open_opponent
from lisjong.policies import (
    PlacementAwareSpeedCallPolicy,
    SingleOpenGenbutsuPlacementAwareSpeedCallPolicy,
)
from lisjong.policies.single_open_genbutsu_placement_aware_speed_call import (
    GenbutsuCandidateEvaluation,
    SingleOpenGenbutsuAnalysis,
    SingleOpenGenbutsuStage,
    open_opponent_seats,
    river_genbutsu_tile_types,
)
from lisjong.policy_contract.action import (
    DiscardAction,
    PonAction,
    RiichiAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.decision_trace import DecisionTraceRecorder
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.policy_execution import (
    execute_policy,
    execute_policy_with_trace,
)
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.wind import Wind

_MODULE = "lisjong.policies.single_open_genbutsu_placement_aware_speed_call"
_FACTORY = f"{_MODULE}:SingleOpenGenbutsuPlacementAwareSpeedCallPolicy"
_STAGE = SingleOpenGenbutsuStage

# 23m 456p 78s 11p 99p + 1z 2z 3z（3z自摸）: 2向聴。Championは1zを切る。
# 字牌3枚はどれを切っても向聴2・受入20・価値0・route 0。
BASE = "23m456p78s1199p1z2z3z"
# 1479m 258p 369s 1234z（4z自摸）: 6向聴。Championは4m（受入59）。9mは受入39。
UKEIRE_HAND = "1479m258p369s1234z"
# 234m 456p 67s 22p 88p 5m 1z（1z自摸）: Championは1z（打牌後は断么compatibleでroute 1）。
# 5mも向聴1・受入12だが、1zが残るのでroute 0。
ROUTE_HAND = "234m456p67s2288p5m1z"
# 23m 456p 78s 11p 99p 1z 2z 5s（2z自摸）: Championは5sを切る。
SOUZU_HAND = "23m456p78s1199p1z2z5s"
# 上の5sを赤5sにした手。赤5sを切るとretained real valueが1下がる。
RED_HAND = "23m456p78s1199p1z2z0s"
# 123m 456p 789s 11z 22z + 5z自摸: 5zを切れば聴牌。
TENPAI_HAND = "123m456p789s1122z5z"

CHAMPION = PlacementAwareSpeedCallPolicy()
CANDIDATE = SingleOpenGenbutsuPlacementAwareSpeedCallPolicy()


def _action(spec, *, tsumogiri=False):
    return DiscardAction(actor=Seat.SEAT_0, tile=_tile(spec), tsumogiri=tsumogiri)


def _ankan(spec):
    return PublicMeld(
        kind=MeldKind.ANKAN, tiles=_hand(spec), from_seat=None, called_tile=None
    )


def _with_player(policy_input, index, player):
    players = list(policy_input.players)
    players[index] = player
    return dataclasses.replace(policy_input, players=tuple(players))


def _situation(hand=BASE, river="2z", *, drawn="3z", open_seat=2, **keywords):
    """`open_seat`が7zをポンした副露者で、その河が`river`の局面。"""
    policy_input = _input(hand, drawn=drawn, **keywords)
    if open_seat is None:
        return policy_input
    return _with_player(
        policy_input, open_seat, _player(melds=(_pon_meld("7z"),), discards=river)
    )


def _decision(policy_input, legal_actions=None):
    return DecisionContext(
        input=policy_input,
        legal_actions=_discards(policy_input)
        if legal_actions is None
        else legal_actions,
    )


def _both(decision):
    return (
        CHAMPION.choose_action_with_analysis(decision),
        CANDIDATE.choose_action_with_analysis(decision),
    )


class ChangedBehaviorTest(unittest.TestCase):
    def test_switches_to_the_open_opponents_river_tile(self):
        champion, candidate = _both(_decision(_situation()))

        self.assertEqual(champion.action, _action("1z"))
        self.assertEqual(candidate.action, _action("2z"))
        analysis = candidate.analysis
        self.assertIs(analysis.stage, _STAGE.SWITCHED)
        self.assertTrue(analysis.action_changed)
        self.assertEqual(analysis.parent_action, champion.action)
        self.assertEqual(analysis.open_opponent_seat, 2)
        self.assertEqual(analysis.genbutsu_tile_types, (_tile("2z").tile_type,))
        self.assertIs(analysis.parent_in_genbutsu, False)
        parent = analysis.parent_evaluation
        (alternative,) = analysis.alternative_evaluations
        self.assertEqual(
            (
                parent.post_discard_shanten,
                parent.current_ukeire_count,
                parent.retained_real_value,
                parent.yaku_route_value,
            ),
            (2, 20, 0, 0),
        )
        self.assertEqual(dataclasses.replace(alternative, action=parent.action), parent)
        self.assertEqual(analysis.qualifying_actions, (_action("2z"),))

    def test_suited_parent_switches_to_an_honor_genbutsu(self):
        champion, candidate = _both(_decision(_situation(SOUZU_HAND, "1z", drawn="2z")))

        self.assertEqual(champion.action, _action("5s"))
        self.assertEqual(candidate.action, _action("1z"))
        self.assertIs(candidate.analysis.stage, _STAGE.SWITCHED)

    def test_equal_alternatives_use_the_canonical_discard_order(self):
        decision = _decision(_situation(river="3z2z"))
        champion, candidate = _both(decision)

        self.assertEqual(champion.action, _action("1z"))
        self.assertEqual(candidate.action, _action("2z"))
        self.assertEqual(
            candidate.analysis.qualifying_actions, (_action("2z"), _action("3z"))
        )
        reversed_decision = _decision(
            decision.input, tuple(reversed(decision.legal_actions))
        )
        self.assertEqual(CANDIDATE.choose_action(reversed_decision), _action("2z"))

    def test_tsumogiri_identity_of_the_alternative_is_kept(self):
        policy_input = _situation(drawn="2z")
        legal = tuple(
            _action("2z", tsumogiri=True) if action.tile == _tile("2z") else action
            for action in _discards(policy_input)
        )

        selected = CANDIDATE.choose_action(_decision(policy_input, legal))

        self.assertEqual(selected, _action("2z", tsumogiri=True))

    def test_closed_hand_with_ankan_counts_as_closed(self):
        policy_input = _situation(
            "23m456p78s11p1z2z", drawn="2z", own_melds=(_ankan("9999p"),)
        )

        champion, candidate = _both(_decision(policy_input))

        self.assertEqual(champion.action, _action("1z"))
        self.assertEqual(candidate.action, _action("2z"))

    def test_all_last_in_normal_mode_still_applies(self):
        policy_input = _situation(round_wind=Wind.SOUTH, hand_number=4)

        _, candidate = _both(_decision(policy_input))

        self.assertEqual(candidate.action, _action("2z"))


class ProxyGateTest(unittest.TestCase):
    def _no_qualifying(self, policy_input, *failing_fields):
        champion, candidate = _both(_decision(policy_input))
        self.assertEqual(candidate.action, champion.action)
        analysis = candidate.analysis
        self.assertIs(analysis.stage, _STAGE.NO_QUALIFYING_ALTERNATIVE)
        self.assertEqual(analysis.qualifying_actions, ())
        self.assertFalse(analysis.action_changed)
        parent = analysis.parent_evaluation
        (alternative,) = analysis.alternative_evaluations
        for field in (
            "post_discard_shanten",
            "current_ukeire_count",
            "retained_real_value",
            "yaku_route_value",
        ):
            if field in failing_fields:
                continue
            self.assertEqual(getattr(alternative, field), getattr(parent, field))
        return parent, alternative

    def test_worse_shanten_keeps_the_parent(self):
        parent, alternative = self._no_qualifying(
            _situation(river="9p"), "post_discard_shanten", "current_ukeire_count"
        )
        self.assertEqual(
            (parent.post_discard_shanten, alternative.post_discard_shanten), (2, 3)
        )

    def test_lower_ukeire_keeps_the_parent(self):
        parent, alternative = self._no_qualifying(
            _situation(UKEIRE_HAND, "9m", drawn="4z"), "current_ukeire_count"
        )
        self.assertEqual(parent.action, _action("4m"))
        self.assertEqual(
            (parent.current_ukeire_count, alternative.current_ukeire_count), (59, 39)
        )

    def test_lower_retained_real_value_keeps_the_parent(self):
        # 1z表示で2zがドラ。2zを切るとretained real valueが1下がる。
        parent, alternative = self._no_qualifying(
            _situation(dora_indicators="1z"), "retained_real_value"
        )
        self.assertEqual(
            (parent.retained_real_value, alternative.retained_real_value), (1, 0)
        )

    def test_lower_yaku_route_value_keeps_the_parent(self):
        parent, alternative = self._no_qualifying(
            _situation(ROUTE_HAND, "5m", drawn="1z"), "yaku_route_value"
        )
        self.assertEqual(parent.action, _action("1z"))
        self.assertEqual(
            (parent.yaku_route_value, alternative.yaku_route_value), (1, 0)
        )

    def test_red_five_alternative_loses_retained_real_value(self):
        # 河は通常5sで、手の赤5sも同じ牌種の現物。赤を切ると価値が下がる。
        parent, alternative = self._no_qualifying(
            _situation(RED_HAND, "5s", drawn="2z"), "retained_real_value"
        )
        self.assertEqual(alternative.action, _action("0s"))

    def test_red_five_in_the_river_makes_the_normal_five_genbutsu(self):
        champion, candidate = _both(_decision(_situation(SOUZU_HAND, "0s", drawn="2z")))

        self.assertEqual(candidate.action, champion.action)
        self.assertEqual(candidate.action, _action("5s"))
        self.assertIs(candidate.analysis.stage, _STAGE.PARENT_ALREADY_GENBUTSU)
        self.assertIsNone(candidate.analysis.parent_evaluation)

    def test_parent_already_genbutsu_is_kept_without_evaluation(self):
        _, candidate = _both(_decision(_situation(river="1z2z")))

        self.assertEqual(candidate.action, _action("1z"))
        analysis = candidate.analysis
        self.assertIs(analysis.stage, _STAGE.PARENT_ALREADY_GENBUTSU)
        self.assertIs(analysis.parent_in_genbutsu, True)
        self.assertEqual(analysis.alternative_evaluations, ())

    def test_no_genbutsu_among_legal_discards_keeps_the_parent(self):
        champion, candidate = _both(_decision(_situation(river="4m6z")))

        self.assertEqual(candidate.action, champion.action)
        self.assertIs(candidate.analysis.stage, _STAGE.NO_GENBUTSU_ALTERNATIVE)
        self.assertIs(candidate.analysis.parent_in_genbutsu, False)
        self.assertIsNone(candidate.analysis.parent_evaluation)

    def test_selection_order_is_ukeire_value_route_then_canonical(self):
        policy_input = _situation(river="2z3z")
        parent = GenbutsuCandidateEvaluation(_action("1z"), 2, 20, 0, 0)
        rows = {
            _action("1z"): parent,
            _action("2z"): GenbutsuCandidateEvaluation(_action("2z"), 2, 21, 0, 0),
            _action("3z"): GenbutsuCandidateEvaluation(_action("3z"), 2, 21, 1, 0),
        }
        with mock.patch.object(
            module, "_evaluate", side_effect=lambda _p, row, _k, _e: rows[row.action]
        ):
            selected = CANDIDATE.choose_action(_decision(policy_input))
            rows[_action("2z")] = GenbutsuCandidateEvaluation(
                _action("2z"), 2, 22, 0, 0
            )
            by_ukeire = CANDIDATE.choose_action(_decision(policy_input))
            rows[_action("2z")] = GenbutsuCandidateEvaluation(
                _action("2z"), 2, 21, 1, 1
            )
            by_route = CANDIDATE.choose_action(_decision(policy_input))

        self.assertEqual(selected, _action("3z"))
        self.assertEqual(by_ukeire, _action("2z"))
        self.assertEqual(by_route, _action("2z"))


class OutOfGateTest(unittest.TestCase):
    def _assert_parent(self, policy_input, stage, legal=None):
        champion, candidate = _both(_decision(policy_input, legal))
        self.assertEqual(candidate.action, champion.action)
        self.assertIs(candidate.analysis.stage, stage)
        self.assertFalse(candidate.analysis.action_changed)
        self.assertIsNone(candidate.analysis.parent_evaluation)
        return candidate.analysis

    def test_post_call_discard_without_drawn_tile(self):
        self._assert_parent(_situation(drawn=None), _STAGE.NOT_NORMAL_TURN)

    def test_self_riichi(self):
        for state in (RiichiState.DECLARED, RiichiState.ACCEPTED):
            with self.subTest(state=state):
                policy_input = _with_player(_situation(), 0, _player(riichi=state))
                self._assert_parent(policy_input, _STAGE.SELF_RIICHI)

    def test_self_open_hand(self):
        policy_input = _situation(
            "23m456p78s11p1z2z", drawn="2z", own_melds=(_chi_meld("789p"),)
        )
        self._assert_parent(policy_input, _STAGE.SELF_OPEN_HAND)

    def test_opponent_riichi_declared_or_accepted(self):
        for state in (RiichiState.DECLARED, RiichiState.ACCEPTED):
            with self.subTest(state=state):
                policy_input = _with_player(
                    _situation(), 3, _player(riichi=state, discards="2z")
                )
                self._assert_parent(policy_input, _STAGE.OPPONENT_RIICHI)

    def test_no_open_opponent(self):
        self._assert_parent(_situation(open_seat=None), _STAGE.NO_OPEN_OPPONENT)

    def test_opponent_with_only_ankan_is_not_open(self):
        policy_input = _with_player(
            _situation(open_seat=None), 2, _player(melds=(_ankan("7777z"),))
        )
        self._assert_parent(policy_input, _STAGE.NO_OPEN_OPPONENT)

    def test_multiple_open_opponents(self):
        policy_input = _with_player(
            _situation(), 3, _player(melds=(_chi_meld("345s"),), discards="3z")
        )
        analysis = self._assert_parent(policy_input, _STAGE.MULTIPLE_OPEN_OPPONENTS)
        self.assertIsNone(analysis.open_opponent_seat)

    def test_all_last_special_mode(self):
        analysis = self._assert_parent(
            _situation(**ALL_LAST_TOP), _STAGE.NOT_NORMAL_SITUATION
        )
        self.assertEqual(analysis.open_opponent_seat, 2)

    def test_tenpai_reachable(self):
        self._assert_parent(
            _situation(TENPAI_HAND, "1z", drawn="5z"), _STAGE.TENPAI_REACHABLE
        )

    def test_non_discard_decisions_are_inherited(self):
        policy_input = _situation()
        discards = _discards(policy_input)
        pon = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_2,
            called_tile=_tile("1p"),
            consumed_tiles=(_tile("1p"), _tile("1p")),
        )
        cases = (
            (*discards, TsumoAction(actor=Seat.SEAT_0, winning_tile=_tile("3z"))),
            (*discards, RiichiAction(actor=Seat.SEAT_0)),
            (pon, PASS),
        )
        for legal in cases:
            with self.subTest(legal=type(legal[-1]).__name__):
                champion, candidate = _both(_decision(policy_input, legal))
                self.assertEqual(candidate, champion)
        self.assertIs(
            SingleOpenGenbutsuPlacementAwareSpeedCallPolicy._decide,
            PlacementAwareSpeedCallPolicy._decide,
        )


class ContractTest(unittest.TestCase):
    def test_open_opponent_definition_matches_is_open_opponent(self):
        inputs = (
            _situation(),
            _situation(open_seat=None),
            _with_player(_situation(), 3, _player(melds=(_chi_meld("345s"),))),
            _with_player(
                _situation(open_seat=None), 2, _player(melds=(_ankan("7777z"),))
            ),
            _with_player(_situation(), 1, _player(riichi=RiichiState.ACCEPTED)),
        )
        for policy_input in inputs:
            with self.subTest(policy_input=policy_input):
                self.assertEqual(
                    open_opponent_seats(policy_input),
                    tuple(
                        seat
                        for seat in range(4)
                        if is_open_opponent(policy_input, seat)
                    ),
                )

    def test_river_genbutsu_includes_called_discards_and_merges_red(self):
        policy_input = _situation(river="0s1z")
        player = policy_input.players[2]
        called = dataclasses.replace(player.discards[1], called_by=Seat.SEAT_3)
        policy_input = _with_player(
            policy_input,
            2,
            dataclasses.replace(player, discards=(player.discards[0], called)),
        )

        self.assertEqual(
            river_genbutsu_tile_types(policy_input, 2),
            frozenset({_tile("5s").tile_type, _tile("1z").tile_type}),
        )

    def test_trace_and_plain_execution_select_the_same_action(self):
        for policy_input in (
            _situation(),
            _situation(river="9p"),
            _situation(**ALL_LAST_TOP),
        ):
            with self.subTest(policy_input=policy_input):
                decision = _decision(policy_input)
                recorder = DecisionTraceRecorder()
                traced = execute_policy_with_trace(CANDIDATE, decision, recorder)
                self.assertEqual(execute_policy(CANDIDATE, decision), traced)
                (trace,) = recorder.snapshot()
                self.assertIsInstance(trace.analysis, SingleOpenGenbutsuAnalysis)
                self.assertEqual(trace.analysis.selected_action, traced)

    def test_identical_policy_input_gives_identical_decision(self):
        # PolicyInputは公開情報と自席の手牌だけを持つ。他家の手牌・山・結果など
        # hidden情報が異なる実局面も同じPolicyInputになり、同じ判断になる。
        first = SingleOpenGenbutsuPlacementAwareSpeedCallPolicy()
        second = SingleOpenGenbutsuPlacementAwareSpeedCallPolicy()

        self.assertEqual(
            first.choose_action_with_analysis(_decision(_situation())),
            second.choose_action_with_analysis(_decision(_situation())),
        )
        self.assertEqual(vars(first), {})

    def test_candidate_adds_no_finite_horizon_or_r5_evaluation(self):
        imported = {
            node.module
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.ImportFrom)
        }

        self.assertFalse(
            {
                "lisjong.policies.finite_horizon_completion",
                "lisjong.policies.targeted_honor_release_terminal_progression",
                "lisjong.belief.exact_wait_ground_truth",
            }
            & imported
        )
        self.assertFalse(any(name.startswith("lisjong.learning") for name in imported))

    def test_analysis_rejects_inconsistent_change(self):
        analysis = CANDIDATE.choose_action_with_analysis(
            _decision(_situation())
        ).analysis
        with self.assertRaises(ValueError):
            dataclasses.replace(analysis, action_changed=False)
        with self.assertRaises(ValueError):
            dataclasses.replace(
                analysis,
                selected_action=analysis.parent_action,
                action_changed=False,
            )
        with self.assertRaises(ValueError):
            dataclasses.replace(analysis, qualifying_actions=())

    def test_public_export_is_stateless_and_picklable(self):
        self.assertIsInstance(CANDIDATE, PlacementAwareSpeedCallPolicy)
        self.assertIs(
            pickle.loads(pickle.dumps(SingleOpenGenbutsuPlacementAwareSpeedCallPolicy)),
            SingleOpenGenbutsuPlacementAwareSpeedCallPolicy,
        )

    def test_static_import_check_passes(self):
        report = check_candidate_imports(_MODULE)
        self.assertTrue(report.passed, msg=report.violations)

    def test_purity_check_passes_across_processes(self):
        decisions = (
            _decision(_situation()),
            _decision(_situation(river="3z2z")),
            _decision(_situation(river="9p")),
            _decision(_situation(UKEIRE_HAND, "9m", drawn="4z")),
        )
        report = check_candidate_purity(_FACTORY, decisions)
        self.assertTrue(report.passed, msg=(report.mismatches, report.errors))
        self.assertEqual(report.decision_count, len(decisions))


if __name__ == "__main__":
    unittest.main()
