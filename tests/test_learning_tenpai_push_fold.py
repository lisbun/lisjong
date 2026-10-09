"""#288: 聴牌PUSH/FOLDのゲート判定・候補選択・対比較用Policy・比較式の境界を固定する。

役・点数の計算は差し替えたfakeで行い、native拡張に依存しない。実際の`evaluate_win()`を使う
testは`_lisjong_native`が使える環境だけで実行し、`LISJONG_REQUIRE_NATIVE=1`ではskipしない。
"""

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_placement_aware_speed_call_policy import (
    ALL_LAST_TOP,
    _chi_meld,
    _discards,
    _input,
    _player,
    _pon_meld,
    _tile,
)

import lisjong.learning.tenpai_push_fold as gate_module
import lisjong.learning.tenpai_push_fold_value as value_module
from lisjong.belief.canonical_axes import tile_type_index
from lisjong.hand_evaluation import calculate_shanten, scoring
from lisjong.hand_evaluation.scoring import (
    EvaluationStatus,
    RiichiStatus,
    ScoringBackendUnavailableError,
    UraDoraExcluded,
    WinEvaluation,
    WinMethod,
)
from lisjong.learning.riichi_wait_estimator import (
    CLIP_EPSILON,
    FEATURE_SET,
    LogisticWaitModel,
)
from lisjong.learning.riichi_wait_mawashi_policy import RiichiWaitModelError
from lisjong.learning.tenpai_push_fold import (
    GateKind,
    OwnWaitValue,
    TenpaiGate,
    TenpaiPushFoldError,
    TenpaiPushFoldPairAnalysis,
    TenpaiPushFoldPairPolicy,
    evaluate_tenpai_gate,
    own_wait_value,
)
from lisjong.learning.tenpai_push_fold_value import (
    BucketSpec,
    PushFoldTables,
    TenpaiSideTable,
    comparison_values,
)
from lisjong.policies import PlacementAwareSpeedCallPolicy
from lisjong.policy_contract.action import DiscardAction, PassAction, RiichiAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat
from lisjong.structural_efficiency import post_discard_concealed_hand

try:
    import _lisjong_native
except ImportError:
    _lisjong_native = None

_REQUIRE_NATIVE = os.environ.get("LISJONG_REQUIRE_NATIVE") == "1"
_NATIVE_SCORING = (
    getattr(_lisjong_native, "SCORING_API_VERSION", None)
    == scoring.REQUIRED_NATIVE_SCORING_API_VERSION
)

# 123m 456p 789s 23s 55s + 1z: 1z切りだけが聴牌を保つ（待ち1s・4s）。
CLOSED_TENPAI = "123m456p789s2355s1z"
# 7zポン + 123m 456p 23s 55s + 1z: 役牌のある副露聴牌。
OPEN_TENPAI = "123m456p23s55s1z"
# 七対子の聴牌で、5z・6zのどちらを切っても聴牌を保つ。
TWO_WAY_TENPAI = "1199m2288p3377s5z6z"
RIICHI = RiichiAction(actor=Seat.SEAT_0)
MODEL = LogisticWaitModel(weights=(("class_rank_1", 3.0),))
CHAMPION = PlacementAwareSpeedCallPolicy()


def _scored(hand, context) -> WinEvaluation:
    points = 2000 if context.method is WinMethod.RON else 2700
    return WinEvaluation(EvaluationStatus.SCORED, SimpleNamespace(winner_points=points))


def _no_yaku(hand, context) -> WinEvaluation:
    return WinEvaluation(EvaluationStatus.NO_YAKU)


def _decision(context, *, riichi: bool = False, discards=None) -> DecisionContext:
    discards = _discards(context) if discards is None else discards
    return DecisionContext(
        input=context, legal_actions=((RIICHI, *discards) if riichi else discards)
    )


def _discard(tile: str) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=_tile(tile), tsumogiri=False)


def _gate(decision, *, c0=None, evaluate=_scored, raw=None) -> TenpaiGate | None:
    c0 = CHAMPION.choose_action_with_analysis(decision) if c0 is None else c0
    if raw is None:
        return evaluate_tenpai_gate(decision, c0, MODEL, evaluate=evaluate)
    with _ron_legal(raw):
        return evaluate_tenpai_gate(decision, c0, MODEL, evaluate=evaluate)


def _ron_legal(raw_by_tile: dict[str, int], default: int = 4000):
    """ロン合法確率のrawを牌種ごとに差し替える（候補選択の順序を固定するため）。"""
    table = [default] * 34
    for tile, raw in raw_by_tile.items():
        table[tile_type_index(_tile(tile).tile_type)] = raw
    return patch.object(
        gate_module,
        "estimate_riichi_ron_legal_belief",
        return_value=SimpleNamespace(ron_legal_probability_raw=tuple(table)),
    )


def _with_own(context, **changes):
    players = list(context.players)
    players[0] = replace(players[0], **changes)
    return replace(context, players=tuple(players))


class GateConditionTest(unittest.TestCase):
    def test_closed_discard_decision_is_kind_b(self):
        gate = _gate(_decision(_input(CLOSED_TENPAI, riichi_discards="3z4z")))
        self.assertIs(gate.kind, GateKind.CLOSED_DISCARD)
        self.assertIs(gate.riichi_seat, Seat.SEAT_1)
        self.assertEqual(gate.c0_action, _discard("1z"))
        self.assertEqual(gate.push_action, _discard("1z"))

    def test_open_discard_decision_is_kind_c(self):
        context = _input(
            OPEN_TENPAI, own_melds=(_pon_meld("7z"),), riichi_discards="3z4z"
        )
        gate = _gate(_decision(context))
        self.assertIs(gate.kind, GateKind.OPEN_DISCARD)
        self.assertEqual(gate.push_action, _discard("1z"))

    def test_riichi_decision_is_kind_a_and_never_uses_the_tenpai_fold_side(self):
        context = _input(CLOSED_TENPAI, riichi_discards="3z4z")
        gate = _gate(_decision(context, riichi=True), raw={"1z": 0})
        self.assertIs(gate.kind, GateKind.RIICHI)
        self.assertEqual(gate.c0_action, RIICHI)
        self.assertEqual(gate.push_action, _discard("1z"))
        self.assertEqual(gate.fold_action, _discard("1z"))
        self.assertTrue(gate.fold_keeps_tenpai)
        self.assertIsNone(gate.fold_wait)

    def test_each_condition_outside_the_gate_returns_none(self):
        base = _input(CLOSED_TENPAI, riichi_discards="3z4z")
        two_riichi = list(base.players)
        two_riichi[2] = _player(riichi=RiichiState.ACCEPTED, discards="2z")
        riichi_without_discard = list(base.players)
        riichi_without_discard[1] = _player(riichi=RiichiState.DECLARED)
        dead_waits = _input("123m456p789s111z2z5z", riichi_discards="2z2z2z5z5z5z")
        yakuless_route = _input(
            OPEN_TENPAI, own_melds=(_pon_meld("9p"),), riichi_discards="3z4z"
        )
        cases = {
            "1 self in riichi": _decision(_with_own(base, riichi=RiichiState.DECLARED)),
            "2 no riichi": _decision(_input(CLOSED_TENPAI)),
            "2 two riichi": _decision(replace(base, players=tuple(two_riichi))),
            "2 riichi seat has no discard": _decision(
                replace(base, players=tuple(riichi_without_discard))
            ),
            "3 all last": _decision(
                _input(CLOSED_TENPAI, riichi_discards="3z4z", **ALL_LAST_TOP)
            ),
            "4 one tile type": _decision(base, discards=(_discard("1z"),)),
            "4 riichi without a discard choice": DecisionContext(
                input=base, legal_actions=(RIICHI,)
            ),
            "5 fold after the role filter": _decision(yakuless_route),
            "5 not tenpai": _decision(
                _input("34m678p68p234s55s9s1m", riichi_discards="3z4z")
            ),
            "7 no live wait": _decision(dead_waits),
        }
        for name, decision in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(_gate(decision))

    def test_condition_6_requires_the_push_discard_to_keep_tenpai(self):
        decision = _decision(_input(CLOSED_TENPAI, riichi_discards="3z4z"))
        self.assertIsNotNone(_gate(decision))
        c0 = PolicyDecision(action=_discard("9s"))
        self.assertIsNone(_gate(decision, c0=c0))

    def test_non_discard_champion_actions_are_outside_the_gate(self):
        decision = _decision(_input(CLOSED_TENPAI, riichi_discards="3z4z"))
        c0 = PolicyDecision(action=PassAction(actor=Seat.SEAT_0))
        self.assertIsNone(_gate(decision, c0=c0))

    def test_yakuless_open_tenpai_is_in_the_gate_with_no_yaku(self):
        context = _input(
            "456p23s55s1z",
            own_melds=(_chi_meld("123m"), _pon_meld("9p")),
            riichi_discards="3z4z",
        )
        gate = _gate(_decision(context), evaluate=_no_yaku)
        self.assertIs(gate.kind, GateKind.OPEN_DISCARD)
        self.assertFalse(gate.push_wait.has_yaku)
        self.assertEqual((gate.push_wait.ron_count, gate.push_wait.tsumo_count), (0, 0))


class FoldCandidateTest(unittest.TestCase):
    def setUp(self):
        self.decision = _decision(_input(CLOSED_TENPAI, riichi_discards="3z4z"))

    def test_lowest_ron_legal_probability_wins_over_shanten(self):
        gate = _gate(self.decision, raw={"7s": 10, "1z": 20})
        self.assertEqual(gate.fold_action, _discard("7s"))
        self.assertEqual((gate.push_ron_legal_raw, gate.fold_ron_legal_raw), (20, 10))
        self.assertTrue(gate.has_fold_candidate)
        self.assertFalse(gate.fold_keeps_tenpai)
        self.assertIsNone(gate.fold_wait)

    def test_equal_probability_prefers_the_smaller_shanten(self):
        gate = _gate(self.decision, raw={"7s": 10, "1z": 10})
        self.assertEqual(gate.fold_action, _discard("1z"))
        self.assertFalse(gate.has_fold_candidate)

    def test_equal_shanten_prefers_the_larger_ukeire(self):
        # 7s切りは受け入れ11枚、9s切りは15枚。canonicalな打牌順は7sが先。
        gate = _gate(self.decision, raw={"7s": 10, "9s": 10})
        self.assertEqual(gate.fold_action, _discard("9s"))

    def test_equal_ukeire_falls_back_to_the_canonical_discard_order(self):
        # 2s切りと3s切りは、どちらも受け入れ19枚。
        gate = _gate(self.decision, raw={"3s": 10, "2s": 10})
        self.assertEqual(gate.fold_action, _discard("2s"))

    def test_red_five_and_tsumogiri_keep_their_action_identity(self):
        context = _input("123m456p789s2305s1z", riichi_discards="3z4z", drawn="5s")
        red = _discard("0s")
        normal = _discard("5s")
        tsumogiri = replace(normal, tsumogiri=True)
        decision = _decision(context, discards=(tsumogiri, red, normal, _discard("1z")))
        gate = _gate(decision, raw={"5s": 10})
        self.assertEqual(gate.fold_action, normal)
        decision = _decision(context, discards=(tsumogiri, red, _discard("1z")))
        self.assertEqual(_gate(decision, raw={"5s": 10}).fold_action, tsumogiri)

    def test_fold_that_keeps_tenpai_gets_a_wait_value_for_discards_only(self):
        context = _input(TWO_WAY_TENPAI, riichi_discards="3z4z")
        c0 = CHAMPION.choose_action(_decision(context))
        other = "6z" if c0 == _discard("5z") else "5z"
        gate = _gate(_decision(context), raw={other: 10})
        self.assertEqual(gate.fold_action, _discard(other))
        self.assertTrue(gate.fold_keeps_tenpai)
        self.assertIsInstance(gate.fold_wait, OwnWaitValue)


class OwnWaitValueTest(unittest.TestCase):
    def test_counts_and_weighted_points_use_the_remaining_tiles(self):
        # 待ちは1s・4s。4sが場に1枚見えているので、残りは4枚と3枚。
        context = _input(CLOSED_TENPAI, riichi_discards="3z4z4s")

        def evaluate(hand, context_):
            points = 1000 if hand.winning_tile == _tile("1s") else 8000
            return WinEvaluation(
                EvaluationStatus.SCORED, SimpleNamespace(winner_points=points)
            )

        value = own_wait_value(context, _discard("1z"), riichi=False, evaluate=evaluate)
        self.assertEqual((value.ron_count, value.tsumo_count), (7, 7))
        self.assertEqual(value.ron_points, (4 * 1000 + 3 * 8000) / 7)
        self.assertFalse(value.discard_furiten)
        self.assertTrue(value.has_yaku)

    def test_tsumo_only_yaku_gives_no_ron_count(self):
        def evaluate(hand, context_):
            if context_.method is WinMethod.RON:
                return WinEvaluation(EvaluationStatus.NO_YAKU)
            return _scored(hand, context_)

        value = own_wait_value(
            _input(CLOSED_TENPAI, riichi_discards="3z4z"),
            _discard("1z"),
            riichi=False,
            evaluate=evaluate,
        )
        self.assertEqual((value.ron_count, value.ron_points), (0, 0.0))
        self.assertEqual((value.tsumo_count, value.tsumo_points), (8, 2700.0))
        self.assertTrue(value.has_yaku)

    def test_a_wait_tile_in_the_own_river_removes_the_ron_side(self):
        context = _input(CLOSED_TENPAI, riichi_discards="3z4z")
        furiten = _with_own(context, discards=_player(discards="4s").discards)
        value = own_wait_value(furiten, _discard("1z"), riichi=False, evaluate=_scored)
        self.assertTrue(value.discard_furiten)
        self.assertEqual((value.ron_count, value.ron_points), (0, 0.0))
        self.assertEqual(value.tsumo_count, 7)
        self.assertTrue(value.has_yaku)

    def test_discarding_a_wait_tile_is_furiten(self):
        # 123m 456p 789s 11s 2s 3s 4s: 1s切りで1s・4s待ち（1sは自分の打牌）。
        context = _input("123m456p789s11234s", riichi_discards="3z4z")
        value = own_wait_value(context, _discard("1s"), riichi=False, evaluate=_scored)
        self.assertTrue(value.discard_furiten)
        self.assertEqual(value.ron_count, 0)
        self.assertGreater(value.tsumo_count, 0)

    def test_riichi_side_scores_with_riichi_and_ura_dora_excluded(self):
        contexts = []

        def evaluate(hand, context_):
            contexts.append(context_)
            return _scored(hand, context_)

        context = _input(CLOSED_TENPAI, riichi_discards="3z4z")
        own_wait_value(context, _discard("1z"), riichi=True, evaluate=evaluate)
        self.assertTrue(
            all(
                c.riichi is RiichiStatus.RIICHI
                and isinstance(c.ura_dora, UraDoraExcluded)
                for c in contexts
            )
        )
        contexts.clear()
        own_wait_value(context, _discard("1z"), riichi=False, evaluate=evaluate)
        self.assertTrue(
            all(
                c.riichi is RiichiStatus.NONE and c.ura_dora.tiles == ()
                for c in contexts
            )
        )

    def test_an_unseen_red_five_is_one_of_the_remaining_copies(self):
        # 123m 456p 789s 34s 11z + 9m: 待ちは2s・5s。赤5sは見えていない。
        winning_tiles = []

        def evaluate(hand, context_):
            if context_.method is WinMethod.RON:
                winning_tiles.append(hand.winning_tile)
            return _scored(hand, context_)

        context = _input("123m456p789s34s11z9m", riichi_discards="3z4z")
        value = own_wait_value(context, _discard("9m"), riichi=False, evaluate=evaluate)
        self.assertEqual(value.ron_count, 8)
        self.assertEqual(
            sorted(winning_tiles, key=repr),
            sorted((_tile("2s"), _tile("0s"), _tile("5s")), key=repr),
        )
        winning_tiles.clear()
        seen = _input(
            "123m456p789s34s11z9m", riichi_discards="3z4z", dora_indicators="0s"
        )
        value = own_wait_value(seen, _discard("9m"), riichi=False, evaluate=evaluate)
        self.assertEqual(value.ron_count, 7)
        self.assertNotIn(_tile("0s"), winning_tiles)

    def test_a_structural_wait_that_is_not_a_win_is_an_error(self):
        with self.assertRaises(ValueError):
            own_wait_value(
                _input(CLOSED_TENPAI, riichi_discards="3z4z"),
                _discard("1z"),
                riichi=False,
                evaluate=lambda hand, context: WinEvaluation(
                    EvaluationStatus.NOT_COMPLETE
                ),
            )

    @unittest.skipUnless(_NATIVE_SCORING or _REQUIRE_NATIVE, "needs _lisjong_native")
    def test_real_scoring_separates_yaku_by_win_method(self):
        # 123m 456p 789s 13s 55s: 嵌2s待ち。門前ロンは役なし、ツモは門前清自摸和。
        closed = _input("123m456p789s1355s1z", riichi_discards="3z4z")
        dama = own_wait_value(closed, _discard("1z"), riichi=False)
        self.assertEqual((dama.ron_count, dama.tsumo_count), (0, 4))
        self.assertEqual(dama.ron_points, 0.0)
        self.assertGreater(dama.tsumo_points, 0)
        self.assertTrue(dama.has_yaku)
        riichi = own_wait_value(closed, _discard("1z"), riichi=True)
        self.assertEqual((riichi.ron_count, riichi.tsumo_count), (4, 4))
        self.assertGreater(riichi.tsumo_points, dama.tsumo_points)
        yakuless = _input(
            "456p23s55s1z",
            own_melds=(_chi_meld("123m"), _pon_meld("9p")),
            riichi_discards="3z4z",
        )
        value = own_wait_value(yakuless, _discard("1z"), riichi=False)
        self.assertEqual((value.ron_count, value.tsumo_count), (0, 0))
        self.assertFalse(value.has_yaku)


class RiichiDeclarationPredictionTest(unittest.TestCase):
    def test_prediction_matches_the_champion_discard_after_declaring(self):
        for hand in (
            CLOSED_TENPAI,
            "234m456p678s2355s9m",
            TWO_WAY_TENPAI,
            "123m456p789s2305s1z",
        ):
            with self.subTest(hand=hand):
                context = _input(hand, riichi_discards="3z4z")
                gate = _gate(_decision(context, riichi=True))
                self.assertIs(gate.kind, GateKind.RIICHI)
                declared = _with_own(context, riichi=RiichiState.DECLARED)
                # リーチ後の次の判断では、聴牌を保つ打牌だけが合法になる。
                legal = tuple(
                    action
                    for action in _discards(context)
                    if calculate_shanten(
                        post_discard_concealed_hand(
                            context.own_hand.concealed_tiles, action.tile
                        )
                    )
                    == 0
                )
                self.assertIn(gate.push_action, legal)
                actual = CHAMPION.choose_action(
                    DecisionContext(input=declared, legal_actions=legal)
                )
                self.assertEqual(gate.push_action, actual)


class PairPolicyTest(unittest.TestCase):
    def setUp(self):
        self.context = _input(CLOSED_TENPAI, riichi_discards="3z4z")
        self.target = _decision(self.context)

    def _policy(self, target=None) -> TenpaiPushFoldPairPolicy:
        return TenpaiPushFoldPairPolicy(
            MODEL, self.target if target is None else target, evaluate=_scored
        )

    def test_only_the_target_decision_discards_the_fold_candidate(self):
        with _ron_legal({"7s": 10}):
            result = self._policy().choose_action_with_analysis(self.target)
        self.assertEqual(result.action, _discard("7s"))
        self.assertIsInstance(result.analysis, TenpaiPushFoldPairAnalysis)
        self.assertEqual(result.analysis.gate.push_action, _discard("1z"))

    def test_riichi_target_discards_without_declaring(self):
        target = _decision(self.context, riichi=True)
        with _ron_legal({"7s": 10}):
            action = self._policy(target).choose_action(target)
        self.assertEqual(action, _discard("7s"))

    def test_other_decisions_return_the_champion_decision(self):
        policy = self._policy()
        other_gate_decision = _decision(_input(CLOSED_TENPAI, riichi_discards="3z4z2z"))
        decisions = (
            other_gate_decision,
            _decision(self.context, riichi=True),
            _decision(_input(CLOSED_TENPAI)),
            _decision(_input("34m678p68p234s55s9s1m", riichi_discards="3z4z")),
            _decision(_input(CLOSED_TENPAI, riichi_discards="3z4z", **ALL_LAST_TOP)),
        )
        with _ron_legal({"7s": 10}) as estimator:
            for decision in decisions:
                with self.subTest(decision=decision):
                    self.assertEqual(
                        policy.choose_action_with_analysis(decision),
                        CHAMPION.choose_action_with_analysis(decision),
                    )
            estimator.assert_not_called()

    def test_target_without_a_fold_candidate_is_an_error(self):
        with _ron_legal({"1z": 0}), self.assertRaises(TenpaiPushFoldError):
            self._policy().choose_action(self.target)
        outside = _decision(_input(CLOSED_TENPAI))
        with self.assertRaises(TenpaiPushFoldError):
            self._policy(outside).choose_action(outside)

    def test_the_policy_keeps_no_state_across_decisions(self):
        policy = self._policy()
        before = dict(vars(policy))
        other = _decision(_input(CLOSED_TENPAI, riichi_discards="3z4z2z"))
        with _ron_legal({"7s": 10}):
            first = policy.choose_action_with_analysis(self.target)
            policy.choose_action_with_analysis(other)
            again = policy.choose_action_with_analysis(self.target)
        self.assertEqual(first, again)
        self.assertEqual(vars(policy), before)

    def test_construction_fails_without_the_scoring_backend(self):
        with (
            patch.object(
                scoring,
                "_load_native",
                side_effect=ScoringBackendUnavailableError("missing"),
            ),
            self.assertRaises(ScoringBackendUnavailableError),
        ):
            TenpaiPushFoldPairPolicy(MODEL, self.target)

    def test_construction_fails_on_a_different_wait_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "lisjong-riichi-wait-selection-v1",
                        "feature_set": FEATURE_SET,
                        "clip_epsilon": CLIP_EPSILON,
                        "models": {
                            "estimator_logistic": {
                                "feature_set": FEATURE_SET,
                                "weights": {"class_rank_1": 3.0},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RiichiWaitModelError):
                TenpaiPushFoldPairPolicy.from_selection(path, self.target)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(gate_module, "require_scoring_backend"):
                policy = TenpaiPushFoldPairPolicy.from_selection(
                    path, self.target, expected_sha256=digest
                )
            self.assertIsInstance(policy, TenpaiPushFoldPairPolicy)

    def test_arguments_are_type_checked(self):
        with self.assertRaises(TypeError):
            TenpaiPushFoldPairPolicy(object(), self.target, evaluate=_scored)
        with self.assertRaises(TypeError):
            TenpaiPushFoldPairPolicy(MODEL, self.context, evaluate=_scored)


def _wait(ron_count, tsumo_count, ron_points=0.0, tsumo_points=0.0) -> OwnWaitValue:
    return OwnWaitValue(
        ron_count=ron_count,
        tsumo_count=tsumo_count,
        ron_points=ron_points,
        tsumo_points=tsumo_points,
        discard_furiten=False,
        has_yaku=bool(ron_count or tsumo_count),
    )


class ComparisonValueTest(unittest.TestCase):
    BUCKETS = BucketSpec(wall_upper_bounds=(20, 40), count_upper_bounds=(2, 4))

    def setUp(self):
        # 残りツモ山50（w2）、本場1、供託2。リーチ者（席1）は親。
        context = _input(CLOSED_TENPAI, riichi_discards="3z4z")
        self.context = replace(
            context, round=replace(context.round, honba=1, riichi_sticks=2)
        )

    def _tables(self, **changes) -> PushFoldTables:
        tables = PushFoldTables(
            buckets=self.BUCKETS,
            loss={"dealer": 9000.0, "non_dealer": 6000.0},
            tenpai={
                "discard": TenpaiSideTable(
                    q_ron={"w2.c3": 0.25},
                    q_tsumo={"w2.c3": 0.125, "w2.c1": 0.0625},
                    r_t={"w2.c3": -1000.0, "w2.c1": -500.0, "w2.c0": -250.0},
                ),
                "riichi": TenpaiSideTable(
                    q_ron={"w2.c3": 0.5},
                    q_tsumo={"w2.c3": 0.25},
                    r_t={"w2.c3": -2000.0},
                ),
            },
            r_f={"discard": {"w2": -1500.0}, "riichi": {"w2": -800.0}},
            uplift={"ron": 1000.0, "tsumo": 2000.0},
        )
        return replace(tables, **changes)

    def _gate(self, kind=GateKind.CLOSED_DISCARD, **changes) -> TenpaiGate:
        gate = TenpaiGate(
            kind=kind,
            riichi_seat=Seat.SEAT_1,
            c0_action=_discard("1z"),
            push_action=_discard("1z"),
            fold_action=_discard("7s"),
            push_ron_legal_raw=2048,
            fold_ron_legal_raw=0,
            fold_keeps_tenpai=False,
            push_wait=_wait(8, 8, 4000.0, 6000.0),
            fold_wait=None,
        )
        return replace(gate, **changes)

    def test_bucket_keys(self):
        buckets = self.BUCKETS
        self.assertEqual(
            [buckets.wall_key(wall) for wall in (0, 20, 21, 40, 41)],
            ["w0", "w0", "w1", "w1", "w2"],
        )
        self.assertEqual(
            [buckets.count_bin(count) for count in (0, 1, 2, 3, 4, 5)],
            [0, 1, 1, 2, 2, 3],
        )
        with self.assertRaises(ValueError):
            BucketSpec(wall_upper_bounds=(40, 20), count_upper_bounds=(2, 4))

    def test_discard_gate_values(self):
        push, fold = comparison_values(self.context, self._gate(), self._tables())
        # 供託2000 + 本場300を和了の収入に、本場300を放銃の支出に足す。
        continuation = 0.25 * (4000 + 2300) + 0.125 * (6000 + 2300) - 1000
        self.assertAlmostEqual(push, -0.25 * 9300 + 0.75 * continuation)
        self.assertAlmostEqual(fold, -1500.0)

    def test_riichi_gate_uses_its_own_tables_and_the_uplift(self):
        gate = self._gate(GateKind.RIICHI, c0_action=RIICHI)
        push, fold = comparison_values(self.context, gate, self._tables())
        continuation = 0.5 * (4000 + 1000 + 2300) + 0.25 * (6000 + 2000 + 2300) - 2000
        self.assertAlmostEqual(push, -0.25 * 9300 + 0.75 * continuation)
        self.assertAlmostEqual(fold, -800.0)

    def test_fold_that_keeps_tenpai_uses_the_tenpai_side(self):
        gate = self._gate(
            fold_keeps_tenpai=True,
            fold_wait=_wait(0, 2, 0.0, 3000.0),
            fold_ron_legal_raw=819,
        )
        _, fold = comparison_values(self.context, gate, self._tables())
        p = 819 / 8192
        self.assertAlmostEqual(
            fold, -p * 9300 + (1 - p) * (0.0625 * (3000 + 2300) - 500)
        )

    def test_zero_count_skips_the_q_table(self):
        gate = self._gate(push_wait=_wait(0, 0))
        push, _ = comparison_values(self.context, gate, self._tables())
        self.assertAlmostEqual(push, -0.25 * 9300 + 0.75 * -250)

    def test_riichi_seat_dealer_selects_the_loss(self):
        gate = self._gate(riichi_seat=Seat.SEAT_2)
        push, _ = comparison_values(self.context, gate, self._tables())
        continuation = 0.25 * (4000 + 2300) + 0.125 * (6000 + 2300) - 1000
        self.assertAlmostEqual(push, -0.25 * 6300 + 0.75 * continuation)

    def test_a_missing_bucket_gives_no_values(self):
        cases = {
            "loss": {"loss": {}},
            "r_f": {"r_f": {"discard": {}, "riichi": {}}},
            "tenpai": {"tenpai": {}},
            "q": {
                "tenpai": {
                    "discard": TenpaiSideTable(q_ron={}, q_tsumo={}, r_t={"w2.c3": 0.0})
                }
            },
        }
        for name, changes in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(
                    comparison_values(
                        self.context, self._gate(), self._tables(**changes)
                    )
                )
        riichi = self._gate(GateKind.RIICHI, c0_action=RIICHI)
        self.assertIsNone(
            comparison_values(self.context, riichi, self._tables(uplift={}))
        )


class InformationBoundaryTest(unittest.TestCase):
    # `lisjong.belief`のpackageが`exact_wait_ground_truth`を読み込むので、この名前は
    # 直接のimportだけを調べる。
    LOADED_FORBIDDEN = (
        "riichi_deal_in_source",
        "ron_legal_ground_truth",
        "ron_legal_source",
        "riichi_ron_label",
        "hand_belief_source",
        "tenpai_push_fold_source",
        "tenpai_push_fold_evaluation",
        "lisjong_arena",
    )
    FORBIDDEN = (*LOADED_FORBIDDEN, "exact_wait_ground_truth")

    def test_inference_modules_do_not_import_label_or_source_modules(self):
        for module in (gate_module, value_module):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            imported = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            } | {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            for forbidden in self.FORBIDDEN:
                self.assertFalse(
                    any(forbidden in name for name in imported), (module, forbidden)
                )

    def test_importing_the_inference_modules_loads_no_label_or_source_module(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "import lisjong.learning.tenpai_push_fold\n"
                "import lisjong.learning.tenpai_push_fold_value\n"
                f"forbidden = {self.LOADED_FORBIDDEN!r}\n"
                "loaded = [name for name in sys.modules "
                "if any(part in name for part in forbidden)]\n"
                "assert not loaded, loaded\n"
                "assert 'torch' not in sys.modules\n"
                "print('ok')\n",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
