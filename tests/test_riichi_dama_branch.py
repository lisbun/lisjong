"""Issue #352: リーチ判断の特徴APIとD arm用のリーチ抑止wrapper（riichi-dama-branch-v1）。

点数は実際の`evaluate_win()`で計算するため、native scoring backendが使えない環境では
skipする（CIは`LISJONG_REQUIRE_NATIVE=1`でskipしない）。
"""

import json
import os
import unittest
from dataclasses import replace

from test_placement_aware_speed_call_policy import (
    ALL_LAST_TOP,
    _discards,
    _input,
    _player,
    _pon_meld,
    _tile,
)

from lisjong.candidate_boundary import check_candidate_imports
from lisjong.hand_evaluation import scoring
from lisjong.hand_evaluation.scoring import EvaluationStatus, ScoreLimit
from lisjong.policies import PlacementAwareSpeedCallPolicy
from lisjong.policies.riichi_dama_branch import (
    RIICHI_DAMA_FEATURES_VERSION,
    RIICHI_FIRST_DAMA_BRANCH_VERSION,
    BranchStatus,
    RiichiDamaAnalysisError,
    RiichiDamaBranchError,
    RiichiDamaBranchPolicy,
    RiichiDecisionFeatures,
    RiichiFirstDamaBranchPolicy,
    RoundIdentity,
    WaitScore,
    analyze_riichi_decision,
    parse_riichi_decision_features,
    riichi_decision_features_to_value,
)
from lisjong.policy_contract.action import (
    DiscardAction,
    RiichiAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.wind import Wind

try:
    import _lisjong_native
except ImportError:
    _lisjong_native = None

_NATIVE_SCORING = (
    getattr(_lisjong_native, "SCORING_API_VERSION", None)
    == scoring.REQUIRED_NATIVE_SCORING_API_VERSION
)
_REQUIRE_NATIVE = os.environ.get("LISJONG_REQUIRE_NATIVE") == "1"
_needs_scoring = unittest.skipUnless(
    _NATIVE_SCORING or _REQUIRE_NATIVE, "native scoring backend is not available"
)

SELF = Seat.SEAT_0
RIICHI = RiichiAction(actor=SELF)
CHAMPION = PlacementAwareSpeedCallPolicy()

# 123m 456p 789s 23s 55s + 1z: 打1zで平和の両面聴牌（待ち1s・4s）。
PINFU = "123m456p789s2355s1z"
# 123m 456p 789s 13s 55s + 1z: 打1zで嵌張2s待ち。ダマのロンは役なし。
KANCHAN = "123m456p789s1355s1z"
# 234m 234p 234s 55m 67s + 1z: 打1zでタンヤオ・平和・三色（5s・8s待ち）。
SANSHOKU = "234m234p234s55m67s1z"
# 平和の聴牌形に4sを加えた14枚。打4sで待ち1s・4s、宣言牌自身で振聴になる。
PINFU_WITH_4S = "123m456p789s234s55s"
# 平和の聴牌形に赤5pを加えた14枚。打0pなら赤なし、打5pなら赤1枚が手に残る。
PINFU_WITH_RED = "123m456p789s2355s0p"
# 七対子の聴牌（打5zで6z単騎）。
CHIITOI = "1199m2288p3377s5z6z"
# 国士無双13面待ち + 5m。
KOKUSHI = "19m19p19s1234567z5m"


def _own_river(context, spec: str, **changes):
    players = list(context.players)
    players[0] = _player(score=players[0].score, discards=spec, **changes)
    return replace(context, players=tuple(players))


def _riichi_decision(concealed: str, *, river: str = "9m", **kwargs) -> DecisionContext:
    context = _input(concealed, **kwargs)
    if river is not None:
        context = _own_river(context, river)
    return DecisionContext(input=context, legal_actions=(RIICHI, *_discards(context)))


def _discard(tile: str, *, tsumogiri: bool = False) -> DiscardAction:
    return DiscardAction(actor=SELF, tile=_tile(tile), tsumogiri=tsumogiri)


def _wait(features: RiichiDecisionFeatures, tile: str):
    (wait,) = [w for w in features.waits if w.tile_type == _tile(tile).tile_type]
    return wait


@_needs_scoring
class WaitAndScoreTest(unittest.TestCase):
    def test_pinfu_waits_counts_and_four_scores(self) -> None:
        features = analyze_riichi_decision(_riichi_decision(PINFU), _tile("1z"))
        self.assertEqual(
            features.wait_tile_types, (_tile("1s").tile_type, _tile("4s").tile_type)
        )
        self.assertEqual([w.unseen_count for w in features.waits], [4, 4])
        self.assertEqual(features.total_unseen_count, 8)
        self.assertFalse(features.furiten)
        wait = _wait(features, "4s")
        self.assertEqual(
            wait.dama_ron,
            WaitScore(
                EvaluationStatus.SCORED,
                han=1,
                fu=30,
                limit=ScoreLimit.NONE,
                winner_points=1000,
                ron_payment=1000,
                red_dora=0,
            ),
        )
        self.assertEqual(wait.winning_tile, _tile("4s"))
        self.assertEqual((wait.dama_tsumo.han, wait.dama_tsumo.fu), (2, 20))
        self.assertEqual(
            (
                wait.dama_tsumo.tsumo_dealer_payment,
                wait.dama_tsumo.tsumo_non_dealer_payment,
            ),
            (700, 400),
        )
        self.assertIsNone(wait.dama_tsumo.ron_payment)
        self.assertEqual((wait.riichi_ron.han, wait.riichi_ron.ron_payment), (2, 2000))
        self.assertEqual(wait.riichi_tsumo.han, 3)

    def test_no_yaku_dama_ron_is_returned_as_status(self) -> None:
        features = analyze_riichi_decision(_riichi_decision(KANCHAN), _tile("1z"))
        wait = _wait(features, "2s")
        self.assertEqual(wait.dama_ron, WaitScore(EvaluationStatus.NO_YAKU))
        self.assertFalse(wait.dama_ron.is_scored)
        self.assertTrue(wait.dama_tsumo.is_scored)
        self.assertEqual((wait.riichi_ron.han, wait.riichi_ron.fu), (1, 40))
        self.assertEqual(wait.riichi_ron.ron_payment, 1300)

    def test_mangan_dama_and_haneman_riichi(self) -> None:
        decision = _riichi_decision(SANSHOKU, dora_indicators="1m")
        features = analyze_riichi_decision(decision, _tile("1z"))
        self.assertEqual(len(features.waits), 2)
        for wait in features.waits:
            self.assertEqual(wait.dama_ron.limit, ScoreLimit.MANGAN)
            self.assertEqual(wait.dama_ron.ron_payment, 8000)
            self.assertEqual(wait.riichi_ron.limit, ScoreLimit.HANEMAN)
        # ドラ表示牌の1mは見えている牌として数える（待ちの5s・8sは4枚とも見えない）。
        self.assertEqual(features.total_unseen_count, 8)

    def test_dealer_payment(self) -> None:
        decision = _riichi_decision(PINFU, dealer_seat=SELF)
        features = analyze_riichi_decision(decision, _tile("1z"))
        self.assertTrue(features.is_dealer)
        self.assertEqual(_wait(features, "4s").dama_ron.ron_payment, 1500)
        self.assertIsNone(_wait(features, "4s").dama_tsumo.tsumo_dealer_payment)

    def test_red_identity_of_declaration_tile_changes_score(self) -> None:
        decision = _riichi_decision(PINFU_WITH_RED)
        plain = analyze_riichi_decision(decision, _tile("0p"))
        red_kept = analyze_riichi_decision(decision, _tile("5p"))
        self.assertEqual(_wait(plain, "4s").dama_ron.ron_payment, 1000)
        self.assertEqual(_wait(red_kept, "4s").dama_ron.han, 2)
        self.assertEqual(_wait(red_kept, "4s").dama_ron.ron_payment, 2000)
        self.assertEqual(red_kept.declaration_tile, _tile("5p"))
        self.assertEqual(plain.wait_tile_types, red_kept.wait_tile_types)

    def test_seven_pairs_and_thirteen_orphans_waits(self) -> None:
        chiitoi = analyze_riichi_decision(_riichi_decision(CHIITOI), _tile("5z"))
        self.assertEqual(chiitoi.wait_tile_types, (_tile("6z").tile_type,))
        self.assertEqual(chiitoi.waits[0].unseen_count, 3)
        self.assertEqual(
            (chiitoi.waits[0].dama_ron.han, chiitoi.waits[0].dama_ron.fu), (2, 25)
        )
        kokushi = analyze_riichi_decision(_riichi_decision(KOKUSHI), _tile("5m"))
        self.assertEqual(len(kokushi.waits), 13)
        # 自分の河の9m（既定の`river`）も見えている牌として数える。
        self.assertEqual([w.unseen_count for w in kokushi.waits], [3, 2, *([3] * 11)])
        for wait in kokushi.waits:
            self.assertEqual(wait.dama_ron.limit, ScoreLimit.YAKUMAN)


@_needs_scoring
class FuritenAndCountTest(unittest.TestCase):
    def test_declaration_tile_causes_furiten_and_is_counted_once(self) -> None:
        features = analyze_riichi_decision(_riichi_decision(PINFU_WITH_4S), _tile("4s"))
        self.assertTrue(features.furiten)
        # 宣言牌の4sは手牌から河へ移るだけで、1枚として数える（4 - 1 = 3）。
        self.assertEqual(_wait(features, "4s").unseen_count, 3)
        self.assertEqual(_wait(features, "1s").unseen_count, 4)

    def test_previous_own_discard_causes_furiten(self) -> None:
        features = analyze_riichi_decision(
            _riichi_decision(PINFU, river="1s"), _tile("1z")
        )
        self.assertTrue(features.furiten)
        self.assertEqual(_wait(features, "1s").unseen_count, 3)
        # 振聴でもロンの点数は計算する。
        self.assertTrue(_wait(features, "1s").dama_ron.is_scored)

    def test_public_discards_and_melds_reduce_unseen_count(self) -> None:
        decision = _riichi_decision(PINFU)
        players = list(decision.input.players)
        players[2] = _player(discards="4s4s")
        context = replace(decision.input, players=tuple(players))
        features = analyze_riichi_decision(
            replace(decision, input=context), _tile("1z")
        )
        self.assertFalse(features.furiten)
        self.assertEqual(_wait(features, "4s").unseen_count, 2)


@_needs_scoring
class SituationTest(unittest.TestCase):
    def test_double_riichi_status(self) -> None:
        decision = _riichi_decision(PINFU, river=None)
        features = analyze_riichi_decision(decision, _tile("1z"))
        self.assertTrue(features.double_riichi)
        self.assertEqual(features.own_discard_count, 0)
        self.assertEqual(_wait(features, "4s").riichi_ron.han, 3)
        self.assertEqual(_wait(features, "4s").dama_ron.han, 1)

    def test_any_meld_prevents_double_riichi(self) -> None:
        decision = _riichi_decision(PINFU, river=None)
        players = list(decision.input.players)
        players[2] = _player(melds=(_pon_meld("7z"),))
        context = replace(decision.input, players=tuple(players))
        features = analyze_riichi_decision(
            replace(decision, input=context), _tile("1z")
        )
        self.assertFalse(features.double_riichi)
        self.assertEqual(_wait(features, "4s").riichi_ron.han, 2)

    def test_turn_opponent_riichi_rank_and_all_last(self) -> None:
        decision = _riichi_decision(
            PINFU,
            river="9m8m",
            riichi_discards="9p",
            scores=(25000, 30000, 25000, 20000),
        )
        features = analyze_riichi_decision(decision, _tile("1z"))
        self.assertEqual(features.own_discard_count, 2)
        self.assertEqual(features.opponent_riichi_count, 1)
        self.assertEqual((features.score, features.rank), (25000, 2))
        self.assertEqual(features.tied_opponent_count, 1)
        self.assertFalse(features.is_dealer)
        self.assertFalse(features.is_all_last)
        self.assertEqual(features.seat, SELF)

        players = list(decision.input.players)
        players[3] = replace(players[3], riichi=RiichiState.DECLARED)
        declared = replace(
            decision, input=replace(decision.input, players=tuple(players))
        )
        self.assertEqual(
            analyze_riichi_decision(declared, _tile("1z")).opponent_riichi_count, 2
        )

        all_last = _riichi_decision(PINFU, **ALL_LAST_TOP)
        features = analyze_riichi_decision(all_last, _tile("1z"))
        self.assertTrue(features.is_all_last)
        self.assertEqual((features.rank, features.tied_opponent_count), (1, 0))

    def test_dama_choice_is_fresh_champion_without_riichi(self) -> None:
        decision = _riichi_decision(PINFU)
        features = analyze_riichi_decision(decision, _tile("1z"))
        dama = DecisionContext(
            input=decision.input,
            legal_actions=tuple(
                action for action in decision.legal_actions if action != RIICHI
            ),
        )
        self.assertEqual(features.dama_choice, CHAMPION.choose_action(dama))
        self.assertEqual(features.dama_choice, _discard("1z"))
        # 解析の前後でChampionの本番行動は変わらない。
        self.assertEqual(CHAMPION.choose_action(decision), RIICHI)


@_needs_scoring
class InvalidInputTest(unittest.TestCase):
    def test_declaration_tile_not_in_hand(self) -> None:
        with self.assertRaises(RiichiDamaAnalysisError):
            analyze_riichi_decision(_riichi_decision(PINFU), _tile("9p"))
        # 赤のidentityも区別する（手牌の5pは赤だけ）。
        decision = _riichi_decision("123m4067p89s2355s1z")
        with self.assertRaises(RiichiDamaAnalysisError):
            analyze_riichi_decision(decision, _tile("5p"))

    def test_riichi_not_legal(self) -> None:
        decision = _riichi_decision(PINFU)
        no_riichi = replace(
            decision,
            legal_actions=tuple(a for a in decision.legal_actions if a != RIICHI),
        )
        with self.assertRaises(RiichiDamaAnalysisError):
            analyze_riichi_decision(no_riichi, _tile("1z"))

    def test_not_tenpai_after_declaration(self) -> None:
        with self.assertRaises(RiichiDamaAnalysisError):
            analyze_riichi_decision(_riichi_decision(PINFU), _tile("2s"))

    def test_own_riichi_already_declared(self) -> None:
        decision = _riichi_decision(PINFU)
        context = _own_river(decision.input, "9m", riichi=RiichiState.ACCEPTED)
        with self.assertRaises(RiichiDamaAnalysisError):
            analyze_riichi_decision(replace(decision, input=context), _tile("1z"))

    def test_wrong_types(self) -> None:
        with self.assertRaises(TypeError):
            analyze_riichi_decision(_riichi_decision(PINFU).input, _tile("1z"))
        with self.assertRaises(TypeError):
            analyze_riichi_decision(_riichi_decision(PINFU), "1z")


@_needs_scoring
@_needs_scoring
class RedFiveWinningTileTest(unittest.TestCase):
    """#354: 通常5の上限を超える待ちだけ赤5で評価し、和了牌と赤の加点を返す。"""

    # Arena #542 rehearsal-2の分岐 472006-9-1: 打7sで 123m 11p 333p 555p 67p（待ち5p・8p）
    THREE_PLAIN_FIVES = "123m11p333p555p67p7s"
    # 123m 789s 111p 4556p + 1z: 打1zで5p待ち（手に通常5pが2枚）
    TWO_PLAIN_FIVES = "123m789s111p4556p1z"

    def test_three_plain_fives_in_hand_evaluate_the_red_five(self) -> None:
        features = analyze_riichi_decision(
            _riichi_decision(self.THREE_PLAIN_FIVES), _tile("7s")
        )
        five, eight = _wait(features, "5p"), _wait(features, "8p")
        self.assertEqual(five.winning_tile, _tile("0p"))
        self.assertEqual(eight.winning_tile, _tile("8p"))
        for score in (five.dama_tsumo, five.riichi_ron, five.riichi_tsumo):
            self.assertEqual(score.red_dora, 1)
        for score in (eight.dama_tsumo, eight.riichi_ron, eight.riichi_tsumo):
            self.assertEqual(score.red_dora, 0)
        # 赤で評価しても残り枚数は公開情報から数える（通常5p 3枚が手にある -> 残り1枚）
        self.assertEqual(five.unseen_count, 1)
        self.assertGreater(five.riichi_ron.han, eight.riichi_ron.han)

    def test_plain_five_is_used_when_it_exists(self) -> None:
        features = analyze_riichi_decision(
            _riichi_decision(self.TWO_PLAIN_FIVES), _tile("1z")
        )
        five = _wait(features, "5p")
        self.assertEqual(five.winning_tile, _tile("5p"))
        self.assertEqual(five.riichi_ron.red_dora, 0)

    def test_dora_indicator_counts_toward_the_plain_limit(self) -> None:
        decision = _riichi_decision(self.TWO_PLAIN_FIVES, dora_indicators="5p")
        five = _wait(analyze_riichi_decision(decision, _tile("1z")), "5p")
        self.assertEqual(five.winning_tile, _tile("0p"))
        self.assertEqual(five.riichi_ron.red_dora, 1)

    def test_no_physical_winning_tile_is_rejected(self) -> None:
        # 通常5p 3枚が手にあり、赤5pはドラ表示牌: 5p待ちはどの牌でも成立しない
        decision = _riichi_decision(self.THREE_PLAIN_FIVES, dora_indicators="0p")
        with self.assertRaises(RiichiDamaAnalysisError):
            analyze_riichi_decision(decision, _tile("7s"))

    def test_winning_tile_and_red_dora_round_trip(self) -> None:
        features = analyze_riichi_decision(
            _riichi_decision(self.THREE_PLAIN_FIVES), _tile("7s")
        )
        value = riichi_decision_features_to_value(features)
        self.assertEqual(value["version"], "riichi-dama-features-v2")
        (five,) = [wait for wait in value["waits"] if wait["tile_type"]["rank"] == 5]
        self.assertEqual(
            five["winning_tile"], {"category": "pinzu", "is_red": True, "rank": 5}
        )
        self.assertEqual(five["riichi_ron"]["red_dora"], 1)
        text = json.dumps(value, sort_keys=True)
        self.assertEqual(parse_riichi_decision_features(json.loads(text)), features)
        broken = json.loads(text)
        broken["waits"][0]["winning_tile"]["rank"] = 9
        with self.assertRaises(RiichiDamaAnalysisError):
            parse_riichi_decision_features(broken)
        broken = json.loads(text)
        broken["waits"][0]["riichi_ron"]["red_dora"] = -1
        with self.assertRaises(RiichiDamaAnalysisError):
            parse_riichi_decision_features(broken)


class JsonRoundTripTest(unittest.TestCase):
    def _features(self) -> list[RiichiDecisionFeatures]:
        return [
            analyze_riichi_decision(_riichi_decision(PINFU), _tile("1z")),
            analyze_riichi_decision(_riichi_decision(KANCHAN), _tile("1z")),
            analyze_riichi_decision(_riichi_decision(PINFU_WITH_RED), _tile("5p")),
            analyze_riichi_decision(_riichi_decision(KOKUSHI), _tile("5m")),
            analyze_riichi_decision(_riichi_decision(PINFU, river=None), _tile("1z")),
        ]

    def test_round_trip_through_json_text(self) -> None:
        for features in self._features():
            value = riichi_decision_features_to_value(features)
            self.assertEqual(value["version"], RIICHI_DAMA_FEATURES_VERSION)
            text = json.dumps(value, sort_keys=True)
            parsed = parse_riichi_decision_features(json.loads(text))
            self.assertEqual(parsed, features)
            self.assertEqual(
                json.dumps(riichi_decision_features_to_value(parsed), sort_keys=True),
                text,
            )

    def test_tsumo_dama_choice_round_trips(self) -> None:
        features = self._features()[0]
        tsumo = replace(
            features, dama_choice=TsumoAction(actor=SELF, winning_tile=_tile("4s"))
        )
        value = riichi_decision_features_to_value(tsumo)
        self.assertEqual(parse_riichi_decision_features(value), tsumo)

    def test_rejects_unknown_version_and_malformed_values(self) -> None:
        value = riichi_decision_features_to_value(self._features()[0])
        cases = [
            {**value, "version": "riichi-dama-features-v0"},
            {**value, "version": "riichi-dama-features-v1"},
            {key: item for key, item in value.items() if key != "version"},
            {**value, "extra": 1},
            {**value, "furiten": 0},
            {**value, "rank": True},
            {**value, "rank": 5},
            {**value, "waits": []},
            {**value, "waits": list(reversed(value["waits"]))},
            {**value, "dama_choice": {"kind": "riichi", "actor": 0}},
            {**value, "dama_choice": {**value["dama_choice"], "actor": 1}},
            {
                **value,
                "waits": [
                    {
                        **value["waits"][0],
                        "dama_ron": {
                            **value["waits"][0]["dama_ron"],
                            "limit": "double",
                        },
                    },
                    *value["waits"][1:],
                ],
            },
            {
                **value,
                "waits": [
                    {
                        **value["waits"][0],
                        "dama_ron": {
                            **value["waits"][0]["dama_ron"],
                            "status": "no_yaku",
                        },
                    },
                    *value["waits"][1:],
                ],
            },
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(RiichiDamaAnalysisError):
                parse_riichi_decision_features(case)
        with self.assertRaises(RiichiDamaAnalysisError):
            parse_riichi_decision_features(json.dumps(value))


class _Scripted:
    """判断ごとに決めた行動を返すinner（呼び出しを記録する）。"""

    def __init__(self, actions) -> None:
        self._actions = list(actions)
        self.calls: list[DecisionContext] = []

    def choose_action(self, decision: DecisionContext):
        self.calls.append(decision)
        action = self._actions.pop(0)
        return action(decision) if callable(action) else action


TARGET = RoundIdentity(Wind.EAST, 1, 0)
OTHER_ROUND = {"round_wind": Wind.EAST, "hand_number": 2}


class BranchPolicyTest(unittest.TestCase):
    def _wrapper(self, inner=CHAMPION, discard=None) -> RiichiDamaBranchPolicy:
        return RiichiDamaBranchPolicy(inner, TARGET, discard or _discard("1z"))

    def test_round_identity_of_round_state(self) -> None:
        decision = _riichi_decision(PINFU)
        self.assertEqual(RoundIdentity.of(decision.input.round), TARGET)
        with self.assertRaises(ValueError):
            RoundIdentity(Wind.EAST, 5, 0)
        with self.assertRaises(TypeError):
            RoundIdentity(Wind.EAST, 1, True)

    def test_other_rounds_pass_through_including_riichi(self) -> None:
        wrapper = self._wrapper()
        decision = _riichi_decision(PINFU, **OTHER_ROUND)
        self.assertEqual(wrapper.choose_action(decision), RIICHI)
        self.assertEqual(wrapper.status, BranchStatus.PENDING)
        self.assertEqual(wrapper.replacement_count, 0)

    def test_branch_point_returns_specified_discard(self) -> None:
        wrapper = self._wrapper()
        decision = _riichi_decision(PINFU)
        self.assertEqual(CHAMPION.choose_action(decision), RIICHI)
        self.assertEqual(wrapper.choose_action(decision), _discard("1z"))
        self.assertEqual(wrapper.status, BranchStatus.BRANCHED)
        self.assertEqual(wrapper.replacement_count, 0)

    def test_branch_discard_must_match_exactly(self) -> None:
        decision = _riichi_decision(PINFU)
        for discard in (
            _discard("1z", tsumogiri=True),
            _discard("9p"),
            DiscardAction(actor=Seat.SEAT_1, tile=_tile("1z"), tsumogiri=False),
        ):
            wrapper = self._wrapper(discard=discard)
            with (
                self.subTest(discard=discard),
                self.assertRaises(RiichiDamaBranchError),
            ):
                wrapper.choose_action(decision)
            self.assertEqual(wrapper.status, BranchStatus.PENDING)
        red_free = _input("123m456p789s2355s5p")
        red_free = DecisionContext(
            input=red_free, legal_actions=(RIICHI, *_discards(red_free))
        )
        with self.assertRaises(RiichiDamaBranchError):
            self._wrapper(discard=_discard("0p")).choose_action(red_free)

    def test_later_riichi_in_target_round_is_replaced(self) -> None:
        wrapper = self._wrapper()
        decision = _riichi_decision(PINFU)
        wrapper.choose_action(decision)
        later = _riichi_decision(PINFU, river="9m1z")
        dama = DecisionContext(
            input=later.input,
            legal_actions=tuple(a for a in later.legal_actions if a != RIICHI),
        )
        expected = CHAMPION.choose_action(dama)
        self.assertEqual(wrapper.choose_action(later), expected)
        self.assertEqual(wrapper.replacement_count, 1)
        self.assertEqual(wrapper.choose_action(later), expected)
        self.assertEqual(wrapper.replacement_count, 2)
        self.assertEqual(wrapper.status, BranchStatus.BRANCHED)

    def test_replacement_reruns_inner_without_riichi(self) -> None:
        inner = _Scripted([RIICHI, RIICHI, lambda decision: decision.legal_actions[0]])
        wrapper = self._wrapper(inner=inner)
        decision = _riichi_decision(PINFU)
        wrapper.choose_action(decision)
        replaced = wrapper.choose_action(decision)
        self.assertNotIn(RIICHI, inner.calls[-1].legal_actions)
        self.assertEqual(inner.calls[-1].input, decision.input)
        self.assertEqual(replaced, inner.calls[-1].legal_actions[0])
        self.assertEqual(wrapper.replacement_count, 1)

    def test_inner_non_riichi_actions_pass_through(self) -> None:
        tsumo = TsumoAction(actor=SELF, winning_tile=_tile("4s"))
        decision = _riichi_decision(PINFU)
        decision = replace(decision, legal_actions=(*decision.legal_actions, tsumo))
        inner = _Scripted([tsumo, _discard("1z")])
        wrapper = self._wrapper(inner=inner)
        self.assertEqual(wrapper.choose_action(decision), tsumo)
        self.assertEqual(wrapper.choose_action(decision), _discard("1z"))
        self.assertEqual(wrapper.status, BranchStatus.PENDING)

    def test_round_left_without_branch_is_missed(self) -> None:
        inner = _Scripted([_discard("1z"), RIICHI])
        wrapper = self._wrapper(inner=inner)
        wrapper.choose_action(_riichi_decision(PINFU))
        self.assertEqual(wrapper.status, BranchStatus.PENDING)
        wrapper.choose_action(_riichi_decision(PINFU, **OTHER_ROUND))
        self.assertEqual(wrapper.status, BranchStatus.MISSED)

    def test_target_round_appearing_again_is_rejected(self) -> None:
        for first in (_discard("1z"), RIICHI):
            inner = _Scripted([first, RIICHI])
            wrapper = self._wrapper(inner=inner)
            wrapper.choose_action(_riichi_decision(PINFU))
            wrapper.choose_action(_riichi_decision(PINFU, **OTHER_ROUND))
            with self.subTest(first=first), self.assertRaises(RiichiDamaBranchError):
                wrapper.choose_action(_riichi_decision(PINFU))

    def test_champion_behaviour_unchanged_outside_branch(self) -> None:
        decisions = [
            _riichi_decision(PINFU, **OTHER_ROUND),
            _riichi_decision(SANSHOKU, **OTHER_ROUND),
            DecisionContext(
                input=_input(KANCHAN), legal_actions=_discards(_input(KANCHAN))
            ),
        ]
        wrapper = self._wrapper()
        for decision in decisions:
            self.assertEqual(
                wrapper.choose_action(decision),
                PlacementAwareSpeedCallPolicy().choose_action(decision),
            )

    def test_constructor_types(self) -> None:
        with self.assertRaises(TypeError):
            RiichiDamaBranchPolicy(object(), TARGET, _discard("1z"))
        with self.assertRaises(TypeError):
            RiichiDamaBranchPolicy(CHAMPION, (Wind.EAST, 1, 0), _discard("1z"))
        with self.assertRaises(TypeError):
            RiichiDamaBranchPolicy(CHAMPION, TARGET, RIICHI)


class FirstBranchPolicyTest(unittest.TestCase):
    """R1（#357）: 分岐点の1回だけ差し替え、以後はリーチを含めinnerへ戻す。"""

    def _wrapper(self, inner=CHAMPION, discard=None) -> RiichiFirstDamaBranchPolicy:
        return RiichiFirstDamaBranchPolicy(inner, TARGET, discard or _discard("1z"))

    def test_identity_is_distinct_from_old_wrapper(self) -> None:
        wrapper = self._wrapper()
        self.assertEqual(wrapper.version, RIICHI_FIRST_DAMA_BRANCH_VERSION)
        self.assertNotIsInstance(wrapper, RiichiDamaBranchPolicy)
        self.assertFalse(hasattr(RiichiDamaBranchPolicy, "version"))

    def test_other_rounds_pass_through_including_riichi(self) -> None:
        wrapper = self._wrapper()
        decision = _riichi_decision(PINFU, **OTHER_ROUND)
        self.assertEqual(wrapper.choose_action(decision), RIICHI)
        self.assertEqual(wrapper.status, BranchStatus.PENDING)
        self.assertEqual(wrapper.riichi_after_branch_count, 0)

    def test_branch_point_returns_specified_discard(self) -> None:
        wrapper = self._wrapper()
        decision = _riichi_decision(PINFU)
        self.assertEqual(CHAMPION.choose_action(decision), RIICHI)
        self.assertEqual(wrapper.choose_action(decision), _discard("1z"))
        self.assertEqual(wrapper.status, BranchStatus.BRANCHED)
        self.assertEqual(wrapper.riichi_after_branch_count, 0)

    def test_branch_discard_must_match_exactly(self) -> None:
        decision = _riichi_decision(PINFU)
        for discard in (
            _discard("1z", tsumogiri=True),
            _discard("9p"),
            DiscardAction(actor=Seat.SEAT_1, tile=_tile("1z"), tsumogiri=False),
        ):
            wrapper = self._wrapper(discard=discard)
            with (
                self.subTest(discard=discard),
                self.assertRaises(RiichiDamaBranchError),
            ):
                wrapper.choose_action(decision)
            self.assertEqual(wrapper.status, BranchStatus.PENDING)

    def test_later_riichi_in_target_round_is_returned_unchanged(self) -> None:
        wrapper = self._wrapper()
        wrapper.choose_action(_riichi_decision(PINFU))
        later = _riichi_decision(PINFU, river="9m1z")
        self.assertEqual(CHAMPION.choose_action(later), RIICHI)
        self.assertEqual(wrapper.choose_action(later), RIICHI)
        self.assertEqual(wrapper.riichi_after_branch_count, 1)
        self.assertEqual(wrapper.status, BranchStatus.BRANCHED)

    def test_only_one_replacement_and_inner_called_once_per_decision(self) -> None:
        inner = _Scripted([RIICHI, RIICHI, _discard("1z"), RIICHI])
        wrapper = self._wrapper(inner=inner)
        decision = _riichi_decision(PINFU)
        returned = [wrapper.choose_action(decision) for _ in range(4)]
        self.assertEqual(returned, [_discard("1z"), RIICHI, _discard("1z"), RIICHI])
        self.assertEqual(len(inner.calls), 4)
        # innerへ渡す合法手は元のまま（リーチを除いて作り直さない）
        for call in inner.calls:
            self.assertIs(call, decision)
        self.assertEqual(wrapper.riichi_after_branch_count, 2)

    def test_non_riichi_actions_pass_through_before_and_after_branch(self) -> None:
        tsumo = TsumoAction(actor=SELF, winning_tile=_tile("4s"))
        decision = _riichi_decision(PINFU)
        decision = replace(decision, legal_actions=(*decision.legal_actions, tsumo))
        inner = _Scripted([tsumo, RIICHI, tsumo])
        wrapper = self._wrapper(inner=inner)
        self.assertEqual(wrapper.choose_action(decision), tsumo)
        self.assertEqual(wrapper.status, BranchStatus.PENDING)
        self.assertEqual(wrapper.choose_action(decision), _discard("1z"))
        self.assertEqual(wrapper.choose_action(decision), tsumo)
        self.assertEqual(wrapper.riichi_after_branch_count, 0)

    def test_round_left_without_branch_is_missed(self) -> None:
        inner = _Scripted([_discard("1z"), RIICHI])
        wrapper = self._wrapper(inner=inner)
        wrapper.choose_action(_riichi_decision(PINFU))
        wrapper.choose_action(_riichi_decision(PINFU, **OTHER_ROUND))
        self.assertEqual(wrapper.status, BranchStatus.MISSED)

    def test_target_round_appearing_again_is_rejected(self) -> None:
        for first in (_discard("1z"), RIICHI):
            inner = _Scripted([first, RIICHI])
            wrapper = self._wrapper(inner=inner)
            wrapper.choose_action(_riichi_decision(PINFU))
            wrapper.choose_action(_riichi_decision(PINFU, **OTHER_ROUND))
            with self.subTest(first=first), self.assertRaises(RiichiDamaBranchError):
                wrapper.choose_action(_riichi_decision(PINFU))

    def test_riichi_in_other_round_after_branch_is_not_counted(self) -> None:
        inner = _Scripted([RIICHI, RIICHI])
        wrapper = self._wrapper(inner=inner)
        wrapper.choose_action(_riichi_decision(PINFU))
        self.assertEqual(
            wrapper.choose_action(_riichi_decision(PINFU, **OTHER_ROUND)), RIICHI
        )
        self.assertEqual(wrapper.riichi_after_branch_count, 0)
        self.assertEqual(wrapper.status, BranchStatus.BRANCHED)

    def test_champion_behaviour_unchanged_outside_branch(self) -> None:
        decisions = [
            _riichi_decision(PINFU, **OTHER_ROUND),
            _riichi_decision(SANSHOKU, **OTHER_ROUND),
            DecisionContext(
                input=_input(KANCHAN), legal_actions=_discards(_input(KANCHAN))
            ),
        ]
        wrapper = self._wrapper()
        for decision in decisions:
            self.assertEqual(
                wrapper.choose_action(decision),
                PlacementAwareSpeedCallPolicy().choose_action(decision),
            )

    def test_input_and_constructor_types(self) -> None:
        with self.assertRaises(TypeError):
            RiichiFirstDamaBranchPolicy(object(), TARGET, _discard("1z"))
        with self.assertRaises(TypeError):
            RiichiFirstDamaBranchPolicy(CHAMPION, (Wind.EAST, 1, 0), _discard("1z"))
        with self.assertRaises(TypeError):
            RiichiFirstDamaBranchPolicy(CHAMPION, TARGET, RIICHI)
        with self.assertRaises(TypeError):
            self._wrapper().choose_action(object())


class BoundaryTest(unittest.TestCase):
    def test_module_passes_information_boundary_check(self) -> None:
        report = check_candidate_imports("lisjong.policies.riichi_dama_branch")
        self.assertTrue(report.passed, msg=report.violations)


if __name__ == "__main__":
    unittest.main()
