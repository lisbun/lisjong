"""Issue #348: 鳴き時のroute打点proxyと、直後の本番打牌後のproxyの整合診断。

鳴き前は本番`PlacementAwareSpeedCallPolicy`にChi/Ponを選ばせ、成立後は
公開情報だけで整合する`DecisionContext`（手牌から喰った牌をexact identityで
除去、自席副露を追加、鳴かれた河の牌に`called_by`、喰い替え禁止牌を除いた
合法打牌、赤牌identity）を同じ本番Policyへ渡す。選択処理のコピーやmockは
使わない。

比較量は鳴き判断と同じ定義のroute shanten / 打点proxy
（`_speed_call_candidates`、`_best_route_evaluation`）であり、実際の翻・得点・
EVではない。分類:

- CONSISTENT: 打牌後のroute shanten / proxyが鳴き判断時と同じ
- BELOW_MAX: route shantenは同じで、proxyが鳴き判断時の最大値より低いが
  mode別の採用閾値以上
- BELOW_THRESHOLD: route shantenは同じで、proxyが採用閾値を下回る

fixtureは手作りの小さな局面に限り、到達頻度・棋力への影響は扱わない。
結果の記録は`docs/speed-call-post-call-value-diagnostic.md`。
"""

import unittest
from collections.abc import Mapping
from dataclasses import dataclass, replace

from test_placement_aware_speed_call_policy import (
    ALL_LAST_FAR_LAST,
    ALL_LAST_TOP,
    _chi_meld,
    _hand,
    _input,
    _pon_meld,
    _tile,
)

import lisjong.policies.finite_horizon_completion as finite_horizon
import lisjong.policies.placement_aware_speed_call as speed_call
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.policies import (
    PlacementAwareSpeedCallPolicy,
    TargetedHonorReleaseTerminalProgressionPolicy,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    YakuhaiCallPolicyError,
    _kuikae_forbidden_tile_types,
    _post_call_stable_hands,
    _remove_exact_consumed_tiles,
)
from lisjong.policy_contract.action import (
    ChiAction,
    DiscardAction,
    PassAction,
    PonAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile
from lisjong.structural_efficiency import (
    known_tile_counts,
    post_discard_concealed_hand,
    ukeire_count,
)

_SELF = Seat.SEAT_0
_PASS = PassAction(actor=_SELF)

CONSISTENT = "CONSISTENT"
BELOW_MAX = "BELOW_MAX"
BELOW_THRESHOLD = "BELOW_THRESHOLD"


def _chi(called: str, consumed: str) -> ChiAction:
    first, second = _hand(consumed)
    return ChiAction(
        actor=_SELF,
        target=Seat.SEAT_3,
        called_tile=_tile(called),
        consumed_tiles=(first, second),
    )


def _pon(called: str) -> PonAction:
    tile = _tile(called)
    return PonAction(
        actor=_SELF, target=Seat.SEAT_2, called_tile=tile, consumed_tiles=(tile, tile)
    )


def _river(*, called_by_self: str = "", plain: str = "") -> tuple[Discard, ...]:
    """自席が以前鳴いた牌（`called_by`付き）→通常の捨て牌の順の河。"""
    tiles = (
        *((tile, _SELF) for tile in _hand(called_by_self)),
        *((tile, None) for tile in _hand(plain)),
    )
    return tuple(
        Discard(tile=tile, tsumogiri=False, order=order, called_by=called_by)
        for order, (tile, called_by) in enumerate(tiles)
    )


def _pre_call_decision(
    concealed: str,
    calls: tuple[ChiAction | PonAction, ...],
    *,
    rivers: Mapping[Seat, tuple[Discard, ...]] | None = None,
    legal_pass: bool = True,
    **input_kwargs: object,
) -> DecisionContext:
    """鳴かれる牌がtargetの河の最後の牌である、鳴き判断のcontext。"""
    policy_input = _input(concealed, **input_kwargs)  # type: ignore[arg-type]
    rivers = dict(rivers or {})
    target = calls[0].target
    called_tile = calls[0].called_tile
    if any(call.target != target or call.called_tile != called_tile for call in calls):
        raise ValueError("fixture calls must share target and called tile")
    target_river = rivers.get(target, ())
    rivers[target] = (
        *target_river,
        Discard(
            tile=called_tile,
            tsumogiri=False,
            order=len(target_river),
            called_by=None,
        ),
    )
    players = list(policy_input.players)
    for seat, river in rivers.items():
        players[int(seat)] = replace(players[int(seat)], discards=river)
    policy_input = replace(policy_input, players=tuple(players))
    derive_remaining_tile_inventory(policy_input)
    legal_actions = ((_PASS,) if legal_pass else ()) + calls
    return DecisionContext(input=policy_input, legal_actions=legal_actions)


def _post_call_decision(
    pre_input: PolicyInput, call: ChiAction | PonAction
) -> DecisionContext:
    """鳴き成立直後の強制打牌context。"""
    players = list(pre_input.players)
    target = players[int(call.target)]
    last = target.discards[-1]
    if last.tile != call.called_tile or last.called_by is not None:
        raise ValueError("called tile must be the target's last unclaimed discard")
    players[int(call.target)] = replace(
        target, discards=(*target.discards[:-1], replace(last, called_by=_SELF))
    )
    meld = PublicMeld(
        kind=MeldKind.CHI if isinstance(call, ChiAction) else MeldKind.PON,
        tiles=(*call.consumed_tiles, call.called_tile),
        from_seat=call.target,
        called_tile=call.called_tile,
    )
    own = players[int(_SELF)]
    players[int(_SELF)] = replace(own, melds=(*own.melds, meld))
    concealed = tuple(
        _remove_exact_consumed_tiles(
            pre_input.own_hand.concealed_tiles, call.consumed_tiles
        )
    )
    post_input = replace(
        pre_input,
        players=tuple(players),
        own_hand=replace(
            pre_input.own_hand, concealed_tiles=concealed, drawn_tile=None
        ),
    )
    forbidden = _kuikae_forbidden_tile_types(call)
    unique_tiles: list[Tile] = []
    for tile in concealed:
        if tile not in unique_tiles and tile.tile_type not in forbidden:
            unique_tiles.append(tile)
    legal_actions = tuple(
        DiscardAction(actor=_SELF, tile=tile, tsumogiri=False) for tile in unique_tiles
    )
    return DecisionContext(input=post_input, legal_actions=legal_actions)


@dataclass(frozen=True, slots=True)
class _DiscardFacts:
    route_shanten: int
    value: int
    completion_mass: int
    ukeire: int


@dataclass(frozen=True, slots=True)
class _Trace:
    call: ChiAction | PonAction
    mode: speed_call.GameSituationMode
    call_route_shanten: int
    call_value: int
    minimum_value: int
    post: DecisionContext
    discard: DiscardAction
    facts: dict[Tile, _DiscardFacts]

    @property
    def chosen(self) -> _DiscardFacts:
        return self.facts[self.discard.tile]

    @property
    def classification(self) -> str:
        chosen = self.chosen
        if chosen.route_shanten != self.call_route_shanten:
            return "ROUTE_SHANTEN_CHANGED"
        if chosen.value == self.call_value:
            return CONSISTENT
        if chosen.value < self.minimum_value:
            return BELOW_THRESHOLD
        return BELOW_MAX


def _minimum_value(
    pre_input: PolicyInput, mode: speed_call.GameSituationMode, route_shanten: int
) -> int:
    acceptance = speed_call._CALL_ACCEPTANCE[mode]
    if not speed_call._is_closed_hand(pre_input):
        return acceptance.open_minimum_value
    if route_shanten == 0:
        return acceptance.closed_tenpai_minimum_value
    return acceptance.closed_non_tenpai_minimum_value


def _post_discard_facts(post: DecisionContext) -> dict[Tile, _DiscardFacts]:
    """各合法打牌の同一定義route shanten / proxyと、速度指標。"""
    post_input = post.input
    meld_tiles = speed_call._own_meld_tiles(post_input)
    routes = speed_call._compatible_routes(meld_tiles)
    discards = tuple(
        action for action in post.legal_actions if isinstance(action, DiscardAction)
    )
    masses = {
        evaluation.action.tile: evaluation.completion_mass
        for evaluation in finite_horizon._evaluate_completion_masses(
            post_input,
            discards,
            finite_horizon._root_remaining_counts(post_input),
            finite_horizon.DEFAULT_HORIZON,
            finite_horizon._FiniteHorizonEvaluator(),
        )
    }
    known = known_tile_counts(post_input)
    facts: dict[Tile, _DiscardFacts] = {}
    for action in discards:
        hand = post_discard_concealed_hand(
            post_input.own_hand.concealed_tiles, action.tile
        )
        route_shanten, value = speed_call._best_route_evaluation(
            hand, meld_tiles, routes, post_input.round.dora_indicators
        )
        facts[action.tile] = _DiscardFacts(
            route_shanten=route_shanten,
            value=value,
            completion_mass=masses[action.tile],
            ukeire=ukeire_count(hand, known),
        )
    return facts


class _DiagnosticCase(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PlacementAwareSpeedCallPolicy()
        self.parent = TargetedHonorReleaseTerminalProgressionPolicy()

    def _trace(self, pre: DecisionContext) -> _Trace:
        """本番Policyの鳴き→成立後の本番Policyの打牌を通す。"""
        self.assertIs(self.parent.choose_action(pre), _PASS)
        call = self.policy.choose_action(pre)
        self.assertIsInstance(call, (ChiAction, PonAction))
        assert isinstance(call, (ChiAction, PonAction))

        mode = speed_call.situation_mode(pre.input)
        calls = tuple(
            action
            for action in pre.legal_actions
            if isinstance(action, (ChiAction, PonAction))
        )
        candidates = {
            candidate.action: candidate
            for candidate in speed_call._speed_call_candidates(pre.input, calls, mode)
        }
        selected = candidates[call]

        post = _post_call_decision(pre.input, call)
        derive_remaining_tile_inventory(post.input)
        discard = self.policy.choose_action(post)
        self.assertIn(discard, post.legal_actions)
        assert isinstance(discard, DiscardAction)

        facts = _post_discard_facts(post)
        # 鳴き判断の(最小route shanten, その最大proxy)は合法打牌の中に実在する。
        best = min((fact.route_shanten, -fact.value) for fact in facts.values())
        self.assertEqual(best, (selected.route_shanten, -selected.value))
        return _Trace(
            call=call,
            mode=mode,
            call_route_shanten=selected.route_shanten,
            call_value=selected.value,
            minimum_value=_minimum_value(pre.input, mode, selected.route_shanten),
            post=post,
            discard=discard,
            facts=facts,
        )

    def _assert_trace(
        self,
        trace: _Trace,
        *,
        call: ChiAction | PonAction,
        mode: speed_call.GameSituationMode,
        call_evaluation: tuple[int, int],
        minimum_value: int,
        discard: str,
        post_evaluation: tuple[int, int],
        classification: str,
    ) -> None:
        self.assertEqual(trace.call, call)
        self.assertIs(trace.mode, mode)
        self.assertEqual((trace.call_route_shanten, trace.call_value), call_evaluation)
        self.assertEqual(trace.minimum_value, minimum_value)
        self.assertEqual(trace.discard.tile, _tile(discard))
        self.assertEqual(
            (trace.chosen.route_shanten, trace.chosen.value), post_evaluation
        )
        self.assertEqual(trace.classification, classification)

    def _assert_value_preserving_alternative(
        self, trace: _Trace, alternative: str, *, ukeire: tuple[int, int]
    ) -> None:
        """失ったproxyを保つ合法代替と、本番打牌が優先した速度指標を固定する。"""
        kept = trace.facts[_tile(alternative)]
        self.assertEqual(
            (kept.route_shanten, kept.value),
            (trace.call_route_shanten, trace.call_value),
        )
        self.assertGreater(trace.chosen.completion_mass, kept.completion_mass)
        self.assertEqual((trace.chosen.ukeire, kept.ukeire), ukeire)


NORMAL = speed_call.GameSituationMode.NORMAL
SPEED = speed_call.GameSituationMode.ALL_LAST_TOP_SPEED
VALUE = speed_call.GameSituationMode.ALL_LAST_LAST_VALUE

# 34m 678m 222p 56p 468s: 門前2向聴。2mチー後の 678m 222p 56p 468s は
# Tanyao 1向聴（8s切りでも4s切りでも1向聴）。8sがドラ（表示7s）のとき
# 鳴き判断の最大proxyは Tanyao 1 + 8s 1 = 2 で、門前非聴牌の閾値2ちょうど。
CLOSED_TWO_SHANTEN = "34m678m56p22p468s2p"
CHI_2M = _chi("2m", "34m")


class PostCallContextTest(unittest.TestCase):
    """成立後contextの公開情報・合法手・喰い替え・赤牌identity。"""

    def test_post_call_context_is_consistent_public_state(self) -> None:
        pre = _pre_call_decision(CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s")
        post = _post_call_decision(pre.input, CHI_2M)
        own = post.input.players[int(_SELF)]
        self.assertEqual(len(post.input.own_hand.concealed_tiles), 11)
        self.assertIsNone(post.input.own_hand.drawn_tile)
        self.assertEqual(own.melds[-1].kind, MeldKind.CHI)
        self.assertEqual(own.melds[-1].called_tile, _tile("2m"))
        river = post.input.players[int(Seat.SEAT_3)].discards
        self.assertEqual(river[-1].tile, _tile("2m"))
        self.assertIs(river[-1].called_by, _SELF)
        # 鳴き前後で公開tile conservationが成立し、未見枚数は変わらない。
        self.assertEqual(
            derive_remaining_tile_inventory(pre.input).remaining_tile_counts,
            derive_remaining_tile_inventory(post.input).remaining_tile_counts,
        )

    def test_kuikae_types_are_not_legal_discards(self) -> None:
        # 2mを34mで下側チー: 2m（現物）と5m（筋）は切れない。
        pre = _pre_call_decision("345m678m222p468s9s", (CHI_2M,))
        post = _post_call_decision(pre.input, CHI_2M)
        legal_types = {action.tile.tile_type for action in post.legal_actions}  # type: ignore[union-attr]
        self.assertNotIn(_tile("5m").tile_type, legal_types)
        self.assertNotIn(_tile("2m").tile_type, legal_types)
        self.assertIn(_tile("8m").tile_type, legal_types)

    def test_red_identity_is_consumed_exactly(self) -> None:
        chi_red = _chi("4p", "06p")
        chi_normal = _chi("4p", "56p")
        pre = _pre_call_decision("50p6p678m22s468s9m1z", (chi_red, chi_normal))
        red_post = _post_call_decision(pre.input, chi_red)
        normal_post = _post_call_decision(pre.input, chi_normal)
        self.assertIn(_tile("5p"), red_post.input.own_hand.concealed_tiles)
        self.assertNotIn(_tile("0p"), red_post.input.own_hand.concealed_tiles)
        self.assertIn(_tile("0p"), normal_post.input.own_hand.concealed_tiles)
        self.assertNotIn(_tile("5p"), normal_post.input.own_hand.concealed_tiles)
        derive_remaining_tile_inventory(red_post.input)
        derive_remaining_tile_inventory(normal_post.input)


class ReproducedFixtureTest(_DiagnosticCase):
    """本番打牌が鳴き判断時の最大proxyを保持しないfixture（REPRODUCED-FIXTURE）。"""

    def test_closed_one_shanten_call_falls_below_acceptance_threshold(self) -> None:
        # 打8s: 678m 222p 56p 46s（proxy 1）。打4s: 678m 222p 56p 68s（proxy 2）。
        # 68s嵌張の7sはドラ表示牌で1枚見えるため、本番打牌は46sを残す。
        trace = self._trace(
            _pre_call_decision(CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s")
        )
        self._assert_trace(
            trace,
            call=CHI_2M,
            mode=NORMAL,
            call_evaluation=(1, 2),
            minimum_value=2,
            discard="8s",
            post_evaluation=(1, 1),
            classification=BELOW_THRESHOLD,
        )
        self._assert_value_preserving_alternative(trace, "4s", ukeire=(24, 23))

    def test_closed_tenpai_call_keeps_threshold_but_loses_dora(self) -> None:
        # 2mチー後 678m 567p 22p 468s。打8sは5s待ち（4枚）、打4sは7s待ち
        # （表示牌で1枚見えて3枚）でドラ8sを残す。
        trace = self._trace(
            _pre_call_decision("34m678m567p22p468s", (CHI_2M,), dora_indicators="7s")
        )
        self._assert_trace(
            trace,
            call=CHI_2M,
            mode=NORMAL,
            call_evaluation=(0, 2),
            minimum_value=0,
            discard="8s",
            post_evaluation=(0, 1),
            classification=BELOW_MAX,
        )
        self._assert_value_preserving_alternative(trace, "4s", ukeire=(4, 3))

    def test_open_hand_additional_pon_loses_dora(self) -> None:
        # 234mチー済み。5pポン後 678m 22p 468s で、同じ46s / 68sの選択。
        pre = _pre_call_decision(
            "678m22p55p468s",
            (_pon("5p"),),
            own_melds=(_chi_meld("234m"),),
            rivers={Seat.SEAT_3: _river(called_by_self="2m")},
            dora_indicators="7s",
        )
        trace = self._trace(pre)
        self._assert_trace(
            trace,
            call=_pon("5p"),
            mode=NORMAL,
            call_evaluation=(0, 2),
            minimum_value=0,
            discard="8s",
            post_evaluation=(0, 1),
            classification=BELOW_MAX,
        )
        self._assert_value_preserving_alternative(trace, "4s", ukeire=(4, 3))

    def test_chinitsu_tenpai_becomes_honitsu_tanki(self) -> None:
        # 6mポン後 111m 345m 789m 9m 1z。打1zは清一色聴牌（6m-9m待ち、proxy 5）、
        # 打9mは1z単騎の混一色聴牌（proxy 2）。受入はどちらも3枚で、
        # completion massの小差だけで1z単騎が選ばれる。
        trace = self._trace(_pre_call_decision("66m111m345m789m9m1z", (_pon("6m"),)))
        self._assert_trace(
            trace,
            call=_pon("6m"),
            mode=NORMAL,
            call_evaluation=(0, 5),
            minimum_value=0,
            discard="9m",
            post_evaluation=(0, 2),
            classification=BELOW_MAX,
        )
        self._assert_value_preserving_alternative(trace, "1z", ukeire=(3, 3))

    def test_honitsu_route_switches_to_tanyao_when_honor_wait_is_scarce(
        self,
    ) -> None:
        # 3mポン後 456m 678m 567m 4p 1z。打4pは1z単騎の混一色（proxy 2）、
        # 打1zは4p単騎のタンヤオ（proxy 1）。1zが2枚見えて1z単騎は残り1枚。
        pre = _pre_call_decision(
            "33m456m678m567m4p1z",
            (_pon("3m"),),
            rivers={Seat.SEAT_1: _river(plain="11z")},
        )
        trace = self._trace(pre)
        self._assert_trace(
            trace,
            call=_pon("3m"),
            mode=NORMAL,
            call_evaluation=(0, 2),
            minimum_value=0,
            discard="1z",
            post_evaluation=(0, 1),
            classification=BELOW_MAX,
        )
        self._assert_value_preserving_alternative(trace, "4p", ukeire=(3, 1))

    def test_speed_mode_boundary(self) -> None:
        # SPEEDは閾値0のため、同じ打8sは閾値割れではなくBELOW_MAX。
        trace = self._trace(
            _pre_call_decision(
                CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s", **ALL_LAST_TOP
            )
        )
        self._assert_trace(
            trace,
            call=CHI_2M,
            mode=SPEED,
            call_evaluation=(1, 2),
            minimum_value=0,
            discard="8s",
            post_evaluation=(1, 1),
            classification=BELOW_MAX,
        )

    def test_value_mode_boundary(self) -> None:
        # 5pもドラ（表示4p）でproxy 3 = VALUE閾値ちょうど。打8sで2へ下がる。
        trace = self._trace(
            _pre_call_decision(
                CLOSED_TWO_SHANTEN,
                (CHI_2M,),
                dora_indicators="7s4p",
                **ALL_LAST_FAR_LAST,
            )
        )
        self._assert_trace(
            trace,
            call=CHI_2M,
            mode=VALUE,
            call_evaluation=(1, 3),
            minimum_value=3,
            discard="8s",
            post_evaluation=(1, 2),
            classification=BELOW_THRESHOLD,
        )


class ConsistentFixtureTest(_DiagnosticCase):
    """本番打牌が鳴き判断時の最大proxyを保持するfixture。"""

    def _assert_consistent(
        self, pre: DecisionContext, *, discard: str, call_evaluation: tuple[int, int]
    ) -> None:
        trace = self._trace(pre)
        self.assertEqual(trace.discard.tile, _tile(discard))
        self.assertEqual((trace.call_route_shanten, trace.call_value), call_evaluation)
        self.assertEqual(trace.classification, CONSISTENT)

    def test_dora_on_faster_side_is_kept(self) -> None:
        # ドラが4s（表示3s）なら速度でも46sを残すため一致する。
        self._assert_consistent(
            _pre_call_decision("34m678m567p22p468s", (CHI_2M,), dora_indicators="3s"),
            discard="8s",
            call_evaluation=(0, 2),
        )

    def test_exact_completion_tie_keeps_dora(self) -> None:
        # 2mチー後 67m 567p 22p 3468s。打6s / 打8sはcompletion massが完全同点で、
        # HandValueAwareの同点処理がドラ8sを残す。
        self._assert_consistent(
            _pre_call_decision("34m67m567p22p3468s", (CHI_2M,), dora_indicators="7s"),
            discard="6s",
            call_evaluation=(1, 2),
        )

    def test_open_hand_additional_pon_with_dora_on_faster_side(self) -> None:
        self._assert_consistent(
            _pre_call_decision(
                "678m22p55p468s",
                (_pon("5p"),),
                own_melds=(_chi_meld("234m"),),
                rivers={Seat.SEAT_3: _river(called_by_self="2m")},
                dora_indicators="3s",
            ),
            discard="8s",
            call_evaluation=(0, 2),
        )

    def test_honitsu_pon_keeps_dora(self) -> None:
        self._assert_consistent(
            _pre_call_decision("1235799m112233z", (_pon("9m"),), dora_indicators="6m"),
            discard="1z",
            call_evaluation=(1, 3),
        )

    def test_chinitsu_kept_when_honor_wait_is_scarce(self) -> None:
        self._assert_consistent(
            _pre_call_decision(
                "66m111m345m789m9m1z",
                (_pon("6m"),),
                rivers={Seat.SEAT_1: _river(plain="11z")},
            ),
            discard="1z",
            call_evaluation=(0, 5),
        )

    def test_route_split_tie_keeps_honitsu(self) -> None:
        self._assert_consistent(
            _pre_call_decision("33m456m678m567m4p1z", (_pon("3m"),)),
            discard="4p",
            call_evaluation=(0, 2),
        )

    def test_red_five_kept_over_normal_five(self) -> None:
        # 2mチー後 678m 50p6p 22p 468s。5p / 0p切りは同じ構造で、通常5pを切る。
        self._assert_consistent(
            _pre_call_decision("34m678m56p22p468s0p", (CHI_2M,)),
            discard="5p",
            call_evaluation=(1, 2),
        )


class NonActivationTest(_DiagnosticCase):
    """本番Policyがspeed callを選ばない局面は反例に数えない。"""

    def test_value_mode_rejects_call_below_threshold(self) -> None:
        pre = _pre_call_decision(
            CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s", **ALL_LAST_FAR_LAST
        )
        self.assertIs(self.policy.choose_action(pre), _PASS)

    def test_opponent_riichi_suppresses_call(self) -> None:
        pre = _pre_call_decision(
            CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s", riichi_discards="1z9s"
        )
        self.assertIs(self.policy.choose_action(pre), _PASS)

    def test_open_yakuhai_hand_uses_parent_decision(self) -> None:
        pre = _pre_call_decision(
            "678m56p22p468s",
            (_pon("2p"),),
            own_melds=(_pon_meld("5z"),),
            rivers={Seat.SEAT_2: _river(called_by_self="5z")},
            dora_indicators="7s",
        )
        self.assertEqual(self.policy.choose_action(pre), self.parent.choose_action(pre))

    def test_non_improving_call_is_rejected(self) -> None:
        pre = _pre_call_decision("34m678m567p22p678s", (CHI_2M,), dora_indicators="7s")
        self.assertIs(self.policy.choose_action(pre), _PASS)

    def test_without_legal_pass_parent_fails_closed(self) -> None:
        # parentの既存契約: 採用する鳴きもPassもなければfail closed。
        # speed callの判定までは進まないため、反例の候補にならない。
        pre = _pre_call_decision(
            CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s", legal_pass=False
        )
        with self.assertRaises(YakuhaiCallPolicyError):
            self.policy.choose_action(pre)

    def test_call_without_kuikae_legal_discard_is_excluded(self) -> None:
        # 234p / 345s / 666sの副露済み。2mを34mでチーすると残り55mが全て喰い替え禁止。
        pre = _pre_call_decision(
            "3455m",
            (CHI_2M,),
            own_melds=(_chi_meld("234p"), _chi_meld("345s"), _pon_meld("6s")),
            rivers={
                Seat.SEAT_2: _river(called_by_self="6s"),
                Seat.SEAT_3: _river(called_by_self="2p3s"),
            },
        )
        self.assertEqual(
            _post_call_stable_hands(pre.input.own_hand.concealed_tiles, CHI_2M), ()
        )
        self.assertEqual(
            speed_call._speed_call_candidates(pre.input, (CHI_2M,), NORMAL), ()
        )
        self.assertIs(self.policy.choose_action(pre), _PASS)


if __name__ == "__main__":
    unittest.main()
