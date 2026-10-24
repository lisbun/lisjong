"""鳴き採用と直後の本番打牌の限定解析API（Issue #350、call-value-shadow-v1）。

current Heuristic Champion `PlacementAwareSpeedCallPolicy`が

1. 鳴き判断で、どの経路（parent判断 / Pass / スピード鳴き）でActionを選んだか
2. スピード鳴き成立直後の強制打牌で、実打牌が鳴き判断時の打点proxyと採用閾値を
   保ったか、保たない場合にどの合法打牌が閾値を保つ代替になるか

を、Policyと同じproduction helperで返す。Arena #533の
call-value-shadow-v1が正本の意味定義であり、Arenaはここで返す値を記録・照合し、
proxy・向聴・受入・completion massを再計算しない。

入口は2つだけである。

- `analyze_speed_call_decision(decision, selected)`:
  鳴き判断のpublic `DecisionContext`とChampionが選んだActionから
  `SpeedCallDecisionAnalysis`を返す。
- `analyze_post_call_discard(decision, call_analysis, selected)`:
  call成立後の実際のmandatory discard `DecisionContext`、同じcall解析値、
  Championの実打牌から`PostCallDiscardAnalysis`を返す。

どちらもChampionの判断を同じ入力で再実行し、渡されたActionと一致しなければ
fail closedする。Policyの挙動・状態は変えず、返却値はimmutableである。
入力は`DecisionContext`だけで、hidden truth・未来event・cross-decision状態を
扱わない（call→打牌の対応付けはArenaの観測側が所有する）。

## 値の意味

- route shanten / proxy: 鳴き判断と同じ`_best_route_evaluation`。proxyは
  route翻相当とroute内ドラ・赤ドラ枚数の和で、実際の翻・得点・EVではない。
- completion mass: 本番打牌選択と同じFiniteHorizon（horizon=3）のexact整数。
  全合法打牌で同じroot remaining inventory（`N`枚）と分母
  `D = N(N-1)(N-2)`を共有する。構造的な指標であり、和了確率やEVではない。
- 受入: `ukeire_count`（構造的な有効牌の残り枚数）。

## 分類と代替

実打牌のroute shantenが鳴き判断時の最小route shantenと異なる場合、および
鳴き判断の(最小route shanten, 最大proxy)が合法打牌に再現しない場合は、
解析の前提が崩れているため`SpeedCallAnalysisError`とする。

- `CONSISTENT`: 実打牌のproxyが鳴き判断時の最大proxyと同じ
- `BELOW_MAX`: 最大proxyより低いが採用閾値以上
- `BELOW_THRESHOLD`: 採用閾値未満

代替は`BELOW_THRESHOLD`だけで選ぶ。route shantenが鳴き判断時と同じで
proxyが採用閾値以上の合法打牌から、completion mass最大 → proxy最大 →
受入最大 → `discard_action_sort_key`最小の順で1つ選ぶ。整数だけで比較し、
合法手の列挙順に依存しない。それ以外の分類では代替は`None`。

速度損失は実打牌mass `B`と代替mass `A`から`L = (B - A) / B`を整数の
分子・分母で返す。`B = A = 0`は`L = 0`（`zero_mass=True`）、`A > B`や
`B = 0 < A`は前提違反として`SpeedCallAnalysisError`とする。
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto

from lisjong.policies import finite_horizon_completion as _finite_horizon
from lisjong.policies.placement_aware_speed_call import (
    GameSituationMode,
    PlacementAwareSpeedCallPolicy,
    _best_route_evaluation,
    _compatible_routes,
    _is_closed_hand,
    _minimum_call_value,
    _own_meld_tiles,
    _select_speed_call,
    situation_mode,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    _kuikae_forbidden_tile_types,
)
from lisjong.policy_contract.action import (
    ChiAction,
    DiscardAction,
    InternalAction,
    PassAction,
    PonAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.meld import MeldKind
from lisjong.structural_efficiency import (
    discard_action_sort_key,
    known_tile_counts,
    post_discard_concealed_hand,
    ukeire_count,
)

CALL_VALUE_SHADOW_VERSION = "call-value-shadow-v1"
"""このAPIが実装する意味定義（Arena #533）のversion。"""

_PRIMARY_TARGET_ROUTE_SHANTEN = 1
_PRIMARY_TARGET_MINIMUM_VALUE = 2
_HAND_SLOTS = 14
_MELD_SLOTS = 3


class SpeedCallAnalysisError(Exception):
    """入力が解析の前提と整合しない場合にfail closedする。"""


class CallDecisionRoute(Enum):
    """鳴き判断でChampionのActionが決まった経路。"""

    PARENT_DECISION = auto()
    """parentがPass以外を選び、そのまま返した（役牌parent経路等）。"""
    PARENT_PASS = auto()
    """parentはPassで、スピード鳴きを採用しなかった。"""
    SPEED_CALL = auto()
    """parentはPassで、スピード鳴きとしてChi/Ponを選んだ。"""


class PostCallClassification(Enum):
    """実打牌のproxyと鳴き判断時のproxy・採用閾値の関係。"""

    CONSISTENT = auto()
    BELOW_MAX = auto()
    BELOW_THRESHOLD = auto()


@dataclass(frozen=True, slots=True)
class SpeedCallDecisionAnalysis:
    """1回の鳴き判断の解析値。

    `call` / `route_shanten` / `value` / `minimum_value`は
    `route is SPEED_CALL`のときだけ値を持ち、それ以外は`None`。
    `route_shanten`はcall後の最小route shanten、`value`はその最小を達成する
    最大proxy、`minimum_value`はmode・call前の門前状態・route shantenに対する
    採用閾値である。
    """

    route: CallDecisionRoute
    action: InternalAction
    mode: GameSituationMode
    closed_before_call: bool
    call: ChiAction | PonAction | None
    route_shanten: int | None
    value: int | None
    minimum_value: int | None

    def __post_init__(self) -> None:
        speed_fields = (self.call, self.route_shanten, self.value, self.minimum_value)
        if self.route is CallDecisionRoute.SPEED_CALL:
            if any(field is None for field in speed_fields):
                raise ValueError("speed call analysis requires call values")
            if self.call != self.action:
                raise ValueError("speed call must be the selected action")
        elif any(field is not None for field in speed_fields):
            raise ValueError("call values are defined only for speed calls")

    @property
    def is_primary_target(self) -> bool:
        """#533の主対象（NORMAL・門前→route 1向聴・採用閾値2のスピード鳴き）。"""
        return (
            self.route is CallDecisionRoute.SPEED_CALL
            and self.mode is GameSituationMode.NORMAL
            and self.closed_before_call
            and self.route_shanten == _PRIMARY_TARGET_ROUTE_SHANTEN
            and self.minimum_value == _PRIMARY_TARGET_MINIMUM_VALUE
        )


@dataclass(frozen=True, slots=True)
class PostCallDiscardFacts:
    """call成立後の1合法打牌の解析値。"""

    action: DiscardAction
    route_shanten: int
    value: int
    completion_mass: int
    ukeire: int


@dataclass(frozen=True, slots=True)
class SpeedLoss:
    """代替に対する実打牌の速度差。比率は整数の分子・分母が正本。

    `mass_difference = B - A`。`relative_*`は`L = (B - A) / B`で、
    `B = A = 0`のときは`0 / 1`かつ`zero_mass=True`。
    """

    mass_difference: int
    relative_numerator: int
    relative_denominator: int
    zero_mass: bool


@dataclass(frozen=True, slots=True)
class PostCallDiscardAnalysis:
    """call成立直後のmandatory discardの解析値。

    `candidates`は全合法打牌を`discard_action_sort_key`順に並べる。
    `horizon` / `hidden_tile_count`（N） / `sequence_denominator`（D）は
    全候補で共通である。
    """

    call_analysis: SpeedCallDecisionAnalysis
    discard: DiscardAction
    horizon: int
    hidden_tile_count: int
    sequence_denominator: int
    candidates: tuple[PostCallDiscardFacts, ...]
    classification: PostCallClassification
    alternative: DiscardAction | None
    speed_loss: SpeedLoss | None

    def facts_for(self, action: DiscardAction) -> PostCallDiscardFacts:
        for facts in self.candidates:
            if facts.action == action:
                return facts
        raise KeyError(action)

    @property
    def chosen(self) -> PostCallDiscardFacts:
        return self.facts_for(self.discard)

    @property
    def alternative_facts(self) -> PostCallDiscardFacts | None:
        if self.alternative is None:
            return None
        return self.facts_for(self.alternative)


def analyze_speed_call_decision(
    decision: DecisionContext, selected: InternalAction
) -> SpeedCallDecisionAnalysis:
    """鳴き判断でChampionが選んだActionの経路と、スピード鳴きの採用値を返す。

    `selected`はChampionが同じ`decision`で選んだActionでなければならない。
    """
    if selected not in decision.legal_actions:
        raise SpeedCallAnalysisError("selected action is not a legal action")
    policy = PlacementAwareSpeedCallPolicy()
    parent_decision = super(PlacementAwareSpeedCallPolicy, policy)._decide(decision)
    candidate = _select_speed_call(decision, parent_decision)
    production = parent_decision.action if candidate is None else candidate.action
    if production != selected:
        raise SpeedCallAnalysisError(
            "selected action differs from the Champion decision"
        )

    policy_input = decision.input
    mode = situation_mode(policy_input)
    closed = _is_closed_hand(policy_input)
    if candidate is None:
        route = (
            CallDecisionRoute.PARENT_PASS
            if isinstance(parent_decision.action, PassAction)
            else CallDecisionRoute.PARENT_DECISION
        )
        return SpeedCallDecisionAnalysis(
            route=route,
            action=selected,
            mode=mode,
            closed_before_call=closed,
            call=None,
            route_shanten=None,
            value=None,
            minimum_value=None,
        )
    return SpeedCallDecisionAnalysis(
        route=CallDecisionRoute.SPEED_CALL,
        action=selected,
        mode=mode,
        closed_before_call=closed,
        call=candidate.action,
        route_shanten=candidate.route_shanten,
        value=candidate.value,
        minimum_value=_minimum_call_value(
            mode, closed=closed, route_shanten=candidate.route_shanten
        ),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpeedCallAnalysisError(message)


def _validate_post_call_context(
    decision: DecisionContext,
    call_analysis: SpeedCallDecisionAnalysis,
    call: ChiAction | PonAction,
) -> tuple[DiscardAction, ...]:
    """成立後contextがこのcallの直後のmandatory discardであることを検査する。"""
    policy_input = decision.input
    actor = call.actor
    _require(policy_input.self_seat == actor, "self seat differs from the call actor")
    own_hand = policy_input.own_hand
    _require(own_hand.drawn_tile is None, "post-call discard must have no drawn tile")
    _require(
        situation_mode(policy_input) is call_analysis.mode,
        "situation mode changed between call and discard",
    )

    melds = policy_input.players[int(actor)].melds
    _require(bool(melds), "post-call context has no meld")
    meld = melds[-1]
    expected_kind = MeldKind.CHI if isinstance(call, ChiAction) else MeldKind.PON
    _require(
        meld.kind is expected_kind
        and meld.from_seat == call.target
        and meld.called_tile == call.called_tile
        and Counter(meld.tiles) == Counter((*call.consumed_tiles, call.called_tile)),
        "last own meld does not match the call",
    )
    _require(
        all(earlier.kind is MeldKind.ANKAN for earlier in melds[:-1])
        is call_analysis.closed_before_call,
        "earlier melds do not match the closed state at call time",
    )
    _require(
        len(own_hand.concealed_tiles) + _MELD_SLOTS * len(melds) == _HAND_SLOTS,
        "concealed tile count is not a mandatory discard state",
    )
    river = policy_input.players[int(call.target)].discards
    _require(
        bool(river)
        and river[-1].tile == call.called_tile
        and river[-1].called_by == actor,
        "called tile is not the target's last discard called by the actor",
    )

    discards: list[DiscardAction] = []
    for action in decision.legal_actions:
        _require(
            isinstance(action, DiscardAction),
            "post-call legal actions must all be discards",
        )
        assert isinstance(action, DiscardAction)
        _require(action.actor == actor, "discard actor differs from the call actor")
        _require(not action.tsumogiri, "post-call discard cannot be tsumogiri")
        discards.append(action)
    forbidden = _kuikae_forbidden_tile_types(call)
    expected_tiles = {
        tile for tile in own_hand.concealed_tiles if tile.tile_type not in forbidden
    }
    tiles = [action.tile for action in discards]
    _require(
        len(tiles) == len(set(tiles)) and set(tiles) == expected_tiles,
        "legal discards do not match the concealed tiles after kuikae",
    )
    return tuple(sorted(discards, key=discard_action_sort_key))


def _alternative(
    candidates: tuple[PostCallDiscardFacts, ...],
    route_shanten: int,
    minimum_value: int,
) -> PostCallDiscardFacts:
    eligible = tuple(
        facts
        for facts in candidates
        if facts.route_shanten == route_shanten and facts.value >= minimum_value
    )
    _require(bool(eligible), "no legal discard keeps the call route and threshold")
    return min(
        eligible,
        key=lambda facts: (
            -facts.completion_mass,
            -facts.value,
            -facts.ukeire,
            discard_action_sort_key(facts.action),
        ),
    )


def _speed_loss(chosen: int, alternative: int) -> SpeedLoss:
    _require(
        alternative <= chosen,
        "alternative completion mass exceeds the chosen discard",
    )
    if chosen == 0:
        return SpeedLoss(
            mass_difference=0,
            relative_numerator=0,
            relative_denominator=1,
            zero_mass=True,
        )
    difference = chosen - alternative
    return SpeedLoss(
        mass_difference=difference,
        relative_numerator=difference,
        relative_denominator=chosen,
        zero_mass=False,
    )


def analyze_post_call_discard(
    decision: DecisionContext,
    call_analysis: SpeedCallDecisionAnalysis,
    selected: DiscardAction,
) -> PostCallDiscardAnalysis:
    """スピード鳴き成立直後の実打牌と全合法打牌の解析値を返す。

    `decision`は成立後の実際のpublic context、`call_analysis`は同じcallの
    `analyze_speed_call_decision()`の結果、`selected`はChampionの実打牌。
    """
    _require(
        call_analysis.route is CallDecisionRoute.SPEED_CALL,
        "post-call analysis requires a speed call",
    )
    call = call_analysis.call
    route_shanten = call_analysis.route_shanten
    call_value = call_analysis.value
    minimum_value = call_analysis.minimum_value
    assert call is not None
    assert route_shanten is not None
    assert call_value is not None
    assert minimum_value is not None

    discards = _validate_post_call_context(decision, call_analysis, call)
    _require(selected in discards, "selected discard is not a legal action")
    production = PlacementAwareSpeedCallPolicy().choose_action(decision)
    _require(
        production == selected, "selected discard differs from the Champion decision"
    )

    policy_input = decision.input
    horizon = _finite_horizon.DEFAULT_HORIZON
    remaining_counts = _finite_horizon._root_remaining_counts(policy_input)
    hidden_tile_count = sum(remaining_counts)
    _require(hidden_tile_count >= horizon, "remaining tiles are fewer than horizon")
    sequence_denominator = _finite_horizon._falling_factorial(
        hidden_tile_count, horizon
    )
    masses = {
        evaluation.action: evaluation.completion_mass
        for evaluation in _finite_horizon._evaluate_completion_masses(
            policy_input,
            discards,
            remaining_counts,
            horizon,
            _finite_horizon._FiniteHorizonEvaluator(),
        )
    }
    _require(
        all(mass <= sequence_denominator for mass in masses.values()),
        "completion mass exceeds the sequence denominator",
    )

    meld_tiles = _own_meld_tiles(policy_input)
    routes = _compatible_routes(meld_tiles)
    _require(bool(routes), "own melds have no compatible yaku route")
    known = known_tile_counts(policy_input)
    concealed = policy_input.own_hand.concealed_tiles
    candidates: list[PostCallDiscardFacts] = []
    for action in discards:
        hand = post_discard_concealed_hand(concealed, action.tile)
        shanten, value = _best_route_evaluation(
            hand, meld_tiles, routes, policy_input.round.dora_indicators
        )
        candidates.append(
            PostCallDiscardFacts(
                action=action,
                route_shanten=shanten,
                value=value,
                completion_mass=masses[action],
                ukeire=ukeire_count(hand, known),
            )
        )
    facts = tuple(candidates)

    _require(
        min((item.route_shanten, -item.value) for item in facts)
        == (route_shanten, -call_value),
        "call route shanten and value are not reproduced by the legal discards",
    )
    chosen = next(item for item in facts if item.action == selected)
    _require(
        chosen.route_shanten == route_shanten,
        "chosen discard changed the call route shanten",
    )
    if chosen.value == call_value:
        classification = PostCallClassification.CONSISTENT
    elif chosen.value < minimum_value:
        classification = PostCallClassification.BELOW_THRESHOLD
    else:
        classification = PostCallClassification.BELOW_MAX

    alternative: DiscardAction | None = None
    speed_loss: SpeedLoss | None = None
    if classification is PostCallClassification.BELOW_THRESHOLD:
        alternative_facts = _alternative(facts, route_shanten, minimum_value)
        alternative = alternative_facts.action
        speed_loss = _speed_loss(
            chosen.completion_mass, alternative_facts.completion_mass
        )

    return PostCallDiscardAnalysis(
        call_analysis=call_analysis,
        discard=selected,
        horizon=horizon,
        hidden_tile_count=hidden_tile_count,
        sequence_denominator=sequence_denominator,
        candidates=facts,
        classification=classification,
        alternative=alternative,
        speed_loss=speed_loss,
    )


__all__ = [
    "CALL_VALUE_SHADOW_VERSION",
    "CallDecisionRoute",
    "PostCallClassification",
    "PostCallDiscardAnalysis",
    "PostCallDiscardFacts",
    "SpeedCallAnalysisError",
    "SpeedCallDecisionAnalysis",
    "SpeedLoss",
    "analyze_post_call_discard",
    "analyze_speed_call_decision",
]
