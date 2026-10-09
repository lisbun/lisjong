"""単独リーチ下の聴牌PUSH/FOLD（H2）のゲート判定・押す/降りる候補・対比較用Policy（lisbun/lisjong#288）。

設計は`docs/tenpai-push-fold-design.md`（#254）の2・3節、wire契約は
`docs/tenpai-push-fold-source.md`にある。このmoduleは推論側で、入力は`PolicyInput`と
合法手だけである。リーチ者の手牌・正解ラベル・学習用sourceは読まない。

- `evaluate_tenpai_gate()`: ゲート条件1〜7と(A)(B)(C)の区分、押す側の打牌`a_push`
  （(A)では宣言牌の予測）、降りる候補`a_fold`を返す。対象外は`None`。条件8（表のbucket）は
  表がまだないので含めない。後の候補Policyも同じ関数を使う
- `own_wait_value()`: 打牌後の自分の待ちについて、役がある残り枚数と和了点の加重平均
  （比較式の`G`の和了点部分）を計算する
- `TenpaiPushFoldPairPolicy`: 指定した1つの判断だけで`a_fold`を切り、それ以外はChampion
  （`PlacementAwareSpeedCallPolicy`）の`PolicyDecision`をそのまま返す、対比較source用のPolicy。
  候補Policyではない

`p(a)`は`estimate_riichi_ron_legal_belief()`のロン合法確率で、校正済みの放銃確率ではない。
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from lisjong.belief.canonical_axes import tile_type_index, wind_for_seat
from lisjong.hand_evaluation.scoring import (
    URA_DORA_EXCLUDED,
    EvaluationStatus,
    RiichiStatus,
    ScoringMeld,
    UraDoraIndicators,
    WinContext,
    WinEvaluation,
    WinMethod,
    WinningHand,
    WinSituation,
    evaluate_win,
    require_scoring_backend,
)
from lisjong.learning.errors import LearningError
from lisjong.learning.riichi_wait_estimator import LogisticWaitModel
from lisjong.learning.riichi_wait_mawashi_policy import (
    SELECTED_WAIT_MODEL_SHA256,
    load_selected_wait_model,
)
from lisjong.learning.ron_legal_estimator import (
    estimate_riichi_ron_legal_belief,
    in_riichi_scope,
)
from lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware import (
    _decide_push_fold,
    _PushFoldDecision,
)
from lisjong.policies.placement_aware_speed_call import (
    PlacementAwareSpeedCallPolicy,
    _is_all_last,
    _is_closed_hand,
    _route_preserving_actions,
)
from lisjong.policy_contract.action import DiscardAction, RiichiAction
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory
from lisjong.structural_efficiency import (
    StructuralShantenEvaluator,
    discard_action_sort_key,
    effective_tile_types,
    evaluate_post_discard_hands,
    known_tile_counts,
    post_discard_concealed_hand,
    ukeire_count,
)

GATE = "tenpai-push-fold-gate.conditions-1-7.v1"
"""ゲート判定と候補選択の版。条件・`a_push`・`a_fold`の決め方を変えたら替える。"""

_MAX_COPIES = 4
_CHAMPION = PlacementAwareSpeedCallPolicy()

EvaluateWin = Callable[[WinningHand, WinContext], WinEvaluation]


class TenpaiPushFoldError(LearningError):
    """対比較用Policyへ指定した判断が、降りる候補のあるゲート判断ではない場合。"""


class GateKind(Enum):
    """ゲート判断の区分（設計1節の(A)(B)(C)）。"""

    RIICHI = "riichi"
    """(A) 門前で`RiichiAction`が合法。Championはリーチする。"""
    CLOSED_DISCARD = "closed_discard"
    """(B) 門前だが`RiichiAction`が合法でない打牌判断。"""
    OPEN_DISCARD = "open_discard"
    """(C) 暗槓以外の副露がある打牌判断。"""


@dataclass(frozen=True, slots=True)
class OwnWaitValue:
    """ある打牌の後の、自分の待ちの枚数と和了点。

    枚数は自分から見た残り枚数のうち、その和了方法で役がある牌種だけを数える。和了点は
    `evaluate_win()`の受取点（本場・供託を含まない）の残り枚数による加重平均で、枚数が0なら
    0.0である。`discard_furiten`なら`ron_count`は0、`ron_points`は0.0になる。
    `has_yaku`は、残りがある待ち牌種のどれかがロンかツモで役を持つこと（フリテンを見ない）。
    """

    ron_count: int
    tsumo_count: int
    ron_points: float
    tsumo_points: float
    discard_furiten: bool
    has_yaku: bool


@dataclass(frozen=True, slots=True)
class TenpaiGate:
    """ゲート条件1〜7を満たす判断1回の、区分と押す/降りる候補。

    `fold_action`は全合法打牌のうちロン合法確率が最小の打牌で、`has_fold_candidate`が
    偽（`push_action`より安全な牌がない）でも値を持つ。`fold_wait`は、(B)(C)で`fold_action`が
    聴牌を保つ場合だけ計算する（(A)の降りる側は聴牌側の式を使わない）。
    """

    kind: GateKind
    riichi_seat: Seat
    c0_action: RiichiAction | DiscardAction
    push_action: DiscardAction
    fold_action: DiscardAction
    push_ron_legal_raw: int
    fold_ron_legal_raw: int
    fold_keeps_tenpai: bool
    push_wait: OwnWaitValue
    fold_wait: OwnWaitValue | None

    @property
    def has_fold_candidate(self) -> bool:
        """`p(a_fold) < p(a_push)`か。偽ならC0を維持する判断である。"""
        return self.fold_ron_legal_raw < self.push_ron_legal_raw


@dataclass(frozen=True, slots=True)
class TenpaiPushFoldPairAnalysis(AnalysisTrace):
    """対比較用Policyが`a_fold`へ替えた判断の記録。替えなかった判断では作らない。"""

    gate: TenpaiGate


def _red_five_is_visible(policy_input: PolicyInput, category: TileCategory) -> bool:
    visible = (
        *policy_input.own_hand.concealed_tiles,
        *policy_input.round.dora_indicators,
        *(
            tile
            for player in policy_input.players
            for meld in player.melds
            for tile in meld.tiles
        ),
        *(
            discard.tile
            for player in policy_input.players
            for discard in player.discards
        ),
    )
    return any(tile.is_red and tile.tile_type.category is category for tile in visible)


def own_wait_value(
    policy_input: PolicyInput,
    action: DiscardAction,
    *,
    riichi: bool,
    evaluate: EvaluateWin = evaluate_win,
) -> OwnWaitValue:
    """`action`を切った後の待ちの、役がある残り枚数と和了点の加重平均を返す。

    `riichi`が真なら、リーチ役を含め裏ドラを除外モードで計算する（(A)の押す側。上乗せ`U`は
    含まない）。偽なら裏ドラなしを明示した完全な点数である。一発・海底などの状況役は
    入れない。未見の赤5は、その牌種の残り1枚を赤として数える。
    """
    hand = post_discard_concealed_hand(
        policy_input.own_hand.concealed_tiles, action.tile
    )
    seat = policy_input.self_seat
    own = policy_input.players[seat]
    known = known_tile_counts(policy_input)
    waits = effective_tile_types(hand, 0)
    river = {discard.tile.tile_type for discard in own.discards}
    river.add(action.tile.tile_type)
    furiten = any(tile_type in river for tile_type in waits)
    melds = tuple(ScoringMeld(meld.kind, meld.tiles) for meld in own.melds)

    counts = {WinMethod.RON: 0, WinMethod.TSUMO: 0}
    points = {WinMethod.RON: 0, WinMethod.TSUMO: 0}
    for tile_type in waits:
        remaining = _MAX_COPIES - known.get(tile_type, 0)
        if remaining <= 0:
            continue
        copies = [(Tile(tile_type), remaining)]
        if (
            tile_type.category is not TileCategory.HONOR
            and tile_type.rank == 5
            and not _red_five_is_visible(policy_input, tile_type.category)
        ):
            copies = [
                (Tile(tile_type, is_red=True), 1),
                (Tile(tile_type), remaining - 1),
            ]
        for method in counts:
            context = WinContext(
                method=method,
                seat_wind=wind_for_seat(seat, policy_input.round.dealer_seat),
                prevailing_wind=policy_input.round.round_wind,
                riichi=RiichiStatus.RIICHI if riichi else RiichiStatus.NONE,
                is_ippatsu=False,
                situation=WinSituation.NORMAL,
                dora_indicators=policy_input.round.dora_indicators,
                ura_dora=URA_DORA_EXCLUDED if riichi else UraDoraIndicators(()),
            )
            for tile, count in copies:
                if count == 0:
                    continue
                evaluation = evaluate(WinningHand(hand, melds, tile), context)
                if evaluation.status is EvaluationStatus.NOT_COMPLETE:
                    raise ValueError(
                        f"structural wait {tile_type} does not complete the hand"
                    )
                if evaluation.status is EvaluationStatus.SCORED:
                    counts[method] += count
                    points[method] += count * evaluation.score.winner_points

    has_yaku = any(counts.values())
    if furiten:
        counts[WinMethod.RON] = points[WinMethod.RON] = 0
    return OwnWaitValue(
        ron_count=counts[WinMethod.RON],
        tsumo_count=counts[WinMethod.TSUMO],
        ron_points=_mean(points[WinMethod.RON], counts[WinMethod.RON]),
        tsumo_points=_mean(points[WinMethod.TSUMO], counts[WinMethod.TSUMO]),
        discard_furiten=furiten,
        has_yaku=has_yaku,
    )


def _mean(total: int, count: int) -> float:
    return total / count if count else 0.0


def _champion_discard(
    policy_input: PolicyInput, discard_actions: tuple[DiscardAction, ...]
) -> DiscardAction:
    action = PlacementAwareSpeedCallPolicy._decide_discard(
        _CHAMPION, policy_input, discard_actions
    ).action
    if not isinstance(action, DiscardAction):
        raise ValueError("the Champion discard path did not return a discard")
    return action


def evaluate_tenpai_gate(
    decision: DecisionContext,
    c0_decision: PolicyDecision,
    model: LogisticWaitModel,
    *,
    evaluate: EvaluateWin = evaluate_win,
) -> TenpaiGate | None:
    """ゲート条件1〜7を満たす判断なら区分と候補を返し、対象外なら`None`を返す。

    `c0_decision`は同じ`decision`に対するChampionの決定である。和了・鳴き・槓・Passを
    Championが選ぶ判断は対象外になる。
    """
    if not isinstance(model, LogisticWaitModel):
        raise TypeError("model must be a LogisticWaitModel")
    policy_input = decision.input
    seat = policy_input.self_seat
    if policy_input.players[seat].riichi is not RiichiState.NONE:
        return None
    riichi_seats = [
        other
        for other in Seat
        if other is not seat and in_riichi_scope(policy_input, other)
    ]
    if len(riichi_seats) != 1 or _is_all_last(policy_input):
        return None
    riichi_seat = riichi_seats[0]
    discards = tuple(
        action for action in decision.legal_actions if isinstance(action, DiscardAction)
    )
    if len({action.tile.tile_type for action in discards}) < 2:
        return None

    evaluator = StructuralShantenEvaluator()
    structural = {
        evaluation.action: evaluation
        for evaluation in evaluate_post_discard_hands(policy_input, discards, evaluator)
    }
    c0 = c0_decision.action
    if isinstance(c0, RiichiAction):
        kind = GateKind.RIICHI
        tenpai_keeping = tuple(
            action
            for action in discards
            if structural[action].post_discard_shanten == 0
        )
        if not tenpai_keeping:
            return None
        push = _champion_discard(policy_input, tenpai_keeping)
    elif isinstance(c0, DiscardAction) and c0 in structural:
        if _is_closed_hand(policy_input):
            kind = GateKind.CLOSED_DISCARD
        else:
            kind = GateKind.OPEN_DISCARD
        role_preserving = _route_preserving_actions(policy_input, discards)
        if _decide_push_fold(policy_input, role_preserving) is _PushFoldDecision.FOLD:
            return None
        push = c0
    else:
        return None

    if structural[push].post_discard_shanten != 0:
        return None
    known = known_tile_counts(policy_input)
    if not any(
        known.get(tile_type, 0) < _MAX_COPIES
        for tile_type in effective_tile_types(
            structural[push].post_discard_hand, 0, evaluator
        )
    ):
        return None

    belief = estimate_riichi_ron_legal_belief(policy_input, riichi_seat, model)
    if belief is None:
        raise ValueError("the riichi seat is outside the ron-legal estimator's scope")
    ron_legal = belief.ron_legal_probability_raw

    def raw(action: DiscardAction) -> int:
        return ron_legal[tile_type_index(action.tile.tile_type)]

    fold = _fold_action(discards, structural, known, evaluator, raw)
    fold_keeps_tenpai = structural[fold].post_discard_shanten == 0
    riichi = kind is GateKind.RIICHI
    return TenpaiGate(
        kind=kind,
        riichi_seat=riichi_seat,
        c0_action=c0,
        push_action=push,
        fold_action=fold,
        push_ron_legal_raw=raw(push),
        fold_ron_legal_raw=raw(fold),
        fold_keeps_tenpai=fold_keeps_tenpai,
        push_wait=own_wait_value(policy_input, push, riichi=riichi, evaluate=evaluate),
        fold_wait=(
            own_wait_value(policy_input, fold, riichi=False, evaluate=evaluate)
            if fold_keeps_tenpai and not riichi
            else None
        ),
    )


def _fold_action(discards, structural, known, evaluator, raw) -> DiscardAction:
    """ロン合法確率が最小 → 打牌後の向聴が小さい → 受け入れが多い → canonicalな打牌順。"""
    safest = min(raw(action) for action in discards)
    candidates = [action for action in discards if raw(action) == safest]
    shanten = min(structural[action].post_discard_shanten for action in candidates)
    candidates = [
        action
        for action in candidates
        if structural[action].post_discard_shanten == shanten
    ]
    if len(candidates) == 1:
        return candidates[0]
    return min(
        candidates,
        key=lambda action: (
            -ukeire_count(
                structural[action].post_discard_hand, known, shanten, evaluator
            ),
            discard_action_sort_key(action),
        ),
    )


class TenpaiPushFoldPairPolicy(PlacementAwareSpeedCallPolicy):
    """#288: 指定した1つのゲート判断だけで`a_fold`を切る、対比較source用のPolicy。

    `fold_target`は、対照（Champion×4）で記録したゲート判断の`DecisionContext`である。
    対局は決定的なので、同じseed・同じ席配置なら、その判断までは対照と同じ入力が現れる。
    判断の特定は入力の一致だけで行い、instanceは判断をまたぐ状態を持たない。
    `fold_target`と一致する判断が、降りる候補のあるゲート判断でなければエラーにする。
    """

    def __init__(
        self,
        model: LogisticWaitModel,
        fold_target: DecisionContext,
        *,
        evaluate: EvaluateWin | None = None,
    ) -> None:
        if not isinstance(model, LogisticWaitModel):
            raise TypeError("model must be a LogisticWaitModel")
        if not isinstance(fold_target, DecisionContext):
            raise TypeError("fold_target must be a DecisionContext")
        if evaluate is None:
            require_scoring_backend()
            evaluate = evaluate_win
        super().__init__()
        self._model = model
        self._fold_target = fold_target
        self._evaluate = evaluate

    @classmethod
    def from_selection(
        cls,
        path: Path,
        fold_target: DecisionContext,
        *,
        expected_sha256: str = SELECTED_WAIT_MODEL_SHA256,
    ) -> "TenpaiPushFoldPairPolicy":
        return cls(
            load_selected_wait_model(path, expected_sha256=expected_sha256), fold_target
        )

    def _decide(self, decision: DecisionContext) -> PolicyDecision:
        c0_decision = super()._decide(decision)
        if decision != self._fold_target:
            return c0_decision
        gate = evaluate_tenpai_gate(
            decision, c0_decision, self._model, evaluate=self._evaluate
        )
        if gate is None or not gate.has_fold_candidate:
            raise TenpaiPushFoldError(
                "the fold target is not a gate decision with a fold candidate"
            )
        return PolicyDecision(
            action=gate.fold_action, analysis=TenpaiPushFoldPairAnalysis(gate=gate)
        )


__all__ = [
    "GATE",
    "GateKind",
    "OwnWaitValue",
    "TenpaiGate",
    "TenpaiPushFoldError",
    "TenpaiPushFoldPairAnalysis",
    "TenpaiPushFoldPairPolicy",
    "evaluate_tenpai_gate",
    "own_wait_value",
]
