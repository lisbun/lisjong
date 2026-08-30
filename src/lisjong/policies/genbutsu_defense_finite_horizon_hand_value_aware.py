"""defenseをPush/Fold・Safetyへ分解し、ValueAwareをHandValueAwareへ拡張したPolicy。

Issue #143のexperimental Policyとして、既存Policyを変更せず次のpriorityを
`TwoStepUkeirePolicy._decide_discard()` extension pointへ実装する。

    Push/Fold decision
    > Discard Safety
    > FiniteHorizon completion mass
    > HandValueAware ranking

current strength baselineである`GenbutsuDefenseFiniteHorizonValueAwarePolicy`
（Issue #122、`combined`）は変更しない。今回の新Policyへその実装を移植・refactor
することも避け、`current Combined = frozen comparison baseline` /
`new Policy = experimental candidate`という区別を維持する。

## Defense decomposition

既存Combinedでは他家リーチ有無・聴牌維持可否・共通現物判定が一続きの
`_genbutsu_eligible_actions()`として実装されている。本Policyはsemanticを変えず、
コード上でだけ次の2段へ明示的に分ける。

    _decide_push_fold()  -> _PushFoldDecision (PUSH / FOLD)
    _discard_safety()    -> _DiscardSafety (COMMON_GENBUTSU / UNKNOWN)

`_PushFoldDecision` / `_DiscardSafety`はこのPolicy専用のprivate typed valueで
あり、public Policy contractやgeneric defense frameworkへは昇格させない。
Push/Fold判定は既存Genbutsu activation semanticをそのまま使う。他家リーチが
なければPUSH、リーチがあってもlegal discardのいずれかが`post_discard_shanten
< 1`を維持できればPUSHとし、Genbutsu constraintを発動しない。全candidateが
非聴牌になるときだけFOLDとする。

FOLD時だけDiscard Safetyを評価する。全リーチ者のdiscard履歴に共通する牌種は
`COMMON_GENBUTSU`、それ以外は`UNKNOWN`（危険ではなく、単に安全根拠未確認）と
する。複数リーチ者では既存Combined / GenbutsuDefenseと同じくintersectionを取る。
`COMMON_GENBUTSU`が1件でもあればそれだけをeligible setにし、1件もなければ
既存Combinedと同じく全legal discardへfallbackする。

## FiniteHorizon / HandValueAware

defense filtering後のeligible setへ、既存`FiniteHorizonCompletionPolicy`の
DP実装をそのまま再利用する。unique positive maximumは即採用、positive
maximum tieではmaximum subsetだけ、all-zeroではeligible set全部をHandValueAware
stageへ渡す。completion massで負けた候補をHandValueAwareで復活させない。

HandValueAware stageは既存`HandValueAwareTwoStepUkeirePolicy`の
`_evaluate_and_choose_discard()`をそのまま呼び出す。retained real value、
completed yakuhai、tanyao / honitsu / chinitsu route value、second-step
rankingはこのPolicy専用に再実装しない。

winning action、Always Riichi、pass、既存fallbackのorchestrationは基底classから
継承する。本Issueではcombined-specific analysisを追加せず、打牌decisionの
`analysis`は`None`とする。
"""

from enum import Enum, auto

from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    FiniteHorizonCandidateEvaluation,
    FiniteHorizonCompletionPolicyError,
    _evaluate_completion_masses,
    _FiniteHorizonEvaluator,
    _root_remaining_counts,
)
from lisjong.policies.genbutsu_defense_two_step_ukeire import (
    _common_genbutsu_tile_types,
    _opponent_riichi_players,
)
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _evaluate_and_choose_discard as _hand_value_aware_evaluate_and_choose_discard,
)
from lisjong.policies.two_step_ukeire import (
    TwoStepUkeirePolicy,
    _DecisionShantenEvaluator,
    _evaluate_post_discard_hands,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import TileType


class _PushFoldDecision(Enum):
    """このPolicy専用のprivate push/fold typed value。"""

    PUSH = auto()
    FOLD = auto()


class _DiscardSafety(Enum):
    """このPolicy専用のprivate discard safety typed value。

    `UNKNOWN`は「危険」を意味せず、このPolicyが安全根拠を確認していないことを
    表す。
    """

    COMMON_GENBUTSU = auto()
    UNKNOWN = auto()


def _decide_push_fold(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> _PushFoldDecision:
    """他家リーチ有無と聴牌維持可否だけでPush/Foldを判定する。

    既存CombinedのGenbutsu activation semanticをそのまま使う。点棒状況、巡目、
    offensive expected value、deal-in probabilityは使用しない。
    """
    riichi_players = _opponent_riichi_players(policy_input)
    if not riichi_players:
        return _PushFoldDecision.PUSH

    evaluator = _DecisionShantenEvaluator()
    evaluated = _evaluate_post_discard_hands(policy_input, discard_actions, evaluator)
    if min(candidate.post_discard_shanten for candidate in evaluated) < 1:
        return _PushFoldDecision.PUSH
    return _PushFoldDecision.FOLD


def _discard_safety(
    action: DiscardAction,
    common_genbutsu_tile_types: frozenset[TileType],
) -> _DiscardSafety:
    """1候補のsafety categoryを返す。boolのsafe / unsafeにはしない。"""
    if action.tile.tile_type in common_genbutsu_tile_types:
        return _DiscardSafety.COMMON_GENBUTSU
    return _DiscardSafety.UNKNOWN


def _defense_eligible_actions(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, ...]:
    """Push/Fold・Safetyの2段decisionでFiniteHorizonのeligible setを返す。

    PUSHでは全legal discardをそのまま返す。FOLDではCOMMON_GENBUTSUの候補だけへ
    絞り、1件もなければ既存Combinedと同じく全legal discardへfallbackする。
    """
    if _decide_push_fold(policy_input, discard_actions) is _PushFoldDecision.PUSH:
        return discard_actions

    riichi_players = _opponent_riichi_players(policy_input)
    common_tile_types = _common_genbutsu_tile_types(riichi_players)
    genbutsu_actions = tuple(
        action
        for action in discard_actions
        if _discard_safety(action, common_tile_types) is _DiscardSafety.COMMON_GENBUTSU
    )
    return genbutsu_actions or discard_actions


def _hand_value_aware_fallback(
    policy_input: PolicyInput,
    evaluations: tuple[FiniteHorizonCandidateEvaluation, ...],
) -> DiscardAction:
    """FiniteHorizonから渡されたcandidate subsetを既存HandValueAwareで選ぶ。"""
    selected, _ = _hand_value_aware_evaluate_and_choose_discard(
        policy_input, tuple(evaluation.action for evaluation in evaluations)
    )
    return selected


def _evaluate_and_choose_discard(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> DiscardAction:
    """Issue #143の4-stage compositionを1回のselection pathで実行する。"""
    eligible_actions = _defense_eligible_actions(policy_input, discard_actions)

    remaining_counts = _root_remaining_counts(policy_input)
    hidden_tile_count = sum(remaining_counts)
    if hidden_tile_count < DEFAULT_HORIZON:
        raise FiniteHorizonCompletionPolicyError(
            "remaining hidden tile count is smaller than the search horizon: "
            f"{hidden_tile_count} hidden tiles cannot fill {DEFAULT_HORIZON} future "
            "self-draw slots"
        )

    evaluations = _evaluate_completion_masses(
        policy_input,
        eligible_actions,
        remaining_counts,
        DEFAULT_HORIZON,
        _FiniteHorizonEvaluator(),
    )
    maximum_mass = max(evaluation.completion_mass for evaluation in evaluations)
    if maximum_mass == 0:
        return _hand_value_aware_fallback(policy_input, evaluations)

    maximum_candidates = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.completion_mass == maximum_mass
    )
    if len(maximum_candidates) == 1:
        return maximum_candidates[0].action
    return _hand_value_aware_fallback(policy_input, maximum_candidates)


class GenbutsuDefenseFiniteHorizonHandValueAwarePolicy(TwoStepUkeirePolicy):
    """Push/Fold > Safety > FiniteHorizon > HandValueAwareのexperimental Policy。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        return PolicyDecision(
            action=_evaluate_and_choose_discard(policy_input, discard_actions),
            analysis=None,
        )
