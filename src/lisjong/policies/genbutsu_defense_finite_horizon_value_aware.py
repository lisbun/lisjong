"""GenbutsuDefense・FiniteHorizon・ValueAwareを明示的に合成するPolicy。

Issue #122のexperimental Policyとして、既存Policyを変更せず次のpriorityを
`TwoStepUkeirePolicy._decide_discard()` extension pointへ実装する。

    Genbutsu safety constraint
    > FiniteHorizon completion mass
    > ValueAware ranking

Genbutsuは既存と同じactivation conditionを使い、発動時は全リーチ者への
common genbutsuだけをeligible setにする。FiniteHorizonのunique positive
maximumは即採用し、positive maximum tieではmaximum subsetだけ、all-zeroでは
eligible set全部をValueAwareへ渡す。completion massで負けた候補をValueAwareで
復活させない。

winning action、Always Riichi、pass、既存fallbackのorchestrationは基底classから
継承する。本Issueではcombined-specific analysisを追加せず、打牌decisionの
`analysis`は`None`とする。
"""

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
from lisjong.policies.two_step_ukeire import (
    TwoStepUkeirePolicy,
    _DecisionShantenEvaluator,
    _evaluate_post_discard_hands,
)
from lisjong.policies.value_aware_two_step_ukeire import (
    _evaluate_and_choose_discard as _value_aware_evaluate_and_choose_discard,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput


def _genbutsu_eligible_actions(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, ...]:
    """既存Genbutsu activation semanticでFiniteHorizonのeligible setを返す。"""
    riichi_players = _opponent_riichi_players(policy_input)
    if not riichi_players:
        return discard_actions

    evaluator = _DecisionShantenEvaluator()
    evaluated = _evaluate_post_discard_hands(policy_input, discard_actions, evaluator)
    if min(candidate.post_discard_shanten for candidate in evaluated) < 1:
        return discard_actions

    common_tile_types = _common_genbutsu_tile_types(riichi_players)
    genbutsu_actions = tuple(
        action
        for action in discard_actions
        if action.tile.tile_type in common_tile_types
    )
    return genbutsu_actions or discard_actions


def _value_aware_fallback(
    policy_input: PolicyInput,
    evaluations: tuple[FiniteHorizonCandidateEvaluation, ...],
) -> DiscardAction:
    """FiniteHorizonから渡されたcandidate subsetを既存ValueAwareで選ぶ。"""
    selected, _ = _value_aware_evaluate_and_choose_discard(
        policy_input, tuple(evaluation.action for evaluation in evaluations)
    )
    return selected


def _evaluate_and_choose_discard(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> DiscardAction:
    """Issue #122の3-stage compositionを1回のselection pathで実行する。"""
    eligible_actions = _genbutsu_eligible_actions(policy_input, discard_actions)

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
        return _value_aware_fallback(policy_input, evaluations)

    maximum_candidates = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.completion_mass == maximum_mass
    )
    if len(maximum_candidates) == 1:
        return maximum_candidates[0].action
    return _value_aware_fallback(policy_input, maximum_candidates)


class GenbutsuDefenseFiniteHorizonValueAwarePolicy(TwoStepUkeirePolicy):
    """Genbutsu > FiniteHorizon > ValueAwareのexperimental Policy。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        return PolicyDecision(
            action=_evaluate_and_choose_discard(policy_input, discard_actions),
            analysis=None,
        )
