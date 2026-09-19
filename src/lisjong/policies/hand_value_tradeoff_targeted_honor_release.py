"""Current Heuristic ChampionへIssue #175 HandValue v2を保守的に合成する。

Issue #182 experimental Policy。current Championの#174 targeted honor-release
`R5_HONOR_ONLY_SWITCH`を最優先で保持し、それ以外のordinary discardでは
exact #175 HandValue-v2 selectionを許可する。

mechanism / defense filteringとFiniteHorizon evaluationは1 decisionにつき1回だけ
実行し、そのprecomputed valuesを#174 gateと#175 selectorで共有する。
"""

from dataclasses import dataclass
from enum import Enum, auto

from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    FiniteHorizonCandidateEvaluation,
    FiniteHorizonCompletionPolicyError,
    _evaluate_completion_masses,
    _FiniteHorizonEvaluator,
    _root_remaining_counts,
)
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _evaluate_and_choose_discard as _hand_value_aware_evaluate_and_choose_discard,
)
from lisjong.policies.hand_value_tradeoff_mechanism_riichi_defense import (
    _select_finite_horizon_hand_value_v2,
)
from lisjong.policies.targeted_honor_release_terminal_progression import (
    TargetedHonorReleaseActivationStage,
    TargetedHonorReleaseAnalysis,
    TargetedHonorReleaseBranch,
    TargetedHonorReleaseTerminalProgressionPolicy,
    _classify_branch,
    _evaluate_all_zero_targeted_honor_release,
    _is_closed_hand,
)
from lisjong.policies.terminal_shanten_progression_mechanism_riichi_defense import (
    _select_from_completion_masses,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput


class HandValueTradeoffTargetedHonorReleaseSelectionSource(Enum):
    """Combined Policyが最終actionを採用したsource。"""

    CHAMPION_TARGETED_HONOR_RELEASE = auto()
    HAND_VALUE_V2 = auto()
    SHARED_ACTION = auto()


@dataclass(frozen=True, slots=True)
class HandValueTradeoffTargetedHonorReleaseAnalysis(AnalysisTrace):
    """#182 compositionの1 ordinary-discard decision trace。"""

    champion_analysis: TargetedHonorReleaseAnalysis | None
    hand_value_v2_action: DiscardAction | None
    champion_action: DiscardAction
    former_parent_action: DiscardAction
    selected_action: DiscardAction
    selection_source: HandValueTradeoffTargetedHonorReleaseSelectionSource
    action_changed_vs_champion: bool
    action_changed_vs_former_parent: bool

    def __post_init__(self) -> None:
        if self.champion_analysis is not None and not isinstance(
            self.champion_analysis, TargetedHonorReleaseAnalysis
        ):
            raise TypeError(
                "champion_analysis must be None or TargetedHonorReleaseAnalysis"
            )
        if self.hand_value_v2_action is not None and not isinstance(
            self.hand_value_v2_action, DiscardAction
        ):
            raise TypeError("hand_value_v2_action must be None or DiscardAction")
        for field_name in (
            "champion_action",
            "former_parent_action",
            "selected_action",
        ):
            if not isinstance(getattr(self, field_name), DiscardAction):
                raise TypeError(f"{field_name} must be a DiscardAction")
        if not isinstance(
            self.selection_source,
            HandValueTradeoffTargetedHonorReleaseSelectionSource,
        ):
            raise TypeError(
                "selection_source must be a "
                "HandValueTradeoffTargetedHonorReleaseSelectionSource"
            )
        for field_name in (
            "action_changed_vs_champion",
            "action_changed_vs_former_parent",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")

        if self.action_changed_vs_champion != (
            self.selected_action != self.champion_action
        ):
            raise ValueError(
                "action_changed_vs_champion must describe selected vs champion"
            )
        if self.action_changed_vs_former_parent != (
            self.selected_action != self.former_parent_action
        ):
            raise ValueError(
                "action_changed_vs_former_parent must describe selected vs former parent"
            )

        source = self.selection_source
        if (
            source
            is HandValueTradeoffTargetedHonorReleaseSelectionSource.CHAMPION_TARGETED_HONOR_RELEASE
        ):
            if self.champion_analysis is None:
                raise ValueError(
                    "Champion targeted selection requires champion_analysis"
                )
            if (
                self.champion_analysis.activation_stage
                is not TargetedHonorReleaseActivationStage.R5_HONOR_ONLY_SWITCH
            ):
                raise ValueError(
                    "Champion targeted selection requires R5_HONOR_ONLY_SWITCH"
                )
            if self.selected_action != self.champion_action:
                raise ValueError(
                    "Champion targeted selection must preserve champion action"
                )
            if self.hand_value_v2_action is not None:
                raise ValueError(
                    "HandValue v2 must not run after a confirmed Champion switch"
                )
            return

        if self.hand_value_v2_action is None:
            raise ValueError(
                "non-Champion-targeted selection requires hand_value_v2_action"
            )
        if self.hand_value_v2_action != self.selected_action:
            raise ValueError("hand_value_v2_action must equal selected_action")

        if (
            source
            is HandValueTradeoffTargetedHonorReleaseSelectionSource.SHARED_ACTION
        ):
            if self.selected_action != self.champion_action:
                raise ValueError("SHARED_ACTION requires Champion and v2 agreement")
        elif self.selected_action == self.champion_action:
            raise ValueError("HAND_VALUE_V2 requires divergence from Champion")


def _current_parent_from_completion(
    policy_input: PolicyInput,
    evaluations: tuple[FiniteHorizonCandidateEvaluation, ...],
) -> DiscardAction:
    """Precomputed FiniteHorizon valuesからformer Champion parent actionを返す。"""
    maximum_mass = max(evaluation.completion_mass for evaluation in evaluations)
    if maximum_mass > 0:
        return _select_from_completion_masses(
            policy_input, evaluations, maximum_mass
        )
    selected, _ = _hand_value_aware_evaluate_and_choose_discard(
        policy_input, tuple(evaluation.action for evaluation in evaluations)
    )
    return selected


def _evaluate_and_choose_discard(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, HandValueTradeoffTargetedHonorReleaseAnalysis]:
    """#174 precedence + exact #175 fallbackをsingle-pass common evaluationで合成する。"""
    discard_actions = tuple(discard_actions)
    if not discard_actions:
        raise ValueError("discard_actions must not be empty")

    branch, eligible_actions = _classify_branch(policy_input, discard_actions)
    closed_hand = _is_closed_hand(policy_input)

    remaining_counts = _root_remaining_counts(policy_input)
    hidden_tile_count = sum(remaining_counts)
    if hidden_tile_count < DEFAULT_HORIZON:
        raise FiniteHorizonCompletionPolicyError(
            "remaining hidden tile count is smaller than the search horizon: "
            f"{hidden_tile_count} hidden tiles cannot fill {DEFAULT_HORIZON} future "
            "self-draw slots"
        )

    completion_evaluations = _evaluate_completion_masses(
        policy_input,
        eligible_actions,
        remaining_counts,
        DEFAULT_HORIZON,
        _FiniteHorizonEvaluator(),
    )
    maximum_completion = max(
        evaluation.completion_mass for evaluation in completion_evaluations
    )

    champion_analysis: TargetedHonorReleaseAnalysis | None = None
    if (
        policy_input.own_hand.drawn_tile is not None
        and closed_hand
        and branch is TargetedHonorReleaseBranch.PUSH
        and maximum_completion == 0
    ):
        champion_action, champion_analysis = (
            _evaluate_all_zero_targeted_honor_release(
                policy_input,
                branch=branch,
                eligible_actions=eligible_actions,
                remaining_counts=remaining_counts,
                hidden_tile_count=hidden_tile_count,
                completion_evaluations=completion_evaluations,
            )
        )
        former_parent_action = champion_analysis.parent_action
        if (
            champion_analysis.activation_stage
            is TargetedHonorReleaseActivationStage.R5_HONOR_ONLY_SWITCH
        ):
            analysis = HandValueTradeoffTargetedHonorReleaseAnalysis(
                champion_analysis=champion_analysis,
                hand_value_v2_action=None,
                champion_action=champion_action,
                former_parent_action=former_parent_action,
                selected_action=champion_action,
                selection_source=(
                    HandValueTradeoffTargetedHonorReleaseSelectionSource.
                    CHAMPION_TARGETED_HONOR_RELEASE
                ),
                action_changed_vs_champion=False,
                action_changed_vs_former_parent=(
                    champion_action != former_parent_action
                ),
            )
            return champion_action, analysis
    else:
        champion_action = _current_parent_from_completion(
            policy_input, completion_evaluations
        )
        former_parent_action = champion_action

    hand_value_v2_action = _select_finite_horizon_hand_value_v2(
        policy_input, completion_evaluations, remaining_counts
    )
    source = (
        HandValueTradeoffTargetedHonorReleaseSelectionSource.SHARED_ACTION
        if hand_value_v2_action == champion_action
        else HandValueTradeoffTargetedHonorReleaseSelectionSource.HAND_VALUE_V2
    )
    analysis = HandValueTradeoffTargetedHonorReleaseAnalysis(
        champion_analysis=champion_analysis,
        hand_value_v2_action=hand_value_v2_action,
        champion_action=champion_action,
        former_parent_action=former_parent_action,
        selected_action=hand_value_v2_action,
        selection_source=source,
        action_changed_vs_champion=hand_value_v2_action != champion_action,
        action_changed_vs_former_parent=(
            hand_value_v2_action != former_parent_action
        ),
    )
    return hand_value_v2_action, analysis


class HandValueTradeoffTargetedHonorReleasePolicy(
    TargetedHonorReleaseTerminalProgressionPolicy
):
    """Current Heuristic Championへexact #175 HandValue v2を追加したcandidate。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        selected, analysis = _evaluate_and_choose_discard(
            policy_input, discard_actions
        )
        return PolicyDecision(action=selected, analysis=analysis)
