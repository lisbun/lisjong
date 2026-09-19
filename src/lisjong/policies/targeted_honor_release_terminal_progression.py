"""限定的なhonor-release conflictだけexact terminal progressionで再判定するPolicy。

Issue #174のexperimental Policy。current promoted heuristic strength baselineである
`MechanismRiichiDefenseYakuhaiCallPolicy`をexact parentとして変更せず、Arena #256で
集中して観測された次のconflictだけを対象にする。

    far hand / FiniteHorizon all-zero
    current parent -> suited discard
    exact R5       -> honor-only best

activationはnormal-turn / closed / PUSH / all-zero / far-handへ限定し、parent actionと
post-discard shanten・current ukeire・retained real valueが同じ候補だけをR5へ渡す。
R5はIssue #169のexact integer evaluatorをsingle-sourceで再利用する。current actionが
R5 bestに含まれる場合、R5 bestがhonor-onlyでない場合、qualifying honor peerがない
場合は必ずparent actionを維持する。

`choose_action()`のAction契約はexact parentと同じまま、traced executionでは
purpose-specificな`TargetedHonorReleaseAnalysis`を同じ1回のdecision計算から返す。
analysisのためにR5やselectionを二重実行しない。
"""

from dataclasses import dataclass
from enum import Enum, auto

from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    FiniteHorizonCompletionPolicyError,
    _evaluate_completion_masses,
    _falling_factorial,
    _FiniteHorizonEvaluator,
    _root_remaining_counts,
)
from lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware import (
    DefenseFilterBranch,
    _evaluate_defense_filter,
)
from lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware import (
    _evaluate_and_choose_discard as _parent_offensive_evaluate_and_choose_discard,
)
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    HandValueCandidateEvaluation,
)
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _evaluate_and_choose_discard as _hand_value_aware_evaluate_and_choose_discard,
)
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    MechanismDefenseActivation,
    MechanismRiichiDefenseYakuhaiCallPolicy,
    _evaluate_mechanism_defense_filter,
)
from lisjong.policies.terminal_shanten_progression_mechanism_riichi_defense import (
    ProgressionCandidateEvaluation,
    _evaluate_progression_candidates,
    _select_from_completion_masses,
    _TerminalShantenProgressionEvaluator,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import TileCategory
from lisjong.structural_efficiency import discard_action_sort_key


class TargetedHonorReleaseBranch(Enum):
    """current parentのproduction filteringから直接導出するbranch。"""

    PUSH = auto()
    FOLD_COMMON_GENBUTSU = auto()
    MECHANISM_DEFENSE_FILTERED = auto()
    OTHER_CURRENT_FALLBACK = auto()


_DEFENSE_BRANCH_MAP = {
    DefenseFilterBranch.PUSH: TargetedHonorReleaseBranch.PUSH,
    DefenseFilterBranch.FOLD_COMMON_GENBUTSU: (
        TargetedHonorReleaseBranch.FOLD_COMMON_GENBUTSU
    ),
    DefenseFilterBranch.FOLD_FALLBACK_ALL_LEGAL: (
        TargetedHonorReleaseBranch.OTHER_CURRENT_FALLBACK
    ),
}


class TargetedHonorReleaseActivationStage(Enum):
    """#174 gateのどこでparent-equivalentへ戻ったか、またはswitchしたか。"""

    NOT_NORMAL_TURN = auto()
    OPEN_HAND = auto()
    NON_PUSH_BRANCH = auto()
    POSITIVE_COMPLETION = auto()
    PARENT_SHANTEN_TOO_CLOSE = auto()
    PARENT_NOT_MINIMUM_SHANTEN = auto()
    PARENT_NOT_MAXIMUM_UKEIRE = auto()
    PARENT_ALREADY_HONOR = auto()
    NO_QUALIFYING_HONOR_PEER = auto()
    R5_PARENT_BEST = auto()
    R5_NON_HONOR_ONLY_BEST = auto()
    R5_HONOR_ONLY_SWITCH = auto()


class HandValueDecisiveStage(Enum):
    """current HVA snapshotから安全に導出できるdecisive stage。"""

    SHANTEN = auto()
    CURRENT_UKEIRE = auto()
    RETAINED_REAL_VALUE = auto()
    YAKU_ROUTE = auto()
    SECOND_STEP = auto()
    STABLE_TIE = auto()
    UNRESOLVED = auto()


@dataclass(frozen=True, slots=True)
class TargetedHonorReleaseAnalysis(AnalysisTrace):
    """1 discard decisionで#174 pathが実際に観測したpurpose-specific値。"""

    activation_stage: TargetedHonorReleaseActivationStage
    parent_action: DiscardAction
    selected_action: DiscardAction
    action_changed: bool
    branch: TargetedHonorReleaseBranch
    closed_hand: bool
    parent_post_discard_shanten: int | None
    parent_current_ukeire: int | None
    parent_retained_real_value: int | None
    eligible_candidate_count: int
    target_candidate_count: int
    honor_target_candidate_count: int
    hva_decisive_stage: HandValueDecisiveStage
    horizon: int
    hidden_tile_count: int | None
    sequence_denominator: int | None
    progression_evaluations: tuple[ProgressionCandidateEvaluation, ...]
    r5_best_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.activation_stage, TargetedHonorReleaseActivationStage):
            raise TypeError(
                "activation_stage must be a TargetedHonorReleaseActivationStage"
            )
        if not isinstance(self.branch, TargetedHonorReleaseBranch):
            raise TypeError("branch must be a TargetedHonorReleaseBranch")
        if not isinstance(self.hva_decisive_stage, HandValueDecisiveStage):
            raise TypeError("hva_decisive_stage must be a HandValueDecisiveStage")
        for field_name in ("parent_action", "selected_action"):
            if not isinstance(getattr(self, field_name), DiscardAction):
                raise TypeError(f"{field_name} must be a DiscardAction")
        if type(self.action_changed) is not bool:
            raise TypeError("action_changed must be a bool")
        if type(self.closed_hand) is not bool:
            raise TypeError("closed_hand must be a bool")
        if self.action_changed != (self.parent_action != self.selected_action):
            raise ValueError(
                "action_changed must describe selected_action against parent_action"
            )
        for field_name in (
            "parent_post_discard_shanten",
            "parent_current_ukeire",
            "parent_retained_real_value",
            "hidden_tile_count",
            "sequence_denominator",
        ):
            value = getattr(self, field_name)
            if value is not None and type(value) is not int:
                raise TypeError(f"{field_name} must be an int or None")
        for field_name in (
            "eligible_candidate_count",
            "target_candidate_count",
            "honor_target_candidate_count",
            "horizon",
            "r5_best_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an int")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.eligible_candidate_count <= 0:
            raise ValueError("eligible_candidate_count must be positive")
        if self.honor_target_candidate_count > self.target_candidate_count:
            raise ValueError(
                "honor_target_candidate_count must not exceed target_candidate_count"
            )
        if type(self.progression_evaluations) is not tuple:
            raise TypeError("progression_evaluations must be a tuple")
        if any(
            not isinstance(evaluation, ProgressionCandidateEvaluation)
            for evaluation in self.progression_evaluations
        ):
            raise TypeError(
                "progression_evaluations must contain only "
                "ProgressionCandidateEvaluation values"
            )
        if self.progression_evaluations:
            if self.sequence_denominator is None or self.hidden_tile_count is None:
                raise ValueError(
                    "R5 evaluation requires hidden_tile_count and sequence_denominator"
                )
            if len(self.progression_evaluations) != self.target_candidate_count:
                raise ValueError(
                    "progression_evaluations must cover every target candidate"
                )
        elif self.r5_best_count != 0:
            raise ValueError("r5_best_count must be zero when R5 was not evaluated")


def _is_closed_hand(policy_input: PolicyInput) -> bool:
    """ANKANだけはclosedとして維持し、それ以外のmeldがあればopenとする。"""
    own_melds = policy_input.players[int(policy_input.self_seat)].melds
    return all(meld.kind is MeldKind.ANKAN for meld in own_melds)


def _classify_branch(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[TargetedHonorReleaseBranch, tuple[DiscardAction, ...]]:
    """production filtering helperだけでcurrent branch / eligible setを導出する。"""
    mechanism_evaluation = _evaluate_mechanism_defense_filter(
        policy_input, discard_actions
    )
    if mechanism_evaluation.activation is MechanismDefenseActivation.ACTIVATED:
        return (
            TargetedHonorReleaseBranch.MECHANISM_DEFENSE_FILTERED,
            mechanism_evaluation.eligible_actions,
        )

    defense_evaluation = _evaluate_defense_filter(
        policy_input, mechanism_evaluation.eligible_actions
    )
    return (
        _DEFENSE_BRANCH_MAP[defense_evaluation.branch],
        defense_evaluation.eligible_actions,
    )


def _parent_action(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> DiscardAction:
    """exact parentのmechanism filter -> offensive compositionをそのまま実行する。"""
    mechanism_evaluation = _evaluate_mechanism_defense_filter(
        policy_input, discard_actions
    )
    return _parent_offensive_evaluate_and_choose_discard(
        policy_input, mechanism_evaluation.eligible_actions
    )


def _snapshot_by_action(
    snapshots: tuple[HandValueCandidateEvaluation, ...],
) -> dict[DiscardAction, HandValueCandidateEvaluation]:
    return {snapshot.action: snapshot for snapshot in snapshots}


def _classify_hva_decisive_stage(
    selected_action: DiscardAction,
    snapshots: tuple[HandValueCandidateEvaluation, ...],
) -> HandValueDecisiveStage:
    """HVAのstaged snapshotだけから確実に言えるdecisive stageを分類する。"""
    by_action = _snapshot_by_action(snapshots)
    selected = by_action.get(selected_action)
    if selected is None:
        return HandValueDecisiveStage.UNRESOLVED

    minimum_shanten = min(snapshot.post_discard_shanten for snapshot in snapshots)
    shanten_finalists = tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.post_discard_shanten == minimum_shanten
    )
    if selected.post_discard_shanten != minimum_shanten:
        return HandValueDecisiveStage.UNRESOLVED
    if len(shanten_finalists) == 1:
        return HandValueDecisiveStage.SHANTEN

    if any(snapshot.current_ukeire_count is None for snapshot in shanten_finalists):
        return HandValueDecisiveStage.UNRESOLVED
    maximum_ukeire = max(
        snapshot.current_ukeire_count for snapshot in shanten_finalists
    )
    ukeire_finalists = tuple(
        snapshot
        for snapshot in shanten_finalists
        if snapshot.current_ukeire_count == maximum_ukeire
    )
    if selected.current_ukeire_count != maximum_ukeire:
        return HandValueDecisiveStage.UNRESOLVED
    if len(ukeire_finalists) == 1:
        return HandValueDecisiveStage.CURRENT_UKEIRE

    if any(snapshot.retained_real_value is None for snapshot in ukeire_finalists):
        return HandValueDecisiveStage.UNRESOLVED
    maximum_real_value = max(
        snapshot.retained_real_value for snapshot in ukeire_finalists
    )
    real_value_finalists = tuple(
        snapshot
        for snapshot in ukeire_finalists
        if snapshot.retained_real_value == maximum_real_value
    )
    if selected.retained_real_value != maximum_real_value:
        return HandValueDecisiveStage.UNRESOLVED
    if len(real_value_finalists) == 1:
        return HandValueDecisiveStage.RETAINED_REAL_VALUE

    if any(snapshot.yaku_route_value is None for snapshot in real_value_finalists):
        return HandValueDecisiveStage.UNRESOLVED
    maximum_route_value = max(
        snapshot.yaku_route_value for snapshot in real_value_finalists
    )
    route_finalists = tuple(
        snapshot
        for snapshot in real_value_finalists
        if snapshot.yaku_route_value == maximum_route_value
    )
    if selected.yaku_route_value != maximum_route_value:
        return HandValueDecisiveStage.UNRESOLVED
    if len(route_finalists) == 1:
        return HandValueDecisiveStage.YAKU_ROUTE

    if minimum_shanten == 0:
        stable = min(
            (snapshot.action for snapshot in route_finalists),
            key=discard_action_sort_key,
        )
        return (
            HandValueDecisiveStage.STABLE_TIE
            if selected_action == stable
            else HandValueDecisiveStage.UNRESOLVED
        )

    if any(snapshot.second_step_ukeire_score is None for snapshot in route_finalists):
        return HandValueDecisiveStage.UNRESOLVED
    maximum_second_step = max(
        snapshot.second_step_ukeire_score for snapshot in route_finalists
    )
    second_step_finalists = tuple(
        snapshot
        for snapshot in route_finalists
        if snapshot.second_step_ukeire_score == maximum_second_step
    )
    if selected.second_step_ukeire_score != maximum_second_step:
        return HandValueDecisiveStage.UNRESOLVED
    if len(second_step_finalists) == 1:
        return HandValueDecisiveStage.SECOND_STEP

    stable = min(
        (snapshot.action for snapshot in second_step_finalists),
        key=discard_action_sort_key,
    )
    return (
        HandValueDecisiveStage.STABLE_TIE
        if selected_action == stable
        else HandValueDecisiveStage.UNRESOLVED
    )


def _target_candidates(
    parent_action: DiscardAction,
    snapshots: tuple[HandValueCandidateEvaluation, ...],
) -> tuple[HandValueCandidateEvaluation, ...]:
    """parentとshanten / ukeire / retained real valueがexactly同じ候補だけ残す。"""
    by_action = _snapshot_by_action(snapshots)
    parent = by_action[parent_action]
    return tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.post_discard_shanten == parent.post_discard_shanten
        and snapshot.current_ukeire_count == parent.current_ukeire_count
        and snapshot.retained_real_value == parent.retained_real_value
    )


def _analysis(
    *,
    stage: TargetedHonorReleaseActivationStage,
    parent_action: DiscardAction,
    selected_action: DiscardAction,
    branch: TargetedHonorReleaseBranch,
    closed_hand: bool,
    parent_snapshot: HandValueCandidateEvaluation | None,
    eligible_candidate_count: int,
    target_candidate_count: int = 0,
    honor_target_candidate_count: int = 0,
    hva_decisive_stage: HandValueDecisiveStage = HandValueDecisiveStage.UNRESOLVED,
    hidden_tile_count: int | None = None,
    sequence_denominator: int | None = None,
    progression_evaluations: tuple[ProgressionCandidateEvaluation, ...] = (),
    r5_best_count: int = 0,
) -> TargetedHonorReleaseAnalysis:
    return TargetedHonorReleaseAnalysis(
        activation_stage=stage,
        parent_action=parent_action,
        selected_action=selected_action,
        action_changed=selected_action != parent_action,
        branch=branch,
        closed_hand=closed_hand,
        parent_post_discard_shanten=(
            None if parent_snapshot is None else parent_snapshot.post_discard_shanten
        ),
        parent_current_ukeire=(
            None if parent_snapshot is None else parent_snapshot.current_ukeire_count
        ),
        parent_retained_real_value=(
            None if parent_snapshot is None else parent_snapshot.retained_real_value
        ),
        eligible_candidate_count=eligible_candidate_count,
        target_candidate_count=target_candidate_count,
        honor_target_candidate_count=honor_target_candidate_count,
        hva_decisive_stage=hva_decisive_stage,
        horizon=DEFAULT_HORIZON,
        hidden_tile_count=hidden_tile_count,
        sequence_denominator=sequence_denominator,
        progression_evaluations=progression_evaluations,
        r5_best_count=r5_best_count,
    )


def _evaluate_and_choose_discard(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, TargetedHonorReleaseAnalysis]:
    """#174のnarrow gateとselectionを1 decision分だけ評価する。"""
    discard_actions = tuple(discard_actions)
    if not discard_actions:
        raise ValueError("discard_actions must not be empty")

    branch, eligible_actions = _classify_branch(policy_input, discard_actions)
    closed_hand = _is_closed_hand(policy_input)

    # current OwnHandStateではdrawn_tileがnormal self-drawのphase metadataであり、
    # post-call discardではNoneになる。metadataがない局面へ#174を外挿せずparent維持。
    if policy_input.own_hand.drawn_tile is None:
        parent_action = _parent_action(policy_input, discard_actions)
        return parent_action, _analysis(
            stage=TargetedHonorReleaseActivationStage.NOT_NORMAL_TURN,
            parent_action=parent_action,
            selected_action=parent_action,
            branch=branch,
            closed_hand=closed_hand,
            parent_snapshot=None,
            eligible_candidate_count=len(eligible_actions),
        )

    if not closed_hand:
        parent_action = _parent_action(policy_input, discard_actions)
        return parent_action, _analysis(
            stage=TargetedHonorReleaseActivationStage.OPEN_HAND,
            parent_action=parent_action,
            selected_action=parent_action,
            branch=branch,
            closed_hand=False,
            parent_snapshot=None,
            eligible_candidate_count=len(eligible_actions),
        )

    if branch is not TargetedHonorReleaseBranch.PUSH:
        parent_action = _parent_action(policy_input, discard_actions)
        return parent_action, _analysis(
            stage=TargetedHonorReleaseActivationStage.NON_PUSH_BRANCH,
            parent_action=parent_action,
            selected_action=parent_action,
            branch=branch,
            closed_hand=True,
            parent_snapshot=None,
            eligible_candidate_count=len(eligible_actions),
        )

    remaining_counts = _root_remaining_counts(policy_input)
    hidden_tile_count = sum(remaining_counts)
    if hidden_tile_count < DEFAULT_HORIZON:
        raise FiniteHorizonCompletionPolicyError(
            "remaining hidden tile count is smaller than the search horizon: "
            f"{hidden_tile_count} hidden tiles cannot fill {DEFAULT_HORIZON} future "
            "self-draw slots"
        )
    sequence_denominator: int | None = None
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
    if maximum_completion > 0:
        parent_action = _select_from_completion_masses(
            policy_input, completion_evaluations, maximum_completion
        )
        return parent_action, _analysis(
            stage=TargetedHonorReleaseActivationStage.POSITIVE_COMPLETION,
            parent_action=parent_action,
            selected_action=parent_action,
            branch=branch,
            closed_hand=True,
            parent_snapshot=None,
            eligible_candidate_count=len(eligible_actions),
            hidden_tile_count=hidden_tile_count,
            sequence_denominator=sequence_denominator,
        )

    parent_action, hva_snapshots = _hand_value_aware_evaluate_and_choose_discard(
        policy_input, eligible_actions
    )
    by_action = _snapshot_by_action(hva_snapshots)
    parent_snapshot = by_action[parent_action]
    decisive_stage = _classify_hva_decisive_stage(parent_action, hva_snapshots)

    minimum_shanten = min(snapshot.post_discard_shanten for snapshot in hva_snapshots)
    if parent_snapshot.post_discard_shanten < 3:
        return parent_action, _analysis(
            stage=TargetedHonorReleaseActivationStage.PARENT_SHANTEN_TOO_CLOSE,
            parent_action=parent_action,
            selected_action=parent_action,
            branch=branch,
            closed_hand=True,
            parent_snapshot=parent_snapshot,
            eligible_candidate_count=len(eligible_actions),
            hva_decisive_stage=decisive_stage,
            hidden_tile_count=hidden_tile_count,
            sequence_denominator=sequence_denominator,
        )
    if parent_snapshot.post_discard_shanten != minimum_shanten:
        return parent_action, _analysis(
            stage=TargetedHonorReleaseActivationStage.PARENT_NOT_MINIMUM_SHANTEN,
            parent_action=parent_action,
            selected_action=parent_action,
            branch=branch,
            closed_hand=True,
            parent_snapshot=parent_snapshot,
            eligible_candidate_count=len(eligible_actions),
            hva_decisive_stage=decisive_stage,
            hidden_tile_count=hidden_tile_count,
            sequence_denominator=sequence_denominator,
        )

    shanten_finalists = tuple(
        snapshot
        for snapshot in hva_snapshots
        if snapshot.post_discard_shanten == minimum_shanten
    )
    ukeire_values = tuple(
        snapshot.current_ukeire_count
        for snapshot in shanten_finalists
        if snapshot.current_ukeire_count is not None
    )
    if (
        parent_snapshot.current_ukeire_count is None
        or not ukeire_values
        or parent_snapshot.current_ukeire_count != max(ukeire_values)
    ):
        return parent_action, _analysis(
            stage=TargetedHonorReleaseActivationStage.PARENT_NOT_MAXIMUM_UKEIRE,
            parent_action=parent_action,
            selected_action=parent_action,
            branch=branch,
            closed_hand=True,
            parent_snapshot=parent_snapshot,
            eligible_candidate_count=len(eligible_actions),
            hva_decisive_stage=decisive_stage,
            hidden_tile_count=hidden_tile_count,
            sequence_denominator=sequence_denominator,
        )

    if parent_action.tile.tile_type.category is TileCategory.HONOR:
        return parent_action, _analysis(
            stage=TargetedHonorReleaseActivationStage.PARENT_ALREADY_HONOR,
            parent_action=parent_action,
            selected_action=parent_action,
            branch=branch,
            closed_hand=True,
            parent_snapshot=parent_snapshot,
            eligible_candidate_count=len(eligible_actions),
            hva_decisive_stage=decisive_stage,
            hidden_tile_count=hidden_tile_count,
            sequence_denominator=sequence_denominator,
        )

    target_snapshots = _target_candidates(parent_action, hva_snapshots)
    honor_target_count = sum(
        snapshot.action.tile.tile_type.category is TileCategory.HONOR
        for snapshot in target_snapshots
    )
    if honor_target_count == 0:
        return parent_action, _analysis(
            stage=TargetedHonorReleaseActivationStage.NO_QUALIFYING_HONOR_PEER,
            parent_action=parent_action,
            selected_action=parent_action,
            branch=branch,
            closed_hand=True,
            parent_snapshot=parent_snapshot,
            eligible_candidate_count=len(eligible_actions),
            target_candidate_count=len(target_snapshots),
            honor_target_candidate_count=0,
            hva_decisive_stage=decisive_stage,
            hidden_tile_count=hidden_tile_count,
            sequence_denominator=sequence_denominator,
        )

    sequence_denominator = _falling_factorial(hidden_tile_count, DEFAULT_HORIZON)
    completion_by_action = {
        evaluation.action: evaluation for evaluation in completion_evaluations
    }
    target_completion_evaluations = tuple(
        completion_by_action[snapshot.action] for snapshot in target_snapshots
    )
    progression_evaluations = _evaluate_progression_candidates(
        policy_input,
        target_completion_evaluations,
        remaining_counts,
        DEFAULT_HORIZON,
        _TerminalShantenProgressionEvaluator(),
    )
    minimum_terminal_mass = min(
        evaluation.terminal_shanten_mass for evaluation in progression_evaluations
    )
    r5_best = tuple(
        evaluation
        for evaluation in progression_evaluations
        if evaluation.terminal_shanten_mass == minimum_terminal_mass
    )

    if any(evaluation.action == parent_action for evaluation in r5_best):
        return parent_action, _analysis(
            stage=TargetedHonorReleaseActivationStage.R5_PARENT_BEST,
            parent_action=parent_action,
            selected_action=parent_action,
            branch=branch,
            closed_hand=True,
            parent_snapshot=parent_snapshot,
            eligible_candidate_count=len(eligible_actions),
            target_candidate_count=len(target_snapshots),
            honor_target_candidate_count=honor_target_count,
            hva_decisive_stage=decisive_stage,
            hidden_tile_count=hidden_tile_count,
            sequence_denominator=sequence_denominator,
            progression_evaluations=progression_evaluations,
            r5_best_count=len(r5_best),
        )

    if any(
        evaluation.action.tile.tile_type.category is not TileCategory.HONOR
        for evaluation in r5_best
    ):
        return parent_action, _analysis(
            stage=TargetedHonorReleaseActivationStage.R5_NON_HONOR_ONLY_BEST,
            parent_action=parent_action,
            selected_action=parent_action,
            branch=branch,
            closed_hand=True,
            parent_snapshot=parent_snapshot,
            eligible_candidate_count=len(eligible_actions),
            target_candidate_count=len(target_snapshots),
            honor_target_candidate_count=honor_target_count,
            hva_decisive_stage=decisive_stage,
            hidden_tile_count=hidden_tile_count,
            sequence_denominator=sequence_denominator,
            progression_evaluations=progression_evaluations,
            r5_best_count=len(r5_best),
        )

    if len(r5_best) == 1:
        selected_action = r5_best[0].action
    else:
        selected_action, _ = _hand_value_aware_evaluate_and_choose_discard(
            policy_input, tuple(evaluation.action for evaluation in r5_best)
        )
    return selected_action, _analysis(
        stage=TargetedHonorReleaseActivationStage.R5_HONOR_ONLY_SWITCH,
        parent_action=parent_action,
        selected_action=selected_action,
        branch=branch,
        closed_hand=True,
        parent_snapshot=parent_snapshot,
        eligible_candidate_count=len(eligible_actions),
        target_candidate_count=len(target_snapshots),
        honor_target_candidate_count=honor_target_count,
        hva_decisive_stage=decisive_stage,
        hidden_tile_count=hidden_tile_count,
        sequence_denominator=sequence_denominator,
        progression_evaluations=progression_evaluations,
        r5_best_count=len(r5_best),
    )


class TargetedHonorReleaseTerminalProgressionPolicy(
    MechanismRiichiDefenseYakuhaiCallPolicy
):
    """#256で集中したsuited-discard / honor-release conflictだけをR5で再判定する。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        selected, analysis = _evaluate_and_choose_discard(policy_input, discard_actions)
        return PolicyDecision(action=selected, analysis=analysis)
