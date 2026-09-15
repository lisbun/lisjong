"""current baseline専用のoffline structural-efficiency regret diagnostic seam。

Issue #172を実装する。current promoted heuristic strength baselineである
`MechanismRiichiDefenseYakuhaiCallPolicy`の`choose_action()` selection behaviorを
一切変更せず、1つのplayer-safe discard decisionについて、現在の選択が各
structural-efficiency metricのbest candidateに対してどの程度regretを持つかを
lisjong-owned semanticsで観測できるようにする。

このmoduleはgeneric heuristic frameworkや universal Q-value APIではない。
`analyze_mechanism_riichi_defense_offensive_efficiency()`という、current
baseline専用のpurpose-specific analysisだけを提供する。Arenaはこのmoduleから
module-level symbolを直接importでき、`lisjong.policies.__init__`のPolicy
inventoryへは追加しない。

## production selection / filtering logicの再利用

`baseline_eligible_actions`と`baseline_selected_action`は、production
`MechanismRiichiDefenseYakuhaiCallPolicy._decide_discard()`が実際に使用する

    legal discard
    -> _mechanism_defense_eligible_actions()          (Issue #163)
    -> _defense_eligible_actions()                     (parent Push/Fold・Safety)
    -> FiniteHorizon completion mass
    -> HandValueAware

と exact same semanticsのhelperをそのまま呼び出して得る。diagnostic用に別の
selection algorithmは実装しない。

branch classification（`PUSH` / `FOLD_COMMON_GENBUTSU` /
`MECHANISM_DEFENSE_FILTERED` / `OTHER_CURRENT_FALLBACK`）も、
`mechanism_riichi_defense_yakuhai_call._evaluate_mechanism_defense_filter()`と
`genbutsu_defense_finite_horizon_hand_value_aware._evaluate_defense_filter()`と
いう、production filteringから切り出したsingle-source typed helperの戻り値だけ
から導出する。`mechanism_eligible_actions != input_actions`のようなset size比較
からactivationを推測しない（danger tieでcandidate setが縮まらないactivationを
区別するため）。

## full-legal / baseline-eligible universeの分離

R1（post-discard shanten）・R4（FiniteHorizon completion mass）は、full legal
action集合について1回だけ評価し、baseline-eligible universeの値は同じ
evaluation結果をmembershipでfilterして再利用する（二重計算しない）。

R2（現在受け入れ）・R3（2段階受け入れ）は、universeごとにstage applicability
（minimum shanten到達・tied-maximum ukeire到達）が異なり得るため、既存
`two_step_ukeire._evaluate_and_choose_prepared()`をuniverseごとに独立実行する。
mutable working stateをuniverse間で共有・再利用しない。

## R5 optional terminal-shanten progression

`include_terminal_progression=True`のexplicit opt-inのときだけ、Issue #169 /
PR #170のexact adaptive expected-terminal-shanten evaluator
（`terminal_shanten_progression_mechanism_riichi_defense._TerminalShantenProgressionEvaluator`）
を起動する。R1/R4と同じくfull-legal universeとbaseline-eligible universeを
分離し、それぞれ`full_legal_terminal_progression_summary` /
`baseline_eligible_terminal_progression_summary`として別々に保持する。

`baseline_eligible_actions`はfull legalのsubsetなので、full legal universeが
all-zeroならbaseline eligible universeも必ずall-zeroである。そのため

    full legal universeがall-zero
        -> full legal action全体でprogression DPを1回だけ実行し、
           full-legalとbaseline-eligibleの両方のsummaryをそのcandidate結果
           から導出する（baseline-eligible側のためにDPを再実行しない）
    full legalは非all-zeroだがbaseline eligible universeがall-zero
        -> baseline eligible actionだけでprogression DPを実行し、
           baseline-eligible側のsummaryだけを持つ（full legal側はNone）
    どちらもall-zeroでない
        -> progression DPを実行せず、両方Noneのまま

とする。default callおよびどちらのuniverseもall-zeroでない場合、progression
DPを絶対に起動しない。canonical値はexact integer（`terminal_shanten_mass` /
`terminal_shanten_regret_mass`）であり、floatはconsumer側の表示用derived
valueにだけ現れる。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto

from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    FiniteHorizonCandidateEvaluation,
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
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    MechanismDefenseActivation,
    _evaluate_mechanism_defense_filter,
    _mechanism_defense_eligible_actions,
)
from lisjong.policies.terminal_shanten_progression_mechanism_riichi_defense import (
    TERMINAL_SHANTEN_AXIS,
    _evaluate_progression_candidates,
    _TerminalShantenProgressionEvaluator,
)
from lisjong.policies.two_step_ukeire import (
    TwoStepUkeireCandidateEvaluation,
    _DecisionShantenEvaluator,
    _discard_action_sort_key,
    _evaluate_and_choose_prepared,
    _evaluate_post_discard_hands,
    _known_tile_counts,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import TileType


class MechanismRiichiDefenseOffensiveEfficiencyDiagnosticError(Exception):
    """diagnostic入力が不整合または未定義の状況をfail closedする場合。"""


class OffensiveEfficiencyBranch(Enum):
    """current baselineが実際に選んだfiltering branchのtyped classification。

    `mechanism_eligible_actions != input_actions`のようなset size比較では
    activationを推測できない（danger tieでcandidate setが縮まらない
    `MECHANISM_DEFENSE_FILTERED`が起こり得る）ため、production filteringの
    single-source typed evaluationから直接導出する。
    """

    PUSH = auto()
    FOLD_COMMON_GENBUTSU = auto()
    MECHANISM_DEFENSE_FILTERED = auto()
    OTHER_CURRENT_FALLBACK = auto()


_DEFENSE_BRANCH_TO_OFFENSIVE_EFFICIENCY_BRANCH = {
    DefenseFilterBranch.PUSH: OffensiveEfficiencyBranch.PUSH,
    DefenseFilterBranch.FOLD_COMMON_GENBUTSU: OffensiveEfficiencyBranch.FOLD_COMMON_GENBUTSU,
    DefenseFilterBranch.FOLD_FALLBACK_ALL_LEGAL: OffensiveEfficiencyBranch.OTHER_CURRENT_FALLBACK,
}


@dataclass(frozen=True, slots=True)
class OffensiveEfficiencyCandidateEvaluation:
    """1打牌候補についてR1-R5が実際に計算したsemantic評価値。

    `post_discard_shanten`（R1）とcompletion mass（R4）はuniverseに依存しない
    candidate固有値なので単一fieldである。R2 / R3はfull-legal universeと
    baseline-eligible universeでstage applicabilityが異なり得るため、universe
    ごとのfieldへ分ける。`None`は「評価未実行 / stage未到達」を表し、評価済み
    result `0`とは区別する。
    """

    action: DiscardAction
    baseline_eligible: bool
    post_discard_shanten: int
    full_legal_current_ukeire_count: int | None
    full_legal_second_step_ukeire_score: int | None
    baseline_eligible_current_ukeire_count: int | None
    baseline_eligible_second_step_ukeire_score: int | None
    completion_mass: int
    terminal_shanten_mass: int | None
    terminal_shanten_counts: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.action, DiscardAction):
            raise TypeError("action must be a DiscardAction")
        if type(self.baseline_eligible) is not bool:
            raise TypeError("baseline_eligible must be a bool")
        if type(self.post_discard_shanten) is not int:
            raise TypeError("post_discard_shanten must be an int")
        if type(self.completion_mass) is not int or self.completion_mass < 0:
            raise ValueError("completion_mass must be a non-negative int")
        for field_name in (
            "full_legal_current_ukeire_count",
            "full_legal_second_step_ukeire_score",
            "baseline_eligible_current_ukeire_count",
            "baseline_eligible_second_step_ukeire_score",
            "terminal_shanten_mass",
        ):
            value = getattr(self, field_name)
            if value is not None and type(value) is not int:
                raise TypeError(f"{field_name} must be an int or None")
        if not self.baseline_eligible and (
            self.baseline_eligible_current_ukeire_count is not None
            or self.baseline_eligible_second_step_ukeire_score is not None
        ):
            raise ValueError(
                "baseline_eligible_* stage values must be None when "
                "baseline_eligible is False"
            )
        counts = self.terminal_shanten_counts
        if counts is not None:
            if type(counts) is not tuple or len(counts) != TERMINAL_SHANTEN_AXIS:
                raise ValueError(
                    "terminal_shanten_counts must be None or a tuple covering "
                    f"structural shanten 0..{TERMINAL_SHANTEN_AXIS - 1}"
                )
            if any(type(count) is not int or count < 0 for count in counts):
                raise ValueError(
                    "terminal_shanten_counts must contain non-negative int counts"
                )
        if (self.terminal_shanten_mass is None) != (counts is None):
            raise ValueError(
                "terminal_shanten_mass and terminal_shanten_counts must be both "
                "None or both present"
            )


@dataclass(frozen=True, slots=True)
class UniverseEfficiencySummary:
    """1 candidate universe（full legal / baseline eligible）のR1-R4集約値。

    `not_applicable`（`None`）と`evaluated`かつ`0`を混同しない。ukeire /
    second-step regretは、`baseline_selected_action`がそのuniverseの
    applicable stage（minimum shanten、続いてmaximum current ukeire tie）へ
    到達した場合だけ数値を持つ。
    """

    action_count: int
    best_post_discard_shanten: int
    selected_post_discard_shanten: int
    shanten_regret: int
    best_current_ukeire: int
    selected_current_ukeire: int | None
    ukeire_regret: int | None
    best_second_step_score: int | None
    selected_second_step_score: int | None
    second_step_regret: int | None
    best_completion_mass: int
    selected_completion_mass: int
    completion_regret: int
    completion_all_zero: bool

    def __post_init__(self) -> None:
        for field_name in (
            "action_count",
            "best_post_discard_shanten",
            "selected_post_discard_shanten",
            "shanten_regret",
            "best_current_ukeire",
            "best_completion_mass",
            "selected_completion_mass",
            "completion_regret",
        ):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an int")
        for field_name in (
            "selected_current_ukeire",
            "ukeire_regret",
            "best_second_step_score",
            "selected_second_step_score",
            "second_step_regret",
        ):
            value = getattr(self, field_name)
            if value is not None and type(value) is not int:
                raise TypeError(f"{field_name} must be an int or None")
        if type(self.completion_all_zero) is not bool:
            raise TypeError("completion_all_zero must be a bool")
        if self.action_count <= 0:
            raise ValueError("action_count must be positive")
        if (self.selected_current_ukeire is None) != (self.ukeire_regret is None):
            raise ValueError(
                "selected_current_ukeire and ukeire_regret must be both None or "
                "both present"
            )
        if (self.selected_second_step_score is None) != (
            self.second_step_regret is None
        ):
            raise ValueError(
                "selected_second_step_score and second_step_regret must be both "
                "None or both present"
            )


@dataclass(frozen=True, slots=True)
class TerminalProgressionSummary:
    """R5 optional expected-terminal-shanten progression regretのexact integer値。

    `expected_terminal_shanten_regret`のようなfloat表示値はここへ含めない。
    consumer側が`terminal_shanten_regret_mass / sequence_denominator`で導出
    する。
    """

    best_terminal_shanten_mass: int
    selected_terminal_shanten_mass: int
    terminal_shanten_regret_mass: int

    def __post_init__(self) -> None:
        for field_name in (
            "best_terminal_shanten_mass",
            "selected_terminal_shanten_mass",
            "terminal_shanten_regret_mass",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative int")


@dataclass(frozen=True, slots=True)
class MechanismRiichiDefenseOffensiveEfficiencyAnalysis:
    """1回の`analyze_mechanism_riichi_defense_offensive_efficiency()`呼び出しの
    typed immutable結果。

    production `MechanismRiichiDefenseYakuhaiCallPolicy.choose_action()`の
    selection behaviorをこの型は一切変更しない。`baseline_selected_action`は
    常にexact production selectionと一致する。
    """

    legal_discard_actions: tuple[DiscardAction, ...]
    baseline_eligible_actions: tuple[DiscardAction, ...]
    baseline_selected_action: DiscardAction
    branch: OffensiveEfficiencyBranch
    horizon: int
    hidden_tile_count: int
    sequence_denominator: int
    include_terminal_progression: bool
    candidate_evaluations: tuple[OffensiveEfficiencyCandidateEvaluation, ...]
    full_legal_summary: UniverseEfficiencySummary
    baseline_eligible_summary: UniverseEfficiencySummary
    full_legal_terminal_progression_summary: TerminalProgressionSummary | None
    baseline_eligible_terminal_progression_summary: TerminalProgressionSummary | None

    def __post_init__(self) -> None:
        if not isinstance(self.branch, OffensiveEfficiencyBranch):
            raise TypeError("branch must be an OffensiveEfficiencyBranch")
        for field_name in ("horizon", "hidden_tile_count", "sequence_denominator"):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an int")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.sequence_denominator <= 0:
            raise ValueError("sequence_denominator must be positive")
        if type(self.include_terminal_progression) is not bool:
            raise TypeError("include_terminal_progression must be a bool")
        if not isinstance(self.baseline_selected_action, DiscardAction):
            raise TypeError("baseline_selected_action must be a DiscardAction")

        try:
            legal_actions = tuple(self.legal_discard_actions)
            eligible_actions = tuple(self.baseline_eligible_actions)
            candidates = tuple(self.candidate_evaluations)
        except TypeError:
            raise TypeError(
                "legal_discard_actions, baseline_eligible_actions, and "
                "candidate_evaluations must be iterable"
            ) from None
        if not legal_actions:
            raise ValueError("legal_discard_actions must not be empty")
        if not eligible_actions:
            raise ValueError("baseline_eligible_actions must not be empty")
        if any(action not in legal_actions for action in eligible_actions):
            raise ValueError(
                "baseline_eligible_actions must be a subset of legal_discard_actions"
            )
        if self.baseline_selected_action not in eligible_actions:
            raise ValueError(
                "baseline_selected_action must be a member of baseline_eligible_actions"
            )
        if any(
            not isinstance(candidate, OffensiveEfficiencyCandidateEvaluation)
            for candidate in candidates
        ):
            raise TypeError(
                "candidate_evaluations must contain only "
                "OffensiveEfficiencyCandidateEvaluation values"
            )
        if not isinstance(self.full_legal_summary, UniverseEfficiencySummary):
            raise TypeError("full_legal_summary must be a UniverseEfficiencySummary")
        if not isinstance(self.baseline_eligible_summary, UniverseEfficiencySummary):
            raise TypeError(
                "baseline_eligible_summary must be a UniverseEfficiencySummary"
            )
        for field_name, summary in (
            (
                "full_legal_terminal_progression_summary",
                self.full_legal_terminal_progression_summary,
            ),
            (
                "baseline_eligible_terminal_progression_summary",
                self.baseline_eligible_terminal_progression_summary,
            ),
        ):
            if summary is not None and not isinstance(
                summary, TerminalProgressionSummary
            ):
                raise TypeError(
                    f"{field_name} must be None or a TerminalProgressionSummary"
                )

        full_legal_all_zero = self.full_legal_summary.completion_all_zero
        baseline_eligible_all_zero = self.baseline_eligible_summary.completion_all_zero

        if not self.include_terminal_progression:
            if self.full_legal_terminal_progression_summary is not None:
                raise ValueError(
                    "full_legal_terminal_progression_summary must be None unless "
                    "include_terminal_progression is True"
                )
            if self.baseline_eligible_terminal_progression_summary is not None:
                raise ValueError(
                    "baseline_eligible_terminal_progression_summary must be None "
                    "unless include_terminal_progression is True"
                )
            if any(
                candidate.terminal_shanten_mass is not None for candidate in candidates
            ):
                raise ValueError(
                    "terminal_shanten_mass must be None on every candidate unless "
                    "R5 is opted in and activated"
                )
        else:
            if full_legal_all_zero:
                if self.full_legal_terminal_progression_summary is None:
                    raise ValueError(
                        "full_legal_terminal_progression_summary must be present "
                        "when include_terminal_progression is True and "
                        "full_legal_summary.completion_all_zero is True"
                    )
            elif self.full_legal_terminal_progression_summary is not None:
                raise ValueError(
                    "full_legal_terminal_progression_summary must be None unless "
                    "full_legal_summary.completion_all_zero is True"
                )

            if baseline_eligible_all_zero:
                if self.baseline_eligible_terminal_progression_summary is None:
                    raise ValueError(
                        "baseline_eligible_terminal_progression_summary must be "
                        "present when include_terminal_progression is True and "
                        "baseline_eligible_summary.completion_all_zero is True"
                    )
            elif self.baseline_eligible_terminal_progression_summary is not None:
                raise ValueError(
                    "baseline_eligible_terminal_progression_summary must be None "
                    "unless baseline_eligible_summary.completion_all_zero is True"
                )

            if (
                not full_legal_all_zero
                and not baseline_eligible_all_zero
                and any(
                    candidate.terminal_shanten_mass is not None
                    for candidate in candidates
                )
            ):
                raise ValueError(
                    "terminal_shanten_mass must be None on every candidate when "
                    "neither universe is completion all-zero"
                )

        object.__setattr__(self, "legal_discard_actions", legal_actions)
        object.__setattr__(self, "baseline_eligible_actions", eligible_actions)
        object.__setattr__(self, "candidate_evaluations", candidates)


def _classify_branch(
    policy_input: PolicyInput,
    legal_discard_actions: tuple[DiscardAction, ...],
) -> tuple[OffensiveEfficiencyBranch, tuple[DiscardAction, ...]]:
    """production filtering single-source helperだけからbranchとeligible setを得る。

    set size比較でactivationを推測しない。mechanism defenseがactivateすれば
    （danger tieでcandidate setが縮まらなくても）常に`MECHANISM_DEFENSE_FILTERED`
    とし、それ以外は親Policyの`_evaluate_defense_filter()`結果をそのまま使う。
    """
    mechanism_evaluation = _evaluate_mechanism_defense_filter(
        policy_input, legal_discard_actions
    )
    if mechanism_evaluation.activation is MechanismDefenseActivation.ACTIVATED:
        return (
            OffensiveEfficiencyBranch.MECHANISM_DEFENSE_FILTERED,
            mechanism_evaluation.eligible_actions,
        )

    defense_evaluation = _evaluate_defense_filter(
        policy_input, mechanism_evaluation.eligible_actions
    )
    return (
        _DEFENSE_BRANCH_TO_OFFENSIVE_EFFICIENCY_BRANCH[defense_evaluation.branch],
        defense_evaluation.eligible_actions,
    )


def _universe_stage_snapshots(
    policy_input: PolicyInput,
    universe_actions: tuple[DiscardAction, ...],
    evaluator: _DecisionShantenEvaluator,
    known_counts: Mapping[TileType, int],
) -> tuple[TwoStepUkeireCandidateEvaluation, ...]:
    """既存TwoStep semanticsを1つのuniverseへ独立実行し、stage-correctなsnapshotを返す。

    universeごとに新しい`_DiscardCandidateWork`集合を作る。full-legal universe
    とbaseline-eligible universeの評価結果を同じmutable working objectで共有
    すると、後段の`_evaluate_and_choose_prepared()`呼び出しが先行universeの
    結果を上書きしてしまうため、これを避ける。
    """
    worklist = _evaluate_post_discard_hands(policy_input, universe_actions, evaluator)
    _, snapshots = _evaluate_and_choose_prepared(
        policy_input, worklist, evaluator, known_counts=known_counts
    )
    return snapshots


def _summarize_universe(
    snapshots: tuple[TwoStepUkeireCandidateEvaluation, ...],
    completion_evaluations: tuple[FiniteHorizonCandidateEvaluation, ...],
    selected_action: DiscardAction,
) -> UniverseEfficiencySummary:
    """R1-R4の`best` / `selected` / `regret`を1つのuniverseについて集約する。"""
    stage_by_action = {snapshot.action: snapshot for snapshot in snapshots}
    selected_stage = stage_by_action[selected_action]

    best_shanten = min(snapshot.post_discard_shanten for snapshot in snapshots)
    selected_shanten = selected_stage.post_discard_shanten

    ukeire_values = tuple(
        snapshot.current_ukeire_count
        for snapshot in snapshots
        if snapshot.current_ukeire_count is not None
    )
    best_ukeire = max(ukeire_values)
    selected_ukeire = selected_stage.current_ukeire_count
    ukeire_regret = (
        best_ukeire - selected_ukeire if selected_ukeire is not None else None
    )

    second_step_values = tuple(
        snapshot.second_step_ukeire_score
        for snapshot in snapshots
        if snapshot.second_step_ukeire_score is not None
    )
    best_second_step = max(second_step_values) if second_step_values else None
    selected_second_step = selected_stage.second_step_ukeire_score
    second_step_regret = (
        best_second_step - selected_second_step
        if selected_second_step is not None and best_second_step is not None
        else None
    )

    completion_by_action = {
        evaluation.action: evaluation.completion_mass
        for evaluation in completion_evaluations
    }
    best_completion = max(completion_by_action.values())
    selected_completion = completion_by_action[selected_action]

    return UniverseEfficiencySummary(
        action_count=len(snapshots),
        best_post_discard_shanten=best_shanten,
        selected_post_discard_shanten=selected_shanten,
        shanten_regret=selected_shanten - best_shanten,
        best_current_ukeire=best_ukeire,
        selected_current_ukeire=selected_ukeire,
        ukeire_regret=ukeire_regret,
        best_second_step_score=best_second_step,
        selected_second_step_score=selected_second_step,
        second_step_regret=second_step_regret,
        best_completion_mass=best_completion,
        selected_completion_mass=selected_completion,
        completion_regret=best_completion - selected_completion,
        completion_all_zero=best_completion == 0,
    )


def _summarize_terminal_progression(
    terminal_by_action: Mapping[DiscardAction, tuple[int, tuple[int, ...]]],
    universe_actions: tuple[DiscardAction, ...],
    selected_action: DiscardAction,
) -> TerminalProgressionSummary:
    """R5 terminal-shanten massの`best` / `selected` / `regret`を1 universeについて集約する。"""
    masses = tuple(terminal_by_action[action][0] for action in universe_actions)
    best_mass = min(masses)
    selected_mass = terminal_by_action[selected_action][0]
    return TerminalProgressionSummary(
        best_terminal_shanten_mass=best_mass,
        selected_terminal_shanten_mass=selected_mass,
        terminal_shanten_regret_mass=selected_mass - best_mass,
    )


def analyze_mechanism_riichi_defense_offensive_efficiency(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
    *,
    include_terminal_progression: bool = False,
) -> MechanismRiichiDefenseOffensiveEfficiencyAnalysis:
    """1 discard decisionについて、current baseline selectionのstructural regretを観測する。

    production `MechanismRiichiDefenseYakuhaiCallPolicy.choose_action()`の
    selection behaviorはこの呼び出しによって一切変更されない。この関数を
    production decision pathから呼び出さない。

    `include_terminal_progression=True`のときだけ、Issue #169のexact adaptive
    terminal-shanten evaluatorを追加実行する。full legal universeが
    completion all-zeroならfull legal全体でDPを1回だけ実行してfull-legal /
    baseline-eligible両方のsummaryを導出し、full legalは非all-zeroだが
    baseline eligible universeだけall-zeroならbaseline eligible universeだけ
    DPを実行する。どちらもall-zeroでなければ、あるいはopt-inしていなければ、
    R5関連の重い評価を一切起動しない。
    """
    legal_discard_actions = tuple(discard_actions)
    if not legal_discard_actions:
        raise MechanismRiichiDefenseOffensiveEfficiencyDiagnosticError(
            "discard_actions must not be empty"
        )

    branch, baseline_eligible_actions = _classify_branch(
        policy_input, legal_discard_actions
    )

    mechanism_eligible_actions = _mechanism_defense_eligible_actions(
        policy_input, legal_discard_actions
    )
    baseline_selected_action = _parent_offensive_evaluate_and_choose_discard(
        policy_input, mechanism_eligible_actions
    )

    baseline_eligible_set = frozenset(baseline_eligible_actions)

    # R1: shanten evaluatorとworklistはfull legal universeで1回だけ評価し、
    # baseline-eligible universeはmembership filterで再利用する。
    shanten_evaluator = _DecisionShantenEvaluator()
    known_counts = _known_tile_counts(policy_input)

    full_legal_snapshots = _universe_stage_snapshots(
        policy_input, legal_discard_actions, shanten_evaluator, known_counts
    )
    baseline_eligible_snapshots = _universe_stage_snapshots(
        policy_input, baseline_eligible_actions, shanten_evaluator, known_counts
    )
    full_legal_stage_by_action = {
        snapshot.action: snapshot for snapshot in full_legal_snapshots
    }
    baseline_eligible_stage_by_action = {
        snapshot.action: snapshot for snapshot in baseline_eligible_snapshots
    }

    # R4: completion massはfull legal universeで1回だけexact DPを実行し、
    # baseline-eligible universeの値は同じevaluationをfilterして再利用する。
    remaining_counts = _root_remaining_counts(policy_input)
    hidden_tile_count = sum(remaining_counts)
    sequence_denominator = _falling_factorial(hidden_tile_count, DEFAULT_HORIZON)
    completion_evaluator = _FiniteHorizonEvaluator()
    full_legal_completion_evaluations = _evaluate_completion_masses(
        policy_input,
        legal_discard_actions,
        remaining_counts,
        DEFAULT_HORIZON,
        completion_evaluator,
    )
    baseline_eligible_completion_evaluations = tuple(
        evaluation
        for evaluation in full_legal_completion_evaluations
        if evaluation.action in baseline_eligible_set
    )
    completion_by_action = {
        evaluation.action: evaluation.completion_mass
        for evaluation in full_legal_completion_evaluations
    }

    full_legal_summary = _summarize_universe(
        full_legal_snapshots,
        full_legal_completion_evaluations,
        baseline_selected_action,
    )
    baseline_eligible_summary = _summarize_universe(
        baseline_eligible_snapshots,
        baseline_eligible_completion_evaluations,
        baseline_selected_action,
    )

    # R5: explicit opt-inのときだけ、full-legal / baseline-eligible universeを
    # 分離してIssue #169のexact evaluatorを適用する。baseline_eligible_actionsは
    # full legalのsubsetなので、full legal universeがall-zeroならbaseline
    # eligible universeも必ずall-zeroである。したがってfull legalがall-zeroの
    # ときはfull legal action全体で1回だけDPを実行し、baseline-eligible側は
    # そのcandidate結果をsubset filterして再利用する（DPを再実行しない）。
    terminal_by_action: dict[DiscardAction, tuple[int, tuple[int, ...]]] = {}
    full_legal_terminal_progression_summary: TerminalProgressionSummary | None = None
    baseline_eligible_terminal_progression_summary: (
        TerminalProgressionSummary | None
    ) = None
    if include_terminal_progression:
        if full_legal_summary.completion_all_zero:
            progression_candidates = _evaluate_progression_candidates(
                policy_input,
                full_legal_completion_evaluations,
                remaining_counts,
                DEFAULT_HORIZON,
                _TerminalShantenProgressionEvaluator(),
            )
            terminal_by_action = {
                candidate.action: (
                    candidate.terminal_shanten_mass,
                    candidate.terminal_shanten_counts,
                )
                for candidate in progression_candidates
            }
            full_legal_terminal_progression_summary = _summarize_terminal_progression(
                terminal_by_action, legal_discard_actions, baseline_selected_action
            )
            baseline_eligible_terminal_progression_summary = (
                _summarize_terminal_progression(
                    terminal_by_action,
                    baseline_eligible_actions,
                    baseline_selected_action,
                )
            )
        elif baseline_eligible_summary.completion_all_zero:
            progression_candidates = _evaluate_progression_candidates(
                policy_input,
                baseline_eligible_completion_evaluations,
                remaining_counts,
                DEFAULT_HORIZON,
                _TerminalShantenProgressionEvaluator(),
            )
            terminal_by_action = {
                candidate.action: (
                    candidate.terminal_shanten_mass,
                    candidate.terminal_shanten_counts,
                )
                for candidate in progression_candidates
            }
            baseline_eligible_terminal_progression_summary = (
                _summarize_terminal_progression(
                    terminal_by_action,
                    baseline_eligible_actions,
                    baseline_selected_action,
                )
            )

    candidate_evaluations = tuple(
        OffensiveEfficiencyCandidateEvaluation(
            action=action,
            baseline_eligible=action in baseline_eligible_set,
            post_discard_shanten=full_legal_stage_by_action[
                action
            ].post_discard_shanten,
            full_legal_current_ukeire_count=(
                full_legal_stage_by_action[action].current_ukeire_count
            ),
            full_legal_second_step_ukeire_score=(
                full_legal_stage_by_action[action].second_step_ukeire_score
            ),
            baseline_eligible_current_ukeire_count=(
                baseline_eligible_stage_by_action[action].current_ukeire_count
                if action in baseline_eligible_set
                else None
            ),
            baseline_eligible_second_step_ukeire_score=(
                baseline_eligible_stage_by_action[action].second_step_ukeire_score
                if action in baseline_eligible_set
                else None
            ),
            completion_mass=completion_by_action[action],
            terminal_shanten_mass=(
                terminal_by_action[action][0] if action in terminal_by_action else None
            ),
            terminal_shanten_counts=(
                terminal_by_action[action][1] if action in terminal_by_action else None
            ),
        )
        for action in sorted(legal_discard_actions, key=_discard_action_sort_key)
    )

    return MechanismRiichiDefenseOffensiveEfficiencyAnalysis(
        legal_discard_actions=legal_discard_actions,
        baseline_eligible_actions=baseline_eligible_actions,
        baseline_selected_action=baseline_selected_action,
        branch=branch,
        horizon=DEFAULT_HORIZON,
        hidden_tile_count=hidden_tile_count,
        sequence_denominator=sequence_denominator,
        include_terminal_progression=include_terminal_progression,
        candidate_evaluations=candidate_evaluations,
        full_legal_summary=full_legal_summary,
        baseline_eligible_summary=baseline_eligible_summary,
        full_legal_terminal_progression_summary=full_legal_terminal_progression_summary,
        baseline_eligible_terminal_progression_summary=(
            baseline_eligible_terminal_progression_summary
        ),
    )
