"""Championの同点数牌打牌だけを公開paijiaで選び直す候補Policy（lisbun/lisjong#330）。

仕様は`docs/placement-aware-speed-call-suited-paijia-tie.md`。未昇格の実験候補で、
現Champion `PlacementAwareSpeedCallPolicy`は書き換えない。Arenaで使うidentity
（案: `placement-aware-speed-call-suited-paijia-tie-v1`、設定は空object）は
登録する側が明示する。

Championの打牌判断を1回だけ実行し、次をすべて満たす通常打牌だけ、その選択を
`kobalab_0004_discard._PublicCounts.paijia()`の最小へ置き換える。

1. 通常自摸後（`drawn_tile`あり）、自席に副露・暗槓がない、自席未リーチ、他家リーチなし
2. 南4局以降ではなく、Championの候補制限後の`_classify_branch()`がPUSH
3. FiniteHorizon completion massが全候補0で、限定R5を実行していない
4. HandValueAwareのstaged評価がsecond-stepまで到達し（向聴0は到達しない）、
   second-step最大の最終tieが2候補以上
5. Championの選択牌が数牌で、最終tieのうち数牌の打牌に異なる牌（牌種または赤/通常）が
   2つ以上ある
6. その数牌の打牌でpaijiaが最小のものを選ぶ。Championの打牌も最小なら維持し、
   それ以外の同点は`discard_action_sort_key`で決める

最終tieはHandValueAwareが実際に計算したsnapshotから読む。second-step値は
second-step段階まで残った候補にだけ計算されるので、`None`でない候補の最大値の
集合が最終tieである。rankingを再実行・複製しない。

和了・立直・鳴きの判断と、上の条件を満たさない打牌判断はChampionと同じactionを返す。
打牌判断の`analysis`は常にこの候補の`SuitedPaijiaTieAnalysis`で、Championの
`TargetedHonorReleaseAnalysis`は流用しない。Championの例外（入力不整合、
未見枚数が探索幅未満等）はそのまま送出する。

PolicyInput以外の状態やhidden informationは使用せず、Policy instanceに
cross-decision stateを保持しない。
"""

from dataclasses import dataclass
from enum import Enum, auto

from lisjong.policies.genbutsu_defense_two_step_ukeire import _opponent_riichi_players
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    HandValueCandidateEvaluation,
)
from lisjong.policies.kobalab_0004_discard import _PublicCounts
from lisjong.policies.placement_aware_speed_call import (
    PlacementAwareSpeedCallPolicy,
    _eligible_discard_actions,
    _is_all_last,
)
from lisjong.policies.targeted_honor_release_terminal_progression import (
    TargetedHonorReleaseActivationStage,
    TargetedHonorReleaseBranch,
    _evaluate_and_choose_discard_with_hva,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.tile import TileCategory
from lisjong.structural_efficiency import discard_action_sort_key

_NO_R5_ALL_ZERO_STAGES = frozenset(
    {
        TargetedHonorReleaseActivationStage.PARENT_SHANTEN_TOO_CLOSE,
        TargetedHonorReleaseActivationStage.PARENT_NOT_MINIMUM_SHANTEN,
        TargetedHonorReleaseActivationStage.PARENT_NOT_MAXIMUM_UKEIRE,
        TargetedHonorReleaseActivationStage.PARENT_ALREADY_HONOR,
        TargetedHonorReleaseActivationStage.NO_QUALIFYING_HONOR_PEER,
    }
)
"""all-zero PUSH pathのうち、限定R5を実行せずにChampionの選択が決まったstage。"""

_R5_EVALUATED_STAGES = frozenset(
    {
        TargetedHonorReleaseActivationStage.R5_PARENT_BEST,
        TargetedHonorReleaseActivationStage.R5_NON_HONOR_ONLY_BEST,
        TargetedHonorReleaseActivationStage.R5_HONOR_ONLY_SWITCH,
    }
)


class SuitedPaijiaTieStage(Enum):
    """1打牌判断がどのgateでChampion維持になったか、または選び直したか。"""

    NOT_NORMAL_TURN = auto()
    HAS_OWN_MELD = auto()
    SELF_RIICHI = auto()
    OPPONENT_RIICHI = auto()
    ALL_LAST = auto()
    NON_PUSH_BRANCH = auto()
    POSITIVE_COMPLETION = auto()
    R5_EVALUATED = auto()
    NO_FINAL_TIE = auto()
    CHAMPION_HONOR = auto()
    SINGLE_SUITED_OPTION = auto()
    CHAMPION_MINIMUM_PAIJIA = auto()
    PAIJIA_SWITCH = auto()


_ACTIVATED_STAGES = frozenset(
    {SuitedPaijiaTieStage.CHAMPION_MINIMUM_PAIJIA, SuitedPaijiaTieStage.PAIJIA_SWITCH}
)


@dataclass(frozen=True, slots=True)
class SuitedPaijiaCandidateEvaluation:
    """最終tieの数牌打牌1つについて、この候補が使ったpaijia。"""

    action: DiscardAction
    paijia: int

    def __post_init__(self) -> None:
        if not isinstance(self.action, DiscardAction):
            raise TypeError("action must be a DiscardAction")
        if type(self.paijia) is not int:
            raise TypeError("paijia must be an int")


@dataclass(frozen=True, slots=True)
class SuitedPaijiaTieAnalysis(AnalysisTrace):
    """#330候補の1打牌判断のpurpose-specific analysis。

    `paijia_evaluations`はpaijiaを評価した判断（`CHAMPION_MINIMUM_PAIJIA` /
    `PAIJIA_SWITCH`）だけが持ち、最終tieの数牌打牌を`discard_action_sort_key`順に並べる。
    """

    stage: SuitedPaijiaTieStage
    champion_action: DiscardAction
    selected_action: DiscardAction
    action_changed: bool
    paijia_evaluations: tuple[SuitedPaijiaCandidateEvaluation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage, SuitedPaijiaTieStage):
            raise TypeError("stage must be a SuitedPaijiaTieStage")
        if not isinstance(self.champion_action, DiscardAction):
            raise TypeError("champion_action must be a DiscardAction")
        if not isinstance(self.selected_action, DiscardAction):
            raise TypeError("selected_action must be a DiscardAction")
        if type(self.action_changed) is not bool:
            raise TypeError("action_changed must be a bool")
        if self.action_changed != (self.champion_action != self.selected_action):
            raise ValueError(
                "action_changed must describe selected_action against champion_action"
            )
        if self.action_changed != (self.stage is SuitedPaijiaTieStage.PAIJIA_SWITCH):
            raise ValueError("only PAIJIA_SWITCH may change the Champion action")
        if type(self.paijia_evaluations) is not tuple:
            raise TypeError("paijia_evaluations must be a tuple")
        if any(
            not isinstance(evaluation, SuitedPaijiaCandidateEvaluation)
            for evaluation in self.paijia_evaluations
        ):
            raise TypeError(
                "paijia_evaluations must contain only "
                "SuitedPaijiaCandidateEvaluation values"
            )
        if (self.stage in _ACTIVATED_STAGES) != bool(self.paijia_evaluations):
            raise ValueError(
                "paijia_evaluations must be present exactly when paijia was evaluated"
            )


def _final_tie(
    snapshots: tuple[HandValueCandidateEvaluation, ...],
) -> tuple[HandValueCandidateEvaluation, ...]:
    """HVAが最後まで区別できなかった候補（second-step最大の集合）を返す。

    HVAはsecond-stepを、向聴・受入・retained real value・yaku routeの各最大を
    通過し、さらに向聴0でない場合にだけ計算する。second-step値を持つ候補の最大
    集合がそのまま最終tieで、1候補以下なら空を返す。
    """
    scored = tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.second_step_ukeire_score is not None
    )
    if not scored:
        return ()
    maximum = max(snapshot.second_step_ukeire_score for snapshot in scored)
    finalists = tuple(
        snapshot for snapshot in scored if snapshot.second_step_ukeire_score == maximum
    )
    return finalists if len(finalists) >= 2 else ()


def _pre_evaluation_stage(policy_input: PolicyInput) -> SuitedPaijiaTieStage | None:
    """Championの打牌評価結果に依存しないgate（条件1・2の局面条件）。"""
    if policy_input.own_hand.drawn_tile is None:
        return SuitedPaijiaTieStage.NOT_NORMAL_TURN
    own_player = policy_input.players[int(policy_input.self_seat)]
    if own_player.melds:
        return SuitedPaijiaTieStage.HAS_OWN_MELD
    if own_player.riichi is not RiichiState.NONE:
        return SuitedPaijiaTieStage.SELF_RIICHI
    if _opponent_riichi_players(policy_input):
        return SuitedPaijiaTieStage.OPPONENT_RIICHI
    if _is_all_last(policy_input):
        return SuitedPaijiaTieStage.ALL_LAST
    return None


def _evaluate_and_choose_discard(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, SuitedPaijiaTieAnalysis]:
    """Championの打牌判断を1回実行し、条件を満たす場合だけpaijiaで選び直す。"""
    eligible_actions = _eligible_discard_actions(policy_input, discard_actions)
    champion_action, hva_snapshots, champion_analysis = (
        _evaluate_and_choose_discard_with_hva(policy_input, eligible_actions)
    )

    def keep(
        stage: SuitedPaijiaTieStage,
    ) -> tuple[DiscardAction, SuitedPaijiaTieAnalysis]:
        return champion_action, SuitedPaijiaTieAnalysis(
            stage=stage,
            champion_action=champion_action,
            selected_action=champion_action,
            action_changed=False,
            paijia_evaluations=(),
        )

    pre_stage = _pre_evaluation_stage(policy_input)
    if pre_stage is not None:
        return keep(pre_stage)
    if champion_analysis.branch is not TargetedHonorReleaseBranch.PUSH:
        return keep(SuitedPaijiaTieStage.NON_PUSH_BRANCH)
    champion_stage = champion_analysis.activation_stage
    if champion_stage is TargetedHonorReleaseActivationStage.POSITIVE_COMPLETION:
        return keep(SuitedPaijiaTieStage.POSITIVE_COMPLETION)
    if champion_stage in _R5_EVALUATED_STAGES:
        return keep(SuitedPaijiaTieStage.R5_EVALUATED)
    if champion_stage not in _NO_R5_ALL_ZERO_STAGES or hva_snapshots is None:
        # 上のgateを通過したclosed PUSHの通常自摸は、必ずall-zero pathへ進む。
        raise AssertionError(
            f"unexpected Champion stage for a closed PUSH turn: {champion_stage}"
        )

    final_tie = _final_tie(hva_snapshots)
    if champion_action not in {snapshot.action for snapshot in final_tie}:
        return keep(SuitedPaijiaTieStage.NO_FINAL_TIE)
    if champion_action.tile.tile_type.category is TileCategory.HONOR:
        return keep(SuitedPaijiaTieStage.CHAMPION_HONOR)
    suited_actions = tuple(
        sorted(
            (
                snapshot.action
                for snapshot in final_tie
                if snapshot.action.tile.tile_type.category is not TileCategory.HONOR
            ),
            key=discard_action_sort_key,
        )
    )
    if len({action.tile for action in suited_actions}) < 2:
        return keep(SuitedPaijiaTieStage.SINGLE_SUITED_OPTION)

    counts = _PublicCounts(policy_input)
    evaluations = tuple(
        SuitedPaijiaCandidateEvaluation(
            action=action, paijia=counts.paijia(action.tile)
        )
        for action in suited_actions
    )
    minimum = min(evaluation.paijia for evaluation in evaluations)
    paijia_by_action = {
        evaluation.action: evaluation.paijia for evaluation in evaluations
    }
    if paijia_by_action[champion_action] == minimum:
        selected_action = champion_action
        stage = SuitedPaijiaTieStage.CHAMPION_MINIMUM_PAIJIA
    else:
        selected_action = min(
            (
                evaluation.action
                for evaluation in evaluations
                if evaluation.paijia == minimum
            ),
            key=discard_action_sort_key,
        )
        stage = SuitedPaijiaTieStage.PAIJIA_SWITCH
    return selected_action, SuitedPaijiaTieAnalysis(
        stage=stage,
        champion_action=champion_action,
        selected_action=selected_action,
        action_changed=selected_action != champion_action,
        paijia_evaluations=evaluations,
    )


class PlacementAwareSpeedCallSuitedPaijiaTiePolicy(PlacementAwareSpeedCallPolicy):
    """#330: Championの同点数牌打牌だけを公開paijiaで選び直す（未昇格の実験候補）。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        selected, analysis = _evaluate_and_choose_discard(policy_input, discard_actions)
        return PolicyDecision(action=selected, analysis=analysis)


__all__ = [
    "PlacementAwareSpeedCallSuitedPaijiaTiePolicy",
    "SuitedPaijiaCandidateEvaluation",
    "SuitedPaijiaTieAnalysis",
    "SuitedPaijiaTieStage",
]
