"""公開済みドラ・赤ドラ保持を打牌比較へ追加するTwoStepUkeire派生Policy。

Issue #107により、`TwoStepUkeirePolicy`の

    打牌後向聴数 > 現在受け入れ > 2段階受け入れ > stable tie-break

を変更せずoffense baselineとして維持しつつ、`ValueAwareTwoStepUkeirePolicy`という
新しいPolicy世代を追加する。selection semanticsは次のとおりである。

    打牌後向聴数
    > 現在受け入れ
    > 打牌後concealed handに残る retained_concealed_dora_count
    > 2段階受け入れ
    > stable tie-break

`retained_concealed_dora_count`は、完成手の実際のhan数、期待得点、winning value、
expected valueのいずれでもない。打牌候補ごとに、打牌後concealed handに残る

    公開済みdora indicator由来のdora count
    +
    赤ドラcount

だけを数えるcandidate-dependent featureである。副露済みのドラ・赤ドラ等、全candidate
に共通する定数分はこの値へ含めない。

## dora mappingの責務境界

`lisjong-engine`には得点評価用の同等semanticが存在するが、`lisjong`から
`lisjong-engine`へのruntime dependencyは導入しない。本moduleが持つ
`_dora_tile_type()`は、Policy-visibleな公開indicatorからcandidate featureを導出する
ための最小限のpure / deterministic helperであり、engineの得点評価を再実装しない。

## selection stagingとsecond-step semantics

Stage構成は既存TwoStepと同じ「不要なstageを計算しない」原則に従う。

    Stage 1 (全candidate)        -> post_discard_shanten
    Stage 2 (最小shanten候補)     -> current_ukeire_count
    Stage 3 (最大現在受け入れ候補) -> retained_concealed_dora_count
    Stage 4 (最大dora候補;
             minimum_shanten > 0 のときのみ) -> second_step_ukeire_score
    Stage 5                      -> stable tie-break

前段がすでに候補を1件へ絞った場合、既存TwoStepと同様に後続stageを評価しない
（`None`のまま残す）。`minimum_shanten == 0`でもdora countは評価する。

value-aware化は**現在decisionのreal legal discard比較だけ**に限定し、既存TwoStepの
第2段仮想branch（`_best_next_ukeire()`等のhypothetical future draw / discard選択）
へdora valueを伝播しない。

## subclass構造

`TwoStepUkeirePolicy._decide_discard()` extension pointだけをoverrideし、
`_decide()` / `choose_action()` / `choose_action_with_analysis()`および
winning action / Always Riichi / pass / fallback handlingは基底classから継承する。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from lisjong.policies.two_step_ukeire import (
    TwoStepUkeirePolicy,
    _DecisionShantenEvaluator,
    _discard_action_sort_key,
    _known_tile_counts,
    _remove_one_matching_tile,
    _second_step_score,
    _ukeire_count,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

_WIND_RANKS = 4
_DRAGON_START_RANK = 5
_DRAGON_RANKS = 3
_SUITED_MAXIMUM_RANK = 9


def _dora_tile_type(indicator_tile_type: TileType) -> TileType:
    """1枚のdora indicatorが示す実ドラの`TileType`を返す。

    赤5 indicatorも通常5 indicatorと同じ基礎牌種として扱うため、この関数は
    `TileType`（赤牌区分を含まない）だけを引数に取る。

        数牌: 1 -> 2 -> ... -> 9 -> 1
        風牌: 東 -> 南 -> 西 -> 北 -> 東
        三元牌: 白 -> 發 -> 中 -> 白
    """
    if indicator_tile_type.category is TileCategory.HONOR:
        if indicator_tile_type.rank <= _WIND_RANKS:
            next_rank = indicator_tile_type.rank % _WIND_RANKS + 1
        else:
            offset = indicator_tile_type.rank - _DRAGON_START_RANK
            next_rank = (offset + 1) % _DRAGON_RANKS + _DRAGON_START_RANK
        return TileType(TileCategory.HONOR, next_rank)

    next_rank = indicator_tile_type.rank % _SUITED_MAXIMUM_RANK + 1
    return TileType(indicator_tile_type.category, next_rank)


def _retained_concealed_dora_count(
    post_discard_hand: Sequence[Tile], dora_indicators: Sequence[Tile]
) -> int:
    """打牌後concealed handに残る、赤ドラ + indicator-derived doraの合計を返す。

    各公開indicatorを独立に扱うため、同じ実ドラを示すindicatorが複数あれば、
    それを保持する牌1枚につきindicatorの枚数分だけ加算される
    (multiplicityを保持する)。1枚の赤5が同時にindicator-derived doraでもある
    場合、赤ドラ分とindicator-derived dora分をそれぞれ独立に加算するため
    合計は2になる。
    """
    dora_tile_types = tuple(
        _dora_tile_type(indicator.tile_type) for indicator in dora_indicators
    )
    count = 0
    for tile in post_discard_hand:
        if tile.is_red:
            count += 1
        count += sum(
            1 for dora_tile_type in dora_tile_types if tile.tile_type == dora_tile_type
        )
    return count


@dataclass(frozen=True, slots=True)
class ValueAwareTwoStepUkeireCandidateEvaluation:
    """ValueAwareTwoStepUkeireが実際に使用した1打牌候補のsemantic評価値。"""

    action: DiscardAction
    post_discard_shanten: int
    current_ukeire_count: int | None
    retained_concealed_dora_count: int | None
    second_step_ukeire_score: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.action, DiscardAction):
            raise TypeError("action must be a DiscardAction")
        if type(self.post_discard_shanten) is not int:
            raise TypeError("post_discard_shanten must be an int")
        if self.current_ukeire_count is not None and (
            type(self.current_ukeire_count) is not int
        ):
            raise TypeError("current_ukeire_count must be an int or None")
        if self.retained_concealed_dora_count is not None and (
            type(self.retained_concealed_dora_count) is not int
        ):
            raise TypeError("retained_concealed_dora_count must be an int or None")
        if self.second_step_ukeire_score is not None and (
            type(self.second_step_ukeire_score) is not int
        ):
            raise TypeError("second_step_ukeire_score must be an int or None")


@dataclass(frozen=True, slots=True)
class ValueAwareTwoStepUkeireAnalysis(AnalysisTrace):
    """ValueAwareTwoStepUkeireが実際に実行した打牌評価のtyped observation payload。

    `candidate_evaluations`は、その1 decisionで実際に生成された
    `ValueAwareTwoStepUkeireCandidateEvaluation`をsource of truthとしてそのまま
    保持する。trace目的でshanten、現在受け入れ、dora count、2段階受け入れscoreを
    再計算しない。`None = stage未評価`と`0 = 評価済み結果0`の区別も、元のcandidate
    evaluation側のsemanticsをそのまま引き継ぐ。
    """

    candidate_evaluations: tuple[ValueAwareTwoStepUkeireCandidateEvaluation, ...]

    def __post_init__(self) -> None:
        try:
            evaluations = tuple(self.candidate_evaluations)
        except TypeError:
            raise TypeError("candidate_evaluations must be an iterable") from None

        if any(
            not isinstance(evaluation, ValueAwareTwoStepUkeireCandidateEvaluation)
            for evaluation in evaluations
        ):
            raise TypeError(
                "candidate_evaluations must contain only "
                "ValueAwareTwoStepUkeireCandidateEvaluation values"
            )

        if not evaluations:
            raise ValueError("candidate_evaluations must not be empty")

        object.__setattr__(self, "candidate_evaluations", evaluations)


@dataclass(slots=True)
class _ValueAwareDiscardCandidateWork:
    """1 decision内だけで使う、後続計算用のprivate mutable候補状態。"""

    action: DiscardAction
    post_discard_hand: list[Tile]
    post_discard_shanten: int
    current_ukeire_count: int | None = None
    retained_concealed_dora_count: int | None = None
    second_step_ukeire_score: int | None = None

    def snapshot(self) -> ValueAwareTwoStepUkeireCandidateEvaluation:
        return ValueAwareTwoStepUkeireCandidateEvaluation(
            action=self.action,
            post_discard_shanten=self.post_discard_shanten,
            current_ukeire_count=self.current_ukeire_count,
            retained_concealed_dora_count=self.retained_concealed_dora_count,
            second_step_ukeire_score=self.second_step_ukeire_score,
        )


def _evaluate_and_choose_discard(
    policy_input: PolicyInput, discard_actions: tuple[DiscardAction, ...]
) -> tuple[DiscardAction, tuple[ValueAwareTwoStepUkeireCandidateEvaluation, ...]]:
    """legal discard集合を1回のevaluation pathでstaged評価し、選択とsnapshotを返す。

    selectionとanalysisで同じ計算結果をsource of truthとして共有し、shanten、
    現在受け入れ、dora count、2段階受け入れscoreを二重計算しない。
    """
    known_counts = _known_tile_counts(policy_input)
    evaluator = _DecisionShantenEvaluator()
    concealed_tiles = policy_input.own_hand.concealed_tiles

    evaluated = tuple(
        _ValueAwareDiscardCandidateWork(
            action=action,
            post_discard_hand=remaining_hand,
            post_discard_shanten=evaluator.calculate(remaining_hand),
        )
        for action in discard_actions
        for remaining_hand in (_remove_one_matching_tile(concealed_tiles, action.tile),)
    )

    minimum_shanten = min(candidate.post_discard_shanten for candidate in evaluated)
    minimum_shanten_candidates = tuple(
        candidate
        for candidate in evaluated
        if candidate.post_discard_shanten == minimum_shanten
    )

    for candidate in minimum_shanten_candidates:
        candidate.current_ukeire_count = _ukeire_count(
            candidate.post_discard_hand,
            known_counts,
            minimum_shanten,
            evaluator,
        )
    maximum_current_ukeire = max(
        candidate.current_ukeire_count for candidate in minimum_shanten_candidates
    )
    ukeire_finalists = tuple(
        candidate
        for candidate in minimum_shanten_candidates
        if candidate.current_ukeire_count == maximum_current_ukeire
    )

    if len(ukeire_finalists) == 1:
        selected = ukeire_finalists[0].action
    else:
        dora_indicators = policy_input.round.dora_indicators
        for candidate in ukeire_finalists:
            candidate.retained_concealed_dora_count = _retained_concealed_dora_count(
                candidate.post_discard_hand, dora_indicators
            )
        maximum_dora = max(
            candidate.retained_concealed_dora_count for candidate in ukeire_finalists
        )
        dora_finalists = tuple(
            candidate
            for candidate in ukeire_finalists
            if candidate.retained_concealed_dora_count == maximum_dora
        )

        if len(dora_finalists) == 1:
            selected = dora_finalists[0].action
        elif minimum_shanten == 0:
            selected = min(
                (candidate.action for candidate in dora_finalists),
                key=_discard_action_sort_key,
            )
        else:
            for candidate in dora_finalists:
                candidate.second_step_ukeire_score = _second_step_score(
                    candidate.post_discard_hand,
                    known_counts,
                    minimum_shanten,
                    evaluator,
                )
            maximum_second_step = max(
                candidate.second_step_ukeire_score for candidate in dora_finalists
            )
            selected = min(
                (
                    candidate.action
                    for candidate in dora_finalists
                    if candidate.second_step_ukeire_score == maximum_second_step
                ),
                key=_discard_action_sort_key,
            )

    snapshots = tuple(
        candidate.snapshot()
        for candidate in sorted(
            evaluated,
            key=lambda candidate: _discard_action_sort_key(candidate.action),
        )
    )
    return selected, snapshots


class ValueAwareTwoStepUkeirePolicy(TwoStepUkeirePolicy):
    """現在受け入れとdora保持の両方を反映するTwoStepUkeire派生Policy。

    winning action、Always Riichi、pass、既存fallback handlingは
    `TwoStepUkeirePolicy._decide()`からそのまま継承し、`_decide_discard()`
    extension pointだけをoverrideする。
    """

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        selected, evaluations = _evaluate_and_choose_discard(
            policy_input, discard_actions
        )
        return PolicyDecision(
            action=selected,
            analysis=ValueAwareTwoStepUkeireAnalysis(candidate_evaluations=evaluations),
        )
