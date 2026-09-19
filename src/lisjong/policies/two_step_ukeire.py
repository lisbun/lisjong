"""同向聴・同受け入れ候補を2段階受け入れで比較するPolicy。

`TwoStepUkeirePolicy`は`UkeirePolicy`を置き換えず、次の独立した戦略世代を
実装する。

    打牌後向聴数 > 現在受け入れ > 2段階受け入れ > stable tie-break

2段目は、最小向聴かつ最大現在受け入れの候補が複数残り、打牌後向聴数が
1以上の場合だけ評価する。各第1有効牌をPolicy-visibleな未見枚数で重み付けし、
仮想ツモ後の最善の純手牌形が持つ次の受け入れを合計する。

山の内部状態、他家の実手牌、belief、未来のlegal actionは使用しない。向聴数は
公開`calculate_shanten()`だけを正本とし、赤5と通常5は仮想branchでは同じ
`TileType`として扱う。実際の最初の`DiscardAction` identityは維持する。

Issue #177により、structural discard / 牌効率semanticそのものは
`lisjong.structural_efficiency`が所有する。このmoduleが所有するのは、
そのreusable semanticを組み合わせたTwoStepUkeire固有のstaged selection
semanticsだけである。

    lisjong.structural_efficiency
        canonical discard ordering
        post-discard concealed hand / structural shanten
        known tile counting
        effective tile types / current ukeire / second-step ukeire
            ↓
    two_step_ukeire
        staged selection（`_evaluate_and_choose_prepared()`）
        `TwoStepUkeireCandidateEvaluation`
        `TwoStepUkeireAnalysis`

Issue #76により、`choose_action()`は次の優先順位でAlways Riichi baselineを
持つ。

    winning action > RiichiAction > 通常打牌評価 > pass > 既存fallback

`decision.legal_actions`に`RiichiAction`が存在することを合法性の正本とし、
面前・聴牌・持ち点等のリーチ条件はここで再計算しない。この判定は通常打牌
評価ロジック（上記2段階受け入れ）とは独立したprivate helperに置き、将来
リーチ / ダマ判断へ差し替えられる責務分離を保つ。

Issue #97により、この優先順位判定は`_decide()`という単一のdecision
orchestrationへまとめ、`choose_action()`と`choose_action_with_analysis()`は
どちらも同じ算法を**1回だけ**実行する。打牌評価branchだけが
`TwoStepUkeireAnalysis`をtyped observation payloadとして返し、和了、リーチ、
pass、既存fallbackでは`analysis`を`None`にする。trace表示のためだけに打牌
評価を追加実行しない。

analysisはIssue #87の`TwoStepUkeireCandidateEvaluation`をsource of truthとして
そのまま再利用し、shanten、現在受け入れ、2段階受け入れscoreをtrace用に
再計算しない。`policy.last_analysis`のようなdecision間mutable stateも持たない。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lisjong.policy_contract.action import (
    DiscardAction,
    InternalAction,
    PassAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import Tile, TileType, tile_sort_key
from lisjong.structural_efficiency import (
    PostDiscardStructuralEvaluation,
    StructuralEfficiencyError,
    StructuralShantenEvaluator,
    discard_action_sort_key,
    evaluate_post_discard_hands,
    known_tile_counts,
    second_step_ukeire_score,
    ukeire_count,
)

_WINNING_ACTION_TYPES = (RonAction, TsumoAction)

TwoStepUkeirePolicyError = StructuralEfficiencyError
"""入力不整合または未定義の状況をPolicyがfail closedする場合。

Issue #177でstructural efficiency semanticのownershipを
`lisjong.structural_efficiency`へ移したため、この名前は
`StructuralEfficiencyError`のcompatibility aliasである。既存のtests / callerが
観測するfail-closed behaviorとexception identityを変えないために維持する。
"""


@dataclass(frozen=True, slots=True)
class TwoStepUkeireCandidateEvaluation:
    """TwoStepUkeireが実際に使用した1打牌候補のsemantic評価値。

    Issue #87のPolicy-specific staged snapshotであり、Issue #177のreusable
    structural componentが返すlow-level valueではない。`None = stage未評価`と
    `0 = 評価済み結果0`の区別はTwoStepUkeireのevaluation pathを表す。
    """

    action: DiscardAction
    post_discard_shanten: int
    current_ukeire_count: int | None
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
        if self.second_step_ukeire_score is not None and (
            type(self.second_step_ukeire_score) is not int
        ):
            raise TypeError("second_step_ukeire_score must be an int or None")


@dataclass(frozen=True, slots=True)
class TwoStepUkeireAnalysis(AnalysisTrace):
    """TwoStepUkeireが実際に実行した打牌評価のtyped observation payload。

    `candidate_evaluations`は、その1 decisionで実際に生成された
    `TwoStepUkeireCandidateEvaluation`をsource of truthとしてそのまま保持する。
    trace目的でshanten、現在受け入れ、2段階受け入れscoreを再計算しない。
    `None = stage未評価`と`0 = 評価済み結果0`の区別も、元のcandidate
    evaluation側のsemanticsをそのまま引き継ぐ。

    このpayloadはpost-discard hand、decision-local shanten cache、mutable
    working state、Policy instance、環境runtimeへの参照を持たない。打牌評価を
    実際に行わなかったdecision（和了、リーチ、pass、既存fallback）では、
    このanalysis自体を生成しない。
    """

    candidate_evaluations: tuple[TwoStepUkeireCandidateEvaluation, ...]

    def __post_init__(self) -> None:
        try:
            evaluations = tuple(self.candidate_evaluations)
        except TypeError:
            raise TypeError("candidate_evaluations must be an iterable") from None

        if any(
            not isinstance(evaluation, TwoStepUkeireCandidateEvaluation)
            for evaluation in evaluations
        ):
            raise TypeError(
                "candidate_evaluations must contain only "
                "TwoStepUkeireCandidateEvaluation values"
            )

        if not evaluations:
            raise ValueError("candidate_evaluations must not be empty")

        object.__setattr__(self, "candidate_evaluations", evaluations)


@dataclass(slots=True)
class _DiscardCandidateWork:
    """1 decision内だけで使う、TwoStep staged評価用のprivate mutable候補状態。

    reusable structural componentのimmutable
    `PostDiscardStructuralEvaluation`から作り、TwoStep固有のstage値だけを
    後から埋める。この型をsupported contractへ露出しない。
    """

    action: DiscardAction
    post_discard_hand: tuple[Tile, ...]
    post_discard_shanten: int
    current_ukeire_count: int | None = None
    second_step_ukeire_score: int | None = None

    def snapshot(self) -> TwoStepUkeireCandidateEvaluation:
        return TwoStepUkeireCandidateEvaluation(
            action=self.action,
            post_discard_shanten=self.post_discard_shanten,
            current_ukeire_count=self.current_ukeire_count,
            second_step_ukeire_score=self.second_step_ukeire_score,
        )


def _winning_action_sort_key(action: RonAction | TsumoAction) -> tuple[object, ...]:
    if isinstance(action, RonAction):
        return (
            0,
            int(action.actor),
            int(action.target),
            tile_sort_key(action.winning_tile),
        )
    return (1, int(action.actor), tile_sort_key(action.winning_tile))


def _choose_discard(
    policy_input: PolicyInput, discard_actions: tuple[DiscardAction, ...]
) -> DiscardAction:
    selected, _ = _evaluate_and_choose_discard(policy_input, discard_actions)
    return selected


def _evaluate_and_choose_discard(
    policy_input: PolicyInput, discard_actions: tuple[DiscardAction, ...]
) -> tuple[DiscardAction, tuple[TwoStepUkeireCandidateEvaluation, ...]]:
    known_counts = known_tile_counts(policy_input)
    evaluator = StructuralShantenEvaluator()
    evaluated = evaluate_post_discard_hands(policy_input, discard_actions, evaluator)
    return _evaluate_and_choose_prepared(
        policy_input,
        evaluated,
        evaluator,
        known_counts=known_counts,
    )


def _evaluate_and_choose_prepared(
    policy_input: PolicyInput,
    evaluated: Sequence[PostDiscardStructuralEvaluation],
    evaluator: StructuralShantenEvaluator,
    *,
    known_counts: Mapping[TileType, int] | None = None,
) -> tuple[DiscardAction, tuple[TwoStepUkeireCandidateEvaluation, ...]]:
    """評価済みpost-discard候補をstagedに比較し、選択とcanonical snapshotを返す。

    これはTwoStepUkeire固有のstaged selection semanticsであり、Issue #177の
    reusable structural componentへ昇格させない。stage値を持つmutable work
    objectはこの呼び出しの内部だけで作るため、同じ
    `PostDiscardStructuralEvaluation`集合を複数のuniverseへ独立に渡しても
    先行結果に汚染されない。
    """
    if known_counts is None:
        known_counts = known_tile_counts(policy_input)
    candidates = tuple(
        _DiscardCandidateWork(
            action=evaluation.action,
            post_discard_hand=evaluation.post_discard_hand,
            post_discard_shanten=evaluation.post_discard_shanten,
        )
        for evaluation in evaluated
    )
    minimum_shanten = min(candidate.post_discard_shanten for candidate in candidates)
    minimum_shanten_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.post_discard_shanten == minimum_shanten
    )

    for candidate in minimum_shanten_candidates:
        candidate.current_ukeire_count = ukeire_count(
            candidate.post_discard_hand,
            known_counts,
            minimum_shanten,
            evaluator,
        )
    maximum_current_ukeire = max(
        candidate.current_ukeire_count for candidate in minimum_shanten_candidates
    )
    finalists = tuple(
        candidate
        for candidate in minimum_shanten_candidates
        if candidate.current_ukeire_count == maximum_current_ukeire
    )

    if len(finalists) == 1:
        selected = finalists[0].action
    elif minimum_shanten == 0:
        selected = min(
            (candidate.action for candidate in finalists),
            key=discard_action_sort_key,
        )
    else:
        for candidate in finalists:
            candidate.second_step_ukeire_score = second_step_ukeire_score(
                candidate.post_discard_hand,
                known_counts,
                minimum_shanten,
                evaluator,
            )
        maximum_second_step = max(
            candidate.second_step_ukeire_score for candidate in finalists
        )
        selected = min(
            (
                candidate.action
                for candidate in finalists
                if candidate.second_step_ukeire_score == maximum_second_step
            ),
            key=discard_action_sort_key,
        )

    snapshots = tuple(
        candidate.snapshot()
        for candidate in sorted(
            candidates,
            key=lambda candidate: discard_action_sort_key(candidate.action),
        )
    )
    return selected, snapshots


def _choose_riichi(
    legal_actions: tuple[InternalAction, ...],
) -> RiichiAction | None:
    """legalなRiichiActionがあればそれを返す、Always Riichi baseline。

    リーチ合法性は`legal_actions`の存在自体を正本とし、面前・聴牌・持ち点等の
    麻雀ルールをここで再計算しない。将来リーチ / ダマ判断へ差し替える際は、
    この関数の中身だけを置き換えればよい。
    """
    for action in legal_actions:
        if isinstance(action, RiichiAction):
            return action
    return None


class TwoStepUkeirePolicy:
    """同向聴・同現在受け入れ候補だけを2段階受け入れで比較するPolicy。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        """このPolicy世代がlegal discard集合から選ぶprivate extension point。

        通常実行とtraced実行で共有するsingle-source評価pathであり、trace用の
        別evaluation pathは作らない。subclassはこのmethodをoverrideすることで、
        選択とanalysis公開の両方を同時に引き継ぐ。`_decide()`側にanalysis生成を
        置かないため、subclassが基底classのanalysis pathを偶然inheritして
        自分のdecision pathを迂回することがない。
        """
        selected, evaluations = _evaluate_and_choose_discard(
            policy_input, discard_actions
        )
        return PolicyDecision(
            action=selected,
            analysis=TwoStepUkeireAnalysis(candidate_evaluations=evaluations),
        )

    def _decide(self, decision: DecisionContext) -> PolicyDecision:
        """1回のdecision計算からactionとoptional analysisを同時に得る。

        analysisを持つのは実際に打牌評価を実行したbranchだけである。和了、
        リーチ、pass、既存fallbackでは、trace表示のためだけの打牌評価を
        追加実行しない。
        """
        winning_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, _WINNING_ACTION_TYPES)
        )
        if winning_actions:
            return PolicyDecision(
                action=min(winning_actions, key=_winning_action_sort_key)
            )

        riichi_action = _choose_riichi(decision.legal_actions)
        if riichi_action is not None:
            return PolicyDecision(action=riichi_action)

        discard_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, DiscardAction)
        )
        if discard_actions:
            return self._decide_discard(decision.input, discard_actions)

        pass_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, PassAction)
        )
        if pass_actions:
            return PolicyDecision(action=pass_actions[0])

        if len(decision.legal_actions) == 1:
            return PolicyDecision(action=decision.legal_actions[0])

        raise TwoStepUkeirePolicyError(
            "no winning action, discard, or pass is available and multiple "
            "non-discard candidates remain without a defined conservative rule"
        )

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        return self._decide(decision).action

    def choose_action_with_analysis(self, decision: DecisionContext) -> PolicyDecision:
        """`choose_action()`と同じdecision算法を1回だけ実行して返す。"""
        return self._decide(decision)
