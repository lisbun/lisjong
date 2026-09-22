"""learned candidate scorerが読むpurpose-specific discard candidate view。

Issue #187のL0.1に対応する。NNへshanten / ukeireを再発見させるのではなく、
`lisjong.structural_efficiency`（Issue #177）が既に所有するcanonical /
player-safeな牌効率semanticを、candidate単位のmodel-facing projectionへ写す。

```text
DecisionContext
    -> legal DiscardActionだけを抽出
    -> canonical structural semantics（#177をsingle sourceとして再利用）
    -> canonical順のDiscardCandidateFeatures tuple
```

このmoduleは新しいshanten / ukeire algorithmを実装しない。次の既存semanticを
そのままsingle sourceとして呼ぶ。

```text
known_tile_counts             Policy-visibleな既知牌counting
StructuralShantenEvaluator    decision-local shanten評価 / memoization
evaluate_post_discard_hands   actual discard identity -> 打牌後純手牌 / shanten
ukeire_count                  current受け入れ
second_step_ukeire_score      2段階受け入れscore
discard_action_sort_key       canonical deterministic ordering
```

## contract identity

```text
lisjong.structural_efficiency
    = canonical reusable Mahjong calculation semantic

lisjong.learning.candidate_features
    = purpose-specific model-facing candidate projection

TwoStepUkeireCandidateEvaluation
    = TwoStep固有のstaged trace snapshot
```

`TwoStepUkeireCandidateEvaluation`（Issue #87）はTwoStepUkeireが実際に辿った
staged evaluation pathを表すPolicy固有のtrace snapshotであり、このmoduleの
candidate feature contractとは別物である。どちらか一方をgeneric ML schemaへ
変更して共有しない。

本contractはIssue #184のflat action vocabulary BC path（`FEATURE_IDENTITY`、
`DATASET_SCHEMA`、`MODEL_ARTIFACT_SCHEMA`、action vocabulary identity）とも
別のpurpose-specific contractとして**追加**される。既存identityを
candidate-centricへ遡及変更しない。

## second-step materialization contract（v1）

```text
post_discard_shanten    全legal discard candidate
current_ukeire_count    全legal discard candidate
second_step_ukeire_score    明示requestされたcandidateだけ
```

`second_step_ukeire_score()`はcandidateあたりのcostが他stageより桁で大きい
（Issue #187 performance preflightを参照）。v1では全candidate mandatoryに
せず、callerが`second_step_actions`で明示したcandidateだけをmaterializeする
selective contractとする。requestしていないcandidateを暗黙に評価しない。

availabilityは3状態を混同せずに表す。

```text
EVALUATED          評価済み。score 0も正当な評価結果である
NOT_MATERIALIZED   applicableだがrequestされていない
NOT_APPLICABLE     semantic上適用しない（打牌後聴牌以上）
```

`post_discard_shanten <= 0`（打牌後聴牌以上）では2段階受け入れを
`NOT_APPLICABLE`とする。第1有効牌のツモが和了そのものであり、「仮想ツモ後に
打牌して次の受け入れを測る」という`second_step_ukeire_score()`のsemanticが
成立しないためである。これはIssue #87のstaged semantics（TwoStepUkeireも
最小向聴が0なら2段目を評価せずtie-breakへ進む）と一致する。
`NOT_APPLICABLE`なcandidateを明示requestしてもfail closedせず、
`NOT_APPLICABLE`のまま返す。requestがfail closedするのは、そのactionが
現在のlegal discard candidateでない場合だけである。

## information boundary

入力は`DecisionContext`（`PolicyInput` + legal actions）だけである。山の内部
状態、王牌、他家の実手牌、未来のevent、`GameTrace`のprivileged truth、
RiichiEnv state、Arena固有dataへ依存しない。decision間で共有するglobal /
cross-decision cacheも持たない。shanten cacheは1 build呼び出し内だけで使う
decision-local optimizationである。

## このmoduleが提供しないもの

```text
candidate scorer model
generic Policy router / dispatcher framework
universal CandidateEvaluation
新しいshanten / ukeire algorithm
```

learned candidate scorerとその最小orchestration seamはL0.2の責務である。
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from lisjong.learning.errors import CandidateFeatureError
from lisjong.policy_contract import DecisionContext, DiscardAction
from lisjong.structural_efficiency import (
    StructuralShantenEvaluator,
    discard_action_sort_key,
    evaluate_post_discard_hands,
    known_tile_counts,
    second_step_ukeire_score,
    ukeire_count,
)

CANDIDATE_FEATURE_IDENTITY = "lisjong-offense-l0.1-discard-candidate-feature-v1"
"""このcandidate feature contractのidentity。

Issue #184のflat action vocabulary path（`FEATURE_IDENTITY`等）とは別の
purpose-specific contractである。fieldの意味、availability semantics、
canonical orderingのいずれかを変える場合は既存identityを書き換えず、
新しいidentityとして追加する。
"""

_SECOND_STEP_APPLICABLE_MINIMUM_SHANTEN = 1
"""2段階受け入れがsemantic上成立する最小のpost-discard向聴数。

打牌後聴牌以上（向聴数0以下）では第1有効牌のツモが和了であり、仮想ツモ後の
「次の受け入れ」が定義されない。
"""


class SecondStepStatus(Enum):
    """1 candidateの2段階受け入れavailability。

    `EVALUATED`と`NOT_MATERIALIZED` / `NOT_APPLICABLE`を混同しない。評価済みの
    score 0は`EVALUATED`であり、未評価や適用外を意味しない。
    """

    EVALUATED = "evaluated"
    """実際に`second_step_ukeire_score()`を評価した。score 0も含む。"""

    NOT_MATERIALIZED = "not_materialized"
    """適用可能だが、このbuildではrequestされなかった。"""

    NOT_APPLICABLE = "not_applicable"
    """打牌後聴牌以上で、2段階受け入れのsemanticが成立しない。"""


@dataclass(frozen=True, slots=True)
class DiscardCandidateFeatures:
    """1 legal discard candidateのmodel-facing structural projection。

    `action`は元の`DecisionContext.legal_actions`にあるcanonical
    `DiscardAction` objectそのものである。赤5 / 通常5、ツモ切り、actorの
    identityを保つため、等価なactionを再構築しない。

    `second_step_ukeire_score`は`second_step_status`と組で読む。
    `EVALUATED`のときだけintであり、それ以外では`None`である。
    """

    action: DiscardAction
    post_discard_shanten: int
    current_ukeire_count: int
    second_step_ukeire_score: int | None
    second_step_status: SecondStepStatus

    def __post_init__(self) -> None:
        if not isinstance(self.action, DiscardAction):
            raise TypeError("action must be a DiscardAction")
        if type(self.post_discard_shanten) is not int:
            raise TypeError("post_discard_shanten must be an int")
        if type(self.current_ukeire_count) is not int:
            raise TypeError("current_ukeire_count must be an int")
        if not isinstance(self.second_step_status, SecondStepStatus):
            raise TypeError("second_step_status must be a SecondStepStatus")
        if self.second_step_status is SecondStepStatus.EVALUATED:
            if type(self.second_step_ukeire_score) is not int:
                raise TypeError(
                    "second_step_ukeire_score must be an int when "
                    "second_step_status is EVALUATED"
                )
        elif self.second_step_ukeire_score is not None:
            raise ValueError(
                "second_step_ukeire_score must be None unless "
                "second_step_status is EVALUATED"
            )


def legal_discard_candidates(
    decision: DecisionContext,
) -> tuple[DiscardAction, ...]:
    """legal `DiscardAction`だけをcanonical順で重複なく返す。

    non-discard legal action（リーチ宣言、鳴き、和了、pass等）は含めない。
    順序は`discard_action_sort_key()`だけで決まり、`legal_actions`の入力順序へ
    依存しない。返すobjectは元のlegal action objectそのものである。
    """
    return tuple(
        sorted(
            (
                action
                for action in decision.legal_actions
                if isinstance(action, DiscardAction)
            ),
            key=discard_action_sort_key,
        )
    )


def _resolve_second_step_requests(
    requested: Iterable[DiscardAction],
    candidates: Sequence[DiscardAction],
) -> frozenset[DiscardAction]:
    """requestされたactionを現在のlegal discard candidateへ解決する。

    現在のlegal discard candidateでないactionはfail closedする。`DiscardAction`
    はfrozen dataclassであり、`DecisionContext`がsemantic重複を禁止するため、
    この照合はcanonical action identityと一対一に対応する。
    """
    try:
        requested_actions = tuple(requested)
    except TypeError:
        raise TypeError("second_step_actions must be an iterable") from None

    candidate_set = frozenset(candidates)
    for action in requested_actions:
        if not isinstance(action, DiscardAction):
            raise TypeError("second_step_actions must contain only DiscardAction")
        if action not in candidate_set:
            raise CandidateFeatureError(
                "second_step_actions must reference a legal discard candidate of "
                f"this decision: {action!r} is not one of them"
            )
    return frozenset(requested_actions)


def build_discard_candidate_features(
    decision: DecisionContext,
    *,
    second_step_actions: Iterable[DiscardAction] = (),
) -> tuple[DiscardCandidateFeatures, ...]:
    """legal discard candidateのcandidate feature tupleをcanonical順で返す。

    全legal discard candidateがちょうど1回ずつ含まれる。post-discard shantenと
    current ukeireは全candidateでmaterializeし、2段階受け入れは
    `second_step_actions`で明示requestされたcandidateだけを評価する
    （module docstringのsecond-step materialization contractを参照）。

    `known_tile_counts()`と`StructuralShantenEvaluator`はこの呼び出しで1回だけ
    用意し、全candidateで共有する。cacheはこのdecisionの範囲を超えない。

    legal discard candidateが存在しないdecision、および現在のlegal discard
    candidateでないactionをrequestした場合は`CandidateFeatureError`で
    fail closedする。
    """
    if not isinstance(decision, DecisionContext):
        raise TypeError("decision must be a DecisionContext")

    candidates = legal_discard_candidates(decision)
    if not candidates:
        raise CandidateFeatureError(
            "decision has no legal DiscardAction to build candidate features from"
        )

    requested_second_step = _resolve_second_step_requests(
        second_step_actions, candidates
    )

    policy_input = decision.input
    known_counts = known_tile_counts(policy_input)
    evaluator = StructuralShantenEvaluator()
    evaluated = evaluate_post_discard_hands(policy_input, candidates, evaluator)

    features: list[DiscardCandidateFeatures] = []
    for evaluation in evaluated:
        shanten = evaluation.post_discard_shanten
        current_ukeire = ukeire_count(
            evaluation.post_discard_hand,
            known_counts,
            shanten,
            evaluator,
        )
        if shanten < _SECOND_STEP_APPLICABLE_MINIMUM_SHANTEN:
            status = SecondStepStatus.NOT_APPLICABLE
            score = None
        elif evaluation.action in requested_second_step:
            status = SecondStepStatus.EVALUATED
            score = second_step_ukeire_score(
                evaluation.post_discard_hand,
                known_counts,
                shanten,
                evaluator,
            )
        else:
            status = SecondStepStatus.NOT_MATERIALIZED
            score = None
        features.append(
            DiscardCandidateFeatures(
                action=evaluation.action,
                post_discard_shanten=shanten,
                current_ukeire_count=current_ukeire,
                second_step_ukeire_score=score,
                second_step_status=status,
            )
        )
    return tuple(features)
