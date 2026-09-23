"""L0.2a semantic-envelope Learned Offense Policy（Issue #191）。

#189のfrozen candidate scorerを、exactに計算できる牌効率hierarchyの内側
だけで使う。hierarchyはNNに近似させる対象ではなく、serving policyの
selection constraintとしてdeterministicに保証する。

```text
DecisionContext
    -> O0 precedence（lisjong.learning._o0、#189と同一）
         WIN / RIICHI / RESPONSE   #189と同じdeterministic guard
         DISCARD
            -> build_scorer_candidates()（#189 two-pass request policy）
            -> S1 minimum post-discard shanten
            -> S2 maximum current ukeire
            -> S3 maximum second-step score（適用条件は下記）
            -> semantic survivors
                 1件   そのcandidateのlegal DiscardAction（scorerは実行しない）
                 複数  full candidate tupleへscoreし、argmaxをsurvivorに限定
```

S3の適用条件は#189 two-pass request policy
（`second_step_finalists()`）と完全に一致させる。

```text
minimum shanten == 0      S3なし。全S2 survivorはNOT_APPLICABLE
elif S2 survivor == 1件   S3なし。そのsurvivorはNOT_MATERIALIZED
else                      全S2 survivorがEVALUATEDであることを要求し、
                          maximum second_step_ukeire_scoreを残す
```

期待statusと異なるcandidateはfail closedする。未評価candidateをscore 0として
扱わず、評価済みcandidateだけの部分比較もしない。

この選択規則は`TwoStepUkeirePolicy`の通常打牌規則（打牌後向聴数 > 現在受け入れ
> 2段階受け入れ > stable tie-break）と同じhierarchyである。したがって
constant scorerではTwoStepと同じactionを返し、learned scorerがTwoStepと異なる
打牌を選べるのはS1 / S2 / S3がすべて同値なresidual candidate間だけである。

固定する原則。

- #189 `LearnedCandidateOffensePolicy` / `CandidateScorerRuntime`のidentityと
  behaviorを変更しない。このmoduleはselection方法だけが異なる別classである
- #184 shared feature、#187 candidate feature、#189 encoding / request policy /
  model / artifact schemaは変更しない。scorerはfull candidate tupleへ#189と
  同じ入力でscoreを出す
- scorer同点はcanonical discard順で最初のsurvivor。非有限score・score数不一致は
  fail closed。survivorが1件でscorerを省略した場合はscore生成も非有限検査も
  行わない（仕様）
- actionを再構築せず、`decision.legal_actions`側のobjectをそのまま返す
- hidden information、PRNG、concrete Policy module（TwoStep）へ依存しない
- runtime identityは#189 artifact identityそのものではなく、
  `SEMANTIC_ENVELOPE_IDENTITY`とartifact identityから決定的に合成する
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from lisjong.learning._canonical import value_digest
from lisjong.learning._o0 import (
    O0DecisionKind,
    classify_o0_decision,
    deterministic_guard_action,
)
from lisjong.learning.candidate_encoding import (
    build_scorer_candidates,
    encode_candidates,
)
from lisjong.learning.candidate_features import (
    DiscardCandidateFeatures,
    SecondStepStatus,
)
from lisjong.learning.candidate_policy import (
    CandidateScorerRuntime,
    load_candidate_scorer_policy_factory,
)
from lisjong.learning.errors import CandidateFeatureError, LearnedPolicyError
from lisjong.learning.features import build_player_safe_feature
from lisjong.policy_contract import DecisionContext, InternalAction

SEMANTIC_ENVELOPE_IDENTITY = "lisjong-offense-l0.2-semantic-envelope-v1"
"""semantic-envelope selection semanticsのidentity。"""


def semantic_envelope_survivors(
    candidates: Sequence[DiscardCandidateFeatures],
) -> tuple[int, ...]:
    """S1 / S2 / S3を通過したcandidateのindexをcanonical順で返す。

    `candidates`は`build_scorer_candidates()`が返すfull canonical tupleである。
    second-step statusがtwo-pass request policyの期待と異なればfail closedする。
    """
    values = tuple(candidates)
    if not values:
        raise CandidateFeatureError("candidate tuple must not be empty")

    minimum_shanten = min(item.post_discard_shanten for item in values)
    stage = [
        index
        for index, item in enumerate(values)
        if item.post_discard_shanten == minimum_shanten
    ]
    maximum_ukeire = max(values[index].current_ukeire_count for index in stage)
    stage = [
        index for index in stage if values[index].current_ukeire_count == maximum_ukeire
    ]

    if minimum_shanten <= 0:
        expected = SecondStepStatus.NOT_APPLICABLE
    elif len(stage) == 1:
        expected = SecondStepStatus.NOT_MATERIALIZED
    else:
        expected = SecondStepStatus.EVALUATED
    if any(values[index].second_step_status is not expected for index in stage):
        raise CandidateFeatureError(
            "semantic envelope survivors do not carry the second-step status "
            f"{expected.value!r} required by the two-pass request policy"
        )
    if expected is SecondStepStatus.EVALUATED:
        maximum_second_step = max(
            values[index].second_step_ukeire_score for index in stage
        )
        stage = [
            index
            for index in stage
            if values[index].second_step_ukeire_score == maximum_second_step
        ]
    return tuple(stage)


@dataclass(frozen=True, slots=True)
class SemanticEnvelopeDecision:
    """1 decisionの選択結果。

    `candidates` / `survivors`はDISCARD branchだけが持ち、guard branchでは
    `None`である。`scores`はscorerを実際に実行したとき（survivorが複数）
    だけ持つ。offline diagnosticsが同じ1回の計算を読むために公開する。
    """

    kind: O0DecisionKind
    action: InternalAction
    candidates: tuple[DiscardCandidateFeatures, ...] | None = None
    survivors: tuple[int, ...] | None = None
    scores: tuple[float, ...] | None = None

    @property
    def scorer_invoked(self) -> bool:
        return self.scores is not None

    @property
    def selected_candidate(self) -> DiscardCandidateFeatures | None:
        if self.candidates is None:
            return None
        return next(
            candidate
            for candidate in self.candidates
            if candidate.action is self.action
        )


@dataclass(frozen=True, slots=True)
class SemanticEnvelopeOffensePolicy:
    """1 seat・1 gameぶんのL0.2a semantic-envelope Policy instance。

    `runtime`は`score(context, candidates)`を持つscorerである。decision間の
    可変stateを持たない。
    """

    runtime: object

    def decide(self, decision: DecisionContext) -> SemanticEnvelopeDecision:
        """O0 precedenceとsemantic envelopeに従ってactionを選ぶ。"""
        if not isinstance(decision, DecisionContext):
            raise LearnedPolicyError("decision must be a DecisionContext")

        kind = classify_o0_decision(decision)
        if kind is not O0DecisionKind.DISCARD:
            return SemanticEnvelopeDecision(
                kind=kind, action=deterministic_guard_action(decision, kind)
            )

        candidates = build_scorer_candidates(decision)
        survivors = semantic_envelope_survivors(candidates)
        scores = None
        if len(survivors) == 1:
            best = survivors[0]
        else:
            encoded = encode_candidates(candidates)
            context = build_player_safe_feature(decision.input)
            scores = tuple(self.runtime.score(context, encoded))
            if len(scores) != len(candidates):
                raise LearnedPolicyError("scorer produced an unexpected score count")
            for index, score in enumerate(scores):
                if not isfinite(score):
                    raise LearnedPolicyError(
                        f"scorer produced a non-finite score for candidate {index}"
                    )
            # survivorsはcanonical discard順であり、strictな`>`で更新するため、
            # 同点時はcanonical順で最初のsurvivorが残る。
            best = survivors[0]
            for index in survivors[1:]:
                if scores[index] > scores[best]:
                    best = index
        action = candidates[best].action
        if not any(action is legal for legal in decision.legal_actions):
            raise LearnedPolicyError(
                "selected candidate is not a canonical legal action object"
            )
        return SemanticEnvelopeDecision(
            kind=kind,
            action=action,
            candidates=candidates,
            survivors=survivors,
            scores=scores,
        )

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        """`Policy` Protocolの入口。`decide()`と同じ算法を1回だけ実行する。"""
        return self.decide(decision).action


def semantic_envelope_runtime_identity(artifact_identity: str) -> str:
    """selection semanticsとfrozen artifact identityから合成したidentity。"""
    return value_digest(
        {
            "candidate_scorer_artifact": artifact_identity,
            "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
        }
    )


@dataclass(frozen=True, slots=True)
class SemanticEnvelopeRuntime:
    """#189 frozen scorerをsemantic-envelope selectionで使うPolicy factory。"""

    scorer: CandidateScorerRuntime

    @property
    def artifact_identity(self) -> str:
        return self.scorer.identity

    @property
    def identity(self) -> str:
        """#189 runtime（artifact identity）と衝突しない合成identity。"""
        return semantic_envelope_runtime_identity(self.scorer.identity)

    def score(
        self,
        context: tuple[float, ...],
        candidates: tuple[tuple[float, ...], ...],
    ) -> tuple[float, ...]:
        return self.scorer.score(context, candidates)

    def create_policy(self) -> SemanticEnvelopeOffensePolicy:
        """game / seatごとのfresh Policy instanceを返す。"""
        return SemanticEnvelopeOffensePolicy(self)

    def __call__(self) -> SemanticEnvelopeOffensePolicy:
        return self.create_policy()


def load_semantic_envelope_policy_factory(path: str | Path) -> SemanticEnvelopeRuntime:
    """#189 candidate scorer artifactをstrict loadし、envelope runtimeを返す。"""
    return SemanticEnvelopeRuntime(load_candidate_scorer_policy_factory(path))


__all__ = [
    "SEMANTIC_ENVELOPE_IDENTITY",
    "SemanticEnvelopeDecision",
    "SemanticEnvelopeOffensePolicy",
    "SemanticEnvelopeRuntime",
    "load_semantic_envelope_policy_factory",
    "semantic_envelope_runtime_identity",
    "semantic_envelope_survivors",
]
