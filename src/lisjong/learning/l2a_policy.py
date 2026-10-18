"""L2aの合成Policy：教師規則guard + 打牌scorer + 反応scorer（Issue #316）。

設計`docs/learning-l2a.md` 2.6節。

```text
DecisionContext
    -> classify_l2a_decision（lisjong.learning.l2a_decision）
    -> FORCED / WIN / RIICHI / PASS_RESPONSE : 教師規則guard（guard_action）
    -> 単一候補行                           : その候補（scorerを呼ばない）
    -> DISCARD : #184 shared context + 打牌encoding（M0 / M1）-> 打牌scorer
    -> CALL    : #184 shared context + 反応encoding（R1）      -> 反応scorer
    -> 最大score（同点はcanonical候補順で最初）
    -> decision.legal_actions側のcanonical objectを返す
```

- 入力は`DecisionContext`だけである。HandBelief、hidden information、PRNGを使わない
- 非有限score、候補数とscore数の不一致、encoding / artifactの不一致はすべて例外にし、
  heuristic fallbackしない
- 打牌artifactと反応artifactは、同じsource identityと同じ分類（guard規則）identityを
  持たなければならない。headが合わないartifactはfail closedする
- factoryはtop-level importableなclassのinstanceであり、Arenaは
  `PolicySpec(identity=..., factory=runtime)`としてそのまま使える

torchはoptional extra（`lisjong[ml]`）であり、lazy importで必要とする。
"""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from lisjong.learning.candidate_model import score_candidates
from lisjong.learning.errors import LearnedPolicyError
from lisjong.learning.features import FEATURE_DIMENSION, build_player_safe_feature
from lisjong.learning.l2a_artifact import LoadedL2aScorerArtifact, load_l2a_artifact
from lisjong.learning.l2a_decision import (
    L2aDecisionKind,
    classify_l2a_decision,
    guard_action,
)
from lisjong.learning.l2a_model import build_l2a_scorer_module
from lisjong.learning.model import require_torch
from lisjong.policy_contract import DecisionContext, InternalAction

INFERENCE_DEVICE = "cpu"


@dataclass(frozen=True, slots=True)
class L2aDecision:
    """1判断の選択結果。scorerを呼んだときだけ候補とscoreを持つ。"""

    kind: L2aDecisionKind
    action: InternalAction
    candidates: tuple[InternalAction, ...] | None = None
    scores: tuple[float, ...] | None = None


def select_best(scores: tuple[float, ...]) -> int:
    """最大scoreの位置を返す。同点はcanonical候補順で最初。非有限はfail closed。"""
    best = None
    for index, score in enumerate(scores):
        if not isfinite(score):
            raise LearnedPolicyError(
                f"scorer produced a non-finite score for candidate {index}"
            )
        if best is None or score > scores[best]:
            best = index
    if best is None:
        raise LearnedPolicyError("scorer produced no score")
    return best


@dataclass(frozen=True, slots=True)
class L2aScorer:
    """1回だけloadしたL2a scorer（打牌または反応）。"""

    artifact: LoadedL2aScorerArtifact
    module: object

    @property
    def encoding(self):
        return self.artifact.encoding

    def score(
        self, context: tuple[float, ...], vectors: tuple[tuple[float, ...], ...]
    ) -> tuple[float, ...]:
        if len(context) != FEATURE_DIMENSION:
            raise LearnedPolicyError("shared context has an unexpected dimension")
        dimension = self.encoding.dimension
        if not vectors or any(len(vector) != dimension for vector in vectors):
            raise LearnedPolicyError("candidate vector has an unexpected dimension")
        torch = require_torch()
        with torch.inference_mode():
            output = score_candidates(
                self.module,
                torch.tensor([list(context)], dtype=torch.float32),
                torch.tensor([list(vector) for vector in vectors], dtype=torch.float32),
                torch.zeros(len(vectors), dtype=torch.long),
            )
        if tuple(output.shape) != (len(vectors),):
            raise LearnedPolicyError("scorer produced an unexpected output shape")
        return tuple(output.tolist())


@dataclass(frozen=True, slots=True)
class L2aPolicy:
    """1 seat・1 gameぶんのL2a合成Policy。判断間の可変stateを持たない。"""

    runtime: "L2aRuntime"

    def decide(self, decision: DecisionContext) -> L2aDecision:
        if not isinstance(decision, DecisionContext):
            raise LearnedPolicyError("decision must be a DecisionContext")
        kind = classify_l2a_decision(decision)
        guarded = guard_action(decision, kind)
        if guarded is not None:
            return L2aDecision(kind=kind, action=guarded)

        scorer = self.runtime.scorer_for(kind)
        encoded = scorer.encoding.encode(decision)
        context = build_player_safe_feature(decision.input)
        scores = scorer.score(context, encoded.vectors)
        if len(scores) != len(encoded.actions):
            raise LearnedPolicyError("scorer produced an unexpected score count")
        action = encoded.actions[select_best(scores)]
        if not any(action is legal for legal in decision.legal_actions):
            raise LearnedPolicyError(
                "selected candidate is not a canonical legal action object"
            )
        return L2aDecision(
            kind=kind, action=action, candidates=encoded.actions, scores=scores
        )

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        """`Policy` Protocolの入口。"""
        return self.decide(decision).action


@dataclass(frozen=True, slots=True)
class L2aRuntime:
    """打牌scorerと反応scorerを1回だけloadし、Policyを生成するfactory。"""

    discard: L2aScorer
    reaction: L2aScorer

    def __post_init__(self) -> None:
        if self.discard.encoding.head is not L2aDecisionKind.DISCARD:
            raise LearnedPolicyError("discard artifact does not have a DISCARD head")
        if self.reaction.encoding.head is not L2aDecisionKind.CALL:
            raise LearnedPolicyError("reaction artifact does not have a CALL head")
        discard_encoding = self.discard.artifact.manifest["encoding"]
        reaction_encoding = self.reaction.artifact.manifest["encoding"]
        if discard_encoding["classification"] != reaction_encoding["classification"]:
            raise LearnedPolicyError("artifacts bind different L2a guard rules")
        if self.discard.artifact.source_identity != (
            self.reaction.artifact.source_identity
        ):
            raise LearnedPolicyError("artifacts were trained from different sources")

    @property
    def identity(self) -> tuple[str, str]:
        """(打牌artifact identity, 反応artifact identity)。"""
        return (self.discard.artifact.identity, self.reaction.artifact.identity)

    def scorer_for(self, kind: L2aDecisionKind) -> L2aScorer:
        if kind is L2aDecisionKind.DISCARD:
            return self.discard
        if kind is L2aDecisionKind.CALL:
            return self.reaction
        raise LearnedPolicyError(f"{kind.name} decision has no scorer")

    def create_policy(self) -> L2aPolicy:
        return L2aPolicy(self)

    def __call__(self) -> L2aPolicy:
        return self.create_policy()


def _load_scorer(path: str | Path) -> L2aScorer:
    torch = require_torch()
    artifact = load_l2a_artifact(path)
    module = build_l2a_scorer_module(artifact.model_config, artifact.weights)
    if module.training:
        raise LearnedPolicyError("inference module must be in eval mode")
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(False)
        if parameter.device.type != INFERENCE_DEVICE:
            raise LearnedPolicyError(f"inference parameter {name} is not on the CPU")
    with torch.inference_mode():
        for name, parameter in module.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise LearnedPolicyError(
                    f"inference parameter {name} contains a non-finite value"
                )
    return L2aScorer(artifact=artifact, module=module)


def load_l2a_policy_factory(
    discard_artifact: str | Path, reaction_artifact: str | Path
) -> L2aRuntime:
    """2つのartifactをstrict loadし、Policy factoryとして使えるruntimeを返す。"""
    return L2aRuntime(
        discard=_load_scorer(discard_artifact),
        reaction=_load_scorer(reaction_artifact),
    )


__all__ = [
    "INFERENCE_DEVICE",
    "L2aDecision",
    "L2aPolicy",
    "L2aRuntime",
    "L2aScorer",
    "load_l2a_policy_factory",
    "select_best",
]
