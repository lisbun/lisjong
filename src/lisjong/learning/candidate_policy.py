"""L0.2 candidate-scoring Learned Offense Policyとそのpublic factory seam。

Issue #189に対応する。win / immediate riichi / O0 no-callをmodel capacityを
使わずdeterministicに保証し、normal discardだけをlearned scalar candidate
scorerで選ぶ。

```text
DecisionContext
    -> O0 precedence（lisjong.learning._o0）
         WIN       canonical deterministic win
         RIICHI    canonical RiichiAction
         RESPONSE  canonical PassAction（無ければfail closed）
         DISCARD
            -> #184 shared context + two-pass candidate features + encoding
            -> candidateごとのscalar score
            -> 最大score（同点はcanonical discard順で最初）のcandidate
            -> そのcandidateが持つlegal DiscardAction objectそのもの
```

固定する原則。

- 入力は`DecisionContext`だけである。hidden information、RiichiEnv /
  RiichiLab固有型、PRNGを使わない
- scorerは`legal_discard_candidates()`由来のcandidateにだけscoreを付けるため、
  legal `DiscardAction`以外を返せない。actionを再構築せず、legal action側の
  objectをそのまま返す
- Ankan / Kakan / KyuushuKyuuhai等のvoluntary actionはO0では選ばない
- 非有限score、candidate数とscore数の不一致、guard不成立はすべて例外にし、
  silent heuristic fallbackを持たない
- Issue #184の`LearnedOffensePolicy`（flat action vocabulary path）は変更しない
- factoryはtop-level importableなclassのinstanceであり、Arenaは
  `PolicySpec(identity=..., factory=runtime)`としてそのまま利用できる

torchはoptional extra（`lisjong[ml]`）であり、lazy importで必要とする。
"""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from lisjong.learning._o0 import (
    O0DecisionKind,
    classify_o0_decision,
    deterministic_guard_action,
)
from lisjong.learning.candidate_artifact import (
    LoadedCandidateScorerArtifact,
    load_candidate_artifact,
)
from lisjong.learning.candidate_encoding import (
    CANDIDATE_ENCODING_DIMENSION,
    build_scorer_candidates,
    encode_candidates,
)
from lisjong.learning.candidate_features import DiscardCandidateFeatures
from lisjong.learning.candidate_model import build_scorer_module, score_candidates
from lisjong.learning.errors import LearnedPolicyError
from lisjong.learning.features import FEATURE_DIMENSION, build_player_safe_feature
from lisjong.learning.model import require_torch
from lisjong.policy_contract import DecisionContext, InternalAction

INFERENCE_DEVICE = "cpu"


@dataclass(frozen=True, slots=True)
class CandidateScorerDecision:
    """1 decisionの選択結果。

    `candidates` / `scores`はDISCARD branchでscorerを実行したときだけ持ち、
    guard branchでは`None`である。offline diagnosticsが同じ1回の計算から
    semantic agreementを読むために公開する。
    """

    kind: O0DecisionKind
    action: InternalAction
    candidates: tuple[DiscardCandidateFeatures, ...] | None = None
    scores: tuple[float, ...] | None = None

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
class LearnedCandidateOffensePolicy:
    """1 seat・1 gameぶんのL0.2 Learned Offense Policy instance。

    decision間の可変stateを持たず、共有するのはimmutableなloaded modelだけで
    ある。
    """

    runtime: "CandidateScorerRuntime"

    def decide(self, decision: DecisionContext) -> CandidateScorerDecision:
        """O0 precedenceに従ってactionを選び、scorer結果と共に返す。"""
        if not isinstance(decision, DecisionContext):
            raise LearnedPolicyError("decision must be a DecisionContext")

        kind = classify_o0_decision(decision)
        if kind is not O0DecisionKind.DISCARD:
            return CandidateScorerDecision(
                kind=kind, action=deterministic_guard_action(decision, kind)
            )

        candidates = build_scorer_candidates(decision)
        encoded = encode_candidates(candidates)
        context = build_player_safe_feature(decision.input)
        scores = tuple(self.runtime.score(context, encoded))
        if len(scores) != len(candidates):
            raise LearnedPolicyError("scorer produced an unexpected score count")

        # candidatesはcanonical discard順であり、strictな`>`で更新するため、
        # 同点時はcanonical順で最初のcandidateが残る。
        best = None
        for index, score in enumerate(scores):
            if not isfinite(score):
                raise LearnedPolicyError(
                    f"scorer produced a non-finite score for candidate {index}"
                )
            if best is None or score > scores[best]:
                best = index
        action = candidates[best].action
        if not any(action is legal for legal in decision.legal_actions):
            raise LearnedPolicyError(
                "selected candidate is not a canonical legal action object"
            )
        return CandidateScorerDecision(
            kind=kind, action=action, candidates=candidates, scores=scores
        )

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        """`Policy` Protocolの入口。`decide()`と同じ算法を1回だけ実行する。"""
        return self.decide(decision).action


@dataclass(frozen=True, slots=True)
class CandidateScorerRuntime:
    """1回だけloadしたscorerと、そこからPolicyを生成するfactory。"""

    artifact: LoadedCandidateScorerArtifact
    module: object

    @property
    def identity(self) -> str:
        """artifact identity。Policy比較対象のidentityはcallerが明示する。"""
        return self.artifact.identity

    def score(
        self,
        context: tuple[float, ...],
        candidates: tuple[tuple[float, ...], ...],
    ) -> tuple[float, ...]:
        """1 decisionのshared contextとcandidate vector列からscore列を返す。"""
        if len(context) != FEATURE_DIMENSION:
            raise LearnedPolicyError("shared context has an unexpected dimension")
        if not candidates:
            raise LearnedPolicyError("normal discard decision has no candidate")
        if any(len(vector) != CANDIDATE_ENCODING_DIMENSION for vector in candidates):
            raise LearnedPolicyError("candidate vector has an unexpected dimension")
        torch = require_torch()
        with torch.inference_mode():
            output = score_candidates(
                self.module,
                torch.tensor([list(context)], dtype=torch.float32),
                torch.tensor(
                    [list(vector) for vector in candidates], dtype=torch.float32
                ),
                torch.zeros(len(candidates), dtype=torch.long),
            )
        if tuple(output.shape) != (len(candidates),):
            raise LearnedPolicyError("scorer produced an unexpected output shape")
        return tuple(output.tolist())

    def create_policy(self) -> LearnedCandidateOffensePolicy:
        """game / seatごとのfresh Policy instanceを返す。"""
        return LearnedCandidateOffensePolicy(self)

    def __call__(self) -> LearnedCandidateOffensePolicy:
        return self.create_policy()


def load_candidate_scorer_policy_factory(path: str | Path) -> CandidateScorerRuntime:
    """artifactをstrict loadし、Policy factoryとして使えるruntimeを返す。"""
    torch = require_torch()
    artifact = load_candidate_artifact(path)
    module = build_scorer_module(artifact.model_config, artifact.weights)
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
    return CandidateScorerRuntime(artifact=artifact, module=module)


__all__ = [
    "INFERENCE_DEVICE",
    "CandidateScorerDecision",
    "CandidateScorerRuntime",
    "LearnedCandidateOffensePolicy",
    "load_candidate_scorer_policy_factory",
]
