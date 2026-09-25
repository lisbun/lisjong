"""L0.3 outcome-Q residual scorer runtime（Issue #200、#79 A2 / §9）。

strict loadしたoutcome-Q artifactを、#191 `SemanticEnvelopeOffensePolicy`の
residual scorerとして使うPolicy factoryである。新しいPolicy classは作らない。

```text
DecisionContext
    -> SemanticEnvelopeOffensePolicy（#191、変更しない）
         O0 guard / S1 / S2 / S3 envelope
         survivor 1件    scorerを呼ばない
         survivor >= 2   full candidate tupleへQをscore
                         -> argmaxをsurvivorに限定（同値はcanonical先頭）
```

baseline arm（`ConstantResidualRuntime`）との違いはresidual scorerだけである。
runtime identityは#79 A2でfreezeした次のdigestである。

```json
{
  "outcome_q_artifact": "<artifact identity>",
  "selection_policy": "lisjong-offense-l0.2-semantic-envelope-v1"
}
```

#189 `CandidateScorerRuntime` / #191 `SemanticEnvelopeRuntime`のidentityと
behaviorは変更しない。torchはoptional extra（`lisjong[ml]`）であり、lazy importで
必要とする。
"""

from dataclasses import dataclass
from pathlib import Path

from lisjong.learning._canonical import value_digest
from lisjong.learning.candidate_encoding import CANDIDATE_ENCODING_DIMENSION
from lisjong.learning.candidate_model import build_scorer_module, score_candidates
from lisjong.learning.envelope_policy import (
    SEMANTIC_ENVELOPE_IDENTITY,
    SemanticEnvelopeOffensePolicy,
)
from lisjong.learning.errors import LearnedPolicyError
from lisjong.learning.features import FEATURE_DIMENSION
from lisjong.learning.model import require_torch
from lisjong.learning.outcome_q_artifact import (
    LoadedOutcomeQArtifact,
    load_outcome_q_artifact,
)

INFERENCE_DEVICE = "cpu"


def outcome_q_runtime_identity(artifact_identity: str) -> str:
    """selection semanticsとoutcome-Q artifact identityから合成したidentity。"""
    return value_digest(
        {
            "outcome_q_artifact": artifact_identity,
            "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
        }
    )


@dataclass(frozen=True, slots=True)
class OutcomeQRuntime:
    """1回だけloadしたoutcome-Q scorerと、そこからPolicyを生成するfactory。"""

    artifact: LoadedOutcomeQArtifact
    module: object

    @property
    def artifact_identity(self) -> str:
        return self.artifact.identity

    @property
    def identity(self) -> str:
        """artifact identityそのものとは異なる、#79 A2の合成identity。"""
        return outcome_q_runtime_identity(self.artifact.identity)

    def score(
        self,
        context: tuple[float, ...],
        candidates: tuple[tuple[float, ...], ...],
    ) -> tuple[float, ...]:
        """1 decisionのshared contextとfull candidate vector列からQ列を返す。"""
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

    def create_policy(self) -> SemanticEnvelopeOffensePolicy:
        """game / seatごとのfresh Policy instanceを返す。"""
        return SemanticEnvelopeOffensePolicy(self)

    def __call__(self) -> SemanticEnvelopeOffensePolicy:
        return self.create_policy()


def load_outcome_q_policy_factory(path: str | Path) -> OutcomeQRuntime:
    """outcome-Q artifactをstrict loadし、Policy factoryとして使えるruntimeを返す。"""
    torch = require_torch()
    artifact = load_outcome_q_artifact(path)
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
    return OutcomeQRuntime(artifact=artifact, module=module)


@dataclass(frozen=True, slots=True)
class OutcomeQPolicyLoader:
    """process worker内でoutcome-Q Policyを生成するpicklableなfactory（Issue #209）。

    artifact pathと期待するartifact identityだけを持ち、torch moduleやPolicy
    instanceをprocess間で渡さない。呼び出すたびにartifactをstrict loadし、
    identityが一致しなければfail closedする。
    """

    artifact_path: str | Path
    artifact_identity: str

    def __call__(self) -> SemanticEnvelopeOffensePolicy:
        runtime = load_outcome_q_policy_factory(self.artifact_path)
        if runtime.artifact_identity != self.artifact_identity:
            raise LearnedPolicyError(
                "outcome-Q artifact identity does not match the policy loader"
            )
        return runtime.create_policy()


__all__ = [
    "INFERENCE_DEVICE",
    "OutcomeQPolicyLoader",
    "OutcomeQRuntime",
    "load_outcome_q_policy_factory",
    "outcome_q_runtime_identity",
]
