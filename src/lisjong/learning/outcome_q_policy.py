"""L0.3 outcome-Q residual runtime（lisjong-project#79 A2 / §9）。

#191 `SemanticEnvelopeOffensePolicy`をそのまま使い、residual scorerだけを
frozen outcome-Q artifactへ差し替えたcandidate runtimeである。

```text
DecisionContext
    -> SemanticEnvelopeOffensePolicy（#191、変更しない）
         O0 guard / S1 / S2 / S3 envelope
         survivor == 1   そのsurvivor（scorerは実行しない）
         survivor >= 2   full candidate tupleへQ scoreを返し、
                         argmaxをsurvivorに限定（同点はcanonical順で最初）
```

baseline（`ConstantResidualRuntime`）とcandidateは同じPolicy code pathを通り、
scientific differenceはresidual scorer runtimeだけになる。新しいserving
Policy classは作らない。

runtime identityは#79 A2でfreezeした次のdigestそのものであり、#189 artifact
identity、#191 `semantic_envelope_runtime_identity()`、constant-zero baseline
identityのいずれとも衝突しない。

```json
{
  "outcome_q_artifact": "<artifact identity>",
  "selection_policy": "lisjong-offense-l0.2-semantic-envelope-v1"
}
```

非有限score・score数不一致はPolicy側で、非有限parameter・identity不一致は
load時にfail closedする。torchはoptional extra（`lisjong[ml]`）である。
"""

from dataclasses import dataclass
from pathlib import Path

from lisjong.learning._canonical import value_digest
from lisjong.learning.candidate_policy import (
    build_inference_module,
    score_with_module,
)
from lisjong.learning.envelope_policy import (
    SEMANTIC_ENVELOPE_IDENTITY,
    SemanticEnvelopeOffensePolicy,
)
from lisjong.learning.model import require_torch
from lisjong.learning.outcome_q_artifact import (
    LoadedOutcomeQArtifact,
    load_outcome_q_artifact,
)


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
    """1回だけloadしたQ scorerと、そこからsemantic-envelope Policyを作るfactory。"""

    artifact: LoadedOutcomeQArtifact
    module: object

    @property
    def artifact_identity(self) -> str:
        return self.artifact.identity

    @property
    def identity(self) -> str:
        return outcome_q_runtime_identity(self.artifact.identity)

    def score(
        self,
        context: tuple[float, ...],
        candidates: tuple[tuple[float, ...], ...],
    ) -> tuple[float, ...]:
        """full candidate tupleの各entryへQ estimateを1つずつ返す。"""
        return score_with_module(self.module, context, candidates)

    def create_policy(self) -> SemanticEnvelopeOffensePolicy:
        """game / seatごとのfresh Policy instanceを返す。"""
        return SemanticEnvelopeOffensePolicy(self)

    def __call__(self) -> SemanticEnvelopeOffensePolicy:
        return self.create_policy()


def load_outcome_q_policy_factory(path: str | Path) -> OutcomeQRuntime:
    """outcome-Q artifactをstrict loadし、Policy factoryとして使えるruntimeを返す。"""
    require_torch()
    artifact = load_outcome_q_artifact(path)
    module = build_inference_module(artifact.model_config, artifact.weights)
    return OutcomeQRuntime(artifact=artifact, module=module)


__all__ = [
    "OutcomeQRuntime",
    "load_outcome_q_policy_factory",
    "outcome_q_runtime_identity",
]
