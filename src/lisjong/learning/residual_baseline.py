"""L0.3 constant-zero residual baseline runtime（Issue #193、#79 A2）。

#191 `SemanticEnvelopeOffensePolicy`をそのまま使い、residual scorerだけを
constant-zeroへ差し替えたbaseline runtimeである。

```text
DecisionContext
    -> SemanticEnvelopeOffensePolicy（#191、変更しない）
         O0 guard / S1 / S2 / S3 envelope
         survivor >= 2   full candidate tupleへ0.0をscore
                         -> strict `>`によりcanonical順で最初のsurvivor
```

用途はL0.3 scientific comparisonのbaseline armと、source生成時の非focal
3 seatである。

固定する原則。

- 新しいPolicy classを作らない。`create_policy()`は既存の
  `SemanticEnvelopeOffensePolicy(self)`を返す
- `envelope_policy.py`の`SemanticEnvelopeRuntime` /
  `load_semantic_envelope_policy_factory` /
  `semantic_envelope_runtime_identity`は#189 artifactのstrict loadに結び付いて
  いるため変更しない。本runtimeはartifactもML runtimeも必要としない
- runtime identityは#79 A2でfreezeした次のdigestそのものである

```json
{
  "residual_scorer": "lisjong-offense-l0.3-constant-zero-residual-scorer-v1",
  "selection_policy": "lisjong-offense-l0.2-semantic-envelope-v1"
}
```
"""

from dataclasses import dataclass

from lisjong.learning._canonical import value_digest
from lisjong.learning.envelope_policy import (
    SEMANTIC_ENVELOPE_IDENTITY,
    SemanticEnvelopeOffensePolicy,
)

CONSTANT_RESIDUAL_SCORER_IDENTITY = (
    "lisjong-offense-l0.3-constant-zero-residual-scorer-v1"
)
"""candidateごとに有限の0.0を返すresidual scorerのidentity。"""

CONSTANT_RESIDUAL_RUNTIME_IDENTITY = value_digest(
    {
        "residual_scorer": CONSTANT_RESIDUAL_SCORER_IDENTITY,
        "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
    }
)
"""baseline runtime identity。source manifestの`baseline_runtime_identity`。"""


@dataclass(frozen=True, slots=True)
class ConstantResidualRuntime:
    """constant-zero residual scorerを使うsemantic-envelope Policy factory。"""

    @property
    def identity(self) -> str:
        return CONSTANT_RESIDUAL_RUNTIME_IDENTITY

    def score(
        self,
        context: tuple[float, ...],
        candidates: tuple[tuple[float, ...], ...],
    ) -> tuple[float, ...]:
        """full candidate tupleの各entryへ0.0を1つずつ返す。"""
        return tuple(0.0 for _ in candidates)

    def create_policy(self) -> SemanticEnvelopeOffensePolicy:
        """game / seatごとのfresh Policy instanceを返す。"""
        return SemanticEnvelopeOffensePolicy(self)

    def __call__(self) -> SemanticEnvelopeOffensePolicy:
        return self.create_policy()


__all__ = [
    "CONSTANT_RESIDUAL_RUNTIME_IDENTITY",
    "CONSTANT_RESIDUAL_SCORER_IDENTITY",
    "ConstantResidualRuntime",
]
