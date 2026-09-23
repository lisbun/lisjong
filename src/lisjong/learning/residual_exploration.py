"""L0.3 focal residual exploration selector（Issue #193、#79 A3）。

source生成でfocal seatだけが使う、generation専用のpure functionである。
serving `Policy`（`choose_action(decision)`）は実装しない。

```text
(DecisionContext, exploration_token)
    -> exploration_tokenを検証（lowercase 64-hex、違えばfail closed）
    -> O0 precedence（lisjong.learning._o0、#189 / #191と同一）
         WIN / RIICHI / RESPONSE   deterministic_guard_action()（tokenは使わない）
         DISCARD
            -> build_scorer_candidates()
            -> semantic_envelope_survivors()（#191、canonical順）
                 1件   そのsurvivor
                 k>=2  bucket = int(exploration_token, 16) % k
                       survivors[bucket]
```

固定する原則。

- ambient / global randomnessや可変のPRNG streamを持たない。同じ入力には
  常に同じ結果を返す
- tokenの導出（SHA-256 / focal decision ordinal管理）はArenaが所有する
  （`lisjong-arena-l0.3-focal-decision-token-sha256-v1`）。lisjongは導出を
  再実装しない
- `envelope_policy.py`はPRNGを使わない契約を維持するため、本moduleは別module
  とする。envelope / O0 / candidate構築は#191と同じ関数を呼ぶだけである
- actionを再構築せず、`decision.legal_actions`側のobjectをそのまま返す
- 返り値はArenaが監査用のsurvivor action列を同じ1回の計算から記録できるよう、
  full candidate tuple、survivor index、選択index、bucketを含む
- outcome source consumer（`outcome_source.py`）は同じ関数で選択を再計算・
  照合する
"""

from dataclasses import dataclass

from lisjong.learning._canonical import expect_digest
from lisjong.learning._o0 import (
    O0DecisionKind,
    classify_o0_decision,
    deterministic_guard_action,
)
from lisjong.learning.candidate_encoding import build_scorer_candidates
from lisjong.learning.candidate_features import DiscardCandidateFeatures
from lisjong.learning.envelope_policy import semantic_envelope_survivors
from lisjong.learning.errors import LearnedPolicyError
from lisjong.policy_contract import DecisionContext, InternalAction

RESIDUAL_EXPLORATION_BEHAVIOR_IDENTITY = (
    "lisjong-offense-l0.3-focal-uniform-residual-exploration-v1"
)
"""focal-only uniform residual exploration behaviorのidentity（#79 A3）。

このplain identity自体が、hash-to-bucket rule
（`survivors[int(exploration_token, 16) % len(survivors)]`）と#191
`SEMANTIC_ENVELOPE_IDENTITY`のsemanticsを表す契約である。source manifestの
`exploration_behavior_identity`にはこの文字列をそのまま記録する。
"""


@dataclass(frozen=True, slots=True)
class ResidualExplorationDecision:
    """1 focal decisionの選択結果。

    `candidates` / `survivors` / `selected_candidate_index`はDISCARD branch
    だけが持ち、guard branchでは`None`である。`bucket`はsurvivorが2件以上で
    tokenを使ったときだけ持つ。
    """

    kind: O0DecisionKind
    action: InternalAction
    candidates: tuple[DiscardCandidateFeatures, ...] | None = None
    survivors: tuple[int, ...] | None = None
    selected_candidate_index: int | None = None
    bucket: int | None = None

    @property
    def survivor_actions(self) -> tuple[InternalAction, ...] | None:
        """canonical順のsurvivor action列（DISCARD以外は`None`）。"""
        if self.candidates is None or self.survivors is None:
            return None
        return tuple(self.candidates[index].action for index in self.survivors)


def validate_exploration_token(token: object) -> str:
    """exploration tokenがlowercase 64-hexであることを検証する。"""
    return expect_digest(token, LearnedPolicyError, "exploration_token")


def select_residual_exploration(
    decision: DecisionContext, exploration_token: str
) -> ResidualExplorationDecision:
    """O0 guardとsemantic envelopeの内側で、tokenからresidual actionを選ぶ。"""
    if not isinstance(decision, DecisionContext):
        raise LearnedPolicyError("decision must be a DecisionContext")
    token = validate_exploration_token(exploration_token)

    kind = classify_o0_decision(decision)
    if kind is not O0DecisionKind.DISCARD:
        return ResidualExplorationDecision(
            kind=kind, action=deterministic_guard_action(decision, kind)
        )

    candidates = build_scorer_candidates(decision)
    survivors = semantic_envelope_survivors(candidates)
    bucket = None
    if len(survivors) == 1:
        selected = survivors[0]
    else:
        bucket = int(token, 16) % len(survivors)
        selected = survivors[bucket]
    action = candidates[selected].action
    if not any(action is legal for legal in decision.legal_actions):
        raise LearnedPolicyError(
            "selected candidate is not a canonical legal action object"
        )
    return ResidualExplorationDecision(
        kind=kind,
        action=action,
        candidates=candidates,
        survivors=survivors,
        selected_candidate_index=selected,
        bucket=bucket,
    )


__all__ = [
    "RESIDUAL_EXPLORATION_BEHAVIOR_IDENTITY",
    "ResidualExplorationDecision",
    "select_residual_exploration",
    "validate_exploration_token",
]
