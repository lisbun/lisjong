"""R1価値モデルのgate Policy（Issue #360、設計 #356）。Championをexact parentとする実験候補。

```text
DecisionContext
    -> Champion（PlacementAwareSpeedCallPolicy）の行動
    -> RiichiActionでなければそのまま返す
    -> 推定宣言牌（project_declaration）・解析・19列・予測
    -> dama_choice = 推定宣言牌（牌・赤・ツモ切りflag）かつ 予測 > 0 のときだけ、その牌をリーチせずに打つ
    -> それ以外（不一致・予測 <= 0・解析できない）はChampionのRiichiActionを返す
```

- 入力は判断の`DecisionContext`とartifactだけ。判断をまたぐ状態を持たない。閾値0は固定
- 未昇格の実験候補で、Championは変えない
"""

from lisjong.learning.r1_value import (
    R1ValueArtifact,
    R1ValueError,
    analyze,
    encode,
    project_declaration,
)
from lisjong.policies import PlacementAwareSpeedCallPolicy
from lisjong.policy_contract import DecisionContext, InternalAction, execute_policy
from lisjong.policy_contract.action import RiichiAction

THRESHOLD = 0.0


class R1ValueGatePolicy:
    """ChampionがリーチするときだけR1価値の予測で同じ宣言牌のダマへ替えるPolicy。"""

    def __init__(self, artifact: R1ValueArtifact) -> None:
        if not isinstance(artifact, R1ValueArtifact):
            raise TypeError("artifact must be an R1ValueArtifact")
        self._artifact = artifact

    @property
    def artifact(self) -> R1ValueArtifact:
        return self._artifact

    def gate(self, decision: DecisionContext) -> InternalAction | None:
        """介入するときの打牌、介入しないときNone（Championがリーチする判断が前提）。"""
        try:
            declaration = project_declaration(decision)
            features = analyze(decision, declaration)
            if features.dama_choice != declaration:
                return None
            prediction = self._artifact.predict(encode(decision, features))
        except R1ValueError:
            return None
        return declaration if prediction > THRESHOLD else None

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        if not isinstance(decision, DecisionContext):
            raise TypeError("decision must be a DecisionContext")
        action = execute_policy(PlacementAwareSpeedCallPolicy(), decision)
        if not isinstance(action, RiichiAction):
            return action
        replacement = self.gate(decision)
        return action if replacement is None else replacement


__all__ = ["THRESHOLD", "R1ValueGatePolicy"]
