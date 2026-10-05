"""#245の待ち推定を危険度に使う、攻撃制限付き回し打ちの実験候補（lisbun/lisjong#249）。

ゲート・攻撃制限・ロンされない牌の扱い・同点処理は
`lisjong.policies.attack_limited_mawashi_placement_aware_speed_call`と同じで、危険度の生の値
だけを、リーチ者の**構造的待ち確率**（`LogisticWaitModel`の出力）に替える。この値は
放銃確率・期待放銃点ではなく、候補牌の相対的な順位付けにだけ使う。現物・スジで生の値を
上書きしない。

推論入力は`PolicyInput`だけである。モデルは#245で選択を固定した`selection.json`から読み、
ファイルのSHA-256が固定値と一致しない場合は読み込まない（別のモデルへ黙って替えない）。
モデルのファイルはrepositoryの外に置く。ML runtimeには依存しない。
"""

import hashlib
import json
from pathlib import Path

from lisjong.learning.errors import LearningError
from lisjong.learning.riichi_wait_estimator import (
    CLIP_EPSILON,
    FEATURE_SET,
    LogisticWaitModel,
)
from lisjong.policies.attack_limited_mawashi_placement_aware_speed_call import (
    decide_attack_limited_mawashi,
)
from lisjong.policies.placement_aware_speed_call import PlacementAwareSpeedCallPolicy
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput

SELECTED_WAIT_MODEL_SHA256 = (
    "14475264d7fe4137a9ac8a23ee1d27b420bff2f4434c6e93ca72ecfcf24ccc38"
)
"""#245の正式評価で使った`selection.json`のSHA-256。"""

# `lisjong.learning.riichi_wait_evaluation.SELECTION_SCHEMA`と同じ値。評価側（ラベルを扱う）を
# 推論経路からimportしないため、ここに書く。
_SELECTION_SCHEMA = "lisjong-riichi-wait-selection-v1"


class RiichiWaitModelError(LearningError):
    """待ち推定モデルの`selection.json`を、固定した条件どおりに読めない場合。"""


def load_selected_wait_model(
    path: Path, *, expected_sha256: str = SELECTED_WAIT_MODEL_SHA256
) -> LogisticWaitModel:
    """`selection.json`から推定器を読む。hash・schema・特徴量セット・clip幅が違えば拒否する。"""
    content = Path(path).read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected_sha256:
        raise RiichiWaitModelError(
            f"selection SHA-256 {actual} differs from the expected {expected_sha256}"
        )
    try:
        document = json.loads(content)
        if document["schema"] != _SELECTION_SCHEMA:
            raise RiichiWaitModelError("not a riichi wait selection document")
        if (
            document["feature_set"] != FEATURE_SET
            or document["clip_epsilon"] != CLIP_EPSILON
        ):
            raise RiichiWaitModelError(
                "the selection's feature set or clip width differs from this code's"
            )
        estimator = document["models"]["estimator_logistic"]
        weights = tuple(
            (str(name), float(weight)) for name, weight in estimator["weights"].items()
        )
        return LogisticWaitModel(weights=weights, feature_set=estimator["feature_set"])
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise RiichiWaitModelError(f"malformed selection document: {error}") from error


class RiichiWaitAttackLimitedMawashiPolicy(PlacementAwareSpeedCallPolicy):
    """#249: 危険度に構造的待ち確率の推定を使う版（未昇格の実験候補）。"""

    def __init__(self, model: LogisticWaitModel) -> None:
        if not isinstance(model, LogisticWaitModel):
            raise TypeError("model must be a LogisticWaitModel")
        super().__init__()
        self._model = model

    @classmethod
    def from_selection(
        cls, path: Path, *, expected_sha256: str = SELECTED_WAIT_MODEL_SHA256
    ) -> "RiichiWaitAttackLimitedMawashiPolicy":
        return cls(load_selected_wait_model(path, expected_sha256=expected_sha256))

    def _raw_danger(self, policy_input: PolicyInput, riichi_seat: int):
        return self._model.predict(policy_input)

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        return decide_attack_limited_mawashi(
            policy_input,
            discard_actions,
            super()._decide_discard(policy_input, discard_actions),
            self._raw_danger,
        )


__all__ = [
    "SELECTED_WAIT_MODEL_SHA256",
    "RiichiWaitAttackLimitedMawashiPolicy",
    "RiichiWaitModelError",
    "load_selected_wait_model",
]
