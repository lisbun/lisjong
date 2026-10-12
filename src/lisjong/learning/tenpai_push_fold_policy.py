"""単独リーチ下の聴牌PUSH/FOLD（H2）の実験候補Policy（lisbun/lisjong#289、推論側）。

設計は`docs/tenpai-push-fold-design.md`（#254）の2〜4節。`PlacementAwareSpeedCallPolicy`
（Champion）をexact parentとし、Championの決定C0を先に求める。ゲート条件1〜8を満たし、
降りる候補`a_fold`があり、比較式で`V(降りる) > V(押す)`のときだけ`a_fold`を切る。
(A)ではリーチせずに切る。それ以外はC0の行動を返す。

- ゲート条件1〜7と2候補は`evaluate_tenpai_gate()`、比較式は`comparison_breakdown()`を使う
  （対比較sourceの生成・表のfitと同じ関数）
- ゲートの外、降りる候補がない判断、表にbucketがない判断（条件8）では、Championの
  `PolicyDecision`をそのまま返す
- `V`を計算した判断では、押す場合も降りる場合も、内訳を`TenpaiPushFoldAnalysis`として返す
- #292のvalidで(A)と(B)(C)の両方が中止の条件を通過したので、両方の区分を対象にする

入力は`PolicyInput`と合法手だけである。待ちモデルと表はrepositoryの外のfileから読み、
SHA-256が登録値と違えば構築時にエラーにする。native拡張が使えない場合も構築時にエラーにする。
instanceは判断をまたぐ状態を持たない。未昇格の実験候補であり、Championは変えない。
"""

from dataclasses import dataclass
from pathlib import Path

from lisjong.hand_evaluation.scoring import evaluate_win, require_scoring_backend
from lisjong.learning.riichi_wait_estimator import LogisticWaitModel
from lisjong.learning.riichi_wait_mawashi_policy import (
    SELECTED_WAIT_MODEL_SHA256,
    load_selected_wait_model,
)
from lisjong.learning.tenpai_push_fold import (
    EvaluateWin,
    TenpaiGate,
    evaluate_tenpai_gate,
)
from lisjong.learning.tenpai_push_fold_value import (
    SELECTED_TABLES_SHA256,
    CandidateValue,
    PushFoldTables,
    comparison_breakdown,
    load_tables,
)
from lisjong.policies.placement_aware_speed_call import PlacementAwareSpeedCallPolicy
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_decision import PolicyDecision


@dataclass(frozen=True, slots=True)
class TenpaiPushFoldAnalysis(AnalysisTrace):
    """`V`を計算した判断の、ゲートの内容と2候補の値の内訳。"""

    gate: TenpaiGate
    push: CandidateValue
    fold: CandidateValue

    @property
    def fold_selected(self) -> bool:
        """`V(降りる) > V(押す)`か。同点は押す（C0）。"""
        return self.fold.value > self.push.value


class TenpaiPushFoldPolicy(PlacementAwareSpeedCallPolicy):
    """#289: 単独リーチ下の聴牌で、押す行動と降りる打牌を`V`で比べる版（未昇格の実験候補）。"""

    def __init__(
        self,
        model: LogisticWaitModel,
        tables: PushFoldTables,
        *,
        evaluate: EvaluateWin | None = None,
    ) -> None:
        if not isinstance(model, LogisticWaitModel):
            raise TypeError("model must be a LogisticWaitModel")
        if not isinstance(tables, PushFoldTables):
            raise TypeError("tables must be a PushFoldTables")
        if evaluate is None:
            require_scoring_backend()
            evaluate = evaluate_win
        super().__init__()
        self._model = model
        self._tables = tables
        self._evaluate = evaluate

    @classmethod
    def from_files(
        cls,
        selection_path: Path,
        tables_path: Path,
        *,
        expected_selection_sha256: str = SELECTED_WAIT_MODEL_SHA256,
        expected_tables_sha256: str = SELECTED_TABLES_SHA256,
    ) -> "TenpaiPushFoldPolicy":
        return cls(
            load_selected_wait_model(
                selection_path, expected_sha256=expected_selection_sha256
            ),
            load_tables(tables_path, expected_sha256=expected_tables_sha256),
        )

    def _decide(self, decision: DecisionContext) -> PolicyDecision:
        c0_decision = super()._decide(decision)
        gate = evaluate_tenpai_gate(
            decision, c0_decision, self._model, evaluate=self._evaluate
        )
        if gate is None or not gate.has_fold_candidate:
            return c0_decision
        breakdown = comparison_breakdown(decision.input, gate, self._tables)
        if breakdown is None:
            return c0_decision
        analysis = TenpaiPushFoldAnalysis(
            gate=gate, push=breakdown[0], fold=breakdown[1]
        )
        return PolicyDecision(
            action=gate.fold_action if analysis.fold_selected else c0_decision.action,
            analysis=analysis,
        )


__all__ = ["TenpaiPushFoldAnalysis", "TenpaiPushFoldPolicy"]
