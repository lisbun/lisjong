"""L0.2 candidate scorerのoffline diagnosticsとfrozen offline gate。

Issue #189に対応する。Arena strength evaluationではない。source recordの
decisionへfrozen artifactのserving path（`LearnedCandidateOffensePolicy.
decide()`）をそのまま適用し、legality、O0 guard、semantic agreementを測る。

```text
TRAIN         optimization
SELECT        epoch / artifact candidate selection
freeze        model / encoding / second-step request policy / thresholds
OFFLINE-EVAL  qualificationのためにexactly once
```

thresholdは`OFFLINE_GATE`としてOFFLINE-EVALを見る前に固定し、identityと
fingerprintで記録する。結果を見た後に同じIssue内でthreshold / feature /
modelを変更しない。

## semantic agreement

learnerが選んだcandidateとteacherが選んだcandidateのsemantic quality
（#187 candidate feature値）を比較するguardrailである。NNがshantenを推論
できるかではなく、選択結果の牌効率semanticを測る。

```text
shanten-stage agreement        learner shanten == teacher shanten
                               （分母: scorer decision全体）
conditional current-ukeire     shanten一致decisionのうち ukeire一致
conditional second-step        shanten・ukeire一致かつteacher candidateが
                               second-step EVALUATEDのdecisionのうち score一致
```

upstream stageの不一致をlater stageの成功として数えない。candidate top-1
agreementは記録するがqualification thresholdにはしない（同等牌効率candidate
間のcanonical tie-break差をfailureと誤判定しないため）。
"""

import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from lisjong.learning._canonical import value_digest
from lisjong.learning._o0 import O0DecisionKind
from lisjong.learning.candidate_features import (
    DiscardCandidateFeatures,
    SecondStepStatus,
)
from lisjong.policy_contract import (
    DecisionContext,
    DiscardAction,
    PassAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)

OFFLINE_GATE_IDENTITY = "lisjong-offense-l0.2-offline-gate-v1"

READY = "SEMANTIC OFFENSE SCORER READY"
NOT_QUALIFIED = "SEMANTIC OFFENSE SCORER NOT QUALIFIED — OFFLINE"
INVALID = "STOP / INVALID"

OFFLINE_GATE = {
    "identity": OFFLINE_GATE_IDENTITY,
    "thresholds": {
        "legality": 1.0,
        "win_guard": 1.0,
        "riichi_guard": 1.0,
        "no_call_guard": 1.0,
        "shanten_stage_agreement": 0.99,
        "conditional_current_ukeire_agreement": 0.95,
        "conditional_second_step_agreement": 0.90,
    },
    "minimum_support": {
        "win_decisions": 30,
        "riichi_decisions": 60,
        "response_decisions": 100,
        "scorer_decisions": 1000,
        "conditional_current_ukeire_support": 500,
        "conditional_second_step_support": 250,
    },
    "diagnostic_only": ["candidate_top1_agreement"],
}
"""OFFLINE-EVALを見る前に固定したgate。変更する場合は新identityにする。"""

_THRESHOLD_SUPPORT = {
    "win_guard": "win_decisions",
    "riichi_guard": "riichi_decisions",
    "no_call_guard": "response_decisions",
    "shanten_stage_agreement": "scorer_decisions",
    "conditional_current_ukeire_agreement": "conditional_current_ukeire_support",
    "conditional_second_step_agreement": "conditional_second_step_support",
}


def offline_gate_fingerprint() -> str:
    return value_digest(OFFLINE_GATE)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(slots=True)
class SemanticAgreement:
    """teacher candidateとlearner candidateのstage別semantic agreement counter。"""

    scorer_decisions: int = 0
    shanten_agreement_count: int = 0
    ukeire_support: int = 0
    ukeire_agreement_count: int = 0
    second_step_support: int = 0
    second_step_agreement_count: int = 0

    def add(
        self, teacher: DiscardCandidateFeatures, learner: DiscardCandidateFeatures
    ) -> None:
        self.scorer_decisions += 1
        if learner.post_discard_shanten != teacher.post_discard_shanten:
            return
        self.shanten_agreement_count += 1
        self.ukeire_support += 1
        if learner.current_ukeire_count != teacher.current_ukeire_count:
            return
        self.ukeire_agreement_count += 1
        if teacher.second_step_status is not SecondStepStatus.EVALUATED:
            return
        if learner.second_step_status is not SecondStepStatus.EVALUATED:
            # 同shanten・同ukeireのteacherがfinalistならlearnerもfinalistで
            # あり、two-pass policy上ここへは到達しない。到達すれば
            # materializationの不整合である。
            raise ValueError("learner candidate lost second-step availability")
        self.second_step_support += 1
        if learner.second_step_ukeire_score == teacher.second_step_ukeire_score:
            self.second_step_agreement_count += 1

    def to_value(self) -> dict[str, float | int]:
        """count群と、分母が正のときだけrateを返す（有限値だけ）。"""
        value: dict[str, float | int] = {
            "scorer_decisions": self.scorer_decisions,
            "shanten_agreement_count": self.shanten_agreement_count,
            "conditional_current_ukeire_support": self.ukeire_support,
            "conditional_current_ukeire_agreement_count": self.ukeire_agreement_count,
            "conditional_second_step_support": self.second_step_support,
            "conditional_second_step_agreement_count": (
                self.second_step_agreement_count
            ),
        }
        rates = {
            "shanten_stage_agreement": _rate(
                self.shanten_agreement_count, self.scorer_decisions
            ),
            "conditional_current_ukeire_agreement": _rate(
                self.ukeire_agreement_count, self.ukeire_support
            ),
            "conditional_second_step_agreement": _rate(
                self.second_step_agreement_count, self.second_step_support
            ),
        }
        value.update({name: rate for name, rate in rates.items() if rate is not None})
        return value


def _expected_kind(decision: DecisionContext) -> O0DecisionKind:
    """O0 branchをlegal actionsから独立に再判定する（policy実装と別経路）。"""
    legal = decision.legal_actions
    if any(isinstance(action, (RonAction, TsumoAction)) for action in legal):
        return O0DecisionKind.WIN
    if any(isinstance(action, RiichiAction) for action in legal):
        return O0DecisionKind.RIICHI
    if any(isinstance(action, DiscardAction) for action in legal):
        return O0DecisionKind.DISCARD
    return O0DecisionKind.RESPONSE


_GUARD_TYPES = {
    O0DecisionKind.WIN: (RonAction, TsumoAction),
    O0DecisionKind.RIICHI: (RiichiAction,),
    O0DecisionKind.RESPONSE: (PassAction,),
    O0DecisionKind.DISCARD: (DiscardAction,),
}


@dataclass(slots=True)
class _KindCounter:
    decisions: int = 0
    correct: int = 0
    teacher_agreement: int = 0
    milliseconds: list[float] = field(default_factory=list)


def _runtime_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_ms": ordered[-1],
    }


def evaluate_candidate_policy(policy, source, splits: Iterable[str]) -> dict:
    """source recordの指定splitへpolicyのserving pathを適用しmetricsを返す。

    teacher labelはsource recordの`teacher_selected_action`だけを使う。
    """
    selected_splits = tuple(splits)
    if not selected_splits:
        raise ValueError("splits must not be empty")
    counters = {kind: _KindCounter() for kind in O0DecisionKind}
    semantic = SemanticAgreement()
    legal_count = 0
    total = 0
    top1 = 0
    for decision in source.decisions():
        if decision.split not in selected_splits:
            continue
        context = DecisionContext(
            input=decision.policy_input, legal_actions=decision.legal_actions
        )
        expected = _expected_kind(context)
        started = time.perf_counter()
        result = policy.decide(context)
        elapsed = (time.perf_counter() - started) * 1000.0

        total += 1
        legal_count += int(
            any(result.action is legal for legal in context.legal_actions)
        )
        counter = counters[expected]
        counter.decisions += 1
        counter.milliseconds.append(elapsed)
        counter.correct += int(
            result.kind is expected
            and isinstance(result.action, _GUARD_TYPES[expected])
        )
        counter.teacher_agreement += int(result.action == decision.selected_action)
        if expected is O0DecisionKind.DISCARD and result.candidates is not None:
            teacher = next(
                (
                    candidate
                    for candidate in result.candidates
                    if candidate.action == decision.selected_action
                ),
                None,
            )
            if teacher is None:
                raise ValueError(
                    "teacher selected action of a scorer decision is not a legal "
                    "discard candidate"
                )
            learner = result.selected_candidate
            semantic.add(teacher, learner)
            top1 += int(learner is teacher)

    if total == 0:
        raise ValueError(f"source record has no decisions in splits {selected_splits}")

    guard_rate = {
        kind: _rate(counters[kind].correct, counters[kind].decisions)
        for kind in O0DecisionKind
    }
    semantic_value = semantic.to_value()
    metrics: dict[str, object] = {
        "legality": legal_count / total,
        "win_guard": guard_rate[O0DecisionKind.WIN],
        "riichi_guard": guard_rate[O0DecisionKind.RIICHI],
        "no_call_guard": guard_rate[O0DecisionKind.RESPONSE],
        "discard_branch": guard_rate[O0DecisionKind.DISCARD],
        "shanten_stage_agreement": semantic_value.get("shanten_stage_agreement"),
        "conditional_current_ukeire_agreement": semantic_value.get(
            "conditional_current_ukeire_agreement"
        ),
        "conditional_second_step_agreement": semantic_value.get(
            "conditional_second_step_agreement"
        ),
        "candidate_top1_agreement": _rate(top1, semantic.scorer_decisions),
    }
    support = {
        "decisions": total,
        "win_decisions": counters[O0DecisionKind.WIN].decisions,
        "riichi_decisions": counters[O0DecisionKind.RIICHI].decisions,
        "response_decisions": counters[O0DecisionKind.RESPONSE].decisions,
        "scorer_decisions": semantic.scorer_decisions,
        "conditional_current_ukeire_support": semantic.ukeire_support,
        "conditional_second_step_support": semantic.second_step_support,
    }
    return {
        "metrics": metrics,
        "semantic_counts": semantic_value,
        "support": support,
        "teacher_agreement_by_branch": {
            kind.value: _rate(
                counters[kind].teacher_agreement, counters[kind].decisions
            )
            for kind in O0DecisionKind
        },
        "runtime_by_branch": {
            kind.value: _runtime_summary(counters[kind].milliseconds)
            for kind in O0DecisionKind
        },
        "splits": list(selected_splits),
    }


def classify_offline_result(evaluation: dict) -> dict:
    """frozen `OFFLINE_GATE`でterminal classificationを決める。

    support不足で評価できないgateは推測でPASSにせず、`STOP / INVALID`として
    理由を明記する。
    """
    thresholds = OFFLINE_GATE["thresholds"]
    minimum_support = OFFLINE_GATE["minimum_support"]
    metrics = evaluation["metrics"]
    support = evaluation["support"]

    insufficient = [
        {"support": name, "observed": support[name], "required": required}
        for name, required in minimum_support.items()
        if support[name] < required
    ]
    failures = []
    for name, required in thresholds.items():
        observed = metrics[name]
        support_name = _THRESHOLD_SUPPORT.get(name)
        if observed is None:
            continue
        if (
            support_name is not None
            and support[support_name] < (minimum_support[support_name])
        ):
            continue
        if observed < required:
            failures.append(
                {"metric": name, "observed": observed, "required": required}
            )

    if insufficient:
        outcome = INVALID
    elif failures:
        outcome = NOT_QUALIFIED
    else:
        outcome = READY
    return {
        "failures": failures,
        "gate": {"fingerprint": offline_gate_fingerprint(), **OFFLINE_GATE},
        "insufficient_support": insufficient,
        "outcome": outcome,
    }


__all__ = [
    "INVALID",
    "NOT_QUALIFIED",
    "OFFLINE_GATE",
    "OFFLINE_GATE_IDENTITY",
    "READY",
    "SemanticAgreement",
    "classify_offline_result",
    "evaluate_candidate_policy",
    "offline_gate_fingerprint",
]
