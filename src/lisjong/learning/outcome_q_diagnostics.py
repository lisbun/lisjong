"""L0.3 outcome-Q runtimeのcheap serving qualification（lisjong-project#79 §10）。

formal Arena strength evaluationの前に、development data（focal outcome sourceの
指定split）へserving pathをそのまま適用し、safety / integration gateと
Q診断値を記録する。strengthの証明ではない。

```text
serving（gate、#191 envelope diagnosticsをそのまま再利用）
    legality / win / immediate-riichi / O0 no-call guard   1.000
    shanten / current-ukeire / second-step regret            0
    reference oracle（指定時）                              constant-scorer
                                                            選択 == reference action
    reference disagreement outside survivors                0

outcome-Q（gate）
    selected action inside #191 survivors（eligible row）   1.000
    Policy選択 == survivor内Q argmax（canonical tie-break）  不一致0
    eligible row                                            >= 1

outcome-Q（診断値、gateではない）
    Q Policy != canonical-first baselineの割合
    survivor内Q spread
    behavior-selected action上のQ MSE（SELECTならSELECT Q loss）
    prediction / target calibration（prediction分位bin）
    runtime（serving block）
```

非有限score・score数不一致・resolve failureはPolicy / runtimeが例外で
fail closedするため、evaluationは完了しない（failure数0以外は結果を返さない）。

`serving.teacher_agreement`はここではbehavior-selected action（exploration）との
一致率であり、gateにも強さの指標にもしない。
"""

import dataclasses
import statistics
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isfinite

from lisjong.learning._canonical import value_digest
from lisjong.learning.candidate_encoding import encode_candidates
from lisjong.learning.envelope_diagnostics import (
    classify_semantic_envelope_result,
    evaluate_semantic_envelope_policy,
)
from lisjong.learning.errors import LearnedPolicyError
from lisjong.learning.features import build_player_safe_feature
from lisjong.learning.outcome_q_policy import OutcomeQRuntime
from lisjong.learning.outcome_source import FocalOutcomeSource, build_outcome_targets
from lisjong.policy_contract import InternalAction, PolicyInput

OUTCOME_Q_SERVING_QUALIFIED = "OUTCOME-Q SERVING QUALIFIED"
OUTCOME_Q_SERVING_INVALID = "OUTCOME-Q SERVING INVALID"

DEFAULT_CALIBRATION_BINS = 10


@dataclass(frozen=True, slots=True)
class _ReplayDecision:
    split: str
    policy_input: PolicyInput
    legal_actions: tuple[InternalAction, ...]
    selected_action: InternalAction


@dataclass(frozen=True, slots=True)
class _FocalReplaySource:
    """focal decision列を#191 envelope diagnosticsが読むdecision形へ写す。"""

    source: FocalOutcomeSource

    def decisions(self) -> Iterator[_ReplayDecision]:
        for game, item in self.source.decisions():
            yield _ReplayDecision(
                split=game.split,
                policy_input=item.decision.input,
                legal_actions=item.decision.legal_actions,
                selected_action=item.selected_action,
            )


def _calibration(pairs: list[tuple[float, float]], bins: int) -> list[dict]:
    """prediction順に等件数binへ分け、binごとの平均prediction / targetを返す。"""
    ordered = sorted(pairs)
    count = len(ordered)
    bins = min(bins, count)
    result = []
    for index in range(bins):
        chunk = ordered[index * count // bins : (index + 1) * count // bins]
        result.append(
            {
                "count": len(chunk),
                "mean_prediction": statistics.fmean(p for p, _ in chunk),
                "mean_target": statistics.fmean(t for _, t in chunk),
            }
        )
    return result


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def evaluate_outcome_q_policy(
    runtime: OutcomeQRuntime,
    source: FocalOutcomeSource,
    splits: Iterable[str],
    *,
    reference=None,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict:
    """指定splitのfocal decisionへQ runtimeのserving pathを適用する。

    `reference`は`choose_action(decision)`を持つPolicy（通常
    `TwoStepUkeirePolicy`）で、callerが注入する。
    """
    if not isinstance(runtime, OutcomeQRuntime):
        raise LearnedPolicyError("runtime must be an OutcomeQRuntime")
    if not isinstance(source, FocalOutcomeSource):
        raise LearnedPolicyError("source must be a strict-read FocalOutcomeSource")
    selected_splits = tuple(splits)
    if not selected_splits:
        raise ValueError("splits must not be empty")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be >= 1")

    policy = runtime()
    serving = evaluate_semantic_envelope_policy(
        policy, _FocalReplaySource(source), selected_splits, reference=reference
    )

    targets = build_outcome_targets(
        dataclasses.replace(
            source,
            games=tuple(g for g in source.games if g.split in selected_splits),
        )
    )
    pairs: list[tuple[float, float]] = []
    spreads: list[float] = []
    inside = 0
    changed = 0
    argmax_mismatch = 0
    for row in targets.rows:
        scores = runtime.score(
            build_player_safe_feature(row.decision.input),
            encode_candidates(row.candidates),
        )
        if len(scores) != len(row.candidates) or any(
            not isfinite(value) for value in scores
        ):
            raise LearnedPolicyError("Q runtime produced an invalid score tuple")
        best = row.survivors[0]
        for index in row.survivors[1:]:
            if scores[index] > scores[best]:
                best = index
        result = policy.decide(row.decision)
        survivor_actions = [row.candidates[i].action for i in row.survivors]
        inside += int(any(result.action is action for action in survivor_actions))
        argmax_mismatch += int(result.action is not row.candidates[best].action)
        changed += int(best != row.survivors[0])
        survivor_scores = [scores[i] for i in row.survivors]
        spreads.append(max(survivor_scores) - min(survivor_scores))
        pairs.append((scores[row.selected_candidate_index], row.target_q))

    eligible = len(targets.rows)
    q_block: dict[str, object] = {
        "argmax_mismatch": argmax_mismatch,
        "behavior_action_mse": statistics.fmean((p - t) ** 2 for p, t in pairs)
        if pairs
        else None,
        "calibration": _calibration(pairs, calibration_bins),
        "eligible_rows": eligible,
        "fraction_q_not_canonical_first": changed / eligible if eligible else None,
        "prediction_mean": statistics.fmean(p for p, _ in pairs) if pairs else None,
        "selected_inside_survivors": inside / eligible if eligible else None,
        "survivor_q_spread": _summary(spreads),
        "target_mean": statistics.fmean(t for _, t in pairs) if pairs else None,
    }
    return {
        "artifact_identity": runtime.artifact_identity,
        "outcome_q": q_block,
        "runtime_identity": runtime.identity,
        "serving": serving,
        "source_identity": source.identity,
        "splits": list(selected_splits),
    }


def classify_outcome_q_serving_result(evaluation: dict) -> dict:
    """serving gateとoutcome-Q gateからterminal classificationを決める。"""
    serving = classify_semantic_envelope_result(evaluation["serving"])
    failures = list(serving["failures"])
    q_block = evaluation["outcome_q"]
    if q_block["eligible_rows"] == 0:
        failures.append({"invariant": "eligible_rows", "observed": 0})
    elif q_block["selected_inside_survivors"] != 1.0:
        failures.append(
            {
                "invariant": "selected_inside_survivors",
                "observed": q_block["selected_inside_survivors"],
                "required": 1.0,
            }
        )
    if q_block["argmax_mismatch"]:
        failures.append(
            {
                "invariant": "policy_matches_survivor_q_argmax",
                "observed": q_block["argmax_mismatch"],
                "required": 0,
            }
        )
    criteria = {
        "serving": serving["criteria"]["fingerprint"],
        "selected_inside_survivors": 1.0,
        "policy_matches_survivor_q_argmax": "canonical-first tie-break",
        "eligible_rows": ">= 1",
    }
    return {
        "criteria": {"fingerprint": value_digest(criteria), **criteria},
        "failures": failures,
        "outcome": OUTCOME_Q_SERVING_INVALID
        if failures
        else OUTCOME_Q_SERVING_QUALIFIED,
    }


__all__ = [
    "DEFAULT_CALIBRATION_BINS",
    "OUTCOME_Q_SERVING_INVALID",
    "OUTCOME_Q_SERVING_QUALIFIED",
    "classify_outcome_q_serving_result",
    "evaluate_outcome_q_policy",
]
