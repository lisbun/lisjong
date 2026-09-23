"""L0.2a semantic-envelope policyのbounded replay diagnostics（Issue #191）。

#189 OFFLINE-EVALを新しいholdoutとして再利用しない。既観測のTRAIN / SELECT
splitへ`SemanticEnvelopeOffensePolicy.decide()`をそのまま適用し、次を記録する。

```text
construction invariants（qualification gate）
    legality / win / riichi / no-call guard       1.000
    shanten regret                                0
    current-ukeire regret（min-shanten内）         0
    second-step regret（S3適用時）                  0
    reference oracle（指定時）                      constant-scorer選択と
                                                  reference actionがobject一致
    reference disagreement outside survivors      0

residual-choice diagnostics（gateではない）
    survivor count distribution / survivor >= 2の割合
    scorer invoked割合 / scorerがcanonical先頭以外を選んだ割合
    residual survivorのsample（tile identityの列挙）
    runtime
```

regretは`semantic_envelope_survivors()`を使わず、candidate semantic値から
このmoduleで独立に再計算する。reference policy（通常`TwoStepUkeirePolicy`）は
callerが注入する。learning packageはconcrete Policy moduleへ依存しない。

teacher top-1 agreementはqualification gateにしない。
"""

import statistics
import time
from collections import Counter
from collections.abc import Iterable

from lisjong.learning._canonical import value_digest
from lisjong.learning._o0 import O0DecisionKind
from lisjong.learning.candidate_features import (
    DiscardCandidateFeatures,
    SecondStepStatus,
)
from lisjong.learning.envelope_policy import SEMANTIC_ENVELOPE_IDENTITY
from lisjong.policy_contract import (
    DecisionContext,
    DiscardAction,
    PassAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.tile import TileCategory

ENVELOPE_READY = "SEMANTIC OFFENSE ENVELOPE READY"
ENVELOPE_INVALID = "SEMANTIC ENVELOPE INVALID"

_GUARD_TYPES = {
    O0DecisionKind.WIN: (RonAction, TsumoAction),
    O0DecisionKind.RIICHI: (RiichiAction,),
    O0DecisionKind.RESPONSE: (PassAction,),
    O0DecisionKind.DISCARD: (DiscardAction,),
}
_SUIT_LETTER = {
    TileCategory.MANZU: "m",
    TileCategory.PINZU: "p",
    TileCategory.SOUZU: "s",
    TileCategory.HONOR: "z",
}


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


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


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


def _describe(action: DiscardAction) -> str:
    """sample記録用の短いtile identity表記（例: `0m`=赤5萬、`*`=ツモ切り）。"""
    tile = action.tile
    rank = "0" if tile.is_red else str(tile.tile_type.rank)
    suffix = "*" if action.tsumogiri else ""
    return f"{rank}{_SUIT_LETTER[tile.tile_type.category]}{suffix}"


def _regrets(
    candidates: tuple[DiscardCandidateFeatures, ...],
    selected: DiscardCandidateFeatures,
) -> tuple[int, int | None, int | None]:
    """選択candidateのshanten / ukeire / second-step regretを独立に計算する。

    ukeire regretはselectedがminimum shantenにあるときだけ、second-step regretは
    S3適用条件（minimum shanten >= 1かつmax-ukeire finalist >= 2）で、selectedが
    finalistであるときだけ定義する。upstreamで外れた場合は`None`とし、
    upstream regretで検出する。
    """
    minimum_shanten = min(item.post_discard_shanten for item in candidates)
    shanten_regret = selected.post_discard_shanten - minimum_shanten
    if shanten_regret:
        return shanten_regret, None, None
    level = [c for c in candidates if c.post_discard_shanten == minimum_shanten]
    maximum_ukeire = max(item.current_ukeire_count for item in level)
    ukeire_regret = maximum_ukeire - selected.current_ukeire_count
    finalists = [c for c in level if c.current_ukeire_count == maximum_ukeire]
    if ukeire_regret or minimum_shanten < 1 or len(finalists) < 2:
        return shanten_regret, ukeire_regret, None
    scores = []
    for item in finalists:
        if item.second_step_status is not SecondStepStatus.EVALUATED:
            raise ValueError("S3 finalist lacks an evaluated second-step score")
        scores.append(item.second_step_ukeire_score)
    return (
        shanten_regret,
        ukeire_regret,
        max(scores) - (selected.second_step_ukeire_score),
    )


def evaluate_semantic_envelope_policy(
    policy,
    source,
    splits: Iterable[str],
    *,
    sample_every: int = 1,
    reference=None,
    residual_sample_limit: int = 12,
) -> dict:
    """source recordの指定splitへenvelope policyのserving pathを適用する。

    `sample_every`はsplit内decisionのdeterministic stride（1なら全件）である。
    `reference`は`choose_action(decision)`を持つPolicyで、指定時だけoracle
    比較を記録する。
    """
    selected_splits = tuple(splits)
    if not selected_splits:
        raise ValueError("splits must not be empty")
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")

    decisions_by_kind: Counter = Counter()
    guard_correct: Counter = Counter()
    milliseconds: dict[O0DecisionKind, list[float]] = {
        kind: [] for kind in O0DecisionKind
    }
    legal_count = 0
    total = 0
    seen = 0
    teacher_agreement = 0

    scorer_decisions = 0
    survivor_counts: Counter = Counter()
    scorer_invoked = 0
    non_canonical_first = 0
    residual_single_tile_type = 0
    residual_samples: list[dict[str, object]] = []
    max_regret = {"shanten": 0, "current_ukeire": 0, "second_step": 0}
    regret_support = {"shanten": 0, "current_ukeire": 0, "second_step": 0}
    regret_violations = {"shanten": 0, "current_ukeire": 0, "second_step": 0}

    reference_decisions = 0
    reference_oracle_agreement = 0
    reference_policy_agreement = 0
    reference_disagreement_outside_survivors = 0

    for decision in source.decisions():
        if decision.split not in selected_splits:
            continue
        seen += 1
        if (seen - 1) % sample_every:
            continue
        context = DecisionContext(
            input=decision.policy_input, legal_actions=decision.legal_actions
        )
        expected = _expected_kind(context)
        started = time.perf_counter()
        result = policy.decide(context)
        elapsed = (time.perf_counter() - started) * 1000.0

        total += 1
        decisions_by_kind[expected] += 1
        milliseconds[expected].append(elapsed)
        legal_count += int(
            any(result.action is legal for legal in context.legal_actions)
        )
        guard_correct[expected] += int(
            result.kind is expected
            and isinstance(result.action, _GUARD_TYPES[expected])
        )
        teacher_agreement += int(result.action == decision.selected_action)

        oracle_action = result.action
        if expected is O0DecisionKind.DISCARD and result.candidates is not None:
            scorer_decisions += 1
            candidates = result.candidates
            survivors = result.survivors
            oracle_action = candidates[survivors[0]].action
            survivor_counts[len(survivors)] += 1
            scorer_invoked += int(result.scorer_invoked)
            selected = result.selected_candidate
            if len(survivors) >= 2:
                non_canonical_first += int(selected.action is not oracle_action)
                tile_types = {candidates[i].action.tile.tile_type for i in survivors}
                residual_single_tile_type += int(len(tile_types) == 1)
                if len(residual_samples) < residual_sample_limit:
                    residual_samples.append(
                        {
                            "split": decision.split,
                            "survivors": [
                                _describe(candidates[i].action) for i in survivors
                            ],
                            "selected": _describe(selected.action),
                        }
                    )
            regrets = _regrets(candidates, selected)
            for name, value in zip(
                ("shanten", "current_ukeire", "second_step"), regrets, strict=True
            ):
                if value is None:
                    continue
                regret_support[name] += 1
                max_regret[name] = max(max_regret[name], value)
                regret_violations[name] += int(value != 0)

        if reference is not None:
            reference_decisions += 1
            reference_action = reference.choose_action(context)
            reference_oracle_agreement += int(reference_action is oracle_action)
            reference_policy_agreement += int(reference_action is result.action)
            if reference_action is not result.action:
                # 不一致はpolicy側・reference側の両actionがsemantic survivor
                # 集合内にあるときだけ許容する。guard branchに許容範囲はない。
                survivor_actions = (
                    ()
                    if result.survivors is None
                    else tuple(result.candidates[i].action for i in result.survivors)
                )
                if not (
                    any(action is result.action for action in survivor_actions)
                    and any(action is reference_action for action in survivor_actions)
                ):
                    reference_disagreement_outside_survivors += 1

    if total == 0:
        raise ValueError(f"source record has no decisions in splits {selected_splits}")

    guard_rate = {
        kind: _rate(guard_correct[kind], decisions_by_kind[kind])
        for kind in O0DecisionKind
    }
    residual = sum(count for size, count in survivor_counts.items() if size >= 2)
    evaluation: dict[str, object] = {
        "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
        "splits": list(selected_splits),
        "sample_every": sample_every,
        "invariants": {
            "legality": legal_count / total,
            "win_guard": guard_rate[O0DecisionKind.WIN],
            "riichi_guard": guard_rate[O0DecisionKind.RIICHI],
            "no_call_guard": guard_rate[O0DecisionKind.RESPONSE],
            "discard_branch": guard_rate[O0DecisionKind.DISCARD],
            "max_regret": max_regret,
            "regret_violations": regret_violations,
            "regret_support": regret_support,
        },
        "support": {
            "decisions": total,
            "win_decisions": decisions_by_kind[O0DecisionKind.WIN],
            "riichi_decisions": decisions_by_kind[O0DecisionKind.RIICHI],
            "response_decisions": decisions_by_kind[O0DecisionKind.RESPONSE],
            "scorer_decisions": scorer_decisions,
        },
        "residual_choice": {
            "survivor_count_distribution": {
                str(size): survivor_counts[size] for size in sorted(survivor_counts)
            },
            "fraction_survivor_count_ge_2": _rate(residual, scorer_decisions),
            "fraction_survivor_count_eq_1": _rate(survivor_counts[1], scorer_decisions),
            "fraction_scorer_invoked": _rate(scorer_invoked, scorer_decisions),
            "fraction_non_canonical_first_given_residual": _rate(
                non_canonical_first, residual
            ),
            "fraction_single_tile_type_given_residual": _rate(
                residual_single_tile_type, residual
            ),
            "samples": residual_samples,
        },
        "teacher_agreement": _rate(teacher_agreement, total),
        "runtime_by_branch": {
            kind.value: _runtime_summary(milliseconds[kind]) for kind in O0DecisionKind
        },
    }
    if reference is not None:
        evaluation["reference"] = {
            "identity": f"{type(reference).__module__}.{type(reference).__qualname__}",
            "decisions": reference_decisions,
            "constant_scorer_oracle_agreement": _rate(
                reference_oracle_agreement, reference_decisions
            ),
            "policy_agreement": _rate(reference_policy_agreement, reference_decisions),
            "disagreement_outside_survivors": (
                reference_disagreement_outside_survivors
            ),
        }
    return evaluation


def classify_semantic_envelope_result(evaluation: dict) -> dict:
    """construction invariantからterminal classificationを決める。

    thresholdはすべてexact（1.0 / regret 0 / 不一致0）である。supportが0の
    guardは推測でPASSにせずfailureとして明記する。
    """
    invariants = evaluation["invariants"]
    failures = []
    for name in ("legality", "win_guard", "riichi_guard", "no_call_guard"):
        observed = invariants[name]
        if observed != 1.0:
            failures.append({"invariant": name, "observed": observed, "required": 1.0})
    if evaluation["support"]["scorer_decisions"] == 0:
        failures.append({"invariant": "scorer_decisions", "observed": 0})
    for name, count in invariants["regret_violations"].items():
        if count:
            failures.append(
                {
                    "invariant": f"{name}_regret",
                    "violations": count,
                    "max_regret": invariants["max_regret"][name],
                }
            )
    reference = evaluation.get("reference")
    if reference is not None:
        if reference["constant_scorer_oracle_agreement"] != 1.0:
            failures.append(
                {
                    "invariant": "constant_scorer_oracle_agreement",
                    "observed": reference["constant_scorer_oracle_agreement"],
                    "required": 1.0,
                }
            )
        if reference["disagreement_outside_survivors"]:
            failures.append(
                {
                    "invariant": "reference_disagreement_outside_survivors",
                    "observed": reference["disagreement_outside_survivors"],
                    "required": 0,
                }
            )
    criteria = {
        "exact_guards": ["legality", "win_guard", "riichi_guard", "no_call_guard"],
        "zero_regret": ["shanten", "current_ukeire", "second_step"],
        "reference_oracle": "constant-scorer selection is the reference action",
        "reference_disagreement": "only within semantic survivors",
        "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
    }
    return {
        "criteria": {"fingerprint": value_digest(criteria), **criteria},
        "failures": failures,
        "outcome": ENVELOPE_INVALID if failures else ENVELOPE_READY,
    }


__all__ = [
    "ENVELOPE_INVALID",
    "ENVELOPE_READY",
    "classify_semantic_envelope_result",
    "evaluate_semantic_envelope_policy",
]
