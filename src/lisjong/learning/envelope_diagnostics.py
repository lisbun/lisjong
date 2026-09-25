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
import sys
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

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


def _selected_splits(splits: Iterable[str], sample_every: int) -> tuple[str, ...]:
    selected_splits = tuple(splits)
    if not selected_splits:
        raise ValueError("splits must not be empty")
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")
    return selected_splits


def _reference_identity(reference) -> str:
    return f"{type(reference).__module__}.{type(reference).__qualname__}"


def _regret_names() -> dict[str, int]:
    return {"shanten": 0, "current_ukeire": 0, "second_step": 0}


@dataclass(slots=True)
class _Tally:
    """replay集計のmerge可能な中間値（Issue #209）。

    整数counter、max regret、decision順のruntime列、global decision index付きの
    residual sampleだけを持つ。rateとruntime summaryはmerge後に`_evaluation()`
    だけが計算する。game順に`merge()`した結果は、同じdecision列を1つの
    `_Tally`へ順に`add()`した結果と一致する。
    """

    decisions_by_kind: Counter = field(default_factory=Counter)
    guard_correct: Counter = field(default_factory=Counter)
    milliseconds: dict[O0DecisionKind, list[float]] = field(
        default_factory=lambda: {kind: [] for kind in O0DecisionKind}
    )
    legal_count: int = 0
    total: int = 0
    teacher_agreement: int = 0
    scorer_decisions: int = 0
    survivor_counts: Counter = field(default_factory=Counter)
    scorer_invoked: int = 0
    non_canonical_first: int = 0
    residual_single_tile_type: int = 0
    residual_samples: list[tuple[int, dict[str, object]]] = field(default_factory=list)
    max_regret: dict[str, int] = field(default_factory=_regret_names)
    regret_support: dict[str, int] = field(default_factory=_regret_names)
    regret_violations: dict[str, int] = field(default_factory=_regret_names)
    reference_identity: str | None = None
    reference_decisions: int = 0
    reference_oracle_agreement: int = 0
    reference_policy_agreement: int = 0
    reference_disagreement_outside_survivors: int = 0

    def add(
        self, policy, reference, decision, *, index: int, sample_limit: int
    ) -> None:
        """1 decisionを積み上げる。`index`は選択split内のglobal decision indexである。"""
        context = DecisionContext(
            input=decision.policy_input, legal_actions=decision.legal_actions
        )
        expected = _expected_kind(context)
        started = time.perf_counter()
        result = policy.decide(context)
        elapsed = (time.perf_counter() - started) * 1000.0

        self.total += 1
        self.decisions_by_kind[expected] += 1
        self.milliseconds[expected].append(elapsed)
        self.legal_count += int(
            any(result.action is legal for legal in context.legal_actions)
        )
        self.guard_correct[expected] += int(
            result.kind is expected
            and isinstance(result.action, _GUARD_TYPES[expected])
        )
        self.teacher_agreement += int(result.action == decision.selected_action)

        oracle_action = result.action
        if expected is O0DecisionKind.DISCARD and result.candidates is not None:
            self.scorer_decisions += 1
            candidates = result.candidates
            survivors = result.survivors
            oracle_action = candidates[survivors[0]].action
            self.survivor_counts[len(survivors)] += 1
            self.scorer_invoked += int(result.scorer_invoked)
            selected = result.selected_candidate
            if len(survivors) >= 2:
                self.non_canonical_first += int(selected.action is not oracle_action)
                tile_types = {candidates[i].action.tile.tile_type for i in survivors}
                self.residual_single_tile_type += int(len(tile_types) == 1)
                if len(self.residual_samples) < sample_limit:
                    self.residual_samples.append(
                        (
                            index,
                            {
                                "split": decision.split,
                                "survivors": [
                                    _describe(candidates[i].action) for i in survivors
                                ],
                                "selected": _describe(selected.action),
                            },
                        )
                    )
            regrets = _regrets(candidates, selected)
            for name, value in zip(
                ("shanten", "current_ukeire", "second_step"), regrets, strict=True
            ):
                if value is None:
                    continue
                self.regret_support[name] += 1
                self.max_regret[name] = max(self.max_regret[name], value)
                self.regret_violations[name] += int(value != 0)

        if reference is not None:
            self.reference_identity = _reference_identity(reference)
            self.reference_decisions += 1
            reference_action = reference.choose_action(context)
            self.reference_oracle_agreement += int(reference_action is oracle_action)
            self.reference_policy_agreement += int(reference_action is result.action)
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
                    self.reference_disagreement_outside_survivors += 1

    def merge(self, other: "_Tally", *, sample_limit: int) -> None:
        """canonical順で後続の`other`を加える（整数の和 / max / 列の連結）。"""
        if other.reference_identity is not None:
            if self.reference_identity not in (None, other.reference_identity):
                raise ValueError("reference identity differs between replay shards")
            self.reference_identity = other.reference_identity
        for counter in ("decisions_by_kind", "guard_correct", "survivor_counts"):
            getattr(self, counter).update(getattr(other, counter))
        for kind in O0DecisionKind:
            self.milliseconds[kind].extend(other.milliseconds[kind])
        for name in (
            "legal_count",
            "total",
            "teacher_agreement",
            "scorer_decisions",
            "scorer_invoked",
            "non_canonical_first",
            "residual_single_tile_type",
            "reference_decisions",
            "reference_oracle_agreement",
            "reference_policy_agreement",
            "reference_disagreement_outside_survivors",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in self.max_regret:
            self.max_regret[name] = max(self.max_regret[name], other.max_regret[name])
            self.regret_support[name] += other.regret_support[name]
            self.regret_violations[name] += other.regret_violations[name]
        # global decision index順に並べ、先頭`sample_limit`件だけを残す
        self.residual_samples = sorted(
            self.residual_samples + other.residual_samples, key=lambda item: item[0]
        )[:sample_limit]


def _evaluation(
    tally: _Tally,
    selected_splits: tuple[str, ...],
    sample_every: int,
    reference_identity: str | None,
) -> dict:
    """merge済み`_Tally`からevaluation dictを作る。rateはここでだけ計算する。"""
    total = tally.total
    if total == 0:
        raise ValueError(f"source record has no decisions in splits {selected_splits}")

    decisions_by_kind = tally.decisions_by_kind
    survivor_counts = tally.survivor_counts
    scorer_decisions = tally.scorer_decisions
    guard_rate = {
        kind: _rate(tally.guard_correct[kind], decisions_by_kind[kind])
        for kind in O0DecisionKind
    }
    residual = sum(count for size, count in survivor_counts.items() if size >= 2)
    evaluation: dict[str, object] = {
        "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
        "splits": list(selected_splits),
        "sample_every": sample_every,
        "invariants": {
            "legality": tally.legal_count / total,
            "win_guard": guard_rate[O0DecisionKind.WIN],
            "riichi_guard": guard_rate[O0DecisionKind.RIICHI],
            "no_call_guard": guard_rate[O0DecisionKind.RESPONSE],
            "discard_branch": guard_rate[O0DecisionKind.DISCARD],
            "max_regret": tally.max_regret,
            "regret_violations": tally.regret_violations,
            "regret_support": tally.regret_support,
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
            "fraction_scorer_invoked": _rate(tally.scorer_invoked, scorer_decisions),
            "fraction_non_canonical_first_given_residual": _rate(
                tally.non_canonical_first, residual
            ),
            "fraction_single_tile_type_given_residual": _rate(
                tally.residual_single_tile_type, residual
            ),
            "samples": [sample for _, sample in tally.residual_samples],
        },
        "teacher_agreement": _rate(tally.teacher_agreement, total),
        "runtime_by_branch": {
            kind.value: _runtime_summary(tally.milliseconds[kind])
            for kind in O0DecisionKind
        },
    }
    if reference_identity is not None:
        reference_decisions = tally.reference_decisions
        evaluation["reference"] = {
            "identity": reference_identity,
            "decisions": reference_decisions,
            "constant_scorer_oracle_agreement": _rate(
                tally.reference_oracle_agreement, reference_decisions
            ),
            "policy_agreement": _rate(
                tally.reference_policy_agreement, reference_decisions
            ),
            "disagreement_outside_survivors": (
                tally.reference_disagreement_outside_survivors
            ),
        }
    return evaluation


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
    selected_splits = _selected_splits(splits, sample_every)
    tally = _Tally()
    seen = 0
    for decision in source.decisions():
        if decision.split not in selected_splits:
            continue
        seen += 1
        if (seen - 1) % sample_every:
            continue
        tally.add(
            policy,
            reference,
            decision,
            index=seen - 1,
            sample_limit=residual_sample_limit,
        )
    return _evaluation(
        tally,
        selected_splits,
        sample_every,
        None if reference is None else _reference_identity(reference),
    )


# worker processごとの状態。`_start_replay_worker()`がfactoryを保持し、最初の
# shardでPolicyを生成する（factoryの例外をそのままshardの例外として返すため）。
_WORKER_STATE: dict[str, object] = {}


def _pin_worker_torch_threads() -> None:
    """worker processのtorch intra-op thread数を1へ固定する。

    このmoduleはtorchをimportしない。factoryがtorchをload済みのときだけ設定する。
    """
    torch = sys.modules.get("torch")
    if torch is not None and torch.get_num_threads() != 1:
        torch.set_num_threads(1)


def _start_replay_worker(policy_factory, reference_factory) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE["factories"] = (policy_factory, reference_factory)


def _replay_shard(policy, reference, shard, sample_limit: int) -> _Tally:
    tally = _Tally()
    for index, decision in shard:
        tally.add(policy, reference, decision, index=index, sample_limit=sample_limit)
    return tally


def _replay_shard_task(shard, sample_limit: int) -> _Tally:
    """process worker用のtop-level entry。Policyはこのworker内で生成する。"""
    if "policies" not in _WORKER_STATE:
        policy_factory, reference_factory = _WORKER_STATE["factories"]
        _WORKER_STATE["policies"] = (
            policy_factory(),
            None if reference_factory is None else reference_factory(),
        )
    _pin_worker_torch_threads()
    policy, reference = _WORKER_STATE["policies"]
    return _replay_shard(policy, reference, shard, sample_limit)


def evaluate_semantic_envelope_policy_by_game(
    policy_factory,
    games,
    splits: Iterable[str],
    *,
    workers: int = 1,
    sample_every: int = 1,
    reference_factory=None,
    residual_sample_limit: int = 12,
) -> dict:
    """`evaluate_semantic_envelope_policy()`をgame単位でprocess並列に実行する（#209）。

    `games`はcanonical順のgame列で、各gameは`split` / `policy_input` /
    `legal_actions` / `selected_action`を持つpicklableなdecisionのcanonical順列で
    ある。`policy_factory` / `reference_factory`は引数なしで新しいPolicyを返す
    picklableなcallableであり、Policy instanceやtorch moduleはprocess間で渡さない。

    `workers == 1`は親processでfactoryを1回ずつ呼び、全gameを順に評価する。
    `workers > 1`ではworkerごとにfactoryを呼び、torch thread数を1に固定する。
    `sample_every`のstrideとresidual sampleの順序は、選択split全体の通し
    decision indexで決める。worker結果はgame順にmergeしてからrateを計算するため、
    runtime以外の結果は`workers`によらず、同じdecision列に対する
    `evaluate_semantic_envelope_policy()`と一致する。1 gameでも失敗すれば例外を
    送出し、部分結果を返さない。`workers > 1`はprocess workerを起動するため、
    呼び出し側scriptは`if __name__ == "__main__":` guardの下で呼ぶ必要がある。
    """
    selected_splits = _selected_splits(splits, sample_every)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be an int >= 1")

    # 親processでglobal decision indexを振り、strideを全game通しで適用する
    shards: list[tuple[tuple[int, object], ...]] = []
    seen = 0
    for decisions in games:
        shard = []
        for decision in decisions:
            if decision.split not in selected_splits:
                continue
            seen += 1
            if (seen - 1) % sample_every:
                continue
            shard.append((seen - 1, decision))
        if shard:
            shards.append(tuple(shard))

    if not shards:
        tallies: list[_Tally] = []
    elif workers == 1:
        policy = policy_factory()
        reference = None if reference_factory is None else reference_factory()
        tallies = [
            _replay_shard(policy, reference, shard, residual_sample_limit)
            for shard in shards
        ]
    else:
        executor = ProcessPoolExecutor(
            max_workers=min(workers, len(shards)),
            initializer=_start_replay_worker,
            initargs=(policy_factory, reference_factory),
        )
        try:
            tallies = list(
                executor.map(
                    _replay_shard_task, shards, [residual_sample_limit] * len(shards)
                )
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    tally = _Tally()
    for shard_tally in tallies:
        tally.merge(shard_tally, sample_limit=residual_sample_limit)
    return _evaluation(
        tally,
        selected_splits,
        sample_every,
        None if reference_factory is None else tally.reference_identity,
    )


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
    "evaluate_semantic_envelope_policy_by_game",
]
