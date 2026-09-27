"""Issue #218 kobalab 0004系Policyのdecision再生profile（development-only）。

`tools/benchmark_tile_efficiency.py`と同じ固定decision列
（`(DecisionContext, InternalAction)`のpickle list）を再生する。

    python tools/profile_kobalab_0004.py timing --decisions decisions.pickle \\
        --policy lisjong.policies:Kobalab0004ReferencePolicy --repeat 5
    python tools/profile_kobalab_0004.py breakdown --decisions decisions.pickle \\
        --policy lisjong.policies:Kobalab0004ReferencePolicy

- `timing`: profilerなし。decisionごとの`choose_action()`時間をdecision種類別に
  集計する。1 passごとにPolicy instanceを作り直し、process最初のpassをcold、
  以降をwarmとして分けて報告する。
- `breakdown`: 下記の対象関数をmodule属性の差し替えでwrapし、呼び出し回数・
  子処理を含む時間（inclusive）・自己時間（self = inclusive − wrapした子処理）を
  集計する。wrap自体のoverheadを含むため、速度比較には`timing`を使う。
  `calculate_shanten`はPython側から見えるnative呼び出しを含む時間であり、
  Rust内部の内訳ではない（Rust内部はkernel modeの`benchmark_tile_efficiency.py`）。

decision pickleは自分で生成したlocal fileだけを読むこと（pickleは任意code
実行を許す）。CIのwall-clock thresholdをここから作らない。
"""

import argparse
import gc
import hashlib
import importlib
import json
import pathlib
import pickle
import platform
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lisjong.hand_evaluation import _shanten_backend  # noqa: E402
from lisjong.policy_contract.action import (  # noqa: E402
    AnkanAction,
    DiscardAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)

_BREAKDOWN_TARGETS = (
    # (module, attribute, label)。存在しない属性はskipし、reportへ記録する。
    ("lisjong.policies.kobalab_0004_reference", "calculate_shanten", "shanten"),
    (
        "lisjong.policies.kobalab_0004_reference",
        # Issue #224: 打牌候補の一括構造評価（打牌後向聴数と改善牌）。
        "evaluate_discards_from_canonical_counts",
        "discard_batch",
    ),
    (
        "lisjong.policies.kobalab_0004_reference",
        "_improving_tile_types",
        "improving_tile_types",
    ),
    ("lisjong.policies.kobalab_0004_reference", "_remove_exact", "remove_exact"),
    (
        "lisjong.policies.kobalab_0004_reference",
        "derive_remaining_tile_inventory",
        "remaining_inventory",
    ),
    ("lisjong.policies.kobalab_0004_reference", "_evaluation_order", "eval_order"),
    ("lisjong.policies.kobalab_0004_reference", "_choose_discard", "choose_discard"),
    (
        "lisjong.policies.kobalab_0004_reference",
        "_allows_riichi_discard",
        "riichi_check",
    ),
    (
        "lisjong.policies.kobalab_0004_reference",
        "_structural_discard_evaluations",
        "structural_eval",
    ),
    (
        "lisjong.policies.kobalab_0004_reference",
        "_paijia_input_from_belief",
        "belief_paijia_input",
    ),
    (
        "lisjong.policies.kobalab_0004_reference",
        "_opponent_concealed_slot_counts_by_wind",
        "belief_slot_counts",
    ),
    (
        "lisjong.policies.kobalab_0004_reference",
        "_estimate_from_conservation",
        "belief_uniform_estimator",
    ),
    (
        # 推定器内部のinventory導出（Issue #220以降、Policy経路では呼ばれない）。
        "lisjong.belief.conditional_uniform_hand_belief",
        "derive_remaining_tile_inventory",
        "belief_estimator_inventory",
    ),
    (
        "lisjong.policies.kobalab_0004_reference",
        "derive_non_player_hidden_belief",
        "belief_non_player_hidden",
    ),
)
_PAIJIA_TARGETS = (
    ("lisjong.policies.kobalab_0004_reference", "_PublicCounts", "paijia"),
    ("lisjong.policies.kobalab_0004_reference", "_PaijiaInput", "paijia"),
)


def _load_decisions(path: str) -> tuple[list, str]:
    payload = pathlib.Path(path).read_bytes()
    return pickle.loads(payload), hashlib.sha256(payload).hexdigest()


def _policy_class(reference: str):
    module_name, _, class_name = reference.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def decision_kind(decision) -> str:
    """legal_actionsだけから決まるdecision種類（Policyの判断結果に依存しない）。"""
    legal = decision.legal_actions
    if any(isinstance(a, (TsumoAction, RonAction)) for a in legal):
        return "win_available"
    if any(isinstance(a, DiscardAction) for a in legal):
        if any(isinstance(a, KyuushuKyuuhaiAction) for a in legal):
            return "discard+kyuushu"
        if any(isinstance(a, (AnkanAction, KakanAction)) for a in legal):
            return "discard+kan"
        if any(isinstance(a, RiichiAction) for a in legal):
            return "discard+riichi"
        return "discard"
    return "response"


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "total_ms": sum(ordered) * 1000.0,
        "mean_ms": statistics.fmean(ordered) * 1000.0,
        "median_ms": statistics.median(ordered) * 1000.0,
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
    }


def _native_revision() -> str | None:
    module = sys.modules.get("_lisjong_native")
    return None if module is None else getattr(module, "SOURCE_REVISION", "unknown")


def _native_call_count() -> int | None:
    """native numeric coreの実計算回数（境界呼び出し回数ではない。Issue #224）。"""
    module = sys.modules.get("_lisjong_native")
    return None if module is None else module.standard_shanten_call_count()


def _native_batch_call_count() -> int | None:
    """一括構造評価の境界呼び出し回数（#224以前のnativeにはないのでNone）。"""
    module = sys.modules.get("_lisjong_native")
    counter = getattr(module, "discard_evaluation_call_count", None)
    return None if counter is None else counter()


def _environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "shanten_backend": _shanten_backend.BACKEND_NAME,
        "native_source_revision": _native_revision(),
    }


def _run_timing(arguments: argparse.Namespace) -> dict[str, object]:
    records, digest = _load_decisions(arguments.decisions)
    policy_class = _policy_class(arguments.policy)
    kinds = [decision_kind(decision) for decision, _ in records]

    passes = []
    mismatches = 0
    native_before = _native_call_count()
    batch_before = _native_batch_call_count()
    per_decision: list[list[float]] = []
    for pass_index in range(arguments.repeat):
        gc.collect()
        policy = policy_class()
        durations = []
        for decision, recorded in records:
            started = time.perf_counter()
            action = policy.choose_action(decision)
            durations.append(time.perf_counter() - started)
            if arguments.compare_recorded and action != recorded:
                mismatches += 1
        per_decision.append(durations)
        by_kind: dict[str, list[float]] = {}
        for kind, duration in zip(kinds, durations, strict=True):
            by_kind.setdefault(kind, []).append(duration)
        passes.append(
            {
                "pass": pass_index,
                "cold": pass_index == 0,
                "all": _summary(durations),
                "by_kind": {k: _summary(v) for k, v in sorted(by_kind.items())},
            }
        )
    native_after = _native_call_count()
    batch_after = _native_batch_call_count()

    warm = per_decision[1:] or per_decision
    medians = [statistics.median(column) for column in zip(*warm, strict=True)]
    slowest = sorted(range(len(records)), key=lambda i: -medians[i])[:10]
    return {
        "mode": "timing",
        "policy": arguments.policy,
        "decisions": len(records),
        "decisions_sha256": digest,
        "kind_counts": {k: kinds.count(k) for k in sorted(set(kinds))},
        "repeat": arguments.repeat,
        "passes": passes,
        "pass_total_ms": [item["all"]["total_ms"] for item in passes],
        "warm_pass_total_ms_median": statistics.median(
            item["all"]["total_ms"] for item in passes[1:] or passes
        ),
        "slowest_warm_median": [
            {"index": i, "kind": kinds[i], "median_ms": medians[i] * 1000.0}
            for i in slowest
        ],
        "action_mismatches": mismatches if arguments.compare_recorded else None,
        "native_standard_shanten_calls": (
            None if native_before is None else native_after - native_before
        ),
        "native_discard_batch_calls": (
            None if batch_before is None else batch_after - batch_before
        ),
        "environment": _environment(),
    }


class _Accumulator:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.inclusive: dict[str, float] = {}
        self.self_time: dict[str, float] = {}
        self._child_stack: list[float] = []

    def wrap(self, label: str, function):
        def wrapped(*args, **kwargs):
            self._child_stack.append(0.0)
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                children = self._child_stack.pop()
                if self._child_stack:
                    self._child_stack[-1] += elapsed
                self.calls[label] = self.calls.get(label, 0) + 1
                self.inclusive[label] = self.inclusive.get(label, 0.0) + elapsed
                self.self_time[label] = (
                    self.self_time.get(label, 0.0) + elapsed - children
                )

        return wrapped


def _run_breakdown(arguments: argparse.Namespace) -> dict[str, object]:
    records, digest = _load_decisions(arguments.decisions)
    policy_class = _policy_class(arguments.policy)
    accumulator = _Accumulator()
    patched: list[tuple[object, str, object]] = []
    skipped: list[str] = []

    for module_name, attribute, label in _BREAKDOWN_TARGETS:
        module = importlib.import_module(module_name)
        original = getattr(module, attribute, None)
        if original is None:
            skipped.append(f"{module_name}.{attribute}")
            continue
        patched.append((module, attribute, original))
        setattr(module, attribute, accumulator.wrap(label, original))
    for module_name, class_name, label in _PAIJIA_TARGETS:
        cls = getattr(importlib.import_module(module_name), class_name, None)
        if cls is None or not hasattr(cls, "_compute_paijia"):
            skipped.append(f"{module_name}.{class_name}._compute_paijia")
            continue
        original = cls._compute_paijia
        patched.append((cls, "_compute_paijia", original))
        cls._compute_paijia = accumulator.wrap(label, original)

    kinds = [decision_kind(decision) for decision, _ in records]
    root = "choose_action"
    try:
        for _ in range(arguments.repeat):
            gc.collect()
            policy = policy_class()
            choose = accumulator.wrap(root, policy.choose_action)
            for decision, _recorded in records:
                choose(decision)
    finally:
        for owner, attribute, original in reversed(patched):
            setattr(owner, attribute, original)

    total = accumulator.inclusive[root]
    rows = {
        label: {
            "calls": accumulator.calls[label],
            "inclusive_ms": accumulator.inclusive[label] * 1000.0,
            "self_ms": accumulator.self_time[label] * 1000.0,
            "self_share": accumulator.self_time[label] / total,
        }
        for label in sorted(accumulator.calls, key=lambda k: -accumulator.self_time[k])
    }
    return {
        "mode": "breakdown",
        "policy": arguments.policy,
        "decisions": len(records),
        "decisions_sha256": digest,
        "kind_counts": {k: kinds.count(k) for k in sorted(set(kinds))},
        "repeat": arguments.repeat,
        "note": (
            "self_ms of choose_action is time not attributed to any wrapped "
            "function; wrapper overhead is included, so use timing mode for speed"
        ),
        "rows": rows,
        "skipped_targets": skipped,
        "environment": _environment(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("timing", "breakdown"))
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--policy", required=True, help="module:Class")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--no-compare-recorded",
        dest="compare_recorded",
        action="store_false",
        help="timing: do not compare with recorded actions (other Policies)",
    )
    arguments = parser.parse_args(argv)
    if arguments.repeat <= 0:
        parser.error("--repeat must be positive")
    runner = _run_timing if arguments.mode == "timing" else _run_breakdown
    print(json.dumps(runner(arguments), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
