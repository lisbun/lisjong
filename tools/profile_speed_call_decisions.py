"""Issue #228 PlacementAwareSpeedCallPolicy decision時間のprofile（development-only）。

Arenaの`scripts/capture_policy_decisions.py`が作る固定decision列
（`(DecisionContext, InternalAction)`のpickle list）を再生する。decision列は
Arena checkoutで次のように作る（Arena側のLocalGameRunner self-play。この
repositoryからArenaへ依存しない）。

    PYTHONPATH=<lisjong>/src python scripts/capture_policy_decisions.py capture \\
        --policy lisjong.policies:PlacementAwareSpeedCallPolicy --seed 0 \\
        --game-mode 4p-red-half --output d.pickle

    python tools/profile_speed_call_decisions.py scan --decisions d.pickle \\
        --top 20 [--export-min-ms 1000 --export slow.pickle]
    python tools/profile_speed_call_decisions.py breakdown --decisions d.pickle \\
        [--indices 12 34 | --min-ms 1000]

- `scan`: profilerなしでdecisionごとの`choose_action_with_analysis()`時間を測り、
  player-visibleなinput特徴（手牌枚数、副露数、向聴数、legal候補数、他家リーチ
  人数、オーラスmode等）、#174 activation stage、FiniteHorizon / R5 DPの
  state・shanten評価数と並べて報告する。DP counterは各evaluatorが元から持つ
  private instrumentationを、decisionごとに数回しか呼ばれない
  `_evaluate_completion_masses` / `_evaluate_progression_candidates`の
  戻り時点で読むだけであり、hot loopへwrapperを入れない。
  rust指定のprocessではR5はnative evaluatorが実行し、同じ4 counterをnativeが
  実測してevaluatorへ保持する（Issue #232。未計測は`None`で、0で埋めない）。
  `--export`は閾値以上のdecisionだけを同じ形式のpickleへ書き出す（local専用。
  repositoryへcommitしない）。
- `breakdown`: stage関数とshanten primitiveをmodule / class属性の差し替えで
  wrapし、選んだdecisionごとに呼び出し回数・inclusive・self時間を出す。
  wrap自体のoverhead（特にshanten primitive）を含むため、速度の絶対値は
  `scan`を使う。rust指定のprocessでは、R5探索本体はnative内で完結するため
  Python関数wrapではnative内部のshanten時間・call数を測れない。その場合は
  R5一括呼出しの時間（`r5_progression_dp`のinclusive）とnativeの実測counterだけを
  報告し、native内部のshanten時間・Python DP overheadは`null` / `not_measured`と
  理由を出す（0秒・0回やPython DP overheadとは表示しない）。

shanten backendはprocess起動時の`LISJONG_SHANTEN_BACKEND`で決まるため、
backend比較は別processで行う。

Policyのsemanticsや選択actionは変えない（`scan`は記録済みactionとの一致を
報告する）。decision pickleは自分で生成したlocal fileだけを読むこと（pickleは
任意code実行を許す）。CIのwall-clock thresholdをここから作らない。
"""

import argparse
import gc
import hashlib
import importlib
import json
import pathlib
import pickle
import platform
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lisjong.hand_evaluation import _shanten_backend, calculate_shanten  # noqa: E402
from lisjong.policies import PlacementAwareSpeedCallPolicy  # noqa: E402
from lisjong.policies import (  # noqa: E402
    terminal_shanten_progression_mechanism_riichi_defense as _progression,
)
from lisjong.policies.genbutsu_defense_two_step_ukeire import (  # noqa: E402
    _opponent_riichi_players,
)
from lisjong.policies.placement_aware_speed_call import situation_mode  # noqa: E402
from lisjong.policy_contract.action import (  # noqa: E402
    AnkanAction,
    ChiAction,
    DiscardAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    PonAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)

_POLICY = "lisjong.policies:PlacementAwareSpeedCallPolicy"
_FH = "lisjong.policies.finite_horizon_completion"
_GFH = "lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware"
_THR = "lisjong.policies.targeted_honor_release_terminal_progression"
_TSP = "lisjong.policies.terminal_shanten_progression_mechanism_riichi_defense"
_MRD = "lisjong.policies.mechanism_riichi_defense_yakuhai_call"
_YC = "lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware"
_PASC = "lisjong.policies.placement_aware_speed_call"
_HVA = "lisjong.policies.hand_value_aware_two_step_ukeire"
_SE = "lisjong.structural_efficiency"
_SH = "lisjong.hand_evaluation.shanten"

_DP_COUNTER_TARGETS = (
    # (module, attribute, label): evaluatorを第5引数に受け取るDP入口。
    (_THR, "_evaluate_completion_masses", "finite_horizon"),
    (_GFH, "_evaluate_completion_masses", "finite_horizon"),
    (_THR, "_evaluate_progression_candidates", "r5_progression"),
)
_DP_COUNTER_FIELDS = (
    "visited_states",
    "cache_hits",
    "cache_misses",
    "shanten_evaluations",
    "completion_predicate_evaluations",
    "tenpai_predicate_evaluations",
)

_NATIVE_EVALUATOR = _progression._NativeTerminalShantenProgressionEvaluator
_R5_COUNTER_FIELDS = (
    "visited_states",
    "cache_hits",
    "cache_misses",
    "shanten_evaluations",
)
_NATIVE_NOT_MEASURED_REASON = (
    "the R5 search runs inside the native extension, so Python function wrappers "
    "cannot time or count its internal shanten evaluations; only the whole native "
    "batch call and the native instrumentation counters are measured"
)

_CLASS_TARGETS = (
    # (module, class, method, label)。super()経由の呼び出しも捕捉する。
    (_PASC, "PlacementAwareSpeedCallPolicy", "_decide", "pasc._decide"),
    (
        _YC,
        "YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy",
        "_decide",
        "yakuhai_call._decide",
    ),
    (
        "lisjong.policies.two_step_ukeire",
        "TwoStepUkeirePolicy",
        "_decide",
        "two_step._decide",
    ),
    (_PASC, "PlacementAwareSpeedCallPolicy", "_decide_discard", "pasc._decide_discard"),
    (
        _THR,
        "TargetedHonorReleaseTerminalProgressionPolicy",
        "_decide_discard",
        "thr._decide_discard",
    ),
)
_FUNCTION_TARGETS = (
    # (module, attribute, label)。呼び出し元moduleの名前空間で差し替える。
    (_PASC, "_eligible_discard_actions", "pasc.eligible_discards"),
    (_PASC, "_route_preserving_actions", "pasc.route_preserving"),
    (_PASC, "_top_fold_actions", "pasc.top_fold"),
    (_PASC, "_speed_call_candidates", "pasc.speed_call_candidates"),
    (_PASC, "_decide_push_fold", "push_fold@pasc"),
    (_YC, "_qualifying_call_candidates", "yakuhai_call.candidates"),
    (_THR, "_classify_branch", "thr.classify_branch"),
    (_MRD, "_evaluate_mechanism_defense_filter", "mechanism_defense_filter"),
    (_GFH, "_evaluate_defense_filter", "defense_filter"),
    (_GFH, "_decide_push_fold", "push_fold@defense"),
    (_THR, "_parent_action", "thr.parent_action"),
    (_THR, "_root_remaining_counts", "remaining_inventory"),
    (_GFH, "_root_remaining_counts", "remaining_inventory"),
    (_THR, "_evaluate_completion_masses", "finite_horizon_dp"),
    (_GFH, "_evaluate_completion_masses", "finite_horizon_dp"),
    (_THR, "_select_from_completion_masses", "thr.select_positive"),
    (_THR, "_evaluate_all_zero_targeted_honor_release", "thr.all_zero_gate"),
    (_THR, "_evaluate_progression_candidates", "r5_progression_dp"),
    (_THR, "_hand_value_aware_evaluate_and_choose_discard", "hand_value_aware"),
    (_GFH, "_hand_value_aware_evaluate_and_choose_discard", "hand_value_aware"),
    (_HVA, "second_step_ukeire_score", "second_step_ukeire"),
    # shanten primitive（呼び出し元別の入口と、共通numeric core）。
    (_FH, "calculate_shanten_from_canonical_counts", "shanten@finite_horizon"),
    (_FH, "is_structurally_tenpai_from_canonical_counts", "tenpai_pred@finite_horizon"),
    (
        _FH,
        "is_structurally_complete_from_canonical_counts",
        "complete_pred@finite_horizon",
    ),
    (_TSP, "calculate_shanten_from_canonical_counts", "shanten@r5_progression"),
    (_SE, "calculate_shanten", "shanten@structural_efficiency"),
    (_YC, "calculate_shanten", "shanten@yakuhai_call"),
    (_PASC, "calculate_shanten", "shanten@pasc"),
    (_PASC, "calculate_restricted_standard_shanten", "restricted_shanten@pasc"),
    (_SH, "_shanten_from_valid_counts", "shanten_core"),
)


def _load_decisions(path: str) -> tuple[list, str]:
    payload = pathlib.Path(path).read_bytes()
    return pickle.loads(payload), hashlib.sha256(payload).hexdigest()


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
    if any(isinstance(a, (ChiAction, PonAction)) for a in legal):
        return "call"
    return "response"


def input_features(decision) -> dict[str, object]:
    """slow decisionと突き合わせるplayer-visible特徴。hidden informationは使わない。"""
    policy_input = decision.input
    own = policy_input.own_hand
    melds = policy_input.players[int(policy_input.self_seat)].melds
    discards = [a for a in decision.legal_actions if isinstance(a, DiscardAction)]
    return {
        "kind": decision_kind(decision),
        "concealed_tiles": len(own.concealed_tiles),
        "melds": len(melds),
        "closed": all(meld.kind.name == "ANKAN" for meld in melds),
        "drawn_tile": own.drawn_tile is not None,
        "shanten": calculate_shanten(own.concealed_tiles),
        "legal_discards": len(discards),
        "discard_tile_types": len({a.tile.tile_type for a in discards}),
        "call_actions": sum(
            isinstance(a, (ChiAction, PonAction)) for a in decision.legal_actions
        ),
        "riichi_opponents": len(_opponent_riichi_players(policy_input)),
        "mode": situation_mode(policy_input).name,
        "live_wall": policy_input.round.live_wall_tiles_remaining,
    }


def _environment() -> dict[str, object]:
    native = sys.modules.get("_lisjong_native")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "shanten_backend": _shanten_backend.BACKEND_NAME,
        "native_source_revision": (
            None if native is None else getattr(native, "SOURCE_REVISION", "unknown")
        ),
    }


def _patch(owner, attribute: str, replacement, patched: list) -> None:
    patched.append((owner, attribute, getattr(owner, attribute)))
    setattr(owner, attribute, replacement)


def _restore(patched: list) -> None:
    for owner, attribute, original in reversed(patched):
        setattr(owner, attribute, original)


class _DpCounters:
    """1 decision分のDP evaluator counterを、DP入口の戻り時点で集める。"""

    def __init__(self) -> None:
        self.current: dict[str, dict[str, int]] = {}

    def wrap(self, label: str, function):
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            finally:
                evaluator = args[4] if len(args) > 4 else kwargs.get("evaluator")
                row = self.current.setdefault(label, {"calls": 0})
                row["calls"] += 1
                if isinstance(evaluator, _NATIVE_EVALUATOR):
                    row["evaluator"] = "native"
                for field in _DP_COUNTER_FIELDS:
                    value = getattr(evaluator, field, None)
                    if value is not None:
                        row[field] = row.get(field, 0) + value

        return wrapped


def _analysis_facts(decision_result) -> dict[str, object]:
    """#174 analysisが観測したactivation stageと候補数（打牌decisionだけ）。"""
    analysis = decision_result.analysis
    stage = getattr(analysis, "activation_stage", None)
    if stage is None:
        return {"thr_stage": None}
    return {
        "thr_stage": stage.name,
        "thr_eligible": analysis.eligible_candidate_count,
        "thr_targets": analysis.target_candidate_count,
        "thr_honor_targets": analysis.honor_target_candidate_count,
    }


def _run_scan(arguments: argparse.Namespace) -> dict[str, object]:
    records, digest = _load_decisions(arguments.decisions)
    counters = _DpCounters()
    patched: list = []
    for module_name, attribute, label in _DP_COUNTER_TARGETS:
        module = importlib.import_module(module_name)
        _patch(
            module, attribute, counters.wrap(label, getattr(module, attribute)), patched
        )

    rows = []
    mismatches = 0
    try:
        gc.collect()
        policy = PlacementAwareSpeedCallPolicy()
        for index, (decision, recorded) in enumerate(records):
            counters.current = {}
            started = time.perf_counter()
            result = policy.choose_action_with_analysis(decision)
            elapsed = time.perf_counter() - started
            matched = result.action == recorded
            mismatches += not matched
            rows.append(
                {
                    "index": index,
                    "ms": elapsed * 1000.0,
                    "matches_recorded": matched,
                    **_analysis_facts(result),
                    **input_features(decision),
                    "dp": counters.current,
                }
            )
    finally:
        _restore(patched)

    ordered = sorted(rows, key=lambda row: -row["ms"])
    total_ms = sum(row["ms"] for row in rows)
    thresholds = (100, 1000, 3000, 10000)
    exported = None
    if arguments.export:
        selected = [
            records[row["index"]]
            for row in rows
            if row["ms"] >= arguments.export_min_ms
        ]
        output = pathlib.Path(arguments.export)
        if output.exists():
            raise SystemExit(f"refusing to overwrite {output}")
        payload = pickle.dumps(selected, protocol=pickle.HIGHEST_PROTOCOL)
        output.write_bytes(payload)
        exported = {
            "path": str(output),
            "decisions": len(selected),
            "min_ms": arguments.export_min_ms,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {
        "mode": "scan",
        "policy": _POLICY,
        "decisions": len(records),
        "decisions_sha256": digest,
        "total_ms": total_ms,
        "action_mismatches": mismatches,
        "count_at_least_ms": {
            str(t): sum(row["ms"] >= t for row in rows) for t in thresholds
        },
        "share_of_total_in_top": {
            str(n): sum(row["ms"] for row in ordered[:n]) / total_ms for n in (1, 5, 20)
        },
        "slowest": ordered[: arguments.top],
        "exported": exported,
        "environment": _environment(),
    }


class _Accumulator:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.inclusive: dict[str, float] = {}
        self.self_time: dict[str, float] = {}
        self._child_stack: list[float] = []

    def reset(self) -> None:
        self.calls.clear()
        self.inclusive.clear()
        self.self_time.clear()

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

    def rows(self, root: str) -> dict[str, dict[str, float]]:
        total = self.inclusive[root]
        return {
            label: {
                "calls": self.calls[label],
                "inclusive_ms": round(self.inclusive[label] * 1000.0, 3),
                "self_ms": round(self.self_time[label] * 1000.0, 3),
                "inclusive_share": round(self.inclusive[label] / total, 4),
            }
            for label in sorted(self.calls, key=lambda k: -self.inclusive[k])
        }


class _R5Recorder:
    """1 decisionが生成したR5 evaluatorを、factory経由で記録する。"""

    def __init__(self) -> None:
        self.evaluators: list = []

    def wrap(self, factory):
        def wrapped(*args, **kwargs):
            evaluator = factory(*args, **kwargs)
            self.evaluators.append(evaluator)
            return evaluator

        return wrapped


def _r5_report(evaluators: list, rows: dict[str, dict]) -> dict[str, object] | None:
    """R5 evaluatorごとの実測counterと、native時の未計測値の明示。

    native evaluatorではnative内部のshanten時間とPython DP overheadを測れないので、
    `r5_progression_dp`行の`self_ms`（Python oracleではDP overhead）を`null`にして
    理由を添える。nativeのshanten call数は`shanten_evaluations` counterとして
    実測済みである。
    """
    if not evaluators:
        return None
    native = [item for item in evaluators if isinstance(item, _NATIVE_EVALUATOR)]
    counters = {
        name: sum(
            value
            for item in evaluators
            if (value := getattr(item, name, None)) is not None
        )
        for name in _R5_COUNTER_FIELDS
    }
    if not native:
        return {
            "backend": "python",
            "evaluators": len(evaluators),
            "counters": counters,
        }
    if len(native) != len(evaluators):
        raise RuntimeError("a decision mixed native and Python R5 evaluators")
    if any(
        getattr(item, name) is None for item in native for name in _R5_COUNTER_FIELDS
    ):
        counters = {name: None for name in _R5_COUNTER_FIELDS}
    row = rows.get("r5_progression_dp")
    if row is not None:
        row["self_ms"] = None
        row["self_ms_status"] = (
            "not_measured: includes the native search, not Python DP overhead"
        )
    return {
        "backend": "rust-native",
        "evaluators": len(evaluators),
        "counters": counters,
        "native_shanten_ms": None,
        "native_shanten_status": "not_measured",
        "native_shanten_reason": _NATIVE_NOT_MEASURED_REASON,
        "python_dp_overhead_ms": None,
        "python_dp_overhead_status": "not_applicable: no Python DP runs",
        "r5_batch_ms": None if row is None else row["inclusive_ms"],
    }


def _run_breakdown(arguments: argparse.Namespace) -> dict[str, object]:
    records, digest = _load_decisions(arguments.decisions)
    accumulator = _Accumulator()
    patched: list = []
    skipped: list[str] = []
    for module_name, class_name, method, label in _CLASS_TARGETS:
        owner = getattr(importlib.import_module(module_name), class_name)
        if method not in vars(owner):
            skipped.append(f"{module_name}.{class_name}.{method}")
            continue
        _patch(owner, method, accumulator.wrap(label, vars(owner)[method]), patched)
    for module_name, attribute, label in _FUNCTION_TARGETS:
        module = importlib.import_module(module_name)
        original = getattr(module, attribute, None)
        if original is None:
            skipped.append(f"{module_name}.{attribute}")
            continue
        _patch(module, attribute, accumulator.wrap(label, original), patched)

    recorder = _R5Recorder()
    thr_module = importlib.import_module(_THR)
    _patch(
        thr_module,
        "_new_progression_evaluator",
        recorder.wrap(thr_module._new_progression_evaluator),
        patched,
    )

    if arguments.indices:
        indices = arguments.indices
    else:
        indices = list(range(len(records)))
    root = "choose_action"
    per_decision = []
    try:
        policy = PlacementAwareSpeedCallPolicy()
        choose = accumulator.wrap(root, policy.choose_action)
        for index in indices:
            decision, _recorded = records[index]
            gc.collect()
            accumulator.reset()
            recorder.evaluators = []
            choose(decision)
            total_ms = accumulator.inclusive[root] * 1000.0
            if total_ms < arguments.min_ms:
                continue
            rows = accumulator.rows(root)
            entry = {
                "index": index,
                "total_ms_with_wrappers": round(total_ms, 3),
                **input_features(decision),
                "rows": rows,
            }
            r5 = _r5_report(recorder.evaluators, rows)
            if r5 is not None:
                entry["r5"] = r5
            per_decision.append(entry)
    finally:
        _restore(patched)

    return {
        "mode": "breakdown",
        "policy": _POLICY,
        "decisions": len(records),
        "decisions_sha256": digest,
        "note": (
            "inclusive/self include wrapper overhead (notably shanten primitives); "
            "labels with the same name in several modules are merged; use scan "
            "for absolute timing"
        ),
        "per_decision": per_decision,
        "skipped_targets": skipped,
        "environment": _environment(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="mode", required=True)
    scan = commands.add_parser("scan")
    scan.add_argument("--decisions", required=True)
    scan.add_argument("--top", type=int, default=20)
    scan.add_argument("--export", help="write decisions >= --export-min-ms here")
    scan.add_argument("--export-min-ms", type=float, default=1000.0)
    breakdown = commands.add_parser("breakdown")
    breakdown.add_argument("--decisions", required=True)
    breakdown.add_argument("--indices", type=int, nargs="*")
    breakdown.add_argument(
        "--min-ms", type=float, default=0.0, help="report only decisions this slow"
    )
    arguments = parser.parse_args(argv)
    runner = _run_scan if arguments.mode == "scan" else _run_breakdown
    print(json.dumps(runner(arguments), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
