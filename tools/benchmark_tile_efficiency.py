"""Issue #213 牌効率計算backendのdevelopment benchmark。

固定decision列（`(DecisionContext, InternalAction)`のpickle list）を1つの
Policyで再生し、Policy判断全体の時間と、記録済みactionとの一致を測る。
shanten backendはprocess起動時の`LISJONG_SHANTEN_BACKEND`で決まるため、
backendごとに別processで実行して比較する。

    python tools/benchmark_tile_efficiency.py policy \\
        --decisions decisions.pickle \\
        --policy lisjong.policies:TwoStepUkeirePolicy --repeat 3

`kernel`は同じdecision列の再生中にnumeric shanten core
（`shanten._shanten_from_valid_counts()`）へ実際に渡された入力を記録し、その
入力列だけをPython coreとnative coreで同一process内で時間計測する
（native拡張が必要）。

    python tools/benchmark_tile_efficiency.py kernel \\
        --decisions decisions.pickle \\
        --policy lisjong.policies:TwoStepUkeirePolicy

decision pickleは自分で生成したlocal fileだけを読むこと（pickleは任意code
実行を許す）。decision列の生成はArenaの`LocalGameRunner`側で行い、この
repositoryからArenaへ依存しない。これはdevelopment-only utilityであり、CIの
wall-clock thresholdをここから作らない。
"""

import argparse
import gc
import hashlib
import importlib
import json
import os
import pathlib
import pickle
import platform
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lisjong.hand_evaluation import _shanten_backend  # noqa: E402


def _load_decisions(path: str) -> tuple[list, str]:
    payload = pathlib.Path(path).read_bytes()
    return pickle.loads(payload), hashlib.sha256(payload).hexdigest()


def _policy_class(reference: str):
    module_name, _, class_name = reference.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def _peak_rss_bytes() -> int | None:
    """process peak working set / max RSS。取得できなければNone。"""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_Counters),
            wintypes.DWORD,
        ]
        if psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "total_s": sum(ordered),
        "mean_ms": statistics.fmean(ordered) * 1000.0,
        "median_ms": statistics.median(ordered) * 1000.0,
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
    }


def _native_call_count() -> int | None:
    module = sys.modules.get("_lisjong_native")
    return None if module is None else module.standard_shanten_call_count()


def _environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "shanten_backend": _shanten_backend.BACKEND_NAME,
    }


def _run_policy(arguments: argparse.Namespace) -> dict[str, object]:
    records, digest = _load_decisions(arguments.decisions)
    policy_class = _policy_class(arguments.policy)

    # artifact / native table loadを最初のdecisionへ混ぜないよう、明示的に
    # 1回だけ計測してから再生する（cold start costとして別に報告する）。
    started = time.perf_counter()
    _shanten_backend.calculate_standard_shanten([0] * 33 + [1], 4)
    table_load_s = time.perf_counter() - started

    passes = []
    mismatches = 0
    native_before = _native_call_count()
    for pass_index in range(arguments.repeat):
        gc.collect()
        policy = policy_class()
        durations = []
        for decision, recorded in records:
            started = time.perf_counter()
            action = policy.choose_action(decision)
            durations.append(time.perf_counter() - started)
            if action != recorded:
                mismatches += 1
        passes.append({"pass": pass_index, **_summary(durations)})
    native_after = _native_call_count()

    return {
        "mode": "policy",
        "policy": arguments.policy,
        "decisions": len(records),
        "decisions_sha256": digest,
        "repeat": arguments.repeat,
        "table_load_s": table_load_s,
        "passes": passes,
        "pass_total_s": [item["total_s"] for item in passes],
        "action_mismatches": mismatches,
        "native_standard_shanten_calls": (
            None if native_before is None else native_after - native_before
        ),
        "peak_rss_bytes": _peak_rss_bytes(),
        "environment": _environment(),
    }


def _run_kernel(arguments: argparse.Namespace) -> dict[str, object]:
    import _lisjong_native

    import lisjong.hand_evaluation.shanten as shanten_module

    if _shanten_backend.BACKEND_NAME != _shanten_backend.PYTHON_BACKEND:
        raise SystemExit("kernel mode records inputs with the default Python backend")
    records, digest = _load_decisions(arguments.decisions)
    if arguments.limit_decisions is not None:
        records = records[: arguments.limit_decisions]
    policy_class = _policy_class(arguments.policy)

    # Issue #213の対象境界であるnumeric shanten core
    # （`shanten._shanten_from_valid_counts()`）へ実際に渡された入力を記録する。
    inputs: list[tuple[tuple[int, ...], int]] = []
    python_core = shanten_module._shanten_from_valid_counts

    def recording(counts, concealed_tile_count):
        inputs.append((tuple(counts), concealed_tile_count))
        return python_core(counts, concealed_tile_count)

    shanten_module._shanten_from_valid_counts = recording
    try:
        policy = policy_class()
        for decision, _recorded in records:
            policy.choose_action(decision)
    finally:
        shanten_module._shanten_from_valid_counts = python_core

    started = time.perf_counter()
    native = _shanten_backend.build_native_table(_lisjong_native)
    native_load_s = time.perf_counter() - started
    native_core = native.shanten_from_valid_counts

    mismatches = sum(
        native_core(counts, size) != python_core(counts, size)
        for counts, size in inputs
    )
    timings: dict[str, list[float]] = {"python": [], "rust": []}
    for _ in range(arguments.repeat):
        for name, function in (("python", python_core), ("rust", native_core)):
            gc.collect()
            started = time.perf_counter()
            for counts, size in inputs:
                function(counts, size)
            timings[name].append(time.perf_counter() - started)

    return {
        "mode": "kernel",
        "boundary": "shanten._shanten_from_valid_counts",
        "policy": arguments.policy,
        "decisions": len(records),
        "decisions_sha256": digest,
        "core_inputs": len(inputs),
        "distinct_inputs": len(set(inputs)),
        "value_mismatches": mismatches,
        "native_table_load_s": native_load_s,
        "repeat": arguments.repeat,
        "python_s": timings["python"],
        "rust_s": timings["rust"],
        "python_us_per_call_median": statistics.median(timings["python"])
        / len(inputs)
        * 1e6,
        "rust_us_per_call_median": statistics.median(timings["rust"])
        / len(inputs)
        * 1e6,
        "environment": _environment(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("policy", "kernel"))
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--policy", required=True, help="module:Class")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--limit-decisions",
        type=int,
        help="kernel mode: record inputs from only the first N decisions",
    )
    arguments = parser.parse_args(argv)
    if arguments.repeat <= 0:
        parser.error("--repeat must be positive")
    runner = _run_policy if arguments.mode == "policy" else _run_kernel
    print(json.dumps(runner(arguments), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
