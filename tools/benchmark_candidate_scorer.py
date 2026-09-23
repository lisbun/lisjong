"""Issue #189 L0.2 candidate scorer serving pathのdevelopment performance preflight。

source recordからnormal-discard（O0 DISCARD）decisionをsampleし、actual L0.2
serving pathのstage別時間と、dataset materializationのprojected durationを出す。

    python tools/benchmark_candidate_scorer.py --source-record <source-record>
    python tools/benchmark_candidate_scorer.py --source-record <source-record> \\
        --artifact <candidate-scorer-artifact> --sample-every 50

`--artifact`を指定した場合だけtorch forwardとLearned Policy全体を測る
（optional ML runtimeが必要）。

これはdevelopment-only utilityである。CI wall-clock thresholdやcorrectness
testの時間thresholdをここから作らない。productionのserving pathへbenchmark
専用hookを入れないため、stageは公開APIを順に呼んで測る。
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lisjong.learning import (  # noqa: E402
    build_discard_candidate_features,
    build_player_safe_feature,
    build_scorer_candidates,
    encode_candidates,
    read_source_record,
)
from lisjong.learning._o0 import O0DecisionKind, classify_o0_decision  # noqa: E402
from lisjong.learning.candidate_encoding import second_step_finalists  # noqa: E402
from lisjong.policy_contract import DecisionContext  # noqa: E402


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(ordered) * 1000.0,
        "median_ms": statistics.median(ordered) * 1000.0,
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
    }


def _timed(function, *args):
    started = time.perf_counter()
    value = function(*args)
    return value, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-record", required=True)
    parser.add_argument("--artifact")
    parser.add_argument("--sample-every", type=int, default=100)
    arguments = parser.parse_args(argv)
    if arguments.sample_every < 1:
        parser.error("--sample-every must be positive")

    runtime = None
    if arguments.artifact:
        from lisjong.learning import load_candidate_scorer_policy_factory

        runtime = load_candidate_scorer_policy_factory(arguments.artifact)
        policy = runtime()

    source = read_source_record(arguments.source_record)
    kinds: dict[str, int] = {}
    stages: dict[str, list[float]] = {
        "pass1_candidate_build": [],
        "two_pass_candidate_build": [],
        "candidate_numeric_encoding": [],
        "shared_context": [],
        "materialization_equivalent": [],
    }
    if runtime is not None:
        stages["torch_forward"] = []
        stages["policy_decide_total"] = []
    finalist_decisions = 0
    sampled = 0
    for ordinal, decision in enumerate(source.decisions()):
        context = DecisionContext(
            input=decision.policy_input, legal_actions=decision.legal_actions
        )
        kind = classify_o0_decision(context)
        kinds[kind.value] = kinds.get(kind.value, 0) + 1
        if kind is not O0DecisionKind.DISCARD or ordinal % arguments.sample_every:
            continue
        sampled += 1
        first, elapsed = _timed(build_discard_candidate_features, context)
        stages["pass1_candidate_build"].append(elapsed)
        finalist_decisions += int(bool(second_step_finalists(first)))
        candidates, two_pass = _timed(build_scorer_candidates, context)
        stages["two_pass_candidate_build"].append(two_pass)
        encoded, encoding = _timed(encode_candidates, candidates)
        stages["candidate_numeric_encoding"].append(encoding)
        shared, shared_elapsed = _timed(
            build_player_safe_feature, decision.policy_input
        )
        stages["shared_context"].append(shared_elapsed)
        stages["materialization_equivalent"].append(
            two_pass + encoding + shared_elapsed
        )
        if runtime is not None:
            _scores, forward = _timed(runtime.score, shared, encoded)
            stages["torch_forward"].append(forward)
            _result, total = _timed(policy.decide, context)
            stages["policy_decide_total"].append(total)

    if not sampled:
        raise SystemExit("no normal-discard decision was sampled")
    scorer_decisions = kinds.get(O0DecisionKind.DISCARD.value, 0)
    per_decision = statistics.fmean(stages["materialization_equivalent"])
    report = {
        "decision_kinds": dict(sorted(kinds.items())),
        "finalist_fraction": finalist_decisions / sampled,
        "materialization_projection": {
            "scorer_decisions": scorer_decisions,
            "scorer_decisions_per_second": 1.0 / per_decision,
            "projected_seconds": scorer_decisions * per_decision,
        },
        "python": sys.version.split()[0],
        "sampled_scorer_decisions": sampled,
        "source_identity": source.identity,
        "stages": {name: _summary(values) for name, values in stages.items()},
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
