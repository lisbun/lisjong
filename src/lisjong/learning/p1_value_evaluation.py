"""P1価値モデルのfitとTEST評価（E2 / E3、Issue #366、設計 #365）。

```text
fit       TRAIN-1〜3 -> artifact（immutable directory）
evaluate  artifact + TEST-1〜3（headerがそのartifact identityを束縛）-> 評価記録（write-once、1回）
```

推定量（TESTの全成立分岐、Δは100点単位、m = TRAIN平均、ŷ = 凍結ridgeの予測）:

- E2（主）: ΔMSE = mean[(Δ − m)² − (Δ − ŷ)²]
- E3（副）: V = mean[I(ŷ > 0)·Δ]（Passへ替えると予測した分岐の差）。成立分岐あたりで判定し、
  全対象判断あたり（ロン取消の候補を分母に含め、その寄与は0）も記録する
- bootstrap: TESTの120 seeds（昇順）。統計量ごとに``random.Random(365)``を新しく初期化し、
  ``randrange(120)``を120回の復元抽出を10,000回。抽出seedの全行を重複込みでまとめて分子（合計）と
  分母（行数）を再計算する。P5 = 昇順500番目（nearest rank）。**P5 > 0で通過**
- 判定: E2不通過 -> NOT-PREDICTIVE。E2通過・E3不通過 -> PREDICTIVE-ONLY。
  両方通過 -> DECISION-VALUE-SIGNAL（E4の設計候補で、採用の根拠ではない）
- 同じartifactでの評価記録は1つだけ（write-once）。TESTで再fitしない
"""

import argparse
import json
import math
import sys
from pathlib import Path

from lisjong.learning._canonical import canonical_json_text, seal
from lisjong.learning._publication import write_new_text
from lisjong.learning.p1_value import (
    SPLIT_SEEDS,
    TEST,
    TRAIN,
    P1ValueError,
    build_artifact,
    encode_row,
    load_artifact,
    read_source,
    rows_of,
    save_artifact,
)
from lisjong.learning.r1_value_evaluation import P5_RANK, cluster_bootstrap

EVALUATION_SCHEMA = "lisjong-p1-value-evaluation-v1"
BOOTSTRAP_SEED = 365
BOOTSTRAP_REPLICATES = 10_000
NOT_PREDICTIVE = "NOT-PREDICTIVE"
PREDICTIVE_ONLY = "PREDICTIVE-ONLY"
DECISION_VALUE_SIGNAL = "DECISION-VALUE-SIGNAL"


def _fail(message: str):
    raise P1ValueError(message)


def evaluation_values(artifact, rows) -> list[dict]:
    """行ごとの値（E2の差、E3の寄与、予測でPassへ替えるか）。"""
    values = []
    for row in rows:
        prediction = artifact.predict(encode_row(row))
        switch = prediction > 0
        values.append(
            {
                "branch_id": row.branch_id,
                "delta": row.delta,
                "e2": (row.delta - artifact.train_mean) ** 2
                - (row.delta - prediction) ** 2,
                "e3": float(row.delta) if switch else 0.0,
                "prediction": prediction,
                "seed": row.seed,
                "switch": switch,
            }
        )
    return values


def _bootstrap(seeds, values, key) -> dict:
    by_seed = {seed: ([], 0) for seed in seeds}
    for item in values:
        parts, size = by_seed[item["seed"]]
        by_seed[item["seed"]] = ([*parts, item[key]], size + 1)
    clusters = {
        seed: (math.fsum(parts), size) for seed, (parts, size) in by_seed.items()
    }
    return cluster_bootstrap(seeds, clusters, rng_seed=BOOTSTRAP_SEED)


def evaluate(artifact, test_blocks) -> dict:
    """TEST blockを評価して記録documentを返す（保存はcaller）。"""
    if any(block.split != TEST for block in test_blocks):
        _fail("evaluation reads TEST blocks only")
    if {block.artifact_identity for block in test_blocks} != {artifact.identity}:
        _fail("the TEST blocks were not generated for this artifact")
    rows = rows_of(test_blocks)
    if not rows:
        _fail("TEST has no rows")
    values = evaluation_values(artifact, rows)
    seeds = SPLIT_SEEDS[TEST]
    n = len(values)
    cancelled = sum(block.ron_cancelled for block in test_blocks)
    e2 = {
        "estimate": math.fsum(item["e2"] for item in values) / n,
        "bootstrap": _bootstrap(seeds, values, "e2"),
    }
    e3_sum = math.fsum(item["e3"] for item in values)
    e3 = {
        "estimate": e3_sum / n,
        "bootstrap": _bootstrap(seeds, values, "e3"),
        "per_target_decision": e3_sum / (n + cancelled),
    }
    e2_pass = e2["bootstrap"]["P5"] > 0
    e3_pass = e2_pass and e3["bootstrap"]["P5"] > 0
    if not e2_pass:
        decision = NOT_PREDICTIVE
    elif not e3_pass:
        decision = PREDICTIVE_ONLY
    else:
        decision = DECISION_VALUE_SIGNAL
    switched = [item for item in values if item["switch"]]
    body = {
        "artifact_identity": artifact.identity,
        "bootstrap": {
            "p5_rank": P5_RANK,
            "replicates": BOOTSTRAP_REPLICATES,
            "rng": f"random.Random({BOOTSTRAP_SEED}) per statistic",
            "seeds": list(seeds),
        },
        "decision": decision,
        "e2": {**e2, "pass": e2_pass},
        "e3": {**e3, "pass": e3_pass},
        "not_a_strength_result": True,
        "population": {
            "branches": n,
            "ron_cancelled": cancelled,
            "target_decisions": n + cancelled,
        },
        "rates": {"predicted_switch": len(switched) / n},
        "rows": n,
        "schema": EVALUATION_SCHEMA,
        "seeds_with_rows": len({item["seed"] for item in values}),
        "switched_mean_delta": (
            math.fsum(float(item["delta"]) for item in switched) / len(switched)
            if switched
            else None
        ),
        "test_blocks": [block.identity() for block in test_blocks],
    }
    return seal(body)


def evaluation_path(record_dir: str | Path, artifact_identity: str) -> Path:
    return Path(record_dir) / f"evaluation-{artifact_identity}.json"


def run_evaluation(artifact_dir, test_dirs, record_dir) -> dict:
    """同じartifactの評価記録が既にあれば拒否する（TESTを1回だけ読む）。"""
    artifact = load_artifact(artifact_dir)
    destination = evaluation_path(record_dir, artifact.identity)
    if destination.exists():
        _fail(f"this artifact has already been evaluated: {destination}")
    document = evaluate(artifact, read_source(test_dirs, TEST))
    Path(record_dir).mkdir(parents=True, exist_ok=True)
    write_new_text(destination, canonical_json_text(document), P1ValueError)
    return document


def run_fit(train_dirs, output) -> dict:
    document = build_artifact(read_source(train_dirs, TRAIN))
    save_artifact(document, output)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lisjong.learning.p1_value_evaluation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser(
        "fit", help="fit the ridge on TRAIN-1..3 and save the artifact"
    )
    fit.add_argument("--train", nargs="+", required=True)
    fit.add_argument("--output", required=True)
    evaluation = sub.add_parser("evaluate", help="evaluate once on TEST-1..3")
    evaluation.add_argument("--artifact", required=True)
    evaluation.add_argument("--test", nargs="+", required=True)
    evaluation.add_argument("--record-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "fit":
        document = run_fit(args.train, args.output)
        print(json.dumps({"identity": document["identity"], **document["diagnostics"]}))
        return 0
    document = run_evaluation(args.artifact, args.test, args.record_dir)
    print(
        json.dumps({"decision": document["decision"], "identity": document["identity"]})
    )
    return 0


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "DECISION_VALUE_SIGNAL",
    "EVALUATION_SCHEMA",
    "NOT_PREDICTIVE",
    "PREDICTIVE_ONLY",
    "evaluate",
    "evaluation_path",
    "evaluation_values",
    "main",
    "run_evaluation",
    "run_fit",
]


if __name__ == "__main__":
    sys.exit(main())
