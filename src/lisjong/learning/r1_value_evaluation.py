"""R1価値モデルのfitとTEST評価（E2 / E3、Issue #360、設計 #356）。

```text
fit       TRAIN-1・TRAIN-2 -> artifact（immutable directory）
evaluate  artifact + TEST-1・TEST-2（headerがそのartifact identityを束縛）-> 評価記録（write-once、1回）
```

推定量（TESTの全分岐、Δは100点単位、m = TRAIN平均、ŷ = 凍結ridgeの予測）:

- E2（主）: ΔMSE = mean[(Δ − m)² − (Δ − ŷ)²]
- E3（副）: V = mean[I(推定宣言牌 = 記録の宣言牌 かつ dama_choice = 記録の宣言牌 かつ ŷ > 0)·Δ]
- bootstrap: TESTの44 seeds（昇順）。統計量ごとに``random.Random(356)``を新しく初期化し、
  ``randrange(44)``を44回の復元抽出を10,000回。抽出seedの全行を重複込みでまとめて分子（合計）と
  分母（行数）を再計算する。P5 = 昇順500番目（nearest rank）。**P5 > 0で通過**
- 判定: E2不通過 -> NOT-PREDICTIVE。E2通過・E3不通過（またはSUPPORT-INVALID）-> PREDICTIVE-ONLY。
  両方通過 -> DECISION-VALUE-SIGNAL（E4の設計候補で、採用の根拠ではない）
- 特徴は記録の宣言牌を基準に作る（学習と同じ）。推定宣言牌と記録が違う行は、推論時に介入しない行として
  E3で数えない。いずれかの不一致がTEST分岐の2%を超えたらE3はSUPPORT-INVALID
- 同じartifactでの評価記録は1つだけ（write-once）。TESTで再fitしない
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

from lisjong.learning._canonical import canonical_json_text, seal
from lisjong.learning._publication import write_new_text
from lisjong.learning.r1_value import (
    SPLIT_SEEDS,
    TEST,
    TRAIN,
    R1ValueError,
    analyze,
    build_artifact,
    encode,
    load_artifact,
    project_declaration,
    read_source,
    rows_of,
    save_artifact,
)

EVALUATION_SCHEMA = "lisjong-r1-value-evaluation-v1"
BOOTSTRAP_SEED = 356
BOOTSTRAP_REPLICATES = 10_000
P5_RANK = 500
"""昇順でのP5の位置（nearest rank: ceil(0.05 × 10,000) = 500番目）。"""
SUPPORT_LIMIT = 0.02
NOT_PREDICTIVE = "NOT-PREDICTIVE"
PREDICTIVE_ONLY = "PREDICTIVE-ONLY"
DECISION_VALUE_SIGNAL = "DECISION-VALUE-SIGNAL"


def _fail(message: str):
    raise R1ValueError(message)


def cluster_bootstrap(seeds, rows_by_seed) -> dict:
    """``rows_by_seed[seed]`` は (合計, 行数)。比 = 抽出seedの合計の和 / 行数の和。"""
    seeds = sorted(seeds)
    m = len(seeds)
    rng = random.Random(BOOTSTRAP_SEED)
    statistics = []
    for _ in range(BOOTSTRAP_REPLICATES):
        total = 0.0
        count = 0
        parts = []
        for _ in range(m):
            value, size = rows_by_seed[seeds[rng.randrange(m)]]
            parts.append(value)
            count += size
        if count == 0:
            _fail("a bootstrap replicate drew no rows")
        total = math.fsum(parts)
        statistics.append(total / count)
    statistics.sort()
    return {
        "P5": statistics[P5_RANK - 1],
        "P50": statistics[5000 - 1],
        "P95": statistics[9500 - 1],
    }


def evaluation_values(artifact, rows) -> dict:
    """行ごとの値（E2の差、E3の寄与、介入・不一致の指標）を作る。"""
    values = []
    for row in rows:
        features = analyze(row.decision, row.declaration)
        prediction = artifact.predict(encode(row.decision, features))
        projected = project_declaration(row.decision)
        projected_match = projected == row.declaration
        dama_match = features.dama_choice == row.declaration
        intervene = projected_match and dama_match and prediction > 0
        values.append(
            {
                "branch_id": row.branch_id,
                "delta": row.delta,
                "e2": (row.delta - artifact.train_mean) ** 2
                - (row.delta - prediction) ** 2,
                "e3": float(row.delta) if intervene else 0.0,
                "dama_match": dama_match,
                "intervene": intervene,
                "predicted_positive": prediction > 0,
                "prediction": prediction,
                "projected_match": projected_match,
                "seed": row.seed,
            }
        )
    return values


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

    def clusters(key):
        by_seed = {seed: ([], 0) for seed in seeds}
        for item in values:
            parts, size = by_seed[item["seed"]]
            by_seed[item["seed"]] = ([*parts, item[key]], size + 1)
        return {
            seed: (math.fsum(parts), size) for seed, (parts, size) in by_seed.items()
        }

    e2 = {
        "estimate": math.fsum(item["e2"] for item in values) / n,
        "bootstrap": cluster_bootstrap(seeds, clusters("e2")),
    }
    e3 = {
        "estimate": math.fsum(item["e3"] for item in values) / n,
        "bootstrap": cluster_bootstrap(seeds, clusters("e3")),
    }
    projected_mismatch = sum(not item["projected_match"] for item in values)
    dama_mismatch = sum(not item["dama_match"] for item in values)
    any_mismatch = sum(
        not (item["projected_match"] and item["dama_match"]) for item in values
    )
    support_invalid = any_mismatch / n > SUPPORT_LIMIT
    e2_pass = e2["bootstrap"]["P5"] > 0
    e3_pass = e2_pass and not support_invalid and e3["bootstrap"]["P5"] > 0
    if not e2_pass:
        decision = NOT_PREDICTIVE
    elif not e3_pass:
        decision = PREDICTIVE_ONLY
    else:
        decision = DECISION_VALUE_SIGNAL
    intervened = [item for item in values if item["intervene"]]
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
        "e3": {**e3, "pass": e3_pass, "support_invalid": support_invalid},
        "not_a_strength_result": True,
        "rates": {
            "actual_intervention": len(intervened) / n,
            "any_mismatch": any_mismatch / n,
            "dama_choice_mismatch": dama_mismatch / n,
            "predicted_intervention": sum(item["predicted_positive"] for item in values)
            / n,
            "projected_declaration_mismatch": projected_mismatch / n,
        },
        "rows": n,
        "intervened_mean_delta": (
            math.fsum(float(item["delta"]) for item in intervened) / len(intervened)
            if intervened
            else None
        ),
        "schema": EVALUATION_SCHEMA,
        "seeds_with_rows": len({item["seed"] for item in values}),
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
    write_new_text(destination, canonical_json_text(document), R1ValueError)
    return document


def run_fit(train_dirs, output) -> dict:
    document = build_artifact(read_source(train_dirs, TRAIN))
    save_artifact(document, output)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lisjong.learning.r1_value_evaluation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser(
        "fit", help="fit the ridge on TRAIN-1 / TRAIN-2 and save the artifact"
    )
    fit.add_argument("--train", nargs="+", required=True)
    fit.add_argument("--output", required=True)
    evaluation = sub.add_parser("evaluate", help="evaluate once on TEST-1 / TEST-2")
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
    "P5_RANK",
    "PREDICTIVE_ONLY",
    "SUPPORT_LIMIT",
    "cluster_bootstrap",
    "evaluate",
    "evaluation_path",
    "evaluation_values",
    "main",
    "run_evaluation",
    "run_fit",
]


if __name__ == "__main__":
    sys.exit(main())
