"""放銃確率推定器とベースラインの学習・比較（#237 S1、学習専用の経路）。

ラベル付きの判断（`read_labelled_source`）を読み、次の手順で比較する。

1. ``select``: ベースライン1（一定値）とベースライン2（古典的危険度scoreのPlatt校正）を
   validで校正し、推定器（logistic回帰）をtrainで学習してL2強度をvalidのlog lossで選ぶ。
   選択結果とvalidの指標を書き出す
2. ``test``: 選択結果を固定したままtestで1回だけ評価する。出力は上書きしない

候補（判断×候補牌種）は独立な標本と扱わず、半荘単位のbootstrapで差のばらつきを示す。
ML runtimeには依存しない（離散特徴を集約したIRLS）。

```text
python -m lisjong.learning.riichi_deal_in_evaluation select SOURCE SELECTION.json
python -m lisjong.learning.riichi_deal_in_evaluation test SOURCE SELECTION.json RESULT.json
python -m lisjong.learning.riichi_deal_in_evaluation posthoc SOURCE SELECTION.json OUT.json
```

``posthoc``はtestを見た後の事後分析（比較条件をそろえたベースライン2、安全が確定していない
牌だけの評価、最安全牌の対比較）で、選択済みのモデルを変えない。
"""

import argparse
import json
import sys
from collections import defaultdict
from math import log
from pathlib import Path
from random import Random
from typing import Protocol, Sequence

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.learning._canonical import canonical_json_text, file_digest
from lisjong.learning.riichi_deal_in_estimator import (
    BIAS,
    FEATURE_SET,
    ClassicalScoreModel,
    ConstantModel,
    LogisticModel,
    _sigmoid,
    candidate_features,
    classical_score,
    riichi_view,
)
from lisjong.learning.riichi_deal_in_source import (
    MANIFEST_FILENAME,
    LabelledDecision,
    RiichiDealInManifest,
    read_labelled_source,
)
from lisjong.policy_contract.tile import TileCategory, TileType

SELECTION_SCHEMA = "lisjong-riichi-deal-in-selection-v1"
RESULT_SCHEMA = "lisjong-riichi-deal-in-test-result-v1"
L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 237
_EPSILON = 1e-15
_RELIABILITY_EDGES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)


class Model(Protocol):
    def predict(
        self, policy_input, candidates: tuple[TileType, ...]
    ) -> dict[TileType, float]: ...


# ---------------------------------------------------------------- fitting


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting（小さな正定値系）。"""
    size = len(vector)
    rows = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(rows[r][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        lead = rows[column][column]
        if abs(lead) < 1e-300:
            raise ValueError("singular system")
        for row in range(column + 1, size):
            factor = rows[row][column] / lead
            if factor:
                for k in range(column, size + 1):
                    rows[row][k] -= factor * rows[column][k]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = rows[row][size] - sum(
            rows[row][k] * solution[k] for k in range(row + 1, size)
        )
        solution[row] = total / rows[row][row]
    return solution


def fit_logistic(
    patterns: dict[tuple[tuple[int, float], ...], tuple[int, int]],
    size: int,
    l2: float,
    *,
    unpenalized: frozenset[int] = frozenset({0}),
) -> list[float]:
    """集約した疎な特徴（index, value）の(件数, 正例数)からridge logisticをIRLSで解く。"""
    weights = [0.0] * size
    for _ in range(100):
        gradient = [0.0 if i in unpenalized else l2 * weights[i] for i in range(size)]
        hessian = [
            [(0.0 if i in unpenalized else l2) if i == j else 0.0 for j in range(size)]
            for i in range(size)
        ]
        for features, (count, positives) in patterns.items():
            p = _sigmoid(sum(weights[i] * v for i, v in features))
            residual = count * p - positives
            curvature = count * p * (1.0 - p)
            for i, vi in features:
                gradient[i] += residual * vi
                for j, vj in features:
                    hessian[i][j] += curvature * vi * vj
        for i in range(size):
            hessian[i][i] += 1e-9
        step = _solve(hessian, gradient)
        weights = [w - s for w, s in zip(weights, step)]
        if max(abs(s) for s in step) < 1e-10:
            return weights
    raise ValueError("logistic fit did not converge")


def _candidates(decision: LabelledDecision) -> tuple[TileType, ...]:
    return tuple(candidate.tile_type for candidate in decision.candidates)


def fit_constant(decisions: Sequence[LabelledDecision]) -> ConstantModel:
    count = positives = 0
    for decision in decisions:
        view = riichi_view(decision.decision.policy_input)
        for candidate in decision.candidates:
            if candidate.tile_type not in view.genbutsu:
                count += 1
                positives += candidate.label_a
    if not count:
        raise ValueError("no non-genbutsu candidate to calibrate")
    return ConstantModel(probability=positives / count)


def fit_classical(
    decisions: Sequence[LabelledDecision], *, safe_zero: bool = False
) -> ClassicalScoreModel:
    """``safe_zero``なら構造的に安全な牌を除いて校正し、予測でも0にする。"""
    patterns: dict[tuple[tuple[int, float], ...], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    for decision in decisions:
        policy_input = decision.decision.policy_input
        view = riichi_view(policy_input)
        for candidate in decision.candidates:
            if safe_zero and candidate.tile_type in view.structurally_safe:
                continue
            score = classical_score(policy_input, view, candidate.tile_type)
            entry = patterns[((0, 1.0), (1, float(score)))]
            entry[0] += 1
            entry[1] += candidate.label_a
    intercept, slope = fit_logistic(
        {key: tuple(value) for key, value in patterns.items()},
        2,
        0.0,
        unpenalized=frozenset({0, 1}),
    )
    return ClassicalScoreModel(intercept=intercept, slope=slope, safe_zero=safe_zero)


def fit_estimator(decisions: Sequence[LabelledDecision], l2: float) -> LogisticModel:
    counts: dict[tuple[tuple[str, float], ...], list[int]] = defaultdict(lambda: [0, 0])
    for decision in decisions:
        policy_input = decision.decision.policy_input
        view = riichi_view(policy_input)
        for candidate in decision.candidates:
            if candidate.tile_type in view.structurally_safe:
                continue
            entry = counts[candidate_features(policy_input, view, candidate.tile_type)]
            entry[0] += 1
            entry[1] += candidate.label_a
    names = sorted({name for features in counts for name, _ in features} - {BIAS})
    index = {BIAS: 0} | {name: position + 1 for position, name in enumerate(names)}
    patterns = {
        tuple(sorted((index[name], value) for name, value in features)): tuple(value)
        for features, value in counts.items()
    }
    weights = fit_logistic(patterns, len(index), l2)
    return LogisticModel(
        weights=tuple((name, weights[position]) for name, position in index.items())
    )


# ---------------------------------------------------------------- metrics


def _tile_class(tile: TileType, view) -> str:
    if tile.category is TileCategory.HONOR:
        return "honor_yakuhai" if tile in view.yakuhai else "honor_guest"
    return {1: "n19", 9: "n19", 2: "n28", 8: "n28", 3: "n37", 7: "n37"}.get(
        tile.rank, "n456"
    )


def _turn(count: int) -> str:
    return "early(<=6)" if count <= 6 else "middle(7-11)" if count <= 11 else "late"


class _Accumulator:
    def __init__(self) -> None:
        self.count = 0
        self.positives = 0
        self.log_loss = 0.0
        self.brier = 0.0
        self.probability = 0.0

    def add(self, probability: float, label: bool) -> None:
        if probability == 0.0 and label:
            raise ValueError("a ron tile was predicted with probability 0")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability outside [0, 1]")
        self.count += 1
        self.positives += label
        clipped = min(max(probability, _EPSILON), 1.0 - _EPSILON)
        if label:
            self.log_loss -= log(clipped)
        elif probability > 0.0:
            self.log_loss -= log(1.0 - clipped)
        self.brier += (probability - label) ** 2
        self.probability += probability

    def summary(self) -> dict[str, object]:
        if not self.count:
            return {"candidates": 0}
        return {
            "candidates": self.count,
            "positives": self.positives,
            "positive_rate": self.positives / self.count,
            "mean_probability": self.probability / self.count,
            "log_loss": self.log_loss / self.count,
            "brier": self.brier / self.count,
        }


def evaluate(
    model: Model,
    decisions: Sequence[LabelledDecision],
    *,
    exclude_safe: bool = False,
) -> dict[str, object]:
    """候補単位の指標・校正・判断内の順位付けを、半荘別の合計とともに返す。

    ``exclude_safe``なら構造的に安全な牌を候補から除き、安全が確定していない牌だけで
    評価する（候補が残らない判断は数えない）。
    """
    evaluated = 0
    overall = _Accumulator()
    groups: dict[str, dict[str, _Accumulator]] = defaultdict(
        lambda: defaultdict(_Accumulator)
    )
    bins = [[0, 0, 0.0] for _ in range(len(_RELIABILITY_EDGES) - 1)]
    per_game: dict[int, _Accumulator] = defaultdict(_Accumulator)
    auc_sum = 0.0
    auc_decisions = 0
    pick_hits = 0.0
    for decision in decisions:
        policy_input = decision.decision.policy_input
        view = riichi_view(policy_input)
        labels = {
            c.tile_type: c.label_a
            for c in decision.candidates
            if not (exclude_safe and c.tile_type in view.structurally_safe)
        }
        if not labels:
            continue
        evaluated += 1
        predictions = model.predict(policy_input, tuple(labels))
        if set(predictions) != set(labels):
            raise ValueError("the model did not score every candidate")
        for tile, label in labels.items():
            probability = predictions[tile]
            overall.add(probability, label)
            per_game[decision.decision.key.seed].add(probability, label)
            groups["tile_class"][_tile_class(tile, view)].add(probability, label)
            groups["riichi_turn"][_turn(view.riichi_discard_count)].add(
                probability, label
            )
            groups["structurally_safe"][str(tile in view.structurally_safe)].add(
                probability, label
            )
            groups["remaining"][str(view.remaining_counts[tile_type_index(tile)])].add(
                probability, label
            )
            for position in range(len(bins)):
                if probability <= _RELIABILITY_EDGES[position + 1]:
                    bins[position][0] += 1
                    bins[position][1] += label
                    bins[position][2] += probability
                    break
        positives = [predictions[t] for t, label in labels.items() if label]
        negatives = [predictions[t] for t, label in labels.items() if not label]
        if positives and negatives:
            auc_decisions += 1
            auc_sum += sum(
                1.0 if p > n else 0.5 if p == n else 0.0
                for p in positives
                for n in negatives
            ) / (len(positives) * len(negatives))
        lowest = min(predictions.values())
        tied = [t for t, p in predictions.items() if p == lowest]
        pick_hits += sum(labels[t] for t in tied) / len(tied)
    return {
        "decisions": evaluated,
        "excluded_structurally_safe": exclude_safe,
        "overall": overall.summary(),
        "groups": {
            name: {key: value.summary() for key, value in sorted(members.items())}
            for name, members in sorted(groups.items())
        },
        "reliability": [
            {
                "upper": _RELIABILITY_EDGES[position + 1],
                "candidates": count,
                "positive_rate": positives / count if count else None,
                "mean_probability": total / count if count else None,
            }
            for position, (count, positives, total) in enumerate(bins)
        ],
        "ranking": {
            "within_decision_auc": auc_sum / auc_decisions if auc_decisions else None,
            "decisions_with_both_labels": auc_decisions,
            "lowest_risk_pick_ron_rate": pick_hits / evaluated if evaluated else None,
        },
        "per_game": {
            str(seed): {
                "candidates": acc.count,
                "log_loss_sum": acc.log_loss,
                "brier_sum": acc.brier,
            }
            for seed, acc in sorted(per_game.items())
        },
    }


def bootstrap_difference(
    candidate: dict[str, object], reference: dict[str, object], metric: str
) -> dict[str, float]:
    """半荘を単位に復元抽出し、(candidate - reference)の平均差の分布を返す。"""
    a, b = candidate["per_game"], reference["per_game"]
    if set(a) != set(b):
        raise ValueError("the two evaluations cover different games")
    games = sorted(a)
    key = f"{metric}_sum"

    def difference(sample: list[str]) -> float:
        count = sum(a[g]["candidates"] for g in sample)
        return (sum(a[g][key] for g in sample) - sum(b[g][key] for g in sample)) / count

    random = Random(BOOTSTRAP_SEED)
    draws = sorted(
        difference([random.choice(games) for _ in games])
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "point": difference(games),
        "low_2.5": draws[int(0.025 * BOOTSTRAP_RESAMPLES)],
        "high_97.5": draws[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
        "games": len(games),
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def _lowest_risk_pick(
    predictions: dict[TileType, float], labels: dict[TileType, bool]
) -> tuple[frozenset[TileType], float]:
    """最も安全と推定した牌の集合（同点を含む）と、そのロン牌率（同点は平均）。"""
    lowest = min(predictions.values())
    tied = frozenset(t for t, p in predictions.items() if p == lowest)
    return tied, sum(labels[t] for t in tied) / len(tied)


def paired_lowest_risk(
    first: Model, second: Model, decisions: Sequence[LabelledDecision]
) -> dict[str, object]:
    """同じ判断で、2モデルが最も安全と推定した牌を対にして比べる。

    選んだ牌（同点の集合）が異なる判断をすべて数え、結果で分類する。差のばらつきは
    半荘を単位にしたpaired bootstrapで示す（判断を独立な標本と扱わない）。
    """
    outcome = dict.fromkeys(
        ("both_safe", "both_ron", "first_worse", "second_worse", "equal_partial"), 0
    )
    different_pick = 0
    first_total = second_total = 0.0
    per_game: dict[int, list[float]] = defaultdict(lambda: [0, 0.0])
    for decision in decisions:
        policy_input = decision.decision.policy_input
        candidates = _candidates(decision)
        labels = {c.tile_type: c.label_a for c in decision.candidates}
        pick_a, a = _lowest_risk_pick(first.predict(policy_input, candidates), labels)
        pick_b, b = _lowest_risk_pick(second.predict(policy_input, candidates), labels)
        first_total += a
        second_total += b
        game = per_game[decision.decision.key.seed]
        game[0] += 1
        game[1] += a - b
        if pick_a == pick_b:
            continue
        different_pick += 1
        if a > b:
            outcome["first_worse"] += 1
        elif b > a:
            outcome["second_worse"] += 1
        elif a == 0.0:
            outcome["both_safe"] += 1
        elif a == 1.0:
            outcome["both_ron"] += 1
        else:
            outcome["equal_partial"] += 1
    games = sorted(per_game)

    def difference(sample: list[int]) -> float:
        count = sum(per_game[g][0] for g in sample)
        return sum(per_game[g][1] for g in sample) / count

    random = Random(BOOTSTRAP_SEED)
    draws = sorted(
        difference([random.choice(games) for _ in games])
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "decisions": len(decisions),
        "first_expected_ron_picks": first_total,
        "second_expected_ron_picks": second_total,
        "decisions_with_different_pick": different_pick,
        "different_pick_outcomes": outcome,
        "first_minus_second_ron_pick_rate": {
            "point": difference(games),
            "low_2.5": draws[int(0.025 * BOOTSTRAP_RESAMPLES)],
            "high_97.5": draws[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
            "games": len(games),
            "resamples": BOOTSTRAP_RESAMPLES,
        },
    }


# ---------------------------------------------------------------- phases


def _split(
    manifest: RiichiDealInManifest, decisions: Sequence[LabelledDecision], name: str
) -> tuple[LabelledDecision, ...]:
    seeds = frozenset(manifest.splits[name])
    return tuple(d for d in decisions if d.decision.key.seed in seeds)


def _source_identity(directory: Path) -> dict[str, object]:
    return file_digest(directory / MANIFEST_FILENAME)


def _write_new(path: Path, document: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(canonical_json_text(document), encoding="utf-8")


def _models_to_value(constant, classical, estimator) -> dict[str, object]:
    return {
        "baseline1_genbutsu_zero_constant": {"probability": constant.probability},
        "baseline2_classical_platt": {
            "intercept": classical.intercept,
            "slope": classical.slope,
        },
        "estimator_logistic": {
            "feature_set": estimator.feature_set,
            "weights": dict(estimator.weights),
        },
    }


def _models_from_value(value: dict[str, object]):
    return (
        ConstantModel(**value["baseline1_genbutsu_zero_constant"]),
        ClassicalScoreModel(**value["baseline2_classical_platt"]),
        LogisticModel(
            weights=tuple(value["estimator_logistic"]["weights"].items()),
            feature_set=value["estimator_logistic"]["feature_set"],
        ),
    )


_MODEL_NAMES = (
    "baseline1_genbutsu_zero_constant",
    "baseline2_classical_platt",
    "estimator_logistic",
)


def _comparison(evaluations: dict[str, dict[str, object]]) -> dict[str, object]:
    reference = evaluations["baseline2_classical_platt"]
    return {
        f"{name}_minus_baseline2": {
            metric: bootstrap_difference(evaluations[name], reference, metric)
            for metric in ("log_loss", "brier")
        }
        for name in ("estimator_logistic", "baseline1_genbutsu_zero_constant")
    }


def select(source: Path, output: Path) -> dict[str, object]:
    manifest, decisions = read_labelled_source(source)
    train = _split(manifest, decisions, "train")
    valid = _split(manifest, decisions, "valid")
    constant = fit_constant(valid)
    classical = fit_classical(valid)
    grid = []
    for l2 in L2_GRID:
        model = fit_estimator(train, l2)
        grid.append((evaluate(model, valid)["overall"]["log_loss"], l2, model))
    _, chosen_l2, estimator = min(grid, key=lambda item: (item[0], item[1]))
    models = (constant, classical, estimator)
    evaluations = {name: evaluate(m, valid) for name, m in zip(_MODEL_NAMES, models)}
    document = {
        "schema": SELECTION_SCHEMA,
        "source_manifest": _source_identity(source),
        "feature_set": FEATURE_SET,
        "fitted_on": {
            "baselines": "valid",
            "estimator": "train",
            "l2_selected_on": "valid",
        },
        "l2_grid": [{"l2": l2, "valid_log_loss": loss} for loss, l2, _ in grid],
        "chosen_l2": chosen_l2,
        "models": _models_to_value(*models),
        "split_sizes": {
            name: len(_split(manifest, decisions, name))
            for name in ("train", "valid", "test")
        },
        "valid": evaluations,
        "valid_comparison": _comparison(evaluations),
    }
    _write_new(output, document)
    return document


def test(source: Path, selection: Path, output: Path) -> dict[str, object]:
    chosen = json.loads(selection.read_text(encoding="utf-8"))
    if chosen.get("schema") != SELECTION_SCHEMA:
        raise ValueError("not a selection document")
    if chosen["source_manifest"] != _source_identity(source):
        raise ValueError("the selection was made on a different source")
    manifest, decisions = read_labelled_source(source)
    evaluations = {
        name: evaluate(model, _split(manifest, decisions, "test"))
        for name, model in zip(_MODEL_NAMES, _models_from_value(chosen["models"]))
    }
    document = {
        "schema": RESULT_SCHEMA,
        "source_manifest": chosen["source_manifest"],
        "selection": file_digest(selection),
        "test": evaluations,
        "test_comparison": _comparison(evaluations),
    }
    _write_new(output, document)
    return document


POSTHOC_SCHEMA = "lisjong-riichi-deal-in-posthoc-v2"


def posthoc(source: Path, selection: Path, output: Path) -> dict[str, object]:
    """事後分析: 選択を変えずに、比較条件をそろえたベースラインと非安全牌だけの評価を行う。

    ベースライン2の変種（構造的に安全な牌=0、残りをvalidで校正）を加える。選択済みの
    モデルは変更しない。testを見た後の分析であり、ここから改良したモデルの最終評価には
    新しい保留データが必要。
    """
    chosen = json.loads(selection.read_text(encoding="utf-8"))
    if chosen.get("schema") != SELECTION_SCHEMA:
        raise ValueError("not a selection document")
    if chosen["source_manifest"] != _source_identity(source):
        raise ValueError("the selection was made on a different source")
    manifest, decisions = read_labelled_source(source)
    valid = _split(manifest, decisions, "valid")
    _, classical, estimator = _models_from_value(chosen["models"])
    aligned = fit_classical(valid, safe_zero=True)
    models = {
        "baseline2_classical_platt": classical,
        "baseline2_safe_zero": aligned,
        "estimator_logistic": estimator,
    }
    document: dict[str, object] = {
        "schema": POSTHOC_SCHEMA,
        "posthoc": True,
        "source_manifest": chosen["source_manifest"],
        "selection": file_digest(selection),
        "baseline2_safe_zero": {
            "intercept": aligned.intercept,
            "slope": aligned.slope,
            "fitted_on": "valid, non-structurally-safe candidates",
        },
    }
    for split in ("valid", "test"):
        rows = _split(manifest, decisions, split)
        result: dict[str, object] = {}
        for scope, exclude in (("all", False), ("non_safe", True)):
            evaluations = {
                name: evaluate(model, rows, exclude_safe=exclude)
                for name, model in models.items()
            }
            reference = evaluations["baseline2_safe_zero"]
            result[scope] = {
                "summary": {
                    name: evaluation["overall"] | evaluation["ranking"]
                    for name, evaluation in evaluations.items()
                },
                "estimator_minus_baseline2_safe_zero": {
                    metric: bootstrap_difference(
                        evaluations["estimator_logistic"], reference, metric
                    )
                    for metric in ("log_loss", "brier")
                },
            }
        result["paired_lowest_risk"] = {
            "estimator_vs_baseline2_safe_zero": paired_lowest_risk(
                estimator, aligned, rows
            ),
            "estimator_vs_baseline2": paired_lowest_risk(estimator, classical, rows),
        }
        document[split] = result
    _write_new(output, document)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=__name__)
    commands = parser.add_subparsers(dest="command", required=True)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("source", type=Path)
    select_parser.add_argument("output", type=Path)
    test_parser = commands.add_parser("test")
    test_parser.add_argument("source", type=Path)
    test_parser.add_argument("selection", type=Path)
    test_parser.add_argument("output", type=Path)
    posthoc_parser = commands.add_parser("posthoc")
    posthoc_parser.add_argument("source", type=Path)
    posthoc_parser.add_argument("selection", type=Path)
    posthoc_parser.add_argument("output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "posthoc":
        document = posthoc(arguments.source, arguments.selection, arguments.output)
        json.dump(
            {split: document[split] for split in ("valid", "test")},
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        print()
        return 0
    if arguments.command == "select":
        document = select(arguments.source, arguments.output)
        summary = {"chosen_l2": document["chosen_l2"], "split": "valid"}
        evaluations = document["valid"]
    else:
        document = test(arguments.source, arguments.selection, arguments.output)
        summary = {"split": "test"}
        evaluations = document["test"]
    summary |= {
        name: evaluations[name]["overall"] | evaluations[name]["ranking"]
        for name in _MODEL_NAMES
    }
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
