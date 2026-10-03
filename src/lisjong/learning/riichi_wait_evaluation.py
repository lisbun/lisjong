"""構造的待ち確率の推定器とベースラインの学習・比較（lisbun/lisjong#245、学習専用の経路）。

ラベル付きの判断（`read_wait_labelled_source`）を読み、次の手順で比較する。

1. ``select``: S1のsource（train / valid）で、ベースライン1（牌種ごとの待ち率）・ベースライン2
   （古典的危険度scoreのPlatt校正）・推定器（logistic回帰）の係数をtrainで当てはめ、
   推定器のL2強度をvalidのlossで選ぶ。選択結果とvalidの指標を書き出す
2. ``test``: 新しいseedのsourceの`test`分割で、選択を固定したまま1回だけ評価する。
   選択に使ったsourceのseed（train / valid / test）と重なるseedは拒否する。出力は上書きしない

```text
python -m lisjong.learning.riichi_wait_evaluation select SOURCE SELECTION.json
python -m lisjong.learning.riichi_wait_evaluation test SOURCE SELECTION.json RESULT.json
```

## 重みと集約（学習・校正・評価で共通）

リーチは重み付けの単位、半荘は再標本化の単位とする。同じリーチを見た判断行（観測者が複数でも）
は、そのリーチの重み1を等分する。損失は、1判断の34牌種で平均し、同一リーチ内の行で平均し、
リーチ間で平均する。

学習の目的関数は ``sum_e sum_{r in e} (1/N_e) sum_t loss(r, t) + (l2/2) * ||w||^2``
（``N_e``はそのsplit内でリーチ``e``を見た判断行の数、切片は正則化しない）で、
34牌種の平均ではなく和を使う（係数が定数倍違うだけで、L2強度の格子は事前に固定する）。

## 比較と合格

``Δ = 推定器のloss - ベースラインのloss``（負なら推定器が良い）。半荘を単位に復元抽出する
paired bootstrapで、ベースライン1・2それぞれとのΔの95%区間を求める。再標本化した後も、
リーチ間の平均を計算し直す。**合格は、両方のΔの区間の上端が0未満**であること。
区間が0をまたぐ場合は「改善を確認できなかった」であり、「効果がない」とは限らない。

ML runtimeには依存しない（離散特徴を集約したIRLS）。
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from math import log
from pathlib import Path
from random import Random
from typing import Protocol

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.learning._canonical import canonical_json_text, file_digest
from lisjong.learning.riichi_deal_in_estimator import riichi_view
from lisjong.learning.riichi_deal_in_evaluation import fit_logistic
from lisjong.learning.riichi_deal_in_source import (
    MANIFEST_FILENAME,
    RiichiDealInManifest,
    RiichiEpisodeKey,
    WaitLabelledDecision,
    read_wait_labelled_source,
)
from lisjong.learning.riichi_wait_estimator import (
    BIAS,
    CLIP_EPSILON,
    FEATURE_SET,
    SCORE,
    TILE_TYPES,
    ClassicalScoreWaitModel,
    LogisticWaitModel,
    PrevalenceWaitModel,
    classical_wait_score,
    wait_feature_table,
)
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import TileCategory, TileType

SELECTION_SCHEMA = "lisjong-riichi-wait-selection-v1"
RESULT_SCHEMA = "lisjong-riichi-wait-test-result-v1"
L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 245
JEFFREYS_PRIOR = 0.5
_RELIABILITY_EDGES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
MODEL_NAMES = (
    "baseline1_prevalence",
    "baseline2_classical_platt",
    "estimator_logistic",
)


class Model(Protocol):
    def predict(self, policy_input: PolicyInput) -> dict[TileType, float]: ...


# ---------------------------------------------------------------- weights


def row_weights(decisions: Sequence[WaitLabelledDecision]) -> list[float]:
    """判断行ごとの重み。同じリーチを見た行（観測者が複数でも）で、そのリーチの重み1を等分する。"""
    counts: dict[RiichiEpisodeKey, int] = defaultdict(int)
    for decision in decisions:
        counts[decision.episode] += 1
    return [1.0 / counts[decision.episode] for decision in decisions]


# ---------------------------------------------------------------- fitting


def _patterns(
    rows: Sequence[tuple[tuple[tuple[str, float], ...], bool, float]],
    names: Sequence[str],
) -> dict[tuple[tuple[int, float], ...], tuple[float, float]]:
    """(特徴, ラベル, 重み)の行を、同じ特徴ごとに(重みの和, 正例の重みの和)へ集約する。"""
    index = {name: position for position, name in enumerate(names)}
    patterns: dict[tuple[tuple[int, float], ...], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    for features, label, weight in rows:
        key = tuple(sorted((index[name], value) for name, value in features))
        patterns[key][0] += weight
        patterns[key][1] += weight * label
    return {key: (total, positive) for key, (total, positive) in patterns.items()}


def _tile_rows(
    decisions: Sequence[WaitLabelledDecision], weights: Sequence[float], features_of
):
    for decision, weight in zip(decisions, weights):
        table = features_of(decision.decision.policy_input)
        for tile, features in zip(TILE_TYPES, table):
            yield features, tile in decision.wait_tile_types, weight


def fit_prevalence(decisions: Sequence[WaitLabelledDecision]) -> PrevalenceWaitModel:
    """牌種ごとの待ち率（重み付き、Jeffreys平滑化）。"""
    weights = row_weights(decisions)
    total = 0.0
    positive = [0.0] * 34
    for decision, weight in zip(decisions, weights):
        riichi_view(decision.decision.policy_input)
        total += weight
        for tile in decision.wait_tile_types:
            positive[tile_type_index(tile)] += weight
    if not total:
        raise ValueError("no decision to calibrate")
    return PrevalenceWaitModel(
        tuple((p + JEFFREYS_PRIOR) / (total + 2 * JEFFREYS_PRIOR) for p in positive)
    )


def _score_features(policy_input: PolicyInput):
    view = riichi_view(policy_input)
    player = policy_input.players[view.riichi_seat]
    return tuple(
        ((BIAS, 1.0), (SCORE, float(classical_wait_score(view, player, tile))))
        for tile in TILE_TYPES
    )


def fit_classical(decisions: Sequence[WaitLabelledDecision]) -> ClassicalScoreWaitModel:
    """古典的危険度scoreのPlatt校正（切片は正則化しない。L2は固定の小さな値）。"""
    rows = list(_tile_rows(decisions, row_weights(decisions), _score_features))
    names = (BIAS, SCORE)
    solved = fit_logistic(_patterns(rows, names), len(names), 1e-6)
    return ClassicalScoreWaitModel(intercept=solved[0], slope=solved[1])


def feature_names(rows) -> tuple[str, ...]:
    return tuple(sorted({name for features, _, _ in rows for name, _ in features}))


def build_estimator_rows(
    decisions: Sequence[WaitLabelledDecision],
) -> list[tuple[tuple[tuple[str, float], ...], bool, float]]:
    return list(_tile_rows(decisions, row_weights(decisions), wait_feature_table))


def fit_estimator_from_rows(rows, l2: float) -> LogisticWaitModel:
    names = feature_names(rows)
    if BIAS not in names:
        raise ValueError("no training rows")
    # BIASを先頭にして切片として扱う（正則化しない）
    ordered = (BIAS, *[name for name in names if name != BIAS])
    solved = fit_logistic(_patterns(rows, ordered), len(ordered), l2)
    return LogisticWaitModel(
        weights=tuple((name, weight) for name, weight in zip(ordered, solved))
    )


def fit_estimator(
    decisions: Sequence[WaitLabelledDecision], l2: float
) -> LogisticWaitModel:
    return fit_estimator_from_rows(build_estimator_rows(decisions), l2)


# ------------------------------------------------------------- evaluation


def _tile_class(tile: TileType, yakuhai: frozenset[TileType]) -> str:
    if tile.category is TileCategory.HONOR:
        return "honor_yakuhai" if tile in yakuhai else "honor_guest"
    return {1: "n19", 9: "n19", 2: "n28", 8: "n28", 3: "n37", 7: "n37"}.get(
        tile.rank, "n456"
    )


def _turn(count: int) -> str:
    return "early(<=6)" if count <= 6 else "middle(7-11)" if count <= 11 else "late"


def _auc(probabilities: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [p for p, label in zip(probabilities, labels) if label]
    negatives = [p for p, label in zip(probabilities, labels) if not label]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


class _Weighted:
    def __init__(self) -> None:
        self.weight = 0.0
        self.positives = 0.0
        self.probability = 0.0
        self.log_loss = 0.0
        self.brier = 0.0

    def add(self, weight: float, probability: float, label: bool) -> None:
        self.weight += weight
        self.positives += weight * label
        self.probability += weight * probability
        self.log_loss += weight * -log(probability if label else 1.0 - probability)
        self.brier += weight * (probability - label) ** 2

    def summary(self) -> dict[str, float]:
        if not self.weight:
            return {"weight": 0.0}
        return {
            "weight": self.weight,
            "positive_rate": self.positives / self.weight,
            "mean_probability": self.probability / self.weight,
            "log_loss": self.log_loss / self.weight,
            "brier": self.brier / self.weight,
        }


def evaluate(
    model: Model, decisions: Sequence[WaitLabelledDecision]
) -> dict[str, object]:
    """リーチ平均の指標・半荘別の合計・副指標を返す。

    主指標は、判断の34牌種のlossの平均 → 同一リーチ内の平均 → リーチ間の平均。
    ``per_game``は、半荘ごとの（リーチ数、リーチごとの平均lossの和）で、bootstrapが使う。
    """
    weights = row_weights(decisions)
    episodes: dict[RiichiEpisodeKey, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    rows_per_episode: dict[RiichiEpisodeKey, int] = defaultdict(int)
    groups: dict[str, dict[str, _Weighted]] = {
        "tile_class": defaultdict(_Weighted),
        "turn": defaultdict(_Weighted),
    }
    bins = [_Weighted() for _ in range(len(_RELIABILITY_EDGES) - 1)]
    auc_by_episode: dict[RiichiEpisodeKey, list[float]] = defaultdict(list)
    excluded_auc = 0
    for decision, weight in zip(decisions, weights):
        policy_input = decision.decision.policy_input
        view = riichi_view(policy_input)
        predictions = model.predict(policy_input)
        if set(predictions) != set(TILE_TYPES):
            raise ValueError("the model must predict all 34 tile types")
        probabilities = [predictions[tile] for tile in TILE_TYPES]
        labels = [tile in decision.wait_tile_types for tile in TILE_TYPES]
        episode = decision.episode
        stats = episodes[episode]
        rows_per_episode[episode] += 1
        stats["expected_kinds"] += weight * sum(probabilities)
        stats["actual_kinds"] += weight * sum(labels)
        row_loss = row_brier = 0.0
        for tile, p, label in zip(TILE_TYPES, probabilities, labels):
            if not CLIP_EPSILON <= p <= 1.0 - CLIP_EPSILON:
                raise ValueError("probability outside the clip range")
            loss = -log(p if label else 1.0 - p)
            row_loss += loss / 34
            row_brier += (p - label) ** 2 / 34
            groups["tile_class"][_tile_class(tile, view.yakuhai)].add(weight, p, label)
            groups["turn"][_turn(view.riichi_discard_count)].add(weight, p, label)
            edge = next(i for i in range(len(bins)) if p <= _RELIABILITY_EDGES[i + 1])
            bins[edge].add(weight, p, label)
        stats["log_loss"] += weight * row_loss
        stats["brier"] += weight * row_brier
        auc = _auc(probabilities, labels)
        if auc is None:
            excluded_auc += 1
        else:
            auc_by_episode[episode].append(auc)

    per_game: dict[str, dict[str, float]] = defaultdict(
        lambda: {"episodes": 0, "log_loss_sum": 0.0, "brier_sum": 0.0}
    )
    for episode, stats in episodes.items():
        game = per_game[str(episode.seed)]
        game["episodes"] += 1
        game["log_loss_sum"] += stats["log_loss"]
        game["brier_sum"] += stats["brier"]
    count = len(episodes)
    if not count:
        raise ValueError("no riichi episode to evaluate")
    episode_aucs = [sum(v) / len(v) for v in auc_by_episode.values()]
    return {
        "riichi_episodes": count,
        "decision_rows": len(decisions),
        "log_loss": sum(g["log_loss_sum"] for g in per_game.values()) / count,
        "brier": sum(g["brier_sum"] for g in per_game.values()) / count,
        "expected_wait_kinds": sum(s["expected_kinds"] for s in episodes.values())
        / count,
        "actual_wait_kinds": sum(s["actual_kinds"] for s in episodes.values()) / count,
        "auc": {
            "mean_over_episodes": (
                sum(episode_aucs) / len(episode_aucs) if episode_aucs else None
            ),
            "episodes_with_auc": len(episode_aucs),
            "decision_rows_without_both_classes": excluded_auc,
        },
        "per_game": dict(sorted(per_game.items())),
        "by_tile_class": {
            k: v.summary() for k, v in sorted(groups["tile_class"].items())
        },
        "by_turn": {k: v.summary() for k, v in sorted(groups["turn"].items())},
        "reliability": [
            {"low": _RELIABILITY_EDGES[i], "high": _RELIABILITY_EDGES[i + 1]}
            | bins[i].summary()
            for i in range(len(bins))
        ],
    }


def bootstrap_difference(
    candidate: dict[str, object], reference: dict[str, object], metric: str
) -> dict[str, float]:
    """半荘を単位に復元抽出し、(candidate - reference)のリーチ平均の差の分布を返す。

    再標本化した半荘の集合で、リーチごとの平均lossの和をリーチ数で割り直す。
    """
    a, b = candidate["per_game"], reference["per_game"]
    if set(a) != set(b) or any(a[g]["episodes"] != b[g]["episodes"] for g in a):
        raise ValueError("the two evaluations cover different games or episodes")
    games = sorted(a)
    key = f"{metric}_sum"

    def difference(sample: list[str]) -> float:
        count = sum(a[g]["episodes"] for g in sample)
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


def compare(evaluations: dict[str, dict[str, object]]) -> dict[str, object]:
    """推定器とベースライン1・2のΔ（推定器 - ベースライン）、と合格判定。"""
    comparison: dict[str, object] = {}
    passed = True
    for baseline in ("baseline1_prevalence", "baseline2_classical_platt"):
        delta = {
            metric: bootstrap_difference(
                evaluations["estimator_logistic"], evaluations[baseline], metric
            )
            for metric in ("log_loss", "brier")
        }
        comparison[f"estimator_minus_{baseline}"] = delta
        passed = passed and delta["log_loss"]["high_97.5"] < 0.0
    comparison["pass_rule"] = (
        "both estimator-minus-baseline log loss deltas have a 95% interval upper end below 0"
    )
    comparison["passed"] = passed
    return comparison


# ------------------------------------------------------------------ driver


def _split(
    manifest: RiichiDealInManifest,
    decisions: Sequence[WaitLabelledDecision],
    name: str,
) -> tuple[WaitLabelledDecision, ...]:
    seeds = frozenset(manifest.splits[name])
    return tuple(d for d in decisions if d.decision.key.seed in seeds)


def _source_identity(directory: Path) -> dict[str, object]:
    return file_digest(directory / MANIFEST_FILENAME)


def _write_new(path: Path, document: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(canonical_json_text(document), encoding="utf-8")


def _models_to_value(prevalence, classical, estimator) -> dict[str, object]:
    return {
        "baseline1_prevalence": {"probabilities": list(prevalence.probabilities)},
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
        PrevalenceWaitModel(tuple(value["baseline1_prevalence"]["probabilities"])),
        ClassicalScoreWaitModel(**value["baseline2_classical_platt"]),
        LogisticWaitModel(
            weights=tuple(value["estimator_logistic"]["weights"].items()),
            feature_set=value["estimator_logistic"]["feature_set"],
        ),
    )


def select(source: Path, output: Path) -> dict[str, object]:
    manifest, decisions = read_wait_labelled_source(source)
    train = _split(manifest, decisions, "train")
    valid = _split(manifest, decisions, "valid")
    prevalence = fit_prevalence(train)
    classical = fit_classical(train)
    rows = build_estimator_rows(train)
    grid = []
    for l2 in L2_GRID:
        model = fit_estimator_from_rows(rows, l2)
        grid.append((evaluate(model, valid)["log_loss"], l2, model))
    _, chosen_l2, estimator = min(grid, key=lambda item: (item[0], item[1]))
    models = (prevalence, classical, estimator)
    evaluations = {name: evaluate(m, valid) for name, m in zip(MODEL_NAMES, models)}
    document = {
        "schema": SELECTION_SCHEMA,
        "source_manifest": _source_identity(source),
        "feature_set": FEATURE_SET,
        "clip_epsilon": CLIP_EPSILON,
        "fitted_on": {
            "baselines": "train",
            "estimator": "train",
            "l2_selected_on": "valid",
        },
        "l2_grid": [{"l2": l2, "valid_log_loss": loss} for loss, l2, _ in grid],
        "chosen_l2": chosen_l2,
        "models": _models_to_value(*models),
        "used_seeds": {name: list(seeds) for name, seeds in manifest.splits.items()},
        "split_sizes": {
            name: len(_split(manifest, decisions, name))
            for name in ("train", "valid", "test")
        },
        "valid": evaluations,
        "valid_comparison": compare(evaluations),
    }
    _write_new(output, document)
    return document


def test(source: Path, selection: Path, output: Path) -> dict[str, object]:
    chosen = json.loads(selection.read_text(encoding="utf-8"))
    if chosen.get("schema") != SELECTION_SCHEMA:
        raise ValueError("not a selection document")
    if chosen["source_manifest"] == _source_identity(source):
        raise ValueError("the test source must differ from the selection source")
    manifest, decisions = read_wait_labelled_source(source)
    used = {seed for seeds in chosen["used_seeds"].values() for seed in seeds}
    test_seeds = set(manifest.splits["test"])
    if not test_seeds:
        raise ValueError("the test source has no test seed")
    if test_seeds & used:
        raise ValueError("the test seeds overlap the seeds used for the selection")
    rows = _split(manifest, decisions, "test")
    evaluations = {
        name: evaluate(model, rows)
        for name, model in zip(MODEL_NAMES, _models_from_value(chosen["models"]))
    }
    document = {
        "schema": RESULT_SCHEMA,
        "source_manifest": _source_identity(source),
        "selection": file_digest(selection),
        "test_seeds": sorted(test_seeds),
        "test": evaluations,
        "test_comparison": compare(evaluations),
    }
    _write_new(output, document)
    return document


def _summary(evaluations: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        name: {
            key: evaluations[name][key]
            for key in (
                "riichi_episodes",
                "log_loss",
                "brier",
                "expected_wait_kinds",
                "actual_wait_kinds",
            )
        }
        | {"auc": evaluations[name]["auc"]["mean_over_episodes"]}
        for name in MODEL_NAMES
    }


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
    arguments = parser.parse_args(argv)
    if arguments.command == "select":
        document = select(arguments.source, arguments.output)
        summary = {
            "split": "valid",
            "chosen_l2": document["chosen_l2"],
            "models": _summary(document["valid"]),
            "comparison": document["valid_comparison"],
        }
    else:
        document = test(arguments.source, arguments.selection, arguments.output)
        summary = {
            "split": "test",
            "models": _summary(document["test"]),
            "comparison": document["test_comparison"],
        }
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
