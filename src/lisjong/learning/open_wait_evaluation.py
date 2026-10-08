"""副露者の構造的待ち確率の推定器と出現率baselineの学習・比較（lisbun/lisjong#259 範囲1）。

学習専用の経路である。#256 v1のsource（``hand_belief_source``、chunkは複数可）を読み、
#257と同じ行・エピソード・集約で、副露者の行だけを比べる。条件は#259の事前登録に従う。

```text
python -m lisjong.learning.open_wait_evaluation select \\
    --train A..B --valid A..B --dev-eval A..B --output SELECTION.json SOURCE [SOURCE ...]
python -m lisjong.learning.open_wait_evaluation test \\
    --test A..B --selection SELECTION.json --selection-sha256 HEX \\
    --output RESULT.json SOURCE [SOURCE ...]
```

1. ``select``: trainで出現率baselineと推定器を当てはめ、推定器の2段のL2強度の組を
   validのlossで選ぶ。dev-eval（sourceの``test``分割）は開発用の確認として同じ指標を出すが、
   改善の主張には使わない
2. ``test``: 選択を固定したまま、新しいseedのsource（``test``分割だけ）で1回だけ評価する。
   selectionのseedと重なるseedは拒否する。出力は上書きしない。sourceのproducerは
   selectionを作ったproducerと全fieldで一致しなければならない。別の実行revisionで生成した
   sourceを使う場合は、事前登録したproducerを ``--test-producer FILE --test-producer-sha256 HEX``
   で渡す（実行revision以外のfieldが違う登録と、登録と違うsourceは拒否する。lisbun/lisjong#279）

## 行・重み・集約（#257と同じ）

- 行: 観測者の1判断 × 副露者1席（リーチしておらず暗槓以外の副露がある他家）
- エピソード: (半荘, 局instance, 他家席)。重み1をそのエピソードの行で等分する
- 1行のlossは34牌種で平均し、エピソード内で行平均し、エピソード間で平均する（episode-macro）
- 学習も同じ重みを使う。目的関数は34牌種の和（係数の定数倍だけの違い）に、切片以外のL2
- 確率は ``[1e-6, 1 - 1e-6]`` にclipしてから指標を計算する

## 比べるもの

- 出現率baseline: trainの副露者の行だけでの牌種ごとの待ち率（Jeffreys 0.5）
- 推定器（``open_wait_estimator``）: P(聴牌) × P(待ち | 聴牌)。聴牌段はtrainの副露者の全行、
  待ち段はtrainの副露者の聴牌行で当てはめる

## 判定（test）

``Δ = 推定器のlog loss - 出現率baselineのlog loss``（副露者の全行、episode-macro）。半荘単位の
paired bootstrap（2,000回、seed 259）の95%区間の上端が0未満なら ``pass``、そうでなければ
``not_confirmed``。testで待ちの正例を含む独立半荘が50未満、または正例エピソードが100未満なら
``held``（判定しない）。
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from math import log
from pathlib import Path

from lisjong.belief.fixed_point import SCALE
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.learning._canonical import canonical_json_text
from lisjong.learning.hand_belief_accuracy import (
    HOLD_MIN_EPISODES,
    HOLD_MIN_HANCHAN,
    HandBeliefAccuracyError,
    _Calibration,
    _Groups,
    bootstrap,
    check_population,
    check_test_producer,
    episode_macro,
    kyoku_instances,
    read_registered_producer,
    seed_range,
    turn_bucket,
)
from lisjong.learning.hand_belief_source import (
    HandBeliefLabelledDecision,
    read_labelled_source,
)
from lisjong.learning.open_wait_estimator import (
    BIAS,
    CLIP_EPSILON,
    FEATURE_SET,
    OpenWaitModel,
    _sigmoid,
    clip_probability,
    is_open_opponent,
    open_view,
    tenpai_features,
    wait_tile_feature_table,
)
from lisjong.learning.riichi_deal_in_evaluation import fit_logistic

SELECTION_SCHEMA = "lisjong-open-wait-selection-v1"
RESULT_SCHEMA = "lisjong-open-wait-test-result-v1"
L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
JEFFREYS = 0.5
BOOTSTRAP_SEED = 259
MODELS = ("estimator", "rate")


class OpenWaitEvaluationError(HandBeliefAccuracyError):
    """入力が事前登録の条件に合わない。"""


_E = OpenWaitEvaluationError


# --- 行 --------------------------------------------------------------------


Features = tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class OpenRow:
    """観測者の1判断 × 副露者1席。牌種はcanonical順。"""

    seed: int
    episode: tuple[int, int]  # (局instance, 他家席)
    turn: str
    unseen: tuple[int, ...]
    tenpai_features: Features
    tile_features: tuple[Features, ...]
    labels: tuple[bool, ...]

    @property
    def tenpai(self) -> bool:
        return any(self.labels)


def build_open_rows(
    labelled: Sequence[HandBeliefLabelledDecision],
) -> Iterator[OpenRow]:
    by_seed = defaultdict(list)
    for row in labelled:
        by_seed[row.decision.key.seed].append(row)
    intern: dict = {}
    for seed in sorted(by_seed):
        instances = kyoku_instances(by_seed[seed])
        for row in by_seed[seed]:
            pi = row.decision.policy_input
            remaining = None
            for opponent in row.opponents:
                if not is_open_opponent(pi, opponent.seat):
                    continue
                if remaining is None:
                    remaining = tuple(
                        derive_remaining_tile_inventory(pi).remaining_tile_counts
                    )
                view = open_view(pi, opponent.seat, remaining)
                raw = opponent.truth.wait_probability_raw
                if raw is None:
                    raise _E("the truth lacks a wait table")
                yield OpenRow(
                    seed=row.decision.key.seed,
                    episode=(instances[row.decision.key.sequence], opponent.seat),
                    turn=turn_bucket(view.discard_count),
                    unseen=remaining,
                    tenpai_features=intern.setdefault(
                        tenpai_features(view), tenpai_features(view)
                    ),
                    tile_features=tuple(
                        intern.setdefault(f, f) for f in wait_tile_feature_table(view)
                    ),
                    labels=tuple(value == SCALE for value in raw),
                )


def episode_weights(rows: Sequence[OpenRow]) -> list[float]:
    counts = defaultdict(int)
    for row in rows:
        counts[(row.seed, row.episode)] += 1
    return [1.0 / counts[(row.seed, row.episode)] for row in rows]


def read_split_rows(
    sources: Sequence[Path], split: str
) -> tuple[list[OpenRow], set[int]]:
    """各sourceのmanifestの``split``に属するseedの副露者の行。"""
    rows, seeds = [], set()
    for source in sources:
        manifest, labelled = read_labelled_source(source)
        wanted = set(manifest.splits[split])
        chosen = [r for r in labelled if r.decision.key.seed in wanted]
        if {r.decision.key.seed for r in chosen} != wanted:
            raise _E(f"{source}: a {split} seed has no decision")
        rows.extend(build_open_rows(chosen))
        seeds |= wanted
    return rows, seeds


# --- 当てはめ ---------------------------------------------------------------


def fit_rate(rows: Sequence[OpenRow]) -> tuple[float, ...]:
    total = 0.0
    positives = [0.0] * 34
    for row, weight in zip(rows, episode_weights(rows)):
        total += weight
        for index, label in enumerate(row.labels):
            if label:
                positives[index] += weight
    if not total:
        raise _E("no train row")
    return tuple((p + JEFFREYS) / (total + 2 * JEFFREYS) for p in positives)


def _names(feature_sets) -> list[str]:
    names = {name for features in feature_sets for name, _ in features}
    names.discard(BIAS)
    return [BIAS, *sorted(names)]


def _patterns(samples, names):
    """(特徴, ラベル, 重み) を同じ特徴ごとに (重みの和, 正例の重みの和) へ集約する。"""
    index = {name: position for position, name in enumerate(names)}
    keys: dict = {}
    patterns: dict = defaultdict(lambda: [0.0, 0.0])
    for features, label, weight in samples:
        key = keys.get(features)
        if key is None:
            key = keys[features] = tuple(
                sorted((index[name], value) for name, value in features)
            )
        cell = patterns[key]
        cell[0] += weight
        cell[1] += weight * label
    return {key: tuple(value) for key, value in patterns.items()}


def tenpai_samples(rows: Sequence[OpenRow]):
    for row, weight in zip(rows, episode_weights(rows)):
        yield row.tenpai_features, row.tenpai, weight


def wait_samples(rows: Sequence[OpenRow]):
    """聴牌行の34牌種。重みは副露者の全行で求めたエピソード重み。"""
    for row, weight in zip(rows, episode_weights(rows)):
        if row.tenpai:
            for features, label in zip(row.tile_features, row.labels):
                yield features, label, weight


def fit_stage(samples, l2_grid: Sequence[float]) -> dict[float, tuple]:
    samples = list(samples)
    names = _names(features for features, _, _ in samples)
    patterns = _patterns(samples, names)
    return {
        l2: tuple(zip(names, fit_logistic(patterns, len(names), l2))) for l2 in l2_grid
    }


# --- 指標 -------------------------------------------------------------------


def _log_loss(p: float, label: bool) -> float:
    return -log(p if label else 1.0 - p)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def within_row_auc(p: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [q for q, y in zip(p, labels) if y]
    negatives = [q for q, y in zip(p, labels) if not y]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if a > b else 0.5 if a == b else 0.0 for a in positives for b in negatives
    )
    return wins / (len(positives) * len(negatives))


def _linear(weights: dict[str, float], features: Features) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features)


def estimator_prediction(
    row: OpenRow, tenpai_weights: dict, wait_weights: dict
) -> tuple[float, tuple[float, ...]]:
    tenpai = _sigmoid(_linear(tenpai_weights, row.tenpai_features))
    return tenpai, tuple(
        clip_probability(tenpai * _sigmoid(_linear(wait_weights, features)))
        for features in row.tile_features
    )


def row_metrics(
    row: OpenRow, predictions: dict[str, tuple[float, ...]], tenpai_p: float
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    labels = row.labels
    for model, p in predictions.items():
        metrics[f"{model}.log_loss"] = _mean(map(_log_loss, p, labels))
        metrics[f"{model}.brier"] = _mean((q - y) ** 2 for q, y in zip(p, labels))
        metrics[f"{model}.predicted_kinds"] = sum(p)
        metrics[f"{model}.predicted_unseen_weighted"] = sum(
            u * q for u, q in zip(row.unseen, p)
        )
        auc = within_row_auc(p, labels)
        if auc is not None:
            metrics[f"{model}.within_row_auc"] = auc
    for metric in ("log_loss", "brier"):
        metrics[f"delta.{metric}"] = (
            metrics[f"estimator.{metric}"] - metrics[f"rate.{metric}"]
        )
    metrics["actual_kinds"] = sum(labels)
    metrics["actual_unseen_weighted"] = sum(u for u, y in zip(row.unseen, labels) if y)
    tenpai_p = min(max(tenpai_p, CLIP_EPSILON), 1.0 - CLIP_EPSILON)
    metrics["estimator.tenpai_log_loss"] = _log_loss(tenpai_p, row.tenpai)
    metrics["estimator.tenpai_probability"] = tenpai_p
    metrics["actual_tenpai"] = float(row.tenpai)
    return metrics


@dataclass
class Report:
    groups: _Groups = field(default_factory=_Groups)
    calibration: dict[str, _Calibration] = field(
        default_factory=lambda: defaultdict(_Calibration)
    )
    sets: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    rows: int = 0
    positive_rows: int = 0

    def add(self, rows: Sequence[OpenRow], model: OpenWaitModel, rate) -> None:
        tenpai_weights = dict(model.tenpai_weights)
        wait_weights = dict(model.wait_weights)
        rate = tuple(clip_probability(p) for p in rate)
        for row, weight in zip(rows, episode_weights(rows)):
            tenpai_p, estimator = estimator_prediction(
                row, tenpai_weights, wait_weights
            )
            values = row_metrics(row, {"estimator": estimator, "rate": rate}, tenpai_p)
            tenpai = "tenpai" if row.tenpai else "not_tenpai"
            for group in ("open", f"open.{tenpai}", f"open.turn.{row.turn}"):
                self.groups.add(row.seed, row.episode, group, values)
            self.calibration["estimator"].add(weight, estimator, row.labels)
            self.calibration["rate"].add(weight, rate, row.labels)
            episode = (row.seed, row.episode)
            self.rows += 1
            self.sets["episodes"].add(episode)
            self.sets["hanchan"].add(row.seed)
            if row.tenpai:
                self.positive_rows += 1
                self.sets["positive_episodes"].add(episode)
                self.sets["positive_hanchan"].add(row.seed)

    def counts(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "positive_rows": self.positive_rows,
            **{
                name: len(self.sets.get(name, ()))
                for name in (
                    "episodes",
                    "hanchan",
                    "positive_episodes",
                    "positive_hanchan",
                )
            },
        }

    def support(self) -> str:
        counts = self.counts()
        if not counts["positive_episodes"]:
            return "not_estimable"
        if (
            counts["positive_hanchan"] < HOLD_MIN_HANCHAN
            or counts["positive_episodes"] < HOLD_MIN_EPISODES
        ):
            return "held"
        return "reported"

    def document(self, seeds: Sequence[int]) -> dict[str, object]:
        per_group = self.groups.per_seed()
        seeds = sorted(seeds)
        targets = [
            (group, metric)
            for group in ("open", "open.tenpai", "open.not_tenpai")
            for metric in (
                "estimator.log_loss",
                "rate.log_loss",
                "delta.log_loss",
                "delta.brier",
            )
        ]
        point = {
            group: {
                metric: episode_macro(per_seed, seeds, metric)
                for metric in sorted({m for s in per_seed.values() for m in s})
            }
            for group, per_seed in sorted(per_group.items())
        }
        return {
            "counts": self.counts(),
            "support": self.support(),
            "episode_macro": point,
            "intervals": bootstrap(per_group, seeds, targets, seed=BOOTSTRAP_SEED),
            "calibration": {
                name: cal.summary() for name, cal in sorted(self.calibration.items())
            },
        }


def verdict(document: dict[str, object]) -> str:
    if document["support"] != "reported":
        return "held"
    high = document["intervals"]["open/delta.log_loss"]["high_97.5"]
    return "pass" if high < 0 else "not_confirmed"


def valid_loss(rows: Sequence[OpenRow], model: OpenWaitModel) -> float:
    """validのepisode-macro log loss（推定器）。"""
    groups = _Groups()
    tenpai_weights, wait_weights = dict(model.tenpai_weights), dict(model.wait_weights)
    for row in rows:
        _, p = estimator_prediction(row, tenpai_weights, wait_weights)
        groups.add(
            row.seed, row.episode, "open", {"m": _mean(map(_log_loss, p, row.labels))}
        )
    per_seed = groups.per_seed()["open"]
    return episode_macro(per_seed, sorted(per_seed), "m")


# --- select / test ------------------------------------------------------------


def _model_value(model: OpenWaitModel, rate) -> dict[str, object]:
    return {
        "estimator": {
            "feature_set": model.feature_set,
            "tenpai_weights": [list(pair) for pair in model.tenpai_weights],
            "wait_weights": [list(pair) for pair in model.wait_weights],
        },
        "rate": list(rate),
    }


def _models_from_value(value) -> tuple[OpenWaitModel, tuple[float, ...]]:
    estimator = value["estimator"]
    model = OpenWaitModel(
        tenpai_weights=tuple((n, float(w)) for n, w in estimator["tenpai_weights"]),
        wait_weights=tuple((n, float(w)) for n, w in estimator["wait_weights"]),
        feature_set=estimator["feature_set"],
    )
    rate = tuple(float(p) for p in value["rate"])
    if len(rate) != 34:
        raise _E("the rate baseline must have 34 values")
    return model, rate


def select(
    sources: Sequence[Path], expected: dict[str, Sequence[int]]
) -> dict[str, object]:
    identity = check_population(sources, expected)
    train, _ = read_split_rows(sources, "train")
    valid, _ = read_split_rows(sources, "valid")
    rate = fit_rate(train)
    tenpai_fits = fit_stage(tenpai_samples(train), L2_GRID)
    wait_fits = fit_stage(wait_samples(train), L2_GRID)
    grid = []
    best = None
    for l2_tenpai in L2_GRID:
        for l2_wait in L2_GRID:
            model = OpenWaitModel(tenpai_fits[l2_tenpai], wait_fits[l2_wait])
            loss = valid_loss(valid, model)
            grid.append({"l2_tenpai": l2_tenpai, "l2_wait": l2_wait, "loss": loss})
            if best is None or loss < best[0]:
                best = (loss, l2_tenpai, l2_wait, model)
    _, l2_tenpai, l2_wait, model = best
    valid_report = Report()
    valid_report.add(valid, model, rate)
    del train, valid
    dev, _ = read_split_rows(sources, "test")
    dev_report = Report()
    dev_report.add(dev, model, rate)
    return {
        "schema": SELECTION_SCHEMA,
        "feature_set": FEATURE_SET,
        "population": identity,
        "splits": {name: sorted(seeds) for name, seeds in expected.items()},
        "selected": {"l2_tenpai": l2_tenpai, "l2_wait": l2_wait},
        "l2_grid": list(L2_GRID),
        "valid_grid": grid,
        "models": _model_value(model, rate),
        "valid": valid_report.document(expected["valid"]),
        "dev_eval": {
            "note": "development check only; not a claim",
            **dev_report.document(expected["eval"]),
        },
        "bootstrap": {"resamples": 2000, "seed": BOOTSTRAP_SEED},
        "clip_epsilon": CLIP_EPSILON,
    }


def _check_selection(selection: dict[str, object]) -> None:
    if selection.get("schema") != SELECTION_SCHEMA:
        raise _E("unknown selection schema")
    if selection.get("feature_set") != FEATURE_SET:
        raise _E("the selection was made with another feature set")


def run_test(
    sources: Sequence[Path],
    test_seeds: Sequence[int],
    selection: dict[str, object],
    test_producer: dict[str, object] | None = None,
) -> dict[str, object]:
    _check_selection(selection)
    used = {seed for seeds in selection["splits"].values() for seed in seeds}
    if used & set(test_seeds):
        raise _E("a test seed was used by the selection")
    identity = check_population(sources, {"train": [], "valid": [], "eval": test_seeds})
    registration = check_test_producer(
        identity, selection["population"]["producer"], test_producer
    )
    model, rate = _models_from_value(selection["models"])
    rows, _ = read_split_rows(sources, "test")
    report = Report()
    report.add(rows, model, rate)
    document = report.document(test_seeds)
    return {
        "schema": RESULT_SCHEMA,
        "feature_set": FEATURE_SET,
        "population": identity,
        "test_seeds": sorted(test_seeds),
        "verdict": verdict(document),
        **document,
        "hold_rule": {
            "min_positive_hanchan": HOLD_MIN_HANCHAN,
            "min_positive_episodes": HOLD_MIN_EPISODES,
        },
        "bootstrap": {"resamples": 2000, "seed": BOOTSTRAP_SEED},
        "clip_epsilon": CLIP_EPSILON,
    } | ({} if registration is None else {"producer_registration": registration})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=__name__)
    commands = parser.add_subparsers(dest="command", required=True)
    chosen = commands.add_parser("select")
    chosen.add_argument("--train", type=seed_range, required=True)
    chosen.add_argument("--valid", type=seed_range, required=True)
    chosen.add_argument("--dev-eval", type=seed_range, required=True)
    tested = commands.add_parser("test")
    tested.add_argument("--test", type=seed_range, required=True)
    tested.add_argument("--selection", type=Path, required=True)
    tested.add_argument("--selection-sha256", required=True)
    tested.add_argument("--test-producer", type=Path)
    tested.add_argument("--test-producer-sha256")
    for command in (chosen, tested):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("sources", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        parser.error(f"refusing to overwrite {arguments.output}")
    if arguments.command == "select":
        document = select(
            arguments.sources,
            {
                "train": arguments.train,
                "valid": arguments.valid,
                "eval": arguments.dev_eval,
            },
        )
        summary = {
            "selected": document["selected"],
            "valid": document["valid"]["intervals"],
            "dev_eval": document["dev_eval"]["intervals"],
        }
    else:
        data = arguments.selection.read_bytes()
        if hashlib.sha256(data).hexdigest() != arguments.selection_sha256:
            parser.error("the selection does not match the registered SHA-256")
        if (arguments.test_producer is None) != (
            arguments.test_producer_sha256 is None
        ):
            parser.error("a test producer needs its registered SHA-256")
        test_producer = (
            None
            if arguments.test_producer is None
            else read_registered_producer(
                arguments.test_producer, arguments.test_producer_sha256
            )
        )
        document = run_test(
            arguments.sources, arguments.test, json.loads(data), test_producer
        )
        document["selection_sha256"] = arguments.selection_sha256
        if test_producer is not None:
            document["producer_registration"]["sha256"] = arguments.test_producer_sha256
        summary = {"verdict": document["verdict"], "intervals": document["intervals"]}
    arguments.output.write_text(canonical_json_text(document), encoding="utf-8")
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
