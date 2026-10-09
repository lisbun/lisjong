"""形別待ちテーブル（Level 2）の推定器と出現率baselineの学習・比較（lisbun/lisjong#260）。

学習専用の経路である。#256 v1のsource（``hand_belief_source``、chunkは複数可）を読み、
#257と同じ行・エピソード・集約で、populationごと・channelごとに比べる。条件は
``docs/wait-shape-belief.md``に従う。

```text
python -m lisjong.learning.wait_shape_evaluation select \\
    --train A..B --valid A..B --dev-eval A..B \\
    --riichi-wait-selection FILE --riichi-wait-sha256 HEX \\
    --open-wait-selection FILE --open-wait-sha256 HEX \\
    --code-revision COMMIT --output SELECTION.json SOURCE [SOURCE ...]
python -m lisjong.learning.wait_shape_evaluation test \\
    --test A..B --selection SELECTION.json --selection-sha256 HEX \\
    （Level 1の4引数と --code-revision は select と同じ） \\
    --output RESULT.json SOURCE [SOURCE ...]
```

1. ``select``: trainで出現率baselineと形別の係数を当てはめ、population・channelごとのL2強度を
   validのlossで選ぶ。dev-eval（sourceの``test``分割）は開発用の確認として同じ指標を出すが、
   改善の主張には使わない
2. ``test``: 選択を固定したまま、新しいseedのsource（``test``分割だけ）で1回だけ評価する。
   selectionのseedと重なるseed、selectionと違うLevel 1の待ち推定器は拒否する。出力は
   上書きしない。producerの照合は``open_wait_evaluation``と同じ（lisbun/lisjong#279）

## population・行・重み

- ``riichi``: #245のS1対象の判断（打牌を選んだ判断で、合法打牌の牌種が2以上、観測者は非リーチ、
  他家のリーチ者が1人）のリーチ者1席。待ち確率は凍結済みの#245の推定器
- ``open``: リーチしておらず暗槓以外の副露がある他家。待ち確率は凍結済みの#259範囲1の推定器
- それ以外の行（門前非リーチ、S1対象外のリーチ者）は未提供として件数だけを数える
- 1行は観測者の1判断 × 他家1席。エピソードは(半荘, 局instance, 他家席)で、重み1をその
  populationに属する行で等分する。2つのpopulationは混ぜない

## 当てはめ（trainだけ）

- 出現率baseline: channel・牌種ごとに ``(重み付き正例数 + 0.5) / (エピソード数 + 1)``。
  channelが構造上占め得ないslotは0
- 推定器: channelごとのlogistic回帰。行のlossはchannelが占め得るslot（静的な集合）で平均し、
  枚数の必要条件を満たさないslotは定義上0なので当てはめに入れない（残ったslotで割り直さない）。
  目的関数は ``Σ_行 エピソード重み × 行のloss + L2 × Σ係数² / 2`` で、L2は切片にも掛ける。
  副露者の国士は構造上のゼロモデルで、当てはめもL2の選択もしない
- L2は ``0.01, 0.1, 1, 10, 100`` から、上限処理・mask・固定小数点化の後のvalidのlog loss
  （静的slot、episode-macro）が最小のものをpopulation・channelごとに選ぶ。同値なら先のもの

## 指標と判定

確率は最終的な``HandBelief``のrawを``SCALE``で割った値を使い、``[1e-6, 1 - 1e-6]``にclipする。
主指標はchannelが占め得るslotだけのlog loss（``log_loss.static``）で、全34 slotの値も併記する。
``Δ = 推定器 − 出現率baseline``を、半荘単位のpaired bootstrap（2,000回、seed 260）で評価する。

判定はpopulation × channelごとに行う。対象行がない、正例がない、正例を含む独立半荘が50未満、
または正例エピソードが100未満なら ``excluded``（判定対象外。理由を記録する）。そうでなければ、
Δの95%区間の上端が0未満で ``improved``、それ以外は ``not_confirmed``。7 channel全体・
両population全体の採否は出さない。
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lisjong.belief.fixed_point import SCALE, probability_to_raw
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.learning import open_wait_evaluation
from lisjong.learning._canonical import canonical_json_text
from lisjong.learning.hand_belief_accuracy import (
    HOLD_MIN_EPISODES,
    HOLD_MIN_HANCHAN,
    HandBeliefAccuracyError,
    Riichi245Estimator,
    _Calibration,
    _Groups,
    bootstrap,
    check_population,
    check_test_producer,
    clip,
    episode_macro,
    kyoku_instances,
    load_245_estimator,
    log_loss,
    read_registered_producer,
    seed_range,
)
from lisjong.learning.hand_belief_source import (
    MANIFEST_SCHEMA,
    SPLITS,
    HandBeliefLabelledDecision,
    read_labelled_source,
)
from lisjong.learning.open_wait_estimator import (
    OpenWaitModel,
    is_open_opponent,
    open_view,
)
from lisjong.learning.riichi_deal_in_estimator import riichi_view
from lisjong.learning.riichi_deal_in_evaluation import fit_logistic
from lisjong.learning.riichi_wait_estimator import BIAS, CLIP_EPSILON
from lisjong.learning.wait_shape_estimator import (
    CHANNEL_NAMES,
    CHANNELS,
    CONSISTENCY,
    FEATURE_SET,
    KOKUSHI,
    POPULATIONS,
    Features,
    WaitShapeModel,
    cap_by_wait,
    open_shape_features,
    riichi_shape_features,
    supported_slots,
)

SELECTION_SCHEMA = "lisjong-wait-shape-selection-v1"
RESULT_SCHEMA = "lisjong-wait-shape-test-result-v1"
L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
JEFFREYS = 0.5
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 260
COVERAGE = (*POPULATIONS, "unprovided")
TARGETS = (
    "estimator.log_loss.static",
    "rate.log_loss.static",
    "delta.log_loss.static",
    "delta.log_loss.all",
    "delta.brier.static",
    "delta_capped_rate.log_loss.static",
)
_STATIC = tuple(tuple(sorted(slots)) for _, _, slots in CHANNELS)


class WaitShapeEvaluationError(HandBeliefAccuracyError):
    """入力が事前登録の条件に合わない。"""


_E = WaitShapeEvaluationError


# --- 固定したLevel 1の待ち推定器 ----------------------------------------------


@dataclass(frozen=True, slots=True)
class Level1:
    """populationごとの凍結済みの待ち推定器と、そのselectionのSHA-256。"""

    riichi: Riichi245Estimator
    open: OpenWaitModel
    open_sha256: str

    def identity(self) -> dict[str, object]:
        return {
            "riichi": {
                "selection_sha256": self.riichi.selection_sha256,
                "feature_set": self.riichi.model.feature_set,
            },
            "open": {
                "selection_sha256": self.open_sha256,
                "feature_set": self.open.feature_set,
            },
        }


def load_level1(
    riichi_selection: Path, riichi_sha256: str, open_selection: Path, open_sha256: str
) -> Level1:
    """#245と#259範囲1のselectionを読む。登録したSHA-256と一致しなければ拒否する。"""
    data = open_selection.read_bytes()
    if hashlib.sha256(data).hexdigest() != open_sha256:
        raise _E("the #259 selection does not match the registered SHA-256")
    chosen = json.loads(data)
    open_wait_evaluation._check_selection(chosen)
    model, _ = open_wait_evaluation._models_from_value(chosen["models"])
    return Level1(
        riichi=load_245_estimator(riichi_selection, riichi_sha256),
        open=model,
        open_sha256=open_sha256,
    )


# --- 行 --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShapeRow:
    """観測者の1判断 × 対象の他家1席。牌種はcanonical順、channelは``CHANNELS``の順。"""

    seed: int
    episode: tuple[int, int]  # (局instance, 他家席)
    tile_features: tuple[Features, ...]
    shared_features: Features
    supported: tuple[frozenset[int], ...]
    wait_raw: tuple[int, ...]
    labels: tuple[tuple[bool, ...], ...]


@dataclass
class SplitRows:
    rows: dict[str, list[ShapeRow]]
    coverage: dict[str, int]
    seeds: set[int]

    @classmethod
    def empty(cls) -> "SplitRows":
        return cls(
            {name: [] for name in POPULATIONS}, dict.fromkeys(COVERAGE, 0), set()
        )


def build_rows(
    labelled: Sequence[HandBeliefLabelledDecision],
    level1: Level1,
    target: SplitRows,
    intern: dict | None = None,
) -> None:
    """ラベル付きの判断から、2つのpopulationの行と未提供の件数を``target``へ加える。"""
    intern = {} if intern is None else intern

    def shared(value):
        return intern.setdefault(value, value)

    by_seed = defaultdict(list)
    for row in labelled:
        by_seed[row.decision.key.seed].append(row)
    for seed in sorted(by_seed):
        instances = kyoku_instances(by_seed[seed])
        for row in by_seed[seed]:
            pi = row.decision.policy_input
            riichi_seat = level1.riichi.scope_seat(row.decision)
            remaining = None
            for opponent in row.opponents:
                seat = opponent.seat
                if seat == riichi_seat:
                    population = "riichi"
                    view = riichi_view(pi)
                    features = riichi_shape_features(view)
                    wait = level1.riichi.predict(pi)
                elif is_open_opponent(pi, seat):
                    population = "open"
                    if remaining is None:
                        remaining = tuple(
                            derive_remaining_tile_inventory(pi).remaining_tile_counts
                        )
                    view = open_view(pi, seat, remaining)
                    features = open_shape_features(view)
                    wait = level1.open.predict_view(view)
                else:
                    target.coverage["unprovided"] += 1
                    continue
                truth = opponent.truth
                if truth.tanki_wait_probability_raw is None:
                    raise _E("the truth lacks the wait mechanism tables")
                target.coverage[population] += 1
                target.rows[population].append(
                    ShapeRow(
                        seed=seed,
                        episode=(instances[row.decision.key.sequence], seat),
                        tile_features=tuple(shared(f) for f in features[0]),
                        shared_features=shared(features[1]),
                        supported=shared(
                            supported_slots(
                                view.remaining_counts, bool(pi.players[seat].melds)
                            )
                        ),
                        wait_raw=tuple(probability_to_raw(p) for p in wait),
                        labels=shared(
                            tuple(
                                tuple(value == SCALE for value in getattr(truth, name))
                                for _, name, _ in CHANNELS
                            )
                        ),
                    )
                )


def read_rows(sources: Sequence[Path], level1: Level1) -> dict[str, SplitRows]:
    """各sourceを1回ずつ読み、manifestのsplitごとに行を作る。"""
    result = {split: SplitRows.empty() for split in SPLITS}
    intern: dict = {}
    for source in sources:
        manifest, labelled = read_labelled_source(source)
        for split in SPLITS:
            wanted = set(manifest.splits[split])
            chosen = [r for r in labelled if r.decision.key.seed in wanted]
            if {r.decision.key.seed for r in chosen} != wanted:
                raise _E(f"{source}: a {split} seed has no decision")
            build_rows(chosen, level1, result[split], intern)
            result[split].seeds |= wanted
    return result


def episode_weights(rows: Sequence[ShapeRow]) -> list[float]:
    counts = defaultdict(int)
    for row in rows:
        counts[(row.seed, row.episode)] += 1
    return [1.0 / counts[(row.seed, row.episode)] for row in rows]


# --- 当てはめ ---------------------------------------------------------------


def fit_rate(rows: Sequence[ShapeRow]) -> tuple[tuple[float, ...], ...]:
    """channel・牌種ごとの出現率（Jeffreys 0.5）。channelが占め得ないslotは0。"""
    total = 0.0
    positives = [[0.0] * 34 for _ in CHANNELS]
    for row, weight in zip(rows, episode_weights(rows)):
        total += weight
        for table, labels in zip(positives, row.labels):
            for index, label in enumerate(labels):
                if label:
                    table[index] += weight
    if not total:
        raise _E("no train row")
    return tuple(
        tuple(
            (value + JEFFREYS) / (total + 2 * JEFFREYS) if index in static else 0.0
            for index, value in enumerate(table)
        )
        for table, (_, _, static) in zip(positives, CHANNELS)
    )


def channel_patterns(rows: Sequence[ShapeRow], channel: int):
    """``channel``の当てはめに使うslotを、同じ特徴ごとに (重みの和, 正例の重みの和) へ集約する。

    1行の重みはエピソード重み ÷ 静的slot数。枚数の必要条件を満たさないslotは入れない。
    """
    static = _STATIC[channel]
    cells: dict = {}
    for row, weight in zip(rows, episode_weights(rows)):
        share = weight / len(static)
        supported = row.supported[channel]
        labels = row.labels[channel]
        for index in static:
            if index in supported:
                key = (row.tile_features[index], row.shared_features)
                cell = cells.get(key)
                if cell is None:
                    cell = cells[key] = [0.0, 0.0]
                cell[0] += share
                cell[1] += share * labels[index]
    names = {name for key in cells for part in key for name, _ in part}
    names.discard(BIAS)
    names = [BIAS, *sorted(names)]
    position = {name: index for index, name in enumerate(names)}
    patterns: dict = defaultdict(lambda: [0.0, 0.0])
    for (tile, shared), (weight, positive) in cells.items():
        cell = patterns[
            tuple(sorted((position[name], value) for name, value in (*tile, *shared)))
        ]
        cell[0] += weight
        cell[1] += positive
    return names, {key: tuple(value) for key, value in patterns.items()}


def fit_channel(
    rows: Sequence[ShapeRow], channel: int, l2_grid: Sequence[float]
) -> dict[float, Features]:
    """L2ごとの係数。L2は切片を含む全係数に掛ける。収束しなければ例外のまま失敗する。"""
    names, patterns = channel_patterns(rows, channel)
    return {
        l2: tuple(
            zip(names, fit_logistic(patterns, len(names), l2, unpenalized=frozenset()))
        )
        for l2 in l2_grid
    }


def is_zero_model(population: str, channel: str) -> bool:
    """係数を持たない構造上のゼロモデルか（副露者の国士）。"""
    return population == "open" and channel == KOKUSHI


# --- 指標 -------------------------------------------------------------------


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def predict_raw(row: ShapeRow, model: WaitShapeModel):
    """上限処理の前の確率と、最終的な``HandBelief``と同じraw。"""
    q = model.probabilities(row.tile_features, row.shared_features, row.supported)
    return q, cap_by_wait(q, row.wait_raw)


def capped_rate_raw(row: ShapeRow, rates) -> tuple[tuple[int, ...], ...]:
    """副比較: 出現率に、推定器と同じmaskと上限処理を掛けたもの。"""
    return cap_by_wait(
        [
            [p if index in slots else 0.0 for index, p in enumerate(rate)]
            for rate, slots in zip(rates, row.supported)
        ],
        row.wait_raw,
    )


def channel_metrics(
    channel: int, labels: Sequence[bool], predictions: dict[str, Sequence[float]]
) -> dict[str, float]:
    static = _STATIC[channel]
    metrics = {"positive_rate.static": _mean(labels[i] for i in static)}
    for model, values in predictions.items():
        p = [clip(value) for value in values]
        losses = [log_loss(q, y) for q, y in zip(p, labels)]
        metrics[f"{model}.log_loss.all"] = _mean(losses)
        metrics[f"{model}.log_loss.static"] = _mean(losses[i] for i in static)
        metrics[f"{model}.brier.static"] = _mean(
            (p[i] - labels[i]) ** 2 for i in static
        )
    for other in ("rate", "capped_rate"):
        prefix = "delta" if other == "rate" else f"delta_{other}"
        for metric in ("log_loss.all", "log_loss.static", "brier.static"):
            metrics[f"{prefix}.{metric}"] = (
                metrics[f"estimator.{metric}"] - metrics[f"{other}.{metric}"]
            )
    return metrics


_SETS = ("episodes", "hanchan")
_CHANNEL_COUNTS = (
    "positive_rows",
    "positive_slots",
    "static_slots",
    "estimated_slots",
    "over_wait_slots_before_cap",
    "over_wait_slots_after_cap",
)


class Report:
    """1つのpopulationの、channelごとの指標・件数・上限処理の診断。"""

    def __init__(self) -> None:
        self.groups = _Groups()
        self.calibration = [_Calibration() for _ in CHANNELS]
        self.rows = 0
        self.sets: dict[str, set] = defaultdict(set)
        self.counts = [dict.fromkeys(_CHANNEL_COUNTS, 0) for _ in CHANNELS]
        self.excess = [0.0 for _ in CHANNELS]

    def add(self, rows: Sequence[ShapeRow], model: WaitShapeModel, rates) -> None:
        for row, weight in zip(rows, episode_weights(rows)):
            episode = (row.seed, row.episode)
            self.rows += 1
            self.sets["episodes"].add(episode)
            self.sets["hanchan"].add(row.seed)
            q, raw = predict_raw(row, model)
            capped = capped_rate_raw(row, rates)
            for channel, (name, _, _) in enumerate(CHANNELS):
                labels = row.labels[channel]
                static = _STATIC[channel]
                estimator = [value / SCALE for value in raw[channel]]
                self.groups.add(
                    row.seed,
                    row.episode,
                    name,
                    channel_metrics(
                        channel,
                        labels,
                        {
                            "estimator": estimator,
                            "rate": rates[channel],
                            "capped_rate": [v / SCALE for v in capped[channel]],
                        },
                    ),
                )
                self.calibration[channel].add(
                    weight,
                    [clip(estimator[i]) for i in static],
                    [labels[i] for i in static],
                )
                counts = self.counts[channel]
                counts["static_slots"] += len(static)
                counts["estimated_slots"] += len(row.supported[channel])
                positives = sum(labels)
                counts["positive_slots"] += positives
                if positives:
                    counts["positive_rows"] += 1
                    self.sets[f"{name}.positive_episodes"].add(episode)
                    self.sets[f"{name}.positive_hanchan"].add(row.seed)
                for index in row.supported[channel]:
                    over = probability_to_raw(q[channel][index]) - row.wait_raw[index]
                    if over > 0:
                        counts["over_wait_slots_before_cap"] += 1
                        self.excess[channel] += over / SCALE
                counts["over_wait_slots_after_cap"] += sum(
                    value > wait for value, wait in zip(raw[channel], row.wait_raw)
                )

    def channel_counts(self, channel: int) -> dict[str, object]:
        name = CHANNEL_NAMES[channel]
        counts = dict(self.counts[channel])
        over = counts["over_wait_slots_before_cap"]
        return counts | {
            "positive_episodes": len(self.sets[f"{name}.positive_episodes"]),
            "positive_hanchan": len(self.sets[f"{name}.positive_hanchan"]),
            "mean_excess_over_wait_before_cap": (
                self.excess[channel] / over if over else None
            ),
        }

    def document(self, seeds: Sequence[int]) -> dict[str, object]:
        per_group = self.groups.per_seed()
        seeds = sorted(seeds)
        intervals = bootstrap(
            per_group,
            seeds,
            [(name, metric) for name in CHANNEL_NAMES for metric in TARGETS],
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        )
        channels = {}
        for channel, name in enumerate(CHANNEL_NAMES):
            per_seed = per_group.get(name, {})
            counts = self.channel_counts(channel)
            channels[name] = {
                "counts": counts,
                **judgement(
                    self.rows,
                    counts,
                    intervals[f"{name}/delta.log_loss.static"]["high_97.5"],
                ),
                "episode_macro": {
                    metric: episode_macro(per_seed, seeds, metric)
                    for metric in sorted({m for s in per_seed.values() for m in s})
                },
                "intervals": {
                    metric: intervals[f"{name}/{metric}"] for metric in TARGETS
                },
                "calibration": self.calibration[channel].summary(),
            }
        return {
            "counts": {"rows": self.rows}
            | {name: len(self.sets[name]) for name in _SETS},
            "channels": channels,
        }


def judgement(
    rows: int, counts: dict[str, object], high: float | None
) -> dict[str, object]:
    """population × channelの判定。判定対象外の理由を先に確かめる。"""
    reason = None
    if not rows:
        reason = "no_rows"
    elif not counts["positive_episodes"]:
        reason = "no_positive"
    elif counts["positive_hanchan"] < HOLD_MIN_HANCHAN:
        reason = "positive_hanchan_below_minimum"
    elif counts["positive_episodes"] < HOLD_MIN_EPISODES:
        reason = "positive_episodes_below_minimum"
    if reason is not None:
        return {"verdict": "excluded", "reason": reason}
    return {"verdict": "improved" if high < 0 else "not_confirmed", "reason": None}


def valid_losses(rows: Sequence[ShapeRow], model: WaitShapeModel) -> list[float]:
    """validの、channelごとのepisode-macro log loss（静的slot、推定器の最終rawで）。"""
    groups = _Groups()
    for row in rows:
        _, raw = predict_raw(row, model)
        groups.add(
            row.seed,
            row.episode,
            "valid",
            {
                name: _mean(
                    log_loss(clip(raw[channel][i] / SCALE), row.labels[channel][i])
                    for i in _STATIC[channel]
                )
                for channel, name in enumerate(CHANNEL_NAMES)
            },
        )
    per_seed = groups.per_seed()["valid"]
    return [episode_macro(per_seed, sorted(per_seed), name) for name in CHANNEL_NAMES]


# --- select / test ------------------------------------------------------------


def _model_value(model: WaitShapeModel) -> dict[str, object]:
    return {
        "population": model.population,
        "feature_set": model.feature_set,
        "weights": {
            name: None if weights is None else [list(pair) for pair in weights]
            for name, weights in zip(CHANNEL_NAMES, model.weights)
        },
    }


def _model_from_value(value, population: str) -> WaitShapeModel:
    if value["population"] != population or set(value["weights"]) != set(CHANNEL_NAMES):
        raise _E("the selection's shape model does not match the registered layout")
    return WaitShapeModel(
        population=population,
        weights=tuple(
            None
            if value["weights"][name] is None
            else tuple((n, float(w)) for n, w in value["weights"][name])
            for name in CHANNEL_NAMES
        ),
        feature_set=value["feature_set"],
    )


def _rates_from_value(value) -> tuple[tuple[float, ...], ...]:
    rates = tuple(tuple(float(p) for p in value[name]) for name in CHANNEL_NAMES)
    if any(
        len(rate) != 34
        or not all(0.0 <= p <= 1.0 for p in rate)
        or any(p and i not in static for i, p in enumerate(rate))
        for rate, (_, _, static) in zip(rates, CHANNELS)
    ):
        raise _E("the rate baseline must be 34 probabilities per channel")
    return rates


def select_population(
    population: str, train: Sequence[ShapeRow], valid: Sequence[ShapeRow]
):
    """trainで当てはめ、validでchannelごとのL2を選ぶ。"""
    if not train or not valid:
        raise _E(f"{population}: no train or valid row")
    rates = fit_rate(train)
    fits = [
        None
        if is_zero_model(population, name)
        else fit_channel(train, channel, L2_GRID)
        for channel, name in enumerate(CHANNEL_NAMES)
    ]
    losses = {
        l2: valid_losses(
            valid,
            WaitShapeModel(
                population, tuple(None if fit is None else fit[l2] for fit in fits)
            ),
        )
        for l2 in L2_GRID
    }
    selected, grid = {}, {}
    for channel, name in enumerate(CHANNEL_NAMES):
        if fits[channel] is None:
            selected[name] = None
            grid[name] = "structural zero model"
            continue
        selected[name] = min(L2_GRID, key=lambda l2: losses[l2][channel])
        grid[name] = [{"l2": l2, "loss": losses[l2][channel]} for l2 in L2_GRID]
    model = WaitShapeModel(
        population,
        tuple(
            None if fit is None else fit[selected[name]]
            for fit, name in zip(fits, CHANNEL_NAMES)
        ),
    )
    return model, rates, selected, grid


def _settings() -> dict[str, object]:
    return {
        "feature_set": FEATURE_SET,
        "consistency": CONSISTENCY,
        "source_schema": MANIFEST_SCHEMA,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "clip_epsilon": CLIP_EPSILON,
        "exclusion_rule": {
            "min_positive_hanchan": HOLD_MIN_HANCHAN,
            "min_positive_episodes": HOLD_MIN_EPISODES,
        },
    }


def select(
    sources: Sequence[Path],
    expected: dict[str, Sequence[int]],
    level1: Level1,
    code_revision: str,
) -> dict[str, object]:
    identity = check_population(sources, expected)
    data = read_rows(sources, level1)
    populations = {}
    for population in POPULATIONS:
        train, valid, dev = (data[split].rows[population] for split in SPLITS)
        model, rates, selected, grid = select_population(population, train, valid)
        reports = {}
        for name, rows, seeds in (
            ("valid", valid, expected["valid"]),
            ("dev_eval", dev, expected["eval"]),
        ):
            report = Report()
            report.add(rows, model, rates)
            reports[name] = report.document(seeds)
        populations[population] = {
            "selected_l2": selected,
            "valid_grid": grid,
            "model": _model_value(model),
            "rate": dict(zip(CHANNEL_NAMES, map(list, rates))),
            "valid": reports["valid"],
            "dev_eval": {
                "note": "development check only; not a claim",
                **reports["dev_eval"],
            },
        }
    return {
        "schema": SELECTION_SCHEMA,
        **_settings(),
        "code_revision": code_revision,
        "population": identity,
        "level1": level1.identity(),
        "splits": {name: sorted(seeds) for name, seeds in expected.items()},
        "coverage": {
            name: data[split].coverage
            for name, split in zip(("train", "valid", "dev_eval"), SPLITS)
        },
        "l2_grid": list(L2_GRID),
        "populations": populations,
    }


def _check_selection(selection: dict[str, object], level1: Level1) -> None:
    if selection.get("schema") != SELECTION_SCHEMA:
        raise _E("unknown selection schema")
    if selection.get("feature_set") != FEATURE_SET:
        raise _E("the selection was made with another feature set")
    if selection.get("consistency") != CONSISTENCY:
        raise _E("the selection was made with another consistency policy")
    if selection.get("level1") != level1.identity():
        raise _E("the wait estimators differ from the selection's")
    if set(selection.get("populations", ())) != set(POPULATIONS):
        raise _E("the selection must have both populations")


def run_test(
    sources: Sequence[Path],
    test_seeds: Sequence[int],
    selection: dict[str, object],
    level1: Level1,
    code_revision: str,
    test_producer: dict[str, object] | None = None,
) -> dict[str, object]:
    _check_selection(selection, level1)
    used = {seed for seeds in selection["splits"].values() for seed in seeds}
    if used & set(test_seeds):
        raise _E("a test seed was used by the selection")
    identity = check_population(sources, {"train": [], "valid": [], "eval": test_seeds})
    registration = check_test_producer(
        identity, selection["population"]["producer"], test_producer
    )
    models = {
        population: (
            _model_from_value(value["model"], population),
            _rates_from_value(value["rate"]),
        )
        for population, value in selection["populations"].items()
    }
    data = read_rows(sources, level1)["test"]
    populations = {}
    for population in POPULATIONS:
        report = Report()
        report.add(data.rows[population], *models[population])
        populations[population] = report.document(test_seeds)
    return {
        "schema": RESULT_SCHEMA,
        **_settings(),
        "code_revision": code_revision,
        "population": identity,
        "level1": level1.identity(),
        "test_seeds": sorted(test_seeds),
        "coverage": data.coverage,
        "verdicts": {
            population: {
                name: value["verdict"] for name, value in document["channels"].items()
            }
            for population, document in populations.items()
        },
        "populations": populations,
    } | ({} if registration is None else {"producer_registration": registration})


def _revision(text: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", text):
        raise argparse.ArgumentTypeError("a full 40-hex commit id is required")
    return text


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
        command.add_argument("--riichi-wait-selection", type=Path, required=True)
        command.add_argument("--riichi-wait-sha256", required=True)
        command.add_argument("--open-wait-selection", type=Path, required=True)
        command.add_argument("--open-wait-sha256", required=True)
        command.add_argument("--code-revision", type=_revision, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("sources", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        parser.error(f"refusing to overwrite {arguments.output}")
    level1 = load_level1(
        arguments.riichi_wait_selection,
        arguments.riichi_wait_sha256,
        arguments.open_wait_selection,
        arguments.open_wait_sha256,
    )
    if arguments.command == "select":
        document = select(
            arguments.sources,
            {
                "train": arguments.train,
                "valid": arguments.valid,
                "eval": arguments.dev_eval,
            },
            level1,
            arguments.code_revision,
        )
        summary = {
            population: {
                "selected_l2": value["selected_l2"],
                "dev_eval": {
                    name: channel["intervals"]["delta.log_loss.static"]
                    for name, channel in value["dev_eval"]["channels"].items()
                },
            }
            for population, value in document["populations"].items()
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
            arguments.sources,
            arguments.test,
            json.loads(data),
            level1,
            arguments.code_revision,
            test_producer,
        )
        document["selection_sha256"] = arguments.selection_sha256
        if test_producer is not None:
            document["producer_registration"]["sha256"] = arguments.test_producer_sha256
        summary = {
            "verdicts": document["verdicts"],
            "delta": {
                population: {
                    name: channel["intervals"]["delta.log_loss.static"]
                    for name, channel in value["channels"].items()
                }
                for population, value in document["populations"].items()
            },
        }
    arguments.output.write_text(canonical_json_text(document), encoding="utf-8")
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
