"""HandBeliefの構造的テーブルの推定精度を測る（lisbun/lisjong#257、学習専用の経路）。

#256 v1のsource（``hand_belief_source``）を読み、現行推定器と単純baselineの**絶対精度**を
測る。改善の合否判定ではない。測定条件は#257の事前登録に従う。

```text
python -m lisjong.learning.hand_belief_accuracy \\
    --train 933000..933159 --valid 933160..933239 --eval 933240..933399 \\
    --selection-245 SELECTION.json --selection-245-sha256 HEX \\
    --output RESULT.json SOURCE [SOURCE ...]
```

sourceは複数（chunk）を受け取る。各chunkのmanifestの ``train`` / ``valid`` / ``test`` を
合わせたものが、指定した train / valid / eval のseed範囲とちょうど一致しなければならない
（重複・欠落・producerの不一致はエラー）。sourceの ``test`` 分割をevalとして使う。

## 比べるもの

- ``expected_count`` / ``red_five_probability``: 条件付き一様推定器
  （``estimate_conditional_uniform_hand_belief``。他家のslot数は ``13 - 3 * 副露数``）
- ``wait_probability``: 牌種ごとの待ち率（trainの全行、Jeffreys 0.5）。#245推定器は
  凍結済みのselectionをそのまま使い、S1の対象条件の行（下記）でだけ値を出す
- 形別7種: channel・牌種ごとの出現率（trainの全行、Jeffreys 0.5）

validで選ぶハイパーパラメータはない。validの行はchunkと一緒に読み込まれるが、fit・選択・
報告のどれにも使わない。

## #245の対象条件（S1と同じ）

観測者は非リーチ、他家のリーチ者がちょうど1人、合法な打牌の牌種が2以上、選んだ行動が打牌。
その判断の**リーチ者の行**だけが対象で、同じ判断の他の他家の行と、対象外の判断の行は
未提供（``None``）として数え、ゼロ予測にしない。#245とbaselineの差は対象の行だけで計算する。

## 標本と集約

- 行: 観測者の1打牌判断 × 他家1席。エピソード: (半荘, 局instance, 他家席)。局instanceは
  半荘内で局が始まった順の通し番号で、(場風, 局, 本場)と1対1でなければエラー
- 層（行ごと、その他家の判断時点の状態）: リーチ者 / 副露者（暗槓以外の副露あり）/
  門前非リーチ者
- 集約: 行内で平均 → エピソード内で行平均 → エピソード間で平均。層別・#245対象は、
  各エピソードの当該行だけで行平均し、その行を持つエピソード間で平均する
- trainのfit（出現率）も同じ重み（エピソード1をその行で等分）を使う
- 確率はすべて ``[1e-6, 1 - 1e-6]`` にclipしてから指標を計算する
- 区間: 半荘単位のpaired bootstrap（2,000回、seed 257）、95%の百分位区間
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from math import log
from pathlib import Path
from random import Random

from lisjong.belief.canonical_axes import (
    red_five_index,
    tile_type_from_index,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
from lisjong.belief.conditional_uniform_hand_belief import (
    estimate_conditional_uniform_hand_belief,
)
from lisjong.belief.fixed_point import SCALE
from lisjong.belief.hand_belief import _WAIT_MECHANISM_FIELDS, HandBelief
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.learning._canonical import canonical_json_text
from lisjong.learning.hand_belief_source import (
    HandBeliefLabelledDecision,
    read_labelled_source,
    read_manifest,
)
from lisjong.learning.riichi_deal_in_estimator import riichi_view
from lisjong.learning.riichi_wait_estimator import CLIP_EPSILON, LogisticWaitModel
from lisjong.learning.riichi_wait_evaluation import (
    _check_selection,
    _models_from_value,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import TileCategory

RESULT_SCHEMA = "lisjong-hand-belief-accuracy-result-v1"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 257
JEFFREYS = 0.5
STRATA = ("riichi", "open", "closed_non_riichi")
RED_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)
CHANNELS = tuple(
    (
        name.removesuffix("_raw").removesuffix("_probability").removesuffix("_wait"),
        name,
        frozenset(v),
    )
    for name, v in _WAIT_MECHANISM_FIELDS
)
TILE_CLASSES = ("n19", "n28", "n37", "n456", "honor")
TURNS = ("early(<=6)", "middle(7-11)", "late(>=12)")
CALIBRATION_BINS = 10


class HandBeliefAccuracyError(ValueError):
    """入力が事前登録の条件に合わない。"""


_E = HandBeliefAccuracyError


def clip(value: float) -> float:
    return min(max(value, CLIP_EPSILON), 1.0 - CLIP_EPSILON)


def log_loss(p: float, label: bool) -> float:
    return -log(p if label else 1.0 - p)


def tile_class(index: int) -> str:
    tile = tile_type_from_index(index)
    if tile.category is TileCategory.HONOR:
        return "honor"
    return {1: "n19", 9: "n19", 2: "n28", 8: "n28", 3: "n37", 7: "n37"}.get(
        tile.rank, "n456"
    )


def turn_bucket(discards: int) -> str:
    return TURNS[0] if discards <= 6 else TURNS[1] if discards <= 11 else TURNS[2]


_TILE_CLASS = tuple(tile_class(index) for index in range(34))


# --- sourceの集合 ----------------------------------------------------------


def seed_range(text: str) -> tuple[int, ...]:
    first, _, last = text.partition("..")
    seeds = tuple(range(int(first), int(last or first) + 1))
    if not seeds:
        raise argparse.ArgumentTypeError("empty seed range")
    return seeds


def check_population(
    sources: Sequence[Path], expected: dict[str, Sequence[int]]
) -> dict[str, object]:
    """chunkのmanifestを照合する。ラベルは読まない。"""
    if not sources:
        raise _E("no source")
    producers, seen = set(), defaultdict(list)
    manifests = {}
    for source in sources:
        manifest = read_manifest(source)
        producers.add(tuple(sorted(manifest.producer.items())))
        for name, split in (("train", "train"), ("valid", "valid"), ("eval", "test")):
            seen[name].extend(manifest.splits[split])
        manifests[str(source)] = hashlib.sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest()
    if len(producers) != 1:
        raise _E("the sources were made by different producers")
    every = [seed for name in seen for seed in seen[name]]
    if len(every) != len(set(every)):
        raise _E("a seed appears in more than one source or split")
    for name in ("train", "valid", "eval"):
        if sorted(seen[name]) != sorted(expected[name]):
            raise _E(f"the {name} seeds differ from the registered range")
    return {"producer": dict(producers.pop()), "manifest_sha256": manifests}


# --- 行 --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    """観測者の1判断 × 他家1席。確率・正解は34牌種のcanonical順。"""

    seed: int
    episode: tuple[int, int]  # (局instance, 他家席)
    stratum: str
    turn: str
    candidates: frozenset[int]
    unseen: tuple[int, ...]
    expected_count: tuple[float, ...]
    red_five: tuple[float, ...]
    truth: HandBelief
    in_245_scope: bool
    wait_245: tuple[float, ...] | None


def _stratum(player) -> str:
    if player.riichi is not RiichiState.NONE:
        return "riichi"
    if any(meld.kind is not MeldKind.ANKAN for meld in player.melds):
        return "open"
    return "closed_non_riichi"


def kyoku_instances(rows: Sequence[HandBeliefLabelledDecision]) -> dict[int, int]:
    """sequence → 局instance（半荘内で局が始まった順）。同じ局が離れて再び現れたらエラー。"""
    order: dict[tuple, int] = {}
    previous = None
    result = {}
    for row in sorted(rows, key=lambda r: r.decision.key.sequence):
        state = row.decision.policy_input.round
        kyoku = (state.round_wind, state.hand_number, state.honba)
        if kyoku != previous:
            if kyoku in order:
                raise _E(f"seed {row.decision.key.seed}: kyoku {kyoku} reappears")
            order[kyoku] = len(order)
            previous = kyoku
        result[row.decision.key.sequence] = order[kyoku]
    return result


def _in_245_scope(row: HandBeliefLabelledDecision) -> int | None:
    """S1の対象ならリーチ者の席を返す。"""
    decision = row.decision
    pi = decision.policy_input
    if not isinstance(decision.selected_action, DiscardAction):
        return None
    candidates = {
        action.tile.tile_type
        for action in decision.legal_actions
        if isinstance(action, DiscardAction)
    }
    if len(candidates) < 2:
        return None
    try:
        return riichi_view(pi).riichi_seat
    except ValueError:
        return None


def build_rows(
    labelled: Sequence[HandBeliefLabelledDecision], model_245: LogisticWaitModel
) -> Iterator[Row]:
    by_seed = defaultdict(list)
    for row in labelled:
        by_seed[row.decision.key.seed].append(row)
    for seed in sorted(by_seed):
        instances = kyoku_instances(by_seed[seed])
        for row in by_seed[seed]:
            yield from _rows_of(row, instances[row.decision.key.sequence], model_245)


def _rows_of(
    row: HandBeliefLabelledDecision, instance: int, model_245: LogisticWaitModel
) -> Iterator[Row]:
    decision = row.decision
    pi = decision.policy_input
    dealer = pi.round.dealer_seat
    slots = [0, 0, 0, 0]
    for seat in Seat:
        if seat != pi.self_seat:
            slots[wind_index(wind_for_seat(seat, dealer))] = 13 - 3 * len(
                pi.players[seat].melds
            )
    uniform = estimate_conditional_uniform_hand_belief(pi, tuple(slots))
    unseen = tuple(derive_remaining_tile_inventory(pi).remaining_tile_counts)
    candidates = frozenset(
        tile_type_index(action.tile.tile_type)
        for action in decision.legal_actions
        if isinstance(action, DiscardAction)
    )
    scope_seat = _in_245_scope(row)
    wait_245 = None
    if scope_seat is not None:
        predicted = model_245.predict(pi)
        wait_245 = tuple(predicted[tile_type_from_index(i)] for i in range(34))
    for opponent in row.opponents:
        player = pi.players[opponent.seat]
        belief = uniform.hand(wind_for_seat(Seat(opponent.seat), dealer))
        in_scope = opponent.seat == scope_seat
        yield Row(
            seed=decision.key.seed,
            episode=(instance, opponent.seat),
            stratum=_stratum(player),
            turn=turn_bucket(len(player.discards)),
            candidates=candidates,
            unseen=unseen,
            expected_count=tuple(raw / SCALE for raw in belief.expected_count_raw),
            red_five=tuple(raw / SCALE for raw in belief.red_five_probability_raw),
            truth=opponent.truth,
            in_245_scope=in_scope,
            wait_245=wait_245 if in_scope else None,
        )


def _labels(raw: tuple[int, ...] | None) -> tuple[bool, ...]:
    if raw is None:
        raise _E("the truth lacks a wait table")
    return tuple(value == SCALE for value in raw)


def wait_labels(row: Row) -> tuple[bool, ...]:
    return _labels(row.truth.wait_probability_raw)


def channel_labels(row: Row, field_name: str) -> tuple[bool, ...]:
    return _labels(getattr(row.truth, field_name))


def red_labels(row: Row) -> tuple[bool, ...]:
    return tuple(
        row.truth.red_five_probability_raw[red_five_index(category)] == SCALE
        for category in RED_CATEGORIES
    )


# --- 出現率baseline（train） -------------------------------------------------


def episode_weights(rows: Sequence[Row]) -> list[float]:
    counts = defaultdict(int)
    for row in rows:
        counts[(row.seed, row.episode)] += 1
    return [1.0 / counts[(row.seed, row.episode)] for row in rows]


@dataclass
class RateFit:
    """牌種（channelごと）の重み付き出現数。"""

    weight: float = 0.0
    positives: dict[str, list[float]] = field(
        default_factory=lambda: {
            name: [0.0] * 34 for name in ("wait", *[c[0] for c in CHANNELS])
        }
    )

    def add(self, rows: Sequence[Row]) -> None:
        for row, weight in zip(rows, episode_weights(rows)):
            self.weight += weight
            tables = {"wait": wait_labels(row)} | {
                name: channel_labels(row, field_name)
                for name, field_name, _ in CHANNELS
            }
            for name, labels in tables.items():
                target = self.positives[name]
                for index, label in enumerate(labels):
                    if label:
                        target[index] += weight

    def probabilities(self) -> dict[str, tuple[float, ...]]:
        if not self.weight:
            raise _E("no train row")
        return {
            name: tuple(
                (value + JEFFREYS) / (self.weight + 2 * JEFFREYS) for value in values
            )
            for name, values in self.positives.items()
        }


# --- 評価 -----------------------------------------------------------------


class _Groups:
    """(半荘, エピソード, group) ごとに指標の和と行数を持ち、episode-macroへ集約する。"""

    def __init__(self) -> None:
        self.cells: dict[tuple[int, tuple, str], dict[str, list[float]]] = {}

    def add(self, seed: int, episode: tuple, group: str, values: dict) -> None:
        cell = self.cells.setdefault((seed, episode, group), {})
        for metric, value in values.items():
            pair = cell.setdefault(metric, [0.0, 0])
            pair[0] += value
            pair[1] += 1

    def per_seed(self) -> dict[str, dict[int, dict[str, list[float]]]]:
        """group → 半荘 → 指標 → [エピソード平均の和, エピソード数]。"""
        result: dict = defaultdict(lambda: defaultdict(dict))
        for (seed, _, group), cell in self.cells.items():
            target = result[group][seed]
            for metric, (total, count) in cell.items():
                pair = target.setdefault(metric, [0.0, 0])
                pair[0] += total / count
                pair[1] += 1
        return result


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def row_metrics(row: Row, rates: dict[str, tuple[float, ...]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    counts = [raw / SCALE for raw in row.truth.expected_count_raw]
    errors = [(p - c) ** 2 for p, c in zip(row.expected_count, counts)]
    metrics["expected_count.mse.all"] = _mean(errors)
    metrics["expected_count.mse.candidates"] = _mean(errors[i] for i in row.candidates)
    reds = red_labels(row)
    red_p = [clip(p) for p in row.red_five]
    metrics["red_five.log_loss"] = _mean(map(log_loss, red_p, reds))
    metrics["red_five.brier"] = _mean((p - y) ** 2 for p, y in zip(red_p, reds))
    waits = wait_labels(row)
    wait_p = [clip(p) for p in rates["wait"]]
    _probability_metrics(metrics, "wait.rate", wait_p, waits, row.candidates)
    metrics["wait.rate.predicted_kinds"] = sum(wait_p)
    metrics["wait.actual_kinds"] = sum(waits)
    metrics["wait.rate.predicted_unseen_weighted"] = sum(
        u * p for u, p in zip(row.unseen, wait_p)
    )
    metrics["wait.actual_unseen_weighted"] = sum(
        u for u, w in zip(row.unseen, waits) if w
    )
    for name, field_name, valid in CHANNELS:
        labels = channel_labels(row, field_name)
        p = [clip(v) for v in rates[name]]
        losses = [log_loss(q, y) for q, y in zip(p, labels)]
        metrics[f"channel.{name}.log_loss.all"] = _mean(losses)
        metrics[f"channel.{name}.log_loss.valid_slots"] = _mean(
            losses[i] for i in valid
        )
        metrics[f"channel.{name}.brier.all"] = _mean(
            (q - y) ** 2 for q, y in zip(p, labels)
        )
    return metrics


def _probability_metrics(metrics, prefix, p, labels, candidates) -> None:
    losses = [log_loss(q, y) for q, y in zip(p, labels)]
    metrics[f"{prefix}.log_loss.all"] = _mean(losses)
    metrics[f"{prefix}.log_loss.candidates"] = _mean(losses[i] for i in candidates)
    metrics[f"{prefix}.brier.all"] = _mean((q - y) ** 2 for q, y in zip(p, labels))
    for name in TILE_CLASSES:
        metrics[f"{prefix}.log_loss.class.{name}"] = _mean(
            loss for loss, c in zip(losses, _TILE_CLASS) if c == name
        )


def scope_metrics(row: Row, rates: dict[str, tuple[float, ...]]) -> dict[str, float]:
    """#245の対象行だけ: #245とbaselineを同じ行で。"""
    metrics: dict[str, float] = {}
    waits = wait_labels(row)
    _probability_metrics(
        metrics, "wait.245", [clip(p) for p in row.wait_245], waits, row.candidates
    )
    _probability_metrics(
        metrics, "wait.rate", [clip(p) for p in rates["wait"]], waits, row.candidates
    )
    metrics["wait.245.predicted_kinds"] = sum(row.wait_245)
    metrics["wait.actual_kinds"] = sum(waits)
    for suffix in ("log_loss.all", "log_loss.candidates", "brier.all"):
        metrics[f"wait.245_minus_rate.{suffix}"] = (
            metrics[f"wait.245.{suffix}"] - metrics[f"wait.rate.{suffix}"]
        )
    return metrics


@dataclass
class _Calibration:
    weight: list[float] = field(default_factory=lambda: [0.0] * CALIBRATION_BINS)
    predicted: list[float] = field(default_factory=lambda: [0.0] * CALIBRATION_BINS)
    observed: list[float] = field(default_factory=lambda: [0.0] * CALIBRATION_BINS)
    slots: list[int] = field(default_factory=lambda: [0] * CALIBRATION_BINS)

    def add(self, weight: float, p: Sequence[float], labels: Sequence[bool]) -> None:
        share = weight / len(p)
        for q, y in zip(p, labels):
            b = min(int(q * CALIBRATION_BINS), CALIBRATION_BINS - 1)
            self.weight[b] += share
            self.predicted[b] += share * q
            self.observed[b] += share * y
            self.slots[b] += 1

    def summary(self) -> list[dict[str, float]]:
        return [
            {
                "bin": f"[{b / CALIBRATION_BINS:.1f},{(b + 1) / CALIBRATION_BINS:.1f})",
                "slots": self.slots[b],
                "weight": self.weight[b],
                "mean_probability": self.predicted[b] / self.weight[b],
                "observed_rate": self.observed[b] / self.weight[b],
            }
            for b in range(CALIBRATION_BINS)
            if self.weight[b]
        ]


@dataclass
class Evaluation:
    groups: _Groups = field(default_factory=_Groups)
    calibration: dict[str, _Calibration] = field(
        default_factory=lambda: defaultdict(_Calibration)
    )
    support: dict[str, dict[str, set]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    rows: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, rows: Sequence[Row], rates: dict[str, tuple[float, ...]]) -> None:
        for row, weight in zip(rows, episode_weights(rows)):
            values = row_metrics(row, rates)
            groups = ["all", f"stratum.{row.stratum}", f"turn.{row.turn}"]
            for group in groups:
                self.groups.add(row.seed, row.episode, group, values)
            self._count(row)
            self.calibration["wait.rate"].add(
                weight, [clip(p) for p in rates["wait"]], wait_labels(row)
            )
            self.calibration["red_five.uniform"].add(
                weight, [clip(p) for p in row.red_five], red_labels(row)
            )
            if row.in_245_scope:
                scoped = scope_metrics(row, rates)
                self.groups.add(row.seed, row.episode, "245_scope", scoped)
                self.groups.add(
                    row.seed, row.episode, f"245_scope.turn.{row.turn}", scoped
                )
        # #245の対象行だけで別に重みを付け直す（エピソード1を対象行で等分）
        scoped_rows = [row for row in rows if row.in_245_scope]
        for row, weight in zip(scoped_rows, episode_weights(scoped_rows)):
            labels = wait_labels(row)
            self.calibration["wait.245.scope"].add(
                weight, [clip(p) for p in row.wait_245], labels
            )
            self.calibration["wait.rate.scope"].add(
                weight, [clip(p) for p in rates["wait"]], labels
            )

    def _count(self, row: Row) -> None:
        episode = (row.seed, row.episode)
        tables = {"wait": wait_labels(row)} | {
            name: channel_labels(row, f) for name, f, _ in CHANNELS
        }
        for group in ("all", f"stratum.{row.stratum}"):
            self.rows[f"{group}.rows"] += 1
            self.support[group]["episodes"].add(episode)
            self.support[group]["hanchan"].add(row.seed)
            self.rows[f"{group}.245_scope_rows"] += row.in_245_scope
            if row.in_245_scope:
                self.support[group]["245_scope_episodes"].add(episode)
            for name, labels in tables.items():
                if any(labels):
                    self.rows[f"{group}.{name}.positive_rows"] += 1
                    self.support[group][f"{name}.positive_episodes"].add(episode)
                    self.support[group][f"{name}.positive_hanchan"].add(row.seed)

    def counts(self) -> dict[str, dict[str, int]]:
        result: dict = defaultdict(dict)
        for key, value in self.rows.items():
            group = (
                ".".join(key.split(".")[:2]) if key.startswith("stratum.") else "all"
            )
            result[group][key[len(group) + 1 :]] = value
        for group, sets in self.support.items():
            for name, values in sets.items():
                result[group][name] = len(values)
        return {group: dict(sorted(values.items())) for group, values in result.items()}


def episode_macro(per_seed: dict[int, dict[str, list[float]]], seeds, metric) -> float:
    total = count = 0.0
    for seed in seeds:
        pair = per_seed.get(seed, {}).get(metric)
        if pair:
            total += pair[0]
            count += pair[1]
    return total / count if count else None


PRIMARY = (
    "expected_count.mse.all",
    "red_five.log_loss",
    "wait.rate.log_loss.all",
    *(f"channel.{name}.log_loss.all" for name, _, _ in CHANNELS),
    *(f"channel.{name}.log_loss.valid_slots" for name, _, _ in CHANNELS),
)
PRIMARY_245 = (
    "wait.245.log_loss.all",
    "wait.rate.log_loss.all",
    "wait.245_minus_rate.log_loss.all",
)


def bootstrap(
    per_group: dict[str, dict[int, dict[str, list[float]]]],
    seeds: Sequence[int],
    targets: Sequence[tuple[str, str]],
) -> dict[str, dict[str, object]]:
    """eval半荘を復元抽出し、各 (group, 指標) のepisode-macroの95%区間を求める。

    全targetで同じ再標本を使う（paired）。
    """
    random = Random(BOOTSTRAP_SEED)
    samples = [
        [random.choice(seeds) for _ in seeds] for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    result = {}
    for group, metric in targets:
        per_seed = per_group.get(group, {})
        draws = sorted(
            value
            for value in (episode_macro(per_seed, s, metric) for s in samples)
            if value is not None
        )
        result[f"{group}/{metric}"] = {
            "point": episode_macro(per_seed, seeds, metric),
            "low_2.5": draws[int(0.025 * len(draws))] if draws else None,
            "high_97.5": draws[int(0.975 * len(draws)) - 1] if draws else None,
            "resamples_with_rows": len(draws),
        }
    return result


def load_245_model(path: Path, sha256: str) -> LogisticWaitModel:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise _E("the #245 selection does not match the registered SHA-256")
    chosen = json.loads(data.decode("utf-8"))
    _check_selection(chosen)
    return _models_from_value(chosen["models"])[2]


def _rows_of_split(source: Path, split: str, model_245) -> list[Row]:
    manifest, labelled = read_labelled_source(source)
    wanted = set(manifest.splits[split])
    chosen = [r for r in labelled if r.decision.key.seed in wanted]
    if {r.decision.key.seed for r in chosen} != wanted:
        raise _E(f"{source}: a {split} seed has no decision")
    return list(build_rows(chosen, model_245))


def evaluate_population(
    sources: Sequence[Path],
    expected: dict[str, Sequence[int]],
    model_245: LogisticWaitModel,
) -> dict[str, object]:
    identity = check_population(sources, expected)
    fit = RateFit()
    for source in sources:
        fit.add(_rows_of_split(source, "train", model_245))
    rates = fit.probabilities()
    evaluation = Evaluation()
    for source in sources:
        evaluation.add(_rows_of_split(source, "test", model_245), rates)
    per_group = evaluation.groups.per_seed()
    seeds = sorted(expected["eval"])
    targets = [
        (group, metric)
        for group in ("all", *(f"stratum.{s}" for s in STRATA))
        for metric in PRIMARY
    ] + [("245_scope", metric) for metric in PRIMARY_245]
    point = {
        group: {
            metric: episode_macro(per_seed, seeds, metric)
            for metric in sorted({m for s in per_seed.values() for m in s})
        }
        for group, per_seed in sorted(per_group.items())
    }
    return {
        "schema": RESULT_SCHEMA,
        "population": identity,
        "splits": {name: sorted(seeds) for name, seeds in expected.items()},
        "valid_used": False,
        "rates": {"train_episode_weight": fit.weight, "probabilities": rates},
        "counts": evaluation.counts(),
        "episode_macro": point,
        "intervals": bootstrap(per_group, seeds, targets),
        "calibration": {
            name: cal.summary() for name, cal in sorted(evaluation.calibration.items())
        },
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "clip_epsilon": CLIP_EPSILON,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("--train", type=seed_range, required=True)
    parser.add_argument("--valid", type=seed_range, required=True)
    parser.add_argument("--eval", type=seed_range, required=True)
    parser.add_argument("--selection-245", type=Path, required=True)
    parser.add_argument("--selection-245-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("sources", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        parser.error(f"refusing to overwrite {arguments.output}")
    model = load_245_model(arguments.selection_245, arguments.selection_245_sha256)
    expected = {
        "train": arguments.train,
        "valid": arguments.valid,
        "eval": arguments.eval,
    }
    document = evaluate_population(arguments.sources, expected, model)
    document["selection_245_sha256"] = arguments.selection_245_sha256
    arguments.output.write_text(canonical_json_text(document), encoding="utf-8")
    json.dump(document["intervals"], sys.stdout, ensure_ascii=False, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
