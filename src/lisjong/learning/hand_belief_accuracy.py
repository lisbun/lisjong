"""HandBeliefの構造的テーブルの推定精度を測る（lisbun/lisjong#257、学習専用の経路）。

#256 v1のsource（``hand_belief_source``）を読み、現行推定器と単純baselineの**絶対精度**を
測る。改善の合否判定ではない。測定条件は#257の事前登録に従う。

```text
python -m lisjong.learning.hand_belief_accuracy \\
    --preset hand-belief-accuracy-257 --preset-sha256 HEX \\
    --selection-245 SELECTION.json --selection-245-sha256 HEX \\
    --output RESULT.json SOURCE [SOURCE ...]
```

## 設定の2層（lisbun/lisjong#274）

- **protocol preset**（``Preset``）: 分割のseed範囲、bootstrap回数・乱数seed、Jeffreys、
  clip幅、判定保留の閾値、``purpose``。version付きのimmutableな値で、事前登録した
  ``--preset-sha256`` と一致しなければラベルを読む前に拒否する。実行時に個別の値を
  上書きするoptionはない。変える場合は新しいpresetを事前登録する（``--preset`` は
  組み込みpresetの名前か、presetのJSON file）。#257の値は ``PRESET_257``
- **差し替え点**: 比べる推定器（``ScopedWaitEstimator``。#245は ``Riichi245Estimator``）。
  推定器が値を出す行（scope）を自分で宣言し、それ以外の行は未提供（``None``）のまま扱う。
  比較baseline（trainの出現率）と集約・指標は共通で、presetの下で固定する

``purpose`` が ``formal-test`` なら、``--allocation-identity`` / ``--ledger-revision`` /
``--arena-revision`` が必須で、結果に記録する。live ledgerとの照合はlisjong-arena側
（``check-allocation``）の責務で、ここでは検査せず記録だけを行う（arenaへ依存しない）。

sourceは複数（chunk）を受け取る。各chunkのmanifestの ``train`` / ``valid`` / ``test`` を
合わせたものが、指定した train / valid / eval のseed範囲とちょうど一致しなければならない
（重複・欠落・producerの不一致はエラー）。sourceの ``test`` 分割をevalとして使う。

## 比べるもの

- ``expected_count`` / ``red_five_probability``: 条件付き一様推定器
  （``estimate_conditional_uniform_hand_belief``。他家のslot数は ``13 - 3 * 副露数``）
- ``wait_probability``: 牌種ごとの待ち率（trainの全行、Jeffreys）。#245推定器は
  凍結済みのselectionをそのまま使い、S1の対象条件の行（下記）でだけ値を出す
- 形別7種: channel・牌種ごとの出現率（trainの全行、Jeffreys 0.5）。そのchannelが構造上
  占め得ないslotは``HandBelief``のcanonical zeroと同じく予測0とする（log lossではclipで
  ``1e-6``になる）

validで選ぶハイパーパラメータはない。validは正例・対象行の件数の報告にだけ使い、fit・選択・
指標には使わない。

## 希少な層の扱い（#257計画の6）

- 待ちの指標は、全行に加えて聴牌行（待ちが1牌種以上）だけ・非聴牌行だけでも出す
- 正例の行数・エピソード数・独立半荘数をtrain / valid / eval別に出す。ラベル・集計は省略しない
- evalで正例を含む独立半荘が``HOLD_MIN_HANCHAN``未満、または正例エピソードが
  ``HOLD_MIN_EPISODES``未満の層・tableは ``held``（判定保留）、正例が0なら ``not_estimable``

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
- 区間: 半荘単位のpaired bootstrap（#257は2,000回、seed 257）、95%の百分位区間
"""

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from math import log
from pathlib import Path
from random import Random
from typing import Protocol

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

RESULT_SCHEMA = "lisjong-hand-belief-accuracy-result-v2"
PRESET_SCHEMA = "lisjong-hand-belief-accuracy-preset-v1"
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
# evalで正例を含む独立半荘・正例エピソードがこれ未満の層・tableは判定を保留する
HOLD_MIN_HANCHAN = 50
HOLD_MIN_EPISODES = 100


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
    manifests, coverages = {}, {}
    for source in sources:
        manifest = read_manifest(source)
        producers.add(tuple(sorted(manifest.producer.items())))
        for name, split in (("train", "train"), ("valid", "valid"), ("eval", "test")):
            seen[name].extend(manifest.splits[split])
        manifests[str(source)] = hashlib.sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest()
        coverage = source / "coverage.json"
        coverages[str(source)] = (
            hashlib.sha256(coverage.read_bytes()).hexdigest()
            if coverage.exists()
            else None
        )
    if len(producers) != 1:
        raise _E("the sources were made by different producers")
    every = [seed for name in seen for seed in seen[name]]
    if len(every) != len(set(every)):
        raise _E("a seed appears in more than one source or split")
    for name in ("train", "valid", "eval"):
        if sorted(seen[name]) != sorted(expected[name]):
            raise _E(f"the {name} seeds differ from the registered range")
    return {
        "producer": dict(producers.pop()),
        "manifest_sha256": manifests,
        "coverage_sha256": coverages,
    }


def check_producer(identity: dict[str, object], registered: dict[str, object]) -> None:
    """testのsourceのproducerが、登録済み（selectionを作った）producerと全fieldで一致するか。

    ラベルを読む前に呼ぶ。field名の集合が違う場合も、値が違う場合も拒否する。
    """
    producer = identity["producer"]
    differing = sorted(
        name
        for name in set(producer) | set(registered)
        if producer.get(name) != registered.get(name)
    )
    if differing:
        raise _E(
            "the test sources were made by another producer: " + ", ".join(differing)
        )


# --- protocol preset（事前登録して固定する値） ---------------------------------

PURPOSES = ("development-baseline", "formal-test")
_PRESET_FIELDS = (
    "schema",
    "preset_id",
    "version",
    "purpose",
    "train",
    "valid",
    "eval",
    "bootstrap_resamples",
    "bootstrap_seed",
    "jeffreys",
    "clip_epsilon",
    "hold_min_hanchan",
    "hold_min_episodes",
)


def _ranges(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise _E(f"preset {name} must be a list of seed ranges")
    for item in value:
        try:
            seed_range(item)
        except (ValueError, argparse.ArgumentTypeError) as error:
            raise _E(f"preset {name}: bad seed range {item!r}") from error
    return tuple(value)


@dataclass(frozen=True, slots=True)
class Preset:
    """測定の事前登録で固定する条件。呼び出し側が実行時に個別に上書きしない。"""

    preset_id: str
    version: int
    purpose: str
    train: tuple[str, ...]
    valid: tuple[str, ...]
    eval: tuple[str, ...]
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED
    jeffreys: float = JEFFREYS
    clip_epsilon: float = CLIP_EPSILON
    hold_min_hanchan: int = HOLD_MIN_HANCHAN
    hold_min_episodes: int = HOLD_MIN_EPISODES

    def __post_init__(self) -> None:
        if not isinstance(self.preset_id, str) or not self.preset_id:
            raise _E("preset_id must be a non-empty string")
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise _E("preset version must be an integer")
        if self.purpose not in PURPOSES:
            raise _E(f"preset purpose must be one of {PURPOSES}")
        for name in ("train", "valid", "eval"):
            _ranges(getattr(self, name), name)
        if not self.train or not self.eval:
            raise _E("preset needs train and eval seeds")
        every = [
            seed for name in ("train", "valid", "eval") for seed in self.seeds(name)
        ]
        if len(every) != len(set(every)):
            raise _E("preset train / valid / eval seeds overlap")
        for name in ("bootstrap_resamples", "bootstrap_seed", "hold_min_hanchan"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise _E(f"preset {name} must be a positive integer")
        if (
            not isinstance(self.hold_min_episodes, int)
            or isinstance(self.hold_min_episodes, bool)
            or self.hold_min_episodes < 1
        ):
            raise _E("preset hold_min_episodes must be a positive integer")
        if not isinstance(self.jeffreys, (int, float)) or not self.jeffreys > 0:
            raise _E("preset jeffreys must be positive")
        # clip幅は#245の学習側と共通のprotocol invariantで、presetでは変えられない
        if self.clip_epsilon != CLIP_EPSILON:
            raise _E(f"preset clip_epsilon is fixed at {CLIP_EPSILON}")

    def seeds(self, name: str) -> tuple[int, ...]:
        return tuple(seed for item in getattr(self, name) for seed in seed_range(item))

    def expected(self) -> dict[str, tuple[int, ...]]:
        return {name: self.seeds(name) for name in ("train", "valid", "eval")}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PRESET_SCHEMA,
            "preset_id": self.preset_id,
            "version": self.version,
            "purpose": self.purpose,
            "train": list(self.train),
            "valid": list(self.valid),
            "eval": list(self.eval),
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "jeffreys": self.jeffreys,
            "clip_epsilon": self.clip_epsilon,
            "hold_min_hanchan": self.hold_min_hanchan,
            "hold_min_episodes": self.hold_min_episodes,
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_text(self.to_dict()).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "Preset":
        if not isinstance(value, dict) or set(value) != set(_PRESET_FIELDS):
            raise _E("the preset does not have exactly the preset fields")
        if value["schema"] != PRESET_SCHEMA:
            raise _E("not a preset document")
        fields = {name: value[name] for name in _PRESET_FIELDS if name != "schema"}
        for name in ("train", "valid", "eval"):
            fields[name] = _ranges(fields[name], name)
        return cls(**fields)


# #257の事前登録（lisbun/lisjong#257）。この値を変えず、再現の基準とする
PRESET_257 = Preset(
    preset_id="hand-belief-accuracy-257",
    version=1,
    purpose="development-baseline",
    train=("933000..933159",),
    valid=("933160..933239",),
    eval=("933240..933399",),
)
PRESETS = {PRESET_257.preset_id: PRESET_257}


def load_preset(name_or_path: str, sha256: str) -> Preset:
    """組み込みpresetの名前かJSON fileを読み、事前登録のhashと一致するか検査する。"""
    if name_or_path in PRESETS:
        preset = PRESETS[name_or_path]
    else:
        try:
            preset = Preset.from_dict(json.loads(Path(name_or_path).read_text("utf-8")))
        except OSError as error:
            raise _E(f"cannot read the preset {name_or_path!r}") from error
    if preset.sha256() != sha256:
        raise _E("the preset does not match the registered SHA-256")
    return preset


# --- 比べる推定器（差し替え点） -----------------------------------------------


class ScopedWaitEstimator(Protocol):
    """待ち確率の推定器。値を出す行（scope）を自分で宣言する。

    ``scope_seat`` が席を返した判断では、その席の行だけが対象内で、同じ判断の他の席と
    scope外の判断の行は未提供（``None``）として扱う。ゼロ予測へ変換しない。
    ``name`` は結果のkey（``wait.<name>.*`` / ``<name>_scope``）に使う。
    """

    name: str

    def identity(self) -> dict[str, object]: ...

    def scope_seat(self, decision) -> int | None: ...

    def predict(self, policy_input) -> Sequence[float]:
        """34牌種のcanonical順の確率。"""
        ...


class Riichi245Estimator:
    """凍結済みの#245推定器。S1の対象条件のリーチ者の行だけで値を出す。"""

    name = "245"

    def __init__(self, model: LogisticWaitModel, selection_sha256: str) -> None:
        self.model = model
        self.selection_sha256 = selection_sha256

    def identity(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "riichi-wait-logistic-245",
            "selection_sha256": self.selection_sha256,
            "feature_set": self.model.feature_set,
        }

    def scope_seat(self, decision) -> int | None:
        """S1の対象ならリーチ者の席を返す。"""
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
            return riichi_view(decision.policy_input).riichi_seat
        except ValueError:
            return None

    def predict(self, policy_input) -> Sequence[float]:
        predicted = self.model.predict(policy_input)
        return tuple(predicted[tile_type_from_index(i)] for i in range(34))


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
    in_scope: bool
    scoped_wait: tuple[float, ...] | None


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


def build_rows(
    labelled: Sequence[HandBeliefLabelledDecision], estimator: ScopedWaitEstimator
) -> Iterator[Row]:
    by_seed = defaultdict(list)
    for row in labelled:
        by_seed[row.decision.key.seed].append(row)
    for seed in sorted(by_seed):
        instances = kyoku_instances(by_seed[seed])
        for row in by_seed[seed]:
            yield from _rows_of(row, instances[row.decision.key.sequence], estimator)


def _rows_of(
    row: HandBeliefLabelledDecision, instance: int, estimator: ScopedWaitEstimator
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
    scope_seat = estimator.scope_seat(decision)
    scoped_wait = None
    if scope_seat is not None:
        scoped_wait = tuple(estimator.predict(pi))
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
            in_scope=in_scope,
            scoped_wait=scoped_wait if in_scope else None,
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

    def probabilities(self, jeffreys: float = JEFFREYS) -> dict[str, tuple[float, ...]]:
        if not self.weight:
            raise _E("no train row")
        valid = {"wait": frozenset(range(34))} | {
            name: slots for name, _, slots in CHANNELS
        }
        return {
            name: tuple(
                (value + jeffreys) / (self.weight + 2 * jeffreys)
                if index in valid[name]
                else 0.0
                for index, value in enumerate(values)
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


def scope_metrics(
    row: Row, rates: dict[str, tuple[float, ...]], name: str = "245"
) -> dict[str, float]:
    """推定器の対象行だけ: 推定器とbaselineを同じ行で。"""
    metrics: dict[str, float] = {}
    waits = wait_labels(row)
    _probability_metrics(
        metrics,
        f"wait.{name}",
        [clip(p) for p in row.scoped_wait],
        waits,
        row.candidates,
    )
    _probability_metrics(
        metrics, "wait.rate", [clip(p) for p in rates["wait"]], waits, row.candidates
    )
    metrics[f"wait.{name}.predicted_kinds"] = sum(row.scoped_wait)
    metrics["wait.actual_kinds"] = sum(waits)
    for suffix in ("log_loss.all", "log_loss.candidates", "brier.all"):
        metrics[f"wait.{name}_minus_rate.{suffix}"] = (
            metrics[f"wait.{name}.{suffix}"] - metrics[f"wait.rate.{suffix}"]
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
    scope: str = "245"
    support: "Support" = field(init=False)

    def __post_init__(self) -> None:
        self.support = Support(self.scope)

    def add(self, rows: Sequence[Row], rates: dict[str, tuple[float, ...]]) -> None:
        for row, weight in zip(rows, episode_weights(rows)):
            values = row_metrics(row, rates)
            tenpai = "tenpai" if any(wait_labels(row)) else "not_tenpai"
            groups = ["all", f"stratum.{row.stratum}", f"turn.{row.turn}"]
            groups += [f"{group}.{tenpai}" for group in groups[:2]]
            for group in groups:
                self.groups.add(row.seed, row.episode, group, values)
            self.support.add(row)
            self.calibration["wait.rate"].add(
                weight, [clip(p) for p in rates["wait"]], wait_labels(row)
            )
            self.calibration["red_five.uniform"].add(
                weight, [clip(p) for p in row.red_five], red_labels(row)
            )
            if row.in_scope:
                scoped = scope_metrics(row, rates, self.scope)
                self.groups.add(row.seed, row.episode, f"{self.scope}_scope", scoped)
                self.groups.add(
                    row.seed, row.episode, f"{self.scope}_scope.turn.{row.turn}", scoped
                )
        # 推定器の対象行だけで別に重みを付け直す（エピソード1を対象行で等分）
        scoped_rows = [row for row in rows if row.in_scope]
        for row, weight in zip(scoped_rows, episode_weights(scoped_rows)):
            labels = wait_labels(row)
            self.calibration[f"wait.{self.scope}.scope"].add(
                weight, [clip(p) for p in row.scoped_wait], labels
            )
            self.calibration["wait.rate.scope"].add(
                weight, [clip(p) for p in rates["wait"]], labels
            )


COUNT_GROUPS = ("all", *(f"stratum.{stratum}" for stratum in STRATA))
COUNT_TABLES = ("wait", *(name for name, _, _ in CHANNELS))


def _row_counts(scope: str) -> tuple[str, ...]:
    return (
        "rows",
        f"{scope}_scope_rows",
        *(f"{table}.positive_rows" for table in COUNT_TABLES),
    )


def _set_counts(scope: str) -> tuple[str, ...]:
    return (
        "episodes",
        "hanchan",
        f"{scope}_scope_episodes",
        *(
            f"{table}.positive_{unit}"
            for table in COUNT_TABLES
            for unit in ("episodes", "hanchan")
        ),
    )


class Support:
    """正例・対象行の件数（行・エピソード・独立半荘）。splitごとに1つ持つ。"""

    def __init__(self, scope: str = "245") -> None:
        self.scope = scope
        self.sets: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        self.rows: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add(self, row: Row) -> None:
        episode = (row.seed, row.episode)
        tables = {"wait": wait_labels(row)} | {
            name: channel_labels(row, f) for name, f, _ in CHANNELS
        }
        for group in ("all", f"stratum.{row.stratum}"):
            rows, sets = self.rows[group], self.sets[group]
            rows["rows"] += 1
            sets["episodes"].add(episode)
            sets["hanchan"].add(row.seed)
            rows[f"{self.scope}_scope_rows"] += row.in_scope
            if row.in_scope:
                sets[f"{self.scope}_scope_episodes"].add(episode)
            for name, labels in tables.items():
                if any(labels):
                    rows[f"{name}.positive_rows"] += 1
                    sets[f"{name}.positive_episodes"].add(episode)
                    sets[f"{name}.positive_hanchan"].add(row.seed)

    def counts(self) -> dict[str, dict[str, int]]:
        """全体と3層を、行がなくても、正例が0でも、すべての件数を0で明示して返す。"""
        result = {}
        for group in COUNT_GROUPS:
            rows, sets = self.rows.get(group, {}), self.sets.get(group, {})
            values = {name: rows.get(name, 0) for name in _row_counts(self.scope)}
            values |= {
                name: len(sets.get(name, ())) for name in _set_counts(self.scope)
            }
            result[group] = dict(sorted(values.items()))
        return result


def judgement(
    counts: dict[str, int],
    table: str,
    min_hanchan: int = HOLD_MIN_HANCHAN,
    min_episodes: int = HOLD_MIN_EPISODES,
) -> str:
    """希少な正例の扱い（#257計画の6）。閾値はpresetで事前登録する。"""
    episodes = counts[f"{table}.positive_episodes"]
    hanchan = counts[f"{table}.positive_hanchan"]
    if not episodes:
        return "not_estimable"
    if hanchan < min_hanchan or episodes < min_episodes:
        return "held"
    return "reported"


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


def primary_scoped(name: str) -> tuple[str, ...]:
    return (
        f"wait.{name}.log_loss.all",
        "wait.rate.log_loss.all",
        f"wait.{name}_minus_rate.log_loss.all",
    )


def bootstrap(
    per_group: dict[str, dict[int, dict[str, list[float]]]],
    seeds: Sequence[int],
    targets: Sequence[tuple[str, str]],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, dict[str, object]]:
    """eval半荘を復元抽出し、各 (group, 指標) のepisode-macroの95%区間を求める。

    全targetで同じ再標本を使う（paired）。
    """
    random = Random(seed)
    samples = [[random.choice(seeds) for _ in seeds] for _ in range(resamples)]
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


def load_245_estimator(path: Path, sha256: str) -> Riichi245Estimator:
    return Riichi245Estimator(load_245_model(path, sha256), sha256)


def _rows_of_splits(
    source: Path, splits: Sequence[str], estimator: ScopedWaitEstimator
) -> dict[str, list[Row]]:
    manifest, labelled = read_labelled_source(source)
    result = {}
    for split in splits:
        wanted = set(manifest.splits[split])
        chosen = [r for r in labelled if r.decision.key.seed in wanted]
        if {r.decision.key.seed for r in chosen} != wanted:
            raise _E(f"{source}: a {split} seed has no decision")
        result[split] = list(build_rows(chosen, estimator))
    return result


REGISTRATION_FIELDS = (
    "allocation_identity",
    "ledger_revision",
    "arena_revision",
    "evaluator_revision",
)


def check_registration(
    preset: Preset, registration: dict[str, str | None]
) -> dict[str, str | None]:
    """formal-testでは、予約の識別情報を記録することを求める（live ledgerの照合はArena側）。"""
    if set(registration) != set(REGISTRATION_FIELDS):
        raise _E("the registration does not have exactly the registration fields")
    if preset.purpose == "formal-test":
        missing = [name for name, value in registration.items() if not value]
        if missing:
            raise _E("a formal-test needs " + ", ".join(missing))
    return dict(registration)


def evaluate_population(
    sources: Sequence[Path],
    preset: Preset,
    estimator: ScopedWaitEstimator,
    registration: dict[str, str | None] | None = None,
) -> dict[str, object]:
    registration = check_registration(
        preset, registration or dict.fromkeys(REGISTRATION_FIELDS)
    )
    if not (estimator.name.isascii() and estimator.name.isalnum()):
        raise _E("the estimator name must be alphanumeric")
    expected = preset.expected()
    identity = check_population(sources, expected)
    fit = RateFit()
    support = {"train": Support(estimator.name), "valid": Support(estimator.name)}
    for source in sources:
        rows = _rows_of_splits(source, ("train", "valid"), estimator)
        fit.add(rows["train"])
        for name in ("train", "valid"):
            for row in rows[name]:
                support[name].add(row)
    rates = fit.probabilities(preset.jeffreys)
    evaluation = Evaluation(scope=estimator.name)
    for source in sources:
        evaluation.add(_rows_of_splits(source, ("test",), estimator)["test"], rates)
    eval_counts = evaluation.support.counts()
    per_group = evaluation.groups.per_seed()
    seeds = sorted(expected["eval"])
    base_groups = ("all", *(f"stratum.{s}" for s in STRATA))
    targets = (
        [(group, metric) for group in base_groups for metric in PRIMARY]
        + [
            (f"{group}.{tenpai}", "wait.rate.log_loss.all")
            for group in base_groups
            for tenpai in ("tenpai", "not_tenpai")
        ]
        + [(f"{estimator.name}_scope", m) for m in primary_scoped(estimator.name)]
    )
    point = {
        group: {
            metric: episode_macro(per_seed, seeds, metric)
            for metric in sorted({m for s in per_seed.values() for m in s})
        }
        for group, per_seed in sorted(per_group.items())
    }
    return {
        "schema": RESULT_SCHEMA,
        "preset": preset.to_dict() | {"sha256": preset.sha256()},
        "purpose": preset.purpose,
        "estimator": estimator.identity(),
        "registration": registration,
        "population": identity,
        "splits": {name: sorted(seeds) for name, seeds in expected.items()},
        "valid_used_for": "support counts only",
        "rates": {"train_episode_weight": fit.weight, "probabilities": rates},
        "counts": {
            "train": support["train"].counts(),
            "valid": support["valid"].counts(),
            "eval": eval_counts,
        },
        "judgement": {
            group: {
                table: judgement(
                    counts, table, preset.hold_min_hanchan, preset.hold_min_episodes
                )
                for table in COUNT_TABLES
            }
            for group, counts in eval_counts.items()
        },
        "hold_rule": {
            "min_positive_hanchan": preset.hold_min_hanchan,
            "min_positive_episodes": preset.hold_min_episodes,
        },
        "episode_macro": point,
        "intervals": bootstrap(
            per_group,
            seeds,
            targets,
            seed=preset.bootstrap_seed,
            resamples=preset.bootstrap_resamples,
        ),
        "calibration": {
            name: cal.summary() for name, cal in sorted(evaluation.calibration.items())
        },
        "bootstrap": {
            "resamples": preset.bootstrap_resamples,
            "seed": preset.bootstrap_seed,
        },
        "clip_epsilon": CLIP_EPSILON,
    }


# 再現の比較に使う、測定値の部分（実行環境・パス・v2で増えた記録は含めない）
REPRODUCED_KEYS = (
    "splits",
    "valid_used_for",
    "rates",
    "counts",
    "judgement",
    "hold_rule",
    "episode_macro",
    "intervals",
    "calibration",
    "bootstrap",
    "clip_epsilon",
)


def _comparable(document: dict) -> dict[str, object]:
    population = document["population"]
    selection = document.get("selection_245_sha256") or (
        document.get("estimator", {}).get("selection_sha256")
    )
    return {key: document.get(key) for key in REPRODUCED_KEYS} | {
        "producer": population["producer"],
        "manifest_sha256": sorted(population["manifest_sha256"].values()),
        "selection_sha256": selection,
    }


def reproduction_differences(recorded: dict, new: dict) -> list[str]:
    """記録済み結果（v1でもv2でも）と新しい結果の測定値の違うkey。空なら再現している。"""
    left, right = _comparable(recorded), _comparable(new)
    return sorted(key for key in left if left[key] != right[key])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--preset-sha256", required=True)
    parser.add_argument("--selection-245", type=Path, required=True)
    parser.add_argument("--selection-245-sha256", required=True)
    parser.add_argument("--allocation-identity")
    parser.add_argument("--ledger-revision")
    parser.add_argument("--arena-revision")
    parser.add_argument("--evaluator-revision")
    parser.add_argument("--reproduce-of", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("sources", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        parser.error(f"refusing to overwrite {arguments.output}")
    started = time.monotonic()
    preset = load_preset(arguments.preset, arguments.preset_sha256)
    estimator = load_245_estimator(
        arguments.selection_245, arguments.selection_245_sha256
    )
    registration = {
        "allocation_identity": arguments.allocation_identity,
        "ledger_revision": arguments.ledger_revision,
        "arena_revision": arguments.arena_revision,
        "evaluator_revision": arguments.evaluator_revision,
    }
    document = evaluate_population(arguments.sources, preset, estimator, registration)
    if arguments.reproduce_of is not None:
        recorded = json.loads(arguments.reproduce_of.read_text(encoding="utf-8"))
        differing = reproduction_differences(
            recorded, json.loads(canonical_json_text(document))
        )
        document["reproduction"] = {
            "of_sha256": hashlib.sha256(
                arguments.reproduce_of.read_bytes()
            ).hexdigest(),
            "differing": differing,
        }
    document["execution"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wall_seconds": round(time.monotonic() - started, 1),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    arguments.output.write_text(canonical_json_text(document), encoding="utf-8")
    json.dump(document["intervals"], sys.stdout, ensure_ascii=False, indent=1)
    print()
    if document.get("reproduction", {}).get("differing"):
        print(
            "NOT reproduced: " + ", ".join(document["reproduction"]["differing"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
