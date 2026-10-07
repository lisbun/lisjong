"""他家の期待枚数・赤5確率の推定器と条件付き一様baselineの学習・比較（lisbun/lisjong#258）。

学習専用の経路である。#256 v1のsource（``hand_belief_source``、chunkは複数可）を読み、
#257と同じ行・エピソード・集約で比べる。条件は#258の事前登録に従う。

```text
python -m lisjong.learning.hand_count_evaluation select \\
    --train A..B --valid A..B --dev-eval A..B --output SELECTION.json SOURCE [SOURCE ...]
python -m lisjong.learning.hand_count_evaluation test \\
    --test A..B --selection SELECTION.json --selection-sha256 HEX \\
    --output RESULT.json SOURCE [SOURCE ...]
```

1. ``select``: trainでθ（Poisson回帰、offset = 条件付き一様の期待枚数）をL2の格子ごとに求め、
   validの期待枚数の二乗誤差で1つ選ぶ。dev-eval（sourceの``test``分割）は開発用の確認として
   同じ指標を出すが、改善の主張には使わない
2. ``test``: 選択を固定したまま、新しいseedのsource（``test``分割だけ）で1回だけ評価する。
   selectionのseedと重なるseedは拒否する。出力は上書きしない

## 行・重み・集約（#257と同じ）

- 行: 観測者の1判断 × 他家1席（全層）。エピソード: (半荘, 局instance, 他家席)。重み1をその
  エピソードの行で等分する。学習（Poisson回帰）も同じ重みを使う
- 期待枚数は34牌種の二乗誤差の平均、赤5は3色のlog loss平均。行内で平均し、エピソード内で
  行平均し、エピソード間で平均する（episode-macro）。層別（リーチ者 / 副露者 / 門前非リーチ者）も出す
- どちらのモデルも``HandBelief``の固定小数点（raw / 8192）の値で比べる。赤5は
  ``[1e-6, 1 - 1e-6]``にclipする

## 判定（test、2つの出力は別々の主張）

- 期待枚数: ``Δ = 推定器の二乗誤差 - 条件付き一様の二乗誤差``（全行、episode-macro）
- 赤5: ``Δ = 推定器のlog loss - 条件付き一様のlog loss``（全行、episode-macro）

それぞれ、半荘単位のpaired bootstrap（2,000回、seed 258）の95%区間の上端が0未満なら
``pass``、そうでなければ``not_confirmed``。赤5は、testで赤5を持つ行を含む独立半荘が50未満、
または正例エピソードが100未満なら``held``。
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from math import exp, log
from pathlib import Path

from lisjong.belief.canonical_axes import red_five_index, wind_for_seat, wind_index
from lisjong.belief.conditional_uniform_hand_belief import (
    estimate_conditional_uniform_hand_belief,
)
from lisjong.belief.fixed_point import SCALE
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.learning._canonical import canonical_json_text
from lisjong.learning.hand_belief_accuracy import (
    HOLD_MIN_EPISODES,
    HOLD_MIN_HANCHAN,
    RED_CATEGORIES,
    STRATA,
    HandBeliefAccuracyError,
    _Groups,
    _stratum,
    bootstrap,
    check_population,
    episode_macro,
    kyoku_instances,
    seed_range,
)
from lisjong.learning.hand_belief_source import (
    HandBeliefLabelledDecision,
    read_labelled_source,
)
from lisjong.learning.hand_count_estimator import (
    BIAS,
    FEATURE_SET,
    CountContext,
    HandCountModel,
    count_context,
    count_feature_table,
    quantize,
    uniform_expected,
)
from lisjong.learning.riichi_deal_in_evaluation import _solve
from lisjong.policy_contract.seat import Seat

SELECTION_SCHEMA = "lisjong-hand-count-selection-v1"
RESULT_SCHEMA = "lisjong-hand-count-test-result-v1"
L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
BOOTSTRAP_SEED = 258
CLIP_EPSILON = 1e-6
GROUPS = ("all", *(f"stratum.{stratum}" for stratum in STRATA))


class HandCountEvaluationError(HandBeliefAccuracyError):
    """入力が事前登録の条件に合わない。"""


_E = HandCountEvaluationError


# --- 判断の展開 ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Opponent:
    """観測者の1判断 × 他家1席の正解と、その判断の文脈での位置。"""

    seed: int
    episode: tuple[int, int]
    stratum: str
    position: int  # context.opponentsの何番目か
    counts: tuple[int, ...]
    reds: tuple[bool, ...]
    uniform_counts: tuple[float, ...]
    uniform_reds: tuple[float, ...]


def decisions_of_seed(
    labelled: Sequence[HandBeliefLabelledDecision],
) -> Iterator[tuple[CountContext, tuple, list[Opponent]]]:
    """1半荘分の判断を (文脈, 特徴表, 他家の行) へ展開する。"""
    instances = kyoku_instances(labelled)
    for row in labelled:
        pi = row.decision.policy_input
        conservation = derive_remaining_tile_inventory(pi)
        context = count_context(pi, conservation)
        dealer = pi.round.dealer_seat
        slots = [0, 0, 0, 0]
        for view in context.opponents:
            slots[wind_index(wind_for_seat(Seat(view.seat), dealer))] = view.slots
        uniform = estimate_conditional_uniform_hand_belief(pi, tuple(slots))
        positions = {view.seat: i for i, view in enumerate(context.opponents)}
        opponents = []
        for opponent in row.opponents:
            belief = uniform.hand(wind_for_seat(Seat(opponent.seat), dealer))
            truth = opponent.truth
            opponents.append(
                Opponent(
                    seed=row.decision.key.seed,
                    episode=(instances[row.decision.key.sequence], opponent.seat),
                    stratum=_stratum(pi.players[opponent.seat]),
                    position=positions[opponent.seat],
                    counts=tuple(raw // SCALE for raw in truth.expected_count_raw),
                    reds=tuple(
                        truth.red_five_probability_raw[red_five_index(c)] == SCALE
                        for c in RED_CATEGORIES
                    ),
                    uniform_counts=tuple(
                        raw / SCALE for raw in belief.expected_count_raw
                    ),
                    uniform_reds=tuple(
                        raw / SCALE for raw in belief.red_five_probability_raw
                    ),
                )
            )
        yield context, count_feature_table(context), opponents


def episode_weights(opponents: Sequence[Opponent]) -> list[float]:
    counts = defaultdict(int)
    for row in opponents:
        counts[(row.seed, row.episode)] += 1
    return [1.0 / counts[(row.seed, row.episode)] for row in opponents]


def iterate_split(
    sources: Sequence[Path], split: str
) -> Iterator[list[tuple[CountContext, tuple, list[Opponent]]]]:
    """各sourceのmanifestの``split``について、半荘ごとの展開を順に返す。"""
    for source in sources:
        manifest, labelled = read_labelled_source(source)
        wanted = set(manifest.splits[split])
        by_seed = defaultdict(list)
        for row in labelled:
            if row.decision.key.seed in wanted:
                by_seed[row.decision.key.seed].append(row)
        if set(by_seed) != wanted:
            raise _E(f"{source}: a {split} seed has no decision")
        del labelled
        for seed in sorted(by_seed):
            yield list(decisions_of_seed(by_seed.pop(seed)))


def _weighted(decisions) -> Iterator[tuple[CountContext, tuple, Opponent, float]]:
    rows = [row for _, _, opponents in decisions for row in opponents]
    weights = iter(episode_weights(rows))
    for context, table, opponents in decisions:
        for row in opponents:
            yield context, table, row, next(weights)


# --- 当てはめ（Poisson回帰、offset = 条件付き一様） ---------------------------


def add_patterns(patterns: dict, decisions) -> None:
    """特徴の組ごとに (重み × 一様の期待枚数の和, 重み × 実際の枚数の和) を足す。"""
    for context, table, row, weight in _weighted(decisions):
        expected = uniform_expected(context)[row.position]
        for index, features in enumerate(table[row.position]):
            if not context.remaining[index]:
                continue
            cell = patterns.setdefault(features, [0.0, 0.0])
            cell[0] += weight * expected[index]
            cell[1] += weight * row.counts[index]


def fit_poisson(patterns: dict, l2: float) -> tuple[tuple[str, float], ...]:
    """``Σ (U e^{θφ} - y θφ) + l2/2 |θ|^2``（切片は正則化しない）をNewton法で解く。"""
    names = sorted({name for features in patterns for name in features} - {BIAS})
    names = [BIAS, *names]
    index = {name: i for i, name in enumerate(names)}
    keys = [
        (tuple(index[n] for n in features), u, y)
        for features, (u, y) in patterns.items()
    ]
    size = len(names)
    weights = [0.0] * size
    for _ in range(100):
        gradient = [0.0 if i == 0 else l2 * weights[i] for i in range(size)]
        hessian = [
            [(0.0 if i == 0 else l2) if i == j else 0.0 for j in range(size)]
            for i in range(size)
        ]
        for features, u, y in keys:
            mean = u * exp(sum(weights[i] for i in features))
            residual = mean - y
            for i in features:
                gradient[i] += residual
                row = hessian[i]
                for j in features:
                    row[j] += mean
        for i in range(size):
            hessian[i][i] += 1e-9
        step = _solve(hessian, gradient)
        weights = [w - s for w, s in zip(weights, step)]
        if max(abs(s) for s in step) < 1e-10:
            return tuple(zip(names, weights))
    raise ValueError("the Poisson fit did not converge")


# --- 指標 -------------------------------------------------------------------


def _clip(p: float) -> float:
    return min(max(p, CLIP_EPSILON), 1.0 - CLIP_EPSILON)


def _log_loss(p: float, label: bool) -> float:
    p = _clip(p)
    return -log(p if label else 1.0 - p)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def row_metrics(row: Opponent, counts, reds) -> dict[str, float]:
    metrics = {}
    for model, c, r in (
        ("uniform", row.uniform_counts, row.uniform_reds),
        ("estimator", counts, reds),
    ):
        metrics[f"{model}.count_mse"] = _mean(
            (p - y) ** 2 for p, y in zip(c, row.counts)
        )
        metrics[f"{model}.red_log_loss"] = _mean(map(_log_loss, r, row.reds))
        metrics[f"{model}.red_brier"] = _mean(
            (_clip(p) - y) ** 2 for p, y in zip(r, row.reds)
        )
    for metric in ("count_mse", "red_log_loss", "red_brier"):
        metrics[f"delta.{metric}"] = (
            metrics[f"estimator.{metric}"] - metrics[f"uniform.{metric}"]
        )
    return metrics


def estimator_values(
    model: HandCountModel, context: CountContext, table
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """他家ごとの (期待枚数, 赤5確率)。固定小数点の値（raw / 8192）で返す。"""
    result = []
    for counts in model.expected_from(context, table):
        belief = quantize(context, counts)
        result.append(
            (
                tuple(raw / SCALE for raw in belief.expected_count_raw),
                tuple(raw / SCALE for raw in belief.red_five_probability_raw),
            )
        )
    return result


@dataclass
class Report:
    groups: _Groups = field(default_factory=_Groups)
    sets: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    rows: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    seeds: set = field(default_factory=set)

    def add(self, decisions, model: HandCountModel) -> None:
        for context, table, opponents in decisions:
            values = estimator_values(model, context, table)
            for row in opponents:
                counts, reds = values[row.position]
                metrics = row_metrics(row, counts, reds)
                for group in ("all", f"stratum.{row.stratum}"):
                    self.groups.add(row.seed, row.episode, group, metrics)
                    self._count(group, row)
            self.seeds.update(row.seed for row in opponents)

    def _count(self, group: str, row: Opponent) -> None:
        episode = (row.seed, row.episode)
        self.rows[group]["rows"] += 1
        self.sets[group]["episodes"].add(episode)
        self.sets[group]["hanchan"].add(row.seed)
        if any(row.reds):
            self.rows[group]["red.positive_rows"] += 1
            self.sets[group]["red.positive_episodes"].add(episode)
            self.sets[group]["red.positive_hanchan"].add(row.seed)

    def counts(self) -> dict[str, dict[str, int]]:
        return {
            group: {
                "rows": self.rows[group]["rows"],
                "red.positive_rows": self.rows[group]["red.positive_rows"],
                **{
                    name: len(self.sets[group][name])
                    for name in (
                        "episodes",
                        "hanchan",
                        "red.positive_episodes",
                        "red.positive_hanchan",
                    )
                },
            }
            for group in GROUPS
        }

    def document(self, seeds: Sequence[int]) -> dict[str, object]:
        per_group = self.groups.per_seed()
        seeds = sorted(seeds)
        targets = [
            (group, f"{model}.{metric}")
            for group in GROUPS
            for model in ("uniform", "estimator", "delta")
            for metric in ("count_mse", "red_log_loss")
        ]
        counts = self.counts()
        return {
            "counts": counts,
            "red_support": red_support(counts["all"]),
            "episode_macro": {
                group: {
                    metric: episode_macro(per_seed, seeds, metric)
                    for metric in sorted({m for s in per_seed.values() for m in s})
                }
                for group, per_seed in sorted(per_group.items())
            },
            "intervals": bootstrap(per_group, seeds, targets, seed=BOOTSTRAP_SEED),
        }


def red_support(counts: dict[str, int]) -> str:
    if not counts["red.positive_episodes"]:
        return "not_estimable"
    if (
        counts["red.positive_hanchan"] < HOLD_MIN_HANCHAN
        or counts["red.positive_episodes"] < HOLD_MIN_EPISODES
    ):
        return "held"
    return "reported"


def verdicts(document: dict[str, object]) -> dict[str, str]:
    intervals = document["intervals"]

    def judge(metric: str) -> str:
        return (
            "pass"
            if intervals[f"all/delta.{metric}"]["high_97.5"] < 0
            else ("not_confirmed")
        )

    return {
        "expected_count": judge("count_mse"),
        "red_five": judge("red_log_loss")
        if document["red_support"] == "reported"
        else "held",
    }


# --- select / test ------------------------------------------------------------


def select(
    sources: Sequence[Path], expected: dict[str, Sequence[int]]
) -> dict[str, object]:
    identity = check_population(sources, expected)
    patterns: dict = {}
    for decisions in iterate_split(sources, "train"):
        add_patterns(patterns, decisions)
    fits = {l2: fit_poisson(patterns, l2) for l2 in L2_GRID}
    models = {l2: HandCountModel(weights) for l2, weights in fits.items()}
    valid = {l2: _Groups() for l2 in L2_GRID}
    for decisions in iterate_split(sources, "valid"):
        for context, table, opponents in decisions:
            for l2, model in models.items():
                values = estimator_values(model, context, table)
                for row in opponents:
                    counts, _ = values[row.position]
                    valid[l2].add(
                        row.seed,
                        row.episode,
                        "all",
                        {"m": _mean((p - y) ** 2 for p, y in zip(counts, row.counts))},
                    )
    grid = []
    best = None
    for l2 in L2_GRID:
        per_seed = valid[l2].per_seed()["all"]
        loss = episode_macro(per_seed, sorted(per_seed), "m")
        grid.append({"l2": l2, "count_mse": loss})
        if best is None or loss < best[0]:
            best = (loss, l2)
    l2 = best[1]
    report = Report()
    for decisions in iterate_split(sources, "test"):
        report.add(decisions, models[l2])
    return {
        "schema": SELECTION_SCHEMA,
        "feature_set": FEATURE_SET,
        "population": identity,
        "splits": {name: sorted(seeds) for name, seeds in expected.items()},
        "selected": {"l2": l2},
        "l2_grid": list(L2_GRID),
        "valid_grid": grid,
        "pattern_count": len(patterns),
        "model": {"feature_set": FEATURE_SET, "weights": [list(p) for p in fits[l2]]},
        "dev_eval": {
            "note": "development check only; not a claim",
            **report.document(expected["eval"]),
        },
        "bootstrap": {"resamples": 2000, "seed": BOOTSTRAP_SEED},
        "clip_epsilon": CLIP_EPSILON,
    }


def _model_from(selection: dict[str, object]) -> HandCountModel:
    if selection.get("schema") != SELECTION_SCHEMA:
        raise _E("unknown selection schema")
    value = selection["model"]
    if value.get("feature_set") != FEATURE_SET or selection.get("feature_set") != (
        FEATURE_SET
    ):
        raise _E("the selection was made with another feature set")
    return HandCountModel(tuple((n, float(w)) for n, w in value["weights"]))


def run_test(
    sources: Sequence[Path], test_seeds: Sequence[int], selection: dict[str, object]
) -> dict[str, object]:
    model = _model_from(selection)
    used = {seed for seeds in selection["splits"].values() for seed in seeds}
    if used & set(test_seeds):
        raise _E("a test seed was used by the selection")
    identity = check_population(sources, {"train": [], "valid": [], "eval": test_seeds})
    report = Report()
    for decisions in iterate_split(sources, "test"):
        report.add(decisions, model)
    document = report.document(test_seeds)
    return {
        "schema": RESULT_SCHEMA,
        "feature_set": FEATURE_SET,
        "population": identity,
        "test_seeds": sorted(test_seeds),
        "verdicts": verdicts(document),
        **document,
        "hold_rule": {
            "min_positive_hanchan": HOLD_MIN_HANCHAN,
            "min_positive_episodes": HOLD_MIN_EPISODES,
        },
        "bootstrap": {"resamples": 2000, "seed": BOOTSTRAP_SEED},
        "clip_epsilon": CLIP_EPSILON,
    }


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
            "valid_grid": document["valid_grid"],
            "dev_eval": document["dev_eval"]["intervals"],
        }
    else:
        data = arguments.selection.read_bytes()
        if hashlib.sha256(data).hexdigest() != arguments.selection_sha256:
            parser.error("the selection does not match the registered SHA-256")
        document = run_test(arguments.sources, arguments.test, json.loads(data))
        document["selection_sha256"] = arguments.selection_sha256
        summary = {"verdicts": document["verdicts"], "intervals": document["intervals"]}
    arguments.output.write_text(canonical_json_text(document), encoding="utf-8")
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
