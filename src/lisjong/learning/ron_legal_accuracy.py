"""Offline absolute-accuracy measurement for #262's two fixed rate baselines.

Each source argument is a population directory containing base/ and ron/.
Expected seeds and the full producer identity must be supplied before labels
are read. valid reports support only; eval is the source manifest's test split.

#277 adds optional estimators. Each is scored only on the rows it provides,
paired with both baselines on those same rows, and reported under "estimators";
the baseline sections are computed exactly as without it. --score valid scores
the valid split instead of eval (tuning and diagnostics) and never reads eval.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from math import log
from pathlib import Path
from random import Random

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.fixed_point import SCALE, probability_to_raw
from lisjong.belief.hand_belief import HandBelief
from lisjong.learning._canonical import canonical_json_text
from lisjong.learning.hand_belief_source import read_manifest
from lisjong.learning.riichi_wait_evaluation import (
    _check_selection,
    _models_from_value,
)
from lisjong.learning.ron_legal_baseline import (
    BASELINES,
    CONTEXT_PROTOCOL,
    STRATA,
    RonRateModel,
    StratumRates,
    public_stratum,
)
from lisjong.learning.ron_legal_estimator import (
    OPEN_SCOPE,
    RIICHI_SCOPE,
    TRANSFORM,
    estimate_open_ron_legal_belief,
    estimate_riichi_ron_legal_belief,
)
from lisjong.learning.ron_legal_source import (
    RULES,
    label_ron_source,
    read_ron_source,
)
from lisjong.policy_contract import DiscardAction, PolicyInput, RiichiState, Seat

RESULT_SCHEMA = "lisjong-ron-legal-accuracy-result-v1"
MODEL_SCHEMA = "lisjong-ron-legal-rate-model-v1"
EPSILON = 1e-6
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 262
GROUPS = ("all", *STRATA)
DIAGNOSTICS = (
    "not_tenpai",
    "own_discard_furiten",
    "temporary_furiten",
    "riichi_missed_furiten",
    "no_yaku",
)
METRICS = ("log_loss", "brier", "candidate_log_loss")
# Per-row slot sums that show how far ron truth falls below the wait estimate.
# "zeroed" slots are those where the estimator's ron is below its own wait.
GAP_SUMS = (
    "predicted_wait",
    "predicted_ron",
    "actual_wait",
    "actual_ron",
    "actual_wait_not_zeroed",
    "actual_ron_zeroed",
)


class RonLegalAccuracyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Row:
    seed: int
    episode: tuple[int, str, int]
    policy_input: PolicyInput
    seat: Seat
    wait: tuple[bool, ...]
    ron: tuple[bool, ...]
    candidates: frozenset[int]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        if len(self.wait) != 34 or len(self.ron) != 34:
            raise RonLegalAccuracyError("truth must have 34 slots")
        if any(type(y) is not bool for y in (*self.wait, *self.ron)):
            raise RonLegalAccuracyError("truth must be binary")
        if any(y and not w for y, w in zip(self.ron, self.wait)):
            raise RonLegalAccuracyError("ron truth exceeds wait truth")
        if not self.candidates or not self.candidates <= frozenset(range(34)):
            raise RonLegalAccuracyError("legal discard tile types are required")
        if self.episode[0] != self.seed or self.episode[2] != int(self.seat):
            raise RonLegalAccuracyError("episode does not match row")
        public_stratum(self.policy_input, self.seat)
        if len(set(self.diagnostics)) != len(self.diagnostics) or any(
            name not in DIAGNOSTICS for name in self.diagnostics
        ):
            raise RonLegalAccuracyError("unknown or duplicate diagnostic group")

    @property
    def stratum(self):
        return public_stratum(self.policy_input, self.seat)

    @property
    def groups(self):
        return ("all", self.stratum, *self.diagnostics)


@dataclass(frozen=True, slots=True)
class Estimator:
    """A PolicyInput-only ron estimator; predict returns None when unprovided."""

    name: str
    identity: dict
    predict: Callable[[PolicyInput, Seat], HandBelief | None]

    def __post_init__(self):
        if not self.name.isidentifier() or self.name in BASELINES:
            raise RonLegalAccuracyError("estimator name is invalid or a baseline")


def _registered_selection(selection, sha256):
    data = selection.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RonLegalAccuracyError("wait selection differs from registered SHA-256")
    return json.loads(data)


def _weights_sha256(weights):
    return hashlib.sha256(canonical_json_text(weights).encode()).hexdigest()


def riichi_wait_zero_estimator(selection, sha256):
    """#245's frozen wait model times the certain zeros; identity fails closed."""
    chosen = _registered_selection(selection, sha256)
    _check_selection(chosen)
    model = _models_from_value(chosen["models"])[2]
    return Estimator(
        "riichi_wait_zero",
        {
            "transform": TRANSFORM,
            "scope": RIICHI_SCOPE,
            "selection_sha256": sha256,
            "selection_schema": chosen["schema"],
            "feature_set": model.feature_set,
            "chosen_l2": chosen["chosen_l2"],
            "weights_sha256": _weights_sha256(dict(model.weights)),
        },
        lambda policy_input, seat: estimate_riichi_ron_legal_belief(
            policy_input, seat, model
        ),
    )


def open_wait_zero_estimator(selection, sha256):
    """#259 range 1's selected wait model times the certain zeros."""
    # Imported here: open_wait_evaluation needs the POSIX-only resource module.
    from lisjong.learning import open_wait_evaluation

    chosen = _registered_selection(selection, sha256)
    open_wait_evaluation._check_selection(chosen)
    model, _ = open_wait_evaluation._models_from_value(chosen["models"])
    return Estimator(
        "open_wait_zero",
        {
            "transform": TRANSFORM,
            "scope": OPEN_SCOPE,
            "selection_sha256": sha256,
            "selection_schema": chosen["schema"],
            "feature_set": model.feature_set,
            "selected_l2": chosen["selected"],
            "weights_sha256": _weights_sha256(
                {
                    "tenpai": dict(model.tenpai_weights),
                    "wait": dict(model.wait_weights),
                }
            ),
        },
        lambda policy_input, seat: estimate_open_ron_legal_belief(
            policy_input, seat, model
        ),
    )


SELECTION_ESTIMATORS = {
    "riichi_wait": riichi_wait_zero_estimator,
    "open_wait": open_wait_zero_estimator,
}


def _binary(raw):
    if raw is None or any(v not in (0, SCALE) for v in raw):
        raise RonLegalAccuracyError("exact wait and ron truth are required")
    return tuple(v == SCALE for v in raw)


def rows_from_source(root):
    source = read_ron_source(root / "ron", base_directory=root / "base")
    labelled = label_ron_source(source)
    result = []
    for decision, snapshot, checkpoint in zip(
        labelled, source.snapshots, source.checkpoints
    ):
        public = decision.decision.policy_input
        candidates = frozenset(
            tile_type_index(a.tile.tile_type)
            for a in decision.decision.legal_actions
            if isinstance(a, DiscardAction)
        )
        for opponent in decision.opponents:
            seat = Seat(opponent.seat)
            wait = _binary(opponent.truth.wait_probability_raw)
            ron = _binary(opponent.truth.ron_legal_probability_raw)
            river = {
                tile_type_index(d.tile.tile_type) for d in public.players[seat].discards
            }
            discarded_wait = any(w and i in river for i, w in enumerate(wait))
            missed = checkpoint.contexts[seat].missed_ron_state.value
            diagnoses = []
            if not any(wait):
                diagnoses.append("not_tenpai")
            if discarded_wait:
                diagnoses.append("own_discard_furiten")
            if missed != "none":
                diagnoses.append(
                    "temporary_furiten"
                    if missed == "temporary"
                    else "riichi_missed_furiten"
                )
            if (
                not discarded_wait
                and missed == "none"
                and any(w and not y for w, y in zip(wait, ron))
            ):
                diagnoses.append("no_yaku")
            result.append(
                Row(
                    decision.decision.key.seed,
                    (decision.decision.key.seed, snapshot.round_id, int(seat)),
                    public,
                    seat,
                    wait,
                    ron,
                    candidates,
                    tuple(diagnoses),
                )
            )
    return source.base_manifest, result


class RateFit:
    def __init__(self):
        self.weight = Counter()
        self.wait = {s: [0.0] * 34 for s in STRATA}
        self.ron = {s: [0.0] * 34 for s in STRATA}

    def add(self, rows):
        # Each episode is contained in exactly one source (checked before fit).
        counts = Counter((r.stratum, r.episode) for r in rows)
        for row in rows:
            stratum = row.stratum
            weight = 1 / counts[stratum, row.episode]
            self.weight[stratum] += weight
            for i in range(34):
                self.wait[stratum][i] += weight * row.wait[i]
                self.ron[stratum][i] += weight * row.ron[i]

    def model(self):
        return RonRateModel(
            tuple(
                StratumRates(
                    tuple(
                        probability_to_raw((v + 0.5) / (self.weight[s] + 1))
                        for v in self.wait[s]
                    ),
                    tuple(
                        probability_to_raw((v + 0.5) / (self.weight[s] + 1))
                        for v in self.ron[s]
                    ),
                )
                if self.weight[s]
                else None
                for s in STRATA
            )
        )


def model_value(model):
    return {
        "schema": MODEL_SCHEMA,
        "context_protocol": model.context_protocol,
        "strata": {
            s: None
            if t is None
            else {"wait_raw": list(t.wait_raw), "ron_raw": list(t.ron_raw)}
            for s, t in zip(STRATA, model.tables)
        },
        "smoothing": "Jeffreys-0.5",
        "fit_weight": "one-per-stratum-episode",
    }


class Support:
    def __init__(self):
        self.rows = Counter()
        self.positive_rows = Counter()
        self.positive_slots = Counter()
        self.declared = Counter()
        self.episodes = defaultdict(set)
        self.seeds = defaultdict(set)
        self.positive_episodes = defaultdict(set)
        self.positive_seeds = defaultdict(set)

    def add(self, row):
        for group in row.groups:
            self.rows[group] += 1
            self.positive_rows[group] += any(row.ron)
            self.positive_slots[group] += sum(row.ron)
            self.declared[group] += (
                row.policy_input.players[row.seat].riichi is RiichiState.DECLARED
            )
            self.episodes[group].add(row.episode)
            self.seeds[group].add(row.seed)
            if any(row.ron):
                self.positive_episodes[group].add(row.episode)
                self.positive_seeds[group].add(row.seed)

    def value(self):
        result = {}
        for group in (*GROUPS, *DIAGNOSTICS):
            positive_episodes = len(self.positive_episodes[group])
            positive_hanchan = len(self.positive_seeds[group])
            result[group] = {
                "rows": self.rows[group],
                "episodes": len(self.episodes[group]),
                "hanchan": len(self.seeds[group]),
                "positive_rows": self.positive_rows[group],
                "positive_slots": self.positive_slots[group],
                "positive_episodes": positive_episodes,
                "positive_hanchan": positive_hanchan,
                "declared_rows": self.declared[group],
                "support_status": "not_estimable"
                if not positive_episodes
                else "held"
                if positive_hanchan < 50 or positive_episodes < 100
                else "reported",
            }
        return result


def metrics(raw, labels, candidates):
    p = tuple(v / SCALE for v in raw)
    losses = tuple(
        -log(
            min(max(q, EPSILON), 1 - EPSILON)
            if y
            else 1 - min(max(q, EPSILON), 1 - EPSILON)
        )
        for q, y in zip(p, labels)
    )
    return {
        "log_loss": sum(losses) / 34,
        "brier": sum((q - y) ** 2 for q, y in zip(p, labels)) / 34,
        "candidate_log_loss": sum(losses[i] for i in candidates) / len(candidates),
    }


def _calibrate(table, raw, labels):
    for p, y in zip(raw, labels):
        probability = p / SCALE
        bucket = table[min(int(probability * 10), 9)]
        bucket[0] += 1
        bucket[1] += probability
        bucket[2] += y


def gap_sums(prediction, row):
    wait, ron = prediction.wait_probability_raw, prediction.ron_legal_probability_raw
    kept = tuple(r >= w for r, w in zip(ron, wait))
    return {
        "predicted_wait": sum(wait) / SCALE,
        "predicted_ron": sum(ron) / SCALE,
        "actual_wait": sum(row.wait),
        "actual_ron": sum(row.ron),
        "actual_wait_not_zeroed": sum(w and k for w, k in zip(row.wait, kept)),
        "actual_ron_zeroed": sum(y and not k for y, k in zip(row.ron, kept)),
    }


class Evaluation:
    def __init__(self, estimators=()):
        self.cells = {}
        self.coverage = {g: Counter() for g in (*GROUPS, *DIAGNOSTICS)}
        self.estimators = tuple(estimators)
        if len({e.name for e in self.estimators}) != len(self.estimators):
            raise RonLegalAccuracyError("duplicate estimator name")
        self.estimator_cells = {e.name: {} for e in self.estimators}
        self.estimator_coverage = {
            e.name: {g: Counter() for g in (*GROUPS, *DIAGNOSTICS)}
            for e in self.estimators
        }

    def add(self, row, model):
        predictions = {
            b: model.predict(row.policy_input, row.seat, b) for b in BASELINES
        }
        available = all(p is not None for p in predictions.values())
        for estimator in self.estimators:
            self._add_estimator(estimator, row, predictions, available)
        for group in row.groups:
            self.coverage[group]["target_rows"] += 1
            self.coverage[group][
                "provided_rows" if available else "unprovided_rows"
            ] += 1
            if not available:
                continue
            cell = self.cells.setdefault(
                (group, row.episode),
                {
                    "rows": 0,
                    "metrics": Counter(),
                    "calibration": {
                        b: [[0, 0.0, 0] for _ in range(10)] for b in BASELINES
                    },
                },
            )
            cell["rows"] += 1
            values = {}
            for baseline, prediction in predictions.items():
                raw = prediction.ron_legal_probability_raw
                values[baseline] = metrics(raw, row.ron, row.candidates)
                for name, value in values[baseline].items():
                    cell["metrics"][f"{baseline}.{name}"] += value
                _calibrate(cell["calibration"][baseline], raw, row.ron)
            cell["metrics"]["wait_genbutsu_minus_ron_rate.log_loss"] += (
                values["wait_genbutsu"]["log_loss"] - values["ron_rate"]["log_loss"]
            )

    def _add_estimator(self, estimator, row, baselines, available):
        name = estimator.name
        prediction = estimator.predict(row.policy_input, row.seat)
        if prediction is not None:
            # An unprovided row stays unprovided; a malformed one is an error.
            if (
                not isinstance(prediction, HandBelief)
                or prediction.ron_legal_probability_raw is None
            ):
                raise RonLegalAccuracyError("estimator must return paired wait and ron")
            if not available:
                raise RonLegalAccuracyError("estimator row has no paired baseline")
            values = {
                name: metrics(
                    prediction.ron_legal_probability_raw, row.ron, row.candidates
                )
            } | {
                b: metrics(p.ron_legal_probability_raw, row.ron, row.candidates)
                for b, p in baselines.items()
            }
            gaps = gap_sums(prediction, row)
        for group in row.groups:
            coverage = self.estimator_coverage[name][group]
            coverage["target_rows"] += 1
            coverage["unprovided_rows" if prediction is None else "provided_rows"] += 1
            if prediction is None:
                continue
            cell = self.estimator_cells[name].setdefault(
                (group, row.episode),
                {
                    "rows": 0,
                    "metrics": Counter(),
                    "calibration": {name: [[0, 0.0, 0] for _ in range(10)]},
                },
            )
            cell["rows"] += 1
            for metric in METRICS:
                for model, value in values.items():
                    cell["metrics"][f"{model}.{metric}"] += value[metric]
                for baseline in BASELINES:
                    cell["metrics"][f"{name}_minus_{baseline}.{metric}"] += (
                        values[name][metric] - values[baseline][metric]
                    )
            for key, value in gaps.items():
                cell["metrics"][f"{name}.{key}"] += value
            _calibrate(
                cell["calibration"][name], prediction.ron_legal_probability_raw, row.ron
            )

    def per_seed(self):
        return _per_seed(self.cells)

    def calibration(self):
        return _calibration(self.cells, BASELINES)

    def estimator_value(self, estimator, seeds):
        name = estimator.name
        cells = self.estimator_cells[name]
        targets = tuple(
            f"{model}.{metric}"
            for metric in METRICS
            for model in (
                name,
                *BASELINES,
                *(f"{name}_minus_{b}" for b in BASELINES),
            )
        ) + tuple(f"{name}.{key}" for key in GAP_SUMS)
        return {
            "identity": estimator.identity,
            "coverage": {
                g: {
                    k: self.estimator_coverage[name][g][k]
                    for k in ("target_rows", "provided_rows", "unprovided_rows")
                }
                for g in self.estimator_coverage[name]
            },
            "metrics": _intervals(_per_seed(cells), seeds, targets),
            "calibration": _calibration(cells, (name,)),
        }


def _per_seed(cells):
    result = defaultdict(lambda: defaultdict(dict))
    for (group, episode), cell in cells.items():
        for metric, total in cell["metrics"].items():
            pair = result[group][episode[0]].setdefault(metric, [0.0, 0])
            pair[0] += total / cell["rows"]
            pair[1] += 1
    return result


def _calibration(cells, names):
    bins = {
        g: {b: [[0, 0.0, 0.0, 0.0] for _ in range(10)] for b in names}
        for g in (*GROUPS, *DIAGNOSTICS)
    }
    for (group, _), cell in cells.items():
        denominator = cell["rows"] * 34
        for baseline in names:
            for target, (slots, predicted, observed) in zip(
                bins[group][baseline], cell["calibration"][baseline]
            ):
                target[0] += slots
                target[1] += slots / denominator
                target[2] += predicted / denominator
                target[3] += observed / denominator
    return {
        g: {
            b: [
                {
                    "bin": i,
                    "slots": slots,
                    "episode_weight": weight,
                    "mean_probability": predicted / weight if weight else None,
                    "observed_rate": observed / weight if weight else None,
                }
                for i, (slots, weight, predicted, observed) in enumerate(table)
            ]
            for b, table in baselines.items()
        }
        for g, baselines in bins.items()
    }


def episode_mean(per_seed, seeds, metric):
    values = [per_seed.get(seed, {}).get(metric, (0, 0)) for seed in seeds]
    count = sum(v[1] for v in values)
    return sum(v[0] for v in values) / count if count else None


def confidence_intervals(evaluation, seeds):
    return _intervals(
        evaluation.per_seed(),
        seeds,
        (
            "ron_rate.log_loss",
            "wait_genbutsu.log_loss",
            "wait_genbutsu_minus_ron_rate.log_loss",
            "ron_rate.brier",
            "wait_genbutsu.brier",
            "ron_rate.candidate_log_loss",
            "wait_genbutsu.candidate_log_loss",
        ),
    )


def _intervals(per_group, seeds, targets):
    # The same RNG seed gives every model the same half-game draws (paired).
    random = Random(BOOTSTRAP_SEED)
    draws = [[random.choice(seeds) for _ in seeds] for _ in range(BOOTSTRAP_RESAMPLES)]
    result = {}
    for group in (*GROUPS, *DIAGNOSTICS):
        per_seed = per_group.get(group, {})
        result[group] = {}
        for metric in targets:
            samples = sorted(
                v
                for draw in draws
                if (v := episode_mean(per_seed, draw, metric)) is not None
            )
            result[group][metric] = {
                "point": episode_mean(per_seed, seeds, metric),
                "low_2.5": samples[int(0.025 * len(samples))] if samples else None,
                "high_97.5": samples[int(0.975 * len(samples)) - 1]
                if samples
                else None,
                "resamples_with_rows": len(samples),
            }
    return result


def check_population(roots, expected, producer):
    if not roots or set(expected) != {"train", "valid", "eval"}:
        raise RonLegalAccuracyError(
            "sources and exact train/valid/eval seeds are required"
        )
    every = [seed for seeds in expected.values() for seed in seeds]
    if len(every) != len(set(every)) or any(type(s) is not int or s < 0 for s in every):
        raise RonLegalAccuracyError("registered seeds must be nonnegative and disjoint")
    if set(producer) != {
        "arena_revision",
        "lisjong_revision",
        "lisjong_engine_revision",
        "policy",
    }:
        raise RonLegalAccuracyError("full producer identity is required")
    seen = defaultdict(list)
    identity = {}
    for root in roots:
        manifest = read_manifest(root / "base")
        if manifest.producer != producer:
            raise RonLegalAccuracyError(
                "source producer differs from registered producer"
            )
        extension = json.loads((root / "ron/manifest.json").read_text())
        if (
            extension["context_protocol"] != CONTEXT_PROTOCOL
            or extension["rules"] != RULES
        ):
            raise RonLegalAccuracyError(
                "source context/rules differ from fixed protocol"
            )
        identity[str(root)] = {
            "base_manifest_sha256": hashlib.sha256(
                (root / "base/manifest.json").read_bytes()
            ).hexdigest(),
            "ron_manifest_sha256": hashlib.sha256(
                (root / "ron/manifest.json").read_bytes()
            ).hexdigest(),
        }
        for name, split in (("train", "train"), ("valid", "valid"), ("eval", "test")):
            seen[name].extend(manifest.splits[split])
    actual = [s for seeds in seen.values() for s in seeds]
    if len(actual) != len(set(actual)) or any(
        sorted(seen[n]) != sorted(expected[n]) for n in expected
    ):
        raise RonLegalAccuracyError(
            "missing, duplicate, or incorrectly split source seeds"
        )
    return identity


def backend_identity(producer):
    from lisjong.belief.ron_legal_ground_truth import require_scoring_backend

    require_scoring_backend()
    import _lisjong_native

    if _lisjong_native.SOURCE_REVISION != producer["lisjong_revision"]:
        raise RonLegalAccuracyError(
            "native source revision differs from registered producer"
        )
    return {
        "native_source_revision": _lisjong_native.SOURCE_REVISION,
        "native_api_version": _lisjong_native.API_VERSION,
        "scoring_api_version": _lisjong_native.SCORING_API_VERSION,
    }


def evaluate_population(
    roots, expected, producer, *, support_only=False, estimators=(), score="eval"
):
    if score not in ("eval", "valid") or (
        support_only and (estimators or score != "eval")
    ):
        raise RonLegalAccuracyError("unknown score split or support-only scoring")
    scored = "test" if score == "eval" else "valid"
    identity = check_population(roots, expected, producer)
    backend = backend_identity(producer)
    if (
        not expected["train"]
        or not expected["valid"]
        or (not support_only and not expected["eval"])
    ):
        raise RonLegalAccuracyError(
            "train, valid and eval are required for measurement"
        )
    fit = RateFit()
    supports = {name: Support() for name in expected}
    # Materialize only one chunk at a time. No labels escape into inference.
    for root in roots:
        manifest = read_manifest(root / "base")
        if not (manifest.splits["train"] or manifest.splits["valid"] or support_only):
            continue
        manifest, rows = rows_from_source(root)
        for row in rows:
            split = {"test": "eval"}.get(
                manifest.split_of(row.seed), manifest.split_of(row.seed)
            )
            if split != "eval" or support_only:
                supports[split].add(row)
        fit.add([r for r in rows if manifest.split_of(r.seed) == "train"])
    model = fit.model()
    evaluation = Evaluation(estimators)
    if not support_only:
        for root in roots:
            if not read_manifest(root / "base").splits[scored]:
                continue
            manifest, rows = rows_from_source(root)
            for row in rows:
                if manifest.split_of(row.seed) == scored:
                    if score == "eval":
                        supports["eval"].add(row)
                    evaluation.add(row, model)
    model_json = json.dumps(model_value(model), sort_keys=True, separators=(",", ":"))
    extension = {}
    if estimators or score != "eval":
        # Absent on the #262 path, so its result document keeps the same keys.
        extension = {
            "scored_split": score,
            "estimators": {
                e.name: evaluation.estimator_value(e, expected[score])
                for e in estimators
            },
        }
    return extension | {
        "schema": RESULT_SCHEMA,
        "mode": "support_only" if support_only else "absolute_accuracy",
        "context_protocol": CONTEXT_PROTOCOL,
        "rules": RULES,
        "producer": producer,
        "sources": identity,
        "seeds": expected,
        "model": model_value(model),
        "model_sha256": hashlib.sha256(model_json.encode()).hexdigest(),
        "support": {name: value.value() for name, value in supports.items()},
        "coverage": {
            g: {
                k: evaluation.coverage[g][k]
                for k in ("target_rows", "provided_rows", "unprovided_rows")
            }
            for g in evaluation.coverage
        },
        "metrics": {}
        if support_only
        else confidence_intervals(evaluation, expected[score]),
        "calibration": {} if support_only else evaluation.calibration(),
        "protocol": {
            "clip_epsilon": EPSILON,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "aggregation": "slot-mean/row-mean/episode-mean",
            "diagnostic_groups_overlap": True,
            "valid_used_for_selection": False,
        },
        "runtime": sys.version,
        "backend": backend,
    }


def seed_range(value):
    if value == "none":
        return []
    pieces = value.split("..")
    if len(pieces) not in (1, 2) or any(not p.isdecimal() for p in pieces):
        raise argparse.ArgumentTypeError(
            "expected a nonnegative seed or FIRST..LAST or none"
        )
    first, last = int(pieces[0]), int(pieces[-1])
    if last < first:
        raise argparse.ArgumentTypeError("seed range is reversed")
    return list(range(first, last + 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "valid", "eval"):
        parser.add_argument(f"--{split}", type=seed_range, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--support-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score", choices=("eval", "valid"), default="eval")
    for kind in SELECTION_ESTIMATORS:
        parser.add_argument(f"--{kind.replace('_', '-')}-selection", type=Path)
        parser.add_argument(f"--{kind.replace('_', '-')}-selection-sha256")
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise RonLegalAccuracyError("refusing to overwrite a measurement result")
    estimators = []
    for kind, build in SELECTION_ESTIMATORS.items():
        selection = getattr(args, f"{kind}_selection")
        sha256 = getattr(args, f"{kind}_selection_sha256")
        if (selection is None) != (sha256 is None):
            raise RonLegalAccuracyError("a wait selection needs its registered SHA-256")
        if selection is not None:
            estimators.append(build(selection, sha256))
    expected = {name: getattr(args, name) for name in ("train", "valid", "eval")}
    result = evaluate_population(
        args.sources,
        expected,
        json.loads(args.producer.read_text()),
        support_only=args.support_only,
        estimators=estimators,
        score=args.score,
    )
    # Result appears only after complete validation and measurement.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
