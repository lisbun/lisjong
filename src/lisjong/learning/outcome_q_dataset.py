"""L0.3 step D — outcome-Q training preflightとin-memory training set（lisjong-project#79 §4 / §7 / §8）。

strict readしたfocal outcome source（#193 / #195）から、Q trainingが読むeligible row
（DISCARD かつ #191 survivor >= 2）とmodel-facing payloadを決定的に構築する。

```text
FocalOutcomeSource
    -> outcome_q_preflight()   #79 §8 report（hard stopは`failures`）
         population_role SCIENTIFIC、split = {TRAIN, SELECT}、seed leakageなし、
         各splitにeligible rowがある、targetが有限、selectedがsurvivor内、
         TRAIN support: canonical-first / non-canonical-first 各20%以上
    -> build_outcome_q_training_set()
         row順 = source順（game順、game内focal_decision_ordinal順）
         #184 shared context、#189 full candidate tuple encoding、
         behavior-selected candidate index、target_q（build_outcome_targets）
```

full candidate tupleをencodeするのは、#189 encodingのdecision内relative gapを
servingと同じ値にするためである。targetは選択candidateにだけ付く。

training setはsource自体がimmutable / strict-readであるためfileへ再publishせず、
identity（source identity、feature / encoding / label block、row provenance /
target、payload digest）をartifactへbindする。preflightとtraining setの構築は
ML runtimeを必要としない。
"""

import dataclasses
import hashlib
from array import array
from collections import Counter
from dataclasses import dataclass
from math import isfinite

from lisjong.learning._canonical import value_digest
from lisjong.learning._o0 import O0_DECOMPOSITION_IDENTITY
from lisjong.learning.candidate_dataset import _float32_array, _float32_bytes
from lisjong.learning.candidate_encoding import (
    CANDIDATE_ENCODING_DIMENSION,
    encode_candidates,
    encoding_block,
)
from lisjong.learning.dataset import feature_block
from lisjong.learning.envelope_policy import SEMANTIC_ENVELOPE_IDENTITY
from lisjong.learning.errors import TrainingError
from lisjong.learning.features import FEATURE_DIMENSION, build_player_safe_feature
from lisjong.learning.outcome_source import (
    OUTCOME_OBJECTIVE_IDENTITY,
    OUTCOME_TARGET_IDENTITY,
    SCIENTIFIC_ROLE,
    FocalOutcomeSource,
    OutcomeTargetRow,
    build_outcome_targets,
    summarize_outcome_targets,
)

TRAIN_SPLIT = "TRAIN"
SELECT_SPLIT = "SELECT"
OUTCOME_Q_SPLITS = (TRAIN_SPLIT, SELECT_SPLIT)
"""source splitの役割。TRAINでoptimize、SELECTでepochだけを選ぶ。"""

OUTCOME_Q_HIDDEN_WIDTH = 64
"""#79 §5でfreezeしたhidden width（64）。"""

MINIMUM_SURVIVORS = 2
"""eligible rowの#191 survivor数下限（#79 §4）。"""

MINIMUM_SUPPORT_RATE = 0.2
"""TRAIN support sanity（#79 §8）: canonical-first / non-canonical-first各20%以上。"""

PREFLIGHT_PASS = "OUTCOME-Q TRAINING PREFLIGHT PASS"
PREFLIGHT_STOP = "STOP / INVALID"


def outcome_q_label_block() -> dict[str, object]:
    """training set / artifactがbindするrow eligibility / target / objective。"""
    return {
        "eligibility": O0_DECOMPOSITION_IDENTITY,
        "minimum_survivors": MINIMUM_SURVIVORS,
        "objective": OUTCOME_OBJECTIVE_IDENTITY,
        "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
        "target": OUTCOME_TARGET_IDENTITY,
    }


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def _split_source(source: FocalOutcomeSource, split: str) -> FocalOutcomeSource:
    return dataclasses.replace(
        source, games=tuple(game for game in source.games if game.split == split)
    )


def _selection_concentration(rows: tuple[OutcomeTargetRow, ...]) -> dict[str, object]:
    """survivor内の選択位置と、選択tile typeの偏りを返す（診断値）。"""
    positions = Counter(
        row.survivors.index(row.selected_candidate_index) for row in rows
    )
    tile_types = Counter(row.selected_candidate.action.tile.tile_type for row in rows)
    return {
        "selected_survivor_position_distribution": {
            str(position): positions[position] for position in sorted(positions)
        },
        "selected_tile_type_count": len(tile_types),
        "selected_tile_type_max_share": max(tile_types.values()) / len(rows)
        if rows
        else None,
    }


def outcome_q_preflight(source: FocalOutcomeSource) -> dict[str, object]:
    """#79 §8のtraining preflight reportを返す。

    hard stopは`failures`へ列挙し、1件でもあれば`outcome`は`STOP / INVALID`
    になる。report自体はtarget分布を含むため、training前に記録する。
    """
    if not isinstance(source, FocalOutcomeSource):
        raise TrainingError("source must be a strict-read FocalOutcomeSource")
    failures: list[dict[str, object]] = []
    if source.population_role != SCIENTIFIC_ROLE:
        failures.append(
            {
                "check": "population_role",
                "observed": source.population_role,
                "required": SCIENTIFIC_ROLE,
            }
        )
    observed_splits = sorted({game.split for game in source.games})
    if observed_splits != sorted(OUTCOME_Q_SPLITS):
        failures.append(
            {
                "check": "splits",
                "observed": observed_splits,
                "required": sorted(OUTCOME_Q_SPLITS),
            }
        )
    seeds = {
        split: {game.seed for game in source.games if game.split == split}
        for split in OUTCOME_Q_SPLITS
    }
    if seeds[TRAIN_SPLIT] & seeds[SELECT_SPLIT]:
        failures.append({"check": "split_leakage", "observed": "seed overlap"})

    splits: dict[str, object] = {}
    for split in OUTCOME_Q_SPLITS:
        targets = build_outcome_targets(_split_source(source, split))
        rows = targets.rows
        if not targets.hanchan_count or not rows:
            failures.append({"check": "eligible_rows", "split": split, "observed": 0})
            splits[split] = None
            continue
        non_finite = sum(1 for row in rows if not isfinite(row.target_q))
        if non_finite:
            failures.append(
                {"check": "finite_target", "split": split, "observed": non_finite}
            )
        outside = sum(
            1
            for row in rows
            if len(row.survivors) < MINIMUM_SURVIVORS
            or row.selected_candidate_index not in row.survivors
        )
        if outside:
            failures.append(
                {"check": "selected_in_survivors", "split": split, "observed": outside}
            )
        splits[split] = {
            **summarize_outcome_targets(targets),
            **_selection_concentration(rows),
        }

    train = splits.get(TRAIN_SPLIT)
    if train is not None:
        for name in (
            "canonical_first_selected_rate",
            "non_canonical_first_selected_rate",
        ):
            if train[name] < MINIMUM_SUPPORT_RATE:
                failures.append(
                    {
                        "check": f"train_{name}",
                        "observed": train[name],
                        "required_minimum": MINIMUM_SUPPORT_RATE,
                    }
                )

    return {
        "failures": failures,
        "label": outcome_q_label_block(),
        "outcome": PREFLIGHT_STOP if failures else PREFLIGHT_PASS,
        "source": source.provenance(),
        "splits": splits,
        "support_rule": {
            "minimum_canonical_first_rate": MINIMUM_SUPPORT_RATE,
            "minimum_non_canonical_first_rate": MINIMUM_SUPPORT_RATE,
            "split": TRAIN_SPLIT,
        },
    }


def require_outcome_q_preflight(source: FocalOutcomeSource) -> dict[str, object]:
    """preflightを実行し、hard stopがあれば`TrainingError`でfail closedする。"""
    report = outcome_q_preflight(source)
    if report["failures"]:
        raise TrainingError(
            f"outcome-Q training preflight failed: {report['failures']}"
        )
    return report


# ---------------------------------------------------------------------------
# in-memory training set
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutcomeQRow:
    """1 eligible rowのprovenanceと、flat candidate payload上の区間。"""

    split: str
    game_ordinal: int
    seed: int
    kyoku_ordinal: int
    focal_decision_ordinal: int
    candidate_offset: int
    candidate_count: int
    survivors: tuple[int, ...]
    selected_candidate_index: int
    target_q: float

    def to_value(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "focal_decision_ordinal": self.focal_decision_ordinal,
            "game_ordinal": self.game_ordinal,
            "kyoku_ordinal": self.kyoku_ordinal,
            "seed": self.seed,
            "selected_candidate_index": self.selected_candidate_index,
            "split": self.split,
            "survivors": list(self.survivors),
            "target_q": self.target_q,
        }


@dataclass(frozen=True, slots=True)
class OutcomeQTrainingSet:
    """source 1つから決定的に構築したeligible rowとmodel-facing payload。

    `identity`はsource identity、feature / encoding / label block、row
    provenance / target、context / candidate payloadのdigestを束ねる。
    """

    source: FocalOutcomeSource
    identity: str
    rows: tuple[OutcomeQRow, ...]
    context: array
    candidates: array

    @property
    def candidate_count(self) -> int:
        return len(self.candidates) // CANDIDATE_ENCODING_DIMENSION

    def row_indices(self, split: str) -> tuple[int, ...]:
        return tuple(index for index, row in enumerate(self.rows) if row.split == split)

    def split_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(row.split for row in self.rows).items()))

    def unique_kyoku_counts(self) -> dict[str, int]:
        kyokus = {(row.split, row.game_ordinal, row.kyoku_ordinal) for row in self.rows}
        return dict(sorted(Counter(split for split, _, _ in kyokus).items()))

    def block(self) -> dict[str, object]:
        """artifactへbindするtraining set block。"""
        return {
            "candidates": self.candidate_count,
            "identity": self.identity,
            "rows": len(self.rows),
            "splits": self.split_counts(),
            "unique_eligible_kyoku": self.unique_kyoku_counts(),
        }


def build_outcome_q_training_set(source: FocalOutcomeSource) -> OutcomeQTrainingSet:
    """eligible rowをsource順（game順、game内focal_decision_ordinal順）で構築する。"""
    if not isinstance(source, FocalOutcomeSource):
        raise TrainingError("source must be a strict-read FocalOutcomeSource")
    targets = build_outcome_targets(source)
    rows: list[OutcomeQRow] = []
    context_payload = bytearray()
    candidate_payload = bytearray()
    offset = 0
    for row in targets.rows:
        shared = build_player_safe_feature(row.decision.input)
        encoded = encode_candidates(row.candidates)
        if len(shared) != FEATURE_DIMENSION or any(
            not isfinite(value) for value in shared
        ):
            raise TrainingError("shared context materialization is invalid")
        context_payload += _float32_bytes(shared)
        for vector in encoded:
            candidate_payload += _float32_bytes(vector)
        rows.append(
            OutcomeQRow(
                split=row.split,
                game_ordinal=row.game_ordinal,
                seed=row.seed,
                kyoku_ordinal=row.kyoku_ordinal,
                focal_decision_ordinal=row.focal_decision_ordinal,
                candidate_offset=offset,
                candidate_count=len(row.candidates),
                survivors=row.survivors,
                selected_candidate_index=row.selected_candidate_index,
                target_q=row.target_q,
            )
        )
        offset += len(row.candidates)
    if not rows:
        raise TrainingError("outcome source contains no eligible residual rows")
    context_bytes = bytes(context_payload)
    candidate_bytes = bytes(candidate_payload)
    identity = value_digest(
        {
            "candidates_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "context_sha256": hashlib.sha256(context_bytes).hexdigest(),
            "encoding": encoding_block(),
            "feature": feature_block(),
            "label": outcome_q_label_block(),
            "rows": [row.to_value() for row in rows],
            "source_identity": source.identity,
        }
    )
    return OutcomeQTrainingSet(
        source=source,
        identity=identity,
        rows=tuple(rows),
        context=_float32_array(context_bytes),
        candidates=_float32_array(candidate_bytes),
    )


__all__ = [
    "MINIMUM_SUPPORT_RATE",
    "MINIMUM_SURVIVORS",
    "OUTCOME_Q_HIDDEN_WIDTH",
    "OUTCOME_Q_SPLITS",
    "PREFLIGHT_PASS",
    "PREFLIGHT_STOP",
    "SELECT_SPLIT",
    "TRAIN_SPLIT",
    "OutcomeQRow",
    "OutcomeQTrainingSet",
    "build_outcome_q_training_set",
    "outcome_q_label_block",
    "outcome_q_preflight",
    "require_outcome_q_preflight",
]
