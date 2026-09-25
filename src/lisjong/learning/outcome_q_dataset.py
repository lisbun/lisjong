"""L0.3 outcome-Q training rowsとtraining preflight（Issue #200、#79 Step D prep）。

strict readしたSCIENTIFIC focal outcome source（#193 / #195）から、selected-action
Monte-Carlo Q regressionの入力をin-memoryで1回だけmaterializeする。

```text
FocalOutcomeSource（SCIENTIFIC、strict read済み）
    -> build_outcome_targets()          eligibility / target_q / selected candidate
                                        （#193 contractをそのまま使い再定義しない）
    -> rowごとに
         #184 shared context            build_player_safe_feature(PolicyInput)
         #189 full candidate encoding   encode_candidates(row.candidates)
    -> OutcomeQRows（flat float32 payload + row metadata + 決定的identity）
```

published datasetは作らない（#200 DP-3）。入力のsourceはsealed / strict-readの
immutable artifactであり、ここでの変換は決定的である。artifactはsource identityと
本moduleのrow identity（provenanceとpayload digestの合成）をbindする。

SCIENTIFIC以外（DIAGNOSTIC / CALIBRATION）のsourceはfail closedする。splitは
`TRAIN`（optimization）と`SELECT`（checkpoint selection）で固定する。

`outcome_q_preflight()`は#79 §8のpreflight recordとhard stopである。target分布は
TRAINだけから求め、SELECTはcountとsupportだけを記録する（SELECT outcomeを読まない）。
"""

import hashlib
import sys
from array import array
from collections import Counter
from dataclasses import dataclass
from math import isfinite

from lisjong.learning._canonical import value_digest
from lisjong.learning.candidate_encoding import (
    CANDIDATE_ENCODING_DIMENSION,
    encode_candidates,
    encoding_block,
)
from lisjong.learning.dataset import feature_block
from lisjong.learning.envelope_policy import SEMANTIC_ENVELOPE_IDENTITY
from lisjong.learning.errors import DatasetError, TrainingError
from lisjong.learning.features import FEATURE_DIMENSION, build_player_safe_feature
from lisjong.learning.outcome_source import (
    OUTCOME_OBJECTIVE_IDENTITY,
    OUTCOME_TARGET_IDENTITY,
    SCIENTIFIC_ROLE,
    FocalOutcomeSource,
    OutcomeTargetRow,
    OutcomeTargets,
    build_outcome_targets,
    summarize_outcome_targets,
)

TRAIN_SPLIT = "TRAIN"
SELECT_SPLIT = "SELECT"
OUTCOME_Q_SPLITS = (TRAIN_SPLIT, SELECT_SPLIT)

OUTCOME_Q_ROWS_IDENTITY = "lisjong-offense-l0.3-outcome-q-training-rows-v1"
"""row materializationのcontract identity（row identityのdigestへ入る）。"""

MINIMUM_SUPPORT_RATE = 0.2
"""#79 §8: eligible TRAINでcanonical-first / non-canonical-firstがそれぞれ20%以上。"""


@dataclass(frozen=True, slots=True)
class OutcomeQRows:
    """materialize済みのeligible rowとflat payload。

    `context`はR x FEATURE_DIMENSION、`candidates`はC x
    CANDIDATE_ENCODING_DIMENSIONのflat float32である。row `i`のcandidateは
    `candidate_offsets[i]`から`len(rows[i].candidates)`個の連続区間にある。
    """

    identity: str
    source: FocalOutcomeSource
    targets: OutcomeTargets
    context: array
    candidates: array
    candidate_offsets: tuple[int, ...]

    @property
    def rows(self) -> tuple[OutcomeTargetRow, ...]:
        return self.targets.rows

    @property
    def candidate_count(self) -> int:
        return len(self.candidates) // CANDIDATE_ENCODING_DIMENSION

    def row_indices(self, split: str) -> tuple[int, ...]:
        return tuple(index for index, row in enumerate(self.rows) if row.split == split)

    def split_counts(self) -> dict[str, int]:
        counts = Counter(row.split for row in self.rows)
        return {split: counts[split] for split in OUTCOME_Q_SPLITS}

    def hanchan_counts(self) -> dict[str, int]:
        counts = Counter(game.split for game in self.source.games)
        return {split: counts[split] for split in OUTCOME_Q_SPLITS}

    def to_value(self) -> dict[str, object]:
        """artifactへbindするrow block。"""
        return {
            "candidates": self.candidate_count,
            "contract": OUTCOME_Q_ROWS_IDENTITY,
            "eligible_rows": len(self.rows),
            "hanchan": self.hanchan_counts(),
            "identity": self.identity,
            "splits": self.split_counts(),
        }


def _float32_bytes(values: array) -> bytes:
    payload = array("f", values)
    if sys.byteorder != "little":
        payload.byteswap()
    return payload.tobytes()


def materialize_outcome_q_rows(source: FocalOutcomeSource) -> OutcomeQRows:
    """SCIENTIFIC sourceのeligible rowをmaterializeする。

    DIAGNOSTIC / CALIBRATION source、TRAIN / SELECTのeligible row欠損、
    非有限targetはfail closedする。
    """
    if not isinstance(source, FocalOutcomeSource):
        raise DatasetError("source must be a strict-read FocalOutcomeSource")
    if source.population_role != SCIENTIFIC_ROLE:
        raise DatasetError(
            f"outcome-Q training requires a {SCIENTIFIC_ROLE} source; got "
            f"{source.population_role!r}"
        )
    targets = build_outcome_targets(source)
    context = array("f")
    candidates = array("f")
    offsets: list[int] = []
    for row in targets.rows:
        if row.split not in OUTCOME_Q_SPLITS:
            raise DatasetError(f"unexpected split {row.split!r} in a SCIENTIFIC source")
        if not isfinite(row.target_q):
            raise DatasetError("outcome target is not finite")
        features = build_player_safe_feature(row.decision.input)
        if len(features) != FEATURE_DIMENSION:
            raise DatasetError("shared context has an unexpected dimension")
        encoded = encode_candidates(row.candidates)
        if len(encoded) != len(row.candidates) or any(
            len(vector) != CANDIDATE_ENCODING_DIMENSION for vector in encoded
        ):
            raise DatasetError("candidate encoding has an unexpected shape")
        offsets.append(len(candidates) // CANDIDATE_ENCODING_DIMENSION)
        context.extend(features)
        for vector in encoded:
            candidates.extend(vector)

    rows = OutcomeQRows(
        identity="",
        source=source,
        targets=targets,
        context=context,
        candidates=candidates,
        candidate_offsets=tuple(offsets),
    )
    missing = [split for split, count in rows.split_counts().items() if not count]
    if missing:
        raise DatasetError(f"outcome source has no eligible rows in {missing}")

    payload = hashlib.sha256()
    payload.update(_float32_bytes(context))
    payload.update(_float32_bytes(candidates))
    identity = value_digest(
        {
            "contract": OUTCOME_Q_ROWS_IDENTITY,
            "encoding": encoding_block(),
            "feature": feature_block(),
            "objective": OUTCOME_OBJECTIVE_IDENTITY,
            "payload_sha256": payload.hexdigest(),
            "rows": [
                [
                    row.split,
                    row.game_ordinal,
                    row.focal_decision_ordinal,
                    len(row.candidates),
                    list(row.survivors),
                    row.selected_candidate_index,
                    row.target_q,
                ]
                for row in targets.rows
            ],
            "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
            "source": source.identity,
            "target": OUTCOME_TARGET_IDENTITY,
        }
    )
    object.__setattr__(rows, "identity", identity)
    return rows


def _selected_position_distribution(rows) -> dict[str, int]:
    """selected candidateがsurvivor列の何番目か（action-support分布）。"""
    counts = Counter(row.survivors.index(row.selected_candidate_index) for row in rows)
    return {str(position): counts[position] for position in sorted(counts)}


def outcome_q_preflight(rows: OutcomeQRows) -> dict[str, object]:
    """#79 §8のpreflight recordを返す。

    `hard_stops`が空でなければtrainingを開始してはならない。target分布は
    TRAINだけから求め、SELECTはcount / supportだけを記録する。
    """
    if not isinstance(rows, OutcomeQRows):
        raise TrainingError("rows must be materialized OutcomeQRows")
    train = tuple(rows.rows[index] for index in rows.row_indices(TRAIN_SPLIT))
    select = tuple(rows.rows[index] for index in rows.row_indices(SELECT_SPLIT))
    hanchan = rows.hanchan_counts()

    select_survivors = Counter(len(row.survivors) for row in select)
    hard_stops = []
    train_summary = None
    if any(not isfinite(row.target_q) for row in rows.rows):
        # 非有限targetは統計を求める前にstopする（TRAIN summaryは作らない）。
        hard_stops.append("non_finite_target")
    else:
        train_summary = summarize_outcome_targets(
            OutcomeTargets(
                source_identity=rows.source.identity,
                rows=train,
                excluded={},
                hanchan_count=hanchan[TRAIN_SPLIT],
            )
        )
        # 除外件数はsplit横断の値しか持たないため、TRAIN summaryには載せない。
        train_summary.pop("excluded")
        train_summary["selected_survivor_position_distribution"] = (
            _selected_position_distribution(train)
        )
        for name in (
            "canonical_first_selected_rate",
            "non_canonical_first_selected_rate",
        ):
            rate = train_summary[name]
            if rate is None or rate < MINIMUM_SUPPORT_RATE:
                hard_stops.append(f"train_{name}_below_{MINIMUM_SUPPORT_RATE}")

    return {
        "behavior": dict(rows.source.behavior),
        "hard_stops": hard_stops,
        "objective_identity": OUTCOME_OBJECTIVE_IDENTITY,
        "rows_identity": rows.identity,
        "select": {
            "eligible_row_count": len(select),
            "hanchan_count": hanchan[SELECT_SPLIT],
            "survivor_count_distribution": {
                str(count): select_survivors[count]
                for count in sorted(select_survivors)
            },
            "unique_eligible_kyoku_count": len(
                {(row.game_ordinal, row.kyoku_ordinal) for row in select}
            ),
        },
        "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
        "source": rows.source.provenance(),
        "target_identity": OUTCOME_TARGET_IDENTITY,
        "train": train_summary,
    }


__all__ = [
    "MINIMUM_SUPPORT_RATE",
    "OUTCOME_Q_ROWS_IDENTITY",
    "OUTCOME_Q_SPLITS",
    "SELECT_SPLIT",
    "TRAIN_SPLIT",
    "OutcomeQRows",
    "materialize_outcome_q_rows",
    "outcome_q_preflight",
]
