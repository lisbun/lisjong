"""L0.2 candidate scorer datasetのmaterializationとstrict read。

Issue #189に対応する。player-safe source record（schema v2）から、O0
precedence上のnormal-discard decisionだけをcandidate-centric datasetへ
materializeする。Issue #184のflat BC dataset（`DATASET_SCHEMA`）とは別の
purpose-specific contractであり、既存datasetを変更しない。

```text
player-safe source record
    -> O0 precedence（WIN / RIICHI / RESPONSEは除外してcountだけ残す）
    -> DISCARD decisionごとに
         #184 shared context（build_player_safe_feature）
         two-pass finalist candidate features（#187）+ numeric encoding
         source teacher_selected_actionのcanonical candidate index
    -> immutable dataset directory
```

1 dataset = 1 immutable directoryである。

```text
<dataset>/
    manifest.json      canonical JSON identity / provenance / payload digest
    decisions.jsonl    1行 = 1 scorer decisionのprovenance / label / candidate semantic
    context.f32        D x FEATURE_DIMENSION little-endian float32 (row-major)
    candidates.f32     C x CANDIDATE_ENCODING_DIMENSION little-endian float32
```

candidate rowはragged / flattenedである。`decisions.jsonl`の各行が
`candidate_offset` / `candidate_count`でcandidates.f32上の連続区間を指し、
paddingを持たない。

## label source

teacher labelはsource recordに記録済みの`teacher_selected_action`だけである。
現在の`TwoStepUkeirePolicy`を再実行してrelabelしない。scorer対象decisionで
teacher actionがlegal `DiscardAction` candidateへ一意に解決できなければfail
closedする。

## strict read

readbackはmanifest identity、bindされたshared feature / candidate feature /
encoding / request policy / label identity、payloadのbyte長とdigest、row
provenanceとsource population、split別count、candidate区間の連続性に加え、
`decisions.jsonl`に保持したtyped candidate semanticから`encode_candidates()`
で再encodeした値がcandidates.f32と一致することを照合する。
"""

import sys
from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from lisjong.learning._canonical import (
    canonical_json_line,
    canonical_json_text,
    expect_digest,
    expect_list,
    expect_non_negative_int,
    expect_object,
    expect_str,
    file_digest,
    parse_json_text,
    seal,
    unseal,
)
from lisjong.learning._o0 import (
    O0_DECOMPOSITION_IDENTITY,
    O0DecisionKind,
    classify_o0_decision,
)
from lisjong.learning._publication import (
    staged_publication,
    write_new_bytes,
    write_new_text,
)
from lisjong.learning._typed_values import action_to_value, parse_action
from lisjong.learning.candidate_encoding import (
    CANDIDATE_ENCODING_DIMENSION,
    build_scorer_candidates,
    encode_candidates,
    encoding_block,
)
from lisjong.learning.candidate_features import (
    DiscardCandidateFeatures,
    SecondStepStatus,
)
from lisjong.learning.dataset import feature_block, validate_source_block
from lisjong.learning.errors import (
    CandidateFeatureError,
    DatasetError,
)
from lisjong.learning.features import FEATURE_DIMENSION, build_player_safe_feature
from lisjong.learning.source_record import (
    SOURCE_RECORD_SCHEMA_V2,
    PlayerSafeSourceRecord,
)
from lisjong.policy_contract import DecisionContext, DiscardAction, Seat

CANDIDATE_DATASET_SCHEMA = "lisjong-offense-l0.2-candidate-scorer-dataset-v1"
"""このcandidate dataset contractのidentity。"""

CANDIDATE_DATASET_KIND = "candidate-scorer-dataset"

CANDIDATE_LABEL_SEMANTICS = "lisjong-offense-l0.2-teacher-selected-discard-candidate-v1"
"""label semanticsのidentity。

1 decisionのlabelは、source recordの`teacher_selected_action`がcanonical
candidate順の何番目のlegal `DiscardAction`かを表すindexである。relabel、
soft target、reward、privileged truthは含まない。
"""

CANDIDATE_TRAINING_OBJECTIVE = "per-decision-legal-discard-softmax-cross-entropy"

MANIFEST_FILENAME = "manifest.json"
DECISIONS_FILENAME = "decisions.jsonl"
CONTEXT_FILENAME = "context.f32"
CANDIDATES_FILENAME = "candidates.f32"

_FLOAT32_BYTES = 4
_ROWS_DTYPE = "canonical-json-line"
_FLOAT_DTYPE = "float32-le"

_MANIFEST_FIELDS = frozenset(
    {
        "dataset_schema",
        "encoding",
        "feature",
        "files",
        "kind",
        "label",
        "rows",
        "source",
    }
)
_LABEL_FIELDS = frozenset({"eligibility", "objective", "semantics"})
_ROWS_FIELDS = frozenset(
    {"candidates", "scorer_decisions", "source_decisions", "splits"}
)
_SPLIT_COUNT_FIELDS = frozenset(
    {
        "candidates",
        "excluded_response",
        "excluded_riichi",
        "excluded_win",
        "scorer_decisions",
        "source_decisions",
    }
)
_PAYLOAD_FIELDS = frozenset({"bytes", "columns", "dtype", "rows", "sha256"})
_DECISION_FIELDS = frozenset(
    {
        "actor_seat",
        "candidate_count",
        "candidate_offset",
        "candidates",
        "decision_ordinal",
        "game_ordinal",
        "seed",
        "split",
        "step_ordinal",
        "teacher_candidate_index",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "action",
        "current_ukeire_count",
        "post_discard_shanten",
        "second_step_status",
        "second_step_ukeire_score",
    }
)
_EXCLUSION_FIELD = {
    O0DecisionKind.WIN: "excluded_win",
    O0DecisionKind.RIICHI: "excluded_riichi",
    O0DecisionKind.RESPONSE: "excluded_response",
}


@dataclass(frozen=True, slots=True)
class CandidateDecisionRow:
    """1 scorer decisionのprovenance、label、typed candidate semantic。"""

    game_ordinal: int
    seed: int
    split: str
    step_ordinal: int
    decision_ordinal: int
    actor_seat: Seat
    candidate_offset: int
    teacher_candidate_index: int
    candidates: tuple[DiscardCandidateFeatures, ...]

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def teacher_candidate(self) -> DiscardCandidateFeatures:
        return self.candidates[self.teacher_candidate_index]


@dataclass(frozen=True, slots=True)
class CandidateDataset:
    """strict readしたcandidate datasetと、そのbindされたidentity。"""

    identity: str
    manifest: Mapping[str, object]
    rows: tuple[CandidateDecisionRow, ...]
    context: array
    candidates: array

    @property
    def decision_count(self) -> int:
        return len(self.rows)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates) // CANDIDATE_ENCODING_DIMENSION

    def row_indices_for_splits(self, splits: tuple[str, ...]) -> tuple[int, ...]:
        """指定splitに属するdecision indexを、dataset順のまま返す。"""
        selected = set(splits)
        return tuple(
            index for index, row in enumerate(self.rows) if row.split in selected
        )

    def split_counts(self) -> dict[str, int]:
        """splitごとのscorer decision数を返す。"""
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.split] = counts.get(row.split, 0) + 1
        return counts


def _float32_bytes(values) -> bytes:
    payload = array("f", values)
    if sys.byteorder != "little":
        payload.byteswap()
    return payload.tobytes()


def _float32_array(data: bytes) -> array:
    payload = array("f")
    payload.frombytes(data)
    if sys.byteorder != "little":
        payload.byteswap()
    return payload


def candidate_label_block() -> dict[str, object]:
    """teacher / label semantics blockを返す。"""
    return {
        "eligibility": O0_DECOMPOSITION_IDENTITY,
        "objective": CANDIDATE_TRAINING_OBJECTIVE,
        "semantics": CANDIDATE_LABEL_SEMANTICS,
    }


def _candidate_to_value(candidate: DiscardCandidateFeatures) -> dict[str, object]:
    return {
        "action": action_to_value(candidate.action, DatasetError, "candidate.action"),
        "current_ukeire_count": candidate.current_ukeire_count,
        "post_discard_shanten": candidate.post_discard_shanten,
        "second_step_status": candidate.second_step_status.value,
        "second_step_ukeire_score": candidate.second_step_ukeire_score,
    }


def _parse_candidate(value: object, context: str) -> DiscardCandidateFeatures:
    item = expect_object(value, _CANDIDATE_FIELDS, DatasetError, context)
    action = parse_action(item["action"], DatasetError, f"{context}.action")
    if not isinstance(action, DiscardAction):
        raise DatasetError(f"{context}.action must be a DiscardAction")
    for field in ("current_ukeire_count", "post_discard_shanten"):
        expect_non_negative_int(item[field], DatasetError, f"{context}.{field}")
    try:
        status = SecondStepStatus(item["second_step_status"])
    except ValueError:
        raise DatasetError(f"{context}.second_step_status is invalid") from None
    score = item["second_step_ukeire_score"]
    if score is not None:
        expect_non_negative_int(
            score, DatasetError, f"{context}.second_step_ukeire_score"
        )
    try:
        return DiscardCandidateFeatures(
            action=action,
            post_discard_shanten=item["post_discard_shanten"],
            current_ukeire_count=item["current_ukeire_count"],
            second_step_ukeire_score=score,
            second_step_status=status,
        )
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{context} is not a valid candidate: {exc}") from exc


def resolve_teacher_candidate(
    candidates: tuple[DiscardCandidateFeatures, ...], teacher_action: object
) -> int:
    """teacher actionをcanonical candidate indexへ一意に解決する。

    legal `DiscardAction` candidateに一致しなければ`DatasetError`で
    fail closedする。
    """
    if not isinstance(teacher_action, DiscardAction):
        raise DatasetError(
            "teacher selected action of a scorer decision is not a DiscardAction: "
            f"{teacher_action!r}"
        )
    matches = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.action == teacher_action
    ]
    if len(matches) != 1:
        raise DatasetError(
            "teacher selected action does not resolve to exactly one legal "
            "discard candidate"
        )
    return matches[0]


def _empty_split_counts() -> dict[str, int]:
    return {field: 0 for field in sorted(_SPLIT_COUNT_FIELDS)}


def materialize_candidate_dataset(
    source: PlayerSafeSourceRecord, destination: str | Path
) -> CandidateDataset:
    """source recordからcandidate datasetを決定的にmaterializeする。

    rowの順序はsource populationの順序（game順、game内はdecision_ordinal順）
    を保つ。split membershipはsourceのものをそのまま使い、再配分しない。
    既存destinationは上書きしない。
    """
    if not isinstance(source, PlayerSafeSourceRecord):
        raise DatasetError("source must be a PlayerSafeSourceRecord")
    if source.decision_count == 0:
        raise DatasetError("source record contains no decisions")
    if source.allocation_bindings is None:
        raise DatasetError(
            "candidate dataset materialization requires a source record with "
            f"Arena allocation provenance (schema {SOURCE_RECORD_SCHEMA_V2!r}); "
            f"got schema {source.schema!r} without allocation_bindings"
        )

    destination = Path(destination)
    with staged_publication(destination, DatasetError) as staging:
        row_lines: list[str] = []
        context_payload = bytearray()
        candidate_payload = bytearray()
        split_counts: dict[str, dict[str, int]] = {}
        candidate_total = 0

        for decision in source.decisions():
            counts = split_counts.setdefault(decision.split, _empty_split_counts())
            counts["source_decisions"] += 1
            context = DecisionContext(
                input=decision.policy_input, legal_actions=decision.legal_actions
            )
            kind = classify_o0_decision(context)
            if kind is not O0DecisionKind.DISCARD:
                counts[_EXCLUSION_FIELD[kind]] += 1
                continue

            try:
                candidates = build_scorer_candidates(context)
                encoded = encode_candidates(candidates)
            except CandidateFeatureError as exc:
                raise DatasetError(
                    f"candidate materialization failed at game "
                    f"{decision.game_ordinal} decision {decision.decision_ordinal}: "
                    f"{exc}"
                ) from exc
            teacher_index = resolve_teacher_candidate(
                candidates, decision.selected_action
            )
            shared = build_player_safe_feature(decision.policy_input)
            if len(shared) != FEATURE_DIMENSION or any(
                not isfinite(value) for value in shared
            ):
                raise DatasetError("shared context materialization is invalid")

            row_lines.append(
                canonical_json_line(
                    {
                        "actor_seat": int(decision.actor_seat),
                        "candidate_count": len(candidates),
                        "candidate_offset": candidate_total,
                        "candidates": [
                            _candidate_to_value(candidate) for candidate in candidates
                        ],
                        "decision_ordinal": decision.decision_ordinal,
                        "game_ordinal": decision.game_ordinal,
                        "seed": decision.seed,
                        "split": decision.split,
                        "step_ordinal": decision.step_ordinal,
                        "teacher_candidate_index": teacher_index,
                    }
                )
            )
            context_payload += _float32_bytes(shared)
            for vector in encoded:
                candidate_payload += _float32_bytes(vector)
            candidate_total += len(candidates)
            counts["scorer_decisions"] += 1
            counts["candidates"] += len(candidates)

        decision_count = len(row_lines)
        if decision_count == 0:
            raise DatasetError("source record contains no scorer decisions")
        decisions_path = staging / DECISIONS_FILENAME
        context_path = staging / CONTEXT_FILENAME
        candidates_path = staging / CANDIDATES_FILENAME
        write_new_text(decisions_path, "".join(row_lines), DatasetError)
        write_new_bytes(context_path, bytes(context_payload), DatasetError)
        write_new_bytes(candidates_path, bytes(candidate_payload), DatasetError)

        manifest = seal(
            {
                "dataset_schema": CANDIDATE_DATASET_SCHEMA,
                "encoding": encoding_block(),
                "feature": feature_block(),
                "files": {
                    CANDIDATES_FILENAME: {
                        **file_digest(candidates_path),
                        "columns": CANDIDATE_ENCODING_DIMENSION,
                        "dtype": _FLOAT_DTYPE,
                        "rows": candidate_total,
                    },
                    CONTEXT_FILENAME: {
                        **file_digest(context_path),
                        "columns": FEATURE_DIMENSION,
                        "dtype": _FLOAT_DTYPE,
                        "rows": decision_count,
                    },
                    DECISIONS_FILENAME: {
                        **file_digest(decisions_path),
                        "columns": 1,
                        "dtype": _ROWS_DTYPE,
                        "rows": decision_count,
                    },
                },
                "kind": CANDIDATE_DATASET_KIND,
                "label": candidate_label_block(),
                "rows": {
                    "candidates": candidate_total,
                    "scorer_decisions": decision_count,
                    "source_decisions": source.decision_count,
                    "splits": dict(sorted(split_counts.items())),
                },
                "source": source.provenance(),
            }
        )
        write_new_text(
            staging / MANIFEST_FILENAME, canonical_json_text(manifest), DatasetError
        )
        published = read_candidate_dataset(staging)

    republished = read_candidate_dataset(destination)
    if republished.identity != published.identity:
        raise DatasetError("dataset readback identity mismatch after publication")
    return republished


def _read_manifest(path: Path) -> dict[str, object]:
    manifest_path = path / MANIFEST_FILENAME
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"dataset manifest cannot be read: {manifest_path}") from exc
    manifest = parse_json_text(text, DatasetError, "dataset manifest")
    body = unseal(manifest, DatasetError, "dataset manifest")
    if text != canonical_json_text(manifest):
        raise DatasetError("dataset manifest is not canonical JSON")
    expect_object(body, _MANIFEST_FIELDS, DatasetError, "dataset manifest")

    if body["dataset_schema"] != CANDIDATE_DATASET_SCHEMA:
        raise DatasetError(
            f"unsupported candidate dataset schema: {body['dataset_schema']!r}; "
            f"this implementation provides {CANDIDATE_DATASET_SCHEMA!r}"
        )
    if body["kind"] != CANDIDATE_DATASET_KIND:
        raise DatasetError("dataset kind mismatch")
    if body["feature"] != feature_block():
        raise DatasetError("dataset shared feature identity/fingerprint mismatch")
    if body["encoding"] != encoding_block():
        raise DatasetError(
            "dataset candidate encoding / candidate feature / second-step request "
            "policy identity mismatch"
        )
    label = expect_object(body["label"], _LABEL_FIELDS, DatasetError, "manifest.label")
    if label != candidate_label_block():
        raise DatasetError("dataset label semantics mismatch")

    source = validate_source_block(body["source"], context="manifest.source")

    rows = expect_object(body["rows"], _ROWS_FIELDS, DatasetError, "manifest.rows")
    for field in ("candidates", "scorer_decisions", "source_decisions"):
        expect_non_negative_int(rows[field], DatasetError, f"manifest.rows.{field}")
    if rows["source_decisions"] != source["decision_count"]:
        raise DatasetError("manifest source decision count mismatch")
    splits = rows["splits"]
    population_splits: dict[str, int] = {}
    for entry in source["population"]:
        population_splits[entry["split"]] = (
            population_splits.get(entry["split"], 0) + entry["decision_count"]
        )
    if type(splits) is not dict or set(splits) != set(population_splits):
        raise DatasetError("manifest.rows.splits does not match the source splits")
    totals = {"candidates": 0, "scorer_decisions": 0}
    for name, counts in splits.items():
        context = f"manifest.rows.splits[{name}]"
        item = expect_object(counts, _SPLIT_COUNT_FIELDS, DatasetError, context)
        for field in _SPLIT_COUNT_FIELDS:
            expect_non_negative_int(item[field], DatasetError, f"{context}.{field}")
        if item["source_decisions"] != population_splits[name]:
            raise DatasetError(f"{context} source decision count mismatch")
        if (
            item["scorer_decisions"]
            + item["excluded_win"]
            + item["excluded_riichi"]
            + item["excluded_response"]
            != item["source_decisions"]
        ):
            raise DatasetError(f"{context} O0 exclusion accounting mismatch")
        totals["candidates"] += item["candidates"]
        totals["scorer_decisions"] += item["scorer_decisions"]
    if totals["candidates"] != rows["candidates"] or (
        totals["scorer_decisions"] != rows["scorer_decisions"]
    ):
        raise DatasetError("manifest.rows split accounting mismatch")

    files = expect_object(
        body["files"],
        frozenset({DECISIONS_FILENAME, CONTEXT_FILENAME, CANDIDATES_FILENAME}),
        DatasetError,
        "manifest.files",
    )
    expected_layout = {
        CANDIDATES_FILENAME: (
            rows["candidates"],
            CANDIDATE_ENCODING_DIMENSION,
            _FLOAT_DTYPE,
        ),
        CONTEXT_FILENAME: (rows["scorer_decisions"], FEATURE_DIMENSION, _FLOAT_DTYPE),
        DECISIONS_FILENAME: (rows["scorer_decisions"], 1, _ROWS_DTYPE),
    }
    for name, payload in files.items():
        context = f"manifest.files[{name}]"
        item = expect_object(payload, _PAYLOAD_FIELDS, DatasetError, context)
        expect_non_negative_int(item["bytes"], DatasetError, f"{context}.bytes")
        expect_digest(item["sha256"], DatasetError, f"{context}.sha256")
        if (item["rows"], item["columns"], item["dtype"]) != expected_layout[name]:
            raise DatasetError(f"{context} payload layout mismatch")
    return manifest


def _read_rows(
    path: Path, manifest: dict[str, object]
) -> tuple[CandidateDecisionRow, ...]:
    population = {
        entry["game_ordinal"]: entry for entry in manifest["source"]["population"]
    }
    rows: list[CandidateDecisionRow] = []
    last_position = (-1, -1)
    candidate_total = 0
    observed: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="\n") as stream:
        for index, line in enumerate(stream):
            context = f"decisions[{index}]"
            value = parse_json_text(line, DatasetError, context)
            if line != canonical_json_line(value):
                raise DatasetError(f"{context} is not canonical JSON")
            expect_object(value, _DECISION_FIELDS, DatasetError, context)
            for field in (
                "actor_seat",
                "candidate_count",
                "candidate_offset",
                "decision_ordinal",
                "game_ordinal",
                "seed",
                "step_ordinal",
                "teacher_candidate_index",
            ):
                expect_non_negative_int(
                    value[field], DatasetError, f"{context}.{field}"
                )
            expect_str(value["split"], DatasetError, f"{context}.split")

            game = population.get(value["game_ordinal"])
            if game is None:
                raise DatasetError(f"{context} references an unknown source hanchan")
            if value["seed"] != game["seed"] or value["split"] != game["split"]:
                raise DatasetError(f"{context} provenance does not match the manifest")
            if value["decision_ordinal"] >= game["decision_count"]:
                raise DatasetError(f"{context} decision ordinal is out of range")
            position = (value["game_ordinal"], value["decision_ordinal"])
            if position <= last_position:
                raise DatasetError(f"{context} source ordering is not preserved")
            last_position = position

            try:
                actor_seat = Seat(value["actor_seat"])
            except ValueError:
                raise DatasetError(f"{context} actor seat is invalid") from None
            candidate_values = expect_list(
                value["candidates"], DatasetError, f"{context}.candidates"
            )
            candidates = tuple(
                _parse_candidate(item, f"{context}.candidates[{ordinal}]")
                for ordinal, item in enumerate(candidate_values)
            )
            if not candidates or value["candidate_count"] != len(candidates):
                raise DatasetError(f"{context} candidate count mismatch")
            if any(candidate.action.actor != actor_seat for candidate in candidates):
                raise DatasetError(f"{context} candidate actor does not match")
            if value["candidate_offset"] != candidate_total:
                raise DatasetError(f"{context} candidate offset is not contiguous")
            if value["teacher_candidate_index"] >= len(candidates):
                raise DatasetError(f"{context} teacher candidate index is out of range")
            candidate_total += len(candidates)

            counts = observed.setdefault(
                value["split"], {"candidates": 0, "scorer_decisions": 0}
            )
            counts["candidates"] += len(candidates)
            counts["scorer_decisions"] += 1
            rows.append(
                CandidateDecisionRow(
                    game_ordinal=value["game_ordinal"],
                    seed=value["seed"],
                    split=value["split"],
                    step_ordinal=value["step_ordinal"],
                    decision_ordinal=value["decision_ordinal"],
                    actor_seat=actor_seat,
                    candidate_offset=value["candidate_offset"],
                    teacher_candidate_index=value["teacher_candidate_index"],
                    candidates=candidates,
                )
            )

    manifest_rows = manifest["rows"]
    if len(rows) != manifest_rows["scorer_decisions"]:
        raise DatasetError("dataset decision count mismatch")
    if candidate_total != manifest_rows["candidates"]:
        raise DatasetError("dataset candidate count mismatch")
    for name, counts in manifest_rows["splits"].items():
        seen = observed.get(name, {"candidates": 0, "scorer_decisions": 0})
        if (
            seen["candidates"] != counts["candidates"]
            or seen["scorer_decisions"] != counts["scorer_decisions"]
        ):
            raise DatasetError("dataset split membership accounting mismatch")
    return tuple(rows)


def read_candidate_dataset(path: str | Path) -> CandidateDataset:
    """candidate dataset directoryをstrict readする。1つでも合わなければfail closed。"""
    root = Path(path)
    if not root.is_dir():
        raise DatasetError(f"dataset directory does not exist: {root}")
    manifest = _read_manifest(root)
    expected_names = {
        MANIFEST_FILENAME,
        DECISIONS_FILENAME,
        CONTEXT_FILENAME,
        CANDIDATES_FILENAME,
    }
    if {child.name for child in root.iterdir()} != expected_names:
        raise DatasetError("missing/unexpected dataset files")

    files = manifest["files"]
    for name in (DECISIONS_FILENAME, CONTEXT_FILENAME, CANDIDATES_FILENAME):
        digest = file_digest(root / name)
        if (
            digest["bytes"] != files[name]["bytes"]
            or digest["sha256"] != files[name]["sha256"]
        ):
            raise DatasetError(f"dataset payload size/digest mismatch: {name}")

    rows = _read_rows(root / DECISIONS_FILENAME, manifest)

    context_bytes = (root / CONTEXT_FILENAME).read_bytes()
    if len(context_bytes) != len(rows) * FEATURE_DIMENSION * _FLOAT32_BYTES:
        raise DatasetError("dataset context payload has an unexpected byte length")
    context = _float32_array(context_bytes)
    if any(not isfinite(value) for value in context):
        raise DatasetError("dataset context payload contains a non-finite value")

    candidate_bytes = (root / CANDIDATES_FILENAME).read_bytes()
    candidate_count = manifest["rows"]["candidates"]
    if len(candidate_bytes) != (
        candidate_count * CANDIDATE_ENCODING_DIMENSION * _FLOAT32_BYTES
    ):
        raise DatasetError("dataset candidate payload has an unexpected byte length")

    # typed candidate semanticから再encodeし、payloadと一致することを照合する。
    # encoding contract（request policyとの整合を含む）に反するrowもここで
    # fail closedする。
    reencoded = bytearray()
    for index, row in enumerate(rows):
        try:
            vectors = encode_candidates(row.candidates)
        except CandidateFeatureError as exc:
            raise DatasetError(f"decisions[{index}] candidates: {exc}") from exc
        for vector in vectors:
            reencoded += _float32_bytes(vector)
    if bytes(reencoded) != candidate_bytes:
        raise DatasetError(
            "dataset candidate payload does not match the typed candidate semantic"
        )

    return CandidateDataset(
        identity=manifest["identity"],
        manifest=manifest,
        rows=rows,
        context=context,
        candidates=_float32_array(candidate_bytes),
    )


__all__ = [
    "CANDIDATES_FILENAME",
    "CANDIDATE_DATASET_KIND",
    "CANDIDATE_DATASET_SCHEMA",
    "CANDIDATE_LABEL_SEMANTICS",
    "CANDIDATE_TRAINING_OBJECTIVE",
    "CONTEXT_FILENAME",
    "DECISIONS_FILENAME",
    "MANIFEST_FILENAME",
    "CandidateDataset",
    "CandidateDecisionRow",
    "candidate_label_block",
    "materialize_candidate_dataset",
    "read_candidate_dataset",
    "resolve_teacher_candidate",
]
