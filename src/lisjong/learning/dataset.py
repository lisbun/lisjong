"""lisjong所有のBehavior Cloning datasetのmaterializationとstrict read。

Issue #184のL0bに対応する。player-safe source recordから、lisjong所有の
feature identityとcanonical action vocabularyへbindしたversioned datasetを
決定的に生成する。

```text
player-safe source record
    -> lisjong-owned feature materialization
    -> versioned dataset (identity-bound, strict-read validated)
```

1 dataset = 1 immutable directoryである。

```text
<dataset>/
    manifest.json     canonical JSON identity / provenance / payload digest
    rows.jsonl        1行 = 1 decisionのprovenanceとteacher label
    features.f32      N x FEATURE_DIMENSION little-endian float32 (row-major)
    legal_mask.u8     N x ACTION_VOCABULARY_SIZE uint8 (0 / 1)
```

manifestがbindするもの。

- source-record identity / lock identity / ordered source population
- feature identity / dimension / fingerprint
- action vocabulary version / size / fingerprint
- teacher / label semantics
- split membership（rowごとのsplitとsplitごとの母数）
- payloadのrow数、列数、byte長、SHA-256

dense featureとlegal maskをJSONへ展開しないのは容量のためだけであり、契約は
変わらない。両fileはrow-major fixed strideで、manifestがすべてのdimension、
row count、byte count、digestを保持し、readbackはそのすべてを照合する。

これはgeneric dataset registryでもexperiment trackerでもない。offense L0
use case 1本のためのversioned contractだけを持ち、任意schemaの登録、database、
code / factory / callableのserializeは提供しない。
"""

import sys
from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from lisjong.action_vocabulary import (
    ACTION_VOCABULARY_SIZE,
    ACTION_VOCABULARY_VERSION,
    build_legal_action_mask,
    encode_action,
)
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
from lisjong.learning._publication import (
    staged_publication,
    write_new_bytes,
    write_new_text,
)
from lisjong.learning._typed_values import vocabulary_fingerprint
from lisjong.learning.errors import DatasetError, SourceRecordError
from lisjong.learning.features import (
    FEATURE_DIMENSION,
    FEATURE_IDENTITY,
    build_player_safe_feature,
    feature_fingerprint,
)
from lisjong.learning.source_record import (
    SOURCE_RECORD_SCHEMA_V2,
    PlayerSafeSourceRecord,
    validate_allocation_bindings,
)
from lisjong.policy_contract import DecisionContext, Seat

DATASET_SCHEMA = "lisjong-offense-l0-bc-dataset-v1"
"""このdataset contractのidentity。schemaを変える場合は新しいversionにする。"""

DATASET_KIND = "behavior-cloning-dataset"

TEACHER_LABEL_SEMANTICS = "lisjong-offense-l0-teacher-selected-action-v1"
"""label semanticsのidentity。

1 rowのlabelは、source recordのteacher selected actionを、bindされたcanonical
action vocabularyでencodeしたindexである。labelはそのdecisionのlegal action
maskの上でのみ意味を持ち、非合法indexへ確率を割り当てる解釈を許さない。
soft target、reward、advantage、privileged truthは含まない。
"""

TRAINING_OBJECTIVE = "masked-action-classification"

MANIFEST_FILENAME = "manifest.json"
ROWS_FILENAME = "rows.jsonl"
FEATURES_FILENAME = "features.f32"
LEGAL_MASK_FILENAME = "legal_mask.u8"

_FLOAT32_BYTES = 4
_FEATURE_ROW_BYTES = FEATURE_DIMENSION * _FLOAT32_BYTES
_MASK_ROW_BYTES = ACTION_VOCABULARY_SIZE

_MANIFEST_FIELDS = frozenset(
    {
        "dataset_schema",
        "kind",
        "source",
        "feature",
        "vocabulary",
        "label",
        "rows",
        "files",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "allocation_bindings",
        "decision_count",
        "game_mode",
        "identity",
        "lock_identity",
        "population",
        "schema",
        "scientific_corpus_identity",
        "source_contract_digest",
    }
)
_POPULATION_FIELDS = frozenset({"decision_count", "game_ordinal", "seed", "split"})
_FEATURE_FIELDS = frozenset({"dimension", "fingerprint", "identity"})
_VOCABULARY_FIELDS = frozenset({"action_vocabulary_version", "fingerprint", "size"})
_LABEL_FIELDS = frozenset({"objective", "semantics"})
_ROWS_FIELDS = frozenset({"count", "splits"})
_PAYLOAD_FIELDS = frozenset({"bytes", "columns", "dtype", "rows", "sha256"})
_ROW_FIELDS = frozenset(
    {
        "actor_seat",
        "decision_ordinal",
        "game_ordinal",
        "legal_action_count",
        "seed",
        "split",
        "step_ordinal",
        "teacher_action_index",
    }
)

_ROWS_DTYPE = "canonical-json-line"
_FEATURES_DTYPE = "float32-le"
_MASK_DTYPE = "uint8"


@dataclass(frozen=True, slots=True)
class DatasetRow:
    """1 decision分のprovenanceとteacher label。featureはpayload側に持つ。"""

    game_ordinal: int
    seed: int
    split: str
    step_ordinal: int
    decision_ordinal: int
    actor_seat: Seat
    legal_action_count: int
    teacher_action_index: int


@dataclass(frozen=True, slots=True)
class LearningDataset:
    """strict readしたdatasetと、そのbindされたidentity。"""

    identity: str
    manifest: Mapping[str, object]
    rows: tuple[DatasetRow, ...]
    features: array
    legal_mask: bytes

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def feature_dimension(self) -> int:
        return FEATURE_DIMENSION

    def feature_row(self, index: int) -> tuple[float, ...]:
        start = index * FEATURE_DIMENSION
        return tuple(self.features[start : start + FEATURE_DIMENSION])

    def legal_mask_row(self, index: int) -> bytes:
        start = index * ACTION_VOCABULARY_SIZE
        return self.legal_mask[start : start + ACTION_VOCABULARY_SIZE]

    def legal_indices(self, index: int) -> tuple[int, ...]:
        row = self.legal_mask_row(index)
        return tuple(position for position, flag in enumerate(row) if flag)

    def row_indices_for_splits(self, splits: tuple[str, ...]) -> tuple[int, ...]:
        """指定splitに属するrow indexを、dataset順のまま返す。"""
        selected = set(splits)
        return tuple(
            index for index, row in enumerate(self.rows) if row.split in selected
        )

    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.split] = counts.get(row.split, 0) + 1
        return counts


def _float32_bytes(values: tuple[float, ...]) -> bytes:
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


def feature_block() -> dict[str, object]:
    """現在のruntimeがbindするfeature identity blockを返す。"""
    return {
        "dimension": FEATURE_DIMENSION,
        "fingerprint": feature_fingerprint(),
        "identity": FEATURE_IDENTITY,
    }


def vocabulary_block() -> dict[str, object]:
    """現在のruntimeがbindするaction vocabulary identity blockを返す。"""
    return {
        "action_vocabulary_version": ACTION_VOCABULARY_VERSION,
        "fingerprint": vocabulary_fingerprint(),
        "size": ACTION_VOCABULARY_SIZE,
    }


def label_block() -> dict[str, object]:
    """teacher / label semantics blockを返す。"""
    return {"objective": TRAINING_OBJECTIVE, "semantics": TEACHER_LABEL_SEMANTICS}


def _payload_block(
    path: Path, *, rows: int, columns: int, dtype: str
) -> dict[str, object]:
    return {**file_digest(path), "columns": columns, "dtype": dtype, "rows": rows}


def validate_source_block(
    value: object, *, context: str = "source"
) -> dict[str, object]:
    """source provenance blockをstrictに検証する。

    `PlayerSafeSourceRecord.provenance()`が生成する記述の正本validationであり、
    datasetとmodel artifactの双方が同じ契約でこのblockを検証する。
    """
    source = expect_object(value, _SOURCE_FIELDS, DatasetError, context)
    for field in (
        "schema",
        "identity",
        "lock_identity",
        "game_mode",
        "scientific_corpus_identity",
    ):
        expect_str(source[field], DatasetError, f"{context}.{field}")
    expect_digest(
        source["source_contract_digest"],
        DatasetError,
        f"{context}.source_contract_digest",
    )
    expect_non_negative_int(
        source["decision_count"], DatasetError, f"{context}.decision_count"
    )
    population = expect_list(
        source["population"], DatasetError, f"{context}.population"
    )
    if not population:
        raise DatasetError(f"{context}.population must not be empty")
    seeds: set[int] = set()
    seeds_by_split: dict[str, list[int]] = {}
    total = 0
    for ordinal, entry in enumerate(population):
        entry_context = f"{context}.population[{ordinal}]"
        item = expect_object(entry, _POPULATION_FIELDS, DatasetError, entry_context)
        expect_non_negative_int(item["seed"], DatasetError, f"{entry_context}.seed")
        expect_non_negative_int(
            item["decision_count"], DatasetError, f"{entry_context}.decision_count"
        )
        expect_str(item["split"], DatasetError, f"{entry_context}.split")
        if item["game_ordinal"] != ordinal:
            raise DatasetError(f"{entry_context} game ordinal is not contiguous")
        if item["seed"] in seeds:
            raise DatasetError(f"{context}.population reuses a seed")
        seeds.add(item["seed"])
        seeds_by_split.setdefault(item["split"], []).append(item["seed"])
        total += item["decision_count"]
    if source["decision_count"] != total:
        raise DatasetError(f"{context} decision accounting mismatch")

    # allocation_bindingsはArena-owned seed allocation ledger（lisjong-arena#346/
    # #347）のper-split provenanceである。datasetとartifactに到達する時点では
    # 常にschema v2由来（materialize_dataset()がv1を拒否する）なので、ここでは
    # Noneを許さず、populationのsplitと矛盾しないbindingを要求する。liveな
    # ledgerへは一切照会しない。
    try:
        validate_allocation_bindings(
            source["allocation_bindings"],
            populations=seeds_by_split,
            context=f"{context}.allocation_bindings",
        )
    except SourceRecordError as error:
        raise DatasetError(str(error)) from error
    return source


def materialize_dataset(
    source: PlayerSafeSourceRecord, destination: str | Path
) -> "LearningDataset":
    """source recordからdatasetを決定的にmaterializeし、strict readで返す。

    rowの順序はsource populationの順序（game順、game内はdecision_ordinal順）
    をそのまま保持する。既存destinationは上書きしない。同じsource recordから
    再materializeすると、同じmanifest identityと同じpayload digestになる。

    `source`はschema v2（Arena allocation binding provenance付き）でなければ
    ならない。historical schema v1のsource recordはallocation provenanceを
    持たないため、それを推測で補完してdatasetを作ることはしない。v1
    readbackはこのIssueのcanonical first-slice datasetの入力にはならない。
    """
    if not isinstance(source, PlayerSafeSourceRecord):
        raise DatasetError("source must be a PlayerSafeSourceRecord")
    if source.decision_count == 0:
        raise DatasetError("source record contains no decisions")
    if source.allocation_bindings is None:
        raise DatasetError(
            "dataset materialization requires a source record with Arena "
            "allocation provenance (schema "
            f"{SOURCE_RECORD_SCHEMA_V2!r}); got schema {source.schema!r} "
            "without allocation_bindings"
        )

    destination = Path(destination)
    with staged_publication(destination, DatasetError) as staging:
        rows_path = staging / ROWS_FILENAME
        features_path = staging / FEATURES_FILENAME
        mask_path = staging / LEGAL_MASK_FILENAME

        row_lines: list[str] = []
        feature_payload = bytearray()
        mask_payload = bytearray()
        split_counts: dict[str, int] = {}

        for decision in source.decisions():
            context = DecisionContext(
                input=decision.policy_input, legal_actions=decision.legal_actions
            )
            mask = build_legal_action_mask(context)
            teacher_index = encode_action(decision.selected_action)
            if not mask[teacher_index]:
                raise DatasetError(
                    "teacher selected action is not legal under the bound vocabulary"
                )
            values = build_player_safe_feature(decision.policy_input)
            if len(values) != FEATURE_DIMENSION:
                raise DatasetError("feature materialization produced a wrong length")
            if any(not isfinite(value) for value in values):
                raise DatasetError(
                    "feature materialization produced a non-finite value"
                )

            legal_count = sum(1 for flag in mask if flag)
            row_lines.append(
                canonical_json_line(
                    {
                        "actor_seat": int(decision.actor_seat),
                        "decision_ordinal": decision.decision_ordinal,
                        "game_ordinal": decision.game_ordinal,
                        "legal_action_count": legal_count,
                        "seed": decision.seed,
                        "split": decision.split,
                        "step_ordinal": decision.step_ordinal,
                        "teacher_action_index": teacher_index,
                    }
                )
            )
            feature_payload += _float32_bytes(values)
            mask_payload += bytes(1 if flag else 0 for flag in mask)
            split_counts[decision.split] = split_counts.get(decision.split, 0) + 1

        row_count = len(row_lines)
        write_new_text(rows_path, "".join(row_lines), DatasetError)
        write_new_bytes(features_path, bytes(feature_payload), DatasetError)
        write_new_bytes(mask_path, bytes(mask_payload), DatasetError)

        manifest = seal(
            {
                "dataset_schema": DATASET_SCHEMA,
                "feature": feature_block(),
                "files": {
                    FEATURES_FILENAME: _payload_block(
                        features_path,
                        rows=row_count,
                        columns=FEATURE_DIMENSION,
                        dtype=_FEATURES_DTYPE,
                    ),
                    LEGAL_MASK_FILENAME: _payload_block(
                        mask_path,
                        rows=row_count,
                        columns=ACTION_VOCABULARY_SIZE,
                        dtype=_MASK_DTYPE,
                    ),
                    ROWS_FILENAME: _payload_block(
                        rows_path, rows=row_count, columns=1, dtype=_ROWS_DTYPE
                    ),
                },
                "kind": DATASET_KIND,
                "label": label_block(),
                "rows": {
                    "count": row_count,
                    "splits": dict(sorted(split_counts.items())),
                },
                "source": source.provenance(),
                "vocabulary": vocabulary_block(),
            }
        )
        write_new_text(
            staging / MANIFEST_FILENAME, canonical_json_text(manifest), DatasetError
        )
        published = read_dataset(staging)

    republished = read_dataset(destination)
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

    if body["dataset_schema"] != DATASET_SCHEMA:
        raise DatasetError(
            f"unsupported dataset schema: {body['dataset_schema']!r}; "
            f"this implementation provides {DATASET_SCHEMA!r}"
        )
    if body["kind"] != DATASET_KIND:
        raise DatasetError("dataset kind mismatch")

    feature = expect_object(
        body["feature"], _FEATURE_FIELDS, DatasetError, "manifest.feature"
    )
    if feature != feature_block():
        raise DatasetError("dataset feature identity/fingerprint mismatch")
    vocabulary = expect_object(
        body["vocabulary"], _VOCABULARY_FIELDS, DatasetError, "manifest.vocabulary"
    )
    if vocabulary != vocabulary_block():
        raise DatasetError("dataset action vocabulary identity/fingerprint mismatch")
    label = expect_object(body["label"], _LABEL_FIELDS, DatasetError, "manifest.label")
    if label != label_block():
        raise DatasetError("dataset label semantics mismatch")

    source = validate_source_block(body["source"], context="manifest.source")
    total = source["decision_count"]

    rows = expect_object(body["rows"], _ROWS_FIELDS, DatasetError, "manifest.rows")
    row_count = expect_non_negative_int(
        rows["count"], DatasetError, "manifest.rows.count"
    )
    if row_count != total:
        raise DatasetError("manifest row count does not match the source population")
    splits = rows["splits"]
    if type(splits) is not dict or not splits:
        raise DatasetError("manifest.rows.splits must be a non-empty JSON object")
    for name, count in splits.items():
        expect_str(name, DatasetError, "manifest.rows.splits key")
        expect_non_negative_int(count, DatasetError, f"manifest.rows.splits[{name}]")
    if sum(splits.values()) != row_count:
        raise DatasetError("manifest.rows.splits accounting mismatch")

    files = expect_object(
        body["files"],
        frozenset({ROWS_FILENAME, FEATURES_FILENAME, LEGAL_MASK_FILENAME}),
        DatasetError,
        "manifest.files",
    )
    for name, payload in files.items():
        context = f"manifest.files[{name}]"
        item = expect_object(payload, _PAYLOAD_FIELDS, DatasetError, context)
        expect_non_negative_int(item["bytes"], DatasetError, f"{context}.bytes")
        expect_digest(item["sha256"], DatasetError, f"{context}.sha256")
        if item["rows"] != row_count:
            raise DatasetError(f"{context} row count mismatch")
    if (
        files[FEATURES_FILENAME]["columns"] != FEATURE_DIMENSION
        or files[FEATURES_FILENAME]["dtype"] != _FEATURES_DTYPE
        or files[LEGAL_MASK_FILENAME]["columns"] != ACTION_VOCABULARY_SIZE
        or files[LEGAL_MASK_FILENAME]["dtype"] != _MASK_DTYPE
        or files[ROWS_FILENAME]["columns"] != 1
        or files[ROWS_FILENAME]["dtype"] != _ROWS_DTYPE
    ):
        raise DatasetError("dataset payload layout mismatch")
    return manifest


def _read_rows(path: Path, manifest: dict[str, object]) -> tuple[DatasetRow, ...]:
    population = manifest["source"]["population"]
    expected_splits = {
        entry["game_ordinal"]: (entry["seed"], entry["split"], entry["decision_count"])
        for entry in population
    }
    rows: list[DatasetRow] = []
    last_game = 0
    ordinal_in_game = 0
    with path.open(encoding="utf-8", newline="\n") as stream:
        for index, line in enumerate(stream):
            context = f"rows[{index}]"
            value = parse_json_text(line, DatasetError, context)
            if line != canonical_json_line(value):
                raise DatasetError(f"{context} is not canonical JSON")
            expect_object(value, _ROW_FIELDS, DatasetError, context)
            for field in (
                "actor_seat",
                "decision_ordinal",
                "game_ordinal",
                "legal_action_count",
                "seed",
                "step_ordinal",
                "teacher_action_index",
            ):
                expect_non_negative_int(
                    value[field], DatasetError, f"{context}.{field}"
                )
            expect_str(value["split"], DatasetError, f"{context}.split")

            game_ordinal = value["game_ordinal"]
            if game_ordinal not in expected_splits:
                raise DatasetError(f"{context} references an unknown source hanchan")
            if game_ordinal < last_game:
                raise DatasetError(f"{context} source ordering is not preserved")
            if game_ordinal != last_game:
                if ordinal_in_game != expected_splits[last_game][2]:
                    raise DatasetError("dataset hanchan decision accounting mismatch")
                last_game, ordinal_in_game = game_ordinal, 0
            seed, split, _count = expected_splits[game_ordinal]
            if value["seed"] != seed or value["split"] != split:
                raise DatasetError(f"{context} provenance does not match the manifest")
            if value["decision_ordinal"] != ordinal_in_game:
                raise DatasetError(f"{context} decision ordinal is not contiguous")
            ordinal_in_game += 1

            if value["teacher_action_index"] >= ACTION_VOCABULARY_SIZE:
                raise DatasetError(f"{context} teacher action index is out of range")
            if not 1 <= value["legal_action_count"] <= ACTION_VOCABULARY_SIZE:
                raise DatasetError(f"{context} legal action count is out of range")
            try:
                actor_seat = Seat(value["actor_seat"])
            except ValueError:
                raise DatasetError(f"{context} actor seat is invalid") from None

            rows.append(
                DatasetRow(
                    game_ordinal=game_ordinal,
                    seed=seed,
                    split=split,
                    step_ordinal=value["step_ordinal"],
                    decision_ordinal=value["decision_ordinal"],
                    actor_seat=actor_seat,
                    legal_action_count=value["legal_action_count"],
                    teacher_action_index=value["teacher_action_index"],
                )
            )

    if rows and ordinal_in_game != expected_splits[last_game][2]:
        raise DatasetError("dataset hanchan decision accounting mismatch")
    if len(rows) != manifest["rows"]["count"]:
        raise DatasetError("dataset row count mismatch")
    if len({row.game_ordinal for row in rows}) != len(expected_splits):
        raise DatasetError("dataset is missing source hanchan rows")
    observed: dict[str, int] = {}
    for row in rows:
        observed[row.split] = observed.get(row.split, 0) + 1
    if observed != manifest["rows"]["splits"]:
        raise DatasetError("dataset split membership accounting mismatch")
    return tuple(rows)


def read_dataset(path: str | Path) -> LearningDataset:
    """dataset directoryをstrict readする。

    manifest identity、bindされたfeature / vocabulary / label identity、
    payloadのbyte長とdigest、row provenance、ordering、split母数、legal mask
    とteacher labelの整合をすべて照合する。1つでも合わなければfail closedする。
    """
    root = Path(path)
    if not root.is_dir():
        raise DatasetError(f"dataset directory does not exist: {root}")
    manifest = _read_manifest(root)
    expected_names = {
        MANIFEST_FILENAME,
        ROWS_FILENAME,
        FEATURES_FILENAME,
        LEGAL_MASK_FILENAME,
    }
    if {child.name for child in root.iterdir()} != expected_names:
        raise DatasetError("missing/unexpected dataset files")

    files = manifest["files"]
    for name in (ROWS_FILENAME, FEATURES_FILENAME, LEGAL_MASK_FILENAME):
        payload = root / name
        digest = file_digest(payload)
        if (
            digest["bytes"] != files[name]["bytes"]
            or digest["sha256"] != files[name]["sha256"]
        ):
            raise DatasetError(f"dataset payload size/digest mismatch: {name}")

    row_count = manifest["rows"]["count"]
    rows = _read_rows(root / ROWS_FILENAME, manifest)

    feature_bytes = (root / FEATURES_FILENAME).read_bytes()
    if len(feature_bytes) != row_count * _FEATURE_ROW_BYTES:
        raise DatasetError("dataset feature payload has an unexpected byte length")
    features = _float32_array(feature_bytes)
    if any(not isfinite(value) for value in features):
        raise DatasetError("dataset feature payload contains a non-finite value")

    legal_mask = (root / LEGAL_MASK_FILENAME).read_bytes()
    if len(legal_mask) != row_count * _MASK_ROW_BYTES:
        raise DatasetError("dataset legal mask payload has an unexpected byte length")
    if any(flag not in (0, 1) for flag in legal_mask):
        raise DatasetError("dataset legal mask payload must contain only 0 / 1")

    for index, row in enumerate(rows):
        start = index * _MASK_ROW_BYTES
        mask_row = legal_mask[start : start + _MASK_ROW_BYTES]
        if sum(mask_row) != row.legal_action_count:
            raise DatasetError(f"rows[{index}] legal action count mismatch")
        if not mask_row[row.teacher_action_index]:
            raise DatasetError(f"rows[{index}] teacher label is not legal")

    return LearningDataset(
        identity=manifest["identity"],
        manifest=manifest,
        rows=rows,
        features=features,
        legal_mask=legal_mask,
    )


__all__ = [
    "DATASET_KIND",
    "DATASET_SCHEMA",
    "FEATURES_FILENAME",
    "LEGAL_MASK_FILENAME",
    "MANIFEST_FILENAME",
    "ROWS_FILENAME",
    "TEACHER_LABEL_SEMANTICS",
    "TRAINING_OBJECTIVE",
    "DatasetRow",
    "LearningDataset",
    "feature_block",
    "label_block",
    "materialize_dataset",
    "read_dataset",
    "validate_source_block",
    "vocabulary_block",
]
