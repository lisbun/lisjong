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
from collections.abc import Iterator, Mapping
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
    appended_new_bytes,
    appended_new_text,
    staged_publication,
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
    DEVELOPMENT_PURPOSE,
    POLICY_SOURCE_RECORD_SCHEMA_V1,
    SCIENTIFIC_PURPOSE,
    SOURCE_RECORD_SCHEMA_V2,
    PlayerSafeSourceRecord,
    StreamingSourceRecord,
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
_POLICY_SOURCE_FIELDS = frozenset(
    {
        "allocation_bindings",
        "decision_count",
        "game_mode",
        "identity",
        "population",
        "purpose",
        "schema",
        "source_contract_digest",
        "teacher",
    }
)
_TEACHER_FIELDS = frozenset(
    {"catalog_identity", "policy_class", "configuration_digest"}
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
    policy = (
        type(value) is dict and value.get("schema") == POLICY_SOURCE_RECORD_SCHEMA_V1
    )
    source = expect_object(
        value,
        _POLICY_SOURCE_FIELDS if policy else _SOURCE_FIELDS,
        DatasetError,
        context,
    )
    for field in (
        ("schema", "identity", "game_mode")
        if policy
        else (
            "schema",
            "identity",
            "lock_identity",
            "game_mode",
            "scientific_corpus_identity",
        )
    ):
        expect_str(source[field], DatasetError, f"{context}.{field}")
    if policy:
        if source["purpose"] not in (DEVELOPMENT_PURPOSE, SCIENTIFIC_PURPOSE):
            raise DatasetError(f"{context}.purpose is not a supported purpose")
        teacher = expect_object(
            source["teacher"], _TEACHER_FIELDS, DatasetError, f"{context}.teacher"
        )
        for field in _TEACHER_FIELDS:
            expect_str(teacher[field], DatasetError, f"{context}.teacher.{field}")
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
    # ledgerへは一切照会しない。唯一の例外はpolicy source recordの
    # DEVELOPMENT populationで、bindingを持たないことを明示している
    # （lisjong-arena#442の接続・費用計測用）。
    if policy and source["purpose"] == DEVELOPMENT_PURPOSE:
        if source["allocation_bindings"] is not None:
            raise DatasetError(
                f"{context} DEVELOPMENT source must not claim allocation bindings"
            )
        return source
    try:
        validate_allocation_bindings(
            source["allocation_bindings"],
            populations=seeds_by_split,
            context=f"{context}.allocation_bindings",
        )
    except SourceRecordError as error:
        raise DatasetError(str(error)) from error
    return source


def _check_source(source: object, *, purpose_text: str) -> None:
    if not isinstance(source, (PlayerSafeSourceRecord, StreamingSourceRecord)):
        raise DatasetError("source must be a PlayerSafeSourceRecord")
    if source.decision_count == 0:
        raise DatasetError("source record contains no decisions")
    if source.allocation_bindings is None and source.purpose != DEVELOPMENT_PURPOSE:
        raise DatasetError(
            f"{purpose_text} requires a source record with Arena "
            "allocation provenance (schema "
            f"{SOURCE_RECORD_SCHEMA_V2!r}); got schema {source.schema!r} "
            "without allocation_bindings"
        )


def publish_dataset(
    source: PlayerSafeSourceRecord | StreamingSourceRecord, destination: str | Path
) -> dict[str, object]:
    """source recordからdatasetを決定的にmaterializeし、検証済みmanifestを返す。

    出力のbytes・順序・manifest identityは`materialize_dataset()`と同一である
    （同じ実装を共有する）。違いはpayloadをメモリへ読み戻さないことだけであり、
    rowは判断ごとにstaging fileへ追記し、公開前後の検証も逐次readで行う
    （Issue #247）。`StreamingSourceRecord`と組み合わせると、変換中に保持する
    のは1 game分のdecisionとmanifestの数え上げだけになる。

    失敗時はstagingを破棄し、destinationを作らない（fail closed）。
    """
    _check_source(source, purpose_text="dataset materialization")

    destination = Path(destination)
    with staged_publication(destination, DatasetError) as staging:
        rows_path = staging / ROWS_FILENAME
        features_path = staging / FEATURES_FILENAME
        mask_path = staging / LEGAL_MASK_FILENAME

        row_count = 0
        split_counts: dict[str, int] = {}
        with (
            appended_new_text(rows_path, DatasetError) as rows_stream,
            appended_new_bytes(features_path, DatasetError) as features_stream,
            appended_new_bytes(mask_path, DatasetError) as mask_stream,
        ):
            for decision in source.decisions():
                context = DecisionContext(
                    input=decision.policy_input, legal_actions=decision.legal_actions
                )
                mask = build_legal_action_mask(context)
                teacher_index = encode_action(decision.selected_action)
                if not mask[teacher_index]:
                    raise DatasetError(
                        "teacher selected action is not legal under the bound "
                        "vocabulary"
                    )
                values = build_player_safe_feature(decision.policy_input)
                if len(values) != FEATURE_DIMENSION:
                    raise DatasetError(
                        "feature materialization produced a wrong length"
                    )
                if any(not isfinite(value) for value in values):
                    raise DatasetError(
                        "feature materialization produced a non-finite value"
                    )

                legal_count = sum(1 for flag in mask if flag)
                rows_stream.write(
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
                features_stream.write(_float32_bytes(values))
                mask_stream.write(bytes(1 if flag else 0 for flag in mask))
                split_counts[decision.split] = split_counts.get(decision.split, 0) + 1
                row_count += 1

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
        with appended_new_text(staging / MANIFEST_FILENAME, DatasetError) as stream:
            stream.write(canonical_json_text(manifest))
        published = verify_dataset(staging)

    republished = verify_dataset(destination)
    if republished["identity"] != published["identity"]:
        raise DatasetError("dataset readback identity mismatch after publication")
    return republished


def materialize_dataset(
    source: PlayerSafeSourceRecord | StreamingSourceRecord, destination: str | Path
) -> "LearningDataset":
    """source recordからdatasetを決定的にmaterializeし、strict readで返す。

    rowの順序はsource populationの順序（game順、game内はdecision_ordinal順）
    をそのまま保持する。既存destinationは上書きしない。同じsource recordから
    再materializeすると、同じmanifest identityと同じpayload digestになる。

    `source`はschema v2（Arena allocation binding provenance付き）でなければ
    ならない。historical schema v1のsource recordはallocation provenanceを
    持たないため、それを推測で補完してdatasetを作ることはしない。v1
    readbackはこのIssueのcanonical first-slice datasetの入力にはならない。

    書出しは`publish_dataset()`と同じであり、戻り値のためにdataset全体を
    メモリへ読む。半荘数に依存しないメモリで変換だけを行う場合は
    `publish_dataset()`を使う。
    """
    manifest = publish_dataset(source, destination)
    dataset = read_dataset(destination)
    if dataset.identity != manifest["identity"]:
        raise DatasetError("dataset readback identity mismatch after publication")
    return dataset


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
    return tuple(_iter_rows(path, manifest))


def _iter_rows(path: Path, manifest: dict[str, object]) -> Iterator[DatasetRow]:
    """rows.jsonlを1行ずつstrict readして列挙する。

    population全体にわたる照合（hanchanごとのdecision数、row数、split母数）は
    最後の行の後に行うため、呼び出し側は最後まで列挙しなければならない。
    """
    population = manifest["source"]["population"]
    expected_splits = {
        entry["game_ordinal"]: (entry["seed"], entry["split"], entry["decision_count"])
        for entry in population
    }
    row_count = 0
    seen_games: set[int] = set()
    observed: dict[str, int] = {}
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

            row_count += 1
            seen_games.add(game_ordinal)
            observed[split] = observed.get(split, 0) + 1
            yield DatasetRow(
                game_ordinal=game_ordinal,
                seed=seed,
                split=split,
                step_ordinal=value["step_ordinal"],
                decision_ordinal=value["decision_ordinal"],
                actor_seat=actor_seat,
                legal_action_count=value["legal_action_count"],
                teacher_action_index=value["teacher_action_index"],
            )

    if row_count and ordinal_in_game != expected_splits[last_game][2]:
        raise DatasetError("dataset hanchan decision accounting mismatch")
    if row_count != manifest["rows"]["count"]:
        raise DatasetError("dataset row count mismatch")
    if len(seen_games) != len(expected_splits):
        raise DatasetError("dataset is missing source hanchan rows")
    if observed != manifest["rows"]["splits"]:
        raise DatasetError("dataset split membership accounting mismatch")


def _open_checked(root: Path) -> dict[str, object]:
    """manifest、file集合、payloadのbyte長とdigestを照合する。"""
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
    return manifest


def _check_mask_row(index: int, row: DatasetRow, mask_row: bytes) -> None:
    if sum(mask_row) != row.legal_action_count:
        raise DatasetError(f"rows[{index}] legal action count mismatch")
    if not mask_row[row.teacher_action_index]:
        raise DatasetError(f"rows[{index}] teacher label is not legal")


def read_dataset(path: str | Path) -> LearningDataset:
    """dataset directoryをstrict readする。

    manifest identity、bindされたfeature / vocabulary / label identity、
    payloadのbyte長とdigest、row provenance、ordering、split母数、legal mask
    とteacher labelの整合をすべて照合する。1つでも合わなければfail closedする。
    """
    root = Path(path)
    manifest = _open_checked(root)

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
        _check_mask_row(index, row, legal_mask[start : start + _MASK_ROW_BYTES])

    return LearningDataset(
        identity=manifest["identity"],
        manifest=manifest,
        rows=rows,
        features=features,
        legal_mask=legal_mask,
    )


def verify_dataset(path: str | Path) -> dict[str, object]:
    """`read_dataset()`と同じ照合を逐次readで行い、検証済みmanifestを返す。

    payloadをメモリへ保持しないため、使用メモリはrow数に依存しない
    （Issue #247）。1つでも合わなければfail closedする。
    """
    root = Path(path)
    manifest = _open_checked(root)
    files = manifest["files"]
    row_count = manifest["rows"]["count"]
    if files[FEATURES_FILENAME]["bytes"] != row_count * _FEATURE_ROW_BYTES:
        raise DatasetError("dataset feature payload has an unexpected byte length")
    if files[LEGAL_MASK_FILENAME]["bytes"] != row_count * _MASK_ROW_BYTES:
        raise DatasetError("dataset legal mask payload has an unexpected byte length")

    with (
        (root / FEATURES_FILENAME).open("rb") as features,
        (root / LEGAL_MASK_FILENAME).open("rb") as legal_mask,
    ):
        for index, row in enumerate(_iter_rows(root / ROWS_FILENAME, manifest)):
            feature_row = features.read(_FEATURE_ROW_BYTES)
            mask_row = legal_mask.read(_MASK_ROW_BYTES)
            if (
                len(feature_row) != _FEATURE_ROW_BYTES
                or len(mask_row) != _MASK_ROW_BYTES
            ):
                raise DatasetError("dataset payload ended before its rows")
            if any(not isfinite(value) for value in _float32_array(feature_row)):
                raise DatasetError(
                    "dataset feature payload contains a non-finite value"
                )
            if any(flag not in (0, 1) for flag in mask_row):
                raise DatasetError("dataset legal mask payload must contain only 0 / 1")
            _check_mask_row(index, row, mask_row)
        if features.read(1) or legal_mask.read(1):
            raise DatasetError("dataset payload has bytes beyond its rows")
    return manifest


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
    "publish_dataset",
    "read_dataset",
    "validate_source_block",
    "verify_dataset",
    "vocabulary_block",
]
