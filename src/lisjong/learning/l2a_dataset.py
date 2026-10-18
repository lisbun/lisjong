"""L2a候補scorer datasetのmaterializationとstrict read（Issue #316）。

設計`docs/learning-l2a.md` 2節・4節。player-safe source record（schema v2）の
全判断を`lisjong.learning.l2a_decision`で分類し、encodingのheadに当たる判断
（打牌M0 / M1はDISCARD、反応R1はCALL）だけをdatasetへ入れる。単一候補行は
入れず、件数だけ数える。L0.2のcandidate dataset（O0 eligibility）とは別の
contractであり、既存datasetを変更しない。

```text
player-safe source record
    -> classify_l2a_decision（全判断を分類し、split別に数える）
    -> headの判断のうち単一候補行でないものごとに
         #184 shared context（build_player_safe_feature）
         L2aEncoding.encode()（servingと同じ関数）
         teacher_selected_actionの候補index（一意に解決できなければfail closed）
    -> immutable dataset directory
```

```text
<dataset>/
    manifest.json      canonical JSON identity / provenance / payload digest
    decisions.jsonl    1行 = 1 scorer判断のprovenance / label / 候補action
    context.f32        D x FEATURE_DIMENSION little-endian float32
    candidates.f32     C x encoding次元 little-endian float32
```

strict readはmanifest identity、bindしたshared feature / encoding /
分類identity / label、payloadのbyte長とdigest、rowのprovenance、split別の
件数と分類の勘定、候補区間の連続性、値が有限であることを照合する。
encodingの値そのもののtrain-serve一致は、datasetとservingが同じ
`L2aEncoding.encode()`を呼ぶことで保ち、評価では同じsourceから再encodeして
照合する（設計5.1節）。
"""

import shutil
from collections import deque
from collections.abc import Iterator, Mapping
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from contextlib import contextmanager
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
from lisjong.learning._publication import (
    appended_new_bytes,
    appended_new_text,
    staged_publication,
)
from lisjong.learning._typed_values import action_to_value, parse_action
from lisjong.learning.candidate_dataset import _float32_array, _float32_bytes
from lisjong.learning.dataset import feature_block, validate_source_block
from lisjong.learning.errors import CandidateFeatureError, DatasetError
from lisjong.learning.features import FEATURE_DIMENSION, build_player_safe_feature
from lisjong.learning.l2a_decision import (
    L2aDecisionKind,
    classify_l2a_decision,
    is_single_candidate,
    teacher_rule_exception,
)
from lisjong.learning.l2a_encoding import L2aEncoding, l2a_encoding
from lisjong.learning.source_record import (
    DEVELOPMENT_PURPOSE,
    PlayerSafeSourceRecord,
    StreamingSourceRecord,
    read_source_game,
)
from lisjong.policy_contract import DecisionContext, InternalAction, Seat

L2A_DATASET_SCHEMA = "lisjong-l2a-candidate-scorer-dataset-v1"
L2A_DATASET_KIND = "l2a-candidate-scorer-dataset"
L2A_LABEL_SEMANTICS = "lisjong-l2a-teacher-selected-scorer-candidate-v1"
"""labelは`teacher_selected_action`がcanonical候補順の何番目かを表すindex。"""
L2A_TRAINING_OBJECTIVE = "per-decision-legal-candidate-softmax-cross-entropy"

MANIFEST_FILENAME = "manifest.json"
DECISIONS_FILENAME = "decisions.jsonl"
CONTEXT_FILENAME = "context.f32"
CANDIDATES_FILENAME = "candidates.f32"

_FLOAT32_BYTES = 4
_ROWS_DTYPE = "canonical-json-line"
_FLOAT_DTYPE = "float32-le"
_SHARD_DIRECTORY = ".shards"
_COPY_CHUNK_BYTES = 1024 * 1024
_VERIFY_CHUNK_FLOATS = 256 * 1024

_KIND_FIELDS = tuple(kind.value for kind in L2aDecisionKind)
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
_ROWS_FIELDS = frozenset(
    {"candidates", "scorer_decisions", "source_decisions", "splits"}
)
_SPLIT_COUNT_FIELDS = frozenset(
    {
        "candidates",
        "scorer_decisions",
        "single_candidate",
        "source_decisions",
        *_KIND_FIELDS,
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


@dataclass(frozen=True, slots=True)
class L2aDecisionRow:
    """1 scorer判断のprovenance、label、canonical順の候補action。"""

    game_ordinal: int
    seed: int
    split: str
    step_ordinal: int
    decision_ordinal: int
    actor_seat: Seat
    candidate_offset: int
    teacher_candidate_index: int
    actions: tuple[InternalAction, ...]

    @property
    def candidate_count(self) -> int:
        return len(self.actions)


@dataclass(frozen=True, slots=True)
class L2aDataset:
    """strict readしたL2a dataset。"""

    identity: str
    manifest: Mapping[str, object]
    encoding: L2aEncoding
    rows: tuple[L2aDecisionRow, ...]
    context: object
    candidates: object

    @property
    def decision_count(self) -> int:
        return len(self.rows)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates) // self.encoding.dimension

    def row_indices_for_splits(self, splits: tuple[str, ...]) -> tuple[int, ...]:
        selected = set(splits)
        return tuple(
            index for index, row in enumerate(self.rows) if row.split in selected
        )

    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.split] = counts.get(row.split, 0) + 1
        return counts


def l2a_label_block(encoding: L2aEncoding) -> dict[str, object]:
    return {
        "head": encoding.head.value,
        "objective": L2A_TRAINING_OBJECTIVE,
        "semantics": L2A_LABEL_SEMANTICS,
    }


def resolve_teacher_index(
    actions: tuple[InternalAction, ...], teacher_action: object
) -> int:
    """教師の行動を候補indexへ一意に解決する。できなければfail closedする。"""
    matches = [
        index for index, action in enumerate(actions) if action == teacher_action
    ]
    if len(matches) != 1:
        raise DatasetError(
            "teacher selected action does not resolve to exactly one scorer "
            f"candidate: {teacher_action!r}"
        )
    return matches[0]


def _empty_split_counts() -> dict[str, int]:
    return {field: 0 for field in sorted(_SPLIT_COUNT_FIELDS)}


def _materialize_decision(
    decision, encoding: L2aEncoding, counts: dict[str, int], candidate_offset: int
) -> tuple[str, bytes, bytes, int] | None:
    """1判断をrow / context / candidate bytesへ変換する。対象外は`None`。"""
    counts["source_decisions"] += 1
    context = DecisionContext(
        input=decision.policy_input, legal_actions=decision.legal_actions
    )
    kind = classify_l2a_decision(context)
    counts[kind.value] += 1
    position = f"game {decision.game_ordinal} decision {decision.decision_ordinal}"
    # 教師規則の例外はheadに関係なく全行で拒否する（設計2.3・3.4節のSTOP条件）。
    exception = teacher_rule_exception(context, kind, decision.selected_action)
    if exception is not None:
        raise DatasetError(f"teacher rule exception at {position}: {exception}")
    if kind is not encoding.head:
        return None
    if is_single_candidate(context, kind):
        counts["single_candidate"] += 1
        return None

    try:
        encoded = encoding.encode(context)
    except CandidateFeatureError as exc:
        raise DatasetError(
            f"{encoding.name} encoding failed at {position}: {exc}"
        ) from exc
    try:
        teacher_index = resolve_teacher_index(encoded.actions, decision.selected_action)
    except DatasetError as exc:
        raise DatasetError(f"{exc} at {position}") from exc
    shared = build_player_safe_feature(decision.policy_input)
    vectors = encoded.vectors
    if len(shared) != FEATURE_DIMENSION or not all(map(isfinite, shared)):
        raise DatasetError(f"shared context materialization is invalid at {position}")
    if not all(isfinite(value) for vector in vectors for value in vector):
        raise DatasetError(f"candidate encoding is not finite at {position}")

    line = canonical_json_line(
        {
            "actor_seat": int(decision.actor_seat),
            "candidate_count": len(encoded.actions),
            "candidate_offset": candidate_offset,
            "candidates": [
                action_to_value(action, DatasetError, "candidate")
                for action in encoded.actions
            ],
            "decision_ordinal": decision.decision_ordinal,
            "game_ordinal": decision.game_ordinal,
            "seed": decision.seed,
            "split": decision.split,
            "step_ordinal": decision.step_ordinal,
            "teacher_candidate_index": teacher_index,
        }
    )
    counts["scorer_decisions"] += 1
    counts["candidates"] += len(encoded.actions)
    return (
        line,
        _float32_bytes(shared),
        b"".join(_float32_bytes(vector) for vector in vectors),
        len(encoded.actions),
    )


@contextmanager
def _payload_streams(root: Path) -> Iterator[tuple]:
    with (
        appended_new_text(root / DECISIONS_FILENAME, DatasetError) as decisions,
        appended_new_bytes(root / CONTEXT_FILENAME, DatasetError) as context,
        appended_new_bytes(root / CANDIDATES_FILENAME, DatasetError) as candidates,
    ):
        yield decisions, context, candidates


def _write_decisions(
    decisions,
    encoding: L2aEncoding,
    streams: tuple,
    split_counts: dict[str, dict[str, int]],
) -> tuple[int, int]:
    decisions_stream, context_stream, candidates_stream = streams
    decision_count = 0
    candidate_total = 0
    for decision in decisions:
        counts = split_counts.setdefault(decision.split, _empty_split_counts())
        result = _materialize_decision(decision, encoding, counts, candidate_total)
        if result is None:
            continue
        line, shared, candidates, count = result
        decisions_stream.write(line)
        context_stream.write(shared)
        candidates_stream.write(candidates)
        decision_count += 1
        candidate_total += count
    return decision_count, candidate_total


def _materialize_shard(
    root: str, summary: dict[str, object], shard: str, schema: str, encoding_name: str
) -> dict:
    """1半荘（shard）を変換する。worker processで実行する。"""
    game = read_source_game(Path(root), summary, schema=schema)
    split_counts: dict[str, dict[str, int]] = {}
    with _payload_streams(Path(shard)) as streams:
        decisions, candidates = _write_decisions(
            game.decisions, l2a_encoding(encoding_name), streams, split_counts
        )
    return {
        "candidates": candidates,
        "scorer_decisions": decisions,
        "split_counts": split_counts,
    }


def _shard_executor(workers: int) -> Executor:
    return ProcessPoolExecutor(max_workers=workers)


def _append_shard(shard: Path, streams: tuple, candidate_base: int) -> None:
    decisions_stream, context_stream, candidates_stream = streams
    with (shard / DECISIONS_FILENAME).open(encoding="utf-8", newline="\n") as rows:
        for line in rows:
            value = parse_json_text(line, DatasetError, "shard decision row")
            value["candidate_offset"] += candidate_base
            decisions_stream.write(canonical_json_line(value))
    for name, stream in (
        (CONTEXT_FILENAME, context_stream),
        (CANDIDATES_FILENAME, candidates_stream),
    ):
        with (shard / name).open("rb") as payload:
            shutil.copyfileobj(payload, stream, _COPY_CHUNK_BYTES)


def _write_shards(
    source: StreamingSourceRecord,
    encoding: L2aEncoding,
    staging: Path,
    streams: tuple,
    split_counts: dict[str, dict[str, int]],
    *,
    workers: int,
    max_pending_shards: int,
) -> tuple[int, int]:
    """半荘ごとのshardを並列に変換し、半荘順に連結する（L0.2 #248と同じ方式）。"""
    shard_root = staging / _SHARD_DIRECTORY
    shard_root.mkdir()
    summaries = iter(source.summaries)
    pending: deque[tuple[Path, Future]] = deque()
    decision_count = 0
    candidate_total = 0
    pool = _shard_executor(workers)
    try:
        while True:
            while len(pending) < max_pending_shards:
                summary = next(summaries, None)
                if summary is None:
                    break
                shard = shard_root / f"game-{summary['game_ordinal']:03d}"
                shard.mkdir()
                future = pool.submit(
                    _materialize_shard,
                    str(source.root),
                    summary,
                    str(shard),
                    source.schema,
                    encoding.name,
                )
                pending.append((shard, future))
            if not pending:
                break
            shard, future = pending.popleft()
            result = future.result()
            _append_shard(shard, streams, candidate_total)
            shutil.rmtree(shard)
            for split, counts in result["split_counts"].items():
                merged = split_counts.setdefault(split, _empty_split_counts())
                for field, count in counts.items():
                    merged[field] += count
            decision_count += result["scorer_decisions"]
            candidate_total += result["candidates"]
    except BaseException:
        terminate = getattr(pool, "terminate_workers", None)
        if terminate is not None:
            terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    shard_root.rmdir()
    return decision_count, candidate_total


def publish_l2a_dataset(
    source: PlayerSafeSourceRecord | StreamingSourceRecord,
    destination: str | Path,
    *,
    encoding: L2aEncoding,
    workers: int = 1,
    max_pending_shards: int | None = None,
) -> dict[str, object]:
    """source recordからL2a datasetをmaterializeし、検証済みmanifestを返す。

    payloadはstaging fileへ逐次追記し、公開前後の検証も逐次readで行う。
    `workers`が2以上なら半荘をshardとしてprocess並列に変換する（出力は同一）。
    失敗時はstagingを破棄し、destinationを作らない。
    """
    if not isinstance(source, (PlayerSafeSourceRecord, StreamingSourceRecord)):
        raise DatasetError("source must be a PlayerSafeSourceRecord")
    if not isinstance(encoding, L2aEncoding):
        raise DatasetError("encoding must be an L2aEncoding")
    if type(workers) is not int or workers < 1:
        raise DatasetError("workers must be a positive integer")
    if max_pending_shards is None:
        max_pending_shards = 2 * workers
    if type(max_pending_shards) is not int or max_pending_shards < 1:
        raise DatasetError("max_pending_shards must be a positive integer")
    if workers > 1 and not isinstance(source, StreamingSourceRecord):
        raise DatasetError("parallel L2a materialization requires open_source_record()")
    if source.decision_count == 0:
        raise DatasetError("source record contains no decisions")
    if source.allocation_bindings is None and source.purpose != DEVELOPMENT_PURPOSE:
        raise DatasetError(
            "L2a dataset materialization requires Arena allocation provenance "
            "unless the source purpose is DEVELOPMENT"
        )

    destination = Path(destination)
    with staged_publication(destination, DatasetError) as staging:
        split_counts: dict[str, dict[str, int]] = {}
        with _payload_streams(staging) as streams:
            if workers == 1:
                decision_count, candidate_total = _write_decisions(
                    source.decisions(), encoding, streams, split_counts
                )
            else:
                decision_count, candidate_total = _write_shards(
                    source,
                    encoding,
                    staging,
                    streams,
                    split_counts,
                    workers=workers,
                    max_pending_shards=max_pending_shards,
                )
        if decision_count == 0:
            raise DatasetError("source record contains no scorer decisions")

        manifest = seal(
            {
                "dataset_schema": L2A_DATASET_SCHEMA,
                "encoding": encoding.block(),
                "feature": feature_block(),
                "files": {
                    CANDIDATES_FILENAME: {
                        **file_digest(staging / CANDIDATES_FILENAME),
                        "columns": encoding.dimension,
                        "dtype": _FLOAT_DTYPE,
                        "rows": candidate_total,
                    },
                    CONTEXT_FILENAME: {
                        **file_digest(staging / CONTEXT_FILENAME),
                        "columns": FEATURE_DIMENSION,
                        "dtype": _FLOAT_DTYPE,
                        "rows": decision_count,
                    },
                    DECISIONS_FILENAME: {
                        **file_digest(staging / DECISIONS_FILENAME),
                        "columns": 1,
                        "dtype": _ROWS_DTYPE,
                        "rows": decision_count,
                    },
                },
                "kind": L2A_DATASET_KIND,
                "label": l2a_label_block(encoding),
                "rows": {
                    "candidates": candidate_total,
                    "scorer_decisions": decision_count,
                    "source_decisions": source.decision_count,
                    "splits": dict(sorted(split_counts.items())),
                },
                "source": source.provenance(),
            }
        )
        with appended_new_text(staging / MANIFEST_FILENAME, DatasetError) as stream:
            stream.write(canonical_json_text(manifest))
        published = verify_l2a_dataset(staging)

    republished = verify_l2a_dataset(destination)
    if republished["identity"] != published["identity"]:
        raise DatasetError("dataset readback identity mismatch after publication")
    return republished


def _manifest_encoding(body: Mapping[str, object]) -> L2aEncoding:
    block = body["encoding"]
    if type(block) is not dict or "name" not in block:
        raise DatasetError("dataset encoding block is invalid")
    try:
        encoding = l2a_encoding(block["name"])
    except CandidateFeatureError as exc:
        raise DatasetError(str(exc)) from exc
    if block != encoding.block():
        raise DatasetError(
            "dataset encoding / classification identity or fingerprint mismatch"
        )
    return encoding


def _read_manifest(root: Path) -> tuple[dict[str, object], L2aEncoding]:
    manifest_path = root / MANIFEST_FILENAME
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"dataset manifest cannot be read: {manifest_path}") from exc
    manifest = parse_json_text(text, DatasetError, "dataset manifest")
    body = unseal(manifest, DatasetError, "dataset manifest")
    if text != canonical_json_text(manifest):
        raise DatasetError("dataset manifest is not canonical JSON")
    expect_object(body, _MANIFEST_FIELDS, DatasetError, "dataset manifest")
    if body["dataset_schema"] != L2A_DATASET_SCHEMA:
        raise DatasetError(
            f"unsupported L2a dataset schema: {body['dataset_schema']!r}; "
            f"this implementation provides {L2A_DATASET_SCHEMA!r}"
        )
    if body["kind"] != L2A_DATASET_KIND:
        raise DatasetError("dataset kind mismatch")
    if body["feature"] != feature_block():
        raise DatasetError("dataset shared feature identity/fingerprint mismatch")
    encoding = _manifest_encoding(body)
    if body["label"] != l2a_label_block(encoding):
        raise DatasetError("dataset label semantics mismatch")
    source = validate_source_block(body["source"], context="manifest.source")

    rows = expect_object(body["rows"], _ROWS_FIELDS, DatasetError, "manifest.rows")
    for field in ("candidates", "scorer_decisions", "source_decisions"):
        expect_non_negative_int(rows[field], DatasetError, f"manifest.rows.{field}")
    if rows["source_decisions"] != source["decision_count"]:
        raise DatasetError("manifest source decision count mismatch")
    population_splits: dict[str, int] = {}
    for entry in source["population"]:
        population_splits[entry["split"]] = (
            population_splits.get(entry["split"], 0) + entry["decision_count"]
        )
    splits = rows["splits"]
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
        if sum(item[field] for field in _KIND_FIELDS) != item["source_decisions"]:
            raise DatasetError(f"{context} classification accounting mismatch")
        if (
            item["scorer_decisions"] + item["single_candidate"]
            != item[encoding.head.value]
        ):
            raise DatasetError(f"{context} scorer decision accounting mismatch")
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
        CANDIDATES_FILENAME: (rows["candidates"], encoding.dimension, _FLOAT_DTYPE),
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
        if name != DECISIONS_FILENAME and item["bytes"] != (
            item["rows"] * item["columns"] * _FLOAT32_BYTES
        ):
            raise DatasetError(f"{context} has an unexpected byte length")
    return manifest, encoding


def _open_checked(root: Path) -> tuple[dict[str, object], L2aEncoding]:
    if not root.is_dir():
        raise DatasetError(f"dataset directory does not exist: {root}")
    manifest, encoding = _read_manifest(root)
    expected = {
        MANIFEST_FILENAME,
        DECISIONS_FILENAME,
        CONTEXT_FILENAME,
        CANDIDATES_FILENAME,
    }
    if {child.name for child in root.iterdir()} != expected:
        raise DatasetError("missing/unexpected dataset files")
    files = manifest["files"]
    for name in (DECISIONS_FILENAME, CONTEXT_FILENAME, CANDIDATES_FILENAME):
        digest = file_digest(root / name)
        if (
            digest["bytes"] != files[name]["bytes"]
            or digest["sha256"] != files[name]["sha256"]
        ):
            raise DatasetError(f"dataset payload size/digest mismatch: {name}")
    return manifest, encoding


def _iter_rows(
    path: Path, manifest: Mapping[str, object], encoding: L2aEncoding
) -> Iterator[L2aDecisionRow]:
    """decisions.jsonlを1行ずつstrict readする。最後まで列挙して勘定を照合する。"""
    population = {
        entry["game_ordinal"]: entry for entry in manifest["source"]["population"]
    }
    row_count = 0
    candidate_total = 0
    last_position = (-1, -1)
    observed: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="\n") as stream:
        for index, line in enumerate(stream):
            context = f"decisions[{index}]"
            value = parse_json_text(line, DatasetError, context)
            if line != canonical_json_line(value):
                raise DatasetError(f"{context} is not canonical JSON")
            expect_object(value, _DECISION_FIELDS, DatasetError, context)
            for field in _DECISION_FIELDS - {"candidates", "split"}:
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
            actions = tuple(
                parse_action(item, DatasetError, f"{context}.candidates[{ordinal}]")
                for ordinal, item in enumerate(
                    expect_list(
                        value["candidates"], DatasetError, f"{context}.candidates"
                    )
                )
            )
            if len(actions) < 2 or value["candidate_count"] != len(actions):
                raise DatasetError(f"{context} candidate count mismatch")
            if len(set(actions)) != len(actions):
                raise DatasetError(f"{context} candidates are not unique")
            if any(action.actor != actor_seat for action in actions):
                raise DatasetError(f"{context} candidate actor does not match")
            if value["candidate_offset"] != candidate_total:
                raise DatasetError(f"{context} candidate offset is not contiguous")
            if value["teacher_candidate_index"] >= len(actions):
                raise DatasetError(f"{context} teacher candidate index is out of range")
            candidate_total += len(actions)
            counts = observed.setdefault(
                value["split"], {"candidates": 0, "scorer_decisions": 0}
            )
            counts["candidates"] += len(actions)
            counts["scorer_decisions"] += 1
            row_count += 1
            yield L2aDecisionRow(
                game_ordinal=value["game_ordinal"],
                seed=value["seed"],
                split=value["split"],
                step_ordinal=value["step_ordinal"],
                decision_ordinal=value["decision_ordinal"],
                actor_seat=actor_seat,
                candidate_offset=value["candidate_offset"],
                teacher_candidate_index=value["teacher_candidate_index"],
                actions=actions,
            )

    rows = manifest["rows"]
    if row_count != rows["scorer_decisions"]:
        raise DatasetError("dataset decision count mismatch")
    if candidate_total != rows["candidates"]:
        raise DatasetError("dataset candidate count mismatch")
    for name, counts in rows["splits"].items():
        seen = observed.get(name, {"candidates": 0, "scorer_decisions": 0})
        if (
            seen["candidates"] != counts["candidates"]
            or seen["scorer_decisions"] != counts["scorer_decisions"]
        ):
            raise DatasetError("dataset split membership accounting mismatch")


def _check_finite(path: Path) -> None:
    chunk_bytes = _VERIFY_CHUNK_FLOATS * _FLOAT32_BYTES
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            if not all(map(isfinite, _float32_array(chunk))):
                raise DatasetError(
                    f"dataset payload contains a non-finite value: {path.name}"
                )


def verify_l2a_dataset(path: str | Path) -> dict[str, object]:
    """`read_l2a_dataset()`と同じ照合を逐次readで行い、manifestを返す。"""
    root = Path(path)
    manifest, encoding = _open_checked(root)
    for _row in _iter_rows(root / DECISIONS_FILENAME, manifest, encoding):
        pass
    _check_finite(root / CONTEXT_FILENAME)
    _check_finite(root / CANDIDATES_FILENAME)
    return manifest


def read_l2a_dataset(path: str | Path) -> L2aDataset:
    """L2a dataset directoryをstrict readする。1つでも合わなければfail closed。"""
    root = Path(path)
    manifest, encoding = _open_checked(root)
    rows = tuple(_iter_rows(root / DECISIONS_FILENAME, manifest, encoding))
    context = _float32_array((root / CONTEXT_FILENAME).read_bytes())
    candidates = _float32_array((root / CANDIDATES_FILENAME).read_bytes())
    if not all(map(isfinite, context)) or not all(map(isfinite, candidates)):
        raise DatasetError("dataset payload contains a non-finite value")
    return L2aDataset(
        identity=manifest["identity"],
        manifest=manifest,
        encoding=encoding,
        rows=rows,
        context=context,
        candidates=candidates,
    )


__all__ = [
    "L2A_DATASET_KIND",
    "L2A_DATASET_SCHEMA",
    "L2A_LABEL_SEMANTICS",
    "L2A_TRAINING_OBJECTIVE",
    "L2aDataset",
    "L2aDecisionRow",
    "l2a_label_block",
    "publish_l2a_dataset",
    "read_l2a_dataset",
    "resolve_teacher_index",
    "verify_l2a_dataset",
]
