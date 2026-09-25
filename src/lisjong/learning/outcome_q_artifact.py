"""L0.3 outcome-Q scorer artifactの生成とstrict load（Issue #200、#79 §9）。

#189 candidate scorer artifact（`CANDIDATE_ARTIFACT_SCHEMA`）とは別の
purpose-specific artifactであり、#189 / #191のartifactとidentityは変更しない。
1 artifact = 1 immutable directoryで、repository外のoperator-owned成果物である。

```text
<artifact>/
    manifest.json   canonical JSON identity / provenance / config / digest
    weights.f32     parameter_layout順のlittle-endian float32 payload
```

manifestがbindするもの（#79 §9の最小binding）。

```text
source           outcome source provenance（schema / identity / SCIENTIFIC role /
                 behavior / allocation bindings / population / source_contract digest）
rows             TRAIN / SELECT row / hanchan数とrow identity（#200 DP-3）
feature          #184 shared feature identity / dimension / fingerprint
encoding         #187 candidate feature / #189 encoding identity / fingerprint /
                 second-step request policy
selection_policy #191 semantic envelope identity
objective        target identity / objective identity / checkpoint rule
model            #189と同じcandidate-scorer MLP family（hidden width 64固定）
training         optimizer / hyperparameter / seed / selected epoch / framework /
                 diagnostics
files            weights payloadのSHA-256とbyte長
lisjong          package version / installed source digest
```

既存pathを上書きせず、callable / factory / pickle / optimizer stateを保存・
復元しない。identity / fingerprint / configの不整合はload時にfail closedする。
weights serialization、training block validation、lisjong provenanceは
#184 / #189と同じprimitiveを再利用する。
"""

from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from lisjong.learning._canonical import (
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
from lisjong.learning.artifact import (
    _lisjong_block,
    _validate_training,
    _weights_array,
    _weights_bytes,
)
from lisjong.learning.candidate_encoding import encoding_block
from lisjong.learning.candidate_model import DEFAULT_HIDDEN_WIDTH, CandidateScorerConfig
from lisjong.learning.dataset import feature_block
from lisjong.learning.envelope_policy import SEMANTIC_ENVELOPE_IDENTITY
from lisjong.learning.errors import ModelArtifactError, OutcomeSourceError
from lisjong.learning.model import ModelParameter
from lisjong.learning.outcome_q_dataset import (
    OUTCOME_Q_ROWS_IDENTITY,
    OUTCOME_Q_SPLITS,
    SELECT_SPLIT,
    TRAIN_SPLIT,
    OutcomeQRows,
)
from lisjong.learning.outcome_source import (
    _SUPPORTED_SCHEMAS,
    EXPECTED_BEHAVIOR,
    OUTCOME_OBJECTIVE_IDENTITY,
    OUTCOME_TARGET_IDENTITY,
    SCIENTIFIC_ROLE,
    _read_allocation_bindings,
)
from lisjong.learning.training import OPTIMIZER

OUTCOME_Q_ARTIFACT_SCHEMA = "lisjong-offense-l0.3-outcome-q-artifact-v1"
"""このoutcome-Q scorer artifact contractのidentity。"""

OUTCOME_Q_ARTIFACT_KIND = "outcome-q-scorer-model"

OUTCOME_Q_CHECKPOINT_RULE = "lisjong-offense-l0.3-min-select-mse-earliest-epoch-v1"
"""SELECT selected-action MSEが最小のepoch（同値は最初のepoch）を採用する。

#200 DP-2: #79でruleの文言はまだfreezeされていない。real training前に確認する。
"""

OUTCOME_Q_MODEL = CandidateScorerConfig(hidden_width=DEFAULT_HIDDEN_WIDTH)
"""#79 §5: #189と同じcandidate-scorer MLP family、hidden width 64。"""

MANIFEST_FILENAME = "manifest.json"
WEIGHTS_FILENAME = "weights.f32"

_WEIGHTS_DTYPE = "float32-le"
_FLOAT32_BYTES = 4

_MANIFEST_FIELDS = frozenset(
    {
        "artifact_schema",
        "encoding",
        "feature",
        "files",
        "kind",
        "lisjong",
        "model",
        "objective",
        "parameters",
        "rows",
        "selection_policy",
        "source",
        "training",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "allocation_bindings",
        "behavior",
        "identity",
        "population",
        "population_role",
        "schema",
        "source_contract_digest",
    }
)
_POPULATION_FIELDS = frozenset({"focal_seat", "game_ordinal", "seed", "split"})
_ROWS_FIELDS = frozenset(
    {"candidates", "contract", "eligible_rows", "hanchan", "identity", "splits"}
)
_LISJONG_FIELDS = frozenset({"package_version", "source_digest"})
_PARAMETER_FIELDS = frozenset({"count", "name", "offset", "shape"})
_WEIGHTS_FIELDS = frozenset({"bytes", "count", "dtype", "sha256"})


def objective_block() -> dict[str, str]:
    """target / objective / checkpoint ruleのidentity block。"""
    return {
        "checkpoint_rule": OUTCOME_Q_CHECKPOINT_RULE,
        "objective": OUTCOME_OBJECTIVE_IDENTITY,
        "target": OUTCOME_TARGET_IDENTITY,
    }


@dataclass(frozen=True, slots=True)
class LoadedOutcomeQArtifact:
    """strict loadしたoutcome-Q scorer artifact。"""

    identity: str
    manifest: Mapping[str, object]
    model_config: CandidateScorerConfig
    parameters: tuple[ModelParameter, ...]
    weights: array

    @property
    def source_identity(self) -> str:
        return self.manifest["source"]["identity"]

    @property
    def rows_identity(self) -> str:
        return self.manifest["rows"]["identity"]

    @property
    def selected_epoch(self) -> int:
        return self.manifest["training"]["selected_epoch"]

    @property
    def diagnostics(self) -> Mapping[str, float | int]:
        return self.manifest["training"]["diagnostics"]


def _validate_outcome_training(value: object) -> dict[str, object]:
    training = _validate_training(value)
    if training["objective"] != OUTCOME_OBJECTIVE_IDENTITY:
        raise ModelArtifactError("artifact training objective mismatch")
    if training["optimizer"] != OPTIMIZER:
        raise ModelArtifactError("artifact training optimizer mismatch")
    if training["train_splits"] != [TRAIN_SPLIT]:
        raise ModelArtifactError("artifact training.train_splits must be [TRAIN]")
    if training["validation_splits"] != [SELECT_SPLIT]:
        raise ModelArtifactError("artifact training.validation_splits must be [SELECT]")
    if not 1 <= training["epochs"] or not 1 <= training["batch_size"]:
        raise ModelArtifactError("artifact training budget must be positive")
    if not 1 <= training["selected_epoch"] <= training["epochs"]:
        raise ModelArtifactError("artifact selected epoch is outside the epoch budget")
    return training


def _validate_source(value: object) -> dict[str, int]:
    """source provenanceを検証し、splitごとのhanchan数を返す。"""
    source = expect_object(value, _SOURCE_FIELDS, ModelArtifactError, "artifact.source")
    schema = source["schema"]
    if type(schema) is not str or schema not in _SUPPORTED_SCHEMAS:
        raise ModelArtifactError("artifact.source.schema is not a supported source")
    for field in ("identity", "source_contract_digest"):
        expect_digest(source[field], ModelArtifactError, f"artifact.source.{field}")
    if source["population_role"] != SCIENTIFIC_ROLE:
        raise ModelArtifactError("artifact source must be a SCIENTIFIC population")
    if source["behavior"] != EXPECTED_BEHAVIOR:
        raise ModelArtifactError("artifact source behavior identity mismatch")

    population = expect_list(
        source["population"], ModelArtifactError, "artifact.source.population"
    )
    seeds_by_split: dict[str, list[int]] = {}
    for ordinal, game in enumerate(population):
        context = f"artifact.source.population[{ordinal}]"
        expect_object(game, _POPULATION_FIELDS, ModelArtifactError, context)
        seed = expect_non_negative_int(game["seed"], ModelArtifactError, context)
        if game["game_ordinal"] != ordinal or game["focal_seat"] != ordinal % 4:
            raise ModelArtifactError(f"{context} ordinal / focal seat mismatch")
        if game["split"] not in OUTCOME_Q_SPLITS:
            raise ModelArtifactError(f"{context}.split is not TRAIN / SELECT")
        seeds_by_split.setdefault(game["split"], []).append(seed)
    seeds = [seed for values in seeds_by_split.values() for seed in values]
    if len(set(seeds)) != len(seeds):
        raise ModelArtifactError("artifact source population reuses a seed")
    try:
        _read_allocation_bindings(
            source["allocation_bindings"],
            schema=schema,
            population_role=SCIENTIFIC_ROLE,
            seeds_by_split=seeds_by_split,
        )
    except OutcomeSourceError as exc:
        raise ModelArtifactError(
            f"artifact source allocation bindings are invalid: {exc}"
        ) from exc
    return {split: len(seeds_by_split.get(split, ())) for split in OUTCOME_Q_SPLITS}


def _validate_rows(value: object, hanchan: dict[str, int]) -> dict[str, object]:
    rows = expect_object(value, _ROWS_FIELDS, ModelArtifactError, "artifact.rows")
    if rows["contract"] != OUTCOME_Q_ROWS_IDENTITY:
        raise ModelArtifactError("artifact rows contract mismatch")
    expect_digest(rows["identity"], ModelArtifactError, "artifact.rows.identity")
    for field in ("candidates", "eligible_rows"):
        expect_non_negative_int(
            rows[field], ModelArtifactError, f"artifact.rows.{field}"
        )
    if rows["hanchan"] != hanchan:
        raise ModelArtifactError("artifact rows hanchan counts contradict the source")
    splits = expect_object(
        rows["splits"],
        frozenset(OUTCOME_Q_SPLITS),
        ModelArtifactError,
        "artifact.rows.splits",
    )
    for split in OUTCOME_Q_SPLITS:
        if (
            expect_non_negative_int(
                splits[split], ModelArtifactError, f"artifact.rows.splits.{split}"
            )
            == 0
        ):
            raise ModelArtifactError(f"artifact rows have no {split} row")
    if sum(splits.values()) != rows["eligible_rows"]:
        raise ModelArtifactError("artifact rows split accounting mismatch")
    if rows["candidates"] < rows["eligible_rows"]:
        raise ModelArtifactError("artifact rows candidate accounting mismatch")
    return rows


def write_outcome_q_artifact(
    destination: str | Path,
    *,
    rows: OutcomeQRows,
    training: Mapping[str, object],
    weights: Sequence[float],
) -> LoadedOutcomeQArtifact:
    """artifactをimmutableに書き出し、strict loadした結果を返す。"""
    if not isinstance(rows, OutcomeQRows):
        raise ModelArtifactError("rows must be materialized OutcomeQRows")
    if len(weights) != OUTCOME_Q_MODEL.parameter_count:
        raise ModelArtifactError("weights length does not match the model layout")
    if any(not isfinite(value) for value in weights):
        raise ModelArtifactError("weights must be finite")

    training_value = _validate_outcome_training(dict(training))
    destination = Path(destination)
    with staged_publication(destination, ModelArtifactError) as staging:
        weights_path = staging / WEIGHTS_FILENAME
        write_new_bytes(weights_path, _weights_bytes(weights), ModelArtifactError)
        manifest = seal(
            {
                "artifact_schema": OUTCOME_Q_ARTIFACT_SCHEMA,
                "encoding": encoding_block(),
                "feature": feature_block(),
                "files": {
                    WEIGHTS_FILENAME: {
                        **file_digest(weights_path),
                        "count": len(weights),
                        "dtype": _WEIGHTS_DTYPE,
                    }
                },
                "kind": OUTCOME_Q_ARTIFACT_KIND,
                "lisjong": _lisjong_block(),
                "model": OUTCOME_Q_MODEL.to_value(),
                "objective": objective_block(),
                "parameters": [
                    parameter.to_value()
                    for parameter in OUTCOME_Q_MODEL.parameter_layout()
                ],
                "rows": rows.to_value(),
                "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
                "source": rows.source.provenance(),
                "training": training_value,
            }
        )
        write_new_text(
            staging / MANIFEST_FILENAME,
            canonical_json_text(manifest),
            ModelArtifactError,
        )
        staged = load_outcome_q_artifact(staging)

    published = load_outcome_q_artifact(destination)
    if published.identity != staged.identity:
        raise ModelArtifactError(
            "artifact readback identity mismatch after publication"
        )
    return published


def load_outcome_q_artifact(path: str | Path) -> LoadedOutcomeQArtifact:
    """artifact directoryをstrict loadする。1つでも合わなければfail closedする。"""
    root = Path(path)
    if not root.is_dir():
        raise ModelArtifactError(f"model artifact directory does not exist: {root}")
    manifest_path = root / MANIFEST_FILENAME
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelArtifactError(
            f"model artifact manifest cannot be read: {manifest_path}"
        ) from exc
    manifest = parse_json_text(text, ModelArtifactError, "artifact manifest")
    body = unseal(manifest, ModelArtifactError, "artifact manifest")
    if text != canonical_json_text(manifest):
        raise ModelArtifactError("artifact manifest is not canonical JSON")
    expect_object(body, _MANIFEST_FIELDS, ModelArtifactError, "artifact manifest")

    if body["artifact_schema"] != OUTCOME_Q_ARTIFACT_SCHEMA:
        raise ModelArtifactError(
            f"unsupported model artifact schema: {body['artifact_schema']!r}; "
            f"this implementation provides {OUTCOME_Q_ARTIFACT_SCHEMA!r}"
        )
    if body["kind"] != OUTCOME_Q_ARTIFACT_KIND:
        raise ModelArtifactError("model artifact kind mismatch")
    if {child.name for child in root.iterdir()} != {
        MANIFEST_FILENAME,
        WEIGHTS_FILENAME,
    }:
        raise ModelArtifactError("missing/unexpected model artifact files")

    if body["feature"] != feature_block():
        raise ModelArtifactError(
            "artifact shared feature identity/fingerprint mismatch"
        )
    if body["encoding"] != encoding_block():
        raise ModelArtifactError(
            "artifact candidate encoding / candidate feature / second-step request "
            "policy identity mismatch"
        )
    if body["selection_policy"] != SEMANTIC_ENVELOPE_IDENTITY:
        raise ModelArtifactError("artifact selection policy identity mismatch")
    if body["objective"] != objective_block():
        raise ModelArtifactError(
            "artifact target / objective / checkpoint rule identity mismatch"
        )

    hanchan = _validate_source(body["source"])
    _validate_rows(body["rows"], hanchan)
    _validate_outcome_training(body["training"])

    lisjong_block = expect_object(
        body["lisjong"], _LISJONG_FIELDS, ModelArtifactError, "artifact.lisjong"
    )
    expect_str(
        lisjong_block["package_version"],
        ModelArtifactError,
        "artifact.lisjong.package_version",
    )
    expect_digest(
        lisjong_block["source_digest"],
        ModelArtifactError,
        "artifact.lisjong.source_digest",
    )

    model_config = CandidateScorerConfig.from_value(body["model"])
    if model_config != OUTCOME_Q_MODEL:
        raise ModelArtifactError("artifact model is not the frozen outcome-Q model")
    layout = model_config.parameter_layout()
    declared = expect_list(
        body["parameters"], ModelArtifactError, "artifact.parameters"
    )
    if len(declared) != len(layout):
        raise ModelArtifactError("artifact parameter layout mismatch")
    for index, (value, parameter) in enumerate(zip(declared, layout, strict=True)):
        expect_object(
            value,
            _PARAMETER_FIELDS,
            ModelArtifactError,
            f"artifact.parameters[{index}]",
        )
        if value != parameter.to_value():
            raise ModelArtifactError(
                f"artifact.parameters[{index}] does not match the model layout"
            )

    files = expect_object(
        body["files"],
        frozenset({WEIGHTS_FILENAME}),
        ModelArtifactError,
        "artifact.files",
    )
    weights_block = expect_object(
        files[WEIGHTS_FILENAME],
        _WEIGHTS_FIELDS,
        ModelArtifactError,
        "artifact.files[weights]",
    )
    expect_digest(
        weights_block["sha256"], ModelArtifactError, "artifact.files[weights].sha256"
    )
    if weights_block["dtype"] != _WEIGHTS_DTYPE:
        raise ModelArtifactError("artifact weights dtype mismatch")
    if weights_block["count"] != model_config.parameter_count:
        raise ModelArtifactError("artifact weights count does not match the layout")
    if weights_block["bytes"] != model_config.parameter_count * _FLOAT32_BYTES:
        raise ModelArtifactError("artifact weights byte length mismatch")

    weights_path = root / WEIGHTS_FILENAME
    digest = file_digest(weights_path)
    if (
        digest["bytes"] != weights_block["bytes"]
        or digest["sha256"] != weights_block["sha256"]
    ):
        raise ModelArtifactError("artifact weights size/digest mismatch")
    weights = _weights_array(weights_path.read_bytes())
    if len(weights) != model_config.parameter_count:
        raise ModelArtifactError("artifact weights payload length mismatch")
    if any(not isfinite(value) for value in weights):
        raise ModelArtifactError("artifact weights contain a non-finite value")

    return LoadedOutcomeQArtifact(
        identity=manifest["identity"],
        manifest=manifest,
        model_config=model_config,
        parameters=layout,
        weights=weights,
    )


__all__ = [
    "MANIFEST_FILENAME",
    "OUTCOME_Q_ARTIFACT_KIND",
    "OUTCOME_Q_ARTIFACT_SCHEMA",
    "OUTCOME_Q_CHECKPOINT_RULE",
    "OUTCOME_Q_MODEL",
    "WEIGHTS_FILENAME",
    "LoadedOutcomeQArtifact",
    "load_outcome_q_artifact",
    "objective_block",
    "write_outcome_q_artifact",
]
