"""L0.3 outcome-Q artifactの生成とstrict load（lisjong-project#79 §9）。

#189 candidate scorer artifact（`CANDIDATE_ARTIFACT_SCHEMA`）とは別の
purpose-specific artifactであり、#184 / #189 / #191のartifactとidentityを変更しない。
1 artifact = 1 immutable directoryで、repository外のoperator-owned成果物として扱う。

```text
<artifact>/
    manifest.json   canonical JSON identity / provenance / config / digest
    weights.f32     parameter_layout順のlittle-endian float32 payload
```

manifestがbindするもの（#79 §9の最小集合を含む）。

- outcome source provenance（schema / identity / SCIENTIFIC role / behavior /
  focal rotation population / TRAIN・SELECT allocation binding）
- training set identity / split別row数 / unique eligible kyoku数
- #184 shared feature identity / fingerprint
- #187 candidate feature identity、#189 candidate encoding identity /
  fingerprint、#189 second-step request policy
- #191 semantic envelope identity、O0 eligibility、survivor下限
- outcome target identity / objective identity
- model architecture / config（#189 family、hidden width 64）/ parameter layout
- training config / seed / selected epoch / framework / SELECT diagnostics
- weights payloadのSHA-256とbyte長、lisjong package provenance

既存pathを上書きせず、callable / factory / pickle / optimizer stateを保存・
復元しない。identity / fingerprint / configの不整合はload時にfail closedする。
weights serialization、training block validation、lisjong provenanceは
#184 / #189 artifactと同じprimitiveを再利用する。
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
from lisjong.learning.candidate_model import CandidateScorerConfig
from lisjong.learning.dataset import feature_block
from lisjong.learning.errors import ModelArtifactError, OutcomeSourceError
from lisjong.learning.model import ModelParameter
from lisjong.learning.outcome_q_dataset import (
    OUTCOME_Q_HIDDEN_WIDTH,
    OUTCOME_Q_SPLITS,
    SELECT_SPLIT,
    TRAIN_SPLIT,
    OutcomeQTrainingSet,
    outcome_q_label_block,
)
from lisjong.learning.outcome_source import (
    OUTCOME_OBJECTIVE_IDENTITY,
    SCIENTIFIC_ROLE,
    validate_outcome_source_provenance,
)
from lisjong.learning.training import OPTIMIZER

OUTCOME_Q_ARTIFACT_SCHEMA = "lisjong-offense-l0.3-outcome-q-artifact-v1"
"""このoutcome-Q artifact contractのidentity。"""

OUTCOME_Q_ARTIFACT_KIND = "outcome-q-residual-scorer-model"
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
        "label",
        "lisjong",
        "model",
        "parameters",
        "source",
        "training",
        "training_set",
    }
)
_TRAINING_SET_FIELDS = frozenset(
    {"candidates", "identity", "rows", "splits", "unique_eligible_kyoku"}
)
_LISJONG_FIELDS = frozenset({"package_version", "source_digest"})
_PARAMETER_FIELDS = frozenset({"count", "name", "offset", "shape"})
_WEIGHTS_FIELDS = frozenset({"bytes", "count", "dtype", "sha256"})


@dataclass(frozen=True, slots=True)
class LoadedOutcomeQArtifact:
    """strict loadしたoutcome-Q artifact。"""

    identity: str
    manifest: Mapping[str, object]
    model_config: CandidateScorerConfig
    parameters: tuple[ModelParameter, ...]
    weights: array

    @property
    def source_identity(self) -> str:
        return self.manifest["source"]["identity"]

    @property
    def training_set_identity(self) -> str:
        return self.manifest["training_set"]["identity"]


def _validate_q_training(value: object) -> dict[str, object]:
    training = _validate_training(value)
    if training["objective"] != OUTCOME_OBJECTIVE_IDENTITY:
        raise ModelArtifactError("artifact training objective mismatch")
    if training["optimizer"] != OPTIMIZER:
        raise ModelArtifactError("artifact training optimizer mismatch")
    if training["train_splits"] != [TRAIN_SPLIT] or training["validation_splits"] != [
        SELECT_SPLIT
    ]:
        raise ModelArtifactError("artifact TRAIN / SELECT split roles mismatch")
    if not 1 <= training["selected_epoch"] <= training["epochs"]:
        raise ModelArtifactError("artifact selected epoch is outside the epoch budget")
    if training["batch_size"] < 1:
        raise ModelArtifactError("artifact training batch size must be positive")
    return training


def _validate_split_counts(value: object, context: str) -> dict[str, int]:
    counts = expect_object(
        value, frozenset(OUTCOME_Q_SPLITS), ModelArtifactError, context
    )
    for split, count in counts.items():
        if expect_non_negative_int(count, ModelArtifactError, f"{context}.{split}") < 1:
            raise ModelArtifactError(f"{context}.{split} must be positive")
    return counts


def _validate_source(value: object) -> dict[str, object]:
    try:
        source = validate_outcome_source_provenance(value, context="artifact.source")
    except OutcomeSourceError as exc:
        raise ModelArtifactError(
            f"artifact source provenance is invalid: {exc}"
        ) from exc
    if source["population_role"] != SCIENTIFIC_ROLE:
        raise ModelArtifactError("artifact source is not a SCIENTIFIC outcome source")
    if {entry["split"] for entry in source["population"]} != set(OUTCOME_Q_SPLITS):
        raise ModelArtifactError("artifact source population is not TRAIN + SELECT")
    return source


def write_outcome_q_artifact(
    destination: str | Path,
    *,
    training_set: OutcomeQTrainingSet,
    model_config: CandidateScorerConfig,
    training: Mapping[str, object],
    weights: Sequence[float],
) -> LoadedOutcomeQArtifact:
    """artifactをimmutableに書き出し、strict loadした結果を返す。"""
    if not isinstance(training_set, OutcomeQTrainingSet):
        raise ModelArtifactError("training_set must be an OutcomeQTrainingSet")
    if not isinstance(model_config, CandidateScorerConfig):
        raise ModelArtifactError("model_config must be a CandidateScorerConfig")
    if len(weights) != model_config.parameter_count:
        raise ModelArtifactError("weights length does not match the model layout")
    if any(not isfinite(value) for value in weights):
        raise ModelArtifactError("weights must be finite")

    training_value = _validate_q_training(dict(training))
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
                "label": outcome_q_label_block(),
                "lisjong": _lisjong_block(),
                "model": model_config.to_value(),
                "parameters": [
                    parameter.to_value()
                    for parameter in model_config.parameter_layout()
                ],
                "source": training_set.source.provenance(),
                "training": training_value,
                "training_set": training_set.block(),
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
    if body["label"] != outcome_q_label_block():
        raise ModelArtifactError(
            "artifact eligibility / selection policy / target / objective mismatch"
        )

    _validate_source(body["source"])
    training_set = expect_object(
        body["training_set"],
        _TRAINING_SET_FIELDS,
        ModelArtifactError,
        "artifact.training_set",
    )
    expect_digest(
        training_set["identity"], ModelArtifactError, "artifact.training_set.identity"
    )
    for field in ("candidates", "rows"):
        expect_non_negative_int(
            training_set[field], ModelArtifactError, f"artifact.training_set.{field}"
        )
    splits = _validate_split_counts(
        training_set["splits"], "artifact.training_set.splits"
    )
    kyokus = _validate_split_counts(
        training_set["unique_eligible_kyoku"],
        "artifact.training_set.unique_eligible_kyoku",
    )
    if sum(splits.values()) != training_set["rows"]:
        raise ModelArtifactError("artifact training set split accounting mismatch")
    if any(kyokus[split] > splits[split] for split in OUTCOME_Q_SPLITS):
        raise ModelArtifactError("artifact training set kyoku accounting mismatch")
    if training_set["candidates"] < 2 * training_set["rows"]:
        raise ModelArtifactError("artifact training set candidate accounting mismatch")

    _validate_q_training(body["training"])

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
    if model_config.hidden_width != OUTCOME_Q_HIDDEN_WIDTH:
        raise ModelArtifactError(
            f"artifact hidden width is not the frozen {OUTCOME_Q_HIDDEN_WIDTH}"
        )
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
    "WEIGHTS_FILENAME",
    "LoadedOutcomeQArtifact",
    "load_outcome_q_artifact",
    "write_outcome_q_artifact",
]
