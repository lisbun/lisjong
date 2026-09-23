"""L0.2 candidate scorer artifactの生成とstrict load。

Issue #189に対応する。Issue #184の`MODEL_ARTIFACT_SCHEMA`とは別の
purpose-specific artifactであり、既存artifactを変更しない。1 artifact =
1 immutable directoryで、repository外のoperator-owned成果物として扱う。

```text
<artifact>/
    manifest.json   canonical JSON identity / provenance / config / digest
    weights.f32     parameter_layout順のlittle-endian float32 payload
```

manifestがbindするもの。

- source identity / provenance（Arena allocation binding含む）
- candidate dataset schema / identity / decision数 / candidate数 / split母数
- #184 shared feature identity / dimension / fingerprint
- #187 candidate feature identity、candidate encoding identity / fingerprint、
  second-step request policy identity
- label semantics / O0 eligibility identity
- model architecture / config / parameter layout
- training config / seed / selected epoch / framework
- weights payloadのSHA-256とbyte長
- lisjong package version / installed source digest（provenance記録）

既存pathを上書きせず、callable / factory / pickle / optimizer stateを保存・
復元しない。identity / fingerprint / configの不整合はload時にfail closedする。
weights serialization、training block validation、lisjong provenanceは
Issue #184のmodel artifactと同じprimitiveを再利用する。
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
from lisjong.learning.candidate_dataset import (
    CANDIDATE_DATASET_SCHEMA,
    CANDIDATE_TRAINING_OBJECTIVE,
    CandidateDataset,
    candidate_label_block,
)
from lisjong.learning.candidate_encoding import encoding_block
from lisjong.learning.candidate_model import CandidateScorerConfig
from lisjong.learning.dataset import feature_block, validate_source_block
from lisjong.learning.errors import DatasetError, ModelArtifactError
from lisjong.learning.model import ModelParameter

CANDIDATE_ARTIFACT_SCHEMA = "lisjong-offense-l0.2-candidate-scorer-artifact-v1"
"""このcandidate scorer artifact contractのidentity。"""

CANDIDATE_ARTIFACT_KIND = "candidate-scorer-model"
MANIFEST_FILENAME = "manifest.json"
WEIGHTS_FILENAME = "weights.f32"

_WEIGHTS_DTYPE = "float32-le"
_FLOAT32_BYTES = 4

_MANIFEST_FIELDS = frozenset(
    {
        "artifact_schema",
        "dataset",
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
    }
)
_DATASET_FIELDS = frozenset(
    {"candidates", "identity", "schema", "scorer_decisions", "splits"}
)
_LISJONG_FIELDS = frozenset({"package_version", "source_digest"})
_PARAMETER_FIELDS = frozenset({"count", "name", "offset", "shape"})
_WEIGHTS_FIELDS = frozenset({"bytes", "count", "dtype", "sha256"})


@dataclass(frozen=True, slots=True)
class LoadedCandidateScorerArtifact:
    """strict loadしたcandidate scorer artifact。"""

    identity: str
    manifest: Mapping[str, object]
    model_config: CandidateScorerConfig
    parameters: tuple[ModelParameter, ...]
    weights: array

    @property
    def dataset_identity(self) -> str:
        return self.manifest["dataset"]["identity"]

    @property
    def source_identity(self) -> str:
        return self.manifest["source"]["identity"]


def _dataset_block(dataset: CandidateDataset) -> dict[str, object]:
    return {
        "candidates": dataset.candidate_count,
        "identity": dataset.identity,
        "schema": CANDIDATE_DATASET_SCHEMA,
        "scorer_decisions": dataset.decision_count,
        "splits": dict(sorted(dataset.split_counts().items())),
    }


def _validate_candidate_training(value: object) -> dict[str, object]:
    training = _validate_training(value)
    if training["objective"] != CANDIDATE_TRAINING_OBJECTIVE:
        raise ModelArtifactError("artifact training objective mismatch")
    if not 1 <= training["selected_epoch"] <= training["epochs"]:
        raise ModelArtifactError("artifact selected epoch is outside the epoch budget")
    return training


def write_candidate_artifact(
    destination: str | Path,
    *,
    dataset: CandidateDataset,
    model_config: CandidateScorerConfig,
    training: Mapping[str, object],
    weights: Sequence[float],
) -> LoadedCandidateScorerArtifact:
    """artifactをimmutableに書き出し、strict loadした結果を返す。"""
    if not isinstance(dataset, CandidateDataset):
        raise ModelArtifactError("dataset must be a strict-read CandidateDataset")
    if not isinstance(model_config, CandidateScorerConfig):
        raise ModelArtifactError("model_config must be a CandidateScorerConfig")
    if len(weights) != model_config.parameter_count:
        raise ModelArtifactError("weights length does not match the model layout")
    if any(not isfinite(value) for value in weights):
        raise ModelArtifactError("weights must be finite")

    training_value = _validate_candidate_training(dict(training))
    destination = Path(destination)
    with staged_publication(destination, ModelArtifactError) as staging:
        weights_path = staging / WEIGHTS_FILENAME
        write_new_bytes(weights_path, _weights_bytes(weights), ModelArtifactError)
        manifest = seal(
            {
                "artifact_schema": CANDIDATE_ARTIFACT_SCHEMA,
                "dataset": _dataset_block(dataset),
                "encoding": encoding_block(),
                "feature": feature_block(),
                "files": {
                    WEIGHTS_FILENAME: {
                        **file_digest(weights_path),
                        "count": len(weights),
                        "dtype": _WEIGHTS_DTYPE,
                    }
                },
                "kind": CANDIDATE_ARTIFACT_KIND,
                "label": candidate_label_block(),
                "lisjong": _lisjong_block(),
                "model": model_config.to_value(),
                "parameters": [
                    parameter.to_value()
                    for parameter in model_config.parameter_layout()
                ],
                "source": dict(dataset.manifest["source"]),
                "training": training_value,
            }
        )
        write_new_text(
            staging / MANIFEST_FILENAME,
            canonical_json_text(manifest),
            ModelArtifactError,
        )
        staged = load_candidate_artifact(staging)

    published = load_candidate_artifact(destination)
    if published.identity != staged.identity:
        raise ModelArtifactError(
            "artifact readback identity mismatch after publication"
        )
    return published


def load_candidate_artifact(path: str | Path) -> LoadedCandidateScorerArtifact:
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

    if body["artifact_schema"] != CANDIDATE_ARTIFACT_SCHEMA:
        raise ModelArtifactError(
            f"unsupported model artifact schema: {body['artifact_schema']!r}; "
            f"this implementation provides {CANDIDATE_ARTIFACT_SCHEMA!r}"
        )
    if body["kind"] != CANDIDATE_ARTIFACT_KIND:
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
    if body["label"] != candidate_label_block():
        raise ModelArtifactError("artifact label semantics mismatch")

    dataset = expect_object(
        body["dataset"], _DATASET_FIELDS, ModelArtifactError, "artifact.dataset"
    )
    if dataset["schema"] != CANDIDATE_DATASET_SCHEMA:
        raise ModelArtifactError("artifact dataset schema mismatch")
    expect_digest(dataset["identity"], ModelArtifactError, "artifact.dataset.identity")
    for field in ("candidates", "scorer_decisions"):
        expect_non_negative_int(
            dataset[field], ModelArtifactError, f"artifact.dataset.{field}"
        )
    splits = dataset["splits"]
    if type(splits) is not dict or not splits:
        raise ModelArtifactError("artifact.dataset.splits must be a non-empty object")
    for name, count in splits.items():
        expect_str(name, ModelArtifactError, "artifact.dataset.splits key")
        expect_non_negative_int(
            count, ModelArtifactError, f"artifact.dataset.splits[{name}]"
        )
    if sum(splits.values()) != dataset["scorer_decisions"]:
        raise ModelArtifactError("artifact dataset split accounting mismatch")

    try:
        validate_source_block(body["source"], context="artifact.source")
    except DatasetError as exc:
        raise ModelArtifactError(
            f"artifact source provenance is invalid: {exc}"
        ) from exc

    training = _validate_candidate_training(body["training"])
    if set(training["train_splits"]) - set(splits) or set(
        training["validation_splits"]
    ) - set(splits):
        raise ModelArtifactError("artifact training splits are not dataset splits")

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

    return LoadedCandidateScorerArtifact(
        identity=manifest["identity"],
        manifest=manifest,
        model_config=model_config,
        parameters=layout,
        weights=weights,
    )


__all__ = [
    "CANDIDATE_ARTIFACT_KIND",
    "CANDIDATE_ARTIFACT_SCHEMA",
    "MANIFEST_FILENAME",
    "WEIGHTS_FILENAME",
    "LoadedCandidateScorerArtifact",
    "load_candidate_artifact",
    "write_candidate_artifact",
]
