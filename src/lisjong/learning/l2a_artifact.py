"""L2a候補scorer artifactの生成とstrict load（Issue #316）。

設計`docs/learning-l2a.md` 4.1節。打牌（M0 / M1）と反応（R1）で1つずつ持つ。
1 artifact = 1 immutable directoryで、repository外のoperator-owned成果物である。

```text
<artifact>/
    manifest.json   canonical JSON identity / provenance / config / digest
    weights.f32     parameter_layout順のlittle-endian float32
```

manifestがbindするもの：source identity / provenance、dataset schema /
identity / 件数 / split母数、#184 shared feature、L2a encoding（M0 / M1 / R1）の
identity / fingerprintとhead、guard規則（L2a分類）のidentity、label semantics、
model architecture / config / parameter layout、学習設定（seed・選択epoch・
epoch別のTRAIN / SELECT cross entropyを含むdiagnostics）、weightsのSHA-256。

既存pathを上書きせず、callable / pickle / optimizer stateを保存・復元しない。
identity / fingerprint / configの不整合はload時にfail closedする。weightsの
serialization、学習blockのvalidation、lisjong provenanceはL0.2 artifactと
同じprimitiveを使う。
"""

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
from lisjong.learning.dataset import feature_block, validate_source_block
from lisjong.learning.errors import DatasetError, ModelArtifactError
from lisjong.learning.l2a_dataset import (
    L2A_DATASET_SCHEMA,
    L2A_TRAINING_OBJECTIVE,
    L2aDataset,
    l2a_label_block,
)
from lisjong.learning.l2a_model import L2aScorerConfig
from lisjong.learning.model import ModelParameter

L2A_ARTIFACT_SCHEMA = "lisjong-l2a-candidate-scorer-artifact-v1"
L2A_ARTIFACT_KIND = "l2a-candidate-scorer-model"
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
class LoadedL2aScorerArtifact:
    """strict loadしたL2a scorer artifact。"""

    identity: str
    manifest: Mapping[str, object]
    model_config: L2aScorerConfig
    parameters: tuple[ModelParameter, ...]
    weights: object

    @property
    def encoding(self):
        return self.model_config.encoding

    @property
    def source_identity(self) -> str:
        return self.manifest["source"]["identity"]

    @property
    def dataset_identity(self) -> str:
        return self.manifest["dataset"]["identity"]


def _validate_l2a_training(value: object) -> dict[str, object]:
    training = _validate_training(value)
    if training["objective"] != L2A_TRAINING_OBJECTIVE:
        raise ModelArtifactError("artifact training objective mismatch")
    if not 1 <= training["selected_epoch"] <= training["epochs"]:
        raise ModelArtifactError("artifact selected epoch is outside the epoch budget")
    return training


def write_l2a_artifact(
    destination: str | Path,
    *,
    dataset: L2aDataset,
    model_config: L2aScorerConfig,
    training: Mapping[str, object],
    weights: Sequence[float],
) -> LoadedL2aScorerArtifact:
    """artifactをimmutableに書き出し、strict loadした結果を返す。"""
    if not isinstance(dataset, L2aDataset):
        raise ModelArtifactError("dataset must be a strict-read L2aDataset")
    if not isinstance(model_config, L2aScorerConfig):
        raise ModelArtifactError("model_config must be an L2aScorerConfig")
    if model_config.encoding is not dataset.encoding:
        raise ModelArtifactError("model encoding does not match the dataset encoding")
    if len(weights) != model_config.parameter_count:
        raise ModelArtifactError("weights length does not match the model layout")
    if any(not isfinite(value) for value in weights):
        raise ModelArtifactError("weights must be finite")

    training_value = _validate_l2a_training(dict(training))
    destination = Path(destination)
    with staged_publication(destination, ModelArtifactError) as staging:
        weights_path = staging / WEIGHTS_FILENAME
        write_new_bytes(weights_path, _weights_bytes(weights), ModelArtifactError)
        manifest = seal(
            {
                "artifact_schema": L2A_ARTIFACT_SCHEMA,
                "dataset": {
                    "candidates": dataset.candidate_count,
                    "identity": dataset.identity,
                    "schema": L2A_DATASET_SCHEMA,
                    "scorer_decisions": dataset.decision_count,
                    "splits": dict(sorted(dataset.split_counts().items())),
                },
                "encoding": dataset.encoding.block(),
                "feature": feature_block(),
                "files": {
                    WEIGHTS_FILENAME: {
                        **file_digest(weights_path),
                        "count": len(weights),
                        "dtype": _WEIGHTS_DTYPE,
                    }
                },
                "kind": L2A_ARTIFACT_KIND,
                "label": l2a_label_block(dataset.encoding),
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
        staged = load_l2a_artifact(staging)

    published = load_l2a_artifact(destination)
    if published.identity != staged.identity:
        raise ModelArtifactError(
            "artifact readback identity mismatch after publication"
        )
    return published


def load_l2a_artifact(path: str | Path) -> LoadedL2aScorerArtifact:
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

    if body["artifact_schema"] != L2A_ARTIFACT_SCHEMA:
        raise ModelArtifactError(
            f"unsupported model artifact schema: {body['artifact_schema']!r}; "
            f"this implementation provides {L2A_ARTIFACT_SCHEMA!r}"
        )
    if body["kind"] != L2A_ARTIFACT_KIND:
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

    model_config = L2aScorerConfig.from_value(body["model"])
    encoding = model_config.encoding
    if body["encoding"] != encoding.block():
        raise ModelArtifactError(
            "artifact encoding / guard classification identity or fingerprint mismatch"
        )
    if body["label"] != l2a_label_block(encoding):
        raise ModelArtifactError("artifact label semantics mismatch")

    dataset = expect_object(
        body["dataset"], _DATASET_FIELDS, ModelArtifactError, "artifact.dataset"
    )
    if dataset["schema"] != L2A_DATASET_SCHEMA:
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

    training = _validate_l2a_training(body["training"])
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

    layout = model_config.parameter_layout()
    declared = expect_list(
        body["parameters"], ModelArtifactError, "artifact.parameters"
    )
    if declared != [parameter.to_value() for parameter in layout]:
        raise ModelArtifactError("artifact parameter layout mismatch")

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

    return LoadedL2aScorerArtifact(
        identity=manifest["identity"],
        manifest=manifest,
        model_config=model_config,
        parameters=layout,
        weights=weights,
    )


__all__ = [
    "L2A_ARTIFACT_KIND",
    "L2A_ARTIFACT_SCHEMA",
    "LoadedL2aScorerArtifact",
    "load_l2a_artifact",
    "write_l2a_artifact",
]
