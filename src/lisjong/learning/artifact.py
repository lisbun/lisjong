"""immutable model artifactの生成とstrict load。

Issue #184のL0cに対応する。1 model artifact = 1 immutable directoryであり、
repository外のoperator-owned成果物として扱う（weights / checkpointはGitへ
commitしない）。

```text
<artifact>/
    manifest.json   canonical JSON identity / provenance / config / digest
    weights.f32     parameter_layout順のlittle-endian float32 payload
```

manifestがbindするもの。

- source-record identity / lock identity / ordered source population
- dataset schema / identity / row数 / split母数
- feature identity / dimension / fingerprint
- action vocabulary version / size / fingerprint
- teacher / label semantics
- model architecture / config / parameter layout
- optimizer / training config / RNG seed / training framework identity
- weights payloadのSHA-256とbyte長
- lisjong package version / installed source digest

固定する原則。

- 既存pathをsilentに上書きしない
- callable、factory、lambda、pickleされたobject、optimizer state、任意codeを
  保存・復元しない。payloadは明示的なlayoutに従うfloat32だけである
- identity / config / digestの不整合はload時にfail closedする。近い設定へ
  丸めたり、mismatchしたweightsを読み込んだりしない
- historical Arena artifact identityを再利用しない。artifact schemaも
  feature / vocabulary fingerprintもlisjong所有のcanonical identityである
"""

import sys
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import lisjong
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
    value_digest,
)
from lisjong.learning._publication import (
    staged_publication,
    write_new_bytes,
    write_new_text,
)
from lisjong.learning.dataset import (
    DATASET_SCHEMA,
    LearningDataset,
    feature_block,
    label_block,
    validate_source_block,
    vocabulary_block,
)
from lisjong.learning.errors import DatasetError, ModelArtifactError
from lisjong.learning.model import ModelConfig, ModelParameter

MODEL_ARTIFACT_SCHEMA = "lisjong-offense-l0-bc-model-artifact-v1"
"""このmodel artifact contractのidentity。"""

MODEL_ARTIFACT_KIND = "behavior-cloning-model"
MANIFEST_FILENAME = "manifest.json"
WEIGHTS_FILENAME = "weights.f32"

_WEIGHTS_DTYPE = "float32-le"
_FLOAT32_BYTES = 4

_MANIFEST_FIELDS = frozenset(
    {
        "artifact_schema",
        "dataset",
        "feature",
        "files",
        "kind",
        "label",
        "lisjong",
        "model",
        "parameters",
        "source",
        "training",
        "vocabulary",
    }
)
_DATASET_FIELDS = frozenset({"identity", "row_count", "schema", "splits"})
_TRAINING_FIELDS = frozenset(
    {
        "batch_size",
        "diagnostics",
        "epochs",
        "framework",
        "learning_rate",
        "objective",
        "optimizer",
        "seed",
        "selected_epoch",
        "train_splits",
        "validation_splits",
        "weight_decay",
    }
)
_FRAMEWORK_FIELDS = frozenset({"name", "version"})
_LISJONG_FIELDS = frozenset({"package_version", "source_digest"})
_PARAMETER_FIELDS = frozenset({"count", "name", "offset", "shape"})
_WEIGHTS_FIELDS = frozenset({"bytes", "count", "dtype", "sha256"})


@dataclass(frozen=True, slots=True)
class LoadedModelArtifact:
    """strict loadしたmodel artifact。

    `weights`は`model_config.parameter_layout()`順のflat float32 valueである。
    実行用modelはこのvalueから構築する。artifactへtorch module objectや
    factoryを保存しないため、load時に任意codeを復元しない。
    """

    identity: str
    manifest: Mapping[str, object]
    model_config: ModelConfig
    parameters: tuple[ModelParameter, ...]
    weights: array

    @property
    def dataset_identity(self) -> str:
        return self.manifest["dataset"]["identity"]

    @property
    def source_identity(self) -> str:
        return self.manifest["source"]["identity"]

    def parameter_values(self, name: str) -> tuple[float, ...]:
        for parameter in self.parameters:
            if parameter.name == name:
                start = parameter.offset
                return tuple(self.weights[start : start + parameter.count])
        raise ModelArtifactError(f"artifact has no parameter named {name!r}")


def _weights_bytes(values: Sequence[float]) -> bytes:
    payload = array("f", values)
    if sys.byteorder != "little":
        payload.byteswap()
    return payload.tobytes()


def _weights_array(data: bytes) -> array:
    payload = array("f")
    payload.frombytes(data)
    if sys.byteorder != "little":
        payload.byteswap()
    return payload


def _lisjong_source_digest() -> str:
    """installed `lisjong`パッケージの実ソースから単一digestを計算する。

    git revisionはtraining実行環境がgit checkoutであることを前提にしてしまい、
    installed packageからのtrainingでは取得できない。代わりに、実際にimportされた
    `lisjong`パッケージ配下の全`.py`ファイルを相対path順に読み、
    `{path: sha256}`のmappingそのものをcanonical digestする。これは
    「training時に実際にどのlisjong実装が動いたか」を、git履歴の有無に
    依存せず記録するsemantic identityである。
    """
    root = Path(lisjong.__file__).parent
    sources = {
        str(path.relative_to(root)).replace("\\", "/"): value_digest(
            path.read_bytes().decode("utf-8", errors="surrogateescape")
        )
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    return value_digest(sources)


def _lisjong_block() -> dict[str, object]:
    return {
        "package_version": lisjong.__version__,
        "source_digest": _lisjong_source_digest(),
    }


def _dataset_block(dataset: LearningDataset) -> dict[str, object]:
    return {
        "identity": dataset.identity,
        "row_count": dataset.row_count,
        "schema": DATASET_SCHEMA,
        "splits": dict(sorted(dataset.split_counts().items())),
    }


def _validate_training(value: object) -> dict[str, object]:
    training = expect_object(value, _TRAINING_FIELDS, ModelArtifactError, "training")
    for field in ("batch_size", "epochs", "selected_epoch"):
        expect_non_negative_int(
            training[field], ModelArtifactError, f"training.{field}"
        )
    expect_non_negative_int(training["seed"], ModelArtifactError, "training.seed")
    for field in ("learning_rate", "weight_decay"):
        item = training[field]
        if type(item) is not float or not isfinite(item) or item < 0.0:
            raise ModelArtifactError(f"training.{field} must be a finite float >= 0")
    expect_str(training["objective"], ModelArtifactError, "training.objective")
    expect_str(training["optimizer"], ModelArtifactError, "training.optimizer")
    for field in ("train_splits", "validation_splits"):
        names = expect_list(training[field], ModelArtifactError, f"training.{field}")
        for index, name in enumerate(names):
            expect_str(name, ModelArtifactError, f"training.{field}[{index}]")
    if not training["train_splits"]:
        raise ModelArtifactError("training.train_splits must not be empty")
    framework = expect_object(
        training["framework"],
        _FRAMEWORK_FIELDS,
        ModelArtifactError,
        "training.framework",
    )
    expect_str(framework["name"], ModelArtifactError, "training.framework.name")
    expect_str(framework["version"], ModelArtifactError, "training.framework.version")
    diagnostics = training["diagnostics"]
    if type(diagnostics) is not dict:
        raise ModelArtifactError("training.diagnostics must be a JSON object")
    for name, item in diagnostics.items():
        expect_str(name, ModelArtifactError, "training.diagnostics key")
        if type(item) not in (int, float) or not isfinite(item):
            raise ModelArtifactError(
                f"training.diagnostics[{name}] must be a finite number"
            )
    return training


def write_model_artifact(
    destination: str | Path,
    *,
    dataset: LearningDataset,
    model_config: ModelConfig,
    training: Mapping[str, object],
    weights: Sequence[float],
) -> LoadedModelArtifact:
    """artifactをimmutableに書き出し、strict loadした結果を返す。

    `dataset`はstrict readしたdatasetであり、そのbindされたsource / dataset
    identityをそのままartifactへ引き継ぐ。
    """
    if not isinstance(dataset, LearningDataset):
        raise ModelArtifactError("dataset must be a strict-read LearningDataset")
    if not isinstance(model_config, ModelConfig):
        raise ModelArtifactError("model_config must be a ModelConfig")
    if len(weights) != model_config.parameter_count:
        raise ModelArtifactError("weights length does not match the model layout")
    if any(not isfinite(value) for value in weights):
        raise ModelArtifactError("weights must be finite")

    training_value = _validate_training(dict(training))
    destination = Path(destination)
    with staged_publication(destination, ModelArtifactError) as staging:
        weights_path = staging / WEIGHTS_FILENAME
        write_new_bytes(weights_path, _weights_bytes(weights), ModelArtifactError)
        manifest = seal(
            {
                "artifact_schema": MODEL_ARTIFACT_SCHEMA,
                "dataset": _dataset_block(dataset),
                "feature": feature_block(),
                "files": {
                    WEIGHTS_FILENAME: {
                        **file_digest(weights_path),
                        "count": len(weights),
                        "dtype": _WEIGHTS_DTYPE,
                    }
                },
                "kind": MODEL_ARTIFACT_KIND,
                "label": label_block(),
                "lisjong": _lisjong_block(),
                "model": model_config.to_value(),
                "parameters": [
                    parameter.to_value()
                    for parameter in model_config.parameter_layout()
                ],
                "source": dict(dataset.manifest["source"]),
                "training": training_value,
                "vocabulary": vocabulary_block(),
            }
        )
        write_new_text(
            staging / MANIFEST_FILENAME,
            canonical_json_text(manifest),
            ModelArtifactError,
        )
        staged = load_model_artifact(staging)

    published = load_model_artifact(destination)
    if published.identity != staged.identity:
        raise ModelArtifactError(
            "artifact readback identity mismatch after publication"
        )
    return published


def load_model_artifact(path: str | Path) -> LoadedModelArtifact:
    """artifact directoryをstrict loadする。

    manifest identity、schema、bindされたfeature / vocabulary / label identity、
    dataset / source provenance、model config、parameter layout、weights payload
    のbyte長とdigestをすべて照合する。1つでも合わなければfail closedする。
    """
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

    if body["artifact_schema"] != MODEL_ARTIFACT_SCHEMA:
        raise ModelArtifactError(
            f"unsupported model artifact schema: {body['artifact_schema']!r}; "
            f"this implementation provides {MODEL_ARTIFACT_SCHEMA!r}"
        )
    if body["kind"] != MODEL_ARTIFACT_KIND:
        raise ModelArtifactError("model artifact kind mismatch")
    if {child.name for child in root.iterdir()} != {
        MANIFEST_FILENAME,
        WEIGHTS_FILENAME,
    }:
        raise ModelArtifactError("missing/unexpected model artifact files")

    if body["feature"] != feature_block():
        raise ModelArtifactError("artifact feature identity/fingerprint mismatch")
    if body["vocabulary"] != vocabulary_block():
        raise ModelArtifactError(
            "artifact action vocabulary identity/fingerprint mismatch"
        )
    if body["label"] != label_block():
        raise ModelArtifactError("artifact label semantics mismatch")

    dataset = expect_object(
        body["dataset"], _DATASET_FIELDS, ModelArtifactError, "artifact.dataset"
    )
    if dataset["schema"] != DATASET_SCHEMA:
        raise ModelArtifactError("artifact dataset schema mismatch")
    expect_str(dataset["identity"], ModelArtifactError, "artifact.dataset.identity")
    row_count = expect_non_negative_int(
        dataset["row_count"], ModelArtifactError, "artifact.dataset.row_count"
    )
    splits = dataset["splits"]
    if type(splits) is not dict or not splits:
        raise ModelArtifactError("artifact.dataset.splits must be a non-empty object")
    for name, count in splits.items():
        expect_str(name, ModelArtifactError, "artifact.dataset.splits key")
        expect_non_negative_int(
            count, ModelArtifactError, f"artifact.dataset.splits[{name}]"
        )
    if sum(splits.values()) != row_count:
        raise ModelArtifactError("artifact dataset split accounting mismatch")

    try:
        validate_source_block(body["source"], context="artifact.source")
    except DatasetError as exc:
        raise ModelArtifactError(
            f"artifact source provenance is invalid: {exc}"
        ) from exc

    training = _validate_training(body["training"])
    if set(training["train_splits"]) - set(splits) or set(
        training["validation_splits"]
    ) - set(splits):
        raise ModelArtifactError("artifact training splits are not dataset splits")
    if training["selected_epoch"] > training["epochs"]:
        raise ModelArtifactError("artifact selected epoch exceeds the epoch budget")

    lisjong_block = expect_object(
        body["lisjong"], _LISJONG_FIELDS, ModelArtifactError, "artifact.lisjong"
    )
    expect_str(
        lisjong_block["package_version"],
        ModelArtifactError,
        "artifact.lisjong.package_version",
    )
    # source_digestはprovenance記録であり、現在installed中のlisjongと一致する
    # ことは要求しない。無関係なmodule変更のたびに既存artifactが読めなくなる
    # ことを避けるためであり、feature / vocabulary fingerprintという
    # load-relevantなidentityは別途厳密に照合する。
    expect_digest(
        lisjong_block["source_digest"],
        ModelArtifactError,
        "artifact.lisjong.source_digest",
    )

    model_config = ModelConfig.from_value(body["model"])
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

    return LoadedModelArtifact(
        identity=manifest["identity"],
        manifest=manifest,
        model_config=model_config,
        parameters=layout,
        weights=weights,
    )


__all__ = [
    "MANIFEST_FILENAME",
    "MODEL_ARTIFACT_KIND",
    "MODEL_ARTIFACT_SCHEMA",
    "WEIGHTS_FILENAME",
    "LoadedModelArtifact",
    "load_model_artifact",
    "write_model_artifact",
]
