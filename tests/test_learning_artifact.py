"""immutable model artifactの生成とstrict loadのcontract test。"""

import hashlib
import json
import tempfile
import unittest
from array import array
from pathlib import Path

import learning_fixtures as fixtures

import lisjong
from lisjong.learning import (
    MODEL_ARTIFACT_SCHEMA,
    ModelArtifactError,
    ModelConfig,
    load_model_artifact,
    materialize_dataset,
    read_source_record,
    write_model_artifact,
)
from lisjong.learning.artifact import MANIFEST_FILENAME, WEIGHTS_FILENAME
from lisjong.learning.dataset import feature_block, label_block, vocabulary_block


class ModelArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = read_source_record(
            fixtures.write_source_record(self.root / "source-record")
        )
        self.dataset = materialize_dataset(source, self.root / "dataset")
        self.model = ModelConfig(hidden_width=2)

    def write(self, name="artifact", **overrides):
        values = {
            "dataset": self.dataset,
            "model_config": self.model,
            "training": fixtures.training_block(),
            "weights": fixtures.zero_weights(self.model),
        }
        values.update(overrides)
        return write_model_artifact(self.root / name, **values)

    def test_write_then_load_binds_every_identity(self) -> None:
        artifact = self.write()
        manifest = artifact.manifest

        self.assertEqual(manifest["artifact_schema"], MODEL_ARTIFACT_SCHEMA)
        self.assertEqual(manifest["feature"], feature_block())
        self.assertEqual(manifest["vocabulary"], vocabulary_block())
        self.assertEqual(manifest["label"], label_block())
        self.assertEqual(manifest["dataset"]["identity"], self.dataset.identity)
        self.assertEqual(manifest["source"], dict(self.dataset.manifest["source"]))
        self.assertEqual(manifest["model"], self.model.to_value())
        self.assertEqual(manifest["training"]["seed"], 0)
        self.assertEqual(
            manifest["lisjong"]["package_version"],
            lisjong.__version__,
        )
        self.assertEqual(len(manifest["lisjong"]["source_digest"]), 64)
        self.assertEqual(len(artifact.weights), self.model.parameter_count)
        self.assertEqual(artifact.source_identity, manifest["source"]["identity"])

        loaded = load_model_artifact(self.root / "artifact")
        self.assertEqual(loaded.identity, artifact.identity)
        self.assertEqual(loaded.model_config, self.model)

    def test_artifact_contains_only_manifest_and_weights(self) -> None:
        self.write()
        root = self.root / "artifact"

        self.assertEqual(
            sorted(child.name for child in root.iterdir()),
            [MANIFEST_FILENAME, WEIGHTS_FILENAME],
        )
        self.assertEqual(
            (root / WEIGHTS_FILENAME).stat().st_size,
            self.model.parameter_count * 4,
        )
        text = (root / MANIFEST_FILENAME).read_text(encoding="utf-8")
        for forbidden in ("pickle", "lambda", "callable", "factory", "__main__"):
            self.assertNotIn(forbidden, text)

    def test_refuses_to_overwrite_an_existing_artifact(self) -> None:
        self.write()

        with self.assertRaisesRegex(ModelArtifactError, "refusing to overwrite"):
            self.write()

    def test_write_rejects_wrong_weight_length(self) -> None:
        with self.assertRaisesRegex(ModelArtifactError, "weights length"):
            self.write(weights=(0.0, 1.0))

    def test_write_rejects_non_finite_weights(self) -> None:
        weights = list(fixtures.zero_weights(self.model))
        weights[0] = float("inf")

        with self.assertRaisesRegex(ModelArtifactError, "finite"):
            self.write(weights=tuple(weights))

    def test_write_rejects_invalid_training_block(self) -> None:
        with self.assertRaisesRegex(ModelArtifactError, "learning_rate"):
            self.write(training=fixtures.training_block(learning_rate=-1.0))
        with self.assertRaisesRegex(ModelArtifactError, "unexpected fields"):
            self.write(
                name="artifact-2",
                training={**fixtures.training_block(), "extra": 1},
            )

    def test_load_rejects_unknown_artifact_schema(self) -> None:
        self.write()
        root = self.root / "artifact"
        fixtures.mutate_manifest(
            root, lambda body: body.update(artifact_schema="lisjong-future-model-v9")
        )

        with self.assertRaisesRegex(ModelArtifactError, "unsupported model artifact"):
            load_model_artifact(root)

    def test_load_rejects_feature_identity_mismatch(self) -> None:
        self.write()
        root = self.root / "artifact"
        fixtures.mutate_manifest(
            root,
            lambda body: body["feature"].update(
                identity="arena-policy-input-feature-v1", dimension=8204
            ),
        )

        with self.assertRaisesRegex(ModelArtifactError, "feature identity"):
            load_model_artifact(root)

    def test_load_rejects_vocabulary_fingerprint_mismatch(self) -> None:
        self.write()
        root = self.root / "artifact"
        fixtures.mutate_manifest(
            root, lambda body: body["vocabulary"].update(fingerprint="0" * 64)
        )

        with self.assertRaisesRegex(ModelArtifactError, "vocabulary identity"):
            load_model_artifact(root)

    def test_load_rejects_label_semantics_mismatch(self) -> None:
        self.write()
        root = self.root / "artifact"
        fixtures.mutate_manifest(
            root, lambda body: body["label"].update(objective="regression")
        )

        with self.assertRaisesRegex(ModelArtifactError, "label semantics"):
            load_model_artifact(root)

    def test_load_rejects_unsupported_architecture(self) -> None:
        self.write()
        root = self.root / "artifact"
        fixtures.mutate_manifest(
            root, lambda body: body["model"].update(architecture="arena-stage2-mlp")
        )

        with self.assertRaisesRegex(
            ModelArtifactError, "unsupported model architecture"
        ):
            load_model_artifact(root)

    def test_load_rejects_model_config_mismatch(self) -> None:
        self.write()
        root = self.root / "artifact"
        fixtures.mutate_manifest(
            root, lambda body: body["model"].update(hidden_width=4)
        )

        with self.assertRaisesRegex(
            ModelArtifactError, "does not match the model layout"
        ):
            load_model_artifact(root)

    def test_load_rejects_parameter_layout_mismatch(self) -> None:
        self.write()
        root = self.root / "artifact"
        fixtures.mutate_manifest(
            root, lambda body: body["parameters"][0].update(offset=1)
        )

        with self.assertRaisesRegex(
            ModelArtifactError, "does not match the model layout"
        ):
            load_model_artifact(root)

    def test_load_rejects_training_split_outside_the_dataset(self) -> None:
        self.write()
        root = self.root / "artifact"
        fixtures.mutate_manifest(
            root, lambda body: body["training"].update(train_splits=["MISSING"])
        )

        with self.assertRaisesRegex(ModelArtifactError, "not dataset splits"):
            load_model_artifact(root)

    def test_load_rejects_selected_epoch_beyond_budget(self) -> None:
        self.write()
        root = self.root / "artifact"
        fixtures.mutate_manifest(
            root, lambda body: body["training"].update(selected_epoch=9)
        )

        with self.assertRaisesRegex(ModelArtifactError, "selected epoch"):
            load_model_artifact(root)

    def test_load_rejects_invalid_source_provenance(self) -> None:
        self.write()
        root = self.root / "artifact"
        fixtures.mutate_manifest(root, lambda body: body["source"].pop("population"))

        with self.assertRaisesRegex(ModelArtifactError, "source provenance is invalid"):
            load_model_artifact(root)

    def test_load_rejects_manifest_identity_tampering(self) -> None:
        self.write()
        root = self.root / "artifact"
        body = fixtures.read_manifest_body(root)
        fixtures.rewrite_manifest(root, body, identity="0" * 64)

        with self.assertRaisesRegex(ModelArtifactError, "identity mismatch"):
            load_model_artifact(root)

    def test_load_rejects_corrupted_manifest(self) -> None:
        self.write()
        root = self.root / "artifact"
        (root / MANIFEST_FILENAME).write_text('{"artifact_schema":', encoding="utf-8")

        with self.assertRaisesRegex(ModelArtifactError, "strict JSON"):
            load_model_artifact(root)

    def test_load_rejects_non_canonical_manifest(self) -> None:
        self.write()
        root = self.root / "artifact"
        manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        (root / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )

        with self.assertRaisesRegex(ModelArtifactError, "canonical JSON"):
            load_model_artifact(root)

    def test_load_rejects_weights_digest_mismatch(self) -> None:
        self.write()
        root = self.root / "artifact"
        payload = root / WEIGHTS_FILENAME
        values = array("f")
        values.frombytes(payload.read_bytes())
        values[0] = 1.5
        payload.write_bytes(values.tobytes())

        with self.assertRaisesRegex(ModelArtifactError, "digest mismatch"):
            load_model_artifact(root)

    def test_load_rejects_truncated_weights(self) -> None:
        self.write()
        root = self.root / "artifact"
        payload = root / WEIGHTS_FILENAME
        data = payload.read_bytes()[:-4]
        payload.write_bytes(data)
        fixtures.mutate_manifest(
            root,
            lambda body: body["files"][WEIGHTS_FILENAME].update(
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
            ),
        )

        with self.assertRaisesRegex(ModelArtifactError, "byte length mismatch"):
            load_model_artifact(root)

    def test_load_rejects_non_finite_weights(self) -> None:
        self.write()
        root = self.root / "artifact"
        payload = root / WEIGHTS_FILENAME
        values = array("f")
        values.frombytes(payload.read_bytes())
        values[1] = float("nan")
        data = values.tobytes()
        payload.write_bytes(data)
        fixtures.mutate_manifest(
            root,
            lambda body: body["files"][WEIGHTS_FILENAME].update(
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
            ),
        )

        with self.assertRaisesRegex(ModelArtifactError, "non-finite"):
            load_model_artifact(root)

    def test_load_rejects_unexpected_files(self) -> None:
        self.write()
        root = self.root / "artifact"
        (root / "optimizer.pt").write_bytes(b"\x00")

        with self.assertRaisesRegex(ModelArtifactError, "missing/unexpected"):
            load_model_artifact(root)

    def test_load_rejects_missing_directory(self) -> None:
        with self.assertRaisesRegex(ModelArtifactError, "does not exist"):
            load_model_artifact(self.root / "absent")

    def test_parameter_values_are_addressable_by_name(self) -> None:
        artifact = self.write()

        self.assertEqual(
            len(artifact.parameter_values("input_layer.bias")),
            self.model.hidden_width,
        )
        with self.assertRaisesRegex(ModelArtifactError, "no parameter named"):
            artifact.parameter_values("missing.weight")


if __name__ == "__main__":
    unittest.main()
