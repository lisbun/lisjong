"""bounded BC trainerのcontract test（optional ML runtimeが必要）。"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import learning_fixtures as fixtures

from lisjong.learning import (
    BehaviorCloningConfig,
    ModelConfig,
    TrainingError,
    load_model_artifact,
    materialize_dataset,
    read_source_record,
    train_behavior_cloning,
)
from lisjong.learning.artifact import WEIGHTS_FILENAME
from lisjong.learning.dataset import TRAINING_OBJECTIVE
from lisjong.learning.training import OPTIMIZER, TRAINING_FRAMEWORK

requires_ml_runtime = unittest.skipIf(
    importlib.util.find_spec("torch") is None,
    "requires the optional ML runtime (lisjong[ml])",
)


class BehaviorCloningConfigTests(unittest.TestCase):
    def test_requires_explicit_train_splits(self) -> None:
        with self.assertRaises(TypeError):
            BehaviorCloningConfig()

    def test_rejects_overlapping_splits(self) -> None:
        with self.assertRaisesRegex(ValueError, "disjoint"):
            BehaviorCloningConfig(train_splits=("TRAIN",), validation_splits=("TRAIN",))

    def test_rejects_duplicate_and_empty_split_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            BehaviorCloningConfig(train_splits=("TRAIN", "TRAIN"))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            BehaviorCloningConfig(train_splits=("",))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            BehaviorCloningConfig(train_splits=())
        with self.assertRaises(TypeError):
            BehaviorCloningConfig(train_splits="TRAIN")

    def test_rejects_out_of_range_hyperparameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "epochs"):
            BehaviorCloningConfig(train_splits=("TRAIN",), epochs=0)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            BehaviorCloningConfig(train_splits=("TRAIN",), batch_size=0)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            BehaviorCloningConfig(train_splits=("TRAIN",), learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "weight_decay"):
            BehaviorCloningConfig(train_splits=("TRAIN",), weight_decay=-1.0)
        with self.assertRaisesRegex(ValueError, "seed"):
            BehaviorCloningConfig(train_splits=("TRAIN",), seed=-1)


class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = read_source_record(
            fixtures.write_source_record(self.root / "source-record")
        )
        self.dataset = materialize_dataset(source, self.root / "dataset")
        self.model = ModelConfig(hidden_width=4)

    def config(self, **overrides) -> BehaviorCloningConfig:
        values = {
            "train_splits": ("TRAIN",),
            "validation_splits": ("SELECT",),
            "epochs": 2,
            "batch_size": 2,
            "model": self.model,
        }
        values.update(overrides)
        return BehaviorCloningConfig(**values)

    def test_rejects_unknown_split(self) -> None:
        with self.assertRaisesRegex(TrainingError, "absent from the dataset"):
            train_behavior_cloning(
                self.dataset,
                self.config(train_splits=("MISSING",), validation_splits=()),
                self.root / "artifact",
            )

    def test_rejects_non_dataset_input(self) -> None:
        with self.assertRaisesRegex(TrainingError, "LearningDataset"):
            train_behavior_cloning(object(), self.config(), self.root / "artifact")

    @requires_ml_runtime
    def test_training_writes_a_bound_immutable_artifact(self) -> None:
        artifact = train_behavior_cloning(
            self.dataset, self.config(), self.root / "artifact"
        )
        training = artifact.manifest["training"]

        self.assertEqual(artifact.dataset_identity, self.dataset.identity)
        self.assertEqual(
            artifact.manifest["source"]["identity"],
            self.dataset.manifest["source"]["identity"],
        )
        self.assertEqual(training["objective"], TRAINING_OBJECTIVE)
        self.assertEqual(training["optimizer"], OPTIMIZER)
        self.assertEqual(training["framework"]["name"], TRAINING_FRAMEWORK)
        self.assertEqual(training["train_splits"], ["TRAIN"])
        self.assertEqual(training["validation_splits"], ["SELECT"])
        self.assertEqual(training["seed"], 0)
        self.assertTrue(1 <= training["selected_epoch"] <= training["epochs"])
        self.assertEqual(training["diagnostics"]["train_rows"], 3)
        self.assertEqual(training["diagnostics"]["validation_rows"], 2)
        self.assertEqual(len(artifact.weights), self.model.parameter_count)
        self.assertEqual(
            load_model_artifact(self.root / "artifact").identity, artifact.identity
        )

    @requires_ml_runtime
    def test_training_is_reproducible_for_the_same_seed(self) -> None:
        first = train_behavior_cloning(
            self.dataset, self.config(), self.root / "artifact-a"
        )
        second = train_behavior_cloning(
            self.dataset, self.config(), self.root / "artifact-b"
        )

        self.assertEqual(first.identity, second.identity)
        self.assertEqual(
            (self.root / "artifact-a" / WEIGHTS_FILENAME).read_bytes(),
            (self.root / "artifact-b" / WEIGHTS_FILENAME).read_bytes(),
        )

    @requires_ml_runtime
    def test_a_different_seed_changes_the_weights(self) -> None:
        first = train_behavior_cloning(
            self.dataset, self.config(), self.root / "artifact-a"
        )
        second = train_behavior_cloning(
            self.dataset, self.config(seed=7), self.root / "artifact-b"
        )

        self.assertNotEqual(first.identity, second.identity)
        self.assertNotEqual(
            (self.root / "artifact-a" / WEIGHTS_FILENAME).read_bytes(),
            (self.root / "artifact-b" / WEIGHTS_FILENAME).read_bytes(),
        )

    @requires_ml_runtime
    def test_training_without_validation_selects_the_final_epoch(self) -> None:
        artifact = train_behavior_cloning(
            self.dataset,
            self.config(validation_splits=(), epochs=3),
            self.root / "artifact",
        )

        self.assertEqual(artifact.manifest["training"]["selected_epoch"], 3)
        self.assertEqual(artifact.manifest["training"]["validation_splits"], [])
        self.assertEqual(
            artifact.manifest["training"]["diagnostics"]["validation_rows"], 0
        )

    @requires_ml_runtime
    def test_training_refuses_to_overwrite_an_existing_artifact(self) -> None:
        train_behavior_cloning(self.dataset, self.config(), self.root / "artifact")

        with self.assertRaisesRegex(Exception, "refusing to overwrite"):
            train_behavior_cloning(self.dataset, self.config(), self.root / "artifact")


if __name__ == "__main__":
    unittest.main()
