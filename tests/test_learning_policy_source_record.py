"""Arena `arena-policy-source-record-v1`（lisjong-arena#442）consumerのcontract test。"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import candidate_fixtures as cf
import learning_fixtures as fixtures

from lisjong.learning import (
    BehaviorCloningConfig,
    DatasetError,
    ModelConfig,
    SourceRecordError,
    load_model_artifact,
    materialize_dataset,
    read_dataset,
    read_source_record,
    train_behavior_cloning,
)
from lisjong.learning.candidate_dataset import (
    materialize_candidate_dataset,
    read_candidate_dataset,
)
from lisjong.learning.dataset import validate_source_block
from lisjong.learning.source_record import (
    DEVELOPMENT_PURPOSE,
    POLICY_SOURCE_RECORD_SCHEMA_V1,
    SCIENTIFIC_PURPOSE,
)

requires_ml_runtime = unittest.skipIf(
    importlib.util.find_spec("torch") is None,
    "requires the optional ML runtime (lisjong[ml])",
)

EXPECTED_TEACHER = {
    "catalog_identity": fixtures.POLICY_TEACHER["catalog_identity"],
    "configuration_digest": fixtures.POLICY_TEACHER["configuration_digest"],
    "policy_class": fixtures.POLICY_TEACHER["policy_class"],
}


class PolicySourceRecordReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def record(self, games=None, **overrides) -> Path:
        return fixtures.write_policy_source_record(
            self.root / "source-record", games, **overrides
        )

    def test_development_record_round_trip_and_provenance(self) -> None:
        record = read_source_record(self.record())

        self.assertEqual(record.schema, POLICY_SOURCE_RECORD_SCHEMA_V1)
        self.assertEqual(record.purpose, DEVELOPMENT_PURPOSE)
        self.assertIsNone(record.allocation_bindings)
        self.assertIsNone(record.lock_identity)
        self.assertIsNone(record.scientific_corpus_identity)
        self.assertEqual(dict(record.teacher), EXPECTED_TEACHER)
        self.assertEqual(record.decision_count, 5)
        provenance = record.provenance()
        self.assertEqual(
            set(provenance),
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
            },
        )
        self.assertEqual(provenance["teacher"], EXPECTED_TEACHER)
        self.assertNotIn("source_contract", provenance)

    def test_scientific_record_requires_and_preserves_bindings(self) -> None:
        record = read_source_record(self.record(purpose=SCIENTIFIC_PURPOSE))
        self.assertEqual(set(record.allocation_bindings), {"TRAIN", "SELECT"})

        root = self.root / "missing"
        fixtures.write_policy_source_record(
            root, purpose=SCIENTIFIC_PURPOSE, allocation_bindings=None
        )
        with self.assertRaisesRegex(SourceRecordError, "population splits"):
            read_source_record(root)

    def test_development_record_must_not_claim_bindings(self) -> None:
        root = self.record(
            allocation_bindings={"TRAIN": fixtures.allocation_binding([100])}
        )
        with self.assertRaisesRegex(SourceRecordError, "must not claim"):
            read_source_record(root)

    def test_manifest_mismatches_fail_closed(self) -> None:
        cases = {
            "purpose": {"purpose": "PILOT"},
            "kind": {"kind": "player-safe-source-record"},
            "population seed": {"populations": {"TRAIN": [101], "SELECT": [200]}},
            "unknown split": {"populations": {"VALID": [100], "SELECT": [200]}},
            "teacher": {
                "source_contract": {
                    "teacher": {"catalog_identity": "placement-aware-speed-call"}
                }
            },
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                root = fixtures.write_policy_source_record(
                    self.root / name.replace(" ", "-"), **overrides
                )
                with self.assertRaises(SourceRecordError):
                    read_source_record(root)

    def test_game_summary_must_not_carry_a_lock_identity(self) -> None:
        root = self.record()
        fixtures.mutate_game_summary(
            root, 0, lambda body: body.__setitem__("lock_identity", "a" * 64)
        )
        with self.assertRaisesRegex(SourceRecordError, "unexpected fields"):
            read_source_record(root)


class PolicySourceDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, purpose=DEVELOPMENT_PURPOSE):
        return read_source_record(
            fixtures.write_policy_source_record(
                self.root / f"source-{purpose}", purpose=purpose
            )
        )

    def test_dataset_keeps_labels_split_and_teacher_provenance(self) -> None:
        for purpose in (DEVELOPMENT_PURPOSE, SCIENTIFIC_PURPOSE):
            with self.subTest(purpose):
                source = self.source(purpose)
                dataset = materialize_dataset(source, self.root / f"bc-{purpose}")
                reread = read_dataset(self.root / f"bc-{purpose}")
                self.assertEqual(reread.manifest, dataset.manifest)
                block = reread.manifest["source"]
                self.assertEqual(block, source.provenance())
                self.assertEqual(block["purpose"], purpose)
                self.assertEqual(block["teacher"], EXPECTED_TEACHER)
                self.assertEqual(
                    [row.split for row in reread.rows],
                    [decision.split for decision in source.decisions()],
                )

    def test_candidate_dataset_accepts_the_policy_source(self) -> None:
        games = []
        for game_ordinal, split in enumerate(("TRAIN", "SELECT", "OFFLINE-EVAL")):
            seed = 100 * (game_ordinal + 1)
            rows = cf.source_rows(
                cf.mixed_decisions(), game_ordinal=game_ordinal, seed=seed, split=split
            )
            games.append((split, seed, rows))
        source = read_source_record(
            fixtures.write_policy_source_record(self.root / "candidate-source", games)
        )
        materialize_candidate_dataset(source, self.root / "candidate")
        dataset = read_candidate_dataset(self.root / "candidate")
        self.assertEqual(dataset.manifest["source"], source.provenance())

    def test_source_block_purpose_and_bindings_are_validated(self) -> None:
        block = self.source().provenance()
        validate_source_block(block)
        cases = {
            "purpose": {**block, "purpose": "PILOT"},
            "bindings on development": {
                **block,
                "allocation_bindings": {
                    "TRAIN": fixtures.allocation_binding([100]),
                    "SELECT": fixtures.allocation_binding([200]),
                },
            },
            "scientific without bindings": {**block, "purpose": SCIENTIFIC_PURPOSE},
            "teacher": {**block, "teacher": {"catalog_identity": "x"}},
            "legacy field": {**block, "lock_identity": "a" * 64},
        }
        for name, value in cases.items():
            with self.subTest(name), self.assertRaises(DatasetError):
                validate_source_block(value)

    @requires_ml_runtime
    def test_development_source_trains_and_verifies(self) -> None:
        dataset = materialize_dataset(self.source(), self.root / "dataset")
        artifact = train_behavior_cloning(
            dataset,
            BehaviorCloningConfig(
                train_splits=("TRAIN",),
                validation_splits=("SELECT",),
                epochs=1,
                batch_size=2,
                model=ModelConfig(hidden_width=4),
            ),
            self.root / "artifact",
        )
        loaded = load_model_artifact(self.root / "artifact")
        self.assertEqual(loaded.manifest, artifact.manifest)
        self.assertEqual(loaded.manifest["source"]["purpose"], DEVELOPMENT_PURPOSE)


if __name__ == "__main__":
    unittest.main()
