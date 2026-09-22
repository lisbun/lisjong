"""lisjong所有のBC dataset materializationとstrict readのcontract test。"""

import hashlib
import json
import tempfile
import unittest
from array import array
from pathlib import Path

import learning_fixtures as fixtures

from lisjong.action_vocabulary import ACTION_VOCABULARY_SIZE, encode_action
from lisjong.learning import (
    DATASET_SCHEMA,
    FEATURE_DIMENSION,
    SOURCE_RECORD_SCHEMA_V1,
    TEACHER_LABEL_SEMANTICS,
    DatasetError,
    build_player_safe_feature,
    materialize_dataset,
    read_dataset,
    read_source_record,
)
from lisjong.learning._canonical import canonical_json_line
from lisjong.learning.dataset import (
    FEATURES_FILENAME,
    LEGAL_MASK_FILENAME,
    MANIFEST_FILENAME,
    ROWS_FILENAME,
    feature_block,
    label_block,
    vocabulary_block,
)


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = read_source_record(
            fixtures.write_source_record(self.root / "source-record")
        )

    def dataset(self, name="dataset"):
        return materialize_dataset(self.source, self.root / name)

    def test_materialization_binds_identity_and_population(self) -> None:
        dataset = self.dataset()
        manifest = dataset.manifest

        self.assertEqual(manifest["dataset_schema"], DATASET_SCHEMA)
        self.assertEqual(manifest["feature"], feature_block())
        self.assertEqual(manifest["vocabulary"], vocabulary_block())
        self.assertEqual(manifest["label"], label_block())
        self.assertEqual(manifest["label"]["semantics"], TEACHER_LABEL_SEMANTICS)
        self.assertEqual(manifest["source"]["identity"], self.source.identity)
        self.assertEqual(
            manifest["source"]["population"], list(self.source.population())
        )
        self.assertEqual(manifest["rows"]["count"], 5)
        self.assertEqual(manifest["rows"]["splits"], {"SELECT": 2, "TRAIN": 3})
        self.assertEqual(dataset.split_counts(), {"TRAIN": 3, "SELECT": 2})

    def test_allocation_bindings_are_preserved_exactly_from_the_source(self) -> None:
        """Arena allocation provenanceはlisjongが再生成せず、そのままbindする。"""
        dataset = self.dataset()

        self.assertEqual(
            dataset.manifest["source"]["allocation_bindings"],
            self.source.provenance()["allocation_bindings"],
        )
        self.assertEqual(
            set(dataset.manifest["source"]["allocation_bindings"]),
            {"TRAIN", "SELECT"},
        )

    def test_materialization_rejects_a_v1_source_without_allocation_bindings(
        self,
    ) -> None:
        v1_source = read_source_record(
            fixtures.write_source_record(
                self.root / "v1-source", schema=SOURCE_RECORD_SCHEMA_V1
            )
        )
        self.assertIsNone(v1_source.allocation_bindings)

        with self.assertRaisesRegex(DatasetError, "allocation provenance"):
            materialize_dataset(v1_source, self.root / "dataset-from-v1")

    def test_row_order_and_labels_follow_the_source(self) -> None:
        dataset = self.dataset()
        decisions = list(self.source.decisions())

        self.assertEqual(dataset.row_count, len(decisions))
        for index, (row, decision) in enumerate(zip(dataset.rows, decisions)):
            self.assertEqual(row.game_ordinal, decision.game_ordinal)
            self.assertEqual(row.seed, decision.seed)
            self.assertEqual(row.split, decision.split)
            self.assertEqual(row.decision_ordinal, decision.decision_ordinal)
            self.assertEqual(row.actor_seat, decision.actor_seat)
            self.assertEqual(
                row.teacher_action_index, encode_action(decision.selected_action)
            )
            self.assertEqual(
                dataset.legal_indices(index),
                tuple(sorted(encode_action(a) for a in decision.legal_actions)),
            )
            self.assertEqual(row.legal_action_count, len(decision.legal_actions))

    def test_feature_payload_matches_the_materialized_feature(self) -> None:
        dataset = self.dataset()
        decisions = list(self.source.decisions())

        for index, decision in enumerate(decisions):
            expected = array(
                "f", build_player_safe_feature(decision.policy_input)
            ).tolist()
            self.assertEqual(len(expected), FEATURE_DIMENSION)
            self.assertEqual(list(dataset.feature_row(index)), expected)

    def test_rematerialization_is_reproducible(self) -> None:
        first = self.dataset("dataset-a")
        second = self.dataset("dataset-b")

        self.assertEqual(first.identity, second.identity)
        for name in (
            MANIFEST_FILENAME,
            ROWS_FILENAME,
            FEATURES_FILENAME,
            LEGAL_MASK_FILENAME,
        ):
            self.assertEqual(
                (self.root / "dataset-a" / name).read_bytes(),
                (self.root / "dataset-b" / name).read_bytes(),
            )

    def test_refuses_to_overwrite_an_existing_destination(self) -> None:
        self.dataset()

        with self.assertRaisesRegex(DatasetError, "refusing to overwrite"):
            self.dataset()

    def test_split_membership_selection_preserves_order(self) -> None:
        dataset = self.dataset()

        self.assertEqual(dataset.row_indices_for_splits(("TRAIN",)), (0, 1, 2))
        self.assertEqual(dataset.row_indices_for_splits(("SELECT",)), (3, 4))
        self.assertEqual(
            dataset.row_indices_for_splits(("SELECT", "TRAIN")), (0, 1, 2, 3, 4)
        )

    def test_unknown_dataset_schema_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(
            root,
            lambda body: body.update(dataset_schema="lisjong-future-dataset-v9"),
        )

        with self.assertRaisesRegex(DatasetError, "unsupported dataset schema"):
            read_dataset(root)

    def test_feature_identity_mismatch_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(
            root,
            lambda body: body["feature"].update(
                identity="arena-policy-input-feature-v1"
            ),
        )

        with self.assertRaisesRegex(DatasetError, "feature identity"):
            read_dataset(root)

    def test_feature_fingerprint_mismatch_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(
            root, lambda body: body["feature"].update(fingerprint="0" * 64)
        )

        with self.assertRaisesRegex(DatasetError, "feature identity"):
            read_dataset(root)

    def test_vocabulary_mismatch_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(root, lambda body: body["vocabulary"].update(size=1))

        with self.assertRaisesRegex(DatasetError, "vocabulary identity"):
            read_dataset(root)

    def test_label_semantics_mismatch_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(
            root, lambda body: body["label"].update(semantics="other-teacher-v1")
        )

        with self.assertRaisesRegex(DatasetError, "label semantics"):
            read_dataset(root)

    def test_source_population_tampering_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(
            root,
            lambda body: body["source"]["population"][0].update(decision_count=99),
        )

        with self.assertRaisesRegex(DatasetError, "accounting mismatch"):
            read_dataset(root)

    def test_source_seed_reuse_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(
            root,
            lambda body: body["source"]["population"][1].update(
                seed=body["source"]["population"][0]["seed"]
            ),
        )

        with self.assertRaisesRegex(DatasetError, "reuses a seed"):
            read_dataset(root)

    def test_tampered_allocation_identity_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(
            root,
            lambda body: body["source"]["allocation_bindings"]["TRAIN"].update(
                allocation_identity="not-a-valid-sha256-digest"
            ),
        )

        with self.assertRaisesRegex(DatasetError, "SHA-256"):
            read_dataset(root)

    def test_tampered_seed_membership_identity_rejected(self) -> None:
        """population(TRAIN=seed 100)と矛盾するbindingはfail closedする。"""
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(
            root,
            lambda body: body["source"]["allocation_bindings"]["TRAIN"].update(
                fixtures.allocation_binding([777])
            ),
        )

        with self.assertRaisesRegex(DatasetError, "contradicts the source population"):
            read_dataset(root)

    def test_allocation_binding_split_mismatch_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(
            root, lambda body: body["source"]["allocation_bindings"].pop("SELECT")
        )

        with self.assertRaisesRegex(
            DatasetError, "do not match the source population splits"
        ):
            read_dataset(root)

    def test_extra_allocation_binding_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(
            root,
            lambda body: body["source"]["allocation_bindings"].__setitem__(
                "OFFLINE-EVAL", fixtures.allocation_binding([999])
            ),
        )

        with self.assertRaisesRegex(
            DatasetError, "do not match the source population splits"
        ):
            read_dataset(root)

    def test_missing_allocation_bindings_field_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        fixtures.mutate_manifest(
            root, lambda body: body["source"].pop("allocation_bindings")
        )

        with self.assertRaisesRegex(DatasetError, "unexpected fields"):
            read_dataset(root)

    def test_manifest_identity_tampering_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        body = fixtures.read_manifest_body(root)
        fixtures.rewrite_manifest(root, body, identity="0" * 64)

        with self.assertRaisesRegex(DatasetError, "identity mismatch"):
            read_dataset(root)

    def test_payload_digest_mismatch_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        payload = root / FEATURES_FILENAME
        payload.write_bytes(payload.read_bytes()[:-4])

        with self.assertRaisesRegex(DatasetError, "digest mismatch"):
            read_dataset(root)

    def test_non_finite_feature_payload_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        payload = root / FEATURES_FILENAME
        values = array("f")
        values.frombytes(payload.read_bytes())
        values[0] = float("nan")
        payload.write_bytes(values.tobytes())
        fixtures.mutate_manifest(
            root,
            lambda body: body["files"][FEATURES_FILENAME].update(
                sha256=hashlib.sha256(payload.read_bytes()).hexdigest()
            ),
        )

        with self.assertRaisesRegex(DatasetError, "non-finite"):
            read_dataset(root)

    def test_row_reordering_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        rows_path = root / ROWS_FILENAME
        lines = rows_path.read_text(encoding="utf-8").splitlines(keepends=True)
        rows_path.write_text(
            "".join([lines[1], lines[0], *lines[2:]]), encoding="utf-8", newline="\n"
        )
        self._refresh_digest(root, ROWS_FILENAME)

        with self.assertRaisesRegex(DatasetError, "decision ordinal"):
            read_dataset(root)

    def test_teacher_label_outside_the_legal_mask_rejected(self) -> None:
        dataset = self.dataset()
        root = self.root / "dataset"
        mask_path = root / LEGAL_MASK_FILENAME
        mask = bytearray(mask_path.read_bytes())
        label = dataset.rows[0].teacher_action_index
        mask[label] = 0
        mask_path.write_bytes(bytes(mask))
        self._refresh_digest(root, LEGAL_MASK_FILENAME)
        self._set_row_field(
            root, 0, "legal_action_count", sum(mask[:ACTION_VOCABULARY_SIZE])
        )

        with self.assertRaisesRegex(DatasetError, "teacher label is not legal"):
            read_dataset(root)

    def test_legal_action_count_mismatch_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        self._set_row_field(root, 0, "legal_action_count", 1)

        with self.assertRaisesRegex(DatasetError, "legal action count mismatch"):
            read_dataset(root)

    def test_unexpected_dataset_file_rejected(self) -> None:
        self.dataset()
        root = self.root / "dataset"
        (root / "notes.txt").write_text("extra\n", encoding="utf-8")

        with self.assertRaisesRegex(DatasetError, "missing/unexpected"):
            read_dataset(root)

    def test_missing_directory_rejected(self) -> None:
        with self.assertRaisesRegex(DatasetError, "does not exist"):
            read_dataset(self.root / "absent")

    def _refresh_digest(self, root: Path, name: str) -> None:
        payload = (root / name).read_bytes()
        fixtures.mutate_manifest(
            root,
            lambda body: body["files"][name].update(
                bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
            ),
        )

    def _set_row_field(self, root: Path, index: int, field: str, value) -> None:
        rows_path = root / ROWS_FILENAME
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        rows[index][field] = value
        rows_path.write_text(
            "".join(canonical_json_line(row) for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        self._refresh_digest(root, ROWS_FILENAME)


if __name__ == "__main__":
    unittest.main()
