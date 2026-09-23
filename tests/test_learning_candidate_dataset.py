"""Issue #189 candidate scorer datasetのmaterializationとstrict readのtest。"""

import json
import tempfile
import unittest
from pathlib import Path

import candidate_fixtures as cf
import learning_fixtures as fixtures

from lisjong.learning import (
    CANDIDATE_DATASET_SCHEMA,
    FEATURE_DIMENSION,
    DatasetError,
    SecondStepStatus,
    build_player_safe_feature,
    build_scorer_candidates,
    encode_candidates,
    materialize_candidate_dataset,
    read_candidate_dataset,
    read_source_record,
)
from lisjong.learning._canonical import canonical_json_line, file_digest
from lisjong.learning.candidate_dataset import (
    CANDIDATES_FILENAME,
    CONTEXT_FILENAME,
    DECISIONS_FILENAME,
)
from lisjong.learning.source_record import SOURCE_RECORD_SCHEMA_V1
from lisjong.policy_contract import AnkanAction, DecisionContext


class CandidateDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = read_source_record(
            cf.write_candidate_source_record(self.root / "source-record")
        )
        self.dataset = materialize_candidate_dataset(self.source, self.root / "dataset")

    def test_row_eligibility_follows_the_serving_o0_precedence(self) -> None:
        splits = self.dataset.manifest["rows"]["splits"]

        for name in ("TRAIN", "SELECT", "OFFLINE-EVAL"):
            with self.subTest(split=name):
                # mixed_decisions: 3 discard / tsumo+ron / riichi / pon response
                self.assertEqual(splits[name]["source_decisions"], 7)
                self.assertEqual(splits[name]["scorer_decisions"], 3)
                self.assertEqual(splits[name]["excluded_win"], 2)
                self.assertEqual(splits[name]["excluded_riichi"], 1)
                self.assertEqual(splits[name]["excluded_response"], 1)
        self.assertEqual(self.dataset.decision_count, 9)
        self.assertEqual(
            self.dataset.manifest["dataset_schema"], CANDIDATE_DATASET_SCHEMA
        )

    def test_teacher_label_resolves_to_the_canonical_candidate(self) -> None:
        scorer_decisions = [
            decision
            for decision in self.source.decisions()
            if decision.decision_ordinal in (0, 2, 5)
        ]
        self.assertEqual(len(scorer_decisions), len(self.dataset.rows))
        for row, decision in zip(self.dataset.rows, scorer_decisions, strict=True):
            with self.subTest(row=row.decision_ordinal):
                self.assertEqual(row.decision_ordinal, decision.decision_ordinal)
                self.assertEqual(row.teacher_candidate.action, decision.selected_action)

    def test_rows_hold_the_serving_candidate_semantics_and_encoding(self) -> None:
        decisions = {
            (d.game_ordinal, d.decision_ordinal): d for d in self.source.decisions()
        }
        for index, row in enumerate(self.dataset.rows):
            decision = decisions[(row.game_ordinal, row.decision_ordinal)]
            context = DecisionContext(
                input=decision.policy_input, legal_actions=decision.legal_actions
            )
            expected = build_scorer_candidates(context)
            with self.subTest(index=index):
                self.assertEqual(row.candidates, expected)
                start = row.candidate_offset * len(encode_candidates(expected)[0])
                vectors = encode_candidates(expected)
                flat = [value for vector in vectors for value in vector]
                stored = self.dataset.candidates[start : start + len(flat)]
                self.assertEqual(list(stored), [float(value) for value in _f32(flat)])
                shared = self.dataset.context[
                    index * FEATURE_DIMENSION : (index + 1) * FEATURE_DIMENSION
                ]
                self.assertEqual(
                    list(shared),
                    list(_f32(build_player_safe_feature(decision.policy_input))),
                )

    def test_candidate_counts_are_variable_and_ragged(self) -> None:
        counts = {row.candidate_count for row in self.dataset.rows}
        self.assertGreater(len(counts), 1)
        offsets = [row.candidate_offset for row in self.dataset.rows]
        self.assertEqual(offsets[0], 0)
        for previous, current in zip(self.dataset.rows, self.dataset.rows[1:]):
            self.assertEqual(
                current.candidate_offset,
                previous.candidate_offset + previous.candidate_count,
            )
        self.assertEqual(
            self.dataset.candidate_count,
            sum(row.candidate_count for row in self.dataset.rows),
        )

    def test_two_pass_availability_is_preserved(self) -> None:
        statuses = {
            candidate.second_step_status
            for row in self.dataset.rows
            for candidate in row.candidates
        }
        self.assertEqual(statuses, set(SecondStepStatus))

    def test_split_membership_is_not_redistributed(self) -> None:
        population = {
            entry["game_ordinal"]: entry["split"]
            for entry in self.source.provenance()["population"]
        }
        for row in self.dataset.rows:
            self.assertEqual(row.split, population[row.game_ordinal])
        self.assertEqual(
            self.dataset.split_counts(),
            {"TRAIN": 3, "SELECT": 3, "OFFLINE-EVAL": 3},
        )

    def test_source_provenance_and_allocation_bindings_propagate(self) -> None:
        source = self.dataset.manifest["source"]

        self.assertEqual(source, self.source.provenance())
        self.assertIsNotNone(source["allocation_bindings"])
        self.assertEqual(
            set(source["allocation_bindings"]), {"TRAIN", "SELECT", "OFFLINE-EVAL"}
        )
        self.assertEqual(
            source["source_contract_digest"], self.source.source_contract_digest
        )

    def test_rematerialization_is_deterministic(self) -> None:
        again = materialize_candidate_dataset(self.source, self.root / "again")
        self.assertEqual(again.identity, self.dataset.identity)

    def test_existing_destination_is_never_overwritten(self) -> None:
        with self.assertRaises(DatasetError):
            materialize_candidate_dataset(self.source, self.root / "dataset")
        self.assertEqual(
            read_candidate_dataset(self.root / "dataset").identity,
            self.dataset.identity,
        )

    def test_historical_v1_source_is_rejected(self) -> None:
        games = [
            (
                "TRAIN",
                100,
                cf.source_rows(
                    cf.mixed_decisions(), game_ordinal=0, seed=100, split="TRAIN"
                ),
            )
        ]
        source = read_source_record(
            fixtures.write_source_record(
                self.root / "v1", games, schema=SOURCE_RECORD_SCHEMA_V1
            )
        )
        with self.assertRaises(DatasetError):
            materialize_candidate_dataset(source, self.root / "from-v1")
        self.assertFalse((self.root / "from-v1").exists())


class TeacherLabelFailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_non_discard_teacher_label_on_a_scorer_decision_fails_closed(self) -> None:
        decision = cf.ankan_decision()
        ankan = next(
            action
            for action in decision.legal_actions
            if isinstance(action, AnkanAction)
        )
        rows = cf.source_rows(
            (decision,),
            game_ordinal=0,
            seed=100,
            split="TRAIN",
            teacher=lambda _decision: ankan,
        )
        source = read_source_record(
            fixtures.write_source_record(self.root / "source", [("TRAIN", 100, rows)])
        )

        with self.assertRaisesRegex(DatasetError, "not a DiscardAction"):
            materialize_candidate_dataset(source, self.root / "dataset")
        self.assertFalse((self.root / "dataset").exists())


class StrictReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = read_source_record(
            cf.write_candidate_source_record(self.root / "source-record")
        )
        self.path = self.root / "dataset"
        materialize_candidate_dataset(source, self.path)

    def _mutate_manifest(self, mutate) -> None:
        fixtures.mutate_manifest(self.path, mutate)

    def test_encoding_fingerprint_mismatch_fails_closed(self) -> None:
        self._mutate_manifest(
            lambda body: body["encoding"].update(fingerprint="0" * 64)
        )
        with self.assertRaisesRegex(DatasetError, "encoding"):
            read_candidate_dataset(self.path)

    def test_request_policy_mismatch_fails_closed(self) -> None:
        self._mutate_manifest(
            lambda body: body["encoding"].update(
                second_step_request_policy="all-candidate"
            )
        )
        with self.assertRaises(DatasetError):
            read_candidate_dataset(self.path)

    def test_shared_feature_fingerprint_mismatch_fails_closed(self) -> None:
        self._mutate_manifest(lambda body: body["feature"].update(fingerprint="0" * 64))
        with self.assertRaisesRegex(DatasetError, "shared feature"):
            read_candidate_dataset(self.path)

    def test_label_semantics_mismatch_fails_closed(self) -> None:
        self._mutate_manifest(lambda body: body["label"].update(semantics="relabel"))
        with self.assertRaises(DatasetError):
            read_candidate_dataset(self.path)

    def test_broken_manifest_seal_fails_closed(self) -> None:
        fixtures.mutate_manifest(self.path, lambda body: None, identity="0" * 64)
        with self.assertRaises(DatasetError):
            read_candidate_dataset(self.path)

    def test_payload_digest_mismatch_fails_closed(self) -> None:
        payload = self.path / CONTEXT_FILENAME
        data = bytearray(payload.read_bytes())
        data[0] ^= 0x01
        payload.write_bytes(bytes(data))
        with self.assertRaisesRegex(DatasetError, "digest"):
            read_candidate_dataset(self.path)

    def test_candidate_semantic_that_disagrees_with_the_payload_fails_closed(
        self,
    ) -> None:
        """digestを合わせてもtyped semanticと再encode結果が一致しなければ拒否する。"""
        decisions = self.path / DECISIONS_FILENAME
        rows = [json.loads(line) for line in decisions.read_text("utf-8").splitlines()]
        rows[0]["candidates"][0]["current_ukeire_count"] += 1
        decisions.write_text(
            "".join(canonical_json_line(row) for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        self._mutate_manifest(
            lambda body: body["files"][DECISIONS_FILENAME].update(
                file_digest(decisions)
            )
        )
        with self.assertRaisesRegex(DatasetError, "typed candidate semantic"):
            read_candidate_dataset(self.path)

    def test_teacher_index_out_of_range_fails_closed(self) -> None:
        decisions = self.path / DECISIONS_FILENAME
        rows = [json.loads(line) for line in decisions.read_text("utf-8").splitlines()]
        rows[0]["teacher_candidate_index"] = rows[0]["candidate_count"]
        decisions.write_text(
            "".join(canonical_json_line(row) for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        self._mutate_manifest(
            lambda body: body["files"][DECISIONS_FILENAME].update(
                file_digest(decisions)
            )
        )
        with self.assertRaisesRegex(DatasetError, "teacher candidate index"):
            read_candidate_dataset(self.path)

    def test_exclusion_accounting_mismatch_fails_closed(self) -> None:
        self._mutate_manifest(
            lambda body: body["rows"]["splits"]["TRAIN"].update(excluded_win=0)
        )
        with self.assertRaisesRegex(DatasetError, "exclusion accounting"):
            read_candidate_dataset(self.path)

    def test_unexpected_file_fails_closed(self) -> None:
        (self.path / "extra.bin").write_bytes(b"")
        with self.assertRaises(DatasetError):
            read_candidate_dataset(self.path)

    def test_missing_payload_fails_closed(self) -> None:
        (self.path / CANDIDATES_FILENAME).unlink()
        with self.assertRaises(DatasetError):
            read_candidate_dataset(self.path)


def _f32(values):
    from array import array

    return array("f", values)


if __name__ == "__main__":
    unittest.main()
