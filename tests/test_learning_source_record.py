"""Arena player-safe source record consumerのcontract test。"""

import json
import tempfile
import unittest
from pathlib import Path

import learning_fixtures as fixtures

from lisjong.learning import (
    SOURCE_RECORD_SCHEMA_V1,
    SOURCE_RECORD_SCHEMA_V2,
    SourceRecordError,
    UnsupportedSourceSchemaError,
    read_source_record,
)
from lisjong.learning._canonical import canonical_json_line, canonical_json_text
from lisjong.learning._typed_values import action_to_value
from lisjong.policy_contract import DiscardAction, Seat


class SourceRecordReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def record(self, games=None, **overrides) -> Path:
        return fixtures.write_source_record(
            self.root / "source-record", games, **overrides
        )

    def test_supported_schema_round_trip_preserves_provenance(self) -> None:
        record = read_source_record(self.record())

        self.assertEqual(record.schema, SOURCE_RECORD_SCHEMA_V2)
        self.assertEqual(record.lock_identity, fixtures.LOCK_IDENTITY)
        self.assertEqual(record.scientific_corpus_identity, fixtures.CORPUS_IDENTITY)
        self.assertEqual(record.game_mode, fixtures.GAME_MODE)
        self.assertEqual(len(record.games), 2)
        self.assertEqual(record.decision_count, 5)
        self.assertEqual(
            [(game.split, game.seed) for game in record.games],
            [("TRAIN", 100), ("SELECT", 200)],
        )
        self.assertEqual(
            record.population(),
            (
                {
                    "decision_count": 3,
                    "game_ordinal": 0,
                    "seed": 100,
                    "split": "TRAIN",
                },
                {
                    "decision_count": 2,
                    "game_ordinal": 1,
                    "seed": 200,
                    "split": "SELECT",
                },
            ),
        )

        decisions = list(record.decisions())
        self.assertEqual([item.decision_ordinal for item in decisions], [0, 1, 2, 0, 1])
        for decision in decisions:
            self.assertEqual(decision.policy_input.self_seat, decision.actor_seat)
            self.assertIn(decision.selected_action, decision.legal_actions)
            for action in decision.legal_actions:
                self.assertEqual(action.actor, decision.actor_seat)

    def test_provenance_binds_source_identity_without_arena_semantics(self) -> None:
        record = read_source_record(self.record())
        provenance = record.provenance()

        self.assertEqual(provenance["schema"], SOURCE_RECORD_SCHEMA_V2)
        self.assertEqual(provenance["identity"], record.identity)
        self.assertEqual(provenance["decision_count"], 5)
        self.assertEqual(len(provenance["source_contract_digest"]), 64)
        # Arena-owned qualification bindingはdigestとしてのみ保持し、
        # historical Arena identityをlisjong側へ持ち上げない。
        self.assertNotIn("source_contract", provenance)
        self.assertNotIn("teacher", json.dumps(provenance))
        self.assertEqual(set(provenance["allocation_bindings"]), {"TRAIN", "SELECT"})

    def test_v2_allocation_bindings_are_read_and_preserved_exactly(self) -> None:
        root = self.record()
        expected = fixtures.read_manifest_body(root)["allocation_bindings"]

        record = read_source_record(root)

        self.assertEqual(record.schema, SOURCE_RECORD_SCHEMA_V2)
        self.assertIsNotNone(record.allocation_bindings)
        self.assertEqual(dict(record.allocation_bindings), expected)
        for split, binding in record.allocation_bindings.items():
            self.assertEqual(
                set(binding),
                {
                    "allocation_identity",
                    "ledger_revision",
                    "owner_repository",
                    "seed_domain",
                    "seed_membership_identity",
                },
            )
            self.assertEqual(binding["owner_repository"], "lisbun/lisjong-arena")

    def test_v1_historical_record_has_no_allocation_bindings(self) -> None:
        root = self.record(schema=SOURCE_RECORD_SCHEMA_V1)

        record = read_source_record(root)

        self.assertEqual(record.schema, SOURCE_RECORD_SCHEMA_V1)
        self.assertIsNone(record.allocation_bindings)
        self.assertIsNone(record.provenance()["allocation_bindings"])

    def test_v1_manifest_rejects_an_allocation_bindings_field(self) -> None:
        """v1はallocation provenanceを持たない: v1へ推測でv2相当を足さない。"""
        root = self.record(schema=SOURCE_RECORD_SCHEMA_V1)
        fixtures.mutate_manifest(
            root,
            lambda body: body.__setitem__(
                "allocation_bindings",
                {"TRAIN": fixtures.allocation_binding([100])},
            ),
        )

        with self.assertRaisesRegex(SourceRecordError, "unexpected fields"):
            read_source_record(root)

    def test_v2_missing_split_binding_rejected(self) -> None:
        root = self.record()
        fixtures.mutate_manifest(
            root, lambda body: body["allocation_bindings"].pop("SELECT")
        )

        with self.assertRaisesRegex(
            SourceRecordError, "do not match the source population splits"
        ):
            read_source_record(root)

    def test_v2_extra_split_binding_rejected(self) -> None:
        root = self.record()
        fixtures.mutate_manifest(
            root,
            lambda body: body["allocation_bindings"].__setitem__(
                "OFFLINE-EVAL", fixtures.allocation_binding([999])
            ),
        )

        with self.assertRaisesRegex(
            SourceRecordError, "do not match the source population splits"
        ):
            read_source_record(root)

    def test_v2_missing_allocation_bindings_field_rejected(self) -> None:
        root = self.record()
        fixtures.mutate_manifest(root, lambda body: body.pop("allocation_bindings"))

        with self.assertRaisesRegex(SourceRecordError, "unexpected fields"):
            read_source_record(root)

    def test_v2_malformed_binding_shape_rejected(self) -> None:
        root = self.record()
        fixtures.mutate_manifest(
            root, lambda body: body["allocation_bindings"]["TRAIN"].pop("seed_domain")
        )

        with self.assertRaisesRegex(SourceRecordError, "unexpected fields"):
            read_source_record(root)

    def test_v2_non_digest_allocation_identity_rejected(self) -> None:
        root = self.record()
        fixtures.mutate_manifest(
            root,
            lambda body: body["allocation_bindings"]["TRAIN"].__setitem__(
                "allocation_identity", "not-a-digest"
            ),
        )

        with self.assertRaisesRegex(SourceRecordError, "SHA-256"):
            read_source_record(root)

    def test_v2_wrong_owner_repository_rejected(self) -> None:
        root = self.record()
        fixtures.mutate_manifest(
            root,
            lambda body: body["allocation_bindings"]["TRAIN"].__setitem__(
                "owner_repository", "someone-else/not-arena"
            ),
        )

        with self.assertRaisesRegex(SourceRecordError, "canonical Arena owner"):
            read_source_record(root)

    def test_v2_invalid_seed_domain_rejected(self) -> None:
        root = self.record()
        fixtures.mutate_manifest(
            root,
            lambda body: body["allocation_bindings"]["TRAIN"].__setitem__(
                "seed_domain", "Not A Valid Domain!"
            ),
        )

        with self.assertRaisesRegex(SourceRecordError, "invalid format"):
            read_source_record(root)

    def test_v2_seed_membership_contradiction_rejected(self) -> None:
        """binding.seed_membership_identityが実populationのseedと矛盾する場合。"""
        root = self.record()
        fixtures.mutate_manifest(
            root,
            lambda body: body["allocation_bindings"]["TRAIN"].update(
                fixtures.allocation_binding([999, 1000])
            ),
        )

        with self.assertRaisesRegex(
            SourceRecordError, "contradicts the source population"
        ):
            read_source_record(root)

    def test_unknown_schema_fails_closed(self) -> None:
        root = self.record()
        fixtures.mutate_manifest(
            root, lambda body: body.update(schema="arena-future-source-record-v9")
        )

        with self.assertRaises(UnsupportedSourceSchemaError):
            read_source_record(root)

    def test_unknown_kind_rejected(self) -> None:
        root = self.record()
        fixtures.mutate_manifest(root, lambda body: body.update(kind="something-else"))

        with self.assertRaisesRegex(SourceRecordError, "kind mismatch"):
            read_source_record(root)

    def test_unexpected_manifest_field_rejected(self) -> None:
        root = self.record()
        fixtures.mutate_manifest(root, lambda body: body.update(privileged_wall=["1m"]))

        with self.assertRaisesRegex(SourceRecordError, "unexpected fields"):
            read_source_record(root)

    def test_manifest_identity_mismatch_rejected(self) -> None:
        root = self.record()
        body = fixtures.read_manifest_body(root)
        fixtures.rewrite_manifest(root, body, identity="0" * 64)

        with self.assertRaisesRegex(SourceRecordError, "identity mismatch"):
            read_source_record(root)

    def test_non_canonical_manifest_rejected(self) -> None:
        root = self.record()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=4, sort_keys=True) + "\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(SourceRecordError, "canonical JSON"):
            read_source_record(root)

    def test_game_provenance_mismatch_rejected(self) -> None:
        root = self.record()
        fixtures.mutate_game_summary(
            root, 0, lambda summary: summary.update(lock_identity="c" * 64)
        )

        with self.assertRaisesRegex(SourceRecordError, "does not match the manifest"):
            read_source_record(root)

    def test_duplicate_source_seed_rejected(self) -> None:
        rows_a = fixtures.decision_rows(game_ordinal=0, seed=100, split="TRAIN")
        rows_b = fixtures.decision_rows(
            game_ordinal=1, seed=100, split="SELECT", count=2
        )
        root = self.record(
            (("TRAIN", 100, rows_a), ("SELECT", 100, rows_b)),
        )

        with self.assertRaisesRegex(SourceRecordError, "reuse a seed"):
            read_source_record(root)

    def test_payload_digest_mismatch_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        fixtures.write_game_rows(root, 0, rows[:-1], update_manifest=False)

        with self.assertRaisesRegex(SourceRecordError, "digest mismatch"):
            read_source_record(root)

    def test_missing_decision_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        fixtures.write_game_rows(root, 0, rows[:-1])
        fixtures.mutate_game_summary(
            root, 0, lambda summary: summary.update(decision_count=3)
        )

        with self.assertRaisesRegex(SourceRecordError, "decision count mismatch"):
            read_source_record(root)

    def test_extra_game_directory_rejected(self) -> None:
        root = self.record()
        (root / "game-002").mkdir()

        with self.assertRaisesRegex(SourceRecordError, "missing/unexpected"):
            read_source_record(root)

    def test_unexpected_payload_file_rejected(self) -> None:
        root = self.record()
        (root / "game-000" / "features.f32").write_bytes(b"\x00")

        with self.assertRaisesRegex(SourceRecordError, "missing/unexpected payload"):
            read_source_record(root)

    def test_duplicate_decision_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        fixtures.write_game_rows(root, 0, [rows[0], rows[0], rows[1], rows[2]])

        with self.assertRaisesRegex(SourceRecordError, "decision ordinal"):
            read_source_record(root)

    def test_row_provenance_mismatch_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        rows[1]["seed"] = 999
        fixtures.write_game_rows(root, 0, rows)

        with self.assertRaisesRegex(SourceRecordError, "provenance"):
            read_source_record(root)

    def test_execution_ordering_mismatch_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        rows[1]["step_ordinal"] = 7
        fixtures.write_game_rows(root, 0, rows)

        with self.assertRaisesRegex(SourceRecordError, "ordering mismatch"):
            read_source_record(root)

    def test_actor_seat_mismatch_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        rows[0]["actor_seat"] = 3
        fixtures.write_game_rows(root, 0, rows)

        with self.assertRaisesRegex(SourceRecordError, "seat does not match"):
            read_source_record(root)

    def test_illegal_selected_action_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        value = fixtures.policy_input(self_seat=Seat.SEAT_0)
        illegal = DiscardAction(
            actor=Seat.SEAT_0, tile=fixtures.tile(fixtures.HONOR, 7), tsumogiri=False
        )
        self.assertNotIn(illegal, fixtures.discard_legal_actions(value))
        rows[0]["teacher_selected_action"] = action_to_value(
            illegal, ValueError, "action"
        )
        fixtures.write_game_rows(root, 0, rows)

        with self.assertRaisesRegex(SourceRecordError, "not legal"):
            read_source_record(root)

    def test_selected_action_actor_mismatch_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        rows[0]["teacher_selected_action"]["actor"] = 2
        fixtures.write_game_rows(root, 0, rows)

        with self.assertRaises(SourceRecordError):
            read_source_record(root)

    def test_non_canonical_legal_action_order_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        rows[0]["legal_actions"] = list(reversed(rows[0]["legal_actions"]))
        fixtures.write_game_rows(root, 0, rows)

        with self.assertRaisesRegex(SourceRecordError, "canonically ordered"):
            read_source_record(root)

    def test_duplicate_legal_action_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        rows[0]["legal_actions"] = sorted(
            [*rows[0]["legal_actions"], rows[0]["legal_actions"][0]],
            key=canonical_json_line,
        )
        fixtures.write_game_rows(root, 0, rows)

        with self.assertRaisesRegex(SourceRecordError, "duplicates"):
            read_source_record(root)

    def test_empty_legal_actions_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        rows[0]["legal_actions"] = []
        fixtures.write_game_rows(root, 0, rows)

        with self.assertRaisesRegex(SourceRecordError, "must not be empty"):
            read_source_record(root)

    def test_non_canonical_row_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        fixtures.write_game_rows(root, 0, rows, raw=raw)

        with self.assertRaisesRegex(SourceRecordError, "canonical JSON"):
            read_source_record(root)

    def test_privileged_row_field_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        rows[0]["opponent_concealed_tiles"] = [[], [], [], []]
        fixtures.write_game_rows(root, 0, rows)

        with self.assertRaisesRegex(SourceRecordError, "unexpected fields"):
            read_source_record(root)

    def test_privileged_policy_input_field_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        rows[0]["policy_input"]["live_wall_tiles"] = [{"category": "manzu", "rank": 1}]
        fixtures.write_game_rows(root, 0, rows)

        with self.assertRaisesRegex(SourceRecordError, "unexpected fields"):
            read_source_record(root)

    def test_non_finite_row_value_rejected(self) -> None:
        root = self.record()
        rows = fixtures.read_game_rows(root, 0)
        raw = canonical_json_line(rows[0]).replace('"score":25000', '"score":NaN', 1)
        fixtures.write_game_rows(
            root,
            0,
            rows,
            raw=raw + "".join(canonical_json_line(row) for row in rows[1:]),
        )

        with self.assertRaisesRegex(SourceRecordError, "non-finite|strict JSON"):
            read_source_record(root)

    def test_missing_directory_rejected(self) -> None:
        with self.assertRaisesRegex(SourceRecordError, "does not exist"):
            read_source_record(self.root / "absent")

    def test_consumer_reads_only_the_source_record(self) -> None:
        """Arena corpus artifactを必要とせず、source recordだけで読めること。"""
        root = self.record()
        self.assertEqual(
            sorted(child.name for child in root.iterdir()),
            ["game-000", "game-001", "manifest.json"],
        )
        record = read_source_record(root)
        self.assertEqual(record.decision_count, 5)
        self.assertNotIn(
            "features",
            canonical_json_text(
                {"population": list(record.population())},
            ),
        )


if __name__ == "__main__":
    unittest.main()
