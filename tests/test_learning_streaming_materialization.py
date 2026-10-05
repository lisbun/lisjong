"""Issue #247 逐次書出しのdataset materializationのtest。

`open_source_record()` + `publish_dataset()` / `publish_candidate_dataset()`が、
全件保持の経路（`read_source_record()` + `materialize_*()`）と同一のbytes・
identityを出し、途中で失敗した場合に部分出力を正式な出力として残さないことを
固定する。
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import candidate_fixtures as cf
import learning_fixtures as fixtures

from lisjong.learning import (
    SOURCE_RECORD_SCHEMA_V1,
    DatasetError,
    SourceRecordError,
    StreamingSourceRecord,
    build_player_safe_feature,
    materialize_candidate_dataset,
    materialize_dataset,
    open_source_record,
    publish_candidate_dataset,
    publish_dataset,
    read_source_record,
)
from lisjong.learning.__main__ import main
from lisjong.learning._canonical import file_digest


def _payload_digests(root: Path) -> dict[str, dict[str, object]]:
    return {child.name: file_digest(child) for child in sorted(root.iterdir())}


class StreamingSourceRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def sources(self):
        yield "v2", fixtures.write_source_record(self.root / "v2")
        yield (
            "v1",
            fixtures.write_source_record(
                self.root / "v1", schema=SOURCE_RECORD_SCHEMA_V1
            ),
        )
        yield "policy", fixtures.write_policy_source_record(self.root / "policy")

    def test_streaming_record_matches_the_full_read(self) -> None:
        for name, path in self.sources():
            with self.subTest(source=name):
                full = read_source_record(path)
                stream = open_source_record(path)

                self.assertIsInstance(stream, StreamingSourceRecord)
                self.assertEqual(stream.identity, full.identity)
                self.assertEqual(stream.decision_count, full.decision_count)
                self.assertEqual(stream.population(), full.population())
                self.assertEqual(stream.provenance(), full.provenance())
                self.assertEqual(list(stream.decisions()), list(full.decisions()))
                self.assertEqual(tuple(stream.games()), full.games)

    def test_manifest_level_violations_fail_when_opening(self) -> None:
        path = fixtures.write_source_record(self.root / "source")
        fixtures.mutate_game_summary(path, 1, lambda summary: summary.update(seed=100))

        with self.assertRaisesRegex(SourceRecordError, "reuse a seed"):
            open_source_record(path)

    def test_game_payload_violation_fails_during_enumeration(self) -> None:
        path = fixtures.write_source_record(self.root / "source")
        rows = fixtures.read_game_rows(path, 1)
        fixtures.write_game_rows(path, 1, rows[:1], update_manifest=False)

        stream = open_source_record(path)
        decisions = stream.decisions()
        first_game = [next(decisions) for _ in range(3)]
        self.assertEqual({decision.game_ordinal for decision in first_game}, {0})
        with self.assertRaisesRegex(SourceRecordError, "digest mismatch"):
            next(decisions)


class StreamingMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()

    def test_bc_dataset_bytes_and_identity_match_the_full_path(self) -> None:
        for name, path in (
            ("v2", fixtures.write_source_record(self.root / "v2")),
            ("policy", fixtures.write_policy_source_record(self.root / "policy")),
        ):
            with self.subTest(source=name):
                full = materialize_dataset(
                    read_source_record(path), self.outputs / f"{name}-full"
                )
                manifest = publish_dataset(
                    open_source_record(path), self.outputs / f"{name}-stream"
                )

                self.assertEqual(manifest["identity"], full.identity)
                self.assertEqual(manifest, dict(full.manifest))
                self.assertEqual(
                    _payload_digests(self.outputs / f"{name}-stream"),
                    _payload_digests(self.outputs / f"{name}-full"),
                )

    def test_candidate_dataset_bytes_and_identity_match_the_full_path(self) -> None:
        path = cf.write_candidate_source_record(self.root / "source")
        full = materialize_candidate_dataset(
            read_source_record(path), self.outputs / "full"
        )
        manifest = publish_candidate_dataset(
            open_source_record(path), self.outputs / "stream"
        )

        self.assertEqual(manifest["identity"], full.identity)
        self.assertEqual(manifest, dict(full.manifest))
        self.assertEqual(
            _payload_digests(self.outputs / "stream"),
            _payload_digests(self.outputs / "full"),
        )

    def test_corrupt_later_game_leaves_no_partial_output(self) -> None:
        path = cf.write_candidate_source_record(self.root / "source")
        rows = fixtures.read_game_rows(path, 1)
        fixtures.write_game_rows(path, 1, rows[:1], update_manifest=False)

        for publish in (publish_dataset, publish_candidate_dataset):
            with self.subTest(publish=publish.__name__):
                destination = self.outputs / publish.__name__
                with self.assertRaisesRegex(SourceRecordError, "digest mismatch"):
                    publish(open_source_record(path), destination)
                self.assertEqual(list(self.outputs.iterdir()), [])

    def test_failure_after_written_rows_leaves_no_partial_output(self) -> None:
        # 先頭gameの行をstaging fileへ書いた後、後続gameの変換で失敗させる。
        path = cf.write_candidate_source_record(self.root / "source")
        for module, publish in (
            ("dataset", publish_dataset),
            ("candidate_dataset", publish_candidate_dataset),
        ):
            with self.subTest(publish=publish.__name__):
                target = f"lisjong.learning.{module}.build_player_safe_feature"
                original = build_player_safe_feature
                calls = []

                def failing(policy_input, calls=calls):
                    calls.append(policy_input)
                    if len(calls) > 3:
                        raise DatasetError("late failure")
                    return original(policy_input)

                with mock.patch(target, side_effect=failing):
                    with self.assertRaisesRegex(DatasetError, "late failure"):
                        publish(open_source_record(path), self.outputs / module)
                self.assertGreater(len(calls), 3)
                self.assertEqual(list(self.outputs.iterdir()), [])

    def test_cli_materializes_without_loading_the_dataset(self) -> None:
        path = fixtures.write_source_record(self.root / "source")
        expected = materialize_dataset(
            read_source_record(path), self.outputs / "expected"
        )
        output = self.outputs / "cli"
        stdout = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            mock.patch(
                "lisjong.learning.dataset.read_dataset",
                side_effect=AssertionError("the CLI must not load the dataset"),
            ),
            mock.patch(
                "lisjong.learning.__main__.read_source_record",
                side_effect=AssertionError("the CLI must not load the source"),
            ),
        ):
            code = main(
                [
                    "materialize-dataset",
                    "--source-record",
                    str(path),
                    "--output",
                    str(output),
                ]
            )

        self.assertEqual(code, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["dataset_identity"], expected.identity)
        self.assertEqual(summary["row_count"], expected.row_count)
        self.assertEqual(summary["splits"], expected.split_counts())
        self.assertEqual(
            _payload_digests(output), _payload_digests(self.outputs / "expected")
        )


if __name__ == "__main__":
    unittest.main()
