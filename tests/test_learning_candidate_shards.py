"""Issue #248 候補datasetのgame単位shard並列変換のtest。

worker数によらず出力bytes・dataset identityが同一であること、1 shardの失敗で
全体が失敗し部分出力が残らないこと、先頭shardが遅い場合も投入済みで未連結の
shard数が上限を超えないことを固定する。
"""

import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import candidate_fixtures as cf
import learning_fixtures as fixtures

from lisjong.learning import (
    DatasetError,
    SourceRecordError,
    open_source_record,
    publish_candidate_dataset,
    read_source_record,
)
from lisjong.learning import candidate_dataset as module
from lisjong.learning.__main__ import main
from lisjong.learning._canonical import file_digest

_SPLITS = ("TRAIN", "TRAIN", "SELECT", "OFFLINE-EVAL", "TRAIN", "SELECT")


def _digests(root: Path) -> dict[str, dict[str, object]]:
    return {child.name: file_digest(child) for child in sorted(root.iterdir())}


class CandidateShardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = cf.write_candidate_source_record(
            self.root / "source", splits=_SPLITS
        )
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.sequential = publish_candidate_dataset(
            open_source_record(self.path), self.outputs / "sequential"
        )

    def assert_same_as_sequential(self, name: str, manifest) -> None:
        self.assertEqual(manifest, self.sequential)
        self.assertEqual(
            _digests(self.outputs / name), _digests(self.outputs / "sequential")
        )

    def test_parallel_output_is_identical_to_one_worker(self) -> None:
        for workers, pending in ((2, None), (3, 1), (3, 2), (4, 8)):
            name = f"workers-{workers}-pending-{pending}"
            with self.subTest(name):
                manifest = publish_candidate_dataset(
                    open_source_record(self.path),
                    self.outputs / name,
                    workers=workers,
                    max_pending_shards=pending,
                )
                self.assert_same_as_sequential(name, manifest)

    def test_one_failed_shard_fails_the_whole_conversion(self) -> None:
        rows = fixtures.read_game_rows(self.path, 3)
        fixtures.write_game_rows(self.path, 3, rows[:1], update_manifest=False)

        with self.assertRaisesRegex(SourceRecordError, "digest mismatch"):
            publish_candidate_dataset(
                open_source_record(self.path), self.outputs / "failed", workers=3
            )
        self.assertEqual(
            sorted(child.name for child in self.outputs.iterdir()), ["sequential"]
        )

    def test_slow_head_shard_does_not_exceed_the_pending_bound(self) -> None:
        workers, bound = 3, 3
        started: list[int] = []
        seen_at_head_release: list[int] = []
        lock = threading.Lock()
        original = module._materialize_shard

        def slow_head(root, summary, shard):
            with lock:
                started.append(summary["game_ordinal"])
            if summary["game_ordinal"] == 0:
                # 他のworkerが空いても、上限を超える投入は起きないことを見る。
                time.sleep(0.5)
                with lock:
                    seen_at_head_release.extend(started)
            return original(root, summary, shard)

        with (
            mock.patch.object(
                module, "_shard_executor", lambda count: ThreadPoolExecutor(count)
            ),
            mock.patch.object(module, "_materialize_shard", side_effect=slow_head),
        ):
            manifest = publish_candidate_dataset(
                open_source_record(self.path),
                self.outputs / "bounded",
                workers=workers,
                max_pending_shards=bound,
            )

        self.assertEqual(sorted(seen_at_head_release), [0, 1, 2])
        self.assertEqual(sorted(started), list(range(len(_SPLITS))))
        self.assert_same_as_sequential("bounded", manifest)

    def test_invalid_parallel_settings_fail_closed(self) -> None:
        for kwargs, message in (
            ({"workers": 0}, "workers"),
            ({"workers": 2, "max_pending_shards": 0}, "max_pending_shards"),
        ):
            with self.subTest(kwargs):
                with self.assertRaisesRegex(DatasetError, message):
                    publish_candidate_dataset(
                        open_source_record(self.path), self.outputs / "x", **kwargs
                    )
        with self.assertRaisesRegex(DatasetError, "open_source_record"):
            publish_candidate_dataset(
                read_source_record(self.path), self.outputs / "x", workers=2
            )
        self.assertFalse((self.outputs / "x").exists())

    def test_cli_workers_produce_the_same_dataset(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(
                [
                    "materialize-candidate-dataset",
                    "--source-record",
                    str(self.path),
                    "--output",
                    str(self.outputs / "cli"),
                    "--workers",
                    "2",
                    "--max-pending-shards",
                    "3",
                ]
            )
        self.assertEqual(code, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["dataset_identity"], self.sequential["identity"])
        self.assertEqual(
            _digests(self.outputs / "cli"), _digests(self.outputs / "sequential")
        )


if __name__ == "__main__":
    unittest.main()
