"""L2a runtime互換確認tool（Issue #369、`tools/l2a_runtime_compat.py`）のtest。

合成fixtureだけを使う。実artifact・実sourceでの互換実測は後続の実行Issueで行う。
"""

import copy
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

import candidate_fixtures as cf
import learning_fixtures as fixtures

from lisjong.learning import (
    _typed_values,
    features,
    l2a_decision,
    l2a_policy,
    source_record,
)
from lisjong.learning._typed_values import policy_input_to_value_v2
from lisjong.learning.errors import LearnedPolicyError
from lisjong.learning.l2a_decision import (
    L2aDecisionKind,
    classify_l2a_decision,
    guard_action,
)
from lisjong.learning.l2a_encoding import ENCODING_M1, ENCODING_R1
from lisjong.learning.source_record import (
    GAME_PAYLOAD_FILENAME,
    POLICY_SOURCE_RECORD_SCHEMA_V2,
    open_source_record,
)
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DecisionContext,
    PassAction,
    RonAction,
    Seat,
    Tile,
    TileCategory,
    TileType,
)

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "l2a_runtime_compat.py"
_SPEC = importlib.util.spec_from_file_location("l2a_runtime_compat", TOOL_PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_OBJECTIVE = "per-decision-legal-candidate-softmax-cross-entropy"
OLD = "a" * 40
NEW = "b" * 40


# --- 手で組んだcapture（torch不要） ---------------------------------------------------


def _manifest(games=(("TRAIN", 0, 100, 3), ("SELECT", 1, 200, 2))):
    body = {
        "artifacts": {"m1": "1" * 64, "r1": "2" * 64},
        "classification": tool.FROZEN["classification"],
        "games": [
            {
                "decision_count": count,
                "decision_ordinals": list(range(count)),
                "game_ordinal": ordinal,
                "payload": {"bytes": 10, "sha256": "3" * 64},
                "seed": seed,
                "split": split,
            }
            for split, ordinal, seed, count in games
        ],
        "record_count": sum(game[3] for game in games),
        "schema": tool.INPUT_SCHEMA,
        "source": {
            "identity": "4" * 64,
            "purpose": "SCIENTIFIC",
            "schema": POLICY_SOURCE_RECORD_SCHEMA_V2,
        },
    }
    return tool.validate_input_manifest(tool.seal(body))


def _action(index):
    return {"actor": 0, "kind": "discard", "tile": index}


def _record(key, *, kind="discard", scores=(1.0, 0.5, -0.25)):
    legal = [_action(index) for index in range(len(scores))]
    record = {
        "action": legal[0],
        "input_sha256": tool.digest(key),
        "key": key,
        "kind": kind,
        "legal_actions": legal,
        "scorer": None,
        "single_candidate": False,
    }
    if kind in tool.SCORER_HEADS:
        record["scorer"] = {
            "candidate_vectors": [
                {"dimension": 5, "sha256": tool.digest([kind, index])}
                for index in range(len(scores))
            ],
            "candidates": legal,
            "encoding": {"dimension": 5, "identity": "m1-id", "name": "m1"},
            "head": kind,
            "scores": [float(score).hex() for score in scores],
            "shared_feature": {"dimension": 9, "sha256": tool.digest(key)},
        }
    return record


def _runtime(revision):
    return {
        "classification_identity": tool.FROZEN["classification"],
        "lisjong": {"method": "git-checkout", "revision": revision},
        "native": {
            "api_version": 3,
            "module_sha256": tool.digest(revision),
            "source_revision": revision,
            "wheel_sha256": tool.digest([revision]),
        },
        "platform": "Linux-test",
        "python": "3.14.7",
        "scorer_loader": "load_l2a_scorer",
        "shanten_backend": "rust",
        "torch": {
            "deterministic_algorithms": True,
            "interop_threads": 1,
            "threads": 1,
            "version": "2.13.0+cpu",
        },
    }


def _capture(manifest, revision, records=None):
    if records is None:
        kinds = ("discard", "call", "forced", "win", "pass_response")
        records = [
            _record(key, kind=kinds[index % len(kinds)])
            for index, key in enumerate(tool.expected_keys(manifest))
        ]
    return _finish(
        {
            "artifacts": {
                name: {"identity": identity, "weights": {"sha256": "6" * 64}}
                for name, identity in manifest["artifacts"].items()
            },
            "input_manifest": manifest,
            "metadata": {"wall_seconds": 1.0},
            "records": records,
            "representation": dict(tool.REPRESENTATION),
            "runtime": _runtime(revision),
            "schema": tool.CAPTURE_SCHEMA,
            "status": "COMPLETE",
            "tool": {"name": tool.TOOL_NAME, "sha256": "5" * 64},
        }
    )


def _finish(body):
    """recordsから導出するfieldを再計算する（runtimeが別の値を出した場合の模擬）。"""
    body = {key: value for key, value in body.items() if key != "identity"}
    body["decision_digest"] = tool.digest(body["records"])
    body["summary"] = tool.summarize(body["records"])
    return body


class _Files(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _write(self, name, value) -> Path:
        path = self.root / name
        tool.write_once(path, value)
        return path


class InputManifestTests(unittest.TestCase):
    def _body(self):
        manifest = _manifest()
        return {key: value for key, value in manifest.items() if key != "identity"}

    def _rejects(self, mutate, message) -> None:
        body = self._body()
        mutate(body)
        with self.assertRaisesRegex(tool.CompatError, message):
            tool.validate_input_manifest(tool.seal(body))

    def test_expected_keys_follow_the_fixed_order(self) -> None:
        keys = tool.expected_keys(_manifest())
        self.assertEqual(len(keys), 5)
        self.assertEqual(
            keys[3],
            {"decision_ordinal": 0, "game_ordinal": 1, "seed": 200, "split": "SELECT"},
        )

    def test_offline_eval_and_unknown_splits_are_rejected(self) -> None:
        for split in ("OFFLINE-EVAL", "TEST", "train"):
            with self.subTest(split=split):
                self._rejects(
                    lambda body, split=split: body["games"][0].update(split=split),
                    "is not one of",
                )

    def test_count_order_and_duplicates_are_rejected(self) -> None:
        cases = (
            (lambda b: b["games"][0].update(decision_ordinals=[0, 0]), "increasing"),
            (lambda b: b["games"][0].update(decision_ordinals=[1, 0]), "increasing"),
            (lambda b: b["games"][0].update(decision_ordinals=[]), "nonempty"),
            (lambda b: b["games"][0].update(decision_ordinals=[0, 3]), "exceed"),
            (lambda b: b.update(record_count=4), "record_count"),
            (lambda b: b["games"].reverse(), "game ordinals"),
            (lambda b: b["games"][1].update(seed=100), "repeats a seed"),
            (lambda b: b["games"][0]["payload"].update(sha256="x"), "SHA-256"),
            (lambda b: b["artifacts"].update(m1="short"), "SHA-256"),
            (lambda b: b.update(extra=1), "exactly the fields"),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                self._rejects(mutate, message)

    def test_identity_must_match_the_content(self) -> None:
        manifest = dict(_manifest(), record_count=4)
        with self.assertRaisesRegex(tool.CompatError, "record_count|identity"):
            tool.validate_input_manifest(manifest)
        manifest = dict(_manifest(), identity="0" * 64)
        with self.assertRaisesRegex(tool.CompatError, "identity"):
            tool.validate_input_manifest(manifest)


class DocumentTests(_Files):
    def test_write_once_never_overwrites(self) -> None:
        path = self._write("a.json", {"x": 1})
        with self.assertRaises(FileExistsError):
            tool.write_once(path, {"x": 2})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"x": 1})

    def test_strict_read_rejects_noncanonical_duplicate_and_nonfinite(self) -> None:
        sealed = tool.seal({"schema": "s", "value": 1})
        good = self._write("good.json", sealed)
        self.assertEqual(tool.read_document(good, schema="s", context="d"), sealed)
        cases = {
            "pretty.json": json.dumps(sealed, indent=1),
            "duplicate.json": '{"schema":"s","schema":"s"}\n',
            "nan.json": '{"schema":"s","value":NaN}\n',
            "identity.json": tool.document_text({**sealed, "value": 2}),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                path = self.root / name
                path.write_text(text, encoding="utf-8", newline="\n")
                with self.assertRaises(tool.CompatError):
                    tool.read_document(path, schema="s", context="d")
        with self.assertRaisesRegex(tool.CompatError, "schema"):
            tool.read_document(good, schema="other", context="d")


class CompareTests(_Files):
    def setUp(self) -> None:
        super().setUp()
        self.manifest = _manifest()
        self.manifest_path = self._write("manifest.json", self.manifest)
        self.old = _capture(self.manifest, OLD)
        self.new = _capture(self.manifest, NEW)

    def _compare(self, old=None, new=None, *, seal_new=True):
        old_path = self.root / "old.json"
        new_path = self.root / "new.json"
        for path in (old_path, new_path):
            path.unlink(missing_ok=True)
        tool.write_once(old_path, tool.seal(old or self.old))
        new = new or self.new
        tool.write_once(new_path, tool.seal(new) if seal_new else new)
        return tool.compare(
            manifest_path=self.manifest_path,
            old_path=old_path,
            new_path=new_path,
            old_revision=OLD,
            new_revision=NEW,
        )

    def _mutated(self, mutate):
        new = copy.deepcopy(self.new)
        mutate(new)
        return _finish(new)

    def test_identical_captures_match(self) -> None:
        report = self._compare()
        self.assertEqual(report["status"], "MATCH")
        self.assertTrue(report["decision_digest_equal"])
        self.assertEqual(report["mismatch"]["records"], 0)
        self.assertEqual(report["score_max_abs_diff"], 0.0)
        self.assertFalse(report["frozen_candidate"])
        self.assertEqual(
            report["captures"]["old"]["runtime"]["lisjong"]["revision"], OLD
        )

    def test_metadata_is_not_part_of_the_match(self) -> None:
        new = dict(self.new, metadata={"wall_seconds": 99.0, "host": "elsewhere"})
        self.assertEqual(self._compare(new=new)["status"], "MATCH")

    def test_decision_differences_are_located_by_record_and_field(self) -> None:
        def swap_candidates(body):
            scorer = body["records"][0]["scorer"]
            for field in ("candidates", "candidate_vectors", "scores"):
                scorer[field].reverse()

        def nudge_score(body):
            scores = body["records"][0]["scorer"]["scores"]
            value = float.fromhex(scores[1])
            scores[1] = math.nextafter(value, math.inf).hex()

        cases = {
            "scorer.candidates": swap_candidates,
            "scorer.shared_feature": lambda body: body["records"][1]["scorer"][
                "shared_feature"
            ].update(sha256="7" * 64),
            "scorer.candidate_vectors": lambda body: body["records"][0]["scorer"][
                "candidate_vectors"
            ][2].update(sha256="7" * 64),
            "scorer.scores": nudge_score,
            "action": lambda body: body["records"][0].update(action=_action(1)),
            "kind": lambda body: body["records"][2].update(kind="pass_response"),
            "legal_actions": lambda body: body["records"][3]["legal_actions"].append(
                _action(9)
            ),
            "input_sha256": lambda body: body["records"][4].update(
                input_sha256="8" * 64
            ),
            "scorer": lambda body: body["records"][2].update(
                scorer=_record(body["records"][2]["key"], kind="call")["scorer"],
                kind="call",
            ),
        }
        for field, mutate in cases.items():
            with self.subTest(field=field):
                report = self._compare(new=self._mutated(mutate))
                self.assertEqual(report["status"], "MISMATCH")
                self.assertFalse(report["decision_digest_equal"])
                self.assertEqual(report["mismatch"]["records"], 1)
                self.assertIn(field, report["mismatch"]["fields"])
                detail = report["mismatch"]["details"][0]
                self.assertIn(field, detail["fields"])
                self.assertIn(field, detail["values"])

    def test_score_difference_is_reported_not_tolerated(self) -> None:
        def nudge(body):
            scores = body["records"][0]["scorer"]["scores"]
            scores[0] = (float.fromhex(scores[0]) + 1e-7).hex()

        report = self._compare(new=self._mutated(nudge))
        self.assertEqual(report["status"], "MISMATCH")
        self.assertEqual(report["mismatch"]["fields"], {"scorer.scores": 1})
        self.assertAlmostEqual(report["score_max_abs_diff"], 1e-7)

    def test_provenance_differences_are_invalid(self) -> None:
        cases = {
            "artifact provenance": lambda body: body["artifacts"]["m1"][
                "weights"
            ].update(sha256="9" * 64),
            "revision is not": lambda body: body["runtime"]["lisjong"].update(
                revision="c" * 40
            ),
            "SOURCE_REVISION": lambda body: body["runtime"]["native"].update(
                source_revision=OLD
            ),
            "shanten backend": lambda body: body["runtime"].update(
                shanten_backend="python"
            ),
            "deterministic": lambda body: body["runtime"]["torch"].update(
                deterministic_algorithms=False
            ),
            "torch threads": lambda body: body["runtime"]["torch"].update(threads=8),
            "different tool": lambda body: body["tool"].update(sha256="0" * 64),
        }
        for message, mutate in cases.items():
            with self.subTest(message=message):
                report = self._compare(new=self._mutated(mutate))
                self.assertEqual(report["status"], "INVALID")
                self.assertTrue(
                    any(message in reason for reason in report["invalid_reasons"]),
                    report["invalid_reasons"],
                )

    def test_runtime_version_differences_are_recorded_without_failing(self) -> None:
        def upgrade(body):
            body["runtime"]["torch"]["version"] = "2.14.0+cpu"
            body["runtime"]["python"] = "3.14.8"

        report = self._compare(new=self._mutated(upgrade))
        self.assertEqual(report["status"], "MATCH")
        self.assertEqual(
            set(report["runtime_differences"]), {"native", "python", "torch"}
        )

    def test_same_revision_on_both_sides_is_invalid(self) -> None:
        old_path = self._write("o.json", tool.seal(self.old))
        report = tool.compare(
            manifest_path=self.manifest_path,
            old_path=old_path,
            new_path=old_path,
            old_revision=OLD,
            new_revision=OLD,
        )
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("must differ", " ".join(report["invalid_reasons"]))

    def test_integrity_and_input_set_violations_are_invalid(self) -> None:
        def drop(body):
            body["records"].pop()

        def reorder(body):
            body["records"][0], body["records"][1] = (
                body["records"][1],
                body["records"][0],
            )

        def nan_score(body):
            body["records"][0]["scorer"]["scores"][0] = "nan"

        def illegal_action(body):
            body["records"][2]["action"] = _action(99)

        def short_scores(body):
            body["records"][0]["scorer"]["scores"].pop()

        cases = {
            "fixed input set": drop,
            "fixed input set and order": reorder,
            "non-finite score": nan_score,
            "not a legal action": illegal_action,
            "counts differ": short_scores,
        }
        for message, mutate in cases.items():
            with self.subTest(message=message):
                report = self._compare(new=self._mutated(mutate))
                self.assertEqual(report["status"], "INVALID")
                self.assertIn(message, report["invalid_reasons"][0])

        other = _manifest(games=(("TRAIN", 0, 100, 2),))
        report = self._compare(new=_capture(other, NEW))
        self.assertIn("another input manifest", report["invalid_reasons"][0])

        tampered = tool.seal(self.new)
        tampered["records"][0]["action"] = _action(1)
        report = self._compare(new=tampered, seal_new=False)
        self.assertIn("identity does not match", report["invalid_reasons"][0])

        stale = tool.seal(dict(self.new, decision_digest="0" * 64))
        report = self._compare(new=stale, seal_new=False)
        self.assertIn("decision digest", report["invalid_reasons"][0])

        stop = {"reason": "x", "schema": tool.CAPTURE_SCHEMA, "status": "STOP"}
        report = self._compare(new=stop)
        self.assertIn("not a COMPLETE capture", report["invalid_reasons"][0])

    def test_cli_compare_is_write_once_and_exit_code_tracks_the_status(self) -> None:
        old = self._write("old-cli.json", tool.seal(self.old))
        new = self._write("new-cli.json", tool.seal(self.new))
        output = self.root / "compare.json"
        arguments = [
            "compare",
            "--manifest",
            str(self.manifest_path),
            "--old",
            str(old),
            "--new",
            str(new),
            "--old-revision",
            OLD,
            "--new-revision",
            NEW,
            "--output",
            str(output),
        ]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(tool.main(arguments), 0)
        report = tool.read_document(
            output, schema=tool.COMPARE_SCHEMA, context="compare"
        )
        self.assertEqual(report["status"], "MATCH")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(arguments)
        self.assertEqual(
            tool.read_document(output, schema=tool.COMPARE_SCHEMA, context="c"),
            report,
        )

        arguments[arguments.index("--new-revision") + 1] = "c" * 40
        arguments[-1] = str(self.root / "compare-invalid.json")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(tool.main(arguments), 2)


class RuntimeBlockTests(unittest.TestCase):
    def test_runtime_rules(self) -> None:
        self.assertEqual(tool.check_runtime_block(_runtime(OLD)), [])
        block = _runtime(OLD)
        block["native"]["source_revision"] = NEW
        block["shanten_backend"] = "python"
        block["scorer_loader"] = "load_something"
        reasons = " ".join(tool.check_runtime_block(block))
        for message in ("SOURCE_REVISION", "shanten backend", "scorer loader"):
            self.assertIn(message, reasons)
        self.assertEqual(
            tool.check_runtime_block({"lisjong": {}}), ["runtime block is malformed"]
        )

    def test_record_hash_uses_the_wheel_record_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_bytes(b"abc")
            self.assertEqual(
                tool.record_hash(path),
                "sha256=ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0",
            )

    def test_native_module_must_be_the_one_in_the_given_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "_lisjong_native.cpython-314-x86_64-linux-gnu.so"
            installed = root / "site" / name
            installed.parent.mkdir()
            installed.write_bytes(b"native-v1")
            module = types.SimpleNamespace(
                __file__=str(installed), API_VERSION=3, SOURCE_REVISION=OLD
            )
            wheel = root / "good.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(f"_lisjong_native/{name}", b"native-v1")
                archive.writestr("lisjong_native-0.1.0.dist-info/RECORD", "")
            value = tool.native_provenance(module, wheel)
            self.assertEqual(value["wheel_sha256"], tool._sha256_file(wheel))
            self.assertEqual(value["module_sha256"], tool._sha256_bytes(b"native-v1"))
            self.assertEqual(value["source_revision"], OLD)

            other = root / "other.whl"
            with zipfile.ZipFile(other, "w") as archive:
                archive.writestr(name, b"native-v2")
            with self.assertRaisesRegex(tool.CompatError, "not the one"):
                tool.native_provenance(module, other)
            empty = root / "empty.whl"
            with zipfile.ZipFile(empty, "w") as archive:
                archive.writestr("README", b"")
            with self.assertRaisesRegex(tool.CompatError, "exactly one"):
                tool.native_provenance(module, empty)
            with self.assertRaisesRegex(tool.CompatError, "cannot be read"):
                tool.native_provenance(module, installed)

    def test_float32_bytes_are_little_endian_and_finite(self) -> None:
        self.assertEqual(
            tool.float32_bytes((1.0, 0), "v"), b"\x00\x00\x80\x3f" + bytes(4)
        )
        for values in ((math.nan,), (math.inf,), (1e300,), (True,), ("1",)):
            with self.subTest(values=values):
                with self.assertRaises(tool.CompatError):
                    tool.float32_bytes(values, "v")


class ImportBoundaryTests(unittest.TestCase):
    def test_tool_imports_neither_torch_nor_lisjong(self) -> None:
        code = (
            "import importlib.util, sys\n"
            "sys.modules['torch'] = None\n"
            f"spec = importlib.util.spec_from_file_location('t', {str(TOOL_PATH)!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "assert 'lisjong' not in sys.modules, 'tool imported lisjong'\n"
            "assert sys.modules['torch'] is None\n"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


# --- 合成source / artifactでのcapture（torch必要） ------------------------------------


def _tile(spec):
    return cf.hand(spec)[0]


def _chi_response():
    value = cf.policy_input("23m456p789s11223z", drawn=False)
    return DecisionContext(
        input=value,
        legal_actions=(
            PassAction(actor=value.self_seat),
            ChiAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_3,
                called_tile=_tile("1m"),
                consumed_tiles=(_tile("2m"), _tile("3m")),
            ),
            ChiAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_3,
                called_tile=_tile("4m"),
                consumed_tiles=(_tile("2m"), _tile("3m")),
            ),
        ),
    )


def _single_candidate_discard():
    value = cf.policy_input("1111m456p789s122z3z")
    return DecisionContext(
        input=value,
        legal_actions=(
            *cf.discard_actions(value, tsumogiri_only=True),
            AnkanAction(actor=value.self_seat, tiles=(_tile("1m"),) * 4),
        ),
    )


def _response(legal):
    value = cf.policy_input("123m456p789s1112z", drawn=False)
    return DecisionContext(input=value, legal_actions=legal(value.self_seat))


_EAST = Tile(TileType(TileCategory.HONOR, 1))
_SOUTH = Tile(TileType(TileCategory.HONOR, 2))


def _decisions():
    """FORCED優先・WIN・RIICHI・PASS_RESPONSE・単一候補・DISCARD・CALLを含む。"""
    return (
        cf.discard_decision(cf.SECOND_NEAR_HAND),
        cf.tsumo_decision(),
        cf.riichi_decision(),
        cf.response_decision(),
        _chi_response(),
        _single_candidate_discard(),
        cf.ron_response_decision(),
        # FORCED優先：合法手が1つなら、PassだけでもRonだけでもFORCED。
        _response(lambda seat: (PassAction(actor=seat),)),
        _response(
            lambda seat: (
                RonAction(actor=seat, target=Seat.SEAT_3, winning_tile=_SOUTH),
            )
        ),
        # PASS_RESPONSE：Chi / Ponがなく、Passと大明槓だけ。
        _response(
            lambda seat: (
                PassAction(actor=seat),
                DaiminkanAction(
                    actor=seat,
                    target=Seat.SEAT_3,
                    called_tile=_EAST,
                    consumed_tiles=(_EAST, _EAST, _EAST),
                ),
            )
        ),
    )


def _teacher(decision):
    kind = classify_l2a_decision(decision)
    action = guard_action(decision, kind)
    if action is not None:
        return action
    if kind is L2aDecisionKind.DISCARD:
        return TwoStepUkeirePolicy().choose_action(decision)
    return ENCODING_R1.encode(decision).actions[-1]


def _write_source(root, *, seed_base=100):
    games = []
    for game_ordinal, split in enumerate(("TRAIN", "SELECT", "OFFLINE-EVAL")):
        seed = seed_base * (game_ordinal + 1)
        rows = []
        for ordinal, decision in enumerate(_decisions()):
            value = decision.input
            row = fixtures.source_row(
                game_ordinal=game_ordinal,
                seed=seed,
                split=split,
                step_ordinal=ordinal,
                decision_ordinal=ordinal,
                policy_input_value=replace(
                    value,
                    players=tuple(
                        replace(player, riichi_declaration_discard_order=None)
                        for player in value.players
                    ),
                ),
                legal_actions=decision.legal_actions,
                selected_action=_teacher(decision),
            )
            row["policy_input"] = policy_input_to_value_v2(value, ValueError, "input")
            rows.append(row)
        games.append((split, seed, rows))
    return fixtures.write_policy_source_record(
        root, games, schema=POLICY_SOURCE_RECORD_SCHEMA_V2
    )


def _provenance(revision):
    def provide(runtime, native_wheel):
        block = _runtime(revision)
        return {
            key: block[key]
            for key in ("lisjong", "native", "platform", "python", "shanten_backend")
        }

    return provide


def _modules(**overrides):
    modules = {
        "contract": sys.modules["lisjong.policy_contract"],
        "decision": l2a_decision,
        "features": features,
        "policy": l2a_policy,
        "source": source_record,
        "values": _typed_values,
    }
    modules.update(overrides)
    return modules


class _CountingScorer:
    """実scorerへ委譲し、呼出し回数を数える。"""

    def __init__(self, scorer, calls) -> None:
        self.artifact = scorer.artifact
        self.encoding = scorer.encoding
        self._scorer = scorer
        self._calls = calls

    def score(self, context, vectors):
        self._calls.append(self.encoding.name)
        return self._scorer.score(context, vectors)


@unittest.skipUnless(_HAS_TORCH, "requires the optional ML runtime")
class CaptureTests(_Files):
    @classmethod
    def setUpClass(cls) -> None:
        from lisjong.learning.l2a_artifact import write_l2a_artifact
        from lisjong.learning.l2a_dataset import publish_l2a_dataset, read_l2a_dataset
        from lisjong.learning.l2a_model import L2aScorerConfig

        cls.shared = tempfile.TemporaryDirectory()
        root = Path(cls.shared.name)
        cls.source_path = _write_source(root / "source")
        source = open_source_record(cls.source_path)
        cls.artifacts = {}
        cls.identities = {}
        for encoding in (ENCODING_M1, ENCODING_R1):
            dataset = root / f"{encoding.name}-dataset"
            publish_l2a_dataset(source, dataset, encoding=encoding)
            config = L2aScorerConfig(encoding=encoding, hidden_width=64)
            path = root / f"{encoding.name}-artifact"
            artifact = write_l2a_artifact(
                path,
                dataset=read_l2a_dataset(dataset),
                model_config=config,
                training=fixtures.training_block(objective=_OBJECTIVE),
                weights=tuple(
                    math.sin(index) * 0.05 for index in range(config.parameter_count)
                ),
            )
            cls.artifacts[encoding.name] = path
            cls.identities[encoding.name] = artifact.identity

    @classmethod
    def tearDownClass(cls) -> None:
        cls.shared.cleanup()

    def setUp(self) -> None:
        super().setUp()
        self.source = open_source_record(self.source_path)
        self.manifest = tool.build_input_manifest(
            self.source,
            game_ordinals=(0, 1),
            payload_filename=GAME_PAYLOAD_FILENAME,
            m1_identity=self.identities["m1"],
            r1_identity=self.identities["r1"],
        )
        self.manifest_path = self._write("manifest.json", self.manifest)

    def _limits(self, **options):
        return tool.Limits(
            max_wall_seconds=options.get("wall", 600.0),
            max_rss_mib=options.get("rss", 1e9),
            **({"clock": options["clock"]} if "clock" in options else {}),
        )

    def _capture(self, revision=OLD, *, modules=None, manifest=None, **options):
        body = tool.capture(
            manifest_path=manifest or self.manifest_path,
            source_path=options.get("source", self.source_path),
            m1_path=options.get("m1", self.artifacts["m1"]),
            r1_path=options.get("r1", self.artifacts["r1"]),
            native_wheel=self.root / "native.whl",
            torch_threads=1,
            limits=options.get("limits") or self._limits(),
            runtime=tool.Runtime(modules or _modules()),
            provenance=_provenance(revision),
        )
        return {
            **body,
            "metadata": {},
            "tool": {"name": tool.TOOL_NAME, "sha256": "5" * 64},
        }

    def test_two_runtimes_with_the_same_behavior_match(self) -> None:
        old = self._capture(OLD)
        new = self._capture(NEW)
        self.assertEqual(old["decision_digest"], new["decision_digest"])
        report = tool.compare_captures(
            self.manifest,
            tool.read_capture(
                self._write("old.json", tool.seal(old)), self.manifest, context="o"
            ),
            tool.read_capture(
                self._write("new.json", tool.seal(new)), self.manifest, context="n"
            ),
            old_revision=OLD,
            new_revision=NEW,
        )
        self.assertEqual(report["status"], "MATCH", report["invalid_reasons"])
        self.assertEqual(report["records"], 2 * len(_decisions()))

    def test_every_branch_is_captured_from_the_actual_rows(self) -> None:
        report = self._capture()
        records = report["records"]
        self.assertEqual(len(records), 2 * len(_decisions()))
        # 合成sourceの2半荘に実際にある件数（PASS_RESPONSEは各半荘1行）。
        self.assertEqual(
            report["summary"]["kinds"],
            {
                "call": 4,
                "discard": 4,
                "forced": 4,
                "pass_response": 2,
                "riichi": 2,
                "win": 4,
            },
        )
        self.assertEqual(report["summary"]["single_candidate"], 2)
        self.assertEqual(report["summary"]["scorer_calls"], {"call": 4, "discard": 2})
        by_ordinal = {record["key"]["decision_ordinal"]: record for record in records}
        # FORCED優先：Passだけ・Ronだけの行は、PASS_RESPONSE / WINではなくFORCED。
        for ordinal, action_type in ((7, "pass"), (8, "ron")):
            record = by_ordinal[ordinal]
            self.assertEqual(record["kind"], "forced")
            self.assertEqual(record["action"]["kind"], action_type)
            self.assertIsNone(record["scorer"])
        self.assertEqual(by_ordinal[9]["kind"], "pass_response")
        self.assertEqual(by_ordinal[9]["action"]["kind"], "pass")
        single = by_ordinal[5]
        self.assertTrue(single["single_candidate"])
        self.assertEqual(single["kind"], "discard")
        self.assertIsNone(single["scorer"])
        for ordinal, head, encoding in ((0, "discard", "m1"), (4, "call", "r1")):
            scorer = by_ordinal[ordinal]["scorer"]
            self.assertEqual(scorer["head"], head)
            self.assertEqual(scorer["encoding"]["name"], encoding)
            self.assertGreaterEqual(len(scorer["scores"]), 2)
            self.assertIn(by_ordinal[ordinal]["action"], scorer["candidates"])
            self.assertEqual(
                scorer["shared_feature"]["dimension"], features.FEATURE_DIMENSION
            )
        self.assertEqual(report["artifacts"]["m1"]["identity"], self.identities["m1"])
        self.assertEqual(
            set(report["artifacts"]["r1"]["files"]), {"manifest.json", "weights.f32"}
        )
        self.assertEqual(report["runtime"]["torch"]["threads"], 1)
        self.assertTrue(report["runtime"]["torch"]["deterministic_algorithms"])

    def test_either_loader_shape_runs_the_given_runtime(self) -> None:
        digests = {}
        for loader in ("_load_scorer", "load_l2a_scorer"):
            with self.subTest(loader=loader):
                factory_calls = []
                score_calls = []

                def load_factory(
                    m1, r1, factory_calls=factory_calls, calls=score_calls
                ):
                    factory_calls.append((Path(m1), Path(r1)))
                    real = l2a_policy.load_l2a_policy_factory(m1, r1)
                    return l2a_policy.L2aRuntime(
                        discard=_CountingScorer(real.discard, calls),
                        reaction=_CountingScorer(real.reaction, calls),
                    )

                policy = types.SimpleNamespace(
                    load_l2a_policy_factory=load_factory,
                    **{loader: l2a_policy.load_l2a_scorer},
                )
                report = self._capture(modules=_modules(policy=policy))
                self.assertEqual(report["runtime"]["scorer_loader"], loader)
                self.assertEqual(
                    factory_calls, [(self.artifacts["m1"], self.artifacts["r1"])]
                )
                scorer_rows = sum(r["scorer"] is not None for r in report["records"])
                # decideと決定性確認の再scoreで、scorer行ごとに2回。
                self.assertEqual(len(score_calls), 2 * scorer_rows)
                digests[loader] = report["decision_digest"]
        self.assertEqual(len(set(digests.values())), 1)

        for names in ((), ("_load_scorer", "load_l2a_scorer")):
            with self.subTest(names=names):
                policy = types.SimpleNamespace(
                    load_l2a_policy_factory=l2a_policy.load_l2a_policy_factory,
                    **{name: l2a_policy.load_l2a_scorer for name in names},
                )
                with self.assertRaisesRegex(tool.CompatError, "exactly one"):
                    tool.Runtime(_modules(policy=policy))

    def test_offline_eval_payload_is_never_opened(self) -> None:
        offline = self.source_path / "game-002" / GAME_PAYLOAD_FILENAME
        original = offline.read_bytes()
        try:
            offline.write_bytes(b"not opened\n")
            self.assertEqual(self._capture()["summary"]["records"], 20)
        finally:
            offline.write_bytes(original)
        with self.assertRaisesRegex(tool.CompatError, "OFFLINE-EVAL"):
            tool.build_input_manifest(
                self.source, game_ordinals=(2,), payload_filename=GAME_PAYLOAD_FILENAME
            )

    def _manifest_variant(self, name, mutate):
        body = {key: value for key, value in self.manifest.items() if key != "identity"}
        body = copy.deepcopy(body)
        mutate(body)
        return self._write(name, tool.seal(body))

    def test_inputs_that_do_not_match_the_manifest_stop(self) -> None:
        cases = {
            "payload": lambda body: body["games"][0]["payload"].update(sha256="0" * 64),
            "seed": lambda body: body["games"][1].update(seed=999),
            "source identity": lambda body: body["source"].update(identity="0" * 64),
            "artifact identities": lambda body: body["artifacts"].update(m1="0" * 64),
            "classification": lambda body: body.update(classification="other"),
        }
        for index, (message, mutate) in enumerate(cases.items()):
            with self.subTest(message=message):
                path = self._manifest_variant(f"m{index}.json", mutate)
                with self.assertRaises(tool.CompatError):
                    self._capture(manifest=path)

        other = _write_source(self.root / "other", seed_base=1000)
        with self.assertRaisesRegex(tool.CompatError, "source identity"):
            self._capture(source=other)
        # 取り違えたartifactは検証対象runtime自身がfail closedする。
        with self.assertRaisesRegex(LearnedPolicyError, "DISCARD head"):
            self._capture(m1=self.artifacts["r1"], r1=self.artifacts["m1"])

    def test_shape_and_nonfinite_features_stop(self) -> None:
        def short(policy_input):
            return features.build_player_safe_feature(policy_input)[:-1]

        def nonfinite(policy_input):
            return (math.nan, *features.build_player_safe_feature(policy_input)[1:])

        for name, build, message in (
            ("shape", short, "dimension"),
            ("nan", nonfinite, "non-finite"),
        ):
            with self.subTest(name=name):
                module = types.SimpleNamespace(
                    FEATURE_DIMENSION=features.FEATURE_DIMENSION,
                    build_player_safe_feature=build,
                )
                with self.assertRaisesRegex(tool.CompatError, message):
                    self._capture(modules=_modules(features=module))

    def test_limits_stop_the_capture(self) -> None:
        ticks = iter(range(0, 10_000, 100))
        limits = self._limits(wall=150.0, clock=lambda: float(next(ticks)))
        with self.assertRaisesRegex(tool.CompatError, "wall time"):
            self._capture(limits=limits)
        with self.assertRaisesRegex(tool.CompatError, "RSS"):
            self._capture(limits=self._limits(rss=1e-3))

    def test_cli_capture_writes_once_and_records_stop(self) -> None:
        output = self.root / "capture.json"
        arguments = [
            "capture",
            "--manifest",
            str(self.manifest_path),
            "--source",
            str(self.source_path),
            "--m1",
            str(self.artifacts["m1"]),
            "--r1",
            str(self.artifacts["r1"]),
            "--native-wheel",
            str(self.root / "native.whl"),
            "--torch-threads",
            "1",
            "--max-wall-seconds",
            "600",
            "--max-rss-mib",
            "1000000",
            "--output",
            str(output),
        ]
        with (
            mock.patch.object(tool, "runtime_provenance", _provenance(OLD)),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(tool.main(arguments), 0)
        report = tool.read_capture(output, self.manifest, context="capture")
        self.assertEqual(report["tool"]["sha256"], tool._sha256_file(TOOL_PATH))
        self.assertEqual(report["metadata"]["limits"]["max_rss_mib"], 1000000.0)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(arguments)
        inside = list(arguments)
        inside[-1] = str(self.artifacts["m1"] / "capture.json")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(inside)
        self.assertFalse((self.artifacts["m1"] / "capture.json").exists())

        # 入力が不正ならSTOP reportを残し、recordは書かない。
        stopped = list(arguments)
        stopped[stopped.index("--manifest") + 1] = str(
            self._manifest_variant(
                "offline.json",
                lambda body: body["games"][0].update(split="OFFLINE-EVAL"),
            )
        )
        stopped[-1] = str(self.root / "stop.json")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(tool.main(stopped), 2)
        stop = json.loads(Path(stopped[-1]).read_text(encoding="utf-8"))
        self.assertEqual(stop["status"], "STOP")
        self.assertIn("OFFLINE-EVAL", stop["reason"])
        self.assertNotIn("records", stop)

    def test_cli_make_manifest_matches_the_library(self) -> None:
        output = self.root / "made.json"
        arguments = [
            "make-manifest",
            "--source",
            str(self.source_path),
            "--game",
            "0",
            "--game",
            "1",
            "--m1-identity",
            self.identities["m1"],
            "--r1-identity",
            self.identities["r1"],
            "--output",
            str(output),
        ]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(tool.main(arguments), 0)
        self.assertEqual(tool.read_input_manifest(output), self.manifest)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(arguments)


if __name__ == "__main__":
    unittest.main()
