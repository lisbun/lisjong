"""R1価値モデル（Issue #360）: source reader・推定宣言牌・19列・ridge・artifact・gate Policy・E2 / E3。

source blockはArena riichi-first-dama-source-v1の形をこのtestで合成する（lisjong_arenaはimportしない）。
解析は実lisjong（native scoring backendが必要）。
"""

import hashlib
import json
import random
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest import mock

from test_riichi_dama_branch import (
    KANCHAN,
    PINFU,
    SANSHOKU,
    _discard,
    _needs_scoring,
    _riichi_decision,
)

from lisjong.learning import r1_value as core
from lisjong.learning import r1_value_evaluation as evaluation
from lisjong.learning import r1_value_policy as policy_module
from lisjong.learning._canonical import canonical_json_line, canonical_json_text
from lisjong.learning._typed_values import action_to_value, policy_input_to_value_v2
from lisjong.policies import PlacementAwareSpeedCallPolicy
from lisjong.policy_contract import DecisionContext
from lisjong.policy_contract.action import RiichiAction
from lisjong.policy_contract.seat import Seat

HANDS = (PINFU, SANSHOKU, KANCHAN)
DECLARATION = _discard("1z")
PLAN = "a" * 64
ALLOCATION = "b" * 64


def _decision(index: int) -> DecisionContext:
    """手牌・巡目・本場・供託・残り山を行ごとに変える分岐点。"""
    river = "".join(("9m", "2z", "3z", "4z", "5z", "6z", "7z")[: 1 + index % 7])
    decision = _riichi_decision(HANDS[index % 3], river=river)
    state = replace(
        decision.input.round,
        honba=index % 4,
        riichi_sticks=index % 3,
        live_wall_tiles_remaining=20 + (index * 7) % 40,
    )
    return replace(decision, input=replace(decision.input, round=state))


def _delta(index: int) -> int:
    """本場と残り山に依存する決定的なΔ（100点単位）。"""
    return 4 * (index % 4) - (20 + (index * 7) % 40) // 10 + (index % 5) - 2


def _line(ordinal: int, seed: int, index: int) -> dict:
    decision = _decision(index)
    g_r = 10 * (index % 3) - 10
    return {
        "branch_id": f"{seed}-{ordinal}-0",
        "branch_ordinal": ordinal,
        "context": {
            "legal_actions": [
                action_to_value(action, ValueError, "a")
                for action in decision.legal_actions
            ],
            "policy_input": policy_input_to_value_v2(decision.input, ValueError, "p"),
        },
        "context_sha256": "c" * 64,
        "d": {
            "branch_ordinal": ordinal,
            "invalid": [],
            "seat": 0,
            "seed": seed,
            "settlement": {"point_deltas": [100 * (g_r + _delta(index)), 0, 0, 0]},
            "status": "branched",
        },
        "d_repeat": None,
        "decision_index": ordinal,
        "declaration": action_to_value(DECLARATION, ValueError, "d"),
        "declaration_decision_index": ordinal + 1,
        "kind": "branch",
        "prefix_sha256": "d" * 64,
        "r_invalid": [],
        "r_post_sha256": "e" * 64,
        "r_settlement": {"point_deltas": [100 * g_r, 0, 0, 0]},
        "riichi": action_to_value(RiichiAction(actor=Seat.SEAT_0), ValueError, "r"),
        "round": {},
        "round_ordinal": 0,
        "seat": 0,
        "seed": seed,
    }


def write_block(
    root: Path, block: str, *, artifact=None, rows=10, start=0, **changes
) -> Path:
    split, seeds = core.BLOCKS[block]
    directory = root / block
    directory.mkdir(parents=True)
    header = {
        "allocation_identity": ALLOCATION,
        "artifact_identity": artifact,
        "block": block,
        "features_version": core.RIICHI_DAMA_FEATURES_VERSION,
        "kind": "header",
        "plan_digest": PLAN,
        "protocol_version": core.SOURCE_PROTOCOL,
        "schema": core.BRANCHES_SCHEMA,
        "split": split,
        "wrapper_version": core.RIICHI_FIRST_DAMA_BRANCH_VERSION,
    }
    header.update(changes.pop("header", {}))
    lines = [header] + [
        _line(ordinal, seeds[ordinal % len(seeds)], start + ordinal)
        for ordinal in range(rows)
    ]
    (directory / "branches.jsonl").write_text(
        "".join(canonical_json_line(line) for line in lines),
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "block": block,
        "decision": core.SOURCE_COMPLETE,
        "invalid": {"branches": [], "count": 0},
        "plan_digest": PLAN,
        "schema": core.SUMMARY_SCHEMA,
        "split": split,
    }
    summary.update(changes.pop("summary", {}))
    (directory / "summary.json").write_text(
        canonical_json_text(summary), encoding="utf-8"
    )
    resum(directory)
    return directory


def resum(directory: Path) -> None:
    text = "".join(
        f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}\n"
        for item in sorted(directory.iterdir())
        if item.name != "SHA256SUMS"
    )
    (directory / "SHA256SUMS").write_text(text, encoding="utf-8", newline="\n")


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)

    def tearDown(self) -> None:
        self._dir.cleanup()


# ---------------------------------------------------------------------------
# source reader
# ---------------------------------------------------------------------------


@_needs_scoring
class SourceReaderTest(_Tmp):
    def test_reads_rows_and_deltas(self) -> None:
        block = core.read_source_block(write_block(self.tmp, "TRAIN-1", rows=4))
        self.assertEqual(block.split, core.TRAIN)
        self.assertEqual(len(block.rows), 4)
        row = block.rows[1]
        self.assertEqual(row.delta, _delta(1))
        self.assertEqual(row.declaration, DECLARATION)
        self.assertEqual(row.decision, _decision(1))

    def test_rejects_tampered_or_incomplete_blocks(self) -> None:
        cases = {
            "schema": {"header": {"schema": "other"}},
            "split": {"header": {"split": "TEST"}},
            "artifact on TRAIN": {"header": {"artifact_identity": "f" * 64}},
            "decision": {"summary": {"decision": "STOP-INVALID"}},
            "invalid": {"summary": {"invalid": {"count": 1}}},
            "plan": {"summary": {"plan_digest": "0" * 64}},
        }
        for name, changes in cases.items():
            with self.subTest(name=name), self.assertRaises(core.R1ValueError):
                core.read_source_block(
                    write_block(self.tmp / name, "TRAIN-1", **changes)
                )
        directory = write_block(self.tmp / "hash", "TRAIN-1")
        (directory / "summary.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(core.R1ValueError, "SHA256SUMS"):
            core.read_source_block(directory)
        directory = write_block(self.tmp / "extra", "TRAIN-1")
        (directory / "extra.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(core.R1ValueError, "missing or extra"):
            core.read_source_block(directory)
        with self.assertRaisesRegex(core.R1ValueError, "artifact"):
            core.read_source_block(write_block(self.tmp / "test", "TEST-1"))

    def test_rejects_rows_outside_the_block_or_invalid_arms(self) -> None:
        directory = write_block(self.tmp / "seed", "TRAIN-1", rows=2)
        lines = [
            json.loads(x)
            for x in (directory / "branches.jsonl").read_text().splitlines()
        ]
        lines[1]["seed"] = lines[1]["d"]["seed"] = 547_100
        (directory / "branches.jsonl").write_text(
            "".join(canonical_json_line(x) for x in lines),
            encoding="utf-8",
            newline="\n",
        )
        resum(directory)
        with self.assertRaisesRegex(core.R1ValueError, "outside the block"):
            core.read_source_block(directory)
        directory = write_block(self.tmp / "arm", "TRAIN-1", rows=2)
        lines = [
            json.loads(x)
            for x in (directory / "branches.jsonl").read_text().splitlines()
        ]
        lines[2]["d"]["status"] = "pending"
        (directory / "branches.jsonl").write_text(
            "".join(canonical_json_line(x) for x in lines),
            encoding="utf-8",
            newline="\n",
        )
        resum(directory)
        with self.assertRaisesRegex(core.R1ValueError, "D arm"):
            core.read_source_block(directory)

    def test_read_source_needs_both_blocks_of_the_split(self) -> None:
        one = write_block(self.tmp, "TRAIN-1", rows=2)
        with self.assertRaisesRegex(core.R1ValueError, "exactly the blocks"):
            core.read_source([one], core.TRAIN)
        two = write_block(self.tmp, "TRAIN-2", rows=2)
        blocks = core.read_source([two, one], core.TRAIN)
        self.assertEqual([block.block for block in blocks], ["TRAIN-1", "TRAIN-2"])
        a = write_block(self.tmp, "TEST-1", rows=2, artifact="1" * 64)
        b = write_block(self.tmp, "TEST-2", rows=2, artifact="2" * 64)
        with self.assertRaisesRegex(core.R1ValueError, "different artifacts"):
            core.read_source([a, b], core.TEST)


# ---------------------------------------------------------------------------
# 推定宣言牌・encoding
# ---------------------------------------------------------------------------


@_needs_scoring
class EncodingTest(unittest.TestCase):
    def test_projected_declaration_keeps_tenpai(self) -> None:
        self.assertEqual(core.project_declaration(_riichi_decision(PINFU)), DECLARATION)
        no_riichi = DecisionContext(
            input=_decision(0).input,
            legal_actions=tuple(
                a for a in _decision(0).legal_actions if not isinstance(a, RiichiAction)
            ),
        )
        with self.assertRaises(core.R1ValueError):
            core.project_declaration(no_riichi)

    def test_projection_uses_a_declared_own_seat_and_tenpai_discards(self) -> None:
        captured = []

        def capture(policy, decision):
            captured.append(decision)
            return decision.legal_actions[0]

        with mock.patch.object(core, "execute_policy", capture):
            core.project_declaration(_riichi_decision(PINFU))
        (projected,) = captured
        self.assertEqual(projected.legal_actions, (DECLARATION,))
        self.assertEqual(projected.input.players[0].riichi.name, "DECLARED")

    def test_column_order_and_values(self) -> None:
        decision = _decision(5)
        vector = core.encode(decision, core.analyze(decision, DECLARATION))
        self.assertEqual(len(vector), 19)
        named = dict(zip(core.COLUMNS, vector, strict=True))
        self.assertEqual(named["honba"], 1.0)
        self.assertEqual(named["riichi_sticks"], 2.0)
        self.assertEqual(named["live_wall_tiles_remaining"], float(20 + 35 % 40))
        self.assertEqual(named["own_discard_count"], 6.0)
        self.assertEqual(named["rank_2"] + named["rank_3"] + named["rank_4"], 0.0)
        self.assertEqual(named["opponent_riichi_1"], 0.0)
        self.assertEqual(core.COLUMNS[12], "unseen_total_over_waits")
        self.assertEqual(core.CONTINUOUS, tuple(range(12, 19)))

    def test_inputs_are_only_the_public_context_and_the_declaration(self) -> None:
        import inspect

        self.assertEqual(
            list(inspect.signature(core.encode).parameters), ["decision", "features"]
        )
        self.assertNotIn("import lisjong_arena", inspect.getsource(core))


# ---------------------------------------------------------------------------
# ridge・artifact
# ---------------------------------------------------------------------------


def _reference_ridge(matrix, targets):
    """Fractionでの正規方程式（切片罰則なし、和の目的関数）。"""
    n = len(matrix)
    scaling = []
    for column in core.CONTINUOUS:
        values = [Fraction(row[column]) for row in matrix]
        mean = sum(values) / n
        scaling.append((mean, sum((v - mean) ** 2 for v in values) / n))
    import math

    design = []
    for row in matrix:
        z = [Fraction(v) for v in row]
        for (mean, var), column in zip(scaling, core.CONTINUOUS, strict=True):
            z[column] = (z[column] - mean) / Fraction(math.sqrt(var))
        design.append([Fraction(1), *z])
    k = len(design[0])
    gram = [[sum(r[i] * r[j] for r in design) for j in range(k)] for i in range(k)]
    for i in range(1, k):
        gram[i][i] += 10
    moment = [
        sum(r[i] * t for r, t in zip(design, targets, strict=True)) for i in range(k)
    ]
    # Gauss-Jordan
    rows = [g + [m] for g, m in zip(gram, moment, strict=True)]
    for c in range(k):
        p = next(r for r in range(c, k) if rows[r][c] != 0)
        rows[c], rows[p] = rows[p], rows[c]
        for r in range(k):
            if r != c and rows[r][c]:
                f = rows[r][c] / rows[c][c]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[c], strict=True)]
    return [rows[i][k] / rows[i][i] for i in range(k)]


class RidgeTest(unittest.TestCase):
    def _data(self, n=40):
        rng = random.Random(7)
        matrix = []
        for _ in range(n):
            row = [float(rng.random() < 0.5) for _ in range(12)]
            row += [float(rng.randint(0, 12)) for _ in range(7)]
            matrix.append(row)
        targets = [int(3 * r[0] - r[13] + rng.randint(-5, 5)) for r in matrix]
        return matrix, targets

    def test_matches_the_reference_normal_equations(self) -> None:
        matrix, targets = self._data()
        scaling, intercept, coefficients = core.fit_ridge(matrix, targets)
        reference = _reference_ridge(matrix, targets)
        self.assertAlmostEqual(intercept, float(reference[0]), places=9)
        for mine, ref in zip(coefficients, reference[1:], strict=True):
            self.assertAlmostEqual(mine, float(ref), places=9)
        # 母分散（nで割る）
        values = [row[12] for row in matrix]
        mean = sum(values) / len(values)
        self.assertAlmostEqual(
            scaling[0][1], sum((v - mean) ** 2 for v in values) / len(values)
        )

    def test_intercept_is_not_penalised(self) -> None:
        matrix, _ = self._data()
        targets = [100] * len(matrix)
        _, intercept, coefficients = core.fit_ridge(matrix, targets)
        self.assertAlmostEqual(intercept, 100.0, places=6)
        self.assertTrue(all(abs(c) < 1e-6 for c in coefficients))

    def test_zero_variance_continuous_column_is_rejected(self) -> None:
        matrix, targets = self._data()
        for row in matrix:
            row[16] = 30.0
        with self.assertRaisesRegex(core.R1ValueError, "zero variance"):
            core.fit_ridge(matrix, targets)


@_needs_scoring
class ArtifactTest(_Tmp):
    def _train(self):
        return core.read_source(
            [
                write_block(self.tmp, "TRAIN-1", rows=24),
                write_block(self.tmp, "TRAIN-2", rows=24, start=24),
            ],
            core.TRAIN,
        )

    def test_fit_save_load_round_trip(self) -> None:
        blocks = self._train()
        document = core.build_artifact(blocks)
        self.assertEqual(document["diagnostics"]["rows"], 48)
        self.assertEqual(document["diagnostics"]["projected_declaration_matches"], 48)
        path = core.save_artifact(document, self.tmp / "artifact")
        artifact = core.load_artifact(path)
        self.assertEqual(artifact.identity, document["identity"])
        rows = core.rows_of(blocks)
        self.assertAlmostEqual(
            artifact.train_mean, sum(row.delta for row in rows) / len(rows)
        )
        with self.assertRaisesRegex(core.R1ValueError, "overwrite"):
            core.save_artifact(document, path)

    def test_tampered_or_foreign_artifacts_are_rejected(self) -> None:
        document = core.build_artifact(self._train())
        cases = {
            "identity": {**document, "identity": "0" * 64},
            "coefficients": {
                **document,
                "fit": {**document["fit"], "coefficients": [0.0] * 18},
            },
        }
        for name, changed in cases.items():
            with self.subTest(name=name), self.assertRaises(core.R1ValueError):
                core.parse_artifact(changed)
        path = core.save_artifact(document, self.tmp / "artifact")
        (path / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(core.R1ValueError, "canonical"):
            core.load_artifact(path)

    def test_fit_uses_train_rows_only(self) -> None:
        test = core.read_source(
            [
                write_block(self.tmp, "TEST-1", rows=4, artifact="1" * 64),
                write_block(self.tmp, "TEST-2", rows=4, artifact="1" * 64),
            ],
            core.TEST,
        )
        with self.assertRaisesRegex(core.R1ValueError, "TRAIN rows only"):
            core.build_artifact(test)


# ---------------------------------------------------------------------------
# gate Policy
# ---------------------------------------------------------------------------


def _artifact(intercept: float) -> core.R1ValueArtifact:
    return core.R1ValueArtifact(
        identity="9" * 64,
        train_mean=0.0,
        scaling=tuple((0.0, 1.0) for _ in core.CONTINUOUS),
        intercept=intercept,
        coefficients=(0.0,) * len(core.COLUMNS),
        source={},
    )


@_needs_scoring
class GatePolicyTest(unittest.TestCase):
    def test_positive_prediction_discards_the_declaration_without_riichi(self) -> None:
        decision = _riichi_decision(PINFU)
        policy = policy_module.R1ValueGatePolicy(_artifact(1.0))
        self.assertEqual(policy.choose_action(decision), DECLARATION)

    def test_non_positive_prediction_keeps_the_champion_riichi(self) -> None:
        decision = _riichi_decision(PINFU)
        for value in (0.0, -1.0):
            with self.subTest(value=value):
                policy = policy_module.R1ValueGatePolicy(_artifact(value))
                self.assertIsInstance(policy.choose_action(decision), RiichiAction)

    def test_mismatch_or_failure_falls_back_to_the_champion(self) -> None:
        decision = _riichi_decision(PINFU)
        policy = policy_module.R1ValueGatePolicy(_artifact(1.0))
        with mock.patch.object(
            policy_module,
            "project_declaration",
            return_value=_discard("1z", tsumogiri=True),
        ):
            self.assertIsInstance(policy.choose_action(decision), RiichiAction)
        with mock.patch.object(
            policy_module, "analyze", side_effect=core.R1ValueError("x")
        ):
            self.assertIsInstance(policy.choose_action(decision), RiichiAction)

    def test_non_riichi_decisions_are_the_champion(self) -> None:
        decision = _decision(3)
        dama = DecisionContext(
            input=decision.input,
            legal_actions=tuple(
                a for a in decision.legal_actions if not isinstance(a, RiichiAction)
            ),
        )
        policy = policy_module.R1ValueGatePolicy(_artifact(1.0))
        self.assertEqual(
            policy.choose_action(dama),
            PlacementAwareSpeedCallPolicy().choose_action(dama),
        )

    def test_constructor_requires_an_artifact(self) -> None:
        with self.assertRaises(TypeError):
            policy_module.R1ValueGatePolicy(object())


# ---------------------------------------------------------------------------
# E2 / E3
# ---------------------------------------------------------------------------


def _reference_bootstrap(seeds, by_seed):
    rng = random.Random(356)
    stats = []
    for _ in range(10_000):
        total = count = 0
        for _ in range(len(seeds)):
            value, size = by_seed[seeds[rng.randrange(len(seeds))]]
            total += value
            count += size
        stats.append(total / count)
    stats.sort()
    return stats[499]


class BootstrapTest(unittest.TestCase):
    def test_matches_the_reference_and_uses_the_500th_value(self) -> None:
        seeds = list(range(547_100, 547_144))
        by_seed = {seed: (float((seed % 7) - 3), 1 + seed % 3) for seed in seeds}
        result = evaluation.cluster_bootstrap(seeds, by_seed)
        self.assertAlmostEqual(
            result["P5"], _reference_bootstrap(seeds, by_seed), places=12
        )
        self.assertEqual(evaluation.P5_RANK, 500)
        self.assertEqual(evaluation.cluster_bootstrap(seeds, by_seed), result)


@_needs_scoring
class EvaluationTest(_Tmp):
    def _test_blocks(self, artifact_identity, rows=12):
        return [
            write_block(
                self.tmp, "TEST-1", rows=rows, artifact=artifact_identity, start=100
            ),
            write_block(
                self.tmp,
                "TEST-2",
                rows=rows,
                artifact=artifact_identity,
                start=100 + rows,
            ),
        ]

    def test_estimators_and_decision(self) -> None:
        artifact = _artifact(1.0)
        blocks = core.read_source(self._test_blocks(artifact.identity), core.TEST)
        document = evaluation.evaluate(artifact, blocks)
        rows = core.rows_of(blocks)
        e2 = sum((r.delta - 0.0) ** 2 - (r.delta - 1.0) ** 2 for r in rows) / len(rows)
        e3 = sum(r.delta for r in rows) / len(rows)  # 全行で一致・予測 > 0
        self.assertAlmostEqual(document["e2"]["estimate"], e2)
        self.assertAlmostEqual(document["e3"]["estimate"], e3)
        self.assertEqual(document["rates"]["actual_intervention"], 1.0)
        self.assertEqual(document["rates"]["any_mismatch"], 0.0)
        self.assertIn(
            document["decision"],
            (
                evaluation.NOT_PREDICTIVE,
                evaluation.PREDICTIVE_ONLY,
                evaluation.DECISION_VALUE_SIGNAL,
            ),
        )
        self.assertEqual(document["e2"]["pass"], document["e2"]["bootstrap"]["P5"] > 0)
        if not document["e2"]["pass"]:
            self.assertEqual(document["decision"], evaluation.NOT_PREDICTIVE)
            self.assertFalse(document["e3"]["pass"])

    def test_support_invalid_and_e3_needs_e2(self) -> None:
        artifact = _artifact(1.0)
        blocks = core.read_source(self._test_blocks(artifact.identity), core.TEST)
        with mock.patch.object(
            evaluation,
            "project_declaration",
            return_value=_discard("1z", tsumogiri=True),
        ):
            document = evaluation.evaluate(artifact, blocks)
        self.assertTrue(document["e3"]["support_invalid"])
        self.assertFalse(document["e3"]["pass"])
        self.assertEqual(document["e3"]["estimate"], 0.0)
        self.assertEqual(document["rates"]["projected_declaration_mismatch"], 1.0)

    def test_test_is_read_once_and_bound_to_the_artifact(self) -> None:
        train = core.read_source(
            [
                write_block(self.tmp / "t", "TRAIN-1", rows=24),
                write_block(self.tmp / "t", "TRAIN-2", rows=24, start=24),
            ],
            core.TRAIN,
        )
        path = core.save_artifact(core.build_artifact(train), self.tmp / "artifact")
        identity = core.load_artifact(path).identity
        test_dirs = self._test_blocks(identity)
        record = self.tmp / "records"
        document = evaluation.run_evaluation(path, test_dirs, record)
        self.assertTrue(evaluation.evaluation_path(record, identity).exists())
        self.assertEqual(document["artifact_identity"], identity)
        with self.assertRaisesRegex(core.R1ValueError, "already been evaluated"):
            evaluation.run_evaluation(path, test_dirs, record)
        other = self.tmp / "other"
        wrong = [
            write_block(other, "TEST-1", rows=2, artifact="1" * 64),
            write_block(other, "TEST-2", rows=2, artifact="1" * 64),
        ]
        with self.assertRaisesRegex(
            core.R1ValueError, "not generated for this artifact"
        ):
            evaluation.run_evaluation(path, wrong, self.tmp / "records-2")


class BoundaryTest(unittest.TestCase):
    def test_no_ml_framework_or_arena_import(self) -> None:
        import subprocess
        import sys

        for module in (core, policy_module, evaluation):
            text = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotIn("import torch", text)
            self.assertNotIn("lisjong_arena", text.split('"""', 2)[-1])
        # 他のtestがtorchを読み込んでいても影響しないよう、新しいprocessで確かめる
        probe = (
            "import sys, lisjong.learning.r1_value, lisjong.learning.r1_value_policy, "
            "lisjong.learning.r1_value_evaluation; "
            "print(sorted(m for m in ('torch', 'lisjong_arena') if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
