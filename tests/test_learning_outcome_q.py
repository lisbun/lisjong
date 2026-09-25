"""lisjong-project#79 step D — outcome-Q preflight / training set / trainer /
artifact / Q residual runtime / cheap serving qualificationのtest。

実Arena sourceは読まない。#193 / #195の合成focal outcome source fixtureだけを使う。
"""

import contextlib
import dataclasses
import importlib.util
import io
import json
import math
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

import learning_fixtures as fixtures
import outcome_fixtures as of

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.learning import __main__ as learning_main
from lisjong.learning._canonical import (
    canonical_json_text,
    file_digest,
    seal,
    value_digest,
)
from lisjong.learning.artifact import _weights_bytes
from lisjong.learning.candidate_artifact import load_candidate_artifact
from lisjong.learning.candidate_encoding import (
    CANDIDATE_BLOCK_OFFSETS,
    CANDIDATE_ENCODING_DIMENSION,
    encode_candidates,
)
from lisjong.learning.candidate_model import CandidateScorerConfig
from lisjong.learning.envelope_policy import (
    SEMANTIC_ENVELOPE_IDENTITY,
    SemanticEnvelopeOffensePolicy,
    semantic_envelope_runtime_identity,
)
from lisjong.learning.errors import (
    ModelArtifactError,
    OutcomeSourceError,
    TrainingError,
)
from lisjong.learning.features import build_player_safe_feature
from lisjong.learning.outcome_q_artifact import (
    MANIFEST_FILENAME,
    OUTCOME_Q_ARTIFACT_SCHEMA,
    WEIGHTS_FILENAME,
    load_outcome_q_artifact,
    write_outcome_q_artifact,
)
from lisjong.learning.outcome_q_dataset import (
    PREFLIGHT_PASS,
    PREFLIGHT_STOP,
    build_outcome_q_training_set,
    outcome_q_label_block,
    outcome_q_preflight,
)
from lisjong.learning.outcome_q_training import (
    FROZEN_OUTCOME_Q_TRAINING_CONFIG,
    OutcomeQTrainingConfig,
    train_outcome_q,
)
from lisjong.learning.outcome_source import (
    OUTCOME_OBJECTIVE_IDENTITY,
    OUTCOME_TARGET_IDENTITY,
    build_outcome_targets,
    read_outcome_source,
    validate_outcome_source_provenance,
)
from lisjong.learning.residual_baseline import (
    CONSTANT_RESIDUAL_RUNTIME_IDENTITY,
    ConstantResidualRuntime,
)

requires_ml_runtime = unittest.skipIf(
    importlib.util.find_spec("torch") is None,
    "requires the optional ML runtime (lisjong[ml])",
)

MODEL = CandidateScorerConfig(hidden_width=64)


def _training_block(**overrides):
    value = fixtures.training_block(objective=OUTCOME_OBJECTIVE_IDENTITY)
    value.update(overrides)
    return value


def _tile_type_weights(tile_type, weight=1.0):
    """score = ReLU(weight * [candidate discard tile type == tile_type])。"""
    values = [0.0] * MODEL.parameter_count
    layout = {parameter.name: parameter for parameter in MODEL.parameter_layout()}
    feature = CANDIDATE_BLOCK_OFFSETS["discard_tile_type"] + tile_type_index(tile_type)
    values[layout["candidate_layer.weight"].offset + feature] = weight
    values[layout["output_layer.weight"].offset] = 1.0
    return tuple(values)


def _read_body(root):
    manifest = json.loads((Path(root) / MANIFEST_FILENAME).read_text("utf-8"))
    manifest.pop("identity")
    return manifest


def _reseal(root, body):
    (Path(root) / MANIFEST_FILENAME).write_text(
        canonical_json_text(seal(body)), encoding="utf-8", newline="\n"
    )


class _SourceCase(unittest.TestCase):
    schema = of.ENGINE_OUTCOME_SOURCE_SCHEMA

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_path = of.write_outcome_source(
            self.root / "source", schema=self.schema
        )
        self.source = read_outcome_source(self.source_path)

    def replace_games(self, games):
        return dataclasses.replace(self.source, games=tuple(games))

    def write_artifact(self, name="artifact", *, weights=None, **training):
        return write_outcome_q_artifact(
            self.root / name,
            training_set=build_outcome_q_training_set(self.source),
            model_config=MODEL,
            training=_training_block(**training),
            weights=weights or fixtures.zero_weights(MODEL),
        )


# ---------------------------------------------------------------------------
# preflight / training set（ML runtime不要）
# ---------------------------------------------------------------------------


class OutcomeQPreflightTests(_SourceCase):
    def test_scientific_fixture_passes_and_reports_per_split_facts(self) -> None:
        report = outcome_q_preflight(self.source)

        self.assertEqual(report["outcome"], PREFLIGHT_PASS)
        self.assertEqual(report["failures"], [])
        self.assertEqual(set(report["splits"]), {"TRAIN", "SELECT"})
        train = report["splits"]["TRAIN"]
        self.assertEqual(train["hanchan_count"], 2)
        self.assertEqual(train["eligible_row_count"], 4)
        self.assertEqual(train["target_identity"], OUTCOME_TARGET_IDENTITY)
        self.assertEqual(
            sum(train["selected_survivor_position_distribution"].values()), 4
        )
        self.assertEqual(report["source"], self.source.provenance())
        self.assertEqual(report["label"], outcome_q_label_block())

    def test_non_scientific_sources_stop(self) -> None:
        for role, games in (
            ("CALIBRATION", of.CALIBRATION_GAMES),
            ("DIAGNOSTIC", of.DIAGNOSTIC_GAMES),
        ):
            with self.subTest(role=role):
                source = read_outcome_source(
                    of.write_outcome_source(
                        self.root / role, games, role=role, schema=self.schema
                    )
                )
                report = outcome_q_preflight(source)
                self.assertEqual(report["outcome"], PREFLIGHT_STOP)
                checks = {failure["check"] for failure in report["failures"]}
                self.assertIn("population_role", checks)
                self.assertIn("splits", checks)

    def test_missing_select_split_stops(self) -> None:
        source = self.replace_games(
            game for game in self.source.games if game.split == "TRAIN"
        )
        report = outcome_q_preflight(source)
        self.assertEqual(report["outcome"], PREFLIGHT_STOP)
        self.assertIn(
            {"check": "eligible_rows", "split": "SELECT", "observed": 0},
            report["failures"],
        )

    def test_split_seed_leakage_stops(self) -> None:
        train_seed = self.source.games[0].seed
        source = self.replace_games(
            dataclasses.replace(game, seed=train_seed)
            if game.split == "SELECT"
            else game
            for game in self.source.games
        )
        report = outcome_q_preflight(source)
        self.assertIn(
            {"check": "split_leakage", "observed": "seed overlap"}, report["failures"]
        )

    def test_train_support_rule_is_enforced(self) -> None:
        # fixture TRAINのcanonical-first率は0.5。下限を超えるruleで配線を固定する。
        with patch("lisjong.learning.outcome_q_dataset.MINIMUM_SUPPORT_RATE", 0.6):
            report = outcome_q_preflight(self.source)
        checks = {failure["check"] for failure in report["failures"]}
        self.assertEqual(
            checks,
            {
                "train_canonical_first_selected_rate",
                "train_non_canonical_first_selected_rate",
            },
        )


class OutcomeQTrainingSetTests(_SourceCase):
    def test_rows_are_eligible_rows_with_selected_candidate_targets(self) -> None:
        training_set = build_outcome_q_training_set(self.source)
        targets = build_outcome_targets(self.source)

        self.assertEqual(len(training_set.rows), len(targets.rows))
        width = CANDIDATE_ENCODING_DIMENSION
        for row, target in zip(training_set.rows, targets.rows, strict=True):
            self.assertGreaterEqual(len(row.survivors), 2)
            self.assertIn(row.selected_candidate_index, row.survivors)
            self.assertEqual(
                row.selected_candidate_index, target.selected_candidate_index
            )
            self.assertEqual(row.target_q, target.target_q)
            self.assertEqual(row.candidate_count, len(target.candidates))
            encoded = encode_candidates(target.candidates)
            start = row.candidate_offset * width
            payload = training_set.candidates[
                start : start + row.candidate_count * width
            ]
            expected = [value for vector in encoded for value in vector]
            self.assertEqual(list(payload), list(array("f", expected)))
        self.assertEqual(training_set.split_counts(), {"SELECT": 2, "TRAIN": 4})

    def test_shared_context_matches_the_serving_feature(self) -> None:
        training_set = build_outcome_q_training_set(self.source)
        targets = build_outcome_targets(self.source)
        shared = build_player_safe_feature(targets.rows[-1].decision.input)
        start = (len(targets.rows) - 1) * len(shared)
        self.assertEqual(
            list(training_set.context[start : start + len(shared)]),
            list(array("f", shared)),
        )

    def test_identity_is_deterministic_and_source_bound(self) -> None:
        first = build_outcome_q_training_set(self.source)
        second = build_outcome_q_training_set(read_outcome_source(self.source_path))
        self.assertEqual(first.identity, second.identity)

        other = read_outcome_source(
            of.write_outcome_source(
                self.root / "other", schema=of.OUTCOME_SOURCE_SCHEMA
            )
        )
        self.assertNotEqual(
            first.identity, build_outcome_q_training_set(other).identity
        )

    def test_source_without_eligible_rows_fails_closed(self) -> None:
        with self.assertRaises(TrainingError):
            build_outcome_q_training_set(self.replace_games(()))


class OutcomeSourceProvenanceTests(_SourceCase):
    def test_accepts_strict_read_provenance(self) -> None:
        provenance = self.source.provenance()
        self.assertEqual(validate_outcome_source_provenance(provenance), provenance)

    def test_rejects_structural_violations(self) -> None:
        cases = {
            "focal rotation": lambda p: p["population"][0].update(focal_seat=1),
            "seed reuse": lambda p: p["population"][1].update(
                seed=p["population"][0]["seed"]
            ),
            "behavior": lambda p: p["behavior"].update(
                exploration_behavior_identity="other"
            ),
            "split": lambda p: p["population"][2].update(split="CALIBRATION"),
            "ordinal": lambda p: p["population"][0].update(game_ordinal=True),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                provenance = json.loads(json.dumps(self.source.provenance()))
                mutate(provenance)
                with self.assertRaises(OutcomeSourceError):
                    validate_outcome_source_provenance(provenance)


# ---------------------------------------------------------------------------
# artifact（ML runtime不要）
# ---------------------------------------------------------------------------


class OutcomeQArtifactTests(_SourceCase):
    def test_round_trip_binds_the_frozen_contract(self) -> None:
        artifact = self.write_artifact()
        loaded = load_outcome_q_artifact(self.root / "artifact")

        self.assertEqual(loaded.identity, artifact.identity)
        manifest = loaded.manifest
        self.assertEqual(manifest["artifact_schema"], OUTCOME_Q_ARTIFACT_SCHEMA)
        self.assertEqual(loaded.source_identity, self.source.identity)
        self.assertEqual(manifest["source"], self.source.provenance())
        self.assertEqual(
            loaded.training_set_identity,
            build_outcome_q_training_set(self.source).identity,
        )
        self.assertEqual(manifest["label"]["target"], OUTCOME_TARGET_IDENTITY)
        self.assertEqual(manifest["label"]["objective"], OUTCOME_OBJECTIVE_IDENTITY)
        self.assertEqual(
            manifest["label"]["selection_policy"], SEMANTIC_ENVELOPE_IDENTITY
        )
        self.assertEqual(manifest["model"]["hidden_width"], 64)
        self.assertEqual(manifest["training_set"]["splits"], {"SELECT": 2, "TRAIN": 4})

    def test_never_overwrites(self) -> None:
        self.write_artifact()
        with self.assertRaises(ModelArtifactError):
            self.write_artifact()

    def test_rejects_non_q_training_blocks_at_write(self) -> None:
        cases = {
            "objective": {
                "objective": "per-decision-legal-discard-softmax-cross-entropy"
            },
            "splits": {"train_splits": ["TRAIN", "SELECT"]},
            "select": {"validation_splits": []},
            "optimizer": {"optimizer": "sgd"},
            "epoch": {"selected_epoch": 2},
        }
        for index, (name, overrides) in enumerate(cases.items()):
            with self.subTest(name):
                with self.assertRaises(ModelArtifactError):
                    self.write_artifact(f"artifact-{index}", **overrides)
                self.assertFalse((self.root / f"artifact-{index}").exists())

    def test_rejects_non_frozen_hidden_width(self) -> None:
        with self.assertRaises(ModelArtifactError):
            write_outcome_q_artifact(
                self.root / "narrow",
                training_set=build_outcome_q_training_set(self.source),
                model_config=CandidateScorerConfig(hidden_width=2),
                training=_training_block(),
                weights=fixtures.zero_weights(CandidateScorerConfig(hidden_width=2)),
            )

    def test_strict_load_rejects_tampering(self) -> None:
        self.write_artifact()
        cases = {
            "label": lambda b: b["label"].update(target="other"),
            "schema": lambda b: b.update(artifact_schema="other"),
            "role": lambda b: b["source"].update(population_role="CALIBRATION"),
            "rotation": lambda b: b["source"]["population"][0].update(focal_seat=3),
            "split accounting": lambda b: b["training_set"].update(rows=7),
            "feature": lambda b: b["feature"].update(fingerprint="0" * 64),
            "encoding": lambda b: b["encoding"].update(identity="other"),
        }
        for index, (name, mutate) in enumerate(cases.items()):
            with self.subTest(name):
                copy = self.root / f"tampered-{index}"
                copy.mkdir()
                for child in (self.root / "artifact").iterdir():
                    (copy / child.name).write_bytes(child.read_bytes())
                body = _read_body(copy)
                mutate(body)
                _reseal(copy, body)
                with self.assertRaises(ModelArtifactError):
                    load_outcome_q_artifact(copy)

    def test_strict_load_rejects_non_finite_weights(self) -> None:
        self.write_artifact()
        root = self.root / "artifact"
        values = list(fixtures.zero_weights(MODEL))
        values[0] = math.nan
        (root / WEIGHTS_FILENAME).chmod(0o644)
        (root / WEIGHTS_FILENAME).write_bytes(_weights_bytes(values))
        body = _read_body(root)
        body["files"][WEIGHTS_FILENAME].update(file_digest(root / WEIGHTS_FILENAME))
        _reseal(root, body)
        with self.assertRaisesRegex(ModelArtifactError, "non-finite"):
            load_outcome_q_artifact(root)

    def test_is_not_a_candidate_scorer_artifact(self) -> None:
        self.write_artifact()
        with self.assertRaises(ModelArtifactError):
            load_candidate_artifact(self.root / "artifact")


# ---------------------------------------------------------------------------
# config / trainer
# ---------------------------------------------------------------------------


class OutcomeQTrainingConfigTests(unittest.TestCase):
    def test_frozen_config_values(self) -> None:
        config = FROZEN_OUTCOME_Q_TRAINING_CONFIG
        self.assertEqual(
            (
                config.epochs,
                config.batch_size,
                config.learning_rate,
                config.weight_decay,
                config.seed,
                config.model.hidden_width,
            ),
            (20, 256, 1.0e-3, 0.0, 0, 64),
        )
        value = config.to_value(selected_epoch=1, framework_version="x", diagnostics={})
        self.assertEqual(value["objective"], OUTCOME_OBJECTIVE_IDENTITY)
        self.assertEqual(value["train_splits"], ["TRAIN"])
        self.assertEqual(value["validation_splits"], ["SELECT"])

    def test_rejects_invalid_or_non_frozen_values(self) -> None:
        for overrides in (
            {"model": CandidateScorerConfig(hidden_width=128)},
            {"epochs": 0},
            {"batch_size": 0},
            {"learning_rate": math.inf},
            {"weight_decay": -1.0},
            {"seed": -1},
        ):
            with self.subTest(overrides):
                with self.assertRaises((TypeError, ValueError)):
                    OutcomeQTrainingConfig(**overrides)


class OutcomeQTrainerBoundaryTests(_SourceCase):
    def test_expected_source_identity_must_match(self) -> None:
        for expected in ("0" * 64, "not-a-digest"):
            with self.subTest(expected):
                with self.assertRaises(TrainingError):
                    train_outcome_q(
                        self.source,
                        OutcomeQTrainingConfig(epochs=1),
                        self.root / "trained",
                        expected_source_identity=expected,
                    )
        self.assertFalse((self.root / "trained").exists())

    def test_non_scientific_source_is_rejected_by_preflight(self) -> None:
        source = read_outcome_source(
            of.write_outcome_source(
                self.root / "calibration",
                of.CALIBRATION_GAMES,
                role="CALIBRATION",
                schema=self.schema,
            )
        )
        with self.assertRaisesRegex(TrainingError, "preflight"):
            train_outcome_q(
                source,
                OutcomeQTrainingConfig(epochs=1),
                self.root / "trained",
                expected_source_identity=source.identity,
            )
        self.assertFalse((self.root / "trained").exists())


@requires_ml_runtime
class OutcomeQTrainerTests(_SourceCase):
    def train(self, name="trained", **overrides):
        return train_outcome_q(
            self.source,
            OutcomeQTrainingConfig(epochs=3, **overrides),
            self.root / name,
            expected_source_identity=self.source.identity,
        )

    def test_trains_and_publishes_a_strict_loadable_artifact(self) -> None:
        artifact = self.train()
        training = artifact.manifest["training"]

        self.assertEqual(load_outcome_q_artifact(self.root / "trained"), artifact)
        self.assertTrue(1 <= training["selected_epoch"] <= 3)
        self.assertEqual(training["objective"], OUTCOME_OBJECTIVE_IDENTITY)
        diagnostics = training["diagnostics"]
        self.assertEqual(diagnostics["train_rows"], 4)
        self.assertEqual(diagnostics["select_rows"], 2)
        for value in diagnostics.values():
            self.assertTrue(math.isfinite(value))

    def test_training_is_deterministic(self) -> None:
        first = self.train("first")
        second = self.train("second")
        self.assertEqual(first.identity, second.identity)
        self.assertEqual(list(first.weights), list(second.weights))

    def test_loss_regresses_only_the_selected_candidate(self) -> None:
        import torch

        from lisjong.learning.outcome_q_training import _Tensors

        training_set = build_outcome_q_training_set(self.source)
        tensors = _Tensors(torch, training_set)
        for index, row in enumerate(training_set.rows):
            flat = row.candidate_offset + row.selected_candidate_index
            self.assertTrue(
                torch.equal(tensors.selected[index], tensors.candidates[flat])
            )
            self.assertEqual(float(tensors.target[index]), row.target_q)
        self.assertEqual(tuple(tensors.selected.shape)[0], len(training_set.rows))


# ---------------------------------------------------------------------------
# runtime / serving qualification
# ---------------------------------------------------------------------------


@requires_ml_runtime
class OutcomeQRuntimeTests(_SourceCase):
    def load(self, **kwargs):
        from lisjong.learning.outcome_q_policy import load_outcome_q_policy_factory

        artifact = self.write_artifact(**kwargs)
        return artifact, load_outcome_q_policy_factory(self.root / "artifact")

    def eligible_rows(self):
        return build_outcome_targets(self.source).rows

    def test_identity_is_the_frozen_a2_digest_and_does_not_collide(self) -> None:
        artifact, runtime = self.load()
        expected = value_digest(
            {
                "outcome_q_artifact": artifact.identity,
                "selection_policy": SEMANTIC_ENVELOPE_IDENTITY,
            }
        )
        self.assertEqual(runtime.identity, expected)
        self.assertNotIn(
            runtime.identity,
            {
                artifact.identity,
                CONSTANT_RESIDUAL_RUNTIME_IDENTITY,
                semantic_envelope_runtime_identity(artifact.identity),
            },
        )

    def test_reuses_the_semantic_envelope_policy_class(self) -> None:
        _artifact, runtime = self.load()
        first, second = runtime(), runtime.create_policy()
        self.assertIs(type(first), SemanticEnvelopeOffensePolicy)
        self.assertIs(type(ConstantResidualRuntime()()), SemanticEnvelopeOffensePolicy)
        self.assertIsNot(first, second)

    def test_equal_q_matches_the_constant_baseline(self) -> None:
        _artifact, runtime = self.load()
        baseline = ConstantResidualRuntime()()
        policy = runtime()
        for _game, item in self.source.decisions():
            result = policy.decide(item.decision)
            self.assertIs(result.action, baseline.choose_action(item.decision))
            self.assertTrue(
                any(result.action is legal for legal in item.decision.legal_actions)
            )
            if result.survivors is not None and len(result.survivors) == 1:
                self.assertFalse(result.scorer_invoked)

    def test_argmax_is_restricted_to_survivors(self) -> None:
        row = self.eligible_rows()[0]
        survivor_types = {
            row.candidates[i].action.tile.tile_type for i in row.survivors
        }
        outside = next(
            c.action.tile.tile_type
            for c in row.candidates
            if c.action.tile.tile_type not in survivor_types
        )
        _artifact, runtime = self.load(weights=_tile_type_weights(outside, 5.0))
        result = runtime().decide(row.decision)

        self.assertTrue(result.scorer_invoked)
        self.assertGreater(max(result.scores), 0.0)
        self.assertIs(result.action, row.candidates[row.survivors[0]].action)

    def test_highest_q_survivor_is_selected(self) -> None:
        row = next(
            row
            for row in self.eligible_rows()
            if row.candidates[row.survivors[-1]].action.tile.tile_type
            != row.candidates[row.survivors[0]].action.tile.tile_type
        )
        target = row.candidates[row.survivors[-1]].action
        _artifact, runtime = self.load(
            weights=_tile_type_weights(target.tile.tile_type, 1.0)
        )
        result = runtime().decide(row.decision)
        self.assertEqual(result.action.tile.tile_type, target.tile.tile_type)
        self.assertTrue(
            any(result.action is legal for legal in row.decision.legal_actions)
        )


@requires_ml_runtime
class OutcomeQServingQualificationTests(_SourceCase):
    def setUp(self) -> None:
        super().setUp()
        from lisjong.learning.outcome_q_diagnostics import evaluate_outcome_q_policy
        from lisjong.learning.outcome_q_policy import load_outcome_q_policy_factory
        from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy

        self.write_artifact()
        self.runtime = load_outcome_q_policy_factory(self.root / "artifact")
        self.evaluation = evaluate_outcome_q_policy(
            self.runtime,
            self.source,
            ["SELECT"],
            reference=TwoStepUkeirePolicy(),
        )

    def test_qualifies_on_development_data(self) -> None:
        from lisjong.learning.outcome_q_diagnostics import (
            OUTCOME_Q_SERVING_QUALIFIED,
            classify_outcome_q_serving_result,
        )

        classification = classify_outcome_q_serving_result(self.evaluation)
        self.assertEqual(classification["outcome"], OUTCOME_Q_SERVING_QUALIFIED)
        q_block = self.evaluation["outcome_q"]
        self.assertEqual(q_block["eligible_rows"], 2)
        self.assertEqual(q_block["selected_inside_survivors"], 1.0)
        self.assertEqual(q_block["argmax_mismatch"], 0)
        self.assertEqual(q_block["fraction_q_not_canonical_first"], 0.0)
        self.assertEqual(sum(b["count"] for b in q_block["calibration"]), 2)
        self.assertEqual(self.evaluation["runtime_identity"], self.runtime.identity)

    def test_q_gate_failures_are_invalid(self) -> None:
        from lisjong.learning.outcome_q_diagnostics import (
            OUTCOME_Q_SERVING_INVALID,
            classify_outcome_q_serving_result,
        )

        for name, update in (
            ("selected_inside_survivors", {"selected_inside_survivors": 0.5}),
            ("policy_matches_survivor_q_argmax", {"argmax_mismatch": 1}),
            ("eligible_rows", {"eligible_rows": 0}),
        ):
            with self.subTest(name):
                evaluation = json.loads(json.dumps(self.evaluation))
                evaluation["outcome_q"].update(update)
                classification = classify_outcome_q_serving_result(evaluation)
                self.assertEqual(classification["outcome"], OUTCOME_Q_SERVING_INVALID)
                self.assertIn(
                    name,
                    {failure["invariant"] for failure in classification["failures"]},
                )


# ---------------------------------------------------------------------------
# executable entry point
# ---------------------------------------------------------------------------


class OutcomeQEntryPointTests(_SourceCase):
    def run_main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = learning_main.main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_preflight_reports_pass(self) -> None:
        code, stdout, _stderr = self.run_main(
            "outcome-q-preflight", "--source", str(self.source_path)
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["outcome"], PREFLIGHT_PASS)

    def test_train_refuses_an_unexpected_source_identity(self) -> None:
        code, _stdout, stderr = self.run_main(
            "train-outcome-q",
            "--source",
            str(self.source_path),
            "--output",
            str(self.root / "trained"),
            "--expected-source-identity",
            "0" * 64,
        )
        self.assertEqual(code, 1)
        self.assertIn("TrainingError", stderr)
        self.assertFalse((self.root / "trained").exists())

    def test_verify_outcome_q_artifact(self) -> None:
        artifact = self.write_artifact()
        code, stdout, _stderr = self.run_main(
            "verify-outcome-q-artifact", "--artifact", str(self.root / "artifact")
        )
        self.assertEqual(code, 0)
        summary = json.loads(stdout)
        self.assertEqual(summary["artifact_identity"], artifact.identity)
        self.assertEqual(summary["source_identity"], self.source.identity)

    @requires_ml_runtime
    def test_train_uses_the_frozen_config(self) -> None:
        code, stdout, _stderr = self.run_main(
            "train-outcome-q",
            "--source",
            str(self.source_path),
            "--output",
            str(self.root / "trained"),
            "--expected-source-identity",
            self.source.identity,
        )
        self.assertEqual(code, 0)
        artifact = load_outcome_q_artifact(self.root / "trained")
        self.assertEqual(json.loads(stdout)["artifact_identity"], artifact.identity)
        training = artifact.manifest["training"]
        self.assertEqual(
            FROZEN_OUTCOME_Q_TRAINING_CONFIG.to_value(
                selected_epoch=training["selected_epoch"],
                framework_version=training["framework"]["version"],
                diagnostics=training["diagnostics"],
            ),
            training,
        )


if __name__ == "__main__":
    unittest.main()
