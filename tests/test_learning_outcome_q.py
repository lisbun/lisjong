"""Issue #200 outcome-Q rows / preflight / trainer / artifact / runtimeのtest。

#374の実データは使わない。#193 / #195の合成outcome source fixtureだけで、
row materialization、#79 §8 preflight、selected-action MSE trainer、
purpose-specific artifactのstrict load、`SemanticEnvelopeOffensePolicy`への
runtime接続を固定する。
"""

import dataclasses
import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

import candidate_fixtures as cf
import outcome_fixtures as of

from lisjong.learning import (
    CONSTANT_RESIDUAL_RUNTIME_IDENTITY,
    OUTCOME_OBJECTIVE_IDENTITY,
    OUTCOME_Q_ARTIFACT_SCHEMA,
    OUTCOME_Q_CHECKPOINT_RULE,
    OUTCOME_TARGET_IDENTITY,
    SEMANTIC_ENVELOPE_IDENTITY,
    ConstantResidualRuntime,
    DatasetError,
    MissingLearningDependencyError,
    ModelArtifactError,
    OutcomeQPolicyLoader,
    OutcomeQTrainingConfig,
    SemanticEnvelopeOffensePolicy,
    TrainingError,
    classify_semantic_envelope_result,
    evaluate_semantic_envelope_policy,
    evaluate_semantic_envelope_policy_by_game,
    load_outcome_q_artifact,
    load_outcome_q_policy_factory,
    materialize_outcome_q_rows,
    outcome_q_preflight,
    outcome_q_runtime_identity,
    read_outcome_source,
    read_source_record,
    semantic_envelope_runtime_identity,
    train_outcome_q,
)
from lisjong.learning._canonical import canonical_json_text, seal, value_digest
from lisjong.learning._o0 import O0DecisionKind
from lisjong.learning.candidate_encoding import (
    CANDIDATE_ENCODING_DIMENSION,
    encoding_block,
)
from lisjong.learning.dataset import feature_block
from lisjong.learning.features import FEATURE_DIMENSION
from lisjong.learning.outcome_q_artifact import (
    MANIFEST_FILENAME,
    OUTCOME_Q_MODEL,
    WEIGHTS_FILENAME,
    write_outcome_q_artifact,
)
from lisjong.learning.outcome_q_dataset import MINIMUM_SUPPORT_RATE
from lisjong.learning.outcome_q_policy import OutcomeQRuntime
from lisjong.learning.outcome_source import OutcomeTargets
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import DecisionContext

HAS_TORCH = importlib.util.find_spec("torch") is not None

GAMES = tuple(("TRAIN", 5000 + index) for index in range(6)) + tuple(
    ("SELECT", 6000 + index) for index in range(2)
)
"""8 hanchan（TRAIN 6 / SELECT 2）。TRAIN supportは#79 §8の20% / 20%を満たす。"""

CONFIG = OutcomeQTrainingConfig(
    epochs=3, batch_size=4, learning_rate=1.0e-2, weight_decay=0.0, seed=7
)
"""test専用のconfig。scientific valueではない（#200 DP-1）。"""


def _with_selection(rows, choose):
    """TRAIN rowのselected candidateを`choose(row)`へ置き換えたrowsを返す。"""
    replaced = tuple(
        dataclasses.replace(row, selected_candidate_index=choose(row))
        if row.split == "TRAIN"
        else row
        for row in rows.rows
    )
    targets = OutcomeTargets(
        source_identity=rows.targets.source_identity,
        rows=replaced,
        excluded=rows.targets.excluded,
        hanchan_count=rows.targets.hanchan_count,
    )
    return dataclasses.replace(rows, targets=targets)


class _SourceCase(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def source(self, name="source", games=GAMES, *, engine=True, role="SCIENTIFIC"):
        write = of.write_engine_outcome_source if engine else of.write_outcome_source
        return read_outcome_source(write(self.root / name, games, role=role))


class RowsTests(_SourceCase):
    def test_materializes_eligible_rows_for_both_lineages(self) -> None:
        for engine in (True, False):
            with self.subTest(engine=engine):
                source = self.source(f"source-{engine}", engine=engine)
                rows = materialize_outcome_q_rows(source)

                self.assertEqual(rows.split_counts(), {"TRAIN": 12, "SELECT": 4})
                self.assertEqual(rows.hanchan_counts(), {"TRAIN": 6, "SELECT": 2})
                self.assertEqual(len(rows.context), len(rows.rows) * FEATURE_DIMENSION)
                self.assertEqual(
                    rows.candidate_count,
                    sum(len(row.candidates) for row in rows.rows),
                )
                self.assertEqual(rows.candidate_offsets[0], 0)
                for row in rows.rows:
                    self.assertGreaterEqual(len(row.survivors), 2)
                    self.assertIn(row.selected_candidate_index, row.survivors)
                self.assertEqual(rows.to_value()["identity"], rows.identity)

    def test_identity_is_deterministic_and_binds_the_source(self) -> None:
        source = self.source()
        first = materialize_outcome_q_rows(source)

        self.assertEqual(first.identity, materialize_outcome_q_rows(source).identity)
        other = materialize_outcome_q_rows(self.source("historical", engine=False))
        self.assertNotEqual(first.identity, other.identity)

    def test_non_scientific_sources_are_rejected(self) -> None:
        for role, games in (
            ("DIAGNOSTIC", of.DIAGNOSTIC_GAMES),
            ("CALIBRATION", of.CALIBRATION_GAMES),
        ):
            with self.subTest(role=role):
                source = self.source(role, games, role=role)
                with self.assertRaises(DatasetError):
                    materialize_outcome_q_rows(source)

    def test_missing_select_rows_are_rejected(self) -> None:
        source = self.source("train-only", GAMES[:6])
        with self.assertRaisesRegex(DatasetError, "SELECT"):
            materialize_outcome_q_rows(source)

    def test_requires_a_strict_read_source(self) -> None:
        with self.assertRaises(DatasetError):
            materialize_outcome_q_rows(object())


class PreflightTests(_SourceCase):
    def setUp(self) -> None:
        super().setUp()
        self.rows = materialize_outcome_q_rows(self.source())

    def test_preflight_record(self) -> None:
        preflight = outcome_q_preflight(self.rows)

        self.assertEqual(preflight["hard_stops"], [])
        self.assertEqual(preflight["target_identity"], OUTCOME_TARGET_IDENTITY)
        self.assertEqual(preflight["objective_identity"], OUTCOME_OBJECTIVE_IDENTITY)
        self.assertEqual(preflight["rows_identity"], self.rows.identity)
        self.assertEqual(preflight["source"]["identity"], self.rows.source.identity)
        train = preflight["train"]
        self.assertEqual(train["eligible_row_count"], 12)
        self.assertEqual(train["hanchan_count"], 6)
        self.assertIsNotNone(train["target_q"])
        self.assertEqual(
            sum(train["selected_survivor_position_distribution"].values()), 12
        )
        self.assertNotIn("excluded", train)

    def test_preflight_does_not_read_select_outcomes(self) -> None:
        select = outcome_q_preflight(self.rows)["select"]

        self.assertEqual(
            set(select),
            {
                "eligible_row_count",
                "hanchan_count",
                "survivor_count_distribution",
                "unique_eligible_kyoku_count",
            },
        )
        # SELECT targetを書き換えてもpreflightは変わらない。
        shifted = dataclasses.replace(
            self.rows,
            targets=OutcomeTargets(
                source_identity=self.rows.targets.source_identity,
                rows=tuple(
                    dataclasses.replace(row, target_q=row.target_q + 100.0)
                    if row.split == "SELECT"
                    else row
                    for row in self.rows.rows
                ),
                excluded=self.rows.targets.excluded,
                hanchan_count=self.rows.targets.hanchan_count,
            ),
        )
        self.assertEqual(outcome_q_preflight(shifted), outcome_q_preflight(self.rows))

    def test_support_hard_stops(self) -> None:
        self.assertEqual(MINIMUM_SUPPORT_RATE, 0.2)
        cases = (
            (lambda row: row.survivors[0], "train_non_canonical_first"),
            (lambda row: row.survivors[-1], "train_canonical_first"),
        )
        for choose, prefix in cases:
            with self.subTest(prefix=prefix):
                rows = _with_selection(self.rows, choose)
                stops = outcome_q_preflight(rows)["hard_stops"]
                self.assertEqual(len(stops), 1)
                self.assertTrue(stops[0].startswith(prefix))
                with self.assertRaisesRegex(TrainingError, "preflight"):
                    train_outcome_q(rows, CONFIG, self.root / prefix)
                self.assertFalse((self.root / prefix).exists())

    def test_non_finite_target_hard_stop(self) -> None:
        rows = dataclasses.replace(
            self.rows,
            targets=OutcomeTargets(
                source_identity=self.rows.targets.source_identity,
                rows=(
                    dataclasses.replace(self.rows.rows[0], target_q=float("nan")),
                    *self.rows.rows[1:],
                ),
                excluded=self.rows.targets.excluded,
                hanchan_count=self.rows.targets.hanchan_count,
            ),
        )
        preflight = outcome_q_preflight(rows)
        self.assertEqual(preflight["hard_stops"], ["non_finite_target"])
        self.assertIsNone(preflight["train"])
        with self.assertRaisesRegex(TrainingError, "non_finite_target"):
            train_outcome_q(rows, CONFIG, self.root / "nan")
        self.assertFalse((self.root / "nan").exists())


class ConfigAndIdentityTests(unittest.TestCase):
    def test_config_has_no_scientific_defaults(self) -> None:
        with self.assertRaises(TypeError):
            OutcomeQTrainingConfig()  # type: ignore[call-arg]
        self.assertFalse(
            any(
                field.default is not dataclasses.MISSING
                or field.default_factory is not dataclasses.MISSING
                for field in dataclasses.fields(OutcomeQTrainingConfig)
            )
        )
        self.assertNotIn(
            "model",
            {field.name for field in dataclasses.fields(OutcomeQTrainingConfig)},
        )

    def test_config_validation(self) -> None:
        base = dataclasses.asdict(CONFIG)
        for field, value in (
            ("epochs", 0),
            ("epochs", 1.0),
            ("batch_size", 0),
            ("learning_rate", 0.0),
            ("learning_rate", 1),
            ("weight_decay", -1.0),
            ("weight_decay", float("nan")),
            ("seed", -1),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    OutcomeQTrainingConfig(**{**base, field: value})

    def test_runtime_identity_is_the_frozen_a2_digest(self) -> None:
        artifact_identity = "c" * 64
        identity = outcome_q_runtime_identity(artifact_identity)

        self.assertEqual(
            identity,
            value_digest(
                {
                    "outcome_q_artifact": artifact_identity,
                    "selection_policy": "lisjong-offense-l0.2-semantic-envelope-v1",
                }
            ),
        )
        self.assertNotEqual(identity, artifact_identity)
        self.assertNotEqual(identity, CONSTANT_RESIDUAL_RUNTIME_IDENTITY)
        self.assertNotEqual(
            identity, semantic_envelope_runtime_identity(artifact_identity)
        )

    def test_frozen_identities(self) -> None:
        self.assertEqual(
            OUTCOME_Q_ARTIFACT_SCHEMA, "lisjong-offense-l0.3-outcome-q-artifact-v1"
        )
        self.assertEqual(
            OUTCOME_Q_CHECKPOINT_RULE,
            "lisjong-offense-l0.3-min-select-mse-earliest-epoch-v1",
        )
        self.assertEqual(OUTCOME_Q_MODEL.hidden_width, 64)
        self.assertEqual(
            OUTCOME_OBJECTIVE_IDENTITY,
            "lisjong-offense-l0.3-selected-action-mc-q-mse-v1",
        )


class MlBoundaryTests(_SourceCase):
    def test_training_and_inference_require_the_ml_runtime(self) -> None:
        rows = materialize_outcome_q_rows(self.source())
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(MissingLearningDependencyError):
                train_outcome_q(rows, CONFIG, self.root / "trained")
            with self.assertRaises(MissingLearningDependencyError):
                load_outcome_q_policy_factory(self.root / "missing")
        self.assertFalse((self.root / "trained").exists())


@unittest.skipUnless(HAS_TORCH, "requires the optional ML runtime")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        cls.source = read_outcome_source(
            of.write_engine_outcome_source(cls.root / "source", GAMES)
        )
        cls.rows = materialize_outcome_q_rows(cls.source)
        cls.artifact = train_outcome_q(cls.rows, CONFIG, cls.root / "artifact")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_artifact_binds_the_frozen_contract(self) -> None:
        manifest = self.artifact.manifest

        self.assertEqual(manifest["artifact_schema"], OUTCOME_Q_ARTIFACT_SCHEMA)
        self.assertEqual(manifest["source"], self.source.provenance())
        self.assertEqual(manifest["rows"], self.rows.to_value())
        self.assertEqual(manifest["feature"], feature_block())
        self.assertEqual(manifest["encoding"], encoding_block())
        self.assertEqual(manifest["selection_policy"], SEMANTIC_ENVELOPE_IDENTITY)
        self.assertEqual(
            manifest["objective"],
            {
                "checkpoint_rule": OUTCOME_Q_CHECKPOINT_RULE,
                "objective": OUTCOME_OBJECTIVE_IDENTITY,
                "target": OUTCOME_TARGET_IDENTITY,
            },
        )
        self.assertEqual(manifest["model"], OUTCOME_Q_MODEL.to_value())
        training = manifest["training"]
        self.assertEqual(training["train_splits"], ["TRAIN"])
        self.assertEqual(training["validation_splits"], ["SELECT"])
        self.assertEqual(training["objective"], OUTCOME_OBJECTIVE_IDENTITY)
        self.assertTrue(1 <= self.artifact.selected_epoch <= CONFIG.epochs)
        self.assertEqual(
            load_outcome_q_artifact(self.root / "artifact").identity,
            self.artifact.identity,
        )

    def test_diagnostics(self) -> None:
        diagnostics = self.artifact.diagnostics

        self.assertEqual(diagnostics["train_rows"], 12)
        self.assertEqual(diagnostics["select_rows"], 4)
        for name in (
            "select_mse",
            "select_mean_survivor_q_spread",
            "select_fraction_q_argmax_not_canonical_first",
            "select_prediction_mean",
            "select_target_mean",
        ):
            self.assertIn(name, diagnostics)
        self.assertTrue(
            0.0 <= diagnostics["select_fraction_q_argmax_not_canonical_first"] <= 1.0
        )
        self.assertEqual(
            diagnostics["select_target_mean"],
            sum(row.target_q for row in self.rows.rows if row.split == "SELECT") / 4,
        )

    def test_training_is_deterministic(self) -> None:
        again = train_outcome_q(self.rows, CONFIG, self.root / "again")
        self.assertEqual(again.identity, self.artifact.identity)
        self.assertEqual(list(again.weights), list(self.artifact.weights))

        other = dataclasses.replace(CONFIG, seed=CONFIG.seed + 1)
        reseeded = train_outcome_q(self.rows, other, self.root / "reseeded")
        self.assertNotEqual(list(reseeded.weights), list(self.artifact.weights))

    def test_unselected_candidates_do_not_enter_the_loss(self) -> None:
        candidates = array("f", self.rows.candidates)
        for offset, row in zip(
            self.rows.candidate_offsets, self.rows.rows, strict=True
        ):
            for index in range(len(row.candidates)):
                if index == row.selected_candidate_index:
                    continue
                start = (offset + index) * CANDIDATE_ENCODING_DIMENSION
                for column in range(CANDIDATE_ENCODING_DIMENSION):
                    candidates[start + column] += 3.0
        perturbed = dataclasses.replace(self.rows, candidates=candidates)
        artifact = train_outcome_q(perturbed, CONFIG, self.root / "perturbed")

        self.assertEqual(list(artifact.weights), list(self.artifact.weights))
        self.assertEqual(artifact.selected_epoch, self.artifact.selected_epoch)

    def test_no_overwrite(self) -> None:
        with self.assertRaises(TrainingError):
            train_outcome_q(self.rows, CONFIG, self.root / "artifact")
        with self.assertRaises(ModelArtifactError):
            write_outcome_q_artifact(
                self.root / "artifact",
                rows=self.rows,
                training=self.artifact.manifest["training"],
                weights=self.artifact.weights,
            )


@unittest.skipUnless(HAS_TORCH, "requires the optional ML runtime")
class ArtifactStrictLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        source = read_outcome_source(
            of.write_engine_outcome_source(cls.root / "source", GAMES)
        )
        cls.rows = materialize_outcome_q_rows(source)
        cls.config = dataclasses.replace(CONFIG, epochs=1)
        cls.artifact = train_outcome_q(cls.rows, cls.config, cls.root / "artifact")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def copy(self, name):
        import shutil

        path = self.root / name
        shutil.copytree(self.root / "artifact", path)
        return path

    def mutate(self, name, mutate):
        path = self.copy(name)
        manifest = json.loads((path / MANIFEST_FILENAME).read_text("utf-8"))
        manifest.pop("identity")
        mutate(manifest)
        (path / MANIFEST_FILENAME).write_text(
            canonical_json_text(seal(manifest)), encoding="utf-8", newline="\n"
        )
        return path

    def test_manifest_tamper_fails_closed(self) -> None:
        cases = {
            "schema": lambda m: m.update(artifact_schema="other-v1"),
            "checkpoint": lambda m: m["objective"].update(checkpoint_rule="x"),
            "target": lambda m: m["objective"].update(target="x"),
            "selection": lambda m: m.update(selection_policy="x"),
            "feature": lambda m: m["feature"].update(fingerprint="0" * 64),
            "encoding": lambda m: m["encoding"].update(fingerprint="0" * 64),
            "diagnostic_role": lambda m: m["source"].update(
                population_role="DIAGNOSTIC"
            ),
            "behavior": lambda m: m["source"]["behavior"].update(
                exploration_behavior_identity="x"
            ),
            "population_split": lambda m: m["source"]["population"][0].update(
                split="DIAGNOSTIC"
            ),
            "focal_seat": lambda m: m["source"]["population"][1].update(focal_seat=0),
            "seed_domain": lambda m: m["source"]["allocation_bindings"]["TRAIN"].update(
                seed_domain="riichienv-4p-red-half-hanchan-v1"
            ),
            "rows_hanchan": lambda m: m["rows"]["hanchan"].update(TRAIN=5),
            "rows_accounting": lambda m: m["rows"]["splits"].update(SELECT=0),
            "model_width": lambda m: m["model"].update(hidden_width=32),
            "validation_splits": lambda m: m["training"].update(
                validation_splits=["TRAIN"]
            ),
            "objective": lambda m: m["training"].update(objective="huber"),
            "selected_epoch": lambda m: m["training"].update(selected_epoch=2),
            "extra": lambda m: m.update(extra=1),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                path = self.mutate(name, mutate)
                with self.assertRaises(ModelArtifactError):
                    load_outcome_q_artifact(path)

    def test_payload_tamper_fails_closed(self) -> None:
        path = self.copy("weights")
        data = bytearray((path / WEIGHTS_FILENAME).read_bytes())
        data[0] ^= 1
        (path / WEIGHTS_FILENAME).write_bytes(bytes(data))
        with self.assertRaises(ModelArtifactError):
            load_outcome_q_artifact(path)

        path = self.copy("extra-file")
        (path / "notes.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(ModelArtifactError):
            load_outcome_q_artifact(path)

        path = self.copy("digest")
        text = (path / MANIFEST_FILENAME).read_text("utf-8")
        (path / MANIFEST_FILENAME).write_text(
            text.replace('"SCIENTIFIC"', '"CALIBRATION"'),
            encoding="utf-8",
            newline="\n",
        )
        with self.assertRaises(ModelArtifactError):
            load_outcome_q_artifact(path)

    def test_non_ml_load_path(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            artifact = load_outcome_q_artifact(self.root / "artifact")
        self.assertEqual(artifact.identity, self.artifact.identity)


@unittest.skipUnless(HAS_TORCH, "requires the optional ML runtime")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        source = read_outcome_source(
            of.write_engine_outcome_source(cls.root / "source", GAMES)
        )
        cls.decisions = tuple(item.decision for _, item in source.decisions())
        rows = materialize_outcome_q_rows(source)
        artifact = train_outcome_q(rows, CONFIG, cls.root / "artifact")
        cls.artifact = artifact
        cls.runtime = load_outcome_q_policy_factory(cls.root / "artifact")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def runtime_with(self, weights):
        from lisjong.learning.candidate_model import build_scorer_module

        return OutcomeQRuntime(
            artifact=self.artifact,
            module=build_scorer_module(OUTCOME_Q_MODEL, weights),
        )

    def test_identity_and_policy_class(self) -> None:
        self.assertEqual(self.runtime.artifact_identity, self.artifact.identity)
        self.assertEqual(
            self.runtime.identity, outcome_q_runtime_identity(self.artifact.identity)
        )
        policy = self.runtime()
        self.assertIs(type(policy), SemanticEnvelopeOffensePolicy)
        self.assertIsNot(policy, self.runtime())

    def test_same_policy_path_as_the_constant_baseline(self) -> None:
        baseline = ConstantResidualRuntime().create_policy()
        policy = self.runtime.create_policy()
        residual = 0
        for decision in self.decisions:
            result = policy.decide(decision)
            expected = baseline.decide(decision)
            with self.subTest(kind=result.kind):
                self.assertIs(result.kind, expected.kind)
                self.assertTrue(
                    any(result.action is legal for legal in decision.legal_actions)
                )
                if result.kind is not O0DecisionKind.DISCARD:
                    self.assertIs(result.action, expected.action)
                    continue
                self.assertEqual(result.survivors, expected.survivors)
                survivor_actions = [
                    result.candidates[index].action for index in result.survivors
                ]
                self.assertTrue(
                    any(result.action is action for action in survivor_actions)
                )
                if len(result.survivors) == 1:
                    self.assertFalse(result.scorer_invoked)
                    self.assertIs(result.action, expected.action)
                else:
                    residual += 1
                    self.assertEqual(len(result.scores), len(result.candidates))
        self.assertGreater(residual, 0)

    def test_equal_q_uses_the_canonical_first_survivor(self) -> None:
        policy = self.runtime_with([0.0] * OUTCOME_Q_MODEL.parameter_count)()
        for decision in self.decisions:
            result = policy.decide(decision)
            if result.scorer_invoked:
                self.assertIs(
                    result.action, result.candidates[result.survivors[0]].action
                )

    def test_non_finite_q_fails_closed(self) -> None:
        from lisjong.learning import LearnedPolicyError

        policy = self.runtime_with([float("nan")] * OUTCOME_Q_MODEL.parameter_count)()
        residual = next(
            decision
            for decision in self.decisions
            if self.runtime().decide(decision).scorer_invoked
        )
        with self.assertRaises(LearnedPolicyError):
            policy.decide(residual)

    def test_score_shape_checks(self) -> None:
        from lisjong.learning import LearnedPolicyError

        with self.assertRaises(LearnedPolicyError):
            self.runtime.score((0.0,), ((0.0,) * CANDIDATE_ENCODING_DIMENSION,))
        with self.assertRaises(LearnedPolicyError):
            self.runtime.score((0.0,) * FEATURE_DIMENSION, ())

    def test_step_e_replay_diagnostics_accept_the_q_runtime(self) -> None:
        """既存の#191 replay qualificationがQ runtimeでそのまま動く（DP-4は未凍結）。"""
        record = read_source_record(
            cf.write_candidate_source_record(self.root / "record")
        )
        evaluation = evaluate_semantic_envelope_policy(
            self.runtime.create_policy(),
            record,
            ["TRAIN", "SELECT"],
            reference=TwoStepUkeirePolicy(),
        )
        classification = classify_semantic_envelope_result(evaluation)

        self.assertEqual(evaluation["invariants"]["legality"], 1.0)
        self.assertEqual(
            evaluation["invariants"]["regret_violations"],
            {"shanten": 0, "current_ukeire": 0, "second_step": 0},
        )
        self.assertEqual(evaluation["reference"]["disagreement_outside_survivors"], 0)
        self.assertEqual(classification["failures"], [])

    def test_policy_loader_is_picklable_and_strictly_loads_the_artifact(self) -> None:
        from lisjong.learning import LearnedPolicyError

        loader = OutcomeQPolicyLoader(self.root / "artifact", self.artifact.identity)
        restored = pickle.loads(pickle.dumps(loader))
        self.assertEqual(restored, loader)
        policy = restored()
        self.assertIs(type(policy), SemanticEnvelopeOffensePolicy)
        for decision in self.decisions:
            self.assertEqual(
                policy.decide(decision).action, self.runtime().decide(decision).action
            )
        with self.assertRaises(LearnedPolicyError):
            OutcomeQPolicyLoader(self.root / "artifact", "0" * 64)()

    def test_by_game_replay_with_the_q_loader_matches_the_sequential_replay(
        self,
    ) -> None:
        record = read_source_record(
            cf.write_candidate_source_record(self.root / "record-by-game")
        )
        expected = evaluate_semantic_envelope_policy(
            self.runtime.create_policy(),
            record,
            ["TRAIN", "SELECT"],
            reference=TwoStepUkeirePolicy(),
        )
        loader = OutcomeQPolicyLoader(self.root / "artifact", self.artifact.identity)
        for workers in (1, 2):
            with self.subTest(workers=workers):
                evaluation = evaluate_semantic_envelope_policy_by_game(
                    loader,
                    [game.decisions for game in record.games],
                    ["TRAIN", "SELECT"],
                    workers=workers,
                    reference_factory=TwoStepUkeirePolicy,
                )
                evaluation.pop("runtime_by_branch")
                self.assertEqual(
                    evaluation,
                    {
                        key: value
                        for key, value in expected.items()
                        if key != "runtime_by_branch"
                    },
                )

    def test_decision_context_is_the_only_input(self) -> None:
        decision = self.decisions[0]
        self.assertIsInstance(decision, DecisionContext)
        self.assertEqual(
            self.runtime().choose_action(decision),
            self.runtime().decide(decision).action,
        )


if __name__ == "__main__":
    unittest.main()
