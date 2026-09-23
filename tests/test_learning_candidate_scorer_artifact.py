"""Issue #189 candidate scorer artifact / training / runtime factoryのtest。"""

import contextlib
import importlib
import importlib.util
import io
import json
import math
import tempfile
import types
import unittest
from pathlib import Path

import candidate_fixtures as cf
import learning_fixtures as fixtures

from lisjong.learning import (
    CANDIDATE_ARTIFACT_SCHEMA,
    CANDIDATE_ENCODING_DIMENSION,
    CandidateScorerConfig,
    CandidateScorerRuntime,
    CandidateScorerTrainingConfig,
    LearnedCandidateOffensePolicy,
    ModelArtifactError,
    TrainingError,
    load_candidate_artifact,
    load_candidate_scorer_policy_factory,
    load_model_artifact,
    materialize_candidate_dataset,
    read_candidate_dataset,
    read_source_record,
    train_candidate_scorer,
    write_candidate_artifact,
)
from lisjong.learning import __main__ as learning_main
from lisjong.learning.candidate_dataset import CANDIDATE_TRAINING_OBJECTIVE
from lisjong.learning.candidate_encoding import CANDIDATE_BLOCK_OFFSETS
from lisjong.policy_contract import DiscardAction, execute_policy

requires_ml_runtime = unittest.skipIf(
    importlib.util.find_spec("torch") is None,
    "requires the optional ML runtime (lisjong[ml])",
)


def _training_block(**overrides):
    value = fixtures.training_block(objective=CANDIDATE_TRAINING_OBJECTIVE)
    value.update(overrides)
    return value


def _feature_weights(config: CandidateScorerConfig, feature: int, weight: float):
    """score = ReLU(weight * candidate[feature])となるweightsを作る。"""
    values = [0.0] * config.parameter_count
    layout = {parameter.name: parameter for parameter in config.parameter_layout()}
    candidate = layout["candidate_layer.weight"]
    values[candidate.offset + feature] = weight  # hidden unit 0
    values[layout["output_layer.weight"].offset] = 1.0
    return tuple(values)


class _DatasetCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = read_source_record(
            cf.write_candidate_source_record(self.root / "source-record")
        )
        self.dataset = materialize_candidate_dataset(source, self.root / "dataset")
        self.model = CandidateScorerConfig(hidden_width=2)

    def write(self, name="artifact", *, weights=None, **training):
        return write_candidate_artifact(
            self.root / name,
            dataset=self.dataset,
            model_config=self.model,
            training=_training_block(**training),
            weights=weights or fixtures.zero_weights(self.model),
        )


class CandidateArtifactTests(_DatasetCase):
    def test_artifact_binds_all_identities_and_strict_loads(self) -> None:
        artifact = self.write()
        manifest = artifact.manifest

        self.assertEqual(manifest["artifact_schema"], CANDIDATE_ARTIFACT_SCHEMA)
        self.assertEqual(manifest["dataset"]["identity"], self.dataset.identity)
        self.assertEqual(manifest["encoding"], self.dataset.manifest["encoding"])
        self.assertEqual(manifest["feature"], self.dataset.manifest["feature"])
        self.assertEqual(manifest["label"], self.dataset.manifest["label"])
        self.assertEqual(manifest["source"], self.dataset.manifest["source"])
        self.assertEqual(
            load_candidate_artifact(self.root / "artifact").identity, artifact.identity
        )

    def test_existing_artifact_is_never_overwritten(self) -> None:
        self.write()
        with self.assertRaises(ModelArtifactError):
            self.write()

    def test_encoding_fingerprint_mismatch_fails_closed(self) -> None:
        self.write()
        fixtures.mutate_manifest(
            self.root / "artifact",
            lambda body: body["encoding"].update(fingerprint="0" * 64),
        )
        with self.assertRaisesRegex(ModelArtifactError, "encoding"):
            load_candidate_artifact(self.root / "artifact")

    def test_request_policy_mismatch_fails_closed(self) -> None:
        self.write()
        fixtures.mutate_manifest(
            self.root / "artifact",
            lambda body: body["encoding"].update(
                second_step_request_policy="all-candidate-second-step"
            ),
        )
        with self.assertRaisesRegex(ModelArtifactError, "request"):
            load_candidate_artifact(self.root / "artifact")

    def test_shared_feature_fingerprint_mismatch_fails_closed(self) -> None:
        self.write()
        fixtures.mutate_manifest(
            self.root / "artifact",
            lambda body: body["feature"].update(fingerprint="0" * 64),
        )
        with self.assertRaises(ModelArtifactError):
            load_candidate_artifact(self.root / "artifact")

    def test_model_config_mismatch_fails_closed(self) -> None:
        self.write()
        fixtures.mutate_manifest(
            self.root / "artifact",
            lambda body: body["model"].update(candidate_dimension=1),
        )
        with self.assertRaises(ModelArtifactError):
            load_candidate_artifact(self.root / "artifact")

    def test_training_objective_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ModelArtifactError, "objective"):
            self.write(objective="masked-action-classification")

    def test_non_finite_weights_are_rejected(self) -> None:
        weights = list(fixtures.zero_weights(self.model))
        weights[0] = math.nan
        with self.assertRaises(ModelArtifactError):
            self.write(weights=tuple(weights))
        self.assertFalse((self.root / "artifact").exists())

    def test_weights_digest_mismatch_fails_closed(self) -> None:
        self.write()
        path = self.root / "artifact" / "weights.f32"
        data = bytearray(path.read_bytes())
        data[0] ^= 0x01
        path.write_bytes(bytes(data))
        with self.assertRaises(ModelArtifactError):
            load_candidate_artifact(self.root / "artifact")

    def test_l0_and_l0_2_artifact_loaders_do_not_accept_each_other(self) -> None:
        self.write()
        with self.assertRaises(ModelArtifactError):
            load_model_artifact(self.root / "artifact")


@requires_ml_runtime
class CandidateRuntimeTests(_DatasetCase):
    def test_factory_is_a_top_level_importable_object(self) -> None:
        self.write()
        factory = load_candidate_scorer_policy_factory(self.root / "artifact")

        self.assertIsInstance(factory, CandidateScorerRuntime)
        self.assertNotIsInstance(factory, types.FunctionType)
        module = importlib.import_module(type(factory).__module__)
        self.assertIs(getattr(module, type(factory).__qualname__), type(factory))
        first, second = factory(), factory()
        self.assertIsNot(first, second)
        self.assertIsInstance(first, LearnedCandidateOffensePolicy)
        self.assertIs(first.runtime, second.runtime)

    def test_loaded_scorer_drives_the_discard_choice(self) -> None:
        red = CANDIDATE_BLOCK_OFFSETS["discard_red"]
        self.write(weights=_feature_weights(self.model, red, 1.0))
        policy = load_candidate_scorer_policy_factory(self.root / "artifact")()
        decision = cf.discard_decision(cf.TENPAI_REACHABLE_HAND)

        action = execute_policy(policy, decision)

        self.assertIsInstance(action, DiscardAction)
        self.assertTrue(action.tile.is_red)
        self.assertTrue(any(action is legal for legal in decision.legal_actions))

    def test_guards_do_not_depend_on_the_scorer(self) -> None:
        self.write()
        policy = load_candidate_scorer_policy_factory(self.root / "artifact")()
        for decision in cf.mixed_decisions():
            action = execute_policy(policy, decision)
            self.assertTrue(any(action is legal for legal in decision.legal_actions))

    def test_artifact_mismatch_fails_closed_on_factory_load(self) -> None:
        self.write()
        fixtures.mutate_manifest(
            self.root / "artifact",
            lambda body: body["encoding"].update(fingerprint="0" * 64),
        )
        with self.assertRaises(ModelArtifactError):
            load_candidate_scorer_policy_factory(self.root / "artifact")


@requires_ml_runtime
class CandidateTrainingTests(_DatasetCase):
    def config(self, **overrides):
        values = {
            "train_splits": ("TRAIN",),
            "select_splits": ("SELECT",),
            "epochs": 3,
            "batch_size": 2,
            "model": self.model,
        }
        values.update(overrides)
        return CandidateScorerTrainingConfig(**values)

    def test_training_is_deterministic_and_records_selection(self) -> None:
        first = train_candidate_scorer(self.dataset, self.config(), self.root / "a")
        second = train_candidate_scorer(self.dataset, self.config(), self.root / "b")

        self.assertEqual(
            first.manifest["files"]["weights.f32"]["sha256"],
            second.manifest["files"]["weights.f32"]["sha256"],
        )
        training = first.manifest["training"]
        self.assertEqual(training["train_splits"], ["TRAIN"])
        self.assertEqual(training["validation_splits"], ["SELECT"])
        self.assertEqual(training["objective"], CANDIDATE_TRAINING_OBJECTIVE)
        self.assertTrue(1 <= training["selected_epoch"] <= 3)
        self.assertEqual(training["diagnostics"]["select_decisions"], 3)
        self.assertEqual(training["diagnostics"]["train_decisions"], 3)

    def test_trained_artifact_serves_through_the_policy_contract(self) -> None:
        train_candidate_scorer(self.dataset, self.config(), self.root / "trained")
        policy = load_candidate_scorer_policy_factory(self.root / "trained")()
        for decision in cf.mixed_decisions():
            action = execute_policy(policy, decision)
            self.assertTrue(any(action is legal for legal in decision.legal_actions))

    def test_offline_eval_is_not_used_by_training(self) -> None:
        with self.assertRaises(ValueError):
            self.config(select_splits=("TRAIN",))
        artifact = train_candidate_scorer(
            self.dataset, self.config(), self.root / "trained"
        )
        used = set(artifact.manifest["training"]["train_splits"]) | set(
            artifact.manifest["training"]["validation_splits"]
        )
        self.assertNotIn("OFFLINE-EVAL", used)

    def test_missing_split_fails_closed(self) -> None:
        with self.assertRaises(TrainingError):
            train_candidate_scorer(
                self.dataset,
                self.config(select_splits=("HOLDOUT",)),
                self.root / "trained",
            )
        self.assertFalse((self.root / "trained").exists())


@requires_ml_runtime
class EvaluateEntryPointTests(_DatasetCase):
    def run_main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = learning_main.main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def arguments(self, split):
        return (
            "evaluate-candidate-scorer",
            "--artifact",
            str(self.root / "artifact"),
            "--source-record",
            str(self.root / "source-record"),
            "--split",
            split,
        )

    def test_training_or_selection_split_is_refused(self) -> None:
        self.write()
        for split in ("TRAIN", "SELECT"):
            code, stdout, stderr = self.run_main(*self.arguments(split))
            with self.subTest(split=split):
                self.assertEqual(code, 1)
                self.assertEqual(stdout, "")
                self.assertIn("used for training or epoch selection", stderr)

    def test_held_out_split_reports_the_frozen_gate_classification(self) -> None:
        self.write()
        code, stdout, stderr = self.run_main(*self.arguments("OFFLINE-EVAL"))

        self.assertEqual(code, 0, msg=stderr)
        report = json.loads(stdout)
        self.assertEqual(report["evaluation"]["splits"], ["OFFLINE-EVAL"])
        # fixtureはsupportが小さいため、PASSと推測せずINVALIDになる。
        self.assertEqual(report["classification"]["outcome"], "STOP / INVALID")


@requires_ml_runtime
class PerDecisionLossTests(unittest.TestCase):
    """lossが同じdecisionのlegal candidateだけで正規化されることを固定する。"""

    def setUp(self) -> None:
        import torch

        from lisjong.learning import candidate_training

        self.torch = torch
        self.loss = candidate_training._per_decision_loss

    def compute(self, scores, counts, teacher):
        torch = self.torch
        counts_tensor = torch.tensor(counts)
        decision_index = torch.repeat_interleave(
            torch.arange(len(counts)), counts_tensor
        )
        local_starts = torch.cumsum(counts_tensor, 0) - counts_tensor
        return self.loss(
            torch,
            torch.tensor(scores, dtype=torch.float32),
            decision_index,
            local_starts,
            torch.tensor(teacher),
            len(counts),
        ).tolist()

    def test_loss_matches_a_per_decision_softmax(self) -> None:
        scores = [1.0, 2.0, 0.5, -1.0, 3.0]
        losses = self.compute(scores, [2, 3], [1, 0])

        first = math.log(math.exp(1.0) + math.exp(2.0)) - 2.0
        second = math.log(math.exp(0.5) + math.exp(-1.0) + math.exp(3.0)) - 0.5
        self.assertAlmostEqual(losses[0], first, places=5)
        self.assertAlmostEqual(losses[1], second, places=5)

    def test_other_decisions_do_not_leak_probability_mass(self) -> None:
        base = self.compute([1.0, 2.0, 0.5, -1.0, 3.0], [2, 3], [1, 0])
        changed = self.compute([1.0, 2.0, 50.0, 40.0, 30.0], [2, 3], [1, 0])
        self.assertAlmostEqual(base[0], changed[0], places=6)

    def test_single_candidate_decision_has_zero_loss(self) -> None:
        losses = self.compute([7.0, 1.0, 2.0], [1, 2], [0, 1])
        self.assertAlmostEqual(losses[0], 0.0, places=6)


class DimensionContractTests(unittest.TestCase):
    def test_model_config_binds_both_input_contracts(self) -> None:
        from lisjong.learning import FEATURE_DIMENSION

        value = CandidateScorerConfig(hidden_width=3).to_value()
        self.assertEqual(value["context_dimension"], FEATURE_DIMENSION)
        self.assertEqual(value["candidate_dimension"], CANDIDATE_ENCODING_DIMENSION)
        self.assertEqual(value["output_dimension"], 1)


class DatasetReadbackForTrainingTests(_DatasetCase):
    def test_training_reads_only_strict_datasets(self) -> None:
        with self.assertRaises(TrainingError):
            train_candidate_scorer(
                object(),
                CandidateScorerTrainingConfig(
                    train_splits=("TRAIN",), select_splits=("SELECT",)
                ),
                self.root / "never",
            )
        self.assertEqual(
            read_candidate_dataset(self.root / "dataset").identity,
            self.dataset.identity,
        )


if __name__ == "__main__":
    unittest.main()
