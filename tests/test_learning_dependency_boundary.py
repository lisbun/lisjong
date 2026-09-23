"""core / Learning間のoptional ML dependency境界のregression test。

`lisjong` coreはML runtime非依存を維持する。`lisjong.learning`のimport、
source record読み取り、feature materialization、dataset生成、model artifact
読み取りもML runtimeなしで成立し、trainingとlearned inferenceだけがoptional
extra（`lisjong[ml]`）をlazy importで要求する。

ML extraがinstall済みの環境でも同じ結論を検証できるよう、ML runtimeなしの
検証はtorchのimportを遮断した別interpreterまたは`sys.modules`遮断で行う。
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import candidate_fixtures
import learning_fixtures as fixtures

from lisjong.learning import (
    BehaviorCloningConfig,
    CandidateScorerConfig,
    CandidateScorerTrainingConfig,
    MissingLearningDependencyError,
    ModelConfig,
    load_candidate_artifact,
    load_candidate_scorer_policy_factory,
    load_learned_policy_factory,
    load_model_artifact,
    materialize_candidate_dataset,
    materialize_dataset,
    read_candidate_dataset,
    read_dataset,
    read_source_record,
    train_behavior_cloning,
    train_candidate_scorer,
    write_candidate_artifact,
    write_model_artifact,
)
from lisjong.learning.candidate_dataset import CANDIDATE_TRAINING_OBJECTIVE

_BLOCKER = """
import sys


class _TorchBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("ML runtime is blocked in this regression")
        return None


sys.meta_path.insert(0, _TorchBlocker())
"""

_CORE_MODULES = (
    "lisjong",
    "lisjong.action_vocabulary",
    "lisjong.belief",
    "lisjong.hand_evaluation",
    "lisjong.policies",
    "lisjong.policy_contract",
    "lisjong.structural_efficiency",
)

_LEARNING_MODULES = (
    "lisjong.learning",
    "lisjong.learning.artifact",
    "lisjong.learning.candidate_artifact",
    "lisjong.learning.candidate_dataset",
    "lisjong.learning.candidate_diagnostics",
    "lisjong.learning.candidate_encoding",
    "lisjong.learning.candidate_model",
    "lisjong.learning.candidate_policy",
    "lisjong.learning.candidate_training",
    "lisjong.learning.dataset",
    "lisjong.learning.errors",
    "lisjong.learning.features",
    "lisjong.learning.model",
    "lisjong.learning.policy",
    "lisjong.learning.source_record",
    "lisjong.learning.training",
    "lisjong.learning.__main__",
)


def _run_without_ml(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + body],
        capture_output=True,
        text=True,
        check=False,
    )


class NoMLRuntimeImportTests(unittest.TestCase):
    def test_core_modules_import_without_ml_runtime(self) -> None:
        body = "".join(f"import {name}\n" for name in _CORE_MODULES)
        body += 'assert "torch" not in sys.modules\nprint("ok")\n'
        result = _run_without_ml(body)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_learning_modules_import_without_ml_runtime(self) -> None:
        body = "".join(f"import {name}\n" for name in _LEARNING_MODULES)
        body += 'assert "torch" not in sys.modules\nprint("ok")\n'
        result = _run_without_ml(body)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_importing_learning_does_not_import_the_ml_runtime(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, lisjong.learning; "
                'assert "torch" not in sys.modules; print("ok")',
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_core_test_modules_run_without_ml_runtime(self) -> None:
        """ML extraなしでもcore testが成立することを別interpreterで確認する。"""
        tests_directory = Path(__file__).resolve().parent
        result = _run_without_ml(
            "import unittest\n"
            "loader = unittest.defaultTestLoader\n"
            f"suite = loader.discover({str(tests_directory)!r}, "
            'pattern="test_policy_input.py")\n'
            "result = unittest.TextTestRunner(verbosity=0).run(suite)\n"
            "assert result.wasSuccessful()\n"
            "assert suite.countTestCases() > 0\n"
            'assert "torch" not in sys.modules\n'
            'print("ok")\n'
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "ok")


class MLRuntimeRequirementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = read_source_record(
            fixtures.write_source_record(self.root / "source-record")
        )
        self.dataset = materialize_dataset(source, self.root / "dataset")
        self.model = ModelConfig(hidden_width=2)
        write_model_artifact(
            self.root / "artifact",
            dataset=self.dataset,
            model_config=self.model,
            training=fixtures.training_block(),
            weights=fixtures.zero_weights(self.model),
        )

    def test_non_ml_path_runs_without_ml_runtime(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            dataset = read_dataset(self.root / "dataset")
            artifact = load_model_artifact(self.root / "artifact")

        self.assertEqual(dataset.identity, self.dataset.identity)
        self.assertEqual(artifact.dataset_identity, self.dataset.identity)

    def test_training_requires_the_ml_runtime(self) -> None:
        config = BehaviorCloningConfig(
            train_splits=("TRAIN",), epochs=1, batch_size=2, model=self.model
        )

        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(MissingLearningDependencyError):
                train_behavior_cloning(self.dataset, config, self.root / "trained")

        self.assertFalse((self.root / "trained").exists())

    def test_learned_inference_requires_the_ml_runtime(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(MissingLearningDependencyError):
                load_learned_policy_factory(self.root / "artifact")


class CandidateScorerMLRuntimeRequirementTests(unittest.TestCase):
    """Issue #189 candidate scorer pathのoptional ML dependency境界。"""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = CandidateScorerConfig(hidden_width=2)
        source_path = candidate_fixtures.write_candidate_source_record(
            self.root / "source-record"
        )
        with patch.dict(sys.modules, {"torch": None}):
            source = read_source_record(source_path)
            self.dataset = materialize_candidate_dataset(source, self.root / "dataset")
            write_candidate_artifact(
                self.root / "artifact",
                dataset=self.dataset,
                model_config=self.model,
                training=fixtures.training_block(
                    objective=CANDIDATE_TRAINING_OBJECTIVE
                ),
                weights=fixtures.zero_weights(self.model),
            )

    def test_non_ml_path_runs_without_ml_runtime(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            dataset = read_candidate_dataset(self.root / "dataset")
            artifact = load_candidate_artifact(self.root / "artifact")

        self.assertEqual(dataset.identity, self.dataset.identity)
        self.assertEqual(artifact.dataset_identity, self.dataset.identity)

    def test_candidate_training_requires_the_ml_runtime(self) -> None:
        config = CandidateScorerTrainingConfig(
            train_splits=("TRAIN",), select_splits=("SELECT",), model=self.model
        )
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(MissingLearningDependencyError):
                train_candidate_scorer(self.dataset, config, self.root / "trained")
        self.assertFalse((self.root / "trained").exists())

    def test_candidate_inference_requires_the_ml_runtime(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(MissingLearningDependencyError):
                load_candidate_scorer_policy_factory(self.root / "artifact")


class NoArenaRuntimeDependencyTests(unittest.TestCase):
    """`lisjong -> lisjong-arena`のruntime依存が存在しないことを固定する。

    `lisjong.learning.source_record`はArenaが生成する`#342`/`#346`/`#347`
    source-record artifactを読むだけであり、Arena allocation ledger
    （`lisjong_arena.seed_registry`）を再実装・再所有・importしない。この
    invariantはML runtime有無とは独立した別のdependency boundaryである。
    """

    def test_learning_package_never_imports_lisjong_arena(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, lisjong.learning; "
                'assert not any(name == "lisjong_arena" or name.startswith('
                '"lisjong_arena.") for name in sys.modules); '
                'print("ok")',
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_source_module_has_no_lisjong_arena_import_statement(self) -> None:
        """importのない環境でも静的に固定できるよう、source上でも確認する。"""
        import ast
        import inspect

        from lisjong.learning import source_record as module

        tree = ast.parse(inspect.getsource(module))
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }

        self.assertFalse(
            any(
                name == "lisjong_arena" or name.startswith("lisjong_arena.")
                for name in imported_names
            )
        )

    def test_allocation_binding_validation_never_touches_the_network_or_disk_ledger(
        self,
    ) -> None:
        """live registryへの再照会が一切ないことを、副作用の不在で確認する。

        `validate_allocation_binding`はpure functionであり、ここでは
        filesystem / networkをmockで完全に塞いだ状態でも成立することを示す。
        """
        import socket

        from lisjong.learning.source_record import validate_allocation_binding

        def _blocked(*_args, **_kwargs):
            raise AssertionError("must not touch the network")

        binding = fixtures.allocation_binding([1, 2, 3])
        with (
            patch.object(socket, "socket", side_effect=_blocked),
            patch("builtins.open", side_effect=_blocked),
        ):
            validated = validate_allocation_binding(
                binding, seeds=[1, 2, 3], context="binding"
            )

        self.assertEqual(validated, binding)


if __name__ == "__main__":
    unittest.main()
