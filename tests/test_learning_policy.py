"""LearnedPolicy inferenceとArena互換public seamのcontract test。"""

import ast
import importlib
import importlib.util
import inspect
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import learning_fixtures as fixtures

from lisjong.action_vocabulary import ACTION_VOCABULARY_SIZE, encode_action
from lisjong.learning import (
    LearnedOffensePolicy,
    LearnedPolicyError,
    LearnedPolicyRuntime,
    ModelArtifactError,
    ModelConfig,
    load_learned_policy_factory,
    materialize_dataset,
    read_source_record,
    write_model_artifact,
)
from lisjong.learning import policy as policy_module
from lisjong.policy_contract import (
    DecisionContext,
    DiscardAction,
    InternalAction,
    execute_policy,
)

requires_ml_runtime = unittest.skipIf(
    importlib.util.find_spec("torch") is None,
    "requires the optional ML runtime (lisjong[ml])",
)


def _bias_weights(model: ModelConfig, bias_values: dict[int, float]):
    """output biasだけを持つweightsを作る。

    input layerのweight / biasが0のため、ReLU後のhiddenも0になり、logitsは
    output biasと一致する。これによりselection規則だけを検証できる。
    """
    values = [0.0] * model.parameter_count
    layout = {parameter.name: parameter for parameter in model.parameter_layout()}
    offset = layout["output_layer.bias"].offset
    for index, value in bias_values.items():
        values[offset + index] = value
    return tuple(values)


class PolicySourceContractTests(unittest.TestCase):
    def test_inference_module_has_no_exception_fallback(self) -> None:
        """例外を飲み込むfallback経路が存在しないことをsource上で固定する。"""
        tree = ast.parse(inspect.getsource(policy_module))
        nodes = list(ast.walk(tree))

        self.assertEqual(
            [node for node in nodes if isinstance(node, ast.ExceptHandler)], []
        )
        self.assertEqual([node for node in nodes if isinstance(node, ast.Lambda)], [])
        imported = {
            alias.name
            for node in nodes
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in nodes
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("random", imported)
        self.assertFalse(any(name.startswith("torch") for name in imported))

    def test_policy_contract_signature(self) -> None:
        signature = inspect.signature(LearnedOffensePolicy.choose_action)

        self.assertEqual(list(signature.parameters), ["self", "decision"])
        self.assertIs(signature.return_annotation, InternalAction)


class LearnedPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = read_source_record(
            fixtures.write_source_record(self.root / "source-record")
        )
        self.dataset = materialize_dataset(source, self.root / "dataset")
        self.model = ModelConfig(hidden_width=2)
        self.decision = self.build_decision()
        self.legal_indices = sorted(
            encode_action(action) for action in self.decision.legal_actions
        )
        self.illegal_index = next(
            index
            for index in range(ACTION_VOCABULARY_SIZE)
            if index not in set(self.legal_indices)
        )

    def build_decision(self) -> DecisionContext:
        value = fixtures.policy_input()
        return DecisionContext(
            input=value,
            legal_actions=fixtures.discard_legal_actions(value, with_riichi=True),
        )

    def artifact(self, bias_values, name="artifact"):
        return write_model_artifact(
            self.root / name,
            dataset=self.dataset,
            model_config=self.model,
            training=fixtures.training_block(),
            weights=_bias_weights(self.model, bias_values),
        )

    @requires_ml_runtime
    def test_inference_returns_the_canonical_legal_action(self) -> None:
        target = self.legal_indices[1]
        self.artifact({target: 1.0})
        factory = load_learned_policy_factory(self.root / "artifact")
        policy = factory()

        action = policy.choose_action(self.decision)

        self.assertTrue(
            any(action is candidate for candidate in self.decision.legal_actions)
        )
        self.assertEqual(encode_action(action), target)

    @requires_ml_runtime
    def test_illegal_index_never_wins_even_with_the_largest_logit(self) -> None:
        target = self.legal_indices[2]
        self.artifact({self.illegal_index: 9.0, target: 1.0})
        policy = load_learned_policy_factory(self.root / "artifact")()

        action = policy.choose_action(self.decision)

        self.assertEqual(encode_action(action), target)
        self.assertIn(action, self.decision.legal_actions)

    @requires_ml_runtime
    def test_inference_passes_the_normal_policy_execution_boundary(self) -> None:
        target = self.legal_indices[0]
        self.artifact({target: 2.0})
        policy = load_learned_policy_factory(self.root / "artifact")()

        action = execute_policy(policy, self.decision)

        self.assertEqual(encode_action(action), target)

    @requires_ml_runtime
    def test_selection_is_deterministic_and_tie_breaks_on_the_lowest_index(
        self,
    ) -> None:
        first, second = self.legal_indices[0], self.legal_indices[1]
        self.artifact({first: 3.0, second: 3.0})
        policy = load_learned_policy_factory(self.root / "artifact")()

        first_action = policy.choose_action(self.decision)
        second_action = policy.choose_action(self.decision)

        self.assertEqual(first_action, second_action)
        self.assertEqual(encode_action(first_action), first)

    @requires_ml_runtime
    def test_factory_creates_a_fresh_policy_instance_per_call(self) -> None:
        self.artifact({self.legal_indices[0]: 1.0})
        factory = load_learned_policy_factory(self.root / "artifact")

        first, second = factory(), factory()

        self.assertIsNot(first, second)
        self.assertIsInstance(first, LearnedOffensePolicy)
        self.assertIs(first.runtime, second.runtime)
        self.assertEqual(
            first.choose_action(self.decision), second.choose_action(self.decision)
        )
        self.assertIsNot(factory.create_policy(), factory.create_policy())

    @requires_ml_runtime
    def test_factory_is_a_top_level_importable_object(self) -> None:
        self.artifact({self.legal_indices[0]: 1.0})
        factory = load_learned_policy_factory(self.root / "artifact")

        self.assertIsInstance(factory, LearnedPolicyRuntime)
        self.assertNotIsInstance(factory, types.FunctionType)
        self.assertNotIsInstance(factory, types.LambdaType)
        module = importlib.import_module(type(factory).__module__)
        self.assertIs(getattr(module, type(factory).__qualname__), type(factory))
        self.assertEqual(type(factory).__module__, "lisjong.learning.policy")
        self.assertTrue(callable(factory))

    @requires_ml_runtime
    def test_arena_style_consumer_needs_only_the_artifact_and_public_seam(
        self,
    ) -> None:
        """consumerはartifact pathとpublic factoryだけでPolicyを実行できる。

        dataset / source recordを削除した別processで実行し、trainer internals、
        dataset artifact、Arena側実装のいずれも必要としないことを固定する。
        """
        target = self.legal_indices[1]
        self.artifact({target: 4.0})
        shutil.rmtree(self.root / "dataset")
        shutil.rmtree(self.root / "source-record")

        program = (
            "import sys\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "import learning_fixtures as fixtures\n"
            "from lisjong.learning import load_learned_policy_factory\n"
            "from lisjong.policy_contract import DecisionContext, execute_policy\n"
            "from lisjong.action_vocabulary import encode_action\n"
            f"factory = load_learned_policy_factory({str(self.root / 'artifact')!r})\n"
            "value = fixtures.policy_input()\n"
            "decision = DecisionContext(\n"
            "    input=value,\n"
            "    legal_actions=fixtures.discard_legal_actions(\n"
            "        value, with_riichi=True\n"
            "    ),\n"
            ")\n"
            "action = execute_policy(factory(), decision)\n"
            "print(encode_action(action))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), str(target))

    @requires_ml_runtime
    def test_artifact_identity_mismatch_fails_closed_on_load(self) -> None:
        self.artifact({self.legal_indices[0]: 1.0})
        fixtures.mutate_manifest(
            self.root / "artifact",
            lambda body: body["feature"].update(fingerprint="0" * 64),
        )

        with self.assertRaises(ModelArtifactError):
            load_learned_policy_factory(self.root / "artifact")

    @requires_ml_runtime
    def test_non_decision_context_input_rejected(self) -> None:
        self.artifact({self.legal_indices[0]: 1.0})
        policy = load_learned_policy_factory(self.root / "artifact")()

        with self.assertRaises(LearnedPolicyError):
            policy.choose_action(self.decision.input)

    @requires_ml_runtime
    def test_illegal_only_decision_fails_closed(self) -> None:
        """vocabularyへencodeできないlegal候補しか無い場合もfallbackしない。"""
        self.artifact({self.legal_indices[0]: 1.0})
        policy = load_learned_policy_factory(self.root / "artifact")()

        class UnknownAction(DiscardAction):
            __slots__ = ()

        decision = DecisionContext(
            input=self.decision.input,
            legal_actions=(
                UnknownAction(
                    actor=self.decision.input.self_seat,
                    tile=self.decision.input.own_hand.concealed_tiles[0],
                    tsumogiri=False,
                ),
            ),
        )

        with self.assertRaises(Exception) as captured:
            policy.choose_action(decision)
        self.assertNotIsInstance(captured.exception, AssertionError)


class BrokenRuntimeTests(unittest.TestCase):
    """logitsが壊れている場合にheuristic fallbackへ落ちないことを固定する。"""

    def setUp(self) -> None:
        value = fixtures.policy_input()
        self.decision = DecisionContext(
            input=value, legal_actions=fixtures.discard_legal_actions(value)
        )

    def test_non_finite_logits_fail_closed(self) -> None:
        class NonFiniteRuntime:
            def logits(self, values):
                return (float("nan"),) * ACTION_VOCABULARY_SIZE

        policy = LearnedOffensePolicy(NonFiniteRuntime())

        with self.assertRaisesRegex(LearnedPolicyError, "non-finite"):
            policy.choose_action(self.decision)

    def test_wrong_logit_count_fails_closed(self) -> None:
        class ShortRuntime:
            def logits(self, values):
                return (0.0, 1.0)

        policy = LearnedOffensePolicy(ShortRuntime())

        with self.assertRaisesRegex(LearnedPolicyError, "logit count"):
            policy.choose_action(self.decision)


if __name__ == "__main__":
    unittest.main()
