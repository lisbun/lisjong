"""Issue #97 `DecisionTrace` / `AnalysisTrace` / recorderのunit test。"""

import ast
import inspect
import unittest
from dataclasses import FrozenInstanceError, dataclass, fields, replace

import lisjong.policy_contract.analysis_trace as analysis_trace_module
import lisjong.policy_contract.decision_trace as decision_trace_module
import lisjong.policy_contract.policy_decision as policy_decision_module
import lisjong.policy_contract.policy_execution as policy_execution_module
from lisjong.policy_contract import (
    AnalysisTrace,
    DecisionTrace,
    DecisionTraceRecorder,
    DecisionTraceSink,
    DiscardAction,
    InternalAction,
    PassAction,
    PolicyDecision,
    RiichiAction,
    Seat,
    Tile,
    TileCategory,
    TileType,
)

MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_6 = Tile(TileType(TileCategory.MANZU, 6))

DISCARD_4M = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
DISCARD_6M = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_6, tsumogiri=False)
PASS = PassAction(actor=Seat.SEAT_0)
RIICHI = RiichiAction(actor=Seat.SEAT_0)


@dataclass(frozen=True, slots=True)
class _StubAnalysis(AnalysisTrace):
    label: str


@dataclass(slots=True)
class _MutableAnalysis(AnalysisTrace):
    label: str


class _NonDataclassAnalysis(AnalysisTrace):
    __slots__ = ()


class _AmbiguousPassAction(PassAction):
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        return True


class AnalysisTraceContractTest(unittest.TestCase):
    def test_root_contract_defines_no_free_form_payload_fields(self) -> None:
        self.assertEqual(AnalysisTrace.__slots__, ())
        self.assertFalse(hasattr(AnalysisTrace, "metrics"))
        self.assertFalse(hasattr(AnalysisTrace, "reason"))

    def test_typed_frozen_dataclass_payload_is_accepted(self) -> None:
        analysis = _StubAnalysis(label="two-step")

        trace = DecisionTrace(
            legal_actions=(DISCARD_4M,),
            selected_action=DISCARD_4M,
            analysis=analysis,
        )

        self.assertIs(trace.analysis, analysis)

    def test_free_form_and_mutable_payloads_are_rejected(self) -> None:
        rejected = (
            {"ukeire": 4},
            {"ukeire": 4.0},
            "この牌が一番良いと思ったから",
            42,
            _MutableAnalysis(label="mutable"),
            _NonDataclassAnalysis(),
        )

        for analysis in rejected:
            with self.subTest(analysis=type(analysis).__name__):
                with self.assertRaisesRegex(TypeError, "analysis must be None"):
                    DecisionTrace(
                        legal_actions=(DISCARD_4M,),
                        selected_action=DISCARD_4M,
                        analysis=analysis,
                    )

    def test_payload_subclass_instances_are_immutable(self) -> None:
        analysis = _StubAnalysis(label="two-step")

        with self.assertRaises(FrozenInstanceError):
            analysis.label = "changed"
        self.assertFalse(hasattr(analysis, "__dict__"))


class DecisionTraceValueTest(unittest.TestCase):
    def test_value_is_an_immutable_three_field_snapshot(self) -> None:
        trace = DecisionTrace(
            legal_actions=(DISCARD_4M, PASS),
            selected_action=PASS,
        )

        self.assertEqual(
            tuple(field.name for field in fields(trace)),
            ("legal_actions", "selected_action", "analysis"),
        )
        self.assertIsNone(trace.analysis)
        self.assertFalse(hasattr(trace, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            trace.selected_action = DISCARD_4M

    def test_legal_actions_are_normalized_to_an_immutable_tuple(self) -> None:
        legal_actions = [DISCARD_4M, PASS]

        trace = DecisionTrace(
            legal_actions=legal_actions,
            selected_action=PASS,
        )
        legal_actions.append(DISCARD_6M)

        self.assertIsInstance(trace.legal_actions, tuple)
        self.assertEqual(trace.legal_actions, (DISCARD_4M, PASS))

    def test_selected_action_is_normalized_to_the_canonical_legal_candidate(
        self,
    ) -> None:
        equivalent = replace(DISCARD_4M)
        self.assertIsNot(equivalent, DISCARD_4M)

        trace = DecisionTrace(
            legal_actions=(DISCARD_4M, PASS),
            selected_action=equivalent,
        )

        self.assertIs(trace.selected_action, DISCARD_4M)

    def test_rejects_empty_legal_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            DecisionTrace(legal_actions=(), selected_action=PASS)

    def test_rejects_non_action_members_and_non_iterable_legal_actions(self) -> None:
        with self.assertRaisesRegex(TypeError, "only InternalAction"):
            DecisionTrace(legal_actions=(PASS, object()), selected_action=PASS)
        with self.assertRaisesRegex(TypeError, "must be an iterable"):
            DecisionTrace(legal_actions=7, selected_action=PASS)

    def test_rejects_semantic_duplicates_in_legal_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate semantic actions"):
            DecisionTrace(
                legal_actions=(PASS, replace(PASS)),
                selected_action=PASS,
            )

    def test_rejects_non_action_selected_action(self) -> None:
        with self.assertRaisesRegex(TypeError, "selected_action must be"):
            DecisionTrace(legal_actions=(PASS,), selected_action=None)

    def test_selected_action_must_match_exactly_one_legal_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "found 0 matches"):
            DecisionTrace(legal_actions=(PASS,), selected_action=DISCARD_4M)

        with self.assertRaisesRegex(ValueError, "found 2 matches"):
            DecisionTrace(
                legal_actions=(PASS, RIICHI),
                selected_action=_AmbiguousPassAction(actor=Seat.SEAT_0),
            )

    def test_none_analysis_means_not_produced_and_is_not_an_empty_payload(self) -> None:
        trace = DecisionTrace(legal_actions=(PASS,), selected_action=PASS)

        self.assertIsNone(trace.analysis)
        self.assertIsNot(trace.analysis, ())
        self.assertNotEqual(
            trace,
            DecisionTrace(
                legal_actions=(PASS,),
                selected_action=PASS,
                analysis=_StubAnalysis(label=""),
            ),
        )

    def test_trace_holds_no_decision_context_or_privileged_field(self) -> None:
        field_names = {field.name for field in fields(DecisionTrace)}

        self.assertEqual(field_names, {"legal_actions", "selected_action", "analysis"})
        for forbidden in (
            "input",
            "decision",
            "policy",
            "sink",
            "game_trace",
            "sequence",
            "observation",
            "wall",
            "reason",
        ):
            self.assertNotIn(forbidden, field_names)


class PolicyDecisionValueTest(unittest.TestCase):
    def test_value_pairs_a_proposed_action_with_optional_analysis(self) -> None:
        analysis = _StubAnalysis(label="two-step")
        decision = PolicyDecision(action=DISCARD_4M, analysis=analysis)

        self.assertIs(decision.action, DISCARD_4M)
        self.assertIs(decision.analysis, analysis)
        self.assertIsNone(PolicyDecision(action=DISCARD_4M).analysis)
        with self.assertRaises(FrozenInstanceError):
            decision.action = PASS

    def test_proposed_action_is_not_matched_against_legal_actions(self) -> None:
        # PolicyDecision.actionはPolicyが「提案した」Actionであり、
        # DecisionTrace.selected_actionのようなcanonical合法Actionではない。
        equivalent = replace(DISCARD_4M)

        self.assertIs(PolicyDecision(action=equivalent).action, equivalent)

    def test_rejects_non_action_and_free_form_analysis(self) -> None:
        with self.assertRaisesRegex(TypeError, "action must be an InternalAction"):
            PolicyDecision(action=None)
        with self.assertRaisesRegex(TypeError, "analysis must be None"):
            PolicyDecision(action=DISCARD_4M, analysis={"score": 1})


class DecisionTraceSinkContractTest(unittest.TestCase):
    def test_sink_protocol_is_one_way_and_returns_none(self) -> None:
        annotations = DecisionTraceSink.on_decision.__annotations__

        self.assertEqual(annotations["trace"], DecisionTrace)
        self.assertIsNone(annotations["return"])
        self.assertFalse(getattr(DecisionTraceSink, "_is_runtime_protocol", False))


class DecisionTraceRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = DecisionTraceRecorder()
        self.first = DecisionTrace(
            legal_actions=(DISCARD_4M, PASS), selected_action=DISCARD_4M
        )
        self.second = DecisionTrace(
            legal_actions=(DISCARD_4M, PASS), selected_action=PASS
        )

    def test_zero_records_snapshot_is_an_empty_tuple(self) -> None:
        snapshot = self.recorder.snapshot()

        self.assertEqual(snapshot, ())
        self.assertIsInstance(snapshot, tuple)

    def test_notification_order_is_preserved(self) -> None:
        self.recorder.on_decision(self.first)
        self.recorder.on_decision(self.second)

        self.assertEqual(self.recorder.snapshot(), (self.first, self.second))

    def test_one_normal_notification_appends_exactly_one_record(self) -> None:
        self.recorder.on_decision(self.first)

        self.assertEqual(len(self.recorder.snapshot()), 1)

    def test_previous_snapshot_is_detached_from_later_records(self) -> None:
        self.recorder.on_decision(self.first)
        snapshot = self.recorder.snapshot()

        self.recorder.on_decision(self.second)

        self.assertEqual(snapshot, (self.first,))
        self.assertEqual(self.recorder.snapshot(), (self.first, self.second))

    def test_recorder_rejects_values_that_are_not_decision_traces(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a DecisionTrace"):
            self.recorder.on_decision(object())

        self.assertEqual(self.recorder.snapshot(), ())

    def test_recorder_keeps_no_mutable_public_state(self) -> None:
        self.recorder.on_decision(self.first)

        self.assertEqual(DecisionTraceRecorder.__slots__, ("_traces",))
        self.assertFalse(hasattr(self.recorder, "__dict__"))


class InternalActionUnionTest(unittest.TestCase):
    def test_trace_uses_the_shared_internal_action_union(self) -> None:
        annotations = DecisionTrace.__annotations__

        self.assertEqual(annotations["legal_actions"], tuple[InternalAction, ...])
        self.assertEqual(annotations["selected_action"], InternalAction)


class TraceDependencyDirectionTest(unittest.TestCase):
    """`policy_contract`が具体Policy実装や環境へ逆依存しないことを固定する。"""

    _TRACE_MODULES = (
        analysis_trace_module,
        decision_trace_module,
        policy_decision_module,
        policy_execution_module,
    )

    _FORBIDDEN_PREFIXES = (
        "lisjong.policies",
        "lisjong.belief",
        "lisjong.hand_evaluation",
        "lisjong_arena",
        "mahjong",
        "riichienv",
        "websockets",
    )

    def test_trace_modules_import_no_policy_implementation_or_environment(
        self,
    ) -> None:
        for module in self._TRACE_MODULES:
            tree = ast.parse(inspect.getsource(module))
            imported = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            } | {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }

            for name in imported:
                for prefix in self._FORBIDDEN_PREFIXES:
                    with self.subTest(module=module.__name__, imported=name):
                        self.assertFalse(name.startswith(prefix))

    def test_trace_modules_reference_no_game_trace_or_correlation_contract(
        self,
    ) -> None:
        for module in self._TRACE_MODULES:
            source = inspect.getsource(module)
            for forbidden in ("GameTrace(", "correlation_id", "global_sequence"):
                with self.subTest(module=module.__name__, symbol=forbidden):
                    self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
