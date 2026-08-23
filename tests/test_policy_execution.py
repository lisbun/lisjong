import inspect
import unittest
from dataclasses import FrozenInstanceError, dataclass, fields, replace

from lisjong.policies import MinimalPolicy
from lisjong.policy_contract import (
    AnalysisTrace,
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DecisionContext,
    DecisionTrace,
    DecisionTraceRecorder,
    DiscardAction,
    InternalAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    PassAction,
    PolicyActionValidationError,
    PolicyDecision,
    PonAction,
    RiichiAction,
    RonAction,
    TsumoAction,
    execute_policy,
    execute_policy_with_trace,
)
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

MANZU_3 = Tile(TileType(TileCategory.MANZU, 3))
MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))
MANZU_6 = Tile(TileType(TileCategory.MANZU, 6))
PINZU_5 = Tile(TileType(TileCategory.PINZU, 5))


def _make_input() -> PolicyInput:
    player = PlayerPublicState(
        score=25000, discards=(), melds=(), riichi=RiichiState.NONE
    )
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(MANZU_3,),
            live_wall_tiles_remaining=70,
        ),
        players=(player, player, player, player),
        own_hand=OwnHandState(
            concealed_tiles=(MANZU_4, MANZU_5, MANZU_6),
            drawn_tile=MANZU_5,
        ),
    )


def _decision(*actions: InternalAction) -> DecisionContext:
    return DecisionContext(input=_make_input(), legal_actions=actions)


class _ReturningPolicy:
    def __init__(self, selected: object) -> None:
        self.selected = selected
        self.received: DecisionContext | None = None

    def choose_action(self, decision: DecisionContext) -> object:
        self.received = decision
        return self.selected


class _RaisingPolicy:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        raise self.error


class _AmbiguousPassAction(PassAction):
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        return True


class _UncomparablePassAction(PassAction):
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("comparison failed")


@dataclass(frozen=True, slots=True)
class _StubAnalysis(AnalysisTrace):
    label: str


class _CountingPolicy:
    """decision algorithmの実行回数を数えるlegacy Policy。"""

    def __init__(self, selected: object) -> None:
        self.selected = selected
        self.calls = 0

    def choose_action(self, decision: DecisionContext) -> object:
        self.calls += 1
        return self.selected


class _CountingAnalysisPolicy:
    """1回のdecision計算からaction + analysisを返すanalysis-capable Policy。"""

    def __init__(self, selected: object, analysis: object = None) -> None:
        self.selected = selected
        self.analysis = analysis
        self.calls = 0
        self.legacy_calls = 0

    def _decide(self, decision: DecisionContext) -> object:
        self.calls += 1
        return PolicyDecision(action=self.selected, analysis=self.analysis)

    def choose_action(self, decision: DecisionContext) -> object:
        self.legacy_calls += 1
        return self._decide(decision).action

    def choose_action_with_analysis(self, decision: DecisionContext) -> object:
        return self._decide(decision)


class _BrokenAnalysisPolicy:
    """analysis capabilityがPolicyDecision以外を返す契約違反Policy。"""

    def __init__(self, result: object) -> None:
        self.result = result

    def choose_action(self, decision: DecisionContext) -> object:
        raise AssertionError("traced execution must use the analysis capability")

    def choose_action_with_analysis(self, decision: DecisionContext) -> object:
        return self.result


class _AnalysisCapableBasePolicy:
    """analysis capabilityを定義する基底Policy。"""

    def __init__(self, base_action: object, analysis: object = None) -> None:
        self.base_action = base_action
        self.analysis = analysis
        self.base_decide_calls = 0

    def _decide(self, decision: DecisionContext) -> PolicyDecision:
        self.base_decide_calls += 1
        return PolicyDecision(action=self.base_action, analysis=self.analysis)

    def choose_action(self, decision: DecisionContext) -> object:
        return self._decide(decision).action

    def choose_action_with_analysis(self, decision: DecisionContext) -> PolicyDecision:
        return self._decide(decision)


class _ChooseActionOverrideOnlySubPolicy(_AnalysisCapableBasePolicy):
    """`choose_action()`だけをoverrideし、analysis capabilityを継承するsubclass。"""

    def __init__(
        self, base_action: object, own_action: object, analysis: object = None
    ) -> None:
        super().__init__(base_action, analysis)
        self.own_action = own_action
        self.own_calls = 0

    def choose_action(self, decision: DecisionContext) -> object:
        self.own_calls += 1
        return self.own_action


class _ExplicitAnalysisOverrideSubPolicy(_AnalysisCapableBasePolicy):
    """analysis capabilityを明示overrideするsubclass。"""

    def __init__(
        self, base_action: object, own_action: object, analysis: object = None
    ) -> None:
        super().__init__(base_action, analysis)
        self.own_action = own_action
        self.own_analysis_calls = 0

    def choose_action_with_analysis(self, decision: DecisionContext) -> PolicyDecision:
        self.own_analysis_calls += 1
        return PolicyDecision(action=self.own_action, analysis=self.analysis)


class _InnerPathOverrideSubPolicy(_AnalysisCapableBasePolicy):
    """analysis pathの内側だけをoverrideするsubclass。"""

    def __init__(
        self, base_action: object, own_action: object, analysis: object = None
    ) -> None:
        super().__init__(base_action, analysis)
        self.own_action = own_action

    def _decide(self, decision: DecisionContext) -> PolicyDecision:
        self.base_decide_calls += 1
        return PolicyDecision(action=self.own_action, analysis=self.analysis)


class _RecordingSink:
    def __init__(self) -> None:
        self.traces: list[DecisionTrace] = []

    def on_decision(self, trace: DecisionTrace) -> None:
        self.traces.append(trace)


class _RaisingSink:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def on_decision(self, trace: DecisionTrace) -> None:
        self.calls += 1
        raise self.error


class PolicyExecutionTest(unittest.TestCase):
    def test_minimal_policy_can_be_called_through_boundary(self) -> None:
        legal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
        decision = _decision(legal, PassAction(actor=Seat.SEAT_0))

        selected = execute_policy(MinimalPolicy(), decision)

        self.assertIs(selected, legal)

    def test_passes_the_exact_decision_context_object_to_policy(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)
        decision = _decision(legal)
        policy = _ReturningPolicy(legal)

        execute_policy(policy, decision)

        self.assertIs(policy.received, decision)

    def test_returns_canonical_legal_candidate_for_equal_distinct_result(self) -> None:
        legal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
        policy_result = replace(legal)
        self.assertIsNot(policy_result, legal)

        selected = execute_policy(_ReturningPolicy(policy_result), _decision(legal))

        self.assertIs(selected, legal)

    def test_uses_one_common_validation_path_for_all_action_variants(self) -> None:
        actions: tuple[InternalAction, ...] = (
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
            RiichiAction(actor=Seat.SEAT_0),
            ChiAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_3,
                called_tile=MANZU_5,
                consumed_tiles=(MANZU_4, MANZU_6),
            ),
            PonAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_1,
                called_tile=PINZU_5,
                consumed_tiles=(PINZU_5, PINZU_5),
            ),
            DaiminkanAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_1,
                called_tile=PINZU_5,
                consumed_tiles=(PINZU_5, PINZU_5, PINZU_5),
            ),
            AnkanAction(
                actor=Seat.SEAT_0,
                tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5),
            ),
            KakanAction(
                actor=Seat.SEAT_0,
                added_tile=PINZU_5,
                from_seat=Seat.SEAT_1,
                called_tile=PINZU_5,
            ),
            RonAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_1,
                winning_tile=PINZU_5,
            ),
            TsumoAction(actor=Seat.SEAT_0, winning_tile=PINZU_5),
            PassAction(actor=Seat.SEAT_0),
            KyuushuKyuuhaiAction(actor=Seat.SEAT_0),
        )

        for legal in actions:
            with self.subTest(action_type=type(legal).__name__):
                selected = execute_policy(
                    _ReturningPolicy(replace(legal)), _decision(legal)
                )
                self.assertIs(selected, legal)

    def test_accepts_duck_typed_policy_without_inheritance(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)

        selected = execute_policy(_ReturningPolicy(legal), _decision(legal))

        self.assertIs(selected, legal)

    def test_rejects_action_outside_legal_actions_with_zero_matches(self) -> None:
        legal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
        outside = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_6, tsumogiri=False)

        with self.assertRaisesRegex(PolicyActionValidationError, "found 0 matches"):
            execute_policy(_ReturningPolicy(outside), _decision(legal))

    def test_rejects_non_action_result(self) -> None:
        with self.assertRaisesRegex(
            PolicyActionValidationError, "must return an InternalAction"
        ):
            execute_policy(
                _ReturningPolicy(None), _decision(PassAction(actor=Seat.SEAT_0))
            )

    def test_rejects_multiple_semantic_matches(self) -> None:
        decision = _decision(
            PassAction(actor=Seat.SEAT_0), RiichiAction(actor=Seat.SEAT_0)
        )

        with self.assertRaisesRegex(PolicyActionValidationError, "found 2 matches"):
            execute_policy(
                _ReturningPolicy(_AmbiguousPassAction(actor=Seat.SEAT_0)),
                decision,
            )

    def test_rejects_result_that_cannot_be_compared_safely(self) -> None:
        with self.assertRaisesRegex(
            PolicyActionValidationError, "could not be compared safely"
        ) as caught:
            execute_policy(
                _ReturningPolicy(_UncomparablePassAction(actor=Seat.SEAT_0)),
                _decision(PassAction(actor=Seat.SEAT_0)),
            )

        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_policy_exception_propagates_unchanged_without_fallback(self) -> None:
        error = RuntimeError("policy failed")

        with self.assertRaises(RuntimeError) as caught:
            execute_policy(
                _RaisingPolicy(error), _decision(PassAction(actor=Seat.SEAT_0))
            )

        self.assertIs(caught.exception, error)

    def test_public_api_requires_only_policy_and_decision(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(execute_policy).parameters),
            ("policy", "decision"),
        )


class TracedPolicyExecutionTest(unittest.TestCase):
    def test_legacy_policy_needs_no_change_and_reports_no_analysis(self) -> None:
        legal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
        decision = _decision(legal, PassAction(actor=Seat.SEAT_0))
        recorder = DecisionTraceRecorder()

        selected = execute_policy_with_trace(MinimalPolicy(), decision, recorder)

        (trace,) = recorder.snapshot()
        self.assertIs(selected, legal)
        self.assertIs(trace.selected_action, legal)
        self.assertIsNone(trace.analysis)

    def test_traced_execution_notifies_the_sink_exactly_once(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)
        sink = _RecordingSink()

        execute_policy_with_trace(_ReturningPolicy(legal), _decision(legal), sink)

        self.assertEqual(len(sink.traces), 1)

    def test_policy_algorithm_runs_exactly_once_per_traced_decision(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)
        legacy = _CountingPolicy(legal)
        analysis_capable = _CountingAnalysisPolicy(legal, _StubAnalysis(label="x"))

        execute_policy_with_trace(legacy, _decision(legal), DecisionTraceRecorder())
        execute_policy_with_trace(
            analysis_capable, _decision(legal), DecisionTraceRecorder()
        )

        self.assertEqual(legacy.calls, 1)
        self.assertEqual(analysis_capable.calls, 1)
        self.assertEqual(analysis_capable.legacy_calls, 0)

    def test_trace_records_the_canonical_action_for_an_equal_distinct_result(
        self,
    ) -> None:
        legal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
        policy_result = replace(legal)
        self.assertIsNot(policy_result, legal)
        recorder = DecisionTraceRecorder()

        selected = execute_policy_with_trace(
            _ReturningPolicy(policy_result), _decision(legal), recorder
        )

        (trace,) = recorder.snapshot()
        self.assertIs(selected, legal)
        self.assertIs(trace.selected_action, legal)
        self.assertIsNot(trace.selected_action, policy_result)

    def test_canonical_action_holds_for_an_analysis_capable_policy(self) -> None:
        legal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
        proposed = replace(legal)
        policy = _CountingAnalysisPolicy(proposed, _StubAnalysis(label="x"))
        recorder = DecisionTraceRecorder()

        selected = execute_policy_with_trace(policy, _decision(legal), recorder)

        (trace,) = recorder.snapshot()
        self.assertIs(selected, legal)
        self.assertIs(trace.selected_action, legal)

    def test_trace_keeps_only_the_presented_legal_actions(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)
        riichi = RiichiAction(actor=Seat.SEAT_0)
        decision = _decision(legal, riichi)
        recorder = DecisionTraceRecorder()

        execute_policy_with_trace(_ReturningPolicy(legal), decision, recorder)

        (trace,) = recorder.snapshot()
        self.assertEqual(trace.legal_actions, decision.legal_actions)

    def test_typed_analysis_is_forwarded_unchanged(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)
        analysis = _StubAnalysis(label="two-step")
        recorder = DecisionTraceRecorder()

        execute_policy_with_trace(
            _CountingAnalysisPolicy(legal, analysis), _decision(legal), recorder
        )

        (trace,) = recorder.snapshot()
        self.assertIs(trace.analysis, analysis)

    def test_invalid_policy_output_emits_no_trace(self) -> None:
        sink = _RecordingSink()
        outside = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_6, tsumogiri=False)
        legal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)

        for result in (None, outside):
            with (
                self.subTest(result=result),
                self.assertRaises(PolicyActionValidationError),
            ):
                execute_policy_with_trace(
                    _ReturningPolicy(result), _decision(legal), sink
                )

        self.assertEqual(sink.traces, [])

    def test_analysis_capability_must_return_a_policy_decision(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)
        sink = _RecordingSink()

        with self.assertRaisesRegex(
            PolicyActionValidationError, "must return a PolicyDecision"
        ):
            execute_policy_with_trace(
                _BrokenAnalysisPolicy(legal), _decision(legal), sink
            )

        self.assertEqual(sink.traces, [])

    def test_policy_exception_emits_no_trace_and_propagates_unchanged(self) -> None:
        error = RuntimeError("policy failed")
        sink = _RecordingSink()

        with self.assertRaises(RuntimeError) as caught:
            execute_policy_with_trace(
                _RaisingPolicy(error), _decision(PassAction(actor=Seat.SEAT_0)), sink
            )

        self.assertIs(caught.exception, error)
        self.assertEqual(sink.traces, [])

    def test_sink_exception_propagates_without_fallback_or_policy_retry(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)
        error = RuntimeError("sink failed")
        policy = _CountingPolicy(legal)
        sink = _RaisingSink(error)

        with self.assertRaises(RuntimeError) as caught:
            execute_policy_with_trace(policy, _decision(legal), sink)

        self.assertIs(caught.exception, error)
        self.assertEqual(policy.calls, 1)
        self.assertEqual(sink.calls, 1)

    def test_rejects_a_sink_without_a_callable_on_decision(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)
        policy = _CountingPolicy(legal)

        for sink in (None, object(), type("_Stub", (), {"on_decision": 1})()):
            with self.subTest(sink=type(sink).__name__):
                with self.assertRaisesRegex(TypeError, "sink must provide"):
                    execute_policy_with_trace(policy, _decision(legal), sink)

        self.assertEqual(policy.calls, 0)

    def test_rejects_a_decision_that_is_not_a_decision_context(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a DecisionContext"):
            execute_policy_with_trace(
                MinimalPolicy(), object(), DecisionTraceRecorder()
            )

    def test_traced_api_is_a_separate_opt_in_three_argument_function(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(execute_policy).parameters),
            ("policy", "decision"),
        )
        self.assertEqual(
            tuple(inspect.signature(execute_policy_with_trace).parameters),
            ("policy", "decision", "sink"),
        )


class TraceNonInterferenceTest(unittest.TestCase):
    def test_traced_and_untraced_execution_select_the_same_action(self) -> None:
        legal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
        other = PassAction(actor=Seat.SEAT_0)

        untraced = execute_policy(MinimalPolicy(), _decision(legal, other))
        traced = execute_policy_with_trace(
            MinimalPolicy(), _decision(legal, other), DecisionTraceRecorder()
        )

        self.assertEqual(untraced, traced)

    def test_untraced_execution_never_uses_the_analysis_capability(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)
        policy = _CountingAnalysisPolicy(legal, _StubAnalysis(label="x"))

        selected = execute_policy(policy, _decision(legal))

        self.assertIs(selected, legal)
        self.assertEqual(policy.legacy_calls, 1)
        self.assertEqual(policy.calls, 1)

    def test_decision_context_carries_no_trace_sink_or_privileged_field(self) -> None:
        field_names = {field.name for field in fields(DecisionContext)}

        self.assertEqual(field_names, {"input", "legal_actions"})

    def test_traced_execution_does_not_mutate_the_decision_context(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)
        decision = _decision(legal)

        execute_policy_with_trace(
            _ReturningPolicy(legal), decision, DecisionTraceRecorder()
        )

        self.assertEqual(decision.legal_actions, (legal,))
        with self.assertRaises(FrozenInstanceError):
            decision.legal_actions = ()


class InheritedAnalysisCapabilityDispatchTest(unittest.TestCase):
    """Issue #97: 偶然inheritしたanalysis pathがdecision semanticsを変えないこと。"""

    def setUp(self) -> None:
        self.base_action = PassAction(actor=Seat.SEAT_0)
        self.own_action = RiichiAction(actor=Seat.SEAT_0)
        self.analysis = _StubAnalysis(label="base")
        self.decision = _decision(self.base_action, self.own_action)

    def test_choose_action_override_only_subclass_bypasses_inherited_capability(
        self,
    ) -> None:
        policy = _ChooseActionOverrideOnlySubPolicy(
            self.base_action, self.own_action, self.analysis
        )
        recorder = DecisionTraceRecorder()

        traced = execute_policy_with_trace(policy, self.decision, recorder)

        (trace,) = recorder.snapshot()
        self.assertIs(traced, self.own_action)
        self.assertIs(trace.selected_action, self.own_action)
        self.assertIsNone(trace.analysis)
        self.assertEqual(policy.own_calls, 1)
        self.assertEqual(policy.base_decide_calls, 0)

    def test_choose_action_override_only_subclass_agrees_with_untraced_execution(
        self,
    ) -> None:
        untraced_policy = _ChooseActionOverrideOnlySubPolicy(
            self.base_action, self.own_action, self.analysis
        )
        traced_policy = _ChooseActionOverrideOnlySubPolicy(
            self.base_action, self.own_action, self.analysis
        )

        untraced = execute_policy(untraced_policy, self.decision)
        traced = execute_policy_with_trace(
            traced_policy, self.decision, DecisionTraceRecorder()
        )

        self.assertIs(untraced, traced)
        self.assertEqual(untraced_policy.own_calls, 1)
        self.assertEqual(traced_policy.own_calls, 1)

    def test_base_policy_still_uses_its_own_analysis_capability(self) -> None:
        policy = _AnalysisCapableBasePolicy(self.base_action, self.analysis)
        recorder = DecisionTraceRecorder()

        traced = execute_policy_with_trace(policy, self.decision, recorder)

        (trace,) = recorder.snapshot()
        self.assertIs(traced, self.base_action)
        self.assertIs(trace.analysis, self.analysis)
        self.assertEqual(policy.base_decide_calls, 1)

    def test_subclass_overriding_the_inner_analysis_path_keeps_the_capability(
        self,
    ) -> None:
        policy = _InnerPathOverrideSubPolicy(
            self.base_action, self.own_action, self.analysis
        )
        recorder = DecisionTraceRecorder()

        traced = execute_policy_with_trace(policy, self.decision, recorder)

        (trace,) = recorder.snapshot()
        self.assertIs(traced, self.own_action)
        self.assertIs(trace.analysis, self.analysis)
        self.assertIs(execute_policy(policy, self.decision), self.own_action)

    def test_subclass_explicitly_overriding_the_capability_keeps_it(self) -> None:
        policy = _ExplicitAnalysisOverrideSubPolicy(
            self.base_action, self.own_action, self.analysis
        )
        recorder = DecisionTraceRecorder()

        traced = execute_policy_with_trace(policy, self.decision, recorder)

        (trace,) = recorder.snapshot()
        self.assertIs(traced, self.own_action)
        self.assertIs(trace.analysis, self.analysis)
        self.assertEqual(policy.own_analysis_calls, 1)

    def test_instance_level_choose_action_override_also_bypasses_the_capability(
        self,
    ) -> None:
        policy = _AnalysisCapableBasePolicy(self.base_action, self.analysis)
        own_action = self.own_action
        policy.choose_action = lambda decision: own_action
        recorder = DecisionTraceRecorder()

        traced = execute_policy_with_trace(policy, self.decision, recorder)

        (trace,) = recorder.snapshot()
        self.assertIs(traced, own_action)
        self.assertIsNone(trace.analysis)
        self.assertEqual(policy.base_decide_calls, 0)


if __name__ == "__main__":
    unittest.main()
