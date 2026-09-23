"""Issue #189 offline diagnosticsとfrozen offline gateのtest。"""

import tempfile
import unittest
from pathlib import Path

import candidate_fixtures as cf

from lisjong.learning import (
    OFFLINE_GATE,
    DiscardCandidateFeatures,
    LearnedCandidateOffensePolicy,
    SecondStepStatus,
    classify_offline_result,
    evaluate_candidate_policy,
    read_source_record,
)
from lisjong.learning.candidate_diagnostics import (
    INVALID,
    NOT_QUALIFIED,
    READY,
    SemanticAgreement,
)
from lisjong.policy_contract import DiscardAction, Seat


def _candidate(tile_spec, shanten, ukeire, status, score=None):
    return DiscardCandidateFeatures(
        action=DiscardAction(
            actor=Seat.SEAT_0, tile=cf.hand(tile_spec)[0], tsumogiri=False
        ),
        post_discard_shanten=shanten,
        current_ukeire_count=ukeire,
        second_step_ukeire_score=score,
        second_step_status=status,
    )


class ConstantRuntime:
    def score(self, context, candidates):
        return tuple(0.0 for _ in candidates)


class SemanticAgreementTests(unittest.TestCase):
    def test_upstream_misses_are_not_later_stage_successes(self) -> None:
        agreement = SemanticAgreement()
        teacher = _candidate("1m", 1, 20, SecondStepStatus.EVALUATED, 100)

        agreement.add(
            teacher, _candidate("2m", 2, 20, SecondStepStatus.NOT_MATERIALIZED)
        )
        agreement.add(
            teacher, _candidate("3m", 1, 18, SecondStepStatus.NOT_MATERIALIZED)
        )
        agreement.add(teacher, _candidate("4m", 1, 20, SecondStepStatus.EVALUATED, 90))
        agreement.add(teacher, _candidate("5m", 1, 20, SecondStepStatus.EVALUATED, 100))
        value = agreement.to_value()

        self.assertEqual(value["scorer_decisions"], 4)
        self.assertEqual(value["shanten_stage_agreement"], 3 / 4)
        self.assertEqual(value["conditional_current_ukeire_support"], 3)
        self.assertEqual(value["conditional_current_ukeire_agreement"], 2 / 3)
        self.assertEqual(value["conditional_second_step_support"], 2)
        self.assertEqual(value["conditional_second_step_agreement"], 1 / 2)

    def test_second_step_support_requires_an_evaluated_teacher(self) -> None:
        agreement = SemanticAgreement()
        teacher = _candidate("1m", 0, 8, SecondStepStatus.NOT_APPLICABLE)

        agreement.add(teacher, _candidate("2m", 0, 8, SecondStepStatus.NOT_APPLICABLE))
        value = agreement.to_value()

        self.assertEqual(value["conditional_second_step_support"], 0)
        self.assertNotIn("conditional_second_step_agreement", value)

    def test_evaluated_zero_scores_are_compared_as_values(self) -> None:
        agreement = SemanticAgreement()
        teacher = _candidate("1m", 1, 20, SecondStepStatus.EVALUATED, 0)

        agreement.add(teacher, _candidate("2m", 1, 20, SecondStepStatus.EVALUATED, 0))

        self.assertEqual(agreement.to_value()["conditional_second_step_agreement"], 1.0)


class OfflineEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = read_source_record(
            cf.write_candidate_source_record(Path(self.temp.name) / "source")
        )

    def test_serving_path_metrics_on_one_split(self) -> None:
        policy = LearnedCandidateOffensePolicy(ConstantRuntime())

        evaluation = evaluate_candidate_policy(policy, self.source, ["OFFLINE-EVAL"])
        metrics = evaluation["metrics"]
        support = evaluation["support"]

        self.assertEqual(support["decisions"], 7)
        self.assertEqual(support["scorer_decisions"], 3)
        self.assertEqual(support["win_decisions"], 2)
        self.assertEqual(support["riichi_decisions"], 1)
        self.assertEqual(support["response_decisions"], 1)
        for name in ("legality", "win_guard", "riichi_guard", "no_call_guard"):
            self.assertEqual(metrics[name], 1.0)
        self.assertEqual(evaluation["splits"], ["OFFLINE-EVAL"])
        self.assertEqual(evaluation["teacher_agreement_by_branch"]["win"], 1.0)

    def test_small_support_is_never_classified_as_ready(self) -> None:
        policy = LearnedCandidateOffensePolicy(ConstantRuntime())
        evaluation = evaluate_candidate_policy(policy, self.source, ["OFFLINE-EVAL"])

        classification = classify_offline_result(evaluation)

        self.assertEqual(classification["outcome"], INVALID)
        self.assertTrue(classification["insufficient_support"])


class ClassificationTests(unittest.TestCase):
    def _evaluation(self, **metric_overrides):
        metrics = {
            "legality": 1.0,
            "win_guard": 1.0,
            "riichi_guard": 1.0,
            "no_call_guard": 1.0,
            "shanten_stage_agreement": 0.995,
            "conditional_current_ukeire_agreement": 0.97,
            "conditional_second_step_agreement": 0.93,
            "candidate_top1_agreement": 0.4,
        }
        metrics.update(metric_overrides)
        support = {
            name: required for name, required in OFFLINE_GATE["minimum_support"].items()
        }
        return {"metrics": metrics, "support": support}

    def test_all_gates_met_is_ready(self) -> None:
        result = classify_offline_result(self._evaluation())
        self.assertEqual(result["outcome"], READY)
        self.assertEqual(result["failures"], [])

    def test_top1_agreement_is_diagnostic_only(self) -> None:
        result = classify_offline_result(self._evaluation(candidate_top1_agreement=0.0))
        self.assertEqual(result["outcome"], READY)

    def test_semantic_threshold_miss_is_not_qualified(self) -> None:
        result = classify_offline_result(
            self._evaluation(conditional_current_ukeire_agreement=0.94)
        )
        self.assertEqual(result["outcome"], NOT_QUALIFIED)
        self.assertEqual(
            [failure["metric"] for failure in result["failures"]],
            ["conditional_current_ukeire_agreement"],
        )

    def test_guard_below_one_is_not_qualified(self) -> None:
        result = classify_offline_result(self._evaluation(no_call_guard=0.999))
        self.assertEqual(result["outcome"], NOT_QUALIFIED)

    def test_gate_is_fingerprinted(self) -> None:
        result = classify_offline_result(self._evaluation())
        self.assertEqual(len(result["gate"]["fingerprint"]), 64)
        self.assertEqual(result["gate"]["thresholds"]["shanten_stage_agreement"], 0.99)


if __name__ == "__main__":
    unittest.main()
