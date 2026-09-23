"""Issue #189 candidate numeric encodingとtwo-pass second-step request policyのtest。"""

import unittest
from unittest.mock import patch

import candidate_fixtures as cf

import lisjong.learning.candidate_encoding as candidate_encoding
import lisjong.learning.candidate_features as candidate_features
from lisjong.learning import (
    CANDIDATE_ENCODING_DIMENSION,
    CANDIDATE_ENCODING_IDENTITY,
    CANDIDATE_FEATURE_IDENTITY,
    FEATURE_IDENTITY,
    SECOND_STEP_REQUEST_POLICY,
    CandidateFeatureError,
    DiscardCandidateFeatures,
    SecondStepStatus,
    build_discard_candidate_features,
    build_scorer_candidates,
    candidate_encoding_fingerprint,
    encode_candidates,
)
from lisjong.learning.candidate_encoding import (
    CANDIDATE_BLOCK_OFFSETS,
    SECOND_STEP_STATUS_AXIS,
    candidate_encoding_specification,
    second_step_finalists,
)
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import DiscardAction, Seat

_STATUS_OFFSET = CANDIDATE_BLOCK_OFFSETS["second_step_status"]


def _candidate(tile_spec, shanten, ukeire, status, score=None, *, tsumogiri=False):
    return DiscardCandidateFeatures(
        action=DiscardAction(
            actor=Seat.SEAT_0, tile=cf.hand(tile_spec)[0], tsumogiri=tsumogiri
        ),
        post_discard_shanten=shanten,
        current_ukeire_count=ukeire,
        second_step_ukeire_score=score,
        second_step_status=status,
    )


def _status_bits(vector):
    return tuple(vector[_STATUS_OFFSET : _STATUS_OFFSET + len(SECOND_STEP_STATUS_AXIS)])


class EncodingContractTests(unittest.TestCase):
    def test_identity_is_a_new_purpose_specific_contract(self) -> None:
        self.assertEqual(
            CANDIDATE_ENCODING_IDENTITY,
            "lisjong-offense-l0.2-discard-candidate-encoding-v1",
        )
        self.assertEqual(
            SECOND_STEP_REQUEST_POLICY,
            "lisjong-offense-l0.2-two-pass-finalist-second-step-v1",
        )
        # #184 / #187 identityは変更しない
        self.assertEqual(FEATURE_IDENTITY, "lisjong-offense-l0-player-safe-feature-v1")
        self.assertEqual(
            CANDIDATE_FEATURE_IDENTITY,
            "lisjong-offense-l0.1-discard-candidate-feature-v1",
        )

    def test_specification_binds_layout_request_policy_and_feature_contract(
        self,
    ) -> None:
        specification = candidate_encoding_specification()

        self.assertEqual(specification["dimension"], CANDIDATE_ENCODING_DIMENSION)
        self.assertEqual(
            sum(block["size"] for block in specification["blocks"]),
            CANDIDATE_ENCODING_DIMENSION,
        )
        self.assertEqual(
            specification["second_step_request_policy"], SECOND_STEP_REQUEST_POLICY
        )
        self.assertEqual(
            specification["candidate_feature_identity"], CANDIDATE_FEATURE_IDENTITY
        )
        self.assertEqual(
            candidate_encoding_fingerprint(), candidate_encoding_fingerprint()
        )

    def test_every_vector_has_the_declared_dimension(self) -> None:
        for spec in (cf.FAR_HAND, cf.NEAR_HAND, cf.TENPAI_REACHABLE_HAND):
            vectors = encode_candidates(
                build_scorer_candidates(cf.discard_decision(spec))
            )
            self.assertTrue(
                all(len(vector) == CANDIDATE_ENCODING_DIMENSION for vector in vectors)
            )


class AvailabilityEncodingTests(unittest.TestCase):
    """`SecondStepStatus`3状態と評価済み0を混同しないことを固定する。"""

    def setUp(self) -> None:
        # 1向聴の同ukeire finalist 2つ（EVALUATED、一方はscore 0）と、
        # 2向聴のNOT_MATERIALIZED candidate。
        self.candidates = (
            _candidate("1m", 1, 10, SecondStepStatus.EVALUATED, 0),
            _candidate("2m", 1, 10, SecondStepStatus.EVALUATED, 40),
            _candidate("3m", 2, 30, SecondStepStatus.NOT_MATERIALIZED),
        )

    def test_three_states_encode_to_distinct_status_bits(self) -> None:
        evaluated_zero, evaluated, not_materialized = encode_candidates(self.candidates)
        tenpai = encode_candidates(
            (
                _candidate("1m", 0, 4, SecondStepStatus.NOT_APPLICABLE),
                _candidate("2m", 1, 20, SecondStepStatus.NOT_MATERIALIZED),
            )
        )[0]

        self.assertEqual(_status_bits(evaluated_zero), (1.0, 0.0, 0.0))
        self.assertEqual(_status_bits(evaluated), (1.0, 0.0, 0.0))
        self.assertEqual(_status_bits(not_materialized), (0.0, 1.0, 0.0))
        self.assertEqual(_status_bits(tenpai), (0.0, 0.0, 1.0))

    def test_evaluated_zero_is_not_confused_with_availability(self) -> None:
        evaluated_zero, _evaluated, not_materialized = encode_candidates(
            self.candidates
        )
        score = CANDIDATE_BLOCK_OFFSETS["second_step_score"]
        gap = CANDIDATE_BLOCK_OFFSETS["second_step_gap"]

        self.assertEqual(evaluated_zero[score], 0.0)
        self.assertEqual(not_materialized[score], 0.0)
        # 評価済み0は最大scoreとのgapを持ち、未評価はgapも0のまま。
        self.assertGreater(evaluated_zero[gap], 0.0)
        self.assertEqual(not_materialized[gap], 0.0)
        self.assertNotEqual(
            _status_bits(evaluated_zero), _status_bits(not_materialized)
        )

    def test_relative_blocks_use_only_the_same_decision(self) -> None:
        vectors = encode_candidates(self.candidates)
        shanten_gap = CANDIDATE_BLOCK_OFFSETS["shanten_gap"]
        ukeire_gap = CANDIDATE_BLOCK_OFFSETS["ukeire_gap_within_shanten"]

        self.assertEqual([vector[shanten_gap] for vector in vectors], [0.0, 0.0, 1 / 8])
        self.assertEqual([vector[ukeire_gap] for vector in vectors], [0.0, 0.0, 0.0])


class EncodingFailClosedTests(unittest.TestCase):
    def test_empty_candidates_fail_closed(self) -> None:
        with self.assertRaises(CandidateFeatureError):
            encode_candidates(())

    def test_non_canonical_order_fails_closed(self) -> None:
        candidates = (
            _candidate("2m", 2, 10, SecondStepStatus.NOT_MATERIALIZED),
            _candidate("1m", 1, 10, SecondStepStatus.NOT_MATERIALIZED),
        )
        with self.assertRaises(CandidateFeatureError):
            encode_candidates(candidates)

    def test_status_that_contradicts_the_request_policy_fails_closed(self) -> None:
        """finalistをNOT_MATERIALIZEDのまま渡すとencodeしない。"""
        candidates = (
            _candidate("1m", 1, 10, SecondStepStatus.NOT_MATERIALIZED),
            _candidate("2m", 1, 10, SecondStepStatus.NOT_MATERIALIZED),
        )
        with self.assertRaisesRegex(CandidateFeatureError, "second-step availability"):
            encode_candidates(candidates)

    def test_all_candidate_second_step_is_rejected(self) -> None:
        """finalist以外まで評価したmaterializationはこのencodingに入らない。"""
        candidates = (
            _candidate("1m", 1, 10, SecondStepStatus.EVALUATED, 5),
            _candidate("2m", 2, 30, SecondStepStatus.EVALUATED, 5),
        )
        with self.assertRaises(CandidateFeatureError):
            encode_candidates(candidates)

    def test_out_of_range_shanten_fails_closed(self) -> None:
        with self.assertRaises(CandidateFeatureError):
            encode_candidates(
                (_candidate("1m", 9, 10, SecondStepStatus.NOT_MATERIALIZED),)
            )


class TwoPassRequestPolicyTests(unittest.TestCase):
    def _spy(self):
        calls = []
        original = candidate_features.build_discard_candidate_features

        def spy(decision, *, second_step_actions=()):
            requested = tuple(second_step_actions)
            calls.append(requested)
            return original(decision, second_step_actions=requested)

        return calls, spy

    def _count_second_step(self):
        counter = {"calls": 0}
        original = candidate_features.second_step_ukeire_score

        def counting(*args, **kwargs):
            counter["calls"] += 1
            return original(*args, **kwargs)

        return counter, counting

    def test_pass_one_never_calls_the_second_step_evaluator(self) -> None:
        counter, counting = self._count_second_step()
        with patch.object(candidate_features, "second_step_ukeire_score", counting):
            build_discard_candidate_features(cf.discard_decision(cf.FAR_HAND))
        self.assertEqual(counter["calls"], 0)

    def test_pass_two_requests_only_the_finalists(self) -> None:
        decision = cf.discard_decision(cf.FAR_HAND)
        first_pass = build_discard_candidate_features(decision)
        finalists = second_step_finalists(first_pass)
        calls, spy = self._spy()
        counter, counting = self._count_second_step()

        with (
            patch.object(candidate_encoding, "build_discard_candidate_features", spy),
            patch.object(candidate_features, "second_step_ukeire_score", counting),
        ):
            candidates = build_scorer_candidates(decision)

        self.assertGreaterEqual(len(finalists), 2)
        self.assertEqual(calls, [(), finalists])
        self.assertEqual(counter["calls"], len(finalists))
        self.assertLess(len(finalists), len(candidates))
        evaluated = {
            item.action
            for item in candidates
            if item.second_step_status is SecondStepStatus.EVALUATED
        }
        self.assertEqual(evaluated, set(finalists))

    def test_no_second_pass_when_minimum_shanten_is_tenpai(self) -> None:
        calls, spy = self._spy()
        with patch.object(candidate_encoding, "build_discard_candidate_features", spy):
            candidates = build_scorer_candidates(
                cf.discard_decision(cf.TENPAI_REACHABLE_HAND)
            )

        self.assertEqual(calls, [()])
        self.assertNotIn(
            SecondStepStatus.EVALUATED,
            {item.second_step_status for item in candidates},
        )

    def test_no_second_pass_with_a_single_finalist(self) -> None:
        candidates = (
            _candidate("1m", 1, 12, SecondStepStatus.NOT_MATERIALIZED),
            _candidate("2m", 1, 10, SecondStepStatus.NOT_MATERIALIZED),
        )
        self.assertEqual(second_step_finalists(candidates), ())

    def test_finalists_match_the_two_step_staged_semantics(self) -> None:
        """#87 TwoStepが2段目を評価したcandidate集合と一致する（再定義しない）。"""
        for spec in (cf.FAR_HAND, cf.NEAR_HAND, cf.TENPAI_REACHABLE_HAND):
            decision = cf.discard_decision(spec)
            analysis = (
                TwoStepUkeirePolicy().choose_action_with_analysis(decision).analysis
            )
            expected = {
                evaluation.action
                for evaluation in analysis.candidate_evaluations
                if evaluation.second_step_ukeire_score is not None
            }
            scores = {
                evaluation.action: evaluation.second_step_ukeire_score
                for evaluation in analysis.candidate_evaluations
            }
            candidates = build_scorer_candidates(decision)
            evaluated = {
                item.action: item.second_step_ukeire_score
                for item in candidates
                if item.second_step_status is SecondStepStatus.EVALUATED
            }
            with self.subTest(spec=spec):
                self.assertEqual(set(evaluated), expected)
                for action, score in evaluated.items():
                    self.assertEqual(score, scores[action])


if __name__ == "__main__":
    unittest.main()
