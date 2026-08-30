"""Determinism tests for the Issue #141 offline artifact generator."""

import pathlib
import unittest

from lisjong.hand_evaluation import _structural_predicates
from tools import generate_structural_predicate_table


class StructuralPredicateGeneratorTest(unittest.TestCase):
    def test_two_generations_are_byte_identical_and_match_the_committed_artifact(
        self,
    ) -> None:
        first, _first_stats = generate_structural_predicate_table.build_artifact(
            progress=False
        )
        second, _second_stats = generate_structural_predicate_table.build_artifact(
            progress=False
        )
        committed = (
            pathlib.Path(_structural_predicates.__file__).parent
            / _structural_predicates.TABLE_RESOURCE
        ).read_bytes()

        self.assertEqual(first, second)
        self.assertEqual(first, committed)


if __name__ == "__main__":
    unittest.main()
