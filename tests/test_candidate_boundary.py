"""候補Policyの情報境界チェック（lisbun/lisjong#296）のregression test。

既存のPolicy（現Champion `PlacementAwareSpeedCallPolicy`）が合格し、意図的に
違反させたfixture Policy（`tests/candidate_boundary_fixtures/`）が不合格に
なることを、import制限と純粋性の両方で固定する。
"""

import unittest

import candidate_fixtures

from lisjong.candidate_boundary import (
    BoundaryViolation,
    check_candidate_imports,
    check_candidate_purity,
)

_CHAMPION_MODULE = "lisjong.policies.placement_aware_speed_call"
_CHAMPION_FACTORY = f"{_CHAMPION_MODULE}:PlacementAwareSpeedCallPolicy"
_FIXTURES = "candidate_boundary_fixtures"


def _rules(violations: tuple[BoundaryViolation, ...]) -> set[str]:
    return {violation.rule for violation in violations}


class ImportBoundaryTests(unittest.TestCase):
    def test_champion_module_passes(self) -> None:
        report = check_candidate_imports(_CHAMPION_MODULE)

        self.assertTrue(report.passed, msg=report.violations)
        self.assertEqual(len(report.files), 1)

    def test_package_with_relative_imports_passes(self) -> None:
        report = check_candidate_imports(f"{_FIXTURES}.clean_package")

        self.assertTrue(report.passed, msg=report.violations)
        self.assertEqual(len(report.files), 3)

    def test_direct_engine_import_is_rejected(self) -> None:
        report = check_candidate_imports(f"{_FIXTURES}.engine_import")

        self.assertFalse(report.passed)
        self.assertIn(
            ("forbidden-import", "lisjong_engine"),
            {(v.rule, v.detail) for v in report.violations},
        )
        self.assertIn("runtime-import", _rules(report.violations))

    def test_frame_access_is_rejected(self) -> None:
        report = check_candidate_imports(f"{_FIXTURES}.frame_peeking")

        self.assertFalse(report.passed)
        self.assertEqual(
            _rules(report.violations), {"import-not-allowlisted", "forbidden-attribute"}
        )
        self.assertIn("_getframe", {v.detail for v in report.violations})

    def test_ground_truth_reexport_is_rejected_by_origin(self) -> None:
        report = check_candidate_imports(f"{_FIXTURES}.ground_truth_reexport")

        self.assertFalse(report.passed)
        (violation,) = report.violations
        self.assertEqual(violation.rule, "forbidden-origin")
        self.assertIn("lisjong.belief.exact_wait_ground_truth", violation.detail)
        self.assertEqual(violation.line, 3)

    def test_ground_truth_attribute_through_module_alias_is_rejected(self) -> None:
        report = check_candidate_imports(f"{_FIXTURES}.ground_truth_attribute")

        self.assertFalse(report.passed)
        (violation,) = report.violations
        self.assertEqual(violation.rule, "forbidden-origin")
        self.assertIn("lisjong.belief.exact_wait_ground_truth", violation.detail)
        self.assertEqual(violation.line, 9)

    def test_dynamic_attribute_lookup_is_rejected(self) -> None:
        report = check_candidate_imports(f"{_FIXTURES}.dynamic_lookup")

        self.assertFalse(report.passed)
        self.assertEqual(_rules(report.violations), {"forbidden-builtin"})

    def test_transitive_engine_import_is_rejected_at_runtime(self) -> None:
        report = check_candidate_imports(f"{_FIXTURES}.transitive_engine")

        self.assertFalse(report.passed)
        self.assertIn("import-not-allowlisted", _rules(report.violations))
        (runtime,) = [v for v in report.violations if v.rule == "runtime-import"]
        self.assertIn("lisjong_engine", runtime.detail)

    def test_module_without_python_source_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            check_candidate_imports(f"{_FIXTURES}.does_not_exist")


class PurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decisions = candidate_fixtures.mixed_decisions()

    def test_champion_is_pure(self) -> None:
        report = check_candidate_purity(_CHAMPION_FACTORY, self.decisions)

        self.assertTrue(report.passed, msg=(report.mismatches, report.errors))
        self.assertEqual(report.decision_count, len(self.decisions))

    def test_hash_order_dependence_is_rejected(self) -> None:
        report = check_candidate_purity(
            f"{_FIXTURES}.hash_order:HashOrderPolicy", self.decisions
        )

        self.assertFalse(report.passed)
        self.assertEqual(report.errors, ())
        self.assertTrue(report.mismatches)

    def test_hidden_call_count_state_is_rejected(self) -> None:
        report = check_candidate_purity(
            f"{_FIXTURES}.call_count:CallCountPolicy", self.decisions
        )

        self.assertFalse(report.passed)
        self.assertEqual(report.errors, ())
        self.assertTrue(report.mismatches)

    def test_forbidden_runtime_import_fails_closed(self) -> None:
        report = check_candidate_purity(
            f"{_FIXTURES}.engine_import:EngineImportPolicy", self.decisions
        )

        self.assertFalse(report.passed)
        self.assertEqual(len(report.errors), 2)
        self.assertIn("lisjong_engine", report.errors[0])

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_candidate_purity(_CHAMPION_FACTORY, ())
        with self.assertRaises(ValueError):
            check_candidate_purity(_CHAMPION_MODULE, self.decisions)
        with self.assertRaises(TypeError):
            check_candidate_purity(_CHAMPION_FACTORY, (object(),))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
