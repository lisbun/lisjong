"""Issue #228 development profile toolのsmoke test。

toolがwrapする関数・methodがrenameで消えていないこと、profile後に元へ戻る
こと、profileしてもPolicyの選択actionが変わらないことだけを固定する。
wall-clock値は検査しない。
"""

import argparse
import importlib
import importlib.util
import pathlib
import pickle
import tempfile
import unittest

from test_placement_aware_speed_call_policy import (
    PASS,
    TANYAO_ONE_SHANTEN,
    _chi,
    _chi_meld,
    _discards,
    _input,
)

from lisjong.policies import PlacementAwareSpeedCallPolicy
from lisjong.policy_contract.decision_context import DecisionContext

_TOOL_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tools"
    / "profile_speed_call_decisions.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "profile_speed_call_decisions", _TOOL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records() -> list:
    call_decision = DecisionContext(
        input=_input(TANYAO_ONE_SHANTEN),
        legal_actions=(_chi("2m", "34m"), PASS),
    )
    open_input = _input("23455s667789p", own_melds=(_chi_meld("234m"),), drawn="7p")
    discard_decision = DecisionContext(
        input=open_input, legal_actions=_discards(open_input)
    )
    policy = PlacementAwareSpeedCallPolicy()
    return [
        (decision, policy.choose_action(decision))
        for decision in (call_decision, discard_decision)
    ]


class ProfileSpeedCallDecisionsToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.decisions = str(pathlib.Path(directory.name) / "decisions.pickle")
        pathlib.Path(cls.decisions).write_bytes(pickle.dumps(_records()))

    def _current_targets(self) -> list[object]:
        targets = [
            vars(getattr(importlib.import_module(module), cls))[method]
            for module, cls, method, _ in self.tool._CLASS_TARGETS
        ]
        targets.extend(
            getattr(importlib.import_module(module), attribute)
            for module, attribute, _ in (
                *self.tool._FUNCTION_TARGETS,
                *self.tool._DP_COUNTER_TARGETS,
            )
        )
        return targets

    def test_scan_keeps_recorded_actions_and_restores_patches(self) -> None:
        before = self._current_targets()
        report = self.tool._run_scan(
            argparse.Namespace(
                decisions=self.decisions, top=5, export=None, export_min_ms=0.0
            )
        )
        self.assertEqual(report["action_mismatches"], 0)
        self.assertEqual(
            {row["kind"] for row in report["slowest"]}, {"call", "discard"}
        )
        self.assertEqual(self._current_targets(), before)

    def test_breakdown_wraps_every_target_and_restores_patches(self) -> None:
        before = self._current_targets()
        report = self.tool._run_breakdown(
            argparse.Namespace(decisions=self.decisions, indices=None, min_ms=0.0)
        )
        self.assertEqual(report["skipped_targets"], [])
        self.assertEqual(len(report["per_decision"]), 2)
        call_rows, discard_rows = (item["rows"] for item in report["per_decision"])
        self.assertIn("pasc.speed_call_candidates", call_rows)
        self.assertIn("pasc.eligible_discards", discard_rows)
        self.assertEqual(self._current_targets(), before)


if __name__ == "__main__":
    unittest.main()
