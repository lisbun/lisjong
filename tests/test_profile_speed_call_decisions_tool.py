"""Issue #228 development profile toolのsmoke test。

toolがwrapする関数・methodがrenameで消えていないこと、profile後に元へ戻る
こと、profileしてもPolicyの選択actionが変わらないことだけを固定する。
wall-clock値は検査しない。
"""

import argparse
import dataclasses
import importlib
import importlib.util
import json
import pathlib
import pickle
import tempfile
import unittest

from test_native_shanten_backend import _lisjong_native, _require_native, _run_python
from test_placement_aware_speed_call_policy import (
    PASS,
    TANYAO_ONE_SHANTEN,
    _chi,
    _chi_meld,
    _discards,
    _input,
)
from test_terminal_shanten_progression_mechanism_riichi_defense_policy import (
    _distinct_discard_actions,
    _restricted_input,
)
from test_terminal_shanten_progression_mechanism_riichi_defense_policy import (
    _hand as _progression_hand,
)

from lisjong.hand_evaluation import _shanten_backend
from lisjong.policies import PlacementAwareSpeedCallPolicy
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.own_hand_state import OwnHandState

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


def _r5_records() -> list:
    """#174 gateがR5を発動するclosed 14枚手のdecision（Issue #232）。"""
    tiles = tuple(_progression_hand("8p5m6m1z3m8s4z7m3s5z7z5p1m1s"))
    policy_input = _restricted_input(tiles, ("6s", "5z", "1s", "6p"))
    policy_input = dataclasses.replace(
        policy_input, own_hand=OwnHandState(concealed_tiles=tiles, drawn_tile=tiles[-1])
    )
    decision = DecisionContext(
        input=policy_input, legal_actions=_distinct_discard_actions(tiles)
    )
    return [(decision, PlacementAwareSpeedCallPolicy().choose_action(decision))]


_R5_COUNTERS = ("visited_states", "cache_hits", "cache_misses", "shanten_evaluations")


class ProfileSpeedCallDecisionsToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.decisions = str(pathlib.Path(directory.name) / "decisions.pickle")
        pathlib.Path(cls.decisions).write_bytes(pickle.dumps(_records()))
        cls.r5_decisions = str(pathlib.Path(directory.name) / "r5_decisions.pickle")
        pathlib.Path(cls.r5_decisions).write_bytes(pickle.dumps(_r5_records()))

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

    def test_scan_reports_measured_r5_counters(self) -> None:
        report = self.tool._run_scan(
            argparse.Namespace(
                decisions=self.r5_decisions, top=5, export=None, export_min_ms=0.0
            )
        )
        (row,) = report["slowest"]
        self.assertEqual(row["thr_stage"], "R5_HONOR_ONLY_SWITCH")
        r5 = row["dp"]["r5_progression"]
        self.assertEqual(r5["calls"], 1)
        for name in ("visited_states", "cache_hits", "cache_misses"):
            self.assertIsInstance(r5[name], int)
        self.assertGreater(r5["visited_states"], 0)
        self.assertGreater(r5["shanten_evaluations"], 0)
        native = _shanten_backend.BACKEND_NAME == _shanten_backend.RUST_BACKEND
        self.assertEqual(r5.get("evaluator"), "native" if native else None)

    def test_breakdown_reports_r5_counters_and_never_fakes_native_timing(
        self,
    ) -> None:
        report = self.tool._run_breakdown(
            argparse.Namespace(decisions=self.r5_decisions, indices=None, min_ms=0.0)
        )
        (entry,) = report["per_decision"]
        r5 = entry["r5"]
        for name in _R5_COUNTERS:
            self.assertIsInstance(r5["counters"][name], int)
        self.assertGreater(r5["counters"]["shanten_evaluations"], 0)
        dp_row = entry["rows"]["r5_progression_dp"]
        if _shanten_backend.BACKEND_NAME == _shanten_backend.RUST_BACKEND:
            self._assert_native_breakdown(r5, entry["rows"], dp_row)
        else:
            self.assertEqual(r5["backend"], "python")
            self.assertIsNotNone(dp_row["self_ms"])
            self.assertIn("shanten@r5_progression", entry["rows"])

    def _assert_native_breakdown(self, r5, rows, dp_row) -> None:
        self.assertEqual(r5["backend"], "rust-native")
        self.assertIsNone(r5["native_shanten_ms"])
        self.assertEqual(r5["native_shanten_status"], "not_measured")
        self.assertIn("native", r5["native_shanten_reason"])
        self.assertIsNone(r5["python_dp_overhead_ms"])
        self.assertIsNone(dp_row["self_ms"])
        self.assertTrue(dp_row["self_ms_status"].startswith("not_measured"))
        self.assertEqual(r5["r5_batch_ms"], dp_row["inclusive_ms"])
        # native内部のshanten評価はPython wrapを通らないので、R5用の行は出ない
        # （0回・0秒として出さない）。
        self.assertNotIn("shanten@r5_progression", rows)

    def test_native_breakdown_in_a_rust_selected_process(self) -> None:
        _require_native(self)
        result = _run_python(
            "import argparse, json, sys\n"
            "sys.path.insert(0, 'tests')\n"
            "from test_profile_speed_call_decisions_tool import _load_tool\n"
            f"arguments = argparse.Namespace(decisions={self.r5_decisions!r}, "
            "indices=None, min_ms=0.0, top=5, export=None, export_min_ms=0.0)\n"
            "tool = _load_tool()\n"
            "scan = tool._run_scan(arguments)\n"
            "breakdown = tool._run_breakdown(arguments)\n"
            "print(json.dumps({'scan': scan['slowest'][0]['dp'], "
            "'r5': breakdown['per_decision'][0]['r5']}))\n",
            backend="rust",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["scan"]["r5_progression"]["evaluator"], "native")
        self.assertEqual(report["r5"]["backend"], "rust-native")
        self.assertIsNone(report["r5"]["native_shanten_ms"])
        self.assertGreater(report["r5"]["counters"]["shanten_evaluations"], 0)
        self.assertIsNotNone(_lisjong_native)


if __name__ == "__main__":
    unittest.main()
