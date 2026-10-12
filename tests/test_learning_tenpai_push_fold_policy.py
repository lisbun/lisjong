"""#289: 聴牌PUSH/FOLDの候補Policyと、表のfileの読み込みの境界を固定する。

ゲート条件1〜7・候補選択・待ちの値・比較式そのものは`test_learning_tenpai_push_fold`が
固定している。ここでは、候補Policyがそれらを使って行動を選ぶ部分と、構築時の検査を扱う。
役・点数の計算は差し替えたfakeで行い、native拡張に依存しない。
"""

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_learning_tenpai_push_fold import (
    CHAMPION,
    CLOSED_TENPAI,
    MODEL,
    OPEN_TENPAI,
    RIICHI,
    TWO_WAY_TENPAI,
    InformationBoundaryTest,
    _decision,
    _discard,
    _ron_legal,
    _scored,
    _with_own,
)
from test_placement_aware_speed_call_policy import (
    ALL_LAST_TOP,
    _input,
    _pon_meld,
    _tile,
)

import lisjong.learning.tenpai_push_fold_policy as policy_module
from lisjong.hand_evaluation import scoring
from lisjong.hand_evaluation.scoring import ScoringBackendUnavailableError
from lisjong.learning._canonical import canonical_json_text, seal
from lisjong.learning.riichi_wait_estimator import CLIP_EPSILON, FEATURE_SET
from lisjong.learning.riichi_wait_mawashi_policy import RiichiWaitModelError
from lisjong.learning.tenpai_push_fold import GateKind
from lisjong.learning.tenpai_push_fold_evaluation import tables_document
from lisjong.learning.tenpai_push_fold_policy import (
    TenpaiPushFoldAnalysis,
    TenpaiPushFoldPolicy,
)
from lisjong.learning.tenpai_push_fold_value import (
    SELECTED_TABLES_SHA256,
    BucketSpec,
    PushFoldTables,
    TenpaiPushFoldTablesError,
    TenpaiSideTable,
    load_tables,
)
from lisjong.policy_contract.action import TsumoAction
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat

# 押す側（1z）はp = 0.25、降りる側（7s）はp = 0。
PUSH_QUARTER = {"1z": 2048, "7s": 0}


def _tables(r_f: float = -2000.0, **changes) -> PushFoldTables:
    """残りツモ山50（w2）・待ち8枚（c3）で、`V(押す) = −0.25 × 8000 = −2000`になる表。

    降りる側が聴牌を崩す判断では`V(降りる) = r_f`である。
    """
    side = TenpaiSideTable(
        q_ron={"w2.c3": 0.0, "w2.c2": 0.5},
        q_tsumo={"w2.c3": 0.0, "w2.c2": 0.5},
        r_t={"w2.c3": 0.0, "w2.c2": -100.0},
    )
    tables = PushFoldTables(
        buckets=BucketSpec(wall_upper_bounds=(20, 40), count_upper_bounds=(2, 4)),
        loss={"dealer": 8000.0, "non_dealer": 8000.0},
        tenpai={"discard": side, "riichi": side},
        r_f={"discard": {"w2": r_f}, "riichi": {"w2": r_f}},
        uplift={"ron": 0.0, "tsumo": 0.0},
    )
    return replace(tables, **changes)


def _policy(tables: PushFoldTables | None = None) -> TenpaiPushFoldPolicy:
    return TenpaiPushFoldPolicy(
        MODEL, _tables() if tables is None else tables, evaluate=_scored
    )


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.context = _input(CLOSED_TENPAI, riichi_discards="3z4z")
        self.decision = _decision(self.context)

    def test_equal_values_keep_c0_and_a_larger_fold_value_folds(self):
        with _ron_legal(PUSH_QUARTER):
            tie = _policy().choose_action_with_analysis(self.decision)
            fold = _policy(_tables(r_f=-1999.5)).choose_action_with_analysis(
                self.decision
            )
            worse = _policy(_tables(r_f=-2000.5)).choose_action(self.decision)
        self.assertEqual(tie.action, _discard("1z"))
        self.assertIsInstance(tie.analysis, TenpaiPushFoldAnalysis)
        self.assertEqual(
            (tie.analysis.push.value, tie.analysis.fold.value), (-2000, -2000)
        )
        self.assertFalse(tie.analysis.fold_selected)
        self.assertEqual(fold.action, _discard("7s"))
        self.assertTrue(fold.analysis.fold_selected)
        self.assertEqual(worse, _discard("1z"))

    def test_analysis_holds_the_gate_and_the_breakdown_of_both_candidates(self):
        with _ron_legal(PUSH_QUARTER):
            analysis = (
                _policy(_tables(r_f=-500.0))
                .choose_action_with_analysis(self.decision)
                .analysis
            )
        self.assertIs(analysis.gate.kind, GateKind.CLOSED_DISCARD)
        self.assertEqual(analysis.gate.push_action, _discard("1z"))
        self.assertEqual(analysis.gate.fold_action, _discard("7s"))
        push, fold = analysis.push, analysis.fold
        self.assertEqual((push.ron_legal_probability, push.deal_in_loss), (0.25, 8000))
        self.assertEqual((push.continuation, push.value), (0.0, -2000.0))
        self.assertEqual((fold.ron_legal_probability, fold.continuation), (0.0, -500.0))
        self.assertEqual(fold.value, -500.0)

    def test_riichi_decision_declares_unless_the_fold_value_is_larger(self):
        decision = _decision(self.context, riichi=True)
        with _ron_legal(PUSH_QUARTER):
            tie = _policy().choose_action_with_analysis(decision)
            fold = _policy(_tables(r_f=-1999.5)).choose_action_with_analysis(decision)
        self.assertEqual(tie.action, RIICHI)
        self.assertIs(tie.analysis.gate.kind, GateKind.RIICHI)
        self.assertEqual(fold.action, _discard("7s"))

    def test_open_hand_decision_is_compared_with_the_discard_tables(self):
        context = _input(
            OPEN_TENPAI, own_melds=(_pon_meld("7z"),), riichi_discards="3z4z"
        )
        tables = _tables(r_f=-1999.5)
        tables = replace(tables, r_f={**tables.r_f, "riichi": {}})
        with _ron_legal({"1z": 2048, "3m": 0}):
            result = _policy(tables).choose_action_with_analysis(_decision(context))
        self.assertIs(result.analysis.gate.kind, GateKind.OPEN_DISCARD)
        self.assertEqual(result.action, _discard("3m"))

    def test_fold_that_keeps_tenpai_uses_the_tenpai_side(self):
        context = _input(TWO_WAY_TENPAI, riichi_discards="3z4z")
        decision = _decision(context)
        c0 = CHAMPION.choose_action(decision)
        other = "6z" if c0 == _discard("5z") else "5z"
        pushed = "5z" if other == "6z" else "6z"
        # 降りる側は単騎3枚（c2）: C = 0.5 × 2000 + 0.5 × 2700 − 100。R_F（−9999）は使わない。
        with _ron_legal({pushed: 2048, other: 0}):
            result = _policy(_tables(r_f=-9999.0)).choose_action_with_analysis(decision)
        self.assertTrue(result.analysis.gate.fold_keeps_tenpai)
        self.assertEqual(result.analysis.fold.continuation, 2250.0)
        self.assertEqual(result.action, _discard(other))


class ChampionIdentityTest(unittest.TestCase):
    def setUp(self):
        self.context = _input(CLOSED_TENPAI, riichi_discards="3z4z")
        self.decision = _decision(self.context)

    def _assert_champion(self, policy, decision):
        self.assertEqual(
            policy.choose_action_with_analysis(decision),
            CHAMPION.choose_action_with_analysis(decision),
        )

    def test_decisions_outside_the_gate_return_the_champion_decision(self):
        decisions = {
            "1 self in riichi": _decision(
                _with_own(self.context, riichi=RiichiState.DECLARED)
            ),
            "2 no riichi": _decision(_input(CLOSED_TENPAI)),
            "3 all last": _decision(
                _input(CLOSED_TENPAI, riichi_discards="3z4z", **ALL_LAST_TOP)
            ),
            "4 one tile type": _decision(self.context, discards=(_discard("1z"),)),
            "5 not tenpai": _decision(
                _input("34m678p68p234s55s9s1m", riichi_discards="3z4z")
            ),
        }
        # 降りる方が大きく有利な表でも、ゲートの外では推定器を呼ばずにChampionの決定を返す。
        policy = _policy(_tables(r_f=0.0))
        with _ron_legal(PUSH_QUARTER) as estimator:
            for name, decision in decisions.items():
                with self.subTest(case=name):
                    self._assert_champion(policy, decision)
            estimator.assert_not_called()

    def test_a_legal_win_is_still_taken(self):
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=_tile("1z"))
        decision = replace(
            self.decision, legal_actions=(tsumo, *self.decision.legal_actions)
        )
        policy = _policy(_tables(r_f=0.0))
        with _ron_legal(PUSH_QUARTER):
            self.assertEqual(policy.choose_action(decision), tsumo)
            self._assert_champion(policy, decision)

    def test_no_safer_tile_returns_the_champion_decision(self):
        policy = _policy(_tables(r_f=0.0))
        for raw in ({"1z": 0}, {"1z": 10, "7s": 10}):
            with self.subTest(raw=raw), _ron_legal(raw):
                self._assert_champion(policy, self.decision)

    def test_a_missing_bucket_returns_the_champion_decision(self):
        # 条件8: 表にこの判断の区分・bucketがなければ、降りる方が有利な値でもC0。
        cases = {
            "r_f": {"r_f": {"discard": {}, "riichi": {}}},
            "loss": {"loss": {}},
            "tenpai": {
                "tenpai": {
                    "discard": TenpaiSideTable(),
                    "riichi": TenpaiSideTable(),
                }
            },
        }
        for name, changes in cases.items():
            with self.subTest(case=name), _ron_legal(PUSH_QUARTER):
                self._assert_champion(
                    _policy(replace(_tables(r_f=0.0), **changes)), self.decision
                )

    def test_a_group_without_tables_stays_with_the_champion(self):
        tables = _tables(r_f=0.0)
        tables = replace(tables, r_f={**tables.r_f, "riichi": {}})
        policy = _policy(tables)
        with _ron_legal(PUSH_QUARTER):
            self._assert_champion(policy, _decision(self.context, riichi=True))
            self.assertEqual(policy.choose_action(self.decision), _discard("7s"))


class ConstructionTest(unittest.TestCase):
    def test_the_policy_keeps_no_state_across_decisions(self):
        policy = _policy(_tables(r_f=-1999.5))
        before = dict(vars(policy))
        context = _input(CLOSED_TENPAI, riichi_discards="3z4z")
        decision = _decision(context)
        with _ron_legal(PUSH_QUARTER):
            first = policy.choose_action_with_analysis(decision)
            policy.choose_action_with_analysis(_decision(context, riichi=True))
            policy.choose_action_with_analysis(_decision(_input(CLOSED_TENPAI)))
            again = policy.choose_action_with_analysis(decision)
        self.assertEqual(first, again)
        self.assertEqual(vars(policy), before)

    def test_construction_fails_without_the_scoring_backend(self):
        with (
            patch.object(
                scoring,
                "_load_native",
                side_effect=ScoringBackendUnavailableError("missing"),
            ),
            self.assertRaises(ScoringBackendUnavailableError),
        ):
            TenpaiPushFoldPolicy(MODEL, _tables())

    def test_arguments_are_type_checked(self):
        with self.assertRaises(TypeError):
            TenpaiPushFoldPolicy(object(), _tables(), evaluate=_scored)
        with self.assertRaises(TypeError):
            TenpaiPushFoldPolicy(MODEL, _tables().to_value(), evaluate=_scored)

    def test_from_files_checks_both_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory) / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "schema": "lisjong-riichi-wait-selection-v1",
                        "feature_set": FEATURE_SET,
                        "clip_epsilon": CLIP_EPSILON,
                        "models": {
                            "estimator_logistic": {
                                "feature_set": FEATURE_SET,
                                "weights": {"class_rank_1": 3.0},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            tables = _write_tables(Path(directory), _tables())
            digests = {
                "expected_selection_sha256": _sha256(selection),
                "expected_tables_sha256": _sha256(tables),
            }
            with self.assertRaises(RiichiWaitModelError):
                TenpaiPushFoldPolicy.from_files(selection, tables)
            with self.assertRaises(TenpaiPushFoldTablesError):
                TenpaiPushFoldPolicy.from_files(
                    selection,
                    tables,
                    expected_selection_sha256=digests["expected_selection_sha256"],
                )
            with patch.object(policy_module, "require_scoring_backend"):
                policy = TenpaiPushFoldPolicy.from_files(selection, tables, **digests)
            self.assertEqual(vars(policy)["_tables"], _tables())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_document(directory: Path, document: object) -> Path:
    path = directory / "tables.json"
    path.write_text(canonical_json_text(document), encoding="utf-8", newline="")
    return path


def _write_tables(directory: Path, tables: PushFoldTables) -> Path:
    source = {"decisions": {"bytes": 1, "rows": 1, "sha256": "d" * 64}}
    return _write_document(
        directory, tables_document(tables, minimum_support=20, source=source)
    )


class LoadTablesTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def _load(self, path: Path) -> PushFoldTables:
        return load_tables(path, expected_sha256=_sha256(path))

    def _body(self, **changes) -> dict[str, object]:
        document = tables_document(_tables(), minimum_support=20, source={})
        body = {key: value for key, value in document.items() if key != "identity"}
        return {**body, **changes}

    def _rejects(self, document: object, message: str):
        path = _write_document(self.directory, document)
        with self.assertRaisesRegex(TenpaiPushFoldTablesError, message):
            self._load(path)

    def test_round_trips_the_fitted_tables(self):
        path = _write_tables(self.directory, _tables())
        self.assertEqual(self._load(path), _tables())

    def test_the_default_hash_is_the_registered_tables_file(self):
        self.assertEqual(
            SELECTED_TABLES_SHA256,
            "9234480ce3bed2f0b6e976cb853de9597f515f0d39b3fcbc35b534dc6efe730c",
        )
        path = _write_tables(self.directory, _tables())
        with self.assertRaisesRegex(TenpaiPushFoldTablesError, "SHA-256"):
            load_tables(path)

    def test_identity_schema_and_canonical_form_are_checked(self):
        sealed = seal(self._body())
        self._rejects({**sealed, "minimum_support": 1}, "identity mismatch")
        self._rejects(self._body(), "missing its identity")
        self._rejects(seal(self._body(schema="other")), "unsupported schema")
        self._rejects(seal({**self._body(), "extra": 1}), "unexpected fields")
        path = self.directory / "tables.json"
        path.write_text(json.dumps(sealed), encoding="utf-8")
        with self.assertRaisesRegex(TenpaiPushFoldTablesError, "not canonical"):
            self._load(path)
        path.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(TenpaiPushFoldTablesError, "not strict JSON"):
            self._load(path)

    def test_table_shape_keys_and_values_are_checked(self):
        tables = _tables().to_value()

        def changed(*path, value):
            copy = json.loads(json.dumps(tables))
            target = copy
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return seal(self._body(tables=copy))

        cases = {
            "unexpected fields": changed("extra", value={}),
            "tables.buckets": changed("buckets", "wall_upper_bounds", value=[40, 20]),
            "tables.r_f has": changed("r_f", value={"discard": {}}),
            "unexpected key 'w3'": changed("r_f", "discard", value={"w3": 0.0}),
            "unexpected key 'w2.c0'": changed(
                "tenpai", "riichi", "q_ron", value={"w2.c0": 0.0}
            ),
            "must be a probability": changed(
                "tenpai", "riichi", "q_tsumo", value={"w2.c3": 1.5}
            ),
            "must be a number": changed("loss", value={"dealer": "8000"}),
            "tables.loss.dealer must be a number": changed(
                "loss", value={"dealer": True}
            ),
            "unexpected key 'other'": changed("uplift", value={"other": 0.0}),
        }
        for message, document in cases.items():
            with self.subTest(case=message):
                self._rejects(document, message)


class PolicyInformationBoundaryTest(unittest.TestCase):
    FORBIDDEN = InformationBoundaryTest.FORBIDDEN
    LOADED_FORBIDDEN = InformationBoundaryTest.LOADED_FORBIDDEN

    def test_the_policy_module_does_not_import_label_or_source_modules(self):
        tree = ast.parse(Path(policy_module.__file__).read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        for forbidden in self.FORBIDDEN:
            self.assertFalse(any(forbidden in name for name in imported), forbidden)

    def test_importing_the_policy_module_loads_no_label_or_source_module(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "import lisjong.learning.tenpai_push_fold_policy\n"
                f"forbidden = {self.LOADED_FORBIDDEN!r}\n"
                "loaded = [name for name in sys.modules "
                "if any(part in name for part in forbidden)]\n"
                "assert not loaded, loaded\n"
                "assert 'torch' not in sys.modules\n"
                "print('ok')\n",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
