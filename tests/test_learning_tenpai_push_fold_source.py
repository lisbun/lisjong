"""#288: 聴牌PUSH/FOLDの対比較sourceのstrict reader・表のfit・validの報告・件数の確認を固定する。"""

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from test_learning_tenpai_push_fold import (
    CLOSED_TENPAI,
    MODEL,
    RIICHI,
    _decision,
    _discard,
    _no_yaku,
    _scored,
    _wait,
)
from test_placement_aware_speed_call_policy import _chi_meld, _input, _pon_meld

from lisjong.learning._canonical import canonical_json_line, file_digest, unseal
from lisjong.learning.riichi_wait_mawashi_policy import SELECTED_WAIT_MODEL_SHA256
from lisjong.learning.tenpai_push_fold import GATE, GateKind, TenpaiGate
from lisjong.learning.tenpai_push_fold_evaluation import (
    count_gate_decisions,
    fit_tables,
    gate_from_record,
    support_counts,
    tables_document,
    valid_report,
)
from lisjong.learning.tenpai_push_fold_source import (
    DECISIONS_FILENAME,
    FOLD,
    MANIFEST_FILENAME,
    OUTCOMES_FILENAME,
    PUSH,
    GateDecisionKey,
    GateDecisionRecord,
    PairRecord,
    RoundOutcome,
    TenpaiPushFoldSourceError,
    decision_to_value,
    manifest_text,
    outcome_to_value,
    read_decisions,
    read_source,
)
from lisjong.learning.tenpai_push_fold_value import BucketSpec
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.seat import Seat

CONTEXT = _input(CLOSED_TENPAI, riichi_discards="3z4z")
PRODUCER = {
    "arena_revision": "a" * 7,
    "lisjong_revision": "b" * 7,
    "lisjong_engine_revision": "c" * 7,
    "policy": "PlacementAwareSpeedCallPolicy",
    "wait_model_sha256": SELECTED_WAIT_MODEL_SHA256,
}


def _record(seed, sequence, ordinal, *, riichi=False, fold_raw=10, **changes):
    decision = _decision(CONTEXT, riichi=riichi)
    record = GateDecisionRecord(
        key=GateDecisionKey(seed=seed, sequence=sequence, seat=0),
        ordinal=ordinal,
        policy_input=decision.input,
        legal_actions=decision.legal_actions,
        kind=GateKind.RIICHI if riichi else GateKind.CLOSED_DISCARD,
        riichi_seat=Seat.SEAT_1,
        c0_action=RIICHI if riichi else _discard("1z"),
        push_action=_discard("1z"),
        fold_action=_discard("7s"),
        push_ron_legal_raw=20,
        fold_ron_legal_raw=fold_raw,
    )
    return replace(record, **changes)


def _outcome(record, side, round_delta=0, **changes) -> RoundOutcome:
    outcome = RoundOutcome(
        key=record.key,
        side=side,
        round_delta=round_delta,
        discard_passed=True,
        win_method=None,
        win_points=None,
        deal_in_to=None,
        deal_in_points=None,
        exhaustive_draw_tenpai=None,
        declaration_discard=(
            _discard("1z") if side == PUSH and record.kind is GateKind.RIICHI else None
        ),
    )
    return replace(outcome, **changes)


def _default_source():
    """seed 1（train）に(B)と(A)、seed 2（valid）に降りる候補のない(B)。"""
    first = _record(1, 10, 0)
    second = _record(1, 30, 1, riichi=True)
    third = _record(2, 5, 0, fold_raw=20)
    decisions = [first, second, third]
    outcomes = [
        _outcome(
            first,
            PUSH,
            -8000,
            discard_passed=False,
            deal_in_to=Seat.SEAT_1,
            deal_in_points=8000,
        ),
        _outcome(first, FOLD, -1000, exhaustive_draw_tenpai=False),
        _outcome(second, PUSH, 9000, win_method="tsumo", win_points=8000),
        _outcome(second, FOLD, 0),
        _outcome(third, PUSH, 1500, exhaustive_draw_tenpai=True),
    ]
    return (
        [decision_to_value(record) for record in decisions],
        [outcome_to_value(outcome) for outcome in outcomes],
    )


def _write(root: Path, decisions, outcomes, *, manifest_changes=None) -> None:
    files = {}
    for name, filename, rows in (
        ("decisions", DECISIONS_FILENAME, decisions),
        ("outcomes", OUTCOMES_FILENAME, outcomes),
    ):
        path = root / filename
        path.write_text(
            "".join(canonical_json_line(row) for row in rows),
            encoding="utf-8",
            newline="",
        )
        files[name] = {**file_digest(path), "rows": len(rows)}
    text = manifest_text(
        producer=PRODUCER, splits={"train": (1,), "valid": (2,)}, files=files
    )
    for old, new in (manifest_changes or {}).items():
        assert old in text
        text = text.replace(old, new)
    (root / MANIFEST_FILENAME).write_text(text, encoding="utf-8", newline="")


class SourceReaderTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.decisions, self.outcomes = _default_source()

    def _read(self, **manifest_changes):
        _write(
            self.root,
            self.decisions,
            self.outcomes,
            manifest_changes=manifest_changes or None,
        )
        return read_source(self.root)

    def _rejects(self, message: str, **manifest_changes):
        with self.assertRaisesRegex(TenpaiPushFoldSourceError, message):
            self._read(**manifest_changes)

    def test_reads_pairs_in_key_order(self):
        manifest, pairs = self._read()
        self.assertEqual(manifest.split_of(1), "train")
        self.assertEqual(
            [(pair.decision.key.seed, pair.decision.ordinal) for pair in pairs],
            [(1, 0), (1, 1), (2, 0)],
        )
        first, second, third = pairs
        self.assertEqual(first.push.deal_in_to, Seat.SEAT_1)
        self.assertEqual(first.fold.round_delta, -1000)
        self.assertEqual(second.push.declaration_discard, _discard("1z"))
        self.assertEqual(second.decision.decision, _decision(CONTEXT, riichi=True))
        self.assertIsNone(third.fold)
        _, decisions = read_decisions(self.root)
        self.assertEqual(decisions, tuple(pair.decision for pair in pairs))

    def test_read_decisions_does_not_open_the_outcomes(self):
        _write(self.root, self.decisions, self.outcomes)
        (self.root / OUTCOMES_FILENAME).unlink()
        self.assertEqual(len(read_decisions(self.root)[1]), 3)

    def test_manifest_schema_gate_and_model_are_checked(self):
        self._rejects("schema", **{"source-manifest-v1": "source-manifest-v2"})
        self._rejects("gate version", **{GATE: GATE + "x"})
        self._rejects("wait model", **{SELECTED_WAIT_MODEL_SHA256: "0" * 64})
        self._rejects("canonical", **{'"train"': '"train" '})

    def test_digest_and_row_count_are_checked(self):
        _write(self.root, self.decisions, self.outcomes)
        path = self.root / OUTCOMES_FILENAME
        path.write_text(
            path.read_text(encoding="utf-8").replace("-1000", "-2000"),
            encoding="utf-8",
            newline="",
        )
        with self.assertRaisesRegex(TenpaiPushFoldSourceError, "digest"):
            read_source(self.root)
        self._rejects("row count", **{'"rows": 5': '"rows": 4'})

    def test_record_schema_and_fields_are_checked(self):
        for rows in (self.decisions, self.outcomes):
            original = copy.deepcopy(rows[0])
            rows[0]["schema"] = "other"
            self._rejects("schema")
            rows[0] = copy.deepcopy(original)
            rows[0]["extra"] = 1
            self._rejects("unexpected fields")
            rows[0] = copy.deepcopy(original)
            del rows[0]["key"]
            self._rejects("unexpected fields")
            rows[0] = original

    def test_duplicate_and_missing_keys_are_rejected(self):
        self.decisions.append(copy.deepcopy(self.decisions[0]))
        self._rejects("duplicate decision key")
        self.decisions.pop()

        self.outcomes.append(copy.deepcopy(self.outcomes[0]))
        self._rejects("duplicate push outcome")
        self.outcomes.pop()

        orphan = copy.deepcopy(self.outcomes[0])
        orphan["key"]["sequence"] = 999
        self.outcomes.append(orphan)
        self._rejects("has no decision")
        self.outcomes.pop()

        removed = self.outcomes.pop(0)
        self._rejects("no push outcome")
        self.outcomes.insert(0, removed)

    def test_push_and_fold_sides_must_match_the_fold_candidate(self):
        removed = self.outcomes.pop(1)
        self._rejects("fold outcome does not match")
        self.outcomes.insert(1, removed)

        extra = copy.deepcopy(self.outcomes[4])
        extra["side"] = FOLD
        self.outcomes.append(extra)
        self._rejects("fold outcome does not match")

    def test_declaration_discard_belongs_to_riichi_push_sides(self):
        declaration = self.outcomes[2]["declaration_discard"]
        self.outcomes[2]["declaration_discard"] = None
        self._rejects("riichi push sides only")
        self.outcomes[2]["declaration_discard"] = declaration
        self.outcomes[0]["declaration_discard"] = declaration
        self._rejects("riichi push sides only")
        self.outcomes[0]["declaration_discard"] = None
        self.outcomes[3]["declaration_discard"] = declaration
        self._rejects("fold side has no declaration")

    def test_decision_consistency_is_checked(self):
        cases = {
            "key.seat": lambda row: row["key"].update(seat=2),
            "kind does not match": lambda row: row.update(kind="open_discard"),
            "differs from the Champion discard": lambda row: row.update(
                push_action=row["fold_action"]
            ),
            "not a single riichi opponent": lambda row: row.update(riichi_seat=2),
            "is not a legal action": lambda row: row["legal_actions"].remove(
                row["fold_action"]
            ),
            "exceeds the probability scale": lambda row: row.update(
                push_ron_legal_raw=8193
            ),
            "ordinals": lambda row: row.update(ordinal=1),
        }
        for message, mutate in cases.items():
            with self.subTest(case=message):
                original = copy.deepcopy(self.decisions[0])
                mutate(self.decisions[0])
                try:
                    self._rejects(message)
                finally:
                    self.decisions[0] = original

    def test_outcome_consistency_is_checked(self):
        cases = {
            "side must be": lambda row: row.update(side="other"),
            "more than one way": lambda row: row.update(
                win={"method": "ron", "points": 1000}
            ),
            "no deal-in": lambda row: row.update(deal_in=None),
            "is the decider": lambda row: row["deal_in"].update(to=0),
            "must be positive": lambda row: row["deal_in"].update(points=0),
            "must be an int": lambda row: row.update(round_delta=1.5),
        }
        for message, mutate in cases.items():
            with self.subTest(case=message):
                original = copy.deepcopy(self.outcomes[0])
                mutate(self.outcomes[0])
                try:
                    self._rejects(message)
                finally:
                    self.outcomes[0] = original

    def test_non_canonical_rows_are_rejected(self):
        _write(self.root, self.decisions, self.outcomes)
        path = self.root / OUTCOMES_FILENAME
        text = path.read_text(encoding="utf-8").replace(
            '"side":"push"', '"side": "push"', 1
        )
        path.write_text(text, encoding="utf-8", newline="")
        files = {
            "decisions": {
                **file_digest(self.root / DECISIONS_FILENAME),
                "rows": 3,
            },
            "outcomes": {**file_digest(path), "rows": 5},
        }
        (self.root / MANIFEST_FILENAME).write_text(
            manifest_text(
                producer=PRODUCER, splits={"train": (1,), "valid": (2,)}, files=files
            ),
            encoding="utf-8",
            newline="",
        )
        with self.assertRaisesRegex(TenpaiPushFoldSourceError, "canonical"):
            read_source(self.root)


BUCKETS = BucketSpec(wall_upper_bounds=(20, 40), count_upper_bounds=(2, 4))


def _pair(seed, sequence, *, riichi=False, push, fold=None, gate_changes=None):
    """（対、待ちの値を固定した`TenpaiGate`）。残りツモ山は50（w2）、待ちは8枚（c3）。"""
    record = _record(seed, sequence, 0, riichi=riichi, fold_raw=10 if fold else 20)
    gate = TenpaiGate(
        kind=record.kind,
        riichi_seat=record.riichi_seat,
        c0_action=record.c0_action,
        push_action=record.push_action,
        fold_action=record.fold_action,
        push_ron_legal_raw=record.push_ron_legal_raw,
        fold_ron_legal_raw=record.fold_ron_legal_raw,
        fold_keeps_tenpai=False,
        push_wait=_wait(8, 8, 2000.0, 2700.0),
        fold_wait=None,
    )
    pair = PairRecord(
        decision=record,
        push=_outcome(record, PUSH, **push),
        fold=None if fold is None else _outcome(record, FOLD, **fold),
    )
    return pair, replace(gate, **(gate_changes or {}))


_DEAL_IN = {"discard_passed": False, "deal_in_to": Seat.SEAT_1}


class FitTablesTest(unittest.TestCase):
    def test_tables_are_fitted_from_the_matching_side_and_group(self):
        pairs = [
            _pair(
                1,
                1,
                push={"round_delta": 2000, "win_method": "ron", "win_points": 2000},
                fold={"round_delta": -500},
            ),
            _pair(1, 2, push={"round_delta": -3000}, fold={"round_delta": -1500}),
            _pair(
                1, 3, push={"round_delta": -8000, **_DEAL_IN, "deal_in_points": 8000}
            ),
            # 降りる側が聴牌を保つ(B)(C)の局は、R_Fに入れない。
            _pair(
                1,
                4,
                push={"round_delta": 1000},
                fold={"round_delta": 9999},
                gate_changes={"fold_keeps_tenpai": True},
            ),
            _pair(
                1,
                5,
                riichi=True,
                push={"round_delta": 9000, "win_method": "tsumo", "win_points": 4700},
                fold={"round_delta": -100, **_DEAL_IN, "deal_in_points": 12000},
            ),
            _pair(
                1,
                6,
                riichi=True,
                push={"round_delta": -1000},
                fold={"round_delta": 300},
            ),
        ]
        tables, support = fit_tables(pairs, BUCKETS, minimum_support=1)
        # リーチ者（席1）は親。押す側・降りる側の、今の打牌での放銃の平均。
        self.assertEqual(tables.loss, {"dealer": 10000.0})
        discard = tables.tenpai["discard"]
        self.assertEqual(discard.q_ron, {"w2.c3": 1 / 3})
        self.assertEqual(discard.q_tsumo, {"w2.c3": 0.0})
        # 和了した局は0、それ以外は局収支（今の打牌が通った3局）。
        self.assertEqual(discard.r_t, {"w2.c3": (0 - 3000 + 1000) / 3})
        self.assertEqual(tables.r_f["discard"], {"w2": -1000.0})
        riichi = tables.tenpai["riichi"]
        self.assertEqual(riichi.q_tsumo, {"w2.c3": 0.5})
        self.assertEqual(riichi.r_t, {"w2.c3": -500.0})
        # (A)の降りる側は聴牌を保つかどうかによらず使うが、通らなかった局は除く。
        self.assertEqual(tables.r_f["riichi"], {"w2": 300.0})
        self.assertEqual(tables.uplift, {"tsumo": 2000.0})
        self.assertEqual(support["tenpai.discard.r_t"], {"w2.c3": 3})
        self.assertEqual(tables.to_value()["loss"], {"dealer": 10000.0})
        source = {"decisions": {"sha256": "d" * 64}}
        document = tables_document(tables, minimum_support=1, source=source)
        body = unseal(document, ValueError, "tables")
        self.assertEqual(body["tables"], tables.to_value())
        self.assertEqual((body["minimum_support"], body["source"]), (1, source))

    def test_buckets_below_the_minimum_support_are_left_out(self):
        pairs = [
            _pair(1, 1, push={"round_delta": 0}, fold={"round_delta": -500}),
            _pair(1, 2, push={"round_delta": 0}),
        ]
        tables, support = fit_tables(pairs, BUCKETS, minimum_support=2)
        self.assertEqual(tables.tenpai["discard"].r_t, {"w2.c3": 0.0})
        self.assertEqual(tables.r_f["discard"], {})
        self.assertEqual(support["r_f.discard"], {"w2": 1})


class ValidReportTest(unittest.TestCase):
    def test_reports_differences_by_group_and_v_decision(self):
        train = [
            _pair(1, 1, push={"round_delta": -4000}, fold={"round_delta": -1000}),
            _pair(
                1, 2, push={"round_delta": -8000, **_DEAL_IN, "deal_in_points": 8000}
            ),
        ]
        tables, _ = fit_tables(train, BUCKETS, minimum_support=1)
        valid = [
            _pair(2, 1, push={"round_delta": -3000}, fold={"round_delta": -1000}),
            _pair(
                3,
                1,
                push={"round_delta": -8000, **_DEAL_IN, "deal_in_points": 8000},
                fold={"round_delta": 0},
            ),
            _pair(3, 2, push={"round_delta": 500}),
            _pair(
                3, 3, riichi=True, push={"round_delta": 0}, fold={"round_delta": 100}
            ),
        ]
        report = valid_report(valid, tables)
        discard = report["round_delta_difference"]["discard"]
        self.assertEqual(discard["all"]["n"], 2)
        self.assertEqual(discard["all"]["games"], 2)
        self.assertEqual(discard["all"]["mean"], 5000.0)
        self.assertLessEqual(discard["all"]["low_2.5"], discard["all"]["high_97.5"])
        # V(押す) = -4000 * (1 - p) - p * 8000 < V(降りる) = -1000
        self.assertEqual(discard["v_fold"]["mean"], 5000.0)
        self.assertEqual(discard["yaku"]["n"], 2)
        self.assertEqual(discard["bucket.w2.c3"]["n"], 2)
        self.assertEqual(
            report["v_decisions"]["discard"],
            {"no_fold_candidate": 1, "v_fold": 2},
        )
        # (A)の表はtrainにないので、Vは計算できない。
        self.assertEqual(report["v_decisions"]["riichi"], {"no_table": 1})
        self.assertEqual(
            report["round_delta_difference"]["riichi"]["all"]["mean"], 100.0
        )
        self.assertEqual(report["riichi_declaration_prediction"], {"match": 1})
        self.assertEqual(report["deal_in_points"], {"other": {"n": 1, "mean": 8000.0}})
        self.assertEqual(sum(row["n"] for row in report["calibration"]), 7)
        deal_ins = sum(row["n"] * row["deal_in_rate"] for row in report["calibration"])
        self.assertAlmostEqual(deal_ins, 1.0)

    def test_dora_discards_are_reported_separately(self):
        context = _input(CLOSED_TENPAI, riichi_discards="3z4z", dora_indicators="4z")
        pair, gate = _pair(
            2,
            1,
            push={"round_delta": -8000, **_DEAL_IN, "deal_in_points": 12000},
            fold={"round_delta": 0},
        )
        pair = replace(pair, decision=replace(pair.decision, policy_input=context))
        tables, _ = fit_tables([], BUCKETS, minimum_support=1)
        report = valid_report([(pair, gate)], tables)
        self.assertEqual(
            report["round_delta_difference"]["discard"]["push_dora_or_red"]["n"], 1
        )
        self.assertEqual(
            report["deal_in_points"], {"dora_or_red": {"n": 1, "mean": 12000.0}}
        )
        self.assertEqual(report["v_decisions"]["discard"], {"no_table": 1})


class GateFromRecordAndCountTest(unittest.TestCase):
    def test_gate_from_record_recomputes_the_wait_values(self):
        record = _record(1, 1, 0)
        pair = PairRecord(decision=record, push=_outcome(record, PUSH), fold=None)
        gate = gate_from_record(pair, evaluate=_scored)
        self.assertEqual(gate.push_wait, _wait(8, 8, 2000.0, 2700.0))
        self.assertFalse(gate.fold_keeps_tenpai)
        self.assertIsNone(gate.fold_wait)
        keeps = replace(record, fold_action=_discard("1z"))
        gate = gate_from_record(replace(pair, decision=keeps), evaluate=_scored)
        self.assertTrue(gate.fold_keeps_tenpai)
        self.assertEqual(gate.fold_wait, gate.push_wait)
        riichi = _record(1, 1, 0, riichi=True, fold_action=_discard("1z"))
        gate = gate_from_record(replace(pair, decision=riichi), evaluate=_scored)
        self.assertIsNone(gate.fold_wait)

    def test_support_counts_use_the_decision_records_only(self):
        records = [
            _record(1, 1, 0),
            _record(1, 2, 1, fold_raw=20),
            _record(1, 3, 2, fold_action=_discard("1z")),
            _record(1, 4, 3, riichi=True, fold_action=_discard("1z")),
        ]
        counts = support_counts(records, evaluate=_scored)
        self.assertEqual(counts["decisions"], {"discard": 3, "riichi": 1})
        self.assertEqual(counts["fold_candidates"], {"discard": 2, "riichi": 1})
        self.assertEqual(counts["riichi_seat"], {"dealer": 4})
        self.assertEqual(counts["wall.discard"], {"50": 3})
        self.assertEqual(counts["tsumo_count.discard"], {"8": 3})
        # (B)(C)は聴牌を崩す降りる候補だけ、(A)は降りる候補のすべてがR_Fの材料になる。
        self.assertEqual(counts["r_f_wall.discard"], {"50": 1})
        self.assertEqual(counts["r_f_wall.riichi"], {"50": 1})
        self.assertFalse(any(name.startswith("bucket.") for name in counts))

        counts = support_counts(records, BUCKETS, evaluate=_scored)
        self.assertEqual(counts["bucket.tenpai.discard.r_t"], {"w2.c3": 3})
        self.assertEqual(counts["bucket.tenpai.riichi.q_ron"], {"w2.c3": 1})
        self.assertEqual(counts["bucket.r_f.discard"], {"w2": 1})

    def test_counts_gate_decisions_by_kind_and_yaku(self):
        yakuless = _input(
            "456p23s55s1z",
            own_melds=(_chi_meld("123m"), _pon_meld("9p")),
            riichi_discards="3z4z",
        )
        decisions = [
            _decision(CONTEXT),
            _decision(yakuless),
            _decision(_input(CLOSED_TENPAI)),
        ]

        def evaluate(hand, context):
            return _no_yaku(hand, context) if hand.melds else _scored(hand, context)

        result = count_gate_decisions(
            (
                (decision, PolicyDecision(action=_discard("1z")))
                for decision in decisions
            ),
            MODEL,
            evaluate=evaluate,
        )
        self.assertEqual(result["decisions"], 3)
        self.assertEqual(result["gate_decisions"], 2)
        self.assertEqual(set(result["by_kind"]), {"closed_discard", "open_discard"})
        self.assertEqual(set(result["by_kind"]["closed_discard"]), {"yaku"})
        self.assertEqual(set(result["by_kind"]["open_discard"]), {"no_yaku"})


if __name__ == "__main__":
    unittest.main()
