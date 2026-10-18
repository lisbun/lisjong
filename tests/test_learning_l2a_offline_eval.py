"""L2a OFFLINE-EVALの評価tool（Issue #317）のtest。"""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import candidate_fixtures as cf
import learning_fixtures as fixtures

from lisjong.learning.__main__ import main
from lisjong.learning._canonical import value_digest
from lisjong.learning._typed_values import policy_input_to_value_v2
from lisjong.learning.errors import LearnedPolicyError, LearningError
from lisjong.learning.features import FEATURE_DIMENSION
from lisjong.learning.l2a_artifact import write_l2a_artifact
from lisjong.learning.l2a_dataset import (
    iter_l2a_dataset_payload,
    publish_l2a_dataset,
    read_l2a_dataset,
)
from lisjong.learning.l2a_decision import (
    L2aDecisionKind,
    classify_l2a_decision,
    guard_action,
)
from lisjong.learning.l2a_encoding import (
    ENCODING_M0,
    ENCODING_M1,
    ENCODING_R1,
    R1_BLOCK_OFFSETS,
)
from lisjong.learning.l2a_model import L2aScorerConfig
from lisjong.learning.l2a_offline_eval import (
    collect_l2a_offline_tally,
    decision_strata,
    record_offline_evaluation,
    run_l2a_offline_eval,
)
from lisjong.learning.l2a_offline_gate import (
    ESTABLISHED,
    EVALUATED,
    HOLD,
    INVALID,
    L2A_OFFLINE_GATE,
    NOT_ESTABLISHED,
    PARTIAL,
    L2aOfflineTally,
    judge_l2a_offline_eval,
    l2a_offline_gate_fingerprint,
    metric_values,
)
from lisjong.learning.l2a_policy import L2aPolicy
from lisjong.learning.source_record import (
    POLICY_SOURCE_RECORD_SCHEMA_V2,
    open_source_record,
)
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import (
    AnkanAction,
    ChiAction,
    DecisionContext,
    DiscardAction,
    PassAction,
    RiichiState,
    Seat,
)
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.wind import Wind

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_KINDS = ("forced", "win", "riichi", "discard", "call", "pass_response")
_OBJECTIVE = "per-decision-legal-candidate-softmax-cross-entropy"


# --- 合成tally ------------------------------------------------------------------


def _game(
    *,
    discard=200,
    m1=170,
    m0=150,
    bts=120,
    c0_rows=80,
    c0_m1=60,
    c0_m0=30,
    call=30,
    chi=(12, 4, 4, 8, 4),
    pon=(12, 4, 4, 8, 4),
):
    """1半荘の数。chi / ponは(合法, 教師, R1のTP, R1のTN, R1の行動一致)。"""
    counts = {
        "discard.rows": discard,
        "discard.m1.match": m1,
        "discard.m0.match": m0,
        "discard.b_ts.match": bts,
        "call.rows": call,
    }
    for name, (legal, teacher, tp, tn, action_match) in (("chi", chi), ("pon", pon)):
        prefix = f"call.{name}"
        counts.update(
            {
                f"{prefix}.legal": legal,
                f"{prefix}.teacher": teacher,
                f"{prefix}.r1.true_positive": tp,
                f"{prefix}.r1.true_negative": tn,
                f"{prefix}.r1.action_match": action_match,
                f"{prefix}.b_pass.true_negative": legal - teacher,
            }
        )
    return {
        "all": counts,
        "efficiency:c0_specific": {
            "discard.rows": c0_rows,
            "discard.m1.match": c0_m1,
            "discard.m0.match": c0_m0,
        },
    }


def _correctness(**overrides):
    value = {
        "decisions": 1000,
        "guard": {kind: {"agree": 50, "rows": 50} for kind in _KINDS},
        "legal": 1000,
        "single_candidate": {"guard_agree": 3, "rows": 3},
        "teacher_rule_exceptions": 0,
        "train_serve": {name: {"match": 10, "rows": 10} for name in ("m1", "m0", "r1")},
    }
    value.update(overrides)
    return value


def _tally(games=None, *, count=120, correctness=None, **game):
    if games is None:
        games = [_game(**game) for _ in range(count)]
    return L2aOfflineTally(
        games=tuple(games),
        correctness=correctness or _correctness(),
        diagnostics={},
    )


def _verdicts(report):
    return {name: item["verdict"] for name, item in report["verdicts"].items()}


class GateTests(unittest.TestCase):
    def test_gate_values_are_the_preregistered_design(self) -> None:
        gate = L2A_OFFLINE_GATE
        self.assertEqual(gate["bootstrap"]["resamples"], 10000)
        self.assertEqual(gate["bootstrap"]["confidence"], 0.95)
        self.assertEqual(gate["bootstrap"]["unit"], "hanchan")
        support = {
            name: (spec["rows"], spec["hanchan"])
            for name, spec in gate["minimum_support"].items()
        }
        self.assertEqual(
            support,
            {
                "discard": (20000, 100),
                "discard_c0_specific": (2000, 80),
                "call": (2000, 80),
                "chi_legal": (1000, 80),
                "chi_selected": (300, 60),
                "pon_legal": (1000, 80),
                "pon_selected": (300, 60),
                "discard_all_last": (1000, 30),
            },
        )
        self.assertEqual(gate["guard_support"], {"riichi": 30, "win": 30})
        discard = gate["verdicts"]["discard"]
        self.assertEqual(
            (discard["primary"]["estimate"], discard["primary"]["threshold"]),
            ("point", 0.85),
        )
        self.assertEqual(
            [(item["estimate"], item["threshold"]) for item in discard["secondary"]],
            [("lower", 0.40), ("lower", 0.0)],
        )
        self.assertEqual(l2a_offline_gate_fingerprint(), value_digest(gate))

    def test_report_carries_the_gate_digest(self) -> None:
        report = judge_l2a_offline_eval(_tally())
        self.assertEqual(report["gate"]["fingerprint"], l2a_offline_gate_fingerprint())
        self.assertEqual(
            [item["action"] for item in report["non_claims"]],
            ["和了 / リーチ", "槓 / 九種九牌", "打牌 / CALL"],
        )

    def test_balanced_accuracy_of_always_pass_is_one_half(self) -> None:
        values = metric_values(
            {
                "call.rows": 10,
                "call.chi.legal": 10,
                "call.chi.teacher": 3,
                "call.chi.b_pass.true_negative": 7,
            }
        )
        self.assertEqual(values["b_pass.chi_balanced_accuracy"], 0.5)
        self.assertEqual(values["b_pass.chi_action_match"], 0.0)

    def test_unknown_quantity_or_stratum_fails_closed(self) -> None:
        for game in ({"all": {"discard.unknown": 1}}, {"other": {"discard.rows": 1}}):
            with self.subTest(game=game), self.assertRaises(LearningError):
                _tally([game])


class VerdictTests(unittest.TestCase):
    def test_established(self) -> None:
        report = judge_l2a_offline_eval(_tally())
        self.assertEqual(report["outcome"], EVALUATED)
        self.assertEqual(
            _verdicts(report),
            {"discard": ESTABLISHED, "chi": ESTABLISHED, "pon": ESTABLISHED},
        )
        self.assertEqual(
            report["conclusion"]["overall"], "L2a: C0の打牌・鳴き判断の蒸留が成立"
        )
        self.assertEqual(report["conclusion"]["call"], "鳴き判断の蒸留が成立")
        discard = report["metrics"]["all"]
        self.assertAlmostEqual(discard["m1.discard_match"]["point"], 0.85)
        self.assertAlmostEqual(discard["b_ts.discard_match"]["point"], 0.6)
        self.assertEqual(discard["b_pass.chi_balanced_accuracy"]["point"], 0.5)

    def test_not_established(self) -> None:
        report = judge_l2a_offline_eval(
            _tally(m1=160, pon=(12, 4, 1, 8, 1))  # 打牌0.80、PonのBA 0.625
        )
        self.assertEqual(
            _verdicts(report),
            {"discard": NOT_ESTABLISHED, "chi": ESTABLISHED, "pon": NOT_ESTABLISHED},
        )
        self.assertFalse(report["verdicts"]["discard"]["primary"]["met"])
        self.assertNotEqual(report["conclusion"]["call"], "鳴き判断の蒸留が成立")

    def test_partial(self) -> None:
        report = judge_l2a_offline_eval(
            # C0固有層0.25（区間下限 < 0.40）、Chiの行動一致0.75
            _tally(c0_m1=20, chi=(12, 4, 4, 8, 3))
        )
        self.assertEqual(
            _verdicts(report),
            {"discard": PARTIAL, "chi": PARTIAL, "pon": ESTABLISHED},
        )
        self.assertNotIn("note", report["verdicts"]["discard"])
        self.assertNotIn("成立", report["conclusion"]["overall"])

    def test_partial_when_only_m1_minus_m0_fails(self) -> None:
        report = judge_l2a_offline_eval(_tally(m0=170))
        discard = report["verdicts"]["discard"]
        self.assertEqual(discard["verdict"], PARTIAL)
        self.assertEqual(
            discard["note"], "C0の打牌は学べたが、encoding追加の効果は示せない"
        )
        self.assertEqual(discard["secondary"][1]["value"], 0.0)

    def test_hold_only_chi_when_chi_support_is_short(self) -> None:
        # 教師がChiを選んだ行は2 x 120 = 240 < 300。Ponは足りる。
        report = judge_l2a_offline_eval(_tally(chi=(12, 2, 2, 10, 2)))
        self.assertEqual(
            _verdicts(report),
            {"discard": ESTABLISHED, "chi": HOLD, "pon": ESTABLISHED},
        )
        self.assertEqual(report["verdicts"]["chi"]["held_support"], ["chi_selected"])
        self.assertEqual(
            report["conclusion"]["call"],
            f"Chiは{HOLD}、Ponは{ESTABLISHED}（CALL全体の成立とは呼ばない）",
        )
        self.assertNotIn("蒸留が成立", report["conclusion"]["overall"])

    def test_hold_only_pon_when_pon_hanchan_support_is_short(self) -> None:
        # Ponの教師行は8 x 50 = 400行あるが、半荘supportは50 < 60。
        games = [
            _game(pon=(12, 8, 8, 4, 8) if index < 50 else (12, 0, 0, 12, 0))
            for index in range(120)
        ]
        report = judge_l2a_offline_eval(_tally(games))
        self.assertEqual(
            _verdicts(report),
            {"discard": ESTABLISHED, "chi": ESTABLISHED, "pon": HOLD},
        )
        support = report["support"]["pon_selected"]
        self.assertEqual((support["rows"], support["hanchan"]), (400, 50))
        self.assertFalse(support["satisfied"])

    def test_hold_discard_when_hanchan_support_is_short(self) -> None:
        # 22,500行あるが半荘supportは90 < 100。
        report = judge_l2a_offline_eval(_tally(count=90, discard=250, m1=220))
        self.assertEqual(_verdicts(report)["discard"], HOLD)
        self.assertEqual(report["verdicts"]["discard"]["held_support"], ["discard"])

    def test_all_last_support_is_report_only(self) -> None:
        report = judge_l2a_offline_eval(_tally())
        self.assertFalse(report["support"]["discard_all_last"]["satisfied"])
        self.assertEqual(_verdicts(report)["discard"], ESTABLISHED)

    def test_short_guard_support_holds_every_action(self) -> None:
        guard = {kind: {"agree": 50, "rows": 50} for kind in _KINDS}
        guard["win"] = {"agree": 10, "rows": 10}
        report = judge_l2a_offline_eval(_tally(correctness=_correctness(guard=guard)))
        self.assertEqual(set(_verdicts(report).values()), {HOLD})
        self.assertEqual(report["verdicts"]["pon"]["held_support"], ["win_guard"])

    def test_stop_invalid(self) -> None:
        guard = {kind: {"agree": 50, "rows": 50} for kind in _KINDS}
        guard["call"] = {"agree": 49, "rows": 50}
        train_serve = {name: {"match": 10, "rows": 10} for name in ("m1", "m0")}
        train_serve["r1"] = {"match": 9, "rows": 10}
        cases = {
            "legality": _correctness(legal=999),
            "guard agreement (call)": _correctness(guard=guard),
            "train-serve agreement (r1)": _correctness(train_serve=train_serve),
            "teacher-rule exception": _correctness(teacher_rule_exceptions=1),
            "no decisions": _correctness(decisions=0, legal=0),
        }
        for expected, correctness in cases.items():
            with self.subTest(reason=expected):
                report = judge_l2a_offline_eval(_tally(correctness=correctness))
                self.assertEqual(report["outcome"], INVALID)
                self.assertEqual(set(_verdicts(report).values()), {INVALID})
                self.assertIn(expected, " ".join(report["stop_reasons"]))

    def test_stop_reason_without_tally_does_not_compute_metrics(self) -> None:
        report = judge_l2a_offline_eval(None, ["OFFLINE-EVAL was already evaluated"])
        self.assertEqual(report["outcome"], INVALID)
        self.assertNotIn("metrics", report)
        with self.assertRaises(LearningError):
            judge_l2a_offline_eval(None)


class BootstrapTests(unittest.TestCase):
    def test_interval_resamples_hanchan_not_rows(self) -> None:
        # 半荘内で一致が完全に相関する。行を独立に扱えば幅は約0.01になる。
        games = [_game(m1=200 if index % 2 else 0) for index in range(100)]
        first = judge_l2a_offline_eval(_tally(games))["metrics"]["all"]
        second = judge_l2a_offline_eval(_tally(games))["metrics"]["all"]
        metric = first["m1.discard_match"]
        self.assertAlmostEqual(metric["point"], 0.5)
        self.assertGreater(metric["upper"] - metric["lower"], 0.1)
        self.assertLess(metric["lower"], 0.5)
        self.assertGreater(metric["upper"], 0.5)
        self.assertEqual(metric["undefined_resamples"], 0)
        self.assertEqual(first, second)

    def test_m1_minus_m0_is_paired_by_hanchan(self) -> None:
        # 半荘ごとにM1はM0より常に10件多い。対で比べれば差は全resampleで0.05。
        games = [
            _game(m1=110 + 4 * index % 80, m0=100 + 4 * index % 80)
            for index in range(100)
        ]
        metric = judge_l2a_offline_eval(_tally(games))["metrics"]["all"][
            "m1_minus_m0.discard_match"
        ]
        self.assertAlmostEqual(metric["point"], 0.05)
        self.assertAlmostEqual(metric["lower"], 0.05)
        self.assertAlmostEqual(metric["upper"], 0.05)


# --- 合成source（#316のtestと同じshape） --------------------------------------------


def _tile(spec):
    return cf.hand(spec)[0]


def _chi(called, consumed):
    return ChiAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_3,
        called_tile=_tile(called),
        consumed_tiles=tuple(_tile(item) for item in consumed),
    )


def _chi_response():
    value = cf.policy_input("23m456p789s11223z", drawn=False)
    return DecisionContext(
        input=value,
        legal_actions=(
            PassAction(actor=value.self_seat),
            _chi("1m", ("2m", "3m")),
            _chi("4m", ("2m", "3m")),
        ),
    )


def _single_candidate_discard():
    value = cf.policy_input("1111m456p789s122z3z")
    return DecisionContext(
        input=value,
        legal_actions=(
            *cf.discard_actions(value, tsumogiri_only=True),
            AnkanAction(actor=value.self_seat, tiles=(_tile("1m"),) * 4),
        ),
    )


def _riichi_opponent_decision():
    value = cf.policy_input(cf.NEAR_HAND)
    players = list(value.players)
    players[1] = replace(
        players[1],
        discards=(fixtures.discard(_tile("9p"), 0), fixtures.discard(_tile("1z"), 1)),
        riichi=RiichiState.ACCEPTED,
        riichi_declaration_discard_order=1,
    )
    value = replace(value, players=tuple(players))
    return DecisionContext(input=value, legal_actions=cf.discard_actions(value))


def _teacher(decision):
    """教師規則に従うfixture教師。DISCARDはTwoStep、CALLは最後の候補（鳴き）。"""
    kind = classify_l2a_decision(decision)
    action = guard_action(decision, kind)
    if action is not None:
        return action
    if kind is L2aDecisionKind.DISCARD:
        return TwoStepUkeirePolicy().choose_action(decision)
    return ENCODING_R1.encode(decision).actions[-1]


def _decisions():
    return (
        cf.discard_decision(cf.SECOND_NEAR_HAND),
        cf.tsumo_decision(),
        cf.riichi_decision(),
        cf.response_decision(),
        _chi_response(),
        _single_candidate_discard(),
        cf.ron_response_decision(),
        _riichi_opponent_decision(),
    )


def _write_source(root, *, seed_base=100):
    games = []
    for game_ordinal, split in enumerate(("TRAIN", "SELECT", "OFFLINE-EVAL")):
        seed = seed_base * (game_ordinal + 1)
        rows = []
        for ordinal, decision in enumerate(_decisions()):
            value = decision.input
            row = fixtures.source_row(
                game_ordinal=game_ordinal,
                seed=seed,
                split=split,
                step_ordinal=ordinal,
                decision_ordinal=ordinal,
                policy_input_value=replace(
                    value,
                    players=tuple(
                        replace(player, riichi_declaration_discard_order=None)
                        for player in value.players
                    ),
                ),
                legal_actions=decision.legal_actions,
                selected_action=_teacher(decision),
            )
            row["policy_input"] = policy_input_to_value_v2(value, ValueError, "input")
            rows.append(row)
        games.append((split, seed, rows))
    return fixtures.write_policy_source_record(
        root, games, schema=POLICY_SOURCE_RECORD_SCHEMA_V2
    )


class _Scorer:
    def __init__(self, encoding, scorer=None) -> None:
        self.encoding = encoding
        self.scorer = scorer or (lambda vector: 0.0)

    def score(self, context, vectors):
        return tuple(self.scorer(vector) for vector in vectors)


class _Runtime:
    def __init__(self, discard, reaction) -> None:
        self.discard = discard
        self.reaction = reaction

    def scorer_for(self, kind):
        return self.discard if kind is L2aDecisionKind.DISCARD else self.reaction


def _policy(discard=ENCODING_M1, reaction=ENCODING_R1, scorer=None):
    return L2aPolicy(_Runtime(_Scorer(discard, scorer), _Scorer(reaction, scorer)))


def _prefer_call(vector):
    """R1ではChi / Ponを選ぶ（canonical順で最初）。打牌では最初の候補。"""
    if len(vector) != ENCODING_R1.dimension:
        return 0.0
    kind = R1_BLOCK_OFFSETS["kind"]
    return vector[kind + 1] + vector[kind + 2]


class _SourceCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source_path = _write_source(self.root / "source")
        self.source = open_source_record(self.source_path)
        self.datasets = {}
        for encoding in (ENCODING_M1, ENCODING_M0, ENCODING_R1):
            path = self.root / f"{encoding.name}-dataset"
            publish_l2a_dataset(self.source, path, encoding=encoding)
            self.datasets[encoding.name] = path

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _collect(self, proposed=None, **options):
        return collect_l2a_offline_tally(
            self.source,
            proposed=proposed or _policy(scorer=_prefer_call),
            baseline_m0=_policy(discard=ENCODING_M0),
            datasets=self.datasets,
            **options,
        )


class DatasetPayloadTests(_SourceCase):
    def test_offline_rows_and_payload_bytes(self) -> None:
        rows = list(
            iter_l2a_dataset_payload(self.datasets["m1"], splits=("OFFLINE-EVAL",))
        )
        self.assertEqual([item.row.split for item in rows], ["OFFLINE-EVAL"] * 2)
        dataset = read_l2a_dataset(self.datasets["m1"])
        width = ENCODING_M1.dimension
        for item in rows:
            index = dataset.rows.index(item.row)
            context = dataset.context[
                index * FEATURE_DIMENSION : (index + 1) * FEATURE_DIMENSION
            ]
            offset = item.row.candidate_offset * width
            candidates = dataset.candidates[
                offset : offset + item.row.candidate_count * width
            ]
            self.assertEqual(item.context, context.tobytes())
            self.assertEqual(item.candidates, candidates.tobytes())


class CollectTests(_SourceCase):
    def test_correctness_and_counts(self) -> None:
        tally = self._collect()
        correctness = tally.correctness
        self.assertEqual((correctness["decisions"], correctness["legal"]), (8, 8))
        self.assertTrue(
            all(item["agree"] == item["rows"] for item in correctness["guard"].values())
        )
        self.assertEqual(correctness["guard"]["win"]["rows"], 2)
        self.assertEqual(correctness["single_candidate"], {"guard_agree": 1, "rows": 1})
        self.assertEqual(
            correctness["train_serve"],
            {name: {"match": 2, "rows": 2} for name in ("m1", "m0", "r1")},
        )
        self.assertEqual(correctness["teacher_rule_exceptions"], 0)

        self.assertEqual(len(tally.games), 1)
        counts = tally.games[0]["all"]
        self.assertEqual((counts["discard.rows"], counts["call.rows"]), (2, 2))
        self.assertEqual(counts["discard.b_ts.match"], 2)
        self.assertEqual(tally.games[0]["efficiency:two_step_same"]["discard.rows"], 2)
        self.assertNotIn("efficiency:c0_specific", tally.games[0])
        # Ponは教師と同じ行動、Chiは種別だけ一致（教師は4mのChi、R1は1mのChi）。
        self.assertEqual(counts["call.pon.r1.action_match"], 1)
        self.assertEqual(counts["call.chi.r1.true_positive"], 1)
        self.assertNotIn("call.chi.r1.action_match", counts)
        self.assertEqual(counts["call.any.teacher"], 2)
        self.assertEqual(counts["call.r1.match"], 1)
        self.assertNotIn("call.b_pass.match", counts)
        self.assertEqual(tally.diagnostics["scores"]["r1"]["rows"], 2)

        report = judge_l2a_offline_eval(tally)
        self.assertEqual(report["outcome"], EVALUATED)
        self.assertEqual(set(_verdicts(report).values()), {HOLD})

    def test_c0_specific_stratum_uses_the_reference_policy(self) -> None:
        def reference(decision):
            two_step = TwoStepUkeirePolicy().choose_action(decision)
            return next(
                action
                for action in decision.legal_actions
                if isinstance(action, DiscardAction) and action.tile != two_step.tile
            )

        tally = self._collect(reference_policy=reference)
        game = tally.games[0]
        self.assertNotIn("discard.b_ts.match", game["all"])
        self.assertEqual(game["efficiency:c0_specific"]["discard.rows"], 2)

    def test_train_serve_mismatch_stops(self) -> None:
        original = ENCODING_M1._encoder

        def perturbed(decision):
            actions, vectors = original(decision)
            return actions, tuple((vector[0] + 1.0, *vector[1:]) for vector in vectors)

        drifted = replace(ENCODING_M1, _encoder=perturbed)
        self.assertEqual(drifted.block(), ENCODING_M1.block())
        tally = self._collect(proposed=_policy(discard=drifted))
        self.assertEqual(
            tally.correctness["train_serve"]["m1"], {"match": 0, "rows": 2}
        )
        report = judge_l2a_offline_eval(tally)
        self.assertEqual(report["outcome"], INVALID)
        self.assertIn("train-serve agreement (m1)", report["stop_reasons"][0])

    def test_dataset_of_another_encoding_fails_closed(self) -> None:
        self.datasets["m1"] = self.datasets["m0"]
        with self.assertRaises(LearnedPolicyError):
            self._collect()


class StrataTests(unittest.TestCase):
    def test_strata_of_a_default_decision(self) -> None:
        strata = decision_strata(
            cf.discard_decision(cf.NEAR_HAND).input, c0_specific=False
        )
        self.assertIn("round:east", strata)
        self.assertIn("rank:1", strata)
        self.assertIn("rank_tied:yes", strata)
        self.assertIn("opponent_riichi:0", strata)
        self.assertIn("hand:closed", strata)
        self.assertIn("efficiency:two_step_same", strata)

    def test_position_rank_riichi_and_open_hand(self) -> None:
        value = _riichi_opponent_decision().input
        players = list(value.players)
        players[0] = replace(
            players[0],
            score=20000,
            melds=(
                PublicMeld(
                    kind=MeldKind.PON,
                    tiles=(_tile("2z"),) * 3,
                    from_seat=Seat.SEAT_2,
                    called_tile=_tile("2z"),
                ),
            ),
        )
        players[2] = replace(players[2], score=30000)
        players[3] = replace(players[3], score=20000)
        value = replace(
            value,
            players=tuple(players),
            round=replace(value.round, round_wind=Wind.SOUTH, hand_number=4),
        )
        strata = decision_strata(value, c0_specific=None)
        self.assertIn("round:all_last", strata)
        self.assertIn("rank:3", strata)
        self.assertIn("rank_tied:yes", strata)
        self.assertIn("opponent_riichi:1", strata)
        self.assertIn("hand:open", strata)
        self.assertFalse(any(item.startswith("efficiency:") for item in strata))


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.ledger = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _record(self, artifacts):
        return record_offline_evaluation(
            self.ledger,
            artifacts=artifacts,
            datasets={"m1": "d1"},
            source_identity="s" * 64,
        )

    def test_second_use_of_an_artifact_is_detected(self) -> None:
        first, prior = self._record({"m1": "a", "m0": "b", "r1": "c"})
        self.assertEqual((first, prior), ("offline-eval-0001.json", []))
        other, prior = self._record({"m1": "x", "m0": "y", "r1": "z"})
        self.assertEqual(prior, [])
        second, prior = self._record({"m1": "a2", "m0": "y2", "r1": "c"})
        self.assertEqual(second, "offline-eval-0003.json")
        self.assertEqual([item["entry"] for item in prior], [first])
        self.assertEqual(len(list(self.ledger.iterdir())), 3)

    def test_tampered_entry_and_missing_directory_fail_closed(self) -> None:
        self._record({"m1": "a", "m0": "b", "r1": "c"})
        path = self.ledger / "offline-eval-0001.json"
        path.write_text(path.read_text(encoding="utf-8").replace('"a"', '"q"'))
        with self.assertRaises(LearningError):
            self._record({"m1": "a", "m0": "b", "r1": "c"})
        with self.assertRaises(LearningError):
            record_offline_evaluation(
                self.ledger / "missing",
                artifacts={},
                datasets={},
                source_identity="s",
            )


class RunTests(_SourceCase):
    def setUp(self) -> None:
        super().setUp()
        self.ledger = self.root / "ledger"
        self.ledger.mkdir()
        self.artifacts = {}
        self.identities = {}
        for encoding in (ENCODING_M1, ENCODING_M0, ENCODING_R1):
            config = L2aScorerConfig(encoding=encoding, hidden_width=64)
            path = self.root / f"{encoding.name}-artifact"
            artifact = write_l2a_artifact(
                path,
                dataset=read_l2a_dataset(self.datasets[encoding.name]),
                model_config=config,
                training=fixtures.training_block(objective=_OBJECTIVE),
                weights=fixtures.zero_weights(config),
            )
            self.artifacts[encoding.name] = path
            self.identities[encoding.name] = artifact.identity

    def _run(self, **overrides):
        options = {
            "artifacts": self.artifacts,
            "datasets": self.datasets,
            "ledger": self.ledger,
            "source_record": self.source_path,
        }
        options.update(overrides)
        return run_l2a_offline_eval(**options)

    def test_identity_mismatch_stops_without_using_the_ledger(self) -> None:
        swapped = dict(self.artifacts, m1=self.artifacts["m0"], m0=self.artifacts["m1"])
        report = self._run(artifacts=swapped)
        self.assertEqual(report["outcome"], INVALID)
        reasons = " ".join(report["stop_reasons"])
        self.assertIn("m1 artifact uses encoding m0", reasons)
        self.assertIn("m1 dataset is not the dataset of the m1 artifact", reasons)
        self.assertEqual(list(self.ledger.iterdir()), [])

        other = _write_source(self.root / "other-source", seed_base=1000)
        report = self._run(source_record=other)
        self.assertIn("trained from another source", " ".join(report["stop_reasons"]))
        self.assertEqual(list(self.ledger.iterdir()), [])

    def test_second_evaluation_stops(self) -> None:
        record_offline_evaluation(
            self.ledger,
            artifacts=self.identities,
            datasets={},
            source_identity=self.source.identity,
        )
        report = self._run()
        self.assertEqual(report["outcome"], INVALID)
        self.assertIn("already evaluated", report["stop_reasons"][0])
        self.assertNotIn("metrics", report)
        self.assertEqual(
            report["evaluation"]["ledger"]["entry"], "offline-eval-0002.json"
        )

    @unittest.skipUnless(_HAS_TORCH, "requires the optional ML runtime")
    def test_cli_evaluates_once_then_stops(self) -> None:
        arguments = ["l2a-offline-eval", "--source-record", str(self.source_path)]
        for name in ("m1", "m0", "r1"):
            arguments += [f"--{name}-artifact", str(self.artifacts[name])]
            arguments += [f"--{name}-dataset", str(self.datasets[name])]
        arguments += ["--ledger", str(self.ledger)]
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(arguments), 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["outcome"], EVALUATED)
        self.assertEqual(report["gate"]["fingerprint"], l2a_offline_gate_fingerprint())
        self.assertEqual(report["correctness"]["decisions"], 8)
        self.assertEqual(
            report["evaluation"]["artifacts"]["m1"]["identity"], self.identities["m1"]
        )

        again = self._run()
        self.assertEqual(again["outcome"], INVALID)
        self.assertEqual(
            [
                item["entry"]
                for item in again["evaluation"]["ledger"]["prior_evaluations"]
            ],
            ["offline-eval-0001.json"],
        )


if __name__ == "__main__":
    unittest.main()
