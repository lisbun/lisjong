"""#324の分解・分母・禁止split・凍結artifact境界。実験dataは使わない。"""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import candidate_fixtures as cf
import learning_fixtures as fixtures
from test_learning_l2a_scorer import _call, _chi, _v2_rows

from lisjong.learning._canonical import seal
from lisjong.learning.errors import LearnedPolicyError
from lisjong.learning.l2a_call_diagnostic import (
    CONFIG,
    CallGroup,
    collect_calls,
    diagnostic_summaries,
    distribution,
    main,
    run,
    validate_inputs,
)
from lisjong.learning.l2a_encoding import ENCODING_R1
from lisjong.learning.source_record import (
    POLICY_SOURCE_RECORD_SCHEMA_V2,
    open_source_record,
)
from lisjong.policy_contract import PassAction, PonAction, Seat
from lisjong.policy_contract.meld import MeldKind, PublicMeld


def candidates():
    # 同じ鳴く牌に対し、通常5/赤5の消費だけが異なる合法Chi。
    return (
        PassAction(actor=Seat.SEAT_0),
        _chi("4m", ("5m", "6m")),
        _chi("4m", ("0m", "6m")),
        PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_3,
            called_tile=cf.hand("4m")[0],
            consumed_tiles=cf.hand("44m"),
        ),
    )


class CountTests(unittest.TestCase):
    def test_teacher_partition_false_positive_and_hanchan_support(self):
        actions = candidates()
        tally = CallGroup()
        for scores, teacher in (
            ((4.0, 3.0, 2.0, 1.0), actions[1]),  # missed Chi
            ((0.0, 1.0, 3.0, 2.0), actions[1]),  # wrong red choice
            ((0.0, 3.0, 1.0, 2.0), actions[1]),  # exact
            ((0.0, 3.0, 1.0, 2.0), actions[0]),  # false Chi
            ((3.0, 0.0, 1.0, 2.0), actions[0]),  # true negative
            ((0.0, 1.0, 2.0, 3.0), actions[3]),  # Pon exact
        ):
            tally.observe(7, actions, scores, teacher)
        report = tally.report()
        counts = report["counts"]
        for key in ("chi.non_kind", "chi.wrong_candidate", "chi.exact", "chi.fp"):
            self.assertEqual(counts[key], {"rows": 1, "hanchan": 1})
        self.assertEqual(counts["chi.related_mismatch"]["rows"], 3)
        self.assertEqual(counts["rows"], {"rows": 6, "hanchan": 1})
        self.assertAlmostEqual(
            report["metrics"]["chi"]["balanced_accuracy"]["value"], 2 / 3
        )
        self.assertAlmostEqual(report["restricted_chi_match"]["value"], 2 / 3)
        self.assertEqual(report["metrics"]["pon"]["action_match"]["value"], 1)

    def test_tie_uses_given_canonical_order(self):
        actions = candidates()
        tally = CallGroup()
        tally.observe(1, actions, (1.0, 1.0, 1.0, 1.0), actions[1])
        report = tally.report()
        self.assertEqual(report["counts"]["chi.non_kind.pass"]["rows"], 1)
        self.assertEqual(report["counts"]["chi.teacher_rank.2"]["rows"], 1)
        self.assertEqual(report["restricted_chi_match"]["value"], 1)
        self.assertEqual(report["margins"]["chi_vs_non_chi"]["zero"], 1)

    def test_one_chi_and_zero_denominators(self):
        actions = candidates()[:2]
        tally = CallGroup()
        tally.observe(1, actions, (0.0, 1.0), actions[1])
        report = tally.report()
        self.assertEqual(report["counts"]["chi.no_other_candidate"]["rows"], 1)
        self.assertIsNone(report["margins"]["teacher_vs_other_chi"]["median"])
        self.assertIsNone(report["metrics"]["chi"]["balanced_accuracy"]["value"])
        self.assertIsNone(report["metrics"]["pon"]["recall"]["value"])
        self.assertIsNotNone(report["metrics"]["pon"]["recall"]["reason"])

    def test_support_counts_distinct_hanchan(self):
        group = CallGroup()
        for seed in (1, 1, 2):
            group.observe(seed, candidates(), (0.0, 3.0, 2.0, 1.0), candidates()[1])
        self.assertEqual(
            group.report()["counts"]["chi.exact"], {"rows": 3, "hanchan": 2}
        )

    def test_invalid_scores_labels_and_candidates_stop(self):
        actions = candidates()
        cases = [
            (actions, (0.0, 1.0), actions[1]),
            (actions, (float("nan"), 1.0, 2.0, 3.0), actions[1]),
            (actions, (float("inf"), 1.0, 2.0, 3.0), actions[1]),
            (actions[:1], (1.0,), actions[1]),
            ((actions[1], actions[1]), (0.0, 1.0), actions[1]),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(LearnedPolicyError):
                CallGroup().observe(1, *args)

    def test_quantiles_have_defined_interpolation(self):
        self.assertEqual(distribution([-2.0, 0.0, 2.0, 4.0])["q25"], -0.5)
        self.assertEqual(distribution([2.0])["q75"], 2.0)
        self.assertIsNone(distribution([])["median"])


class SourceBoundaryTests(unittest.TestCase):
    def source(self, root):
        decision = _call("45506m789p123s11z", *candidates()[1:3])
        games = []
        for ordinal, split in enumerate(("TRAIN", "SELECT", "OFFLINE-EVAL")):
            seed = 100 + ordinal
            rows = _v2_rows(
                [decision, decision],
                game_ordinal=ordinal,
                seed=seed,
                split=split,
                teacher=lambda d: d.legal_actions[1],
            )
            games.append((split, seed, rows))
        fixtures.write_policy_source_record(
            root, games, schema=POLICY_SOURCE_RECORD_SCHEMA_V2
        )
        return open_source_record(root)

    def test_forbidden_payload_not_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.source(Path(tmp) / "source")
            forbidden = source.root / "game-002" / "source-record.jsonl"
            forbidden.write_text("malformed and forbidden", encoding="utf-8")
            # open_source_record検証はmanifestのみ。禁止payloadは破損していても未読。
            scorer = SimpleNamespace(
                encoding=ENCODING_R1,
                score=lambda context, vectors: tuple(
                    float(i) for i in range(len(vectors))
                ),
            )
            with patch(
                "lisjong.learning.l2a_call_diagnostic.read_source_game",
                wraps=__import__(
                    "lisjong.learning.source_record", fromlist=["read_source_game"]
                ).read_source_game,
            ) as reader:
                report = collect_calls(
                    source, scorer, diagnostic_summaries(source, ("TRAIN", "SELECT"))
                )
                self.assertEqual(
                    [c.args[1]["split"] for c in reader.call_args_list],
                    ["TRAIN", "SELECT"],
                )
            self.assertEqual(report["TRAIN"]["games"], 1)
            counts = report["TRAIN"]["groups"]["closed"]["2"]["counts"]
            self.assertEqual(counts["rows"], {"rows": 2, "hanchan": 1})
            self.assertEqual(report["TRAIN"]["closed_chi_contribution"]["value"], 1)
            with patch(
                "lisjong.learning.l2a_call_diagnostic.read_source_game"
            ) as reader:
                for splits in (("OFFLINE-EVAL",), (), ("TRAIN", "TRAIN")):
                    with self.assertRaises(LearnedPolicyError):
                        diagnostic_summaries(source, splits)
                with self.assertRaises(LearnedPolicyError):
                    collect_calls(source, scorer, (source.summaries[-1],))
                reader.assert_not_called()

    def test_ankan_is_closed_and_open_comparison_uses_same_population(self):
        base = _call("45506m123s11z", *candidates()[1:3])
        decisions = []
        tile = cf.hand("9p")[0]
        for kind in (MeldKind.ANKAN, MeldKind.PON):
            meld = PublicMeld(
                kind=kind,
                tiles=(tile,) * (4 if kind is MeldKind.ANKAN else 3),
                from_seat=None if kind is MeldKind.ANKAN else Seat.SEAT_2,
                called_tile=None if kind is MeldKind.ANKAN else tile,
            )
            players = list(base.input.players)
            players[0] = replace(players[0], melds=(meld,))
            decisions.append(
                replace(base, input=replace(base.input, players=tuple(players)))
            )
        rows = _v2_rows(
            decisions,
            game_ordinal=0,
            seed=100,
            split="TRAIN",
            teacher=lambda d: d.legal_actions[1],
        )
        with tempfile.TemporaryDirectory() as tmp:
            fixtures.write_policy_source_record(
                Path(tmp) / "source",
                [("TRAIN", 100, rows)],
                schema=POLICY_SOURCE_RECORD_SCHEMA_V2,
            )
            source = open_source_record(Path(tmp) / "source")
            scorer = SimpleNamespace(
                encoding=ENCODING_R1,
                score=lambda context, vectors: tuple(
                    float(i) for i in range(len(vectors))
                ),
            )
            report = collect_calls(
                source, scorer, diagnostic_summaries(source, ("TRAIN",))
            )
        groups = report["TRAIN"]["groups"]
        for hand in ("closed", "open"):
            self.assertEqual(
                groups[hand]["chi_legal"]["counts"]["rows"], {"rows": 1, "hanchan": 1}
            )
            self.assertEqual(groups[hand]["2"], groups[hand]["chi_legal"])
        self.assertEqual(groups["all"]["all"]["counts"]["rows"]["rows"], 2)

    def test_limit_failure_stops_before_payload_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.source(Path(tmp) / "source")
            with patch(
                "lisjong.learning.l2a_call_diagnostic.read_source_game"
            ) as reader:

                def stop():
                    raise LearnedPolicyError("limit")

                with self.assertRaisesRegex(LearnedPolicyError, "limit"):
                    collect_calls(
                        source, None, diagnostic_summaries(source, ("TRAIN",)), stop
                    )
                reader.assert_not_called()

    def test_non_frozen_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.source(Path(tmp) / "source")
            with self.assertRaisesRegex(LearnedPolicyError, "identity"):
                validate_inputs(source, None)

    @unittest.skipUnless(
        sys.platform == "linux", "Linux production resource accounting"
    )
    def test_full_requires_matching_preflight_before_payload_collection(self):
        module = "lisjong.learning.l2a_call_diagnostic."
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(module + "open_source_record"),
            patch(module + "load_l2a_scorer"),
            patch(module + "validate_inputs"),
            patch(
                module + "runtime_record",
                return_value={"backend": "rust", "revision": "current"},
            ),
            patch(module + "collect_calls") as collect,
        ):
            with self.assertRaisesRegex(
                LearnedPolicyError, "requires matching preflight"
            ):
                run("source", "r1", phase="full")
            path = Path(tmp) / "preflight.json"
            for changed in (
                {"within_budget": False},
                {"environment": {"backend": "rust", "revision": "old"}},
                {"status": "STOP"},
                {"config": {}},
            ):
                report = {
                    "config": CONFIG,
                    "phase": "preflight",
                    "status": "COMPLETE",
                    "environment": {"backend": "rust", "revision": "current"},
                    "within_budget": True,
                }
                report.update(changed)
                path.write_text(json.dumps(seal(report)))
                with self.assertRaisesRegex(LearnedPolicyError, "preflight.*mismatch"):
                    run("source", "r1", phase="full", preflight_path=path)
            collect.assert_not_called()

    def test_cli_failure_does_not_claim_completion_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "stop.json"
            argv = [
                "--source",
                str(Path(tmp) / "missing"),
                "--r1",
                str(Path(tmp) / "r1"),
                "--phase",
                "preflight",
                "--output",
                str(output),
            ]
            self.assertEqual(main(argv), 2)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "STOP")
            before = output.read_bytes()
            with self.assertRaises(SystemExit):
                main(argv)
            self.assertEqual(before, output.read_bytes())
            self.assertEqual(report["config"], CONFIG)


if __name__ == "__main__":
    unittest.main()
