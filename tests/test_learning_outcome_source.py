"""Issue #193 focal outcome source consumerとoutcome targetのtest。

Arena producer（lisbun/lisjong-arena#359）の論理contractを合成fixtureで書き、
strict read、#191 survivor / exploration selectionの再計算照合、`target_q`、
fail closedを固定する。
"""

import ast
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import outcome_fixtures as of

from lisjong.learning import (
    EXPLORATION_TOKEN_IDENTITY,
    OUTCOME_OBJECTIVE_IDENTITY,
    OUTCOME_TARGET_IDENTITY,
    LearningError,
    OutcomeSourceError,
    UnsupportedSourceSchemaError,
    build_outcome_targets,
    read_outcome_source,
    summarize_outcome_targets,
)
from lisjong.learning import outcome_source as outcome_module
from lisjong.learning._o0 import O0DecisionKind
from lisjong.learning.outcome_source import (
    DECISION_PAYLOAD_FILENAME,
    FOCAL_ROTATION_RULE,
    KYOKU_PAYLOAD_FILENAME,
    MANIFEST_FILENAME,
)

KYOKUS = KYOKU_PAYLOAD_FILENAME
DECISIONS = DECISION_PAYLOAD_FILENAME

ELIGIBLE_INDICES = (0, 5)
"""fixtureのfocal decisionのうちsurvivor >= 2のもの（kyoku 0とkyoku 2）。"""

GUARD_INDEX = 1
"""WIN（tsumo）guard decision。"""


class _SourceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def write(self, games=of.SCIENTIFIC_GAMES, **options):
        return of.write_outcome_source(self.root / "source", games, **options)


class ReadAndTargetTests(_SourceTestCase):
    def test_reads_scientific_source_across_the_train_select_boundary(self) -> None:
        source = read_outcome_source(self.write())

        self.assertEqual(source.population_role, "SCIENTIFIC")
        self.assertEqual(
            [
                (game.game_ordinal, game.split, int(game.focal_seat))
                for game in source.games
            ],
            [(0, "TRAIN", 0), (1, "TRAIN", 1), (2, "SELECT", 2)],
        )
        self.assertEqual(set(source.allocation_bindings), {"TRAIN", "SELECT"})
        self.assertEqual(source.provenance()["identity"], source.identity)

    def test_reads_calibration_source(self) -> None:
        source = read_outcome_source(
            self.write(of.CALIBRATION_GAMES, role="CALIBRATION")
        )

        self.assertEqual(source.population_role, "CALIBRATION")
        self.assertEqual({game.split for game in source.games}, {"CALIBRATION"})

    def test_target_is_the_focal_kyoku_point_delta(self) -> None:
        targets = build_outcome_targets(read_outcome_source(self.write()))

        self.assertEqual(targets.target_identity, OUTCOME_TARGET_IDENTITY)
        self.assertEqual(len(targets.rows), 2 * len(of.SCIENTIFIC_GAMES))
        for row in targets.rows:
            with self.subTest(game=row.game_ordinal, kyoku=row.kyoku_ordinal):
                focal = row.game_ordinal % 4
                self.assertEqual(int(row.decision.input.self_seat), focal)
                expected = {0: 8.0, 2: of.FINAL_KYOKU_TARGET}[row.kyoku_ordinal]
                self.assertEqual(row.target_q, expected)

    def test_final_kyoku_uses_the_pre_final_adjustment_boundary(self) -> None:
        source = read_outcome_source(self.write())
        for game in source.games:
            final = game.kyokus[-1]
            focal = int(game.focal_seat)
            with self.subTest(game=game.game_ordinal):
                self.assertTrue(final.is_final_kyoku)
                self.assertEqual(final.riichi_sticks_after, 1)
                # fixtureではfocal seatがtopで、最終帰属の+1000を受け取る。
                self.assertEqual(
                    game.hanchan_final_scores[focal] - final.points_after_kyoku[focal],
                    1000,
                )
        rows = [
            row for row in build_outcome_targets(source).rows if row.kyoku_ordinal == 2
        ]
        self.assertTrue(rows)
        self.assertTrue(all(row.target_q == of.FINAL_KYOKU_TARGET for row in rows))

    def test_changing_hanchan_final_scores_does_not_change_the_target(self) -> None:
        path = self.write()
        before = [
            row.target_q
            for row in build_outcome_targets(read_outcome_source(path)).rows
        ]
        for ordinal in range(len(of.SCIENTIFIC_GAMES)):
            of.mutate_game_summary(
                path,
                ordinal,
                lambda summary: summary.update(hanchan_final_scores=[0, 0, 0, 100000]),
            )
        after = [
            row.target_q
            for row in build_outcome_targets(read_outcome_source(path)).rows
        ]

        self.assertEqual(before, after)

    def test_eligible_rows_point_at_the_selected_candidate_only(self) -> None:
        source = read_outcome_source(self.write())
        selected = {
            (game.game_ordinal, item.focal_decision_ordinal): item.selected_action
            for game, item in source.decisions()
        }
        targets = build_outcome_targets(source)

        self.assertEqual(
            [(row.game_ordinal, row.focal_decision_ordinal) for row in targets.rows],
            [
                (game, index)
                for game in range(len(of.SCIENTIFIC_GAMES))
                for index in ELIGIBLE_INDICES
            ],
        )
        for row in targets.rows:
            with self.subTest(
                game=row.game_ordinal, decision=row.focal_decision_ordinal
            ):
                self.assertGreaterEqual(len(row.survivors), 2)
                self.assertIn(row.selected_candidate_index, row.survivors)
                self.assertIs(
                    row.selected_candidate.action,
                    row.candidates[row.selected_candidate_index].action,
                )
                self.assertEqual(
                    row.selected_candidate.action,
                    selected[(row.game_ordinal, row.focal_decision_ordinal)],
                )
                self.assertTrue(
                    any(
                        row.selected_candidate.action is legal
                        for legal in row.decision.legal_actions
                    )
                )
                # targetはrowの1値だけで、他のsurvivorへ付与するfieldを持たない。
                self.assertIsInstance(row.target_q, float)
        # fixtureは非canonical-first survivorを選ぶrowも含む。
        self.assertTrue(
            any(
                row.selected_candidate_index != row.survivors[0] for row in targets.rows
            )
        )

    def test_exclusions_and_summary(self) -> None:
        targets = build_outcome_targets(read_outcome_source(self.write()))
        summary = summarize_outcome_targets(targets)

        self.assertEqual(
            targets.excluded,
            {"response": 3, "riichi": 3, "single_survivor": 3, "win": 3},
        )
        self.assertEqual(summary["eligible_row_count"], 6)
        self.assertEqual(summary["unique_eligible_kyoku_count"], 6)
        self.assertEqual(summary["unique_eligible_kyoku_per_hanchan"], 2.0)
        self.assertEqual(summary["survivor_count_distribution"], {"2": 6})
        self.assertAlmostEqual(
            summary["canonical_first_selected_rate"]
            + summary["non_canonical_first_selected_rate"],
            1.0,
        )
        self.assertEqual(
            summary["target_q"],
            {
                "max": 8.0,
                "mean": 5.0,
                "median": 5.0,
                "min": 2.0,
                "quantiles": {
                    "0.05": 2.0,
                    "0.25": 2.0,
                    "0.50": 5.0,
                    "0.75": 8.0,
                    "0.95": 8.0,
                },
                "std": 3.0,
                "zero_fraction": 0.0,
            },
        )
        self.assertEqual(summary["target_identity"], OUTCOME_TARGET_IDENTITY)
        self.assertEqual(json.loads(json.dumps(summary)), summary)
        self.assertEqual(summarize_outcome_targets(targets), summary)

    def test_identities(self) -> None:
        self.assertEqual(
            OUTCOME_TARGET_IDENTITY,
            "lisjong-offense-l0.3-focal-kyoku-point-delta-1000-v1",
        )
        self.assertEqual(
            OUTCOME_OBJECTIVE_IDENTITY,
            "lisjong-offense-l0.3-selected-action-mc-q-mse-v1",
        )
        self.assertEqual(
            EXPLORATION_TOKEN_IDENTITY,
            "lisjong-arena-l0.3-focal-decision-token-sha256-v1",
        )


class FailClosedTests(_SourceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.path = self.write()

    def assertRejected(self, error=OutcomeSourceError):
        with self.assertRaises(error):
            read_outcome_source(self.path)

    def mutate_decision(self, index, mutate, *, game=0):
        def apply(rows):
            mutate(rows[index])

        of.mutate_rows(self.path, game, DECISIONS, apply)

    def mutate_kyoku(self, index, mutate, *, game=0):
        def apply(rows):
            mutate(rows[index])

        of.mutate_rows(self.path, game, KYOKUS, apply)

    # -- manifest ----------------------------------------------------------

    def test_unsupported_schema(self) -> None:
        of.mutate_manifest(self.path, lambda body: body.update(schema="other-v2"))
        self.assertRejected(UnsupportedSourceSchemaError)

    def test_kind_mismatch(self) -> None:
        of.mutate_manifest(self.path, lambda body: body.update(kind="other"))
        self.assertRejected()

    def test_manifest_digest_mismatch(self) -> None:
        manifest = self.path / MANIFEST_FILENAME
        text = manifest.read_text("utf-8").replace('"SCIENTIFIC"', '"CALIBRATION"')
        manifest.write_text(text, encoding="utf-8", newline="\n")
        self.assertRejected()

    def test_non_canonical_manifest(self) -> None:
        manifest = self.path / MANIFEST_FILENAME
        value = json.loads(manifest.read_text("utf-8"))
        manifest.write_text(json.dumps(value), encoding="utf-8", newline="\n")
        self.assertRejected()

    def test_manifest_missing_or_extra_field(self) -> None:
        for mutate in (
            lambda body: body.pop("allocation_bindings"),
            lambda body: body.update(extra=1),
        ):
            with self.subTest(mutate=mutate):
                path = self.root / f"case-{id(mutate)}"
                of.write_outcome_source(path)
                of.mutate_manifest(path, mutate)
                with self.assertRaises(OutcomeSourceError):
                    read_outcome_source(path)

    def test_behavior_identity_mismatch(self) -> None:
        for field in (
            "exploration_behavior_identity",
            "baseline_runtime_identity",
            "exploration_token_identity",
            "focal_rotation_rule",
        ):
            with self.subTest(field=field):
                path = self.root / f"behavior-{field}"
                of.write_outcome_source(path)
                of.mutate_manifest(
                    path, lambda body: body["behavior"].update({field: "0" * 64})
                )
                with self.assertRaises(OutcomeSourceError):
                    read_outcome_source(path)
        self.assertEqual(
            FOCAL_ROTATION_RULE,
            "lisjong-arena-l0.3-focal-seat-game-ordinal-mod-4-v1",
        )

    def test_behavior_extra_field(self) -> None:
        of.mutate_manifest(self.path, lambda body: body["behavior"].update(extra="x"))
        self.assertRejected()

    def test_allocation_binding_mismatch(self) -> None:
        of.mutate_manifest(
            self.path, lambda body: body["allocation_bindings"].pop("SELECT")
        )
        self.assertRejected()

    # -- population / ordinal ---------------------------------------------

    def test_game_ordinal_restart_at_split_boundary(self) -> None:
        of.mutate_game_summary(
            self.path, 2, lambda summary: summary.update(game_ordinal=0, focal_seat=0)
        )
        self.assertRejected()

    def test_game_ordinal_order_swap(self) -> None:
        def swap(body):
            body["games"][0], body["games"][1] = body["games"][1], body["games"][0]

        of.mutate_manifest(self.path, swap)
        self.assertRejected()

    def test_focal_seat_is_not_the_rotation(self) -> None:
        of.mutate_game_summary(
            self.path, 1, lambda summary: summary.update(focal_seat=0)
        )
        self.assertRejected()

    def test_seed_reuse_across_splits(self) -> None:
        path = of.write_outcome_source(
            self.root / "dup", (("TRAIN", 1001), ("SELECT", 1001))
        )
        with self.assertRaises(OutcomeSourceError):
            read_outcome_source(path)

    def test_mixed_population_role(self) -> None:
        cases = (
            ("SCIENTIFIC", (("TRAIN", 1), ("CALIBRATION", 2))),
            ("CALIBRATION", (("CALIBRATION", 1), ("TRAIN", 2))),
            ("CALIBRATION", (("SELECT", 1),)),
            ("OFFLINE", (("TRAIN", 1),)),
        )
        for index, (role, games) in enumerate(cases):
            with self.subTest(role=role, games=games):
                path = of.write_outcome_source(
                    self.root / f"mix-{index}", games, role=role
                )
                with self.assertRaises(OutcomeSourceError):
                    read_outcome_source(path)

    def test_missing_or_extra_game_directory(self) -> None:
        (self.path / "game-003").mkdir()
        self.assertRejected()

    def test_game_summary_extra_field(self) -> None:
        of.mutate_game_summary(self.path, 0, lambda summary: summary.update(extra=1))
        self.assertRejected()

    # -- payload -----------------------------------------------------------

    def test_payload_digest_mismatch(self) -> None:
        payload = self.path / "game-000" / DECISIONS
        payload.write_text(
            payload.read_text("utf-8") + "\n", encoding="utf-8", newline="\n"
        )
        self.assertRejected()

    def test_non_canonical_row(self) -> None:
        directory = self.path / "game-000"
        rows = of.read_rows(self.path, 0, KYOKUS)
        (directory / KYOKUS).write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        digest = outcome_module.file_digest(directory / KYOKUS)
        of.mutate_game_summary(
            self.path,
            0,
            lambda summary: summary.update(files={**summary["files"], KYOKUS: digest}),
        )
        self.assertRejected()

    def test_row_missing_or_extra_field(self) -> None:
        cases = (
            (DECISIONS, lambda row: row.pop("exploration_token")),
            (DECISIONS, lambda row: row.update(extra=None)),
            (KYOKUS, lambda row: row.pop("end")),
            (KYOKUS, lambda row: row.update(extra=None)),
        )
        for index, (name, mutate) in enumerate(cases):
            with self.subTest(name=name, index=index):
                path = of.write_outcome_source(self.root / f"row-{index}")
                of.mutate_rows(path, 0, name, lambda rows: mutate(rows[0]))
                with self.assertRaises(OutcomeSourceError):
                    read_outcome_source(path)

    def test_focal_decision_ordinal_gap(self) -> None:
        def drop(rows):
            del rows[2]

        of.mutate_rows(self.path, 0, DECISIONS, drop)
        of.mutate_game_summary(
            self.path, 0, lambda summary: summary.update(decision_count=5)
        )
        self.assertRejected()

    def test_decision_count_mismatch(self) -> None:
        of.mutate_game_summary(
            self.path, 0, lambda summary: summary.update(decision_count=7)
        )
        self.assertRejected()

    def test_step_ordering_mismatch(self) -> None:
        self.mutate_decision(3, lambda row: row.update(step_ordinal=0))
        self.assertRejected()

    # -- decision ----------------------------------------------------------

    def test_actor_seat_is_not_the_focal_seat(self) -> None:
        self.mutate_decision(0, lambda row: row.update(actor_seat=1))
        self.assertRejected()

    def test_policy_input_seat_is_not_the_focal_seat(self) -> None:
        path = of.write_outcome_source(self.root / "seat")
        other = of.read_rows(path, 1, DECISIONS)[0]

        def replace(rows):
            rows[0]["policy_input"] = other["policy_input"]

        of.mutate_rows(path, 0, DECISIONS, replace)
        with self.assertRaises(OutcomeSourceError):
            read_outcome_source(path)

    def test_legal_actions_empty_duplicate_or_unordered(self) -> None:
        cases = (
            lambda row: row.update(legal_actions=[]),
            lambda row: row["legal_actions"].insert(0, row["legal_actions"][0]),
            lambda row: row["legal_actions"].reverse(),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(index=index):
                path = of.write_outcome_source(self.root / f"legal-{index}")
                of.mutate_rows(path, 0, DECISIONS, lambda rows: mutate(rows[0]))
                with self.assertRaises(OutcomeSourceError):
                    read_outcome_source(path)

    def test_selected_action_is_not_legal(self) -> None:
        other = of.read_rows(self.path, 0, DECISIONS)[3]["legal_actions"][0]
        self.mutate_decision(0, lambda row: row.update(selected_action=other))
        self.assertRejected()

    def test_guard_selection_mismatch(self) -> None:
        def pick_discard(row):
            discard = next(
                action for action in row["legal_actions"] if action["kind"] == "discard"
            )
            row["selected_action"] = discard

        self.mutate_decision(GUARD_INDEX, pick_discard)
        self.assertRejected()

    def test_guard_with_producer_survivors(self) -> None:
        self.mutate_decision(
            GUARD_INDEX,
            lambda row: row.update(producer_survivor_actions=[row["selected_action"]]),
        )
        self.assertRejected()

    def test_discard_without_producer_survivors(self) -> None:
        self.mutate_decision(0, lambda row: row.update(producer_survivor_actions=None))
        self.assertRejected()

    def test_producer_survivors_disagree(self) -> None:
        cases = (
            lambda row: row["producer_survivor_actions"].pop(),
            lambda row: row["producer_survivor_actions"].reverse(),
            lambda row: row["producer_survivor_actions"].append(
                row["legal_actions"][0]
            ),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(index=index):
                path = of.write_outcome_source(self.root / f"survivors-{index}")
                of.mutate_rows(path, 0, DECISIONS, lambda rows: mutate(rows[0]))
                with self.assertRaises(OutcomeSourceError):
                    read_outcome_source(path)

    def test_selected_action_is_not_the_bucket_survivor(self) -> None:
        def other_survivor(row):
            survivors = row["producer_survivor_actions"]
            row["selected_action"] = next(
                action for action in survivors if action != row["selected_action"]
            )

        self.mutate_decision(0, other_survivor)
        self.assertRejected()

    def test_token_does_not_explain_the_selection(self) -> None:
        def flip(row):
            value = int(row["exploration_token"], 16) ^ 1
            row["exploration_token"] = f"{value:064x}"

        self.mutate_decision(0, flip)
        self.assertRejected()

    def test_invalid_token(self) -> None:
        for index, value in enumerate(("A" * 64, "0" * 63, 7)):
            with self.subTest(token=value):
                path = of.write_outcome_source(self.root / f"token-{index}")
                of.mutate_rows(
                    path,
                    0,
                    DECISIONS,
                    lambda rows: rows[GUARD_INDEX].update(exploration_token=value),
                )
                with self.assertRaises(OutcomeSourceError):
                    read_outcome_source(path)

    def test_missing_kyoku_reference(self) -> None:
        self.mutate_decision(5, lambda row: row.update(kyoku_ordinal=3))
        self.assertRejected()

    def test_policy_input_round_does_not_match_the_kyoku(self) -> None:
        # decision 1はE1 honba0、kyoku 1はE2。
        self.mutate_decision(GUARD_INDEX, lambda row: row.update(kyoku_ordinal=1))
        self.assertRejected()

    def test_kyoku_honba_does_not_match_the_policy_input(self) -> None:
        self.mutate_kyoku(2, lambda row: row.update(honba=2))
        self.assertRejected()

    # -- kyoku -------------------------------------------------------------

    def test_score_snapshot_shape(self) -> None:
        cases = (
            lambda row: row["points_after_kyoku"].pop(),
            lambda row: row["points_before_kyoku"].append(0),
            lambda row: row["points_after_kyoku"].__setitem__(0, 25000.0),
            lambda row: row["points_after_kyoku"].__setitem__(0, True),
            lambda row: row["points_after_kyoku"].__setitem__(0, "25000"),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(index=index):
                path = of.write_outcome_source(self.root / f"points-{index}")
                of.mutate_rows(path, 0, KYOKUS, lambda rows: mutate(rows[2]))
                with self.assertRaises(OutcomeSourceError):
                    read_outcome_source(path)

    def test_hanchan_final_scores_shape(self) -> None:
        of.mutate_game_summary(
            self.path, 0, lambda summary: summary.update(hanchan_final_scores=[1, 2, 3])
        )
        self.assertRejected()

    def test_points_before_do_not_continue(self) -> None:
        def shift(row):
            row["points_before_kyoku"][0] += 100
            row["points_after_kyoku"][0] += 100

        self.mutate_kyoku(1, shift)
        self.assertRejected()

    def test_riichi_sticks_do_not_continue(self) -> None:
        self.mutate_kyoku(2, lambda row: row.update(riichi_sticks_before=1))
        self.assertRejected()

    def test_kyoku_ordinal_gap(self) -> None:
        self.mutate_kyoku(1, lambda row: row.update(kyoku_ordinal=2))
        self.assertRejected()

    def test_final_kyoku_flag(self) -> None:
        for index, (kyoku, value) in enumerate(((2, False), (0, True))):
            with self.subTest(kyoku=kyoku):
                path = of.write_outcome_source(self.root / f"final-{index}")
                of.mutate_rows(
                    path,
                    0,
                    KYOKUS,
                    lambda rows: rows[kyoku].update(is_final_kyoku=value),
                )
                with self.assertRaises(OutcomeSourceError):
                    read_outcome_source(path)

    def test_invalid_end(self) -> None:
        cases = (
            {"kind": "win", "winner_seats": []},
            {"kind": "win", "winner_seats": [0, 0]},
            {"kind": "win", "winner_seats": [4]},
            {"draw_kind": "", "kind": "draw"},
            {"kind": "abort"},
        )
        for index, end in enumerate(cases):
            with self.subTest(end=end):
                path = of.write_outcome_source(self.root / f"end-{index}")
                of.mutate_rows(path, 0, KYOKUS, lambda rows: rows[0].update(end=end))
                with self.assertRaises(OutcomeSourceError):
                    read_outcome_source(path)

    def test_fail_closed_errors_are_learning_errors(self) -> None:
        self.assertTrue(issubclass(OutcomeSourceError, LearningError))


class ModuleBoundaryTests(unittest.TestCase):
    def test_consumer_does_not_import_arena_or_ml_runtime(self) -> None:
        tree = ast.parse(inspect.getsource(outcome_module))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }

        self.assertFalse(
            any(
                name.startswith(("lisjong_arena", "torch", "riichienv", "random"))
                for name in imported
            )
        )

    def test_guard_kinds_are_the_o0_kinds(self) -> None:
        self.assertEqual(
            {kind.value for kind in O0DecisionKind} - {"discard"},
            {"win", "riichi", "response"},
        )


if __name__ == "__main__":
    unittest.main()
