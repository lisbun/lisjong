"""Issue #195 lisjong-engine focal outcome source lineageのconsumer test。

Arena producer（lisbun/lisjong-arena#370）の論理contractを合成fixtureで書き、
engine schemaのstrict read、authoritative score factの検証、`step_ordinal`を
合成しないこと、selection再計算、population / DIAGNOSTIC境界、targetが
RiichiEnv lineageと同じ式・identityであることを固定する。
"""

import tempfile
import unittest
from pathlib import Path

import learning_fixtures as fixtures
import outcome_fixtures as of

from lisjong.learning import (
    ENGINE_OUTCOME_SOURCE_SCHEMA,
    OUTCOME_SOURCE_SCHEMA,
    OUTCOME_TARGET_IDENTITY,
    OutcomeSourceError,
    UnsupportedSourceSchemaError,
    build_outcome_targets,
    read_outcome_source,
)
from lisjong.learning.outcome_source import (
    DECISION_PAYLOAD_FILENAME,
    KYOKU_PAYLOAD_FILENAME,
    LISJONG_ENGINE_SEED_DOMAIN,
)

KYOKUS = KYOKU_PAYLOAD_FILENAME
DECISIONS = DECISION_PAYLOAD_FILENAME

GUARD_INDEX = 1
"""WIN（tsumo）guard decision。"""


def _targets(path):
    return [
        (
            row.game_ordinal,
            row.kyoku_ordinal,
            row.selected_candidate_index,
            row.target_q,
        )
        for row in build_outcome_targets(read_outcome_source(path)).rows
    ]


class _EngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.path = self.write("source")

    def write(self, name, games=of.SCIENTIFIC_GAMES, **options):
        return of.write_engine_outcome_source(self.root / name, games, **options)

    def assertRejected(self, path=None, error=OutcomeSourceError, regex=None):
        with self.assertRaises(error) as caught:
            read_outcome_source(self.path if path is None else path)
        if regex is not None:
            self.assertRegex(str(caught.exception), regex)

    def mutate_kyoku(self, index, mutate, *, game=0):
        of.mutate_rows(self.path, game, KYOKUS, lambda rows: mutate(rows[index]))

    def mutate_decision(self, index, mutate, *, game=0):
        of.mutate_rows(self.path, game, DECISIONS, lambda rows: mutate(rows[index]))


class EngineReadTests(_EngineTestCase):
    def test_reads_engine_source(self) -> None:
        source = read_outcome_source(self.path)

        self.assertEqual(source.schema, ENGINE_OUTCOME_SOURCE_SCHEMA)
        self.assertEqual(source.provenance()["schema"], ENGINE_OUTCOME_SOURCE_SCHEMA)
        self.assertEqual(source.population_role, "SCIENTIFIC")
        self.assertEqual(
            [(game.split, int(game.focal_seat)) for game in source.games],
            [("TRAIN", 0), ("TRAIN", 1), ("SELECT", 2)],
        )
        self.assertEqual(
            {binding["seed_domain"] for binding in source.allocation_bindings.values()},
            {LISJONG_ENGINE_SEED_DOMAIN},
        )
        for game in source.games:
            with self.subTest(game=game.game_ordinal):
                self.assertIsNone(game.hanchan_final_scores)
                self.assertIsNone(game.hanchan_final_riichi_sticks)
                self.assertEqual(game.match_end_reason, "final_round")
                self.assertEqual(
                    game.final_riichi_stick_awards, ((game.focal_seat, 1000),)
                )
                final = game.kyokus[-1]
                self.assertEqual(
                    game.hanchan_final_raw_scores[int(game.focal_seat)]
                    - final.points_after_kyoku[int(game.focal_seat)],
                    1000,
                )
                for kyoku in game.kyokus:
                    self.assertEqual(
                        kyoku.point_deltas,
                        tuple(
                            after - before
                            for before, after in zip(
                                kyoku.points_before_kyoku,
                                kyoku.points_after_kyoku,
                                strict=True,
                            )
                        ),
                    )

    def test_reads_calibration_and_diagnostic_sources(self) -> None:
        for role, games in (
            ("CALIBRATION", of.CALIBRATION_GAMES),
            ("DIAGNOSTIC", of.DIAGNOSTIC_GAMES),
        ):
            with self.subTest(role=role):
                source = read_outcome_source(self.write(role, games, role=role))
                self.assertEqual(source.population_role, role)
                self.assertEqual({game.split for game in source.games}, {role})

    def test_historical_source_has_no_engine_facts(self) -> None:
        source = read_outcome_source(of.write_outcome_source(self.root / "riichienv"))

        self.assertEqual(source.schema, OUTCOME_SOURCE_SCHEMA)
        self.assertEqual(source.provenance()["schema"], OUTCOME_SOURCE_SCHEMA)
        for game, decision in source.decisions():
            self.assertIsNone(game.hanchan_final_raw_scores)
            self.assertIsNone(game.final_riichi_stick_awards)
            self.assertIsNone(game.match_end_reason)
            self.assertIsInstance(decision.step_ordinal, int)
        self.assertTrue(
            all(
                kyoku.point_deltas is None
                for game in source.games
                for kyoku in game.kyokus
            )
        )

    def test_engine_source_never_fabricates_step_ordinal(self) -> None:
        source = read_outcome_source(self.path)
        targets = build_outcome_targets(source)

        self.assertTrue(targets.rows)
        self.assertTrue(
            all(decision.step_ordinal is None for _, decision in source.decisions())
        )
        self.assertTrue(all(row.step_ordinal is None for row in targets.rows))

    def test_targets_use_the_frozen_identity_and_formula(self) -> None:
        targets = build_outcome_targets(read_outcome_source(self.path))

        self.assertEqual(targets.target_identity, OUTCOME_TARGET_IDENTITY)
        self.assertEqual(
            OUTCOME_TARGET_IDENTITY,
            "lisjong-offense-l0.3-focal-kyoku-point-delta-1000-v1",
        )
        self.assertEqual(len(targets.rows), 2 * len(of.SCIENTIFIC_GAMES))
        for row in targets.rows:
            with self.subTest(game=row.game_ordinal, kyoku=row.kyoku_ordinal):
                expected = {0: 8.0, 2: of.FINAL_KYOKU_TARGET}[row.kyoku_ordinal]
                self.assertEqual(row.target_q, expected)
        self.assertEqual(
            _targets(self.path),
            _targets(of.write_outcome_source(self.root / "riichienv")),
        )

    def test_final_audit_facts_cannot_alter_the_target(self) -> None:
        before = _targets(self.path)

        def move_award(summary):
            # 最終配分を別seatへ移し、raw scoreも整合させる（audit fact同士は整合）。
            focal = summary["focal_seat"]
            other = (focal + 1) % 4
            scores = list(summary["hanchan_final_raw_scores"])
            scores[focal] -= 1000
            scores[other] += 1000
            summary.update(
                final_riichi_stick_awards=[{"amount": 1000, "recipient_seat": other}],
                hanchan_final_raw_scores=scores,
                match_end_reason="target_reached",
            )

        for ordinal in range(len(of.SCIENTIFIC_GAMES)):
            of.mutate_game_summary(self.path, ordinal, move_award)
        source = read_outcome_source(self.path)

        self.assertEqual(_targets(self.path), before)
        for game in source.games:
            self.assertEqual(game.match_end_reason, "target_reached")
            self.assertNotIn(
                game.focal_seat,
                [seat for seat, _ in game.final_riichi_stick_awards],
            )


class EngineFailClosedTests(_EngineTestCase):
    # -- schema ------------------------------------------------------------

    def test_unknown_schema(self) -> None:
        for index, schema in enumerate(
            (
                "arena-offense-l0.3-lisjong-engine-focal-outcome-source-v2",
                "other-v1",
                ["list"],
            )
        ):
            with self.subTest(schema=schema):
                path = self.write(f"schema-{index}")
                of.mutate_manifest(path, lambda body: body.update(schema=schema))
                self.assertRejected(path, UnsupportedSourceSchemaError)

    def test_schemas_are_not_reinterpreted_across_lineages(self) -> None:
        """engine rowをRiichiEnv schemaとして、またはその逆として読まない。"""
        of.mutate_manifest(
            self.path, lambda body: body.update(schema=OUTCOME_SOURCE_SCHEMA)
        )
        self.assertRejected()
        historical = of.write_outcome_source(self.root / "riichienv")
        of.mutate_manifest(
            historical, lambda body: body.update(schema=ENGINE_OUTCOME_SOURCE_SCHEMA)
        )
        self.assertRejected(historical)

    def test_game_summary_field_set(self) -> None:
        cases = (
            lambda summary: summary.pop("match_end_reason"),
            lambda summary: summary.update(hanchan_final_riichi_sticks=0),
            lambda summary: summary.update(match_end_reason=1),
            lambda summary: summary.update(match_end_reason="unknown_reason"),
            lambda summary: summary.update(hanchan_final_raw_scores=[1, 2, 3]),
            lambda summary: summary.update(final_riichi_stick_awards={}),
            lambda summary: summary["final_riichi_stick_awards"][0].update(extra=1),
            lambda summary: summary["final_riichi_stick_awards"][0].update(
                recipient_seat=4
            ),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(index=index):
                path = self.write(f"summary-{index}")
                of.mutate_game_summary(path, 0, mutate)
                self.assertRejected(path)

    # -- authoritative score facts ------------------------------------------

    def test_point_deltas_tamper(self) -> None:
        def tamper(row):
            row["point_deltas"][0] += 1000

        self.mutate_kyoku(0, tamper)
        self.assertRejected(regex="point_deltas")

    def test_point_deltas_shape_or_missing(self) -> None:
        cases = (
            lambda row: row.pop("point_deltas"),
            lambda row: row.update(point_deltas=[0, 0, 0]),
            lambda row: row.update(point_deltas=[0, 0, 0, "0"]),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(index=index):
                path = self.write(f"deltas-{index}")
                of.mutate_rows(path, 0, KYOKUS, lambda rows: mutate(rows[0]))
                self.assertRejected(path)

    def test_boundary_tamper_consistent_with_deltas_breaks_continuity(self) -> None:
        def tamper(row):
            row["point_deltas"][0] += 1000
            row["point_deltas"][1] -= 1000
            row["points_after_kyoku"][0] += 1000
            row["points_after_kyoku"][1] -= 1000

        self.mutate_kyoku(0, tamper)
        self.assertRejected(regex="continue")

    def test_score_riichi_stick_conservation(self) -> None:
        self.mutate_kyoku(1, lambda row: row.update(riichi_sticks_after=1))
        self.assertRejected(regex="conservation")

    def test_riichi_sticks_do_not_continue(self) -> None:
        def tamper(row):
            row.update(riichi_sticks_before=1, riichi_sticks_after=2)

        self.mutate_kyoku(2, tamper)
        self.assertRejected()

    def test_round_identity_repeats(self) -> None:
        def repeat(rows):
            rows[2].update(hand_number=rows[1]["hand_number"], honba=0, dealer_seat=1)

        of.mutate_rows(self.path, 0, KYOKUS, repeat)
        self.assertRejected(regex="round identity")

    def test_final_audit_inconsistent_with_final_kyoku(self) -> None:
        cases = (
            lambda summary: summary["hanchan_final_raw_scores"].__setitem__(
                0, summary["hanchan_final_raw_scores"][0] + 1000
            ),
            lambda summary: summary.update(final_riichi_stick_awards=[]),
            lambda summary: summary["final_riichi_stick_awards"][0].update(amount=2000),
            lambda summary: summary["final_riichi_stick_awards"][0].update(amount=0),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(index=index):
                path = self.write(f"audit-{index}")
                of.mutate_game_summary(path, 0, mutate)
                self.assertRejected(path)

    # -- decision ordering / step_ordinal -------------------------------------

    def test_engine_decision_rows_reject_step_ordinal(self) -> None:
        self.mutate_decision(0, lambda row: row.update(step_ordinal=1))
        self.assertRejected()

    def test_kyoku_ordinal_must_not_decrease(self) -> None:
        self.mutate_decision(2, lambda row: row.update(kyoku_ordinal=0))
        self.assertRejected()

    def test_focal_decision_ordinal_gap(self) -> None:
        def drop(rows):
            del rows[2]

        of.mutate_rows(self.path, 0, DECISIONS, drop)
        of.mutate_game_summary(
            self.path, 0, lambda summary: summary.update(decision_count=5)
        )
        self.assertRejected()

    # -- selection recomputation --------------------------------------------

    def test_token_tamper(self) -> None:
        def flip(row):
            value = int(row["exploration_token"], 16) ^ 1
            row["exploration_token"] = f"{value:064x}"

        self.mutate_decision(0, flip)
        self.assertRejected()

    def test_survivor_tamper(self) -> None:
        cases = (
            lambda row: row["producer_survivor_actions"].pop(),
            lambda row: row["producer_survivor_actions"].reverse(),
            lambda row: row.update(producer_survivor_actions=None),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(index=index):
                path = self.write(f"survivors-{index}")
                of.mutate_rows(path, 0, DECISIONS, lambda rows: mutate(rows[0]))
                self.assertRejected(path)

    def test_selected_action_tamper(self) -> None:
        def other_survivor(row):
            row["selected_action"] = next(
                action
                for action in row["producer_survivor_actions"]
                if action != row["selected_action"]
            )

        def guard_discard(row):
            row["selected_action"] = next(
                action for action in row["legal_actions"] if action["kind"] == "discard"
            )

        for index, (decision, mutate) in enumerate(
            ((0, other_survivor), (GUARD_INDEX, guard_discard))
        ):
            with self.subTest(index=index):
                path = self.write(f"selected-{index}")
                of.mutate_rows(path, 0, DECISIONS, lambda rows: mutate(rows[decision]))
                self.assertRejected(path)


class EnginePopulationTests(_EngineTestCase):
    def test_diagnostic_source_is_not_scientific_input(self) -> None:
        source = read_outcome_source(
            self.write("diagnostic", of.DIAGNOSTIC_GAMES, role="DIAGNOSTIC")
        )
        targets = build_outcome_targets(source)

        self.assertEqual(source.population_role, "DIAGNOSTIC")
        self.assertEqual(source.allocation_bindings, {})
        self.assertEqual(source.provenance()["population_role"], "DIAGNOSTIC")
        self.assertTrue(targets.rows)
        self.assertEqual({row.split for row in targets.rows}, {"DIAGNOSTIC"})

    def test_diagnostic_role_rejects_scientific_splits_and_bindings(self) -> None:
        for index, games in enumerate(
            (
                (("TRAIN", 1),),
                (("DIAGNOSTIC", 1), ("SELECT", 2)),
                (("CALIBRATION", 1),),
            )
        ):
            with self.subTest(games=games):
                path = self.write(f"diag-split-{index}", games, role="DIAGNOSTIC")
                self.assertRejected(path)

        path = self.write("diag-binding", of.DIAGNOSTIC_GAMES, role="DIAGNOSTIC")
        binding = fixtures.allocation_binding(
            [seed for _, seed in of.DIAGNOSTIC_GAMES],
            seed_domain=LISJONG_ENGINE_SEED_DOMAIN,
        )
        of.mutate_manifest(
            path,
            lambda body: body.update(allocation_bindings={"DIAGNOSTIC": binding}),
        )
        self.assertRejected(path, regex="DIAGNOSTIC")

    def test_scientific_roles_reject_diagnostic_split(self) -> None:
        for role in ("SCIENTIFIC", "CALIBRATION"):
            with self.subTest(role=role):
                path = self.write(f"sci-{role}", of.DIAGNOSTIC_GAMES, role=role)
                self.assertRejected(path)

    def test_scientific_relabel_of_diagnostic_source_is_rejected(self) -> None:
        path = self.write("relabel", of.DIAGNOSTIC_GAMES, role="DIAGNOSTIC")
        of.mutate_manifest(path, lambda body: body.update(population_role="SCIENTIFIC"))
        self.assertRejected(path)

    def test_scientific_source_requires_engine_seed_domain_bindings(self) -> None:
        cases = (
            lambda body: body.update(allocation_bindings={}),
            lambda body: body["allocation_bindings"].pop("SELECT"),
            lambda body: body["allocation_bindings"]["TRAIN"].update(
                seed_domain=fixtures.ALLOCATION_SEED_DOMAIN
            ),
            lambda body: body["allocation_bindings"]["SELECT"].update(
                seed_membership_identity="0" * 64
            ),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(index=index):
                path = self.write(f"binding-{index}")
                of.mutate_manifest(path, mutate)
                self.assertRejected(path)

    def test_calibration_source_requires_engine_seed_domain(self) -> None:
        path = self.write("calibration", of.CALIBRATION_GAMES, role="CALIBRATION")
        of.mutate_manifest(
            path,
            lambda body: body["allocation_bindings"]["CALIBRATION"].update(
                seed_domain="riichienv-4p-red-half-hanchan-v1"
            ),
        )
        self.assertRejected(path, regex="seed_domain")

    def test_focal_seat_rotation_and_unique_seeds(self) -> None:
        of.mutate_game_summary(
            self.path, 1, lambda summary: summary.update(focal_seat=0)
        )
        self.assertRejected()
        path = self.write("dup", (("TRAIN", 1001), ("SELECT", 1001)))
        self.assertRejected(path)

    def test_historical_source_rejects_diagnostic_role(self) -> None:
        path = of.write_outcome_source(
            self.root / "riichienv-diagnostic",
            of.DIAGNOSTIC_GAMES,
            role="DIAGNOSTIC",
        )
        self.assertRejected(path)

    def test_historical_seed_domain_validation_is_unchanged(self) -> None:
        """RiichiEnv lineageにはengine seed domainの制約を課さない。"""
        source = read_outcome_source(of.write_outcome_source(self.root / "riichienv"))
        self.assertEqual(
            {binding["seed_domain"] for binding in source.allocation_bindings.values()},
            {fixtures.ALLOCATION_SEED_DOMAIN},
        )
        self.assertNotEqual(fixtures.ALLOCATION_SEED_DOMAIN, LISJONG_ENGINE_SEED_DOMAIN)


if __name__ == "__main__":
    unittest.main()
