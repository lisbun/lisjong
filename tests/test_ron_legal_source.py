"""Synthetic source/replay contracts; native CI separately verifies real yaku."""

import copy
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_learning_hand_belief_source import (
    B_HANDS,
    B_PLAYERS,
    SEED,
    _player,
    make_decision,
    tiles,
    tt,
    write_source,
)

from lisjong.belief.ron_legal_ground_truth import (
    MissedRonState,
    RonSeatContext,
    actual_ron_is_legal,
    exact_hand_belief_with_ron,
    require_scoring_backend,
)
from lisjong.hand_evaluation.scoring import (
    RiichiStatus,
    ScoringBackendUnavailableError,
    WinSituation,
)
from lisjong.learning import ron_legal_source as source
from lisjong.learning._canonical import (
    canonical_json_line,
    canonical_json_text,
    file_digest,
)
from lisjong.learning._typed_values import (
    action_to_value,
    policy_input_to_value,
    tile_to_value,
)
from lisjong.learning.hand_belief_source import read_decisions
from lisjong.policy_contract import (
    Discard,
    DiscardAction,
    OwnHandState,
    PassAction,
    RiichiAction,
    RiichiState,
    RonAction,
    Seat,
    Wind,
)

NONE = RonSeatContext(MissedRonState.NONE, RiichiStatus.NONE, False)


def context_value(context):
    return {
        "missed_ron_state": context.missed_ron_state.value,
        "riichi_status": context.riichi_status.value,
        "is_ippatsu": context.is_ippatsu,
    }


class World:
    def __init__(self):
        self.hands = [
            tiles("45m678p567s888m44z"),
            tiles(B_HANDS[1][0]),
            tiles("12m567m111z99m234s"),
            tiles(B_HANDS[3][0]),
        ]
        self.drawn = [None] * 4
        self.players = [_player() for _ in range(4)]
        self.contexts = [NONE] * 4
        self.round = make_decision(
            0, "345m678p567s888m44z", B_PLAYERS
        ).policy_input.round

    def views(self):
        template = make_decision(
            0, "345m678p567s888m44z", tuple(self.players)
        ).policy_input
        return tuple(
            replace(
                template,
                self_seat=Seat(s),
                round=self.round,
                own_hand=OwnHandState(
                    self.hands[s] + (() if self.drawn[s] is None else (self.drawn[s],)),
                    self.drawn[s],
                ),
            )
            for s in range(4)
        )

    def checkpoint(self):
        return {
            "views": [
                policy_input_to_value(v, ValueError, "view") for v in self.views()
            ],
            "contexts": [context_value(c) for c in self.contexts],
        }

    def draw(self, seat, tile):
        self.drawn[seat] = tiles(tile)[0]
        self.round = replace(
            self.round,
            live_wall_tiles_remaining=self.round.live_wall_tiles_remaining - 1,
        )
        if self.contexts[seat].missed_ron_state is MissedRonState.TEMPORARY:
            self.contexts[seat] = replace(
                self.contexts[seat], missed_ron_state=MissedRonState.NONE
            )

    def discard(self, seat, tile):
        tile = tiles(tile)[0]
        hand = list(self.hands[seat]) + (
            [] if self.drawn[seat] is None else [self.drawn[seat]]
        )
        hand.remove(tile)
        self.hands[seat], self.drawn[seat] = tuple(hand), None
        river = self.players[seat].discards
        order = max((d.order for p in self.players for d in p.discards), default=-1) + 1
        self.players[seat] = replace(
            self.players[seat], discards=(*river, Discard(tile, True, order, None))
        )
        self.contexts[seat] = replace(self.contexts[seat], is_ippatsu=False)

    def decision(self, sequence, seat, tile):
        view = self.views()[seat]
        own = "345m678p567s888m44z"  # only the helper's valid legal-action seed
        action = DiscardAction(Seat(seat), tiles(tile)[0], True)
        return replace(
            make_decision(
                sequence,
                own,
                tuple(self.players),
                seat=seat,
                legal=(action,),
                selected=action,
            ),
            policy_input=view,
        )

    def facts(self, sequence, seat):
        return source.base.HandBeliefHandFacts(
            source.base.DecisionKey(SEED, sequence, seat),
            tuple(
                source.base.OpponentHand(
                    s, sequence, self.hands[s], self.players[s].melds
                )
                for s in range(4)
                if s != seat
            ),
        )


def fixture():
    world = World()
    history, decisions, hands, facts = [], [], [], []

    def record(event, selectors=()):
        history.append(
            {
                "schema": source.HISTORY_SCHEMA,
                "seed": SEED,
                "round_id": "round-0",
                "index": len(history),
                "steps": [
                    {
                        "selector_sequences": list(selectors),
                        "event": event,
                        "checkpoint": world.checkpoint(),
                    }
                ],
            }
        )

    def snapshot(seq, seat, tile):
        decisions.append(world.decision(seq, seat, tile))
        hands.append(world.facts(seq, seat))
        facts.append(
            {
                "schema": source.FACT_SCHEMA,
                "key": {"seed": SEED, "sequence": seq, "seat": seat},
                "round_id": "round-0",
                "history_boundary": len(history),
                "opponents": [
                    {
                        "seat": s,
                        "sequence": seq,
                        "context": context_value(world.contexts[s]),
                    }
                    for s in range(4)
                    if s != seat
                ],
            }
        )

    record({"kind": "round_start"})
    world.draw(0, "3m")
    record({"kind": "draw", "seat": 0, "draw_kind": "normal"})
    snapshot(0, 0, "3m")
    world.discard(0, "3m")
    record(
        {
            "kind": "progress",
            "sequence": 0,
            "action": action_to_value(
                decisions[0].selected_action, ValueError, "action"
            ),
        },
        (0,),
    )
    ron = RonAction(Seat.SEAT_2, Seat.SEAT_0, tiles("3m")[0])
    passed = PassAction(Seat.SEAT_2)
    proof = {
        "reaction_id": "reaction-0",
        "origin": "discard",
        "source_seat": 0,
        "winning_tile": tile_to_value(tiles("3m")[0], ValueError, "tile"),
        "discard_draw_kind": "normal",
        "candidates": [
            {
                "seat": s,
                "sequence": 1 if s == 2 else None,
                "legal_actions": [
                    action_to_value(a, ValueError, "action") for a in (ron, passed)
                ]
                if s == 2
                else [],
                "selected_action": action_to_value(passed, ValueError, "action")
                if s == 2
                else None,
            }
            for s in (1, 2, 3)
        ],
        "ron_capable": [2],
        "ron_selected": [],
        "ron_awarded": [],
        "ron_passed": [2],
        "resolution": "pass",
        "resolved_action": None,
    }
    world.contexts[2] = replace(NONE, missed_ron_state=MissedRonState.TEMPORARY)
    record({"kind": "reaction", "evidence": proof}, (1,))
    world.draw(1, "7p")
    record({"kind": "draw", "seat": 1, "draw_kind": "normal"})
    snapshot(2, 1, "7p")
    world.discard(1, "7p")
    record(
        {
            "kind": "progress",
            "sequence": 2,
            "action": action_to_value(
                decisions[1].selected_action, ValueError, "action"
            ),
        },
        (2,),
    )
    world.draw(2, "6z")
    record({"kind": "draw", "seat": 2, "draw_kind": "normal"})
    record({"kind": "round_result"})
    record({"kind": "round_end"})
    coverage = [
        {
            "seed": SEED,
            "round_id": "round-0",
            "transitions": len(history),
            "reactions": 1,
            "decisions": 2,
            "selectors": 3,
        }
    ]
    return decisions, hands, facts, history, coverage


class SourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / "base"
        self.extra = self.root / "extra"
        self.base.mkdir()
        self.extra.mkdir()
        self.decisions, self.hands, self.facts, self.history, self.coverage = fixture()
        # These temporal/codec tests stub only yaku and backend availability.
        # NativeTruthTest and native SourceTest exercise the real scorer in CI.
        self.backend = patch(
            "lisjong.belief.ron_legal_ground_truth.require_scoring_backend"
        )
        self.yaku = patch(
            "lisjong.belief.ron_legal_ground_truth._has_yaku", return_value=True
        )
        self.backend.start()
        self.yaku.start()
        self.addCleanup(self.backend.stop)
        self.addCleanup(self.yaku.stop)
        self.addCleanup(self.tmp.cleanup)

    def write(self):
        write_source(self.base, self.decisions, self.hands)
        files = {}
        for name, filename, rows in (
            ("ron_facts", source.FACTS_FILENAME, self.facts),
            ("ron_history", source.HISTORY_FILENAME, self.history),
        ):
            path = self.extra / filename
            path.write_text(
                "".join(canonical_json_line(row) for row in rows), encoding="utf-8"
            )
            files[name] = {**file_digest(path), "rows": len(rows)}
        base_manifest = source.base.read_manifest(self.base)
        self.manifest = {
            "schema": source.MANIFEST_SCHEMA,
            "base_manifest_sha256": file_digest(self.base / "manifest.json")["sha256"],
            "context_protocol": source.CONTEXT_PROTOCOL,
            "rules": source.RULES,
            "producer": base_manifest.producer,
            "splits": {k: list(v) for k, v in base_manifest.splits.items()},
            "files": files,
            "coverage": self.coverage,
        }
        self.manifest_path = self.extra / "manifest.json"
        self.manifest_path.write_text(
            canonical_json_text(self.manifest), encoding="utf-8"
        )

    def read(self):
        self.write()
        return source.read_ron_source(self.extra, base_directory=self.base)

    def assert_bad(self, pattern):
        with self.assertRaisesRegex(source.RonLegalSourceError, pattern):
            self.read()

    def test_prefix_snapshot_ignores_future_pass_and_future_clear(self):
        result = self.read()
        self.assertEqual(result.checkpoints[0].contexts[2], NONE)
        self.assertEqual(
            result.checkpoints[1].contexts[2].missed_ron_state, MissedRonState.TEMPORARY
        )
        replay, _, _ = source._replay(self.history, self.coverage)
        self.assertEqual(replay[SEED, "round-0"][-1].contexts[2], NONE)

    def test_temporary_is_not_cleared_by_another_player_draw_or_discard(self):
        self.history[4]["steps"][0]["checkpoint"]["contexts"][2] = context_value(NONE)
        self.assert_bad("replayed")

    def test_missing_facts_are_rejected_for_all_population(self):
        self.facts.pop()
        self.assert_bad("complete base population")

    def test_unknown_reason_and_missing_field_are_rejected(self):
        for mutate in (
            lambda c: c.pop("missed_ron_state"),
            lambda c: c.update(missed_ron_state="discard"),
            lambda c: c.update(is_ippatsu=1),
        ):
            with self.subTest(mutate=mutate):
                self.facts = fixture()[2]
                mutate(self.facts[0]["opponents"][0]["context"])
                self.assert_bad("context|ippatsu")

    def test_riichi_reason_without_established_riichi_is_rejected(self):
        self.facts[0]["opponents"][0]["context"]["missed_ron_state"] = "riichi"
        self.assert_bad("require established")

    def test_snapshot_boundary_and_same_sequence_are_exact(self):
        for mutate in (
            lambda f: f.update(history_boundary=3),
            lambda f: f["opponents"][0].update(sequence=1),
            lambda f: f.update(round_id="missing"),
        ):
            with self.subTest(mutate=mutate):
                self.facts = fixture()[2]
                mutate(self.facts[0])
                self.assert_bad("snapshot|boundary")

    def test_snapshot_context_cannot_incorporate_future(self):
        self.facts[0]["opponents"][1]["context"]["missed_ron_state"] = "temporary"
        self.assert_bad("half-open")

    def test_unknown_schema_noncontiguous_index_and_missing_end(self):
        for mutate in (
            lambda h: h[0].update(schema="unknown"),
            lambda h: h[3].update(index=5),
            lambda h: h[-1]["steps"][0]["event"].update(kind="draw"),
        ):
            with self.subTest(mutate=mutate):
                self.history = fixture()[3]
                mutate(self.history)
                self.assert_bad("contiguous|end checkpoint|terminal")

    def test_independent_coverage_mismatch_is_rejected(self):
        for field in ("transitions", "reactions", "decisions", "selectors"):
            with self.subTest(field=field):
                self.coverage = fixture()[4]
                self.coverage[0][field] += 1
                self.assert_bad("coverage|count")

    def test_opportunity_bool_and_pass_set_are_not_trusted(self):
        proof = self.history[3]["steps"][0]["event"]["evidence"]
        proof["ron_passed"] = []
        self.assert_bad("ron_passed")
        self.history = fixture()[3]
        proof = self.history[3]["steps"][0]["event"]["evidence"]
        proof["candidates"][1]["legal_actions"] = [
            proof["candidates"][1]["selected_action"]
        ]
        proof["ron_capable"] = []
        proof["ron_passed"] = []
        self.assert_bad("offered ron")

    def test_selected_ron_is_never_a_pass(self):
        proof = self.history[3]["steps"][0]["event"]["evidence"]
        proof["candidates"][1]["selected_action"] = proof["candidates"][1][
            "legal_actions"
        ][0]
        proof.update(ron_selected=[2], ron_awarded=[2], ron_passed=[], resolution="ron")
        before = source._checkpoint(self.history[2]["steps"][0]["checkpoint"], "before")
        contexts = source._reaction(
            proof, before, (1,), set(), {0: (tiles("3m")[0], "normal")}
        )
        self.assertEqual(contexts[2], NONE)

    def test_physical_history_and_snapshot_hands_cannot_differ(self):
        self.history[1]["steps"][0]["checkpoint"]["views"][1]["own_hand"][
            "concealed_tiles"
        ][0]["rank"] = 9
        self.assert_bad("hand|tiles|round-trip")

    def test_player_safe_reader_never_opens_extension(self):
        self.write()
        for path in self.extra.iterdir():
            path.unlink()
        _, decisions = read_decisions(self.base)
        self.assertEqual(decisions, tuple(self.decisions))

    def test_base_digest_and_rule_types_are_strict(self):
        self.write()
        for field, value in (
            ("base_manifest_sha256", "0" * 64),
            ("rules", {**source.RULES, "kuitan_enabled": 1}),
        ):
            manifest = copy.deepcopy(self.manifest)
            manifest[field] = value
            self.manifest_path.write_text(
                canonical_json_text(manifest), encoding="utf-8"
            )
            with self.assertRaises(source.RonLegalSourceError):
                source.read_ron_source(self.extra, base_directory=self.base)

    def test_digest_corruption_is_rejected(self):
        self.write()
        with (self.extra / source.HISTORY_FILENAME).open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(source.RonLegalSourceError, "digest"):
            source.read_ron_source(self.extra, base_directory=self.base)

    def test_duplicate_key_opponent_order_and_declared_population_are_rejected(self):
        self.facts.append(copy.deepcopy(self.facts[0]))
        self.assert_bad("duplicate")
        self.facts = fixture()[2]
        self.facts[0]["opponents"].reverse()
        self.assert_bad("seat order")
        self.facts = fixture()[2]
        self.write()
        self.manifest["splits"]["train"].append(SEED + 1)
        self.manifest_path.write_text(
            canonical_json_text(self.manifest), encoding="utf-8"
        )
        with self.assertRaisesRegex(source.RonLegalSourceError, "exact splits"):
            source.read_ron_source(self.extra, base_directory=self.base)

    def test_split_seeds_are_strict_non_negative_integers(self):
        self.write()
        name = next(k for k, v in self.manifest["splits"].items() if v)
        for value in (float(self.manifest["splits"][name][0]), True, -1):
            with self.subTest(value=value):
                manifest = copy.deepcopy(self.manifest)
                manifest["splits"][name][0] = value
                self.manifest_path.write_text(
                    canonical_json_text(manifest), encoding="utf-8"
                )
                with self.assertRaisesRegex(source.RonLegalSourceError, "split seed"):
                    source.read_ron_source(self.extra, base_directory=self.base)

    def test_selector_omission_and_snapshot_commit_action_are_rejected(self):
        self.history[3]["steps"][0]["selector_sequences"] = [10]
        self.assert_bad("omitted")
        self.history = fixture()[3]
        self.history[2]["steps"][0]["event"]["action"]["tsumogiri"] = False
        self.assert_bad("action")

    def test_actual_houtei_requires_normal_draw_not_just_empty_wall(self):
        proof = self.history[3]["steps"][0]["event"]["evidence"]
        before = source._checkpoint(self.history[2]["steps"][0]["checkpoint"], "before")
        views = tuple(
            replace(v, round=replace(v.round, live_wall_tiles_remaining=0))
            for v in before.views
        )
        before = replace(before, views=views)
        for draw_kind, expected in (
            ("normal", WinSituation.HOUTEI),
            ("rinshan", WinSituation.NORMAL),
            (None, WinSituation.NORMAL),
        ):
            with (
                self.subTest(draw_kind=draw_kind),
                patch.object(
                    source,
                    "actual_ron_is_legal",
                    side_effect=lambda *a, **k: bool(
                        a[0] == before.views[2].own_hand.concealed_tiles
                    ),
                ) as scorer,
            ):
                proof["discard_draw_kind"] = draw_kind
                source._reaction(
                    proof, before, (1,), set(), {0: (tiles("3m")[0], draw_kind)}
                )
                self.assertTrue(
                    all(
                        call.kwargs["situation"] is expected
                        for call in scorer.call_args_list
                    )
                )


def riichi_journal(*, pass_after_riichi=True):
    """Compound engine commit: reaction -> establishment; no middle snapshot."""
    world = World()
    history = []

    def step(event, selectors=()):
        return {
            "event": event,
            "selector_sequences": list(selectors),
            "checkpoint": world.checkpoint(),
        }

    def commit(*steps):
        history.append(
            {
                "schema": source.HISTORY_SCHEMA,
                "seed": SEED,
                "round_id": "riichi-round",
                "index": len(history),
                "steps": list(steps),
            }
        )

    def discard(seat, tile, sequence):
        world.discard(seat, tile)
        action = DiscardAction(Seat(seat), tiles(tile)[0], True)
        commit(
            step(
                {
                    "kind": "progress",
                    "sequence": sequence,
                    "action": action_to_value(action, ValueError, "action"),
                },
                (sequence,),
            )
        )

    def passed(source_seat, tile, winner, sequence, reaction_id):
        ron = RonAction(Seat(winner), Seat(source_seat), tiles(tile)[0])
        choice = PassAction(Seat(winner))
        proof = {
            "reaction_id": reaction_id,
            "origin": "discard",
            "source_seat": source_seat,
            "winning_tile": tile_to_value(tiles(tile)[0], ValueError, "tile"),
            "discard_draw_kind": "normal",
            "candidates": [
                {
                    "seat": s,
                    "sequence": sequence if s == winner else None,
                    "legal_actions": [
                        action_to_value(a, ValueError, "action") for a in (ron, choice)
                    ]
                    if s == winner
                    else [],
                    "selected_action": action_to_value(choice, ValueError, "action")
                    if s == winner
                    else None,
                }
                for s in range(4)
                if s != source_seat
            ],
            "ron_capable": [winner],
            "ron_selected": [],
            "ron_awarded": [],
            "ron_passed": [winner],
            "resolution": "pass",
            "resolved_action": None,
        }
        context = world.contexts[winner]
        world.contexts[winner] = replace(
            context,
            missed_ron_state=MissedRonState.TEMPORARY
            if context.riichi_status is RiichiStatus.NONE
            else MissedRonState.RIICHI,
        )
        return step({"kind": "reaction", "evidence": proof}, (sequence,))

    commit(step({"kind": "round_start"}))
    world.draw(2, "6z")
    commit(step({"kind": "draw", "seat": 2, "draw_kind": "normal"}))
    world.players[2] = replace(world.players[2], riichi=RiichiState.DECLARED)
    commit(
        step(
            {
                "kind": "progress",
                "sequence": 0,
                "action": action_to_value(
                    RiichiAction(Seat.SEAT_2), ValueError, "action"
                ),
            },
            (0,),
        )
    )
    discard(2, "6z", 1)
    reaction_step = passed(2, "6z", 3, 2, "declaration-reaction")
    world.players[2] = replace(world.players[2], riichi=RiichiState.ACCEPTED)
    world.contexts[2] = RonSeatContext(
        MissedRonState.NONE, RiichiStatus.DOUBLE_RIICHI, True
    )
    commit(
        reaction_step,
        step(
            {
                "kind": "riichi_established",
                "seat": 2,
                "riichi_status": "double_riichi",
                "is_ippatsu": True,
                "reaction_id": "declaration-reaction",
            }
        ),
    )
    if pass_after_riichi:
        world.draw(0, "3m")
        commit(step({"kind": "draw", "seat": 0, "draw_kind": "normal"}))
        discard(0, "3m", 3)
        commit(passed(0, "3m", 2, 4, "later-reaction"))
    world.draw(2, "3m")
    commit(step({"kind": "draw", "seat": 2, "draw_kind": "normal"}))
    discard(2, "3m", 5 if pass_after_riichi else 3)
    commit(step({"kind": "round_result"}))
    commit(step({"kind": "round_end"}))
    coverage = [
        {
            "seed": SEED,
            "round_id": "riichi-round",
            "transitions": len(history),
            "reactions": 2 if pass_after_riichi else 1,
            "decisions": 0,
            "selectors": 6 if pass_after_riichi else 4,
        }
    ]
    return history, coverage


class ContextReplayTest(unittest.TestCase):
    def setUp(self):
        backend = patch("lisjong.belief.ron_legal_ground_truth.require_scoring_backend")
        yaku = patch(
            "lisjong.belief.ron_legal_ground_truth._has_yaku", return_value=True
        )
        backend.start()
        yaku.start()
        self.addCleanup(backend.stop)
        self.addCleanup(yaku.stop)

    def test_compound_commit_establishes_riichi_at_outer_boundary(self):
        history, coverage = riichi_journal()
        states, _, _ = source._replay(history, coverage)
        states = states[SEED, "riichi-round"]
        self.assertEqual(states[3].contexts[2], NONE)
        self.assertEqual(
            states[4].contexts[2].riichi_status, RiichiStatus.DOUBLE_RIICHI
        )
        self.assertTrue(states[4].contexts[2].is_ippatsu)
        self.assertEqual(states[-1].contexts[2].missed_ron_state, MissedRonState.RIICHI)
        self.assertFalse(states[-1].contexts[2].is_ippatsu)

    def test_skipped_riichi_tsumo_does_not_create_a_missed_ron(self):
        history, coverage = riichi_journal(pass_after_riichi=False)
        states, _, _ = source._replay(history, coverage)
        self.assertEqual(
            states[SEED, "riichi-round"][-1].contexts[2].missed_ron_state,
            MissedRonState.NONE,
        )

    def test_permanent_reason_cannot_be_cleared_by_draw(self):
        history, coverage = riichi_journal()
        draw = [
            row
            for row in history
            if row["steps"][0]["event"]
            == {"kind": "draw", "seat": 2, "draw_kind": "normal"}
        ][-1]
        draw["steps"][0]["checkpoint"]["contexts"][2]["missed_ron_state"] = "none"
        with self.assertRaisesRegex(source.RonLegalSourceError, "replayed"):
            source._replay(history, coverage)

    def test_establishment_requires_its_actual_reaction(self):
        history, coverage = riichi_journal()
        history[4]["steps"][1]["event"]["reaction_id"] = "unknown"
        with self.assertRaisesRegex(source.RonLegalSourceError, "declaration reaction"):
            source._replay(history, coverage)

    def test_declaration_tile_ron_cancels_without_establishing_riichi(self):
        history, coverage = riichi_journal()
        history = history[:5]
        reaction, cancellation = history[4]["steps"]
        proof = reaction["event"]["evidence"]
        candidate = proof["candidates"][-1]
        candidate["selected_action"] = candidate["legal_actions"][0]
        proof.update(ron_selected=[3], ron_awarded=[3], ron_passed=[], resolution="ron")
        reaction["checkpoint"]["contexts"][3] = context_value(NONE)
        cancellation["event"] = {
            "kind": "riichi_cancelled",
            "seat": 2,
            "reaction_id": "declaration-reaction",
        }
        cancellation["checkpoint"]["contexts"][2] = context_value(NONE)
        cancellation["checkpoint"]["contexts"][3] = context_value(NONE)
        for view in cancellation["checkpoint"]["views"]:
            view["players"][2]["riichi"] = "none"
        for kind in ("round_result", "round_end"):
            history.append(
                {
                    "schema": source.HISTORY_SCHEMA,
                    "seed": SEED,
                    "round_id": "riichi-round",
                    "index": len(history),
                    "steps": [
                        {
                            "selector_sequences": [],
                            "event": {"kind": kind},
                            "checkpoint": copy.deepcopy(cancellation["checkpoint"]),
                        }
                    ],
                }
            )
        coverage[0].update(transitions=len(history), reactions=1, selectors=3)
        states, _, _ = source._replay(history, coverage)
        self.assertEqual(states[SEED, "riichi-round"][-1].contexts[2], NONE)


class BackendTest(unittest.TestCase):
    def test_scoring_errors_are_never_negative_labels(self):
        with (
            patch("lisjong.belief.ron_legal_ground_truth.require_scoring_backend"),
            patch(
                "lisjong.belief.ron_legal_ground_truth.evaluate_win",
                side_effect=ValueError("invalid scoring input"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "invalid scoring input"):
                exact_hand_belief_with_ron(
                    tiles("789m456p789s11z23m"),
                    (),
                    (),
                    NONE,
                    seat_wind=Wind.SOUTH,
                    prevailing_wind=Wind.EAST,
                )

    def test_unavailable_backend_cannot_become_all_zero_truth(self):
        with patch(
            "lisjong.belief.ron_legal_ground_truth._load_native",
            side_effect=ScoringBackendUnavailableError("missing"),
        ):
            with self.assertRaises(ScoringBackendUnavailableError):
                exact_hand_belief_with_ron(
                    tiles("13579m13579p135s"),
                    (),
                    (),
                    NONE,
                    seat_wind=Wind.SOUTH,
                    prevailing_wind=Wind.EAST,
                )


try:
    require_scoring_backend()
    HAS_NATIVE = True
except ScoringBackendUnavailableError:
    HAS_NATIVE = False


@unittest.skipUnless(
    HAS_NATIVE or os.environ.get("LISJONG_REQUIRE_NATIVE") == "1",
    "native scorer unavailable",
)
class NativeTruthTest(unittest.TestCase):
    def truth(self, hand, *, melds=(), river="", context=NONE):
        return exact_hand_belief_with_ron(
            tiles(hand),
            melds,
            tiles(river),
            context,
            seat_wind=Wind.SOUTH,
            prevailing_wind=Wind.EAST,
        )

    def test_menzen_no_yaku_and_riichi_and_declared(self):
        hand = "789m456p789s11z23m"
        self.assertEqual(sum(self.truth(hand).ron_legal_probability_raw), 0)
        riichi = replace(NONE, riichi_status=RiichiStatus.RIICHI)
        truth = self.truth(hand, context=riichi)
        self.assertEqual(truth.ron_legal_probability_raw, truth.wait_probability_raw)

    def test_open_yaku_and_no_yaku(self):
        truth = self.truth(B_HANDS[2][0], melds=B_HANDS[2][1])
        self.assertEqual(truth.ron_legal_probability(tt("3m")), 1)
        from test_learning_hand_belief_source import chi

        no_yaku = self.truth("12m567m333z99m", melds=(chi("234s", "2s", Seat.SEAT_1),))
        self.assertEqual(sum(no_yaku.ron_legal_probability_raw), 0)

    def test_one_river_wait_blocks_all_waits_including_called_tile(self):
        riichi = replace(NONE, riichi_status=RiichiStatus.RIICHI)
        truth = self.truth("123m406p789s11z23m", river="1m", context=riichi)
        self.assertGreater(sum(truth.wait_probability_raw), 0)
        self.assertEqual(sum(truth.ron_legal_probability_raw), 0)

    def test_missed_temporary_and_riichi_block_all_waits(self):
        for reason in (MissedRonState.TEMPORARY, MissedRonState.RIICHI):
            context = RonSeatContext(reason, RiichiStatus.RIICHI, True)
            self.assertEqual(
                sum(
                    self.truth(
                        "123m456p789s11z23m", context=context
                    ).ron_legal_probability_raw
                ),
                0,
            )

    def test_native_source_labels_before_and_after_pass(self):
        test = SourceTest()
        test.setUp()
        try:
            test.backend.stop()
            test.yaku.stop()
            test.write()
            _, rows = source.read_labelled_ron_source(
                test.extra, base_directory=test.base
            )
            self.assertEqual(
                rows[0].opponents[1].truth.ron_legal_probability(tt("3m")), 1
            )
            self.assertEqual(
                rows[1].opponents[1].truth.ron_legal_probability(tt("3m")), 0
            )
        finally:
            test.doCleanups()

    def test_no_yaku_pass_does_not_set_furiten_and_special_history_can(self):
        hand = tiles("789m456p789s11z23m")
        for situation, expected in (
            (WinSituation.NORMAL, False),
            (WinSituation.HOUTEI, True),
            (WinSituation.CHANKAN, True),
        ):
            self.assertEqual(
                actual_ron_is_legal(
                    hand,
                    (),
                    (),
                    NONE,
                    winning_tile=tiles("4m")[0],
                    seat_wind=Wind.SOUTH,
                    prevailing_wind=Wind.EAST,
                    situation=situation,
                ),
                expected,
            )

    def test_red_normal_invariant_and_three_plain_fives_choose_red(self):
        for hand in ("345m567m456p111z5m", "340m567m456p111z5m"):
            self.assertEqual(self.truth(hand).ron_legal_probability(tt("5m")), 1)
        with self.assertRaisesRegex(ValueError, "physical inventory"):
            actual_ron_is_legal(
                tiles("345m567m456p111z5m"),
                (),
                (),
                NONE,
                winning_tile=tiles("5m")[0],
                seat_wind=Wind.SOUTH,
                prevailing_wind=Wind.EAST,
                situation=WinSituation.NORMAL,
            )

    def test_exhausted_public_type_still_has_counterfactual_truth(self):
        # Three own 1m plus a publicly visible fourth (e.g. dora indicator)
        # must still permit the counterfactual 1m. Yaku projection excludes dora.
        truth = self.truth(
            "111m23m456p789s55z",
            context=replace(NONE, riichi_status=RiichiStatus.RIICHI),
        )
        self.assertEqual(truth.ron_legal_probability(tt("1m")), 1)

    def test_real_scoring_compound_riichi_history(self):
        source._replay(*riichi_journal())
