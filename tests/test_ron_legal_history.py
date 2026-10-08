"""Additional history and online information-boundary regressions for #262."""

import json
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_ron_legal_source import HAS_NATIVE, NONE, SEED, World, riichi_journal, tiles

from lisjong.belief.ron_legal_ground_truth import MissedRonState
from lisjong.learning import ron_legal_source as source
from lisjong.learning._typed_values import (
    action_to_value,
    policy_input_to_value,
    tile_to_value,
)
from lisjong.policy_contract import (
    DaiminkanAction,
    DiscardAction,
    KakanAction,
    MeldKind,
    PassAction,
    PonAction,
    PublicMeld,
    RonAction,
    Seat,
)


def daiminkan_journal():
    world = World()
    world.hands[0] = tiles("888m67m234p456s55p")
    history = []

    def record(event, selectors=()):
        history.append(
            {
                "schema": source.HISTORY_SCHEMA,
                "seed": SEED,
                "round_id": "kan-round",
                "index": len(history),
                "steps": [
                    {
                        "event": event,
                        "selector_sequences": list(selectors),
                        "checkpoint": world.checkpoint(),
                    }
                ],
            }
        )

    record({"kind": "round_start"})
    world.draw(1, "8m")
    record({"kind": "draw", "seat": 1, "draw_kind": "normal"})
    world.discard(1, "8m")
    record(
        {
            "kind": "progress",
            "sequence": 0,
            "action": action_to_value(
                DiscardAction(Seat.SEAT_1, tiles("8m")[0], True), ValueError, "action"
            ),
        },
        (0,),
    )
    ron = RonAction(Seat.SEAT_0, Seat.SEAT_1, tiles("8m")[0])
    call = DaiminkanAction(Seat.SEAT_0, Seat.SEAT_1, tiles("8m")[0], tiles("888m"))
    action = action_to_value(call, ValueError, "call")
    proof = {
        "reaction_id": "kan-reaction",
        "origin": "discard",
        "source_seat": 1,
        "winning_tile": tile_to_value(tiles("8m")[0], ValueError, "tile"),
        "discard_draw_kind": "normal",
        "candidates": [
            {
                "seat": s,
                "sequence": 1 if s == 0 else None,
                "legal_actions": [
                    action_to_value(a, ValueError, "action")
                    for a in (ron, PassAction(Seat.SEAT_0), call)
                ]
                if s == 0
                else [],
                "selected_action": action if s == 0 else None,
            }
            for s in (0, 2, 3)
        ],
        "ron_capable": [0],
        "ron_selected": [],
        "ron_awarded": [],
        "ron_passed": [0],
        "resolution": "call",
        "resolved_action": action,
    }
    world.contexts[0] = replace(NONE, missed_ron_state=MissedRonState.TEMPORARY)
    world.hands[0] = tiles("67m234p456s55p")
    world.players[0] = replace(
        world.players[0],
        melds=(
            PublicMeld(MeldKind.DAIMINKAN, tiles("8888m"), Seat.SEAT_1, tiles("8m")[0]),
        ),
    )
    river = world.players[1].discards
    world.players[1] = replace(
        world.players[1],
        discards=(*river[:-1], replace(river[-1], called_by=Seat.SEAT_0)),
    )
    record({"kind": "reaction", "evidence": proof}, (1,))
    world.draw(0, "6z")
    record({"kind": "draw", "seat": 0, "draw_kind": "rinshan"})
    record({"kind": "round_result"})
    record({"kind": "round_end"})
    coverage = [
        {
            "seed": SEED,
            "round_id": "kan-round",
            "transitions": len(history),
            "reactions": 1,
            "decisions": 0,
            "selectors": 2,
        }
    ]
    return history, coverage


def kakan_journal():
    """Pon, then kakan with the held fourth tile: declaration -> chankan -> kan."""
    world = World()
    world.hands[0] = tiles("888m67m234p456s55p")
    history = []

    def record(event, selectors=()):
        history.append(
            {
                "schema": source.HISTORY_SCHEMA,
                "seed": SEED,
                "round_id": "kakan-round",
                "index": len(history),
                "steps": [
                    {
                        "event": event,
                        "selector_sequences": list(selectors),
                        "checkpoint": world.checkpoint(),
                    }
                ],
            }
        )

    def value(action):
        return action_to_value(action, ValueError, "action")

    record({"kind": "round_start"})
    world.draw(1, "8m")
    record({"kind": "draw", "seat": 1, "draw_kind": "normal"})
    world.discard(1, "8m")
    discard = DiscardAction(Seat.SEAT_1, tiles("8m")[0], True)
    record({"kind": "progress", "sequence": 0, "action": value(discard)}, (0,))
    eight = tiles("8m")[0]
    ron = RonAction(Seat.SEAT_0, Seat.SEAT_1, eight)
    pon = PonAction(Seat.SEAT_0, Seat.SEAT_1, eight, tiles("88m"))
    pon_reaction = {
        "kind": "reaction",
        "evidence": {
            "reaction_id": "pon",
            "origin": "discard",
            "source_seat": 1,
            "winning_tile": tile_to_value(eight, ValueError, "tile"),
            "discard_draw_kind": "normal",
            "candidates": [
                {
                    "seat": s,
                    "sequence": 1 if s == 0 else None,
                    "legal_actions": [
                        value(a) for a in (ron, PassAction(Seat.SEAT_0), pon)
                    ]
                    if s == 0
                    else [],
                    "selected_action": value(pon) if s == 0 else None,
                }
                for s in (0, 2, 3)
            ],
            "ron_capable": [0],
            "ron_selected": [],
            "ron_awarded": [],
            "ron_passed": [0],
            "resolution": "call",
            "resolved_action": value(pon),
        },
    }
    world.contexts[0] = replace(NONE, missed_ron_state=MissedRonState.TEMPORARY)
    world.hands[0] = tiles("8m67m234p456s55p")
    pon_meld = PublicMeld(MeldKind.PON, tiles("888m"), Seat.SEAT_1, eight)
    world.players[0] = replace(world.players[0], melds=(pon_meld,))
    river = world.players[1].discards
    world.players[1] = replace(
        world.players[1],
        discards=(*river[:-1], replace(river[-1], called_by=Seat.SEAT_0)),
    )
    record(pon_reaction, (1,))
    world.discard(0, "6m")
    discard = DiscardAction(Seat.SEAT_0, tiles("6m")[0], False)
    world.players[0] = replace(
        world.players[0],
        discards=(replace(world.players[0].discards[-1], tsumogiri=False),),
    )
    record({"kind": "progress", "sequence": 2, "action": value(discard)}, (2,))
    world.draw(0, "6z")
    record({"kind": "draw", "seat": 0, "draw_kind": "normal"})
    kakan = KakanAction(Seat.SEAT_0, eight, Seat.SEAT_1, eight)
    record({"kind": "progress", "sequence": 3, "action": value(kakan)}, (3,))
    record(
        {
            "kind": "reaction",
            "evidence": {
                "reaction_id": "chankan",
                "origin": "kakan",
                "source_seat": 0,
                "winning_tile": tile_to_value(eight, ValueError, "tile"),
                "discard_draw_kind": None,
                "candidates": [
                    {
                        "seat": s,
                        "sequence": None,
                        "legal_actions": [],
                        "selected_action": None,
                    }
                    for s in (1, 2, 3)
                ],
                "ron_capable": [],
                "ron_selected": [],
                "ron_awarded": [],
                "ron_passed": [],
                "resolution": "pass",
                "resolved_action": None,
            },
        }
    )
    hand = list(world.hands[0]) + [world.drawn[0]]
    hand.remove(eight)
    world.hands[0], world.drawn[0] = tuple(hand), None
    world.players[0] = replace(
        world.players[0],
        melds=(PublicMeld(MeldKind.KAKAN, tiles("8888m"), Seat.SEAT_1, eight),),
    )
    record({"kind": "kan_confirmed", "seat": 0})
    world.draw(0, "7z")
    record({"kind": "draw", "seat": 0, "draw_kind": "rinshan"})
    record({"kind": "round_result"})
    record({"kind": "round_end"})
    coverage = [
        {
            "seed": SEED,
            "round_id": "kakan-round",
            "transitions": len(history),
            "reactions": 2,
            "decisions": 0,
            "selectors": 4,
        }
    ]
    return history, coverage


class TemporalTest(unittest.TestCase):
    def setUp(self):
        backend = patch("lisjong.belief.ron_legal_ground_truth.require_scoring_backend")
        yaku = patch(
            "lisjong.belief.ron_legal_ground_truth._has_yaku", return_value=True
        )
        backend.start()
        yaku.start()
        self.addCleanup(backend.stop)
        self.addCleanup(yaku.stop)

    def test_daiminkan_keeps_temporary_until_actual_rinshan_draw(self):
        states, _, _ = source._replay(*daiminkan_journal())
        self.assertEqual(
            states[SEED, "kan-round"][3].contexts[0].missed_ron_state,
            MissedRonState.TEMPORARY,
        )
        self.assertEqual(
            states[SEED, "kan-round"][4].contexts[0].missed_ron_state,
            MissedRonState.NONE,
        )

    def test_call_cannot_clear_missed_ron_early(self):
        history, coverage = daiminkan_journal()
        history[3]["steps"][0]["checkpoint"]["contexts"][0]["missed_ron_state"] = "none"
        with self.assertRaisesRegex(source.RonLegalSourceError, "replayed"):
            source._replay(history, coverage)

    def kakan_variant(self, edit, reactions=2):
        history, coverage = kakan_journal()
        edit(history)
        for index, row in enumerate(history):
            row["index"] = index
        coverage[0].update(transitions=len(history), reactions=reactions)
        return history, coverage

    def test_kakan_declaration_chankan_reaction_and_kan_are_bound(self):
        states, _, _ = source._replay(*kakan_journal())
        self.assertEqual(
            states[SEED, "kakan-round"][8].views[0].players[0].melds[0].kind,
            MeldKind.KAKAN,
        )

    def test_kakan_without_its_chankan_reaction_is_rejected(self):
        with self.assertRaisesRegex(source.RonLegalSourceError, "chankan reaction"):
            source._replay(*self.kakan_variant(lambda h: h.pop(7), reactions=1))

    def test_chankan_reaction_on_another_tile_is_rejected(self):
        def edit(history):
            evidence = history[7]["steps"][0]["event"]["evidence"]
            evidence["winning_tile"] = tile_to_value(tiles("7m")[0], ValueError, "t")

        with self.assertRaisesRegex(source.RonLegalSourceError, "added tile"):
            source._replay(*self.kakan_variant(edit))

    def test_duplicate_chankan_reaction_is_rejected(self):
        def edit(history):
            again = json.loads(json.dumps(history[7]))
            again["steps"][0]["event"]["evidence"]["reaction_id"] = "chankan-2"
            history.insert(8, again)

        with self.assertRaisesRegex(source.RonLegalSourceError, "duplicate reaction"):
            source._replay(*self.kakan_variant(edit, reactions=3))

    def test_kan_reaction_needs_a_matching_pending_declaration(self):
        def wrong_origin(history):
            history[7]["steps"][0]["event"]["evidence"]["origin"] = "ankan"

        def wrong_seat(history):
            history[7]["steps"][0]["event"]["evidence"]["source_seat"] = 1

        for edit, pattern in (
            (wrong_origin, "origin"),
            (wrong_seat, "pending declaration"),
        ):
            with (
                self.subTest(pattern),
                self.assertRaisesRegex(source.RonLegalSourceError, pattern),
            ):
                source._replay(*self.kakan_variant(edit))

    def test_play_cannot_continue_before_the_pending_kan_resolves(self):
        def edit(history):
            # the rinshan draw moved before the chankan reaction
            history.insert(7, history.pop(9))

        with self.assertRaisesRegex(source.RonLegalSourceError, "pending kan"):
            source._replay(*self.kakan_variant(edit))

    def test_round_reset_has_no_inherited_permanent_furiten(self):
        history, coverage = riichi_journal()
        next_history, next_coverage = daiminkan_journal()
        for row in next_history:
            for step in row["steps"]:
                step["selector_sequences"] = [
                    seq + 6 for seq in step["selector_sequences"]
                ]
                event = step["event"]
                if event["kind"] == "progress":
                    event["sequence"] += 6
                if event["kind"] == "reaction":
                    for candidate in event["evidence"]["candidates"]:
                        if candidate["sequence"] is not None:
                            candidate["sequence"] += 6
        states, _, _ = source._replay(history + next_history, coverage + next_coverage)
        self.assertEqual(states[SEED, "kan-round"][0].contexts, (NONE,) * 4)

    def test_round_start_cannot_reset_inherited_reason_mid_round(self):
        history, coverage = riichi_journal()
        history[5]["steps"][0]["event"] = {"kind": "round_start"}
        with self.assertRaises(source.RonLegalSourceError):
            source._replay(history, coverage)


@unittest.skipUnless(
    HAS_NATIVE or os.environ.get("LISJONG_REQUIRE_NATIVE") == "1",
    "native scorer unavailable",
)
class NativeHistoryTest(unittest.TestCase):
    def test_real_scoring_daiminkan_then_rinshan(self):
        source._replay(*daiminkan_journal())

    def test_real_scoring_pon_then_kakan_with_chankan_window(self):
        source._replay(*kakan_journal())

    def test_three_selected_ron_keep_engine_e2_awards_and_are_not_passes(self):
        world = World()
        world.hands = [
            tiles("234m456m678p345s5p"),
            *(tiles("19m19p19s1234567z") for _ in range(3)),
        ]
        world.draw(0, "1z")
        world.discard(0, "1z")
        before = source._checkpoint(world.checkpoint(), "triple ron")
        proof = {
            "reaction_id": "triple",
            "origin": "discard",
            "source_seat": 0,
            "winning_tile": tile_to_value(tiles("1z")[0], ValueError, "tile"),
            "discard_draw_kind": "normal",
            "candidates": [],
            "ron_capable": [1, 2, 3],
            "ron_selected": [1, 2, 3],
            "ron_awarded": [1, 2, 3],
            "ron_passed": [],
            "resolution": "ron",
            "resolved_action": None,
        }
        for seat in (1, 2, 3):
            action = action_to_value(
                RonAction(Seat(seat), Seat.SEAT_0, tiles("1z")[0]), ValueError, "action"
            )
            proof["candidates"].append(
                {
                    "seat": seat,
                    "sequence": seat,
                    "legal_actions": [
                        action,
                        action_to_value(PassAction(Seat(seat)), ValueError, "action"),
                    ],
                    "selected_action": action,
                }
            )
        result = source._reaction(
            proof, before, (1, 2, 3), set(), {0: (tiles("1z")[0], "normal")}
        )
        self.assertEqual(result, (NONE,) * 4)
        proof["ron_awarded"] = []
        with self.assertRaisesRegex(source.RonLegalSourceError, "award/resolution"):
            source._reaction(
                proof, before, (1, 2, 3), set(), {0: (tiles("1z")[0], "normal")}
            )


class OnlineBoundaryTest(unittest.TestCase):
    def test_online_estimation_imports_no_ron_truth_or_source(self):
        program = """
import importlib.abc, sys, json
class DenyTruth(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('lisjong.belief.ron_legal_ground_truth', 'lisjong.learning.ron_legal_source', 'lisjong.learning.hand_belief_source'):
            raise ImportError('privileged truth disabled')
sys.meta_path.insert(0, DenyTruth())
from lisjong.learning._typed_values import parse_policy_input
from lisjong.belief.conditional_uniform_hand_belief import estimate_conditional_uniform_hand_belief
observation = parse_policy_input(json.loads(sys.stdin.read()), ValueError, 'observation')
slots = tuple(0 if seat == observation.self_seat else 13 - 3 * len(player.melds) for seat, player in enumerate(observation.players))
result = estimate_conditional_uniform_hand_belief(observation, slots)
assert result is not None
"""
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent.resolve()))
        result = subprocess.run(
            [sys.executable, "-c", program],
            env=env,
            capture_output=True,
            text=True,
            input=json.dumps(
                policy_input_to_value(World().views()[0], ValueError, "view")
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
