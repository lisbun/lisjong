import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

from lisjong.belief.riichi_ron_label import RiichiFuritenReason
from lisjong.learning._canonical import canonical_json_line, file_digest
from lisjong.learning.riichi_deal_in_source import (
    DECISIONS_FILENAME,
    LABEL_FACTS_FILENAME,
    MANIFEST_FILENAME,
    CandidateLabel,
    DecisionKey,
    RiichiDealInDecision,
    RiichiDealInLabelFacts,
    RiichiDealInSourceError,
    WinOption,
    decision_to_value,
    label_decisions,
    label_facts_to_value,
    manifest_text,
    read_decisions,
    read_labelled_source,
)
from lisjong.policy_contract.action import (
    DiscardAction,
    PassAction,
)
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

_CATEGORIES = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}
S, R = Seat.SEAT_0, Seat.SEAT_1


def tiles(spec: str) -> tuple[Tile, ...]:
    out, ranks = [], ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        out += [Tile(TileType(_CATEGORIES[character], int(rank))) for rank in ranks]
        ranks = ""
    return tuple(out)


def tt(spec: str) -> TileType:
    return tiles(spec)[0].tile_type


# リーチ者の打牌後13枚: 123m 456p 789s 11z 23m（待ち 1m / 4m）
R_HAND = tiles("123m456p789s11z23m")
KEY = DecisionKey(seed=931000, sequence=20, seat=0)


def _player(riichi=RiichiState.NONE, discards=""):
    return PlayerPublicState(
        score=25000,
        discards=tuple(
            Discard(tile=tile, tsumogiri=False, order=order, called_by=None)
            for order, tile in enumerate(tiles(discards))
        ),
        melds=(),
        riichi=riichi,
    )


def make_decision(
    selected="5z",
    *,
    r_discards="9p",
    self_riichi=RiichiState.NONE,
    x_riichi=RiichiState.NONE,
    key=KEY,
    concealed="14m5z",
):
    hand = tiles(concealed)
    policy_input = PolicyInput(
        self_seat=S,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=1,
            dora_indicators=(),
            live_wall_tiles_remaining=40,
        ),
        players=(
            _player(self_riichi),
            _player(RiichiState.ACCEPTED, r_discards),
            _player(x_riichi),
            _player(),
        ),
        own_hand=OwnHandState(concealed_tiles=hand, drawn_tile=None),
    )
    legal = tuple(
        DiscardAction(actor=S, tile=tile, tsumogiri=False)
        for tile in dict.fromkeys(hand)
    )
    selected_action = next(a for a in legal if a.tile.tile_type == tt(selected))
    return RiichiDealInDecision(
        key=key,
        policy_input=policy_input,
        legal_actions=legal,
        selected_action=selected_action,
    )


def make_facts(
    *,
    key=KEY,
    riichi_seat=1,
    declared=5,
    hand_sequence=5,
    concealed=R_HAND,
    melds=(),
    win_options=(),
    ron_offered=False,
    dealt_in=False,
):
    return RiichiDealInLabelFacts(
        key=key,
        riichi_seat=riichi_seat,
        riichi_declared_sequence=declared,
        hand_sequence=hand_sequence,
        concealed_tiles=concealed,
        melds=melds,
        win_options=tuple(win_options),
        ron_offered=ron_offered,
        dealt_in=dealt_in,
    )


def labels_of(labelled):
    return {c.tile_type: c.label_a for c in labelled.candidates}


class LabelDecisionsTest(unittest.TestCase):
    def test_waits_are_positive_and_others_negative(self):
        (labelled,) = label_decisions([make_decision()], [make_facts()])
        self.assertEqual(
            labels_of(labelled),
            {tt("1m"): True, tt("4m"): True, tt("5z"): False},
        )
        self.assertEqual(labelled.furiten_reasons, frozenset())
        self.assertFalse(labelled.selected_ron_offered)

    def test_only_the_selected_discard_carries_engine_facts(self):
        self.assertEqual(
            [field.name for field in fields(CandidateLabel)], ["tile_type", "label_a"]
        )

    def test_selected_wait_must_match_the_engine_offer(self):
        (labelled,) = label_decisions(
            [make_decision("1m")], [make_facts(ron_offered=True, dealt_in=True)]
        )
        self.assertTrue(labelled.selected_dealt_in)
        with self.assertRaises(RiichiDealInSourceError):
            label_decisions([make_decision("1m")], [make_facts(ron_offered=False)])
        with self.assertRaises(RiichiDealInSourceError):
            label_decisions([make_decision("5z")], [make_facts(ron_offered=True)])
        with self.assertRaises(RiichiDealInSourceError):
            label_decisions([make_decision("5z")], [make_facts(dealt_in=True)])

    def test_own_discard_furiten_is_computed_from_the_river(self):
        (labelled,) = label_decisions(
            [make_decision(r_discards="9p4m")], [make_facts()]
        )
        self.assertFalse(any(labels_of(labelled).values()))
        self.assertEqual(
            labelled.furiten_reasons, frozenset({RiichiFuritenReason.OWN_DISCARD})
        )

    def test_passed_win_options_cause_furiten(self):
        for option in (
            WinOption(
                sequence=8,
                winning_tiles=tiles("4m"),
                selected_action=PassAction(actor=R),
            ),
            WinOption(
                sequence=8,
                winning_tiles=tiles("1m"),
                selected_action=DiscardAction(
                    actor=R, tile=tiles("1m")[0], tsumogiri=True
                ),
            ),
        ):
            (labelled,) = label_decisions(
                [make_decision()], [make_facts(win_options=[option])]
            )
            self.assertFalse(any(labels_of(labelled).values()))
            self.assertEqual(
                labelled.furiten_reasons,
                frozenset({RiichiFuritenReason.PASSED_AFTER_RIICHI}),
            )

    def test_win_options_and_hand_must_precede_the_decision(self):
        option = WinOption(
            sequence=KEY.sequence,
            winning_tiles=tiles("4m"),
            selected_action=PassAction(actor=R),
        )
        for facts in (
            make_facts(win_options=[option]),
            make_facts(hand_sequence=KEY.sequence),
            make_facts(
                win_options=[replace(option, sequence=4)]
            ),  # 宣言打牌より前の見逃しは窓の外
            make_facts(declared=6, hand_sequence=5),
        ):
            with self.assertRaises(RiichiDealInSourceError):
                label_decisions([make_decision()], [facts])

    def test_ankan_after_riichi_uses_the_13_equivalent_hand(self):
        meld = PublicMeld(
            kind=MeldKind.ANKAN, tiles=tiles("7777s"), from_seat=None, called_tile=None
        )
        (labelled,) = label_decisions(
            [make_decision()],
            [make_facts(concealed=tiles("123m456p11z23m"), melds=(meld,))],
        )
        self.assertTrue(labels_of(labelled)[tt("1m")])

    def test_keys_must_match_exactly(self):
        other = DecisionKey(seed=931000, sequence=21, seat=0)
        for decisions, facts in (
            ([make_decision()], [make_facts(key=other)]),
            ([make_decision()], [make_facts(), make_facts()]),
            ([make_decision(), make_decision()], [make_facts()]),
            ([make_decision()], []),
        ):
            with self.assertRaises(RiichiDealInSourceError):
                label_decisions(decisions, facts)

    def test_scope_and_riichi_seat_are_checked(self):
        for decision, facts in (
            (make_decision(self_riichi=RiichiState.ACCEPTED), make_facts()),
            (make_decision(x_riichi=RiichiState.ACCEPTED), make_facts()),
            (make_decision(), make_facts(riichi_seat=2)),
        ):
            with self.assertRaises(RiichiDealInSourceError):
                label_decisions([decision], [facts])


def write_source(root: Path, decisions, facts, *, splits=None):
    lines = {
        DECISIONS_FILENAME: [decision_to_value(d) for d in decisions],
        LABEL_FACTS_FILENAME: [label_facts_to_value(f) for f in facts],
    }
    files = {}
    for filename, rows in lines.items():
        path = root / filename
        path.write_text(
            "".join(canonical_json_line(row) for row in rows), encoding="utf-8"
        )
        name = "decisions" if filename == DECISIONS_FILENAME else "label_facts"
        files[name] = {**file_digest(path), "rows": len(rows)}
    (root / MANIFEST_FILENAME).write_text(
        manifest_text(
            producer={
                "arena_revision": "a" * 40,
                "lisjong_revision": "b" * 40,
                "lisjong_engine_revision": "c" * 40,
                "policy": "PlacementAwareSpeedCallPolicy",
            },
            splits=splits or {"train": [931000], "valid": [], "test": []},
            files=files,
        ),
        encoding="utf-8",
    )


class SourceFilesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_and_separate_reading_paths(self):
        write_source(self.root, [make_decision()], [make_facts()])
        manifest, labelled = read_labelled_source(self.root)
        self.assertEqual(manifest.split_of(931000), "train")
        self.assertEqual(len(labelled), 1)
        (self.root / LABEL_FACTS_FILENAME).unlink()
        _, decisions = read_decisions(self.root)  # 推論入力の経路はラベルを読まない
        self.assertEqual(decisions[0].key, KEY)

    def test_tampered_or_non_canonical_files_fail_closed(self):
        write_source(self.root, [make_decision()], [make_facts()])
        path = self.root / LABEL_FACTS_FILENAME
        path.write_text(path.read_text(encoding="utf-8").replace("false", "true", 1))
        with self.assertRaises(RiichiDealInSourceError):
            read_labelled_source(self.root)

    def test_seed_without_split_fails_closed(self):
        write_source(
            self.root,
            [make_decision()],
            [make_facts()],
            splits={"train": [1], "valid": [], "test": []},
        )
        with self.assertRaises(RiichiDealInSourceError):
            read_decisions(self.root)

    def test_overlapping_splits_fail_closed(self):
        write_source(
            self.root,
            [make_decision()],
            [make_facts()],
            splits={"train": [931000], "valid": [931000], "test": []},
        )
        with self.assertRaises(RiichiDealInSourceError):
            read_decisions(self.root)


if __name__ == "__main__":
    unittest.main()
