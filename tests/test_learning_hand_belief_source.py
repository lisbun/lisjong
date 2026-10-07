import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from lisjong.belief.canonical_axes import tile_type_from_index
from lisjong.learning._canonical import canonical_json_line, file_digest
from lisjong.learning.hand_belief_source import (
    DECISIONS_FILENAME,
    HAND_FACTS_FILENAME,
    MANIFEST_FILENAME,
    DecisionKey,
    HandBeliefDecision,
    HandBeliefHandFacts,
    HandBeliefSourceError,
    OpponentHand,
    decision_to_value,
    hand_facts_to_value,
    label_decisions,
    manifest_text,
    read_decisions,
    read_labelled_source,
)
from lisjong.policy_contract.action import DiscardAction, PassAction
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
SEED = 900001


def tiles(spec: str) -> tuple[Tile, ...]:
    """``0`` は赤5（例: ``406p`` = 4p 赤5p 6p）。"""
    out, ranks = [], ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        category = _CATEGORIES[character]
        out += [
            Tile(TileType(category, int(rank) or 5), is_red=rank == "0")
            for rank in ranks
        ]
        ranks = ""
    return tuple(out)


def tt(spec: str) -> TileType:
    return tiles(spec)[0].tile_type


def types(spec: str) -> frozenset[TileType]:
    return frozenset(tile.tile_type for tile in tiles(spec))


def pon(spec: str, from_seat: Seat) -> PublicMeld:
    meld_tiles = tiles(spec)
    return PublicMeld(
        kind=MeldKind.PON,
        tiles=meld_tiles,
        from_seat=from_seat,
        called_tile=meld_tiles[0],
    )


def chi(spec: str, called: str, from_seat: Seat) -> PublicMeld:
    return PublicMeld(
        kind=MeldKind.CHI,
        tiles=tiles(spec),
        from_seat=from_seat,
        called_tile=tiles(called)[0],
    )


def ankan(spec: str) -> PublicMeld:
    return PublicMeld(
        kind=MeldKind.ANKAN, tiles=tiles(spec), from_seat=None, called_tile=None
    )


def _player(riichi=RiichiState.NONE, discards=(), melds=()):
    return PlayerPublicState(
        score=25000,
        discards=tuple(
            Discard(tile=tile, tsumogiri=False, order=order, called_by=called_by)
            for order, (tile, called_by) in enumerate(discards)
        ),
        melds=tuple(melds),
        riichi=riichi,
    )


def make_decision(sequence, own, players, *, seat=0, legal=None, selected=None):
    hand = tiles(own)
    policy_input = PolicyInput(
        self_seat=Seat(seat),
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(),
            live_wall_tiles_remaining=40,
        ),
        players=players,
        own_hand=OwnHandState(concealed_tiles=hand, drawn_tile=None),
    )
    if legal is None:
        legal = tuple(
            DiscardAction(actor=Seat(seat), tile=tile, tsumogiri=False)
            for tile in dict.fromkeys(hand)
        )
    return HandBeliefDecision(
        key=DecisionKey(seed=SEED, sequence=sequence, seat=seat),
        policy_input=policy_input,
        legal_actions=legal,
        selected_action=selected or legal[0],
    )


def make_facts(sequence, hands, *, seat=0, hand_sequence=None):
    opponents = tuple(
        OpponentHand(
            seat=opponent,
            sequence=sequence if hand_sequence is None else hand_sequence,
            concealed_tiles=tiles(concealed),
            melds=tuple(melds),
        )
        for opponent, (concealed, melds) in hands.items()
    )
    return HandBeliefHandFacts(
        key=DecisionKey(seed=SEED, sequence=sequence, seat=seat), opponents=opponents
    )


# 判断A: リーチ者（赤5・両面）、非リーチ・ポンあり・不聴、非リーチ・暗槓あり・嵌張
A_HANDS = {
    1: ("123m406p789s11z23m", ()),
    2: ("1357m2468p99s", (pon("555z", Seat.SEAT_1),)),
    3: ("678m22p666z35s", (ankan("9999m"),)),
}
A_PLAYERS = (
    _player(),
    _player(
        RiichiState.ACCEPTED,
        discards=((tiles("9p")[0], None), (tiles("5z")[0], Seat.SEAT_2)),
    ),
    _player(melds=A_HANDS[2][1]),
    _player(melds=A_HANDS[3][1]),
)
A_OWN = "234567p456s444z77z"

# 判断B: 非リーチ・双碰、非リーチ・チーあり・辺張、非リーチ・国士13面
B_HANDS = {
    1: ("123p456p789s22z33z", ()),
    2: ("12m567m111z99m", (chi("234s", "2s", Seat.SEAT_1),)),
    3: ("19m19p19s1234567z", ()),
}
B_PLAYERS = (
    _player(),
    _player(discards=((tiles("2s")[0], Seat.SEAT_2),)),
    _player(melds=B_HANDS[2][1]),
    _player(),
)
B_OWN = "345m678p567s888m44z"

# 判断C: リーチ者（単騎）、非リーチ・不聴 x2
C_HANDS = {
    1: ("111m222p333s444z5z", ()),
    2: ("13579m13579p135s", ()),
    3: ("2468m2468p2468s6z", ()),
}
C_PLAYERS = (
    _player(),
    _player(RiichiState.ACCEPTED, discards=((tiles("8s")[0], None),)),
    _player(),
    _player(),
)
C_OWN = "777s999s77z66z55z7m8p"


def all_decisions():
    return [
        make_decision(20, A_OWN, A_PLAYERS),
        make_decision(30, B_OWN, B_PLAYERS),
        make_decision(40, C_OWN, C_PLAYERS),
    ]


def all_facts():
    return [make_facts(20, A_HANDS), make_facts(30, B_HANDS), make_facts(40, C_HANDS)]


_MECHANISMS = (
    "tanki_wait_probability",
    "shanpon_wait_probability",
    "kanchan_wait_probability",
    "penchan_wait_probability",
    "ryanmen_low_side_probability",
    "ryanmen_high_side_probability",
    "kokushi_wait_probability",
)
_ALL_TYPES = tuple(tile_type_from_index(index) for index in range(34))


def positives(belief, table):
    return frozenset(t for t in _ALL_TYPES if getattr(belief, table)(t) == 1.0)


def counts(belief):
    return {t: belief.expected_count(t) for t in _ALL_TYPES if belief.expected_count(t)}


def count_of(spec):
    out = {}
    for tile in tiles(spec):
        out[tile.tile_type] = out.get(tile.tile_type, 0) + 1
    return out


def reds(belief):
    return tuple(
        belief.red_five_probability(category)
        for category in (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)
    )


class LabelDecisionsTest(unittest.TestCase):
    def setUp(self):
        labelled = label_decisions(all_decisions(), all_facts())
        self.truth = {
            (row.decision.key.sequence, opponent.seat): opponent.truth
            for row in labelled
            for opponent in row.opponents
        }

    def assert_waits(self, belief, wait, **mechanisms):
        self.assertEqual(positives(belief, "wait_probability"), wait)
        for table in _MECHANISMS:
            expected = mechanisms.get(table.split("_")[0], frozenset())
            if table.startswith("ryanmen"):
                expected = mechanisms.get(table.split("_")[1], frozenset())
            self.assertEqual(positives(belief, table), expected, table)

    def test_every_decision_labels_the_three_opponents_in_seat_order(self):
        labelled = label_decisions(all_decisions(), all_facts())
        self.assertEqual(
            [[o.seat for o in row.opponents] for row in labelled], [[1, 2, 3]] * 3
        )

    def test_riichi_hand_with_red_five_and_ryanmen(self):
        belief = self.truth[(20, 1)]
        self.assertEqual(counts(belief), count_of("123m456p789s11z23m"))
        self.assertEqual(reds(belief), (0.0, 1.0, 0.0))
        self.assert_waits(belief, types("14m"), low=types("1m"), high=types("4m"))

    def test_non_riichi_pon_hand_is_not_tenpai_and_excludes_meld_tiles(self):
        belief = self.truth[(20, 2)]
        self.assertEqual(counts(belief), count_of("1357m2468p99s"))
        self.assertEqual(reds(belief), (0.0, 0.0, 0.0))
        self.assert_waits(belief, frozenset())

    def test_ankan_hand_counts_the_kan_as_one_meld(self):
        belief = self.truth[(20, 3)]
        self.assertEqual(counts(belief), count_of("678m22p666z35s"))
        self.assert_waits(belief, types("4s"), kanchan=types("4s"))

    def test_shanpon_penchan_and_kokushi(self):
        self.assert_waits(
            self.truth[(30, 1)], types("23z"), shanpon=types("23z"), tanki=frozenset()
        )
        self.assert_waits(self.truth[(30, 2)], types("3m"), penchan=types("3m"))
        thirteen = types("19m19p19s1234567z")
        self.assert_waits(self.truth[(30, 3)], thirteen, kokushi=thirteen)

    def test_tanki_and_non_riichi_non_tenpai(self):
        self.assert_waits(self.truth[(40, 1)], types("5z"), tanki=types("5z"))
        self.assert_waits(self.truth[(40, 2)], frozenset())
        self.assert_waits(self.truth[(40, 3)], frozenset())


class JoinChecksTest(unittest.TestCase):
    def assert_rejected(self, decisions, facts):
        with self.assertRaises(HandBeliefSourceError):
            label_decisions(decisions, facts)

    def test_keys_must_match_exactly(self):
        decisions, facts = all_decisions(), all_facts()
        self.assert_rejected(decisions, facts[:2])
        self.assert_rejected(decisions[:2], facts)
        self.assert_rejected(decisions, facts + [facts[0]])
        self.assert_rejected(decisions + [decisions[0]], facts)

    def test_opponents_must_be_the_other_three_seats_in_order(self):
        decision = make_decision(20, A_OWN, A_PLAYERS)
        reordered = make_facts(20, {2: A_HANDS[2], 1: A_HANDS[1], 3: A_HANDS[3]})
        self.assert_rejected([decision], [reordered])
        missing = make_facts(20, {1: A_HANDS[1], 2: A_HANDS[2]})
        self.assert_rejected([decision], [missing])
        with_self = make_facts(20, {0: A_HANDS[1], 2: A_HANDS[2], 3: A_HANDS[3]})
        self.assert_rejected([decision], [with_self])

    def test_hand_from_after_the_decision_is_rejected(self):
        decision = make_decision(20, A_OWN, A_PLAYERS)
        label_decisions([decision], [make_facts(20, A_HANDS, hand_sequence=19)])
        self.assert_rejected([decision], [make_facts(20, A_HANDS, hand_sequence=21)])

    def test_melds_must_match_the_public_state(self):
        decision = make_decision(20, A_OWN, A_PLAYERS)
        hands = dict(A_HANDS)
        hands[2] = ("1357m2468p99s", (pon("777s", Seat.SEAT_1),))
        self.assert_rejected([decision], [make_facts(20, hands)])

    def test_hand_must_be_13_equivalent(self):
        decision = make_decision(20, A_OWN, A_PLAYERS)
        hands = dict(A_HANDS)
        hands[1] = ("123m406p789s11z23m8s", ())
        self.assert_rejected([decision], [make_facts(20, hands)])

    def test_hands_must_fit_in_the_unseen_tiles(self):
        decision = make_decision(20, A_OWN, A_PLAYERS)
        hands = dict(A_HANDS)
        hands[3] = ("678m22p444z35s", A_HANDS[3][1])  # 自分が444zを持っている
        self.assert_rejected([decision], [make_facts(20, hands)])
        hands = dict(A_HANDS)
        hands[2] = ("1357m2408p99s", A_HANDS[2][1])  # 赤5pが2枚
        self.assert_rejected([decision], [make_facts(20, hands)])

    def test_decision_scope_and_selected_action(self):
        legal = (PassAction(actor=Seat.SEAT_0),)
        with self.assertRaises(HandBeliefSourceError):
            write_and_read([make_decision(20, A_OWN, A_PLAYERS, legal=legal)], [])
        decision = make_decision(20, A_OWN, A_PLAYERS)
        outside = replace(
            decision,
            selected_action=DiscardAction(
                actor=Seat.SEAT_0, tile=tiles("1z")[0], tsumogiri=False
            ),
        )
        with self.assertRaises(HandBeliefSourceError):
            write_and_read([outside], [make_facts(20, A_HANDS)])


def write_source(root: Path, decisions, facts, *, splits=None):
    lines = {
        DECISIONS_FILENAME: [decision_to_value(d) for d in decisions],
        HAND_FACTS_FILENAME: [hand_facts_to_value(f) for f in facts],
    }
    files = {}
    for filename, rows in lines.items():
        path = root / filename
        path.write_text(
            "".join(canonical_json_line(row) for row in rows), encoding="utf-8"
        )
        name = "decisions" if filename == DECISIONS_FILENAME else "hand_facts"
        files[name] = {**file_digest(path), "rows": len(rows)}
    (root / MANIFEST_FILENAME).write_text(
        manifest_text(
            producer={
                "arena_revision": "a" * 40,
                "lisjong_revision": "b" * 40,
                "lisjong_engine_revision": "c" * 40,
                "policy": "MinimalPolicy",
            },
            splits=splits or {"train": [SEED], "valid": [], "test": []},
            files=files,
        ),
        encoding="utf-8",
    )


def write_and_read(decisions, facts):
    with tempfile.TemporaryDirectory() as directory:
        write_source(Path(directory), decisions, facts)
        return read_labelled_source(directory)


class SourceFilesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_source(self.root, all_decisions(), all_facts())

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        manifest, labelled = read_labelled_source(self.root)
        self.assertEqual(manifest.split_of(SEED), "train")
        self.assertEqual(labelled, label_decisions(all_decisions(), all_facts()))

    def test_decision_path_never_opens_the_hand_facts(self):
        opened = []
        original_open = Path.open
        original_read_text = Path.read_text

        def tracking_open(path, *args, **kwargs):
            opened.append(Path(path).name)
            return original_open(path, *args, **kwargs)

        def tracking_read_text(path, *args, **kwargs):
            opened.append(Path(path).name)
            return original_read_text(path, *args, **kwargs)

        with (
            mock.patch.object(Path, "open", tracking_open),
            mock.patch.object(Path, "read_text", tracking_read_text),
            mock.patch("builtins.open", side_effect=AssertionError("builtins.open")),
        ):
            _, decisions = read_decisions(self.root)
        self.assertEqual(len(decisions), 3)
        self.assertNotIn(HAND_FACTS_FILENAME, opened)
        self.assertIn(DECISIONS_FILENAME, opened)
        (self.root / HAND_FACTS_FILENAME).unlink()
        self.assertEqual(len(read_decisions(self.root)[1]), 3)

    def test_tampered_or_non_canonical_files_fail_closed(self):
        path = self.root / HAND_FACTS_FILENAME
        path.write_text(path.read_text(encoding="utf-8").replace("2", "3", 1))
        with self.assertRaises(HandBeliefSourceError):
            read_labelled_source(self.root)

    def test_unknown_versions_fail_closed(self):
        for filename, old, new in (
            (MANIFEST_FILENAME, "source-manifest-v1", "source-manifest-v2"),
            (MANIFEST_FILENAME, "decisions.v1", "decisions.v2"),
            (DECISIONS_FILENAME, "decision-record-v1", "decision-record-v2"),
            (HAND_FACTS_FILENAME, "hand-fact-record-v1", "hand-fact-record-v2"),
        ):
            with self.subTest(filename=filename, new=new):
                write_source(self.root, all_decisions(), all_facts())
                path = self.root / filename
                text = path.read_text(encoding="utf-8")
                self.assertIn(old, text)
                path.write_text(text.replace(old, new), encoding="utf-8")
                if filename != MANIFEST_FILENAME:
                    _refresh_digests(self.root)
                with self.assertRaises(HandBeliefSourceError):
                    read_labelled_source(self.root)

    def test_seed_without_split_fails_closed(self):
        write_source(
            self.root,
            all_decisions(),
            all_facts(),
            splits={"train": [1], "valid": [], "test": []},
        )
        with self.assertRaises(HandBeliefSourceError):
            read_decisions(self.root)


def _refresh_digests(root: Path) -> None:
    """改変したrowのdigestだけを合わせ、schema検査そのものを確かめる。"""
    files = {
        name: {**file_digest(root / filename), "rows": rows}
        for name, filename, rows in (
            ("decisions", DECISIONS_FILENAME, 3),
            ("hand_facts", HAND_FACTS_FILENAME, 3),
        )
    }
    (root / MANIFEST_FILENAME).write_text(
        manifest_text(
            producer={
                "arena_revision": "a" * 40,
                "lisjong_revision": "b" * 40,
                "lisjong_engine_revision": "c" * 40,
                "policy": "MinimalPolicy",
            },
            splits={"train": [SEED], "valid": [], "test": []},
            files=files,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
