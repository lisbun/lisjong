import unittest

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.exact_wait_ground_truth import (
    exact_hand_belief_with_waits,
    exact_hand_belief_with_waits_for_own_hand_state,
)
from lisjong.belief.fixed_point import SCALE
from lisjong.hand_evaluation.shanten import calculate_shanten
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

MANZU = TileCategory.MANZU
PINZU = TileCategory.PINZU
SOUZU = TileCategory.SOUZU
HONOR = TileCategory.HONOR

MECHANISM_ACCESSOR_NAMES = (
    "tanki_wait_probability",
    "shanpon_wait_probability",
    "kanchan_wait_probability",
    "penchan_wait_probability",
    "ryanmen_low_side_probability",
    "ryanmen_high_side_probability",
    "kokushi_wait_probability",
)


def m(rank: int, red: bool = False) -> Tile:
    return Tile(TileType(MANZU, rank), is_red=red)


def p(rank: int, red: bool = False) -> Tile:
    return Tile(TileType(PINZU, rank), is_red=red)


def s(rank: int, red: bool = False) -> Tile:
    return Tile(TileType(SOUZU, rank), is_red=red)


def z(rank: int) -> Tile:
    return Tile(TileType(HONOR, rank))


def run(category, low_rank: int) -> tuple[Tile, ...]:
    return (
        Tile(TileType(category, low_rank)),
        Tile(TileType(category, low_rank + 1)),
        Tile(TileType(category, low_rank + 2)),
    )


def pair(tile: Tile) -> tuple[Tile, ...]:
    return (tile, tile)


def triplet(tile: Tile) -> tuple[Tile, ...]:
    return (tile, tile, tile)


def _all_active_slots(belief) -> dict[TileType, dict[str, float]]:
    """non-zeroなwait / mechanism slotだけを{TileType: {name: value}}へ集約する。"""
    active: dict[TileType, dict[str, float]] = {}
    for category in (MANZU, PINZU, SOUZU):
        for rank in range(1, 10):
            _record_if_active(active, TileType(category, rank), belief)
    for rank in range(1, 8):
        _record_if_active(active, TileType(HONOR, rank), belief)
    return active


def _record_if_active(active, tile_type, belief) -> None:
    entries = {}
    wait = belief.wait_probability(tile_type)
    if wait:
        entries["wait"] = wait
    for name in MECHANISM_ACCESSOR_NAMES:
        value = getattr(belief, f"{name}")(tile_type)
        if value:
            entries[name] = value
    if entries:
        active[tile_type] = entries


def _assert_exact_ground_truth_invariants(test: unittest.TestCase, belief) -> None:
    """Issue #84が課すexact ground-truth contractを、どのhandにも共通に検証する。"""
    test.assertTrue(belief.has_wait_belief)
    test.assertTrue(belief.has_wait_mechanism_belief)

    for category in (MANZU, PINZU, SOUZU):
        for rank in range(1, 10):
            _assert_slot_is_binary_and_matches_or(
                test, belief, TileType(category, rank)
            )
    for rank in range(1, 8):
        _assert_slot_is_binary_and_matches_or(test, belief, TileType(HONOR, rank))


def _assert_slot_is_binary_and_matches_or(test, belief, tile_type: TileType) -> None:
    index = tile_type_index(tile_type)
    raw_values = [belief.wait_probability_raw[index]] + [
        getattr(belief, f"{name}_raw")[index] for name in MECHANISM_ACCESSOR_NAMES
    ]
    for raw in raw_values:
        test.assertIn(raw, (0, SCALE))

    mechanism_hit = any(
        getattr(belief, f"{name}")(tile_type) == 1.0
        for name in MECHANISM_ACCESSOR_NAMES
    )
    test.assertEqual(belief.wait_probability(tile_type) == 1.0, mechanism_hit)


class StandardHandMechanismTest(unittest.TestCase):
    def test_tanki_wait(self) -> None:
        # 123m 456p 789s 123p + single 5m(tanki)
        concealed = (
            run(MANZU, 1) + run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + (m(5),)
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        self.assertEqual(belief.wait_probability(TileType(MANZU, 5)), 1.0)
        self.assertEqual(belief.tanki_wait_probability(TileType(MANZU, 5)), 1.0)
        active = _all_active_slots(belief)
        self.assertEqual(set(active.keys()), {TileType(MANZU, 5)})
        self.assertEqual(
            active[TileType(MANZU, 5)], {"wait": 1.0, "tanki_wait_probability": 1.0}
        )

    def test_shanpon_wait(self) -> None:
        # 123m 456p 789s + pair(5s) + pair(7z), waits on 5s or 7z
        concealed = (
            run(MANZU, 1) + run(PINZU, 4) + run(SOUZU, 7) + pair(s(5)) + pair(z(7))
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        for tile_type in (TileType(SOUZU, 5), TileType(HONOR, 7)):
            self.assertEqual(belief.wait_probability(tile_type), 1.0)
            self.assertEqual(belief.shanpon_wait_probability(tile_type), 1.0)

        active = _all_active_slots(belief)
        self.assertEqual(set(active.keys()), {TileType(SOUZU, 5), TileType(HONOR, 7)})

    def test_kanchan_wait(self) -> None:
        # 456p 789s 123p + pair(5s) + kanchan(2m,4m) waits on 3m
        concealed = (
            run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + pair(s(5)) + (m(2), m(4))
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        target = TileType(MANZU, 3)
        self.assertEqual(belief.wait_probability(target), 1.0)
        self.assertEqual(belief.kanchan_wait_probability(target), 1.0)
        active = _all_active_slots(belief)
        self.assertEqual(active[target], {"wait": 1.0, "kanchan_wait_probability": 1.0})

    def test_penchan_low_edge_completes_at_high_end(self) -> None:
        # 456p 789s 123p + pair(5s) + (1m,2m) waits on 3m (penchan)
        concealed = (
            run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + pair(s(5)) + (m(1), m(2))
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        target = TileType(MANZU, 3)
        self.assertEqual(belief.penchan_wait_probability(target), 1.0)
        active = _all_active_slots(belief)
        self.assertEqual(active[target], {"wait": 1.0, "penchan_wait_probability": 1.0})

    def test_penchan_high_edge_completes_at_low_end(self) -> None:
        # 456p 789s 123p + pair(5s) + (8m,9m) waits on 7m (penchan)
        concealed = (
            run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + pair(s(5)) + (m(8), m(9))
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        target = TileType(MANZU, 7)
        self.assertEqual(belief.penchan_wait_probability(target), 1.0)
        active = _all_active_slots(belief)
        self.assertEqual(active[target], {"wait": 1.0, "penchan_wait_probability": 1.0})

    def test_ryanmen_low_and_high_sides_are_distinguished(self) -> None:
        # 456p 789s 123p + pair(5s) + (2m,3m) waits on 1m(low) / 4m(high)
        concealed = (
            run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + pair(s(5)) + (m(2), m(3))
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        low = TileType(MANZU, 1)
        high = TileType(MANZU, 4)
        self.assertEqual(belief.ryanmen_low_side_probability(low), 1.0)
        self.assertEqual(belief.ryanmen_high_side_probability(high), 1.0)
        self.assertEqual(belief.ryanmen_high_side_probability(low), 0.0)
        self.assertEqual(belief.ryanmen_low_side_probability(high), 0.0)
        active = _all_active_slots(belief)
        self.assertEqual(set(active.keys()), {low, high})

    def test_honor_tanki(self) -> None:
        concealed = (
            run(MANZU, 1) + run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + (z(1),)
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        target = TileType(HONOR, 1)
        self.assertEqual(belief.wait_probability(target), 1.0)
        self.assertEqual(belief.tanki_wait_probability(target), 1.0)

    def test_honor_shanpon(self) -> None:
        concealed = (
            run(MANZU, 1) + run(PINZU, 4) + run(SOUZU, 7) + pair(z(1)) + pair(z(2))
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        for tile_type in (TileType(HONOR, 1), TileType(HONOR, 2)):
            self.assertEqual(belief.shanpon_wait_probability(tile_type), 1.0)


class OpenHandMeldTest(unittest.TestCase):
    def _chi_meld(self, category, low_rank: int) -> PublicMeld:
        tiles = run(category, low_rank)
        return PublicMeld(
            kind=MeldKind.CHI,
            tiles=tiles,
            from_seat=Seat.SEAT_1,
            called_tile=tiles[0],
        )

    def _pon_meld(self, tile: Tile) -> PublicMeld:
        tiles = triplet(tile)
        return PublicMeld(
            kind=MeldKind.PON,
            tiles=tiles,
            from_seat=Seat.SEAT_2,
            called_tile=tile,
        )

    def test_chi_and_pon_open_hand_ryanmen(self) -> None:
        # melds: chi(1p2p3p) + pon(9s9s9s); concealed: 123m + (4p,5p) ryanmen + pair(6s)
        own_melds = (self._chi_meld(PINZU, 1), self._pon_meld(s(9)))
        concealed = run(MANZU, 1) + (p(4), p(5)) + pair(s(6))
        belief = exact_hand_belief_with_waits(concealed, own_melds)
        _assert_exact_ground_truth_invariants(self, belief)

        self.assertEqual(belief.ryanmen_low_side_probability(TileType(PINZU, 3)), 1.0)
        self.assertEqual(belief.ryanmen_high_side_probability(TileType(PINZU, 6)), 1.0)
        # meld内の牌はexpected_countへ混入しない。
        self.assertEqual(belief.expected_count(TileType(PINZU, 1)), 0.0)
        self.assertEqual(belief.expected_count(TileType(SOUZU, 9)), 0.0)

    def test_multiple_melds_standard_hand(self) -> None:
        # melds: pon(1z) + pon(2z) + chi(1s2s3s); concealed: (2m,3m) ryanmen + pair(4p)
        own_melds = (
            self._pon_meld(z(1)),
            self._pon_meld(z(2)),
            self._chi_meld(SOUZU, 1),
        )
        concealed = (m(2), m(3)) + pair(p(4))
        belief = exact_hand_belief_with_waits(concealed, own_melds)
        _assert_exact_ground_truth_invariants(self, belief)

        self.assertEqual(belief.ryanmen_low_side_probability(TileType(MANZU, 1)), 1.0)
        self.assertEqual(belief.ryanmen_high_side_probability(TileType(MANZU, 4)), 1.0)

    def test_meld_order_does_not_affect_result(self) -> None:
        own_melds_a = (self._chi_meld(PINZU, 1), self._pon_meld(s(9)))
        own_melds_b = tuple(reversed(own_melds_a))
        concealed = run(MANZU, 1) + (p(4), p(5)) + pair(s(6))

        belief_a = exact_hand_belief_with_waits(concealed, own_melds_a)
        belief_b = exact_hand_belief_with_waits(concealed, own_melds_b)
        self.assertEqual(belief_a, belief_b)


class KanStructuralAndPhysicalCountTest(unittest.TestCase):
    def test_daiminkan_is_one_completed_meld_and_four_physical_tiles(self) -> None:
        kan_tile = s(9)
        daiminkan = PublicMeld(
            kind=MeldKind.DAIMINKAN,
            tiles=(kan_tile, kan_tile, kan_tile, kan_tile),
            from_seat=Seat.SEAT_3,
            called_tile=kan_tile,
        )
        # structural: kan(+3) + concealed(10) == 13. concealed: 123m 456p 789p + single 2s
        concealed = run(MANZU, 1) + run(PINZU, 4) + run(PINZU, 7) + (s(2),)
        belief = exact_hand_belief_with_waits(concealed, (daiminkan,))
        _assert_exact_ground_truth_invariants(self, belief)

        # kan tiles do not leak into expected_count (concealed-only marginal).
        self.assertEqual(belief.expected_count(TileType(SOUZU, 9)), 0.0)
        # a 5th physical copy of the kan's tile kind must be rejected as a candidate.
        self.assertEqual(belief.wait_probability(TileType(SOUZU, 9)), 0.0)

    def test_kakan_is_one_completed_meld_and_four_physical_tiles(self) -> None:
        kan_tile = s(9)
        kakan = PublicMeld(
            kind=MeldKind.KAKAN,
            tiles=(kan_tile, kan_tile, kan_tile, kan_tile),
            from_seat=Seat.SEAT_3,
            called_tile=kan_tile,
        )
        concealed = run(MANZU, 1) + run(PINZU, 4) + run(PINZU, 7) + (s(2),)
        belief = exact_hand_belief_with_waits(concealed, (kakan,))
        _assert_exact_ground_truth_invariants(self, belief)

        self.assertEqual(belief.expected_count(TileType(SOUZU, 9)), 0.0)
        self.assertEqual(belief.wait_probability(TileType(SOUZU, 9)), 0.0)

    def test_ankan_is_one_completed_meld_and_four_physical_tiles(self) -> None:
        kan_tile = s(9)
        ankan = PublicMeld(
            kind=MeldKind.ANKAN,
            tiles=(kan_tile, kan_tile, kan_tile, kan_tile),
            from_seat=None,
            called_tile=None,
        )
        concealed = run(MANZU, 1) + run(PINZU, 4) + run(PINZU, 7) + (s(2),)
        belief = exact_hand_belief_with_waits(concealed, (ankan,))
        _assert_exact_ground_truth_invariants(self, belief)

        self.assertEqual(belief.expected_count(TileType(SOUZU, 9)), 0.0)
        self.assertEqual(belief.wait_probability(TileType(SOUZU, 9)), 0.0)

    def test_concealed_four_of_a_kind_is_not_auto_interpreted_as_ankan(self) -> None:
        # concealed contains 4x 9s without any own_melds entry: still just plain
        # concealed counts, decomposed as triplet(9s9s9s) + leftover 9s (tanki),
        # not folded into a meld automatically.
        concealed = (
            run(MANZU, 1) + run(PINZU, 4) + run(PINZU, 7) + (s(9), s(9), s(9), s(9))
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        # the leftover 9s cannot pair with a 5th physical copy: 5th 9s is illegal.
        self.assertEqual(belief.wait_probability(TileType(SOUZU, 9)), 0.0)
        self.assertEqual(belief.tanki_wait_probability(TileType(SOUZU, 9)), 0.0)


class ChiitoitsuTest(unittest.TestCase):
    def test_seven_pairs_wait_is_expressed_as_tanki_only(self) -> None:
        concealed = (
            pair(z(1))
            + pair(z(2))
            + pair(z(3))
            + pair(z(4))
            + pair(z(5))
            + pair(z(6))
            + (z(7),)
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        target = TileType(HONOR, 7)
        self.assertEqual(belief.wait_probability(target), 1.0)
        self.assertEqual(belief.tanki_wait_probability(target), 1.0)
        self.assertEqual(belief.kokushi_wait_probability(target), 0.0)
        active = _all_active_slots(belief)
        self.assertEqual(set(active.keys()), {target})

    def test_four_of_a_kind_is_not_counted_as_two_pairs(self) -> None:
        # 5 honor pairs (10 tiles) + 3x manzu-1 (3 tiles) = 13 concealed tiles.
        # Completing with a 4th manzu-1 gives only 6 distinct kinds (5 honor +
        # manzu-1), not 7: a naive "pair_count = sum(count // 2)" style
        # implementation would wrongly count the manzu-1 quad as 2 pairs and
        # reach 7, so this guards against that specific bug.
        concealed = (
            pair(z(1))
            + pair(z(2))
            + pair(z(3))
            + pair(z(4))
            + pair(z(5))
            + (m(1), m(1), m(1))
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        # completing with the 4th manzu-1 would make manzu-1 count 4 with only
        # 6 distinct kinds present, not a valid 7-distinct-pairs shape, and
        # standard hand cannot complete disconnected honor pairs into melds.
        self.assertEqual(belief.wait_probability(TileType(MANZU, 1)), 0.0)


class KokushiTest(unittest.TestCase):
    _TERMINAL_AND_HONOR_TILES = (
        m(1),
        m(9),
        p(1),
        p(9),
        s(1),
        s(9),
        z(1),
        z(2),
        z(3),
        z(4),
        z(5),
        z(6),
        z(7),
    )

    def test_kokushi_thirteen_sided_wait(self) -> None:
        concealed = self._TERMINAL_AND_HONOR_TILES
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        for tile in self._TERMINAL_AND_HONOR_TILES:
            self.assertEqual(belief.kokushi_wait_probability(tile.tile_type), 1.0)
            self.assertEqual(belief.wait_probability(tile.tile_type), 1.0)
            self.assertEqual(belief.tanki_wait_probability(tile.tile_type), 0.0)

    def test_kokushi_single_wait(self) -> None:
        # pair on m1, missing z7 (the last tile of the 13-tile tuple) -> only
        # z7 completes kokushi.
        concealed = self._TERMINAL_AND_HONOR_TILES[:-1] + (m(1),)
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        winning = TileType(HONOR, 7)
        self.assertEqual(belief.kokushi_wait_probability(winning), 1.0)
        self.assertEqual(belief.wait_probability(winning), 1.0)
        for tile in self._TERMINAL_AND_HONOR_TILES:
            if tile.tile_type != winning:
                self.assertEqual(belief.wait_probability(tile.tile_type), 0.0)


class MeldPresentDisablesChiitoitsuAndKokushiTest(unittest.TestCase):
    def test_ankan_present_disables_chiitoitsu(self) -> None:
        ankan_tile = s(9)
        ankan = PublicMeld(
            kind=MeldKind.ANKAN,
            tiles=(ankan_tile, ankan_tile, ankan_tile, ankan_tile),
            from_seat=None,
            called_tile=None,
        )
        # 5 honor pairs (10 tiles) + ankan meld: structurally 13-equivalent.
        concealed = pair(z(1)) + pair(z(2)) + pair(z(3)) + pair(z(4)) + pair(z(5))
        belief = exact_hand_belief_with_waits(concealed, (ankan,))
        _assert_exact_ground_truth_invariants(self, belief)

        # without the meld this shape would be one step from chiitoitsu; with a
        # meld present, chiitoitsu must not be evaluated, and standard hand
        # cannot complete disconnected honor pairs into melds either.
        self.assertEqual(belief.wait_probability(TileType(HONOR, 6)), 0.0)
        active = _all_active_slots(belief)
        self.assertEqual(active, {})

    def test_ankan_present_disables_kokushi(self) -> None:
        ankan_tile = s(5)
        ankan = PublicMeld(
            kind=MeldKind.ANKAN,
            tiles=(ankan_tile, ankan_tile, ankan_tile, ankan_tile),
            from_seat=None,
            called_tile=None,
        )
        # 10 distinct terminal/honor singles + ankan meld: near-kokushi shape,
        # but meld_count > 0 must disable kokushi evaluation.
        concealed = (m(1), m(9), p(1), p(9), s(1), s(9), z(1), z(2), z(3), z(4))
        belief = exact_hand_belief_with_waits(concealed, (ankan,))
        _assert_exact_ground_truth_invariants(self, belief)

        active = _all_active_slots(belief)
        self.assertEqual(active, {})


class AmbiguousDecompositionGoldenTest(unittest.TestCase):
    def test_kanchan_and_tanki_from_a_single_decomposition(self) -> None:
        # concealed: 2m,3m,3m,4m (a kanchan taatsu 2m-4m sharing a pair-or-run
        # ambiguity with 3m) + three fixed complete runs to reach 13 tiles.
        concealed = (
            (m(2), m(3), m(3), m(4)) + run(PINZU, 1) + run(PINZU, 4) + run(PINZU, 7)
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        target = TileType(MANZU, 3)
        self.assertEqual(belief.wait_probability(target), 1.0)
        self.assertEqual(belief.tanki_wait_probability(target), 1.0)
        self.assertEqual(belief.kanchan_wait_probability(target), 1.0)
        self.assertEqual(belief.shanpon_wait_probability(target), 0.0)
        self.assertEqual(belief.penchan_wait_probability(target), 0.0)
        self.assertEqual(belief.ryanmen_low_side_probability(target), 0.0)
        self.assertEqual(belief.ryanmen_high_side_probability(target), 0.0)

        # cross-check against the existing shanten oracle: adding 3m completes.
        self.assertEqual(calculate_shanten(concealed + (m(3),)), -1)

    def test_standard_and_chiitoitsu_both_complete(self) -> None:
        # 1122334455667m: both a 4-run + pair decomposition and chiitoitsu
        # complete on the same candidate (7m).
        concealed = (
            m(1),
            m(1),
            m(2),
            m(2),
            m(3),
            m(3),
            m(4),
            m(4),
            m(5),
            m(5),
            m(6),
            m(6),
            m(7),
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        target = TileType(MANZU, 7)
        self.assertEqual(belief.wait_probability(target), 1.0)
        self.assertEqual(belief.tanki_wait_probability(target), 1.0)
        self.assertEqual(calculate_shanten(concealed + (m(7),)), -1)


class NonTenpaiTest(unittest.TestCase):
    def test_non_tenpai_is_level_two_all_zero(self) -> None:
        concealed = (
            m(1),
            m(4),
            m(7),
            p(1),
            p(4),
            p(7),
            s(1),
            s(4),
            s(7),
            z(1),
            z(3),
            z(5),
            z(7),
        )
        belief = exact_hand_belief_with_waits(concealed, ())

        self.assertTrue(belief.has_wait_belief)
        self.assertTrue(belief.has_wait_mechanism_belief)
        self.assertEqual(sum(belief.wait_probability_raw), 0)
        for name in MECHANISM_ACCESSOR_NAMES:
            self.assertEqual(sum(getattr(belief, f"{name}_raw")), 0)


class PhysicalTileCountTest(unittest.TestCase):
    def test_own_physical_four_of_a_kind_excludes_fifth_candidate(self) -> None:
        pon_tile = s(9)
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=triplet(pon_tile),
            from_seat=Seat.SEAT_1,
            called_tile=pon_tile,
        )
        # meld holds 3 physical 9s; concealed holds a lone 9s (tanki candidate)
        # plus three fixed complete melds, but the 4th physical 9s is already
        # spoken for, so the 5th (candidate) copy must be excluded.
        concealed = (s(9),) + run(MANZU, 1) + run(PINZU, 4) + run(PINZU, 7)
        belief = exact_hand_belief_with_waits(concealed, (pon,))
        _assert_exact_ground_truth_invariants(self, belief)

        self.assertEqual(belief.wait_probability(TileType(SOUZU, 9)), 0.0)
        active = _all_active_slots(belief)
        self.assertEqual(active, {})

    def test_red_five_counts_as_the_same_base_kind_as_normal_five(self) -> None:
        concealed = (
            (m(5, red=True), m(5))
            + run(PINZU, 4)
            + run(PINZU, 7)
            + run(SOUZU, 1)
            + (p(2), p(3))
        )
        belief = exact_hand_belief_with_waits(concealed, ())
        _assert_exact_ground_truth_invariants(self, belief)

        self.assertEqual(belief.expected_count(TileType(MANZU, 5)), 2.0)
        self.assertEqual(belief.red_five_probability(MANZU), 1.0)
        self.assertEqual(belief.ryanmen_low_side_probability(TileType(PINZU, 1)), 1.0)
        self.assertEqual(belief.ryanmen_high_side_probability(TileType(PINZU, 4)), 1.0)

    def test_rejects_more_than_four_physical_copies_of_a_base_kind(self) -> None:
        concealed = (
            (m(1), m(1), m(1), m(1), m(1))
            + run(PINZU, 4)
            + run(PINZU, 7)
            + (p(2), p(3))
        )
        with self.assertRaises(ValueError):
            exact_hand_belief_with_waits(concealed, ())

    def test_rejects_more_than_four_physical_copies_split_across_meld(self) -> None:
        pon_tile = s(9)
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=triplet(pon_tile),
            from_seat=Seat.SEAT_1,
            called_tile=pon_tile,
        )
        # meld holds 3 physical 9s, concealed holds 2 more -> 5 total.
        concealed = (s(9), s(9)) + run(MANZU, 1) + run(PINZU, 4) + (p(6), p(7))
        with self.assertRaises(ValueError):
            exact_hand_belief_with_waits(concealed, (pon,))

    def test_rejects_more_than_one_red_five_per_suit(self) -> None:
        concealed = (
            (m(5, red=True), m(5, red=True))
            + run(PINZU, 4)
            + run(PINZU, 7)
            + run(SOUZU, 1)
            + (p(2), p(3))
        )
        with self.assertRaises(ValueError):
            exact_hand_belief_with_waits(concealed, ())

    def test_rejects_red_five_split_across_concealed_and_meld(self) -> None:
        # meld holds 1 red-5m (its only legal red copy) + 2 normal 5m;
        # concealed independently holds another red-5m -> 2 reds total.
        called = m(5)
        pon_with_red = PublicMeld(
            kind=MeldKind.PON,
            tiles=(m(5, red=True), called, called),
            from_seat=Seat.SEAT_1,
            called_tile=called,
        )
        concealed = (
            (m(5, red=True),) + run(PINZU, 4) + run(PINZU, 7) + (p(2), p(3), p(4))
        )
        with self.assertRaises(ValueError):
            exact_hand_belief_with_waits(concealed, (pon_with_red,))


class StableStateContractTest(unittest.TestCase):
    def test_rejects_wrong_concealed_tile_count_with_no_melds(self) -> None:
        concealed = (
            run(MANZU, 1) + run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1)
        )  # 12 tiles
        with self.assertRaises(ValueError):
            exact_hand_belief_with_waits(concealed, ())

    def test_rejects_fourteen_equivalent_drawn_state(self) -> None:
        concealed = (
            run(MANZU, 1) + run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + (m(5), m(6))
        )  # 14 tiles
        with self.assertRaises(ValueError):
            exact_hand_belief_with_waits(concealed, ())

    def test_rejects_meld_count_inconsistent_with_concealed_length(self) -> None:
        pon_tile = z(1)
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=triplet(pon_tile),
            from_seat=Seat.SEAT_1,
            called_tile=pon_tile,
        )
        # 13 concealed tiles + 1 meld => structural count 16, not 13.
        concealed = run(MANZU, 1) + run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1)
        with self.assertRaises(ValueError):
            exact_hand_belief_with_waits(concealed, (pon,))

    def test_own_hand_state_with_drawn_tile_is_rejected(self) -> None:
        drawn = m(5)
        concealed = (
            run(MANZU, 1) + run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + (drawn,)
        )  # 14 tiles including the draw
        own_hand_state = OwnHandState(concealed_tiles=concealed, drawn_tile=drawn)
        with self.assertRaises(ValueError):
            exact_hand_belief_with_waits_for_own_hand_state(own_hand_state, ())

    def test_own_hand_state_without_drawn_tile_delegates_to_direct_builder(
        self,
    ) -> None:
        concealed = (
            run(MANZU, 1) + run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + (m(5),)
        )
        own_hand_state = OwnHandState(concealed_tiles=concealed, drawn_tile=None)

        via_state = exact_hand_belief_with_waits_for_own_hand_state(own_hand_state, ())
        via_direct = exact_hand_belief_with_waits(concealed, ())
        self.assertEqual(via_state, via_direct)


class DeterminismTest(unittest.TestCase):
    def test_concealed_tile_order_does_not_affect_result(self) -> None:
        concealed = (
            run(MANZU, 1) + run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + (m(5),)
        )
        shuffled = tuple(reversed(concealed))

        belief_a = exact_hand_belief_with_waits(concealed, ())
        belief_b = exact_hand_belief_with_waits(shuffled, ())
        self.assertEqual(belief_a, belief_b)


class ExpectedCountConcealedOnlyTest(unittest.TestCase):
    def test_meld_tiles_never_appear_in_expected_count_or_red_five_probability(
        self,
    ) -> None:
        pon_tile = m(5)
        pon = PublicMeld(
            kind=MeldKind.PON,
            # only 1 red-5m exists canonically: this pon holds it, the other
            # 2 physical copies are normal 5m.
            tiles=(m(5, red=True), pon_tile, pon_tile),
            from_seat=Seat.SEAT_1,
            called_tile=pon_tile,
        )
        concealed = run(PINZU, 4) + run(PINZU, 7) + run(SOUZU, 1) + (p(2),)
        belief = exact_hand_belief_with_waits(concealed, (pon,))

        self.assertEqual(belief.expected_count(TileType(MANZU, 5)), 0.0)
        self.assertEqual(belief.red_five_probability(MANZU), 0.0)
        self.assertEqual(sum(belief.expected_count_raw), len(concealed) * SCALE)


class ShantenOracleCrossCheckTest(unittest.TestCase):
    """`calculate_shanten()`は、own meld tiles自体を見ないmeldlessなoracleとして
    使う。ここでは own_melds=() のhandだけを対象に、全34候補についてbuilderの
    `wait[t]`と`calculate_shanten(concealed + [t]) == -1`が一致することを確認する。
    """

    def _cross_check(self, concealed: tuple[Tile, ...]) -> None:
        belief = exact_hand_belief_with_waits(concealed, ())
        for category in (MANZU, PINZU, SOUZU):
            for rank in range(1, 10):
                self._assert_matches_oracle(belief, concealed, TileType(category, rank))
        for rank in range(1, 8):
            self._assert_matches_oracle(belief, concealed, TileType(HONOR, rank))

    def _assert_matches_oracle(self, belief, concealed, tile_type: TileType) -> None:
        candidate = Tile(tile_type)
        try:
            shanten = calculate_shanten(concealed + (candidate,))
        except ValueError:
            return  # candidate itself is physically illegal (5th copy); skip.
        expected_wait = shanten == -1
        self.assertEqual(
            belief.wait_probability(tile_type) == 1.0,
            expected_wait,
            msg=f"mismatch at {tile_type}",
        )

    def test_cross_check_tanki_hand(self) -> None:
        concealed = (
            run(MANZU, 1) + run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + (m(5),)
        )
        self._cross_check(concealed)

    def test_cross_check_ryanmen_hand(self) -> None:
        concealed = (
            run(PINZU, 4) + run(SOUZU, 7) + run(PINZU, 1) + pair(s(5)) + (m(2), m(3))
        )
        self._cross_check(concealed)

    def test_cross_check_ambiguous_golden_hand(self) -> None:
        concealed = (
            (m(2), m(3), m(3), m(4)) + run(PINZU, 1) + run(PINZU, 4) + run(PINZU, 7)
        )
        self._cross_check(concealed)

    def test_cross_check_kokushi_hand(self) -> None:
        concealed = (
            m(1),
            m(9),
            p(1),
            p(9),
            s(1),
            s(9),
            z(1),
            z(2),
            z(3),
            z(4),
            z(5),
            z(6),
            z(7),
        )
        self._cross_check(concealed)

    def test_cross_check_non_tenpai_hand(self) -> None:
        concealed = (
            m(1),
            m(4),
            m(7),
            p(1),
            p(4),
            p(7),
            s(1),
            s(4),
            s(7),
            z(1),
            z(3),
            z(5),
            z(7),
        )
        self._cross_check(concealed)


if __name__ == "__main__":
    unittest.main()
