"""待ち形の成立可能性の切り出し（#245）がChampionの古典的危険度scoreを変えないことの同値test。

`_reference_score`は切り出し前（lisjong main f07ff9b）の`_classical_riichi_danger_score`を、
河を直接受け取る形に直しただけで、判定ロジックはそのまま写している。
"""

import random
import unittest

import lisjong.policies.mechanism_riichi_defense_yakuhai_call as mechanism
from lisjong.belief.canonical_axes import tile_type_from_index, tile_type_index
from lisjong.belief.wait_shape_support import river_relation, wait_shape_support
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.tile import Tile, TileCategory, TileType


def _reference_score(
    candidate: TileType,
    river: frozenset[TileType],
    remaining_tile_counts: tuple[int, ...],
) -> mechanism._ClassicalRiichiDangerBreakdown:
    """公開河とremaining inventoryから固定v1 relative scoreを返す。"""
    opponent_river = river
    if candidate in opponent_river:
        return mechanism._ZERO_DANGER

    def remaining(rank: int) -> int:
        return remaining_tile_counts[
            tile_type_index(TileType(candidate.category, rank))
        ]

    same_tile = mechanism._same_tile_contribution(
        candidate, remaining_tile_counts[tile_type_index(candidate)]
    )
    if candidate.category is TileCategory.HONOR:
        return mechanism._ClassicalRiichiDangerBreakdown(same_tile, 0, 0, 0, 0)

    rank = candidate.rank
    penchan = 0
    if rank == 3 and remaining(1) > 0 and remaining(2) > 0:
        penchan = 3
    elif rank == 7 and remaining(8) > 0 and remaining(9) > 0:
        penchan = 3

    kanchan = (
        3
        if 2 <= rank <= 8 and remaining(rank - 1) > 0 and remaining(rank + 1) > 0
        else 0
    )
    ryanmen_low_side = (
        10
        if 1 <= rank <= 6
        and remaining(rank + 1) > 0
        and remaining(rank + 2) > 0
        and TileType(candidate.category, rank + 3) not in opponent_river
        else 0
    )
    ryanmen_high_side = (
        10
        if 4 <= rank <= 9
        and remaining(rank - 2) > 0
        and remaining(rank - 1) > 0
        and TileType(candidate.category, rank - 3) not in opponent_river
        else 0
    )
    return mechanism._ClassicalRiichiDangerBreakdown(
        same_tile,
        penchan,
        kanchan,
        ryanmen_low_side,
        ryanmen_high_side,
    )


def _all_tile_types() -> list[TileType]:
    return [tile_type_from_index(index) for index in range(34)]


def _random_remaining(rng: random.Random) -> tuple[int, ...]:
    # 観測者から見た未見枚数。候補牌自身は観測者の手牌にあるため0..3、他は0..4
    return tuple(rng.choice((0, 0, 1, 2, 3, 4)) for _ in range(34))


def _random_river(rng: random.Random) -> frozenset[TileType]:
    return frozenset(rng.sample(_all_tile_types(), rng.choice((0, 1, 3, 6, 10, 18))))


def _opponent(river: frozenset[TileType]) -> PlayerPublicState:
    discards = tuple(
        Discard(Tile(tile_type, False), False, order, None)
        for order, tile_type in enumerate(sorted(river, key=tile_type_index))
    )
    return PlayerPublicState(25000, discards, (), RiichiState.ACCEPTED)


class ClassicalScoreEquivalenceTest(unittest.TestCase):
    def test_every_candidate_matches_the_reference_on_random_inputs(self) -> None:
        rng = random.Random(245)
        checked = 0
        for _ in range(3000):
            remaining = _random_remaining(rng)
            river = _random_river(rng)
            opponent = _opponent(river)
            for candidate in _all_tile_types():
                own = remaining[tile_type_index(candidate)]
                if own > 3 and candidate not in river:
                    with self.assertRaises(ValueError):
                        _reference_score(candidate, river, remaining)
                    with self.assertRaises(ValueError):
                        mechanism._classical_riichi_danger_score(
                            candidate, opponent, remaining
                        )
                    continue
                self.assertEqual(
                    _reference_score(candidate, river, remaining),
                    mechanism._classical_riichi_danger_score(
                        candidate, opponent, remaining
                    ),
                    (candidate, river),
                )
                checked += 1
        self.assertGreater(checked, 50_000)

    def test_genbutsu_is_zero_even_when_the_remaining_count_is_invalid(self) -> None:
        remaining = (4,) * 34
        for candidate in _all_tile_types():
            river = frozenset({candidate})
            self.assertEqual(
                mechanism._classical_riichi_danger_score(
                    candidate, _opponent(river), remaining
                ),
                mechanism._ZERO_DANGER,
            )

    def test_fixed_scenarios_cover_walls_honors_and_suji(self) -> None:
        m = TileCategory.MANZU
        full = (3,) * 34
        scenarios = {
            "all open": (TileType(m, 5), frozenset()),
            "wall low": (TileType(m, 5), frozenset()),
            "suji low": (TileType(m, 2), frozenset({TileType(m, 5)})),
            "suji high": (TileType(m, 8), frozenset({TileType(m, 5)})),
            "honor": (TileType(TileCategory.HONOR, 7), frozenset()),
            "terminal": (TileType(m, 1), frozenset({TileType(m, 4)})),
        }
        for name, (candidate, river) in scenarios.items():
            remaining = list(full)
            if name == "wall low":
                remaining[tile_type_index(TileType(m, 6))] = 0
                remaining[tile_type_index(TileType(m, 3))] = 0
            remaining = tuple(remaining)
            self.assertEqual(
                _reference_score(candidate, river, remaining),
                mechanism._classical_riichi_danger_score(
                    candidate, _opponent(river), remaining
                ),
                name,
            )


class WaitShapeSupportTest(unittest.TestCase):
    def test_support_follows_the_remaining_counts_of_the_constituent_tiles(
        self,
    ) -> None:
        m = TileCategory.MANZU
        remaining = [0] * 34
        for rank in (3, 4, 6, 7):
            remaining[tile_type_index(TileType(m, rank))] = 1
        support = wait_shape_support(TileType(m, 5), tuple(remaining))
        self.assertTrue(support.kanchan)  # 4と6
        self.assertTrue(support.ryanmen_low_side)  # 6と7
        self.assertTrue(support.ryanmen_high_side)  # 3と4
        self.assertFalse(support.tanki)  # 5は残っていない
        self.assertFalse(support.shanpon)
        self.assertFalse(support.penchan)

    def test_tanki_and_shanpon_need_one_and_two_remaining_copies(self) -> None:
        tile = TileType(TileCategory.HONOR, 1)
        for count, tanki, shanpon in (
            (0, False, False),
            (1, True, False),
            (2, True, True),
        ):
            remaining = [0] * 34
            remaining[tile_type_index(tile)] = count
            support = wait_shape_support(tile, tuple(remaining))
            self.assertEqual((support.tanki, support.shanpon), (tanki, shanpon))
            self.assertEqual(support.count, int(tanki) + int(shanpon))

    def test_penchan_only_for_three_and_seven(self) -> None:
        p = TileCategory.PINZU
        remaining = (4,) * 34
        for rank in range(1, 10):
            support = wait_shape_support(TileType(p, rank), remaining)
            self.assertEqual(support.penchan, rank in (3, 7), rank)
            self.assertEqual(support.kanchan, 2 <= rank <= 8, rank)
            self.assertEqual(support.ryanmen_low_side, rank <= 6, rank)
            self.assertEqual(support.ryanmen_high_side, rank >= 4, rank)

    def test_river_relation_is_independent_of_the_support(self) -> None:
        s = TileCategory.SOUZU
        river = frozenset({TileType(s, 5), TileType(s, 8)})
        relation = river_relation(TileType(s, 2), river)
        self.assertEqual(
            (
                relation.in_river,
                relation.ryanmen_low_far_end_in_river,
                relation.ryanmen_high_far_end_in_river,
            ),
            (False, True, False),
        )
        relation = river_relation(TileType(s, 5), river)
        self.assertTrue(relation.in_river)
        # 河にある牌でも、枚数制約の層は成立可能性を変えない
        self.assertTrue(wait_shape_support(TileType(s, 5), (3,) * 34).kanchan)

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            wait_shape_support(TileType(TileCategory.MANZU, 1), (1,) * 33)
        with self.assertRaises(TypeError):
            river_relation("1m", frozenset())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
