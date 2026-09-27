"""Issue #221 canonical牌種集合のmask適用・集約のunit test。

期待値は手計算したmanual golden valueで、単純なindex列挙による合計とも照合する。
"""

import random
import unittest

from lisjong.belief import (
    EXPECTED_COUNT_MAX_RAW,
    SCALE,
    STANDARD_TILE_COUNTS,
    HandBelief,
    NonPlayerHiddenBelief,
    TileConservationResult,
    intersect_tile_type_sets,
    mask_tile_type_values,
    red_five_index,
    sum_tile_type_values,
    tile_type_from_index,
    tile_type_index,
    tile_type_set,
)
from lisjong.policy_contract.tile import TileCategory, TileType

M2 = tile_type_index(TileType(TileCategory.MANZU, 2))
M5 = tile_type_index(TileType(TileCategory.MANZU, 5))
P5 = tile_type_index(TileType(TileCategory.PINZU, 5))
CHUN = tile_type_index(TileType(TileCategory.HONOR, 7))


class TileTypeSetTest(unittest.TestCase):
    def test_normalizes_to_sorted_unique_canonical_indices(self) -> None:
        self.assertEqual(tile_type_set([M5, M2, M5, M2]), (M2, M5))
        self.assertEqual(tile_type_set(()), ())
        self.assertEqual(tile_type_set([33, 0]), (0, 33))
        self.assertEqual(tile_type_set(range(33, -1, -1)), tuple(range(34)))
        # 列挙は既存のcanonical axisでTileTypeへ戻せる。
        self.assertEqual(
            [tile_type_from_index(i) for i in tile_type_set([M5, M2])],
            [TileType(TileCategory.MANZU, 2), TileType(TileCategory.MANZU, 5)],
        )

    def test_rejects_out_of_range_and_non_int_indices(self) -> None:
        for bad in ([-1], [34], [0, 100]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tile_type_set(bad)
        for bad in ([True], [1.0], ["1"], [None]):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                tile_type_set(bad)
        with self.assertRaises(TypeError):
            tile_type_set(5)  # type: ignore[arg-type]

    def test_intersection(self) -> None:
        self.assertEqual(
            intersect_tile_type_sets((0, M2, M5, CHUN), (M5, 30, CHUN)), (M5, CHUN)
        )
        self.assertEqual(intersect_tile_type_sets((M2,), ()), ())
        self.assertEqual(intersect_tile_type_sets((), (M2,)), ())
        full = tuple(range(34))
        self.assertEqual(intersect_tile_type_sets(full, (M2, M5)), (M2, M5))

    def test_positive_remaining_subset_is_a_separate_intersection(self) -> None:
        # 構造上の改善牌は未見0の牌種も含む。可用な改善牌は別の集合演算で得る。
        remaining = [4] * 34
        remaining[M5] = 0
        improving = tile_type_set([M2, M5])
        available = intersect_tile_type_sets(
            improving, tile_type_set(i for i, n in enumerate(remaining) if n > 0)
        )
        self.assertEqual(improving, (M2, M5))
        self.assertEqual(available, (M2,))
        self.assertEqual(sum_tile_type_values(remaining, improving), 4)


class MaskAndSumTest(unittest.TestCase):
    def test_exact_counts_hand_example(self) -> None:
        # 有効牌2萬・5萬、未見枚数: 2萬3枚・5萬(赤5込み)2枚、他は4枚。
        remaining = [4] * 34
        remaining[M2] = 3
        remaining[M5] = 2
        before = tuple(remaining)
        improving = tile_type_set([M5, M2])
        masked = mask_tile_type_values(remaining, improving)

        self.assertEqual(len(masked), 34)
        self.assertEqual(masked[M2], 3)
        self.assertEqual(masked[M5], 2)
        self.assertEqual(sum(1 for v in masked if v != 0), 2)
        self.assertEqual(sum_tile_type_values(remaining, improving), 5)
        self.assertEqual(sum(masked), 5)
        self.assertEqual(tuple(remaining), before)

    def test_empty_full_and_edge_indices(self) -> None:
        values = tuple(range(1, 35))
        self.assertEqual(sum_tile_type_values(values, ()), 0)
        self.assertEqual(mask_tile_type_values(values, ()), (0,) * 34)
        full = tuple(range(34))
        self.assertEqual(sum_tile_type_values(values, full), sum(values))
        self.assertEqual(mask_tile_type_values(values, full), values)
        self.assertEqual(sum_tile_type_values(values, (0, 33)), 1 + 34)
        masked = mask_tile_type_values(values, (0, 33))
        self.assertEqual((masked[0], masked[33]), (1, 34))
        self.assertEqual(masked[1:33], (0,) * 32)

    def test_zero_and_four_counts_and_total_above_four(self) -> None:
        remaining = [0] * 34
        remaining[M2] = 4
        remaining[P5] = 4
        remaining[CHUN] = 4
        improving = tile_type_set([M2, M5, P5, CHUN])
        # 未見0の5萬は集合に残り、合計は1牌種の上限(4枚)で切り詰めない。
        self.assertIn(M5, improving)
        self.assertEqual(sum_tile_type_values(remaining, improving), 12)
        self.assertEqual(mask_tile_type_values(remaining, improving)[M5], 0)

    def test_duplicate_indices_do_not_double_count(self) -> None:
        values = [1] * 34
        self.assertEqual(sum_tile_type_values(values, tile_type_set([M2, M2, M2])), 1)

    def test_values_length_is_validated(self) -> None:
        for bad in ((1,) * 33, (1,) * 35, ()):
            with self.subTest(length=len(bad)):
                with self.assertRaises(ValueError):
                    sum_tile_type_values(bad, (0,))
                with self.assertRaises(ValueError):
                    mask_tile_type_values(bad, (0,))

    def test_matches_plain_index_enumeration(self) -> None:
        generator = random.Random(221)
        for _ in range(200):
            values = [
                generator.randrange(0, EXPECTED_COUNT_MAX_RAW + 1) for _ in range(34)
            ]
            indices = [
                generator.randrange(34) for _ in range(generator.randrange(0, 40))
            ]
            tile_types = tile_type_set(indices)
            expected = sum(values[i] for i in sorted(set(indices)))
            self.assertEqual(sum_tile_type_values(values, tile_types), expected)
            masked = mask_tile_type_values(values, tile_types)
            self.assertEqual(
                masked,
                tuple(v if i in tile_types else 0 for i, v in enumerate(values)),
            )


def _hand_belief(expected: dict[int, int], red: tuple[int, int, int]) -> HandBelief:
    raw = [0] * 34
    for index, value in expected.items():
        raw[index] = value
    return HandBelief(expected_count_raw=tuple(raw), red_five_probability_raw=red)


class ExistingBeliefTypesTest(unittest.TestCase):
    """既存の`HandBelief` / `NonPlayerHiddenBelief` / `TileConservationResult`へ
    同じ牌種集合を適用し、尺度と元データを保つことを固定する。"""

    def test_hand_belief_raw_sum_keeps_fixed_point_scale(self) -> None:
        # 2萬 0.25枚（小数相当）、5萬 3.5枚（赤5確率0.5を含む）、中 4枚。
        belief = _hand_belief(
            {M2: SCALE // 4, M5: 3 * SCALE + SCALE // 2, CHUN: 4 * SCALE},
            (SCALE // 2, 0, 0),
        )
        before = (belief.expected_count_raw, belief.red_five_probability_raw)
        improving = tile_type_set([M2, M5, CHUN])

        total = sum_tile_type_values(belief.expected_count_raw, improving)
        # raw 2048 + 28672 + 32768 = 63488（= 7.75枚。4 * SCALEで切り詰めない）
        self.assertEqual(total, 63488)
        self.assertGreater(total, EXPECTED_COUNT_MAX_RAW)
        self.assertIsInstance(total, int)
        masked = mask_tile_type_values(belief.expected_count_raw, improving)
        self.assertEqual(masked[M2], SCALE // 4)
        self.assertEqual(masked[P5], 0)
        # 赤5は34牌種の5に含まれ、受入合計へ追加加算しない。
        self.assertEqual(
            sum_tile_type_values(belief.expected_count_raw, tile_type_set([M5])),
            belief.expected_count_raw[M5],
        )
        self.assertEqual(
            (belief.expected_count_raw, belief.red_five_probability_raw), before
        )
        self.assertEqual(
            belief.red_five_probability_raw[red_five_index(TileCategory.MANZU)],
            SCALE // 2,
        )

    def test_non_player_hidden_belief_and_conservation(self) -> None:
        remaining = list(STANDARD_TILE_COUNTS)
        remaining[M2] = 3
        accounted = [s - r for s, r in zip(STANDARD_TILE_COUNTS, remaining)]
        conservation = TileConservationResult(
            exact_accounted_counts=tuple(accounted),
            exact_accounted_red_five_counts=(0, 0, 0),
            remaining_tile_counts=tuple(remaining),
            remaining_red_five_counts=(1, 1, 1),
        )
        residual_raw = [count * SCALE for count in remaining]
        residual_raw[M2] = 3 * SCALE - SCALE // 3
        residual = NonPlayerHiddenBelief(
            expected_count_raw=tuple(residual_raw),
            red_five_probability_raw=(SCALE, SCALE, SCALE),
        )
        improving = tile_type_set([M2, M5])

        self.assertEqual(
            sum_tile_type_values(conservation.remaining_tile_counts, improving), 7
        )
        self.assertEqual(
            sum_tile_type_values(residual.expected_count_raw, improving),
            3 * SCALE - SCALE // 3 + 4 * SCALE,
        )
        self.assertEqual(tuple(residual.expected_count_raw), tuple(residual_raw))


if __name__ == "__main__":
    unittest.main()
