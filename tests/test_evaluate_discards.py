"""Issue #224: 打牌候補の一括構造評価（`evaluate_discards_from_canonical_counts`）。

期待値はTile入口の`calculate_shanten()`から独立に作り、Python実装と
native実装（`_lisjong_native`がimportできる場合）を同じ期待値・同じ例外契約で
照合する。一括入口自体を期待値の生成に使わない。
"""

import random
import unittest

import lisjong.hand_evaluation.shanten as shanten_module
from lisjong.hand_evaluation import _shanten_backend, calculate_shanten
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from tests.test_native_shanten_backend import (
    _hand,
    _lisjong_native,
    _random_counts,
    _require_native,
    _run_python,
)

_CATEGORIES = (
    TileCategory.MANZU,
    TileCategory.PINZU,
    TileCategory.SOUZU,
    TileCategory.HONOR,
)
_DISCARD_SIZES = (2, 5, 8, 11, 14)


def _tile(index: int) -> Tile:
    return Tile(TileType(_CATEGORIES[index // 9], index % 9 + 1))


def _tiles(counts) -> list[Tile]:
    return [_tile(index) for index, count in enumerate(counts) for _ in range(count)]


def _counts(tiles) -> list[int]:
    counts = [0] * 34
    for tile in tiles:
        tile_type = tile.tile_type
        counts[_CATEGORIES.index(tile_type.category) * 9 + tile_type.rank - 1] += 1
    return counts


def _expected(counts, discard_indexes, improving_max_shanten):
    """Tile入口だけから作る期待値。"""
    shanten_after = []
    improving_after = []
    for index in discard_indexes:
        after = list(counts)
        after[index] -= 1
        tiles = _tiles(after)
        value = calculate_shanten(tiles)
        shanten_after.append(value)
        if improving_max_shanten is None or value <= improving_max_shanten:
            improving_after.append(
                tuple(
                    drawn
                    for drawn in range(34)
                    if after[drawn] < 4
                    and calculate_shanten([*tiles, _tile(drawn)]) < value
                )
            )
        else:
            improving_after.append(None)
    return tuple(shanten_after), tuple(improving_after)


def _implementations():
    implementations = [("python", shanten_module._python_evaluate_discards)]
    if _lisjong_native is not None:
        table = _shanten_backend.build_native_table(_lisjong_native)
        implementations.append(("native", table.evaluate_discards))
    return implementations


class EvaluateDiscardsContractTest(unittest.TestCase):
    """Python / native実装がTile入口の期待値と一致する（nativeは存在時のみ）。"""

    def _assert_matches(self, counts, discard_indexes, improving_max_shanten) -> None:
        expected = _expected(counts, discard_indexes, improving_max_shanten)
        snapshot = list(counts)
        for name, evaluate in _implementations():
            with self.subTest(
                implementation=name,
                counts=counts,
                discard_indexes=discard_indexes,
                improving_max_shanten=improving_max_shanten,
            ):
                self.assertEqual(
                    evaluate(counts, discard_indexes, improving_max_shanten), expected
                )
                self.assertEqual(counts, snapshot)

    def test_random_hands_of_every_discard_size(self) -> None:
        generator = random.Random(224)
        for size in _DISCARD_SIZES:
            for _ in range(25):
                counts = _random_counts(generator, size)
                held = [index for index, count in enumerate(counts) if count]
                generator.shuffle(held)
                current = calculate_shanten(_tiles(counts))
                for threshold in (None, current):
                    self._assert_matches(counts, held, threshold)

    def test_special_forms_four_copies_and_red_fives(self) -> None:
        for notation in (
            "1133557799m11p3p5z",
            "19m19p19s1234567z1m",
            "1111m2222p3333s44z",
            "123456789p1111s5z",
            "0555s1p123456789m",
            "123m456p789s111z22z",
        ):
            counts = _counts(_hand(notation))
            held = [index for index, count in enumerate(counts) if count]
            current = calculate_shanten(_hand(notation))
            for threshold in (None, current, -1):
                self._assert_matches(counts, held, threshold)

    def test_subsets_order_and_empty_candidates(self) -> None:
        counts = _counts(_hand("123m456p789s1234z5z"))
        self._assert_matches(counts, [31, 0], None)
        self._assert_matches(counts, (1, 31), 1)
        for name, evaluate in _implementations():
            with self.subTest(implementation=name):
                self.assertEqual(evaluate(counts, []), ((), ()))
                # iterableを受け付け、出力は入力順に対応する。
                self.assertEqual(
                    evaluate(counts, iter([31, 0])), evaluate(counts, (31, 0))
                )

    def test_unevaluated_is_none_and_distinct_from_evaluated(self) -> None:
        # 和了形（向聴-1）: thresholdが-1なら打牌後向聴0の候補は未評価（None）、
        # thresholdなしでは評価済みで空でない。
        counts = _counts(_hand("123m456p789s111z22z"))
        for name, evaluate in _implementations():
            with self.subTest(implementation=name):
                shanten_after, improving = evaluate(counts, [28, 0], -1)
                self.assertEqual(improving, (None, None))
                self.assertTrue(all(value >= 0 for value in shanten_after))
                _, improving = evaluate(counts, [28, 0], None)
                self.assertTrue(all(len(item) > 0 for item in improving))

    def test_invalid_inputs_raise_the_same_exception_types(self) -> None:
        valid = _counts(_hand("123m456p789s1234z5z"))
        too_many = list(valid)
        too_many[0] = 5
        too_many[27:31] = [0, 0, 0, 0]
        thirteen = list(valid)
        thirteen[31] -= 1
        cases = (
            (ValueError, valid[:33], [0], None),
            (ValueError, too_many, [1], None),
            (ValueError, thirteen, [0], None),
            (TypeError, valid, [True], None),
            (TypeError, valid, ["0"], None),
            (TypeError, valid, [0.0], None),
            (ValueError, valid, [-1], None),
            (ValueError, valid, [34], None),
            (ValueError, valid, [2**70], None),
            (ValueError, valid, [0, 0], None),
            (ValueError, valid, [8], None),
            (TypeError, valid, [0], True),
            (TypeError, valid, [0], 0.5),
            (ValueError, valid, [0], 2**70),
        )
        for name, evaluate in _implementations():
            for error, counts, indexes, threshold in cases:
                with self.subTest(
                    implementation=name, indexes=indexes, threshold=threshold
                ):
                    snapshot = list(counts)
                    with self.assertRaises(error):
                        evaluate(counts, indexes, threshold)
                    self.assertEqual(counts, snapshot)

    def test_selected_backend_dispatch(self) -> None:
        expected = (
            shanten_module._python_evaluate_discards
            if _shanten_backend.native_evaluate_discards is None
            else _shanten_backend.native_evaluate_discards
        )
        self.assertIs(shanten_module._evaluate_discards, expected)


class NativeEvaluateDiscardsCounterTest(unittest.TestCase):
    """nativeの計数器: 実計算回数と境界呼び出し回数を別々に数える。"""

    def setUp(self) -> None:
        _require_native(self)

    def test_counters_separate_boundary_calls_and_evaluations(self) -> None:
        table = _shanten_backend.build_native_table(_lisjong_native)
        counts = _counts(_hand("123m456p789s1234z5z"))
        evaluations = _lisjong_native.standard_shanten_call_count()
        calls = _lisjong_native.discard_evaluation_call_count()
        # 候補2つ（打牌後2回）、改善牌は1候補だけ（手中4枚の牌種なし: 34回）。
        shanten_after, improving = table.evaluate_discards(counts, [31, 0], -99)
        self.assertEqual(improving, (None, None))
        table.evaluate_discards(counts, [31], None)
        self.assertEqual(_lisjong_native.discard_evaluation_call_count(), calls + 2)
        self.assertEqual(
            _lisjong_native.standard_shanten_call_count(), evaluations + 2 + 1 + 34
        )
        # 失敗した呼び出しはどちらも数えない。
        with self.assertRaises(ValueError):
            table.evaluate_discards(counts, [8])
        self.assertEqual(_lisjong_native.discard_evaluation_call_count(), calls + 2)


class NativeApiVersionSelectionTest(unittest.TestCase):
    """rust選択時のnative API version検査（native拡張の有無によらず実行する）。"""

    def _run_with_fake_native(self, api_line: str, backend: str):
        return _run_python(
            "import sys, types\n"
            "fake = types.ModuleType('_lisjong_native')\n"
            f"{api_line}\n"
            "sys.modules['_lisjong_native'] = fake\n"
            "import lisjong.hand_evaluation\n"
            "import lisjong.policies\n",
            backend=backend,
        )

    def test_old_wheel_without_api_version_fails_closed_under_rust(self) -> None:
        result = self._run_with_fake_native("pass", "rust")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ShantenBackendError", result.stderr)
        self.assertIn("API_VERSION 2, got 1", result.stderr)

    def test_mismatched_api_version_fails_closed_under_rust(self) -> None:
        result = self._run_with_fake_native("fake.API_VERSION = 3", "rust")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ShantenBackendError", result.stderr)

    def test_python_backend_ignores_an_old_wheel(self) -> None:
        result = self._run_with_fake_native("pass", "python")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
