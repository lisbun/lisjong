"""Issue #332: 仕様由来fixtureで向聴numeric coreの3入口を固定する。

同じ`shanten_spec_fixtures`を次の入口へ通し、手計算で固定した期待値と比較する。

- 公開Tile入口: `calculate_shanten()`
- trusted canonical-count入口: `calculate_shanten_from_canonical_counts()`
- native numeric入口: `StandardShantenTable.shanten_from_valid_counts()`

期待値にPython lookup / 再帰DFS / 外部libraryの計算出力を使わない（このmodule
はそれらをimportしない。`NoComputedOracleTest`で固定）。native tableの構築には
既存のPython artifact読込（`_shanten_backend.build_native_table`）を使うが、
これは入力tableの読込であって期待値の計算ではない。

native入口は`_lisjong_native`がimportできる環境で実行し、
`LISJONG_REQUIRE_NATIVE=1`ではskipせずfailさせる。Rust backend選択の
full suite（native CI job）では公開入口・canonical入口もnative numeric coreを
通ったことをcall counterで確認する。
"""

import ast
import os
import pathlib
import unittest

from shanten_spec_fixtures import (
    FIFTH_COPY_EXCLUDED,
    QUAD_IS_ONE_PAIR,
    RED_FIVE,
    SEVEN_PAIRS,
    SHANTEN_SPEC_FIXTURES,
    SPECIAL_FORMS_IGNORED,
    STANDARD,
    THIRTEEN_ORPHANS,
    ShantenSpecFixture,
)

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.hand_evaluation import _shanten_backend, calculate_shanten
from lisjong.hand_evaluation.shanten import calculate_shanten_from_canonical_counts
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

try:
    import _lisjong_native
except ImportError:
    _lisjong_native = None

_REQUIRE_NATIVE = os.environ.get("LISJONG_REQUIRE_NATIVE") == "1"
_VALID_HAND_SIZES = frozenset({1, 2, 4, 5, 7, 8, 10, 11, 13, 14})
_SUIT_BY_SUFFIX = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}
_NUMBER_SUITS = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)


def _hand(notation: str) -> list[Tile]:
    """`"123m0p"`形式（`0`は赤5）をTile listへ展開する。"""
    tiles: list[Tile] = []
    ranks: list[int] = []
    for character in notation:
        if character.isdigit():
            ranks.append(int(character))
            continue
        category = _SUIT_BY_SUFFIX[character]
        for rank in ranks:
            red = rank == 0
            tiles.append(Tile(TileType(category, 5 if red else rank), is_red=red))
        ranks = []
    if ranks:
        raise ValueError(f"notation has trailing ranks: {notation!r}")
    return tiles


def _counts(tiles: list[Tile]) -> list[int]:
    """canonical axisの正本`tile_type_index()`でTile列を34 countへ変換する。"""
    counts = [0] * 34
    for tile in tiles:
        counts[tile_type_index(tile.tile_type)] += 1
    return counts


def _fixed_meld_count(concealed_tile_count: int) -> int:
    return 4 - (concealed_tile_count - 1) // 3


def _require_native(test: unittest.TestCase) -> None:
    if _lisjong_native is not None:
        return
    if _REQUIRE_NATIVE:
        test.fail("LISJONG_REQUIRE_NATIVE=1 but _lisjong_native is not importable")
    test.skipTest("opt-in native extension _lisjong_native is not installed")


def _permute_suits(tiles: list[Tile], order: tuple[TileCategory, ...]) -> list[Tile]:
    mapping = dict(zip(_NUMBER_SUITS, order, strict=True))
    return [
        tile
        if tile.tile_type.category is TileCategory.HONOR
        else Tile(
            TileType(mapping[tile.tile_type.category], tile.tile_type.rank),
            is_red=tile.is_red,
        )
        for tile in tiles
    ]


def _reflect_ranks(tiles: list[Tile]) -> list[Tile]:
    """数牌の1↔9（rank r → 10 - r）反転。字牌と赤5（5 → 5）はそのまま。"""
    return [
        tile
        if tile.tile_type.category is TileCategory.HONOR
        else Tile(
            TileType(tile.tile_type.category, 10 - tile.tile_type.rank),
            is_red=tile.is_red,
        )
        for tile in tiles
    ]


def _toggle_red_fives(tiles: list[Tile]) -> list[Tile]:
    """数牌の5を赤5⇔通常5に入れ替える（基礎牌種は変えない）。"""
    return [
        Tile(tile.tile_type, is_red=not tile.is_red)
        if tile.tile_type.category is not TileCategory.HONOR
        and tile.tile_type.rank == 5
        else tile
        for tile in tiles
    ]


class FixtureDataTest(unittest.TestCase):
    """fixture自体の形とcoverageを固定する。"""

    def test_fixtures_are_well_formed(self) -> None:
        names = [fixture.name for fixture in SHANTEN_SPEC_FIXTURES]
        self.assertEqual(len(names), len(set(names)))
        known_tags = {
            STANDARD,
            SEVEN_PAIRS,
            THIRTEEN_ORPHANS,
            QUAD_IS_ONE_PAIR,
            FIFTH_COPY_EXCLUDED,
            SPECIAL_FORMS_IGNORED,
            RED_FIVE,
        }
        for fixture in SHANTEN_SPEC_FIXTURES:
            with self.subTest(fixture=fixture.name):
                tiles = _hand(fixture.notation)
                self.assertIn(len(tiles), _VALID_HAND_SIZES)
                self.assertLessEqual(max(_counts(tiles)), 4)
                self.assertGreaterEqual(fixture.expected, -1)
                self.assertTrue(fixture.rationale.strip())
                self.assertTrue(fixture.tags)
                self.assertLessEqual(fixture.tags, known_tags)
                if fixture.tags & {SEVEN_PAIRS, THIRTEEN_ORPHANS}:
                    self.assertIn(len(tiles), (13, 14))
                if RED_FIVE in fixture.tags:
                    self.assertTrue(any(tile.is_red for tile in tiles))

    def test_fixtures_cover_the_issue_boundaries(self) -> None:
        sizes = {len(_hand(f.notation)) for f in SHANTEN_SPEC_FIXTURES}
        self.assertEqual(sizes, _VALID_HAND_SIZES)
        self.assertEqual({_fixed_meld_count(size) for size in sizes}, set(range(5)))

        values = {f.expected for f in SHANTEN_SPEC_FIXTURES}
        self.assertLessEqual({-1, 0, 1}, values)
        self.assertTrue(any(value >= 2 for value in values))
        for fixed in range(5):
            fixed_values = {
                f.expected
                for f in SHANTEN_SPEC_FIXTURES
                if _fixed_meld_count(len(_hand(f.notation))) == fixed
            }
            with self.subTest(fixed_meld_count=fixed):
                self.assertLessEqual({-1, 0}, fixed_values)

        tags = set().union(*(f.tags for f in SHANTEN_SPEC_FIXTURES))
        self.assertEqual(
            tags,
            {
                STANDARD,
                SEVEN_PAIRS,
                THIRTEEN_ORPHANS,
                QUAD_IS_ONE_PAIR,
                FIFTH_COPY_EXCLUDED,
                SPECIAL_FORMS_IGNORED,
                RED_FIVE,
            },
        )
        for tag in (SEVEN_PAIRS, THIRTEEN_ORPHANS):
            tagged = {f.expected for f in SHANTEN_SPEC_FIXTURES if tag in f.tags}
            with self.subTest(tag=tag):
                self.assertLessEqual({-1, 0, 1}, tagged)


class NoComputedOracleTest(unittest.TestCase):
    """期待値がPython lookup / DFS / 外部libraryの計算出力に依存しない。"""

    def test_spec_modules_do_not_import_computed_oracles(self) -> None:
        forbidden = {
            "_python_shanten",
            "_lookup_shanten",
            "_shanten_frontier",
            "_python_shanten_from_valid_counts",
            "mahjong",
        }
        tests_dir = pathlib.Path(__file__).resolve().parent
        for name in ("shanten_spec_fixtures.py", pathlib.Path(__file__).name):
            tree = ast.parse((tests_dir / name).read_text(encoding="utf-8"))
            imported: set[str] = set()
            attributes: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.update(alias.name.split("."))
                elif isinstance(node, ast.ImportFrom):
                    imported.update((node.module or "").split("."))
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.Attribute):
                    attributes.add(node.attr)
            with self.subTest(module=name):
                self.assertFalse(imported & forbidden, sorted(imported & forbidden))
                self.assertFalse(attributes & forbidden, sorted(attributes & forbidden))


class _EntryAssertions:
    """各入口へfixtureと変換後fixtureを通す共通部分。subclassが`_shanten`を持つ。"""

    def _shanten(self, tiles: list[Tile]) -> int:
        raise NotImplementedError

    def _assert_fixture(self, fixture: ShantenSpecFixture, tiles: list[Tile]) -> None:
        self.assertEqual(
            self._shanten(tiles),
            fixture.expected,
            msg=f"{fixture.name} ({fixture.notation}): {fixture.rationale}",
        )

    def test_spec_fixtures(self) -> None:
        for fixture in SHANTEN_SPEC_FIXTURES:
            with self.subTest(fixture=fixture.name):
                self._assert_fixture(fixture, _hand(fixture.notation))

    def test_suit_permutation_invariance(self) -> None:
        # 色の置換で通常形・七対子・国士の構造は変わらない。
        permutations = (
            (TileCategory.PINZU, TileCategory.SOUZU, TileCategory.MANZU),
            (TileCategory.SOUZU, TileCategory.MANZU, TileCategory.PINZU),
            (TileCategory.PINZU, TileCategory.MANZU, TileCategory.SOUZU),
        )
        for fixture in SHANTEN_SPEC_FIXTURES:
            for order in permutations:
                with self.subTest(fixture=fixture.name, order=order):
                    self._assert_fixture(
                        fixture, _permute_suits(_hand(fixture.notation), order)
                    )

    def test_terminal_reflection_invariance(self) -> None:
        # 数牌のrank r → 10 - rは順子・塔子の隣接関係と么九牌集合を保つ。
        for fixture in SHANTEN_SPEC_FIXTURES:
            with self.subTest(fixture=fixture.name):
                self._assert_fixture(fixture, _reflect_ranks(_hand(fixture.notation)))

    def test_input_order_invariance(self) -> None:
        for fixture in SHANTEN_SPEC_FIXTURES:
            tiles = _hand(fixture.notation)
            rotated = tiles[len(tiles) // 2 :] + tiles[: len(tiles) // 2]
            for label, ordered in (("reversed", tiles[::-1]), ("rotated", rotated)):
                with self.subTest(fixture=fixture.name, order=label):
                    self._assert_fixture(fixture, ordered)

    def test_red_and_normal_five_are_aggregated(self) -> None:
        # 5を含むfixtureで赤5⇔通常5を入れ替えても同じ牌種として数える。
        toggled_any = False
        for fixture in SHANTEN_SPEC_FIXTURES:
            tiles = _hand(fixture.notation)
            toggled = _toggle_red_fives(tiles)
            if toggled == tiles:
                continue
            toggled_any = True
            with self.subTest(fixture=fixture.name):
                self._assert_fixture(fixture, toggled)
        self.assertTrue(toggled_any)


class PublicTileEntryTest(_EntryAssertions, unittest.TestCase):
    def _shanten(self, tiles: list[Tile]) -> int:
        return calculate_shanten(tiles)


class TrustedCanonicalCountEntryTest(_EntryAssertions, unittest.TestCase):
    def _shanten(self, tiles: list[Tile]) -> int:
        return calculate_shanten_from_canonical_counts(_counts(tiles))


class NativeNumericEntryTest(_EntryAssertions, unittest.TestCase):
    """native numeric入口を直接呼ぶ。backend選択に依存しない。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.table = (
            None
            if _lisjong_native is None
            else _shanten_backend.build_native_table(_lisjong_native)
        )

    def setUp(self) -> None:
        _require_native(self)

    def _shanten(self, tiles: list[Tile]) -> int:
        counts = _counts(tiles)
        before = _lisjong_native.standard_shanten_call_count()
        value = self.table.shanten_from_valid_counts(counts, sum(counts))
        self.assertEqual(_lisjong_native.standard_shanten_call_count(), before + 1)
        return value

    def test_native_boundary_rejects_malformed_counts(self) -> None:
        # 既存のnative境界契約（Issue #213）。trusted Python入口へvalidationを
        # 足すものではなく、例外型もnative側の現状どおりに確認する。
        valid = [0] * 34
        valid[0] = 1
        cases = (
            ("short", valid[:-1], 1, ValueError),
            ("long", valid + [0], 1, ValueError),
            ("above four", [5] + [0] * 33, 5, ValueError),
            ("negative", [-1, 2] + [0] * 32, 1, ValueError),
            ("non-int count", ["1"] + [0] * 33, 1, TypeError),
            ("not a sequence", 12, 1, TypeError),
            ("size mismatch", valid, 2, ValueError),
            ("invalid size", [3] + [0] * 33, 3, ValueError),
            ("empty", [0] * 34, 0, ValueError),
        )
        for label, counts, concealed, error in cases:
            with self.subTest(case=label):
                with self.assertRaises(error):
                    self.table.shanten_from_valid_counts(counts, concealed)


class SelectedBackendRoutingTest(unittest.TestCase):
    """Rust backend選択processで公開・canonical入口がnativeを通ったことを確認する。"""

    def setUp(self) -> None:
        if _shanten_backend.BACKEND_NAME != "rust":
            self.skipTest("only meaningful when LISJONG_SHANTEN_BACKEND=rust")

    def test_public_and_canonical_entries_use_the_native_core(self) -> None:
        fixture = SHANTEN_SPEC_FIXTURES[0]
        tiles = _hand(fixture.notation)
        for label, call in (
            ("public", lambda: calculate_shanten(tiles)),
            (
                "canonical",
                lambda: calculate_shanten_from_canonical_counts(_counts(tiles)),
            ),
        ):
            with self.subTest(entry=label):
                before = _lisjong_native.standard_shanten_call_count()
                self.assertEqual(call(), fixture.expected)
                self.assertEqual(
                    _lisjong_native.standard_shanten_call_count(), before + 1
                )


if __name__ == "__main__":
    unittest.main()
