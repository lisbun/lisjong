import ast
import os
import pathlib
import random
import struct
import subprocess
import sys
import unittest

import lisjong.hand_evaluation.shanten as shanten_module
from lisjong.belief.canonical_axes import tile_type_index
from lisjong.hand_evaluation import (
    _shanten_backend,
    _shanten_common,
    _shanten_frontier,
    _shanten_table,
    calculate_shanten,
)
from lisjong.hand_evaluation.shanten import calculate_shanten_from_canonical_counts
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

_CATEGORY_BY_SUFFIX = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}


def hand(notation: str) -> list[Tile]:
    """`"123m11p1z"`のようなtest記法をTile listへ展開する。

    赤5は`"0"`と書く。牌種と枚数だけを表すtest用の入力記法であり、lisjongの
    公開契約ではない。
    """
    tiles: list[Tile] = []
    ranks: list[int] = []
    for character in notation:
        if character.isdigit():
            ranks.append(int(character))
            continue
        category = _CATEGORY_BY_SUFFIX[character]
        for rank in ranks:
            if rank == 0:
                tiles.append(Tile(TileType(category, 5), is_red=True))
            else:
                tiles.append(Tile(TileType(category, rank)))
        ranks = []
    if ranks:
        raise ValueError(f"notation has trailing ranks: {notation!r}")
    return tiles


class HandNotationTest(unittest.TestCase):
    """test記法自体が意図どおりTileへ展開されることを確認する。"""

    def test_expands_each_suit_and_red_five(self) -> None:
        self.assertEqual(
            hand("10m1z"),
            [
                Tile(TileType(TileCategory.MANZU, 1)),
                Tile(TileType(TileCategory.MANZU, 5), is_red=True),
                Tile(TileType(TileCategory.HONOR, 1)),
            ],
        )

    def test_rejects_trailing_ranks(self) -> None:
        with self.assertRaises(ValueError):
            hand("123")


class StandardShantenTest(unittest.TestCase):
    def test_completed_hand_is_minus_one(self) -> None:
        self.assertEqual(calculate_shanten(hand("123456789m11122p")), -1)
        self.assertEqual(calculate_shanten(hand("111222333444m55p")), -1)

    def test_tenpai_is_zero(self) -> None:
        cases = (
            "123456789m1122p",  # シャンポン待ち
            "123456789m11p13s",  # 嵌張待ち
            "123456789m11p34s",  # 両面待ち
            "123456789m111p2s",  # タンキ待ち
            "123456789m11p24s7z",  # 14枚。7zを切って嵌張聴牌。
        )
        for notation in cases:
            with self.subTest(notation=notation):
                self.assertEqual(calculate_shanten(hand(notation)), 0)

    def test_one_shanten(self) -> None:
        cases = (
            "123456789m1p24s7z",  # 3面子 + 嵌張。雀頭が無い。
            "123456789m12p45s7z",  # 14枚。3面子 + 塔子2つで雀頭が無い。
        )
        for notation in cases:
            with self.subTest(notation=notation):
                self.assertEqual(calculate_shanten(hand(notation)), 1)

    def test_two_shanten(self) -> None:
        # 3面子だけで、雀頭も4つ目の面子候補も無い。
        self.assertEqual(calculate_shanten(hand("123456789m1p4p7p1s")), 2)

    def test_distant_hands_with_fixed_melds(self) -> None:
        # 副露済みの手では七対子・国士が下限を作らないので、通常形の遠さが出る。
        self.assertEqual(calculate_shanten(hand("147m147p147s1z")), 6)
        self.assertEqual(calculate_shanten(hand("147m147p1s")), 4)

    def test_overlapping_meld_pair_and_partial_candidates(self) -> None:
        """面子・雀頭・塔子候補が重なる牌姿でgreedyな誤判定をしない。"""
        # 234m + 345m と取ると1向聴だが、234m + 234m + 55mシャンポンなら聴牌。
        self.assertEqual(calculate_shanten(hand("22334455m678p99s")), 0)
        # 111m + 222m と取ると雀頭が消える。111m + 12m + 22m で聴牌になる。
        self.assertEqual(calculate_shanten(hand("1111m222m")), 0)
        # 111222333m は3刻子とも3順子とも解釈でき、44mを雀頭に残せる。
        self.assertEqual(calculate_shanten(hand("111222333m44m567p")), -1)


class ImpossibleFifthCopyTest(unittest.TestCase):
    """同じ牌種を5枚必要とする分解を、和了への最短経路に数えないことを確認する。"""

    def test_four_copies_cannot_become_the_pair(self) -> None:
        # 確定面子3個 + 1111m。111mの余り1枚は5枚目を引けないので雀頭にできず、
        # 11m雀頭 + 11mシャンポンも同じ牌種を5枚必要とする。
        self.assertEqual(calculate_shanten(hand("1111m")), 1)
        # 余り牌が別の牌種なら、その牌でタンキ聴牌になる。
        self.assertEqual(calculate_shanten(hand("111m2m")), 0)
        # 牌種が分かれていればシャンポン聴牌として成立する。
        self.assertEqual(calculate_shanten(hand("1122m")), 0)

    def test_four_copies_with_a_separate_triplet(self) -> None:
        self.assertEqual(calculate_shanten(hand("1111p666z")), 1)

    def test_dead_honor_spares_cannot_seed_missing_blocks(self) -> None:
        # 字牌の4枚使いは刻子1つ分にしかならず、余り1枚は面子にも雀頭にも育たない。
        # 種牌として使えるのは1mだけで、面子枠と雀頭の両方は賄えない。
        self.assertEqual(calculate_shanten(hand("111122223333z1m")), 3)


class SevenPairsTest(unittest.TestCase):
    def test_completed_hand(self) -> None:
        self.assertEqual(calculate_shanten(hand("11m22m33p44s55z66z77z")), -1)

    def test_tenpai(self) -> None:
        self.assertEqual(calculate_shanten(hand("11m22m33p44s55z66z7z")), 0)

    def test_kind_count_correction_below_seven_kinds(self) -> None:
        # 対子は6つあるが牌種も6しかないため、牌種数補正で1向聴になる。
        self.assertEqual(calculate_shanten(hand("111m22m33p44s55z66z")), 1)

    def test_four_copies_count_as_a_single_pair(self) -> None:
        # 1111mは対子1つ分にしかならず、牌種も1しか増えない。
        self.assertEqual(calculate_shanten(hand("1111m22m33p44s55z66z")), 1)


class ThirteenOrphansTest(unittest.TestCase):
    def test_completed_hand(self) -> None:
        self.assertEqual(calculate_shanten(hand("19m19p19s12345677z")), -1)

    def test_thirteen_kinds_is_tenpai(self) -> None:
        self.assertEqual(calculate_shanten(hand("19m19p19s1234567z")), 0)

    def test_twelve_kinds_with_a_pair_is_tenpai(self) -> None:
        self.assertEqual(calculate_shanten(hand("19m19p19s1234556z")), 0)

    def test_missing_one_kind_without_a_pair(self) -> None:
        self.assertEqual(calculate_shanten(hand("19m19p19s123456z5m")), 1)

    def test_far_from_thirteen_orphans(self) -> None:
        self.assertEqual(calculate_shanten(hand("19m19p19s1234z456m")), 3)


class CombinedHandTypeTest(unittest.TestCase):
    def test_picks_seven_pairs_over_standard_form(self) -> None:
        self.assertEqual(calculate_shanten(hand("1199m1199p1199s1z")), 0)

    def test_picks_thirteen_orphans_over_seven_pairs(self) -> None:
        # 七対子なら6向聴、国士なら6向聴、通常形なら8向聴になる牌姿。
        self.assertEqual(calculate_shanten(hand("147m147p147s1234z")), 6)

    def test_picks_standard_form_over_seven_pairs(self) -> None:
        self.assertEqual(calculate_shanten(hand("123456789m1122p")), 0)

    def test_special_hand_types_are_ignored_with_fixed_melds(self) -> None:
        """確定面子がある手では、七対子・国士を候補にしない。"""
        # 么九牌10種。13枚の手なら国士3向聴だが、確定面子1個の10枚では
        # 通常形だけを評価するため6向聴になる。
        self.assertEqual(calculate_shanten(hand("19m19p19s1234z")), 6)


class FixedMeldTest(unittest.TestCase):
    """副露・槓で確定した面子を、純手牌枚数だけから扱えることを確認する。"""

    def test_one_fixed_meld(self) -> None:
        self.assertEqual(calculate_shanten(hand("123456789m11p")), -1)
        self.assertEqual(calculate_shanten(hand("123456789m1p")), 0)
        self.assertEqual(calculate_shanten(hand("12345678m11p")), 0)
        self.assertEqual(calculate_shanten(hand("12356m89p11s4z")), 1)

    def test_two_fixed_melds(self) -> None:
        self.assertEqual(calculate_shanten(hand("123456m11p")), -1)
        self.assertEqual(calculate_shanten(hand("123456m1p")), 0)
        self.assertEqual(calculate_shanten(hand("12m56m11p4z")), 1)

    def test_three_fixed_melds(self) -> None:
        self.assertEqual(calculate_shanten(hand("123m11p")), -1)
        self.assertEqual(calculate_shanten(hand("123m1p")), 0)
        self.assertEqual(calculate_shanten(hand("11m59p")), 1)

    def test_four_fixed_melds(self) -> None:
        self.assertEqual(calculate_shanten(hand("11p")), -1)
        self.assertEqual(calculate_shanten(hand("1p")), 0)
        self.assertEqual(calculate_shanten(hand("1m9p")), 0)

    def test_meld_kind_is_never_required(self) -> None:
        """Chi / Pon / Kanの区別を渡さずに、同じ純手牌なら同じ結果になる。

        11枚の純手牌がChiの結果でもPonの結果でもAnkanの結果でも、
        `calculate_shanten()`へ渡すのは純手牌だけである。
        """
        concealed = hand("234567m11p234s")
        self.assertEqual(len(concealed), 11)
        self.assertEqual(calculate_shanten(concealed), -1)


class RedFiveTest(unittest.TestCase):
    def test_red_five_does_not_change_shanten(self) -> None:
        cases = (
            ("123456789m11p345s", "123456789m11p340s"),
            ("345m11p123s456p", "340m11p123s456p"),
            ("555m11p123s456p", "055m11p123s456p"),
        )
        for notation, red_notation in cases:
            with self.subTest(notation=notation):
                self.assertEqual(
                    calculate_shanten(hand(notation)),
                    calculate_shanten(hand(red_notation)),
                )

    def test_red_five_shares_the_base_tile_kind_count(self) -> None:
        # 通常5が3枚 + 赤5が1枚で、5mの基礎牌種としては4枚使いになる。
        self.assertEqual(calculate_shanten(hand("5550m")), 1)
        self.assertEqual(calculate_shanten(hand("1111m")), 1)


class InputOrderTest(unittest.TestCase):
    def test_shuffled_input_gives_the_same_result(self) -> None:
        rng = random.Random(50)
        for notation in (
            "123456789m1122p",
            "11m22m33p44s55z66z7z",
            "19m19p19s1234567z",
            "234567m11p234s",
            "1111m222m",
        ):
            tiles = hand(notation)
            expected = calculate_shanten(tiles)
            for _ in range(20):
                shuffled = tiles[:]
                rng.shuffle(shuffled)
                with self.subTest(notation=notation, shuffled=tuple(shuffled)):
                    self.assertEqual(calculate_shanten(shuffled), expected)

    def test_accepts_any_iterable(self) -> None:
        tiles = hand("123456789m1122p")
        expected = calculate_shanten(tiles)
        self.assertEqual(calculate_shanten(iter(tiles)), expected)
        self.assertEqual(calculate_shanten(tile for tile in tiles), expected)
        self.assertEqual(calculate_shanten(tuple(tiles)), expected)


class InvalidInputTest(unittest.TestCase):
    def test_rejects_non_iterable(self) -> None:
        for value in (None, 13, object()):
            with self.subTest(value=value), self.assertRaises(TypeError):
                calculate_shanten(value)

    def test_rejects_non_tile_elements(self) -> None:
        tiles = hand("123456789m1122p")
        with self.assertRaises(TypeError):
            calculate_shanten([*tiles[:-1], "1p"])
        with self.assertRaises(TypeError):
            calculate_shanten([*tiles[:-1], TileType(TileCategory.PINZU, 1)])

    def test_rejects_impossible_tile_counts(self) -> None:
        supply = hand("123456789m123456789p12s")
        for count in (0, 3, 6, 9, 12, 15, 16, 20):
            with self.subTest(count=count), self.assertRaises(ValueError):
                calculate_shanten(supply[:count])

    def test_accepts_every_possible_tile_count(self) -> None:
        supply = hand("123456789m123456789p")
        for count in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14):
            with self.subTest(count=count):
                self.assertIsInstance(calculate_shanten(supply[:count]), int)

    def test_rejects_five_copies_of_one_base_tile_kind(self) -> None:
        with self.assertRaises(ValueError):
            calculate_shanten(hand("11111m23p"))

    def test_rejects_five_copies_across_normal_and_red_five(self) -> None:
        # 通常5が4枚 + 赤5が1枚で、基礎牌種としては5枚になる。
        with self.assertRaises(ValueError):
            calculate_shanten(hand("55550m23p"))

    def test_element_type_is_checked_before_hand_size(self) -> None:
        with self.assertRaises(TypeError):
            calculate_shanten(["1m", "2m", "3m"])


_TILE_KIND_COUNT = 34
"""canonical 34牌種axisの要素数（test-local定数）。"""


def _canonical_counts(tiles: list[Tile]) -> list[int]:
    """既存canonical helperだけを使ってTile列を34牌種countへ変換する。

    `hand_evaluation`側のindex mappingを再実装せず、`belief.canonical_axes`の
    `tile_type_index()`をcanonical axisの正本として使う。これにより、
    FiniteHorizon / beliefが使うaxisと`hand_evaluation` backendの解釈が
    一致していること（axis driftがないこと）をtestで固定できる。
    """
    counts = [0] * _TILE_KIND_COUNT
    for tile in tiles:
        counts[tile_type_index(tile.tile_type)] += 1
    return counts


def _corpus_notation(seed_index: int, size: int) -> list[Tile]:
    """乱数を使わない決定的な規則で作る、再現可能なcorpus hand。"""
    counts = [0] * _TILE_KIND_COUNT
    index = seed_index % _TILE_KIND_COUNT
    step = 1 + seed_index % 7
    total = 0
    while total < size:
        if counts[index] < 4:
            counts[index] += 1
            total += 1
        index = (index + step) % _TILE_KIND_COUNT
    tiles: list[Tile] = []
    for tile_index, count in enumerate(counts):
        for category, base, span in (
            (TileCategory.MANZU, 0, 9),
            (TileCategory.PINZU, 9, 9),
            (TileCategory.SOUZU, 18, 9),
            (TileCategory.HONOR, 27, 7),
        ):
            if base <= tile_index < base + span:
                tiles.extend([Tile(TileType(category, tile_index - base + 1))] * count)
                break
    return tiles


class CanonicalCountHotPathTest(unittest.TestCase):
    """Tile公開APIとpackage-internal count-native pathの一致を固定する。

    count-native pathは新しい向聴semanticではなく、同じsemantic coreへの
    別入口である（Issue #113）。したがって同じ牌姿に対して両者は常に同じ値を
    返さなければならない。
    """

    def _assert_paths_agree(self, tiles: list[Tile]) -> int:
        expected = calculate_shanten(tiles)
        actual = calculate_shanten_from_canonical_counts(_canonical_counts(tiles))
        self.assertEqual(
            actual,
            expected,
            msg=f"count-native path disagreed with the Tile API for {tiles}",
        )
        return expected

    def test_standard_forms_agree(self) -> None:
        fixtures = {
            "123456789m123p11s": -1,
            "123456789m123p1s2s": 0,
            "123456789m123p1s3s": 0,
            "19m19p19s1234567z": 0,
            "123456789m12p13s5z": 1,
        }
        for notation, expected in fixtures.items():
            with self.subTest(notation=notation):
                self.assertEqual(self._assert_paths_agree(hand(notation)), expected)

    def test_chiitoitsu_and_kokushi_routes_agree(self) -> None:
        for notation in (
            "1199m2288p3355s6z",
            "1199m2288p3355s66z",
            "19m19p19s1234567z",
            "119m19p19s123456z",
            "1122334455667z",
        ):
            with self.subTest(notation=notation):
                self._assert_paths_agree(hand(notation))

    def test_every_fixed_meld_context_agrees(self) -> None:
        fixtures = {
            0: "13579m13p246s777z",
            1: "13579m13p246s",
            2: "1357m13p24s",
            3: "13m57p7z",
            4: "1m5p",
        }
        for fixed_melds, notation in fixtures.items():
            tiles = hand(notation)
            with self.subTest(fixed_melds=fixed_melds, size=len(tiles)):
                self.assertEqual(4 - (len(tiles) - 1) // 3, fixed_melds)
                self._assert_paths_agree(tiles)

    def test_shanten_depth_range_agrees(self) -> None:
        # complete / tenpai / 1-shanten / deeper shanten をまたいで一致する。
        observed = set()
        for notation in (
            "123456789m123p11s",
            "123456789m123p1s2s",
            "123456789m12p13s5z",
            "147m258p369s1357z",
            "147m2588p369s135z",
        ):
            with self.subTest(notation=notation):
                observed.add(self._assert_paths_agree(hand(notation)))
        self.assertLessEqual({-1, 0, 1}, observed)
        self.assertTrue(any(value >= 2 for value in observed))

    def test_red_and_normal_five_are_structurally_equivalent(self) -> None:
        red = hand("123406789m123p11s")
        normal = hand("123456789m123p11s")

        self.assertEqual(_canonical_counts(red), _canonical_counts(normal))
        self.assertEqual(
            calculate_shanten_from_canonical_counts(_canonical_counts(red)),
            calculate_shanten_from_canonical_counts(_canonical_counts(normal)),
        )
        self._assert_paths_agree(red)

    def test_deterministic_corpus_agrees_on_the_canonical_axis(self) -> None:
        # canonical axisがずれると通常形以外の解釈が食い違うため、決定的な
        # corpus全体でTile path == count-native pathを固定する。
        for size in (14, 13, 11, 10, 8, 7, 5, 4, 2, 1):
            for seed_index in range(_TILE_KIND_COUNT):
                tiles = _corpus_notation(seed_index, size)
                self._assert_paths_agree(tiles)

    def test_seeded_sample_corpus_agrees(self) -> None:
        generator = random.Random(20260824)
        for size in (14, 13, 11, 8, 5, 2):
            for _ in range(12):
                counts = [0] * _TILE_KIND_COUNT
                total = 0
                while total < size:
                    index = generator.randrange(_TILE_KIND_COUNT)
                    if counts[index] == 4:
                        continue
                    counts[index] += 1
                    total += 1
                tiles: list[Tile] = []
                for tile_index, count in enumerate(counts):
                    for category, base, span in (
                        (TileCategory.MANZU, 0, 9),
                        (TileCategory.PINZU, 9, 9),
                        (TileCategory.SOUZU, 18, 9),
                        (TileCategory.HONOR, 27, 7),
                    ):
                        if base <= tile_index < base + span:
                            tiles.extend(
                                [Tile(TileType(category, tile_index - base + 1))]
                                * count
                            )
                            break
                self._assert_paths_agree(tiles)

    def test_count_native_path_accepts_a_tuple_state(self) -> None:
        # FiniteHorizonのDP stateはtupleなので、tuple入力でも同じ値を返す。
        tiles = hand("123456789m123p11s")
        counts = _canonical_counts(tiles)

        self.assertEqual(
            calculate_shanten_from_canonical_counts(tuple(counts)),
            calculate_shanten(tiles),
        )


def _single_group_counts(span: int, seed_index: int, target: int) -> list[int]:
    """1 groupだけを決定的に埋める。stepとspanが互いに素でなくても停止する。

    巡回stepが全indexを回らない場合があるため、試行回数で必ず打ち切る。
    埋まりきらない場合はその時点のstateをそのまま使う（valid group state）。
    """
    group = [0] * span
    index = seed_index % span
    step = 1 + seed_index % 5
    total = 0
    for _ in range(span * 8):
        if total >= target:
            break
        if group[index] < 4:
            group[index] += 1
            total += 1
        index = (index + step) % span
    return group


class LookupTableArtifactTest(unittest.TestCase):
    """artifact format / key encoding / fail closedを固定する。"""

    def test_group_key_encoding_is_deterministic_and_positional(self) -> None:
        self.assertEqual(_shanten_frontier.group_key([0] * 9), 0)
        self.assertEqual(_shanten_frontier.group_key([1] + [0] * 8), 5**8)
        self.assertEqual(_shanten_frontier.group_key([0] * 8 + [1]), 1)
        self.assertEqual(
            _shanten_frontier.group_key([1, 2, 3]),
            1 * 25 + 2 * 5 + 3,
        )

    def test_entry_packing_round_trips(self) -> None:
        for blocks_used in range(5):
            for head_used in range(2):
                for meld_seeds in range(6):
                    for head_seeds in range(6):
                        for score in range(10):
                            packed = _shanten_table.pack_entry(
                                blocks_used, head_used, meld_seeds, head_seeds, score
                            )
                            state = packed >> _shanten_table._SCORE_SHIFT
                            self.assertEqual(packed & _shanten_table._SCORE_MASK, score)
                            self.assertEqual(
                                _shanten_table._decode_state(state),
                                (blocks_used, head_used, meld_seeds, head_seeds),
                            )

    def test_artifact_header_matches_its_declared_dimensions(self) -> None:
        payload = (
            pathlib.Path(_PACKAGE_ROOT) / _shanten_table.TABLE_RESOURCE
        ).read_bytes()

        self.assertTrue(payload.startswith(_shanten_table.MAGIC))
        # 宣言dimensionと実サイズの不一致はloaderがfail closedする。
        _shanten_table._ShantenTable(payload)

    def test_same_size_structural_corruption_fails_closed(self) -> None:
        # file sizeが宣言dimensionと一致していても、frontier idやspanが壊れて
        # いれば範囲外参照になり得る。どちらも`ShantenTableError`へ統一され、
        # `IndexError`やsilentに短いsliceにならないことを固定する。
        payload = (
            pathlib.Path(_PACKAGE_ROOT) / _shanten_table.TABLE_RESOURCE
        ).read_bytes()
        header_size = struct.calcsize(_shanten_table.HEADER_FORMAT)

        # spanをpool末尾より先へ伸ばす（サイズは不変）。load時に弾かれる。
        broken_span = bytearray(payload)
        span_offset = (
            header_size
            + (_shanten_table.SUIT_KEY_SPACE + _shanten_table.HONOR_KEY_SPACE) * 2
        )
        broken_span[span_offset : span_offset + 4] = (0xFFFFFF).to_bytes(4, "little")
        with self.assertRaisesRegex(
            _shanten_table.ShantenTableError, "past the end of its entry pool"
        ):
            _shanten_table._ShantenTable(bytes(broken_span))

        # frontier id（参照時に検査）の破損はnative backendの参照時検査が担う
        # （`test_native_shanten_backend.NativeArtifactIntegrityTest`）。

    def test_corrupted_artifacts_fail_closed(self) -> None:
        payload = bytearray(
            (pathlib.Path(_PACKAGE_ROOT) / _shanten_table.TABLE_RESOURCE).read_bytes()
        )

        with self.assertRaises(_shanten_table.ShantenTableError):
            _shanten_table._ShantenTable(b"")
        with self.assertRaises(_shanten_table.ShantenTableError):
            _shanten_table._ShantenTable(b"NOTALISJ" + bytes(payload[8:]))
        bad_version = bytearray(payload)
        bad_version[8:12] = (_shanten_table.FORMAT_VERSION + 1).to_bytes(4, "little")
        with self.assertRaises(_shanten_table.ShantenTableError):
            _shanten_table._ShantenTable(bytes(bad_version))
        with self.assertRaises(_shanten_table.ShantenTableError):
            _shanten_table._ShantenTable(bytes(payload[:-2]))


class ProductionBackendBoundaryTest(unittest.TestCase):
    """Issue #346: numeric入口はnative実装へ直接結び付き、Python計算moduleはない。"""

    def test_numeric_entries_are_bound_to_the_native_table(self) -> None:
        self.assertEqual(_shanten_backend.BACKEND_NAME, _shanten_backend.RUST_BACKEND)
        self.assertIs(
            shanten_module._shanten_from_valid_counts,
            _shanten_backend.native_shanten_from_valid_counts,
        )
        self.assertIs(
            shanten_module._evaluate_discards,
            _shanten_backend.native_evaluate_discards,
        )
        for name in ("_lookup_shanten", "_python_shanten"):
            with self.subTest(module=name):
                self.assertFalse((_PACKAGE_ROOT / f"{name}.py").exists())
                self.assertNotIn(f"lisjong.hand_evaluation.{name}", sys.modules)


class LocalFrontierValidationTest(unittest.TestCase):
    """local frontier generatorの性質を代表stateで固定する。

    全suit key (405,350) / honor key (43,130) のexhaustive validationは
    offline（`tools/`のgenerator実行時とPR記録）で行う。Python lookup / DFS
    oracleとの単一group比較はIssue #346で削除した（値は仕様fixtureと外部
    reference検証が固定する）。
    """

    def test_dominant_frontier_preserves_the_reachable_optimum(self) -> None:
        # exact-safe dominanceがoptimumを落としていないことを、縮約前後の
        # 合成結果で確認する。
        for seed_index in range(12):
            group = _single_group_counts(9, seed_index, 9)
            raw = _shanten_frontier.local_frontier(group, suited=True)
            reduced = _shanten_frontier.dominant_frontier(raw)
            with self.subTest(seed=seed_index):
                self.assertLessEqual(len(reduced), len(raw))
                for (bu, hu, ms, hs), score in raw.items():
                    self.assertTrue(
                        any(
                            rb == bu
                            and rh == hu
                            and rms >= ms
                            and rhs >= hs
                            and rscore >= score
                            for (rb, rh, rms, rhs), rscore in reduced.items()
                        ),
                        msg=f"dominance dropped {(bu, hu, ms, hs)} -> {score}",
                    )


_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "src" / "lisjong" / "hand_evaluation"


def _imported_modules(path: pathlib.Path) -> set[str]:
    """source fileがimportするmodule名（`from X import Y`はX.Yも含む）を返す。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imported


def _run_python(code: str, backend: str | None) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment.pop(_shanten_backend.BACKEND_ENVIRONMENT_VARIABLE, None)
    if backend is not None:
        environment[_shanten_backend.BACKEND_ENVIRONMENT_VARIABLE] = backend
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        cwd=_REPOSITORY_ROOT,
        timeout=120,
    )


_FAKE_NATIVE_RUST_PROGRAM = """
import sys
import types

fake = types.ModuleType("_lisjong_native")
fake.API_VERSION = 3


class StandardShantenTable:
    standard_shanten = shanten_from_valid_counts = None
    evaluate_discards = evaluate_progression = None

    def __init__(self, payload, combine, penalty, error):
        fake.arguments = (payload, combine, penalty, error)


fake.StandardShantenTable = StandardShantenTable
sys.modules["_lisjong_native"] = fake

from lisjong.hand_evaluation import _shanten_backend, _shanten_table

assert _shanten_backend.BACKEND_NAME == "rust"
assert "lisjong.hand_evaluation._shanten_backend" in sys.modules
payload, combine, penalty, error = fake.arguments
assert payload == _shanten_table.read_table_payload()
assert combine == _shanten_backend._little_endian_bytes(_shanten_table._COMBINE)
assert penalty == b"".join(table.tobytes() for table in _shanten_table._PENALTY)
assert error is _shanten_table.ShantenTableError
for name in ("_lookup_shanten", "_python_shanten"):
    assert "lisjong.hand_evaluation." + name not in sys.modules, name
print("ok")
"""


_GENERATOR_IMPORT_PROGRAM = """
import sys

sys.modules["_lisjong_native"] = None
from tools import generate_shanten_table, generate_structural_predicate_table

for name in ("shanten", "_shanten_backend"):
    assert "lisjong.hand_evaluation." + name not in sys.modules, name
print("ok")
"""


class ArtifactSupplySeparationTest(unittest.TestCase):
    """Issue #338 / #346: artifact供給・共有helperとnumeric計算の依存分離を固定する。

    native table構築、structural predicate、offline generatorが、Issue #346で
    削除したPython numeric計算本体（`_lookup_shanten`、`_python_shanten`）にも、
    native拡張にも依存しないことを、import graphと実行時のmodule読込で確認する。
    削除済みmodule名もforbiddenに残し、再導入を検出する。
    """

    _CALCULATION_MODULES = ("_lookup_shanten", "_python_shanten")

    def _assert_no_import(self, path: pathlib.Path, forbidden: tuple[str, ...]) -> None:
        imported = _imported_modules(path)
        for name in forbidden:
            with self.subTest(path=path.name, forbidden=name):
                self.assertFalse(
                    any(name in module.split(".") for module in imported),
                    sorted(imported),
                )

    def test_shared_helper_module_has_no_package_dependencies(self) -> None:
        imported = _imported_modules(_PACKAGE_ROOT / "_shanten_common.py")
        self.assertFalse(any(name.startswith("lisjong") for name in imported))

    def test_artifact_supply_does_not_import_calculation_modules(self) -> None:
        self._assert_no_import(
            _PACKAGE_ROOT / "_shanten_table.py",
            (
                *self._CALCULATION_MODULES,
                "_shanten_frontier",
                "_shanten_backend",
                "shanten",
                "_lisjong_native",
            ),
        )

    def test_frontier_and_generators_do_not_import_calculation_or_native(
        self,
    ) -> None:
        forbidden = (*self._CALCULATION_MODULES, "_shanten_backend", "_lisjong_native")
        self._assert_no_import(_PACKAGE_ROOT / "_shanten_frontier.py", forbidden)
        for name in (
            "generate_shanten_table.py",
            "generate_structural_predicate_table.py",
        ):
            self._assert_no_import(_REPOSITORY_ROOT / "tools" / name, forbidden)

    def test_semantic_module_does_not_import_the_old_search(self) -> None:
        # 七対子・国士無双と共有定数は`_shanten_common`から読む。structural
        # predicateの特殊形dispatchもこの経路を共有する。
        self._assert_no_import(_PACKAGE_ROOT / "shanten.py", ("_python_shanten",))
        self.assertIs(shanten_module._shanten_common, _shanten_common)

    def test_generators_import_without_native_extension(self) -> None:
        # 環境変数未設定（default rust）でもnative拡張なしでimportできる。
        result = _run_python(_GENERATOR_IMPORT_PROGRAM, backend=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_rust_selection_builds_native_table_without_python_calculation(
        self,
    ) -> None:
        # native拡張の代わりに引数を記録するfakeを入れ、rust選択processが
        # `_shanten_table`の供給だけでnative tableを作り、Python計算moduleを
        # importしないことを確認する（API_VERSION検査は実物と同じ経路を通る）。
        for backend in (None, "rust"):
            with self.subTest(backend=backend):
                result = _run_python(_FAKE_NATIVE_RUST_PROGRAM, backend=backend)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "ok")


class PublicSurfaceTest(unittest.TestCase):
    def test_package_exports_only_the_public_contract(self) -> None:
        import lisjong.hand_evaluation as hand_evaluation

        self.assertEqual(hand_evaluation.__all__, ["calculate_shanten"])

    def test_count_native_hot_path_is_not_a_top_level_public_api(self) -> None:
        # Issue #113のcount-native pathはpackage-internal contractであり、
        # package rootの一般public surfaceへは載せない。
        import lisjong.hand_evaluation as hand_evaluation

        self.assertNotIn(
            "calculate_shanten_from_canonical_counts", hand_evaluation.__all__
        )
        self.assertFalse(
            hasattr(hand_evaluation, "calculate_shanten_from_canonical_counts")
        )
        for predicate in (
            "is_structurally_complete_from_canonical_counts",
            "is_structurally_tenpai_from_canonical_counts",
        ):
            self.assertNotIn(predicate, hand_evaluation.__all__)
            self.assertFalse(hasattr(hand_evaluation, predicate))

    def test_hand_evaluation_does_not_depend_on_the_belief_layer(self) -> None:
        # canonical axisの正本はbelief側だが、hand_evaluationからbelief layerへ
        # 依存させない（count-native pathは34-countをそのまま受け取る）。
        for path in _PACKAGE_ROOT.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    with self.subTest(path=path.name, module=node.module):
                        self.assertFalse(node.module.startswith("lisjong.belief"))

    def test_backend_stays_behind_a_private_module_name(self) -> None:
        # Issue #115のnumeric backend、Issue #141のpredicate backend、
        # Issue #213のbackend選択module、Issue #263の役・点数計算binding
        # （`scoring.py`、計算coreはnative拡張）を含む。
        # 期待するprivate architectureをexact setとして固定する（public
        # surfaceは増やさない）。
        modules = {path.name for path in _PACKAGE_ROOT.glob("*.py")}
        self.assertEqual(
            modules,
            {
                "__init__.py",
                "shanten.py",
                "_shanten_frontier.py",
                "_shanten_backend.py",
                "_shanten_common.py",
                "_shanten_table.py",
                "_structural_predicates.py",
                "scoring.py",
            },
        )

    def test_lookup_table_artifact_ships_with_the_package(self) -> None:
        for resource in ("_shanten_table.bin", "_structural_predicate_table.bin"):
            with self.subTest(resource=resource):
                artifact = _PACKAGE_ROOT / resource
                self.assertTrue(artifact.is_file())
                self.assertGreater(artifact.stat().st_size, 0)

    def test_does_not_depend_on_external_environments(self) -> None:
        """RiichiEnv / RiichiLab / transportへ依存しないことを、importから確認する。"""
        forbidden = ("riichienv", "riichilab", "websockets", "asyncio")
        for path in _PACKAGE_ROOT.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            for name in imported:
                root = name.split(".")[0]
                with self.subTest(path=path.name, module=name):
                    self.assertNotIn(root, forbidden)


if __name__ == "__main__":
    unittest.main()
