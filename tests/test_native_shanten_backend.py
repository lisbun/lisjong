"""Issue #213: opt-in native shanten backendの選択・同値性・fail closedを固定する。

backend選択testはnative拡張の有無に依存せず常に実行する。nativeの同値性testは
`_lisjong_native`がimportできる環境だけで実行し、`LISJONG_REQUIRE_NATIVE=1`
（native CI job）ではskipせずfailさせる。
"""

import os
import pathlib
import random
import subprocess
import sys
import threading
import unittest

import lisjong.hand_evaluation.shanten as shanten_module
from lisjong.hand_evaluation import (
    _lookup_shanten,
    _python_shanten,
    _shanten_backend,
    calculate_shanten,
)
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

try:
    import _lisjong_native
except ImportError:
    _lisjong_native = None

_REQUIRE_NATIVE = os.environ.get("LISJONG_REQUIRE_NATIVE") == "1"
_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
_TABLE_PATH = (
    _REPOSITORY_ROOT
    / "src"
    / "lisjong"
    / "hand_evaluation"
    / _lookup_shanten.TABLE_RESOURCE
)
_HAND_SIZES = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)


def _fixed_meld_count(concealed_tile_count: int) -> int:
    return 4 - (concealed_tile_count - 1) // 3


def _random_counts(generator: random.Random, size: int) -> list[int]:
    counts = [0] * 34
    total = 0
    while total < size:
        index = generator.randrange(34)
        if counts[index] < 4:
            counts[index] += 1
            total += 1
    return counts


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


def _require_native(test: unittest.TestCase) -> None:
    if _lisjong_native is not None:
        return
    if _REQUIRE_NATIVE:
        test.fail("LISJONG_REQUIRE_NATIVE=1 but _lisjong_native is not importable")
    test.skipTest("opt-in native extension _lisjong_native is not installed")


class BackendSelectionTest(unittest.TestCase):
    """native拡張がなくても成立するbackend選択contract。"""

    def test_default_backend_is_the_python_lookup_backend(self) -> None:
        result = _run_python(
            "from lisjong.hand_evaluation import _lookup_shanten, _shanten_backend\n"
            "assert _shanten_backend.BACKEND_NAME == 'python'\n"
            "assert (_shanten_backend.calculate_standard_shanten\n"
            "        is _lookup_shanten.calculate_standard_shanten)\n"
            "import sys\n"
            "assert '_lisjong_native' not in sys.modules\n",
            backend=None,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_explicit_python_backend_does_not_import_native(self) -> None:
        result = _run_python(
            "import sys\n"
            "from lisjong.hand_evaluation import calculate_shanten, _shanten_backend\n"
            "assert _shanten_backend.BACKEND_NAME == 'python'\n"
            "assert '_lisjong_native' not in sys.modules\n",
            backend="python",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_backend_name_fails_closed(self) -> None:
        result = _run_python("import lisjong.hand_evaluation\n", backend="Rust")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ShantenBackendError", result.stderr)

    def test_rust_backend_without_native_extension_does_not_fall_back(self) -> None:
        # sys.modulesのNone entryはImportErrorを起こす。native拡張の有無に
        # よらず「rust指定で利用不能ならimport時にfail closed」を固定する。
        result = _run_python(
            "import sys\n"
            "sys.modules['_lisjong_native'] = None\n"
            "import lisjong.hand_evaluation\n",
            backend="rust",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ShantenBackendError", result.stderr)
        self.assertIn("refusing to fall back", result.stderr)

    def test_core_package_does_not_require_native_extension(self) -> None:
        result = _run_python(
            "import sys\n"
            "sys.modules['_lisjong_native'] = None\n"
            "import lisjong, lisjong.policies\n"
            "from lisjong.hand_evaluation import calculate_shanten\n",
            backend=None,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class NativeStandardShantenEquivalenceTest(unittest.TestCase):
    """native frontier combineがPython lookupと独立DFS oracleに一致する。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.table = (
            None
            if _lisjong_native is None
            else _shanten_backend.build_native_table(_lisjong_native)
        )

    def setUp(self) -> None:
        _require_native(self)

    def _assert_all_backends_match(self, counts, fixed_meld_count: int) -> None:
        native = self.table.standard_shanten(counts, fixed_meld_count)
        self.assertEqual(
            native,
            _lookup_shanten.calculate_standard_shanten(counts, fixed_meld_count),
            msg=f"counts={counts} fixed_meld_count={fixed_meld_count}",
        )
        # 独立oracle（Issue #115以前の再帰探索）とも比較し、Python lookupと
        # nativeが同じ誤りを共有するだけの検証にしない。
        self.assertEqual(
            native,
            _python_shanten.calculate_standard_shanten(counts, fixed_meld_count),
            msg=f"counts={counts} fixed_meld_count={fixed_meld_count}",
        )

    def test_seeded_valid_hands_match_lookup_and_oracle(self) -> None:
        generator = random.Random(213_2026)
        for size in _HAND_SIZES:
            for _ in range(60):
                counts = _random_counts(generator, size)
                with self.subTest(size=size, counts=counts):
                    self._assert_all_backends_match(counts, _fixed_meld_count(size))

    def test_edge_shapes_match_lookup_and_oracle(self) -> None:
        shapes = {
            "empty-heavy single": [0] * 33 + [1],
            "four copies": [4, 4, 4] + [0] * 30 + [2],
            "four copies with honors": [0] * 27 + [4, 4, 4, 1, 1, 0, 0],
            "all terminals": [1, 0, 0, 0, 0, 0, 0, 0, 1] * 3 + [1] * 7,
            "pure suit": [3, 1, 1, 1, 1, 1, 1, 1, 3] + [0] * 25,
            "pairs": [2, 2, 2, 2, 2, 2, 2] + [0] * 27,
        }
        for label, counts in shapes.items():
            size = sum(counts)
            with self.subTest(label=label):
                self._assert_all_backends_match(counts, _fixed_meld_count(size))

    def test_every_fixed_meld_count_matches_the_python_lookup(self) -> None:
        # precondition上の枚数とfixed_meld_countが食い違う組でも、count 0..4 /
        # fixed 0..4の範囲ならbackend間で同じ値を返す。
        generator = random.Random(2130)
        for _ in range(300):
            counts = _random_counts(generator, generator.randrange(0, 15))
            for fixed_meld_count in range(5):
                try:
                    expected = _lookup_shanten.calculate_standard_shanten(
                        counts, fixed_meld_count
                    )
                except _lookup_shanten.ShantenTableError as error:
                    with self.assertRaises(_lookup_shanten.ShantenTableError) as raised:
                        self.table.standard_shanten(counts, fixed_meld_count)
                    self.assertEqual(str(raised.exception), str(error))
                    continue
                with self.subTest(counts=counts, fixed=fixed_meld_count):
                    self.assertEqual(
                        self.table.standard_shanten(counts, fixed_meld_count),
                        expected,
                    )

    def test_input_order_and_container_type_do_not_change_results(self) -> None:
        counts = [1, 1, 1, 0, 2, 0, 1, 1, 1, 0, 3, 0, 0, 0, 0, 0, 1, 1] + [0] * 16
        expected = self.table.standard_shanten(tuple(counts), 1)
        self.assertEqual(self.table.standard_shanten(list(counts), 1), expected)
        self.assertEqual(
            self.table.standard_shanten(_ArraySequence(counts), 1), expected
        )
        # 同じ入力の繰り返し（warm）でも結果は決定的。
        for _ in range(3):
            self.assertEqual(self.table.standard_shanten(tuple(counts), 1), expected)

    def test_invalid_inputs_are_rejected_at_the_python_boundary(self) -> None:
        valid = [0] * 34
        valid[0] = 1
        with self.assertRaises(ValueError):
            self.table.standard_shanten(valid[:-1], 4)
        with self.assertRaises(ValueError):
            self.table.standard_shanten(valid + [0], 4)
        for bad_value in (5, -1, 2**70):
            bad = list(valid)
            bad[3] = bad_value
            with self.subTest(value=bad_value):
                with self.assertRaises((ValueError, OverflowError)):
                    self.table.standard_shanten(bad, 4)
        for bad_fixed in (-1, 5):
            with self.subTest(fixed=bad_fixed):
                with self.assertRaises(ValueError):
                    self.table.standard_shanten(valid, bad_fixed)
        with self.assertRaises(TypeError):
            self.table.standard_shanten(["1"] * 34, 0)
        with self.assertRaises(TypeError):
            self.table.standard_shanten(12, 0)

    def test_concurrent_calls_are_consistent(self) -> None:
        generator = random.Random(21301)
        inputs = [
            (counts, _fixed_meld_count(sum(counts)))
            for counts in (_random_counts(generator, 14) for _ in range(200))
        ]
        expected = [self.table.standard_shanten(c, f) for c, f in inputs]
        failures: list[str] = []

        def worker() -> None:
            for (counts, fixed), value in zip(inputs, expected, strict=True):
                if self.table.standard_shanten(counts, fixed) != value:
                    failures.append(repr(counts))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(failures, [])

    def test_call_counter_proves_the_native_path_ran(self) -> None:
        before = _lisjong_native.standard_shanten_call_count()
        self.table.standard_shanten([0] * 33 + [1], 4)
        self.assertEqual(_lisjong_native.standard_shanten_call_count(), before + 1)


def _oracle_numeric_shanten(counts) -> int:
    """独立oracle: 再帰探索の通常形と、閉じた手の七対子・国士の最小値。"""
    size = sum(counts)
    value = _python_shanten.calculate_standard_shanten(counts, _fixed_meld_count(size))
    if size in (13, 14):
        value = min(
            value,
            _python_shanten.calculate_seven_pairs_shanten(counts),
            _python_shanten.calculate_thirteen_orphans_shanten(counts),
        )
    return value


class NativeNumericCoreEquivalenceTest(unittest.TestCase):
    """native numeric core（通常形 + 特殊形dispatch）がPython coreに一致する。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.table = (
            None
            if _lisjong_native is None
            else _shanten_backend.build_native_table(_lisjong_native)
        )

    def setUp(self) -> None:
        _require_native(self)

    def _assert_core_matches(self, counts) -> None:
        size = sum(counts)
        native = self.table.shanten_from_valid_counts(counts, size)
        message = f"counts={counts}"
        self.assertEqual(
            native,
            shanten_module._python_shanten_from_valid_counts(counts, size),
            msg=message,
        )
        self.assertEqual(native, _oracle_numeric_shanten(counts), msg=message)

    def test_seeded_valid_hands_match_python_core_and_oracle(self) -> None:
        generator = random.Random(213_0013)
        for size in _HAND_SIZES:
            for _ in range(80):
                counts = _random_counts(generator, size)
                with self.subTest(size=size, counts=counts):
                    self._assert_core_matches(counts)

    def test_special_hand_shapes_match_python_core_and_oracle(self) -> None:
        # 七対子・国士が通常形より小さくなる形、4枚持ちで七対子を数えない形、
        # 么九牌が偏った形を含める。
        generator = random.Random(213_0714)
        terminals = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
        for size in (13, 14):
            for _ in range(150):
                counts = [0] * 34
                while sum(counts) < size:
                    pool = terminals if generator.random() < 0.7 else range(34)
                    index = generator.choice(tuple(pool))
                    if counts[index] < 4:
                        counts[index] += 1
                with self.subTest(size=size, counts=counts):
                    self._assert_core_matches(counts)
            for _ in range(150):
                counts = [0] * 34
                while sum(counts) < size:
                    index = generator.randrange(34)
                    add = min(2, 4 - counts[index], size - sum(counts))
                    counts[index] += add
                with self.subTest(size=size, pair_heavy=counts):
                    self._assert_core_matches(counts)

    def test_invalid_core_inputs_are_rejected(self) -> None:
        counts = [0] * 34
        counts[0] = 3
        for concealed_tile_count in (2, 4, 3):
            with self.subTest(concealed=concealed_tile_count):
                with self.assertRaises(ValueError):
                    self.table.shanten_from_valid_counts(counts, concealed_tile_count)
        counts[1] = 3
        counts[2] = 3
        counts[3] = 3
        counts[4] = 3
        # sum == 15は有効な純手牌枚数ではない。
        with self.assertRaises(ValueError):
            self.table.shanten_from_valid_counts(counts, 15)


class _ArraySequence:
    """tuple / list以外のsequence入力経路を通すための最小sequence。"""

    def __init__(self, values) -> None:
        self._values = list(values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]


class NativeArtifactIntegrityTest(unittest.TestCase):
    """native側のartifact検証がPython backendと同じくfail closedする。"""

    def setUp(self) -> None:
        _require_native(self)
        self.payload = _TABLE_PATH.read_bytes()

    def _build(self, payload: bytes, combine=None, penalties=None):
        return _lisjong_native.StandardShantenTable(
            payload,
            (
                _shanten_backend._little_endian_bytes(_lookup_shanten._COMBINE)
                if combine is None
                else combine
            ),
            (
                b"".join(table.tobytes() for table in _lookup_shanten._PENALTY)
                if penalties is None
                else penalties
            ),
            _lookup_shanten.ShantenTableError,
        )

    def test_corrupted_artifacts_fail_closed(self) -> None:
        payload = bytearray(self.payload)
        bad_version = bytearray(payload)
        bad_version[8:12] = (_lookup_shanten.FORMAT_VERSION + 1).to_bytes(4, "little")
        for label, broken in (
            ("empty", b""),
            ("magic", b"NOTALISJ" + bytes(payload[8:])),
            ("version", bytes(bad_version)),
            ("truncated", bytes(payload[:-2])),
            ("extended", bytes(payload) + b"\x00\x00"),
        ):
            with self.subTest(label=label):
                with self.assertRaises(_lookup_shanten.ShantenTableError):
                    self._build(broken)

    def test_broken_frontier_span_fails_closed_at_load(self) -> None:
        header_size = 28
        span_offset = (
            header_size
            + (_lookup_shanten.SUIT_KEY_SPACE + _lookup_shanten.HONOR_KEY_SPACE) * 2
        )
        broken = bytearray(self.payload)
        broken[span_offset : span_offset + 4] = (0xFFFFFFF0).to_bytes(4, "little")
        with self.assertRaisesRegex(
            _lookup_shanten.ShantenTableError, "past the end of its entry pool"
        ):
            self._build(bytes(broken))

    def test_broken_frontier_id_fails_closed_on_reference(self) -> None:
        broken = bytearray(self.payload)
        broken[28:30] = (0xFFFF).to_bytes(2, "little")
        table = self._build(bytes(broken))
        counts = [0] * 34
        for kind in (9, 10, 11, 18, 19, 20, 27, 27, 28, 28, 29, 29, 30, 30):
            counts[kind] += 1
        with self.assertRaisesRegex(
            _lookup_shanten.ShantenTableError, "frontier that does not exist"
        ):
            table.standard_shanten(counts, 0)

    def test_inconsistent_helper_tables_fail_closed(self) -> None:
        with self.assertRaises(_lookup_shanten.ShantenTableError):
            self._build(self.payload, combine=b"\x00\x00")
        with self.assertRaises(_lookup_shanten.ShantenTableError):
            self._build(self.payload, combine=b"\x00\x7f" * (360 * 360))
        with self.assertRaises(_lookup_shanten.ShantenTableError):
            self._build(self.payload, penalties=b"\x00")


class NativeBackendSelectedProcessTest(unittest.TestCase):
    """`LISJONG_SHANTEN_BACKEND=rust`のprocessで公開APIの結果が変わらない。"""

    def setUp(self) -> None:
        _require_native(self)

    def test_public_shanten_matches_under_the_rust_backend(self) -> None:
        notations = (
            "123456789m123p11s",
            "123456789m123p1s",
            "1122334455667z",
            "19m19p19s1234567z",
            "19m19p19s123456z1m",
            "1111m2222p3333s44z",
            "147m258p369s1357z",
            "0m55p0s123456789m",
            "1111m",
            "12m",
            "1z",
        )
        expected = [calculate_shanten(_hand(notation)) for notation in notations]
        result = _run_python(
            "import sys\n"
            "import _lisjong_native\n"
            "from lisjong.hand_evaluation import calculate_shanten, _shanten_backend\n"
            "from tests.test_native_shanten_backend import _hand\n"
            "assert _shanten_backend.BACKEND_NAME == 'rust'\n"
            f"notations = {notations!r}\n"
            "values = [calculate_shanten(_hand(n)) for n in notations]\n"
            "assert _lisjong_native.standard_shanten_call_count() == len(values)\n"
            "print(values)\n",
            backend="rust",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), repr(expected))


_CATEGORY_BY_SUFFIX = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}


def _hand(notation: str) -> list[Tile]:
    """`"123m0p"`形式（0は赤5）をTile listへ展開するtest用記法。"""
    tiles: list[Tile] = []
    ranks: list[str] = []
    for character in notation:
        if character.isdigit():
            ranks.append(character)
            continue
        category = _CATEGORY_BY_SUFFIX[character]
        for rank in ranks:
            red = rank == "0"
            tiles.append(Tile(TileType(category, 5 if red else int(rank)), is_red=red))
        ranks = []
    return tiles


if __name__ == "__main__":
    unittest.main()
