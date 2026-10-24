"""Issue #213 / #346: native shanten backendの必須化・境界・fail closedを固定する。

Issue #346でPython計算実装を削除し、native拡張が必須になった。backend選択testは
subprocessでnative拡張を隠す・偽装するため、実物の有無に依存せず実行する。
numeric値の正本は`test_shanten_spec_fixtures.py`（仕様由来fixture）と
`reference_tests/`（外部実装）であり、本moduleはnative境界の契約を扱う。

このmodule自体が`shanten`をimportするため、native拡張がなければimport時に
失敗する（skipにならない）。`_require_native`は他moduleとの互換のために残す。
"""

import os
import pathlib
import random
import subprocess
import sys
import threading
import unittest

from lisjong.hand_evaluation import (
    _shanten_backend,
    _shanten_table,
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
    / _shanten_table.TABLE_RESOURCE
)


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


_ROUTING_PROGRAM = """
import sys
import _lisjong_native
import lisjong.hand_evaluation.shanten as shanten
from lisjong.hand_evaluation import _shanten_backend

assert _shanten_backend.BACKEND_NAME == "rust"
table = _shanten_backend._NATIVE_TABLE
assert type(table) is _lisjong_native.StandardShantenTable
assert shanten._shanten_from_valid_counts == table.shanten_from_valid_counts
assert shanten._evaluate_discards == table.evaluate_discards
assert _shanten_backend.calculate_standard_shanten == table.standard_shanten
assert _shanten_backend.native_evaluate_progression == table.evaluate_progression
for name in ("_lookup_shanten", "_python_shanten"):
    assert "lisjong.hand_evaluation." + name not in sys.modules, name
print("ok")
"""


_HIDDEN_NATIVE = "import sys\nsys.modules['_lisjong_native'] = None\n"


class BackendSelectionTest(unittest.TestCase):
    """Issue #346: 未設定 / rustはnative必須、pythonと未知値は拒否する。"""

    def _assert_fails(self, result, *fragments: str) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("ShantenBackendError", result.stderr)
        for fragment in fragments:
            self.assertIn(fragment, result.stderr)

    def test_default_and_explicit_rust_route_every_entry_to_native(self) -> None:
        for backend in (None, "rust"):
            with self.subTest(backend=backend):
                result = _run_python(_ROUTING_PROGRAM, backend=backend)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "ok")

    def test_retired_python_backend_is_rejected_with_migration_guidance(
        self,
    ) -> None:
        # 旧値を黙ってRustへ読み替えない。native拡張の有無によらず拒否する。
        for prefix in ("", _HIDDEN_NATIVE):
            with self.subTest(native_hidden=bool(prefix)):
                result = _run_python(
                    prefix + "import lisjong.hand_evaluation.shanten\n",
                    backend="python",
                )
                self._assert_fails(result, "no longer supported", "Issue #346")

    def test_unknown_backend_name_fails_closed(self) -> None:
        for name in ("Rust", "", "native"):
            with self.subTest(name=name):
                result = _run_python(
                    "import lisjong.hand_evaluation.shanten\n", backend=name
                )
                self._assert_fails(result, "must be unset or 'rust'")

    def test_missing_native_extension_fails_at_shanten_import(self) -> None:
        # 失敗位置は`shanten`（と`_shanten_backend`）のimport時。Pythonへ
        # fallbackする経路は存在しない。
        for backend in (None, "rust"):
            for statement in (
                "import lisjong.hand_evaluation.shanten",
                "from lisjong.hand_evaluation import calculate_shanten",
                "import lisjong.policies",
            ):
                with self.subTest(backend=backend, statement=statement):
                    result = _run_python(
                        _HIDDEN_NATIVE + statement + "\n", backend=backend
                    )
                    self._assert_fails(
                        result, "requires the native extension '_lisjong_native'"
                    )

    def test_packages_without_shanten_import_without_native_extension(self) -> None:
        # package root、scoring binding、artifact供給・共有helper・frontier・
        # structural predicateはnative拡張なしでimportできる。
        result = _run_python(
            _HIDDEN_NATIVE + "import lisjong, lisjong.hand_evaluation\n"
            "from lisjong.hand_evaluation import (\n"
            "    scoring, _shanten_common, _shanten_frontier, _shanten_table,\n"
            "    _structural_predicates,\n"
            ")\n"
            "assert 'lisjong.hand_evaluation.shanten' not in sys.modules\n"
            "assert 'lisjong.hand_evaluation._shanten_backend' not in sys.modules\n"
            "print('ok')\n",
            backend=None,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_corrupted_table_fails_closed_at_shanten_import(self) -> None:
        # artifact破損はimport時のnative table構築で`ShantenTableError`になる。
        result = _run_python(
            "from lisjong.hand_evaluation import _shanten_table\n"
            "payload = _shanten_table.read_table_payload()\n"
            "_shanten_table.read_table_payload = lambda: payload[:-2]\n"
            "import lisjong.hand_evaluation.shanten\n",
            backend=None,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("ShantenTableError", result.stderr)


class NativeStandardShantenBoundaryTest(unittest.TestCase):
    """native通常形入口の入力境界・決定性・並行呼出し・counter。

    Python lookup / DFS oracleとの差分test（Issue #213）はIssue #346で削除した。
    値は`test_shanten_spec_fixtures.py`と`reference_tests/`が固定する。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.table = (
            None
            if _lisjong_native is None
            else _shanten_backend.build_native_table(_lisjong_native)
        )

    def setUp(self) -> None:
        _require_native(self)

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


class NativeNumericCoreBoundaryTest(unittest.TestCase):
    """native numeric core（通常形 + 特殊形dispatch）の入力境界。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.table = (
            None
            if _lisjong_native is None
            else _shanten_backend.build_native_table(_lisjong_native)
        )

    def setUp(self) -> None:
        _require_native(self)

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
    """native側のartifact検証がfail closedする。"""

    def setUp(self) -> None:
        _require_native(self)
        self.payload = _TABLE_PATH.read_bytes()

    def _build(self, payload: bytes, combine=None, penalties=None):
        return _lisjong_native.StandardShantenTable(
            payload,
            (
                _shanten_backend._little_endian_bytes(_shanten_table._COMBINE)
                if combine is None
                else combine
            ),
            (
                b"".join(table.tobytes() for table in _shanten_table._PENALTY)
                if penalties is None
                else penalties
            ),
            _shanten_table.ShantenTableError,
        )

    def test_corrupted_artifacts_fail_closed(self) -> None:
        payload = bytearray(self.payload)
        bad_version = bytearray(payload)
        bad_version[8:12] = (_shanten_table.FORMAT_VERSION + 1).to_bytes(4, "little")
        for label, broken in (
            ("empty", b""),
            ("magic", b"NOTALISJ" + bytes(payload[8:])),
            ("version", bytes(bad_version)),
            ("truncated", bytes(payload[:-2])),
            ("extended", bytes(payload) + b"\x00\x00"),
        ):
            with self.subTest(label=label):
                with self.assertRaises(_shanten_table.ShantenTableError):
                    self._build(broken)

    def test_broken_frontier_span_fails_closed_at_load(self) -> None:
        header_size = 28
        span_offset = (
            header_size
            + (_shanten_table.SUIT_KEY_SPACE + _shanten_table.HONOR_KEY_SPACE) * 2
        )
        broken = bytearray(self.payload)
        broken[span_offset : span_offset + 4] = (0xFFFFFFF0).to_bytes(4, "little")
        with self.assertRaisesRegex(
            _shanten_table.ShantenTableError, "past the end of its entry pool"
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
            _shanten_table.ShantenTableError, "frontier that does not exist"
        ):
            table.standard_shanten(counts, 0)

    def test_inconsistent_helper_tables_fail_closed(self) -> None:
        with self.assertRaises(_shanten_table.ShantenTableError):
            self._build(self.payload, combine=b"\x00\x00")
        with self.assertRaises(_shanten_table.ShantenTableError):
            self._build(self.payload, combine=b"\x00\x7f" * (360 * 360))
        with self.assertRaises(_shanten_table.ShantenTableError):
            self._build(self.payload, penalties=b"\x00")

    def test_out_of_range_pool_states_fail_closed_without_aborting(self) -> None:
        # pool entryの上位12 bitはresource state（0..359）としてscore配列と
        # combine行の添字に使われる。4095のような配列外の値と、360のように
        # combine table内だが別の行を読む値の両方を、table構築時に
        # ShantenTableErrorで拒否しなければならない。release buildは
        # panic=abortなので、範囲外アクセスはprocess全体を落とし得る。
        # そのためsubprocessで実行し、正常終了して期待した例外になることを
        # 確認する。
        cases = (
            ("suit", 0, 0xFFFF),
            ("suit", 0, 360 << 4),
            ("suit", -1, (360 << 4) | 0xF),
            ("honor", 0, 0xFFF0),
            ("honor", -1, 361 << 4),
        )
        for group, position, packed in cases:
            with self.subTest(group=group, position=position, packed=packed):
                result = _run_python(
                    _CORRUPT_POOL_PROGRAM.format(
                        group=group, position=position, packed=packed
                    ),
                    backend=None,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    result.stdout.strip(),
                    "ShantenTableError: unknown resource state",
                    result.stderr,
                )


_CORRUPT_POOL_PROGRAM = """
import struct
import _lisjong_native
from lisjong.hand_evaluation import _shanten_backend, _shanten_table

payload = bytearray(_shanten_table.read_table_payload())
header = struct.unpack_from(_shanten_table.HEADER_FORMAT, payload)
_magic, _version, suit_frontiers, honor_frontiers, suit_pool, honor_pool = header
suit_pool_offset = (
    struct.calcsize(_shanten_table.HEADER_FORMAT)
    + (_shanten_table.SUIT_KEY_SPACE + _shanten_table.HONOR_KEY_SPACE) * 2
    + (suit_frontiers + honor_frontiers) * 5
)
offset, entries = (
    (suit_pool_offset, suit_pool)
    if "{group}" == "suit"
    else (suit_pool_offset + suit_pool * 2, honor_pool)
)
index = {position} % entries
payload[offset + index * 2 : offset + index * 2 + 2] = ({packed}).to_bytes(2, "little")
try:
    table = _lisjong_native.StandardShantenTable(
        bytes(payload),
        _shanten_backend._little_endian_bytes(_shanten_table._COMBINE),
        b"".join(table.tobytes() for table in _shanten_table._PENALTY),
        _shanten_table.ShantenTableError,
    )
except _shanten_table.ShantenTableError as error:
    assert "unknown resource state" in str(error), str(error)
    print("ShantenTableError: unknown resource state")
else:
    print("accepted")
"""


class NativeBuildIdentityTest(unittest.TestCase):
    """Issue #216: wheelが自身のbuild元lisjong revisionを報告する。"""

    def setUp(self) -> None:
        _require_native(self)

    def test_source_revision_is_a_full_commit_or_unknown(self) -> None:
        revision = _lisjong_native.SOURCE_REVISION
        self.assertIsInstance(revision, str)
        if revision != "unknown":
            self.assertRegex(revision, r"^[0-9a-f]{40}$")


class NativeBackendSelectedProcessTest(unittest.TestCase):
    """環境変数未設定の別processでも公開APIがnativeを通り、同じ値を返す。"""

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
            backend=None,
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
