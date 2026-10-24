"""numeric shanten計算をnative拡張へ結び付けるprivate module（Issue #213 / #346）。

`shanten`のnumeric core、`calculate_restricted_standard_shanten()`の通常形、
打牌候補の一括評価、R5（expected terminal shanten progression）の探索本体は、
すべてnative拡張`_lisjong_native`（`native/`のRust crate）で実行する。Issue #346で
Python計算実装（`_lookup_shanten`、`_python_shanten`、Python版の一括打牌評価、
R5のPython evaluator）を削除したため、Pythonへのfallbackや切替はない。

```text
LISJONG_SHANTEN_BACKEND   動作
未設定 / "rust"           native拡張を要求する
"python"                  ShantenBackendError（Python backendは廃止済み）
それ以外                  ShantenBackendError
```

本moduleはimport時にnative拡張のimport、`API_VERSION`の確認、native tableの
構築まで行う。native拡張の不在・API不一致・artifact破損は最初のdecisionではなく
import時に`ShantenBackendError` / `_shanten_table.ShantenTableError`でfail closed
する（hot pathにlazy-load用のwrapper frameを挟まない）。失敗するのは本moduleを
importする`lisjong.hand_evaluation.shanten`とその利用者（Policy等）であり、
`lisjong`、`lisjong.hand_evaluation`、`scoring`、`_shanten_table`、
`_shanten_common`、`_shanten_frontier`、`_structural_predicates`とoffline
generatorはnative拡張なしでimportできる（package rootの`calculate_shanten`は
遅延公開）。

native tableは`_shanten_table`（Issue #338で分離した供給module）から
`_shanten_table.bin`、resource state combine table、penalty tableをbytesで受け取る。
artifact integrity failureは`_shanten_table.ShantenTableError`になる。

`API_VERSION`はnative入口の組を表す。`REQUIRED_NATIVE_API_VERSION`と一致しない
wheel（#224以前の`API_VERSION`属性なし = 1を含む）は拒否する。lisjong単体の
この検査は`SOURCE_REVISION`の一致を保証しないため、同じlisjong revisionから
buildしたwheelを使う（`docs/rust-backend-distribution.md`）。

native入口は入力preconditionを実行時にも検査する（34要素、各count 0..4、
`fixed_meld_count` 0..4）。環境変数で選ぶため、Windows `spawn`を含むworker
processにも親processの指定がそのまま継承される。
"""

import os
import sys

from lisjong.hand_evaluation import _shanten_table

BACKEND_ENVIRONMENT_VARIABLE = "LISJONG_SHANTEN_BACKEND"
RUST_BACKEND = "rust"
RETIRED_PYTHON_BACKEND = "python"
"""Issue #346で廃止した旧backend名。黙ってRustへ読み替えず、移行案内付きで拒否する。"""

REQUIRED_NATIVE_API_VERSION = 3
"""要求する`_lisjong_native.API_VERSION`（Issue #224 / #232）。"""


class ShantenBackendError(Exception):
    """native shanten backendを利用できない場合のfail closed例外。"""


def _little_endian_bytes(values) -> bytes:
    if sys.byteorder == "big" and values.itemsize > 1:
        values = type(values)(values.typecode, values)
        values.byteswap()
    return values.tobytes()


def build_native_table(native_module):
    """`_shanten_table`のartifactとcombine / penalty表からnative tableを作る。"""
    return native_module.StandardShantenTable(
        _shanten_table.read_table_payload(),
        _little_endian_bytes(_shanten_table._COMBINE),
        b"".join(table.tobytes() for table in _shanten_table._PENALTY),
        _shanten_table.ShantenTableError,
    )


def _import_native_module():
    try:
        import _lisjong_native
    except ImportError as error:
        raise ShantenBackendError(
            "lisjong requires the native extension '_lisjong_native' built from "
            "the same lisjong revision (see docs/rust-backend-distribution.md); "
            "the Python shanten backend was removed in Issue #346"
        ) from error
    # API_VERSIONを持たない#224以前のwheelは1として扱う。
    api_version = getattr(_lisjong_native, "API_VERSION", 1)
    if api_version != REQUIRED_NATIVE_API_VERSION:
        raise ShantenBackendError(
            f"lisjong requires _lisjong_native API_VERSION "
            f"{REQUIRED_NATIVE_API_VERSION}, got {api_version!r}; rebuild or "
            "reinstall the native extension from the same lisjong revision"
        )
    return _lisjong_native


def _select_backend_name() -> str:
    name = os.environ.get(BACKEND_ENVIRONMENT_VARIABLE, RUST_BACKEND)
    if name == RETIRED_PYTHON_BACKEND:
        raise ShantenBackendError(
            f"{BACKEND_ENVIRONMENT_VARIABLE}={RETIRED_PYTHON_BACKEND!r} is no "
            "longer supported: the Python shanten backend was removed in Issue "
            f"#346. Unset {BACKEND_ENVIRONMENT_VARIABLE} (or set it to "
            f"{RUST_BACKEND!r}) and install the native extension from the same "
            "lisjong revision; to reproduce a Python-backend run, use the "
            "recorded lisjong revision and environment"
        )
    if name != RUST_BACKEND:
        raise ShantenBackendError(
            f"{BACKEND_ENVIRONMENT_VARIABLE} must be unset or {RUST_BACKEND!r}, "
            f"got {name!r}"
        )
    return name


BACKEND_NAME = _select_backend_name()
"""このprocessのbackend名。現在は常に`"rust"`（記録・診断用に残す）。"""

_NATIVE_TABLE = build_native_table(_import_native_module())
calculate_standard_shanten = _NATIVE_TABLE.standard_shanten
"""通常形向聴数（`calculate_restricted_standard_shanten()`が使う）。"""

native_shanten_from_valid_counts = _NATIVE_TABLE.shanten_from_valid_counts
"""numeric shanten core全体（通常形と閉じた13 / 14枚の七対子・国士無双の最小値）。"""

native_evaluate_discards = _NATIVE_TABLE.evaluate_discards
"""`shanten.evaluate_discards_from_canonical_counts()`の実装（Issue #224）。"""

native_evaluate_progression = _NATIVE_TABLE.evaluate_progression
"""R5 progression探索（Issue #232）。policy側の`_new_progression_evaluator()`だけが使う。"""
