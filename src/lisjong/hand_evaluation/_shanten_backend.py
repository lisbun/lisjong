"""通常形shanten計算backendの明示選択（Issue #213）。

`shanten._shanten_from_valid_counts()`と
`shanten.calculate_restricted_standard_shanten()`が使う
`calculate_standard_shanten(counts, fixed_meld_count)`と、numeric shanten
core（`shanten._shanten_from_valid_counts()`）の実装を、process起動時に1回だけ
選ぶprivate moduleである。shanten semantic、Policy、公開APIは変えない。

```text
LISJONG_SHANTEN_BACKEND   backend
未設定 / "python"         _lookup_shanten（default、native拡張不要）
"rust"                    opt-in native拡張 `_lisjong_native`
それ以外                  ShantenBackendError
```

`rust`を明示した場合、native拡張をimportできなければimport時に
`ShantenBackendError`でfail closedし、Python backendへ黙って切り替えない。
native backendは`_lookup_shanten`と同じ`_shanten_table.bin`、同じresource
state combine table、同じpenalty tableをbytesで受け取り、frontier combineの
loopと、閉じた13 / 14枚の七対子・国士無双との最小値dispatchをnativeで実行する
（Issue #213の計測でcore全体がchampion Policyの主要costだったため）。
artifact integrity failureは同じ`ShantenTableError`になる。

dispatchと特殊形の定義はPython core（`shanten._shanten_from_valid_counts()`、
`_python_shanten`）が正本であり、native側はその写しである。差分は
`tests/test_native_shanten_backend.py`のdifferential testsと、rust選択下の
full test suite（CI `native-backend` job）で検出する。

選択は環境変数で行うため、Windows `spawn`を含むworker processにも親processの
指定がそのまま継承される。runtimeに切り替えるAPIや、Python / Rustを混在させる
設定は持たない。

native backendは入力preconditionを実行時にも検査する（34要素、各count 0..4、
`fixed_meld_count` 0..4）。Python backendはtrusted callerを前提に検査しない。
precondition違反の入力に対する例外型だけはbackendで異なり得るが、
precondition内の入力に対する結果は一致する（differential testsで固定）。

Issue #224で、打牌候補の構造評価を一括で行う
`shanten.evaluate_discards_from_canonical_counts()`のnative実装
（`StandardShantenTable.evaluate_discards`）を追加した。native拡張は入口の組を
`API_VERSION`で識別し、rust選択時はimport時に`REQUIRED_NATIVE_API_VERSION`と
一致することを確認する。#224以前のwheel（`API_VERSION`属性なし = 1）や将来の
不一致なwheelは`ShantenBackendError`でfail closedし、一部の入口だけPythonへ
切り替えることはしない。python選択時はnative拡張をimportしないため、旧wheelが
installされていても影響しない。

Issue #232でAPI_VERSION 3として、R5（expected terminal shanten progression）の
探索本体`StandardShantenTable.evaluate_progression`を追加した。rust選択時は
R5もnativeで実行し、R5だけPythonへ黙って戻ることはない（環境変数名は向聴専用に
見えるが、適用範囲はnumeric shanten core・打牌一括評価・R5探索である）。
"""

import os
import sys

from lisjong.hand_evaluation import _lookup_shanten

BACKEND_ENVIRONMENT_VARIABLE = "LISJONG_SHANTEN_BACKEND"
PYTHON_BACKEND = "python"
RUST_BACKEND = "rust"
REQUIRED_NATIVE_API_VERSION = 3
"""rust選択時に要求する`_lisjong_native.API_VERSION`（Issue #224 / #232）。"""


class ShantenBackendError(Exception):
    """明示されたshanten backendを利用できない場合のfail closed例外。"""


def _little_endian_bytes(values) -> bytes:
    if sys.byteorder == "big" and values.itemsize > 1:
        values = type(values)(values.typecode, values)
        values.byteswap()
    return values.tobytes()


def build_native_table(native_module):
    """`_lookup_shanten`のartifactとtableからnative tableを作る。"""
    return native_module.StandardShantenTable(
        _lookup_shanten.read_table_payload(),
        _little_endian_bytes(_lookup_shanten._COMBINE),
        b"".join(table.tobytes() for table in _lookup_shanten._PENALTY),
        _lookup_shanten.ShantenTableError,
    )


def _import_native_module():
    try:
        import _lisjong_native
    except ImportError as error:
        raise ShantenBackendError(
            f"{BACKEND_ENVIRONMENT_VARIABLE}={RUST_BACKEND!r} requires the opt-in "
            "native extension '_lisjong_native' (install it with "
            "'python -m pip install ./native'); refusing to fall back to the "
            "Python backend"
        ) from error
    # API_VERSIONを持たない#224以前のwheelは1として扱う。
    api_version = getattr(_lisjong_native, "API_VERSION", 1)
    if api_version != REQUIRED_NATIVE_API_VERSION:
        raise ShantenBackendError(
            f"{BACKEND_ENVIRONMENT_VARIABLE}={RUST_BACKEND!r} requires "
            f"_lisjong_native API_VERSION {REQUIRED_NATIVE_API_VERSION}, got "
            f"{api_version!r}; rebuild or reinstall the native extension from "
            "the same lisjong revision"
        )
    return _lisjong_native


def _select_backend_name() -> str:
    name = os.environ.get(BACKEND_ENVIRONMENT_VARIABLE, PYTHON_BACKEND)
    if name not in (PYTHON_BACKEND, RUST_BACKEND):
        raise ShantenBackendError(
            f"{BACKEND_ENVIRONMENT_VARIABLE} must be {PYTHON_BACKEND!r} or "
            f"{RUST_BACKEND!r}, got {name!r}"
        )
    return name


BACKEND_NAME = _select_backend_name()
"""このprocessで選択されたbackend名。"""

native_shanten_from_valid_counts = None
"""rust選択時だけ設定される、numeric shanten core全体のnative実装。

`shanten._shanten_from_valid_counts()`と同じ入力（trusted canonical counts,
`concealed_tile_count == sum(counts)`）を受け取り、通常形と閉じた13 / 14枚の
七対子・国士無双の最小値を返す。python選択時はNoneで、既存のPython coreを
そのまま使う。
"""

native_evaluate_discards = None
"""rust選択時だけ設定される、`shanten.evaluate_discards_from_canonical_counts()`の
native実装（Issue #224）。python選択時はNoneで、Python実装をそのまま使う。
"""

native_evaluate_progression = None
"""rust選択時だけ設定される、R5 progression探索のnative実装（Issue #232）。

policy側の`_new_progression_evaluator()`だけが使う。python選択時はNoneで、
Python oracle（`_TerminalShantenProgressionEvaluator`）を使う。
"""

if BACKEND_NAME == RUST_BACKEND:
    # rustを明示した場合はimport時にnative API versionの確認とnative table構築まで
    # 行う。artifactやnative拡張の問題を最初のdecisionではなく起動時にfail closedで
    # 検出し、hot pathにlazy-load用のPython wrapper frameを挟まない。
    _NATIVE_TABLE = build_native_table(_import_native_module())
    calculate_standard_shanten = _NATIVE_TABLE.standard_shanten
    native_shanten_from_valid_counts = _NATIVE_TABLE.shanten_from_valid_counts
    native_evaluate_discards = _NATIVE_TABLE.evaluate_discards
    native_evaluate_progression = _NATIVE_TABLE.evaluate_progression
else:
    calculate_standard_shanten = _lookup_shanten.calculate_standard_shanten
