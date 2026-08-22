"""非公開手牌belief用のfixed-point数値表現。

Issue #59が固定したcanonical storage表現である。

```text
storage = unsigned 16-bit integer
SCALE   = 8192 = 2^13
raw     = round(semantic_value * SCALE)
```

expected-count（0.0..4.0）はraw 0..32768、probability（0.0..1.0。red-five
probability、wait probability等）はraw 0..8192として表す。0/1/2/3/4枚および
0.0/1.0はquantization errorなしでexactに表現できる。storage width（uint16）と
中間演算幅は分離し、中間演算ではPythonの任意精度intをそのまま使う
（NumPyのような追加dependencyは導入しない）。

semantic→raw変換（`expected_count_to_raw()` / `probability_to_raw()` /
`red_five_probability_to_raw()`）は、quantize（`value * SCALE`のround）する
前に、semantic value自体が`0.0 <= value <= 4.0`（expected count）または
`0.0 <= value <= 1.0`（probability）の範囲内かをfail-closedで検証する。
round後にraw rangeへ収まるかどうかでは判定しない。境界からわずかに外れた値
（例: expected countの`-1e-9`や`4.0 + 1e-9`）も、round結果に関わらず拒否する。

丸め規則はPython組み込み`round()`のround-half-to-even（銀行家丸め、
IEEE 754 roundTiesToEven相当）をそのまま採用する。half-way値
（例: `0.5 / SCALE`ちょうど）でも、独自の四捨五入や切り捨てへ変更しない。
"""

SCALE = 8192  # 2 ** 13

EXPECTED_COUNT_MAX_RAW = 4 * SCALE  # 32768
PROBABILITY_MAX_RAW = SCALE  # 8192
RED_FIVE_PROBABILITY_MAX_RAW = PROBABILITY_MAX_RAW  # 8192

_STORAGE_MAX_RAW = 0xFFFF  # unsigned 16-bit integerが表現できる上限


def raw_to_semantic(raw: int) -> float:
    """storage raw値（unsigned 16-bit整数）を意味上の値（float）へ変換する。"""
    _require_storage_raw(raw)
    return raw / SCALE


def _require_storage_raw(raw: int) -> None:
    if type(raw) is not int:
        raise TypeError("raw must be an int")
    if not 0 <= raw <= _STORAGE_MAX_RAW:
        raise ValueError("raw must fit within an unsigned 16-bit integer")


def expected_count_to_raw(value: float) -> int:
    """expected-countの意味上の値（0.0..4.0）をraw fixed-point値へ変換する。

    量子化（`round(value * SCALE)`）を行う前に、`value`自体が0.0..4.0の
    範囲内であることを検証する。roundした結果がraw range内に収まるとしても、
    quantize前のsemantic valueが範囲外なら拒否する。
    """
    return _semantic_to_raw(value, 0.0, 4.0, "expected_count")


def probability_to_raw(value: float) -> int:
    """probabilityの意味上の値（0.0..1.0）をraw fixed-point値へ変換する。

    `[0.0, 1.0]`のprobability channel全般（wait belief等）が共有する
    canonical変換である。量子化前に`value`自体が0.0..1.0の範囲内であることを
    検証し、丸めは既存のround-half-to-even contractに従う。
    """
    return _semantic_to_raw(value, 0.0, 1.0, "probability")


def red_five_probability_to_raw(value: float) -> int:
    """red-five probabilityの意味上の値（0.0..1.0）をraw fixed-point値へ変換する。

    量子化前に`value`自体が0.0..1.0の範囲内であることを検証する。
    """
    return _semantic_to_raw(value, 0.0, 1.0, "red_five_probability")


def _semantic_to_raw(
    value: float, min_value: float, max_value: float, field_name: str
) -> int:
    if type(value) is not float and type(value) is not int:
        raise TypeError(f"{field_name} must be a float or int")
    if type(value) is bool:
        raise TypeError(f"{field_name} must be a float or int")

    if not min_value <= value <= max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}")

    # round()はPythonの既定であるround-half-to-even(銀行家丸め)を使う。
    # このcanonical semantic→raw変換の丸め規則として固定し、独自の
    # 四捨五入・切り捨てへは変更しない。
    return round(value * SCALE)


def round_half_to_even_ratio(numerator: int, denominator: int) -> int:
    """`numerator / denominator`をexact rational上でround-half-to-evenする。

    Python組み込み`round()`と同じcanonical丸め規則（round-half-to-even /
    IEEE 754 roundTiesToEven相当）を、binary floatを経由せず整数算術だけで
    再現する。`round(numerator / denominator)`のように一度floatへ変換すると、
    floatの丸め誤差がcanonical rounding contractへ混入し得るため、比率の
    quantizationをexact integer domainで行う必要がある場合はこちらを使う。

    `numerator`は非負、`denominator`は正のintであることを要求する。
    """
    if type(numerator) is not int or type(denominator) is not int:
        raise TypeError("numerator and denominator must be int")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator < 0:
        raise ValueError("numerator must not be negative")

    quotient, remainder = divmod(numerator, denominator)
    twice_remainder = 2 * remainder
    if twice_remainder < denominator:
        return quotient
    if twice_remainder > denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1
