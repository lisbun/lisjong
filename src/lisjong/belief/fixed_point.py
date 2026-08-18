"""非公開手牌belief用のfixed-point数値表現。

Issue #59が固定したcanonical storage表現である。

```text
storage = unsigned 16-bit integer
SCALE   = 8192 = 2^13
raw     = round(semantic_value * SCALE)
```

expected-count（0.0..4.0）はraw 0..32768、red-five probability（0.0..1.0）は
raw 0..8192として表す。0/1/2/3/4枚および0.0/1.0はquantization errorなしで
exactに表現できる。storage width（uint16）と中間演算幅は分離し、中間演算では
Pythonの任意精度intをそのまま使う（NumPyのような追加dependencyは導入しない）。
"""

SCALE = 8192  # 2 ** 13

EXPECTED_COUNT_MAX_RAW = 4 * SCALE  # 32768
RED_FIVE_PROBABILITY_MAX_RAW = SCALE  # 8192

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
    """expected-countの意味上の値（0.0..4.0）をraw fixed-point値へ変換する。"""
    return _semantic_to_raw(value, EXPECTED_COUNT_MAX_RAW, "expected_count")


def red_five_probability_to_raw(value: float) -> int:
    """red-five probabilityの意味上の値（0.0..1.0）をraw fixed-point値へ変換する。"""
    return _semantic_to_raw(value, RED_FIVE_PROBABILITY_MAX_RAW, "red_five_probability")


def _semantic_to_raw(value: float, max_raw: int, field_name: str) -> int:
    if type(value) is not float and type(value) is not int:
        raise TypeError(f"{field_name} must be a float or int")
    if type(value) is bool:
        raise TypeError(f"{field_name} must be a float or int")

    raw = round(value * SCALE)
    if not 0 <= raw <= max_raw:
        raise ValueError(f"{field_name} must be within its fixed-point range")
    return raw
