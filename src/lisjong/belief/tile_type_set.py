"""canonical 34牌種axis上の牌種集合と、値配列へのmask適用・集約（Issue #221）。

有効牌等の牌種集合を、`tile_type_index()`のcanonical index（0..33）を昇順・
重複なしに並べたtupleで表す。新しい牌種順序やbitset / 専用classは作らない
（試作・計測の記録は`docs/kobalab-0004-belief-paijia.md` §7）。

```text
tile type set   tuple[int, ...]   canonical index昇順・重複なし・各0..33
empty set       ()                空判定は`len(tile_types) == 0`で明示的に行う
```

集合は牌種だけを表し、物理的な牌の重複、赤 / 通常、手出し / ツモ切りの
identityを持たない。34牌種の5は赤5を含むため、5の牌種を選ぶと赤5も
その中に数えられる。赤5専用のaxisはここでは扱わない。

値配列（`values`）は同じcanonical axisの長さ34の非負整数列で、例えば次を渡す。

| values | 尺度 | 選択位置の合計の意味 |
| --- | --- | --- |
| `TileConservationResult.remaining_tile_counts` | exact枚数 | 未見枚数ベースの受入枚数 |
| `HandBelief.expected_count_raw` | fixed-point raw（`SCALE`） | その相手が集合の牌を保持する期待枚数 |
| `NonPlayerHiddenBelief.expected_count_raw` | fixed-point raw（`SCALE`） | 集合の牌の非プレイヤー残余期待枚数 |

`mask_tile_type_values()` / `sum_tile_type_values()`は入力の尺度をそのまま保つ。
exact枚数を渡せばexact枚数、rawを渡せばraw（`SCALE = 8192`倍）の整数を返し、
float化・整数枚数への丸め戻し・1牌種分の上限（4枚、`4 * SCALE`）による
切り詰めを行わない。元の`values`は変更しない。mask後の値は元の手牌belief等の
完全な表現ではないため、`HandBelief`等として再構築しない。残余にはツモ山以外
（王牌等）も含まれ、mask後の期待枚数はツモ確率・和了確率ではない。

検証境界：`tile_type_set()`が入力indexを検証・正規化する唯一の境界である。
`intersect_tile_type_sets()` / `mask_tile_type_values()` /
`sum_tile_type_values()`は、`tile_type_set()`の結果か、同じ契約を構築時に
満たす集合（例：0004の`_DiscardStructures.improving_after()`）を前提とし、
呼び出しごとに集合を再検証しない。`values`は長さ34だけを検証する。
"""

from collections.abc import Iterable, Sequence

from lisjong.belief.tile_inventory import TILE_TYPE_COUNT


def tile_type_set(indices: Iterable[int]) -> tuple[int, ...]:
    """canonical 34牌種indexの列を、昇順・重複なしの牌種集合へ正規化する。

    各indexは`int`（`bool`は不可）かつ0..33でなければならない。重複は集合として
    除去し、合計へ二重に数えない。
    """
    try:
        values = tuple(indices)
    except TypeError:
        raise TypeError("indices must be an iterable of int") from None
    for index in values:
        if type(index) is not int:
            raise TypeError("indices must contain only int values")
        if not 0 <= index < TILE_TYPE_COUNT:
            raise ValueError(f"indices must be between 0 and {TILE_TYPE_COUNT - 1}")
    return tuple(sorted(set(values)))


def intersect_tile_type_sets(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    """2つの牌種集合の交差（canonical index昇順）。"""
    selected = set(right)
    return tuple(index for index in left if index in selected)


def _check_values(values: Sequence[int]) -> None:
    if len(values) != TILE_TYPE_COUNT:
        raise ValueError(f"values must contain exactly {TILE_TYPE_COUNT} values")


def mask_tile_type_values(
    values: Sequence[int], tile_types: tuple[int, ...]
) -> tuple[int, ...]:
    """集合の位置の値をそのまま残し、それ以外を0にした長さ34の値を返す。

    選択値だけを詰めた短い配列ではなく、同じcanonical axisの34要素である。
    """
    _check_values(values)
    masked = [0] * TILE_TYPE_COUNT
    for index in tile_types:
        masked[index] = values[index]
    return tuple(masked)


def sum_tile_type_values(values: Sequence[int], tile_types: tuple[int, ...]) -> int:
    """集合の位置の値の正確な合計（`values`と同じ尺度）。

    34要素の中間配列は作らない。空集合の合計は0である。
    """
    _check_values(values)
    return sum([values[index] for index in tile_types])


__all__ = [
    "intersect_tile_type_sets",
    "mask_tile_type_values",
    "sum_tile_type_values",
    "tile_type_set",
]
