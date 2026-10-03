"""待ち形の成立可能性を、観測者の枚数制約と河との関係に分けて導出する。

リーチ者が牌種`t`を待つ形（単騎・シャンポン・辺張・嵌張・両面の低い側/高い側）の
うち、どの形がリーチ者の手牌として成り立ち得るかを、観測者に見える情報だけから求める。
Championの古典的危険度score（`_classical_riichi_danger_score`）の形判定を、
意味の異なる層に分けて切り出したもので、Issue lisbun/lisjong#245の第1層・第2層に当たる。

```text
第1層 wait_shape_support  観測者から見た残り枚数だけで決まる形の成立可能性
第2層 river_relation      リーチ者の河と候補牌の関係（現物・両面の反対側の牌）
第3層                     Championの固定weight・安全牌処理（policies側が持つ）
```

「構造的な不可能性」と呼べるのは第1層だけである。リーチ者の手牌に必要な牌は、観測者から見て
未見（remaining）の牌でなければならないからである。第2層は、河の牌が待ちに含まれると
フリテンになるというロンの安全度の話であり、手牌の構造としてその形が成立しないことを意味しない
（フリテンのまま待つこともある）。第2層を構造的な不可能性として扱わない。

`remaining_tile_counts`は`derive_remaining_tile_inventory()`が返す34牌種の未見枚数
（0..4。赤5は通常5と同じ牌種として数える）である。
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.policy_contract.tile import TileCategory, TileType


@dataclass(frozen=True, slots=True)
class WaitShapeSupport:
    """牌種`t`を待つ各形が、残り枚数の上でリーチ者の手牌として成り立ち得るか。

    - ``tanki``: 手牌に`t`が1枚あり得る（残り1枚以上）
    - ``shanpon``: 手牌に`t`が2枚あり得る（残り2枚以上）
    - ``kanchan``: 手牌に`t-1`と`t+1`がありうる（数牌の2..8）
    - ``penchan``: 3なら`1,2`、7なら`8,9`がありうる
    - ``ryanmen_low_side``: `t+1,t+2`がありうる（数牌の1..6。`t`は両面の低い側）
    - ``ryanmen_high_side``: `t-2,t-1`がありうる（数牌の4..9。`t`は両面の高い側）

    字牌は``tanki``と``shanpon``だけが成り立ち得る。国士無双の待ち（手牌に待ち牌を
    持たない形）は扱わない。
    """

    tanki: bool
    shanpon: bool
    kanchan: bool
    penchan: bool
    ryanmen_low_side: bool
    ryanmen_high_side: bool

    @property
    def count(self) -> int:
        return (
            self.tanki
            + self.shanpon
            + self.kanchan
            + self.penchan
            + self.ryanmen_low_side
            + self.ryanmen_high_side
        )


@dataclass(frozen=True, slots=True)
class RiverRelation:
    """リーチ者の河と候補牌の関係。ロンの安全度に関わり、構造的な不可能性ではない。

    - ``in_river``: 候補牌と同じ牌種がリーチ者の河にある（現物）
    - ``ryanmen_low_far_end_in_river``: `t+3`が河にある（`t`を低い側とする両面の反対側）
    - ``ryanmen_high_far_end_in_river``: `t-3`が河にある（`t`を高い側とする両面の反対側）
    """

    in_river: bool
    ryanmen_low_far_end_in_river: bool
    ryanmen_high_far_end_in_river: bool


def _remaining(
    remaining_tile_counts: Sequence[int], category: TileCategory, rank: int
) -> int:
    return remaining_tile_counts[tile_type_index(TileType(category, rank))]


def wait_shape_support(
    candidate: TileType, remaining_tile_counts: Sequence[int]
) -> WaitShapeSupport:
    """第1層: `candidate`を待つ各形の、残り枚数による成立可能性を返す。"""
    if not isinstance(candidate, TileType):
        raise TypeError("candidate must be a TileType")
    if len(remaining_tile_counts) != 34:
        raise ValueError("remaining_tile_counts must have 34 entries")

    own = remaining_tile_counts[tile_type_index(candidate)]
    tanki = own >= 1
    shanpon = own >= 2
    if candidate.category is TileCategory.HONOR:
        return WaitShapeSupport(tanki, shanpon, False, False, False, False)

    category, rank = candidate.category, candidate.rank

    def remaining(target: int) -> int:
        return _remaining(remaining_tile_counts, category, target)

    penchan = (rank == 3 and remaining(1) > 0 and remaining(2) > 0) or (
        rank == 7 and remaining(8) > 0 and remaining(9) > 0
    )
    kanchan = 2 <= rank <= 8 and remaining(rank - 1) > 0 and remaining(rank + 1) > 0
    ryanmen_low_side = (
        1 <= rank <= 6 and remaining(rank + 1) > 0 and remaining(rank + 2) > 0
    )
    ryanmen_high_side = (
        4 <= rank <= 9 and remaining(rank - 2) > 0 and remaining(rank - 1) > 0
    )
    return WaitShapeSupport(
        tanki, shanpon, kanchan, penchan, ryanmen_low_side, ryanmen_high_side
    )


def river_relation(candidate: TileType, river: Collection[TileType]) -> RiverRelation:
    """第2層: `candidate`とリーチ者の河（牌種の集まり）の関係を返す。"""
    if not isinstance(candidate, TileType):
        raise TypeError("candidate must be a TileType")
    in_river = candidate in river
    if candidate.category is TileCategory.HONOR:
        return RiverRelation(in_river, False, False)
    category, rank = candidate.category, candidate.rank
    return RiverRelation(
        in_river=in_river,
        ryanmen_low_far_end_in_river=rank + 3 <= 9
        and TileType(category, rank + 3) in river,
        ryanmen_high_far_end_in_river=rank - 3 >= 1
        and TileType(category, rank - 3) in river,
    )


__all__ = [
    "RiverRelation",
    "WaitShapeSupport",
    "river_relation",
    "wait_shape_support",
]
