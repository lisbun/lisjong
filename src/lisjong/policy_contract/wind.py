"""lisjong内部の風（東南西北）value型。

docs/policy-input-schema.md「RoundState」のround_wind、および自風
（(self_seat - dealer_seat) mod 4 から導出する値）の表現に使う基本domain値である。

WindはSeat（固定player座席位置）とは異なる概念であり、意図的にIntEnumではなく
通常のEnum（str値）として表現する。これにより、int値の一致だけでSeatと
Windが混同されることを型として防ぐ。
"""

from enum import Enum


class Wind(Enum):
    """場風または自風を表す。Seatとは別型であり、相互変換・比較しない。"""

    EAST = "east"
    SOUTH = "south"
    WEST = "west"
    NORTH = "north"
