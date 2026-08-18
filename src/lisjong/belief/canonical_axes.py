"""非公開手牌beliefのcanonical axis変換。

Issue #59が固定した、Wind axis・34牌種axis・赤5 axisの明示的な相互変換を
提供する。`list(Enum).index(...)`、Enum定義順、dict iteration order、
hash、object identity等の偶然の順序には依存せず、すべて明示的なlookup
tableで変換する。

player axisはSeat（固定player座席位置）ではなくWindである。EASTは常に
現在のdealerを表し、`RoundState.dealer_seat`から`wind_for_seat()` /
`seat_for_wind()`で明示的に解決する。SeatとWindを相互変換できるvalueを
Seat自体やWind自体へは持たせない。
"""

from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import TileCategory, TileType
from lisjong.policy_contract.wind import Wind

_WIND_INDEX = {
    Wind.EAST: 0,
    Wind.SOUTH: 1,
    Wind.WEST: 2,
    Wind.NORTH: 3,
}
_INDEX_WIND = {index: wind for wind, index in _WIND_INDEX.items()}

# category → (canonical indexの開始位置, 有効なrank数)
_CATEGORY_BASE_INDEX = {
    TileCategory.MANZU: (0, 9),
    TileCategory.PINZU: (9, 9),
    TileCategory.SOUZU: (18, 9),
    TileCategory.HONOR: (27, 7),
}

# indexからcategoryへの逆変換。dict iteration orderへ依存しないよう、
# 明示的なrange境界の固定tupleとして持つ（0..8=MANZU, 9..17=PINZU,
# 18..26=SOUZU, 27..33=HONOR）。
_INDEX_CATEGORY_RANGES = (
    (0, 9, TileCategory.MANZU),
    (9, 9, TileCategory.PINZU),
    (18, 9, TileCategory.SOUZU),
    (27, 7, TileCategory.HONOR),
)

_RED_FIVE_INDEX = {
    TileCategory.MANZU: 0,
    TileCategory.PINZU: 1,
    TileCategory.SOUZU: 2,
}


def wind_index(wind: Wind) -> int:
    """Windをcanonical player axis index（0=EAST..3=NORTH）へ変換する。"""
    if not isinstance(wind, Wind):
        raise TypeError("wind must be a Wind")
    return _WIND_INDEX[wind]


def wind_from_index(index: int) -> Wind:
    """canonical player axis index（0..3）をWindへ変換する。"""
    if type(index) is not int:
        raise TypeError("index must be an int")
    if index not in _INDEX_WIND:
        raise ValueError("index must be between 0 and 3")
    return _INDEX_WIND[index]


def tile_type_index(tile_type: TileType) -> int:
    """TileTypeをcanonical 34牌種index（0..33）へ変換する。"""
    if not isinstance(tile_type, TileType):
        raise TypeError("tile_type must be a TileType")
    base, _size = _CATEGORY_BASE_INDEX[tile_type.category]
    return base + (tile_type.rank - 1)


def tile_type_from_index(index: int) -> TileType:
    """canonical 34牌種index（0..33）をTileTypeへ変換する。

    `_INDEX_CATEGORY_RANGES`の明示的なrange境界だけで解決し、dictの
    iteration orderには依存しない。
    """
    if type(index) is not int:
        raise TypeError("index must be an int")
    for base, size, category in _INDEX_CATEGORY_RANGES:
        if base <= index < base + size:
            return TileType(category, index - base + 1)
    raise ValueError("index must be between 0 and 33")


def red_five_index(category: TileCategory) -> int:
    """数牌categoryをcanonical赤5 index（0=5m, 1=5p, 2=5s）へ変換する。"""
    if not isinstance(category, TileCategory):
        raise TypeError("category must be a TileCategory")
    if category not in _RED_FIVE_INDEX:
        raise ValueError("category must be a suited TileCategory (manzu/pinzu/souzu)")
    return _RED_FIVE_INDEX[category]


def wind_for_seat(seat: Seat, dealer_seat: Seat) -> Wind:
    """固定seat位置と現在のdealer_seatから、そのseatの自風を解決する。"""
    if not isinstance(seat, Seat):
        raise TypeError("seat must be a Seat")
    if not isinstance(dealer_seat, Seat):
        raise TypeError("dealer_seat must be a Seat")
    return wind_from_index((int(seat) - int(dealer_seat)) % 4)


def seat_for_wind(wind: Wind, dealer_seat: Seat) -> Seat:
    """自風と現在のdealer_seatから、対応する固定seat位置を解決する。"""
    if not isinstance(dealer_seat, Seat):
        raise TypeError("dealer_seat must be a Seat")
    return Seat((wind_index(wind) + int(dealer_seat)) % 4)


def concealed_hand_offset(wind: Wind, tile_type: TileType) -> int:
    """`concealed_hand_belief`（Wind-major / row-major, shape [4, 34]）の
    flattened offsetを返す。

    ```text
    offset = wind_index * 34 + tile_type_index
    ```
    """
    return wind_index(wind) * 34 + tile_type_index(tile_type)


def red_five_offset(wind: Wind, category: TileCategory) -> int:
    """`concealed_red_five_belief`（Wind-major / row-major, shape [4, 3]）の
    flattened offsetを返す。

    ```text
    offset = wind_index * 3 + red_five_index
    ```
    """
    return wind_index(wind) * 3 + red_five_index(category)
