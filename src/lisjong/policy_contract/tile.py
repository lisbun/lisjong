"""lisjong内部の麻雀牌value型。

docs/policy-input-schema.md「Tileの意味契約」および
docs/action-identity.md「Tile identity」の意味契約を実装する。

Tile identity = base tile kind（TileType） + red distinction（is_red）

RiichiEnvのphysical tile ID（136枚ID）やcopy_indexは持たない。同じ基礎牌種かつ
同じ赤牌区分のphysical copy間の差は、lisjongのTileへ持ち込まない。
"""

from dataclasses import dataclass
from enum import Enum


class TileCategory(Enum):
    """萬子・筒子・索子・字牌の牌種category。"""

    MANZU = "manzu"
    PINZU = "pinzu"
    SOUZU = "souzu"
    HONOR = "honor"


@dataclass(frozen=True, slots=True)
class TileType:
    """赤牌区分を除いた基礎牌種（base tile kind）。

    数牌は1..9、字牌は1..7の範囲を持つ。字牌の順序自体はここでは固定しない。
    """

    category: TileCategory
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.category, TileCategory):
            raise TypeError("category must be a TileCategory")
        if type(self.rank) is not int:
            raise TypeError("rank must be an int")

        maximum_rank = 7 if self.category is TileCategory.HONOR else 9
        if not 1 <= self.rank <= maximum_rank:
            raise ValueError(f"rank must be between 1 and {maximum_rank}")


@dataclass(frozen=True, slots=True)
class Tile:
    """lisjong内部の麻雀牌value。physical copy identityを持たない。

    同じ基礎牌種かつ同じ赤牌区分ならvalueとして等しい。
    """

    tile_type: TileType
    is_red: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.tile_type, TileType):
            raise TypeError("tile_type must be a TileType")
        if type(self.is_red) is not bool:
            raise TypeError("is_red must be a bool")
        if self.is_red and (
            self.tile_type.category is TileCategory.HONOR or self.tile_type.rank != 5
        ):
            raise ValueError("only suited fives can be red")


# TileCategory.HONORのrank 1..7が具体的にどの字牌を指すかは、参考にした
# python-studyのtile.py自体には定義がなく、離れた場所（yaku.py）の役判定
# ロジックへ暗黙に埋め込まれていた（1..4=Wind順の風牌、5..7=白發中）。
# lisjongではこれを暗黙のまま引き継がず、MJAI（"E","S","W","N","P","F","C"）を
# 含む一般的な麻雀ソフトウェアの慣例に合わせた、明示的なlisjongの設計判断として
# 固定する。
EAST_WIND = TileType(TileCategory.HONOR, 1)
SOUTH_WIND = TileType(TileCategory.HONOR, 2)
WEST_WIND = TileType(TileCategory.HONOR, 3)
NORTH_WIND = TileType(TileCategory.HONOR, 4)
WHITE_DRAGON = TileType(TileCategory.HONOR, 5)
GREEN_DRAGON = TileType(TileCategory.HONOR, 6)
RED_DRAGON = TileType(TileCategory.HONOR, 7)


_CATEGORY_ORDER = {
    TileCategory.MANZU: 0,
    TileCategory.PINZU: 1,
    TileCategory.SOUZU: 2,
    TileCategory.HONOR: 3,
}


def tile_sort_key(tile: Tile) -> tuple[int, int, bool]:
    """Tileのcanonical順序を返す。

    docs/action-identity.mdが後続実装事項として残していたcanonical sort keyの
    具体化である。category → rank → red distinctionの順で比較し、
    TileCategoryのEnum文字列value、Python hash、object identity等の偶然の
    順序には依存しない。multiset canonicalization（consumed_tiles等）の
    ソートkeyとして使用する。
    """
    if not isinstance(tile, Tile):
        raise TypeError("tile must be a Tile")

    return (
        _CATEGORY_ORDER[tile.tile_type.category],
        tile.tile_type.rank,
        tile.is_red,
    )


def _canonicalize_tile_multiset(
    tiles: object, expected_count: int | None, field_name: str
) -> tuple[Tile, ...]:
    """multiset fieldを、入力順序に依存しないcanonical tupleへ正規化する。

    要素数と型だけを検証し、physical copy identityが存在しないlisjongの
    Tileでは正常な重複（同一semantic Tileの複数枚）を拒否しない。
    `expected_count`が`None`の場合、要素数を検証しない
    （`OwnHandState.concealed_tiles`等、副露数やdecision phaseにより
    枚数が変わる固定長を持たないmultiset向け）。`InternalAction`の
    consumed_tiles/tiles、`PublicMeld.tiles`、`OwnHandState.concealed_tiles`が
    使用する共通のcanonicalizationロジックである。
    """
    try:
        values = tuple(tiles)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable of Tile") from None
    if any(not isinstance(tile, Tile) for tile in values):
        raise TypeError(f"{field_name} must contain only Tile instances")
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(f"{field_name} must contain exactly {expected_count} tiles")

    return tuple(sorted(values, key=tile_sort_key))


def _require_uniform_tile_type(
    tiles: tuple[Tile, ...], reference: Tile, field_name: str
) -> None:
    """tilesのすべてが、referenceと同じ基礎牌種（TileType）であることを検証する。

    赤牌区分は無視し、`InternalAction`のPon/Daiminkan/Ankan/Kakanと、
    `PublicMeld`の対応するkindが共通して使用する。
    """
    if any(tile.tile_type != reference.tile_type for tile in tiles):
        raise ValueError(f"{field_name} must share the same base tile kind")
