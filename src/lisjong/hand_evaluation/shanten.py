"""向聴数計算の公開契約。

Issue #50で確定した、外部環境に依存しない牌姿評価の入口である。入力はlisjong
内部の`Tile`列だけで、RiichiEnv / RiichiLab / mjai等の外部型を受け取らない。

このmoduleは入力のsnapshot、validation、34牌種countへの正規化、確定面子数の
判断までを担当し、実際の探索は`_python_shanten`へ委譲する。34牌種countは
privateな内部表現であり、公開APIにはしない。
"""

from collections.abc import Iterable

from lisjong.hand_evaluation import _python_shanten
from lisjong.policy_contract.tile import Tile, TileCategory

_CATEGORY_OFFSETS = {
    TileCategory.MANZU: 0,
    TileCategory.PINZU: 9,
    TileCategory.SOUZU: 18,
    TileCategory.HONOR: 27,
}
"""34牌種canonical representationのindex起点。

    0..8    manzu 1..9
    9..17   pinzu 1..9
    18..26  souzu 1..9
    27..33  honor 1..7
"""

_MAX_COPIES_PER_TILE_KIND = 4
"""赤5と通常5を合算した基礎牌種1種あたりの上限枚数。"""

_VALID_CONCEALED_TILE_COUNTS = frozenset({1, 2, 4, 5, 7, 8, 10, 11, 13, 14})
"""確定面子0..4個に対応する、有効な純手牌枚数。"""

_MELDLESS_TILE_COUNTS = frozenset({13, 14})
"""確定面子が0で、七対子・国士無双を候補にできる純手牌枚数。"""


def calculate_shanten(tiles: Iterable[Tile]) -> int:
    """純手牌（concealed tiles）から向聴数を返す。

    返り値は一般的な定義に従う。

        和了形: -1
        聴牌:    0
        一向聴:  1
        以下同様

    `tiles`には、副露・槓で既に確定したmeldの牌を含めない。確定面子数は純手牌
    枚数から判断するため、Chi / Pon / Daiminkan / Ankan / Kakanの具体的な
    identityや`PublicMeld`をこのAPIへ渡す必要はない。

        13 / 14枚 -> 確定面子 0
        10 / 11枚 -> 確定面子 1
         7 /  8枚 -> 確定面子 2
         4 /  5枚 -> 確定面子 3
         1 /  2枚 -> 確定面子 4

    `OwnHandState.drawn_tile`は`concealed_tiles`に含まれる既存契約上のmetadata
    であるため、追加の1枚として渡さない。

    確定面子が0の13 / 14枚の場合だけ、通常形・七対子・国士無双を比較して最小の
    向聴数を返す。それ以外の有効な純手牌枚数（11枚以下）は、既に確定面子が
    ある状態なので通常形だけを評価する。

    赤5と通常5は牌姿構造上同じ牌種として扱う。`Tile`自体のred distinctionは
    変更しない。

    向聴計算と無関係なphysical tile conservation（実牌セット上の赤5枚数等）は
    検証しない。

    Raises:
        TypeError: `tiles`がiterableでない、または`Tile`以外を含む場合。
        ValueError: 純手牌枚数が有効な集合に含まれない場合、または赤5と通常5を
            合算した基礎牌種が5枚以上ある場合。
    """
    snapshot = _snapshot_tiles(tiles)

    if len(snapshot) not in _VALID_CONCEALED_TILE_COUNTS:
        raise ValueError(
            "tiles must contain a concealed hand size of "
            f"{sorted(_VALID_CONCEALED_TILE_COUNTS)}, got {len(snapshot)}"
        )

    counts = _count_tile_kinds(snapshot)
    fixed_meld_count = _fixed_meld_count(len(snapshot))

    shanten = _python_shanten.calculate_standard_shanten(counts, fixed_meld_count)
    if len(snapshot) in _MELDLESS_TILE_COUNTS:
        shanten = min(
            shanten,
            _python_shanten.calculate_seven_pairs_shanten(counts),
            _python_shanten.calculate_thirteen_orphans_shanten(counts),
        )
    return shanten


def _snapshot_tiles(tiles: Iterable[Tile]) -> tuple[Tile, ...]:
    """入力をその場でtupleへ固定し、要素型を検証する。

    後段の判断がiterableの再走査や遅延評価に依存しないよう、最初に1回だけ
    消費してsnapshotを取る。
    """
    try:
        snapshot = tuple(tiles)
    except TypeError:
        raise TypeError("tiles must be an iterable of Tile") from None
    if any(not isinstance(tile, Tile) for tile in snapshot):
        raise TypeError("tiles must contain only Tile instances")
    return snapshot


def _count_tile_kinds(tiles: tuple[Tile, ...]) -> list[int]:
    """`Tile`列を34牌種countへ正規化する。

    赤5と通常5は同じindexへ加算する。入力順序に依存しない。
    """
    counts = [0] * _python_shanten.TILE_KIND_COUNT
    for tile in tiles:
        tile_type = tile.tile_type
        counts[_CATEGORY_OFFSETS[tile_type.category] + tile_type.rank - 1] += 1

    if any(count > _MAX_COPIES_PER_TILE_KIND for count in counts):
        raise ValueError(
            "tiles must not contain more than "
            f"{_MAX_COPIES_PER_TILE_KIND} copies of the same base tile kind"
        )
    return counts


def _fixed_meld_count(concealed_tile_count: int) -> int:
    """純手牌枚数から、副露・槓で確定済みの面子数を求める。"""
    return 4 - (concealed_tile_count - 1) // 3
