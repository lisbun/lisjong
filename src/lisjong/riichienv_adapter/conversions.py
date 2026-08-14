"""RiichiEnv外部値とlisjong内部valueの間の薄いvalue conversion utility。

docs/内部Actionモデル・Action identityが要求する「physical tile copy identityを
Policy契約へ持ち込まない」境界を実装するための最小限の変換関数を提供する。

Issue #29のコメント「#28との並行実装境界」が示すとおり、ここには
physical RiichiEnv tile ID → lisjong Tile、RiichiEnv player index → lisjong
Seatという薄い値変換だけを置き、state trackerやAction mappingの責務は
混ぜない。#28（materialized state）が同種の変換を先に必要とする場合、この
moduleを共通利用してよい。

RiichiEnv 0.4.8の物理牌ID（0..135）は、`riichienv-core`の内部表現として
`index = physical_id // 4`が牌種（0..33: 萬子1-9, 筒子1-9, 索子1-9,
東南西北白發中）、`copy = physical_id % 4`が同一牌種内の4枚のうちどれかを
表す。赤牌は5m/5p/5sそれぞれの`copy == 0`に固定される。この対応は
Issue #29着手前の実測（CPython 3.14.0rc2、RiichiEnv 0.4.8、136物理牌ID全件を
複数seedの初期配牌・ツモから収集し、`Action.to_mjai()`のMJAI表現と突き合わせて
確認）で裏付けている。`to_mjai()`だけを変換の正本にはせず、この算術的対応を
正本とする。
"""

from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

_SUIT_CATEGORY_BY_INDEX_GROUP = (
    TileCategory.MANZU,
    TileCategory.PINZU,
    TileCategory.SOUZU,
)
_HONOR_RANK_COUNT = 7
_SUITED_RANK_COUNT = 9
_TILE_KIND_COUNT = 34


def tile_from_physical_id(physical_id: int) -> Tile:
    """RiichiEnvの物理牌ID(0..135)をlisjongの`Tile`へ変換する。

    physical copy identity（`physical_id % 4`が表す同一牌種内のどのコピーか）は
    破棄し、赤牌区分だけを保持する。
    """
    if type(physical_id) is not int:
        raise TypeError("physical_id must be an int")
    if not 0 <= physical_id < _TILE_KIND_COUNT * 4:
        raise ValueError(
            f"physical_id must be between 0 and {_TILE_KIND_COUNT * 4 - 1}"
        )

    kind_index, copy_index = divmod(physical_id, 4)

    if kind_index < len(_SUIT_CATEGORY_BY_INDEX_GROUP) * _SUITED_RANK_COUNT:
        suit_group, rank_within_suit = divmod(kind_index, _SUITED_RANK_COUNT)
        category = _SUIT_CATEGORY_BY_INDEX_GROUP[suit_group]
        rank = rank_within_suit + 1
        is_red = copy_index == 0 and rank == 5
    else:
        category = TileCategory.HONOR
        rank = kind_index - len(_SUIT_CATEGORY_BY_INDEX_GROUP) * _SUITED_RANK_COUNT + 1
        is_red = False

    return Tile(TileType(category, rank), is_red)


def seat_from_player_index(player_index: int) -> Seat:
    """RiichiEnvのplayer index(0..3)をlisjongの`Seat`へ変換する。"""
    if type(player_index) is not int:
        raise TypeError("player_index must be an int")
    try:
        return Seat(player_index)
    except ValueError:
        raise ValueError(
            f"player_index must be between 0 and 3, got {player_index}"
        ) from None
