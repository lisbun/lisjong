"""lisjong共通Policy契約の基本value型。

Issue #11で確定した共通Policy境界、Policy入力、InternalAction、Action identityの
意味契約を、Python valueとして表現するための最小限の型を提供する。

RiichiEnv、RiichiLab、mjai、WebSocket等の外部library固有型へは依存しない。
Policy、RiichiEnv Adapter、Local game runner、RiichiLab Clientが共通して利用する。
"""

from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import (
    EAST_WIND,
    GREEN_DRAGON,
    NORTH_WIND,
    RED_DRAGON,
    SOUTH_WIND,
    WEST_WIND,
    WHITE_DRAGON,
    Tile,
    TileCategory,
    TileType,
)
from lisjong.policy_contract.wind import Wind

__all__ = [
    "EAST_WIND",
    "GREEN_DRAGON",
    "MeldKind",
    "NORTH_WIND",
    "RED_DRAGON",
    "RiichiState",
    "SOUTH_WIND",
    "Seat",
    "Tile",
    "TileCategory",
    "TileType",
    "WEST_WIND",
    "WHITE_DRAGON",
    "Wind",
]
