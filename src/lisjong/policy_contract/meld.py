"""lisjong内部の副露・槓state型。

docs/policy-input-schema.md「PublicMeld」の意味契約を実装する。

`PublicMeld`はevent履歴ではなく、そのplayerが現在保持する副露・槓状態の
snapshotである。`ChiAction`等のInternalActionそのものではない
（action.pyを参照）。`MeldKind`はこのvalueが表す副露・槓種別であり、
`TileType`と`Tile`の関係と同様にPublicMeldとは独立した小さなenumだが、
PublicMeldから直接使用する密結合な概念のため、同じmoduleにまとめている。
"""

from dataclasses import dataclass
from enum import Enum

from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import (
    Tile,
    TileCategory,
    _canonicalize_tile_multiset,
    _require_uniform_tile_type,
)


class MeldKind(Enum):
    """PublicMeld.kindが区別する副露・槓種別。"""

    CHI = "chi"
    PON = "pon"
    DAIMINKAN = "daiminkan"
    ANKAN = "ankan"
    KAKAN = "kakan"


_TILE_COUNT_BY_KIND = {
    MeldKind.CHI: 3,
    MeldKind.PON: 3,
    MeldKind.DAIMINKAN: 4,
    MeldKind.ANKAN: 4,
    MeldKind.KAKAN: 4,
}


def _require_meld_kind(value: object, field_name: str) -> None:
    if not isinstance(value, MeldKind):
        raise TypeError(f"{field_name} must be a MeldKind")


def _require_optional_seat(value: object, field_name: str) -> None:
    if value is not None and not isinstance(value, Seat):
        raise TypeError(f"{field_name} must be a Seat or None")


def _require_optional_tile(value: object, field_name: str) -> None:
    if value is not None and not isinstance(value, Tile):
        raise TypeError(f"{field_name} must be a Tile or None")


@dataclass(frozen=True, slots=True)
class PublicMeld:
    """あるseatが現在保持する副露・槓1件のsnapshot。

    kindごとに、正本文書どおり次のfield意味を持つ。

    CHI / PON / DAIMINKAN: from_seat・called_tileは必須。
    ANKAN:                 from_seat・called_tileはNone。
    KAKAN:                 from_seat・called_tileは元Ponの値を維持する
                            （Noneにしない）。

    ownerとなるseat自体はこのvalueへ保持しない。「from_seatがownerの上家で
    あること」等、他seatとの関係を要するContext整合条件は、この値単体では
    検証できないため、後続のAdapter / Policy呼び出し境界の責務として残す。
    source_meld_id、source_meld_index、Python object identityによる元Pon
    参照は持たない。
    """

    kind: MeldKind
    tiles: tuple[Tile, ...]
    from_seat: Seat | None
    called_tile: Tile | None

    def __post_init__(self) -> None:
        _require_meld_kind(self.kind, "kind")
        _require_optional_seat(self.from_seat, "from_seat")
        _require_optional_tile(self.called_tile, "called_tile")

        expected_count = _TILE_COUNT_BY_KIND[self.kind]
        tiles = _canonicalize_tile_multiset(self.tiles, expected_count, "tiles")

        if self.kind is MeldKind.ANKAN:
            if self.from_seat is not None:
                raise ValueError("ankan must not have a from_seat")
            if self.called_tile is not None:
                raise ValueError("ankan must not have a called_tile")

            _require_uniform_tile_type(tiles, tiles[0], "tiles")
        else:
            if self.from_seat is None:
                raise ValueError(f"{self.kind} must have a from_seat")
            if self.called_tile is None:
                raise ValueError(f"{self.kind} must have a called_tile")
            if self.called_tile not in tiles:
                raise ValueError("called_tile must be one of tiles")

            if self.kind is MeldKind.CHI:
                category = self.called_tile.tile_type.category
                if category is TileCategory.HONOR:
                    raise ValueError("chi must not use honor tiles")
                if any(tile.tile_type.category is not category for tile in tiles):
                    raise ValueError("chi tiles must be the same suit")

                ranks = sorted(tile.tile_type.rank for tile in tiles)
                if ranks != list(range(ranks[0], ranks[0] + 3)):
                    raise ValueError("chi tiles must form three consecutive ranks")
            else:
                _require_uniform_tile_type(tiles, self.called_tile, "tiles")

        object.__setattr__(self, "tiles", tiles)
