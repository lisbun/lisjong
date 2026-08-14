"""lisjong内部の打牌履歴value型。

docs/policy-input-schema.md「Discard」の意味契約を実装する。

`Discard`は「今この牌を切るという選択」を表す`DiscardAction`
（action.pyを参照）とは別物であり、「すでに行われた打牌履歴」を表す
state値である。混同を避けるため、`action.py`とは別moduleに置く。
"""

from dataclasses import dataclass

from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile


def _require_tile(value: object, field_name: str) -> None:
    if not isinstance(value, Tile):
        raise TypeError(f"{field_name} must be a Tile")


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")


def _require_optional_seat(value: object, field_name: str) -> None:
    if value is not None and not isinstance(value, Seat):
        raise TypeError(f"{field_name} must be a Seat or None")


@dataclass(frozen=True, slots=True)
class Discard:
    """あるseatがこの局で行った打牌1件の履歴。

    現在河に残っている牌だけを表すのではなく、鳴きに利用された打牌も含めて
    履歴から削除しない。discarder自身のseatはこのvalueへ保持しない
    （正本文書の概念schemaどおりであり、PlayerPublicState側がどのseatの
    履歴かを文脈として持つ）。

    以下はDiscard単体では検証できないため、ここでは検証しない。

    - 同一局でorderが一意であること、連番であること
    - called_byが実際にその牌を鳴いたseatであること
    - called_by != discarder（discarder seat自体を持たないため）
    - 鳴き種別との整合、ron対象であること／ないこと

    これらは後続のAdapter / Policy呼び出し境界の責務として残す。
    """

    tile: Tile
    tsumogiri: bool
    order: int
    called_by: Seat | None

    def __post_init__(self) -> None:
        _require_tile(self.tile, "tile")
        _require_bool(self.tsumogiri, "tsumogiri")
        if type(self.order) is not int:
            raise TypeError("order must be an int")
        if self.order < 0:
            raise ValueError("order must not be negative")
        _require_optional_seat(self.called_by, "called_by")
