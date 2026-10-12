"""lisjong内部のPlayerPublicState型。

docs/policy-input-schema.md「PlayerPublicState」の意味契約を実装する。

`PlayerPublicState`は、自席を含む1 seat分の公開状態のsnapshotである。
`score`/`discards`/`melds`/`riichi`は`docs/policy-input-schema.md`の
`score` / `Discard` / `PublicMeld` / `RiichiState`をそれぞれ正本とする。
`riichi_declaration_discard_order`は同文書の「リーチ宣言牌位置」を正本とする。

このvalueはどのseatの状態かを自ら保持しない。`PolicyInput.players`側で
`players[index]のindex == Seat`によってseat identityが決まる設計であり、
`PlayerPublicState(seat=...)`のような二重保持は行わない。二重保持すると
`players[1].seat == Seat.SEAT_2`のような矛盾状態を型として作れてしまう。

他家のconcealed hand、drawn tile、waits、shanten、furiten内部状態、
非公開dora等はこのvalueへ含めない。自席の非公開情報は後続のOwnHandStateへ
分離する。
"""

from dataclasses import dataclass

from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import PublicMeld
from lisjong.policy_contract.riichi import RiichiState


def _normalize_tuple(values: object, element_type: type, field_name: str) -> tuple:
    """iterableをtupleへ正規化するだけで、要素の並び順は変更しない。

    `discards`は局全体のDiscard.orderと対応する時間関係を持つ履歴、
    `melds`はkakan時のsequence位置が未確定のまま正本文書へ残る collection
    であり、いずれも境界側が渡した並び順をこのvalueが勝手に修復
    （canonical sort）しない。
    """
    try:
        items = tuple(values)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable") from None
    if any(not isinstance(item, element_type) for item in items):
        raise TypeError(
            f"{field_name} must contain only {element_type.__name__} instances"
        )

    return items


@dataclass(frozen=True, slots=True)
class PlayerPublicState:
    """1 seat分の公開状態のsnapshot。

    ```text
    PlayerPublicState
    ├── score
    ├── discards
    ├── melds
    ├── riichi
    └── riichi_declaration_discard_order
    ```

    `discards`と`melds`はどちらも入力sequenceの並び順をそのまま保持する。

    `riichi_declaration_discard_order`は、このseatのリーチ宣言牌の
    `Discard.order`（局内の打牌の通し番号）である。このseatの何打目かではない。
    リーチ宣言と宣言牌は全seatに見える公開情報であり、hidden informationを
    増やさない。リーチ未宣言、宣言牌を切る前の`DECLARED`、値を持たない古い記録の
    互換読み込みでは`None`になる。

    次の不整合は構築時に拒否し、値を捨てたり補正したりしない。

    - `riichi`が`NONE`なのに値がある
    - 値があるのに、その`order`を持つ打牌が`discards`にない

    宣言牌より後の打牌がツモ切りであること、`ACCEPTED`なのに値がないこと、
    seatをまたぐ`order`の一意性は、採用規則や外部表現に依存するためここでは
    検証しない。Adapter側の整合検証の責務とする。
    """

    score: int
    discards: tuple[Discard, ...]
    melds: tuple[PublicMeld, ...]
    riichi: RiichiState
    riichi_declaration_discard_order: int | None = None

    def __post_init__(self) -> None:
        if type(self.score) is not int:
            raise TypeError("score must be an int")

        discards = _normalize_tuple(self.discards, Discard, "discards")
        melds = _normalize_tuple(self.melds, PublicMeld, "melds")

        if not isinstance(self.riichi, RiichiState):
            raise TypeError("riichi must be a RiichiState")

        declaration_order = self.riichi_declaration_discard_order
        if declaration_order is not None:
            if type(declaration_order) is not int:
                raise TypeError(
                    "riichi_declaration_discard_order must be an int or None"
                )
            if self.riichi is RiichiState.NONE:
                raise ValueError(
                    "riichi_declaration_discard_order must be None when riichi is NONE"
                )
            if all(discard.order != declaration_order for discard in discards):
                raise ValueError(
                    "riichi_declaration_discard_order must refer to one of discards"
                )

        object.__setattr__(self, "discards", discards)
        object.__setattr__(self, "melds", melds)
