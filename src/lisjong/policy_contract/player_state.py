"""lisjong内部のPlayerPublicState型。

docs/policy-input-schema.md「PlayerPublicState」の意味契約を実装する。

`PlayerPublicState`は、自席を含む1 seat分の公開状態のsnapshotである。
`score`/`discards`/`melds`/`riichi`は`docs/policy-input-schema.md`の
`score` / `Discard` / `PublicMeld` / `RiichiState`をそれぞれ正本とする。

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
    └── riichi
    ```

    `discards`と`melds`はどちらも入力sequenceの並び順をそのまま保持する。
    """

    score: int
    discards: tuple[Discard, ...]
    melds: tuple[PublicMeld, ...]
    riichi: RiichiState

    def __post_init__(self) -> None:
        if type(self.score) is not int:
            raise TypeError("score must be an int")

        discards = _normalize_tuple(self.discards, Discard, "discards")
        melds = _normalize_tuple(self.melds, PublicMeld, "melds")

        if not isinstance(self.riichi, RiichiState):
            raise TypeError("riichi must be a RiichiState")

        object.__setattr__(self, "discards", discards)
        object.__setattr__(self, "melds", melds)
