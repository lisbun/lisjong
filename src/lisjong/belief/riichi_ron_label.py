"""単独リーチ者に対する牌ごとの放銃ラベル（lisbun/lisjong#237 S1、ラベルA）。

完全情報（リーチ者の手牌）から、「判断時点でリーチ者がその牌種でロンできるか」
を決めるtraining-only truthである。推定器の教師ラベルと診断にだけ使い、
Policyの判断入力（online decision path）へは渡さない。

```text
ron_tile_types = wait_tile_types   if not furiten
               = ∅                 if furiten
```

- 構造的な待ち（`wait_tile_types`）は、既存の
  `exact_wait_ground_truth.exact_hand_belief_with_waits()`の
  `wait_probability`が正の牌種をそのまま使う。待ちの判定をここで再実装しない
- 役: リーチ者はリーチ役を持つため、役なしでロンできない待ちは生じない
- フリテン:
  - 捨て牌フリテン: リーチ者自身の捨て牌（他家に鳴かれた牌を含む）に待ちの
    牌種が1つでもあれば、すべての待ちでロンできない
  - リーチ後の見逃しフリテン: リーチ宣言打牌より後に、リーチ者が和了できた
    牌（他家の打牌・加槓牌・自摸牌）を和了しなかった場合、以後すべての待ちで
    ロンできない。どの牌を見逃したかは呼び出し側が`passed_tile_types`で渡す
- ラベルB（実際にロンされたか）は対局結果から別に求め、本moduleでは扱わない

リーチ者は聴牌しているはずなので、待ちが1つもない入力はデータ不整合として
`ValueError`にする（silentに「ロンできない」へ変換しない）。
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from lisjong.belief.canonical_axes import tile_type_from_index
from lisjong.belief.exact_wait_ground_truth import exact_hand_belief_with_waits
from lisjong.policy_contract.meld import PublicMeld
from lisjong.policy_contract.tile import Tile, TileType

_TILE_KIND_COUNT = 34


class RiichiFuritenReason(Enum):
    """リーチ者がロンできない理由。両方が成立する場合は両方を記録する。"""

    OWN_DISCARD = "own_discard"
    PASSED_AFTER_RIICHI = "passed_after_riichi"


@dataclass(frozen=True, slots=True)
class RiichiRonLabel:
    """1人のリーチ者について、判断時点の牌種ごとのロン可否。"""

    wait_tile_types: frozenset[TileType]
    furiten_reasons: frozenset[RiichiFuritenReason]

    def __post_init__(self) -> None:
        if not self.wait_tile_types:
            raise ValueError("a riichi player must have at least one wait")
        if any(not isinstance(t, TileType) for t in self.wait_tile_types):
            raise TypeError("wait_tile_types must contain only TileType values")
        if any(not isinstance(r, RiichiFuritenReason) for r in self.furiten_reasons):
            raise TypeError("furiten_reasons must contain RiichiFuritenReason values")

    @property
    def furiten(self) -> bool:
        return bool(self.furiten_reasons)

    @property
    def ron_tile_types(self) -> frozenset[TileType]:
        return frozenset() if self.furiten else self.wait_tile_types

    def can_ron(self, tile_type: TileType) -> bool:
        if not isinstance(tile_type, TileType):
            raise TypeError("tile_type must be a TileType")
        return tile_type in self.ron_tile_types


def _tile_types(values: Iterable[object], name: str) -> frozenset[TileType]:
    result = set()
    for value in values:
        if isinstance(value, Tile):
            result.add(value.tile_type)
        elif isinstance(value, TileType):
            result.add(value)
        else:
            raise TypeError(f"{name} must contain only Tile or TileType values")
    return frozenset(result)


def riichi_ron_label(
    concealed_tiles: Iterable[Tile],
    own_melds: Iterable[PublicMeld] = (),
    *,
    own_discards: Iterable[Tile | TileType] = (),
    passed_tile_types: Iterable[Tile | TileType] = (),
) -> RiichiRonLabel:
    """リーチ者の完全情報から、判断時点のロン可否ラベルを返す。

    Args:
        concealed_tiles: リーチ者の手牌（打牌後のstable 13-equivalent）。
        own_melds: リーチ者の副露（暗槓のみの想定だが制限しない）。
        own_discards: リーチ者がこの局で捨てた牌すべて（鳴かれた牌を含む）。
        passed_tile_types: リーチ宣言打牌より後に、リーチ者が和了できる状況で
            見送った牌。見送ったかどうかの判定は呼び出し側の責務で、ここでは
            渡された牌種と待ちの交差だけを見る。
    """
    belief = exact_hand_belief_with_waits(concealed_tiles, own_melds)
    waits = frozenset(
        tile_type
        for tile_type in (tile_type_from_index(i) for i in range(_TILE_KIND_COUNT))
        if (belief.wait_probability(tile_type) or 0) > 0
    )
    if not waits:
        raise ValueError("the riichi player's hand is not tenpai")
    discards = _tile_types(own_discards, "own_discards")
    passed = _tile_types(passed_tile_types, "passed_tile_types")
    reasons = set()
    if waits & discards:
        reasons.add(RiichiFuritenReason.OWN_DISCARD)
    if waits & passed:
        reasons.add(RiichiFuritenReason.PASSED_AFTER_RIICHI)
    return RiichiRonLabel(wait_tile_types=waits, furiten_reasons=frozenset(reasons))


__all__ = [
    "RiichiFuritenReason",
    "RiichiRonLabel",
    "riichi_ron_label",
]
