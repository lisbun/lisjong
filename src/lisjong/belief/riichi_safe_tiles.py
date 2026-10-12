"""リーチ者ごとの確定安全牌を、判断時点の公開情報だけから導く。

Issue lisbun/lisjong#302（設計: `docs/riichi-passed-tile-genbutsu-design.md` 3節）。

リーチ者`r`に対する確定安全牌種を次で定める。

```text
safe(r) = river(r) ∪ passed(r)

river(r)   r の捨て牌履歴にある牌種（鳴かれた牌・宣言牌を含む）
passed(r)  r のリーチ宣言牌より後に、r 以外の席（自席を含む）が切った牌の牌種
```

「確定安全」は**`r`にロンされない**ことだけを意味する。リーチしていない他家へのロン、
`r`のツモ、放銃以外の損失には何も言わない。「全リーチ者に安全」は「全員に安全」ではない。

## 前提とする規則

`passed(r)`が`r`にロンされないのは、次の2つの規則による。

1. リーチ後の見逃しは、その局の終わりまでフリテンになる
2. リーチ後は待ちが変わらない（打牌はツモ切りだけで、暗槓は待ちを変えない場合にだけ許される）

この規則を採らない環境では`passed(r)`は確定安全ではない。対応を確認した環境は
`docs/riichi-safe-tiles.md`に記録する。未確認の環境へ自動的には適用しない。この前提の
identityは`RIICHI_SAFE_TILE_RULE_ASSUMPTION`で、利用する側が自分の設定・環境identityへ結び付ける。

## 利用できる時点

**自席の打牌判断からだけ呼ぶ。** この時点では、履歴にあるすべての打牌へのロン応答が解決済み
である。他家の打牌への応答判断（鳴き・ロン・Pass）では、履歴の最後の打牌へのロン応答が未解決で、
その牌が通ったかどうかを`Discard.order`だけからは区別できない。

誤用を検出する補助として、自席の手牌が打牌前の枚数（3n+2枚）でない入力を拒否する。
これは補助チェックであり、枚数が合うことは「ロン応答が解決済み」を保証しない。
呼び出し側の契約と併用する。

## 宣言牌の位置が分からないリーチ者

`PlayerPublicState.riichi_declaration_discard_order`が`None`のリーチ者（宣言牌を切る前の
`DECLARED`、または古い記録の互換読み込み）は、`passed(r)`を空とし、河だけを使う。
位置を推測して広げない。
"""

from dataclasses import dataclass

from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import TileType

RIICHI_SAFE_TILE_RULE_ASSUMPTION = "riichi-missed-win-furiten-and-fixed-waits-v1"
"""前提とする規則のidentity。確認していない環境との組合せで使わない。"""


@dataclass(frozen=True, slots=True)
class RiichiPlayerSafeTiles:
    """1人のリーチ者に対する確定安全牌種。そのリーチ者にロンされない牌種だけを表す。"""

    riichi_seat: Seat
    river_tile_types: frozenset[TileType]
    passed_tile_types: frozenset[TileType]
    declaration_known: bool

    @property
    def safe_tile_types(self) -> frozenset[TileType]:
        """このリーチ者にロンされない牌種。他のリーチ者・非リーチ者への安全は含まない。"""
        return self.river_tile_types | self.passed_tile_types


@dataclass(frozen=True, slots=True)
class RiichiSafeTiles:
    """判断時点のリーチ者全員について、リーチ者ごとの確定安全牌種を持つ。"""

    by_riichi_player: tuple[RiichiPlayerSafeTiles, ...]

    @property
    def safe_against_all_riichi_players(self) -> frozenset[TileType]:
        """すべてのリーチ者にロンされない牌種。

        リーチしていない他家への安全は含まない。リーチ者がいないときは空集合を返す
        （「どの牌も安全」という意味にはしない）。
        """
        if not self.by_riichi_player:
            return frozenset()
        safe = self.by_riichi_player[0].safe_tile_types
        for player in self.by_riichi_player[1:]:
            safe = safe & player.safe_tile_types
        return safe

    @property
    def all_declarations_known(self) -> bool:
        """全リーチ者の宣言牌の位置が分かっているか。"""
        return all(player.declaration_known for player in self.by_riichi_player)


def derive_riichi_safe_tiles(policy_input: PolicyInput) -> RiichiSafeTiles:
    """自席の打牌判断の`PolicyInput`から、リーチ者ごとの確定安全牌種を導く。

    リーチ者は、リーチ宣言済み（成立前を含む）の他家である。席の順に返す。
    自席の手牌が打牌前の枚数でない入力は`ValueError`で拒否する。
    """
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("policy_input must be a PolicyInput")
    if len(policy_input.own_hand.concealed_tiles) % 3 != 2:
        raise ValueError(
            "riichi safe tiles are defined only for the own discard decision "
            "(concealed tile count must be 3n+2)"
        )

    self_index = int(policy_input.self_seat)
    players = policy_input.players
    derived = []
    for index, player in enumerate(players):
        if index == self_index or player.riichi is RiichiState.NONE:
            continue
        declaration_order = player.riichi_declaration_discard_order
        passed: frozenset[TileType] = frozenset()
        if declaration_order is not None:
            passed = frozenset(
                discard.tile.tile_type
                for other_index, other in enumerate(players)
                if other_index != index
                for discard in other.discards
                if discard.order > declaration_order
            )
        derived.append(
            RiichiPlayerSafeTiles(
                riichi_seat=Seat(index),
                river_tile_types=frozenset(
                    discard.tile.tile_type for discard in player.discards
                ),
                passed_tile_types=passed,
                declaration_known=declaration_order is not None,
            )
        )
    return RiichiSafeTiles(by_riichi_player=tuple(derived))


__all__ = [
    "RIICHI_SAFE_TILE_RULE_ASSUMPTION",
    "RiichiPlayerSafeTiles",
    "RiichiSafeTiles",
    "derive_riichi_safe_tiles",
]
