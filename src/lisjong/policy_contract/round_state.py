"""lisjong内部のRoundState型。

docs/policy-input-schema.md「RoundState」の意味契約を実装する。

RiichiEnv固有の整数値、`len(env.wall)`等の環境固有container size、具体的な
`live_wall_tiles_remaining`算出algorithmは持ち込まない。fieldの意味契約と
具体的な算出方法を分離し、算出責務は後続のRiichiEnv Adapterへ残す。
"""

from dataclasses import dataclass

from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile
from lisjong.policy_contract.wind import Wind


def _normalize_tile_sequence(values: object, field_name: str) -> tuple[Tile, ...]:
    """iterableをtupleへ正規化するだけで、要素の並び順は変更しない。

    dora_indicatorsは公開された順序を持つsequenceであり、multisetではない。
    境界側が渡した並び順をこのvalueが勝手に修復（canonical sort）しない。
    """
    try:
        items = tuple(values)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable") from None
    if any(not isinstance(item, Tile) for item in items):
        raise TypeError(f"{field_name} must contain only Tile instances")

    return items


@dataclass(frozen=True, slots=True)
class RoundState:
    """局全体で共有される公開状態のsnapshot。

    ```text
    RoundState
    ├── round_wind
    ├── hand_number
    ├── dealer_seat
    ├── honba
    ├── riichi_sticks
    ├── dora_indicators
    └── live_wall_tiles_remaining
    ```

    `round_wind`はEAST/SOUTHへ制限せず、`Wind`の4値すべてを許容する。正本文書は
    「場風を表すlisjong内部の麻雀ドメイン値」とだけ定義しており、初期rule上の
    想定局数からEAST/SOUTHだけに値型を閉じる根拠がない。RiichiEnv調査記録でも
    `4p-red-half`で西1〜西4への遷移を実測しており、西場を排除すると既実測の
    局面を表現できなくなる。

    `dealer_seat`が自風・`round_wind`・`hand_number`と整合するか等、複数fieldに
    またがる整合や他stateとの照合は、この値単体では検証できないため後続の
    Adapter / PolicyInput境界へ残す。

    `dora_indicators`は空tupleも許容する。通常の4人麻雀の実decisionでは
    公開済み表示牌が1枚以上あるが、それはPolicy decisionを生成する環境・
    タイミング側の条件であり、`RoundState`単一値の構造的不変条件ではない。
    正本文書も「現在公開済みのドラ表示牌だけを、公開された順序で保持する」と
    定義するのみで非空を明記していないため、この値型では下限を課さない。
    非空であることが必要な場合は、後続のAdapter / DecisionContext境界で
    context整合条件として検証する。
    """

    round_wind: Wind
    hand_number: int
    dealer_seat: Seat
    honba: int
    riichi_sticks: int
    dora_indicators: tuple[Tile, ...]
    live_wall_tiles_remaining: int

    def __post_init__(self) -> None:
        if not isinstance(self.round_wind, Wind):
            raise TypeError("round_wind must be a Wind")

        if type(self.hand_number) is not int:
            raise TypeError("hand_number must be an int")
        if not 1 <= self.hand_number <= 4:
            raise ValueError("hand_number must be between 1 and 4")

        if not isinstance(self.dealer_seat, Seat):
            raise TypeError("dealer_seat must be a Seat")

        if type(self.honba) is not int:
            raise TypeError("honba must be an int")
        if self.honba < 0:
            raise ValueError("honba must not be negative")

        if type(self.riichi_sticks) is not int:
            raise TypeError("riichi_sticks must be an int")
        if self.riichi_sticks < 0:
            raise ValueError("riichi_sticks must not be negative")

        dora_indicators = _normalize_tile_sequence(
            self.dora_indicators, "dora_indicators"
        )

        if type(self.live_wall_tiles_remaining) is not int:
            raise TypeError("live_wall_tiles_remaining must be an int")
        if self.live_wall_tiles_remaining < 0:
            raise ValueError("live_wall_tiles_remaining must not be negative")

        object.__setattr__(self, "dora_indicators", dora_indicators)
