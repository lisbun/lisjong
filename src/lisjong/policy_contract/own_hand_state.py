"""lisjong内部のOwnHandState型。

docs/policy-input-schema.md「OwnHandState」の意味契約を実装する。

自席だけの非公開情報を、`PlayerPublicState`等の他家公開情報から分離して
保持する。`concealed_tiles`はphysical copy identityを持たず、同じ
semantic Tileの複数枚保持や、`drawn_tile`がどのphysical copyかの区別は
行わない。
"""

from dataclasses import dataclass

from lisjong.policy_contract.tile import Tile, _canonicalize_tile_multiset


@dataclass(frozen=True, slots=True)
class OwnHandState:
    """自席の非公開手牌状態のsnapshot。

    ```text
    OwnHandState
    ├── concealed_tiles
    └── drawn_tile
    ```

    `concealed_tiles`は意味上順序を持たないmultisetであり、生成時に
    canonical tupleへ正規化する（`discards` / `melds` / `dora_indicators`とは
    異なり、正本文書が明示的に順序なしと定義している）。固定枚数制約
    （13/14枚等）や非空制約は課さない。副露数やdecision phaseによる枚数変化は
    Context整合条件として後続のAdapter / PolicyInput境界へ残す。

    `drawn_tile`はconcealed_tiles内のmetadataであり、追加の1枚として
    数えない。`None`でない場合、`concealed_tiles`内に完全なsemantic Tile
    equality（赤牌区分を含む）で存在することを検証する。physical copy
    identityは使用しない。どのphysical copyを引いたかの区別、
    `drawn_tile_index`等は導入しない。
    """

    concealed_tiles: tuple[Tile, ...]
    drawn_tile: Tile | None

    def __post_init__(self) -> None:
        concealed_tiles = _canonicalize_tile_multiset(
            self.concealed_tiles, None, "concealed_tiles"
        )

        if self.drawn_tile is not None:
            if not isinstance(self.drawn_tile, Tile):
                raise TypeError("drawn_tile must be a Tile or None")
            if self.drawn_tile not in concealed_tiles:
                raise ValueError("drawn_tile must be one of concealed_tiles")

        object.__setattr__(self, "concealed_tiles", concealed_tiles)
