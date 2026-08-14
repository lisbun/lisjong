"""lisjong内部の`InternalAction`型。

docs/internal-action-model.mdの11variantと、docs/action-identity.mdの
semantic identity・multiset canonicalizationを実装する。

各variantはactorを含むrequired fieldだけを持つ、frozen dataclassである。
共通base classは作らない。「すべてのvariantがactorを持つ」ことは正本文書の
契約だが、「同じclass hierarchyに属する」ことは契約ではないため、継承を
公開契約であるかのように見せない。variant固有のsemantic fieldに基づく
Python value equality（__eq__ / __hash__）がそのままsemantic identityの
一致判定になる。ただし、Python hash値そのものをidentityの正本にはしない。
hash collisionが起きても、最終的にはvalue equalityで判定される。

各Actionは、Action値不変条件（単一Actionのfieldだけで検証できるもの）だけを
生成時に検証する。Context整合条件（同じdecisionのPolicyInput、
materialized state、legal candidateとの照合が必要なもの）はここでは検証せず、
後続のAdapter / Policy呼び出し境界の責務として残す。

RiichiEnv、RiichiLab、mjai、WebSocket等の外部library固有型、physical tile ID、
game state mutation、hand / river / meld実体へのobject referenceは持たない。
"""

from dataclasses import dataclass

from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, tile_sort_key


def _require_seat(value: object, field_name: str) -> None:
    if not isinstance(value, Seat):
        raise TypeError(f"{field_name} must be a Seat")


def _require_tile(value: object, field_name: str) -> None:
    if not isinstance(value, Tile):
        raise TypeError(f"{field_name} must be a Tile")


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")


def _require_distinct(actor: Seat, other: Seat, other_field_name: str) -> None:
    if actor == other:
        raise ValueError(f"actor must not equal {other_field_name}")


def _kamicha(seat: Seat) -> Seat:
    """seatから見た上家（直前に打牌し、そのdiscardをchiできる相手）を返す。

    Seatの契約 (seat + 1) mod 4 = 下家 から、上家は (seat - 1) mod 4 になる。
    """
    return Seat((int(seat) - 1) % 4)


def _canonicalize_tile_multiset(
    tiles: object, expected_count: int, field_name: str
) -> tuple[Tile, ...]:
    """multiset fieldを、入力順序に依存しないcanonical tupleへ正規化する。

    要素数と型だけを検証し、physical copy identityが存在しないlisjongの
    Tileでは正常な重複（同一semantic Tileの複数枚）を拒否しない。
    """
    try:
        values = tuple(tiles)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable of Tile") from None
    if any(not isinstance(tile, Tile) for tile in values):
        raise TypeError(f"{field_name} must contain only Tile instances")
    if len(values) != expected_count:
        raise ValueError(f"{field_name} must contain exactly {expected_count} tiles")

    return tuple(sorted(values, key=tile_sort_key))


def _require_uniform_tile_type(
    tiles: tuple[Tile, ...], reference: Tile, field_name: str
) -> None:
    if any(tile.tile_type != reference.tile_type for tile in tiles):
        raise ValueError(f"{field_name} must share the same base tile kind")


@dataclass(frozen=True, slots=True)
class DiscardAction:
    """actorがtileを打牌する操作。"""

    actor: Seat
    tile: Tile
    tsumogiri: bool

    def __post_init__(self) -> None:
        _require_seat(self.actor, "actor")
        _require_tile(self.tile, "tile")
        _require_bool(self.tsumogiri, "tsumogiri")


@dataclass(frozen=True, slots=True)
class RiichiAction:
    """actorがリーチ宣言を開始する操作。宣言牌はfieldへ含めない。"""

    actor: Seat

    def __post_init__(self) -> None:
        _require_seat(self.actor, "actor")


@dataclass(frozen=True, slots=True)
class ChiAction:
    """actorがtargetのcalled_tileに対し、consumed_tiles 2枚でchiする操作。"""

    actor: Seat
    target: Seat
    called_tile: Tile
    consumed_tiles: tuple[Tile, Tile]

    def __post_init__(self) -> None:
        _require_seat(self.actor, "actor")
        _require_seat(self.target, "target")
        _require_tile(self.called_tile, "called_tile")
        consumed_tiles = _canonicalize_tile_multiset(
            self.consumed_tiles, 2, "consumed_tiles"
        )

        _require_distinct(self.actor, self.target, "target")
        if self.target != _kamicha(self.actor):
            raise ValueError("target must be actor's kamicha for chi")

        category = self.called_tile.tile_type.category
        if category is TileCategory.HONOR:
            raise ValueError("chi must not use honor tiles")
        if any(tile.tile_type.category is not category for tile in consumed_tiles):
            raise ValueError("chi tiles must be the same suit as called_tile")

        ranks = sorted(
            (
                self.called_tile.tile_type.rank,
                *(tile.tile_type.rank for tile in consumed_tiles),
            )
        )
        if ranks != list(range(ranks[0], ranks[0] + 3)):
            raise ValueError("chi tiles must form three consecutive ranks")

        object.__setattr__(self, "consumed_tiles", consumed_tiles)


@dataclass(frozen=True, slots=True)
class PonAction:
    """actorがtargetのcalled_tileに対し、consumed_tiles 2枚でponする操作。"""

    actor: Seat
    target: Seat
    called_tile: Tile
    consumed_tiles: tuple[Tile, Tile]

    def __post_init__(self) -> None:
        _require_seat(self.actor, "actor")
        _require_seat(self.target, "target")
        _require_tile(self.called_tile, "called_tile")
        consumed_tiles = _canonicalize_tile_multiset(
            self.consumed_tiles, 2, "consumed_tiles"
        )

        _require_distinct(self.actor, self.target, "target")
        _require_uniform_tile_type(consumed_tiles, self.called_tile, "consumed_tiles")

        object.__setattr__(self, "consumed_tiles", consumed_tiles)


@dataclass(frozen=True, slots=True)
class DaiminkanAction:
    """actorがtargetのcalled_tileに対し、consumed_tiles 3枚でdaiminkanする操作。"""

    actor: Seat
    target: Seat
    called_tile: Tile
    consumed_tiles: tuple[Tile, Tile, Tile]

    def __post_init__(self) -> None:
        _require_seat(self.actor, "actor")
        _require_seat(self.target, "target")
        _require_tile(self.called_tile, "called_tile")
        consumed_tiles = _canonicalize_tile_multiset(
            self.consumed_tiles, 3, "consumed_tiles"
        )

        _require_distinct(self.actor, self.target, "target")
        _require_uniform_tile_type(consumed_tiles, self.called_tile, "consumed_tiles")

        object.__setattr__(self, "consumed_tiles", consumed_tiles)


@dataclass(frozen=True, slots=True)
class AnkanAction:
    """actorが自席のtiles 4枚でankanする操作。"""

    actor: Seat
    tiles: tuple[Tile, Tile, Tile, Tile]

    def __post_init__(self) -> None:
        _require_seat(self.actor, "actor")
        tiles = _canonicalize_tile_multiset(self.tiles, 4, "tiles")

        _require_uniform_tile_type(tiles, tiles[0], "tiles")

        object.__setattr__(self, "tiles", tiles)


@dataclass(frozen=True, slots=True)
class KakanAction:
    """actorが、from_seatのcalled_tileから成立した既存Ponへadded_tileを加える操作。

    元Ponはこのfieldへ重複保持しない。source_meld_id、source_meld_index、
    PublicMeld object参照、Python object identityによる元Ponの識別は行わない
    （docs/internal-action-model.md「KakanAction」を参照）。
    """

    actor: Seat
    added_tile: Tile
    from_seat: Seat
    called_tile: Tile

    def __post_init__(self) -> None:
        _require_seat(self.actor, "actor")
        _require_tile(self.added_tile, "added_tile")
        _require_seat(self.from_seat, "from_seat")
        _require_tile(self.called_tile, "called_tile")

        _require_distinct(self.actor, self.from_seat, "from_seat")
        _require_uniform_tile_type((self.added_tile,), self.called_tile, "added_tile")


@dataclass(frozen=True, slots=True)
class RonAction:
    """actorがtargetの牌winning_tileでronする操作。"""

    actor: Seat
    target: Seat
    winning_tile: Tile

    def __post_init__(self) -> None:
        _require_seat(self.actor, "actor")
        _require_seat(self.target, "target")
        _require_tile(self.winning_tile, "winning_tile")

        _require_distinct(self.actor, self.target, "target")


@dataclass(frozen=True, slots=True)
class TsumoAction:
    """actorがwinning_tileでツモする操作。通常ツモ・嶺上ツモ・天和等を含む。"""

    actor: Seat
    winning_tile: Tile

    def __post_init__(self) -> None:
        _require_seat(self.actor, "actor")
        _require_tile(self.winning_tile, "winning_tile")


@dataclass(frozen=True, slots=True)
class PassAction:
    """actorが現在の応答機会で、鳴きや和了を行わないことを明示的に選ぶ操作。"""

    actor: Seat

    def __post_init__(self) -> None:
        _require_seat(self.actor, "actor")


@dataclass(frozen=True, slots=True)
class KyuushuKyuuhaiAction:
    """actorが九種九牌による途中流局を宣言する操作。"""

    actor: Seat

    def __post_init__(self) -> None:
        _require_seat(self.actor, "actor")


type InternalAction = (
    DiscardAction
    | RiichiAction
    | ChiAction
    | PonAction
    | DaiminkanAction
    | AnkanAction
    | KakanAction
    | RonAction
    | TsumoAction
    | PassAction
    | KyuushuKyuuhaiAction
)
