"""fixed-sizeかつversionedなmodel-facing action vocabularyとcodec。

docs/action-vocabulary.mdの意味契約を実装する。

learned Policyは固定長のaction出力を持つため、`InternalAction`のsemantic
identityと、固定長vector上のnumeric indexを対応付ける表現が必要になる。本module
はその対応付けだけを所有する。

```text
semantic identity
    = InternalAction dataclass value equality

model action index
    = versioned adapter representation
```

model action indexは新しいAction identityではない。合法性の根拠でもなく、
`DecisionContext.legal_actions`のtuple indexでもない。indexからActionへ戻す際は、
同じdecisionのcanonical legal Actionを`legal_mask.py`側で解決する
（docs/action-identity.md「Model-facing action vocabulary」を参照）。

vocabularyは`InternalAction`の値空間全体を損失なく表現する。すなわち、Action値
不変条件を満たす任意のInternalActionは、`actor`を除くすべてのsemantic fieldを
保ったままちょうど1つのindexへencodeでき、同じ`actor`の下でindexから同じ値へ
decodeできる。`actor`だけはvocabularyへ持ち込まず、decode時のcontextから復元する
（`DecisionContext`のlegal actionsはすべて`input.self_seat`をactorとする）。

RiichiEnv、RiichiLab、mjai等の外部engine固有action表現、physical tile ID、
hidden information、ML runtime（NumPy / PyTorch等）へは依存しない。
"""

from collections.abc import Mapping
from types import MappingProxyType

from lisjong.action_vocabulary.errors import (
    ActionEncodingError,
    ActionIndexError,
    ActionVocabularyError,
    UnsupportedActionVocabularyVersionError,
)
from lisjong.policy_contract.action import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DiscardAction,
    InternalAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType, tile_sort_key

ACTION_VOCABULARY_VERSION = "lisjong-action-vocabulary-1"
"""vocabularyの意味とnumeric assignmentを識別するversion。

index layout、block順序、block内の列挙順、fieldのencoding規則、vocabulary size、
またはvariant集合を変更する場合は、この文字列も更新する。後続のmodel artifactは
weightsと同じ場所へこのversionを記録し、読み込み時に照合する。
"""

_SUIT_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)
_SUIT_ORDER: Mapping[TileCategory, int] = MappingProxyType(
    {category: order for order, category in enumerate(_SUIT_CATEGORIES)}
)

# 相対seat。actorから見た下家 / 対面 / 上家であり、absolute seatを
# vocabularyへ持ち込まない。actor自身(0)はどのvariantでも合法でない。
_RELATIVE_SHIMOCHA = 1
_RELATIVE_TOIMEN = 2
_RELATIVE_KAMICHA = 3
_RELATIVE_SEATS = (_RELATIVE_SHIMOCHA, _RELATIVE_TOIMEN, _RELATIVE_KAMICHA)


def _is_red_capable(tile_type: TileType) -> bool:
    """その基礎牌種に赤牌が存在し得るかを返す（Tile不変条件と同じ判定）。"""
    return tile_type.category is not TileCategory.HONOR and tile_type.rank == 5


def _build_base_tile_types() -> tuple[TileType, ...]:
    """34種の基礎牌種を、tile_sort_keyと同じcategory → rank順で返す。"""
    types = [
        TileType(category, rank)
        for category in _SUIT_CATEGORIES
        for rank in range(1, 10)
    ]
    types.extend(TileType(TileCategory.HONOR, rank) for rank in range(1, 8))
    return tuple(types)


_BASE_TILE_TYPES = _build_base_tile_types()
_BASE_TILE_TYPE_TO_INDEX: Mapping[TileType, int] = MappingProxyType(
    {tile_type: index for index, tile_type in enumerate(_BASE_TILE_TYPES)}
)


def _build_vocabulary_tiles() -> tuple[Tile, ...]:
    """37種のTile値（34基礎牌種 + 赤5m/5p/5s）をcanonical順で返す。

    順序は`lisjong.policy_contract.tile.tile_sort_key`と一致させる。
    vocabulary側で独自のtile順序を発明しない。
    """
    tiles: list[Tile] = []
    for tile_type in _BASE_TILE_TYPES:
        tiles.append(Tile(tile_type))
        if _is_red_capable(tile_type):
            tiles.append(Tile(tile_type, is_red=True))
    return tuple(sorted(tiles, key=tile_sort_key))


_VOCABULARY_TILES = _build_vocabulary_tiles()
_TILE_TO_INDEX: Mapping[Tile, int] = MappingProxyType(
    {tile: index for index, tile in enumerate(_VOCABULARY_TILES)}
)

# vocabulary keyのvariant tagにはAction dataclass型そのものを使う。文字列kindや
# ActionKind enumを新設すると、docs/action-identity.mdが禁じる「別のcanonical key /
# action ID」の外観を持つ表現をcontract側へ増やすことになるため採用しない。
type _VocabularyKey = tuple[object, ...]
"""vocabulary index 1件分のcanonical key。先頭要素はAction variantの型である。"""


def _red_flags(tile_type: TileType) -> tuple[bool, ...]:
    """単一牌の赤牌区分として到達可能な値だけを返す。"""
    return (False, True) if _is_red_capable(tile_type) else (False,)


def _red_counts(tile_type: TileType, tile_count: int) -> tuple[int, ...]:
    """multiset内の赤牌枚数として到達可能な値だけを返す。"""
    if not _is_red_capable(tile_type):
        return (0,)
    return tuple(range(tile_count + 1))


def _chi_contains_five(low_rank: int) -> bool:
    return low_rank <= 5 <= low_rank + 2


def _build_vocabulary_keys() -> tuple[_VocabularyKey, ...]:
    """vocabulary index順のcanonical key列を構築する。

    variantごとに連続したblockを割り当て、block内は下のnested loop順で
    列挙する。到達不能な組み合わせ（赤牌になり得ない牌種の赤牌flag等）は
    生成しないため、range内にholeやunreachable indexを作らない。
    """
    keys: list[_VocabularyKey] = []

    # Discard: tile(37) x tsumogiri(2)
    for tile_index in range(len(_VOCABULARY_TILES)):
        for tsumogiri in (False, True):
            keys.append((DiscardAction, tile_index, tsumogiri))

    # Riichi: 宣言牌を持たない単一index
    keys.append((RiichiAction,))

    # Chi: suit(3) x 順子の最小rank(1..7) x called位置(0..2) x 5の赤牌区分
    for suit_order in range(len(_SUIT_CATEGORIES)):
        for low_rank in range(1, 8):
            for called_offset in range(3):
                red_flags = (False, True) if _chi_contains_five(low_rank) else (False,)
                for red_five in red_flags:
                    keys.append(
                        (ChiAction, suit_order, low_rank, called_offset, red_five)
                    )

    # Pon / Daiminkan: 相対target(3) x 基礎牌種(34) x called赤 x consumed赤枚数
    for variant, consumed_count in ((PonAction, 2), (DaiminkanAction, 3)):
        for relative_target in _RELATIVE_SEATS:
            for base_index, tile_type in enumerate(_BASE_TILE_TYPES):
                for called_red in _red_flags(tile_type):
                    for consumed_reds in _red_counts(tile_type, consumed_count):
                        keys.append(
                            (
                                variant,
                                relative_target,
                                base_index,
                                called_red,
                                consumed_reds,
                            )
                        )

    # Ankan: 基礎牌種(34) x 4枚中の赤牌枚数
    for base_index, tile_type in enumerate(_BASE_TILE_TYPES):
        for red_count in _red_counts(tile_type, 4):
            keys.append((AnkanAction, base_index, red_count))

    # Kakan: 相対from_seat(3) x 基礎牌種(34) x added赤 x called赤
    for relative_from_seat in _RELATIVE_SEATS:
        for base_index, tile_type in enumerate(_BASE_TILE_TYPES):
            for added_red in _red_flags(tile_type):
                for called_red in _red_flags(tile_type):
                    keys.append(
                        (
                            KakanAction,
                            relative_from_seat,
                            base_index,
                            added_red,
                            called_red,
                        )
                    )

    # Ron: 相対target(3) x winning tile(37)
    for relative_target in _RELATIVE_SEATS:
        for tile_index in range(len(_VOCABULARY_TILES)):
            keys.append((RonAction, relative_target, tile_index))

    # Tsumo: winning tile(37)
    for tile_index in range(len(_VOCABULARY_TILES)):
        keys.append((TsumoAction, tile_index))

    keys.append((PassAction,))
    keys.append((KyuushuKyuuhaiAction,))

    return tuple(keys)


_INDEX_TO_KEY = _build_vocabulary_keys()
_KEY_TO_INDEX: Mapping[_VocabularyKey, int] = MappingProxyType(
    {key: index for index, key in enumerate(_INDEX_TO_KEY)}
)

if len(_KEY_TO_INDEX) != len(_INDEX_TO_KEY):
    # vocabulary定義の編集で2つのindexが同じ意味へaliasした場合、encodeが
    # silentに情報を失う。import時点でfail closedする。
    raise ActionVocabularyError("action vocabulary keys must be unique")

ACTION_VOCABULARY_SIZE = len(_INDEX_TO_KEY)
"""fixed-size action vocabularyの総index数。decisionによって変化しない。"""


def _build_vocabulary_blocks() -> Mapping[type, range]:
    """variantごとの連続index rangeを返す。"""
    bounds: dict[type, list[int]] = {}
    for index, key in enumerate(_INDEX_TO_KEY):
        variant = key[0]
        if not isinstance(variant, type):  # pragma: no cover - 構築側の防御
            raise ActionVocabularyError("vocabulary key must start with a variant type")
        block = bounds.get(variant)
        if block is None:
            bounds[variant] = [index, index + 1]
        else:
            block[1] = index + 1
    return MappingProxyType(
        {variant: range(start, stop) for variant, (start, stop) in bounds.items()}
    )


ACTION_VOCABULARY_BLOCKS: Mapping[type, range] = _build_vocabulary_blocks()
"""Action variantごとのindex range。read-onlyなlayout記述であり、identityではない。"""


def _require_supported_version(version: object) -> None:
    """未対応versionをfail closedで拒否する。"""
    if version != ACTION_VOCABULARY_VERSION:
        raise UnsupportedActionVocabularyVersionError(
            f"unsupported action vocabulary version: {version!r}; "
            f"this implementation provides {ACTION_VOCABULARY_VERSION!r}"
        )


def _require_vocabulary_index(index: object) -> int:
    """vocabulary index範囲内のintであることを検証する。"""
    if type(index) is not int:
        raise TypeError("index must be an int")
    if not 0 <= index < ACTION_VOCABULARY_SIZE:
        raise ActionIndexError(
            f"index must be in range(0, {ACTION_VOCABULARY_SIZE}); got {index}"
        )
    return index


def _require_actor(actor: object) -> Seat:
    if not isinstance(actor, Seat):
        raise TypeError("actor must be a Seat")
    return actor


def _relative_seat(actor: Seat, other: Seat) -> int:
    """actorから見たotherの相対位置(1=下家, 2=対面, 3=上家)を返す。"""
    return (int(other) - int(actor)) % 4


def _seat_from_relative(actor: Seat, relative: int) -> Seat:
    return Seat((int(actor) + relative) % 4)


def _tile_index(tile: Tile) -> int:
    index = _TILE_TO_INDEX.get(tile)
    if index is None:
        raise ActionEncodingError(f"tile is not part of the vocabulary: {tile!r}")
    return index


def _base_tile_type_index(tile_type: TileType) -> int:
    index = _BASE_TILE_TYPE_TO_INDEX.get(tile_type)
    if index is None:
        raise ActionEncodingError(
            f"tile type is not part of the vocabulary: {tile_type!r}"
        )
    return index


def _chi_key(action: ChiAction) -> _VocabularyKey:
    """Chiを(suit, 最小rank, called位置, 5の赤牌区分)へ正規化する。

    順子は3つの連続する異なるrankから成るため、5は高々1枚しか含まれない。
    したがって赤牌構成は1 bitで損失なく表現でき、その5がcalled tile側か
    consumed側かはcalled位置から一意に決まる。
    """
    tiles = (action.called_tile, *action.consumed_tiles)
    suit_order = _SUIT_ORDER.get(action.called_tile.tile_type.category)
    if suit_order is None:
        raise ActionEncodingError("chi called_tile must be a suited tile")

    ranks = sorted(tile.tile_type.rank for tile in tiles)
    low_rank = ranks[0]
    called_offset = action.called_tile.tile_type.rank - low_rank
    red_five = any(tile.tile_type.rank == 5 and tile.is_red for tile in tiles)
    return (ChiAction, suit_order, low_rank, called_offset, red_five)


def _called_meld_key(
    variant: type, action: PonAction | DaiminkanAction
) -> _VocabularyKey:
    """Pon / Daiminkanを(相対target, 基礎牌種, called赤, consumed赤枚数)へ正規化する。

    called tileとconsumed tilesは同じ基礎牌種であることがAction値不変条件なので、
    牌種は1つで足りる。consumed multisetは順序を持たず、赤牌枚数だけが
    semantic distinctionを構成する。
    """
    return (
        variant,
        _relative_seat(action.actor, action.target),
        _base_tile_type_index(action.called_tile.tile_type),
        action.called_tile.is_red,
        sum(1 for tile in action.consumed_tiles if tile.is_red),
    )


def _action_key(action: object) -> _VocabularyKey:
    """InternalActionをvocabulary keyへ正規化する。actorは含めない。"""
    match action:
        case DiscardAction():
            return (DiscardAction, _tile_index(action.tile), action.tsumogiri)
        case RiichiAction():
            return (RiichiAction,)
        case ChiAction():
            return _chi_key(action)
        case PonAction():
            return _called_meld_key(PonAction, action)
        case DaiminkanAction():
            return _called_meld_key(DaiminkanAction, action)
        case AnkanAction():
            return (
                AnkanAction,
                _base_tile_type_index(action.tiles[0].tile_type),
                sum(1 for tile in action.tiles if tile.is_red),
            )
        case KakanAction():
            return (
                KakanAction,
                _relative_seat(action.actor, action.from_seat),
                _base_tile_type_index(action.called_tile.tile_type),
                action.added_tile.is_red,
                action.called_tile.is_red,
            )
        case RonAction():
            return (
                RonAction,
                _relative_seat(action.actor, action.target),
                _tile_index(action.winning_tile),
            )
        case TsumoAction():
            return (TsumoAction, _tile_index(action.winning_tile))
        case PassAction():
            return (PassAction,)
        case KyuushuKyuuhaiAction():
            return (KyuushuKyuuhaiAction,)
        case _:
            raise ActionEncodingError(
                f"action must be an InternalAction; got {type(action).__name__} instead"
            )


def _red_composition(
    tile_type: TileType, tile_count: int, red_count: int
) -> tuple[Tile, ...]:
    """同一牌種のmultisetを、通常牌 → 赤牌のcanonical順で構築する。"""
    normal = tuple(Tile(tile_type) for _ in range(tile_count - red_count))
    red = tuple(Tile(tile_type, is_red=True) for _ in range(red_count))
    return normal + red


def _action_from_key(key: _VocabularyKey, actor: Seat) -> InternalAction:
    """vocabulary keyとactorからInternalActionを再構築する。"""
    variant = key[0]

    if variant is DiscardAction:
        _, tile_index, tsumogiri = key
        return DiscardAction(
            actor=actor,
            tile=_VOCABULARY_TILES[tile_index],
            tsumogiri=tsumogiri,
        )
    if variant is RiichiAction:
        return RiichiAction(actor=actor)
    if variant is ChiAction:
        _, suit_order, low_rank, called_offset, red_five = key
        category = _SUIT_CATEGORIES[suit_order]
        run = tuple(
            Tile(TileType(category, rank), is_red=(red_five and rank == 5))
            for rank in range(low_rank, low_rank + 3)
        )
        return ChiAction(
            actor=actor,
            target=_seat_from_relative(actor, _RELATIVE_KAMICHA),
            called_tile=run[called_offset],
            consumed_tiles=tuple(
                tile for offset, tile in enumerate(run) if offset != called_offset
            ),
        )
    if variant in (PonAction, DaiminkanAction):
        _, relative_target, base_index, called_red, consumed_reds = key
        tile_type = _BASE_TILE_TYPES[base_index]
        consumed_count = 2 if variant is PonAction else 3
        return variant(
            actor=actor,
            target=_seat_from_relative(actor, relative_target),
            called_tile=Tile(tile_type, is_red=called_red),
            consumed_tiles=_red_composition(tile_type, consumed_count, consumed_reds),
        )
    if variant is AnkanAction:
        _, base_index, red_count = key
        return AnkanAction(
            actor=actor,
            tiles=_red_composition(_BASE_TILE_TYPES[base_index], 4, red_count),
        )
    if variant is KakanAction:
        _, relative_from_seat, base_index, added_red, called_red = key
        tile_type = _BASE_TILE_TYPES[base_index]
        return KakanAction(
            actor=actor,
            added_tile=Tile(tile_type, is_red=added_red),
            from_seat=_seat_from_relative(actor, relative_from_seat),
            called_tile=Tile(tile_type, is_red=called_red),
        )
    if variant is RonAction:
        _, relative_target, tile_index = key
        return RonAction(
            actor=actor,
            target=_seat_from_relative(actor, relative_target),
            winning_tile=_VOCABULARY_TILES[tile_index],
        )
    if variant is TsumoAction:
        _, tile_index = key
        return TsumoAction(actor=actor, winning_tile=_VOCABULARY_TILES[tile_index])
    if variant is PassAction:
        return PassAction(actor=actor)
    if variant is KyuushuKyuuhaiAction:
        return KyuushuKyuuhaiAction(actor=actor)

    raise ActionIndexError(  # pragma: no cover - 構築側の防御
        f"vocabulary key has no decoder: {key!r}"
    )


def encode_action(
    action: InternalAction, *, version: str = ACTION_VOCABULARY_VERSION
) -> int:
    """InternalActionをmodel action indexへencodeする。

    `actor`はvocabularyへ含めないため、同じsemantic操作は`actor`によらず同じ
    indexになる。`target` / `from_seat`はactor相対位置として、tileはTile
    identity（基礎牌種 + 赤牌区分）としてencodeする。

    encodeできないvalueはfail closedとし、近いindexやfallbackへ丸めない。
    """
    _require_supported_version(version)

    key = _action_key(action)
    index = _KEY_TO_INDEX.get(key)
    if index is None:
        raise ActionEncodingError(f"action cannot be encoded losslessly: {action!r}")
    return index


def decode_action(
    index: int, actor: Seat, *, version: str = ACTION_VOCABULARY_VERSION
) -> InternalAction:
    """model action indexと`actor`からInternalActionを再構築する。

    これはvocabulary上の意味を復元するだけであり、合法性の主張ではない。
    Policy decisionからActionを得る経路では、canonicalな合法候補を返す
    `resolve_legal_action()`を使用する。
    """
    _require_supported_version(version)
    _require_actor(actor)
    _require_vocabulary_index(index)

    return _action_from_key(_INDEX_TO_KEY[index], actor)
