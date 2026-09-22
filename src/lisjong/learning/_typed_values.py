"""policy contract valueのcanonical JSON projectionとvocabulary fingerprint。

Learning pathは、typed valueをPython object、pickle、factory、callable、
`repr()`、generic dataclass serializationとしては保存しない。`Tile` /
`Discard` / `PublicMeld` / `PolicyInput` / `InternalAction`のsemantic fieldだけを
明示的なJSON objectへ射影し、同じ射影でparseし直せることを検証する。

```text
typed policy contract value
    <-> explicit semantic JSON object
```

この射影は`arena-offense-o0-player-safe-source-record-v1`のwire shapeと一致する。
一致は意図的で、consumerはrowをparseしたあと同じ射影へ戻し、byte単位で
round tripすることを要求する（`source_record.py`）。将来source schemaのwire
shapeが変わる場合は、schema versionごとのcodecとして分岐させ、既存versionの
意味を書き換えない。

`vocabulary_fingerprint()`は、この射影を使ってfixed-size action vocabulary
全indexのcanonical descriptionをdigestする。index layout、block順序、field
encodingのいずれが変わってもfingerprintが変わるため、dataset / model artifact
はこの値をbindしてload時に照合できる。
"""

from collections.abc import Sequence
from functools import cache

from lisjong.action_vocabulary import (
    ACTION_VOCABULARY_SIZE,
    ACTION_VOCABULARY_VERSION,
    decode_action,
)
from lisjong.learning._canonical import (
    expect_bool,
    expect_int,
    expect_list,
    expect_object,
    expect_str,
    value_digest,
)
from lisjong.policy_contract import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    Discard,
    DiscardAction,
    InternalAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    MeldKind,
    OwnHandState,
    PassAction,
    PlayerPublicState,
    PolicyInput,
    PonAction,
    PublicMeld,
    RiichiAction,
    RiichiState,
    RonAction,
    RoundState,
    Seat,
    Tile,
    TileCategory,
    TileType,
    TsumoAction,
    Wind,
)

_TILE_FIELDS = frozenset({"category", "is_red", "rank"})
_DISCARD_FIELDS = frozenset({"called_by", "order", "tile", "tsumogiri"})
_MELD_FIELDS = frozenset({"called_tile", "from_seat", "kind", "tiles"})
_OWN_HAND_FIELDS = frozenset({"concealed_tiles", "drawn_tile"})
_PLAYER_FIELDS = frozenset({"discards", "melds", "riichi", "score"})
_ROUND_FIELDS = frozenset(
    {
        "dealer_seat",
        "dora_indicators",
        "hand_number",
        "honba",
        "live_wall_tiles_remaining",
        "riichi_sticks",
        "round_wind",
    }
)
_POLICY_INPUT_FIELDS = frozenset({"own_hand", "players", "round", "self_seat"})

_ACTION_KINDS: dict[type, str] = {
    DiscardAction: "discard",
    RiichiAction: "riichi",
    ChiAction: "chi",
    PonAction: "pon",
    DaiminkanAction: "daiminkan",
    AnkanAction: "ankan",
    KakanAction: "kakan",
    RonAction: "ron",
    TsumoAction: "tsumo",
    PassAction: "pass",
    KyuushuKyuuhaiAction: "kyuushu-kyuuhai",
}
_KIND_ACTIONS = {kind: variant for variant, kind in _ACTION_KINDS.items()}
_ACTION_FIELDS: dict[str, frozenset[str]] = {
    "discard": frozenset({"actor", "kind", "tile", "tsumogiri"}),
    "riichi": frozenset({"actor", "kind"}),
    "chi": frozenset({"actor", "kind", "target", "called_tile", "consumed_tiles"}),
    "pon": frozenset({"actor", "kind", "target", "called_tile", "consumed_tiles"}),
    "daiminkan": frozenset(
        {"actor", "kind", "target", "called_tile", "consumed_tiles"}
    ),
    "ankan": frozenset({"actor", "kind", "tiles"}),
    "kakan": frozenset({"actor", "kind", "added_tile", "from_seat", "called_tile"}),
    "ron": frozenset({"actor", "kind", "target", "winning_tile"}),
    "tsumo": frozenset({"actor", "kind", "winning_tile"}),
    "pass": frozenset({"actor", "kind"}),
    "kyuushu-kyuuhai": frozenset({"actor", "kind"}),
}

_FINGERPRINT_ACTOR = Seat.SEAT_0


def _construct(factory, error: type[Exception], context: str, **values):
    """value型のconstructor例外を、呼び出し側のfail-closed例外へ翻訳する。"""
    try:
        return factory(**values)
    except (TypeError, ValueError) as exc:
        raise error(f"{context} is invalid: {exc}") from exc


def _parse_enum(enum_type: type, value: object, error: type[Exception], context: str):
    raw = expect_str(value, error, context)
    try:
        return enum_type(raw)
    except ValueError:
        raise error(f"{context} has an unsupported value: {raw!r}") from None


def _parse_seat(value: object, error: type[Exception], context: str) -> Seat:
    raw = expect_int(value, error, context)
    try:
        return Seat(raw)
    except ValueError:
        raise error(f"{context} is not a valid seat") from None


def tile_to_value(tile: object, error: type[Exception], context: str) -> dict:
    if not isinstance(tile, Tile):
        raise error(f"{context} must be a Tile")
    return {
        "category": tile.tile_type.category.value,
        "is_red": tile.is_red,
        "rank": tile.tile_type.rank,
    }


def parse_tile(value: object, error: type[Exception], context: str) -> Tile:
    raw = expect_object(value, _TILE_FIELDS, error, context)
    tile_type = _construct(
        TileType,
        error,
        f"{context}.tile_type",
        category=_parse_enum(
            TileCategory, raw["category"], error, f"{context}.category"
        ),
        rank=expect_int(raw["rank"], error, f"{context}.rank"),
    )
    return _construct(
        Tile,
        error,
        context,
        tile_type=tile_type,
        is_red=expect_bool(raw["is_red"], error, f"{context}.is_red"),
    )


def tiles_to_value(
    tiles: Sequence[Tile], error: type[Exception], context: str
) -> list[dict]:
    return [
        tile_to_value(tile, error, f"{context}[{index}]")
        for index, tile in enumerate(tiles)
    ]


def parse_tiles(
    value: object, error: type[Exception], context: str
) -> tuple[Tile, ...]:
    return tuple(
        parse_tile(item, error, f"{context}[{index}]")
        for index, item in enumerate(expect_list(value, error, context))
    )


def discard_to_value(discard: object, error: type[Exception], context: str) -> dict:
    if not isinstance(discard, Discard):
        raise error(f"{context} must be a Discard")
    return {
        "called_by": None if discard.called_by is None else int(discard.called_by),
        "order": discard.order,
        "tile": tile_to_value(discard.tile, error, f"{context}.tile"),
        "tsumogiri": discard.tsumogiri,
    }


def parse_discard(value: object, error: type[Exception], context: str) -> Discard:
    raw = expect_object(value, _DISCARD_FIELDS, error, context)
    return _construct(
        Discard,
        error,
        context,
        tile=parse_tile(raw["tile"], error, f"{context}.tile"),
        tsumogiri=expect_bool(raw["tsumogiri"], error, f"{context}.tsumogiri"),
        order=expect_int(raw["order"], error, f"{context}.order"),
        called_by=None
        if raw["called_by"] is None
        else _parse_seat(raw["called_by"], error, f"{context}.called_by"),
    )


def meld_to_value(meld: object, error: type[Exception], context: str) -> dict:
    if not isinstance(meld, PublicMeld):
        raise error(f"{context} must be a PublicMeld")
    return {
        "called_tile": None
        if meld.called_tile is None
        else tile_to_value(meld.called_tile, error, f"{context}.called_tile"),
        "from_seat": None if meld.from_seat is None else int(meld.from_seat),
        "kind": meld.kind.value,
        "tiles": tiles_to_value(meld.tiles, error, f"{context}.tiles"),
    }


def parse_meld(value: object, error: type[Exception], context: str) -> PublicMeld:
    raw = expect_object(value, _MELD_FIELDS, error, context)
    return _construct(
        PublicMeld,
        error,
        context,
        kind=_parse_enum(MeldKind, raw["kind"], error, f"{context}.kind"),
        tiles=parse_tiles(raw["tiles"], error, f"{context}.tiles"),
        from_seat=None
        if raw["from_seat"] is None
        else _parse_seat(raw["from_seat"], error, f"{context}.from_seat"),
        called_tile=None
        if raw["called_tile"] is None
        else parse_tile(raw["called_tile"], error, f"{context}.called_tile"),
    )


def policy_input_to_value(
    policy_input: object, error: type[Exception], context: str
) -> dict:
    """player-safe PolicyInputをcanonical JSON objectへ射影する。

    `PolicyInput`自体が当該seatの観測可能範囲しか持たないため、この射影は
    privileged情報を追加しない。逆にfieldを落とすこともせず、round trip
    可能な全semantic fieldだけを含める。
    """
    if not isinstance(policy_input, PolicyInput):
        raise error(f"{context} must be a PolicyInput")
    round_state = policy_input.round
    return {
        "own_hand": {
            "concealed_tiles": tiles_to_value(
                policy_input.own_hand.concealed_tiles,
                error,
                f"{context}.own_hand.concealed_tiles",
            ),
            "drawn_tile": None
            if policy_input.own_hand.drawn_tile is None
            else tile_to_value(
                policy_input.own_hand.drawn_tile,
                error,
                f"{context}.own_hand.drawn_tile",
            ),
        },
        "players": [
            {
                "discards": [
                    discard_to_value(
                        item, error, f"{context}.players[{seat}].discards[{index}]"
                    )
                    for index, item in enumerate(player.discards)
                ],
                "melds": [
                    meld_to_value(
                        item, error, f"{context}.players[{seat}].melds[{index}]"
                    )
                    for index, item in enumerate(player.melds)
                ],
                "riichi": player.riichi.value,
                "score": player.score,
            }
            for seat, player in enumerate(policy_input.players)
        ],
        "round": {
            "dealer_seat": int(round_state.dealer_seat),
            "dora_indicators": tiles_to_value(
                round_state.dora_indicators, error, f"{context}.round.dora_indicators"
            ),
            "hand_number": round_state.hand_number,
            "honba": round_state.honba,
            "live_wall_tiles_remaining": round_state.live_wall_tiles_remaining,
            "riichi_sticks": round_state.riichi_sticks,
            "round_wind": round_state.round_wind.value,
        },
        "self_seat": int(policy_input.self_seat),
    }


def parse_policy_input(
    value: object, error: type[Exception], context: str
) -> PolicyInput:
    raw = expect_object(value, _POLICY_INPUT_FIELDS, error, context)
    round_raw = expect_object(raw["round"], _ROUND_FIELDS, error, f"{context}.round")
    own_raw = expect_object(
        raw["own_hand"], _OWN_HAND_FIELDS, error, f"{context}.own_hand"
    )
    players_raw = expect_list(raw["players"], error, f"{context}.players")
    players = []
    for index, item in enumerate(players_raw):
        player_context = f"{context}.players[{index}]"
        player = expect_object(item, _PLAYER_FIELDS, error, player_context)
        players.append(
            _construct(
                PlayerPublicState,
                error,
                player_context,
                score=expect_int(player["score"], error, f"{player_context}.score"),
                discards=tuple(
                    parse_discard(row, error, f"{player_context}.discards[{row_index}]")
                    for row_index, row in enumerate(
                        expect_list(
                            player["discards"], error, f"{player_context}.discards"
                        )
                    )
                ),
                melds=tuple(
                    parse_meld(row, error, f"{player_context}.melds[{row_index}]")
                    for row_index, row in enumerate(
                        expect_list(player["melds"], error, f"{player_context}.melds")
                    )
                ),
                riichi=_parse_enum(
                    RiichiState, player["riichi"], error, f"{player_context}.riichi"
                ),
            )
        )
    drawn = own_raw["drawn_tile"]
    return _construct(
        PolicyInput,
        error,
        context,
        self_seat=_parse_seat(raw["self_seat"], error, f"{context}.self_seat"),
        round=_construct(
            RoundState,
            error,
            f"{context}.round",
            round_wind=_parse_enum(
                Wind, round_raw["round_wind"], error, f"{context}.round.round_wind"
            ),
            hand_number=expect_int(
                round_raw["hand_number"], error, f"{context}.round.hand_number"
            ),
            dealer_seat=_parse_seat(
                round_raw["dealer_seat"], error, f"{context}.round.dealer_seat"
            ),
            honba=expect_int(round_raw["honba"], error, f"{context}.round.honba"),
            riichi_sticks=expect_int(
                round_raw["riichi_sticks"], error, f"{context}.round.riichi_sticks"
            ),
            dora_indicators=parse_tiles(
                round_raw["dora_indicators"],
                error,
                f"{context}.round.dora_indicators",
            ),
            live_wall_tiles_remaining=expect_int(
                round_raw["live_wall_tiles_remaining"],
                error,
                f"{context}.round.live_wall_tiles_remaining",
            ),
        ),
        players=tuple(players),
        own_hand=_construct(
            OwnHandState,
            error,
            f"{context}.own_hand",
            concealed_tiles=parse_tiles(
                own_raw["concealed_tiles"],
                error,
                f"{context}.own_hand.concealed_tiles",
            ),
            drawn_tile=None
            if drawn is None
            else parse_tile(drawn, error, f"{context}.own_hand.drawn_tile"),
        ),
    )


def action_to_value(action: object, error: type[Exception], context: str) -> dict:
    """InternalActionをcanonical JSON objectへ射影する。

    variantはexact typeで判定する。subclassはbase variantのsemantic fieldしか
    射影できず、round tripでsemantic distinctionを失うため拒否する。
    """
    kind = _ACTION_KINDS.get(type(action))
    if kind is None:
        raise error(
            f"{context} must be exactly one of the 11 InternalAction variants; "
            f"got {type(action).__name__} instead"
        )
    value: dict[str, object] = {"actor": int(action.actor), "kind": kind}
    if kind == "discard":
        value.update(
            tile=tile_to_value(action.tile, error, f"{context}.tile"),
            tsumogiri=action.tsumogiri,
        )
    elif kind in ("chi", "pon", "daiminkan"):
        value.update(
            target=int(action.target),
            called_tile=tile_to_value(
                action.called_tile, error, f"{context}.called_tile"
            ),
            consumed_tiles=tiles_to_value(
                action.consumed_tiles, error, f"{context}.consumed_tiles"
            ),
        )
    elif kind == "ankan":
        value["tiles"] = tiles_to_value(action.tiles, error, f"{context}.tiles")
    elif kind == "kakan":
        value.update(
            added_tile=tile_to_value(action.added_tile, error, f"{context}.added_tile"),
            from_seat=int(action.from_seat),
            called_tile=tile_to_value(
                action.called_tile, error, f"{context}.called_tile"
            ),
        )
    elif kind == "ron":
        value.update(
            target=int(action.target),
            winning_tile=tile_to_value(
                action.winning_tile, error, f"{context}.winning_tile"
            ),
        )
    elif kind == "tsumo":
        value["winning_tile"] = tile_to_value(
            action.winning_tile, error, f"{context}.winning_tile"
        )
    return value


def parse_action(value: object, error: type[Exception], context: str) -> InternalAction:
    if type(value) is not dict or "kind" not in value:
        raise error(f"{context} must be a JSON object with a kind")
    kind = expect_str(value["kind"], error, f"{context}.kind")
    variant = _KIND_ACTIONS.get(kind)
    if variant is None:
        raise error(f"{context} has an unsupported action kind: {kind!r}")
    raw = expect_object(value, _ACTION_FIELDS[kind], error, context)
    values: dict[str, object] = {
        "actor": _parse_seat(raw["actor"], error, f"{context}.actor")
    }
    if kind == "discard":
        values.update(
            tile=parse_tile(raw["tile"], error, f"{context}.tile"),
            tsumogiri=expect_bool(raw["tsumogiri"], error, f"{context}.tsumogiri"),
        )
    elif kind in ("chi", "pon", "daiminkan"):
        values.update(
            target=_parse_seat(raw["target"], error, f"{context}.target"),
            called_tile=parse_tile(raw["called_tile"], error, f"{context}.called_tile"),
            consumed_tiles=parse_tiles(
                raw["consumed_tiles"], error, f"{context}.consumed_tiles"
            ),
        )
    elif kind == "ankan":
        values["tiles"] = parse_tiles(raw["tiles"], error, f"{context}.tiles")
    elif kind == "kakan":
        values.update(
            added_tile=parse_tile(raw["added_tile"], error, f"{context}.added_tile"),
            from_seat=_parse_seat(raw["from_seat"], error, f"{context}.from_seat"),
            called_tile=parse_tile(raw["called_tile"], error, f"{context}.called_tile"),
        )
    elif kind == "ron":
        values.update(
            target=_parse_seat(raw["target"], error, f"{context}.target"),
            winning_tile=parse_tile(
                raw["winning_tile"], error, f"{context}.winning_tile"
            ),
        )
    elif kind == "tsumo":
        values["winning_tile"] = parse_tile(
            raw["winning_tile"], error, f"{context}.winning_tile"
        )
    return _construct(variant, error, context, **values)


@cache
def vocabulary_fingerprint() -> str:
    """fixed-size action vocabulary layoutのfingerprintを返す。

    vocabulary version、size、および全indexのcanonical action descriptionを
    digestする。indexとsemantic actionの対応が変われば必ず値が変わるため、
    dataset / model artifactはこのfingerprintをbindしてload時に照合できる。

    decodeのactorは固定seatであり、fingerprintはactor絶対値の選択に依存する
    ことなくindex layoutの同一性を表す（vocabularyはactorを持たず、
    `target` / `from_seat`はactor相対位置としてencodeされる）。
    """
    description = {
        "action_vocabulary_version": ACTION_VOCABULARY_VERSION,
        "actions": [
            action_to_value(
                decode_action(index, _FINGERPRINT_ACTOR),
                ValueError,
                f"vocabulary[{index}]",
            )
            for index in range(ACTION_VOCABULARY_SIZE)
        ],
        "fingerprint_actor": int(_FINGERPRINT_ACTOR),
        "size": ACTION_VOCABULARY_SIZE,
    }
    return value_digest(description)
