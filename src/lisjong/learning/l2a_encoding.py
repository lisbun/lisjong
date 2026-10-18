"""L2aの候補encoding：打牌M0 / M1と反応R1（Issue #316）。

設計の正本は`docs/learning-l2a.md`の2.6節・4.2節である。datasetとservingは、
このmoduleの`L2aEncoding.encode()`だけを使う（train-serve一致）。

```text
DecisionContext（分類はlisjong.learning.l2a_decision）
    DISCARD -> M0  L0.2の打牌候補encoding（無変更・同じidentity）
            -> M1  M0 + player-safe解析値block（下のM1_BLOCKS）
    CALL    -> R1  Pass・Chi・Ponの反応候補encoding（下のR1_BLOCKS）
    -> (canonical順の候補action, 候補ごとのfloat vector)
```

入力は`DecisionContext`だけである。解析値は`PolicyInput`から決定的に計算できる
ものに限り、HandBelief・他家の伏せ牌・wall・C0の内部値（`GameSituationMode`、
危険度の順位、選択branch）は使わない。concrete Policy moduleのprivate helperは
importせず、役のroute・喰い替え・ドラの規則はこのmoduleが独自に持つ。

## M1の追加block（M0の後ろへ連結）

```text
wait_shape_support      6  単騎 / シャンポン / 嵌張 / 辺張 / 両面低 / 両面高が、
                           観測者から見た残り枚数で成立し得るか（wait_shape_support）
remaining_same_tile     1  打牌牌種の残り枚数 / 4
opponent_<k>            5  k = 下家 / 対面 / 上家。リーチ（宣言を含む）、
  x 3                      河に同じ牌種（現物）、リーチ後の通過牌（#302の確定安全牌）、
                           両面低の反対側（t+3）が河、両面高の反対側（t-3）が河
safe_against_all_riichi 1  リーチ者が1人以上いて、全リーチ者に確定安全
discard_dora            1  打牌牌種を指すドラ表示牌の数 / 4
retained_dora           1  打牌後の純手牌のドラ枚数（表示牌の重複を数える）/ 8
retained_red            1  打牌後の純手牌の赤5枚数 / 3
tile_class              3  么九（数牌の1・9）、字牌、役牌の翻数（連風牌は2）/ 2
open_route_<r>          2  r = Tanyao / Honitsu萬 / 筒 / 索。副露時（暗槓以外の副露あり）
  x 4                      かつ既存の副露がrouteと矛盾しないときmask 1と打牌後のroute向聴 / 13。
                           それ以外はmask 0・値0
```

## R1（反応候補）

候補はPassを先頭に、Chi・Ponを（種別、鳴く牌、消費牌）のcanonical順に並べる。

```text
kind                 3  Pass / Chi / Pon one-hot
called_tile_type    34  鳴く牌の牌種one-hot（Passは0）
called_red           1  鳴く牌が赤5
consumed_tile_type  34  消費牌の牌種ごとの枚数 / 2（Passは0）
consumed_red         1  消費牌の赤5枚数
route_<r>            4  r = 通常 / Tanyao / Honitsu萬 / 筒 / 索。
  x 5                   成立mask、route向聴 / 13、差の成立mask、
                        (route向聴 - Pass候補のroute向聴) / 13
yakuhai_triplet      1  その鳴きで役牌の刻子ができる（役牌のPon）
own_yakuhai          1  自席に役牌の副露（Pon・槓）または純手牌の暗刻がある（判断で共通）
```

route向聴は`calculate_restricted_standard_shanten`によるroute外の牌を浮き牌とする
通常形向聴である。Chi / Ponの値は、消費牌を除いた手牌から喰い替え規則で許される
打牌をそれぞれ1枚行った手を列挙し、routeごとに独立した最小値を取る。Pass候補の
値は鳴かない現在の手の向聴である。既存の副露・鳴く牌・消費牌がrouteと矛盾する
場合は成立maskを0、値を0にし、大きな向聴で代用しない。差の欄も、どちらかが
不成立ならmask・値とも0にする。

layout・scale・規則を変える場合は既存identityを書き換えず新しいidentityにする。
"""

from collections import Counter
from collections.abc import Callable, Container, Sequence
from dataclasses import dataclass

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.riichi_safe_tiles import (
    RIICHI_SAFE_TILE_RULE_ASSUMPTION,
    derive_riichi_safe_tiles,
)
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.belief.wait_shape_support import river_relation, wait_shape_support
from lisjong.hand_evaluation.shanten import calculate_restricted_standard_shanten
from lisjong.learning._canonical import value_digest
from lisjong.learning.candidate_encoding import (
    CANDIDATE_ENCODING_DIMENSION,
    build_scorer_candidates,
    encode_candidates,
)
from lisjong.learning.candidate_encoding import (
    encoding_block as m0_encoding_block,
)
from lisjong.learning.errors import CandidateFeatureError
from lisjong.learning.l2a_decision import (
    L2A_DECISION_CLASSIFICATION_IDENTITY,
    L2aDecisionKind,
    scorer_candidates,
)
from lisjong.policy_contract import (
    ChiAction,
    DecisionContext,
    DiscardAction,
    InternalAction,
    PassAction,
    PolicyInput,
    PonAction,
    RiichiState,
    Tile,
    TileCategory,
    TileType,
)
from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.tile import tile_sort_key
from lisjong.policy_contract.wind import Wind
from lisjong.structural_efficiency import post_discard_concealed_hand

M1_ENCODING_IDENTITY = "lisjong-l2a-discard-candidate-encoding-m1-v1"
R1_ENCODING_IDENTITY = "lisjong-l2a-reaction-candidate-encoding-r1-v1"

TILE_TYPE_AXIS_SIZE = 34
ROUTE_SHANTEN_SCALE = 13.0
MAX_ROUTE_SHANTEN = 13
"""route向聴の上限。route外の牌は浮き牌なので、使える牌が無い14枚の手で13になる
（通常形の上限8より大きい）。範囲外はfail closedする。"""
REMAINING_SCALE = 4.0
DISCARD_DORA_SCALE = 4.0
RETAINED_DORA_SCALE = 8.0
RETAINED_RED_SCALE = 3.0
YAKUHAI_SCALE = 2.0
CONSUMED_SCALE = 2.0

_NUMBER_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)
_ALL_TILE_TYPES = frozenset(
    TileType(category, rank)
    for category in TileCategory
    for rank in range(1, 8 if category is TileCategory.HONOR else 10)
)
_HONOR_TILE_TYPES = frozenset(
    tile_type
    for tile_type in _ALL_TILE_TYPES
    if tile_type.category is TileCategory.HONOR
)


@dataclass(frozen=True, slots=True)
class _Route:
    name: str
    usable_tile_types: frozenset[TileType]


_NORMAL_ROUTE = _Route("normal", _ALL_TILE_TYPES)
_TANYAO_ROUTE = _Route(
    "tanyao",
    frozenset(
        tile_type
        for tile_type in _ALL_TILE_TYPES
        if tile_type.category is not TileCategory.HONOR and 2 <= tile_type.rank <= 8
    ),
)
_HONITSU_ROUTES = tuple(
    _Route(
        f"honitsu_{category.value}",
        frozenset(
            tile_type for tile_type in _ALL_TILE_TYPES if tile_type.category is category
        )
        | _HONOR_TILE_TYPES,
    )
    for category in _NUMBER_CATEGORIES
)
M1_OPEN_ROUTES = (_TANYAO_ROUTE, *_HONITSU_ROUTES)
R1_ROUTES = (_NORMAL_ROUTE, _TANYAO_ROUTE, *_HONITSU_ROUTES)

_OPPONENT_NAMES = ("shimocha", "toimen", "kamicha")
_WIND_RANK = {Wind.EAST: 1, Wind.SOUTH: 2, Wind.WEST: 3, Wind.NORTH: 4}
_DRAGON_RANKS = frozenset({5, 6, 7})


def _layout(blocks: Sequence[tuple[str, int]]) -> tuple[dict[str, int], int]:
    offsets: dict[str, int] = {}
    offset = 0
    for name, size in blocks:
        offsets[name] = offset
        offset += size
    return offsets, offset


M1_BLOCKS = (
    ("wait_shape_support", 6),
    ("remaining_same_tile", 1),
    *((f"opponent_{name}", 5) for name in _OPPONENT_NAMES),
    ("safe_against_all_riichi", 1),
    ("discard_dora", 1),
    ("retained_dora", 1),
    ("retained_red", 1),
    ("tile_class", 3),
    *((f"open_route_{route.name}", 2) for route in M1_OPEN_ROUTES),
)
M1_BLOCK_OFFSETS, M1_EXTRA_DIMENSION = _layout(M1_BLOCKS)
M1_ENCODING_DIMENSION = CANDIDATE_ENCODING_DIMENSION + M1_EXTRA_DIMENSION

R1_BLOCKS = (
    ("kind", 3),
    ("called_tile_type", TILE_TYPE_AXIS_SIZE),
    ("called_red", 1),
    ("consumed_tile_type", TILE_TYPE_AXIS_SIZE),
    ("consumed_red", 1),
    *((f"route_{route.name}", 4) for route in R1_ROUTES),
    ("yakuhai_triplet", 1),
    ("own_yakuhai", 1),
)
R1_BLOCK_OFFSETS, R1_ENCODING_DIMENSION = _layout(R1_BLOCKS)


# --- player-safe解析値 -------------------------------------------------------


def _dora_tile_type(indicator: TileType) -> TileType:
    """表示牌が指すドラの牌種（数牌は9の次が1、風牌・三元牌はそれぞれ循環）。"""
    rank = indicator.rank
    if indicator.category is not TileCategory.HONOR:
        return TileType(indicator.category, rank % 9 + 1)
    if rank <= 4:
        return TileType(TileCategory.HONOR, rank % 4 + 1)
    return TileType(TileCategory.HONOR, (rank - 5 + 1) % 3 + 5)


def _dora_multiplicity(policy_input: PolicyInput) -> Counter[TileType]:
    return Counter(
        _dora_tile_type(tile.tile_type) for tile in policy_input.round.dora_indicators
    )


def _yakuhai_value(policy_input: PolicyInput, tile_type: TileType) -> int:
    """役牌の翻数（三元牌1、場風1、自風1。連風牌は2）。"""
    if tile_type.category is not TileCategory.HONOR:
        return 0
    if tile_type.rank in _DRAGON_RANKS:
        return 1
    seat_wind_rank = (
        int(policy_input.self_seat) - int(policy_input.round.dealer_seat)
    ) % 4 + 1
    round_wind_rank = _WIND_RANK[policy_input.round.round_wind]
    return int(tile_type.rank == seat_wind_rank) + int(
        tile_type.rank == round_wind_rank
    )


def _own_melds(policy_input: PolicyInput):
    return policy_input.players[int(policy_input.self_seat)].melds


def _route_compatible(route: _Route, tiles: Sequence[Tile]) -> bool:
    return all(tile.tile_type in route.usable_tile_types for tile in tiles)


def _route_shanten(tiles: Sequence[Tile], usable: Container[TileType]) -> int:
    shanten = calculate_restricted_standard_shanten(tiles, usable)
    if not 0 <= shanten <= MAX_ROUTE_SHANTEN:
        raise CandidateFeatureError(
            f"route shanten {shanten} is outside 0..{MAX_ROUTE_SHANTEN}"
        )
    return shanten


# --- M1 -----------------------------------------------------------------------


def _m1_extra(decision: DecisionContext, actions: Sequence[DiscardAction]):
    """候補ごとのM1追加block（長さM1_EXTRA_DIMENSION）を返す。"""
    policy_input = decision.input
    self_index = int(policy_input.self_seat)
    try:
        safe_tiles = derive_riichi_safe_tiles(policy_input)
    except ValueError as exc:
        raise CandidateFeatureError(f"riichi safe tiles are undefined: {exc}") from exc
    safe_by_seat = {int(item.riichi_seat): item for item in safe_tiles.by_riichi_player}
    safe_against_all = (
        safe_tiles.safe_against_all_riichi_players
        if safe_tiles.by_riichi_player
        else frozenset()
    )
    remaining = derive_remaining_tile_inventory(policy_input).remaining_tile_counts
    opponents = []
    for offset in (1, 2, 3):
        index = (self_index + offset) % 4
        player = policy_input.players[index]
        opponents.append(
            (
                player.riichi is not RiichiState.NONE,
                frozenset(discard.tile.tile_type for discard in player.discards),
                safe_by_seat.get(index),
            )
        )
    dora = _dora_multiplicity(policy_input)
    melds = _own_melds(policy_input)
    is_open = any(meld.kind is not MeldKind.ANKAN for meld in melds)
    meld_tiles = tuple(tile for meld in melds for tile in meld.tiles)
    open_routes = tuple(
        is_open and _route_compatible(route, meld_tiles) for route in M1_OPEN_ROUTES
    )
    concealed = policy_input.own_hand.concealed_tiles
    offsets = M1_BLOCK_OFFSETS

    extras: list[list[float]] = []
    for action in actions:
        tile_type = action.tile.tile_type
        vector = [0.0] * M1_EXTRA_DIMENSION
        support = wait_shape_support(tile_type, remaining)
        for position, value in enumerate(
            (
                support.tanki,
                support.shanpon,
                support.kanchan,
                support.penchan,
                support.ryanmen_low_side,
                support.ryanmen_high_side,
            )
        ):
            vector[offsets["wait_shape_support"] + position] = float(value)
        vector[offsets["remaining_same_tile"]] = (
            remaining[tile_type_index(tile_type)] / REMAINING_SCALE
        )
        for name, (riichi, river, safe) in zip(_OPPONENT_NAMES, opponents, strict=True):
            relation = river_relation(tile_type, river)
            base = offsets[f"opponent_{name}"]
            vector[base] = float(riichi)
            vector[base + 1] = float(relation.in_river)
            vector[base + 2] = float(
                safe is not None and tile_type in safe.passed_tile_types
            )
            vector[base + 3] = float(relation.ryanmen_low_far_end_in_river)
            vector[base + 4] = float(relation.ryanmen_high_far_end_in_river)
        vector[offsets["safe_against_all_riichi"]] = float(
            tile_type in safe_against_all
        )
        retained = post_discard_concealed_hand(concealed, action.tile)
        vector[offsets["discard_dora"]] = dora[tile_type] / DISCARD_DORA_SCALE
        vector[offsets["retained_dora"]] = (
            sum(dora[tile.tile_type] for tile in retained) / RETAINED_DORA_SCALE
        )
        vector[offsets["retained_red"]] = (
            sum(tile.is_red for tile in retained) / RETAINED_RED_SCALE
        )
        is_honor = tile_type.category is TileCategory.HONOR
        vector[offsets["tile_class"]] = float(not is_honor and tile_type.rank in (1, 9))
        vector[offsets["tile_class"] + 1] = float(is_honor)
        vector[offsets["tile_class"] + 2] = (
            _yakuhai_value(policy_input, tile_type) / YAKUHAI_SCALE
        )
        for route, established in zip(M1_OPEN_ROUTES, open_routes, strict=True):
            if not established:
                continue
            base = offsets[f"open_route_{route.name}"]
            vector[base] = 1.0
            vector[base + 1] = (
                _route_shanten(retained, route.usable_tile_types) / ROUTE_SHANTEN_SCALE
            )
        extras.append(vector)
    return extras


def _encode_discard_m0(decision: DecisionContext):
    candidates = build_scorer_candidates(decision)
    actions = tuple(candidate.action for candidate in candidates)
    expected = scorer_candidates(decision, L2aDecisionKind.DISCARD)
    if len(actions) != len(expected) or set(actions) != set(expected):
        raise CandidateFeatureError(
            "M0 candidates do not match the L2a DISCARD scorer candidates"
        )
    return actions, encode_candidates(candidates)


def _encode_discard_m1(decision: DecisionContext):
    actions, base = _encode_discard_m0(decision)
    extras = _m1_extra(decision, actions)
    return actions, tuple(
        (*vector, *extra) for vector, extra in zip(base, extras, strict=True)
    )


# --- R1 -----------------------------------------------------------------------


def _reaction_sort_key(action: InternalAction) -> tuple[object, ...]:
    if isinstance(action, PassAction):
        return (0,)
    return (
        1 if isinstance(action, ChiAction) else 2,
        tile_sort_key(action.called_tile),
        tuple(sorted(tile_sort_key(tile) for tile in action.consumed_tiles)),
    )


def _remove_consumed(concealed: Sequence[Tile], consumed: Sequence[Tile]) -> list[Tile]:
    remaining = list(concealed)
    for tile in consumed:
        try:
            remaining.remove(tile)
        except ValueError:
            raise CandidateFeatureError(
                "call consumes a tile that is not in the concealed hand"
            ) from None
    return remaining


def _kuikae_forbidden(action: ChiAction | PonAction) -> frozenset[TileType]:
    """鳴いた直後に切れない牌種（現物の喰い替えと、Chiの筋喰い替え）。"""
    called = action.called_tile.tile_type
    if isinstance(action, PonAction):
        return frozenset({called})
    ranks = sorted(
        (called.rank, *(tile.tile_type.rank for tile in action.consumed_tiles))
    )
    forbidden = {called}
    if called.rank == ranks[0] and ranks[-1] + 1 <= 9:
        forbidden.add(TileType(called.category, ranks[-1] + 1))
    elif called.rank == ranks[-1] and ranks[0] - 1 >= 1:
        forbidden.add(TileType(called.category, ranks[0] - 1))
    return frozenset(forbidden)


def _call_route_values(
    concealed: Sequence[Tile],
    meld_tiles: Sequence[Tile],
    action: ChiAction | PonAction,
) -> tuple[int | None, ...]:
    """routeごとの独立した最小向聴。routeが成立しなければNone。"""
    after_call = _remove_consumed(concealed, action.consumed_tiles)
    forbidden = _kuikae_forbidden(action)
    hands = [
        (*after_call[:index], *after_call[index + 1 :])
        for index, tile in enumerate(after_call)
        if tile.tile_type not in forbidden
    ]
    if not hands:
        raise CandidateFeatureError("call leaves no discard allowed by kuikae")
    call_tiles = (*meld_tiles, action.called_tile, *action.consumed_tiles)
    values: list[int | None] = []
    for route in R1_ROUTES:
        if not _route_compatible(route, call_tiles):
            values.append(None)
            continue
        values.append(
            min(_route_shanten(hand, route.usable_tile_types) for hand in hands)
        )
    return tuple(values)


def _has_own_yakuhai(policy_input: PolicyInput) -> bool:
    for meld in _own_melds(policy_input):
        if meld.kind is not MeldKind.CHI and _yakuhai_value(
            policy_input, meld.tiles[0].tile_type
        ):
            return True
    counts = Counter(tile.tile_type for tile in policy_input.own_hand.concealed_tiles)
    return any(
        count >= 3 and _yakuhai_value(policy_input, tile_type)
        for tile_type, count in counts.items()
    )


def _encode_reaction_r1(decision: DecisionContext):
    actions = tuple(
        sorted(
            scorer_candidates(decision, L2aDecisionKind.CALL), key=_reaction_sort_key
        )
    )
    if not actions or not isinstance(actions[0], PassAction):
        raise CandidateFeatureError("reaction candidates must include PassAction")
    if len({_reaction_sort_key(action) for action in actions}) != len(actions):
        raise CandidateFeatureError("reaction candidates are not unique")
    policy_input = decision.input
    concealed = policy_input.own_hand.concealed_tiles
    meld_tiles = tuple(tile for meld in _own_melds(policy_input) for tile in meld.tiles)
    pass_values = tuple(
        _route_shanten(concealed, route.usable_tile_types)
        if _route_compatible(route, meld_tiles)
        else None
        for route in R1_ROUTES
    )
    own_yakuhai = float(_has_own_yakuhai(policy_input))
    offsets = R1_BLOCK_OFFSETS

    vectors: list[tuple[float, ...]] = []
    for action in actions:
        vector = [0.0] * R1_ENCODING_DIMENSION
        if isinstance(action, PassAction):
            vector[offsets["kind"]] = 1.0
            values = pass_values
        else:
            vector[offsets["kind"] + (1 if isinstance(action, ChiAction) else 2)] = 1.0
            called = action.called_tile
            vector[offsets["called_tile_type"] + tile_type_index(called.tile_type)] = (
                1.0
            )
            vector[offsets["called_red"]] = float(called.is_red)
            for tile in action.consumed_tiles:
                vector[
                    offsets["consumed_tile_type"] + tile_type_index(tile.tile_type)
                ] += 1.0 / CONSUMED_SCALE
                vector[offsets["consumed_red"]] += float(tile.is_red)
            vector[offsets["yakuhai_triplet"]] = float(
                isinstance(action, PonAction)
                and _yakuhai_value(policy_input, called.tile_type) > 0
            )
            values = _call_route_values(concealed, meld_tiles, action)
        for route, value, pass_value in zip(
            R1_ROUTES, values, pass_values, strict=True
        ):
            if value is None:
                continue
            base = offsets[f"route_{route.name}"]
            vector[base] = 1.0
            vector[base + 1] = value / ROUTE_SHANTEN_SCALE
            if pass_value is not None:
                vector[base + 2] = 1.0
                vector[base + 3] = (value - pass_value) / ROUTE_SHANTEN_SCALE
        vector[offsets["own_yakuhai"]] = own_yakuhai
        vectors.append(tuple(vector))
    return actions, tuple(vectors)


# --- encoding contract --------------------------------------------------------

type _Encoder = Callable[
    [DecisionContext],
    tuple[tuple[InternalAction, ...], tuple[tuple[float, ...], ...]],
]


@dataclass(frozen=True, slots=True)
class L2aEncodedDecision:
    """1判断のcanonical順の候補actionと、同じ順の候補vector。

    `actions`は`decision.legal_actions`側のobjectそのものである。
    """

    actions: tuple[InternalAction, ...]
    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class L2aEncoding:
    """L2aの1つの候補encoding contract（M0 / M1 / R1）。"""

    name: str
    identity: str
    head: L2aDecisionKind
    dimension: int
    specification: dict[str, object]
    _encoder: _Encoder

    @property
    def fingerprint(self) -> str:
        return value_digest(self.specification)

    def block(self) -> dict[str, object]:
        """dataset / artifactがbindするencoding identity blockを返す。"""
        return {
            "classification": L2A_DECISION_CLASSIFICATION_IDENTITY,
            "dimension": self.dimension,
            "fingerprint": self.fingerprint,
            "head": self.head.value,
            "identity": self.identity,
            "name": self.name,
        }

    def encode(self, decision: DecisionContext) -> L2aEncodedDecision:
        """scorer判断を候補actionとvectorへ写す。datasetとservingの共通入口。"""
        if not isinstance(decision, DecisionContext):
            raise TypeError("decision must be a DecisionContext")
        actions, vectors = self._encoder(decision)
        if len(actions) < 2 or len(vectors) != len(actions):
            raise CandidateFeatureError(
                f"{self.name} encoding needs two or more candidates"
            )
        if any(len(vector) != self.dimension for vector in vectors):
            raise CandidateFeatureError(f"{self.name} vector has a wrong dimension")
        return L2aEncodedDecision(actions=actions, vectors=vectors)


def _blocks_value(blocks, offsets) -> list[dict[str, object]]:
    return [
        {"name": name, "offset": offsets[name], "size": size} for name, size in blocks
    ]


_ROUTE_SPECIFICATION = {
    route.name: sorted(
        tile_type_index(tile_type) for tile_type in route.usable_tile_types
    )
    for route in R1_ROUTES
}

ENCODING_M0 = L2aEncoding(
    name="m0",
    identity=m0_encoding_block()["identity"],
    head=L2aDecisionKind.DISCARD,
    dimension=CANDIDATE_ENCODING_DIMENSION,
    specification={"l0.2": m0_encoding_block()},
    _encoder=_encode_discard_m0,
)
"""M0：L0.2の打牌候補encodingをidentityごと無変更で使う。"""

ENCODING_M1 = L2aEncoding(
    name="m1",
    identity=M1_ENCODING_IDENTITY,
    head=L2aDecisionKind.DISCARD,
    dimension=M1_ENCODING_DIMENSION,
    specification={
        "base": m0_encoding_block(),
        "blocks": _blocks_value(M1_BLOCKS, M1_BLOCK_OFFSETS),
        "dimension": M1_ENCODING_DIMENSION,
        "identity": M1_ENCODING_IDENTITY,
        "opponent_order": "self+1,self+2,self+3",
        "riichi_safe_tile_rule": RIICHI_SAFE_TILE_RULE_ASSUMPTION,
        "routes": {
            route.name: _ROUTE_SPECIFICATION[route.name] for route in M1_OPEN_ROUTES
        },
        "scales": {
            "discard_dora": DISCARD_DORA_SCALE,
            "remaining_same_tile": REMAINING_SCALE,
            "retained_dora": RETAINED_DORA_SCALE,
            "retained_red": RETAINED_RED_SCALE,
            "route_shanten": ROUTE_SHANTEN_SCALE,
            "yakuhai": YAKUHAI_SCALE,
        },
    },
    _encoder=_encode_discard_m1,
)

ENCODING_R1 = L2aEncoding(
    name="r1",
    identity=R1_ENCODING_IDENTITY,
    head=L2aDecisionKind.CALL,
    dimension=R1_ENCODING_DIMENSION,
    specification={
        "blocks": _blocks_value(R1_BLOCKS, R1_BLOCK_OFFSETS),
        "candidate_order": "pass,then(chi<pon,called tile,consumed tiles)",
        "dimension": R1_ENCODING_DIMENSION,
        "identity": R1_ENCODING_IDENTITY,
        "kuikae": "called-tile-and-chi-suji-swap-v1",
        "route_value": "independent-minimum-over-kuikae-allowed-discards",
        "routes": _ROUTE_SPECIFICATION,
        "scales": {"consumed": CONSUMED_SCALE, "route_shanten": ROUTE_SHANTEN_SCALE},
    },
    _encoder=_encode_reaction_r1,
)

L2A_ENCODINGS = {item.name: item for item in (ENCODING_M0, ENCODING_M1, ENCODING_R1)}


def l2a_encoding(name: str) -> L2aEncoding:
    """名前（m0 / m1 / r1）からencodingを返す。未知の名前はfail closedする。"""
    try:
        return L2A_ENCODINGS[name]
    except KeyError:
        raise CandidateFeatureError(
            f"unknown L2a encoding {name!r}; expected one of {sorted(L2A_ENCODINGS)}"
        ) from None


__all__ = [
    "ENCODING_M0",
    "ENCODING_M1",
    "ENCODING_R1",
    "L2A_ENCODINGS",
    "M1_BLOCK_OFFSETS",
    "M1_ENCODING_DIMENSION",
    "M1_ENCODING_IDENTITY",
    "R1_BLOCK_OFFSETS",
    "R1_ENCODING_DIMENSION",
    "R1_ENCODING_IDENTITY",
    "L2aEncodedDecision",
    "L2aEncoding",
    "l2a_encoding",
]
