"""副露者の構造的待ち確率の推定（lisbun/lisjong#259 範囲1、推論側）。

推論は`PolicyInput`だけを入力にする。他家の手牌・ラベルなど学習専用の事実はこのmoduleへ
渡さず、学習・評価は`lisjong.learning.open_wait_evaluation`が行う。学習時の特徴も、ここの
`tenpai_features()` / `wait_tile_feature_table()`が同じ`PolicyInput`から作る。
ML runtimeには依存しない。

対象は、観測者から見た他家のうち、リーチしておらず暗槓以外の副露を1つ以上持つ席（#257の
「副露者」層）である。推定するのは34牌種ごとの**構造的待ち**（`HandBelief.wait_probability`の
意味。フリテン・役の有無を含まない）で、聴牌でなければ全牌種0が正解になる。

```text
P(W_t | I) = P(聴牌 | I) * P(W_t | 聴牌, I)
```

の2段に分け、どちらもlogistic回帰で推定する。現物・河由来の特徴があっても確率を0に固定しない
（構造的な待ちはフリテンでも成り立つ）。役の有無も同様で、副露の形は特徴にするだけで、
役のない待ちを0にしない。
"""

from dataclasses import dataclass, replace
from math import exp

from lisjong.belief.canonical_axes import (
    tile_type_from_index,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
from lisjong.belief.conditional_uniform_hand_belief import (
    estimate_conditional_uniform_hand_belief,
)
from lisjong.belief.fixed_point import probability_to_raw
from lisjong.belief.hand_belief import HandBelief
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.belief.wait_shape_support import river_relation, wait_shape_support
from lisjong.learning.riichi_deal_in_estimator import _suji, _turn_bucket
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _WIND_RANK,
    _yakuhai_han_value,
)
from lisjong.policies.value_aware_two_step_ukeire import _dora_tile_type
from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import TileCategory, TileType

FEATURE_SET = "open-wait-features-v1"
BIAS = "bias"
CLIP_EPSILON = 1e-6
TILE_TYPES = tuple(tile_type_from_index(index) for index in range(34))
_CALLS = (MeldKind.CHI, MeldKind.PON, MeldKind.DAIMINKAN)
_TRIPLETS = (MeldKind.PON, MeldKind.DAIMINKAN, MeldKind.ANKAN, MeldKind.KAKAN)

_SUPPORT_FIELD = {
    "tanki": "tanki",
    "shanpon": "shanpon",
    "kanchan": "kanchan",
    "penchan": "penchan",
    "ryanmen_low": "ryanmen_low_side",
    "ryanmen_high": "ryanmen_high_side",
}


def clip_probability(value: float) -> float:
    return min(max(value, CLIP_EPSILON), 1.0 - CLIP_EPSILON)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    z = exp(value)
    return z / (1.0 + z)


def is_open_opponent(policy_input: PolicyInput, seat: int) -> bool:
    """観測者以外で、リーチしておらず暗槓以外の副露がある席か（#257の副露者層）。"""
    if seat == int(policy_input.self_seat):
        return False
    player = policy_input.players[seat]
    return player.riichi is RiichiState.NONE and any(
        meld.kind is not MeldKind.ANKAN for meld in player.melds
    )


@dataclass(frozen=True, slots=True)
class OpenView:
    """1判断・副露者1席について、牌種に依らない公開情報の要約。"""

    seat: int
    meld_count: int
    discard_count: int
    discards_since_call: int
    recent_tedashi: int
    last_tedashi: bool
    all_triplets: bool
    yakuhai_meld: bool
    all_simples: bool
    flush_suit: TileCategory | None
    dora_in_melds: int
    melded_types: frozenset[TileType]
    river: frozenset[TileType]
    passed_since_last_discard: frozenset[TileType]
    remaining_counts: tuple[int, ...]
    dora: frozenset[TileType]
    yakuhai: frozenset[TileType]


def open_view(
    policy_input: PolicyInput,
    seat: int,
    remaining_counts: tuple[int, ...] | None = None,
) -> OpenView:
    """副露者`seat`の要約。`remaining_counts`は同じ判断の他の席と共有するときに渡す。"""
    if not is_open_opponent(policy_input, seat):
        raise ValueError("the seat is not an open, non-riichi opponent")
    player = policy_input.players[seat]
    round_state = policy_input.round
    seat_wind_rank = (seat - int(round_state.dealer_seat)) % 4 + 1
    yakuhai = frozenset(
        TileType(TileCategory.HONOR, rank)
        for rank in range(1, 8)
        if _yakuhai_han_value(
            TileType(TileCategory.HONOR, rank),
            seat_wind_rank=seat_wind_rank,
            round_wind_rank=_WIND_RANK[round_state.round_wind],
        )
        > 0
    )
    dora = frozenset(
        _dora_tile_type(indicator.tile_type)
        for indicator in round_state.dora_indicators
    )
    # 最後に鳴いた時点: 他家の捨て牌のうち、この席が鳴いたものの最大order
    calls = [
        discard.order
        for other in policy_input.players
        for discard in other.discards
        if discard.called_by is not None and int(discard.called_by) == seat
    ]
    last_call = max(calls) if calls else None
    discards = player.discards
    last_order = max((d.order for d in discards), default=None)
    recent = sorted(discards, key=lambda d: d.order)[-3:]
    meld_tiles = [tile for meld in player.melds for tile in meld.tiles]
    suits = {
        tile.tile_type.category
        for tile in meld_tiles
        if tile.tile_type.category is not TileCategory.HONOR
    }
    passed = (
        frozenset(
            discard.tile.tile_type
            for index, other in enumerate(policy_input.players)
            if index != seat
            for discard in other.discards
            if discard.order > last_order
        )
        if last_order is not None
        else frozenset()
    )
    return OpenView(
        seat=seat,
        meld_count=len(player.melds),
        discard_count=len(discards),
        discards_since_call=sum(
            last_call is not None and d.order > last_call for d in discards
        ),
        recent_tedashi=sum(not d.tsumogiri for d in recent),
        last_tedashi=bool(recent) and not recent[-1].tsumogiri,
        all_triplets=all(meld.kind in _TRIPLETS for meld in player.melds),
        yakuhai_meld=any(
            meld.kind in _TRIPLETS and meld.tiles[0].tile_type in yakuhai
            for meld in player.melds
        ),
        all_simples=all(
            tile.tile_type.category is not TileCategory.HONOR
            and 2 <= tile.tile_type.rank <= 8
            for tile in meld_tiles
        ),
        flush_suit=suits.pop() if len(suits) == 1 else None,
        dora_in_melds=sum(
            (tile.tile_type in dora) + tile.is_red for tile in meld_tiles
        ),
        melded_types=frozenset(tile.tile_type for tile in meld_tiles),
        river=frozenset(d.tile.tile_type for d in discards),
        passed_since_last_discard=passed,
        remaining_counts=remaining_counts
        if remaining_counts is not None
        else derive_remaining_tile_inventory(policy_input).remaining_tile_counts,
        dora=dora,
        yakuhai=yakuhai,
    )


def _discard_bucket(count: int) -> str:
    for upper, name in ((3, "0-3"), (6, "4-6"), (9, "7-9"), (12, "10-12")):
        if count <= upper:
            return name
    return "13+"


def tenpai_features(view: OpenView) -> tuple[tuple[str, float], ...]:
    """聴牌かどうかの特徴（すべて値1の離散特徴）。"""
    names = {
        BIAS,
        f"melds_{view.meld_count}",
        f"discards_{_discard_bucket(view.discard_count)}",
        f"since_call_{min(view.discards_since_call, 3)}",
        f"recent_tedashi_{view.recent_tedashi}",
        f"dora_in_melds_{min(view.dora_in_melds, 2)}",
    }
    signals = {
        "melds_all_triplets": view.all_triplets,
        "melds_yakuhai": view.yakuhai_meld,
        "melds_all_simples": view.all_simples,
        "melds_one_suit": view.flush_suit is not None,
    }
    names.update(name for name, on in signals.items() if on)
    if not any(signals.values()):
        names.add("melds_no_visible_yaku")
    if view.last_tedashi:
        names.add("last_tedashi")
    return tuple(sorted((name, 1.0) for name in names))


def _remaining(view: OpenView, tile: TileType, rank: int) -> int:
    return view.remaining_counts[tile_type_index(TileType(tile.category, rank))]


def wait_tile_features(view: OpenView, tile: TileType) -> tuple[tuple[str, float], ...]:
    """聴牌のとき牌種`tile`を待つかどうかの特徴（名前と値の組）。

    #245（`riichi_wait_estimator.wait_features`）と同じ形・河の層に、副露の形との関係を加える。
    """
    remaining = view.remaining_counts[tile_type_index(tile)]
    names = {
        BIAS,
        f"turn_{_turn_bucket(view.discard_count)}",
        f"remaining_self_{remaining}",
        f"melds_{view.meld_count}",
    }
    numeric: list[tuple[str, float]] = []
    support = wait_shape_support(tile, view.remaining_counts)
    relation = river_relation(tile, view.river)
    names.add(f"support_count_{min(support.count, 3)}")
    for shape, field in _SUPPORT_FIELD.items():
        if getattr(support, field):
            names.add(f"support_{shape}")
    if relation.in_river:
        names.add("genbutsu")
    if tile in view.passed_since_last_discard:
        names.add("passed_since_last_discard")
    if tile in view.dora:
        names.add("dora")
    if tile in view.melded_types:
        names.add("in_meld")

    honor = tile.category is TileCategory.HONOR
    if view.flush_suit is not None:
        if honor:
            names.add("flush_honor")
        elif tile.category is view.flush_suit:
            names.add("flush_in_suit")
        else:
            names.add("flush_off_suit")
    if view.all_simples:
        names.add(
            "simples_terminal_or_honor"
            if honor or tile.rank in (1, 9)
            else "simples_simple"
        )

    if honor:
        kind = "yakuhai" if tile in view.yakuhai else "guest"
        names.add(f"class_honor_{kind}")
        return tuple(sorted((name, 1.0) for name in names))
    rank = tile.rank
    names.add(f"class_rank_{rank}")
    names.add(f"suji_{_suji(tile, view.river)}")
    if support.ryanmen_low_side and not relation.ryanmen_low_far_end_in_river:
        names.add("ryanmen_low_open")
    if support.ryanmen_high_side and not relation.ryanmen_high_far_end_in_river:
        names.add("ryanmen_high_open")
    # 構成牌の残り枚数の積（壁の度合い）。成立可能な形だけ値を持つ
    combos = (
        ("combo_kanchan", support.kanchan, (rank - 1, rank + 1)),
        ("combo_penchan", support.penchan, (1, 2) if rank == 3 else (8, 9)),
        ("combo_ryanmen_low", support.ryanmen_low_side, (rank + 1, rank + 2)),
        ("combo_ryanmen_high", support.ryanmen_high_side, (rank - 2, rank - 1)),
    )
    for name, on, (a, b) in combos:
        if on:
            numeric.append(
                (name, _remaining(view, tile, a) * _remaining(view, tile, b) / 16.0)
            )
    return tuple(sorted([(name, 1.0) for name in names] + numeric))


def wait_tile_feature_table(
    view: OpenView,
) -> tuple[tuple[tuple[str, float], ...], ...]:
    """34牌種（canonical順）それぞれの特徴。学習・推論の両方がこの関数を使う。"""
    return tuple(wait_tile_features(view, tile) for tile in TILE_TYPES)


def _linear(weights: dict[str, float], features) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features)


@dataclass(frozen=True, slots=True)
class OpenWaitModel:
    """推定器: 聴牌のlogistic × 聴牌時の牌種ごとのlogistic。"""

    tenpai_weights: tuple[tuple[str, float], ...]
    wait_weights: tuple[tuple[str, float], ...]
    feature_set: str = FEATURE_SET

    def __post_init__(self) -> None:
        if self.feature_set != FEATURE_SET:
            raise ValueError(f"unknown feature set {self.feature_set!r}")

    def tenpai_probability(self, view: OpenView) -> float:
        return _sigmoid(_linear(dict(self.tenpai_weights), tenpai_features(view)))

    def predict_view(self, view: OpenView) -> tuple[float, ...]:
        tenpai = self.tenpai_probability(view)
        weights = dict(self.wait_weights)
        return tuple(
            clip_probability(tenpai * _sigmoid(_linear(weights, features)))
            for features in wait_tile_feature_table(view)
        )

    def predict(self, policy_input: PolicyInput, seat: int) -> dict[TileType, float]:
        return dict(zip(TILE_TYPES, self.predict_view(open_view(policy_input, seat))))


def estimate_open_wait_belief(
    policy_input: PolicyInput, seat: int, model: OpenWaitModel
) -> HandBelief:
    """副露者`seat`の`HandBelief`（Level 1: `wait_probability`あり、形別テーブルなし）。

    牌のmarginalは既存の条件付き一様推定器の値をそのまま使い、待ち確率だけを置き換える。
    """
    predictions = model.predict(policy_input, seat)
    dealer = policy_input.round.dealer_seat
    slots = [0, 0, 0, 0]
    for other in Seat:
        if other != policy_input.self_seat:
            slots[wind_index(wind_for_seat(other, dealer))] = 13 - 3 * len(
                policy_input.players[other].melds
            )
    beliefs = estimate_conditional_uniform_hand_belief(policy_input, tuple(slots))
    return replace(
        beliefs.hand(wind_for_seat(Seat(seat), dealer)),
        wait_probability_raw=tuple(
            probability_to_raw(predictions[tile]) for tile in TILE_TYPES
        ),
    )


__all__ = [
    "BIAS",
    "CLIP_EPSILON",
    "FEATURE_SET",
    "TILE_TYPES",
    "OpenView",
    "OpenWaitModel",
    "clip_probability",
    "estimate_open_wait_belief",
    "is_open_opponent",
    "open_view",
    "tenpai_features",
    "wait_tile_feature_table",
    "wait_tile_features",
]
