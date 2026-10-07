"""他家の期待枚数・赤5確率の推定（lisbun/lisjong#258、推論側）。

推論は`PolicyInput`だけを入力にする。他家の手牌・ラベルなど学習専用の事実はこのmoduleへ
渡さず、学習・評価は`lisjong.learning.hand_count_evaluation`が行う。学習時の特徴も、ここの
`count_feature_table()`が同じ`PolicyInput`から作る。ML runtimeには依存しない。

条件付き一様baseline（`estimate_conditional_uniform_hand_belief`）は、未見の牌を他家の
非公開slotと山（王牌を含む）へ一様に配る。この推定器は、他家`c`が牌種`t`を持ちやすい度合い
``w(c, t) = exp(θ · φ(c, t))``（山は1）を公開情報の特徴`φ`から求め、

```text
K(t, c) = 未見枚数(t) * w(c, t)
```

を、行和 = 未見枚数(t)、列和 = 他家`c`のslot数（山は残り）になるよう反復比例配分（IPF）
して期待枚数とする。`w`がすべて1なら条件付き一様baselineと一致する。行和・列和を保つので、
牌種ごとの物理的な保存則（他家の期待枚数の和 ≤ 未見枚数）が成り立つ。

赤5は、同じ色の5の中で赤が交換可能だと仮定し、``期待枚数(c, 5) * 未見の赤5 / 未見の5``
とする（赤5のための追加の学習はしない）。

θはtrainで、条件付き一様の期待枚数をoffsetとするPoisson回帰で求める（学習側）。
"""

from collections import Counter
from dataclasses import dataclass
from math import exp, floor

from lisjong.belief.canonical_axes import (
    red_five_index,
    tile_type_from_index,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
from lisjong.belief.concealed_hand_belief import ConcealedHandBelief
from lisjong.belief.fixed_point import SCALE
from lisjong.belief.hand_belief import HandBelief
from lisjong.belief.self_belief import exact_self_belief
from lisjong.belief.tile_conservation import (
    TileConservationResult,
    derive_remaining_tile_inventory,
)
from lisjong.learning.riichi_deal_in_estimator import _turn_bucket
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

FEATURE_SET = "hand-count-features-v1"
BIAS = "bias"
TILE_TYPES = tuple(tile_type_from_index(index) for index in range(34))
SUITS = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)
FIVE_INDICES = tuple(tile_type_index(TileType(suit, 5)) for suit in SUITS)
IPF_MAX_ITERATIONS = 500
IPF_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class OpponentView:
    """1判断・他家1席について、牌種に依らない公開情報の要約。"""

    seat: int
    slots: int
    stratum: str  # riichi / open / closed
    turn: str
    river_counts: tuple[int, ...]
    early_discards: frozenset[TileType]
    recent_tedashi: frozenset[TileType]
    melded: frozenset[TileType]
    flush_suit: TileCategory | None
    yakuhai: frozenset[TileType]


@dataclass(frozen=True, slots=True)
class CountContext:
    """1判断の公開情報。他家はseat順。"""

    remaining: tuple[int, ...]
    remaining_red: tuple[int, ...]
    dora: frozenset[TileType]
    opponents: tuple[OpponentView, ...]

    @property
    def wall_slots(self) -> int:
        return sum(self.remaining) - sum(view.slots for view in self.opponents)


def _yakuhai(policy_input: PolicyInput, seat: int) -> frozenset[TileType]:
    round_state = policy_input.round
    seat_wind_rank = (seat - int(round_state.dealer_seat)) % 4 + 1
    return frozenset(
        TileType(TileCategory.HONOR, rank)
        for rank in range(1, 8)
        if _yakuhai_han_value(
            TileType(TileCategory.HONOR, rank),
            seat_wind_rank=seat_wind_rank,
            round_wind_rank=_WIND_RANK[round_state.round_wind],
        )
        > 0
    )


def _opponent_view(policy_input: PolicyInput, seat: int) -> OpponentView:
    player = policy_input.players[seat]
    if player.riichi is not RiichiState.NONE:
        stratum = "riichi"
    elif any(meld.kind is not MeldKind.ANKAN for meld in player.melds):
        stratum = "open"
    else:
        stratum = "closed"
    discards = sorted(player.discards, key=lambda d: d.order)
    river = Counter(tile_type_index(d.tile.tile_type) for d in discards)
    meld_tiles = [tile.tile_type for meld in player.melds for tile in meld.tiles]
    suits = {t.category for t in meld_tiles if t.category is not TileCategory.HONOR}
    return OpponentView(
        seat=seat,
        slots=13 - 3 * len(player.melds),
        stratum=stratum,
        turn=_turn_bucket(len(discards)),
        river_counts=tuple(river.get(index, 0) for index in range(34)),
        early_discards=frozenset(d.tile.tile_type for d in discards[:6]),
        recent_tedashi=frozenset(
            d.tile.tile_type for d in discards[-3:] if not d.tsumogiri
        ),
        melded=frozenset(meld_tiles),
        flush_suit=suits.pop() if len(suits) == 1 and stratum == "open" else None,
        yakuhai=_yakuhai(policy_input, seat),
    )


def count_context(
    policy_input: PolicyInput, conservation: TileConservationResult | None = None
) -> CountContext:
    """`conservation`は同じ`policy_input`から導出したものを共有するときに渡す。"""
    if conservation is None:
        conservation = derive_remaining_tile_inventory(policy_input)
    return CountContext(
        remaining=tuple(conservation.remaining_tile_counts),
        remaining_red=tuple(conservation.remaining_red_five_counts),
        dora=frozenset(
            _dora_tile_type(indicator.tile_type)
            for indicator in policy_input.round.dora_indicators
        ),
        opponents=tuple(
            _opponent_view(policy_input, seat)
            for seat in range(4)
            if seat != int(policy_input.self_seat)
        ),
    )


def _group(tile: TileType) -> str:
    if tile.category is TileCategory.HONOR:
        return "honor"
    return "terminal" if tile.rank in (1, 9) else "simple"


def cell_features(
    context: CountContext, view: OpponentView, index: int
) -> tuple[str, ...]:
    """他家`view`が牌種`index`を持ちやすいかの特徴（すべて値1の離散特徴、名前順）。"""
    tile = TILE_TYPES[index]
    group = _group(tile)
    names = {
        BIAS,
        f"{view.stratum}_x_{group}",
        f"turn_{view.turn}_x_{group}",
        f"unseen_{context.remaining[index]}",
    }
    if tile.category is TileCategory.HONOR:
        names.add("class_honor_" + ("yakuhai" if tile in view.yakuhai else "guest"))
    else:
        names.add(f"class_rank_{tile.rank}")
        for distance in (1, 2):
            if any(
                1 <= tile.rank + step <= 9 and view.river_counts[index + step] > 0
                for step in (-distance, distance)
            ):
                names.add(f"near{distance}_in_river")
    river = view.river_counts[index]
    if river:
        names.add(f"river_{min(river, 2)}")
    if tile in view.early_discards:
        names.add("discarded_early")
    if tile in view.recent_tedashi:
        names.add("recent_tedashi")
    if tile in view.melded:
        names.add("in_meld")
    if tile in context.dora:
        names.add("dora")
    if view.flush_suit is not None:
        if tile.category is TileCategory.HONOR:
            names.add("flush_honor")
        elif tile.category is view.flush_suit:
            names.add("flush_in_suit")
        else:
            names.add("flush_off_suit")
    return tuple(sorted(names))


def count_feature_table(
    context: CountContext,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """他家（seat順）× 34牌種の特徴。学習・推論の両方がこの関数を使う。"""
    return tuple(
        tuple(cell_features(context, view, index) for index in range(34))
        for view in context.opponents
    )


def balance(
    kernel: list[list[float]], rows: tuple[int, ...], columns: tuple[int, ...]
) -> list[list[float]]:
    """`kernel`（牌種 × 列）を、行和`rows`・列和`columns`へ反復比例配分する。

    行和が0の牌種は0のまま。最後に列をそろえ、行和の誤差が許容値未満になるまで繰り返す。
    """
    matrix = [
        [value for value in row] if total else [0.0] * len(row)
        for row, total in zip(kernel, rows)
    ]
    for _ in range(IPF_MAX_ITERATIONS):
        for row, total in zip(matrix, rows):
            current = sum(row)
            if current:
                factor = total / current
                for j in range(len(row)):
                    row[j] *= factor
        for j, total in enumerate(columns):
            current = sum(row[j] for row in matrix)
            if current:
                factor = total / current
                for row in matrix:
                    row[j] *= factor
        error = max(abs(sum(row) - total) for row, total in zip(matrix, rows))
        if error < IPF_TOLERANCE:
            return matrix
    raise ValueError("the allocation did not converge")


def uniform_expected(context: CountContext) -> tuple[tuple[float, ...], ...]:
    """条件付き一様の期待枚数（他家seat順 × 34）。"""
    total = sum(context.remaining)
    return tuple(
        tuple(remaining * view.slots / total for remaining in context.remaining)
        for view in context.opponents
    )


@dataclass(frozen=True, slots=True)
class HandCountModel:
    """推定器: 持ちやすさの重み × 反復比例配分。"""

    weights: tuple[tuple[str, float], ...]
    feature_set: str = FEATURE_SET

    def __post_init__(self) -> None:
        if self.feature_set != FEATURE_SET:
            raise ValueError(f"unknown feature set {self.feature_set!r}")

    def expected_from(
        self, context: CountContext, table=None
    ) -> tuple[tuple[float, ...], ...]:
        """他家seat順 × 34の期待枚数（量子化前）。"""
        if table is None:
            table = count_feature_table(context)
        weights = dict(self.weights)
        kernel = [
            [
                remaining * exp(sum(weights.get(name, 0.0) for name in table[c][index]))
                for c in range(len(context.opponents))
            ]
            + [float(remaining)]
            for index, remaining in enumerate(context.remaining)
        ]
        columns = (*(view.slots for view in context.opponents), context.wall_slots)
        matrix = balance(kernel, context.remaining, columns)
        return tuple(
            tuple(matrix[index][c] for index in range(34))
            for c in range(len(context.opponents))
        )

    def predict(self, policy_input: PolicyInput) -> ConcealedHandBelief:
        conservation = derive_remaining_tile_inventory(policy_input)
        context = count_context(policy_input, conservation)
        expected = self.expected_from(context)
        dealer = policy_input.round.dealer_seat
        hands: list[HandBelief | None] = [None] * 4
        hands[wind_index(wind_for_seat(policy_input.self_seat, dealer))] = (
            exact_self_belief(policy_input.own_hand)
        )
        for view, counts in zip(context.opponents, expected):
            hands[wind_index(wind_for_seat(Seat(view.seat), dealer))] = quantize(
                context, counts
            )
        return ConcealedHandBelief(hands=tuple(hands))


def red_five_probabilities(
    context: CountContext, counts: tuple[float, ...]
) -> tuple[float, ...]:
    """赤5の確率（色順）。同じ色の5の中で赤が交換可能だとして期待枚数から導く。"""
    result = []
    for suit, five in zip(SUITS, FIVE_INDICES):
        remaining = context.remaining[five]
        red = context.remaining_red[red_five_index(suit)]
        result.append(counts[five] * red / remaining if remaining else 0.0)
    return tuple(result)


def quantize(context: CountContext, counts: tuple[float, ...]) -> HandBelief:
    """固定小数点へ切り捨てる。保存則（他家の和 ≤ 未見）は切り捨てで保たれる。"""
    expected = [
        min(floor(value * SCALE), remaining * SCALE)
        for value, remaining in zip(counts, context.remaining)
    ]
    reds = []
    for suit, five in zip(SUITS, FIVE_INDICES):
        remaining = context.remaining[five]
        red = context.remaining_red[red_five_index(suit)]
        if not remaining:
            reds.append(0)
            continue
        red_raw = min(floor(counts[five] * red / remaining * SCALE), red * SCALE)
        normal_raw = min(
            floor(counts[five] * (remaining - red) / remaining * SCALE),
            (remaining - red) * SCALE,
        )
        expected[five] = red_raw + normal_raw
        reds.append(red_raw)
    return HandBelief(
        expected_count_raw=tuple(expected), red_five_probability_raw=tuple(reds)
    )


__all__ = [
    "BIAS",
    "FEATURE_SET",
    "CountContext",
    "HandCountModel",
    "OpponentView",
    "balance",
    "cell_features",
    "count_context",
    "count_feature_table",
    "quantize",
    "red_five_probabilities",
    "uniform_expected",
]
