"""他家の形別待ちテーブル（`HandBelief`のLevel 2）の推定（lisbun/lisjong#260、推論側）。

推論は`PolicyInput`だけを入力にする。他家の手牌・ラベルなど学習専用の事実はこのmoduleへ
渡さず、学習・評価は`lisjong.learning.wait_shape_evaluation`が行う。ML runtimeには依存しない。
設計は`docs/wait-shape-belief.md`を正本とする。

対象は2つのpopulationで、モデルは別々に持つ。

- ``riichi``: #245（`riichi_wait_estimator`）の提供範囲の単独リーチ者
- ``open``: #259範囲1（`open_wait_estimator`）の提供範囲の副露者

どちらも、固定したLevel 1の待ち推定器が返す`HandBelief`へ7channelの形別テーブルを加える。
範囲外の席への要求は`ValueError`で拒否し、ゼロ予測や部分的な形別groupは作らない。

channelごとに独立したlogistic回帰で`q[c,t]`を求める（multi-label。channel間で正規化しない）。
構造上0にするslotは次の3つだけである。現物・河・役の有無・待ち牌自身の残り0枚では0にしない。

1. channelが構造上占め得ないslot（`HandBelief`のconstructorが拒否するslot）
2. 六つの通常形で、`wait_shape_support()`の枚数の必要条件を満たさないslot
3. 国士で、対象席に副露（暗槓を含む）が1つでもある場合の全slot

確率の整合性は**待ち確率による上限処理**（`cap-by-wait-v1`）で保つ。形の事象は構造的な待ちの
部分集合なので、`p[c,t] = min(q[c,t], w[t])`とする。`w`はLevel 1のraw値で、`q`を同じ
`probability_to_raw()`で丸めてから比べるので、rawでも`channel <= wait`になる。
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import exp, isfinite

from lisjong.belief.fixed_point import probability_to_raw
from lisjong.belief.hand_belief import _WAIT_MECHANISM_FIELDS, HandBelief
from lisjong.belief.wait_shape_support import wait_shape_support
from lisjong.learning.open_wait_estimator import (
    OpenView,
    OpenWaitModel,
    estimate_open_wait_belief,
    open_view,
    tenpai_features,
    wait_tile_features,
)
from lisjong.learning.riichi_deal_in_estimator import RiichiView, riichi_view
from lisjong.learning.riichi_wait_estimator import (
    BIAS,
    TILE_TYPES,
    LogisticWaitModel,
    estimate_riichi_wait_belief,
    wait_features,
)
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.seat import Seat

FEATURE_SET = "wait-shape-features-v1"
CONSISTENCY = "cap-by-wait-v1"
POPULATIONS = ("riichi", "open")
KOKUSHI = "kokushi"
# (channel名, HandBeliefのfield名, 構造上占め得るslot)。順序は`HandBelief`のfield順
CHANNELS = tuple(
    (
        name.removesuffix("_raw").removesuffix("_probability").removesuffix("_wait"),
        name,
        frozenset(slots),
    )
    for name, slots in _WAIT_MECHANISM_FIELDS
)
CHANNEL_NAMES = tuple(name for name, _, _ in CHANNELS)

Features = tuple[tuple[str, float], ...]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    z = exp(value)
    return z / (1.0 + z)


def riichi_shape_features(view: RiichiView) -> tuple[tuple[Features, ...], Features]:
    """単独リーチ者の特徴: 34牌種ごとの#245の特徴と、牌種に依らない特徴（なし）。"""
    return tuple(wait_features(view, tile) for tile in TILE_TYPES), ()


def open_shape_features(view: OpenView) -> tuple[tuple[Features, ...], Features]:
    """副露者の特徴: 34牌種ごとの#259の待ち特徴と、牌種に依らない#259の聴牌特徴。

    2つの特徴群は`tile.` / `tenpai.`のprefixで分け、biasは1つにする。
    """
    table = tuple(
        tuple(
            (name if name == BIAS else f"tile.{name}", value)
            for name, value in wait_tile_features(view, tile)
        )
        for tile in TILE_TYPES
    )
    shared = tuple(
        (f"tenpai.{name}", value)
        for name, value in tenpai_features(view)
        if name != BIAS
    )
    return table, shared


def supported_slots(
    remaining_counts: Sequence[int], has_meld: bool
) -> tuple[frozenset[int], ...]:
    """channelごと（`CHANNELS`の順）の、確率を推定するslot。これ以外のslotは構造上0にする。

    `has_meld`は対象席に副露（暗槓を含む）が1つでもあるか。国士だけが使う。
    """
    supports = [wait_shape_support(tile, remaining_counts) for tile in TILE_TYPES]
    return tuple(
        (frozenset() if has_meld else static)
        if channel == KOKUSHI
        else frozenset(index for index in static if getattr(supports[index], channel))
        for channel, _, static in CHANNELS
    )


@dataclass(frozen=True, slots=True)
class WaitShapeModel:
    """1つのpopulationの形別推定器。channelごとに別の係数を持つ。

    `weights`は`CHANNELS`の順。`None`は係数を持たない構造上のゼロモデルで、副露者の国士だけが
    これに当たる（副露があれば国士は成立しない）。それ以外の欠損は拒否する。
    """

    population: str
    weights: tuple[Features | None, ...]
    feature_set: str = FEATURE_SET

    def __post_init__(self) -> None:
        if self.feature_set != FEATURE_SET:
            raise ValueError(f"unknown feature set {self.feature_set!r}")
        if self.population not in POPULATIONS:
            raise ValueError(f"unknown population {self.population!r}")
        if len(self.weights) != len(CHANNELS):
            raise ValueError("weights must cover the seven channels in order")
        for channel, weights in zip(CHANNEL_NAMES, self.weights):
            zero_model = self.population == "open" and channel == KOKUSHI
            if (weights is None) is not zero_model:
                raise ValueError(f"{self.population}: bad {channel} model availability")
            if weights is not None and not all(
                type(name) is str and type(value) is float and isfinite(value)
                for name, value in weights
            ):
                raise ValueError(f"{channel} weights must be finite floats")

    def probabilities(
        self,
        tile_features: Sequence[Features],
        shared_features: Features,
        supported: Sequence[frozenset[int]],
    ) -> tuple[tuple[float, ...], ...]:
        """上限処理の前の`q[c][t]`（7channel × 34牌種）。構造上0のslotは0.0。

        `supported`は同じ判断・対象席の`supported_slots()`の値。
        """
        tables = []
        for weights, slots in zip(self.weights, supported, strict=True):
            if weights is None or not slots:
                tables.append((0.0,) * 34)
                continue
            lookup = dict(weights)
            shared = sum(lookup.get(name, 0.0) * v for name, v in shared_features)
            tables.append(
                tuple(
                    _sigmoid(
                        shared + sum(lookup.get(name, 0.0) * v for name, v in features)
                    )
                    if index in slots
                    else 0.0
                    for index, features in enumerate(tile_features)
                )
            )
        return tuple(tables)


def cap_by_wait(
    probabilities: Sequence[Sequence[float]], wait_raw: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """`q`をrawへ丸め、同じ牌種のwaitのrawを上限にする（`cap-by-wait-v1`）。"""
    return tuple(
        tuple(min(probability_to_raw(q), wait) for q, wait in zip(table, wait_raw))
        for table in probabilities
    )


def _with_shapes(
    belief: HandBelief,
    model: WaitShapeModel,
    features: tuple[tuple[Features, ...], Features],
    remaining_counts: Sequence[int],
    has_meld: bool,
) -> HandBelief:
    tables = cap_by_wait(
        model.probabilities(*features, supported_slots(remaining_counts, has_meld)),
        belief.wait_probability_raw,
    )
    return replace(
        belief,
        **{field: table for (_, field, _), table in zip(CHANNELS, tables)},
    )


def _check(policy_input, seat, model, population: str) -> None:
    if not isinstance(policy_input, PolicyInput) or not isinstance(seat, Seat):
        raise TypeError("a PolicyInput and Seat are required")
    if not isinstance(model, WaitShapeModel):
        raise TypeError("a WaitShapeModel is required")
    if model.population != population:
        raise ValueError(f"the shape model is not for the {population} population")


def estimate_riichi_wait_shape_belief(
    policy_input: PolicyInput,
    seat: Seat,
    wait_model: LogisticWaitModel,
    shape_model: WaitShapeModel,
) -> HandBelief:
    """単独リーチ者`seat`のLevel 2の`HandBelief`。`wait_probability`は`wait_model`の値のまま。

    #245の提供範囲（観測者は非リーチ、他家のリーチ者は`seat`だけ、`seat`に捨て牌がある）の外は
    `ValueError`。
    """
    _check(policy_input, seat, shape_model, "riichi")
    if not isinstance(wait_model, LogisticWaitModel):
        raise TypeError("a LogisticWaitModel is required")
    view = riichi_view(policy_input)
    if view.riichi_seat != int(seat):
        raise ValueError("the seat is not the single riichi opponent")
    return _with_shapes(
        estimate_riichi_wait_belief(policy_input, wait_model),
        shape_model,
        riichi_shape_features(view),
        view.remaining_counts,
        bool(policy_input.players[seat].melds),
    )


def estimate_open_wait_shape_belief(
    policy_input: PolicyInput,
    seat: Seat,
    wait_model: OpenWaitModel,
    shape_model: WaitShapeModel,
) -> HandBelief:
    """副露者`seat`のLevel 2の`HandBelief`。`wait_probability`は`wait_model`の値のまま。

    #259範囲1の提供範囲（リーチしておらず暗槓以外の副露がある他家）の外は`ValueError`。
    """
    _check(policy_input, seat, shape_model, "open")
    if not isinstance(wait_model, OpenWaitModel):
        raise TypeError("an OpenWaitModel is required")
    view = open_view(policy_input, int(seat))
    return _with_shapes(
        estimate_open_wait_belief(policy_input, int(seat), wait_model),
        shape_model,
        open_shape_features(view),
        view.remaining_counts,
        True,
    )


__all__ = [
    "CHANNELS",
    "CHANNEL_NAMES",
    "CONSISTENCY",
    "FEATURE_SET",
    "KOKUSHI",
    "POPULATIONS",
    "WaitShapeModel",
    "cap_by_wait",
    "estimate_open_wait_shape_belief",
    "estimate_riichi_wait_shape_belief",
    "open_shape_features",
    "riichi_shape_features",
    "supported_slots",
]
