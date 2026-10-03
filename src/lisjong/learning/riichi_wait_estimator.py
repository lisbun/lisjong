"""単独リーチ者の構造的待ち確率の推定（lisbun/lisjong#245、推論側）。

推論は`PolicyInput`だけを入力にする。リーチ者の手牌・ラベルなど学習専用の事実は
このmoduleへ渡さず、学習・評価は`lisjong.learning.riichi_wait_evaluation`が行う。
学習時の特徴も、ここの`wait_feature_table()`が同じ`PolicyInput`から作る。
ML runtimeには依存しない。

推定対象は、リーチ者の34牌種ごとの**構造的待ち**（`HandBelief.wait_probability`の意味。
その牌を1枚加えると手牌が完成形になる確率）である。フリテンやロンの可否とは分け、
現物・河由来の特徴があっても確率を0に固定しない（フリテンのまま待つこともある）。
ロンの可否は`lisjong.learning.riichi_deal_in_estimator`（S1）の範囲である。

特徴は`lisjong.belief.wait_shape_support`の層に沿って作る。

- 枚数制約の層: 形別の成立可能性と、構成牌の残り枚数の積（壁の度合い）
- 河の層: 現物、リーチ後に他家が切った牌、スジ、両面の反対側の牌が河にあるか
- 牌の分類・残り枚数・ドラ・リーチ宣言後の巡目

ベースライン1は牌種ごとの待ち率（Jeffreys平滑化）、ベースライン2はChampionの古典的危険度score
（固定weightの合計）をlogistic（Platt）で確率へ校正したもの。確率はすべて同じ幅
（`CLIP_EPSILON`）でclipして使う。
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
from lisjong.belief.wait_shape_support import river_relation, wait_shape_support
from lisjong.learning.riichi_deal_in_estimator import (
    RiichiView,
    _suji,
    _turn_bucket,
    riichi_view,
)
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    _classical_riichi_danger_score,
)
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import TileCategory, TileType

FEATURE_SET = "riichi-wait-features-v1"
BIAS = "bias"
SCORE = "classical_score"
CLIP_EPSILON = 1e-6
TILE_TYPES = tuple(tile_type_from_index(index) for index in range(34))

_SUPPORT_FIELD = {
    "tanki": "tanki",
    "shanpon": "shanpon",
    "kanchan": "kanchan",
    "penchan": "penchan",
    "ryanmen_low": "ryanmen_low_side",
    "ryanmen_high": "ryanmen_high_side",
}


def clip_probability(value: float) -> float:
    """全モデル共通のclip。0や1ちょうどのlog lossが発散しないようにする。"""
    return min(max(value, CLIP_EPSILON), 1.0 - CLIP_EPSILON)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    z = exp(value)
    return z / (1.0 + z)


def _remaining(view: RiichiView, tile: TileType, rank: int) -> int:
    return view.remaining_counts[tile_type_index(TileType(tile.category, rank))]


def wait_features(view: RiichiView, tile: TileType) -> tuple[tuple[str, float], ...]:
    """牌種`tile`を待つかどうかの特徴（名前と値の組）。

    古典的危険度scoreではなく、形の成立可能性と河との関係を別々の特徴にする。
    連続値の特徴（残り枚数の積）以外は、有効なら値1の離散特徴。
    """
    remaining = view.remaining_counts[tile_type_index(tile)]
    names = {
        BIAS,
        f"turn_{_turn_bucket(view.riichi_discard_count)}",
        f"remaining_self_{remaining}",
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
    if tile in view.passed_since_last_riichi_discard:
        names.add("passed_since_riichi_discard")
    if tile in view.dora:
        names.add("dora")

    if tile.category is TileCategory.HONOR:
        kind = "yakuhai" if tile in view.yakuhai else "guest"
        names.add(f"class_honor_{kind}")
    else:
        rank = tile.rank
        names.add(f"class_rank_{rank}")
        names.add(f"suji_{_suji(tile, view.river)}")
        if support.ryanmen_low_side and not relation.ryanmen_low_far_end_in_river:
            names.add("ryanmen_low_open")
        if support.ryanmen_high_side and not relation.ryanmen_high_far_end_in_river:
            names.add("ryanmen_high_open")
        # 構成牌の残り枚数の積（壁の度合い）。成立可能な形だけ値を持つ
        if support.kanchan:
            numeric.append(
                (
                    "combo_kanchan",
                    _remaining(view, tile, rank - 1)
                    * _remaining(view, tile, rank + 1)
                    / 16.0,
                )
            )
        if support.penchan:
            low, high = (1, 2) if rank == 3 else (8, 9)
            numeric.append(
                (
                    "combo_penchan",
                    _remaining(view, tile, low) * _remaining(view, tile, high) / 16.0,
                )
            )
        if support.ryanmen_low_side:
            numeric.append(
                (
                    "combo_ryanmen_low",
                    _remaining(view, tile, rank + 1)
                    * _remaining(view, tile, rank + 2)
                    / 16.0,
                )
            )
        if support.ryanmen_high_side:
            numeric.append(
                (
                    "combo_ryanmen_high",
                    _remaining(view, tile, rank - 2)
                    * _remaining(view, tile, rank - 1)
                    / 16.0,
                )
            )
    return tuple(sorted([(name, 1.0) for name in names] + numeric))


def wait_feature_table(
    policy_input: PolicyInput,
) -> tuple[tuple[tuple[str, float], ...], ...]:
    """1判断について、34牌種（canonical順）それぞれの特徴を返す。

    学習・推論の両方がこの関数を使う。入力は`PolicyInput`だけで、ラベルは別経路で結合する。
    """
    view = riichi_view(policy_input)
    return tuple(wait_features(view, tile) for tile in TILE_TYPES)


def classical_wait_score(view: RiichiView, riichi_player, tile: TileType) -> int:
    """Championの古典的危険度score（固定weightの合計）を、34牌種すべてへ適用する。

    Championは手牌にある牌（残り3枚以下）だけを評価する。手牌にない牌種で残りが4枚のとき
    だけ、同一牌のweightに残り3枚の値を使う。それ以外はChampionと同じ値である。
    """
    remaining = list(view.remaining_counts)
    index = tile_type_index(tile)
    remaining[index] = min(remaining[index], 3)
    return _classical_riichi_danger_score(tile, riichi_player, tuple(remaining)).total


@dataclass(frozen=True, slots=True)
class PrevalenceWaitModel:
    """ベースライン1: 牌種ごとの待ち率（文脈に依らない）。"""

    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.probabilities) != 34 or any(
            not 0.0 <= p <= 1.0 for p in self.probabilities
        ):
            raise ValueError("probabilities must be 34 values in [0, 1]")

    def predict(self, policy_input: PolicyInput) -> dict[TileType, float]:
        riichi_view(policy_input)  # 対象範囲の検査
        return {
            tile: clip_probability(p) for tile, p in zip(TILE_TYPES, self.probabilities)
        }


@dataclass(frozen=True, slots=True)
class ClassicalScoreWaitModel:
    """ベースライン2: 古典的危険度scoreをlogisticで待ち確率へ校正したもの。"""

    intercept: float
    slope: float

    def predict(self, policy_input: PolicyInput) -> dict[TileType, float]:
        view = riichi_view(policy_input)
        riichi_player = policy_input.players[view.riichi_seat]
        return {
            tile: clip_probability(
                _sigmoid(
                    self.intercept
                    + self.slope * classical_wait_score(view, riichi_player, tile)
                )
            )
            for tile in TILE_TYPES
        }


@dataclass(frozen=True, slots=True)
class LogisticWaitModel:
    """推定器: 特徴のlogistic回帰。現物・河由来の特徴も確率を0に固定しない。"""

    weights: tuple[tuple[str, float], ...]
    feature_set: str = FEATURE_SET

    def __post_init__(self) -> None:
        if self.feature_set != FEATURE_SET:
            raise ValueError(f"unknown feature set {self.feature_set!r}")

    def predict(self, policy_input: PolicyInput) -> dict[TileType, float]:
        weights = dict(self.weights)
        table = wait_feature_table(policy_input)
        return {
            tile: clip_probability(
                _sigmoid(sum(weights.get(name, 0.0) * v for name, v in features))
            )
            for tile, features in zip(TILE_TYPES, table)
        }


def estimate_riichi_wait_belief(
    policy_input: PolicyInput,
    model: PrevalenceWaitModel | ClassicalScoreWaitModel | LogisticWaitModel,
) -> HandBelief:
    """リーチ者の`HandBelief`（Level 1: `wait_probability`あり、形別テーブルなし）を返す。

    牌のmarginal（`expected_count` / `red_five_probability`）は既存の条件付き一様推定器の値
    をそのまま使い、待ち確率だけを`model`の予測で置き換える。他家の非公開手牌のslot数は、
    非手番の席の門前手牌枚数（13 - 3 * 副露数）とする。
    """
    view = riichi_view(policy_input)
    dealer = policy_input.round.dealer_seat
    slots = [0, 0, 0, 0]
    for seat in Seat:
        if seat != policy_input.self_seat:
            melds = policy_input.players[seat].melds
            slots[wind_index(wind_for_seat(seat, dealer))] = 13 - 3 * len(melds)
    beliefs = estimate_conditional_uniform_hand_belief(policy_input, tuple(slots))
    riichi_wind = wind_for_seat(Seat(view.riichi_seat), dealer)
    predictions = model.predict(policy_input)
    return replace(
        beliefs.hand(riichi_wind),
        wait_probability_raw=tuple(
            probability_to_raw(predictions[tile]) for tile in TILE_TYPES
        ),
    )


__all__ = [
    "BIAS",
    "CLIP_EPSILON",
    "FEATURE_SET",
    "SCORE",
    "TILE_TYPES",
    "ClassicalScoreWaitModel",
    "LogisticWaitModel",
    "PrevalenceWaitModel",
    "classical_wait_score",
    "clip_probability",
    "estimate_riichi_wait_belief",
    "wait_feature_table",
    "wait_features",
]
