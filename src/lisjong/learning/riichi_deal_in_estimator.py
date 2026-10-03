"""単独リーチ者に対する打牌候補ごとの放銃確率推定（#237 S1、推論側）。

推論はplayer-safeな`PolicyInput`だけを入力にする。ラベル・リーチ者の手牌など
学習専用の事実はこのmoduleへ渡さず、学習・評価は
`lisjong.learning.riichi_deal_in_evaluation`が行う。ML runtimeには依存しない。

構造的に安全な牌（ラベルAが定義上必ず偽になる牌）は確率0とする。

- 現物: リーチ者自身の捨て牌にある牌種（捨て牌フリテン）
- リーチ者の最後の打牌より後に他家が切った牌種。最後の打牌はリーチ宣言打牌以後なので、
  その牌が待ちなら見逃しフリテンになり、待ちでなければロンできない

推定器（`LogisticModel`）は、それ以外の候補を特徴（古典的危険度scoreの数値と区分、
牌の分類×筋、残り枚数、形の成立可能性、ドラ、リーチ者の巡目）のlogistic回帰で推定する。
ベースライン1は現物=0・それ以外一定、ベースライン2は既存の古典的危険度scoreを
logistic（Platt）で確率へ校正したもの。
"""

from dataclasses import dataclass
from math import exp

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _WIND_RANK,
    _yakuhai_han_value,
)
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    _classical_riichi_danger_score,
)
from lisjong.policies.value_aware_two_step_ukeire import _dora_tile_type
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.tile import TileCategory, TileType

FEATURE_SET = "riichi-deal-in-features-v3"
BIAS = "bias"
SCORE = "classical_score"


@dataclass(frozen=True, slots=True)
class RiichiView:
    """1判断について、候補に依らない公開情報の要約。"""

    riichi_seat: int
    genbutsu: frozenset[TileType]
    passed_since_last_riichi_discard: frozenset[TileType]
    river: frozenset[TileType]
    remaining_counts: tuple[int, ...]
    dora: frozenset[TileType]
    riichi_discard_count: int
    yakuhai: frozenset[TileType]

    @property
    def structurally_safe(self) -> frozenset[TileType]:
        return self.genbutsu | self.passed_since_last_riichi_discard


def riichi_view(policy_input: PolicyInput) -> RiichiView:
    """対象範囲（自分は非リーチ・他家1人がリーチ）の判断から要約を作る。"""
    seat = int(policy_input.self_seat)
    if policy_input.players[seat].riichi is not RiichiState.NONE:
        raise ValueError("the decider is in riichi")
    opponents = [
        index
        for index, player in enumerate(policy_input.players)
        if index != seat and player.riichi is not RiichiState.NONE
    ]
    if len(opponents) != 1:
        raise ValueError("not exactly one riichi opponent")
    riichi_seat = opponents[0]
    riichi_player = policy_input.players[riichi_seat]
    if not riichi_player.discards:
        raise ValueError("the riichi player has no discard")
    last_order = max(discard.order for discard in riichi_player.discards)
    passed = frozenset(
        discard.tile.tile_type
        for index, player in enumerate(policy_input.players)
        if index != riichi_seat
        for discard in player.discards
        if discard.order > last_order
    )
    round_state = policy_input.round
    seat_wind_rank = (riichi_seat - int(round_state.dealer_seat)) % 4 + 1
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
    river = frozenset(discard.tile.tile_type for discard in riichi_player.discards)
    return RiichiView(
        riichi_seat=riichi_seat,
        genbutsu=river,
        passed_since_last_riichi_discard=passed,
        river=river,
        remaining_counts=derive_remaining_tile_inventory(
            policy_input
        ).remaining_tile_counts,
        dora=frozenset(
            _dora_tile_type(indicator.tile_type)
            for indicator in round_state.dora_indicators
        ),
        riichi_discard_count=len(riichi_player.discards),
        yakuhai=yakuhai,
    )


def classical_score(policy_input: PolicyInput, view: RiichiView, tile: TileType) -> int:
    """既存の古典的危険度score（固定weightの相対値）。"""
    return _classical_riichi_danger_score(
        tile, policy_input.players[view.riichi_seat], view.remaining_counts
    ).total


def _rank_class(rank: int) -> str:
    return {1: "19", 9: "19", 2: "28", 8: "28", 3: "37", 7: "37"}.get(rank, "456")


def _suji(tile: TileType, river: frozenset[TileType]) -> str:
    def seen(rank: int) -> bool:
        return 1 <= rank <= 9 and TileType(tile.category, rank) in river

    sides = [rank for rank in (tile.rank - 3, tile.rank + 3) if 1 <= rank <= 9]
    hit = sum(seen(rank) for rank in sides)
    if hit == len(sides):
        return "full"
    return "half" if hit else "none"


def _score_bucket(score: int) -> str:
    for upper, name in ((0, "0"), (3, "1-3"), (8, "4-8"), (13, "9-13"), (20, "14-20")):
        if score <= upper:
            return name
    return "21+"


def _turn_bucket(count: int) -> str:
    if count <= 6:
        return "early"
    return "middle" if count <= 11 else "late"


def candidate_features(
    policy_input: PolicyInput, view: RiichiView, tile: TileType
) -> tuple[tuple[str, float], ...]:
    """構造的に安全でない候補の特徴（名前と値の組）。

    古典的危険度score（10で割った数値）以外は、有効なら値1の離散特徴。
    """
    remaining = view.remaining_counts[tile_type_index(tile)]
    names = {BIAS, f"turn_{_turn_bucket(view.riichi_discard_count)}"}
    if tile in view.dora:
        names.add("dora")
    score = classical_score(policy_input, view, tile)
    names.add(f"classical_{_score_bucket(score)}")
    if tile.category is TileCategory.HONOR:
        kind = "yakuhai" if tile in view.yakuhai else "guest"
        names.add(f"honor_{kind}_remaining_{remaining}")
        return _with_score(names, score)
    names.add(f"number_{_rank_class(tile.rank)}_suji_{_suji(tile, view.river)}")
    names.add(f"number_remaining_{remaining}")
    breakdown = _classical_riichi_danger_score(
        tile, policy_input.players[view.riichi_seat], view.remaining_counts
    )
    for name in ("penchan", "kanchan", "ryanmen_low_side", "ryanmen_high_side"):
        if getattr(breakdown, name):
            names.add(f"shape_{name}")
    return _with_score(names, score)


def _with_score(names: set[str], score: int) -> tuple[tuple[str, float], ...]:
    return tuple(sorted([(name, 1.0) for name in names] + [(SCORE, score / 10.0)]))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    z = exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True, slots=True)
class ConstantModel:
    """ベースライン1: 現物=0、それ以外は一定。"""

    probability: float

    def predict(
        self, policy_input: PolicyInput, candidates: tuple[TileType, ...]
    ) -> dict[TileType, float]:
        view = riichi_view(policy_input)
        return {
            tile: 0.0 if tile in view.genbutsu else self.probability
            for tile in candidates
        }


@dataclass(frozen=True, slots=True)
class ClassicalScoreModel:
    """ベースライン2: 古典的危険度scoreをlogisticで確率へ校正したもの。"""

    intercept: float
    slope: float

    def predict(
        self, policy_input: PolicyInput, candidates: tuple[TileType, ...]
    ) -> dict[TileType, float]:
        view = riichi_view(policy_input)
        return {
            tile: _sigmoid(
                self.intercept + self.slope * classical_score(policy_input, view, tile)
            )
            for tile in candidates
        }


@dataclass(frozen=True, slots=True)
class LogisticModel:
    """推定器: 構造的に安全な牌は0、それ以外は離散特徴のlogistic回帰。"""

    weights: tuple[tuple[str, float], ...]
    feature_set: str = FEATURE_SET

    def __post_init__(self) -> None:
        if self.feature_set != FEATURE_SET:
            raise ValueError(f"unknown feature set {self.feature_set!r}")

    def predict(
        self, policy_input: PolicyInput, candidates: tuple[TileType, ...]
    ) -> dict[TileType, float]:
        view = riichi_view(policy_input)
        weights = dict(self.weights)
        return {
            tile: 0.0
            if tile in view.structurally_safe
            else _sigmoid(
                sum(
                    weights.get(name, 0.0) * value
                    for name, value in candidate_features(policy_input, view, tile)
                )
            )
            for tile in candidates
        }


__all__ = [
    "BIAS",
    "FEATURE_SET",
    "ClassicalScoreModel",
    "ConstantModel",
    "LogisticModel",
    "RiichiView",
    "SCORE",
    "candidate_features",
    "classical_score",
    "riichi_view",
]
