"""リーチを受けた1向聴の押し引きを、役を含む打点推定で判断する実験候補（#384）。

仕様は`docs/riichi-response-hand-value-candidate.md`。未昇格の実験候補で、Championと
#377の候補（dora版）は書き換えない。exact parentは`PlacementAwareSpeedCallPolicy`。

#377の候補は手の価値をdora枚数で測った。この候補は、聴牌した場合のロン点を
`evaluate_win`で計算し、門前と副露で判断を分ける。

## 打点推定 V（`estimate_one_shanten_hand_value`）

判断時点の公開情報（自分の手牌・副露、見えている牌、ドラ表示牌、場風・自風、親）だけを使う。

1. 打牌後も1向聴の各打牌 d について、打牌後の純手牌を作る
2. 見えていない牌種 t（残り n_t > 0）を引いて聴牌になるとき、聴牌を保つ打牌のうち
   3の値が最大のものを選ぶ
3. 聴牌形の値は、待ち牌ごとのロン点（`winner_points`、本場・供託を含まない）を、
   待ち牌の見えない残り枚数（引いた t を1枚除く）で加重平均した値。
   門前はリーチしてロンした点（裏ドラ`URA_DORA_EXCLUDED`、一発なし）、副露はロンした点。
   役がなければ0点。待ちの残りが0なら0点。振聴は考慮しない
4. V(d) は2の値を n_t で加重平均した値（聴牌にならない牌は除く。1枚もなければ0）
5. V = max_d V(d)。自席が親なら 2/3 を掛けて子の点に換算する

値は`Fraction`で返し、浮動小数点の誤差を持たない。

## 判断規則

ゲートは#377と同じ（他家リーチあり、点数状況mode `NORMAL`、どの打牌でも聴牌にならず
打牌後の最小向聴数が1）。それ以外の判断はparentと同じ。

    押し返し（門前）: 共通現物あり、門前、V >= push_closed_minimum_points、
                      残り山 >= push_minimum_live_wall
    押し返し（副露）: 共通現物あり、副露、V >= push_open_minimum_points、
                      残り山 >= push_minimum_live_wall
    降り:             共通現物なし、V <= fold_maximum_points

押し返しは他家リーチがないものとしてparentに打牌を選ばせ、降りは古典的危険度score
（全リーチ者に対する最大値）が最小の打牌へ絞ってparentへ渡す（#377と同じ動作）。
閾値がNoneの規則は使わない。

PolicyInput以外の情報（他家の手牌・山・裏ドラ）は使わず、instanceに状態を持たない。
"""

from dataclasses import dataclass
from fractions import Fraction

from lisjong.belief.canonical_axes import (
    tile_type_from_index,
    tile_type_index,
    wind_for_seat,
)
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.hand_evaluation import calculate_shanten
from lisjong.hand_evaluation.scoring import (
    URA_DORA_EXCLUDED,
    EvaluationStatus,
    RiichiStatus,
    ScoringMeld,
    UraDoraIndicators,
    WinContext,
    WinMethod,
    WinningHand,
    WinSituation,
    evaluate_win,
)
from lisjong.policies.genbutsu_defense_two_step_ukeire import (
    _common_genbutsu_tile_types,
    _opponent_riichi_players,
)
from lisjong.policies.placement_aware_speed_call import (
    GameSituationMode,
    PlacementAwareSpeedCallPolicy,
    _is_closed_hand,
    situation_mode,
)
from lisjong.policies.riichi_dama_branch import winning_tile_for
from lisjong.policies.riichi_response_placement_aware_speed_call import (
    _minimum_danger_actions,
    _without_opponent_riichi,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import Tile
from lisjong.structural_efficiency import (
    effective_tile_types,
    post_discard_concealed_hand,
)

HAND_VALUE_ESTIMATE_VERSION = "riichi-response-hand-value-estimate-v1"
RIICHI_RESPONSE_HAND_VALUE_SEARCH_SPACE_VERSION = (
    "riichi-response-hand-value-search-space-v2"
)
_IDENTITY_PREFIX = "placement-aware-speed-call-riichi-response-hand-value-v1"
_TILE_TYPE_COUNT = 34
_DEALER_FACTOR = Fraction(2, 3)


class HandValueEstimateError(Exception):
    """打点推定の入力が推定の前提（1向聴など）を満たさない。"""


def _tile_key(tile: Tile) -> tuple[int, bool]:
    return tile_type_index(tile.tile_type), tile.is_red


def _distinct(tiles: tuple[Tile, ...]) -> tuple[Tile, ...]:
    seen: list[Tile] = []
    for tile in tiles:
        if tile not in seen:
            seen.append(tile)
    return tuple(seen)


def estimate_one_shanten_hand_value(
    policy_input: PolicyInput, discard_actions: tuple[DiscardAction, ...]
) -> Fraction:
    """打牌後も1向聴の打牌について、聴牌した場合のロン点の推定（子換算）の最大値。

    打牌後に1向聴になる打牌がなければ`HandValueEstimateError`。
    """
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("policy_input must be a PolicyInput")
    seat = policy_input.self_seat
    own = policy_input.players[int(seat)]
    round_state = policy_input.round
    closed = _is_closed_hand(policy_input)
    remaining = derive_remaining_tile_inventory(policy_input).remaining_tile_counts
    melds = tuple(ScoringMeld(meld.kind, meld.tiles) for meld in own.melds)
    meld_tiles = tuple(tile for meld in own.melds for tile in meld.tiles)
    context = WinContext(
        method=WinMethod.RON,
        seat_wind=wind_for_seat(seat, round_state.dealer_seat),
        prevailing_wind=round_state.round_wind,
        riichi=RiichiStatus.RIICHI if closed else RiichiStatus.NONE,
        is_ippatsu=False,
        situation=WinSituation.NORMAL,
        dora_indicators=round_state.dora_indicators,
        ura_dora=URA_DORA_EXCLUDED if closed else UraDoraIndicators(()),
    )
    fixed_physical = (*meld_tiles, *round_state.dora_indicators)
    points_cache: dict[tuple, int] = {}

    def ron_points(hand: tuple[Tile, ...], wait: Tile) -> int:
        key = (tuple(sorted(map(_tile_key, hand))), _tile_key(wait))
        if key not in points_cache:
            try:
                evaluation = evaluate_win(WinningHand(hand, melds, wait), context)
            except ValueError as error:
                raise HandValueEstimateError(
                    f"the wait {wait} cannot be evaluated: {error}"
                ) from None
            if evaluation.status is EvaluationStatus.NO_YAKU:
                points_cache[key] = 0
            elif evaluation.status is EvaluationStatus.SCORED:
                points_cache[key] = evaluation.score.winner_points
            else:
                raise HandValueEstimateError("a structural wait does not complete")
        return points_cache[key]

    def tenpai_value(hand: tuple[Tile, ...], drawn_index: int) -> Fraction:
        physical = (*hand, *fixed_physical)
        weight = 0
        total = 0
        for wait_type in effective_tile_types(hand, 0):
            index = tile_type_index(wait_type)
            count = remaining[index] - (1 if index == drawn_index else 0)
            if count <= 0:
                continue
            weight += count
            total += count * ron_points(hand, winning_tile_for(wait_type, physical))
        return Fraction(total, weight) if weight else Fraction(0)

    def one_shanten_value(hand: tuple[Tile, ...]) -> Fraction:
        weight = 0
        total = Fraction(0)
        for index in range(_TILE_TYPE_COUNT):
            count = remaining[index]
            if count <= 0:
                continue
            drawn = winning_tile_for(
                tile_type_from_index(index), (*hand, *fixed_physical)
            )
            drawn_hand = (*hand, drawn)
            best: Fraction | None = None
            for tile in _distinct(drawn_hand):
                if tile == drawn:
                    continue
                tenpai = post_discard_concealed_hand(drawn_hand, tile)
                if calculate_shanten(tenpai) != 0:
                    continue
                value = tenpai_value(tenpai, index)
                if best is None or value > best:
                    best = value
            if best is not None:
                weight += count
                total += count * best
        return total / weight if weight else Fraction(0)

    concealed = policy_input.own_hand.concealed_tiles
    estimate: Fraction | None = None
    for action in discard_actions:
        if not isinstance(action, DiscardAction):
            raise TypeError("discard_actions must contain only DiscardAction")
    for tile in _distinct(tuple(action.tile for action in discard_actions)):
        hand = post_discard_concealed_hand(concealed, tile)
        if calculate_shanten(hand) != 1:
            continue
        value = one_shanten_value(hand)
        if estimate is None or value > estimate:
            estimate = value
    if estimate is None:
        raise HandValueEstimateError("no discard keeps the hand at one shanten")
    if seat == round_state.dealer_seat:
        estimate *= _DEALER_FACTOR
    return estimate


@dataclass(frozen=True, slots=True)
class RiichiResponseHandValueParameters:
    """押し返し（門前・副露）と降りの閾値（子換算の点）。Noneはその規則を使わない。"""

    push_closed_minimum_points: int | None
    push_open_minimum_points: int | None
    push_minimum_live_wall: int
    fold_maximum_points: int | None

    def __post_init__(self) -> None:
        for name in (
            "push_closed_minimum_points",
            "push_open_minimum_points",
            "fold_maximum_points",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be None or a non-negative int")
        if type(self.push_minimum_live_wall) is not int or (
            self.push_minimum_live_wall < 0
        ):
            raise ValueError("push_minimum_live_wall must be a non-negative int")

    @property
    def identity(self) -> str:
        """パラメータ値だけから決まる候補identity（class名から導出しない）。"""

        def text(value: int | None) -> str:
            return "none" if value is None else str(value)

        return (
            f"{_IDENTITY_PREFIX}:push-closed-{text(self.push_closed_minimum_points)}"
            f":push-open-{text(self.push_open_minimum_points)}"
            f":push-wall-{self.push_minimum_live_wall}"
            f":fold-{text(self.fold_maximum_points)}"
        )


_SEARCH_PUSH_CLOSED = (5200, 8000)
_SEARCH_PUSH_OPEN = (None, 8000)
_SEARCH_FOLD = (2000, 3900)
_SEARCH_PUSH_MINIMUM_LIVE_WALL = 24


def riichi_response_hand_value_search_space() -> tuple[
    RiichiResponseHandValueParameters, ...
]:
    """探索空間v2の8候補を固定順（門前押し、副露押し、降りの順に外側から）で返す。"""
    return tuple(
        RiichiResponseHandValueParameters(
            push_closed_minimum_points=push_closed,
            push_open_minimum_points=push_open,
            push_minimum_live_wall=_SEARCH_PUSH_MINIMUM_LIVE_WALL,
            fold_maximum_points=fold,
        )
        for push_closed in _SEARCH_PUSH_CLOSED
        for push_open in _SEARCH_PUSH_OPEN
        for fold in _SEARCH_FOLD
    )


class RiichiResponseHandValuePlacementAwareSpeedCallPolicy(
    PlacementAwareSpeedCallPolicy
):
    """Issue #384: リーチを受けた1向聴の、打点推定による押し返し / 降り。"""

    def __init__(self, parameters: RiichiResponseHandValueParameters) -> None:
        if not isinstance(parameters, RiichiResponseHandValueParameters):
            raise TypeError("parameters must be RiichiResponseHandValueParameters")
        super().__init__()
        self._parameters = parameters

    @property
    def parameters(self) -> RiichiResponseHandValueParameters:
        return self._parameters

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        riichi_players = _opponent_riichi_players(policy_input)
        if (
            not riichi_players
            or situation_mode(policy_input) is not GameSituationMode.NORMAL
        ):
            return super()._decide_discard(policy_input, discard_actions)
        concealed_tiles = policy_input.own_hand.concealed_tiles
        minimum_shanten = min(
            calculate_shanten(post_discard_concealed_hand(concealed_tiles, action.tile))
            for action in discard_actions
        )
        if minimum_shanten != 1:
            return super()._decide_discard(policy_input, discard_actions)

        parameters = self._parameters
        genbutsu = _common_genbutsu_tile_types(riichi_players)
        has_genbutsu = any(
            action.tile.tile_type in genbutsu for action in discard_actions
        )
        if has_genbutsu:
            push_minimum = (
                parameters.push_closed_minimum_points
                if _is_closed_hand(policy_input)
                else parameters.push_open_minimum_points
            )
            if (
                push_minimum is not None
                and policy_input.round.live_wall_tiles_remaining
                >= parameters.push_minimum_live_wall
                and estimate_one_shanten_hand_value(policy_input, discard_actions)
                >= push_minimum
            ):
                return super()._decide_discard(
                    _without_opponent_riichi(policy_input), discard_actions
                )
        elif (
            parameters.fold_maximum_points is not None
            and estimate_one_shanten_hand_value(policy_input, discard_actions)
            <= parameters.fold_maximum_points
        ):
            return super()._decide_discard(
                policy_input, _minimum_danger_actions(policy_input, discard_actions)
            )
        return super()._decide_discard(policy_input, discard_actions)


__all__ = [
    "HAND_VALUE_ESTIMATE_VERSION",
    "RIICHI_RESPONSE_HAND_VALUE_SEARCH_SPACE_VERSION",
    "HandValueEstimateError",
    "RiichiResponseHandValueParameters",
    "RiichiResponseHandValuePlacementAwareSpeedCallPolicy",
    "estimate_one_shanten_hand_value",
    "riichi_response_hand_value_search_space",
]
