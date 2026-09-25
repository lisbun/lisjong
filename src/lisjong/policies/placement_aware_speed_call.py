"""役確定型スピード鳴き・鳴き後の役保持打牌・オーラス点数状況modeを加えたPolicy。

Issue #199のexperimental Policy。current Heuristic Champion
`TargetedHonorReleaseTerminalProgressionPolicy`をexact parentとして変更せず、
次の3つの限定拡張だけを加える。

A. 役確定型スピード鳴き

    parentが明示的なPassを選び、他家リーチがない局面だけで、Tanyao /
    Honitsu（Chinitsuを含む）routeと矛盾しないChi/Ponを評価する。
    exact consume + current kuikaeを適用したmandatory discard後の最小
    route shantenが現在の向聴数より厳密に小さく、mode別の採用基準を満たす
    callだけを選ぶ。

B. 鳴き後の役保持打牌

    副露済みで役牌副露・役牌暗刻による役の確保がない手は、PUSH局面に限り
    route shantenが最小の打牌へ絞ってからparentへ渡す。parentの打牌選択は
    構造的な和了確率を優先するため、役のない副露手が么九牌を残して役なしに
    なることを防ぐ。

C. オーラス点数状況mode

    南4局以降（西入を含む）だけ、公開scoreから単独トップ目をSPEED、
    3着との差が大きい子の単独ラス目をVALUEとする。SPEEDは鳴きの採用基準を
    緩め、他家リーチを受けた非聴牌時は1向聴でも公開mechanism危険度最小へ
    絞る。VALUEは高打点の鳴きだけを採用する。同点を含む局面ではmodeを
    発動しない。

route shantenはroute外の牌を面子・塔子・雀頭に使えない浮き牌として扱う
通常形向聴数であり、actual han、expected score、call EVのいずれでもない。
打点proxyはroute翻相当（Tanyao 1 / Honitsu 2 / Chinitsu 5）と、route内で
保持する牌および副露牌のdora・赤ドラ枚数の和である。

PolicyInput以外の状態やhidden informationは使用せず、Policy instanceに
cross-decision stateを保持しない。
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.hand_evaluation import calculate_shanten
from lisjong.hand_evaluation.shanten import calculate_restricted_standard_shanten
from lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware import (
    _decide_push_fold,
    _PushFoldDecision,
)
from lisjong.policies.genbutsu_defense_two_step_ukeire import (
    _common_genbutsu_tile_types,
    _opponent_riichi_players,
)
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    _classical_riichi_danger_score,
)
from lisjong.policies.targeted_honor_release_terminal_progression import (
    TargetedHonorReleaseTerminalProgressionPolicy,
)
from lisjong.policies.value_aware_two_step_ukeire import (
    _retained_concealed_dora_count,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    _call_action_sort_key,
    _CallAction,
    _has_open_yakuhai,
    _is_yakuhai,
    _post_call_stable_hands,
)
from lisjong.policy_contract.action import (
    ChiAction,
    DiscardAction,
    PassAction,
    PonAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind
from lisjong.structural_efficiency import post_discard_concealed_hand

_SUITS = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)
_HONOR_TYPES = frozenset(TileType(TileCategory.HONOR, rank) for rank in range(1, 8))
_VALUE_MODE_MINIMUM_GAP = 4000
"""VALUE modeを発動する、ラス目子と3着の最小点差（この値を超えると発動）。"""


class GameSituationMode(Enum):
    """公開score / round stateだけから導出する点数状況mode。"""

    NORMAL = auto()
    ALL_LAST_TOP_SPEED = auto()
    ALL_LAST_LAST_VALUE = auto()


@dataclass(frozen=True, slots=True)
class _YakuRoute:
    """副露手で目指す役route。usable外の牌はrouteの面子に使えない。"""

    usable_tile_types: frozenset[TileType]
    han: int


_TANYAO_ROUTE = _YakuRoute(
    usable_tile_types=frozenset(
        TileType(suit, rank) for suit in _SUITS for rank in range(2, 9)
    ),
    han=1,
)
_HONITSU_ROUTES = tuple(
    _YakuRoute(
        usable_tile_types=frozenset(TileType(suit, rank) for rank in range(1, 10))
        | _HONOR_TYPES,
        han=2,
    )
    for suit in _SUITS
)
_ALL_ROUTES = (_TANYAO_ROUTE, *_HONITSU_ROUTES)
_CHINITSU_HAN = 5


@dataclass(frozen=True, slots=True)
class _CallAcceptance:
    """modeごとのスピード鳴き採用基準。"""

    maximum_route_shanten: int
    closed_tenpai_minimum_value: int
    closed_non_tenpai_minimum_value: int
    open_minimum_value: int


_CALL_ACCEPTANCE = {
    GameSituationMode.NORMAL: _CallAcceptance(
        maximum_route_shanten=1,
        closed_tenpai_minimum_value=0,
        closed_non_tenpai_minimum_value=2,
        open_minimum_value=0,
    ),
    GameSituationMode.ALL_LAST_TOP_SPEED: _CallAcceptance(
        maximum_route_shanten=2,
        closed_tenpai_minimum_value=0,
        closed_non_tenpai_minimum_value=0,
        open_minimum_value=0,
    ),
    GameSituationMode.ALL_LAST_LAST_VALUE: _CallAcceptance(
        maximum_route_shanten=1,
        closed_tenpai_minimum_value=3,
        closed_non_tenpai_minimum_value=3,
        open_minimum_value=3,
    ),
}


@dataclass(frozen=True, slots=True)
class _SpeedCallCandidate:
    action: _CallAction
    route_shanten: int
    value: int


def _is_all_last(policy_input: PolicyInput) -> bool:
    """南4局、または西入以降の局をオーラス相当として扱う。"""
    round_state = policy_input.round
    if round_state.round_wind in (Wind.WEST, Wind.NORTH):
        return True
    return round_state.round_wind is Wind.SOUTH and round_state.hand_number == 4


def situation_mode(policy_input: PolicyInput) -> GameSituationMode:
    """公開scoreから点数状況modeを返す。同点が絡むrank判定は行わない。"""
    if not _is_all_last(policy_input):
        return GameSituationMode.NORMAL
    self_index = int(policy_input.self_seat)
    own_score = policy_input.players[self_index].score
    other_scores = tuple(
        player.score
        for index, player in enumerate(policy_input.players)
        if index != self_index
    )
    if all(own_score > score for score in other_scores):
        return GameSituationMode.ALL_LAST_TOP_SPEED
    if (
        policy_input.self_seat != policy_input.round.dealer_seat
        and all(own_score < score for score in other_scores)
        and min(other_scores) - own_score > _VALUE_MODE_MINIMUM_GAP
    ):
        return GameSituationMode.ALL_LAST_LAST_VALUE
    return GameSituationMode.NORMAL


def _own_meld_tiles(policy_input: PolicyInput) -> tuple[Tile, ...]:
    own_melds = policy_input.players[int(policy_input.self_seat)].melds
    return tuple(tile for meld in own_melds for tile in meld.tiles)


def _is_closed_hand(policy_input: PolicyInput) -> bool:
    own_melds = policy_input.players[int(policy_input.self_seat)].melds
    return all(meld.kind is MeldKind.ANKAN for meld in own_melds)


def _compatible_routes(meld_tiles: Sequence[Tile]) -> tuple[_YakuRoute, ...]:
    return tuple(
        route
        for route in _ALL_ROUTES
        if all(tile.tile_type in route.usable_tile_types for tile in meld_tiles)
    )


def _has_concealed_yakuhai_triplet(
    policy_input: PolicyInput, concealed_tiles: Sequence[Tile]
) -> bool:
    counts = Counter(tile.tile_type for tile in concealed_tiles)
    return any(
        count >= 3 and _is_yakuhai(policy_input, tile_type)
        for tile_type, count in counts.items()
    )


def _route_value(
    route: _YakuRoute,
    concealed_tiles: Sequence[Tile],
    meld_tiles: Sequence[Tile],
    dora_indicators: Sequence[Tile],
) -> int:
    """route翻相当 + route内で保持する牌・副露牌のdora / 赤ドラ枚数。"""
    retained = (
        *(
            tile
            for tile in concealed_tiles
            if tile.tile_type in route.usable_tile_types
        ),
        *meld_tiles,
    )
    han = route.han
    if route is not _TANYAO_ROUTE and all(
        tile.tile_type.category is not TileCategory.HONOR for tile in retained
    ):
        han = _CHINITSU_HAN
    return han + _retained_concealed_dora_count(retained, dora_indicators)


def _best_route_evaluation(
    concealed_tiles: Sequence[Tile],
    meld_tiles: Sequence[Tile],
    routes: Sequence[_YakuRoute],
    dora_indicators: Sequence[Tile],
) -> tuple[int, int]:
    """最小route shantenと、その最小を達成するrouteの最大打点proxyを返す。"""
    evaluations = tuple(
        (
            calculate_restricted_standard_shanten(
                concealed_tiles, route.usable_tile_types
            ),
            _route_value(route, concealed_tiles, meld_tiles, dora_indicators),
        )
        for route in routes
    )
    best_shanten = min(shanten for shanten, _ in evaluations)
    best_value = max(value for shanten, value in evaluations if shanten == best_shanten)
    return best_shanten, best_value


def _speed_call_candidates(
    policy_input: PolicyInput,
    call_actions: tuple[_CallAction, ...],
    mode: GameSituationMode,
) -> tuple[_SpeedCallCandidate, ...]:
    """current modeの採用基準を満たす役確定型Chi/Ponだけを返す。"""
    concealed_tiles = policy_input.own_hand.concealed_tiles
    current_meld_tiles = _own_meld_tiles(policy_input)
    closed = _is_closed_hand(policy_input)
    dora_indicators = policy_input.round.dora_indicators
    if closed:
        current_shanten = calculate_shanten(concealed_tiles)
    else:
        current_routes = _compatible_routes(current_meld_tiles)
        if not current_routes:
            return ()
        current_shanten, _ = _best_route_evaluation(
            concealed_tiles, current_meld_tiles, current_routes, dora_indicators
        )

    acceptance = _CALL_ACCEPTANCE[mode]
    tile_type_counts = Counter(tile.tile_type for tile in concealed_tiles)
    candidates: list[_SpeedCallCandidate] = []
    for action in call_actions:
        if (
            isinstance(action, PonAction)
            and tile_type_counts[action.called_tile.tile_type] != 2
        ):
            continue
        meld_tiles = (*current_meld_tiles, *action.consumed_tiles, action.called_tile)
        routes = _compatible_routes(meld_tiles)
        if not routes:
            continue
        stable_hands = _post_call_stable_hands(concealed_tiles, action)
        if not stable_hands:
            continue
        route_shanten, value = min(
            (
                _best_route_evaluation(stable, meld_tiles, routes, dora_indicators)
                for stable in stable_hands
            ),
            key=lambda evaluation: (evaluation[0], -evaluation[1]),
        )
        if route_shanten >= current_shanten:
            continue
        if route_shanten > acceptance.maximum_route_shanten:
            continue
        if not closed:
            minimum_value = acceptance.open_minimum_value
        elif route_shanten == 0:
            minimum_value = acceptance.closed_tenpai_minimum_value
        else:
            minimum_value = acceptance.closed_non_tenpai_minimum_value
        if value < minimum_value:
            continue
        candidates.append(_SpeedCallCandidate(action, route_shanten, value))
    return tuple(candidates)


def _route_preserving_actions(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, ...]:
    """役の確保がない副露手のPUSH局面だけ、route shantenが最小の打牌へ絞る。"""
    if _is_closed_hand(policy_input) or _has_open_yakuhai(policy_input):
        return discard_actions
    meld_tiles = _own_meld_tiles(policy_input)
    routes = _compatible_routes(meld_tiles)
    if not routes:
        return discard_actions
    if _decide_push_fold(policy_input, discard_actions) is _PushFoldDecision.FOLD:
        return discard_actions

    concealed_tiles = policy_input.own_hand.concealed_tiles
    evaluated: list[tuple[DiscardAction, int]] = []
    for action in discard_actions:
        post_discard_hand = post_discard_concealed_hand(concealed_tiles, action.tile)
        if _has_concealed_yakuhai_triplet(policy_input, post_discard_hand):
            route_shanten = calculate_shanten(post_discard_hand)
        else:
            route_shanten, _ = _best_route_evaluation(
                post_discard_hand, meld_tiles, routes, ()
            )
        evaluated.append((action, route_shanten))
    minimum = min(route_shanten for _, route_shanten in evaluated)
    return tuple(
        action for action, route_shanten in evaluated if route_shanten == minimum
    )


def _top_fold_actions(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, ...]:
    """オーラストップ目の非聴牌FOLDで共通現物がなければ最小危険度へ絞る。"""
    riichi_players = _opponent_riichi_players(policy_input)
    if not riichi_players:
        return discard_actions
    if _decide_push_fold(policy_input, discard_actions) is _PushFoldDecision.PUSH:
        return discard_actions
    common_genbutsu = _common_genbutsu_tile_types(riichi_players)
    if any(action.tile.tile_type in common_genbutsu for action in discard_actions):
        return discard_actions

    remaining_counts = derive_remaining_tile_inventory(
        policy_input
    ).remaining_tile_counts
    dangers = tuple(
        (
            action,
            max(
                _classical_riichi_danger_score(
                    action.tile.tile_type, opponent, remaining_counts
                ).total
                for opponent in riichi_players
            ),
        )
        for action in discard_actions
    )
    minimum = min(danger for _, danger in dangers)
    return tuple(action for action, danger in dangers if danger == minimum)


class PlacementAwareSpeedCallPolicy(TargetedHonorReleaseTerminalProgressionPolicy):
    """Issue #199: 役確定型スピード鳴き + 役保持打牌 + オーラスmode。"""

    def _decide(self, decision: DecisionContext) -> PolicyDecision:
        parent_decision = super()._decide(decision)
        if not isinstance(parent_decision.action, PassAction):
            return parent_decision

        call_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, (ChiAction, PonAction))
        )
        if not call_actions or _opponent_riichi_players(decision.input):
            return parent_decision
        if _has_open_yakuhai(decision.input):
            return parent_decision

        candidates = _speed_call_candidates(
            decision.input, call_actions, situation_mode(decision.input)
        )
        if not candidates:
            return parent_decision
        selected = min(
            candidates,
            key=lambda candidate: (
                candidate.route_shanten,
                -candidate.value,
                _call_action_sort_key(candidate.action),
            ),
        )
        return PolicyDecision(action=selected.action)

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        eligible_actions = _route_preserving_actions(policy_input, discard_actions)
        if situation_mode(policy_input) is GameSituationMode.ALL_LAST_TOP_SPEED:
            eligible_actions = _top_fold_actions(policy_input, eligible_actions)
        return super()._decide_discard(policy_input, eligible_actions)


__all__ = ["GameSituationMode", "PlacementAwareSpeedCallPolicy", "situation_mode"]
