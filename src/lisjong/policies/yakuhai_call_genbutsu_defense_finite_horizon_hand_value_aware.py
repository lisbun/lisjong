"""役牌Ponを起点に向聴改善Chi/Ponだけを許可するcall-aware Policy。

Issue #145のexperimental Policyとして、既存のno-call parentである
`GenbutsuDefenseFiniteHorizonHandValueAwarePolicy`を変更せず再利用する。
parentのdecision orchestrationを先に1回だけ実行するため、winning action、Always
Riichi、ordinary discardのselection semanticはそのまま維持される。parentが
call-responseで選んだ明示的なPassだけを、次の保守的なcall strategyで置換できる。

    CLOSED_CALL_MODE:
        役牌Pon + concealed exact pair + strict shanten improvement

    YAKUHAI_OPEN_CALL_MODE:
        Chi/Pon + strict shanten improvement

call modeは毎decision、自席の役牌PON/KAKAN PublicMeldから導出する。Policy
instanceにcross-decision stateを保持しない。call qualificationは
`Action.consumed_tiles`をexact identityで除去した後、current standard kuikaeで
許されるmandatory discardを1枚行ったstable concealed handのminimum shantenを
使う。このhypothetical discardは保存せず、actual next decisionではfresh
`legal_actions`をparentが選択する。
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware import (
    GenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
)
from lisjong.policies.genbutsu_defense_two_step_ukeire import (
    _common_genbutsu_tile_types,
    _opponent_riichi_players,
)
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _WIND_RANK,
    _seat_wind_rank,
    _yakuhai_han_value,
)
from lisjong.policy_contract.action import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    KakanAction,
    PassAction,
    PonAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import Tile, TileType, tile_sort_key

type _CallAction = ChiAction | PonAction
_KAN_ACTION_TYPES = (DaiminkanAction, AnkanAction, KakanAction)
_YAKUHAI_OPEN_MELD_KINDS = frozenset({MeldKind.PON, MeldKind.KAKAN})


class YakuhaiCallPolicyError(Exception):
    """call qualificationが入力不整合を検出してfail closedする場合。"""


@dataclass(frozen=True, slots=True)
class _CallCandidateEvaluation:
    action: _CallAction
    best_post_call_shanten: int


def _is_yakuhai(policy_input: PolicyInput, tile_type: TileType) -> bool:
    """HandValueAwareと同じ役牌semanticで1牌種を判定する。"""
    return (
        _yakuhai_han_value(
            tile_type,
            seat_wind_rank=_seat_wind_rank(policy_input),
            round_wind_rank=_WIND_RANK[policy_input.round.round_wind],
        )
        > 0
    )


def _has_open_yakuhai(policy_input: PolicyInput) -> bool:
    """自席current public meld snapshotだけからcall modeを導出する。"""
    own_melds = policy_input.players[int(policy_input.self_seat)].melds
    return any(
        meld.kind in _YAKUHAI_OPEN_MELD_KINDS
        and _is_yakuhai(policy_input, meld.tiles[0].tile_type)
        for meld in own_melds
    )


def _remove_exact_consumed_tiles(
    concealed_tiles: Sequence[Tile], consumed_tiles: Sequence[Tile]
) -> list[Tile]:
    """red distinctionを含むAction上のexact Tile identityでconsumeする。"""
    remaining = list(concealed_tiles)
    for consumed in consumed_tiles:
        for index, candidate in enumerate(remaining):
            if candidate == consumed:
                del remaining[index]
                break
        else:
            raise YakuhaiCallPolicyError(
                "call consumed_tiles contains a tile with no exact matching "
                "identity in own_hand.concealed_tiles"
            )
    return remaining


def _kuikae_forbidden_tile_types(action: _CallAction) -> frozenset[TileType]:
    """Issue #145限定のhypothetical mandatory-discard禁止牌種を返す。"""
    called_type = action.called_tile.tile_type
    forbidden = {called_type}
    if isinstance(action, PonAction):
        return frozenset(forbidden)

    sequence_ranks = sorted(
        (called_type.rank, *(tile.tile_type.rank for tile in action.consumed_tiles))
    )
    if called_type.rank == sequence_ranks[0]:
        extra_rank = sequence_ranks[-1] + 1
        if extra_rank <= 9:
            forbidden.add(TileType(called_type.category, extra_rank))
    elif called_type.rank == sequence_ranks[-1]:
        extra_rank = sequence_ranks[0] - 1
        if extra_rank >= 1:
            forbidden.add(TileType(called_type.category, extra_rank))
    return frozenset(forbidden)


def _best_post_call_shanten(
    concealed_tiles: Sequence[Tile], action: _CallAction
) -> int | None:
    """call後に許されるmandatory discardを行ったstable stateの最小向聴数。"""
    post_call_tiles = _remove_exact_consumed_tiles(
        concealed_tiles, action.consumed_tiles
    )
    forbidden_types = _kuikae_forbidden_tile_types(action)
    candidate_shanten = tuple(
        calculate_shanten([*post_call_tiles[:index], *post_call_tiles[index + 1 :]])
        for index, tile in enumerate(post_call_tiles)
        if tile.tile_type not in forbidden_types
    )
    if not candidate_shanten:
        return None
    return min(candidate_shanten)


def _call_action_sort_key(action: _CallAction) -> tuple[object, ...]:
    """legal-action入力順に依存しないChi/Ponの明示的canonical順。"""
    action_kind = 0 if isinstance(action, ChiAction) else 1
    return (
        action_kind,
        int(action.actor),
        int(action.target),
        tile_sort_key(action.called_tile),
        tuple(tile_sort_key(tile) for tile in action.consumed_tiles),
    )


def _defense_suppresses_call(policy_input: PolicyInput, current_shanten: int) -> bool:
    """被立直・非聴牌・common genbutsu所持ならcallよりfoldを優先する。"""
    riichi_players = _opponent_riichi_players(policy_input)
    if not riichi_players or current_shanten < 1:
        return False
    common_tile_types = _common_genbutsu_tile_types(riichi_players)
    return any(
        tile.tile_type in common_tile_types
        for tile in policy_input.own_hand.concealed_tiles
    )


def _qualifying_call_candidates(
    policy_input: PolicyInput,
    call_actions: tuple[_CallAction, ...],
    current_shanten: int,
) -> tuple[_CallCandidateEvaluation, ...]:
    """current call modeでstrictly improvingなChi/Ponだけを返す。"""
    concealed_tiles = policy_input.own_hand.concealed_tiles
    tile_type_counts = Counter(tile.tile_type for tile in concealed_tiles)
    open_mode = _has_open_yakuhai(policy_input)
    evaluations: list[_CallCandidateEvaluation] = []
    for action in call_actions:
        if isinstance(action, PonAction):
            if tile_type_counts[action.called_tile.tile_type] != 2:
                continue
            if not open_mode and not _is_yakuhai(
                policy_input, action.called_tile.tile_type
            ):
                continue
        elif not open_mode:
            continue

        best_post_call_shanten = _best_post_call_shanten(concealed_tiles, action)
        if (
            best_post_call_shanten is not None
            and best_post_call_shanten < current_shanten
        ):
            evaluations.append(_CallCandidateEvaluation(action, best_post_call_shanten))
    return tuple(evaluations)


class YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy(
    GenbutsuDefenseFiniteHorizonHandValueAwarePolicy
):
    """役牌起点のstrict向聴改善callを加えたstateless experimental Policy。"""

    def _decide(self, decision: DecisionContext) -> PolicyDecision:
        parent_decision = super()._decide(decision)
        if not isinstance(
            parent_decision.action,
            (PassAction, ChiAction, PonAction, *_KAN_ACTION_TYPES),
        ):
            return parent_decision

        pass_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, PassAction)
        )
        call_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, (ChiAction, PonAction))
        )
        if not call_actions:
            if pass_actions:
                return PolicyDecision(action=pass_actions[0])
            raise YakuhaiCallPolicyError(
                "no selectable non-Kan action or explicit PassAction is available"
            )

        current_shanten = calculate_shanten(decision.input.own_hand.concealed_tiles)
        if _defense_suppresses_call(decision.input, current_shanten):
            if pass_actions:
                return PolicyDecision(action=pass_actions[0])
            raise YakuhaiCallPolicyError(
                "defense gate requires PassAction, but no explicit pass is legal"
            )

        candidates = _qualifying_call_candidates(
            decision.input, call_actions, current_shanten
        )
        if candidates:
            selected = min(
                candidates,
                key=lambda candidate: (
                    candidate.best_post_call_shanten,
                    _call_action_sort_key(candidate.action),
                ),
            )
            return PolicyDecision(action=selected.action)

        if pass_actions:
            return PolicyDecision(action=pass_actions[0])
        raise YakuhaiCallPolicyError(
            "no call qualifies and no explicit PassAction is available"
        )
