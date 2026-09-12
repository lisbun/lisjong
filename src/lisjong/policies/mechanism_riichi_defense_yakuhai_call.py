"""公開情報だけのwait mechanism危険度で遠い手の打牌を絞るPolicy。

Issue #163のexperimental Policy。exact current ``yakuhai-call`` parentを変更せず、
他家リーチあり・全legal discard後の最小向聴数が2以上・common genbutsuなし、
の3条件が揃った通常打牌だけを拡張する。危険度は確率ではなく、same-tile、
penchan、kanchan、ryanmenの成立可能性へ固定weightを与えたrelative scoreである。
learned HandBelief、one-chanceの段階weight、chiitoitsu / kokushi専用mechanism、
calibrated放銃確率は扱わない。
"""

from dataclasses import dataclass

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware import (
    _evaluate_and_choose_discard as _parent_offensive_evaluate_and_choose_discard,
)
from lisjong.policies.genbutsu_defense_two_step_ukeire import (
    _common_genbutsu_tile_types,
    _opponent_riichi_players,
)
from lisjong.policies.two_step_ukeire import (
    _DecisionShantenEvaluator,
    _evaluate_post_discard_hands,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import TileCategory, TileType


@dataclass(frozen=True, slots=True)
class _ClassicalRiichiDangerBreakdown:
    """1 opponent・1基礎牌種のrelative mechanism score内訳。"""

    same_tile: int
    penchan: int
    kanchan: int
    ryanmen_low_side: int
    ryanmen_high_side: int

    @property
    def total(self) -> int:
        return (
            self.same_tile
            + self.penchan
            + self.kanchan
            + self.ryanmen_low_side
            + self.ryanmen_high_side
        )


_ZERO_DANGER = _ClassicalRiichiDangerBreakdown(0, 0, 0, 0, 0)


def _same_tile_contribution(tile_type: TileType, remaining_count: int) -> int:
    if remaining_count == 0:
        return 0
    if remaining_count == 1:
        return 1
    if remaining_count == 2:
        return 3
    if remaining_count == 3:
        return 8 if tile_type.category is TileCategory.HONOR else 3
    raise ValueError(
        "candidate remaining count must be between 0 and 3 after exact accounting"
    )


def _classical_riichi_danger_score(
    candidate: TileType,
    opponent: PlayerPublicState,
    remaining_tile_counts: tuple[int, ...],
) -> _ClassicalRiichiDangerBreakdown:
    """公開河とremaining inventoryから固定v1 relative scoreを返す。"""
    opponent_river = frozenset(discard.tile.tile_type for discard in opponent.discards)
    if candidate in opponent_river:
        return _ZERO_DANGER

    def remaining(rank: int) -> int:
        return remaining_tile_counts[
            tile_type_index(TileType(candidate.category, rank))
        ]

    same_tile = _same_tile_contribution(
        candidate, remaining_tile_counts[tile_type_index(candidate)]
    )
    if candidate.category is TileCategory.HONOR:
        return _ClassicalRiichiDangerBreakdown(same_tile, 0, 0, 0, 0)

    rank = candidate.rank
    penchan = 0
    if rank == 3 and remaining(1) > 0 and remaining(2) > 0:
        penchan = 3
    elif rank == 7 and remaining(8) > 0 and remaining(9) > 0:
        penchan = 3

    kanchan = (
        3
        if 2 <= rank <= 8 and remaining(rank - 1) > 0 and remaining(rank + 1) > 0
        else 0
    )
    ryanmen_low_side = (
        10
        if 1 <= rank <= 6
        and remaining(rank + 1) > 0
        and remaining(rank + 2) > 0
        and TileType(candidate.category, rank + 3) not in opponent_river
        else 0
    )
    ryanmen_high_side = (
        10
        if 4 <= rank <= 9
        and remaining(rank - 2) > 0
        and remaining(rank - 1) > 0
        and TileType(candidate.category, rank - 3) not in opponent_river
        else 0
    )
    return _ClassicalRiichiDangerBreakdown(
        same_tile,
        penchan,
        kanchan,
        ryanmen_low_side,
        ryanmen_high_side,
    )


def _mechanism_defense_eligible_actions(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, ...]:
    """Issue #163 activation時だけminimum worst-opponent scoreへhard filterする。"""
    riichi_players = _opponent_riichi_players(policy_input)
    if not riichi_players:
        return discard_actions

    evaluator = _DecisionShantenEvaluator()
    evaluated = _evaluate_post_discard_hands(policy_input, discard_actions, evaluator)
    if min(candidate.post_discard_shanten for candidate in evaluated) < 2:
        return discard_actions

    common_genbutsu = _common_genbutsu_tile_types(riichi_players)
    if any(action.tile.tile_type in common_genbutsu for action in discard_actions):
        return discard_actions

    remaining_counts = derive_remaining_tile_inventory(
        policy_input
    ).remaining_tile_counts
    overall_dangers = tuple(
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
    minimum_danger = min(score for _, score in overall_dangers)
    return tuple(action for action, score in overall_dangers if score == minimum_danger)


class MechanismRiichiDefenseYakuhaiCallPolicy(
    YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy
):
    """2向聴以上・common genbutsuなしだけをmechanism守備で拡張するPolicy。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        eligible_actions = _mechanism_defense_eligible_actions(
            policy_input, discard_actions
        )
        return PolicyDecision(
            action=_parent_offensive_evaluate_and_choose_discard(
                policy_input, eligible_actions
            ),
            analysis=None,
        )
