"""Championの現物なし1向聴FOLDへ既存mechanism守備を加える実験候補。

危険度は公開情報からの相対scoreであり、放銃確率ではない。
聴牌機会とのtrade-offは未評価で、Championの置き換えではない。
"""

from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.policies.genbutsu_defense_two_step_ukeire import (
    _common_genbutsu_tile_types,
    _opponent_riichi_players,
)
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    _classical_riichi_danger_score,
)
from lisjong.policies.placement_aware_speed_call import PlacementAwareSpeedCallPolicy
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.structural_efficiency import (
    StructuralShantenEvaluator,
    evaluate_post_discard_hands,
)


def _one_shanten_defense_actions(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, ...]:
    """元の合法打牌集合でgateを評価し、発動時だけ最小危険度へ絞る。"""
    opponents = _opponent_riichi_players(policy_input)
    if not opponents:
        return discard_actions
    common_genbutsu = _common_genbutsu_tile_types(opponents)
    if any(action.tile.tile_type in common_genbutsu for action in discard_actions):
        return discard_actions
    evaluated = evaluate_post_discard_hands(
        policy_input, discard_actions, StructuralShantenEvaluator()
    )
    if min(candidate.post_discard_shanten for candidate in evaluated) != 1:
        return discard_actions

    remaining = derive_remaining_tile_inventory(policy_input).remaining_tile_counts
    dangers = tuple(
        max(
            _classical_riichi_danger_score(
                action.tile.tile_type, opponent, remaining
            ).total
            for opponent in opponents
        )
        for action in discard_actions
    )
    minimum = min(dangers)
    return tuple(
        action
        for action, danger in zip(discard_actions, dangers, strict=True)
        if danger == minimum
    )


class OneShantenDefensePlacementAwareSpeedCallPolicy(PlacementAwareSpeedCallPolicy):
    """#234: 現物なし1向聴FOLDの守備を追加する未昇格のChampion派生候補。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        return super()._decide_discard(
            policy_input, _one_shanten_defense_actions(policy_input, discard_actions)
        )
