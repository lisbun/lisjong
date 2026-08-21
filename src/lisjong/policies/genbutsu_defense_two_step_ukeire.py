"""他家リーチ時に共通現物を優先するTwoStepUkeire Policy。"""

from lisjong.policies.two_step_ukeire import (
    TwoStepUkeirePolicy,
    _choose_discard,
    _DecisionShantenEvaluator,
    _evaluate_post_discard_hands,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import TileType


def _opponent_riichi_players(
    policy_input: PolicyInput,
) -> tuple[PlayerPublicState, ...]:
    """PolicyInputのseat identityに従い、リーチ中の他家だけを返す。"""
    return tuple(
        player
        for seat_index, player in enumerate(policy_input.players)
        if Seat(seat_index) != policy_input.self_seat
        and player.riichi is not RiichiState.NONE
    )


def _common_genbutsu_tile_types(
    riichi_players: tuple[PlayerPublicState, ...],
) -> frozenset[TileType]:
    """全リーチ者のdiscard履歴に共通する基礎牌種を返す。"""
    if not riichi_players:
        return frozenset()

    discard_tile_types = (
        frozenset(discard.tile.tile_type for discard in player.discards)
        for player in riichi_players
    )
    common = next(discard_tile_types)
    for tile_types in discard_tile_types:
        common = common.intersection(tile_types)
    return common


class GenbutsuDefenseTwoStepUkeirePolicy(TwoStepUkeirePolicy):
    """非聴牌なら全リーチ者への共通現物を優先するstateless baseline。"""

    def _choose_discard_action(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> DiscardAction:
        riichi_players = _opponent_riichi_players(policy_input)
        if not riichi_players:
            return super()._choose_discard_action(policy_input, discard_actions)

        evaluator = _DecisionShantenEvaluator()
        evaluated = _evaluate_post_discard_hands(
            policy_input, discard_actions, evaluator
        )
        if min(shanten for shanten, _, _ in evaluated) < 1:
            return super()._choose_discard_action(policy_input, discard_actions)

        common_tile_types = _common_genbutsu_tile_types(riichi_players)
        genbutsu_actions = tuple(
            action
            for action in discard_actions
            if action.tile.tile_type in common_tile_types
        )
        if genbutsu_actions:
            return _choose_discard(policy_input, genbutsu_actions)

        return super()._choose_discard_action(policy_input, discard_actions)
