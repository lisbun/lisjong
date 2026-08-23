"""他家リーチ時に共通現物を優先するTwoStepUkeire Policy。

Issue #97のDecisionTrace / AnalysisTrace基盤に対して、このPolicyは
**explicit override**でanalysis非公開を選択する。基底`TwoStepUkeirePolicy`が
持つanalysis付き打牌評価pathを偶然inheritして、defense decision pathを迂回する
ことがないよう、single decision extension pointである`_decide_discard()`自身を
overrideする。

本Issueの範囲ではGenbutsuDefense固有analysisを追加しない。したがって、

    selected action = 既存のGenbutsuDefense decision
    analysis        = None

をtrace有効時の期待behaviorとする。trace有無でselected Actionは一致し、
defense activationの順序も変わらない。Policyも1回しか実行しない。
"""

from lisjong.policies.two_step_ukeire import (
    TwoStepUkeirePolicy,
    _choose_discard,
    _DecisionShantenEvaluator,
    _evaluate_and_choose_prepared,
    _evaluate_post_discard_hands,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_decision import PolicyDecision
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

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        """defense decision pathの結果を、analysis非公開のまま返す。

        基底classのanalysis付き打牌評価pathへ委譲しない。本Issueでは
        GenbutsuDefense固有analysisを追加しないため、`analysis`は常に`None`と
        する。`None`は「analysisを生成していない」ことを表し、評価結果0や
        empty evaluationの意味には流用しない。
        """
        return PolicyDecision(
            action=self._choose_defense_discard(policy_input, discard_actions),
            analysis=None,
        )

    def _choose_defense_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> DiscardAction:
        """既存のdefense decision pathそのもの。判定順序を変更しない。"""
        riichi_players = _opponent_riichi_players(policy_input)
        if not riichi_players:
            return _choose_discard(policy_input, discard_actions)

        evaluator = _DecisionShantenEvaluator()
        evaluated = _evaluate_post_discard_hands(
            policy_input, discard_actions, evaluator
        )
        if min(candidate.post_discard_shanten for candidate in evaluated) < 1:
            selected, _ = _evaluate_and_choose_prepared(
                policy_input, evaluated, evaluator
            )
            return selected

        common_tile_types = _common_genbutsu_tile_types(riichi_players)
        genbutsu_actions = tuple(
            action
            for action in discard_actions
            if action.tile.tile_type in common_tile_types
        )
        if genbutsu_actions:
            genbutsu_evaluated = tuple(
                candidate
                for candidate in evaluated
                if candidate.action in genbutsu_actions
            )
            selected, _ = _evaluate_and_choose_prepared(
                policy_input, genbutsu_evaluated, evaluator
            )
            return selected

        selected, _ = _evaluate_and_choose_prepared(policy_input, evaluated, evaluator)
        return selected
