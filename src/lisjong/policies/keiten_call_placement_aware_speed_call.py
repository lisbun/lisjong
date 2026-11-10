"""終盤だけ、聴牌を取る鳴きと聴牌維持の打牌を加える実験候補（#375）。

仕様は`docs/keiten-call-candidate.md`。未昇格の実験候補で、Championは書き換えない。
exact parentは`PlacementAwareSpeedCallPolicy`。

## 動作

`RoundState.live_wall_tiles_remaining`が`KEITEN_MAXIMUM_LIVE_WALL`以下の判断でだけ、
parentの判断を次の2点で変える。それ以外の判断はparentと完全に同じである。

A. 聴牌を取る鳴き

    parentが明示的なPassを選び、自席の純手牌が非聴牌で、Chi / Ponのあと喰い替えを
    守った打牌を1枚して聴牌になる組合せがあるとき、その鳴きを選ぶ。他家リーチが
    ある場合は、その打牌が全リーチ者への共通現物である組合せに限る。

B. 聴牌維持の打牌

    副露済みの手で、聴牌を維持する打牌があるとき、候補をその集合へ絞ってparentへ
    渡す。他家リーチがあり、その集合に共通現物があれば、共通現物だけへ絞る。

聴牌は純手牌の向聴数0で判定する。役の有無は判定しない（形式聴牌を含む）。待ち牌が
すべて見えている純カラ聴も区別しない。鳴き後の打牌は実際の次の判断でparentが
fresh `legal_actions`から選ぶため、Aで想定した打牌を保存しない（Bがその集合を残す）。

PolicyInput以外の状態やhidden informationは使用せず、Policy instanceに
cross-decision stateを保持しない。
"""

from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies.genbutsu_defense_two_step_ukeire import (
    _common_genbutsu_tile_types,
    _opponent_riichi_players,
)
from lisjong.policies.placement_aware_speed_call import (
    PlacementAwareSpeedCallPolicy,
    _is_closed_hand,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    _call_action_sort_key,
    _CallAction,
    _kuikae_forbidden_tile_types,
    _remove_exact_consumed_tiles,
)
from lisjong.policy_contract.action import (
    ChiAction,
    DiscardAction,
    PassAction,
    PonAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import TileType
from lisjong.structural_efficiency import post_discard_concealed_hand

KEITEN_MAXIMUM_LIVE_WALL = 12
"""A / Bを適用する`live_wall_tiles_remaining`の上限（この値以下で適用）。"""


def _is_keiten_window(policy_input: PolicyInput) -> bool:
    return policy_input.round.live_wall_tiles_remaining <= KEITEN_MAXIMUM_LIVE_WALL


def _riichi_genbutsu(policy_input: PolicyInput) -> frozenset[TileType] | None:
    """他家リーチがなければNone、あれば全リーチ者への共通現物（空集合を含む）。"""
    riichi_players = _opponent_riichi_players(policy_input)
    if not riichi_players:
        return None
    return _common_genbutsu_tile_types(riichi_players)


def _call_reaches_tenpai(
    policy_input: PolicyInput,
    action: _CallAction,
    genbutsu: frozenset[TileType] | None,
) -> bool:
    """鳴いて、許される打牌を1枚した後に聴牌になる組合せがあるか。

    `genbutsu`がNoneでなければ、その打牌は共通現物でなければならない。
    """
    post_call_tiles = _remove_exact_consumed_tiles(
        policy_input.own_hand.concealed_tiles, action.consumed_tiles
    )
    forbidden = _kuikae_forbidden_tile_types(action)
    checked: set[TileType] = set()
    for index, tile in enumerate(post_call_tiles):
        tile_type = tile.tile_type
        if tile_type in forbidden or tile_type in checked:
            continue
        checked.add(tile_type)
        if genbutsu is not None and tile_type not in genbutsu:
            continue
        stable_hand = (*post_call_tiles[:index], *post_call_tiles[index + 1 :])
        if calculate_shanten(stable_hand) == 0:
            return True
    return False


def _select_keiten_call(
    decision: DecisionContext, parent_decision: PolicyDecision
) -> _CallAction | None:
    """parentがPassした終盤の判断で、聴牌を取る鳴きを返す（なければNone）。"""
    if not isinstance(parent_decision.action, PassAction):
        return None
    policy_input = decision.input
    if not _is_keiten_window(policy_input):
        return None
    call_actions = tuple(
        action
        for action in decision.legal_actions
        if isinstance(action, (ChiAction, PonAction))
    )
    if not call_actions:
        return None
    if calculate_shanten(policy_input.own_hand.concealed_tiles) < 1:
        return None
    genbutsu = _riichi_genbutsu(policy_input)
    qualifying = tuple(
        action
        for action in call_actions
        if _call_reaches_tenpai(policy_input, action, genbutsu)
    )
    if not qualifying:
        return None
    return min(qualifying, key=_call_action_sort_key)


def _tenpai_keeping_actions(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, ...]:
    """終盤の副露手だけ、聴牌維持（リーチ下は可能なら共通現物）の打牌へ絞る。"""
    if not _is_keiten_window(policy_input) or _is_closed_hand(policy_input):
        return discard_actions
    concealed_tiles = policy_input.own_hand.concealed_tiles
    keeping = tuple(
        action
        for action in discard_actions
        if calculate_shanten(post_discard_concealed_hand(concealed_tiles, action.tile))
        == 0
    )
    if not keeping:
        return discard_actions
    genbutsu = _riichi_genbutsu(policy_input)
    if genbutsu:
        safe_keeping = tuple(
            action for action in keeping if action.tile.tile_type in genbutsu
        )
        if safe_keeping:
            return safe_keeping
    return keeping


class KeitenCallPlacementAwareSpeedCallPolicy(PlacementAwareSpeedCallPolicy):
    """Issue #375: 終盤の聴牌を取る鳴き + 聴牌維持打牌。"""

    def _decide(self, decision: DecisionContext) -> PolicyDecision:
        parent_decision = super()._decide(decision)
        selected = _select_keiten_call(decision, parent_decision)
        if selected is None:
            return parent_decision
        return PolicyDecision(action=selected)

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        return super()._decide_discard(
            policy_input, _tenpai_keeping_actions(policy_input, discard_actions)
        )


__all__ = ["KEITEN_MAXIMUM_LIVE_WALL", "KeitenCallPlacementAwareSpeedCallPolicy"]
