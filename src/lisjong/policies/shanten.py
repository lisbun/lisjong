"""向聴数計算を利用した、最初の麻雀戦略的Policy。

`MinimalPolicy`はPolicy契約・合法手境界・決定性を確認するためのbaselineで
あり、麻雀の目的そのものは考慮しない。`ShantenPolicy`はIssue #50で確定した
`lisjong.hand_evaluation.calculate_shanten()`を利用し、「和了できるなら和了し、
自分の打牌番では打牌後向聴数を最小にする」ことだけを目的とする最初のPolicy
である。

選択規則:

1. `RonAction` / `TsumoAction`が存在すれば、和了候補だけを対象にする。
   複数の和了候補があり得る場合も、semantic fieldだけのstable deterministicな
   sort keyで1件を選ぶ。
2. `DiscardAction`が1件以上あれば、各候補について
   `decision.input.own_hand.concealed_tiles`から`action.tile`と一致する牌を
   ちょうど1枚だけ除いた純手牌を作り、`calculate_shanten()`で打牌後向聴数を
   比較する。最小向聴数の候補が複数あれば、`tile_sort_key(action.tile)`と
   `action.tsumogiri`によるtie-breakで1件を選ぶ。
3. 打牌候補がなく`PassAction`が合法なら、それを選ぶ。
4. 和了・打牌・Passのいずれもなく、合法候補が1件だけならその強制候補を返す。
5. それ以外（鳴き・槓・立直・九種九牌等が複数残る場合）は、根拠のない
   action type順で選ばずfail closedする。

このPolicyはIssue #51のscope内で、受け入れ枚数比較・打点評価・鳴きや立直の
期待値評価・守備を一切行わない。それらは後続Issueの責務である。
"""

from lisjong.hand_evaluation import calculate_shanten
from lisjong.policy_contract.action import (
    DiscardAction,
    InternalAction,
    PassAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.tile import Tile, tile_sort_key

_WINNING_ACTION_TYPES = (RonAction, TsumoAction)


class ShantenPolicyError(Exception):
    """`ShantenPolicy`が入力の不整合または未定義の状況をfail closedする場合。"""


def _winning_action_sort_key(action: RonAction | TsumoAction) -> tuple[object, ...]:
    """和了候補だけを対象にした、semantic fieldだけのstable deterministic key。"""
    if isinstance(action, RonAction):
        return (
            0,
            int(action.actor),
            int(action.target),
            tile_sort_key(action.winning_tile),
        )
    return (1, int(action.actor), tile_sort_key(action.winning_tile))


def _remove_one_matching_tile(
    concealed_tiles: tuple[Tile, ...], tile: Tile
) -> list[Tile]:
    """`concealed_tiles`から`tile`とsemantic equalityで一致する牌を1枚だけ除く。

    同じ牌が複数枚あっても、除くのは1枚だけである。一致する牌が存在しない
    場合、それはDiscardAction / OwnHandState間の不整合なのでfail closedする
    （別の同牌種で代用しない）。
    """
    remaining = list(concealed_tiles)
    for index, candidate in enumerate(remaining):
        if candidate == tile:
            del remaining[index]
            return remaining
    raise ShantenPolicyError(
        "DiscardAction.tile has no matching tile in own_hand.concealed_tiles"
    )


def _post_discard_shanten(concealed_tiles: tuple[Tile, ...], tile: Tile) -> int:
    remaining_hand = _remove_one_matching_tile(concealed_tiles, tile)
    return calculate_shanten(remaining_hand)


def _discard_sort_key(
    concealed_tiles: tuple[Tile, ...], action: DiscardAction
) -> tuple[object, ...]:
    """打牌後向聴数を先頭に置き、同向聴はsemantic fieldだけでtie-breakするkey。"""
    return (
        _post_discard_shanten(concealed_tiles, action.tile),
        tile_sort_key(action.tile),
        action.tsumogiri,
    )


class ShantenPolicy:
    """向聴数を最小化する、最初の麻雀戦略的Policy。"""

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        winning_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, _WINNING_ACTION_TYPES)
        )
        if winning_actions:
            return min(winning_actions, key=_winning_action_sort_key)

        discard_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, DiscardAction)
        )
        if discard_actions:
            concealed_tiles = decision.input.own_hand.concealed_tiles
            return min(
                discard_actions,
                key=lambda action: _discard_sort_key(concealed_tiles, action),
            )

        pass_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, PassAction)
        )
        if pass_actions:
            return pass_actions[0]

        if len(decision.legal_actions) == 1:
            return decision.legal_actions[0]

        raise ShantenPolicyError(
            "no winning action, discard, or pass is available and multiple "
            "non-discard candidates remain without a defined conservative rule"
        )
