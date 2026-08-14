"""境界検証用の決定的な最小Policy。

このPolicyは麻雀AIとしての強さを目的としない。`DecisionContext`だけを参照し、
合法候補の入力順序に依存しない明示的なtotal orderで1件を選択する。

選択規則:
1. `RonAction` / `TsumoAction` が存在すれば、和了候補だけを対象にする。
2. 対象候補をvariant種別と各Actionのsemantic fieldから作るsort keyで比較し、
   最小のActionを選ぶ。

variant間の順序はtotal orderを成立させるためだけの実装上の固定順であり、
麻雀上の戦略的優先度を表さない。Action内のTileは`tile_sort_key`を使用し、
multiset fieldはAction生成時にcanonicalize済みの順序をそのままkeyへ写す。

`DecisionContext`が保証するlegal_actions非空、actor一致、semantic identity一意等を
ここで重複検証しない。PRNGや呼び出し履歴など、選択へ影響するhidden mutable
stateも持たない。想定外のAction型にはfallbackせずfail closedする。
"""

from typing import assert_never

from lisjong.policy_contract.action import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DiscardAction,
    InternalAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.tile import Tile, tile_sort_key

_WINNING_ACTION_TYPES = (RonAction, TsumoAction)


def _tiles_sort_key(tiles: tuple[Tile, ...]) -> tuple[tuple[int, int, bool], ...]:
    return tuple(tile_sort_key(tile) for tile in tiles)


def _action_sort_key(action: InternalAction) -> tuple[object, ...]:
    """semantic fieldだけから、全InternalAction上の安定したtotal orderを作る。"""
    if isinstance(action, DiscardAction):
        return (0, int(action.actor), tile_sort_key(action.tile), action.tsumogiri)
    if isinstance(action, RiichiAction):
        return (1, int(action.actor))
    if isinstance(action, ChiAction):
        return (
            2,
            int(action.actor),
            int(action.target),
            tile_sort_key(action.called_tile),
            _tiles_sort_key(action.consumed_tiles),
        )
    if isinstance(action, PonAction):
        return (
            3,
            int(action.actor),
            int(action.target),
            tile_sort_key(action.called_tile),
            _tiles_sort_key(action.consumed_tiles),
        )
    if isinstance(action, DaiminkanAction):
        return (
            4,
            int(action.actor),
            int(action.target),
            tile_sort_key(action.called_tile),
            _tiles_sort_key(action.consumed_tiles),
        )
    if isinstance(action, AnkanAction):
        return (5, int(action.actor), _tiles_sort_key(action.tiles))
    if isinstance(action, KakanAction):
        return (
            6,
            int(action.actor),
            tile_sort_key(action.added_tile),
            int(action.from_seat),
            tile_sort_key(action.called_tile),
        )
    if isinstance(action, RonAction):
        return (
            7,
            int(action.actor),
            int(action.target),
            tile_sort_key(action.winning_tile),
        )
    if isinstance(action, TsumoAction):
        return (8, int(action.actor), tile_sort_key(action.winning_tile))
    if isinstance(action, PassAction):
        return (9, int(action.actor))
    if isinstance(action, KyuushuKyuuhaiAction):
        return (10, int(action.actor))
    assert_never(action)


class MinimalPolicy:
    """Issue #22の境界検証用Policy。"""

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        winning_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, _WINNING_ACTION_TYPES)
        )
        candidates = winning_actions or decision.legal_actions
        return min(candidates, key=_action_sort_key)
