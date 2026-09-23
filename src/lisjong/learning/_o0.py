"""L0.2 candidate scorer用のO0 decision precedence（Issue #189）。

servingのorchestrationとcandidate datasetのrow eligibilityが同じ判定を使う
ための、最小のprivate seamである。generic router / dispatcher frameworkでは
ない。

```text
winning action (Ron / Tsumo) exists
    -> WIN       deterministic win（canonical deterministic tie-break）
elif RiichiAction exists
    -> RIICHI    canonical RiichiAction（O0 immediate riichi）
elif no legal DiscardAction
    -> RESPONSE  canonical PassAction（O0 closed-hand / no-call）
else
    -> DISCARD   learned candidate scorer over legal DiscardAction
```

deterministic win / immediate riichiのsemanticsは`TwoStepUkeirePolicy`
（Issue #76 Always Riichi）と一致させる。concrete Policy moduleのprivate
helperへは依存せず（docs/architecture.mdの依存方向）、
各Policy世代と同じく同じ意味のwinning action keyをここに持つ。一致はtestで
固定する。`TwoStepUkeirePolicy`をbase classとして継承もしない。

own-turnのAnkan / Kakan / KyuushuKyuuhai等のvoluntary actionはO0では選ばない。
DiscardActionがlegalであれば、それらが併存してもDISCARDへ分類する。
"""

from enum import Enum

from lisjong.learning.errors import LearnedPolicyError
from lisjong.policy_contract import (
    DecisionContext,
    DiscardAction,
    InternalAction,
    PassAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.tile import tile_sort_key

O0_DECOMPOSITION_IDENTITY = "lisjong-offense-l0.2-o0-decomposition-v1"
"""上記precedenceのidentity。datasetのrow eligibilityとartifactがbindする。"""

_WINNING_ACTION_TYPES = (RonAction, TsumoAction)


def _winning_action_sort_key(action: RonAction | TsumoAction) -> tuple[object, ...]:
    """複数winning actionのcanonical deterministic tie-break key。

    Ronを先に、actor / target / winning tileのcanonical順で比較する。
    `TwoStepUkeirePolicy`のwinning action選択と同じ順序である。
    """
    if isinstance(action, RonAction):
        return (
            0,
            int(action.actor),
            int(action.target),
            tile_sort_key(action.winning_tile),
        )
    return (1, int(action.actor), tile_sort_key(action.winning_tile))


class O0DecisionKind(Enum):
    """1 decisionがO0 precedence上どのbranchに属するか。"""

    WIN = "win"
    RIICHI = "riichi"
    RESPONSE = "response"
    DISCARD = "discard"


def classify_o0_decision(decision: DecisionContext) -> O0DecisionKind:
    """legal actionsだけからO0 branchを決める。"""
    legal_actions = decision.legal_actions
    if any(isinstance(action, _WINNING_ACTION_TYPES) for action in legal_actions):
        return O0DecisionKind.WIN
    if any(isinstance(action, RiichiAction) for action in legal_actions):
        return O0DecisionKind.RIICHI
    if not any(isinstance(action, DiscardAction) for action in legal_actions):
        return O0DecisionKind.RESPONSE
    return O0DecisionKind.DISCARD


def deterministic_guard_action(
    decision: DecisionContext, kind: O0DecisionKind
) -> InternalAction | None:
    """WIN / RIICHI / RESPONSEのcanonical legal actionを返す。

    DISCARDでは`None`を返す（scorerの責務）。RESPONSEでPassActionがlegalで
    ない場合は推測で別actionを選ばず`LearnedPolicyError`でfail closedする。返すobject
    は常に`decision.legal_actions`側のものである。
    """
    legal_actions = decision.legal_actions
    if kind is O0DecisionKind.WIN:
        return min(
            (
                action
                for action in legal_actions
                if isinstance(action, _WINNING_ACTION_TYPES)
            ),
            key=_winning_action_sort_key,
        )
    if kind is O0DecisionKind.RIICHI:
        # RiichiActionはactorだけをfieldに持ち、DecisionContextがsemantic重複を
        # 禁止するため、legalなRiichiActionは高々1つである。
        return next(
            action for action in legal_actions if isinstance(action, RiichiAction)
        )
    if kind is O0DecisionKind.RESPONSE:
        for action in legal_actions:
            if isinstance(action, PassAction):
                return action
        raise LearnedPolicyError(
            "response decision without a legal DiscardAction has no legal "
            "PassAction; O0 defines no other no-call action"
        )
    return None


__all__ = [
    "O0_DECOMPOSITION_IDENTITY",
    "O0DecisionKind",
    "classify_o0_decision",
    "deterministic_guard_action",
]
