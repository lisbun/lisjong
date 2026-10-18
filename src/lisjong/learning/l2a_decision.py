"""L2a（C0蒸留）の判断の分類・scorer候補・教師規則guard（Issue #315）。

設計の正本は`docs/learning-l2a.md`の2.1節・2.3節である。support計数
（`lisjong.learning.l2a_support`）、dataset、servingは、この1つの実装だけを
使う。generic router / dispatcherではない。

```text
legal actionが1つ                    -> FORCED         その行動
elif Ron / Tsumoがlegal              -> WIN            和了（canonical tie-break）
elif RiichiActionがlegal             -> RIICHI         リーチ
elif DiscardActionがlegal            -> DISCARD        打牌scorer（候補はDiscardActionだけ）
elif Chi / Ponがlegal かつ Passがlegal -> CALL           反応scorer（候補はPass・Chi・Pon）
elif Passがlegal                     -> PASS_RESPONSE  Pass
else                                 -> fail closed
```

分類は`legal_actions`全体で行う。DISCARD / CALLで、Ankan / Kakan /
KyuushuKyuuhai / Daiminkanをscorer候補から外した後に候補が1つだけになる行を
**単一候補行**とし、scorerを呼ばずその候補を返す。単一候補行は学習datasetに
入れず、評価の分母からも外す。

和了の選択規則は`lisjong.learning._o0`と同じcanonical順である（L0.2 O0・
C0と一致）。concrete Policy moduleのprivate helperには依存しない。
"""

from enum import Enum

from lisjong.learning._o0 import _WINNING_ACTION_TYPES, _winning_action_sort_key
from lisjong.learning.errors import LearnedPolicyError
from lisjong.policy_contract import (
    ChiAction,
    DecisionContext,
    DiscardAction,
    InternalAction,
    PassAction,
    PonAction,
    RiichiAction,
)

L2A_DECISION_CLASSIFICATION_IDENTITY = "lisjong-l2a-decision-classification-v1"
"""上記の分類・scorer候補・guard規則のidentity。下流のdataset / artifactがbindする。"""

_CALL_ACTION_TYPES = (ChiAction, PonAction)


class L2aDecisionKind(Enum):
    """1判断がL2aの分類のどれに属するか。"""

    FORCED = "forced"
    WIN = "win"
    RIICHI = "riichi"
    DISCARD = "discard"
    CALL = "call"
    PASS_RESPONSE = "pass_response"


SCORER_DECISION_KINDS = frozenset({L2aDecisionKind.DISCARD, L2aDecisionKind.CALL})
"""scorerが候補を選ぶ分類。単一候補行はこの中でもscorerを呼ばない。"""


def classify_l2a_decision(decision: DecisionContext) -> L2aDecisionKind:
    """`legal_actions`だけから分類を決める。どれにも入らなければfail closedする。"""
    legal_actions = decision.legal_actions
    if len(legal_actions) == 1:
        return L2aDecisionKind.FORCED
    if any(isinstance(action, _WINNING_ACTION_TYPES) for action in legal_actions):
        return L2aDecisionKind.WIN
    if any(isinstance(action, RiichiAction) for action in legal_actions):
        return L2aDecisionKind.RIICHI
    if any(isinstance(action, DiscardAction) for action in legal_actions):
        return L2aDecisionKind.DISCARD
    has_pass = any(isinstance(action, PassAction) for action in legal_actions)
    if any(isinstance(action, _CALL_ACTION_TYPES) for action in legal_actions):
        if has_pass:
            return L2aDecisionKind.CALL
    elif has_pass:
        return L2aDecisionKind.PASS_RESPONSE
    raise LearnedPolicyError(
        "decision does not match any L2a classification: "
        + ", ".join(sorted({type(action).__name__ for action in legal_actions}))
    )


def scorer_candidates(
    decision: DecisionContext, kind: L2aDecisionKind
) -> tuple[InternalAction, ...]:
    """DISCARD / CALLのscorer候補を`legal_actions`の順で返す。

    DISCARDはDiscardActionだけ、CALLはPass・Chi・Ponだけを候補にする。
    返すobjectは`decision.legal_actions`側のものである。
    """
    if kind is L2aDecisionKind.DISCARD:
        candidate_types: tuple[type, ...] = (DiscardAction,)
    elif kind is L2aDecisionKind.CALL:
        candidate_types = (PassAction, *_CALL_ACTION_TYPES)
    else:
        raise LearnedPolicyError(f"{kind.name} decision has no scorer candidates")
    return tuple(
        action
        for action in decision.legal_actions
        if isinstance(action, candidate_types)
    )


def is_single_candidate(decision: DecisionContext, kind: L2aDecisionKind) -> bool:
    """DISCARD / CALLで、scorer候補が1つだけの単一候補行かどうか。"""
    return kind in SCORER_DECISION_KINDS and len(scorer_candidates(decision, kind)) == 1


def guard_action(
    decision: DecisionContext, kind: L2aDecisionKind
) -> InternalAction | None:
    """教師規則guardが決める行動を返す。scorerが選ぶ判断では`None`を返す。

    FORCED / WIN / RIICHI / PASS_RESPONSEと単一候補行が対象である。返すobjectは
    常に`decision.legal_actions`側のものである。
    """
    legal_actions = decision.legal_actions
    if kind is L2aDecisionKind.FORCED:
        return legal_actions[0]
    if kind is L2aDecisionKind.WIN:
        return min(
            (
                action
                for action in legal_actions
                if isinstance(action, _WINNING_ACTION_TYPES)
            ),
            key=_winning_action_sort_key,
        )
    if kind is L2aDecisionKind.RIICHI:
        # RiichiActionはactorだけをfieldに持ち、semantic重複は禁止されるため1つだけ。
        return next(
            action for action in legal_actions if isinstance(action, RiichiAction)
        )
    if kind is L2aDecisionKind.PASS_RESPONSE:
        return next(
            action for action in legal_actions if isinstance(action, PassAction)
        )
    candidates = scorer_candidates(decision, kind)
    if len(candidates) == 1:
        return candidates[0]
    return None


def teacher_rule_exception(
    decision: DecisionContext, kind: L2aDecisionKind, selected: InternalAction
) -> str | None:
    """教師の選択がguardの前提（設計2.3節）に反する理由を返す。反しなければ`None`。

    guardが行動を決める判断では、教師の選択がguardの行動と一致することを求める。
    scorerの判断では、教師の選択がscorer候補（槓・九種九牌以外）であることを求める。
    """
    expected = guard_action(decision, kind)
    if expected is not None:
        if selected != expected:
            return (
                f"{kind.name}: teacher selected {type(selected).__name__}, "
                f"guard selects {type(expected).__name__}"
            )
        return None
    if selected not in scorer_candidates(decision, kind):
        return (
            f"{kind.name}: teacher selected {type(selected).__name__}, "
            "which is not a scorer candidate"
        )
    return None


__all__ = [
    "L2A_DECISION_CLASSIFICATION_IDENTITY",
    "SCORER_DECISION_KINDS",
    "L2aDecisionKind",
    "classify_l2a_decision",
    "guard_action",
    "is_single_candidate",
    "scorer_candidates",
    "teacher_rule_exception",
]
