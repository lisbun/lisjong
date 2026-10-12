"""違反（純粋性）: processごとに変わるstr hashの順序で合法手を選ぶ。"""

from lisjong.policy_contract.action import InternalAction
from lisjong.policy_contract.decision_context import DecisionContext


class HashOrderPolicy:
    def choose_action(self, decision: DecisionContext) -> InternalAction:
        return min(decision.legal_actions, key=lambda action: hash(repr(action)))
