"""違反（純粋性）: 呼び出し回数という隠れた可変状態で合法手を選ぶ。"""

from lisjong.policy_contract.action import InternalAction
from lisjong.policy_contract.decision_context import DecisionContext


class CallCountPolicy:
    def __init__(self) -> None:
        self._calls = 0

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        self._calls += 1
        actions = decision.legal_actions
        return actions[self._calls % len(actions)]
