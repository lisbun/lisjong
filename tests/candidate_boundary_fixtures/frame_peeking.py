"""違反: 呼び出し元のframeから`DecisionContext`以外の状態を読む。"""

import sys

from lisjong.policies.minimal import MinimalPolicy


class FramePeekingPolicy(MinimalPolicy):
    def choose_action(self, decision):
        caller = sys._getframe(1)
        del caller
        return super().choose_action(decision)
