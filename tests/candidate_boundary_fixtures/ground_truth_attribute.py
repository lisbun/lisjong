"""違反: allowlist上のmoduleをimportし、属性参照でtraining-only truthへ届く。"""

import lisjong.belief as belief
from lisjong.policies.minimal import MinimalPolicy


class GroundTruthAttributePolicy(MinimalPolicy):
    def choose_action(self, decision):
        builder = belief.exact_hand_belief_with_waits
        del builder
        return super().choose_action(decision)
