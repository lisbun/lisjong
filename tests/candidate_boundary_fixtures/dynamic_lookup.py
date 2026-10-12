"""違反: 動的な属性参照で、関数のmodule globalsへ届く。"""

from lisjong.policies.minimal import MinimalPolicy


class DynamicLookupPolicy(MinimalPolicy):
    def choose_action(self, decision):
        namespace = getattr(super().choose_action, "__globals__")
        del namespace
        return super().choose_action(decision)
