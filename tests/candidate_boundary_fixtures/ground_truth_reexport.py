"""違反: allowlist上の`lisjong.belief`経由で、training-only truthのbuilderを取り込む。"""

from lisjong.belief import exact_hand_belief_with_waits  # noqa: F401
from lisjong.policies.minimal import MinimalPolicy


class GroundTruthPolicy(MinimalPolicy):
    pass
