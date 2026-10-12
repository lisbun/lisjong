"""違反: allowlist外のhelper経由で、推移的に`lisjong_engine`をimportする。"""

from candidate_boundary_fixtures import _engine_helper  # noqa: F401
from lisjong.policies.minimal import MinimalPolicy


class TransitiveEnginePolicy(MinimalPolicy):
    pass
