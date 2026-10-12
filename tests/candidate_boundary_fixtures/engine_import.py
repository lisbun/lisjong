"""違反: 環境objectを持つ`lisjong_engine`を直接importする。"""

import lisjong_engine  # noqa: F401

from lisjong.policies.minimal import MinimalPolicy


class EngineImportPolicy(MinimalPolicy):
    pass
