"""違反: allowlist上のmoduleに存在しない名前をimportする。"""

from lisjong.policies.minimal import DoesNotExistPolicy  # noqa: F401
