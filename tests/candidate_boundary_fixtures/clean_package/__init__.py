"""合格: package内の相対importと、allowlist上のmoduleだけを使う候補。"""

from .policy import CleanPackagePolicy

__all__ = ["CleanPackagePolicy"]
