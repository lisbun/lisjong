"""lisjong内部のリーチ状態value型。

docs/policy-input-schema.md「RiichiState」の意味契約を実装する。
単純な riichi_declared: bool ではなく、reach Action実行前、実行済み・未成立、
成立後の3状態を区別する。
"""

from enum import Enum


class RiichiState(Enum):
    """reach Actionからreach_acceptedまでの状態を区別する。

    NONE: reach Actionを実行する前
    DECLARED: reach Actionを実行済みだが、まだ成立前
    ACCEPTED: リーチ成立後
    """

    NONE = "none"
    DECLARED = "declared"
    ACCEPTED = "accepted"
