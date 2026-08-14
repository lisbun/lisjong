"""lisjong内部の副露・槓種別value型。

docs/policy-input-schema.md「PublicMeld」のkindの意味契約を実装する。
PublicMeld本体（kind、tiles、from_seat、called_tile一式）は本Issueのこの
実装単位には含めず、後続で追加する。
"""

from enum import Enum


class MeldKind(Enum):
    """PublicMeld.kindが区別する副露・槓種別。"""

    CHI = "chi"
    PON = "pon"
    DAIMINKAN = "daiminkan"
    ANKAN = "ankan"
    KAKAN = "kakan"
