"""Policy decisionが生成したtyped intermediate valueのone-way観測contract。

`AnalysisTrace`は、1回のPolicy decisionで**実際に生成・使用された**
lisjong-ownedなintermediate valueを、decisionの外側からone-wayで観測する
ためのroot contractである。

このroot contract自身は、向聴数、受け入れ、belief、value / riskといった
AI intermediate valueのsemanticsを新しく定義しない。semanticsの正本は常に
各AI domain value側（例:
`lisjong.policies.two_step_ukeire.TwoStepUkeireCandidateEvaluation`）に置き、
`AnalysisTrace`はそれをobservation payloadとして束ねるだけである。

`AnalysisTrace` is output / observation, not Policy input。Policyへ
`AnalysisTrace`を渡さない。`DecisionContext`へも追加しない。

concrete analysisは`dict[str, object]`、`dict[str, float]`、
`Mapping[str, object]`、`reason: str`のようなfree-form telemetryではなく、
immutableでtypedなsemantic payloadとする。

root contractがruntime検証するのは、`AnalysisTrace`のsubclassかつfrozen
dataclassであるという**最低限の構造条件**だけである。これはfree-form dict /
string / mutable payloadをcanonical representationから排除するための境界であり、
deep immutabilityまでは保証しない。frozen dataclassでもfieldに`list`等の
mutable objectを持てば、その中身は変更可能である。field値まで含めた
immutabilityとdetachmentは、各concrete analysis payload側の責務とする
（例: `TwoStepUkeireAnalysis`はimmutableなcandidate valueのtupleだけを持つ）。

`policy_contract`は具体Policy実装へ逆依存しないため、concrete analysis型は
各Policy実装側のpackage（例: `lisjong.policies`）が所有する。
"""

from dataclasses import is_dataclass


class AnalysisTrace:
    """typed analysis payloadのmarker root contract。

    自身はfieldを持たず、共通のtemplate methodも定義しない。concrete analysis
    payloadは、このclassを継承したfrozen dataclassとして各Policy実装側で
    定義する。`__slots__`を空にすることで、frozen + slotsなsubclassが
    余分な`__dict__`を持たないようにする。
    """

    __slots__ = ()


def _is_analysis_trace(value: object) -> bool:
    """値がimmutableなtyped analysis payloadとして受理可能かを返す。"""
    if not isinstance(value, AnalysisTrace):
        return False
    analysis_type = type(value)
    if not is_dataclass(analysis_type):
        return False
    return analysis_type.__dataclass_params__.frozen


def _require_optional_analysis_trace(value: object, field_name: str) -> None:
    """`None`または受理可能なtyped analysis payloadであることを検証する。"""
    if value is None or _is_analysis_trace(value):
        return
    raise TypeError(
        f"{field_name} must be None or a frozen dataclass AnalysisTrace payload"
    )
