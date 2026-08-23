"""analysis-capable Policyが1 decisionで返すdecision + optional analysis。

`PolicyDecision`は、1回のdecision calculationから得られた、

- Policyが提案した`InternalAction`
- そのdecisionで実際に生成されたtyped analysis、またはanalysisなしを表す`None`

をまとめるimmutable valueである。

`PolicyDecision.action`は**Policyが提案したAction**であり、
`DecisionTrace.selected_action`は**既存のlegal-action validationを通過した
canonicalな合法Action**である。この2つを同一視しない。

`Policy.choose_action()`の既存契約は変更しない。analysisを提供できるPolicyは
`AnalysisCapablePolicy`のoptional capability methodを追加で実装してよい。
capabilityを実装しないPolicyは一切変更せずtraced executionから利用できる。

capability methodは、`choose_action()`とdecision algorithmを二重実装せず、
1回のdecision calculationからactionとanalysisの両方を得ること。
`policy.last_analysis`のようなdecision間mutable stateをanalysisの
transport mechanismにしない。
"""

from dataclasses import dataclass
from typing import Protocol

from lisjong.policy_contract.action import InternalAction, _is_internal_action
from lisjong.policy_contract.analysis_trace import (
    AnalysisTrace,
    _require_optional_analysis_trace,
)
from lisjong.policy_contract.decision_context import DecisionContext


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """1回のPolicy decisionが提案したactionと、optionalなtyped analysis。"""

    action: InternalAction
    analysis: AnalysisTrace | None = None

    def __post_init__(self) -> None:
        if not _is_internal_action(self.action):
            raise TypeError("action must be an InternalAction")
        _require_optional_analysis_trace(self.analysis, "analysis")


class AnalysisCapablePolicy(Protocol):
    """typed analysisも提供できるPolicyのoptional structural capability。

    `Policy`と同じくstructural Protocolであり、明示的な継承を要求しない。
    `@runtime_checkable`も付けない。traced execution境界は、method名の有無を
    duck typingで判定するだけであり、このProtocolをruntime検証には使わない。
    """

    def choose_action(self, decision: DecisionContext) -> InternalAction: ...

    def choose_action_with_analysis(
        self, decision: DecisionContext
    ) -> PolicyDecision: ...
