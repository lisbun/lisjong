"""1 Policy decisionのone-way観測value、observer contract、in-memory recorder。

`DecisionTrace`は、

    what canonical action lisjong selected for one Policy decision

を表すimmutable valueであり、`AnalysisTrace`は、

    which typed lisjong intermediate values were actually produced / used
    in that Policy decision

を表す。objective execution factsを表す`lisjong-arena`側の`GameTrace`
（what happened in execution）とは責務が異なり、相互にschemaを混ぜない。

DecisionTraceはobservabilityであり、Policy decisionへfeedbackしない。
trace有無をPolicy input、decision feature、tie-break input、hidden mutable
stateとして利用しない。

`DecisionTrace`へ含めてよいのは次の3つだけである。

1. `DecisionContext`としてPolicyへ提示されたlegal actions
2. Policyがそこから実際に生成・使用したtyped intermediate value
3. 既存validation後のcanonical selected action

opponentの実手牌、山 / 王牌の実状態、環境のprivileged state、GameTraceの
privileged observer state、未来のevent、offline ground truth、Arena固有の
metric / stateを混入しない。

game-global sequence、GameTrace sequence、GameTrace join ID、environment
event IDも持たない。`DecisionTraceRecorder`が保証するのは、同一recorder内の
notification orderだけである。
"""

from dataclasses import dataclass
from typing import Protocol

from lisjong.policy_contract.action import InternalAction, _is_internal_action
from lisjong.policy_contract.analysis_trace import (
    AnalysisTrace,
    _require_optional_analysis_trace,
)


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """1回のPolicy decisionを表す、detachedなimmutable observation value。"""

    legal_actions: tuple[InternalAction, ...]
    selected_action: InternalAction
    analysis: AnalysisTrace | None = None

    def __post_init__(self) -> None:
        try:
            legal_actions = tuple(self.legal_actions)
        except TypeError:
            raise TypeError("legal_actions must be an iterable") from None

        if any(not _is_internal_action(action) for action in legal_actions):
            raise TypeError("legal_actions must contain only InternalAction instances")

        if not legal_actions:
            raise ValueError("legal_actions must not be empty")

        if len(set(legal_actions)) != len(legal_actions):
            raise ValueError(
                "legal_actions must not contain duplicate semantic actions"
            )

        if not _is_internal_action(self.selected_action):
            raise TypeError("selected_action must be an InternalAction")

        matches = tuple(
            candidate
            for candidate in legal_actions
            if candidate == self.selected_action
        )
        if len(matches) != 1:
            raise ValueError(
                "selected_action must match exactly one legal action; "
                f"found {len(matches)} matches"
            )

        _require_optional_analysis_trace(self.analysis, "analysis")

        object.__setattr__(self, "legal_actions", legal_actions)
        object.__setattr__(self, "selected_action", matches[0])


class DecisionTraceSink(Protocol):
    """完成済み`DecisionTrace`を受け取るone-way observer contract。

    `DecisionTrace`は1 decisionごとに完成済みvalueとして生成されるため、
    start / event / completeのlifecycleは持たない。`on_decision()`は
    決してPolicy decisionへ値を返さない。
    """

    def on_decision(self, trace: DecisionTrace) -> None: ...


class DecisionTraceRecorder:
    """通知順を保持する標準in-memory `DecisionTraceSink`。

    正常な1回の`on_decision()`につき1件だけrecordする。`snapshot()`は
    その時点までのrecordをimmutable tupleとして返し、以後の追加recordは
    取得済みsnapshotを変更しない。
    """

    __slots__ = ("_traces",)

    def __init__(self) -> None:
        self._traces: list[DecisionTrace] = []

    def on_decision(self, trace: DecisionTrace) -> None:
        if not isinstance(trace, DecisionTrace):
            raise TypeError("trace must be a DecisionTrace")
        self._traces.append(trace)

    def snapshot(self) -> tuple[DecisionTrace, ...]:
        """記録済み`DecisionTrace`をnotification順のimmutable tupleで返す。"""
        return tuple(self._traces)
