"""環境非依存のPolicy呼び出し・返却値validation境界。

1 seat x 1 decisionだけを扱う。受け取った``DecisionContext``を変更せず
``Policy.choose_action()``へ渡し、返却値をAction dataclassの既存value equalityで
同Contextの合法候補へ照合する。外部Action変換、transport、timeout、retry、
fallback選択は扱わない。

trace付きexecutionは``execute_policy_with_trace()``としてopt-inで提供する。
既存の``execute_policy(policy, decision)``へtrace用引数は追加しない。両APIは
``_select_canonical_legal_action()``という単一のprivate validation pathを共有し、
legal-action validationを二重実装しない。
"""

from lisjong.policy_contract.action import InternalAction, _is_internal_action
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.decision_trace import DecisionTrace, DecisionTraceSink
from lisjong.policy_contract.policy import Policy
from lisjong.policy_contract.policy_decision import PolicyDecision

_ANALYSIS_CAPABILITY_METHOD = "choose_action_with_analysis"


class PolicyActionValidationError(Exception):
    """Policy返却値をちょうど1件の合法Actionとして検証できない場合。"""


def _require_decision_context(decision: DecisionContext) -> None:
    if not isinstance(decision, DecisionContext):
        raise TypeError("decision must be a DecisionContext")


def _select_canonical_legal_action(
    selected: object, decision: DecisionContext
) -> InternalAction:
    """Policy提案Actionを合法候補へ照合し、canonicalな候補を返す。

    traced / non-tracedいずれのexecutionもこのpathだけを使う。返り値は常に
    ``decision.legal_actions``側のobjectであり、Policyが返したequalだが別の
    objectではない。
    """
    if not _is_internal_action(selected):
        raise PolicyActionValidationError(
            "Policy.choose_action() must return an InternalAction"
        )

    try:
        matches = tuple(
            candidate for candidate in decision.legal_actions if candidate == selected
        )
    except Exception as error:
        raise PolicyActionValidationError(
            "Policy result could not be compared safely with legal_actions"
        ) from error

    if len(matches) != 1:
        raise PolicyActionValidationError(
            "Policy result must match exactly one legal action; "
            f"found {len(matches)} matches"
        )

    return matches[0]


def _decide_once(
    policy: Policy, decision: DecisionContext
) -> tuple[object, AnalysisTrace | None]:
    """Policyを**ちょうど1回**実行し、提案Actionとoptional analysisを返す。

    analysis capabilityを持たないPolicyは``choose_action()``だけを呼ぶ。
    capabilityを持つPolicyでも、analysis取得のためにdecision algorithmを
    再実行しない。
    """
    decide_with_analysis = getattr(policy, _ANALYSIS_CAPABILITY_METHOD, None)
    if not callable(decide_with_analysis):
        return policy.choose_action(decision), None

    proposed = decide_with_analysis(decision)
    if not isinstance(proposed, PolicyDecision):
        raise PolicyActionValidationError(
            f"Policy.{_ANALYSIS_CAPABILITY_METHOD}() must return a PolicyDecision"
        )
    return proposed.action, proposed.analysis


def execute_policy(policy: Policy, decision: DecisionContext) -> InternalAction:
    """Policyを呼び出し、一意に一致したcanonicalな合法候補を返す。

    ``Policy.choose_action()``が送出した例外は変更せず伝播する。返却値が
    ``InternalAction``でない、安全に比較できない、または候補へちょうど1件一致
    しない場合は``PolicyActionValidationError``を送出する。代替Actionやfallbackは
    選択しない。
    """
    _require_decision_context(decision)

    return _select_canonical_legal_action(policy.choose_action(decision), decision)


def execute_policy_with_trace(
    policy: Policy, decision: DecisionContext, sink: DecisionTraceSink
) -> InternalAction:
    """opt-inのtrace付きexecution。canonicalな合法候補を返す。

    処理順は次で固定する。

        Policy decision once
            -> Policy result validation
            -> canonical legal action
            -> DecisionTrace construction
            -> sink.on_decision(trace)
            -> return canonical action

    Policyの例外、返却値のvalidation失敗ではDecisionTraceをemitしない。
    ``sink.on_decision()``が送出した例外は握り潰さず、そのまま伝播する。
    このときfallback Actionを返さず、Policyも再実行しない。

    trace有無でsemantic selected actionは変わらない。``execute_policy()``と
    同じ``_select_canonical_legal_action()``だけをvalidation pathとして使う。
    """
    _require_decision_context(decision)
    if not callable(getattr(sink, "on_decision", None)):
        raise TypeError("sink must provide a callable on_decision()")

    proposed_action, analysis = _decide_once(policy, decision)
    selected = _select_canonical_legal_action(proposed_action, decision)

    sink.on_decision(
        DecisionTrace(
            legal_actions=decision.legal_actions,
            selected_action=selected,
            analysis=analysis,
        )
    )
    return selected
