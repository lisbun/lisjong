"""環境非依存のPolicy呼び出し・返却値validation境界。

1 seat x 1 decisionだけを扱う。受け取った``DecisionContext``を変更せず
``Policy.choose_action()``へ渡し、返却値をAction dataclassの既存value equalityで
同Contextの合法候補へ照合する。外部Action変換、transport、timeout、retry、
fallback選択は扱わない。

trace付きexecutionは``execute_policy_with_trace()``としてopt-inで提供する。
既存の``execute_policy(policy, decision)``へtrace用引数は追加しない。両APIは
``_select_canonical_legal_action()``という単一のprivate validation pathを共有し、
legal-action validationを二重実装しない。

optional analysis capabilityのdispatchは、method名の有無だけでなくMRO上の
method ownerも見る。subclassが``choose_action()``だけをoverrideし、analysis
capabilityを基底classから偶然inheritしているだけの場合はcapabilityを使わず、
そのsubclass自身の``choose_action()``へfallbackする。
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


def _defining_class(policy_type: type, method_name: str) -> type | None:
    """MRO上でそのmethodを実際に定義しているclassを返す。"""
    for klass in policy_type.__mro__:
        if method_name in vars(klass):
            return klass
    return None


def _analysis_capability_is_shadowed(policy: Policy) -> bool:
    """基底classのanalysis capabilityだけをinheritしているかを返す。

    subclassが``choose_action()``だけをoverrideし、analysis capabilityを
    より上位のclassからinheritしているだけの場合、inheritしたanalysis pathは
    そのsubclassのdecision semanticsを表さない。このままtraced executionが
    inherited pathを呼ぶと、trace有無で異なるdecision algorithmを通り得る。

    これはIssue #97が禁止する「subclassが基底classのanalysis methodを偶然
    inheritした結果、decision semanticsが変わる」状態そのものなので、この場合は
    capabilityを使わず``choose_action()``のlegacy pathへfallbackする
    （``analysis``は``None``になる）。

    判定はMRO上のmethod ownerで行い、次を期待する扱いとする。

    - 同じclassが両方を定義している           -> capabilityを使う
    - subclassがanalysis pathの内側だけを
      overrideしている                        -> capabilityを使う
    - subclassが``choose_action()``だけを
      overrideしている                        -> capabilityを使わない
    - subclassがanalysis capabilityを明示
      overrideしている                        -> capabilityを使う

    instance属性として明示的にbindされたmethodは、偶然のinheritではないため
    そのまま尊重する。
    """
    instance_attributes = getattr(policy, "__dict__", {})
    if _ANALYSIS_CAPABILITY_METHOD in instance_attributes:
        return False

    policy_type = type(policy)
    analysis_owner = _defining_class(policy_type, _ANALYSIS_CAPABILITY_METHOD)
    if analysis_owner is None:
        return False

    if "choose_action" in instance_attributes:
        return True

    choose_action_owner = _defining_class(policy_type, "choose_action")
    if choose_action_owner is None:
        return False

    return not issubclass(analysis_owner, choose_action_owner)


def _decide_once(
    policy: Policy, decision: DecisionContext
) -> tuple[object, AnalysisTrace | None]:
    """Policyを**ちょうど1回**実行し、提案Actionとoptional analysisを返す。

    analysis capabilityを持たないPolicy、および基底classのanalysis capabilityを
    偶然inheritしただけのPolicyは``choose_action()``だけを呼ぶ。capabilityを
    使う場合も、analysis取得のためにdecision algorithmを再実行しない。
    """
    decide_with_analysis = getattr(policy, _ANALYSIS_CAPABILITY_METHOD, None)
    if not callable(decide_with_analysis) or _analysis_capability_is_shadowed(policy):
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
