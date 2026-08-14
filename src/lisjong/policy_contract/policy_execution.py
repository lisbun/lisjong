"""環境非依存のPolicy呼び出し・返却値validation境界。

1 seat x 1 decisionだけを扱う。受け取った``DecisionContext``を変更せず
``Policy.choose_action()``へ渡し、返却値をAction dataclassの既存value equalityで
同Contextの合法候補へ照合する。外部Action変換、transport、timeout、retry、
fallback選択は扱わない。
"""

from lisjong.policy_contract.action import InternalAction, _is_internal_action
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy import Policy


class PolicyActionValidationError(Exception):
    """Policy返却値をちょうど1件の合法Actionとして検証できない場合。"""


def execute_policy(policy: Policy, decision: DecisionContext) -> InternalAction:
    """Policyを呼び出し、一意に一致したcanonicalな合法候補を返す。

    ``Policy.choose_action()``が送出した例外は変更せず伝播する。返却値が
    ``InternalAction``でない、安全に比較できない、または候補へちょうど1件一致
    しない場合は``PolicyActionValidationError``を送出する。代替Actionやfallbackは
    選択しない。
    """
    if not isinstance(decision, DecisionContext):
        raise TypeError("decision must be a DecisionContext")

    selected = policy.choose_action(decision)

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
