"""`DecisionContext.legal_actions`から導出するfixed-size legal maskとresolve。

docs/action-vocabulary.md「Legal mask」「Resolve」の意味契約を実装する。

```text
DecisionContext
    -> fixed-size legal mask
    -> model-selected action index
    -> canonical legal InternalAction
```

maskは純粋なPython contract（`tuple[bool, ...]`）として表現し、NumPy等のML
runtimeを持ち込まない。learned Policy側でtensorへ変換するのはconsumerの責務で
あり、その変換semanticsは本Issueのscope外である。

いずれのAPIも`DecisionContext.legal_actions`と`input.self_seat`だけを読む。
自席手牌、他家の非公開情報、外部engine objectを参照しない。並び順にも依存せず、
`legal_actions`のpermutationは結果を変えない。
"""

from lisjong.action_vocabulary.action_codec import (
    ACTION_VOCABULARY_SIZE,
    ACTION_VOCABULARY_VERSION,
    _require_supported_version,
    _require_vocabulary_index,
    encode_action,
)
from lisjong.action_vocabulary.errors import (
    ActionIndexCollisionError,
    IllegalActionIndexError,
)
from lisjong.policy_contract.action import InternalAction
from lisjong.policy_contract.decision_context import DecisionContext


def _require_decision_context(decision: object) -> DecisionContext:
    if not isinstance(decision, DecisionContext):
        raise TypeError("decision must be a DecisionContext")
    return decision


def encode_legal_actions(
    decision: DecisionContext, *, version: str = ACTION_VOCABULARY_VERSION
) -> dict[int, InternalAction]:
    """当該decisionのlegal actionsを`index -> canonical legal Action`へencodeする。

    値は常に`decision.legal_actions`側のobjectそのものであり、equalな別objectでは
    ない。返すmappingはcaller所有の新しいdictで、index昇順に反復する。したがって
    `legal_actions`のpermutationは、集合としてもiteration orderとしても結果を
    変えない。

    同一decision内で2つのlegal actionsが同じindexへ衝突した場合は、どちらかを
    採用せず`ActionIndexCollisionError`でfail closedする。
    """
    _require_supported_version(version)
    _require_decision_context(decision)

    encoded: dict[int, InternalAction] = {}
    for action in decision.legal_actions:
        index = encode_action(action, version=version)
        if index in encoded:
            raise ActionIndexCollisionError(
                f"legal actions collide at action index {index}: "
                f"{encoded[index]!r} and {action!r}"
            )
        encoded[index] = action

    return dict(sorted(encoded.items()))


def build_legal_action_mask(
    decision: DecisionContext, *, version: str = ACTION_VOCABULARY_VERSION
) -> tuple[bool, ...]:
    """fixed-sizeなlegal maskを返す。

    長さは常に`ACTION_VOCABULARY_SIZE`であり、trueなindexの集合は
    `encode_legal_actions()`のkey集合と完全一致する。
    """
    encoded = encode_legal_actions(decision, version=version)
    return tuple(index in encoded for index in range(ACTION_VOCABULARY_SIZE))


def resolve_legal_action(
    index: int,
    decision: DecisionContext,
    *,
    version: str = ACTION_VOCABULARY_VERSION,
) -> InternalAction:
    """model-selected indexを、同じdecisionのcanonical legal Actionへ解決する。

    返り値は`decision.legal_actions`側のobjectである。model indexを新しいAction
    identityとして扱わず、`execute_policy()`のvalidationも迂回しない。Policy実装は
    ここで得たActionをそのまま返し、既存の実行境界が改めて合法候補へ照合する。

    次はいずれもfail closedとし、他のActionへ置換しない。

    - 未対応のvocabulary version
    - int以外、またはvocabulary範囲外のindex
    - vocabulary上は有効だが当該decisionでlegalでないindex（mask上illegal）
    - 同一decision内でindexが衝突しているlegal actions
    """
    _require_supported_version(version)
    _require_decision_context(decision)
    _require_vocabulary_index(index)

    encoded = encode_legal_actions(decision, version=version)
    action = encoded.get(index)
    if action is None:
        raise IllegalActionIndexError(
            f"action index {index} is not legal in this decision"
        )
    return action
