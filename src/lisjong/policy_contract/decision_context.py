"""lisjong内部のDecisionContext型。

docs/policy-contract.md「DecisionContext」「legal_actionsの事前条件」、
docs/policy-input-schema.md「DecisionContext」の意味契約を実装する。

```text
DecisionContext
├── input: PolicyInput
└── legal_actions: immutable sequence of InternalAction
```

正本文書がPolicy呼び出し時点でのmust-level条件として明記する次を、この型
自身の構造的不変条件として生成時に検証する。いずれもDecisionContext自身が
直接持つ2 fieldだけで判定でき、Adapter履歴等の外部stateを必要としない。

- legal_actionsが1件以上存在する
- 全legal_actionsのactorがinput.self_seatと一致する
  （`all legal_actions.actor == input.self_seat`）
- legal_actions同士がsemantic identity上重複しない

一方、次はここでは検証しない。DecisionContextを麻雀ルールvalidatorへ
膨らませないため、Adapter / environment側のlegal action materialization
責務として残す。

- 各legal actionが実際に麻雀上合法か（例: DiscardAction.tileがown_handに
  存在するか、ChiAction.called_tileが直前discardか、RonAction.winning_tileが
  本当に和了牌か等）
- Ron/Tsumoの特殊優先順位付け、Passの自動補完、和了actionの優先mark、
  RiichiとDiscardの統合

legal_actionsの並び順は、docs/policy-contract.mdが「並び順に契約上の意味を
持たない」「list indexや順番をAction identityまたは優先順位として扱わない」
と明記するとおり、入力sequenceをそのままtuple化して保持する。これは境界側が
渡した順序を勝手にsortしないという意味であり、Policyが順序を意思決定の
意味として利用してよいという意味ではない。
"""

from dataclasses import dataclass

from lisjong.policy_contract.action import InternalAction, _is_internal_action
from lisjong.policy_contract.policy_input import PolicyInput


def _normalize_legal_actions(values: object) -> tuple[InternalAction, ...]:
    """iterableをtupleへ正規化する。並び順は一切変更しない。"""
    try:
        items = tuple(values)
    except TypeError:
        raise TypeError("legal_actions must be an iterable") from None
    if any(not _is_internal_action(item) for item in items):
        raise TypeError("legal_actions must contain only InternalAction instances")

    return items


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """1 seat・1 decisionを表す、整合した不変スナップショット。"""

    input: PolicyInput
    legal_actions: tuple[InternalAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.input, PolicyInput):
            raise TypeError("input must be a PolicyInput")

        legal_actions = _normalize_legal_actions(self.legal_actions)

        if not legal_actions:
            raise ValueError("legal_actions must not be empty")

        if any(action.actor != self.input.self_seat for action in legal_actions):
            raise ValueError("all legal_actions must have actor == input.self_seat")

        if len(set(legal_actions)) != len(legal_actions):
            raise ValueError(
                "legal_actions must not contain duplicate semantic actions"
            )

        object.__setattr__(self, "legal_actions", legal_actions)
