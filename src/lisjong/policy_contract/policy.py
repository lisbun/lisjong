"""lisjong内部のPolicy契約。

docs/policy-contract.md「基本契約」「Policyの出力と事後条件」「決定性」
「Policyが所有してよい状態」「Policyが隠れて所有してはいけない状態」
「fail closed」の意味契約を実装する。

`Policy`はduck typing / structural typingで適合可能なstructural Protocol
である。明示的な継承を要求しない。`@runtime_checkable`は付けていない。
`isinstance(obj, Policy)`は`choose_action`というmethod名の有無しか検査
できず、引数・戻り値の型、決定性、legal match等の契約は検証できないため、
「Policy契約をruntimeで検証できる」という誤解を招く。現時点でruntime
`isinstance`判定が必要な具体的ユースケースもない。

`choose_action`のtype signature（引数・戻り値の型）だけがこのProtocolで
強制される契約であり、以下はPythonの型システムでは表現できない
behavioral contractとして、実装者が守るべき責務にとどまる。実行時の違反
検出はここでは実装せず、後続のPolicy契約testおよび最小Policy実装Issueへ
残す。共通base classやtemplate method（`choose_action`を呼び出し先で
wrapしてvalidationを挟む設計等）もここでは導入しない。

- 意思決定に`DecisionContext`以外の入力を使わない。RiichiEnv
  Observation、外部Action、PRNG、seed、runner state、以前のdecision等を
  追加引数・隠れた依存として使わない
- 前回呼び出しの結果、呼び出し順序、hidden PRNG状態等の隠れた可変状態に
  依存しない。ただし、不変なmodel parameter、明示的なPolicy設定、
  1回のdecision中だけ使う一時状態、最終選択へ影響しないcache / metrics /
  statisticsは保持してよい
- 同じ意味内容の`DecisionContext`に対して、action identity上で意味的に
  同じ`InternalAction`を返す（決定性）。GPU/hardware差を含む内部数値計算の
  bit-exactな再現性までは要求しない
- 返却する`InternalAction`は、`decision.legal_actions`内の候補へ、
  action identity（このcodebaseではdataclass value equalityと同じ）上
  ちょうど1件一致する
- legal_actions外のActionや、契約違反を隠すarbitrary fallback
  （例: 先頭合法手、暗黙のpass）を返さない
"""

from typing import Protocol

from lisjong.policy_contract.action import InternalAction
from lisjong.policy_contract.decision_context import DecisionContext


class Policy(Protocol):
    """1 seat・1 decisionの意思決定契約。

    `choose_action`は`DecisionContext`だけを受け取り、`InternalAction`を
    1件返す。RiichiEnv、RiichiLab、mjai、WebSocket固有型を引数・戻り値
    いずれにも持たない。
    """

    def choose_action(self, decision: DecisionContext) -> InternalAction: ...
