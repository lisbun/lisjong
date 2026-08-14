# Policy契約

## 目的と位置付け

本書は、Issue #11「共通Policy境界と内部Action表現を設計する」のうち、
Policyの公開契約を定める。上位の責務と依存方向は
[Architecture](architecture.md)、RiichiEnv 0.4.8の公式情報、実測、
推測・未確認事項、設計判断の区別は
[RiichiEnv調査記録](riichienv-investigation.md)を正本とする。
`DecisionContext`に含まれるPolicy入力の具体的な許可fieldと意味契約は、
[Policy入力の最小スキーマ](policy-input-schema.md)を正本とする。
Policyが選択する`InternalAction`のvariant、field、意味契約は、
[内部Actionモデル](internal-action-model.md)を正本とする。
`InternalAction`のsemantic identity、外部候補の集約、decision-local mappingは
[Action identity](action-identity.md)を正本とする。

本書の非空な合法手集合等の条件はlisjongの設計判断である。特に、
RiichiEnvの`legal_actions()`が常に1件以上を返すことを確認済みという意味では
ない。外部環境から空の合法手集合が渡され得るかは、引き続き未確認事項である。

## 基本契約

Policyは概念上、次の公開契約を持つ。

```python
class Policy(Protocol):
    def choose_action(
        self,
        decision: DecisionContext,
    ) -> InternalAction: ...
```

`Policy`、`choose_action`、`DecisionContext`、`InternalAction`を、
Issue #11の設計上の用語として一貫して使用する。本書はPython実装を追加せず、
具体的なclass表現やpackage構成を固定しない。

PolicyはRiichiEnv、RiichiLab、mjai、WebSocket固有型を受け取らず、それらの型を
返さない。

## 判断単位

Policyの論理的な判断単位は、1 seat × 1 decisionである。

RiichiEnv等から複数playerへ同時にActionが要求された場合、Local game runner
またはRiichiLab Clientが各seatを独立した判断単位へ分離し、Policyを個別に
呼び出す。

Policyは次を所有しない。

- 複数seatの呼び出し順序
- 複数seatの進行管理
- 対局loop

## DecisionContext

`DecisionContext`は、1 seat・1 decisionを表す、整合した不変スナップショット
である。

- Policy入力と`legal_actions`は、同じseat・同じ判断時点から生成する
- 1つのContext内で観測状態と合法手の時点をずらさない
- すべての`legal_actions.actor`はPolicy入力の`self_seat`と一致する
- Policy評価中にContextの意味内容を変更しない
- Policy自身もContextや合法候補を変更しない
- 外部環境が次の状態へ進んだ後、古いContextを新しいdecisionへ再利用しない

Policy入力の具体的なschema、不変性、canonicalizationは
[Policy入力の最小スキーマ](policy-input-schema.md)で定める。immutable class、
`tuple`、frozen dataclass等の具体的な実装方式は後続実装で決定する。

## legal_actionsの事前条件

Policyを正常に呼び出す時点で、`DecisionContext.legal_actions`は次を満たす。

- 1件以上存在する
- [Action identity](action-identity.md)で定義するsemantic identity上、
  候補同士が重複しない
- 並び順に契約上の意味を持たない
- list indexや順番をAction identityまたは優先順位として扱わない

pass / noneが合法な選択肢である場合は、空集合ではなく明示的なAction候補として
含める。`legal_actions = []`を暗黙のpassとして扱わない。

これらはlisjongがPolicyを呼び出すための事前条件であり、RiichiEnvの
`legal_actions()`に対する実測結果ではない。

## Policyの出力と事後条件

Policyは`InternalAction`を1件返す。返却Actionは
`DecisionContext.legal_actions`内の候補へ、action identity上で意味的に
ちょうど1件一致しなければならない。

照合は次に依存しない。

- Python object identity
- hash
- list index
- 合法手候補の並び順

`InternalAction`の具体的なvariantとfieldは
[内部Actionモデル](internal-action-model.md)、action identityの正規化規則は
[Action identity](action-identity.md)で定める。

## 合法性検証の責務

合法性検証は、異なる境界の条件を混同しないよう段階ごとに分離する。

| 段階 | 責務 |
| --- | --- |
| Policy呼び出し境界 | Policy返却Actionを`DecisionContext.legal_actions`へaction identityで照合し、意味的にちょうど1件一致することを確認する |
| RiichiEnv Adapter | 検証済み内部Actionを元のRiichiEnv合法Actionへ対応付け、外部環境へ返す前に再検証する |
| RiichiLab Client | オンライン経路で、送信Actionが`request_action.possible_actions`と整合することを送信前に再検証する |

Policy呼び出し境界の検証はPolicy実装自身へ重複実装させない。Local game runner
およびRiichiLab ClientがPolicyを呼び出す際に利用する共通責務とするが、
具体的なclass名、wrapper、module構成は本書で確定しない。

semantic identityと共通の照合原則は
[Action identity](action-identity.md)で定める。RiichiLabの実際の
`possible_actions`との具体的なtranslation、serialization、照合規則には
未実測事項があるため、引き続き確定しない。

## Policyが所有してよい状態

Policyを完全なstateless objectには限定しない。次は保持してよい。

- model
- 固定されたmodel parameter
- 明示的なPolicy設定。Policy instanceへbindされた不変なMatchRulesを含み得る
- 1回のdecision中だけ使用する探索・推論用の一時状態
- 最終Action選択へ影響しないcache
- metrics
- statistics

境界の基準はmutableかimmutableかではなく、Policyの出力へ影響する呼び出し間状態を
暗黙の内部状態として保持しないことである。

固定MatchRulesとPolicy設定の分離および初期`DecisionContext`へrulesetを複製しない
方針は、[Policy入力の最小スキーマ](policy-input-schema.md)を参照する。

## Policyが隠れて所有してはいけない状態

Policyは少なくとも次を隠れた内部状態として所有しない。

- 前回までのPolicy呼び出し履歴によって次のActionを変更する状態
- seat別の局・対局進行状態
- 前回のObservation
- 前回event
- 対局step
- 複数seatの進行状態
- 隠れたPRNG状態
- `request_id`
- network、transport、session状態
- Policyの呼び出し順序によってActionを変更する状態

ゲーム進行上の情報が判断に必要な場合は、隠れた状態ではなく明示的なPolicy入力
として与える。将来recurrent model等で呼び出し間stateが必要になった場合は、
暗黙状態を追加せず、明示的なstate契約として別途設計する。

## 決定性

初期Policy契約は、最終的なAction選択について論理的な決定性を要求する。

同じ意味内容の`DecisionContext`、同じPolicy実装、同じmodel parameter・
明示設定、同じ宣言済み実行条件に対して、action identity上で意味的に同じ
`InternalAction`を返す。

GPUやhardware差等を含む内部数値計算のbit-exactな再現性までは要求しない。
RiichiEnv constructorや`reset(seed=...)`等の外部環境側seedをPolicy契約へ
持ち込まない。隠れたPRNG状態によって同一入力の結果が変わる設計は、
初期契約では採用しない。

## Policy評価失敗

次は正常なPolicy判断として扱わない。

- `legal_actions`が空
- `legal_actions`にaction identity上の重複がある
- Policyが例外を送出する
- Policyが想定した`InternalAction`以外を返す
- Policyが`None`を返す
- caller側のtimeoutまたはcancellation等によりPolicy判断が完了しない
- Policy返却Actionと合法候補の一致が0件
- Policy返却Actionと合法候補の一致が複数件

具体的な例外class、timeout値、retry方法は本書で確定しない。

## fail closed

Policy契約におけるfail closedは、検証されていないActionを外部環境へ送信しない
ことを意味する。必ずprocess全体を終了するという意味ではない。

失敗後にLocal game runnerを終了する、RiichiLab Clientを安全に切断する、
timeoutとして扱う、ログを記録する等の処理は、各callerまたは後続設計の責務と
する。

RunnerおよびClientが勝手に次へ置換して外部へ送信することは禁止する。

- 先頭の合法Action
- 暗黙のpass
- 任意のfallback Action

将来fallbackを導入する場合は、明示的なPolicyまたは明示的な契約として別途
設計する。

## 後続項目

次はIssue #20または各componentの後続実装Issueで決定し、本書では確定しない。

- Pythonでの具体的なaction equality、hash、canonical key表現
- `Tile`の具体符号化
- 外部候補のdeterministic representativeを選ぶ具体的なtie-break key
- Python package、module、class構成
- 具体的な例外class
- timeout値、retry方法
- asyncおよびbatch interface
- thread safetyの詳細
- recurrent model等の明示的な呼び出し間state契約
- RiichiLab `possible_actions`との具体的な照合規則

Policy入力の具体的なfieldと、raw event履歴を初期入力へ含めない判断は、
[Policy入力の最小スキーマ](policy-input-schema.md)で確定済みである。
`InternalAction`のvariant、field、麻雀上の意味、不変条件は、
[内部Actionモデル](internal-action-model.md)で確定済みである。
semantic identity、multiset canonicalization、外部候補のsemantic aggregation、
decision-local mapping、representative選択の要件は
[Action identity](action-identity.md)で確定済みである。
