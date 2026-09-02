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
learned Policy向けのfixed-size action vocabulary、codec、legal maskは
[Model-facing action vocabulary](action-vocabulary.md)を正本とする。

本書の非空な合法手集合等の条件はlisjongの設計判断である。特に、
RiichiEnvの`legal_actions()`が常に1件以上を返すことを確認済みという意味では
ない。外部環境から空の合法手集合が渡され得るかは、引き続き未確認事項である。

## 基本契約

Policyは`lisjong.policy_contract.policy`で、次のstructural `Protocol`として
実装する。

```python
class Policy(Protocol):
    def choose_action(
        self,
        decision: DecisionContext,
    ) -> InternalAction: ...
```

`Policy`、`choose_action`、`DecisionContext`、`InternalAction`を公開契約の用語
として一貫して使用する。`Policy`は明示的な継承を要求せず、
`@runtime_checkable`を付けない。Protocolが型として表現するのは
`choose_action`の引数と戻り値であり、決定性、hidden state非依存、合法候補との
semantic match等は、型シグネチャだけでは強制できないbehavioral contractである。

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

`DecisionContext`は`input: PolicyInput`と`legal_actions`を持つfrozen dataclass
として実装する。`legal_actions`は入力順を変更せずtupleへ正規化し、生成時に
非空、全Actionのactor一致、semantic identity上の重複禁止を検証する。

RiichiEnv経路では、Adapterの`build_decision()`が同じseat・同じ
`Observation`からIssue #28の`PolicyInput`とIssue #29のdecision-local mappingを
生成し、mappingのsemantic unique candidateを`legal_actions`として本型へ渡す。
返される`RiichiEnvDecision.context`だけがPolicy入力であり、対応mappingが保持する
raw RiichiEnv ActionはPolicyへ渡さない。

Policy入力の具体的なschema、不変性、canonicalizationは
[Policy入力の最小スキーマ](policy-input-schema.md)で定める。同じdecisionへの同期や
各Actionの麻雀上の合法性等、外部stateを必要とするContext整合条件は
`DecisionContext` constructorへ取り込まず、Adapter等の境界で検証する。

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

`InternalAction`のsemantic identityは、実装では各variantのdataclass value
equalityと一致する。照合は次に依存しない。

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
およびRiichiLab Clientは、`lisjong.policy_contract.execute_policy(policy,
decision)`を共通境界として利用する。この関数は受け取った`DecisionContext`を
そのまま`Policy.choose_action()`へ渡し、返却値を既存のdataclass value equalityで
`decision.legal_actions`へ照合する。ちょうど1件一致した場合は、Policyが返した
objectではなく`legal_actions`側のcanonicalな候補を返す。

Policy返却値が`InternalAction`でない、安全に比較できない、または一致件数が0件・
複数件の場合は`PolicyActionValidationError`を送出する。Policy自身が送出した例外は
捕捉・置換せず、そのままcallerへ伝播させる。いずれの失敗でもfallback Actionを
返さない。

semantic identityと共通の照合原則は
[Action identity](action-identity.md)で定める。RiichiLabの実際の
`possible_actions`との具体的なtranslation、serialization、照合規則には
未実測事項があるため、引き続き確定しない。

## Model-facing action vocabularyとの関係

learned Policyは固定長のaction出力を持つため、`InternalAction`と固定長vector上の
numeric indexを対応付ける表現を必要とする。Issue #149で追加した
`lisjong.action_vocabulary`は、その対応付けだけを所有するoptionalなadapter層で
あり、本書のPolicy契約を変更しない。意味契約の正本は
[Model-facing action vocabulary](action-vocabulary.md)である。

```text
semantic identity
    = InternalAction dataclass value equality

model action index
    = versioned adapter representation
```

- `Policy`の`choose_action()`は引き続き`InternalAction`を返す。model action index
  を返すPolicy interfaceは追加しない
- model action indexは新しいAction identityではなく、合法性の根拠でもなく、
  `legal_actions`のtuple indexでもない
- `resolve_legal_action()`は同じdecisionの`legal_actions`側のcanonical
  `InternalAction` objectを返す。Policy実装はそれをそのまま返し、
  `execute_policy()`のvalidationを迂回しない
- `execute_policy()`のsignature、validation、例外semanticsは変更しない。
  vocabularyのfail closedは`ActionVocabularyError`階層で表し、
  `PolicyActionValidationError`のsemanticsを変更しない
- codecとlegal maskは`DecisionContext.legal_actions`と`input.self_seat`だけを
  読む。`PolicyInput` / `DecisionContext`へ新しい情報を追加しない

```text
DecisionContext
    -> fixed-size legal mask
    -> model-selected action index
    -> canonical legal InternalAction
    -> execute_policy() の既存validation
```

feature encoder、tensor schema、HandBelief consumer seam、model architecture、
trainingは後続Issueで扱い、本書とvocabulary contractへ先行して固定しない。

## DecisionTrace / AnalysisTrace

Issue #97で、1回のPolicy decisionをone-wayで観測するcontractを追加した。
observability契約であり、Policy契約の入力側は一切拡張しない。

責務差は次のとおりで、相互にschemaを混ぜない。

```text
GameTrace
    = what happened in execution

DecisionTrace
    = what canonical action lisjong selected for one Policy decision

AnalysisTrace
    = which typed lisjong intermediate values were actually produced / used
      in that Policy decision
```

ADR 0002以降のcurrent ownershipでは、objective `GameTrace`は`lisjong-arena`、
`DecisionTrace` / `AnalysisTrace`は`lisjong`が所有する。

```text
AnalysisTrace is output / observation, not Policy input.
AnalysisTrace does not own the semantics of AI intermediate values.
```

`AnalysisTrace`はfree-form telemetryではなく、immutableでtypedな
lisjong-owned semantic payloadとする。`dict[str, object]`、
`dict[str, float]`、`Mapping[str, object]`、`reason: str`のような表現、および
free-form natural-language reasoningをcanonical schemaにしない。向聴数、
受け入れ、belief、value / riskといったintermediate valueのsemanticsは、各AI
domain value側（例: `TwoStepUkeireCandidateEvaluation`）を正本とし、
`AnalysisTrace`側へ複製しない。

root contractがruntime検証するのは、`AnalysisTrace` subclassかつfrozen
dataclassであるという最低限の構造条件だけである。これはfree-form dict /
string / mutable payloadを排除するための境界であり、deep immutabilityまでは
保証しない。field値まで含めたimmutabilityとdetachmentは、各concrete analysis
payload側の責務とする。

`DecisionTrace`は1 decisionを表すimmutable valueで、次だけを持つ。

- `legal_actions`: Policyへ提示された`decision.legal_actions`のimmutable snapshot
- `selected_action`: 既存validation後のcanonicalな合法`InternalAction`
- `analysis`: 許可されたtyped analysis、またはanalysis未生成を表す`None`

value自身は、`legal_actions`が非空でsemantic重複を持たないこと、全要素が
`InternalAction`であること、`selected_action`が`legal_actions`へちょうど1件
semantic matchすること、`analysis`が許可されたtyped payloadまたは`None`である
ことだけを構造的に検証する。麻雀ルール上の合法性は再検証せず、
`DecisionContext`や実行境界の責務を重複実装しない。

`analysis`の`None`は「analysisを生成していない」ことを表す。評価結果の`0`、
empty evaluation、neutral scoreの意味へ流用しない。

`DecisionTrace`が保持してよいのは次の3つだけである。

1. `DecisionContext`としてPolicyへ合法的に提示された情報
2. Policy自身がそこから実際に導出したtyped intermediate value
3. validation後のcanonical selected action

他家の実手牌、山 / 王牌の実状態、環境のprivileged state、GameTraceの
privileged observer data、未来のevent、offline ground truth、Arena固有の
state / metricを混入しない。`DecisionContext`へtrace、sink、GameTrace、observer、
privileged stateを追加しない。DecisionTrace導入を理由にPolicy-visible
informationを拡張しない。

`DecisionTrace`はgame-global sequence、GameTrace sequence、GameTrace join ID、
environment event IDを持たない。保証するのは同一`DecisionTraceRecorder`内の
notification orderだけであり、Action equalityやtuple indexを暗黙のjoin keyとして
契約化しない。

## trace付きexecutionと analysis capability

既存の`execute_policy(policy, decision)`は変更しない。名前、2引数signature、
例外semantics、canonical legal-action validationをすべて維持し、trace用引数も
追加しない。traceを利用しない既存caller、既存Policyは一切変更不要である。

trace付きexecutionは別のopt-in APIとして追加する。

```python
execute_policy_with_trace(
    policy: Policy,
    decision: DecisionContext,
    sink: DecisionTraceSink,
) -> InternalAction
```

両APIはlegal-action validation logicを二重実装せず、共通のprivate validation
pathだけを使う。

```text
execute_policy()
        \
         +--> common private validation --> canonical action
        /
execute_policy_with_trace()
```

`Policy.choose_action()`の契約も変更しない。analysisを提供できるPolicyは、
optional capability methodを追加してよい。

```python
class AnalysisCapablePolicy(Protocol):
    def choose_action(
        self,
        decision: DecisionContext,
    ) -> InternalAction: ...

    def choose_action_with_analysis(
        self,
        decision: DecisionContext,
    ) -> PolicyDecision: ...
```

`PolicyDecision`は、Policyが提案したActionとoptional typed analysisを持つ
immutable valueである。

```text
PolicyDecision.action
    = Policyが提案したAction

DecisionTrace.selected_action
    = validation後のcanonical legal Action
```

この2つを同一視しない。

traced execution境界は、capabilityのdispatchをmethod名の有無だけで決めない。
MRO上のmethod ownerを見て、次のとおり扱う。

```text
同じclassが両方を定義している
    -> analysis capabilityを使う

subclassがanalysis pathの内側だけをoverrideしている
    -> analysis capabilityを使う

subclassが`choose_action()`だけをoverrideし、
analysis capabilityを基底classからinheritしているだけ
    -> analysis capabilityを使わない
    -> subclass自身の`choose_action()`へfallbackし、analysisは`None`

subclassがanalysis capabilityを明示overrideしている
    -> analysis capabilityを使う
```

これにより、subclassが基底classのanalysis methodを偶然inheritした結果、trace
有無で異なるdecision algorithmを通ることがない。

analysis-capable Policyは次を守る。

- trace取得のためにPolicyを2回実行しない
- `choose_action()`とanalysis-capable pathへdecision algorithmを二重実装しない
- analysis生成のために向聴数、受け入れ等のintermediate valueを再計算しない
- `policy.last_analysis`等のdecision間mutable stateをcanonical transportにしない
- trace有効 / 無効でsemantic selected actionを変えない
- subclassが基底classのanalysis pathを偶然inheritした結果、自分のdecision
  semanticsが変わる設計にしない

1回のdecision計算からactionとimmutable analysisの両方を得る。

### traced executionの順序と失敗時の扱い

```text
Policy decision once
        ↓
Policy result validation
        ↓
canonical legal action
        ↓
DecisionTrace construction
        ↓
sink.on_decision(trace)
        ↓
return canonical selected action
```

Policy自身の例外、またはPolicy返却値のvalidation失敗では、DecisionTraceを
emitしない。`DecisionTraceSink.on_decision()`が例外を送出した場合は、その例外を
黙って握り潰さず、原則として変更せず伝播する。このときfallback Actionを返さず、
sink失敗をtrace成功として扱わず、Policyも再実行しない。generic sink内部の
atomicityまでは実行境界が保証しない。

### DecisionTraceSinkとDecisionTraceRecorder

`DecisionTrace`は1 decisionごとに完成済みvalueとして生成されるため、GameTraceの
start / event / completeのlifecycleは持ち込まない。

```python
class DecisionTraceSink(Protocol):
    def on_decision(
        self,
        trace: DecisionTrace,
    ) -> None: ...
```

標準in-memory recorderは次を保証する。

- record 0件では`snapshot() == ()`
- `snapshot()`はimmutable tupleを返す
- 同一recorderへのnotification orderを保持する
- snapshot取得後にrecordが追加されても、取得済みsnapshotは変化しない
- 正常な1回の`on_decision()`につき1件だけrecordする

### non-interference

同じPolicy、同じ意味内容の`DecisionContext`について、`execute_policy(...)`と
標準recorderを渡した`execute_policy_with_trace(...)`は、semantic selected action
を一致させる。trace有無をPolicy input、decisionへ影響するfeature flag、
tie-break input、hidden mutable stateとして使用しない。

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

Policy返却値のvalidation失敗は`PolicyActionValidationError`で表す。Policy自身の
例外は変更せず伝播する。timeout値、retry方法は本書で確定しない。

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

## Python実装で確定した契約

共通Policy契約型は`src/lisjong/policy_contract/`へ配置する。

- `Policy`は最小のstructural `Protocol`とする
- `DecisionContext`、`PolicyInput`、それらを構成するstate値は、再帰的に
  immutableなfrozen dataclassまたはEnumとする
- `InternalAction`は共通base classや`ActionKind` fieldを持たない11個の独立した
  frozen dataclassとし、`InternalAction`はそれらのtype alias unionとする
- Actionのdataclass value equalityをsemantic identityとし、別のaction IDや
  canonical keyを設けない
- 順序なしmultiset fieldだけを生成時にcanonical tupleへ正規化し、履歴、公開順、
  seat位置を持つsequenceは並べ替えない
- `policy_execution.py`の`execute_policy()`は`DecisionContext`だけをPolicyへ渡し、
  一意に一致した`legal_actions`側の`InternalAction`を返す
- Policy返却値を検証できない場合は`PolicyActionValidationError`でfail closedし、
  Policy自身の例外は変更せず伝播する
- `analysis_trace.py`の`AnalysisTrace`はtyped analysis payloadのroot contract
  とし、concrete analysis型は`lisjong.policies`側が所有する。`policy_contract`
  から具体Policy実装へ逆依存しない
- `decision_trace.py`の`DecisionTrace`、`DecisionTraceSink`、
  `DecisionTraceRecorder`と、`policy_decision.py`の`PolicyDecision`、
  `AnalysisCapablePolicy`は、いずれもimmutable value / structural Protocolとし、
  Policy runtimeのmutable stateを保持しない
- `execute_policy_with_trace()`はopt-in APIとし、`execute_policy()`と同じprivate
  validation pathだけを共有する
- learned Policy向けのmodel-facing action vocabularyは、`policy_contract`ではなく
  `lisjong.action_vocabulary`が所有する。`policy_contract`側の型、`execute_policy()`
  のvalidation、例外semanticsは変更しない

## 引き続き未確定の項目

次は各componentの後続実装Issueで決定し、本書では確定しない。

- 外部候補のdeterministic representativeを選ぶ具体的なtie-break key
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
