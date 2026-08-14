# Action identity

## 目的と位置付け

本書は、Issue #11「共通Policy境界と内部Action表現を設計する」のうち、
「4. Action identity」の正本である。

- Policyの公開契約は[Policy契約](policy-contract.md)を正本とする
- Policy入力の許可fieldと意味契約は
  [Policy入力の最小スキーマ](policy-input-schema.md)を正本とする
- `InternalAction`のvariant、field、意味契約は
  [内部Actionモデル](internal-action-model.md)を正本とする
- 上位の責務と依存方向は[Architecture](architecture.md)を正本とする
- RiichiEnv 0.4.8の公式情報、source確認、実測、推測・未確認事項、
  lisjongの設計判断の区別は
  [RiichiEnv調査記録](riichienv-investigation.md)を正本とする

本書は、Policyが選ぶ麻雀上の操作を、外部engineのphysical objectや候補の並び順に
依存せず照合するsemantic identityを定める。具体的なPythonのequality、hash、
canonical keyの型、Tile符号化、package、module構成は確定しない。

RiichiEnvで未実測のAction種別やRiichiLabオンライン経路を、本書によって実測済みへ
格上げしない。本書のidentity規則はlisjongの設計判断である。

## Semantic identityの定義

2つのActionが、同じseat・同じdecisionにおいてPolicyから見て同じ麻雀上の選択肢を
表すとき、両者のsemantic identityは一致する。

```text
same semantic identity
    = same action variant
    + same actor
    + same variant-specific semantic fields
```

次はidentityの根拠にしない。

- Python object identity
- Python hash
- external objectへの参照
- physical tile copy ID
- source meld IDまたはindex
- legal candidateのlist index
- `legal_actions()`または`possible_actions`の並び順
- `request_id`等のtransport field

variantが異なるActionはidentityも異なる。共通fieldの`actor`もすべてのvariantで
identityへ含める。これにより、同じ操作内容でも別seatのActionを同一視しない。

## Tile identity

Action identityで使用する`Tile`は、次の意味で比較する。

```text
Tile identity = base tile kind + red distinction
```

- 萬子、筒子、索子、字牌の基礎牌種を区別する
- 赤牌と同じ基礎牌種の通常牌を区別する
- 同じ基礎牌種かつ同じ赤牌区分のphysical copyは区別しない
- RiichiEnvのphysical tile IDまたはMJAI文字列そのものをidentityにしない

したがって、通常5pと赤5pは一致しない。一方、同じ通常4sを表すphysical copy間の
差だけなら一致する。具体的なTile encodingとcanonical sort keyは後続実装で
定める。

## Variant別identity

| variant | semantic identityに含めるfield |
| --- | --- |
| `DiscardAction` | variant、`actor`、`tile`、`tsumogiri` |
| `RiichiAction` | variant、`actor` |
| `ChiAction` | variant、`actor`、`target`、`called_tile`、`consumed_tiles` multiset |
| `PonAction` | variant、`actor`、`target`、`called_tile`、`consumed_tiles` multiset |
| `DaiminkanAction` | variant、`actor`、`target`、`called_tile`、`consumed_tiles` multiset |
| `AnkanAction` | variant、`actor`、`tiles` multiset |
| `KakanAction` | variant、`actor`、`added_tile`、`from_seat`、`called_tile` |
| `RonAction` | variant、`actor`、`target`、`winning_tile` |
| `TsumoAction` | variant、`actor`、`winning_tile` |
| `PassAction` | variant、`actor` |
| `KyuushuKyuuhaiAction` | variant、`actor` |

Action値不変条件とContext整合条件は
[内部Actionモデル](internal-action-model.md)を正本とする。identity一致は合法性を
単独で証明しない。variant固有の意味fieldが一致しても、同じdecisionで合法である
ことを別途検証する。

### Discard

`tsumogiri`はsemantic identityへ含める。

```text
Discard(actor=0, tile=5p, tsumogiri=False)
!=
Discard(actor=0, tile=5p, tsumogiri=True)
```

RiichiEnv 0.4.8では、drawn tileと同じ牌種について手出しとツモ切りが別の合法
Actionとして存在することを実測している。一方、通常牌のphysical copy差だけは
Policyの選択肢へ持ち込まない。

### Chi、Pon、Daiminkan

`consumed_tiles`は順序なしmultisetとして比較する。

- 要素の並び順はidentityへ影響しない
- Tileの重複枚数を保持する
- 赤牌構成を保持する
- physical copy IDを保持しない

例えば、`(4m, 5mr)`と`(5mr, 4m)`は同じmultisetである。一方、
`(4m, 5m)`とは赤牌構成が異なるため一致しない。

### Ankan

`tiles`4枚も順序なしmultisetとして比較する。4枚の順序差だけはidentityへ
影響しないが、赤牌構成と重複枚数は保持する。

### Kakan

Kakanは、現在のContextに存在する元Ponを更新する操作として照合する。

```text
Kakan identity
    = variant
    + actor
    + added_tile
    + from_seat
    + called_tile
```

同じdecisionの`PolicyInput.players[actor].melds`から、少なくとも次を満たす元Ponを
検索する。

- meld kindがPonである
- `from_seat`がActionと一致する
- `called_tile`がTile identity上でActionと一致する
- meldの基礎牌種と`added_tile`の基礎牌種が一致する

一致する元Ponはちょうど1件でなければならない。0件または複数件ならContext整合
違反としてfail closedとし、未検証Actionを外部へ送らない。

元Ponの全構成牌を`KakanAction`へ重複保持しない。赤牌を含む元Ponの構成は同じ
Contextの`PublicMeld.tiles`に保持され、元PonはContext上で一意に解決される。

Kakan後の`PublicMeld`をmeld sequenceのどの位置へ置くかは、Action identityでは
なく結果stateのcanonicalizationである。本書では確定せず、materialized stateの
更新規則と同期testを設計する際に決定する。

### RonとTsumo

Ronは`target`と`winning_tile`、Tsumoは`winning_tile`をidentityへ含める。
和了理由、役、符、翻、得点はidentityへ含めない。

RiichiEnvにおけるron、tsumoの完全なAction / MJAI round-tripと、RiichiLabの
winning tile表現は未実測事項を含む。外部候補からrequired fieldを取得できない
場合、外部境界は未確認値を推測して補完しない。同じseat-visibleなdecision
contextから明示的に導出できることを検証するか、変換をfail closedとする。

## Multiset canonicalization

`consumed_tiles`およびAnkanの`tiles`は、順序ではなくTile identityごとの個数で
比較する。

```text
multiset identity = each Tile identity and its multiplicity
```

実装はcanonical sort済みsequenceまたはcount表現等を使用できるが、具体的な
Python表現は固定しない。どの表現でも次を満たす。

- 同じ要素と個数なら入力順にかかわらず一致する
- 赤牌と通常牌を別要素として数える
- physical tile copyの差を要素へ持ち込まない
- Python hashだけをidentityの正本にしない

## RiichiEnv ActionからInternalActionへの代表変換例

本節は新しいAction設計を追加するものではない。RiichiEnv 0.4.8の実測記録と、
本書および[内部Actionモデル](internal-action-model.md)で確定済みの設計を対応付け、
Issue #11の設計確認例として整理する。

### 実測した範囲

[RiichiEnv調査記録](riichienv-investigation.md)では、各`Observation`の
`legal_actions()`について次の経路を実行した。

```text
Action
  -> Action.to_mjai()
  -> json.loads()
  -> Observation.select_action_from_mjai()
  -> Action
```

90 stepの実行経路で、通常牌打牌、赤牌を含む打牌、chi、pon、noneが出現し、
round-tripに成功した。赤牌のMJAI表現として`5pr`、`5sr`が出現した。一方、ron、
tsumo agari、riichi、kan各種はこの一括probeに出現せず、完全なround-tripは
未確認である。RiichiLab `possible_actions`との変換も未実測である。

### 確定済み設計への正規化

次の表の「実測」はAction種別とround-tripの確認範囲を表す。「正規化」は、
その実測を根拠としてlisjongで確定した設計判断である。保存されていない局面の
physical tile ID、`actor`、`target`、`consumed_tiles`の具体値は例示しない。

| RiichiEnv 0.4.8で実測した操作 | 変換境界が取得・検証する意味情報 | lisjongへの正規化結果 |
| --- | --- | --- |
| 通常牌打牌 | 同じdecisionのRiichiEnv `Action`、`Observation`、seat-visible contextから`actor`、牌種、赤牌区分、手出し / ツモ切りを取得・検証する。physical tile copy IDは除外する | `DiscardAction(actor, tile, tsumogiri)` |
| 赤牌打牌 | MJAI表面では`5pr`、`5sr`等が実測された。境界でMJAI文字列そのものではないlisjong `Tile`へ変換し、red distinctionを保持する。physical tile copy IDは除外する | `DiscardAction(actor, tile, tsumogiri)`。`tile`が赤牌区分を保持する |
| chi | 同じdecisionのRiichiEnv `Action`、`Observation`、seat-visible contextから`actor`、`target`、`called_tile`、自席から使用する2枚を取得・検証する。各牌をlisjong `Tile`へ変換する | `ChiAction(actor, target, called_tile, consumed_tiles)` |
| pon | 同じdecisionのRiichiEnv `Action`、`Observation`、seat-visible contextから`actor`、`target`、`called_tile`、自席から使用する2枚を取得・検証する。各牌をlisjong `Tile`へ変換する | `PonAction(actor, target, called_tile, consumed_tiles)` |
| none | 外部のnoneを、Policyが値を返さない状態ではなく、当該seatが応答機会をpassする明示的な選択として正規化する | `PassAction(actor)` |

chiおよびponの`consumed_tiles`はphysical copy identityを持たず、赤牌構成と重複枚数を
保持する。semantic identityでは順序なしmultisetとして比較する。

noneを`None`、空の`legal_actions`、Actionなしへ変換しない。passが合法なdecision
では、`PassAction(actor)`を明示的なlegal candidateとしてPolicyへ渡す。

### `to_mjai()`だけに依存しない変換

上記round-tripは外部変換経路の実測であり、RiichiEnv Adapterの正規化algorithmを
`to_mjai()`だけで構成するという意味ではない。

RiichiEnv 0.4.8の追加実測では、同じ牌種の別physical Discard Actionが同一の
MJAI打牌へ変換されながら、`is_drawn`等の差によって手出し / ツモ切りという
semantic differenceを持つケースを確認した。このため、変換境界は同じdecisionの
RiichiEnv `Action`、`Observation`、seat-visible contextを合わせて使用し、
`DiscardAction.tsumogiri`を取得・検証する。

通常牌のphysical copy差だけで麻雀上の意味差がない候補は同じsemantic groupへ
集約できる。一方、手出し / ツモ切り差、通常牌 / 赤牌差、chi / ponの
`consumed_tiles`の赤牌構成差は別のsemantic identityとして保持する。

## External Actionのsemantic aggregation

外部環境は、同じsemantic identityへ正規化される複数のphysical candidateを返す
場合がある。RiichiEnv 0.4.8では、同じ通常牌のphysical copyを表す複数のDiscard
Actionが、同じMJAI打牌へ変換される例を実測している。

Policyへ渡す前に、外部候補をsemantic identityでgroup化する。

```text
external legal candidates
        ↓ normalize and validate
semantic groups
        ↓ one InternalAction per group
DecisionContext.legal_actions
```

各groupは次を保持する。

```text
semantic InternalAction -> one or more external candidates
```

Policyへは各groupを代表する`InternalAction`を1件だけ提示する。これにより、
`DecisionContext.legal_actions`はPolicy契約が要求するidentity上の重複禁止を満たす。

aggregationは、差がPolicyから不要という理由だけで行わない。各external candidateが
同じ麻雀上の選択と、lisjongが観測・送信する範囲で同じ外部意味結果を表すことを
境界側で確認できる場合だけ行う。赤牌、手出し / ツモ切り、consumed構成、target等に
意味差がある候補は別groupとする。同値性を確認できない差は潰さない。

## Decision-local mapping

semantic groupとexternal candidateの対応は、1 seat・1 decisionだけに有効な
境界側の一時mappingとする。

- Policyへ渡さない
- Policyのhidden stateにしない
- requestやtransport fieldをInternalActionへ混入させない
- 外部環境が次のstateへ進んだ後に再利用しない
- 別seatまたは別decisionの候補を混在させない

Policy返却Actionを受け取った境界は、次を順に確認する。

1. Policy返却値が有効な`InternalAction`である
2. semantic identityが当該decisionのgroupへちょうど1件一致する
3. 一致groupにexternal candidateが1件以上存在する
4. group内からdeterministicなrepresentativeを選ぶ
5. representativeを現在の外部合法候補へ送信前に再検証する

0件一致、複数group一致、空group、stale mapping、再検証失敗はいずれも
fail closedとする。

## Deterministic representative

1 semantic groupに複数のexternal candidateがある場合、境界側は安定した外部fieldに
基づく全順序でrepresentativeを決定する。

次へ依存しない。

- random selection
- object identityまたはmemory address
- Python hash iteration order
- 外部candidate listの受信順またはindex

external physical tile ID等は、Policyへ公開せず、同値なexternal candidateを解決する
境界内部のtie-breakに使用してよい。ただし、そのfieldが同じdecision内で安定し、
一意な全順序を構成できることを確認する。安定したrepresentativeを定義できない
外部候補群は、任意の候補へfallbackせずfail closedとする。

RiichiEnvおよびRiichiLabそれぞれの具体的なtie-break keyは、Adapter / Clientの
実装と外部schemaの実測に合わせて後続で定める。

## Revalidationとfail closed

semantic identity一致は、外部環境へ送信可能であることの最終確認ではない。

```text
Policy internal legality
        ↓
decision-local external mapping
        ↓
RiichiEnv external legality
        ↓
RiichiLab online legality
```

- Policy呼び出し境界は、返却Actionを内部合法候補へidentityで照合する
- RiichiEnv Adapterは、選択groupのrepresentativeを元のRiichiEnv合法Actionへ
  対応付け、外部へ返す前に再検証する
- RiichiLab Clientはオンライン経路で、送信Actionを
  `request_action.possible_actions`へ送信前に再検証する

fail closedは、検証されていないActionを外部へ送信しないことを意味する。
先頭候補、暗黙のpass、任意のfallback Actionへ置換しない。具体的な例外class、
終了、切断、timeout処理は各callerまたは後続実装で定める。

## RiichiEnvとRiichiLabの共通原則

identity、semantic aggregation、decision-local mapping、revalidation、fail closedは
lisjong共通の原則とする。外部表現とのtranslationは境界ごとに分離する。

```text
RiichiEnv Action
    ↕ RiichiEnv Adapter
InternalAction semantic identity
    ↕ RiichiLab Client boundary
RiichiLab possible_action
```

RiichiLab側も同じsemantic identityへnormalizeする。ただし、次は未実測または
未確定である。

- `possible_actions`の具体schema
- 同じsemantic identityのpossible actionが複数存在するか
- physical identityの有無
- 赤牌、tsumogiri、consumedの具体表現
- Kakan source情報の具体表現
- Ron / Tsumo winning tileの具体表現
- representative選択が必要になるケースとtie-break key
- 具体的なserializationと照合規則
- transport、session、timeoutの詳細

RiichiEnvで確認した事実をRiichiLabでも確認済みとは扱わない。RiichiLab候補から
semantic identityに必要な情報を安全に得られない場合、未確認値を既定値で補わず、
明示的に導出可能であることを検証するかfail closedとする。

## 牌譜学習への引継ぎ

raw牌譜がphysical tile identityを保持する場合も、Policyのsemantic action spaceへ
同じ差をそのまま持ち込まない。教師信号として意味のある次の差を保持する。

- Tileの基礎牌種
- 赤牌区分
- Discardの手出し / ツモ切り
- callまたはAnkanの構成牌multiset
- variant、actor、target、鳴き元、和了牌等のvariant固有field

physical copy差だけの教師ラベルは同じsemantic identityへ正規化する。raw dataの
保存形式と学習datasetのversioningは本書では確定しない。

## 後続実装testへの引継ぎ

少なくとも次をtest観点とする。

- 同variant、同actor、同semantic fieldならidentityが一致する
- variant差またはactor差があればidentityが一致しない
- 通常牌と赤牌の差があればidentityが一致しない
- 同じ通常牌のphysical copy差だけならidentityが一致する
- Discardの`tsumogiri`差があればidentityが一致しない
- RiichiEnv Discard変換が`to_mjai()`だけに依存せず、手出し / ツモ切り差を保持する
- RiichiEnv赤牌打牌がMJAI文字列ではなく、red distinctionを持つlisjong `Tile`へ
  変換される
- `consumed_tiles`の順序差だけならidentityが一致する
- `consumed_tiles`の赤牌構成または重複枚数差があればidentityが一致しない
- RiichiEnv chi / pon変換でphysical copy identityを除外し、赤牌構成を保持する
- Ankan `tiles`の順序差だけならidentityが一致する
- RiichiEnv noneが`None`や空集合ではなく、明示的な`PassAction`へ変換される
- Kakan元Ponが0件または複数件ならfail closedとなる
- 複数external candidateから1つのsemantic InternalActionを生成する
- 意味差を確認できないexternal candidateを同じgroupへ集約しない
- Policy返却Actionに対応するsemantic groupまたはexternal candidateがなければ
  fail closedとなる
- decisionまたはseatをまたいでmappingを再利用しない
- representative選択が同じ外部候補集合に対してdeterministicである
- representative選択がlist順、object identity、hash iterationへ依存しない
- RiichiLab固有の未確認事項をRiichiEnv実測で代用しない

## 引き続き未確定の事項

次は後続実装、実測、または別の設計項目で確定する。

- Pythonでの具体的なequality、hash、canonical key表現
- `Tile`の具体符号化とcanonical sort key
- RiichiEnvおよびRiichiLabごとのrepresentative tie-break key
- Kakan後の`PublicMeld`のsequence位置
- decision-local mappingの具体的な型と所有component内の実装構造
- RiichiLab `possible_actions`の具体schema、translation、serialization、照合規則
- 具体的な例外class、timeout、終了または切断処理
- Adapter、Policy、Runner、Clientの本実装
