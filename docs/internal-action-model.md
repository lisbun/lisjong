# 内部Actionモデル

## 目的と位置付け

本書は、Issue #11「共通Policy境界と内部Action表現を設計する」のうち、
「3. 内部Actionモデル」の正本である。

- Policyの公開契約は[Policy契約](policy-contract.md)を正本とする
- Policy入力の許可fieldと意味契約は
  [Policy入力の最小スキーマ](policy-input-schema.md)を正本とする
- 上位の責務と依存方向は[Architecture](architecture.md)を正本とする
- RiichiEnv 0.4.8の公式情報、source確認、実測、推測・未確認事項、
  lisjongの設計判断の区別は
  [RiichiEnv調査記録](riichienv-investigation.md)を正本とする

本書は`InternalAction`のvariant、field、麻雀上の意味、不変条件を定める。
Python実装は`lisjong.policy_contract.action`を正本とし、semantic equality、
canonicalization、deduplication、外部合法Actionとの対応は
[Action identity](action-identity.md)を正本とする。

RiichiEnv調査記録で未実測とされているAction種別やRiichiLabオンライン経路を、
本書によって実測済みへ格上げしない。本書のvariantとfieldはlisjongの設計判断で
ある。

## 責務

`InternalAction`は、Policyが1 seat・1 decisionの`legal_actions`から選択する、
環境非依存の麻雀上の操作である。

```text
InternalAction = playerがそのdecisionで何を選択したか
```

RiichiEnv、RiichiLab、MJAI等の外部protocol固有型を持たない。次のようなActionの
実行結果およびゲーム進行は`InternalAction`へ含めない。

- 次の手番、phase、next player
- 局または対局の進行
- 河、meld、手牌等のstate更新そのもの
- 点数処理、役、符、翻、得点
- 局終了または対局終了の処理
- 複数playerの応答競合解決

プレイヤー操作、state mutation、手番更新、局進行を分離する。Local game runner
またはRiichiLab Clientが複数seatの判断をオーケストレーションし、外部環境が
Action適用後の進行を担う。

## Variant一覧

初期4人麻雀の`InternalAction`は次のvariantを持つ。

```text
InternalAction
├── DiscardAction
├── RiichiAction
├── ChiAction
├── PonAction
├── DaiminkanAction
├── AnkanAction
├── KakanAction
├── RonAction
├── TsumoAction
├── PassAction
└── KyuushuKyuuhaiAction
```

3人麻雀固有の操作は初期対象外である。自動成立する途中流局や、Action実行後の
結果eventはvariantへ含めない。

### Python表現

11 variantはそれぞれ独立した`@dataclass(frozen=True, slots=True)`として実装する。
共通Action base classと`ActionKind` fieldは設けない。`InternalAction`は11 classの
type alias unionである。各dataclassのvalue equalityがsemantic identityと一致し、
variant、`actor`、variant固有fieldを比較する。

constructorは単一Actionだけで検証できるAction値不変条件を検証する。
PolicyInput、materialized state、legal candidateとの照合が必要なContext整合条件は
各Action constructorへ取り込まず、Adapter / Policy呼び出し境界へ残す。

## 共通field: actor

すべてのvariantは次のrequired fieldを持つ。

```text
actor: Seat
```

`actor`は現在手番等の進行情報ではなく、そのActionを行う主体である。

```text
action.actor == DecisionContext.input.self_seat
```

この不変条件により、Action単体で主体を明示し、複数seatが同時に判断を要求された
場合の混線を検出できる。`turn`、`phase`、`next_player`はActionへ持たせない。

`actor`は[Action identity](action-identity.md)で、すべてのvariantのsemantic
identityへ含める。

## 不変条件の区分

不変条件は次の2種類を区別する。

| 区分 | 意味 | 例 |
| --- | --- | --- |
| Action値不変条件 | Action値だけで検証できる | tile枚数、`actor != target`、順子または刻子の構成 |
| Context整合条件 | 同じdecisionのPolicyInput、materialized state、legal candidateとの照合が必要 | 対象discardが直前のclaim対象であること、Kakan対象の元Ponが存在すること |

どちらも合法な内部Action候補を生成する境界で満たす必要がある。ただし、
具体的なContext整合検証の実装方法は本書で確定しない。semantic identityは
[Action identity](action-identity.md)を参照する。

## Tileに関する共通用語

すべてのvariantは、Policy入力と同じlisjong内部`Tile`概念を使用する。

- RiichiEnvのphysical tile IDではない
- MJAI文字列表現そのものではない
- 麻雀牌種を表す
- 赤牌を区別する
- 同じ牌種かつ同じ赤牌区分ならvalueとして等しい
- physical copy identityを持たない
- unknownまたは`?`を表す値を持たない

本書の「同じ麻雀牌種」は、赤牌か通常牌かの差を無視した基礎牌種が同じことを
意味する。一方、各fieldに保持する`Tile`値は赤牌差を失わない。

`Tile`は`TileType(category, rank)`と`is_red`から成るfrozen dataclassとして実装する。
Tile identityは基礎牌種と赤牌区分で構成し、physical copyを含めない。詳細は
[Action identity](action-identity.md)を参照する。

## DiscardAction

```text
DiscardAction
├── actor: Seat
├── tile: Tile
└── tsumogiri: bool
```

`actor`が`tile`を打牌する操作である。`tsumogiri`は、その打牌が現在のdrawn tileを
切る操作かを表す。

RiichiEnv 0.4.8の追加実測では、drawn tileと同じ牌種の牌を手牌に持つ局面で、
手出しとツモ切りが別の合法Actionとして存在し得ることを確認している。したがって、
`tsumogiri`を麻雀上の意味情報として保持する。

RiichiEnvのphysical tile IDは持たない。同じ通常牌のphysical copy間に麻雀上の
意味差がない場合、そのidentityを内部Actionへ持ち込まない。

## RiichiAction

```text
RiichiAction
└── actor: Seat
```

`actor`がリーチ宣言を開始する操作である。宣言牌を`RiichiAction`へ埋め込まない。

RiichiEnv 0.4.8の追加実測では、reach Action後にreach eventが発生し、別decisionで
宣言牌をdiscardする状態遷移を確認している。したがって、宣言牌は後続の
`DiscardAction`として表す。

`RiichiAction`自体は、宣言後のscore減算、供託棒増加、リーチ成立、宣言牌の
打牌結果を表さない。それらはstateおよびゲーム進行側の責務である。

## ChiAction

```text
ChiAction
├── actor: Seat
├── target: Seat
├── called_tile: Tile
└── consumed_tiles: 2 × Tile
```

`actor`が`target`の`called_tile`に対し、自席の`consumed_tiles`2枚を使ってchiする
操作である。

Action値不変条件は次である。

- `actor != target`
- `target`は`actor`の上家である
- `len(consumed_tiles) == 2`
- `called_tile`と`consumed_tiles`が有効な順子を構成する
- 字牌ではchiしない

Context整合条件として、`target`と`called_tile`は同じdecisionでclaim可能な直前の
打牌に対応し、`consumed_tiles`はactorのconcealed handから使用可能でなければ
ならない。

`consumed_tiles`は赤牌差を保持する。並び順に麻雀上の意味を持たせず、具体的な
semantic identityでは順序なしmultisetとしてcanonicalizeする。詳細は
[Action identity](action-identity.md)を参照する。

## PonAction

```text
PonAction
├── actor: Seat
├── target: Seat
├── called_tile: Tile
└── consumed_tiles: 2 × Tile
```

`actor`が`target`の`called_tile`に対し、自席の同じ麻雀牌種2枚を使用してponする
操作である。

Action値不変条件は次である。

- `actor != target`
- `len(consumed_tiles) == 2`
- `consumed_tiles`の2枚と`called_tile`は同じ麻雀牌種である

Context整合条件として、`target`と`called_tile`は同じdecisionでclaim可能な直前の
打牌に対応し、`consumed_tiles`はactorのconcealed handから使用可能でなければ
ならない。

赤牌差は保持する。`consumed_tiles`はsemantic identity上、順序なしmultisetとして
比較する。

## DaiminkanAction

```text
DaiminkanAction
├── actor: Seat
├── target: Seat
├── called_tile: Tile
└── consumed_tiles: 3 × Tile
```

`actor`が`target`の`called_tile`に対し、自席の同じ麻雀牌種3枚を使用して
daiminkanする操作である。

Action値不変条件は次である。

- `actor != target`
- `len(consumed_tiles) == 3`
- `consumed_tiles`の3枚と`called_tile`は同じ麻雀牌種である

Context整合条件として、`target`と`called_tile`は同じdecisionでclaim可能な直前の
打牌に対応し、`consumed_tiles`はactorのconcealed handから使用可能でなければ
ならない。

赤牌差は保持する。`consumed_tiles`はsemantic identity上、順序なしmultisetとして
比較する。

## AnkanAction

```text
AnkanAction
├── actor: Seat
└── tiles: 4 × Tile
```

`tiles`は、赤牌を区別した4枚のlisjong `Tile`値のmultisetである。

```text
(5p, 5p, 5p, 5pr)
```

上のように赤牌と通常牌を含む構成を保持できる。Action値不変条件は次である。

- `len(tiles) == 4`
- 4枚すべてが同じ麻雀牌種である

Context整合条件として、4枚はactorのconcealed handから使用可能であり、その
decisionでankanが合法でなければならない。

`tiles`の並び順に意味を持たせない。RiichiEnvのphysical tile IDを含めず、
semantic identityでは順序なしmultisetとしてcanonicalizeする。詳細は
[Action identity](action-identity.md)を参照する。

牌種だけのfieldにしない。`AnkanAction(actor, tile_kind)`では赤牌構成が失われるため、
PolicyInput、PublicMeld、InternalActionで共通する`Tile`概念を使用する。

## KakanAction

```text
KakanAction
├── actor: Seat
├── added_tile: Tile
├── from_seat: Seat
└── called_tile: Tile
```

`actor`が、`from_seat`の`called_tile`から成立した既存Ponへ`added_tile`を加え、
Kakanへ更新する操作である。Kakanは新規meldをゼロから作る操作ではない。

Action値不変条件は次である。

- `actor != from_seat`
- `added_tile`と`called_tile`は同じ麻雀牌種である

Context整合条件は次である。

- actorの現在meldに対応する元Ponが存在する
- 元Ponの`from_seat`と`called_tile`がActionのfieldに対応する
- `added_tile`をactorのconcealed handから使用できる
- 同じdecisionでそのKakanが合法である

次は導入しない。

- `source_meld_id`
- `source_meld_index`
- Python object identity
- `PublicMeld` objectへの直接参照

元Ponを外部または実装object identityで識別しない。同じContextで`from_seat`と
`called_tile`に対応する元Ponをちょうど1件へ照合する。0件または複数件なら
fail closedとする。semantic identityと外部合法Actionへの対応は
[Action identity](action-identity.md)を参照する。

Kakan後のmeld sequence位置はAction identityではなく結果stateの
canonicalizationであり、materialized stateの更新規則を設計する際に決定する。

## RonAction

```text
RonAction
├── actor: Seat
├── target: Seat
└── winning_tile: Tile
```

`actor`が`target`の牌`winning_tile`でronする操作である。Action値不変条件として
`actor != target`を満たす。

Context整合条件として、`target`と`winning_tile`は同じdecisionでron可能な対象に
対応し、そのronが合法でなければならない。

多家和では、和了する各seatについて独立した`RonAction`を表す。複数応答の競合、
採用順、点数処理はLocal game runner、RiichiLab Client、外部環境またはルール処理の
責務であり、Policyは他seat分をまとめて判断しない。

槍槓等も別の`ChankanAction`を設けず、槓を行ったplayerを`target`、槍槓対象牌を
`winning_tile`とする同じron操作として表現できる。役判定と和了理由は
`InternalAction`へ含めない。

## TsumoAction

```text
TsumoAction
├── actor: Seat
└── winning_tile: Tile
```

`actor`が`winning_tile`によってtsumoする操作である。`winning_tile`はツモ和了を
成立させる和了牌を表す。

通常ツモと嶺上ツモでは、`winning_tile == own_hand.drawn_tile`となることを期待できる。
ただし、この関係をすべての`TsumoAction`に対するAction値不変条件とはしない。
天和等も同じvariantで表し、`winning_tile`と`OwnHandState.drawn_tile`を常に同一の
概念とは定義しない。各環境から合法候補を生成できることはContext整合条件である。

`TenhouAction`、`RinshanAction`等の和了理由別variantは設けない。役、符、翻、点数、
点棒移動はInternalActionの責務ではない。

## PassAction

```text
PassAction
└── actor: Seat
```

`actor`が現在の応答機会で、提示された鳴き、和了等を行わないことを明示的に選ぶ
操作である。

`None`または空の`legal_actions`をpassとして扱わない。pass可能なdecisionでは、
`PassAction`自体を明示的な合法候補に含める。

## KyuushuKyuuhaiAction

```text
KyuushuKyuuhaiAction
└── actor: Seat
```

`actor`が九種九牌による途中流局を宣言する操作である。playerが合法Actionとして
明示的に選択する途中流局なので、初期`InternalAction`へ含める。

genericな`AbortiveDrawAction(reason=...)`は初期モデルで導入しない。九種九牌と、
四風連打、四家立直、四開槓等のgame stateから自動成立する途中流局との責務差を
維持するためである。

`InternalAction`にはplayerがlegal actionとして明示的に選択する操作だけを含め、
自動成立するゲーム状態または途中流局を含めない。将来、別のplayer選択型途中流局が
必要になった場合は専用variantを追加し、種類が増えた時点でgeneric化を再検討する。

## Actionと結果stateの分離

Actionとその実行結果を同じ型にしない。

```text
ChiAction
    ↓ 実行結果
PublicMeld(kind=CHI)

PonAction
    ↓ 実行結果
PublicMeld(kind=PON)

KakanAction
    ↓ 実行結果
既存PublicMeld(PON)をKAKANへ更新
```

Actionは今何を選ぶかを表し、PublicMeld等は現在どのような状態かを表す。
state mutation、手番更新、局進行はInternalActionへ埋め込まない。

## Action identityとの関係

variant、`actor`、variant固有のsemantic fieldから構成するidentity、Tileの赤牌区分、
`consumed_tiles`およびAnkan `tiles`のmultiset比較、Kakan元PonのContext照合、
Ron / Tsumoの`winning_tile`比較は
[Action identity](action-identity.md)を正本とする。

同じsemantic identityへ正規化される複数のexternal candidateはPolicyへ渡す前に
集約し、decision-local mappingで外部候補を保持する。Action dataclassのvalue
equalityをsemantic identityとし、`consumed_tiles`とAnkan `tiles`は生成時に
canonical tupleへ正規化する。Python hash値そのものをidentityの正本にせず、
別のcanonical keyやaction IDも設けない。外部環境ごとのrepresentative tie-breakは
後続実装で定める。
RiichiLab `possible_actions`の具体的なtranslationと照合規則は、未実測事項として
後続へ残す。

## 検証境界とtest観点

共通型の単体testとAdapter等の後続境界testを合わせ、少なくとも次を確認する。

- すべてのvariantがrequired `actor`を持つ
- `action.actor == DecisionContext.input.self_seat`
- Action値不変条件に違反する値を正常な合法候補として生成しない
- call Actionの`target`、`called_tile`、`consumed_tiles`が同じdecisionと整合する
- Discardの手出しとツモ切りを区別できる
- physical tile IDがInternalActionへ残らない
- 赤牌構成がDiscard、call、Ankan、Kakanで失われない
- Kakanが対応する既存Ponと一意に整合する
- 多家和の各Ronをseat別の独立Actionとして扱う
- passを`None`または空集合に変換しない
- 自動途中流局をplayer-selectable Actionとして生成しない
- Action適用結果のstateまたは進行情報をAction値へ混入させない

RiichiEnvで未実測のAction種別、実際のRiichiLabオンライン経路、action identityの
完全なround-tripは、それぞれ後続の実測およびtestで確認する。
