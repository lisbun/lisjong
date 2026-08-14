# Policy入力の最小スキーマ

## 目的と位置付け

本書は、Issue #11「共通Policy境界と内部Action表現を設計する」のうち、
「2. Policy入力の最小スキーマ」の正本である。

- Policyの公開契約は[Policy契約](policy-contract.md)を正本とする
- 上位の責務と依存方向は[Architecture](architecture.md)を正本とする
- RiichiEnv 0.4.8の公式情報、source確認、実測、推測・未確認事項、
  lisjongの設計判断の区別は
  [RiichiEnv調査記録](riichienv-investigation.md)を正本とする

本書はPolicyへ渡す情報を許可リスト方式で定義する。Pythonの具体的なclass、
dataclass、enum、collection型、package、module構成は確定しない。
`InternalAction`の全schemaとaction identityは、Issue #11の後続項目
「3. 内部Action表現」と「4. Action identity」で確定する。

## DecisionContext

`DecisionContext`は、1 seat・1 decisionを表す、整合した不変スナップショット
である。概念上、次の構造を持つ。

```text
DecisionContext
├── input: PolicyInput
└── legal_actions: immutable sequence of InternalAction
```

`PolicyInput`は当該seatから観測可能な状態を正規化したsnapshotである。
`legal_actions`は同じdecisionで選択できる内部Action候補であり、
`PolicyInput`へ重複保持しない。

追加の不変条件は次である。

```text
all legal_actions.actor == input.self_seat
```

`legal_actions`の非空、action identity上の重複禁止、順序に契約上の意味を
持たせないこと、pass / noneを明示候補とすることは、Policy契約を正本とする。

## PolicyInputの概念schema

初期の`PolicyInput`を次の許可fieldで構成する。

```text
PolicyInput
├── self_seat
│
├── round
│   ├── round_wind
│   ├── hand_number
│   ├── dealer_seat
│   ├── honba
│   ├── riichi_sticks
│   ├── dora_indicators
│   └── live_wall_tiles_remaining
│
├── players[4]
│   ├── score
│   ├── discards
│   │   ├── tile
│   │   ├── tsumogiri
│   │   ├── order
│   │   └── called_by
│   │
│   ├── melds
│   │   ├── kind
│   │   ├── tiles
│   │   ├── from_seat
│   │   └── called_tile
│   │
│   └── riichi
│
└── own_hand
    ├── concealed_tiles
    └── drawn_tile
```

具体的なPython型は後続実装で定義する。

## Seat

`self_seat`はlisjong内部の4人麻雀seatを表す。

```text
Seat = 0, 1, 2, 3
```

RiichiEnvの`player_id`と数値が一致する場合も、RiichiEnv型または外部identityを
Policy契約へ公開する意味ではない。外部seatとの対応付けはRiichiEnv Adapter
またはRiichiLab Client側の境界が所有する。

席順は次の関係を持つ。

```text
(seat + 1) mod 4 = 下家
```

自風は、現在の親との相対位置から導出できる。

```text
seat_wind_index = (self_seat - dealer_seat) mod 4
```

`PolicyInput`は自席をseat 0へrotateしない。自席基準の表現が必要なPolicyまたは
model encoderが、入力境界の内側で変換する。

## RoundState

`round`は次のrequired fieldを持つ。

```text
RoundState
├── round_wind
├── hand_number
├── dealer_seat
├── honba
├── riichi_sticks
├── dora_indicators
└── live_wall_tiles_remaining
```

### round_wind

場風を表すlisjong内部の麻雀ドメイン値である。RiichiEnv固有の整数値を
Policy契約とせず、境界側で変換する。

### hand_number

場風内の局番号を`1..4`の1-based値で表す。RiichiEnvの`kyoku_index`等が
0-basedの場合は、境界側で変換する。

### dealer_seat

現在の親を`Seat`で表す。他fieldから導出できる場合もPolicyInputでは明示保持する。
導出結果と明示値が矛盾する入力は、境界側の契約違反である。

### honba

現在の本場数を表す。

### riichi_sticks

現在供託されているリーチ棒の本数を表す。

### dora_indicators

現在公開済みのドラ表示牌だけを、公開された順序で保持する。

次は含めない。

- 裏ドラ
- 未公開の槓ドラ
- その他の非公開牌

### live_wall_tiles_remaining

現在のdecision時点で、live wallから今後取得可能な残り牌数を表す。
環境固有containerのサイズではなく、麻雀ドメイン上の意味である。
`len(env.wall)`そのものをPolicy契約にしない。

RiichiEnv調査記録では、通常ツモとdaiminkan、ankan、kakan後の嶺上ツモに伴う
wall変化をRiichiEnv 0.4.8で実測している。一方、localとonlineの双方で同じ値を
生成する具体的なcounter algorithmは未確定であり、後続のAdapter実装とtestで
検証する。fieldの意味契約と具体的な算出方法を分離する。

## PlayerPublicState

`players`は4要素固定で、自席を含む全seatの公開状態を保持する。

```text
len(players) == 4
players[index]のindex == Seat
```

各要素は次の構造を持つ。

```text
PlayerPublicState
├── score
├── discards
├── melds
└── riichi
```

他家のconcealed handを保持するfieldは設けない。

### score

当該seatの現在の公開点数を表す。

## Discard

`discards`は、現在河に残っている牌だけではなく、そのseatがこの局で行った
打牌履歴である。鳴きに利用された打牌も履歴から削除しない。

RiichiEnv 0.4.8の追加実測では、chiされた打牌が
`Observation.discards`から削除されないことを確認した。この実測範囲を、
RiichiEnvの全versionおよび全局面へ一般化しない。

各entryは次の意味を持つ。

```text
Discard
├── tile
├── tsumogiri
├── order
└── called_by
```

### tile

打牌されたlisjong内部`Tile`である。

### tsumogiri

その打牌がツモ切りだったかを表す。

RiichiEnv 0.4.8の追加実測では、drawn tileと同じ牌種について、ツモ切りと
手出しが別の合法Actionとして存在することを確認した。取得できない値を
`False`等で補完してはならない。

### order

局内の全seatに共通する0-basedの打牌通番である。最初の打牌を0とし、打牌が
発生するたびに1増える。decision番号またはturn countではない。全seatの打牌間の
時間関係を表現する。

### called_by

chi、pon、daiminkanでそのdiscardが利用された場合のcaller `Seat`を保持し、
利用されていない場合は`None`とする。鳴かれてもdiscard entryを削除しない。

初期契約ではronを`called_by`に含めない。ronは副露ではなく局終了Actionであり、
その後のPolicy decisionで河snapshotとして利用しないためである。

## PublicMeld

`melds`はevent履歴ではなく、そのplayerが現在保持する副露・槓状態のsnapshot
である。

```text
PublicMeld
├── kind
├── tiles
├── from_seat
└── called_tile
```

### kind

少なくとも次を区別できる。

```text
CHI
PON
DAIMINKAN
ANKAN
KAKAN
```

具体的なenum表現は後続で定義する。

### tiles

meldを構成する牌のmultisetである。表示順に意味を持たせず、canonical
representationへ正規化する。赤牌は区別する。

### from_seat

Action種別ごとの意味は次である。

```text
CHI / PON / DAIMINKAN -> Seat
ANKAN                 -> None
```

kakanでは元ponの`from_seat`を維持する。

### called_tile

Action種別ごとの意味は次である。

```text
CHI / PON / DAIMINKAN -> Tile
ANKAN                 -> None
```

kakanでは元ponの`called_tile`を維持する。

### kakan時のmeld順序

`melds`は現在状態の所定の順序で保持する。ただし、kakanで既存ponを更新した際に、
sequence上の位置を元pon成立時の位置に維持するか、kakan発生時点へ移すかは
未確定である。Issue #11の「3. 内部Action表現」または
「4. Action identity」で必ず固定する。

## RiichiState

単純な`riichi_declared: bool`ではなく、次の状態を区別する。

```text
RiichiState
├── NONE
├── DECLARED
└── ACCEPTED
```

### NONE

reach Actionを実行する前の状態である。

### DECLARED

reach Actionを実行済みだが、まだ成立前の状態である。

RiichiEnv 0.4.8の追加実測では、reach event発生後、宣言牌discard前に、
`riichi_declared == False`、score減算前、`riichi_sticks`増加前の独立した
Observationを確認した。

### ACCEPTED

リーチ成立後の状態である。

同じ追加実測では、宣言牌discard後の同一`step()`内で`reach_accepted`まで進み、
次のObservationで`riichi_declared == True`、宣言者のscoreが1000点減少し、
`riichi_sticks`が1増加していた。

### リーチ宣言牌位置

`riichi_sutehais`は追加実測で更新を確認できなかったため、取得元として依存しない。
初期`RiichiState`へ`declaration_discard_order`をrequired fieldとして追加しない。

将来`ippatsu_active`等を追加する場合は、リーチ宣言牌位置または
`declaration_discard_order`相当が必要になる可能性がある。これは後続拡張へ
引き継ぐ既知依存である。

## OwnHandState

自席だけの非公開情報を、他家の公開情報から分離して保持する。

```text
OwnHandState
├── concealed_tiles
└── drawn_tile
```

### concealed_tiles

現在自席が保持するconcealed tiles全体である。自摸直後はdrawn tileも含み、
副露牌は含まない。順序に意味を持たせず、canonical orderへ正規化する。

### drawn_tile

現在のdecisionに対応するdrawn tileである。追加の1枚として数えず、
`concealed_tiles`内のmetadataとして扱う。

```text
drawn_tile is not None
    =>
drawn_tileがmultisetとしてconcealed_tiles内に存在する
```

鳴き後の打牌判断等、対応するdrawn tileがない場合は`None`とする。通常ツモと
嶺上ツモは、いずれも現在のdrawn tileという同じ意味で扱う。

## Tileの意味契約

具体的な符号化はIssue #11の後続項目で確定する。初期schemaでは次の意味を
固定する。

- lisjong内部の麻雀牌値である
- RiichiEnvのphysical tile IDではない
- MJAI文字列表現そのものではない
- 牌種を表す
- 赤牌を区別する
- 同じ牌種かつ同じ赤牌区分なら、valueとして等しい
- physical copy identityを持たない
- unknownまたは`?`を表す値を持たない
- `PolicyInput`と`InternalAction`で同じ`Tile`概念を使用する
- deterministicなcanonical orderを定義できる

34-index、37-index、enum、dataclass等の具体表現は後続で決定する。

## Materialized state

RiichiEnv AdapterおよびRiichiLab Clientは、seat-visibleなObservationとevent deltaを
継続的に処理し、`PolicyInput`生成に必要な現在状態を正規化してmaterializeして
よい。

```text
RiichiEnv / RiichiLab
        ↓
seat-visible Observation + event delta
        ↓
境界側 normalized materialized state
        ↓
immutable DecisionContext
        ↓
Policy
```

これはPolicyのhidden recurrent stateではない。境界側が保持してよいのは、
PolicyInputへ必要なseat-visibleな現在状態を外部表現から正規化するためのstateに
限る。

次はmaterialized stateへ含めない。

- Policyの過去判断
- AI内部memory
- 他家の非公開情報
- `env.mjai_log`
- 完全な山情報
- Policy判断
- 対局loopそのもの
- transport固有情報をPolicy入力化した値

### 同期不変条件

`DecisionContext`生成時には、次の3者を同じseat・同じdecision時点まで同期する。

```text
materialized state
Observation
legal_actions
```

機械的な検証方法は、後続のAdapter実装とtestで具体化する。同期条件を後続Issueの
明示的なtest項目へ引き継ぐ。

## Canonicalization

不変化するだけでなく、意味的に同じ局面を同じ`PolicyInput`へ正規化する。

| collection | canonicalization規則 |
| --- | --- |
| `players` | seat 0から3の順 |
| `discards` | 各seatの打牌順。各entryはglobal `order`を持つ |
| `melds` | 現在meldの所定の順序。kakan更新時の順序規則は後続で確定 |
| `dora_indicators` | 公開順 |
| `concealed_tiles` | 意味上順序なし。canonical order |
| `PublicMeld.tiles` | multisetとしてcanonical representation |
| `legal_actions` | priorityとしての順序なし。不変sequence |

意味的に同じ`PolicyInput`は、同じcanonical representationを持つ。
Pythonの`__eq__`等の具体実装は後続で定義する。

## Immutable snapshot

`PolicyInput`と`DecisionContext`は再帰的な値snapshotである。外部環境が所有する
`Observation`、mutable `list`、event buffer、live object等をそのまま参照しない。
Policy入力から、外部環境所有の可変objectへ到達できないようにする。

具体的な`tuple`、frozen dataclass等は後続実装で決定する。不変性は情報境界の
代替ではない。非公開情報はimmutable化して渡すのではなく、そもそも
PolicyInputへ含めない。

## MatchRulesとPolicy設定

固定rulesetはdecisionごとの`PolicyInput`へ複製せず、Policy instanceを1つの
明示的かつ不変なMatchRulesまたはPolicy configurationへbindする。

少なくとも次のように、合法性または戦略評価へ影響する固定ruleを暗黙のglobal
stateにしない。

- 赤牌
- 喰いタン
- 対局長またはgame mode
- 終了・延長条件
- 順位評価に必要なルール
- 飛び条件
- その他の固定rule

同じPolicy instanceの対局途中でrulesetを変更せず、異なるrulesetへ黙って
使い回さない。これはPolicy契約で認める明示的なPolicy設定に含まれる。
各`DecisionContext`への`ruleset_id`追加は初期契約では行わない。

## 初期schemaに含めない情報

次は初期`PolicyInput`へ含めない。

### 外部型と完全ログ

- raw RiichiEnv `Observation`
- `Observation.to_dict()`全体
- `env.mjai_log`
- raw MJAI eventまたはevent履歴

### transportとsession

- RiichiLab `request_id`
- `possible_actions`そのもの
- WebSocket
- timeout
- transportまたはsession状態

### 非公開情報

- 他家のconcealed hand
- 完全な山の並び
- 裏ドラ
- 未公開の槓ドラ
- unknownまたはmasked tile

### 今回採用しない状態と派生特徴

- `turn_count`
- `own_furiten`
- `ippatsu_active`
- waits
- `is_tenpai`
- shanten
- ukeire
- 期待値
- 危険度
- その他の評価済み特徴量

`own_furiten`は通常、同巡内、リーチ後見逃し等を安全に統合する確認が不足して
いる。ronの合法性は`legal_actions`で表す。未確認値を`False`で補完しない。
`ippatsu_active`は将来拡張とする。waits、`is_tenpai`、shanten、ukeire等は
派生・評価情報であり、Policy契約を外部engineの評価結果へ依存させない。

### 拡張用field

- generic `extras`またはarbitrary `dict`
- runtime `schema_version`

## InternalAction設計への引継ぎ

RiichiEnv 0.4.8の追加実測では、同じ牌種について、ツモ切りと手出しが別の
RiichiEnv `Action`でありながら、`to_mjai()`が同一になるケースを確認した。

したがって、Discard `InternalAction`はphysical tile identityを使用せず、
少なくともtileとtsumogiriの意味差を保持できなければならない。

本書では`InternalAction`の全schema、最終的なaction identity、`consumed`の
正規化規則は確定しない。

## 後方互換性

Policy入力は許可リスト方式を維持し、`extras`を設けない。

- 既存fieldの意味を黙って変更しない
- field追加は明示的な契約変更とする
- required field追加は原則breaking changeとする
- fieldの削除、rename、意味変更はbreaking changeとする
- optional field追加も自動的に後方互換とは扱わない

runtime `schema_version`は現時点で導入しない。将来、model artifact、dataset、
Replay、cross-process Policy、plugin API、persistent serialization等で互換性が
必要になった時点でversioningを設計する。model artifact側metadataへ入力schemaの
識別子を保持する方式は将来候補である。

## 後続実装testへの引継ぎ

少なくとも次を後続実装のtest観点とする。

- `players`は4要素で、indexが`Seat`に一致する
- すべての`legal_actions.actor`が`self_seat`に一致する
- `drawn_tile`が`None`でない場合、multisetとして`concealed_tiles`内に存在する
- 他家hidden handまたは山内容だけが異なる局面から、同じ`PolicyInput`を生成する
- `PolicyInput`からraw `Observation`または`env.mjai_log`へ到達できない
- 同じ意味局面はcanonicalization後に同じ`PolicyInput`となる
- `Discard.order`は局内で一意であり、打牌ごとに単調増加する
- 鳴かれたdiscardは履歴から消えず、`called_by`で表現する
- 公開済みのdora indicatorだけを含む
- materialized state、Observation、`legal_actions`が同じdecision時点で同期する

## 引き続き未確定の事項

次は後続設計または実装で確定する。

- Pythonの具体的なclass、enum、collection、package、module構成
- `Tile`の具体符号化
- `InternalAction`の全schema
- action identityと`consumed`の正規化規則
- kakanで既存ponを更新した際のmeld sequence上の位置
- 将来`ippatsu_active`等を追加する場合のリーチ宣言牌位置の表現
- `live_wall_tiles_remaining`をlocalとonlineで生成する具体的なcounter algorithm
- materialized state、Observation、`legal_actions`の同期を機械的に検証する方法
- `own_furiten`、`ippatsu_active`、正規化済みevent等の将来拡張
