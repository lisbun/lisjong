# Model-facing action vocabulary

## 目的と位置付け

本書は、Issue #149「Learned Policy Stage 0 — InternalAction ↔ model-facing action
vocabularyとlegal maskの契約を確立する」で確定した、fixed-sizeかつversionedな
model-facing action vocabulary、`InternalAction`とのcodec、
`DecisionContext.legal_actions`から導出するlegal maskの正本である。

- Policyの公開契約は[Policy契約](policy-contract.md)を正本とする
- `InternalAction`のsemantic identityは[Action identity](action-identity.md)を
  正本とする
- `InternalAction`のvariant、field、意味契約は
  [内部Actionモデル](internal-action-model.md)を正本とする
- Policy入力の許可fieldと意味契約は
  [Policy入力の最小スキーマ](policy-input-schema.md)を正本とする
- 上位の責務と依存方向は[Architecture](architecture.md)を正本とする

learned Policyは固定長のaction出力を持つため、`InternalAction`と固定長vector上の
numeric indexを対応付ける表現が必要になる。本書とその実装
（`src/lisjong/action_vocabulary/`）はその対応付けだけを所有する。

feature encoder、tensor schema、HandBelief consumer seam、model architecture、
behavior cloning、RL / self-play、teacher data生成、Arena evaluation、model
artifact formatは本書のscope外であり、後続Issueで扱う。本vocabularyは、後続の
modelがweightsとvocabulary semanticsを対応付けられる最小のversion contractまでを
提供する。

## Semantic identityとmodel action indexの区別

```text
semantic identity
    = InternalAction dataclass value equality

model action index
    = versioned adapter representation
```

model action indexは新しいAction identityではない。次のいずれとしても扱わない。

- Action identityの正本または代替canonical key
- 麻雀上の合法性の根拠
- `DecisionContext.legal_actions`のtuple index
- 永続的なAction ID

[Action identity](action-identity.md)が「別のcanonical key / action IDを設けない」
と定めるのはsemantic identityについての契約であり、本書のindexはそれを置換しない。
indexからActionへ戻す`resolve_legal_action()`は、同じdecisionの
`legal_actions`側のcanonical `InternalAction` objectを返す。Policy実装は、
resolveしたActionをそのまま返し、既存の`execute_policy()`のvalidationを迂回しない。

## Vocabulary version

```text
ACTION_VOCABULARY_VERSION = "lisjong-action-vocabulary-1"
ACTION_VOCABULARY_SIZE    = 802
```

version identityは、vocabularyの意味とnumeric assignmentの組を表す。実装は同時に
1つのversionだけを提供し、他のversion値はfail closedで拒否する
（`UnsupportedActionVocabularyVersionError`）。

次のいずれかを変更する場合は、必ずversion文字列も更新する。

- vocabulary size
- blockの順序またはindex range
- block内の列挙順
- fieldのencoding規則（tile order、相対seat定義、赤牌構成の表現等）
- 扱うAction variantの集合
- context復元に使うfieldの取り決め

同じversionの範囲では、numeric assignmentはstableである。size、block range、
bijectionの検証だけではblock内の列挙順（tile順、`tsumogiri`の順、赤牌構成の順
など）の入れ替えを検出できず、同じversionのweightsが別Actionへdecodeされ得る。
そのためtestは、全indexのcanonical semanticsから生成したfingerprintを
`version -> fingerprint`のliteralとして固定する。

```text
_VOCABULARY_FINGERPRINTS = {
    "lisjong-action-vocabulary-1":
        "543c6bca832069dd88b22554b8546ddcd958840a7be7ed291b4ebab6302d7952",
}
```

fingerprintはindexごとの意味を、実装内部のkey表現とは独立に文字列化して
sha256したものである。layoutや列挙順を変更するとversionを更新しない限りCIが
失敗する。意図的な変更では`ACTION_VOCABULARY_VERSION`を更新し、新しいversionの
fingerprintを追加する（既存entryは書き換えない）。各blockの先頭 / 末尾indexの
意味も、変更内容を人手で追跡できるanchorとして併記する。

後続のmodel artifactはweightsと同じ場所へこのversionを記録し、読み込み時に
照合する。artifact formatそのものは本書で確定しない。

## Index layout

vocabularyはvariantごとの連続blockへ分割する。

| block | index range | size | block内の構成 |
| --- | --- | --- | --- |
| `DiscardAction` | 0–73 | 74 | tile(37) × `tsumogiri`(2) |
| `RiichiAction` | 74 | 1 | 単一index |
| `ChiAction` | 75–164 | 90 | suit(3) × 順子最小rank(7) × called位置(3) × 5の赤牌区分 |
| `PonAction` | 165–311 | 147 | 相対target(3) × 49 |
| `DaiminkanAction` | 312–476 | 165 | 相対target(3) × 55 |
| `AnkanAction` | 477–522 | 46 | 基礎牌種(34) × 4枚中の赤牌枚数 |
| `KakanAction` | 523–651 | 129 | 相対from_seat(3) × 43 |
| `RonAction` | 652–762 | 111 | 相対target(3) × winning tile(37) |
| `TsumoAction` | 763–799 | 37 | winning tile(37) |
| `PassAction` | 800 | 1 | 単一index |
| `KyuushuKyuuhaiAction` | 801 | 1 | 単一index |
| 合計 | 0–801 | 802 | |

blockは重複せず、`range(0, 802)`を過不足なく分割する。到達不能な組み合わせ
（赤牌になり得ない牌種の赤牌flag等）はindexを消費しないため、range内にhole、
unreachable index、意図しないaliasを作らない。実装は
`ACTION_VOCABULARY_BLOCKS`（variant型 → `range`のread-only mapping）として同じ
layoutを公開する。

### Tile order

37種のTile値（34基礎牌種 + 赤5m / 赤5p / 赤5s）は、
`lisjong.policy_contract.tile.tile_sort_key`と同じcanonical順に並べる。

```text
manzu 1,2,3,4,5,5r,6,7,8,9
pinzu 1,2,3,4,5,5r,6,7,8,9
souzu 1,2,3,4,5,5r,6,7,8,9
honor 1..7 (東南西北白發中)
```

赤牌区分を持たない基礎牌種の順序（34種）は、上記から赤牌を除いた並びとする。
vocabulary側で独自のtile順序を発明せず、既存canonical sort keyへ従う。

### Actor

`actor`はvocabularyへ含めない。`DecisionContext`のすべてのlegal actionは
`input.self_seat`をactorとするため、actorはdecode時のcontextから一意に復元できる。
同じsemantic操作は、どのseatが行っても同じindexになる。

### Target / from seat

`target`と`from_seat`はabsolute seatではなく、actorからの相対位置でencodeする。

```text
relative = (target - actor) mod 4
1 = 下家, 2 = 対面, 3 = 上家
```

`ChiAction.target`はActionの値不変条件により常にactorの上家であるため、Chi block
はtargetをindexへ含めず、decode時にactorから復元する。`PonAction`、
`DaiminkanAction`、`KakanAction`、`RonAction`は相対位置をindexへ含める。

### Discard

`discard index = 2 × tile index + (1 if tsumogiri else 0)`。赤5の打牌と通常5の打牌、
同一牌種の手出しとツモ切りは、いずれも別indexになる。

### Chi

Chiは順子全体を(suit, 順子の最小rank, called位置, 5の赤牌区分)へ正規化する。

- suit: 萬子 / 筒子 / 索子（字牌はChiの値不変条件で存在しない）
- 最小rank: 1..7
- called位置: 最小rankからのoffset 0..2。`consumed_tiles`は残りの2枚
- 5の赤牌区分: 順子が5を含む場合だけ2値。順子は3つの連続する異なるrankから成り、
  5は高々1枚しか含まれないため、赤牌構成は1 bitで損失なく表現できる。その5が
  called tile側かconsumed側かはcalled位置から一意に決まる

同一called tileに対する複数のconsumed pair（例: 5mに対する`(3m,4m)` / `(4m,6m)` /
`(6m,7m)`）は、called位置が異なるため別indexになる。「chi = 1 index」への集約は
行わない。

### Pon / Daiminkan

called tileとconsumed tilesは同じ基礎牌種であることがAction値不変条件なので、
牌種は1つで足りる。

```text
(相対target, 基礎牌種, called tileの赤牌区分, consumed中の赤牌枚数)
```

`consumed_tiles`は順序なしmultisetであり、赤牌枚数だけがsemantic distinctionを
構成する。相対targetあたりのsizeは、Ponが `31 + 3 × (2 × 3) = 49`、
Daiminkanが `31 + 3 × (2 × 4) = 55` である（31は赤牌になり得ない基礎牌種の数）。

### Ankan

```text
(基礎牌種, 4枚中の赤牌枚数)
```

赤牌になり得る牌種は0..4枚、それ以外は0枚だけを取る（`31 + 3 × 5 = 46`）。

### Kakan

```text
(相対from_seat, 基礎牌種, added_tileの赤牌区分, called_tileの赤牌区分)
```

`added_tile`と`called_tile`は同じ基礎牌種であり、赤牌区分だけが独立する。元Ponの
全構成牌はKakan indexへ持ち込まない。元Ponは
[Action identity](action-identity.md)のとおり、同じContextの`melds`から
`from_seat`と`called_tile`でちょうど1件へ解決する。

赤牌区分は、`added_tile`と`called_tile`について独立した2値として扱う。1 suitあたり
赤5が何枚存在するかはmatch ruleの問題であり、`InternalAction`の値不変条件では
ないため、vocabulary側でruleを仮定して組み合わせを削らない。

### Ron / Tsumo

winning tileはindexへ明示的に含める（相対target × 37、および37）。同じdecisionの
直前打牌やdrawn tileから推測して補完しない。

### Riichi / Pass / KyuushuKyuuhai

いずれもactor以外のsemantic fieldを持たないため、単一indexとする。
`RiichiAction`はリーチ宣言の開始であり、宣言牌は別の`DiscardAction`として
表現される（[内部Actionモデル](internal-action-model.md)）。

## Public API

`lisjong.action_vocabulary` package rootが次を公開する。

| 名前 | 役割 |
| --- | --- |
| `ACTION_VOCABULARY_VERSION` | vocabulary version identity |
| `ACTION_VOCABULARY_SIZE` | fixed-sizeなindex総数 |
| `ACTION_VOCABULARY_BLOCKS` | variant型 → index rangeのread-only mapping |
| `encode_action(action, *, version)` | `InternalAction` → model action index |
| `decode_action(index, actor, *, version)` | index + actor → `InternalAction`値 |
| `encode_legal_actions(decision, *, version)` | index → canonical legal Actionのmapping |
| `build_legal_action_mask(decision, *, version)` | fixed-sizeなlegal mask |
| `resolve_legal_action(index, decision, *, version)` | index → canonical legal Action |

```text
DecisionContext
    -> build_legal_action_mask()
    -> model-selected action index
    -> resolve_legal_action()
    -> canonical legal InternalAction
    -> execute_policy() の既存validation
```

### Encode

`encode_action()`は`InternalAction`のsemantic fieldからindexを導出する。Python
object identity、Python hash値、`legal_actions`の並び順、外部engineのphysical
tile IDは使用しない。`actor`を含めないため、同じsemantic操作はactorによらず同じ
indexになる。

Action値不変条件を満たす11 variantのexact typeは、すべて損失なくencodeできる。
variantのsubclassを含め、encodeできない値はfail closedとし、近いindexやfallback
へ丸めない。

### Decode

`decode_action()`はindexとactorから`InternalAction`の値を再構築する。これは
vocabulary上の意味の復元であり、合法性の主張ではない。

### Legal mask

`build_legal_action_mask()`は長さ`ACTION_VOCABULARY_SIZE`の`tuple[bool, ...]`を
返す。trueなindexの集合は、そのdecisionのencoded legal action集合と完全一致する。
maskは純粋なPython contractとして表現し、NumPy / PyTorch等のML runtimeへ依存
しない。tensorへの変換はconsumer側の責務である。

### Resolve

`resolve_legal_action()`は、model-selected indexを同じdecisionのcanonical legal
Actionへ解決し、`decision.legal_actions`側のobjectそのものを返す。equalな別object
ではない。

`encode_legal_actions()`はindex昇順に反復する新しいmappingを返す。mask、encoded
mapping、resolve結果のいずれも`legal_actions`のpermutationで変化しない。

### 情報境界

codecとmaskは`DecisionContext.legal_actions`と`input.self_seat`だけを読む。自席の
非公開手牌、他家の非公開情報、materialized state、外部engine objectを参照しない。
`PolicyInput` / `DecisionContext`へ新しい情報を追加せず、
[Architecture](architecture.md)のseat-safe information boundaryを変更しない。

## Fail closed

次はいずれも例外とし、任意のfallback Actionへ置換しない。

| 条件 | 例外 |
| --- | --- |
| 未対応のvocabulary version | `UnsupportedActionVocabularyVersionError` |
| 11 variantのexact typeでない値（subclassを含む）、または損失なくencodeできない値 | `ActionEncodingError` |
| vocabulary範囲外のindex | `ActionIndexError` |
| vocabulary上は有効だが当該decisionでlegalでないindex（mask上illegal） | `IllegalActionIndexError` |
| 同一decision内で複数legal actionsが同じindexへ衝突 | `ActionIndexCollisionError` |
| int以外のindex、`Seat`でないactor、`DecisionContext`でないdecision | `TypeError` |

上表のうち、`ActionVocabularyError`を基底に持つのはこのpackageが定義する5つの
例外である。引数の型不正はPython標準の`TypeError`とし、`ActionVocabularyError`
階層へは含めない。いずれも`lisjong.policy_contract`の
`PolicyActionValidationError`とは別であり、`execute_policy()`のsignature、
validation、例外semanticsを変更しない。

`InternalAction` variantのsubclassは`isinstance`上variantに一致するが、
vocabularyはbase variantのsemantic fieldしか表現できず、decodeもbase variantを
返す。subclassをencodeすると`decode(encode(a)) != a`となりsemantic distinctionを
silentに失うため、encodeはexact typeで判定してfail closedとする
（`DecisionContext`自体はsubclassを受理するため、maskとresolveも同じ地点で
拒否する）。

versionの検証は他の検証より先に行う。unsupported versionの下でindex範囲や
legalityの判断を行わない。

vocabularyはactorを固定した値空間上でinjectiveであり、`DecisionContext`は
semantic重複を禁止するため、collisionは正常な構築経路では発生しない。それでも
collision検出はdefensive guardとして残し、衝突時にどちらかのActionを採用しない。

## 検証境界とtest観点

`tests/test_action_vocabulary.py`が少なくとも次を固定する。

- vocabulary version identityと総size
- documented index layoutとblockが`range(0, 802)`を分割すること
- `version -> fingerprint`のliteral固定（同じversion内でindexの意味が動かないこと）と、
  block境界indexの意味のliteral固定
- 全indexの機械的検査（decode → encode round-trip、alias / hole / range重複の不在、
  decode結果が自分のvariant blockへ収まること）
- 11 variantすべてのencode / decode
- 同じdecisionでの`resolve_legal_action(encode_action(a), decision)`が
  `legal_actions`側のcanonical objectを返すこと
- actorがindexへ含まれず、contextから復元されること
- target / from_seatがactor相対であること
- 赤5を含むdiscard、同一牌種の手出し / ツモ切り
- 同一called tileに複数consumed pairがあるChi、Chi / Pon / Daiminkan / Ankan /
  Kakanの赤牌構成、multiset順序非依存
- Ron / Tsumoのwinning tile、Riichi / Pass / KyuushuKyuuhaiの単一index
- maskのtrue index集合とencoded legal action集合の完全一致
- `legal_actions` permutationへの非依存、`PolicyInput`内容への非依存
- out-of-range index、非int index、mask上illegalなindex、unsupported version、
  collision、variant subclassのfail closed
- `execute_policy()`のsignatureとcanonical action返却が変わらないこと

## 引き続き未確定の事項

- model artifact format、weightsとversionの具体的な保存方法
- PolicyInput feature encoderとtensor schema
- HandBelief consumer seam、model architecture、training
- vocabulary version 2以降で扱う可能性のある追加variantや集約
