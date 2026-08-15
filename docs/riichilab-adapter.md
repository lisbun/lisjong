# RiichiLab request_action Adapter

この文書は、[Issue #38](https://github.com/lisbun/lisjong/issues/38) で実装した
`src/lisjong/riichilab_adapter/` の責務境界、RiichiLab公式protocolについて確認した事実、
lisjongでの実測事実、未確認事項、設計判断を分離して記録する。

対象範囲、既存境界(#23 `build_decision()`、#34 `execute_policy()`)の再利用方針、
WebSocket・token・`request_id` lifecycle・timeout schedulerを対象外とすることは
[Issue #38本文](https://github.com/lisbun/lisjong/issues/38)を正本とする。本書は
実装で確定した具体的なnormalization規則と、公式仕様との既知の差異を記録する。

## 区分

`docs/riichienv-investigation.md`と同じ区分を用いる。

| 区分 | 意味 |
| --- | --- |
| 公式情報 | RiichiLab公式文書、または公式文書を引用したIssue本文で確認した情報 |
| 実測 | 実RiichiEnv 0.4.8、または実RiichiLab `/ws/validate`で確認した情報 |
| 推測・未確認 | 公式情報と実測のどちらでも確認できていない事項 |
| 設計判断 | 調査結果からlisjongの実装へ引き継ぐ判断 |

## 公式文書へのアクセス制限(実装時点の記録)

Issue #38実装時点(2026-08-14)、本実装を行ったAI実行環境からは
`https://riichi.dev/docs/protocol`、`https://riichi.dev/docs/local-testing`、
`https://riichi.dev/docs/validation`を含む`riichi.dev`ドメインへのnetwork
egressそのものがproxy policyにより完全にblockされており、実装時点での
公式仕様の再取得ができなかった。

このため本実装は、次を情報源として進めた。

- Issue #38 / #39本文が引用する、起票時点(2026-08-15)の公式仕様の要約
- 本repositoryの既存文書(`docs/architecture.md`、`docs/riichienv-investigation.md`)
- 実RiichiEnv 0.4.8 SDKに対する新規実測(下記)

したがって、本書が「公式情報」と記載する項目は、実際にはIssue本文が転記した
時点の公式仕様であり、`riichi.dev`を直接参照して独自に確認したものではない。
将来この制限が解消された時点、または#39実装時点で、本書と実際の公式文書との
差異を再確認することが望ましい。

## RiichiEnv 0.4.8実測: `Action.to_mjai()`の出力

`riichienv.Action.to_mjai()`を、実RiichiEnv 0.4.8（`RiichiEnv(seed=..., game_mode="4p-red-half")`）の
複数seed・複数kyokuにわたる`legal_actions()`から得たActionへ実際に呼び出し、
出力JSONの構造を確認した(実測)。

| Action type | `to_mjai()`が含むfield | `to_mjai()`が含まないfield |
| --- | --- | --- |
| `dahai` | `type`, `actor`, `pai` | `tsumogiri` |
| `chi` / `pon` / `daiminkan` | `type`, `actor`, `pai`, `consumed` | `target` |
| `ankan` | `type`, `actor`, `consumed` | - |
| `kakan` | `type`, `actor`, `pai`, `consumed`(元Ponの構成牌) | - |
| `reach` | `type`, `actor` | - |
| `hora`(ron/tsumo共通) | `type`, `actor` | `pai`, `target` |
| `none`(Pass) | `type`, `actor` | - |
| `ryukyoku`(九種九牌) | `type`, `actor` | `pai`(constructorのtile省略時、`0`つまり`"1m"`相当が紛れ込み得るため、この経路では明示的に無視する) |

`hora`が`pai`(和了牌)を含まない点、call系(chi/pon/daiminkan)とron相当が
`target`(呼ばれた/放銃元のseat)を含まない点は、Issue本文だけからは確認できず、
本実装で新たに実測した事実である。

## 設計判断: MJAI response構築における必要最小限のnormalization

Issue #38本文が要求する「`Action.to_mjai()`を優先し、差異がある場合だけ必要
最小限の正規化を行う」方針を、上記実測に基づき次のように具体化した
(`src/lisjong/riichilab_adapter/mjai_response.py`)。

- `to_mjai()`の出力をbaseとして使用する(全variantの独自再実装をしない)
- `actor`は、resolve済みcanonical `InternalAction.actor`から明示的に上書きする
- `dahai`には、`InternalAction.tsumogiri`から`tsumogiri`を追加する
- `chi` / `pon` / `daiminkan`には、`InternalAction.target`から`target`を追加する
- `ron`(`hora`)には、`InternalAction.target`から`target`を、
  `InternalAction.winning_tile`から`pai`を追加する
- `tsumo`(`hora`)には、`target = actor`を、`InternalAction.winning_tile`から
  `pai`を追加する
- `ankan` / `kakan` / `reach` / `none` / `ryukyoku`は、`to_mjai()`の出力を
  そのまま使用する(`actor`の上書きを除く)

tile文字列の生成には、既存`tile_from_mjai()`の逆変換として新設した
`lisjong.riichienv_adapter.tile_conversion.tile_to_mjai()`を使用する。

## レビューで判明した事実: possible_actions candidate schemaとBot response schemaの分離

**この節の内容は、Claude Codeが`riichi.dev`へ直接アクセスして確認したもの
ではない。** 学習者(レビュー担当)がIssue #38 ([comment-5298618558](https://github.com/lisbun/lisjong/issues/38#issuecomment-5298618558))で
RiichiLab公式Protocolを確認し、blocking findingとして報告した事実に基づく
修正である。Claude Code自身は引き続き`riichi.dev`ドメインへのnetwork
egressが実行環境から到達できないため、この修正はレビュー指摘への対応として
行った。

初回実装(Issue #38最初のhandoff)は、server提示`possible_actions`
candidateを、Bot-to-Server response(送信するAction)と同じschemaで
parseしていた。具体的には、candidate全件に`actor`を要求し、`dahai`へ
`tsumogiri`、`chi`/`pon`/`daiminkan`へ`target`を要求していた。しかし
RiichiLab公式`possible_actions`は、Bot responseより意図的に小さい最小
candidate表現であり、例えば公式形の`{"type": "dahai", "pai": "1m"}`は
`actor`も`tsumogiri`も持たない。そのため初回実装は、この公式形candidateを
malformedとして扱い、意味上は合法な選択を0件一致でfail closedし得る
状態だった。

この修正で、**candidate側のsemantic identity**と**Bot response
serialization**を明確に分離した。

続く再レビュー([comment-5298736327](https://github.com/lisbun/lisjong/issues/38#issuecomment-5298736327))
では、同じくレビュー担当が公式Protocolを確認したうえで、さらに次の2点が
blocking findingとして報告された。本書の該当記述はその指摘に基づき更新して
いる(この2点もClaude Code自身が`riichi.dev`で直接確認したものではない)。

- 公式`possible_actions` field表では`kakan` candidateが`pai`に加えて
  `consumed`を持つため、`pai`だけをidentityにすると元Pon構成が異なる
  candidateを誤って受理し得る
- 公式Protocolの`possible_actions`の具体例とAction別field表の間には
  一部記述差があり、`actor` / `target`が候補側に一切現れないとまでは
  断言できない

さらに続く第3回レビュー([comment-5298855018](https://github.com/lisbun/lisjong/issues/38#issuecomment-5298855018))
で、`hora`について次のblocking findingが報告された(この事実も
Claude Code自身が確認したものではない)。

- 公式`request_action`例には、`possible_actions`の`hora` candidateとして
  `{"type": "hora"}`というminimal形が掲載されている。一方、同じ公式
  文書のAction別field表には`hora`へ追加fieldが記載されており、
  公式文書内で例とfield表が食い違っている
- 修正前の実装は`hora`を`dahai`と同様の「`pai`必須」type(下記表の
  `_PAI_ONLY_TYPES`相当)として扱っていたため、この公式minimal例
  そのものをmalformedとして拒否していた

## 設計判断: possible_actions送信前semantic validation

`src/lisjong/riichilab_adapter/possible_action_validation.py`は、**これから
serverへ送ろうとしているBot-to-Server response**と、**server提示
`possible_actions`の各candidate**の両方を、同一のcandidate semantic
identityへprojectionしてから比較する。

```text
send-ready Bot response --projection--> candidate semantic identity
server candidate        --projection--> candidate semantic identity
                                        -> semantic equality
```

照合対象をcanonical `InternalAction`ではなく実際の送信内容にしているのは、
`KakanAction`のようにInternalAction側が保持しない外部semantic情報(元Pon
の`consumed`)を落とさずに検証するためである(Issue #38
[再レビュー](https://github.com/lisbun/lisjong/issues/38#issuecomment-5298736327)
のblocking finding)。InternalActionモデル自体はこの都合で変更していない。

| Action type (mjai) | candidate必須identity(照合に使うfield) | candidateに存在する場合だけ整合確認するfield |
| --- | --- | --- |
| `dahai` | tile(`pai`) | actor, tsumogiri(要求しない) |
| `reach` | (type一致のみ) | actor |
| `chi` / `pon` / `daiminkan` | called tile(`pai`), consumed tile multiset(`consumed`) | actor, target |
| `ankan` | tile multiset(`consumed`、4枚) | actor |
| `kakan` | added tile(`pai`), 元Ponのtile multiset(`consumed`、3枚) | actor |
| `hora`(ron/tsumo共通) | (type一致のみ) | pai(和了牌), actor, target |
| `none` | (type一致のみ) | actor |
| `ryukyoku` | (type一致のみ) | actor |

`hora`だけは、公式`request_action`例が示す`{"type": "hora"}`という
minimal candidateを拒否しないために、`pai`をcandidate必須identityに
含めていない(下記「`hora`のminimal candidate対応」を参照)。

- 比較はraw dict完全一致ではなく、上記のsemantic identityの一致で行う
- list index、候補の列挙順には依存しない
- candidateへ`actor`/`target`/`tsumogiri`が存在しなくても拒否理由にしない
  (公式のminimal candidate形をそのまま受理する)
- tile文字列は既存`tile_from_mjai()`で正規化し、赤五と通常五、字牌表記の
  違いを保持する
- multiset field(`consumed`)は牌のcanonical順序でソートしてから比較し、
  入力側の順序差を無視する。枚数は`chi`/`pon` 2枚、`daiminkan`/`kakan`
  3枚、`ankan` 4枚を要求する
- semantic identity上、match件数が0件ならfail closed
  (`PossibleActionsValidationError`)、1件以上ならacceptする

### Issue #39 live実測: duplicate candidate

**公式情報**: RiichiLab Protocolは、botがserver提示`possible_actions`の
いずれかに対応する合法responseを返すことを要求する。一方、candidate list内で
semantic matchが一意になることまでは要求していない。

**lisjong live実測**: Issue #39で、Windows / Python 3.14 / 実BOT_TOKENの
環境から実RiichiLab `/ws/validate`へ接続した。WebSocket connection、
Authorization、`start_game.id`、`request_action`受信、Observation deserialize、
Policy実行までは成功したが、selected responseへ同じsemantic identityで一致する
candidateが2件存在した。旧実装はこれを`found 2 ambiguous matches`として拒否し、
送信前validationで停止した。

**lisjong設計判断**: live実測に合わせ、semantic match 0件はreject、1件以上は
acceptする。`possible_actions`全candidateをprojectする処理は維持し、malformed
またはunknown Action typeが1件でもあればvalidation全体をfail closedする。
candidate order、candidate object identity、list indexはAction identityに使わず、
send-ready responseをserver candidateへ置換しない。送信payloadの正本は引き続き
Policyからresolveしたcanonical Actionを変換したresponseである。

### malformed / unknown candidateはfail closed

forward compatibilityとして許容するのは**既知Action typeのunknown追加
field**までであり、legal candidateそのもののunknown Action typeや
required field欠落までsilent ignoreはしない(Issue #38 再レビュー
blocking finding)。

- 許容する: 既知typeのcandidateに`display_name`等の未知fieldが増えている
- fail closedする: candidateがmappingでない / `type`欠落 / 未知Action type /
  既知typeだがrequired field欠落・型不正・tile parse不能・`consumed`不正

これらが`possible_actions`内に1件でも存在する場合、他に一致candidateが
あるかどうかにかかわらずvalidation全体をfail closedする。個々のcandidateを
skipして残りだけで成功させない。

送信予定response側をcandidate identityへprojectionできない場合も、同様に
payloadを返さずfail closedする。

### candidateが任意で持つsemantic fieldとの整合

公式Protocolは`possible_actions`の具体例(minimal形)とAction別field表の
間に一部記述差があり、candidateが`actor` / `target`を持ち得ないとまでは
断言できない(Issue #38 再レビュー)。そのため、

- candidateにこれらのfieldが**無ければ**identityだけで判定する
- candidateにこれらのfieldが**あれば**、送信予定responseの同名fieldと
  矛盾しないことも確認し、矛盾する場合はそのcandidateを非一致として扱う
  (結果として一致0件になればfail closedする。誤受理側には倒れない)

`tsumogiri`はこの整合確認の対象に含めない。打牌はcandidate identityの
`pai`で一意に定まり、公式のminimal candidate例も`tsumogiri`を持たないため、
仮にcandidate側へ付随していても識別材料にも矛盾判定材料にもしない。

### `hora`のminimal candidate対応(Issue #38 第3回レビュー)

公式`request_action`例が示す`hora` candidateのminimal形
`{"type": "hora"}`を拒否しないため、`hora`の必須identityは`type`のみと
する。`pai`(和了牌)は、`actor` / `target`と同様に**candidate側に存在
する場合だけ**送信予定responseと矛盾しないことを確認する
(`_optional_tile_consistency_agrees`に相当する処理)。

- `{"type": "hora"}` → 常にidentityが一致すれば受理する(`pai`での絞り込み
  なし)
- `{"type": "hora", "pai": "5m"}` → responseの`pai`と一致する場合だけ受理、
  不一致なら非一致として扱う(結果として一致0件ならfail closed)
- `{"type": "hora", "pai": "99z"}`のように`pai`が存在するのに牌として
  parseできない場合は、無視して非一致にするのではなく、candidate
  malformedとしてvalidation全体をfail closedする(`actor` / `target`の
  型不正時のsilent非一致とは扱いが異なる。`pai`は和了牌を区別する唯一の
  optional fieldであり、parse不能を無視すると意味的に別のhora候補を
  誤って受理し得るため)

`ron`と`tsumo`はどちらもmjai `hora`へ変換されるため、同じcandidate
schemaを共有する。
`actor` / `target` / `tsumogiri`は引き続きBot responseへ必要に応じて
付与する(「設計判断: MJAI response構築における必要最小限のnormalization」を
参照)。`possible_actions` validationとBot response serializationは別
責務であり、一方のschema変更が他方の出力に影響しない。

### `tsumogiri`の最終的な扱い

初回実装では、RiichiLab実サーバーの`possible_actions`が`tsumogiri` field
を含むかどうかを未確認としていたが、レビューで公式Protocolを確認できた
結果、**公式`possible_actions`の`dahai` candidateは`tsumogiri`を含まない**
ことが判明した。そのため、`possible_actions` validationの`dahai` semantic
identityから`tsumogiri`を除外した。selected側の`InternalAction.tsumogiri`
は、引き続きBot response serialization(`mjai_response.py`)でのみ使用する。

### 残る未確認事項・既知の前提

- 公式`possible_actions`の具体例とAction別field表の記述差(`reach` /
  `hora`等でfield表の方が多い)については、実サーバーが実際にどこまでの
  fieldをcandidateへ付けるかが未確認である。現在の実装は「無ければ
  identityだけで判定、あれば矛盾だけ確認」という両対応にしてあるが
  (`hora`は`pai`まで、その他は`actor` / `target`まで)、実データでの確認は
  後続live validationで行う。特に`target`を相対seat等で表現するserver実装
  だった場合、現在の整合確認は誤ってfail closed側へ倒れるため、実測してから
  必要なら見直す
- honor牌の文字列表記(`E`/`S`/`W`/`N`/`P`/`F`/`C`)は、既存`tile_conversion.py`が
  RiichiEnv 0.4.8のevent JSONに対して実測した表記であり、RiichiLab
  server側の`possible_actions`が同じ表記を使うことは未確認である
  (公式一般的なmjai表記は`1z`-`7z`の数値表記を使う場合もある)。差異が
  あれば、該当candidateはmalformedとして扱われ、fail closedする側に働く。

## 設計判断: request_action入力境界

`src/lisjong/riichilab_adapter/request_action.py`は、Issue #38本文が示す
最低限の必須fieldだけを検証する。

- `type == "request_action"`
- `request_id`: `int`(`bool`は明示的に除外)。RiichiLab Protocol v2の
  `request_id`はgame内で一意なmonotonically increasing integerであるため
  (Issue #38 review: blocking 2)、初回実装が許容していた`str`は受理しない。
  monotonicity検証、previous requestとの比較、stale/duplicate判定、
  `action_ack`との対応付けはこの境界の責務ではなく、#39で扱う。この境界が
  行うのは、現在の`request_id`が仕様どおりの`int`であることの確認と、その
  値をresponseへechoすることまでである
- `possible_actions`: list-likeなcollection(内容の検証はvalidation側で行う)
- `observation`: base64文字列。`riichienv.Observation.deserialize_from_base64()`で
  復元できない場合はfail closed
- `time`は存在すれば`ParsedRequestAction.time`として保持するが、
  `DecisionContext`やPolicyへは一切渡さない
- 上記以外の未知fieldは、それだけを理由に拒否しない(forward compatibility)

## 責務境界(実装確定)

`src/lisjong/riichilab_adapter/`の公開APIは次のとおりである。

- `RiichiLabSeatAdapter(self_seat, policy)`: 1 game x 1 seatへbindされる
  stateful runtime。`SeatMaterializedState`と`RiichiEnvActionMappingSession`を
  constructorで1回だけ生成し、`process_request_action()`呼び出しをまたいで
  継続保持する
- `RiichiLabSeatAdapter.process_request_action(raw_request_action) -> SendReadyResponse`:
  1件の`request_action`を、送信前validation済みのpayloadまで処理する
- `SendReadyResponse(request_id, action)`: 後続#39がそのままJSON化して
  送信できるpayload

内部処理順は`build_decision()`(#23) → `execute_policy()`(#34) →
`mapping.resolve()` → MJAI response変換 → `possible_actions`検証 →
`request_id`bindである。WebSocket接続、token、`start_game` /
`action_ack` / `validation_result` / `end_game`、`request_id`のgame内
lifecycle管理、timeout schedulerはこのpackageの責務ではなく、#39で実装する。

`docs/architecture.md`の「RiichiLab Client」節が定める情報境界
(Policyへ渡してよいのは`DecisionContext`だけ)は、このpackageでも維持する。

## Issue #39実装後の補足

WebSocket接続、`start_game` / `action_ack` / `validation_result` /
`end_game`、`request_id`のgame内lifecycle管理はIssue #39で
`src/lisjong/riichilab_client/`として実装済みである。詳細は
[RiichiLab WebSocket Client](riichilab-client.md)を参照する。

Issue #39実装時点でも、本書冒頭に記載した`riichi.dev`ドメインへの
network egress blockは、実装を行ったAI実行環境では解消していない
(2026-08-14確認)。一方、2026-08-15に学習者のWindows環境からlive
validationを実行し、`possible_actions`のduplicate semantic candidateが
実serverから提示され得ることは確認済みとなった。`actor`/`target`の実際の
表現、honor牌表記の実サーバー一致等は引き続き未確認である。
