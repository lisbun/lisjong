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
| 実測 | 実RiichiEnv 0.4.8で実際に確認した情報 |
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

## 設計判断: possible_actions送信前semantic validation

`src/lisjong/riichilab_adapter/possible_action_validation.py`は、送信予定の
canonical `InternalAction`をserver candidateと同じ最小identity空間へ
projectionし、server提示`possible_actions`内の各候補を、Action typeごとに
**公式candidate schemaが実際に持つfieldだけ**へ正規化してから比較する。
`actor` / `target` / `tsumogiri`はBot response専用fieldであり、candidate
側のsemantic identityには含めない。

| Action type (mjai) | candidate semantic identity(このmoduleが照合するfield) | Bot response専用field(candidate側では要求しない) |
| --- | --- | --- |
| `dahai` | tile(`pai`) | actor, tsumogiri |
| `reach` | (type一致のみ) | actor |
| `chi` / `pon` / `daiminkan` | called tile(`pai`), consumed tile multiset(`consumed`) | actor, target |
| `ankan` | tile multiset(`consumed`) | actor |
| `kakan` | added tile(`pai`) | actor |
| `hora`(ron/tsumo共通) | winning tile(`pai`) | actor, target |
| `none` | (type一致のみ) | actor |
| `ryukyoku` | (type一致のみ) | actor |

- 比較はraw dict完全一致ではなく、上記のsemantic identityの一致で行う
- list index、候補の列挙順には依存しない
- candidateへ`actor`/`target`/`tsumogiri`が存在しなくても拒否理由にしない。
  逆に存在しても(server実装が将来これらを付加する場合に備えて)無視する
- 1 request_actionの`possible_actions`は、このAdapterがbindされた1 seat
  分のcandidateだけであるため、actorは常に自明であり識別に不要である。
  hora candidateのtargetについても同様に、1つのrequestが表す和了機会は
  常に一意であるためcandidate側の識別には使わない
- tile文字列は既存`tile_from_mjai()`で正規化し、赤五と通常五、字牌表記の
  違いを保持する
- multiset field(`consumed`、`ankan`の`consumed`)は牌のcanonical順序で
  ソートしてから比較し、入力側の順序差を無視する
- semantic identity上、match件数が0件または複数件の場合はfail closed
  (`PossibleActionsValidationError`)とする
- 個々のcandidateがmalformed、またはtypeが未知の場合、そのcandidate単体は
  「一致しない候補」として扱い(validation全体を中断させない)、選択中の
  Actionがどの候補とも一致しなければ結果的に0件一致として拒否する

Bot response側(`mjai_response.py`)は、この節の変更による影響を受けない。
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

### 未確認事項・既知の前提(#39で要確認)

- `possible_actions`が、同一semantic Actionに対応する複数の重複candidate
  (例: 手牌中の同じ牌が2枚あり、どちらを打牌しても同じ結果になる場合の
  candidateが2件listされる等)を含むかどうかは、今回のレビューでも未確認の
  ままである。現在の実装は複数件一致を無条件にambiguousとしてfail closed
  するため、仮に実サーバーが重複candidateを送る設計であった場合、正当な
  打牌が誤って拒否される可能性がある。この点はIssue #38の判断(「複数一致は
  安全側でfail closed」)を優先し、#39の実サーバー接続で実際の
  `possible_actions`の重複有無を確認したうえで、必要なら再検討する。
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
