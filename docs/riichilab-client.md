# RiichiLab WebSocket Client

この文書は、[Issue #39](https://github.com/lisbun/lisjong/issues/39)で実装した
`src/lisjong/riichilab_client/`の責務境界、RiichiLab公式protocolについて確認した
事実、lisjongでの実測事実、未確認事項、設計判断を分離して記録する。

Policy判断、Observation変換、Action mapping、`possible_actions` semantic
validationの責務境界は[RiichiLab request_action Adapter](riichilab-adapter.md)
(#38)を正本とし、本書ではWebSocket transport lifecycle固有の判断だけを記録する。

## 区分

`docs/riichienv-investigation.md`、`docs/riichilab-adapter.md`と同じ区分を用いる。

| 区分 | 意味 |
| --- | --- |
| 公式情報 | RiichiLab公式文書、または公式文書を引用したIssue本文で確認した情報 |
| 実測 | 実RiichiLab `/ws/validate`への接続で実際に確認した情報 |
| 推測・未確認 | 公式情報と実測のどちらでも確認できていない事項 |
| 設計判断 | 調査結果からlisjongの実装へ引き継ぐ判断 |

## 公式文書へのアクセス制限(実装時点の記録)

Issue #38実装時点と同様、Issue #39実装時点(2026-08-14)でも、本実装を行った
AI実行環境から`https://riichi.dev/docs/protocol`、
`https://riichi.dev/docs/local-testing`、`https://riichi.dev/docs/validation`を
含む`riichi.dev`ドメインへのnetwork egressがproxy policyによりblockされており
(`curl`で`403`)、公式仕様の再取得ができなかった。またこの実行環境には実
`BOT_TOKEN`も注入されておらず、実RiichiLab `/ws/validate`へのlive validationも
実行できなかった。

このため本実装は、次を情報源として進めた。

- Issue #39本文、および[実装前の設計方針コメント](https://github.com/lisbun/lisjong/issues/39#issuecomment-5298990637)
  が引用する公式仕様の要約
- 本repositoryの既存文書(`docs/architecture.md`、`docs/riichilab-adapter.md`)
- Issue #38が確定した`RiichiLabSeatAdapter`公開API(実装は再検証済み、
  仕様は変更していない)

したがって、本書が「公式情報」と記載する項目は、実際にはIssue本文・コメントが
転記した時点の公式仕様であり、`riichi.dev`を直接参照して独自に確認したもの
ではない。live validationを含む実測の再確認は、学習者環境で`BOT_TOKEN`と
network egressが利用可能な状態で行うことが望ましい(下記「live validation」を
参照)。

## 責務境界(実装確定)

`src/lisjong/riichilab_client/`の公開APIは次のとおりである。

- `run_validation(policy: Policy, token: str, *, url: str = DEFAULT_VALIDATION_URL) -> ValidationResult`:
  `wss://game.riichi.dev/ws/validate`(既定値)へBearer token付きで接続し、
  1 validation gameを完走する。`token`は接続確立のためだけに使い、戻り値へは
  含めない
- `ValidationResult`: `passed`(`validation_result.passed`をそのまま採用、
  成功の正本)、`validation_result_received`、`end_game_received`、
  `failure_reason`、`requests_received`、`responses_sent`、
  `ack_history`(`request_id`ごとの`action_ack` statusの履歴)を持つ
  frozen dataclass。tokenやraw Observation、raw `request_action`全文等の
  secretは含まない
- `python -m lisjong.riichilab_client.validation`: 環境変数`BOT_TOKEN`から
  tokenを読み込むCLI entry point。secretはstdout/stderrへ出力しない

内部構造(`ValidationSession` / `Transport`等)は次の「package構成」を参照する。

WebSocket接続、`request_id`のgame内lifecycle管理、`action_ack`対応付け、
`start_game` / `end_game` / `validation_result`処理はこのpackageの責務である。
Policy判断(`build_decision()`、`execute_policy()`)、Observation deserialize、
Action mapping、`possible_actions` semantic validation、MJAI response
serializationは`riichilab_adapter`(#38)を再利用し、この境界へ再実装しない。

## package構成

```text
src/lisjong/riichilab_client/
    __init__.py     公開API re-export
    errors.py       RiichiLabClientError / ProtocolError / TransportError /
                     UnexpectedDisconnectError
    session.py       ValidationSession(pure transport lifecycle state)
    transport.py      Transport protocol、WebSocketTransport、
                       connect_validation_transport()、
                       drive_validation_session()
    validation.py      run_validation()、ValidationResult、CLI entry point
```

- `ValidationSession`(`session.py`)は、WebSocket接続・asyncioから完全に
  独立したpure state machineである。parsed済みJSON event(mapping)を
  `handle_event()`で受け取り、送信すべきpayloadがあればそのdictを返す。
  fake/local transport testは、実接続なしにこのclassだけで
  lifecycle全体を確認できる(下記「テスト方針」を参照)
- `Transport` protocol(`transport.py`)は`recv()` / `send()` / `close()`
  だけを要求する最小限のasync interfaceであり、`WebSocketTransport`が
  `websockets` libraryの実接続をこのprotocolへ適合させる
- `drive_validation_session()`が、`Transport`からの受信・JSON parse・
  binary frame判定・`ValidationSession`への委譲・送信を1つのloopとして
  実装する。`validation_result`を受信するまでloopし続け、受信後は
  呼び出し側(`run_validation()`)が接続を閉じる
- `websockets`への依存はこのpackage内(`transport.py`)だけで使用し、
  `policy_contract` / `policies` / `riichienv_adapter`へは逆流させない
  (設計判断、Issue #39本文セクション38)

## WebSocket library

**設計判断**: 依存として`websockets==17.0.1`を採用した。実装開始時点
(2026-08-14)でPyPIから取得可能な最新の安定版である。公式Local Testing文書
(network egress blockのため本実装からは直接確認できていない)がこの
libraryを例示していたIssue本文の記述を踏襲した。generic HTTP client、
他のtransport frameworkは追加していない。

`websockets.connect(url, additional_headers=headers)`は、17.x系のasyncio
client APIが提供するkeyword引数である(実測: `inspect.signature`で確認)。
`websockets.connect()`はcontext manager使用時に自動reconnectする
iteratorとしても使えるが、本実装では`await websockets.connect(...)`で
1回だけ接続を確立し、`async for`によるreconnect loopは使用しない
(mid-game reconnect非対応の設計判断と整合させるため)。

## Token境界

**設計判断**: `BOT_TOKEN`はruntime secretとして`run_validation(policy, token)`
の明示引数から注入する。secret管理frameworkは導入していない。

- `token`はAuthorization header(`Bearer <token>`)を設定する目的だけに
  使用し、`Transport`実装・`ValidationSession`・`ValidationResult`の
  いずれにも保持しない
- CLI(`python -m lisjong.riichilab_client.validation`)は環境変数
  `BOT_TOKEN`から読み込む。未設定時はsecretを含まないエラーメッセージを
  stderrへ出力し、非zero exit codeを返す
- 例外メッセージ・ログ・test fixtureへtoken文字列を含めない設計とした
  (`test_riichilab_client_validation.py`の`SecretHandlingTest`相当の
  確認を`RunValidationEndToEndTest`内で行っている)

## `start_game` / seat bind

**設計判断**(公式field名は未確認、Issue #39本文セクション12の要求を
具体化): `start_game` eventの`seat` fieldを`int`として読み取り、
`Seat`へ変換して`RiichiLabSeatAdapter(self_seat=Seat.SEAT_0, policy=policy)`
を1回だけ生成する。

- `seat`が欠落・`bool`・`int`以外・`0`-`3`範囲外の場合はfail closed
  (`ProtocolError`)
- validationでは`seat == 0`を要求する。`0`以外はsilent補正せずfail closed
- `start_game`前に`request_action`を受信した場合はfail closed
- duplicate `start_game`は安全側で扱う: 同一seatを再度報告した場合は
  既存`RiichiLabSeatAdapter` runtimeをそのまま維持し、作り直さない。
  異なるseatを報告した場合はfail closed(silent補正しない)

**推測・未確認**: `start_game`の実際のfield名(`seat`という名前自体を
含む)は、network egress blockにより実サーバーで確認できていない。
学習者環境でのlive validationで実測し、異なる場合は本書と
`session.py`の実装を更新する必要がある。

## `request_action` / `request_id` lifecycle

`request_action`受信時の処理順序は次のとおりである。

1. `start_game`未受信ならfail closed
2. `request_id`が`int`(`bool`除外)であることを確認
3. `request_id`が既に受理済み(duplicate)、または直前に受理した値以下
   (decrease)ならfail closed。**`+1`ずつの連番とは仮定しない**
   (`request_id`はgame内でmonotonically increasing integerであるという
   公式契約だけを前提とする、Issue #39最新設計方針)
4. `time`が存在する場合、`grace_ms` / `bank_ms` / `deadline_ms`の型だけを
   検証する(数値であること。値そのものはdeadline enforcementへ使わない)
5. 受理済みrequest_idとして記録したうえで、`RiichiLabSeatAdapter.process_request_action(raw_request_action)`
   (#38)へそのままeventを渡す
6. Adapterが返す`SendReadyResponse.request_id`がcurrent request_idと
   一致することを確認(mismatchならfail closed)
7. send直前にも、current requestへのbindを再確認する(cross-request
   payload再利用の禁止、Issue #39本文セクション15)
8. 同じrequest_idへの二重sendを禁止する
9. `SendReadyResponse.action`にrequest_idを付与したdictを、送信対象
   payloadとして返す

**設計判断(未確認のBot-to-Server response schema)**: `SendReadyResponse`
は`request_id`と`action`(MJAI action dict)を別々に保持するが(#38)、
RiichiLab Bot-to-Server responseの実際のJSON schemaは本実装からは確認
できていない。本実装は、`action`の各fieldをtop-levelへ展開したdictへ
`request_id`を追加した形(`{**action, "request_id": request_id}`)を
送信payloadとして採用した。これは「responseはcurrent request_idを
echoする」という公式契約(Issue本文)を満たす最小の実装判断であり、
実サーバーが別のwrapper形式(例: `{"type": "action", "request_id": ...,
"action": {...}}`)を要求する場合は、live validationでの実測後に
`session.py`の`_handle_request_action()`を更新する必要がある。

## time budget / timeout

**設計判断**(Issue #39最新コメントを採用): client-side deadline
cancellationはMVPでは実装しない。

- `request_action.time`の`grace_ms` / `bank_ms` / `deadline_ms`は
  transport metadataとして型検証のみ行い、値そのものをdeadline
  enforcementへ使わない
- `DecisionContext` / Policyへ`time`情報を一切渡さない
- timeout時にClient独自の`none` / tsumogiri / `possible_actions[0]`等の
  arbitrary fallback Actionを生成しない。server側のtimeout/default
  action semanticsへ任せる
- 現在のPolicy / #38 Adapterは同期APIであり、deadline enforcementを
  導入するとthread/executor/cancellationとstateful Adapterの整合管理が
  必要になるため、#39 MVPでは導入しないという判断はIssue #39最新
  コメントの理由をそのまま採用した

## `action_ack`

**設計判断**(Issue #39最新コメントを採用): `action_ack`を「1 request =
1 ack」と仮定せず、`request_id`ごとのstatus historyとして保持する。

現在の公式定義として扱うstatus:

| status | 扱い |
| --- | --- |
| `accepted` | 正常。historyへ記録するのみ |
| `rejected` | chomboにつながり得る重大statusのため、historyへ記録した後fail closed |
| `unparseable` | 同上 |
| `stale` | non-fatal。historyへ記録するのみ |
| `defaulted` | non-fatal。historyへ記録するのみ |

- `request_id`または`status`が欠落・型不正・未知の場合はfail closed
- 対応する`request_action`をまだ受理していない`request_id`(unknown /
  future)への`action_ack`は成功扱いせずfail closed
- duplicate ack(同じ`request_id`への複数`accepted`等)は、別requestの
  成功として扱わず、単にhistoryへ積む
- `action_ack`はPolicyへ一切渡さない

**推測・未確認**: `rejected` / `unparseable`受信時に即座にvalidation
lifecycle全体をfail closedする(=validation失敗として扱う)という
本実装の判断は、Issue #39最新コメントの「chomboにつながるためfail」という
記述をそのまま採用したものであり、実サーバーが`rejected`後も
`validation_result`まで通信を継続する挙動を取るかどうかは実測できて
いない。live validationで異なる挙動が確認された場合、この判断を
見直す必要がある。

## forward compatibility

- `type`が欠落、または既知5種(`start_game` / `request_action` /
  `action_ack` / `validation_result` / `end_game`)以外の未知event typeは、
  それだけを理由にfail closedしない。standard informational MJAI event
  (`tsumo`等)を含め、Policy stateの正本は#38がdeserializeする
  Observationであるため、Client側でPolicy stateへ二重適用しない
- 既知eventのunknown追加fieldは許容する(例: `start_game`へ`game_id`が
  含まれていても無視する)
- 既知eventの必須field欠落・型不正はfail closedする

## `end_game` / `validation_result`

**設計判断**(Issue #39最新コメントを採用): validation modeでは
`end_game`受信時に即disconnectせず、`validation_result`を待つ。

```text
end_game受信
→ end_game_received = true
→ connectionを即closeしない
→ validation_resultを待つ
→ passed / reasonを結果へ記録
→ close (drive_validation_sessionのloop終了後、run_validation()が接続を閉じる)
```

- `validation_result.passed`が成功の正本である。Client独自条件だけで
  成功を推測しない
- `passed`が`bool`でない場合はfail closed
- `reason`(なければ`message`)を任意のfailure reasonとして保持する。
  型が`str`でない場合はfail closed

**推測・未確認**: `end_game`と`validation_result`の実際の受信順序、
`validation_result`受信後にserverが自発的にconnectionを閉じるかどうかは、
network egress blockにより実測できていない。本実装は
`validation_result`受信後にClient側から`await connection.close()`する
設計とした(`run_validation()`が`async with connect_validation_transport(...)`
のcontext exitで行う)。live validationでの実測後、この節を更新する
ことが望ましい。

## binary frame

**設計判断**(Issue #39最新コメントを採用): binary frameはprotocol
failureとしてClient全体を落とさず無視する。text frameだけJSON parse
対象とする。

## JSON parse

- text frameはJSON top-level objectとしてparseする
- JSON syntax errorはfail closed
- top-level JSONがobject(dict)でない場合はfail closed
- unknown fieldは許容する(forward compatibility)

## fail closed一覧(実装確定)

Issue #39本文セクション25が要求する項目を、本実装ではすべて
`ProtocolError` / `TransportError` / `UnexpectedDisconnectError`
(いずれも`RiichiLabClientError`のsubclass)、または#38 / #34 / #23が
送出する既存例外の伝播として実装した。

- JSON parse failure、known lifecycle event malformed、
  `request_action` before `start_game`、validation seat != 0、
  duplicate/old/decreasing `request_id`、Adapter response request_id
  mismatch、`action_ack`のlifecycle不整合(unknown request_id、
  unknown status、`rejected`/`unparseable`)、`validation_result`の
  malformed `passed` → `ProtocolError`
- JSON serialization failure、WebSocket send failure → `ProtocolError`
  または`TransportError`(送信前serializationは`ProtocolError`、
  送信そのものの失敗は`TransportError`)
- WebSocket receive failure、unexpected disconnect →
  `UnexpectedDisconnectError`
- Adapter error、Policy errorの伝播、mapping/validation errorの伝播 →
  #38 / #34 / #23の例外をそのまま伝播(変更・再wrapしない)

いずれのケースでもarbitrary fallback Actionを生成しない。

## reconnect

mid-game reconnectは実装していない。unexpected disconnectは
`UnexpectedDisconnectError`として明示的な失敗を返し、自動retry loop、
connection pool、旧`ValidationSession`/`RiichiLabSeatAdapter` stateの
再利用は行わない。将来必要になった場合は、RiichiLabの仕様と必要性を
確認し、別Issueで合意したうえで検討する。

## テスト方針(実装確定)

Issue #39最新コメントが提示した4層構成で実装した。

1. **pure lifecycle unit test** (`tests/test_riichilab_client_session.py`):
   `ValidationSession`を、`RiichiLabSeatAdapter`をfake stubへ差し替えた
   状態で直接test する。`start_game` seat bind、`request_id`
   monotonicity(gap許容、duplicate/decrease拒否)、`action_ack`
   status history、forward compatibility、`end_game`/`validation_result`、
   time metadata型検証、fail closedを実WebSocket・実RiichiEnvなしに確認する
2. **fake WebSocket transport test** (`tests/test_riichilab_client_transport.py`):
   `Transport` protocolのfake実装で`drive_validation_session()`を駆動し、
   JSON text送受信、binary frame無視、JSON parse failure、送信・受信
   failureの例外変換、serialization failureを確認する
3. **#38 + `MinimalPolicy` integration** (`tests/test_riichilab_client_validation.py`):
   実RiichiEnv 0.4.8の`Observation`を使い、`ValidationSession`が
   `RiichiLabSeatAdapter` + `MinimalPolicy`を経て送信前validation済み
   MJAI responseまで届くこと、Policyへ`request_id`/`time`/`ack`/
   WebSocket固有情報が漏れないことを確認する。あわせて
   `run_validation()`をfake transportで駆動するend-to-end testで、
   `ValidationResult`の内容とtoken非露出を確認する
4. **manual live validation**: 下記「live validation」を参照

## live validation

**未実施**。理由は次の2点である。

- 本実装を行ったAI実行環境には実`BOT_TOKEN`が注入されておらず、有効な
  botとして接続できない
- `riichi.dev`ドメインへのnetwork egressがproxy policyによりblockされて
  おり(`curl`で`403`)、`wss://game.riichi.dev/ws/validate`への接続
  そのものを試行できない

学習者環境からlive validationを実行する場合、次のコマンドを使用する。

```powershell
$env:BOT_TOKEN = "<実RiichiLab bot token>"
python -m lisjong.riichilab_client.validation
```

成功時は`RiichiLab validation passed`と、request/response件数、
`end_game`受信有無を標準出力へ表示する(token、Authorization header、
raw Observationは表示しない)。失敗時は非zero exit codeを返し、
`RiichiLabClientError`のtypeとmessageをstderrへ出力する(いずれも
tokenを含まない)。

live validation実行後、確認できた実際の`start_game` field名、
Bot-to-Server response schema、`end_game`/`validation_result`の受信
順序、`action_ack` `rejected`/`unparseable`後のserver側継続有無等を
本書へ反映することが望ましい。
