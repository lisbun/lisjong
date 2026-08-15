# RiichiLab WebSocket Client

この文書は、[Issue #39](https://github.com/lisbun/lisjong/issues/39)と
[Issue #42](https://github.com/lisbun/lisjong/issues/42)で実装した
`src/lisjong/riichilab_client/`の責務境界、RiichiLab公式protocolについて
確認した事実、lisjongでの実測事実、未確認事項、設計判断を分離して記録する。

Policy判断、Observation変換、Action mapping、`possible_actions` semantic
validationの責務境界は[RiichiLab request_action Adapter](riichilab-adapter.md)
(#38)を正本とし、本書ではWebSocket transport lifecycle固有の判断だけを記録する。

## 区分

`docs/riichienv-investigation.md`、`docs/riichilab-adapter.md`と同じ区分を用いる。

| 区分 | 意味 |
| --- | --- |
| 公式情報 | RiichiLab公式文書、または公式文書を引用したIssue本文で確認した情報 |
| 実測 | 実RiichiLabへの接続、または実RiichiEnvを使ったtestで確認した情報 |
| 推測・未確認 | 公式情報と実測のどちらでも確認できていない事項 |
| 設計判断 | 調査結果からlisjongの実装へ引き継ぐ判断 |

## 公式仕様の再確認(2026-08-15)

Issue #42実装前に、次のRiichiLab公式文書を直接再確認した。

- [MJAI Protocol](https://riichi.dev/docs/protocol): `/ws/ranked`と
  `/ws/validate`はいずれもBearer `BOT_TOKEN`で接続する。serverがgame loopを
  駆動し、botは`request_action`へのresponseだけを送る。`start_game.id`は
  0..3、`request_id`はgame内で単調増加するinteger、binary frameと未知event /
  fieldはignoreする。文書上の`end_game`例はfinal `scores`を含み、受信後は
  disconnectすると記載されている
- [Ranked Matches](https://riichi.dev/docs/ranked): active botで
  `wss://game.riichi.dev/ws/ranked`へ接続するとmatchmaking queueへ入り、4 botで
  full hanchanを行う。終了後にratingがfinal placementから更新され、serverが
  `end_game`を送る
- [Matchmaking](https://riichi.dev/docs/matchmaking): ranked endpointへの接続が
  queue参加であり、botはmatchedまたはdisconnectまでqueueに残る。Client側の
  join payload、polling、matchmaking algorithmは不要である
- [Rating System](https://riichi.dev/docs/rating): ratingはOpenSkillで管理される。
  Clientの`end_game` payloadでratingが通知されるという保証はないため、
  `RankedGameResult`へratingを含めない

Issue #38/#39実装時は実装環境のnetwork制約により公式文書を直接取得できなかった
が、その履歴は当時の制約として扱い、Issue #42では上記の最新公式文書を正本と
して設計を再照合した。実`BOT_TOKEN`は本実装環境へ注入していないため、live
ranked smoke testは学習者環境で行う。

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
- `run_ranked_game(policy: Policy, token: str, *, url: str = DEFAULT_RANKED_URL) -> RankedGameResult`:
  `wss://game.riichi.dev/ws/ranked`へ1回だけ接続し、queue待ちから1 full
  hanchanの`end_game`まで処理して終了する
- `RankedGameResult`: `end_game_received`、自seat、request/response件数、
  `ack_history`、optionalなfinal `scores`を持つfrozen dataclass。実serverの
  `end_game`にscoresがない場合は`None`とし、rank / placement / ratingや
  score値を推測・補完しない
- `python -m lisjong.riichilab_client.ranked`: 検証用botのtokenを環境変数
  `BOT_TOKEN`から読み込み、`MinimalPolicy`で1半荘だけ実行するCLI entry point

validation/ranked双方のresult/runnerは各実行moduleから直接importできるほか、
`lisjong.riichilab_client` package rootからlazy exportする。package import時に
`validation` / `ranked` moduleをeager importせず、`python -m`実行時のrunpy
二重import warningを発生させない。

内部構造(`ValidationSession` / `Transport`等)は次の「package構成」を参照する。

WebSocket接続、`request_id`のgame内lifecycle管理、`action_ack`対応付け、
`start_game` / `end_game` / `validation_result`処理はこのpackageの責務である。
Policy判断(`build_decision()`、`execute_policy()`)、Observation deserialize、
Action mapping、`possible_actions` semantic validation、MJAI response
serializationは`riichilab_adapter`(#38)を再利用し、この境界へ再実装しない。

## package構成

```text
src/lisjong/riichilab_client/
    __init__.py     公開API lazy re-export
    errors.py       RiichiLabClientError / ProtocolError / TransportError /
                     UnexpectedDisconnectError
    session.py       validation/ranked共通lifecycle、ValidationSession、
                     RankedSession
    transport.py     Transport protocol、WebSocketTransport、共通connect/driver、
                     validation/ranked互換wrapper
    validation.py      run_validation()、ValidationResult、CLI entry point
    ranked.py          run_ranked_game()、RankedGameResult、CLI entry point
```

- 非公開`_GameSession`へseat bind、request_id lifecycle、`action_ack`
  history、#38 Adapter呼び出し、`end_game` flagを1回だけ実装する。
  `ValidationSession`はseat 0限定と`validation_result` terminal、
  `RankedSession`はseat 0..3と`end_game` terminalだけを差分として持つ
- `Transport` protocol(`transport.py`)は`recv()` / `send()` / `close()`
  だけを要求する最小限のasync interfaceであり、`WebSocketTransport`が
  `websockets` libraryの実接続をこのprotocolへ適合させる
- `connect_transport()` / `drive_session()`が接続・JSON parse・binary判定・
  sessionへの委譲・response送信を共通実装する。既存validation wrapper APIは
  維持し、ranked wrapperも同じ共通処理を利用する。接続直後にsendする処理は
  なく、rankedでも`request_action`を受信した場合だけresponseを送る
- `websockets`への依存はこのpackage内(`transport.py`)だけで使用し、
  `policy_contract` / `policies` / `riichienv_adapter`へは逆流させない
  (設計判断、Issue #39本文セクション38)

## WebSocket library

**設計判断**: 依存として`websockets==17.0.1`を採用した。実装開始時点
(2026-08-14)でPyPIから取得可能な最新の安定版である。RiichiLab公式
Local Testing文書がこのlibraryを例示している。generic HTTP client、
他のtransport frameworkは追加していない。

`websockets.connect(url, additional_headers=headers)`は、17.x系のasyncio
client APIが提供するkeyword引数である(実測: `inspect.signature`で確認)。
`websockets.connect()`はcontext manager使用時に自動reconnectする
iteratorとしても使えるが、本実装では`await websockets.connect(...)`で
1回だけ接続を確立し、`async for`によるreconnect loopは使用しない
(mid-game reconnect非対応の設計判断と整合させるため)。

## Token境界

**設計判断**: `BOT_TOKEN`はruntime secretとして`run_validation()` /
`run_ranked_game()`の明示引数から注入する。secret管理frameworkは導入していない。

- `token`はAuthorization header(`Bearer <token>`)を設定する目的だけに
  使用し、`Transport`、各session、各resultのいずれにも保持しない
- validation/ranked CLIは環境変数`BOT_TOKEN`から読み込む。未設定時はsecretを
  含まないエラーメッセージをstderrへ出力し、非zero exit codeを返す
- 例外メッセージ・ログ・test fixtureへtoken文字列を含めない設計とした
  (`test_riichilab_client_validation.py`の`SecretHandlingTest`相当の
  確認を`RunValidationEndToEndTest`内で行っている)

## `start_game` / seat bind

**公式情報**: RiichiLab公式Protocolの
`start_game` eventは、bot seat indexを`seat`ではなく**`id`** fieldで
表す。公式例は次の形である。

```json
{"type": "start_game", "id": 0}
```

**設計判断**: `start_game.id`を`int`として読み取り`Seat`へ変換し、通知された
self seatへ`RiichiLabSeatAdapter`を1 gameにつき1回だけbindする。

- `id`が欠落・`bool`・`int`以外・`0`-`3`範囲外の場合はfail closed
  (`ProtocolError`)
- validationでは`id == 0`を要求する。`0`以外はsilent補正せずfail closed
- rankedでは`id == 0..3`をすべて正常として受理し、そのseatを結果へ記録する
- `start_game`前に`request_action`を受信した場合はfail closed
- duplicate `start_game`は安全側で扱う: 同一`id`を再度報告した場合は
  既存`RiichiLabSeatAdapter` runtimeをそのまま維持し、作り直さない。
  異なる`id`を報告した場合はfail closed(silent補正しない)
- `seat` fieldをfallbackとして併用しない。`{"type": "start_game", "seat": 0}`
  のように`id`を伴わないeventは、`id`欠落としてfail closedする
  (初回実装の誤り、`tests/test_riichilab_client_session.py`の
  `test_legacy_seat_field_alone_is_not_treated_as_id`で回帰防止済み)。
  `id`と`seat`が両方存在する場合、正本は`id`であり`seat`はunknown extra
  fieldとしてforward-compatibleに無視する

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

**公式情報・実測**: Bot-to-Server responseはMJAI actionのtop-levelへcurrent
`request_id`をechoする。Issue #39の実`/ws/validate`ではこの形で108 responsesを
送信し、validationを完走した。Issue #42のranked pathも同じ#38 Adapterと共通
session処理を再利用し、別schemaやwrapperを導入しない。

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

validationとrankedは、同じgame lifecycleを利用するがterminal条件が異なる。

**validation設計判断**: `end_game`受信時に即disconnectせず、
`validation_result`を待つ。

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

**ranked公式情報**: Protocol文書では`end_game`が1 full hanchanのterminal
eventであり、4 seatのfinal `scores`を含む例が掲載され、受信後はdisconnectする
と記載されている。

**ranked実測(2026-08-15)**: 検証用botで実RiichiLab `/ws/ranked`を2回実行した。
1回目はscores必須validationで失敗し、secret-safeなshape診断を追加して再実行した
結果、受信したranked `end_game`のtop-level keyは`type`だけで、`scores` fieldは
存在しなかった(`event_keys=['type']; scores_type=NoneType; scores_length=None`)。
公式文書の例と実ranked serverのeventには、この点で差がある。

**ranked設計判断**: `start_game`後の有効な`end_game`受信そのものを正常終了条件
とし、scoresを必須条件にしない。`validation_result`や次gameのeventは待たない。

- scoresが存在しない場合、`SessionStatus.scores` / `RankedGameResult.scores`は
  `None`とする。0や別fieldからの推測値で補完しない
- scoresが存在する場合は、4個の`int`(`bool`除外)だけをtupleとして保持する。
  fieldが存在するのに型・要素数が不正な場合は、値をdumpせずkey一覧・型・
  list長だけを示してfail closedする
- rank / final placement / ratingは必須化・推測しない
- rankedの`end_game`前にconnectionが切れた場合、server側ではdefault actionで
  gameが継続しても、lisjongのsmoke testは`UnexpectedDisconnectError`で失敗する
- `end_game`後はcontext exitでconnectionを閉じ、自動再queueしない

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
  `request_action` before `start_game`、validation seat != 0、ranked seat範囲外、
  duplicate/old/decreasing `request_id`、Adapter response request_id
  mismatch、`action_ack`のlifecycle不整合(unknown request_id、
  unknown status、`rejected`/`unparseable`)、`validation_result`の
  malformed `passed`、ranked `end_game`に存在する不正scores → `ProtocolError`
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
再利用は行わない。rankedではqueue retry、auto requeue、`end_game`後の
next game loopも行わず、1 connection / 1 hanchanで終了する。
将来必要になった場合は、RiichiLabの仕様と必要性を
確認し、別Issueで合意したうえで検討する。

## テスト方針(実装確定)

Issue #39の4層構成を維持し、Issue #42のranked差分を同じ境界で追加した。

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
   `ValidationResult`の内容とtoken非露出を確認する。CLIをtoken未設定で
   subprocess実行し、`python -m`でrunpy `RuntimeWarning`が発生しないことも
   回帰testで固定する
4. **ranked unit / fake transport / integration**
   (`tests/test_riichilab_client_ranked.py`): seat 0..3、validation seat 0回帰、
   scoresなし`end_game` terminal、optionalなvalid scores、不正scores、
   no join payload、exactly one game、
   binary/unknown event、unexpected disconnect、#38 + `MinimalPolicy` integration、
   secret-safe result、ranked CLIのRuntimeWarning非再発を確認する
5. **manual live validation / ranked smoke test**: 下記を参照

## live validation

2026-08-15に学習者のWindows / Python 3.14環境から実BOT_TOKENをruntime
注入し、duplicate semantic candidate対応後の実`/ws/validate`を再実行した。

```text
RiichiLab validation passed
requests: 108
responses: 108
end_game: yes
```

これにより、WebSocket connection / Authorization、`start_game.id`、108件の
`request_action`、Observation deserialize、#23 `build_decision()`、#34
`execute_policy()`、`MinimalPolicy`、mapping resolve、MJAI response送信、
`end_game`、`validation_result.passed`まで実serverで完走済みとなった。

再実行する場合は次のコマンドを使用する。

```powershell
$env:BOT_TOKEN = "<実RiichiLab bot token>"
python -m lisjong.riichilab_client.validation
```

## live ranked smoke test

### 公式情報

公式文書ではranked接続のterminal eventを`end_game`としている。実装当初は
同eventにfinal scoresが含まれる前提としていた。

### 実測

2026-08-15、検証用botで実RiichiLab `/ws/ranked`を実行した。最初の2回は
matchmakingから`end_game`受信までは到達したが、Clientがscoresを必須としたため
`ProtocolError`で終了した。2回目のsecret-safe診断により、実serverの
`end_game`はtop-level keyが`type`だけの`{"type":"end_game"}`相当で、scoresを
含まないことを確認した。

この実測を反映した修正後、同日の3回目のlive smoke testで1半荘を完走した。

```text
RiichiLab ranked game completed
seat: 3
requests: 85
responses: 85
end_game: yes
scores: unavailable
```

Clientが観測する`rejected` / `unparseable`、protocol error、chombo相当のfatal
error、unexpected disconnectは発生せず、`end_game`後に再queue・次gameへ進まず
processが終了した。これによりIssue #42のlive ranked smoke test成功条件を満たした。

### 設計判断

ranked Clientは有効な`end_game`受信そのものを正常終了条件とし、scoreの取得を
成功条件にしない。scoresがない場合は`None`として保持し、値を捏造しない。serverが
validな4整数scoresを通知した場合だけtupleとして保持・表示し、scores fieldが存在
するが不正な場合はfail closedを維持する。

本実装環境へtokenは注入していない。live smoke testには検証用botだけを使用し、
本命bot `lisjong`は使用していない。Policy versionはbot名を増やさずGit commit/tagで
管理する。順位・score・ratingは観測してよいが成功条件にしない。

```powershell
$env:BOT_TOKEN = "<検証用RiichiLab bot token>"
python -m lisjong.riichilab_client.ranked
```

再実行時は次を確認する。

```text
RiichiLab ranked game completed
seat: 0..3
requests: <受信件数>
responses: <送信件数>
end_game: yes
scores: unavailable
```

serverがvalidな4整数scoresを通知した場合だけ、`scores:`にはその4値を表示する。
加えて、`rejected` / `unparseable`、protocol error、chombo、unexpected
disconnectがなく、`end_game`後に再queue・次gameへ進まずprocessが終了することを
確認する。
