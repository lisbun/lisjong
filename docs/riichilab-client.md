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
- `run_ranked_game(policy: Policy, token: str, *, url: str = DEFAULT_RANKED_URL) -> RankedGameResult`:
  `wss://game.riichi.dev/ws/ranked`へ1回だけ接続し、queue待ちから1 full
  hanchanの`end_game`まで処理して終了する
- `RankedGameResult`: `end_game_received`、自seat、request/response件数、
  `ack_history`、optionalなfinal `scores`を持つfrozen dataclass。実serverの
  `end_game`にscoresがない場合は`None`とし、rank / placement / ratingや
  score値を推測・補完しない

`run_validation()` / `run_ranked_game()`は、Policyとcredentialを明示的に
受け取る実行境界として維持する(Issue #44)。`python -m
lisjong.riichilab_client.validation --profile <name>` / `ranked --profile
<name>`は、Issue #44の`lisjong.riichilab_client.profile` / `cli`が解決した
Policy・credential・trace pathをこの境界へ注入するだけのCLI entry point
である。profileの詳細は下記「profile(Issue #44)」を参照する。

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
    trace.py         protocol trace(Issue #45)。JsonlProtocolTraceWriter、
                     ProtocolTraceError
    transport.py     Transport protocol、WebSocketTransport、共通connect/driver、
                     validation/ranked互換wrapper
    profile.py        bot実行profile(Issue #44)。RuntimeProfile、
                     resolve_profile()、resolve_credential()、runtime_root()、
                     default_trace_path()、runtime summary
    cli.py             profile CLI共通の引数解析・trace path解決
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

**設計判断**: tokenはruntime secretとして`run_validation()` /
`run_ranked_game()`の明示引数から注入する。secret管理frameworkは導入していない。

- `token`はAuthorization header(`Bearer <token>`)を設定する目的だけに
  使用し、`Transport`、各session、各resultのいずれにも保持しない
- validation/ranked CLIは、Issue #44の`--profile`が選択したprofile専用の
  環境変数(下記「profile(Issue #44)」を参照)から読み込む。未設定時は
  secretを含まないエラーメッセージをstderrへ出力し、非zero exit codeを返す
- 例外メッセージ・ログ・test fixtureへtoken文字列を含めない設計とした
  (`test_riichilab_client_validation.py`の`SecretHandlingTest`相当の
  確認を`RunValidationEndToEndTest`内で行っている)

## profile(Issue #44)

**設計判断**: [Issue #44](https://github.com/lisbun/lisjong/issues/44)で、
`src/lisjong/riichilab_client/profile.py` / `cli.py`にRiichiLab bot実行
profileのcomposition/configuration layerを実装した。目的は、bot identity・
credential source・使用Policy・runtime namespace・runtime output policyを
明示的なprofileとして分離し、誤ったcredentialやPolicyでbotを誤起動しにくい
構造にすることである。

### 責務境界

profileは次を一方向に解決するだけの、`run_validation()` /
`run_ranked_game()`より上位のlayerである。

```text
profile -> bot identity -> credential環境変数名 -> Policy -> runtime
namespace -> trace/runtime output policy
```

- `profile.py`はPolicy契約(`DecisionContext`)、`RiichiLabSeatAdapter`、
  `ValidationSession` / `RankedSession`、`Transport`のいずれにも依存しない。
  `run_validation(policy, token, ...)` / `run_ranked_game(policy, token,
  ...)`という既存の低レベル公開APIはIssue #44でも変更せず、profileは
  これらへ渡す`policy`(`RuntimeProfile.policy_factory()`)と`token`
  (`resolve_credential()`)を組み立てるだけである
- `cli.py`は`--profile` / `--trace` / `--trace-path`のCLI引数解析と、
  Issue #45 `RIICHILAB_TRACE_PATH`との優先順位解決だけを担当する薄いlayer
  であり、Session/Transport/Adapter/Policyへ責務を持ち込まない

### 3 profile

少なくとも次の3 profileを`PROFILE_NAMES`として固定する
(`tests/test_riichilab_client_profile.py`の`ProfileMappingTest`で
mappingをregression test化している)。

| profile | 用途 | credential環境変数 | Policy | runtime namespace |
| --- | --- | --- | --- | --- |
| `lisjong-dev` | 開発・smoke test・protocol調査用 | `LISJONG_DEV_BOT_TOKEN` | `MinimalPolicy` | `lisjong-dev` |
| `lisjong-baseline` | Policy性能比較の決定的な基準 | `LISJONG_BASELINE_BOT_TOKEN` | `MinimalPolicy` | `lisjong-baseline` |
| `lisjong` | 本番運用(十分に検証済みのPolicyのみ) | `LISJONG_BOT_TOKEN` | `MinimalPolicy` | `lisjong` |

3 profileとも現時点では`MinimalPolicy`(決定的、hidden mutable stateなし)
を使用する。Issue #44はPolicyの強さそのものを改善する対象ではなく、profileを
区別するためだけの不要なPolicy classも追加しない。`lisjong-baseline`には
比較基準として固定可能な決定的Policyを割り当てるという完了条件を、既存の
`MinimalPolicy`で満たす。各`RuntimeProfile.policy_factory`は独立した
callableであるため、将来`lisjong-dev` / `lisjong-baseline` / `lisjong`を
別Policyへ個別に差し替えることができる。

### credential解決とfail closed

`resolve_credential(profile, env=os.environ)`は、`profile.credential_env_var`
**だけ**を参照する。

- 他profileの環境変数を探索・流用しない(production → dev/baseline、
  逆方向のいずれも実装しない、`ResolveCredentialTest`で固定)
- 対応する環境変数が未設定・空文字列の場合は`MissingCredentialError`で
  fail closedする。例外メッセージには環境変数の**名前**だけを含め、値は
  含めない
- `resolve_profile(name)`は、`name`が`None`・空文字列・未知profileの
  いずれの場合も`UnknownProfileError`でfail closedする。profile未指定を
  production等へ暗黙fallbackさせない
- token値・token fingerprintはprofile identityとして使用しない
  (`RuntimeProfile`はcredential**環境変数名**だけを保持し、値は保持しない)

### CLI

```powershell
$env:LISJONG_DEV_BOT_TOKEN = "<dev検証用bot token>"
python -m lisjong.riichilab_client.ranked --profile lisjong-dev

$env:LISJONG_BASELINE_BOT_TOKEN = "<baseline検証用bot token>"
python -m lisjong.riichilab_client.ranked --profile lisjong-baseline

$env:LISJONG_BOT_TOKEN = "<本番bot token>"
python -m lisjong.riichilab_client.ranked --profile lisjong
```

`validation`も同じ`--profile`引数を持つ(`python -m
lisjong.riichilab_client.validation --profile <name>`)。

- `--profile`は必須引数である。未指定はargparseの標準fail-closed挙動
  (non-zero exit、usageをstderrへ出力)に委ね、production等へ暗黙
  fallbackしない
- `--profile`は`choices=PROFILE_NAMES`で制限し、未知profileも
  `resolve_profile()`と同様に明確に拒否する
- 旧CLI(profile概念導入前)が読んでいた単一の`BOT_TOKEN`環境変数は、
  Issue #44で3profile専用の環境変数へ置き換えた(破壊的変更。fail-closed
  原則を優先し、`BOT_TOKEN`への後方互換fallbackは実装していない)

### runtime output / trace保存先

profile経由の既定trace保存先は、repository配下ではなくOSユーザーローカル
領域を使用する(`profile.runtime_root()`、標準libraryだけで実装、新規
dependencyは追加していない)。

| OS | root |
| --- | --- |
| Windows | `%LOCALAPPDATA%\lisjong`(未設定時は`~\AppData\Local\lisjong`) |
| macOS | `~/Library/Application Support/lisjong` |
| Linux等 | `$XDG_DATA_HOME/lisjong`(未設定時は`~/.local/share/lisjong`) |

`default_trace_path(profile)`は、`<root>/traces/<runtime_namespace>/
<timestamp>-<uuid4>.jsonl`の形でpathを作る。timestamp(UTC, マイクロ秒まで)
とUUID4を組み合わせることで、同一profileを複数回・複数processで実行しても
既定trace fileが同じfileへ意図せず混在しない
(`DefaultTracePathTest.test_concurrent_calls_do_not_collide`で
複数threadからの同時呼び出しでも衝突しないことを確認している)。filename/
directory名にはtimestampとUUID4しか使わず、credential値・断片は一切
含めない。

trace pathの解決優先順位は次のとおりである(`cli.resolve_trace_path()`)。

1. `--trace-path <path>`による明示指定
2. 既存`RIICHILAB_TRACE_PATH`環境変数(Issue #45、後方互換として維持)
3. `--trace`指定時のprofile既定path(上記`default_trace_path()`)
4. どれも指定がなければtrace無効(`trace_path=None`)

**trace既定値の判断**: Issue #45の設計原則(tracingは既定OFFのopt-in)を
profile層でも維持し、`lisjong-dev`を含むどのprofileも既定ではtraceを
生成しない。`lisjong-dev`だけ既定ONにする案も検討したが、同じCLI呼び出しが
profileによって暗黙に異なる副作用(trace file生成)を持つことは「明示的な
opt-in」という#45の契約と整合しないと判断した。dev用途でtraceを使う場合は
`--trace`(profile既定path)または`--trace-path`/`RIICHILAB_TRACE_PATH`
(明示path)で毎回明示する。この判断は`RuntimeSummaryTest` /
`ResolveTracePathTest`のtrace既定OFF確認で固定している。

### 起動時のsecret-free runtime summary

profile CLIは、実行を開始する前に`build_runtime_summary()` /
`format_runtime_summary()`が組み立てるsummaryを表示する。

```text
profile: lisjong-baseline
policy: MinimalPolicy
mode: ranked
trace: off
```

`trace`が有効な場合は`trace path: <path>`をあわせて表示する。summaryは
profile名、Policy識別子、mode、trace ON/OFF、trace有効時のpathだけを
表示し、BOT token・Authorization header・token fingerprint・credential
環境変数の値は一切含めない。credential環境変数の**名前**も、利便性より
情報露出の最小化を優先し表示しない。

### multi-process independence

別processから`lisjong-dev`と`lisjong-baseline`を同時起動しても、
credential source・Policy selection・runtime namespace・trace/output path
が混線しないことを、次の2種類のtestで確認している。

- `tests/test_riichilab_client_profile.py`: profile解決・credential解決・
  `default_trace_path()`が共有mutable stateを持たない純粋関数であることを
  確認する(`MultiProfileIndependenceTest`、`DefaultTracePathTest`の
  concurrent呼び出しtestを含む)
- `tests/test_riichilab_client_ranked.py`の
  `MultiProcessProfileIndependenceTest`: 実OS processを2つ同時起動し
  (`subprocess.Popen`)、`lisjong-dev`向けprocessが`LISJONG_DEV_BOT_TOKEN`
  だけを、`lisjong-baseline`向けprocessが`LISJONG_BASELINE_BOT_TOKEN`
  だけを参照すること、互いのcredential環境変数名や値がもう一方の
  stdout/stderrへ漏れないことを確認する。credential解決はnetwork接続前に
  fail closedするため、live RiichiLab接続やprocess supervisorは不要である

process supervisor、4 bot一括起動orchestrator、reconnect、auto requeueは
Issue #44の非スコープであり実装していない。RiichiLab側で同一user所有の
別bot同士が同一ranked matchへ選ばれるかどうかは、lisjong側のprofile分離
とは独立した外部サービスの挙動であり、本Issueでは調査・保証しない
(必要であれば別Issueで扱う)。

## protocol trace(Issue #45)

**設計判断**: [Issue #45](https://github.com/lisbun/lisjong/issues/45)で、
RiichiLab validation / ranked sessionの送受信protocol eventを、任意で
secret-safeなJSON Lines(JSONL)へ保存できる観測機能を実装した。目的は、
PR #43 / Issue #42のranked live smoke testで実測した
`ProtocolError: ranked end_game must contain four final scores`のような
protocol差異・unknown event・`ProtocolError`原因調査を、実RiichiLabに対して
安全に継続できるようにすることである。Policy契約
(`DecisionContext` / `InternalAction`)やDecisionContext・
`RiichiLabSeatAdapter`へこの責務を持ち込まない。

### 責務境界

`src/lisjong/riichilab_client/trace.py`が次のみを担当する。

- `JsonlProtocolTraceWriter(path)`: 出力先pathだけを受け取るwriter。
  JSONL recordの生成、UTC ISO 8601 timestampの付与、fileへのappendを行う
- `ProtocolTraceError`(`RiichiLabClientError`のsubclass): trace file
  のopen/write/close失敗時に送出する専用例外

`Policy`、`DecisionContext`、`RiichiLabSeatAdapter`はtrace writerを一切
知らない。trace挿入点は`connect_transport()`(BOT token / Authorization
headerを知る唯一の場所)ではなく、credentialを持たない共通
`transport.drive_session()`である。

### opt-inであること

tracingは既定で無効である。

- `run_validation(policy, token, *, url=..., trace_path=None)` /
  `run_ranked_game(policy, token, *, url=..., trace_path=None)`は、
  `trace_path`(`str` / `os.PathLike`)を明示的に渡した場合だけ
  `JsonlProtocolTraceWriter`を生成し、`drive_session()`へ渡す
- `trace_path=None`(既定)では trace fileはまったく作られず、既存の
  `ValidationResult` / `RankedGameResult` / CLI出力の挙動は変わらない
- validation/ranked CLI(`python -m lisjong.riichilab_client.validation` /
  `ranked`)は、profile credential(Issue #44、下記「profile(Issue #44)」を
  参照)とは独立した環境変数`RIICHILAB_TRACE_PATH`が設定されている場合だけ、
  そのpathを`trace_path`として使用する。credential環境変数を設定しても
  tracingは有効化されない。Issue #44導入後は、`--trace` CLI引数による
  profile既定pathでのopt-inも利用できる(優先順位は「profile(Issue #44)」
  の「runtime output / trace保存先」を参照)

### JSONL schema

1行が1 protocol event/actionの独立したvalid JSONである。

```json
{"timestamp": "2026-08-15T12:00:00.123456+00:00", "direction": "recv", "event_type": "start_game", "payload": {"type": "start_game", "id": 0}}
{"timestamp": "2026-08-15T12:00:01.000000+00:00", "direction": "recv", "event_type": "request_action", "payload": {"type": "request_action", "request_id": 17, "...": "..."}}
{"timestamp": "2026-08-15T12:00:01.050000+00:00", "direction": "send", "event_type": "dahai", "payload": {"type": "dahai", "actor": 0, "pai": "1m", "request_id": 17}}
```

| field | 内容 |
| --- | --- |
| `timestamp` | `datetime.now(timezone.utc).isoformat()`によるtimezone-aware UTC ISO 8601 |
| `direction` | `"recv"` または `"send"` |
| `event_type` | recvは受信event `payload["type"]`の値(欠落時は`null`)、sendは送信action `payload["type"]`の値 |
| `payload` | 受信済みparsed JSON event、または送信直前にJSON serialization済みのaction dictそのもの |

`payload`はJSON syntax errorや非objectなど受理できなかったtext frameを
含まない(下記「recvの記録タイミング」を参照)。session state、Policy
state、Authorization metadataはrecordへ追加しない。

### recv/sendの記録タイミング

`transport.drive_session()`の順序は次のとおりである。

```text
recv -> frame種別判定 -> (binary: ignore, 記録なし)
      -> JSON parse (失敗: ProtocolError, 記録なし)
      -> trace記録(“recv”)
      -> session.handle_event() (ProtocolErrorの可能性あり)
      -> outgoing payload -> JSON serialization (失敗: ProtocolError, 記録なし)
      -> trace記録(“send”)
      -> transport.send() (失敗: TransportError)
```

- **recv**: `session.handle_event()`より前に記録する。これにより、
  unknown eventや、既知eventのmalformed fieldが原因で`ProtocolError`に
  なる場合でも、その原因となった受信eventがtraceへ残る
  (`ProtocolErrorになる前のrecv trace`)。binary frameとJSON syntax
  errorは、そもそも扱えるprotocol payloadではないため記録しない
  (Issue #45の「各行がvalid JSON」という契約を優先する)
- **send**: `session.handle_event()`が返したoutgoing payloadのJSON
  serializationに成功した後、実`transport.send()`の前に記録する。
  serializationに失敗したpayloadは「送信済み」として記録しない。
  一方、record自体は「送信を試みた」ことだけを表し、直後の実
  `transport.send()`が`TransportError`で失敗した場合でもrecordは
  そのまま残る(=「相手へ届いた」ことの証明ではない)

### secret境界

- `JsonlProtocolTraceWriter`のconstructorは出力先pathしか受け取らない。
  BOT_TOKEN、Authorization header、その他credentialを引数として渡す
  経路自体が存在しない
- `connect_transport()`だけがBOT token / Authorization headerを知り、
  `Transport`・`drive_session()`・trace writerのいずれにもcredentialを
  渡さない、という既存の責務境界を変更していない
- 単純な文字列redaction(既知secret文字列をtraceへ書いてから置換する)
  には依存していない。secretがtrace boundaryを通る経路自体を作らない
  構造でsecret-safeを担保する
- protocol payload自体(`request_action.observation`、`possible_actions`
  等)はredactionせずそのまま記録する。protocol調査能力を優先し、
  Issue #45に個別のredaction要求は定義されていない

### 保存場所

trace出力先はrepositoryへ誤commitしにくいよう、専用directory
`/traces/`をGit管理対象外とした(`.gitignore`)。`trace_path`は
呼び出し側が指定する任意のpathであり、`/traces/`配下を使うことを
推奨するが、writer自体はpathを強制しない。ファイル名にはsecretを
含めないこと。

### trace writer failure

trace書き込み失敗をsilentに無視しない。`JsonlProtocolTraceWriter`は
open/write/close失敗時に`ProtocolTraceError`(`RiichiLabClientError`の
subclass)を送出する。`run_validation()` / `run_ranked_game()`は
既存の`RiichiLabClientError`捕捉(CLIの`except RiichiLabClientError`)に
そのまま乗り、既存のexception hierarchyや`ProtocolError` /
`TransportError`の意味を変更しない。tracingを有効化した利用者が
trace保存失敗に気づけず「通常成功したように見える」挙動は発生しない。

### validation / ranked共通化

trace実装は共通`transport.drive_session()`に1回だけあり、
`drive_validation_session()` / `drive_ranked_session()`双方が同じ
`trace`引数を透過的に渡す。mode固有のprotocol semantics
(`ValidationSession` / `RankedSession`のterminal条件等)はSession側の
責務のまま変更していない。

### 有効化方法

```python
result = await run_validation(policy, token, trace_path="traces/validate.jsonl")
result = await run_ranked_game(policy, token, trace_path="traces/ranked.jsonl")
```

```powershell
$env:LISJONG_DEV_BOT_TOKEN = "<dev検証用bot token>"
$env:RIICHILAB_TRACE_PATH = "traces/ranked.jsonl"
python -m lisjong.riichilab_client.ranked --profile lisjong-dev
```

`RIICHILAB_TRACE_PATH`を設定しない場合、CLIは従来どおりtrace fileを
作らない。Issue #44導入後は、`RIICHILAB_TRACE_PATH`を設定せずとも
`--trace`だけでprofile既定path(OSユーザーローカル領域)へtraceを
保存できる(`python -m lisjong.riichilab_client.ranked --profile
lisjong-dev --trace`)。

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
   failureの例外変換、serialization failureを確認する。あわせて
   `ProtocolTraceIntegrationTest`が、Issue #45のprotocol traceを
   `drive_session()`の境界で確認する: recv eventが
   `session.handle_event()`より前に記録されること、malformed known
   eventが`ProtocolError`になる直前でもrecv traceが残ること、unknown
   eventが記録されforward-compatible挙動を壊さないこと、recv/sendの
   記録順序、binary frameを記録しないこと、serialization失敗を
   送信済みとして記録しないこと、実`transport.send()`失敗後もsend
   recordが残ること、trace writer failureがsilentに無視されないこと、
   実`JsonlProtocolTraceWriter`で複数recordを1行1JSONとして読み戻せる
   こと、BOT_TOKEN/Authorization/Bearerに相当する文字列がtrace出力へ
   含まれないことを確認する
3. **#38 + `MinimalPolicy` integration** (`tests/test_riichilab_client_validation.py`):
   実RiichiEnv 0.4.8の`Observation`を使い、`ValidationSession`が
   `RiichiLabSeatAdapter` + `MinimalPolicy`を経て送信前validation済み
   MJAI responseまで届くこと、Policyへ`request_id`/`time`/`ack`/
   WebSocket固有情報が漏れないことを確認する。あわせて
   `run_validation()`をfake transportで駆動するend-to-end testで、
   `ValidationResult`の内容とtoken非露出を確認する。CLIをtoken未設定で
   subprocess実行し、`python -m`でrunpy `RuntimeWarning`が発生しないことも
   回帰testで固定する。`RunValidationTraceOptInTest`が、`trace_path`
   未指定時にtrace fileが作られないこと、`trace_path`指定時に
   `run_validation()`経由でJSONL recordが書き出され、tokenを含まない
   ことを確認する
4. **ranked unit / fake transport / integration**
   (`tests/test_riichilab_client_ranked.py`): seat 0..3、validation seat 0回帰、
   scoresなし`end_game` terminal、optionalなvalid scores、不正scores、
   no join payload、exactly one game、
   binary/unknown event、unexpected disconnect、#38 + `MinimalPolicy` integration、
   secret-safe result、ranked CLIのRuntimeWarning非再発を確認する。
   `RunRankedGameTraceOptInTest`が`run_ranked_game()`側の同じopt-in
   trace挙動を、`RankedModuleCliTest`が`--profile`必須・未知profile拒否・
   credential未設定fail closed・`RIICHILAB_TRACE_PATH`環境変数から
   `trace_path`が独立して渡されることを確認し、validation/ranked双方が
   共通`drive_session()`のtrace実装を利用することを固定する。
   `MultiProcessProfileIndependenceTest`が、Issue #44セクション9の
   multi-process independence(下記「profile(Issue #44)」を参照)を
   実subprocessで確認する
5. **`JsonlProtocolTraceWriter`のunit test**
   (`tests/test_riichilab_client_trace.py`): 実WebSocket/RiichiLabなしに、
   JSONL record生成、timezone-aware ISO 8601 timestamp、複数recordの
   1行1JSON読み戻し、親directory自動作成、open/write失敗時に
   `ProtocolTraceError`をsilentに無視せず送出することを確認する
6. **profile / CLI layerのunit test**
   (`tests/test_riichilab_client_profile.py`、
   `tests/test_riichilab_client_cli.py`): 3 profileのmapping固定、
   profile未指定・未知profile・credential未設定のfail closed、他profile
   credentialへのfallbackがないこと、`runtime_root()`のOS別解決、
   `default_trace_path()`の複数run/複数thread同時呼び出しでの非衝突、
   secret-freeなruntime summary、`--profile` / `--trace` /
   `--trace-path`の引数解析とtrace path優先順位を実WebSocket/RiichiLab
   なしに確認する
7. **manual live validation / ranked smoke test**: 下記を参照

## live validation

2026-08-15に学習者のWindows / Python 3.14環境から実BOT_TOKENをruntime
注入し、duplicate semantic candidate対応後の実`/ws/validate`を再実行した
(Issue #44導入前の`BOT_TOKEN`単一環境変数によるCLIで実行した記録)。

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

再実行する場合は、Issue #44導入後のprofile CLIで次のコマンドを使用する。

```powershell
$env:LISJONG_DEV_BOT_TOKEN = "<dev検証用bot token>"
python -m lisjong.riichilab_client.validation --profile lisjong-dev
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

この実測はIssue #44導入前の`BOT_TOKEN`単一環境変数によるCLIで実行した記録
である。再実行する場合は、Issue #44導入後のprofile CLIで次のコマンドを
使用する(検証用botには`lisjong-dev`または`lisjong-baseline`を使用し、
本命bot `lisjong`は使用しない)。

```powershell
$env:LISJONG_DEV_BOT_TOKEN = "<検証用RiichiLab bot token>"
python -m lisjong.riichilab_client.ranked --profile lisjong-dev
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
