# RiichiLab WebSocket Client

この文書は、`src/lisjong/riichilab_client/`に現在残るRiichiLab lower-level runtimeの
責務境界と、RiichiLab protocolについて確認済みの事実・実測・設計判断を記録する。

project-wideなrepository責務は[`lisjong-project`](https://github.com/lisbun/lisjong-project)
を正本とする。Policy判断、Observation変換、Action mapping、`possible_actions`
semantic validationの詳細は[RiichiLab request_action Adapter](riichilab-adapter.md)を
正本とする。

## 現在のownership

2026-08-22時点で、RiichiLab ranked / validation one-game orchestrationと
execution profile / credential / common CLI compositionのcanonical ownerは
`lisjong-arena`である。

- `lisbun/lisjong-arena#17` / PR #18で、Arena-local
  `lisjong_arena.riichilab.ranked.RankedGameResult`と`run_ranked_game()`、
  first-party ranked CLIを実装した
- `lisbun/lisjong#86`で、lisjong側のlegacy
  `RankedGameResult` / `run_ranked_game()` / `python -m lisjong.riichilab_client.ranked`
  とpackage-root exportを除去済みである
- `lisbun/lisjong-arena#19` / PR #20で、Arena-local
  `lisjong_arena.riichilab.validation.ValidationResult`と`run_validation()`、
  first-party validation CLI、execution profile / credential / common CLI /
  trace-path composition(`lisjong_arena.riichilab.profile` / `lisjong_arena.riichilab.cli`)
  を実装した。Arena ranked CLIもこの時点でlisjong側profile / CLI helperへの依存を
  解消し、Arena-local compositionへ切り替えている
- `lisbun/lisjong#89`で、lisjong側のlegacy `ValidationResult` / `run_validation()` /
  `python -m lisjong.riichilab_client.validation`、`profile.py`、`cli.py`と
  package-root exportを除去済みである
- compatibility re-exportは作らず、`lisjong -> lisjong-arena`のreverse dependencyを
  導入しない
- Session / transport / protocol trace writer / Adapterは、physical migrationが
  完了するまでlisjongに残り、Arenaがtemporaryにconsumerとなる

現在の境界は次のとおりである。

```text
lisjong-arena
    canonical ranked / validation one-game orchestration
    RankedGameResult / run_ranked_game()
    ValidationResult / run_validation()
    first-party ranked / validation CLI
    execution profile / credential / common CLI composition
        |
        v
lisjong
    RankedSession / ValidationSession
    Transport / WebSocketTransport
    connect_ranked_transport() / connect_validation_transport()
    drive_ranked_session() / drive_validation_session()
    JsonlProtocolTraceWriter
    RiichiLabSeatAdapter
    Policy contract
```

Arenaへのdependency directionは`lisjong-arena -> lisjong`である。lower-level runtimeが
Arenaへ逆依存してはならない。

## 区分

| 区分 | 意味 |
| --- | --- |
| 公式情報 | RiichiLab公式文書で確認した情報 |
| 実測 | 実RiichiLab接続または実RiichiEnvを使ったtestで確認した情報 |
| 推測・未確認 | 公式情報と実測のどちらでも確認できていない事項 |
| 設計判断 | 確認結果からlisjong / Arenaの実装へ引き継ぐ判断 |

## 公式仕様の再確認履歴

Issue #42実装前の2026-08-15に、RiichiLab公式文書を再確認した。

- [MJAI Protocol](https://riichi.dev/docs/protocol): `/ws/ranked`と`/ws/validate`は
  Bearer tokenで接続する。serverがgame loopを駆動し、botは`request_action`への
  responseを送る。`start_game.id`は0..3、`request_id`はgame内で単調増加するinteger。
  binary frameと未知event / fieldはignoreする
- [Ranked Matches](https://riichi.dev/docs/ranked): ranked endpointへ接続すると
  matchmaking queueへ入り、4 botでfull hanchanを行い、serverが`end_game`を送る
- [Matchmaking](https://riichi.dev/docs/matchmaking): endpointへの接続自体がqueue参加で
  あり、Client側のjoin payloadやpollingは不要
- [Rating System](https://riichi.dev/docs/rating): ratingはserver側で管理され、
  `end_game` payloadにratingが含まれる保証はない

これは2026-08-15時点の確認記録であり、将来の仕様変更を固定するものではない。

## lisjong側の公開境界

Issue #89後もlisjong側に残す主要な公開APIは次である。

- session:
  `RankedSession`、`ValidationSession`、`SessionStatus`
- transport:
  `Transport`、`connect_transport()`、`connect_validation_transport()`、
  `connect_ranked_transport()`、`drive_session()`、`drive_validation_session()`、
  `drive_ranked_session()`、`DEFAULT_VALIDATION_URL`、`DEFAULT_RANKED_URL`
- trace:
  `JsonlProtocolTraceWriter`、`ProtocolTraceError`
- errors:
  `RiichiLabClientError`、`ProtocolError`、`TransportError`、
  `UnexpectedDisconnectError`

`RankedGameResult` / `run_ranked_game()`、`ValidationResult` / `run_validation()`は
いずれもlisjong側の公開APIではない。one-game orchestrationが必要なconsumerは
Arenaのcanonical APIを使う。execution profile / credential resolution / common CLI
引数解析 / trace-path解決も同様にlisjong側にはなく、`lisjong_arena.riichilab.profile` /
`lisjong_arena.riichilab.cli`がcanonicalである。

## package構成

```text
src/lisjong/riichilab_client/
    __init__.py     lower-level public API
    errors.py       client error hierarchy
    session.py      validation/ranked共通lifecycle
    trace.py        secret-safe protocol trace
    transport.py    Transport / WebSocketTransport / connect / drive
```

Issue #86で`ranked.py`、Issue #89で`validation.py` / `profile.py` / `cli.py`は
削除済みである。rankedのSession / transport差分は`session.py` / `transport.py`に
残るため、file削除とlower-level runtime削除を同一視しない。

## lower-level Session / transport contract

### `start_game` / seat bind

公式protocolではbot seat indexは`start_game.id`で通知される。

- `id`は`int`（`bool`除外）かつ0..3を要求する
- validationではseat 0のみを受理する
- rankedではseat 0..3を受理する
- `start_game`前の`request_action`はfail closed
- 同一seatのduplicate `start_game`は既存Adapterを保持する
- 異なるseatへのduplicate bindはfail closed
- legacy `seat` fieldだけをfallbackとして扱わない

### `request_action` / `request_id`

`request_id`はgame内で単調増加するintegerとして扱い、`+1`ずつの連番は仮定しない。

1. `start_game`未受信ならfail closed
2. `request_id`の型を検証する
3. duplicate / decreaseを拒否する
4. `time`がある場合はmetadataの型だけを検証し、Policyへ渡さない
5. `RiichiLabSeatAdapter.process_request_action()`へ委譲する
6. Adapter resultの`request_id`がcurrent requestと一致することを確認する
7. 同じrequestへの二重sendを拒否する
8. send直前にもcurrent requestへのbindを再検証する

`request_id`、time budget、ack、transport object等を`DecisionContext`へ混入させない。

### `action_ack`

`action_ack`は`request_id`ごとのstatus historyとして保持し、1 request = 1 ackとは
仮定しない。

- `accepted`: historyへ記録
- `stale`: non-fatalとしてhistoryへ記録
- `defaulted`: non-fatalとしてhistoryへ記録
- `rejected`: historyへ記録後fail closed
- `unparseable`: historyへ記録後fail closed
- unknown request ID、未知status、malformed fieldはfail closed

### validation terminal

validationでは`end_game`だけで完了せず、`validation_result`を待つ。
`validation_result.passed`が成功判定の正本であり、Client独自条件から成功を推測しない。

### ranked terminal

rankedでは、`start_game`後の有効な`end_game`をterminalとする。

2026-08-15の実ranked serverでは、`end_game`が`{"type":"end_game"}`相当で
`scores`を含まないケースを観測した。したがってlower-level `SessionStatus.scores`は
optionalである。

- scores欠落: `None`
- scores存在時: 4個の`int`（`bool`除外）だけを受理
- 不正scores: 値をdumpせずshape情報だけでfail closed
- rank / placement / ratingを推測しない
- `end_game`前のdisconnectは`UnexpectedDisconnectError`

Arenaの`RankedGameResult`も、このlower-level status contractをconsumerとして扱う。

## transport / WebSocket boundary

`Transport` protocolは`recv()` / `send()` / `close()`だけを要求する。
`WebSocketTransport`が`websockets` libraryをこの最小interfaceへ適合させる。

Issue #39実装時は`websockets==17.0.1`を採用した。`websockets` dependencyは
`riichilab_client`内へ閉じ込め、`policy_contract` / `policies` /
`riichienv_adapter`へ逆流させない。

`drive_session()`の基本順序は次である。

```text
recv
 -> frame種別判定
 -> JSON parse
 -> optional recv trace
 -> session.handle_event()
 -> outgoing JSON serialize
 -> optional send trace
 -> transport.send()
```

- binary frameはignoreし、traceにも書かない
- JSON syntax error / top-level非objectは`ProtocolError`
- unknown event typeやknown eventのunknown追加fieldはforward-compatibleに許容する
- known eventの必須field欠落・型不正はfail closed
- send / recv transport failureはclient error hierarchyへ変換する
- arbitrary fallback Actionは生成しない

## reconnect / continuous execution

lisjongに残るlower-level runtimeはmid-game reconnectを行わない。
unexpected disconnectは成功扱いせず`UnexpectedDisconnectError`とする。

rankedのretry / backoff / requeue / continuous participationはArena側の上位
orchestration責務であり、Issue #86のcleanupと同時には実装しない。

## Token境界

BOT tokenはruntime secretとして扱う。

- validationではArenaの`run_validation(policy, token, ...)`へ明示注入し、Arenaが
  lisjongのlower-level `connect_validation_transport()`へ渡す
- rankedではArenaの`run_ranked_game(policy, token, ...)`へ明示注入し、Arenaが
  lisjongのlower-level `connect_ranked_transport()`へ渡す
- tokenはAuthorization header設定にのみ使い、Session / result / trace payloadへ
  保持しない
- token値、Authorization header、token fingerprintをlog / exception / result /
  test fixtureへ含めない
- credential環境変数の値をrepositoryへcommitしない

## execution profile / credential / CLI composition (Arena-owned, Issues #44 / #45 / #19)

bot identity、credential source、Policy、runtime namespace、trace output policyを
一方向に解決するcomposition/configuration layer(旧`profile.py` / `cli.py`)は、
`lisjong-arena` Issue #19でcanonical implementationがArenaへ移り、本Issue #89で
lisjong側のlegacy実装を削除した。

```text
profile
 -> bot identity
 -> credential環境変数名
 -> Policy factory
 -> runtime namespace
 -> trace path policy
```

は`lisjong_arena.riichilab.profile` / `lisjong_arena.riichilab.cli`がcanonical +
physical ownerである。少なくとも次の3 profileを提供する(mapping自体の正本はArena側)。

| profile | credential環境変数 | Policy | runtime namespace |
| --- | --- | --- | --- |
| `lisjong-dev` | `LISJONG_DEV_BOT_TOKEN` | `TwoStepUkeirePolicy` | `lisjong-dev` |
| `lisjong-baseline` | `LISJONG_BASELINE_BOT_TOKEN` | `MinimalPolicy` | `lisjong-baseline` |
| `lisjong` | `LISJONG_BOT_TOKEN` | `MinimalPolicy` | `lisjong` |

- 他profileのcredentialへfallbackしない
- profile未指定・未知profile・credential未設定はfail closed
- credentialの値ではなく環境変数名だけを設定として保持する
- lisjongが引き続き所有するのは、profile mappingが参照するPolicy class自体
  (`MinimalPolicy` / `TwoStepUkeirePolicy`等)だけである

ranked / validation双方のfirst-party CLIはArenaを使用する。

```powershell
$env:LISJONG_DEV_BOT_TOKEN = "<dev用bot token>"
python -m lisjong_arena.riichilab.ranked --profile lisjong-dev
python -m lisjong_arena.riichilab.validation --profile lisjong-dev
```

## protocol trace (Issue #45)

`JsonlProtocolTraceWriter`は、送受信protocol eventを任意のJSON Linesへ保存する。
tracingは既定OFFのopt-inである。

1 recordは次のfieldを持つ。

```json
{"timestamp":"...","direction":"recv","event_type":"start_game","payload":{"type":"start_game","id":0}}
```

- `timestamp`: timezone-aware UTC ISO 8601
- `direction`: `recv` / `send`
- `event_type`: payloadの`type`
- `payload`: parsed protocol object

trace writerはtoken / Authorization headerを受け取らない。credentialをtrace boundaryへ
渡す経路自体を作らないことでsecret-safeを担保する。

record timingは次の契約とする。

- recv: JSON parse成功後、`session.handle_event()`より前
- send: JSON serialize成功後、実`transport.send()`より前
- binary frame / JSON syntax error / serialization failureは記録しない
- writer open/write/close failureは`ProtocolTraceError`としてfail closed

`drive_validation_session()` / `drive_ranked_session()`は共通`drive_session()`のtrace
実装を利用する。validation / ranked双方でArena orchestrationがwriterを生成し、この
lower-level driverへ渡す。

trace path解決の優先順位は、canonical implementationである
`lisjong_arena.riichilab.cli`が次のとおり維持する。

1. `--trace-path`
2. `RIICHILAB_TRACE_PATH`
3. `--trace`指定時のprofile既定path
4. 無効

profile既定pathはrepository配下ではなくOS user-local領域を使い、profile別directoryと
UTC timestamp + UUID4で衝突を避ける。この解決ロジック自体はlisjongには存在しない。

## fail-closed原則

次をsilent補正しない。

- JSON parse failure
- known lifecycle event malformed
- invalid seat bind
- `request_action` before `start_game`
- duplicate / decreasing request ID
- Adapter response request ID mismatch
- invalid / fatal `action_ack`
- malformed validation result
- ranked `end_game`に存在する不正scores
- WebSocket send / receive failure
- unexpected disconnect
- Adapter / Policy / action mapping / possible-action validation failure
- trace writer failure

どのfailureでも`possible_actions[0]`、tsumogiri、`none`等のarbitrary fallbackを
Client側から生成しない。

## test ownership

Issue #89後もlisjong側で保持するtestはlower-level runtimeのcontractだけを対象とする。

- `tests/test_riichilab_client_session.py`: 共通lifecycle / validation / ranked terminal差分
- `tests/test_riichilab_client_transport.py`: transport / JSON / trace integration
- `tests/test_riichilab_client_validation.py`: `ValidationSession` lifecycleと#38 Adapter +
  Policy integration(lower-level onlyへ縮小済み)
- `tests/test_riichilab_client_trace.py`: trace writer
- `tests/test_riichilab_client_ranked.py`: `RankedSession`、ranked terminal、fake transport、
  Adapter + Policy integration、およびlegacy orchestration APIがlisjongから消えたこと

`test_riichilab_client_ranked.py` / `test_riichilab_client_validation.py`をfile単位では
削除しない。`RunRankedGameTest` / legacy ranked CLI / `RankedGameResult` contract、
`RunValidationEndToEndTest` / legacy validation CLI / `ValidationResult` contract等の
上位orchestration coverageだけを除去する。Arena-local `run_ranked_game()` /
`RankedGameResult` / ranked CLIと`run_validation()` / `ValidationResult` / validation CLI
のcoverageはcanonical ownerである`lisjong-arena`側が担当する。

`tests/test_riichilab_client_profile.py` / `tests/test_riichilab_client_cli.py`は
Issue #89でlisjong-owned responsibilityが残らないことを確認したうえでfile自体を
削除した。profile / credential / CLI compositionのcoverageは`lisjong-arena`側の
`tests/test_riichilab_profile.py` / `tests/test_riichilab_cli.py`が担当する。

## live実測履歴

### validation

2026-08-15、実RiichiLab `/ws/validate`でvalidationを完走した。

```text
RiichiLab validation passed
requests: 108
responses: 108
end_game: yes
```

これにより、WebSocket authorization、`start_game.id`、`request_action`、Observation
deserialize、Policy execution、Action mapping、MJAI response、`end_game`、
`validation_result.passed`までの経路を実serverで確認した。

この記録は当時のlisjong-owned validation runnerで取得したhistorical実測である。
現在のfirst-party validation executionはArenaへ移管済みであり、このhistoryを
lisjong側APIが現存する根拠として扱わない。

### ranked

2026-08-15、実RiichiLab `/ws/ranked`で初期実装が`scores`を必須としたため
`ProtocolError`となり、その後のsecret-safe shape診断で実serverの`end_game`に
`scores`がないケースを確認した。optional scoresへ修正後、1半荘を完走した。

```text
RiichiLab ranked game completed
seat: 3
requests: 85
responses: 85
end_game: yes
scores: unavailable
```

この記録は当時のlisjong-owned ranked runnerで取得したhistorical実測である。
現在のfirst-party ranked executionはArenaへ移管済みであり、このhistoryを
lisjong側APIが現存する根拠として扱わない。

## Issue #86のbreaking change

Issue #86はpublic APIのbreaking removalを含む。

削除対象:

- `lisjong.riichilab_client.ranked`
- `lisjong.riichilab_client.RankedGameResult`
- `lisjong.riichilab_client.run_ranked_game`
- `python -m lisjong.riichilab_client.ranked`

first-party replacementはArena #17 / PR #18で成立済みである。deprecation wrapperや
re-exportをlisjongに残すと、reverse dependencyまたはlegacy ownershipの固定化に
つながるため設けない。

## Issue #89のbreaking change

Issue #89もpublic APIのbreaking removalを含む。

削除対象:

- `lisjong.riichilab_client.validation`
- `lisjong.riichilab_client.ValidationResult`
- `lisjong.riichilab_client.run_validation`
- `python -m lisjong.riichilab_client.validation`
- `lisjong.riichilab_client.profile`(execution profile / credential resolution)
- `lisjong.riichilab_client.cli`(common CLI引数解析 / trace-path解決)

first-party replacementはArena #19 / PR #20で成立済みである。`profile.py` / `cli.py`は
package rootでstable public APIとして保証していたsurfaceではなく、Arenaへcanonical
移管済みのlegacy execution-composition implementation surfaceとして削除した。
deprecation wrapperやre-exportをlisjongに残すと、reverse dependencyまたはlegacy
ownershipの固定化につながるため設けない。`MinimalPolicy` / `TwoStepUkeirePolicy`等の
Policy class自体、`ValidationSession` / `RankedSession` / transport / trace writer /
Adapter、および`DEFAULT_VALIDATION_URL`(lower-level transport APIの一部)は削除して
いない。

## 今後のmigration

Issue #89後も、次はtemporaryにlisjongへ残る。

- Session
- transport
- protocol trace writer
- RiichiLab Adapter / possible-action validation

これらのphysical migrationは別Issueで段階的に行う。Arena側dependency pinの更新も
lisjong #86 / #89には含めず、cleanup後のSHAへ更新するArena側changeとして扱う。

Issue #86 / #89ではretry / reconnect / requeue / continuous participation、raw online
game record、Policy / DecisionContext / InternalAction変更、generic runtime抽象化を
行わない。