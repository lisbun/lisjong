# lisjong

Personal Japanese riichi mahjong AI for RiichiEnv and RiichiLab.

> [!IMPORTANT]
> lisjong is an independent personal Japanese mahjong AI project developed by
> [lisbun](https://github.com/lisbun). It is not affiliated with any other
> project using the LisJong or lisjong name.

## 概要

lisjongは、日本式立直麻雀AIを自作し、ローカル対局からオンライン対局まで
同じAI Policyを再利用できる形で開発・評価するプロジェクトです。

現在は初期開発段階です。最初の到達目標は、学習済みmodelの強さではなく、
決定的な最小PolicyをRiichiEnvとRiichiLabへ安全に接続し、半荘を完走する
ことです。ロードマップと完了条件は
[親Issue](https://github.com/lisbun/lisjong/issues/1)で管理します。

lisjong ecosystem全体のrepository責務、repository間依存方向、長期ロードマップは
[`lisjong-project`](https://github.com/lisbun/lisjong-project) を正本とします。
本repositoryでは `lisjong` 内部のPolicy、AI戦略、Adapter、integrationのarchitectureと実装を管理します。

## AI vision

lisjongは、不完全情報ゲームであるリーチ麻雀において、観測可能な情報から
hidden stateに対するbeliefを構築し、その不確実性と牌効率・打点・リスク等の
structural / value evaluationを組み合わせて意思決定へ活用する麻雀AIを目指します。

hidden-information inferenceの中では、**各他家のconcealed handに各牌種が何枚
存在するかの期待値を高精度に推定すること**を主要な研究・開発テーマとします。
将来的に既存手法と定量比較可能な評価基盤を整備したうえで、最高水準の推定能力を
目標とします。これは将来目標であり、現在の性能について未検証な達成済み主張は
行いません。

現在の`HandBelief`やconditional-uniform estimatorは、このvisionへ向けた
AI-side representation / baselineです。現在の具体的shape、fixed-point表現、
classやestimator方式をlong-term requirementとして固定せず、将来のheuristic / learned
inferenceや追加belief targetへ発展できる余地を残します。

推定精度そのものと麻雀AIとしての強さは別の評価対象として扱います。HandBeliefの
accuracy / calibration等のcomponent-specific evaluationは`lisjong`が扱い、Policyへ
統合した後のcontrolled performance comparisonは
[`lisjong-arena`](https://github.com/lisbun/lisjong-arena)へ接続します。
技術的な責務境界は[Architecture](docs/architecture.md)を参照してください。

## 位置づけ

| 対象 | 役割 |
| --- | --- |
| lisjong | 自作麻雀AI、Policy、学習・推論、接続Adapter、AI component評価 |
| [RiichiEnv](https://riichi.dev/docs/local-testing) | ローカル対局・開発・回帰評価環境 |
| [RiichiLab](https://riichi.dev/) | オンライン接続先 |
| Mortal | 比較対象・互換性確認用の外部AI |
| [python-study](https://github.com/lisbun/python-study) | 将来の接続先となる自作麻雀基盤を含む学習repository |

Mortalやpython-studyのコード・modelをlisjongの内部実装として取り込むことは
初期目標に含めません。

責務と依存方向の詳細は[Architecture](docs/architecture.md)を参照してください。

## 開発方針

- 初期実装は通常版CPython 3.14を基準とする
- RiichiEnv、RiichiLabなど外部環境の型・protocolをAI Policyから分離する
- 各プレイヤーから観測可能な情報だけを判断へ使用する
- Policyは合法手からactionを選択し、外部送信前にも合法手を検証する
- 再現可能なseed、version、評価条件を記録する
- AIの強さより先に、接続の正しさ、半荘完走、test可能性を確立する
- Rustは先行導入せず、profilingで必要性が確認された処理に限って検討する

## 開発環境

初期基準は通常版CPython 3.14です。[RiichiLabの公式Local Testing要件](https://riichi.dev/docs/local-testing)
であるPython 3.12以上を満たし、[RiichiEnvの配布package](https://pypi.org/project/riichienv/#files)
に含まれるCPython 3.14向けwheelを利用します。
free-threaded build（3.14t）は、依存libraryを含む互換性を個別に検証するまで
対象外とします。

repositoryを取得後、次のコマンドで開発環境を準備します。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS / Linuxでは、activateコマンドを次のように読み替えます。

```bash
source .venv/bin/activate
```

### 品質確認

ローカルとCIで同じコマンドを使用します。

```text
python -m ruff format --check .
python -m ruff check .
python -m unittest discover -s tests -v
```

testではPolicy契約、RiichiEnv Adapter、共通Policy実行境界、Local game runnerを
単体確認し、RiichiEnv 0.4.8を使う固定seed半荘のintegration testも実行します。

## Local game runner

`LocalGameRunner`は4 seatそれぞれのPolicyをRiichiEnvへ接続し、`env.done()`まで
1半荘を進行します。再現性のためseedは`RiichiEnv`のconstructorへ渡します。

```python
from lisjong.game_trace import GameTraceRecorder
from lisjong.local_game_runner import LocalGameRunner
from lisjong.policies import MinimalPolicy
from lisjong.policy_contract import Seat

policies = {seat: MinimalPolicy() for seat in Seat}
recorder = GameTraceRecorder()
result = LocalGameRunner(
    policies,
    seed=12345,
    game_mode="4p-red-half",
    max_steps=10_000,
    trace_sink=recorder,
).run()
trace = recorder.snapshot()

print(result.scores, result.ranks)
print(len(trace.events), trace.events[-1].event)
```

返却される`LocalGameResult`にはseed、game mode、最終scores / ranks、step数、
Policy判断数が含まれます。`max_steps`はhang防止用の安全上限であり、対局終了前に
到達した場合は正常結果を返さず`StepLimitExceededError`で失敗します。

`trace_sink`はopt-inです。標準`GameTraceRecorder`は、RiichiEnvが生成したMJAI
eventを対局中に0-basedの連続順で受け取り、正常な`LocalGameResult`の構築後だけ
immutableなcompleted `GameTrace`を返します。各`GameTraceEvent.event`はruntimeの
mutable `dict`から切り離したMJAI JSON文字列です。途中失敗時はpartial traceを公開せず、
sinkの例外も無視せず対局失敗として伝播します。GameTraceはprivileged observer outputで
あり、Policy inputにはなりません。

## RiichiLab bot実行profile

RiichiLab botのprofile定義・credential解決・runtime output設定は、Issue #44で
`lisjong.riichilab_client.profile` / `cli`へ実装しました。現在はlisjongの
validation CLIと、`lisjong-arena`が所有するranked first-party CLIの双方から
この設定層をtemporaryに再利用します。少なくとも次の3 profileを提供します。

| profile | 用途 | credential環境変数 | Policy |
| --- | --- | --- | --- |
| `lisjong-dev` | 開発・smoke test・protocol調査用 | `LISJONG_DEV_BOT_TOKEN` | `TwoStepUkeirePolicy` |
| `lisjong-baseline` | Policy性能比較の決定的な基準 | `LISJONG_BASELINE_BOT_TOKEN` | `MinimalPolicy` |
| `lisjong` | 本番運用(十分に検証済みのPolicyのみ) | `LISJONG_BOT_TOKEN` | `MinimalPolicy` |

`lisjong.policies`は、守備なし比較基準の`TwoStepUkeirePolicy`に加え、
非聴牌かつ他家リーチ時に全リーチ者への共通現物を優先する
`GenbutsuDefenseTwoStepUkeirePolicy`を公開する。後者は比較用の独立したPolicy世代で
あり、上記runtime profileへは自動的に割り当てない。

各profileは自分専用のcredential環境変数だけを参照し、他profileの
credentialやPolicyへ暗黙fallbackしません。profile設計の詳細
(責務境界、fail closed、runtime output/trace保存先、independence)は
[RiichiLab WebSocket Client](docs/riichilab-client.md)の「profile(Issue #44)」を
参照してください。

## RiichiLab validation

`run_validation(policy, token)`は、RiichiLab `/ws/validate`へBearer token付き
WebSocket接続し、1 validation gameを完走して`ValidationResult`(`passed`を含む)
を返します。Policy判断・Observation変換・`possible_actions` validationは
`RiichiLabSeatAdapter`(#38)を再利用し、WebSocket transport lifecycle
(`start_game` / `request_id` / `action_ack` / `end_game` / `validation_result`)
だけをこのpackageが担当します。責務境界と設計判断の詳細は
[RiichiLab WebSocket Client](docs/riichilab-client.md)を参照してください。

profile経由のlive validationは、次のコマンドで学習者環境から実行します。

```powershell
$env:LISJONG_DEV_BOT_TOKEN = "<dev検証用RiichiLab bot token>"
python -m lisjong.riichilab_client.validation --profile lisjong-dev
```

credential環境変数はrepositoryへcommitせず、実行時に注入してください。

## RiichiLab ranked smoke test

RiichiLab ranked one-game orchestrationのcanonical implementationとfirst-party CLIは、
`lisjong-arena` Issue #17でArenaへ移管しました。`lisjong` Issue #86ではlegacyな
`lisjong.riichilab_client.RankedGameResult` / `run_ranked_game()`と
`python -m lisjong.riichilab_client.ranked`を削除します。compatibility re-exportや
`lisjong -> lisjong-arena`のreverse dependencyは設けません。

現在の公開境界は次です。

```text
lisjong-arena
    lisjong_arena.riichilab.ranked.RankedGameResult
    lisjong_arena.riichilab.ranked.run_ranked_game()
    first-party ranked CLI

lisjong
    RankedSession
    connect_ranked_transport() / drive_ranked_session()
    protocol trace
    profile / credential helpers
    RiichiLab Adapter
```

lower-level runtimeはまだlisjongに物理的に存在し、Arenaがそのpublic APIをtemporaryに
利用します。ranked 1半荘の実行は次のArena entry pointから行います。

```powershell
$env:LISJONG_DEV_BOT_TOKEN = "<検証用RiichiLab bot token>"
python -m lisjong_arena.riichilab.ranked --profile lisjong-dev
```

本命bot `lisjong`ではなく`lisjong-dev` / `lisjong-baseline`profileの検証botを
使用し、原則1半荘だけ実行します。順位・score・ratingは成功条件ではありません。
tokenはstdout/stderr、結果、test、docs、Issue / PRへ保存しません。

profileごとのcredential source・Policy selection・runtime namespaceの独立性は
`tests/test_riichilab_client_profile.py`で直接検証します。ranked process orchestration
自体のtestはcanonical ownerである`lisjong-arena`側が担当します。

## RiichiLab protocol trace / runtime output

RiichiLab validation/ranked sessionの送受信protocol eventは、opt-inで
secret-safeなJSON Lines(JSONL)として保存できます(Issue #45)。既定は
無効(trace file非生成)です。

- validationは`run_validation(..., trace_path=...)`、rankedはArenaの
  `run_ranked_game(..., trace_path=...)`からopt-inする。ranked側もlisjongに残る
  `JsonlProtocolTraceWriter`と`drive_ranked_session()`をtemporaryに再利用する
- CLIの`RIICHILAB_TRACE_PATH` / `--trace` / `--trace-path`解決規則は既存helperを
  共用し、Arena ranked CLIも同じprofile / trace-path semanticsを利用する
- profile既定pathはOSユーザーローカル領域配下
  (Windowsは`%LOCALAPPDATA%\lisjong\...`等、repository配下は使わない)の
  profile別runtime namespace(`.../traces/<profile>/...`)へ、timestamp + UUID4で
  衝突しないfilenameを自動生成する
- BOT token、Authorization header、環境変数の値はtrace・runtime summary・
  filename/directory名のいずれにも含めない

詳細は[RiichiLab WebSocket Client](docs/riichilab-client.md)の
「protocol trace(Issue #45)」「profile(Issue #44)」を参照してください。

## ロードマップ

1. Python package、test、CI、文書の初期整備
2. RiichiEnvの実API・依存条件の調査
3. 共通Policy境界の設計
4. 学習modelを使わない最小Policyの実装
5. RiichiEnvで最初の局終了・半荘完走
6. RiichiLab validation・ランク戦1半荘
7. 接続MVP完了後に学習・評価を開始

Issue単位の現在地は
[GitHub Issues](https://github.com/lisbun/lisjong/issues)を正本とします。

## データ・model・秘密情報

このrepositoryには、次をcommitまたは再配布しません。

- RiichiLabのBot token、API key、その他の秘密情報
- 利用条件を確認していない牌譜・学習データ
- Mortalなど外部プロジェクトのmodel weight
- 大容量の生成model、raw data、実験artifact

外部データやmodelを利用する場合は、提供元、license、version、取得方法、
hashなどを確認し、repository本体とは分離して管理します。

## 開発状況

Python 3.14、Ruff、GitHub Actions CIを開発基盤とし、共通Policy契約、最小Policy、
RiichiEnv Adapter、共通Policy実行境界、Local game runner、RiichiLab
`request_action` Adapter、RiichiLab `/ws/validate` WebSocket Clientまで実装し、
validationを完走しています。RiichiLab rankedのone-game orchestration / first-party
CLIは`lisjong-arena`へcanonical移管済みで、lisjongにはArenaがtemporaryに利用する
`RankedSession` / transport / protocol trace / profile helpers / Adapterが残っています。
`lisjong-dev` / `lisjong-baseline` / `lisjong`のbot実行profileによりcredential・Policy・
runtime outputを分離しています。AI-sideではremaining tile inventoryと
conditional-uniform `HandBelief` baselineまで実装済みで、observation-aware heuristic /
learned estimatorや学習済みmodelは未導入です。

## License

lisjong自身のsource codeは[MIT License](LICENSE)で公開します。外部library、
model、牌譜、学習データには、それぞれの提供元のlicenseと利用条件が適用されます。
