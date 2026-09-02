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
本repositoryでは、環境非依存のAI decision coreとして、Policy、AI-side contract、牌効率・belief・value / risk等のAI semanticsと実装を管理します。RiichiEnv / RiichiLabへのexternal execution / observationは`lisjong-arena`がcanonical + physical ownerです。

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
公開Policyのcurrent roleとcurrent strength baselineは[Policy current status](docs/policy-status.md)を参照してください。

## 位置づけ

| 対象 | 役割 |
| --- | --- |
| lisjong | 自作麻雀AI decision core、Policy、学習・推論、AI component評価 |
| [lisjong-arena](https://github.com/lisbun/lisjong-arena) | RiichiEnv / RiichiLab integration、execution / observation、Policy比較・評価 |
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
であるPython 3.12以上を満たします。lisjong自体は`riichienv`へdirect dependencyを
持たないため(`lisbun/lisjong#100`)、`pip install -e ".[dev]"`は[RiichiEnvの配布
package](https://pypi.org/project/riichienv/#files)をinstallしません。RiichiEnv
integrationの開発・実行は`lisjong-arena`側の開発環境で行います。
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

testではPolicy契約を単体確認します。lisjongはRiichiEnvへdirect dependencyを
持たず(`lisbun/lisjong#100`)、RiichiEnv Adapterと、RiichiEnv 0.4.8を使う固定seed
半荘のintegration testはいずれもArena側(`lisjong-arena`)にあります。

ADR 0002に基づくexternal execution / observationのphysical migrationは完了しており、
Arenaのcurrent exact lisjong pinは
`376f69088a134b5a9bcc33a69b95e3f779eb2b0e`です。lisjongはexternal execution用の
runtime dependencyを持たず、AI decision coreとして成立します。

## Local game runner

`LocalGameRunner` / `LocalGameResult`のcanonical implementationとcanonical
physical implementationは`lisjong-arena` Issue #31 / PR #32で
`lisjong_arena.riichienv.local_game_runner`へ移管しました。`lisjong` Issue #98で
lisjong側legacy `src/lisjong/local_game_runner.py`と、そのrunner-owned /
Policy-specific integration testsを削除済みです。compatibility re-exportや
`lisjong -> lisjong-arena`のreverse dependencyは設けていません。

```text
lisjong-arena
    lisjong_arena.riichienv.local_game_runner.LocalGameRunner
    lisjong_arena.riichienv.local_game_runner.LocalGameResult
    lisjong_arena.riichienv.adapter (RiichiEnv Adapter, canonical + physical)
    lisjong_arena.game_trace (GameTrace, canonical + physical)
```

RiichiEnv Adapterのcanonical physical implementationはArena側
`lisjong_arena.riichienv.adapter`です(`lisjong-arena` Issue #39 / PR #40)。
lisjong main側legacy `lisjong.riichienv_adapter`とそのAdapter-owned testsは
`lisbun/lisjong#100` / PR #101で削除し、`riichienv`をlisjongのruntime dependencyから
完全に除去しました。その後Arena Issue #41 / PR #42でexact lisjong dependency pinを
cleanup merge commit `3505321b62e7a2be204cc555924b485a898c8f31`へ同期したため、
RiichiEnv Adapter pillarのphysical duplicateは完全解消済みです。compatibility
re-exportや`lisjong -> lisjong-arena`のreverse dependencyは設けていません。

`GameTraceRecorder` / `GameTraceSink` / `GameTrace`のcanonical ownerとphysical
implementationは、`lisjong-arena` Issue #43 / PR #44でArena側
`lisjong_arena.game_trace`へ移管済みです。lisjong側legacy `lisjong.game_trace`と
そのowned testはIssue #102で削除し、compatibility re-exportや
`lisjong -> lisjong-arena`のreverse dependencyは設けていません。
標準`GameTraceRecorder`は、RiichiEnvが生成したMJAI eventを対局中に0-basedの
連続順で受け取り、正常な対局結果の構築後だけimmutableなcompleted `GameTrace`を
返します。各`GameTraceEvent.event`はruntimeのmutable `dict`から切り離したMJAI
JSON文字列です。途中失敗時はpartial traceを公開せず、sinkの例外も無視せず対局
失敗として伝播します。GameTraceはprivileged observer outputであり、Policy input
にはなりません。

lisjong Issue #102 / PR #103でlegacy GameTraceを削除した後、Arena Issue #45 /
PR #46でexact lisjong dependency pinをcleanup merge commit
`376f69088a134b5a9bcc33a69b95e3f779eb2b0e`へ同期しました。fresh installed dependencyでも
`lisjong.game_trace`が存在せず、`lisjong_arena.game_trace`がimport可能であることを確認済みです。
これにより`lisjong_arena.game_trace`がcanonicalかつsole physical implementationとなり、
GameTrace pillarのphysical duplicateは完全解消済みです。

## RiichiLab execution profile / credential / CLI composition

RiichiLab botのprofile定義・credential解決・common CLI引数解析・trace path解決は、
Issue #44/#45でlisjong側`lisjong.riichilab_client.profile` / `cli`として実装しましたが、
`lisjong-arena` Issue #19でcanonical implementationをArenaへ移管し、`lisbun/lisjong#89`で
lisjong側のlegacy実装を削除しました。この設定層は現在`lisjong_arena.riichilab.profile` /
`lisjong_arena.riichilab.cli`がcanonical + physical ownerであり、ranked / validation CLIの
双方がここからprofile・credential・trace pathを解決します。

lisjongが引き続き所有するのは、profile mappingが参照するPolicy class自体
(`MinimalPolicy` / `TwoStepUkeirePolicy`等)です。Arena側profile mappingは次の3 profileを
提供します(mapping自体の正本はArena側)。

| profile | 用途 | credential環境変数 | Policy |
| --- | --- | --- | --- |
| `lisjong-dev` | 開発・smoke test・protocol調査用 | `LISJONG_DEV_BOT_TOKEN` | `TwoStepUkeirePolicy` |
| `lisjong-baseline` | Policy性能比較の決定的な基準 | `LISJONG_BASELINE_BOT_TOKEN` | `MinimalPolicy` |
| `lisjong` | 本番運用(十分に検証済みのPolicyのみ) | `LISJONG_BOT_TOKEN` | `MinimalPolicy` |

`lisjong.policies`は、守備なし比較基準の`TwoStepUkeirePolicy`に加え、
非聴牌かつ他家リーチ時に全リーチ者への共通現物を優先する
`GenbutsuDefenseTwoStepUkeirePolicy`を公開する。後者は比較用の独立したPolicy世代で
あり、上記runtime profileへは自動的に割り当てない。

`lisjong.policies`はさらに、`TwoStepUkeirePolicy`のselection semanticsを変更せず
offense baselineとして維持したまま、打点価値の最小世代を打牌比較へ追加する
`ValueAwareTwoStepUkeirePolicy`を公開する。

```text
TwoStepUkeirePolicy:
shanten > current ukeire > second-step ukeire > stable tie-break

ValueAwareTwoStepUkeirePolicy:
shanten > current ukeire > retained concealed dora count > second-step ukeire > stable tie-break
```

`retained_concealed_dora_count`は、打牌候補ごとに打牌後concealed handへ残る
「公開済みdora indicator由来のdora count + 赤ドラcount」だけを数える
candidate-dependent featureであり、`actual han` / `total hand han` /
`expected score` / `expected value`のいずれでもない。`PolicyInput.round.dora_indicators`
（PolicyInput上すでに公開済みのindicatorのみ。未公開槓ドラ・裏ドラは含まない）
だけを使う。こちらも比較用の独立したPolicy世代であり、上記runtime profileへは
自動的に割り当てない。詳細は[Architecture](docs/architecture.md)を参照。

Issue #125のexperimental generationとして、`lisjong.policies`は
`HandValueAwareTwoStepUkeirePolicy`も公開する。

```text
HandValueAwareTwoStepUkeirePolicy:
shanten
> current ukeire
> retained real value
> yaku route value
> second-step ukeire
> stable tie-break
```

`retained_real_value`は、post-discard自手に残る公開indicator由来dora、赤ドラ、
完成済み役牌刻子・槓の翻相当値の和である。`yaku_route_value`はtanyao / honitsu /
chinitsu compatibilityだけを表すlightweight heuristicであり、actual han、
expected han、expected score、expected valueではない。value-awareなのは現在の
real discard比較だけで、第2段のhypothetical branchは既存TwoStepのstructural
semanticsを維持する。このPolicyもruntime profileへは自動的に割り当てない。
詳細は[Architecture](docs/architecture.md)を参照。

`lisjong.policies`はさらに、Policy-visibleなremaining uncertaintyに対する
exact finite-horizon dynamic programmingで打牌を選ぶ
`FiniteHorizonCompletionPolicy`を公開する。

```text
TwoStepUkeirePolicy:
heuristic two-step structural efficiency

FiniteHorizonCompletionPolicy:
exact conditional k-self-draw structural completion probability
completion mass > (tie / all-zeroのときだけ) existing TwoStep ranking
```

初期世代はhorizon 3固定で、「今後3個のself-draw slotsが存在すると条件付けた
conditional-uniform / exchangeable model上で、3回以内にstructural completionへ
到達する確率」をexact integer massとして比較する。向聴数はcompletion massより
上位のhard filterにしない。

次の2点はこのPolicyのsemanticとして明示的な非同一である。

```text
remaining inventory != live wall
completion probability != actual probability of winning within 3 turns
```

remaining tile inventoryは他家concealed hand・live wall・dead wall・未開示裏ドラ
表示牌等をまとめた残余inventoryであり、山ではない。completion probabilityも、
流局・他家和了・実際に残るツモ回数・future riichi / call等を含まない
structural hand-development valueである。こちらも比較用の独立したPolicy世代で
あり、上記runtime profileへは自動的に割り当てない。詳細は
[Architecture](docs/architecture.md)を参照。

Issue #122のexperimental generationとして、`lisjong.policies`は
`GenbutsuDefenseFiniteHorizonValueAwarePolicy`も公開する。このPolicyは既存4 Policyを
変更せず、legal discardへ次のpriorityを適用する。

```text
Genbutsu safety constraint
> FiniteHorizon completion mass
> ValueAware ranking
```

FiniteHorizonがunique positive maximumを持つ場合はその候補を即採用し、positive
maximum tieではmaximum-mass subsetだけ、all-zeroではGenbutsu適用後のeligible set
全部をValueAwareへ渡す。初期実装のcombined-specific analysisは`None`であり、この
Policyもruntime profileやArena policy catalogへ自動登録しない。

各profileは自分専用のcredential環境変数だけを参照し、他profileの
credentialやPolicyへ暗黙fallbackしません。profile / credential compositionの詳細は
`lisjong-arena`側の文書を参照してください。

## Model-facing action vocabulary

`lisjong.action_vocabulary`は、learned Policyが固定長のaction出力を扱えるように、
fixed-sizeかつversionedなmodel-facing action vocabulary、`InternalAction`との
codec、`DecisionContext.legal_actions`から導出するfixed-size legal maskを提供
します(Issue #149)。ML runtime(NumPy / PyTorch等)へは依存せず、maskは純粋な
Python contract(`tuple[bool, ...]`)として表現します。

```text
semantic identity
    = InternalAction dataclass value equality

model action index
    = versioned adapter representation
```

model action indexは新しいAction identityではありません。麻雀上の合法性の根拠でも、
`legal_actions`のtuple indexでもありません。`resolve_legal_action()`は同じ
decisionの`legal_actions`側のcanonical `InternalAction`を返し、
`execute_policy()`のsignature・validation・例外semanticsは変更しません。

```python
from lisjong.action_vocabulary import (
    ACTION_VOCABULARY_SIZE,  # 802
    ACTION_VOCABULARY_VERSION,  # "lisjong-action-vocabulary-1"
    build_legal_action_mask,
    encode_action,
    resolve_legal_action,
)

mask = build_legal_action_mask(decision)  # len(mask) == ACTION_VOCABULARY_SIZE
index = ...  # modelがmask上で選んだindex
action = resolve_legal_action(index, decision)  # canonical legal InternalAction

encode_action(action) == index  # 同じdecisionでround-tripする
```

encodeできないAction、mask上illegalなindex、範囲外のindex、同一decision内の
index衝突、未対応のvocabulary versionはいずれもfail closedとし、fallback Action
へ置換しません。index layout、encoding規則、version更新規則は
[Model-facing action vocabulary](docs/action-vocabulary.md)を正本とします。
feature encoder、tensor schema、HandBelief consumer seam、model architecture、
trainingは後続Issueで扱います。

## RiichiLab ranked / validation execution

RiichiLab ranked one-game orchestrationのcanonical implementationとfirst-party CLIは
`lisjong-arena` Issue #17で、validation one-game orchestration・CLI・
execution profile / credential / common CLI compositionのcanonical implementationは
`lisjong-arena` Issue #19でそれぞれArenaへ移管しました。`lisjong` Issue #86 / #89で
lisjong側legacyな`RankedGameResult` / `run_ranked_game()` / `ValidationResult` /
`run_validation()`とそれぞれのCLI(`python -m lisjong.riichilab_client.ranked` /
`python -m lisjong.riichilab_client.validation`)、および`profile.py` / `cli.py`を
削除済みです。compatibility re-exportや`lisjong -> lisjong-arena`のreverse dependency
は設けていません。

`lisjong-arena` Issue #23 / PR #24で、client errors、Session、Transport、protocol
trace writerもArena-local implementationへcanonical + physical migrationしました。
そのtakeoverを確認したうえで、`lisjong` Issue #91ではlegacy
`lisjong.riichilab_client` package全体を削除しています。

現在の公開境界は次です。

```text
lisjong-arena
    lisjong_arena.riichilab.ranked.RankedGameResult
    lisjong_arena.riichilab.ranked.run_ranked_game()
    lisjong_arena.riichilab.validation.ValidationResult
    lisjong_arena.riichilab.validation.run_validation()
    first-party ranked / validation CLI
    execution profile / credential / common CLI composition
    client errors / Session / Transport / protocol trace writer
    RiichiLabSeatAdapter / SendReadyResponse
    request_action parsing / MJAI response conversion
    possible_actions semantic validation / Adapter-specific errors

lisjong
    Policy / DecisionContext / InternalAction / semantic contracts
```

RiichiEnv Adapter(`build_decision()` / `SeatMaterializedState` /
`RiichiEnvActionMappingSession` / `RiichiEnvActionMapping` / `build_policy_input()`)
はcanonical + physical implementationとして`lisjong_arena.riichienv.adapter`に
あります(`lisjong-arena` Issue #39 / PR #40)。lisjong側は`lisbun/lisjong#100`で
legacy実装を削除し、`riichienv`へのdirect dependencyを持ちません。

`RiichiLabSeatAdapter` / request_action parsing / MJAI response conversion /
possible_actions semantic validation / Adapter-specific errorsは、`lisjong-arena`
Issue #27 / PR #28でArena-local implementation(`lisjong_arena.riichilab.adapter`)へ
canonical + physical migrationし、`lisjong` Issue #94でlisjong側legacy
`src/lisjong/riichilab_adapter/`を削除しています。compatibility re-exportや
`lisjong -> lisjong-arena`のreverse dependencyは設けていません。

Issue #91 / #94それぞれのcleanup merge直後はArenaのlisjong dependency pinがcleanup前revisionを
参照していましたが、その後Arena Issue #25 / #29で各cleanup merge SHAへexact pin syncを
完了しています。RiichiLab lower-level runtimeとprotocol-facing decision bridgeのlegacy
physical duplicateは現在いずれも解消済みです。

ranked 1半荘・validation 1 gameの実行は、いずれも次のArena entry pointから行います。

```powershell
$env:LISJONG_DEV_BOT_TOKEN = "<検証用RiichiLab bot token>"
python -m lisjong_arena.riichilab.ranked --profile lisjong-dev
python -m lisjong_arena.riichilab.validation --profile lisjong-dev
```

本命bot `lisjong`ではなく`lisjong-dev` / `lisjong-baseline`profileの検証botを
使用し、原則1半荘・1 gameだけ実行します。順位・score・ratingは成功条件ではありません。
tokenはstdout/stderr、結果、test、docs、Issue / PRへ保存しません。

profileごとのcredential source・Policy selection・runtime namespaceの独立性、
ranked / validation process orchestration、Session / Transport / trace / client error、
`RiichiLabSeatAdapter` / request_action parsing / MJAI response conversion /
possible_actions semantic validationのtestは、canonical ownerである`lisjong-arena`側が
担当します。lisjong側は`tests/test_policy_execution.py`等でPolicy contractの
regressionを保持します。

## RiichiLab protocol trace / runtime output

RiichiLab validation/ranked sessionの送受信protocol eventは、opt-inで
secret-safeなJSON Lines(JSONL)として保存できます(Issue #45)。既定は
無効(trace file非生成)です。

- Arena-local `run_validation(..., trace_path=...)` / `run_ranked_game(..., trace_path=...)`
  からopt-inする。writerとdrive処理はArena-local lower-level runtimeを利用する
- `RIICHILAB_TRACE_PATH` / `--trace` / `--trace-path`解決規則、profile既定pathは
  Arena側の`lisjong_arena.riichilab.cli` / `lisjong_arena.riichilab.profile`が
  canonical implementationとして所有する
- profile既定pathはOSユーザーローカル領域配下
  (Windowsは`%LOCALAPPDATA%\lisjong\...`等、repository配下は使わない)の
  profile別runtime namespace(`.../traces/<profile>/...`)へ、timestamp + UUID4で
  衝突しないfilenameを自動生成する
- BOT token、Authorization header、環境変数の値はtrace・runtime summary・
  filename/directory名のいずれにも含めない

trace writer実装(`JsonlProtocolTraceWriter` / `ProtocolTraceError`)とtrace schemaの
current contractは、Arena側
[RiichiLab client runtime contract](https://github.com/lisbun/lisjong-arena/blob/main/docs/riichilab-client.md)
を正本とします。

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
共通Policy実行境界まで実装しています。RiichiEnv Adapter / Local game runnerの
canonical + physical implementationはいずれも`lisjong-arena`にあります。
RiichiLab `/ws/validate` / `/ws/ranked` lower-level runtimeとrankedのone-game
orchestration / first-party CLIは`lisjong-arena` Issue #17で、validationの
one-game orchestration / first-party CLI / execution profile・credential・
common CLI compositionは同Issue #19でそれぞれcanonical移管済みです。client errors /
Session / Transport / protocol trace writerもArena Issue #23 / PR #24で
canonical + physical migrationし、lisjong Issue #91でlegacy copyを削除しました。
RiichiLab protocol-facing decision bridge(`RiichiLabSeatAdapter` / request_action
parsing / MJAI response conversion / possible_actions semantic validation)も
Arena Issue #27 / PR #28でcanonical + physical migrationし、lisjong Issue #94で
legacy `src/lisjong/riichilab_adapter/`を削除しました。lisjongに残るのはPolicy /
DecisionContext / InternalAction等のAI-side semanticsです。RiichiEnv Adapterは
Arena Issue #39 / PR #40でcanonical + physical migrationし、lisjong Issue #100で
legacy `src/lisjong/riichienv_adapter/`を削除し、Arena Issue #41でpost-cleanup
exact pin syncまで完了しました。GameTraceもArena Issue #43 / PR #44でcanonical +
physical migrationし、lisjong Issue #102 / PR #103でlegacy copyを削除、Arena Issue #45 /
PR #46でexact pinを`376f69088a134b5a9bcc33a69b95e3f779eb2b0e`へ同期済みです。
`lisjong-dev` / `lisjong-baseline` / `lisjong`のbot実行profile mappingはArena側が
所有し、mappingが参照するPolicy class自体はlisjongが引き続き所有します。AI-sideでは
remaining tile inventoryとconditional-uniform `HandBelief` baselineまで実装済みで、
observation-aware heuristic / learned estimatorや学習済みmodelは未導入です。
learned Policy向けには、Issue #149でfixed-sizeかつversionedなmodel-facing action
vocabulary、`InternalAction` codec、legal mask contract(`lisjong.action_vocabulary`)
まで実装済みです。feature encoder、model、trainingは未導入です。

## License

lisjong自身のsource codeは[MIT License](LICENSE)で公開します。外部library、
model、牌譜、学習データには、それぞれの提供元のlicenseと利用条件が適用されます。
