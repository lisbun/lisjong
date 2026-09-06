# lisjong

Personal Japanese riichi mahjong AI project.

> [!IMPORTANT]
> lisjong is an independent personal Japanese mahjong AI project developed by
> [lisbun](https://github.com/lisbun). It is not affiliated with any other
> project using the LisJong or lisjong name.

## 概要

`lisjong` は、日本式立直麻雀AIの **decision core** を開発するrepositoryです。

初期段階で重視していた、

```text
Policy contract
    ↓
RiichiEnv / RiichiLabへ安全に接続
    ↓
局・半荘を再現可能に完走
```

という接続・実行基盤は、現在はecosystem内で独立した責務へ整理されています。
RiichiEnv / RiichiLabへのexternal execution / observationは
[`lisjong-arena`](https://github.com/lisbun/lisjong-arena) がcanonical ownerです。

現在の中心は、**再現可能な研究ループの中でPolicyを実際に強くするresearch phase**です。

主な研究方向は次の2本柱です。

```text
A. General Policy Strength
   Learned Policy / structural efficiency / value / defense / long-horizon utility

B. Explicit HandBelief Research
   hidden informationを明示的に推定し、decision valueへ接続する
```

総合strengthのNorth Starはfixed-protocolの半荘performanceとしつつ、
日常のcandidate iterationでは、より安いdiagnostic / round-level evidenceを先に使って
明らかな退化を早期にrejectします。

```text
cheap diagnostic
    ↓
low-complexity opponent evaluation
    ↓
interactive single-round evaluation
    ↓
hanchan evaluation / North Star
```

`cheap metric` は最終optimization targetではなく、reject / triage / prioritizationのための
signalとして扱います。

lisjong ecosystem全体のrepository責務、依存方向、長期ロードマップは
[`lisjong-project`](https://github.com/lisbun/lisjong-project) を正本とします。

- [Project architecture](https://github.com/lisbun/lisjong-project/blob/main/docs/architecture.md)
- [Project roadmap](https://github.com/lisbun/lisjong-project/blob/main/docs/roadmap.md)
- [Policy current status](docs/policy-status.md)
- [lisjong architecture](docs/architecture.md)

## Current research phase

### Learned Policy

Learned Policyは、hand-crafted Policyとは別世界のruntimeを作らず、既存の
Policy contract / action semantics / execution / evaluationを再利用して研究します。

現在までに、model-facing action vocabulary、player-safe feature representation、
dataset / training / servingのbounded vertical slice、Behavior Cloning、Offline Q系の
controlled experimentまで成立しています。

一方、technical servingの成立とgame strengthは別問題として扱います。
Behavior Cloningやsimple Offline Qの実験では、teacher agreementや学習の成立だけでは
strengthを保証できないことが確認されています。特にOffline Qのfailure diagnosisでは、
Qによるaction selectionがhand progressionを悪化させるmechanism-level evidenceが得られました。

そのため、現在の研究方針は次です。

```text
strong existing AI / literature
        ↓
mechanism-level design principle
        ↓
existing lisjong evidence
        ↓
one-axis bounded experiment
        ↓
cheap diagnostic first
        ↓
higher-fidelity strength evaluation only when warranted
```

強い既存AIで使われているarchitectureやalgorithmをそのままcanonical designとはせず、
`representation`、`objective`、`data scale`、`offline support`、`auxiliary supervision`等へ
分解し、原則1軸ずつlisjong自身のevidenceで検証します。

初期Learned Policyでは、最初から麻雀の全能力を同時に獲得できることを前提にせず、

```text
hand progression
    ↓
tenpai
    ↓
riichi / win
    ↓
calling / hand value
    ↓
defense / push-fold
    ↓
opponent belief / placement / long-horizon utility
```

のように基礎能力を診断できるcurriculumも重視します。

### HandBelief

lisjongは、不完全情報ゲームであるリーチ麻雀において、観測可能な情報から
hidden stateに対するbeliefを構築し、その不確実性を意思決定へ利用することを
独自の重要な研究軸とします。

特に、各他家のconcealed handに各牌種が何枚存在するかの期待値を推定する
`HandBelief` を中心に、heuristic / learned estimator、history-aware representation、
data scale、decision-value integrationを段階的に検証します。

ただし、次の3つは別のquality claimです。

```text
belief prediction quality
        !=
Policy decision quality
        !=
overall game strength
```

estimatorのaccuracy / calibrationが改善しても、それだけでPolicy strength improvementとは
みなしません。Policyへ利用する段階では、controlled Arena evaluationでdecision valueと
実際のgame performanceを別途確認します。

## Repository responsibilities

| Repository | 主な責務 |
| --- | --- |
| `lisjong` | AI decision core、Policy / AI-side contract、action semantics、牌効率・belief・value / risk等のstable AI semantics |
| [`lisjong-engine`](https://github.com/lisbun/lisjong-engine) | 日本式立直麻雀のルール、状態遷移、合法手、round / hanchan progression |
| [`lisjong-arena`](https://github.com/lisbun/lisjong-arena) | external / local execution、objective observation、experiment-local dataset / training / analysis、Policy strength evaluation |
| [`lisjong-play`](https://github.com/lisbun/lisjong-play) | Human Play / GUI / presentation consumer |

`lisjong` がstableなAI-side semanticsを所有し、Arena内のbounded experimentが
一時的なfeature layout、dataset schema、training harness、model artifactを所有する場合があります。
experiment-localな成功を、そのままlisjongのproduction / stable contractへ昇格させません。

## AI-side contract

lisjongの中心的な境界は、観測可能な情報から合法Actionを選択することです。

```text
DecisionContext
      |
      v
    Policy
      |
      v
InternalAction
```

- 外部environment固有型をPolicy contractへ漏らさない
- seat-visibleな情報だけをdecisionへ利用する
- Policyはlegal actionから選択する
- semantic identityと外部表現上のindexを分離する
- privileged observer dataをruntime Policy inputへ逆流させない

詳細は [Architecture](docs/architecture.md) と
[Policy contract](docs/policy-contract.md) を参照してください。

## Model-facing action vocabulary

`lisjong.action_vocabulary` は、Learned Policyが固定長のmodel outputを扱うための
versioned action vocabularyとlegal mask contractを提供します。

```text
semantic identity
    = InternalAction dataclass value equality

model action index
    = versioned adapter representation
```

model action indexは麻雀上の新しいAction identityでも、合法性の根拠でもありません。
合法性のauthorityは常にcurrent decisionのlegal actionsです。

詳細は [Model-facing action vocabulary](docs/action-vocabulary.md) を参照してください。

## Policy status

公開Policyのcurrent role、current strength baseline、training coverage source等の
**現在の解釈**は [Policy current status](docs/policy-status.md) を正本とします。

historicalな評価値やpromotion / rejectionの全履歴をREADMEへ複製しません。
Policy strength comparisonのmeasurement source of truthは`lisjong-arena`側の
artifact / evaluation policyです。

## Development principles

- correctness / information boundary / reproducibilityをstrength claimより先に守る
- strong-AI referenceを採用理由ではなくresearch evidenceとして扱う
- 原則 one bounded experiment = one primary research question
- negative / inconclusive / invalid resultも次の判断に使えるevidenceとして保持する
- cheap proxyを最終目的化しない
- fixed-protocol hanchan performanceを総合strengthのNorth Starとして維持する
- resultを見てseedやthresholdを都合よく追加しない
- large model / large data / native backend / cloud scaleはmeasured need後に導入する
- 高頻度のsmall loopはLLMに依存させず、LLM / coding agentは低頻度のresearch decisionに利用できる形を優先する
- Rustは先行導入せず、profilingで必要性が確認された処理に限って検討する

## Development environment

初期基準は通常版CPython 3.14です。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS / Linuxでは:

```bash
source .venv/bin/activate
```

品質確認:

```text
python -m ruff format --check .
python -m ruff check .
python -m unittest discover -s tests -v
```

`lisjong` 自体はRiichiEnv / RiichiLab execution runtimeを所有しません。
external/local integration testや実対局の実行方法は
[`lisjong-arena`](https://github.com/lisbun/lisjong-arena) を参照してください。

## Data, models, and secrets

このrepositoryには、次をcommitまたは再配布しません。

- RiichiLabのBot token、API key、その他の秘密情報
- 利用条件を確認していない牌譜・学習データ
- Mortalなど外部プロジェクトのmodel weight
- 大容量の生成model、raw data、実験artifact

外部データやmodelを利用する場合は、提供元、license、version、取得方法、hash等を確認し、
repository本体とは分離して管理します。

## Roadmap and work tracking

project-wideな長期方向は
[`lisjong-project/docs/roadmap.md`](https://github.com/lisbun/lisjong-project/blob/main/docs/roadmap.md)
を正本とします。

個別の実験、実装、acceptance criteria、current next actionは各repositoryのGitHub Issues / PRsを
正本とし、READMEを短期進捗表として使いません。

## License

lisjong自身のsource codeは[MIT License](LICENSE)で公開します。外部library、model、牌譜、
学習データには、それぞれの提供元のlicenseと利用条件が適用されます。
