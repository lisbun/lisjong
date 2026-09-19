# Architecture

## 目的

`lisjong` は、日本式立直麻雀AIの **decision core** と、stable / productionなAI-side semanticsを所有するrepositoryである。

外部環境のprotocolや型をPolicyから分離し、各seatが判断時点で観測可能な情報だけをPolicyへ渡すことを最優先の境界とする。

lisjong ecosystem全体のrepository責務、repository間依存方向、experiment-local researchのownership、promotion boundaryは、
[`lisjong-project` のArchitecture](https://github.com/lisbun/lisjong-project/blob/main/docs/architecture.md)をproject-wideな正本とする。

本書は、その横断境界の内側にある`lisjong`固有の次を正本とする。

- Policy / Policy contract
- `PolicyInput` / `DecisionContext` / `InternalAction`等のAI-side semantic contract
- shanten / ukeire / HandBelief / risk / value / utility等のstable domain semantics
- Policy-internal analysis semantics
- production / public Learned Policyやlearned estimatorへ昇格した後のstable inference semantics
- hidden-information / player-safe information boundary

external / local execution、observation、bounded experiment-local dataset / training / analysis、Policy / game evaluationは、canonical ownerである[`lisjong-arena`](https://github.com/lisbun/lisjong-arena)側を正本とする。

現在のPolicy role / strength baselineは[Policy current status](policy-status.md)を正本とする。historicalなPolicy世代、migration、実験結果、current work statusはGitHub Issues / PRsまたはpurpose-specific documentへ委ね、本書へ進捗ログとして重複させない。

## Source of truth

```text
lisjong-project/docs/architecture.md
    project-wide repository responsibility
    dependency direction
    experiment-local research ownership
    promotion boundary

lisjong/docs/architecture.md
    stable AI-side architecture
    Policy / belief / value / action semantics
    hidden-information boundary

purpose-specific lisjong docs
    concrete stable contract

lisjong-arena
    execution / observation
    bounded experiment-local research / ML
    reproducible evaluation

GitHub Issues / PRs
    current work
    experiment protocol / result
    adoption / promotion decision
```

Policyの公開契約は[Policy契約](policy-contract.md)、Policy入力の許可fieldと意味契約は[Policy入力の最小スキーマ](policy-input-schema.md)、内部Actionのvariant / field / semanticsは[内部Actionモデル](internal-action-model.md)、Action semantic identityは[Action identity](action-identity.md)、model-facing action indexは[Model-facing action vocabulary](action-vocabulary.md)を正本とする。

---

# Core ownership boundary

短く言うと、次の境界を維持する。

```text
lisjong
    = what stable AI decisions / features / beliefs / values mean

lisjong-arena Execution / Observation
    = what happened

lisjong-arena Experiment-local Research / ML
    = how a bounded experiment materializes / trains / diagnoses evidence

lisjong-arena Evaluation
    = how candidates are compared reproducibly
```

重要なのは、ArenaでMLを実装できることと、stable AI semanticsをArenaが所有することは別だという点である。

```text
experiment-local feature schema
!= stable PolicyInput / production feature contract

experiment-local model
!= canonical production Learned Policy architecture

experiment-local checkpoint
!= production Policy

experiment result
!= stable public API
```

逆に、`lisjong`がstable semanticsを所有することは、すべてのdataset builder / trainer / model artifact / experiment harnessを`lisjong`へ置くことを意味しない。

bounded research questionに対するpurpose-specificなdataset / tensor / trainer / checkpoint / diagnostic / calibration studyは、Arenaのcontrolled evidence pipelineと強く結び付く場合、Arena-owned experiment-local implementationとして扱う。

---

# Long-term AI architecture

`lisjong` は、不完全情報ゲームである立直麻雀において、観測可能な情報からhidden stateに対するbeliefを構築し、その不確実性とstructural / value evaluationを組み合わせて意思決定へ利用するAIを長期的に目指す。

```text
observable information
        ↓
canonical player-safe state
        ↓
physical accounting
        ↓
hidden-state inference / belief
        │
        ├──────────────┐
        │              │
        ▼              ▼
structural        score / risk /
evaluation        value estimation
        │              │
        └──────┬───────┘
               ▼
        Policy / decision
```

この図は特定のmodel architectureやruntime call graphを固定するものではない。representation、inference、value、decisionを独立に改善できるstable semantic boundaryを示す。

特に`HandBelief`は、各他家のconcealed handに各牌種が何枚存在するかのexpected countや、red-five / structural wait等のmarginal beliefを表すAI-side conceptとして扱う。

現在の`HandBelief` / `ConcealedHandBelief`、canonical 34牌種axis、red-five companion、fixed-point representation、conditional-uniform baseline等はcurrent stable contract / baselineであり、将来のmodel familyを固定するものではない。

将来、history-aware learned estimator、joint representation、追加head、learned latent等へ進んでも、stable consumerがexperiment-specific tensor / framework objectへ直接依存しないことを優先する。

---

# Quality claimsを分離する

AI研究では少なくとも次の3つを別claimとして扱う。

```text
Inference / component quality
    prediction accuracy
    calibration
    physical consistency

Decision quality
    same decision contextでのAction quality
    ranking / local objective improvement

Game performance
    controlled round / hanchan result
```

これらを同一視しない。

```text
better prediction
!= better decision
!= stronger Mahjong AI
```

## Stable semantic correctness vs empirical measurement

ここでownershipをさらに分ける。

### `lisjong` が正本とするもの

- `HandBelief` fieldの意味
- structural wait / ron legality / risk / valueの区別
- canonical axis / units / availability semantics
- physical-validity / conservation semantics
- stable inference input / output contract
- Policy-visible feature / Action semantics
- production consumerが依存するsemantic correctness

### Arena experiment-local researchが所有できるもの

- learned estimator training corpus / dataset
- experiment-local feature / tensor schema
- bounded trainer / model architecture / loss / optimizer
- checkpoint / result artifact
- MAE / log loss / Brier score / calibration study
- scale study / distribution-shift measurement
- failure diagnosis
- experiment-specific classification rule

したがって、以前のように

```text
component accuracy / calibration = 必ずlisjongで測る
```

とはしない。

**semantic correctnessはlisjong、bounded empirical measurementはArenaに置ける**という境界を採用する。

測定結果がpositiveでも、それだけでstable contractへのpromotionは起きない。

---

# Experiment → stable contract promotion boundary

research implementationは次の順序で扱う。

```text
bounded hypothesis
    ↓
experiment-local implementation
    ↓
result / evidence
    ↓
negative / inconclusive
    -> historical experimentとして保持

repeatedly useful / adoption warranted
    ↓
owner review
    ↓
stable semanticsを定義
    ↓
必要ならlisjongへformalize / integrate
```

promotion時には最低限次を確認する。

- semanticsが特定experimentを超えてstableか
- production / multiple consumerで必要か
- stable public APIとしてversioningする価値があるか
- experiment-local schema / checkpoint identityをそのままstable contractへ流用してよいか
- model runtime / weights delivery / inference code ownerをどこに置くか
- `lisjong -> lisjong-arena` のreverse dependencyを作らず成立するか
- player-safe / hidden-information boundaryを維持できるか

「Arenaで学習できた」「TESTで改善した」だけではpromotion理由にしない。

---

# Policy contract

Policyは、1 seat・1 decision分のenvironment-independentな入力を受け、`InternalAction`を選ぶ。

概念上:

```text
PolicyInput
    +
legal InternalAction candidates
        ↓
DecisionContext
        ↓
Policy
        ↓
InternalAction
        ↓
execute_policy() validation
```

固定する原則:

- RiichiEnv / RiichiLab / mjai / WebSocket固有型をPolicy contractへ入れない
- `DecisionContext`は同じseat・同じdecision時点の整合したsnapshot
- `legal_actions`はsemantic identity上一意で、tuple順に意味を持たせない
- pass / none相当が合法なら明示Actionとして表す
- Policyは渡された合法候補からだけ選ぶ
- external environmentのlegal Action mappingはArena execution boundaryの責務
- Policy返却Actionは`execute_policy()`でcanonical legal candidateへ照合する
- validation failureでは未検証Actionを外部へ送らない
- Policy自身の例外は意味を変えず伝播する
- Policy選択に影響する未宣言mutable state / hidden PRNG stateを持たない

詳細は[Policy契約](policy-contract.md)を正本とする。

---

# Policy-visible information boundary

Policyへ渡してよい情報は、そのseatが判断時点で観測可能な情報に限定する。

許可される代表例:

- 自席concealed hand / visible drawn tile
- public discard / meld / riichi / dora indicator
- public score / round / dealer / honba等
- decision時点のlegal actions
- rulesetに由来する明示的configuration

禁止する代表例:

- 他家の実concealed hand
- live wall / dead wallの実配列
- future event / future outcome
- unmasked omniscient game log
- Arena observerだけが持つprivileged state
- training-only ground truth

```text
player-safe observation
        ↓
Policy input / online inference

omniscient / privileged truth
        ↓
training label / offline evaluation only
```

この2経路を混ぜない。

完全対局logやArena GameTraceがoffline analysisに利用可能でも、その情報を`PolicyInput`へ逆流させない。

Policy入力の具体field / canonicalization / invariantsは[Policy入力の最小スキーマ](policy-input-schema.md)を正本とする。

---

# AI-side stable components

## `policy_contract`

`lisjong.policy_contract`は環境非依存のAI-side root contractを所有する。

主なstable value / API:

- `Policy`
- `PolicyInput`
- `DecisionContext`
- `InternalAction` variants
- `execute_policy()`
- `execute_policy_with_trace()`
- `AnalysisTrace`
- `DecisionTrace`
- `PolicyDecision`

`policy_contract`はRiichiEnv / RiichiLab / mjai / WebSocket / PyTorch等へ依存しない。

## `hand_evaluation`

`lisjong.hand_evaluation`は、Policy-visibleな手牌からshanten等のstable structural semanticsを計算するAI-side domain componentである。

重要なのはbackendではなくsemantic contractである。

- general public APIはvalidated `Tile` inputを使う
- 和了形 = `-1`、聴牌 = `0`というnumeric shanten semanticsを維持する
- standard / 七対子 / 国士無双をcurrent contractに従って評価し、Policy moduleごとに別semanticを再実装しない
- package-internal optimized pathを追加しても新しいshanten semanticsを作らない
- lookup / native / cache等のbackend optimizationで結果semanticを変えない
- fail closedなinput validationを維持する
- count-native hot pathやpackage-internal predicateは一般public APIへ自動昇格させない

具体的backend / lookup artifactはimplementation detailとして扱う。

## `belief`

`lisjong.belief`は、hidden-information inferenceに必要なstable AI-side representation / physical-accounting semanticsを所有する。

代表的なstable concept:

- canonical 34 tile-type axis
- red-five companion axis
- `HandBelief`
- `ConcealedHandBelief`
- public tile provenance
- remaining tile inventory
- tile conservation
- conditional-uniform baseline estimator
- structural wait belief
- exact wait ground-truth builder for offline validation

current representationでは、player axisはWind semanticsを明示的に扱い、expected-count / probabilityはcanonical fixed-point domainを使う。storage表現をconsumer側semanticへ漏らさず、semantic accessorを優先する。

重要なsemantic boundary:

```text
remaining tile inventory
!= live wall

HandBelief marginal set
!= full joint posterior

structural wait
!= ron-legal wait
!= deal-in probability
!= hand EV
!= game EV
```

`remaining tile inventory`は、exact accounted public / self informationをstandard physical inventoryから差し引いた残余であり、他家concealed / live wall / dead wall等のlocation posteriorを意味しない。

conditional-uniform baselineは、exact観測で条件付けたremaining physical tilesがremaining hidden slotsへexchangeableに配置されているというmodel assumptionを使う。これはground truthではない。

wait beliefでは`None = feature unavailable`と`all-zero = estimatorがzeroと評価`を区別する。mechanism channelはmulti-labelであり、単純sumからjoint distributionを復元できるとは仮定しない。

exact / player-safe informationとlearned uncertaintyを混同しない。physical conservationを壊すbeliefを黙って正常値として扱わない。

## `action_vocabulary`

`lisjong.action_vocabulary`は、stable `InternalAction` semantic identityとmodel-facing fixed vocabularyの対応を所有する。

```text
InternalAction semantic identity
        ↓
versioned model-facing index
```

model action indexは麻雀上の合法性の根拠ではない。`DecisionContext.legal_actions`を唯一のcurrent decision legality authorityとして維持する。

このpackageはaction semanticsを再定義せず、tensor / model architecture / trainingを所有しない。

## `structural_efficiency`

`lisjong.structural_efficiency`は、複数のPolicy / diagnosticが共有するstructural discard / 牌効率calculation semanticsを所有するreusable AI-domain componentである。

ownership boundaryを次のように分ける。

```text
policy_contract
    = Policy境界 / visible state / action contract

structural-efficiency component
    = reusable AI-domain structural calculation semantics

concrete Policy
    = reusable semanticsを組み合わせてselection behaviorを定義
```

このcomponentが所有するsupported semantic:

- canonical `DiscardAction` ordering
- actual discard identityによるpost-discard concealed hand導出
- Policy-visible known tile counting
- decision-local structural shanten evaluation / memoization
- post-discard structural shanten evaluation
- effective tile types
- current ukeire
- second-step ukeire

shantenは`hand_evaluation`の公開`calculate_shanten()`だけを正本とし、このcomponentは別のshanten algorithmを持たない。known tile countingはown concealed tiles / public melds / uncalled public discards / dora indicatorsを数え、called discardとmeldで同一物理牌を二重計上せず、4枚を超える不整合をfail closedする。ukeireは「shantenを実際に下げる34基礎牌種 × Policy-visible remaining copies」であり、second-stepは`Σ remaining(t) * best_next_ukeire(t)`のexact integerである。赤5と通常5はstructuralには同じ`TileType`として扱い、actual `DiscardAction` identityは維持する。

supported semanticとimplementation detailを区別する。module-level publicな名前が内部consumerの依存先であり、decision-local shanten cacheのような具体実装は引き続きprivateである。supported evaluatorは公開するが、内部dict cacheやmutable work objectをsupported contractへ露出しない。

```text
supported structural evaluator
    ↓ internally
private decision-local cache
```

このcomponentは次ではない。

```text
lisjong external / top-level public APIの永久固定
external semver contract
generic Policy framework
universal CandidateEvaluation
ML feature schema
```

concrete Policyのstaged selection semanticsはPolicy側が所有する。例えばTwoStepUkeireの`TwoStepUkeireCandidateEvaluation`（Issue #87）は、`post_discard_shanten` / `current_ukeire_count` / `second_step_ukeire_score`の`None = stage未評価`と`0 = 評価済み結果0`を区別するPolicy-specific staged snapshotであり、このreusable componentのlow-level structural valueとは別物である。

```text
reusable low-level structural evaluation
        ↓
Policy固有のstaged evaluation
        ↓
TwoStepUkeireCandidateEvaluation
```

依存方向はconcrete Policy / diagnostic → structural-efficiency componentの一方向とし、逆依存を作らない。新しい牌効率系Policyは、他のconcrete Policy moduleのprivate helperへ依存せずこのcomponentをreuseする。

このcomponentは`PolicyInput`-visible informationだけを使用し、山・王牌・他家concealed truth・future event・`GameTrace` privileged truth・RiichiEnv / Arena固有情報へ依存しない。

歴史的に似たhelper実装を持つlegacy `UkeirePolicy` / `ShantenPolicy`は、Policy世代の独立性のため意図的に小さな重複を保持しており、このcomponentへ移行しない。

## `policies`

`lisjong.policies`はstable / first-party Policy implementationを所有する。

Policy世代ごとのhistorical design詳細やcurrent strength roleをarchitectureへ重複せず、[Policy current status](policy-status.md)を参照する。

ただし、stable consumerが誤解しやすいalgorithmic semanticsはowner側で維持する。

### Finite-horizon structural completion semantics

finite-horizon completion系Policyが扱う値は、

```text
k個のfuture self-draw slotsが存在すると条件付けた
conditional-uniform structural hand-completion value
```

であり、実対局で「k巡以内に和了する確率」ではない。

他家和了、流局、future call / riichi legality、実際に残るself-draw回数等を同じ値へ暗黙に混ぜない。remaining inventoryをlive wallとして扱わない。

selectionでexact integer completion massを使う実装は、float probabilityを新しいcanonical semanticへしない。optimization / pruningはselection semanticを変えないexact-safe reductionに限定する。

新Policyは既存stable semanticsをreuseし、experimentで必要になっただけのgeneric abstractionをproduction coreへ先行追加しない。

---

# Decision observability

objective execution observationとAI decision observationを分離する。

```text
GameTrace          -> lisjong-arena
    what happened

DecisionTrace      -> lisjong
    what canonical Action was selected

AnalysisTrace      -> lisjong
    which typed AI-side intermediate values were produced / used
```

`AnalysisTrace`はPolicy inputではない。

concrete analysis payloadの意味は各AI domain value / Policy implementationが所有し、`AnalysisTrace` root contractがshanten / ukeire / value / defense等を再定義しない。

free-form natural-language reasoningや`dict[str, object]`をcanonical AI semanticsとして固定しない。

trace取得のためにPolicyを二重実行せず、traced / untraced executionでsemantic selected Actionを変えない。`DecisionTrace.selected_action`はvalidation後のcanonical legal `InternalAction`を表す。

`analysis=None`は「analysisを生成していない」を意味し、評価値zeroやempty evaluationへ流用しない。

Policy exception / Action validation failure時に成功したDecisionTraceを偽装しない。observer / sinkの存在をPolicy-visible inputへ入れない。

Arenaは`DecisionTrace` / `AnalysisTrace`をtransport / persistできるが、payload semanticsを再定義しない。

---

# Arenaとのboundary

Arenaは少なくとも次の3責務を持つ。

```text
Execution / Observation
Experiment-local Research / ML
Evaluation
```

## Execution / Observation

Arenaが所有する代表例:

- RiichiEnv / RiichiLab integration
- local / external runner
- session / retry / reconnect
- raw game record / protocol trace
- environment-applied Action record
- external Observation -> lisjong-owned Policy contract projection
- `InternalAction` -> external legal Action mapping / revalidation

このlayerはPolicy strategyやtraining objectiveを所有しない。

## Experiment-local Research / ML

Arenaがbounded experimentとして所有できる代表例:

- purpose-specific feature / tensor schema
- dataset / split / manifest
- trainer / model / loss / optimizer
- checkpoint / model artifact
- offline Q / BC等のexperiment logic
- HandBelief training / scale study / calibration measurement
- failure diagnosis

このlayerはstable `HandBelief` / shanten / action semanticsを再定義しない。

## Evaluation

Arenaが所有する代表例:

- candidate / baseline matchup
- fixed seed set
- seat rotation
- comparison protocol
- strength / diagnostic metric aggregation
- immutable evaluation artifact
- external benchmark orchestration

Arenaのpositive resultはstable adoptionの入力にはなるが、production promotionそのものではない。

---

# Learned Policy / learned estimator boundary

Learned Policy研究では、hand-crafted Policyとは別のAI contractを作らず、既存の`PolicyInput` / `InternalAction` / action vocabulary / execution / evaluation boundaryをreuseする。

ただし、stable input contractとexperiment-local tensorを区別する。

```text
PolicyInput
    stable lisjong semantic contract
        ↓
experiment-local encoder / tensor
    Arena research contract
        ↓
experiment-local model / checkpoint
        ↓
experiment-local Policy adapter
        ↓
Arena evaluation
```

research結果がadoption-worthyになった場合にのみ、production inference boundaryを再設計する。

その際、次を明示的に決める。

- canonical feature extraction owner
- model loading / inference owner
- ML runtime dependency
- weights distribution / storage
- stable versioning
- fallback / failure semantics
- `lisjong` coreへ入れる範囲

experiment-local PyTorch model classやtensor schemaを、そのままstable public APIにしない。

---

# HandBelief learned-estimator boundary

HandBelief研究でも同じ分離を使う。

```text
lisjong
    HandBelief semantics
    physical validity
    canonical inference / consumer boundary

lisjong-arena research
    corpus / dataset
    bounded trainer
    learned model
    MAE / calibration / scale study
    artifact / provenance

lisjong-arena evaluation
    decision / game-strength effect
```

learned estimatorのprediction qualityが改善しても、そのままdecision valueやgame strengthを主張しない。

production統合時には、learned-model固有tensor / framework objectを`HandBelief` consumerへ漏らさず、stable domain valueへprojectionする。

---

# Dependency direction

project-wide dependency directionを維持する。

```text
lisjong-arena -> lisjong
lisjong-play  -> lisjong

lisjong -X-> lisjong-arena
```

lisjong内部の概念的依存は、下位stable contractから上位decision implementationへ向ける。

```text
policy_contract
    ↑
hand_evaluation / belief / action_vocabulary
    ↑
structural_efficiency
    ↑
policies / stable AI consumers
```

実際には各component間で必要な依存だけを持ち、循環依存を作らない。

特に:

- `policy_contract`は`policies`へ依存しない
- `policy_contract`はshanten / ukeire / second-step等の具体AI evaluation semanticsを所有しない
- `policy_contract`はArenaへ依存しない
- `belief`はPolicy implementationへ依存しない
- `structural_efficiency`はconcrete Policy implementationへ依存しない
- concrete Policy / diagnosticは、共有structural semanticについて他Policy moduleのprivate helperへ依存しない
- experiment-local Arena model codeをlisjong stable modulesからimportしない
- Arenaがstable lisjong APIをconsumerとして利用する

production promotion時もreverse dependencyを作らず成立するplacementを選ぶ。

---

# External execution / migration history

RiichiEnv / RiichiLabのexecution / observation implementationは既に`lisjong-arena`へ移管されている。

`lisjong`はexternal runtimeを再実装しない。

current contractはArena側の以下を正本とする。

- Arena architecture
- RiichiEnv local execution / Adapter
- RiichiLab client / protocol bridge
- GameTrace / durable observation contract

lisjong側の`riichienv-investigation.md`、`riichilab-client.md`、`riichilab-adapter.md`等に残るhistorical migration情報は、current ownershipを上書きしない。

architectureへ特定のArena dependency pinやmigration時点のcommit SHAをcurrent contractとして固定しない。必要なexact revisionはconsumer repositoryのdependency / provenance contractを正本とする。

---

# Validation principles

stable semanticsを変更する場合は、semantic correctnessをowner repositoryで固定する。

例:

- Action identity
- shanten result semantics
- HandBelief representation / physical validity
- wait semantics
- PolicyInput visibility
- DecisionTrace semantics

一方、learned modelのquality / calibration / statistical effectはexperiment protocol側で測定できる。

correctness testとresearch resultを混同しない。

```text
stable semantic test failure
    = contract defect

experiment negative result
    = valid research outcomeになり得る
```

negative / inconclusive / invalid experimentを区別する。

---

# Data / artifact / secret boundary

repositoryへ次を直接commitしない。

- credential / token / API key
- external model weight / generated weight
- usage rights未確認の牌譜 / raw data
- large generated experiment artifact
- machine-local secret / identity

training / evaluation artifactはowner experimentのprovenance contractへbindする。

source revision、rules、Policy population、dataset identity、model identity等を解決できない場合、完全に再現可能であると偽装しない。

privileged ground truthをartifactへ保存できる場合でも、online Policy inputへ流入させない。

---

# Current non-goals

`lisjong` architectureとして、次をdefault responsibilityにしない。

- RiichiEnv / RiichiLab session lifecycleやrunner
- Arena evaluation orchestration
- generic experiment platform / generic trainer framework
- experiment-local dataset / tensor / checkpointの集約
- HPO platform / model registry / cloud scheduler
- external model weightsのrepository内配布
- omniscient stateを使うonline Policy
- project-wide canonical GameRecord / replay format
- 3人麻雀対応
- profiling evidenceなしのnative rewrite

一方、次は「非目標」ではない。

- learned Policy研究
- learned HandBelief estimator研究
- stable production inference contractへの将来promotion
- HandBeliefをdecisionへ利用するconsumer

これらは、Arena experiment-local researchでevidenceを得た後、promotion reviewを経て`lisjong`のstable AI semanticsへ統合し得る。

---

# Detailed contract documents

- [Policy契約](policy-contract.md)
- [Policy入力の最小スキーマ](policy-input-schema.md)
- [内部Actionモデル](internal-action-model.md)
- [Action identity](action-identity.md)
- [Model-facing action vocabulary](action-vocabulary.md)
- [Policy current status](policy-status.md)
- [RiichiEnv investigation — historical / validation reference](riichienv-investigation.md)
- [RiichiLab client — historical migration pointer](riichilab-client.md)
- [RiichiLab Adapter — historical migration pointer](riichilab-adapter.md)

project-wide responsibility / roadmap:

- [lisjong-project Architecture](https://github.com/lisbun/lisjong-project/blob/main/docs/architecture.md)
- [lisjong-project Roadmap](https://github.com/lisbun/lisjong-project/blob/main/docs/roadmap.md)

Arena ownership / evaluation:

- [lisjong-arena Architecture](https://github.com/lisbun/lisjong-arena/blob/main/docs/architecture.md)
- [lisjong-arena Roadmap](https://github.com/lisbun/lisjong-arena/blob/main/docs/roadmap.md)
- [Policy strength evaluation policy](https://github.com/lisbun/lisjong-arena/blob/main/docs/policy-strength-evaluation.md)
