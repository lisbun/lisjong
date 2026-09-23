# Architecture

## 目的

`lisjong` は、日本式立直麻雀AIの **decision core** と、Heuristic / Learnedを問わない **canonical Learning / Learned Policy capability** を所有するrepositoryである。

外部環境のprotocolや型をPolicyから分離し、各seatが判断時点で観測可能な情報だけをPolicy / learned inferenceへ渡すことを最優先の境界とする。

lisjong ecosystem全体のrepository責務と依存方向は、[`lisjong-project` のArchitecture](https://github.com/lisbun/lisjong-project/blob/main/docs/architecture.md)とADR 0008をproject-wideな正本とする。

本書は、その横断境界の内側にある`lisjong`固有の次を正本とする。

- Policy / Policy contract
- `PolicyInput` / `DecisionContext` / `InternalAction`等のAI-side semantic contract
- shanten / ukeire / HandBelief / risk / value / utility等のstable domain semantics
- model-facing feature representation / action vocabulary
- Learning dataset semantics / teacher / label
- model architecture / training objective / trainer
- model artifact identity / load semantics
- Learned Policy / learned estimator inference
- Learning objective固有diagnostics / intrinsic metric definition
- hidden-information / player-safe information boundary

external / local execution、observation、Arena-executed population provenance、Policy / game-strength evaluationは[`lisjong-arena`](https://github.com/lisbun/lisjong-arena)を正本とする。

historicalなArena Learning implementationやartifactはhistorical provenanceを維持する。ownership変更を理由に遡及移動・改名・再指定しない。

## Source of truth

```text
lisjong-project/docs/architecture.md + ADR 0008
    project-wide repository responsibility
    dependency direction
    Learning ownership / migration boundary

lisjong/docs/architecture.md
    AI-side architecture
    feature / dataset / training / artifact / inference boundary
    hidden-information boundary

purpose-specific lisjong docs
    concrete stable / Learning contract

lisjong-arena
    execution / observation
    source-record / population provenance
    reproducible strength evaluation
    historical locked experiment implementation

GitHub Issues / PRs
    current work
    experiment protocol / result
    adoption decision
```

Policyの公開契約は[Policy契約](policy-contract.md)、Policy入力の許可fieldと意味契約は[Policy入力の最小スキーマ](policy-input-schema.md)、内部Actionのvariant / field / semanticsは[内部Actionモデル](internal-action-model.md)、Action semantic identityは[Action identity](action-identity.md)、model-facing action indexは[Model-facing action vocabulary](action-vocabulary.md)を正本とする。

---

# Core ownership boundary

短く言うと、次の境界を維持する。

```text
lisjong
    = how the AI is built
      semantics / feature / dataset / teacher
      training / artifact / inference / Policy

lisjong-arena Execution / Observation
    = what happened

lisjong-arena Evaluation
    = how candidates are compared reproducibly
```

Learningは`lisjong`内でtrainingとinferenceを一続きのcanonical capabilityとして所有する。Arenaで成立したhistorical implementationはrequirements / failure modes / provenance disciplineのreferenceとして利用できるが、同等のcanonical semanticsを両repoで独立維持しない。

```text
historical Arena feature / dataset / checkpoint identity
    != new lisjong canonical identity
```

bulk code movementは行わず、concrete use caseごとに必要なcapabilityをreconstructする。

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

## Semantic correctness, Learning metrics, and strength evidence

ownershipを次のように分ける。

### `lisjong` が正本とするもの

- `HandBelief` field / shanten / risk / value等の意味
- player-safe feature representation / action vocabulary
- dataset interpretation / teacher / label semantics
- model architecture / training objective / trainer
- model artifact identity / load semantics
- Learned Policy / learned estimator inference
- Learning objective固有metricのdefinition / semantic threshold
- production consumerが依存するsemantic correctness

### `lisjong-arena` が正本とするもの

- Arena-executed population / rotation / seed provenance
- formal Policy / game-strength comparison
- evaluation artifact / statistical evidence
- external benchmark

intrinsic metricがinteractive executionを必要とする場合は、definition / execution / artifactを分離する。metricの意味は`lisjong`、Arenaが実行するpopulationのprovenanceはArena、artifact schemaはproducerが所有する。

```text
better prediction / lower training loss
!=
better decision
!=
stronger Mahjong AI
```

# Historical experiment → canonical capability boundary

既存Arena experimentはhistorical evidence / reference implementationとして扱い、ownership変更だけでartifact identityやprotocolを変更しない。

新しいcanonical capabilityは次の順序で構築する。

```text
concrete Learning use case
    ↓
existing Arena implementation / evidenceをinspect
    ↓
requirements / invariants / failure modesを抽出
    ↓
lisjong-owned contractを設計
    ↓
必要なcapabilityだけ実装
    ↓
Arenaでstrength evaluation
```

generic ML platformやbulk migrationを先行させない。historical Arena codeの削除もこのarchitecture変更の完了条件にしない。

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

Learning側のconsumerも同じ一方向依存に従う。`lisjong.learning.candidate_features`（Issue #187）は、このcomponentのsupported semanticをsingle sourceとして呼ぶpurpose-specificなmodel-facing candidate projectionであり、別のshanten / ukeire algorithmを持たない。`TwoStepUkeireCandidateEvaluation`とも別contractである。詳細は[Learning L0](learning-l0.md)「Candidate feature contract」を正本とする。L0.2（Issue #189）のlearned candidate scorerは、このcandidate projectionをtwo-pass finalist requestで呼ぶだけであり、shanten / ukeire / second-stepを再実装せず、concrete Policy moduleのprivate helperにも依存しない。詳細は同文書「Candidate-centric Learned Offense Policy」を正本とする。L0.2a（Issue #191）のsemantic-envelope Policyも同じcandidate projectionの値だけでS1 shanten → S2 current ukeire → S3 second-stepのsurvivorを決め、TwoStepUkeire moduleへ依存しない（TwoStepとの一致はtest / replayのreferenceとしてだけ注入する）。詳細は同文書「Semantic-envelope Learned Offense Policy」を正本とする。

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

Arenaのcurrent target responsibilityは次の2つである。

```text
Execution / Observation
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
- reusable player-safe source-record schema
- Arena-executed population allocation / seed provenance

ArenaがLearning consumer向けsource recordを生成する場合、player-safe observation、legal actions、selected / applied action、provenanceを保持し、encode済みfeature tensorやtraining objective固有labelをcanonical sourceにしない。

## Evaluation

Arenaが所有する代表例:

- candidate / baseline matchup
- fixed seed set / seat rotation
- comparison protocol
- strength metric / statistical comparison
- immutable evaluation artifact
- external benchmark orchestration

historical / already-locked Arena Learning implementationは残せるが、新しいcanonical Learning capabilityのownerではない。

---

# Learned Policy / learned estimator boundary

Learned Policyはhand-crafted Policyと同じ`PolicyInput` / `InternalAction` / action vocabulary / execution boundaryをreuseし、Learning capability自体を`lisjong`が所有する。

```text
player-safe source record
        ↓
lisjong feature / dataset materialization
        ↓
teacher / label
        ↓
training objective / trainer
        ↓
immutable model artifact
        ↓
LearnedPolicy / learned estimator inference
        ↓
Arena strength evaluation
```

## Runtime dependency boundary

- core（`policy_contract` / `hand_evaluation` / `belief` / `action_vocabulary` / non-ML policies）はML framework非依存を維持する
- training / learned inferenceはoptional extra + lazy importとする
- ML framework未導入環境でもcore importを成功させる
- ML CIはpinned CPU buildでLearning pathを検証する
- `lisjong -> lisjong-arena` のruntime dependencyを作らない

## Artifact / serving boundary

- canonical feature extraction owner: `lisjong`
- model loading / inference owner: `lisjong`
- ML runtime dependency: optional extra、coreはML-free
- weights distribution / storage: repository外のoperator-owned artifact
- stable versioning: versioned identity + fail-closed load
- fallback / failure semantics: silent fallbackを作らずfail closed
- `lisjong` coreへ入れる範囲: core contractはML-free、Learning実装は独立package / optional path

model artifactは実行用model objectと分離したimmutable snapshotとし、factory / callable / arbitrary codeを保存・復元しない。既存artifactを上書きせず、teacher identity、source provenance、dataset identity、feature fingerprint、vocabulary fingerprint、training config、RNG、checkpoint digest等の必要なidentityをbindする。

# HandBelief learned-estimator boundary

HandBeliefでも同じownershipを使う。

```text
lisjong
    HandBelief semantics / physical validity
    feature / label semantics
    learned-estimator training / artifact / inference
    intrinsic prediction metric definition

lisjong-arena
    Arena-executed population provenance
    Policy / game-strength evaluation
```

learned estimatorのprediction qualityが改善しても、そのままdecision valueやgame strengthを主張しない。model固有tensor / framework objectをconsumerへ漏らさず、stable domain valueへprojectionする。

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
- historical Arena model codeをlisjong Learning modulesからimportしない
- Arenaがstable lisjong AI / Learning APIをconsumerとして利用する
- Learning packageからArena evaluator / runnerへ依存しない

cross-repository artifactを使う場合もowner / version / provenanceを明示し、artifact経由のhidden reverse dependencyを作らない。

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

Learning artifactはimmutable / versioned / fail-closedを基本とし、source population identity、teacher identity、dataset identity、feature / vocabulary fingerprint、training config、RNG、checkpoint digest等を解決可能にする。

Arena source recordからmaterializeしたdatasetは、source schema versionとsource population identityを記録する。unknown source version、identity mismatch、欠損 / extra recordを推測でacceptしない。

privileged ground truthをoffline label / validationへ利用する場合も、online Policy / learned inference inputへ流入させない。

---

# Current non-goals

`lisjong` architectureとして、次をdefault responsibilityにしない。

- RiichiEnv / RiichiLab session lifecycleやrunner
- Arena evaluation orchestration
- generic ML platform / HPO platform / model registry / cloud scheduler
- external model weightsのrepository内配布
- omniscient stateを使うonline Policy
- project-wide canonical GameRecord / replay format
- 3人麻雀対応
- profiling evidenceなしのnative rewrite
- Arena historical Learning codeの一括移動 / 一括削除
- long-term training / rollout hosting ownerの先行固定

一方、feature / dataset / teacher / training / model artifact / Learned Policy / learned estimatorは、concrete use caseから必要になった範囲で`lisjong`のcanonical Learning capabilityとして実装する。

---

# Detailed contract documents

- [Policy契約](policy-contract.md)
- [Policy入力の最小スキーマ](policy-input-schema.md)
- [内部Actionモデル](internal-action-model.md)
- [Action identity](action-identity.md)
- [Model-facing action vocabulary](action-vocabulary.md)
- [Learning L0 — canonical player-safe dataset / BC artifact / LearnedPolicy](learning-l0.md)
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
