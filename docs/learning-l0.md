# Learning L0 — canonical player-safe dataset / BC artifact / LearnedPolicy

## 目的と位置付け

本書は、lisbun/lisjong#184 で確立した最初のlisjong-native Learning vertical
sliceの正本である。

```text
versioned player-safe source record   (lisjong-arena producer)
    -> lisjong-owned feature materialization
    -> lisjong-owned dataset
    -> bounded BC training
    -> immutable model artifact
    -> lisjong-owned LearnedPolicy
    -> 通常のlisjong Policy契約
    -> Arenaが実行
```

- ownership境界の正本は[Architecture](architecture.md)「Learned Policy /
  learned estimator boundary」であり、本書はそのconcreteなidentity / schemaを
  記録する
- Policyの公開契約は[Policy契約](policy-contract.md)、`PolicyInput`は
  [Policy入力の最小スキーマ](policy-input-schema.md)、model-facing action
  vocabularyは[Model-facing action vocabulary](action-vocabulary.md)を正本とする
- 実装は`src/lisjong/learning/`、testは`tests/test_learning_*.py`

historicalなlisjong-arena実装（#331/#332、`arena-policy-input-feature-v1`等）
はreferenceであり、本書のidentityとは別である。本書のidentityを再利用・改名して
historical identityと混同しない。

## Source-record consumer

`lisjong.learning.read_source_record()`は、lisjong-arena#342が生成する
versioned player-safe source recordをstrict readする。

```text
schema      arena-offense-o0-player-safe-source-record-v2  (current)
            arena-offense-o0-player-safe-source-record-v1  (historical, readback-only)
```

- 未対応schemaは`UnsupportedSourceSchemaError`でfail closedする
- manifest identity、game provenance（seed / split / lock identity /
  game mode）、row間のexecution ordering、actor seatとPolicyInput / legal
  actions / selected actionの整合をすべて検証する
- Arenaのencoded feature tensor（`features.f32`等の locked #331 scientific
  corpus）は読まない。読むのはtyped player-safe source recordだけである
- 返り値の`provenance()`が、下流のdataset / artifactへbindするsource identity
  とordered populationを提供する

### Arena allocation provenance（v2、lisjong-arena#346/#347）

schema v2のmanifestは、split別のArena seed-allocation binding
（`allocation_identity` / `ledger_revision` / `owner_repository` /
`seed_domain` / `seed_membership_identity`）を`allocation_bindings`として持つ。
`read_source_record()`は次をstrict validateし、`PlayerSafeSourceRecord.
allocation_bindings`（v1なら`None`）へそのまま保持する。

- binding集合が実際のsource populationのsplit集合と完全一致すること
  （欠損・余剰はfail closed）
- 各fieldのshape（SHA-256 hex、canonical owner repository、`seed_domain`の
  形式）
- `seed_membership_identity`が、そのsplitの実際のseed集合から
  `lisjong.learning.seed_membership_identity()`（Arena
  `seed_registry.seed_membership_identity()`とbyte-for-byte一致する
  独立実装）で再計算した値と一致すること

lisjongはArena allocation ledgerを再生成・再所有せず、live ledgerへの
再照会も行わない（`lisjong_arena`への runtime import は存在しない）。
`allocation_bindings`はdataset manifestの`source.allocation_bindings`、
model artifact manifestの`source.allocation_bindings`へそのまま伝播する。

`materialize_dataset()`はschema v2（`allocation_bindings`を持つsource
record）だけを受け付ける。historical v1 source recordはallocation
provenanceを持たないため、推測で補完せず`DatasetError`でfail closedする。
v1は historical readback（`read_source_record()`単体の呼び出し）にのみ使う。

## Canonical player-safe feature representation

```text
identity     lisjong-offense-l0-player-safe-feature-v1
```

現在のdimensionは`lisjong.learning.FEATURE_DIMENSION`として実装から取得する
（layout変更に追随するため、本書では固定値として複製しない）。

`lisjong.learning.build_player_safe_feature(policy_input)`は、`PolicyInput`
だけから決定的にfeature vectorを構築する。`lisjong.learning.feature_fingerprint()`
がlayout全体のfingerprintを返し、dataset / artifactはこの値をbindする。

- 入力は`PolicyInput`のみ。legal actions、engine state、HandBelief、
  shanten / ukeire等のPolicy-internal analysis、opponent非公開情報、
  wall / dead wall実配列は使わない
- axisはactor相対seat（self / shimocha / toimen / kamicha）、canonical
  34牌種index（`lisjong.belief.canonical_axes`）、canonical Wind axisだけで
  決まり、Enum定義順やdict iteration orderに依存しない
- tile conservation違反（1牌種5枚以上等）や、宣言済み物理bound超過（濃厚牌14枚
  超、副露5つ以上、dora表示牌6枚以上、残り牌136枚超）は`FeatureError`で
  fail closedする

historical `arena-policy-input-feature-v1`（8204次元）とは別identityであり、
再利用・改名・再定義しない。

## Dataset materialization

```text
schema              lisjong-offense-l0-bc-dataset-v1
label semantics      lisjong-offense-l0-teacher-selected-action-v1
training objective    masked-action-classification
```

`lisjong.learning.materialize_dataset(source_record, destination)`は、1
dataset = 1 immutable directoryとして次を書く。

```text
<dataset>/
    manifest.json     canonical JSON identity / provenance / payload digest
    rows.jsonl        1行 = 1 decisionのprovenanceとteacher label
    features.f32      N x FEATURE_DIMENSION little-endian float32 (row-major)
    legal_mask.u8     N x ACTION_VOCABULARY_SIZE uint8 (0 / 1)
```

manifestは、source-record identity、feature identity/fingerprint、action
vocabulary version/fingerprint、teacher/label semantics、ordered source
population、split別row数、Arena allocation binding（`source.
allocation_bindings`、上記参照）をbindする。既存destinationは上書きせず、
`lisjong.learning.read_dataset()`がbindされた全identityとpayload digestを
照合してstrict readする。同じsource recordから再materializeすると同一identityに
なる。`materialize_dataset()`はschema v2のsource recordだけを受け付ける。

## Bounded BC trainer

`lisjong.learning.train_behavior_cloning(dataset, config, destination)`は、
offense L0のための最小限のBehavior Cloningだけを提供する。

```text
architecture   lisjong-offense-l0-bc-mlp-v1  (Linear -> ReLU -> Linear)
optimizer      Adam
objective      masked cross entropy (illegal indexへ確率質量を与えない)
```

`BehaviorCloningConfig`はtrain / validation splitの明示、epoch数、batch size、
learning rate、weight decay、seedを固定する。generic experiment framework、
hyperparameter search、self-play/RLは提供しない。

## Immutable model artifact

```text
schema   lisjong-offense-l0-bc-model-artifact-v1
```

`lisjong.learning.write_model_artifact()` / `load_model_artifact()`は、1
artifact = 1 immutable directory（`manifest.json` + `weights.f32`）として
repository外に置く。manifestは、source/dataset identity（Arena allocation
binding含む）、feature fingerprint、action vocabulary fingerprint、model
architecture/config、optimizer/training config、RNG seed、選択epoch、
weights digest、lisjong package versionと実際にimportされたlisjongソースの
digest（`source_digest`）をbindする。`source_digest`はprovenance記録であり、load時に現在の
installed lisjongと一致することは要求しない（load-relevantなidentityは
feature / vocabulary fingerprintが別途厳密に照合する）。

- 既存artifactを上書きしない
- identity / fingerprint / configのmismatchはload時にfail closedする
- weightsはparameter layout順のflat float32 payloadであり、callable /
  factory / lambda / pickleされたobjectは保存・復元しない

## LearnedPolicy inference

`lisjong.learning.load_learned_policy_factory(path)`は、artifactをstrict
loadし、`LearnedPolicyRuntime`（top-level importableなfactory）を返す。

```text
LearnedPolicyRuntime()   -> fresh LearnedOffensePolicy instance
LearnedOffensePolicy.choose_action(decision: DecisionContext) -> InternalAction
```

- `choose_action`は通常の`lisjong` Policy契約（`Policy` Protocol）を満たす
- 返すActionは常に`decision.legal_actions`側のcanonical objectであり、
  `execute_policy()`のvalidationをそのまま通る
- illegal indexへは確率質量を与えない。非有限logits、legal candidate不在、
  vocabulary version不一致はすべて例外にし、silent heuristic fallbackはない
- Arenaは`LearnedPolicyRuntime`インスタンスをそのまま
  `PolicySpec(identity=..., factory=runtime)`のfactoryとして使える

## Candidate feature contract（L0.1、#187）

```text
identity   lisjong-offense-l0.1-discard-candidate-feature-v1
実装        src/lisjong/learning/candidate_features.py
test        tests/test_learning_candidate_features.py
```

`lisjong.learning.build_discard_candidate_features(decision)`は、1 decisionの
legal `DiscardAction`ごとにcandidate単位のmodel-facing projectionを返す。
将来のlearned candidate scorer（L0.2）が、NNへshanten / ukeireを再発見させずに
canonical牌効率semanticを使えるようにするためのseamである。

```text
DecisionContext
    -> legal DiscardActionだけを抽出
    -> canonical structural semantics
    -> canonical順のDiscardCandidateFeatures tuple
```

### responsibility identity

3つのcontractを混同しない。

```text
lisjong.structural_efficiency
    = canonical reusable Mahjong calculation semantic
      （shanten / ukeire / second-stepの正本）

lisjong.learning.candidate_features
    = purpose-specific model-facing candidate projection
      （#187が追加するLearning-side contract）

TwoStepUkeireCandidateEvaluation
    = TwoStep固有のstaged trace semantic
      != learned candidate feature contract
```

candidate viewは新しいshanten / ukeire algorithmを実装せず、
[Architecture](architecture.md)「`structural_efficiency`」が所有する次の
supported APIをsingle sourceとして呼ぶ。

```text
known_tile_counts             Policy-visibleな既知牌counting
StructuralShantenEvaluator    decision-local shanten評価 / memoization
evaluate_post_discard_hands   actual discard identity -> 打牌後純手牌 / shanten
ukeire_count                  current受け入れ
second_step_ukeire_score      2段階受け入れscore
discard_action_sort_key       canonical deterministic ordering
```

Issue #87の`TwoStepUkeireCandidateEvaluation`はTwoStepUkeireが実際に辿った
staged evaluation pathのtrace snapshotであり、generic ML schemaへ変更しない。
TwoStepはminimum-shanten候補だけcurrent ukeireを評価するが、candidate viewは
全candidateでmaterializeする。両者のstage semanticsは独立である。

本contractは上記L0（`FEATURE_IDENTITY` / `DATASET_SCHEMA` /
`MODEL_ARTIFACT_SCHEMA` / action vocabulary identity）とは別の
purpose-specific contractとして**追加**される。#187を理由に既存identityを
candidate-centricへ遡及変更しない。`LearnedOffensePolicy`のflat action
vocabulary pathも現状のまま残す。

### 各candidateが持つ値

```text
action                    元のlegal actionのcanonical DiscardAction object
post_discard_shanten      全legal discard candidate
current_ukeire_count      全legal discard candidate
second_step_ukeire_score  EVALUATEDのときだけint、それ以外はNone
second_step_status        SecondStepStatus
```

`action`は`DecisionContext.legal_actions`側のobjectそのものであり、赤5 /
通常5、ツモ切り、actorのidentityを再構築しない。結果は
`discard_action_sort_key()`によるcanonical deterministic orderで返し、
legal action入力順へ依存しない。入力は`DecisionContext`だけで、wall truth、
dead-wall truth、opponent concealed truth、future event、`GameTrace`
privileged truth、RiichiEnv state、Arena固有dataへ依存しない。

### Second-step materialization contract（v1: selective）

```text
shanten         all candidates
current ukeire  all candidates
second-step     explicit selective materialization
```

`second_step_actions`で明示requestされたcandidateだけ2段階受け入れを評価する。
requestしていないcandidateを暗黙に評価せず、request対象が現在のlegal discard
candidateでない場合は`CandidateFeatureError`でfail closedする。
cross-decision cacheは持たず、`known_tile_counts()`と
`StructuralShantenEvaluator`は1 build呼び出しにつき1つを全candidateで共有する。

根拠は#187のperformance preflight実測である（`tools/benchmark_candidate_features.py`、
CPython 3.14.6 / Windows、repeat=9のmedian、decisionあたりms）。

| decision | candidates | shanten | all-candidate ukeire | all-candidate 2nd | selective 2nd |
| --- | --- | --- | --- | --- | --- |
| far wide hand | 14 | 0.34 | 11.9 | 1188.6 | 323.1 |
| mid honor-pair hand | 12 | 0.37 | 13.3 | 243.4 | 118.0 |
| two-step relevant hand | 10 | 0.35 | 12.1 | 247.1 | 82.3 |
| wide suited hand | 13 | 0.48 | 15.2 | 224.2 | 54.0 |
| tenpai-reachable hand | 12 | 0.39 | 12.9 | 0.4 | 12.9 |

shantenとcurrent ukeireは全candidate materializeしてもdecisionあたり
0.3〜15msに収まる。一方all-candidate second-stepは224〜1189msであり、同じ
decisionの他stageより1〜2桁大きい。よってv1では全candidate mandatoryにせず、
selective contractとする（`SECOND-STEP MATERIALIZATION READY — SELECTIVE`）。
この計測はdevelopment-only utilityであり、CI wall-clock thresholdや
correctness testの時間thresholdは追加しない。

### Availability semantics

`evaluated result == 0`と`not materialized` / `not applicable`を混同しない。
`second_step_ukeire_score: int | None`だけで`None`に複数の意味を持たせず、
`SecondStepStatus`で明示する。

```text
EVALUATED          評価済み。score 0も正当な評価結果である
NOT_MATERIALIZED   applicableだがrequestされていない
NOT_APPLICABLE     semantic上適用しない（打牌後聴牌以上）
```

`post_discard_shanten <= 0`では2段階受け入れを`NOT_APPLICABLE`とする。第1
有効牌のツモが和了そのものであり、「仮想ツモ後に打牌して次の受け入れを測る」
という`second_step_ukeire_score()`のsemanticが成立しないためである。これは
Issue #87のstaged semantics（TwoStepUkeireも最小向聴が0なら2段目を評価せず
tie-breakへ進む）と一致する。`NOT_APPLICABLE`なcandidateを明示requestしても
fail closedせず`NOT_APPLICABLE`のまま返す。fail closedするのは、requestされた
actionが現在のlegal discard candidateでない場合だけである。

### Policy seamはL0.2へdeferする

#187ではcandidate scorer modelを実装しないため、新しいgeneric Policy
router / dispatcher frameworkを追加しない（`POLICY SEAM REFORMULATE`）。

`TwoStepUkeirePolicy`は既に`winning action -> RiichiAction -> 通常打牌評価 ->
pass -> fallback`というO0相当のdecompositionを`_decide()`が持ち、打牌選択は
`_decide_discard()`というprivate extension pointへ分離されている。必要な
decompositionは現行architectureが既に示しているが、実際のcandidate scorerが
存在しない段階でgeneric routerを抽出するのはspeculative abstractionになる。

```text
existing architecture already demonstrates the required O0 decomposition,
but extracting a generic router before a real candidate scorer exists would
be speculative abstraction.

The minimal reusable O0 orchestration seam should be extracted in L0.2
together with the first real learned candidate scorer.
```

`TwoStepUkeirePolicy`をLearning base classとして再利用することもしない。

## Candidate-centric Learned Offense Policy（L0.2、#189）

```text
O0 decomposition     lisjong-offense-l0.2-o0-decomposition-v1
request policy       lisjong-offense-l0.2-two-pass-finalist-second-step-v1
candidate encoding   lisjong-offense-l0.2-discard-candidate-encoding-v1
dataset schema       lisjong-offense-l0.2-candidate-scorer-dataset-v1
label semantics      lisjong-offense-l0.2-teacher-selected-discard-candidate-v1
model architecture   lisjong-offense-l0.2-candidate-scorer-mlp-v1
artifact schema      lisjong-offense-l0.2-candidate-scorer-artifact-v1
offline gate         lisjong-offense-l0.2-offline-gate-v1
実装                  src/lisjong/learning/_o0.py, candidate_encoding.py,
                     candidate_dataset.py, candidate_model.py,
                     candidate_training.py, candidate_artifact.py,
                     candidate_policy.py, candidate_diagnostics.py
test                 tests/test_learning_candidate_*.py
```

#331のflat 802-way single headにwin / no-call / 牌効率hierarchyを丸ごと学習
させる設計から、**deterministic O0 guard + normal-discard candidate scorer**へ
進む最初のslice。上記L0の`FEATURE_IDENTITY` / `DATASET_SCHEMA` /
`MODEL_ARTIFACT_SCHEMA` / action vocabulary identity / `LearnedOffensePolicy`は
historical / current L0としてそのまま残し、L0.2は別identityとして**追加**する。

### Accepted source / teacher lineage

teacher labelはsource recordに記録済みの`teacher_selected_action`だけであり、
現在の`TwoStepUkeirePolicy`を再実行してrelabelしない。L0.2のaccepted lineageは
次のretained evidenceで確認した（Issue #189コメントに記録）。

```text
source record schema      arena-offense-o0-player-safe-source-record-v2
source record identity    8c1899c528f3fd551b57d7d8e4a7d44f42cfbf03426f611bd6794823ced3fcc7
lock identity             89100db2833e4e8274f602eeb1c1853c21d1ba4d34772915f62e9aac39a6148b
scientific corpus         bc8563112797f7c6e9a6607354e1663414e5a1b9f2bb4dd08e2b0ad9b98a8155
teacher (Arena binding)   lisjong.policies.TwoStepUkeirePolicy @ lisjong 15799e5, 4 seats
population                TRAIN 100 / SELECT 20 / OFFLINE-EVAL 20 hanchan, 90,579 decisions
```

同じidentityを#331 retention manifest、#331 preflight、#332 phase-B
scientific lockがbindしている。この確認はoperatorがretained evidenceを読んで
行ったものであり、production dataset codeはArena-owned `source_contract`を
parseせず`source_contract_digest`だけを保持する。

### O0 decomposition

```text
winning action (Ron / Tsumo) exists  -> deterministic win（canonical tie-break）
elif RiichiAction exists             -> canonical RiichiAction
elif no legal DiscardAction          -> canonical PassAction（無ければfail closed）
else                                 -> learned candidate scorer over legal DiscardAction
```

`lisjong.learning._o0`がservingとdataset row eligibilityの両方で使う唯一の
precedenceである。Ankan / Kakan / KyuushuKyuuhai等のown-turn voluntary actionは
O0では選ばない。win / immediate riichiのsemanticsは`TwoStepUkeirePolicy`と
一致することをtestで固定し、`TwoStepUkeirePolicy`を継承せず、concrete Policy
moduleのprivate helperへも依存しない。generic router / dispatcherは作らない。

scorer branchは`legal_discard_candidates()`由来のcandidateだけにscoreを付け、
選んだcandidateが持つ`decision.legal_actions`側の`DiscardAction` objectを
そのまま返す。同点は`discard_action_sort_key()`順で最初のcandidate、非有限
score・score数不一致はfail closedであり、silent heuristic fallbackはない。
riichi宣言牌decision、riichi中のforced tsumogiriも同じpathで扱う。

### Shared context / candidate encoding

shared decision contextは#184の`build_player_safe_feature()`をそのまま再利用
する（`FEATURE_IDENTITY` / fingerprint / layoutは不変）。per-candidateは#187の
`DiscardCandidateFeatures`を次のfixed-width vectorへ写す。

```text
discard_tile_type 34 / discard_red 1 / tsumogiri 1 /
post_discard_shanten 9 (0..8 one-hot) / shanten_gap 1 /
current_ukeire 1 / ukeire_gap_within_shanten 1 /
second_step_status 3 (EVALUATED / NOT_MATERIALIZED / NOT_APPLICABLE) /
second_step_score 1 / second_step_gap 1
```

gap blockは同じdecisionのcandidate tupleだけから決まる。status one-hotにより
評価済みscore 0とNOT_MATERIALIZED / NOT_APPLICABLEは別vectorになる。現在の
dimension / fingerprintは`CANDIDATE_ENCODING_DIMENSION` /
`candidate_encoding_fingerprint()`から取得する。encodingはsecond-step
availabilityがrequest policyの結果と一致することを要求し、一致しない
materializationはfail closedする。

### Second-step request policy（two-pass finalists）

```text
pass 1   second-step requestなしでbuild（全candidateのshanten / current ukeire）
pass 2   minimum shanten > 0 かつ（最小shanten内で最大ukeireの）finalist数 >= 2
         のときだけ、finalistsだけをrequestして再build
```

#187の`build_discard_candidate_features()`を最大2回呼ぶだけで、generic lazy
feature framework / cacheは持たない。finalist集合は#87 TwoStepが2段目を評価する
candidate集合と一致する（testで固定）。

### Dataset / model / training / artifact

- dataset: O0 DISCARD decisionだけを1 row = 1 decisionでmaterializeし、
  WIN / RIICHI / RESPONSE rowはsplit別の除外countとして残す。candidateは
  paddingなしのragged / flattened rowで、`decisions.jsonl`がtyped candidate
  semantic・offset・teacher candidate indexを持つ。strict readはtyped semantic
  から再encodeした値とpayloadの一致まで照合する。teacher actionがlegal
  `DiscardAction` candidateへ一意に解決できなければfail closedする
- model: `hidden = ReLU(context_layer(shared) + candidate_layer(candidate))`、
  `score = output_layer(hidden)`（concat入力のLinear -> ReLU -> Linearと同値）
- training: 1 decisionのlegal discard candidateだけで正規化するsoftmax cross
  entropy。TRAINでoptimize、SELECTのcross entropy最小epochを採用。OFFLINE-EVALは
  trainingにもselectionにも使わない
- artifact: source / dataset / shared feature / candidate feature / encoding /
  request policy / label / model / training / seed / selected epoch / weights
  digestをbindし、mismatchはload時にfail closedする。
  `load_candidate_scorer_policy_factory(path)`がtop-level importableな
  `CandidateScorerRuntime`を返し、Arenaは`PolicySpec(identity=..., factory=runtime)`
  としてそのまま使える

### TRAIN / SELECT / OFFLINE-EVAL roles と frozen offline gate

```text
TRAIN         optimization
SELECT        epoch / artifact candidate selection
freeze        model / encoding / second-step request policy / thresholds
OFFLINE-EVAL  qualificationのためにexactly once
```

thresholdはOFFLINE-EVALを見る前に`OFFLINE_GATE`として固定し、Issue #189へ
記録した（fingerprint `ebde74b5…15258f9`）。

| metric | required |
| --- | --- |
| legality | 1.000 |
| deterministic win guard | 1.000 |
| deterministic immediate-riichi guard | 1.000 |
| O0 no-call guard | 1.000 |
| post-discard shanten-stage agreement | >= 0.99 |
| conditional current-ukeire agreement | >= 0.95 |
| conditional second-step agreement | >= 0.90 |
| candidate top-1 agreement vs teacher | diagnostic only |

minimum support（win 30 / riichi 60 / response 100 / scorer decisions 1000 /
conditional ukeire 500 / conditional second-step 250）を下回るgateは推測で
PASSにせず`STOP / INVALID`とする。semantic agreementはlearner選択candidateと
teacher選択candidateの#187 semantic値を比較するguardrailであり、upstream stageの
不一致をlater stageの成功として数えない。

### Runtime preflight

actual L0.2 serving pathを`tools/benchmark_candidate_scorer.py`で測定した
（accepted source recordのscorer decisionを100件ごとにsample、675 decision、
CPython 3.14.6 / Windows / torch 2.13.0+cpu、decisionあたりms）。

| stage | median | mean | p95 | max |
| --- | --- | --- | --- | --- |
| pass-1 candidate build | 15.1 | 13.8 | 18.8 | 36.6 |
| two-pass finalist candidate build | 18.8 | 71.9 | 305.0 | 885.4 |
| candidate numeric encoding | 0.05 | 0.06 | 0.09 | 0.57 |
| shared context（#184） | 0.10 | 0.11 | 0.16 | 0.90 |
| torch forward | 0.39 | 0.44 | 0.79 | 1.38 |
| Learned Policy `decide()` total | 20.3 | 72.6 | 295.8 | 881.9 |

finalist second-stepが必要なdecisionは約48%で、costのほぼ全てがtwo-pass
candidate buildにある。all-candidate second-step（#187: 224〜1189ms）より
十分小さく、two-pass policyは不合理ではない（REFORMULATE不要）。

dataset materializationの実測は90,579 source decision / 67,996 scorer decision
を4,918秒（約82分、13.8 scorer decision/秒、single process）でmaterializeし、
sampleからのprojection（4,899秒）と一致した。CI wall-clock thresholdは作らない。

### Terminal result

```text
SEMANTIC OFFENSE SCORER NOT QUALIFIED — OFFLINE
```

frozen artifactとdataset。

```text
candidate dataset   12092740c4d22b37bf3c3fbe6bb0ad771033bf823b78b80e30867b0553df985a
artifact            36d77f8cc1ba28df2859be491be538a1ed5b6a62126b3b52992ecee39dd1d4e0
selected epoch      13 / 20（SELECT cross entropy最小）
```

| split | source decisions | scorer decisions | candidates | excluded win / riichi / response |
| --- | --- | --- | --- | --- |
| TRAIN | 66,239 | 49,721 | 514,959 | 934 / 1,764 / 13,820 |
| SELECT | 12,702 | 9,560 | 98,878 | 180 / 345 / 2,617 |
| OFFLINE-EVAL | 11,638 | 8,715 | 89,699 | 178 / 350 / 2,395 |

OFFLINE-EVAL（frozen gate `ebde74b5…`、全support minimum充足）。

| metric | observed | required | support |
| --- | --- | --- | --- |
| legality | 1.000 | 1.000 | 11,638 |
| deterministic win guard | 1.000 | 1.000 | 178 |
| deterministic immediate-riichi guard | 1.000 | 1.000 | 350 |
| O0 no-call guard | 1.000 | 1.000 | 2,395 |
| shanten-stage agreement | 0.9991 | >= 0.99 | 8,715 |
| conditional current-ukeire agreement | **0.9225** | >= 0.95 | 8,707 |
| conditional second-step agreement | 0.9579 | >= 0.90 | 4,136 |
| candidate top-1 agreement（diagnostic） | 0.8538 | — | 8,715 |

参考: SELECTでは shanten 0.9976 / ukeire 0.9263 / second-step 0.9751 /
top-1 0.8731、final-epoch train cross entropy 0.019に対しSELECT cross entropy
0.380であり、同じ不足がSELECTでも見えていた。#331（flat 802-way）の同じ
OFFLINE-EVALではwin recall 0.607、shanten 0.793、ukeire 0.759、no-call 0.970
だったため、deterministic guardとcandidate semantic encodingでwin / riichi /
no-call / shantenは解消したが、同shanten内のcurrent-ukeire選択が0.95に届かない。

Issue #189の規則に従い、同じIssue内でretrain / threshold / feature / modelの
変更によるrescueは行わない。次の仮説は別Issueとする。

記録上の注意: OFFLINE-EVALの最初の実行は、結果のstdout出力がWindows console
encoding（cp932）でem dashを含むoutcome文字列をencodeできず失われた（metricsは
未観測）。CLIをUTF-8出力へ修正し、同じfrozen artifactに対する同じ決定的評価を
出力記録のためだけに再実行した。retrain、threshold / feature / model変更は
行っていない。

## Semantic-envelope Learned Offense Policy（L0.2a、#191）

```text
selection policy     lisjong-offense-l0.2-semantic-envelope-v1
runtime identity     value_digest({candidate_scorer_artifact, selection_policy})
実装                  src/lisjong/learning/envelope_policy.py,
                     envelope_diagnostics.py, tools/replay_semantic_envelope.py
test                 tests/test_learning_semantic_envelope.py
```

Issue #189のfailure（conditional current-ukeire 0.9225）を再学習で救済せず、
exactに計算できる牌効率hierarchyをserving policyのselection constraintにする
engineering integrationである。#184 shared feature、#187 candidate feature、
そして#189のencoding / request policy / dataset / model / artifact schema / frozen
artifact・result、`LearnedCandidateOffensePolicy` / `CandidateScorerRuntime`の
identityとbehaviorは変更しない。

```text
O0 guard（#189と同一）
normal discard
    -> build_scorer_candidates()（#189 two-pass request policy）
    -> S1 minimum post-discard shanten
    -> S2 maximum current ukeire
    -> S3 maximum second-step score
         minimum shanten == 0   S3なし、全survivorはNOT_APPLICABLE
         elif survivor == 1件   S3なし、NOT_MATERIALIZED
         else                   全survivorがEVALUATEDであることを要求
    -> survivor 1件   そのlegal DiscardAction（scorerは実行しない）
       survivor複数   full candidate tupleへ#189 scorerを適用し、argmaxだけを
                      survivorへ限定（同点はcanonical順で最初）
```

期待statusと異なるsurvivorはfail closedし、未評価candidateをscore 0として扱わず、
評価済みcandidateだけの部分比較もしない。scorerを省略した場合はscore生成も
非有限検査も行わない（仕様）。

S1 → S3は`TwoStepUkeirePolicy`の通常打牌規則と同じhierarchyである。したがって
constant scorerではTwoStepと同じaction objectを返し（oracle testで固定）、
learned scorerがTwoStepと異なる打牌を選べるのはS1 / S2 / S3がすべて同値な
residual candidate間だけである。semantic regret 0はqualification evidenceでは
なくconstruction invariantの確認である。

runtime identityは#189 artifact identityを返さず、selection policy identityと
artifact identityから決定的に合成する（#189 artifactでは
`f412508d042aa890180dfa5a9a7c529dd07fc0ac3aa0db5b79c83e5afd532a72`）。

### Bounded engineering replay

Issue #189のOFFLINE-EVALは新しいholdoutとして再利用しない。#189 frozen artifact
（`36d77f8c…`）と同じsource record（`8c1899c5…`、retained #332 scientific
archiveから展開しidentityを照合）で、既観測のTRAIN / SELECTだけを使った。
`tools/replay_semantic_envelope.py`はartifactのtraining / selection split以外を
拒否し、referenceとして`TwoStepUkeirePolicy`を注入する。regretは
`semantic_envelope_survivors()`を使わずcandidate semantic値から独立に再計算する。

```text
python tools/replay_semantic_envelope.py --artifact <candidate-artifact> \
    --source-record <source-record> --split SELECT
python tools/replay_semantic_envelope.py ... --split TRAIN --sample-every 5
```

| invariant | SELECT（全件） | TRAIN（1/5 stride） |
| --- | --- | --- |
| decisions / scorer decisions | 12,702 / 9,560 | 13,248 / 9,941 |
| legality | 1.000 | 1.000 |
| win / riichi / no-call guard | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| shanten / ukeire / second-step max regret | 0 / 0 / 0 | 0 / 0 / 0 |
| second-step regret support | 4,621 | 4,836 |
| constant-scorer oracle == TwoStep action object | 1.000 | 1.000 |
| TwoStepとの不一致のうちsurvivor外 | 0 | 0 |

Residual-choice diagnostics（gateではない）。

| diagnostic | SELECT | TRAIN |
| --- | --- | --- |
| survivor 1件 | 0.537 | 0.542 |
| survivor >= 2（= scorer invoked） | 0.463 | 0.458 |
| survivor >= 2のうち単一tile type（赤5/通常5、ツモ切り/手出しのみ） | 0.046 | 0.058 |
| scorerがcanonical先頭以外を選んだ割合（survivor >= 2） | 0.0864 | 0.0002 |
| combined policyとTwoStepの一致 | 0.9698 | 0.9999 |

survivor数分布（SELECT）: 1: 5,129 / 2: 2,254 / 3: 1,255 / 4: 600 / 5: 82 /
6: 171 / 7: 4 / 8: 60 / 10: 5。代表sampleは`6m 9p 2z 6z`、`3z 7z`、
`1s 5s 8s 7z`、`9s 6z*`のように異なるtile typeの完全同値candidateが大半である。

読み方。

- strict envelopeでも約46%のnormal discardで2件以上の完全同値candidateが
  残り、その約95%は異なるtile typeである。learned residual choiceの自由度は
  「赤5 / ツモ切りの区別」だけに縮退していない
- ただし#189 scorerはteacher（canonical tie-break）を模倣したものであり、
  TRAINではresidual choiceの99.98%でcanonical先頭を選ぶ。SELECTの8.6%は
  tie-break再現の汎化誤差であり、outcome上の意味を持つ選択ではない。
  residual choiceを改善するにはL0.3のoutcome-aware objectiveが必要である

### L0.2a runtime

SELECTの全scorer decision（9,560件）を単独processで測定した（CPython 3.14.6 /
Windows / torch 2.13.0+cpu、decisionあたりms）。

| stage | median | mean | p95 | max |
| --- | --- | --- | --- | --- |
| semantic envelope（S1〜S3 survivor決定） | 0.004 | 0.005 | 0.008 | 1.8 |
| Semantic-envelope Policy `decide()` total | 15.5 | 58.1 | 251.8 | 898.4 |

envelopeの追加costは無視できる。survivor 1件のdecision（54%）ではscorerを
省略するため、#189の`decide()`（20.3 / 72.6 / 295.8 / 881.9）と同等かそれ以下で
あり、costの中心は引き続きtwo-pass candidate buildである。replay実行中の
discard max（114s / 144s）はreplayを並列実行中の一時的なPCスタンバイによる
wall-clock外れ値であり、上記の単独測定では再現しない。CI timing thresholdは
作らない。

### L0.2a terminal result

```text
SEMANTIC OFFENSE ENVELOPE READY
```

READYは#189 learned scorer単体が後からqualifiedになったことを意味しない。
deterministic exact semantics + frozen #189 residual scorerのcombined serving
policyが、L0 offenseのsemantic safety boundaryを満たしたことを意味する。

## L0.3 step B — exploration selector / baseline runtime / outcome consumer（#193）

lisjong-project#79 A-freeze（A1〜A4）のlisjong所有分を、source生成に必要な最小の
seamとして実装する。producerはlisjong-arena#359である。dataset publication、
trainer / MSE objective、Q artifact / Q runtimeは#79 step Dで扱う。

```text
実装    src/lisjong/learning/residual_baseline.py
        src/lisjong/learning/residual_exploration.py
        src/lisjong/learning/outcome_source.py
test    tests/test_learning_residual_exploration.py
        tests/test_learning_outcome_source.py
```

**Constant-zero baseline runtime（A2）**。`ConstantResidualRuntime`はfull candidate
tupleの各entryへ`0.0`を返すscorerで、`create_policy()`は既存の
`SemanticEnvelopeOffensePolicy(self)`を返す（新しいPolicy classは作らない）。
artifactもML runtimeも不要。identityは
`value_digest({"residual_scorer": "lisjong-offense-l0.3-constant-zero-residual-scorer-v1", "selection_policy": SEMANTIC_ENVELOPE_IDENTITY})`
（`CONSTANT_RESIDUAL_RUNTIME_IDENTITY`）であり、#189 / #191のartifact-backed
runtime identityとは衝突しない。constant scorerなのでactionはTwoStepと同じobjectになる。

**Focal residual exploration selector（A3）**。`select_residual_exploration(decision,
exploration_token)`はgeneration専用のpure functionで、Policy Protocolは実装しない。

```text
token        lowercase 64-hex以外はfail closed
O0 guard     deterministic_guard_action()（tokenは使わない）
DISCARD      build_scorer_candidates() -> semantic_envelope_survivors()
             survivor 1件   そのsurvivor
             survivor k>=2  survivors[int(token, 16) % k]
```

PRNGは使わない。token導出はArena（`lisjong-arena-l0.3-focal-decision-token-sha256-v1`）
が所有し、lisjongは再実装しない。返り値はO0 kind、`decision.legal_actions`側の
action object、full candidate tuple、canonical順survivor、`selected_candidate_index`、
`bucket`を持つ。behavior identityは
`lisjong-offense-l0.3-focal-uniform-residual-exploration-v1`、hash-to-bucket rule、
そして#191 envelope identityをbindしたdigest（`RESIDUAL_EXPLORATION_RUNTIME_IDENTITY`）である。

**Outcome source consumer + target（A1）**。`read_outcome_source()`はArena-owned
`arena-offense-l0.3-focal-outcome-source-v1`をstrict readする。`lisjong_arena`は
importしない。wire layout（`manifest.json` + `game-NNN/kyokus.jsonl` /
`focal-decisions.jsonl`）とfail-closed条件の一覧はmodule docstringを正本とする。
focal decisionごとにselectorを再計算し、guard action、producerが記録したsurvivor
action列、bucket選択がすべて一致しなければfail closedする。

```text
OUTCOME_TARGET_IDENTITY  lisjong-offense-l0.3-focal-kyoku-point-delta-1000-v1
target_q = (points_after_kyoku[focal] - points_before_kyoku[focal]) / 1000.0
```

`points_after_kyoku`はArenaが記録した**hanchan最終調整前**の事実値をそのまま使う。
lisjongは精算を再実装せず、backendの最終調整を逆算もしない。
`hanchan_final_scores`はaudit factとしてshapeだけを検証し、targetには使わない。

`build_outcome_targets()`はeligible row（DISCARD かつ survivor >= 2）だけへ、
選択したcandidateの`target_q`を付ける。選ばれなかったsurvivorにはtargetを付けない。
除外件数（win / riichi / response / single_survivor）も返す。
`summarize_outcome_targets()`はC0 pilot用の最小deterministic summaryを返し、
統計的なqualificationは行わない。

## Optional ML dependency boundary

```text
core (policy_contract / hand_evaluation / belief / action_vocabulary / policies)
    -> ML runtime不要

lisjong.learning
    -> importはML runtime不要
    -> source record読み取り / feature materialization / dataset生成 /
       artifact読み取りはML runtimeなしで動く
    -> trainingとLearnedPolicy inferenceだけがoptional extra `lisjong[ml]`を
       lazy importで要求する
```

`tests/test_learning_dependency_boundary.py`が、core / Learning importが
ML runtimeなしで成立すること、および training / inferenceがML runtime不在時に
`MissingLearningDependencyError`でfail closedすることを固定する。

## Executable entry point

```text
python -m lisjong.learning materialize-dataset \
    --source-record <source-record> --output <dataset>

python -m lisjong.learning train \
    --dataset <dataset> --output <artifact> \
    --train-split TRAIN --validation-split SELECT

python -m lisjong.learning verify-artifact --artifact <artifact>

python -m lisjong.learning materialize-candidate-dataset \
    --source-record <source-record> --output <candidate-dataset>

python -m lisjong.learning train-candidate-scorer \
    --dataset <candidate-dataset> --output <candidate-artifact> \
    --train-split TRAIN --select-split SELECT

python -m lisjong.learning verify-candidate-artifact --artifact <candidate-artifact>

python -m lisjong.learning evaluate-candidate-scorer \
    --artifact <candidate-artifact> --source-record <source-record> \
    --split OFFLINE-EVAL
```

`train` / `train-candidate-scorer` / `evaluate-candidate-scorer`だけがoptional
ML runtimeを必要とする。`evaluate-candidate-scorer`はartifactのtraining /
selection splitを評価splitに指定するとfail closedし、source record identityが
artifactと一致することを要求する。

## Non-goals（このsliceでは扱わない）

- Arena ML codeのbulk migration、historical research moduleの削除
- generic ML platform / dataset registry / experiment tracking
- self-play / RL / defense / calls / open-hand learning
- Champion promotion、#331/#332 locked scientific protocolの変更
- `arena-policy-input-feature-v1`、#331 dataset/checkpoint identity、
  #331 teacher semantics/thresholds、#332 execution result/protocolの改名・再定義
- generic Policy router / framework、universal CandidateEvaluation、generic
  Dataset / Artifact / Trainer framework
- L0.2の範囲外: formal Arena strength / paired evaluation、outcome-aware / RL /
  Q learning（L0.3）、riichi vs dama、learned calls、defense / push-fold、
  structured tile-axis encoder、current `TwoStepUkeirePolicy`によるrelabel
