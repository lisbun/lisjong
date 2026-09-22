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
schema      arena-offense-o0-player-safe-source-record-v1
```

- 未対応schemaは`UnsupportedSourceSchemaError`でfail closedする
- manifest identity、game provenance（seed / split / lock identity /
  game mode）、row間のexecution ordering、actor seatとPolicyInput / legal
  actions / selected actionの整合をすべて検証する
- Arenaのencoded feature tensor（`features.f32`等の locked #331 scientific
  corpus）は読まない。読むのはtyped player-safe source recordだけである
- 返り値の`provenance()`が、下流のdataset / artifactへbindするsource identity
  とordered populationを提供する

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
population、split別row数をbindする。既存destinationは上書きせず、
`lisjong.learning.read_dataset()`がbindされた全identityとpayload digestを
照合してstrict readする。同じsource recordから再materializeすると同一identityに
なる。

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
repository外に置く。manifestは、source/dataset identity、feature
fingerprint、action vocabulary fingerprint、model architecture/config、
optimizer/training config、RNG seed、選択epoch、weights digest、lisjong
package versionと実際にimportされたlisjongソースのdigest（`source_digest`）を
bindする。`source_digest`はprovenance記録であり、load時に現在の
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
```

`train`だけがoptional ML runtimeを必要とする。

## Non-goals（このsliceでは扱わない）

- Arena ML codeのbulk migration、historical research moduleの削除
- generic ML platform / dataset registry / experiment tracking
- self-play / RL / defense / calls / open-hand learning
- Champion promotion、#331/#332 locked scientific protocolの変更
- `arena-policy-input-feature-v1`、#331 dataset/checkpoint identity、
  #331 teacher semantics/thresholds、#332 execution result/protocolの改名・再定義
