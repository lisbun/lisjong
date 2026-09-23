"""lisjong所有のcanonical Learning capability。

Issue #184のLearning L0 vertical sliceと、Issue #189のL0.2 candidate-centric
Learned Offense Policy（deterministic O0 guard + learned normal-discard candidate
scorer）、Issue #191のL0.2a semantic-envelope Policy（exact牌効率hierarchyを
selection constraintとし、#189 scorerはresidual choiceだけを担当）を提供する。

```text
versioned player-safe source record   (lisjong-arena producer)
    -> lisjong-owned feature materialization
    -> lisjong-owned versioned dataset
    -> bounded Behavior Cloning training
    -> immutable model artifact
    -> lisjong-owned LearnedPolicy inference
    -> 通常のlisjong Policy契約
```

ownership境界。

- `lisjong`が所有するもの: player-safe model-facing representation、feature
  identity / fingerprint、dataset semantics / materialization、teacher / label
  interpretation、training objective / trainer、model artifact contract、
  LearnedPolicy load / inference、Learning固有diagnostics
- `lisjong-arena`が所有し続けるもの: environment execution / observation、
  source record schema、Arena-owned seed / population provenance、matchup /
  rotation / holdout、game-strength evaluation、evaluation-specific artifact

`lisjong`から`lisjong-arena`へのruntime依存は作らない。Arena source recordは
file artifactとして読むだけである。

dependency境界。

```text
core（policy contract / hand / belief / action vocabulary）
    -> ML runtime不要

learning / learned inference
    -> optional extra `lisjong[ml]` + lazy import
```

本packageのimport自体はML runtimeを必要としない。source record読み取り、
feature materialization、dataset生成、artifact読み取りはML runtimeなしで
実行でき、trainingとlearned inferenceだけがlazy importでtorchを要求する。
"""

from lisjong.learning.artifact import (
    MODEL_ARTIFACT_SCHEMA,
    LoadedModelArtifact,
    load_model_artifact,
    write_model_artifact,
)
from lisjong.learning.candidate_artifact import (
    CANDIDATE_ARTIFACT_SCHEMA,
    LoadedCandidateScorerArtifact,
    load_candidate_artifact,
    write_candidate_artifact,
)
from lisjong.learning.candidate_dataset import (
    CANDIDATE_DATASET_SCHEMA,
    CANDIDATE_LABEL_SEMANTICS,
    CandidateDataset,
    CandidateDecisionRow,
    materialize_candidate_dataset,
    read_candidate_dataset,
)
from lisjong.learning.candidate_diagnostics import (
    OFFLINE_GATE,
    classify_offline_result,
    evaluate_candidate_policy,
)
from lisjong.learning.candidate_encoding import (
    CANDIDATE_ENCODING_DIMENSION,
    CANDIDATE_ENCODING_IDENTITY,
    SECOND_STEP_REQUEST_POLICY,
    build_scorer_candidates,
    candidate_encoding_fingerprint,
    encode_candidates,
)
from lisjong.learning.candidate_features import (
    CANDIDATE_FEATURE_IDENTITY,
    DiscardCandidateFeatures,
    SecondStepStatus,
    build_discard_candidate_features,
    legal_discard_candidates,
)
from lisjong.learning.candidate_model import (
    CANDIDATE_SCORER_ARCHITECTURE,
    CandidateScorerConfig,
)
from lisjong.learning.candidate_policy import (
    CandidateScorerDecision,
    CandidateScorerRuntime,
    LearnedCandidateOffensePolicy,
    load_candidate_scorer_policy_factory,
)
from lisjong.learning.candidate_training import (
    CandidateScorerTrainingConfig,
    train_candidate_scorer,
)
from lisjong.learning.dataset import (
    DATASET_SCHEMA,
    TEACHER_LABEL_SEMANTICS,
    DatasetRow,
    LearningDataset,
    materialize_dataset,
    read_dataset,
)
from lisjong.learning.envelope_diagnostics import (
    ENVELOPE_INVALID,
    ENVELOPE_READY,
    classify_semantic_envelope_result,
    evaluate_semantic_envelope_policy,
)
from lisjong.learning.envelope_policy import (
    SEMANTIC_ENVELOPE_IDENTITY,
    SemanticEnvelopeDecision,
    SemanticEnvelopeOffensePolicy,
    SemanticEnvelopeRuntime,
    load_semantic_envelope_policy_factory,
    semantic_envelope_runtime_identity,
    semantic_envelope_survivors,
)
from lisjong.learning.errors import (
    CandidateFeatureError,
    DatasetError,
    FeatureError,
    LearnedPolicyError,
    LearningError,
    MissingLearningDependencyError,
    ModelArtifactError,
    OutcomeSourceError,
    SourceRecordError,
    TrainingError,
    UnsupportedSourceSchemaError,
)
from lisjong.learning.features import (
    FEATURE_DIMENSION,
    FEATURE_IDENTITY,
    build_player_safe_feature,
    feature_fingerprint,
    feature_specification,
)
from lisjong.learning.model import MODEL_ARCHITECTURE, ModelConfig
from lisjong.learning.outcome_source import (
    EXPLORATION_TOKEN_IDENTITY,
    OUTCOME_OBJECTIVE_IDENTITY,
    OUTCOME_SOURCE_SCHEMA,
    OUTCOME_TARGET_IDENTITY,
    FocalOutcomeSource,
    OutcomeTargetRow,
    OutcomeTargets,
    build_outcome_targets,
    read_outcome_source,
    summarize_outcome_targets,
)
from lisjong.learning.policy import (
    LearnedOffensePolicy,
    LearnedPolicyRuntime,
    load_learned_policy_factory,
)
from lisjong.learning.residual_baseline import (
    CONSTANT_RESIDUAL_RUNTIME_IDENTITY,
    CONSTANT_RESIDUAL_SCORER_IDENTITY,
    ConstantResidualRuntime,
)
from lisjong.learning.residual_exploration import (
    RESIDUAL_EXPLORATION_BEHAVIOR_IDENTITY,
    ResidualExplorationDecision,
    select_residual_exploration,
)
from lisjong.learning.source_record import (
    EXPECTED_ALLOCATION_OWNER_REPOSITORY,
    SOURCE_RECORD_SCHEMA_V1,
    SOURCE_RECORD_SCHEMA_V2,
    SUPPORTED_SOURCE_RECORD_SCHEMAS,
    PlayerSafeSourceRecord,
    SourceDecision,
    SourceGame,
    read_source_record,
    seed_membership_identity,
    validate_allocation_binding,
    validate_allocation_bindings,
)
from lisjong.learning.training import (
    BehaviorCloningConfig,
    train_behavior_cloning,
)

__all__ = [
    "OutcomeSourceError",
    "EXPLORATION_TOKEN_IDENTITY",
    "OUTCOME_OBJECTIVE_IDENTITY",
    "OUTCOME_SOURCE_SCHEMA",
    "OUTCOME_TARGET_IDENTITY",
    "FocalOutcomeSource",
    "OutcomeTargetRow",
    "OutcomeTargets",
    "build_outcome_targets",
    "read_outcome_source",
    "summarize_outcome_targets",
    "CONSTANT_RESIDUAL_RUNTIME_IDENTITY",
    "CONSTANT_RESIDUAL_SCORER_IDENTITY",
    "ConstantResidualRuntime",
    "RESIDUAL_EXPLORATION_BEHAVIOR_IDENTITY",
    "ResidualExplorationDecision",
    "select_residual_exploration",
    "CANDIDATE_ARTIFACT_SCHEMA",
    "CANDIDATE_DATASET_SCHEMA",
    "CANDIDATE_ENCODING_DIMENSION",
    "CANDIDATE_ENCODING_IDENTITY",
    "CANDIDATE_FEATURE_IDENTITY",
    "CANDIDATE_LABEL_SEMANTICS",
    "CANDIDATE_SCORER_ARCHITECTURE",
    "DATASET_SCHEMA",
    "ENVELOPE_INVALID",
    "ENVELOPE_READY",
    "EXPECTED_ALLOCATION_OWNER_REPOSITORY",
    "FEATURE_DIMENSION",
    "FEATURE_IDENTITY",
    "MODEL_ARCHITECTURE",
    "MODEL_ARTIFACT_SCHEMA",
    "OFFLINE_GATE",
    "SECOND_STEP_REQUEST_POLICY",
    "SEMANTIC_ENVELOPE_IDENTITY",
    "SOURCE_RECORD_SCHEMA_V1",
    "SOURCE_RECORD_SCHEMA_V2",
    "SUPPORTED_SOURCE_RECORD_SCHEMAS",
    "TEACHER_LABEL_SEMANTICS",
    "BehaviorCloningConfig",
    "CandidateDataset",
    "CandidateDecisionRow",
    "CandidateFeatureError",
    "CandidateScorerConfig",
    "CandidateScorerDecision",
    "CandidateScorerRuntime",
    "CandidateScorerTrainingConfig",
    "DatasetError",
    "DatasetRow",
    "DiscardCandidateFeatures",
    "FeatureError",
    "LearnedCandidateOffensePolicy",
    "LearnedOffensePolicy",
    "LearnedPolicyError",
    "LearnedPolicyRuntime",
    "LearningDataset",
    "LearningError",
    "LoadedCandidateScorerArtifact",
    "LoadedModelArtifact",
    "MissingLearningDependencyError",
    "ModelArtifactError",
    "ModelConfig",
    "PlayerSafeSourceRecord",
    "SecondStepStatus",
    "SemanticEnvelopeDecision",
    "SemanticEnvelopeOffensePolicy",
    "SemanticEnvelopeRuntime",
    "SourceDecision",
    "SourceGame",
    "SourceRecordError",
    "TrainingError",
    "UnsupportedSourceSchemaError",
    "build_discard_candidate_features",
    "build_player_safe_feature",
    "build_scorer_candidates",
    "candidate_encoding_fingerprint",
    "classify_offline_result",
    "classify_semantic_envelope_result",
    "encode_candidates",
    "evaluate_candidate_policy",
    "evaluate_semantic_envelope_policy",
    "feature_fingerprint",
    "feature_specification",
    "legal_discard_candidates",
    "load_candidate_artifact",
    "load_candidate_scorer_policy_factory",
    "load_learned_policy_factory",
    "load_model_artifact",
    "load_semantic_envelope_policy_factory",
    "materialize_candidate_dataset",
    "materialize_dataset",
    "read_candidate_dataset",
    "read_dataset",
    "read_source_record",
    "seed_membership_identity",
    "semantic_envelope_runtime_identity",
    "semantic_envelope_survivors",
    "train_behavior_cloning",
    "train_candidate_scorer",
    "validate_allocation_binding",
    "validate_allocation_bindings",
    "write_candidate_artifact",
    "write_model_artifact",
]
