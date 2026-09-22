"""lisjong-owned Learning pathのfail-closed例外階層。

Issue #184のLearning L0 vertical sliceは、source record、feature、dataset、
model artifact、learned inferenceのいずれの段でも、意味が確定しない入力を
推測で受け入れない。すべての失敗を例外としてcallerへ伝播させ、heuristic
fallbackやdefault値への丸めを行わない。

`lisjong.action_vocabulary`の`ActionVocabularyError`階層とは別の階層とする。
model-facing action indexのcodec契約と、Learning artifactのidentity契約は
別の責務であり、片方の例外をもう片方のvalidation根拠として扱わない。
"""


class LearningError(Exception):
    """lisjong-owned Learning pathのfail-closed基底例外。"""


class MissingLearningDependencyError(LearningError):
    """optional ML runtime（`lisjong[ml]`）が未installの場合。

    core importはML runtimeを要求しないため、trainingとlearned inferenceだけが
    この例外を発生させ得る。silentなCPU fallbackやheuristic代替は行わない。
    """


class SourceRecordError(LearningError):
    """player-safe source recordをstrictに読み取れない場合。"""


class UnsupportedSourceSchemaError(SourceRecordError):
    """source recordのschema / versionをこの実装が提供しない場合。"""


class FeatureError(LearningError):
    """player-safe featureをmaterializeできない場合。"""


class DatasetError(LearningError):
    """Learning datasetを生成・strict readできない場合。"""


class ModelArtifactError(LearningError):
    """model artifactを生成・strict loadできない場合。"""


class TrainingError(LearningError):
    """bounded BC trainingの前提条件が成立しない場合。"""


class LearnedPolicyError(LearningError):
    """learned inferenceがPolicy契約を満たせない場合。"""
