"""L0.2 candidate scorerのsecond-step request policyとmodel-facing numeric encoding。

Issue #189に対応する。Issue #187のtyped `DiscardCandidateFeatures`を、learned
candidate scorerが読むfixed-width float vectorへ写すpurpose-specific contract
である。shanten / ukeire / second-stepは`build_discard_candidate_features()`を
single sourceとし、再実装しない。

```text
DecisionContext
    -> two-pass finalist second-step request（SECOND_STEP_REQUEST_POLICY）
    -> #187 DiscardCandidateFeatures（canonical順）
    -> candidateごとのCANDIDATE_ENCODING_DIMENSION次元float vector
```

shared decision contextはIssue #184の`build_player_safe_feature()`をそのまま
再利用し、このmoduleは再定義しない。

## second-step request policy（two-pass finalists）

```text
pass 1   second-step requestなしでbuild（全candidateのshanten / current ukeire）
finalists
         minimum post-discard shantenのcandidateのうちcurrent ukeire最大のもの
pass 2   minimum shanten > 0 かつ finalist数 >= 2 のときだけ、
         finalistsだけをsecond-step requestして再build
         それ以外はpass 1の結果をそのまま使う
```

`TwoStepUkeireCandidateEvaluation`（Issue #87）をML schemaとして再利用する
ものではなく、TwoStep系teacherのtie-breakに必要な情報だけを#187 semanticから
materializeする。generic lazy-feature frameworkやcross-decision cacheは持たない。

## numeric encoding

candidate vectorは次のblockを順に連結する。decision内のrelative block
（gap）は同じdecisionのcandidate tupleだけから決まり、decision外の情報を
使わない。

```text
discard_tile_type           34  canonical 34牌種index one-hot
discard_red                  1  赤5なら1
tsumogiri                    1  ツモ切りなら1
post_discard_shanten         9  0..8 one-hot
shanten_gap                  1  (shanten - decision内最小shanten) / 8
current_ukeire               1  current ukeire / 136
ukeire_gap_within_shanten    1  (同shanten candidate内の最大ukeire - ukeire) / 136
second_step_status           3  EVALUATED / NOT_MATERIALIZED / NOT_APPLICABLE one-hot
second_step_score            1  EVALUATEDならscore / 1000、それ以外0
second_step_gap              1  EVALUATEDなら(EVALUATED内最大score - score) / 1000、
                                それ以外0
```

`second_step_status`のone-hotがavailabilityを表すため、評価済みscore 0
（EVALUATED + score 0）と`NOT_MATERIALIZED` / `NOT_APPLICABLE`は異なるvectorに
なる。layout・scale・axis・request policyを変える場合は既存identityを書き換えず
新しいidentityにする。
"""

from collections.abc import Sequence

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.learning._canonical import value_digest
from lisjong.learning.candidate_features import (
    CANDIDATE_FEATURE_IDENTITY,
    DiscardCandidateFeatures,
    SecondStepStatus,
    build_discard_candidate_features,
)
from lisjong.learning.errors import CandidateFeatureError
from lisjong.policy_contract import DecisionContext, DiscardAction
from lisjong.structural_efficiency import discard_action_sort_key

SECOND_STEP_REQUEST_POLICY = "lisjong-offense-l0.2-two-pass-finalist-second-step-v1"
"""上記two-pass finalist request policyのidentity。"""

CANDIDATE_ENCODING_IDENTITY = "lisjong-offense-l0.2-discard-candidate-encoding-v1"
"""このnumeric encoding contractのidentity。"""

TILE_TYPE_AXIS_SIZE = 34
MAX_POST_DISCARD_SHANTEN = 8
"""打牌後純手牌（13枚以下）の標準形shantenの物理上限。これを超えればfail closed。"""

SHANTEN_GAP_SCALE = float(MAX_POST_DISCARD_SHANTEN)
UKEIRE_SCALE = 136.0
SECOND_STEP_SCALE = 1000.0
SECOND_STEP_STATUS_AXIS = (
    SecondStepStatus.EVALUATED,
    SecondStepStatus.NOT_MATERIALIZED,
    SecondStepStatus.NOT_APPLICABLE,
)

_BLOCKS = (
    ("discard_tile_type", TILE_TYPE_AXIS_SIZE),
    ("discard_red", 1),
    ("tsumogiri", 1),
    ("post_discard_shanten", MAX_POST_DISCARD_SHANTEN + 1),
    ("shanten_gap", 1),
    ("current_ukeire", 1),
    ("ukeire_gap_within_shanten", 1),
    ("second_step_status", len(SECOND_STEP_STATUS_AXIS)),
    ("second_step_score", 1),
    ("second_step_gap", 1),
)


def _block_offsets() -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for name, size in _BLOCKS:
        offsets[name] = offset
        offset += size
    return offsets


CANDIDATE_BLOCK_OFFSETS = _block_offsets()
CANDIDATE_ENCODING_DIMENSION = sum(size for _name, size in _BLOCKS)

_STATUS_INDEX = {status: index for index, status in enumerate(SECOND_STEP_STATUS_AXIS)}
_SCALES = {
    "current_ukeire": UKEIRE_SCALE,
    "second_step_gap": SECOND_STEP_SCALE,
    "second_step_score": SECOND_STEP_SCALE,
    "shanten_gap": SHANTEN_GAP_SCALE,
    "ukeire_gap_within_shanten": UKEIRE_SCALE,
}


def candidate_encoding_specification() -> dict[str, object]:
    """fingerprint対象となるencoding contractのcanonical記述を返す。"""
    return {
        "axes": {
            "post_discard_shanten": list(range(MAX_POST_DISCARD_SHANTEN + 1)),
            "second_step_status": [status.value for status in SECOND_STEP_STATUS_AXIS],
            "tile_type": "lisjong-canonical-34-tile-type-index",
        },
        "blocks": [
            {
                "name": name,
                "offset": CANDIDATE_BLOCK_OFFSETS[name],
                "scale": _SCALES.get(name),
                "size": size,
            }
            for name, size in _BLOCKS
        ],
        "candidate_feature_identity": CANDIDATE_FEATURE_IDENTITY,
        "candidate_order": "discard_action_sort_key",
        "dimension": CANDIDATE_ENCODING_DIMENSION,
        "identity": CANDIDATE_ENCODING_IDENTITY,
        "second_step_request_policy": SECOND_STEP_REQUEST_POLICY,
    }


def candidate_encoding_fingerprint() -> str:
    """encoding contractのfingerprint。dataset / artifactはこの値をbindする。"""
    return value_digest(candidate_encoding_specification())


def encoding_block() -> dict[str, object]:
    """dataset / artifactがbindするcandidate encoding identity blockを返す。"""
    return {
        "candidate_feature_identity": CANDIDATE_FEATURE_IDENTITY,
        "dimension": CANDIDATE_ENCODING_DIMENSION,
        "fingerprint": candidate_encoding_fingerprint(),
        "identity": CANDIDATE_ENCODING_IDENTITY,
        "second_step_request_policy": SECOND_STEP_REQUEST_POLICY,
    }


def second_step_finalists(
    candidates: Sequence[DiscardCandidateFeatures],
) -> tuple[DiscardAction, ...]:
    """pass 1のcandidateからsecond-stepをrequestするfinalistsを返す。

    minimum shanten <= 0、またはfinalistが1つ以下なら空tupleを返す。
    """
    if not candidates:
        raise CandidateFeatureError("candidate tuple must not be empty")
    minimum_shanten = min(item.post_discard_shanten for item in candidates)
    if minimum_shanten <= 0:
        return ()
    minimum = tuple(
        item for item in candidates if item.post_discard_shanten == minimum_shanten
    )
    maximum_ukeire = max(item.current_ukeire_count for item in minimum)
    finalists = tuple(
        item.action for item in minimum if item.current_ukeire_count == maximum_ukeire
    )
    return finalists if len(finalists) >= 2 else ()


def build_scorer_candidates(
    decision: DecisionContext,
) -> tuple[DiscardCandidateFeatures, ...]:
    """two-pass finalist request policyでcandidate featureをbuildする。"""
    first_pass = build_discard_candidate_features(decision)
    finalists = second_step_finalists(first_pass)
    if not finalists:
        return first_pass
    return build_discard_candidate_features(decision, second_step_actions=finalists)


def _validate_candidates(
    candidates: Sequence[DiscardCandidateFeatures],
) -> tuple[DiscardCandidateFeatures, ...]:
    values = tuple(candidates)
    if not values:
        raise CandidateFeatureError("candidate tuple must not be empty")
    for item in values:
        if not isinstance(item, DiscardCandidateFeatures):
            raise CandidateFeatureError(
                "candidates must contain only DiscardCandidateFeatures"
            )
        if not 0 <= item.post_discard_shanten <= MAX_POST_DISCARD_SHANTEN:
            raise CandidateFeatureError(
                "post_discard_shanten is outside the encodable range "
                f"0..{MAX_POST_DISCARD_SHANTEN}"
            )
        if item.current_ukeire_count < 0:
            raise CandidateFeatureError("current_ukeire_count must be non-negative")
        if (
            item.second_step_ukeire_score is not None
            and item.second_step_ukeire_score < 0
        ):
            raise CandidateFeatureError("second_step_ukeire_score must be non-negative")
    keys = [discard_action_sort_key(item.action) for item in values]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise CandidateFeatureError(
            "candidates must be unique and in canonical discard order"
        )
    if len({item.action.actor for item in values}) != 1:
        raise CandidateFeatureError("candidates must share one actor")

    # encodingはrequest policyにbindされる。serving / trainingが異なる
    # materializationを混ぜないよう、second-step availabilityがtwo-pass
    # policyの結果と一致することを要求する。
    requested = frozenset(second_step_finalists(values))
    for item in values:
        if item.post_discard_shanten <= 0:
            expected = SecondStepStatus.NOT_APPLICABLE
        elif item.action in requested:
            expected = SecondStepStatus.EVALUATED
        else:
            expected = SecondStepStatus.NOT_MATERIALIZED
        if item.second_step_status is not expected:
            raise CandidateFeatureError(
                "second-step availability does not follow "
                f"{SECOND_STEP_REQUEST_POLICY!r}"
            )
    return values


def encode_candidates(
    candidates: Sequence[DiscardCandidateFeatures],
) -> tuple[tuple[float, ...], ...]:
    """1 decisionのcandidate tupleを、同じ順序のfloat vector tupleへ写す。"""
    values = _validate_candidates(candidates)
    minimum_shanten = min(item.post_discard_shanten for item in values)
    maximum_ukeire_by_shanten: dict[int, int] = {}
    for item in values:
        current = maximum_ukeire_by_shanten.get(item.post_discard_shanten)
        if current is None or item.current_ukeire_count > current:
            maximum_ukeire_by_shanten[item.post_discard_shanten] = (
                item.current_ukeire_count
            )
    evaluated_scores = [
        item.second_step_ukeire_score
        for item in values
        if item.second_step_status is SecondStepStatus.EVALUATED
    ]
    maximum_second_step = max(evaluated_scores) if evaluated_scores else None

    offsets = CANDIDATE_BLOCK_OFFSETS
    encoded: list[tuple[float, ...]] = []
    for item in values:
        vector = [0.0] * CANDIDATE_ENCODING_DIMENSION
        tile = item.action.tile
        vector[offsets["discard_tile_type"] + tile_type_index(tile.tile_type)] = 1.0
        vector[offsets["discard_red"]] = 1.0 if tile.is_red else 0.0
        vector[offsets["tsumogiri"]] = 1.0 if item.action.tsumogiri else 0.0
        vector[offsets["post_discard_shanten"] + item.post_discard_shanten] = 1.0
        vector[offsets["shanten_gap"]] = (
            item.post_discard_shanten - minimum_shanten
        ) / SHANTEN_GAP_SCALE
        vector[offsets["current_ukeire"]] = item.current_ukeire_count / UKEIRE_SCALE
        vector[offsets["ukeire_gap_within_shanten"]] = (
            maximum_ukeire_by_shanten[item.post_discard_shanten]
            - item.current_ukeire_count
        ) / UKEIRE_SCALE
        vector[
            offsets["second_step_status"] + _STATUS_INDEX[item.second_step_status]
        ] = 1.0
        if item.second_step_status is SecondStepStatus.EVALUATED:
            score = item.second_step_ukeire_score
            vector[offsets["second_step_score"]] = score / SECOND_STEP_SCALE
            vector[offsets["second_step_gap"]] = (
                maximum_second_step - score
            ) / SECOND_STEP_SCALE
        encoded.append(tuple(vector))
    return tuple(encoded)


__all__ = [
    "CANDIDATE_BLOCK_OFFSETS",
    "CANDIDATE_ENCODING_DIMENSION",
    "CANDIDATE_ENCODING_IDENTITY",
    "MAX_POST_DISCARD_SHANTEN",
    "SECOND_STEP_REQUEST_POLICY",
    "SECOND_STEP_SCALE",
    "SECOND_STEP_STATUS_AXIS",
    "UKEIRE_SCALE",
    "build_scorer_candidates",
    "candidate_encoding_fingerprint",
    "candidate_encoding_specification",
    "encode_candidates",
    "encoding_block",
    "second_step_finalists",
]
