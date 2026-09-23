"""L0.2 candidate scorerのbounded supervised warm start。

Issue #189に対応する。1 decision内のlegal discard candidateだけを対象にした
softmax cross entropyで、teacherが実際に選んだcandidateを学習する。

```text
strict-read candidate dataset
    -> TRAIN split（optimization）/ SELECT split（epoch selection）
    -> scalar scorer + per-decision softmax cross entropy
    -> deterministic Adam + fixed epoch budget
    -> SELECT cross entropy最小のepoch weights
    -> immutable candidate scorer artifact
```

candidateはragged / flattenedのまま扱う。各candidateのscoreは所属decisionの
candidate集合の中だけで正規化され、他decisionのcandidate、illegal action、
paddingへ確率質量を与えない（paddingは存在しない）。

OFFLINE-EVAL splitはtrainingにもepoch selectionにも使わない。trainerは
hanchan / decisionをsplit間で再配分しない。

意図的に持たないもの: generic trainer / experiment framework、hyperparameter
search、scheduler、HPO、GPU / distributed実行。torchはoptional extra
（`lisjong[ml]`）でありlazy importする。CPU / single threadのprocess全体設定は
Issue #184 trainerと同じくtraining entry pointとして明示的に行う。
"""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

from lisjong.learning.candidate_artifact import (
    LoadedCandidateScorerArtifact,
    write_candidate_artifact,
)
from lisjong.learning.candidate_dataset import (
    CANDIDATE_TRAINING_OBJECTIVE,
    CandidateDataset,
)
from lisjong.learning.candidate_diagnostics import SemanticAgreement
from lisjong.learning.candidate_encoding import CANDIDATE_ENCODING_DIMENSION
from lisjong.learning.candidate_model import (
    CandidateScorerConfig,
    build_scorer_module,
    score_candidates,
    scorer_module_weights,
)
from lisjong.learning.errors import TrainingError
from lisjong.learning.features import FEATURE_DIMENSION
from lisjong.learning.model import require_torch
from lisjong.learning.training import (
    MAX_EPOCHS,
    MAX_SEED,
    OPTIMIZER,
    TRAINING_FRAMEWORK,
    _configure_runtime,
    _normalize_splits,
)


@dataclass(frozen=True, slots=True)
class CandidateScorerTrainingConfig:
    """1回のbounded candidate scorer trainingを完全に決める不変の実行条件。

    `batch_size`はdecision数である（candidate数ではない）。
    """

    train_splits: tuple[str, ...]
    select_splits: tuple[str, ...]
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    seed: int = 0
    model: CandidateScorerConfig = field(default_factory=CandidateScorerConfig)

    def __post_init__(self) -> None:
        train_splits = _normalize_splits(self.train_splits, "train_splits")
        select_splits = _normalize_splits(self.select_splits, "select_splits")
        if not train_splits:
            raise ValueError("train_splits must not be empty")
        if not select_splits:
            raise ValueError("select_splits must not be empty")
        if set(train_splits) & set(select_splits):
            raise ValueError("train_splits and select_splits must be disjoint")
        if type(self.epochs) is not int or not 1 <= self.epochs <= MAX_EPOCHS:
            raise ValueError(f"epochs must be an int in 1..{MAX_EPOCHS}")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive int")
        if (
            type(self.learning_rate) is not float
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be a finite positive float")
        if (
            type(self.weight_decay) is not float
            or not isfinite(self.weight_decay)
            or self.weight_decay < 0.0
        ):
            raise ValueError("weight_decay must be a finite non-negative float")
        if type(self.seed) is not int or not 0 <= self.seed <= MAX_SEED:
            raise ValueError(f"seed must be an int in 0..{MAX_SEED}")
        if not isinstance(self.model, CandidateScorerConfig):
            raise TypeError("model must be a CandidateScorerConfig")
        object.__setattr__(self, "train_splits", train_splits)
        object.__setattr__(self, "select_splits", select_splits)

    def to_value(self, *, selected_epoch: int, framework_version: str, diagnostics):
        """artifactへbindするtraining blockを返す。

        SELECT splitは#184 artifactと同じ`validation_splits` fieldへ記録する。
        """
        return {
            "batch_size": self.batch_size,
            "diagnostics": dict(sorted(diagnostics.items())),
            "epochs": self.epochs,
            "framework": {"name": TRAINING_FRAMEWORK, "version": framework_version},
            "learning_rate": self.learning_rate,
            "objective": CANDIDATE_TRAINING_OBJECTIVE,
            "optimizer": OPTIMIZER,
            "seed": self.seed,
            "selected_epoch": selected_epoch,
            "train_splits": list(self.train_splits),
            "validation_splits": list(self.select_splits),
            "weight_decay": self.weight_decay,
        }


def _split_decisions(
    dataset: CandidateDataset, splits: tuple[str, ...], field_name: str
) -> tuple[int, ...]:
    available = dataset.split_counts()
    missing = [name for name in splits if name not in available]
    if missing:
        raise TrainingError(
            f"{field_name} references splits without scorer decisions: "
            f"{sorted(missing)}"
        )
    indices = dataset.row_indices_for_splits(splits)
    if not indices:
        raise TrainingError(f"{field_name} selected no scorer decisions")
    return indices


class _Tensors:
    """datasetのflat payloadをtorch tensorとして1回だけ保持する。"""

    def __init__(self, torch, dataset: CandidateDataset) -> None:
        self.context = torch.frombuffer(
            bytearray(dataset.context.tobytes()), dtype=torch.float32
        ).reshape(-1, FEATURE_DIMENSION)
        self.candidates = torch.frombuffer(
            bytearray(dataset.candidates.tobytes()), dtype=torch.float32
        ).reshape(-1, CANDIDATE_ENCODING_DIMENSION)
        self.offsets = torch.tensor(
            [row.candidate_offset for row in dataset.rows], dtype=torch.long
        )
        self.counts = torch.tensor(
            [row.candidate_count for row in dataset.rows], dtype=torch.long
        )
        self.teacher = torch.tensor(
            [row.teacher_candidate_index for row in dataset.rows], dtype=torch.long
        )


def _batch_scores(torch, module, tensors: _Tensors, batch):
    """decision batchのflatなscoreと、decision-local indexを返す。"""
    counts = tensors.counts.index_select(0, batch)
    starts = tensors.offsets.index_select(0, batch)
    local_starts = torch.cumsum(counts, 0) - counts
    total = int(counts.sum())
    decision_index = torch.repeat_interleave(torch.arange(len(batch)), counts)
    within = torch.arange(total) - local_starts.repeat_interleave(counts)
    candidate_index = starts.repeat_interleave(counts) + within
    scores = score_candidates(
        module,
        tensors.context.index_select(0, batch),
        tensors.candidates.index_select(0, candidate_index),
        decision_index,
    )
    return scores, decision_index, local_starts, within


def _per_decision_loss(torch, scores, decision_index, local_starts, teacher, size):
    """decisionごとのsoftmax cross entropy（そのdecisionのcandidateだけで正規化）。"""
    shift = torch.full((size,), float("-inf")).scatter_reduce(
        0, decision_index, scores.detach(), reduce="amax", include_self=True
    )
    shifted = scores - shift.index_select(0, decision_index)
    sums = torch.zeros(size).index_add(0, decision_index, shifted.exp())
    teacher_scores = shifted.index_select(0, local_starts + teacher)
    return sums.log() - teacher_scores


def _argmax_positions(torch, scores, decision_index, within, size):
    """decisionごとの最大score candidateのlocal位置を返す。

    同点はcanonical順で最初のcandidate（最小local位置）を選ぶ。
    """
    best = torch.full((size,), float("-inf")).scatter_reduce(
        0, decision_index, scores, reduce="amax", include_self=True
    )
    is_best = scores == best.index_select(0, decision_index)
    # 各decisionは1つ以上のcandidateを持つため、全local位置より大きい値を
    # 「最大scoreでない」印として使える。
    beyond = int(within.max()) + 1
    positions = torch.where(is_best, within, torch.full_like(within, beyond))
    return torch.full((size,), beyond, dtype=torch.long).scatter_reduce(
        0, decision_index, positions, reduce="amin", include_self=True
    )


def _evaluate(torch, module, tensors: _Tensors, indices, batch_size: int):
    """平均cross entropyと、decisionごとのlearner選択位置を返す。"""
    selector = torch.tensor(list(indices), dtype=torch.long)
    total_loss = 0.0
    chosen: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(selector), batch_size):
            batch = selector[start : start + batch_size]
            scores, decision_index, local_starts, within = _batch_scores(
                torch, module, tensors, batch
            )
            if not bool(torch.isfinite(scores).all()):
                raise TrainingError("scorer produced a non-finite score")
            teacher = tensors.teacher.index_select(0, batch)
            loss = _per_decision_loss(
                torch, scores, decision_index, local_starts, teacher, len(batch)
            )
            total_loss += float(loss.sum())
            chosen.extend(
                _argmax_positions(
                    torch, scores, decision_index, within, len(batch)
                ).tolist()
            )
    return total_loss / len(selector), chosen


def _select_diagnostics(dataset, indices, chosen) -> dict[str, float | int]:
    agreement = SemanticAgreement()
    top1 = 0
    for index, position in zip(indices, chosen, strict=True):
        row = dataset.rows[index]
        agreement.add(row.teacher_candidate, row.candidates[position])
        top1 += int(position == row.teacher_candidate_index)
    values = {f"select_{name}": value for name, value in agreement.to_value().items()}
    values["select_candidate_top1_agreement"] = top1 / len(indices)
    return values


def train_candidate_scorer(
    dataset: CandidateDataset,
    config: CandidateScorerTrainingConfig,
    destination: str | Path,
) -> LoadedCandidateScorerArtifact:
    """bounded candidate scorer trainingを実行し、immutable artifactを書き出す。

    SELECT split上の平均cross entropyが最小のepochのweightsを採用する
    （事前に固定したselection rule）。採用epochとSELECT diagnosticsを
    artifactへ記録する。
    """
    if not isinstance(dataset, CandidateDataset):
        raise TrainingError("dataset must be a strict-read CandidateDataset")
    if not isinstance(config, CandidateScorerTrainingConfig):
        raise TrainingError("config must be a CandidateScorerTrainingConfig")

    train_indices = _split_decisions(dataset, config.train_splits, "train_splits")
    select_indices = _split_decisions(dataset, config.select_splits, "select_splits")

    torch = require_torch()
    _configure_runtime(torch, config.seed)
    tensors = _Tensors(torch, dataset)
    train_selector = torch.tensor(list(train_indices), dtype=torch.long)

    module = build_scorer_module(config.model)
    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    selected_weights = None
    selected_epoch = 0
    selected_objective = None
    for epoch in range(1, config.epochs + 1):
        module.train()
        permutation = train_selector.index_select(
            0, torch.randperm(len(train_selector), generator=generator)
        )
        for start in range(0, len(permutation), config.batch_size):
            batch = permutation[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            scores, decision_index, local_starts, _within = _batch_scores(
                torch, module, tensors, batch
            )
            loss = _per_decision_loss(
                torch,
                scores,
                decision_index,
                local_starts,
                tensors.teacher.index_select(0, batch),
                len(batch),
            ).mean()
            if not torch.isfinite(loss):
                raise TrainingError(f"training loss became non-finite at epoch {epoch}")
            loss.backward()
            optimizer.step()

        module.eval()
        objective, _chosen = _evaluate(
            torch, module, tensors, select_indices, config.batch_size
        )
        if not isfinite(objective):
            raise TrainingError(f"SELECT objective became non-finite at epoch {epoch}")
        if selected_objective is None or objective < selected_objective:
            selected_objective = objective
            selected_epoch = epoch
            selected_weights = scorer_module_weights(config.model, module)

    if selected_weights is None or selected_epoch == 0:
        raise TrainingError("training produced no selected weights")

    module.eval()
    final_train_ce, _chosen = _evaluate(
        torch, module, tensors, train_indices, config.batch_size
    )
    selected_module = build_scorer_module(config.model, selected_weights)
    select_ce, select_chosen = _evaluate(
        torch, selected_module, tensors, select_indices, config.batch_size
    )
    diagnostics = {
        "epochs_run": config.epochs,
        "final_epoch_train_cross_entropy": final_train_ce,
        "select_cross_entropy": select_ce,
        "select_decisions": len(select_indices),
        "train_decisions": len(train_indices),
        **_select_diagnostics(dataset, select_indices, select_chosen),
    }

    return write_candidate_artifact(
        destination,
        dataset=dataset,
        model_config=config.model,
        training=config.to_value(
            selected_epoch=selected_epoch,
            framework_version=str(torch.__version__),
            diagnostics=diagnostics,
        ),
        weights=selected_weights,
    )


__all__ = [
    "CandidateScorerTrainingConfig",
    "train_candidate_scorer",
]
