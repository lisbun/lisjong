"""L0.3 selected-action Monte-Carlo Q regressionのone-shot trainer（Issue #200）。

#79 Step Dのlisjong側実装である。real-data trainingは#200のBlock
（lisbun/lisjong-arena#374のreadback PASSと、DP-1 / DP-2の#79でのfreeze）が
解除されるまで実行しない。本moduleはreal sourceを読むCLI entry pointを持たない。

```text
OutcomeQRows（SCIENTIFIC source、TRAIN / SELECT）
    -> outcome_q_preflight()のhard stopがあればtraining前にstop
    -> #189と同じcandidate-scorer MLP（hidden width 64）
    -> Q(shared_context, selected_candidate)だけを予測し、
       MSE(Q, target_q)を最小化（OUTCOME_OBJECTIVE_IDENTITY）
    -> deterministic Adam + 明示したepoch budget
    -> SELECT selected-action MSE最小のepoch（同値は最初）を採用
    -> immutable outcome-Q artifact
```

lossはrowごとにbehavior-selected candidate 1件だけへかかる。選ばれなかった
survivorは入力にもlossにも現れず、counterfactual labelとして扱わない。
Huber / SmoothL1、reward shaping、target clipping、scheduler、HPOは持たない。

hyperparameterは`OutcomeQTrainingConfig`で呼び出し側がすべて明示する。
scientific defaultは持たない（#200 DP-1、値は#79でfreezeする）。modelは
`OUTCOME_Q_MODEL`で固定であり、configから変更できない。

diagnostics（gateではない）はselected checkpointでのSELECT MSE、survivor間Q
spread、Q argmaxがcanonical-first baselineと異なる割合、prediction / target
calibration summaryである。CPU / single threadのprocess全体設定は#184 / #189
trainerと同じくtraining entry pointで明示的に行う。
"""

import statistics
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from lisjong.learning.candidate_encoding import CANDIDATE_ENCODING_DIMENSION
from lisjong.learning.candidate_model import (
    build_scorer_module,
    score_candidates,
    scorer_module_weights,
)
from lisjong.learning.errors import TrainingError
from lisjong.learning.features import FEATURE_DIMENSION
from lisjong.learning.model import require_torch
from lisjong.learning.outcome_q_artifact import (
    OUTCOME_Q_MODEL,
    LoadedOutcomeQArtifact,
    write_outcome_q_artifact,
)
from lisjong.learning.outcome_q_dataset import (
    SELECT_SPLIT,
    TRAIN_SPLIT,
    OutcomeQRows,
    outcome_q_preflight,
)
from lisjong.learning.outcome_source import OUTCOME_OBJECTIVE_IDENTITY
from lisjong.learning.training import (
    MAX_EPOCHS,
    MAX_SEED,
    OPTIMIZER,
    TRAINING_FRAMEWORK,
    _configure_runtime,
)


@dataclass(frozen=True, slots=True)
class OutcomeQTrainingConfig:
    """1回のoutcome-Q trainingを完全に決める不変の実行条件。

    すべてのfieldが必須である（#200 DP-1）。`batch_size`はrow数
    （= selected candidate数）である。
    """

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int

    def __post_init__(self) -> None:
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

    def to_value(self, *, selected_epoch: int, framework_version: str, diagnostics):
        """artifactへbindするtraining block。SELECTは`validation_splits`へ記録する。"""
        return {
            "batch_size": self.batch_size,
            "diagnostics": dict(sorted(diagnostics.items())),
            "epochs": self.epochs,
            "framework": {"name": TRAINING_FRAMEWORK, "version": framework_version},
            "learning_rate": self.learning_rate,
            "objective": OUTCOME_OBJECTIVE_IDENTITY,
            "optimizer": OPTIMIZER,
            "seed": self.seed,
            "selected_epoch": selected_epoch,
            "train_splits": [TRAIN_SPLIT],
            "validation_splits": [SELECT_SPLIT],
            "weight_decay": self.weight_decay,
        }


class _Tensors:
    """rowsのflat payloadをtorch tensorとして1回だけ保持する。"""

    def __init__(self, torch, rows: OutcomeQRows) -> None:
        self.context = torch.tensor(list(rows.context), dtype=torch.float32).reshape(
            -1, FEATURE_DIMENSION
        )
        self.candidates = torch.tensor(
            list(rows.candidates), dtype=torch.float32
        ).reshape(-1, CANDIDATE_ENCODING_DIMENSION)
        self.selected = torch.tensor(
            [
                offset + row.selected_candidate_index
                for offset, row in zip(rows.candidate_offsets, rows.rows, strict=True)
            ],
            dtype=torch.long,
        )
        self.target = torch.tensor(
            [row.target_q for row in rows.rows], dtype=torch.float32
        )


def _selected_q(torch, module, tensors: _Tensors, batch):
    """batch rowのselected candidateだけのQを返す（長さ = batch row数）。"""
    return score_candidates(
        module,
        tensors.context.index_select(0, batch),
        tensors.candidates.index_select(0, tensors.selected.index_select(0, batch)),
        torch.arange(len(batch)),
    )


def _mse(torch, module, tensors: _Tensors, indices, batch_size: int) -> float:
    selector = torch.tensor(list(indices), dtype=torch.long)
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(selector), batch_size):
            batch = selector[start : start + batch_size]
            q = _selected_q(torch, module, tensors, batch)
            if not bool(torch.isfinite(q).all()):
                raise TrainingError("outcome-Q model produced a non-finite value")
            total += float(((q - tensors.target.index_select(0, batch)) ** 2).sum())
    return total / len(selector)


def _row_scores(torch, module, tensors: _Tensors, rows: OutcomeQRows, index: int):
    """1 rowのfull candidate tupleのQを返す。"""
    offset = rows.candidate_offsets[index]
    count = len(rows.rows[index].candidates)
    with torch.inference_mode():
        scores = score_candidates(
            module,
            tensors.context[index : index + 1],
            tensors.candidates[offset : offset + count],
            torch.zeros(count, dtype=torch.long),
        )
    values = tuple(scores.tolist())
    if any(not isfinite(value) for value in values):
        raise TrainingError("outcome-Q model produced a non-finite value")
    return values


def _survivor_argmax(scores, survivors) -> int:
    """`SemanticEnvelopeOffensePolicy`と同じ規則（strict `>`、同値はcanonical先頭）。"""
    best = survivors[0]
    for index in survivors[1:]:
        if scores[index] > scores[best]:
            best = index
    return best


def _select_diagnostics(torch, module, tensors, rows, indices) -> dict[str, float]:
    spreads = []
    changed = 0
    predictions = []
    targets = []
    for index in indices:
        row = rows.rows[index]
        scores = _row_scores(torch, module, tensors, rows, index)
        survivor_scores = [scores[i] for i in row.survivors]
        spreads.append(max(survivor_scores) - min(survivor_scores))
        changed += int(_survivor_argmax(scores, row.survivors) != row.survivors[0])
        predictions.append(scores[row.selected_candidate_index])
        targets.append(row.target_q)
    return {
        "select_fraction_q_argmax_not_canonical_first": changed / len(indices),
        "select_mean_survivor_q_spread": statistics.fmean(spreads),
        "select_max_survivor_q_spread": max(spreads),
        "select_prediction_mean": statistics.fmean(predictions),
        "select_prediction_std": statistics.pstdev(predictions),
        "select_target_mean": statistics.fmean(targets),
        "select_target_std": statistics.pstdev(targets),
    }


def train_outcome_q(
    rows: OutcomeQRows,
    config: OutcomeQTrainingConfig,
    destination: str | Path,
) -> LoadedOutcomeQArtifact:
    """one-shot outcome-Q trainingを実行し、immutable artifactを書き出す。

    preflight hard stop、非有限loss / Q、既存destinationはfail closedする。
    """
    if not isinstance(rows, OutcomeQRows):
        raise TrainingError("rows must be materialized OutcomeQRows")
    if not isinstance(config, OutcomeQTrainingConfig):
        raise TrainingError("config must be an OutcomeQTrainingConfig")
    if Path(destination).exists():
        raise TrainingError(f"artifact destination already exists: {destination}")
    preflight = outcome_q_preflight(rows)
    if preflight["hard_stops"]:
        raise TrainingError(f"outcome-Q preflight hard stop: {preflight['hard_stops']}")
    train_indices = rows.row_indices(TRAIN_SPLIT)
    select_indices = rows.row_indices(SELECT_SPLIT)

    torch = require_torch()
    _configure_runtime(torch, config.seed)
    tensors = _Tensors(torch, rows)
    train_selector = torch.tensor(list(train_indices), dtype=torch.long)

    module = build_scorer_module(OUTCOME_Q_MODEL)
    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    selected_weights = None
    selected_epoch = 0
    selected_mse = None
    for epoch in range(1, config.epochs + 1):
        module.train()
        permutation = train_selector.index_select(
            0, torch.randperm(len(train_selector), generator=generator)
        )
        for start in range(0, len(permutation), config.batch_size):
            batch = permutation[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            q = _selected_q(torch, module, tensors, batch)
            loss = ((q - tensors.target.index_select(0, batch)) ** 2).mean()
            if not torch.isfinite(loss):
                raise TrainingError(f"training loss became non-finite at epoch {epoch}")
            loss.backward()
            optimizer.step()

        module.eval()
        mse = _mse(torch, module, tensors, select_indices, config.batch_size)
        if not isfinite(mse):
            raise TrainingError(f"SELECT MSE became non-finite at epoch {epoch}")
        # OUTCOME_Q_CHECKPOINT_RULE: 最小SELECT MSE、同値は最初のepoch。
        if selected_mse is None or mse < selected_mse:
            selected_mse = mse
            selected_epoch = epoch
            selected_weights = scorer_module_weights(OUTCOME_Q_MODEL, module)

    if selected_weights is None or selected_epoch == 0:
        raise TrainingError("training produced no selected weights")

    module.eval()
    final_train_mse = _mse(torch, module, tensors, train_indices, config.batch_size)
    selected_module = build_scorer_module(OUTCOME_Q_MODEL, selected_weights)
    diagnostics = {
        "epochs_run": config.epochs,
        "final_epoch_train_mse": final_train_mse,
        "select_mse": _mse(
            torch, selected_module, tensors, select_indices, config.batch_size
        ),
        "select_rows": len(select_indices),
        "selected_epoch_train_mse": _mse(
            torch, selected_module, tensors, train_indices, config.batch_size
        ),
        "train_rows": len(train_indices),
        **_select_diagnostics(torch, selected_module, tensors, rows, select_indices),
    }

    return write_outcome_q_artifact(
        destination,
        rows=rows,
        training=config.to_value(
            selected_epoch=selected_epoch,
            framework_version=str(torch.__version__),
            diagnostics=diagnostics,
        ),
        weights=selected_weights,
    )


__all__ = [
    "OutcomeQTrainingConfig",
    "train_outcome_q",
]
