"""L0.3 step D — selected-action Monte-Carlo Q trainer（lisjong-project#79 §4〜§8）。

strict readしたSCIENTIFIC focal outcome sourceから、1回のfrozen Q trainingを
実行してimmutable outcome-Q artifactを書き出す。

```text
FocalOutcomeSource（read_outcome_source、#193 / #195）
    -> expected source identityとの一致（operatorが明示。下記）
    -> training preflight（hard stopはfail closed、#79 §8）
    -> eligible row（DISCARD かつ #191 survivor >= 2）の in-memory training set
         #184 shared context + #189 full candidate tuple encoding
         behavior-selected candidate index + target_q（#193 build_outcome_targets）
    -> #189と同じcandidate-scorer MLP（hidden width 64、scalar Q / candidate）
    -> MSE( Q(context, selected candidate), target_q )  TRAINでoptimize
    -> SELECT MSE最小のepoch（同値は早いepoch）
    -> immutable outcome-Q artifact（outcome_q_artifact.py）
```

固定する原則。

- lossはbehavior-selected candidateのQだけに掛かる。選ばれなかったsurvivorを
  observed counterfactual labelとして扱わない。full candidate tupleは#189
  encodingのrelative gapを保つためにencodeするだけである
- objectiveはMSEだけ（`OUTCOME_OBJECTIVE_IDENTITY`）。Huber / SmoothL1や
  delta tuningは持たない
- TRAIN / SELECTはsourceのsplitをそのまま使う（protocol invariantであり
  callerが変更できない）。decision rowをsplit間で再配分しない
- architecture familyは#189と同一で、hidden widthは64に固定する
- optimizer / learning rate / weight decay / batch size / epoch数 / seed /
  checkpoint ruleはconfig valueとしてartifactへbindする。executable entry point
  はfrozen default（`FROZEN_OUTCOME_Q_TRAINING_CONFIG`）だけを使い、HPOや
  scheduler searchのflagを持たない

## Training boundary

実際のscientific trainingは、lisbun/lisjong-arena#374が
`ENGINE SCIENTIFIC SOURCE READBACK PASS`を記録するまで行わない。trainerは
`expected_source_identity`（#374 readbackが記録したsource identity）を必須とし、
読んだsourceのidentityと一致しなければ何も書かずにfail closedする。これは
「どのsourceで1回だけtrainingするか」をoperatorが事前に明示する境界であり、
CALIBRATION / DIAGNOSTIC sourceはidentityが一致してもpreflightで拒否する。

torchはoptional extra（`lisjong[ml]`）でありlazy importする。preflightと
training setの構築（`outcome_q_dataset.py`）はML runtimeなしで動く。
"""

import statistics
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

from lisjong.learning._canonical import expect_digest
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
from lisjong.learning.outcome_q_artifact import (
    LoadedOutcomeQArtifact,
    write_outcome_q_artifact,
)
from lisjong.learning.outcome_q_dataset import (
    OUTCOME_Q_HIDDEN_WIDTH,
    SELECT_SPLIT,
    TRAIN_SPLIT,
    OutcomeQTrainingSet,
    build_outcome_q_training_set,
    require_outcome_q_preflight,
)
from lisjong.learning.outcome_source import (
    OUTCOME_OBJECTIVE_IDENTITY,
    FocalOutcomeSource,
)
from lisjong.learning.training import (
    MAX_EPOCHS,
    MAX_SEED,
    OPTIMIZER,
    TRAINING_FRAMEWORK,
    _configure_runtime,
)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutcomeQTrainingConfig:
    """1回のQ trainingを完全に決める不変の実行条件。

    `batch_size`はeligible row数である。TRAIN / SELECT split、optimizer（Adam）、
    objective（MSE）、hidden width（64）はprotocol invariantで変更できない。
    """

    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    seed: int = 0
    model: CandidateScorerConfig = field(default_factory=CandidateScorerConfig)

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
        if not isinstance(self.model, CandidateScorerConfig):
            raise TypeError("model must be a CandidateScorerConfig")
        if self.model.hidden_width != OUTCOME_Q_HIDDEN_WIDTH:
            raise ValueError(
                f"outcome-Q hidden width is frozen at {OUTCOME_Q_HIDDEN_WIDTH}"
            )

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


FROZEN_OUTCOME_Q_TRAINING_CONFIG = OutcomeQTrainingConfig()
"""executable entry pointが使う唯一のconfig（#189 candidate scorerと同じ値）。

checkpoint ruleはSELECT MSE最小のepoch（同値は早いepoch）である。
"""


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


class _Tensors:
    def __init__(self, torch, training_set: OutcomeQTrainingSet) -> None:
        rows = training_set.rows
        self.context = torch.frombuffer(
            bytearray(training_set.context.tobytes()), dtype=torch.float32
        ).reshape(-1, FEATURE_DIMENSION)
        self.candidates = torch.frombuffer(
            bytearray(training_set.candidates.tobytes()), dtype=torch.float32
        ).reshape(-1, CANDIDATE_ENCODING_DIMENSION)
        self.selected = self.candidates.index_select(
            0,
            torch.tensor(
                [row.candidate_offset + row.selected_candidate_index for row in rows],
                dtype=torch.long,
            ),
        )
        self.target = torch.tensor([row.target_q for row in rows], dtype=torch.float32)


def _selected_q(torch, module, tensors: _Tensors, batch):
    """batch rowのbehavior-selected candidateのQを返す（長さ = batch）。"""
    return score_candidates(
        module,
        tensors.context.index_select(0, batch),
        tensors.selected.index_select(0, batch),
        torch.arange(len(batch)),
    )


def _mse(torch, module, tensors: _Tensors, indices, batch_size: int) -> float:
    selector = torch.tensor(list(indices), dtype=torch.long)
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(selector), batch_size):
            batch = selector[start : start + batch_size]
            prediction = _selected_q(torch, module, tensors, batch)
            if not bool(torch.isfinite(prediction).all()):
                raise TrainingError("Q model produced a non-finite prediction")
            error = prediction - tensors.target.index_select(0, batch)
            total += float((error * error).sum())
    return total / len(selector)


def _select_diagnostics(torch, module, tensors, training_set, indices, train_mean):
    """SELECT rowのcalibration / residual-choice診断値（gateではない）。"""
    predictions: list[float] = []
    targets: list[float] = []
    spreads: list[float] = []
    changed = 0
    with torch.inference_mode():
        for index in indices:
            row = training_set.rows[index]
            start = row.candidate_offset
            scores = score_candidates(
                module,
                tensors.context[index : index + 1],
                tensors.candidates[start : start + row.candidate_count],
                torch.zeros(row.candidate_count, dtype=torch.long),
            ).tolist()
            if any(not isfinite(value) for value in scores):
                raise TrainingError("Q model produced a non-finite score")
            survivor_scores = [scores[position] for position in row.survivors]
            best = row.survivors[0]
            for position in row.survivors[1:]:
                if scores[position] > scores[best]:
                    best = position
            changed += int(best != row.survivors[0])
            spreads.append(max(survivor_scores) - min(survivor_scores))
            predictions.append(scores[row.selected_candidate_index])
            targets.append(row.target_q)
    count = len(indices)
    return {
        "select_fraction_q_argmax_not_canonical_first": changed / count,
        "select_mean_survivor_q_spread": statistics.fmean(spreads),
        "select_prediction_mean": statistics.fmean(predictions),
        "select_target_mean": statistics.fmean(targets),
        "select_train_mean_baseline_mse": statistics.fmean(
            (target - train_mean) ** 2 for target in targets
        ),
    }


def train_outcome_q(
    source: FocalOutcomeSource,
    config: OutcomeQTrainingConfig,
    destination: str | Path,
    *,
    expected_source_identity: str,
) -> LoadedOutcomeQArtifact:
    """1回のfrozen Q trainingを実行し、immutable outcome-Q artifactを書き出す。"""
    if not isinstance(source, FocalOutcomeSource):
        raise TrainingError("source must be a strict-read FocalOutcomeSource")
    if not isinstance(config, OutcomeQTrainingConfig):
        raise TrainingError("config must be an OutcomeQTrainingConfig")
    expected = expect_digest(
        expected_source_identity, TrainingError, "expected_source_identity"
    )
    if source.identity != expected:
        raise TrainingError(
            "outcome source identity does not match the expected source identity: "
            f"{source.identity} != {expected}"
        )
    require_outcome_q_preflight(source)
    training_set = build_outcome_q_training_set(source)
    train_indices = training_set.row_indices(TRAIN_SPLIT)
    select_indices = training_set.row_indices(SELECT_SPLIT)

    torch = require_torch()
    _configure_runtime(torch, config.seed)
    tensors = _Tensors(torch, training_set)
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
            error = _selected_q(torch, module, tensors, batch) - (
                tensors.target.index_select(0, batch)
            )
            loss = (error * error).mean()
            if not torch.isfinite(loss):
                raise TrainingError(f"training loss became non-finite at epoch {epoch}")
            loss.backward()
            optimizer.step()

        module.eval()
        objective = _mse(torch, module, tensors, select_indices, config.batch_size)
        if not isfinite(objective):
            raise TrainingError(f"SELECT objective became non-finite at epoch {epoch}")
        if selected_objective is None or objective < selected_objective:
            selected_objective = objective
            selected_epoch = epoch
            selected_weights = scorer_module_weights(config.model, module)

    if selected_weights is None or selected_epoch == 0:
        raise TrainingError("training produced no selected weights")

    module.eval()
    final_train_mse = _mse(torch, module, tensors, train_indices, config.batch_size)
    selected_module = build_scorer_module(config.model, selected_weights)
    train_mean = statistics.fmean(training_set.rows[i].target_q for i in train_indices)
    diagnostics = {
        "epochs_run": config.epochs,
        "final_epoch_train_mse": final_train_mse,
        "select_mse": _mse(
            torch, selected_module, tensors, select_indices, config.batch_size
        ),
        "select_rows": len(select_indices),
        "train_rows": len(train_indices),
        "train_target_mean": train_mean,
        **_select_diagnostics(
            torch, selected_module, tensors, training_set, select_indices, train_mean
        ),
    }

    return write_outcome_q_artifact(
        destination,
        training_set=training_set,
        model_config=config.model,
        training=config.to_value(
            selected_epoch=selected_epoch,
            framework_version=str(torch.__version__),
            diagnostics=diagnostics,
        ),
        weights=selected_weights,
    )


__all__ = [
    "FROZEN_OUTCOME_Q_TRAINING_CONFIG",
    "OutcomeQTrainingConfig",
    "train_outcome_q",
]
