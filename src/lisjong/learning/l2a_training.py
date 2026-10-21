"""L2a候補scorerの学習（Issue #316）。

設計`docs/learning-l2a.md` 4.3節。判断ごとに合法候補（scorer候補）だけで正規化した
softmax cross entropyで、教師が選んだ候補を学習する。打牌と反応は別々に学習する。

```text
strict-read L2a dataset（M0 / M1 / R1）
    -> TRAIN split（最適化）/ SELECT split（epoch選択）
    -> L2a scorer + 判断ごとのsoftmax cross entropy
    -> deterministic Adam + 固定epoch数
    -> SELECT cross entropy最小のepochのweights
    -> immutable L2a scorer artifact（epoch別のTRAIN / SELECT CEをdiagnosticsへ）
```

1回の呼び出しは1つの幅だけを学習する。幅{64, 256}の比較は、各artifactの
`select_cross_entropy`で行う（設計4.3節の選択規則）。OFFLINE-EVAL splitは学習にも
選択にも使わない。損失・argmaxの計算はL0.2の候補学習と同じ関数を共有する。
torchはoptional extraでありlazy importする。
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from lisjong.learning.candidate_model import scorer_module_weights
from lisjong.learning.candidate_training import (
    _argmax_positions,
    _batch_scores,
    _per_decision_loss,
)
from lisjong.learning.errors import TrainingError
from lisjong.learning.features import FEATURE_DIMENSION
from lisjong.learning.l2a_artifact import LoadedL2aScorerArtifact, write_l2a_artifact
from lisjong.learning.l2a_dataset import (
    L2A_HELD_OUT_SPLIT,
    L2A_TRAINING_OBJECTIVE,
    L2aDataset,
)
from lisjong.learning.l2a_model import L2aScorerConfig, build_l2a_scorer_module
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
class L2aTrainingConfig:
    """1回のL2a scorer学習を完全に決める不変の実行条件。

    `batch_size`は判断数である。encodingは学習するdatasetから決まる。
    """

    train_splits: tuple[str, ...]
    select_splits: tuple[str, ...]
    hidden_width: int = 64
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        train_splits = _normalize_splits(self.train_splits, "train_splits")
        select_splits = _normalize_splits(self.select_splits, "select_splits")
        if not train_splits or not select_splits:
            raise ValueError("train_splits and select_splits must not be empty")
        if set(train_splits) & set(select_splits):
            raise ValueError("train_splits and select_splits must be disjoint")
        if L2A_HELD_OUT_SPLIT in (*train_splits, *select_splits):
            raise ValueError(
                f"{L2A_HELD_OUT_SPLIT} must not be used for training or selection"
            )
        if type(self.epochs) is not int or not 1 <= self.epochs <= MAX_EPOCHS:
            raise ValueError(f"epochs must be an int in 1..{MAX_EPOCHS}")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive int")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if type(value) is not float or not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative float")
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        if type(self.seed) is not int or not 0 <= self.seed <= MAX_SEED:
            raise ValueError(f"seed must be an int in 0..{MAX_SEED}")
        object.__setattr__(self, "train_splits", train_splits)
        object.__setattr__(self, "select_splits", select_splits)


class _Tensors:
    """datasetのflat payloadをtorch tensorとして1回だけ保持する。"""

    def __init__(self, torch, dataset: L2aDataset) -> None:
        self.context = torch.frombuffer(
            bytearray(dataset.context.tobytes()), dtype=torch.float32
        ).reshape(-1, FEATURE_DIMENSION)
        self.candidates = torch.frombuffer(
            bytearray(dataset.candidates.tobytes()), dtype=torch.float32
        ).reshape(-1, dataset.encoding.dimension)
        self.offsets = torch.tensor(
            [row.candidate_offset for row in dataset.rows], dtype=torch.long
        )
        self.counts = torch.tensor(
            [row.candidate_count for row in dataset.rows], dtype=torch.long
        )
        self.teacher = torch.tensor(
            [row.teacher_candidate_index for row in dataset.rows], dtype=torch.long
        )


def _split_indices(dataset: L2aDataset, splits, field_name: str) -> tuple[int, ...]:
    missing = [name for name in splits if name not in dataset.split_counts()]
    if missing:
        raise TrainingError(
            f"{field_name} references splits without scorer decisions: {sorted(missing)}"
        )
    return dataset.row_indices_for_splits(splits)


def _evaluate(torch, module, tensors: _Tensors, indices, batch_size: int):
    """平均cross entropyと、判断ごとのlearner選択位置を返す。"""
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
            loss = _per_decision_loss(
                torch,
                scores,
                decision_index,
                local_starts,
                tensors.teacher.index_select(0, batch),
                len(batch),
            )
            total_loss += float(loss.sum())
            chosen.extend(
                _argmax_positions(
                    torch, scores, decision_index, within, len(batch)
                ).tolist()
            )
    return total_loss / len(selector), chosen


def train_l2a_scorer(
    dataset: L2aDataset,
    config: L2aTrainingConfig,
    destination: str | Path,
    *,
    on_epoch: Callable[[int, float, float], None] | None = None,
) -> LoadedL2aScorerArtifact:
    """L2a scorerを学習し、SELECT CE最小のepochのweightsでartifactを書き出す。

    `on_epoch(epoch, train_ce, select_ce)`は各epochの評価後に呼ぶ。例外を送出すると
    学習を中止し、artifactを書き出さない（予算超過の停止に使う）。
    """
    if not isinstance(dataset, L2aDataset):
        raise TrainingError("dataset must be a strict-read L2aDataset")
    if not isinstance(config, L2aTrainingConfig):
        raise TrainingError("config must be an L2aTrainingConfig")
    model_config = L2aScorerConfig(
        encoding=dataset.encoding, hidden_width=config.hidden_width
    )
    train_indices = _split_indices(dataset, config.train_splits, "train_splits")
    select_indices = _split_indices(dataset, config.select_splits, "select_splits")

    torch = require_torch()
    _configure_runtime(torch, config.seed)
    tensors = _Tensors(torch, dataset)
    train_selector = torch.tensor(list(train_indices), dtype=torch.long)
    module = build_l2a_scorer_module(model_config)
    optimizer = torch.optim.Adam(
        module.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    diagnostics: dict[str, float | int] = {}
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
        train_ce, _ = _evaluate(
            torch, module, tensors, train_indices, config.batch_size
        )
        select_ce, _ = _evaluate(
            torch, module, tensors, select_indices, config.batch_size
        )
        if not isfinite(train_ce) or not isfinite(select_ce):
            raise TrainingError(f"cross entropy became non-finite at epoch {epoch}")
        diagnostics[f"epoch_{epoch:03d}_train_cross_entropy"] = train_ce
        diagnostics[f"epoch_{epoch:03d}_select_cross_entropy"] = select_ce
        if selected_objective is None or select_ce < selected_objective:
            selected_objective = select_ce
            selected_epoch = epoch
            selected_weights = scorer_module_weights(model_config, module)
        if on_epoch is not None:
            on_epoch(epoch, train_ce, select_ce)

    if selected_weights is None:
        raise TrainingError("training produced no selected weights")
    selected_module = build_l2a_scorer_module(model_config, selected_weights)
    select_ce, _ = _evaluate(
        torch, selected_module, tensors, select_indices, config.batch_size
    )
    diagnostics.update(
        {
            "select_cross_entropy": select_ce,
            "select_decisions": len(select_indices),
            "train_decisions": len(train_indices),
        }
    )
    training = {
        "batch_size": config.batch_size,
        "diagnostics": dict(sorted(diagnostics.items())),
        "epochs": config.epochs,
        "framework": {"name": TRAINING_FRAMEWORK, "version": str(torch.__version__)},
        "learning_rate": config.learning_rate,
        "objective": L2A_TRAINING_OBJECTIVE,
        "optimizer": OPTIMIZER,
        "seed": config.seed,
        "selected_epoch": selected_epoch,
        "train_splits": list(config.train_splits),
        "validation_splits": list(config.select_splits),
        "weight_decay": config.weight_decay,
    }
    return write_l2a_artifact(
        destination,
        dataset=dataset,
        model_config=model_config,
        training=training,
        weights=selected_weights,
    )


__all__ = ["L2aTrainingConfig", "train_l2a_scorer"]
