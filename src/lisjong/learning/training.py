"""offense L0のbounded Behavior Cloning trainer。

Issue #184のL0cに対応する。teacherが実際に選んだcanonical actionを、同じ
decisionのlegal action上のclassification targetとして学習する最小のpathだけを
提供する。

```text
strict-read dataset
    -> train / validation split membership
    -> bounded MLP + masked cross entropy
    -> deterministic Adam + fixed epoch budget
    -> selected epoch weights
    -> immutable model artifact
```

意図的に持たないもの。

- generic experiment framework / trainer abstraction / plugin registry
- hyperparameter search / HPO / scheduler / auto-tuning
- self-play / RL / DAgger / online aggregation
- teacher semanticsの変更、soft target、label smoothing
- distributed / GPU実行、mixed precision

固定する原則。

- train splitとvalidation splitは呼び出し側が明示する。protocol identityを
  構成するsplit名をtrainerのdefaultへ埋め込まない
- split membershipはdataset側のsource population（hanchan単位）に従う。
  trainerはrowをsplit間で再配分しない
- 同じdataset、同じconfig、同じseedでは同じweightsになる（同一platform /
  同一framework version上でのdeterministic実行）
- illegal actionへ確率質量を与えない。maskはlegal indexだけを残す
- torchはoptional extra（`lisjong[ml]`）であり、lazy importで必要とする

本moduleはCPU実行とsingle threadを前提にprocess全体のtorch設定を固定する。
training entry pointとしての意図的なglobal設定であり、library内の暗黙の
副作用として隠さない。
"""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

from lisjong.learning.artifact import LoadedModelArtifact, write_model_artifact
from lisjong.learning.dataset import TRAINING_OBJECTIVE, LearningDataset
from lisjong.learning.errors import TrainingError
from lisjong.learning.model import (
    ModelConfig,
    build_module,
    module_weights,
    require_torch,
)

OPTIMIZER = "adam"
TRAINING_FRAMEWORK = "torch"
TORCH_THREADS = 1
TRAINING_DEVICE = "cpu"
MAX_EPOCHS = 200
MAX_SEED = 2**32 - 1

_MASKED_LOGIT = -1.0e9
"""illegal action indexへ与えるlogit。

float32のsoftmaxでは`exp(-1e9 - max)`が0へunderflowするため、illegal index
へ確率質量が残らない。`-inf`を使うと一部のautograd kernelでNaNが伝播し得る
ため、有限な下限値を使う。
"""


def _normalize_splits(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a tuple of split names")
    try:
        names = tuple(values)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable of split names") from None
    if any(type(name) is not str or not name for name in names):
        raise ValueError(f"{field_name} must contain only non-empty split names")
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must not contain duplicates")
    return names


@dataclass(frozen=True, slots=True)
class BehaviorCloningConfig:
    """1回のbounded BC trainingを完全に決める不変の実行条件。"""

    train_splits: tuple[str, ...]
    validation_splits: tuple[str, ...] = ()
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        train_splits = _normalize_splits(self.train_splits, "train_splits")
        validation_splits = _normalize_splits(
            self.validation_splits, "validation_splits"
        )
        if not train_splits:
            raise ValueError("train_splits must not be empty")
        if set(train_splits) & set(validation_splits):
            raise ValueError("train_splits and validation_splits must be disjoint")
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
        if not isinstance(self.model, ModelConfig):
            raise TypeError("model must be a ModelConfig")
        object.__setattr__(self, "train_splits", train_splits)
        object.__setattr__(self, "validation_splits", validation_splits)

    def to_value(self, *, selected_epoch: int, framework_version: str, diagnostics):
        """artifactへbindするtraining blockを返す。"""
        return {
            "batch_size": self.batch_size,
            "diagnostics": dict(sorted(diagnostics.items())),
            "epochs": self.epochs,
            "framework": {"name": TRAINING_FRAMEWORK, "version": framework_version},
            "learning_rate": self.learning_rate,
            "objective": TRAINING_OBJECTIVE,
            "optimizer": OPTIMIZER,
            "seed": self.seed,
            "selected_epoch": selected_epoch,
            "train_splits": list(self.train_splits),
            "validation_splits": list(self.validation_splits),
            "weight_decay": self.weight_decay,
        }


def _configure_runtime(torch, seed: int) -> None:
    """deterministicなCPU実行条件を固定する。"""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(TORCH_THREADS)
    torch.manual_seed(seed)


def _split_rows(
    dataset: LearningDataset, splits: tuple[str, ...], field_name: str
) -> tuple[int, ...]:
    available = dataset.split_counts()
    missing = [name for name in splits if name not in available]
    if missing:
        raise TrainingError(
            f"{field_name} references splits that are absent from the dataset: "
            f"{sorted(missing)}"
        )
    return dataset.row_indices_for_splits(splits)


def _tensors(torch, dataset: LearningDataset, indices: tuple[int, ...]):
    dimension = dataset.feature_dimension
    features = torch.frombuffer(
        bytearray(dataset.features.tobytes()), dtype=torch.float32
    ).reshape(-1, dimension)
    mask = (
        torch.frombuffer(bytearray(dataset.legal_mask), dtype=torch.uint8)
        .reshape(len(dataset.rows), -1)
        .bool()
    )
    labels = torch.tensor(
        [row.teacher_action_index for row in dataset.rows], dtype=torch.long
    )
    selector = torch.tensor(list(indices), dtype=torch.long)
    return (
        features.index_select(0, selector),
        mask.index_select(0, selector),
        labels.index_select(0, selector),
    )


def _masked_logits(torch, module, features, mask):
    return module(features).masked_fill(~mask, _MASKED_LOGIT)


def _evaluate(torch, module, features, mask, labels, batch_size: int):
    """masked cross entropyとteacher top-1 agreementを返す。"""
    total_loss = 0.0
    agreed = 0
    rows = features.shape[0]
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            logits = _masked_logits(
                torch, module, features[start:stop], mask[start:stop]
            )
            target = labels[start:stop]
            total_loss += float(
                torch.nn.functional.cross_entropy(logits, target, reduction="sum")
            )
            agreed += int((logits.argmax(dim=1) == target).sum())
    return total_loss / rows, agreed / rows


def train_behavior_cloning(
    dataset: LearningDataset,
    config: BehaviorCloningConfig,
    destination: str | Path,
) -> LoadedModelArtifact:
    """bounded BC trainingを実行し、immutable model artifactを書き出す。

    validation splitを指定した場合はvalidation masked cross entropyが最小の
    epochのweightsを採用する。指定しない場合は最終epochのweightsを採用する。
    採用epochはartifactへ記録する。
    """
    if not isinstance(dataset, LearningDataset):
        raise TrainingError("dataset must be a strict-read LearningDataset")
    if not isinstance(config, BehaviorCloningConfig):
        raise TrainingError("config must be a BehaviorCloningConfig")

    # split membershipはpure Pythonで検証できるcontract違反であり、ML runtime
    # の有無に関わらず同じ例外を出す。torchの要求より先に検証することで、
    # ML extra未installでも呼び出し側の設定ミスを正しく報告できる。
    train_indices = _split_rows(dataset, config.train_splits, "train_splits")
    if not train_indices:
        raise TrainingError("train_splits selected no dataset rows")
    validation_indices = (
        _split_rows(dataset, config.validation_splits, "validation_splits")
        if config.validation_splits
        else ()
    )
    if config.validation_splits and not validation_indices:
        raise TrainingError("validation_splits selected no dataset rows")

    torch = require_torch()
    _configure_runtime(torch, config.seed)

    train_features, train_mask, train_labels = _tensors(torch, dataset, train_indices)
    validation = (
        _tensors(torch, dataset, validation_indices) if validation_indices else None
    )

    module = build_module(config.model)
    module.train()
    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    rows = len(train_indices)
    selected_weights = None
    selected_epoch = 0
    selected_objective = None
    for epoch in range(1, config.epochs + 1):
        permutation = torch.randperm(rows, generator=generator)
        module.train()
        for start in range(0, rows, config.batch_size):
            batch = permutation[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = _masked_logits(
                torch,
                module,
                train_features.index_select(0, batch),
                train_mask.index_select(0, batch),
            )
            loss = torch.nn.functional.cross_entropy(
                logits, train_labels.index_select(0, batch)
            )
            if not torch.isfinite(loss):
                raise TrainingError(f"training loss became non-finite at epoch {epoch}")
            loss.backward()
            optimizer.step()

        module.eval()
        if validation is None:
            objective, _agreement = _evaluate(
                torch,
                module,
                train_features,
                train_mask,
                train_labels,
                config.batch_size,
            )
        else:
            objective, _agreement = _evaluate(
                torch, module, *validation, config.batch_size
            )
        if not isfinite(objective):
            raise TrainingError(f"objective became non-finite at epoch {epoch}")
        # validationがある場合はvalidation objective最小のepochを採用し、無い
        # 場合は最終epochを採用する。どちらもresult exposure後の選び直しではなく、
        # 事前に固定したselection ruleである。
        improved = (
            selected_weights is None
            or validation is None
            or objective < selected_objective
        )
        if improved:
            selected_objective = objective
            selected_epoch = epoch
            selected_weights = module_weights(config.model, module)

    if selected_weights is None or selected_epoch == 0:
        raise TrainingError("training produced no selected weights")

    module.eval()
    final_train_ce, final_train_agreement = _evaluate(
        torch, module, train_features, train_mask, train_labels, config.batch_size
    )
    diagnostics = {
        "epochs_run": config.epochs,
        "final_epoch_train_masked_cross_entropy": final_train_ce,
        "final_epoch_train_teacher_agreement": final_train_agreement,
        "selected_objective_masked_cross_entropy": selected_objective,
        "train_rows": rows,
        "validation_rows": len(validation_indices),
    }

    return write_model_artifact(
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
    "MAX_EPOCHS",
    "OPTIMIZER",
    "TORCH_THREADS",
    "TRAINING_DEVICE",
    "TRAINING_FRAMEWORK",
    "BehaviorCloningConfig",
    "train_behavior_cloning",
]
