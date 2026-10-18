"""Learning L0のexecutable entry point。

training semanticsとその実行entry pointは`lisjong`が所有する。CLIは
`lisjong.learning`のcontractを呼ぶ薄いwrapperであり、experiment framework、
job scheduler、config file loader、artifact registryを持たない。

```text
python -m lisjong.learning materialize-dataset \
    --source-record <source-record> --output <dataset>

python -m lisjong.learning train \
    --dataset <dataset> --output <artifact> \
    --train-split TRAIN --validation-split SELECT

python -m lisjong.learning verify-artifact --artifact <artifact>
```

Issue #189のL0.2 candidate scorer pathは次のsubcommandで扱う。

```text
python -m lisjong.learning materialize-candidate-dataset     --source-record <source-record> --output <dataset>

python -m lisjong.learning train-candidate-scorer     --dataset <dataset> --output <artifact>     --train-split TRAIN --select-split SELECT

python -m lisjong.learning verify-candidate-artifact --artifact <artifact>

python -m lisjong.learning evaluate-candidate-scorer     --artifact <artifact> --source-record <source-record> --split OFFLINE-EVAL
```

Issue #315のL2a support計数（設計`docs/learning-l2a.md` 2.3・3.3節）。

```text
python -m lisjong.learning l2a-source-support --source-record <source-record>
```

Issue #316のL2a候補scorer（設計4節。打牌M0 / M1、反応R1）。

```text
python -m lisjong.learning l2a-dataset     --source-record <source-record> --output <dataset> --encoding m1

python -m lisjong.learning l2a-train     --dataset <dataset> --output <artifact>     --train-split TRAIN --select-split SELECT --hidden-width 64

python -m lisjong.learning verify-l2a-artifact --artifact <artifact>
```

`train` / `train-candidate-scorer` / `evaluate-candidate-scorer` / `l2a-train`だけがoptional
ML runtimeを必要とする。dataset materializationとartifact検証はML runtimeなしで
実行できる。
"""

import argparse
import sys

from lisjong.learning._canonical import canonical_json_text
from lisjong.learning.artifact import load_model_artifact
from lisjong.learning.candidate_artifact import load_candidate_artifact
from lisjong.learning.candidate_dataset import (
    publish_candidate_dataset,
    read_candidate_dataset,
)
from lisjong.learning.candidate_diagnostics import (
    classify_offline_result,
    evaluate_candidate_policy,
)
from lisjong.learning.candidate_model import CandidateScorerConfig
from lisjong.learning.candidate_policy import load_candidate_scorer_policy_factory
from lisjong.learning.candidate_training import (
    CandidateScorerTrainingConfig,
    train_candidate_scorer,
)
from lisjong.learning.dataset import publish_dataset, read_dataset
from lisjong.learning.errors import LearningError
from lisjong.learning.l2a_artifact import load_l2a_artifact
from lisjong.learning.l2a_dataset import publish_l2a_dataset, read_l2a_dataset
from lisjong.learning.l2a_encoding import L2A_ENCODINGS, l2a_encoding
from lisjong.learning.l2a_model import L2A_HIDDEN_WIDTHS
from lisjong.learning.l2a_support import count_source_support
from lisjong.learning.l2a_training import L2aTrainingConfig, train_l2a_scorer
from lisjong.learning.model import ModelConfig
from lisjong.learning.source_record import open_source_record, read_source_record
from lisjong.learning.training import BehaviorCloningConfig, train_behavior_cloning


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lisjong.learning",
        description="lisjong-owned Learning L0 entry point",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    materialize = commands.add_parser(
        "materialize-dataset",
        help="strict-read a player-safe source record into a versioned dataset",
    )
    materialize.add_argument("--source-record", required=True)
    materialize.add_argument("--output", required=True)

    train = commands.add_parser(
        "train", help="run bounded behavior cloning and write a model artifact"
    )
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--train-split", required=True, action="append", default=None)
    train.add_argument("--validation-split", action="append", default=None)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1.0e-3)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--hidden-width", type=int, default=ModelConfig().hidden_width)

    verify = commands.add_parser(
        "verify-artifact", help="strict-load a model artifact without an ML runtime"
    )
    verify.add_argument("--artifact", required=True)

    candidate_materialize = commands.add_parser(
        "materialize-candidate-dataset",
        help="strict-read a player-safe source record into an L0.2 candidate dataset",
    )
    candidate_materialize.add_argument("--source-record", required=True)
    candidate_materialize.add_argument("--output", required=True)
    candidate_materialize.add_argument(
        "--workers",
        type=int,
        default=1,
        help="convert games as shards in this many processes (output is identical)",
    )
    candidate_materialize.add_argument(
        "--max-pending-shards",
        type=int,
        default=None,
        help="bound on submitted, not yet concatenated shards (default 2 x workers)",
    )

    candidate_train = commands.add_parser(
        "train-candidate-scorer",
        help="train the bounded L0.2 candidate scorer and write an artifact",
    )
    candidate_train.add_argument("--dataset", required=True)
    candidate_train.add_argument("--output", required=True)
    candidate_train.add_argument("--train-split", required=True, action="append")
    candidate_train.add_argument("--select-split", required=True, action="append")
    candidate_train.add_argument("--epochs", type=int, default=20)
    candidate_train.add_argument("--batch-size", type=int, default=256)
    candidate_train.add_argument("--learning-rate", type=float, default=1.0e-3)
    candidate_train.add_argument("--weight-decay", type=float, default=0.0)
    candidate_train.add_argument("--seed", type=int, default=0)
    candidate_train.add_argument(
        "--hidden-width", type=int, default=CandidateScorerConfig().hidden_width
    )

    candidate_verify = commands.add_parser(
        "verify-candidate-artifact",
        help="strict-load an L0.2 candidate scorer artifact without an ML runtime",
    )
    candidate_verify.add_argument("--artifact", required=True)

    candidate_evaluate = commands.add_parser(
        "evaluate-candidate-scorer",
        help="run the frozen L0.2 offline gate on one held-out split",
    )
    candidate_evaluate.add_argument("--artifact", required=True)
    candidate_evaluate.add_argument("--source-record", required=True)
    candidate_evaluate.add_argument("--split", required=True, action="append")

    l2a_support = commands.add_parser(
        "l2a-source-support",
        help="count L2a decision kinds, action support and teacher-rule exceptions",
    )
    l2a_support.add_argument("--source-record", required=True)

    l2a_dataset = commands.add_parser(
        "l2a-dataset",
        help="materialize an L2a candidate dataset (discard M0 / M1, reaction R1)",
    )
    l2a_dataset.add_argument("--source-record", required=True)
    l2a_dataset.add_argument("--output", required=True)
    l2a_dataset.add_argument("--encoding", required=True, choices=sorted(L2A_ENCODINGS))
    l2a_dataset.add_argument("--workers", type=int, default=1)
    l2a_dataset.add_argument("--max-pending-shards", type=int, default=None)

    l2a_train = commands.add_parser(
        "l2a-train", help="train one L2a candidate scorer and write an artifact"
    )
    l2a_train.add_argument("--dataset", required=True)
    l2a_train.add_argument("--output", required=True)
    l2a_train.add_argument("--train-split", required=True, action="append")
    l2a_train.add_argument("--select-split", required=True, action="append")
    l2a_train.add_argument(
        "--hidden-width", type=int, required=True, choices=L2A_HIDDEN_WIDTHS
    )
    l2a_train.add_argument("--epochs", type=int, default=20)
    l2a_train.add_argument("--batch-size", type=int, default=256)
    l2a_train.add_argument("--learning-rate", type=float, default=1.0e-3)
    l2a_train.add_argument("--weight-decay", type=float, default=0.0)
    l2a_train.add_argument("--seed", type=int, default=0)

    l2a_verify = commands.add_parser(
        "verify-l2a-artifact",
        help="strict-load an L2a candidate scorer artifact without an ML runtime",
    )
    l2a_verify.add_argument("--artifact", required=True)

    return parser


def _materialize(arguments) -> dict[str, object]:
    # source / datasetを全件保持せずに変換する（Issue #247）。
    source = open_source_record(arguments.source_record)
    manifest = publish_dataset(source, arguments.output)
    return {
        "dataset_identity": manifest["identity"],
        "output": str(arguments.output),
        "row_count": manifest["rows"]["count"],
        "source_identity": source.identity,
        "splits": dict(sorted(manifest["rows"]["splits"].items())),
    }


def _train(arguments) -> dict[str, object]:
    dataset = read_dataset(arguments.dataset)
    config = BehaviorCloningConfig(
        train_splits=tuple(arguments.train_split),
        validation_splits=tuple(arguments.validation_split or ()),
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        seed=arguments.seed,
        model=ModelConfig(hidden_width=arguments.hidden_width),
    )
    artifact = train_behavior_cloning(dataset, config, arguments.output)
    return {
        "artifact_identity": artifact.identity,
        "dataset_identity": artifact.dataset_identity,
        "output": str(arguments.output),
        "selected_epoch": artifact.manifest["training"]["selected_epoch"],
        "source_identity": artifact.source_identity,
        "training_diagnostics": artifact.manifest["training"]["diagnostics"],
    }


def _verify(arguments) -> dict[str, object]:
    artifact = load_model_artifact(arguments.artifact)
    return {
        "artifact_identity": artifact.identity,
        "dataset_identity": artifact.dataset_identity,
        "feature": artifact.manifest["feature"],
        "model": artifact.manifest["model"],
        "source_identity": artifact.source_identity,
        "vocabulary": artifact.manifest["vocabulary"],
    }


def _materialize_candidates(arguments) -> dict[str, object]:
    source = open_source_record(arguments.source_record)
    manifest = publish_candidate_dataset(
        source,
        arguments.output,
        workers=arguments.workers,
        max_pending_shards=arguments.max_pending_shards,
    )
    return {
        "dataset_identity": manifest["identity"],
        "output": str(arguments.output),
        "rows": manifest["rows"],
        "source_identity": source.identity,
    }


def _train_candidates(arguments) -> dict[str, object]:
    dataset = read_candidate_dataset(arguments.dataset)
    config = CandidateScorerTrainingConfig(
        train_splits=tuple(arguments.train_split),
        select_splits=tuple(arguments.select_split),
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        seed=arguments.seed,
        model=CandidateScorerConfig(hidden_width=arguments.hidden_width),
    )
    artifact = train_candidate_scorer(dataset, config, arguments.output)
    return {
        "artifact_identity": artifact.identity,
        "dataset_identity": artifact.dataset_identity,
        "output": str(arguments.output),
        "selected_epoch": artifact.manifest["training"]["selected_epoch"],
        "source_identity": artifact.source_identity,
        "training_diagnostics": artifact.manifest["training"]["diagnostics"],
    }


def _verify_candidates(arguments) -> dict[str, object]:
    artifact = load_candidate_artifact(arguments.artifact)
    return {
        "artifact_identity": artifact.identity,
        "dataset_identity": artifact.dataset_identity,
        "encoding": artifact.manifest["encoding"],
        "feature": artifact.manifest["feature"],
        "model": artifact.manifest["model"],
        "source_identity": artifact.source_identity,
    }


def _evaluate_candidates(arguments) -> dict[str, object]:
    runtime = load_candidate_scorer_policy_factory(arguments.artifact)
    training = runtime.artifact.manifest["training"]
    used = set(training["train_splits"]) | set(training["validation_splits"])
    if used & set(arguments.split):
        raise ValueError(
            "evaluation split was used for training or epoch selection: "
            f"{sorted(used & set(arguments.split))}"
        )
    source = read_source_record(arguments.source_record)
    if source.identity != runtime.artifact.source_identity:
        raise ValueError("source record identity does not match the artifact source")
    evaluation = evaluate_candidate_policy(runtime(), source, arguments.split)
    return {
        "artifact_identity": runtime.identity,
        "classification": classify_offline_result(evaluation),
        "evaluation": evaluation,
        "source_identity": source.identity,
    }


def _l2a_source_support(arguments) -> dict[str, object]:
    # gameを1つずつ読み、保持する判断を半荘数から切り離す。
    return count_source_support(open_source_record(arguments.source_record))


def _l2a_dataset(arguments) -> dict[str, object]:
    source = open_source_record(arguments.source_record)
    manifest = publish_l2a_dataset(
        source,
        arguments.output,
        encoding=l2a_encoding(arguments.encoding),
        workers=arguments.workers,
        max_pending_shards=arguments.max_pending_shards,
    )
    return {
        "dataset_identity": manifest["identity"],
        "encoding": manifest["encoding"],
        "output": str(arguments.output),
        "rows": manifest["rows"],
        "source_identity": source.identity,
    }


def _l2a_train(arguments) -> dict[str, object]:
    dataset = read_l2a_dataset(arguments.dataset)
    config = L2aTrainingConfig(
        train_splits=tuple(arguments.train_split),
        select_splits=tuple(arguments.select_split),
        hidden_width=arguments.hidden_width,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        seed=arguments.seed,
    )
    artifact = train_l2a_scorer(dataset, config, arguments.output)
    return {
        "artifact_identity": artifact.identity,
        "dataset_identity": artifact.dataset_identity,
        "output": str(arguments.output),
        "selected_epoch": artifact.manifest["training"]["selected_epoch"],
        "source_identity": artifact.source_identity,
        "training_diagnostics": artifact.manifest["training"]["diagnostics"],
    }


def _verify_l2a(arguments) -> dict[str, object]:
    artifact = load_l2a_artifact(arguments.artifact)
    return {
        "artifact_identity": artifact.identity,
        "dataset_identity": artifact.dataset_identity,
        "encoding": artifact.manifest["encoding"],
        "model": artifact.manifest["model"],
        "selected_epoch": artifact.manifest["training"]["selected_epoch"],
        "source_identity": artifact.source_identity,
    }


def _emit(summary: dict[str, object]) -> None:
    """summaryをUTF-8のcanonical JSONとしてstdoutへ書く。

    terminal classification等は非ASCII（例: em dash）を含むため、console
    encoding（Windowsのcp932等）に依存せずUTF-8で出力する。
    """
    text = canonical_json_text(summary)
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        sys.stdout.write(text)
        return
    sys.stdout.flush()
    buffer.write(text.encode("utf-8"))
    buffer.flush()


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    handlers = {
        "materialize-dataset": _materialize,
        "train": _train,
        "verify-artifact": _verify,
        "materialize-candidate-dataset": _materialize_candidates,
        "train-candidate-scorer": _train_candidates,
        "verify-candidate-artifact": _verify_candidates,
        "evaluate-candidate-scorer": _evaluate_candidates,
        "l2a-source-support": _l2a_source_support,
        "l2a-dataset": _l2a_dataset,
        "l2a-train": _l2a_train,
        "verify-l2a-artifact": _verify_l2a,
    }
    try:
        summary = handlers[arguments.command](arguments)
    except (LearningError, TypeError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    _emit(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
