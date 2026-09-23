"""Issue #191 L0.2a semantic-envelope policyのbounded engineering replay。

#189 frozen candidate scorer artifactを`SemanticEnvelopeRuntime`で包み、既観測の
TRAIN / SELECT splitへserving pathを適用してconstruction invariantと
residual-choice diagnosticsを出す。referenceとして`TwoStepUkeirePolicy`を
注入し、constant-scorer oracleとのaction-object一致を記録する。

    python tools/replay_semantic_envelope.py \\
        --artifact <candidate-scorer-artifact> --source-record <source-record> \\
        --split SELECT
    python tools/replay_semantic_envelope.py ... --split TRAIN --sample-every 5

#189 OFFLINE-EVALを新しいholdoutとして再利用しないため、artifactの
training / selectionに使ったsplit以外は拒否する。optional ML runtimeが必要。
CI wall-clock thresholdはここから作らない。
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lisjong.learning import read_source_record  # noqa: E402
from lisjong.learning._canonical import canonical_json_text  # noqa: E402
from lisjong.learning.envelope_diagnostics import (  # noqa: E402
    classify_semantic_envelope_result,
    evaluate_semantic_envelope_policy,
)
from lisjong.learning.envelope_policy import (  # noqa: E402
    load_semantic_envelope_policy_factory,
)
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--source-record", required=True)
    parser.add_argument("--split", required=True, action="append")
    parser.add_argument("--sample-every", type=int, default=1)
    arguments = parser.parse_args(argv)

    runtime = load_semantic_envelope_policy_factory(arguments.artifact)
    training = runtime.scorer.artifact.manifest["training"]
    observed = set(training["train_splits"]) | set(training["validation_splits"])
    unobserved = sorted(set(arguments.split) - observed)
    if unobserved:
        parser.error(
            f"splits {unobserved} were not observed by the #189 artifact; "
            "the #191 replay uses TRAIN / SELECT only"
        )
    source = read_source_record(arguments.source_record)
    if source.identity != runtime.scorer.artifact.source_identity:
        parser.error("source record identity does not match the artifact source")

    started = time.perf_counter()
    evaluation = evaluate_semantic_envelope_policy(
        runtime(),
        source,
        arguments.split,
        sample_every=arguments.sample_every,
        reference=TwoStepUkeirePolicy(),
    )
    summary = {
        "artifact_identity": runtime.artifact_identity,
        "classification": classify_semantic_envelope_result(evaluation),
        "evaluation": evaluation,
        "python": sys.version.split()[0],
        "runtime_identity": runtime.identity,
        "source_identity": source.identity,
        "wall_seconds": time.perf_counter() - started,
    }
    sys.stdout.flush()
    sys.stdout.buffer.write(canonical_json_text(summary).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
