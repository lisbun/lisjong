"""lisjong-project#79 step E — outcome-Q runtimeのcheap serving qualification。

frozen outcome-Q artifactを`OutcomeQRuntime`で包み、artifactが学習した
focal outcome sourceの指定splitへserving pathを適用してsafety / integration
gateとQ診断値を出す。referenceとして`TwoStepUkeirePolicy`を注入し、
constant-scorer oracleとのaction-object一致を記録する。strengthの証明ではない。

    python tools/qualify_outcome_q.py \\
        --artifact <outcome-q-artifact> --source <outcome-source> --split SELECT

sourceのidentityがartifactのsource identityと一致しなければ拒否する。
optional ML runtimeが必要。CI wall-clock thresholdはここから作らない。
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lisjong.learning._canonical import canonical_json_text  # noqa: E402
from lisjong.learning.outcome_q_diagnostics import (  # noqa: E402
    classify_outcome_q_serving_result,
    evaluate_outcome_q_policy,
)
from lisjong.learning.outcome_q_policy import (  # noqa: E402
    load_outcome_q_policy_factory,
)
from lisjong.learning.outcome_source import read_outcome_source  # noqa: E402
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--split", required=True, action="append")
    arguments = parser.parse_args(argv)

    runtime = load_outcome_q_policy_factory(arguments.artifact)
    source = read_outcome_source(arguments.source)
    if source.identity != runtime.artifact.source_identity:
        parser.error("outcome source identity does not match the artifact source")

    started = time.perf_counter()
    evaluation = evaluate_outcome_q_policy(
        runtime, source, arguments.split, reference=TwoStepUkeirePolicy()
    )
    summary = {
        "classification": classify_outcome_q_serving_result(evaluation),
        "evaluation": evaluation,
        "python": sys.version.split()[0],
        "wall_seconds": time.perf_counter() - started,
    }
    sys.stdout.flush()
    sys.stdout.buffer.write(canonical_json_text(summary).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
