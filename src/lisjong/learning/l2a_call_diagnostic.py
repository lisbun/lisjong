"""#324: 凍結R1のTRAIN/SELECT限定診断。学習・OFFLINE-EVALは行わない。

python -m lisjong.learning.l2a_call_diagnostic --help
production入口は#318のsource/artifactとseedを固定する。集計helperは合成test用。
"""

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

from lisjong.learning._canonical import canonical_json_text, file_digest, seal, unseal
from lisjong.learning._publication import write_new_text
from lisjong.learning.errors import LearnedPolicyError, LearningError
from lisjong.learning.features import build_player_safe_feature
from lisjong.learning.l2a_decision import (
    L2aDecisionKind,
    classify_l2a_decision,
    teacher_rule_exception,
)
from lisjong.learning.l2a_encoding import ENCODING_R1
from lisjong.learning.l2a_policy import load_l2a_scorer, select_best
from lisjong.learning.source_record import (
    POLICY_SOURCE_RECORD_SCHEMA_V2,
    SCIENTIFIC_PURPOSE,
    open_source_record,
    read_source_game,
)
from lisjong.policy_contract import ChiAction, DecisionContext, PassAction, PonAction
from lisjong.policy_contract.meld import MeldKind

SOURCE_ID = "ba02b59163e17e128b998c80f59dc386c49796fc3dd35f67b88dc2c756421aaa"
R1_ID = "d565713f51a2b9f9cbdd2146bf01cb4bfb13ff859b639de962c74562eaf01efc"
SPLITS = ("TRAIN", "SELECT")
KINDS = ("pass", "chi", "pon")
GROUPS = ("all", "chi_legal", "1", "2", "3+")
CONFIG = {
    "identity": "lisjong-l2a-call-diagnostic-324-v1",
    "source": SOURCE_ID,
    "r1": R1_ID,
    "splits": list(SPLITS),
    "seed_ranges": {"TRAIN": [506000, 506599], "SELECT": [506600, 506699]},
    "preflight_games": 10,
    "max_wall_seconds": 7200,
    "max_rss_mib": 8192,
    "chi_candidate_bins": ["1", "2", "3+"],
    "quantile": "linear interpolation at (n-1)*p",
    "interpretation": "exploratory; no independent evaluation or strength claim",
}


def action_kind(action):
    for cls, name in ((PassAction, "pass"), (ChiAction, "chi"), (PonAction, "pon")):
        if isinstance(action, cls):
            return name
    raise LearnedPolicyError("non-CALL candidate")


def fraction(numerator, denominator):
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "reason": None if denominator else "zero denominator",
    }


def distribution(values):
    ordered = sorted(values)

    def quantile(p):
        if not ordered:
            return None
        position = (len(ordered) - 1) * p
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(ordered),
        "reason": None if ordered else "zero denominator",
        "q25": quantile(0.25),
        "median": quantile(0.5),
        "q75": quantile(0.75),
        "negative": sum(v < 0 for v in ordered),
        "zero": sum(v == 0 for v in ordered),
        "positive": sum(v > 0 for v in ordered),
    }


class CallGroup:
    """分子・分母それぞれの半荘supportを保持。個別入力は保存しない。"""

    def __init__(self):
        self.counts = Counter()
        self.support = defaultdict(set)
        self.margins = defaultdict(list)

    def add(self, seed, key):
        self.counts[key] += 1
        self.support[key].add(seed)

    def observe(self, seed, candidates, scores, teacher):
        if len(candidates) != len(scores) or len(set(candidates)) != len(candidates):
            raise LearnedPolicyError("candidate/score count or uniqueness mismatch")
        if candidates.count(teacher) != 1:
            raise LearnedPolicyError("teacher does not resolve uniquely")
        kinds = tuple(action_kind(a) for a in candidates)
        if "pass" not in kinds or not any(k in kinds for k in ("chi", "pon")):
            raise LearnedPolicyError("CALL requires Pass and Chi/Pon")
        predicted_index = select_best(scores)  # finite validation, canonical tie-break
        predicted = candidates[predicted_index]
        true_kind, predicted_kind = action_kind(teacher), kinds[predicted_index]
        teacher_index = candidates.index(teacher)
        self.add(seed, "rows")
        self.add(seed, f"matrix.{true_kind}.{predicted_kind}")
        if teacher != predicted:
            self.add(seed, "mismatch")
        for name in ("chi", "pon"):
            if name not in kinds:
                continue
            self.add(seed, f"{name}.legal")
            positive, selected = true_kind == name, predicted_kind == name
            cell = (
                ("tp" if selected else "fn")
                if positive
                else ("fp" if selected else "tn")
            )
            self.add(seed, f"{name}.{cell}")
            self.add(seed, f"{name}.teacher_{'positive' if positive else 'negative'}")
            if positive:
                if predicted_kind != name:
                    self.add(seed, f"{name}.non_kind")
                    self.add(seed, f"{name}.non_kind.{predicted_kind}")
                elif predicted != teacher:
                    self.add(seed, f"{name}.wrong_candidate")
                else:
                    self.add(seed, f"{name}.exact")
        if (
            "chi" in kinds
            and (true_kind == "chi" or predicted_kind == "chi")
            and predicted != teacher
        ):
            self.add(seed, "chi.related_mismatch")
        if true_kind != "chi":
            return
        chi = [i for i, k in enumerate(kinds) if k == "chi"]
        non_chi = [i for i, k in enumerate(kinds) if k != "chi"]
        best_chi = chi[select_best(tuple(scores[i] for i in chi))]
        if candidates[best_chi] == teacher:
            self.add(seed, "chi.restricted_exact")
        rank = (
            sorted(range(len(scores)), key=lambda i: (-scores[i], i)).index(
                teacher_index
            )
            + 1
        )
        self.add(seed, f"chi.teacher_rank.{rank}")
        self.margins["chi_vs_non_chi"].append(
            max(scores[i] for i in chi) - max(scores[i] for i in non_chi)
        )
        others = [i for i in chi if i != teacher_index]
        if others:
            self.margins["teacher_vs_other_chi"].append(
                scores[teacher_index] - max(scores[i] for i in others)
            )
        else:
            self.add(seed, "chi.no_other_candidate")

    def report(self):
        c = self.counts
        keys = {
            "rows",
            "mismatch",
            "chi.related_mismatch",
            "chi.restricted_exact",
            "chi.no_other_candidate",
        }
        keys.update(f"matrix.{a}.{b}" for a in KINDS for b in KINDS)
        for name in ("chi", "pon"):
            keys.update(
                f"{name}.{suffix}"
                for suffix in (
                    "legal",
                    "tp",
                    "fn",
                    "fp",
                    "tn",
                    "teacher_positive",
                    "teacher_negative",
                    "non_kind",
                    "wrong_candidate",
                    "exact",
                )
            )
        keys.update(c)
        metrics = {}
        for name in ("chi", "pon"):
            positive, negative = (
                c[f"{name}.teacher_positive"],
                c[f"{name}.teacher_negative"],
            )
            if (
                c[f"{name}.non_kind"]
                + c[f"{name}.wrong_candidate"]
                + c[f"{name}.exact"]
                != positive
            ):
                raise LearnedPolicyError("teacher decomposition accounting mismatch")
            recall = fraction(c[f"{name}.tp"], positive)
            specificity = fraction(c[f"{name}.tn"], negative)
            metrics[name] = {
                "recall": recall,
                "specificity": specificity,
                "balanced_accuracy": {
                    "value": (recall["value"] + specificity["value"]) / 2
                    if positive and negative
                    else None,
                    "reason": None
                    if positive and negative
                    else "positive or negative denominator is zero",
                },
                "action_match": fraction(c[f"{name}.exact"], positive),
            }
        expected = c["chi.non_kind"] + c["chi.wrong_candidate"] + c["chi.fp"]
        if c["chi.related_mismatch"] != expected:
            raise LearnedPolicyError("Chi mismatch accounting mismatch")
        return {
            "counts": {
                key: {"rows": c[key], "hanchan": len(self.support[key])}
                for key in sorted(keys)
            },
            "metrics": metrics,
            "restricted_chi_match": fraction(
                c["chi.restricted_exact"], c["chi.teacher_positive"]
            ),
            "margins": {
                name: distribution(self.margins[name])
                for name in ("chi_vs_non_chi", "teacher_vs_other_chi")
            },
        }


def diagnostic_summaries(source, splits):
    """拒否をpayload openより先に行う。全splitをsource.games()で展開しない。"""
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(SPLITS):
        raise LearnedPolicyError("diagnostic permits TRAIN/SELECT only")
    return tuple(s for s in source.summaries if s["split"] in splits)


def collect_calls(source, scorer, summaries, check_limit=lambda: None):
    groups = {
        (split, hand, count): CallGroup()
        for split in SPLITS
        for hand in ("all", "closed", "open")
        for count in GROUPS
    }
    exclusions = {split: Counter() for split in SPLITS}
    exclusion_support = {split: defaultdict(set) for split in SPLITS}
    games = {split: 0 for split in SPLITS}
    for summary in summaries:
        # 二重の境界: callerが未検証summaryを渡しても禁止splitを開かない。
        if summary["split"] not in SPLITS or summary not in source.summaries:
            raise LearnedPolicyError("forbidden or unknown diagnostic game")
        check_limit()
        game = read_source_game(source.root, summary, schema=source.schema)
        games[game.split] += 1
        for row in game.decisions:
            check_limit()
            decision = DecisionContext(
                input=row.policy_input, legal_actions=row.legal_actions
            )
            kind = classify_l2a_decision(decision)
            if teacher_rule_exception(decision, kind, row.selected_action) is not None:
                raise LearnedPolicyError("teacher rule exception")
            if kind is not L2aDecisionKind.CALL:
                exclusions[row.split][kind.value] += 1
                exclusion_support[row.split][kind.value].add(row.seed)
                continue
            encoded = scorer.encoding.encode(decision)
            if any(
                not any(a is legal for legal in decision.legal_actions)
                for a in encoded.actions
            ):
                raise LearnedPolicyError("non-canonical legal candidate")
            scores = scorer.score(
                build_player_safe_feature(decision.input), encoded.vectors
            )
            own = row.policy_input.players[int(row.actor_seat)]
            hand = (
                "open"
                if any(m.kind is not MeldKind.ANKAN for m in own.melds)
                else "closed"
            )
            chi_count = sum(isinstance(a, ChiAction) for a in encoded.actions)
            bins = (
                ("all",)
                if chi_count == 0
                else ("all", "chi_legal", str(chi_count) if chi_count < 3 else "3+")
            )
            for h in ("all", hand):
                for count in bins:
                    groups[row.split, h, count].observe(
                        row.seed, encoded.actions, scores, row.selected_action
                    )
    result = {}
    for split in SPLITS:
        denominator = groups[split, "all", "all"].counts["mismatch"]
        result[split] = {
            "games": games[split],
            "excluded": {
                kind.value: {
                    "rows": exclusions[split][kind.value],
                    "hanchan": len(exclusion_support[split][kind.value]),
                }
                for kind in L2aDecisionKind
                if kind is not L2aDecisionKind.CALL
            },
            "groups": {
                hand: {count: groups[split, hand, count].report() for count in GROUPS}
                for hand in ("all", "closed", "open")
            },
            "closed_chi_contribution": fraction(
                groups[split, "closed", "all"].counts["chi.related_mismatch"],
                denominator,
            ),
        }
    return result


def validate_inputs(source, scorer):
    if (
        source.identity != SOURCE_ID
        or source.schema != POLICY_SOURCE_RECORD_SCHEMA_V2
        or source.purpose != SCIENTIFIC_PURPOSE
    ):
        raise LearnedPolicyError("#318 source identity/schema/purpose mismatch")
    if (
        scorer.artifact.identity != R1_ID
        or scorer.artifact.source_identity != source.identity
        or scorer.encoding is not ENCODING_R1
    ):
        raise LearnedPolicyError("frozen #318 R1 identity/source/encoding mismatch")
    for split, (first, last) in CONFIG["seed_ranges"].items():
        seeds = [s["seed"] for s in diagnostic_summaries(source, (split,))]
        if seeds != list(range(first, last + 1)):
            raise LearnedPolicyError("diagnostic population/seed mismatch")


def runtime_record():
    from lisjong.hand_evaluation._shanten_backend import BACKEND_NAME
    from lisjong.learning.model import require_torch

    torch = require_torch()
    import _lisjong_native

    native = {
        "api_version": _lisjong_native.API_VERSION,
        "source_revision": _lisjong_native.SOURCE_REVISION,
        "file": file_digest(Path(_lisjong_native.__file__)),
    }
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise LearnedPolicyError("commit the diagnostic implementation before running")
    return {
        "revision": revision,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "backend": BACKEND_NAME,
        "native": native,
    }


def run(source_path, artifact_path, *, phase, preflight_path=None):
    if sys.platform != "linux":
        raise LearnedPolicyError("diagnostic resource accounting requires Linux")
    import resource

    if phase not in ("preflight", "full"):
        raise LearnedPolicyError("unknown diagnostic phase")
    started = perf_counter()

    def check_limit():
        if perf_counter() - started > CONFIG["max_wall_seconds"]:
            raise LearnedPolicyError("wall time limit exceeded")
        if (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            > CONFIG["max_rss_mib"]
        ):
            raise LearnedPolicyError("RSS limit exceeded")

    source = open_source_record(source_path)  # manifest only; no payload
    scorer = load_l2a_scorer(artifact_path)
    validate_inputs(source, scorer)
    environment = runtime_record()
    if environment["backend"] != "rust":
        raise LearnedPolicyError("production diagnosis requires the Rust backend")
    if phase == "full":
        if preflight_path is None:
            raise LearnedPolicyError("full diagnosis requires matching preflight")
        preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
        unseal(preflight, LearnedPolicyError, "preflight report")
        if (
            preflight.get("config") != CONFIG
            or preflight.get("phase") != "preflight"
            or preflight.get("status") != "COMPLETE"
            or preflight.get("environment") != environment
            or preflight.get("within_budget") is not True
        ):
            raise LearnedPolicyError("preflight identity/environment/budget mismatch")
        summaries = diagnostic_summaries(source, SPLITS)
    else:
        summaries = diagnostic_summaries(source, ("TRAIN",))[:10]
    collected = collect_calls(source, scorer, summaries, check_limit)
    check_limit()
    wall = perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    # setupも70倍する保守的な外挿。対象10半荘の偏りは残るので実行中も監視。
    extrapolated = wall * 70 if phase == "preflight" else None
    return seal(
        {
            "config": CONFIG,
            "phase": phase,
            "status": "COMPLETE",
            "environment": environment,
            "source_contract_digest": source.source_contract_digest,
            "allocation_bindings": source.allocation_bindings,
            "artifact_training": scorer.artifact.manifest["training"],
            "artifact_dataset": scorer.artifact.manifest["dataset"],
            "artifact_encoding": scorer.artifact.manifest["encoding"],
            "artifact_weights": scorer.artifact.manifest["files"],
            "summary": collected,
            "wall_seconds": wall,
            "peak_rss_mib": peak,
            "extrapolated_full_wall_seconds": extrapolated,
            "within_budget": peak <= CONFIG["max_rss_mib"]
            and (extrapolated is None or extrapolated <= CONFIG["max_wall_seconds"]),
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--r1", required=True)
    parser.add_argument("--phase", choices=("preflight", "full"), required=True)
    parser.add_argument("--preflight")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # 失敗でも旧report・source・artifactを上書きしない。
    if args.output.exists():
        parser.error("output already exists")
    for protected in (Path(args.source), Path(args.r1)):
        if args.output.resolve().is_relative_to(protected.resolve()):
            parser.error("output must be outside source/artifact")
    try:
        report = run(
            args.source, args.r1, phase=args.phase, preflight_path=args.preflight
        )
    except (LearningError, OSError, ValueError, subprocess.SubprocessError) as exc:
        report = seal(
            {
                "config": CONFIG,
                "phase": args.phase,
                "status": "STOP",
                "reason": str(exc),
            }
        )
    write_new_text(args.output, canonical_json_text(report), LearnedPolicyError)
    print(
        json.dumps(
            {"status": report["status"], "output": str(args.output)}, ensure_ascii=False
        )
    )
    return 0 if report["status"] == "COMPLETE" and report["within_budget"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
