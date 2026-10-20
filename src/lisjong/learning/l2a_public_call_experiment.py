"""#326: 凍結R1（A）とR1-public-call（B）の限定開発比較。TRAIN/SELECTだけを開く。

```text
preflight    TRAINの最初の10半荘でA/B encodingと採点を計時し、時間・memoryを見積もる
lock         Aのmanifestから学習条件を全数値継承し、identity・環境・予算を固定する
baseline     Aを再採点して#324の件数と照合し、未知のbaseline件数を機械算出する
materialize  TRAIN/SELECTだけに限定したsource viewからBのdatasetを作る
fit          Bを幅256で1回だけscratchから学習する（最初のepochで残予算を判定）
compare      A/Bを同じ行で採点し、paired cluster bootstrapと事前固定の判定を出す
```

python -m lisjong.learning.l2a_public_call_experiment <phase> --help
各phaseのreportは前のreportのdigestにbindする。OFFLINE-EVALのpayloadは開かない。
集計helperは合成testから使う。
"""

import argparse
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from math import exp, log
from pathlib import Path
from time import perf_counter

from lisjong.learning._canonical import (
    canonical_json_text,
    file_digest,
    seal,
    unseal,
    value_digest,
)
from lisjong.learning._publication import write_new_text
from lisjong.learning.errors import LearnedPolicyError, LearningError
from lisjong.learning.features import FEATURE_DIMENSION, build_player_safe_feature
from lisjong.learning.l2a_call_diagnostic import (
    GROUPS as REFERENCE_GROUPS,
)
from lisjong.learning.l2a_call_diagnostic import (
    R1_ID,
    SOURCE_ID,
    CallGroup,
    diagnostic_summaries,
    fraction,
    runtime_record,
    validate_inputs,
)
from lisjong.learning.l2a_decision import (
    L2aDecisionKind,
    classify_l2a_decision,
    teacher_rule_exception,
)
from lisjong.learning.l2a_encoding import ENCODING_R1, ENCODING_R1_PUBLIC_CALL
from lisjong.learning.l2a_model import L2aScorerConfig
from lisjong.learning.source_record import read_source_game
from lisjong.policy_contract import ChiAction, DecisionContext
from lisjong.policy_contract.meld import MeldKind

SPLITS = ("TRAIN", "SELECT")
HANDS = ("all", "closed", "open")
BINS = ("all", "chi_legal", "pon_only", "1", "2", "3+")
REFERENCE_324_SHA256 = (
    "6d95bd0881ad95611bdd01a8e3257e9d91242b2d6f7b3393a763891bca2e9f58"
)
"""#324の`full.json`（凍結Aの診断report）。baselineの照合先。"""

CRITERIA = {
    "closed_chi_recall_gain_minimum": 0.15,
    "closed_chi_recall_gain_ci_lower_exclusive": 0.0,
    "closed_chi_gap_ratio_maximum": 0.5,
    "closed_chi_train_recall_drop_maximum": 0.02,
    "closed_false_chi_maximum": 6,
    "open_false_chi_maximum": 36,
    "open_chi_recall_minimum": [114, 188],
    "chi_wrong_candidate_maximum": 0,
    "pon_recall_minimum": [565, 727],
    "false_pon_not_above_a": ["all", "closed", "open"],
    "closed_pon_only_false_pon_maximum": 196,
    "all_mismatch_maximum": 548,
    "closed_mismatch_maximum": 396,
}
"""Issue #326の主判定（SELECT）。Bの結果を見る前に固定した開発上の選別基準。"""

EXPECTED_A = {
    "TRAIN/closed/chi_legal/chi.tp": 679,
    "TRAIN/closed/chi_legal/chi.teacher_positive": 699,
    "SELECT/closed/chi_legal/chi.tp": 21,
    "SELECT/closed/chi_legal/chi.teacher_positive": 93,
    "SELECT/closed/chi_legal/chi.fp": 6,
    "SELECT/open/chi_legal/chi.fp": 36,
    "SELECT/open/chi_legal/chi.tp": 114,
    "SELECT/open/chi_legal/chi.teacher_positive": 188,
    "SELECT/all/all/pon.tp": 565,
    "SELECT/all/all/pon.teacher_positive": 727,
    "SELECT/closed/pon_only/pon.fp": 196,
    "SELECT/all/all/mismatch": 548,
    "SELECT/closed/all/mismatch": 396,
}
"""Issueに記載した既知のA件数。baselineで再採点結果と機械照合する。"""

SEED_KEYS = (
    "rows",
    "mismatch",
    "chi.tp",
    "chi.fp",
    "chi.teacher_positive",
    "chi.wrong_candidate",
    "pon.tp",
    "pon.fp",
    "pon.teacher_positive",
)
SEED_GROUPS = (
    ("all", "all"),
    ("closed", "all"),
    ("open", "all"),
    ("closed", "chi_legal"),
    ("open", "chi_legal"),
    ("closed", "pon_only"),
)
"""半荘ごとに保存する件数（bootstrapと半荘ごとの対応差）。"""

BOOTSTRAP_METRICS = (
    ("closed_chi_recall", ("closed", "chi_legal"), "chi.tp", "chi.teacher_positive"),
    ("open_chi_recall", ("open", "chi_legal"), "chi.tp", "chi.teacher_positive"),
    ("pon_recall", ("all", "all"), "pon.tp", "pon.teacher_positive"),
    ("closed_false_chi", ("closed", "chi_legal"), "chi.fp", None),
    ("open_false_chi", ("open", "chi_legal"), "chi.fp", None),
    ("all_mismatch", ("all", "all"), "mismatch", None),
    ("closed_mismatch", ("closed", "all"), "mismatch", None),
)
"""B-Aの差。先頭が主指標、残りは記述用。分母なしは件数差。"""

CONFIG = {
    "identity": "lisjong-l2a-public-call-experiment-326-v1",
    "source": SOURCE_ID,
    "a": {"identity": R1_ID, "selected_epoch": 5},
    "b_encoding": ENCODING_R1_PUBLIC_CALL.block(),
    "splits": list(SPLITS),
    "seed_ranges": {"TRAIN": [506000, 506599], "SELECT": [506600, 506699]},
    "preflight_games": 10,
    "budget": {"max_wall_seconds": 7200, "max_rss_mib": 8192, "margin": 1.5},
    "reference_324_sha256": REFERENCE_324_SHA256,
    "groups": {"hands": list(HANDS), "bins": list(BINS)},
    "bootstrap": {
        "replicates": 10000,
        "seed": 324,
        "rng": "random.Random(seed).choices(ascending SELECT seeds, k=100)",
        "statistic": "pooled numerator/denominator over sampled hanchan (B-A)",
        "quantiles": [0.025, 0.975],
        "interpolation": "linear at (n-1)*p",
        "zero_denominator": "indeterminate -> STOP",
    },
    "criteria": CRITERIA,
    "expected_a": EXPECTED_A,
    "interpretation": (
        "exploratory development comparison on reused TRAIN/SELECT; one training "
        "seed; not an independent evaluation, strength or promotion claim"
    ),
}


# --- 集計 -----------------------------------------------------------------------


class SeedCallGroup(CallGroup):
    """#324のCallGroupに、半荘ごとの件数を加えたもの。"""

    def __init__(self):
        super().__init__()
        self.by_seed = defaultdict(Counter)

    def add(self, seed, key):
        super().add(seed, key)
        self.by_seed[seed][key] += 1

    def seed_counts(self):
        return {
            str(seed): {key: counts[key] for key in SEED_KEYS}
            for seed, counts in sorted(self.by_seed.items())
        }


def hand_kind(policy_input, actor_seat):
    melds = policy_input.players[int(actor_seat)].melds
    return "open" if any(m.kind is not MeldKind.ANKAN for m in melds) else "closed"


def row_bins(actions):
    chi_count = sum(isinstance(a, ChiAction) for a in actions)
    if chi_count == 0:
        return ("all", "pon_only")
    return ("all", "chi_legal", str(chi_count) if chi_count < 3 else "3+")


def cross_entropy(scores, teacher_index):
    top = max(scores)
    return top + log(sum(exp(s - top) for s in scores)) - scores[teacher_index]


class ModelTally:
    """1 modelのsplit・手・候補数bin別の集計とCE。"""

    def __init__(self):
        self.groups = {
            (split, hand, b): SeedCallGroup()
            for split in SPLITS
            for hand in HANDS
            for b in BINS
        }
        self.ce = {(split, hand): [0.0, 0] for split in SPLITS for hand in HANDS}

    def observe(self, split, hand, seed, actions, scores, teacher):
        bins = row_bins(actions)
        for h in ("all", hand):
            for b in bins:
                self.groups[split, h, b].observe(seed, actions, scores, teacher)
            ce = cross_entropy(scores, actions.index(teacher))
            self.ce[split, h][0] += ce
            self.ce[split, h][1] += 1

    def report(self):
        return {
            split: {
                "groups": {
                    hand: {b: self.groups[split, hand, b].report() for b in BINS}
                    for hand in HANDS
                },
                "cross_entropy": {
                    hand: {
                        "sum": self.ce[split, hand][0],
                        "rows": self.ce[split, hand][1],
                        "mean": self.ce[split, hand][0] / self.ce[split, hand][1]
                        if self.ce[split, hand][1]
                        else None,
                    }
                    for hand in HANDS
                },
                "by_seed": {
                    f"{hand}/{b}": self.groups[split, hand, b].seed_counts()
                    for hand, b in SEED_GROUPS
                },
            }
            for split in SPLITS
        }


def count(report, path):
    """`SPLIT/hand/bin/key`の行数。"""
    split, hand, b, key = path.split("/")
    return report[split]["groups"][hand][b]["counts"][key]["rows"]


def _float32_bytes(values):
    from array import array

    return array("f", values).tobytes()


def collect(source, scorers, summaries, check_limit=lambda: None, payload=None):
    """TRAIN/SELECTのCALL行を全modelで同じ順に採点する。

    `payload`（Bのdatasetの`iter_l2a_dataset_payload`）を渡すと、Bの候補vector・
    shared context・教師indexがdatasetとbyte単位で一致することを照合する。
    """
    tallies = {name: ModelTally() for name in scorers}
    games = Counter()
    payload = iter(payload) if payload is not None else None
    for summary in summaries:
        if summary["split"] not in SPLITS or summary not in source.summaries:
            raise LearnedPolicyError("forbidden or unknown experiment game")
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
                continue
            context = build_player_safe_feature(decision.input)
            hand = hand_kind(row.policy_input, row.actor_seat)
            actions = None
            for name, scorer in scorers.items():
                encoded = scorer.encoding.encode(decision)
                if actions is None:
                    actions = encoded.actions
                elif encoded.actions != actions:
                    raise LearnedPolicyError("A/B candidate order mismatch")
                if any(
                    not any(a is legal for legal in decision.legal_actions)
                    for a in encoded.actions
                ):
                    raise LearnedPolicyError("non-canonical legal candidate")
                if encoded.actions.count(row.selected_action) != 1:
                    raise LearnedPolicyError("teacher does not resolve uniquely")
                if payload is not None and scorer.encoding is ENCODING_R1_PUBLIC_CALL:
                    _check_payload(next(payload, None), row, context, encoded)
                scores = scorer.score(context, encoded.vectors)
                tallies[name].observe(
                    row.split,
                    hand,
                    row.seed,
                    encoded.actions,
                    scores,
                    row.selected_action,
                )
    if payload is not None and next(payload, None) is not None:
        raise LearnedPolicyError("dataset has rows that were not re-encoded")
    return {
        "games": {split: games[split] for split in SPLITS},
        "models": {name: tally.report() for name, tally in tallies.items()},
    }


def _check_payload(item, row, context, encoded):
    if item is None:
        raise LearnedPolicyError("dataset ended before the source CALL rows")
    stored = item.row
    if (stored.game_ordinal, stored.decision_ordinal, stored.seed) != (
        row.game_ordinal,
        row.decision_ordinal,
        row.seed,
    ):
        raise LearnedPolicyError("dataset row order does not match the source")
    if (
        stored.actions != encoded.actions
        or stored.actions[stored.teacher_candidate_index] != row.selected_action
    ):
        raise LearnedPolicyError("dataset candidates/label do not match serving")
    if item.context != _float32_bytes(context) or item.candidates != b"".join(
        _float32_bytes(vector) for vector in encoded.vectors
    ):
        raise LearnedPolicyError("train-serve vector mismatch")


# --- baseline照合 ---------------------------------------------------------------


def reconcile_with_324(a_report, reference):
    """#324の件数（選択action由来）と完全一致、marginは丸め差として分けて記録する。"""
    count_mismatches = []
    margin_differences = []
    for split in SPLITS:
        for hand in HANDS:
            for b in REFERENCE_GROUPS:
                ours = a_report[split]["groups"][hand][b]
                theirs = reference["summary"][split]["groups"][hand][b]
                if ours["counts"] != theirs["counts"]:
                    keys = sorted(set(ours["counts"]) | set(theirs["counts"]))
                    count_mismatches.extend(
                        f"{split}/{hand}/{b}/{key}"
                        for key in keys
                        if ours["counts"].get(key) != theirs["counts"].get(key)
                    )
                for name, ours_margin in ours["margins"].items():
                    theirs_margin = theirs["margins"][name]
                    for field, value in ours_margin.items():
                        other = theirs_margin[field]
                        if value == other:
                            continue
                        if (
                            isinstance(value, float)
                            and isinstance(other, float)
                            and abs(value - other) <= 1e-6 * max(1.0, abs(other))
                        ):
                            margin_differences.append(
                                f"{split}/{hand}/{b}/{name}.{field}"
                            )
                        else:
                            count_mismatches.append(
                                f"{split}/{hand}/{b}/{name}.{field}"
                            )
    return {
        "status": "MATCH" if not count_mismatches else "MISMATCH",
        "action_count_mismatches": count_mismatches,
        "margin_rounding_differences": margin_differences,
    }


def expected_a_check(a_report):
    observed = {path: count(a_report, path) for path in EXPECTED_A}
    return {
        "status": "MATCH" if observed == EXPECTED_A else "MISMATCH",
        "observed": observed,
    }


def derived_thresholds(a_report):
    """Bの結果を見る前に固定Aから機械算出するbaseline件数。"""
    return {
        f"SELECT/{hand}/all/pon.fp": count(a_report, f"SELECT/{hand}/all/pon.fp")
        for hand in HANDS
    }


# --- bootstrap ------------------------------------------------------------------


def percentile(ordered, p):
    position = (len(ordered) - 1) * p
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_bootstrap(a_by_seed, b_by_seed, *, replicates, seed):
    """SELECT半荘を復元抽出し、同じmultisetでA/Bの差を計算する。

    `*_by_seed`は`{"hand/bin": {seed: {key: count}}}`。各replicateの比率は抽出した
    行の分子・分母を合算する。主指標の分母0が出たら`INDETERMINATE`。
    """
    seeds = sorted(
        {int(s) for group in a_by_seed.values() for s in group}
        | {int(s) for group in b_by_seed.values() for s in group}
    )
    if not seeds:
        raise LearnedPolicyError("bootstrap has no SELECT hanchan")
    rng = random.Random(seed)
    empty = Counter()

    def total(by_seed, group, key, sample):
        counts = by_seed[f"{group[0]}/{group[1]}"]
        return sum(counts.get(str(s), empty).get(key, 0) for s in sample)

    differences = {name: [] for name, *_ in BOOTSTRAP_METRICS}
    indeterminate = {name: 0 for name, *_ in BOOTSTRAP_METRICS}
    for _ in range(replicates):
        sample = rng.choices(seeds, k=len(seeds))
        for name, group, numerator, denominator in BOOTSTRAP_METRICS:
            a = total(a_by_seed, group, numerator, sample)
            b = total(b_by_seed, group, numerator, sample)
            if denominator is None:
                differences[name].append(float(b - a))
                continue
            d = total(a_by_seed, group, denominator, sample)
            if d != total(b_by_seed, group, denominator, sample):
                raise LearnedPolicyError("A/B teacher denominators differ")
            if d == 0:
                indeterminate[name] += 1
                continue
            differences[name].append((b - a) / d)
    result = {}
    for name, *_ in BOOTSTRAP_METRICS:
        values = sorted(differences[name])
        if indeterminate[name]:
            result[name] = {
                "status": "INDETERMINATE",
                "zero_denominator_replicates": indeterminate[name],
            }
            continue
        result[name] = {
            "status": "COMPLETE",
            "replicates": len(values),
            "lower": percentile(values, 0.025),
            "upper": percentile(values, 0.975),
        }
    return {"hanchan": len(seeds), "metrics": result}


# --- 判定 -----------------------------------------------------------------------


def _ratio(report, split, hand, b, numerator, denominator):
    n = count(report, f"{split}/{hand}/{b}/{numerator}")
    d = count(report, f"{split}/{hand}/{b}/{denominator}")
    return fraction(n, d)


def judge(a, b, bootstrap, baseline_thresholds):
    """Issue #326の主判定。全条件を満たすときだけPROMISING-DEVELOPMENT。"""
    c = CRITERIA
    checks = []

    def check(name, passed, **values):
        checks.append({"name": name, "passed": bool(passed), **values})

    recall = {
        (model, split): _ratio(
            report, split, "closed", "chi_legal", "chi.tp", "chi.teacher_positive"
        )
        for model, report in (("A", a), ("B", b))
        for split in SPLITS
    }
    if any(item["value"] is None for item in recall.values()):
        raise LearnedPolicyError("closed Chi recall has a zero denominator")
    gain = recall["B", "SELECT"]["value"] - recall["A", "SELECT"]["value"]
    gained_rows = (
        recall["B", "SELECT"]["numerator"] - recall["A", "SELECT"]["numerator"]
    )
    positives = recall["A", "SELECT"]["denominator"]
    primary = bootstrap["metrics"]["closed_chi_recall"]
    if primary["status"] != "COMPLETE":
        return {
            "verdict": "STOP/INVALID",
            "reason": "primary bootstrap interval is indeterminate",
            "checks": checks,
        }
    check(
        "1.closed_chi_recall_gain",
        gained_rows >= c["closed_chi_recall_gain_minimum"] * positives
        and primary["lower"] > c["closed_chi_recall_gain_ci_lower_exclusive"],
        a=recall["A", "SELECT"],
        b=recall["B", "SELECT"],
        gain=gain,
        interval=[primary["lower"], primary["upper"]],
    )
    gap_a = recall["A", "TRAIN"]["value"] - recall["A", "SELECT"]["value"]
    gap_b = recall["B", "TRAIN"]["value"] - recall["B", "SELECT"]["value"]
    drop = recall["A", "TRAIN"]["value"] - recall["B", "TRAIN"]["value"]
    check(
        "2.closed_chi_train_select_gap",
        abs(gap_b) <= c["closed_chi_gap_ratio_maximum"] * abs(gap_a)
        and drop <= c["closed_chi_train_recall_drop_maximum"],
        a_gap=gap_a,
        b_gap=gap_b,
        gap_limit=c["closed_chi_gap_ratio_maximum"] * abs(gap_a),
        a_train=recall["A", "TRAIN"],
        b_train=recall["B", "TRAIN"],
        train_drop=drop,
    )
    closed_fp = count(b, "SELECT/closed/chi_legal/chi.fp")
    open_fp = count(b, "SELECT/open/chi_legal/chi.fp")
    open_tp = count(b, "SELECT/open/chi_legal/chi.tp")
    open_positive = count(b, "SELECT/open/chi_legal/chi.teacher_positive")
    numerator, denominator = c["open_chi_recall_minimum"]
    wrong = {
        f"{hand}/{bin_}": count(b, f"SELECT/{hand}/{bin_}/chi.wrong_candidate")
        for hand in HANDS
        for bin_ in BINS
    }
    check(
        "3.false_chi_open_chi_wrong_candidate",
        closed_fp <= c["closed_false_chi_maximum"]
        and open_fp <= c["open_false_chi_maximum"]
        and open_tp * denominator >= numerator * open_positive
        and max(wrong.values()) <= c["chi_wrong_candidate_maximum"],
        closed_false_chi=closed_fp,
        open_false_chi=open_fp,
        open_chi_recall=fraction(open_tp, open_positive),
        wrong_candidate=wrong,
    )
    pon_tp = count(b, "SELECT/all/all/pon.tp")
    pon_positive = count(b, "SELECT/all/all/pon.teacher_positive")
    numerator, denominator = c["pon_recall_minimum"]
    false_pon = {
        hand: {
            "a": baseline_thresholds[f"SELECT/{hand}/all/pon.fp"],
            "b": count(b, f"SELECT/{hand}/all/pon.fp"),
        }
        for hand in c["false_pon_not_above_a"]
    }
    pon_only_fp = count(b, "SELECT/closed/pon_only/pon.fp")
    check(
        "4.pon",
        pon_tp * denominator >= numerator * pon_positive
        and all(item["b"] <= item["a"] for item in false_pon.values())
        and pon_only_fp <= c["closed_pon_only_false_pon_maximum"],
        pon_recall=fraction(pon_tp, pon_positive),
        false_pon=false_pon,
        closed_pon_only_false_pon=pon_only_fp,
    )
    all_mismatch = count(b, "SELECT/all/all/mismatch")
    closed_mismatch = count(b, "SELECT/closed/all/mismatch")
    check(
        "5.call_mismatch",
        all_mismatch <= c["all_mismatch_maximum"]
        and closed_mismatch <= c["closed_mismatch_maximum"],
        all_mismatch=all_mismatch,
        closed_mismatch=closed_mismatch,
    )
    passed = all(item["passed"] for item in checks)
    return {
        "verdict": "PROMISING-DEVELOPMENT" if passed else "NOT-SUPPORTED-DEVELOPMENT",
        "checks": checks,
    }


# --- 実行phase ------------------------------------------------------------------


def restricted_source(source):
    """TRAIN/SELECTのgameとallocation bindingだけを持つsource view。

    identityは元のsourceのまま、populationが限定されたことをprovenanceに残す。
    OFFLINE-EVALのsummaryを含まないので、materializationはそのpayloadを開けない。
    """
    summaries = diagnostic_summaries(source, SPLITS)
    if [s["game_ordinal"] for s in summaries] != list(range(len(summaries))):
        raise LearnedPolicyError("TRAIN/SELECT are not the leading source games")
    bindings = source.allocation_bindings
    return replace(
        source,
        summaries=summaries,
        allocation_bindings=None
        if bindings is None
        else {split: bindings[split] for split in SPLITS},
    )


def _read_report(path, phase, lock_digest=None):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unseal(report, LearnedPolicyError, f"{phase} report")
    if (
        report.get("config") != CONFIG
        or report.get("phase") != phase
        or report.get("status") != "COMPLETE"
    ):
        raise LearnedPolicyError(f"{phase} report config/phase/status mismatch")
    if lock_digest is not None and report.get("lock") != lock_digest:
        raise LearnedPolicyError(f"{phase} report is bound to another lock")
    return report


class Budget:
    """lockの予算に対する協調的check。OS側の監視（timeout・time -v）と併用する。"""

    def __init__(self, spent_seconds=0.0):
        import resource

        self.resource = resource
        self.spent = spent_seconds
        self.started = perf_counter()

    @property
    def elapsed(self):
        return perf_counter() - self.started

    def remaining(self):
        return CONFIG["budget"]["max_wall_seconds"] - self.spent - self.elapsed

    def peak_rss_mib(self):
        usage = self.resource.getrusage
        return (
            max(
                usage(self.resource.RUSAGE_SELF).ru_maxrss,
                usage(self.resource.RUSAGE_CHILDREN).ru_maxrss,
            )
            / 1024
        )

    def check(self):
        if self.remaining() < 0:
            raise LearnedPolicyError("wall time budget exceeded")
        if self.peak_rss_mib() > CONFIG["budget"]["max_rss_mib"]:
            raise LearnedPolicyError("RSS budget exceeded")


def _locked_budget(args, bound_phases):
    """lockと前段reportを検証し、それらのwall合計を消費済みとするBudgetを返す。"""
    preflight = _read_report(args.preflight, "preflight")
    lock = _read_report(args.lock, "lock")
    if lock["preflight"] != value_digest(preflight):
        raise LearnedPolicyError("lock is bound to another preflight")
    lock_digest = value_digest(lock)
    reports = {
        phase: _read_report(getattr(args, phase), phase, lock_digest)
        for phase in bound_phases
    }
    spent = preflight["wall_seconds"] + lock["wall_seconds"]
    spent += sum(report["wall_seconds"] for report in reports.values())
    return lock, lock_digest, reports, Budget(spent)


def _open_inputs(source_path, a_path):
    from lisjong.learning.l2a_policy import load_l2a_scorer
    from lisjong.learning.source_record import open_source_record

    source = open_source_record(source_path)  # manifestだけ。payloadは開かない
    a = load_l2a_scorer(a_path)
    validate_inputs(source, a)
    if (
        a.artifact.manifest["training"]["selected_epoch"]
        != (CONFIG["a"]["selected_epoch"])
    ):
        raise LearnedPolicyError("frozen A selected epoch mismatch")
    return source, a


def _environment():
    environment = runtime_record()
    if environment["backend"] != "rust":
        raise LearnedPolicyError("experiment requires the Rust shanten backend")
    if environment["torch_threads"] != 1:
        raise LearnedPolicyError("experiment requires one torch thread")
    return environment


def phase_preflight(args):
    budget = Budget()
    source, a = _open_inputs(args.source, args.a)
    environment = _environment()
    summaries = diagnostic_summaries(source, ("TRAIN",))[: CONFIG["preflight_games"]]
    timers = Counter()
    rows = Counter()
    for summary in summaries:
        budget.check()
        started = perf_counter()
        game = read_source_game(source.root, summary, schema=source.schema)
        decisions = []
        for row in game.decisions:
            decision = DecisionContext(
                input=row.policy_input, legal_actions=row.legal_actions
            )
            if classify_l2a_decision(decision) is L2aDecisionKind.CALL:
                decisions.append(decision)
        timers["read_classify"] += perf_counter() - started
        for decision in decisions:
            started = perf_counter()
            context = build_player_safe_feature(decision.input)
            timers["shared_context"] += perf_counter() - started
            started = perf_counter()
            encoded_a = ENCODING_R1.encode(decision)
            timers["encode_a"] += perf_counter() - started
            started = perf_counter()
            encoded_b = ENCODING_R1_PUBLIC_CALL.encode(decision)
            timers["encode_b"] += perf_counter() - started
            started = perf_counter()
            a.score(context, encoded_a.vectors)
            timers["score"] += perf_counter() - started
            if encoded_a.actions != encoded_b.actions:
                raise LearnedPolicyError("A/B candidate order mismatch")
            rows["call"] += 1
            rows["candidates"] += len(encoded_b.actions)
    scale = sum(len(diagnostic_summaries(source, (split,))) for split in SPLITS) / len(
        summaries
    )
    t = {name: value * scale for name, value in timers.items()}
    estimates = {
        "baseline": t["read_classify"]
        + t["shared_context"]
        + t["encode_a"]
        + t["score"],
        "materialize": t["read_classify"] + t["shared_context"] + t["encode_b"],
        "compare": t["read_classify"]
        + t["shared_context"]
        + t["encode_a"]
        + t["encode_b"]
        + 2 * t["score"],
    }
    call_rows = rows["call"] * scale
    candidates = rows["candidates"] * scale
    payload_bytes = 4 * (
        call_rows * FEATURE_DIMENSION + candidates * ENCODING_R1_PUBLIC_CALL.dimension
    )
    wall = budget.elapsed
    return {
        "environment": environment,
        "preflight_games": [s["seed"] for s in summaries],
        "sample": {"call_rows": rows["call"], "candidates": rows["candidates"]},
        "sample_seconds": dict(sorted(timers.items())),
        "extrapolation_factor": scale,
        "estimated_seconds": estimates,
        "estimated_payload_bytes": payload_bytes,
        "estimated_training_memory_mib": 4 * payload_bytes / 2**20,
        "wall_seconds": wall,
        "peak_rss_mib": budget.peak_rss_mib(),
        "note": "fit time is measured by the first formal epoch; not estimated here",
        "within_budget": (
            wall + CONFIG["budget"]["margin"] * sum(estimates.values())
            <= CONFIG["budget"]["max_wall_seconds"]
        )
        and 4 * payload_bytes / 2**20 <= CONFIG["budget"]["max_rss_mib"],
    }


def phase_lock(args):
    import os

    from lisjong.learning.model import require_torch

    budget = Budget()
    preflight = _read_report(args.preflight, "preflight")
    if preflight["within_budget"] is not True:
        raise LearnedPolicyError("preflight is not within budget")
    source, a = _open_inputs(args.source, args.a)
    environment = _environment()
    if preflight["environment"] != environment:
        raise LearnedPolicyError("preflight environment mismatch")
    manifest = a.artifact.manifest
    training = manifest["training"]
    torch = require_torch()
    if training["framework"]["version"] != str(torch.__version__):
        raise LearnedPolicyError("torch version differs from frozen A")
    fit = {
        "hidden_width": manifest["model"]["hidden_width"],
        "epochs": training["epochs"],
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "seed": training["seed"],
        "train_splits": training["train_splits"],
        "select_splits": training["validation_splits"],
        "objective": training["objective"],
        "optimizer": training["optimizer"],
        "framework": training["framework"],
        "selection": "minimum SELECT mean cross entropy over all epochs; first on tie",
    }
    if fit["train_splits"] != ["TRAIN"] or fit["select_splits"] != ["SELECT"]:
        raise LearnedPolicyError("frozen A splits are not TRAIN/SELECT")
    a_parameters = a.artifact.model_config.parameter_count
    b_parameters = L2aScorerConfig(
        encoding=ENCODING_R1_PUBLIC_CALL, hidden_width=fit["hidden_width"]
    ).parameter_count
    return {
        "preflight": value_digest(preflight),
        "environment": environment,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "source": {
            "identity": source.identity,
            "schema": source.schema,
            "purpose": source.purpose,
            "source_contract_digest": source.source_contract_digest,
            "allocation_bindings": source.allocation_bindings,
        },
        "a": {
            "identity": a.artifact.identity,
            "encoding": manifest["encoding"],
            "dataset": manifest["dataset"],
            "weights": manifest["files"],
            "training": training,
            "parameters": a_parameters,
        },
        "b": {
            "encoding": ENCODING_R1_PUBLIC_CALL.block(),
            "specification": ENCODING_R1_PUBLIC_CALL.specification,
            "parameters": b_parameters,
            "added_parameters": b_parameters - a_parameters,
        },
        "fit": fit,
        "budget": CONFIG["budget"],
        "preflight_estimates": preflight["estimated_seconds"],
        "wall_seconds": budget.elapsed,
        "peak_rss_mib": budget.peak_rss_mib(),
    }


def phase_baseline(args):
    lock, lock_digest, _reports, budget = _locked_budget(args, ())
    reference_path = Path(args.reference_324)
    if file_digest(reference_path)["sha256"] != REFERENCE_324_SHA256:
        raise LearnedPolicyError("#324 reference report digest mismatch")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    source, a = _open_inputs(args.source, args.a)
    if _environment() != lock["environment"]:
        raise LearnedPolicyError("environment differs from lock")
    collected = collect(
        source, {"A": a}, diagnostic_summaries(source, SPLITS), budget.check
    )
    a_report = collected["models"]["A"]
    reconciliation = reconcile_with_324(a_report, reference)
    expected = expected_a_check(a_report)
    status = (
        "COMPLETE"
        if reconciliation["status"] == "MATCH" and expected["status"] == "MATCH"
        else "STOP"
    )
    return {
        "status": status,
        "lock": lock_digest,
        "reconciliation_324": reconciliation,
        "expected_a": expected,
        "derived_thresholds": derived_thresholds(a_report),
        "games": collected["games"],
        "a": a_report,
        "wall_seconds": budget.elapsed,
        "peak_rss_mib": budget.peak_rss_mib(),
    }


def phase_materialize(args):
    from lisjong.learning.l2a_dataset import publish_l2a_dataset
    from lisjong.learning.source_record import open_source_record

    lock, lock_digest, _reports, budget = _locked_budget(args, ("baseline",))
    source = open_source_record(args.source)
    if source.identity != SOURCE_ID:
        raise LearnedPolicyError("source identity mismatch")
    if _environment() != lock["environment"]:
        raise LearnedPolicyError("environment differs from lock")
    view = restricted_source(source)
    manifest = publish_l2a_dataset(
        view, args.dataset, encoding=ENCODING_R1_PUBLIC_CALL, workers=1
    )
    budget.check()
    return {
        "lock": lock_digest,
        "dataset": {
            "identity": manifest["identity"],
            "files": manifest["files"],
            "rows": manifest["rows"],
            "encoding": manifest["encoding"],
            "population_splits": sorted(
                {entry["split"] for entry in manifest["source"]["population"]}
            ),
        },
        "wall_seconds": budget.elapsed,
        "peak_rss_mib": budget.peak_rss_mib(),
    }


def phase_fit(args):
    from lisjong.learning.l2a_dataset import read_l2a_dataset
    from lisjong.learning.l2a_training import L2aTrainingConfig, train_l2a_scorer

    lock, lock_digest, reports, budget = _locked_budget(
        args, ("baseline", "materialize")
    )
    materialized = reports["materialize"]
    if _environment() != lock["environment"]:
        raise LearnedPolicyError("environment differs from lock")
    payload_bytes = sum(
        materialized["dataset"]["files"][name]["bytes"]
        for name in ("context.f32", "candidates.f32")
    )
    # float32 payload・bytearray copy・tensor・batch中間値の保守的な見積り。
    memory_estimate_mib = 4 * payload_bytes / 2**20
    if memory_estimate_mib > CONFIG["budget"]["max_rss_mib"]:
        raise LearnedPolicyError("pre-fit memory estimate exceeds the RSS budget")
    dataset = read_l2a_dataset(args.dataset)
    if (
        dataset.identity != materialized["dataset"]["identity"]
        or dataset.encoding is not ENCODING_R1_PUBLIC_CALL
        or dataset.manifest["source"]["identity"] != SOURCE_ID
        or set(dataset.split_counts()) != set(SPLITS)
    ):
        raise LearnedPolicyError("dataset identity/encoding/source/split mismatch")
    fit = lock["fit"]
    config = L2aTrainingConfig(
        train_splits=tuple(fit["train_splits"]),
        select_splits=tuple(fit["select_splits"]),
        hidden_width=fit["hidden_width"],
        epochs=fit["epochs"],
        batch_size=fit["batch_size"],
        learning_rate=fit["learning_rate"],
        weight_decay=fit["weight_decay"],
        seed=fit["seed"],
    )
    loaded_at = budget.elapsed
    epochs = []
    compare_estimate = lock["preflight_estimates"]["compare"]

    def on_epoch(epoch, train_ce, select_ce):
        epochs.append(
            {
                "epoch": epoch,
                "elapsed_seconds": budget.elapsed,
                "peak_rss_mib": budget.peak_rss_mib(),
            }
        )
        budget.check()
        if epoch == 1:
            first = budget.elapsed - loaded_at
            remaining = first * (config.epochs - 1) + compare_estimate
            epochs[0]["projected_remaining_seconds"] = remaining
            if CONFIG["budget"]["margin"] * remaining > budget.remaining():
                raise LearnedPolicyError(
                    "first epoch projects the run beyond the remaining budget"
                )

    artifact = train_l2a_scorer(dataset, config, args.artifact, on_epoch=on_epoch)
    return {
        "lock": lock_digest,
        "dataset_identity": dataset.identity,
        "pre_fit_memory_estimate_mib": memory_estimate_mib,
        "dataset_load_seconds": loaded_at,
        "artifact": {
            "identity": artifact.identity,
            "training": artifact.manifest["training"],
            "weights": artifact.manifest["files"],
            "parameters": artifact.model_config.parameter_count,
        },
        "epochs": epochs,
        "wall_seconds": budget.elapsed,
        "peak_rss_mib": budget.peak_rss_mib(),
    }


def phase_compare(args):
    from lisjong.learning.l2a_dataset import iter_l2a_dataset_payload
    from lisjong.learning.l2a_policy import load_l2a_scorer

    lock, lock_digest, reports, budget = _locked_budget(
        args, ("baseline", "materialize", "fit")
    )
    baseline, materialized, fitted = (
        reports["baseline"],
        reports["materialize"],
        reports["fit"],
    )
    source, a = _open_inputs(args.source, args.a)
    if _environment() != lock["environment"]:
        raise LearnedPolicyError("environment differs from lock")
    b = load_l2a_scorer(args.b)
    training = b.artifact.manifest["training"]
    fit = lock["fit"]
    if (
        b.artifact.identity != fitted["artifact"]["identity"]
        or b.encoding is not ENCODING_R1_PUBLIC_CALL
        or b.artifact.source_identity != SOURCE_ID
        or b.artifact.dataset_identity != materialized["dataset"]["identity"]
        or b.artifact.model_config.hidden_width != fit["hidden_width"]
        or any(
            training[key] != fit[key]
            for key in (
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "seed",
                "objective",
                "optimizer",
                "framework",
            )
        )
        or training["train_splits"] != fit["train_splits"]
        or training["validation_splits"] != fit["select_splits"]
    ):
        raise LearnedPolicyError("B artifact does not match the lock / fit report")
    payload = iter_l2a_dataset_payload(args.dataset, splits=SPLITS)
    collected = collect(
        source,
        {"A": a, "B": b},
        diagnostic_summaries(source, SPLITS),
        budget.check,
        payload=payload,
    )
    a_report = collected["models"]["A"]
    b_report = collected["models"]["B"]
    if value_digest(a_report) != value_digest(baseline["a"]):
        raise LearnedPolicyError("A rescoring differs from the locked baseline")
    bootstrap = paired_bootstrap(
        a_report["SELECT"]["by_seed"],
        b_report["SELECT"]["by_seed"],
        replicates=CONFIG["bootstrap"]["replicates"],
        seed=CONFIG["bootstrap"]["seed"],
    )
    judgement = judge(a_report, b_report, bootstrap, baseline["derived_thresholds"])
    return {
        "lock": lock_digest,
        "baseline": value_digest(baseline),
        "a_identity": a.artifact.identity,
        "b_identity": b.artifact.identity,
        "b_selected_epoch": training["selected_epoch"],
        "b_epoch_cross_entropy": training["diagnostics"],
        "games": collected["games"],
        "a": a_report,
        "b": b_report,
        "bootstrap": bootstrap,
        "judgement": judgement,
        "wall_seconds": budget.elapsed,
        "peak_rss_mib": budget.peak_rss_mib(),
    }


PHASES = {
    "preflight": phase_preflight,
    "lock": phase_lock,
    "baseline": phase_baseline,
    "materialize": phase_materialize,
    "fit": phase_fit,
    "compare": phase_compare,
}
_ARGUMENTS = {
    "preflight": ("source", "a"),
    "lock": ("source", "a", "preflight"),
    "baseline": ("source", "a", "preflight", "lock", "reference_324"),
    "materialize": ("source", "preflight", "lock", "baseline", "dataset"),
    "fit": ("preflight", "lock", "baseline", "materialize", "dataset", "artifact"),
    "compare": (
        "source",
        "a",
        "b",
        "dataset",
        "preflight",
        "lock",
        "baseline",
        "materialize",
        "fit",
    ),
}


def run(phase, args):
    if sys.platform != "linux":
        raise LearnedPolicyError("experiment resource accounting requires Linux")
    body = PHASES[phase](args)
    return seal({"config": CONFIG, "phase": phase, "status": "COMPLETE", **body})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase, names in _ARGUMENTS.items():
        command = commands.add_parser(phase)
        for name in names:
            command.add_argument(f"--{name.replace('_', '-')}", required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # 失敗でも旧report・source・artifactを上書きしない。
    if args.output.exists():
        parser.error("output already exists")
    for name in ("source", "a", "dataset"):
        protected = getattr(args, name, None)
        if protected is not None and args.output.resolve().is_relative_to(
            Path(protected).resolve()
        ):
            parser.error("output must be outside inputs")
    try:
        report = run(args.phase, args)
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
    status = report["status"]
    within = report.get("within_budget", True)
    verdict = report.get("judgement", {}).get("verdict")
    print(
        json.dumps(
            {"status": status, "verdict": verdict, "output": str(args.output)},
            ensure_ascii=False,
        )
    )
    return 0 if status == "COMPLETE" and within else 2


if __name__ == "__main__":
    raise SystemExit(main())
