"""L2a OFFLINE-EVALの凍結した判定（Issue #317）。

設計`docs/learning-l2a.md`の3.2節（support）、3.4節（判定保留とSTOP）、5節（指標）、
6節（成立条件）を、結果を見る前に`L2A_OFFLINE_GATE`として固定する。gateのdigest
（`l2a_offline_gate_fingerprint()`）を評価reportへ必ず含める。閾値・support下限・
bootstrap設定・層を変える場合は既存identityを書き換えず新しいidentityにする。

このmoduleは集計済みの半荘別の値（`L2aOfflineTally`）だけから判定する。source・
artifact・Policyを読まず、ML runtimeも使わない。集計は
`lisjong.learning.l2a_offline_eval`が行う。

```text
半荘別・層別の数（quantity）
    -> 点推定  = 全半荘の数の和から計算
    -> 95%区間 = 半荘をresampling単位とするbootstrap（10,000回、seed固定）の
                 percentile区間。各resampleで同じ指標関数を数の和へ適用する
    -> 5.1の正しさ・教師規則の例外 -> STOP / INVALID
    -> 3.2のsupport下限            -> 行動別の判定保留
    -> 6節の条件                    -> ESTABLISHED / PARTIAL / NOT ESTABLISHED
```

率は行を独立標本とせず、resampleした半荘の分子の和 / 分母の和として求める。
M1−M0の差は同じresampleの半荘で両者を数える（半荘単位の対）。分母が0になった
resampleは区間から除き、その回数を`undefined_resamples`として報告する。
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from operator import mul

from lisjong.learning._canonical import value_digest
from lisjong.learning.errors import LearningError

L2A_OFFLINE_GATE_IDENTITY = "lisjong-l2a-offline-eval-gate-v1"
L2A_OFFLINE_REPORT_SCHEMA = "lisjong-l2a-offline-eval-report-v1"
L2A_OFFLINE_METRICS_IDENTITY = "lisjong-l2a-offline-eval-metrics-v1"
L2A_OFFLINE_SPLIT = "OFFLINE-EVAL"

ESTABLISHED = "ESTABLISHED"
PARTIAL = "PARTIAL"
NOT_ESTABLISHED = "NOT ESTABLISHED"
HOLD = "HOLD — INSUFFICIENT SUPPORT"
"""判定保留（support不足）。推測でPASSにしない（設計3.4節）。"""
INVALID = "STOP / INVALID"
EVALUATED = "EVALUATED"

DISCARD_LEARNERS = ("m1", "m0", "b_ts")
"""打牌の比較対象。M1が提案、M0とB-TS（TwoStepの選択）がbaseline。"""
CALL_LEARNERS = ("r1", "b_pass")
"""CALLの比較対象。R1が提案、B-PASS（常にPass）がbaseline。"""
CALL_TYPES = {"chi": "call.chi", "pon": "call.pon", "call": "call.any"}
"""CALLの指標名と数の接頭辞。`call`はChi / Pon合算（診断）。"""

STRATA = (
    "all",
    "round:east",
    "round:south_1_3",
    "round:all_last",
    "rank:1",
    "rank:2",
    "rank:3",
    "rank:4",
    "rank_tied:no",
    "rank_tied:yes",
    "opponent_riichi:0",
    "opponent_riichi:1",
    "opponent_riichi:2+",
    "shanten:0",
    "shanten:1",
    "shanten:2",
    "shanten:3+",
    "hand:closed",
    "hand:open",
    "riichi_declaration:no",
    "riichi_declaration:yes",
    "efficiency:two_step_same",
    "efficiency:c0_specific",
)
"""設計2.5節の層。`efficiency:*`はDISCARDの行だけが持つ（C0固有層）。"""


def _quantity_names() -> tuple[str, ...]:
    names = ["discard.rows", "call.rows"]
    for learner in DISCARD_LEARNERS:
        names += [
            f"discard.{learner}.{name}"
            for name in ("match", "tsumogiri_match", "shanten_match", "ukeire_match")
        ]
    for prefix in CALL_TYPES.values():
        names += [f"{prefix}.legal", f"{prefix}.teacher"]
        for learner in CALL_LEARNERS:
            names += [
                f"{prefix}.{learner}.{name}"
                for name in ("true_positive", "true_negative", "action_match")
            ]
    names += [f"call.{learner}.match" for learner in CALL_LEARNERS]
    return tuple(names)


QUANTITIES = _quantity_names()
"""半荘・層ごとに数える量。指標はこれらの和の比である。

```text
discard.rows                       単一候補行を除くDISCARD行
discard.<L>.match                  Lの打牌が教師と同じ牌（牌種と赤の有無）
discard.<L>.tsumogiri_match        同じ牌かつツモ切り / 手出しも同じ（診断）
discard.<L>.shanten_match          打牌後の向聴が教師と同じ（診断）
discard.<L>.ukeire_match           向聴が同じで受入も教師と同じ（診断）
call.rows                          単一候補行を除くCALL行
call.<T>.legal / teacher           種別Tが合法な行 / うち教師がTを選んだ行
call.<T>.<L>.true_positive         教師がTを選び、LもTを選んだ
call.<T>.<L>.true_negative         Tが合法で教師がTを選ばず、LもTを選ばなかった
call.<T>.<L>.action_match          教師がTを選び、Lが同じ行動（種別・消費牌）を選んだ
call.<L>.match                     Lの反応がPassを含め教師と同じ（診断）
```

`<T>`はchi / pon / any（Chi・Pon合算）である。
"""

_QUANTITY_SET = frozenset(QUANTITIES)

L2A_OFFLINE_GATE: dict[str, object] = {
    "identity": L2A_OFFLINE_GATE_IDENTITY,
    "design": "docs/learning-l2a.md 3.2 / 3.4 / 5 / 6",
    "split": L2A_OFFLINE_SPLIT,
    "metrics": L2A_OFFLINE_METRICS_IDENTITY,
    "strata": list(STRATA),
    "correctness": {
        "guard": 1.0,
        "legality": 1.0,
        "teacher_rule_exceptions": 0,
        "train_serve": 1.0,
    },
    "bootstrap": {
        "confidence": 0.95,
        "interval": "percentile",
        "quantile": "linear-interpolation-at-(n-1)p",
        "resamples": 10000,
        "seed": 317,
        "undefined_resample": "excluded-and-counted",
        "unit": "hanchan",
    },
    "minimum_support": {
        "discard": {
            "hanchan": 100,
            "quantity": "discard.rows",
            "rows": 20000,
            "stratum": "all",
        },
        "discard_c0_specific": {
            "hanchan": 80,
            "quantity": "discard.rows",
            "rows": 2000,
            "stratum": "efficiency:c0_specific",
        },
        "call": {
            "hanchan": 80,
            "quantity": "call.rows",
            "rows": 2000,
            "stratum": "all",
        },
        "chi_legal": {
            "hanchan": 80,
            "quantity": "call.chi.legal",
            "rows": 1000,
            "stratum": "all",
        },
        "chi_selected": {
            "hanchan": 60,
            "quantity": "call.chi.teacher",
            "rows": 300,
            "stratum": "all",
        },
        "pon_legal": {
            "hanchan": 80,
            "quantity": "call.pon.legal",
            "rows": 1000,
            "stratum": "all",
        },
        "pon_selected": {
            "hanchan": 60,
            "quantity": "call.pon.teacher",
            "rows": 300,
            "stratum": "all",
        },
        "discard_all_last": {
            "hanchan": 30,
            "quantity": "discard.rows",
            "rows": 1000,
            "stratum": "round:all_last",
        },
    },
    "report_only_support": ["discard_all_last"],
    "guard_support": {"riichi": 30, "win": 30},
    "verdicts": {
        "discard": {
            "support": ["discard", "discard_c0_specific"],
            "primary": {
                "comparison": ">=",
                "estimate": "point",
                "metric": "m1.discard_match",
                "stratum": "all",
                "threshold": 0.85,
            },
            "secondary": [
                {
                    "comparison": ">=",
                    "estimate": "lower",
                    "metric": "m1.discard_match",
                    "stratum": "efficiency:c0_specific",
                    "threshold": 0.40,
                },
                {
                    "comparison": ">",
                    "estimate": "lower",
                    "metric": "m1_minus_m0.discard_match",
                    "stratum": "all",
                    "threshold": 0.0,
                },
            ],
        },
        "chi": {
            "support": ["call", "chi_legal", "chi_selected"],
            "primary": {
                "comparison": ">=",
                "estimate": "point",
                "metric": "r1.chi_balanced_accuracy",
                "stratum": "all",
                "threshold": 0.85,
            },
            "secondary": [
                {
                    "comparison": ">=",
                    "estimate": "point",
                    "metric": "r1.chi_action_match",
                    "stratum": "all",
                    "threshold": 0.85,
                }
            ],
        },
        "pon": {
            "support": ["call", "pon_legal", "pon_selected"],
            "primary": {
                "comparison": ">=",
                "estimate": "point",
                "metric": "r1.pon_balanced_accuracy",
                "stratum": "all",
                "threshold": 0.85,
            },
            "secondary": [
                {
                    "comparison": ">=",
                    "estimate": "point",
                    "metric": "r1.pon_action_match",
                    "stratum": "all",
                    "threshold": 0.85,
                }
            ],
        },
    },
}
"""OFFLINE-EVALを見る前に固定したgate（設計3.2・6節）。変更するなら新identityにする。

`estimate`の`point`は点推定、`lower`は95%区間の下限で判定する。WIN / RIICHIの
`guard_support`を下回るときは、guardの正しさを確かめられないため全行動を判定保留にする。
"""

NON_CLAIMS = (
    {
        "action": "和了 / リーチ",
        "result": "教師規則guard（C0と同じ決定的規則）。学習成立の対象外",
    },
    {
        "action": "槓 / 九種九牌",
        "result": "教師がsupportを持たない（選ばない）ため対象外。guardは選ばない",
    },
    {"action": "打牌 / CALL", "result": "6節の判定"},
)
"""結果の記録に必ず付ける設計2.2節の表。"""


def l2a_offline_gate_fingerprint() -> str:
    return value_digest(L2A_OFFLINE_GATE)


@dataclass(frozen=True, slots=True)
class L2aOfflineTally:
    """OFFLINE-EVALの集計。判定はこの値だけから決まる。

    `games`は半荘ごとに`{層: {quantity: 数}}`を持つ（行の無い半荘も含める）。
    `correctness`は設計5.1節の正しさと教師規則の例外の数、`diagnostics`は判定に
    使わない5.3節の値である。
    """

    games: tuple[Mapping[str, Mapping[str, int]], ...]
    correctness: Mapping[str, object]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        for index, game in enumerate(self.games):
            for stratum, counts in game.items():
                if stratum not in STRATA:
                    raise LearningError(
                        f"game {index} has an unknown stratum {stratum}"
                    )
                unknown = set(counts) - _QUANTITY_SET
                if unknown:
                    raise LearningError(
                        f"game {index} has unknown quantities {sorted(unknown)}"
                    )
                if any(
                    type(value) is not int or value < 0 for value in counts.values()
                ):
                    raise LearningError(f"game {index} has an invalid count")


# --- 指標 -----------------------------------------------------------------------


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _balanced(
    true_positive: float, positive: float, true_negative: float, negative: float
) -> float | None:
    if not positive or not negative:
        return None
    return 0.5 * (true_positive / positive + true_negative / negative)


def metric_values(sums: Mapping[str, float]) -> dict[str, float | None]:
    """数の和から全指標を計算する。点推定と各bootstrap resampleで同じ関数を使う。"""

    def get(name: str) -> float:
        return sums.get(name, 0)

    values: dict[str, float | None] = {}
    rows = get("discard.rows")
    for learner in DISCARD_LEARNERS:
        prefix = f"discard.{learner}"
        shanten = get(f"{prefix}.shanten_match")
        values[f"{learner}.discard_match"] = _ratio(get(f"{prefix}.match"), rows)
        values[f"{learner}.tsumogiri_match"] = _ratio(
            get(f"{prefix}.tsumogiri_match"), rows
        )
        values[f"{learner}.shanten_match"] = _ratio(shanten, rows)
        values[f"{learner}.conditional_ukeire_match"] = _ratio(
            get(f"{prefix}.ukeire_match"), shanten
        )
    values["m1_minus_m0.discard_match"] = _ratio(
        get("discard.m1.match") - get("discard.m0.match"), rows
    )
    for learner in CALL_LEARNERS:
        values[f"{learner}.reaction_match"] = _ratio(
            get(f"call.{learner}.match"), get("call.rows")
        )
        for name, prefix in CALL_TYPES.items():
            positive = get(f"{prefix}.teacher")
            negative = get(f"{prefix}.legal") - positive
            values[f"{learner}.{name}_balanced_accuracy"] = _balanced(
                get(f"{prefix}.{learner}.true_positive"),
                positive,
                get(f"{prefix}.{learner}.true_negative"),
                negative,
            )
            values[f"{learner}.{name}_action_match"] = _ratio(
                get(f"{prefix}.{learner}.action_match"), positive
            )
    return values


# --- bootstrap ------------------------------------------------------------------


@lru_cache(maxsize=4)
def _resample_multiplicities(
    game_count: int, resamples: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """各resampleで各半荘が何回選ばれたか。全指標・全層で同じresampleを使う。"""
    rng = random.Random(seed)
    population = range(game_count)
    result = []
    for _ in range(resamples):
        counts = [0] * game_count
        for index in rng.choices(population, k=game_count):
            counts[index] += 1
        result.append(tuple(counts))
    return tuple(result)


def _quantile(ordered: Sequence[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _stratum_metrics(
    games: Sequence[Mapping[str, Mapping[str, int]]],
    stratum: str,
    multiplicities: Sequence[Sequence[int]],
    confidence: float,
) -> dict[str, dict[str, object]] | None:
    vectors: dict[str, list[int]] = {}
    for name in QUANTITIES:
        vector = [game.get(stratum, {}).get(name, 0) for game in games]
        if any(vector):
            vectors[name] = vector
    if not vectors:
        return None
    point = metric_values({name: sum(vector) for name, vector in vectors.items()})
    resampled = {
        name: [sum(map(mul, counts, vector)) for counts in multiplicities]
        for name, vector in vectors.items()
    }
    distributions: dict[str, list[float]] = {name: [] for name in point}
    for index in range(len(multiplicities)):
        values = metric_values({name: sums[index] for name, sums in resampled.items()})
        for name, value in values.items():
            if value is not None:
                distributions[name].append(value)
    tail = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, object]] = {}
    for name, estimate in point.items():
        if estimate is None:
            continue
        ordered = sorted(distributions[name])
        result[name] = {
            "lower": _quantile(ordered, tail) if ordered else None,
            "upper": _quantile(ordered, 1.0 - tail) if ordered else None,
            "point": estimate,
            "undefined_resamples": len(multiplicities) - len(ordered),
        }
    return result


def _support(
    games: Sequence[Mapping[str, Mapping[str, int]]],
) -> dict[str, dict[str, dict[str, int]]]:
    """層ごとのDISCARD / CALLの行数と半荘support（行を1件以上含む半荘の数）。"""
    result: dict[str, dict[str, dict[str, int]]] = {}
    for stratum in STRATA:
        item = {}
        for name in ("discard.rows", "call.rows"):
            values = [game.get(stratum, {}).get(name, 0) for game in games]
            if any(values):
                item[name.split(".")[0]] = {
                    "hanchan": sum(1 for value in values if value),
                    "rows": sum(values),
                }
        if item:
            result[stratum] = item
    return result


def _minimum_support(
    games: Sequence[Mapping[str, Mapping[str, int]]],
) -> dict[str, dict[str, object]]:
    result = {}
    for name, spec in L2A_OFFLINE_GATE["minimum_support"].items():
        values = [
            game.get(spec["stratum"], {}).get(spec["quantity"], 0) for game in games
        ]
        rows = sum(values)
        hanchan = sum(1 for value in values if value)
        result[name] = {
            "hanchan": hanchan,
            "required_hanchan": spec["hanchan"],
            "required_rows": spec["rows"],
            "rows": rows,
            "satisfied": rows >= spec["rows"] and hanchan >= spec["hanchan"],
        }
    return result


# --- 判定 -----------------------------------------------------------------------


def _rate_check(name: str, counts: Mapping[str, int], key: str) -> list[str]:
    rows = counts["rows"]
    if rows and counts[key] < rows:
        return [f"{name} is {counts[key]}/{rows}, below 1.000"]
    return []


def correctness_stop_reasons(correctness: Mapping[str, object]) -> list[str]:
    """設計5.1節の正しさと2.3節の教師規則からSTOP / INVALIDの理由を返す。"""
    reasons = []
    decisions = correctness["decisions"]
    if decisions == 0:
        reasons.append(f"{L2A_OFFLINE_SPLIT} has no decisions")
    elif correctness["legal"] < decisions:
        reasons.append(f"legality is {correctness['legal']}/{decisions}, below 1.000")
    for kind, counts in correctness["guard"].items():
        reasons += _rate_check(f"guard agreement ({kind})", counts, "agree")
    for name, counts in correctness["train_serve"].items():
        reasons += _rate_check(f"train-serve agreement ({name})", counts, "match")
    if correctness["teacher_rule_exceptions"]:
        reasons.append(
            f"{correctness['teacher_rule_exceptions']} teacher-rule exception(s)"
        )
    return reasons


def _condition(
    spec: Mapping[str, object], metrics: Mapping[str, Mapping[str, Mapping]]
) -> dict[str, object]:
    metric = metrics.get(spec["stratum"], {}).get(spec["metric"])
    value = None if metric is None else metric[spec["estimate"]]
    if value is None:
        met = False
    elif spec["comparison"] == ">=":
        met = value >= spec["threshold"]
    else:
        met = value > spec["threshold"]
    return {**spec, "met": met, "value": value}


def _verdict(
    name: str,
    spec: Mapping[str, object],
    metrics: Mapping[str, Mapping[str, Mapping]],
    support: Mapping[str, Mapping[str, object]],
    guard_hold: Sequence[str],
) -> dict[str, object]:
    held = [item for item in spec["support"] if not support[item]["satisfied"]]
    if held or guard_hold:
        return {
            "held_support": held + list(guard_hold),
            "verdict": HOLD,
        }
    primary = _condition(spec["primary"], metrics)
    secondary = [_condition(item, metrics) for item in spec["secondary"]]
    if not primary["met"]:
        verdict = NOT_ESTABLISHED
    elif all(item["met"] for item in secondary):
        verdict = ESTABLISHED
    else:
        verdict = PARTIAL
    result = {"primary": primary, "secondary": secondary, "verdict": verdict}
    if (
        name == "discard"
        and verdict == PARTIAL
        and secondary[0]["met"]
        and not secondary[1]["met"]
    ):
        result["note"] = "C0の打牌は学べたが、encoding追加の効果は示せない"
    return result


def _conclusion(verdicts: Mapping[str, Mapping[str, object]]) -> dict[str, str]:
    """設計6節の記録の書き方。成立を呼ぶのは全条件を満たしたときだけ。"""
    labels = {"discard": "打牌", "chi": "Chi", "pon": "Pon"}
    states = {name: item["verdict"] for name, item in verdicts.items()}
    if states["chi"] == ESTABLISHED and states["pon"] == ESTABLISHED:
        call = "鳴き判断の蒸留が成立"
    else:
        call = (
            f"Chiは{states['chi']}、Ponは{states['pon']}（CALL全体の成立とは呼ばない）"
        )
    if all(state == ESTABLISHED for state in states.values()):
        overall = "L2a: C0の打牌・鳴き判断の蒸留が成立"
    else:
        overall = "L2a: " + "、".join(
            f"{labels[name]}は{state}" for name, state in states.items()
        )
    return {"call": call, "overall": overall}


def judge_l2a_offline_eval(
    tally: L2aOfflineTally | None, stop_reasons: Sequence[str] = ()
) -> dict[str, object]:
    """凍結したgateでOFFLINE-EVALを判定し、reportを返す。

    `stop_reasons`は集計の前に分かったSTOP / INVALIDの理由（identityの不整合、
    OFFLINE-EVALの2回目の使用等）である。`tally`が`None`なら指標を計算しない。
    """
    reasons = list(stop_reasons)
    if tally is None and not reasons:
        raise LearningError("judging without a tally requires a STOP reason")
    report: dict[str, object] = {
        "gate": {**L2A_OFFLINE_GATE, "fingerprint": l2a_offline_gate_fingerprint()},
        "non_claims": [dict(item) for item in NON_CLAIMS],
        "schema": L2A_OFFLINE_REPORT_SCHEMA,
    }
    if tally is not None:
        reasons += correctness_stop_reasons(tally.correctness)
        bootstrap = L2A_OFFLINE_GATE["bootstrap"]
        multiplicities = _resample_multiplicities(
            len(tally.games), bootstrap["resamples"], bootstrap["seed"]
        )
        metrics = {}
        for stratum in STRATA:
            values = _stratum_metrics(
                tally.games, stratum, multiplicities, bootstrap["confidence"]
            )
            if values is not None:
                metrics[stratum] = values
        support = _minimum_support(tally.games)
        guard_rows = tally.correctness["guard"]
        guard_hold = [
            f"{kind}_guard"
            for kind, required in L2A_OFFLINE_GATE["guard_support"].items()
            if guard_rows.get(kind, {"rows": 0})["rows"] < required
        ]
        report.update(
            {
                "correctness": dict(tally.correctness),
                "diagnostics": dict(tally.diagnostics),
                "hanchan": len(tally.games),
                "metrics": metrics,
                "strata_support": _support(tally.games),
                "support": support,
            }
        )
    if reasons:
        report["outcome"] = INVALID
        report["stop_reasons"] = reasons
        report["verdicts"] = {
            name: {"verdict": INVALID} for name in L2A_OFFLINE_GATE["verdicts"]
        }
        report["conclusion"] = {"overall": f"L2a: {INVALID}（判定しない）"}
        return report
    verdicts = {
        name: _verdict(name, spec, metrics, support, guard_hold)
        for name, spec in L2A_OFFLINE_GATE["verdicts"].items()
    }
    report.update(
        {
            "conclusion": _conclusion(verdicts),
            "outcome": EVALUATED,
            "stop_reasons": [],
            "verdicts": verdicts,
        }
    )
    return report


__all__ = [
    "CALL_LEARNERS",
    "CALL_TYPES",
    "DISCARD_LEARNERS",
    "ESTABLISHED",
    "EVALUATED",
    "HOLD",
    "INVALID",
    "L2A_OFFLINE_GATE",
    "L2A_OFFLINE_GATE_IDENTITY",
    "L2A_OFFLINE_METRICS_IDENTITY",
    "L2A_OFFLINE_REPORT_SCHEMA",
    "L2A_OFFLINE_SPLIT",
    "NON_CLAIMS",
    "NOT_ESTABLISHED",
    "PARTIAL",
    "QUANTITIES",
    "STRATA",
    "L2aOfflineTally",
    "correctness_stop_reasons",
    "judge_l2a_offline_eval",
    "l2a_offline_gate_fingerprint",
    "metric_values",
]
