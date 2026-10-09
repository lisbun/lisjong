"""聴牌PUSH/FOLD（H2）の件数の確認・表のfit・validの報告（lisbun/lisjong#288、学習専用の経路）。

設計は`docs/tenpai-push-fold-design.md`の5節。

- `count`: 消費済みの開発source（#237 S1の`decisions.jsonl`）で、ゲート判断の件数を区分別・
  役の有無別に数える。新しい対局は使わない。S1は打牌判断だけを持つので(A)は数えられない
- `report`: 対比較sourceのtrainから表（`q`・`R_T`・`R_F`・`L`・`U`）を作り、validで
  降りる − 押す の局収支差などを報告する

表の作り方（すべて今の打牌が通った局だけを使う。`L`を除く）:

- `q_ron` / `q_tsumo` / `R_T`: 押す側の局。`R_T`は判断者が和了した局を0として平均する
- `R_F`: 降りる側の局。(B)(C)は`a_fold`が聴牌を崩す判断だけ、(A)は降りる側のすべて
- `L`: 今の打牌でリーチ者に放銃した局の支払点の平均（押す側・降りる側の両方。リーチ者が親か子か）
- `U`: (A)の押す側で和了した局の、実際の和了点 − 判断時点の除外モードの和了点の加重平均

役の判定と和了点の計算にnative拡張が要る。bucketの境界とsupportの下限は引数で与える
（この値はまだ事前登録していない）。
"""

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from random import Random

from lisjong.belief.fixed_point import raw_to_semantic
from lisjong.hand_evaluation import calculate_shanten
from lisjong.hand_evaluation.scoring import evaluate_win
from lisjong.learning.riichi_deal_in_source import read_decisions as read_s1_decisions
from lisjong.learning.riichi_wait_estimator import LogisticWaitModel
from lisjong.learning.riichi_wait_mawashi_policy import load_selected_wait_model
from lisjong.learning.tenpai_push_fold import (
    EvaluateWin,
    GateKind,
    TenpaiGate,
    evaluate_tenpai_gate,
    own_wait_value,
)
from lisjong.learning.tenpai_push_fold_source import (
    PairRecord,
    RoundOutcome,
    read_source,
)
from lisjong.learning.tenpai_push_fold_value import (
    DEALER,
    GROUPS,
    NON_DEALER,
    RIICHI_GROUP,
    BucketSpec,
    PushFoldTables,
    TenpaiSideTable,
    comparison_values,
    gate_group,
)
from lisjong.policies.value_aware_two_step_ukeire import _retained_concealed_dora_count
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.structural_efficiency import post_discard_concealed_hand

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 288
CALIBRATION_UPPER_BOUNDS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0)
"""`p`の校正表の区分の上限（各区分は直前の上限より大きく、この値以下）。"""


def count_gate_decisions(
    decisions: Iterable[tuple[DecisionContext, PolicyDecision]],
    model: LogisticWaitModel,
    *,
    evaluate: EvaluateWin = evaluate_win,
) -> dict[str, object]:
    """（判断、Championの決定）の列から、ゲート判断の件数を区分別・役の有無別に数える。"""
    total = 0
    counts: Counter[tuple[str, str, str]] = Counter()
    for decision, c0_decision in decisions:
        total += 1
        gate = evaluate_tenpai_gate(decision, c0_decision, model, evaluate=evaluate)
        if gate is not None:
            counts[
                gate.kind.value,
                "yaku" if gate.push_wait.has_yaku else "no_yaku",
                "fold_candidate" if gate.has_fold_candidate else "no_fold_candidate",
            ] += 1
    by_kind: dict[str, dict[str, dict[str, int]]] = {}
    for (kind, yaku, fold), count in sorted(counts.items()):
        by_kind.setdefault(kind, {}).setdefault(yaku, {})[fold] = count
    return {
        "decisions": total,
        "gate_decisions": sum(counts.values()),
        "by_kind": by_kind,
    }


def gate_from_record(pair: PairRecord, *, evaluate: EvaluateWin = evaluate_win):
    """記録された候補から、待ちの値を計算し直した`TenpaiGate`を作る（モデルは使わない）。"""
    record = pair.decision
    policy_input = record.policy_input
    riichi = record.kind is GateKind.RIICHI
    fold_keeps_tenpai = (
        calculate_shanten(
            post_discard_concealed_hand(
                policy_input.own_hand.concealed_tiles, record.fold_action.tile
            )
        )
        == 0
    )
    return TenpaiGate(
        kind=record.kind,
        riichi_seat=record.riichi_seat,
        c0_action=record.c0_action,
        push_action=record.push_action,
        fold_action=record.fold_action,
        push_ron_legal_raw=record.push_ron_legal_raw,
        fold_ron_legal_raw=record.fold_ron_legal_raw,
        fold_keeps_tenpai=fold_keeps_tenpai,
        push_wait=own_wait_value(
            policy_input, record.push_action, riichi=riichi, evaluate=evaluate
        ),
        fold_wait=(
            own_wait_value(
                policy_input, record.fold_action, riichi=False, evaluate=evaluate
            )
            if fold_keeps_tenpai and not riichi
            else None
        ),
    )


def _riichi_deal_in(pair: PairRecord, outcome: RoundOutcome) -> bool:
    """今の打牌でリーチ者に放銃したか。"""
    return (
        not outcome.discard_passed and outcome.deal_in_to is pair.decision.riichi_seat
    )


class _Mean:
    __slots__ = ("count", "total")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value


def fit_tables(
    pairs: Sequence[tuple[PairRecord, TenpaiGate]],
    buckets: BucketSpec,
    *,
    minimum_support: int,
) -> tuple[PushFoldTables, dict[str, dict[str, int]]]:
    """trainの対から表を作る。件数が`minimum_support`未満のbucketは表に入れない。

    戻り値の2つ目は、除外したbucketを含む全bucketの件数である。
    """
    cells: defaultdict[str, defaultdict[str, _Mean]] = defaultdict(
        lambda: defaultdict(_Mean)
    )
    for pair, gate in pairs:
        policy_input = pair.decision.policy_input
        wall = policy_input.round.live_wall_tiles_remaining
        group = gate_group(gate.kind)
        riichi_is_dealer = gate.riichi_seat is policy_input.round.dealer_seat
        for outcome in (pair.push, pair.fold):
            if outcome is not None and _riichi_deal_in(pair, outcome):
                cells["loss"][DEALER if riichi_is_dealer else NON_DEALER].add(
                    outcome.deal_in_points
                )

        push, wait = pair.push, gate.push_wait
        if push.discard_passed:
            won = push.win_method is not None
            cells[f"tenpai.{group}.r_t"][
                buckets.tenpai_key(wall, wait.tsumo_count)
            ].add(0 if won else push.round_delta)
            for method, count in (("ron", wait.ron_count), ("tsumo", wait.tsumo_count)):
                if count > 0:
                    cells[f"tenpai.{group}.q_{method}"][
                        buckets.tenpai_key(wall, count)
                    ].add(push.win_method == method)
            if won and group == RIICHI_GROUP:
                base = (
                    wait.ron_points if push.win_method == "ron" else wait.tsumo_points
                )
                cells["uplift"][push.win_method].add(push.win_points - base)

        fold = pair.fold
        if (
            fold is not None
            and fold.discard_passed
            and (group == RIICHI_GROUP or not gate.fold_keeps_tenpai)
        ):
            cells[f"r_f.{group}"][buckets.wall_key(wall)].add(fold.round_delta)

    def table(name: str) -> dict[str, float]:
        return {
            key: cell.total / cell.count
            for key, cell in sorted(cells[name].items())
            if cell.count >= minimum_support
        }

    tables = PushFoldTables(
        buckets=buckets,
        loss=table("loss"),
        tenpai={
            group: TenpaiSideTable(
                q_ron=table(f"tenpai.{group}.q_ron"),
                q_tsumo=table(f"tenpai.{group}.q_tsumo"),
                r_t=table(f"tenpai.{group}.r_t"),
            )
            for group in GROUPS
        },
        r_f={group: table(f"r_f.{group}") for group in GROUPS},
        uplift=table("uplift"),
    )
    support = {
        name: {key: cell.count for key, cell in sorted(table_cells.items())}
        for name, table_cells in sorted(cells.items())
    }
    return tables, support


def _bootstrap_mean(items: Sequence[tuple[int, float]]) -> dict[str, object]:
    """（seed、値）の平均と、半荘（seed）を単位に復元抽出した95%区間。"""
    if not items:
        return {"n": 0, "games": 0}
    by_seed: defaultdict[int, list[float]] = defaultdict(list)
    for seed, value in items:
        by_seed[seed].append(value)
    seeds = sorted(by_seed)
    random = Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [by_seed[random.choice(seeds)] for _ in seeds]
        draws.append(sum(map(sum, sample)) / sum(map(len, sample)))
    draws.sort()
    return {
        "n": len(items),
        "games": len(seeds),
        "mean": sum(value for _, value in items) / len(items),
        "low_2.5": draws[int(0.025 * BOOTSTRAP_RESAMPLES)],
        "high_97.5": draws[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
    }


def _is_dora_or_red(policy_input: PolicyInput, action: DiscardAction) -> bool:
    return (
        _retained_concealed_dora_count(
            (action.tile,), policy_input.round.dora_indicators
        )
        > 0
    )


def valid_report(
    pairs: Sequence[tuple[PairRecord, TenpaiGate]], tables: PushFoldTables
) -> dict[str, object]:
    """validの対について、設計5節の手順3の項目を報告する。

    局収支差は 降りる − 押す で、降りる候補がある判断だけが持つ。区間は半荘単位のbootstrap。
    """
    buckets = tables.buckets
    differences: defaultdict[str, defaultdict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    v_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    calibration: defaultdict[int, list[tuple[float, bool]]] = defaultdict(list)
    deal_in_points: defaultdict[str, list[int]] = defaultdict(list)
    declaration = Counter()

    for pair, gate in pairs:
        record = pair.decision
        policy_input = record.policy_input
        group = gate_group(gate.kind)
        seed = record.key.seed
        sides = [(pair.push, record.push_action, record.push_ron_legal_raw)]
        if pair.fold is not None:
            sides.append((pair.fold, record.fold_action, record.fold_ron_legal_raw))
        for outcome, action, raw in sides:
            p = raw_to_semantic(raw)
            dealt_in = _riichi_deal_in(pair, outcome)
            index = next(
                index
                for index, bound in enumerate(CALIBRATION_UPPER_BOUNDS)
                if p <= bound
            )
            calibration[index].append((p, dealt_in))
            if dealt_in:
                dora = _is_dora_or_red(policy_input, action)
                deal_in_points["dora_or_red" if dora else "other"].append(
                    outcome.deal_in_points
                )
        if gate.kind is GateKind.RIICHI:
            matched = pair.push.declaration_discard == record.push_action
            declaration["match" if matched else "mismatch"] += 1

        if pair.fold is None:
            v_counts[group]["no_fold_candidate"] += 1
            continue
        item = (seed, pair.fold.round_delta - pair.push.round_delta)
        wall = policy_input.round.live_wall_tiles_remaining
        target = differences[group]
        target["all"].append(item)
        target[f"bucket.{buckets.tenpai_key(wall, gate.push_wait.tsumo_count)}"].append(
            item
        )
        target["yaku" if gate.push_wait.has_yaku else "no_yaku"].append(item)
        if _is_dora_or_red(policy_input, record.push_action):
            target["push_dora_or_red"].append(item)
        values = comparison_values(policy_input, gate, tables)
        if values is None:
            v_counts[group]["no_table"] += 1
        elif values[1] > values[0]:
            v_counts[group]["v_fold"] += 1
            target["v_fold"].append(item)
        else:
            v_counts[group]["v_push"] += 1
            target["v_push"].append(item)

    return {
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "round_delta_difference": {
            group: {
                name: _bootstrap_mean(items)
                for name, items in sorted(differences[group].items())
            }
            for group in GROUPS
        },
        "v_decisions": {
            group: dict(sorted(v_counts[group].items())) for group in GROUPS
        },
        "calibration": [
            {
                "upper_bound": CALIBRATION_UPPER_BOUNDS[index],
                "n": len(rows),
                "mean_predicted": sum(p for p, _ in rows) / len(rows),
                "deal_in_rate": sum(dealt_in for _, dealt_in in rows) / len(rows),
            }
            for index, rows in sorted(calibration.items())
        ],
        "deal_in_points": {
            name: {"n": len(points), "mean": sum(points) / len(points)}
            for name, points in sorted(deal_in_points.items())
        },
        "riichi_declaration_prediction": dict(sorted(declaration.items())),
    }


def _bounds(text: str) -> tuple[int, int]:
    try:
        first, second = (int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected two ints as A,B") from None
    return first, second


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=__name__)
    commands = parser.add_subparsers(dest="command", required=True)
    count_parser = commands.add_parser("count")
    count_parser.add_argument("--s1-source", type=Path, required=True)
    count_parser.add_argument("--selection", type=Path, required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--source", type=Path, required=True)
    report_parser.add_argument("--wall-upper-bounds", type=_bounds, required=True)
    report_parser.add_argument("--count-upper-bounds", type=_bounds, required=True)
    report_parser.add_argument("--minimum-support", type=int, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "count":
        manifest, decisions = read_s1_decisions(arguments.s1_source)
        result = count_gate_decisions(
            (
                (
                    DecisionContext(
                        input=decision.policy_input,
                        legal_actions=decision.legal_actions,
                    ),
                    PolicyDecision(action=decision.selected_action),
                )
                for decision in decisions
            ),
            load_selected_wait_model(arguments.selection),
        )
        result["games"] = sum(len(seeds) for seeds in manifest.splits.values())
    else:
        manifest, records = read_source(arguments.source)
        by_split: dict[str, list] = {"train": [], "valid": []}
        for pair in records:
            by_split[manifest.split_of(pair.decision.key.seed)].append(
                (pair, gate_from_record(pair))
            )
        buckets = BucketSpec(
            wall_upper_bounds=arguments.wall_upper_bounds,
            count_upper_bounds=arguments.count_upper_bounds,
        )
        tables, support = fit_tables(
            by_split["train"], buckets, minimum_support=arguments.minimum_support
        )
        result = {
            "minimum_support": arguments.minimum_support,
            "tables": tables.to_value(),
            "train_support": support,
            "valid": valid_report(by_split["valid"], tables),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
