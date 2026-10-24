"""L2a OFFLINE-EVALの評価tool（Issue #317）。

設計`docs/learning-l2a.md`の5節を、凍結したartifactとOFFLINE-EVALのsourceへ1回だけ
適用する。判定は`lisjong.learning.l2a_offline_gate`の凍結したgateで行う。

```text
identityの照合（source・artifact M1 / M0 / R1・dataset M1 / M0 / R1）
    -> 不一致ならSTOP / INVALID（ledgerへ記録しない。OFFLINE-EVALを読まない）
ledgerの照合と記録（同じartifactの評価が既にあればSTOP / INVALID）
    -> OFFLINE-EVALの全行を1行ずつ
         提案（M1 + R1の合成Policy）       : 合法性・guardの一致・指標
         servingの経路で再encodeした値     : datasetの値とbyte単位で照合（train-serve一致）
         M0（同じguard + M0打牌scorer）     : DISCARD行の指標
         B-TS（TwoStepUkeirePolicy）        : DISCARD行の指標とC0固有層の判定
         B-PASS（常にPass）                 : CALL行の指標
    -> 半荘別・層別の数 -> 凍結したgateで判定
```

ledgerは評価の記録を置くdirectoryで、1つのL2a実験を通じて同じdirectoryを使う。
OFFLINE-EVALを読む前に1評価につき1つのentryを新規作成する。同じsource identityの
OFFLINE-EVALを使ったentry、または同じartifact identity（M1 / M0 / R1のどれか）を
含むentryが既にあれば、その評価はOFFLINE-EVALの2回目の使用であり、指標を計算せず
STOP / INVALIDにする（設計3.4・4.5節）。artifactをすべて再学習しても同じsourceの
OFFLINE-EVALは再利用できない。途中で失敗した評価のentryも残るため、再実行も2回目として
検出する。

B-TSとC0固有層の判定に`TwoStepUkeirePolicy`を同じPolicyInputとlegal actionsで
再実行する。これはofflineの比較であり、L2aのserving経路では使わない。
"""

import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

from lisjong.hand_evaluation import _shanten_backend
from lisjong.hand_evaluation.shanten import calculate_shanten
from lisjong.learning._canonical import (
    canonical_json_text,
    expect_object,
    parse_json_text,
    seal,
    unseal,
)
from lisjong.learning._publication import write_new_text
from lisjong.learning.candidate_dataset import _float32_bytes
from lisjong.learning.candidate_encoding import build_scorer_candidates
from lisjong.learning.errors import DatasetError, LearnedPolicyError, LearningError
from lisjong.learning.features import build_player_safe_feature
from lisjong.learning.l2a_artifact import load_l2a_artifact
from lisjong.learning.l2a_dataset import (
    iter_l2a_dataset_payload,
    read_l2a_dataset_manifest,
    resolve_teacher_index,
)
from lisjong.learning.l2a_decision import (
    L2aDecisionKind,
    classify_l2a_decision,
    guard_action,
    is_single_candidate,
    scorer_candidates,
    teacher_rule_exception,
)
from lisjong.learning.l2a_offline_gate import (
    L2A_OFFLINE_SPLIT,
    L2aOfflineTally,
    judge_l2a_offline_eval,
    l2a_offline_gate_fingerprint,
)
from lisjong.learning.l2a_support import is_all_last_equivalent
from lisjong.learning.source_record import (
    StreamingSourceRecord,
    open_source_record,
    read_source_game,
)
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import (
    ChiAction,
    DecisionContext,
    DiscardAction,
    InternalAction,
    PassAction,
    PolicyInput,
    PonAction,
    RiichiState,
)
from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.wind import Wind

L2A_OFFLINE_LEDGER_SCHEMA = "lisjong-l2a-offline-eval-ledger-entry-v1"
LEDGER_ENTRY_PREFIX = "offline-eval-"

ARTIFACT_NAMES = ("m1", "m0", "r1")
"""評価に使うartifact / datasetの名前。名前はencodingの名前と一致しなければならない。"""

_LEDGER_FIELDS = frozenset(
    {
        "artifacts",
        "datasets",
        "gate_fingerprint",
        "schema",
        "sequence",
        "source_identity",
        "split",
    }
)
_CALL_TYPES = (
    ("call.chi", ChiAction),
    ("call.pon", PonAction),
    ("call.any", (ChiAction, PonAction)),
)
_RANK_BUCKETS = ("2", "3", "4", "5+")

type ReferencePolicy = Callable[[DecisionContext], InternalAction]


# --- 層（設計2.5節） ---------------------------------------------------------------


def decision_strata(
    policy_input: PolicyInput, *, c0_specific: bool | None
) -> tuple[str, ...]:
    """1判断が属する層。公開情報と自席の手牌だけで決める。

    向聴は打牌（DISCARD）なら打牌前、反応（CALL）なら鳴く前の純手牌の向聴で、
    和了形（-1）は0に含める。`c0_specific`はDISCARDだけが持つ（他は`None`）。
    """
    seat = int(policy_input.self_seat)
    players = policy_input.players
    own = players[seat]
    others = [player for index, player in enumerate(players) if index != seat]
    if is_all_last_equivalent(policy_input):
        position = "all_last"
    elif policy_input.round.round_wind is Wind.EAST:
        position = "east"
    else:
        position = "south_1_3"
    rank = 1 + sum(player.score > own.score for player in others)
    tied = any(player.score == own.score for player in others)
    riichi = sum(player.riichi is not RiichiState.NONE for player in others)
    shanten = max(calculate_shanten(policy_input.own_hand.concealed_tiles), 0)
    is_open = any(meld.kind is not MeldKind.ANKAN for meld in own.melds)
    strata = [
        "all",
        f"round:{position}",
        f"rank:{rank}",
        f"rank_tied:{'yes' if tied else 'no'}",
        f"opponent_riichi:{min(riichi, 2)}{'+' if riichi >= 2 else ''}",
        f"shanten:{shanten}" if shanten < 3 else "shanten:3+",
        f"hand:{'open' if is_open else 'closed'}",
        "riichi_declaration:" + ("yes" if own.riichi is RiichiState.DECLARED else "no"),
    ]
    if c0_specific is not None:
        strata.append(
            "efficiency:c0_specific" if c0_specific else "efficiency:two_step_same"
        )
    return tuple(strata)


# --- 集計 -----------------------------------------------------------------------


def _offline_games(source) -> Iterator:
    if isinstance(source, StreamingSourceRecord):
        for summary in source.summaries:
            if summary["split"] == L2A_OFFLINE_SPLIT:
                yield read_source_game(source.root, summary, schema=source.schema)
        return
    for game in source.games:
        if game.split == L2A_OFFLINE_SPLIT:
            yield game


def _runtime_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "max_ms": ordered[-1],
        "median_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


class _ScoreDiagnostics:
    """scorerのcross entropyと、top-1でない行の教師候補の順位（設計5.3節）。"""

    def __init__(self) -> None:
        self.rows = 0
        self.cross_entropy = 0.0
        self.ranks: Counter[str] = Counter()

    def add(self, candidates, scores, teacher: InternalAction) -> None:
        index = candidates.index(teacher)
        peak = max(scores)
        log_partition = peak + math.log(sum(math.exp(s - peak) for s in scores))
        self.rows += 1
        self.cross_entropy += log_partition - scores[index]
        target = scores[index]
        # select_bestと同じ順：score降順、同点はcanonical候補順。
        rank = 1 + sum(
            score > target or (score == target and position < index)
            for position, score in enumerate(scores)
        )
        if rank > 1:
            self.ranks[str(rank) if rank < 5 else "5+"] += 1

    def to_value(self) -> dict[str, object]:
        return {
            "mean_cross_entropy": self.cross_entropy / self.rows if self.rows else None,
            "rows": self.rows,
            "teacher_rank_when_missed": {
                bucket: self.ranks[bucket] for bucket in _RANK_BUCKETS
            },
        }


class _Collector:
    def __init__(
        self,
        proposed,
        baseline_m0,
        datasets: Mapping[str, str | Path],
        reference_policy: ReferencePolicy,
    ) -> None:
        self.proposed = proposed
        self.baseline_m0 = baseline_m0
        self.reference_policy = reference_policy
        self.encodings = {
            "m1": proposed.runtime.scorer_for(L2aDecisionKind.DISCARD).encoding,
            "m0": baseline_m0.runtime.scorer_for(L2aDecisionKind.DISCARD).encoding,
            "r1": proposed.runtime.scorer_for(L2aDecisionKind.CALL).encoding,
        }
        for name, encoding in self.encodings.items():
            manifest = read_l2a_dataset_manifest(datasets[name])
            if manifest["encoding"] != encoding.block():
                raise LearnedPolicyError(
                    f"{name} dataset encoding does not match the serving encoding"
                )
        self.payloads = {
            name: iter_l2a_dataset_payload(datasets[name], splits=(L2A_OFFLINE_SPLIT,))
            for name in self.encodings
        }
        self.games: list[dict[str, dict[str, int]]] = []
        self.decisions = 0
        self.legal = 0
        self.guard = {kind.value: {"agree": 0, "rows": 0} for kind in L2aDecisionKind}
        self.single = {"guard_agree": 0, "rows": 0}
        self.train_serve = {name: {"match": 0, "rows": 0} for name in self.encodings}
        self.exceptions = 0
        self.scores = {name: _ScoreDiagnostics() for name in self.encodings}
        self.decide_ms: dict[str, list[float]] = defaultdict(list)
        self.stage_ms: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def run(self, source) -> L2aOfflineTally:
        for game in _offline_games(source):
            counts: dict[str, Counter[str]] = defaultdict(Counter)
            for row in game.decisions:
                self._row(row, counts)
            self.games.append({key: dict(value) for key, value in counts.items()})
        for name, payloads in self.payloads.items():
            if next(payloads, None) is not None:
                raise LearnedPolicyError(
                    f"{name} dataset has {L2A_OFFLINE_SPLIT} rows that are not "
                    "scorer decisions of the source"
                )
        return L2aOfflineTally(
            games=tuple(self.games),
            correctness={
                "decisions": self.decisions,
                "guard": self.guard,
                "legal": self.legal,
                "single_candidate": self.single,
                "teacher_rule_exceptions": self.exceptions,
                "train_serve": self.train_serve,
            },
            diagnostics={
                "runtime": {
                    kind: {
                        "decide": _runtime_summary(self.decide_ms[kind]),
                        **{
                            stage: _runtime_summary(values)
                            for stage, values in sorted(self.stage_ms[kind].items())
                        },
                    }
                    for kind in sorted(self.decide_ms)
                },
                "scores": {
                    name: value.to_value() for name, value in self.scores.items()
                },
                "shanten_backend": _shanten_backend.BACKEND_NAME,
            },
        )

    def _row(self, row, counts: dict[str, Counter[str]]) -> None:
        context = DecisionContext(
            input=row.policy_input, legal_actions=row.legal_actions
        )
        kind = classify_l2a_decision(context)
        started = time.perf_counter()
        result = self.proposed.decide(context)
        self.decide_ms[kind.value].append((time.perf_counter() - started) * 1000.0)
        for stage, value in (result.timings_ms or {}).items():
            self.stage_ms[kind.value][stage].append(value)
        action = result.action

        self.decisions += 1
        self.legal += any(action is legal for legal in context.legal_actions)
        guard = self.guard[kind.value]
        guard["rows"] += 1
        expected = guard_action(context, kind)
        if expected is not None:
            agree = result.kind is kind and action == expected
            if is_single_candidate(context, kind):
                self.single["rows"] += 1
                self.single["guard_agree"] += agree
        else:
            agree = action in scorer_candidates(context, kind)
        guard["agree"] += agree

        teacher = row.selected_action
        if teacher_rule_exception(context, kind, teacher) is not None:
            self.exceptions += 1
            return
        if expected is not None:
            return
        position = (row.game_ordinal, row.decision_ordinal)
        if kind is L2aDecisionKind.DISCARD:
            self._train_serve("m1", context, teacher, position)
            self._train_serve("m0", context, teacher, position)
            self._discard(context, result, teacher, counts)
        else:
            self._train_serve("r1", context, teacher, position)
            self._call(context, result, teacher, counts)

    def _train_serve(self, name, context, teacher, position) -> None:
        payload = next(self.payloads[name], None)
        if (
            payload is None
            or (
                payload.row.game_ordinal,
                payload.row.decision_ordinal,
            )
            != position
        ):
            raise LearnedPolicyError(
                f"{name} dataset does not hold the scorer decision at game "
                f"{position[0]} decision {position[1]}"
            )
        encoded = self.encodings[name].encode(context)
        shared = build_player_safe_feature(context.input)
        try:
            teacher_index = resolve_teacher_index(encoded.actions, teacher)
        except DatasetError:
            teacher_index = None
        match = (
            payload.row.actions == encoded.actions
            and payload.row.teacher_candidate_index == teacher_index
            and payload.context == _float32_bytes(shared)
            and payload.candidates
            == b"".join(_float32_bytes(vector) for vector in encoded.vectors)
        )
        counter = self.train_serve[name]
        counter["rows"] += 1
        counter["match"] += match

    def _discard(self, context, result, teacher, counts) -> None:
        baseline = self.baseline_m0.decide(context)
        if baseline.kind is not L2aDecisionKind.DISCARD:
            raise LearnedPolicyError("M0 baseline did not use its DISCARD scorer")
        reference = self.reference_policy(context)
        features = {item.action: item for item in build_scorer_candidates(context)}
        teacher_features = features[teacher]
        increments = Counter({"discard.rows": 1})
        for learner, action in (
            ("m1", result.action),
            ("m0", baseline.action),
            ("b_ts", reference),
        ):
            prefix = f"discard.{learner}"
            match = isinstance(action, DiscardAction) and action.tile == teacher.tile
            increments[f"{prefix}.match"] += match
            increments[f"{prefix}.tsumogiri_match"] += (
                match and action.tsumogiri == teacher.tsumogiri
            )
            chosen = features.get(action)
            if chosen is None:
                continue
            if chosen.post_discard_shanten == teacher_features.post_discard_shanten:
                increments[f"{prefix}.shanten_match"] += 1
                increments[f"{prefix}.ukeire_match"] += (
                    chosen.current_ukeire_count == teacher_features.current_ukeire_count
                )
        self.scores["m1"].add(result.candidates, result.scores, teacher)
        self.scores["m0"].add(baseline.candidates, baseline.scores, teacher)
        c0_specific = not increments["discard.b_ts.match"]
        strata = decision_strata(context.input, c0_specific=c0_specific)
        _add(counts, strata, increments)

    def _call(self, context, result, teacher, counts) -> None:
        passing = next(
            action for action in result.candidates if isinstance(action, PassAction)
        )
        increments = Counter({"call.rows": 1})
        for prefix, action_type in _CALL_TYPES:
            if not any(
                isinstance(action, action_type) for action in context.legal_actions
            ):
                continue
            teacher_calls = isinstance(teacher, action_type)
            increments[f"{prefix}.legal"] += 1
            increments[f"{prefix}.teacher"] += teacher_calls
            for learner, action in (("r1", result.action), ("b_pass", passing)):
                learner_calls = isinstance(action, action_type)
                base = f"{prefix}.{learner}"
                increments[f"{base}.true_positive"] += teacher_calls and learner_calls
                increments[f"{base}.true_negative"] += not (
                    teacher_calls or learner_calls
                )
                increments[f"{base}.action_match"] += (
                    teacher_calls and action == teacher
                )
        increments["call.r1.match"] += result.action == teacher
        increments["call.b_pass.match"] += passing == teacher
        self.scores["r1"].add(result.candidates, result.scores, teacher)
        _add(counts, decision_strata(context.input, c0_specific=None), increments)


def _add(counts, strata, increments: Counter) -> None:
    values = {name: int(value) for name, value in increments.items() if value}
    for stratum in strata:
        counts[stratum].update(values)


def collect_l2a_offline_tally(
    source,
    *,
    proposed,
    baseline_m0,
    datasets: Mapping[str, str | Path],
    reference_policy: ReferencePolicy | None = None,
) -> L2aOfflineTally:
    """OFFLINE-EVALの全行へ提案とbaselineを適用し、半荘別・層別に数える。

    `proposed`はM1 + R1、`baseline_m0`はM0 + R1の合成Policy（`L2aPolicy`）で、
    `datasets`は`{"m1", "m0", "r1"}`のdataset pathである。`reference_policy`は
    B-TSとC0固有層の基準で、既定は`TwoStepUkeirePolicy().choose_action`である。
    """
    if reference_policy is None:
        reference_policy = TwoStepUkeirePolicy().choose_action
    collector = _Collector(proposed, baseline_m0, datasets, reference_policy)
    return collector.run(source)


# --- ledger（OFFLINE-EVALの使用記録） ------------------------------------------------


def _read_ledger(root: Path) -> list[dict[str, object]]:
    entries = []
    for path in sorted(root.glob(f"{LEDGER_ENTRY_PREFIX}*.json")):
        context = f"ledger entry {path.name}"
        value = parse_json_text(
            path.read_text(encoding="utf-8"), LearningError, context
        )
        body = expect_object(
            unseal(value, LearningError, context),
            _LEDGER_FIELDS,
            LearningError,
            context,
        )
        if body["schema"] != L2A_OFFLINE_LEDGER_SCHEMA:
            raise LearningError(f"{context} has an unsupported schema")
        entries.append({"entry": path.name, **body})
    return entries


def record_offline_evaluation(
    ledger: str | Path,
    *,
    artifacts: Mapping[str, str],
    datasets: Mapping[str, str],
    source_identity: str,
) -> tuple[str, list[dict[str, object]]]:
    """評価のentryを新規作成し、(entry名, 使用済みと判定した過去のentry)を返す。

    同じsource identityの同じsplitを評価したentry、または同じartifact identityを
    含むentryを使用済みとする。既存entryは書き換えない。過去のentryが1つでもあれば、
    この評価はOFFLINE-EVALの2回目以降の使用である。
    """
    root = Path(ledger)
    if not root.is_dir():
        raise LearningError(f"ledger directory does not exist: {root}")
    entries = _read_ledger(root)
    identities = set(artifacts.values())
    prior = [
        entry
        for entry in entries
        if (
            entry["source_identity"] == source_identity
            and entry["split"] == L2A_OFFLINE_SPLIT
        )
        or identities & set(entry["artifacts"].values())
    ]
    sequence = 1 + max((entry["sequence"] for entry in entries), default=0)
    name = f"{LEDGER_ENTRY_PREFIX}{sequence:04d}.json"
    body = {
        "artifacts": dict(artifacts),
        "datasets": dict(datasets),
        "gate_fingerprint": l2a_offline_gate_fingerprint(),
        "schema": L2A_OFFLINE_LEDGER_SCHEMA,
        "sequence": sequence,
        "source_identity": source_identity,
        "split": L2A_OFFLINE_SPLIT,
    }
    write_new_text(root / name, canonical_json_text(seal(body)), LearningError)
    return name, prior


# --- 実行 -----------------------------------------------------------------------


def _identity_stop_reasons(source, artifacts, datasets) -> list[str]:
    reasons = []
    for name in ARTIFACT_NAMES:
        artifact = artifacts[name]
        if artifact.encoding.name != name:
            reasons.append(
                f"{name} artifact uses encoding {artifact.encoding.name}, not {name}"
            )
        if artifact.source_identity != source.identity:
            reasons.append(f"{name} artifact was trained from another source")
        if artifact.dataset_identity != datasets[name]["identity"]:
            reasons.append(f"{name} dataset is not the dataset of the {name} artifact")
        if datasets[name]["source"]["identity"] != source.identity:
            reasons.append(f"{name} dataset was materialized from another source")
    classifications = {
        artifact.manifest["encoding"]["classification"]
        for artifact in artifacts.values()
    }
    if len(classifications) != 1:
        reasons.append("artifacts bind different L2a guard rules")
    if not any(game["split"] == L2A_OFFLINE_SPLIT for game in source.population()):
        reasons.append(f"source has no {L2A_OFFLINE_SPLIT} hanchan")
    return reasons


def _provenance(source, artifacts, datasets) -> dict[str, object]:
    return {
        "artifacts": {
            name: {
                "dataset_identity": artifact.dataset_identity,
                "encoding": artifact.manifest["encoding"],
                "identity": artifact.identity,
                "model": artifact.manifest["model"],
                "training": artifact.manifest["training"],
            }
            for name, artifact in artifacts.items()
        },
        "datasets": {name: manifest["identity"] for name, manifest in datasets.items()},
        "source": {
            "identity": source.identity,
            "purpose": source.purpose,
            "schema": source.schema,
            "teacher": None if source.teacher is None else dict(source.teacher),
        },
    }


def run_l2a_offline_eval(
    *,
    source_record: str | Path,
    artifacts: Mapping[str, str | Path],
    datasets: Mapping[str, str | Path],
    ledger: str | Path,
    reference_policy: ReferencePolicy | None = None,
) -> dict[str, object]:
    """OFFLINE-EVALを1回評価し、凍結したgateで判定したreportを返す。

    `artifacts` / `datasets`は`{"m1", "m0", "r1"}`のpathである。identityが
    合わなければledgerへ記録せずSTOP / INVALIDを返す。ML runtimeが必要である。
    """
    from lisjong.learning.l2a_policy import load_l2a_policy_factory

    if set(artifacts) != set(ARTIFACT_NAMES) or set(datasets) != set(ARTIFACT_NAMES):
        raise LearningError(f"artifacts and datasets must be {list(ARTIFACT_NAMES)}")
    source = open_source_record(source_record)
    loaded = {name: load_l2a_artifact(artifacts[name]) for name in ARTIFACT_NAMES}
    manifests = {
        name: read_l2a_dataset_manifest(datasets[name]) for name in ARTIFACT_NAMES
    }
    provenance = _provenance(source, loaded, manifests)
    reasons = _identity_stop_reasons(source, loaded, manifests)
    if reasons:
        return {**judge_l2a_offline_eval(None, reasons), "evaluation": provenance}

    entry, prior = record_offline_evaluation(
        ledger,
        artifacts={name: artifact.identity for name, artifact in loaded.items()},
        datasets={name: manifest["identity"] for name, manifest in manifests.items()},
        source_identity=source.identity,
    )
    provenance["ledger"] = {"entry": entry, "prior_evaluations": prior}
    if prior:
        reason = (
            f"{L2A_OFFLINE_SPLIT} of this source or these artifacts was already "
            "evaluated: " + ", ".join(item["entry"] for item in prior)
        )
        return {**judge_l2a_offline_eval(None, [reason]), "evaluation": provenance}

    proposed = load_l2a_policy_factory(artifacts["m1"], artifacts["r1"])
    baseline = load_l2a_policy_factory(artifacts["m0"], artifacts["r1"])
    if proposed.identity != (loaded["m1"].identity, loaded["r1"].identity) or (
        baseline.identity[0] != loaded["m0"].identity
    ):
        raise LearnedPolicyError("artifacts changed while the evaluation was starting")
    tally = collect_l2a_offline_tally(
        source,
        proposed=proposed(),
        baseline_m0=baseline(),
        datasets=datasets,
        reference_policy=reference_policy,
    )
    return {**judge_l2a_offline_eval(tally), "evaluation": provenance}


__all__ = [
    "ARTIFACT_NAMES",
    "L2A_OFFLINE_LEDGER_SCHEMA",
    "LEDGER_ENTRY_PREFIX",
    "collect_l2a_offline_tally",
    "decision_strata",
    "record_offline_evaluation",
    "run_l2a_offline_eval",
]
