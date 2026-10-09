"""聴牌PUSH/FOLD（H2）の、同じ局面からの押す／降りる対比較sourceの入力契約（lisbun/lisjong#288）。

lisjong-arenaが対局を実行して記録し、lisjongがこの契約で読む。lisjongはArenaのmoduleを
importしない。文書は`docs/tenpai-push-fold-source.md`。

```text
<source directory>/
    manifest.json     lisjong-tenpai-push-fold-source-manifest-v1
    decisions.jsonl   lisjong-tenpai-push-fold-decision-record-v1   player-safe
    outcomes.jsonl    lisjong-tenpai-push-fold-outcome-record-v1    局の結果（学習専用）
```

- `decisions.jsonl`: 対照（Champion×4）のゲート判断1回につき1行。判断者から見える情報
  （`PolicyInput`・合法手）と、`evaluate_tenpai_gate()`の区分・候補・ロン合法確率のrawを持つ
- `outcomes.jsonl`: ゲート判断1回につき、押す側（対照）の局の結果を1行、降りる候補がある判断
  （`fold_ron_legal_raw < push_ron_legal_raw`）ではさらに降りる側の局の結果を1行持つ。
  降りる側は、その判断だけで`a_fold`を切り、それ以外はChampionのまま進めた局である
- 他家の手牌・山などの隠し情報は、どちらのfileにも入れない

schema・digest・行数・キーの欠落/重複・押す側と降りる側の対応の不一致は、すべてエラーにする。
"""

from dataclasses import dataclass
from pathlib import Path

from lisjong.belief.fixed_point import PROBABILITY_MAX_RAW
from lisjong.learning._canonical import (
    canonical_json_line,
    canonical_json_text,
    expect_bool,
    expect_digest,
    expect_int,
    expect_list,
    expect_non_negative_int,
    expect_object,
    expect_str,
    file_digest,
    parse_json_text,
)
from lisjong.learning._typed_values import (
    _parse_enum,
    _parse_seat,
    action_to_value,
    parse_action,
    parse_policy_input,
    policy_input_to_value,
)
from lisjong.learning.riichi_wait_mawashi_policy import SELECTED_WAIT_MODEL_SHA256
from lisjong.learning.ron_legal_estimator import in_riichi_scope
from lisjong.learning.tenpai_push_fold import GATE, GateKind
from lisjong.policies.placement_aware_speed_call import _is_closed_hand
from lisjong.policy_contract.action import DiscardAction, InternalAction, RiichiAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.seat import Seat

MANIFEST_SCHEMA = "lisjong-tenpai-push-fold-source-manifest-v1"
DECISION_SCHEMA = "lisjong-tenpai-push-fold-decision-record-v1"
OUTCOME_SCHEMA = "lisjong-tenpai-push-fold-outcome-record-v1"

MANIFEST_FILENAME = "manifest.json"
DECISIONS_FILENAME = "decisions.jsonl"
OUTCOMES_FILENAME = "outcomes.jsonl"
SPLITS = ("train", "valid")
PUSH = "push"
FOLD = "fold"
WIN_METHODS = ("ron", "tsumo")

_MANIFEST_FIELDS = frozenset({"schema", "gate", "producer", "splits", "files"})
_PRODUCER_FIELDS = frozenset(
    {
        "arena_revision",
        "lisjong_revision",
        "lisjong_engine_revision",
        "policy",
        "wait_model_sha256",
    }
)
_FILE_FIELDS = frozenset({"bytes", "sha256", "rows"})
_KEY_FIELDS = frozenset({"seed", "sequence", "seat"})
_DECISION_FIELDS = frozenset(
    {
        "schema",
        "key",
        "ordinal",
        "policy_input",
        "legal_actions",
        "kind",
        "riichi_seat",
        "c0_action",
        "push_action",
        "fold_action",
        "push_ron_legal_raw",
        "fold_ron_legal_raw",
    }
)
_OUTCOME_FIELDS = frozenset(
    {
        "schema",
        "key",
        "side",
        "round_delta",
        "discard_passed",
        "win",
        "deal_in",
        "exhaustive_draw_tenpai",
        "declaration_discard",
    }
)
_WIN_FIELDS = frozenset({"method", "points"})
_DEAL_IN_FIELDS = frozenset({"to", "points"})


class TenpaiPushFoldSourceError(ValueError):
    """入力契約に合わない、または記録の間に矛盾がある。"""


_E = TenpaiPushFoldSourceError


@dataclass(frozen=True, slots=True, order=True)
class GateDecisionKey:
    """判断を一意に指すキー。`sequence`は半荘内の全席を通した判断の通し番号。"""

    seed: int
    sequence: int
    seat: int


@dataclass(frozen=True, slots=True)
class GateDecisionRecord:
    """対照のゲート判断1回（player-safe）。`ordinal`は半荘内のゲート判断の通し番号（0始まり）。"""

    key: GateDecisionKey
    ordinal: int
    policy_input: PolicyInput
    legal_actions: tuple[InternalAction, ...]
    kind: GateKind
    riichi_seat: Seat
    c0_action: RiichiAction | DiscardAction
    push_action: DiscardAction
    fold_action: DiscardAction
    push_ron_legal_raw: int
    fold_ron_legal_raw: int

    @property
    def has_fold_candidate(self) -> bool:
        return self.fold_ron_legal_raw < self.push_ron_legal_raw

    @property
    def decision(self) -> DecisionContext:
        """`TenpaiPushFoldPairPolicy`の`fold_target`に渡す値。"""
        return DecisionContext(
            input=self.policy_input, legal_actions=self.legal_actions
        )


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    """ゲート判断の後、その局が終わるまでの判断者の結果。

    - `round_delta`: 判断者のその局の収支（供託・本場・リーチ棒を含む）
    - `discard_passed`: 今の打牌（押す側は`push_action`、降りる側は`fold_action`）が誰にも
      ロンされなかったか。偽なら`deal_in`はその打牌での放銃である
    - `win_method` / `win_points`: 判断者が和了した場合の方法と和了点。和了点は本場・供託・
      リーチ棒を含まない（`evaluate_win()`の受取点と同じ単位）
    - `deal_in_to` / `deal_in_points`: 判断者が放銃した場合の和了者の席と支払点（本場を含まない）
    - `exhaustive_draw_tenpai`: 荒牌流局で終わった場合の判断者の聴牌。それ以外の終わり方は`None`
    - `declaration_discard`: (A)の押す側だけが持つ、対照が実際に切った宣言牌の打牌
    """

    key: GateDecisionKey
    side: str
    round_delta: int
    discard_passed: bool
    win_method: str | None
    win_points: int | None
    deal_in_to: Seat | None
    deal_in_points: int | None
    exhaustive_draw_tenpai: bool | None
    declaration_discard: DiscardAction | None


@dataclass(frozen=True, slots=True)
class PairRecord:
    """ゲート判断と、同じ局面からの押す側・降りる側の結果。降りる候補がなければ`fold`は`None`。"""

    decision: GateDecisionRecord
    push: RoundOutcome
    fold: RoundOutcome | None


@dataclass(frozen=True, slots=True)
class TenpaiPushFoldManifest:
    producer: dict[str, str]
    splits: dict[str, tuple[int, ...]]
    files: dict[str, dict[str, object]]

    def split_of(self, seed: int) -> str:
        for name, seeds in self.splits.items():
            if seed in seeds:
                return name
        raise _E(f"seed {seed} is not assigned to a split")


# --- projection (Arenaはこのwire shapeへ変換して書く) -------------------------


def _key_to_value(key: GateDecisionKey) -> dict[str, int]:
    return {"seat": key.seat, "seed": key.seed, "sequence": key.sequence}


def decision_to_value(record: GateDecisionRecord) -> dict[str, object]:
    return {
        "c0_action": action_to_value(record.c0_action, _E, "c0_action"),
        "fold_action": action_to_value(record.fold_action, _E, "fold_action"),
        "fold_ron_legal_raw": record.fold_ron_legal_raw,
        "key": _key_to_value(record.key),
        "kind": record.kind.value,
        "legal_actions": [
            action_to_value(action, _E, f"legal_actions[{index}]")
            for index, action in enumerate(record.legal_actions)
        ],
        "ordinal": record.ordinal,
        "policy_input": policy_input_to_value(record.policy_input, _E, "input"),
        "push_action": action_to_value(record.push_action, _E, "push_action"),
        "push_ron_legal_raw": record.push_ron_legal_raw,
        "riichi_seat": int(record.riichi_seat),
        "schema": DECISION_SCHEMA,
    }


def outcome_to_value(outcome: RoundOutcome) -> dict[str, object]:
    return {
        "deal_in": (
            None
            if outcome.deal_in_to is None
            else {"points": outcome.deal_in_points, "to": int(outcome.deal_in_to)}
        ),
        "declaration_discard": (
            None
            if outcome.declaration_discard is None
            else action_to_value(outcome.declaration_discard, _E, "declaration")
        ),
        "discard_passed": outcome.discard_passed,
        "exhaustive_draw_tenpai": outcome.exhaustive_draw_tenpai,
        "key": _key_to_value(outcome.key),
        "round_delta": outcome.round_delta,
        "schema": OUTCOME_SCHEMA,
        "side": outcome.side,
        "win": (
            None
            if outcome.win_method is None
            else {"method": outcome.win_method, "points": outcome.win_points}
        ),
    }


def manifest_text(
    *,
    producer: dict[str, str],
    splits: dict[str, tuple[int, ...]],
    files: dict[str, dict[str, object]],
) -> str:
    """manifest.jsonの正準テキスト（Arenaは同じshapeで書く）。"""
    return canonical_json_text(
        {
            "files": files,
            "gate": GATE,
            "producer": producer,
            "schema": MANIFEST_SCHEMA,
            "splits": {name: list(splits[name]) for name in SPLITS},
        }
    )


# --- strict readers -----------------------------------------------------------


def _parse_key(value: object, context: str) -> GateDecisionKey:
    raw = expect_object(value, _KEY_FIELDS, _E, context)
    return GateDecisionKey(
        seed=expect_non_negative_int(raw["seed"], _E, f"{context}.seed"),
        sequence=expect_non_negative_int(raw["sequence"], _E, f"{context}.sequence"),
        seat=int(_parse_seat(raw["seat"], _E, f"{context}.seat")),
    )


def _read_lines(path: Path) -> list[tuple[int, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise _E(f"{path.name}:{number} is not a complete line")
            value = parse_json_text(line[:-1], _E, f"{path.name}:{number}")
            if canonical_json_line(value) != line:
                raise _E(f"{path.name}:{number} is not canonical JSON")
            rows.append((number, value))
    return rows


def _parse_discard(value: object, context: str) -> DiscardAction:
    action = parse_action(value, _E, context)
    if not isinstance(action, DiscardAction):
        raise _E(f"{context} must be a discard")
    return action


def _parse_ron_legal_raw(value: object, context: str) -> int:
    raw = expect_non_negative_int(value, _E, context)
    if raw > PROBABILITY_MAX_RAW:
        raise _E(f"{context} exceeds the probability scale")
    return raw


def _parse_decision(value: object, context: str) -> GateDecisionRecord:
    raw = expect_object(value, _DECISION_FIELDS, _E, context)
    if raw["schema"] != DECISION_SCHEMA:
        raise _E(f"{context} has an unsupported schema")
    c0_action = parse_action(raw["c0_action"], _E, f"{context}.c0_action")
    if not isinstance(c0_action, (RiichiAction, DiscardAction)):
        raise _E(f"{context}.c0_action must be a riichi declaration or a discard")
    record = GateDecisionRecord(
        key=_parse_key(raw["key"], f"{context}.key"),
        ordinal=expect_non_negative_int(raw["ordinal"], _E, f"{context}.ordinal"),
        policy_input=parse_policy_input(raw["policy_input"], _E, f"{context}.input"),
        legal_actions=tuple(
            parse_action(item, _E, f"{context}.legal_actions[{index}]")
            for index, item in enumerate(
                expect_list(raw["legal_actions"], _E, f"{context}.legal_actions")
            )
        ),
        kind=_parse_enum(GateKind, raw["kind"], _E, f"{context}.kind"),
        riichi_seat=_parse_seat(raw["riichi_seat"], _E, f"{context}.riichi_seat"),
        c0_action=c0_action,
        push_action=_parse_discard(raw["push_action"], f"{context}.push_action"),
        fold_action=_parse_discard(raw["fold_action"], f"{context}.fold_action"),
        push_ron_legal_raw=_parse_ron_legal_raw(
            raw["push_ron_legal_raw"], f"{context}.push_ron_legal_raw"
        ),
        fold_ron_legal_raw=_parse_ron_legal_raw(
            raw["fold_ron_legal_raw"], f"{context}.fold_ron_legal_raw"
        ),
    )
    if decision_to_value(record) != value:
        raise _E(f"{context} does not round-trip through the canonical projection")
    _check_decision(record, context)
    return record


def _check_decision(record: GateDecisionRecord, context: str) -> None:
    policy_input = record.policy_input
    seat = policy_input.self_seat
    if int(seat) != record.key.seat:
        raise _E(f"{context}.key.seat does not match the decider")
    try:
        record.decision
    except (TypeError, ValueError) as error:
        raise _E(f"{context}.legal_actions are not a decision: {error}") from error
    for name in ("c0_action", "push_action", "fold_action"):
        if getattr(record, name) not in record.legal_actions:
            raise _E(f"{context}.{name} is not a legal action")
    if record.riichi_seat is seat or not in_riichi_scope(
        policy_input, record.riichi_seat
    ):
        raise _E(f"{context} is outside the gate: not a single riichi opponent")
    if isinstance(record.c0_action, RiichiAction):
        expected = GateKind.RIICHI
    elif record.c0_action != record.push_action:
        raise _E(f"{context}.push_action differs from the Champion discard")
    elif _is_closed_hand(policy_input):
        expected = GateKind.CLOSED_DISCARD
    else:
        expected = GateKind.OPEN_DISCARD
    if record.kind is not expected:
        raise _E(f"{context}.kind does not match the decision")


def _parse_outcome(value: object, context: str) -> RoundOutcome:
    raw = expect_object(value, _OUTCOME_FIELDS, _E, context)
    if raw["schema"] != OUTCOME_SCHEMA:
        raise _E(f"{context} has an unsupported schema")
    if raw["side"] not in (PUSH, FOLD):
        raise _E(f"{context}.side must be push or fold")
    win_method = win_points = deal_in_to = deal_in_points = None
    if raw["win"] is not None:
        win = expect_object(raw["win"], _WIN_FIELDS, _E, f"{context}.win")
        if win["method"] not in WIN_METHODS:
            raise _E(f"{context}.win.method must be ron or tsumo")
        win_method = win["method"]
        win_points = _positive(win["points"], f"{context}.win.points")
    if raw["deal_in"] is not None:
        deal_in = expect_object(
            raw["deal_in"], _DEAL_IN_FIELDS, _E, f"{context}.deal_in"
        )
        deal_in_to = _parse_seat(deal_in["to"], _E, f"{context}.deal_in.to")
        deal_in_points = _positive(deal_in["points"], f"{context}.deal_in.points")
    draw = raw["exhaustive_draw_tenpai"]
    outcome = RoundOutcome(
        key=_parse_key(raw["key"], f"{context}.key"),
        side=raw["side"],
        round_delta=expect_int(raw["round_delta"], _E, f"{context}.round_delta"),
        discard_passed=expect_bool(
            raw["discard_passed"], _E, f"{context}.discard_passed"
        ),
        win_method=win_method,
        win_points=win_points,
        deal_in_to=deal_in_to,
        deal_in_points=deal_in_points,
        exhaustive_draw_tenpai=(
            None
            if draw is None
            else expect_bool(draw, _E, f"{context}.exhaustive_draw_tenpai")
        ),
        declaration_discard=(
            None
            if raw["declaration_discard"] is None
            else _parse_discard(raw["declaration_discard"], f"{context}.declaration")
        ),
    )
    if outcome_to_value(outcome) != value:
        raise _E(f"{context} does not round-trip through the canonical projection")
    endings = (
        outcome.win_method is not None,
        outcome.deal_in_to is not None,
        outcome.exhaustive_draw_tenpai is not None,
    )
    if sum(endings) > 1:
        raise _E(f"{context} has more than one way the round ended")
    if not outcome.discard_passed and outcome.deal_in_to is None:
        raise _E(f"{context}: the discard did not pass but there is no deal-in")
    if outcome.deal_in_to is not None and int(outcome.deal_in_to) == outcome.key.seat:
        raise _E(f"{context}.deal_in.to is the decider")
    return outcome


def _positive(value: object, context: str) -> int:
    if expect_int(value, _E, context) <= 0:
        raise _E(f"{context} must be positive")
    return value


def read_manifest(
    directory: str | Path,
    *,
    expected_wait_model_sha256: str = SELECTED_WAIT_MODEL_SHA256,
) -> TenpaiPushFoldManifest:
    """manifestを読む。ゲートの版・待ちモデルのSHA-256がこのcodeと違えば拒否する。"""
    root = Path(directory)
    text = (root / MANIFEST_FILENAME).read_text(encoding="utf-8")
    raw = expect_object(
        parse_json_text(text, _E, "manifest"), _MANIFEST_FIELDS, _E, "manifest"
    )
    if raw["schema"] != MANIFEST_SCHEMA:
        raise _E("manifest has an unsupported schema")
    if raw["gate"] != GATE:
        raise _E("manifest was produced with a different gate version")
    producer = expect_object(raw["producer"], _PRODUCER_FIELDS, _E, "producer")
    for name, value in producer.items():
        expect_str(value, _E, f"producer.{name}")
    if producer["wait_model_sha256"] != expected_wait_model_sha256:
        raise _E("manifest was produced with a different wait model")
    splits_raw = expect_object(raw["splits"], frozenset(SPLITS), _E, "splits")
    splits: dict[str, tuple[int, ...]] = {}
    seen: set[int] = set()
    for name in SPLITS:
        seeds = tuple(
            expect_non_negative_int(seed, _E, f"splits.{name}")
            for seed in expect_list(splits_raw[name], _E, f"splits.{name}")
        )
        if seen & set(seeds) or len(set(seeds)) != len(seeds):
            raise _E("splits must not share or repeat a seed")
        seen |= set(seeds)
        splits[name] = seeds
    files_raw = expect_object(
        raw["files"], frozenset({"decisions", "outcomes"}), _E, "files"
    )
    files = {}
    for name, value in files_raw.items():
        entry = expect_object(value, _FILE_FIELDS, _E, f"files.{name}")
        expect_digest(entry["sha256"], _E, f"files.{name}.sha256")
        expect_non_negative_int(entry["bytes"], _E, f"files.{name}.bytes")
        expect_non_negative_int(entry["rows"], _E, f"files.{name}.rows")
        files[name] = entry
    if text != manifest_text(producer=producer, splits=splits, files=files):
        raise _E("manifest is not canonical JSON")
    return TenpaiPushFoldManifest(producer=dict(producer), splits=splits, files=files)


def _read_file(root: Path, filename: str, entry: dict[str, object], parse) -> tuple:
    digest = file_digest(root / filename)
    if digest != {"bytes": entry["bytes"], "sha256": entry["sha256"]}:
        raise _E(f"{filename} does not match the manifest digest")
    rows = tuple(
        parse(value, f"{filename}:{number}")
        for number, value in _read_lines(root / filename)
    )
    if len(rows) != entry["rows"]:
        raise _E(f"{filename} row count does not match the manifest")
    return rows


def _read_decisions(
    root: Path, manifest: TenpaiPushFoldManifest
) -> dict[GateDecisionKey, GateDecisionRecord]:
    by_key: dict[GateDecisionKey, GateDecisionRecord] = {}
    for record in _read_file(
        root, DECISIONS_FILENAME, manifest.files["decisions"], _parse_decision
    ):
        if record.key in by_key:
            raise _E(f"duplicate decision key {record.key}")
        manifest.split_of(record.key.seed)
        by_key[record.key] = record
    per_seed: dict[int, list[int]] = {}
    for key in sorted(by_key):
        per_seed.setdefault(key.seed, []).append(by_key[key].ordinal)
    for seed, ordinals in per_seed.items():
        if ordinals != list(range(len(ordinals))):
            raise _E(f"seed {seed}: gate ordinals are not 0..n-1 in sequence order")
    return by_key


def read_decisions(
    directory: str | Path,
    *,
    expected_wait_model_sha256: str = SELECTED_WAIT_MODEL_SHA256,
) -> tuple[TenpaiPushFoldManifest, tuple[GateDecisionRecord, ...]]:
    """player-safeなゲート判断の記録だけを読む。局の結果のfileは開かない。"""
    root = Path(directory)
    manifest = read_manifest(
        root, expected_wait_model_sha256=expected_wait_model_sha256
    )
    by_key = _read_decisions(root, manifest)
    return manifest, tuple(by_key[key] for key in sorted(by_key))


def read_source(
    directory: str | Path,
    *,
    expected_wait_model_sha256: str = SELECTED_WAIT_MODEL_SHA256,
) -> tuple[TenpaiPushFoldManifest, tuple[PairRecord, ...]]:
    """ゲート判断と局の結果を読み、押す側・降りる側を対にして返す（キー順）。"""
    root = Path(directory)
    manifest = read_manifest(
        root, expected_wait_model_sha256=expected_wait_model_sha256
    )
    by_key = _read_decisions(root, manifest)
    outcomes: dict[tuple[GateDecisionKey, str], RoundOutcome] = {}
    for outcome in _read_file(
        root, OUTCOMES_FILENAME, manifest.files["outcomes"], _parse_outcome
    ):
        if outcome.key not in by_key:
            raise _E(f"outcome key {outcome.key} has no decision")
        if (outcome.key, outcome.side) in outcomes:
            raise _E(f"duplicate {outcome.side} outcome for {outcome.key}")
        outcomes[outcome.key, outcome.side] = outcome

    pairs = []
    for key in sorted(by_key):
        record = by_key[key]
        push = outcomes.get((key, PUSH))
        fold = outcomes.get((key, FOLD))
        if push is None:
            raise _E(f"decision {key} has no push outcome")
        if (fold is not None) != record.has_fold_candidate:
            raise _E(
                f"decision {key}: the fold outcome does not match the fold candidate"
            )
        if (push.declaration_discard is not None) != (record.kind is GateKind.RIICHI):
            raise _E(
                f"decision {key}: declaration_discard is for riichi push sides only"
            )
        if fold is not None and fold.declaration_discard is not None:
            raise _E(f"decision {key}: a fold side has no declaration discard")
        if (
            push.declaration_discard is not None
            and int(push.declaration_discard.actor) != key.seat
        ):
            raise _E(f"decision {key}: declaration_discard is not the decider's")
        pairs.append(PairRecord(decision=record, push=push, fold=fold))
    return manifest, tuple(pairs)


__all__ = [
    "DECISION_SCHEMA",
    "DECISIONS_FILENAME",
    "FOLD",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "OUTCOME_SCHEMA",
    "OUTCOMES_FILENAME",
    "PUSH",
    "SPLITS",
    "WIN_METHODS",
    "GateDecisionKey",
    "GateDecisionRecord",
    "PairRecord",
    "RoundOutcome",
    "TenpaiPushFoldManifest",
    "TenpaiPushFoldSourceError",
    "decision_to_value",
    "manifest_text",
    "outcome_to_value",
    "read_decisions",
    "read_manifest",
    "read_source",
]
