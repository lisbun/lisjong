"""L0.3 focal outcome sourceのstrict consumerとoutcome target（Issue #193、#79 A1）。

Arenaが生成するfocal outcome source（lisbun/lisjong-arena#359）を読み、
#191 survivorとfocal exploration selectionを再計算・照合したうえで、
lisjong所有のtarget `target_q`を算出する。

```text
Arena-owned focal outcome source（arena-offense-l0.3-focal-outcome-source-v1）
    -> strict schema / digest / provenance / behavior identity validation
    -> focal decisionごとにselect_residual_exploration()を再計算して照合
    -> eligible row（DISCARD かつ survivor >= 2）
         target_q = (points_after_kyoku[focal] - points_before_kyoku[focal]) / 1000
```

source schemaはproducerであるArenaが所有する論理contractであり、lisjongは
strict consumerになる。`lisjong_arena`はimportせず、Arena package codeも
再利用しない。既存の`source_record.py`（#342 / #189）は変更しない。

## Wire layout

```text
<root>/
  manifest.json            sealed canonical JSON document
  game-000/
    kyokus.jsonl           kyoku row（kyoku_ordinal順、全kyoku）
    focal-decisions.jsonl  focal decision row（focal_decision_ordinal順）
  game-001/
  ...
```

manifest body（`identity`はbodyのcanonical digest）。

```text
schema               arena-offense-l0.3-focal-outcome-source-v1
kind                 focal-outcome-source-record
population_role      CALIBRATION | SCIENTIFIC
games[]              sealed game summary（下記）、game_ordinal順
allocation_bindings  {split: #346 / #347 per-split binding}（v2 source recordと同じshape）
behavior             exploration_behavior_identity / baseline_runtime_identity /
                     exploration_token_identity / focal_rotation_rule
source_contract      Arena-owned JSON object（digestとしてのみ保持）
```

game summary: `game_ordinal`, `seed`, `split`, `focal_seat`, `kyoku_count`,
`decision_count`, `hanchan_final_scores[4]`, `hanchan_final_riichi_sticks`,
`files`（payload filenameごとの`bytes` / `sha256`）。

kyoku row: `game_ordinal`, `kyoku_ordinal`, `round_wind`, `hand_number`,
`honba`, `dealer_seat`, `riichi_sticks_before`, `riichi_sticks_after`,
`points_before_kyoku[4]`, `points_after_kyoku[4]`, `end`, `is_final_kyoku`。
`end`は`{"kind": "win", "winner_seats": [...]}`または
`{"kind": "draw", "draw_kind": "..."}`である。

focal decision row: `game_ordinal`, `kyoku_ordinal`, `focal_decision_ordinal`,
`step_ordinal`, `actor_seat`, `policy_input`, `legal_actions`,
`selected_action`, `exploration_token`, `producer_survivor_actions`
（DISCARDではcanonical順のsurvivor action列、それ以外は`null`）。

## Score boundary

`points_after_kyoku`はArenaが記録した**hanchan最終調整前**の事実値をそのまま
使う。lisjongは精算を再実装せず、backendの最終調整を逆算・推定もしない。
`hanchan_final_scores` / `hanchan_final_riichi_sticks`はshapeだけを検証する
audit factであり、targetには使わない。

## Fail closed

未対応schema / kind、digest不一致、非canonical JSON、fieldの欠損・余剰、
behavior identity不一致、game_ordinalの欠番・重複・split境界でのrestart、
`focal_seat != game_ordinal % 4`、seed重複、population構成違反、
focal_decision_ordinalの欠番、focal decision間で`step_ordinal`が狭義増加しない /
`kyoku_ordinal`が減少する、actor / PolicyInput seat不一致、legal actions
の空・重複・非canonical順、非legalなselected action、guard action / survivor
action列 / bucket選択の不一致、存在しないkyokuの参照、PolicyInputとkyokuの
round_wind / hand_number / honba不一致、点数snapshotのshape違反、kyoku間の
`points_before` / 供託本数の不連続。
"""

import statistics
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

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
    unseal,
    value_digest,
)
from lisjong.learning._o0 import O0DecisionKind
from lisjong.learning._typed_values import (
    _construct,
    _parse_enum,
    action_to_value,
    parse_action,
    parse_policy_input,
    policy_input_to_value,
)
from lisjong.learning.candidate_features import DiscardCandidateFeatures
from lisjong.learning.errors import (
    OutcomeSourceError,
    SourceRecordError,
    UnsupportedSourceSchemaError,
)
from lisjong.learning.residual_baseline import CONSTANT_RESIDUAL_RUNTIME_IDENTITY
from lisjong.learning.residual_exploration import (
    RESIDUAL_EXPLORATION_BEHAVIOR_IDENTITY,
    ResidualExplorationDecision,
    select_residual_exploration,
)
from lisjong.learning.source_record import validate_allocation_bindings
from lisjong.policy_contract import (
    DecisionContext,
    InternalAction,
    Seat,
    Wind,
)

OUTCOME_SOURCE_SCHEMA = "arena-offense-l0.3-focal-outcome-source-v1"
"""consumeするArena-owned source schema（lisbun/lisjong-arena#359）。"""

OUTCOME_SOURCE_KIND = "focal-outcome-source-record"

EXPLORATION_TOKEN_IDENTITY = "lisjong-arena-l0.3-focal-decision-token-sha256-v1"
"""Arena-owned token derivationのidentity。固定文字列として照合するだけで、
lisjongはtokenのSHA-256を再計算しない。"""

FOCAL_ROTATION_RULE = "lisjong-arena-l0.3-focal-seat-game-ordinal-mod-4-v1"
"""`focal_seat = game_ordinal % 4`（ordinalはsource全体のglobal ordinal）。"""

OUTCOME_TARGET_IDENTITY = "lisjong-offense-l0.3-focal-kyoku-point-delta-1000-v1"
"""`target_q = (points_after_kyoku[focal] - points_before_kyoku[focal]) / 1000`。"""

OUTCOME_OBJECTIVE_IDENTITY = "lisjong-offense-l0.3-selected-action-mc-q-mse-v1"
"""#79 Aでfreeze済みのobjective identity。trainer実装は#79 step Dで行う。"""

MANIFEST_FILENAME = "manifest.json"
KYOKU_PAYLOAD_FILENAME = "kyokus.jsonl"
DECISION_PAYLOAD_FILENAME = "focal-decisions.jsonl"

CALIBRATION_ROLE = "CALIBRATION"
SCIENTIFIC_ROLE = "SCIENTIFIC"
_ROLE_SPLITS = {
    CALIBRATION_ROLE: frozenset({"CALIBRATION"}),
    SCIENTIFIC_ROLE: frozenset({"TRAIN", "SELECT"}),
}

EXPECTED_BEHAVIOR = {
    "baseline_runtime_identity": CONSTANT_RESIDUAL_RUNTIME_IDENTITY,
    "exploration_behavior_identity": RESIDUAL_EXPLORATION_BEHAVIOR_IDENTITY,
    "exploration_token_identity": EXPLORATION_TOKEN_IDENTITY,
    "focal_rotation_rule": FOCAL_ROTATION_RULE,
}
"""source manifestの`behavior`が完全一致すべき値。"""

SUMMARY_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "population_role",
        "games",
        "allocation_bindings",
        "behavior",
        "source_contract",
    }
)
_GAME_FIELDS = frozenset(
    {
        "game_ordinal",
        "seed",
        "split",
        "focal_seat",
        "kyoku_count",
        "decision_count",
        "hanchan_final_scores",
        "hanchan_final_riichi_sticks",
        "files",
    }
)
_PAYLOAD_FILES = frozenset({KYOKU_PAYLOAD_FILENAME, DECISION_PAYLOAD_FILENAME})
_PAYLOAD_FIELDS = frozenset({"bytes", "sha256"})
_KYOKU_FIELDS = frozenset(
    {
        "game_ordinal",
        "kyoku_ordinal",
        "round_wind",
        "hand_number",
        "honba",
        "dealer_seat",
        "riichi_sticks_before",
        "riichi_sticks_after",
        "points_before_kyoku",
        "points_after_kyoku",
        "end",
        "is_final_kyoku",
    }
)
_WIN_END_FIELDS = frozenset({"kind", "winner_seats"})
_DRAW_END_FIELDS = frozenset({"kind", "draw_kind"})
_DECISION_FIELDS = frozenset(
    {
        "game_ordinal",
        "kyoku_ordinal",
        "focal_decision_ordinal",
        "step_ordinal",
        "actor_seat",
        "policy_input",
        "legal_actions",
        "selected_action",
        "exploration_token",
        "producer_survivor_actions",
    }
)


@dataclass(frozen=True, slots=True)
class OutcomeKyoku:
    """1 kyokuの事実値（Arenaが記録したpre-hanchan-final-adjustment境界）。"""

    kyoku_ordinal: int
    round_wind: Wind
    hand_number: int
    honba: int
    dealer_seat: Seat
    riichi_sticks_before: int
    riichi_sticks_after: int
    points_before_kyoku: tuple[int, int, int, int]
    points_after_kyoku: tuple[int, int, int, int]
    end: dict[str, object]
    is_final_kyoku: bool


@dataclass(frozen=True, slots=True)
class OutcomeDecision:
    """1 focal decisionと、lisjongが再計算して照合したselection結果。"""

    kyoku_ordinal: int
    focal_decision_ordinal: int
    step_ordinal: int
    decision: DecisionContext
    selected_action: InternalAction
    exploration_token: str
    selection: ResidualExplorationDecision


@dataclass(frozen=True, slots=True)
class OutcomeGame:
    """1 hanchanのprovenance、kyoku列、focal decision列。"""

    game_ordinal: int
    seed: int
    split: str
    focal_seat: Seat
    hanchan_final_scores: tuple[int, int, int, int]
    hanchan_final_riichi_sticks: int
    kyokus: tuple[OutcomeKyoku, ...]
    decisions: tuple[OutcomeDecision, ...]


@dataclass(frozen=True, slots=True)
class FocalOutcomeSource:
    """strict readした完全なfocal outcome sourceとそのArena-owned provenance。"""

    identity: str
    population_role: str
    behavior: dict[str, str]
    allocation_bindings: dict[str, dict[str, str]]
    source_contract_digest: str
    games: tuple[OutcomeGame, ...]

    def decisions(self) -> Iterator[tuple[OutcomeGame, OutcomeDecision]]:
        """game順、game内はfocal_decision_ordinal順に列挙する。"""
        for game in self.games:
            for decision in game.decisions:
                yield game, decision

    def provenance(self) -> dict[str, object]:
        return {
            "allocation_bindings": {
                split: dict(binding)
                for split, binding in self.allocation_bindings.items()
            },
            "behavior": dict(self.behavior),
            "identity": self.identity,
            "population": [
                {
                    "focal_seat": int(game.focal_seat),
                    "game_ordinal": game.game_ordinal,
                    "seed": game.seed,
                    "split": game.split,
                }
                for game in self.games
            ],
            "population_role": self.population_role,
            "schema": OUTCOME_SOURCE_SCHEMA,
            "source_contract_digest": self.source_contract_digest,
        }


@dataclass(frozen=True, slots=True)
class OutcomeTargetRow:
    """eligible row（DISCARD かつ survivor >= 2）1件。

    `target_q`は選択したcandidate（`selected_candidate_index`）だけに付く。
    選ばれなかったsurvivorにはtargetを付与しない。
    """

    split: str
    game_ordinal: int
    seed: int
    kyoku_ordinal: int
    focal_decision_ordinal: int
    step_ordinal: int
    decision: DecisionContext
    candidates: tuple[DiscardCandidateFeatures, ...]
    survivors: tuple[int, ...]
    selected_candidate_index: int
    target_q: float

    @property
    def selected_candidate(self) -> DiscardCandidateFeatures:
        return self.candidates[self.selected_candidate_index]


@dataclass(frozen=True, slots=True)
class OutcomeTargets:
    """source全体のeligible rowと除外件数。"""

    source_identity: str
    rows: tuple[OutcomeTargetRow, ...]
    excluded: dict[str, int]
    hanchan_count: int

    @property
    def target_identity(self) -> str:
        return OUTCOME_TARGET_IDENTITY


# ---------------------------------------------------------------------------
# strict read
# ---------------------------------------------------------------------------


def _read_manifest(root: Path) -> dict[str, object]:
    path = root / MANIFEST_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OutcomeSourceError(
            f"outcome source manifest cannot be read: {path}"
        ) from exc
    manifest = parse_json_text(text, OutcomeSourceError, "outcome source manifest")
    body = unseal(manifest, OutcomeSourceError, "outcome source manifest")
    if text != canonical_json_text(manifest):
        raise OutcomeSourceError("outcome source manifest is not canonical JSON")
    if body.get("schema") != OUTCOME_SOURCE_SCHEMA:
        raise UnsupportedSourceSchemaError(
            f"unsupported outcome source schema: {body.get('schema')!r}; "
            f"this implementation consumes {OUTCOME_SOURCE_SCHEMA!r}"
        )
    expect_object(body, _MANIFEST_FIELDS, OutcomeSourceError, "outcome manifest")
    if body["kind"] != OUTCOME_SOURCE_KIND:
        raise OutcomeSourceError("outcome source kind mismatch")
    if body["population_role"] not in _ROLE_SPLITS:
        raise OutcomeSourceError("manifest.population_role is not supported")
    behavior = expect_object(
        body["behavior"],
        frozenset(EXPECTED_BEHAVIOR),
        OutcomeSourceError,
        "manifest.behavior",
    )
    for field, expected in EXPECTED_BEHAVIOR.items():
        if behavior[field] != expected:
            raise OutcomeSourceError(f"manifest.behavior.{field} mismatch")
    if type(body["source_contract"]) is not dict:
        raise OutcomeSourceError("manifest.source_contract must be a JSON object")
    games = expect_list(body["games"], OutcomeSourceError, "manifest.games")
    if not games:
        raise OutcomeSourceError("outcome source must contain at least one hanchan")
    return manifest


def _expect_scores(value: object, context: str) -> tuple[int, int, int, int]:
    scores = expect_list(value, OutcomeSourceError, context)
    if len(scores) != 4:
        raise OutcomeSourceError(f"{context} must have exactly 4 seats")
    return tuple(
        expect_int(score, OutcomeSourceError, f"{context}[{seat}]")
        for seat, score in enumerate(scores)
    )


def _expect_seat(value: object, context: str) -> Seat:
    raw = expect_non_negative_int(value, OutcomeSourceError, context)
    if raw > 3:
        raise OutcomeSourceError(f"{context} is not a valid seat")
    return Seat(raw)


def _read_game_summary(
    value: object, *, game_ordinal: int, population_role: str
) -> dict[str, object]:
    context = f"manifest.games[{game_ordinal}]"
    body = unseal(value, OutcomeSourceError, context)
    expect_object(body, _GAME_FIELDS, OutcomeSourceError, context)
    for field in ("game_ordinal", "seed", "kyoku_count", "decision_count"):
        expect_non_negative_int(body[field], OutcomeSourceError, f"{context}.{field}")
    if body["game_ordinal"] != game_ordinal:
        raise OutcomeSourceError(
            f"{context} game_ordinal is not the contiguous global ordinal"
        )
    split = expect_str(body["split"], OutcomeSourceError, f"{context}.split")
    if split not in _ROLE_SPLITS[population_role]:
        raise OutcomeSourceError(
            f"{context}.split {split!r} is not allowed for population "
            f"role {population_role!r}"
        )
    if _expect_seat(body["focal_seat"], f"{context}.focal_seat") != Seat(
        game_ordinal % 4
    ):
        raise OutcomeSourceError(f"{context}.focal_seat != game_ordinal % 4")
    if body["kyoku_count"] == 0:
        raise OutcomeSourceError(f"{context} must contain at least one kyoku")
    _expect_scores(body["hanchan_final_scores"], f"{context}.hanchan_final_scores")
    expect_non_negative_int(
        body["hanchan_final_riichi_sticks"],
        OutcomeSourceError,
        f"{context}.hanchan_final_riichi_sticks",
    )
    files = expect_object(
        body["files"], _PAYLOAD_FILES, OutcomeSourceError, f"{context}.files"
    )
    for name in sorted(_PAYLOAD_FILES):
        payload = expect_object(
            files[name], _PAYLOAD_FIELDS, OutcomeSourceError, f"{context}.files[{name}]"
        )
        expect_non_negative_int(
            payload["bytes"], OutcomeSourceError, f"{context}.files[{name}].bytes"
        )
        expect_digest(
            payload["sha256"], OutcomeSourceError, f"{context}.files[{name}].sha256"
        )
    return body


def _read_line(line: str, fields: frozenset[str], context: str) -> dict[str, object]:
    row = parse_json_text(line, OutcomeSourceError, context)
    if line != canonical_json_line(row):
        raise OutcomeSourceError(f"{context} is not canonical JSON")
    return expect_object(row, fields, OutcomeSourceError, context)


def _read_end(value: object, context: str) -> dict[str, object]:
    if type(value) is not dict:
        raise OutcomeSourceError(f"{context} must be a JSON object")
    kind = value.get("kind")
    if kind == "win":
        expect_object(value, _WIN_END_FIELDS, OutcomeSourceError, context)
        winners = expect_list(
            value["winner_seats"], OutcomeSourceError, f"{context}.winner_seats"
        )
        seats = [
            _expect_seat(seat, f"{context}.winner_seats[{index}]")
            for index, seat in enumerate(winners)
        ]
        if not seats or len(set(seats)) != len(seats):
            raise OutcomeSourceError(
                f"{context}.winner_seats must be non-empty and unique"
            )
    elif kind == "draw":
        expect_object(value, _DRAW_END_FIELDS, OutcomeSourceError, context)
        expect_str(value["draw_kind"], OutcomeSourceError, f"{context}.draw_kind")
    else:
        raise OutcomeSourceError(f"{context}.kind must be 'win' or 'draw'")
    return value


def _read_kyoku(line: str, *, game_ordinal: int, index: int, context: str):
    row = _read_line(line, _KYOKU_FIELDS, context)
    if row["game_ordinal"] != game_ordinal:
        raise OutcomeSourceError(f"{context}.game_ordinal does not match the game")
    if row["kyoku_ordinal"] != index:
        raise OutcomeSourceError(f"{context}.kyoku_ordinal is not contiguous")
    hand_number = expect_int(
        row["hand_number"], OutcomeSourceError, f"{context}.hand_number"
    )
    if not 1 <= hand_number <= 4:
        raise OutcomeSourceError(f"{context}.hand_number must be between 1 and 4")
    return OutcomeKyoku(
        kyoku_ordinal=index,
        round_wind=_parse_enum(
            Wind, row["round_wind"], OutcomeSourceError, f"{context}.round_wind"
        ),
        hand_number=hand_number,
        honba=expect_non_negative_int(
            row["honba"], OutcomeSourceError, f"{context}.honba"
        ),
        dealer_seat=_expect_seat(row["dealer_seat"], f"{context}.dealer_seat"),
        riichi_sticks_before=expect_non_negative_int(
            row["riichi_sticks_before"],
            OutcomeSourceError,
            f"{context}.riichi_sticks_before",
        ),
        riichi_sticks_after=expect_non_negative_int(
            row["riichi_sticks_after"],
            OutcomeSourceError,
            f"{context}.riichi_sticks_after",
        ),
        points_before_kyoku=_expect_scores(
            row["points_before_kyoku"], f"{context}.points_before_kyoku"
        ),
        points_after_kyoku=_expect_scores(
            row["points_after_kyoku"], f"{context}.points_after_kyoku"
        ),
        end=_read_end(row["end"], f"{context}.end"),
        is_final_kyoku=expect_bool(
            row["is_final_kyoku"], OutcomeSourceError, f"{context}.is_final_kyoku"
        ),
    )


def _parse_actions(value: object, context: str) -> tuple[InternalAction, ...]:
    values = expect_list(value, OutcomeSourceError, context)
    actions = tuple(
        parse_action(item, OutcomeSourceError, f"{context}[{index}]")
        for index, item in enumerate(values)
    )
    roundtripped = [
        action_to_value(action, OutcomeSourceError, f"{context}[{index}]")
        for index, action in enumerate(actions)
    ]
    if roundtripped != values:
        raise OutcomeSourceError(f"{context} does not round trip")
    return actions


def _read_decision(
    line: str,
    *,
    game_ordinal: int,
    focal_seat: Seat,
    index: int,
    kyokus: tuple[OutcomeKyoku, ...],
    context: str,
) -> OutcomeDecision:
    row = _read_line(line, _DECISION_FIELDS, context)
    if row["game_ordinal"] != game_ordinal:
        raise OutcomeSourceError(f"{context}.game_ordinal does not match the game")
    if row["focal_decision_ordinal"] != index:
        raise OutcomeSourceError(f"{context}.focal_decision_ordinal is not contiguous")
    step_ordinal = expect_non_negative_int(
        row["step_ordinal"], OutcomeSourceError, f"{context}.step_ordinal"
    )
    kyoku_ordinal = expect_non_negative_int(
        row["kyoku_ordinal"], OutcomeSourceError, f"{context}.kyoku_ordinal"
    )
    if kyoku_ordinal >= len(kyokus):
        raise OutcomeSourceError(f"{context} references a missing kyoku")
    kyoku = kyokus[kyoku_ordinal]
    if _expect_seat(row["actor_seat"], f"{context}.actor_seat") != focal_seat:
        raise OutcomeSourceError(f"{context}.actor_seat is not the focal seat")

    policy_input = parse_policy_input(
        row["policy_input"], OutcomeSourceError, f"{context}.policy_input"
    )
    if (
        policy_input_to_value(
            policy_input, OutcomeSourceError, f"{context}.policy_input"
        )
        != row["policy_input"]
    ):
        raise OutcomeSourceError(f"{context}.policy_input does not round trip")
    if policy_input.self_seat != focal_seat:
        raise OutcomeSourceError(f"{context} PolicyInput seat is not the focal seat")
    round_state = policy_input.round
    if (
        round_state.round_wind is not kyoku.round_wind
        or round_state.hand_number != kyoku.hand_number
        or round_state.honba != kyoku.honba
    ):
        raise OutcomeSourceError(
            f"{context} PolicyInput round_wind / hand_number / honba does not "
            "match the kyoku record"
        )

    legal_actions = _parse_actions(row["legal_actions"], f"{context}.legal_actions")
    if not legal_actions:
        raise OutcomeSourceError(f"{context}.legal_actions must not be empty")
    canonical_lines = [canonical_json_line(value) for value in row["legal_actions"]]
    if canonical_lines != sorted(canonical_lines):
        raise OutcomeSourceError(f"{context}.legal_actions is not canonically ordered")
    if len(set(canonical_lines)) != len(canonical_lines) or len(
        set(legal_actions)
    ) != len(legal_actions):
        raise OutcomeSourceError(f"{context}.legal_actions contains duplicates")
    (selected,) = _parse_actions([row["selected_action"]], f"{context}.selected_action")
    if selected not in legal_actions:
        raise OutcomeSourceError(f"{context}.selected_action is not legal")
    token = expect_digest(
        row["exploration_token"], OutcomeSourceError, f"{context}.exploration_token"
    )

    decision = _construct(
        DecisionContext,
        OutcomeSourceError,
        context,
        input=policy_input,
        legal_actions=legal_actions,
    )
    selection = select_residual_exploration(decision, token)
    producer_survivors = row["producer_survivor_actions"]
    if selection.kind is O0DecisionKind.DISCARD:
        if producer_survivors is None:
            raise OutcomeSourceError(
                f"{context}.producer_survivor_actions is missing for a DISCARD decision"
            )
        recorded = _parse_actions(
            producer_survivors, f"{context}.producer_survivor_actions"
        )
        if recorded != selection.survivor_actions:
            raise OutcomeSourceError(
                f"{context}.producer_survivor_actions does not match the "
                "recomputed semantic-envelope survivors"
            )
        if selected != selection.action:
            raise OutcomeSourceError(
                f"{context}.selected_action does not match the exploration "
                "bucket selection"
            )
    else:
        if producer_survivors is not None:
            raise OutcomeSourceError(
                f"{context}.producer_survivor_actions must be null for an O0 "
                "guard decision"
            )
        if selected != selection.action:
            raise OutcomeSourceError(
                f"{context}.selected_action does not match the deterministic "
                "O0 guard action"
            )
    return OutcomeDecision(
        kyoku_ordinal=kyoku_ordinal,
        focal_decision_ordinal=index,
        step_ordinal=step_ordinal,
        decision=decision,
        selected_action=selection.action,
        exploration_token=token,
        selection=selection,
    )


def _read_game(path: Path, summary: dict[str, object]) -> OutcomeGame:
    game_ordinal = summary["game_ordinal"]
    focal_seat = Seat(summary["focal_seat"])
    context = f"game-{game_ordinal:03d}"
    if not path.is_dir():
        raise OutcomeSourceError(f"missing outcome source game directory: {context}")
    if {child.name for child in path.iterdir()} != _PAYLOAD_FILES:
        raise OutcomeSourceError(f"missing/unexpected payload files in {context}")
    for name in sorted(_PAYLOAD_FILES):
        if file_digest(path / name) != summary["files"][name]:
            raise OutcomeSourceError(f"{context}/{name} size/digest mismatch")

    kyokus: list[OutcomeKyoku] = []
    with (path / KYOKU_PAYLOAD_FILENAME).open(encoding="utf-8", newline="\n") as stream:
        for index, line in enumerate(stream):
            kyoku = _read_kyoku(
                line,
                game_ordinal=game_ordinal,
                index=index,
                context=f"{context}.kyokus[{index}]",
            )
            if kyokus and (
                kyoku.points_before_kyoku != kyokus[-1].points_after_kyoku
                or kyoku.riichi_sticks_before != kyokus[-1].riichi_sticks_after
            ):
                raise OutcomeSourceError(
                    f"{context}.kyokus[{index}] points_before / riichi_sticks_before "
                    "do not continue from the previous kyoku"
                )
            kyokus.append(kyoku)
    if len(kyokus) != summary["kyoku_count"]:
        raise OutcomeSourceError(f"{context} kyoku count mismatch")
    if [kyoku.is_final_kyoku for kyoku in kyokus] != [False] * (len(kyokus) - 1) + [
        True
    ]:
        raise OutcomeSourceError(f"{context} is_final_kyoku must mark only the last")

    decisions: list[OutcomeDecision] = []
    with (path / DECISION_PAYLOAD_FILENAME).open(
        encoding="utf-8", newline="\n"
    ) as stream:
        for index, line in enumerate(stream):
            decision = _read_decision(
                line,
                game_ordinal=game_ordinal,
                focal_seat=focal_seat,
                index=index,
                kyokus=tuple(kyokus),
                context=f"{context}.decisions[{index}]",
            )
            if decisions and (
                decision.step_ordinal <= decisions[-1].step_ordinal
                or decision.kyoku_ordinal < decisions[-1].kyoku_ordinal
            ):
                raise OutcomeSourceError(
                    f"{context}.decisions[{index}] execution ordering mismatch"
                )
            decisions.append(decision)
    if len(decisions) != summary["decision_count"]:
        raise OutcomeSourceError(f"{context} decision count mismatch")

    return OutcomeGame(
        game_ordinal=game_ordinal,
        seed=summary["seed"],
        split=summary["split"],
        focal_seat=focal_seat,
        hanchan_final_scores=tuple(summary["hanchan_final_scores"]),
        hanchan_final_riichi_sticks=summary["hanchan_final_riichi_sticks"],
        kyokus=tuple(kyokus),
        decisions=tuple(decisions),
    )


def read_outcome_source(path: str | Path) -> FocalOutcomeSource:
    """focal outcome source directoryをstrict readし、検証済みsourceを返す。

    validationは全体としてfail closedである。1 rowでも不整合があれば、
    成功分だけを返さず例外を送出する。
    """
    root = Path(path)
    if not root.is_dir():
        raise OutcomeSourceError(f"outcome source directory does not exist: {root}")
    manifest = _read_manifest(root)
    population_role = manifest["population_role"]
    summaries = [
        _read_game_summary(value, game_ordinal=ordinal, population_role=population_role)
        for ordinal, value in enumerate(manifest["games"])
    ]
    expected_names = {MANIFEST_FILENAME} | {
        f"game-{ordinal:03d}" for ordinal in range(len(summaries))
    }
    if {child.name for child in root.iterdir()} != expected_names:
        raise OutcomeSourceError("missing/unexpected outcome source files")

    seeds: set[int] = set()
    seeds_by_split: dict[str, list[int]] = {}
    for summary in summaries:
        if summary["seed"] in seeds:
            raise OutcomeSourceError(
                "outcome source population must not reuse a seed across hanchan/splits"
            )
        seeds.add(summary["seed"])
        seeds_by_split.setdefault(summary["split"], []).append(summary["seed"])
    try:
        allocation_bindings = validate_allocation_bindings(
            manifest["allocation_bindings"],
            populations=seeds_by_split,
            context="manifest.allocation_bindings",
        )
    except SourceRecordError as exc:
        raise OutcomeSourceError(str(exc)) from exc

    games = tuple(
        _read_game(root / f"game-{summary['game_ordinal']:03d}", summary)
        for summary in summaries
    )
    return FocalOutcomeSource(
        identity=manifest["identity"],
        population_role=population_role,
        behavior=dict(manifest["behavior"]),
        allocation_bindings=allocation_bindings,
        source_contract_digest=value_digest(manifest["source_contract"]),
        games=games,
    )


# ---------------------------------------------------------------------------
# target
# ---------------------------------------------------------------------------


def outcome_target_q(kyoku: OutcomeKyoku, focal_seat: Seat) -> float:
    """`OUTCOME_TARGET_IDENTITY`のtarget。hanchan最終調整後のscoreは使わない。"""
    seat = int(focal_seat)
    return (kyoku.points_after_kyoku[seat] - kyoku.points_before_kyoku[seat]) / 1000.0


def build_outcome_targets(source: FocalOutcomeSource) -> OutcomeTargets:
    """eligible row（DISCARD かつ survivor >= 2）へtarget_qを付与する。"""
    rows: list[OutcomeTargetRow] = []
    excluded = Counter(
        {kind.value: 0 for kind in O0DecisionKind if kind is not O0DecisionKind.DISCARD}
    )
    excluded["single_survivor"] = 0
    for game, item in source.decisions():
        selection = item.selection
        if selection.kind is not O0DecisionKind.DISCARD:
            excluded[selection.kind.value] += 1
            continue
        if len(selection.survivors) == 1:
            excluded["single_survivor"] += 1
            continue
        rows.append(
            OutcomeTargetRow(
                split=game.split,
                game_ordinal=game.game_ordinal,
                seed=game.seed,
                kyoku_ordinal=item.kyoku_ordinal,
                focal_decision_ordinal=item.focal_decision_ordinal,
                step_ordinal=item.step_ordinal,
                decision=item.decision,
                candidates=selection.candidates,
                survivors=selection.survivors,
                selected_candidate_index=selection.selected_candidate_index,
                target_q=outcome_target_q(
                    game.kyokus[item.kyoku_ordinal], game.focal_seat
                ),
            )
        )
    return OutcomeTargets(
        source_identity=source.identity,
        rows=tuple(rows),
        excluded=dict(sorted(excluded.items())),
        hanchan_count=len(source.games),
    )


def _quantile(ordered: list[float], q: float) -> float:
    """inclusive linear interpolation（numpy `linear`と同じ定義）。"""
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_outcome_targets(targets: OutcomeTargets) -> dict[str, object]:
    """C0 pilot用の最小deterministic summary。統計的qualificationは行わない。"""
    rows = targets.rows
    kyokus = {(row.game_ordinal, row.kyoku_ordinal) for row in rows}
    survivor_counts = Counter(len(row.survivors) for row in rows)
    canonical_first = sum(
        1 for row in rows if row.selected_candidate_index == row.survivors[0]
    )
    values = sorted(row.target_q for row in rows)
    target_summary: dict[str, object] | None = None
    if values:
        target_summary = {
            "max": values[-1],
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": values[0],
            "quantiles": {f"{q:.2f}": _quantile(values, q) for q in SUMMARY_QUANTILES},
            "std": statistics.pstdev(values),
            "zero_fraction": sum(1 for value in values if value == 0.0) / len(values),
        }
    return {
        "canonical_first_selected_rate": canonical_first / len(rows) if rows else None,
        "eligible_row_count": len(rows),
        "excluded": dict(targets.excluded),
        "hanchan_count": targets.hanchan_count,
        "non_canonical_first_selected_rate": (len(rows) - canonical_first) / len(rows)
        if rows
        else None,
        "source_identity": targets.source_identity,
        "survivor_count_distribution": {
            str(count): survivor_counts[count] for count in sorted(survivor_counts)
        },
        "target_identity": OUTCOME_TARGET_IDENTITY,
        "target_q": target_summary,
        "unique_eligible_kyoku_count": len(kyokus),
        "unique_eligible_kyoku_per_hanchan": len(kyokus) / targets.hanchan_count,
    }


__all__ = [
    "CALIBRATION_ROLE",
    "DECISION_PAYLOAD_FILENAME",
    "EXPECTED_BEHAVIOR",
    "EXPLORATION_TOKEN_IDENTITY",
    "FOCAL_ROTATION_RULE",
    "KYOKU_PAYLOAD_FILENAME",
    "MANIFEST_FILENAME",
    "OUTCOME_OBJECTIVE_IDENTITY",
    "OUTCOME_SOURCE_KIND",
    "OUTCOME_SOURCE_SCHEMA",
    "OUTCOME_TARGET_IDENTITY",
    "SCIENTIFIC_ROLE",
    "FocalOutcomeSource",
    "OutcomeDecision",
    "OutcomeGame",
    "OutcomeKyoku",
    "OutcomeTargetRow",
    "OutcomeTargets",
    "build_outcome_targets",
    "outcome_target_q",
    "read_outcome_source",
    "summarize_outcome_targets",
]
