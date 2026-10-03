"""単独リーチ者に対する放銃確率推定器（lisbun/lisjong#237 S1）の入力契約とラベル処理。

lisjong-arenaが対局を実行して**観測事実**を記録し、lisjongがこの契約で読み込んで
ラベルを計算する。lisjongはArenaのmoduleをimportしない（逆依存なし）。

```text
<source directory>/
    manifest.json        lisjong-riichi-deal-in-source-manifest-v1
    decisions.jsonl      lisjong-riichi-deal-in-decision-record-v1   player-safe
    label_facts.jsonl    lisjong-riichi-deal-in-label-fact-record-v1 学習専用
```

- ``decisions.jsonl`` は判断者から見える情報だけを持つ（``PolicyInput``・合法手・
  選んだ行動）。推定器の推論入力はこのfileだけから作る。``read_decisions()`` は
  ``label_facts.jsonl`` を開かない
- ``label_facts.jsonl`` は隠し情報（リーチ者の手牌）と、選んだ打牌のその後の事実を
  持つ学習専用の記録。``read_labelled_source()`` だけが読み、``DecisionKey`` で
  判断記録と結合する。キーの欠落・重複・不一致はエラー
- フリテンかどうか、ラベルAが何かは、記録された事実から
  ``lisjong.belief.riichi_ron_label.riichi_ron_label()`` で**lisjongが計算する**。
  Arenaは判定しない
- 時点: リーチ者の手牌と和了選択肢の記録は、対象判断より前（``sequence`` が小さい）
  でなければならない。対象判断より後の情報が入っていればエラー
- engineとの照合（リーチ者へRonActionが提示されたか）は**選んだ打牌だけ**に適用する。
  選ばなかった候補には照合値を持たせない

同じ記録を、構造的な待ち（lisbun/lisjong#245）のラベルにも使う（``read_wait_labelled_source()``）。
待ちは各行に結合されたリーチ者の手牌から``exact_hand_belief_with_waits()``で求め、フリテン・
ロン可否は含まない。判断記録との結合と時点の検査はラベルAと共通である。

対象範囲（``SCOPE``）: 他家1人がリーチ中（宣言済みを含む）・自分は非リーチ・
候補牌種2以上の打牌判断。#237の「他家単独リーチ」全体より狭い。
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from lisjong.belief.canonical_axes import tile_type_from_index
from lisjong.belief.exact_wait_ground_truth import exact_hand_belief_with_waits
from lisjong.belief.riichi_ron_label import RiichiFuritenReason, riichi_ron_label
from lisjong.learning._canonical import (
    canonical_json_line,
    canonical_json_text,
    expect_bool,
    expect_digest,
    expect_list,
    expect_non_negative_int,
    expect_object,
    expect_str,
    file_digest,
    parse_json_text,
)
from lisjong.learning._typed_values import (
    _parse_seat,
    action_to_value,
    meld_to_value,
    parse_action,
    parse_meld,
    parse_policy_input,
    parse_tiles,
    policy_input_to_value,
    tiles_to_value,
)
from lisjong.policy_contract.action import (
    DiscardAction,
    InternalAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.meld import PublicMeld
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.tile import Tile, TileType

MANIFEST_SCHEMA = "lisjong-riichi-deal-in-source-manifest-v1"
DECISION_SCHEMA = "lisjong-riichi-deal-in-decision-record-v1"
LABEL_FACT_SCHEMA = "lisjong-riichi-deal-in-label-fact-record-v1"
SCOPE = "single-opponent-riichi.self-not-riichi.at-least-two-candidate-tile-types.v1"

MANIFEST_FILENAME = "manifest.json"
DECISIONS_FILENAME = "decisions.jsonl"
LABEL_FACTS_FILENAME = "label_facts.jsonl"
SPLITS = ("train", "valid", "test")

_MANIFEST_FIELDS = frozenset({"schema", "scope", "producer", "splits", "files"})
_PRODUCER_FIELDS = frozenset(
    {"arena_revision", "lisjong_revision", "lisjong_engine_revision", "policy"}
)
_FILE_FIELDS = frozenset({"bytes", "sha256", "rows"})
_KEY_FIELDS = frozenset({"seed", "sequence", "seat"})
_DECISION_FIELDS = frozenset(
    {"schema", "key", "policy_input", "legal_actions", "selected_action"}
)
_FACT_FIELDS = frozenset(
    {
        "schema",
        "key",
        "riichi_seat",
        "riichi_declared_sequence",
        "riichi_hand",
        "win_options",
        "selected_outcome",
    }
)
_HAND_FIELDS = frozenset({"sequence", "concealed_tiles", "melds"})
_WIN_OPTION_FIELDS = frozenset({"sequence", "winning_tiles", "selected_action"})
_OUTCOME_FIELDS = frozenset({"ron_offered", "dealt_in"})


class RiichiDealInSourceError(ValueError):
    """入力契約に合わない、または事実の間に矛盾がある。"""


_E = RiichiDealInSourceError


@dataclass(frozen=True, slots=True, order=True)
class DecisionKey:
    """判断を一意に指すキー。``sequence`` は半荘内の全席を通した判断の通し番号。"""

    seed: int
    sequence: int
    seat: int


@dataclass(frozen=True, slots=True)
class RiichiDealInDecision:
    """player-safeな判断記録（推論入力の元）。"""

    key: DecisionKey
    policy_input: PolicyInput
    legal_actions: tuple[InternalAction, ...]
    selected_action: DiscardAction


@dataclass(frozen=True, slots=True)
class WinOption:
    """リーチ者に和了（Ron/Tsumo）の選択肢が提示された時点と、選んだ行動。"""

    sequence: int
    winning_tiles: tuple[Tile, ...]
    selected_action: InternalAction

    @property
    def passed(self) -> bool:
        return not isinstance(self.selected_action, (RonAction, TsumoAction))


@dataclass(frozen=True, slots=True)
class RiichiDealInLabelFacts:
    """学習専用の観測事実。判断者には見えない情報を含む。"""

    key: DecisionKey
    riichi_seat: int
    riichi_declared_sequence: int
    hand_sequence: int
    concealed_tiles: tuple[Tile, ...]
    melds: tuple[PublicMeld, ...]
    win_options: tuple[WinOption, ...]
    ron_offered: bool
    dealt_in: bool


@dataclass(frozen=True, slots=True)
class CandidateLabel:
    tile_type: TileType
    label_a: bool


@dataclass(frozen=True, slots=True)
class LabelledDecision:
    """ラベル付きの判断。``selected_*`` は選んだ打牌だけが持つ。"""

    decision: RiichiDealInDecision
    riichi_seat: int
    candidates: tuple[CandidateLabel, ...]
    furiten_reasons: frozenset[RiichiFuritenReason]
    selected_tile_type: TileType
    selected_ron_offered: bool
    selected_dealt_in: bool


@dataclass(frozen=True, slots=True, order=True)
class RiichiEpisodeKey:
    """リーチ1回を指すキー。リーチ宣言の通し番号は半荘内で一意なので、seedと合わせて一意。"""

    seed: int
    riichi_seat: int
    declared_sequence: int


@dataclass(frozen=True, slots=True)
class WaitLabelledDecision:
    """構造的な待ちのラベル付き判断。``wait_tile_types``は学習専用の正解。"""

    decision: RiichiDealInDecision
    riichi_seat: int
    episode: RiichiEpisodeKey
    wait_tile_types: frozenset[TileType]


@dataclass(frozen=True, slots=True)
class RiichiDealInManifest:
    producer: dict[str, str]
    splits: dict[str, tuple[int, ...]]
    files: dict[str, dict[str, object]]

    def split_of(self, seed: int) -> str:
        for name, seeds in self.splits.items():
            if seed in seeds:
                return name
        raise _E(f"seed {seed} is not assigned to a split")


# --- projection (Arenaはこのwire shapeへ変換して書く) -------------------------


def _key_to_value(key: DecisionKey) -> dict[str, int]:
    return {"seat": key.seat, "seed": key.seed, "sequence": key.sequence}


def decision_to_value(decision: RiichiDealInDecision) -> dict[str, object]:
    return {
        "key": _key_to_value(decision.key),
        "legal_actions": [
            action_to_value(action, _E, f"legal_actions[{index}]")
            for index, action in enumerate(decision.legal_actions)
        ],
        "policy_input": policy_input_to_value(decision.policy_input, _E, "input"),
        "schema": DECISION_SCHEMA,
        "selected_action": action_to_value(decision.selected_action, _E, "selected"),
    }


def label_facts_to_value(facts: RiichiDealInLabelFacts) -> dict[str, object]:
    return {
        "key": _key_to_value(facts.key),
        "riichi_declared_sequence": facts.riichi_declared_sequence,
        "riichi_hand": {
            "concealed_tiles": tiles_to_value(facts.concealed_tiles, _E, "hand"),
            "melds": [
                meld_to_value(meld, _E, f"melds[{index}]")
                for index, meld in enumerate(facts.melds)
            ],
            "sequence": facts.hand_sequence,
        },
        "riichi_seat": facts.riichi_seat,
        "schema": LABEL_FACT_SCHEMA,
        "selected_outcome": {
            "dealt_in": facts.dealt_in,
            "ron_offered": facts.ron_offered,
        },
        "win_options": [
            {
                "selected_action": action_to_value(
                    option.selected_action, _E, f"win_options[{index}].selected"
                ),
                "sequence": option.sequence,
                "winning_tiles": tiles_to_value(
                    option.winning_tiles, _E, f"win_options[{index}].winning_tiles"
                ),
            }
            for index, option in enumerate(facts.win_options)
        ],
    }


# --- strict readers -----------------------------------------------------------


def _parse_key(value: object, context: str) -> DecisionKey:
    raw = expect_object(value, _KEY_FIELDS, _E, context)
    seat = int(_parse_seat(raw["seat"], _E, f"{context}.seat"))
    return DecisionKey(
        seed=expect_non_negative_int(raw["seed"], _E, f"{context}.seed"),
        sequence=expect_non_negative_int(raw["sequence"], _E, f"{context}.sequence"),
        seat=seat,
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


def _parse_decision(value: object, context: str) -> RiichiDealInDecision:
    raw = expect_object(value, _DECISION_FIELDS, _E, context)
    if raw["schema"] != DECISION_SCHEMA:
        raise _E(f"{context} has an unsupported schema")
    key = _parse_key(raw["key"], f"{context}.key")
    decision = RiichiDealInDecision(
        key=key,
        policy_input=parse_policy_input(raw["policy_input"], _E, f"{context}.input"),
        legal_actions=tuple(
            parse_action(item, _E, f"{context}.legal_actions[{index}]")
            for index, item in enumerate(
                expect_list(raw["legal_actions"], _E, f"{context}.legal_actions")
            )
        ),
        selected_action=parse_action(raw["selected_action"], _E, f"{context}.selected"),
    )
    if decision_to_value(decision) != value:
        raise _E(f"{context} does not round-trip through the canonical projection")
    _check_decision(decision, context)
    return decision


def _discard_candidates(decision: RiichiDealInDecision) -> frozenset[TileType]:
    return frozenset(
        action.tile.tile_type
        for action in decision.legal_actions
        if isinstance(action, DiscardAction)
    )


def _riichi_opponent(decision: RiichiDealInDecision, context: str) -> int:
    pi = decision.policy_input
    seat = int(pi.self_seat)
    if pi.players[seat].riichi is not RiichiState.NONE:
        raise _E(f"{context} is outside the scope: the decider is in riichi")
    opponents = [
        index
        for index, player in enumerate(pi.players)
        if index != seat and player.riichi is not RiichiState.NONE
    ]
    if len(opponents) != 1:
        raise _E(f"{context} is outside the scope: not exactly one riichi opponent")
    return opponents[0]


def _check_decision(decision: RiichiDealInDecision, context: str) -> None:
    pi = decision.policy_input
    if int(pi.self_seat) != decision.key.seat:
        raise _E(f"{context}.key.seat does not match the decider")
    if not isinstance(decision.selected_action, DiscardAction):
        raise _E(f"{context}.selected must be a discard")
    if decision.selected_action not in decision.legal_actions:
        raise _E(f"{context}.selected is not a legal action")
    if len(_discard_candidates(decision)) < 2:
        raise _E(f"{context} is outside the scope: fewer than two candidate types")
    _riichi_opponent(decision, context)


def _parse_facts(value: object, context: str) -> RiichiDealInLabelFacts:
    raw = expect_object(value, _FACT_FIELDS, _E, context)
    if raw["schema"] != LABEL_FACT_SCHEMA:
        raise _E(f"{context} has an unsupported schema")
    hand = expect_object(raw["riichi_hand"], _HAND_FIELDS, _E, f"{context}.hand")
    outcome = expect_object(
        raw["selected_outcome"], _OUTCOME_FIELDS, _E, f"{context}.outcome"
    )
    options = []
    for index, item in enumerate(
        expect_list(raw["win_options"], _E, f"{context}.win_options")
    ):
        option_context = f"{context}.win_options[{index}]"
        option = expect_object(item, _WIN_OPTION_FIELDS, _E, option_context)
        options.append(
            WinOption(
                sequence=expect_non_negative_int(
                    option["sequence"], _E, f"{option_context}.sequence"
                ),
                winning_tiles=parse_tiles(
                    option["winning_tiles"], _E, f"{option_context}.winning_tiles"
                ),
                selected_action=parse_action(
                    option["selected_action"], _E, f"{option_context}.selected"
                ),
            )
        )
    facts = RiichiDealInLabelFacts(
        key=_parse_key(raw["key"], f"{context}.key"),
        riichi_seat=int(_parse_seat(raw["riichi_seat"], _E, f"{context}.riichi_seat")),
        riichi_declared_sequence=expect_non_negative_int(
            raw["riichi_declared_sequence"], _E, f"{context}.riichi_declared_sequence"
        ),
        hand_sequence=expect_non_negative_int(
            hand["sequence"], _E, f"{context}.hand.sequence"
        ),
        concealed_tiles=parse_tiles(
            hand["concealed_tiles"], _E, f"{context}.hand.concealed_tiles"
        ),
        melds=tuple(
            parse_meld(item, _E, f"{context}.hand.melds[{index}]")
            for index, item in enumerate(
                expect_list(hand["melds"], _E, f"{context}.hand.melds")
            )
        ),
        win_options=tuple(options),
        ron_offered=expect_bool(outcome["ron_offered"], _E, f"{context}.ron_offered"),
        dealt_in=expect_bool(outcome["dealt_in"], _E, f"{context}.dealt_in"),
    )
    if label_facts_to_value(facts) != value:
        raise _E(f"{context} does not round-trip through the canonical projection")
    return facts


def _unique_by_key(items: Iterable, name: str) -> dict[DecisionKey, object]:
    result: dict[DecisionKey, object] = {}
    for item in items:
        if item.key in result:
            raise _E(f"duplicate {name} key {item.key}")
        result[item.key] = item
    return result


def read_manifest(directory: str | Path) -> RiichiDealInManifest:
    root = Path(directory)
    raw = expect_object(
        parse_json_text(
            (root / MANIFEST_FILENAME).read_text(encoding="utf-8"), _E, "manifest"
        ),
        _MANIFEST_FIELDS,
        _E,
        "manifest",
    )
    if raw["schema"] != MANIFEST_SCHEMA:
        raise _E("manifest has an unsupported schema")
    if raw["scope"] != SCOPE:
        raise _E("manifest has an unsupported scope")
    producer = expect_object(raw["producer"], _PRODUCER_FIELDS, _E, "producer")
    for name, value in producer.items():
        expect_str(value, _E, f"producer.{name}")
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
        raw["files"], frozenset({"decisions", "label_facts"}), _E, "files"
    )
    files = {}
    for name, value in files_raw.items():
        entry = expect_object(value, _FILE_FIELDS, _E, f"files.{name}")
        expect_digest(entry["sha256"], _E, f"files.{name}.sha256")
        expect_non_negative_int(entry["bytes"], _E, f"files.{name}.bytes")
        expect_non_negative_int(entry["rows"], _E, f"files.{name}.rows")
        files[name] = entry
    if (root / MANIFEST_FILENAME).read_text(encoding="utf-8") != manifest_text(
        producer=producer, splits=splits, files=files
    ):
        raise _E("manifest is not canonical JSON")
    return RiichiDealInManifest(producer=dict(producer), splits=splits, files=files)


def _check_file(root: Path, filename: str, entry: dict[str, object]) -> None:
    digest = file_digest(root / filename)
    if digest != {"bytes": entry["bytes"], "sha256": entry["sha256"]}:
        raise _E(f"{filename} does not match the manifest digest")


def read_decisions(
    directory: str | Path,
) -> tuple[RiichiDealInManifest, tuple[RiichiDealInDecision, ...]]:
    """player-safeな判断記録だけを読む（推論入力の経路）。ラベルの記録は開かない。"""
    root = Path(directory)
    manifest = read_manifest(root)
    _check_file(root, DECISIONS_FILENAME, manifest.files["decisions"])
    decisions = tuple(
        _parse_decision(value, f"{DECISIONS_FILENAME}:{number}")
        for number, value in _read_lines(root / DECISIONS_FILENAME)
    )
    if len(decisions) != manifest.files["decisions"]["rows"]:
        raise _E("decision row count does not match the manifest")
    _unique_by_key(decisions, "decision")
    for decision in decisions:
        manifest.split_of(decision.key.seed)
    return manifest, decisions


def _paired_with_facts(
    decisions: Sequence[RiichiDealInDecision],
    facts: Sequence[RiichiDealInLabelFacts],
) -> list[tuple[RiichiDealInDecision, RiichiDealInLabelFacts, int]]:
    """判断記録と観測事実を`DecisionKey`で結合し、時点の整合を検査する。

    戻り値の各要素は（判断、事実、リーチ者の席）で、キー順に並ぶ。
    """
    by_decision = _unique_by_key(decisions, "decision")
    by_facts = _unique_by_key(facts, "label fact")
    if set(by_decision) != set(by_facts):
        raise _E("decision and label fact keys do not match")
    paired = []
    for key in sorted(by_decision):
        decision, fact = by_decision[key], by_facts[key]
        context = f"decision {key}"
        riichi = _riichi_opponent(decision, context)
        if fact.riichi_seat != riichi:
            raise _E(f"{context}: label facts name a different riichi seat")
        if not (fact.riichi_declared_sequence <= fact.hand_sequence < key.sequence):
            raise _E(f"{context}: the riichi hand is not from before the decision")
        sequences = [option.sequence for option in fact.win_options]
        if sequences != sorted(set(sequences)) or any(
            not fact.riichi_declared_sequence < sequence < key.sequence
            for sequence in sequences
        ):
            raise _E(f"{context}: win options are outside the riichi..decision window")
        paired.append((decision, fact, riichi))
    return paired


def label_decisions(
    decisions: Sequence[RiichiDealInDecision],
    facts: Sequence[RiichiDealInLabelFacts],
) -> tuple[LabelledDecision, ...]:
    """観測事実からラベルAを計算し、判断記録と結合する（学習専用）。"""
    labelled = []
    for decision, fact, riichi in _paired_with_facts(decisions, facts):
        key = decision.key
        context = f"decision {key}"
        label = riichi_ron_label(
            fact.concealed_tiles,
            fact.melds,
            own_discards=[
                d.tile for d in decision.policy_input.players[riichi].discards
            ],
            passed_tile_types=[
                tile.tile_type
                for option in fact.win_options
                if option.passed
                for tile in option.winning_tiles
            ],
        )
        selected = decision.selected_action.tile.tile_type
        if label.can_ron(selected) != fact.ron_offered:
            raise _E(f"{context}: label A disagrees with the engine's ron offer")
        if fact.dealt_in and not fact.ron_offered:
            raise _E(f"{context}: dealt in without a ron offer")
        labelled.append(
            LabelledDecision(
                decision=decision,
                riichi_seat=riichi,
                candidates=tuple(
                    CandidateLabel(
                        tile_type=tile_type, label_a=label.can_ron(tile_type)
                    )
                    for tile_type in sorted(
                        _discard_candidates(decision),
                        key=lambda t: (t.category.value, t.rank),
                    )
                ),
                furiten_reasons=label.furiten_reasons,
                selected_tile_type=selected,
                selected_ron_offered=fact.ron_offered,
                selected_dealt_in=fact.dealt_in,
            )
        )
    return tuple(labelled)


def label_waits(
    decisions: Sequence[RiichiDealInDecision],
    facts: Sequence[RiichiDealInLabelFacts],
) -> tuple[WaitLabelledDecision, ...]:
    """観測事実から構造的な待ち（lisjong#245のラベル）を計算し、判断記録と結合する。

    待ちは、各行に結合されたリーチ者の手牌（`hand_sequence`時点の門前牌と副露）から
    `exact_hand_belief_with_waits()`で求める。ラベルはその行の手牌から作り、行をまたいで
    使い回さない。フリテン・ロン可否は含まない（構造的な待ち）。学習専用。
    """
    labelled = []
    for decision, fact, riichi in _paired_with_facts(decisions, facts):
        belief = exact_hand_belief_with_waits(fact.concealed_tiles, fact.melds)
        waits = frozenset(
            tile_type_from_index(index)
            for index in range(34)
            if (belief.wait_probability(tile_type_from_index(index)) or 0) > 0
        )
        if not waits:
            raise _E(f"decision {decision.key}: the riichi hand is not tenpai")
        labelled.append(
            WaitLabelledDecision(
                decision=decision,
                riichi_seat=riichi,
                episode=RiichiEpisodeKey(
                    seed=decision.key.seed,
                    riichi_seat=riichi,
                    declared_sequence=fact.riichi_declared_sequence,
                ),
                wait_tile_types=waits,
            )
        )
    return tuple(labelled)


def read_labelled_source(
    directory: str | Path,
) -> tuple[RiichiDealInManifest, tuple[LabelledDecision, ...]]:
    """判断記録とラベルの記録を読み、ラベル付きの判断を返す（学習専用の経路）。"""
    root = Path(directory)
    manifest, decisions = read_decisions(root)
    _check_file(root, LABEL_FACTS_FILENAME, manifest.files["label_facts"])
    facts = tuple(
        _parse_facts(value, f"{LABEL_FACTS_FILENAME}:{number}")
        for number, value in _read_lines(root / LABEL_FACTS_FILENAME)
    )
    if len(facts) != manifest.files["label_facts"]["rows"]:
        raise _E("label fact row count does not match the manifest")
    return manifest, label_decisions(decisions, facts)


def read_wait_labelled_source(
    directory: str | Path,
) -> tuple[RiichiDealInManifest, tuple[WaitLabelledDecision, ...]]:
    """判断記録とラベルの記録を読み、構造的な待ちのラベル付き判断を返す（学習専用の経路）。"""
    root = Path(directory)
    manifest, decisions = read_decisions(root)
    _check_file(root, LABEL_FACTS_FILENAME, manifest.files["label_facts"])
    facts = tuple(
        _parse_facts(value, f"{LABEL_FACTS_FILENAME}:{number}")
        for number, value in _read_lines(root / LABEL_FACTS_FILENAME)
    )
    if len(facts) != manifest.files["label_facts"]["rows"]:
        raise _E("label fact row count does not match the manifest")
    return manifest, label_waits(decisions, facts)


def manifest_text(
    *,
    producer: dict[str, str],
    splits: dict[str, Sequence[int]],
    files: dict[str, dict[str, object]],
) -> str:
    """manifest.json の正準テキスト（Arenaは同じshapeで書く）。"""
    return canonical_json_text(
        {
            "files": files,
            "producer": producer,
            "schema": MANIFEST_SCHEMA,
            "scope": SCOPE,
            "splits": {name: list(splits[name]) for name in SPLITS},
        }
    )


__all__ = [
    "DECISION_SCHEMA",
    "DECISIONS_FILENAME",
    "LABEL_FACT_SCHEMA",
    "LABEL_FACTS_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "SCOPE",
    "SPLITS",
    "CandidateLabel",
    "DecisionKey",
    "LabelledDecision",
    "RiichiDealInDecision",
    "RiichiDealInLabelFacts",
    "RiichiDealInManifest",
    "RiichiDealInSourceError",
    "RiichiEpisodeKey",
    "WaitLabelledDecision",
    "WinOption",
    "decision_to_value",
    "label_decisions",
    "label_waits",
    "label_facts_to_value",
    "manifest_text",
    "read_decisions",
    "read_labelled_source",
    "read_manifest",
    "read_wait_labelled_source",
]
