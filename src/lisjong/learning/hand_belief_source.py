"""HandBelief精度評価（lisbun/lisjong#255）用の、他家全員の手牌正解の入力契約とラベル計算。

lisbun/lisjong#256。lisjong-arenaが対局を実行して**観測事実**を記録し、lisjongがこの
契約で読み込んでラベルを計算する（#237 S1の``riichi_deal_in_source``と同じ方式）。
lisjongはArenaのmoduleをimportしない（逆依存なし）。

```text
<source directory>/
    manifest.json      lisjong-hand-belief-source-manifest-v1
    decisions.jsonl    lisjong-hand-belief-decision-record-v1   player-safe
    hand_facts.jsonl   lisjong-hand-belief-hand-fact-record-v1  学習専用
```

- ``decisions.jsonl`` は観測者から見える情報だけを持つ（判断時点の``PolicyInput``・
  合法手・選んだ行動）。推定器の推論入力はこのfileだけから作る。``read_decisions()`` は
  ``hand_facts.jsonl`` を開かない
- ``hand_facts.jsonl`` は他家3席それぞれのconcealed tiles（赤5を区別する）と副露を持つ
  学習専用の記録。``read_labelled_source()`` だけが読み、``DecisionKey`` で判断記録と
  結合する。キーの欠落・重複・不一致はエラー
- 時点: 各他家の手牌の ``sequence`` は判断と同じでなければならない。これはその判断の行動を
  適用する前に取ったsnapshotを意味する（観測者の行動は他家の手牌を変えないので、判断時点の
  他家の手牌そのものである）。前の時点の手牌は途中のツモ・打牌で変わっている可能性があり、
  副露・牌保存則の検査だけでは現在の手牌と保証できないので、前後どちらもエラーにする
- 判断時点との整合: 各他家の副露は ``PolicyInput`` の公開副露と一致しなければならない。
  他家3席のconcealed tilesの合計は、``PolicyInput`` から見て未確定の牌
  （``derive_remaining_tile_inventory()``）を牌種別・赤5別・通常5別に超えてはならない
- ラベル: 各他家について、手牌から ``exact_hand_belief_with_waits()`` で正解の
  ``HandBelief``（``expected_count`` / ``red_five_probability`` / ``wait_probability`` /
  形別7テーブル、すべて0か``SCALE``）を求める。待ち判定は再実装しない。手牌は13枚相当
  （``len(concealed) + 3 * len(melds) == 13``）でなければならない

対象範囲（``SCOPE``）: 観測者の打牌判断すべて（合法手に打牌を1つ以上含む判断）。
リーチの有無、選んだ行動の種類では絞らない。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lisjong.belief.canonical_axes import red_five_index, tile_type_index
from lisjong.belief.exact_wait_ground_truth import exact_hand_belief_with_waits
from lisjong.belief.hand_belief import HandBelief
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.learning._canonical import (
    canonical_json_line,
    canonical_json_text,
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
from lisjong.learning.riichi_deal_in_source import DecisionKey
from lisjong.policy_contract.action import DiscardAction, InternalAction
from lisjong.policy_contract.meld import PublicMeld
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

MANIFEST_SCHEMA = "lisjong-hand-belief-source-manifest-v1"
DECISION_SCHEMA = "lisjong-hand-belief-decision-record-v1"
HAND_FACT_SCHEMA = "lisjong-hand-belief-hand-fact-record-v1"
SCOPE = "all-observer-discard-decisions.v1"

MANIFEST_FILENAME = "manifest.json"
DECISIONS_FILENAME = "decisions.jsonl"
HAND_FACTS_FILENAME = "hand_facts.jsonl"
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
_FACT_FIELDS = frozenset({"schema", "key", "opponents"})
_HAND_FIELDS = frozenset({"seat", "sequence", "concealed_tiles", "melds"})
_SEAT_COUNT = 4
_SUITED_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)


class HandBeliefSourceError(ValueError):
    """入力契約に合わない、または事実の間に矛盾がある。"""


_E = HandBeliefSourceError


@dataclass(frozen=True, slots=True)
class HandBeliefDecision:
    """player-safeな判断記録（推論入力の元）。"""

    key: DecisionKey
    policy_input: PolicyInput
    legal_actions: tuple[InternalAction, ...]
    selected_action: InternalAction


@dataclass(frozen=True, slots=True)
class OpponentHand:
    """他家1席の、``sequence`` 時点の手牌（学習専用）。"""

    seat: int
    sequence: int
    concealed_tiles: tuple[Tile, ...]
    melds: tuple[PublicMeld, ...]


@dataclass(frozen=True, slots=True)
class HandBeliefHandFacts:
    """学習専用の観測事実。他家3席の手牌を席順に持つ。"""

    key: DecisionKey
    opponents: tuple[OpponentHand, ...]


@dataclass(frozen=True, slots=True)
class OpponentTruth:
    """他家1席の正解。``truth`` はLevel 2のexact ``HandBelief``（値は0か``SCALE``）。"""

    seat: int
    truth: HandBelief


@dataclass(frozen=True, slots=True)
class HandBeliefLabelledDecision:
    """ラベル付きの判断。``opponents`` は観測者以外の3席を席順に持つ。"""

    decision: HandBeliefDecision
    opponents: tuple[OpponentTruth, ...]


@dataclass(frozen=True, slots=True)
class HandBeliefManifest:
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


def decision_to_value(decision: HandBeliefDecision) -> dict[str, object]:
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


def hand_facts_to_value(facts: HandBeliefHandFacts) -> dict[str, object]:
    return {
        "key": _key_to_value(facts.key),
        "opponents": [
            {
                "concealed_tiles": tiles_to_value(
                    hand.concealed_tiles, _E, f"opponents[{index}].concealed_tiles"
                ),
                "melds": [
                    meld_to_value(meld, _E, f"opponents[{index}].melds[{position}]")
                    for position, meld in enumerate(hand.melds)
                ],
                "seat": hand.seat,
                "sequence": hand.sequence,
            }
            for index, hand in enumerate(facts.opponents)
        ],
        "schema": HAND_FACT_SCHEMA,
    }


# --- strict readers -----------------------------------------------------------


def _parse_key(value: object, context: str) -> DecisionKey:
    raw = expect_object(value, _KEY_FIELDS, _E, context)
    return DecisionKey(
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


def _check_decision(decision: HandBeliefDecision, context: str) -> None:
    if int(decision.policy_input.self_seat) != decision.key.seat:
        raise _E(f"{context}.key.seat does not match the decider")
    if decision.selected_action not in decision.legal_actions:
        raise _E(f"{context}.selected is not a legal action")
    if not any(isinstance(a, DiscardAction) for a in decision.legal_actions):
        raise _E(f"{context} is outside the scope: no legal discard")


def _parse_decision(value: object, context: str) -> HandBeliefDecision:
    raw = expect_object(value, _DECISION_FIELDS, _E, context)
    if raw["schema"] != DECISION_SCHEMA:
        raise _E(f"{context} has an unsupported schema")
    decision = HandBeliefDecision(
        key=_parse_key(raw["key"], f"{context}.key"),
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


def _parse_facts(value: object, context: str) -> HandBeliefHandFacts:
    raw = expect_object(value, _FACT_FIELDS, _E, context)
    if raw["schema"] != HAND_FACT_SCHEMA:
        raise _E(f"{context} has an unsupported schema")
    opponents = []
    for index, item in enumerate(
        expect_list(raw["opponents"], _E, f"{context}.opponents")
    ):
        hand_context = f"{context}.opponents[{index}]"
        hand = expect_object(item, _HAND_FIELDS, _E, hand_context)
        opponents.append(
            OpponentHand(
                seat=int(_parse_seat(hand["seat"], _E, f"{hand_context}.seat")),
                sequence=expect_non_negative_int(
                    hand["sequence"], _E, f"{hand_context}.sequence"
                ),
                concealed_tiles=parse_tiles(
                    hand["concealed_tiles"], _E, f"{hand_context}.concealed_tiles"
                ),
                melds=tuple(
                    parse_meld(meld, _E, f"{hand_context}.melds[{position}]")
                    for position, meld in enumerate(
                        expect_list(hand["melds"], _E, f"{hand_context}.melds")
                    )
                ),
            )
        )
    facts = HandBeliefHandFacts(
        key=_parse_key(raw["key"], f"{context}.key"), opponents=tuple(opponents)
    )
    if hand_facts_to_value(facts) != value:
        raise _E(f"{context} does not round-trip through the canonical projection")
    return facts


def _unique_by_key(items, name: str) -> dict[DecisionKey, object]:
    result: dict[DecisionKey, object] = {}
    for item in items:
        if item.key in result:
            raise _E(f"duplicate {name} key {item.key}")
        result[item.key] = item
    return result


def read_manifest(directory: str | Path) -> HandBeliefManifest:
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
        raw["files"], frozenset({"decisions", "hand_facts"}), _E, "files"
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
    return HandBeliefManifest(producer=dict(producer), splits=splits, files=files)


def _check_file(root: Path, filename: str, entry: dict[str, object]) -> None:
    digest = file_digest(root / filename)
    if digest != {"bytes": entry["bytes"], "sha256": entry["sha256"]}:
        raise _E(f"{filename} does not match the manifest digest")


def read_decisions(
    directory: str | Path,
) -> tuple[HandBeliefManifest, tuple[HandBeliefDecision, ...]]:
    """player-safeな判断記録だけを読む（推論入力の経路）。手牌の記録は開かない。"""
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


def _check_conservation(
    decision: HandBeliefDecision, opponents: Sequence[OpponentHand], context: str
) -> None:
    try:
        remaining = derive_remaining_tile_inventory(decision.policy_input)
    except ValueError as error:
        raise _E(f"{context}: the public state is not tile-consistent") from error
    tile_counts = [0] * len(remaining.remaining_tile_counts)
    red_counts = [0] * len(remaining.remaining_red_five_counts)
    for hand in opponents:
        for tile in hand.concealed_tiles:
            tile_counts[tile_type_index(tile.tile_type)] += 1
            if tile.is_red:
                red_counts[red_five_index(tile.tile_type.category)] += 1
    over = any(
        count > limit
        for count, limit in zip(tile_counts, remaining.remaining_tile_counts)
    ) or any(
        count > limit
        for count, limit in zip(red_counts, remaining.remaining_red_five_counts)
    )
    # 5は「総数」と「赤」だけでなく「通常5（総数 - 赤）」も残数以下でなければならない
    for category in _SUITED_CATEGORIES:
        five = tile_type_index(TileType(category, 5))
        red = red_five_index(category)
        normal = tile_counts[five] - red_counts[red]
        remaining_normal = (
            remaining.remaining_tile_counts[five]
            - remaining.remaining_red_five_counts[red]
        )
        over = over or normal > remaining_normal
    if over:
        raise _E(f"{context}: opponents' hands exceed the unseen tiles")


def label_decisions(
    decisions: Sequence[HandBeliefDecision],
    facts: Sequence[HandBeliefHandFacts],
) -> tuple[HandBeliefLabelledDecision, ...]:
    """手牌の記録から他家3席の正解``HandBelief``を計算し、判断記録と結合する（学習専用）。"""
    by_decision = _unique_by_key(decisions, "decision")
    by_facts = _unique_by_key(facts, "hand fact")
    if set(by_decision) != set(by_facts):
        raise _E("decision and hand fact keys do not match")
    labelled = []
    for key in sorted(by_decision):
        decision, fact = by_decision[key], by_facts[key]
        context = f"decision {key}"
        expected_seats = [seat for seat in range(_SEAT_COUNT) if seat != key.seat]
        if [hand.seat for hand in fact.opponents] != expected_seats:
            raise _E(f"{context}: hand facts must list the three opponents in order")
        truths = []
        for hand in fact.opponents:
            hand_context = f"{context} seat {hand.seat}"
            if hand.sequence != key.sequence:
                raise _E(f"{hand_context}: the hand is not the decision-point snapshot")
            public = decision.policy_input.players[hand.seat]
            if tuple(hand.melds) != tuple(public.melds):
                raise _E(f"{hand_context}: melds differ from the public state")
            try:
                truth = exact_hand_belief_with_waits(hand.concealed_tiles, hand.melds)
            except ValueError as error:
                raise _E(f"{hand_context}: {error}") from error
            truths.append(OpponentTruth(seat=hand.seat, truth=truth))
        _check_conservation(decision, fact.opponents, context)
        labelled.append(
            HandBeliefLabelledDecision(decision=decision, opponents=tuple(truths))
        )
    return tuple(labelled)


def read_labelled_source(
    directory: str | Path,
) -> tuple[HandBeliefManifest, tuple[HandBeliefLabelledDecision, ...]]:
    """判断記録と手牌の記録を読み、ラベル付きの判断を返す（学習専用の経路）。"""
    root = Path(directory)
    manifest, decisions = read_decisions(root)
    _check_file(root, HAND_FACTS_FILENAME, manifest.files["hand_facts"])
    facts = tuple(
        _parse_facts(value, f"{HAND_FACTS_FILENAME}:{number}")
        for number, value in _read_lines(root / HAND_FACTS_FILENAME)
    )
    if len(facts) != manifest.files["hand_facts"]["rows"]:
        raise _E("hand fact row count does not match the manifest")
    return manifest, label_decisions(decisions, facts)


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
    "HAND_FACT_SCHEMA",
    "HAND_FACTS_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "SCOPE",
    "SPLITS",
    "DecisionKey",
    "HandBeliefDecision",
    "HandBeliefHandFacts",
    "HandBeliefLabelledDecision",
    "HandBeliefManifest",
    "HandBeliefSourceError",
    "OpponentHand",
    "OpponentTruth",
    "decision_to_value",
    "hand_facts_to_value",
    "label_decisions",
    "manifest_text",
    "read_decisions",
    "read_labelled_source",
    "read_manifest",
]
