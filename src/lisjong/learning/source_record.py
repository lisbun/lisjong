"""Arena-produced player-safe source recordのstrict consumer。

Issue #184のL0aに対応する。`lisjong-arena`が生成するversioned player-safe
source record（lisbun/lisjong-arena#342）をcanonical inputとして読み、
lisjong側のfeature / datasetをmaterializeできる形へ復元する。

```text
Arena-owned versioned source record
    -> strict schema / identity / provenance validation
    -> typed PolicyInput + canonical legal actions + selected action
```

固定する境界。

- source recordのschemaはproducerである`lisjong-arena`が所有するversioned
  contractである。未対応schemaは推測でacceptせずfail closedする
- Arenaのencoded feature tensorをcanonical inputとして使わない。本moduleは
  typed player-safe rowだけを読む
- corpus（`rows.jsonl` / `features.f32`等のArena scientific artifact）は読まない。
  source recordとlocked corpusのcross-checkはArena側の責務であり、lisjongが
  Arena artifact semanticsを再計算・再検証しない
- source population / hanchan / decision provenanceは捨てずに保持し、下流の
  dataset identityへbindできる形で返す
- `manifest.json`の`source_contract`（Arena revision、installed package
  identity、historical teacher / feature / vocabulary fingerprint等のArena-owned
  qualification binding）は意味を再解釈せず、digestとしてのみ保持する。
  historical Arena identityをlisjong側のcanonical identityへ持ち上げない
- schema v2の`allocation_bindings`（Arena-owned seed allocation ledger、
  lisbun/lisjong-arena#346/#347のper-split binding）はshapeをstrict validateし、
  値をそのまま保持して下流へbindする。lisjongはArena allocation ledgerを
  再生成・再所有せず、live registryへも再照会しない
- schema v1（historical、allocation provenanceを持たない）は引き続き
  historical readback用途として読める。v1をv2相当へ推測で補完しない

読み取り時にfail closedする代表的条件。

- 未対応 / 不整合なschema、kind、identity
- canonical JSONでないmanifest / row
- manifestとpayloadのbyte長 / SHA-256不一致
- 欠損 / 余剰のgame directory、payload file、manifest field
- game provenance（ordinal / seed / split / lock identity / game mode）不一致
- population内のseed重複（split間leakageを含む）
- decision orderingやdecision countの不整合
- actor seatとPolicyInput / legal actions / selected actionの不一致
- legal actionsの重複、canonical順序違反、空
- legal actionsに含まれないselected action
- （v2のみ）allocation bindingのfield shape不正、owner_repository不一致、
  splitの欠損 / 余剰、source population内のseed membershipとの矛盾
"""

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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
    unseal,
    value_digest,
)
from lisjong.learning._typed_values import (
    action_to_value,
    parse_action,
    parse_policy_input,
    policy_input_to_value,
)
from lisjong.learning.errors import SourceRecordError, UnsupportedSourceSchemaError
from lisjong.policy_contract import InternalAction, PolicyInput, Seat

SOURCE_RECORD_SCHEMA_V1 = "arena-offense-o0-player-safe-source-record-v1"
"""Historical schema（Arena-owned contract）。allocation provenanceを持たない。"""

SOURCE_RECORD_SCHEMA_V2 = "arena-offense-o0-player-safe-source-record-v2"
"""現在consumeするschema identity（Arena-owned contract、lisjong-arena#347）。

v1へ`allocation_bindings`（Arena-owned seed allocation ledgerのper-split
binding）を追加したものであり、v1 identityの意味は変更しない。
"""

SUPPORTED_SOURCE_RECORD_SCHEMAS = frozenset(
    {SOURCE_RECORD_SCHEMA_V1, SOURCE_RECORD_SCHEMA_V2}
)
"""対応schemaの集合。ここに無いschemaはfail closedとする。"""

SOURCE_RECORD_KIND = "player-safe-source-record"
MANIFEST_FILENAME = "manifest.json"
GAME_PAYLOAD_FILENAME = "source-record.jsonl"

EXPECTED_ALLOCATION_OWNER_REPOSITORY = "lisbun/lisjong-arena"
"""allocation bindingが指すcanonical Arena owner repository。

lisjongはallocation ledgerを所有しないため、この値はlisjong側が新たに
定義するidentityではなく、Arena-owned `seed_registry.OWNER_REPOSITORY`と
一致するべき固定値として検証する。
"""

_ALLOCATION_BINDING_FIELDS = frozenset(
    {
        "allocation_identity",
        "ledger_revision",
        "owner_repository",
        "seed_domain",
        "seed_membership_identity",
    }
)
_SEED_DOMAIN_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

_MANIFEST_FIELDS_V1 = frozenset(
    {
        "schema",
        "kind",
        "lock_identity",
        "scientific_corpus_identity",
        "game_mode",
        "source_contract",
        "games",
    }
)
_MANIFEST_FIELDS_V2 = _MANIFEST_FIELDS_V1 | {"allocation_bindings"}
_GAME_FIELDS = frozenset(
    {
        "game_ordinal",
        "seed",
        "split",
        "lock_identity",
        "game_mode",
        "decision_count",
        "steps",
        "files",
    }
)
_ROW_FIELDS = frozenset(
    {
        "game_ordinal",
        "seed",
        "split",
        "step_ordinal",
        "decision_ordinal",
        "actor_seat",
        "policy_input",
        "legal_actions",
        "teacher_selected_action",
    }
)
_PAYLOAD_FIELDS = frozenset({"bytes", "sha256"})


def _seed_membership_document(seeds: Sequence[int]) -> dict[str, object]:
    """Arena `seed_registry.seed_membership_document()`と同じcanonical形。

    lisjongはArena allocation ledgerを再実装・再所有しないが、この
    小さく安定したwire-level encoding（連続range or explicit sorted list）
    だけは、source populationの実際のseed集合とbindingの
    `seed_membership_identity`が矛盾していないかを検証するために、
    ここで独立に再現する。ledgerそのものへは一切アクセスしない。
    """
    ordered = tuple(sorted(seeds))
    if not ordered:
        raise SourceRecordError("allocation binding seed membership must not be empty")
    if ordered == tuple(range(ordered[0], ordered[-1] + 1)):
        return {"first": ordered[0], "kind": "range", "last": ordered[-1]}
    return {"kind": "explicit", "seeds": list(ordered)}


def seed_membership_identity(seeds: Sequence[int]) -> str:
    """splitのseed集合から、Arena `seed_registry.seed_membership_identity()`と
    byte-for-byte一致するidentityを計算する。

    productionのvalidationとtest fixtureの両方が、同じ1つの実装からこの
    identityを得るための公開関数である（Arena本体と同様の構成）。
    """
    text = json.dumps(
        _seed_membership_document(seeds),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_allocation_binding(
    value: object, *, seeds: Sequence[int], context: str
) -> dict[str, str]:
    """1 splitぶんのArena allocation bindingをstrict validateする。

    liveなseed ledgerへは一切照会しない。shape（field集合、SHA-256形式、
    canonical owner、seed_domainの形式）と、このsplitの実際のsource
    populationから独立に再計算した`seed_membership_identity`との一致だけを
    検証する。
    """
    binding = expect_object(
        value, _ALLOCATION_BINDING_FIELDS, SourceRecordError, context
    )
    for field in ("allocation_identity", "ledger_revision", "seed_membership_identity"):
        expect_digest(binding[field], SourceRecordError, f"{context}.{field}")
    owner = expect_str(
        binding["owner_repository"], SourceRecordError, f"{context}.owner_repository"
    )
    if owner != EXPECTED_ALLOCATION_OWNER_REPOSITORY:
        raise SourceRecordError(
            f"{context}.owner_repository is not the canonical Arena owner"
        )
    domain = expect_str(
        binding["seed_domain"], SourceRecordError, f"{context}.seed_domain"
    )
    if not _SEED_DOMAIN_PATTERN.fullmatch(domain):
        raise SourceRecordError(f"{context}.seed_domain has an invalid format")
    if binding["seed_membership_identity"] != seed_membership_identity(seeds):
        raise SourceRecordError(
            f"{context}.seed_membership_identity contradicts the source population"
        )
    return binding


def validate_allocation_bindings(
    value: object, *, populations: Mapping[str, Sequence[int]], context: str
) -> dict[str, dict[str, str]]:
    """splitごとのArena allocation bindingをまとめてstrict validateする。

    `populations`のsplit集合と完全一致しないbinding集合（欠損・余剰）は
    fail closedとする。
    """
    if type(value) is not dict or set(value) != set(populations):
        raise SourceRecordError(f"{context} do not match the source population splits")
    return {
        split: validate_allocation_binding(
            value[split], seeds=seeds, context=f"{context}[{split}]"
        )
        for split, seeds in populations.items()
    }


@dataclass(frozen=True, slots=True)
class SourceDecision:
    """1 decision分のplayer-safe source row。

    `policy_input`は当該seatから観測可能な状態だけを持つ。`legal_actions`は
    そのdecisionのcanonical legal候補であり、`selected_action`はsource
    teacherが実際に選択したactionである。いずれもAI-side analysis、reward、
    privileged truthを含まない。
    """

    game_ordinal: int
    seed: int
    split: str
    step_ordinal: int
    decision_ordinal: int
    actor_seat: Seat
    policy_input: PolicyInput
    legal_actions: tuple[InternalAction, ...]
    selected_action: InternalAction


@dataclass(frozen=True, slots=True)
class SourceGame:
    """1 hanchan分のsource provenanceとdecision列。"""

    game_ordinal: int
    seed: int
    split: str
    step_count: int
    decisions: tuple[SourceDecision, ...]


@dataclass(frozen=True, slots=True)
class PlayerSafeSourceRecord:
    """strict readした完全なsource recordと、そのArena-owned provenance。

    `allocation_bindings`はschema v2だけが持つ、split別のArena
    seed-allocation binding（`allocation_identity` / `ledger_revision` /
    `owner_repository` / `seed_domain` / `seed_membership_identity`）である。
    historical schema v1はこのprovenanceを持たないため`None`となり、
    v2相当の値を推測で補わない。
    """

    schema: str
    identity: str
    lock_identity: str
    scientific_corpus_identity: str
    game_mode: str
    source_contract_digest: str
    allocation_bindings: Mapping[str, Mapping[str, str]] | None
    games: tuple[SourceGame, ...]

    @property
    def decision_count(self) -> int:
        return sum(len(game.decisions) for game in self.games)

    def decisions(self) -> Iterator[SourceDecision]:
        """population順（game順、game内はdecision_ordinal順）に列挙する。"""
        for game in self.games:
            yield from game.decisions

    def population(self) -> tuple[dict[str, object], ...]:
        """ordered source populationのcanonical記述を返す。"""
        return tuple(
            {
                "decision_count": len(game.decisions),
                "game_ordinal": game.game_ordinal,
                "seed": game.seed,
                "split": game.split,
            }
            for game in self.games
        )

    def provenance(self) -> dict[str, object]:
        """下流artifactへbindするsource identity / population記述を返す。

        `allocation_bindings`はv2なら`{split: binding}`のJSON化可能なdict、
        v1なら`None`である。呼び出し側（`materialize_dataset()`）が
        `None`をどう扱うかを決め、この関数自体は値を補完しない。
        """
        return {
            "allocation_bindings": None
            if self.allocation_bindings is None
            else {
                split: dict(binding)
                for split, binding in self.allocation_bindings.items()
            },
            "decision_count": self.decision_count,
            "game_mode": self.game_mode,
            "identity": self.identity,
            "lock_identity": self.lock_identity,
            "population": list(self.population()),
            "schema": self.schema,
            "scientific_corpus_identity": self.scientific_corpus_identity,
            "source_contract_digest": self.source_contract_digest,
        }


def _read_manifest(path: Path) -> dict[str, object]:
    manifest_path = path / MANIFEST_FILENAME
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SourceRecordError(
            f"source record manifest cannot be read: {manifest_path}"
        ) from exc
    manifest = parse_json_text(text, SourceRecordError, "source record manifest")
    body = unseal(manifest, SourceRecordError, "source record manifest")
    if text != canonical_json_text(manifest):
        raise SourceRecordError("source record manifest is not canonical JSON")

    schema = body.get("schema")
    if schema not in SUPPORTED_SOURCE_RECORD_SCHEMAS:
        raise UnsupportedSourceSchemaError(
            f"unsupported source record schema: {schema!r}; "
            f"this implementation consumes {sorted(SUPPORTED_SOURCE_RECORD_SCHEMAS)}"
        )
    expected_fields = (
        _MANIFEST_FIELDS_V2
        if schema == SOURCE_RECORD_SCHEMA_V2
        else _MANIFEST_FIELDS_V1
    )
    expect_object(body, expected_fields, SourceRecordError, "source record manifest")

    if body["kind"] != SOURCE_RECORD_KIND:
        raise SourceRecordError("source record kind mismatch")
    if type(body["source_contract"]) is not dict:
        raise SourceRecordError("manifest.source_contract must be a JSON object")
    expect_str(body["lock_identity"], SourceRecordError, "manifest.lock_identity")
    expect_str(
        body["scientific_corpus_identity"],
        SourceRecordError,
        "manifest.scientific_corpus_identity",
    )
    expect_str(body["game_mode"], SourceRecordError, "manifest.game_mode")
    games = expect_list(body["games"], SourceRecordError, "manifest.games")
    if not games:
        raise SourceRecordError("source record must contain at least one hanchan")
    return manifest


def _read_game_summary(
    value: object, *, game_ordinal: int, manifest: dict[str, object]
) -> dict[str, object]:
    context = f"manifest.games[{game_ordinal}]"
    body = unseal(value, SourceRecordError, context)
    expect_object(body, _GAME_FIELDS, SourceRecordError, context)
    for field in ("game_ordinal", "seed", "decision_count", "steps"):
        expect_non_negative_int(body[field], SourceRecordError, f"{context}.{field}")
    expect_str(body["split"], SourceRecordError, f"{context}.split")
    if body["game_ordinal"] != game_ordinal:
        raise SourceRecordError(f"{context} game ordinal is not contiguous")
    if (
        body["lock_identity"] != manifest["lock_identity"]
        or body["game_mode"] != manifest["game_mode"]
    ):
        raise SourceRecordError(f"{context} provenance does not match the manifest")
    files = expect_object(
        body["files"],
        frozenset({GAME_PAYLOAD_FILENAME}),
        SourceRecordError,
        f"{context}.files",
    )
    payload = expect_object(
        files[GAME_PAYLOAD_FILENAME],
        _PAYLOAD_FIELDS,
        SourceRecordError,
        f"{context}.files[{GAME_PAYLOAD_FILENAME}]",
    )
    expect_non_negative_int(
        payload["bytes"], SourceRecordError, f"{context}.files.bytes"
    )
    expect_digest(payload["sha256"], SourceRecordError, f"{context}.files.sha256")
    return body


def _read_row(line: str, *, context: str) -> tuple[dict[str, object], SourceDecision]:
    row = parse_json_text(line, SourceRecordError, context)
    if line != canonical_json_line(row):
        raise SourceRecordError(f"{context} is not canonical JSON")
    expect_object(row, _ROW_FIELDS, SourceRecordError, context)
    for field in (
        "game_ordinal",
        "seed",
        "step_ordinal",
        "decision_ordinal",
        "actor_seat",
    ):
        expect_non_negative_int(row[field], SourceRecordError, f"{context}.{field}")
    expect_str(row["split"], SourceRecordError, f"{context}.split")

    policy_input = parse_policy_input(
        row["policy_input"], SourceRecordError, f"{context}.policy_input"
    )
    if (
        policy_input_to_value(
            policy_input, SourceRecordError, f"{context}.policy_input"
        )
        != row["policy_input"]
    ):
        raise SourceRecordError(f"{context}.policy_input does not round trip")

    legal_values = expect_list(
        row["legal_actions"], SourceRecordError, f"{context}.legal_actions"
    )
    if not legal_values:
        raise SourceRecordError(f"{context}.legal_actions must not be empty")
    legal_actions = tuple(
        parse_action(value, SourceRecordError, f"{context}.legal_actions[{index}]")
        for index, value in enumerate(legal_values)
    )
    roundtripped = [
        action_to_value(action, SourceRecordError, f"{context}.legal_actions[{index}]")
        for index, action in enumerate(legal_actions)
    ]
    if roundtripped != legal_values:
        raise SourceRecordError(f"{context}.legal_actions does not round trip")
    canonical_lines = [canonical_json_line(value) for value in legal_values]
    if canonical_lines != sorted(canonical_lines):
        raise SourceRecordError(f"{context}.legal_actions is not canonically ordered")
    if len(set(canonical_lines)) != len(canonical_lines) or len(
        set(legal_actions)
    ) != len(legal_actions):
        raise SourceRecordError(f"{context}.legal_actions contains duplicates")

    selected = parse_action(
        row["teacher_selected_action"],
        SourceRecordError,
        f"{context}.teacher_selected_action",
    )
    if (
        action_to_value(
            selected, SourceRecordError, f"{context}.teacher_selected_action"
        )
        != row["teacher_selected_action"]
    ):
        raise SourceRecordError(
            f"{context}.teacher_selected_action does not round trip"
        )

    actor_seat = row["actor_seat"]
    if int(policy_input.self_seat) != actor_seat:
        raise SourceRecordError(f"{context} PolicyInput seat does not match the actor")
    if any(int(action.actor) != actor_seat for action in legal_actions):
        raise SourceRecordError(f"{context} legal action actor does not match")
    if int(selected.actor) != actor_seat:
        raise SourceRecordError(f"{context} selected action actor does not match")
    if selected not in legal_actions:
        raise SourceRecordError(f"{context} selected action is not legal")

    decision = SourceDecision(
        game_ordinal=row["game_ordinal"],
        seed=row["seed"],
        split=row["split"],
        step_ordinal=row["step_ordinal"],
        decision_ordinal=row["decision_ordinal"],
        actor_seat=policy_input.self_seat,
        policy_input=policy_input,
        legal_actions=legal_actions,
        selected_action=selected,
    )
    return row, decision


def _read_game(path: Path, summary: dict[str, object]) -> SourceGame:
    game_ordinal = summary["game_ordinal"]
    context = f"game-{game_ordinal:03d}"
    if not path.is_dir():
        raise SourceRecordError(f"missing source record game directory: {context}")
    if {child.name for child in path.iterdir()} != {GAME_PAYLOAD_FILENAME}:
        raise SourceRecordError(f"missing/unexpected payload files in {context}")
    payload = path / GAME_PAYLOAD_FILENAME
    if file_digest(payload) != summary["files"][GAME_PAYLOAD_FILENAME]:
        raise SourceRecordError(f"{context} payload size/digest mismatch")

    decisions: list[SourceDecision] = []
    last_step = -1
    last_seat = -1
    with payload.open(encoding="utf-8", newline="\n") as stream:
        for index, line in enumerate(stream):
            row, decision = _read_row(line, context=f"{context}.rows[{index}]")
            if (
                decision.game_ordinal != game_ordinal
                or decision.seed != summary["seed"]
                or decision.split != summary["split"]
            ):
                raise SourceRecordError(
                    f"{context}.rows[{index}] provenance does not match the manifest"
                )
            if decision.decision_ordinal != index:
                raise SourceRecordError(
                    f"{context}.rows[{index}] decision ordinal is not contiguous"
                )
            step = decision.step_ordinal
            seat = int(decision.actor_seat)
            if step not in (last_step, last_step + 1) or (
                step == last_step and seat <= last_seat
            ):
                raise SourceRecordError(
                    f"{context}.rows[{index}] execution ordering mismatch"
                )
            last_step, last_seat = step, seat
            decisions.append(decision)

    if len(decisions) != summary["decision_count"]:
        raise SourceRecordError(f"{context} decision count mismatch")
    if last_step + 1 != summary["steps"]:
        raise SourceRecordError(f"{context} step accounting mismatch")
    return SourceGame(
        game_ordinal=game_ordinal,
        seed=summary["seed"],
        split=summary["split"],
        step_count=summary["steps"],
        decisions=tuple(decisions),
    )


def read_source_record(path: str | Path) -> PlayerSafeSourceRecord:
    """source record directoryをstrict readし、検証済みrecordを返す。

    validationは全体としてfail closedである。1 rowでも不整合があれば、
    成功分だけを返さず例外を送出する。
    """
    root = Path(path)
    if not root.is_dir():
        raise SourceRecordError(f"source record directory does not exist: {root}")

    manifest = _read_manifest(root)
    summaries = [
        _read_game_summary(value, game_ordinal=ordinal, manifest=manifest)
        for ordinal, value in enumerate(manifest["games"])
    ]
    expected_names = {MANIFEST_FILENAME} | {
        f"game-{ordinal:03d}" for ordinal in range(len(summaries))
    }
    if {child.name for child in root.iterdir()} != expected_names:
        raise SourceRecordError("missing/unexpected source record files")

    seeds: set[int] = set()
    seeds_by_split: dict[str, list[int]] = {}
    games: list[SourceGame] = []
    for summary in summaries:
        if summary["seed"] in seeds:
            raise SourceRecordError(
                "source population must not reuse a seed across hanchan/splits"
            )
        seeds.add(summary["seed"])
        seeds_by_split.setdefault(summary["split"], []).append(summary["seed"])
        games.append(_read_game(root / f"game-{summary['game_ordinal']:03d}", summary))

    allocation_bindings: dict[str, dict[str, str]] | None = None
    if manifest["schema"] == SOURCE_RECORD_SCHEMA_V2:
        allocation_bindings = validate_allocation_bindings(
            manifest["allocation_bindings"],
            populations=seeds_by_split,
            context="manifest.allocation_bindings",
        )

    return PlayerSafeSourceRecord(
        schema=manifest["schema"],
        identity=manifest["identity"],
        lock_identity=manifest["lock_identity"],
        scientific_corpus_identity=manifest["scientific_corpus_identity"],
        game_mode=manifest["game_mode"],
        source_contract_digest=value_digest(manifest["source_contract"]),
        allocation_bindings=allocation_bindings,
        games=tuple(games),
    )


__all__ = [
    "EXPECTED_ALLOCATION_OWNER_REPOSITORY",
    "GAME_PAYLOAD_FILENAME",
    "MANIFEST_FILENAME",
    "SOURCE_RECORD_KIND",
    "SOURCE_RECORD_SCHEMA_V1",
    "SOURCE_RECORD_SCHEMA_V2",
    "SUPPORTED_SOURCE_RECORD_SCHEMAS",
    "PlayerSafeSourceRecord",
    "SourceDecision",
    "SourceGame",
    "read_source_record",
    "seed_membership_identity",
    "validate_allocation_binding",
    "validate_allocation_bindings",
]
