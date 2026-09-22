"""Learning L0 testで共有するsource record fixture builder。

`lisjong-arena`が生成する`arena-offense-o0-player-safe-source-record-v2`
artifactと同じwire形式・canonical JSON・sealing規則でfixtureを組み立てる
（`schema=SOURCE_RECORD_SCHEMA_V1`を明示すればhistorical v1も書ける）。
production側のconsumerは読み取りだけを所有するため、writerはtest側に置く。

fail closed testのために、manifestやrow payloadを意図的に壊すhelperも提供する。
"""

import json
from pathlib import Path

from lisjong.learning._canonical import (
    canonical_json_line,
    canonical_json_text,
    file_digest,
    seal,
    value_digest,
)
from lisjong.learning._typed_values import action_to_value, policy_input_to_value
from lisjong.learning.source_record import (
    EXPECTED_ALLOCATION_OWNER_REPOSITORY,
    GAME_PAYLOAD_FILENAME,
    MANIFEST_FILENAME,
    SOURCE_RECORD_KIND,
    SOURCE_RECORD_SCHEMA_V2,
    seed_membership_identity,
)
from lisjong.policy_contract import (
    Discard,
    DiscardAction,
    OwnHandState,
    PlayerPublicState,
    PolicyInput,
    RiichiAction,
    RiichiState,
    RoundState,
    Seat,
    Tile,
    TileCategory,
    TileType,
    Wind,
)

GAME_MODE = "4p-red-half"
LOCK_IDENTITY = "a" * 64
CORPUS_IDENTITY = "b" * 64
SOURCE_CONTRACT = {"arena_revision": "0" * 40, "teacher": "example-teacher"}
ALLOCATION_SEED_DOMAIN = "riichienv-4p-red-half-hanchan-v1"

MANZU = TileCategory.MANZU
PINZU = TileCategory.PINZU
SOUZU = TileCategory.SOUZU
HONOR = TileCategory.HONOR


def tile(category, rank, *, red=False):
    return Tile(TileType(category, rank), is_red=red)


def hand(seat_offset=0):
    """seatごとに牌種が重複しない13枚の手牌を返す（tile conservationを保つ）。"""
    suits = (MANZU, PINZU, SOUZU)
    category = suits[seat_offset % 3]
    other = suits[(seat_offset + 1) % 3]
    return (
        tile(category, 1),
        tile(category, 2),
        tile(category, 3),
        tile(category, 4),
        tile(category, 5),
        tile(category, 6),
        tile(category, 7),
        tile(category, 8),
        tile(category, 9),
        tile(other, 1 + seat_offset),
        tile(other, 2 + seat_offset),
        tile(other, 3 + seat_offset),
        tile(HONOR, 1 + (seat_offset % 7)),
    )


def player_state(*, score=25000, discards=(), melds=(), riichi=RiichiState.NONE):
    return PlayerPublicState(
        score=score, discards=tuple(discards), melds=tuple(melds), riichi=riichi
    )


def discard(tile_value, order, *, tsumogiri=False, called_by=None):
    return Discard(
        tile=tile_value, tsumogiri=tsumogiri, order=order, called_by=called_by
    )


def round_state(**overrides):
    values = {
        "round_wind": Wind.EAST,
        "hand_number": 1,
        "dealer_seat": Seat.SEAT_0,
        "honba": 0,
        "riichi_sticks": 0,
        "dora_indicators": (tile(HONOR, 4),),
        "live_wall_tiles_remaining": 60,
    }
    values.update(overrides)
    return RoundState(**values)


def policy_input(*, self_seat=Seat.SEAT_0, concealed=None, drawn=None, **overrides):
    """1 decision分のplayer-safe PolicyInputを組み立てる。"""
    seat = Seat(self_seat)
    concealed = hand(int(seat)) if concealed is None else tuple(concealed)
    if drawn is None:
        drawn = concealed[0]
    elif drawn is False:
        drawn = None
    players = overrides.pop(
        "players",
        tuple(
            player_state(
                discards=(discard(tile(HONOR, 5 + index % 3), index),),
                riichi=RiichiState.NONE,
            )
            for index in range(4)
        ),
    )
    return PolicyInput(
        self_seat=seat,
        round=overrides.pop("round", round_state(**overrides)),
        players=players,
        own_hand=OwnHandState(concealed_tiles=concealed, drawn_tile=drawn),
    )


def discard_legal_actions(value, *, with_riichi=False):
    """concealed手牌から重複しないDiscard候補（+任意のRiichi）を作る。"""
    seat = value.self_seat
    actions = [
        DiscardAction(actor=seat, tile=item, tsumogiri=False)
        for item in dict.fromkeys(value.own_hand.concealed_tiles)
    ]
    if value.own_hand.drawn_tile is not None:
        actions.append(
            DiscardAction(actor=seat, tile=value.own_hand.drawn_tile, tsumogiri=True)
        )
    if with_riichi:
        actions.append(RiichiAction(actor=seat))
    return tuple(actions)


def source_row(
    *,
    game_ordinal,
    seed,
    split,
    step_ordinal,
    decision_ordinal,
    policy_input_value,
    legal_actions,
    selected_action,
):
    """canonical順序のlegal actionsを持つsource record rowを返す。"""
    legal = [
        action_to_value(action, ValueError, "legal_actions") for action in legal_actions
    ]
    legal.sort(key=canonical_json_line)
    return {
        "actor_seat": int(policy_input_value.self_seat),
        "decision_ordinal": decision_ordinal,
        "game_ordinal": game_ordinal,
        "legal_actions": legal,
        "policy_input": policy_input_to_value(
            policy_input_value, ValueError, "policy_input"
        ),
        "seed": seed,
        "split": split,
        "step_ordinal": step_ordinal,
        "teacher_selected_action": action_to_value(
            selected_action, ValueError, "teacher_selected_action"
        ),
    }


def decision_rows(*, game_ordinal, seed, split, count=3, seats=None):
    """1 hanchan分の連続したdecision rowを作る。"""
    seats = seats or [Seat(index % 4) for index in range(count)]
    rows = []
    for ordinal in range(count):
        seat = seats[ordinal]
        value = policy_input(self_seat=seat)
        legal = discard_legal_actions(value, with_riichi=ordinal == 0)
        rows.append(
            source_row(
                game_ordinal=game_ordinal,
                seed=seed,
                split=split,
                step_ordinal=ordinal,
                decision_ordinal=ordinal,
                policy_input_value=value,
                legal_actions=legal,
                selected_action=legal[0],
            )
        )
    return rows


def allocation_binding(
    seeds,
    *,
    allocation_identity=None,
    ledger_revision=None,
    owner_repository=EXPECTED_ALLOCATION_OWNER_REPOSITORY,
    seed_domain=ALLOCATION_SEED_DOMAIN,
):
    """1 splitぶんのArena allocation binding fixtureを組み立てる。

    `seed_membership_identity`はlisjongの公開実装（Arena
    `seed_registry.seed_membership_identity()`とbyte-for-byte一致）で
    計算するため、実Arena artifactと同じ検証規則をそのまま通せる。
    """
    return {
        "allocation_identity": allocation_identity or ("3" * 64),
        "ledger_revision": ledger_revision or ("4" * 64),
        "owner_repository": owner_repository,
        "seed_domain": seed_domain,
        "seed_membership_identity": seed_membership_identity(seeds),
    }


def _game_summary(*, game_ordinal, seed, split, rows, payload_path):
    steps = 0 if not rows else max(row["step_ordinal"] for row in rows) + 1
    return seal(
        {
            "decision_count": len(rows),
            "files": {GAME_PAYLOAD_FILENAME: file_digest(payload_path)},
            "game_mode": GAME_MODE,
            "game_ordinal": game_ordinal,
            "lock_identity": LOCK_IDENTITY,
            "seed": seed,
            "split": split,
            "steps": steps,
        }
    )


def write_source_record(
    root, games=None, *, schema=SOURCE_RECORD_SCHEMA_V2, **manifest_overrides
):
    """source record directoryを書き出し、そのpathを返す。

    `games`は`(split, seed, rows)`のsequenceである。省略時は2 hanchan分の
    既定populationを使う。既定`schema`は現行producerと同じv2であり、
    実populationから決定的に導出したper-split allocation bindingを含める。
    `schema=SOURCE_RECORD_SCHEMA_V1`を明示した場合はallocation_bindingsを
    持たないhistorical記録を書く。
    """
    root = Path(root)
    root.mkdir(parents=True)
    if games is None:
        games = (
            ("TRAIN", 100, decision_rows(game_ordinal=0, seed=100, split="TRAIN")),
            (
                "SELECT",
                200,
                decision_rows(game_ordinal=1, seed=200, split="SELECT", count=2),
            ),
        )

    summaries = []
    seeds_by_split = {}
    for game_ordinal, (split, seed, rows) in enumerate(games):
        game_root = root / f"game-{game_ordinal:03d}"
        game_root.mkdir()
        payload = game_root / GAME_PAYLOAD_FILENAME
        payload.write_text(
            "".join(canonical_json_line(row) for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        summaries.append(
            _game_summary(
                game_ordinal=game_ordinal,
                seed=seed,
                split=split,
                rows=rows,
                payload_path=payload,
            )
        )
        seeds_by_split.setdefault(split, []).append(seed)

    body = {
        "game_mode": GAME_MODE,
        "games": summaries,
        "kind": SOURCE_RECORD_KIND,
        "lock_identity": LOCK_IDENTITY,
        "schema": schema,
        "scientific_corpus_identity": CORPUS_IDENTITY,
        "source_contract": dict(SOURCE_CONTRACT),
    }
    if schema == SOURCE_RECORD_SCHEMA_V2:
        body["allocation_bindings"] = {
            split: allocation_binding(seeds) for split, seeds in seeds_by_split.items()
        }
    body.update(manifest_overrides)
    (root / MANIFEST_FILENAME).write_text(
        canonical_json_text(seal(body)), encoding="utf-8", newline="\n"
    )
    return root


def read_manifest_body(root):
    manifest = json.loads((Path(root) / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return {key: value for key, value in manifest.items() if key != "identity"}


def rewrite_manifest(root, body, *, identity=None):
    """manifest bodyを書き戻す。`identity`を渡すとsealを壊せる。"""
    document = dict(body)
    document["identity"] = value_digest(body) if identity is None else identity
    (Path(root) / MANIFEST_FILENAME).write_text(
        canonical_json_text(document), encoding="utf-8", newline="\n"
    )
    return root


def mutate_manifest(root, mutate, *, identity=None):
    body = read_manifest_body(root)
    mutate(body)
    return rewrite_manifest(root, body, identity=identity)


def mutate_game_summary(root, game_ordinal, mutate):
    """manifest内の1 game summaryを書き換えて再sealする。"""
    body = read_manifest_body(root)
    summary = {
        key: value
        for key, value in body["games"][game_ordinal].items()
        if key != "identity"
    }
    mutate(summary)
    body["games"][game_ordinal] = seal(summary)
    return rewrite_manifest(root, body)


def read_game_rows(root, game_ordinal=0):
    payload = Path(root) / f"game-{game_ordinal:03d}" / GAME_PAYLOAD_FILENAME
    return [
        json.loads(line)
        for line in payload.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_game_rows(root, game_ordinal, rows, *, update_manifest=True, raw=None):
    """game payloadを書き換える。manifest側のdigest更新は選択できる。"""
    root = Path(root)
    payload = root / f"game-{game_ordinal:03d}" / GAME_PAYLOAD_FILENAME
    text = raw if raw is not None else "".join(canonical_json_line(row) for row in rows)
    payload.write_text(text, encoding="utf-8", newline="\n")
    if not update_manifest:
        return root

    body = read_manifest_body(root)
    summary = {
        key: value
        for key, value in body["games"][game_ordinal].items()
        if key != "identity"
    }
    summary["decision_count"] = len(rows)
    summary["steps"] = 0 if not rows else max(row["step_ordinal"] for row in rows) + 1
    summary["files"] = {GAME_PAYLOAD_FILENAME: file_digest(payload)}
    body["games"][game_ordinal] = seal(summary)
    return rewrite_manifest(root, body)


def training_block(**overrides):
    """artifact testで使う妥当なtraining blockを返す。"""
    value = {
        "batch_size": 2,
        "diagnostics": {"epochs_run": 1, "train_rows": 3},
        "epochs": 1,
        "framework": {"name": "torch", "version": "0.0.0-test"},
        "learning_rate": 0.001,
        "objective": "masked-action-classification",
        "optimizer": "adam",
        "seed": 0,
        "selected_epoch": 1,
        "train_splits": ["TRAIN"],
        "validation_splits": ["SELECT"],
        "weight_decay": 0.0,
    }
    value.update(overrides)
    return value


def zero_weights(model_config):
    """artifact write testで使う決定的なweights payloadを返す。"""
    return tuple(0.0 for _ in range(model_config.parameter_count))
