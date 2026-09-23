"""Issue #193 focal outcome source（lisbun/lisjong-arena#359）の合成fixture。

実Arena producerを使わず、consumerが読むwire layoutを直接書く。focal decision
のselected action / producer survivor action列は、Arenaのgeneration-only
adapterと同じく`select_residual_exploration()`の1回の結果から記録する。
tokenはtest専用の任意の64-hex値であり、Arenaのtoken導出は再現しない。
"""

import atexit
import dataclasses
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import candidate_fixtures as cf
import learning_fixtures as fixtures

from lisjong.learning._canonical import (
    canonical_json_line,
    canonical_json_text,
    file_digest,
    seal,
)
from lisjong.learning._typed_values import action_to_value, policy_input_to_value
from lisjong.learning.errors import SourceRecordError
from lisjong.learning.outcome_source import (
    DECISION_PAYLOAD_FILENAME,
    EXPECTED_BEHAVIOR,
    KYOKU_PAYLOAD_FILENAME,
    MANIFEST_FILENAME,
    OUTCOME_SOURCE_KIND,
    OUTCOME_SOURCE_SCHEMA,
)
from lisjong.learning.residual_exploration import select_residual_exploration
from lisjong.policy_contract import DecisionContext, Seat, Wind

SCIENTIFIC_GAMES = (("TRAIN", 1001), ("TRAIN", 1002), ("SELECT", 2001))
CALIBRATION_GAMES = (("CALIBRATION", 3001), ("CALIBRATION", 3002))

KYOKU_IDENTITIES = ((Wind.EAST, 1, 0), (Wind.EAST, 2, 0), (Wind.EAST, 2, 1))
"""3 kyoku。最後はhonba付きの連荘kyoku（hanchan最終kyoku）。"""

FINAL_KYOKU_TARGET = 2.0
"""最終kyokuのfocal target（riichi控除-1000 + 流局聴牌料+3000）。"""


def token(game_ordinal, focal_decision_ordinal):
    """test専用の64-hex token（Arena token導出の再実装ではない）。"""
    text = f"test-token:{game_ordinal}:{focal_decision_ordinal}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _action_line(action):
    return canonical_json_line(action_to_value(action, SourceRecordError, "action"))


def canonical_decision(decision, kyoku_ordinal):
    """legal actionsをcanonical順へ並べ、PolicyInputのkyoku identityを合わせる。"""
    wind, hand_number, honba = KYOKU_IDENTITIES[kyoku_ordinal]
    round_state = fixtures.round_state(
        round_wind=wind,
        hand_number=hand_number,
        dealer_seat=Seat(hand_number - 1),
        honba=honba,
    )
    policy_input = dataclasses.replace(decision.input, round=round_state)
    legal = tuple(sorted(decision.legal_actions, key=_action_line))
    return DecisionContext(input=policy_input, legal_actions=legal)


def focal_decisions(seat):
    """`(kyoku_ordinal, DecisionContext)`のfocal decision列。

    eligible（survivor >= 2）が2件、single-survivor、WIN / RIICHI / RESPONSE
    guardを1件ずつ含む。
    """
    raw = (
        (0, cf.discard_decision(cf.NEAR_HAND, self_seat=seat)),
        (0, cf.tsumo_decision(self_seat=seat)),
        (1, cf.discard_decision("123m456p789s11z23s9m", self_seat=seat)),
        (1, cf.response_decision(self_seat=seat)),
        (2, cf.riichi_decision(self_seat=seat)),
        (2, cf.discard_decision(cf.SECOND_NEAR_HAND, self_seat=seat)),
    )
    return tuple(
        (kyoku, canonical_decision(decision, kyoku)) for kyoku, decision in raw
    )


def decision_row(game_ordinal, index, kyoku_ordinal, decision, *, step_ordinal=None):
    value = token(game_ordinal, index)
    selection = select_residual_exploration(decision, value)
    survivors = selection.survivor_actions
    return {
        "actor_seat": int(decision.input.self_seat),
        "exploration_token": value,
        "focal_decision_ordinal": index,
        "game_ordinal": game_ordinal,
        "kyoku_ordinal": kyoku_ordinal,
        "legal_actions": [
            action_to_value(action, SourceRecordError, "legal")
            for action in decision.legal_actions
        ],
        "policy_input": policy_input_to_value(
            decision.input, SourceRecordError, "policy_input"
        ),
        "producer_survivor_actions": None
        if survivors is None
        else [
            action_to_value(action, SourceRecordError, "survivor")
            for action in survivors
        ],
        "selected_action": action_to_value(
            selection.action, SourceRecordError, "selected"
        ),
        "step_ordinal": 3 * index + 1 if step_ordinal is None else step_ordinal,
    }


def kyoku_rows(game_ordinal, focal):
    """focal視点で +8000（和了）/ -1500（ノーテン）/ +2000（riichi後の聴牌流局）。

    最終kyokuは供託1本を残して終わる。hanchan最終調整後のscoreは
    `hanchan_final`として別に返す。
    """
    other = (focal + 1) % 4
    points = [25000] * 4
    rows = []

    def row(ordinal, before, after, sticks_before, sticks_after, end):
        wind, hand_number, honba = KYOKU_IDENTITIES[ordinal]
        return {
            "dealer_seat": hand_number - 1,
            "end": end,
            "game_ordinal": game_ordinal,
            "hand_number": hand_number,
            "honba": honba,
            "is_final_kyoku": ordinal == len(KYOKU_IDENTITIES) - 1,
            "kyoku_ordinal": ordinal,
            "points_after_kyoku": list(after),
            "points_before_kyoku": list(before),
            "riichi_sticks_after": sticks_after,
            "riichi_sticks_before": sticks_before,
            "round_wind": wind.value,
        }

    after = list(points)
    after[focal] += 8000
    after[other] -= 8000
    rows.append(row(0, points, after, 0, 0, {"kind": "win", "winner_seats": [focal]}))

    points, after = after, list(after)
    for seat in range(4):
        after[seat] += -1500 if seat == focal else 500
    rows.append(
        row(1, points, after, 0, 0, {"draw_kind": "exhaustive", "kind": "draw"})
    )

    points, after = after, list(after)
    for seat in range(4):
        after[seat] += 2000 if seat == focal else -1000
    rows.append(
        row(2, points, after, 0, 1, {"draw_kind": "exhaustive", "kind": "draw"})
    )

    final = list(after)
    final[max(range(4), key=lambda seat: (after[seat], -seat))] += 1000
    return rows, final


def _write_lines(path, rows):
    path.write_text(
        "".join(canonical_json_line(row) for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _game_summary(root, game_ordinal, seed, split, *, final_scores, kyokus, decisions):
    directory = root / f"game-{game_ordinal:03d}"
    return seal(
        {
            "decision_count": decisions,
            "files": {
                name: file_digest(directory / name)
                for name in (DECISION_PAYLOAD_FILENAME, KYOKU_PAYLOAD_FILENAME)
            },
            "focal_seat": game_ordinal % 4,
            "game_ordinal": game_ordinal,
            "hanchan_final_riichi_sticks": 0,
            "hanchan_final_scores": final_scores,
            "kyoku_count": kyokus,
            "seed": seed,
            "split": split,
        }
    )


_TEMPLATES = {}
_TEMPLATE_ROOT = Path(tempfile.mkdtemp(prefix="lisjong-outcome-fixtures-"))
atexit.register(shutil.rmtree, _TEMPLATE_ROOT, ignore_errors=True)


def write_outcome_source(root, games=SCIENTIFIC_GAMES, *, role="SCIENTIFIC"):
    """`games`（`(split, seed)`列）のfocal outcome sourceを書いてpathを返す。

    selection計算を伴う生成は`(games, role)`ごとに1回だけ行い、以降は
    生成済みdirectoryをcopyする。
    """
    key = (tuple(games), role)
    if key not in _TEMPLATES:
        _TEMPLATES[key] = _write_outcome_source(
            _TEMPLATE_ROOT / f"template-{len(_TEMPLATES)}", games, role=role
        )
    root = Path(root)
    shutil.copytree(_TEMPLATES[key], root)
    return root


def _write_outcome_source(root, games, *, role):
    root = Path(root)
    root.mkdir(parents=True)
    summaries = []
    seeds_by_split = {}
    for game_ordinal, (split, seed) in enumerate(games):
        seat = Seat(game_ordinal % 4)
        directory = root / f"game-{game_ordinal:03d}"
        directory.mkdir()
        kyokus, final_scores = kyoku_rows(game_ordinal, int(seat))
        decisions = [
            decision_row(game_ordinal, index, kyoku, decision)
            for index, (kyoku, decision) in enumerate(focal_decisions(seat))
        ]
        _write_lines(directory / KYOKU_PAYLOAD_FILENAME, kyokus)
        _write_lines(directory / DECISION_PAYLOAD_FILENAME, decisions)
        summaries.append(
            _game_summary(
                root,
                game_ordinal,
                seed,
                split,
                final_scores=final_scores,
                kyokus=len(kyokus),
                decisions=len(decisions),
            )
        )
        seeds_by_split.setdefault(split, []).append(seed)
    body = {
        "allocation_bindings": {
            split: fixtures.allocation_binding(seeds)
            for split, seeds in seeds_by_split.items()
        },
        "behavior": dict(EXPECTED_BEHAVIOR),
        "games": summaries,
        "kind": OUTCOME_SOURCE_KIND,
        "population_role": role,
        "schema": OUTCOME_SOURCE_SCHEMA,
        "source_contract": {"arena_revision": "test", "game_mode": "4p-hanchan"},
    }
    write_manifest(root, body)
    return root


def read_manifest_body(root):
    manifest = json.loads((Path(root) / MANIFEST_FILENAME).read_text("utf-8"))
    manifest.pop("identity")
    return manifest


def write_manifest(root, body):
    (Path(root) / MANIFEST_FILENAME).write_text(
        canonical_json_text(seal(body)), encoding="utf-8", newline="\n"
    )


def mutate_manifest(root, mutate):
    body = read_manifest_body(root)
    mutate(body)
    write_manifest(root, body)


def mutate_game_summary(root, game_ordinal, mutate):
    """game summaryを変更し、summaryとmanifestを再sealする。"""

    def apply(body):
        summary = dict(body["games"][game_ordinal])
        summary.pop("identity")
        mutate(summary)
        body["games"][game_ordinal] = seal(summary)

    mutate_manifest(root, apply)


def read_rows(root, game_ordinal, name):
    path = Path(root) / f"game-{game_ordinal:03d}" / name
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def write_rows(root, game_ordinal, name, rows):
    """payloadを書き換え、digestを更新してmanifestを再sealする。"""
    directory = Path(root) / f"game-{game_ordinal:03d}"
    _write_lines(directory / name, rows)
    digest = file_digest(directory / name)

    def update(summary):
        summary["files"] = {**summary["files"], name: digest}

    mutate_game_summary(root, game_ordinal, update)


def mutate_rows(root, game_ordinal, name, mutate):
    rows = read_rows(root, game_ordinal, name)
    mutate(rows)
    write_rows(root, game_ordinal, name, rows)
