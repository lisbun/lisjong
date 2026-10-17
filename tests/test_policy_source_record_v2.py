"""Issue #307: policy source record v2（宣言牌の位置を保持）のreader test。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from learning_fixtures import (
    decision_rows,
    discard_legal_actions,
    source_row,
    write_policy_source_record,
    write_source_record,
)
from learning_fixtures import policy_input as fixture_policy_input

from lisjong.learning._typed_values import policy_input_to_value_v2
from lisjong.learning.errors import SourceRecordError, UnsupportedSourceSchemaError
from lisjong.learning.source_record import (
    POLICY_SOURCE_RECORD_SCHEMA_V1,
    POLICY_SOURCE_RECORD_SCHEMA_V2,
    open_source_record,
    read_source_game,
    read_source_record,
)
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat

FIELD = "riichi_declaration_discard_order"


def _riichi_input(seat):
    """seat 1がorder 1の打牌でリーチを宣言している入力。"""
    value = fixture_policy_input(self_seat=seat)
    players = list(value.players)
    players[1] = replace(
        players[1], riichi=RiichiState.ACCEPTED, riichi_declaration_discard_order=1
    )
    return replace(value, players=tuple(players))


def _v2_rows(*, game_ordinal, seed, split, count=3, with_order=True):
    rows = []
    for ordinal in range(count):
        seat = Seat(ordinal % 4)
        value = (
            _riichi_input(seat)
            if with_order and ordinal == 1
            else fixture_policy_input(self_seat=seat)
        )
        legal = discard_legal_actions(value, with_riichi=ordinal == 0)
        without_order = replace(
            value,
            players=tuple(
                replace(player, riichi_declaration_discard_order=None)
                for player in value.players
            ),
        )
        row = source_row(
            game_ordinal=game_ordinal,
            seed=seed,
            split=split,
            step_ordinal=ordinal,
            decision_ordinal=ordinal,
            policy_input_value=without_order,
            legal_actions=legal,
            selected_action=legal[0],
        )
        row["policy_input"] = policy_input_to_value_v2(
            value, ValueError, "policy_input"
        )
        rows.append(row)
    return rows


def _v2_games(**kwargs):
    return (
        ("TRAIN", 100, _v2_rows(game_ordinal=0, seed=100, split="TRAIN", **kwargs)),
        (
            "SELECT",
            200,
            _v2_rows(game_ordinal=1, seed=200, split="SELECT", count=2, **kwargs),
        ),
    )


class PolicySourceRecordV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _write_v2(self, name="v2", **kwargs):
        return write_policy_source_record(
            self.root / name, _v2_games(**kwargs), schema=POLICY_SOURCE_RECORD_SCHEMA_V2
        )

    def test_v2_record_keeps_the_declaration_order(self) -> None:
        record = read_source_record(self._write_v2())

        self.assertEqual(record.schema, POLICY_SOURCE_RECORD_SCHEMA_V2)
        decision = record.games[0].decisions[1]
        self.assertEqual(decision.policy_input, _riichi_input(Seat.SEAT_1))
        self.assertEqual(
            decision.policy_input.players[1].riichi_declaration_discard_order, 1
        )

    def test_streaming_read_keeps_the_declaration_order(self) -> None:
        source = open_source_record(self._write_v2())

        decisions = list(source.decisions())

        self.assertEqual(
            decisions[1].policy_input.players[1].riichi_declaration_discard_order, 1
        )
        self.assertEqual(len(decisions), source.decision_count)

    def test_v2_record_without_any_order_is_read_with_none(self) -> None:
        record = read_source_record(self._write_v2(with_order=False))

        for game in record.games:
            for decision in game.decisions:
                for player in decision.policy_input.players:
                    self.assertIsNone(player.riichi_declaration_discard_order)

    def test_v2_record_rejects_v1_rows(self) -> None:
        games = (
            ("TRAIN", 100, decision_rows(game_ordinal=0, seed=100, split="TRAIN")),
        )
        path = write_policy_source_record(
            self.root / "mixed", games, schema=POLICY_SOURCE_RECORD_SCHEMA_V2
        )

        with self.assertRaises(SourceRecordError):
            read_source_record(path)

    def test_v1_record_rejects_v2_rows(self) -> None:
        path = write_policy_source_record(
            self.root / "mixed",
            _v2_games(with_order=False),
            schema=POLICY_SOURCE_RECORD_SCHEMA_V1,
        )

        with self.assertRaises(SourceRecordError):
            read_source_record(path)

    def test_v2_rows_with_an_inconsistent_order_are_rejected(self) -> None:
        games = list(_v2_games())
        split, seed, rows = games[0]
        rows[1]["policy_input"]["players"][1][FIELD] = 99
        games[0] = (split, seed, rows)
        path = write_policy_source_record(
            self.root / "bad", tuple(games), schema=POLICY_SOURCE_RECORD_SCHEMA_V2
        )

        with self.assertRaises(SourceRecordError):
            read_source_record(path)

    def test_existing_schemas_are_read_as_before(self) -> None:
        v1 = read_source_record(write_policy_source_record(self.root / "policy-v1"))
        o0 = read_source_record(write_source_record(self.root / "o0-v2"))

        self.assertEqual(v1.schema, POLICY_SOURCE_RECORD_SCHEMA_V1)
        for record in (v1, o0):
            for game in record.games:
                for decision in game.decisions:
                    for player in decision.policy_input.players:
                        self.assertIsNone(player.riichi_declaration_discard_order)

    def test_read_source_game_requires_the_schema(self) -> None:
        source = open_source_record(self._write_v2())
        summary = source.summaries[0]

        with self.assertRaises(TypeError):
            read_source_game(source.root, summary)
        with self.assertRaises(SourceRecordError):
            read_source_game(
                source.root, summary, schema=POLICY_SOURCE_RECORD_SCHEMA_V1
            )
        with self.assertRaises(UnsupportedSourceSchemaError):
            read_source_game(source.root, summary, schema="unknown-schema")
        game = read_source_game(
            source.root, summary, schema=POLICY_SOURCE_RECORD_SCHEMA_V2
        )
        self.assertEqual(
            game.decisions[1].policy_input.players[1].riichi_declaration_discard_order,
            1,
        )


if __name__ == "__main__":
    unittest.main()
