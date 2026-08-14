import unittest
from dataclasses import FrozenInstanceError, fields

from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

MANZU_3 = Tile(TileType(TileCategory.MANZU, 3))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))


def _make_player(score: int = 25000) -> PlayerPublicState:
    return PlayerPublicState(
        score=score, discards=(), melds=(), riichi=RiichiState.NONE
    )


def _make_players() -> tuple[
    PlayerPublicState, PlayerPublicState, PlayerPublicState, PlayerPublicState
]:
    return (_make_player(), _make_player(), _make_player(), _make_player())


def _make_round() -> RoundState:
    return RoundState(
        round_wind=Wind.EAST,
        hand_number=1,
        dealer_seat=Seat.SEAT_0,
        honba=0,
        riichi_sticks=0,
        dora_indicators=(MANZU_3,),
        live_wall_tiles_remaining=70,
    )


def _make_own_hand() -> OwnHandState:
    return OwnHandState(concealed_tiles=(MANZU_5,), drawn_tile=MANZU_5)


def _make(**overrides: object) -> PolicyInput:
    kwargs: dict[str, object] = {
        "self_seat": Seat.SEAT_0,
        "round": _make_round(),
        "players": _make_players(),
        "own_hand": _make_own_hand(),
    }
    kwargs.update(overrides)
    return PolicyInput(**kwargs)


class PolicyInputTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        state = _make()
        self.assertEqual(state.self_seat, Seat.SEAT_0)
        self.assertIsInstance(state.round, RoundState)
        self.assertEqual(len(state.players), 4)
        self.assertIsInstance(state.own_hand, OwnHandState)

    def test_is_immutable(self) -> None:
        state = _make()
        with self.assertRaises(FrozenInstanceError):
            state.self_seat = Seat.SEAT_1

    def test_equal_values_compare_equal_and_are_hashable(self) -> None:
        first = _make()
        second = _make()
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)

    def test_normalizes_list_players_into_tuple(self) -> None:
        state = _make(players=list(_make_players()))
        self.assertIsInstance(state.players, tuple)

    def test_preserves_players_order(self) -> None:
        distinct_players = (
            _make_player(score=25000),
            _make_player(score=24000),
            _make_player(score=26000),
            _make_player(score=23000),
        )
        state = _make(players=distinct_players)
        self.assertEqual(state.players, distinct_players)

    def test_does_not_rotate_players_for_non_zero_self_seat(self) -> None:
        distinct_players = (
            _make_player(score=25000),
            _make_player(score=24000),
            _make_player(score=26000),
            _make_player(score=23000),
        )
        state = _make(self_seat=Seat.SEAT_2, players=distinct_players)
        self.assertEqual(state.players, distinct_players)
        self.assertEqual(state.players[0].score, 25000)
        self.assertEqual(state.players[2].score, 26000)

    def test_rejects_plain_int_self_seat(self) -> None:
        with self.assertRaises(TypeError):
            _make(self_seat=0)

    def test_rejects_non_round_state_round(self) -> None:
        with self.assertRaises(TypeError):
            _make(round="round")

    def test_rejects_non_iterable_players(self) -> None:
        with self.assertRaises(TypeError):
            _make(players=123)

    def test_rejects_non_player_public_state_in_players(self) -> None:
        with self.assertRaises(TypeError):
            _make(
                players=(_make_player(), _make_player(), _make_player(), "not a player")
            )

    def test_rejects_non_own_hand_state_own_hand(self) -> None:
        with self.assertRaises(TypeError):
            _make(own_hand="hand")

    def test_rejects_zero_players(self) -> None:
        with self.assertRaises(ValueError):
            _make(players=())

    def test_rejects_three_players(self) -> None:
        with self.assertRaises(ValueError):
            _make(players=_make_players()[:3])

    def test_rejects_five_players(self) -> None:
        with self.assertRaises(ValueError):
            _make(players=_make_players() + (_make_player(),))

    def test_accepts_four_players(self) -> None:
        state = _make(players=_make_players())
        self.assertEqual(len(state.players), 4)

    def test_has_exactly_the_documented_fields(self) -> None:
        field_names = {field.name for field in fields(PolicyInput)}
        self.assertEqual(field_names, {"self_seat", "round", "players", "own_hand"})
        for forbidden in (
            "external_player_id",
            "riichienv_player_id",
            "rotated_players",
            "current_player",
            "seat_wind",
            "legal_actions",
        ):
            self.assertNotIn(forbidden, field_names)


if __name__ == "__main__":
    unittest.main()
