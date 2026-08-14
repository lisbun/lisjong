import unittest
from dataclasses import FrozenInstanceError, fields

from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

DORA_1 = Tile(TileType(TileCategory.MANZU, 3))
DORA_2 = Tile(TileType(TileCategory.PINZU, 7))
DORA_3 = Tile(TileType(TileCategory.SOUZU, 1))


def _make(**overrides: object) -> RoundState:
    kwargs: dict[str, object] = {
        "round_wind": Wind.EAST,
        "hand_number": 1,
        "dealer_seat": Seat.SEAT_0,
        "honba": 0,
        "riichi_sticks": 0,
        "dora_indicators": (DORA_1,),
        "live_wall_tiles_remaining": 70,
    }
    kwargs.update(overrides)
    return RoundState(**kwargs)


class RoundStateTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        state = _make(
            round_wind=Wind.SOUTH,
            hand_number=3,
            dealer_seat=Seat.SEAT_2,
            honba=2,
            riichi_sticks=1,
            dora_indicators=(DORA_1, DORA_2),
            live_wall_tiles_remaining=42,
        )
        self.assertEqual(state.round_wind, Wind.SOUTH)
        self.assertEqual(state.hand_number, 3)
        self.assertEqual(state.dealer_seat, Seat.SEAT_2)
        self.assertEqual(state.honba, 2)
        self.assertEqual(state.riichi_sticks, 1)
        self.assertEqual(state.dora_indicators, (DORA_1, DORA_2))
        self.assertEqual(state.live_wall_tiles_remaining, 42)

    def test_accepts_all_four_winds(self) -> None:
        for wind in (Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH):
            with self.subTest(wind=wind):
                state = _make(round_wind=wind)
                self.assertEqual(state.round_wind, wind)

    def test_is_immutable(self) -> None:
        state = _make()
        with self.assertRaises(FrozenInstanceError):
            state.honba = 1

    def test_equal_values_compare_equal_and_are_hashable(self) -> None:
        first = _make(dora_indicators=(DORA_1, DORA_2))
        second = _make(dora_indicators=(DORA_1, DORA_2))
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)

    def test_rejects_non_wind_round_wind(self) -> None:
        with self.assertRaises(TypeError):
            _make(round_wind="east")

    def test_rejects_boolean_hand_number(self) -> None:
        with self.assertRaises(TypeError):
            _make(hand_number=True)

    def test_rejects_hand_number_out_of_range(self) -> None:
        for hand_number in (0, 5):
            with self.subTest(hand_number=hand_number), self.assertRaises(ValueError):
                _make(hand_number=hand_number)

    def test_accepts_hand_number_boundaries(self) -> None:
        for hand_number in (1, 4):
            with self.subTest(hand_number=hand_number):
                state = _make(hand_number=hand_number)
                self.assertEqual(state.hand_number, hand_number)

    def test_rejects_non_seat_dealer_seat(self) -> None:
        with self.assertRaises(TypeError):
            _make(dealer_seat=0)

    def test_rejects_boolean_honba(self) -> None:
        with self.assertRaises(TypeError):
            _make(honba=True)

    def test_rejects_negative_honba(self) -> None:
        with self.assertRaises(ValueError):
            _make(honba=-1)

    def test_rejects_boolean_riichi_sticks(self) -> None:
        with self.assertRaises(TypeError):
            _make(riichi_sticks=True)

    def test_rejects_negative_riichi_sticks(self) -> None:
        with self.assertRaises(ValueError):
            _make(riichi_sticks=-1)

    def test_allows_riichi_sticks_greater_than_four(self) -> None:
        # 局をまたぐ供託があり得るため、4本という上限は課さない。
        state = _make(riichi_sticks=6)
        self.assertEqual(state.riichi_sticks, 6)

    def test_rejects_non_tile_in_dora_indicators(self) -> None:
        with self.assertRaises(TypeError):
            _make(dora_indicators=(DORA_1, "5m"))

    def test_allows_empty_dora_indicators(self) -> None:
        # 非空はRoundState単一値の構造的不変条件ではなく、Policy decisionを
        # 生成する環境・タイミング側の条件である。必要ならcontext整合条件として
        # 後続のAdapter / DecisionContext境界で検証する。
        state = _make(dora_indicators=())
        self.assertEqual(state.dora_indicators, ())

    def test_normalizes_list_input_into_tuple(self) -> None:
        state = _make(dora_indicators=[DORA_1, DORA_2])
        self.assertIsInstance(state.dora_indicators, tuple)
        self.assertEqual(state.dora_indicators, (DORA_1, DORA_2))

    def test_does_not_reorder_dora_indicators(self) -> None:
        # 公開順を保持するsequenceであり、tile_sort_key等で並べ替えない。
        state = _make(dora_indicators=(DORA_3, DORA_1, DORA_2))
        self.assertEqual(state.dora_indicators, (DORA_3, DORA_1, DORA_2))

    def test_rejects_boolean_live_wall_tiles_remaining(self) -> None:
        with self.assertRaises(TypeError):
            _make(live_wall_tiles_remaining=True)

    def test_rejects_negative_live_wall_tiles_remaining(self) -> None:
        with self.assertRaises(ValueError):
            _make(live_wall_tiles_remaining=-1)

    def test_allows_live_wall_tiles_remaining_without_upper_bound(self) -> None:
        # 136枚・70枚等の具体上限を持ち込まない。
        state = _make(live_wall_tiles_remaining=1000)
        self.assertEqual(state.live_wall_tiles_remaining, 1000)

    def test_has_no_hidden_or_environment_specific_fields(self) -> None:
        field_names = {field.name for field in fields(RoundState)}
        self.assertEqual(
            field_names,
            {
                "round_wind",
                "hand_number",
                "dealer_seat",
                "honba",
                "riichi_sticks",
                "dora_indicators",
                "live_wall_tiles_remaining",
            },
        )
        for forbidden in (
            "kyoku_index",
            "wall",
            "ura_dora_indicators",
            "kan_count",
            "env",
        ):
            self.assertNotIn(forbidden, field_names)


if __name__ == "__main__":
    unittest.main()
