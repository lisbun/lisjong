"""Issue #127 HandValueAware retained_real_value regression tests."""

import unittest

from lisjong.policies.hand_value_aware_two_step_ukeire import _retained_real_value
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind


def _tile(category: TileCategory, rank: int, *, red: bool = False) -> Tile:
    return Tile(TileType(category, rank), is_red=red)


def _player(*, melds: tuple[PublicMeld, ...] = ()) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=(),
        melds=melds,
        riichi=RiichiState.NONE,
    )


def _input(
    *,
    own_melds: tuple[PublicMeld, ...],
    dora_indicators: tuple[Tile, ...],
) -> PolicyInput:
    players = [_player() for _ in range(4)]
    players[0] = _player(melds=own_melds)
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=dora_indicators,
            live_wall_tiles_remaining=70,
        ),
        players=tuple(players),
        own_hand=OwnHandState(concealed_tiles=(), drawn_tile=None),
    )


def _pon(tiles: tuple[Tile, Tile, Tile]) -> PublicMeld:
    return PublicMeld(
        kind=MeldKind.PON,
        tiles=tiles,
        from_seat=Seat.SEAT_1,
        called_tile=tiles[0],
    )


class HandValueAwareConcealedDoraRegressionTest(unittest.TestCase):
    def test_meld_dora_and_aka_dora_are_not_retained_real_value(self) -> None:
        red_five = _tile(TileCategory.SOUZU, 5, red=True)
        normal_five = _tile(TileCategory.SOUZU, 5)
        indicator = _tile(TileCategory.SOUZU, 4)
        policy_input = _input(
            own_melds=(_pon((red_five, normal_five, normal_five)),),
            dora_indicators=(indicator,),
        )

        self.assertEqual(_retained_real_value((), policy_input), 0)

    def test_meld_yakuhai_remains_retained_real_value(self) -> None:
        white = _tile(TileCategory.HONOR, 5)
        policy_input = _input(
            own_melds=(_pon((white, white, white)),),
            dora_indicators=(),
        )

        self.assertEqual(_retained_real_value((), policy_input), 1)


if __name__ == "__main__":
    unittest.main()
