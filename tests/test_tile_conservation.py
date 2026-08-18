import unittest

from lisjong.belief.tile_conservation import (
    TileConservationResult,
    derive_remaining_tile_inventory,
)
from lisjong.belief.tile_inventory import TOTAL_PHYSICAL_TILE_COUNT
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

MANZU_3 = TileType(TileCategory.MANZU, 3)
MANZU_5 = TileType(TileCategory.MANZU, 5)
PINZU_7 = TileType(TileCategory.PINZU, 7)


def _tile(tile_type: TileType, *, is_red: bool = False) -> Tile:
    return Tile(tile_type, is_red=is_red)


def _discard(tile: Tile, order: int, called_by: Seat | None = None) -> Discard:
    return Discard(tile=tile, tsumogiri=False, order=order, called_by=called_by)


def _player(
    discards: tuple[Discard, ...] = (), melds: tuple[PublicMeld, ...] = ()
) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000, discards=discards, melds=melds, riichi=RiichiState.NONE
    )


def _empty_players() -> tuple[
    PlayerPublicState, PlayerPublicState, PlayerPublicState, PlayerPublicState
]:
    return (_player(), _player(), _player(), _player())


def _round(
    dealer_seat: Seat = Seat.SEAT_0,
    dora_indicators: tuple[Tile, ...] = (),
    live_wall_tiles_remaining: int = 70,
) -> RoundState:
    return RoundState(
        round_wind=Wind.EAST,
        hand_number=1,
        dealer_seat=dealer_seat,
        honba=0,
        riichi_sticks=0,
        dora_indicators=dora_indicators,
        live_wall_tiles_remaining=live_wall_tiles_remaining,
    )


def _own_hand(concealed_tiles: tuple[Tile, ...] = ()) -> OwnHandState:
    return OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=None)


def _policy_input(
    players: tuple[
        PlayerPublicState, PlayerPublicState, PlayerPublicState, PlayerPublicState
    ] = None,
    dealer_seat: Seat = Seat.SEAT_0,
    dora_indicators: tuple[Tile, ...] = (),
    concealed_tiles: tuple[Tile, ...] = (),
    live_wall_tiles_remaining: int = 70,
    self_seat: Seat = Seat.SEAT_0,
) -> PolicyInput:
    if players is None:
        players = _empty_players()
    return PolicyInput(
        self_seat=self_seat,
        round=_round(
            dealer_seat=dealer_seat,
            dora_indicators=dora_indicators,
            live_wall_tiles_remaining=live_wall_tiles_remaining,
        ),
        players=players,
        own_hand=_own_hand(concealed_tiles),
    )


class EmptyStateTest(unittest.TestCase):
    def test_empty_self_and_public_state_yields_full_remaining_inventory(self) -> None:
        result = derive_remaining_tile_inventory(_policy_input())
        self.assertEqual(sum(result.exact_accounted_counts), 0)
        self.assertEqual(sum(result.exact_accounted_red_five_counts), 0)
        self.assertEqual(sum(result.remaining_tile_counts), TOTAL_PHYSICAL_TILE_COUNT)
        self.assertEqual(sum(result.remaining_red_five_counts), 3)

    def test_rejects_non_policy_input(self) -> None:
        with self.assertRaises(TypeError):
            derive_remaining_tile_inventory(None)


class SelfAndDoraTest(unittest.TestCase):
    def test_self_hand_and_dora_indicator_reduce_remaining(self) -> None:
        result = derive_remaining_tile_inventory(
            _policy_input(
                concealed_tiles=(_tile(MANZU_3),),
                dora_indicators=(_tile(MANZU_3),),
            )
        )
        index = 2  # MANZU_3 canonical index (0-based rank 3 -> index 2)
        self.assertEqual(result.exact_accounted_counts[index], 2)
        self.assertEqual(result.remaining_tile_counts[index], 2)


class CalledDiscardAndPonTest(unittest.TestCase):
    def _policy_input_with_pon(self) -> PolicyInput:
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(_tile(PINZU_7), _tile(PINZU_7), _tile(PINZU_7)),
            from_seat=Seat.SEAT_0,
            called_tile=_tile(PINZU_7),
        )
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(
            discards=(_discard(_tile(PINZU_7), order=0, called_by=Seat.SEAT_1),)
        )
        players[Seat.SEAT_1] = _player(melds=(pon,))
        return _policy_input(players=tuple(players))

    def test_called_tile_not_double_counted(self) -> None:
        result = derive_remaining_tile_inventory(self._policy_input_with_pon())
        pinzu_index = 9 + 6  # PINZU base 9, rank 7 -> +6
        self.assertEqual(result.exact_accounted_counts[pinzu_index], 3)
        self.assertEqual(result.remaining_tile_counts[pinzu_index], 1)


class OverCountingTest(unittest.TestCase):
    def _players_with_self_discard_and_meld(
        self, extra_dora: bool = False
    ) -> PolicyInput:
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(_tile(PINZU_7), _tile(PINZU_7), _tile(PINZU_7)),
            from_seat=Seat.SEAT_2,
            called_tile=_tile(PINZU_7),
        )
        players = list(_empty_players())
        players[Seat.SEAT_2] = _player(
            discards=(_discard(_tile(PINZU_7), order=0, called_by=Seat.SEAT_0),)
        )
        players[Seat.SEAT_0] = _player(melds=(pon,))
        return _policy_input(
            players=tuple(players),
            concealed_tiles=(_tile(PINZU_7),),
            dora_indicators=(_tile(PINZU_7),) if extra_dora else (),
        )

    def test_self_plus_discard_plus_meld_hand_origin_is_fully_accounted(self) -> None:
        result = derive_remaining_tile_inventory(
            self._players_with_self_discard_and_meld()
        )
        pinzu_index = 9 + 6
        self.assertEqual(result.exact_accounted_counts[pinzu_index], 4)
        self.assertEqual(result.remaining_tile_counts[pinzu_index], 0)

    def test_additional_dora_indicator_overflows_inventory_and_fails_closed(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            derive_remaining_tile_inventory(
                self._players_with_self_discard_and_meld(extra_dora=True)
            )


class RedFiveCrossSourceTest(unittest.TestCase):
    def test_red_five_accounted_from_self_and_discard_is_rejected(self) -> None:
        players = list(_empty_players())
        players[Seat.SEAT_1] = _player(
            discards=(_discard(_tile(MANZU_5, is_red=True), order=0),)
        )
        policy_input = _policy_input(
            players=tuple(players), concealed_tiles=(_tile(MANZU_5, is_red=True),)
        )
        with self.assertRaises(ValueError):
            derive_remaining_tile_inventory(policy_input)


class SelfDuplicateRedFiveTest(unittest.TestCase):
    def test_own_hand_state_with_two_red_fives_is_rejected(self) -> None:
        # exact_self_belief()/HandBeliefを経由していれば、赤5はpresence（0/1）
        # として扱われ、この不正な重複が隠れてしまう。self countは
        # OwnHandState.concealed_tilesから直接countすることの回帰テスト。
        policy_input = _policy_input(
            concealed_tiles=(
                _tile(MANZU_5, is_red=True),
                _tile(MANZU_5, is_red=True),
            )
        )
        with self.assertRaises(ValueError):
            derive_remaining_tile_inventory(policy_input)


class ImpossibleRedFiveInventoryTest(unittest.TestCase):
    def test_accounted_four_normal_fives_with_no_red_is_rejected(self) -> None:
        # accounted 5m = 4 (すべて通常5), accounted red5m = 0 の場合、
        # remaining 5m = 0 / remaining red5m = 1 となり、standard inventoryと
        # 矛盾するため拒否する。
        policy_input = _policy_input(
            concealed_tiles=tuple(_tile(MANZU_5) for _ in range(4))
        )
        with self.assertRaises(ValueError):
            derive_remaining_tile_inventory(policy_input)

    def test_accounted_four_fives_with_one_red_is_valid(self) -> None:
        concealed_tiles = (
            _tile(MANZU_5),
            _tile(MANZU_5),
            _tile(MANZU_5),
            _tile(MANZU_5, is_red=True),
        )
        result = derive_remaining_tile_inventory(
            _policy_input(concealed_tiles=concealed_tiles)
        )
        five_index = 4  # MANZU rank5 -> index 4
        self.assertEqual(result.exact_accounted_counts[five_index], 4)
        self.assertEqual(result.exact_accounted_red_five_counts[0], 1)
        self.assertEqual(result.remaining_tile_counts[five_index], 0)
        self.assertEqual(result.remaining_red_five_counts[0], 0)


class DealerAndWindMappingTest(unittest.TestCase):
    def test_dealer_change_does_not_change_aggregate_remaining_inventory(self) -> None:
        players = list(_empty_players())
        players[Seat.SEAT_2] = _player(discards=(_discard(_tile(MANZU_3), order=0),))
        players = tuple(players)

        result_dealer_0 = derive_remaining_tile_inventory(
            _policy_input(players=players, dealer_seat=Seat.SEAT_0)
        )
        result_dealer_3 = derive_remaining_tile_inventory(
            _policy_input(players=players, dealer_seat=Seat.SEAT_3)
        )

        self.assertEqual(
            result_dealer_0.exact_accounted_counts,
            result_dealer_3.exact_accounted_counts,
        )
        self.assertEqual(
            result_dealer_0.remaining_tile_counts, result_dealer_3.remaining_tile_counts
        )


class LiveWallIndependenceTest(unittest.TestCase):
    def test_live_wall_tiles_remaining_does_not_affect_result(self) -> None:
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(discards=(_discard(_tile(MANZU_3), order=0),))
        players = tuple(players)

        result_a = derive_remaining_tile_inventory(
            _policy_input(players=players, live_wall_tiles_remaining=70)
        )
        result_b = derive_remaining_tile_inventory(
            _policy_input(players=players, live_wall_tiles_remaining=12)
        )

        self.assertEqual(result_a, result_b)


class DeterminismTest(unittest.TestCase):
    def test_same_policy_input_yields_same_result(self) -> None:
        policy_input = _policy_input(
            concealed_tiles=(_tile(MANZU_3),), dora_indicators=(_tile(PINZU_7),)
        )
        first = derive_remaining_tile_inventory(policy_input)
        second = derive_remaining_tile_inventory(policy_input)
        self.assertEqual(first, second)


class GlobalConservationTest(unittest.TestCase):
    def test_every_tile_type_satisfies_accounted_plus_remaining_equals_four(
        self,
    ) -> None:
        policy_input = _policy_input(
            concealed_tiles=(_tile(MANZU_3), _tile(MANZU_5, is_red=True)),
            dora_indicators=(_tile(PINZU_7),),
        )
        result = derive_remaining_tile_inventory(policy_input)
        for accounted, remaining in zip(
            result.exact_accounted_counts, result.remaining_tile_counts
        ):
            self.assertEqual(accounted + remaining, 4)

    def test_global_136_tile_conservation(self) -> None:
        policy_input = _policy_input(
            concealed_tiles=(_tile(MANZU_3), _tile(MANZU_5, is_red=True)),
            dora_indicators=(_tile(PINZU_7),),
        )
        result = derive_remaining_tile_inventory(policy_input)
        self.assertEqual(
            sum(result.exact_accounted_counts) + sum(result.remaining_tile_counts),
            TOTAL_PHYSICAL_TILE_COUNT,
        )


class TileConservationResultConstructionTest(unittest.TestCase):
    def _valid_kwargs(self) -> dict:
        accounted = [0] * 34
        remaining = [4] * 34
        accounted_red = [0, 0, 0]
        remaining_red = [1, 1, 1]
        return dict(
            exact_accounted_counts=tuple(accounted),
            exact_accounted_red_five_counts=tuple(accounted_red),
            remaining_tile_counts=tuple(remaining),
            remaining_red_five_counts=tuple(remaining_red),
        )

    def test_valid_all_zero_accounted_state(self) -> None:
        result = TileConservationResult(**self._valid_kwargs())
        self.assertEqual(sum(result.remaining_tile_counts), TOTAL_PHYSICAL_TILE_COUNT)

    def test_rejects_negative_remaining(self) -> None:
        kwargs = self._valid_kwargs()
        accounted = [5] + [0] * 33
        remaining = [-1] + [4] * 33
        kwargs["exact_accounted_counts"] = tuple(accounted)
        kwargs["remaining_tile_counts"] = tuple(remaining)
        with self.assertRaises(ValueError):
            TileConservationResult(**kwargs)

    def test_rejects_mismatched_length(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["exact_accounted_counts"] = tuple([0] * 33)
        with self.assertRaises(ValueError):
            TileConservationResult(**kwargs)


if __name__ == "__main__":
    unittest.main()
