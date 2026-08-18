import unittest

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.public_provenance import (
    BASE_TILE_COUNT_MAX,
    RED_FIVE_COUNT_MAX,
    PublicTileProvenance,
    TileProvenanceCounts,
    WindTileProvenanceCounts,
    encode_public_tile_provenance,
)
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

SOUZU_9 = TileType(TileCategory.SOUZU, 9)
PINZU_7 = TileType(TileCategory.PINZU, 7)
MANZU_2 = TileType(TileCategory.MANZU, 2)
MANZU_3 = TileType(TileCategory.MANZU, 3)
MANZU_4 = TileType(TileCategory.MANZU, 4)
MANZU_5 = TileType(TileCategory.MANZU, 5)
NORTH_WIND_TYPE = TileType(TileCategory.HONOR, 4)


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


def _round(
    dealer_seat: Seat = Seat.SEAT_0, dora_indicators: tuple[Tile, ...] = ()
) -> RoundState:
    return RoundState(
        round_wind=Wind.EAST,
        hand_number=1,
        dealer_seat=dealer_seat,
        honba=0,
        riichi_sticks=0,
        dora_indicators=dora_indicators,
        live_wall_tiles_remaining=70,
    )


def _own_hand() -> OwnHandState:
    return OwnHandState(concealed_tiles=(), drawn_tile=None)


def _policy_input(
    players: tuple[
        PlayerPublicState, PlayerPublicState, PlayerPublicState, PlayerPublicState
    ],
    dealer_seat: Seat = Seat.SEAT_0,
    dora_indicators: tuple[Tile, ...] = (),
    self_seat: Seat = Seat.SEAT_0,
) -> PolicyInput:
    return PolicyInput(
        self_seat=self_seat,
        round=_round(dealer_seat=dealer_seat, dora_indicators=dora_indicators),
        players=players,
        own_hand=_own_hand(),
    )


def _empty_players() -> tuple[
    PlayerPublicState, PlayerPublicState, PlayerPublicState, PlayerPublicState
]:
    return (_player(), _player(), _player(), _player())


class TileProvenanceCountsTest(unittest.TestCase):
    def test_rejects_base_count_above_four(self) -> None:
        tile_counts = [0] * 34
        tile_counts[tile_type_index(SOUZU_9)] = BASE_TILE_COUNT_MAX + 1
        with self.assertRaises(ValueError):
            TileProvenanceCounts(
                tile_counts=tuple(tile_counts), red_five_counts=(0, 0, 0)
            )

    def test_rejects_red_five_count_above_one(self) -> None:
        with self.assertRaises(ValueError):
            TileProvenanceCounts(
                tile_counts=tuple([0] * 34),
                red_five_counts=(RED_FIVE_COUNT_MAX + 1, 0, 0),
            )

    def test_rejects_red_five_count_exceeding_base_five_count(self) -> None:
        tile_counts = [0] * 34
        tile_counts[tile_type_index(MANZU_5)] = 0
        with self.assertRaises(ValueError):
            TileProvenanceCounts(
                tile_counts=tuple(tile_counts), red_five_counts=(1, 0, 0)
            )

    def test_allows_red_five_count_equal_to_base_five_count(self) -> None:
        tile_counts = [0] * 34
        tile_counts[tile_type_index(MANZU_5)] = 1
        counts = TileProvenanceCounts(
            tile_counts=tuple(tile_counts), red_five_counts=(1, 0, 0)
        )
        self.assertEqual(counts.red_five_count(TileCategory.MANZU), 1)


class WindTileProvenanceCountsTest(unittest.TestCase):
    def test_rejects_wrong_number_of_winds(self) -> None:
        empty = TileProvenanceCounts(
            tile_counts=tuple([0] * 34), red_five_counts=(0, 0, 0)
        )
        with self.assertRaises(ValueError):
            WindTileProvenanceCounts(winds=(empty, empty, empty))


class EmptyPublicStateTest(unittest.TestCase):
    def test_empty_public_state_is_all_zero(self) -> None:
        provenance = encode_public_tile_provenance(_policy_input(_empty_players()))

        self.assertEqual(sum(provenance.discards.flattened_tile_counts), 0)
        self.assertEqual(sum(provenance.discards.flattened_red_five_counts), 0)
        self.assertEqual(sum(provenance.meld_hand_origin.flattened_tile_counts), 0)
        self.assertEqual(sum(provenance.meld_hand_origin.flattened_red_five_counts), 0)
        self.assertEqual(sum(provenance.dora_indicators.tile_counts), 0)
        self.assertEqual(sum(provenance.dora_indicators.red_five_counts), 0)

    def test_rejects_non_policy_input(self) -> None:
        with self.assertRaises(TypeError):
            encode_public_tile_provenance(None)


class SeatWindMappingTest(unittest.TestCase):
    def test_dealer_other_than_seat_0(self) -> None:
        players = list(_empty_players())
        players[Seat.SEAT_2] = _player(discards=(_discard(_tile(MANZU_3), order=0),))
        provenance = encode_public_tile_provenance(
            _policy_input(tuple(players), dealer_seat=Seat.SEAT_2)
        )

        # dealer(SEAT_2)がEAST、discardした本人がdealerなのでEAST側へ加算される。
        self.assertEqual(provenance.discards.tile_count(Wind.EAST, MANZU_3), 1)
        self.assertEqual(provenance.discards.tile_count(Wind.SOUTH, MANZU_3), 0)

    def test_dealer_seat_1_maps_seat_0_to_north(self) -> None:
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(discards=(_discard(_tile(MANZU_2), order=0),))
        provenance = encode_public_tile_provenance(
            _policy_input(tuple(players), dealer_seat=Seat.SEAT_1)
        )

        # dealer=SEAT_1のときSEAT_0は北家(NORTH)。
        self.assertEqual(provenance.discards.tile_count(Wind.NORTH, MANZU_2), 1)
        self.assertEqual(provenance.discards.tile_count(Wind.EAST, MANZU_2), 0)


class DiscardProvenanceTest(unittest.TestCase):
    def test_discards_normal_tile(self) -> None:
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(
            discards=(
                _discard(_tile(MANZU_3), order=0),
                _discard(_tile(MANZU_3), order=1),
            )
        )
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))
        self.assertEqual(provenance.discards.tile_count(Wind.EAST, MANZU_3), 2)

    def test_discards_red_five(self) -> None:
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(
            discards=(_discard(_tile(MANZU_5, is_red=True), order=0),)
        )
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))
        self.assertEqual(provenance.discards.tile_count(Wind.EAST, MANZU_5), 1)
        self.assertEqual(
            provenance.discards.red_five_count(Wind.EAST, TileCategory.MANZU), 1
        )

    def test_called_discard_still_counts_on_discarder_side(self) -> None:
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(
            discards=(_discard(_tile(PINZU_7), order=0, called_by=Seat.SEAT_1),)
        )
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))
        self.assertEqual(provenance.discards.tile_count(Wind.EAST, PINZU_7), 1)


class MeldHandOriginProvenanceTest(unittest.TestCase):
    def test_chi(self) -> None:
        meld = PublicMeld(
            kind=MeldKind.CHI,
            tiles=(_tile(MANZU_2), _tile(MANZU_3), _tile(MANZU_4)),
            from_seat=Seat.SEAT_3,
            called_tile=_tile(MANZU_3),
        )
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(melds=(meld,))
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))

        self.assertEqual(provenance.meld_hand_origin.tile_count(Wind.EAST, MANZU_2), 1)
        self.assertEqual(provenance.meld_hand_origin.tile_count(Wind.EAST, MANZU_4), 1)
        self.assertEqual(provenance.meld_hand_origin.tile_count(Wind.EAST, MANZU_3), 0)

    def test_pon_excludes_exactly_one_occurrence_of_called_tile(self) -> None:
        meld = PublicMeld(
            kind=MeldKind.PON,
            tiles=(_tile(PINZU_7), _tile(PINZU_7), _tile(PINZU_7)),
            from_seat=Seat.SEAT_3,
            called_tile=_tile(PINZU_7),
        )
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(melds=(meld,))
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))

        self.assertEqual(provenance.meld_hand_origin.tile_count(Wind.EAST, PINZU_7), 2)

    def test_daiminkan_excludes_exactly_one_occurrence_of_called_tile(self) -> None:
        meld = PublicMeld(
            kind=MeldKind.DAIMINKAN,
            tiles=tuple(_tile(SOUZU_9) for _ in range(4)),
            from_seat=Seat.SEAT_3,
            called_tile=_tile(SOUZU_9),
        )
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(melds=(meld,))
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))

        self.assertEqual(provenance.meld_hand_origin.tile_count(Wind.EAST, SOUZU_9), 3)

    def test_ankan_counts_all_four_tiles(self) -> None:
        meld = PublicMeld(
            kind=MeldKind.ANKAN,
            tiles=tuple(_tile(SOUZU_9) for _ in range(4)),
            from_seat=None,
            called_tile=None,
        )
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(melds=(meld,))
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))

        self.assertEqual(provenance.meld_hand_origin.tile_count(Wind.EAST, SOUZU_9), 4)

    def test_kakan_excludes_exactly_one_occurrence_of_called_tile(self) -> None:
        meld = PublicMeld(
            kind=MeldKind.KAKAN,
            tiles=tuple(_tile(SOUZU_9) for _ in range(4)),
            from_seat=Seat.SEAT_3,
            called_tile=_tile(SOUZU_9),
        )
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(melds=(meld,))
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))

        self.assertEqual(provenance.meld_hand_origin.tile_count(Wind.EAST, SOUZU_9), 3)

    def test_called_tile_is_not_double_counted_between_discard_and_meld(self) -> None:
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
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))

        self.assertEqual(provenance.discards.tile_count(Wind.EAST, PINZU_7), 1)
        self.assertEqual(provenance.meld_hand_origin.tile_count(Wind.SOUTH, PINZU_7), 2)
        self.assertEqual(
            sum(provenance.discards.flattened_tile_counts)
            + sum(provenance.meld_hand_origin.flattened_tile_counts),
            3,
        )

    def test_called_tile_is_red_five(self) -> None:
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(_tile(MANZU_5), _tile(MANZU_5), _tile(MANZU_5, is_red=True)),
            from_seat=Seat.SEAT_0,
            called_tile=_tile(MANZU_5, is_red=True),
        )
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(
            discards=(
                _discard(_tile(MANZU_5, is_red=True), order=0, called_by=Seat.SEAT_1),
            )
        )
        players[Seat.SEAT_1] = _player(melds=(pon,))
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))

        self.assertEqual(provenance.discards.tile_count(Wind.EAST, MANZU_5), 1)
        self.assertEqual(
            provenance.discards.red_five_count(Wind.EAST, TileCategory.MANZU), 1
        )
        self.assertEqual(provenance.meld_hand_origin.tile_count(Wind.SOUTH, MANZU_5), 2)
        self.assertEqual(
            provenance.meld_hand_origin.red_five_count(Wind.SOUTH, TileCategory.MANZU),
            0,
        )

    def test_called_tile_is_normal_five_with_red_five_hand_origin(self) -> None:
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(_tile(MANZU_5), _tile(MANZU_5), _tile(MANZU_5, is_red=True)),
            from_seat=Seat.SEAT_0,
            called_tile=_tile(MANZU_5),
        )
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(
            discards=(_discard(_tile(MANZU_5), order=0, called_by=Seat.SEAT_1),)
        )
        players[Seat.SEAT_1] = _player(melds=(pon,))
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))

        self.assertEqual(provenance.discards.tile_count(Wind.EAST, MANZU_5), 1)
        self.assertEqual(
            provenance.discards.red_five_count(Wind.EAST, TileCategory.MANZU), 0
        )
        self.assertEqual(provenance.meld_hand_origin.tile_count(Wind.SOUTH, MANZU_5), 2)
        self.assertEqual(
            provenance.meld_hand_origin.red_five_count(Wind.SOUTH, TileCategory.MANZU),
            1,
        )

    def test_ankan_red_five(self) -> None:
        meld = PublicMeld(
            kind=MeldKind.ANKAN,
            tiles=(
                _tile(MANZU_5),
                _tile(MANZU_5),
                _tile(MANZU_5),
                _tile(MANZU_5, is_red=True),
            ),
            from_seat=None,
            called_tile=None,
        )
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(melds=(meld,))
        provenance = encode_public_tile_provenance(_policy_input(tuple(players)))

        self.assertEqual(provenance.meld_hand_origin.tile_count(Wind.EAST, MANZU_5), 4)
        self.assertEqual(
            provenance.meld_hand_origin.red_five_count(Wind.EAST, TileCategory.MANZU), 1
        )


class DoraIndicatorProvenanceTest(unittest.TestCase):
    def test_normal_tile(self) -> None:
        provenance = encode_public_tile_provenance(
            _policy_input(_empty_players(), dora_indicators=(_tile(NORTH_WIND_TYPE),))
        )
        self.assertEqual(provenance.dora_indicators.tile_count(NORTH_WIND_TYPE), 1)

    def test_red_five(self) -> None:
        provenance = encode_public_tile_provenance(
            _policy_input(
                _empty_players(), dora_indicators=(_tile(MANZU_5, is_red=True),)
            )
        )
        self.assertEqual(provenance.dora_indicators.tile_count(MANZU_5), 1)
        self.assertEqual(
            provenance.dora_indicators.red_five_count(TileCategory.MANZU), 1
        )


class DeterminismTest(unittest.TestCase):
    def test_same_semantic_input_yields_same_result(self) -> None:
        meld = PublicMeld(
            kind=MeldKind.PON,
            tiles=(_tile(PINZU_7), _tile(PINZU_7), _tile(PINZU_7)),
            from_seat=Seat.SEAT_3,
            called_tile=_tile(PINZU_7),
        )
        players = list(_empty_players())
        players[Seat.SEAT_0] = _player(
            discards=(_discard(_tile(MANZU_3), order=0),), melds=(meld,)
        )
        policy_input = _policy_input(
            tuple(players), dora_indicators=(_tile(MANZU_5, is_red=True),)
        )

        first = encode_public_tile_provenance(policy_input)
        second = encode_public_tile_provenance(policy_input)
        self.assertEqual(first, second)


class PublicTileProvenanceTypeTest(unittest.TestCase):
    def test_rejects_wrong_field_types(self) -> None:
        empty = TileProvenanceCounts(
            tile_counts=tuple([0] * 34), red_five_counts=(0, 0, 0)
        )
        wind_empty = WindTileProvenanceCounts(winds=(empty, empty, empty, empty))
        with self.assertRaises(TypeError):
            PublicTileProvenance(
                discards=wind_empty,
                meld_hand_origin=wind_empty,
                dora_indicators=wind_empty,
            )


if __name__ == "__main__":
    unittest.main()
