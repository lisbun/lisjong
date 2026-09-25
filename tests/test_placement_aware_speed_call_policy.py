"""Issue #199 PlacementAwareSpeedCallPolicyのfocused deterministic tests。"""

import unittest

import lisjong.policies.placement_aware_speed_call as speed_call
from lisjong.hand_evaluation import calculate_shanten
from lisjong.hand_evaluation.shanten import calculate_restricted_standard_shanten
from lisjong.policies import (
    PlacementAwareSpeedCallPolicy,
    TargetedHonorReleaseTerminalProgressionPolicy,
)
from lisjong.policies.placement_aware_speed_call import (
    GameSituationMode,
    situation_mode,
)
from lisjong.policy_contract.action import (
    ChiAction,
    DiscardAction,
    PassAction,
    PonAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
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

_CATEGORIES = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}


def _tile(spec: str) -> Tile:
    rank = int(spec[0])
    return Tile(
        TileType(_CATEGORIES[spec[1]], 5 if rank == 0 else rank), is_red=rank == 0
    )


def _hand(spec: str) -> tuple[Tile, ...]:
    tiles: list[Tile] = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        tiles.extend(_tile(rank + character) for rank in ranks)
        ranks = ""
    if ranks:
        raise ValueError(f"hand spec has trailing ranks: {spec!r}")
    return tuple(tiles)


PASS = PassAction(actor=Seat.SEAT_0)


def _chi(called: str, consumed: str) -> ChiAction:
    first, second = _hand(consumed)
    return ChiAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_3,
        called_tile=_tile(called),
        consumed_tiles=(first, second),
    )


def _pon(tile: str) -> PonAction:
    called = _tile(tile)
    return PonAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_2,
        called_tile=called,
        consumed_tiles=(called, called),
    )


def _chi_meld(spec: str) -> PublicMeld:
    tiles = _hand(spec)
    return PublicMeld(
        kind=MeldKind.CHI, tiles=tiles, from_seat=Seat.SEAT_3, called_tile=tiles[0]
    )


def _pon_meld(tile: str) -> PublicMeld:
    called = _tile(tile)
    return PublicMeld(
        kind=MeldKind.PON,
        tiles=(called,) * 3,
        from_seat=Seat.SEAT_2,
        called_tile=called,
    )


def _player(
    *,
    score: int = 25000,
    melds: tuple[PublicMeld, ...] = (),
    riichi: RiichiState = RiichiState.NONE,
    discards: str = "",
) -> PlayerPublicState:
    return PlayerPublicState(
        score=score,
        discards=tuple(
            Discard(tile=tile, tsumogiri=False, order=order, called_by=None)
            for order, tile in enumerate(_hand(discards))
        ),
        melds=melds,
        riichi=riichi,
    )


def _input(
    concealed: str,
    *,
    own_melds: tuple[PublicMeld, ...] = (),
    drawn: str | None = None,
    scores: tuple[int, int, int, int] = (25000, 25000, 25000, 25000),
    round_wind: Wind = Wind.EAST,
    hand_number: int = 1,
    dealer_seat: Seat = Seat.SEAT_1,
    dora_indicators: str = "",
    riichi_discards: str | None = None,
) -> PolicyInput:
    players = [
        _player(score=scores[0], melds=own_melds),
        _player(score=scores[1]),
        _player(score=scores[2]),
        _player(score=scores[3]),
    ]
    if riichi_discards is not None:
        players[1] = _player(
            score=scores[1], riichi=RiichiState.ACCEPTED, discards=riichi_discards
        )
    concealed_tiles = _hand(concealed)
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=round_wind,
            hand_number=hand_number,
            dealer_seat=dealer_seat,
            honba=0,
            riichi_sticks=0,
            dora_indicators=_hand(dora_indicators),
            live_wall_tiles_remaining=50,
        ),
        players=tuple(players),  # type: ignore[arg-type]
        own_hand=OwnHandState(
            concealed_tiles=concealed_tiles,
            drawn_tile=None if drawn is None else _tile(drawn),
        ),
    )


def _discards(policy_input: PolicyInput) -> tuple[DiscardAction, ...]:
    seen: list[Tile] = []
    for tile in policy_input.own_hand.concealed_tiles:
        if tile not in seen:
            seen.append(tile)
    return tuple(
        DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=False) for tile in seen
    )


ALL_LAST_TOP = {
    "round_wind": Wind.SOUTH,
    "hand_number": 4,
    "scores": (40000, 25000, 20000, 15000),
}
ALL_LAST_FAR_LAST = {
    "round_wind": Wind.SOUTH,
    "hand_number": 4,
    "scores": (10000, 35000, 30000, 25000),
}

# 34m 678p 67p 9p 234s 55s: 1向聴。2mチー・打9pでタンヤオ聴牌。
TANYAO_ONE_SHANTEN = "34m667789p23455s"
# 23m 22p 567p 44s 67s 9s 1z: 2向聴。4sポンでタンヤオ1向聴（doraなしproxy 1）。
TANYAO_TWO_SHANTEN = "23m22567p44679s1z"


class RestrictedShantenTest(unittest.TestCase):
    def test_all_usable_matches_standard_shanten_for_open_hand_size(self) -> None:
        tiles = _hand("23m44s568p2679s")[:10]
        everything = frozenset(tile.tile_type for tile in tiles)
        self.assertEqual(
            calculate_restricted_standard_shanten(tiles, everything),
            calculate_shanten(tiles),
        )

    def test_excluded_tiles_cannot_form_blocks(self) -> None:
        tiles = _hand("789p234s55s67p")[:10] + _hand("1m")
        tanyao = speed_call._TANYAO_ROUTE.usable_tile_types
        self.assertEqual(calculate_shanten(tiles), 0)
        self.assertGreater(calculate_restricted_standard_shanten(tiles, tanyao), 0)

    def test_rejects_invalid_hand_size(self) -> None:
        with self.assertRaises(ValueError):
            calculate_restricted_standard_shanten(_hand("123m"), frozenset())


class SituationModeTest(unittest.TestCase):
    def test_non_all_last_is_normal_even_when_top(self) -> None:
        policy_input = _input(
            TANYAO_ONE_SHANTEN,
            round_wind=Wind.SOUTH,
            hand_number=3,
            scores=(40000, 20000, 20000, 20000),
        )
        self.assertIs(situation_mode(policy_input), GameSituationMode.NORMAL)

    def test_all_last_sole_top_is_speed(self) -> None:
        self.assertIs(
            situation_mode(_input(TANYAO_ONE_SHANTEN, **ALL_LAST_TOP)),
            GameSituationMode.ALL_LAST_TOP_SPEED,
        )

    def test_west_round_counts_as_all_last(self) -> None:
        policy_input = _input(
            TANYAO_ONE_SHANTEN,
            round_wind=Wind.WEST,
            hand_number=1,
            scores=(29000, 28000, 22000, 21000),
        )
        self.assertIs(
            situation_mode(policy_input), GameSituationMode.ALL_LAST_TOP_SPEED
        )

    def test_tied_top_is_normal(self) -> None:
        policy_input = _input(
            TANYAO_ONE_SHANTEN,
            round_wind=Wind.SOUTH,
            hand_number=4,
            scores=(30000, 30000, 20000, 20000),
        )
        self.assertIs(situation_mode(policy_input), GameSituationMode.NORMAL)

    def test_far_last_non_dealer_is_value(self) -> None:
        self.assertIs(
            situation_mode(_input(TANYAO_ONE_SHANTEN, **ALL_LAST_FAR_LAST)),
            GameSituationMode.ALL_LAST_LAST_VALUE,
        )

    def test_last_dealer_is_normal(self) -> None:
        policy_input = _input(
            TANYAO_ONE_SHANTEN, dealer_seat=Seat.SEAT_0, **ALL_LAST_FAR_LAST
        )
        self.assertIs(situation_mode(policy_input), GameSituationMode.NORMAL)

    def test_close_last_is_normal(self) -> None:
        policy_input = _input(
            TANYAO_ONE_SHANTEN,
            round_wind=Wind.SOUTH,
            hand_number=4,
            scores=(21000, 35000, 25000, 25000),
        )
        self.assertIs(situation_mode(policy_input), GameSituationMode.NORMAL)


class SpeedCallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PlacementAwareSpeedCallPolicy()
        self.parent = TargetedHonorReleaseTerminalProgressionPolicy()

    def _choose(self, concealed: str, actions: tuple[object, ...], **kwargs) -> object:
        decision = DecisionContext(
            input=_input(concealed, **kwargs), legal_actions=actions
        )
        self.assertIs(self.parent.choose_action(decision), PASS)
        return self.policy.choose_action(decision)

    def test_tanyao_chi_to_tenpai_is_taken(self) -> None:
        action = _chi("2m", "34m")
        self.assertIs(self._choose(TANYAO_ONE_SHANTEN, (PASS, action)), action)

    def test_call_without_compatible_route_is_rejected(self) -> None:
        terminal_chi = _chi("2m", "13m")
        self.assertIs(self._choose("13m667789p23455s", (PASS, terminal_chi)), PASS)

    def test_honitsu_pon_to_one_shanten_is_taken(self) -> None:
        action = _pon("9m")
        self.assertIs(self._choose("1235799m112233z", (PASS, action)), action)

    def test_opponent_riichi_suppresses_speed_call(self) -> None:
        action = _chi("2m", "34m")
        self.assertIs(
            self._choose(TANYAO_ONE_SHANTEN, (PASS, action), riichi_discards="1z9s"),
            PASS,
        )

    def test_cheap_call_to_one_shanten_is_rejected_in_normal_mode(self) -> None:
        self.assertIs(self._choose(TANYAO_TWO_SHANTEN, (PASS, _pon("4s"))), PASS)

    def test_valued_call_to_one_shanten_is_taken_in_normal_mode(self) -> None:
        action = _pon("4s")
        self.assertIs(
            self._choose(TANYAO_TWO_SHANTEN, (PASS, action), dora_indicators="3s"),
            action,
        )

    def test_cheap_call_to_one_shanten_is_taken_when_top_at_all_last(self) -> None:
        action = _pon("4s")
        self.assertIs(
            self._choose(TANYAO_TWO_SHANTEN, (PASS, action), **ALL_LAST_TOP), action
        )

    def test_cheap_tenpai_call_is_rejected_when_far_last_at_all_last(self) -> None:
        action = _chi("2m", "34m")
        self.assertIs(
            self._choose(TANYAO_ONE_SHANTEN, (PASS, action), **ALL_LAST_FAR_LAST),
            PASS,
        )

    def test_non_improving_call_is_rejected(self) -> None:
        # 既に聴牌しているのでstrict improvementにならない。
        action = _chi("2m", "34m")
        self.assertIs(self._choose("34m678p234s55678s", (PASS, action)), PASS)


class RoutePreservingDiscardTest(unittest.TestCase):
    def test_open_tanyao_hand_discards_route_breaking_tiles(self) -> None:
        # 234m chi済み。789pは構造上完成面子だがtanyao routeには使えない。
        policy_input = _input(
            "789p234s55s67p1z", own_melds=(_chi_meld("234m"),), drawn="1z"
        )
        eligible = speed_call._route_preserving_actions(
            policy_input, _discards(policy_input)
        )
        route_junk = {_tile("9p"), _tile("1z")}
        self.assertEqual({action.tile for action in eligible}, route_junk)
        decision = DecisionContext(
            input=policy_input, legal_actions=_discards(policy_input)
        )
        self.assertIn(
            PlacementAwareSpeedCallPolicy().choose_action(decision).tile,  # type: ignore[union-attr]
            route_junk,
        )

    def test_yakuhai_open_hand_is_not_filtered(self) -> None:
        policy_input = _input(
            "789p234s55s67p1z", own_melds=(_pon_meld("5z"),), drawn="1z"
        )
        actions = _discards(policy_input)
        self.assertEqual(
            speed_call._route_preserving_actions(policy_input, actions), actions
        )

    def test_closed_hand_is_not_filtered(self) -> None:
        policy_input = _input("789p234s55s67p123m1z", drawn="1z")
        actions = _discards(policy_input)
        self.assertEqual(
            speed_call._route_preserving_actions(policy_input, actions), actions
        )

    def test_fold_branch_is_not_filtered(self) -> None:
        policy_input = _input(
            "789p24s55s68p19m",
            own_melds=(_chi_meld("234m"),),
            drawn="1m",
            riichi_discards="9p1z",
        )
        actions = _discards(policy_input)
        self.assertEqual(
            speed_call._route_preserving_actions(policy_input, actions), actions
        )


class AllLastTopFoldTest(unittest.TestCase):
    CONCEALED = "34m678p68p234s55s9s7z"  # 1向聴、共通現物なし

    def test_top_fold_filters_to_minimum_danger_when_not_tenpai(self) -> None:
        policy_input = _input(
            self.CONCEALED, drawn="7z", riichi_discards="3z4z", **ALL_LAST_TOP
        )
        actions = _discards(policy_input)
        eligible = speed_call._top_fold_actions(policy_input, actions)
        self.assertLess(len(eligible), len(actions))
        selected = PlacementAwareSpeedCallPolicy().choose_action(
            DecisionContext(input=policy_input, legal_actions=actions)
        )
        self.assertIn(selected, eligible)

    def test_no_filter_without_riichi(self) -> None:
        policy_input = _input(self.CONCEALED, drawn="7z", **ALL_LAST_TOP)
        actions = _discards(policy_input)
        self.assertEqual(speed_call._top_fold_actions(policy_input, actions), actions)

    def test_no_filter_when_common_genbutsu_exists(self) -> None:
        policy_input = _input(
            self.CONCEALED, drawn="7z", riichi_discards="9s3z", **ALL_LAST_TOP
        )
        actions = _discards(policy_input)
        self.assertEqual(speed_call._top_fold_actions(policy_input, actions), actions)

    def test_normal_mode_keeps_parent_discard(self) -> None:
        policy_input = _input(self.CONCEALED, drawn="7z", riichi_discards="3z4z")
        decision = DecisionContext(
            input=policy_input, legal_actions=_discards(policy_input)
        )
        self.assertEqual(
            PlacementAwareSpeedCallPolicy().choose_action(decision),
            TargetedHonorReleaseTerminalProgressionPolicy().choose_action(decision),
        )


if __name__ == "__main__":
    unittest.main()
