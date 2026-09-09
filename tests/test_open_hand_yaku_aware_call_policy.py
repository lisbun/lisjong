"""Issue #157 OpenHandYakuAwareCallPolicyのfocused deterministic tests。"""

import ast
import inspect
import itertools
import pickle
import unittest
from unittest.mock import patch

import lisjong.policies.open_hand_yaku_aware_call as open_yaku_call
from lisjong.policies import (
    OpenHandYakuAwareCallPolicy,
    YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
)
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _yaku_route_value_for_tiles,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    _best_post_call_shanten,
    _post_call_stable_hands,
)
from lisjong.policy_contract.action import (
    ChiAction,
    DiscardAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
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


def _tile(category: TileCategory, rank: int, *, red: bool = False) -> Tile:
    return Tile(TileType(category, rank), is_red=red)


def _hand(spec: str) -> tuple[Tile, ...]:
    categories = {
        "m": TileCategory.MANZU,
        "p": TileCategory.PINZU,
        "s": TileCategory.SOUZU,
        "z": TileCategory.HONOR,
    }
    tiles: list[Tile] = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        category = categories[character]
        for rank_character in ranks:
            rank = int(rank_character)
            tiles.append(_tile(category, 5 if rank == 0 else rank, red=rank == 0))
        ranks = ""
    if ranks:
        raise ValueError(f"hand spec has trailing ranks: {spec!r}")
    return tuple(tiles)


MANZU_1 = _tile(TileCategory.MANZU, 1)
MANZU_2 = _tile(TileCategory.MANZU, 2)
MANZU_3 = _tile(TileCategory.MANZU, 3)
MANZU_4 = _tile(TileCategory.MANZU, 4)
MANZU_5 = _tile(TileCategory.MANZU, 5)
MANZU_5_RED = _tile(TileCategory.MANZU, 5, red=True)
MANZU_6 = _tile(TileCategory.MANZU, 6)
PINZU_2 = _tile(TileCategory.PINZU, 2)
PINZU_9 = _tile(TileCategory.PINZU, 9)
SOUZU_7 = _tile(TileCategory.SOUZU, 7)
WHITE = _tile(TileCategory.HONOR, 5)
PASS = PassAction(actor=Seat.SEAT_0)


def _meld(kind: MeldKind, tile: Tile) -> PublicMeld:
    return PublicMeld(
        kind=kind,
        tiles=(tile,) * 3,
        from_seat=Seat.SEAT_1,
        called_tile=tile,
    )


OPEN_WHITE = (_meld(MeldKind.PON, WHITE),)


def _discard_history(tile: Tile, order: int) -> Discard:
    return Discard(tile=tile, tsumogiri=False, order=order, called_by=None)


def _player(
    *,
    melds: tuple[PublicMeld, ...] = (),
    riichi: RiichiState = RiichiState.NONE,
    discards: tuple[Discard, ...] = (),
) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=discards,
        melds=melds,
        riichi=riichi,
    )


def _input(
    concealed_tiles: tuple[Tile, ...],
    *,
    own_melds: tuple[PublicMeld, ...] = (),
    threat_discards: tuple[tuple[Tile, ...], ...] = (),
) -> PolicyInput:
    players = [_player(melds=own_melds), _player(), _player(), _player()]
    for offset, discards in enumerate(threat_discards, start=1):
        players[offset] = _player(
            riichi=RiichiState.ACCEPTED,
            discards=tuple(
                _discard_history(tile, order) for order, tile in enumerate(discards)
            ),
        )
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(),
            live_wall_tiles_remaining=60,
        ),
        players=tuple(players),  # type: ignore[arg-type]
        own_hand=OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=None),
    )


def _decision(
    concealed_tiles: tuple[Tile, ...],
    actions: tuple[object, ...],
    **input_kwargs,
) -> DecisionContext:
    return DecisionContext(
        input=_input(concealed_tiles, **input_kwargs),
        legal_actions=actions,
    )


def _chi(called: Tile, consumed: tuple[Tile, Tile]) -> ChiAction:
    return ChiAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_3,
        called_tile=called,
        consumed_tiles=consumed,
    )


def _pon(tile: Tile) -> PonAction:
    return PonAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_1,
        called_tile=tile,
        consumed_tiles=(tile, tile),
    )


class SupportedRouteCallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = OpenHandYakuAwareCallPolicy()

    def test_tanyao_route_allows_initial_chi(self) -> None:
        action = _chi(MANZU_2, (MANZU_3, MANZU_4))
        concealed = _hand("223467m22566p37s")
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (PASS, action))), action
        )

    def test_tanyao_route_allows_initial_pon(self) -> None:
        action = _pon(MANZU_2)
        concealed = _hand("223444m23466p57s")
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (action, PASS))), action
        )

    def test_honitsu_route_allows_initial_chi(self) -> None:
        action = _chi(MANZU_2, (MANZU_3, MANZU_4))
        concealed = _hand("11134468m12267z")
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (PASS, action))), action
        )

    def test_honitsu_route_allows_initial_pon(self) -> None:
        action = _pon(MANZU_2)
        concealed = _hand("112247789m1777z")
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (action, PASS))), action
        )

    def test_chinitsu_route_allows_initial_chi(self) -> None:
        action = _chi(MANZU_2, (MANZU_3, MANZU_4))
        concealed = _hand("2334677778899m")
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (action, PASS))), action
        )

    def test_chinitsu_route_allows_initial_pon(self) -> None:
        action = _pon(MANZU_2)
        concealed = _hand("1223337777889m")
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (action, PASS))), action
        )


class RouteAndShantenRejectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = OpenHandYakuAwareCallPolicy()
        self.action = _chi(MANZU_2, (MANZU_3, MANZU_4))

    def test_no_supported_route_matches_yakuhai_call_pass(self) -> None:
        concealed = _hand("1222347m69p37s12z")
        decision = _decision(concealed, (self.action, PASS))
        baseline = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()
        self.assertIs(baseline.choose_action(decision), PASS)
        self.assertIs(self.policy.choose_action(decision), PASS)
        self.assertLess(
            _best_post_call_shanten(concealed, self.action),
            open_yaku_call.calculate_shanten(concealed),
        )

    def test_same_shanten_route_is_passed(self) -> None:
        concealed = _hand("2344556677889m")
        self.assertEqual(
            _best_post_call_shanten(concealed, self.action),
            open_yaku_call.calculate_shanten(concealed),
        )
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (self.action, PASS))),
            PASS,
        )

    def test_mixed_route_preserving_and_breaking_discards_reject_call(self) -> None:
        concealed = _hand("34458m2348p266s6z")
        stable_hands = _post_call_stable_hands(concealed, self.action)
        new_meld_tiles = (*self.action.consumed_tiles, self.action.called_tile)
        current_shanten = open_yaku_call.calculate_shanten(concealed)
        stable_evaluations = tuple(
            (
                open_yaku_call.calculate_shanten(stable_hand),
                _yaku_route_value_for_tiles((*stable_hand, *new_meld_tiles)),
            )
            for stable_hand in stable_hands
        )

        self.assertTrue(
            any(
                post_call_shanten < current_shanten and route_value > 0
                for post_call_shanten, route_value in stable_evaluations
            )
        )
        self.assertTrue(any(route_value == 0 for _, route_value in stable_evaluations))
        self.assertIsNone(
            open_yaku_call._route_compatible_post_call_shanten(
                _input(concealed), self.action, current_shanten
            )
        )
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (self.action, PASS))),
            PASS,
        )

    def test_two_terminals_prevent_tanyao_and_flush_routes(self) -> None:
        concealed = _hand("2234m12345p456s9s")
        with patch.object(open_yaku_call, "calculate_shanten") as shanten:
            shanten.side_effect = lambda tiles: 2 if len(tuple(tiles)) == 13 else 1
            self.assertIs(
                self.policy.choose_action(_decision(concealed, (self.action, PASS))),
                PASS,
            )

    def test_two_honors_prevent_tanyao_and_flush_routes(self) -> None:
        concealed = _hand("2234m2345p456s15z")
        with patch.object(open_yaku_call, "calculate_shanten") as shanten:
            shanten.side_effect = lambda tiles: 2 if len(tuple(tiles)) == 13 else 1
            self.assertIs(
                self.policy.choose_action(_decision(concealed, (PASS, self.action))),
                PASS,
            )

    def test_second_numbered_suit_prevents_honitsu_route(self) -> None:
        concealed = _hand("1234m234456p115z")
        with patch.object(open_yaku_call, "calculate_shanten") as shanten:
            shanten.side_effect = lambda tiles: 2 if len(tuple(tiles)) == 13 else 1
            self.assertIs(
                self.policy.choose_action(_decision(concealed, (self.action, PASS))),
                PASS,
            )

    def test_incompatible_existing_meld_prevents_a_flush_route(self) -> None:
        concealed = _hand("2334677778899m")
        incompatible_meld = _meld(MeldKind.PON, PINZU_9)
        self.assertIs(
            self.policy.choose_action(
                _decision(
                    concealed,
                    (self.action, PASS),
                    own_melds=(incompatible_meld,),
                )
            ),
            PASS,
        )

    def test_new_terminal_call_meld_prevents_tanyao_route(self) -> None:
        action = _chi(MANZU_1, (MANZU_2, MANZU_3))
        concealed = _hand("223m23445p45678s")
        with patch.object(open_yaku_call, "calculate_shanten") as shanten:
            shanten.side_effect = lambda tiles: 2 if len(tuple(tiles)) == 13 else 1
            self.assertIs(
                self.policy.choose_action(_decision(concealed, (action, PASS))),
                PASS,
            )


class BaselinePreservationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = OpenHandYakuAwareCallPolicy()
        self.baseline = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()

    def test_existing_initial_yakuhai_pon_is_unchanged(self) -> None:
        action = _pon(WHITE)
        concealed = _hand("123456m789p19s") + (WHITE, WHITE)
        decision = _decision(concealed, (PASS, action))
        self.assertIs(self.baseline.choose_action(decision), action)
        self.assertIs(self.policy.choose_action(decision), action)

    def test_existing_post_yakuhai_open_chi_is_unchanged(self) -> None:
        action = _chi(MANZU_1, (MANZU_2, MANZU_3))
        decision = _decision(
            _hand("23m123456p19s"),
            (action, PASS),
            own_melds=OPEN_WHITE,
        )
        self.assertIs(self.baseline.choose_action(decision), action)
        self.assertIs(self.policy.choose_action(decision), action)

    def test_existing_post_yakuhai_open_pon_is_unchanged(self) -> None:
        action = _pon(MANZU_2)
        decision = _decision(
            _hand("22m123456p19s"),
            (PASS, action),
            own_melds=OPEN_WHITE,
        )
        self.assertIs(self.baseline.choose_action(decision), action)
        self.assertIs(self.policy.choose_action(decision), action)

    def test_non_improving_yakuhai_pon_is_still_passed(self) -> None:
        action = _pon(WHITE)
        decision = _decision(_hand("123456m789p12s") + (WHITE, WHITE), (action, PASS))
        self.assertIs(self.baseline.choose_action(decision), PASS)
        self.assertIs(self.policy.choose_action(decision), PASS)

    def test_winning_action_is_parent_equivalent(self) -> None:
        ron = RonAction(Seat.SEAT_0, Seat.SEAT_1, MANZU_2)
        action = _pon(MANZU_2)
        decision = _decision(_hand("223444m23466p57s"), (action, PASS, ron))
        self.assertIs(self.baseline.choose_action(decision), ron)
        self.assertIs(self.policy.choose_action(decision), ron)

    def test_riichi_action_is_parent_equivalent(self) -> None:
        riichi = RiichiAction(actor=Seat.SEAT_0)
        decision = _decision((), (PASS, riichi))
        self.assertIs(self.baseline.choose_action(decision), riichi)
        self.assertIs(self.policy.choose_action(decision), riichi)

    def test_ordinary_discard_is_parent_equivalent(self) -> None:
        discard_1 = DiscardAction(Seat.SEAT_0, MANZU_1, False)
        discard_9 = DiscardAction(Seat.SEAT_0, PINZU_9, False)
        decision = _decision(_hand("123456m789p11s"), (discard_1, discard_9))
        with patch.object(
            open_yaku_call,
            "_route_compatible_post_call_shanten",
            side_effect=AssertionError(
                "actual discard must stay a fresh baseline decision"
            ),
        ):
            self.assertEqual(
                self.policy.choose_action(decision),
                self.baseline.choose_action(decision),
            )


class StructuralAndDefenseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = OpenHandYakuAwareCallPolicy()

    def test_supported_route_does_not_bypass_defense_gate(self) -> None:
        action = _chi(MANZU_2, (MANZU_3, MANZU_4))
        concealed = _hand("223467m22566p37s")
        selected = self.policy.choose_action(
            _decision(
                concealed,
                (action, PASS),
                threat_discards=((SOUZU_7,),),
            )
        )
        self.assertIs(selected, PASS)

    def test_multiple_riichi_keep_common_genbutsu_suppression(self) -> None:
        action = _chi(MANZU_2, (MANZU_3, MANZU_4))
        concealed = _hand("223467m22566p37s")
        selected = self.policy.choose_action(
            _decision(
                concealed,
                (PASS, action),
                threat_discards=((SOUZU_7, MANZU_1), (SOUZU_7, PINZU_9)),
            )
        )
        self.assertIs(selected, PASS)

    def test_concealed_triplet_is_not_broken_for_any_supported_route(self) -> None:
        fixtures = {
            "tanyao": _hand("222345m23466p57s"),
            "honitsu": _hand("122234789m1777z"),
            "chinitsu": _hand("1222337777889m"),
        }
        action = _pon(MANZU_2)
        for route, concealed in fixtures.items():
            with self.subTest(route=route):
                self.assertEqual(
                    sum(tile.tile_type == MANZU_2.tile_type for tile in concealed),
                    3,
                )
                with patch.object(
                    open_yaku_call,
                    "_route_compatible_post_call_shanten",
                    side_effect=AssertionError("triplet Pon must be rejected first"),
                ):
                    self.assertIs(
                        self.policy.choose_action(_decision(concealed, (action, PASS))),
                        PASS,
                    )

    def test_kuikae_forbidden_discard_cannot_create_false_improvement(self) -> None:
        action = _chi(MANZU_3, (MANZU_4, MANZU_5))
        concealed = (MANZU_4, MANZU_5, MANZU_6, PINZU_2)
        observed: list[tuple[Tile, ...]] = []

        def shanten(tiles) -> int:
            materialized = tuple(tiles)
            observed.append(materialized)
            if len(materialized) == 4:
                return 1
            return 0 if materialized == (PINZU_2,) else 1

        with patch.object(open_yaku_call, "calculate_shanten", side_effect=shanten):
            selected = self.policy.choose_action(_decision(concealed, (action, PASS)))

        self.assertIs(selected, PASS)
        self.assertIn((MANZU_6,), observed)
        self.assertNotIn((PINZU_2,), observed)

    def test_exact_consumed_identity_and_red_distinction_are_reused(self) -> None:
        action = _chi(MANZU_6, (MANZU_4, MANZU_5))
        concealed = (MANZU_5_RED, MANZU_5, MANZU_4, PINZU_2)
        stable_hands = _post_call_stable_hands(concealed, action)
        self.assertTrue(stable_hands)
        self.assertIn((MANZU_5_RED,), stable_hands)
        self.assertNotIn((MANZU_5,), stable_hands)

    def test_legal_action_order_does_not_change_selection(self) -> None:
        canonical = _chi(MANZU_2, (MANZU_3, MANZU_4))
        other = _chi(MANZU_5, (MANZU_3, MANZU_4))
        concealed = _hand("223467m22566p37s")
        with (
            patch.object(open_yaku_call, "calculate_shanten", return_value=2),
            patch.object(
                open_yaku_call,
                "_route_compatible_post_call_shanten",
                return_value=1,
            ),
        ):
            selected = {
                self.policy.choose_action(_decision(concealed, actions))
                for actions in itertools.permutations((canonical, other, PASS))
            }
        self.assertEqual(selected, {canonical})

    def test_minimum_post_call_shanten_is_hard_priority(self) -> None:
        canonical = _chi(MANZU_2, (MANZU_3, MANZU_4))
        lower_shanten = _chi(MANZU_5, (MANZU_3, MANZU_4))
        concealed = _hand("223467m22566p37s")
        post_call_shanten = {canonical: 1, lower_shanten: 0}
        with (
            patch.object(open_yaku_call, "calculate_shanten", return_value=2),
            patch.object(
                open_yaku_call,
                "_route_compatible_post_call_shanten",
                side_effect=lambda policy_input, action, current: post_call_shanten[
                    action
                ],
            ),
        ):
            selected = self.policy.choose_action(
                _decision(concealed, (canonical, PASS, lower_shanten))
            )
        self.assertIs(selected, lower_shanten)


class ReuseAndInformationBoundaryTest(unittest.TestCase):
    def test_route_helper_preserves_existing_values(self) -> None:
        self.assertEqual(_yaku_route_value_for_tiles(_hand("234m456p678s")), 1)
        self.assertEqual(_yaku_route_value_for_tiles(_hand("123456789p11z")), 2)
        self.assertEqual(_yaku_route_value_for_tiles(_hand("123456789p")), 3)

    def test_public_export_pickle_statelessness_and_dependency_boundary(self) -> None:
        from lisjong.policies import OpenHandYakuAwareCallPolicy as imported

        self.assertIs(imported, open_yaku_call.OpenHandYakuAwareCallPolicy)
        self.assertIs(
            pickle.loads(pickle.dumps(OpenHandYakuAwareCallPolicy)),
            OpenHandYakuAwareCallPolicy,
        )
        self.assertEqual(vars(OpenHandYakuAwareCallPolicy()), {})
        tree = ast.parse(inspect.getsource(open_yaku_call))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertFalse(
            any(
                module.startswith(prefix)
                for module in imported_modules
                for prefix in (
                    "lisjong_engine",
                    "lisjong_arena",
                    "riichienv",
                    "mahjong",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
