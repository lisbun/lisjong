"""Issue #161 CheapFarGuardOpenHandYakuAwareCallPolicyのfocused deterministic tests。"""

import ast
import inspect
import itertools
import pickle
import unittest
from unittest.mock import patch

import lisjong.policies.cheap_far_guard_open_hand_yaku_aware_call as cheap_far_guard
from lisjong.policies import (
    CheapFarGuardOpenHandYakuAwareCallPolicy,
    OpenHandYakuAwareCallPolicy,
)
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _yaku_route_value_for_tiles,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    YakuhaiCallPolicyError,
)
from lisjong.policy_contract.action import (
    AnkanAction,
    ChiAction,
    DiscardAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
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


MANZU_2 = _tile(TileCategory.MANZU, 2)
MANZU_3 = _tile(TileCategory.MANZU, 3)
MANZU_4 = _tile(TileCategory.MANZU, 4)
MANZU_5_RED = _tile(TileCategory.MANZU, 5, red=True)
PINZU_1 = _tile(TileCategory.PINZU, 1)
PINZU_2 = _tile(TileCategory.PINZU, 2)
PINZU_9 = _tile(TileCategory.PINZU, 9)
PASS = PassAction(actor=Seat.SEAT_0)

# Conservative all-stable-hands fixtures for CheapFarEvaluationBoundaryTest.
# Each hand's real `calculate_shanten()` / `_current_visible_value_proxy()`
# result is asserted directly by that test class via `_evaluate()`, so the
# expected far/cheap outcome below is never an unchecked magic number.
HAND_SHANTEN_2_CHEAP = (
    _tile(TileCategory.MANZU, 4),
    _tile(TileCategory.MANZU, 6),
    _tile(TileCategory.MANZU, 6),
    _tile(TileCategory.MANZU, 8),
    _tile(TileCategory.PINZU, 2),
    _tile(TileCategory.PINZU, 5),
    _tile(TileCategory.PINZU, 5),
    _tile(TileCategory.PINZU, 6),
    _tile(TileCategory.PINZU, 6),
    _tile(TileCategory.PINZU, 8),
)
HAND_SHANTEN_1_CHEAP = (
    _tile(TileCategory.MANZU, 4),
    _tile(TileCategory.PINZU, 2),
    _tile(TileCategory.PINZU, 4),
    _tile(TileCategory.PINZU, 6),
    _tile(TileCategory.PINZU, 6),
    _tile(TileCategory.SOUZU, 2),
    _tile(TileCategory.SOUZU, 3),
    _tile(TileCategory.SOUZU, 4),
    _tile(TileCategory.SOUZU, 7),
    _tile(TileCategory.SOUZU, 8),
)
HAND_SHANTEN_2_EXPENSIVE = tuple(
    _tile(TileCategory.MANZU, rank) for rank in (2, 2, 3, 3, 5, 6, 6, 8, 8, 9)
)
HAND_SHANTEN_3_CHEAP = (
    _tile(TileCategory.MANZU, 3),
    _tile(TileCategory.MANZU, 6),
    _tile(TileCategory.MANZU, 7),
    _tile(TileCategory.PINZU, 2),
    _tile(TileCategory.PINZU, 3),
    _tile(TileCategory.PINZU, 7),
    _tile(TileCategory.PINZU, 8),
    _tile(TileCategory.SOUZU, 2),
    _tile(TileCategory.SOUZU, 5),
    _tile(TileCategory.SOUZU, 6),
)


def _player() -> PlayerPublicState:
    return PlayerPublicState(
        score=25000, discards=(), melds=(), riichi=RiichiState.NONE
    )


def _input(concealed_tiles: tuple[Tile, ...]) -> PolicyInput:
    # East dealer at seat 0 (seat_wind_rank=1, round_wind_rank=2), matching
    # the seat/round assumptions used throughout ValueProxyTest.
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.SOUTH,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(),
            live_wall_tiles_remaining=60,
        ),
        players=(_player(), _player(), _player(), _player()),
        own_hand=OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=None),
    )


def _decision(
    concealed_tiles: tuple[Tile, ...], actions: tuple[object, ...]
) -> DecisionContext:
    return DecisionContext(
        input=_input(concealed_tiles),
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


# Real fixtures shared with test_open_hand_yaku_aware_call_policy.py's
# SupportedRouteCallTest. Their post-call shanten/value are exercised for
# real (no mocking) below, and their exact properties are pinned by
# RealFixturePropertyTest.
TANYAO_CHI_CONCEALED = _hand("223467m22566p37s")
TANYAO_CHI_ACTION = _chi(MANZU_2, (MANZU_3, MANZU_4))
TANYAO_PON_CONCEALED = _hand("223444m23466p57s")
TANYAO_PON_ACTION = _pon(MANZU_2)
HONITSU_CHI_CONCEALED = _hand("11134468m12267z")
HONITSU_CHI_ACTION = _chi(MANZU_2, (MANZU_3, MANZU_4))
HONITSU_PON_CONCEALED = _hand("112247789m1777z")
HONITSU_PON_ACTION = _pon(MANZU_2)

# TANYAO_CHI_CONCEALED also contains a concealed MANZU_2 pair, so the exact
# same hand additionally qualifies a real cheap+far *Pon* of MANZU_2 (as
# opposed to the Chi above). This pins the Pon side of the Required tests'
# "selected cheap+far Pon -> legal Pass" case with real parent selection,
# mirroring the Chi coverage above rather than only the boundary-level
# `_evaluate_cheap_far_call()` unit tests.
TANYAO_PON_FAR_CONCEALED = TANYAO_CHI_CONCEALED
TANYAO_PON_FAR_ACTION = _pon(MANZU_2)


class RealFixturePropertyTest(unittest.TestCase):
    """boundary/suppression testsが依拠する実手牌のshanten/valueを固定する。"""

    def test_tanyao_chi_is_two_shanten_and_one_han_equivalent(self) -> None:
        stable_hands = cheap_far_guard._post_call_stable_hands(
            TANYAO_CHI_CONCEALED, TANYAO_CHI_ACTION
        )
        shantens = [cheap_far_guard.calculate_shanten(h) for h in stable_hands]
        values = [
            cheap_far_guard._current_visible_value_proxy(
                h, (), seat_wind_rank=1, round_wind_rank=2
            )
            for h in stable_hands
        ]
        self.assertGreaterEqual(min(shantens), 2)
        self.assertLess(max(values), 3)

    def test_tanyao_pon_is_zero_shanten(self) -> None:
        stable_hands = cheap_far_guard._post_call_stable_hands(
            TANYAO_PON_CONCEALED, TANYAO_PON_ACTION
        )
        shantens = [cheap_far_guard.calculate_shanten(h) for h in stable_hands]
        self.assertLess(min(shantens), 2)

    def test_honitsu_chi_is_two_shanten_and_two_han_equivalent(self) -> None:
        stable_hands = cheap_far_guard._post_call_stable_hands(
            HONITSU_CHI_CONCEALED, HONITSU_CHI_ACTION
        )
        shantens = [cheap_far_guard.calculate_shanten(h) for h in stable_hands]
        values = [
            cheap_far_guard._current_visible_value_proxy(
                h, (), seat_wind_rank=1, round_wind_rank=2
            )
            for h in stable_hands
        ]
        self.assertGreaterEqual(min(shantens), 2)
        self.assertLess(max(values), 3)

    def test_honitsu_pon_is_one_shanten(self) -> None:
        stable_hands = cheap_far_guard._post_call_stable_hands(
            HONITSU_PON_CONCEALED, HONITSU_PON_ACTION
        )
        shantens = [cheap_far_guard.calculate_shanten(h) for h in stable_hands]
        self.assertLess(min(shantens), 2)

    def test_tanyao_pon_far_is_two_shanten_and_one_han_equivalent(self) -> None:
        stable_hands = cheap_far_guard._post_call_stable_hands(
            TANYAO_PON_FAR_CONCEALED, TANYAO_PON_FAR_ACTION
        )
        shantens = [cheap_far_guard.calculate_shanten(h) for h in stable_hands]
        values = [
            cheap_far_guard._current_visible_value_proxy(
                h, (), seat_wind_rank=1, round_wind_rank=2
            )
            for h in stable_hands
        ]
        self.assertGreaterEqual(min(shantens), 2)
        self.assertLess(max(values), 3)


class BaselinePreservationTest(unittest.TestCase):
    """parent `OpenHandYakuAwareCallPolicy`のwinning/Riichi/discard/Kanをそのまま返す。"""

    def setUp(self) -> None:
        self.policy = CheapFarGuardOpenHandYakuAwareCallPolicy()
        self.baseline = OpenHandYakuAwareCallPolicy()

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
        discard_1 = DiscardAction(Seat.SEAT_0, MANZU_2, False)
        discard_9 = DiscardAction(Seat.SEAT_0, PINZU_9, False)
        decision = _decision(_hand("223456m789p11s"), (discard_1, discard_9))
        self.assertEqual(
            self.policy.choose_action(decision),
            self.baseline.choose_action(decision),
        )

    def test_plain_pass_is_parent_equivalent(self) -> None:
        decision = _decision(_hand("123456m789p11s"), (PASS,))
        self.assertIs(self.baseline.choose_action(decision), PASS)
        self.assertIs(self.policy.choose_action(decision), PASS)

    def test_kan_legal_action_is_parent_equivalent(self) -> None:
        concealed = (MANZU_2,) * 4
        ankan = AnkanAction(actor=Seat.SEAT_0, tiles=concealed)
        decision = _decision(concealed, (ankan, PASS))
        self.assertIs(self.baseline.choose_action(decision), PASS)
        self.assertIs(self.policy.choose_action(decision), PASS)

    def test_non_qualifying_call_pass_is_parent_equivalent(self) -> None:
        action = _chi(MANZU_2, (MANZU_3, MANZU_4))
        concealed = _hand("1222347m69p37s12z")
        decision = _decision(concealed, (action, PASS))
        self.assertIs(self.baseline.choose_action(decision), PASS)
        self.assertIs(self.policy.choose_action(decision), PASS)


class RealCallSuppressionTest(unittest.TestCase):
    """実手牌上でcheap AND farのselected callだけをPassへ置換する。"""

    def setUp(self) -> None:
        self.policy = CheapFarGuardOpenHandYakuAwareCallPolicy()
        self.baseline = OpenHandYakuAwareCallPolicy()

    def test_two_shanten_one_han_equivalent_chi_is_suppressed(self) -> None:
        decision = _decision(TANYAO_CHI_CONCEALED, (PASS, TANYAO_CHI_ACTION))
        self.assertIs(self.baseline.choose_action(decision), TANYAO_CHI_ACTION)
        self.assertIs(self.policy.choose_action(decision), PASS)

    def test_two_shanten_two_han_equivalent_chi_is_suppressed(self) -> None:
        decision = _decision(HONITSU_CHI_CONCEALED, (PASS, HONITSU_CHI_ACTION))
        self.assertIs(self.baseline.choose_action(decision), HONITSU_CHI_ACTION)
        self.assertIs(self.policy.choose_action(decision), PASS)

    def test_two_shanten_one_han_equivalent_pon_is_suppressed(self) -> None:
        decision = _decision(TANYAO_PON_FAR_CONCEALED, (PASS, TANYAO_PON_FAR_ACTION))
        self.assertIs(self.baseline.choose_action(decision), TANYAO_PON_FAR_ACTION)
        self.assertIs(self.policy.choose_action(decision), PASS)

    def test_zero_shanten_cheap_pon_is_not_suppressed(self) -> None:
        decision = _decision(TANYAO_PON_CONCEALED, (TANYAO_PON_ACTION, PASS))
        self.assertIs(self.baseline.choose_action(decision), TANYAO_PON_ACTION)
        self.assertIs(self.policy.choose_action(decision), TANYAO_PON_ACTION)

    def test_one_shanten_cheap_pon_is_not_suppressed(self) -> None:
        decision = _decision(HONITSU_PON_CONCEALED, (HONITSU_PON_ACTION, PASS))
        self.assertIs(self.baseline.choose_action(decision), HONITSU_PON_ACTION)
        self.assertIs(self.policy.choose_action(decision), HONITSU_PON_ACTION)

    def test_suppressed_decision_carries_no_analysis(self) -> None:
        decision = _decision(TANYAO_CHI_CONCEALED, (PASS, TANYAO_CHI_ACTION))
        result = self.policy.choose_action_with_analysis(decision)
        self.assertIs(result.action, PASS)


class NoAlternateCallRescueTest(unittest.TestCase):
    def test_suppressed_call_does_not_fall_back_to_an_alternate_call(self) -> None:
        alternate = _pon(PINZU_2)
        decision = _decision(
            TANYAO_CHI_CONCEALED,
            (PASS, TANYAO_CHI_ACTION, alternate),
        )
        baseline = OpenHandYakuAwareCallPolicy()
        self.assertIs(baseline.choose_action(decision), TANYAO_CHI_ACTION)

        candidate = CheapFarGuardOpenHandYakuAwareCallPolicy()
        selected = candidate.choose_action(decision)
        self.assertIs(selected, PASS)
        self.assertNotIsInstance(selected, (ChiAction, PonAction))


class StructuralTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = CheapFarGuardOpenHandYakuAwareCallPolicy()

    def test_legal_action_order_does_not_change_suppressed_result(self) -> None:
        alternate = _pon(PINZU_2)
        selected = {
            self.policy.choose_action(_decision(TANYAO_CHI_CONCEALED, actions))
            for actions in itertools.permutations((PASS, TANYAO_CHI_ACTION, alternate))
        }
        self.assertEqual(selected, {PASS})

    def test_deterministic_repeat(self) -> None:
        decision = _decision(TANYAO_CHI_CONCEALED, (PASS, TANYAO_CHI_ACTION))
        first = self.policy.choose_action(decision)
        second = self.policy.choose_action(decision)
        self.assertIs(first, PASS)
        self.assertIs(second, PASS)

    def test_missing_pass_action_raises_when_call_is_suppressed(self) -> None:
        decision = _decision(TANYAO_CHI_CONCEALED, (TANYAO_CHI_ACTION,))
        with self.assertRaises(YakuhaiCallPolicyError):
            self.policy.choose_action(decision)

    def test_public_export_pickle_statelessness_and_dependency_boundary(self) -> None:
        from lisjong.policies import (
            CheapFarGuardOpenHandYakuAwareCallPolicy as imported,
        )

        self.assertIs(
            imported, cheap_far_guard.CheapFarGuardOpenHandYakuAwareCallPolicy
        )
        self.assertIs(
            pickle.loads(pickle.dumps(CheapFarGuardOpenHandYakuAwareCallPolicy)),
            CheapFarGuardOpenHandYakuAwareCallPolicy,
        )
        self.assertEqual(vars(CheapFarGuardOpenHandYakuAwareCallPolicy()), {})
        tree = ast.parse(inspect.getsource(cheap_far_guard))
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


class ValueProxyTest(unittest.TestCase):
    """`_current_visible_value_proxy`の各componentをplayer-visible情報だけで検証する。"""

    def test_tanyao_only(self) -> None:
        tiles = _hand("234m456p678s")
        self.assertEqual(
            cheap_far_guard._current_visible_value_proxy(
                tiles, (), seat_wind_rank=1, round_wind_rank=2
            ),
            1,
        )

    def test_open_honitsu_only(self) -> None:
        tiles = _hand("123456789m1z")
        self.assertEqual(
            cheap_far_guard._current_visible_value_proxy(
                tiles, (), seat_wind_rank=1, round_wind_rank=2
            ),
            2,
        )

    def test_open_chinitsu_only(self) -> None:
        tiles = _hand("123456789m")
        self.assertEqual(
            cheap_far_guard._current_visible_value_proxy(
                tiles, (), seat_wind_rank=1, round_wind_rank=2
            ),
            5,
        )

    def test_tanyao_and_chinitsu_coexist(self) -> None:
        tiles = _hand("22345678m")
        self.assertEqual(
            cheap_far_guard._current_visible_value_proxy(
                tiles, (), seat_wind_rank=1, round_wind_rank=2
            ),
            6,
        )

    def test_completed_dragon_yakuhai(self) -> None:
        tiles = _hand("555z234m456p")
        self.assertEqual(
            cheap_far_guard._current_visible_value_proxy(
                tiles, (), seat_wind_rank=2, round_wind_rank=3
            ),
            1,
        )

    def test_completed_seat_wind_yakuhai(self) -> None:
        tiles = _hand("111z234m456p")
        self.assertEqual(
            cheap_far_guard._current_visible_value_proxy(
                tiles, (), seat_wind_rank=1, round_wind_rank=2
            ),
            1,
        )

    def test_completed_round_wind_yakuhai(self) -> None:
        tiles = _hand("111z234m456p")
        self.assertEqual(
            cheap_far_guard._current_visible_value_proxy(
                tiles, (), seat_wind_rank=2, round_wind_rank=1
            ),
            1,
        )

    def test_completed_double_wind_yakuhai(self) -> None:
        tiles = _hand("111z234m456p")
        self.assertEqual(
            cheap_far_guard._current_visible_value_proxy(
                tiles, (), seat_wind_rank=1, round_wind_rank=1
            ),
            2,
        )

    def test_currently_owned_dora(self) -> None:
        indicators = (PINZU_1,)
        tiles = _hand("22p345m")
        self.assertEqual(
            cheap_far_guard._current_visible_value_proxy(
                tiles, indicators, seat_wind_rank=1, round_wind_rank=2
            ),
            3,
        )

    def test_aka_dora(self) -> None:
        tiles = (MANZU_5_RED,) + _hand("234p567s")
        self.assertEqual(
            cheap_far_guard._current_visible_value_proxy(
                tiles, (), seat_wind_rank=1, round_wind_rank=2
            ),
            2,
        )

    def test_coexisting_routes_add_deterministically(self) -> None:
        tiles = _hand("555z234m") + (PINZU_2,)
        indicators = (PINZU_1,)
        self.assertEqual(
            cheap_far_guard._current_visible_value_proxy(
                tiles, indicators, seat_wind_rank=2, round_wind_rank=3
            ),
            2,
        )

    def test_route_value_helper_is_not_reused_as_actual_han(self) -> None:
        chinitsu_tiles = _hand("123456789m")
        proxy_value = cheap_far_guard._current_visible_value_proxy(
            chinitsu_tiles, (), seat_wind_rank=1, round_wind_rank=2
        )
        old_route_value = _yaku_route_value_for_tiles(chinitsu_tiles)
        self.assertNotEqual(proxy_value, old_route_value)


class CheapFarEvaluationBoundaryTest(unittest.TestCase):
    """conservative all-stable-hands semanticsをstable hand集合単位で検証する。"""

    def setUp(self) -> None:
        self.policy_input = _input(())
        self.action = TANYAO_CHI_ACTION

    def _evaluate(self, stable_hands) -> cheap_far_guard._CheapFarEvaluation:
        with patch.object(
            cheap_far_guard, "_post_call_stable_hands", return_value=stable_hands
        ):
            return cheap_far_guard._evaluate_cheap_far_call(
                self.policy_input, self.action
            )

    def test_empty_stable_hands_does_not_suppress(self) -> None:
        evaluation = self._evaluate(())
        self.assertFalse(evaluation.suppress)

    def test_one_stable_hand_at_or_below_one_shanten_prevents_suppression(
        self,
    ) -> None:
        evaluation = self._evaluate((HAND_SHANTEN_2_CHEAP, HAND_SHANTEN_1_CHEAP))
        self.assertFalse(evaluation.far)
        self.assertFalse(evaluation.suppress)

    def test_one_stable_hand_at_or_above_three_value_prevents_suppression(
        self,
    ) -> None:
        evaluation = self._evaluate((HAND_SHANTEN_2_CHEAP, HAND_SHANTEN_2_EXPENSIVE))
        self.assertTrue(evaluation.far)
        self.assertFalse(evaluation.cheap)
        self.assertFalse(evaluation.suppress)

    def test_two_shanten_cheap_stable_hands_suppress(self) -> None:
        evaluation = self._evaluate((HAND_SHANTEN_2_CHEAP,))
        self.assertTrue(evaluation.suppress)

    def test_three_or_more_shanten_cheap_stable_hands_suppress(self) -> None:
        evaluation = self._evaluate((HAND_SHANTEN_3_CHEAP,))
        self.assertTrue(evaluation.suppress)

    def test_actual_post_call_discard_is_not_precommitted(self) -> None:
        with patch.object(
            cheap_far_guard,
            "_post_call_stable_hands",
            return_value=(HAND_SHANTEN_2_CHEAP,),
        ) as stable_hands_mock:
            cheap_far_guard._evaluate_cheap_far_call(self.policy_input, self.action)
        stable_hands_mock.assert_called_once_with(
            self.policy_input.own_hand.concealed_tiles, self.action
        )


if __name__ == "__main__":
    unittest.main()
