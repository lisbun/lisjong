"""Issue #151 `KanCoverageYakuhaiCallPolicy`のfocused deterministic tests。"""

import inspect
import itertools
import unittest

from lisjong.policies import YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy
from lisjong.policies.kan_coverage_yakuhai_call import KanCoverageYakuhaiCallPolicy
from lisjong.policy_contract.action import (
    AnkanAction,
    DaiminkanAction,
    DiscardAction,
    KakanAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_execution import execute_policy
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
MANZU_6 = _tile(TileCategory.MANZU, 6)
PINZU_2 = _tile(TileCategory.PINZU, 2)
PINZU_5 = _tile(TileCategory.PINZU, 5)
SOUZU_9 = _tile(TileCategory.SOUZU, 9)
WHITE_DRAGON = _tile(TileCategory.HONOR, 5)

_TWO_STEP_HAND = _hand("345m56679s333577z")


def _player(*, discards: tuple = ()) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000, discards=discards, melds=(), riichi=RiichiState.NONE
    )


def _input(concealed_tiles: tuple[Tile, ...]) -> PolicyInput:
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
        players=(_player(), _player(), _player(), _player()),
        own_hand=OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=None),
    )


def _decision(concealed_tiles: tuple[Tile, ...], *actions: object) -> DecisionContext:
    return DecisionContext(input=_input(concealed_tiles), legal_actions=actions)


def _discard(tile: Tile) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=False)


def _ron(tile: Tile = MANZU_5) -> RonAction:
    return RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=tile)


def _tsumo(tile: Tile = MANZU_5) -> TsumoAction:
    return TsumoAction(actor=Seat.SEAT_0, winning_tile=tile)


def _daiminkan(tile: Tile = MANZU_3, target: Seat = Seat.SEAT_1) -> DaiminkanAction:
    return DaiminkanAction(
        actor=Seat.SEAT_0,
        target=target,
        called_tile=tile,
        consumed_tiles=(tile, tile, tile),
    )


def _ankan(tile: Tile = MANZU_2) -> AnkanAction:
    return AnkanAction(actor=Seat.SEAT_0, tiles=(tile, tile, tile, tile))


def _kakan(tile: Tile = MANZU_2, from_seat: Seat = Seat.SEAT_1) -> KakanAction:
    return KakanAction(
        actor=Seat.SEAT_0,
        added_tile=tile,
        from_seat=from_seat,
        called_tile=tile,
    )


PASS = PassAction(actor=Seat.SEAT_0)
NEUTRAL_HAND = (MANZU_4, MANZU_5, MANZU_6, PINZU_5)


class SelectionPriorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = KanCoverageYakuhaiCallPolicy()

    def test_ron_beats_kan(self) -> None:
        ron = _ron()
        ankan = _ankan()

        chosen = self.policy.choose_action(_decision(NEUTRAL_HAND, ankan, ron))

        self.assertIs(chosen, ron)

    def test_tsumo_beats_kan(self) -> None:
        tsumo = _tsumo()
        kakan = _kakan()

        chosen = self.policy.choose_action(_decision(NEUTRAL_HAND, kakan, tsumo))

        self.assertIs(chosen, tsumo)

    def test_daiminkan_beats_pass(self) -> None:
        daiminkan = _daiminkan()

        chosen = self.policy.choose_action(_decision(NEUTRAL_HAND, PASS, daiminkan))

        self.assertIs(chosen, daiminkan)

    def test_ankan_beats_discard(self) -> None:
        ankan = _ankan()
        discard = _discard(MANZU_4)

        chosen = self.policy.choose_action(_decision(NEUTRAL_HAND, discard, ankan))

        self.assertIs(chosen, ankan)

    def test_kakan_beats_discard(self) -> None:
        kakan = _kakan()
        discard = _discard(MANZU_4)

        chosen = self.policy.choose_action(_decision(NEUTRAL_HAND, discard, kakan))

        self.assertIs(chosen, kakan)

    def test_winning_beats_kan_and_discard_together(self) -> None:
        ron = _ron()
        ankan = _ankan()
        discard = _discard(MANZU_4)

        chosen = self.policy.choose_action(_decision(NEUTRAL_HAND, discard, ankan, ron))

        self.assertIs(chosen, ron)

    def test_no_kan_or_winning_action_delegates_to_yakuhai_call_discard(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        decision = _decision(_TWO_STEP_HAND, discard_9s, discard_white)

        delegate = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()
        expected = delegate.choose_action(decision)

        chosen = self.policy.choose_action(decision)

        self.assertEqual(chosen, expected)

    def test_no_kan_or_winning_action_delegates_to_yakuhai_call_pon(self) -> None:
        concealed = _hand("123456m789p19s") + (WHITE_DRAGON, WHITE_DRAGON)
        pon = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=WHITE_DRAGON,
            consumed_tiles=(WHITE_DRAGON, WHITE_DRAGON),
        )
        decision = _decision(concealed, PASS, pon)

        delegate = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()
        expected = delegate.choose_action(decision)

        chosen = self.policy.choose_action(decision)

        self.assertEqual(chosen, expected)
        self.assertIs(chosen, pon)


class DeterminismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = KanCoverageYakuhaiCallPolicy()

    def test_priority_choices_are_independent_of_legal_action_order(self) -> None:
        ron = _ron()
        ankan = _ankan()
        discard = _discard(MANZU_4)

        results = {
            self.policy.choose_action(_decision(NEUTRAL_HAND, *permutation))
            for permutation in itertools.permutations((discard, ankan, ron))
        }

        self.assertEqual(results, {ron})

    def test_multiple_ankan_candidates_are_stable(self) -> None:
        low = _ankan(MANZU_2)
        high = _ankan(MANZU_5)
        discard = _discard(MANZU_4)

        results = {
            self.policy.choose_action(_decision(NEUTRAL_HAND, *permutation))
            for permutation in itertools.permutations((discard, low, high))
        }

        self.assertEqual(results, {low})

    def test_multiple_kakan_candidates_are_stable(self) -> None:
        low = _kakan(MANZU_2, from_seat=Seat.SEAT_2)
        high = _kakan(MANZU_5, from_seat=Seat.SEAT_1)
        discard = _discard(MANZU_4)

        results = {
            self.policy.choose_action(_decision(NEUTRAL_HAND, *permutation))
            for permutation in itertools.permutations((discard, low, high))
        }

        self.assertEqual(results, {low})

    def test_multiple_kan_kinds_are_stable_with_daiminkan_first(self) -> None:
        daiminkan = _daiminkan()
        ankan = _ankan()
        kakan = _kakan()
        discard = _discard(MANZU_4)

        results = {
            self.policy.choose_action(_decision(NEUTRAL_HAND, *permutation))
            for permutation in itertools.permutations(
                (discard, kakan, ankan, daiminkan)
            )
        }

        self.assertEqual(results, {daiminkan})

    def test_repeated_and_interleaved_calls_do_not_change_choice(self) -> None:
        ankan = _ankan()
        discard = _discard(MANZU_4)
        kan_decision = _decision(NEUTRAL_HAND, discard, ankan)
        unrelated_decision = _decision(NEUTRAL_HAND, PASS)

        first = self.policy.choose_action(kan_decision)
        self.policy.choose_action(unrelated_decision)
        second = self.policy.choose_action(kan_decision)

        self.assertIs(first, ankan)
        self.assertIs(second, ankan)

    def test_delegate_instance_is_not_recreated_between_calls(self) -> None:
        before = self.policy._delegate

        self.policy.choose_action(_decision(NEUTRAL_HAND, PASS))
        self.policy.choose_action(_decision(NEUTRAL_HAND, _ankan(), _discard(MANZU_4)))

        self.assertIs(self.policy._delegate, before)


class ContractSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = KanCoverageYakuhaiCallPolicy()

    def test_selected_action_is_always_an_original_legal_action_member(self) -> None:
        decisions = (
            _decision(NEUTRAL_HAND, _ron(), _ankan()),
            _decision(NEUTRAL_HAND, _discard(MANZU_4), _kakan()),
            _decision(NEUTRAL_HAND, PASS, _daiminkan()),
            _decision(NEUTRAL_HAND, PASS),
        )
        for decision in decisions:
            with self.subTest(decision=decision):
                chosen = self.policy.choose_action(decision)
                self.assertTrue(
                    any(chosen is action for action in decision.legal_actions)
                )

    def test_does_not_intercept_riichi_or_chi_when_no_kan_is_legal(self) -> None:
        riichi_hand = _hand("123456789m123p22s")
        riichi = RiichiAction(actor=Seat.SEAT_0)
        discard = _discard(MANZU_1)
        decision = _decision(riichi_hand, riichi, discard)

        delegate = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()
        expected = delegate.choose_action(decision)

        chosen = self.policy.choose_action(decision)

        self.assertEqual(chosen, expected)
        self.assertIs(chosen, riichi)

    def test_execute_policy_integration_path_accepts_kan_selection(self) -> None:
        ankan = _ankan()
        discard = _discard(MANZU_4)
        decision = _decision(NEUTRAL_HAND, discard, ankan)

        selected = execute_policy(self.policy, decision)

        self.assertIs(selected, ankan)

    def test_execute_policy_integration_path_accepts_winning_selection(self) -> None:
        ron = _ron()
        ankan = _ankan()
        decision = _decision(NEUTRAL_HAND, ankan, ron)

        selected = execute_policy(self.policy, decision)

        self.assertIs(selected, ron)

    def test_public_api_requires_only_decision_context(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(KanCoverageYakuhaiCallPolicy).parameters), ()
        )
        self.assertEqual(
            tuple(
                inspect.signature(KanCoverageYakuhaiCallPolicy.choose_action).parameters
            ),
            ("self", "decision"),
        )


class ExistingPolicySemanticsUnchangedTest(unittest.TestCase):
    """既存`MinimalPolicy` / `yakuhai-call`のkan非選択semanticsが不変であることの回帰確認。"""

    def test_minimal_policy_still_prefers_discard_over_self_turn_kan(self) -> None:
        from lisjong.policies import MinimalPolicy

        ankan = _ankan()
        discard = _discard(MANZU_4)
        decision = _decision(NEUTRAL_HAND, discard, ankan)

        chosen = MinimalPolicy().choose_action(decision)

        self.assertIs(chosen, discard)

    def test_yakuhai_call_policy_does_not_select_kan_itself(self) -> None:
        ankan = _ankan()
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        decision = _decision(_TWO_STEP_HAND, discard_9s, discard_white, ankan)

        delegate = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()
        chosen = delegate.choose_action(decision)

        self.assertIn(chosen, (discard_9s, discard_white))


if __name__ == "__main__":
    unittest.main()
