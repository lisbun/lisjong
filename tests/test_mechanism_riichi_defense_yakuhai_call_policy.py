"""Issue #163 mechanism-based riichi defense Policy tests."""

import ast
import inspect
import itertools
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import lisjong.policies.mechanism_riichi_defense_yakuhai_call as mechanism
from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.tile_inventory import STANDARD_TILE_COUNTS
from lisjong.policies import (
    MechanismRiichiDefenseYakuhaiCallPolicy,
    YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
)
from lisjong.policy_contract.action import (
    AnkanAction,
    DiscardAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind


def _type(category: TileCategory, rank: int) -> TileType:
    return TileType(category, rank)


def _tile(category: TileCategory, rank: int, *, red: bool = False) -> Tile:
    return Tile(_type(category, rank), is_red=red)


def _hand(spec: str) -> tuple[Tile, ...]:
    categories = {
        "m": TileCategory.MANZU,
        "p": TileCategory.PINZU,
        "s": TileCategory.SOUZU,
        "z": TileCategory.HONOR,
    }
    result: list[Tile] = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        for raw_rank in ranks:
            rank = int(raw_rank)
            result.append(
                _tile(categories[character], 5 if rank == 0 else rank, red=rank == 0)
            )
        ranks = ""
    if ranks:
        raise ValueError("trailing ranks")
    return tuple(result)


def _discard_history(tile: Tile, order: int = 0) -> Discard:
    return Discard(tile=tile, tsumogiri=False, order=order, called_by=None)


def _player(
    *,
    riichi: RiichiState = RiichiState.NONE,
    discards: tuple[Tile, ...] = (),
) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=tuple(
            _discard_history(tile, order) for order, tile in enumerate(discards)
        ),
        melds=(),
        riichi=riichi,
    )


def _input(
    concealed: tuple[Tile, ...],
    *,
    threats: tuple[tuple[Tile, ...], ...] = (),
    dora_indicators: tuple[Tile, ...] = (),
) -> PolicyInput:
    players = [_player() for _ in range(4)]
    for offset, discards in enumerate(threats, start=1):
        players[offset] = _player(
            riichi=RiichiState.ACCEPTED,
            discards=discards,
        )
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
        own_hand=OwnHandState(concealed_tiles=concealed, drawn_tile=None),
    )


def _action(tile: Tile) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=False)


def _remaining(**overrides: int) -> tuple[int, ...]:
    counts = list(STANDARD_TILE_COUNTS)
    categories = {
        "m": TileCategory.MANZU,
        "p": TileCategory.PINZU,
        "s": TileCategory.SOUZU,
        "z": TileCategory.HONOR,
    }
    for name, count in overrides.items():
        rank = int(name[:-1])
        counts[tile_type_index(_type(categories[name[-1]], rank))] = count
    return tuple(counts)


def _score(
    candidate: TileType,
    *,
    river: tuple[Tile, ...] = (),
    remaining: tuple[int, ...] | None = None,
) -> mechanism._ClassicalRiichiDangerBreakdown:
    return mechanism._classical_riichi_danger_score(
        candidate,
        _player(riichi=RiichiState.ACCEPTED, discards=river),
        _remaining(
            **{
                f"{candidate.rank}{'z' if candidate.category is TileCategory.HONOR else 'm'}": 3
            }
        )
        if remaining is None
        else remaining,
    )


class MechanismScoreTest(unittest.TestCase):
    def test_candidate_genbutsu_zeroes_every_contribution(self) -> None:
        candidate = _type(TileCategory.MANZU, 4)
        self.assertEqual(
            _score(candidate, river=(_tile(TileCategory.MANZU, 4),)),
            mechanism._ZERO_DANGER,
        )

    def test_other_opponent_river_does_not_make_candidate_genbutsu(self) -> None:
        candidate = _type(TileCategory.MANZU, 4)
        self.assertGreater(_score(candidate).total, 0)

    def test_same_tile_fixed_v1_table(self) -> None:
        suited = _type(TileCategory.MANZU, 1)
        honor = _type(TileCategory.HONOR, 1)
        expected = {0: 0, 1: 1, 2: 3, 3: 3}
        for count, contribution in expected.items():
            with self.subTest(count=count):
                self.assertEqual(
                    _score(suited, remaining=_remaining(**{"1m": count})).same_tile,
                    contribution,
                )
        self.assertEqual(
            _score(honor, remaining=_remaining(**{"1z": 3})).same_tile,
            8,
        )

    def test_honor_has_no_sequence_contribution(self) -> None:
        result = _score(
            _type(TileCategory.HONOR, 5),
            remaining=_remaining(**{"5z": 3}),
        )
        self.assertEqual(result.same_tile, 8)
        self.assertEqual((result.penchan, result.kanchan), (0, 0))
        self.assertEqual((result.ryanmen_low_side, result.ryanmen_high_side), (0, 0))

    def test_penchan_three_and_seven_require_both_supporting_tiles(self) -> None:
        three = _type(TileCategory.MANZU, 3)
        seven = _type(TileCategory.MANZU, 7)
        self.assertEqual(_score(three).penchan, 3)
        self.assertEqual(_score(seven).penchan, 3)
        self.assertEqual(
            _score(three, remaining=_remaining(**{"3m": 3, "1m": 0})).penchan,
            0,
        )
        self.assertEqual(
            _score(seven, remaining=_remaining(**{"7m": 3, "9m": 0})).penchan,
            0,
        )

    def test_kanchan_rank_boundaries_and_supporting_tiles(self) -> None:
        for rank in range(1, 10):
            with self.subTest(rank=rank):
                result = _score(_type(TileCategory.MANZU, rank))
                self.assertEqual(result.kanchan, 3 if 2 <= rank <= 8 else 0)
        self.assertEqual(
            _score(
                _type(TileCategory.MANZU, 5),
                remaining=_remaining(**{"5m": 3, "4m": 0}),
            ).kanchan,
            0,
        )

    def test_all_valid_ryanmen_directions_and_furiten_counterparts(self) -> None:
        for rank in range(1, 10):
            candidate = _type(TileCategory.MANZU, rank)
            with self.subTest(rank=rank, direction="low"):
                self.assertEqual(
                    _score(candidate).ryanmen_low_side,
                    10 if rank <= 6 else 0,
                )
                if rank <= 6:
                    counterpart = _tile(TileCategory.MANZU, rank + 3)
                    self.assertEqual(
                        _score(candidate, river=(counterpart,)).ryanmen_low_side,
                        0,
                    )
            with self.subTest(rank=rank, direction="high"):
                self.assertEqual(
                    _score(candidate).ryanmen_high_side,
                    10 if rank >= 4 else 0,
                )
                if rank >= 4:
                    counterpart = _tile(TileCategory.MANZU, rank - 3)
                    self.assertEqual(
                        _score(candidate, river=(counterpart,)).ryanmen_high_side,
                        0,
                    )

    def test_four_candidate_loses_each_ryanmen_independently(self) -> None:
        candidate = _type(TileCategory.MANZU, 4)
        one = _tile(TileCategory.MANZU, 1)
        seven = _tile(TileCategory.MANZU, 7)
        self.assertEqual(
            (_score(candidate).ryanmen_low_side, _score(candidate).ryanmen_high_side),
            (10, 10),
        )
        self.assertEqual(
            (
                _score(candidate, river=(one,)).ryanmen_low_side,
                _score(candidate, river=(one,)).ryanmen_high_side,
            ),
            (10, 0),
        )
        self.assertEqual(
            (
                _score(candidate, river=(seven,)).ryanmen_low_side,
                _score(candidate, river=(seven,)).ryanmen_high_side,
            ),
            (0, 10),
        )
        result = _score(candidate, river=(one, seven))
        self.assertEqual((result.ryanmen_low_side, result.ryanmen_high_side), (0, 0))
        self.assertGreater(result.same_tile + result.kanchan, 0)

    def test_wall_removes_only_affected_mechanism(self) -> None:
        result = _score(
            _type(TileCategory.MANZU, 4),
            remaining=_remaining(**{"4m": 3, "2m": 0}),
        )
        self.assertEqual(result.ryanmen_high_side, 0)
        self.assertEqual(result.ryanmen_low_side, 10)
        self.assertEqual(result.kanchan, 3)

    def test_one_chance_is_still_fully_possible_in_v1(self) -> None:
        result = _score(
            _type(TileCategory.MANZU, 1),
            remaining=_remaining(**{"1m": 3, "2m": 1, "3m": 1}),
        )
        self.assertEqual(result.ryanmen_low_side, 10)

    def test_four_in_river_makes_one_lower_than_unseen_honor(self) -> None:
        one = _score(
            _type(TileCategory.MANZU, 1),
            river=(_tile(TileCategory.MANZU, 4),),
            remaining=_remaining(**{"1m": 3}),
        )
        honor = _score(
            _type(TileCategory.HONOR, 5),
            remaining=_remaining(**{"5z": 3}),
        )
        self.assertEqual((one.total, honor.total), (3, 8))

    def test_red_five_uses_base_tile_type(self) -> None:
        normal = _tile(TileCategory.MANZU, 5)
        red = _tile(TileCategory.MANZU, 5, red=True)
        opponent = _player(riichi=RiichiState.ACCEPTED, discards=(normal,))
        counts = _remaining(**{"5m": 2})
        self.assertEqual(
            mechanism._classical_riichi_danger_score(
                red.tile_type, opponent, counts
            ).total,
            0,
        )


class ActivationAndFilteringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.one = _tile(TileCategory.MANZU, 1)
        self.honor = _tile(TileCategory.HONOR, 5)
        self.actions = (_action(self.one), _action(self.honor))

    def _eligible(
        self,
        shanten: int,
        *,
        threats: tuple[tuple[Tile, ...], ...] = ((_tile(TileCategory.PINZU, 9),),),
        danger_by_rank: dict[int, int] | None = None,
        actions: tuple[DiscardAction, ...] | None = None,
    ) -> tuple[DiscardAction, ...]:
        selected_actions = self.actions if actions is None else actions
        policy_input = _input((self.one, self.honor), threats=threats)
        dangers = danger_by_rank or {1: 3, 5: 8}

        def score(candidate, opponent, remaining_counts):
            return mechanism._ClassicalRiichiDangerBreakdown(
                dangers[candidate.rank], 0, 0, 0, 0
            )

        with (
            patch.object(
                mechanism,
                "_evaluate_post_discard_hands",
                return_value=tuple(
                    SimpleNamespace(action=action, post_discard_shanten=shanten)
                    for action in selected_actions
                ),
            ),
            patch.object(mechanism, "_classical_riichi_danger_score", score),
        ):
            return mechanism._mechanism_defense_eligible_actions(
                policy_input, selected_actions
            )

    def test_no_riichi_preserves_every_action(self) -> None:
        self.assertEqual(self._eligible(3, threats=()), self.actions)

    def test_tenpai_and_one_shanten_preserve_every_action(self) -> None:
        for shanten in (0, 1):
            with self.subTest(shanten=shanten):
                self.assertEqual(self._eligible(shanten), self.actions)

    def test_common_genbutsu_preserves_parent_filtering_path(self) -> None:
        self.assertEqual(self._eligible(2, threats=((self.one,),)), self.actions)

    def test_two_and_three_shanten_activate(self) -> None:
        for shanten in (2, 3):
            with self.subTest(shanten=shanten):
                self.assertEqual(self._eligible(shanten), (self.actions[0],))

    def test_real_shanten_and_public_inventory_filter_both_distances(self) -> None:
        cases = (
            ("345m56679s333517z", 2),
            ("345m56679s337124z", 3),
        )
        for hand_spec, expected_shanten in cases:
            with self.subTest(hand=hand_spec):
                concealed = _hand(hand_spec)
                manzu_three = _action(_tile(TileCategory.MANZU, 3))
                honor = _action(_tile(TileCategory.HONOR, 7))
                policy_input = _input(
                    concealed,
                    threats=((_tile(TileCategory.PINZU, 9),),),
                )
                actions = (manzu_three, honor)
                evaluated = mechanism._evaluate_post_discard_hands(
                    policy_input, actions, mechanism._DecisionShantenEvaluator()
                )
                self.assertGreaterEqual(
                    min(candidate.post_discard_shanten for candidate in evaluated),
                    expected_shanten,
                )
                self.assertEqual(
                    mechanism._mechanism_defense_eligible_actions(
                        policy_input, actions
                    ),
                    (honor,),
                )

    def test_equal_minimum_danger_retains_all_ties(self) -> None:
        self.assertEqual(
            self._eligible(2, danger_by_rank={1: 4, 5: 4}),
            self.actions,
        )

    def test_multiple_riichi_uses_maximum_per_opponent(self) -> None:
        policy_input = _input(
            (self.one, self.honor),
            threats=(
                (_tile(TileCategory.PINZU, 8),),
                (_tile(TileCategory.SOUZU, 8),),
            ),
        )
        per_opponent = iter((0, 9, 5, 5))
        with (
            patch.object(
                mechanism,
                "_evaluate_post_discard_hands",
                return_value=tuple(
                    SimpleNamespace(action=action, post_discard_shanten=2)
                    for action in self.actions
                ),
            ),
            patch.object(
                mechanism,
                "_classical_riichi_danger_score",
                side_effect=lambda *args: mechanism._ClassicalRiichiDangerBreakdown(
                    next(per_opponent), 0, 0, 0, 0
                ),
            ),
        ):
            eligible = mechanism._mechanism_defense_eligible_actions(
                policy_input, self.actions
            )
        self.assertEqual(eligible, (self.actions[1],))

    def test_legal_action_order_does_not_change_selected_identity(self) -> None:
        results = {
            self._eligible(2, actions=permutation)[0]
            for permutation in itertools.permutations(self.actions)
        }
        self.assertEqual(results, {self.actions[0]})

    def test_minimum_subset_alone_reaches_existing_offensive_ranking(self) -> None:
        policy_input = _input(
            (self.one, self.honor),
            threats=((_tile(TileCategory.PINZU, 9),),),
        )
        observed: list[tuple[DiscardAction, ...]] = []
        with (
            patch.object(
                mechanism,
                "_mechanism_defense_eligible_actions",
                return_value=(self.actions[0],),
            ),
            patch.object(
                mechanism,
                "_parent_offensive_evaluate_and_choose_discard",
                side_effect=lambda _, actions: observed.append(actions) or actions[0],
            ),
        ):
            chosen = (
                MechanismRiichiDefenseYakuhaiCallPolicy()
                ._decide_discard(policy_input, self.actions)
                .action
            )
        self.assertIs(chosen, self.actions[0])
        self.assertEqual(observed, [(self.actions[0],)])

    def test_remaining_inventory_is_derived_once_per_activation(self) -> None:
        policy_input = _input(
            (self.one, self.honor),
            threats=((_tile(TileCategory.PINZU, 9),),),
            dora_indicators=(_tile(TileCategory.SOUZU, 2),),
        )
        actual = mechanism.derive_remaining_tile_inventory(policy_input)
        with (
            patch.object(
                mechanism,
                "_evaluate_post_discard_hands",
                return_value=tuple(
                    SimpleNamespace(action=action, post_discard_shanten=2)
                    for action in self.actions
                ),
            ),
            patch.object(
                mechanism,
                "derive_remaining_tile_inventory",
                wraps=mechanism.derive_remaining_tile_inventory,
            ) as derive,
        ):
            mechanism._mechanism_defense_eligible_actions(policy_input, self.actions)
        derive.assert_called_once_with(policy_input)
        self.assertEqual(
            actual.remaining_tile_counts[tile_type_index(self.one.tile_type)], 3
        )
        self.assertEqual(
            actual.remaining_tile_counts[tile_type_index(_type(TileCategory.PINZU, 9))],
            3,
        )
        self.assertEqual(
            actual.remaining_tile_counts[tile_type_index(_type(TileCategory.SOUZU, 2))],
            3,
        )


class ParentAndStructuralPreservationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = MechanismRiichiDefenseYakuhaiCallPolicy()
        self.parent = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()

    def _decision(
        self, concealed: tuple[Tile, ...], *actions: object
    ) -> DecisionContext:
        return DecisionContext(input=_input(concealed), legal_actions=actions)

    def test_winning_and_riichi_are_exact_parent_decisions(self) -> None:
        tile = _tile(TileCategory.MANZU, 1)
        ron = RonAction(Seat.SEAT_0, Seat.SEAT_1, tile)
        riichi = RiichiAction(Seat.SEAT_0)
        pass_action = PassAction(Seat.SEAT_0)
        for decision in (
            self._decision((tile,), pass_action, ron),
            self._decision((), pass_action, riichi),
        ):
            with self.subTest(actions=decision.legal_actions):
                self.assertEqual(
                    self.policy.choose_action(decision),
                    self.parent.choose_action(decision),
                )

    def test_kan_and_call_response_are_exact_parent_decisions(self) -> None:
        white = _tile(TileCategory.HONOR, 5)
        ankan = AnkanAction(Seat.SEAT_0, (white,) * 4)
        pass_action = PassAction(Seat.SEAT_0)
        pon = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=white,
            consumed_tiles=(white, white),
        )
        decisions = (
            self._decision(
                (white,) * 4 + _hand("123456m789p1s"),
                ankan,
                _action(white),
            ),
            self._decision(_hand("123456m789p19s") + (white, white), pass_action, pon),
        )
        for decision in decisions:
            with self.subTest(actions=decision.legal_actions):
                self.assertEqual(
                    self.policy.choose_action(decision),
                    self.parent.choose_action(decision),
                )

    def test_public_export_pickle_repeat_and_statelessness(self) -> None:
        self.assertIs(
            pickle.loads(pickle.dumps(self.policy)).__class__,
            MechanismRiichiDefenseYakuhaiCallPolicy,
        )
        self.assertEqual(vars(self.policy), {})
        decision = self._decision((), PassAction(Seat.SEAT_0))
        first = self.policy.choose_action(decision)
        second = self.policy.choose_action(decision)
        self.assertIs(first, second)

    def test_exact_parent_and_policy_visible_dependency_boundary(self) -> None:
        self.assertEqual(
            MechanismRiichiDefenseYakuhaiCallPolicy.__bases__,
            (YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy,),
        )
        tree = ast.parse(inspect.getsource(mechanism))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertFalse(
            any(
                module.startswith(prefix)
                for module in imported
                for prefix in (
                    "lisjong.belief.exact_wait_ground_truth",
                    "lisjong_engine",
                    "mahjong",
                    "riichienv",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
