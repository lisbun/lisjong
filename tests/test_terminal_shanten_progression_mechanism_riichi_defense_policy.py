"""Issue #169 expected-terminal-shanten progression Policy tests."""

import ast
import inspect
import itertools
import pickle
import random
import unittest
from unittest.mock import patch

import lisjong.policies.terminal_shanten_progression_mechanism_riichi_defense as progression
from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.belief.tile_inventory import TILE_TYPE_COUNT
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies import (
    MechanismRiichiDefenseYakuhaiCallPolicy,
    TerminalShantenProgressionMechanismRiichiDefensePolicy,
    TwoStepUkeirePolicy,
)
from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    FiniteHorizonCompletionPolicyError,
    _evaluate_completion_masses,
    _falling_factorial,
    _FiniteHorizonEvaluator,
)
from lisjong.policies.two_step_ukeire import _known_tile_counts, _ukeire_count
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
from lisjong.policy_contract.meld import PublicMeld
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
_CATEGORY_OFFSETS = {
    TileCategory.MANZU: 0,
    TileCategory.PINZU: 9,
    TileCategory.SOUZU: 18,
    TileCategory.HONOR: 27,
}
_CATEGORY_SIZES = {
    TileCategory.MANZU: 9,
    TileCategory.PINZU: 9,
    TileCategory.SOUZU: 9,
    TileCategory.HONOR: 7,
}
_MAX_COPIES_PER_TILE_TYPE = 4


def _type(category: TileCategory, rank: int) -> TileType:
    return TileType(category, rank)


def _tile(category: TileCategory, rank: int, *, red: bool = False) -> Tile:
    return Tile(_type(category, rank), is_red=red)


def _hand(spec: str) -> tuple[Tile, ...]:
    """`"345m0p"`のようなspecをTile列へ展開する（`0`は赤5）。"""
    result: list[Tile] = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        for raw_rank in ranks:
            rank = int(raw_rank)
            result.append(
                _tile(_CATEGORIES[character], 5 if rank == 0 else rank, red=rank == 0)
            )
        ranks = ""
    if ranks:
        raise ValueError("trailing ranks")
    return tuple(result)


def _all_tile_types() -> tuple[TileType, ...]:
    return tuple(
        _type(category, rank)
        for category, size in _CATEGORY_SIZES.items()
        for rank in range(1, size + 1)
    )


def _counts(**tiles: int) -> tuple[int, ...]:
    """`m3=1`のようなkeyword指定から34牌種countを組み立てる。"""
    counts = [0] * TILE_TYPE_COUNT
    for key, count in tiles.items():
        counts[_CATEGORY_OFFSETS[_CATEGORIES[key[0]]] + int(key[1:]) - 1] += count
    return tuple(counts)


def _tile_type_counts(tiles: tuple[Tile, ...]) -> tuple[int, ...]:
    counts = [0] * TILE_TYPE_COUNT
    for tile in tiles:
        counts[tile_type_index(tile.tile_type)] += 1
    return tuple(counts)


def _tiles_for_oracle(counts: tuple[int, ...]) -> list[Tile]:
    """test側だけで使う、34牌種countから`Tile`列への独立変換。"""
    tiles: list[Tile] = []
    for index, count in enumerate(counts):
        for category, size in _CATEGORY_SIZES.items():
            offset = _CATEGORY_OFFSETS[category]
            if offset <= index < offset + size:
                tiles.extend([_tile(category, index - offset + 1)] * count)
                break
    return tiles


def _player(
    *,
    discards: tuple[Discard, ...] = (),
    melds: tuple[PublicMeld, ...] = (),
    riichi: RiichiState = RiichiState.NONE,
) -> PlayerPublicState:
    return PlayerPublicState(score=25000, discards=discards, melds=melds, riichi=riichi)


def _river(tiles: tuple[Tile, ...]) -> tuple[Discard, ...]:
    return tuple(
        Discard(tile=tile, tsumogiri=False, order=order, called_by=None)
        for order, tile in enumerate(tiles)
    )


def _make_input(
    concealed_tiles: tuple[Tile, ...],
    *,
    players: tuple[PlayerPublicState, ...] | None = None,
    dora_indicators: tuple[Tile, ...] = (),
) -> PolicyInput:
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=dora_indicators,
            live_wall_tiles_remaining=20,
        ),
        players=players if players is not None else (_player(),) * 4,
        own_hand=OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=None),
    )


def _restricted_input(
    concealed_tiles: tuple[Tile, ...],
    drawable_specs: tuple[str, ...],
    *,
    riichi_seats: tuple[int, ...] = (),
    extra_discards: tuple[tuple[Tile, ...], ...] = ((), (), (), ()),
) -> PolicyInput:
    """remaining inventoryをdrawable牌種だけへ絞ったPolicyInputを組み立てる。

    exact progression DPの探索量はdrawable牌種数の関数なので、focused testでは
    「盤面にほとんどの牌種が見えている終盤」をPolicy-visibleな公開河として
    構成する。hidden情報は一切使っておらず、DPへ渡るremaining inventoryは
    Issue #63の`derive_remaining_tile_inventory()`が導出するものだけである。
    """
    drawable = {_hand(spec)[0].tile_type for spec in drawable_specs}
    accounted: dict[TileType, int] = {}
    for tile in concealed_tiles:
        accounted[tile.tile_type] = accounted.get(tile.tile_type, 0) + 1
    for river in extra_discards:
        for tile in river:
            accounted[tile.tile_type] = accounted.get(tile.tile_type, 0) + 1
    red_seen = {tile.tile_type.category for tile in concealed_tiles if tile.is_red}

    visible: list[Tile] = []
    for tile_type in _all_tile_types():
        if tile_type in drawable:
            continue
        copies = _MAX_COPIES_PER_TILE_TYPE - accounted.get(tile_type, 0)
        needs_red = (
            tile_type.rank == 5
            and tile_type.category is not TileCategory.HONOR
            and tile_type.category not in red_seen
        )
        for index in range(copies):
            visible.append(Tile(tile_type, is_red=needs_red and index == 0))

    rivers: list[list[Tile]] = [list(river) for river in extra_discards]
    for index, tile in enumerate(visible):
        rivers[index % 4].append(tile)
    players = tuple(
        _player(
            discards=_river(tuple(river)),
            riichi=(RiichiState.ACCEPTED if seat in riichi_seats else RiichiState.NONE),
        )
        for seat, river in enumerate(rivers)
    )
    return _make_input(concealed_tiles, players=players)


def _discard(tile: Tile, *, tsumogiri: bool = False) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=tsumogiri)


def _distinct_discard_actions(
    concealed_tiles: tuple[Tile, ...],
) -> tuple[DiscardAction, ...]:
    seen: set[Tile] = set()
    actions: list[DiscardAction] = []
    for tile in concealed_tiles:
        if tile in seen:
            continue
        seen.add(tile)
        actions.append(_discard(tile))
    return tuple(actions)


# --------------------------------------------------------------------------
# test-local reference evaluator
# --------------------------------------------------------------------------


def _reference_distribution(
    hand_counts: tuple[int, ...], remaining_counts: tuple[int, ...], depth: int
) -> tuple[int, ...]:
    """memoization・枝刈り・closed formを一切持たないbrute-force reference。

    productionへ`use_cache`のような二重execution pathを追加せず、
    「production DP == unpruned reference」を確認するためだけの独立実装である。
    `calculate_shanten()`のTile入口を使うので、count-native hot pathとも独立
    している。min nodeのtie-breakだけはproductionと同じcanonical順
    （post-discard shanten昇順、次にhand counts昇順）に揃えており、massだけで
    なくdiagnostic distributionも比較できるようにしてある。
    """
    if depth <= 0:
        counts = [0] * progression.TERMINAL_SHANTEN_AXIS
        counts[calculate_shanten(_tiles_for_oracle(hand_counts))] = 1
        return tuple(counts)

    counts = [0] * progression.TERMINAL_SHANTEN_AXIS
    for drawn in range(TILE_TYPE_COUNT):
        available = remaining_counts[drawn]
        if available == 0:
            continue
        draw_hand = list(hand_counts)
        draw_hand[drawn] += 1
        draw_hand_counts = tuple(draw_hand)
        next_remaining = list(remaining_counts)
        next_remaining[drawn] -= 1
        next_remaining_counts = tuple(next_remaining)

        children: list[tuple[int, tuple[int, ...]]] = []
        for discarded in range(TILE_TYPE_COUNT):
            if draw_hand_counts[discarded] == 0:
                continue
            child = list(draw_hand_counts)
            child[discarded] -= 1
            child_counts = tuple(child)
            children.append(
                (calculate_shanten(_tiles_for_oracle(child_counts)), child_counts)
            )
        children.sort()

        best: tuple[int, ...] | None = None
        best_mass = 0
        for _child_shanten, child_counts in children:
            distribution = _reference_distribution(
                child_counts, next_remaining_counts, depth - 1
            )
            mass = sum(index * count for index, count in enumerate(distribution))
            if best is None or mass < best_mass:
                best, best_mass = distribution, mass
        assert best is not None
        for index, value in enumerate(best):
            counts[index] += available * value
    return tuple(counts)


def _reference_best_post_discard_shanten(draw_hand_counts: tuple[int, ...]) -> int:
    """closed formを使わず、実際に全打牌を列挙する参照実装。"""
    best = None
    for discarded in range(TILE_TYPE_COUNT):
        if draw_hand_counts[discarded] == 0:
            continue
        child = list(draw_hand_counts)
        child[discarded] -= 1
        value = calculate_shanten(_tiles_for_oracle(tuple(child)))
        best = value if best is None else min(best, value)
    assert best is not None
    return best


def _mass(distribution: tuple[int, ...]) -> int:
    return sum(index * count for index, count in enumerate(distribution))


class _RecordingEvaluator(progression._TerminalShantenProgressionEvaluator):
    """productionへ二重pathを足さずにDP遷移を観測するtest-local subclass。"""

    __slots__ = ("child_states", "draw_hands", "root_queries")

    def __init__(self) -> None:
        super().__init__()
        self.root_queries: list[tuple[int, ...]] = []
        self.child_states: list[tuple[tuple[int, ...], int]] = []
        self.draw_hands: list[tuple[int, ...]] = []

    def terminal_shanten_distribution(self, hand_counts, remaining_counts, depth):
        self.root_queries.append(remaining_counts)
        return super().terminal_shanten_distribution(
            hand_counts, remaining_counts, depth
        )

    def _distribution(self, hand_counts, remaining_counts, depth, remaining_total):
        self.child_states.append((remaining_counts, depth))
        return super()._distribution(
            hand_counts, remaining_counts, depth, remaining_total
        )

    def best_post_discard_shanten(self, draw_hand_counts):
        self.draw_hands.append(draw_hand_counts)
        return super().best_post_discard_shanten(draw_hand_counts)


# 探索量を小さく保つためのreference比較用fixture。
_FOUR_MELD_HAND = _tile_type_counts(_hand("3m"))
"""4副露済みで純手牌1枚。"""

_THREE_MELD_HAND = _tile_type_counts(_hand("1m5m9s3z"))
"""3副露済みで純手牌4枚（孤立牌だけの2向聴）。"""

_TWO_MELD_HAND = _tile_type_counts(_hand("1m4m8m9s1z3z7z"))
"""2副露済みで純手牌7枚。"""

_CLOSED_FAR_HAND = _tile_type_counts(_hand("147m258p369s1357z"))
"""どの打牌後もhorizon 3では完成できない、向聴数の深い13枚。"""


class BestPostDiscardShantenTest(unittest.TestCase):
    """depth-1 closed formが全打牌列挙とexactに一致することを固定する。"""

    def test_closed_form_matches_exhaustive_discard_enumeration(self) -> None:
        evaluator = progression._TerminalShantenProgressionEvaluator()
        generator = random.Random(169)
        for draw_size in (14, 11, 8, 5, 2):
            for _ in range(120):
                counts = [0] * TILE_TYPE_COUNT
                drawn = 0
                while drawn < draw_size:
                    index = generator.randrange(TILE_TYPE_COUNT)
                    if counts[index] < _MAX_COPIES_PER_TILE_TYPE:
                        counts[index] += 1
                        drawn += 1
                draw_hand_counts = tuple(counts)
                with self.subTest(draw_size=draw_size, hand=draw_hand_counts):
                    self.assertEqual(
                        evaluator.best_post_discard_shanten(draw_hand_counts),
                        _reference_best_post_discard_shanten(draw_hand_counts),
                    )

    def test_complete_draw_hand_falls_back_to_tenpai(self) -> None:
        evaluator = progression._TerminalShantenProgressionEvaluator()
        complete = _tile_type_counts(_hand("123456789m11122p"))
        self.assertEqual(calculate_shanten(_tiles_for_oracle(complete)), -1)
        self.assertEqual(evaluator.best_post_discard_shanten(complete), 0)


class ReferenceEvaluatorAgreementTest(unittest.TestCase):
    """optimized production DPがunpruned exact referenceと一致することを固定する。"""

    def _assert_matches_reference(
        self,
        hand_counts: tuple[int, ...],
        remaining_counts: tuple[int, ...],
        depth: int,
    ) -> None:
        evaluator = progression._TerminalShantenProgressionEvaluator()
        self.assertEqual(
            evaluator.terminal_shanten_distribution(
                hand_counts, remaining_counts, depth
            ),
            _reference_distribution(hand_counts, remaining_counts, depth),
        )

    def test_horizon_one_matches_the_reference(self) -> None:
        for hand_counts in (
            _FOUR_MELD_HAND,
            _THREE_MELD_HAND,
            _TWO_MELD_HAND,
            _CLOSED_FAR_HAND,
        ):
            with self.subTest(hand=hand_counts):
                self._assert_matches_reference(
                    hand_counts, _counts(m2=3, m5=2, p4=4, z3=1), 1
                )

    def test_horizon_two_matches_the_reference(self) -> None:
        for hand_counts in (_FOUR_MELD_HAND, _THREE_MELD_HAND, _TWO_MELD_HAND):
            with self.subTest(hand=hand_counts):
                self._assert_matches_reference(
                    hand_counts, _counts(m2=3, m6=2, s9=2, z3=1), 2
                )

    def test_horizon_three_matches_the_reference_for_open_hands(self) -> None:
        for hand_counts in (_FOUR_MELD_HAND, _THREE_MELD_HAND):
            with self.subTest(hand=hand_counts):
                self._assert_matches_reference(
                    hand_counts, _counts(m2=3, m5=2, s9=2, z3=1), DEFAULT_HORIZON
                )

    def test_horizon_three_matches_the_reference_for_a_closed_hand(self) -> None:
        self._assert_matches_reference(
            _CLOSED_FAR_HAND, _counts(m2=3, m6=2), DEFAULT_HORIZON
        )

    def test_cache_reuse_does_not_change_the_result(self) -> None:
        remaining = _counts(m2=3, m6=2, s9=2, z3=1)
        shared = progression._TerminalShantenProgressionEvaluator()
        for hand_counts in (_THREE_MELD_HAND, _TWO_MELD_HAND, _THREE_MELD_HAND):
            fresh = progression._TerminalShantenProgressionEvaluator()
            expected = fresh.terminal_shanten_distribution(hand_counts, remaining, 2)
            first = shared.terminal_shanten_distribution(hand_counts, remaining, 2)
            second = shared.terminal_shanten_distribution(hand_counts, remaining, 2)
            self.assertEqual((first, second), (expected, expected))
        self.assertGreater(shared.cache_hits, 0)


class ProgressionRecurrenceTest(unittest.TestCase):
    """progression DPのexact semanticとmass boundsを固定する。"""

    def test_distribution_sums_to_the_ordered_sequence_denominator(self) -> None:
        remaining = _counts(m2=3, m6=2, s9=2, z3=1)
        hidden = sum(remaining)
        evaluator = progression._TerminalShantenProgressionEvaluator()
        for depth in (1, 2, DEFAULT_HORIZON):
            with self.subTest(depth=depth):
                distribution = evaluator.terminal_shanten_distribution(
                    _THREE_MELD_HAND, remaining, depth
                )
                self.assertEqual(sum(distribution), _falling_factorial(hidden, depth))

    def test_mass_stays_inside_the_exact_shanten_bounds(self) -> None:
        remaining = _counts(m2=3, m6=2, s9=2, z3=1)
        hidden = sum(remaining)
        evaluator = progression._TerminalShantenProgressionEvaluator()
        for hand_counts in (_THREE_MELD_HAND, _TWO_MELD_HAND, _CLOSED_FAR_HAND):
            shanten = calculate_shanten(_tiles_for_oracle(hand_counts))
            for depth in (1, 2, DEFAULT_HORIZON):
                denominator = _falling_factorial(hidden, depth)
                distribution = evaluator.terminal_shanten_distribution(
                    hand_counts, remaining, depth
                )
                with self.subTest(hand=hand_counts, depth=depth):
                    self.assertGreaterEqual(
                        _mass(distribution),
                        max(0, shanten - depth) * denominator,
                    )
                    self.assertLessEqual(_mass(distribution), shanten * denominator)

    def test_structural_tenpai_collapses_to_zero_terminal_shanten(self) -> None:
        tenpai = _tile_type_counts(_hand("345m9s"))
        self.assertEqual(calculate_shanten(_tiles_for_oracle(tenpai)), 0)
        remaining = _counts(m2=3, m6=2, s9=2, z3=1)
        evaluator = progression._TerminalShantenProgressionEvaluator()
        distribution = evaluator.terminal_shanten_distribution(
            tenpai, remaining, DEFAULT_HORIZON
        )
        self.assertEqual(_mass(distribution), 0)
        self.assertEqual(
            distribution[0], _falling_factorial(sum(remaining), DEFAULT_HORIZON)
        )

    def test_future_draw_uses_without_replacement_inventory(self) -> None:
        """single-copy牌種は同じsequence内で二度drawされない。"""
        hand_counts = _FOUR_MELD_HAND
        remaining = _counts(m2=1, m6=1)
        evaluator = progression._TerminalShantenProgressionEvaluator()
        distribution = evaluator.terminal_shanten_distribution(
            hand_counts, remaining, 2
        )
        self.assertEqual(sum(distribution), _falling_factorial(2, 2))

    def test_hypothetical_discard_is_not_returned_to_the_inventory(self) -> None:
        evaluator = _RecordingEvaluator()
        remaining = _counts(m2=2, m6=1)
        evaluator.terminal_shanten_distribution(_THREE_MELD_HAND, remaining, 2)
        depth_one_states = [child for child in evaluator.child_states if child[1] == 1]
        self.assertTrue(depth_one_states)
        for child_remaining, _depth in depth_one_states:
            self.assertEqual(sum(child_remaining), sum(remaining) - 1)
            self.assertTrue(
                all(
                    child <= root
                    for child, root in zip(child_remaining, remaining, strict=True)
                )
            )

    def test_every_drawable_tile_type_opens_a_future_branch(self) -> None:
        """improvement drawのbranchを「即効牌」で限定していないことを固定する。"""
        hand_counts = _THREE_MELD_HAND
        shanten = calculate_shanten(_tiles_for_oracle(hand_counts))
        remaining = _counts(m2=4, m5=4, s9=4, p7=4)
        evaluator = _RecordingEvaluator()
        evaluator.terminal_shanten_distribution(hand_counts, remaining, 1)
        drawn_indices = {
            index
            for draw_hand_counts in evaluator.draw_hands
            for index in range(TILE_TYPE_COUNT)
            if draw_hand_counts[index] > hand_counts[index]
        }
        expected = {index for index, count in enumerate(remaining) if count}
        self.assertEqual(drawn_indices, expected)
        non_effective = {
            index
            for index in expected
            if calculate_shanten(
                _tiles_for_oracle(
                    tuple(
                        value + (1 if position == index else 0)
                        for position, value in enumerate(hand_counts)
                    )
                )
            )
            >= shanten
        }
        self.assertTrue(non_effective)

    def test_completion_impossible_hand_still_has_progression_value(self) -> None:
        """completion不能 != progression valueなしを固定するregression。"""
        remaining = _counts(m2=4, m5=4, m8=4)
        hidden = sum(remaining)
        completion = _FiniteHorizonEvaluator().completion_mass(
            _CLOSED_FAR_HAND, remaining, DEFAULT_HORIZON
        )
        self.assertEqual(completion, 0)
        shanten = calculate_shanten(_tiles_for_oracle(_CLOSED_FAR_HAND))
        distribution = progression._TerminalShantenProgressionEvaluator().terminal_shanten_distribution(
            _CLOSED_FAR_HAND, remaining, DEFAULT_HORIZON
        )
        self.assertLess(
            _mass(distribution),
            shanten * _falling_factorial(hidden, DEFAULT_HORIZON),
        )

    def test_selection_values_are_exact_integers(self) -> None:
        remaining = _counts(m2=3, m6=2, s9=2, z3=1)
        distribution = progression._TerminalShantenProgressionEvaluator().terminal_shanten_distribution(
            _THREE_MELD_HAND, remaining, DEFAULT_HORIZON
        )
        self.assertTrue(all(type(count) is int for count in distribution))
        self.assertIs(type(_mass(distribution)), int)


class ProgressionActivationTest(unittest.TestCase):
    """all-zero completion gateの内側だけでactivateすることを固定する。"""

    def setUp(self) -> None:
        self.concealed = _hand("147m258p369s13577z")
        self.actions = _distinct_discard_actions(self.concealed)

    def test_positive_completion_does_not_run_the_progression_dp(self) -> None:
        concealed = _hand("234m567m234p567p5s7z")
        actions = _distinct_discard_actions(concealed)
        policy_input = _make_input(concealed)
        with patch.object(
            progression,
            "_evaluate_progression_candidates",
            side_effect=AssertionError("progression must not run"),
        ):
            selected, analysis = progression._evaluate_and_choose_discard(
                policy_input, actions
            )
        self.assertFalse(analysis.progression_activated)
        self.assertEqual(analysis.candidate_evaluations, ())
        self.assertIsNone(analysis.current_all_zero_fallback_action)
        self.assertFalse(analysis.action_changed)
        self.assertIn(selected, actions)

    def test_all_zero_completion_activates_progression(self) -> None:
        policy_input = _restricted_input(
            self.concealed, tuple(f"{rank}m" for rank in range(1, 10))
        )
        _, analysis = progression._evaluate_and_choose_discard(
            policy_input, self.actions
        )
        self.assertTrue(analysis.progression_activated)
        self.assertEqual(len(analysis.candidate_evaluations), len(self.actions))
        self.assertTrue(
            all(
                candidate.completion_mass == 0
                for candidate in analysis.candidate_evaluations
            )
        )

    def test_exact_progression_tie_delegates_to_hand_value_aware(self) -> None:
        policy_input = _restricted_input(
            self.concealed, tuple(f"{rank}m" for rank in range(1, 10))
        )
        candidates = tuple(
            progression.ProgressionCandidateEvaluation(
                action=action,
                completion_mass=0,
                root_post_discard_shanten=4,
                terminal_shanten_mass=0,
                terminal_shanten_counts=(1,) + (0,) * 8,
            )
            for action in self.actions
        )
        fallback = self.actions[-1]
        self.assertIs(
            progression._select_from_progression(policy_input, candidates, fallback),
            fallback,
        )

    def test_unique_minimum_progression_mass_wins_without_fallback(self) -> None:
        policy_input = _make_input(self.concealed)
        candidates = tuple(
            progression.ProgressionCandidateEvaluation(
                action=action,
                completion_mass=0,
                root_post_discard_shanten=4,
                terminal_shanten_mass=index,
                terminal_shanten_counts=(0,) * index + (1,) + (0,) * (8 - index),
            )
            for index, action in enumerate(self.actions[:4], start=1)
        )
        with patch.object(
            progression,
            "_hand_value_aware_fallback",
            side_effect=AssertionError("fallback must not run"),
        ):
            selected = progression._select_from_progression(
                policy_input, candidates, self.actions[0]
            )
        self.assertIs(selected, candidates[0].action)

    def test_strict_progression_subset_tie_uses_only_the_tied_subset(self) -> None:
        policy_input = _make_input(self.concealed)
        masses = (5, 5, 7, 7)
        candidates = tuple(
            progression.ProgressionCandidateEvaluation(
                action=action,
                completion_mass=0,
                root_post_discard_shanten=4,
                terminal_shanten_mass=mass,
                terminal_shanten_counts=(0,) * mass + (1,) + (0,) * (8 - mass),
            )
            for action, mass in zip(self.actions[:4], masses, strict=True)
        )
        observed: list[tuple[DiscardAction, ...]] = []

        def capture(_policy_input, evaluations):
            actions = tuple(evaluation.action for evaluation in evaluations)
            observed.append(actions)
            return actions[0]

        with patch.object(progression, "_hand_value_aware_fallback", capture):
            progression._select_from_progression(
                policy_input, candidates, self.actions[0]
            )
        self.assertEqual(observed, [(candidates[0].action, candidates[1].action)])


class ParentPreservationTest(unittest.TestCase):
    """positive completion側でexact parent Actionを返すことを固定する。"""

    def setUp(self) -> None:
        self.policy = TerminalShantenProgressionMechanismRiichiDefensePolicy()
        self.parent = MechanismRiichiDefenseYakuhaiCallPolicy()

    def _decision(
        self, concealed: tuple[Tile, ...], *actions: object
    ) -> DecisionContext:
        return DecisionContext(input=_make_input(concealed), legal_actions=actions)

    def test_exact_parent_class_and_boundary(self) -> None:
        self.assertEqual(
            TerminalShantenProgressionMechanismRiichiDefensePolicy.__bases__,
            (MechanismRiichiDefenseYakuhaiCallPolicy,),
        )
        tree = ast.parse(inspect.getsource(progression))
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

    def test_positive_completion_discards_match_the_parent(self) -> None:
        cases = (
            ("unique positive maximum", "234m567m234p567p5s7z"),
            ("seven pairs tenpai", "1199m2288p3355s67z"),
            ("thirteen orphans tenpai", "119m19p19s1234567z"),
            ("ordinary two shanten", "12344m5678p123s34z"),
        )
        for label, spec in cases:
            with self.subTest(case=label):
                concealed = _hand(spec)
                actions = _distinct_discard_actions(concealed)
                decision = DecisionContext(
                    input=_make_input(concealed), legal_actions=actions
                )
                masses = _evaluate_completion_masses(
                    decision.input,
                    actions,
                    derive_remaining_tile_inventory(
                        decision.input
                    ).remaining_tile_counts,
                    DEFAULT_HORIZON,
                    _FiniteHorizonEvaluator(),
                )
                self.assertGreater(
                    max(evaluation.completion_mass for evaluation in masses), 0
                )
                self.assertEqual(
                    self.policy.choose_action(decision),
                    self.parent.choose_action(decision),
                )

    def test_positive_completion_tie_matches_the_parent(self) -> None:
        concealed = _hand("234m567m234p567p5s7z")
        actions = _distinct_discard_actions(concealed)
        decision = DecisionContext(input=_make_input(concealed), legal_actions=actions)
        self.assertEqual(
            self.policy.choose_action(decision), self.parent.choose_action(decision)
        )

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
                (white,) * 4 + _hand("123456m789p1s"), ankan, _discard(white)
            ),
            self._decision(_hand("123456m789p19s") + (white, white), pass_action, pon),
        )
        for decision in decisions:
            with self.subTest(actions=decision.legal_actions):
                self.assertEqual(
                    self.policy.choose_action(decision),
                    self.parent.choose_action(decision),
                )

    def test_defense_filtering_is_unchanged_outside_the_all_zero_branch(self) -> None:
        cases = (
            ("riichi and best tenpai", "234m567m234p567p5s7z", (("1m",),)),
            ("riichi and best one shanten", "345m56679s333577z", (("1m",),)),
            (
                "riichi and legal common genbutsu",
                "345m56679s333517z",
                (("3m",),),
            ),
        )
        for label, spec, threat_specs in cases:
            with self.subTest(case=label):
                concealed = _hand(spec)
                actions = _distinct_discard_actions(concealed)
                players = [_player() for _ in range(4)]
                for offset, river in enumerate(threat_specs, start=1):
                    players[offset] = _player(
                        discards=_river(tuple(_hand(item)[0] for item in river)),
                        riichi=RiichiState.ACCEPTED,
                    )
                decision = DecisionContext(
                    input=_make_input(concealed, players=tuple(players)),
                    legal_actions=actions,
                )
                self.assertEqual(
                    self.policy.choose_action(decision),
                    self.parent.choose_action(decision),
                )

    def test_mechanism_filter_reaches_the_progression_stage_unchanged(self) -> None:
        concealed = _hand("147m258p369s13577z")
        actions = _distinct_discard_actions(concealed)
        policy_input = _make_input(concealed)
        observed: list[tuple[DiscardAction, ...]] = []
        with (
            patch.object(
                progression,
                "_mechanism_defense_eligible_actions",
                return_value=(actions[0],),
            ),
            patch.object(
                progression,
                "_evaluate_and_choose_discard",
                side_effect=lambda _, eligible: (
                    observed.append(eligible) or (eligible[0], None)
                ),
            ),
        ):
            chosen = self.policy._decide_discard(policy_input, actions).action
        self.assertIs(chosen, actions[0])
        self.assertEqual(observed, [(actions[0],)])

    def test_legal_action_order_does_not_change_the_selected_identity(self) -> None:
        concealed = _hand("147m258p369s13577z")
        actions = _distinct_discard_actions(concealed)[:4]
        policy_input = _restricted_input(
            concealed, tuple(f"{rank}m" for rank in range(1, 6))
        )
        selected = {
            progression._evaluate_and_choose_discard(policy_input, permutation)[0]
            for permutation in itertools.permutations(actions)
        }
        self.assertEqual(len(selected), 1)

    def test_public_export_pickle_repeat_and_statelessness(self) -> None:
        self.assertIs(
            pickle.loads(pickle.dumps(self.policy)).__class__,
            TerminalShantenProgressionMechanismRiichiDefensePolicy,
        )
        self.assertEqual(vars(self.policy), {})
        decision = self._decision((), PassAction(Seat.SEAT_0))
        self.assertIs(
            self.policy.choose_action(decision), self.policy.choose_action(decision)
        )

    def test_discard_decision_analysis_stays_none(self) -> None:
        concealed = _hand("234m567m234p567p5s7z")
        actions = _distinct_discard_actions(concealed)
        decision = self._decision(concealed, *actions)
        self.assertIsNone(self.policy.choose_action_with_analysis(decision).analysis)


class InformationBoundaryTest(unittest.TestCase):
    """Policy-visible情報だけを使うことと、fail closed境界を固定する。"""

    def test_remaining_inventory_comes_from_the_shared_derivation(self) -> None:
        concealed = _hand("147m258p369s13577z")
        actions = _distinct_discard_actions(concealed)
        policy_input = _restricted_input(
            concealed, tuple(f"{rank}m" for rank in range(1, 6))
        )
        expected = derive_remaining_tile_inventory(policy_input).remaining_tile_counts
        recorded: list[_RecordingEvaluator] = []

        def build() -> _RecordingEvaluator:
            evaluator = _RecordingEvaluator()
            recorded.append(evaluator)
            return evaluator

        with patch.object(progression, "_TerminalShantenProgressionEvaluator", build):
            progression._evaluate_and_choose_discard(policy_input, actions)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(len(recorded[0].root_queries), len(actions))
        self.assertTrue(all(counts == expected for counts in recorded[0].root_queries))

    def test_red_and_normal_five_share_one_structural_state(self) -> None:
        evaluator = progression._TerminalShantenProgressionEvaluator()
        remaining = _counts(m2=3, m6=2, s9=2)
        red = _tile_type_counts(_hand("30m9s") + (_tile(TileCategory.SOUZU, 9),))
        normal = _tile_type_counts(_hand("35m9s") + (_tile(TileCategory.SOUZU, 9),))
        self.assertEqual(red, normal)
        self.assertEqual(
            evaluator.terminal_shanten_distribution(red, remaining, 2),
            evaluator.terminal_shanten_distribution(normal, remaining, 2),
        )

    def test_open_hand_fixed_meld_context_uses_current_shanten_semantics(
        self,
    ) -> None:
        evaluator = progression._TerminalShantenProgressionEvaluator()
        remaining = _counts(m2=3, m6=2, s9=2)
        distribution = evaluator.terminal_shanten_distribution(
            _THREE_MELD_HAND, remaining, 1
        )
        self.assertEqual(
            distribution, _reference_distribution(_THREE_MELD_HAND, remaining, 1)
        )
        self.assertEqual(
            evaluator.shanten(_THREE_MELD_HAND),
            calculate_shanten(_tiles_for_oracle(_THREE_MELD_HAND)),
        )

    def test_malformed_inventory_fails_closed(self) -> None:
        concealed = _hand("147m258p369s13577z")
        actions = _distinct_discard_actions(concealed)
        impossible = _player(discards=_river((_tile(TileCategory.MANZU, 1),) * 4))
        policy_input = _make_input(
            concealed,
            players=(_player(), impossible, _player(), _player()),
        )
        with self.assertRaises(ValueError):
            progression._evaluate_and_choose_discard(policy_input, actions)

    def test_short_hidden_inventory_fails_closed(self) -> None:
        concealed = _hand("147m258p369s13577z")
        actions = _distinct_discard_actions(concealed)
        policy_input = _make_input(concealed)
        with patch.object(
            progression, "_root_remaining_counts", return_value=_counts(m2=2)
        ):
            with self.assertRaises(FiniteHorizonCompletionPolicyError):
                progression._evaluate_and_choose_discard(policy_input, actions)

    def test_out_of_axis_terminal_shanten_fails_closed(self) -> None:
        with self.assertRaises(progression.TerminalShantenProgressionPolicyError):
            progression._unit_distribution(progression.TERMINAL_SHANTEN_AXIS)
        with self.assertRaises(progression.TerminalShantenProgressionPolicyError):
            progression._unit_distribution(-1)


class AnalysisValueTest(unittest.TestCase):
    """typed diagnostics valueのvalidationを固定する。"""

    def setUp(self) -> None:
        self.action = _discard(_tile(TileCategory.MANZU, 1))
        self.other = _discard(_tile(TileCategory.MANZU, 2))

    def _candidate(self, **overrides: object) -> object:
        values: dict[str, object] = {
            "action": self.action,
            "completion_mass": 0,
            "root_post_discard_shanten": 4,
            "terminal_shanten_mass": 6,
            "terminal_shanten_counts": (0, 0, 3, 0, 0, 0, 0, 0, 0),
        }
        values.update(overrides)
        return progression.ProgressionCandidateEvaluation(**values)

    def _analysis(self, **overrides: object):
        values: dict[str, object] = {
            "horizon": DEFAULT_HORIZON,
            "hidden_tile_count": 10,
            "sequence_denominator": 3,
            "progression_activated": True,
            "selected_action": self.action,
            "current_all_zero_fallback_action": self.action,
            "action_changed": False,
            "candidate_evaluations": (self._candidate(),),
        }
        values.update(overrides)
        return progression.ProgressionDecisionAnalysis(**values)

    def test_valid_candidate_and_analysis(self) -> None:
        self.assertEqual(self._analysis().candidate_evaluations[0].action, self.action)

    def test_mass_must_match_the_distribution(self) -> None:
        with self.assertRaises(ValueError):
            self._candidate(terminal_shanten_mass=5)

    def test_distribution_axis_is_fixed(self) -> None:
        with self.assertRaises(ValueError):
            self._candidate(terminal_shanten_counts=(0, 0, 3))

    def test_negative_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._candidate(completion_mass=-1)

    def test_inactive_analysis_must_not_carry_candidates(self) -> None:
        with self.assertRaises(ValueError):
            self._analysis(progression_activated=False)

    def test_active_analysis_requires_zero_completion_mass(self) -> None:
        with self.assertRaises(ValueError):
            self._analysis(candidate_evaluations=(self._candidate(completion_mass=1),))

    def test_active_analysis_requires_denominator_consistency(self) -> None:
        with self.assertRaises(ValueError):
            self._analysis(sequence_denominator=4)

    def test_action_changed_must_describe_the_selection(self) -> None:
        with self.assertRaises(ValueError):
            self._analysis(action_changed=True)
        self.assertTrue(
            self._analysis(
                selected_action=self.other,
                current_all_zero_fallback_action=self.action,
                action_changed=True,
            ).action_changed
        )


class ImprovementDrawCoreTest(unittest.TestCase):
    """Issue #169の中心仮説をそのまま固定するcore regression。

    固定する局面は次のとおりである。

    ```text
    純手牌   233368899m 26p 38s 5z
    drawable 1m..9m だけ（pinzu / souzu / 字牌はすべて公開河で見えている）
    ```

    全打牌候補のFiniteHorizon completion massは0で、打牌後向聴数は3以上になる。
    `2p`切りと`3m`切りは

    ```text
    打牌後向聴数  3 で同じ
    現在受け入れ  6 で同じ
    ```

    なので、immediate effective tileの観点ではAが優位ではない。既存
    `TwoStepUkeirePolicy`とcurrent baselineのall-zero fallbackはどちらも`3m`切りを
    選ぶ。

    一方、`2p`切り（死に牌を捨てて第3の`3m`を残す）側では、

    ```text
    4m / 5m / 7m 等をtsumo
        ↓ 向聴数は3のまま
        ↓ best structural discard後の受け入れが 6 -> 27 へ増える
        ↓ 次のtsumoで2向聴、3向聴目で更に前進
    ```

    というnon-effective improvement drawのtrajectoryが多く残る。`3m`切り側の
    improvement drawは本数も改善幅も小さい。horizon=3のexpected terminal shanten
    はこの差を拾うので、新Policyだけが`2p`切りを選ぶ。
    """

    CONCEALED_SPEC = "233368899m26p38s5z"
    DRAWABLE_SPECS = tuple(f"{rank}m" for rank in range(1, 10))
    CANDIDATE_A_SPEC = "2p"
    CANDIDATE_B_SPEC = "3m"

    def setUp(self) -> None:
        self.concealed = _hand(self.CONCEALED_SPEC)
        self.actions = _distinct_discard_actions(self.concealed)
        self.policy_input = _restricted_input(self.concealed, self.DRAWABLE_SPECS)
        self.known_counts = _known_tile_counts(self.policy_input)
        self.remaining = derive_remaining_tile_inventory(
            self.policy_input
        ).remaining_tile_counts
        self.candidate_a = _discard(_hand(self.CANDIDATE_A_SPEC)[0])
        self.candidate_b = _discard(_hand(self.CANDIDATE_B_SPEC)[0])

    def _post_discard_hand(self, discarded: DiscardAction) -> tuple[Tile, ...]:
        remaining = list(self.concealed)
        remaining.remove(discarded.tile)
        return tuple(remaining)

    def _ukeire(self, hand: tuple[Tile, ...]) -> int:
        return _ukeire_count(hand, self.known_counts, calculate_shanten(hand), None)

    def _improvement_draws(self, hand: tuple[Tile, ...]) -> dict[TileType, int]:
        """向聴数を下げないのに、best structural discard後の受け入れを増やすdraw。"""
        base_shanten = calculate_shanten(hand)
        base_ukeire = self._ukeire(hand)
        improvements: dict[TileType, int] = {}
        for index, count in enumerate(self.remaining):
            if count == 0:
                continue
            drawn = _tiles_for_oracle(
                tuple(1 if position == index else 0 for position in range(34))
            )[0]
            draw_hand = (*hand, drawn)
            if calculate_shanten(draw_hand) < base_shanten:
                continue
            best = base_ukeire
            for position in range(len(draw_hand)):
                child = draw_hand[:position] + draw_hand[position + 1 :]
                if calculate_shanten(child) != base_shanten:
                    continue
                best = max(best, self._ukeire(child))
            if best > base_ukeire:
                improvements[drawn.tile_type] = best
        return improvements

    def test_progression_prefers_the_improvement_draw_candidate(self) -> None:
        selected, analysis = progression._evaluate_and_choose_discard(
            self.policy_input, self.actions
        )
        self.assertTrue(analysis.progression_activated)
        self.assertTrue(analysis.action_changed)
        self.assertEqual(selected, self.candidate_a)
        self.assertEqual(analysis.current_all_zero_fallback_action, self.candidate_b)

        by_action = {
            candidate.action: candidate for candidate in analysis.candidate_evaluations
        }
        first = by_action[self.candidate_a]
        second = by_action[self.candidate_b]
        self.assertEqual(
            (first.root_post_discard_shanten, second.root_post_discard_shanten),
            (3, 3),
        )
        self.assertLess(first.terminal_shanten_mass, second.terminal_shanten_mass)
        self.assertEqual(
            first.terminal_shanten_mass,
            min(
                candidate.terminal_shanten_mass
                for candidate in analysis.candidate_evaluations
            ),
        )
        self.assertTrue(
            all(
                candidate.completion_mass == 0
                for candidate in analysis.candidate_evaluations
            )
        )

    def test_immediate_and_two_step_views_do_not_prefer_the_candidate(self) -> None:
        hand_a = self._post_discard_hand(self.candidate_a)
        hand_b = self._post_discard_hand(self.candidate_b)
        self.assertEqual(calculate_shanten(hand_a), calculate_shanten(hand_b))
        self.assertEqual(self._ukeire(hand_a), self._ukeire(hand_b))
        decision = DecisionContext(input=self.policy_input, legal_actions=self.actions)
        self.assertEqual(
            TwoStepUkeirePolicy().choose_action(decision), self.candidate_b
        )
        self.assertEqual(
            MechanismRiichiDefenseYakuhaiCallPolicy().choose_action(decision),
            self.candidate_b,
        )

    def test_the_advantage_comes_from_non_effective_improvement_draws(self) -> None:
        hand_a = self._post_discard_hand(self.candidate_a)
        hand_b = self._post_discard_hand(self.candidate_b)
        improvements_a = self._improvement_draws(hand_a)
        improvements_b = self._improvement_draws(hand_b)
        self.assertTrue(improvements_a)
        self.assertGreater(len(improvements_a), len(improvements_b))
        self.assertGreater(max(improvements_a.values()), max(improvements_b.values()))
        self.assertGreater(max(improvements_a.values()), self._ukeire(hand_a))

    def test_full_policy_selects_the_improvement_draw_candidate(self) -> None:
        decision = DecisionContext(input=self.policy_input, legal_actions=self.actions)
        self.assertEqual(
            TerminalShantenProgressionMechanismRiichiDefensePolicy().choose_action(
                decision
            ),
            self.candidate_a,
        )


if __name__ == "__main__":
    unittest.main()
