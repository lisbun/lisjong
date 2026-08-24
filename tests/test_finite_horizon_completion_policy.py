"""Issue #109 `FiniteHorizonCompletionPolicy`のunit test。"""

import ast
import inspect
import itertools
import pickle
import random
import unittest
from dataclasses import FrozenInstanceError, fields
from unittest.mock import patch

import lisjong.policies.finite_horizon_completion as finite_horizon
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.belief.tile_inventory import TILE_TYPE_COUNT
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies import FiniteHorizonCompletionPolicy, TwoStepUkeirePolicy
from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    FiniteHorizonCandidateEvaluation,
    FiniteHorizonCompletionAnalysis,
    FiniteHorizonCompletionPolicyError,
    _evaluate_and_choose_discard,
    _falling_factorial,
    _FiniteHorizonEvaluator,
    _root_remaining_counts,
    _tile_type_counts,
)
from lisjong.policies.two_step_ukeire import (
    TwoStepUkeireAnalysis,
    TwoStepUkeirePolicyError,
    _remove_one_matching_tile,
)
from lisjong.policies.two_step_ukeire import (
    _evaluate_and_choose_discard as two_step_evaluate_and_choose_discard,
)
from lisjong.policy_contract.action import (
    DiscardAction,
    PassAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.decision_trace import DecisionTraceRecorder
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_execution import (
    execute_policy,
    execute_policy_with_trace,
)
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

_CATEGORY_OFFSETS = {
    TileCategory.MANZU: 0,
    TileCategory.PINZU: 9,
    TileCategory.SOUZU: 18,
    TileCategory.HONOR: 27,
}


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


def _index(category: TileCategory, rank: int) -> int:
    return _CATEGORY_OFFSETS[category] + rank - 1


def _counts(**tiles: int) -> tuple[int, ...]:
    """`m3=1`のようなkeyword指定から34牌種countを組み立てる。"""
    categories = {
        "m": TileCategory.MANZU,
        "p": TileCategory.PINZU,
        "s": TileCategory.SOUZU,
        "z": TileCategory.HONOR,
    }
    counts = [0] * TILE_TYPE_COUNT
    for key, count in tiles.items():
        counts[_index(categories[key[0]], int(key[1:]))] += count
    return tuple(counts)


MANZU_3 = _tile(TileCategory.MANZU, 3)
MANZU_4 = _tile(TileCategory.MANZU, 4)
MANZU_5 = _tile(TileCategory.MANZU, 5)
MANZU_5_RED = _tile(TileCategory.MANZU, 5, red=True)
PINZU_1 = _tile(TileCategory.PINZU, 1)
SOUZU_2 = _tile(TileCategory.SOUZU, 2)
SOUZU_5 = _tile(TileCategory.SOUZU, 5)
SOUZU_9 = _tile(TileCategory.SOUZU, 9)
EAST = _tile(TileCategory.HONOR, 1)
RED_DRAGON = _tile(TileCategory.HONOR, 7)

_TENPAI_HAND = _hand("234m567m234p567p5s7z")
"""7z切りで5s単騎、5s切りで7z単騎になる14枚。"""

_CHIITOITSU_HAND = _hand("1199m2288p3355s67z")
"""6対子 + 6z + 7zで、7z切りが6z待ちの七対子聴牌になる14枚。"""

_KOKUSHI_HAND = _hand("119m19p19s1234567z")
"""1m切りで13面待ち国士無双聴牌になる14枚。"""

_FAR_FROM_COMPLETION_HAND = _hand("147m258p369s13577z")
"""どの打牌後もhorizon 3では完成できない、向聴数の深い14枚。"""


def _player(
    discards: tuple[Discard, ...] = (), melds: tuple[PublicMeld, ...] = ()
) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=discards,
        melds=melds,
        riichi=RiichiState.NONE,
    )


def _make_input(
    concealed_tiles: tuple[Tile, ...],
    *,
    players: tuple[PlayerPublicState, ...] | None = None,
    dora_indicators: tuple[Tile, ...] = (),
    live_wall_tiles_remaining: int = 70,
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
            live_wall_tiles_remaining=live_wall_tiles_remaining,
        ),
        players=players if players is not None else (_player(),) * 4,
        own_hand=OwnHandState(concealed_tiles=concealed_tiles, drawn_tile=None),
    )


def _decision(
    concealed_tiles: tuple[Tile, ...],
    actions: tuple[object, ...],
    *,
    players: tuple[PlayerPublicState, ...] | None = None,
) -> DecisionContext:
    return DecisionContext(
        input=_make_input(concealed_tiles, players=players),
        legal_actions=actions,
    )


def _discard(tile: Tile, *, tsumogiri: bool = False) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=tsumogiri)


_OPEN_HAND_CONCEALED = _hand("345m99s")
"""3副露済みで純手牌5枚。9s切りで9s待ち、3m切りで3m/6m待ちになる。"""

_RED_FIVE_CONCEALED = _hand("3045m") + (SOUZU_9,)
"""3副露済みで、赤5mと通常5mの両方を持つ純手牌5枚。"""

_MAX_COPIES_PER_TILE_TYPE = 4
"""1牌種あたりのphysical上限枚数（corpus生成用のtest-local定数）。"""

_ISOLATED_OPEN_HAND = _tile_type_counts(_hand("1m5p9s3z"))
"""3副露 + 孤立牌4枚。どの牌をtsumoしても打牌後2向聴のままになる。"""

_ONE_SHANTEN_OPEN_HAND = _tile_type_counts(_hand("34m9s1z"))
"""3副露 + 1向聴の純手牌4枚。無関係牌をtsumoしても1向聴を維持できる。"""

_TENPAI_OPEN_HAND = _tile_type_counts(_hand("345m9s"))
"""3副露 + 9s単騎聴牌の純手牌4枚。"""


def _open_hand_players() -> tuple[PlayerPublicState, ...]:
    """自席3副露と、対応する被副露discardを持つ牌保存則整合のplayers。"""
    melds = (
        PublicMeld(
            kind=MeldKind.PON,
            tiles=(PINZU_1, PINZU_1, PINZU_1),
            from_seat=Seat.SEAT_1,
            called_tile=PINZU_1,
        ),
        PublicMeld(
            kind=MeldKind.PON,
            tiles=(SOUZU_2, SOUZU_2, SOUZU_2),
            from_seat=Seat.SEAT_2,
            called_tile=SOUZU_2,
        ),
        PublicMeld(
            kind=MeldKind.PON,
            tiles=(EAST, EAST, EAST),
            from_seat=Seat.SEAT_3,
            called_tile=EAST,
        ),
    )
    return (
        _player(melds=melds),
        _player(
            (Discard(tile=PINZU_1, tsumogiri=False, order=0, called_by=Seat.SEAT_0),)
        ),
        _player(
            (Discard(tile=SOUZU_2, tsumogiri=False, order=1, called_by=Seat.SEAT_0),)
        ),
        _player((Discard(tile=EAST, tsumogiri=False, order=2, called_by=Seat.SEAT_0),)),
    )


def _open_hand_input() -> PolicyInput:
    return _make_input(_OPEN_HAND_CONCEALED, players=_open_hand_players())


def _open_hand_decision(actions: tuple[object, ...]) -> DecisionContext:
    return DecisionContext(input=_open_hand_input(), legal_actions=actions)


def _tiles_for_oracle(counts: tuple[int, ...]) -> list[Tile]:
    """test側だけで使う、34牌種countから`Tile`列への独立変換。"""
    tiles: list[Tile] = []
    for index, count in enumerate(counts):
        for category, offset, size in (
            (TileCategory.MANZU, 0, 9),
            (TileCategory.PINZU, 9, 9),
            (TileCategory.SOUZU, 18, 9),
            (TileCategory.HONOR, 27, 7),
        ):
            if offset <= index < offset + size:
                tiles.extend([_tile(category, index - offset + 1)] * count)
                break
    return tiles


def _uncached_oracle_mass(
    hand_counts: tuple[int, ...], remaining_counts: tuple[int, ...], depth: int
) -> int:
    """memoization・枝刈りを一切持たないtest-local brute-force reference。

    productionへ`use_cache`のような二重execution pathを追加せず、
    「production memoized DP == uncached oracle」を確認するためだけの
    独立実装である。
    """
    if depth <= 0:
        return 0
    hidden = sum(remaining_counts)
    total = 0
    for drawn in range(TILE_TYPE_COUNT):
        available = remaining_counts[drawn]
        if available == 0:
            continue
        draw_hand = list(hand_counts)
        draw_hand[drawn] += 1
        draw_hand_counts = tuple(draw_hand)
        if calculate_shanten(_tiles_for_oracle(draw_hand_counts)) == -1:
            suffix = 1
            for step in range(depth - 1):
                suffix *= hidden - 1 - step
            total += available * suffix
            continue
        next_remaining = list(remaining_counts)
        next_remaining[drawn] -= 1
        next_remaining_counts = tuple(next_remaining)
        best = 0
        for discarded in range(TILE_TYPE_COUNT):
            if draw_hand_counts[discarded] == 0:
                continue
            next_hand = list(draw_hand_counts)
            next_hand[discarded] -= 1
            best = max(
                best,
                _uncached_oracle_mass(
                    tuple(next_hand), next_remaining_counts, depth - 1
                ),
            )
        total += available * best
    return total


class _RecordedEvaluation:
    """1 decision分のDP呼び出しを記録するcontext helper。"""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []

    def __enter__(self) -> "_RecordedEvaluation":
        original = _FiniteHorizonEvaluator.completion_mass
        calls = self.calls

        def recording(evaluator, hand_counts, remaining_counts, depth):
            calls.append((hand_counts, remaining_counts, depth))
            return original(evaluator, hand_counts, remaining_counts, depth)

        self._patcher = patch.object(
            _FiniteHorizonEvaluator, "completion_mass", recording
        )
        self._patcher.start()
        return self

    def __exit__(self, *exception_info: object) -> None:
        self._patcher.stop()


class FallingFactorialTest(unittest.TestCase):
    def test_ordered_draw_sequence_count(self) -> None:
        self.assertEqual(_falling_factorial(122, 0), 1)
        self.assertEqual(_falling_factorial(122, 1), 122)
        self.assertEqual(_falling_factorial(122, 3), 122 * 121 * 120)
        self.assertEqual(_falling_factorial(5, 3), 60)

    def test_denominator_matches_the_analysis_contract(self) -> None:
        policy_input = _make_input(_TENPAI_HAND)
        remaining = _root_remaining_counts(policy_input)

        _, analysis = _evaluate_and_choose_discard(
            policy_input, (_discard(RED_DRAGON),), horizon=1
        )

        self.assertEqual(analysis.hidden_tile_count, sum(remaining))
        self.assertEqual(
            analysis.sequence_denominator, _falling_factorial(sum(remaining), 1)
        )


class CanonicalCountRepresentationTest(unittest.TestCase):
    def test_red_and_normal_five_share_one_structural_tile_type(self) -> None:
        red = _tile_type_counts((MANZU_5_RED,))
        normal = _tile_type_counts((MANZU_5,))

        self.assertEqual(red, normal)
        self.assertEqual(red[_index(TileCategory.MANZU, 5)], 1)

    def test_counts_use_the_shared_canonical_34_axis(self) -> None:
        counts = _tile_type_counts(_hand("119m1z"))

        self.assertEqual(len(counts), TILE_TYPE_COUNT)
        self.assertEqual(counts[_index(TileCategory.MANZU, 1)], 2)
        self.assertEqual(counts[_index(TileCategory.MANZU, 9)], 1)
        self.assertEqual(counts[_index(TileCategory.HONOR, 1)], 1)


class CompletionMassRecurrenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = _FiniteHorizonEvaluator()

    def _post_discard(self, hand: tuple[Tile, ...], discarded: Tile) -> tuple[int, ...]:
        remaining = list(hand)
        remaining.remove(discarded)
        return _tile_type_counts(remaining)

    def test_horizon_one_tenpai_mass_equals_remaining_winning_tile_count(self) -> None:
        policy_input = _make_input(_TENPAI_HAND)
        remaining = _root_remaining_counts(policy_input)
        post_discard = self._post_discard(_TENPAI_HAND, RED_DRAGON)

        mass = self.evaluator.completion_mass(post_discard, remaining, 1)

        # 7z切りの5s単騎は、自分の1枚を除いた残り3枚だけが完成牌である。
        self.assertEqual(remaining[_index(TileCategory.SOUZU, 5)], 3)
        self.assertEqual(mass, 3)

    def test_horizon_one_mass_tracks_remaining_multiplicity(self) -> None:
        winning_tile_index = _index(TileCategory.SOUZU, 5)
        post_discard = self._post_discard(_TENPAI_HAND, RED_DRAGON)
        masses = []
        for available in (0, 1, 2, 3):
            remaining = list(_root_remaining_counts(_make_input(_TENPAI_HAND)))
            remaining[winning_tile_index] = available
            masses.append(
                self.evaluator.completion_mass(post_discard, tuple(remaining), 1)
            )

        self.assertEqual(masses, [0, 1, 2, 3])

    def test_horizon_one_chiitoitsu_route_is_recognized(self) -> None:
        policy_input = _make_input(_CHIITOITSU_HAND)
        remaining = _root_remaining_counts(policy_input)
        post_discard = self._post_discard(_CHIITOITSU_HAND, RED_DRAGON)
        green_dragon = _index(TileCategory.HONOR, 6)

        self.assertEqual(calculate_shanten(_tiles_for_oracle(post_discard)), 0)
        self.assertEqual(remaining[green_dragon], 3)
        self.assertEqual(
            self.evaluator.completion_mass(post_discard, remaining, 1),
            remaining[green_dragon],
        )

    def test_horizon_one_kokushi_route_counts_every_orphan(self) -> None:
        policy_input = _make_input(_KOKUSHI_HAND)
        remaining = _root_remaining_counts(policy_input)
        post_discard = self._post_discard(_KOKUSHI_HAND, _tile(TileCategory.MANZU, 1))
        orphan_indexes = [
            _index(TileCategory.MANZU, 1),
            _index(TileCategory.MANZU, 9),
            _index(TileCategory.PINZU, 1),
            _index(TileCategory.PINZU, 9),
            _index(TileCategory.SOUZU, 1),
            _index(TileCategory.SOUZU, 9),
            *(_index(TileCategory.HONOR, rank) for rank in range(1, 8)),
        ]

        mass = self.evaluator.completion_mass(post_discard, remaining, 1)

        # 13面待ちなので、么九13種の未見枚数の合計がそのままmassになる。
        self.assertEqual(mass, sum(remaining[index] for index in orphan_indexes))
        self.assertEqual(mass, 2 + 12 * 3)

    def test_horizon_one_open_hand_uses_the_fixed_meld_semantics(self) -> None:
        policy_input = _open_hand_input()
        remaining = _root_remaining_counts(policy_input)
        post_discard = self._post_discard(_OPEN_HAND_CONCEALED, SOUZU_9)

        self.assertEqual(sum(post_discard), 4)
        self.assertEqual(calculate_shanten(_tiles_for_oracle(post_discard)), 0)
        self.assertEqual(
            self.evaluator.completion_mass(post_discard, remaining, 1),
            remaining[_index(TileCategory.SOUZU, 9)],
        )

    def test_horizon_two_hand_computed_certain_completion(self) -> None:
        # 純手牌`345m9s` + 3副露。残りinventoryを9s2枚と1z1枚に限定すると、
        # 2回のdrawで必ず9sを引くので mass == F(3, 2) となる。
        hand = _counts(m3=1, m4=1, m5=1, s9=1)
        remaining = _counts(s9=2, z1=1)

        mass = self.evaluator.completion_mass(hand, remaining, 2)

        self.assertEqual(mass, _falling_factorial(3, 2))
        self.assertEqual(mass, 6)

    def test_horizon_two_hand_computed_prefix_completion_suffix_mass(self) -> None:
        # t=9s(1枚)で即完成したbranchは残り1 slotの内容によらず成功なので
        # F(2, 1) = 2を寄与する。t=1z(2枚)は最善のhypothetical discardでも
        # 1通りしか完成しないので 2 * 1 を寄与し、合計4になる。
        hand = _counts(m3=1, m4=1, m5=1, s9=1)
        remaining = _counts(s9=1, z1=2)

        mass = self.evaluator.completion_mass(hand, remaining, 2)

        self.assertEqual(mass, 1 * _falling_factorial(2, 1) + 2 * 1)
        self.assertEqual(mass, 4)

    def test_zero_depth_is_a_failure_for_an_incomplete_hand(self) -> None:
        hand = _counts(m3=1, m4=1, m5=1, s9=1)

        self.assertEqual(self.evaluator.completion_mass(hand, _counts(s9=2), 0), 0)

    def test_shanten_lower_bound_prunes_to_exact_zero(self) -> None:
        policy_input = _make_input(_TENPAI_HAND)
        remaining = _root_remaining_counts(policy_input)
        # 面子を崩した1向聴の打牌後手牌は、horizon 1では完成できない。
        post_discard = self._post_discard(_TENPAI_HAND, _tile(TileCategory.MANZU, 2))

        self.assertEqual(calculate_shanten(_tiles_for_oracle(post_discard)), 1)
        self.assertEqual(self.evaluator.completion_mass(post_discard, remaining, 1), 0)
        self.assertGreater(
            self.evaluator.completion_mass(post_discard, remaining, 2), 0
        )

    def test_far_from_completion_hand_has_zero_mass_within_the_horizon(self) -> None:
        policy_input = _make_input(_FAR_FROM_COMPLETION_HAND)
        remaining = _root_remaining_counts(policy_input)
        post_discard = self._post_discard(
            _FAR_FROM_COMPLETION_HAND, _tile(TileCategory.HONOR, 7)
        )

        self.assertGreater(
            calculate_shanten(_tiles_for_oracle(post_discard)) + 1, DEFAULT_HORIZON
        )
        self.assertEqual(
            self.evaluator.completion_mass(post_discard, remaining, DEFAULT_HORIZON), 0
        )

    def test_mass_never_exceeds_the_ordered_sequence_denominator(self) -> None:
        policy_input = _open_hand_input()
        remaining = _root_remaining_counts(policy_input)
        hidden = sum(remaining)

        for horizon in (1, 2, 3):
            for discarded in dict.fromkeys(_OPEN_HAND_CONCEALED):
                with self.subTest(horizon=horizon, discarded=discarded):
                    mass = self.evaluator.completion_mass(
                        self._post_discard(_OPEN_HAND_CONCEALED, discarded),
                        remaining,
                        horizon,
                    )
                    self.assertGreaterEqual(mass, 0)
                    self.assertLessEqual(mass, _falling_factorial(hidden, horizon))

    def test_red_and_normal_five_give_the_same_structural_mass(self) -> None:
        red_hand = _hand("340m") + (SOUZU_9, SOUZU_9)
        normal_hand = _hand("345m") + (SOUZU_9, SOUZU_9)
        remaining = _counts(s9=2, z1=1)

        red_mass = self.evaluator.completion_mass(
            self._post_discard(red_hand, SOUZU_9), remaining, 2
        )
        normal_mass = self.evaluator.completion_mass(
            self._post_discard(normal_hand, SOUZU_9), remaining, 2
        )

        self.assertEqual(red_mass, normal_mass)

    def test_a_future_draw_removes_exactly_the_drawn_tile(self) -> None:
        # drawした牌種だけがremaining inventoryから1枚減り、仮想discardは
        # どのbranchでもinventoryへ戻らない。
        # 2z / 3z はどちらもdraw後に七対子聴牌を保つため、depth 2の
        # exact-safe parent pruning（`draw_shanten > depth - 2`）に掛からず、
        # 仮想discard childrenが実際に生成される。
        hand = _counts(m3=3, m4=2, m5=2, s9=2, z1=2, z2=1, z3=1)
        remaining = _counts(z2=1, z3=1)

        with _RecordedEvaluation() as recorded:
            self.evaluator.completion_mass(hand, remaining, 2)

        child_inventories = {
            remaining_counts
            for _hand, remaining_counts, depth in recorded.calls
            if depth == 1
        }
        self.assertEqual(child_inventories, {_counts(z3=1), _counts(z2=1)})

    def test_hypothetical_discards_are_deduplicated_per_tile_type(self) -> None:
        # 仮想discardは牌種単位なので、drawした14枚目を含む7牌種から生まれる
        # 子stateはちょうど7件になる（同じ牌種のcopy Aとcopy Bを別branchに
        # しない）。2z drawは七対子聴牌を保つのでparent pruningに掛からず、
        # 仮想discardの列挙が実際に行われる。
        hand = _counts(m3=3, m4=2, m5=2, s9=2, z1=2, z2=1, z3=1)
        remaining = _counts(z2=1)
        drawn_hand_tile_types = 7

        self.evaluator.completion_mass(hand, remaining, 2)

        self.assertEqual(self.evaluator.visited_states, 1 + drawn_hand_tile_types)


class MemoizationEquivalenceTest(unittest.TestCase):
    """production memoized DPとtest-local uncached oracleの一致を固定する。"""

    def _assert_matches_oracle(
        self, hand: tuple[int, ...], remaining: tuple[int, ...], depth: int
    ) -> None:
        evaluator = _FiniteHorizonEvaluator()

        self.assertEqual(
            evaluator.completion_mass(hand, remaining, depth),
            _uncached_oracle_mass(hand, remaining, depth),
        )

    def test_horizon_one_matches_the_uncached_oracle(self) -> None:
        hand = _counts(m3=1, m4=1, m5=1, s9=1)
        self._assert_matches_oracle(hand, _counts(s9=2, z1=1, m6=4), 1)

    def test_horizon_two_matches_the_uncached_oracle(self) -> None:
        hand = _counts(m3=1, m4=1, m5=1, s9=1)
        for remaining in (
            _counts(s9=2, z1=1),
            _counts(s9=1, z1=2),
            _counts(m3=3, m6=4, s9=2, z1=2),
        ):
            with self.subTest(remaining=remaining):
                self._assert_matches_oracle(hand, remaining, 2)

    def test_horizon_two_matches_the_oracle_for_a_seven_tile_hand(self) -> None:
        hand = _counts(m3=1, m4=1, m5=1, p7=1, p8=1, s9=1, z1=1)
        self._assert_matches_oracle(hand, _counts(p6=2, p9=2, s9=1, z1=1), 2)


class TranspositionCacheTest(unittest.TestCase):
    def test_the_same_state_is_computed_only_once(self) -> None:
        evaluator = _FiniteHorizonEvaluator()
        hand = _counts(m3=1, m4=1, m5=1, s9=1)
        remaining = _counts(s9=2, z1=1)

        first = evaluator.completion_mass(hand, remaining, 2)
        misses_after_first = evaluator.cache_misses
        hits_after_first = evaluator.cache_hits
        second = evaluator.completion_mass(hand, remaining, 2)

        self.assertEqual(first, second)
        self.assertEqual(evaluator.cache_misses, misses_after_first)
        self.assertEqual(evaluator.cache_hits, hits_after_first + 1)

    def test_root_candidates_share_one_decision_local_cache(self) -> None:
        # 異なるroot discardからでも、同じfuture stateへ合流したbranchは
        # 共有cacheで再利用される。
        evaluator = _FiniteHorizonEvaluator()
        remaining = _counts(m3=3, m6=4, s9=2, z1=2)
        keep_nine_souzu = _counts(m3=1, m4=1, m5=1, s9=1)
        keep_east = _counts(m3=1, m4=1, m5=1, z1=1)

        evaluator.completion_mass(keep_nine_souzu, remaining, 2)
        hits_after_first_root = evaluator.cache_hits
        evaluator.completion_mass(keep_east, remaining, 2)

        self.assertGreater(evaluator.cache_hits, hits_after_first_root)

    def test_one_decision_creates_exactly_one_shared_evaluator(self) -> None:
        policy_input = _open_hand_input()
        discard_actions = tuple(
            _discard(tile) for tile in dict.fromkeys(_OPEN_HAND_CONCEALED)
        )

        with patch.object(
            finite_horizon,
            "_FiniteHorizonEvaluator",
            wraps=_FiniteHorizonEvaluator,
        ) as evaluator_type:
            _evaluate_and_choose_discard(policy_input, discard_actions, horizon=2)

        self.assertEqual(evaluator_type.call_count, 1)

    def test_policy_keeps_no_cross_decision_cache(self) -> None:
        policy = FiniteHorizonCompletionPolicy()
        decision = _open_hand_decision(
            tuple(_discard(tile) for tile in dict.fromkeys(_OPEN_HAND_CONCEALED))
        )

        policy.choose_action(decision)

        self.assertEqual(vars(policy), {})
        self.assertFalse(hasattr(policy, "last_analysis"))


class RootRemainingInventoryTest(unittest.TestCase):
    def test_root_inventory_is_the_issue_63_remaining_tile_counts(self) -> None:
        policy_input = _make_input(_TENPAI_HAND)

        self.assertEqual(
            _root_remaining_counts(policy_input),
            derive_remaining_tile_inventory(policy_input).remaining_tile_counts,
        )

    def test_root_inventory_is_not_the_live_wall(self) -> None:
        policy_input = _make_input(_TENPAI_HAND, live_wall_tiles_remaining=70)

        remaining = _root_remaining_counts(policy_input)

        self.assertNotEqual(
            sum(remaining), policy_input.round.live_wall_tiles_remaining
        )
        self.assertEqual(sum(remaining), 136 - len(_TENPAI_HAND))

    def test_live_wall_count_does_not_change_the_decision(self) -> None:
        discard_actions = tuple(
            _discard(tile) for tile in dict.fromkeys(_OPEN_HAND_CONCEALED)
        )
        selections = []
        for live_wall in (70, 12, 0):
            policy_input = _make_input(
                _OPEN_HAND_CONCEALED,
                players=_open_hand_players(),
                live_wall_tiles_remaining=live_wall,
            )
            selected, analysis = _evaluate_and_choose_discard(
                policy_input, discard_actions, horizon=2
            )
            selections.append(
                (
                    selected,
                    tuple(
                        evaluation.completion_mass
                        for evaluation in analysis.candidate_evaluations
                    ),
                )
            )

        self.assertEqual(len(set(selections)), 1)

    def test_every_root_candidate_uses_the_same_remaining_inventory(self) -> None:
        policy_input = _open_hand_input()
        root_remaining = _root_remaining_counts(policy_input)
        discard_actions = tuple(
            _discard(tile) for tile in dict.fromkeys(_OPEN_HAND_CONCEALED)
        )

        with _RecordedEvaluation() as recorded:
            _evaluate_and_choose_discard(policy_input, discard_actions, horizon=2)

        root_calls = [call for call in recorded.calls if call[2] == 2]
        self.assertEqual(len(root_calls), len(discard_actions))
        for _hand_counts, remaining_counts, _depth in root_calls:
            self.assertEqual(remaining_counts, root_remaining)

    def test_root_discard_is_not_returned_to_the_remaining_inventory(self) -> None:
        policy_input = _open_hand_input()
        root_remaining = _root_remaining_counts(policy_input)
        souzu_nine = _index(TileCategory.SOUZU, 9)

        with _RecordedEvaluation() as recorded:
            _evaluate_and_choose_discard(policy_input, (_discard(SOUZU_9),), horizon=2)

        (root_call,) = [call for call in recorded.calls if call[2] == 2]
        self.assertEqual(root_call[1][souzu_nine], root_remaining[souzu_nine])
        self.assertEqual(root_call[1][souzu_nine], 2)

    def test_future_draws_remove_one_tile_and_discards_never_restore_it(self) -> None:
        policy_input = _open_hand_input()
        root_remaining = _root_remaining_counts(policy_input)
        hidden = sum(root_remaining)
        horizon = 2

        with _RecordedEvaluation() as recorded:
            _evaluate_and_choose_discard(
                policy_input,
                tuple(_discard(tile) for tile in dict.fromkeys(_OPEN_HAND_CONCEALED)),
                horizon=horizon,
            )

        self.assertGreater(len(recorded.calls), len(_OPEN_HAND_CONCEALED))
        for _hand_counts, remaining_counts, depth in recorded.calls:
            # depthが1段進むごとにhidden inventoryはちょうど1枚減り、
            # hypothetical discardで戻ることはない。
            self.assertEqual(sum(remaining_counts), hidden - (horizon - depth))
            for index, count in enumerate(remaining_counts):
                self.assertLessEqual(count, root_remaining[index])


class RootSelectionPrecedenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy_input = _make_input(_TENPAI_HAND)
        self.discard_5s = _discard(SOUZU_5)
        self.discard_7z = _discard(RED_DRAGON)
        self.discard_2m = _discard(_tile(TileCategory.MANZU, 2))
        self.discard_actions = (self.discard_5s, self.discard_7z, self.discard_2m)

    def _with_masses(self, masses: dict[Tile, int]):
        """DP massだけを固定し、selection precedenceを独立に検証する。"""

        def fake_evaluate(
            policy_input, discard_actions, remaining_counts, horizon, evaluator
        ):
            return tuple(
                FiniteHorizonCandidateEvaluation(
                    action=action, completion_mass=masses[action.tile]
                )
                for action in sorted(
                    discard_actions, key=finite_horizon._discard_action_sort_key
                )
            )

        return patch.object(
            finite_horizon, "_evaluate_completion_masses", fake_evaluate
        )

    def test_unique_positive_maximum_skips_the_two_step_fallback(self) -> None:
        masses = {SOUZU_5: 5, RED_DRAGON: 9, _tile(TileCategory.MANZU, 2): 1}

        with (
            self._with_masses(masses),
            patch.object(
                finite_horizon,
                "_two_step_evaluate_and_choose_discard",
                wraps=finite_horizon._two_step_evaluate_and_choose_discard,
            ) as two_step,
        ):
            selected, analysis = _evaluate_and_choose_discard(
                self.policy_input, self.discard_actions
            )

        self.assertIs(selected, self.discard_7z)
        self.assertEqual(two_step.call_count, 0)
        self.assertIsNone(analysis.two_step_tiebreak_analysis)

    def test_positive_exact_tie_passes_only_the_maximum_mass_subset(self) -> None:
        masses = {SOUZU_5: 9, RED_DRAGON: 9, _tile(TileCategory.MANZU, 2): 8}

        with (
            self._with_masses(masses),
            patch.object(
                finite_horizon,
                "_two_step_evaluate_and_choose_discard",
                wraps=finite_horizon._two_step_evaluate_and_choose_discard,
            ) as two_step,
        ):
            selected, analysis = _evaluate_and_choose_discard(
                self.policy_input, self.discard_actions
            )

        self.assertEqual(two_step.call_count, 1)
        self.assertEqual(two_step.call_args.args[1], (self.discard_5s, self.discard_7z))
        self.assertIn(selected, (self.discard_5s, self.discard_7z))
        self.assertIsInstance(
            analysis.two_step_tiebreak_analysis, TwoStepUkeireAnalysis
        )

    def test_a_losing_candidate_is_not_revived_by_the_two_step_fallback(self) -> None:
        # 2m切りは打牌後向聴数が最小ではないが、TwoStepだけならtie-break対象に
        # なり得る。completion massで負けた候補をfallbackで復活させない。
        masses = {SOUZU_5: 9, RED_DRAGON: 9, _tile(TileCategory.MANZU, 2): 9999}
        with self._with_masses(masses):
            selected, _ = _evaluate_and_choose_discard(
                self.policy_input, self.discard_actions
            )
        self.assertIs(selected, self.discard_2m)

        masses = {SOUZU_5: 9, RED_DRAGON: 9, _tile(TileCategory.MANZU, 2): 1}
        with self._with_masses(masses):
            selected, analysis = _evaluate_and_choose_discard(
                self.policy_input, self.discard_actions
            )

        self.assertNotEqual(selected, self.discard_2m)
        self.assertEqual(
            [
                evaluation.action
                for evaluation in analysis.two_step_tiebreak_analysis.candidate_evaluations
            ],
            [self.discard_5s, self.discard_7z],
        )

    def test_all_zero_passes_every_candidate_to_the_two_step_ranking(self) -> None:
        masses = {SOUZU_5: 0, RED_DRAGON: 0, _tile(TileCategory.MANZU, 2): 0}

        with (
            self._with_masses(masses),
            patch.object(
                finite_horizon,
                "_two_step_evaluate_and_choose_discard",
                wraps=finite_horizon._two_step_evaluate_and_choose_discard,
            ) as two_step,
        ):
            selected, analysis = _evaluate_and_choose_discard(
                self.policy_input, self.discard_actions
            )

        self.assertEqual(two_step.call_count, 1)
        self.assertEqual(set(two_step.call_args.args[1]), set(self.discard_actions))
        self.assertEqual(
            selected,
            TwoStepUkeirePolicy().choose_action(
                _decision(_TENPAI_HAND, self.discard_actions)
            ),
        )
        self.assertIsInstance(
            analysis.two_step_tiebreak_analysis, TwoStepUkeireAnalysis
        )

    def test_single_candidate_with_zero_mass_uses_the_all_zero_fallback(self) -> None:
        masses = {RED_DRAGON: 0}

        with (
            self._with_masses(masses),
            patch.object(
                finite_horizon,
                "_two_step_evaluate_and_choose_discard",
                wraps=finite_horizon._two_step_evaluate_and_choose_discard,
            ) as two_step,
        ):
            selected, analysis = _evaluate_and_choose_discard(
                self.policy_input, (self.discard_7z,)
            )

        self.assertIs(selected, self.discard_7z)
        self.assertEqual(two_step.call_count, 1)
        self.assertEqual(two_step.call_args.args[1], (self.discard_7z,))
        self.assertIsNotNone(analysis.two_step_tiebreak_analysis)

    def test_single_candidate_with_positive_mass_skips_the_fallback(self) -> None:
        masses = {RED_DRAGON: 3}

        with (
            self._with_masses(masses),
            patch.object(
                finite_horizon,
                "_two_step_evaluate_and_choose_discard",
                wraps=finite_horizon._two_step_evaluate_and_choose_discard,
            ) as two_step,
        ):
            selected, analysis = _evaluate_and_choose_discard(
                self.policy_input, (self.discard_7z,)
            )

        self.assertIs(selected, self.discard_7z)
        self.assertEqual(two_step.call_count, 0)
        self.assertIsNone(analysis.two_step_tiebreak_analysis)

    def test_shanten_is_not_a_hard_filter_above_completion_mass(self) -> None:
        # 打牌後向聴数が最小でない候補でも、completion massが最大なら選ぶ。
        two_step_selected = TwoStepUkeirePolicy().choose_action(
            _decision(_TENPAI_HAND, self.discard_actions)
        )
        masses = {SOUZU_5: 1, RED_DRAGON: 1, _tile(TileCategory.MANZU, 2): 2}

        with self._with_masses(masses):
            selected, _ = _evaluate_and_choose_discard(
                self.policy_input, self.discard_actions
            )

        self.assertIs(selected, self.discard_2m)
        self.assertNotEqual(selected, two_step_selected)


class DecisionResultTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = FiniteHorizonCompletionPolicy()
        self.discard_actions = tuple(
            _discard(tile) for tile in dict.fromkeys(_OPEN_HAND_CONCEALED)
        )

    def test_horizon_three_decision_produces_a_consistent_analysis(self) -> None:
        decision = _open_hand_decision(self.discard_actions)

        proposed = self.policy.choose_action_with_analysis(decision)

        self.assertIsInstance(proposed, PolicyDecision)
        self.assertIn(proposed.action, self.discard_actions)
        analysis = proposed.analysis
        self.assertIsInstance(analysis, FiniteHorizonCompletionAnalysis)
        self.assertEqual(analysis.horizon, DEFAULT_HORIZON)
        self.assertEqual(
            analysis.hidden_tile_count,
            sum(_root_remaining_counts(decision.input)),
        )
        self.assertEqual(
            analysis.sequence_denominator,
            _falling_factorial(analysis.hidden_tile_count, DEFAULT_HORIZON),
        )
        self.assertEqual(len(analysis.candidate_evaluations), len(self.discard_actions))
        for evaluation in analysis.candidate_evaluations:
            self.assertGreaterEqual(evaluation.completion_mass, 0)
            self.assertLessEqual(
                evaluation.completion_mass, analysis.sequence_denominator
            )

    def test_horizon_three_masses_dominate_the_shorter_horizons(self) -> None:
        # P(k回以内に完成) は k について単調非減少なので、exact integer massは
        # `M_k >= M_(k-1) * (N - k + 1)` を満たす。
        policy_input = _open_hand_input()
        remaining = _root_remaining_counts(policy_input)
        hidden = sum(remaining)
        evaluator = _FiniteHorizonEvaluator()
        post_discard = _tile_type_counts(_hand("345m9s"))

        masses = [
            evaluator.completion_mass(post_discard, remaining, horizon)
            for horizon in (1, 2, 3)
        ]

        self.assertGreater(masses[0], 0)
        for horizon in (2, 3):
            self.assertGreaterEqual(
                masses[horizon - 1], masses[horizon - 2] * (hidden - horizon + 1)
            )

    def test_decision_is_independent_of_the_legal_action_order(self) -> None:
        selections = set()
        for permutation in itertools.permutations(self.discard_actions):
            selections.add(self.policy.choose_action(_open_hand_decision(permutation)))

        self.assertEqual(len(selections), 1)

    def test_red_five_and_normal_five_actions_share_the_completion_mass(self) -> None:
        # 同じstructural root stateへ落ちる別々のactual action identityは、
        # 同じcompletion massを持ち、共有cacheで再計算されない。
        policy_input = _make_input(_RED_FIVE_CONCEALED, players=_open_hand_players())

        _, analysis = _evaluate_and_choose_discard(
            policy_input,
            (_discard(MANZU_5), _discard(MANZU_5_RED)),
            horizon=2,
        )

        masses = {
            evaluation.action.tile.is_red: evaluation.completion_mass
            for evaluation in analysis.candidate_evaluations
        }
        self.assertEqual(len(analysis.candidate_evaluations), 2)
        self.assertEqual(masses[True], masses[False])

    def test_same_structural_root_state_reuses_the_shared_cache(self) -> None:
        policy_input = _make_input(_RED_FIVE_CONCEALED, players=_open_hand_players())
        remaining = _root_remaining_counts(policy_input)
        evaluator = _FiniteHorizonEvaluator()
        post_discard = _tile_type_counts(_hand("34m") + (MANZU_5, SOUZU_9))

        first = evaluator.completion_mass(post_discard, remaining, 2)
        misses_before = evaluator.cache_misses
        second = evaluator.completion_mass(post_discard, remaining, 2)

        self.assertEqual(first, second)
        self.assertEqual(evaluator.cache_misses, misses_before)

    def test_tsumogiri_identity_is_preserved_for_root_actions(self) -> None:
        tedashi = _discard(SOUZU_9)
        tsumogiri = _discard(SOUZU_9, tsumogiri=True)

        _, analysis = _evaluate_and_choose_discard(
            _open_hand_input(), (tedashi, tsumogiri), horizon=2
        )

        actions = [evaluation.action for evaluation in analysis.candidate_evaluations]
        self.assertEqual(actions, [tedashi, tsumogiri])
        self.assertEqual(
            analysis.candidate_evaluations[0].completion_mass,
            analysis.candidate_evaluations[1].completion_mass,
        )

    def test_decision_is_deterministic_for_the_same_input(self) -> None:
        decision = _open_hand_decision(self.discard_actions)

        first = self.policy.choose_action(decision)
        second = self.policy.choose_action(decision)

        self.assertIs(first, second)

    def test_fail_closed_when_the_discard_is_not_in_the_concealed_hand(self) -> None:
        with self.assertRaises(TwoStepUkeirePolicyError):
            self.policy.choose_action(_decision((), (_discard(SOUZU_9),)))

    def test_fail_closed_when_hidden_tiles_cannot_fill_the_horizon(self) -> None:
        policy_input = _open_hand_input()

        with patch.object(
            finite_horizon, "_root_remaining_counts", return_value=_counts(s9=2)
        ):
            with self.assertRaises(FiniteHorizonCompletionPolicyError):
                _evaluate_and_choose_discard(policy_input, (_discard(SOUZU_9),))


class InheritedOrchestrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = FiniteHorizonCompletionPolicy()

    def _traced(self, decision: DecisionContext):
        recorder = DecisionTraceRecorder()
        selected = execute_policy_with_trace(self.policy, decision, recorder)
        (trace,) = recorder.snapshot()
        return selected, trace

    def test_policy_extends_the_two_step_orchestration(self) -> None:
        self.assertTrue(issubclass(FiniteHorizonCompletionPolicy, TwoStepUkeirePolicy))
        self.assertIs(
            FiniteHorizonCompletionPolicy._decide, TwoStepUkeirePolicy._decide
        )
        self.assertIs(
            FiniteHorizonCompletionPolicy.choose_action,
            TwoStepUkeirePolicy.choose_action,
        )
        self.assertIs(
            FiniteHorizonCompletionPolicy.choose_action_with_analysis,
            TwoStepUkeirePolicy.choose_action_with_analysis,
        )

    def test_winning_action_outranks_the_discard_evaluation(self) -> None:
        ron = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_3)
        decision = _open_hand_decision((_discard(SOUZU_9), ron))

        with patch.object(
            finite_horizon,
            "_evaluate_and_choose_discard",
            wraps=finite_horizon._evaluate_and_choose_discard,
        ) as evaluate:
            selected, trace = self._traced(decision)

        self.assertIs(selected, ron)
        self.assertEqual(evaluate.call_count, 0)
        self.assertIsNone(trace.analysis)

    def test_tsumo_and_riichi_branches_report_no_finite_horizon_analysis(self) -> None:
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_3)
        riichi = RiichiAction(actor=Seat.SEAT_0)

        for actions, expected in (
            ((_discard(SOUZU_9), tsumo), tsumo),
            ((_discard(SOUZU_9), riichi), riichi),
        ):
            with self.subTest(expected=expected):
                with patch.object(
                    finite_horizon,
                    "_evaluate_and_choose_discard",
                    wraps=finite_horizon._evaluate_and_choose_discard,
                ) as evaluate:
                    selected, trace = self._traced(_open_hand_decision(actions))

                self.assertIs(selected, expected)
                self.assertEqual(evaluate.call_count, 0)
                self.assertIsNone(trace.analysis)

    def test_pass_fallback_reports_no_finite_horizon_analysis(self) -> None:
        pass_action = PassAction(actor=Seat.SEAT_0)

        with patch.object(
            finite_horizon,
            "_evaluate_and_choose_discard",
            wraps=finite_horizon._evaluate_and_choose_discard,
        ) as evaluate:
            selected, trace = self._traced(_decision((), (pass_action,)))

        self.assertIs(selected, pass_action)
        self.assertEqual(evaluate.call_count, 0)
        self.assertIsNone(trace.analysis)

    def test_discard_evaluation_runs_exactly_once_for_a_traced_decision(self) -> None:
        decision = _open_hand_decision(
            tuple(_discard(tile) for tile in dict.fromkeys(_OPEN_HAND_CONCEALED))
        )

        with patch.object(
            finite_horizon,
            "_evaluate_and_choose_discard",
            wraps=finite_horizon._evaluate_and_choose_discard,
        ) as evaluate:
            _, trace = self._traced(decision)

        self.assertEqual(evaluate.call_count, 1)
        self.assertIsInstance(trace.analysis, FiniteHorizonCompletionAnalysis)

    def test_untraced_execution_matches_traced_execution(self) -> None:
        decision = _open_hand_decision(
            tuple(_discard(tile) for tile in dict.fromkeys(_OPEN_HAND_CONCEALED))
        )

        untraced = execute_policy(self.policy, decision)
        traced = execute_policy_with_trace(
            self.policy, decision, DecisionTraceRecorder()
        )

        self.assertIs(untraced, traced)


class AnalysisValueTest(unittest.TestCase):
    def _evaluation(self, mass: int = 3) -> FiniteHorizonCandidateEvaluation:
        return FiniteHorizonCandidateEvaluation(
            action=_discard(SOUZU_9), completion_mass=mass
        )

    def _analysis(self, **overrides: object) -> FiniteHorizonCompletionAnalysis:
        arguments: dict[str, object] = {
            "horizon": 3,
            "hidden_tile_count": 122,
            "sequence_denominator": 122 * 121 * 120,
            "candidate_evaluations": (self._evaluation(),),
            "two_step_tiebreak_analysis": None,
        }
        arguments.update(overrides)
        return FiniteHorizonCompletionAnalysis(**arguments)

    def test_candidate_evaluation_is_an_immutable_typed_value(self) -> None:
        action = _discard(SOUZU_9)
        evaluation = FiniteHorizonCandidateEvaluation(action, 7)

        self.assertIs(evaluation.action, action)
        self.assertEqual(
            tuple(field.name for field in fields(evaluation)),
            ("action", "completion_mass"),
        )
        self.assertFalse(hasattr(evaluation, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            evaluation.completion_mass = 1

    def test_candidate_evaluation_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(TypeError, "action"):
            FiniteHorizonCandidateEvaluation(object(), 1)
        with self.assertRaisesRegex(TypeError, "completion_mass"):
            FiniteHorizonCandidateEvaluation(_discard(SOUZU_9), True)
        with self.assertRaisesRegex(ValueError, "negative"):
            FiniteHorizonCandidateEvaluation(_discard(SOUZU_9), -1)

    def test_analysis_is_an_immutable_typed_payload(self) -> None:
        analysis = self._analysis()

        self.assertIsInstance(analysis, AnalysisTrace)
        self.assertEqual(
            tuple(field.name for field in fields(analysis)),
            (
                "horizon",
                "hidden_tile_count",
                "sequence_denominator",
                "candidate_evaluations",
                "two_step_tiebreak_analysis",
            ),
        )
        self.assertFalse(hasattr(analysis, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            analysis.horizon = 1

    def test_analysis_normalizes_to_a_detached_tuple(self) -> None:
        evaluations = [self._evaluation()]

        analysis = self._analysis(candidate_evaluations=evaluations)
        evaluations.clear()

        self.assertIsInstance(analysis.candidate_evaluations, tuple)
        self.assertEqual(len(analysis.candidate_evaluations), 1)

    def test_analysis_validates_the_semantic_invariants(self) -> None:
        with self.assertRaisesRegex(ValueError, "horizon must be positive"):
            self._analysis(horizon=0)
        with self.assertRaisesRegex(ValueError, "hidden_tile_count"):
            self._analysis(horizon=3, hidden_tile_count=2)
        with self.assertRaisesRegex(ValueError, "sequence_denominator"):
            self._analysis(sequence_denominator=0)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            self._analysis(
                sequence_denominator=2, candidate_evaluations=(self._evaluation(3),)
            )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self._analysis(candidate_evaluations=())

    def test_analysis_rejects_free_form_and_wrongly_typed_payloads(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be an iterable"):
            self._analysis(candidate_evaluations=7)
        with self.assertRaisesRegex(TypeError, "FiniteHorizonCandidateEvaluation"):
            self._analysis(candidate_evaluations=({"mass": 4},))
        with self.assertRaisesRegex(TypeError, "horizon must be an int"):
            self._analysis(horizon=True)
        with self.assertRaisesRegex(TypeError, "two_step_tiebreak_analysis"):
            self._analysis(two_step_tiebreak_analysis="two-step")

    def test_analysis_keeps_no_float_probability(self) -> None:
        analysis = self._analysis()

        self.assertTrue(
            all(
                type(getattr(analysis, field.name)) is not float
                for field in fields(analysis)
            )
        )
        self.assertTrue(
            all(
                type(evaluation.completion_mass) is int
                for evaluation in analysis.candidate_evaluations
            )
        )


def _corpus_hand_counts(size: int, seed_index: int) -> tuple[int, ...]:
    """乱数を使わない決定的な規則だけで作る、再現可能なcorpus hand。

    `seed_index`から固定stepで牌種を巡回し、1牌種`_MAX_COPIES_PER_TILE_TYPE`枚
    を上限に`size`枚まで積む。同じ引数からは常に同じ牌姿になるので、CIでも
    結果が揺れない。
    """
    counts = [0] * TILE_TYPE_COUNT
    index = seed_index % TILE_TYPE_COUNT
    step = 1 + seed_index % 7
    total = 0
    while total < size:
        if counts[index] < _MAX_COPIES_PER_TILE_TYPE:
            counts[index] += 1
            total += 1
        index = (index + step) % TILE_TYPE_COUNT
    return tuple(counts)


def _pair_heavy_corpus_counts(size: int, seed_index: int) -> tuple[int, ...]:
    """対子中心の決定的corpus hand。七対子解釈が効きやすい牌姿を作る。"""
    counts = [0] * TILE_TYPE_COUNT
    index = seed_index % TILE_TYPE_COUNT
    total = 0
    while total < size:
        take = min(2, size - total, _MAX_COPIES_PER_TILE_TYPE - counts[index])
        counts[index] += take
        total += take
        index = (index + 1 + seed_index % 3) % TILE_TYPE_COUNT
    return tuple(counts)


def _orphan_heavy_corpus_counts(size: int, seed_index: int) -> tuple[int, ...]:
    """么九牌中心の決定的corpus hand。国士解釈が効きやすい牌姿を作る。"""
    orphans = (
        _index(TileCategory.MANZU, 1),
        _index(TileCategory.MANZU, 9),
        _index(TileCategory.PINZU, 1),
        _index(TileCategory.PINZU, 9),
        _index(TileCategory.SOUZU, 1),
        _index(TileCategory.SOUZU, 9),
        *(_index(TileCategory.HONOR, rank) for rank in range(1, 8)),
    )
    counts = [0] * TILE_TYPE_COUNT
    position = seed_index % len(orphans)
    total = 0
    while total < size:
        index = orphans[position % len(orphans)]
        if counts[index] < _MAX_COPIES_PER_TILE_TYPE:
            counts[index] += 1
            total += 1
        position += 1
    return tuple(counts)


def _sampled_hand_counts(generator: random.Random, size: int) -> tuple[int, ...]:
    """固定seedのgeneratorから引く補助corpus（CI上はdeterministic）。"""
    counts = [0] * TILE_TYPE_COUNT
    total = 0
    while total < size:
        index = generator.randrange(TILE_TYPE_COUNT)
        if counts[index] == _MAX_COPIES_PER_TILE_TYPE:
            continue
        counts[index] += 1
        total += 1
    return tuple(counts)


_DRAW_HAND_SIZES = (14, 11, 8, 5, 2)
"""同じfixed-meld contextで`size -> size - 1`のdeletionが定義できる牌数。"""


class ShantenDeletionMonotonicityTest(unittest.TestCase):
    """exact-safe parent pruningの前提`shanten(D - d) >= shanten(D)`を固定する。

    この性質が崩れると`draw_shanten > depth - 2`のparent pruningがexactで
    なくなるため、`FiniteHorizonCompletionPolicy`側のimplementation
    regression testとしてここで固定する。`calculate_shanten()`のpublic
    contract自体は変更しない。
    """

    def _assert_deletion_monotone(self, hand_counts: tuple[int, ...]) -> None:
        draw_shanten = calculate_shanten(_tiles_for_oracle(hand_counts))
        for index, count in enumerate(hand_counts):
            if count == 0:
                continue
            reduced = list(hand_counts)
            reduced[index] -= 1
            reduced_shanten = calculate_shanten(_tiles_for_oracle(tuple(reduced)))
            self.assertGreaterEqual(
                reduced_shanten,
                draw_shanten,
                msg=(
                    f"deleting tile index {index} from {hand_counts} lowered "
                    f"shanten from {draw_shanten} to {reduced_shanten}"
                ),
            )

    def test_closed_standard_hand_is_deletion_monotone(self) -> None:
        self._assert_deletion_monotone(_tile_type_counts(_TENPAI_HAND))

    def test_chiitoitsu_relevant_hand_is_deletion_monotone(self) -> None:
        self._assert_deletion_monotone(_tile_type_counts(_CHIITOITSU_HAND))

    def test_kokushi_relevant_hand_is_deletion_monotone(self) -> None:
        self._assert_deletion_monotone(_tile_type_counts(_KOKUSHI_HAND))

    def test_every_fixed_meld_context_is_deletion_monotone(self) -> None:
        # 14 -> 13 / 11 -> 10 / 8 -> 7 / 5 -> 4 / 2 -> 1 はいずれも同じ確定
        # 面子数を共有するので、同一fixed-meld context内のdeletionになる。
        fixtures = {
            0: _TENPAI_HAND,
            1: _hand("13579m13p246s7z"),
            2: _hand("1357m13p24s"),
            3: _hand("13m57p7z"),
            4: _hand("1m5p"),
        }
        for fixed_melds, tiles in fixtures.items():
            with self.subTest(fixed_melds=fixed_melds, size=len(tiles)):
                self.assertEqual(4 - (len(tiles) - 1) // 3, fixed_melds)
                # 和了形(-1)は下限なのでdeletion testのteethが弱くなる。
                # 未完成fixtureであることを明示的に固定する。
                self.assertGreaterEqual(calculate_shanten(tiles), 0)
                self._assert_deletion_monotone(_tile_type_counts(tiles))

    def test_deterministic_corpus_is_deletion_monotone(self) -> None:
        for size in _DRAW_HAND_SIZES:
            for seed_index in range(TILE_TYPE_COUNT):
                self._assert_deletion_monotone(_corpus_hand_counts(size, seed_index))

    def test_pair_heavy_corpus_is_deletion_monotone(self) -> None:
        for size in _DRAW_HAND_SIZES:
            for seed_index in range(TILE_TYPE_COUNT):
                self._assert_deletion_monotone(
                    _pair_heavy_corpus_counts(size, seed_index)
                )

    def test_orphan_heavy_corpus_is_deletion_monotone(self) -> None:
        for size in _DRAW_HAND_SIZES:
            for seed_index in range(13):
                self._assert_deletion_monotone(
                    _orphan_heavy_corpus_counts(size, seed_index)
                )

    def test_seeded_sample_corpus_is_deletion_monotone(self) -> None:
        # random samplingはあくまで補助だが、固定seedなのでCI上は決定的である。
        generator = random.Random(20260824)
        for size in _DRAW_HAND_SIZES:
            for _ in range(12):
                self._assert_deletion_monotone(_sampled_hand_counts(generator, size))


class ParentPruningBehaviorTest(unittest.TestCase):
    """`draw_shanten > depth - 2`のexact-safe parent pruningを直接固定する。"""

    def _evaluate_with_children(
        self,
        hand_counts: tuple[int, ...],
        remaining_counts: tuple[int, ...],
        depth: int,
    ) -> tuple[int, list[tuple[tuple[int, ...], tuple[int, ...], int]]]:
        evaluator = _FiniteHorizonEvaluator()
        with _RecordedEvaluation() as recorded:
            mass = evaluator.completion_mass(hand_counts, remaining_counts, depth)
        children = [call for call in recorded.calls if call[2] == depth - 1]
        return mass, children

    def _draw_shanten(self, hand_counts: tuple[int, ...], drawn_index: int) -> int:
        draw_counts = list(hand_counts)
        draw_counts[drawn_index] += 1
        return calculate_shanten(_tiles_for_oracle(tuple(draw_counts)))

    def test_draw_shanten_above_the_bound_generates_no_child_state(self) -> None:
        # depth 3、draw後2向聴。2 > 3 - 2 なので、どの牌を切っても残り2draw
        # では完成できず、hypothetical discard childrenを生成しない。
        green_dragon = _index(TileCategory.HONOR, 6)
        self.assertEqual(self._draw_shanten(_ISOLATED_OPEN_HAND, green_dragon), 2)

        mass, children = self._evaluate_with_children(
            _ISOLATED_OPEN_HAND, _counts(z6=1), 3
        )

        self.assertEqual(children, [])
        self.assertEqual(mass, 0)

    def test_draw_shanten_at_the_bound_still_generates_children(self) -> None:
        # depth 3、draw後1向聴。1 > 1 はfalseなのでpruneしてはならない。
        green_dragon = _index(TileCategory.HONOR, 6)
        self.assertEqual(self._draw_shanten(_ONE_SHANTEN_OPEN_HAND, green_dragon), 1)

        _mass, children = self._evaluate_with_children(
            _ONE_SHANTEN_OPEN_HAND, _counts(z6=1), 3
        )

        self.assertNotEqual(children, [])

    def test_draw_shanten_below_the_bound_generates_children(self) -> None:
        # depth 3、draw後聴牌。0 < 1 なので当然pruneしない。
        green_dragon = _index(TileCategory.HONOR, 6)
        self.assertEqual(self._draw_shanten(_TENPAI_OPEN_HAND, green_dragon), 0)

        _mass, children = self._evaluate_with_children(
            _TENPAI_OPEN_HAND, _counts(z6=1), 3
        )

        self.assertNotEqual(children, [])

    def test_depth_two_boundary_uses_the_same_strict_comparison(self) -> None:
        # depth 2ではboundが0になる。聴牌(0 > 0 はfalse)はpruneせず、
        # 1向聴(1 > 0)はpruneする。
        drawn = _counts(z6=1)

        _tenpai_mass, tenpai_children = self._evaluate_with_children(
            _TENPAI_OPEN_HAND, drawn, 2
        )
        one_shanten_mass, one_shanten_children = self._evaluate_with_children(
            _ONE_SHANTEN_OPEN_HAND, drawn, 2
        )

        self.assertNotEqual(tenpai_children, [])
        self.assertEqual(one_shanten_children, [])
        self.assertEqual(one_shanten_mass, 0)

    def test_completed_draw_keeps_the_suffix_success_mass(self) -> None:
        # completion判定はpruning条件より先に行い、既存のsuffix mass計算を
        # そのまま維持する。
        remaining = _counts(s9=2, z1=1)

        mass, children = self._evaluate_with_children(_TENPAI_OPEN_HAND, remaining, 2)

        self.assertEqual(mass, _falling_factorial(3, 2))
        # 9s draw(完成)はchildを作らず、1z draw(聴牌維持)だけがchildを作る。
        self.assertNotEqual(children, [])
        for _child_hand, child_remaining, _depth in children:
            self.assertEqual(child_remaining, _counts(s9=2))

    def test_pruning_adds_no_extra_shanten_evaluation(self) -> None:
        # parent pruningはcompletion判定で評価済みのdraw_shantenを再利用する。
        # 1 draw branchあたりのshanten評価はdraw_hand 1件だけである。
        evaluator = _FiniteHorizonEvaluator()

        evaluator.completion_mass(_ISOLATED_OPEN_HAND, _counts(z6=1), 3)

        # 評価されるのはparent hand自身とdraw_handの2件だけ。
        self.assertEqual(evaluator.shanten_evaluations, 2)

    def test_pruned_branches_match_the_unpruned_oracle(self) -> None:
        # pruneされたbranchのcontributionが本当に0であることを、pruningを
        # 一切持たないtest-local oracleとの一致で確認する。
        for hand, remaining, depth in (
            (_ISOLATED_OPEN_HAND, _counts(z6=1, m1=3), 3),
            (_ONE_SHANTEN_OPEN_HAND, _counts(z6=1, m2=4, m5=4), 3),
            (_ONE_SHANTEN_OPEN_HAND, _counts(z6=2, m2=1), 2),
        ):
            with self.subTest(hand=hand, depth=depth):
                evaluator = _FiniteHorizonEvaluator()
                self.assertEqual(
                    evaluator.completion_mass(hand, remaining, depth),
                    _uncached_oracle_mass(hand, remaining, depth),
                )


class ExactnessAgainstUnprunedOracleTest(unittest.TestCase):
    """optimized production DPがunpruned exact referenceと一致することを固定する。"""

    _INVENTORIES = (
        _counts(s9=2, z1=1),
        _counts(s9=1, z1=2),
        _counts(m3=3, m6=4, s9=2, z1=2),
        _counts(m2=2, m5=2, s9=1, z1=1, z6=1),
    )

    def test_horizon_one_and_two_match_the_unpruned_oracle(self) -> None:
        for hand in (_TENPAI_OPEN_HAND, _ONE_SHANTEN_OPEN_HAND, _ISOLATED_OPEN_HAND):
            for remaining in self._INVENTORIES:
                for depth in (1, 2):
                    with self.subTest(hand=hand, remaining=remaining, depth=depth):
                        evaluator = _FiniteHorizonEvaluator()
                        self.assertEqual(
                            evaluator.completion_mass(hand, remaining, depth),
                            _uncached_oracle_mass(hand, remaining, depth),
                        )

    def test_horizon_three_matches_the_unpruned_oracle(self) -> None:
        for hand in (_TENPAI_OPEN_HAND, _ONE_SHANTEN_OPEN_HAND, _ISOLATED_OPEN_HAND):
            for remaining in (
                _counts(s9=2, z1=1),
                _counts(m2=2, m5=1, s9=1, z6=1),
            ):
                with self.subTest(hand=hand, remaining=remaining):
                    evaluator = _FiniteHorizonEvaluator()
                    self.assertEqual(
                        evaluator.completion_mass(hand, remaining, 3),
                        _uncached_oracle_mass(hand, remaining, 3),
                    )

    def test_every_root_candidate_mass_matches_the_unpruned_oracle(self) -> None:
        # selected actionだけでなく、各root candidateのcompletion massそのものを
        # unpruned oracleと突き合わせる。
        policy_input = _open_hand_input()
        remaining = _root_remaining_counts(policy_input)
        discard_actions = tuple(
            _discard(tile) for tile in dict.fromkeys(_OPEN_HAND_CONCEALED)
        )
        concealed = policy_input.own_hand.concealed_tiles

        for horizon in (1, 2):
            with self.subTest(horizon=horizon):
                _selected, analysis = _evaluate_and_choose_discard(
                    policy_input, discard_actions, horizon=horizon
                )
                self.assertNotEqual(analysis.candidate_evaluations, ())
                for evaluation in analysis.candidate_evaluations:
                    post_discard = _tile_type_counts(
                        _remove_one_matching_tile(concealed, evaluation.action.tile)
                    )
                    self.assertEqual(
                        evaluation.completion_mass,
                        _uncached_oracle_mass(post_discard, remaining, horizon),
                    )


class PolicyGenerationAndScopeTest(unittest.TestCase):
    def test_all_seven_policy_generations_are_public(self) -> None:
        import lisjong.policies as policies

        self.assertEqual(
            set(policies.__all__),
            {
                "FiniteHorizonCompletionPolicy",
                "GenbutsuDefenseTwoStepUkeirePolicy",
                "MinimalPolicy",
                "ShantenPolicy",
                "TwoStepUkeirePolicy",
                "UkeirePolicy",
                "ValueAwareTwoStepUkeirePolicy",
            },
        )

    def test_policy_is_importable_from_the_package(self) -> None:
        from lisjong.policies import FiniteHorizonCompletionPolicy as imported

        self.assertIs(imported, finite_horizon.FiniteHorizonCompletionPolicy)

    def test_policy_is_defined_at_module_level(self) -> None:
        self.assertEqual(
            FiniteHorizonCompletionPolicy.__module__,
            "lisjong.policies.finite_horizon_completion",
        )

    def test_policy_class_is_picklable_for_spawn(self) -> None:
        # Windows `spawn` + `ProcessPoolExecutor`同様、classをpickleで
        # 再構築できることを確認する。
        roundtrip = pickle.loads(pickle.dumps(FiniteHorizonCompletionPolicy))

        self.assertIs(roundtrip, FiniteHorizonCompletionPolicy)

    def test_module_reuses_the_shared_inventory_and_shanten_contracts(self) -> None:
        tree = ast.parse(inspect.getsource(finite_horizon))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }

        self.assertIn("lisjong.belief.tile_conservation", imported_modules)
        self.assertIn("lisjong.belief.canonical_axes", imported_modules)
        # Issue #113: shanten評価はhand_evaluationが所有するcount-native
        # contract経由で行い、private backendへは触れない。
        self.assertIn("lisjong.hand_evaluation.shanten", imported_modules)
        self.assertNotIn("lisjong.hand_evaluation._python_shanten", imported_modules)
        self.assertNotIn(
            "lisjong.belief.conditional_uniform_hand_belief", imported_modules
        )

    def test_module_never_references_the_private_shanten_backend(self) -> None:
        # importだけでなく、name / attribute参照としても`_python_shanten`へ
        # 触れないことを固定する（backend exchangeabilityはhand_evaluation層の
        # 責務）。docstringでの言及は境界の説明なので対象外とし、実際のcode
        # referenceだけをASTで見る。
        tree = ast.parse(inspect.getsource(finite_horizon))
        referenced: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                referenced.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    referenced.add(node.module)
                referenced.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)

        self.assertFalse(
            any("_python_shanten" in name for name in referenced),
            msg="the Policy must not reference the private shanten backend",
        )

    def test_shanten_hot_path_does_not_rebuild_tiles_from_counts(self) -> None:
        # counts -> Tile -> counts のround-tripがproduction hot pathから
        # 消えていることを固定する（Issue #113）。
        source = inspect.getsource(finite_horizon._FiniteHorizonEvaluator.shanten)

        self.assertIn("calculate_shanten_from_canonical_counts(hand_counts)", source)
        self.assertNotIn("_tiles_from_counts", source)
        self.assertFalse(hasattr(finite_horizon, "_tiles_from_counts"))
        self.assertFalse(hasattr(finite_horizon, "_CANONICAL_TILES"))

    def test_module_has_no_hidden_or_environment_dependency(self) -> None:
        tree = ast.parse(inspect.getsource(finite_horizon))
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
                    "mahjong",
                    "riichienv",
                    "websockets",
                )
            )
        )

    def test_policy_has_an_independent_error_boundary(self) -> None:
        self.assertFalse(
            issubclass(FiniteHorizonCompletionPolicyError, TwoStepUkeirePolicyError)
        )
        self.assertFalse(
            issubclass(TwoStepUkeirePolicyError, FiniteHorizonCompletionPolicyError)
        )


class TwoStepBaselineRegressionTest(unittest.TestCase):
    """baselineのTwoStep behaviorが#109で変わらないことを確認する。"""

    def test_two_step_second_step_selection_is_unchanged(self) -> None:
        # 既存TwoStep fixtureの2段階受け入れscoreと選択結果を#109で変えない。
        two_step_hand = _hand("345m56679s333577z")
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(_tile(TileCategory.HONOR, 5))

        selected, evaluations = two_step_evaluate_and_choose_discard(
            _make_input(two_step_hand), (discard_9s, discard_white)
        )

        scores = {
            evaluation.action: evaluation.second_step_ukeire_score
            for evaluation in evaluations
        }
        self.assertIs(selected, discard_white)
        self.assertEqual(scores[discard_9s], 122)
        self.assertEqual(scores[discard_white], 126)
        self.assertIs(
            TwoStepUkeirePolicy().choose_action(
                _decision(two_step_hand, (discard_9s, discard_white))
            ),
            discard_white,
        )

    def test_two_step_analysis_type_is_unchanged(self) -> None:
        decision = _decision(_TENPAI_HAND, (_discard(SOUZU_5), _discard(RED_DRAGON)))

        proposed = TwoStepUkeirePolicy().choose_action_with_analysis(decision)

        self.assertIsInstance(proposed.analysis, TwoStepUkeireAnalysis)


if __name__ == "__main__":
    unittest.main()
