"""Issue #58 `TwoStepUkeirePolicy`のunit test。"""

import ast
import inspect
import itertools
import unittest
from unittest.mock import patch

import lisjong.policies.two_step_ukeire as two_step
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies import TwoStepUkeirePolicy, UkeirePolicy
from lisjong.policies.two_step_ukeire import (
    TwoStepUkeirePolicyError,
    _best_next_ukeire,
    _known_counts_after_draw,
    _known_tile_counts,
    _remove_one_matching_tile,
    _second_step_score,
    _ukeire_count,
)
from lisjong.policy_contract.action import (
    DiscardAction,
    KyuushuKyuuhaiAction,
    PassAction,
    RiichiAction,
    RonAction,
    TsumoAction,
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
MANZU_4 = _tile(TileCategory.MANZU, 4)
MANZU_5 = _tile(TileCategory.MANZU, 5)
MANZU_5_RED = _tile(TileCategory.MANZU, 5, red=True)
MANZU_7 = _tile(TileCategory.MANZU, 7)
PINZU_1 = _tile(TileCategory.PINZU, 1)
PINZU_5 = _tile(TileCategory.PINZU, 5)
SOUZU_9 = _tile(TileCategory.SOUZU, 9)
EAST = _tile(TileCategory.HONOR, 1)
SOUTH = _tile(TileCategory.HONOR, 2)
WHITE_DRAGON = _tile(TileCategory.HONOR, 5)
RED_DRAGON = _tile(TileCategory.HONOR, 7)

MANZU_1_TYPE = MANZU_1.tile_type
MANZU_2_TYPE = MANZU_2.tile_type
MANZU_5_TYPE = MANZU_5.tile_type

_TWO_STEP_HAND = _hand("345m56679s333577z")
"""9s/5z切りが同じ1向聴・受け入れ21で、2段目だけが異なる14枚。"""

_NINE_MANZU_HAND = _hand("123456789m111p23p")
"""2m/4m切りが同向聴で、4m切りのcurrent受け入れが多い14枚。"""

_TANKI_VERSUS_ONE_SHANTEN_HAND = _hand("234m567m234p567p5s7z")
"""7z切りで聴牌を維持し、面子を崩す候補は1向聴になる14枚。"""


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
            live_wall_tiles_remaining=70,
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


class PolicyPriorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = TwoStepUkeirePolicy()

    def test_winning_action_has_priority(self) -> None:
        discard = _discard(SOUZU_9)
        ron = RonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            winning_tile=MANZU_1,
        )
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_1)

        for actions in itertools.permutations((discard, ron, tsumo)):
            with self.subTest(actions=actions):
                self.assertEqual(
                    self.policy.choose_action(_decision(_TWO_STEP_HAND, actions)), ron
                )

    def test_winning_action_has_priority_over_riichi(self) -> None:
        # Issue #76: RonAction/TsumoAction still outrank RiichiAction, and
        # the existing winning-action stable tie-break is unaffected by the
        # new Always Riichi baseline.
        riichi = RiichiAction(actor=Seat.SEAT_0)
        ron = RonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            winning_tile=MANZU_1,
        )
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_1)

        for actions in itertools.permutations((riichi, ron, tsumo)):
            with self.subTest(actions=actions):
                self.assertEqual(
                    self.policy.choose_action(_decision(_TWO_STEP_HAND, actions)), ron
                )

    def test_lower_shanten_cannot_be_reversed(self) -> None:
        keep_tenpai = _discard(RED_DRAGON)
        break_meld = _discard(MANZU_4)
        decision = _decision(
            _TANKI_VERSUS_ONE_SHANTEN_HAND,
            (break_meld, keep_tenpai),
        )

        with patch.object(
            two_step,
            "_second_step_score",
            side_effect=AssertionError("second step must not evaluate worse shanten"),
        ):
            self.assertEqual(self.policy.choose_action(decision), keep_tenpai)

    def test_higher_current_ukeire_cannot_be_reversed(self) -> None:
        lower_current_ukeire = _discard(MANZU_2)
        higher_current_ukeire = _discard(MANZU_4)
        decision = _decision(
            _NINE_MANZU_HAND,
            (lower_current_ukeire, higher_current_ukeire),
        )

        with patch.object(
            two_step,
            "_second_step_score",
            side_effect=AssertionError("second step must not evaluate lower ukeire"),
        ):
            self.assertEqual(self.policy.choose_action(decision), higher_current_ukeire)

    def test_two_step_breaks_only_same_shanten_same_current_ukeire_tie(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        policy_input = _make_input(_TWO_STEP_HAND)
        known_counts = _known_tile_counts(policy_input)

        rows = []
        for action in (discard_9s, discard_white):
            hand = _remove_one_matching_tile(_TWO_STEP_HAND, action.tile)
            shanten = calculate_shanten(hand)
            rows.append(
                (
                    action,
                    shanten,
                    _ukeire_count(hand, known_counts, shanten),
                    _second_step_score(hand, known_counts, shanten),
                )
            )

        self.assertEqual(
            [(shanten, ukeire) for _, shanten, ukeire, _ in rows],
            [(1, 21), (1, 21)],
        )
        self.assertEqual([score for _, _, _, score in rows], [122, 126])
        self.assertEqual(
            UkeirePolicy().choose_action(
                _decision(_TWO_STEP_HAND, (discard_white, discard_9s))
            ),
            discard_9s,
        )
        self.assertEqual(
            self.policy.choose_action(
                _decision(_TWO_STEP_HAND, (discard_9s, discard_white))
            ),
            discard_white,
        )

    def test_equal_two_step_score_uses_stable_semantic_tie_break(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)

        with patch.object(two_step, "_second_step_score", return_value=100):
            for actions in itertools.permutations((discard_9s, discard_white)):
                with self.subTest(actions=actions):
                    self.assertEqual(
                        self.policy.choose_action(_decision(_TWO_STEP_HAND, actions)),
                        discard_9s,
                    )

    def test_tenpai_tie_does_not_expand_second_step(self) -> None:
        concealed = (MANZU_1, MANZU_1, MANZU_1, EAST, SOUTH)
        discard_east = _discard(EAST)
        discard_south = _discard(SOUTH)

        with patch.object(
            two_step,
            "_second_step_score",
            side_effect=AssertionError("winning draw must end the lookahead"),
        ):
            for actions in itertools.permutations((discard_east, discard_south)):
                self.assertEqual(
                    self.policy.choose_action(_decision(concealed, actions)),
                    discard_east,
                )


class AlwaysRiichiTest(unittest.TestCase):
    """Issue #76: legalなRiichiActionは通常打牌評価より優先される。"""

    def setUp(self) -> None:
        self.policy = TwoStepUkeirePolicy()

    def test_riichi_is_chosen_over_multiple_discards(self) -> None:
        riichi = RiichiAction(actor=Seat.SEAT_0)
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)

        # 通常打牌評価(2段階受け入れ)は一切呼ばれないはず: 混在させない責務分離の確認。
        with patch.object(
            two_step,
            "_choose_discard",
            side_effect=AssertionError("discard evaluation must not run"),
        ):
            for actions in itertools.permutations((riichi, discard_9s, discard_white)):
                with self.subTest(actions=actions):
                    self.assertEqual(
                        self.policy.choose_action(_decision(_TWO_STEP_HAND, actions)),
                        riichi,
                    )

    def test_riichi_choice_is_order_independent_and_deterministic(self) -> None:
        riichi = RiichiAction(actor=Seat.SEAT_0)
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        pass_action = PassAction(actor=Seat.SEAT_0)

        results = {
            self.policy.choose_action(_decision(_TWO_STEP_HAND, actions))
            for actions in itertools.permutations(
                (riichi, discard_9s, discard_white, pass_action)
            )
        }

        self.assertEqual(results, {riichi})

    def test_selected_riichi_action_is_the_legal_action_instance(self) -> None:
        # Policy must not independently construct a RiichiAction; the chosen
        # value must be identical to the one presented in legal_actions.
        riichi = RiichiAction(actor=Seat.SEAT_0)
        pass_action = PassAction(actor=Seat.SEAT_0)

        chosen = self.policy.choose_action(_decision((), (pass_action, riichi)))

        self.assertIs(chosen, riichi)

    def test_discard_only_decision_is_unaffected_by_riichi_baseline(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)

        self.assertEqual(
            self.policy.choose_action(
                _decision(_TWO_STEP_HAND, (discard_9s, discard_white))
            ),
            discard_white,
        )


class TwoStepScoreTest(unittest.TestCase):
    def test_decision_cache_reuses_red_and_normal_structural_hand(self) -> None:
        evaluator = two_step._DecisionShantenEvaluator()

        with patch.object(
            two_step, "calculate_shanten", wraps=calculate_shanten
        ) as shanten:
            normal_result = evaluator.calculate((MANZU_1, MANZU_5))
            red_result = evaluator.calculate((MANZU_1, MANZU_5_RED))

        self.assertEqual(normal_result, red_result)
        self.assertEqual(shanten.call_count, 1)

    def test_first_draw_remaining_count_is_the_integer_weight(self) -> None:
        known_counts = {MANZU_1_TYPE: 3, MANZU_2_TYPE: 1}

        def next_ukeire(_hand, tile_type, after_draw, _evaluator):
            expected_count = 4 if tile_type == MANZU_1_TYPE else 2
            self.assertEqual(after_draw[tile_type], expected_count)
            return 10 if tile_type == MANZU_1_TYPE else 2

        with (
            patch.object(
                two_step,
                "_effective_tile_types",
                return_value=(MANZU_1_TYPE, MANZU_2_TYPE),
            ),
            patch.object(two_step, "_best_next_ukeire", side_effect=next_ukeire),
        ):
            score = _second_step_score((PINZU_1,), known_counts, current_shanten=2)

        self.assertEqual(score, 1 * 10 + 3 * 2)

    def test_branch_minimizes_shanten_before_comparing_next_ukeire(self) -> None:
        def branch_shanten(hand):
            return 0 if hand[0].tile_type == MANZU_2_TYPE else 1

        with (
            patch.object(two_step, "calculate_shanten", side_effect=branch_shanten),
            patch.object(two_step, "_ukeire_count", return_value=7) as ukeire,
        ):
            result = _best_next_ukeire((MANZU_1,), MANZU_2_TYPE, {MANZU_2_TYPE: 1})

        self.assertEqual(result, 7)
        self.assertEqual(ukeire.call_count, 1)
        self.assertEqual(ukeire.call_args.args[0], [MANZU_2])

    def test_same_branch_shanten_uses_maximum_next_ukeire(self) -> None:
        visited: list[tuple[Tile, ...]] = []

        def branch_ukeire(hand, _known_counts, _shanten, _evaluator):
            snapshot = tuple(hand)
            visited.append(snapshot)
            return 9 if snapshot == (MANZU_1,) else 3

        with (
            patch.object(two_step, "calculate_shanten", return_value=0),
            patch.object(two_step, "_ukeire_count", side_effect=branch_ukeire),
        ):
            result = _best_next_ukeire((MANZU_1,), MANZU_2_TYPE, {MANZU_2_TYPE: 1})

        self.assertEqual(result, 9)
        self.assertEqual(set(visited), {(MANZU_1,), (MANZU_2,)})

    def test_virtual_discard_does_not_decrement_known_count(self) -> None:
        seen_counts: list[int] = []

        def capture_known(_hand, known_counts, _shanten, _evaluator):
            seen_counts.append(known_counts[MANZU_2_TYPE])
            return 1

        with (
            patch.object(two_step, "calculate_shanten", return_value=0),
            patch.object(two_step, "_ukeire_count", side_effect=capture_known),
        ):
            _best_next_ukeire((MANZU_1,), MANZU_2_TYPE, {MANZU_2_TYPE: 2})

        self.assertEqual(seen_counts, [2, 2])


class KnownCountAndZeroRemainingTest(unittest.TestCase):
    def test_known_counts_cover_every_policy_visible_source_once(self) -> None:
        called_discard = Discard(
            tile=PINZU_5,
            tsumogiri=False,
            order=0,
            called_by=Seat.SEAT_2,
        )
        uncalled_discard = Discard(
            tile=MANZU_5,
            tsumogiri=False,
            order=1,
            called_by=None,
        )
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(PINZU_5, PINZU_5, PINZU_5),
            from_seat=Seat.SEAT_1,
            called_tile=PINZU_5,
        )
        players = (
            _player(),
            _player((called_discard, uncalled_discard)),
            _player(melds=(pon,)),
            _player(),
        )
        policy_input = _make_input(
            (MANZU_5,),
            players=players,
            dora_indicators=(SOUZU_9,),
        )

        counts = _known_tile_counts(policy_input)

        self.assertEqual(counts[MANZU_5_TYPE], 2)
        self.assertEqual(counts[PINZU_5.tile_type], 3)
        self.assertEqual(counts[SOUZU_9.tile_type], 1)

    def test_virtual_draw_adds_one_without_mutating_original_counts(self) -> None:
        original = {MANZU_1_TYPE: 2}

        updated = _known_counts_after_draw(original, MANZU_1_TYPE)

        self.assertEqual(original[MANZU_1_TYPE], 2)
        self.assertEqual(updated[MANZU_1_TYPE], 3)
        self.assertIsNot(updated, original)

    def test_first_candidate_discard_does_not_reduce_known_count(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        captured: list[dict[TileType, int]] = []

        def capture(_hand, known_counts, _shanten, _evaluator):
            captured.append(dict(known_counts))
            return 0

        with patch.object(two_step, "_second_step_score", side_effect=capture):
            TwoStepUkeirePolicy().choose_action(
                _decision(_TWO_STEP_HAND, (discard_9s, discard_white))
            )

        original = _known_tile_counts(_make_input(_TWO_STEP_HAND))
        self.assertEqual(captured, [original, original])

    def test_zero_remaining_effective_tile_is_not_expanded(self) -> None:
        with (
            patch.object(
                two_step,
                "_effective_tile_types",
                return_value=(MANZU_1_TYPE,),
            ),
            patch.object(two_step, "_best_next_ukeire") as next_ukeire,
        ):
            score = _second_step_score((PINZU_1,), {MANZU_1_TYPE: 4}, current_shanten=1)

        self.assertEqual(score, 0)
        next_ukeire.assert_not_called()

    def test_no_positive_first_branch_has_zero_score(self) -> None:
        with patch.object(two_step, "_effective_tile_types", return_value=()):
            self.assertEqual(_second_step_score((PINZU_1,), {}, current_shanten=1), 0)

    def test_drawing_a_fifth_visible_copy_fails_closed(self) -> None:
        with self.assertRaises(TwoStepUkeirePolicyError):
            _known_counts_after_draw({MANZU_1_TYPE: 4}, MANZU_1_TYPE)


class HandSizeAndRedFiveTest(unittest.TestCase):
    def test_closed_and_open_hands_use_n_to_n_plus_one_to_n(self) -> None:
        cases = (
            _hand("123456789m1p24s7z"),
            _hand("19m19p19s1234z"),
            _hand("1111p666z"),
            _hand("1111m"),
        )
        for hand in cases:
            known_counts = two_step._count_tile_types(hand)
            with (
                self.subTest(size=len(hand)),
                patch.object(
                    two_step, "calculate_shanten", wraps=calculate_shanten
                ) as shanten,
            ):
                score = _second_step_score(hand, known_counts)

                observed_sizes = {len(call.args[0]) for call in shanten.call_args_list}
                self.assertIsInstance(score, int)
                self.assertIn(len(hand), observed_sizes)
                self.assertIn(len(hand) + 1, observed_sizes)

    def test_red_and_normal_five_have_the_same_structural_score(self) -> None:
        normal = _hand("345m5667s333577z")
        red = _hand("340m5667s333577z")
        normal_known = two_step._count_tile_types(normal)
        red_known = two_step._count_tile_types(red)

        self.assertEqual(normal_known, red_known)
        self.assertEqual(
            _second_step_score(normal, normal_known),
            _second_step_score(red, red_known),
        )

    def test_actual_first_discard_keeps_red_identity(self) -> None:
        concealed = (MANZU_5, MANZU_5, MANZU_5, MANZU_5_RED, MANZU_2)
        discard_normal = _discard(MANZU_5)
        discard_red = _discard(MANZU_5_RED)

        for actions in itertools.permutations((discard_normal, discard_red)):
            with self.subTest(actions=actions):
                self.assertEqual(
                    TwoStepUkeirePolicy().choose_action(_decision(concealed, actions)),
                    discard_normal,
                )


class DeterminismAndErrorBoundaryTest(unittest.TestCase):
    def test_legal_action_order_does_not_change_choice(self) -> None:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        results = {
            TwoStepUkeirePolicy().choose_action(_decision(_TWO_STEP_HAND, actions))
            for actions in itertools.permutations((discard_9s, discard_white))
        }

        self.assertEqual(results, {discard_white})

    def test_known_count_above_four_fails_closed(self) -> None:
        discards = tuple(
            Discard(tile=MANZU_5, tsumogiri=False, order=index, called_by=None)
            for index in range(3)
        )
        players = (_player(), _player(discards), _player(), _player())
        concealed = (MANZU_5, MANZU_5, MANZU_2)

        with self.assertRaises(TwoStepUkeirePolicyError):
            TwoStepUkeirePolicy().choose_action(
                _decision(concealed, (_discard(MANZU_2),), players=players)
            )

    def test_missing_discard_tile_fails_closed(self) -> None:
        with self.assertRaises(TwoStepUkeirePolicyError):
            TwoStepUkeirePolicy().choose_action(
                _decision((MANZU_1, MANZU_2), (_discard(MANZU_7),))
            )

    def test_ambiguous_non_discard_decision_still_prefers_riichi(self) -> None:
        # Issue #76: Always Riichi baseline. This combination is synthetic
        # (RiichiAction and KyuushuKyuuhaiAction do not co-occur in a real
        # game), but legal_actions is the source of truth for legality, so
        # the winning > Riichi > ... priority applies uniformly.
        riichi = RiichiAction(actor=Seat.SEAT_0)
        kyuushu = KyuushuKyuuhaiAction(actor=Seat.SEAT_0)

        for actions in itertools.permutations((riichi, kyuushu)):
            with self.subTest(actions=actions):
                self.assertEqual(
                    TwoStepUkeirePolicy().choose_action(_decision((), actions)),
                    riichi,
                )

    def test_riichi_is_chosen_over_pass(self) -> None:
        # Issue #76: Always Riichi baseline outranks the conservative Pass
        # fallback whenever a legal RiichiAction is present.
        pass_action = PassAction(actor=Seat.SEAT_0)
        riichi = RiichiAction(actor=Seat.SEAT_0)

        for actions in itertools.permutations((riichi, pass_action)):
            with self.subTest(actions=actions):
                self.assertEqual(
                    TwoStepUkeirePolicy().choose_action(_decision((), actions)),
                    riichi,
                )


class PolicyGenerationAndScopeTest(unittest.TestCase):
    def test_all_five_policy_generations_are_public(self) -> None:
        import lisjong.policies as policies

        self.assertEqual(
            set(policies.__all__),
            {
                "GenbutsuDefenseTwoStepUkeirePolicy",
                "MinimalPolicy",
                "ShantenPolicy",
                "UkeirePolicy",
                "TwoStepUkeirePolicy",
            },
        )

    def test_policy_module_has_no_belief_reference_or_environment_import(self) -> None:
        tree = ast.parse(inspect.getsource(two_step))
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
                    "lisjong.belief",
                    "mahjong",
                    "riichienv",
                    "websockets",
                )
            )
        )

    def test_two_step_policy_has_an_independent_error_boundary(self) -> None:
        from lisjong.policies.ukeire import UkeirePolicyError

        self.assertFalse(issubclass(TwoStepUkeirePolicyError, UkeirePolicyError))
        self.assertFalse(issubclass(UkeirePolicyError, TwoStepUkeirePolicyError))


if __name__ == "__main__":
    unittest.main()
