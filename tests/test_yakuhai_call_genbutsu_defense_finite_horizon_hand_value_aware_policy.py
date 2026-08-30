"""Issue #145 call-aware Policyのfocused deterministic tests。"""

import ast
import inspect
import itertools
import pickle
import unittest
from unittest.mock import patch

import lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware as parent
import lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware as call_policy
from lisjong.policies import (
    GenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
    YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    _best_post_call_shanten,
    _defense_suppresses_call,
    _has_open_yakuhai,
    _kuikae_forbidden_tile_types,
    _remove_exact_consumed_tiles,
)
from lisjong.policy_contract.action import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DiscardAction,
    KakanAction,
    PassAction,
    PonAction,
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


EAST = _tile(TileCategory.HONOR, 1)
SOUTH = _tile(TileCategory.HONOR, 2)
WEST = _tile(TileCategory.HONOR, 3)
WHITE = _tile(TileCategory.HONOR, 5)
MANZU_1 = _tile(TileCategory.MANZU, 1)
MANZU_2 = _tile(TileCategory.MANZU, 2)
MANZU_3 = _tile(TileCategory.MANZU, 3)
MANZU_4 = _tile(TileCategory.MANZU, 4)
MANZU_5 = _tile(TileCategory.MANZU, 5)
MANZU_5_RED = _tile(TileCategory.MANZU, 5, red=True)
MANZU_6 = _tile(TileCategory.MANZU, 6)
MANZU_7 = _tile(TileCategory.MANZU, 7)
MANZU_8 = _tile(TileCategory.MANZU, 8)
MANZU_9 = _tile(TileCategory.MANZU, 9)
PINZU_9 = _tile(TileCategory.PINZU, 9)
SOUZU_9 = _tile(TileCategory.SOUZU, 9)


def _meld(kind: MeldKind, tile: Tile) -> PublicMeld:
    if kind is MeldKind.ANKAN:
        return PublicMeld(
            kind=kind,
            tiles=(tile,) * 4,
            from_seat=None,
            called_tile=None,
        )
    tile_count = 4 if kind in {MeldKind.DAIMINKAN, MeldKind.KAKAN} else 3
    return PublicMeld(
        kind=kind,
        tiles=(tile,) * tile_count,
        from_seat=Seat.SEAT_1,
        called_tile=tile,
    )


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


def _players(
    *,
    own_melds: tuple[PublicMeld, ...] = (),
    threat_discards: tuple[tuple[Tile, ...], ...] = (),
) -> tuple[PlayerPublicState, ...]:
    result = [_player(melds=own_melds), _player(), _player(), _player()]
    for offset, discards in enumerate(threat_discards, start=1):
        result[offset] = _player(
            riichi=RiichiState.ACCEPTED,
            discards=tuple(
                _discard_history(tile, order) for order, tile in enumerate(discards)
            ),
        )
    return tuple(result)  # type: ignore[return-value]


def _input(
    concealed_tiles: tuple[Tile, ...],
    *,
    own_melds: tuple[PublicMeld, ...] = (),
    threat_discards: tuple[tuple[Tile, ...], ...] = (),
    dealer_seat: Seat = Seat.SEAT_0,
    round_wind: Wind = Wind.EAST,
) -> PolicyInput:
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=round_wind,
            hand_number=1,
            dealer_seat=dealer_seat,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(),
            live_wall_tiles_remaining=60,
        ),
        players=_players(
            own_melds=own_melds,
            threat_discards=threat_discards,
        ),
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


def _pon(tile: Tile, consumed: tuple[Tile, Tile] | None = None) -> PonAction:
    return PonAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_1,
        called_tile=tile,
        consumed_tiles=consumed if consumed is not None else (tile, tile),
    )


def _chi(called: Tile, consumed: tuple[Tile, Tile]) -> ChiAction:
    return ChiAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_3,
        called_tile=called,
        consumed_tiles=consumed,
    )


PASS = PassAction(actor=Seat.SEAT_0)
OPEN_WHITE = (_meld(MeldKind.PON, WHITE),)


class InitialYakuhaiPonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()

    def _improving_pair_hand(self, tile: Tile) -> tuple[Tile, ...]:
        return _hand("123456m789p19s") + (tile, tile)

    def test_dragon_pair_with_strict_improvement_is_ponned(self) -> None:
        action = _pon(WHITE)
        selected = self.policy.choose_action(
            _decision(self._improving_pair_hand(WHITE), (PASS, action))
        )
        self.assertIs(selected, action)

    def test_seat_wind_pair_with_strict_improvement_is_ponned(self) -> None:
        action = _pon(SOUTH)
        selected = self.policy.choose_action(
            _decision(
                self._improving_pair_hand(SOUTH),
                (PASS, action),
                dealer_seat=Seat.SEAT_3,
            )
        )
        self.assertIs(selected, action)

    def test_round_wind_pair_with_strict_improvement_is_ponned(self) -> None:
        action = _pon(WEST)
        selected = self.policy.choose_action(
            _decision(
                self._improving_pair_hand(WEST),
                (PASS, action),
                round_wind=Wind.WEST,
            )
        )
        self.assertIs(selected, action)

    def test_double_wind_uses_the_shared_yakuhai_semantic(self) -> None:
        action = _pon(EAST)
        selected = self.policy.choose_action(
            _decision(self._improving_pair_hand(EAST), (PASS, action))
        )
        self.assertIs(selected, action)

    def test_same_shanten_yakuhai_pon_is_passed(self) -> None:
        action = _pon(WHITE)
        concealed = _hand("123456m789p12s") + (WHITE, WHITE)
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (action, PASS))), PASS
        )

    def test_non_yakuhai_pon_before_unlock_is_passed(self) -> None:
        action = _pon(SOUTH)
        concealed = self._improving_pair_hand(SOUTH)
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (action, PASS))), PASS
        )

    def test_chi_before_unlock_is_passed(self) -> None:
        action = _chi(MANZU_1, (MANZU_2, MANZU_3))
        concealed = _hand("23123456m19p123s")
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (action, PASS))), PASS
        )


class ConcealedTripletAndModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()

    def test_yakuhai_concealed_triplet_does_not_unlock_call_mode(self) -> None:
        policy_input = _input(_hand("123456m19s12p") + (WHITE,) * 3)
        self.assertFalse(_has_open_yakuhai(policy_input))

    def test_yakuhai_triplet_and_improving_chi_still_passes(self) -> None:
        action = _chi(MANZU_1, (MANZU_2, MANZU_3))
        concealed = _hand("23123456p19s") + (WHITE,) * 3
        with patch.object(call_policy, "_best_post_call_shanten", return_value=-1):
            self.assertIs(
                self.policy.choose_action(_decision(concealed, (action, PASS))), PASS
            )

    def test_initial_pon_never_breaks_a_concealed_triplet(self) -> None:
        action = _pon(WHITE)
        concealed = _hand("123456m19s12p") + (WHITE,) * 3
        self.assertIs(
            self.policy.choose_action(_decision(concealed, (action, PASS))), PASS
        )

    def test_unlocked_pon_never_breaks_a_concealed_triplet(self) -> None:
        action = _pon(MANZU_2)
        concealed = (MANZU_2,) * 3 + _hand("123p19s45p")
        with patch.object(call_policy, "_best_post_call_shanten", return_value=-1):
            self.assertIs(
                self.policy.choose_action(
                    _decision(concealed, (action, PASS), own_melds=OPEN_WHITE)
                ),
                PASS,
            )

    def test_ankan_and_daiminkan_do_not_unlock_but_lossless_kakan_does(self) -> None:
        concealed = _hand("23m123456p19s")
        ankan_input = _input(concealed, own_melds=(_meld(MeldKind.ANKAN, WHITE),))
        daiminkan_input = _input(
            concealed, own_melds=(_meld(MeldKind.DAIMINKAN, WHITE),)
        )
        kakan_input = _input(concealed, own_melds=(_meld(MeldKind.KAKAN, WHITE),))
        self.assertFalse(_has_open_yakuhai(ankan_input))
        self.assertFalse(_has_open_yakuhai(daiminkan_input))
        self.assertTrue(_has_open_yakuhai(kakan_input))


class PostYakuhaiPonCallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()

    def test_improving_chi_is_selected(self) -> None:
        action = _chi(MANZU_1, (MANZU_2, MANZU_3))
        concealed = _hand("23m123456p19s")
        self.assertIs(
            self.policy.choose_action(
                _decision(concealed, (PASS, action), own_melds=OPEN_WHITE)
            ),
            action,
        )

    def test_improving_non_yakuhai_pon_is_selected(self) -> None:
        action = _pon(MANZU_2)
        concealed = _hand("22m123456p19s")
        self.assertIs(
            self.policy.choose_action(
                _decision(concealed, (action, PASS), own_melds=OPEN_WHITE)
            ),
            action,
        )

    def test_same_shanten_chi_and_pon_are_passed(self) -> None:
        chi = _chi(MANZU_1, (MANZU_2, MANZU_3))
        pon = _pon(MANZU_2)
        self.assertIs(
            self.policy.choose_action(
                _decision(
                    _hand("23m123456p11s"),
                    (chi, PASS),
                    own_melds=OPEN_WHITE,
                )
            ),
            PASS,
        )
        self.assertIs(
            self.policy.choose_action(
                _decision(
                    _hand("22m123456p11s"),
                    (PASS, pon),
                    own_melds=OPEN_WHITE,
                )
            ),
            PASS,
        )

    def test_worsening_call_is_passed(self) -> None:
        action = _chi(MANZU_1, (MANZU_2, MANZU_3))
        concealed = _hand("23m123456p19s")
        with patch.object(call_policy, "_best_post_call_shanten", return_value=2):
            self.assertIs(
                self.policy.choose_action(
                    _decision(concealed, (action, PASS), own_melds=OPEN_WHITE)
                ),
                PASS,
            )

    def test_multiple_calls_choose_minimum_post_call_shanten(self) -> None:
        lower = _chi(MANZU_1, (MANZU_2, MANZU_3))
        higher = _chi(MANZU_4, (MANZU_2, MANZU_3))
        concealed = _hand("23m123456p19s")
        best = {lower: 0, higher: 1}
        with (
            patch.object(call_policy, "calculate_shanten", return_value=2),
            patch.object(
                call_policy,
                "_best_post_call_shanten",
                side_effect=lambda hand, action: best[action],
            ),
        ):
            selected = self.policy.choose_action(
                _decision(
                    concealed,
                    (higher, PASS, lower),
                    own_melds=OPEN_WHITE,
                )
            )
        self.assertIs(selected, lower)

    def test_tie_break_is_canonical_and_input_order_independent(self) -> None:
        canonical = _chi(MANZU_1, (MANZU_2, MANZU_3))
        other = _chi(MANZU_4, (MANZU_2, MANZU_3))
        concealed = _hand("23m123456p19s")
        with (
            patch.object(call_policy, "calculate_shanten", return_value=2),
            patch.object(call_policy, "_best_post_call_shanten", return_value=1),
        ):
            results = {
                self.policy.choose_action(
                    _decision(
                        concealed,
                        permutation,
                        own_melds=OPEN_WHITE,
                    )
                )
                for permutation in itertools.permutations((canonical, other, PASS))
            }
        self.assertEqual(results, {canonical})


class KuikaeAndStableShantenTest(unittest.TestCase):
    def test_pon_forbids_the_called_tile_type(self) -> None:
        action = _pon(MANZU_5)
        self.assertEqual(
            _kuikae_forbidden_tile_types(action), frozenset({MANZU_5.tile_type})
        )

    def test_chi_called_tile_and_low_high_edge_suji_are_forbidden(self) -> None:
        low = _chi(MANZU_3, (MANZU_4, MANZU_5))
        high = _chi(MANZU_5, (MANZU_3, MANZU_4))
        self.assertEqual(
            _kuikae_forbidden_tile_types(low),
            frozenset({MANZU_3.tile_type, MANZU_6.tile_type}),
        )
        self.assertEqual(
            _kuikae_forbidden_tile_types(high),
            frozenset({MANZU_5.tile_type, MANZU_2.tile_type}),
        )

    def test_middle_chi_has_no_extra_suji_restriction(self) -> None:
        action = _chi(MANZU_4, (MANZU_3, MANZU_5))
        self.assertEqual(
            _kuikae_forbidden_tile_types(action), frozenset({MANZU_4.tile_type})
        )

    def test_chi_does_not_create_out_of_range_suji_types(self) -> None:
        high_123 = _chi(MANZU_3, (MANZU_1, MANZU_2))
        low_789 = _chi(MANZU_7, (MANZU_8, MANZU_9))
        self.assertEqual(
            _kuikae_forbidden_tile_types(high_123),
            frozenset({MANZU_3.tile_type}),
        )
        self.assertEqual(
            _kuikae_forbidden_tile_types(low_789),
            frozenset({MANZU_7.tile_type}),
        )

    def test_kuikae_forbidden_best_discard_cannot_qualify_a_call(self) -> None:
        action = _chi(MANZU_3, (MANZU_4, MANZU_5))
        concealed = (MANZU_4, MANZU_5, MANZU_6, PINZU_9)

        def shanten(tiles) -> int:
            materialized = tuple(tiles)
            if len(materialized) == 4:
                return 1
            (remaining,) = materialized
            return 0 if remaining == PINZU_9 else 1

        with patch.object(call_policy, "calculate_shanten", side_effect=shanten):
            restricted_best = _best_post_call_shanten(concealed, action)
            selected = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy().choose_action(
                _decision(
                    concealed,
                    (action, PASS),
                    own_melds=OPEN_WHITE,
                )
            )

        self.assertEqual(restricted_best, 1)
        self.assertEqual(shanten((PINZU_9,)), 0)
        self.assertIs(selected, PASS)

    def test_only_mandatory_discard_stable_hands_are_evaluated(self) -> None:
        action = _pon(WHITE)
        concealed = _hand("123456m789p19s") + (WHITE, WHITE)
        observed_lengths: list[int] = []

        def shanten(tiles) -> int:
            materialized = tuple(tiles)
            observed_lengths.append(len(materialized))
            return 0

        with patch.object(call_policy, "calculate_shanten", side_effect=shanten):
            self.assertEqual(_best_post_call_shanten(concealed, action), 0)
        self.assertEqual(set(observed_lengths), {10})

    def test_exact_consumed_identity_preserves_the_other_five(self) -> None:
        remaining = _remove_exact_consumed_tiles(
            (MANZU_5_RED, MANZU_5, MANZU_6),
            (MANZU_5, MANZU_6),
        )
        self.assertEqual(remaining, [MANZU_5_RED])

    def test_red_and_normal_five_have_the_same_structural_shanten(self) -> None:
        consume_red = _chi(MANZU_7, (MANZU_5_RED, MANZU_6))
        consume_normal = _chi(MANZU_7, (MANZU_5, MANZU_6))
        concealed = (MANZU_5_RED, MANZU_5, MANZU_6, PINZU_9)
        self.assertEqual(
            _best_post_call_shanten(concealed, consume_red),
            _best_post_call_shanten(concealed, consume_normal),
        )


class DefenseAndPriorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()
        self.improving_chi = _chi(MANZU_1, (MANZU_2, MANZU_3))
        self.concealed = _hand("23m123456p19s")

    def test_riichi_non_tenpai_common_genbutsu_suppresses_call(self) -> None:
        selected = self.policy.choose_action(
            _decision(
                self.concealed,
                (self.improving_chi, PASS),
                own_melds=OPEN_WHITE,
                threat_discards=((SOUZU_9,),),
            )
        )
        self.assertIs(selected, PASS)

    def test_multiple_riichi_use_common_genbutsu_intersection(self) -> None:
        policy_input = _input(
            self.concealed,
            own_melds=OPEN_WHITE,
            threat_discards=((SOUZU_9, MANZU_9), (SOUZU_9, PINZU_9)),
        )
        self.assertTrue(_defense_suppresses_call(policy_input, current_shanten=1))

    def test_no_common_genbutsu_continues_normal_call_logic(self) -> None:
        selected = self.policy.choose_action(
            _decision(
                self.concealed,
                (PASS, self.improving_chi),
                own_melds=OPEN_WHITE,
                threat_discards=((MANZU_9,),),
            )
        )
        self.assertIs(selected, self.improving_chi)

    def test_tenpai_does_not_activate_the_defense_gate(self) -> None:
        policy_input = _input(
            self.concealed,
            own_melds=OPEN_WHITE,
            threat_discards=((SOUZU_9,),),
        )
        self.assertFalse(_defense_suppresses_call(policy_input, current_shanten=0))

    def test_ron_has_priority_over_pon_and_chi(self) -> None:
        ron = RonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            winning_tile=MANZU_1,
        )
        pon = _pon(WHITE)
        pon_hand = _hand("123456m789p19s") + (WHITE, WHITE)
        self.assertIs(
            self.policy.choose_action(_decision(pon_hand, (PASS, pon, ron))), ron
        )
        self.assertIs(
            self.policy.choose_action(
                _decision(
                    self.concealed,
                    (self.improving_chi, ron, PASS),
                    own_melds=OPEN_WHITE,
                )
            ),
            ron,
        )

    def test_kan_is_never_selected_when_pass_is_legal(self) -> None:
        kan_actions = (
            DaiminkanAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_1,
                called_tile=WHITE,
                consumed_tiles=(WHITE, WHITE, WHITE),
            ),
            AnkanAction(actor=Seat.SEAT_0, tiles=(WHITE,) * 4),
            KakanAction(
                actor=Seat.SEAT_0,
                added_tile=WHITE,
                from_seat=Seat.SEAT_1,
                called_tile=WHITE,
            ),
        )
        for action in kan_actions:
            with self.subTest(action=action):
                self.assertIs(
                    self.policy.choose_action(_decision((WHITE,) * 4, (action, PASS))),
                    PASS,
                )

    def test_actual_next_discard_is_fresh_parent_decision(self) -> None:
        discard_1 = DiscardAction(Seat.SEAT_0, MANZU_1, False)
        discard_9 = DiscardAction(Seat.SEAT_0, MANZU_9, False)
        decision = _decision(
            _hand("123456m789p11s"),
            (discard_1, discard_9),
            own_melds=OPEN_WHITE,
        )
        with (
            patch.object(
                parent,
                "_evaluate_and_choose_discard",
                return_value=discard_9,
            ) as parent_evaluate,
            patch.object(
                call_policy,
                "_best_post_call_shanten",
                side_effect=AssertionError(
                    "actual discard must not use call simulation"
                ),
            ),
        ):
            selected = self.policy.choose_action(decision)
        self.assertIs(selected, discard_9)
        parent_evaluate.assert_called_once()

    def test_ordinary_discard_matches_the_no_call_parent(self) -> None:
        discard_1 = DiscardAction(Seat.SEAT_0, MANZU_1, False)
        discard_9 = DiscardAction(Seat.SEAT_0, MANZU_9, False)
        decision = _decision(_hand("123456m789p11s"), (discard_1, discard_9))
        with patch.object(
            parent,
            "_evaluate_and_choose_discard",
            return_value=discard_1,
        ):
            self.assertIs(
                self.policy.choose_action(decision),
                GenbutsuDefenseFiniteHorizonHandValueAwarePolicy().choose_action(
                    decision
                ),
            )


class StatelessAndPublicContractTest(unittest.TestCase):
    def test_reusing_an_instance_creates_no_mutable_call_history(self) -> None:
        policy = YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()
        initial = _pon(WHITE)
        initial_hand = _hand("123456m789p19s") + (WHITE, WHITE)
        closed_chi = _chi(MANZU_1, (MANZU_2, MANZU_3))
        closed_hand = _hand("23123456m19p123s")

        self.assertIs(
            policy.choose_action(_decision(initial_hand, (PASS, initial))), initial
        )
        self.assertIs(
            policy.choose_action(_decision(closed_hand, (closed_chi, PASS))), PASS
        )
        self.assertEqual(vars(policy), {})

    def test_public_export_module_level_pickle_and_dependency_boundary(self) -> None:
        from lisjong.policies import (
            YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy as imported,
        )

        self.assertIs(
            imported,
            call_policy.YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
        )
        self.assertIs(
            pickle.loads(
                pickle.dumps(
                    YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy
                )
            ),
            YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
        )
        tree = ast.parse(inspect.getsource(call_policy))
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

    def test_policy_adds_no_analysis_recomputation(self) -> None:
        action = _pon(WHITE)
        concealed = _hand("123456m789p19s") + (WHITE, WHITE)
        decision = _decision(concealed, (PASS, action))
        with patch.object(
            call_policy,
            "_best_post_call_shanten",
            wraps=call_policy._best_post_call_shanten,
        ) as evaluate:
            policy_decision = self.policy_decision(decision)
        self.assertIs(policy_decision.action, action)
        self.assertIsNone(policy_decision.analysis)
        self.assertEqual(evaluate.call_count, 1)

    @staticmethod
    def policy_decision(decision: DecisionContext):
        return YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()._decide(
            decision
        )


if __name__ == "__main__":
    unittest.main()
