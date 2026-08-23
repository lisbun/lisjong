"""Issue #78 `GenbutsuDefenseTwoStepUkeirePolicy`のunit test。"""

import ast
import inspect
import itertools
import unittest
from unittest.mock import patch

import lisjong.policies.genbutsu_defense_two_step_ukeire as genbutsu
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies import (
    GenbutsuDefenseTwoStepUkeirePolicy,
    TwoStepUkeirePolicy,
)
from lisjong.policy_contract.action import (
    DiscardAction,
    PassAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.decision_trace import DecisionTraceRecorder
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_execution import execute_policy_with_trace
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
        tiles.extend(
            _tile(category, 5 if rank == "0" else int(rank), red=rank == "0")
            for rank in ranks
        )
        ranks = ""
    if ranks:
        raise ValueError(f"hand spec has trailing ranks: {spec!r}")
    return tuple(tiles)


MANZU_1 = _tile(TileCategory.MANZU, 1)
MANZU_4 = _tile(TileCategory.MANZU, 4)
MANZU_5 = _tile(TileCategory.MANZU, 5)
MANZU_5_RED = _tile(TileCategory.MANZU, 5, red=True)
MANZU_8 = _tile(TileCategory.MANZU, 8)
PINZU_9 = _tile(TileCategory.PINZU, 9)
SOUZU_9 = _tile(TileCategory.SOUZU, 9)
EAST = _tile(TileCategory.HONOR, 1)
WHITE_DRAGON = _tile(TileCategory.HONOR, 5)
RED_DRAGON = _tile(TileCategory.HONOR, 7)

_TWO_STEP_HAND = _hand("345m56679s333577z")
_FAR_HAND = _hand("489m128p14558s144z")
_TANKI_VERSUS_ONE_SHANTEN_HAND = _hand("234m567m234p567p5s7z")


def _player(
    *,
    riichi: RiichiState = RiichiState.NONE,
    discards: tuple[Discard, ...] = (),
    melds: tuple[PublicMeld, ...] = (),
) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=discards,
        melds=melds,
        riichi=riichi,
    )


def _history(
    tile: Tile,
    *,
    order: int = 0,
    called_by: Seat | None = None,
) -> Discard:
    return Discard(
        tile=tile,
        tsumogiri=False,
        order=order,
        called_by=called_by,
    )


def _players(
    *threats: tuple[Seat, RiichiState, tuple[Discard, ...]],
    self_seat: Seat = Seat.SEAT_0,
    self_riichi: RiichiState = RiichiState.NONE,
) -> tuple[PlayerPublicState, ...]:
    players = [_player() for _ in Seat]
    players[int(self_seat)] = _player(riichi=self_riichi)
    for seat, state, discards in threats:
        players[int(seat)] = _player(riichi=state, discards=discards)
    return tuple(players)


def _decision(
    concealed_tiles: tuple[Tile, ...],
    actions: tuple[object, ...],
    *,
    players: tuple[PlayerPublicState, ...] | None = None,
    self_seat: Seat = Seat.SEAT_0,
) -> DecisionContext:
    return DecisionContext(
        input=PolicyInput(
            self_seat=self_seat,
            round=RoundState(
                round_wind=Wind.EAST,
                hand_number=1,
                dealer_seat=Seat.SEAT_0,
                honba=0,
                riichi_sticks=0,
                dora_indicators=(),
                live_wall_tiles_remaining=70,
            ),
            players=players if players is not None else _players(self_seat=self_seat),
            own_hand=OwnHandState(
                concealed_tiles=concealed_tiles,
                drawn_tile=None,
            ),
        ),
        legal_actions=actions,
    )


def _discard(tile: Tile, *, actor: Seat = Seat.SEAT_0) -> DiscardAction:
    return DiscardAction(actor=actor, tile=tile, tsumogiri=False)


def _post_discard_shanten(hand: tuple[Tile, ...], action: DiscardAction) -> int:
    remaining = list(hand)
    remaining.remove(action.tile)
    return calculate_shanten(remaining)


def _single_threat(
    *safe_tiles: Tile,
    state: RiichiState = RiichiState.ACCEPTED,
    called_by: Seat | None = None,
) -> tuple[PlayerPublicState, ...]:
    discards = tuple(
        _history(tile, order=order, called_by=called_by)
        for order, tile in enumerate(safe_tiles)
    )
    return _players((Seat.SEAT_1, state, discards))


class PriorityAndActivationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GenbutsuDefenseTwoStepUkeirePolicy()

    def test_winning_action_still_has_priority_during_opponent_riichi(self) -> None:
        discard = _discard(SOUZU_9)
        ron = RonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            winning_tile=MANZU_1,
        )
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_1)
        players = _single_threat(SOUZU_9)

        for actions in itertools.permutations((discard, ron, tsumo)):
            with self.subTest(actions=actions):
                self.assertIs(
                    self.policy.choose_action(
                        _decision(_TWO_STEP_HAND, actions, players=players)
                    ),
                    ron,
                )

    def test_own_riichi_still_has_priority_during_opponent_riichi(self) -> None:
        riichi = RiichiAction(actor=Seat.SEAT_0)
        actions = (riichi, _discard(SOUZU_9), _discard(WHITE_DRAGON))
        players = _single_threat(SOUZU_9)

        for ordered_actions in itertools.permutations(actions):
            with self.subTest(actions=ordered_actions):
                self.assertIs(
                    self.policy.choose_action(
                        _decision(
                            _TWO_STEP_HAND,
                            ordered_actions,
                            players=players,
                        )
                    ),
                    riichi,
                )

    def test_one_shanten_hand_prefers_only_legal_genbutsu(self) -> None:
        safe = _discard(MANZU_4)
        efficient = _discard(MANZU_5)
        decision = _decision(
            _TWO_STEP_HAND,
            (efficient, safe),
            players=_single_threat(MANZU_4),
        )

        self.assertIs(TwoStepUkeirePolicy().choose_action(decision), efficient)
        self.assertIs(self.policy.choose_action(decision), safe)

    def test_two_or_more_shanten_hand_also_prefers_genbutsu(self) -> None:
        safe = _discard(MANZU_8)
        efficient = _discard(EAST)
        decision = _decision(
            _FAR_HAND,
            (efficient, safe),
            players=_single_threat(MANZU_8),
        )
        post_discard_shanten = tuple(
            _post_discard_shanten(_FAR_HAND, action) for action in (efficient, safe)
        )

        self.assertGreaterEqual(min(post_discard_shanten), 2)
        self.assertIs(TwoStepUkeirePolicy().choose_action(decision), efficient)
        self.assertIs(self.policy.choose_action(decision), safe)

    def test_tenpai_available_uses_all_legal_discards_before_filtering(self) -> None:
        safe_but_breaks_tenpai = _discard(MANZU_4)
        dangerous_but_keeps_tenpai = _discard(RED_DRAGON)
        decision = _decision(
            _TANKI_VERSUS_ONE_SHANTEN_HAND,
            (safe_but_breaks_tenpai, dangerous_but_keeps_tenpai),
            players=_single_threat(MANZU_4),
        )

        self.assertEqual(
            _post_discard_shanten(
                _TANKI_VERSUS_ONE_SHANTEN_HAND,
                safe_but_breaks_tenpai,
            ),
            1,
        )
        self.assertEqual(
            _post_discard_shanten(
                _TANKI_VERSUS_ONE_SHANTEN_HAND,
                dangerous_but_keeps_tenpai,
            ),
            0,
        )
        self.assertIs(
            TwoStepUkeirePolicy().choose_action(decision),
            dangerous_but_keeps_tenpai,
        )
        self.assertIs(
            self.policy.choose_action(decision),
            dangerous_but_keeps_tenpai,
        )


class GenbutsuSetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GenbutsuDefenseTwoStepUkeirePolicy()
        self.discard_9s = _discard(SOUZU_9)
        self.discard_white = _discard(WHITE_DRAGON)

    def test_multiple_genbutsu_use_existing_two_step_semantics(self) -> None:
        decision = _decision(
            _TWO_STEP_HAND,
            (self.discard_9s, self.discard_white),
            players=_single_threat(SOUZU_9, WHITE_DRAGON),
        )

        self.assertIs(self.policy.choose_action(decision), self.discard_white)

    def test_no_genbutsu_falls_back_to_normal_two_step(self) -> None:
        decision = _decision(
            _TWO_STEP_HAND,
            (self.discard_9s, self.discard_white),
            players=_single_threat(EAST),
        )

        self.assertIs(self.policy.choose_action(decision), self.discard_white)

    def test_no_opponent_riichi_falls_back_to_normal_two_step(self) -> None:
        decision = _decision(
            _TWO_STEP_HAND,
            (self.discard_9s, self.discard_white),
        )

        self.assertIs(self.policy.choose_action(decision), self.discard_white)

    def test_multiple_riichi_require_common_genbutsu(self) -> None:
        players = _players(
            (
                Seat.SEAT_1,
                RiichiState.DECLARED,
                (_history(SOUZU_9), _history(WHITE_DRAGON, order=1)),
            ),
            (
                Seat.SEAT_2,
                RiichiState.ACCEPTED,
                (_history(SOUZU_9),),
            ),
        )
        decision = _decision(
            _TWO_STEP_HAND,
            (self.discard_white, self.discard_9s),
            players=players,
        )

        self.assertIs(self.policy.choose_action(decision), self.discard_9s)

    def test_partial_safety_without_intersection_is_not_genbutsu(self) -> None:
        players = _players(
            (Seat.SEAT_1, RiichiState.ACCEPTED, (_history(SOUZU_9),)),
            (Seat.SEAT_2, RiichiState.ACCEPTED, (_history(WHITE_DRAGON),)),
        )
        decision = _decision(
            _TWO_STEP_HAND,
            (self.discard_9s, self.discard_white),
            players=players,
        )

        self.assertIs(self.policy.choose_action(decision), self.discard_white)

    def test_called_discard_is_still_genbutsu(self) -> None:
        players = _single_threat(SOUZU_9, called_by=Seat.SEAT_2)
        decision = _decision(
            _TWO_STEP_HAND,
            (self.discard_white, self.discard_9s),
            players=players,
        )

        self.assertIs(self.policy.choose_action(decision), self.discard_9s)

    def test_meld_tiles_are_not_reverse_derived_as_genbutsu(self) -> None:
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(MANZU_4, MANZU_4, MANZU_4),
            from_seat=Seat.SEAT_0,
            called_tile=MANZU_4,
        )
        players = list(_players())
        players[int(Seat.SEAT_1)] = _player(
            riichi=RiichiState.ACCEPTED,
            melds=(pon,),
        )
        decision = _decision(
            _TWO_STEP_HAND,
            (_discard(MANZU_4), _discard(MANZU_5)),
            players=tuple(players),
        )

        with patch.object(
            genbutsu,
            "_evaluate_and_choose_prepared",
            side_effect=lambda _input, candidates, _evaluator: (
                candidates[0].action,
                (),
            ),
        ) as filtered_choice:
            selected = self.policy.choose_action(decision)

        candidate_actions = tuple(
            candidate.action for candidate in filtered_choice.call_args.args[1]
        )
        self.assertEqual(candidate_actions, decision.legal_actions)
        self.assertIs(selected, decision.legal_actions[0])

    def test_red_and_normal_five_share_safety_but_keep_action_identity(self) -> None:
        normal = _discard(MANZU_5)
        red = _discard(MANZU_5_RED)
        concealed = (MANZU_1, MANZU_5, MANZU_5_RED, PINZU_9, EAST)
        players = _single_threat(MANZU_5)

        with patch.object(
            genbutsu,
            "_evaluate_and_choose_prepared",
            side_effect=lambda _input, candidates, _evaluator: (
                candidates[0].action,
                (),
            ),
        ) as choose_discard:
            selected = self.policy.choose_action(
                _decision(concealed, (normal, red), players=players)
            )

        self.assertEqual(
            tuple(candidate.action for candidate in choose_discard.call_args.args[1]),
            (normal, red),
        )
        self.assertIs(selected, normal)


class ThreatIdentityAndDeterminismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GenbutsuDefenseTwoStepUkeirePolicy()

    def test_declared_and_accepted_are_both_threats(self) -> None:
        for state in (RiichiState.DECLARED, RiichiState.ACCEPTED):
            safe = _discard(SOUZU_9)
            efficient = _discard(WHITE_DRAGON)
            with self.subTest(state=state):
                self.assertIs(
                    self.policy.choose_action(
                        _decision(
                            _TWO_STEP_HAND,
                            (efficient, safe),
                            players=_single_threat(SOUZU_9, state=state),
                        )
                    ),
                    safe,
                )

    def test_self_riichi_is_excluded_for_both_active_states(self) -> None:
        for state in (RiichiState.DECLARED, RiichiState.ACCEPTED):
            safe_if_self_counted = _discard(SOUZU_9)
            efficient = _discard(WHITE_DRAGON)
            players = _players(self_riichi=state)
            with self.subTest(state=state):
                self.assertIs(
                    self.policy.choose_action(
                        _decision(
                            _TWO_STEP_HAND,
                            (safe_if_self_counted, efficient),
                            players=players,
                        )
                    ),
                    efficient,
                )

    def test_player_index_is_seat_identity_when_self_is_not_seat_zero(self) -> None:
        self_seat = Seat.SEAT_2
        safe = _discard(SOUZU_9, actor=self_seat)
        efficient = _discard(WHITE_DRAGON, actor=self_seat)
        players = _players(
            (Seat.SEAT_0, RiichiState.ACCEPTED, (_history(SOUZU_9),)),
            self_seat=self_seat,
        )

        self.assertIs(
            self.policy.choose_action(
                _decision(
                    _TWO_STEP_HAND,
                    (efficient, safe),
                    players=players,
                    self_seat=self_seat,
                )
            ),
            safe,
        )

    def test_legal_action_and_discard_history_order_do_not_change_choice(self) -> None:
        safe = _discard(SOUZU_9)
        efficient = _discard(WHITE_DRAGON)
        choices = set()
        for actions in itertools.permutations((safe, efficient)):
            for safe_history in (
                (_history(SOUZU_9), _history(EAST, order=1)),
                (_history(EAST, order=1), _history(SOUZU_9)),
            ):
                players = _players((Seat.SEAT_1, RiichiState.ACCEPTED, safe_history))
                choices.add(
                    self.policy.choose_action(
                        _decision(_TWO_STEP_HAND, actions, players=players)
                    )
                )

        self.assertEqual(choices, {safe})

    def test_selected_defensive_action_is_original_legal_instance(self) -> None:
        safe = _discard(SOUZU_9)
        selected = self.policy.choose_action(
            _decision(
                _TWO_STEP_HAND,
                (_discard(WHITE_DRAGON), safe),
                players=_single_threat(SOUZU_9),
            )
        )

        self.assertIs(selected, safe)

    def test_pass_fallback_is_inherited_when_no_discard_exists(self) -> None:
        pass_action = PassAction(actor=Seat.SEAT_0)

        self.assertIs(
            self.policy.choose_action(
                _decision((), (pass_action,), players=_single_threat(SOUZU_9))
            ),
            pass_action,
        )


class PublicGenerationAndScopeTest(unittest.TestCase):
    def test_all_five_policy_generations_are_public(self) -> None:
        import lisjong.policies as policies

        self.assertEqual(
            set(policies.__all__),
            {
                "GenbutsuDefenseTwoStepUkeirePolicy",
                "MinimalPolicy",
                "ShantenPolicy",
                "TwoStepUkeirePolicy",
                "UkeirePolicy",
            },
        )

    def test_policy_module_uses_no_hidden_or_environment_dependency(self) -> None:
        tree = ast.parse(inspect.getsource(genbutsu))
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


class GenbutsuDefenseTraceBehaviorTest(unittest.TestCase):
    """Issue #97: trace on/offでdefense behaviorが完全に一致することを固定する。"""

    def setUp(self) -> None:
        self.policy = GenbutsuDefenseTwoStepUkeirePolicy()

    def _scenarios(self) -> dict[str, DecisionContext]:
        discard_9s = _discard(SOUZU_9)
        discard_white = _discard(WHITE_DRAGON)
        multiple_riichi_players = _players(
            (
                Seat.SEAT_1,
                RiichiState.DECLARED,
                (_history(SOUZU_9), _history(WHITE_DRAGON, order=1)),
            ),
            (Seat.SEAT_2, RiichiState.ACCEPTED, (_history(SOUZU_9),)),
        )
        return {
            "no opponent riichi": _decision(
                _TWO_STEP_HAND, (discard_9s, discard_white)
            ),
            "non-tenpai with common genbutsu": _decision(
                _TWO_STEP_HAND,
                (_discard(MANZU_5), _discard(MANZU_4)),
                players=_single_threat(MANZU_4),
            ),
            "non-tenpai without common genbutsu": _decision(
                _TWO_STEP_HAND,
                (discard_9s, discard_white),
                players=_single_threat(EAST),
            ),
            "tenpai stays available": _decision(
                _TANKI_VERSUS_ONE_SHANTEN_HAND,
                (_discard(MANZU_4), _discard(RED_DRAGON)),
                players=_single_threat(MANZU_4),
            ),
            "multiple genbutsu": _decision(
                _TWO_STEP_HAND,
                (discard_9s, discard_white),
                players=_single_threat(SOUZU_9, WHITE_DRAGON),
            ),
            "multiple riichi need common genbutsu": _decision(
                _TWO_STEP_HAND,
                (discard_white, discard_9s),
                players=multiple_riichi_players,
            ),
            "two or more shanten": _decision(
                _FAR_HAND,
                (_discard(EAST), _discard(MANZU_8)),
                players=_single_threat(MANZU_8),
            ),
        }

    def test_selected_action_is_identical_with_and_without_trace(self) -> None:
        for name, decision in self._scenarios().items():
            with self.subTest(scenario=name):
                untraced = self.policy.choose_action(decision)
                recorder = DecisionTraceRecorder()

                traced = execute_policy_with_trace(self.policy, decision, recorder)

                (trace,) = recorder.snapshot()
                self.assertIs(traced, untraced)
                self.assertIs(trace.selected_action, untraced)

    def test_defense_decisions_report_no_analysis_in_this_issue(self) -> None:
        for name, decision in self._scenarios().items():
            with self.subTest(scenario=name):
                recorder = DecisionTraceRecorder()

                execute_policy_with_trace(self.policy, decision, recorder)

                self.assertIsNone(recorder.snapshot()[0].analysis)

    def test_traced_defense_does_not_degrade_to_the_pure_two_step_path(self) -> None:
        safe = _discard(MANZU_4)
        efficient = _discard(MANZU_5)
        decision = _decision(
            _TWO_STEP_HAND,
            (efficient, safe),
            players=_single_threat(MANZU_4),
        )
        base_recorder = DecisionTraceRecorder()
        defense_recorder = DecisionTraceRecorder()

        base_selected = execute_policy_with_trace(
            TwoStepUkeirePolicy(), decision, base_recorder
        )
        defense_selected = execute_policy_with_trace(
            self.policy, decision, defense_recorder
        )

        self.assertIs(base_selected, efficient)
        self.assertIs(defense_selected, safe)
        self.assertIsNotNone(base_recorder.snapshot()[0].analysis)
        self.assertIsNone(defense_recorder.snapshot()[0].analysis)

    def test_analysis_capability_is_explicitly_overridden_not_inherited(self) -> None:
        self.assertIsNot(
            GenbutsuDefenseTwoStepUkeirePolicy._decide_discard,
            TwoStepUkeirePolicy._decide_discard,
        )
        self.assertIn(
            "_decide_discard",
            vars(GenbutsuDefenseTwoStepUkeirePolicy),
        )

    def test_defense_decision_path_runs_exactly_once_per_traced_decision(self) -> None:
        decision = _decision(
            _TWO_STEP_HAND,
            (_discard(MANZU_5), _discard(MANZU_4)),
            players=_single_threat(MANZU_4),
        )

        with patch.object(
            GenbutsuDefenseTwoStepUkeirePolicy,
            "_choose_defense_discard",
            autospec=True,
            side_effect=GenbutsuDefenseTwoStepUkeirePolicy._choose_defense_discard,
        ) as defense_path:
            execute_policy_with_trace(self.policy, decision, DecisionTraceRecorder())

        self.assertEqual(defense_path.call_count, 1)

    def test_analysis_capable_decision_returns_the_defense_action(self) -> None:
        safe = _discard(MANZU_4)
        efficient = _discard(MANZU_5)
        decision = _decision(
            _TWO_STEP_HAND,
            (efficient, safe),
            players=_single_threat(MANZU_4),
        )

        proposed = self.policy.choose_action_with_analysis(decision)

        self.assertIsInstance(proposed, PolicyDecision)
        self.assertIs(proposed.action, safe)
        self.assertIsNone(proposed.analysis)

    def test_defense_policy_keeps_the_inherited_analysis_capability(self) -> None:
        # GenbutsuDefenseは`choose_action()`をoverrideしないため、traced
        # executionはinheritしたanalysis capabilityを通り、そこから
        # override済みのdefense decision pathへdispatchされる。
        decision = _decision(
            _TWO_STEP_HAND,
            (_discard(MANZU_5), _discard(MANZU_4)),
            players=_single_threat(MANZU_4),
        )
        recorder = DecisionTraceRecorder()

        with patch.object(
            TwoStepUkeirePolicy,
            "choose_action",
            side_effect=AssertionError("analysis capability must be used"),
        ):
            traced = execute_policy_with_trace(self.policy, decision, recorder)

        self.assertIs(traced, decision.legal_actions[1])
        self.assertIsNone(recorder.snapshot()[0].analysis)


if __name__ == "__main__":
    unittest.main()
