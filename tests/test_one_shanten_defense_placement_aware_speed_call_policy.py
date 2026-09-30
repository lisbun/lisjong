"""#234の発動境界と既存Championへの委譲を固定する。"""

import unittest
from dataclasses import replace
from unittest.mock import patch

from test_placement_aware_speed_call_policy import (
    ALL_LAST_TOP,
    PASS,
    _discards,
    _input,
    _player,
    _tile,
)

from lisjong.policies import (
    OneShantenDefensePlacementAwareSpeedCallPolicy,
    PlacementAwareSpeedCallPolicy,
)
from lisjong.policies.one_shanten_defense_placement_aware_speed_call import (
    _one_shanten_defense_actions,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat


class OneShantenDefenseTest(unittest.TestCase):
    HAND = "34m678p68p234s55s9s7z"

    def test_real_one_shanten_filters_to_honor(self):
        context = _input(self.HAND, drawn="7z", riichi_discards="3z4z")
        actions = _discards(context)
        self.assertEqual(
            _one_shanten_defense_actions(context, actions),
            tuple(a for a in actions if a.tile == _tile("7z")),
        )

    def test_no_riichi_common_genbutsu_tenpai_and_two_shanten_are_unchanged(self):
        cases = (
            _input(self.HAND),
            _input(self.HAND, riichi_discards="9s3z"),
            _input("123m123p123s789s11z", riichi_discards="3z4z"),
            _input("345m56679s333517z", riichi_discards="9p"),
        )
        for context in cases:
            with self.subTest(context=context):
                actions = _discards(context)
                self.assertIs(_one_shanten_defense_actions(context, actions), actions)
                expected = PolicyDecision(action=actions[0])
                with patch.object(
                    PlacementAwareSpeedCallPolicy,
                    "_decide_discard",
                    return_value=expected,
                ) as parent:
                    actual = OneShantenDefensePlacementAwareSpeedCallPolicy()._decide_discard(
                        context, actions
                    )
                self.assertIs(actual, expected)
                parent.assert_called_once_with(context, actions)

    def test_all_last_top_selection_matches_champion(self):
        context = _input(self.HAND, drawn="7z", riichi_discards="3z4z", **ALL_LAST_TOP)
        decision = DecisionContext(input=context, legal_actions=_discards(context))
        self.assertEqual(
            OneShantenDefensePlacementAwareSpeedCallPolicy().choose_action(decision),
            PlacementAwareSpeedCallPolicy().choose_action(decision),
        )

    def test_multiple_riichi_uses_worst_opponent_and_not_union_genbutsu(self):
        context = _input(self.HAND, riichi_discards="7z")
        players = list(context.players)
        players[2] = _player(riichi=RiichiState.ACCEPTED, discards="9s")
        context = replace(context, players=tuple(players))
        actions = _discards(context)
        # 7z: 0 against one opponent, 8 against the other. 9s: 13 and 0.
        # There is no common genbutsu. The worst-opponent minimum is 7z.
        self.assertEqual(
            _one_shanten_defense_actions(context, actions),
            tuple(a for a in actions if a.tile == _tile("7z")),
        )

    def test_original_action_identity_and_order_survive_filter(self):
        context = _input("34m678p68p234s05s9s7z", drawn="7z", riichi_discards="3z4z")
        actions = _discards(context)
        hand_discard = next(a for a in actions if a.tile == _tile("7z"))
        draw_discard = replace(hand_discard, tsumogiri=True)
        actions = (draw_discard, *actions)
        eligible = _one_shanten_defense_actions(context, actions)
        self.assertEqual(eligible, (draw_discard, hand_discard))
        self.assertIs(eligible[0], draw_discard)
        self.assertIs(eligible[1], hand_discard)
        # A red five is never normalized into a replacement action.
        red = DiscardAction(actor=Seat.SEAT_0, tile=_tile("0s"), tsumogiri=False)
        safe_context = _input("34m678p68p234s05s9s7z", riichi_discards="5s")
        restricted = (red, hand_discard)
        self.assertIs(
            _one_shanten_defense_actions(safe_context, restricted), restricted
        )

    def test_filtered_subset_is_passed_to_parent_once(self):
        context = _input(self.HAND, riichi_discards="3z4z")
        actions = _discards(context)
        selected = next(a for a in actions if a.tile == _tile("7z"))
        expected = PolicyDecision(action=selected)
        with patch.object(
            PlacementAwareSpeedCallPolicy, "_decide_discard", return_value=expected
        ) as parent:
            actual = OneShantenDefensePlacementAwareSpeedCallPolicy()._decide_discard(
                context, actions
            )
        self.assertIs(actual, expected)
        parent.assert_called_once_with(context, (selected,))

    def test_trace_and_untraced_actions_match_and_are_legal(self):
        context = _input(self.HAND, riichi_discards="3z4z")
        decision = DecisionContext(input=context, legal_actions=_discards(context))
        candidate = OneShantenDefensePlacementAwareSpeedCallPolicy()
        selected = candidate.choose_action(decision)
        self.assertIn(selected, decision.legal_actions)
        self.assertEqual(
            candidate.choose_action_with_analysis(decision).action, selected
        )

    def test_non_discard_orchestration_is_inherited(self):
        self.assertIs(
            OneShantenDefensePlacementAwareSpeedCallPolicy._decide,
            PlacementAwareSpeedCallPolicy._decide,
        )
        decision = DecisionContext(
            input=_input("123m123p123s78s11z"), legal_actions=(PASS,)
        )
        self.assertEqual(
            OneShantenDefensePlacementAwareSpeedCallPolicy().choose_action(decision),
            PASS,
        )


if __name__ == "__main__":
    unittest.main()
