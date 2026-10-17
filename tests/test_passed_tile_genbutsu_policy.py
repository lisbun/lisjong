"""#312: 単独リーチ・非聴牌FOLDで確定安全牌を現物として使う候補の動作を固定する。"""

import unittest
from dataclasses import replace

from test_placement_aware_speed_call_policy import (
    ALL_LAST_TOP,
    PASS,
    _discards,
    _hand,
    _input,
    _pon,
    _tile,
)

from lisjong.candidate_boundary import check_candidate_imports, check_candidate_purity
from lisjong.policies import PlacementAwareSpeedCallPolicy
from lisjong.policies.passed_tile_genbutsu_placement_aware_speed_call import (
    PassedTileGenbutsuPlacementAwareSpeedCallPolicy,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat

_MODULE = "lisjong.policies.passed_tile_genbutsu_placement_aware_speed_call"
_FACTORY = f"{_MODULE}:PassedTileGenbutsuPlacementAwareSpeedCallPolicy"

# 34m 678p 68p 234s 55s + 9s・1m: 1向聴。FOLDの攻撃評価ではChampionは1mを切る。
ONE_SHANTEN = "34m678p68p234s55s9s1m"
# 13m 5p 79s 2468p 1357s 5z: 2向聴。河の現物がなければChampionは5zへ絞る。
TWO_SHANTEN = "13m5p79s2468p1357s5z"
# 123m 456p 789s 1122z + ツモ5z: 5zを切れば聴牌を保てる（PUSH）。
TENPAI_KEEPING = "123m456p789s1122z5z"

CHAMPION = PlacementAwareSpeedCallPolicy()
CANDIDATE = PassedTileGenbutsuPlacementAwareSpeedCallPolicy()


def _discards_of(specs, *, start=0):
    return tuple(
        Discard(tile=_tile(spec), tsumogiri=False, order=order, called_by=None)
        for order, spec in enumerate(specs, start)
    )


def _situation(
    hand,
    *,
    river=("3z", "4z"),
    declaration=1,
    others=None,
    second_riichi=False,
    riichi=True,
    **keywords,
):
    """seat 1がorder `declaration`でリーチした局面。`others`は{席: [(order, 牌)]}。"""
    context = _input(hand, **keywords)
    players = list(context.players)
    if riichi:
        players[1] = replace(
            players[1],
            discards=_discards_of(river),
            riichi=RiichiState.ACCEPTED,
            riichi_declaration_discard_order=declaration,
        )
    for seat, items in (others or {}).items():
        players[seat] = replace(
            players[seat],
            discards=tuple(
                Discard(tile=_tile(spec), tsumogiri=False, order=order, called_by=None)
                for order, spec in items
            ),
        )
    if second_riichi:
        players[3] = replace(
            players[3],
            discards=_discards_of(("3z", "4z"), start=10),
            riichi=RiichiState.ACCEPTED,
            riichi_declaration_discard_order=11,
        )
    return replace(context, players=tuple(players))


def _decision(context, actions=None):
    return DecisionContext(
        input=context,
        legal_actions=_discards(context) if actions is None else actions,
    )


def _choose(policy, context):
    return policy.choose_action(_decision(context))


def _passed_by_seat2(spec, *, order=2):
    return {2: [(order, spec)]}


class ChangedBehaviorTest(unittest.TestCase):
    def test_passed_only_tile_is_discarded_at_one_shanten(self):
        context = _situation(ONE_SHANTEN, others=_passed_by_seat2("9s"))

        self.assertEqual(_choose(CHAMPION, context).tile, _tile("1m"))
        self.assertEqual(_choose(CANDIDATE, context).tile, _tile("9s"))

    def test_passed_only_tile_is_discarded_at_two_shanten(self):
        context = _situation(TWO_SHANTEN, others=_passed_by_seat2("1m"))

        self.assertEqual(_choose(CHAMPION, context).tile, _tile("5z"))
        self.assertEqual(_choose(CANDIDATE, context).tile, _tile("1m"))

    def test_all_last_top_does_not_prefilter_by_danger(self):
        one = _situation(ONE_SHANTEN, others=_passed_by_seat2("9s"), **ALL_LAST_TOP)
        two = _situation(TWO_SHANTEN, others=_passed_by_seat2("1m"), **ALL_LAST_TOP)

        self.assertEqual(_choose(CHAMPION, one).tile, _tile("1m"))
        self.assertEqual(_choose(CANDIDATE, one).tile, _tile("9s"))
        self.assertEqual(_choose(CHAMPION, two).tile, _tile("5z"))
        self.assertEqual(_choose(CANDIDATE, two).tile, _tile("1m"))

    def test_passed_tile_widens_the_river_genbutsu_candidates(self):
        context = _situation(
            ONE_SHANTEN, river=("3z", "8p"), others=_passed_by_seat2("9s")
        )

        self.assertEqual(_choose(CHAMPION, context).tile, _tile("8p"))
        self.assertEqual(_choose(CANDIDATE, context).tile, _tile("9s"))

    def test_tile_passed_by_the_own_seat_counts(self):
        context = _situation(ONE_SHANTEN, others={0: [(2, "9s")]})

        self.assertEqual(_choose(CANDIDATE, context).tile, _tile("9s"))

    def test_tile_before_the_declaration_is_not_safe(self):
        context = _situation(
            ONE_SHANTEN,
            declaration=3,
            river=("3z", "4z", "5z", "6z"),
            others=_passed_by_seat2("9s", order=2),
        )

        self.assertEqual(_choose(CANDIDATE, context), _choose(CHAMPION, context))
        self.assertEqual(_choose(CANDIDATE, context).tile, _tile("1m"))

    def test_chosen_action_is_one_of_the_confirmed_safe_actions(self):
        context = _situation(ONE_SHANTEN, others={2: [(2, "9s"), (3, "1m"), (4, "6p")]})

        chosen = _choose(CANDIDATE, context)

        self.assertIn(chosen.tile, (_tile("9s"), _tile("1m")))

    def test_original_action_identity_and_red_five_survive(self):
        context = _situation("34m678p68p234s05s9s1m", others=_passed_by_seat2("5s"))
        red = DiscardAction(actor=Seat.SEAT_0, tile=_tile("0s"), tsumogiri=False)
        normal = DiscardAction(actor=Seat.SEAT_0, tile=_tile("5s"), tsumogiri=False)
        rest = tuple(
            action
            for action in _discards(context)
            if action.tile.tile_type != _tile("5s").tile_type
        )
        actions = (red, normal, *rest)

        chosen = CANDIDATE.choose_action(_decision(context, actions))

        self.assertIn(chosen, (red, normal))
        self.assertTrue(any(chosen is action for action in actions))


class UnchangedBehaviorTest(unittest.TestCase):
    def assertSameAction(self, context):
        decision = _decision(context)
        self.assertEqual(
            CANDIDATE.choose_action(decision), CHAMPION.choose_action(decision)
        )

    def assertSameAsChampion(self, context):
        """Championの経路へ委譲する入力では、analysisも含めて同じ`PolicyDecision`を返す。"""
        self.assertSameAction(context)
        self.assertEqual(
            CANDIDATE._decide_discard(context, _discards(context)),
            CHAMPION._decide_discard(context, _discards(context)),
        )

    def test_no_confirmed_safe_tile_among_legal_discards(self):
        self.assertSameAsChampion(
            _situation(ONE_SHANTEN, others=_passed_by_seat2("7z"))
        )
        self.assertSameAsChampion(
            _situation(TWO_SHANTEN, others=_passed_by_seat2("7z"))
        )

    def test_river_genbutsu_only_matches_champion(self):
        # 河の現物が合法打牌にあるときは確定安全牌の経路を通るが、選ぶ打牌はChampionと同じ。
        self.assertSameAction(_situation(ONE_SHANTEN, river=("3z", "8p")))

    def test_unknown_declaration_position_uses_the_river_only(self):
        self.assertSameAsChampion(
            _situation(ONE_SHANTEN, declaration=None, others=_passed_by_seat2("9s"))
        )
        self.assertSameAsChampion(
            _situation(TWO_SHANTEN, declaration=None, others=_passed_by_seat2("1m"))
        )
        self.assertSameAction(
            _situation(
                ONE_SHANTEN,
                river=("3z", "8p"),
                declaration=None,
                others=_passed_by_seat2("9s"),
            )
        )

    def test_tenpai_keeping_push_matches_champion(self):
        self.assertSameAsChampion(
            _situation(TENPAI_KEEPING, others=_passed_by_seat2("5z"))
        )

    def test_multiple_riichi_matches_champion(self):
        self.assertSameAsChampion(
            _situation(ONE_SHANTEN, others=_passed_by_seat2("9s"), second_riichi=True)
        )

    def test_no_riichi_matches_champion(self):
        self.assertSameAsChampion(
            _situation(ONE_SHANTEN, riichi=False, others=_passed_by_seat2("9s"))
        )

    def test_call_decision_and_call_suppression_match_champion(self):
        context = _situation("34m678p6p234s55s11z", others=_passed_by_seat2("9s"))
        context = replace(context, own_hand=replace(context.own_hand, drawn_tile=None))
        decision = DecisionContext(input=context, legal_actions=(PASS, _pon("1z")))

        self.assertEqual(
            CANDIDATE.choose_action(decision), CHAMPION.choose_action(decision)
        )

    def test_only_the_discard_decision_is_overridden(self):
        overridden = set(vars(PassedTileGenbutsuPlacementAwareSpeedCallPolicy))
        self.assertLessEqual(
            overridden
            - {
                "__module__",
                "__doc__",
                "__annotations__",
                "__firstlineno__",
                "__static_attributes__",
                "__qualname__",
            },
            {"RULE_ASSUMPTION", "_decide_discard"},
        )

    def test_rule_assumption_is_declared(self):
        self.assertEqual(
            CANDIDATE.RULE_ASSUMPTION, "riichi-missed-win-furiten-and-fixed-waits-v1"
        )


class InvalidInputTest(unittest.TestCase):
    def test_hand_not_before_discard_is_rejected_when_the_foundation_is_used(self):
        context = _situation(ONE_SHANTEN, others=_passed_by_seat2("9s"))
        context = replace(
            context,
            own_hand=replace(
                context.own_hand,
                concealed_tiles=_hand(ONE_SHANTEN)[:-1],
                drawn_tile=None,
            ),
        )
        actions = _discards(context)

        with self.assertRaises(ValueError):
            CANDIDATE._decide_discard(context, actions)


class BoundaryTest(unittest.TestCase):
    def test_static_import_check_passes(self):
        report = check_candidate_imports(_MODULE)

        self.assertTrue(report.passed, msg=report.violations)

    def test_purity_check_passes_across_processes(self):
        contexts = (
            _situation(ONE_SHANTEN, others=_passed_by_seat2("9s")),
            _situation(TWO_SHANTEN, others=_passed_by_seat2("1m")),
            _situation(ONE_SHANTEN, river=("3z", "8p"), others=_passed_by_seat2("9s")),
            _situation(ONE_SHANTEN, declaration=None, others=_passed_by_seat2("9s")),
            _situation(TENPAI_KEEPING, others=_passed_by_seat2("5z")),
        )
        decisions = tuple(_decision(context) for context in contexts)

        report = check_candidate_purity(_FACTORY, decisions)

        self.assertTrue(report.passed, msg=(report.mismatches, report.errors))
        self.assertEqual(report.decision_count, len(decisions))


if __name__ == "__main__":
    unittest.main()
