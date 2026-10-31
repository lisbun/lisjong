"""Issue #363: 門前の最初のPonだけPassへ替えるD arm wrapperと分岐点の共有判定（P1）。"""

import unittest

from test_placement_aware_speed_call_policy import (
    _chi_meld,
    _input,
    _pon,
    _pon_meld,
    _tile,
)

from lisjong.candidate_boundary import check_candidate_imports
from lisjong.policies import PlacementAwareSpeedCallPolicy
from lisjong.policies.pon_first_pass_branch import (
    PON_FIRST_PASS_BRANCH_VERSION,
    PonFirstPassBranchError,
    PonFirstPassBranchPolicy,
    is_closed_for_pon_branch,
    is_pon_branch_candidate,
)
from lisjong.policies.riichi_dama_branch import (
    RIICHI_FIRST_DAMA_BRANCH_VERSION,
    BranchStatus,
    RoundIdentity,
)
from lisjong.policy_contract.action import (
    ChiAction,
    DiscardAction,
    PassAction,
    PonAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.wind import Wind

SELF = Seat.SEAT_0
PASS = PassAction(actor=SELF)
CHAMPION = PlacementAwareSpeedCallPolicy()
TARGET = RoundIdentity(Wind.EAST, 1, 0)
OTHER_ROUND = {"round_wind": Wind.EAST, "hand_number": 2}

# 1235799m 112233z: 9mのPonでHonitsuの1向聴になり、ChampionはPonを選ぶ（既存test）。
HONITSU = "1235799m112233z"
PON_9M = _pon("9m")


def _red_pon(*, red: bool) -> PonAction:
    """5pのPon。消費牌に赤5pを含むかどうかだけが違う候補。"""
    plain = _tile("5p")
    return PonAction(
        actor=SELF,
        target=Seat.SEAT_2,
        called_tile=plain,
        consumed_tiles=(_tile("0p") if red else plain, plain),
    )


def _decision(concealed=HONITSU, actions=(PASS, PON_9M), **kwargs) -> DecisionContext:
    return DecisionContext(input=_input(concealed, **kwargs), legal_actions=actions)


def _ankan_meld(tile: str) -> PublicMeld:
    called = _tile(tile)
    return PublicMeld(
        kind=MeldKind.ANKAN, tiles=(called,) * 4, from_seat=None, called_tile=None
    )


def _discard(tile: str) -> DiscardAction:
    return DiscardAction(actor=SELF, tile=_tile(tile), tsumogiri=False)


class _Scripted:
    """判断ごとに決めた行動を返すinner（呼び出しを記録する）。"""

    def __init__(self, actions) -> None:
        self._actions = list(actions)
        self.calls: list[DecisionContext] = []

    def choose_action(self, decision: DecisionContext):
        self.calls.append(decision)
        return self._actions.pop(0)


class ClosedPredicateTest(unittest.TestCase):
    def test_no_meld_and_ankan_only_are_closed(self) -> None:
        self.assertTrue(is_closed_for_pon_branch(_input(HONITSU)))
        self.assertTrue(
            is_closed_for_pon_branch(
                _input("1235799m1122z", own_melds=(_ankan_meld("3z"),))
            )
        )

    def test_open_melds_are_not_closed(self) -> None:
        kakan_tile = _tile("4z")
        melds = {
            "chi": _chi_meld("234s"),
            "pon": _pon_meld("4z"),
            "daiminkan": PublicMeld(
                kind=MeldKind.DAIMINKAN,
                tiles=(kakan_tile,) * 4,
                from_seat=Seat.SEAT_2,
                called_tile=kakan_tile,
            ),
            "kakan": PublicMeld(
                kind=MeldKind.KAKAN,
                tiles=(kakan_tile,) * 4,
                from_seat=Seat.SEAT_2,
                called_tile=kakan_tile,
            ),
        }
        for name, meld in melds.items():
            with self.subTest(meld=name):
                self.assertFalse(
                    is_closed_for_pon_branch(_input("1235799m1z", own_melds=(meld,)))
                )

    def test_predicate_types(self) -> None:
        with self.assertRaises(TypeError):
            is_closed_for_pon_branch(object())
        with self.assertRaises(TypeError):
            is_pon_branch_candidate(object(), PON_9M)


class CandidatePredicateTest(unittest.TestCase):
    def test_closed_legal_own_pon_is_candidate(self) -> None:
        self.assertTrue(is_pon_branch_candidate(_decision(), PON_9M))

    def test_non_candidates(self) -> None:
        open_hand = _decision("1235799m1122z", own_melds=(_pon_meld("3z"),))
        not_legal = _decision(actions=(PASS,))
        chi = ChiAction(
            actor=SELF,
            target=Seat.SEAT_3,
            called_tile=_tile("4m"),
            consumed_tiles=(_tile("2m"), _tile("3m")),
        )
        cases = {
            "pass": (_decision(), PASS),
            "chi": (_decision(actions=(PASS, chi)), chi),
            "open hand": (open_hand, PON_9M),
            "not legal": (not_legal, PON_9M),
            "not an action": (_decision(), object()),
        }
        for name, (decision, action) in cases.items():
            with self.subTest(case=name):
                self.assertFalse(is_pon_branch_candidate(decision, action))

    def test_uses_only_the_decision_and_the_chosen_action(self) -> None:
        # 同じ判断・同じ行動なら、呼び出し順や他の判断に依存しない（状態を持たない）。
        decision = _decision()
        results = {is_pon_branch_candidate(decision, PON_9M) for _ in range(3)}
        self.assertEqual(results, {True})


class PonFirstPassBranchPolicyTest(unittest.TestCase):
    def _wrapper(self, inner=CHAMPION, pon=PON_9M) -> PonFirstPassBranchPolicy:
        return PonFirstPassBranchPolicy(inner, TARGET, pon)

    def test_identity_is_new(self) -> None:
        wrapper = self._wrapper()
        self.assertEqual(wrapper.version, PON_FIRST_PASS_BRANCH_VERSION)
        self.assertNotEqual(wrapper.version, RIICHI_FIRST_DAMA_BRANCH_VERSION)

    def test_champion_pons_and_branch_point_returns_pass(self) -> None:
        decision = _decision()
        self.assertEqual(CHAMPION.choose_action(decision), PON_9M)
        wrapper = self._wrapper()
        self.assertEqual(wrapper.choose_action(decision), PASS)
        self.assertEqual(wrapper.status, BranchStatus.BRANCHED)
        self.assertEqual(wrapper.calls_after_branch_count, 0)

    def test_other_rounds_pass_through_including_closed_pon(self) -> None:
        wrapper = self._wrapper()
        self.assertEqual(wrapper.choose_action(_decision(**OTHER_ROUND)), PON_9M)
        self.assertEqual(wrapper.status, BranchStatus.PENDING)

    def test_recorded_pon_must_match_exactly(self) -> None:
        plain, red = _red_pon(red=False), _red_pon(red=True)
        decision = _decision("1230559p112233z", actions=(PASS, plain, red))
        # innerが記録と同じ候補を選べば分岐する
        for chosen in (plain, red):
            wrapper = PonFirstPassBranchPolicy(_Scripted([chosen]), TARGET, chosen)
            with self.subTest(chosen=chosen):
                self.assertEqual(wrapper.choose_action(decision), PASS)
        # 赤の有無だけ違う候補、別の牌のPonはエラー
        for recorded, chosen in ((plain, red), (red, plain), (PON_9M, plain)):
            wrapper = PonFirstPassBranchPolicy(_Scripted([chosen]), TARGET, recorded)
            with (
                self.subTest(recorded=recorded, chosen=chosen),
                self.assertRaises(PonFirstPassBranchError),
            ):
                wrapper.choose_action(decision)
            self.assertEqual(wrapper.status, BranchStatus.PENDING)

    def test_missing_pass_is_rejected(self) -> None:
        # DecisionContextは他席の行動を合法手に持てないので、自席のPassが無い場合だけを見る。
        wrapper = self._wrapper(inner=_Scripted([PON_9M]))
        with self.assertRaises(PonFirstPassBranchError):
            wrapper.choose_action(_decision(actions=(PON_9M,)))
        self.assertEqual(wrapper.status, BranchStatus.PENDING)

    def test_only_one_replacement_and_later_calls_are_counted(self) -> None:
        chi = ChiAction(
            actor=SELF,
            target=Seat.SEAT_3,
            called_tile=_tile("4m"),
            consumed_tiles=(_tile("2m"), _tile("3m")),
        )
        inner = _Scripted([PON_9M, PON_9M, _discard("1m"), chi])
        wrapper = self._wrapper(inner=inner)
        decision = _decision(actions=(PASS, PON_9M, chi))
        returned = [wrapper.choose_action(decision) for _ in range(4)]
        self.assertEqual(returned, [PASS, PON_9M, _discard("1m"), chi])
        self.assertEqual(wrapper.calls_after_branch_count, 2)
        # innerへは元の判断をそのまま渡し、作り直さない
        for call in inner.calls:
            self.assertIs(call, decision)

    def test_pon_from_open_hand_is_not_a_branch_point(self) -> None:
        open_decision = _decision(
            "1235799m1122z", actions=(PASS, PON_9M), own_melds=(_pon_meld("3z"),)
        )
        wrapper = self._wrapper(inner=_Scripted([PON_9M, PON_9M]))
        self.assertEqual(wrapper.choose_action(open_decision), PON_9M)
        self.assertEqual(wrapper.status, BranchStatus.PENDING)

    def test_non_pon_actions_pass_through_before_branch(self) -> None:
        wrapper = self._wrapper(inner=_Scripted([PASS, _discard("1m")]))
        self.assertEqual(wrapper.choose_action(_decision()), PASS)
        self.assertEqual(wrapper.choose_action(_decision()), _discard("1m"))
        self.assertEqual(wrapper.status, BranchStatus.PENDING)

    def test_unreached_round_is_left_pending_then_missed(self) -> None:
        # 未到達は例外にせず状態として残し、INVALIDの判定は呼び出し側（Arena）が行う。
        wrapper = self._wrapper(inner=_Scripted([PASS, PASS]))
        wrapper.choose_action(_decision())
        self.assertEqual(wrapper.status, BranchStatus.PENDING)
        wrapper.choose_action(_decision(**OTHER_ROUND))
        self.assertEqual(wrapper.status, BranchStatus.MISSED)

    def test_target_round_appearing_again_is_rejected(self) -> None:
        for first in (PASS, PON_9M):
            wrapper = self._wrapper(inner=_Scripted([first, PASS, PON_9M]))
            wrapper.choose_action(_decision())
            wrapper.choose_action(_decision(**OTHER_ROUND))
            with (
                self.subTest(first=first),
                self.assertRaises(PonFirstPassBranchError),
            ):
                wrapper.choose_action(_decision())

    def test_calls_in_other_round_after_branch_are_not_counted(self) -> None:
        wrapper = self._wrapper(inner=_Scripted([PON_9M, PON_9M]))
        wrapper.choose_action(_decision())
        self.assertEqual(wrapper.choose_action(_decision(**OTHER_ROUND)), PON_9M)
        self.assertEqual(wrapper.calls_after_branch_count, 0)
        self.assertEqual(wrapper.status, BranchStatus.BRANCHED)

    def test_champion_behaviour_unchanged_outside_branch(self) -> None:
        decisions = [
            _decision("1235799m1122z", own_melds=(_pon_meld("3z"),)),
            _decision(actions=(PASS,)),
            _decision(**OTHER_ROUND),
        ]
        wrapper = self._wrapper()
        for decision in decisions:
            with self.subTest(decision=decision.legal_actions):
                self.assertEqual(
                    wrapper.choose_action(decision),
                    PlacementAwareSpeedCallPolicy().choose_action(decision),
                )
        self.assertEqual(wrapper.status, BranchStatus.MISSED)

    def test_input_and_constructor_types(self) -> None:
        with self.assertRaises(TypeError):
            PonFirstPassBranchPolicy(object(), TARGET, PON_9M)
        with self.assertRaises(TypeError):
            PonFirstPassBranchPolicy(CHAMPION, (Wind.EAST, 1, 0), PON_9M)
        with self.assertRaises(TypeError):
            PonFirstPassBranchPolicy(CHAMPION, TARGET, PASS)
        with self.assertRaises(TypeError):
            self._wrapper().choose_action(object())

    def test_red_identity_is_part_of_the_recorded_pon(self) -> None:
        self.assertNotEqual(_red_pon(red=False), _red_pon(red=True))


class BoundaryTest(unittest.TestCase):
    def test_module_passes_information_boundary_check(self) -> None:
        report = check_candidate_imports("lisjong.policies.pon_first_pass_branch")
        self.assertTrue(report.passed, msg=report.violations)


if __name__ == "__main__":
    unittest.main()
