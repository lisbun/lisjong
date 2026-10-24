"""Issue #350: 鳴き採用・直後打牌の限定解析API（call-value-shadow-v1）。

#348の局面fixture（`test_speed_call_post_call_value_diagnostic`）を、
production-ownedな`lisjong.policies.speed_call_analysis`で再現する。
代替の選択規則、速度損失、入力不整合の拒否、本番Action不変、情報境界を固定する。
"""

import unittest
from dataclasses import replace

from test_speed_call_post_call_value_diagnostic import (
    ALL_LAST_FAR_LAST,
    ALL_LAST_TOP,
    CHI_2M,
    CLOSED_TWO_SHANTEN,
    _chi_meld,
    _pon,
    _pon_meld,
    _post_call_decision,
    _pre_call_decision,
    _river,
    _tile,
)

from lisjong.candidate_boundary import check_candidate_imports
from lisjong.policies import PlacementAwareSpeedCallPolicy
from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    _falling_factorial,
)
from lisjong.policies.placement_aware_speed_call import GameSituationMode
from lisjong.policies.speed_call_analysis import (
    CallDecisionRoute,
    PostCallClassification,
    PostCallDiscardAnalysis,
    PostCallDiscardFacts,
    SpeedCallAnalysisError,
    SpeedCallDecisionAnalysis,
    _alternative,
    _speed_loss,
    analyze_post_call_discard,
    analyze_speed_call_decision,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    YakuhaiCallPolicyError,
)
from lisjong.policy_contract.action import DiscardAction, PassAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.seat import Seat
from lisjong.structural_efficiency import discard_action_sort_key

_SELF = Seat.SEAT_0
NORMAL = GameSituationMode.NORMAL
SPEED = GameSituationMode.ALL_LAST_TOP_SPEED
VALUE = GameSituationMode.ALL_LAST_LAST_VALUE
CONSISTENT = PostCallClassification.CONSISTENT
BELOW_MAX = PostCallClassification.BELOW_MAX
BELOW_THRESHOLD = PostCallClassification.BELOW_THRESHOLD

# 2mチー後 55m 678m 22p 56p 48s。喰い替えで5mは切れない。打8sは閾値割れ。
KUIKAE_HAND = "34m55m678m22p56p48s"


def _discard(tile: str) -> DiscardAction:
    return DiscardAction(actor=_SELF, tile=_tile(tile), tsumogiri=False)


def _reversed(decision: DecisionContext) -> DecisionContext:
    return replace(decision, legal_actions=tuple(reversed(decision.legal_actions)))


class _AnalysisCase(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PlacementAwareSpeedCallPolicy()

    def _call(self, pre: DecisionContext) -> SpeedCallDecisionAnalysis:
        return analyze_speed_call_decision(pre, self.policy.choose_action(pre))

    def _post(
        self, pre: DecisionContext
    ) -> tuple[DecisionContext, PostCallDiscardAnalysis]:
        call_analysis = self._call(pre)
        self.assertIs(call_analysis.route, CallDecisionRoute.SPEED_CALL)
        assert call_analysis.call is not None
        post = _post_call_decision(pre.input, call_analysis.call)
        analysis = analyze_post_call_discard(
            post, call_analysis, self.policy.choose_action(post)
        )  # type: ignore[arg-type]
        return post, analysis

    def _analysis(self, pre: DecisionContext) -> PostCallDiscardAnalysis:
        return self._post(pre)[1]


class CallDecisionAnalysisTest(_AnalysisCase):
    def test_primary_target_speed_call(self) -> None:
        analysis = self._call(
            _pre_call_decision(CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s")
        )
        self.assertIs(analysis.route, CallDecisionRoute.SPEED_CALL)
        self.assertEqual(analysis.call, CHI_2M)
        self.assertEqual(analysis.action, CHI_2M)
        self.assertIs(analysis.mode, NORMAL)
        self.assertTrue(analysis.closed_before_call)
        self.assertEqual(
            (analysis.route_shanten, analysis.value, analysis.minimum_value), (1, 2, 2)
        )
        self.assertTrue(analysis.is_primary_target)

    def test_speed_and_value_modes_are_not_primary(self) -> None:
        speed = self._call(
            _pre_call_decision(
                CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s", **ALL_LAST_TOP
            )
        )
        self.assertIs(speed.mode, SPEED)
        self.assertEqual((speed.route_shanten, speed.minimum_value), (1, 0))
        self.assertFalse(speed.is_primary_target)
        value = self._call(
            _pre_call_decision(
                CLOSED_TWO_SHANTEN,
                (CHI_2M,),
                dora_indicators="7s4p",
                **ALL_LAST_FAR_LAST,
            )
        )
        self.assertIs(value.mode, VALUE)
        self.assertEqual((value.value, value.minimum_value), (3, 3))
        self.assertFalse(value.is_primary_target)

    def test_closed_tenpai_and_open_hand_calls_are_not_primary(self) -> None:
        tenpai = self._call(
            _pre_call_decision("34m678m567p22p468s", (CHI_2M,), dora_indicators="7s")
        )
        self.assertEqual((tenpai.route_shanten, tenpai.minimum_value), (0, 0))
        self.assertFalse(tenpai.is_primary_target)
        open_hand = self._call(
            _pre_call_decision(
                "678m22p55p468s",
                (_pon("5p"),),
                own_melds=(_chi_meld("234m"),),
                rivers={Seat.SEAT_3: _river(called_by_self="2m")},
                dora_indicators="7s",
            )
        )
        self.assertFalse(open_hand.closed_before_call)
        self.assertEqual(open_hand.minimum_value, 0)
        self.assertFalse(open_hand.is_primary_target)

    def test_non_activation_is_parent_pass(self) -> None:
        for pre in (
            _pre_call_decision(
                CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s", **ALL_LAST_FAR_LAST
            ),
            _pre_call_decision(
                CLOSED_TWO_SHANTEN,
                (CHI_2M,),
                dora_indicators="7s",
                riichi_discards="1z9s",
            ),
            _pre_call_decision("34m678m567p22p678s", (CHI_2M,), dora_indicators="7s"),
        ):
            with self.subTest(pre=pre):
                analysis = self._call(pre)
                self.assertIs(analysis.route, CallDecisionRoute.PARENT_PASS)
                self.assertIsInstance(analysis.action, PassAction)
                self.assertIsNone(analysis.call)
                self.assertIsNone(analysis.minimum_value)
                self.assertFalse(analysis.is_primary_target)

    def test_yakuhai_parent_call_is_not_a_speed_call(self) -> None:
        # 5zは役牌。parentが役牌ポンを選ぶ経路はスピード鳴きに混ぜない。
        pre = _pre_call_decision("55z678m56p22p468s9m", (_pon("5z"),))
        action = self.policy.choose_action(pre)
        self.assertEqual(action, _pon("5z"))
        analysis = analyze_speed_call_decision(pre, action)
        self.assertIs(analysis.route, CallDecisionRoute.PARENT_DECISION)
        self.assertIsNone(analysis.call)
        self.assertFalse(analysis.is_primary_target)

    def test_open_yakuhai_hand_stays_on_parent_route(self) -> None:
        pre = _pre_call_decision(
            "678m56p22p468s",
            (_pon("2p"),),
            own_melds=(_pon_meld("5z"),),
            rivers={Seat.SEAT_2: _river(called_by_self="5z")},
            dora_indicators="7s",
        )
        analysis = self._call(pre)
        self.assertIsNot(analysis.route, CallDecisionRoute.SPEED_CALL)

    def test_selected_action_must_be_the_champion_decision(self) -> None:
        pre = _pre_call_decision(CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s")
        with self.assertRaises(SpeedCallAnalysisError):
            analyze_speed_call_decision(pre, PassAction(actor=_SELF))
        with self.assertRaises(SpeedCallAnalysisError):
            analyze_speed_call_decision(pre, _pon("2m"))

    def test_parent_fail_closed_is_propagated(self) -> None:
        pre = _pre_call_decision(
            CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s", legal_pass=False
        )
        with self.assertRaises(YakuhaiCallPolicyError):
            analyze_speed_call_decision(pre, CHI_2M)

    def test_inconsistent_call_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SpeedCallDecisionAnalysis(
                route=CallDecisionRoute.PARENT_PASS,
                action=PassAction(actor=_SELF),
                mode=NORMAL,
                closed_before_call=True,
                call=CHI_2M,
                route_shanten=1,
                value=2,
                minimum_value=2,
            )
        with self.assertRaises(ValueError):
            SpeedCallDecisionAnalysis(
                route=CallDecisionRoute.SPEED_CALL,
                action=CHI_2M,
                mode=NORMAL,
                closed_before_call=True,
                call=CHI_2M,
                route_shanten=1,
                value=None,
                minimum_value=2,
            )


class ReproducedFixtureTest(_AnalysisCase):
    """#348のREPRODUCED-FIXTUREを新APIで再現する。"""

    def test_primary_below_threshold_with_alternative(self) -> None:
        analysis = self._analysis(
            _pre_call_decision(CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s")
        )
        self.assertIs(analysis.classification, BELOW_THRESHOLD)
        self.assertTrue(analysis.call_analysis.is_primary_target)
        self.assertEqual(analysis.discard, _discard("8s"))
        self.assertEqual(analysis.alternative, _discard("4s"))
        chosen, alternative = analysis.chosen, analysis.alternative_facts
        assert alternative is not None
        self.assertEqual(
            (chosen.route_shanten, chosen.value, chosen.completion_mass, chosen.ukeire),
            (1, 1, 45980, 24),
        )
        self.assertEqual(
            (
                alternative.route_shanten,
                alternative.value,
                alternative.completion_mass,
                alternative.ukeire,
            ),
            (1, 2, 40176, 23),
        )
        self.assertEqual(analysis.horizon, DEFAULT_HORIZON)
        self.assertEqual(analysis.hidden_tile_count, 121)
        self.assertEqual(
            analysis.sequence_denominator, _falling_factorial(121, DEFAULT_HORIZON)
        )
        loss = analysis.speed_loss
        assert loss is not None
        self.assertEqual(
            (loss.mass_difference, loss.relative_numerator, loss.relative_denominator),
            (45980 - 40176, 45980 - 40176, 45980),
        )
        self.assertFalse(loss.zero_mass)

    def test_kuikae_fixture_below_threshold(self) -> None:
        analysis = self._analysis(
            _pre_call_decision(KUIKAE_HAND, (CHI_2M,), dora_indicators="7s")
        )
        self.assertNotIn(
            _tile("5m").tile_type,
            {facts.action.tile.tile_type for facts in analysis.candidates},
        )
        self.assertIs(analysis.classification, BELOW_THRESHOLD)
        self.assertTrue(analysis.call_analysis.is_primary_target)
        self.assertEqual(analysis.discard, _discard("8s"))
        self.assertEqual(analysis.alternative, _discard("4s"))
        loss = analysis.speed_loss
        assert loss is not None
        self.assertEqual(
            (loss.relative_numerator, loss.relative_denominator), (320, 24576)
        )

    def test_below_max_has_no_alternative(self) -> None:
        cases = {
            "closed_tenpai": (
                _pre_call_decision(
                    "34m678m567p22p468s", (CHI_2M,), dora_indicators="7s"
                ),
                "8s",
            ),
            "open_pon": (
                _pre_call_decision(
                    "678m22p55p468s",
                    (_pon("5p"),),
                    own_melds=(_chi_meld("234m"),),
                    rivers={Seat.SEAT_3: _river(called_by_self="2m")},
                    dora_indicators="7s",
                ),
                "8s",
            ),
            "chinitsu": (
                _pre_call_decision("66m111m345m789m9m1z", (_pon("6m"),)),
                "9m",
            ),
            "route_split": (
                _pre_call_decision(
                    "33m456m678m567m4p1z",
                    (_pon("3m"),),
                    rivers={Seat.SEAT_1: _river(plain="11z")},
                ),
                "1z",
            ),
            "speed_mode": (
                _pre_call_decision(
                    CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s", **ALL_LAST_TOP
                ),
                "8s",
            ),
        }
        for name, (pre, discard) in cases.items():
            with self.subTest(name):
                analysis = self._analysis(pre)
                self.assertIs(analysis.classification, BELOW_MAX)
                self.assertEqual(analysis.discard, _discard(discard))
                self.assertIsNone(analysis.alternative)
                self.assertIsNone(analysis.alternative_facts)
                self.assertIsNone(analysis.speed_loss)
                self.assertFalse(analysis.call_analysis.is_primary_target)

    def test_value_mode_below_threshold_is_separate_from_primary(self) -> None:
        analysis = self._analysis(
            _pre_call_decision(
                CLOSED_TWO_SHANTEN,
                (CHI_2M,),
                dora_indicators="7s4p",
                **ALL_LAST_FAR_LAST,
            )
        )
        self.assertIs(analysis.classification, BELOW_THRESHOLD)
        self.assertIs(analysis.call_analysis.mode, VALUE)
        self.assertFalse(analysis.call_analysis.is_primary_target)
        self.assertEqual(analysis.alternative, _discard("4s"))
        self.assertEqual(analysis.chosen.value, 2)


class ConsistentFixtureTest(_AnalysisCase):
    def test_consistent_fixtures(self) -> None:
        cases = (
            (
                _pre_call_decision(
                    "34m678m567p22p468s", (CHI_2M,), dora_indicators="3s"
                ),
                "8s",
            ),
            (
                _pre_call_decision(
                    "34m67m567p22p3468s", (CHI_2M,), dora_indicators="7s"
                ),
                "6s",
            ),
            (
                _pre_call_decision(
                    "1235799m112233z", (_pon("9m"),), dora_indicators="6m"
                ),
                "1z",
            ),
            (_pre_call_decision("33m456m678m567m4p1z", (_pon("3m"),)), "4p"),
        )
        for pre, discard in cases:
            with self.subTest(discard=discard):
                analysis = self._analysis(pre)
                self.assertIs(analysis.classification, CONSISTENT)
                self.assertEqual(analysis.discard, _discard(discard))
                self.assertEqual(analysis.chosen.value, analysis.call_analysis.value)
                self.assertIsNone(analysis.alternative)
                self.assertIsNone(analysis.speed_loss)

    def test_red_five_identity_is_kept_distinct(self) -> None:
        analysis = self._analysis(_pre_call_decision("34m678m56p22p468s0p", (CHI_2M,)))
        actions = [facts.action for facts in analysis.candidates]
        self.assertIn(_discard("5p"), actions)
        self.assertIn(_discard("0p"), actions)
        self.assertIs(analysis.classification, CONSISTENT)
        self.assertEqual(analysis.discard, _discard("5p"))


class DeterminismAndInvarianceTest(_AnalysisCase):
    def test_candidates_are_canonical_and_order_independent(self) -> None:
        pre = _pre_call_decision(CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s")
        post, analysis = self._post(pre)
        self.assertEqual(
            [facts.action for facts in analysis.candidates],
            sorted(post.legal_actions, key=discard_action_sort_key),  # type: ignore[arg-type]
        )
        reversed_call = analyze_speed_call_decision(
            _reversed(pre), self.policy.choose_action(_reversed(pre))
        )
        self.assertEqual(reversed_call, analysis.call_analysis)
        reversed_post = analyze_post_call_discard(
            _reversed(post), reversed_call, analysis.discard
        )
        self.assertEqual(reversed_post, analysis)

    def test_masses_share_denominator_bounds(self) -> None:
        analysis = self._analysis(
            _pre_call_decision(CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s")
        )
        for facts in analysis.candidates:
            self.assertGreaterEqual(facts.completion_mass, 0)
            self.assertLessEqual(facts.completion_mass, analysis.sequence_denominator)

    def test_champion_actions_unchanged_by_analysis(self) -> None:
        pre = _pre_call_decision(CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s")
        call_before = self.policy.choose_action(pre)
        post = _post_call_decision(pre.input, CHI_2M)
        discard_before = self.policy.choose_action(post)

        first = analyze_post_call_discard(
            post,
            analyze_speed_call_decision(pre, call_before),
            discard_before,  # type: ignore[arg-type]
        )
        second = analyze_post_call_discard(
            post,
            analyze_speed_call_decision(pre, call_before),
            discard_before,  # type: ignore[arg-type]
        )
        self.assertEqual(first, second)
        self.assertEqual(self.policy.choose_action(pre), call_before)
        self.assertEqual(self.policy.choose_action(post), discard_before)
        self.assertEqual(
            PlacementAwareSpeedCallPolicy().choose_action(post), discard_before
        )

    def test_analysis_values_are_immutable(self) -> None:
        analysis = self._analysis(
            _pre_call_decision(CLOSED_TWO_SHANTEN, (CHI_2M,), dora_indicators="7s")
        )
        self.assertIsInstance(analysis.candidates, tuple)
        with self.assertRaises(AttributeError):
            analysis.alternative = None  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            analysis.chosen.value = 5  # type: ignore[misc]

    def test_module_passes_information_boundary_check(self) -> None:
        report = check_candidate_imports("lisjong.policies.speed_call_analysis")
        self.assertTrue(report.passed, msg=report.violations)


class AlternativeRuleTest(unittest.TestCase):
    """代替の決定的順位（mass → value → 受入 → sort key）とSpeedLoss。"""

    def _facts(
        self, tile: str, *, shanten: int = 1, value: int = 2, mass: int, ukeire: int
    ) -> PostCallDiscardFacts:
        return PostCallDiscardFacts(
            action=_discard(tile),
            route_shanten=shanten,
            value=value,
            completion_mass=mass,
            ukeire=ukeire,
        )

    def _assert_choice(
        self, facts: tuple[PostCallDiscardFacts, ...], expected: str
    ) -> None:
        for candidates in (facts, tuple(reversed(facts))):
            self.assertEqual(_alternative(candidates, 1, 2).action, _discard(expected))

    def test_completion_mass_first(self) -> None:
        self._assert_choice(
            (
                self._facts("4s", value=5, mass=10, ukeire=30),
                self._facts("6s", value=2, mass=11, ukeire=1),
            ),
            "6s",
        )

    def test_value_then_ukeire(self) -> None:
        self._assert_choice(
            (
                self._facts("4s", value=2, mass=10, ukeire=30),
                self._facts("6s", value=3, mass=10, ukeire=1),
            ),
            "6s",
        )
        self._assert_choice(
            (
                self._facts("4s", mass=10, ukeire=3),
                self._facts("6s", mass=10, ukeire=4),
            ),
            "6s",
        )

    def test_exact_tie_uses_sort_key_with_red_identity(self) -> None:
        facts = (
            self._facts("0p", mass=10, ukeire=3),
            self._facts("5p", mass=10, ukeire=3),
            self._facts("6p", mass=10, ukeire=3),
        )
        expected = min(
            (item.action for item in facts), key=discard_action_sort_key
        ).tile
        self.assertEqual(expected, _tile("5p"))
        self._assert_choice(facts, "5p")

    def test_excludes_other_route_shanten_and_below_threshold(self) -> None:
        self._assert_choice(
            (
                self._facts("1s", shanten=2, mass=99, ukeire=40),
                self._facts("2s", value=1, mass=98, ukeire=40),
                self._facts("4s", mass=1, ukeire=1),
            ),
            "4s",
        )
        with self.assertRaises(SpeedCallAnalysisError):
            _alternative((self._facts("2s", value=1, mass=9, ukeire=3),), 1, 2)

    def test_speed_loss(self) -> None:
        loss = _speed_loss(10, 10)
        self.assertEqual((loss.relative_numerator, loss.relative_denominator), (0, 10))
        self.assertFalse(loss.zero_mass)
        zero = _speed_loss(0, 0)
        self.assertEqual(
            (zero.mass_difference, zero.relative_numerator, zero.relative_denominator),
            (0, 0, 1),
        )
        self.assertTrue(zero.zero_mass)
        for chosen, alternative in ((0, 1), (5, 6)):
            with self.subTest(chosen=chosen, alternative=alternative):
                with self.assertRaises(SpeedCallAnalysisError):
                    _speed_loss(chosen, alternative)


class InvalidPostContextTest(_AnalysisCase):
    """call成立後contextの不整合はfail closedする。"""

    def setUp(self) -> None:
        super().setUp()
        self.pre = _pre_call_decision(KUIKAE_HAND, (CHI_2M,), dora_indicators="7s")
        self.call_analysis = self._call(self.pre)
        self.post = _post_call_decision(self.pre.input, CHI_2M)
        self.discard = self.policy.choose_action(self.post)

    def _assert_rejected(
        self,
        post: DecisionContext,
        discard: DiscardAction | None = None,
        call_analysis: SpeedCallDecisionAnalysis | None = None,
    ) -> None:
        with self.assertRaises(SpeedCallAnalysisError):
            analyze_post_call_discard(
                post,
                call_analysis or self.call_analysis,
                discard or self.discard,  # type: ignore[arg-type]
            )

    def _with_input(self, **changes: object) -> DecisionContext:
        return replace(self.post, input=replace(self.post.input, **changes))

    def test_valid_context_is_accepted(self) -> None:
        analyze_post_call_discard(self.post, self.call_analysis, self.discard)  # type: ignore[arg-type]

    def test_requires_speed_call_analysis(self) -> None:
        non_speed = self._call(
            _pre_call_decision("34m678m567p22p678s", (CHI_2M,), dora_indicators="7s")
        )
        self._assert_rejected(self.post, call_analysis=non_speed)

    def test_discard_must_be_champion_decision_and_legal(self) -> None:
        self._assert_rejected(self.post, discard=_discard("4s"))
        self._assert_rejected(self.post, discard=_discard("5m"))

    def test_kuikae_forbidden_discard_in_legal_actions(self) -> None:
        self._assert_rejected(
            replace(
                self.post,
                legal_actions=(*self.post.legal_actions, _discard("5m")),
            )
        )

    def test_missing_or_tsumogiri_legal_discards(self) -> None:
        legal = self.post.legal_actions
        missing = tuple(action for action in legal if action != _discard("6m"))
        self._assert_rejected(replace(self.post, legal_actions=missing))
        tsumogiri = tuple(
            replace(action, tsumogiri=True)  # type: ignore[type-var]
            for action in legal
        )
        self._assert_rejected(replace(self.post, legal_actions=tsumogiri))
        self._assert_rejected(
            replace(self.post, legal_actions=(*legal, PassAction(actor=_SELF)))
        )

    def test_drawn_tile_is_rejected(self) -> None:
        own_hand = self.post.input.own_hand
        concealed = own_hand.concealed_tiles
        self._assert_rejected(
            self._with_input(own_hand=replace(own_hand, drawn_tile=concealed[-1]))
        )

    def test_meld_and_river_must_match_call(self) -> None:
        players = list(self.post.input.players)
        own = players[int(_SELF)]
        meld = own.melds[-1]
        wrong_meld = list(players)
        wrong_meld[int(_SELF)] = replace(
            own, melds=(replace(meld, from_seat=Seat.SEAT_2),)
        )
        self._assert_rejected(self._with_input(players=tuple(wrong_meld)))

        target = players[int(Seat.SEAT_3)]
        last = target.discards[-1]
        unclaimed = list(players)
        unclaimed[int(Seat.SEAT_3)] = replace(
            target, discards=(*target.discards[:-1], replace(last, called_by=None))
        )
        self._assert_rejected(self._with_input(players=tuple(unclaimed)))

    def test_situation_mode_change_is_rejected(self) -> None:
        players = tuple(
            replace(player, score=score)
            for player, score in zip(
                self.post.input.players, (40000, 20000, 20000, 20000), strict=True
            )
        )
        round_state = replace(
            self.post.input.round,
            round_wind=ALL_LAST_TOP["round_wind"],
            hand_number=ALL_LAST_TOP["hand_number"],
        )
        self._assert_rejected(self._with_input(players=players, round=round_state))


if __name__ == "__main__":
    unittest.main()
