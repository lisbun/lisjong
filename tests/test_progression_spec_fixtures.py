"""#337: 手計算由来のR5期待値。DP/referenceの出力から期待値を作らない。

distributionの9軸は打牌後の向聴数0..8。4枚の字牌手は確定3面子を表す。
字牌に順子はなく、残る1面子+雀頭について、2対子または刻子+単騎は0向聴、
対子+異なる2単騎は1向聴、異なる4単騎は2向聴となる。
fixtureの導出と3 callerの既存coverageはdocs/r5-spec-verification.mdを参照。
"""

import unittest
from unittest.mock import patch

import lisjong.policies.mechanism_riichi_defense_offensive_efficiency_diagnostic as diagnostic
import lisjong.policies.terminal_shanten_progression_mechanism_riichi_defense as progression
from tests.test_native_progression_backend import _native_evaluator
from tests.test_native_shanten_backend import _require_native
from tests.test_terminal_shanten_progression_mechanism_riichi_defense_policy import (
    _counts,
    _discard,
    _hand,
    _restricted_input,
)

# H=1123z, R=22 + 4 + 5 + 6z (N=5)。2zを一度でも引けば2対子で聴牌。
# 異なる4/5/6zだけでは対子を増やせない。非聴牌の列数は3, 3*2, 3*2*1。
# 同種2zの物理的多重度2を含む総列数5,20,60との差を聴牌軸に入れる。
_EFFECTIVE_DRAW_CASES = (
    (1, 5, (2, 3, 0, 0, 0, 0, 0, 0, 0)),
    (2, 20, (14, 6, 0, 0, 0, 0, 0, 0, 0)),
    (3, 60, (54, 6, 0, 0, 0, 0, 0, 0, 0)),
)

# R=55 + 6z (N=3)。最初の5zは向聴を下げないが、残せば2枚目で対子になる。
# depth2の物理列は55が2通り、56/65が各2通り。depth3は全6列に55が含まれる。
# A=1123zとC=1124zは同分布。B=1234zは常にA/Cより1向聴遠い。
_IMPROVEMENT_CASES = (
    (1, 3, (0, 3, 0, 0, 0, 0, 0, 0, 0), (0, 0, 3, 0, 0, 0, 0, 0, 0)),
    (2, 6, (2, 4, 0, 0, 0, 0, 0, 0, 0), (0, 2, 4, 0, 0, 0, 0, 0, 0)),
    (3, 6, (6, 0, 0, 0, 0, 0, 0, 0, 0), (0, 6, 0, 0, 0, 0, 0, 0, 0)),
)


class NativeProgressionSpecTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_native(self)

    def test_effective_draw_multiplicity_without_replacement(self) -> None:
        root = _counts(z1=2, z2=1, z3=1)
        remaining = _counts(z2=2, z4=1, z5=1, z6=1)
        for horizon, denominator, distribution in _EFFECTIVE_DRAW_CASES:
            with self.subTest(horizon=horizon):
                actual = _native_evaluator().evaluate_roots((root,), remaining, horizon)
                self.assertEqual(actual, ((1, distribution),))
                self.assertEqual(sum(actual[0][1]), denominator)

    def test_improvement_draws_equal_distributions_root_order_and_duplicates(
        self,
    ) -> None:
        a = _counts(z1=2, z2=1, z3=1)
        b = _counts(z1=1, z2=1, z3=1, z4=1)
        c = _counts(z1=2, z2=1, z4=1)
        remaining = _counts(z5=2, z6=1)
        for horizon, denominator, near, far in _IMPROVEMENT_CASES:
            for roots, expected in (
                ((a, b, c, a), ((1, near), (2, far), (1, near), (1, near))),
                ((b, a, a, c), ((2, far), (1, near), (1, near), (1, near))),
            ):
                with self.subTest(horizon=horizon, roots=roots):
                    actual = _native_evaluator().evaluate_roots(
                        roots, remaining, horizon
                    )
                    self.assertEqual(actual, expected)
                    for _, distribution in actual:
                        self.assertEqual(sum(distribution), denominator)

    def test_tenpai_stays_zero_even_after_a_completing_draw(self) -> None:
        # 1122zは聴牌。1zで和了しても、打牌後の軸は-1ではなく0。
        root = _counts(z1=2, z2=2)
        remaining = _counts(z1=1, z5=2)
        for horizon, distribution in (
            (1, (3, 0, 0, 0, 0, 0, 0, 0, 0)),
            (2, (6, 0, 0, 0, 0, 0, 0, 0, 0)),
            (3, (6, 0, 0, 0, 0, 0, 0, 0, 0)),
        ):
            with self.subTest(horizon=horizon):
                self.assertEqual(
                    _native_evaluator().evaluate_roots((root,), remaining, horizon),
                    ((0, distribution),),
                )


def _caller_fixture():
    # 3完成面子 + 11234z。公開河で残りを55+6zだけにする。
    # 1z切りは4単騎、3z/4z切りは対子+2単騎。3ツモ後は必ず55の対子ができ、
    # terminalはそれぞれ1/0/0。刻子/雀頭を完成できずcompletion massは全候補0。
    tiles = _hand("111m222p333s11234z")
    policy_input = _restricted_input(
        tiles,
        ("5z", "6z"),
        extra_discards=(_hand("55666z"), (), (), ()),
    )
    actions = tuple(_discard(_hand(spec)[0]) for spec in ("1z", "3z", "4z"))
    return policy_input, actions


class NativeProgressionCallerSpecTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_native(self)

    def test_progression_tie_uses_hand_value_stable_order_not_root_order(self) -> None:
        policy_input, actions = _caller_fixture()
        for ordered in (actions, tuple(reversed(actions))):
            with self.subTest(actions=ordered):
                with patch.object(
                    progression, "_new_progression_evaluator", _native_evaluator
                ):
                    selected, analysis = progression._evaluate_and_choose_discard(
                        policy_input, ordered
                    )
                # 西/北の保持価値・役route・受入は同じ。安定順で3z切りを選ぶ。
                self.assertEqual(selected, actions[1])
                self.assertTrue(analysis.progression_activated)
                self.assertEqual(analysis.sequence_denominator, 6)
                self.assertEqual(
                    {
                        c.action: (
                            c.completion_mass,
                            c.root_post_discard_shanten,
                            c.terminal_shanten_mass,
                            c.terminal_shanten_counts,
                        )
                        for c in analysis.candidate_evaluations
                    },
                    {
                        actions[0]: (0, 2, 6, (0, 6, 0, 0, 0, 0, 0, 0, 0)),
                        actions[1]: (0, 1, 0, (6, 0, 0, 0, 0, 0, 0, 0, 0)),
                        actions[2]: (0, 1, 0, (6, 0, 0, 0, 0, 0, 0, 0, 0)),
                    },
                )

    def test_diagnostic_histograms_and_both_summaries_have_fixed_values(self) -> None:
        policy_input, actions = _caller_fixture()
        with patch.object(diagnostic, "_new_progression_evaluator", _native_evaluator):
            analysis = diagnostic.analyze_mechanism_riichi_defense_offensive_efficiency(
                policy_input, actions, include_terminal_progression=True
            )
        self.assertEqual(analysis.baseline_selected_action, actions[1])
        self.assertEqual(set(analysis.baseline_eligible_actions), set(actions))
        self.assertEqual(analysis.sequence_denominator, 6)
        self.assertEqual(
            {
                c.action: (c.terminal_shanten_mass, c.terminal_shanten_counts)
                for c in analysis.candidate_evaluations
            },
            {
                actions[0]: (6, (0, 6, 0, 0, 0, 0, 0, 0, 0)),
                actions[1]: (0, (6, 0, 0, 0, 0, 0, 0, 0, 0)),
                actions[2]: (0, (6, 0, 0, 0, 0, 0, 0, 0, 0)),
            },
        )
        for summary in (
            analysis.full_legal_terminal_progression_summary,
            analysis.baseline_eligible_terminal_progression_summary,
        ):
            self.assertIsNotNone(summary)
            self.assertEqual(summary.best_terminal_shanten_mass, 0)
            self.assertEqual(summary.selected_terminal_shanten_mass, 0)
            self.assertEqual(summary.terminal_shanten_regret_mass, 0)
