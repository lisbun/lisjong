"""lisbun/lisjong#330 `PlacementAwareSpeedCallSuitedPaijiaTiePolicy`のfocused test。

gateの内外ではChampion `PlacementAwareSpeedCallPolicy`と同じactionになること、
発動した判断でHandValueAwareの5段階の評価値を落とさず、公開paijia最小の数牌を
選ぶこと（Championの打牌も最小なら維持）を固定する。paijiaの期待値は
`_PaijiaInput`の式を未見枚数から手計算した値で、改善の証明には使わない。

gate単体のtestはChampionの打牌評価（`_evaluate_and_choose_discard_with_hva`）を
差し替え、paijiaだけ実際の`_PublicCounts`で計算する。end-to-endのtestは差し替えない。
"""

import dataclasses
import pickle
import unittest
from types import SimpleNamespace
from unittest import mock

import lisjong.policies.placement_aware_speed_call_suited_paijia_tie as candidate_module
import lisjong.policies.targeted_honor_release_terminal_progression as targeted_module
from lisjong.candidate_boundary import check_candidate_imports, check_candidate_purity
from lisjong.policies import (
    PlacementAwareSpeedCallPolicy,
    PlacementAwareSpeedCallSuitedPaijiaTiePolicy,
)
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    HandValueCandidateEvaluation,
)
from lisjong.policies.placement_aware_speed_call_suited_paijia_tie import (
    SuitedPaijiaCandidateEvaluation,
    SuitedPaijiaTieAnalysis,
    SuitedPaijiaTieStage,
)
from lisjong.policies.targeted_honor_release_terminal_progression import (
    TargetedHonorReleaseActivationStage,
    TargetedHonorReleaseBranch,
)
from lisjong.policy_contract.action import (
    DiscardAction,
    RiichiAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.wind import Wind
from tests.test_placement_aware_speed_call_policy import (
    _discards,
    _hand,
    _input,
    _player,
    _pon_meld,
    _tile,
)

_MODULE = "lisjong.policies.placement_aware_speed_call_suited_paijia_tie"
_FACTORY = f"{_MODULE}:PlacementAwareSpeedCallSuitedPaijiaTiePolicy"
_STAGE = TargetedHonorReleaseActivationStage

# 1479m 258p 369s 1357z + 7z自摸。Championは4mを切る（FH all-zero、R5なし）。
# HVAの最終tieは4m 7m 2p 5p 8p 3s 6s（向聴6・受入63・価値0・route 0・second 2847）。
# 未見枚数は手牌の牌種が3、それ以外が4。paijia: 4m=20、7m=19、2p=8p=15、
# 5p=20（赤5p未見の+1を含む）、3s=6s=20。最小の2p / 8pのうちsort順で2pを選ぶ。
SWITCH_HAND = "1479m258p369s1357z"
# 147m147p147s1357z + 7z自摸。最終tieの4m 7m 4p 7p 4s 7sはpaijiaがすべて20。
KEEP_HAND = "147m147p147s13577z"


def _action(spec: str, *, tsumogiri: bool = False) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=_tile(spec), tsumogiri=tsumogiri)


def _push_input(hand: str = SWITCH_HAND, **kwargs) -> PolicyInput:
    return _input(hand, drawn="7z", **kwargs)


def _with_player(policy_input: PolicyInput, index: int, player) -> PolicyInput:
    players = list(policy_input.players)
    players[index] = player
    return dataclasses.replace(policy_input, players=tuple(players))


def _decision(policy_input: PolicyInput, legal_actions=None) -> DecisionContext:
    return DecisionContext(
        input=policy_input,
        legal_actions=_discards(policy_input)
        if legal_actions is None
        else legal_actions,
    )


def _hva(
    action: DiscardAction,
    *,
    shanten: int = 6,
    ukeire: int | None = 63,
    retained: int | None = 0,
    route: int | None = 0,
    second: int | None = 2847,
) -> HandValueCandidateEvaluation:
    return HandValueCandidateEvaluation(
        action=action,
        post_discard_shanten=shanten,
        current_ukeire_count=ukeire,
        retained_real_value=retained,
        yaku_route_value=route,
        second_step_ukeire_score=second,
    )


def _run_gate(
    policy_input: PolicyInput,
    champion_action: DiscardAction,
    snapshots,
    *,
    stage: TargetedHonorReleaseActivationStage = _STAGE.NO_QUALIFYING_HONOR_PEER,
    branch: TargetedHonorReleaseBranch = TargetedHonorReleaseBranch.PUSH,
):
    """Championの打牌評価だけを差し替えて候補の判断を実行する。"""
    champion_analysis = SimpleNamespace(branch=branch, activation_stage=stage)
    with mock.patch.object(
        candidate_module,
        "_evaluate_and_choose_discard_with_hva",
        return_value=(champion_action, snapshots, champion_analysis),
    ):
        return candidate_module._evaluate_and_choose_discard(
            policy_input, _discards(policy_input)
        )


class PolicyBoundaryTest(unittest.TestCase):
    def test_public_export_is_stateless_picklable_and_inherits_non_discard(self):
        policy = PlacementAwareSpeedCallSuitedPaijiaTiePolicy()
        self.assertIsInstance(policy, PlacementAwareSpeedCallPolicy)
        self.assertEqual(vars(policy), {})
        self.assertIs(
            pickle.loads(pickle.dumps(PlacementAwareSpeedCallSuitedPaijiaTiePolicy)),
            PlacementAwareSpeedCallSuitedPaijiaTiePolicy,
        )
        self.assertIs(
            PlacementAwareSpeedCallSuitedPaijiaTiePolicy._decide,
            PlacementAwareSpeedCallPolicy._decide,
        )

    def test_static_import_check_passes(self):
        report = check_candidate_imports(_MODULE)
        self.assertTrue(report.passed, msg=report.violations)

    def test_purity_check_passes_across_processes(self):
        decisions = (
            _decision(_push_input()),
            _decision(_push_input(KEEP_HAND)),
        )
        report = check_candidate_purity(_FACTORY, decisions)
        self.assertTrue(report.passed, msg=(report.mismatches, report.errors))
        self.assertEqual(report.decision_count, len(decisions))


class ChampionHelperRegressionTest(unittest.TestCase):
    def test_extracted_helper_matches_champion_entry_points(self):
        for hand in (SWITCH_HAND, KEEP_HAND):
            policy_input = _push_input(hand)
            actions = _discards(policy_input)
            with self.subTest(hand=hand):
                action, snapshots, analysis = (
                    targeted_module._evaluate_and_choose_discard_with_hva(
                        policy_input, actions
                    )
                )
                self.assertEqual(
                    targeted_module._evaluate_and_choose_discard(policy_input, actions),
                    (action, analysis),
                )
                self.assertIsNotNone(snapshots)
                self.assertEqual(
                    {snapshot.action for snapshot in snapshots}, set(actions)
                )

    def test_helper_returns_no_snapshots_outside_all_zero_push(self):
        policy_input = _input(SWITCH_HAND, drawn=None)
        action, snapshots, analysis = (
            targeted_module._evaluate_and_choose_discard_with_hva(
                policy_input, _discards(policy_input)
            )
        )
        self.assertIsNone(snapshots)
        self.assertIs(analysis.activation_stage, _STAGE.NOT_NORMAL_TURN)
        self.assertEqual(
            targeted_module._evaluate_and_choose_discard(
                policy_input, _discards(policy_input)
            ),
            (action, analysis),
        )


class EndToEndTest(unittest.TestCase):
    def _both(self, decision: DecisionContext):
        return (
            PlacementAwareSpeedCallPolicy().choose_action_with_analysis(decision),
            PlacementAwareSpeedCallSuitedPaijiaTiePolicy().choose_action_with_analysis(
                decision
            ),
        )

    def test_switch_keeps_all_five_stages_and_picks_minimum_paijia(self):
        decision = _decision(_push_input())
        champion, candidate = self._both(decision)

        self.assertEqual(champion.action, _action("4m"))
        self.assertEqual(candidate.action, _action("2p"))
        self.assertIn(candidate.action, decision.legal_actions)
        analysis = candidate.analysis
        self.assertIsInstance(analysis, SuitedPaijiaTieAnalysis)
        self.assertIs(analysis.stage, SuitedPaijiaTieStage.PAIJIA_SWITCH)
        self.assertEqual(analysis.champion_action, champion.action)
        self.assertTrue(analysis.action_changed)
        self.assertEqual(
            tuple(
                (evaluation.action.tile, evaluation.paijia)
                for evaluation in analysis.paijia_evaluations
            ),
            tuple(
                (_tile(spec), paijia)
                for spec, paijia in (
                    ("4m", 20),
                    ("7m", 19),
                    ("2p", 15),
                    ("5p", 20),
                    ("8p", 15),
                    ("3s", 20),
                    ("6s", 20),
                )
            ),
        )

        _, snapshots, _ = targeted_module._evaluate_and_choose_discard_with_hva(
            decision.input, decision.legal_actions
        )
        by_action = {snapshot.action: snapshot for snapshot in snapshots}
        champion_values = dataclasses.replace(
            by_action[champion.action], action=candidate.action
        )
        self.assertEqual(by_action[candidate.action], champion_values)
        self.assertIsNotNone(champion_values.second_step_ukeire_score)

    def test_switch_is_deterministic_and_traced_path_matches(self):
        decision = _decision(_push_input())
        policy = PlacementAwareSpeedCallSuitedPaijiaTiePolicy()
        traced = policy.choose_action_with_analysis(decision)
        self.assertEqual(policy.choose_action(decision), traced.action)
        self.assertEqual(
            PlacementAwareSpeedCallSuitedPaijiaTiePolicy().choose_action_with_analysis(
                decision
            ),
            traced,
        )
        reversed_decision = _decision(
            decision.input, tuple(reversed(decision.legal_actions))
        )
        self.assertEqual(policy.choose_action(reversed_decision), traced.action)

    def test_equal_minimum_paijia_keeps_champion(self):
        champion, candidate = self._both(_decision(_push_input(KEEP_HAND)))
        self.assertEqual(candidate.action, champion.action)
        self.assertIs(
            candidate.analysis.stage, SuitedPaijiaTieStage.CHAMPION_MINIMUM_PAIJIA
        )
        self.assertFalse(candidate.analysis.action_changed)
        self.assertEqual(
            {evaluation.paijia for evaluation in candidate.analysis.paijia_evaluations},
            {20},
        )

    def test_out_of_gate_inputs_match_champion_action(self):
        base = _push_input()
        cases = {
            "post-call / no drawn tile": (
                _input(SWITCH_HAND, drawn=None),
                SuitedPaijiaTieStage.NOT_NORMAL_TURN,
            ),
            "opponent riichi": (
                _push_input(riichi_discards="9m"),
                SuitedPaijiaTieStage.OPPONENT_RIICHI,
            ),
            "all last": (
                _push_input(round_wind=Wind.SOUTH, hand_number=4),
                SuitedPaijiaTieStage.ALL_LAST,
            ),
            "self riichi declared": (
                _with_player(base, 0, _player(riichi=RiichiState.DECLARED)),
                SuitedPaijiaTieStage.SELF_RIICHI,
            ),
        }
        for name, (policy_input, stage) in cases.items():
            with self.subTest(name):
                champion, candidate = self._both(_decision(policy_input))
                self.assertEqual(candidate.action, champion.action)
                self.assertIs(candidate.analysis.stage, stage)
                self.assertFalse(candidate.analysis.action_changed)
                self.assertEqual(candidate.analysis.paijia_evaluations, ())

    def test_open_hand_matches_champion_action(self):
        policy_input = _input(
            "1479m258p369s7z", drawn="7z", own_melds=(_pon_meld("1z"),)
        )
        champion, candidate = self._both(_decision(policy_input))
        self.assertEqual(candidate.action, champion.action)
        self.assertIs(candidate.analysis.stage, SuitedPaijiaTieStage.HAS_OWN_MELD)

    def test_win_and_riichi_priority_are_inherited(self):
        policy_input = _push_input()
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=_tile("7z"))
        riichi = RiichiAction(actor=Seat.SEAT_0)
        discards = _discards(policy_input)
        for legal, expected in (
            ((*discards, tsumo), tsumo),
            ((*discards, riichi), riichi),
        ):
            with self.subTest(expected=type(expected).__name__):
                champion, candidate = self._both(_decision(policy_input, legal))
                self.assertEqual(candidate, champion)
                self.assertEqual(candidate.action, expected)

    def test_inconsistent_input_raises_like_champion(self):
        # 手牌に1枚ある1mを他家が4枚捨てた（5枚目）入力。
        policy_input = _with_player(_push_input(), 2, _player(discards="1111m"))
        decision = _decision(policy_input)
        with self.assertRaises(Exception) as champion_error:
            PlacementAwareSpeedCallPolicy().choose_action(decision)
        with self.assertRaises(type(champion_error.exception)):
            PlacementAwareSpeedCallSuitedPaijiaTiePolicy().choose_action(decision)


class GateTest(unittest.TestCase):
    def test_champion_stages_outside_no_r5_all_zero_push_keep_champion(self):
        policy_input = _push_input()
        snapshots = (_hva(_action("4m")), _hva(_action("2p")))
        cases = (
            (
                TargetedHonorReleaseBranch.FOLD_COMMON_GENBUTSU,
                _STAGE.NON_PUSH_BRANCH,
                SuitedPaijiaTieStage.NON_PUSH_BRANCH,
            ),
            (
                TargetedHonorReleaseBranch.MECHANISM_DEFENSE_FILTERED,
                _STAGE.NON_PUSH_BRANCH,
                SuitedPaijiaTieStage.NON_PUSH_BRANCH,
            ),
            (
                TargetedHonorReleaseBranch.PUSH,
                _STAGE.POSITIVE_COMPLETION,
                SuitedPaijiaTieStage.POSITIVE_COMPLETION,
            ),
            (
                TargetedHonorReleaseBranch.PUSH,
                _STAGE.R5_PARENT_BEST,
                SuitedPaijiaTieStage.R5_EVALUATED,
            ),
            (
                TargetedHonorReleaseBranch.PUSH,
                _STAGE.R5_NON_HONOR_ONLY_BEST,
                SuitedPaijiaTieStage.R5_EVALUATED,
            ),
            (
                TargetedHonorReleaseBranch.PUSH,
                _STAGE.R5_HONOR_ONLY_SWITCH,
                SuitedPaijiaTieStage.R5_EVALUATED,
            ),
        )
        for branch, champion_stage, expected in cases:
            with self.subTest(branch=branch, champion_stage=champion_stage):
                selected, analysis = _run_gate(
                    policy_input,
                    _action("4m"),
                    snapshots,
                    stage=champion_stage,
                    branch=branch,
                )
                self.assertEqual(selected, _action("4m"))
                self.assertIs(analysis.stage, expected)

    def test_every_no_r5_all_zero_stage_can_activate(self):
        snapshots = (_hva(_action("4m")), _hva(_action("2p")))
        for champion_stage in candidate_module._NO_R5_ALL_ZERO_STAGES:
            with self.subTest(champion_stage=champion_stage):
                selected, analysis = _run_gate(
                    _push_input(), _action("4m"), snapshots, stage=champion_stage
                )
                self.assertEqual(selected, _action("2p"))
                self.assertIs(analysis.stage, SuitedPaijiaTieStage.PAIJIA_SWITCH)

    def test_ankan_counts_as_own_meld(self):
        ankan = PublicMeld(
            kind=MeldKind.ANKAN,
            tiles=_hand("1111z"),
            from_seat=None,
            called_tile=None,
        )
        policy_input = _input("1479m258p369s7z", drawn="7z", own_melds=(ankan,))
        selected, analysis = _run_gate(
            policy_input, _action("4m"), (_hva(_action("4m")), _hva(_action("2p")))
        )
        self.assertEqual(selected, _action("4m"))
        self.assertIs(analysis.stage, SuitedPaijiaTieStage.HAS_OWN_MELD)

    def test_west_round_counts_as_all_last(self):
        policy_input = _push_input(round_wind=Wind.WEST, hand_number=1)
        selected, analysis = _run_gate(
            policy_input, _action("4m"), (_hva(_action("4m")), _hva(_action("2p")))
        )
        self.assertEqual(selected, _action("4m"))
        self.assertIs(analysis.stage, SuitedPaijiaTieStage.ALL_LAST)

    def test_incomplete_or_unique_final_tie_keeps_champion(self):
        cases = {
            "second-step unique maximum": (
                _hva(_action("4m"), second=30),
                _hva(_action("2p"), second=29),
            ),
            "tenpai (second-step not evaluated)": (
                _hva(_action("4m"), shanten=0, second=None),
                _hva(_action("2p"), shanten=0, second=None),
            ),
            "decided at retained real value": (
                _hva(_action("4m"), retained=1, route=None, second=None),
                _hva(_action("2p"), retained=0, route=None, second=None),
            ),
        }
        for name, snapshots in cases.items():
            with self.subTest(name):
                selected, analysis = _run_gate(_push_input(), _action("4m"), snapshots)
                self.assertEqual(selected, _action("4m"))
                self.assertIs(analysis.stage, SuitedPaijiaTieStage.NO_FINAL_TIE)

    def test_honor_champion_keeps_champion(self):
        snapshots = (_hva(_action("1z")), _hva(_action("2p")), _hva(_action("8p")))
        selected, analysis = _run_gate(_push_input(), _action("1z"), snapshots)
        self.assertEqual(selected, _action("1z"))
        self.assertIs(analysis.stage, SuitedPaijiaTieStage.CHAMPION_HONOR)

    def test_single_suited_option_keeps_champion(self):
        cases = {
            "suited + honor only": (_hva(_action("4m")), _hva(_action("1z"))),
            "same tile tedashi / tsumogiri": (
                _hva(_action("4m")),
                _hva(_action("4m", tsumogiri=True)),
            ),
        }
        for name, snapshots in cases.items():
            with self.subTest(name):
                selected, analysis = _run_gate(_push_input(), _action("4m"), snapshots)
                self.assertEqual(selected, _action("4m"))
                self.assertIs(analysis.stage, SuitedPaijiaTieStage.SINGLE_SUITED_OPTION)

    def test_honor_tie_members_are_not_candidates(self):
        # 最終tieに字牌があっても比較対象は数牌だけで、数牌の最小（2p=15）を選ぶ。
        snapshots = (_hva(_action("4m")), _hva(_action("2p")), _hva(_action("1z")))
        selected, analysis = _run_gate(_push_input(), _action("4m"), snapshots)
        self.assertEqual(selected, _action("2p"))
        self.assertEqual(
            tuple(evaluation.action for evaluation in analysis.paijia_evaluations),
            (_action("4m"), _action("2p")),
        )


class PaijiaSelectionTest(unittest.TestCase):
    """未見枚数・ドラ・赤を変えたとき、手計算したpaijiaどおりに選ぶこと。"""

    def _select(self, policy_input: PolicyInput, *specs: str):
        snapshots = tuple(_hva(_action(spec)) for spec in specs)
        return _run_gate(policy_input, _action(specs[0]), snapshots)

    def _paijia(self, analysis) -> dict[str, int]:
        names = {_tile(spec): spec for spec in ("4m", "7m", "2p", "8p")}
        return {
            names[evaluation.action.tile]: evaluation.paijia
            for evaluation in analysis.paijia_evaluations
        }

    def test_baseline_switches_to_lower_paijia(self):
        # 4m: (4,4,3,4,4)=19 + 赤5m未見1 = 20。7m: (4,4,3,4,3)=18 + 1 = 19。
        selected, analysis = self._select(_push_input(), "4m", "7m")
        self.assertEqual(self._paijia(analysis), {"4m": 20, "7m": 19})
        self.assertEqual(selected, _action("7m"))
        self.assertIs(analysis.stage, SuitedPaijiaTieStage.PAIJIA_SWITCH)

    def test_public_discards_lower_champion_paijia_and_keep_it(self):
        # 2m 3mが公開され未見3: 4m: (3,3,3,4,4)=17 + 1 = 18 < 7m 19。
        policy_input = _with_player(_push_input(), 2, _player(discards="23m"))
        selected, analysis = self._select(policy_input, "4m", "7m")
        self.assertEqual(self._paijia(analysis), {"4m": 18, "7m": 19})
        self.assertEqual(selected, _action("4m"))
        self.assertIs(analysis.stage, SuitedPaijiaTieStage.CHAMPION_MINIMUM_PAIJIA)

    def test_seen_red_five_removes_bonus_and_equal_minimum_keeps_champion(self):
        # 赤5mが公開され5mは未見3・赤未見0。
        # 4m: left 4, center min(4,3)=3, right min(3,4)=3 -> (4,4,3,3,3)=17。
        # 7m: left min(3,4)=3, center 4, right min(4,3)=3 -> (3,4,3,4,3)=17。
        policy_input = _with_player(_push_input(), 2, _player(discards="0m"))
        selected, analysis = self._select(policy_input, "4m", "7m")
        self.assertEqual(self._paijia(analysis), {"4m": 17, "7m": 17})
        self.assertEqual(selected, _action("4m"))
        self.assertIs(analysis.stage, SuitedPaijiaTieStage.CHAMPION_MINIMUM_PAIJIA)

    def test_dora_raises_peer_paijia_and_keeps_champion(self):
        # ドラ表示1p（未見3）でドラ2p: 2p: (0,3,3,4,4)、2p項を2倍 -> 3+6+4+4=17、
        # 全体を2倍して34。4m=20が最小。表示牌がなければ2p=15でswitchする。
        without_dora = self._select(_push_input(), "4m", "2p")
        self.assertEqual(self._paijia(without_dora[1]), {"4m": 20, "2p": 15})
        self.assertEqual(without_dora[0], _action("2p"))

        selected, analysis = self._select(_push_input(dora_indicators="1p"), "4m", "2p")
        self.assertEqual(self._paijia(analysis), {"4m": 20, "2p": 34})
        self.assertEqual(selected, _action("4m"))

    def test_equal_minimum_peers_use_stable_sort_order(self):
        # 2p / 8pはともに15。入力順によらずsort順の2pを選ぶ。
        selected, analysis = self._select(_push_input(), "4m", "8p", "2p")
        self.assertEqual(selected, _action("2p"))
        self.assertEqual(self._paijia(analysis), {"4m": 20, "2p": 15, "8p": 15})


class AnalysisContractTest(unittest.TestCase):
    def test_analysis_rejects_inconsistent_change_flag_and_payload(self):
        evaluation = SuitedPaijiaCandidateEvaluation(action=_action("4m"), paijia=20)
        with self.assertRaises(ValueError):
            SuitedPaijiaTieAnalysis(
                stage=SuitedPaijiaTieStage.PAIJIA_SWITCH,
                champion_action=_action("4m"),
                selected_action=_action("4m"),
                action_changed=False,
                paijia_evaluations=(evaluation,),
            )
        with self.assertRaises(ValueError):
            SuitedPaijiaTieAnalysis(
                stage=SuitedPaijiaTieStage.NO_FINAL_TIE,
                champion_action=_action("4m"),
                selected_action=_action("4m"),
                action_changed=False,
                paijia_evaluations=(evaluation,),
            )
        with self.assertRaises(ValueError):
            SuitedPaijiaTieAnalysis(
                stage=SuitedPaijiaTieStage.CHAMPION_MINIMUM_PAIJIA,
                champion_action=_action("4m"),
                selected_action=_action("4m"),
                action_changed=False,
                paijia_evaluations=(),
            )


if __name__ == "__main__":
    unittest.main()
