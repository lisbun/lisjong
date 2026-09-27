"""Issue #230 `PlacementAwareSpeedCallKobalab0004BeliefPaijiaDiscardPolicy`のfocused test。

同じ固定decision入力で、既存の0004合成版
`PlacementAwareSpeedCallKobalab0004DiscardPolicy`と比べて、PUSH打牌のpaijia入力
だけがBelief由来の残余期待枚数へ替わることを固定する。候補制限・向聴数・
改善牌・受入（実残り枚数）・合成版fallback・非PUSH・非打牌判断は合成版と同じで、
非PUSH・非打牌経路ではBelief導出を行わない。

非一様beliefの期待値は、合成beliefから残余を手で求め、paijiaの窓（萬子rank ±2）
へ入るかどうかで順位が変わることから導く。選択関数自体を呼んで期待値を作らない。
"""

import ast
import unittest
from pathlib import Path
from unittest import mock

import lisjong.policies.placement_aware_speed_call_kobalab_0004_belief_paijia_discard as belief_candidate_module
import lisjong.policies.placement_aware_speed_call_kobalab_0004_discard as candidate_module
from lisjong import policies
from lisjong.belief import (
    SCALE,
    ConcealedHandBelief,
    HandBelief,
    derive_remaining_tile_inventory,
    exact_self_belief,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
from lisjong.belief import conditional_uniform_hand_belief as uniform_module
from lisjong.policies import (
    PlacementAwareSpeedCallKobalab0004BeliefPaijiaDiscardPolicy,
    PlacementAwareSpeedCallKobalab0004DiscardPolicy,
    PlacementAwareSpeedCallPolicy,
)
from lisjong.policies import kobalab_0004_discard as discard_module
from lisjong.policies.kobalab_0004_discard import (
    KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR,
    Kobalab0004ReferencePolicyError,
    _PublicCounts,
)
from lisjong.policies.targeted_honor_release_terminal_progression import (
    TargetedHonorReleaseBranch,
)
from lisjong.policy_contract.action import DiscardAction, RiichiAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.seat import Seat
from tests.test_placement_aware_speed_call_kobalab_0004_discard_policy import (
    FALLBACK_CONCEALED,
    TENPAI_TIE,
    NonPushBranchTest,
    OpenHandTest,
    _branch,
    _decision,
    _discard,
    _oracle_ukeire,
    _with_only_unseen,
)
from tests.test_placement_aware_speed_call_policy import (
    ALL_LAST_TOP,
    PASS,
    TANYAO_ONE_SHANTEN,
    _chi,
    _chi_meld,
    _discards,
    _hand,
    _input,
    _tile,
)

Candidate = PlacementAwareSpeedCallKobalab0004BeliefPaijiaDiscardPolicy
Composite = PlacementAwareSpeedCallKobalab0004DiscardPolicy


def _push_inputs() -> dict[str, tuple[PolicyInput, tuple[DiscardAction, ...]]]:
    """合成版testのPUSH fixture（門前・他家リーチ下・副露手・副露直後・fallback・赤5）。"""
    red = _input("123p456p789s1z4m0m6m5m", drawn="5m")
    fallback = _input(FALLBACK_CONCEALED, own_melds=(_chi_meld("234m"),), drawn="5z")
    open_hand = _input(OpenHandTest.OPEN, own_melds=(_chi_meld("234m"),), drawn="1z")
    post_call = _input(OpenHandTest.OPEN, own_melds=(_chi_meld("234m"),))
    tie = _input(TENPAI_TIE, drawn="2z")
    return {
        "closed": (tie, _discards(tie)),
        "opponent_riichi": (
            _input(TENPAI_TIE, drawn="2z", riichi_discards="1z9m"),
            _discards(tie),
        ),
        "declaration_discard": (tie, (_discard("4m"), _discard("7m"))),
        "red_five": (
            red,
            (
                _discard("4m"),
                _discard("0m"),
                _discard("6m"),
                _discard("1z"),
                _discard("1p"),
                _discard("5m", tsumogiri=True),
            ),
        ),
        "open_hand": (open_hand, _discards(open_hand)),
        "post_call": (post_call, _discards(post_call)),
        "worsening_fallback": (fallback, _discards(fallback)),
    }


def _belief(
    policy_input: PolicyInput, mass: dict[tuple[Seat, str], int]
) -> ConcealedHandBelief:
    """selfはexact、他家は`mass`（(seat, 牌) -> raw）だけを持つ合成belief。"""
    rows = [[0] * 34 for _ in range(4)]
    for (seat, spec), raw in mass.items():
        wind = wind_for_seat(seat, policy_input.round.dealer_seat)
        rows[wind_index(wind)][tile_type_index(_tile(spec).tile_type)] += raw
    self_wind = wind_for_seat(policy_input.self_seat, policy_input.round.dealer_seat)
    return ConcealedHandBelief(
        hands=tuple(
            exact_self_belief(policy_input.own_hand)
            if number == wind_index(self_wind)
            else HandBelief(
                expected_count_raw=tuple(rows[number]),
                red_five_probability_raw=(0,) * 3,
            )
            for number in range(4)
        )
    )


def _patched_belief(mass: dict[tuple[Seat, str], int]):
    return mock.patch.object(
        discard_module,
        "_estimate_concealed_hand_belief",
        lambda policy_input, conservation: _belief(policy_input, mass),
    )


def _forbid_belief():
    return mock.patch.object(
        discard_module,
        "_estimate_from_conservation",
        mock.Mock(side_effect=AssertionError("belief derived off the PUSH path")),
    )


def _spy_selection():
    """合成版の候補選択へ渡る`_PublicCounts`（受入・paijia入力）を記録する。"""
    received: list[_PublicCounts] = []
    original = candidate_module._choose_minimum_shanten_discard

    def spy(counts, discard_actions, structures):
        received.append(counts)
        return original(counts, discard_actions, structures)

    return (
        mock.patch.object(candidate_module, "_choose_minimum_shanten_discard", spy),
        received,
    )


class IdentityTest(unittest.TestCase):
    def test_separate_exported_class_without_identity_mechanism(self) -> None:
        self.assertIn(
            "PlacementAwareSpeedCallKobalab0004BeliefPaijiaDiscardPolicy",
            policies.__all__,
        )
        self.assertTrue(issubclass(Candidate, Composite))
        self.assertIsNot(Candidate, Composite)
        for policy in (Candidate, Composite, PlacementAwareSpeedCallPolicy):
            self.assertFalse(hasattr(policy, "identity"), policy.__name__)
        self.assertEqual(
            Candidate.belief_estimator, KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR
        )
        self.assertIn("conditional-uniform", Candidate.belief_estimator)
        # 比較基準の合成版はpaijia入力を変えない。
        self.assertFalse(Composite._belief_paijia)
        self.assertFalse(hasattr(Composite, "belief_estimator"))

    def test_module_does_not_import_other_policy_rules_or_native(self) -> None:
        tree = ast.parse(
            Path(belief_candidate_module.__file__).read_text(encoding="utf-8")
        )
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("lisjong.policies.kobalab_0004_reference", imported)
        self.assertNotIn("_lisjong_native", {name.split(".")[0] for name in imported})


class OutsidePushTest(unittest.TestCase):
    """非PUSH打牌・非打牌判断は合成版と同じで、Belief導出を行わない。"""

    def _assert_same_without_belief(self, decision: DecisionContext) -> None:
        with _forbid_belief():
            candidate = Candidate().choose_action_with_analysis(decision)
        self.assertEqual(candidate, Composite().choose_action_with_analysis(decision))

    def test_non_push_discard_branches(self) -> None:
        concealed = NonPushBranchTest.CONCEALED
        cases = {
            TargetedHonorReleaseBranch.FOLD_COMMON_GENBUTSU: _input(
                concealed, drawn="7z", riichi_discards="9s3z"
            ),
            TargetedHonorReleaseBranch.OTHER_CURRENT_FALLBACK: _input(
                concealed, drawn="7z", riichi_discards="3z4z"
            ),
            TargetedHonorReleaseBranch.MECHANISM_DEFENSE_FILTERED: _input(
                "34m68p68p24s59s1z4z7z3z", drawn="3z", riichi_discards="6z2z"
            ),
        }
        cases["all_last_top"] = _input(
            concealed, drawn="7z", riichi_discards="3z4z", **ALL_LAST_TOP
        )
        for key, policy_input in cases.items():
            with self.subTest(case=str(key)):
                self.assertIsNot(_branch(policy_input), TargetedHonorReleaseBranch.PUSH)
                self._assert_same_without_belief(_decision(policy_input))

    def test_riichi_and_call_decisions(self) -> None:
        tie = _input(TENPAI_TIE, drawn="2z")
        riichi = RiichiAction(actor=Seat.SEAT_0)
        self._assert_same_without_belief(_decision(tie, (*_discards(tie), riichi)))
        calls = _input(TANYAO_ONE_SHANTEN)
        for legal in ((_chi("2m", "34m"), PASS), (PASS,)):
            with self.subTest(legal=legal):
                self._assert_same_without_belief(_decision(calls, legal))

    def test_non_push_keeps_the_champion_exception(self) -> None:
        policy_input = _with_only_unseen(
            _input(NonPushBranchTest.CONCEALED, drawn="7z", riichi_discards="9s3z"),
            "1z2z",
        )
        with self.assertRaises(Exception) as composite_error:
            Composite().choose_action(_decision(policy_input))
        with _forbid_belief(), self.assertRaises(Exception) as candidate_error:
            Candidate().choose_action(_decision(policy_input))
        self.assertIs(type(candidate_error.exception), type(composite_error.exception))


class PushPaijiaInputTest(unittest.TestCase):
    def test_only_paijia_input_differs_from_composite(self) -> None:
        # 受入は実残り枚数、paijiaは`未見 − 他家期待枚数`（raw）。自手・赤5の二重計上なし。
        for name, (policy_input, actions) in _push_inputs().items():
            with self.subTest(case=name):
                self.assertIs(_branch(policy_input), TargetedHonorReleaseBranch.PUSH)
                patch, received = _spy_selection()
                with patch:
                    Candidate().choose_action(_decision(policy_input, actions))
                (counts,) = received
                inventory = derive_remaining_tile_inventory(policy_input)
                self.assertEqual(
                    counts.remaining_tile_counts, inventory.remaining_tile_counts
                )
                belief = discard_module._estimate_concealed_hand_belief(
                    policy_input, inventory
                )
                self_wind = wind_index(
                    wind_for_seat(
                        policy_input.self_seat, policy_input.round.dealer_seat
                    )
                )
                opponents = [
                    hand
                    for number, hand in enumerate(belief.hands)
                    if number != self_wind
                ]
                self.assertEqual(
                    counts.paijia_input._counts,
                    tuple(
                        inventory.remaining_tile_counts[index] * SCALE
                        - sum(hand.expected_count_raw[index] for hand in opponents)
                        for index in range(34)
                    ),
                )
                self.assertEqual(
                    counts.paijia_input._red,
                    tuple(
                        inventory.remaining_red_five_counts[color] * SCALE
                        - sum(
                            hand.red_five_probability_raw[color] for hand in opponents
                        )
                        for color in range(3)
                    ),
                )

    def test_red_five_residual_is_on_its_own_axis(self) -> None:
        policy_input, actions = _push_inputs()["red_five"]
        patch, received = _spy_selection()
        with patch:
            Candidate().choose_action(_decision(policy_input, actions))
        (counts,) = received
        inventory = derive_remaining_tile_inventory(policy_input)
        # 自手に赤5萬があるため萬子の赤5は未見0で、残余も0（負にならない）。
        self.assertEqual(inventory.remaining_red_five_counts[0], 0)
        self.assertEqual(counts.paijia_input._red[0], 0)
        self.assertGreater(counts.paijia_input._red[1], 0)
        # 34牌種の5（赤5を含む）の残余は赤5の残余以上（赤5を別に足し込まない）。
        for color, five in enumerate(("5m", "5p", "5s")):
            index = tile_type_index(_tile(five).tile_type)
            self.assertGreaterEqual(
                counts.paijia_input._counts[index], counts.paijia_input._red[color]
            )

    def test_belief_is_derived_once_from_the_shared_inventory(self) -> None:
        policy_input, actions = _push_inputs()["closed"]
        with (
            mock.patch.object(
                discard_module,
                "derive_remaining_tile_inventory",
                wraps=discard_module.derive_remaining_tile_inventory,
            ) as policy_side,
            mock.patch.object(
                uniform_module,
                "derive_remaining_tile_inventory",
                wraps=uniform_module.derive_remaining_tile_inventory,
            ) as estimator_side,
            mock.patch.object(
                discard_module,
                "_estimate_from_conservation",
                wraps=discard_module._estimate_from_conservation,
            ) as estimator,
        ):
            Candidate().choose_action(_decision(policy_input, actions))
        self.assertEqual(policy_side.call_count, 1)
        self.assertEqual(estimator_side.call_count, 0)
        self.assertEqual(estimator.call_count, 1)
        (call,) = estimator.call_args_list
        self.assertIs(call.args[0], policy_input)
        self.assertEqual(call.args[1], derive_remaining_tile_inventory(policy_input))

    def test_unseen_zero_tile_type_has_zero_residual(self) -> None:
        # 待ちの3sが4枚とも公開済み（未見0）。受入は6sの4枚で4m / 7mとも同数。
        policy_input = _input(TENPAI_TIE, drawn="2z", riichi_discards="3s3s3s3s")
        self.assertIs(_branch(policy_input), TargetedHonorReleaseBranch.PUSH)
        for spec in ("4m", "7m"):
            self.assertEqual(_oracle_ukeire(policy_input, _tile(spec)), (0, 4))
        patch, received = _spy_selection()
        with patch:
            action = Candidate().choose_action(_decision(policy_input))
        (counts,) = received
        self.assertEqual(
            counts.paijia_input._counts[tile_type_index(_tile("3s").tile_type)], 0
        )
        self.assertEqual(action, Composite().choose_action(_decision(policy_input)))

    def test_belief_exceeding_remaining_mass_fails_closed(self) -> None:
        policy_input, actions = _push_inputs()["closed"]
        # 未見の3mは4枚。他家が5枚分持つbeliefは保存則違反で、clampしない。
        with _patched_belief({(Seat.SEAT_1, "3m"): 5 * SCALE}):
            with self.assertRaises(Kobalab0004ReferencePolicyError):
                Candidate().choose_action(_decision(policy_input, actions))

    def test_too_few_unseen_tiles_for_opponent_slots_fails_closed(self) -> None:
        # 他家の手牌より未見が少ない契約外入力。合成版は打牌を選ぶが、Belief由来の
        # paijia入力は導出できず拒否する（0004 Belief版と同じ契約）。
        policy_input = _with_only_unseen(_input(TENPAI_TIE, drawn="2z"), "3s6s")
        decision = _decision(policy_input)
        self.assertEqual(Composite().choose_action(decision), _discard("7m"))
        with self.assertRaises(Kobalab0004ReferencePolicyError):
            Candidate().choose_action(decision)


class PushSelectionTest(unittest.TestCase):
    def test_common_scale_belief_keeps_every_composite_selection(self) -> None:
        # 他家の期待枚数0: 残余 = 未見 × SCALE（丸めなし）。paijiaは共通倍率になり、
        # 候補制限・fallbackを含め合成版と同じActionになる。
        for name, (policy_input, actions) in _push_inputs().items():
            with self.subTest(case=name), _patched_belief({}):
                decision = _decision(policy_input, actions)
                candidate = Candidate().choose_action_with_analysis(decision)
                composite = Composite().choose_action_with_analysis(decision)
                self.assertEqual(candidate, composite)
                self.assertIsNone(candidate.analysis)

    def test_uniform_estimator_quantization_breaks_only_exact_paijia_ties(
        self,
    ) -> None:
        # 現行の一様推定器（丸め規則は変更しない）。4m / 7mのpaijia同点
        # （closed / 宣言牌）だけ、量子化で順位が変わる。推定器は赤5と通常5を別の
        # physical poolとして丸めるため、未見3枚どうしでも5mの残余だけが1 raw
        # 小さく、5mを窓に含む4mのpaijiaが7mより1 raw低くなり4mを先に評価する。
        # 受入は同数のままで、他のfixtureは合成版と同じActionになる。
        quantized = {"closed", "declaration_discard"}
        for name, (policy_input, actions) in _push_inputs().items():
            with self.subTest(case=name):
                decision = _decision(policy_input, actions)
                candidate = Candidate().choose_action_with_analysis(decision)
                composite = Composite().choose_action_with_analysis(decision)
                if name not in quantized:
                    self.assertEqual(candidate, composite)
                    continue
                self.assertEqual(composite.action, _discard("7m"))
                self.assertEqual(candidate.action, _discard("4m"))
                self.assertIsNone(candidate.analysis)
                counts = discard_module._belief_paijia_counts(policy_input)
                residual = counts.paijia_input._counts
                index = {
                    spec: tile_type_index(_tile(spec).tile_type)
                    for spec in ("4m", "5m", "6m", "7m")
                }
                self.assertEqual(residual[index["5m"]] + 1, residual[index["4m"]])
                self.assertEqual(residual[index["6m"]], residual[index["7m"]])
                self.assertEqual(
                    counts.paijia(_tile("4m")) + 1, counts.paijia(_tile("7m"))
                )

    def test_non_uniform_belief_changes_the_tie_break(self) -> None:
        # 4m / 7mはどちらを切っても45sの両面（3s6s、未見8枚）で聴牌し受入同数。
        # 合成版はpaijia同点（4m・7mの窓の未見が対称）でsource順の7mを切る。他家が
        # 3mを2枚持つbeliefでは、3mは4mのpaijiaの窓（2m..6m）にだけ入るため4mの
        # paijiaが下がり、4mを先に評価して切る。受入・候補集合は変わらない。
        policy_input, actions = _push_inputs()["closed"]
        for spec in ("4m", "7m"):
            self.assertEqual(_oracle_ukeire(policy_input, _tile(spec)), (0, 8))
        decision = _decision(policy_input, actions)
        self.assertEqual(Composite().choose_action(decision), _discard("7m"))
        composite_counts = _PublicCounts(policy_input)
        self.assertEqual(
            composite_counts.paijia(_tile("4m")), composite_counts.paijia(_tile("7m"))
        )
        patch, received = _spy_selection()
        with _patched_belief({(Seat.SEAT_2, "3m"): 2 * SCALE}), patch:
            traced = Candidate().choose_action_with_analysis(decision)
            action = Candidate().choose_action(decision)
        counts = received[0]
        self.assertEqual(
            counts.paijia_input._counts[tile_type_index(_tile("3m").tile_type)],
            2 * SCALE,
        )
        self.assertLess(counts.paijia(_tile("4m")), counts.paijia(_tile("7m")))
        self.assertEqual(
            counts.remaining_tile_counts, composite_counts.remaining_tile_counts
        )
        self.assertEqual(traced.action, _discard("4m"))
        self.assertIsNone(traced.analysis)
        self.assertEqual(action, traced.action)
        self.assertIn(action, actions)

    def test_non_uniform_belief_stays_within_the_route_restriction(self) -> None:
        # 副露手ではroute制限後の候補（9p / 1z）だけを比較する。beliefで制限外の牌が
        # 評価順の先頭になっても、制限外のActionを返さない。
        policy_input, actions = _push_inputs()["open_hand"]
        with _patched_belief({(Seat.SEAT_2, "8p"): 2 * SCALE}):
            action = Candidate().choose_action(_decision(policy_input, actions))
        self.assertIn(action.tile, set(_hand("9p1z")))
        self.assertEqual(action, _discard("1z"))

    def test_choose_action_matches_traced_and_ignores_legal_order(self) -> None:
        for name, (policy_input, actions) in _push_inputs().items():
            with self.subTest(case=name):
                policy = Candidate()
                results = set()
                for legal in (actions, tuple(reversed(actions))):
                    decision = _decision(policy_input, legal)
                    action = policy.choose_action(decision)
                    self.assertEqual(
                        action, policy.choose_action_with_analysis(decision).action
                    )
                    self.assertIn(action, legal)
                    results.add(action)
                self.assertEqual(len(results), 1)

    def test_discard_not_in_hand_fails_closed(self) -> None:
        policy_input, actions = _push_inputs()["closed"]
        with self.assertRaises(Kobalab0004ReferencePolicyError):
            Candidate().choose_action(
                _decision(policy_input, (*actions, _discard("0m")))
            )


if __name__ == "__main__":
    unittest.main()
