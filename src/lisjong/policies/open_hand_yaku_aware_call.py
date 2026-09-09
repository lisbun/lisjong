"""supported open-yaku routeでinitial Chi/Ponを限定解禁するPolicy。

Issue #157のexperimental Policyとして、current strength baseline
`YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy`を先に1回だけ
実行する。baselineが選ぶwinning、Riichi、discard、Yakuhai call、
post-Yakuhai-open callはそのまま返し、明示的なPassだけを置換対象にする。

役牌open前のChi / non-Yakuhai Ponは、baselineと同じdefense、exact consume、
kuikae、concealed-triplet preservationを維持し、mandatory discard後の同じ
stable handでstrict shanten improvementと既存HandValueAware由来のTanyao /
Honitsu / Chinitsu compatibilityが同時に成立する場合だけ選べる。

Tanyao routeはcurrent first-party open-Tanyao rulesetへscopeを固定する。
hypothetical discardは保存せず、actual next decisionではfresh legal_actionsを
baselineへ渡す。PolicyInput以外の状態やhidden informationは使用しない。
"""

from collections import Counter
from dataclasses import dataclass

from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _own_melds,
    _yaku_route_value_for_tiles,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
    _call_action_sort_key,
    _CallAction,
    _defense_suppresses_call,
    _post_call_stable_hands,
)
from lisjong.policy_contract.action import ChiAction, PassAction, PonAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput


@dataclass(frozen=True, slots=True)
class _OpenYakuCallCandidateEvaluation:
    action: _CallAction
    best_post_call_shanten: int


def _route_compatible_post_call_shanten(
    policy_input: PolicyInput,
    action: _CallAction,
    current_shanten: int,
) -> int | None:
    """routeを保ったstrictly-improving stable handの最小向聴数を返す。"""
    current_meld_tiles = tuple(
        tile for meld in _own_melds(policy_input) for tile in meld.tiles
    )
    new_meld_tiles = (*action.consumed_tiles, action.called_tile)
    qualifying_shanten = tuple(
        post_call_shanten
        for stable_hand in _post_call_stable_hands(
            policy_input.own_hand.concealed_tiles, action
        )
        for post_call_shanten in (calculate_shanten(stable_hand),)
        if post_call_shanten < current_shanten
        and _yaku_route_value_for_tiles(
            (*stable_hand, *current_meld_tiles, *new_meld_tiles)
        )
        > 0
    )
    if not qualifying_shanten:
        return None
    return min(qualifying_shanten)


def _qualifying_initial_open_yaku_calls(
    policy_input: PolicyInput,
    call_actions: tuple[_CallAction, ...],
    current_shanten: int,
) -> tuple[_OpenYakuCallCandidateEvaluation, ...]:
    """supported routeを持つstrictly-improving initial Chi/Ponだけを返す。"""
    concealed_counts = Counter(
        tile.tile_type for tile in policy_input.own_hand.concealed_tiles
    )
    evaluations: list[_OpenYakuCallCandidateEvaluation] = []
    for action in call_actions:
        if (
            isinstance(action, PonAction)
            and concealed_counts[action.called_tile.tile_type] != 2
        ):
            continue
        best_post_call_shanten = _route_compatible_post_call_shanten(
            policy_input, action, current_shanten
        )
        if best_post_call_shanten is not None:
            evaluations.append(
                _OpenYakuCallCandidateEvaluation(action, best_post_call_shanten)
            )
    return tuple(evaluations)


class OpenHandYakuAwareCallPolicy(
    YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy
):
    """Tanyao / Honitsu / Chinitsu routeを加えたstateless call Policy。"""

    def _decide(self, decision: DecisionContext) -> PolicyDecision:
        baseline_decision = super()._decide(decision)
        if not isinstance(baseline_decision.action, PassAction):
            return baseline_decision

        call_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, (ChiAction, PonAction))
        )
        if not call_actions:
            return baseline_decision

        current_shanten = calculate_shanten(decision.input.own_hand.concealed_tiles)
        if _defense_suppresses_call(decision.input, current_shanten):
            return baseline_decision

        candidates = _qualifying_initial_open_yaku_calls(
            decision.input, call_actions, current_shanten
        )
        if not candidates:
            return baseline_decision

        selected = min(
            candidates,
            key=lambda candidate: (
                candidate.best_post_call_shanten,
                _call_action_sort_key(candidate.action),
            ),
        )
        return PolicyDecision(action=selected.action)
