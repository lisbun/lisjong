"""Issue #175 HandValue v2: bounded speed/value trade-off for the heuristic baseline.

The current ``mechanism-riichi-defense`` Policy remains unchanged. This module adds
one independent experimental generation that preserves:

    Push/Fold > Safety > mechanism defense > FiniteHorizon unique positive winner

and replaces only the HandValue fallback used by positive FiniteHorizon ties or
all-zero completion-mass states.

Within that fallback, shanten remains a hard priority. Same-shanten candidates
within four current-ukeire tiles of the fastest candidate may trade speed for
visible retained value plus one best supported yaku potential (live Yakuhai pair
or current Tanyao compatibility). No expected score, hidden information, future
draw truth, or runtime tuning is introduced.
"""

from collections import Counter

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    FiniteHorizonCandidateEvaluation,
    FiniteHorizonCompletionPolicyError,
    _evaluate_completion_masses,
    _FiniteHorizonEvaluator,
    _root_remaining_counts,
)
from lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware import (
    _defense_eligible_actions,
)
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _WIND_RANK,
    _is_tanyao_compatible,
    _own_melds,
    _retained_real_value,
    _seat_wind_rank,
    _yakuhai_han_value,
    _yaku_route_value,
)
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    MechanismRiichiDefenseYakuhaiCallPolicy,
    _mechanism_defense_eligible_actions,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import Tile
from lisjong.structural_efficiency import (
    StructuralShantenEvaluator,
    discard_action_sort_key,
    evaluate_post_discard_hands,
    known_tile_counts,
    second_step_ukeire_score,
    ukeire_count,
)

UKEIRE_TRADEOFF_BAND = 4
"""Maximum current-ukeire loss that may enter the HandValue-v2 comparison."""


def _yakuhai_pair_potential(
    post_discard_hand: tuple[Tile, ...],
    policy_input: PolicyInput,
    remaining_counts: tuple[int, ...],
) -> int:
    """Return the best live Yakuhai-pair han-equivalent potential.

    Only concealed pairs with exactly two copies are eligible. A pair whose
    player-visible remaining inventory is zero contributes no potential.
    """
    counts = Counter(tile.tile_type for tile in post_discard_hand)
    seat_wind_rank = _seat_wind_rank(policy_input)
    round_wind_rank = _WIND_RANK[policy_input.round.round_wind]

    best = 0
    for tile_type, count in counts.items():
        if count != 2:
            continue
        han_value = _yakuhai_han_value(
            tile_type,
            seat_wind_rank=seat_wind_rank,
            round_wind_rank=round_wind_rank,
        )
        if han_value == 0:
            continue
        if remaining_counts[tile_type_index(tile_type)] == 0:
            continue
        best = max(best, han_value)
    return best


def _tanyao_potential(
    post_discard_hand: tuple[Tile, ...], policy_input: PolicyInput
) -> int:
    """Return 1 only when concealed tiles plus current own melds are Tanyao-safe."""
    all_own_tiles = tuple(post_discard_hand) + tuple(
        tile for meld in _own_melds(policy_input) for tile in meld.tiles
    )
    return int(_is_tanyao_compatible(all_own_tiles))


def _supported_yaku_han_potential(
    post_discard_hand: tuple[Tile, ...],
    policy_input: PolicyInput,
    remaining_counts: tuple[int, ...],
) -> int:
    """Return the single best supported incomplete-yaku potential."""
    return max(
        _yakuhai_pair_potential(post_discard_hand, policy_input, remaining_counts),
        _tanyao_potential(post_discard_hand, policy_input),
    )


def _hand_value_v2_fallback(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
    *,
    remaining_counts: tuple[int, ...] | None = None,
) -> DiscardAction:
    """Select from a FiniteHorizon-approved subset using HandValue-v2 semantics."""
    known_counts = known_tile_counts(policy_input)
    root_remaining_counts = (
        _root_remaining_counts(policy_input)
        if remaining_counts is None
        else remaining_counts
    )
    evaluator = StructuralShantenEvaluator()
    evaluated = evaluate_post_discard_hands(policy_input, discard_actions, evaluator)

    minimum_shanten = min(candidate.post_discard_shanten for candidate in evaluated)
    shanten_finalists = tuple(
        candidate
        for candidate in evaluated
        if candidate.post_discard_shanten == minimum_shanten
    )

    ukeire_by_action = {
        candidate.action: ukeire_count(
            candidate.post_discard_hand,
            known_counts,
            minimum_shanten,
            evaluator,
        )
        for candidate in shanten_finalists
    }
    maximum_ukeire = max(ukeire_by_action.values())
    if maximum_ukeire > 0:
        speed_band = tuple(
            candidate
            for candidate in shanten_finalists
            if ukeire_by_action[candidate.action] > 0
            and ukeire_by_action[candidate.action]
            >= maximum_ukeire - UKEIRE_TRADEOFF_BAND
        )
    else:
        speed_band = shanten_finalists

    if len(speed_band) == 1:
        return speed_band[0].action

    visible_value_by_action = {
        candidate.action: (
            _retained_real_value(candidate.post_discard_hand, policy_input)
            + _supported_yaku_han_potential(
                candidate.post_discard_hand,
                policy_input,
                root_remaining_counts,
            )
        )
        for candidate in speed_band
    }
    maximum_visible_value = max(visible_value_by_action.values())
    value_finalists = tuple(
        candidate
        for candidate in speed_band
        if visible_value_by_action[candidate.action] == maximum_visible_value
    )
    if len(value_finalists) == 1:
        return value_finalists[0].action

    maximum_finalist_ukeire = max(
        ukeire_by_action[candidate.action] for candidate in value_finalists
    )
    ukeire_finalists = tuple(
        candidate
        for candidate in value_finalists
        if ukeire_by_action[candidate.action] == maximum_finalist_ukeire
    )
    if len(ukeire_finalists) == 1:
        return ukeire_finalists[0].action

    melds = _own_melds(policy_input)
    route_value_by_action = {
        candidate.action: _yaku_route_value(candidate.post_discard_hand, melds)
        for candidate in ukeire_finalists
    }
    maximum_route_value = max(route_value_by_action.values())
    route_finalists = tuple(
        candidate
        for candidate in ukeire_finalists
        if route_value_by_action[candidate.action] == maximum_route_value
    )
    if len(route_finalists) == 1:
        return route_finalists[0].action

    if minimum_shanten == 0:
        return min(
            (candidate.action for candidate in route_finalists),
            key=discard_action_sort_key,
        )

    second_step_by_action = {
        candidate.action: second_step_ukeire_score(
            candidate.post_discard_hand,
            known_counts,
            minimum_shanten,
            evaluator,
        )
        for candidate in route_finalists
    }
    maximum_second_step = max(second_step_by_action.values())
    return min(
        (
            candidate.action
            for candidate in route_finalists
            if second_step_by_action[candidate.action] == maximum_second_step
        ),
        key=discard_action_sort_key,
    )


def _hand_value_v2_from_finite_horizon(
    policy_input: PolicyInput,
    evaluations: tuple[FiniteHorizonCandidateEvaluation, ...],
    remaining_counts: tuple[int, ...],
) -> DiscardAction:
    return _hand_value_v2_fallback(
        policy_input,
        tuple(evaluation.action for evaluation in evaluations),
        remaining_counts=remaining_counts,
    )


def _evaluate_finite_horizon_hand_value_v2(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> DiscardAction:
    """Preserve current defense/FiniteHorizon semantics and replace only fallback."""
    eligible_actions = _defense_eligible_actions(policy_input, discard_actions)
    remaining_counts = _root_remaining_counts(policy_input)
    hidden_tile_count = sum(remaining_counts)
    if hidden_tile_count < DEFAULT_HORIZON:
        raise FiniteHorizonCompletionPolicyError(
            "remaining hidden tile count is smaller than the search horizon: "
            f"{hidden_tile_count} hidden tiles cannot fill {DEFAULT_HORIZON} future "
            "self-draw slots"
        )

    evaluations = _evaluate_completion_masses(
        policy_input,
        eligible_actions,
        remaining_counts,
        DEFAULT_HORIZON,
        _FiniteHorizonEvaluator(),
    )
    maximum_mass = max(evaluation.completion_mass for evaluation in evaluations)
    if maximum_mass == 0:
        return _hand_value_v2_from_finite_horizon(
            policy_input, evaluations, remaining_counts
        )

    maximum_candidates = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.completion_mass == maximum_mass
    )
    if len(maximum_candidates) == 1:
        return maximum_candidates[0].action
    return _hand_value_v2_from_finite_horizon(
        policy_input, maximum_candidates, remaining_counts
    )


class HandValueTradeoffMechanismRiichiDefensePolicy(
    MechanismRiichiDefenseYakuhaiCallPolicy
):
    """MechanismRiichiDefense with bounded HandValue-v2 fallback only."""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        mechanism_eligible = _mechanism_defense_eligible_actions(
            policy_input, discard_actions
        )
        return PolicyDecision(
            action=_evaluate_finite_horizon_hand_value_v2(
                policy_input, mechanism_eligible
            ),
            analysis=None,
        )
