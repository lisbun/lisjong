"""Track B用の使い捨てHandBelief decision-sensitivity consumer。

lisjong-project #20のoracle spikeだけで使うexperimental moduleであり、production
Policyとしてexportしない。TwoStepUkeireの最初の2段階、

    打牌後向聴数 > public current ukeire

をhard filterとして維持し、両方がtieした候補にだけcanonical
`ConcealedHandBelief.expected_count`を使った1段のrankingを適用する。

rankingは各候補のstructural effective tileについて、Policy-visibleな未見枚数から
他家concealed handのexpected countを引いたmassを合計する。これはlive-wall draw
probabilityではない。dead wall等も含む「他家concealed handに割り当てられていない
hidden mass」であり、Track Bでbaseline beliefとoracle beliefのdecision sensitivityを
同じconsumerで比較するためだけのproxyである。

wait belief、hidden state、future action、rollout、push/fold、learned modelは使用しない。
"""

from dataclasses import dataclass

from lisjong.belief.canonical_axes import wind_for_seat, wind_index
from lisjong.belief.concealed_hand_belief import ConcealedHandBelief
from lisjong.policies.two_step_ukeire import (
    _DecisionShantenEvaluator,
    _discard_action_sort_key,
    _effective_tile_types,
    _evaluate_post_discard_hands,
    _known_tile_counts,
    _ukeire_count,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import TileType

_MAX_COPIES_PER_TILE_TYPE = 4


class HandBeliefSensitivityError(Exception):
    """experimental consumerの入力またはbelief accountingが不整合な場合。"""


@dataclass(frozen=True, slots=True)
class HandBeliefSensitivityCandidateEvaluation:
    """1 discard candidateについてTrack B consumerが実際に使った値。"""

    action: DiscardAction
    post_discard_shanten: int
    current_ukeire_count: int | None
    non_opponent_effective_tile_mass: float | None


@dataclass(frozen=True, slots=True)
class HandBeliefSensitivityDecision:
    """paired comparison用のselected actionとconsumer activation。"""

    action: DiscardAction
    consumer_active: bool
    candidate_evaluations: tuple[HandBeliefSensitivityCandidateEvaluation, ...]


def _opponent_expected_count(
    policy_input: PolicyInput,
    belief: ConcealedHandBelief,
    tile_type: TileType,
) -> float:
    self_wind = wind_for_seat(
        policy_input.self_seat,
        policy_input.round.dealer_seat,
    )
    self_wind_number = wind_index(self_wind)
    return sum(
        hand.expected_count(tile_type)
        for wind_number, hand in enumerate(belief.hands)
        if wind_number != self_wind_number
    )


def _non_opponent_effective_tile_mass(
    policy_input: PolicyInput,
    belief: ConcealedHandBelief,
    post_discard_hand: tuple | list,
    shanten: int,
    known_counts: dict,
    evaluator: _DecisionShantenEvaluator,
) -> float:
    mass = 0.0
    for tile_type in _effective_tile_types(post_discard_hand, shanten, evaluator):
        unseen_count = _MAX_COPIES_PER_TILE_TYPE - known_counts.get(tile_type, 0)
        opponent_count = _opponent_expected_count(policy_input, belief, tile_type)
        available = unseen_count - opponent_count
        if available < 0.0:
            raise HandBeliefSensitivityError(
                "opponent expected count exceeds Policy-visible unseen tile count"
            )
        mass += available
    return mass


def evaluate_hand_belief_sensitive_discard(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
    belief: ConcealedHandBelief,
) -> HandBeliefSensitivityDecision:
    """同一positionへ与えたbeliefだけを変えてdiscard sensitivityを評価する。"""
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("policy_input must be a PolicyInput")
    if not isinstance(belief, ConcealedHandBelief):
        raise TypeError("belief must be a ConcealedHandBelief")
    if not isinstance(discard_actions, tuple) or not discard_actions:
        raise ValueError("discard_actions must be a non-empty tuple")
    if any(not isinstance(action, DiscardAction) for action in discard_actions):
        raise TypeError("discard_actions must contain only DiscardAction values")

    known_counts = _known_tile_counts(policy_input)
    evaluator = _DecisionShantenEvaluator()
    evaluated = _evaluate_post_discard_hands(policy_input, discard_actions, evaluator)

    minimum_shanten = min(candidate.post_discard_shanten for candidate in evaluated)
    minimum_shanten_candidates = tuple(
        candidate
        for candidate in evaluated
        if candidate.post_discard_shanten == minimum_shanten
    )
    for candidate in minimum_shanten_candidates:
        candidate.current_ukeire_count = _ukeire_count(
            candidate.post_discard_hand,
            known_counts,
            minimum_shanten,
            evaluator,
        )

    maximum_current_ukeire = max(
        candidate.current_ukeire_count for candidate in minimum_shanten_candidates
    )
    finalists = tuple(
        candidate
        for candidate in minimum_shanten_candidates
        if candidate.current_ukeire_count == maximum_current_ukeire
    )

    scores_by_action: dict[DiscardAction, float] = {}
    consumer_active = len(finalists) > 1
    if consumer_active:
        for candidate in finalists:
            scores_by_action[candidate.action] = _non_opponent_effective_tile_mass(
                policy_input,
                belief,
                candidate.post_discard_hand,
                minimum_shanten,
                known_counts,
                evaluator,
            )
        best_score = max(scores_by_action.values())
        selected = min(
            (
                candidate.action
                for candidate in finalists
                if scores_by_action[candidate.action] == best_score
            ),
            key=_discard_action_sort_key,
        )
    else:
        selected = finalists[0].action

    snapshots = tuple(
        HandBeliefSensitivityCandidateEvaluation(
            action=candidate.action,
            post_discard_shanten=candidate.post_discard_shanten,
            current_ukeire_count=candidate.current_ukeire_count,
            non_opponent_effective_tile_mass=scores_by_action.get(candidate.action),
        )
        for candidate in sorted(
            evaluated,
            key=lambda candidate: _discard_action_sort_key(candidate.action),
        )
    )
    return HandBeliefSensitivityDecision(
        action=selected,
        consumer_active=consumer_active,
        candidate_evaluations=snapshots,
    )
