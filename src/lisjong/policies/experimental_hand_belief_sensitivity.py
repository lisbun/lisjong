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

lisjong-project #22 Phase 0.5では、expected countだけを出力するdisposable
estimatorを同じrankingで比較する必要があるため、consumerのbelief入力を
`OpponentExpectedCounts`（viewer視点で他家3人分を合算した34牌種canonical
expected count table）へ縮約するpure seamを追加した。`ConcealedHandBelief`を
受け取る既存entry pointはこのseamへ委譲するだけで、consumerのranking、
consumer activation条件、fail-closed accountingは変更していない。
`red_five_probability`の捏造、omniscient truthの混入、production
`ConcealedHandBelief` contractの変更は行わない。
"""

from dataclasses import dataclass

from lisjong.belief.canonical_axes import (
    tile_type_from_index,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
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
_OPPONENT_COUNT = 3
_TILE_TYPE_COUNT = 34
_MAX_OPPONENT_EXPECTED_COUNT = float(_MAX_COPIES_PER_TILE_TYPE * _OPPONENT_COUNT)
_CANONICAL_TILE_TYPES: tuple[TileType, ...] = tuple(
    tile_type_from_index(index) for index in range(_TILE_TYPE_COUNT)
)


class HandBeliefSensitivityError(Exception):
    """experimental consumerの入力またはbelief accountingが不整合な場合。"""


@dataclass(frozen=True, slots=True)
class OpponentExpectedCounts:
    """viewer視点で他家3人分を合算したconcealed expected countのcanonical table。

    `counts`は`canonical_axes.tile_type_index()`順のlength 34 tableであり、
    値は「self以外の3 windのconcealed handに存在すると推定される同一base tile
    kindの合計expected count」である。self rowは定義上含めない。

    上限は3 opponents × 4 copiesというstructural boundだけを検証する。
    viewer-visibleな未見枚数を超えるかどうかはposition依存の
    conservation条件であり、consumer側の
    `HandBeliefSensitivityError`としてfail closedで扱う。
    """

    counts: tuple[float, ...]

    def __post_init__(self) -> None:
        try:
            values = tuple(self.counts)
        except TypeError:
            raise TypeError("counts must be an iterable of float") from None
        if len(values) != _TILE_TYPE_COUNT:
            raise ValueError(f"counts must contain exactly {_TILE_TYPE_COUNT} values")
        normalized: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("counts must contain only real number values")
            count = float(value)
            if not 0.0 <= count <= _MAX_OPPONENT_EXPECTED_COUNT:
                raise ValueError(
                    f"counts values must be within 0.0..{_MAX_OPPONENT_EXPECTED_COUNT}"
                )
            normalized.append(count)
        object.__setattr__(self, "counts", tuple(normalized))

    def total(self, tile_type: TileType) -> float:
        """`tile_type`について他家3人分を合算したexpected countを返す。"""
        return self.counts[tile_type_index(tile_type)]


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


def opponent_expected_counts_from_belief(
    policy_input: PolicyInput,
    belief: ConcealedHandBelief,
) -> OpponentExpectedCounts:
    """canonical `ConcealedHandBelief`をviewer視点の他家合算tableへ縮約する。"""
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("policy_input must be a PolicyInput")
    if not isinstance(belief, ConcealedHandBelief):
        raise TypeError("belief must be a ConcealedHandBelief")

    self_wind = wind_for_seat(
        policy_input.self_seat,
        policy_input.round.dealer_seat,
    )
    self_wind_number = wind_index(self_wind)
    return OpponentExpectedCounts(
        counts=tuple(
            sum(
                hand.expected_count(tile_type)
                for wind_number, hand in enumerate(belief.hands)
                if wind_number != self_wind_number
            )
            for tile_type in _CANONICAL_TILE_TYPES
        )
    )


def _non_opponent_effective_tile_mass(
    opponent_counts: OpponentExpectedCounts,
    post_discard_hand: tuple | list,
    shanten: int,
    known_counts: dict,
    evaluator: _DecisionShantenEvaluator,
) -> float:
    mass = 0.0
    for tile_type in _effective_tile_types(post_discard_hand, shanten, evaluator):
        unseen_count = _MAX_COPIES_PER_TILE_TYPE - known_counts.get(tile_type, 0)
        opponent_count = opponent_counts.total(tile_type)
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
    return evaluate_expected_count_sensitive_discard(
        policy_input,
        discard_actions,
        opponent_expected_counts_from_belief(policy_input, belief),
    )


def evaluate_expected_count_sensitive_discard(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
    opponent_counts: OpponentExpectedCounts,
) -> HandBeliefSensitivityDecision:
    """expected countだけを入力として同じTrack B rankingを評価する。"""
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("policy_input must be a PolicyInput")
    if not isinstance(opponent_counts, OpponentExpectedCounts):
        raise TypeError("opponent_counts must be an OpponentExpectedCounts")
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
                opponent_counts,
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
