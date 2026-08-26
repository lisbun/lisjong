"""役・ドラの軽量な手牌価値を加えるTwoStepUkeire派生Policy。

`HandValueAwareTwoStepUkeirePolicy`は既存Policyを変更せず、real legal discardを
次のlexicographic priorityで比較する独立したexperimental generationである。

    打牌後向聴数
    > 現在受け入れ
    > retained real value
    > yaku route value
    > 2段階受け入れ
    > stable tie-break

`retained_real_value`は、打牌後の自手に残る公開indicator由来dora、赤ドラ、
完成済み役牌翻相当値の和である。`yaku_route_value`はtanyao / honitsu /
chinitsu compatibilityだけを表す軽量heuristicであり、actual han、expected han、
expected score、expected valueのいずれでもない。

value-aware化するのは現在decisionのreal discardだけである。第2段のhypothetical
draw / discard branchは既存TwoStepのstructural semanticsをそのまま使う。
winning action、Always Riichi、pass、fallbackは基底classから継承する。
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from lisjong.policies.two_step_ukeire import (
    TwoStepUkeirePolicy,
    _DecisionShantenEvaluator,
    _discard_action_sort_key,
    _known_tile_counts,
    _remove_one_matching_tile,
    _second_step_score,
    _ukeire_count,
)
from lisjong.policies.value_aware_two_step_ukeire import (
    _retained_concealed_dora_count,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import Tile, TileCategory
from lisjong.policy_contract.wind import Wind

_WIND_RANK = {
    Wind.EAST: 1,
    Wind.SOUTH: 2,
    Wind.WEST: 3,
    Wind.NORTH: 4,
}
_DRAGON_MINIMUM_RANK = 5
_YAKUHAI_MELD_KINDS = frozenset(
    {MeldKind.PON, MeldKind.DAIMINKAN, MeldKind.ANKAN, MeldKind.KAKAN}
)


@dataclass(frozen=True, slots=True)
class HandValueCandidateEvaluation:
    """HandValueAwareが実際に使用した1打牌候補のsemantic評価値。"""

    action: DiscardAction
    post_discard_shanten: int
    current_ukeire_count: int | None
    retained_real_value: int | None
    yaku_route_value: int | None
    second_step_ukeire_score: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.action, DiscardAction):
            raise TypeError("action must be a DiscardAction")
        if type(self.post_discard_shanten) is not int:
            raise TypeError("post_discard_shanten must be an int")
        for field_name in (
            "current_ukeire_count",
            "retained_real_value",
            "yaku_route_value",
            "second_step_ukeire_score",
        ):
            value = getattr(self, field_name)
            if value is not None and type(value) is not int:
                raise TypeError(f"{field_name} must be an int or None")


@dataclass(frozen=True, slots=True)
class HandValueAwareTwoStepUkeireAnalysis(AnalysisTrace):
    """1 decisionで実際に生成したhand-value候補評価のtyped payload。"""

    candidate_evaluations: tuple[HandValueCandidateEvaluation, ...]

    def __post_init__(self) -> None:
        try:
            evaluations = tuple(self.candidate_evaluations)
        except TypeError:
            raise TypeError("candidate_evaluations must be an iterable") from None
        if any(
            not isinstance(evaluation, HandValueCandidateEvaluation)
            for evaluation in evaluations
        ):
            raise TypeError(
                "candidate_evaluations must contain only "
                "HandValueCandidateEvaluation values"
            )
        if not evaluations:
            raise ValueError("candidate_evaluations must not be empty")
        object.__setattr__(self, "candidate_evaluations", evaluations)


@dataclass(slots=True)
class _HandValueDiscardCandidateWork:
    action: DiscardAction
    post_discard_hand: list[Tile]
    post_discard_shanten: int
    current_ukeire_count: int | None = None
    retained_real_value: int | None = None
    yaku_route_value: int | None = None
    second_step_ukeire_score: int | None = None

    def snapshot(self) -> HandValueCandidateEvaluation:
        return HandValueCandidateEvaluation(
            action=self.action,
            post_discard_shanten=self.post_discard_shanten,
            current_ukeire_count=self.current_ukeire_count,
            retained_real_value=self.retained_real_value,
            yaku_route_value=self.yaku_route_value,
            second_step_ukeire_score=self.second_step_ukeire_score,
        )


def _own_melds(policy_input: PolicyInput) -> tuple[PublicMeld, ...]:
    return policy_input.players[int(policy_input.self_seat)].melds


def _seat_wind_rank(policy_input: PolicyInput) -> int:
    return (int(policy_input.self_seat) - int(policy_input.round.dealer_seat)) % 4 + 1


def _completed_yakuhai_value(
    post_discard_hand: Sequence[Tile],
    melds: Sequence[PublicMeld],
    *,
    seat_wind_rank: int,
    round_wind_rank: int,
) -> int:
    """完成済みの三元牌・自風・場風groupが持つ翻相当値を返す。"""
    concealed_counts = Counter(tile.tile_type for tile in post_discard_hand)
    completed_types = {
        tile_type
        for tile_type, count in concealed_counts.items()
        if tile_type.category is TileCategory.HONOR and count >= 3
    }
    completed_types.update(
        meld.tiles[0].tile_type
        for meld in melds
        if meld.kind in _YAKUHAI_MELD_KINDS
        and meld.tiles[0].tile_type.category is TileCategory.HONOR
    )

    value = 0
    for tile_type in completed_types:
        if tile_type.rank >= _DRAGON_MINIMUM_RANK:
            value += 1
        if tile_type.rank == seat_wind_rank:
            value += 1
        if tile_type.rank == round_wind_rank:
            value += 1
    return value


def _retained_real_value(
    post_discard_hand: Sequence[Tile], policy_input: PolicyInput
) -> int:
    """post-discard自手のretained dora / aka-dora / yakuhai価値を返す。"""
    melds = _own_melds(policy_input)
    retained_tiles = tuple(post_discard_hand) + tuple(
        tile for meld in melds for tile in meld.tiles
    )
    dora_value = _retained_concealed_dora_count(
        retained_tiles, policy_input.round.dora_indicators
    )
    yakuhai_value = _completed_yakuhai_value(
        post_discard_hand,
        melds,
        seat_wind_rank=_seat_wind_rank(policy_input),
        round_wind_rank=_WIND_RANK[policy_input.round.round_wind],
    )
    return dora_value + yakuhai_value


def _yaku_route_value(
    post_discard_hand: Sequence[Tile], melds: Sequence[PublicMeld]
) -> int:
    """tanyao / honitsu / chinitsu compatibilityの軽量heuristicを返す。"""
    tiles = tuple(post_discard_hand) + tuple(
        tile for meld in melds for tile in meld.tiles
    )
    if not tiles:
        return 0

    value = 0
    if all(
        tile.tile_type.category is not TileCategory.HONOR
        and 2 <= tile.tile_type.rank <= 8
        for tile in tiles
    ):
        value += 1

    suited_categories = {
        tile.tile_type.category
        for tile in tiles
        if tile.tile_type.category is not TileCategory.HONOR
    }
    if len(suited_categories) == 1:
        if any(tile.tile_type.category is TileCategory.HONOR for tile in tiles):
            value += 2
        else:
            value += 3
    return value


def _evaluate_and_choose_discard(
    policy_input: PolicyInput, discard_actions: tuple[DiscardAction, ...]
) -> tuple[DiscardAction, tuple[HandValueCandidateEvaluation, ...]]:
    """selectionとanalysisで共有する1回のstaged candidate evaluation。"""
    known_counts = _known_tile_counts(policy_input)
    evaluator = _DecisionShantenEvaluator()
    concealed_tiles = policy_input.own_hand.concealed_tiles
    evaluated = tuple(
        _HandValueDiscardCandidateWork(
            action=action,
            post_discard_hand=remaining_hand,
            post_discard_shanten=evaluator.calculate(remaining_hand),
        )
        for action in discard_actions
        for remaining_hand in (_remove_one_matching_tile(concealed_tiles, action.tile),)
    )

    minimum_shanten = min(candidate.post_discard_shanten for candidate in evaluated)
    shanten_finalists = tuple(
        candidate
        for candidate in evaluated
        if candidate.post_discard_shanten == minimum_shanten
    )
    for candidate in shanten_finalists:
        candidate.current_ukeire_count = _ukeire_count(
            candidate.post_discard_hand,
            known_counts,
            minimum_shanten,
            evaluator,
        )
    maximum_current_ukeire = max(
        candidate.current_ukeire_count for candidate in shanten_finalists
    )
    ukeire_finalists = tuple(
        candidate
        for candidate in shanten_finalists
        if candidate.current_ukeire_count == maximum_current_ukeire
    )

    if len(ukeire_finalists) == 1:
        selected = ukeire_finalists[0].action
    else:
        for candidate in ukeire_finalists:
            candidate.retained_real_value = _retained_real_value(
                candidate.post_discard_hand, policy_input
            )
        maximum_real_value = max(
            candidate.retained_real_value for candidate in ukeire_finalists
        )
        real_value_finalists = tuple(
            candidate
            for candidate in ukeire_finalists
            if candidate.retained_real_value == maximum_real_value
        )

        if len(real_value_finalists) == 1:
            selected = real_value_finalists[0].action
        else:
            melds = _own_melds(policy_input)
            for candidate in real_value_finalists:
                candidate.yaku_route_value = _yaku_route_value(
                    candidate.post_discard_hand, melds
                )
            maximum_route_value = max(
                candidate.yaku_route_value for candidate in real_value_finalists
            )
            route_finalists = tuple(
                candidate
                for candidate in real_value_finalists
                if candidate.yaku_route_value == maximum_route_value
            )

            if len(route_finalists) == 1:
                selected = route_finalists[0].action
            elif minimum_shanten == 0:
                selected = min(
                    (candidate.action for candidate in route_finalists),
                    key=_discard_action_sort_key,
                )
            else:
                for candidate in route_finalists:
                    candidate.second_step_ukeire_score = _second_step_score(
                        candidate.post_discard_hand,
                        known_counts,
                        minimum_shanten,
                        evaluator,
                    )
                maximum_second_step = max(
                    candidate.second_step_ukeire_score for candidate in route_finalists
                )
                selected = min(
                    (
                        candidate.action
                        for candidate in route_finalists
                        if candidate.second_step_ukeire_score == maximum_second_step
                    ),
                    key=_discard_action_sort_key,
                )

    snapshots = tuple(
        candidate.snapshot()
        for candidate in sorted(
            evaluated,
            key=lambda candidate: _discard_action_sort_key(candidate.action),
        )
    )
    return selected, snapshots


class HandValueAwareTwoStepUkeirePolicy(TwoStepUkeirePolicy):
    """牌効率をhard priorityとし、同速候補だけを軽量hand valueで比較する。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        selected, evaluations = _evaluate_and_choose_discard(
            policy_input, discard_actions
        )
        return PolicyDecision(
            action=selected,
            analysis=HandValueAwareTwoStepUkeireAnalysis(
                candidate_evaluations=evaluations
            ),
        )
