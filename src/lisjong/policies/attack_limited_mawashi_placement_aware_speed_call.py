"""Championの単独リーチ・門前・1向聴・現物なしFALLBACKへ、攻撃制限付き回し打ちを加える実験候補。

lisbun/lisjong#249（親: #236）。Championを直接書き換えず、`PlacementAwareSpeedCallPolicy`を
exact parentとする。対象の判断でだけ、Championが選ぶ打牌（C0）を基準に、攻撃面の評価が
C0から一定以上は下がらない打牌へ候補を絞り、その中で危険度が最小の打牌を選ぶ。
対象外の判断は、同じ入力に対してChampionの決定をそのまま返す。

危険度は入れ替えられる。このmoduleの`ClassicalAttackLimitedMawashiPolicy`は古典的危険度score
（固定weightの相対値）を使う比較版であり、#245の待ち推定を使う版は
`lisjong.learning.riichi_wait_mawashi_policy`にある。どちらの危険度も放銃確率・期待放銃点
ではない。

ロンされない牌の扱いはPolicy側が持ち、危険度の生の値とは分ける（`ron_safe_tile_types()`）。
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    _evaluate_completion_masses,
    _FiniteHorizonEvaluator,
    _root_remaining_counts,
)
from lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware import (
    DefenseFilterBranch,
    _evaluate_defense_filter,
)
from lisjong.policies.hand_value_aware_two_step_ukeire import _retained_real_value
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    _classical_riichi_danger_score,
)
from lisjong.policies.placement_aware_speed_call import (
    GameSituationMode,
    PlacementAwareSpeedCallPolicy,
    _is_closed_hand,
    situation_mode,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.tile import TileType
from lisjong.structural_efficiency import (
    StructuralShantenEvaluator,
    discard_action_sort_key,
    evaluate_post_discard_hands,
    known_tile_counts,
    ukeire_count,
)

ATTACK_LIMIT_NUMERATOR = 3
ATTACK_LIMIT_DENOMINATOR = 4
"""受け入れ枚数とFH completion massは、C0の3/4以上を保つ打牌だけを候補に残す。"""

RawDanger = Callable[[PolicyInput, int], Mapping[TileType, float]]
"""(判断の入力, リーチ者の席index) -> 牌種ごとの危険度の生の値。小さいほど安全。"""


@dataclass(frozen=True, slots=True)
class MawashiCandidateEvaluation:
    """対象の判断で、1つの合法打牌について実際に計算した値。

    `danger`は、ロンされない牌を0にした後の値である（生の値ではない）。
    """

    action: DiscardAction
    post_discard_shanten: int
    ukeire_count: int
    completion_mass: int
    retained_real_value: int
    within_attack_limit: bool
    danger: float


@dataclass(frozen=True, slots=True)
class AttackLimitedMawashiAnalysis(AnalysisTrace):
    """対象の判断1回の記録。対象外の判断では作らない。

    `selected_action == c0_action`なら、Championの打牌を維持した判断である。
    """

    c0_action: DiscardAction
    selected_action: DiscardAction
    candidate_evaluations: tuple[MawashiCandidateEvaluation, ...]


def single_riichi_opponent(policy_input: PolicyInput) -> int | None:
    """自分が非リーチで、他家のちょうど1人がリーチ中なら、その席indexを返す。"""
    seat = int(policy_input.self_seat)
    if policy_input.players[seat].riichi is not RiichiState.NONE:
        return None
    opponents = [
        index
        for index, player in enumerate(policy_input.players)
        if index != seat and player.riichi is not RiichiState.NONE
    ]
    return opponents[0] if len(opponents) == 1 else None


def ron_safe_tile_types(
    policy_input: PolicyInput, riichi_seat: int
) -> frozenset[TileType]:
    """公開情報だけから、このリーチ者にロンされないと確定する牌種を返す。

    - リーチ者自身の捨て牌にある牌種（捨て牌フリテン）
    - リーチ者の最後の打牌より後に他家が切った牌種（待ちなら見逃しフリテン、待ちでなければ
      ロンできない）

    スジ・壁などの推測は含めない。観測していないフリテンを確定扱いしない。
    """
    riichi_player = policy_input.players[riichi_seat]
    safe = {discard.tile.tile_type for discard in riichi_player.discards}
    if riichi_player.discards:
        last_order = max(discard.order for discard in riichi_player.discards)
        safe.update(
            discard.tile.tile_type
            for index, player in enumerate(policy_input.players)
            if index != riichi_seat
            for discard in player.discards
            if discard.order > last_order
        )
    return frozenset(safe)


def mawashi_target_riichi_seat(
    policy_input: PolicyInput, discard_actions: tuple[DiscardAction, ...]
) -> int | None:
    """対象の判断ならリーチ者の席indexを返し、対象外なら`None`を返す。

    対象は、他家1人がリーチ・自分は非リーチ・門前・合法打牌が2牌種以上で、Championが
    共通現物なしの非聴牌FOLD（`FOLD_FALLBACK_ALL_LEGAL`）と分類し、打牌後の最小向聴が1の判断。
    オーラストップ目の事前守備（`ALL_LAST_TOP_SPEED`）が働く判断は、Championの処理を優先して
    対象外とする。
    """
    riichi_seat = single_riichi_opponent(policy_input)
    if riichi_seat is None or not _is_closed_hand(policy_input):
        return None
    if len({action.tile.tile_type for action in discard_actions}) < 2:
        return None
    if situation_mode(policy_input) is GameSituationMode.ALL_LAST_TOP_SPEED:
        return None
    branch = _evaluate_defense_filter(policy_input, discard_actions).branch
    if branch is not DefenseFilterBranch.FOLD_FALLBACK_ALL_LEGAL:
        return None
    evaluated = evaluate_post_discard_hands(
        policy_input, discard_actions, StructuralShantenEvaluator()
    )
    if min(candidate.post_discard_shanten for candidate in evaluated) != 1:
        return None
    return riichi_seat


def classical_raw_danger(
    policy_input: PolicyInput, riichi_seat: int
) -> dict[TileType, float]:
    """自分の手牌にある牌種の古典的危険度score（Championのmechanism守備と同じ値）。"""
    remaining = derive_remaining_tile_inventory(policy_input).remaining_tile_counts
    riichi_player = policy_input.players[riichi_seat]
    return {
        tile_type: _classical_riichi_danger_score(
            tile_type, riichi_player, remaining
        ).total
        for tile_type in {
            tile.tile_type for tile in policy_input.own_hand.concealed_tiles
        }
    }


def _evaluate_candidates(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
    c0_action: DiscardAction,
    dangers: Mapping[TileType, float],
) -> tuple[MawashiCandidateEvaluation, ...]:
    evaluator = StructuralShantenEvaluator()
    known_counts = known_tile_counts(policy_input)
    masses = {
        evaluation.action: evaluation.completion_mass
        for evaluation in _evaluate_completion_masses(
            policy_input,
            discard_actions,
            _root_remaining_counts(policy_input),
            DEFAULT_HORIZON,
            _FiniteHorizonEvaluator(),
        )
    }
    rows = tuple(
        (
            structural.action,
            structural.post_discard_shanten,
            ukeire_count(
                structural.post_discard_hand,
                known_counts,
                structural.post_discard_shanten,
                evaluator,
            ),
            masses[structural.action],
            _retained_real_value(structural.post_discard_hand, policy_input),
        )
        for structural in evaluate_post_discard_hands(
            policy_input, discard_actions, evaluator
        )
    )
    _, c0_shanten, c0_ukeire, c0_mass, c0_value = next(
        row for row in rows if row[0] == c0_action
    )
    return tuple(
        MawashiCandidateEvaluation(
            action=action,
            post_discard_shanten=shanten,
            ukeire_count=ukeire,
            completion_mass=mass,
            retained_real_value=value,
            within_attack_limit=(
                shanten == c0_shanten
                and ATTACK_LIMIT_DENOMINATOR * ukeire
                >= ATTACK_LIMIT_NUMERATOR * c0_ukeire
                and ATTACK_LIMIT_DENOMINATOR * mass >= ATTACK_LIMIT_NUMERATOR * c0_mass
                and value >= c0_value
            ),
            danger=dangers[action.tile.tile_type],
        )
        for action, shanten, ukeire, mass, value in rows
    )


def decide_attack_limited_mawashi(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
    c0_decision: PolicyDecision,
    raw_danger: RawDanger,
) -> PolicyDecision:
    """対象の判断だけ、攻撃制限の範囲で危険度が最小の打牌へ替える。

    対象外では`c0_decision`をそのまま返す。対象でも、C0より危険度が厳密に小さい候補が
    制限内になければC0を維持する（C0は必ず制限内なので、候補集合は空にならない）。
    危険度が同じ候補は、FH completion mass・受け入れ枚数・手牌価値の大きい順、最後に
    canonicalな打牌順で選ぶ。
    """
    riichi_seat = mawashi_target_riichi_seat(policy_input, discard_actions)
    c0_action = c0_decision.action
    if riichi_seat is None or c0_action not in discard_actions:
        return c0_decision

    raw = raw_danger(policy_input, riichi_seat)
    safe = ron_safe_tile_types(policy_input, riichi_seat)
    dangers = {
        tile_type: 0.0 if tile_type in safe else raw[tile_type]
        for tile_type in {action.tile.tile_type for action in discard_actions}
    }
    evaluations = _evaluate_candidates(
        policy_input, discard_actions, c0_action, dangers
    )
    allowed = tuple(
        evaluation for evaluation in evaluations if evaluation.within_attack_limit
    )
    minimum = min(evaluation.danger for evaluation in allowed)
    if dangers[c0_action.tile.tile_type] <= minimum:
        selected = c0_action
    else:
        selected = min(
            (evaluation for evaluation in allowed if evaluation.danger == minimum),
            key=lambda evaluation: (
                -evaluation.completion_mass,
                -evaluation.ukeire_count,
                -evaluation.retained_real_value,
                discard_action_sort_key(evaluation.action),
            ),
        ).action
    return PolicyDecision(
        action=selected,
        analysis=AttackLimitedMawashiAnalysis(
            c0_action=c0_action,
            selected_action=selected,
            candidate_evaluations=evaluations,
        ),
    )


class ClassicalAttackLimitedMawashiPolicy(PlacementAwareSpeedCallPolicy):
    """#249: 危険度に古典的危険度scoreを使う比較版（未昇格の実験候補）。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        return decide_attack_limited_mawashi(
            policy_input,
            discard_actions,
            super()._decide_discard(policy_input, discard_actions),
            classical_raw_danger,
        )


__all__ = [
    "ATTACK_LIMIT_DENOMINATOR",
    "ATTACK_LIMIT_NUMERATOR",
    "AttackLimitedMawashiAnalysis",
    "ClassicalAttackLimitedMawashiPolicy",
    "MawashiCandidateEvaluation",
    "classical_raw_danger",
    "decide_attack_limited_mawashi",
    "mawashi_target_riichi_seat",
    "ron_safe_tile_types",
    "single_riichi_opponent",
]
