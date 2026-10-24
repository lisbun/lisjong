"""単独副露者の河の現物を、同向聴・攻撃proxy非減の範囲で優先する実験候補（#343）。

仕様は`docs/single-open-genbutsu-candidate.md`。未昇格の実験候補で、Championは書き換えない。
exact parentは`PlacementAwareSpeedCallPolicy`。

## 動作

parentの打牌（C0）を先に得てから、次のgateをすべて満たす通常打牌の判断でだけ、C0を
単独副露者の河にある牌種（現物）へ替えることを検討する。

- 自席のツモ後の打牌（`drawn_tile`あり。鳴き直後の打牌は対象外）
- 自席が非リーチ・門前（暗槓だけは門前扱い）
- 他家に宣言済みを含むリーチ者がいない
- リーチしておらず暗槓以外の副露がある他家（副露者）がちょうど1人
- 点数状況modeが`GameSituationMode.NORMAL`
- どの合法打牌でも聴牌にならない（C0の打牌後向聴数も1以上）

C0がすでに現物なら維持する。そうでなければ、現物の合法打牌のうち、C0と比べて打牌後向聴数が
同じで、現在受け入れ・retained real value・yaku route valueがいずれも下がらないものだけを
代替候補とする。候補がなければC0を維持し、あれば現在受け入れ・retained real value・
yaku route valueの大きい順、canonicalな打牌順で1つ選ぶ。

各値は`structural_efficiency`と`hand_value_aware_two_step_ukeire`と同じ定義で、C0と現物の
合法打牌についてだけ計算する。valueは得点期待値ではなく既存のproxyである。

河の現物はその副露者に現在ロンされない根拠であり、他の2人への安全、将来の安全、収支の改善を
保証しない。和了・立直・鳴き・Passの判断と、打牌以外の応答はparentを継承し、変えない。
"""

from dataclasses import dataclass
from enum import Enum, auto

from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _own_melds,
    _retained_real_value,
    _yaku_route_value,
)
from lisjong.policies.placement_aware_speed_call import (
    GameSituationMode,
    PlacementAwareSpeedCallPolicy,
    _is_closed_hand,
    situation_mode,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.tile import Tile, TileType, tile_sort_key
from lisjong.structural_efficiency import (
    PostDiscardStructuralEvaluation,
    StructuralShantenEvaluator,
    discard_action_sort_key,
    evaluate_post_discard_hands,
    known_tile_counts,
    ukeire_count,
)


class SingleOpenGenbutsuStage(Enum):
    """gateのどこでC0を維持したか、または打牌を替えたか。"""

    NOT_NORMAL_TURN = auto()
    SELF_RIICHI = auto()
    SELF_OPEN_HAND = auto()
    OPPONENT_RIICHI = auto()
    NO_OPEN_OPPONENT = auto()
    MULTIPLE_OPEN_OPPONENTS = auto()
    NOT_NORMAL_SITUATION = auto()
    TENPAI_REACHABLE = auto()
    PARENT_ALREADY_GENBUTSU = auto()
    NO_GENBUTSU_ALTERNATIVE = auto()
    NO_QUALIFYING_ALTERNATIVE = auto()
    SWITCHED = auto()


@dataclass(frozen=True, slots=True)
class GenbutsuCandidateEvaluation:
    """C0、または現物の合法打牌1つについて実際に計算した比較値。"""

    action: DiscardAction
    post_discard_shanten: int
    current_ukeire_count: int
    retained_real_value: int
    yaku_route_value: int


@dataclass(frozen=True, slots=True)
class SingleOpenGenbutsuAnalysis(AnalysisTrace):
    """1回の打牌判断の記録。対象外の判断でも除外理由を残す。

    `open_opponent_seat`・`genbutsu_tile_types`は副露者が1人に決まった段階から、
    `parent_in_genbutsu`はC0の現物判定に進んだ段階から、`parent_evaluation`・
    `alternative_evaluations`（C0以外の現物の合法打牌）・`qualifying_actions`（そのうち
    代替候補の条件を満たすもの）は比較値を計算した段階から記録する。未計算の値は0で埋めない。
    """

    stage: SingleOpenGenbutsuStage
    parent_action: DiscardAction
    selected_action: DiscardAction
    action_changed: bool
    open_opponent_seat: int | None
    genbutsu_tile_types: tuple[TileType, ...]
    parent_in_genbutsu: bool | None
    parent_evaluation: GenbutsuCandidateEvaluation | None
    alternative_evaluations: tuple[GenbutsuCandidateEvaluation, ...]
    qualifying_actions: tuple[DiscardAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage, SingleOpenGenbutsuStage):
            raise TypeError("stage must be a SingleOpenGenbutsuStage")
        if self.action_changed != (self.parent_action != self.selected_action):
            raise ValueError(
                "action_changed must describe selected_action against parent_action"
            )
        if self.action_changed != (self.stage is SingleOpenGenbutsuStage.SWITCHED):
            raise ValueError("only the SWITCHED stage may change the action")
        if self.action_changed and self.selected_action not in self.qualifying_actions:
            raise ValueError("selected_action must be one of qualifying_actions")


def open_opponent_seats(policy_input: PolicyInput) -> tuple[int, ...]:
    """リーチしておらず、暗槓以外の副露がある他家の席index。"""
    self_index = int(policy_input.self_seat)
    return tuple(
        index
        for index, player in enumerate(policy_input.players)
        if index != self_index
        and player.riichi is RiichiState.NONE
        and any(meld.kind is not MeldKind.ANKAN for meld in player.melds)
    )


def river_genbutsu_tile_types(
    policy_input: PolicyInput, seat: int
) -> frozenset[TileType]:
    """その席自身の河にある牌種（鳴かれた打牌の記録を含む。赤/通常5は同牌種）。"""
    return frozenset(
        discard.tile.tile_type for discard in policy_input.players[seat].discards
    )


def _evaluate(
    policy_input: PolicyInput,
    structural: PostDiscardStructuralEvaluation,
    known_counts: dict[TileType, int],
    evaluator: StructuralShantenEvaluator,
) -> GenbutsuCandidateEvaluation:
    return GenbutsuCandidateEvaluation(
        action=structural.action,
        post_discard_shanten=structural.post_discard_shanten,
        current_ukeire_count=ukeire_count(
            structural.post_discard_hand,
            known_counts,
            structural.post_discard_shanten,
            evaluator,
        ),
        retained_real_value=_retained_real_value(
            structural.post_discard_hand, policy_input
        ),
        yaku_route_value=_yaku_route_value(
            structural.post_discard_hand, _own_melds(policy_input)
        ),
    )


def _qualifies(
    candidate: GenbutsuCandidateEvaluation, parent: GenbutsuCandidateEvaluation
) -> bool:
    return (
        candidate.post_discard_shanten == parent.post_discard_shanten
        and candidate.current_ukeire_count >= parent.current_ukeire_count
        and candidate.retained_real_value >= parent.retained_real_value
        and candidate.yaku_route_value >= parent.yaku_route_value
    )


def _gate_stage(policy_input: PolicyInput) -> SingleOpenGenbutsuStage | None:
    """副露者の数より前のgateで対象外なら、その理由を返す。"""
    if policy_input.own_hand.drawn_tile is None:
        return SingleOpenGenbutsuStage.NOT_NORMAL_TURN
    self_index = int(policy_input.self_seat)
    if policy_input.players[self_index].riichi is not RiichiState.NONE:
        return SingleOpenGenbutsuStage.SELF_RIICHI
    if not _is_closed_hand(policy_input):
        return SingleOpenGenbutsuStage.SELF_OPEN_HAND
    if any(
        player.riichi is not RiichiState.NONE
        for index, player in enumerate(policy_input.players)
        if index != self_index
    ):
        return SingleOpenGenbutsuStage.OPPONENT_RIICHI
    return None


def decide_single_open_genbutsu(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
    parent_decision: PolicyDecision,
) -> PolicyDecision:
    """対象の判断だけC0を単独副露者の現物へ替え、全判断で記録を付けて返す。"""
    parent_action = parent_decision.action
    if not isinstance(parent_action, DiscardAction):
        raise TypeError("parent discard decision must select a DiscardAction")

    def finish(
        stage: SingleOpenGenbutsuStage,
        selected: DiscardAction = parent_action,
        *,
        open_seat: int | None = None,
        genbutsu: frozenset[TileType] = frozenset(),
        parent_in_genbutsu: bool | None = None,
        parent_evaluation: GenbutsuCandidateEvaluation | None = None,
        alternatives: tuple[GenbutsuCandidateEvaluation, ...] = (),
        qualifying: tuple[DiscardAction, ...] = (),
    ) -> PolicyDecision:
        return PolicyDecision(
            action=selected,
            analysis=SingleOpenGenbutsuAnalysis(
                stage=stage,
                parent_action=parent_action,
                selected_action=selected,
                action_changed=selected != parent_action,
                open_opponent_seat=open_seat,
                genbutsu_tile_types=tuple(
                    sorted(
                        genbutsu,
                        key=lambda tile_type: tile_sort_key(Tile(tile_type)),
                    )
                ),
                parent_in_genbutsu=parent_in_genbutsu,
                parent_evaluation=parent_evaluation,
                alternative_evaluations=alternatives,
                qualifying_actions=qualifying,
            ),
        )

    gate = _gate_stage(policy_input)
    if gate is not None:
        return finish(gate)
    open_seats = open_opponent_seats(policy_input)
    if not open_seats:
        return finish(SingleOpenGenbutsuStage.NO_OPEN_OPPONENT)
    if len(open_seats) > 1:
        return finish(SingleOpenGenbutsuStage.MULTIPLE_OPEN_OPPONENTS)
    (open_seat,) = open_seats
    genbutsu = river_genbutsu_tile_types(policy_input, open_seat)
    recorded = {"open_seat": open_seat, "genbutsu": genbutsu}
    if situation_mode(policy_input) is not GameSituationMode.NORMAL:
        return finish(SingleOpenGenbutsuStage.NOT_NORMAL_SITUATION, **recorded)

    evaluator = StructuralShantenEvaluator()
    structurals = evaluate_post_discard_hands(policy_input, discard_actions, evaluator)
    parent_structural = next(
        (row for row in structurals if row.action == parent_action), None
    )
    if parent_structural is None:
        raise ValueError("parent discard action is not one of the legal discards")
    if min(row.post_discard_shanten for row in structurals) == 0:
        return finish(SingleOpenGenbutsuStage.TENPAI_REACHABLE, **recorded)

    parent_in_genbutsu = parent_action.tile.tile_type in genbutsu
    if parent_in_genbutsu:
        return finish(
            SingleOpenGenbutsuStage.PARENT_ALREADY_GENBUTSU,
            parent_in_genbutsu=True,
            **recorded,
        )
    alternative_rows = sorted(
        (row for row in structurals if row.action.tile.tile_type in genbutsu),
        key=lambda row: discard_action_sort_key(row.action),
    )
    if not alternative_rows:
        return finish(
            SingleOpenGenbutsuStage.NO_GENBUTSU_ALTERNATIVE,
            parent_in_genbutsu=False,
            **recorded,
        )

    known_counts = known_tile_counts(policy_input)
    parent_evaluation = _evaluate(
        policy_input, parent_structural, known_counts, evaluator
    )
    alternatives = tuple(
        _evaluate(policy_input, row, known_counts, evaluator)
        for row in alternative_rows
    )
    qualifying = tuple(
        candidate
        for candidate in alternatives
        if _qualifies(candidate, parent_evaluation)
    )
    evaluated = {
        "parent_in_genbutsu": False,
        "parent_evaluation": parent_evaluation,
        "alternatives": alternatives,
        "qualifying": tuple(candidate.action for candidate in qualifying),
        **recorded,
    }
    if not qualifying:
        return finish(SingleOpenGenbutsuStage.NO_QUALIFYING_ALTERNATIVE, **evaluated)
    selected = min(
        qualifying,
        key=lambda candidate: (
            -candidate.current_ukeire_count,
            -candidate.retained_real_value,
            -candidate.yaku_route_value,
            discard_action_sort_key(candidate.action),
        ),
    ).action
    return finish(SingleOpenGenbutsuStage.SWITCHED, selected, **evaluated)


class SingleOpenGenbutsuPlacementAwareSpeedCallPolicy(PlacementAwareSpeedCallPolicy):
    """#343: 単独副露者の河の現物を同向聴・攻撃proxy非減で優先する（未昇格の実験候補）。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        return decide_single_open_genbutsu(
            policy_input,
            discard_actions,
            super()._decide_discard(policy_input, discard_actions),
        )


__all__ = [
    "GenbutsuCandidateEvaluation",
    "SingleOpenGenbutsuAnalysis",
    "SingleOpenGenbutsuPlacementAwareSpeedCallPolicy",
    "SingleOpenGenbutsuStage",
    "decide_single_open_genbutsu",
    "open_opponent_seats",
    "river_genbutsu_tile_types",
]
