"""リーチを受けた1向聴の押し引きを、手の価値でパラメータ化した実験候補（#377）。

仕様は`docs/riichi-response-candidate.md`。未昇格の実験候補で、Championは書き換えない。
exact parentは`PlacementAwareSpeedCallPolicy`。

## 動作

parentは他家リーチ下の非聴牌手を、手の価値を見ずに次のように扱う。

    共通現物がある   -> 共通現物だけへ絞る（降りる）
    共通現物がない   -> 1向聴なら全打牌から攻撃基準で選ぶ（押す）

この候補は、点数状況modeが`NORMAL`で、どの打牌でも聴牌にならず、打牌後の最小
向聴数がちょうど1の打牌判断でだけ、明示的な`RiichiResponseParameters`に従って
この2つを入れ替える。

A. 押し返し（`push_minimum_dora`がNoneでないとき）

    共通現物があり、自席が門前で、手のdora枚数が`push_minimum_dora`以上、
    残り山が`push_minimum_live_wall`以上なら、他家リーチがないものとして
    parentに打牌を選ばせる。

B. 降り（`fold_maximum_dora`がNoneでないとき）

    共通現物がなく、手のdora枚数が`fold_maximum_dora`以下なら、古典的危険度score
    （全リーチ者に対する最大値）が最小の打牌へ絞ってparentへ渡す。

dora枚数は、純手牌と自席の副露牌にある赤ドラとindicator由来doraの合計で、既存の
`_retained_concealed_dora_count`と同じ数え方をする。得点期待値ではない。

和了・立直・鳴き・Passの判断、聴牌時と2向聴以上の打牌はparentを継承し、変えない。
PolicyInput以外の状態やhidden informationは使用せず、Policy instanceに
cross-decision stateを保持しない。
"""

from dataclasses import dataclass, replace

from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies.genbutsu_defense_two_step_ukeire import (
    _common_genbutsu_tile_types,
    _opponent_riichi_players,
)
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    _classical_riichi_danger_score,
)
from lisjong.policies.placement_aware_speed_call import (
    GameSituationMode,
    PlacementAwareSpeedCallPolicy,
    _is_closed_hand,
    _own_meld_tiles,
    situation_mode,
)
from lisjong.policies.value_aware_two_step_ukeire import (
    _retained_concealed_dora_count,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.structural_efficiency import post_discard_concealed_hand

RIICHI_RESPONSE_SEARCH_SPACE_VERSION = "riichi-response-search-space-v1"
_IDENTITY_PREFIX = "placement-aware-speed-call-riichi-response-v1"


@dataclass(frozen=True, slots=True)
class RiichiResponseParameters:
    """押し返しAと降りBの閾値。Noneはその規則を使わないことを表す。"""

    push_minimum_dora: int | None
    push_minimum_live_wall: int
    fold_maximum_dora: int | None

    def __post_init__(self) -> None:
        for name in ("push_minimum_dora", "fold_maximum_dora"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be None or a non-negative int")
        if type(self.push_minimum_live_wall) is not int or (
            self.push_minimum_live_wall < 0
        ):
            raise ValueError("push_minimum_live_wall must be a non-negative int")

    @property
    def identity(self) -> str:
        """パラメータ値だけから決まる候補identity（class名から導出しない）。"""
        push = "none" if self.push_minimum_dora is None else self.push_minimum_dora
        fold = "none" if self.fold_maximum_dora is None else self.fold_maximum_dora
        return (
            f"{_IDENTITY_PREFIX}:push-dora-{push}"
            f":push-wall-{self.push_minimum_live_wall}:fold-dora-{fold}"
        )


_SEARCH_PUSH_MINIMUM_DORA = (None, 1, 2)
_SEARCH_PUSH_MINIMUM_LIVE_WALL = 24
_SEARCH_FOLD_MAXIMUM_DORA = (None, 0, 1)


def riichi_response_search_space() -> tuple[RiichiResponseParameters, ...]:
    """探索空間v1の全候補を固定順で返す。両規則とも使わない組合せは含めない。

    順序はpush（None, 1, 2）を外側、fold（None, 0, 1）を内側とする。
    """
    return tuple(
        RiichiResponseParameters(
            push_minimum_dora=push,
            push_minimum_live_wall=_SEARCH_PUSH_MINIMUM_LIVE_WALL,
            fold_maximum_dora=fold,
        )
        for push in _SEARCH_PUSH_MINIMUM_DORA
        for fold in _SEARCH_FOLD_MAXIMUM_DORA
        if push is not None or fold is not None
    )


def _without_opponent_riichi(policy_input: PolicyInput) -> PolicyInput:
    """他家のリーチ状態だけを外したPolicyInput（押し返しの打牌選択に使う）。"""
    players = tuple(
        player
        if index == int(policy_input.self_seat)
        else replace(
            player, riichi=RiichiState.NONE, riichi_declaration_discard_order=None
        )
        for index, player in enumerate(policy_input.players)
    )
    return replace(policy_input, players=players)


def _minimum_danger_actions(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, ...]:
    riichi_players = _opponent_riichi_players(policy_input)
    remaining_counts = derive_remaining_tile_inventory(
        policy_input
    ).remaining_tile_counts
    dangers = tuple(
        (
            action,
            max(
                _classical_riichi_danger_score(
                    action.tile.tile_type, opponent, remaining_counts
                ).total
                for opponent in riichi_players
            ),
        )
        for action in discard_actions
    )
    minimum = min(danger for _, danger in dangers)
    return tuple(action for action, danger in dangers if danger == minimum)


class RiichiResponsePlacementAwareSpeedCallPolicy(PlacementAwareSpeedCallPolicy):
    """Issue #377: リーチを受けた1向聴の、手の価値による押し返し / 降り。"""

    def __init__(self, parameters: RiichiResponseParameters) -> None:
        if not isinstance(parameters, RiichiResponseParameters):
            raise TypeError("parameters must be RiichiResponseParameters")
        super().__init__()
        self._parameters = parameters

    @property
    def parameters(self) -> RiichiResponseParameters:
        return self._parameters

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        riichi_players = _opponent_riichi_players(policy_input)
        if (
            not riichi_players
            or situation_mode(policy_input) is not GameSituationMode.NORMAL
        ):
            return super()._decide_discard(policy_input, discard_actions)
        concealed_tiles = policy_input.own_hand.concealed_tiles
        minimum_shanten = min(
            calculate_shanten(post_discard_concealed_hand(concealed_tiles, action.tile))
            for action in discard_actions
        )
        if minimum_shanten != 1:
            return super()._decide_discard(policy_input, discard_actions)

        parameters = self._parameters
        dora = _retained_concealed_dora_count(
            (*concealed_tiles, *_own_meld_tiles(policy_input)),
            policy_input.round.dora_indicators,
        )
        genbutsu = _common_genbutsu_tile_types(riichi_players)
        has_genbutsu = any(
            action.tile.tile_type in genbutsu for action in discard_actions
        )
        if has_genbutsu:
            if (
                parameters.push_minimum_dora is not None
                and _is_closed_hand(policy_input)
                and dora >= parameters.push_minimum_dora
                and policy_input.round.live_wall_tiles_remaining
                >= parameters.push_minimum_live_wall
            ):
                return super()._decide_discard(
                    _without_opponent_riichi(policy_input), discard_actions
                )
        elif (
            parameters.fold_maximum_dora is not None
            and dora <= parameters.fold_maximum_dora
        ):
            return super()._decide_discard(
                policy_input, _minimum_danger_actions(policy_input, discard_actions)
            )
        return super()._decide_discard(policy_input, discard_actions)


__all__ = [
    "RIICHI_RESPONSE_SEARCH_SPACE_VERSION",
    "RiichiResponseParameters",
    "RiichiResponsePlacementAwareSpeedCallPolicy",
    "riichi_response_search_space",
]
