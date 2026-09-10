"""安くて遠いChi/Ponだけを抑止するOpenHandYakuAwareCallPolicy派生Policy。

Issue #161のbounded experimental axisとして、exact
`OpenHandYakuAwareCallPolicy`をbaselineに1回だけ実行する。baselineが
winning、Riichi、discard、Kan、Pass、既存Yakuhai/open-yaku callとして返した
decisionはそのまま返し、baselineが実際に選んだChi/Ponが

    cheap AND far

の場合だけ、`DecisionContext.legal_actions`上の既存legal Passへ置換する。
抑止したselected callの代わりに別のChi/Ponを再探索・再順位付けする
alternate-call rescueは行わない。

far / cheapはいずれも、selected callのexisting
`_post_call_stable_hands()`（exact consume + current kuikaeを適用した
mandatory-discard後のstable concealed hand群）から、conservativeに

    far   = stable hands全体でのminimum shantenが2以上
    cheap = stable hands全体でのmaximum current visible value proxyが3han-equivalent未満

として判定する。1つでも`<=1`向聴、または`>=3`han-equivalentのstable hand
があれば抑止しない。

`current visible value proxy`はactual han、expected han、expected score、
call EVのいずれでもない、本Issue専用のsmall pure proxy evaluatorである。
既存`_yaku_route_value()` / `_retained_real_value()`の整数値をそのまま
actual hanとして流用せず、Issueが明示したTanyao / Honitsu / Chinitsu
compatibility、完成済み役牌（三元牌・自風・場風・ダブル風）、現在保有する
dora・赤ドラだけを積み上げる。
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies.hand_value_aware_two_step_ukeire import (
    _WIND_RANK,
    _own_melds,
    _seat_wind_rank,
    _yakuhai_han_value,
)
from lisjong.policies.open_hand_yaku_aware_call import OpenHandYakuAwareCallPolicy
from lisjong.policies.value_aware_two_step_ukeire import (
    _retained_concealed_dora_count,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    YakuhaiCallPolicyError,
    _CallAction,
    _post_call_stable_hands,
)
from lisjong.policy_contract.action import ChiAction, PassAction, PonAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import Tile, TileCategory

_FAR_SHANTEN_THRESHOLD = 2
_CHEAP_VALUE_THRESHOLD = 3
_TANYAO_MINIMUM_RANK = 2
_TANYAO_MAXIMUM_RANK = 8
_HONITSU_VALUE = 2
_CHINITSU_VALUE = 5
_TANYAO_VALUE = 1


@dataclass(frozen=True, slots=True)
class _CheapFarEvaluation:
    """selected Chi/Ponに対する1回のconservative guard判定。"""

    far: bool
    cheap: bool

    @property
    def suppress(self) -> bool:
        return self.far and self.cheap


def _current_visible_value_proxy(
    owned_tiles: Sequence[Tile],
    dora_indicators: Sequence[Tile],
    *,
    seat_wind_rank: int,
    round_wind_rank: int,
) -> int:
    """player-visibleな情報だけから1 stable handのcurrent value proxyを返す。"""
    if all(
        tile.tile_type.category is not TileCategory.HONOR
        and _TANYAO_MINIMUM_RANK <= tile.tile_type.rank <= _TANYAO_MAXIMUM_RANK
        for tile in owned_tiles
    ):
        route_value = _TANYAO_VALUE
    else:
        route_value = 0

    suited_categories = {
        tile.tile_type.category
        for tile in owned_tiles
        if tile.tile_type.category is not TileCategory.HONOR
    }
    has_honor = any(
        tile.tile_type.category is TileCategory.HONOR for tile in owned_tiles
    )
    if len(suited_categories) == 1:
        route_value += _HONITSU_VALUE if has_honor else _CHINITSU_VALUE

    tile_type_counts = Counter(tile.tile_type for tile in owned_tiles)
    completed_yakuhai_value = sum(
        _yakuhai_han_value(
            tile_type,
            seat_wind_rank=seat_wind_rank,
            round_wind_rank=round_wind_rank,
        )
        for tile_type, count in tile_type_counts.items()
        if tile_type.category is TileCategory.HONOR and count >= 3
    )

    # counts red + indicator-derived dora over whatever tiles are passed in;
    # callers here pass the combined concealed + meld tile set so that dora
    # held inside an already-open meld is not undercounted.
    owned_dora_value = _retained_concealed_dora_count(owned_tiles, dora_indicators)

    return route_value + completed_yakuhai_value + owned_dora_value


def _evaluate_cheap_far_call(
    policy_input: PolicyInput, action: _CallAction
) -> _CheapFarEvaluation:
    """selected callの全allowed stable handsをconservativeに評価する。"""
    stable_hands = _post_call_stable_hands(
        policy_input.own_hand.concealed_tiles, action
    )
    if not stable_hands:
        return _CheapFarEvaluation(far=False, cheap=False)

    far = (
        min(calculate_shanten(stable_hand) for stable_hand in stable_hands)
        >= _FAR_SHANTEN_THRESHOLD
    )

    current_meld_tiles = tuple(
        tile for meld in _own_melds(policy_input) for tile in meld.tiles
    )
    new_meld_tiles = (*action.consumed_tiles, action.called_tile)
    seat_wind_rank = _seat_wind_rank(policy_input)
    round_wind_rank = _WIND_RANK[policy_input.round.round_wind]
    dora_indicators = policy_input.round.dora_indicators

    cheap = (
        max(
            _current_visible_value_proxy(
                (*stable_hand, *current_meld_tiles, *new_meld_tiles),
                dora_indicators,
                seat_wind_rank=seat_wind_rank,
                round_wind_rank=round_wind_rank,
            )
            for stable_hand in stable_hands
        )
        < _CHEAP_VALUE_THRESHOLD
    )

    return _CheapFarEvaluation(far=far, cheap=cheap)


class CheapFarGuardOpenHandYakuAwareCallPolicy(OpenHandYakuAwareCallPolicy):
    """安くて遠いselected Chi/PonだけをPassへ置換するexperimental Policy。"""

    def _decide(self, decision: DecisionContext) -> PolicyDecision:
        parent_decision = super()._decide(decision)
        if not isinstance(parent_decision.action, (ChiAction, PonAction)):
            return parent_decision

        if not _evaluate_cheap_far_call(
            decision.input, parent_decision.action
        ).suppress:
            return parent_decision

        pass_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, PassAction)
        )
        if not pass_actions:
            raise YakuhaiCallPolicyError(
                "cheap-far guard suppressed the selected call, but no explicit "
                "PassAction is legal"
            )
        return PolicyDecision(action=pass_actions[0])
