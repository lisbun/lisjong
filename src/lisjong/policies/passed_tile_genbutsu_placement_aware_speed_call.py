"""単独リーチ・非聴牌FOLDの打牌に、リーチ者の確定安全牌を現物として使う候補（lisbun/lisjong#312）。

設計は`docs/riichi-passed-tile-genbutsu-design.md`、候補の仕様は
`docs/passed-tile-genbutsu-candidate.md`。未昇格の実験候補で、Championは書き換えない。

## 動作

他家のリーチ者（宣言済みを含む）がちょうど1人で、現行のFOLD（どの合法打牌でも聴牌に
できない）のとき、Championが「共通現物」として使う河の牌種を、そのリーチ者の確定安全牌
`safe(r) = river(r) ∪ passed(r)`（`lisjong.belief.riichi_safe_tiles`）へ置き換える。
合法打牌に確定安全牌があれば、確定安全牌へ絞り、攻撃側の評価
（FH completion mass → 手牌価値）で選ぶ。

Championの打牌経路で共通現物を使う3箇所（オーラストップ目の事前絞り込み、mechanism defense、
FOLD時の絞り込み）は、確定安全牌が合法打牌にあるとき、どれも同じ集合を使う。事前の2箇所は
「合法打牌に現物があれば発動しない」条件なので、確定安全牌があるこの入力では発動しない。
そのため、この候補は事前の2箇所を通らず、FOLD時の絞り込みを確定安全牌で行う経路へ直接進む。

次の入力では、Championの`PolicyDecision`をそのまま返す。

- 他家のリーチ者が0人、または2人以上（複数リーチ）
- 聴牌を保てる打牌がある（PUSH）
- 合法打牌に確定安全牌がない（宣言牌の位置が不明で河の現物も合法打牌にない場合を含む）

宣言牌の位置が不明なら`safe(r) = river(r)`で、選ぶ打牌はChampionと同じである。ただし合法打牌に
河の現物があるときは上の確定安全牌の経路を通るため、`analysis`は`None`になり、Championの
`TargetedHonorReleaseAnalysis`とは異なる。確定安全牌の経路の`analysis`は常に`None`である。

聴牌の判断、鳴きの抑制、和了・立直・鳴きの応答は継承し、変えない。

## 前提

確定安全牌は`RIICHI_SAFE_TILE_RULE_ASSUMPTION`の規則（リーチ後の見逃しフリテンの継続、
リーチ後の待ち不変）を前提とする。確認済みの環境は`docs/riichi-safe-tiles.md`に記録する。
**未確認の環境（RiichiLab）へ自動適用しない。** 環境への結び付けは、このclassを登録する側
（評価環境）の責務である。

自席の打牌判断からだけ使う。`derive_riichi_safe_tiles`は、手牌が打牌前の枚数でない入力を
`ValueError`で拒否する。
"""

from lisjong.belief.riichi_safe_tiles import (
    RIICHI_SAFE_TILE_RULE_ASSUMPTION,
    derive_riichi_safe_tiles,
)
from lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware import (
    _decide_push_fold,
    _evaluate_and_choose_discard,
    _PushFoldDecision,
)
from lisjong.policies.genbutsu_defense_two_step_ukeire import _opponent_riichi_players
from lisjong.policies.placement_aware_speed_call import PlacementAwareSpeedCallPolicy
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput


class PassedTileGenbutsuPlacementAwareSpeedCallPolicy(PlacementAwareSpeedCallPolicy):
    """#312: 単独リーチ・非聴牌FOLDで、リーチ後に通った牌も現物として扱う（未昇格の実験候補）。"""

    RULE_ASSUMPTION = RIICHI_SAFE_TILE_RULE_ASSUMPTION
    """前提とする規則のidentity。確認していない環境との組合せで使わない。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        if len(_opponent_riichi_players(policy_input)) != 1:
            return super()._decide_discard(policy_input, discard_actions)
        if (
            _decide_push_fold(policy_input, discard_actions)
            is not _PushFoldDecision.FOLD
        ):
            return super()._decide_discard(policy_input, discard_actions)

        (riichi_player,) = derive_riichi_safe_tiles(policy_input).by_riichi_player
        safe_tile_types = riichi_player.safe_tile_types
        if not any(
            action.tile.tile_type in safe_tile_types for action in discard_actions
        ):
            return super()._decide_discard(policy_input, discard_actions)

        return PolicyDecision(
            action=_evaluate_and_choose_discard(
                policy_input, discard_actions, genbutsu_tile_types=safe_tile_types
            ),
            analysis=None,
        )


__all__ = ["PassedTileGenbutsuPlacementAwareSpeedCallPolicy"]
