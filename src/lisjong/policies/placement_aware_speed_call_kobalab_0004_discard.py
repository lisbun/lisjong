"""現Championの攻撃打牌だけをkobalab 0004の牌効率で選ぶ候補Policy。

Issue #226のexperimental Policy。現Champion `PlacementAwareSpeedCallPolicy`
（Issue #199）を変更せず、その打牌decisionのうちPUSHと分類された攻撃打牌の
最終比較だけを、`lisjong.policies.kobalab_0004_discard`の0004打牌評価へ
置き換える。0004の立直・槓・和了・九種九牌の判断規則は移植しない。

打牌decisionの経路（Championと同じ入力・順序）:

1. `_eligible_discard_actions()`: 役保持（副露・役未確保・PUSH時の最小route
   shanten）と、オーラストップ目SPEEDの非聴牌FOLDでの最小危険度制限。
   各filter内部のPUSH / FOLD判定はChampionのまま。
2. 制限後の候補集合を`_classify_branch()`（mechanism filter -> defense filter）
   へ渡し、そのbranchとeligible集合をsingle-sourceで使う。元の未制限集合で
   PUSHを再判定しない。
3. branchが`PUSH`なら、eligible集合を0004方式で比較する（下記）。門前・副露手、
   通常自摸・副露直後（`drawn_tile is None`）・立直宣言牌decisionを区別しない。
4. それ以外（共通現物FOLD・mechanism防御・全候補fallback）はChampionの親
   `TargetedHonorReleaseTerminalProgressionPolicy`の打牌選択へ、同じ制限後
   集合をそのまま渡す（選択・analysis・例外ともChampionと同じ）。

PUSHでの0004比較は、Championの攻撃評価（FiniteHorizon和了評価、
HandValueAwareの打点proxy・second-step、限定R5字牌切り補正）を実行せずに
置き換えるもので、それらを重ねて後から上書きしない。役routeは候補制限にだけ
効き、0004の向聴数・改善牌は通常の構造評価（route外の牌も改善牌に数える）で
ある。「役を保つ候補制限 + 構造的受入」であり、役付き和了への距離を一貫して
評価する方式ではない。

- eligible集合に打牌後向聴数が打牌前向聴数以下の候補があれば、その中で
  実残り枚数の受入が最大の候補を、paijia / source順で選ぶ（参照版と同じ）。
- なければ、eligible集合で最小の打牌後向聴数を達成する候補の中を同じ規則で
  比較する（この合成版固有のfallback。参照版の「評価順の先頭」ではない）。

意図した差: ChampionのPUSH経路は未見枚数がFiniteHorizonの探索幅より少ないと
`FiniteHorizonCompletionPolicyError`を送出するが、このPolicyのPUSH経路は
FiniteHorizonを通らないため打牌を選ぶ。非PUSH経路ではChampionの例外が残る。
未見枚数の再構成（`derive_remaining_tile_inventory()`）による入力検証は省かない。

和了・立直（Always Riichi: 合法な`RiichiAction`があれば宣言）・Chi / Pon・
Passの判断はChampionの`_decide()`をそのまま継承する。Chi / Ponの採否は
仮想mandatory discard後の最小向聴数で決まり、打牌選択を呼ばないため、
同じdecision入力では変わらない。0004の`_allows_riichi_discard()`や即リー規則は
持ち込まない。

traced executionでは、PUSHの0004選択は`analysis=None`を返す（Championの
R5 analysisを流用しない）。非PUSH経路はChampionと同じanalysisを返す。
`choose_action()`とtraced経路は同じ1回の判断から同じActionを返す。

PolicyInput以外の状態やhidden informationは使用せず、Policy instanceに
cross-decision stateを保持しない。Arena等での識別はclass名
（`lisjong.policies.PlacementAwareSpeedCallKobalab0004DiscardPolicy`）で行い、
新しい`identity`属性は持たない。
"""

from lisjong.policies.kobalab_0004_discard import (
    _choose_minimum_shanten_discard,
    _DiscardStructures,
    _PublicCounts,
)
from lisjong.policies.placement_aware_speed_call import (
    PlacementAwareSpeedCallPolicy,
    _eligible_discard_actions,
)
from lisjong.policies.targeted_honor_release_terminal_progression import (
    TargetedHonorReleaseBranch,
    _classify_branch,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput


def _choose_push_discard(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> DiscardAction:
    """PUSHのeligible集合を0004方式（合成版fallback付き）で比較する。"""
    structures = _DiscardStructures(
        policy_input.own_hand.concealed_tiles,
        tuple(action.tile for action in discard_actions),
    )
    counts = _PublicCounts(policy_input)
    return _choose_minimum_shanten_discard(counts, discard_actions, structures)


class PlacementAwareSpeedCallKobalab0004DiscardPolicy(PlacementAwareSpeedCallPolicy):
    """Issue #226: Championの攻撃打牌（PUSH）だけを0004の牌効率で選ぶ。"""

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        eligible_actions = _eligible_discard_actions(policy_input, discard_actions)
        branch, branch_actions = _classify_branch(policy_input, eligible_actions)
        if branch is not TargetedHonorReleaseBranch.PUSH:
            # Championの`_decide_discard()`から候補制限を除いた残り（parent）と同じ。
            return super(PlacementAwareSpeedCallPolicy, self)._decide_discard(
                policy_input, eligible_actions
            )
        return PolicyDecision(
            action=_choose_push_discard(policy_input, branch_actions), analysis=None
        )


__all__ = ["PlacementAwareSpeedCallKobalab0004DiscardPolicy"]
