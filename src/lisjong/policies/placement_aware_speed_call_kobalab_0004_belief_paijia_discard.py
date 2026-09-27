"""0004合成版のPUSH打牌のpaijia入力だけをBelief由来にした別候補Policy。

Issue #230のexperimental Policy。`PlacementAwareSpeedCallKobalab0004DiscardPolicy`
（Issue #226 / #227。以下「合成版」）を比較基準として変更せず残し、PUSHと
分類された攻撃打牌の0004比較で、同点処理に使うpaijiaの入力だけを
`Kobalab0004BeliefPaijiaPolicy`（Issue #218 / #219）と同じBelief由来の
残余期待枚数へ替える。

合成版と同じもの:

- 候補制限（`_eligible_discard_actions()`）とbranch分類（`_classify_branch()`）。
- PUSHの比較規則: 打牌前向聴数以下の候補で実残り枚数の受入が最大の候補、
  なければ制限後集合の最小打牌後向聴数の候補で同じ比較（合成版fallback）。
  向聴数・改善牌の定義、Rust一括構造評価、受入の入力（実残り枚数）と集計。
- 門前・副露手・副露直後・立直宣言牌の対象範囲。役routeは候補制限にだけ効く。
- 非PUSHの打牌（選択・analysis・例外）と、和了・立直・鳴き・槓・九種九牌等の
  非打牌判断（Championの判断をそのまま継承）。
- traced executionでPUSHは`analysis=None`。`choose_action()`とtraced経路は
  同じ1回の判断から同じActionを返す。

違うもの（PUSHの打牌だけ）: paijiaの入力を
`未見枚数 − 他家3人の手牌内期待枚数`（fixed-point raw、赤5は別axis）にする。
推定器・他家slot導出・固定小数点尺度・丸め・保存則検証は0004 Belief版と同じ
共通処理（`lisjong.policies.kobalab_0004_discard._belief_paijia_counts()`）で、
同じdecisionの未見枚数を受入・推定器・残余で共有する。Belief導出はPUSHの打牌
decisionごとに1回だけ行い、非PUSH・非打牌経路では行わない。保存則違反は
clampせず`Kobalab0004ReferencePolicyError`で拒否する。

現行の条件付き一様推定器では残余期待枚数が丸めを除いて未見枚数の共通倍率となり、
paijiaの順位は保たれる。この接続だけで打牌が改善するとは主張しない。将来の
非一様推定器を接続する準備として置く。

PolicyInput以外の状態やhidden informationは使用せず、Policy instanceに
cross-decision stateを保持しない。Arena等での識別はclass名
（`lisjong.policies.PlacementAwareSpeedCallKobalab0004BeliefPaijiaDiscardPolicy`）で
行い、新しい`identity`属性は持たない。推定器の記録は`belief_estimator`に置く。
"""

from lisjong.policies.kobalab_0004_discard import KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR
from lisjong.policies.placement_aware_speed_call_kobalab_0004_discard import (
    PlacementAwareSpeedCallKobalab0004DiscardPolicy,
)


class PlacementAwareSpeedCallKobalab0004BeliefPaijiaDiscardPolicy(
    PlacementAwareSpeedCallKobalab0004DiscardPolicy
):
    """Issue #230: 0004合成版のPUSH打牌のpaijia入力だけをBelief由来にした候補。"""

    belief_estimator = KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR
    _belief_paijia = True


__all__ = ["PlacementAwareSpeedCallKobalab0004BeliefPaijiaDiscardPolicy"]
