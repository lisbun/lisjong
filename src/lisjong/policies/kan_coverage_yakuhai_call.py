"""HandBelief Stage 3 Entry Gate向けのdeterministic kan-capable coverage Policy。

Issue #151の背景: `lisjong-arena #131`のdevelopment-only pilotで、current
first-party Policy familyがdaiminkan / ankan / kakan / rinshan_drawを1件も
選ばないというstructural coverage holeが判明した。`MinimalPolicy`はfull
`InternalAction` total orderを持つが、自席decisionでは`DiscardAction`が
kan variantより先着するsort key（`minimal._action_sort_key`参照）を意図的に
維持しており、ankan / kakan coverage sourceとしては使えない。

このPolicyは、HandBelief / Arena実験のtraining coverage source
（kan / rinshan trajectoryをfirst-party deterministic executionから発生
可能にするための最小Policy）として追加する。

```text
kan-capable coverage source
!= stronger Policy
!= current strength baseline
!= production recommendation
```

`docs/policy-status.md`のcurrent strength baseline / strength hierarchyには
属さない。Arena `POLICY_CATALOG`への登録はこのPolicyの責務ではなく、Arena
#120のexplicit import referenceから`lisjong.policies.kan_coverage_yakuhai_call:
KanCoverageYakuhaiCallPolicy`として利用する想定である。

## Selection semantics

```text
1. RonAction / TsumoAction        (winning action)
2. DaiminkanAction / AnkanAction / KakanAction  (legal kan action)
3. delegated normal-play decision
```

winning actionは常にkanより優先する。winning actionが無く、legalなkan action
（種別・枚数を問わない）が存在すれば、そのいずれかを必ず選ぶ。winning actionも
kanも無い場合だけ、既存first-party deterministic Policyへdelegateする。

すべてのstageで`decision.legal_actions`だけを参照し、`isinstance`によるvariant
判定以外の合法性判定を行わない。Policy側でriichi後ankan等の細かなlegalityを
再判定せず、`DecisionContext.legal_actions`を唯一の正本として扱う。選択した
Actionはconstructし直さず、`decision.legal_actions`内のobjectをそのまま返す。

## Multiple kan candidates

複数のkan候補（同種複数tile、または種別違い）が同時にlegalな場合、
`_kan_action_sort_key()`が示すsemantic fieldだけのtotal orderで1件を選ぶ。
入力`legal_actions`の順序やPRNGには依存しない。

kan種別間の固定順序は次のとおりであり、determinismを保証するための
implementation choiceに過ぎず、麻雀上の優劣を意味しない。

```text
DaiminkanAction < AnkanAction < KakanAction
```

同種内の複数候補（例: 複数tile種別のankan）は、関与するtile / seatの
`tile_sort_key` / seat indexだけで比較する。

winning actionが複数legal（例えば合成fixtureでRonAction/TsumoActionが同時に
legalな場合）も同様に、`_winning_action_sort_key()`によるsemantic field由来の
total orderで決定する。RonAction/TsumoAction間の固定順序も同じくimplementation
choiceであり、麻雀上の優劣を意味しない。

## Normal-play delegation

kanもwinning actionも無いdecisionは、既存の`yakuhai-call`実装
(`YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy`)へ
compositionでdelegateする。delegate instanceは`__init__`で1回だけ生成し、
decisionごとに再生成しない。delegateが返したActionを後処理で別semanticへ
変換せず、そのまま返す。既存`yakuhai-call` / `MinimalPolicy`のclass自体の
semanticsは変更しない。
"""

from typing import assert_never

from lisjong.policies.minimal import _tiles_sort_key
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy,
)
from lisjong.policy_contract.action import (
    AnkanAction,
    DaiminkanAction,
    InternalAction,
    KakanAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy import Policy
from lisjong.policy_contract.tile import tile_sort_key

_WINNING_ACTION_TYPES = (RonAction, TsumoAction)
_KAN_ACTION_TYPES = (DaiminkanAction, AnkanAction, KakanAction)

type _WinningAction = RonAction | TsumoAction
type _KanAction = DaiminkanAction | AnkanAction | KakanAction


def _winning_action_sort_key(action: _WinningAction) -> tuple[object, ...]:
    """winning candidateだけを対象にした、semantic fieldだけのtotal order。

    RonAction/TsumoActionの固定順序はcoverage Policyのdeterminismを固定する
    implementation choiceであり、麻雀上の優劣を意味しない。
    """
    if isinstance(action, RonAction):
        return (
            0,
            int(action.actor),
            int(action.target),
            tile_sort_key(action.winning_tile),
        )
    if isinstance(action, TsumoAction):
        return (1, int(action.actor), tile_sort_key(action.winning_tile))
    assert_never(action)


def _kan_action_sort_key(action: _KanAction) -> tuple[object, ...]:
    """legal kan candidateだけを対象にした、semantic fieldだけのtotal order。

    kan種別間の固定順序(daiminkan < ankan < kakan)はdeterminismを固定する
    implementation choiceであり、麻雀上の優劣を意味しない。
    """
    if isinstance(action, DaiminkanAction):
        return (
            0,
            int(action.actor),
            int(action.target),
            tile_sort_key(action.called_tile),
            _tiles_sort_key(action.consumed_tiles),
        )
    if isinstance(action, AnkanAction):
        return (1, int(action.actor), _tiles_sort_key(action.tiles))
    if isinstance(action, KakanAction):
        return (
            2,
            int(action.actor),
            tile_sort_key(action.added_tile),
            int(action.from_seat),
            tile_sort_key(action.called_tile),
        )
    assert_never(action)


class KanCoverageYakuhaiCallPolicy:
    """HandBelief training coverage source用のdeterministic kan-capable Policy。

    strength baselineではなく、legal kan actionをfirst-party deterministic
    executionから実際に発生可能にするためのcoverage sourceである。
    """

    def __init__(self) -> None:
        self._delegate: Policy = (
            YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy()
        )

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        winning_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, _WINNING_ACTION_TYPES)
        )
        if winning_actions:
            return min(winning_actions, key=_winning_action_sort_key)

        kan_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, _KAN_ACTION_TYPES)
        )
        if kan_actions:
            return min(kan_actions, key=_kan_action_sort_key)

        return self._delegate.choose_action(decision)
