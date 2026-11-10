# リーチ下1向聴の押し引き候補（#377）

`lisjong.policies.RiichiResponsePlacementAwareSpeedCallPolicy` の仕様である。未昇格の実験候補で、
現Heuristic Champion `PlacementAwareSpeedCallPolicy` をexact parentとし、Championは書き換えない。
Familyは Heuristic（手書き規則の閾値を、事前に定めた探索空間から選ぶ。学習モデルなし）。

```python
from lisjong.policies import (
    RiichiResponseParameters,
    RiichiResponsePlacementAwareSpeedCallPolicy,
    riichi_response_search_space,
)

parameters = RiichiResponseParameters(
    push_minimum_dora=2, push_minimum_live_wall=24, fold_maximum_dora=None
)
policy = RiichiResponsePlacementAwareSpeedCallPolicy(parameters)
parameters.identity  # 値だけから決まる候補identity
```

パラメータは必ず明示的な値で渡す（defaultなし）。identityはclass名ではなくパラメータ値から決まる。

## 対象

Championは他家リーチ下の非聴牌手を、手の価値を見ずに扱う。

| 状況 | Championの打牌 |
| --- | --- |
| 共通現物がある | 共通現物だけへ絞る（降りる） |
| 共通現物がなく1向聴 | 全打牌から攻撃基準で選ぶ（押す） |

この候補は、次をすべて満たす打牌判断でだけ、上の2つを手のdora枚数で入れ替える。
それ以外の判断はChampionと同じである。

- 他家リーチがある
- 点数状況modeが`NORMAL`
- どの打牌でも聴牌にならず、打牌後の最小向聴数が1

## A. 押し返し

`push_minimum_dora`がNoneでなく、共通現物があり、自席が門前で、dora枚数が
`push_minimum_dora`以上、残り山が`push_minimum_live_wall`以上のとき、他家リーチが
ないものとしてChampionに打牌を選ばせる。

## B. 降り

`fold_maximum_dora`がNoneでなく、共通現物がなく、dora枚数が`fold_maximum_dora`以下のとき、
古典的危険度score（全リーチ者に対する最大値）が最小の打牌へ絞ってChampionへ渡す。

## 定義と限界

- dora枚数は、純手牌と自席の副露牌にある赤ドラと表示牌由来doraの合計
  （`_retained_concealed_dora_count`と同じ数え方）。得点期待値ではない
- 共通現物は、全リーチ者の河に共通する牌種
- 押し返しでは、リーチ者への危険度を一切見ない（Championの非リーチ時の打牌そのもの）
- 受け入れ枚数・待ちの形・巡目以外の局面情報は条件に使わない
- 和了・立直・鳴き・Pass、聴牌時と2向聴以上の打牌は変えない
- PolicyInput以外の情報を使わず、instanceに状態を持たない

## 探索空間 v1

`RIICHI_RESPONSE_SEARCH_SPACE_VERSION = "riichi-response-search-space-v1"`。
`riichi_response_search_space()`は次の8候補を固定順（pushが外側、foldが内側）で返す。

| パラメータ | 値 |
| --- | --- |
| `push_minimum_dora` | なし / 1 / 2 |
| `fold_maximum_dora` | なし / 0 / 1 |
| `push_minimum_live_wall` | 24（固定） |

両方「なし」の組合せ（Championと同じ挙動）は含めない。

## 評価

評価計画と結果は [#377](https://github.com/lisbun/lisjong/issues/377) を正本とする。

#377のdevelopment screenは、各seedの東1局だけを実行し、候補とChampionの判断が最初に分かれた席の
局収支差（点）を標本にした。相手はChampion 3席。

| 段階 | seed | 対象 | 標本 | 平均 | 区間 |
| --- | --- | --- | ---: | ---: | --- |
| SELECT | 377000..378199 | 押し返し dora>=1 | 556 | +354.1 | 90% [+38.7, +689.6] |
| SELECT | 同上 | 押し返し dora>=2 | 200 | +248.5 | 90% [−368.6, +883.4] |
| SELECT | 同上 | 降り dora<=0 | 348 | +494.3 | 90% [+149.5, +842.0] |
| SELECT | 同上 | 降り dora<=1 | 607 | +382.4 | 90% [+100.5, +664.6] |
| CONFIRM | 378200..379399 | 押し返し dora>=1 ＋ 降り dora<=1 | 1,062 | +260.6 | 95% [−18.3, +544.2] |

CONFIRMの判定は **INCONCLUSIVE-DEVELOPMENT**（平均は正、95%区間が0を含む）。東1局・点数平坦の
1局収支だけを測った結果で、半荘順位への効果、点差のある局面、別の相手での再現性は測っていない。
この候補を既定の採用候補にしない。追加seed・閾値変更・条件の後付けによる救済はしない。
