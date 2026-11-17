# リーチ下1向聴の押し引き: 打点推定版の候補（#384）

`lisjong.policies.RiichiResponseHandValuePlacementAwareSpeedCallPolicy` の仕様である。
未昇格の実験候補で、現Heuristic Champion `PlacementAwareSpeedCallPolicy` をexact parentとし、
Championと #377 の候補（dora版、[仕様](riichi-response-candidate.md)）は書き換えない。
Familyは Heuristic（手書き規則の閾値を、事前に定めた探索空間から選ぶ。学習モデルなし）。

```python
from lisjong.policies import (
    RiichiResponseHandValueParameters,
    RiichiResponseHandValuePlacementAwareSpeedCallPolicy,
    riichi_response_hand_value_search_space,
)

parameters = RiichiResponseHandValueParameters(
    push_closed_minimum_points=5200,
    push_open_minimum_points=None,
    push_minimum_live_wall=24,
    fold_maximum_points=2000,
)
policy = RiichiResponseHandValuePlacementAwareSpeedCallPolicy(parameters)
parameters.identity  # 値だけから決まる候補identity
```

## #377の候補との違い

#377の候補は手の価値をdora枚数で測った。この候補は、役を含む和了時の打点推定 V で判断し、
押し返しの閾値を門前と副露で分ける。

| 項目 | #377（dora版） | #384（打点推定版） |
| --- | --- | --- |
| 手の価値 | dora枚数 | 聴牌した場合のロン点の推定 V（役・符・dora、親は子換算） |
| 押し返し | 門前だけ | 門前と副露で別の閾値（副露は「なし」も可） |
| 降り | 門前・副露共通のdora閾値 | 門前・副露共通のV閾値。副露で役がなければ V = 0 |

ゲート（他家リーチあり、点数状況mode `NORMAL`、どの打牌でも聴牌にならず打牌後の最小向聴数が1）と、
押し返し・降りの動作そのもの（他家リーチがないものとしてChampionに選ばせる / 古典的危険度score
最小の打牌へ絞る）は#377と同じである。

## 打点推定 V

`estimate_one_shanten_hand_value(policy_input, discard_actions)`（version
`riichi-response-hand-value-estimate-v1`）。判断時点の公開情報（自分の手牌・副露、見えている牌、
ドラ表示牌、場風・自風、親）だけを使い、他家の手牌・山・裏ドラは使わない。

1. 打牌後も1向聴の各打牌 d について、打牌後の純手牌を作る
2. 見えていない牌種 t（残り n_t > 0）を引いて聴牌になるとき、聴牌を保つ打牌のうち3の値が
   最大のものを選ぶ
3. 聴牌形の値は、待ち牌ごとのロン点（`evaluate_win`の`winner_points`、本場・供託を含まない）を、
   待ち牌の見えない残り枚数（引いた t を1枚除く）で加重平均した値
   - 門前: リーチしてロンした点（リーチ1翻を含む。裏ドラは`URA_DORA_EXCLUDED`、一発なし）
   - 副露: ロンした点。役がなければ0点
   - 待ちの残りが0なら0点。振聴は考慮しない
4. V(d) は3の値を n_t で加重平均した値（聴牌にならない牌は除く。1枚もなければ0）
5. V = max_d V(d)。自席が親なら2/3を掛けて子の点に換算する

値は`Fraction`で、浮動小数点の誤差を持たない。打牌後に1向聴になる打牌がなければ
`HandValueEstimateError`でfail closedする。

V は「聴牌した場合のロン点」で、聴牌確率・和了確率・放銃確率は見ない。押し引きの期待値そのものでは
ない。

## 判断規則

| 分岐 | 条件 | 動作 |
| --- | --- | --- |
| 押し返し（門前） | 共通現物あり、門前、`push_closed_minimum_points`がNoneでなく V ≥ その値、残り山 ≥ `push_minimum_live_wall` | 他家リーチがないものとしてChampionに選ばせる |
| 押し返し（副露） | 共通現物あり、副露、`push_open_minimum_points`がNoneでなく V ≥ その値、残り山 ≥ `push_minimum_live_wall` | 同上 |
| 降り | 共通現物なし、`fold_maximum_points`がNoneでなく V ≤ その値 | 古典的危険度score最小の打牌へ絞ってChampionに選ばせる |

門前は暗槓だけを持つ手を含む（Championの`_is_closed_hand`と同じ）。
それ以外の判断（和了・立直・鳴き・Pass、聴牌時と2向聴以上の打牌）はChampionと同じである。
PolicyInput以外の情報を使わず、instanceに状態を持たない。

## 探索空間 v2

`RIICHI_RESPONSE_HAND_VALUE_SEARCH_SPACE_VERSION = "riichi-response-hand-value-search-space-v2"`。
`riichi_response_hand_value_search_space()`は次の8候補を固定順（門前押し、副露押し、降りの順に
外側から）で返す。

| パラメータ | 値 |
| --- | --- |
| `push_closed_minimum_points` | 5,200 / 8,000 |
| `push_open_minimum_points` | なし / 8,000 |
| `fold_maximum_points` | 2,000 / 3,900 |
| `push_minimum_live_wall` | 24（固定） |

## 評価

評価計画と結果は [#384](https://github.com/lisbun/lisjong/issues/384) を正本とする。
比較対象は#377の候補（`push-dora-1 / push-wall-24 / fold-dora-1`）である。
