# 終盤の形式聴牌候補（#375）

`lisjong.policies.KeitenCallPlacementAwareSpeedCallPolicy` の仕様である。未昇格の実験候補で、
現Heuristic Champion `PlacementAwareSpeedCallPolicy` をexact parentとし、Championは書き換えない。
Familyは Heuristic（学習モデル・閾値探索なし）。identity案は
`placement-aware-speed-call-keiten-call-v1`。

```python
from lisjong.policies import KeitenCallPlacementAwareSpeedCallPolicy

policy = KeitenCallPlacementAwareSpeedCallPolicy()  # 設定なし
```

## 目的

荒牌流局の聴牌 / 不聴は、不聴罰符の受け払いと親の連荘を決める。Championは役routeに合わない
鳴きをしないため、流局間際に「鳴けば聴牌」の機会があってもPassする。終盤だけ、聴牌を取る
鳴きと聴牌維持の打牌を加える。

## 適用範囲

`RoundState.live_wall_tiles_remaining <= 12`（`KEITEN_MAXIMUM_LIVE_WALL`）の判断だけ。
それ以外の判断はChampionと完全に同じである。

## A. 聴牌を取る鳴き

Championが明示的にPassした判断だけを置き換える。次をすべて満たすChi / Ponを選ぶ。

1. 自席の純手牌が非聴牌（向聴数 1以上）
2. 鳴いたあと、喰い替えを守った打牌を1枚して純手牌が聴牌（向聴数 0）になる
3. 他家リーチがある場合、2の打牌が全リーチ者への共通現物である

複数あれば既存のcanonical順（`_call_action_sort_key`）の最小を選ぶ。`legal_actions`の
入力順には依存しない。

## B. 聴牌維持の打牌

副露済みの手だけ。聴牌を維持する打牌があれば、候補をその集合へ絞ってChampionへ渡す。
他家リーチがあり、その集合に共通現物があれば、共通現物だけへ絞る。聴牌を維持する打牌が
なければ何も変えない。

Aで想定した打牌は保存しない。鳴いた直後の実際の打牌は、Bが残した集合からChampionが選ぶ。

## 判定の定義と限界

- 聴牌は純手牌の向聴数0で判定する。役の有無は見ない（形式聴牌を含む）
- 待ち牌がすべて見えている純カラ聴を区別しない
- 共通現物は、全リーチ者の河に共通する牌種（Championの`_common_genbutsu_tile_types`）
- リーチ下で、聴牌維持の打牌に共通現物がない副露手は、Championと同じく聴牌を維持して押す
- 門前の手のBは対象外（門前聴牌はChampionのリーチ判断に従う）
- PolicyInput以外の情報を使わず、instanceに状態を持たない

## 評価

評価計画と結果は [#375](https://github.com/lisbun/lisjong/issues/375) を正本とする。

#375のdevelopment screen（既使用seed 940000..940099、同一局の対比較）は **STOP-LOW-ACTIVATION**。
Championと判断が分かれたのは400 seat-hanchans中19件（約0.05件/席・半荘）で、事前に固定した
下限20件に届かなかった。強さは未確定で、この候補を既定の採用候補にしない。追加seed・閾値変更・
条件緩和による救済はしない。
