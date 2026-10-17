# 確定安全牌を現物として使う候補（#312）

Issue: [#312](https://github.com/lisbun/lisjong/issues/312)（親: [#299](https://github.com/lisbun/lisjong/issues/299)、設計: [riichi-passed-tile-genbutsu-design.md](riichi-passed-tile-genbutsu-design.md) 4節）

`lisjong.policies.PassedTileGenbutsuPlacementAwareSpeedCallPolicy` の仕様である。未昇格の実験候補で、
Championは書き換えていない。**実装と局面テストまでが範囲で、強さは評価していない。** 確定安全牌を
正しく認識できることと、この候補が強くなることは別である。採否は、別の開発データでの総合収支と
処理時間で判断する。

```python
from lisjong.policies import PassedTileGenbutsuPlacementAwareSpeedCallPolicy

policy = PassedTileGenbutsuPlacementAwareSpeedCallPolicy()
```

exact parentは `PlacementAwareSpeedCallPolicy`。Arenaで使うidentityはclass名から導出せず、
登録する側が明示する。

## 動作

Championは他家リーチに降りるとき、リーチ者の河にある牌（共通現物）だけを現物として扱う。この候補は、
次の2つがともに成り立つ打牌判断だけ、その現物を `safe(r) = river(r) ∪ passed(r)`
（[riichi-safe-tiles.md](riichi-safe-tiles.md)）へ置き換える。

1. 他家のリーチ者（宣言済みを含む）がちょうど1人
2. どの合法打牌でも聴牌にできない（Championと同じFOLD条件）

合法打牌に確定安全牌があるとき、確定安全牌へ絞り、攻撃側の評価（FH completion mass → 手牌価値）で
1つ選ぶ。

Championが共通現物を使う打牌経路の3箇所は、この入力では同じ集合を使う。

| 順 | Championの処理 | この候補 |
|---|---|---|
| 1 オーラストップ目 | 河の現物が合法打牌になければ危険度最小へ絞る | 確定安全牌が合法打牌にあるので絞らない |
| 2 mechanism defense | 最小向聴数2以上で河の現物が合法打牌になければ危険度最小へ絞る | 同上 |
| 4 FOLD時の絞り込み | 河の現物へ絞る | **確定安全牌へ絞る** |

順1・順2は「合法打牌に現物があれば発動しない」条件なので、確定安全牌が合法打牌にある入力では
発動しない。実装は順1・順2を通らず、順4を確定安全牌で行う経路へ直接進む（Champion側は
`_evaluate_defense_filter` などに、集合を渡す任意の引数 `genbutsu_tile_types` を足しただけで、
省略時の動作は変わらない）。

### Championと同じ動作になる入力

次の入力では、候補はChampionの `PolicyDecision` をそのまま返す（`analysis` も同じ）。

- 他家のリーチ者が0人、または2人以上（複数リーチ）
- 聴牌を保てる打牌がある（PUSH）
- 合法打牌に確定安全牌がない

宣言牌の位置が不明（`None`）なリーチ者では `safe(r) = river(r)` になり、選ぶ打牌はChampionと
同じである。合法打牌に河の現物があるときは、確定安全牌の経路を通るため `analysis` は `None`
になるが、打牌は変わらない。

和了・立直・鳴きの応答と鳴きの抑制は継承し、変えない。危険度scoreの式も変えない。

### `analysis` について

確定安全牌の経路で返す `PolicyDecision.analysis` は `None` である。Championの
`TargetedHonorReleaseAnalysis` は返さない。

## 前提と適用してよい環境

確定安全牌の `passed(r)` は、次の2つの規則を前提とする。前提のidentityは
`PassedTileGenbutsuPlacementAwareSpeedCallPolicy.RULE_ASSUMPTION`
（`riichi-missed-win-furiten-and-fixed-waits-v1`）で、評価の環境identity（RuleSet・engineの
固定version）と結び付けるのは、この候補を登録する側（Arena）の責務である。

1. リーチ後の見逃しは、その局の終わりまでフリテンになる
2. リーチ後は待ちが変わらない（打牌はツモ切りだけ。暗槓は待ちを変えない場合にだけ許される）

| 環境 | 状態 |
|---|---|
| RiichiEnv 0.4.10 | 確認済み（局面テスト）。適用してよい |
| project標準RuleSet / lisjong-engine `91af75e` | 確認済み（規則文書と実装） |
| RiichiLab（対局server） | **未確認。適用対象に含めない** |

確認の根拠は [riichi-safe-tiles.md](riichi-safe-tiles.md) の「環境ごとの確認状況」にある。
RiichiLabの規則確認は #302 の残件で、対局記録で `passed(r)` にロン違反が0件だったという観測は
補助証拠であり、規則確認の代わりにしない。RiichiLabでこの候補を稼働・評価するかは、規則確認の後に
判断する。

## 情報境界

- 入力は `PolicyInput` だけである。判断をまたぐ状態を持たない。
- 基盤は自席の打牌判断からだけ使う。手牌が打牌前の枚数（3n+2枚）でない入力は、確定安全牌を使う
  経路で `ValueError` になる（補助チェック）。
- `docs/candidate-boundary.md` のimport制限（静的検査）と、PYTHONHASHSEEDの異なるprocess間の
  純粋性検査を `tests/test_passed_tile_genbutsu_policy.py` で行う。静的検査は候補moduleが対象で、
  継承するChampion側の module は対象外（既存Policyへの適用は要求しない）。継承を含む候補全体の
  純粋性は、純粋性検査が実際の `choose_action` を呼ぶことで確かめる。

## 対象外

強さの評価、影響範囲の計数（設計6節）、開発比較、複数リーチの行動変更、聴牌での確定安全牌の
利用（#254の領域）、鳴きの抑制の変更、Championの昇格・降格。
