# 同点の数牌打牌を公開paijiaで選び直す候補（#330）

Issue: [#330](https://github.com/lisbun/lisjong/issues/330)（親: lisbun/lisjong-project#84、探索対象選定: [#297](https://github.com/lisbun/lisjong/issues/297)、評価: lisbun/lisjong-arena#520）

`lisjong.policies.PlacementAwareSpeedCallSuitedPaijiaTiePolicy` の仕様である。未昇格の実験候補で、
Championは書き換えていない。**実装と局面テストまでが範囲で、強さは評価していない。** 発動頻度も
未測定で、評価Issue側の変更頻度gateで判定する。実装Issueの完了は強化の成功を意味しない。

```python
from lisjong.policies import PlacementAwareSpeedCallSuitedPaijiaTiePolicy

policy = PlacementAwareSpeedCallSuitedPaijiaTiePolicy()  # 設定なし
```

| 項目 | 値 |
|---|---|
| 候補identity（案） | `placement-aware-speed-call-suited-paijia-tie-v1`（Arenaで登録する側が明示する。class名から導出しない） |
| 設定 | 空object。閾値・係数なし |
| 学習artifact | なし |
| exact parent | `PlacementAwareSpeedCallPolicy`（現Champion） |
| 対照B | 同じlisjong revisionの `PlacementAwareSpeedCallPolicy`（歴史的固定C0ではない） |

## 動作

Championの打牌判断を1回だけ実行し、次の1〜5をすべて満たす通常打牌だけ、選択を6で置き換える。

1. 通常自摸後（`drawn_tile`あり）、自席に副露・暗槓がない、自席が未リーチ（宣言済みを含めて対象外）、
   他家リーチなし
2. 南4局以降（西入を含む）ではない。Championの候補制限（`_eligible_discard_actions`）後の集合を
   `_classify_branch` がPUSHと判定する
3. FiniteHorizon completion massが候補全件0で、限定R5を実行していない
   （`TargetedHonorReleaseActivationStage` が `PARENT_SHANTEN_TOO_CLOSE` / `PARENT_NOT_MINIMUM_SHANTEN` /
   `PARENT_NOT_MAXIMUM_UKEIRE` / `PARENT_ALREADY_HONOR` / `NO_QUALIFYING_HONOR_PEER`）。R5を評価した判断は、
   選択が変わったかによらず対象外
4. HandValueAwareのstaged評価で、最小向聴・最大受入・最大retained real value・最大yaku route・
   最大second-stepをすべて通過した最終tieが2候補以上ある。向聴0（second-stepを計算しない）は対象外
5. Championの選択牌が数牌で、最終tieのうち数牌の打牌に異なる牌（牌種、または赤/通常）が2つ以上ある。
   同じ牌の手出し / ツモ切りは1つと数える
6. その数牌の打牌のうち `_PublicCounts.paijia(action.tile)`（kobalab 0004の牌価値、公開未見枚数）が最小の
   ものを選ぶ。Championの打牌も最小なら維持する。それ以外の同点は `discard_action_sort_key` で決める
   （0004のsource順は使わない）

最終tieはHandValueAwareが実際に計算したsnapshotから読む。second-step値は、向聴・受入・retained real value・
yaku routeの各最大を通過し、さらに向聴0でない候補にだけ計算されるので、second-step値を持つ候補のうち
最大値の集合が最終tieである。`None`は同点や0として扱わない。rankingを再実行・複製しない。

このため、置き換えた打牌はChampionの打牌と5段階すべてで同じ評価値を持つ。ここでの「同点」はこの
heuristicの評価値が同じという意味で、真のEVが同じことや悪化しないことは保証しない。

赤5の打牌はretained real valueが下がるため、通常牌と同じ最終tieに入らない。赤は未見の赤5枚数による
paijiaの加点（`_PaijiaInput`の式）として効く。

## 変えないもの

役牌・その他字牌の切り方、FiniteHorizon評価、受入、価値proxy、second-step、限定R5、鳴き・立直・和了・
守備の規則はChampionと同じである。和了・立直・Chi/Pon・Passの判断は `_decide()` を継承する。
ただし打牌が変わると以後の展開が変わるので、対局全体の鳴き頻度や放銃率が不変とは主張しない。

入力不整合や未見枚数が探索幅未満等でChampionが送出する例外は、そのまま送出する。

## `analysis`

打牌判断の `PolicyDecision.analysis` は常に `SuitedPaijiaTieAnalysis` で、Championの
`TargetedHonorReleaseAnalysis` は流用しない。

| field | 内容 |
|---|---|
| `stage` | `SuitedPaijiaTieStage`。どのgateでChampion維持になったか（`NOT_NORMAL_TURN` … `SINGLE_SUITED_OPTION`）、paijiaで維持した（`CHAMPION_MINIMUM_PAIJIA`）か、選び直した（`PAIJIA_SWITCH`）か |
| `champion_action` / `selected_action` / `action_changed` | Championの打牌、この候補の打牌、変更有無。変更するのは `PAIJIA_SWITCH` だけ |
| `paijia_evaluations` | paijiaを評価した判断だけ、最終tieの数牌打牌とpaijiaを `discard_action_sort_key` 順に持つ |

計数はこのPolicy側の目的限定analysis、対局結果はArenaの客観recordで扱う。

## Champion側の変更

最終tieを読むため、`targeted_honor_release_terminal_progression` に、同じ1回の計算で使った
HandValueAwareのsnapshotも返す内部helper（`_evaluate_and_choose_discard_with_hva` /
`_evaluate_all_zero_targeted_honor_release_with_hva`）を切り出した。既存の
`_evaluate_and_choose_discard` / `_evaluate_all_zero_targeted_honor_release` はそれを包むだけで、
選択・analysis・例外は変わらない。

## 情報境界

入力は `PolicyInput` だけ（公開未見枚数・ドラ表示牌・自風/場風）。hidden truth、学習source、Arena、
engineをimportしない（`check_candidate_imports` / `check_candidate_purity` をtestで確認）。
Policy instanceにcross-decision stateを持たない。

## 範囲外と限界

- 強さ・変更頻度は未評価。最終tie候補が実戦でどれだけあるかはArena側の頻度gateで判定し、不足しても
  gateを緩めない
- arena #432の旧0004＋Belief全面置換候補の再評価ではなく、全面置換を避けた別介入である
- Champion昇格・稼働変更、旧#423/#436/#503の救済、LearningのCALL再学習は対象外
- #517のpair・pin・plan・実行checkoutは変更しない。新候補用のproducer/wheel/pinは評価Issueで固定する
