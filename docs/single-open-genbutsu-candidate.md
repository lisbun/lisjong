# 単独副露者の河の現物を優先する候補（#343）

Issue: [#343](https://github.com/lisbun/lisjong/issues/343)（親: lisbun/lisjong-project#84、探索対象選定: [#297](https://github.com/lisbun/lisjong/issues/297)、評価: lisbun/lisjong-arena#524）

`lisjong.policies.SingleOpenGenbutsuPlacementAwareSpeedCallPolicy` の仕様である。未昇格の実験候補で、
Championは書き換えていない。**実装と局面テストまでが範囲で、強さは評価していない。** 発動頻度も
未測定で、評価Issue側の変更頻度gateで判定する。実装Issueの完了は強化の成功を意味しない。

```python
from lisjong.policies import SingleOpenGenbutsuPlacementAwareSpeedCallPolicy

policy = SingleOpenGenbutsuPlacementAwareSpeedCallPolicy()  # 設定なし
```

| 項目 | 値 |
|---|---|
| 候補identity（案） | `placement-aware-speed-call-single-open-genbutsu-v1`（Arenaで登録する側が明示する。class名から導出しない） |
| 設定 | 空object。閾値・係数なし |
| 学習artifact | なし（判定のためにモデルを読まない） |
| exact parent | `PlacementAwareSpeedCallPolicy`（現Champion） |

## 動作

Championの打牌判断（C0）を1回だけ実行し、次の1〜6をすべて満たす通常打牌だけ、7〜8で打牌を替えるかを
決める。和了・立直・鳴き・槓・Passなど打牌以外の判断は `_decide()` を継承し、変えない。

1. 通常自摸後の打牌（`drawn_tile` あり）。鳴き直後の打牌は対象外
2. 自席が未リーチ（`RiichiState.NONE`）で、自席に暗槓以外の副露がない（暗槓は門前扱い）
3. 他家に宣言済み（`DECLARED`）を含むリーチ者がいない
4. 副露者がちょうど1人。副露者は「リーチしておらず、暗槓以外の副露がある他家」で、
   `lisjong.learning.open_wait_estimator.is_open_opponent` と同じ定義（testで一致を確認）
5. 点数状況modeが `GameSituationMode.NORMAL`（オーラスのSPEED / VALUE modeは対象外。オーラスでもNORMALなら対象）
6. どの合法打牌でも聴牌にならない（C0の打牌後向聴数も1以上）
7. 現物は、その副露者自身の河にある牌種。鳴かれて河に記録が残る打牌も含み、赤5と通常5は同じ牌種。
   C0がすでに現物なら維持する
8. C0以外の現物の合法打牌のうち、C0と比べて次をすべて満たすものを代替候補とする
   - 打牌後向聴数が同じ
   - 現在受入（`structural_efficiency.ukeire_count`）が下がらない
   - retained real value が下がらない
   - yaku route value が下がらない

   候補がなければC0を維持する。あれば現在受入、retained real value、yaku route valueの大きい順、
   最後に `discard_action_sort_key` 順で1つ選ぶ。合法候補の `DiscardAction` をそのまま返すので、
   手出し / ツモ切りと赤5のidentityは保たれる。ランダム性はない

retained real value・yaku route valueは `hand_value_aware_two_step_ukeire` と同じ定義の既存proxyで、
得点期待値ではない。値はC0と現物の合法打牌についてだけ計算する。FiniteHorizon・R5・second-stepの
評価は追加しない（Champion自身の計算はそのまま1回行う）。

「最後の打牌後に通った牌」、未解決の打牌、学習した確率は使わない。

## 保証しないこと

- 河の現物は、その副露者に現在ロンされない根拠にすぎない。他の2人への安全、将来の安全、収支の改善は保証しない
- R4 completion mass・R5・second-stepの値が下がらないことは保証しない。攻撃の損失はありうるので、
  局・半荘の収支で検証する
- 打牌が変わると以後の展開が変わるので、対局全体の鳴き頻度や放銃率が不変とは主張しない

## `analysis`

打牌判断の `PolicyDecision.analysis` は常に `SingleOpenGenbutsuAnalysis` で、Championの
`TargetedHonorReleaseAnalysis` は含めない。Arenaは判断式を再実装せず、このrecordを集計する。

| field | 内容 |
|---|---|
| `stage` | `SingleOpenGenbutsuStage`。どのgateでC0を維持したか（`NOT_NORMAL_TURN` / `SELF_RIICHI` / `SELF_OPEN_HAND` / `OPPONENT_RIICHI` / `NO_OPEN_OPPONENT` / `MULTIPLE_OPEN_OPPONENTS` / `NOT_NORMAL_SITUATION` / `TENPAI_REACHABLE` / `PARENT_ALREADY_GENBUTSU` / `NO_GENBUTSU_ALTERNATIVE` / `NO_QUALIFYING_ALTERNATIVE`）、または替えた（`SWITCHED`）か |
| `parent_action` / `selected_action` / `action_changed` | C0、この候補の打牌、変更有無。変更するのは `SWITCHED` だけ |
| `open_opponent_seat` / `genbutsu_tile_types` | 副露者が1人に決まった判断だけ、その席と河の現物（牌種順） |
| `parent_in_genbutsu` | C0の現物判定まで進んだ判断だけ記録（それ以外は `None`） |
| `parent_evaluation` / `alternative_evaluations` | 比較値を計算した判断だけ、C0と、C0以外の現物の合法打牌の `GenbutsuCandidateEvaluation`（打牌後向聴数・現在受入・retained real value・yaku route value）。未計算の値を0で埋めない |
| `qualifying_actions` | 代替候補の条件を満たした打牌（`discard_action_sort_key` 順） |

## 情報境界

入力は `PolicyInput` だけ（自席の手牌、公開の河・副露・リーチ状態・点数・ドラ表示牌）。他家の伏せ牌、
山、結果、将来のevent、学習artifactは使わない。hidden truth、Learning、Arena、engineをimportしない
（`check_candidate_imports` / `check_candidate_purity` をtestで確認）。Policy instanceに
cross-decision stateを持たない。

## 範囲外

- 強さ・変更頻度の評価、対局、seed予約、Champion変更。評価の手順と停止条件は lisbun/lisjong-arena#524 で扱う
- #330 paijia、#312 確定安全牌、#254 聴牌PUSH/FOLDの再調整ではない
