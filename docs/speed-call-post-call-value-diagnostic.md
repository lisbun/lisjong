# 鳴き時の打点proxyと直後の本番打牌の整合診断（Issue #348）

親: [#297](https://github.com/lisbun/lisjong/issues/297)、
[project #84](https://github.com/lisbun/lisjong-project/issues/84)。
H4（鳴き）の次の仮説を具体化するための有限の診断であり、Policyの挙動は変えない。

**結論: `REPRODUCED-FIXTURE`。** current Heuristic Champion
`PlacementAwareSpeedCallPolicy`が鳴きの採否・順位に使ったroute打点proxyを、
直後の本番打牌が保持しない局面を手作りfixtureで確認した。採用閾値を下回るケース
（`BELOW_THRESHOLD`）と、閾値以上だが最大proxyより低いケース（`BELOW_MAX`）の両方がある。
**到達頻度・損失点・棋力への影響は未確認。**

- 基準revision: `89dd00d54d3651c1a8a720b2bea3933bb2ca0bd5`（#346 / #347後のRust必須revision）
- 実行環境: WSL2 Ubuntu 24.04、CPython 3.14.7、同revisionのnative bundle（API_VERSION 3、
  CI run 40070934802）
- test: [`tests/test_speed_call_post_call_value_diagnostic.py`](../tests/test_speed_call_post_call_value_diagnostic.py)

## 比較する量

鳴き判断と同じ定義のroute shanten / 打点proxyを、鳴き判断時と直後の打牌後で比べる。

- 鳴き判断時: `_speed_call_candidates`が選んだcallの値。喰い替えを除いた打牌後手牌のうち、
  最小route shantenと、その最小を達成するrouteの最大proxy
- 打牌後: 本番Policyが実際に選んだ打牌の後の手牌を`_best_route_evaluation`で評価した値
  （同じ互換route、同じドラ表示牌）
- proxyはroute翻相当（Tanyao 1 / Honitsu 2 / Chinitsu 5）とroute内で保持するドラ・赤ドラの枚数の和で、
  実際の翻・得点・EVではない。proxyの差を損失点や勝率に換算しない

分類:

| 分類 | 条件 |
| --- | --- |
| `CONSISTENT` | 打牌後のroute shanten・proxyが鳴き判断時と同じ |
| `BELOW_MAX` | route shantenは同じ、proxyは鳴き判断時の最大値より低いがmode別の採用閾値以上 |
| `BELOW_THRESHOLD` | route shantenは同じ、proxyが採用閾値を下回る |

速度指標として、各合法打牌のFiniteHorizon completion mass（本番の打牌選択が使う値）と
構造的な受入枚数（`ukeire_count`）を記録する。

## code path監査

1. 鳴き判断（`_speed_call_candidates`）はproxyを採否・順位に使う。
2. 鳴き直後の打牌では、`_route_preserving_actions`がroute shanten最小の打牌へ絞る。
   ここではproxy（ドラ・route翻）を使わない。
3. 絞った候補は親の打牌選択へ渡る。鳴き直後は`drawn_tile`が`None`なので
   `TargetedHonorReleaseTerminalProgressionPolicy`の#174拡張は発動せず、親の
   mechanism defense filter → `TerminalShantenProgression...`の選択になる。この選択は
   構造的completion mass（役・ドラを見ない）の最大を取り、HandValueAware（ドラ等を見る）は
   completion massが**完全に同点**のときだけ使う。

したがってproxyが保たれるのは、proxyを保つ打牌がcompletion mass最大と一致する場合か、
完全同点の場合に限られる。route shantenは手順1と2が同じ最小値を使うため一致する
（全fixtureで一致した）。

## 成立後contextの構成

鳴き前は本番Policyに`PassAction`とChi/Ponを渡してcallを選ばせ、親Policy単体が
`PassAction`を選ぶこと（speed call経路であること）を確認する。成立後は次の公開情報だけで
`DecisionContext`を作り、同じ本番Policyへ渡す。選択処理のコピーやmockは使わない。

- 手牌から`consumed_tiles`をexact identity（赤5 / 通常5を区別）で除去し、`drawn_tile=None`
- 自席の副露に`PublicMeld`を追加。鳴かれた牌はtargetの河の最後の牌で、`called_by`を自席にする
- 合法打牌は手牌の牌identityごとに1つ。現物・筋の喰い替え禁止牌種は除く
- 鳴き前後とも`derive_remaining_tile_inventory`でtile conservationを検査する

## fixture matrixと結果

matrixの行はIssueの指定どおり。具体的な手牌は手で導出し、exploratory runで分類を確認してから
testに固定した。exploratory runで試してtestに入れなかった局面（下記「限界」）も記録する。
mass / 受入は「本番の打牌 vs proxyを保つ合法代替」。

### 反例（REPRODUCED-FIXTURE）

| # | 条件 | 鳴き前（自摸なし13枚）→ call | 鳴き判断 (rsh, proxy) / 閾値 | 本番打牌 → (rsh, proxy) | proxyを保つ代替 | mass | 受入 | 分類 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | NORMAL・門前→1向聴・Tanyao Chi・ドラ8s（表示7s） | 34m 678m 222p 56p 468s → 2mチー(34m) | (1, 2) / 2 | 8s → (1, 1) | 4s (1, 2) | 45980 vs 40176 | 24 vs 23 | `BELOW_THRESHOLD` |
| R2 | NORMAL・門前→聴牌・Tanyao Chi・ドラ8s | 34m 678m 567p 22p 468s → 2mチー | (0, 2) / 0 | 8s → (0, 1) | 4s (0, 2) | 175590 vs 133994 | 4 vs 3 | `BELOW_MAX` |
| R3 | NORMAL・既副露(234mチー)→追加Pon・ドラ8s | 678m 22p 55p 468s → 5pポン | (0, 2) / 0 | 8s → (0, 1) | 4s (0, 2) | 175590 vs 133910 | 4 vs 3 | `BELOW_MAX` |
| R4 | NORMAL・Chinitsu Pon | 66m 111m 345m 789m 9m 1z → 6mポン | (0, 5) / 0 | 9m → (0, 2)（1z単騎の混一色） | 1z (0, 5)（清一色） | 149599 vs 149371 | 3 vs 3 | `BELOW_MAX` |
| R5 | NORMAL・Honitsu/Tanyaoの同向聴2route・Pon・1zが2枚見え | 33m 456m 678m 567m 4p 1z → 3mポン | (0, 2) / 0 | 1z → (0, 1)（4p単騎のタンヤオ） | 4p (0, 2)（1z単騎の混一色） | 132202 vs 105218 | 3 vs 1 | `BELOW_MAX` |
| S1 | SPEED（オーラス単独トップ）境界・R1と同じ手 | R1と同じ | (1, 2) / 0 | 8s → (1, 1) | 4s (1, 2) | 45980 vs 40176 | 24 vs 23 | `BELOW_MAX` |
| V1 | VALUE（オーラス大差ラス目）境界・R1＋5pもドラ（表示7s・4p） | R1と同じ | (1, 3) / 3 | 8s → (1, 2) | 4s (1, 3) | 41034 vs 35565 | 23 vs 22 | `BELOW_THRESHOLD` |

手計算での説明（R1）: 2mチー後の 678m 222p 56p 468s から、打8sは 678m 222p 56p 46s、打4sは
678m 222p 56p 68s で、どちらもTanyao 1向聴。proxyは打4sが Tanyao 1 + ドラ8s 1 = 2、
打8sが1。鳴きはドラ8sを含めたproxy 2で門前非聴牌の閾値2をちょうど満たして採用されたが、
本番打牌はcompletion massが大きい46sを残してドラを切る。68s嵌張の待ち7sは表示牌として1枚見えている。
失う速度はmass 40176/45980（約13%減）、受入1枚。

R4は受入が同じ3枚で、massの差は約0.15%しかないが、proxyは5から2へ下がる。

### 一致（CONSISTENT）

| 条件 | 局面 | 理由 |
| --- | --- | --- |
| 門前→聴牌・ドラが速い側（4s、表示3s） | R2の手 | 速度とproxyが同じ打8sを指す |
| 門前→1向聴・完全同点 | 34m 67m 567p 22p 3468s → 2mチー・ドラ8s | 打6s / 打8sのmassが完全同点で、HandValueAwareがドラを残す |
| 既副露→追加Pon・ドラが速い側 | R3の手・表示3s | 同上（速度と一致） |
| Honitsu Pon・ドラ（表示6m） | 1235799m 112233z → 9mポン | 打1zでproxy 3を保つ |
| Chinitsu・1zが2枚見え | R4の手 | 9m単騎は残り1枚のため、速度でも清一色側 |
| 同向聴2route・完全同点 | R5の手（1z見えなし） | 4p単騎 / 1z単騎が完全同点で、混一色を残す |
| 赤ドラ | 34m 678m 56p 0p 22p 468s → 2mチー | 5p / 0p切りが同構造で、通常5pを切って赤を残す |

### 不発動・除外（反例に数えない）

| 条件 | 結果 |
| --- | --- |
| VALUE・proxy 2（R1の手） | 閾値3未満でPass |
| 他家リーチ | Pass |
| 役牌（5z）ポン済み | speed call経路に入らず親の判断 |
| 非改善（既に聴牌） | Pass |
| 合法Passなし | 親が既存契約どおりfail closed（`YakuhaiCallPolicyError`） |
| 喰い替えで打てる牌がない（3副露、残り55m） | 候補から除外されPass |

## 限界

- 手作りfixtureだけで、random探索・実対局・教師データ・閾値gridは使っていない。
  **到達頻度・損失点・棋力への影響は未確認。**頻度の値を推測で補完しない
- 赤ドラだけでproxyを失う反例は作っていない（赤5は形の中央に入りやすく、試した局面では
  同点か速度側と一致した）。不可能とは主張しない
- Tanyao / Honitsuの2route分岐は聴牌（単騎の選択）でだけ作った。門前→1向聴では、
  pon前に構造上の余裕がある手は既に1向聴になり、route制限でpon後も改善しないため、
  試した範囲では鳴き候補にならなかった
- route shantenの不一致（`ROUTE_SHANTEN_CHANGED`）は、code path上も起きない想定で、fixtureでも出なかった

## 後続への条件

- 本結果だけでは#481の候補列やS1/S2へ接続しない
- 「鳴いた直後の打牌で速度と保持価値をどう交換するか」を再設計するなら、先に**一度だけ**限定shadowの
  計画を別途固定し、実際に打牌が変わる頻度と速度 / 打点tradeoffを確認してから、候補仕様・identity・
  S1/S2予算を決める
- 限定shadow（Arena #533 call-value-shadow-v1）用に、本診断の比較量をproduction-ownedな解析APIへ
  移した（#350、[解析API](call-value-shadow-analysis-api.md)）。本testは診断記録として維持する
- R4のように速度差がごく小さいままproxyが大きく下がる局面と、R2 / R3のように受入1枚と
  ドラ1枚を交換する局面は性質が違う。shadowではこの2つを分けて数えるのがよい
