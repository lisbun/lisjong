# P1価値モデル：source reader・10列encoding・ridge・E2 / E3（Issue #366）

## 実験の現在地（2026-11-10 JST）

[Arena #558の結果レビュー](https://github.com/lisbun/lisjong-arena/issues/558#issuecomment-6661359139)により、
今回の10特徴・ridge λ=10の実験は**NOT-PREDICTIVEで完了**。
E2 ΔMSE −64.25、bootstrap P5 −127.63 / P95 −2.06で事前条件を満たさなかった。
追加seed・再fit・TEST再評価・E4へは進まない。P1系列やLearning全体の不可能性は主張しない。
TRAIN 556000..556119・TEST 556200..556319は使用済みで、TESTを調整や独立評価へ流用しない。
下記は実装契約の記録であり再実行指示ではない。次の新学習仮説は
[project #74](https://github.com/lisbun/lisjong-project/issues/74) / [#84](https://github.com/lisbun/lisjong-project/issues/84)で別に選定する。

設計の正本: [#365](https://github.com/lisbun/lisjong/issues/365)（[設計案](https://github.com/lisbun/lisjong/issues/365#issuecomment-6599819983)・
[予算と配分](https://github.com/lisbun/lisjong/issues/365#issuecomment-6600146234)・[完了記録](https://github.com/lisbun/lisjong/issues/365#issuecomment-6600644792)、DESIGN-READY）。
source producer: [Arena #556](https://github.com/lisbun/lisjong-arena/issues/556)（`pon-first-pass-source-v1`）。pilot: Arena #555。
**本書は実装の契約で、生成・fit・TEST評価の実行は別の実行Issueで行う。** Championは変えない。強さ（E4）とgate Policyは範囲外。

| module | 内容 |
| --- | --- |
| [`lisjong.learning.p1_value`](../src/lisjong/learning/p1_value.py) | source reader、10列encoding、ridge、artifact |
| [`lisjong.learning.p1_value_evaluation`](../src/lisjong/learning/p1_value_evaluation.py) | fit / evaluate CLI、E2 / E3 |

ML frameworkに依存しない。`lisjong_arena`をimportしない。R1（#360）のsource helper・`fit_ridge`・`cluster_bootstrap`を
引数（列・λ・乱数seed）付きで再利用する（R1の既定値は変えない）。

## 1. 目標量と母集団

分岐点（各席・各局で、門前のときにChampionがPonを選んだ最初の判断、#362）で、a_R = Pon、a_D = Pass（以後はChampion）の、
分岐席の対象局の局収支の差 Δ = G_D − G_R（100点単位）。学習・評価の対象は**他家ロンで取り消されなかった成立分岐**。
ロン取消の候補は分岐にしないが件数を読み、E3の全対象判断あたりの分母に含める。

## 2. source（Arena #556が出すblock directoryの契約）

```text
<block>/
  SHA256SUMS       directory内の全file（SHA256SUMS以外）と完全一致
  branches.jsonl   1行目header、以降1候補1行（Arena #553の候補行と同じfield）
  summary.json     schema / decision（SOURCE-COMPLETE）/ plan_digest / block / split / invalid.count（0）を読む
  actions.jsonl・hanchans.json・timings.json   内容は読まない（file集合とhashだけ照合）
```

- directoryのfile集合は上の6つと完全一致（`SOURCE_FILES`）。SHA256SUMSと実directoryの双方から欠けていても拒否する
- 1つのsplitの3ブロックは同じ`allocation_identity`（予約はsplit単位）。TESTの3ブロックは同じ`artifact_identity`

- header: `allocation_identity`・`artifact_identity`（TESTだけ）・`block`・`call_analysis_version`（`call-value-shadow-v1`）・
  `kind`・`plan_digest`・`protocol_version`（`pon-first-pass-source-v1`）・`schema`・`split`・`wrapper_version`
  （`pon-first-pass-branch-wrapper-v1`）
- block: TRAIN-1〜3（556000..556039 / ..079 / ..119）、TEST-1〜3（556200..556239 / ..279 / ..319）、各40 seeds
- **特徴の入力は候補行の`decision_input`（context・Pon・鳴き解析）だけ**。`selection`（`called_discard_order`・`resolution`）は
  標本の選択（`ESTABLISHED`は分岐、`CANCELLED-RON`は取消として数える。それ以外はエラー）にだけ使う。D armの値は精算と
  照合（`status` branched・`invalid`空・`pass_applied` true）にだけ使う
- 未知のschema・field、field集合の混在、欠落・追加file、hash不一致はエラー

## 3. encoding（`lisjong-p1-value-encoding-v1`、fingerprint付き）

| # | 列 | 型 |
| --- | --- | --- |
| 1 | 鳴き解析（#350）のrouteがSPEED_CALL（PARENT_DECISIONなら0） | 二値 |
| 2 | 鳴かれた牌が役牌（Championと同じ判定。三元牌・場風・自風） | 二値 |
| 3 | Pon前の向聴（自手13枚） | 連続 |
| 4 | Pon後の最小向聴（消費牌2枚を除き、食い替えで禁止される牌種を除いた打牌の中で最小） | 連続 |
| 5 | ドラ枚数（自手・自席の暗槓・鳴かれた牌。Ponの消費牌は手牌側で1回だけ数える。公開ドラ表示牌から、赤5を1枚ずつ加える） | 連続 |
| 6 | 自分の打牌数 | 連続 |
| 7 | 残り山の枚数 | 連続 |
| 8 | 他家にリーチ（宣言中を含む）がいる | 二値 |
| 9 | 親 | 二値 |
| 10 | トップとの点差 / 1000（0以上） | 連続 |

- routeはsourceに保存された値を使い、保存したcontextとPonからの再導出（Championが実際にそのPonを選ぶことを含む）との一致を要求する
- 欠損・非有限値はエラー（行を除外して続けない）

## 4. ridgeとartifact

- minimize Σ_i (Δ_i − b − x_iβ)² + 10 Σ_j β_j²（切片は罰則なし、和）。連続列（3〜7・10列目）はTRAINで母分散標準化し、分散0ならエラー
- artifact: `lisjong-p1-value-artifact-v1`、model `lisjong-p1-value-ridge-v1`。immutable directory（`manifest.json`、sealed、strict load）。
  TRAINの3 blockのidentity・行数・取消数を記録する。TRAIN以外の行ではfitしない

## 5. E2 / E3（TESTを1回だけ読む）

| 項目 | 定義 |
| --- | --- |
| E2（主） | ΔMSE = TEST全成立分岐の平均 of [(Δ − m)² − (Δ − ŷ)²]（m = TRAIN平均） |
| E3（副） | V = TEST全成立分岐の平均 of I[ŷ > 0]·Δ。全対象判断あたり（ロン取消を分母に含める）も記録 |
| bootstrap | TEST 120 seeds（昇順）、統計量ごとに`random.Random(365)`、`randrange`で復元抽出10,000回、抽出seedの全行を重複込みで合算、P5 = 昇順500番目 |
| 判定 | E2のP5 > 0でなければNOT-PREDICTIVE。E2通過・E3のP5 > 0でなければPREDICTIVE-ONLY。両方通過でDECISION-VALUE-SIGNAL（E4の設計候補で、採用の根拠ではない） |
| 1回だけ | 正規の評価先はartifactの実体pathの親の`p1-value-evaluation-<artifact identity>/`（保存先は指定できない）。TESTを読む前にこのdirectoryを排他的に作り（`mkdir`）、開始記録`started.json`を書く。作れなければ（完了・実行中・途中失敗のいずれでも）TESTを読まずに止まる。途中失敗は再実行せず報告する。結果は同じdirectoryの`evaluation.json`。TESTで再fitしない |

E2の検出力は未取得（#365）。平均差の規模計算（K = 296）はE2に使わない。

CLI:

```bash
python -m lisjong.learning.p1_value_evaluation fit --train <TRAIN-1> <TRAIN-2> <TRAIN-3> --output <artifact>
python -m lisjong.learning.p1_value_evaluation evaluate --artifact <artifact> --test <TEST-1> <TEST-2> <TEST-3>
```

artifactを別の場所へ複製すると評価先も変わるので、実行Issueではfit直後にartifactの正規path（とidentity）を記録し、
評価はそのpathだけで行う。
