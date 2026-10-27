# R1価値モデル：推定宣言牌・19列・ridge・gate Policy・E2 / E3（Issue #360）

設計の正本: [#356](https://github.com/lisbun/lisjong/issues/356)（[固定案](https://github.com/lisbun/lisjong/issues/356#issuecomment-6417097132)・
[改訂・引継ぎ版](https://github.com/lisbun/lisjong/issues/356#issuecomment-6418811229)）。source producer: [Arena #548](https://github.com/lisbun/lisjong-arena/issues/548)。
**本書は実装の契約で、生成・fit・TEST評価の実行は別の実行Issueで行う。** Championは変えない。強さ（E4）は範囲外。

| module | 内容 |
| --- | --- |
| [`lisjong.learning.r1_value`](../src/lisjong/learning/r1_value.py) | source reader、推定宣言牌、19列encoding、ridge、artifact |
| [`lisjong.learning.r1_value_policy`](../src/lisjong/learning/r1_value_policy.py) | gate Policy（Champion exact parent） |
| [`lisjong.learning.r1_value_evaluation`](../src/lisjong/learning/r1_value_evaluation.py) | fit / evaluate CLI、E2 / E3 |

ML frameworkに依存しない（lisjong core、Pythonの浮動小数）。`lisjong_arena`をimportしない。

## 1. 目標量

分岐点（各席・各局でChampionが最初に`RiichiAction`を選ぶ判断）で、a_R = リーチ（宣言牌X）と a_D = Xをリーチせずに打つ
（以後はリーチを含めChampion）の、分岐席の局収支の差 Δ = G_D − G_R（100点単位、リーチ棒・供託・本場を含む、Rで場に残った棒は
−1000、終局時の供託精算は除く）。A(s) = E[Δ | s]。局収支であり、順位の改善と読み替えない。

## 2. source（Arena #548が出すblock directoryの契約）

lisjongはこの形をstrictに読む（未知のschema・field、欠落・追加file、hash不一致はエラー）。

```text
<block>/
  SHA256SUMS       "<sha256>  <name>" の行。directory内の全file（SHA256SUMS以外）と完全一致
  branches.jsonl   1行目header、以降1分岐1行（Arena #544のbranch行と同じfield）
  summary.json     schema / decision / plan_digest / block / split / invalid.count を読む
  （その他のfileはSHA256SUMSに含まれていればよい）
```

| header field | 値 |
| --- | --- |
| `kind` / `schema` | `header` / `arena-riichi-first-dama-source-v1-branches-v1` |
| `protocol_version` | `riichi-first-dama-source-v1` |
| `features_version` / `wrapper_version` | `riichi-dama-features-v2` / `riichi-first-dama-branch-wrapper-v1` |
| `block` / `split` | `TRAIN-1`（547000..547021）・`TRAIN-2`（547022..547043）・`TEST-1`（547100..547121）・`TEST-2`（547122..547143） / `TRAIN`・`TEST` |
| `plan_digest` / `allocation_identity` | SHA-256 hex |
| `artifact_identity` | TRAINは`null`。TESTは凍結artifactのidentity（必須） |

summaryは `schema = arena-riichi-first-dama-source-v1-summary-v1`、`decision = SOURCE-COMPLETE`、`plan_digest`・`block`・`split`が
headerと一致、`invalid.count = 0` を要求する。

分岐行から読む値: `seed`（blockの範囲内）、`seat`、`context`（`legal_actions`と`policy_input`。lisjongの`parse_policy_input_v2` /
`parse_action`で読む。Arenaの保存形式と一致することを確認済み）、`declaration`（分岐席の`DiscardAction`）、
`r_settlement.point_deltas[seat]`（G_R）、`d.settlement.point_deltas[seat]`（G_D）。`r_invalid`が空、`d.status = branched`、
`d.invalid`が空、`d`のseed・seat・branch ordinalが行と一致、局収支が100点の倍数であることを要求する。

`read_source(dirs, split)` はそのsplitの2 blockちょうどを要求し、TESTの2 blockは同じartifactに束縛されていることを要求する。

## 3. 推定宣言牌（推論時）

Championは常にリーチし、宣言牌は次の判断（自席DECLARED）で選ぶ。`project_declaration(decision)` は分岐点の`DecisionContext`から、
自席の公開リーチ状態をDECLAREDにし、聴牌を保つ打牌（元の順序）だけを合法手とした判断を作り、新しいChampionが選ぶ打牌を返す。
既存の開発記録（Arena #542の286分岐・#546の203分岐）では、推定宣言牌はRの記録した宣言牌と全件一致した（形式・挙動の確認だけで、
Δは読んでいない）。

## 4. 19列（encoding identity `lisjong-r1-value-encoding-v1`）

宣言牌を基準に`analyze_riichi_decision`（`riichi-dama-features-v2`）と`DecisionContext`の公開値から作る（学習時はRの記録した宣言牌、
推論時は推定宣言牌）。W_live = 見えない残り1枚以上の待ち。W_live空なら15・16列は0。

| # | 列 | # | 列 |
| --- | --- | --- | --- |
| 1 | ダマのロンに役がある待ちあり | 11 | 順位3 |
| 2 | 振聴 | 12 | 順位4 |
| 3 | 親 | 13 | Wの見えない残り合計 |
| 4 | オーラス相当 | 14 | 自分の打牌数 |
| 5 | ダブルリーチ | 15 | W_liveのダマのロン最高点 / 1000（役なし0） |
| 6 | 他家リーチ1人 | 16 | W_liveのリーチによる上積み最大（リーチのロン − ダマのロン）/ 1000 |
| 7 | 他家リーチ2人以上 | 17 | 残り山の枚数 |
| 8 | 待ち種類数2 | 18 | 供託本数 |
| 9 | 待ち種類数3以上 | 19 | 本場 |
| 10 | 順位2 | | |

1〜12は0 / 1、13〜19は連続列。非有限値・解析の拒否はエラー（行を除外して続けない）。結果・山・他家の手牌・分岐後の行動・終端値は
入力にしない。

## 5. ridgeとartifact

- **minimize Σ_i (Δ_i − b − x_iβ)² + 10 Σ_j β_j²**（切片bは罰則なし、行の重みは等しい、和）。正規方程式を部分pivot付きGauss消去で解く
- 連続列（13〜19）はTRAIN行の平均と母分散（nで割る）で標準化。分散0の連続列はエラー。0 / 1列はそのまま
- TRAIN行だけでfitする（splitを検査）。TRAIN平均mも保存する
- artifact: 1 directory = `manifest.json`だけ（canonical JSON、`identity` = 本体のcanonical digest）。encoding / model identity・
  fingerprint、fit（λ・目的関数・標準化・切片・係数・m）、TRAIN blockのidentity、診断（行数・推定宣言牌の一致数）を持つ。
  callable / pickleを保存しない。既存pathを上書きしない。load時にidentity・field・値の有限性・列・versionを検査してfail closed

## 6. gate Policy（`R1ValueGatePolicy`）

Championの行動が`RiichiAction`のときだけ評価し、**`dama_choice` = 推定宣言牌（牌・赤・ツモ切りflag）かつ 予測 > 0** のときだけ
推定宣言牌をリーチせずに打つ。それ以外（不一致・予測 ≤ 0・解析できない）はChampionの`RiichiAction`を返す。閾値0は固定。
判断をまたぐ状態を持たない。分岐点以外のリーチ判断にも同じ規則を適用するので、同じ半荘で繰り返し介入しうる（E3は強さ評価E4の
代わりにならない）。

## 7. E2 / E3（TESTで1回）

| 項目 | 定義 |
| --- | --- |
| E2（主） | ΔMSE = TEST全分岐の平均 of [(Δ − m)² − (Δ − ŷ)²]（ŷは凍結ridge、特徴は記録の宣言牌が基準） |
| E3（副） | V = TEST全分岐の平均 of I[推定宣言牌 = 記録の宣言牌 かつ `dama_choice` = 記録の宣言牌 かつ ŷ > 0]·Δ |
| bootstrap | TESTの44 seeds（昇順、行のないseedも含む）。統計量ごとに`random.Random(356)`を新しく初期化し、`randrange(44)`×44の復元抽出を10,000回。抽出seedの全行を重複込みでまとめ、分子（合計）・分母（行数）を再計算。P5 = 昇順500番目 |
| 通過 | E2: P5 > 0。E3: E2通過、SUPPORT-INVALIDでない、P5 > 0 |
| SUPPORT-INVALID | 推定宣言牌の不一致または`dama_choice`の不一致のある行がTEST分岐の2%超 |
| 判定 | NOT-PREDICTIVE（E2不通過）/ PREDICTIVE-ONLY（E2のみ）/ DECISION-VALUE-SIGNAL（E2・E3通過。E4の設計候補で、採用の根拠ではない） |
| 報告 | 推定宣言牌の不一致率、`dama_choice`の不一致率、予測上の介入率、実際の介入率、介入した行の平均Δ |

NOT-PREDICTIVEは「この固定設計・予算では確認できなかった」と解釈する。TESTで再fit・特徴変更・閾値変更をしない。

## 8. CLI

```bash
python -m lisjong.learning.r1_value_evaluation fit --train <TRAIN-1> <TRAIN-2> --output <artifact-dir>
python -m lisjong.learning.r1_value_evaluation evaluate --artifact <artifact-dir> \
  --test <TEST-1> <TEST-2> --record-dir <records>
```

`evaluate`は `<records>/evaluation-<artifact identity>.json` をwrite-onceで書き、同じartifactの記録があれば拒否する
（TESTを1回だけ読む）。TEST blockのheaderの`artifact_identity`がartifactと違えば拒否する。

## 9. 検証

[`tests/test_r1_value.py`](../tests/test_r1_value.py)（source blockはtestで合成。解析は実lisjong、native scoringが必要）:
readerのfail closed（schema・split・TRAINのartifact・summary・invalid・plan・hash・追加file・blockの範囲外・D armの不備・
blockの組・TESTのartifact不一致）、推定宣言牌（DECLAREDの射影と聴牌を保つ合法手）、19列の順序と値、ridgeの参照実装（Fractionの
正規方程式）との一致・切片罰則なし・母分散・分散0の拒否、artifactのround trip・上書き拒否・改ざん・非canonical・TRAIN限定、
gate Policyの介入・予測 ≤ 0・不一致と解析失敗のfallback・対象外の判断でChampionと同一、bootstrapの参照実装との一致と500番目、
E2 / E3の推定量・SUPPORT-INVALID・TESTを1回だけ読むこと・artifactへの束縛、ML framework / Arenaをimportしないこと。
