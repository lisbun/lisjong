# L2a後続：公開解析値をR1へ明示する限定開発比較（#326）

## 状態と位置付け

前段の#324で門前Chiの模倣が汎化しなかった（門前teacher-Chi recall TRAIN 679/699、SELECT 21/93）。
本Issueは、教師C0が公開情報から計算する「鳴く前後の向聴」「最小向聴を達成する
（打牌、route）のvalue」「点数状況mode」をR1へ1組の介入として明示すると、
既存SELECTでCALL模倣が改善するかだけを見る。

- 比較対照A：#318の凍結R1 `r1-w256`（identity `d565713f…`、selected epoch 5）。再fitしない
- 候補B：R1の95次元を変えずに下記11次元を足した `r1-public-call` を、幅256で1回だけscratchから学習
- データ：#506 source（identity `ba02b591…`）のTRAIN 600半荘・SELECT 100半荘だけ。
  OFFLINE-EVAL 506700..506849はmanifestの契約確認以外に開かない
- TRAIN / SELECTは開発・epoch選択・仮説形成に使用済みである。本比較は探索的な開発比較の
  明示的な例外であり、独立評価・棋力・昇格の根拠にしない。1 training seedの結果である

## encoding `r1-public-call`

正本は`lisjong.learning.l2a_encoding`の`ENCODING_R1_PUBLIC_CALL`（identity
`lisjong-l2a-reaction-candidate-encoding-r1-public-call-v1`、106次元）と
`public_call_analysis()`である。R1の95次元は意味・順序・候補順・赤牌区別とも同じで、
R1 fingerprint `732ca955…`はAのmanifestと一致する（testで固定）。

| block | 次元 | 定義 |
|---|---|---|
| current_shanten | 2 | mask、向聴 / 13。門前（暗槓のみを含む）は七対子・国士を含む`calculate_shanten`。副露は既存副露と矛盾しない役route（Tanyao / Honitsu萬・筒・索）の最小route向聴。routeなしはmask 0 |
| joint_route_shanten | 2 | mask、向聴 / 13。Chi / Ponは喰い替えで許される打牌と適合routeの全組の最小。Passは鳴かない現在の手の役route最小。routeか許される打牌が無ければmask 0 |
| shanten_improvement | 2 | mask、(current − joint) / 13。両方有効な場合だけ有効。Passは差の基準として値0（maskはcurrentのmask） |
| joint_route_value | 2 | mask、value / 64。最小向聴を達成する組の最大value。Tanyao 1 / Honitsu 2 / 字牌の無いHonitsuはChinitsu 5 + route内の純手牌と副露牌のドラ（表示牌の重複を数える）・赤5 |
| situation_mode | 3 | NORMAL / ALL_LAST_TOP_SPEED / ALL_LAST_LAST_VALUE one-hot。南4局・西入以降で単独トップ、または子の単独ラスで3着との差が4000点超 |

- 入力は`PolicyInput`の公開情報だけである。教師の選択・mode別採用閾値の通過・順位・logit、
  parent Policyのbranchは入れない。新しいguardや候補削除もしない
- 教師Policyのhelperはimportせず、規則をこのmoduleが独自に持つ。教師との一致は
  testが教師のhelper（`_best_route_evaluation`、`_post_call_stable_hands`、`situation_mode`）と
  fixtureごとに照合する
- valueが0..64、向聴が0..13を外れたら丸めずfail closedする。R1自体の喰い替え不能時の
  fail closedは変えない

## 実行手順

実行CLIは`python -m lisjong.learning.l2a_public_call_experiment <phase>`。
判定基準・bootstrap・予算は`CONFIG`に固定し、各phaseのreportは前段のdigestへbindする。

1. preflight：TRAINのseed順最初の10半荘でA/B encodingとAの採点を計時し、70倍で見積もる
2. lock：Aのmanifestから幅・epoch数・batch・学習率・weight decay・seed・split・objective・
   optimizer・torch versionを全数値継承し、code revision・source / encoding identity・環境・
   thread数・予算・第1層のparameter増を記録する。不明値をdefaultで補わない
3. baseline：Aを同じ環境で再採点し、#324 `full.json`（SHA-256 `6d95bd08…`）の件数と
   完全照合する（marginの丸め差は別記）。Issue記載のA件数も照合し、未知のbaseline件数
   （false Pon総数等）をBの結果を見る前に算出する。不一致ならSTOP
4. materialize：TRAIN / SELECTのsummaryとallocation bindingだけを持つsource viewから
   Bのdatasetを1 processで作る。OFFLINE-EVALのsummaryを含まないので、そのpayloadは開けない
5. fit：Aと同じ条件で幅256を1回だけ学習し、全epochのTRAIN / SELECT CEを記録して
   SELECT平均CE最小のepochを選ぶ（同値は最初）。最初の正式epochの時間から残りepochと
   compareを見積もり、1.5倍が残予算を超えれば停止する
6. compare：A/Bを同じ行で採点し、Bのdataset payloadとservingのvectorをbyte単位で照合する。
   Aの再採点がbaselineと一致しなければ停止する。集計・bootstrap・判定を出す

予算は全phase合計wall 2時間・peak RSS 8GiB。各phaseを`timeout`と`/usr/bin/time -v`の下で
実行し、協調的checkだけを厳密なhard capとは呼ばない。source復元の時間・diskは別記する。

## 集計と判定

#324の分類・門前定義・候補数bin（1 / 2 / 3+）・赤牌区別・ゼロ分母null・半荘supportを
再利用し、門前のPonのみ合法行（`pon_only`）を加える。TRAIN / SELECTを別に集計し、
教師Pass / Chi / Pon×予測の3x3、Chi / PonのTP / FN / FP / TN・recall・specificity・BA、
teacher-positiveの分解、Chi限定argmax、rank、margin四分位、CALLのCE、半荘ごとの件数を
A / Bとも保存する。

主判定（SELECT）はIssue #326の5条件（`CRITERIA`）をすべて満たすときだけ
`PROMISING-DEVELOPMENT`とし、1つでも未達なら`NOT-SUPPORTED-DEVELOPMENT`、
不正入力・契約違反・予算超過・実行不能は`STOP/INVALID`とする。
主指標の区間はSELECT 100半荘のpaired cluster bootstrap（10,000回、
`random.Random(324).choices`、抽出した半荘の分子・分母を合算、percentile 2.5 / 97.5%の
線形補間）。正例分母0のreplicateが出たら区間判定不能として停止側に分類する。
区間は固定モデル・既知SELECTの半荘変動だけを表し、独立検定や学習seedの不確実性ではない。

## 実測結果（2026-10-21、1回）

**結論：`NOT-SUPPORTED-DEVELOPMENT`。** 今回の特徴追加案を支持する証拠は得られなかったので、
CALL追加実験は見送る。入力表現仮説全体の棄却ではない。門前Chi recallはSELECTで大きく上がったが、
TRAIN−SELECT差の縮小、門前false Chi、副露false Ponが事前固定の基準を満たさなかった。

### 実行条件

- code：PR #327 head `be2b3b5`。WSL Ubuntu 24.04、CPython 3.14.7、torch 2.13.0+cpu、Rust backend
  （native API 3、`SOURCE_REVISION` `e46fc19`、#324と同じwheel）、1 process・torch 1 thread
- 各phaseを`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`・残予算の`timeout`・
  `/usr/bin/time -v`の下で実行した。新規AWS 0、source生成 0
- source：archive SHA-256 `f7c5d0a3…`とevidence（replay mismatch 0）を検証して新しいdirへ復元
  （wall 2:58、2.8GB。予算外として別記）
- lock・baselineはBの最初のfit前にIssue #326へ記録した。最初のlock / baseline呼び出しはshellの
  変数展開の誤りで入力pathが空になってSTOPした（データ未読）。そのreportは退避し、同じcodeで再実行した

| phase | wall（OS） | peak RSS | 内容 |
|---|---:|---:|---|
| preflight | 5.8秒 | 267MiB | TRAIN 10半荘・CALL 1,092行。見積り計約651秒 |
| lock | 0.9秒 | 236MiB | 学習条件をAのmanifestから継承 |
| baseline | 3:42 | 279MiB | Aの再採点。#324と全件一致 |
| materialize | 4:06 | 253MiB | TRAIN 73,491 / SELECT 12,996判断、候補188,433 |
| fit | 0:29 | 867MiB | 最初のepoch後の残り見積り318秒（予算内） |
| compare | 4:50 | 302MiB | A/B採点・train-serve照合・bootstrap |
| 合計 | 約13.2分 | 867MiB | 予算 2時間・8GiBの範囲内 |

### baselineの照合

- Aを同じ環境で再採点し、#324の`full.json`と、TRAIN / SELECT × all / closed / open × 全bin の
  件数がすべて一致した（mismatch 0件、margin四分位の丸め差もなし）。
  Issueに記載したAの件数13項目もすべて一致した
- 機械算出したbaseline：SELECT false Pon 全Pon合法行 198、門前 196、副露 2
- compareでのAの再採点はbaseline reportとdigestで一致した。Bのdataset（identity `036b483d…`、
  populationはTRAIN / SELECTだけ）は、servingで再encodeした候補vector・shared context・教師indexと
  全行byte単位で一致した

### B（`r1-public-call`、幅256）

- artifact identity `b054e2fd65f6088c9feb21a5cf0413696a51b2d490886b263708b07b8d440756`、
  weights SHA-256 `50c18a62…`、parameter 117,249（Aより第1層の2,816だけ多い）
- 学習条件はAと同じ（20 epochs、batch 256、lr 0.001、weight decay 0、seed 0、Adam）。
  SELECT CE最小でepoch 5を選んだ（偶然Aと同じepoch）

| epoch | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN CE | 0.1301 | 0.0758 | 0.0439 | 0.0277 | **0.0198** | 0.0127 | 0.0087 | 0.0065 | 0.0036 | 0.0010 | 0.0003 |
| SELECT CE | 0.1417 | 0.1180 | 0.1014 | 0.0932 | **0.0891** | 0.0928 | 0.0924 | 0.0989 | 0.1104 | 0.1207 | 0.1305 |

全CALLのCE（A → B）：TRAIN 0.0155 → 0.0198、SELECT 0.1085 → 0.0891（門前 0.0896 → 0.0655、
副露 0.3260 → 0.3603）。

### 主判定（SELECT）

| 条件 | 基準 | A | B | 判定 |
|---|---|---|---|---|
| 1. 門前Chi recallの上昇 | B−A ≥ +0.15、95%区間の下限 > 0 | 21/93 = 0.226 | 46/93 = 0.495 | **pass**（+0.269、区間 [0.188, 0.356]） |
| 2. 門前ChiのTRAIN−SELECT差 | \|B差\| ≤ Aの差の50%（0.373）、TRAIN低下 ≤ 0.02 | 差 0.746（TRAIN 679/699） | 差 0.478（TRAIN 680/699） | **fail** |
| 3. false Chi・副露Chi・wrong-candidate | 門前 ≤ 6、副露 ≤ 36、副露Chi ≥ 114/188、wrong 0 | 6 / 36 / 114 / 0 | **24** / 25 / 126 / 0 | **fail**（門前false Chi） |
| 4. Pon | recall ≥ 565/727、false Pon（全・門前・副露）≤ A、門前Ponのみ合法 ≤ 196 | 565 / 198・196・2 / 196 | 594 / 147・134・**13** / 134 | **fail**（副露false Pon） |
| 5. CALL mismatch | 全 ≤ 548、門前 ≤ 396 | 548 / 396 | 438 / 308 | pass |

区間（記述用、B−A、SELECT半荘のpaired cluster bootstrap 10,000回）：副露Chi recall [+0.025, +0.109]、
Pon recall [+0.018, +0.063]、門前false Chi件数 [+11, +26]、副露false Chi件数 [−18, −4]、
全CALL mismatch件数 [−138, −83]、門前mismatch件数 [−113, −64]。

### 補足の観察（判定には使わない）

- 門前ChiではPassへの取りこぼし（72 → 47件）が減った一方、教師Passの行でChiを選ぶ誤り（6 → 24件）が
  増えた。Chi候補が1つの行に集中している（recall 15/70 → 38/70、false Chi 6 → 24）。
  teacher-Chiでのchi-vs-non-chi marginの中央値は −1.69 → −0.01
- 門前のPonのみ合法行では、false Pon（196 → 134）とPonの取りこぼし（122 → 103）が両方減った。
  副露ではfalse Ponが2 → 13件に増えた
- TRAINでは、Bの全CALL mismatchが104 → 206件、false Ponが53 → 136件に増え、teacher-Chiの
  wrong-candidateが7件出た（Aは0件。条件3はSELECTだけを対象とする）。TRAIN CEもepoch 5で0.0155 → 0.0198と高い
- これらは1 training seed、使用済みのSELECTでの観察であり、個々の特徴の効果、情報の欠落、
  学習乱数への頑健性は検証していない

### 成果物

repository外の`C:\Dev\lisjong-artifacts\issue-326-l2a-public-call\`（`SHA256SUMS`付き）に保存した：
各phaseのreport（compare `9c973288…`、fit `87522952…`、materialize `b7869bef…`、
baseline `b0a85323…`、lock `b9c39ea4…`、preflight `1031bfe6…`）、stdout / stderr / exit、
B artifact（manifest・weights）、B datasetのmanifest、実行script。
datasetのpayload（230MB）はWSL上の`~/l2a-326/b-dataset`にだけ残した。weights・raw dataはGitに入れない。
