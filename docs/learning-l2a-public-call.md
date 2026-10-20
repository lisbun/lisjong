# L2a後続：公開解析値をR1へ明示する限定開発比較（#326）

## 状態と位置付け

#324で門前Chiの模倣が汎化しなかった（門前teacher-Chi recall TRAIN 679/699、SELECT 21/93）。
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

## 実測結果

（実行後に追記する）
