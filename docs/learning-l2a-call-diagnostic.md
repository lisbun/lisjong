# L2a後続の限定CALL診断（#324）

## 状態と対象

診断CLI・合成test・静的照合を実装し、凍結sourceとR1で実データのpreflight / 本診断を
2026-10-21に1回実行した（結果は「実測結果」節）。静的差分と本診断だけで
門前Chiの失敗原因を確定しない。

対象は#318の凍結R1（r1-w256、selected epoch 5）と既存sourceだけである。
identityは `lisjong.learning.l2a_call_diagnostic.CONFIG` に固定する。
TRAIN 506000..506599、SELECT 506600..506699を別々に集計する。
両方とも開発に使用済みなので、独立評価・棋力改善の根拠にはしない。
OFFLINE-EVAL 506700..506849はmanifestの存在確認以外に開かない。
source生成、再学習、再選抜、ledger更新、新しいAWS実行は行わない。

## 教師とR1入力の静的照合

参照元は `placement_aware_speed_call.py` の `_decide` / `_speed_call_candidates` /
`_best_route_evaluation` / `_route_value`、`l2a_encoding.py` の
`_encode_reaction_r1` / `_call_route_values`、`features.py` のshared context。
教師のsource生成revision `174bf48556e6ee0b7f4cda5cb8b93262140e7386` と
実装開始時main `a70c1b4` のplacement / yakuhai policyには差分がない。
この診断runtimeは教師のprivate helperをimportしない。

| 教師の判断材料 | R1 + shared contextとの対応 | 分類・制約 |
|---|---|---|
| Pass / Chi / Pon、鳴く牌・消費牌・赤牌 | kind、called、consumed blocks | 直接表現 |
| 他家リーチならspeed callしない | 各家riichi | 直接表現 |
| 門前／副露 | 自席のmeld-kind counts | 導出可能。暗槓のみなら門前 |
| score / roundによるNORMAL、ALL_LAST_TOP_SPEED、ALL_LAST_LAST_VALUE | 点数・局・親等のshared context | mode自体は非明示。公開情報から導出可能 |
| 鳴く前との厳密な向聴改善 | 各routeの向聴とPassとの差 | 直接表現。ただし教師の門前基準は七対子・国士も含む向聴、R1通常routeは通常形。特殊形は手牌から導出可能だが非明示 |
| 喰い替え後の打牌・役routeの適合 | routeごとの成立mask・独立最小向聴 | 直接表現。ただし教師のroute/value共同選択とは一致しない |
| Tanyao / Honitsu / Chinitsuの役翻と、同じ向聴を達成する打牌で残すドラ・赤によるvalue | Tanyao / Honitsu route、手牌・赤・ドラ表示牌 | Chinitsu専用route、最小向聴を達成する打牌とvalueの結び付きは非明示。暗槓のない門前では公開入力から導出可能。不可逆な情報欠落とは断定しない |
| mode別value閾値 | modeとvalueの合成結果は入力にない | 門前通常時は聴牌0／非聴牌2、トップ速度時0、ラス打点時3。学習器が合成する必要がある |
| 親policyが先に選んだPon等を優先 | 親のbranch / 選択結果は入力にない | 非明示。合法候補・手牌等からの同値導出は未検証 |
| 開いた役牌の有無 | own_yakuhaiは副露役牌と手牌暗刻をまとめる | 教師のopen-yakuhai判定と別物。既存副露の牌ごとの構成はshared contextに直接ない。R1派生値を合わせた復元可能性は未確認 |
| 最小route向聴、最大value、action sort順の辞書式選択 | 個別候補scoreとcanonical argmax | 選択規則は非明示。candidate tieも教師と同じとは仮定しない |

「確認済みの不可逆欠落」とするには、同一の全model入力で教師判断が異なる例等が必要。
今回はその証拠がなく、非明示な解析値を一律に欠落と呼ばない。
特に副露比較群・暗槓持ちの復元可能性には未確認事項が残る。

## 学習目的の照合

`candidate_training._per_decision_loss` と `l2a_training.train_l2a_scorer` は、
各判断の合法候補scoreに対するsoftmax CEを計算し、batch内の判断で平均する。
候補ごとの独立二値lossではない。TRAINの判断をepochごとにshuffleして各1回使い、
class weighting / oversamplingは行わない。最後の小さいbatchもそのbatch平均で更新する。
SELECTの判断平均CEが最小のepochを選び、同値なら先のepochを維持する。

したがってChi-positiveの少なさが全体目的への寄与を小さくする可能性はあるが、
それだけで誤りの原因とは言えない。本診断ではCALL全体と門前Chi合法のteacher-positive /
negativeの件数・半荘supportを対比し、凍結manifestのepoch別TRAIN / SELECT CEも併記する。
R1のepoch 5選択だけから過学習・学習不足を断定しない。manifestのdiagnosticsはreportの
`artifact_training`へ保存する。実測値の解釈は「実測結果」節に記す。

## 集計の読み方

- `groups.closed.chi_legal` が主対象、`groups.open.chi_legal` が同定義の比較群。
  `all` はPonのみ合法も含むCALL全体、`1` / `2` / `3+` はChi候補数別。
- 3x3 matrixはteacher行・prediction列。各件数にdistinct seed（半荘）supportを付ける。
  seat行を独立標本として信頼区間を作らない。
- Chi / Ponのrecall、specificity、balanced accuracyは当該行動が合法な判断が分母。
  teacher-positiveはnon-kind、wrong-candidate、exactの排他的3区分。
  赤牌消費だけ異なるChiも別候補。ゼロ分母はnullと理由を返す。
- `closed_chi_contribution` は門前Chi関連mismatch（non-Chi予測 + wrong-Chi + false-Chi）
  / 同splitの全CALL mismatch。双方の件数と半荘supportは各groupのcountsで確認できる。
- teacher-Chiに限って全候補中rank、Chi限定argmax一致率、maxChi−maxNonChi、
  teacherChi−maxOtherChiの四分位・中央値・符号を出す。同点はserving同様canonical先頭。
  他Chiなしのmarginは対象外件数を明記する。score差は確率ではない。
- 非CALL行は分類別除外数として残す。teacher-rule exception、非有限score、
  teacher不在、不正候補・件数不一致はSTOPとする。

## 実行手順と停止条件

必要な既存入力は以下（repositoryにはcommitしない）。

1. #506 archiveの `source-record.tar.gz` と `evidence.json`。
   所有者保存先: `C:\Dev\lisjong-artifacts\issue-506-l2a-source\archive\`。
   既存の検証済み復元手順で展開したsource rootを指定する。CLIはarchive自体を展開しない。
2. #318の `r1-w256/manifest.json` と `weights.f32`。
   所有者保存先: `C:\Dev\lisjong-artifacts\aws-ec2\lisjong-318-l2a-train-eval-20261019T021418Z-c821ce91\evidence\artifacts\r1-w256\`。

既存Linux環境、通常CPython 3.14、`lisjong[ml]` と互換Rust native backendを用いる。
READMEのnative install手順に従う。診断commitがcleanなcheckoutで実行し、出力はrepository /
source / artifact外へ置く。Python / torch / threads / native binary hash / git SHAを記録する。
次の環境変数とpathは例であり、source/artifact identityの固定は解除できない。

```bash
export LISJONG_SHANTEN_BACKEND=rust
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m lisjong.learning.l2a_call_diagnostic \
  --source /data/l2a-source --r1 /data/r1-w256 \
  --phase preflight --output /data/diagnostic-324/preflight.json
python -m lisjong.learning.l2a_call_diagnostic \
  --source /data/l2a-source --r1 /data/r1-w256 \
  --phase full --preflight /data/diagnostic-324/preflight.json \
  --output /data/diagnostic-324/full.json
```

preflightはseed順の最初のTRAIN 10半荘のみを読み、setup込みwall timeを70倍して
700半荘の所要時間を保守的に外挿する。外挿2時間超、peak RSS 8GiB超ならfullへ進まない。
fullは同一config・commit・runtimeのsealed preflightと予算内判定を必須とする。
実行中も各game読込前・各判断前・完了時にwall / peak RSSを検査する。
これは協調的な監視であり、単一読込・native callの途中を強制停止するものではない。
厳密な実行上限が必要な環境では既存のprocess supervisorにも2時間 / 8GiBを設定する。
外挿には最初の10半荘の偏りが残り、完走を保証しない。

入力不足・不整合・上限超過はSTOP reportとexit 2。既存reportを上書きしない。
import時のnative非互換、OSによる強制終了等はreport生成前に停止し得るので、stderr /
exit statusも保存し、reportがない実行を完了と扱わない。
追加AWS、代替source生成、weights再取得のための再学習へ自動的に切り替えない。

## 実測後の完了条件

preflight / full JSONと短い日本語の診断をIssueへ添付し、TRAIN / SELECTを混合せず、
行動種別・Chi内候補・入力表現・目的関数のどこまで観測で説明できるかを明記する。
次の介入は最大1案とし、反証条件と対象を絞る。根拠不足なら「未解決／追加実験非推奨」とする。

## 実測結果（2026-10-21）

WSL2（x86_64）、Python 3.14.7、torch 2.13.0+cpu（1 thread）、rust backend（native API 3、
PR CIのmanylinux wheel。本PRはnativeを変更しない）、診断revision `f59c03c` のclean clone。
sourceはarchiveをArenaの `restore` で新規に復元し、identity・SHA-256・replay証跡が一致した。

| phase | status | wall | peak RSS | report identity |
|---|---|---:|---:|---|
| preflight（TRAIN 10半荘） | COMPLETE、予算内（外挿357秒） | 5.1秒 | 266 MiB | `f22c4c54…d922f` |
| full（TRAIN 600 / SELECT 100） | COMPLETE | 228秒 | 268 MiB | `bbf58d28…a3359` |

reportは `C:\Dev\lisjong-artifacts\issue-324-l2a-call-diagnostic\` に置く（repositoryにはcommitしない）。

### 主対象（Chi合法の判断。括弧内は半荘support）

| split / 層 | 判断 | teacher Chi | 一致 | Pass予測 | 別Chi | false Chi | Chi recall | balanced acc. | Chi限定argmax一致 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN 門前 | 41,933 | 699（419） | 679 | 20（20） | 0 | 8 | 0.971 | 0.986 | 0.994 |
| SELECT 門前 | 7,317 | 93（57） | 21 | 72（52） | 0 | 6 | 0.226 | 0.612 | 1.000 |
| TRAIN 副露 | 4,144 | 1,122（474） | 1,120 | 2（2） | 0 | 5 | 0.998 | 0.998 | 1.000 |
| SELECT 副露 | 762 | 188（70） | 114 | 74（53） | 0 | 36 | 0.606 | 0.772 | 1.000 |

- SELECT門前のChi候補数別recallは1候補 15/70、2候補 6/9、3候補以上 0/14（TRAINは453/466、145/146、81/87）。
- SELECT門前のteacher-Chiでのmax Chi − max non-Chi scoreは中央値 −1.69（四分位 −2.56 / −0.20）、正21 / 負72。
- 門前のteacher-Chi比率はTRAIN 1.67%、SELECT 1.27%。副露は27.1%、24.7%。
- 凍結manifestのCEはepoch 5（選択）でTRAIN 0.0155 / SELECT 0.1085。SELECT CEはepoch 5が最小で、以後増加する。

### CALL不一致の内訳

| split | CALL不一致 | うち門前 | 門前Chi合法の行 | 門前のPonのみ合法の行（Pon見逃し / false Pon） |
|---|---:|---:|---:|---:|
| TRAIN | 104 | 96 | 28（`closed_chi_contribution` 0.269） | 68（15 / 53） |
| SELECT | 548 | 396 | 78（0.142） | 318（122 / 196） |

SELECTでのPon recallは565/727（0.777）、TRAINは3,859/3,875。

### 解釈（探索的。TRAIN / SELECTとも開発に使用済み）

1. **Chi内の候補選択は原因でない。** 両split・全層でwrong-candidateは0、SELECTのChi限定argmaxは
   teacher-Chiと全件一致する。teacher-Chiの不一致はすべて予測Pass（「Chiするか」の誤り）である。
   これとは別に、teacherがChiしない判断でのfalse ChiがSELECT門前6件・副露36件ある。
2. **TRAINでは再現でき、SELECTで落ちる。** 門前Chi recallは0.971 → 0.226、副露でも0.998 → 0.606。
   失敗は門前固有ではなく、teacher-positiveの少ない門前で最大になる汎化差として観測される。
   TRAINで再現できているので、batch平均CEで希少行動の寄与が小さい点はTRAINへの適合の妨げにはなっていない。
   class weighting / oversamplingでSELECTとの差が縮むかは未検証であり、この結果から支持しない。
3. **門前のCALL不一致の大半は門前Chiではない。** SELECTの門前不一致396件のうちChi合法の行は78件で、
   318件はPonのみ合法の行（主にfalse Pon）。#318の「CALL不一致の71.3%が門前」を門前Chiの寄与と
   読まないという前提を、SELECTでも観測として確認した（OFFLINE-EVALでは再集計していない）。
4. **入力表現は未確定。** 静的照合で非明示だった教師の判断量（mode別value閾値、特殊形を含む向聴改善、
   route / valueの共同選択）を学習器が代理特徴で覚えた可能性は、TRAIN適合とSELECT低下に整合する。
   ただし門前teacher-Chi 699行 / 419半荘というsupportの少なさでも同じ形になり得るため、
   本診断ではこの2つを区別できない。不可逆な情報欠落の証拠も得ていない。

### 次の介入案（1案。実行は別Issueで判断する）

R1の反応encodingに、教師の判断量のうち公開情報から導出できるもの（Pass比の特殊形を含む向聴改善、
最小向聴を達成する打牌でのroute value、mode）を候補特徴として明示した版を1つ作り、
同じsource・TRAIN / SELECT・学習設定・選択規則でR1と比べる。CALL headの変更なのでPonにも影響するため、
Ponの改善・悪化を併せて確認する（Ponへの効果は未検証）。OFFLINE-EVALは使わない。
中止条件：SELECTの門前Chi recallがR1の21/93から半荘clusterで有意に改善しない、
または改善してもTRAIN / SELECTの差が同程度に残るなら、今回の特徴追加案を支持する証拠は得られなかった
として、CALLについての追加実験を見送る。これは入力表現の仮説全体の棄却を意味しない。
