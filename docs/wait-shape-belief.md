# 形別待ちテーブル（Level 2）の推定・評価設計（#260）

[lisbun/lisjong#260](https://github.com/lisbun/lisjong/issues/260)（親: #255）の設計。
既存契約は`HandBelief`と[手牌正解データ契約](hand-belief-accuracy-source.md)を正本とする。
推論・学習・評価のcodeは実装済みだが、select・formal testは未実行である。
この文書は精度合格・Issue完了を意味しない。

## 現在地と実行の前提

#257は開発用測定まで完了し、#259の副露者推定はPR #266でmainへ入った
（merge commit `de786f75d2bd03d7943f621522c67ebd8545544c`）。#259範囲1のformal testは
2026-10-09に1回実行され、結果はpassだった（#259の結果コメント）。
#259のselectを再実行せず、登録済みのselection fileをSHA-256で照合して使う。

この文書が決めるのは対象、特徴量の範囲、比較baseline、整合性、形別評価条件である。
selectの実行commitとformal testのseed予約は未確定なので、この文書だけでは
実験開始用の事前登録は完了しない。実行commitを#260へ追記してからselectへ進み、
testの予約を記録してから生成・評価へ進む。

| 待ち推定器 | 固定するもの | 現在の扱い |
|---|---|---|
| 単独リーチ者 | #245の凍結済みselection、`riichi-wait-features-v1`、L2=1.0 | #257で使ったSHA-256 `14475264d7fe4137a9ac8a23ee1d27b420bff2f4434c6e93ca72ecfcf24ccc38`を指定し、受領時に実体と照合する |
| 副露者 | #259の`open-wait-features-v1`、修正後のselectのselection SHA-256・実行commit・選択L2 | SHA-256 `92608e4df7bbce35c1df26942aec29e11efe88526efc62b0bb7fd934e88165f6`（select commit `804b42b49228769d7c8e1c45e0699f430ef6c519`、L2は聴牌段0.01・待ち段0.01）。受領時に実体と照合し、再調整しない |

#259のmergeを精度合格と扱わない。別モデルへの差し替えは登録を更新してから行う。
#260の途中で#259のリーチ者改善モデルへ差し替えない。

## 対象とavailability

推論時の観測入力は`PolicyInput`だけとする。推定先の席と凍結したモデルは設定であり、
他家のconcealed tiles・正解・source recordを引数に持たせない。

| population | 推定を提供する行 | 学習・判定 |
|---|---|---|
| 単独リーチ者 | #245のS1対象条件: 他家1人がリーチ中（宣言済みを含む）、自分は非リーチ、合法打牌候補の牌種が2以上。そのリーチ者1席 | この対象内だけで独立に行う |
| 副露者 | `is_open_opponent()`と同じ。他家で非リーチ、暗槓以外の副露を1つ以上持つ席。観測者のリーチ・他家の別のリーチの有無では追加除外しない | 単独リーチ者と混ぜず、この層の行だけで行う |
| 門前非リーチ者、S1対象外のリーチ者、自席 | 今回の形別推定を未提供とする | coverage件数を記録し、改善判定へ入れない |

対象内では7channel全部を持つLevel 2の`HandBelief`を返す。対象外では形別groupは
全`None`とし、既存のLevel 0/1のavailabilityを保持する。部分的な形別groupや、
未提供を全ゼロへ置換した結果は作らない。unsupportedな席への直接推論要求は拒否する。
統計上の判定対象外は、推論のavailabilityとは別の評価結果である。

牌種marginal（期待枚数・赤5）は既存の条件付き一様推定器を使い、#258の改善を混ぜない。
`wait_probability`は固定したLevel 1推定器の値を保持する。

## 7channelと成立可能性

| channel | 構造上占め得るslot（静的集合） | slot数 |
|---|---|---:|
| tanki | 全牌種。七対子の待ちを含む | 34 |
| shanpon | 全牌種 | 34 |
| kanchan | 数牌2〜8 | 21 |
| penchan | 数牌3・7 | 6 |
| ryanmen low-side | 数牌1〜6 | 18 |
| ryanmen high-side | 数牌4〜9 | 18 |
| kokushi | 数牌1・9と字牌。tankiへ含めない | 13 |

静的集合の外はcanonical zeroとする。low/highは別の重み・指標・正例件数を持ち、
#257の丸め値が近いことから同一視したり、片側から反対側の確率をコピーしたりしない。

六つの通常形では`wait_shape_support()`が返す未見枚数の必要条件を使う。
必要条件を満たさない形のslotは0にできるが、満たすだけで待ちや聴牌とは断定しない。
待ち牌自身の未見枚数が0でも、その牌を手牌中に必要としない嵌張・辺張・両面の形を
一律0にはしない。河・現物・見逃し・役の有無で構造的な形を0にしない。

`wait_shape_support()`は国士を扱わない。国士へ六形のsupportを流用しない。
既知の副露が1つでもある席（暗槓を含む）では国士は構造上0。
副露がない席の国士は静的13slotを対象に推定し、この初版では追加の枚数maskを設けない。

## 推定方式と確率の整合性

populationごと・channelごとに、構造的形ラベルを目的変数とするL2付きlogistic回帰を作る。
channel間でsoftmaxや和=waitの正規化をしない。同じ牌種の複数channelが同時に正例になる
multi-labelをそのまま保持する。

整合性方針は**待ち確率による上限処理**とする。静的slot内で枚数supportがあるとき、
独立logisticの出力を`q[c,t]`、固定したLevel 1の出力を`w[t]`とすると、
最終出力は`p[c,t] = min(q[c,t], w[t])`。静的slot外・必要条件不成立・副露時の国士は0。
`HandBelief`のconstructorへ新しい制約は追加しない。

`probability_to_raw()`による既存のround-half-to-evenでwとpを同じcanonical axisへ変換する。
丸めの単調性によりrawでも`channel_raw <= wait_raw`を維持することをtestで固定する。
ゼロをepsilonへ置換するのはloss計算時だけとし、推論出力やcanonical zeroへは適用しない。

上限処理は誤入力を救済するfallbackではない。非有限係数、不正確率、未知feature set、
identity不一致、欠損モデルは拒否する。固定モデルの正常な予測にだけ上限処理を適用する。

特徴量は、単独リーチ者では#245の`wait_features()`、副露者では#259の
`wait_tile_features()`と`tenpai_features()`の公開情報を再利用する。同名のbiasは1つとし、
副露者の二つの特徴群にはprefixを付けて衝突を避ける。7channelは同じ特徴集合で別係数を持つ。
新しい手出し・宣言牌・裏スジ等の履歴特徴追加はこの初版に混ぜない。
feature identityは`wait-shape-features-v1`とし、population・channel順・特徴順も固定する。

学習では各populationのtrain行を使い、静的slot内を対象とする。必要条件不成立のslotは
定義上0なので係数学習へ寄与させない。静的slot数で行内平均を定義し、残るslot数で
行ごとに再正規化しない。元のpopulation内episode重みを使う。
L2は切片を含む全係数へ適用し、正例ゼロ・完全分離でも無正則化切片を発散させない。
目的関数は`Σ_行 episode重み × (静的slot内平均のbinary cross entropy) + L2 * Σ係数² / 2`とする。
第1項はepisode数で割らない（#259と同じく、episode重みの和の尺度で持つ）。episode数で割ると
格子の最小値0.01でも切片への罰則が損失より大きくなり、希少channelの確率を系統的に
押し上げるためである。この尺度は実装時（select実行前、結果を見る前）に固定した。
副露者の国士は構造上のゼロモデルとして明記し、係数fit・L2選択をしない。
solver failureはbaselineへの自動置換や候補の黙示的除外をせず失敗にする。

## baseline、select、開発データ

主比較baselineは**population・channel・牌種別の出現率**。population内のtrainだけで、
1episodeの重み1をそのpopulationに属する行へ等分する。静的slot内では
`(重み付き正例数 + 0.5) / (総episode重み + 1)`、静的slot外では0とする。
#257の全層を混ぜたbaselineの改善量を流用しない。

上限処理・枚数maskだけによる差を識別する副比較として、同じ出現率へ推定器と同じ
上限処理・maskを適用したbaselineも報告する。主比較を結果後にこちらへ変更しない。

| 分割 | seed | 用途 |
|---|---|---|
| train | 933000..933159（160半荘） | baselineと形別係数のfit |
| valid | 933160..933239（80半荘） | population・channel別のL2選択 |
| dev-eval | 933240..933399（160半荘） | 固定selectionの開発診断。合格判定に使わない |

L2格子は`0.01, 0.1, 1, 10, 100`の順。各population・channelについて、上限処理・mask・
fixed-point化後のvalidの静的slot内log lossが最小のものを選ぶ。同値なら格子の先を選ぶ。
判定対象外となる希少channelも、この規則を結果後に変えない。
校正の追加fitやdev-evalを使った特徴・閾値変更をしない。変更する場合は新しいfeature
identityと登録を作り、既閲覧の結果を未使用test扱いしない。

selectionには全候補のvalid指標、選択L2・係数、両Level 1 selectionのdigest、実行commit、
feature identity、source schema・manifest digest・producer・全split seed、整合性方針を記録する。
immutable snapshotとして保存し、コード・factory・任意callableを復元しない。

## formal testと形別判定

規模はChampion×4、座席rotationなしの**200半荘**、1 seed=1半荘とする。
#257と同じ#256 v1 / Arena #453のsource経路で全seedをtest分割へ記録する。
具体的seedは未予約。lisbunの環境でlive ledgerの**全履歴**と照合・予約し、allocation、
ledger revision、producer revision、正確なseed集合を#260へ記録してから生成する。
既知の931000..931999、932000..932099、933000..933399だけを禁止一覧にするのでなく、
#259で新しく使用・予約されたformal seedを含む全履歴を照合する。
#259のformal testは#260の未使用testにも開発データにも流用しない。

testはselectionを固定したまま1回。sourceをまたぐseed重複・欠落、split重なり、
producer不一致、coverage・digest不一致は評価前に拒否する。
ラベルは学習・評価moduleだけが`read_labelled_source()`から既存ground truthで生成する。
同じsequenceの行動適用前snapshot契約と、非聴牌は形別全0の意味を変更しない。

主指標はpopulation・channelごとのlog loss。全34slotと**静的な占有可能slotだけ**の両方を
報告し、後者を判定に使う。情報次第で変わる枚数support集合を評価の分母にはしない。
すべての比較は同じ対象行で行い、確率のloss用clipは`[1e-6, 1-1e-6]`で共通とする。
最終`HandBelief`のrawを`SCALE`で割った値を評価し、float予測だけの評価と混同しない。

1行は観測者の1判断×他家1席。episodeは(半荘, 一意な局instance, 他家席)。
slot内平均→population内の当該episodeの行平均→episode間平均の順で集約する。
半荘単位paired bootstrapを2,000回、RNG seed 260で行い、同じ半荘内の全観測者・全他家・
反復判断をまとめて再標本化する。各標本でepisode-macroの分子・分母を再集計する。
差は`Δ = 推定器 − 主baseline`、区間は95%百分位区間。

判定は**population×channelごと**に次の順で行う。

1. 対象行がない、正例が0、正例を含む独立半荘が50未満、または正例episodeが100未満:
   「判定対象外」。理由と件数を明記する。対象行があればloss自体は報告する。
2. 上の条件を満たし、Δ区間の上端が0未満: そのpopulation・channelで「改善を確認」。
3. それ以外: 「改善を確認できなかった」。悪化方向の差も省略しない。

これはchannel別の比較であり、7channel全体・両population全体の改善、十分な検出力、
同時信頼区間を主張しない。全体採否や多重性を考慮した主張が必要なら、結果閲覧前に
別の規則を登録する。#257で副露tanki 97episode・penchan 68episode、国士は全体2episode
だったことは設計材料であり、今回のtestの判定を先取りしない。

副指標は各channelのBrier、静的slot内の正例率・校正表、正例slot数・行数・episode数・
独立半荘数、全対象の行数・episode数・半荘数、提供/未提供coverageとする。
上限処理前の`q > w`率、平均超過量、処理後の違反件数をchannel別に報告し、
static slot数で希釈しないようmask適用後に推定するslotの件数も分母として併記する。

## 実装・引き継ぎの完了条件

実装は推論用`lisjong.learning.wait_shape_estimator`と学習・評価用
`lisjong.learning.wait_shape_evaluation`へ分ける。
既存source reader・canonical builderを再利用し、Arenaへのruntime依存を作らない。
推論・学習ともML frameworkに依存しない（当てはめは既存の`fit_logistic`を使う）。

推論は`estimate_riichi_wait_shape_belief()` / `estimate_open_wait_shape_belief()`が
`PolicyInput`・対象席・固定したLevel 1モデル・形別モデルを受け取り、Level 2の`HandBelief`を返す。
提供範囲外の席は`ValueError`で拒否する。評価の行は、単独リーチ者では#257と同じS1対象の判断
（打牌を選んだ判断で合法打牌の牌種が2以上）に限る。

```text
python -m lisjong.learning.wait_shape_evaluation select \
    --train 933000..933159 --valid 933160..933239 --dev-eval 933240..933399 \
    --riichi-wait-selection FILE --riichi-wait-sha256 HEX \
    --open-wait-selection FILE --open-wait-sha256 HEX \
    --code-revision COMMIT --output SELECTION.json SOURCE [SOURCE ...]
python -m lisjong.learning.wait_shape_evaluation test \
    --test A..B --selection SELECTION.json --selection-sha256 HEX \
    （Level 1の4引数と --code-revision は select と同じ） \
    [--test-producer FILE --test-producer-sha256 HEX] \
    --output RESULT.json SOURCE [SOURCE ...]
```

selectionは両Level 1 selectionのSHA-256を記録し、testは同じSHA-256のfileでなければ拒否する。
判定は結果の`verdicts`に、population×channelごとに`improved` / `not_confirmed` /
`excluded`（判定対象外。理由は`reason`）で記録する。

focused testで次を固定する。

- 推論側へ正解系module・source readerをimportできない状態でも、`PolicyInput`だけで推論できる。
  player-safeなreaderが`hand_facts.jsonl`を開かない既存境界も維持する。
- Level 2のconstructorを通り、canonical zero、7channelのall-or-none、rawのchannel≤waitを満たす。
  丸め境界、wait=0/1、複数形の同時正例、low/highの非同値を含む。
- 現物・役なし・待ち牌の未見枚数0を理由に、構造上可能な待ちを一律0にしない。
  国士をtankiや六形supportへ混ぜず、副露時の国士0を守る。
- 対象外は未提供、identity/schema/seed/coverage違反はfail closed。
- baselineのtrain限定fit、population内episode重み、静的slotの分母、独立半荘のsupport判定、
  同一行集合でのpaired比較、testとselect seed重複拒否を固定する。

実装PRのmerge後、selectのcommit・selection SHA-256を#260へ記録し、未使用seedの
formal testを1回実行する。両population×7channelの結果（判定対象外を含む）、
source/結果digest、実行条件・終了コード・未実施確認をIssueへ記録して完了判断する。
設計PRは`Refs #260`とし、Issueをcloseしない。

対象外: ロン合法の形別テーブル、#262の実装、門前非リーチ用の別population設計、
Policyへの統合、#236/#249の再開、対局での強さの評価。
