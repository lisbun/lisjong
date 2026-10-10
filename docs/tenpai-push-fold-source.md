# 聴牌PUSH/FOLDの対比較source v1（#288）

Issue: [#288](https://github.com/lisbun/lisjong/issues/288)（親: [#254](https://github.com/lisbun/lisjong/issues/254)、設計: [tenpai-push-fold-design.md](tenpai-push-fold-design.md) 5節）

同じ局面から「押す（ChampionのC0）」と「降りる（`a_fold`）」を1回ずつ進め、その局の結果を対で記録する
sourceのwire契約である。対局の実行と記録はlisjong-arena（producer）、ゲート判定・候補選択・読み込み・
表のfitはlisjongが持つ。lisjongはArenaをimportしない。**このsourceはまだ生成していない。**
seed・規模・bucketの境界はproducerのpilot後に事前登録する。

| module | 役割 | 経路 |
|---|---|---|
| `lisjong.learning.tenpai_push_fold` | ゲート判定・`a_push`・`a_fold`、対比較用Policy | 推論側（`PolicyInput`と合法手だけ） |
| `lisjong.learning.tenpai_push_fold_value` | 表の形と比較式`V` | 推論側 |
| `lisjong.learning.tenpai_push_fold_source` | このwire契約のprojectionとstrict reader | 学習専用 |
| `lisjong.learning.tenpai_push_fold_evaluation` | 件数の確認、表のfit、validの報告 | 学習専用 |

## producerの手順

待ちモデルは#245の`selection.json`（SHA-256 `14475264…ccc38`、`SELECTED_WAIT_MODEL_SHA256`）を
`load_selected_wait_model()`で読む。役・点数の計算にnative拡張（`_lisjong_native`）が要る。

1. **対照**: Champion（`PlacementAwareSpeedCallPolicy`）×4で半荘を実行する。各判断について、Championの
   `PolicyDecision`を`evaluate_tenpai_gate(decision, c0_decision, model)`へ渡す。`None`でなければ
   ゲート判断であり、半荘内の出現順に`ordinal`（0始まり）を付けて`decisions.jsonl`へ1行書く。
   その局の終わりまでの判断者の結果を、`side = "push"`として`outcomes.jsonl`へ1行書く
2. **降りる側**: `TenpaiGate.has_fold_candidate`が真のゲート判断ごとに、同じseed・同じ席配置で半荘を
   最初から実行する。判断者の席だけを
   `TenpaiPushFoldPairPolicy.from_selection(path, fold_target=<その判断のDecisionContext>)`にし、
   他の3席はChampionにする。対局は決定的なので、その判断までは対照と同じ入力が現れ、その判断だけで
   `a_fold`を切る（(A)ではリーチせずに切る）。その局の終わりまで実行すれば足りる。結果を
   `side = "fold"`として1行書く
3. 降りる側の実行で、指定した判断の入力が現れなかった場合（その局が終わるまでにPolicyが`a_fold`を
   選ばなかった場合）は、決定性が崩れている。記録せずにエラーにする

`TenpaiPushFoldPairPolicy`は、`fold_target`と一致する判断以外ではChampionの`PolicyDecision`を
そのまま返す。判断の特定は`DecisionContext`（`PolicyInput`と合法手）の一致だけで行い、instanceは
判断をまたぐ状態を持たない。`fold_target`が降りる候補のあるゲート判断でなければ
`TenpaiPushFoldError`、待ちモデルのSHA-256が違えば`RiichiWaitModelError`、native拡張が使えなければ
`ScoringBackendUnavailableError`を、前の1つは判断時、後の2つは構築時に送出する。

## fileと正準形式

```text
<source directory>/
    manifest.json     lisjong-tenpai-push-fold-source-manifest-v1
    decisions.jsonl   lisjong-tenpai-push-fold-decision-record-v1   player-safe
    outcomes.jsonl    lisjong-tenpai-push-fold-outcome-record-v1    学習専用
```

JSONの正準形式は既存のLearning sourceと同じ（manifestはsorted / indent=2 / LF、JSONLはsorted /
compact / LF、UTF-8）。producerは`manifest_text()`・`decision_to_value()`・`outcome_to_value()`と
同じshapeで書く。牌・行動・`PolicyInput`の値は`lisjong.learning._typed_values`のprojectionである。
他家の手牌・山などの隠し情報は、どのfileにも入れない。

### manifest.json

| field | 値 |
|---|---|
| `schema` | `lisjong-tenpai-push-fold-source-manifest-v1` |
| `gate` | `tenpai-push-fold-gate.conditions-1-7.v1`（`lisjong.learning.tenpai_push_fold.GATE`） |
| `producer` | `arena_revision`, `lisjong_revision`, `lisjong_engine_revision`, `policy`, `wait_model_sha256`（すべて空でない文字列） |
| `splits` | `train`, `valid`のseed配列（重複・共有なし）。分割は半荘（seed）単位 |
| `files` | `decisions`, `outcomes`各々の`bytes`, `rows`, `sha256` |

### decisions.jsonl（対照のゲート判断1回につき1行）

| field | 値 |
|---|---|
| `schema` | `lisjong-tenpai-push-fold-decision-record-v1` |
| `key` | `seed`, `sequence`（半荘内の全席を通した判断の通し番号）, `seat`（判断者） |
| `ordinal` | 半荘内のゲート判断の通し番号（0始まり、`sequence`順） |
| `policy_input`, `legal_actions` | 判断者から見える入力と合法手。降りる側の`fold_target`はこの2つから作る |
| `kind` | `riichi`（A）, `closed_discard`（B）, `open_discard`（C） |
| `riichi_seat` | リーチ者の席 |
| `c0_action` | Championの行動。(A)は`RiichiAction`、(B)(C)は打牌 |
| `push_action` | 押す側の打牌`a_push`。(A)は宣言牌の予測、(B)(C)は`c0_action`と同じ |
| `fold_action` | 降りる候補`a_fold`。より安全な牌がない判断でも値を持つ |
| `push_ron_legal_raw`, `fold_ron_legal_raw` | 各打牌のロン合法確率 `p` のraw（0〜8192。`HandBelief.ron_legal_probability_raw`） |

`kind`から`fold_action`までは`TenpaiGate`の同名fieldをそのまま書く。降りる候補があるのは
`fold_ron_legal_raw < push_ron_legal_raw`の判断である。待ちの枚数・和了点・役の有無・
切った牌がドラ／赤ドラかどうかは、`policy_input`と打牌から決まるので記録しない
（lisjongが`gate_from_record()`と報告の中で計算する）。

### outcomes.jsonl（局の結果。押す側は全判断、降りる側は降りる候補がある判断だけ）

| field | 値 |
|---|---|
| `schema` | `lisjong-tenpai-push-fold-outcome-record-v1` |
| `key` | 対応する判断の`key` |
| `side` | `push`（対照）または`fold` |
| `round_delta` | 判断者のその局の収支（局の開始時と終了時の持ち点の差。供託・本場・リーチ棒を含む） |
| `discard_passed` | 今の打牌（押す側は対照が実際に切った牌、降りる側は`a_fold`）が誰にもロンされなかったか |
| `win` | 判断者が和了した場合 `{"method": "ron"｜"tsumo", "points": 和了点}`、それ以外は`null` |
| `deal_in` | 判断者が放銃した場合 `{"to": 和了者の席, "points": 支払点}`、それ以外は`null` |
| `exhaustive_draw_tenpai` | 荒牌流局で終わった場合の判断者の聴牌（bool）、それ以外の終わり方は`null` |
| `declaration_discard` | (A)の押す側だけ、対照が実際に切った宣言牌の打牌。それ以外は`null` |

- `win.points`・`deal_in.points`は本場・供託・リーチ棒を含まない（`evaluate_win()`の受取点と同じ単位）。
  どちらも正の整数
- `win`・`deal_in`・`exhaustive_draw_tenpai`は、高々1つだけが`null`でない。他家のツモ・他家間のロン・
  途中流局で終わった局は3つとも`null`
- `discard_passed`が偽なら、`deal_in`はその打牌での放銃である。真で`deal_in`があれば、後の打牌での放銃である
- (A)の押す側で宣言牌がロンされた場合も、`declaration_discard`にその打牌を書き、`discard_passed`を偽にする

## readerが拒否するもの（fail closed）

`read_source()`は次をすべて`TenpaiPushFoldSourceError`にする。`read_decisions()`は
`outcomes.jsonl`を開かず、player-safeな記録だけを検査する。

- manifest: schema・`gate`の版・`wait_model_sha256`がこのcodeと違う、正準形式でない、fieldの過不足、
  splitのseedの重複
- file: manifestの`bytes`・`sha256`・`rows`と一致しない、正準形式でない行、未知fieldや欠落のある行
- 判断: キーの重複、splitにないseed、`ordinal`が半荘ごとに`sequence`順の0..n-1でない、
  `key.seat`が判断者でない、`c0_action`・`push_action`・`fold_action`が合法手にない、
  リーチ者が単独リーチの提供範囲にない、`kind`が判断と合わない（`RiichiAction`なら`riichi`、
  打牌なら門前／副露で`closed_discard`／`open_discard`、(B)(C)では`push_action == c0_action`）
- 結果: 判断のないキー、同じ判断・同じ`side`の重複、押す側の欠落、降りる候補の有無と降りる側の
  有無の不一致、局の終わり方の矛盾、通らなかったのに放銃がない、(A)の押す側以外の`declaration_discard`

記録された候補がこのcodeのゲート判定と一致するかは、readerでは再計算しない（待ちモデルが要るため）。
`gate`の版と`wait_model_sha256`の照合で代える。

## 表のfitと報告

`python -m lisjong.learning.tenpai_push_fold_evaluation report --source <dir>
--wall-upper-bounds A,B --count-upper-bounds C,D --minimum-support N` が、trainで表を作り、
validの報告と合わせてJSONで出力する。bucketの境界とsupportの下限は引数で与える
（登録した値は[下の節](#表のfitvalidの報告判定2922026-10-10)）。`--tables-output <path>`を付けると、
表をidentityつきのcanonical JSON（schema `lisjong-tenpai-push-fold-tables-v1`。`minimum_support`、
sourceの`files`のdigest、`tables`を持つ）として書く。

`support --source <dir> [--wall-upper-bounds A,B --count-upper-bounds C,D]` は、trainの
`decisions.jsonl`だけを読み、残りツモ山・待ちの枚数の値ごとの件数を区分別に出す。境界を与えると、
その境界でのbucket別の件数（表のsupportの上限）も出す。`outcomes.jsonl`は開かない。

bucketのキー: 残りツモ山は`w0`〜`w2`、役がある待ちの残り枚数は0枚の`c0`と1枚以上の`c1`〜`c3`。
`q_ron`はロンの枚数、`q_tsumo`と`R_T`はツモの枚数の区分で引く（`R_T`だけ`c0`を持つ）。
`R_F`は残りツモ山だけで引く。表は(A)（`riichi`）と(B)(C)（`discard`）で別に作る。

| 表 | trainでの作り方 |
|---|---|
| `q_ron` / `q_tsumo` | 押す側で今の打牌が通った局のうち、判断者がその方法で和了した割合 |
| `R_T` | 押す側で今の打牌が通った局の収支の平均。判断者が和了した局は0 |
| `R_F` | 降りる側で今の打牌が通った局の収支の平均。(B)(C)は`a_fold`が聴牌を崩す判断だけ、(A)は全部 |
| `L` | 今の打牌でリーチ者に放銃した局の支払点の平均（押す側・降りる側の両方）。リーチ者が親か子か |
| `U` | (A)の押す側で和了した局の、実際の和了点 − 判断時点の除外モードの和了点（残り枚数の加重平均）。ロン・ツモ別 |

件数が`--minimum-support`未満のbucketは表に入れず、`V`はそのbucketで計算できない（ゲート条件8）。

validの報告（局収支差は 降りる − 押す。区間は半荘単位のbootstrap、2,000回、固定seed）:

- `round_delta_difference`: (A)と(B)(C)の別に、全体（`all`）、bucket別、役の有無別、
  `V`が降りる／押すと判定した判断別（`v_fold`が設計5節の中止条件の値）、C0がドラ・赤ドラの判断だけ
- `v_decisions`: `V`の判定の件数（降りる候補なし、表なし、降りる、押す）
- `calibration`: 押す側・降りる側の打牌の`p`の区分ごとの、今の打牌でのリーチ者への放銃率（件数つき、区間なし）
- `deal_in_points`: 今の打牌でのリーチ者への放銃点の平均を、切った牌がドラ・赤ドラかどうかで分けた値
- `riichi_declaration_prediction`: (A)で、対照が実際に切った宣言牌と`a_push`の予測が一致した数・しなかった数

## 件数の確認（設計5節の手順1、2026-10-09）

消費済みの開発source（#237 S1、seeds 931200..931399、200半荘、`decisions.jsonl`のSHA-256
`d0805dbe…c0be5`）で数えた。新しい対局・seedは使っていない。C0は記録済みの選択
（lisjong `f6e0f0c`時点のChampion）、ゲートはこのIssueのcode、役の判定はnative拡張（WSL）。

```text
python -m lisjong.learning.tenpai_push_fold_evaluation count --s1-source <S1 source> --selection <selection.json>
```

| 区分 | 役 | ゲート判断 | うち降りる候補あり |
|---|---|---|---|
| (B) 門前・リーチ不可 | あり | 30 | 26 |
| (B) | なし | 0 | 0 |
| (C) 副露聴牌 | あり | 1,078 | 727 |
| (C) | なし | 26 | 17 |
| 合計 | | 1,134 | 770 |

- 全判断は22,579。200半荘あたり、(B)(C)のゲート判断は約5.7回/半荘、降りる候補のある判断は約3.9回/半荘
- (A)は数えられない。S1は打牌を選んだ判断だけを持ち、リーチ宣言の判断を含まない。対照の対局で数える
- 「役あり」は、残りのある待ち牌種のどれかがロンかツモで役を持つこと
- この件数は規模の見積りにだけ使う。押し引きの良し悪しは示さない

## 表のfit・validの報告・判定（#292、2026-10-10）

入力はlisjong-arena#475の対比較source（400半荘。train seed 939000..939199、valid 939200..939399。
`manifest.json`のSHA-256 `1e42b79b…99043`、`decisions.jsonl` `003c0141…500ff`、`outcomes.jsonl`
`1470cb9d…0a6d6`）。code はlisjong `333ff4c`、役・点数はlisjong `6be9b90`のCI wheel
（SHA-256 `0c0e3dc4…6833e`、WSL）。

### 事前登録

`report`の実行前に、[#292へ記録した](https://github.com/lisbun/lisjong/issues/292#issuecomment-6098406165)。
見たのはtrainの`decisions.jsonl`の件数と`p`だけである。

| 項目 | 値 | 区分 |
|---|---|---|
| 残りツモ山 | `19,31` | `w0`: 19以下、`w1`: 20〜31、`w2`: 32以上（trainのゲート判断1,733件の三分位） |
| 待ちの残り枚数 | `3,4` | `c0`: 0枚、`c1`: 1〜3枚、`c2`: 4枚、`c3`: 5枚以上 |
| supportの下限 | `20` | |

- ダブロンの放銃（`outcomes.jsonl`の16行）はそのまま使った。リーチ者が遠い側の和了者だった行は`L`・校正表で
  リーチ者への放銃に数えられず、近い側だった行は`points`に他家への支払いを含む。どの行かはsourceから分からない
- (B)だけを分けた報告は出さない

### 表

```text
python -m lisjong.learning.tenpai_push_fold_evaluation report --source <dir>   --wall-upper-bounds 19,31 --count-upper-bounds 3,4 --minimum-support 20 --tables-output tables.json
```

| file | SHA-256 |
|---|---|
| `tables.json`（identity `fa2a328b36792c60f42640664e133c44ef337a4408b80e1be1b2a42ceb3d5960`） | `9234480ce3bed2f0b6e976cb853de9597f515f0d39b3fcbc35b534dc6efe730c` |
| `report.json`（`report`の標準出力） | `9f9cd775fbb28eb640c7533a3635669f67b4c297c4f95c0f932daa0e9befabdb` |

置き場所はローカルの`C:\Dev\lisjong-artifacts\issue-292-tenpai-push-fold-fit\`（Gitには置かない）。
#289は`tables.json`をこのSHA-256で照合して読む。

`L`（点、括弧はtrainの件数）: リーチ者が親 8,956（32）、子 5,189（81）。
`U`: ロン 1,544（140）、ツモ 2,589（78）。`R_F`: (A) `w0` −970（138）/ `w1` −721（178）/ `w2` −334（191）、
(B)(C) `w0` −1,406（269）/ `w1` −1,456（228）/ `w2` −692（249）。

(A)の`q_ron` / `q_tsumo` / `R_T`（`R_T`の件数）:

| | `c1` | `c2` | `c3` |
|---|---|---|---|
| `w0` | 0.100 / 0.032 / −900（63） | 0.135 / 0.026 / −641（39） | 0.222 / 0.128 / −1,008（39） |
| `w1` | 0.323 / 0.079 / −1,671（63） | 0.250 / 0.179 / −1,854（56） | 0.364 / 0.190 / −1,736（84） |
| `w2` | 0.220 / 0.153 / −1,683（59） | 0.304 / 0.217 / −1,520（69） | 0.321 / 0.190 / −1,578（79） |

(B)(C)の`q_ron` / `q_tsumo` / `R_T`（`R_T`の件数）:

| | `c1` | `c2` | `c3` |
|---|---|---|---|
| `w0` | 0.158 / 0.088 / 177（91） | 0.056 / 0.074 / −593（162） | 0.257 / 0.110 / 23（155） |
| `w1` | 0.275 / 0.122 / −1,706（49） | 0.205 / 0.182 / −1,025（121） | 0.397 / 0.147 / −729（150） |
| `w2` | 0.289 / 0.128 / −2,432（47） | 0.295 / 0.123 / −1,507（130） | 0.341 / 0.268 / −643（138） |

supportが20未満で表に入らなかったのは、(B)(C)の`R_T`の`c0`（ツモで役がある枚数が0）の3 bucket（各5件）だけである。

### validの報告

局収支差は 降りる − 押す（点）。区間は半荘単位のbootstrapの95%区間。

| 区分 | 対象 | 件数（半荘） | 平均 | 区間 |
|---|---|---|---|---|
| (A) | 降りる候補のある判断すべて | 548（197） | −1,679 | −2,245 〜 −1,123 |
| (A) | **`V`が降りると判定（`v_fold`）** | 14（14） | **+1,650** | +136 〜 +3,307 |
| (A) | `V`が押すと判定 | 534（197） | −1,766 | −2,364 〜 −1,199 |
| (A) | C0がドラ・赤ドラ | 18（18） | −1,639 | −5,294 〜 +1,994 |
| (B)(C) | 降りる候補のある判断すべて | 925（170） | −1,103 | −1,528 〜 −670 |
| (B)(C) | **`V`が降りると判定（`v_fold`）** | 122（65） | **+399** | −407 〜 +1,297 |
| (B)(C) | `V`が押すと判定 | 781（163） | −1,350 | −1,811 〜 −855 |
| (B)(C) | 役あり / 役なし | 912（167） / 13（8） | −1,131 / +846 | −1,566 〜 −673 / −914 〜 +3,350 |
| (B)(C) | C0がドラ・赤ドラ | 49（43） | −237 | −1,483 〜 +1,038 |

`V`の判定の件数: (A)は降りる候補なし90、降りる14、押す534。(B)(C)は降りる候補なし374、表なし22、降りる122、押す781。
(A)の降りる候補のある判断はすべて役ありである。

bucket別の平均（件数）:

| | `c0` | `c1` | `c2` | `c3` |
|---|---|---|---|---|
| (A) `w0` | - | −1,963（54） | +97（32） | −1,265（31） |
| (A) `w1` | - | +95（85） | −1,413（45） | −3,514（65） |
| (A) `w2` | - | −1,523（62） | −1,714（77） | −2,760（97） |
| (B)(C) `w0` | +2,000（8） | −1,744（79） | −883（89） | −2,413（113） |
| (B)(C) `w1` | −1,250（2） | −1,108（59） | −1,424（116） | −1,531（142） |
| (B)(C) `w2` | −833（3） | +1,142（52） | −555（108） | −609（154） |

`p`の校正（押す側・降りる側の打牌。今の打牌でのリーチ者への放銃率）:

| `p`の区分 | 件数 | `p`の平均 | 放銃率 |
|---|---|---|---|
| 0 | 1,373 | 0.0000 | 0.0000 |
| 0超〜0.01 | 256 | 0.0043 | 0.0039 |
| 〜0.02 | 143 | 0.0148 | 0.0280 |
| 〜0.05 | 491 | 0.0347 | 0.0387 |
| 〜0.1 | 935 | 0.0728 | 0.0856 |
| 〜0.2 | 212 | 0.1154 | 0.1509 |

- `p`が0.01を超える区分では、放銃率が`p`の平均より高い（0.1超の区分で0.115に対し0.151）。`p`はロン合法確率であり、
  校正済みの放銃確率ではない（設計の仮定A7）
- 放銃点の平均: 切った牌がドラ・赤ドラ 9,200（8件）、それ以外 6,040（128件）。件数が少ないが、`L`を牌によらない
  平均とする仮定A3は、ドラを切る放銃を過小に見る向きである
- `riichi_declaration_prediction`: 一致638、不一致0

### 判定（設計5節の手順4）

| 区分 | `v_fold`の平均（件数） | 判定 |
|---|---|---|
| (A) リーチ宣言 | +1,650（14） | 0より大きい。候補の対象に残す |
| (B)(C) リーチ不可・副露 | +399（122） | 0より大きい。候補の対象に残す |

両方の区分が中止の条件を通過したので、#289（候補Policyの実装）に着手する。

- (B)(C)の区間は0をまたぐ（−407 〜 +1,297）。(A)は14件・14半荘だけである。どちらも、降りた方が良いことを
  示したとは言えない。中止の条件は平均の符号だけで決めている
- `V`が降りると判定した判断は、降りる候補のある判断のうち(A)で2.6%、(B)(C)で13%である。
  それ以外の判断では、降りる側の局収支が押す側より低い
- これは同じ局面からの局収支の比較であり、半荘の強さの評価ではない。相手はChampion×3である
- 判定の後に境界・support・式を変えない。変える場合は新しいsourceで事前登録からやり直す
