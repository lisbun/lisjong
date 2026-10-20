# L2a — first-party C0による小規模蒸留の実験仕様（設計）

Issue: [#267](https://github.com/lisbun/lisjong/issues/267)（親: lisbun/lisjong-project#74、#84）

固定教師C0（`PlacementAwareSpeedCallPolicy`）の行動を小規模モデルへ蒸留し、入力・合法手・教師label・
推論経路が正しく、行動種別ごとに学習が成立するかを確認する実験の仕様である。本書は**設計だけ**であり、
source生成・学習・対局は後続Issue（[9節](#9-担当と後続issue)）で行う。OFFLINE-EVALの結果は
[10節](#10-結果)に追記した。1〜9節の閾値・事前登録は結果を見た後も変更していない。

本書は次を主張しない。

- 蒸留の成立は教師超え・棋力改善・Champion昇格ではない。教師を超える仮説はRL・探索・より強い教師/データで別に検証する
- 下の閾値・規模は結果を見る前の事前登録値であり、達成見込みの主張ではない
- #259の直接待ち推定器、#254のHeuristic設計とは別の仮説である。外部sourceのHOLD解消（project #86）を前提にしない

用語: 「半荘support」は、ある行動の行を1件以上含む半荘の数。行数とは区別する。

## 1. 教師とsourceのidentity

### 1.1 教師

| 項目 | 固定する値 |
|---|---|
| Policy | `lisjong.policies.placement_aware_speed_call.PlacementAwareSpeedCallPolicy`（Heuristic Champion C0） |
| Arena catalog | `placement-aware-speed-call`。`source_contract.teacher`の`configuration_digest`をそのまま保持する |
| lisjong revision | 生成時にArenaがpinしているlisjong commit（起票時点のArena pinは`174bf48`）。manifestの`runtime`が記録するVCS revision・`lisjong_source_digest`を正とする |
| native backend | `LISJONG_SHANTEN_BACKEND=rust`。manifestにnative API versionとextension fileのSHA-256が残る。Python backendは使わない（C0で約6倍遅い、arena #442予備計測） |
| 席 | 4席ともC0の自己対局（`seat_policies`） |

- 教師の行動が同じであることは、生成したrecordの**全件replay-verify（不一致0）**で確かめる。#442の生成revision
  `e6346ed`との行動同一性は仮定しない。C0の継承chainは`e6346ed`以降に`genbutsu_defense_finite_horizon_hand_value_aware.py`
  と`mechanism_riichi_defense_yakuhai_call.py`の整理（#246・#313）を受けており、同一性はreplayで確かめる
- 1つのL2a実験のTRAIN / SELECT / OFFLINE-EVALは、同じ教師identity・同じruntimeで生成した1つのsource recordから取る。
  revisionの異なるsourceを混ぜない

### 1.2 C0の行動（Policyのコードから導いたもの）

`TwoStepUkeirePolicy._decide`を基底とするC0の判断順は次のとおり（lisjong `174bf48`で確認）。

| 順 | 条件 | C0の行動 |
|---|---|---|
| 1 | Ron / Tsumoが合法 | 必ず和了（canonical順で最初） |
| 2 | `RiichiAction`が合法 | 必ずリーチ（Always Riichi） |
| 3 | 合法打牌がある | 打牌経路（受入・2段目・FH completion・手牌価値・現物/mechanism守備・么九整理・route保持・オーラスmode） |
| 4 | 打牌なし・Passあり | Pass。ただし役牌Pon系（#145）・役確定型スピード鳴き（#199）の条件を満たせばChi / Ponへ置換 |
| 5 | 合法手が1つだけ | その行動 |

したがって、Daiminkan / Ankan / Kakan / 九種九牌は、他の合法手がある限りC0は選ばない。和了とリーチは
合法なら必ず選ぶ。これは学習の対象範囲（[2節](#2-対象行動と教師label)）を決める前提であり、生成した
sourceで例外0件であることを確かめてから使う（[2.3](#23-教師規則の確認)）。

### 1.3 source

| 項目 | 内容 |
|---|---|
| producer | Arena `lisjong_arena.policy_source_record`（[arena docs/policy-source-record.md](https://github.com/lisbun/lisjong-arena/blob/main/docs/policy-source-record.md)） |
| schema | `arena-policy-source-record-v2`（PolicyInput v2、宣言牌の位置を持つ）。lisjongの`read_source_record()` / `open_source_record()`で読める |
| purpose | 本番は`SCIENTIFIC`（splitごとのallocation binding必須）。support計数のpilotは`DEVELOPMENT` |
| 実行環境 | RiichiEnv `LocalGameRunner`、`4p-red-half`。engine runnerはproducer未対応 |
| 行 | 全席・全判断。`teacher_selected_action`は同じtraced executionの`DecisionTrace.selected_action` |
| 保存・回収 | replay-verify済みrecordを`archive`（tar.gz + `evidence.json`）で回収し、`restore`がidentity・SHA-256・replay証跡を照合する（arena #449）。datasetは派生物で既定では保存しない |

### 1.4 既存基盤で足りる範囲と不足

| 範囲 | 既存基盤 | L2aでの扱い |
|---|---|---|
| C0 source生成・replay・archive | arena #442 / #445 / #449 | そのまま使う |
| source読込み（v2・教師identity・purpose） | lisjong #244 / #307 | そのまま使う |
| 変換のRSS・並列化 | lisjong #247 / #248 | そのまま使う |
| 打牌の候補scorer（dataset・model・artifact・runtime） | L0.2（#189） | baseline M0として無変更で使う。提案モデルは候補encodingを追加する（[4節](#4-モデル学習目的選択baseline)） |
| 鳴き・Passの学習 | なし（BCの802-way flat分類はあるが、#331で行動の階層を丸ごと学習させて未達） | **不足**。反応判断の候補scorerを新設する |
| C0の行動に合わせたguard | L0.2 O0はC0と同じ和了・リーチ規則だが、反応ではPassを固定で選ぶ | **不足**。L2a用のguardと合成runtimeを新設する |
| 行動別のsupport計数・教師規則の確認 | なし | **不足**。lisjongのCLIとして追加する |
| 半荘clusterでの区間推定・層別報告 | なし | **不足**。評価toolへ追加する |
| 適用された行動（他家の優先で取り消された鳴き） | rowに無い | L2aでは不要（2.4）。結果を報酬に使う後続段階で必要 |
| 学習側の逐次読込み | 無い（全件をメモリへ読む） | 1k半荘程度までは不要（学習RSS約5〜7 GiB、#442）。それを超える規模では必要 |

## 2. 対象行動と教師label

### 2.1 判断の分類と採用範囲

各行を`legal_actions`だけで次の1つへ分類する。分類は結果を見る前の規則であり、servingとdatasetで同じ関数を使う。

| 分類 | 条件（上から順に判定） | L2aでの扱い |
|---|---|---|
| FORCED | 合法手が1つ | 学習・評価の分母から外し、件数だけ報告 |
| WIN | Ron / Tsumoが合法 | **教師規則guard**（必ず和了）。学習しない |
| RIICHI | `RiichiAction`が合法 | **教師規則guard**（必ずリーチ）。学習しない |
| DISCARD | 合法打牌がある | **学習対象1**：打牌scorer |
| CALL | Chi / Ponが合法で、Passが合法 | **学習対象2**：反応scorer（Pass・各Chi・各Ponから1つ） |
| PASS_RESPONSE | Passが合法で、打牌・Chi・Ponが無い | 教師規則guard（Pass）。学習しない |

- 上記のどれにも入らない行が出たら、推測で振り分けずfail closedする
- Ankan / Kakan / 九種九牌は、合法でも打牌などの他の合法手と同時にしか出ないため、該当行はDISCARDに入り、
  guardはそれらを候補から外す。CALL行に同時に出るDaiminkanも反応scorerの候補に入れずguardで外す
- 分類は必ず上の表どおりに`legal_actions`全体で行う。FORCEDは「合法手が1つ」だけであり、合法打牌が1つでも
  暗槓・ツモ等が同時に合法なら表の次の行（WIN / DISCARD等）へ進む
- **単一候補行**：DISCARD / CALL行のうち、guardが槓・九種九牌・Daiminkanを外した後にscorerの候補が1つだけになる行
  （例：リーチ中で、ツモ切り1つと暗槓が合法）。scorerを呼ばずその候補を返す。学習datasetには入れず、評価では
  分母から外して件数とguard一致だけを別に集計する。自明な正解で一致率が上がることを防ぐためである。CALL行は
  Passと1つ以上のChi / Ponを必ず持つため、単一候補行になるのは実際にはDISCARDだけである
- リーチ宣言後の打牌（宣言牌）は、候補が2つ以上ならDISCARDの層（2.5）として別に報告する。リーチ中のツモ切りは、
  合法手が1つならFORCED、暗槓等が同時に合法なら単一候補行になる
- 打牌scorerは全合法打牌を候補にする。L0.2aのsemantic envelope（最小向聴→最大受入→最大2段目）は**使わない**。
  C0は守備・打点・么九整理・route保持で牌効率の順序を破るため、envelopeを掛けるとC0の行動を表現できない。
  牌効率との一致はgateではなく層別の診断（5.3）として測る

### 2.2 成立を主張しない行動

WIN / RIICHI / KAN系 / 九種九牌は学習しない。L2aの結果は「打牌とCALLの蒸留」についてだけ述べ、
全Policyの学習成立とは呼ばない。結果の記録には必ず次の表を付ける。

| 行動 | 結果の表記 |
|---|---|
| 和了 / リーチ | 教師規則guard（C0と同じ決定的規則）。学習成立の対象外 |
| 槓 / 九種九牌 | 教師がsupportを持たない（選ばない）ため対象外。guardは選ばない |
| 打牌 / CALL | 6節の判定 |

リーチ判断・槓・九種九牌を学習対象へ入れるには、それを選ぶ教師または価値目的が必要であり、L2aの範囲外である。

### 2.3 教師規則の確認

guardに置く規則（WIN→和了、RIICHI→リーチ、PASS_RESPONSE→Pass、Ankan / Kakan / 九種九牌 / Daiminkanを選ばない）は、
pilotと本番の両sourceで全行を数え、**例外0件**を確かめる。1件でも例外があれば、学習を始める前にその行動を
guardから外し、本書を改訂してから進める（結果を見た後でguardを変えない）。

### 2.4 教師labelと合法手の対応

- labelは`teacher_selected_action`だけを使う。C0を再実行してrelabelしない
- labelは同じ行の`legal_actions`へ一意に解決できなければならない。解決できない行はfail closedする（L0.2と同じ）
- CALL行のlabelは**教師が選んだ行動**であり、他家のRonが優先されてその鳴きが適用されなかった場合も行は残り、
  labelは教師の選択のままである（runnerは全席の選択を記録してから`env.step()`を呼ぶ）。模倣の目的には
  これが正しいlabelである。適用されたかどうかはrowに無く、L2aでは使わない。結果を報酬にする後続段階では、
  engineの`REACTION_RESOLVED`（capable / selected / awarded / passed）等で別に扱う

### 2.5 層（結果を見る前に固定）

すべての指標を全体と次の層で報告する。層は公開情報だけで決める。

| 層 | 定義 |
|---|---|
| 局の位置 | 東場 / 南1〜3局 / オーラス相当（南4局以降、西入を含む。C0の`_is_all_last`と同じ条件） |
| 順位 | 判断時点の公開scoreでの自席順位（1〜4位、同点を含むかどうか） |
| 他家リーチ | 無し / 1人 / 2人以上 |
| 自分の状態 | 打牌前の向聴（0 / 1 / 2 / 3以上）、門前か副露か、宣言牌か |
| 牌効率との関係（DISCARDのみ） | 教師の打牌牌（牌種と赤の有無）が`TwoStepUkeirePolicy`の選択と同じか。異なる行を「C0固有層」と呼ぶ。TwoStepは同じPolicyInputと合法手で再実行する |

オーラス相当局は**学習に含め、層別に記録する**（調査コメント1の(c)）。L2aは模倣なので、C0の順位依存の行動も
labelどおりに学ぶ。局素点を目的にする後段では、オーラス相当局の判断を学習対象に含めるか・guardへ置くかを
その後段の設計で決める（Suphx `RL-basic`の失敗例。本書では決めない）。

### 2.6 player-safe入力と推論経路

```text
DecisionContext（PolicyInput + legal_actions）
    -> 分類（2.1）
    -> FORCED / WIN / RIICHI / PASS_RESPONSE : 教師規則guard
    -> 単一候補行（2.1）: その候補
    -> DISCARD : 共有context（#184） + 打牌候補encoding -> 打牌scorer -> argmax（同点はcanonical順）
    -> CALL    : 共有context（#184） + 反応候補encoding -> 反応scorer -> argmax（同点はcanonical順）
    -> decision.legal_actions側のcanonical objectを返す
```

- 入力は`PolicyInput`と`legal_actions`だけ。HandBelief、他家の伏せ牌、wall、将来の事象、C0の内部状態・分析値は使わない
- 候補encodingに加える値は、PolicyInputから決定的に計算できるplayer-safeな解析値に限る（4.2）。concrete Policy
  moduleのprivate helperをimportしない（候補Policyの情報境界チェック#296と同じ考え方）
- 非有限score・候補数不一致・vocabulary / encoding不一致はfail closedし、heuristic fallbackしない
- datasetとservingは同じencoding関数を使う。評価では、OFFLINE-EVALの行をservingの経路で再encodeした値と
  datasetの値の一致を確かめる（5.1）

## 3. 分割・support・判定保留

### 3.1 分割

- 半荘（seed）単位でTRAIN / SELECT / OFFLINE-EVALへ分ける。同じ半荘の行は1つのsplitだけに入る
- TRAINは最適化、SELECTはepoch・幅の選択、OFFLINE-EVALは凍結後に1回だけ使う
- support計数のpilot（3.3）は別seedのDEVELOPMENT sourceで行い、学習・選択・評価に使わない
- 具体的なseedの予約は実行計画で行う（本書では予約しない）

### 3.2 support（OFFLINE-EVALで必要な最小値）

| 指標 | 行数 | 半荘support |
|---|---:|---:|
| DISCARD全体 | 20,000 | 100 |
| DISCARDのC0固有層（TwoStepと不一致） | 2,000 | 80 |
| CALL全体 | 2,000 | 80 |
| Chiが合法なCALL行 / うち教師がChiを選んだ行 | 1,000 / 300 | 80 / 60 |
| Ponが合法なCALL行 / うち教師がPonを選んだ行 | 1,000 / 300 | 80 / 60 |
| オーラス相当局のDISCARD（報告のみ） | 1,000 | 30 |
| WIN / RIICHI guard | 各30 | — |

- 半荘clusterの区間推定は、半荘をresampling単位とするbootstrap（10,000回、seed固定）で求める。行を独立標本として
  区間を作らない
- 1半荘に同じ行動が多数あっても、半荘supportは1と数える。下限は行数と半荘supportの両方で判定する
- Chi / Ponは種別ごとに下限を判定し、足りない種別はその種別だけ判定保留にする。合算の件数で片方の不足を補わない
- これらの下限は判定を始めるための出発点であり、指標の精度を保証する根拠ではない。精度は半荘cluster bootstrapの
  区間として報告する
- 単一候補行（2.1）はどの下限の行数にも数えない

### 3.3 support計数のpilotと規模の固定

学習の前に、DEVELOPMENT sourceを50半荘生成し（seedは実行計画で決める）、次を数える。

- 分類別・行動別（打牌・リーチ・Chi・Pon・各槓・九種九牌・和了・Pass）の行数と半荘support
- CALL行のうちChi / Ponそれぞれが合法な行と教師がそれを選んだ行、オーラス相当局の行、DISCARDのC0固有層の行、
  単一候補行
- 教師規則の例外（2.3）

本番の規模は、pilotの1半荘あたりの値から3.2の下限を満たすOFFLINE-EVAL半荘数を求めて決める。初期案は
TRAIN 600 / SELECT 100 / OFFLINE-EVAL 150半荘とし、OFFLINE-EVALは最大250半荘まで増やせる。250半荘でもChi / Ponの
どちらかの下限に届かない見込みなら、その種別の判定を学習前に**判定保留（support不足）**と宣言して進める。黙って除外しない。
pilotでは一致率などの学習結果を見ない。

### 3.4 判定保留とSTOP

| 結果 | 条件 |
|---|---|
| STOP / INVALID | 合法性・guard・train-serve一致が1.000未満、教師規則の例外、source identity・replayの不整合、OFFLINE-EVALを2回以上使った |
| 判定保留（support不足） | 3.2の下限を下回る指標。推測でPASSにしない |
| 6節の判定 | 上記に当たらない指標 |

## 4. モデル・学習目的・選択・baseline

### 4.1 モデル（小規模に限定）

L0.2の候補scorerと同じ形の小さなMLPを、打牌と反応で別々に持つ。

```text
hidden = ReLU(context_layer(shared) + candidate_layer(candidate))
score  = output_layer(hidden)
```

- 幅は64と256の2つだけ。weightsはflat float32（`weights.f32`）で、Rustで直接推論できる規模に留める。
  Mortal（192ch×40 ResNet）・Suphx（256ch×50）級の構造はADR 0010のNN推論形式を先に決める必要があるため、
  L2aでは採らない
- artifactは既存と同じくimmutable directoryで、source / dataset / feature / encoding / guard規則 / model / 学習設定 /
  seed / 選択epoch / weights digestをbindし、不一致はload時にfail closedする

### 4.2 候補encoding

| 対象 | encoding |
|---|---|
| 打牌（M0） | L0.2の`lisjong-offense-l0.2-discard-candidate-encoding-v1`を無変更で使う（牌種・赤・ツモ切り・打牌後向聴・受入・2段目） |
| 打牌（M1、提案） | M0に次のplayer-safe解析値を加える：リーチ者ごとの現物・確定安全牌（#302）、筋・壁などの公開mechanism危険度、打牌後に残るdora / 赤の枚数、么九・役牌・字牌の区別、副露時のTanyao / Honitsu route向聴（打牌後） |
| 反応（R1、提案） | 下の「反応候補の値」 |

反応候補の値（R1）。配列の配置は実装Issueで固定するが、値の意味は次で固定する。

- routeは通常形（制限なし）、Tanyao、Honitsu（萬・筒・索の3つ、Chinitsuを含む）の5つ。route向聴は
  `lisjong.hand_evaluation.shanten.calculate_restricted_standard_shanten`による、route外の牌を浮き牌とする通常形向聴とする。
  route定義はLearningのfeature moduleが独自に持ち、Policy moduleのprivate定数をimportしない
- **Chi / Pon候補**：消費牌を除いた手牌から、喰い替え規則で許される打牌をそれぞれ1枚行った手を列挙し、routeごとに
  **独立した最小値**を取る（routeごとに最小を与える打牌は異なってよい。同じ打牌での評価値には揃えない）
- **Pass候補**：鳴かない現在の手（打牌前の13枚相当）の各route向聴
- 共通の値：行動種別one-hot（Pass / Chi / Pon）、鳴く牌の牌種と赤、消費牌、各routeの値とPass候補の値との差、
  その鳴きで役牌の刻子ができるか、自席に役牌の副露・暗刻があるか。Pass候補では鳴く牌・消費牌の欄を0にする
- **routeが成立しない場合**（既存の副露・鳴く牌・消費牌がそのrouteと矛盾する。例：么九牌を含む副露とTanyao）は、
  そのrouteの成立maskを0、値を0にする。成立しない場合を大きな向聴で代用しない。差の欄も、どちらかが不成立なら
  maskを0、値を0にする

M1・R1の値は、C0が規則で参照する情報の族（守備・打点・役のroute）をplayer-safeな解析値として渡すものである。
値を渡すだけで、制約（envelope）にはしない。C0の内部値（`GameSituationMode`、危険度の順位、選択branch）は渡さない。
オーラスmodeは共有contextの公開score・局・本場から学ぶ。

固定したidentityと次元（#316）は次のとおり。layoutの正本は`lisjong.learning.l2a_encoding`のmodule docstringである。

| encoding | identity | 次元 |
|---|---|---:|
| M0 | `lisjong-offense-l0.2-discard-candidate-encoding-v1`（L0.2と同一） | 53 |
| M1 | `lisjong-l2a-discard-candidate-encoding-m1-v1`（M0 + 解析値37次元） | 90 |
| R1 | `lisjong-l2a-reaction-candidate-encoding-r1-v1` | 95 |

- route向聴はroute外の牌を浮き牌とするため、使える牌が無い手では13まで大きくなる（通常形の上限8を超える）。
  scaleは13とする
- 喰い替え規則は、鳴いた牌と同じ牌種、およびChiで鳴いた牌が順子の端のときの反対側の牌種を、鳴いた直後に切れない牌とする
- datasetは全行で教師規則の例外（2.3）を確かめ、1件でもあればfail closedする

#442の校正（record200、DEVELOPMENT、M0相当のencoding）では、C0教師に対するSELECTの打牌top-1は65.16%だった。
M1の追加は「不足はデータ量より、C0の守備・打点・役の判断に使われる情報がencodingに無いこと」という仮説である。
データ量だけでは解消すると仮定しないため、同じデータでM0とM1を比べる。この比較で分かるのは**追加した特徴量一式**の
効果であり、現物・危険度・ドラ等それぞれの寄与ではない。個別の寄与は必要になったら別Issueで調べる。

### 4.3 学習目的と選択

- 目的：判断ごとに合法候補だけで正規化したsoftmax cross entropy（L0.2と同じ）。打牌・反応は別々に学習する
- 選択：幅{64, 256}× 20 epochをTRAINで学習し、SELECTのcross entropy最小の（幅, epoch）を採る。seedは1つに固定。
  これ以外のhyperparameter探索はしない
- epoch別のTRAIN / SELECT cross entropyをartifactのdiagnosticsに残す（#442で推移が残らなかったため）
- 凍結対象：モデル・encoding・guard・選択規則・閾値。凍結後にOFFLINE-EVALを1回だけ評価する

### 4.4 比較baseline

| 名前 | 内容 | 用途 |
|---|---|---|
| B-TS | `TwoStepUkeirePolicy`の選択（規則、学習なし） | 牌効率だけで説明できる一致の水準 |
| B-PASS | CALLで常にPass | 反応の多数派の水準 |
| M0 | M0 encodingの打牌scorer（同じデータ・同じ選択規則） | encoding追加の効果をデータ量から分ける |
| M1 / R1 | 提案 | 判定対象 |

### 4.5 停止条件

- L2aの中で、OFFLINE-EVALを見た後の再学習・閾値変更・特徴量変更による救済はしない。未達なら記録して終える
- 次の仮説（特徴量・モデル・規模の変更）は新しいIssueと新しいseedで行う
- STOP / INVALID（3.4）が起きたら、原因を直してから同じ事前登録で生成からやり直すかを、新しいseedで判断する

## 5. 評価

### 5.1 正しさ（すべて1.000が必要）

| 指標 | 対象 |
|---|---|
| 合法性 | OFFLINE-EVALの全行 |
| guardの一致（和了・リーチ・Pass・槓等を選ばない） | WIN / RIICHI / PASS_RESPONSE行と、DISCARD / CALL行で槓・九種九牌を選ばないこと |
| train-serve一致 | OFFLINE-EVALの単一候補行（2.1）を除く全DISCARD / CALL行で、servingの経路で作ったencodingとdatasetの値が一致（単一候補行はdatasetを作らない） |
| 情報境界 | 候補Policy・Learning runtimeのimport制限testが通る |

### 5.2 学習の成立（6節で判定に使う）

| 行動 | 主指標 | 定義 |
|---|---|---|
| DISCARD | 打牌一致率 | 学習器と教師の打牌が同じ牌（牌種と赤の有無）。ツモ切り / 手出しの区別は診断として別に報告 |
| DISCARD | C0固有層の一致率 | C0固有層（2.5）での打牌一致率 |
| CALL（Chi / Ponごと） | 種別Tのbalanced accuracy | Tが合法なCALL行で、「教師がTを選んだ行で学習器もTを選んだ割合」と「教師がTを選ばなかった行で学習器もTを選ばなかった割合」の平均 |
| CALL（Chi / Ponごと） | 種別Tの鳴き行動の一致率 | 教師がTを選んだ行のうち、学習器が同じ行動（同じ種別・消費牌）を選んだ割合 |

CALL全体（Chi / Pon合算）の鳴く/鳴かないのbalanced accuracyと鳴き行動の一致率は、診断として報告する。

### 5.3 診断（判定に使わない）

- 層別（2.5）の全指標。特にオーラス相当局と順位
- 打牌後の向聴が教師と同じ割合、同じ向聴内で受入が教師と同じ割合（L0.2の牌効率指標）
- top-1でない場合の教師候補の順位、cross entropy、epoch推移
- 推論時間（判断あたりの中央値・p95・最大、候補構築とforwardを分けて）。shanten backendと特徴量計算の内訳を記録する

### 5.4 対局での動作確認

凍結したモデルの合成Policyを1席、C0を3席にして16半荘を開発seedで実行する（Arena）。
失敗0、全判断が合法、推論時間とRSSを記録する。強さ・局素点の比較は行わず、結果を強さの根拠に使わない。

## 6. 成立条件（OFFLINE-EVAL、結果を見る前に固定）

5.1がすべて1.000で、3.2のsupportを満たした行動について、次で判定する。区間は半荘cluster bootstrapの95%区間。
**「≥ 0.85」は点推定、「区間下限」と書いたものは95%区間の下限**で判定する（区間は全指標で報告する）。
閾値は今回の探索的な成立目標であり、0.85 / 0.40という値そのものに外部の根拠はない。

| 行動 | ESTABLISHED（成立）の条件 |
|---|---|
| DISCARD | (1) M1の打牌一致率 ≥ 0.85（点推定）、(2) M1のC0固有層の一致率の区間下限 ≥ 0.40（B-TSは定義上0）、(3) M1−M0の打牌一致率の差（半荘単位の対）の区間下限 > 0 |
| Chi | (1) R1のChiのbalanced accuracy ≥ 0.85（点推定。B-PASSは0.50）、(2) R1のChiの鳴き行動の一致率 ≥ 0.85（点推定） |
| Pon | (1) R1のPonのbalanced accuracy ≥ 0.85（点推定。B-PASSは0.50）、(2) R1のPonの鳴き行動の一致率 ≥ 0.85（点推定） |

- 各行動について、全条件を満たせばESTABLISHED、(1)を満たさなければNOT ESTABLISHED、(1)は満たすが他の条件を
  満たさなければPARTIAL（記録上の区別。成立とは呼ばない）
- DISCARDの(3)だけを満たさない場合は「C0の打牌は学べたが、encoding追加の効果は示せない」と記録する
- CALLはChiとPonの両方がESTABLISHEDのときだけ「鳴き判断の蒸留が成立」と呼ぶ。片方だけなら「Ponの鳴き判断は成立、
  Chiは判定保留」のように種別ごとに書き、CALL全体の成立とは呼ばない
- 全体の結論は行動別の結果の組で書き、2.2の表を必ず付ける。DISCARD・Chi・PonがすべてESTABLISHEDのときだけ
  「L2a: C0の打牌・鳴き判断の蒸留が成立」と呼ぶ
- 閾値の根拠：L0.2（TwoStep教師）の打牌top-1は0.854、#442（C0教師、200半荘、M0相当）のSELECT top-1は0.652だった。
  0.85は「TwoStep教師を模倣したときの水準をC0教師でも出せる」ことを、0.40は「C0固有の判断の少なくとも
  一部を再現する」ことを求める値である。結果を見た後に変えない

### 6.1 評価tool（#317）

`python -m lisjong.learning l2a-offline-eval`（`lisjong.learning.l2a_offline_eval`）が5節と6節を1回の実行で
判定する。閾値・support下限・bootstrap（10,000回、seed 317、percentile区間）・層は
`lisjong.learning.l2a_offline_gate.L2A_OFFLINE_GATE`（identity `lisjong-l2a-offline-eval-gate-v1`）に固定し、
reportにそのdigestを含める。本書の表から実装で次のとおり具体化した。

- 「CALL全体」の下限（3.2）はChi・Ponの両方の判定に要求する。WIN / RIICHIの各30行を下回るときは
  guardの正しさを確かめられないため、全行動を判定保留にする
- train-serve一致は、M1・M0・R1のdatasetの値とservingの経路で再encodeした値（候補順、教師index、
  shared context、候補vector）のfloat32 byte一致で数える
- 層の向聴は、DISCARDは打牌前、CALLは鳴く前の純手牌の向聴とし、和了形（-1）は0に含める
- `--ledger`のdirectoryは1つのL2a実験を通じて同じ保存先を使う。OFFLINE-EVALを読む前に評価ごとの
  entryを新規作成し、同じsource identityのOFFLINE-EVALを使ったentry、または同じartifact
  （M1 / M0 / R1のどれか）のentryが既にあれば指標を計算せずSTOP / INVALIDにする。artifactを
  すべて再学習しても同じsourceのOFFLINE-EVALは再利用できない（4.5）。途中で失敗した評価の
  entryも残るため、再実行も2回目として扱う。identityが合わない場合はentryを作らずSTOPにする
- 推論時間は、合成Policyのserving呼出しの中で候補encoding・shared context（#184）・forward
  （scorer実行）を分けて測り、判断全体（分類・guardを含む）の時間と並べて報告する

## 7. 時間・RSS・費用・保存の見積り

根拠はarena #442の訂正済み計測（[校正結果](https://github.com/lisbun/lisjong-arena/issues/442#issuecomment-5979691377)、
[訂正](https://github.com/lisbun/lisjong-arena/issues/442#issuecomment-5994798362)）と#247 / #248の結果。m7i.xlarge（4 vCPU =
2物理コア×2 SMT、16 GiB）、rust backend。線形外挿であり、保証値ではない。

### 7.1 1半荘あたり

| 工程 | CPU秒 | 根拠 |
|---|---:|---|
| C0生成 | 92.0 | #442実測 |
| 全件replay-verify | 93.2 | #442実測 |
| 打牌dataset（M0） | 28〜32 | #442の推定処理量 |
| 打牌dataset（M1） | 29〜33（暫定） | #316実測でM1 / M0 = 1.02（下記）。M0の値に比を掛けた暫定値 |
| 反応dataset（R1） | 約1（暫定） | #316実測でR1 / M0 = 0.03（下記）。暫定値 |
| 学習（幅2つ×20 epoch、3モデル） | 約2 | #442の候補学習0.23 s/半荘を3モデル×2幅で概算 |
| 計 | 約245〜255 | |

実測（#316。arena#505のpilot source 50半荘・34,299判断、DEVELOPMENT、rust backend、WSL2・Intel Core Ultra 7 258V、
各encodingを`--workers 1`で1 processずつ並行に実行）。pilot sourceは計測にだけ使い、学習・評価はしていない。

| encoding | CPU秒/半荘 | 変換のmax RSS | dataset容量/半荘 | 読込み+tensor化のRSS増分/半荘 |
|---|---:|---:|---:|---:|
| M0 | 16.8 | 61 MiB | 2.46 MB | 5.8 MiB |
| M1 | 17.1 | 61 MiB | 3.26 MB | 7.9 MiB |
| R1 | 0.52 | 56 MiB | 0.35 MB | 0.9 MiB |

- 258VはPコアとEコアの混成で、WSLからは実行コアを区別できないため、半荘あたりの絶対値は環境依存である。
  m7i.xlargeへの換算には、M0との比（M1 1.02、R1 0.03）を#442のM0推定値へ掛けた暫定値を使う。
  M0とM1は別のprocessで並行に測ったため、実行コアの違いが比に入り得る
- 今回の計測でのM1とM0の差は約0.3 s/半荘である。M1の追加処理（安全牌・待ち形・ドラ・route向聴）だけの時間の内訳は未計測
- 変換のmax RSSは半荘数に依存しない（逐次読込み）。並列変換はworkerあたり同程度
- 「読込み+tensor化」は`read_l2a_dataset()`と学習用tensorの作成までのRSSから、torchのimportだけのRSS
  （221 MiB）を引いて半荘数で割った値である。学習本体は実行していない（pilotの学習結果を見ないため）

### 7.2 L2a全体（初期案 TRAIN 600 / SELECT 100 / OFFLINE-EVAL 150 = 850半荘）

| 項目 | 見積り |
|---|---|
| CPU | 850 × 約250 s（M1 / R1は7.1の暫定値）+ pilot 50半荘（生成・replay 約185 s/半荘）+ 動作確認16半荘 ≈ 約62 vCPU-h |
| 費用（AWS） | 4 vCPUへ詰めた下限で約USD 4.2（USD 0.0673/vCPU-h）。#442ではwallがCPU下限の約1.3倍だったため、概算で約USD 5.5、OFFLINE-EVALを250半荘にすると約USD 6。上限ではない |
| wall | m7i.xlargeで約16〜20 h。vCPUの多い型なら比例して短くなる（型と単価は実行計画で確認） |
| RSS | 変換は半荘数に依存せず、workerあたり約60 MiB（#316実測）。学習は1モデルずつ全件読込みで、読込み+tensor化はM1 約7.9 MiB/半荘、M0 約5.8 MiB/半荘、R1 約0.9 MiB/半荘（#316実測）。850半荘でM1 約6.6 GiB + torch約0.2 GiB。16 GiBの型に収まるが、学習本体の追加分は未計測で、それを超える規模では逐次読込みが要る |
| 保存 | source archive約69 KB/半荘（gzip）→ 約60 MB、保存費は無視できる。datasetは既定では保存せず、sourceから再生成する。dataset容量はM0 2.46 MB、M1 3.26 MB、R1 0.35 MB/半荘（#316実測、非圧縮）で、850半荘の3つ合計で約5.1 GB |
| 回収 | source archive・`evidence.json`・replay summary・dataset / artifactのmanifest・artifact（weights含む）・評価report・環境記録。datasetは回収しない |

- 月USD 20は既支出込みであり、C0生成・Learningの暫定枠はUSD 8（#84）。当月の支出（例：arena #503の上限見積り
  USD 2.68）を差し引いた残額を実行計画で確認してから規模を決める。残額をUSD 20やUSD 8と仮定しない
- pilot（50半荘）は約2.6 vCPU-hで、AWS以外の環境でも実行できる。Windowsではnative backendが使えないため、
  WSLかLinuxで実行する
- 本番の生成・replayは数十時間規模になるため、ユーザーのPCでは長時間の占有になる。AWSで実行する場合は個別の起動承認が要る

## 8. 決めたことの一覧

| 項目 | 決定 |
|---|---|
| 教師 | C0、生成時のArena pinのlisjong revision、rust backend、全件replay-verifyで同一性を確認 |
| source | `arena-policy-source-record-v2`、本番SCIENTIFIC、RiichiEnv、archiveで回収 |
| 学習する行動 | 打牌、CALL（Pass・Chi・Pon） |
| guardに置く行動 | 和了、リーチ、槓・九種九牌を選ばない、Chi / Ponの無い反応はPass。例外0件を確認してから使う |
| envelope | 使わない。牌効率は診断として測る |
| オーラス相当局 | 学習に含め、層別に報告。後段の扱いは後段で決める |
| 割り込まれた反応 | 教師の選択をlabelにする。適用の有無はL2aでは使わない |
| 分割 | 半荘単位、TRAIN / SELECT / OFFLINE-EVAL、pilotは別seed |
| モデル | L0.2形の小MLP、幅{64, 256}、weights.f32、Rustで推論できる規模 |
| baseline | B-TS、B-PASS、M0 |
| 成立条件 | 6節（DISCARD・Chi・Ponを別に判定）。0.85は点推定、0.40は95%区間下限。support不足は種別ごとに判定保留 |
| 規模・費用 | 初期案850半荘、約62 vCPU-h、AWSなら概算USD 5〜6（M1 / R1は#316で実測済み）。実行前に当月残額を確認 |

## 9. 担当と後続Issue

| 順 | 内容 | repository | 依存 |
|---|---|---|---|
| A | 分類関数・教師規則の確認・行動別support計数のCLI（source recordを読む） | lisjong | なし |
| B | pilot 50半荘のDEVELOPMENT生成・replay-verify・archive、Aの実行、規模の確定 | Arena（生成）＋lisjong（計数） | A |
| C | 打牌encoding M1・反応候補dataset / scorer R1・L2a guardと合成runtime・artifact | lisjong | なし（Bと並行可） |
| D | OFFLINE-EVALの評価tool（半荘cluster bootstrap、層別、train-serve一致、baseline B-TS / B-PASS / M0、閾値の凍結） | lisjong | C |
| E | 実行計画：seedの予約（SCIENTIFIC allocation）、型・費用・当月残額の確認、AWS起動の承認、生成・replay・archive | Arena | B、C（encodingのidentity固定） |
| F | dataset変換・学習・選択・凍結・OFFLINE-EVAL 1回・対局動作確認16半荘・結果記録 | lisjong（学習・評価）＋Arena（対局） | D、E |

- 生成・実行・対局評価はArena、feature・教師label・学習・artifact・推論はlisjongが持つ（ADR 0008）。
- project #74 / #84へは、本書の決定（8節）と、残る依存（当月の予算残額、Bのsupport計数、AWS起動の承認）を引き渡す。

## 10. 結果

#318でOFFLINE-EVALを1回評価した結果である。正本：[#318の結果](https://github.com/lisbun/lisjong/issues/318#issuecomment-6270387381)。source `ba02b591…`
（arena#506、SCIENTIFIC、850半荘）、lisjong `999553b`、gate `lisjong-l2a-offline-eval-gate-v1`（digest `77b87f79…fdd4`）。
OFFLINE-EVAL（150半荘・104,180判断）は1回だけ評価した（ledger entry `offline-eval-0001.json`）。

### 10.1 判定（6節）

**L2a: 打牌はNOT ESTABLISHED、ChiはNOT ESTABLISHED、PonはPARTIAL。** 「C0の打牌・鳴き判断の蒸留が成立」とは
呼ばない。CALL全体の成立とも呼ばない。[]は半荘cluster bootstrapの95%区間。

| 行動 | 判定 | 条件 |
|---|---|---|
| DISCARD | NOT ESTABLISHED | (1) M1打牌一致 0.818 [0.815, 0.821] < 0.85（未達）、(2) C0固有層の区間下限 0.721 ≥ 0.40、(3) M1−M0の区間下限 +0.112 > 0 |
| Chi | NOT ESTABLISHED | (1) balanced accuracy 0.771 [0.747, 0.795] < 0.85（未達）、(2) 行動一致 0.522 < 0.85 |
| Pon | PARTIAL | (1) balanced accuracy 0.899 [0.891, 0.907] ≥ 0.85、(2) 行動一致 0.828 [0.812, 0.844] < 0.85（未達） |

| 行動 | 結果の表記（2.2節） |
|---|---|
| 和了 / リーチ | 教師規則guard（C0と同じ決定的規則）。学習成立の対象外 |
| 槓 / 九種九牌 | 教師がsupportを持たない（選ばない）ため対象外。guardは選ばない |
| 打牌 / CALL | 上の判定 |

- 5.1の正しさはすべて1.000（合法性104,180 / 104,180、guard、train-serve一致はM1・M0 72,467行、R1 20,142行）。
  教師規則の例外0。3.2の下限はすべて満たし、判定保留は無い
- **PASS_RESPONSE行はpilot・本番とも0行で、そのguardは確認されていない**
- 4.5節により、#318の中で再学習・閾値変更・特徴量変更による救済はしない

### 10.2 追加特徴量（M1）の効果

同じデータ・同じ選択規則で、M1の打牌一致率はM0より **+0.118**（区間 [+0.112, +0.123]、半荘単位の対）高い。
C0固有層ではM1 0.726に対しM0 0.423である（B-TSは定義上0）。これは4.2節の「追加した特徴量一式」の効果であり、
個々の特徴量の寄与ではない。また、蒸留の成立（10.1）・棋力改善とは別の結果として扱う。

| 指標 | M1 / R1 | M0 | B-TS / B-PASS |
|---|---:|---:|---:|
| 打牌一致率 | 0.818 | 0.700 | 0.710 |
| C0固有層の打牌一致率（20,996行） | 0.726 | 0.423 | 0 |
| Chi / Pon balanced accuracy | 0.771 / 0.899 | — | 0.500 / 0.500 |
| 反応全体の一致（Passを含む） | 0.961 | — | 0.915 |

### 10.3 不一致の所在（既存reportの分析、新しい評価はしていない）

一致率の低い層だけでなく、行数 × 不一致率（不一致件数）で見る。打牌の不一致は計13,192件。

| 層 | 打牌一致率 | 不一致件数に占める割合 |
|---|---:|---:|
| 打牌前の向聴1 / 2 | 0.765 / 0.804 | 40.2% / 38.8% |
| 門前 / 副露 | 0.813 / 0.859 | 91.1% / 8.9% |
| C0固有層 / TwoStepと同じ層 | 0.726 / 0.856 | 43.7% / 56.3% |
| 他家リーチ2人以上 | 0.730 | 5.2% |
| リーチ宣言牌 | 0.713 | 1.8% |

- 一致率の低い他家リーチ2人以上・宣言牌は、件数では小さい。不一致の大半は門前の1〜2向聴にある
- 反応（CALL）の不一致は計777件で、門前が71.3%（反応一致0.970）、副露が28.7%（0.869）。R1のChi balanced accuracyは
  門前0.583 / 副露0.849と差が大きい。ただしreportには層別のChi / Pon別件数が無いため、門前Chiの寄与件数は
  このreportからは出せない
- いずれも原因は断定しない。次の仮説は新しいIssue・新しいseedで扱う

### 10.4 残り・注意

- 対局での動作確認16半荘（5.4節）は実施済みでPASS（10.5節）
- 時間：dataset 2.4 h、学習6回で14分、OFFLINE-EVAL 4.66 h（単一process）、計7.44 h（c7i.2xlarge）
- 費用は推計約USD 3.43（instance稼働時間 × 単価）。請求反映後の実額は未確定

### 10.5 対局での動作確認（5.4節）

凍結したM1 `c976ed4a…`・R1 `d565713f…`を1席、C0を3席にした16半荘（arena#507、seed `506900..506915`、DEVELOPMENT）。
正本：[arena#507の結果](https://github.com/lisbun/lisjong-arena/issues/507#issuecomment-6278840709)。
**動作の正常性の確認であり、強さ・局素点の比較はしていない。** 10.1の判定は変わらず、蒸留は成立していない。

- **PASS**：完走16 / 16、非合法0、例外0。L2a席の判断2,309（scorer：DISCARD 1,571、CALL 405）
- 推論時間（判断ごと、中央値 / p95 / 最大）：DISCARD 14.0 / 122.7 / 409.8 ms（大半は候補encoding）、
  CALL 1.18 / 4.89 / 7.27 ms
- process RSS：load直後243 MiB、終了時321 MiB、peak 479 MiB
- 実行条件：Arena `606759b`（arena#508のhead。squash merge後のmain `6942d00`と同じtree）、lisjong `999553b`、Rust backend、torch 2.13.0+cpu、WSL（8 cores）、1 process
- native wheelは`999553b`でbuildした`9aa0ca4e…`（`SOURCE_REVISION` `999553b`）。学習・OFFLINE-EVALは同じlisjong `999553b`に
  wheel `ee02748f…`（`SOURCE_REVISION` `174bf48`）を入れて実行した。native sourceは同じで、埋め込まれたrevisionだけが違う

## L2a後続の限定診断

#324の集計定義、教師規則との静的照合、実行手順と入力待ちの状態は
[限定CALL診断](learning-l2a-call-diagnostic.md)を参照。
