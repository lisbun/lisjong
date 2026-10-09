# ロン合法確率の表現・正解・測定設計（#262）

[lisbun/lisjong#262](https://github.com/lisbun/lisjong/issues/262)（親: #255）の設計と実装状況。
段階Aの表現追加・availability・raw整合性検証は実装済み。
B1の正解計算・追加source reader・履歴検証は実装済み。
producerと2半荘pilotはlisjong-arena#457で完了した。段階Cの本測定は400半荘
（seed 936000..936399、生成はlisjong-arena#460）で実施し、#262はcloseした。
推定器（#277）は段階1（リーチ者）・段階2（副露者）とも同じsourceで測定し、成功条件を満たした。
数値・digest・実行条件は[#262の結果](https://github.com/lisbun/lisjong/issues/262#issuecomment-6060675233)と
[#277](https://github.com/lisbun/lisjong/issues/277)のコメントを正本とする。
936000..936399のevalは段階1・2で使用済みであり、後続の最終確認には新seedのsourceが要る。
[追加source v1のwire契約](ron-legal-source.md)を正本とする。
#263は[役・符・点数計算](hand-scoring.md)として完成済みで、正解の役判定に利用する。

## テーブルの意味とcontext

`ron_legal_probability(t)`は、観測者の情報Iと固定context protocol Cの下で、
「tが他家から出た場合に、対象windのロンが合法である」同時確率である。

```text
Y(t) = [t ∈ W] AND [W ∩ 対象席の河 = ∅]
       AND [見逃しフリテン状態 = none] AND [手牌+tに役がある]
P(Y(t) | I, C) <= P(t ∈ W | I)
```

Wは真の構造的待ち集合。待ち・非フリテン・役の周辺確率の独立な積ではない。
相手が実際にロンを選ぶか、頭ハネ・複数ロンの解決、精算、期待失点は含めない。
#237のラベルB（実際にロンされたか）や放銃確率そのものとも区別する。

Cのidentityは`project-standard-normal-discard-ron-v1`とする。
採用ルールはlisjong-engineの`PROJECT_STANDARD_RULES`（`project-standard-v1`）。
時点は観測者の判断と同じsequenceの行動適用前で、対象席はstable 13-equivalent。
その時点の手牌・副露・河・見逃し状態を保持したまま、対象牌種が他家の**通常打牌**として
出た反実仮想を評価する。河底・海底・槍槓・嶺上・天和・地和を付けない。
自風・場風と、成立したリーチ/ダブルリーチ・一発はsnapshot時点の実状態を用いる。
リーチの宣言だけを成立済みのリーチ役へ読み替えない。

context protocolは固定の評価規則であり、未知の他家手牌や見逃し状態を推論時に渡す
追加contextではない。正解経路は実状態を使うが、推論側は`PolicyInput`に含まれない
事実について周辺化する。現在の`PolicyInput`はダブルリーチ・一発を個別に持たないため、
正解用contextを特徴へ漏らさない。成立済みリーチならこれらの違いによらず役はある。

34牌種を評価し、観測者が実際に切れる牌だけへ限定しない。場全体に4枚見えていても、
対象手牌に1枚加える構造として成立すれば非ゼロでよい。対象手牌・副露自身に同種4枚が
ある場合の5枚目は既存structural builderが待ちに含めない。この反実仮想値を、実行可能な
全候補のaction legalityと混同しない。contextを変える場合は別protocolを定義し、
既存テーブルの意味を黙って変更しない。形別ロン合法テーブルは追加しない。

## 段階A: availabilityと整合性

`HandBelief`に`ron_legal_probability_raw: tuple[int, ...] | None = None`を追加した。
34牌種canonical順、各要素は厳密なintで`0..PROBABILITY_MAX_RAW`。
`None`は未提供、全0はロン合法確率0の推定であり、区別を維持する。

| structural wait | 形別group | ron legal | 許可 |
|---|---|---|---|
| None（Level 0） | 全None | None | 許可 |
| None | 全None | 34値 | 拒否 |
| 34値（Level 1） | 全None | Noneまたは34値 | 許可 |
| 34値（Level 2） | 7形全部あり | Noneまたは34値 | 許可 |

既存のLevel 0/1/2はstructural waitのavailabilityのまま使う。ロン合法の有無は独立の
optional fieldであり、第4levelや形別groupとのall-or-none条件を作らない。

**constructorで同じslotの`ron_legal_raw <= wait_raw`を強制する。** 長さ・型・範囲、
waitなしのron提供、rawで1 unitでも上限を超える入力はfail closedに拒否する。
constructorは修復・上限処理をしない。既存constructor呼出しは既定のNoneで互換性を維持する。
既存の構造的待ち・形別groupの定義やconstructor制約は変えない。

推定器側は同じ判断・対象席・情報Iのwaitとronを組にして出力する。
正常な未制約予測qに対しては`min(q, wait)`を適用し、既存`probability_to_raw()`で
両方をround-half-to-evenにする。丸め後にも大小関係を検証する。
不正値・非有限値・identity不一致をclipで救済しない。
waitが未提供ならronも未提供とし、別のwaitモデルへの黙示的fallbackをしない。
上限処理前の違反率・超過量、処理後の違反件数を診断に記録する。
loss用epsilonは出力を変えるためには使わない。

段階Aは#259の固定モデルを必要としない。test用の明示的な値で表現契約を検証し、
本番推定器や学習を先行追加しない。architectureとbelief packageの説明も段階Aへ更新済み。

## フリテン規則の照合

照合対象はengine revision `96b9796c76ef5db8f3968f689a1ca6f3dfc9aa3b`。
`furiten.py`、`ron_legality.py`、`reaction.py`、`player_state.py`、`round_state.py`を確認した。
以下はこのprojectの実装規則であり、外部サービス一般の規則とは断定しない。
producerが別revision/RuleSetを使う場合は、生成前に差分と同値性を照合して記録する。

照合の正本: [フリテンの導出・更新](https://github.com/lisbun/lisjong-engine/blob/96b9796c76ef5db8f3968f689a1ca6f3dfc9aa3b/src/lisjong_engine/furiten.py)、
[合法ロン条件](https://github.com/lisbun/lisjong-engine/blob/96b9796c76ef5db8f3968f689a1ca6f3dfc9aa3b/src/lisjong_engine/ron_legality.py)、
[反応選択と見逃しの区別](https://github.com/lisbun/lisjong-engine/blob/96b9796c76ef5db8f3968f689a1ca6f3dfc9aa3b/src/lisjong_engine/reaction.py)、
[局の状態遷移](https://github.com/lisbun/lisjong-engine/blob/96b9796c76ef5db8f3968f689a1ca6f3dfc9aa3b/src/lisjong_engine/round_state.py)。

| 理由・事象 | 採用engineの処理 | 正解への反映 |
|---|---|---|
| 捨て牌フリテン | `derive_furiten_reasons()`が真の待ち集合と河の交差から毎回導出。鳴かれた牌も河に残る | 3m/6m待ちで河に3mがあれば、6mを含む全待ちがロン不可 |
| 未リーチの合法ロン見逃し | `ReactionResolution.ron_passed_seats = ron_capable_seats − ron_selected_seats`に対してTEMPORARYを記録 | 対象席の次の実ツモまで全牌種のロンを不可にする |
| 成立済みリーチ後の合法ロン見逃し | `is_riichi_established`ならRIICHIを記録 | 局の終わりまで保持。手牌が変わっても現在Wとの再交差で解除しない |
| 一時フリテンの解除 | `_draw_into_hand()`で自席のツモ時に解除。通常ツモ・嶺上ツモの両方 | 他家のツモや自席のチー/ポン後打牌では解除しない |
| 役なしで構造だけ完成する牌を通過 | 合法Ron actionがないので`ron_capable_seats`に入らない | 新しい見逃しフリテンを立てない。既存フリテンはそのまま |
| リーチ後のツモ和了を選ばず打牌 | 見逃し更新はreaction resolutionの合法ロンだけ。ツモ見逃しによるRIICHI更新はない | このrevisionでは見逃しフリテンを立てない |
| ロンを選択したが頭ハネで不成立 | `ron_selected_seats`には残り、`ron_passed_seats`に入らない | 「実際に和了しなかった」を見逃しと扱わない |
| 河底/槍槓等で合法ロンを見逃した過去 | 実際のreaction origin・状況・役で合法候補を決めてから同じ見逃し更新 | 出力Cが通常打牌でも、過去の特殊contextによる見逃し状態は保持する |

現在のCで河底・槍槓を数えないことと、過去の和了機会を通常打牌へ読み替えることは別である。
履歴の合法性を再検証するときは、当時のcontext・RuleSetを使う。
暗槓への槍槓なら国士限定等の採用ルールも適用する。

既存`riichi_ron_label()`は非リーチ者のTEMPORARYを扱わず、呼出側がリーチ後のツモ牌も
`passed_tile_types`へ渡せる契約である。#262の新正解builderとして流用しない。
S1ラベルAとの一致を主張できるのは、リーチ成立・stable hand・同じcontext・見逃し定義が
一致する範囲だけ。過去のS1 source/結果を遡及的に変更しない。

## 段階B: 追加factと正解計算

### v1を保持する追加source

[#256 v1](hand-belief-accuracy-source.md)のschema、同一sequence、手牌・副露、
player-safeな`decisions.jsonl`を変更しない。新しい拡張sourceを別directoryに置き、
次の3fileでv1を参照する。具体的なfieldとcommit内の順序は[追加source v1](ron-legal-source.md)に定義する。

| file / schema | 必須の内容 |
|---|---|
| `manifest.json` / `lisjong-ron-legal-source-manifest-v1` | 元v1 manifestのSHA-256、C identity、RuleSet identityと意味上の値、全producer revision、seed/split、追加2fileのbytes/rows/SHA-256、履歴coverage |
| `ron_facts.jsonl` / `lisjong-ron-legal-fact-record-v1` | v1と同じDecisionKey、局instance ID、行動適用前の履歴境界、他家3席それぞれのsequence・見逃し状態・成立riichi status・一発状態 |
| `ron_history.jsonl` / `lisjong-ron-legal-history-record-v1` | 半荘/局instance/連続履歴index、局開始からの見逃し更新・自席ツモ・リーチ成立等の時系列と、reactionの合法和了機会・選択・解決の証拠 |

元v1 directoryの場所はreaderの明示的引数で解決し、manifest内の任意pathを実行・探索しない。
元manifest digestと全file digestを検査する。旧v1単独はstructural truth用として引き続き読めるが、
ロン合法ラベル用readerへ渡すと追加fact欠測として拒否する。欠測をnoneや全0にしない。

`ron_facts`の他家3席は席順でちょうど1回ずつ。各席に以下を必須とする。

- `missed_ron_state`: `none` / `temporary` / `riichi`。`none`は確認済みの状態で、欠測ではない。
  DISCARDはここに持たず、現在Wと公開河からlisjongが導出する。
- `riichi_status`: `none` / `riichi` / `double_riichi`（成立済みの状態）。
  `is_ippatsu`: 明示的bool。公開`RiichiState`のNONE/DECLARED/ACCEPTEDと整合を検査する。
  DECLAREDを無条件に役ありにせず、状態遷移の履歴とengine実状態を照合する。
- snapshotの`sequence`: DecisionKeyと同値。局instanceと履歴境界は同じ局の同じ状態を参照する。

### 見逃しを検証する履歴

v1のsequenceはselector呼出しの通し番号であり、ツモ・reaction解決・成立をすべて番号付け
してはいない。別に**局開始からのcommit済み遷移順**で0始まりの連続履歴indexを持つ。
snapshotの境界kは半開prefix `[0,k)`を意味し、進行中の判断・未来の解決を含めない。
履歴中のselector sequenceとの対応も明記し、単なる牌のorderをsequenceに読み替えない。

| 履歴の事象 | 記録する事実・検証 |
|---|---|
| 局開始 | 一意な局instance、初期見逃し状態none、初期riichi状態。前局の理由を持ち越さない |
| 自席の実ツモ | 席、通常/嶺上の区別、commit境界。TEMPORARYだけを解除する起点 |
| リーチ宣言/成立/一発変化、鳴き/槓・打牌 | 前後の状態と順序。riichi見逃し理由の選択とsnapshot contextを検証できる事実 |
| reaction解決 | 起点（discard/kakan/ankan）、発生元席・対象牌（赤5区別）、全候補席の合法action集合、選択action、ron capable/selected/awarded/passedの各席集合 |
| 見逃し更新 | 対応reaction ID、対象席、更新前後の理由。合法Ronが提示され、Ronを選ばなかったことと一致する |

reactionの和了機会には、当時の対象席のstable手牌・副露・河、成立riichi/一発、
和了方法・origin・last-tile状態・ルールを検証用factとして結び付ける。
`ron_capable=true`というboolだけを正解として受け入れず、lisjong側でstructural truth・
フリテン状態・#263の役判定を用いて合法機会を照合する。
合法actionと選択はengineの観測事実であり、Arenaが独自の役・ロンラベルを計算しない。
物理IDは必要なengine側対応付けにだけ使い、lisjong wire shapeへ任意の内部objectを復元しない。

通常打牌だけでなく槍槓・河底の実contextも履歴へ残す。頭ハネの不成立をpassとしない。
ツモ和了を選ばなかった事実があっても、採用規則にない見逃し更新を追加しない。
reaction候補を持たずselectorを呼ばない進行も、commit履歴から漏らさない。
複数席の選択収集中には見逃し状態を先行更新せず、reaction解決のcommitでだけ反映する。

Arenaは局末までの履歴をflushし、独立に数えた局・遷移・reaction・対象判断のcoverageを
追加manifestへ記録する。連続index・終端・snapshot境界・reaction IDの対応を検査する。
「snapshotとログの両方から同じ見逃しを落とした」場合も、runnerのcoverage参照と一致しなければ
拒否する。readerだけでproducerの記録の真実性を証明できるとは扱わない。

学習専用fact・履歴はPolicy、`PolicyInput`、通常のplayer-safe traceへ渡さない。
engineの`RoundEvidence`は意図的にMissedRon/ron capableを射影しないため、そこから
非公開事実を復元しない。producerの専用記録経路で内部状態・イベントを読み取る。

### readerと欠測

新readerはv1の完全性検査の後に追加fileを結合し、局開始から状態をreplayして
snapshotの理由・contextと完全一致することを検証する。未来の履歴は現在ラベルへ反映しない。
必要file/field/局開始の欠測、key欠落・重複、時点ずれ、未知schema/理由/ルール、
未リーチでRIICHI理由、一発なのに未成立、合法機会とpassの不一致、解除順序の不一致は拒否する。
非聴牌や捨て牌フリテンの行でも、欠測検査を省いてラベル0を返さない。
欠測行を黙って除外した部分populationの評価結果は出さない。

### #263を使う正解builder

検証済み手牌から`exact_hand_belief_with_waits()`を呼んでWを取得し、公開河との交差と
replay済みの見逃し理由を使う。待ち判定・分解・役計算を新しく二重実装しない。
非聴牌なら34slotとも0。フリテンなら、役ありかどうかによらず全待ちのロンが不可。
役の有無は`evaluate_win(WinningHand(...), WinContext(...), PROJECT_STANDARD_SCORING_RULES)`で
確認する。concealed tilesには和了牌を含めず、`WinMethod.RON`、`WinSituation.NORMAL`、
実際の自風・場風・成立riichi・一発、`URA_DORA_EXCLUDED`を明示する。

このCは牌種の反実仮想なので、#263の物理牌検査との接続を次のように固定する。

- 役の有無だけを問い合わせるprojectionでは表ドラ表示牌を明示的に除外して空tupleにする。
  元の表示牌はsourceに保持・検査するが、点数計算結果は使用しない。これは未知の表ドラを
  0枚と推定する処理ではなく、ドラが役を作らない性質を使った役判定専用projectionである。
  実際の表示牌を足すと反実仮想tが同種5枚目になり得るため、物理牌例外をラベル0へ変えない。
- 和了牌の赤/通常は対象の手牌＋副露内で物理的に追加できるものを選ぶ。5は通常を優先し、
  通常5を既に3枚持つ場合は赤5を選ぶ。赤を選べることも検査する。場全体の残り枚数では
  候補を消さない。赤/通常で役の有無が変わらないことをtestで固定し、点数は解釈しない。
- 履歴における**実際の**合法和了機会の照合には、当時の実牌と実contextを使う。
  通常打牌用Cへ置き換えない。表ドラを除外する場合も役の有無だけを見る同じ明示projectionを使う。

`SCORED`は役あり、`NO_YAKU`は正常な役なし。W外のslotは0とする。
W内なのに`NOT_COMPLETE`なら既存builderとの不整合として拒否する。
`ValueError`や`ScoringBackendUnavailableError`を役なしに変換しない。
native/API versionを事前確認し、利用不可なら正解生成・評価を止める。
既存coreのimportや段階Aにはnativeを必須化せず、engine/Arenaへのruntime依存も追加しない。

## 後続作業の分割と完了条件

| 作業 | owner / 依存 | 成果物 |
|---|---|---|
| A: 表現追加 | lisjong、設計の確定後。#259/#260/Bに依存しない | optional field、availability/raw整合性test、architecture/belief説明 |
| B1: 追加契約・正解 | lisjong、Aと#263 | strict reader、履歴検証、正解builder、合成fixtureによる境界test |
| producer | lisjong-arena、B1のwire契約・readerが確定後に別Issue化 | engine記録経路、key/履歴coverage、少数seed pilot、consumer読込成功 |
| C: baseline測定 | lisjong、producer pilot完了後 | 事前登録、新seed source、絶対精度・coverage/support・制約のIssue記録 |

B1のfixtureは実測の代わりではない。producerは#453の実際のengine selector wrapperを
拡張する経路を候補とし、RiichiEnv用`LocalGameRunner`へ戻さない。selectorだけで全遷移を
拾えない場合のhook追加はproducer Issueで具体化し、未確認のengine APIを設計上の既存APIと扱わない。
[v1 producerの専用文書](https://github.com/lisbun/lisjong-arena/blob/f6c8e8132ea8236491250667d38a5e3e7ddce64f/docs/hand-belief-source-453.md)と
`scripts/generate_hand_belief_source_255.py`を起点とする。
今回のPRは`Refs #262`。後続PRも全段階が完了するまではIssueをcloseしない。

focused testには、Level 0/1/2とronの全組合せ、型/範囲/丸め境界、
リーチ成立/宣言のみ、門前と副露の役あり/役なし＋ドラ、待ちの一部だけが河にある例、
鳴かれた河牌、同巡解除前後・鳴きでは解除しない例、リーチ後見逃し、ツモ見逃し非更新、
頭ハネ、過去の槍槓/河底見逃し、赤5と表示牌による反実仮想の物理境界を含める。
正解系moduleをimport禁止にしても推論が`PolicyInput`だけで動き、player-safe readerが
追加fact/historyを開かないことも固定する。

## 段階C: baselineの絶対精度

#259の結果を待たず測定できる主baselineとして、trainの**層・牌種別ロン合法出現率**を使う。
対応するwait出現率も同じtrain・重み・Jeffreys 0.5でfitし、waitを伴うHandBeliefとして提供する。
ラベルYがwaitの部分集合なので、同じ分母と平滑化では出現率でもron≤waitとなる。
推論時には固定したrateと公開層だけを使い、truthや履歴を参照しない。

副baselineは同じtrainのwait出現率に、対象席の公開河にtがあれば0とするmaskを掛けたもの。
これも対応waitを伴う。この現物処理は確定安全な近似であり、Wの別の牌が河にあるケースや
非公開の見逃し・役なしを完全には除けない。構造的待ち推定器の凍結モデルを用いる追加比較は
別途identityと提供範囲を事前登録できた場合だけ行い、#259の必須依存にしない。

#257と同じ観測者の打牌判断×他家1席を標本とする。リーチ者（DECLARED/ACCEPTED）、
門前非リーチ者、副露者で層別し、DECLARED件数も併記する。
episode=(半荘,一意な局instance,他家席)、全34slot平均→episode内行平均→episode間平均。
層別は当該層の行だけで重みを再正規化し、全体値は層別平均から作らない。
主指標log loss、副指標Brier・10bin校正表・合法打牌候補牌種内のloss・フリテン理由/役なし別診断。
真値による診断層はoffline報告だけで使い、baseline推論のmaskへ流さない。
loss用clipは`[1e-6,1-1e-6]`で共通。raw/SCALEの最終予測を評価する。

分割はseed単位train/valid/eval、fitはtrainのみ。初版のrate baselineに選択対象はなく、
validはsupport確認だけ。半荘単位paired bootstrap 2,000回、RNG seed 262、95%百分位区間。
pilotで費用・層別supportを確認し、**本測定の生成前**に半荘数・正確な分割/seed・producer・
RuleSet・C・計算backend・source/モデルidentityをIssueへ登録する。数やseedをこの設計で推測しない。
live ledgerの全履歴（#259/#260の予約・使用分も含む）と照合して新seedを予約する。
v1の過去snapshotから欠測した見逃し状態を埋めたり、過去のsourceを拡張済みと扱ったりしない。

対象/未提供coverage、正例slot・行・episode・独立半荘数をsplit/層別に報告する。
正例0は「正例への能力を評価できない」、正例独立半荘50未満またはepisode100未満は
「support不足」と明記する。負例が多いためlossが小さい層を精度合格としない。
段階Cは絶対精度の記録であり、改善合格・強さ改善の判定を行わない。
結果・digest・実行条件・未実施確認を#262へ記録し、その後の推定器改善は別Issueとする。

### 測定moduleの実装状況

baselineの推論は`lisjong.learning.ron_legal_baseline`（`PolicyInput`と固定rateだけを使い、
正解系module・nativeをimportしない）、fit・集約・区間は`lisjong.learning.ron_legal_accuracy`に置く。
入力は`base/`（#256 v1）と`ron/`（追加source）を持つpopulation directoryで、登録した
train/valid/evalのseedとproducer identityを、ラベルを読む前に全sourceのmanifestと照合する。
evalはmanifestの`test`分割に対応する。native scorerの`SOURCE_REVISION`が登録した
producerのlisjong revisionと異なる場合は測定を止める。既存の結果fileは上書きしない。

```text
python -m lisjong.learning.ron_legal_accuracy   --train FIRST..LAST --valid FIRST..LAST --eval FIRST..LAST   --producer producer.json --output result.json POPULATION_DIR...
```

`--support-only`はfitとsupport集計だけを行い、指標・校正表を出さない（`--eval none`可）。
#457のpilot 2半荘（935000 train / 935001 valid）でこのmodeの読込・ラベル付け・集計が
通ることを確認した（WSL、wall 30秒、最大RSS 約0.5GB）。これは接続確認であり、
精度の測定結果ではない。pilot seedを本測定へ再利用しない。

## 推定器: 待ち推定 × 確実な0（#277）

[lisbun/lisjong#277](https://github.com/lisbun/lisjong/issues/277)の推定器は、学習済みの構造的待ち推定器の出力に、
`PolicyInput`だけで確実にロン不可と言える牌種を0にする変換（`certain-ron-illegal-zero-v1`）を
掛ける。新しい学習と確率的な補正は行わない。推論は`lisjong.learning.ron_legal_estimator`に置き、
正解系module・native・学習用sourceをimportしない。

0にする牌種は次の2つで、赤5と通常5は同じ牌種として扱う。

1. 対象席の河にある牌種（現物）。wait_genbutsu baselineと同じ処理
2. 対象席の最後の打牌より後に他家（観測者を含む）が切った牌種。対象席はその間ツモも副露も
   していないので手牌は同じで、待ちでロン合法だったなら見逃しフリテン、役なしなら今もロン不可。
   `Discard.order`は4席全体で一意・連番なので計算できる。対象席の打牌がまだない場合は
   配牌から手牌が変わっていないので、他家の打牌すべてが該当する

リーチ成立後に通った牌の全体（`PolicyInput`はリーチ宣言牌の位置を持たない）、河にある別の
待ち牌による捨て牌フリテン、副露者の役なしは確定できないので0にせず、過大評価として残る。
ronはwaitと同じrawか0なので、丸め後も`ron <= wait`になる。

提供範囲は待ち推定器と同じにする。段階1（リーチ者）は#245の凍結モデル
（`riichi-wait-features-v1`）を使い、観測者が非リーチ・対象席が唯一のリーチ者・対象席に打牌がある
判断だけに値を出す。範囲外は未提供（`None`）で、ゼロ予測や別の待ちモデルへ置き換えない。
段階2（副露者）は#259範囲1で選択済みのモデル（`open-wait-features-v1`）を使い、リーチしておらず
暗槓以外の副露がある他家だけに値を出す（観測者の状態は問わない）。門前非リーチは対象外である。
2つの推定器は別の名前（`riichi_wait_zero` / `open_wait_zero`）で別々に採点し、主張も分ける。

測定は`ron_legal_accuracy`へ推定器を差し込んで行う。推定器は値を出した行だけで採点し、
同じ行の2 baselineと対にした差を`estimators`に報告する（提供行・未提供行の数を併記）。
baselineの節は推定器なしの場合と同じ計算で、推定器・`--score`を指定しない実行の結果は
#262の時と同じである。待ちモデルのselectionは登録したSHA-256と一致しなければ読まない。

```text
python -m lisjong.learning.ron_legal_accuracy --train ... --valid ... --eval ... \
  --producer producer.json --output result.json [--score valid] \
  [--riichi-wait-selection selection.json --riichi-wait-selection-sha256 HEX] \
  [--open-wait-selection selection.json --open-wait-selection-sha256 HEX] POPULATION_DIR...
```

`--score valid`はevalの代わりにvalidを採点する（調整・診断用。evalのラベルは読まない）。
診断として、行ごとの牌種の和（待ち推定値、ron推定値、実際の待ち、実際のロン合法、0にしなかった
牌種の実際の待ち、0にした牌種の実際のロン合法）を同じ集約で出す。最後の値は変換が正しければ0になる。

### 結果の要約（#277）

eval（936240..936399）の提供行で、推定器 − wait_genbutsu baselineのlog loss差（95%区間）は、
段階1（リーチ者、17,651行）で −0.00431 [−0.00503, −0.00354]、段階2（副露者、22,443行）で
−0.00631 [−0.00718, −0.00541]。どちらも0にした牌種の実際のロン合法は0だった。
これは推定精度の結果であり、強さの評価ではない。

0にする変換で落とせない過大は、リーチ者で実際の待ちの約1%、副露者で約9%（主因は役なし）。
副露者に一律・副露形別の割合を掛ける補正はvalidで効果がなく見送った。
待ちモデルの学習source（lisjong `e6346ed` / engine `8735e89`）と測定source
（lisjong `994f529` / engine `91af75e`）はrevisionが異なる点を制約として残す。
