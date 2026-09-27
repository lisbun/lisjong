# Championの攻撃打牌へのkobalab 0004牌効率の組み込み（Issue #226）

Issue: [lisbun/lisjong#226](https://github.com/lisbun/lisjong/issues/226)

現Champion `PlacementAwareSpeedCallPolicy`（[#199](https://github.com/lisbun/lisjong/issues/199)）を変更せず、
その攻撃打牌（PUSH）の最終比較だけをkobalab 0004の牌効率へ置き換えたexperimental候補
`PlacementAwareSpeedCallKobalab0004DiscardPolicy`を追加した。あわせて0004の打牌評価を
参照版から共通moduleへ切り出した。**新候補の強さは未評価**で、Arena評価は後続作業とする。
新候補はkobalab氏の原botやRiichiLab「牌効率くん」と同一とは扱わない。

## 1. 識別と利用方法

| 項目 | 値 |
| --- | --- |
| class | `lisjong.policies.PlacementAwareSpeedCallKobalab0004DiscardPolicy`（`lisjong.policies.__all__`へexport） |
| module | `lisjong.policies.placement_aware_speed_call_kobalab_0004_discard` |
| 生成 | 引数なしの`PlacementAwareSpeedCallKobalab0004DiscardPolicy()`。instanceにcross-decision stateはない |
| identity | Champion系列と同じくclass名とexportで識別し、`identity`属性・定数は持たない。Arenaの`PolicySpec.identity`は登録時に明示的に定める |
| 共通処理 | `lisjong.policies.kobalab_0004_discard`（package-internal。新しい公開APIではない） |
| 由来 | kobalab/majiang-ai legacy 0004（commit `e75a9720a12b84c03e6c61c3960c1844b8982eb4`、MIT License）の`select_dapai()` / `SuanPai.paijia()`をlisjong exact shantenへ移植した評価（[参照版の記録](kobalab-0004-reference.md)） |

## 2. 統合境界

打牌decisionはChampionと同じ入力・順序で次のように処理する。

1. **候補制限**（Championの`_eligible_discard_actions()`）：役のない副露手のPUSHでの最小route shanten制限
   （`_route_preserving_actions()`）、オーラストップ目SPEEDの非聴牌FOLDでの最小危険度制限
   （`_top_fold_actions()`）。各filter内部のPUSH / FOLD判定はChampionのまま。
2. **分類**：制限後の候補集合を`_classify_branch()`（mechanism filter → defense filter）へ渡し、返った
   branchとeligible集合だけを使う。元の未制限集合でPUSHを再判定しない。0004で分類前に候補を絞らない。
3. **PUSH**：eligible集合を0004方式で比較する（§3）。門前・副露手、通常自摸・副露直後
   （`drawn_tile is None`）・立直宣言牌decisionを区別しない。Championの攻撃評価（FiniteHorizon和了評価、
   HandValueAwareの打点proxy・second-step、限定R5字牌切り補正）は実行せずに置き換える。
4. **PUSH以外**（`FOLD_COMMON_GENBUTSU` / `MECHANISM_DEFENSE_FILTERED` / `OTHER_CURRENT_FALLBACK`）：
   Championの親`TargetedHonorReleaseTerminalProgressionPolicy`の打牌選択へ同じ制限後集合を渡す。
   Action・analysis・例外ともChampionと同じである（分類はそこで再度行われる。§5の費用参照）。

| decision | 新候補 |
| --- | --- |
| 和了 | Championと同じ（和了優先） |
| 立直 | Championと同じAlways Riichi（合法な`RiichiAction`があれば宣言）。0004の`_allows_riichi_discard()`・即リー規則は持ち込まない |
| 立直宣言牌 | 合法候補（聴牌維持の打牌）の中で上記1–4。PUSHなら0004 |
| 通常自摸・副露後の打牌 | 上記1–4 |
| Chi / Pon / Pass | Championと同じ。Chi / Ponの採否は仮想mandatory discard後の最小向聴数で決まり、打牌選択を呼ばない |
| 槓・九種九牌 | Championと同じ（Championはこれらを積極的に選ばない） |

同じdecision入力での非打牌判断は変わらないが、打牌が変わった後の対局展開まで同値とは主張しない。

**役routeとの関係**：副露手でも0004の向聴数・改善牌は通常の構造評価で、役route外の牌も改善牌に数える。
役routeは候補制限にだけ効く。「役を保つ候補制限 + 構造的受入」であり、役付き和了への距離や受入を一貫して
評価する方式ではない。

**traced execution**：PUSHの0004選択は`analysis=None`を返し、ChampionのR5 analysisを流用しない。
非PUSHはChampionと同じ`TargetedHonorReleaseAnalysis`を返す。`choose_action()`とtraced経路は同じ1回の
判断から同じActionを返す。

## 3. 候補集合との契約

- 入力はそのdecisionで許可された非空の打牌候補集合（上記の制限後集合）で、返すActionはその集合内に限る。
- 打牌後向聴数が打牌前向聴数以下の候補があれば、その中で実残り枚数の受入が最大の候補を選ぶ。
  受入は厳密に増えた場合だけ更新し、評価順はpaijia昇順・source順（ツモ切り → 字牌 → 索子 → 筒子 → 萬子の
  rank降順、通常5 → 赤5）。参照版と同じ規則で、`legal_actions`の順序に依存しない。
- **合成版fallback**：そのような候補がなければ、制限後集合で最小の打牌後向聴数を達成する候補に限り、
  同じ規則で比較する。参照版の「評価順の先頭」fallbackは使わない（参照版には適用しない）。
  改善牌はその候補分を既存の一括API（`evaluate_discards_from_canonical_counts()`）で1回で求め、
  未評価を0で代用しない。fallbackの頻度は§4の固定入力では0件で、実対局での頻度は未測定である。
- 未見枚数は同一decisionの公開情報から再構成し（`derive_remaining_tile_inventory()`）、仮に切る牌を
  未見へ戻さない。赤5 / 通常5・手出し / ツモ切りは構造評価を共有し、paijia・Actionとしては区別する。
- 手中にない牌の候補は`Kobalab0004ReferencePolicyError`で拒否する（参照版と同じ契約）。

**意図した例外差**：ChampionのPUSH経路は未見枚数がFiniteHorizonの探索幅（3）より少ないと
`FiniteHorizonCompletionPolicyError`を送出する。新候補のPUSH経路はFiniteHorizonを通らないため打牌を選ぶ。
非PUSH経路ではChampionの例外が残る。未見枚数の再構成による入力検証は省かない。

## 4. 固定入力での判断差（Championとの比較）

[kobalab 0004 profile](kobalab-0004-belief-paijia.md#1-計測条件固定)§1の固定decision列3本で、
各decisionをChampionと新候補へ独立に与えた（Rust backend。新候補のPython backendでの一致は§5.1）。

| 入力（decisions） | PUSH打牌 | うち行動変化 | 変化のうち受入同数 | 変化のうち受入増 | 合成版fallback | 非PUSH打牌（すべて一致） | その他decision（すべて一致） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| champion同士（708） | 436（門前通常自摸419、副露手自摸13、副露直後4） | 117（門前114、副露手3） | 103 | 14 | 0 | 107 | 165 |
| two-step同士（684） | 424 | 124 | 118 | 6 | 0 | 90 | 170 |
| 0004参照同士（630） | 392 | 123 | 115 | 8 | 0 | 92 | 146 |
| 計（2,022） | 1,252 | 364（29%） | 336 | 28 | 0 | 289 | 481 |

受入は新候補の評価（同一decisionの未見枚数、形の改善牌）で数えた値で、変化した364件はすべて
打牌後向聴数がChampionと同じだった（向聴数を落とす変化はない）。「その他decision」は和了・立直可・
応答（Pass等）である。

- PUSH以外の打牌、和了・立直可のdecision、応答（Pass / Chi / Pon）では、全件でActionとanalysisが
  Championと一致した。
- PUSHで行動が変わった例の92%（336 / 364）は、受入が同数の候補間でpaijia / source順とChampionの評価
  （FiniteHorizon・HandValueAware・R5）が異なる選択をしたものである。残る28件は新候補が受入の多い
  打牌を選んだ（Championの比較は受入枚数だけではなく、受入の少ない打牌を選んでいた。個々の理由は分類していない）。
- 全入力で`choose_action()`とtraced経路のActionが一致した。

## 5. 同値性と性能

### 5.1 切り出しの同値性

基準は`main` `8bdfd3f5`（本Issue起票時）のsourceをlocal artifactへcopyしたものとし、本branchと同じ手順で
記録を比較した。記録は[#224 §8.4](kobalab-0004-belief-paijia.md#84-同値性)と同じfixtureを拡張したもので、
固定decision 3入力（計2,022 decision）と合成入力について次を含む。

- 参照版・Belief版の最終行動（`legal_actions`を逆順にした場合も）、評価順、候補ごとの打牌後向聴数・
  改善牌・立直可否、選択で比較したukeire列、選択結果、例外（型とmessage）
- 現Champion `PlacementAwareSpeedCallPolicy`の`choose_action_with_analysis()`（Action・analysis・例外）。
  候補制限の関数化がChampionを変えていないことの確認

合成入力は赤5、未見0の待ちでの立直、聴牌立直、和了形で和了actionがない立直、応答、4枚持ち、暗槓後の11枚、
同点、手中にない赤5の打牌（fail closed）、未見枚数の不整合を含む。

| backend | 基準 `8bdfd3f` | 本branch |
| --- | --- | --- |
| Rust | `b44071ef…547380` | `b44071ef…547380` |
| Python | `b44071ef…547380` | `b44071ef…547380` |

4通りすべてで内容のSHA-256が一致した（`b44071efcc87fcd208097abd4455ae11b378bf979dae703021c078d1f9547380`）。
契約外入力（打牌decisionで純手牌13枚）の記録も基準と一致した。upstream差分試験
（`test_kobalab_0004_upstream_differential.py`）は既知差分の分類を変えずにpassする。
新候補については、Python backendで同じ3入力を再生し、traced判断（Action・analysis・例外）がRust
backendの記録と全件一致することを確認した（708 / 684 / 630件）。

### 5.2 計測条件

| 項目 | 値 |
| --- | --- |
| 基準（B） | `main` `8bdfd3f5`（sourceのcopyを別processで実行） |
| 変更後（N / 新候補） | 本branch |
| native | #224のlocal source build（`.pyd` SHA-256 `8ca87921…446e1c`、`API_VERSION = 2`）。B・Nとも同じbuild。本Issueで`native/`は変更していない |
| backend | `LISJONG_SHANTEN_BACKEND=rust` |
| 環境 | CPython 3.14.7（MSC v.1944, 64 bit）、Windows 11 Home 10.0.26200、Intel64 Family 6 Model 189（8 logical CPU）、1 process、profilerなし |
| 手順 | `tools/profile_kobalab_0004.py timing`。passごとにPolicy instanceを作り直し、pass 0をcold、以降をwarmとする。各processのwarm pass合計の中央値をとり、その中央値を代表値とする。cross-decision cacheはない |

計測の順序と反復は計測前に`run_timing.sh`へ固定した。0004は0004入力（630 decision）で1 process 7 pass、
各版3 process、順序`B N N B B N`。Champion / 新候補はchampion入力（708 decision）で1 process 4 pass、
各2 process、順序`C K K C`。

### 5.3 0004の切り出し前後（Rust）

| 版 | B [ms] | N [ms] |
| --- | --- | --- |
| 参照版 | **128.8**（135.7 / 128.8 / 124.7） | **127.3**（127.3 / 126.3 / 127.4） |
| Belief版 | **186.5**（180.8 / 196.4 / 186.5） | **192.6**（198.0 / 192.6 / 180.4） |

一括構造評価の境界呼び出しはB・Nとも7 passで3,514回（1 passあたり打牌decision数502）で、切り出しによる
native往復・canonical変換の増加はない。差（参照版 −1.2%、Belief版 +3.3%）はどちらもBのprocess間の
ばらつき（Belief版で180.8–196.4 ms）の範囲内で、退行とは判断しない。
直前に同じ手順で行った計測は途中で中断した（計測中に作業treeを分割commit用に一時変更し、Belief版の
N 3回目がimportに失敗した）。完了していた分も同じ傾向だった（参照版 B 103.2 / 107.7 / 106.3、
N 104.6 / 102.7 / 104.0 ms。process間の水準の違いは計測時点の機械状態による）。中断した計測は代表値に使わない。

### 5.4 現Championと新候補（Rust、champion入力708 decision）

| 版 | warm pass合計 [ms] | うち打牌decision（543件） [ms] |
| --- | --- | --- |
| Champion `PlacementAwareSpeedCallPolicy` | 114,429.5 / 112,465.7 | 114,384.7 / 112,423.4 |
| 新候補 | 4,397.6 / 4,639.1 | 4,360.3 / 4,599.7 |

- 新候補はChampionの約1/25で、差はほぼすべて打牌decisionにある。PUSHの打牌でChampionの
  FiniteHorizon和了評価（horizon 3のexact DP）・HandValueAware・R5を実行しないためである。
  非PUSH（107件）は従来どおりChampionの評価を実行する。
- 新候補の一括構造評価は1 passあたり436回で、PUSHの打牌decision数と一致する（1 decisionあたり1回。
  合成版fallbackによる追加評価は発生しなかった）。
- 親評価の二重実行はない（PUSHでChampionの攻撃評価を呼ばないことはtestでも固定）。非PUSHでは
  候補制限後の分類（`_classify_branch()`）を新候補とChampionの親とで2回行うが、FiniteHorizon評価に比べて
  小さく、個別には計測していない。
- 応答・和了・立直可のdecisionの時間はChampionと同程度である。
- 対局全体の時間短縮率ではなくPolicy計算の比較である。行動が変わると以降の局面が変わるため、同じdecision列の
  再生で比べている（新候補はこの記録の708件中117件で記録と異なる行動を返す）。

raw JSON・計測script・基準sourceのcopyはlocal artifact root
`C:\Dev\lisjong-artifacts\issue-226-kobalab-0004-discard\`（`results/`、`scripts/`、`baseline-8bdfd3f/`）にある。
再現は`scripts/run_fixture.sh`（同値性）、`compare_champion.py`（§4）、`candidate_backend_equal.py`
（新候補のbackend一致）、`run_timing.sh`（性能）。

## 6. 検証

- `tests/test_placement_aware_speed_call_kobalab_0004_discard_policy.py`：非PUSH分岐（共通現物FOLD・
  mechanism防御・全候補fallback・オーラストップFOLD制限）のChampion一致、門前通常自摸・他家リーチ下の
  聴牌維持・副露手（route制限内の選択）・副露直後のPUSHでの0004選択、合成版fallback（route制限後の全候補が
  1 → 2向聴、受入最大の9pを選び、参照版規則なら白）、改善牌の一括評価、赤5 / 通常5・ツモ切り、
  `legal_actions`の順序、traced Action一致、PUSHでChampionの攻撃評価を呼ばないこと、立直・宣言牌・
  鳴き判断、探索幅不足時の例外差、手中にない牌の拒否。期待値はTile入口の向聴数計算と未見枚数から求める。
- `tests/test_kobalab_0004_reference_policy.py` / `test_kobalab_0004_upstream_differential.py`：
  既存の参照版・Belief版・upstream差分試験を共通moduleへ向けて維持（既知差分の分類は変更なし）。
- CIはPython backendのfull suiteと、`native-backend` jobでRust選択下のfull suiteを実行する。

## 7. Arenaへの引継ぎ

- 置き換え対象：Championの打牌decisionのうちPUSHと分類された攻撃打牌の最終比較だけ（§2）。
- 変化した判断：§4の固定入力で、PUSH打牌の29%（大半は受入同数の同点処理の差）。非打牌判断・守備分岐は同じdecision入力で不変。
- 後続作業：Arena catalog登録（`PolicySpec.identity`の決定）、seed選定、Champion半荘比較。
  鳴きがあるため、棒聴即リーprotocol（lisbun/lisjong-arena#406）へそのまま投入できるとは仮定せず、
  protocol適合性の確認から行う。
