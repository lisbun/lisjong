# 0004合成版のPUSH打牌へのBelief-paijia接続（Issue #230）

Issue: [lisbun/lisjong#230](https://github.com/lisbun/lisjong/issues/230)

`PlacementAwareSpeedCallKobalab0004DiscardPolicy`（[#226](https://github.com/lisbun/lisjong/issues/226) /
#227、以下「合成版」。[記録](placement-aware-speed-call-kobalab-0004-discard.md)）のPUSH打牌の比較で、
paijiaの入力だけを`Kobalab0004BeliefPaijiaPolicy`（[#218](https://github.com/lisbun/lisjong/issues/218) /
#219、[記録](kobalab-0004-belief-paijia.md)）と同じBelief由来の残余期待枚数へ替えたexperimental候補
`PlacementAwareSpeedCallKobalab0004BeliefPaijiaDiscardPolicy`を追加した。合成版は比較基準として変更せず残す。
**新候補の強さは未評価**で、本Issueでは強さの向上を主張しない。

## 1. 識別と生成

| 項目 | 値 |
| --- | --- |
| class | `lisjong.policies.PlacementAwareSpeedCallKobalab0004BeliefPaijiaDiscardPolicy`（`lisjong.policies.__all__`へexport） |
| module | `lisjong.policies.placement_aware_speed_call_kobalab_0004_belief_paijia_discard` |
| 生成 | 引数なしの`PlacementAwareSpeedCallKobalab0004BeliefPaijiaDiscardPolicy()`。合成版のsubclassで、instanceにcross-decision stateはない |
| identity | Champion系列・合成版と同じくclass名とexportで識別し、`identity`属性・定数は持たない。Arenaの`PolicySpec.identity`は登録時に明示的に定める |
| 推定器 | `belief_estimator = KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR`（0004 Belief版と同じ条件付き一様推定器、他家slot = `13 - 3 * 公開副露数`、`derive_non_player_hidden_belief()`）。推定器を変える場合は別の候補名にする |
| 比較基準 | 合成版`PlacementAwareSpeedCallKobalab0004DiscardPolicy`（変更点はPUSHのpaijia入力だけ） |

## 2. 変更点と維持するもの

打牌decisionは合成版と同じく、候補制限（`_eligible_discard_actions()`）→ 制限後集合の分類
（`_classify_branch()`）→ PUSHなら0004比較、それ以外はChampionの親の打牌選択、の順に処理する。

**変更点（PUSHの打牌だけ）**：0004比較の評価順（同点処理）に使うpaijiaの入力を
`未見枚数 − 他家3人の手牌内期待枚数`（fixed-point raw、尺度`SCALE`）にする。赤5は34牌種の5とは別axisで渡す。

**維持するもの**：

- 向聴数・改善牌の定義、Rust一括構造評価（`evaluate_discards_from_canonical_counts()`）、受入の入力
  （実残り枚数）と集計、受入が厳密に増えた場合だけ更新する規則、完全同点でのsource順。
- 合成版fallback（非悪化候補がなければ制限後集合の最小打牌後向聴数の候補を比較）。参照版の「評価順先頭」
  fallbackは持ち込まない。
- 門前・副露手・副露直後・立直宣言牌の対象範囲。役routeは候補制限にだけ効く。
- 非PUSHの打牌（選択・analysis・例外）、和了・立直（Always Riichi）・鳴き・槓・九種九牌の判断。
- traced executionでPUSHは`analysis=None`。`choose_action()`とtraced経路は同じActionを返す。
- 返すActionは制限後の合法候補集合内。

## 3. Belief経路の共有

Belief由来のpaijia入力の導出（他家slot数、推定器境界、残余の導出）を`kobalab_0004_reference`から共通module
`lisjong.policies.kobalab_0004_discard`へ移し、入口を`_belief_paijia_counts(policy_input)`にした。0004 Belief版と
新候補は同じ関数を使い、Policy間の依存（新候補 → 参照版module）は作らない。移動は関数の置き場所だけで、
処理・例外・fixed-point尺度・丸めは変えていない（§5.1で同値性を確認）。

- 入力は当該seatの`PolicyInput`だけ。真の他家手牌や山は参照しない。
- 未見枚数は`_PublicCounts`が1度だけ導出し、同じ`conservation`を受入・推定器・残余で共有する（#220）。
  候補ごとのBelief再生成やsnapshotを跨ぐcacheはない。
- 自手は`conservation`で既知として数えるため差し引かない。赤5を二重計上しない。整数枚数へ丸め戻さない。
- 保存則違反は`derive_non_player_hidden_belief()`がclampせずに拒否し、`Kobalab0004ReferencePolicyError`になる。
- Belief導出はPUSHの打牌decisionごとに1回で、非PUSH・非打牌経路では行わない（testと§4で確認）。

**意図した例外差**：他家の手牌枚数より未見枚数が少ない契約外入力では、合成版は打牌を選ぶが、新候補は
Belief由来の入力を導出できず拒否する（0004 Belief版と同じ契約）。実対局の入力では発生しない
（§4の固定入力で0件）。

## 4. 固定入力での判断差（合成版との比較）

**入力**：lisjong-engineの`RuleSet.default()`半荘をArenaのengine bridge
（`lisjong_arena.lisjong_engine.hanchan.run_policy_hanchan`）で同じPolicy 4体・seed 0で実行し、全seatの
`(DecisionContext, InternalAction)`を記録した（local artifact、repositoryへはcommitしない。
seedは開発用の非科学的runで、seed予約は行っていない）。記録時のlisjongは`main` `6261d9d`。

| 入力 | decisions | pickle SHA-256 | semantic digest |
| --- | ---: | --- | --- |
| 0004参照同士 | 1,247 | `853170a7…31bdd7e` | `0990bd0f…7350b0` |
| 合成版同士 | 892 | `a9e5c0ed…3ca125e` | `68abdc67…42393b` |
| Champion同士 | 1,011 | `d1a9d2a9…52cc050` | `176bdd67…29e757` |

engine `lisbun/lisjong-engine@96b9796`、Arena `lisbun/lisjong-arena@0603ec7`。RiichiEnvは導入できない環境だったため、
Arenaのimport連鎖にだけ現れる`riichienv`をimport-only stubで満たした（engine経路はRiichiEnvを呼ばない）。
§1（#213）のRiichiEnv固定decision列とは別の入力である。

各decisionを合成版と新候補へ独立に与えた（Python backend）。

| 入力 | PUSH打牌 | うち行動変化 | 変化のうち受入同数 | 合成版fallback | 非PUSH打牌（すべて一致） | その他decision（すべて一致） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0004参照同士（1,247） | 481 | 2 | 2 | 0 | 161 | 605 |
| 合成版同士（892） | 338 | 1 | 1 | 0 | 150 | 404 |
| Champion同士（1,011） | 372 | 0 | 0 | 0 | 175 | 464 |
| 計（3,150） | 1,191 | 3（0.25%） | 3 | 0 | 486 | 1,473 |

- 行動が変わった3件はすべて、合成版でpaijiaが完全同点（source順で決着）だった受入同数の候補間で、
  一様推定器の丸めにより残余paijiaが1 raw差になったものである（**量子化による同点崩れ**。
  例：合成版paijia 13 / 13 → 新候補 69,631 / 69,632）。受入の増減や向聴数の変化はない。
  推定器は赤5と通常5を別のphysical poolとして丸めるため、5を窓に含む牌で1 raw差が生じやすい
  （test fixture `TENPAI_TIE`でも同じ現象を固定した）。丸め規則は変更していない。
- 非PUSH打牌・その他decision（和了・立直可・応答）は全件でActionとanalysisが合成版と一致した。
- Belief導出（`_estimate_from_conservation()`）はPUSH打牌decisionごとに1回（計1,191回）、非PUSH・非打牌では0回。
- 全入力で`choose_action()`とtraced経路のActionが一致し、新候補の例外は0件。

非一様推定器では同点処理以外の順位も変わり得る。非一様な合成Beliefでpaijia順位と最終Actionが変わる例は
testで固定した（§6）。

## 5. 同値性と性能

### 5.1 既存Policyの同値性

基準（B）は`main` `6261d9d`のworktree、変更後（N）は本branch。§4の3入力の全decisionについて、各decisionで
新しいPolicy instanceを作り、次の記録のSHA-256を比較した（Python backend）。

- 0004参照版・0004 Belief版：`choose_action()`（`legal_actions`を逆順にした場合も）と例外（型・message）
- 合成版：上記に加え`choose_action_with_analysis()`（Action・analysis・例外）
- Champion `PlacementAwareSpeedCallPolicy`：合成版同士入力（892件）の`choose_action_with_analysis()`

| 入力 | 0004参照版 | 0004 Belief版 | 合成版 |
| --- | --- | --- | --- |
| 0004参照同士 | `604de719…45141d` | `7056b56a…a599ea` | `85dc1bb9…d6a472` |
| 合成版同士 | `3eca16c4…166839` | `4bd5a433…479141` | `3681254d…ce53c1` |
| Champion同士 | `c6a97278…017f12` | `c6a97278…017f12` | `ea094467…3b5a054` |

B・Nのすべての記録が一致した。Championの記録も一致した（`2437dcdf…07a59a`）。

**Rust backend**：この作業環境ではcrates.ioへ到達できずnative拡張をbuildできなかったため、Rust backendでの
固定入力の再生は行っていない。Rust選択下の一致はCIの`native-backend` job（Rust選択でfull suite、
新候補のfocused testを含む）で確認する。

### 5.2 計測条件

| 項目 | 値 |
| --- | --- |
| 比較 | 合成版（C）と新候補（N）、同じsource（本branch） |
| backend | Python（`LISJONG_SHANTEN_BACKEND`未設定）。Rustは§5.1の理由で未計測 |
| 環境 | CPython 3.14.0rc2（Clang 20.1.4）、Linux、Intel Xeon @ 2.10GHz（2 logical CPU）、1 process、profilerなし |
| 手順 | 1 process 8 pass、passごとにPolicy instanceを作り直し、pass 0をcold、以降7 passをwarmとする。各processのwarm pass合計の中央値をとり、process間の中央値を代表値とする。順序は`C N N C C N` |
| 入力 | 合成版同士入力（§4）から、合法手が打牌だけで合成版の分類がPUSHのdecision 338件（PUSH判断の時間）。Policy全体は同入力の全892件（1 process 4 pass、順序`C N N C`） |

### 5.3 結果

| 対象 | 合成版 [ms] | 新候補 [ms] | うちBelief導出 [ms] |
| --- | --- | --- | --- |
| PUSH打牌338件（warm pass合計） | **776.3**（776.3 / 735.1 / 794.9） | **846.0**（842.8 / 846.0 / 866.2） | 約88–93 |
| Policy全体892件（warm pass合計） | 8,060.1 / 8,019.4 | 8,275.5 / 8,033.1 | 約92–97 |

- PUSH打牌では1 decisionあたり約2.30 ms → 約2.50 ms（+9%）。増分は`_belief_paijia_counts()`の推定器と
  残余導出（1 decisionあたり約0.27 ms）でほぼ説明できる。
- Policy全体では非PUSH打牌のChampion評価が支配的で、差（約1–3%）はprocess間のばらつきと同程度である。
- 対局全体の時間ではなく、同じdecision列の再生によるPolicy計算の比較である。速度・強さの改善閾値は設けない。

raw JSON・計測scriptはlocal artifact（作業環境のscratchpad `m230/`：`capture.py`、`compare.py`、
`champion_equiv.py`、`push_subset.py`）で、repositoryへはcommitしない。

## 6. 検証

- `tests/test_placement_aware_speed_call_kobalab_0004_belief_paijia_discard_policy.py`：
  別classとexport・identity機構なし・推定器記録、非PUSH分岐（共通現物FOLD・全候補fallback・mechanism防御・
  オーラストップFOLD制限）と立直・鳴き判断が合成版と一致しBelief導出を行わないこと、PUSH fixture（門前・
  他家リーチ下・宣言牌・赤5・副露手・副露直後・全候補向聴悪化fallback）で受入入力が実残り枚数のままpaijia入力
  だけが残余になること、赤5の別axis、未見枚数・推定器の1回導出、未見0の牌種、保存則違反と契約外入力の拒否、
  共通倍率（他家期待0）で全fixtureが合成版と一致すること、一様推定器の量子化による同点崩れ、非一様な合成
  Beliefで受入同点の候補のpaijia順位と最終Actionが変わること、route制限外を返さないこと、traced / `legal_actions`
  順序の一致、手中にない牌の拒否。
- `tests/test_kobalab_0004_reference_policy.py`：Belief経路の移動に合わせ、推定器のpatch先を共通moduleへ変更
  （期待値は不変）。`tools/profile_kobalab_0004.py`のwrap対象も共通moduleへ変更。
- CIはPython backendのfull suiteと、`native-backend` jobでRust選択下のfull suiteを実行する。

## 7. 限界と後続

- 現行の条件付き一様推定器の残余は丸めを除き未見枚数の共通倍率で、paijiaの順位を保つ。差は量子化による
  同点崩れ（§4で0.25%）に限られ、この接続だけで戦略的改善は期待しない。将来の非一様推定器を接続する準備である。
- 後続の強度評価では、まず合成版との比較でBelief接続の効果を切り分ける。鳴き・守備を持つため、棒聴即リー
  protocolへそのまま投入せず適合性を確認する。
- 対象外：受入枚数のBelief化（別Issue）、非一様推定器の開発、Champion/default置換、Rust化範囲の拡大、
  Arena登録・対局評価、seed予約、AWS実行、RiichiLab配備。
