# kobalab 0004 正規版採用準備：profile・同値最適化・Belief対応（Issue #218）

Issue: [lisbun/lisjong#218](https://github.com/lisbun/lisjong/issues/218)

`Kobalab0004ReferencePolicy`（[#211](https://github.com/lisbun/lisjong/issues/211)、
[参照仕様](kobalab-0004-reference.md)）を、Rust向聴数backend有効時に計測し、
同値な最適化を行った記録と、paijiaの入力を既存Beliefへ接続した対応版
`Kobalab0004BeliefPaijiaPolicy`の仕様・検証・採用準備の判断をまとめる。
固定decisionでの性能・同値性確認であり、強さの評価ではない。

## 1. 計測条件（固定）

| 項目 | 値 |
| --- | --- |
| 基準（最適化前） | lisjong `main` `2553c1b9f22545bb2fcb914adce1879d15cdc58d` |
| 最適化後・Belief対応版 | 本PR branchの`2d5a384`（同値最適化は`3324887` + `79569cd`、Belief対応は`2d5a384`。以降の変更は文書のみ） |
| Rust backend | `native/`を`2553c1b`から`LISJONG_NATIVE_SOURCE_REVISION`付きでlocal build（`SOURCE_REVISION = 2553c1b…`、`.pyd` SHA-256 `70acd5be7261e8ba93b1403d0705bf9d2ade187f468c57705d8fe8194446b0c3`）。本Issueでnative sourceは変更していない |
| backend選択 | `LISJONG_SHANTEN_BACKEND=rust`。native core呼び出し回数（`standard_shanten_call_count()`）でRust pathの実行を確認 |
| Python | CPython 3.14.7（MSC v.1944, 64 bit）、通常版 |
| OS / CPU | Windows 11 Home 10.0.26200 / Intel64 Family 6 Model 189（8 logical CPU） |
| worker | 1 process、profilerなし（timing） |

**入力**：[#213](rust-backend-prototype.md)で採取した固定decision列（local artifact、repositoryへはcommitしない）。

| 入力 | decisions | bytes SHA-256 | 用途 |
| --- | ---: | --- | --- |
| 0004参照同士の半荘（seed 0, `4p-red-half`） | 630 | `f5782dc3…d287a8` | 計測・同値性・Belief差分 |
| two-step同士の半荘 | 684 | `31d0448e…b8a4cf` | 同値性・Belief差分（0004で再生） |
| champion同士の半荘（副露手を含む） | 708 | `275ca281…18a91b` | 同値性・Belief差分（0004で再生） |

主対象の0004入力のdecision構成（`legal_actions`だけで分類）：打牌483、打牌+立直可19、
打牌+槓可1、和了可10、応答（Pass等）117。純手牌は14枚505、13枚124、11枚1。0004は鳴かないため、
副露後の手牌（8 / 10 / 11枚）はchampion入力で補った。

**手順（計測前に固定）**：`tools/profile_kobalab_0004.py timing`で1 processあたり7 pass
（pass 0 = process起動直後のcold、pass 1..6 = warm。passごとにPolicy instanceを作り直す）を
3 process実行し、warm pass合計の中央値を各processで取り、その3値の中央値を代表値とする。
Policyのcacheはdecision-localだけで、cross-decision cacheはない（Rust backendには値のcacheがない）。
内訳は`breakdown`（対象関数をwrapし、呼び出し回数・inclusive・self時間を3 pass分集計）で求め、
wrapのoverheadを含むため速度比較には使わない。

raw JSON、補助script、入力のcopyはlocal artifact root
`C:\Dev\lisjong-artifacts\issue-218-kobalab-0004\`（`results/`、`scripts/`、`inputs/`）にある。

## 2. 現行版のボトルネック（最適化前）

Policy判断全体（Rust、profilerなし）：warm pass合計 **536.0 ms**（3 process: 536.0 / 529.6 / 542.9）、
cold pass 543–546 ms。decision種類別（warm、中央値）：

| 種類 | 件数 | mean [ms] | p95 [ms] | max [ms] |
| --- | ---: | ---: | ---: | ---: |
| 打牌 | 483 | 1.082 | 2.015 | 2.978 |
| 打牌+立直可 | 19 | 0.625 | 0.844 | 0.844 |
| 打牌+槓可 | 1 | 0.016 | — | — |
| 応答 | 117 | 0.004 | 0.006 | 0.013 |
| 和了可 | 10 | 0.002 | 0.004 | 0.004 |

打牌decisionが時間の約98%を占める。遅い局面は多数の候補が同じ向聴数に残る打牌（最大約2.5 ms）。

内訳（`breakdown`、1 passあたり、wrap overhead込み）：

| 対象 | 呼び出し | self時間の割合 |
| --- | ---: | ---: |
| `calculate_shanten`（Tile入口、native呼び出しを含む） | 101,312 | 62% |
| `_improving_tile_types`（Tile列の組み立て等、self） | 2,730 | 19% |
| 未見枚数の再構成（`derive_remaining_tile_inventory`） | 502 | 7% |
| paijia計算 | 5,138 | 6% |
| 候補比較・評価順・その他 | — | 6% |

`calculate_shanten`をさらに分けると（同じ入力列の再計測）、1 callあたり約3.4–3.5 µsのうち
native core（`_shanten_from_valid_counts`、PyO3呼び出しを含む）は約0.36 µs（全入力で約37 ms / pass、
判断全体の約7%）で、残りはPython側のTile検証と34-count変換だった。
したがって律速はRust内部ではなく、候補ごと・仮想ツモごとに13 / 14枚のTile列を作り直して
変換する構造であり、Rust内部の調査・Rust化の拡大には進まなかった。

## 3. 同値最適化

| 候補 | 判断 | 理由 |
| --- | --- | --- |
| 打牌後向聴数・改善牌を34-countで求め、同じ牌種の手出し / ツモ切り、赤5 / 通常5で共有 | **採用** | Tile変換を1 decision 1回にし、以降はcount ±1だけで`calculate_shanten_from_canonical_counts()`を呼ぶ。native呼び出しは約6%減（共有分） |
| 打牌後向聴数・改善牌を立直判定で再利用 | **採用** | 同じ構造評価instanceを渡すだけで追加の複雑さがない（立直可のdecisionは少なく効果は小さい） |
| paijiaの式を34-index配列上で計算（ドラweightを1回だけ集計） | **採用** | Belief対応の入力分離と同時に行い、`TileType`生成を除いた |
| 未見枚数0の牌種への仮想ツモ評価を省く | **見送り** | 全仮想ツモ89,624件のうち該当は2.6%。改善牌は立直の和了牌判定にも使うため（形としての改善牌）、分離の複雑さと立直判定への流用リスクに見合わない |
| 候補評価をまとめてRust側へ渡す / paijiaのRust化 | **見送り** | 最適化後のnative core呼び出しは判断全体の約3割で、残りはPythonの軽いloop・未見枚数再構成・paijia。Rust境界を広げる根拠がない |
| 未見枚数再構成の高速化 | **見送り（本Issue外）** | 最適化後に約2割を占めるが、共有のbelief基盤（`derive_remaining_tile_inventory`）の変更になり他Policyへ影響する |

保持した条件：exact shantenの定義（5枚目を要する分解を数えない）、赤牌・手出し・ツモ切りの区別
（`legal_actions`上の候補はそのまま評価順に並べ、構造評価だけを共有）、paijiaと同点時の評価順、
立直・槓・九種九牌・和了の判断、`legal_actions`をauthorityとする契約、手中にない牌の打牌候補を
`Kobalab0004ReferencePolicyError`で拒否するfail-closed。構造評価のcacheは牌姿だけに依存し、
公開枚数・ドラ・Beliefに依存する値（ukeire・paijia）とは分け、decision-localで破棄する。

**同値性**：

- 単体test：共有構造評価がTile単位の定義（打牌後`calculate_shanten` / `_improving_tile_types`）と、
  固定seedの全純手牌枚数（14 / 11 / 8 / 5 / 2枚）、4枚持ち、赤5、七対子・国士寄りで一致。
  upstream differential testでも全候補の中間値を照合する。Python / Rust両backendでpass。
- 固定decision：`main`の実装と、3入力（計2,022 decision、打牌候補16,833件）で候補ごとの
  paijia・打牌後向聴数・ukeire・評価順・最終行動がすべて一致。0004入力では記録済みactionとも不一致0。

**性能**（Rust、profilerなし、630 decision）：

| 版 | warm pass合計 [ms] | 打牌decision mean / p95 / max [ms] | native core呼び出し / pass |
| --- | ---: | --- | ---: |
| 最適化前（`2553c1b`） | 536.0 | 1.082 / 2.015 / 2.978 | 101,312 |
| 同値最適化後 | **121.3**（117.8 / 121.3 / 121.4） | 0.241 / 0.356 / 0.496 | 95,233 |

約77%短縮（約4.4倍）。参考としてPython backendでは（1 process）2,462 ms → 1,750 ms（約29%短縮）。
Python backendではcold passよりwarm passが遅くなる傾向が最適化前後で共通に見られた
（本Issueの変更とは独立。原因は調べていない）。

## 4. Belief対応版 `Kobalab0004BeliefPaijiaPolicy`

```text
identity           kobalab-0004-tile-efficiency-belief-paijia-v1
base               kobalab-0004-tile-efficiency-reference-v1 と同じ判断規則
changed            paijia の入力だけ（ukeire・向聴数・候補filter・立直・槓・九種九牌・和了は不変）
paijia input       未見枚数 − 他家3人の手牌内期待枚数（NonPlayerHiddenBelief、fixed-point raw）
belief_estimator   KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR
                   = conditional-uniform HandBelief（#65 / #68）
                     + opponent slots = 13 - 3 * public melds
                     -> derive_non_player_hidden_belief（#67）
role               正規版への採用準備の候補。Champion候補・default Policyではない
```

### 4.1 意味

- `NonPlayerHiddenBelief`は、未見枚数から他家3人のconcealed hand beliefを差し引いた残余
  （通常のツモ山・王牌・未開示の裏ドラ表示牌等を含む）である。残余内の配置が交換可能という仮定では、
  通常のツモ山への分配は全牌種共通の倍率になるため、paijiaの順位比較に残余期待枚数をそのまま使う。
  ツモ山専用の型や分解処理は新設していない。
- 現行の条件付き一様推定器では、残余期待枚数は丸め誤差を除いて未見枚数と共通比率
  （`(未見合計 − 他家slot) / 未見合計`）になる。この接続だけで打牌が改善するとは主張しない。
- 期待枚数を代入したpaijiaはヒューリスティックな牌価であり、厳密な牌価の期待値・ツモ確率・
  和了確率ではない。
- 参照版`Kobalab0004ReferencePolicy`は挙動・identity・参照仕様を維持し、比較基準として残す。

### 4.2 Beliefの生成・供給責務

現行`DecisionContext` / `PolicyInput`にはBelief fieldがなく、本Issueでも契約を拡張しない。

| 項目 | 責務・導出元 |
| --- | --- |
| 生成・供給 | Policy自身が、打牌候補を評価するdecisionごとに**同じ`PolicyInput`から1回だけ**導出する（候補ごとに再計算しない）。ukeireと同じsnapshotの`TileConservationResult`を`derive_non_player_hidden_belief()`へ渡す |
| 他家concealed slot数 | 公開情報だけから`13 − 3 × 副露・槓の数`（暗槓・加槓・大明槓も1面子）。自分の打牌decisionでは他家は手番外で常にこの枚数。Seatは`wind_for_seat(seat, dealer_seat)`で自風へ対応付け、自分のentryは0 |
| 推定器 | `_estimate_concealed_hand_belief()`（`estimate_conditional_uniform_hand_belief()`と同じ結果の内部経路`_estimate_from_conservation()`へ、同じdecisionで導出済みの未見枚数を渡す。§6）。**将来の推定器の接続境界**：同じ`PolicyInput`から`ConcealedHandBelief`を返す関数へ差し替え、別のPolicy identityを与える。汎用Belief frameworkは作っていない |
| 再現情報 | Policy classの`identity`と`belief_estimator`（推定器と設定の記述）。推定器・slot導出を変える場合は両方を変える |
| 不正・欠落入力 | 副露数が手牌で保持できる範囲を超える、推定器入力の不整合、Beliefが残余massを超える保存則違反はclampせず、`Kobalab0004ReferencePolicyError`でfail closedする。未見枚数の導出自体の不整合は参照版と同じく`derive_remaining_tile_inventory()`が拒否する |

情報境界：入力は`PolicyInput`（当該seatの観測可能範囲）だけで、対局環境の非公開手牌・山の実値は使わない。
moduleの依存はlisjongと標準libraryだけ（既存testで固定）。

### 4.3 数値表現

- 34牌種（5は赤5を含む）と赤5（3色）を同じ推定元・同じ尺度（`SCALE = 8192`のraw）で与え、
  途中で整数枚数へ丸め戻さない。
- paijiaの式はmin / max / 加算 / 正の整数倍だけなので、入力を共通倍率で拡大するとpaijiaも同じ倍率になる。
  正規化・除算を持ち込まず、0 massでもすべて0になるだけで例外にならない。
- 自手は`TileConservationResult`で既知として数え済みのため差し引かない。赤5は基本5牌の値に含まれる
  1枚分を別axisで持つだけで、二重加算しない（参照版の未見枚数と同じ構造）。

### 4.4 検証

単体test（`tests/test_kobalab_0004_reference_policy.py`）：

- 共通倍率（丸めなし）：任意の未見量を`SCALE`倍した入力でpaijiaが厳密に`SCALE`倍になる。
  他家massが0のBeliefではBelief版paijiaが参照版の`SCALE`倍になり、行動も一致する。
- 保存則：一様推定の残余が`未見 × SCALE − 他家3人の和`（34牌種・赤5とも）に一致し、自手を差し引かない。
- 非一様Belief：他家1人が6zを期待2枚持つBeliefでは6zの残余が1枚分になり、参照版が同点順で
  7zを切る局面で6zを切る（ukeireは不変）。
- 保存則違反（他家期待枚数 > 未見枚数）はfail closed。slot数の導出（副露・暗槓・親の位置）と
  不正な副露数のfail closed、立直・Pass等の判断が参照版と同じこと。

固定decision（3入力、打牌decision 1,601件）での一様推定器の量子化の影響：

| 項目 | 結果 |
| --- | --- |
| 参照版paijiaが異なる候補対の順位反転 | 82,095対中 **0** |
| 参照版paijiaが同点の候補対 | 4,786対中1,468対（31%）で1–数raw unitの差がつき同点が崩れた |
| paijiaの相対誤差（期待する共通比率に対して） | 最大 1.7 × 10⁻⁴ |
| 最終行動の差 | **3件**（0004入力 #298・#528、two-step入力 #120）。champion入力は0件 |

3件はいずれも、参照版でukeireとpaijiaが同点の候補（`2p`ツモ切り vs `3s`、`2z` vs `1z`、
`9s` vs `8p`）の間で、Belief版のpaijiaが1–2 raw unit異なったため評価順が入れ替わったもので、
**量子化による同点崩れ**に分類した。接続上の不具合（保存則違反、赤5の二重計上、順位反転）は見つからなかった。
一様版との完全行動一致は要件にしていない。

### 4.5 処理負担（Belief導出を含むend-to-end）

| 版 | warm pass合計 [ms]（Rust） | 打牌decision mean / p95 / max [ms] |
| --- | ---: | --- |
| 同値最適化後の参照版 | 121.3 | 0.241 / 0.356 / 0.496 |
| Belief対応版 | **236.3**（236.3 / 237.7 / 233.1） | 0.472 / 0.604 / 0.878 |

追加は打牌decisionあたり約0.23 msで、最適化前の参照版（536 ms）より速い。内訳では一様推定器が
追加分の大半を占める（推定器内部で同じ`PolicyInput`から未見枚数をもう一度導出する分を含む）。
`derive_non_player_hidden_belief()`の保存則検証とslot導出は追加分の約1割である。
Python backendでは1,750 ms → 1,914 ms（1 process）。

## 5. 採用準備の判断

- **同値最適化：採用**。中間値・最終行動の同値性を確認し、Rust有効時に約77%短縮した。
  参照版identityは`kobalab-0004-tile-efficiency-reference-v1`のまま。
- **Belief対応版：実装として採用準備が整った**。paijia計算と入力導出を分離し、既存Beliefを
  保存則検証込みで接続し、別identityと推定器の再現情報を持つ。量子化の影響は同点崩れだけで、
  追加costは許容範囲（最適化前の参照版より速い）。
- ただし一様推定器では参照版と意味上ほぼ同じ（共通比率）であり、**正規版のdefault採用・強さの評価は
  行っていない**。固定decisionの性能・同値性だけで強さの向上は主張しない。

後続Issue候補：

1. 非一様な推定器（河・立直・副露等を使うHandBelief推定）を接続した版を別identityで作り、
   参照版・一様Belief版とArenaのpure-offense benchmark（lisjong-arena#406系）で記述的に比較する。
   一様Belief版単独の比較は、差が量子化による同点崩れに限られるため優先度は低い。
2. 正規版としてdefault / Arena catalogへ登録するかの判断（参照版の単純な改名は行わない）。
3. ~~一様推定器と`derive_remaining_tile_inventory()`の二重導出の解消~~ →
   [#220](https://github.com/lisbun/lisjong/issues/220)で実施（§6）。

## 6. 未見枚数の共有（Issue #220）

Issue: [lisbun/lisjong#220](https://github.com/lisbun/lisjong/issues/220)。
推定能力・丸め・Policy identityは変えず、同じdecision内の未見枚数（`TileConservationResult`）の
二重導出だけを解消した。

### 6.1 重複の実測（基準 = `1263db5`、Rust、`breakdown`、1 passあたり）

| 対象 | 呼び出し | self [ms] |
| --- | ---: | ---: |
| Policy側の未見枚数導出（`_PublicCounts`） | 502 | 37.0 |
| 推定器内部の未見枚数の再導出（重複） | 502 | 34.9 |
| 推定器の固定小数点配分（`HandBelief`生成・検証を含む） | 502 | 54.4 |
| 推定器内の自手exact belief | 502 | 5.9 |
| 残余の導出・保存則検証（`derive_non_player_hidden_belief`） | 502 | 11.3 |
| 自手の34牌種count構築（`_DiscardStructures`初期化、self） | 502 | 5.1 |
| 参照版paijia入力の生成（Belief版で置き換えられる分を含む） | 1,004 | 2.3 |

wrapのoverheadを含むため比率の目安であり、速度比較には§6.3の`timing`を使う。

### 6.2 共有経路と整合性

- 推定器module（`conditional_uniform_hand_belief.py`）に内部関数`_estimate_from_conservation(policy_input,
  conservation, slots)`を設けた。公開API`estimate_conditional_uniform_hand_belief(policy_input, slots)`は
  未見枚数を導出してから同じ配分関数を呼ぶだけで、引数・結果・検証・例外の順序は変わらない。
  公開API（`__all__`）は増やしていない。
- **生成責務・有効期間**：`_choose_discard()`が判断ごとに作る`_PublicCounts`が、その判断の
  `PolicyInput`から未見枚数を1回だけ導出する。同じ`counts.conservation`をukeire、推定器、
  `derive_non_player_hidden_belief()`へ渡し、判断をまたいで保持しない（cacheなし）。
  `TileConservationResult`は不変のtupleで、共有先は変更しない。
- snapshotの照合（再導出して比較する等）は行わない。組み合わせは上記の内部経路でだけ作られ、
  照合のために同じinventoryを再導出すると共有の意味がなくなるため。
- 保存則検証（推定器のslot上限、`derive_non_player_hidden_belief()`のclampしない拒否）は維持した。

### 6.3 同値性と性能

**同値性**：比較元は変更前commit `1263db5`のsourceを別processで実行して生成した。
固定decision 3入力（§1の0004 / two-step / champion、計2,022 decision、打牌1,601件）と合成入力
（赤5、ポン・暗槓・加槓と鳴かれた捨て牌、親の位置4通り × 自席2通り、残余0、未見枚数の不整合、
公開APIの不正slot・不正型）について、公開API・Policy推定器境界の期待枚数raw・赤5 raw、残余raw、
両版のpaijia・評価順・最終行動・例外（型とmessage）を記録した。基準・変更後 × Rust / Python backend の
4通りで内容のSHA-256が一致した（`a3087c6189930d9d199c6f18bbda2dabdeeb3ce5669167521e341847f06519d7`）。
Issue #218で観測した量子化による同点崩れもそのまま維持されている。

**性能**（Rust、0004入力630 decision、1 process 7 pass、warm pass合計の中央値、各版3 process）。
版の実行順は`B N N B B N`（B = 基準、N = 変更後）と反転組`N B B N N B`で、計測前に固定した。
計測時の機械状態が#218時点と異なるため、絶対値は§4.5と直接比較しない。

| 版 | 組1 `B N N B B N` [ms] | 組2 `N B B N N B` [ms] |
| --- | --- | --- |
| Belief版 基準 | **301.3**（318.2 / 301.3 / 298.1） | **268.7**（268.6 / 268.7 / 269.8） |
| Belief版 変更後 | **276.4**（288.4 / 276.4 / 249.7） | **233.4**（228.9 / 237.5 / 233.4） |
| 参照版 基準 | 149.1（149.1 / 149.4 / 145.2） | 141.9（140.5 / 141.9 / 143.7） |
| 参照版 変更後 | 146.8（147.2 / 146.8 / 145.6） | 143.3（142.1 / 158.8 / 143.3） |

組1はprocess間の変動が大きく（時間経過とともに全体が速くなった）、組2は安定していた。
Belief版は組2で約35 ms / pass（約13%）短縮し、§6.1の重複導出分と一致する。参照版はコード経路が
変わらず、差は変動の範囲内。Python backendでは組1で基準2,074.4 ms・変更後1,998.6 msだが、
向聴数計算が支配的でpass間の変動（±150 ms程度）の範囲内である。
変更後の`breakdown`では推定器内部の未見枚数導出は0回、Policy側は打牌decisionごとに1回（502回）。

### 6.4 判断

- **未見枚数の共有：採用**。変更は内部関数1つと呼び出し側の引数追加だけで、公開APIと挙動を維持したまま
  重複分を除けた。
- **自手countの共有：見送り**。自手の34牌種化は未見枚数の導出、`exact_self_belief()`、
  `_DiscardStructures`でそれぞれ行うが、合計しても1 passあたり十数ms未満（wrap込み）で、共有には
  `exact_self_belief()`や`TileConservationResult`の入力型（Tile列 → count）を変える必要がある。
  型境界を崩すほどの効果はない。今後、非一様推定器等で自手countを使う処理が増え、`breakdown`で
  主要な割合を占めるようになった場合に再検討する。
- 残りの主な追加costは推定器の配分（`HandBelief`生成・検証を含む）で、推定器の実装自体の変更に
  なるため本Issueの範囲外とした。

raw JSON・補助script・基準sourceのcopyはlocal artifact root
`C:\Dev\lisjong-artifacts\issue-220-belief-inventory\`（`results/`、`scripts/`、`baseline-1263db5/`）にある。
Rust backendは§1と同じlocal build（`SOURCE_REVISION = 2553c1b…`）、CPython 3.14.7。

## 7. canonical有効牌集合のmask適用・集約（Issue #221）

Issue: [lisbun/lisjong#221](https://github.com/lisbun/lisjong/issues/221)

有効牌等の牌種集合を34牌種canonical axis上で表し、確定枚数（`TileConservationResult.remaining_tile_counts`）と
fixed-point期待枚数（`HandBelief` / `NonPlayerHiddenBelief`の`expected_count_raw`）へ同じ集合を適用・合計する
共通処理を`lisjong.belief.tile_type_set`に置き、0004の従来の受入集計へ統合した。新しい推定器・打牌規則は導入していない。

### 7.1 契約（正本はmodule docstring）

| 項目 | 契約 |
| --- | --- |
| axis | 既存の`tile_type_index()`（0..33）だけを使い、新しい牌種順序を作らない |
| 集合の表現 | canonical indexを昇順・重複なしに並べた`tuple[int, ...]`（既存0004の`improving_after()`と同じ）。赤 / 通常、手出し / ツモ切りのidentityは持たない |
| 空集合 | `()`。空判定は`len(...) == 0` / `> 0`で明示する（0004の立直判定も`bool(...)`から明示判定へ変更、結果は同じ） |
| 検証境界 | `tile_type_set()`が入力indexを検証（`int`かつ0..33、`bool`不可）し、重複を集合として除去して昇順にする唯一の境界。交差・mask・合計は正規化済みの集合を前提とし、呼び出しごとに再検証しない（`values`の長さ34だけを検証）。0004の`_DiscardStructures`は構築時にこの契約を満たす |
| 交差 | `intersect_tile_type_sets()`（昇順を保つ）。「未見枚数が正の改善牌」は構造上の集合を狭めず、別の交差として得る |
| mask出力 | `mask_tile_type_values()`は選択位置の値をそのまま、非選択位置を0にした同じaxisの34要素。選択値だけを詰めた配列ではない |
| 合計 | `sum_tile_type_values()`は34要素の中間配列を作らず、入力と同じ尺度の正確な整数和を返す。exact枚数ならexact枚数、rawなら`SCALE = 8192`倍のraw。float化・丸め戻し・1牌種上限（4枚 / `4 * SCALE`）での切り詰めをしない。bitwise ANDは値へ適用しない |
| 赤5 | 34牌種の5は赤5を含むため、受入合計へ赤5を追加加算しない。赤5の内訳は本Issueでは返さない |
| 派生値 | 元の`values` / beliefは変更しない。mask後の値は完全な手牌beliefではなく`HandBelief`等へ再構築しない。残余は王牌等を含み、mask後の期待枚数はツモ確率・和了確率ではない |

構造上の改善牌（向聴数を下げる手中4枚未満の牌種、未見枚数0を含む）の算出は既存の向聴数評価のままで、
mask演算で有効牌そのものを求めるわけではない。Belief由来の合計を打牌選択へ導入する変更は行っておらず、
参照版・Belief版とも受入は実残り枚数、paijia規則も従来どおりである。

### 7.2 表現候補の試作（micro計測）

基準`fafa6c7`（#220を含む`main`）のPolicyで0004入力（§1、630 decision）を再生し、向聴数を悪化させない
打牌候補2,717件の（改善牌、未見枚数、残余raw）を採取した（集合サイズ平均7.9、最大27）。各候補表現について
**集合の生成・検証・合計を含めて**同じ入力で計測した（15回、順序を交互に反転、中央値）。

| 候補 | 実経路（1配列）[ms] | 再利用例（未見枚数 + 残余rawの2配列）[ms] |
| --- | ---: | ---: |
| A: 既存index tuple + inline `sum(genexpr)`（基準） | 1.00 | 1.98 |
| B: index tuple + helper（`values`長の検証 + `sum([...])`） | 0.63 | 1.21 |
| C: int bitset（bit i = index i、範囲検証 + set bit走査） | 2.43 | 4.16 |
| D: 0/1の34要素配列（乗算で合計） | 4.03 | 8.09 |

全候補で2,717件すべての合計が基準Aと一致した。差は1 passあたり数msで、Policy全体（§7.3、Rustで約130 ms）に
比べて小さい。bitsetは交差・範囲検証がO(1)だが、Pythonでは合計時のbit走査が遅く、構造評価がすでにindexを
昇順に列挙するためtuple→bitset変換も追加になる。bitset・専用class・SIMD / Rust化は採用しない。
再利用例の数値は同じ集合を2配列へ適用した場合の参考であり、実Policyの高速化とは扱わない。

### 7.3 同値性と性能（Policy全体）

**同値性**：比較元は固定基準`fafa6c78673f278a797ff5bd29ea8ac4f33e136e`のsourceを別processで実行して生成した。
固定decision 3入力（§1、計2,022 decision、打牌trace 3,202件）と合成入力（赤5、待ちの未見0での立直、聴牌立直、
残余0、保存則違反）について、両Policyの最終行動（立直を含む）、評価順、候補ごとの打牌後向聴数・改善牌集合・
立直可否、`_choose_discard()`が評価順に比較したukeire列と選択結果、例外を記録した。ukeire列は、基準では
fixture側で基準の式（inline sum）を適用し、変更後では`sum_tile_type_values`をspyしてPolicy経路で実際に
計算された値を記録しており、新規helperで期待値を作っていない。基準・変更後 × Rust / Python backendの4通りで
内容のSHA-256が一致した（`1ff15cb07d2e721e28e9f482f15328123c039d032906b28417d0356956cd7543`）。
Belief版の量子化による同点崩れ（記録済み行動との差14件）も基準と同じである。

**性能**：§6.3と同じ手順（0004入力630 decision、1 process 7 pass、warm pass合計の中央値、各版3 process、
実行順`B N N B B N`と反転組`N B B N N B`を計測前に固定、B = 基準`fafa6c7`、N = 変更後）。
Rust backendは§1と同じlocal build（`SOURCE_REVISION = 2553c1b…`）、CPython 3.14.7。

| 版 | 組1 `B N N B B N` [ms] | 組2 `N B B N N B` [ms] |
| --- | --- | --- |
| 参照版 基準（Rust） | **127.6**（124.6 / 130.2 / 127.6） | **133.1**（129.8 / 133.1 / 139.7） |
| 参照版 変更後（Rust） | **127.3**（127.3 / 127.9 / 124.6） | **133.4**（135.1 / 133.4 / 128.2） |
| Belief版 基準（Rust） | **212.0**（212.0 / 225.4 / 210.2） | **224.2**（222.9 / 224.2 / 251.0） |
| Belief版 変更後（Rust） | **211.4**（209.3 / 211.4 / 217.5） | **215.4**（215.4 / 214.1 / 257.9） |
| 参照版 基準（Python） | **1,430.0**（1,430.0 / 1,465.0 / 1,420.0） | — |
| 参照版 変更後（Python） | **1,434.9**（1,434.9 / 1,438.8 / 1,406.9） | — |
| Belief版 基準（Python） | **1,502.9**（1,510.5 / 1,486.3 / 1,502.9） | — |
| Belief版 変更後（Python） | **1,487.5**（1,487.5 / 1,528.1 / 1,483.3） | — |

いずれも差はprocess間の変動の範囲内で、性能上の改善・悪化とは判断しない（受入集計は§7.2のとおり
1 passあたり約1 msで、Policy全体の1%未満）。

### 7.4 判断

- **採用**（既存index tuple表現を正本とする最小の共通処理）。集合の意味・空判定・尺度・赤5・派生値の契約を
  1か所に固定し、同じ集合を確定枚数とBelief rawへ同じ関数で適用できるようにした。表現変更による高速化は
  前提にしておらず、Policy全体の時間は変わらない。
- 0004では従来の受入集計を`sum_tile_type_values()`へ置き換え、立直判定を明示的な空判定にした。
  構造評価のcache（打牌後の牌種ごとの改善牌tuple）はそのまま使い、tuple→mask→tupleの往復変換は追加していない。
- 全Policyへの展開、Belief由来の合計の打牌選択への導入、赤5内訳の返却は範囲外とした。

raw JSON・補助script・基準sourceのcopyはlocal artifact root
`C:\Dev\lisjong-artifacts\issue-221-tile-type-set\`（`results/`、`scripts/`、`baseline-fafa6c7/`）にある。
