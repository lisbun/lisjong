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
| 最適化後・Belief対応版 | 本PRのbranch（commit別は§3・§4） |
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
| 推定器 | `_estimate_concealed_hand_belief()`（`estimate_conditional_uniform_hand_belief()`を呼ぶ）。**将来の推定器の接続境界**：同じ`PolicyInput`から`ConcealedHandBelief`を返す関数へ差し替え、別のPolicy identityを与える。汎用Belief frameworkは作っていない |
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
3. 一様推定器と`derive_remaining_tile_inventory()`の二重導出の解消（推定器APIへ導出済みinventoryを
   渡せるようにする等）。共有belief基盤の変更になるため別Issueで扱う。
