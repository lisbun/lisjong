# 通常ツモDLVAT型補正のV1・p・単位補正

Issue #391で実装した`lisjong.draw_correction`の契約。設計の正本は#390
（[改訂設計](https://github.com/lisbun/lisjong/issues/390#issuecomment-7238939804)、
[focused re-review](https://github.com/lisbun/lisjong/issues/390#issuecomment-7239075099)）。

lisjongは、通常ツモ1回の補正について**価値と補正のcanonical semantics**だけを所有する。
MLに依存しない純粋関数で、Arena・engine・ML frameworkをimportしない。
online Policyへの経路は持たない。

| 責務 | 所有 |
|---|---|
| allowlistの射影型、p、V1、単位補正（κ = 1）、V1の特徴定義 | lisjong（本module） |
| `pre_draw`からの射影、行のjoin・集約、rank contrast、κの推定・凍結・適用 | Arena（lisbun/lisjong-arena#598） |
| 未公開牌の置換不変性fixture（pの成立条件） | Arena（既存pinのintegration）又はengine |

## カテゴリ軸

37カテゴリ。物理copy IDは持たない。

| index | 内容 | 標準在庫 |
|---|---|---|
| 0..33 | 34牌種（`belief.canonical_axes`と同じ順: 0..8 萬子、9..17 筒子、18..26 索子、27..33 字牌）の**赤でない**牌 | 4（4 / 13 / 22 の五は3） |
| 34 / 35 / 36 | 赤5m / 赤5p / 赤5s | 1 |

基礎牌種ごとの合計（通常の五 + 赤五を含む）は4、全体で136枚。

## 射影型（allowlist）

どちらもfrozen dataclassで、`from_mapping(dict)`はfieldと**完全に一致するkey集合**だけを受け取る。
identity（seed・placement・draw index等）、実現牌、post digest、他の行は型として渡せない
（未知のkeyは`ValueError`、未知のkeyword引数は`TypeError`）。値はJSON-nativeなint・list
（tupleも可）で、`bool`・`float`は拒否する。

### `DrawProbabilityProjection`（p用）

| field | 内容 |
|---|---|
| `hand_counts` | 4席分（席順）の純手牌の37カテゴリ枚数 |
| `meld_counts` | 4席分の副露（鳴かれた牌を含む）の37カテゴリ枚数 |
| `uncalled_discard_counts` | 4席分の**鳴かれていない**河の37カテゴリ枚数 |
| `revealed_dora_indicator_counts` | 公開済みのドラ表示牌の37カテゴリ枚数（1..5枚） |
| `live_tiles_remaining` | 山の残り枚数（≥ 1。Nの照合用） |

検査（不一致はfail closed）:

- 各カテゴリの枚数が0..標準在庫で、既知牌の合計も標準在庫を超えない（赤五の重複・通常の五の4枚目を含む）
- 各席の純手牌がツモ直前の1 / 4 / 7 / 10 / 13枚
- N = 136 − 既知牌の合計 が `live_tiles_remaining + 14 − 公開表示牌数` と一致する

### `ShantenValueProjection`（V用）

`drawer_seat`（0..3）と`drawer_hand_counts`（drawerのツモ直前の純手牌、1 / 4 / 7 / 10 / 13枚）だけ。

## API

| 関数 | 返り値 |
|---|---|
| `draw_probabilities(p用)` | support（n_a > 0）の`{カテゴリ: Fraction(n_a, N)}`、カテゴリ昇順 |
| `shanten_values(V用, a)` | 候補aのV1の4席分の値（`Fraction`） |
| `unit_corrections_from_values(p, values, 実現)` | 与えたVによる4席分の単位補正（zero-V等の検算用の核） |
| `unit_draw_corrections(p用, V用, 実現)` | V1による4席分の単位補正 |
| `v1_definition()` | V1の特徴定義（JSON-nativeなdict、version 1） |

### p

p(a) = n_a / N。n_a = 標準在庫 − 既知牌、N = Σ n_a。n_a = 0のカテゴリはsupportに入れず、Vも評価しない。

### V1（κ = 1の単位値）

g(a) = drawerの純手牌 + aの向聴数（`calculate_shanten`、和了形は−1）。評価する手牌は2 / 5 / 8 / 11 / 14枚に
限る。赤五は向聴計算では通常の五と合算される（`calculate_shanten`の契約）。

- drawer d: V_d(a) = +g(a)
- 他の3席: V_o(a) = −g(a) / 3

**符号:** gは小さいほど良く、順位（1..4）も小さいほど良い。良いツモ（gが下がる）は
drawerの補正を負にし、順位の期待値を下げる向きと揃う。4席の和は常に0。

### 単位補正

1. supportとpを求める
2. **support全体で**Vを計算する（実現牌に合わせて候補を選ばない）
3. そのあとで実現カテゴリをlookupする。supportにない実現カテゴリはfail closed
4. c_s = V_s(a_realized) − Σ_a p(a)·V_s(a) を4席分、`fractions.Fraction`で返す

κは掛けない。V用射影の純手牌がp用射影のdrawerの手牌と一致しなければfail closedする。
構成上 Σ_a p(a)·c_s(a) = 0 と Σ_s c_s = 0 が厳密に成り立つ（testで固定）。

## #390の仮定（このAPIは証明しない）

- **山の理想化:** 局ごとの山を「局ごとに独立な一様順列」と理想化する。seedの事前予約は厳密な一様性を保証しない
- **filtration:** E[c | F_t] = 0（martingale）は**各placementを単独で見た履歴**に対してだけ主張する
- **未公開牌:** 未公開の牌の値が非終局の合法手・選択・公開情報・継続条件に影響しないこと。置換不変性fixtureが確認できなければpは NOT-ESTABLISHED（Arena側の完了条件）
- **停止時刻:** E[N_h] < ∞ と、分散比の議論に要る E[N_h²] < ∞ は未証明の仮定。正式な不偏性を証明済みとは表現しない

## 対象外

κの推定・適用、join・集約、未公開牌の置換不変性fixture、`final_points`の符号・単位（後続）、V2、online Policyへの接続。
