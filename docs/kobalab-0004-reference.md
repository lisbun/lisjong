# kobalab 0004 tile-efficiency reference Policy

Issue: [lisbun/lisjong#211](https://github.com/lisbun/lisjong/issues/211)

## 位置付け

`Kobalab0004ReferencePolicy`（`lisjong.policies.kobalab_0004_reference`）は、
公開されている kobalab/majiang-ai の legacy 思考ルーチン 0004 の仕様を、lisjong の
`DecisionContext -> InternalAction` 契約と既存評価器の上で独立実装した
deterministic reference Policy である。

```text
identity           kobalab-0004-tile-efficiency-reference-v1
reference source   kobalab/majiang-ai legacy 0004
role               external documented reference / pure-offense benchmark arm
                   (not a Champion candidate)
RiichiLab「牌効率くん」との完全同一性   not established
```

RiichiLab の「牌効率くん」（単純牌効率 / 聴牌即リー / 鳴きなし / オリなし）は 0004 と
説明上よく似ているが、同一実装であることは確認していない。本Policyは RiichiLab bot の
clone ではなく、公開 source で仕様を検証できる kobalab 0004 reference として扱う。
upstream とのfixture一致も RiichiLab bot との同一性の証明には使わない。

## Upstream と attribution

| 項目 | 値 |
| --- | --- |
| repository | <https://github.com/kobalab/majiang-ai> |
| commit | `e75a9720a12b84c03e6c61c3960c1844b8982eb4`（対象ファイルの最終変更は `0a3019b3b414501115ffd9e6854ce325d1330bd1`） |
| files | [`legacy/player-0004.js`](https://github.com/kobalab/majiang-ai/blob/e75a9720a12b84c03e6c61c3960c1844b8982eb4/legacy/player-0004.js), [`legacy/suanpai-0004.js`](https://github.com/kobalab/majiang-ai/blob/e75a9720a12b84c03e6c61c3960c1844b8982eb4/legacy/suanpai-0004.js) |
| core | `@kobalab/majiang-core` **1.4.1**（tag `v1.4.1`, `d32e0f8957ff1961ee77dce67f0727722c091f68`）。majiang-ai の `^1.3.4` 範囲ではなく exact version を参照仕様とする |
| rules | `Majiang.rule()` default（赤牌 各色1枚、喰い替えなし、途中流局あり、ノーテン宣言なし、ツモ番なしリーチなし、リーチ後暗槓許可レベル 2） |
| license | MIT License, Copyright (c) Satoshi Kobayashi |

lisjong は upstream の JS package、Node.js、majiang-core を runtime dependency に
しない。source code は逐語移植せず、式と判断順序を lisjong の型と既存評価器で再実装した。

## 判断順序

sourceの `action_zimo()` / `action_dapai()` / `action_gang()` に対応する。

| 局面 | source | lisjong |
| --- | --- | --- |
| 和了 | `select_hule()` → `allow_hule()` | 合法な `TsumoAction` / `RonAction` を選ぶ。合法性は `legal_actions` をauthorityとする |
| 暗槓へのロン | `select_hule()` が `/^[mpsz]\d{4}$/` の槓を拒否 | target が同じ牌種の `ANKAN` を公開している `RonAction` は選ばない（同牌種4枚が暗槓内にあるため、この判定は snapshot から一意） |
| 九種九牌 | `xiangting >= 4` かつ `allow_pingju()` | `KyuushuKyuuhaiAction` が合法かつ14枚向聴数 >= 4 |
| 槓 | `get_gang_mianzi()` 列挙順で最初の `after == before` | `AnkanAction` / `KakanAction` を `m -> p -> s -> z`、rank昇順に並べ、槓後向聴数が槓前と**等しい**最初の候補（`<=` ではない） |
| 打牌 | `select_dapai()` | 下記の0004牌効率選択 |
| 立直 | 打牌選択後 `allow_lizhi(shoupai, dapai)` | 選んだ打牌の後に聴牌かつ和了牌（手中4枚でない改善牌）がある場合、合法な `RiichiAction` を返す。二段階立直の宣言牌 decision では同じ選択規則を再適用し、同じ打牌が選ばれる |
| 鳴き | `select_fulou()` は何も返さない | Chi / Pon / Daiminkan は選ばず `PassAction` |
| 守備 | なし | なし（他家立直でも挙動を変えない） |
| 流局時 `select_daopai()` | `allow_no_daopai()`。default rule の「ノーテン宣言あり: false」では常に false | lisjong の `InternalAction` に対応する decision がなく適用外 |

## 打牌選択

1. 打牌前14枚の向聴数 `n` を求める（`calculate_shanten()` は14 / 11 / 8枚を扱える）。
2. 候補を source の `get_dapai().reverse().sort(paijia)` と同じ順で評価する。
   - `paijia` 昇順（安定sort）。
   - 同値は `get_dapai()` 列挙の逆順: ツモ切り → 字牌 7..1 → 索子 9..1 → 筒子 → 萬子。
     5 は source が赤5 → 通常5 の順に列挙するため、逆順で通常5 → 赤5。
   - `legal_actions` の入力順には依存しない。
3. 打牌後向聴数が `n` を超える候補は除外する。
4. 残りの候補で ukeire（改善牌ごとの実残り枚数の合計）を求め、**厳密に大きい**場合だけ
   選択を更新する。受入・paijia が同値なら評価順で先の候補が残る。
5. 全候補が除外された場合は評価順の最初の候補を返す（source の初期値 fallback）。
   通常は14枚向聴数と最良打牌後の向聴数が一致するため到達しない。

### 実残り枚数

source の `SuanPai` は event 累積で、初期手牌・自分のツモ・他家の打牌・他家の副露の
手牌由来牌・他家の槓・ドラ表示牌を減算する。lisjong は判断時点の `PolicyInput` から
`derive_remaining_tile_inventory()` で同じ値を full recomputation する。

- 自手 `concealed_tiles`、全員の捨て牌（鳴かれた牌を含む）、副露の手牌由来牌
  （`called_tile` を1枚だけ除外、ANKAN は4枚）、ドラ表示牌を既知とする。
- 鳴かれた牌は捨て牌側だけで数え、河と副露で二重に減算しない。加槓の追加牌、
  暗槓、新ドラ表示牌も既知枚数へ入る。
- 自分の過去の打牌・仮に捨てる牌も既知のまま扱う（全候補で同じ判断時点の count）。
- 五の残り枚数は赤五を含む。赤五の残りは別axisで保持し、受入枚数では赤五を二重加算しない。
- 判断を左右する event 履歴を Policy instance に保持しない。

### paijia

`suanpai-0004.js` の `paijia()` の式をそのまま再現する（捨てる牌の将来価値が低い方を
先に評価し、搭子形成力・ドラ等の価値が高い牌を残す）。

- 数牌: 左 / 中 / 右の搭子形成枚数 `min(num[n±k], ...)` を ±2 の範囲で重み付き合計。
  1 / 9 等の端は範囲外を 0 とする。
- 赤5 が未見なら 3..7 に赤5関連 bonus、赤5 自体は 2 倍。
- 字牌: 未見枚数、場風・自風・三元牌でそれぞれ 2 倍。
- dora: 公開ドラ表示牌ごとに weight を 2 倍。対象牌自身の weight は最後にもう一度
  掛けるため、ドラの字牌は weight が二乗になる。

このtie-breakは lisjong の second-step ukeire とは別semanticであり、同一視しない。

## 同値性の確認と分類済み差分

### 向聴数・改善牌

lisjong の `calculate_shanten()` を再利用する。通常形・七対子（同一牌種4枚は1対子）・
国士無双・槓後（確定面子あり、七対子 / 国士なし）の定義は majiang-core と一致する。
ただし次の差がある。

- **向聴定義差（evaluator差）**: majiang-core の `xiangting()` は同一牌種の5枚目を
  必要とする分解（例: `123456789p1111s` を 1s 単騎の聴牌とする）を数えるが、lisjong の
  exact shanten は数えない。差は自手に同一牌種を4枚持つ手牌でだけ生じ、lisjong 側が +1 になる。
- 開発時に majiang-core の式を Python で書き起こした一時oracle（repositoryには含めない）で
  確認した結果、random hand 200,000件で差は0件、4枚持ちを必ず含む手牌に絞った
  sampling で見つかった差はすべて lisjong が +1 の形だった。4枚持ちを含む14枚手牌
  17,592件で打牌選択を比較しても、選ばれる牌種の差は0件だった。この差は reference の目的上
  許容し、differential fixture では該当手牌を中間値の厳密比較から除外して、差がこの形に
  限られることを検証する。
- 聴牌判定だけで和了牌がない（待ち牌を自手に4枚持つ）場合、source の `allow_lizhi()` は
  `tingpai().length > 0` で立直しない。lisjong でも同じく立直しない。

### 表現差

- source の `get_dapai()` は、ツモ牌が唯一の通常5で赤5も持つ場合、実在しない手出し通常5を
  列挙する。この候補は同じ牌種のツモ切りより評価順が後で必ず同点になるため、選択には
  影響しない。lisjong の合法手には現れない。
- ツモ切り / 手出しは `DiscardAction.tsumogiri` で区別する。
- 合法性の authority は `DecisionContext.legal_actions` である。lisjong 側の合法手集合が
  upstream と異なる（喰い替え、リーチ後暗槓、流局条件等）場合、Policy は黙って補正せず、
  与えられた合法手の中で上記の規則を適用する。

## Differential validation

- manual source-derived golden fixture: `tests/test_kobalab_0004_reference_policy.py`
- upstream differential: `tests/test_kobalab_0004_upstream_differential.py` が
  `tests/fixtures/kobalab_0004_upstream.json` を読み、向聴数・改善牌、SuanPai 残り枚数、
  paijia、候補ごとの ukeire、評価順、最終 decision（和了 / 九種九牌 / 槓 / 立直 / 打牌）、
  暗槓・加槓へのロン判定を照合する。fixture が無い場合は skip する。
- fixture 生成: `tools/kobalab_0004_reference/generate_upstream_fixture.js`
  （pinned majiang-ai commit と majiang-core 1.4.1 を `npm install` し、seed固定の
  0004 同士の対局から sampling した decision と、crafted scenario、random 手牌の向聴数を出力）。
  Node.js は development-only で、CI では実行しない。

```text
cd tools/kobalab_0004_reference
npm install
node generate_upstream_fixture.js > ../../tests/fixtures/kobalab_0004_upstream.json
```

## 現行 lisjong 牌効率 lineage との差

```text
kobalab 0004
  shanten (majiang-core式)
  -> actual remaining ukeire
  -> paijia tie-break
  -> source逆順

lisjong UkeirePolicy / TwoStep / semantic envelope lineage
  exact shanten
  -> current ukeire (4 - visible count)
  -> second-step score
  -> stable / residual tie-break
```

| 観点 | kobalab 0004 | lisjong canonical lineage |
| --- | --- | --- |
| 向聴数 | 5枚目分解を数える式 | exact（5枚目分解を数えない） |
| 候補filter | 14枚向聴数を超えない候補 | 最小打牌後向聴数の候補 |
| current ukeire | 改善牌の未見枚数合計（手中4枚の牌種は除外） | 同じ定義（`UkeirePolicy`） |
| visible accounting | event累積 `SuanPai` | snapshot から再構成（同値） |
| 同点処理 | `paijia` → source逆順 | second-step ukeire → `tile_sort_key` 等の stable order |
| 赤5 | paijia で 2 倍・周辺 bonus（赤を残す） | ukeire 上は同一牌種。value-aware 系は別途評価 |
| ドラ | paijia weight | value-aware 系で評価 |
| 字牌 | 未見枚数と役牌倍率 | second-step の構造評価 |
| 聴牌時 | 待ち枚数最大、同点は paijia | 待ち枚数最大、同点は second-step / stable order |
| 立直 | 聴牌即リー（和了牌がある場合） | Policy ごとに異なる |
| 槓 | 向聴数不変なら常に槓 | Policy ごとに異なる |
| 九種九牌 | 向聴数 >= 4 なら宣言 | Policy ごとに異なる |
| 鳴き / 守備 | なし | Policy ごとに異なる |

どちらが正しい / 強いかは source inspection だけで結論しない。比較は Arena の
pure-offense benchmark（lisbun/lisjong-arena#389 系）で descriptive / development
benchmark として行い、Champion promotion や overall strength claim には使わない。
