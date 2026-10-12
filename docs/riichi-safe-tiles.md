# リーチ者ごとの確定安全牌

Issue: [#302](https://github.com/lisbun/lisjong/issues/302)（設計: [riichi-passed-tile-genbutsu-design.md](riichi-passed-tile-genbutsu-design.md) 3節）

`lisjong.belief.riichi_safe_tiles.derive_riichi_safe_tiles()` は、判断時点の公開情報
（`PolicyInput`）だけから、リーチ者ごとの確定安全牌種を導く。

```python
from lisjong.belief.riichi_safe_tiles import derive_riichi_safe_tiles

safe_tiles = derive_riichi_safe_tiles(decision.input)
for player in safe_tiles.by_riichi_player:  # リーチ者ごと（席の順）
    player.riichi_seat
    player.safe_tile_types  # そのリーチ者にロンされない牌種
safe_tiles.safe_against_all_riichi_players  # すべてのリーチ者にロンされない牌種
```

この関数を使うPolicyはまだない。現Championの行動は変わらない。

## 意味

リーチ者 `r` に対して、次を返す。

```text
safe(r) = river(r) ∪ passed(r)

river(r)   r の捨て牌履歴にある牌種（鳴かれた牌・宣言牌を含む）
passed(r)  r のリーチ宣言牌より後に、r 以外の席（自席を含む）が切った牌の牌種
```

- 「確定安全」は、**`r` にロンされない**ことだけを意味する。
- **対象はリーチ者だけである。** `safe_against_all_riichi_players` は「全リーチ者に安全」で
  あって、「全員に安全」ではない。リーチしていない他家（副露者・ダマ）には当たりうる。
- リーチ者は、リーチ宣言済み（成立前を含む）の他家である。現Championの数え方と同じ。
- リーチ者がいないとき、`by_riichi_player` は空で、`safe_against_all_riichi_players` は
  空集合になる。「どの牌も安全」という意味にはしない。
- 牌種は `TileType` で比べる（赤5と通常の5は同じ）。順序は `Discard.order` だけで決まり、
  鳴きで手番が飛んでも影響しない。鳴かれた打牌も入る。槓の牌は打牌ではないので入らない。
- `r` のツモ、放銃以外の損失には何も言わない。

## 利用できる時点（呼び出し側の契約）

**自席の打牌判断からだけ呼ぶ。** この時点では、履歴にあるすべての打牌へのロン応答が
解決済みである。自席が直前の打牌を鳴いた後の打牌判断も含む（ロンは鳴きより優先されるので、
鳴けたならその打牌は通っている）。

他家の打牌への応答判断（鳴き・ロン・Pass）では呼ばない。履歴の最後の打牌へのロン応答が
未解決で、その牌が通ったかどうかを `Discard.order` だけからは区別できない。

誤用を検出する補助として、自席の手牌が打牌前の枚数（3n+2枚）でない入力を `ValueError` で
拒否する。**これは補助チェックであり、枚数が合うことは「ロン応答が解決済み」を保証しない。**
契約と併用する。枚数の検証だけに頼らない。

## 宣言牌の位置が分からないリーチ者

`PlayerPublicState.riichi_declaration_discard_order` が `None` のリーチ者は、`passed(r)` を
空とし、河だけを使う。位置を推測して広げない。`declaration_known` で区別できる。

- 単独リーチで位置が不明なら、結果は現行の共通現物（河）と同じ集合になる。
- 複数リーチで一部だけ不明な場合、`safe_against_all_riichi_players` は現行の共通現物と
  同じとは限らない。位置が分かっているリーチ者に通り、位置が不明なリーチ者の河にある
  牌種は入る（全リーチ者に対して確定安全なので、集合としては正しい）。

位置が `None` になってよいのは、宣言牌を切る前の `DECLARED` と、古い記録の明示的な
互換読み込みだけである（[policy-input-schema.md](policy-input-schema.md) の
「リーチ宣言牌位置」）。

## 前提とする規則

`passed(r)` が `r` にロンされないのは、次の2つの規則による。

1. **リーチ後の見逃しは、その局の終わりまでフリテンになる。**
2. **リーチ後は待ちが変わらない。** 打牌はツモ切りだけで、暗槓は待ちの牌種を変えない場合に
   だけ許される。

宣言牌より後に切られて `r` にロンされなかった牌種は、「ロン牌でなかった」（2により以後も
ロン牌にならない）か、「見逃された」（1により以後ロンできない）のどちらかである。

前提のidentityは `RIICHI_SAFE_TILE_RULE_ASSUMPTION`
（`riichi-missed-win-furiten-and-fixed-waits-v1`）である。利用する側（候補Policy）は、
この値を自分の不変な設定と、評価の環境identity（RuleSet・engineの固定version）へ結び付ける。
**確認していない環境へ自動的には適用しない。**

### 環境ごとの確認状況

| 環境 | 規則1（見逃しのフリテン継続） | 規則2（待ちの不変） | 状態 |
|---|---|---|---|
| project標準のRuleSet（lisjongの定義） | [riichi-deal-in-label.md](riichi-deal-in-label.md) の「リーチ後の見逃しフリテン」：宣言打牌より後に和了できた牌を和了しなければ、以後すべての待ちでロンできない | 同文書はリーチ後の手牌不変を前提に待ちを定める | 規則文書で確認 |
| lisjong-engine `91af75e` | `src/lisjong_engine/furiten.py`：`next_missed_ron_reason` は立直成立後の見逃しを `RIICHI` とし、`cleared_temporary_reason` は自分のツモ番でも `RIICHI` を解除しない | `docs/rules.md` の「RiichiAnkanPolicy」：送り槓禁止と待ち牌種類の不変は全policy共通の必須条件 | 規則文書と実装で確認 |
| RiichiEnv 0.4.10 | `riichienv-core` の合法手生成が `riichi_declared && missed_agari_riichi` をフリテンとして扱う（lisbun/lisjong-arena#247 の調査）。replayのforward stateがこのflagを更新しない既知の差があり、Arena側で補っている | 未確認 | **局面テストで確認中**（lisbun/lisjong-arena#493） |
| RiichiLab（対局server） | 未確認 | 未確認 | **未確認**。候補を稼働・評価する前に確認する |

RiichiEnvとRiichiLabの確認は、環境に依存する局面テストとしてlisjong-arena側に置く。
確認できた時点で、この表を更新する。

補助証拠として、Champion実戦423席（lisbun/lisjong-arena#489）では、通った牌を切った
710 打牌のうち、ロンされたものは0件だった。0件であることは規則の成立を証明しないので、
主な根拠にはしない。

## 情報境界

- 入力は `PolicyInput` だけである。他家の伏せ牌・山・将来の状態を使わない。
- `docs/candidate-boundary.md` のimport制限（静的検査）を通る。候補Policyからimportできる。
- 判断をまたぐ状態を持たない。同じ `PolicyInput` からは常に同じ結果になる。
