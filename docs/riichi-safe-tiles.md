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

この関数を使うのは、未昇格の候補 `PassedTileGenbutsuPlacementAwareSpeedCallPolicy`
（[passed-tile-genbutsu-candidate.md](passed-tile-genbutsu-candidate.md)）だけである。
現Championの行動は変わらない。

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

基盤を適用してよいのは、下の「状態」が**確認済み**の環境だけである。**未確認の環境は、
基盤の自動適用の対象に含めない。** 確認できない点を推測で埋めない。

| 環境 | 確認した version / revision | 状態 |
|---|---|---|
| project標準のRuleSet（規則の定義） | lisjong `85c81f5` の規則文書 | 規則1は確認済み。規則2は規則文書に明文がなく、engineの規則で確認 |
| lisjong-engine | `91af75e3aa11520c3b0543719dc74bb4c517ee06` | 確認済み（規則文書・実装・2026-11-06の固定局面test。下記参照） |
| RiichiEnv | 0.4.10 | 確認済み（局面テスト） |
| RiichiLab（対局server） | — | **未確認** |

#### project標準のRuleSet

- **採用規則**：[riichi-deal-in-label.md](riichi-deal-in-label.md) のフリテンの定義。
- **規則1の根拠**：同文書の「リーチ後の見逃しフリテン」。リーチ宣言打牌より後に、和了できた牌
  （他家の打牌・加槓牌・自摸牌）を和了しなかった場合、以後すべての待ちでロンできない。
- **規則2の根拠**：lisjongの規則文書には、リーチ後の暗槓の条件の明文がない。下のlisjong-engineの
  規則（待ち牌種類の不変は必須条件）で確認する。
- **確認方法**：公開仕様（規則文書）。

#### lisjong-engine `91af75e`

- **採用規則**：`docs/rules.md`（RuleSet）。`riichi_ankan_policy` の既定は
  `PRESERVE_WAIT_AND_DECOMPOSITION`。
- **規則1の根拠**：`src/lisjong_engine/furiten.py`。`next_missed_ron_reason` は立直成立後の
  見逃しを `RIICHI` とし、一度 `RIICHI` になった席を `TEMPORARY` へ弱めない。
  `cleared_temporary_reason` は自分のツモ番で同巡フリテンだけを解除し、`RIICHI` を維持する。
- **規則2の根拠**：`docs/rules.md` の「RiichiAnkanPolicy」。送り槓禁止（暗槓にツモ牌を含むこと）と
  待ち牌種類の不変は、全policy共通の必須条件である。
- **確認方法**：公開仕様（規則文書）と実装確認に加え、2026-11-06 JSTに固定revisionの
  既存32 testsをCPython 3.14.9 / Linuxで実行し、すべてPASSした。
  `test_furiten`、`test_kan.ValidateRiichiAnkanTest`、
  `test_round_state.RoundStateFuritenTest`、
  `RoundStateRiichiTest.test_an_established_riichi_seat_may_only_discard_the_drawn_tile`
  が対象。待ち不変の暗槓の許可、待ちを変える暗槓・送り槓の拒否、成立後のツモ切り限定を含む。
- **追加の局面test**：
  [engine PR #65](https://github.com/lisbun/lisjong-engine/pull/65) の
  `test_riichi_passed_tile_rules.py` は、5s/8s待ちの合成配牌から公開APIだけで進め、
  立直成立→ロン可能な牌を見逃し→自分のツモ番とツモ切り→同じ待ち牌または別の待ち牌の
  打牌、でロンが合法手に出ないことを確認する。5s→5s、5s→8s、8s→5sの3ケース。
  各待ちが見逃し前にロン可能な正の対照も含む。
  この1 test / 3 subcasesを、上記`91af75e`のsourceとfixture、および
  `f11d925f8ce32fa487a4d2159939049d50fb4917`のsourceとfixtureで実行し、両方PASSした。
  外部serverのテストでも、半荘対局による検査でもない。

旧revisionでの再現は、追加testだけを別ディレクトリに置き、旧checkoutで次を実行する。
`/path/to/extra-tests`には追加testだけを置く（新しいfixtureやsourceを混ぜない）。

```sh
PYTHONPATH=src:tests:/path/to/extra-tests python -m unittest -v \
  test_furiten test_kan.ValidateRiichiAnkanTest \
  test_round_state.RoundStateFuritenTest \
  test_round_state.RoundStateRiichiTest.test_an_established_riichi_seat_may_only_discard_the_drawn_tile \
  test_riichi_passed_tile_rules
```


#### RiichiEnv 0.4.10

- **採用規則**：RiichiEnv 0.4.10の合法手生成（MJAI replayの `Kyoku.steps()` が各判断時点で
  提示する合法手）。
- **規則1の根拠**：リーチ後に待ち牌を見逃すと、同じ牌種が再び切られても、別の待ち牌が
  切られても、リーチ者自身のツモ番を挟んだ後でも、ロンが合法手に提示されない。見逃しの前は
  提示される。
- **規則2の根拠**：リーチ後、待ちを変えない暗槓は合法手に出るが、待ちを変える暗槓は出ない。
  リーチ後のツモで、打牌の合法手はツモ牌の1つだけである。
- **確認方法**：局面テスト。lisjong-arenaの `tests/test_riichienv_riichi_safe_tile_rules.py`
  （lisbun/lisjong-arena#494）が、`riichienv.MjaiReplay` の `Kyoku.steps()` が提示する合法手を
  合成局面で確かめる。versionが0.4.10であることもtestで確認する。
- **確認していない範囲**：局面テストはreplayの経路で行った。対局を進める `RiichiEnv` の
  step経路は、同じ合法手生成の条件を共有していると把握している（lisbun/lisjong-arena#247の
  調査）が、同じ局面テストを別には行っていない。
- **既知の差**：MJAI logを `RiichiEnv.apply_event()` で進めるforward stateは、見逃しによる
  フリテンのflagを更新しない（lisbun/lisjong-arena#247）。これはreplay経路の再現の差であり、
  対局中の規則の差ではない。Arena側のreplayが補っている。

#### RiichiLab（対局server）

**未確認のまま。自動適用は解禁しない。** 2026-11-06 JSTに公開根拠を調べ直した。

- [公式Local Testing文書](https://riichi.dev/docs/local-testing) は、serverが
  `riichienv-core`を使い、Python packageが同じRust coreを包むと説明している。
  共通実装であることの根拠であり、稼働中serverが検証済み0.4.10そのものだという
  versionの証拠にはしない。
- [公式告知 #252（2026-10-28）](https://github.com/smly/RiichiEnv/discussions/252) は、
  稼働engineのversion表示を追加したと説明している。ただし今回の取得では表示値を
  確認できなかった。最新release番号から稼働versionを推測しない。
- [upstream #192](https://github.com/smly/RiichiEnv/issues/192) にはreleaseとserverへの
  deployに時間差があった事例もある。これは現在の不具合を示すものではなく、releaseと
  deployを同一視しない理由である。

残る確認は、(1) serverに表示されるengine versionと確認日時・出典、(2) その版の
見逃しフリテン継続と立直後の待ち不変の実装・設定、(3) 固定版での該当局面testの対応付け。
版が0.4.10なら既存のreplay testとの対応を確認する。異なる版ならその版で確認し、
0.4.10の結果をそのまま転用しない。現在のversionが分かっても、過去の#489の対局時点に
遡って同じ版だったとは扱わない。bot接続・新規対局による確認は今回行っていない。

### 補助証拠の区別（2026-11-06 JST照合）

次はGitHubの実行記録の照合であり、local artifactの再集計・hashの独立再検証ではない。

| 記録 | 環境・入力と分母 | 確認できること | 確認できないこと |
|---|---|---|---|
| [Arena #489結果](https://github.com/lisbun/lisjong-arena/issues/489#issuecomment-6127205027) | RiichiLab実戦dev 209席・baseline 214席（共有66対局）。旧診断のR1 7,878打牌のうち、最小向聴1以上・合法打牌の現物が「通った牌だけ」で、実際に通った牌を切った710打牌（dev 324 / baseline 386） | この限定集合でロンされた打牌0件という実行報告 | 全423席の全打牌での検査、更新後Adapterでの`passed(r)`再検査、server規則の証明 |
| [Arena #493移行照合](https://github.com/lisbun/lisjong/issues/302#issuecomment-6234825958) | 同じ423席を更新後Adapterでoffline replay。67,090判断行、宣言位置の一意な比較15,517件、旧診断で複数候補の30行は照合対象外 | 宣言位置の不一致0件という報告 | ロン違反の計数。この実行では実施していない |
| [Arena #503補助検査](https://github.com/lisbun/lisjong/issues/302#issuecomment-6245467517) | engine `91af75e`、lisjong `174bf48`、Arena `e7affd6`。1,300半荘中の記録対象席で、自席非リーチ・他家リーチ者がちょうど1人の49,294打牌（通った牌だけ7,253） | `safe(r)`の打牌を対象リーチ者にロンされた違反0件という報告 | RiichiLabの規則・実戦検査、規則の証明、候補の棋力 |

#489はArena `50b8082ecf89bfeea0887138f7f0357201df1df5`、lisjong
`51e832e50a0ee71eac65e4a017f46590e7438c04`、RiichiEnv 0.4.10 / Rustによる旧診断。
出力正本はユーザーPCの
`C:\Dev\lisjong-artifacts\issue-483-r1-choice-loss\issue-489-champion-play.json`（schema v3）。
#493のAdapter供給開始はArena `1a1cce76b2a052986c9f6746702a237c63af9e7c`であり、
#489の「0件」をこの更新後の再検査として扱わない。

したがって、過去コメントの「実対局検査は未実施」と「423席は別途0件」は、
**更新後Adapterでのロン検査は未実施／旧診断の限定710打牌では0件と報告済み**と
読み分ける。「423席すべてを検査して安全性確認済み」という要約は使わない。
更新後AdapterでのRiichiLab補助検査は、対象artifactがこの作業環境にないため未実施。
追加の実戦集計・再取得を暗黙に開始しない。いずれの0件も規則成立の主根拠にせず、
RiichiLabを確認済みとする根拠にもしない。

## 情報境界

- 入力は `PolicyInput` だけである。他家の伏せ牌・山・将来の状態を使わない。
- `docs/candidate-boundary.md` のimport制限（静的検査）を通る。候補Policyからimportできる。
- 判断をまたぐ状態を持たない。同じ `PolicyInput` からは常に同じ結果になる。
