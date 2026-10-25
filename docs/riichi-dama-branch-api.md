# リーチ判断の特徴APIとD arm用リーチ抑止wrapper（Issue #352）

設計の正本: [Arena #539](https://github.com/lisbun/lisjong-arena/issues/539)の「設計確定」§2・§4
（**riichi-dama-branch-v1**）。consumer: [Arena #540](https://github.com/lisbun/lisjong-arena/issues/540)。
**Champion（`PlacementAwareSpeedCallPolicy`）と既存Policyの挙動は変えない。**候補Policyではない。

module: [`lisjong.policies.riichi_dama_branch`](../src/lisjong/policies/riichi_dama_branch.py)
（`RIICHI_DAMA_FEATURES_VERSION = "riichi-dama-features-v2"`。v2は
[lisjong #354](https://github.com/lisbun/lisjong/issues/354)で和了牌の選択と`winning_tile` / `red_dora`を追加した。v1は読まない）

| 入口 | 用途 |
| --- | --- |
| `analyze_riichi_decision(decision, declaration_tile)` → `RiichiDecisionFeatures` | 判断時点の特徴（§4） |
| `riichi_decision_features_to_value(features)` / `parse_riichi_decision_features(value)` | version付きJSON objectとの往復 |
| `RiichiDamaBranchPolicy(inner, target_round, branch_discard)` | D armのwrapper（§2） |

## 1. 解析API

### 入力と情報境界

- `decision`: `RiichiAction`（自席）が合法な判断のpublic `DecisionContext`
- `declaration_tile`: 自分の宣言牌。その判断の手牌に含まれる`Tile`（赤を区別する）

他家の手牌、山、裏ドラ、privilegedな観測は受け取らない。結果は宣言牌を打った後の13枚
（副露があればその分少ない）について計算する。

### `RiichiDecisionFeatures`

| field | 意味 |
| --- | --- |
| `seat` / `declaration_tile` | 自席と宣言牌 |
| `waits` | 待ち牌の種類Wごとの`WaitFeatures`（牌種のcanonical順、1件以上） |
| `furiten` | 打牌後の自分の河（鳴かれた牌を含む）と宣言牌の牌種に、Wのどれかがある |
| `double_riichi` | 自分がまだ打牌しておらず、どの席にも副露（暗槓を含む）がない。engineの`was_first_discard and not had_prior_call`と同じ |
| `own_discard_count` | 巡目。判断時点の自分の打牌数（宣言牌を含まない。鳴かれた牌を含む） |
| `opponent_riichi_count` | 公開されたリーチ状態が`NONE`でない他家の人数（`DECLARED`を含む） |
| `is_dealer` | 自席が親か |
| `is_all_last` | オーラス相当か。Championの`_is_all_last`と同じ（南4局、または西入以降） |
| `score` | 現在の持ち点 |
| `rank` / `tied_opponent_count` | `1 + 自分より持ち点が多い他家の数`と、同点の他家の数。起家が`PolicyInput`にないため、同点の順位はここで決めない |
| `dama_choice` | 診断用。`RiichiAction`を除いた合法手で、新しいChampion instanceが選ぶ行動。宣言牌と一致するかの判定はArena側で行う |

`wait_tile_types`と`total_unseen_count`（Wの見えない残り枚数の合計）はpropertyで引ける。

### `WaitFeatures`

| field | 意味 |
| --- | --- |
| `tile_type` | 待ち牌種。打牌後の手の向聴数を下げる（和了形にする）基礎牌種で、通常形・七対子・国士無双を含む（`effective_tile_types`）。手牌に4枚ある牌種は含めない |
| `winning_tile` | 4通りの点数を評価した和了牌。原則は赤でない牌で、通常牌の上限を超える場合だけ赤牌（下記） |
| `unseen_count` | 4 − (自分の手牌 + 河・副露・ドラ表示牌で見えている枚数)。`derive_remaining_tile_inventory`で数える |
| `dama_ron` / `dama_tsumo` / `riichi_ron` / `riichi_tsumo` | `evaluate_win`の結果（`WaitScore`） |

見えない残り枚数: 宣言牌は手牌から河へ移るだけなので、判断時点の手牌で1回だけ数えた値が
そのまま打牌後の値になる（宣言牌を二重に数えない）。

和了牌（#354）: 原則は赤でない牌。赤ありのルール（`PROJECT_STANDARD_SCORING_RULES.red_dora_enabled`）では
各色の5は通常3枚＋赤1枚なので、評価入力の物理牌（宣言後の手牌・副露・ドラ表示牌）の通常5と和了牌の合計が3枚を
超える場合だけ赤5で評価する（`winning_tile_for`）。`evaluate_win`が同じ物理牌の集合で枚数を検査するため、ドラ表示牌も
数える。赤5でも成立しない（赤5が既に評価入力にある等）場合は`RiichiDamaAnalysisError`。`unseen_count`は公開情報から
別に数えるので、赤で評価してもその赤が見えていないとは限らない。赤の加点でS1（ダマ満貫）の判定が変わりうる。

点数の条件: 和了牌は上記、一発なし、状況役なし（`WinSituation.NORMAL`）。ダマは裏ドラ
なしを明示し（`UraDoraIndicators(())`）、リーチは`URA_DORA_EXCLUDED`。`double_riichi`なら
リーチ側は`RiichiStatus.DOUBLE_RIICHI`で評価する。振聴でもロンの点数は計算する（振聴は
`furiten`で判定する）。

### `WaitScore`

| field | `SCORED` | `NO_YAKU` |
| --- | --- | --- |
| `status` | `EvaluationStatus.SCORED` | `EvaluationStatus.NO_YAKU` |
| `han` / `limit` / `winner_points` | 翻・`ScoreLimit`・和了者の受取点 | `None` |
| `fu` | 符（役満では`None`になりうる） | `None` |
| `ron_payment` | ロンの支払点（ツモでは`None`） | `None` |
| `tsumo_dealer_payment` / `tsumo_non_dealer_payment` | ツモの親・子の支払点（親のツモでは前者が`None`、ロンでは両方`None`） | `None` |
| `red_dora` | 翻に含まれる赤ドラの枚数（和了牌が赤ならそれを含む）。役満などドラの内訳がない場合は`None` | `None` |

支払点は本場・供託を含まない。部分集合（S1 / S3）の閾値はこのAPIで判定しない。たとえば
「W_liveの全ての待ちでダマのロンが`SCORED`かつ満貫以上」は、`unseen_count >= 1`の待ちについて
`dama_ron.is_scored`と`dama_ron.limit`の比較だけで判定できる（`ScoreLimit`は
`NONE` → `MANGAN` → `HANEMAN` → `BAIMAN` → `SANBAIMAN` → `YAKUMAN`の順）。

### `RiichiDamaAnalysisError`（fail closed）

- `RiichiAction`がその判断で合法でない、自席がすでにリーチしている
- 宣言牌が手牌にない（赤のidentityを含めて照合する）
- 宣言牌を打った後が聴牌でない、聴牌でも待ち牌種がない
- 構造上の待ちが`evaluate_win`で和了形にならない、赤5を含めてどの物理牌でも和了牌にできない
- JSON値が下記の形でない

tile conservationの不整合は`derive_remaining_tile_inventory`の例外がそのまま出る。

### JSON（`riichi-dama-features-v2`）

`riichi_decision_features_to_value()`はJSONへそのまま書けるobjectを返す。

```text
{
  "version": "riichi-dama-features-v2",
  "seat": 0,
  "declaration_tile": {"category": "honor", "rank": 1, "is_red": false},
  "waits": [
    {"tile_type": {"category": "souzu", "rank": 1},
     "winning_tile": {"category": "souzu", "rank": 1, "is_red": false}, "unseen_count": 4,
     "dama_ron": {"status": "scored", "han": 1, "fu": 30, "limit": "none",
                  "winner_points": 1000, "ron_payment": 1000,
                  "tsumo_dealer_payment": null, "tsumo_non_dealer_payment": null,
                  "red_dora": 0},
     "dama_tsumo": {...}, "riichi_ron": {...}, "riichi_tsumo": {...}},
    ...
  ],
  "furiten": false, "double_riichi": false, "own_discard_count": 1,
  "opponent_riichi_count": 0, "is_dealer": false, "is_all_last": false,
  "score": 25000, "rank": 1, "tied_opponent_count": 3,
  "dama_choice": {"kind": "discard", "actor": 0,
                  "tile": {"category": "honor", "rank": 1, "is_red": false},
                  "tsumogiri": false}
}
```

- enumは`.value`の文字列、`Seat`はint。`dama_choice.kind`は`discard` / `tsumo` / `ankan` /
  `kakan` / `kyuushu_kyuuhai`（自分の手番の行動）で、fieldは各Actionのsemantic fieldだけ
- `parse_riichi_decision_features()`は、未知の`version`、fieldの過不足、型の違い（boolとint
  を区別する）、値の範囲・整合（待ちの順序・重複、`NO_YAKU`に値がある等）を
  `RiichiDamaAnalysisError`で拒否する。読み戻した値は元の値と等しく、再射影したJSONも同じになる
- 意味定義かwire shapeを変える場合はversionを上げる。既存versionの意味は書き換えない

## 2. D arm wrapper `RiichiDamaBranchPolicy`

| 引数 | 意味 |
| --- | --- |
| `inner` | 内側のPolicy（Champion）。`choose_action`を持つ |
| `target_round` | 対象局`RoundIdentity(round_wind, hand_number, honba)`。`RoundIdentity.of(round_state)`で作れる |
| `branch_discard` | 分岐点で打つ`DiscardAction`。Rの宣言牌選択で選ばれた打牌（赤のidentityとツモ切りflagを含む） |

| 判断 | 返す行動 |
| --- | --- |
| 対象局以外 | `inner.choose_action(decision)`をそのまま返す（リーチを含む） |
| 対象局で`inner`がリーチ以外を選んだ | `inner`の行動のまま（和了・鳴き・打牌） |
| 対象局で`inner`が初めて`RiichiAction`を選んだ（分岐点） | `branch_discard`。合法手に完全一致で含まれなければ`RiichiDamaBranchError` |
| 分岐後、同じ局で`inner`が`RiichiAction`を選んだ | `RiichiAction`を除いた合法手で`DecisionContext`を作り直し、`execute_policy(inner, ...)`の結果。`replacement_count`を1増やす |

状態（Arena側でINVALIDを区別するため）:

| `status` / 例外 | 条件 |
| --- | --- |
| `BranchStatus.PENDING` | 分岐点はまだ来ていない（対象局の前、または対象局の途中） |
| `BranchStatus.BRANCHED` | 分岐点で`branch_discard`を返した |
| `BranchStatus.MISSED` | 分岐点が来ないまま対象局を出た（対象局の判断の後に別の局の判断が来た） |
| `RiichiDamaBranchError` | 対象局を出た後に再び対象局の判断が来た（分岐点がもう一度来うる）、指定の打牌が合法でない、リーチを除くと合法手が残らない |

- D armでは対象局の精算後に打ち切るため、局の終わりをwrapperは見ない。Arenaは打ち切った時点で
  `status is BRANCHED`を要求し、`PENDING`（分岐点が来ないまま局が終わった）をINVALIDにする
- wrapperは分岐の有無という判断をまたぐ状態を意図的に持つ。対比較実行専用で、1つの対局・1つの席
  ごとに新しいinstanceを使う。分岐の特定には局のidentityと`inner`の選択だけを使い、他席・山の情報は
  受け取らない

## 検証

[`tests/test_riichi_dama_branch.py`](../tests/test_riichi_dama_branch.py)で次を固定する
（点数は実際の`evaluate_win`。native scoring backendが必要）。

- 待ち・残り枚数（河・副露・ドラ表示牌）、平和の4通りの点数、ダマのロンの役なし、満貫 / 跳満、
  親の支払、赤5（宣言牌の赤identityで点数が変わる）、七対子・国士無双13面待ち
- 和了牌（#354）: 手に通常5が3枚ある待ちは赤5で評価（Arena #542の分岐 472006-9-1）、通常5で評価できる場合、
  ドラ表示牌の通常5で上限を超える場合、赤5でも成立しない場合の拒否、`winning_tile`・`red_dora`のJSON往復、v1の拒否
- 宣言牌自身による振聴と宣言牌の二重計数防止、過去の自分の打牌による振聴
- ダブルリーチ（副露があれば不成立）、巡目、他家のリーチ人数（宣言中を含む）、順位と同点、オーラス相当
- ダマでの選択が新しいChampionの選択と一致し、解析の前後でChampionの行動が変わらない
- 入力不整合の拒否、JSONの往復と不正値の拒否
- wrapper: 対象外の局の素通し、分岐点で指定の打牌、指定の打牌が合法でない場合（ツモ切りflag・赤・actor）の
  拒否、それ以降のリーチの差し替えと回数、innerがリーチを選ばない場合、`MISSED`、対象局の再出現の拒否、
  分岐以外でChampionと同じ行動
- `check_candidate_imports`による情報境界（#296）の検査

## consumerへの引継ぎ

- Arenaはここで返す値を記録・照合し、点数・待ち・役・振聴を再計算しない。部分集合の判定は
  返した値の比較だけで行う
- 実行に使うのは**本APIのmerge後main revision**のwheel。CIのnative wheel artifactを取得し、
  wheel SHA-256、`SOURCE_REVISION`、native API version、本moduleのimportをrun planへ束縛する。
  過去revisionのwheelやRust-only gateのPASSを同revisionの証拠として使わない
