# 鳴き採用・直後打牌の限定解析API（Issue #350）

親設計: [Arena #533](https://github.com/lisbun/lisjong-arena/issues/533)の**call-value-shadow-v1**
（意味定義の正本）。consumer: [Arena #534](https://github.com/lisbun/lisjong-arena/issues/534)。
[#348](speed-call-post-call-value-diagnostic.md)のtest-local診断helperを、production-ownedな狭い
解析入口へ移したもの。**Policy / Championの行動は変えない。**候補Policyやcatalogは追加しない。

module: [`lisjong.policies.speed_call_analysis`](../src/lisjong/policies/speed_call_analysis.py)
（`CALL_VALUE_SHADOW_VERSION = "call-value-shadow-v1"`）

## 入口

| 関数 | 入力 | 返却 |
| --- | --- | --- |
| `analyze_speed_call_decision(decision, selected)` | 鳴き判断のpublic `DecisionContext`、Championが選んだAction | `SpeedCallDecisionAnalysis` |
| `analyze_post_call_discard(decision, call_analysis, selected)` | call成立後の**実際の**mandatory discard `DecisionContext`、同じcallの`SpeedCallDecisionAnalysis`、Championの実打牌 | `PostCallDiscardAnalysis` |

どちらもChampion（`PlacementAwareSpeedCallPolicy`）の判断を同じ入力で再実行し、渡された
Actionと一致しなければ`SpeedCallAnalysisError`でfail closedする。入力は`DecisionContext`だけで、
hidden truth・未来event・cross-decision状態を使わない。call意図と成立/取消の区別、
call→直後打牌の一対一対応はArenaの観測側が所有する。返却値はすべてfrozen dataclass。

## `SpeedCallDecisionAnalysis`

| field | 意味 |
| --- | --- |
| `route` | `PARENT_DECISION`（parentがPass以外を選んだ。役牌parent経路等）/ `PARENT_PASS`（parentはPass、スピード鳴き不採用）/ `SPEED_CALL` |
| `action` | Championが選んだAction |
| `mode` | `GameSituationMode`（NORMAL / ALL_LAST_TOP_SPEED / ALL_LAST_LAST_VALUE） |
| `closed_before_call` | call前の門前状態（暗槓のみは門前） |
| `call` | exact Chi/Pon identity（actor / target / called_tile / consumed_tiles。赤を区別） |
| `route_shanten` | call後の最小route shanten |
| `value` | その最小を達成する最大打点proxy |
| `minimum_value` | mode・門前状態・route shantenに対する採用閾値 |
| `is_primary_target` | #533の主対象: `SPEED_CALL`・NORMAL・門前・route shanten 1・閾値2 |

`call`以下の4値は`SPEED_CALL`のときだけ値を持ち、それ以外は`None`。スピード鳴きの判定は
Policyの`_decide`と同じ`_select_speed_call`を使い、採用閾値も同じ`_minimum_call_value`を使う。

## `PostCallDiscardAnalysis`

| field | 意味 |
| --- | --- |
| `call_analysis` / `discard` | 入力のcall解析値と実打牌 |
| `horizon` / `hidden_tile_count` / `sequence_denominator` | 全候補で共通のFiniteHorizon horizon（3）、root remaining inventory合計`N`、`D = N(N-1)(N-2)` |
| `candidates` | 全合法打牌の`PostCallDiscardFacts`（`discard_action_sort_key`順） |
| `classification` | `CONSISTENT` / `BELOW_MAX` / `BELOW_THRESHOLD` |
| `alternative` | `BELOW_THRESHOLD`のときだけの決定的代替。それ以外は`None` |
| `speed_loss` | `BELOW_THRESHOLD`のときだけの`SpeedLoss`。それ以外は`None` |

`PostCallDiscardFacts`は`action`、`route_shanten`、`value`（打点proxy）、`completion_mass`、
`ukeire`（構造的受入枚数）。`chosen` / `alternative_facts` / `facts_for(action)`で引ける。

- 打点proxyはroute翻相当とroute内ドラ・赤ドラ枚数の和で、実際の翻・得点・EVではない
- completion massは本番打牌選択と同じexact整数で、`0 <= M <= D`を検査する。分母はlive wallではなく、
  `M / D`は実対局の和了確率ではない。異なる局面のraw massを直接比べない

### 分類

| 分類 | 条件（実打牌のroute shantenはcall時と同じ） |
| --- | --- |
| `CONSISTENT` | 実打牌proxy = call時の最大proxy |
| `BELOW_MAX` | 採用閾値 <= 実打牌proxy < 最大proxy |
| `BELOW_THRESHOLD` | 実打牌proxy < 採用閾値 |

VALUE / SPEED modeや門前→聴牌・既副露の鳴きも同じ分類で返す。主対象の判定は
`is_primary_target`で行い、別集計にする。

### 代替と速度損失

代替候補は「route shanten = call時の最小route shanten、かつproxy >= 採用閾値」の合法打牌。
completion mass最大 → proxy最大 → 受入最大 → `discard_action_sort_key`最小で1つ選ぶ。
整数比較だけで、合法手の列挙順に依存しない。赤5と通常5は別Actionとして扱う。
post-callのmandatory discardではtsumogiriは常に`False`なので、代替と実打牌の差は牌種か赤identityの差である。

`SpeedLoss`は実打牌mass `B`、代替mass `A`から:

- `mass_difference = B - A`
- `relative_numerator / relative_denominator = (B - A) / B`。`B = A = 0`は`0 / 1`で`zero_mass=True`
- `A > B`、`B = 0 < A`は前提違反として`SpeedCallAnalysisError`

completion proxy差`(B - A) / D`、受入差、打点proxy差は`candidates`の整数から導ける。floatで判定しない。

## `SpeedCallAnalysisError`（STOP-INVALID）

- 渡されたActionがChampionの判断・合法手と一致しない
- `call_analysis`が`SPEED_CALL`でない
- 成立後contextがこのcallの直後のmandatory discardでない: self seat、ツモ牌なし、mode不変、
  最後の自席副露（種類・from_seat・called_tile・exact牌identity）、それ以前の副露と門前状態、
  手牌枚数、targetの河の最後の牌と`called_by`、合法手（全て手出し打牌・actor一致・重複なし・
  手牌の牌identityから喰い替え禁止牌種を除いた集合と一致）
- remaining inventoryがhorizon未満、massが`D`超
- call時の(最小route shanten, 最大proxy)が合法打牌に再現しない、実打牌がroute shantenを変える
- 代替候補が空、速度損失の前提違反

tile conservationの不整合は`derive_remaining_tile_inventory`の例外がそのまま出る。

## 検証

[`tests/test_speed_call_analysis.py`](../tests/test_speed_call_analysis.py)で次を固定する。

- #348の反例・一致・不発動fixtureの再現（R1は主対象の閾値割れ、代替4s、mass 45980 / 40176、受入24 / 23）
- 喰い替えを含む閾値割れfixture、BELOW_MAXとの分離
- 代替のtie順（mass → proxy → 受入 → 赤identityを含むsort key）、0 mass、候補順序非依存
- 不正call / post contextの拒否、APIの前後で本番Action不変、#296のimport境界検査

#348の元のtest（23件）はそのまま維持する。

## consumerへの引継ぎ

- Arenaはここで返す値を記録・照合し、proxy・向聴・受入・completion mass・代替・分類を再計算しない
- 実行に使うのは**本APIのmerge後main revision**のwheel。CIのnative wheel artifactを取得し、
  wheel SHA-256、`SOURCE_REVISION`、native API version、本moduleのimportをrun planへ束縛する。
  旧`89dd00d`のwheelや#531のgate PASSを同revisionの証拠として使わない
