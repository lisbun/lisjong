# 門前の最初のPonをPassへ替えるD arm wrapper（Issue #363、P1）

設計の正本: [#362](https://github.com/lisbun/lisjong/issues/362)（[設計案](https://github.com/lisbun/lisjong/issues/362#issuecomment-6545854838)・
[改訂](https://github.com/lisbun/lisjong/issues/362#issuecomment-6546073400)・[完了記録](https://github.com/lisbun/lisjong/issues/362#issuecomment-6546102219)、DESIGN-READY）。
consumer: [Arena #553](https://github.com/lisbun/lisjong-arena/issues/553)（`pon-first-pass-branch-v1`）。
**Champion（`PlacementAwareSpeedCallPolicy`）・既存Policy・既存wrapperの挙動は変えない。**候補Policyではない。

module: [`lisjong.policies.pon_first_pass_branch`](../src/lisjong/policies/pon_first_pass_branch.py)
（`PON_FIRST_PASS_BRANCH_VERSION = "pon-first-pass-branch-wrapper-v1"`）

| 入口 | 用途 |
| --- | --- |
| `is_closed_for_pon_branch(policy_input)` | 自席が門前か（暗槓以外の副露なし） |
| `is_pon_branch_candidate(decision, action)` | 選んだ行動が、門前の自席の合法な`PonAction`か（分岐点の候補） |
| `PonFirstPassBranchPolicy(inner, target_round, branch_pon)` | D armのwrapper |

## 1. 分岐点（共有predicate）

- **門前**: 自席の副露にChi / Pon / 大明槓 / 加槓がない。暗槓は門前に含める
- **分岐点の候補**: 自席が門前で、選んだ行動がその判断の合法手に含まれる自席の`PonAction`
- **分岐点**: 対象局で候補に当たる**最初の**判断。「最初」は呼び出し側（wrapper、ArenaのR観測）が局ごとに数える

ArenaのR観測はこの関数を使い、門前の判定を再実装しない。判定は分岐席の判断時点の
`DecisionContext`と選んだ行動だけを見る。同じ反応windowの他席の選択、windowの解決結果
（Ponが成立したか、他家のロン・チー）、分岐後の情報は受け取らない（情報境界）。

## 2. wrapper `PonFirstPassBranchPolicy`

入力は`inner`（Champion）、対象局（`RoundIdentity`: 場風・局・本場）、Rで記録したPon
（`PonAction`。called tile・消費牌（赤のidentityを含む）・放銃者）。

| 項目 | 契約 |
| --- | --- |
| 席の指定 | **対象席にだけ装着する**。席は受け取らない（R1の`RiichiFirstDamaBranchPolicy`と同じ）。どの席に装着するかはArenaが決める |
| 対象局以外 | `inner`の選択をそのまま返す |
| 分岐点 | その判断の合法手にある自席の`PassAction`を返す |
| エラー（`PonFirstPassBranchError`） | 分岐点で`inner`の選んだPonが記録と完全一致しない（赤の有無だけの違いを含む）、自席の`PassAction`が合法手にちょうど1つない、対象局を出た後に再び対象局の判断が来た |
| 差し替え | 1回だけ。分岐後は（同じ局で再び門前からPonを選んでも）`inner`のまま |
| 副露後のPon | 分岐点の候補ではない。`inner`のまま |
| `status` | 分岐前`PENDING`、分岐後`BRANCHED`、分岐点が来ないまま対象局を出たら`MISSED`（`riichi_dama_branch.BranchStatus`を共有） |
| `calls_after_branch_count` | 分岐後に、対象局で`inner`の鳴き（Chi / Pon / 大明槓）をそのまま返した回数（Arenaの照合・記述用） |

**未到達の扱い**: wrapperは`PENDING` / `MISSED`を保持するだけで例外にしない。対象局の精算は
判断として届かないため、wrapperからは局の終わりを知れない。D armを対象局の精算で打ち切った
時点で`BRANCHED`でなければINVALIDとする判定はArenaが行う。

判断をまたぐ状態（分岐の有無・対象局の出入り・分岐後の鳴きの回数）を意図的に持つ、対比較実行
専用のPolicyである。1つの対局・1つの席ごとに新しいinstanceを使う。

## 3. 範囲外

- 将来の学習の特徴encoding（pilot後の本学習設計で、TRAIN生成前に固定する）
- Ponの成立 / ロン取消の判定、D armの照合、配分判定（Arena #553）
