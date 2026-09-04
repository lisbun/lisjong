# Policy current status

## Purpose

本書は、`lisjong`が公開するPolicy implementationのcurrent roleを示すrepository-owned snapshotである。
Policy strengthに関する情報は、次のownerへ分離する。

```text
lisjong Policy status
    = current interpretation / current snapshot

lisjong-arena evaluation artifact
    = measurement source of truth

bounded GitHub Issue / PR
    = individual work / decision history
```

本書はhistorical evaluation logではない。過去runの数値や採用・保留・棄却の全経緯を
複製せず、current roleと代表的evidenceへのreferenceだけを保持する。Policy strengthを
どのように比較するかは、Arena-ownedの
[Policy strength evaluation policy](https://github.com/lisbun/lisjong-arena/blob/main/docs/policy-strength-evaluation.md)
を正本とする。

Snapshot date: **2026-09-01**

## Current strength baseline

current strength baselineは次のPolicyである。

| Field | Current value |
| --- | --- |
| Arena identity | `yakuhai-call` |
| Implementation class | `YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy` |
| Family | defense + finite-horizon + hand-value + conservative call |
| Role | current overall-strength baseline |

`yakuhai-call`は、fresh holdout 10,000-game comparisonで当時のbaseline `combined`を
上回ったdecisionに基づき昇格した。current interpretationの根拠は
[lisjong #121のGate 2 decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5471486662)
に残す。historical measurement数値は本書へ重複転記しない。

current baselineはruntime profileへの自動配備を意味しない。RiichiLab等のexecution profile
mappingは`lisjong-arena`が所有し、strength statusとは独立に変更・検証する。

## Public Policy inventory

`lisjong.policies`のpublic exportsを、current management roleとともに示す。
`Arena identity`が`—`のPolicyは、public implementationではあるがcurrent Arena
`single_round_compare` catalogへは登録されていない。

| Public Policy | Arena identity | Family | Current role | Major capabilities / runtime characteristics |
| --- | --- | --- | --- | --- |
| `MinimalPolicy` | — | deterministic contract baseline | foundational / boundary validation | legal actionをstable total orderで選ぶ。strengthを目的とせず、計算costは小さい |
| `ShantenPolicy` | — | structural efficiency | foundational comparator | 向聴数中心のdiscard selection。打点・守備・lookaheadを扱わない |
| `UkeirePolicy` | — | structural efficiency | foundational comparator | current ukeireを加えた局所的な牌効率。future lookaheadを扱わない |
| `TwoStepUkeirePolicy` | `two-step` | structural efficiency | historical structural baseline / cheap comparator | shanten、current ukeire、second-step ukeireを順に比較。exact finite-horizon DPより軽量 |
| `GenbutsuDefenseTwoStepUkeirePolicy` | — | structural efficiency + defense | component comparator | 非聴牌かつ被立直時に全リーチ者への共通現物を優先。generic risk / push-fold EVではない |
| `ValueAwareTwoStepUkeirePolicy` | — | structural efficiency + lightweight value | component comparator | 公開dora indicator由来doraと赤ドラ保持をtie-breakへ追加。actual score / EVではない |
| `HandValueAwareTwoStepUkeirePolicy` | `hand-value-aware` | structural efficiency + hand value heuristic | evaluated component candidate | 役牌・dora・赤dora・軽量yaku routeを同一shanten / ukeire候補内で比較。future branchはstructural semanticsを維持 |
| `FiniteHorizonCompletionPolicy` | `finite-horizon` | exact finite-horizon structural search | component comparator | conditional 3-self-draw completion massをexact DPで比較。runtime costが高く、actual 3巡以内和了確率ではない |
| `GenbutsuDefenseFiniteHorizonValueAwarePolicy` | `combined` | defense + finite-horizon + lightweight value | predecessor strength baseline / comparator | 共通現物constraint、exact finite-horizon DP、ValueAware fallbackを合成。`yakuhai-call`より前のbaseline |
| `GenbutsuDefenseFiniteHorizonHandValueAwarePolicy` | `extended-combined` | defense + finite-horizon + hand value heuristic | no-call parent / historical causal comparator; not promoted | `combined`のvalue stageをHandValueAwareへ拡張。400-game screenでpromotionせず、DP由来の高costを持つ |
| `YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy` | `yakuhai-call` | defense + finite-horizon + hand value + conservative call | **current strength baseline** | no-call parentに役牌Pon起点のstrict shanten-improving callを追加。generic call EVではなく、DP由来の高costを持つ |

ここでの`current role`はPolicy-strength管理上の位置づけであり、public APIの安定性、
deprecation、runtime profile assignmentを表さない。implementation時のmodule / class docstringに
`experimental`とある場合も、current strength roleは本書のsnapshotを参照する。

## Training coverage source Policy (not part of the strength hierarchy)

`lisjong #151`は、`lisjong-arena #131`のHandBelief Stage 3 Entry Gate pilotが
daiminkan / ankan / kakan / rinshan_drawを1件も観測しなかったcoverage holeへの
対応として、次のPolicyを追加した。

| Field | Value |
| --- | --- |
| Implementation class | `KanCoverageYakuhaiCallPolicy` |
| Public import reference | `lisjong.policies.kan_coverage_yakuhai_call:KanCoverageYakuhaiCallPolicy` |
| Arena identity | — (Arena `POLICY_CATALOG`未登録。explicit import referenceで利用する) |
| Role | HandBelief / experiment **training coverage source**。current strength baselineでも、strength hierarchy上のcomparatorでもない |

```text
kan-capable coverage source
!= stronger Policy
!= current strength baseline
!= production recommendation
```

selection semanticsは次の優先順位で固定する。

```text
1. RonAction / TsumoAction        (winning action)
2. DaiminkanAction / AnkanAction / KakanAction  (legal kan action、種別を問わない)
3. delegated normal-play decision (yakuhai-call: YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy)
```

合法性は常に`DecisionContext.legal_actions`を正本とし、Policy側でkan legality
（riichi後ankan等を含む）を再判定しない。複数kan候補が同時にlegalな場合は、
semantic fieldだけから作るdeterministic total orderで1件を選ぶ。kan種別間の
固定順序（`daiminkan < ankan < kakan`）はdeterminismを固定するimplementation
choiceであり、麻雀上の優劣を意味しない。

winning actionもkanも無いdecisionは、既存`yakuhai-call`
（`YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy`）へ
compositionでdelegateする。delegate instanceは構築時に1回だけ生成し、既存
`yakuhai-call` / `MinimalPolicy`のclass自体のsemanticsは変更していない。

本Policyの追加だけでは「kan / rinshan coverageが十分」とは結論しない。実際の
daiminkan / ankan / kakan / rinshan frequencyとHandBelief training-sourceとして
のcoverage adequacyは、後続の`lisjong-arena`側bounded pilotで測定する。

## Representative evidence

代表的なdecision / evidenceだけを案内する。数値と完全な経緯はlink先のhistorical recordに残す。

| Policy / role | Representative reference |
| --- | --- |
| `yakuhai-call` current baseline | [fresh Gate 2 promotion decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5471486662)、[Arena evaluation wiring](https://github.com/lisbun/lisjong-arena/blob/main/docs/yakuhai-call-evaluation.md) |
| `combined` predecessor baseline | [fresh holdout promotion evidence](https://github.com/lisbun/lisjong/issues/121#issuecomment-5462935934) |
| `extended-combined` not promoted | [bounded Gate 1 decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5466162346)、[Arena historical evaluation document](https://github.com/lisbun/lisjong-arena/blob/main/docs/extended-combined-evaluation.md) |
| `finite-horizon` component comparator | [10,000-game follow-up decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5439919623) |
| `hand-value-aware` evaluated component candidate | [4,000-game follow-up interpretation](https://github.com/lisbun/lisjong/issues/121#issuecomment-5431486646) |
| foundational / unevaluated exports | implementation / contract baselinesであり、本書ではformal strength claimを行わない |

current Policy-vs-Policy ABBB runは、Arena #110で追加されたversioned immutable artifactと
reaggregationをmeasurement boundaryとして利用する。artifact対応範囲と制約は
[Arena policy](https://github.com/lisbun/lisjong-arena/blob/main/docs/policy-strength-evaluation.md#measurement-source-of-truth)
を参照する。過去runにartifactが存在しない場合、存在するものとして扱わない。

## Status update workflow

今後のPolicy workは終了条件のあるbounded Issue / PRとして進める。

```text
concrete Policy work
    -> bounded lisjong Issue / implementation PR

strength evaluation
    -> bounded lisjong-arena Issue / Arena artifact / decision

current interpretation changed
    -> relevant bounded PRで本書を更新
```

新しいPolicyや将来candidateの存在だけを理由に、long-lived Policy-strength tracking Issueを
作成しない。speculative directionはconcrete consumer requirementが生じるまでimplementation
Issueへ変換しない。過去のwork / decision historyは対応するclosed Issue / PRを参照する。
