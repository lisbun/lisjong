# Policy current status

## Purpose

本書は、`lisjong`が公開するPolicy implementationの**current role**を示すrepository-owned snapshotである。
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

Snapshot date: **2026-09-06**

## Current strength baseline

current strength baselineは次のPolicyである。

| Field | Current value |
| --- | --- |
| Arena identity | `yakuhai-call` |
| Implementation class | `YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy` |
| Family | defense + finite-horizon + hand-value + conservative call |
| Role | current overall-strength baseline |

`yakuhai-call`は、fresh holdout comparisonで当時のbaseline `combined`を上回ったdecisionに基づき昇格した。
current interpretationの根拠は
[lisjong #121のGate 2 decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5471486662)
に残す。historical measurement数値は本書へ重複転記しない。

current baselineはruntime profileへの自動配備を意味しない。RiichiLab等のexecution profile
mappingは`lisjong-arena`が所有し、strength statusとは独立に変更・検証する。

また、Learned PolicyやHandBelief experimentで別candidateが生成されても、bounded research resultだけで
current strength baselineを自動更新しない。promotionにはpurpose-appropriateなArena evaluationと明示的な
current-status更新が必要である。

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
| `FiniteHorizonCompletionPolicy` | `finite-horizon` | exact finite-horizon structural search | component comparator / possible research teacher-comparator | conditional 3-self-draw completion massをexact DPで比較。runtime costが高く、actual 3巡以内和了確率ではない |
| `GenbutsuDefenseFiniteHorizonValueAwarePolicy` | `combined` | defense + finite-horizon + lightweight value | predecessor strength baseline / comparator | 共通現物constraint、exact finite-horizon DP、ValueAware fallbackを合成。`yakuhai-call`より前のbaseline |
| `GenbutsuDefenseFiniteHorizonHandValueAwarePolicy` | `extended-combined` | defense + finite-horizon + hand value heuristic | no-call parent / historical causal comparator; not promoted | `combined`のvalue stageをHandValueAwareへ拡張。promotionせず、DP由来の高costを持つ |
| `YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy` | `yakuhai-call` | defense + finite-horizon + hand value + conservative call | **current strength baseline** | no-call parentに役牌Pon起点のstrict shanten-improving callを追加。generic call EVではなく、DP由来の高costを持つ |
| `KanCoverageYakuhaiCallPolicy` | — | deterministic coverage source | **HandBelief Stage 3 augmentation source; not in strength hierarchy** | winning action > legal kan > `yakuhai-call` delegate。kan / rinshan trajectoryをtraining distributionへ供給する目的で使用 |

ここでの`current role`はPolicy-strength / research management上の位置づけであり、public APIの安定性、
deprecation、runtime profile assignmentを表さない。implementation時のmodule / class docstringに
`experimental`とある場合も、current roleは本書のsnapshotを参照する。

## Training coverage source Policy — current Stage 3 role

`KanCoverageYakuhaiCallPolicy`は、HandBelief Stage 3 Entry Gateで確認されたkan / rinshan coverage holeへの
対応として追加された、**strength改善を目的としないdeterministic first-party coverage source**である。

```text
kan-capable coverage source
!= stronger Policy
!= current strength baseline
!= production recommendation
```

selection semanticsは次の優先順位で固定する。

```text
1. RonAction / TsumoAction
2. DaiminkanAction / AnkanAction / KakanAction
3. delegated normal-play decision (`yakuhai-call`)
```

合法性は常に`DecisionContext.legal_actions`を正本とし、Policy側でkan legalityを再判定しない。
複数kan候補が同時にlegalな場合は、semantic fieldだけから作るdeterministic total orderで1件を選ぶ。
kan種別間の固定順序はdeterminismを固定するimplementation choiceであり、麻雀上の優劣を意味しない。

### Qualification / population handoff

このPolicyは現在、単なる「将来測定予定」のsourceではない。
Arena側のbounded researchで次の順序まで進んでいる。

```text
Stage 3 Entry Gate
    -> kan / rinshan coverage hole identified

Arena #146 / #147
    -> KAN COVERAGE SOURCE QUALIFIED FOR MIX DESIGN

Arena #148 / #149
    -> MIX LOCKED — 12.5% AUGMENTATION

Arena #150
    -> PHASE10 SCALE SIGNAL
```

#146のqualificationでは、fresh first-party generation上でlegal kan opportunityからselected kan、
publicly confirmed kan、rinshanまでをaccountでき、coverage sourceとしてmix designへ進める根拠が得られた。

その後のpopulation-mix pilotでは、Stage 3 training recipeを次としてlockした。

```text
primary source       yakuhai-call
augmentation source  KanCoverageYakuhaiCallPolicy
augmentation         12.5% of seat slots
construction         at most 1 coverage-source seat / hanchan
seat balancing       canonical E/S/W/N balanced
split semantics      whole hanchan / TRAIN + VALIDATION / formal TESTなし
```

この12.5%は**Policy strength rankingではなくHandBelief training population recipe**である。
`KanCoverageYakuhaiCallPolicy`自体をbaseline、recommended gameplay、production Policyへ昇格させたものではない。

Phase 10では、このrecipeとselected sequential HandBelief familyを固定したままTRAIN data量を増やし、
Belief qualityへpositive scale signalが出ることを確認した。これもPolicy strength claimではなく、
**locked Stage 3 recipeでadditional training dataがcomponent quality改善へ変換された**というevidenceである。

## Learned Policy research status is separate from public Policy status

ArenaにはBehavior Cloning / Offline Q等のexperiment-local Learned Policy researchが存在するが、
これらのcheckpoint / experiment adapterは、現時点では本書のpublic Policy strength inventoryへ自動登録しない。

```text
Arena experiment-local model / checkpoint
!= lisjong public Policy
!= current strength baseline
!= production Policy
```

current Learned Policy研究では、simple Offline Q candidateのfailure diagnosisにより
**hand-progression degradation**がmechanism-level evidenceとして確認され、次のresearch axisでは
hand-progression structureを明示するbounded experimentへ進んでいる。

このresearch statusは、current strength baseline `yakuhai-call` を変更しない。
Learned Policy candidateがstable AI semantics / production Policyへ昇格する場合は、
Arena experiment-local ownershipからlisjong-owned stable contractへのpromotion boundaryを別途明示する。

## Representative evidence

代表的なdecision / evidenceだけを案内する。数値と完全な経緯はlink先のhistorical recordに残す。

| Policy / role | Representative reference |
| --- | --- |
| `yakuhai-call` current baseline | [fresh Gate 2 promotion decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5471486662)、[Arena evaluation wiring](https://github.com/lisbun/lisjong-arena/blob/main/docs/yakuhai-call-evaluation.md) |
| `combined` predecessor baseline | [fresh holdout promotion evidence](https://github.com/lisbun/lisjong/issues/121#issuecomment-5462935934) |
| `extended-combined` not promoted | [bounded Gate 1 decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5466162346)、[Arena historical evaluation document](https://github.com/lisbun/lisjong-arena/blob/main/docs/extended-combined-evaluation.md) |
| `finite-horizon` component comparator | [10,000-game follow-up decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5439919623) |
| `hand-value-aware` evaluated component candidate | [4,000-game follow-up interpretation](https://github.com/lisbun/lisjong/issues/121#issuecomment-5431486646) |
| `KanCoverageYakuhaiCallPolicy` Stage 3 coverage source | [Arena #146 qualification](https://github.com/lisbun/lisjong-arena/issues/146)、[Arena #148 population mix lock](https://github.com/lisbun/lisjong-arena/issues/148)、[Arena #150 Phase 10 scale study](https://github.com/lisbun/lisjong-arena/issues/150) |
| foundational / unevaluated exports | implementation / contract baselinesであり、本書ではformal strength claimを行わない |

current Policy-vs-Policy ABBB runは、Arenaのversioned immutable artifactとreaggregationを
measurement boundaryとして利用する。artifact対応範囲と制約は
[Arena policy](https://github.com/lisbun/lisjong-arena/blob/main/docs/policy-strength-evaluation.md#measurement-source-of-truth)
を参照する。過去runにartifactが存在しない場合、存在するものとして扱わない。

## Status update workflow

今後のPolicy workは終了条件のあるbounded Issue / PRとして進める。

```text
concrete stable Policy work
    -> bounded lisjong Issue / implementation PR

experiment-local ML / diagnostic work
    -> bounded lisjong-arena research Issue / artifact

strength evaluation
    -> bounded lisjong-arena Issue / Arena artifact / decision

current interpretation changed
    -> relevant bounded PRで本書を更新
```

新しいPolicyやfuture candidateの存在だけを理由に、long-lived Policy-strength tracking Issueを作成しない。
research experimentがpositiveでもstable/public Policyへのpromotionを自動化しない。
過去のwork / decision historyは対応するclosed Issue / PRを参照する。
