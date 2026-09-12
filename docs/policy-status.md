# Policy current status

## Purpose

本書は、`lisjong` が公開する Policy implementation の **current role** を示す repository-owned snapshot である。

```text
lisjong Policy status
    = current interpretation / current snapshot

lisjong-arena evaluation artifact
    = measurement source of truth

bounded GitHub Issue / PR
    = individual work / decision history
```

本書は historical evaluation log ではない。過去 run の数値や全経緯を複製せず、現在の役割と代表的 evidence だけを保持する。Policy strength comparison の規律は Arena-owned の [Policy strength evaluation policy](https://github.com/lisbun/lisjong-arena/blob/main/docs/policy-strength-evaluation.md) を正本とする。

Snapshot date: **2026-09-12**

## Current strength baseline

| Field | Current value |
| --- | --- |
| Arena identity | `yakuhai-call` |
| Implementation class | `YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy` |
| Family | defense + finite-horizon + hand-value + conservative call |
| Role | **current overall-strength baseline** |

`yakuhai-call` は fresh holdout comparison に基づいて昇格した current baseline である。Learned Policy、HandBelief、bounded heuristic candidate が存在しても、それだけで baseline は変更しない。promotion には purpose-appropriate な Arena evaluation と明示的な status update が必要である。

Runtime profile への配備は別責務であり、RiichiLab 等の execution profile mapping は `lisjong-arena` が所有する。

## Public Policy inventory

`Arena identity` が `—` の Policy は public implementation ではあるが、current Arena `single_round_compare` curated catalog へ登録されていない場合がある。

| Public Policy | Arena identity | Current role | Notes |
| --- | --- | --- | --- |
| `MinimalPolicy` | — | foundational / boundary validation | deterministic total-order selector。strength 目的ではない |
| `ShantenPolicy` | — | foundational comparator | shanten 中心 |
| `UkeirePolicy` | — | foundational comparator | current ukeire を追加 |
| `TwoStepUkeirePolicy` | `two-step` | historical structural baseline / cheap comparator | second-step ukeire まで比較 |
| `GenbutsuDefenseTwoStepUkeirePolicy` | — | component comparator | 被立直時の共通現物優先 |
| `ValueAwareTwoStepUkeirePolicy` | — | component comparator | lightweight value tie-break |
| `HandValueAwareTwoStepUkeirePolicy` | `hand-value-aware` | evaluated component candidate | 役牌・dora・赤dora・軽量 yaku route |
| `FiniteHorizonCompletionPolicy` | `finite-horizon` | component comparator | exact finite-horizon structural search。runtime cost が高い |
| `GenbutsuDefenseFiniteHorizonValueAwarePolicy` | `combined` | predecessor strength baseline / comparator | `yakuhai-call` より前の baseline |
| `GenbutsuDefenseFiniteHorizonHandValueAwarePolicy` | `extended-combined` | historical causal comparator; not promoted | hand-value 拡張。promotion なし |
| `YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy` | `yakuhai-call` | **current strength baseline** | conservative Yakuhai call を追加 |
| `MechanismRiichiDefenseYakuhaiCallPolicy` | — | experimental strength candidate; not evaluated or promoted | exact `yakuhai-call` parentにbounded mechanism-based riichi defenseを追加。#163のevaluation対象 |
| `OpenHandYakuAwareCallPolicy` | — | **evaluated experimental candidate; inconclusive; not promoted** | Tanyao / Honitsu / Chinitsu-compatible strictly-improving Chi/Pon を追加 |
| `CheapFarGuardOpenHandYakuAwareCallPolicy` | — | **evaluated experimental candidate; inconclusive; not promoted** | selected cheap+far Chi/Pon のみ Pass へ置換 |
| `KanCoverageYakuhaiCallPolicy` | — | **HandBelief Stage 3 augmentation source; not in strength hierarchy** | kan / rinshan coverage 用 deterministic source |

`current role` は Policy-strength / research management 上の位置づけであり、public API stability、deprecation、runtime profile assignment を表さない。

## Recent bounded call-policy evidence

### `OpenHandYakuAwareCallPolicy`

Arena #196 の fresh bounded screen では `yakuhai-call` に対して **INCONCLUSIVE** だった。したがって、open-yaku call 拡張は baseline を置き換えていない。

Representative reference: [Arena #196](https://github.com/lisbun/lisjong-arena/issues/196)

### `CheapFarGuardOpenHandYakuAwareCallPolicy`

`OpenHandYakuAwareCallPolicy` を exact parent とし、selected call が conservative に `cheap AND far` の場合だけ Pass へ置換する bounded hypothesis を検証した。

Fresh 400-game screen の formal outcome は:

```text
CHEAP-FAR GUARD INCONCLUSIVE
```

positive mean direction は観測されたが locked 95% interval が zero を跨いだため、promotion も `yakuhai-call` との follow-up strength comparison も開始していない。

Representative reference: [lisjong #161 final result](https://github.com/lisbun/lisjong/issues/161#issuecomment-5622890134)

## Current unevaluated strength candidate

`MechanismRiichiDefenseYakuhaiCallPolicy` は current baseline `yakuhai-call` を parent とする bounded riichi-defense candidate である。implementationは存在するが、現時点では strength result は未確定であり promotion されていない。

Current work: [lisjong #163](https://github.com/lisbun/lisjong/issues/163)

## `KanCoverageYakuhaiCallPolicy` — training coverage role

`KanCoverageYakuhaiCallPolicy` は strength 改善を目的としない HandBelief training-population source である。

```text
kan-capable coverage source
!= stronger Policy
!= current strength baseline
!= production recommendation
```

Current Stage 3 recipe:

```text
primary source       yakuhai-call
augmentation source  KanCoverageYakuhaiCallPolicy
augmentation         12.5% of seat slots
construction         at most 1 coverage-source seat / hanchan
seat balancing       canonical E/S/W/N balanced
split semantics      whole hanchan / TRAIN + VALIDATION / formal TESTなし
```

これは Policy strength ranking ではなく、HandBelief population design である。Current HandBelief research state は project-wide parent [lisjong-project #36](https://github.com/lisbun/lisjong-project/issues/36) を参照する。

## Learned Policy research is separate from public Policy status

Arena の Behavior Cloning / Offline Q 等は experiment-local research であり、その checkpoint / adapter は public Policy inventory へ自動昇格しない。

```text
Arena experiment-local model / checkpoint
!= lisjong public Policy
!= current strength baseline
!= production Policy
```

Current Learned Policy research は P8 の **data scale / data source** 軸まで進んでいる。

```text
Arena #190  COMPLETE — same-source data-scale signal
Arena #170  COMPLETE — bounded RiichiLab strong-bot corpus acquisition
Arena #203  COMPLETE — player-safe reconstruction / supervision qualification
Arena #211  CURRENT  — matched flat-BC RiichiLab source pilot
```

この研究進捗は current strength baseline `yakuhai-call` を変更しない。stable AI semantics / production Policy へ昇格する場合は、Arena experiment-local ownership から `lisjong` stable contract への promotion boundary を別途明示する。

Parent roadmap: [lisjong-project #45](https://github.com/lisbun/lisjong-project/issues/45)

## Representative evidence

| Policy / role | Representative reference |
| --- | --- |
| `yakuhai-call` current baseline | [lisjong #121 Gate 2 decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5471486662) |
| `combined` predecessor baseline | [lisjong #121 historical promotion evidence](https://github.com/lisbun/lisjong/issues/121#issuecomment-5462935934) |
| `extended-combined` not promoted | [lisjong #121 bounded Gate 1 decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5466162346) |
| `MechanismRiichiDefenseYakuhaiCallPolicy` | [lisjong #163](https://github.com/lisbun/lisjong/issues/163) — current unevaluated bounded candidate |
| `OpenHandYakuAwareCallPolicy` | [Arena #196](https://github.com/lisbun/lisjong-arena/issues/196) — bounded strength inconclusive |
| `CheapFarGuardOpenHandYakuAwareCallPolicy` | [lisjong #161](https://github.com/lisbun/lisjong/issues/161#issuecomment-5622890134) — `CHEAP-FAR GUARD INCONCLUSIVE` |
| `KanCoverageYakuhaiCallPolicy` | [Arena #146](https://github.com/lisbun/lisjong-arena/issues/146), [#148](https://github.com/lisbun/lisjong-arena/issues/148), [#150](https://github.com/lisbun/lisjong-arena/issues/150) |

Historical measurement numbers belong in the corresponding Issue / immutable Arena artifact rather than this snapshot.

## Status update workflow

```text
concrete stable Policy work
    -> bounded lisjong Issue / implementation PR

experiment-local ML / diagnostic work
    -> bounded lisjong-arena research Issue / artifact

strength evaluation
    -> bounded lisjong-arena Issue / Arena artifact / decision

current interpretation changed
    -> update this snapshot
```

Do not create a long-lived Policy-strength tracking Issue merely because a future candidate exists. Negative / inconclusive evidence remains historical evidence; it does not need to be copied into this document beyond the current role it establishes.
