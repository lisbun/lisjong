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
| Arena identity | `mechanism-riichi-defense` |
| Implementation class | `MechanismRiichiDefenseYakuhaiCallPolicy` |
| Family | `yakuhai-call` + bounded mechanism-based riichi defense |
| Role | **current heuristic strength baseline (protocol-scoped)** |
| Evidence scope | `ABBB` / `4p-red-single` |

`MechanismRiichiDefenseYakuhaiCallPolicy` は Arena #217 の strict-read / provenance-locked 10,000-game Gate 2 評価で `yakuhai-call` に対する positive delta が確定し、`mechanism-riichi-defense` として Arena #219 経由で curated `POLICY_CATALOG` alias に登録された。この promotion は **`ABBB` / `4p-red-single` protocol の評価結果に限定** される。hanchan superiority、universal/generalized superiority、production default、runtime profile default、Champion 等の広い主張はこの evidence からは行わない。

`yakuhai-call`（`YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy`）は predecessor / historical anchor として残る。過去の fresh holdout comparison に基づく promotion evidence、および Arena #211 等で明示的に retain されている research source としての役割は変更されない。

```text
current gameplay strength baseline promotion
!= migration of locked teacher/source populations
```

今回の strength baseline promotion は、Arena #211 の `Arm Y`（exact retained `yakuhai-call` source）や `Arm R`（exact qualified RiichiLab source）等、既存 research protocol が明示的に固定した teacher/source population を自動的に移行するものではない。

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
| `YakuhaiCallGenbutsuDefenseFiniteHorizonHandValueAwarePolicy` | `yakuhai-call` | predecessor strength baseline / historical anchor | conservative Yakuhai call を追加。#211等で明示的にlockされたresearch source としては継続利用 |
| `MechanismRiichiDefenseYakuhaiCallPolicy` | `mechanism-riichi-defense` | **current promoted heuristic strength baseline (`ABBB` / `4p-red-single` scope)** | exact `yakuhai-call` parentにbounded mechanism-based riichi defenseを追加。Arena #217 Gate 2で昇格、Arena #219でcurated alias登録 |
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

## `MechanismRiichiDefenseYakuhaiCallPolicy` — promotion evidence

`MechanismRiichiDefenseYakuhaiCallPolicy` は predecessor baseline `yakuhai-call` を parent とする bounded riichi-defense heuristic である。

Current status:

```text
implementation         COMPLETE / merged
strength evaluation    Gate 2 CONFIRMED POSITIVE (Arena #216 / #217)
protocol scope         ABBB / 4p-red-single
Arena curated alias     mechanism-riichi-defense (Arena #219)
promotion               YES — protocol-scoped heuristic strength baseline
```

Arena #217 (10,000 games, strict artifact readback / canonical re-aggregation / provenance validation済み) は次を確定した。

```text
mean delta      +109.2
95% interval    [+73.0, +145.5]
```

この結果は `ABBB` / `4p-red-single` protocol に限定された評価であり、hanchan や他 protocol への一般化、production/runtime default 化、Champion 指定を含意しない。

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

この研究進捗は current strength baseline を変更しない。Arena #211 は `Arm Y`（exact retained `yakuhai-call` source）を引き続き明示的に固定しており、今回の strength baseline promotion はこの locked research source population を移行しない。stable AI semantics / production Policy へ昇格する場合は、Arena experiment-local ownership から `lisjong` stable contract への promotion boundary を別途明示する。

Parent roadmap: [lisjong-project #45](https://github.com/lisbun/lisjong-project/issues/45)

## Representative evidence

| Policy / role | Representative reference |
| --- | --- |
| `yakuhai-call` predecessor / historical anchor | [lisjong #121 Gate 2 decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5471486662) |
| `combined` predecessor baseline | [lisjong #121 historical promotion evidence](https://github.com/lisbun/lisjong/issues/121#issuecomment-5462935934) |
| `extended-combined` not promoted | [lisjong #121 bounded Gate 1 decision](https://github.com/lisbun/lisjong/issues/121#issuecomment-5466162346) |
| `MechanismRiichiDefenseYakuhaiCallPolicy` current heuristic baseline (`ABBB` / `4p-red-single`) | [lisjong-arena #217](https://github.com/lisbun/lisjong-arena/issues/217) — Gate 2 CONFIRMED POSITIVE; [lisjong-arena #216](https://github.com/lisbun/lisjong-arena/issues/216) — prior evaluation step; [lisjong-arena #219](https://github.com/lisbun/lisjong-arena/issues/219) — curated `mechanism-riichi-defense` alias |
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

Do not create a long-lived Policy-strength tracking Issue merely because a future candidate exists。Negative / inconclusive evidence remains historical evidence; it does not need to be copied into this document beyond the current role it establishes。
