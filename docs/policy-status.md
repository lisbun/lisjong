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

Snapshot date: **2026-09-25**

## Current strength baseline

| Field | Current value |
| --- | --- |
| Arena identity | `placement-aware-speed-call` |
| Implementation class | `PlacementAwareSpeedCallPolicy` |
| Family | placement-aware speed-call heuristic over `targeted-honor-release-terminal-progression` |
| Role | **current Heuristic Champion / heuristic strength baseline (protocol-scoped)** |
| Evidence scope | family-internal `arena-heuristic-candidate-aabb-half-v1` / `4p-red-half` |

`PlacementAwareSpeedCallPolicy` は lisjong-project #80 により current Heuristic Champion へ昇格する。authoritative promotion evidence は Arena #375 であり、lisjong revision `2a9debebdbbe4d10841fa4371a6cf6bf19ce9de1` の exact candidate を exact incumbent `TargetedHonorReleaseTerminalProgressionPolicy` と frozen family-internal protocol `arena-heuristic-candidate-aabb-half-v1` で比較した。

```text
classification          CANDIDATE SUPERIOR
result identity         2936df10acc7d171b73451fe2ef3dd95ddd3d7ae1b626e4479db94c9b6c180e2
```

数値の詳細は下記 [`PlacementAwareSpeedCallPolicy` section](#placementawarespeedcallpolicy--current-heuristic-champion) と Arena #375 を正本とする。評価済み revision は strength evidence のanchorとして保持し、same class name / module pathだけをcontinuityの根拠にはしない。

`targeted-honor-release-terminal-progression` は predecessor Heuristic Champion / historical comparator として残す（Arena #270 / project #67 evidenceは変更しない）。`mechanism-riichi-defense` はさらに前の predecessor Heuristic Champion / historical strength baseline、`yakuhai-call` はさらに前の predecessor / historical anchor として、明示的にlockされたresearch-source roleを維持する。

Overall Champion designationはこの文書では確立しない。lisjong-project #69の条件が満たされる場合のOverall designation（`INHERITED`）は、project #80 completion recordで別途扱う。`PlacementAwareSpeedCallPolicy` 自身は direct cross-family superiority を統計的に示していない。

```text
Heuristic Champion promotion
!= migration of locked teacher/source populations
!= Overall Champion designation
!= direct cross-family superiority
!= production/runtime default
```

Arena #211 `Arm Y` / `Arm R` 等のhistorically locked research populationは、このdesignationによってsilent migrationしない。Runtime profile deploymentも引き続き別責務とする。

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
| `MechanismRiichiDefenseYakuhaiCallPolicy` | `mechanism-riichi-defense` | **predecessor Heuristic Champion / historical strength baseline** | Arena #217 Gate 2で`yakuhai-call`を上回り初代Heuristic Championとなったが、Arena #270 / project #67で後継へ交代 |
| `OpenHandYakuAwareCallPolicy` | — | **evaluated experimental candidate; inconclusive; not promoted** | Tanyao / Honitsu / Chinitsu-compatible strictly-improving Chi/Pon を追加 |
| `CheapFarGuardOpenHandYakuAwareCallPolicy` | — | **evaluated experimental candidate; inconclusive; not promoted** | selected cheap+far Chi/Pon のみ Pass へ置換 |
| `TerminalShantenProgressionMechanismRiichiDefensePolicy` | — | **evaluated experimental candidate; inconclusive; not promoted** | exact `mechanism-riichi-defense` parentの all-zero completion branch だけを expected terminal shanten 最小化へ置換 |
| `TargetedHonorReleaseTerminalProgressionPolicy` | `targeted-honor-release-terminal-progression` | **predecessor Heuristic Champion / historical comparator** | far closed PUSH / all-zeroのtargeted honor-release conflictだけをexact R5で再判定。Arena #270でexact incumbentにCONFIRMED POSITIVE、project #67で昇格したが、Arena #375 / project #80で後継へ交代 |
| `HandValueTradeoffTargetedHonorReleasePolicy` | — | **experimental candidate; evaluation pending** | predecessor Heuristic Champion `TargetedHonorReleaseTerminalProgressionPolicy` のconfirmed `R5_HONOR_ONLY_SWITCH`を保持し、それ以外のordinary discardへ#175 HandValue v2を合成 |
| `PlacementAwareSpeedCallPolicy` | `placement-aware-speed-call` | **current Heuristic Champion / current heuristic strength baseline** | `TargetedHonorReleaseTerminalProgressionPolicy`をexact parentとし、Tanyao / Honitsu route確定型のスピード鳴き、役のない副露手のroute保持打牌、オーラス点数状況mode（トップ目SPEED / 大差ラス目VALUE）を追加（[#199](https://github.com/lisbun/lisjong/issues/199)）。Arena #375の`arena-heuristic-candidate-aabb-half-v1`でexact incumbentに対しCANDIDATE SUPERIOR、project #80で昇格 |
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

## `TerminalShantenProgressionMechanismRiichiDefensePolicy` — evaluated broad progression candidate

then-current promoted baseline `MechanismRiichiDefenseYakuhaiCallPolicy` を exact parent とし、FiniteHorizon completion mass が全 root discard candidate で 0 の通常打牌 branch を horizon=3 の exact adaptive expected-terminal-shanten progression へ置換した broad hypothesis である。

Arena #252 では exact #170 semantics を変更せず、100 paired seed blocks / 400 games per arm の passive-x3 development evaluationを実施した。

```text
mean paired score delta  +123
95% interval             [-118.957642, +364.957642]
classification           PROGRESSION DEVELOPMENT INCONCLUSIVE
```

したがって broad progression candidate は `mechanism-riichi-defense` を置き換えておらず、result-drivenなseed追加・horizon変更・objective変更も行っていない。

Representative evidence: [Arena #252 final result](https://github.com/lisbun/lisjong-arena/issues/252#issuecomment-5664824746)

## `TargetedHonorReleaseTerminalProgressionPolicy` — predecessor Champion evidence

Arena #256 で broad #169 のdisagreementが特定のshapeへ集中したことを受け、#174はexact parentを変更せず、far closed PUSH / all-zeroの限定された suited-discard / honor-release conflictだけをexact R5で再判定する。

Arena #263の100-block development populationはINCONCLUSIVEだったためconfirmation observationには使用せず、Arena #270で unchanged candidate を exact incumbent `mechanism-riichi-defense` と fresh independent 2,200 paired seed blocksで評価した。

```text
seed blocks             2,200
games                   17,600
mean paired delta       +49.54545454545455
sample SD               830.1450958429708
SE                      17.698752911281826
95% interval            [+14.855898839342167, +84.23501025156693]
classification          TARGETED HONOR-RELEASE CONFIRMED POSITIVE
paired result identity  04cc12834b25365e3146dde64115162378b329117ba007dd368abb1affcb95dc
```

Project #67では、このexact lineageを次のHeuristic Championへ昇格した（その後 Arena #375 / project #80 で `PlacementAwareSpeedCallPolicy` へ交代し、現在は predecessor Heuristic Champion / historical comparator）。

```text
Heuristic Champion
targeted-honor-release-terminal-progression
TargetedHonorReleaseTerminalProgressionPolicy
```

このfamily designationはcontrolled evidenceのscopeに限定する。Overall Champion、hanchan superiority、RiichiLab superiority、production deployment、universal Mahjong superiorityは確立しない。

Evaluated strength revision は `f29d129c67e5232d06563c6e457754377734ed14`。current-main semantic continuity は、#177 / PR #178 の明示的 behavior-preserving extraction と、terminal-progression Policyを含む byte-identical deterministic before/after harnessによって確認する。

Implementation hypothesis: [lisjong #174](https://github.com/lisbun/lisjong/issues/174)  
Promotion evidence: [Arena #270](https://github.com/lisbun/lisjong-arena/issues/270)  
Champion governance: [lisjong-project #67](https://github.com/lisbun/lisjong-project/issues/67)

## `PlacementAwareSpeedCallPolicy` — current Heuristic Champion

[#199](https://github.com/lisbun/lisjong/issues/199) / PR #202（merge commit `2a9debebdbbe4d10841fa4371a6cf6bf19ce9de1`）で追加した、当時のcurrent Heuristic Champion `TargetedHonorReleaseTerminalProgressionPolicy` をexact parentとするcandidateである。オーラス点数状況modeは1局評価では発動しないため、#202の1局development screen（INCONCLUSIVE）はPolicy全体の評価とみなさない。

Arena #375では、exact lisjong revision `2a9debebdbbe4d10841fa4371a6cf6bf19ce9de1` のcandidateと当時のcurrent Heuristic Championを、frozen family-internal protocol `arena-heuristic-candidate-aabb-half-v1`（`4p-red-half`、AABB 4 rotations / seed、primary metric = ウマオカ込み最終スコア: 25000/30000、uma +30/+10/-10/-30、oka +20）で比較した。

```text
protocol                arena-heuristic-candidate-aabb-half-v1
seed blocks             100
hanchan                 400
mean D                  +5.6125
sample SD               17.87341802705059
SE                      1.787341802705059
95% interval            [+2.109310066698084, +9.115689933301915]
block signs (+ / 0 / -) 44 / 32 / 24
classification          CANDIDATE SUPERIOR
result identity         2936df10acc7d171b73451fe2ef3dd95ddd3d7ae1b626e4479db94c9b6c180e2
```

remoteのclassification / result identityはlocal strict verificationと一致し、artifact SHA-256 verificationもpassした。

Arena #375の記録自体は評価結果だけを表し、Champion promotion / governanceはArena protocolの外にある。この結果をpromotion evidenceとして、lisjong-project #80で `PlacementAwareSpeedCallPolicy` を current Heuristic Champion へ昇格した。

```text
Heuristic Champion
placement-aware-speed-call
PlacementAwareSpeedCallPolicy
```

このfamily designationはfamily-internalな半荘controlled evidenceのscopeに限定する。Overall Champion、direct cross-family superiority、RiichiLab superiority、production deploymentは確立しない。Overall designationの扱いはproject #80 completion recordを正本とする。

- Implementation hypothesis: [lisjong #199](https://github.com/lisbun/lisjong/issues/199)
- Evaluation evidence: [Arena #375](https://github.com/lisbun/lisjong-arena/issues/375)
- Status update: [lisjong #204](https://github.com/lisbun/lisjong/issues/204), [lisjong #207](https://github.com/lisbun/lisjong/issues/207)
- Champion governance: [lisjong-project #80](https://github.com/lisbun/lisjong-project/issues/80)

## `MechanismRiichiDefenseYakuhaiCallPolicy` — predecessor Champion evidence

`MechanismRiichiDefenseYakuhaiCallPolicy` は predecessor baseline `yakuhai-call` を parent とする bounded riichi-defense heuristic である。

Historical status:

```text
implementation         COMPLETE / merged
Gate 1                 INCONCLUSIVE / Gate 2 eligible (Arena #216)
Gate 2                 CONFIRMED POSITIVE (Arena #217)
protocol scope         ABBB / 4p-red-single
Arena curated alias    mechanism-riichi-defense (Arena #219)
Champion role          predecessor Heuristic Champion after project #67
```

Arena #217 (10,000 games, strict artifact readback / canonical re-aggregation / provenance validation済み) は次を確定した。

```text
mean delta      +109.2
95% interval    [+73.0, +145.5]
```

このhistorical resultは初代Heuristic Champion establishment (#66) の根拠となった。Arena #270 / project #67で後継が昇格した後も、`mechanism-riichi-defense` はpredecessor / historical comparatorとして保持する。

Implementation / original hypothesis: [lisjong #163](https://github.com/lisbun/lisjong/issues/163)

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
| `PlacementAwareSpeedCallPolicy` current Heuristic Champion | [lisjong-arena #375](https://github.com/lisbun/lisjong-arena/issues/375) — `arena-heuristic-candidate-aabb-half-v1` CANDIDATE SUPERIOR (result identity `2936df10acc7d171b73451fe2ef3dd95ddd3d7ae1b626e4479db94c9b6c180e2`); [lisjong-project #80](https://github.com/lisbun/lisjong-project/issues/80) — family promotion governance |
| `TargetedHonorReleaseTerminalProgressionPolicy` predecessor Heuristic Champion | [lisjong-arena #270](https://github.com/lisbun/lisjong-arena/issues/270) — independent confirmation CONFIRMED POSITIVE; [lisjong-project #67](https://github.com/lisbun/lisjong-project/issues/67) — family promotion governance |
| `MechanismRiichiDefenseYakuhaiCallPolicy` predecessor Heuristic Champion | [lisjong-arena #217](https://github.com/lisbun/lisjong-arena/issues/217) — Gate 2 CONFIRMED POSITIVE; [lisjong-project #66](https://github.com/lisbun/lisjong-project/issues/66) — initial Heuristic Champion establishment; [lisjong-arena #219](https://github.com/lisbun/lisjong-arena/issues/219) — curated alias |
| `OpenHandYakuAwareCallPolicy` | [Arena #196](https://github.com/lisbun/lisjong-arena/issues/196) — bounded strength inconclusive |
| `CheapFarGuardOpenHandYakuAwareCallPolicy` | [lisjong #161](https://github.com/lisbun/lisjong/issues/161#issuecomment-5622890134) — `CHEAP-FAR GUARD INCONCLUSIVE` |
| `TerminalShantenProgressionMechanismRiichiDefensePolicy` | [lisjong #169](https://github.com/lisbun/lisjong/issues/169) — experimental candidate; strength evaluation blocked on exact-safe performance work |
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
