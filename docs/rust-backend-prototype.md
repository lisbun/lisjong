# Rust backend試作: numeric shanten core（Issue #213）

牌効率計算の律速を計測し、1つの計算境界だけをopt-in Rust backendとして試作・評価した記録である。
アルゴリズムと麻雀上の意味は変更しない。default backendはPythonのままである。

## 1. 計測条件（固定）

| 項目 | 値 |
| --- | --- |
| lisjong base | `main` `0408d62573a67b3d9f10f9ff8dc0312ab129ae8f`（#210 merge後） |
| decision採取 | lisjong-arena `9f748cd`（`LocalGameRunner`, RiichiEnv 0.4.10, `4p-red-half`, seed 0, 4席同一Policy） |
| OS / CPU | Windows 11 Home 10.0.26200 / Intel64 Family 6 Model 189（8 logical CPU） |
| Python | CPython 3.14.7（MSC v.1944, 64 bit）、通常版（free-threadedではない） |
| worker | 1 process（single worker）。新しいthread pool / 分散実行は使わない |
| Rust | rustc / cargo 1.98.1（`native/rust-toolchain.toml`で固定）、target `x86_64-pc-windows-msvc`、release（`opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"`） |
| binding | PyO3 0.29.2 + maturin 1.15.0（build backend） |

固定decision列は、ArenaのLocalGameRunnerで各Policyを1半荘実行し、Policyへ渡った
`(DecisionContext, InternalAction)`をそのままpickleしたlocal artifactである（repositoryへはcommitしない）。

| Policy（identity） | decisions | sha256 |
| --- | ---: | --- |
| `TwoStepUkeirePolicy`（`two-step`、一段階・二段階受入） | 684 | `31d0448e…b8a4cf` |
| `PlacementAwareSpeedCallPolicy`（`placement-aware-speed-call`、current Heuristic Champion） | 708 | `275ca281…b8ea` |
| `Kobalab0004ReferencePolicy`（#211参照Policy） | 630 | `f5782dc3…87a8` |

採取時の半荘wall-clock（Python backend、Arena経由、profilerなし）：two-step 59.8s、
champion 623.7s、0004参照 2.9s。

## 2. profile結果と対象選定

`cProfile`は箇所の特定だけに使った（速度比較には使わない）。

**two-step**（profile下110.9s）：

- `StructuralShantenEvaluator.calculate` cumulative 108.2s のうち、`calculate_shanten` は35.5s。
  残りは主に`_count_tile_types`（Tile / TileType dataclass・enumの`hash`、`dict.get`）による
  cache key生成であり、Pythonのobject変換コストが支配的だった。
- 通常形backend `calculate_standard_shanten` は cumulative 14.1s（約13%）。

**champion**（profile下1105.6s）：

- numeric shanten core `shanten._shanten_from_valid_counts` が cumulative 737.0s（**約67%**）、
  25.8M call。
  - 通常形 `_lookup_shanten.calculate_standard_shanten`: 450.1s（約41%）
  - 閉じた13 / 14枚の七対子・国士 `_meldless_special_shanten`: 282.1s（約25%）
- 呼び出し元はPolicy側の探索（`TerminalShantenProgression…._search` 等）と
  FiniteHorizonのcount-native DPで、入力はすでにcanonical 34-countである。

**選定した対象**：count-native numeric shanten core
（`shanten._shanten_from_valid_counts(counts, concealed_tile_count)`）。

- 理由：全Policy共通のleafで、championの主要cost（約67%）を占め、入力がすでに34-countなので
  変換コストが小さい。Tile / Policy / hidden informationに触れない。
- 境界：34 count + 純手牌枚数 → int。通常形は同じ`_shanten_table.bin`と、`_lookup_shanten`が
  構築したresource-state combine / penalty tableをbytesで受け取る。native側はlookup・combine loopと
  七対子・国士とのmin dispatchだけを実行する。
- 最大改善余地（Amdahl）：champion判断時間の約67%を0にしても約3.0倍が上限。two-stepは
  shanten core全体でも約32%（上限約1.5倍）で、その主因はRustの対象外（Tile変換）である。
- 対象外：Policy側の探索構造、`structural_efficiency`のTile-based API、structural predicate、
  engine / I/O。

## 3. 実装前に固定した採用判断基準

以下はPolicy判断全体・小規模対局の計測前に記録した。（kernel単体のspot-checkは基準記録前に
一度実行している：0004参照の実入力でPython 12.8µs/call → Rust 0.42µs/call。）

- **同値性（必須）**：differential test不一致0、固定decision再生・小規模対局の行動列が
  Python backendと完全一致。Rust選択下でfull test suiteがpass。
- **本採用を推奨**：single workerのchampion Policy判断全体（変換・呼び出し込み）で所要時間
  **30%以上短縮**（半荘/時 1.43倍以上）、かつworkerあたりの追加メモリ **20 MB以下**、
  起動（import + table load）の追加 **100 ms以下**、かつ利用環境（Windows / AWS Linux）へ
  compiler不要の配布手段がある。
- **追加検討**：上記の性能条件は満たすが、配布・対応環境の実証が不足する。
- **見送り**：champion判断全体の短縮が15%未満、または保守負担が効果を上回る。

## 4. 結果

すべて同一マシン・single process・profilerなし。backendはprocessごとに
`LISJONG_SHANTEN_BACKEND`で選択し、同じdecision列・同じcache条件（Policy内部の
decision-local cacheのみ、cross-decision cacheなし）で比較した。
計測tool：`tools/benchmark_tile_efficiency.py`（`policy` / `kernel`）。

### 4.1 対象計算単体（kernel）

各Policyの固定decision再生中にnumeric shanten coreへ実際に渡された入力列を記録し、
同一process内で両backendへ同じ入力列を流した（Pythonは既存のlookup backend、既存最適化を維持）。

| 入力元 | 入力数（distinct） | 値の不一致 | Python µs/call | Rust µs/call | 比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0004参照（全630 decision） | 101,312（74,137） | 0 | 18.33 | 0.515 | 35.6x |
| two-step（全684 decision） | 663,235（528,512） | 0 | 17.31 | 0.591 | 29.3x |
| champion（先頭80 decision） | 5,244,566（5,098,211） | 0 | 13.42 | 0.347 | 38.7x |

中央値（repeat 5 / 5 / 3）。Rust側は34-count extractionとPyO3呼び出しを含む。
native table構築（artifact parse）は3–5 ms。

### 4.2 Policy判断全体（変換・呼び出し込み）

`choose_action()`の合計時間。1 passごとにPolicy instanceを作り直す。
1 pass目がcold（process起動直後）、以降がwarm。

| Policy | backend | pass合計 [s] | mean / median / p95 [ms/decision] | 記録actionとの不一致 |
| --- | --- | --- | --- | ---: |
| champion | Python | 534.2, 569.1 | 754.6 / 74.2 / 748.3（pass 1） | 0 |
| champion | Rust | 153.0, 167.5 | 216.1 / 27.3 / 422.1（pass 1） | 0 |
| two-step | Python | 26.73, 27.02, 26.40 | 39.1 / 8.3 / 212.0 | 0 |
| two-step | Rust | 16.81, 17.18, 16.83 | 24.6 / 4.9 / 130.9 | 0 |
| 0004参照 | Python | 1.87, 1.93, 1.94, 1.95, 1.94 | 3.0 / 2.8 / 7.0 | 0 |
| 0004参照 | Rust | 0.57, 0.56, 0.61, 0.57, 0.57 | 0.9 / 0.9 / 2.1 | 0 |

- champion：**約71%短縮（約3.5倍）**。profile上のAmdahl上限（約3.0倍）を上回ったのは、
  cProfileがPython関数呼び出しの多いcoreを過大に計上していたためと考えられる。
- two-step：約37%短縮（約1.6倍）。残りの主因はTile-based cache key生成で、Rust化の対象外。
- 0004参照：約70%短縮（約3.4倍）。
- Rust選択時は、native coreの呼び出し回数（champion 2 passで51,650,404回）で
  Rust pathが実際に実行されたことを確認した。
- champion の最大decision時間は Python 29.3s → Rust 7.7s。

### 4.3 小規模対局（Arena、半荘1回）

lisjong-arena `9f748cd`の`LocalGameRunner`、`4p-red-half`、seed 0、4席同一Policy。
Policy以外の条件（seed、engine、Arena revision、worker数 1）は固定。

| Policy | Python [s] | Rust [s] | 半荘/時（Python → Rust） | 行動列 |
| --- | ---: | ---: | --- | --- |
| two-step | 27.71 | 17.62 | 130 → 204 | 4席とも完全一致 |
| champion | 544.88 | 158.92 | 6.6 → 22.7 | 4席とも完全一致 |

行動列の比較は席ごとの`(DecisionContext, InternalAction)`列で行い、両backendで
DecisionContext列・action列・最終得点がすべて一致した。同じstep内の複数席の記録順は
backendに関係なくrunごとに入れ替わり得るため、全席を連結した列では比較していない。
（1回だけの計測であり、半荘時間のばらつきは評価していない。採取時の最初のtwo-step半荘は
59.8sだったが、同時に別のprofileを実行していたため比較には使わない。）

### 4.4 起動・メモリ・worker

新しいprocessで`import` + 初回shanten計算を10回計測した（中央値）。

| 項目 | Python | Rust |
| --- | ---: | ---: |
| import + 初回call | 86 ms | 81 ms |
| うちtable load（初回call / import時） | 8.3 ms（初回call時） | import時に構築、初回callは0.016 ms |
| peak RSS（起動直後） | 35.2 MB | 31.4 MB |
| peak RSS（champion 2 pass後） | 668.2 MB | 668.7 MB |
| spawn Pool(4)、8 task | 0.21 s、各worker 37.6 MB | 0.18 s、各worker 32.4 MB |

- Rust選択時はnative tableをimport時に構築する（失敗を起動時にfail closedするため）。
  Python backendの`_ShantenTable`を読み込まないため、起動直後のRSSはむしろ小さい。
- champion実行時のRSSはPolicyのdecision-local cacheが支配的で、backend差はない。
- Windows `spawn` workerは環境変数を継承し、全workerでRust backendが選択された。
- native binaryは約210 KB（wheel圧縮後 約110 KB）。

## 5. 同値性の検証

- `tests/test_native_shanten_backend.py`
  - Python core（`shanten._python_shanten_from_valid_counts`）と、Issue #115以前の独立した
    再帰探索oracle（`_python_shanten`）の**両方**との一致。固定seedの有効手（全純手牌枚数）、
    七対子・国士寄りの13 / 14枚、4枚持ち、么九牌偏重、対子偏重を含む。
  - count 0..4 / `fixed_meld_count` 0..4の範囲では、precondition外の枚数でも
    Python lookupと同じ値・同じ`ShantenTableError`になる。
  - 入力container（tuple / list / 一般sequence）と反復（cold / warm）で結果が変わらない。
  - thread並行呼び出しの一貫性。
  - 不正入力（34以外の長さ、count 5 / 負数 / 巨大整数、`fixed_meld_count`範囲外、
    `concealed_tile_count`不一致・無効枚数、非整数）はPython境界で`ValueError` /
    `OverflowError` / `TypeError`になり、未定義動作にならない。
  - artifact破損（空、magic、version、truncate、延長、frontier span、frontier id）と
    helper tableの不整合は`ShantenTableError`でfail closedする。
  - `LISJONG_SHANTEN_BACKEND=rust`でnative拡張がない場合、Pythonへfallbackせず起動時に失敗する。
  - 不明なbackend名は失敗し、defaultとpython指定ではnative拡張をimportしない。
  - rust選択processで公開`calculate_shanten()`の結果が一致し、native pathが実行されたことを
    call counterで確認する。
- `LISJONG_SHANTEN_BACKEND=rust`でfull test suite（1,977 tests）がローカルでpass。
  CIの`native-backend` jobでも同じ構成を実行する。
- 固定decision再生（3 Policy、計2,022 decision × 複数pass）と小規模対局2半荘で
  action不一致0。
- 値域：entry scoreは4 bit、4 group合算で0..60、結果は-1..8。入力をcount 0..4・34要素・
  `fixed_meld_count` 0..4に制限しているため`i16` / `i32`でoverflowしない。
  artifact寸法の検証は`u64`で行う。整数比較をfloatへ変えていない。
- #211で採用したexact shantenの定義（5枚目待ちを数えない）は、同じtableと同じ特殊形定義を
  使うため変わらない。

Policy identity、Policy module、default backendは変更していない。backend差は実験記録側に
backend名とnative build（本文書のtoolchain / build option）として残す。

## 6. 導入・保守

| 項目 | 状況 |
| --- | --- |
| 対応Python | CPython 3.14（非abi3、free-threaded buildは未対応・未検証） |
| 開発時 | Rust 1.98.1（`native/rust-toolchain.toml`）、Windowsは加えてMSVC Build Tools |
| build | `python -m pip install ./native`（maturin 1.15 build backend、PyPIから取得） |
| 利用時のcompiler | source installでは必要。prebuilt wheelがあれば不要（未整備） |
| 配布 | wheelの配布経路・Arenaのdependency pinへの組み込みは未整備 |
| CI | `native-backend` job（ubuntu-latest）：rustfmt check、native build、同値性test、rust選択下full suite |
| Python-only経路 | default。native拡張なしのinstall・import・全testは既存jobで維持 |

実証済み：Windows 11 / CPython 3.14.7 / MSVC target（build、全test、計測、Arena対局、spawn worker）。
CI（GitHub ubuntu-latest）でのLinux build / testはPRのCI結果を正本とする。

未確認：AWS Linux実機でのbuild・性能、manylinux wheel、macOS、Arena worker（AWS runner）での
導入手順、free-threaded Python、複数workerでの長時間対局のメモリ推移。

保守負担：Rust約500行（うちartifact検証が半分程度）。通常形の意味論はPython側table /
combine / penaltyを共有するが、特殊形（七対子・国士）とdispatchはRust側に写しがあるため、
Python coreを変更する場合は両方を更新し、differential testで検出する運用になる。

## 7. 判断

**追加検討**。

- 同値性：必須条件をすべて満たした（不一致0、full suite pass、行動列一致）。
- 性能：champion Policy判断全体で約71%短縮（基準30%）、半荘/時 6.6 → 22.7。
  追加メモリなし（基準20 MB）、起動の追加なし（基準100 ms）。
- 未達：利用環境（AWS Linux / Arena worker）へcompiler不要で配布する手段が未整備で、
  AWS Linuxでの実証もない。したがって「本採用を推奨」の条件を満たさない。

default backendの切替は本Issueの対象外であり、行っていない。

## 8. 残課題（後続Issue候補）

1. 配布：CIでWindows / manylinux wheelをbuildし、Arenaのdependency pin・AWS runnerへ
   opt-inで導入する手順と、AWS Linux上の同値性・性能確認（本採用判断の前提）。
2. two-stepの主因である`structural_efficiency`のTile-based cache key生成
   （TileType hash）の削減。Rustではなくcount-native化で対処できる可能性がある。
3. Python core側の改善余地：閉じた13 / 14枚では毎回七対子・国士を計算しており、
   championでは約25%を占める。Python実装を改善した場合、Rustとの差は縮む可能性がある
   （本Issueでは既存Python実装をそのまま比較基準にした）。
4. AGENTS.mdのcommon core（「Rust等の高速化は…Issueで合意されるまで導入しない」）と、
   試作・学習を許容する方針との文言整合。ecosystem共通文書のため、担当（ChatGPT WORK）で
   repository横断に扱う。

## 付録：再現手順

```powershell
# native build（opt-in）
python -m pip install ./native

# 固定decision再生（backendごとに別process）
$env:LISJONG_SHANTEN_BACKEND = "python"   # または "rust"
python tools/benchmark_tile_efficiency.py policy `
  --decisions decisions-placement-aware-speed-call.pickle `
  --policy lisjong.policies:PlacementAwareSpeedCallPolicy --repeat 2

# 対象計算単体（Python backendで入力を記録し、同一processで両backendを計測）
Remove-Item Env:LISJONG_SHANTEN_BACKEND
python tools/benchmark_tile_efficiency.py kernel `
  --decisions decisions-placement-aware-speed-call.pickle `
  --policy lisjong.policies:PlacementAwareSpeedCallPolicy --repeat 3 --limit-decisions 80
```

decision列は、Arenaの`LocalGameRunner`へ各席のPolicyを「`choose_action()`を呼んで
`(decision, action)`をlistへ追記するwrapper」で包んで1半荘実行し、listをpickleして作った。
Arena側へ採取用codeは追加していない（lisjongからArenaへ依存しないため、採取scriptは
repositoryへcommitしていない）。
