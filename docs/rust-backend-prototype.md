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

採取条件：lisjong-arena `9f748cd48d5b28673ecca2d4abf5b8eddaabdc4b`のvenv
（RiichiEnv 0.4.10）で、`PYTHONPATH`にlisjong `0408d62573a67b3d9f10f9ff8dc0312ab129ae8f`
（clean）の`src`を指定した。game modeは`4p-red-half`、seedは0、4席とも同じPolicy（席ごとに別instance）、
native拡張はなし。seed 0はArena seed ledgerの`arena-legacy-declared-v1` RETIRED quarantine
範囲（0..646）にあり、新しい評価seedは使っていない。強さ評価ではない。

| Policy（identity） | decisions | 元artifactのbytes SHA-256 | semantic digest |
| --- | ---: | --- | --- |
| `TwoStepUkeirePolicy`（Arena catalog `two-step`、一段階・二段階受入） | 684 | `31d0448e067ad5af08d992b939c02b66ad2f8120d919c4efe22670c750b8a4cf` | `7057aee210f8175e0e130fca7179875d304a788e40a5ccd148ef7a7082371f26` |
| `PlacementAwareSpeedCallPolicy`（Arena catalog `placement-aware-speed-call`、current Heuristic Champion） | 708 | `275ca281f3b57bb4e61ed16d9c95837fdca05d232297c0db56789a56adbcb8ea` | `1d80405ec92e2915227aefa4038c4a6ede1a822506ab2ef09bb89cd21418d91b` |
| `lisjong.policies:Kobalab0004ReferencePolicy`（#211参照Policy） | 630 | `f5782dc31aab06f9e2ddf8a088519e8c2c95cd803b6762af223e745401d287a8` | `ec0e51b3498e6e1a9c89285a941a452f047f77664cb9f41e9b88950a804d3c36` |

**同一性の2つの意味**：bytes SHA-256は元のpickle fileそのものを識別する。同じ内容のdecision列でも、
同じstep内の複数席の記録順とpickleのmemoizationは実行ごとに変わるため、再生成したfileのbytes SHA-256は
一致しない。再生成した入力の中身が同じかどうかは、semantic digestで確認する。semantic digestは、
席ごとにdecision順に並べた`repr((decision, action))`列のSHA-256である。

- 保存先：元artifact、raw測定結果、log、採取・計測script、Arena対局に使ったwheel、全fileのSHA-256 manifestは、
  project既定のlocal artifact root `C:\Dev\lisjong-artifacts\issue-213-rust-backend-prototype\`
  （単一マシン、外部公開・複製なし）にある。同じ場所の`README.md`に、fileごとのrevisionと実行commandを記録した。
- 再生成：採取scriptはlisjong-arena#398 / PR #399の`scripts/capture_policy_decisions.py`として保存した。
  lisjongからArenaへの依存は作らない。
  元条件（lisjong `0408d62`、seed 0、`4p-red-half`）で0004参照とtwo-stepを再生成し、semantic digestが
  元artifactと一致することを確認した（bytes SHA-256は異なる）。championは再生成していない。ただし、
  同じ条件で後から2回（Python / Rust backend）実行したcaptureのsemantic digestが、元artifactと一致している（§4.3）。

採取時の半荘wall-clock（Python backend、Arena経由、profilerなし）：two-step 59.8s、
champion 623.7s、0004参照 2.9s。two-stepとchampionの採取は、同時に実行していた別のprofileと時間が
重なっているため、半荘時間の比較には§4.3の値を使う。

## 2. profile結果と対象選定

`cProfile`は箇所の特定だけに使った（速度比較には使わない）。profile対象はlisjong `0408d62`
（native拡張なし）。championのprofile出力はartifact rootの`logs/profile-champion.log`にある。
two-stepの値は実行時のconsole出力から転記したもので、raw fileは残っていない。

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
- 改善余地の目安（Amdahl）：profile下の割合を使うと、championは約67%を0にしても約3.0倍、
  two-stepはshanten core全体で約32%（約1.5倍）になる。ただし、profiler下の時間割合は通常実行での割合とは
  異なる。そのため、これらの値を通常実行での厳密な上限としては扱わない（§4.2）。two-stepの主因は
  Rustの対象外（Tile変換）である。
- 対象外：Policy側の探索構造、`structural_efficiency`のTile-based API、structural predicate、
  engine / I/O。

## 3. 採用判断基準（Policy判断全体・対局の計測前に記録）

記録の時点：以下の基準は、Rust実装と同じcommit
（`736931dc0241cc20bcfbdce0b5ce83e57e8ebb8c`、2026-09-26T23:44:15+09:00）で記録した。
Policy判断全体の計測（`run_bench.sh`の開始は23:44:26）と小規模対局は、このcommitより後に行った。
一方、Rust実装の後で基準を記録する前に、次の2つのspot-checkを実行して結果を見ている
（raw fileはなく、値は作業記録からの転記）。

- 無作為の20,000手・通常形のみ（`micro.py`）：Python 6.01 µs/call、Rust 0.37 µs/call、不一致0
- kernel toolの初版（通常形`calculate_standard_shanten`境界）、0004参照の実入力101,312件：
  Python 12.82 µs/call、Rust 0.419 µs/call、不一致0

したがって、この基準は「実装前」ではなく、「kernel単体の速度を一部知った後、Policy判断全体と
対局の結果を見る前」に固定したものである。

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
§4.1、§4.2、§4.4はlisjong `736931dc0241cc20bcfbdce0b5ce83e57e8ebb8c`で計測した（native拡張も
同じcommitからbuild）。raw JSONはartifact rootの`results/`、実行順と時刻は`results/progress.log`にある。
PR #214のreview修正（§5のpool state検証、`16fee52`）はtable構築時の検証だけでhot pathを変えていない。
修正後に0004参照のkernelを再計測した結果（`results/kernel-0004-after-pool-validation.json`）は
Python 18.89 µs/call、Rust 0.504 µs/call、不一致0、table構築3.2 msで、修正前と同水準だった。

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

- champion：**約71%短縮（約3.5倍）**。profile下の割合（約67%）から求めた目安（約3.0倍）を
  上回った。profiler下の時間割合は通常実行での割合とは異なるため、その値から求めた上限は通常実行の
  厳密な上限にならない。通常実行での対象割合は測定していないので、差の原因はここでは特定しない。
- two-step：約37%短縮（約1.6倍）。残りの主因はTile-based cache key生成で、Rust化の対象外。
- 0004参照：約70%短縮（約3.4倍）。
- Rust選択時は、native coreの呼び出し回数（champion 2 passで51,650,404回）で
  Rust pathが実際に実行されたことを確認した。
- champion の最大decision時間は Python 29.3s → Rust 7.7s。

### 4.3 小規模対局（Arena、半荘1回）

lisjong-arena `9f748cd`の`LocalGameRunner`、`4p-red-half`、seed 0、4席同一Policy。
Policy以外の条件（seed、engine、Arena revision、worker数 1）は固定。
lisjongは`736931d`（未commitの変更は文書編集だけ）。native拡張は同じcommitからbuildしたwheel
（SHA-256 `636dcc24d2d711bd546ca425ed101af3d1646ccc7cb86f54b6d1a9346ce67d03`）で、Arenaのvenvへは対局の実行中だけinstallした。
対局ごとのpickle、`games.log`、wheelはartifact rootの`games/`と`wheel/`にある。

| Policy | Python [s] | Rust [s] | 半荘/時（Python → Rust） | 行動列 |
| --- | ---: | ---: | --- | --- |
| two-step | 27.71 | 17.62 | 130 → 204 | 4席とも完全一致 |
| champion | 544.88 | 158.92 | 6.6 → 22.7 | 4席とも完全一致 |

行動列の比較は席ごとの`(DecisionContext, InternalAction)`列で行い、両backendで
DecisionContext列・action列・最終得点がすべて一致した。semantic digestも両backendで一致し、
§1の元artifactとも一致する（two-step `7057aee2…`、champion `1d80405e…`）。同じstep内の複数席の記録順は
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
  - pool entryのresource state（`packed >> 4`）が360以上の場合も、table構築時に`ShantenTableError`で
    拒否する。対象は、score配列の範囲外になる値（例：4095）と、combine table内に収まるが別の行を
    読む値（例：360）の両方。release buildは`panic = "abort"`なので、この検証がないと範囲外アクセスが
    process全体を終了させ得る。回帰testはsubprocessで実行し、正常終了して期待した例外になることを確認する。
    hot pathが添字に使うstateは、この検証と`parse_combine()`の検証によって常に360未満になる。
  - `LISJONG_SHANTEN_BACKEND=rust`でnative拡張がない場合、Pythonへfallbackせず起動時に失敗する。
  - 不明なbackend名は失敗し、defaultとpython指定ではnative拡張をimportしない。
  - rust選択processで公開`calculate_shanten()`の結果が一致し、native pathが実行されたことを
    call counterで確認する。
- `LISJONG_SHANTEN_BACKEND=rust`でfull test suite（1,980 tests）がローカルでpass。
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
| 利用時のcompiler | source installでは必要。AWS worker向けprebuilt wheel（#216、[配布記録](rust-backend-distribution.md)）では不要 |
| 配布 | #216でAmazon Linux 2023 x86_64 / CPython 3.14向けmanylinux_2_28 wheelをCIでbuild（[配布記録](rust-backend-distribution.md)）。Arenaへの組み込みはArena側で扱う |
| CI | #216以降：`native-wheel` job（manylinux_2_28 containerでrustfmt check・wheel build）と`native-backend` job（compilerなしの`amazonlinux:2023`へwheelをinstallし、同値性test・rust選択下full suite） |
| Python-only経路 | default。native拡張なしのinstall・import・全testは既存jobで維持 |

実証済み：Windows 11 / CPython 3.14.7 / MSVC target（build、全test、計測、Arena対局、spawn worker）。
CI（GitHub ubuntu-latest）でのLinux build / testはPRのCI結果を正本とする。

未確認：AWS Linux実機でのbuild・性能、manylinux wheel、macOS、Arena worker（AWS runner）での
導入手順、free-threaded Python、複数workerでの長時間対局のメモリ推移。

保守負担：Rust約530行（うち約半分はartifact検証）。通常形の意味論はPython側table /
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
# 1. 入力の再生成（lisjong-arena checkout、Arena venv。lisjongは計測基準revisionのcheckout）
#    scriptはlisjong-arena#398 / PR #399。bytes SHA-256ではなくsemantic digestを§1の値と照合する
git -C <lisjong> worktree add --detach <lisjong-0408d62> 0408d62573a67b3d9f10f9ff8dc0312ab129ae8f
$env:PYTHONPATH = "<lisjong-0408d62>\src;<lisjong-arena>\src"
python scripts/capture_policy_decisions.py capture --policy placement-aware-speed-call `
  --seed 0 --game-mode 4p-red-half --output decisions-placement-aware-speed-call.pickle
#    --policy two-step / lisjong.policies:Kobalab0004ReferencePolicy も同様
python scripts/capture_policy_decisions.py digest decisions-placement-aware-speed-call.pickle
Remove-Item Env:PYTHONPATH

# 2. native build（opt-in、lisjong checkout）
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

元artifactの採取に使ったscript（`capture_decisions.py`）は、artifact rootの`scripts/`にそのまま保存した。
Arena PR #399の`scripts/capture_policy_decisions.py`は、同じwrapper方式にmanifestとdigestを加えたものである。
元の採取scriptとの同等性は、0004参照とtwo-stepのsemantic digest一致で確認した。
lisjongからArenaへのproduction依存はない。§4の測定を追検証するには、元artifact（artifact root `inputs/`）を
bytes SHA-256で確認して使う。元artifactが使えない場合は、再生成した入力をsemantic digestで照合して使う。
