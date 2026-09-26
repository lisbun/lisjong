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

（計測後に記入）
