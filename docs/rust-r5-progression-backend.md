# R5 progression探索のnative化（Issue #232）

Championが使うR5（expected terminal shanten progression, Issue #169 / #174）の探索本体を
opt-in Rust backendへ移した記録である。評価値・選択actionは変えない。default backendはPythonのままである。

## 1. 適用範囲と有効化

`LISJONG_SHANTEN_BACKEND=rust`は名前が向聴専用に見えるが、実際の適用範囲は次の3つである
（process単位の選択で、processの途中では切り替えない）。

| 入口 | 導入 | rust指定時 |
| --- | --- | --- |
| numeric shanten core | #213 | native |
| 打牌候補の一括構造評価`evaluate_discards` | #224 | native |
| R5 progression探索`evaluate_progression` | #232 | native |

未設定 / `python`ではnative拡張を読み込まず、R5はPython oracleが実行する。`rust`指定でnative拡張が
未導入・`API_VERSION`不一致（#232以前のwheelは2以下）の場合は、`_shanten_backend`のimport時に
`ShantenBackendError`で失敗する。R5だけPythonへ黙って戻ることはない。

build・wheel・CIは#213 / #216から変わらない（`docs/rust-backend-distribution.md`）。追加crateはない。
`native/src/progression.rs`が探索本体、`native/src/lib.rs`がPyO3境界である。

## 2. 構造

```text
_TerminalShantenProgressionEvaluator              Python oracle（独立した型として残す）
_NativeTerminalShantenProgressionEvaluator        native呼出しのadapter（別の型）
_new_progression_evaluator()                      本番3呼出し元が共通で使うfactory
    <- terminal_shanten_progression_mechanism_riichi_defense._evaluate_and_choose_discard
    <- targeted_honor_release_terminal_progression._evaluate_and_choose_discard（Champion gate）
    <- mechanism_riichi_defense_offensive_efficiency_diagnostic（full-legal / baseline-eligible）
```

- factoryは`_shanten_backend.native_evaluate_progression`が設定されていればnative evaluatorを、なければPython oracleを返す。
  既存型の内部をbackendに応じて置き換えることはしない。
- `_evaluate_progression_candidates()`は両evaluatorの共通入口`evaluate_roots(root_hands, remaining, horizon)`を使う。
  root候補群を1回でまとめて評価し、root候補ごとの`root_post_discard_shanten`もその結果に含まれる（Python側で再計算しない）。
- PolicyはPyO3拡張を直接importしない。nativeの入口は`_shanten_backend`が保持し、policy側の`_new_progression_evaluator()`だけが使う。
  Policy identityと公開decision契約は変わらない。
- 診断のopt-in条件、2つの候補集合、full-legal結果からbaseline-eligibleへsubsetを取る規則は変更していない。
- `_RecordingEvaluator`による遷移testはPython oracleを対象として残し、本番配線のtestはfactoryへの注入に変えた。

## 3. 維持する契約（Python oracleとの対応）

nativeはPython oracleを1行ずつ写している。

- horizonは3 self-drawのまま。root入口が受け付けるhorizonは整数1–3（boolと範囲外は拒否）。内部再帰のdepth=0は終端。
- 残余枚数、非復元抽出の重み、全remaining牌種のdraw、仮想discardを残余へ戻さない規則、各draw観測後にdiscardを選ぶ逐次DP。
- 内部childは`(shanten, hand_counts)`の辞書順、best更新は厳密な`<`、下界による打切りは`>=`。tenpai shortcut、depth-1 closed form、
  min-node branch and boundもそのまま。
- `terminal_shanten_counts`・mass・分母はexact integer。nativeの各countは`u64`で、Pythonへ返すときにPythonの`int`になる。
  massはPython側で`_distribution_mass()`が計算する。
- root群は同一decision内で1つのcache（向聴cacheと状態cache）を共有し、cacheは1回の`evaluate_progression`呼出しが終わると破棄される。
  decision間・対局間・evaluator間へ持ち越さない。native evaluatorは1 instanceにつき1 batchだけ評価できる（2回目は`RuntimeError`）。

### 計測counter

nativeは`visited_states` / `cache_hits` / `cache_misses` / `shanten_evaluations`をPython oracleと同じ位置・定義で数え、root群を
共有評価したdecision全体の値を返す。探索順とcache共有範囲が同じため、同じroot候補群ではoracleと**完全に一致する**
（testで固定）。evaluator生成直後・評価前は`None`で、未計測を0で埋めない。

## 4. GIL解放とデータの所有関係

採用したAPIはPyO3 0.29.2の`Python::detach`（旧`allow_threads`）である。

1. Python objectの読取り・検証は**すべてGIL保持中に**行い、root hands（`Vec<[u8; 34]>`）、remaining（`[u8; 34]`）、horizon（`u8`）を
   Rustが所有する値へ変換する。この段階で型・長さ・範囲・手牌サイズ・remaining合計≥horizonを検証する。
2. 不変のnumeric shantenデータ（frontier table、combine table、penalty table）は`ShantenCore`としてPython objectを含まない構造へ
   分離し、`Arc<ShantenCore>`で共有する。`StandardShantenTable`（`#[pyclass(frozen)]`）は`Arc<ShantenCore>`と例外型
   `Py<PyType>`だけを持つ。
3. `py.detach(move || search_roots(&core, &roots, &remaining, horizon))`。解放区間へ入るのは`Arc`のcloneとRustの値だけで、
   `Bound` / `PyRef` / `Py`などPython objectやその借用は入らない。戻り値はRustの`Result<SearchOutput, SearchError>`である。
4. GIL再取得後に、結果をPythonのtupleへ、`SearchError`をPython例外へ変換する。

GIL解放だけで探索がキャンセル可能になるわけではない。呼出しは探索が終わるまで戻らず、時間による途中打切りや別actionへの
fallbackは導入していない。別Python threadが進行できること（ArenaのI/O・deadline管理を阻害しないこと）はtestで確認する
（`NativeProgressionGilTest`）。

## 5. 異常時の扱い

`native/Cargo.toml`のreleaseは`panic = "abort"`で、release既定のoverflow検査に依存できない。そのため次の設計にした。

- 変換（narrowing cast・配列アクセス・減算）の前に入力を検証する：長さ34、各count 0–4、root手牌が有効な純手牌枚数、
  remaining合計≥horizon、horizonは`int`（bool不可）かつ1–3。範囲外の巨大intは`OverflowError`になる。
- 探索中に入力由来の違反（手牌とremainingの合計が同じ牌種で4枚を超える、有効でない手牌枚数）が現れた場合は`ValueError`。
- distributionの各count・mass・分母は`u64`で、加算・乗算・減算は`checked_*`。オーバーフローは`TerminalShantenProgressionPolicyError`。
- terminal shantenが0–8を外れる場合、draw後の手牌に打牌候補がない場合も`Result`で伝搬し、`TerminalShantenProgressionPolicyError`になる。
  例外型は呼出し側（policy側のadapter）がnativeへ渡す。nativeがPolicyをimportすることはない。
- table artifactの不整合は既存の`ShantenTableError`。
- 入力由来の違反に`unwrap` / `expect` / `assert!`は使っていない。

正常入力では起きない内部異常（u64 overflow、axis外、打牌候補なし）は、Rust unit test（`cargo test`、`native/src/progression.rs`の`tests`）で
エラー伝搬を検証する。production向けの異常注入APIは追加していない。release buildでabortしないことは
subprocess testで確認する（`NativeProgressionReleaseBuildAbortTest`）。Rust unit testは`pyo3`をlinkするためlibpythonのあるdev環境で
`cd native && cargo test`として実行する（manylinux wheel用のCI jobには追加していない）。

## 6. 検証

- Python / Rust同等性（`tests/test_native_progression_backend.py`）：horizon 1–3、`_reference_distribution`（枝刈りなしreference）との一致、
  root順序・重複、同mass childのdiagnostic distribution、counter完全一致、入力境界、3呼出し元すべて
  （Policy、Champion gate（`R5_PARENT_BEST` / `R5_NON_HONOR_ONLY_BEST` / `R5_HONOR_ONLY_SWITCH`の3 stage、最終actionとanalysis全体）、
  診断（full-legal / baseline-eligible両summary））のPython oracle対native比較、rust指定processで3呼出し元がnativeを実行すること。
- 既存の回帰test（リーチ・和了・鳴き・防御、R5非発動経路、Python oracleの遷移test）は変更していない。
- `tools/profile_speed_call_decisions.py`：`scan`はnative evaluatorのcounterを取得する。`breakdown`はnative時にR5一括呼出しの
  時間（`r5.r5_batch_ms`）と実測counterを報告し、native内部のshanten時間は`null` / `not_measured`、`r5_progression_dp`行の`self_ms`
  （Python oracleではDP overhead）は`null`、`shanten@r5_progression`行は出さない（0秒・0回として表示しない）。
- full suite：Python backend、`LISJONG_SHANTEN_BACKEND=rust`（`LISJONG_REQUIRE_NATIVE=1`）の両方で2109 tests OK（skipped 46は従来と同じ
  ML / native optional分）。CIがpre-mergeの正本である。

## 7. 性能測定（単独process、ローカル）

比較元は変更前revision `58ef82aeb10ac77cb66290d54e67a42426919d5b`（Python R5探索 + Rust向聴計算）、比較先は本変更
（Rust R5探索 + Rust向聴計算）である。各revisionの専用checkout / 仮想環境と、そのrevisionからrelease buildしたwheelを使い、
両方とも`LISJONG_SHANTEN_BACKEND=rust`で実行した。同一の外部driver（下記）で両revisionを測り、counterは各revision自身のtoolで取る方針に従った
（比較元のtoolはnative evaluatorのcounterを扱えないため、両者で共通のdriverはdecision全体の時間だけを測る）。

| 項目 | 値 |
| --- | --- |
| CPU / OS | Intel Xeon 2.1 GHz、4 vCPU / Linux 6.18（Claude Codeのcloud container） |
| Python / Rust | CPython 3.14.0rc2（通常版）/ rustc 1.94.1、maturin 1.15.0、`--release`（`opt-level=3`, `lto="fat"`, `panic="abort"`） |
| 比較元wheel | `lisjong_native-0.1.0-cp314-cp314-manylinux_2_34_x86_64.whl` SHA-256 `fe628f3f…af7ecc14`（`58ef82a`からbuild、`API_VERSION = 2`） |
| 比較先wheel | 同名 SHA-256 `eec6256a…bb6db44`（本変更からbuild、`API_VERSION = 3`） |
| 並列数 / 反復 | 1 process、1 thread、warm-upなし（初回呼出しを別掲）、1 decisionにつき1回 |
| 入力 | 固定seed 232で生成した合成closed 14枚手24 decision（河・副露・ドラ表示なし、live wall 70、全牌種が残る盤面）。`PlacementAwareSpeedCallPolicy.choose_action_with_analysis()`の時間 |

再現：`driver.py`は両revisionで動く（`PlacementAwareSpeedCallPolicy`、`DecisionContext`等の公開型だけを使う）。`random.Random(232)`で
136牌の山を24回shuffleして先頭14枚を手牌とし、各手牌の重複排除した打牌を`legal_actions`とする。

| | 比較元 | 比較先 |
| --- | ---: | ---: |
| 選択action（24 decision） | — | **全decisionで一致** |
| activation stage（24 decision） | — | **全decisionで一致** |
| R5発動 decision数 | 6 | 6 |
| R5発動 p50 | 9,851 ms | 1,842 ms |
| R5発動 p95 / p99 / max（n=6のため参考値） | 16,229 ms | 2,827 ms |
| R5発動 3秒超過 | 6 / 6 | 0 / 6 |
| R5発動 合計 | 63,551 ms | 11,658 ms |
| R5非発動 p50 / p95 / max（n=18） | 120 / 346 / 417 ms | 129 / 332 / 463 ms |
| 初回decision（R5発動） | 6,240 ms | 1,304 ms |
| peak RSS（process全体） | 694 MB | 197 MB |

R5発動decisionの個別値（ms）：比較元 6240 / 16229 / 8877 / 10244 / 9851 / 12109、比較先 1304 / 2827 / 1842 / 1814 / 1925 / 1945。
R5発動decisionの短縮は約4.8〜6.0倍である。非発動decisionはnativeの影響を受けず、差は測定ばらつきの範囲である。

**この数値の限界**：n=6のR5発動decisionはp95 / p99を統計的に意味づけられない。入力は合成した早巡の盤面であり、RiichiLabの実対局分布でも
#229のslow入力でもない（Arenaのcapture成果物は入手できなかった）。再現手順とseedを上に固定した。
連続対局入力（局・seat・順序付き）、R5 kernel時間とdecision全体の分離、実運用相当の並列数、メモリの多worker下での値、AWSでの値は未測定である。
ローカルの結果をAWSへ直接外挿しない。

## 8. RiichiLabの時間規定に対する状態

[RiichiLab公式protocol](https://riichi.dev/docs/protocol)（2026-09-30確認）：各応答3秒のgrace、各プレイヤー・各局15秒のbank
（毎局リセット、3秒超過分だけを消費）、deadlineは通信往復を含む。実際には毎回の`request_action.time`が正本である。

| 項目 | 状態 |
| --- | --- |
| 判断結果の同等性（action・stage・分布・counter） | **確認済み**（§6） |
| 固定した合成入力でのdecision全体の時間（単独process、ローカル） | **確認済み**：R5発動6件は1.3〜2.8秒で、3秒graceに収まった。ただし最大2.8秒は通信・周辺処理の余裕（Arena既存の安全余裕0.5秒を含めても0.2秒程度）がほぼない |
| 実運用相当のCPU・並列数・AWSでの時間適合 | **未確認** |
| 連続対局入力での局内bank消費（局・seat別の積算） | **未確認**（連続対局入力を用意できていない） |
| RiichiLab上の`action_ack`（elapsed / bank消費 / defaulted / stale）照合 | **未確認**（online確認は未実施） |

native化のみを根拠に「DCなし」「期限超過なし」とは判断しない。

引き継ぎ事項：
- Arena側：`pyproject.toml`のlisjong revision pinと、AWS wheelのbuild・SHA-256・`SOURCE_REVISION`を#232を含むrevisionへ更新する
  （`API_VERSION = 3`のwheelでなければrust指定のprocessは起動時に失敗する）。
- 実運用相当の並列数とCPUで、#229のslow入力（または固定した連続対局入力）を両revisionで再測定する。
- 最大2.8秒の余裕が小さいため、さらに短縮が必要なら残る負荷はnative内のnumeric shanten評価（測定した1 decisionあたり数十万回のオーダー）である。
  このIssueでは探索semanticsを変えないため、shanten core自体の高速化は別Issueで扱う。

## 9. 対象外（変更していない）

R5のgate縮小、horizon短縮、近似探索、時間による途中打切り、別actionへのfallback、新しい探索目的、向聴定義、押し引き、鳴き、
FiniteHorizon completion DP全体やChampion全体のRust化、AWS長時間実行・RiichiLab参戦の自動開始。
