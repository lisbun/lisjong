# Python計算backendの段階的廃止

親: [#328](https://github.com/lisbun/lisjong/issues/328)、棚卸: [#334](https://github.com/lisbun/lisjong/issues/334)。
2026-10-24 JST時点の設計。Rustを重複計算の正本にするが、PythonのPolicy・学習・制御・bindingは残す。
この文書のmergeは実装削除、対応OSの縮小、Arena pin変更、実験開始の承認ではない。

調査基準はlisjong [`6e545c0`](https://github.com/lisbun/lisjong/tree/6e545c0aa8340660b13ed8180d8b16d523fc1994)、
Arena [`86ea715`](https://github.com/lisbun/lisjong-arena/tree/86ea7152b0ce3de8241eefca30bbcedcaf00b2ea)。
#332（PR #335）のnumeric仕様fixture、#333（PR #336）のbatched discard Rust core化は完了済み。
既存の移行経緯は[rust-migration.md](rust-migration.md)、配布契約は
[rust-backend-distribution.md](rust-backend-distribution.md)を参照する。

## 1. 処理とconsumer

以下のpathは特記のない限りlisjong。private名も実在consumerがあるため、単なる検索置換で削除しない。

| 処理 | Python / Rustの現状 | consumer・契約 | 移行後の扱い / 前提 |
| --- | --- | --- | --- |
| numeric向聴（通常形・七対子・国士） | `hand_evaluation/shanten.py`、`_lookup_shanten.py`、`_python_shanten.py` / `native/src/lib.rs` | 公開`calculate_shanten`、trusted canonical-count入口、restricted standard入口。赤牌集約、concealed枚数からの確定面子数、入力検証を維持 | runtime Python数値計算・DFS oracleは検証移管後に廃止。wrapperの入力契約は保持 |
| batched discard・有効牌 | `shanten.py::_python_evaluate_discards` / `native/src/discards.rs`、`python.rs` | policy側の一括呼出。候補順、threshold、None/empty、4枚制限、戻り値、counter | #333でRust loop分離済み。Python loop廃止はnumericと同段階。binding validation/counterは残す |
| R5 terminal progression DP | `policies/terminal_shanten_progression_mechanism_riichi_defense.py::_TerminalShantenProgressionEvaluator` / `native/src/progression.rs` | `_new_progression_evaluator`経由でprogression、targeted honor release、offensive efficiency diagnostic。root群、distribution、同点時の選択、decision内cache | #337完了後にPython DPと比較専用testを廃止。native wrapper、policy選択・diagnosticは残す |
| structural complete/tenpai predicate | `_structural_predicates.py`と`shanten.py::_meldless_special_shanten` | numeric APIとは別の構造判定。後者は`_shanten_common`の七対子/国士helperを共有（#338で`_python_shanten`から分離済み） | 保持。Rust移植をこの廃止の前提に追加しない |
| artifact読込・補助配列 | `_shanten_table.py`（format定数、`read_table_payload`、`_ShantenTable`、`_COMBINE`、`_PENALTY`、`ShantenTableError`）/ native table parser | `_shanten_backend.build_native_table`が`_shanten_table`だけからpayloadと配列を渡す。rust選択processは`_lookup_shanten`をimportしない | #338で計算loopから分離済み。保持。format/version、endian、破損時の明示的失敗を維持 |
| artifact生成・frontier | `tools/generate_shanten_table.py`、`generate_structural_predicate_table.py`、`_shanten_frontier.py` | `_shanten_table`のpack/header/resource state/reader、`_shanten_common`の共通定数を使用（#338以降、`_lookup_shanten`/`_python_shanten`/nativeをimportしない） | offline Python generatorを保持。生成再現性とimport graphをtestで固定済み。native必須化で生成不能にしない |
| scoring | `scoring.py`は型・validation・binding / `native/src/scoring.rs` | scoring API、`SCORING_API_VERSION=1`。nativeなしでもimport/型構築可、評価時に明示的失敗 | 既にRust計算のみ。削除対象のPython scorerはない。lazy import契約を別判断なく変更しない |
| Python Policy / Learning / 制御 | `policies`、feature/dataset/training、Python API | ML optional + lazy import、観測情報境界、immutable artifact | 残す。Python非依存化や全repositoryのRust化を目的にしない |

`src/lisjong/`を上表のPython source pathに補う。`_python_shanten`全体を削除する前提だった
`TILE_KIND_COUNT`等の共有定数・特殊形helper・frontierの依存は、#338で`_shanten_common`へ移して解消した。
Python計算本体の削除対象は`_lookup_shanten`（lookup計算loop）と`_python_shanten`（DFS oracle）であり、
`_shanten_common`・`_shanten_table`・`_shanten_frontier`・generatorは保持対象である。
依存境界は`tests/test_shanten.py::ArtifactSupplySeparationTest`で固定する。

## 2. backend選択と互換性

現行`_shanten_backend.py`は`LISJONG_SHANTEN_BACKEND`未設定/`python`ならPython、
`rust`ならimport時にnativeを要求する。Rustの不在・API不一致時にPythonへ自動fallbackしない。
要求`API_VERSION=3`であり、lisjong単体のこの検査は`SOURCE_REVISION`一致を保証しない。
Arenaはさらにrevision、wheel SHA-256、実際のロードfile、native call counterを検査している。

切替実装の推奨契約は、未設定/`rust`でRustを要求し、`python`は移行案内付きで明示的に拒否すること。
旧値を黙ってRustへ読み替えない。未知の値も拒否する。まず旧selectorと両計算を維持して前提を整え、
切替PRでdefault変更とPython計算削除を揃える。import時か初回計算時かの失敗位置は現在のimport graphを
確認して切替Issueに固定する。scoringやoffline generatorの既存契約を巻き込まない。

API_VERSIONはnative入口・型・戻り値の契約変更に応じて判断する。Python backend廃止だけを理由に
機械的に上げず、API=3だけで旧wheelを互換とみなさない。同一source revisionのwheelが必要。
scoringのAPI versionは別契約である。

## 3. 配布・開発環境の前提

| 環境 | 現在の証拠 / 未確認 | Rust必須化前に必要なこと |
| --- | --- | --- |
| CPython 3.14通常版、manylinux_2_28 x86_64 | CIでrelease wheelをbuild。native coreは`--no-default-features`でPyO3なしtest | 同一revisionのwheel供給、checksum、SOURCE_REVISIONを維持 |
| AL2023 x86_64、distro CPython 3.14 | compilerなしcontainerでbinary-only install、native/full suite。live workerは別の実行証拠 | 配布手順とartifact保存を維持。CI合格をlive実測と呼ばない |
| Ubuntu CIのquality / ml-tests / reference-validation | 現在はnativeなしのPython計算経路。mlはCPU torch、referenceは外部mahjong | 同revision wheelを先にinstallしRust検証へ移す。ML未導入coreの回帰も残す |
| Windows開発環境 | 保存先のWindows pathは配布文書にあるが、Windows native wheelの検証証拠ではない | native build/wheel/install/API smokeを用意するか、検証済みLinux/WSL利用への合意が必要 |
| macOS / aarch64 / 他Python / 3.14t | 初期wheel対象外。source build可能性は検証済みsupportではない | 対応追加か対象外の明示判断。推測で対応済みにしない |

root packageと`lisjong-native`は別distributionで、rootのdependenciesは現在空。
wheelはActions artifact（90日）と既存の外部保存であり、PyPI/Releaseへの恒久公開は未整備。
必須化前に「明示的な同revision二段install」を推奨案として文書化・CIで検証する。
自動dependency解決や単一package化は別案で、無断で公開先を追加しない。
main pushのwheelを採用し、PR merge-ref wheelをruntimeの正本にしない。

未決判断D1: Windowsをnative対応するか、検証済みLinux開発経路を採るか。
未決判断D2: wheel永続保存・取得手順と二段installを運用標準にできるか。
推奨は既存Linux経路を先に維持し、必要なWindows consumerを確認すること。
D1/D2は配布・必須化をblockするが、#337/#338と本設計の完了はblockしない。

## 4. 代替検証の充足と不足

| 対象 | 既存の検証 | 残す / 補う |
| --- | --- | --- |
| numeric | #332の`shanten_spec_fixtures.py`、`test_shanten_spec_fixtures.py`。全有効枚数・確定面子、特殊形、赤、public/canonical/native、対称性、入力不正、routing | 定数と根拠を正本にする。旧DFS/lookupとの差分test削除時にcoverageを対応付ける。fixtureを再作成しない |
| batched discard | #333のRust core unit test、既存native binding tests。候補順、None/empty、threshold、4枚、table error | synthetic scorerのloop testと実際の向聴計算testを区別。入力検証・counter・実table integrationは残す |
| R5 | `test_native_progression_backend.py`のPython/unpruned差分、horizon 1〜3、root順/重複、境界、counter/cache、3 caller、GIL。Rustはoverflow等 | #337で`test_progression_spec_fixtures.py`に手計算distribution・tie・caller固定値を追加。[導出とcoverage](r5-spec-verification.md)参照。既存差分/境界/GIL検証は保持 |
| artifact・binding | format/parser、破損、配列、API不一致、release subprocess、実native routing | #338の移動後も維持。generatorの再現性、nativeなしimport経路、rust選択時に計算moduleを読まないことを固定 |
| 配布・provenance | manylinux build、AL2023 binary-only、SOURCE_REVISION。Arenaでwheel hash/ロードfile照合 | Python比較を消しても保持。wheelなし・不整合が成功/skipにならないtestを切替時に固定 |

Python oracleの削除は「testがpassした」だけでは進めない。各差分testについて仕様fixture・property・
境界・binding契約のどれが代替するかを切替PRへ記録する。外部reference検証まで一括削除しない。

## 5. Arena consumerと過去artifact

Arenaの現在のlisjong pinは`0fff8918ec349eeec129907f6a723e2d7314b95a`。
lisjong mainより古いが、本設計で更新しない。`pyproject.toml`、
`shanten_backend_verification/backend.py`、CIのnative source installがこのpinを使用する。

| Arena consumer | 影響・引継ぎ |
| --- | --- |
| `shanten_backend_verification` probe/games/compare/report | explicit env、Python非import条件、旧Python/Rust比較と現在wheel identityを分離。新規実行はRustのみへ、historical readerは別契約 |
| `pure_offense_benchmark` | `--shanten-backend {python,rust}`とrecordのpair、parent/全worker検査。新規実行でpythonを受理して黙ってRust実行しない |
| `riichilab/aws_backend.py` / `aws_instance_run.py` | backend指定・expected-backend、保存済み記録のverify。実行CLIと読取CLIの受理範囲を分離 |
| `l2a_runtime_smoke` | 既にRust要求。pin/wheel/provenance整合を維持 |
| `heuristic_candidate_engine_aabb`ほか既存AABB protocol | #517のP2 Python/Rust証拠、protocol version/digestを固定。新gateを旧protocolへ上書きしない |
| Arena CI / environment identity | package pinとnative build pinを同時更新し、同revision wheelとidentity testを揃える。旧pinで先行互換準備を完了する |

「旧revisionを保持して当時の環境でverify/reproduceできること」と「最新コードで旧artifactをverifyできること」は別。
前者はrecorded Arena/lisjong revision・wheel・hash・protocol・依存環境の保存が必要で、git SHAだけでは不十分。
後者を保証するには、schema/version別readerと過去fixtureの回帰testが必要。現行codeは古いrunの再現を
記録済みArena revisionへ委ねる部分があるため、最新コードがすべてverifyできるとは主張しない。

未決判断D3: 最新版で保証するhistorical verifyの範囲。推奨は読取/整合性検証を可能な範囲で保持し、
旧Python計算を要する再導出・再実行は保存した旧環境へ明示的に案内すること。
Arena ownerの後続IssueでCLI別・schema別に保証/非対応と根拠を固定する。
旧artifactを新schemaへ無言変換せず、新規Rust-only gateは新versionのprotocolとして設計する。
seed管理、P3、候補固有の確認や正式評価承認は省略しない。新規対局を互換準備の完了条件にしない。

## 6. 実装順序・owner・完了単位

| 順序・状態 | Issue / owner | 完了条件と依存 |
| --- | --- | --- |
| 完了 | #332 / #333、lisjong | numeric fixtureとbatched Rust core。追加作業なし |
| 1 検証実装済み | [#337](https://github.com/lisbun/lisjong/issues/337)、lisjong | R5の独立fixtureと選択契約の不足補完。完了状態・CI証拠はIssue/PRを参照。削除なし |
| 2 実装済み | [#338](https://github.com/lisbun/lisjong/issues/338)、Claude Code | artifact/helper供給分離、bytes/API/default不変、generator回帰。完了状態・CI証拠はIssue/PRを参照 |
| 3 判断後に起票 | 配布準備 / lisjong、環境確認はユーザー管理環境 | D1/D2を確定し対象platformのbuild/installを検証。同revision二段install、永続取得と失敗動作を文書化 |
| 4 設計から起票 | Arena互換準備 / Arena owner | D3、CLI/schema別検証、旧protocol保持、新gate仕様を確定。旧pinで読取/新旧gate準備をmergeし、まだ実行切替しない |
| 5 BLOCKED | Rust必須化・Python計算削除 / lisjong | #337/#338、配布、Arena準備完了後。numeric/batched/R5と比較専用test/CIを限定削除。保持項目、env失敗契約、同revision CI/full suiteを確認 |
| 6 BLOCKED | Arena採用 / Arena owner | 5のmain wheelを保存・hash確定後、package/native pin・新規CLI/gate/CIを整合して更新。歴史的証拠は保持 |

3〜6は後続Issue案でありREADYの実装依頼ではない。各Issue起票時に具体的な対象file・検証command・
未決判断の決定記録を追加する。#328はこれらが完了するまでopenを維持する。
対象fileの入口は、3がroot/nativeの`pyproject.toml`、`.github/workflows/ci.yml`と配布文書、
4がArenaの`shanten_backend_verification`、`heuristic_candidate_engine_aabb`と各CLI/test・評価文書、
5が本書§1のnumeric/batched/R5・selectorと対応test/CI、6がArenaの`pyproject.toml`、
`shanten_backend_verification/backend.py`、CI・新規実行CLI・評価文書である。
中間commitでArenaのpinが未配布nativeを要求する状態を作らない。必要なら5を処理別PRに分けるが、
default変更と利用可能なwheel/失敗契約を切り離して壊れた状態をmergeしない。

### rollbackと停止条件

準備作業の比較基準はlisjong `6e545c0aa8340660b13ed8180d8b16d523fc1994`。
現在のArena運用の復帰基準はArena `86ea7152b0ce3de8241eefca30bbcedcaf00b2ea`と、
そのpin `0fff8918ec349eeec129907f6a723e2d7314b95a`、同revision wheel
（SHA-256 `130929229b16b0508a61e77d04c0a219ab65a15f7e79a74d60c43725c0cb8fba`）の組である。
この記載はwheel現物の保存確認を代替しない。採用Issueで切替直前の組・保存先・復元手順を再確認する。

削除後はenvを`python`へ戻すだけではrollbackできない。lisjong/Arena revision、wheel、環境を組で戻す。
wheel欠落、provenance不一致、既存consumerが実行不能、旧証拠を誤って成功扱いする場合は切替を停止する。
本設計でChampion、実行済み評価、seed ledger、正式評価gateを変更しない。棋力・速度改善も主張しない。
