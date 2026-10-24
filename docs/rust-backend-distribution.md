# Rust backend wheel配布（Issue #216、lisjong側）

[#213の試作](rust-backend-prototype.md)で実装したopt-in Rust numeric shanten backendを、
lisjong-arenaのAWS workerへcompilerなしで導入するためのwheel生成・配布・組み合わせ固定の記録である。
向聴数の定義、Rust化した計算範囲、default backend（Python）は変更しない。

本書はlisjongの担当範囲（wheel生成・配布、本体との組み合わせ、計算の同値性）を扱う。
Arena側の導入・worker設定・AWS実行と、測定条件・結果・利用判断はlisjong#216と関連Arena Issue / PRに記録する。

## 1. 対象環境

初期対象は、lisjong-arenaの既存AWS bootstrap（`scripts/aws/bootstrap-*.sh`）が前提にしているworker環境1種類に限る。

| 項目 | 値 | 根拠 |
| --- | --- | --- |
| OS | Amazon Linux 2023 | bootstrapが`/etc/os-release`の`ID=amzn` / `VERSION_ID=2023`を検査する |
| CPU architecture | x86_64 | bootstrapが`uname -m`を検査する。instanceはc7i系 |
| Python | distributionの`python3.14`（`dnf install python3.14`）、CPython 3.14、通常版（abiflagsなし） | bootstrapのinstall手順 |
| glibc | 2.34 | AL2023 |

`amazonlinux:2023` containerで確認した値：CPython 3.14.7（GCC 11.5.0 build）、`linux-x86_64`、abiflagsなし、glibc 2.34。
container上の確認は実機AWS workerでの確認を代替しない（実機はArena側で確認する）。

対象外：Windows / macOS / aarch64 / 他のPython version / free-threaded build（3.14t）向けwheel。
Windows上の開発・実行はWSL（Linux x86_64）を標準とする（Issue #342のD1、§7）。

## 2. wheel

| 項目 | 値 |
| --- | --- |
| 配布名 / module | `lisjong-native` / `_lisjong_native` |
| wheel tag | `cp314-cp314-manylinux_2_28_x86_64`（非abi3、CPython 3.14専用） |
| 互換性方針 | manylinux_2_28（glibc 2.28以上）。対象のglibc 2.34を満たす。maturinがbuild時にmanylinux policy適合を検査する |
| build環境 | `quay.io/pypa/manylinux_2_28_x86_64` container内の`/opt/python/cp314-cp314` |
| toolchain | Rust 1.98.1（`native/rust-toolchain.toml`）、maturin 1.15.0、PyO3 0.29.2、`cargo --locked`、release profile（`native/Cargo.toml`） |
| build元revision | `_lisjong_native.SOURCE_REVISION`へbuild時に埋め込む（完全なcommit id。source buildでは`"unknown"`） |

**build環境で動くことと対象環境で動くことの区別**：wheelはmanylinux container内でbuildするが、そこでは検証しない。
検証は別jobで、compilerを持たない`amazonlinux:2023` containerへwheelだけをinstallして行う（§4）。

manylinux imageはtagを固定していない。そのためwheelのbytesは再buildで一致するとは限らない。
wheelの同一性は、再buildの再現性ではなく、保持したwheel fileのSHA-256で扱う（§3）。

## 3. 本体との組み合わせと同一性

native拡張は、Python側の`_shanten_table.bin`、resource state combine table、penalty tableをbytesで受け取り、
七対子・国士無双とのdispatchをPython coreから写している。したがって、wheelは**build元と同じlisjong revision**との組み合わせでだけ検証済みとして扱う。
汎用的な互換性管理機構は設けない。

組み合わせは次の3値で固定する。

1. lisjong revision（完全なcommit id）：Arenaが`pyproject.toml`でpinするrevision
2. wheel SHA-256：CIが`SHA256SUMS`に記録した値
3. `_lisjong_native.SOURCE_REVISION`：1と一致しなければならない

利用側（Arena）は、導入前に2を照合し、導入後に3が1と一致することを確認する。一致しなければ対局開始前に失敗させる。

**native API version（Issue #224 / #232）**：上記とは別に、native拡張は提供する入口の組を`_lisjong_native.API_VERSION`で示す
（1 = `standard_shanten` / `shanten_from_valid_counts`、#224以前のwheelは属性なしで1とみなす。
2 = 打牌候補の一括構造評価`evaluate_discards`を追加。
3 = R5（expected terminal shanten progression）探索本体`evaluate_progression`を追加、Issue #232）。
`LISJONG_SHANTEN_BACKEND=rust`のprocessは
`_shanten_backend`のimport時に`API_VERSION`が本体の`REQUIRED_NATIVE_API_VERSION`（現在3）と一致することを確認し、
不一致なら`ShantenBackendError`で失敗する（一部の入口だけPythonへ切り替えない。API_VERSION 2以前のwheelでR5だけPythonへ黙って戻ることもない）。
python選択時はnative拡張をimportしないため、旧wheelがinstallされていても影響しない。これは古いwheelの取り違えを起動時に検出する最低限の検査であり、
§3の3値による組み合わせ固定（SHA-256・`SOURCE_REVISION`照合）を置き換えない。

## 4. CI

`.github/workflows/ci.yml`の2 job。default installと他のjobはPython-onlyのままで、Rust toolchainを要求しない。

- `native-wheel`：manylinux_2_28 container内でrustfmt check、wheel build、wheel tagの検査、
  `SHA256SUMS`と`BUILD-INFO.txt`（wheel名、SHA-256、lisjong revision、event / ref、run URL、Python・rustc・maturin version）を作成し、
  artifact `lisjong-native-wheel-<commit>`としてuploadする（保持90日）。
- Issue #232のR5探索は`native/src/progression.rs`に追加した。wheel build手順・CI job・ArenaのAWS wheel構築 / 取得 / SHA-256照合の手順は変わらない
  （追加の依存crateはなく、manylinux_2_28 wheelのtagも同じ）。ただしAPI_VERSIONが3になったため、**Arenaが取得するwheelは
  #232以降のrevisionでbuildしたものでなければrust指定のprocessが起動時に失敗する**。Arena側の`pyproject.toml`のlisjong revision pinと
  wheel SHA-256 / `SOURCE_REVISION`は、#232を含むrevisionへ揃えて更新する必要がある（連携事項。Arena側の変更はこのPRの対象外）。
- `native-backend`：`amazonlinux:2023` containerで`dnf`の`python3.14`だけを入れ、`cc` / `gcc` / `g++` / `clang` / `rustc` / `cargo`が
  存在しないことを確認してから、artifactのSHA-256を照合し、wheelを`--only-binary=:all: --no-index --no-deps`でinstallする。
  `SOURCE_REVISION`がcheckout中のcommitと一致することを確認し、native同値性test（`tests.test_native_shanten_backend`）と、
  `LISJONG_SHANTEN_BACKEND=rust`でのfull test suiteを実行する。

source buildした環境での成功は、wheel導入の確認として扱わない（CIはsource buildを行わない）。
`python -m pip install ./native`は開発用のsource buildとして残る。

## 5. 取得先と保持

1. 配布に使うwheelは、`main`へのpush（merge後）で実行されたCI runの`native-wheel` artifactを使う。
   pull_requestのrunはmerge commitからbuildされ、そのrevisionは`main`に残らないため使わない。
2. ユーザー管理環境でartifactを取得し、`sha256sum --strict -c SHA256SUMS`で照合する。
   ```bash
   gh run download <run-id> --repo lisbun/lisjong --name lisjong-native-wheel-<commit> --dir wheelhouse
   (cd wheelhouse && sha256sum --strict -c SHA256SUMS && cat BUILD-INFO.txt)
   ```
3. 保持：GitHub Actions artifactは90日で消えるため、正本の保持先はproject既定のlocal artifact root
   （`C:\Dev\lisjong-artifacts\issue-216-rust-wheel\`、wheel・`SHA256SUMS`・`BUILD-INFO.txt`）とする。
   PyPIやGitHub Releaseでの公開は行わない。
   Issue #342以降は、wheelと同revisionのsource archiveを組にしたbundleとして保存する（§7）。
4. AWS workerへの受け渡しは、Arena runnerの既存run input（S3 transfer bucketの`input/`、`manifest.sha256`でSHA-256照合）を使う。
   workerがGitHubから直接取得する経路は作らない。

wheelを更新する（lisjong revisionを変える）場合は、新しい`main` runのartifactを取得し、§3の3値を記録し直し、
Arena側でAWS対象環境の同値性確認をやり直す。

## 6. 導入（Arena worker向けの手順）

```bash
# wheelはrun inputとして取得・SHA-256照合済み
python3.14 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: --no-index --no-deps \
    "$LISJONG_INPUT_DIR/lisjong_native-0.1.0-cp314-cp314-manylinux_2_28_x86_64.whl"
.venv/bin/python -c 'import _lisjong_native; print(_lisjong_native.SOURCE_REVISION)'   # pinしたlisjong revisionと一致
export LISJONG_SHANTEN_BACKEND=rust   # worker processへ環境変数で継承される
```

- `--only-binary=:all: --no-index`により、wheelが対象環境に合わない場合は
  `... is not a supported wheel on this platform`でinstallが失敗し、source buildへ回らない。
- `LISJONG_SHANTEN_BACKEND=rust`で`_lisjong_native`を読み込めない場合、lisjongはimport時に`ShantenBackendError`で失敗し、
  Pythonへfallbackしない（#213）。
- Pythonへ戻すには`LISJONG_SHANTEN_BACKEND`を未設定または`python`にする（wheelのuninstallは不要）。
  Python-onlyの実行にはwheelもRust toolchainも要らない。

## 7. 同revision source・wheel bundle（Issue #342）

2026-10-24の決定（#342）：

- D1：通常版CPython 3.14 / Linux x86_64を優先する。Windows上の開発・実行はWSLを標準とし、
  Windows native wheel、macOS、aarch64、3.14tは追加しない。WSL個別環境の互換性は導入時に検証する
  （方針決定を動作確認済みと扱わない）。
- D2：exact lisjong revisionと、そのmain push CIでbuildしたwheelを組で保存する。
  保存先は既存のlocal artifact root（`C:\Dev\lisjong-artifacts\issue-342-native-bundle\<revision>\`）で、
  Actions artifactの90日期限だけに依存しない。自動公開・PyPI・新しいS3/課金・credential自動処理・
  latestへの追随は行わない。

### bundleの構成と作成（ユーザー管理環境）

```text
<bundle>/
  bundle-manifest.json        # tools/native_bundle.py manifest が作る
  lisjong-<revision>.tar      # git archive --format=tar --prefix=lisjong/ <revision>
  wheel/                      # §5のCI artifactをそのまま（*.whl、SHA256SUMS、BUILD-INFO.txt）
```

```bash
R=<main revision>   # 完全なcommit id
gh run download <main push run-id> --repo lisbun/lisjong --name lisjong-native-wheel-$R --dir <bundle>/wheel
git archive --format=tar --prefix=lisjong/ -o <bundle>/lisjong-$R.tar $R   # WSL/Linuxのgitで作る
python3.14 tools/native_bundle.py manifest <bundle>
```

source archiveは改行変換のないgit（WSL/Linux、`core.autocrlf`未設定）で作る。Windowsの
`core.autocrlf=true`では`git archive`がCRLFへ変換し、別bytesのarchiveになる。
`manifest`は、archiveのcommit id（git archiveのpax header）、`BUILD-INFO.txt`の
wheel名・SHA-256・revision・`push` / `refs/heads/main`、`SHA256SUMS`、wheel tag
（`cp314-cp314-manylinux_2_28_x86_64`）、sourceの`REQUIRED_NATIVE_API_VERSION`を照合してから、
revision、source archive / wheelのfilename・bytes・SHA-256、期待する`SOURCE_REVISION` / `API_VERSION`、
CI run URL、build Python、対象Python / platformを記録する。既存manifestは上書きしない。

### 導入・検証（1 command）

```bash
python3.14 tools/native_bundle.py install <bundle> <新規venv path>
```

- 開始前：実行中interpreterがCPython 3.14通常版 / `linux-x86_64`であること、manifest形式、
  source archiveとwheelのbytes・SHA-256、archiveのcommit idとrevision、sourceの要求API versionを照合する。
  不一致・wheel欠落・出力先が既に存在する場合は何も作らずに失敗する（既存環境を削除・上書きしない）。
- 導入：新規venvへwheelを`--only-binary=:all: --no-index --no-deps`で、展開したsourceを`--no-deps`で入れる
  （sourceのbuild backend setuptoolsだけはpip既定のbuild isolationで取得する）。
- 導入後：`LISJONG_SHANTEN_BACKEND=rust`の別process（`python -I`）で、`SOURCE_REVISION` / `API_VERSION`、
  `_lisjong_native`と拡張`.so`・`lisjong`がそのvenv内からロードされたこと、公開`calculate_shanten`と
  R5 evaluator（`_new_progression_evaluator`）がnative counterを進めること、`_lookup_shanten` /
  `_python_shanten`をimportしていないことを確認する。比較用のPython対局は起動しない。
- 成功時だけ`<venv>/lisjong-native-bundle-install.json`（manifest、host、検証結果）を書く。
  途中で失敗した場合は、このcommandが作ったvenvを削除し成功記録を残さない。

拒否の各条件はnetwork不要のfocused test（`tests/test_native_bundle_tool.py`）で固定する。
実native拡張での検証scriptは`_lisjong_native`がある環境（CI `native-backend`）で実行する。

### 責務と復元

| 段階 | 担当 / 手順 |
| --- | --- |
| build | lisjong CI（main push、§4） |
| 取得・bundle作成・保存 | ユーザー管理環境。上記手順でlocal artifact rootへ保存し、READMEにrevision・hash・run・確認環境を記録 |
| 復元 | 保存済みbundleから`install`で新規venvを作る。`main`のlatestへは追随せず、記録したrevisionで再現する |
| Arena worker | 従来どおりArena run input経由（§5.4、§6）。本toolはArena側の導入手順を置き換えない |

過去artifactの再現は記録済みrevision・環境のbundleで行う。全過去artifactを最新codeで再実行する基盤は作らない
（CLI別の保証範囲はArena互換準備で確定する）。

### 保存済みbundleと確認環境

| revision | 保存 | 導入確認 |
| --- | --- | --- |
| `c0e1dfbc220ea3c114cf18e788608897b7853244`（#338 / PR #341） | 保存済み（2026-10-24、`issue-342-native-bundle\c0e1dfb…\`、wheel SHA-256 `e7cccce1…0a49`、CI run 39996162089） | WSL2 Ubuntu 24.04.4（glibc 2.39）、uv管理CPython 3.14.7で`install`成功（2026-10-24） |

未確認：Ubuntu以外のWSL distribution、distro提供のCPython 3.14、ユーザーの他のWSL / Linux開発環境。
AL2023のbinary-only導入はCI `native-backend`の証拠であり、本toolでの導入は未実施。
`issue-216-rust-wheel`のwheel（`2553c1b`、#224以前のためAPI_VERSION 1）は現行lisjongのrust選択では起動時に拒否され、
source archiveと組にもなっていないため、本bundle形式の保存済みbundleとして扱わない。
