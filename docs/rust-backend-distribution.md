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

## 4. CI

`.github/workflows/ci.yml`の2 job。default installと他のjobはPython-onlyのままで、Rust toolchainを要求しない。

- `native-wheel`：manylinux_2_28 container内でrustfmt check、wheel build、wheel tagの検査、
  `SHA256SUMS`と`BUILD-INFO.txt`（wheel名、SHA-256、lisjong revision、event / ref、run URL、Python・rustc・maturin version）を作成し、
  artifact `lisjong-native-wheel-<commit>`としてuploadする（保持90日）。
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
