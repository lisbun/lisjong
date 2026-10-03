# Rustへの段階移行

決定日: 2026-10-03。ユーザー承認済み。
横断作業の正本: [lisjong-project #87](https://github.com/lisbun/lisjong-project/issues/87)。
最初の実装: [lisjong #240](https://github.com/lisbun/lisjong/issues/240)。

## 到達目標

対局・判断の基盤はRustに集約し、学習・統計分析・実験はPythonを継続する。
本番の対局処理とPolicy判断がPythonなしで完結する構成を目指す。
Pythonの学習・分析consumerは、薄いbindingを介して本番と同じRustの特徴量・計算を使う。
ML framework非依存coreと、学習専用truth / online player-safe入力の分離を維持する。
将来のNN推論形式・runtimeは、採用するmodelの要件に基づいて別途決定する。

言語の選択はrepository責務を変更しない。

| owner | Rustへ移す責務 |
| --- | --- |
| lisjong | Policy入力・ActionのAI契約、特徴量、牌効率、探索、belief、risk/value、Policy・本番推論 |
| lisjong-engine | 麻雀ルール、合法手生成、game state transition |
| lisjong-arena | 環境アダプター、対局接続、本番runner |

Arenaの評価・分析やAWS起動・収集などの運用は、Python等の既存手段を必要に応じて維持する。
external executionをlisjongへ戻さず、既存の依存方向を守る。

## 進め方と切替条件

1. 既存native計算をPython bindingから分離する。
2. 局面・Action契約、特徴量・評価、Policy、engine/runner接続を、依存関係に沿った子Issueへ分ける。
3. 各段階で基準revision、入力fixture、対象consumerを固定し、移植と戦術変更を別々に検証する。
4. 結果・選択Action・合法性・情報境界・同点時順序・乱数・例外・traceを照合する。
   数値許容差が必要な場合は基準を事前に明記し、未説明の判断差を残さない。
5. 実行時間・メモリ・配布・起動・障害時動作を対象runtimeで確認してから切り替える。
   rollback可能なrevisionと条件を記録する。
6. 検証後はRustを正式実装とし、Python計算版はconsumer確認後に廃止またはhistorical referenceへ隔離する。

直近の棋力改善と並行して進める。移行中のPython実装は同値性検証のoracleとして利用できる。
二重保守の恒久化や、一括書き換えを目標にしない。
移行作業は承認されたarchitecture上の目的を持つ。各段階の採用を速度向上だけでは判断しないが、
高速化・棋力向上を主張する場合には別途実測を必要とする。

## 第1段階の到達範囲

`native/src/lib.rs`にtable parser・numeric shanten core、`native/src/progression.rs`にR5探索を置く。
Python依存の型変換・例外変換・module登録・診断カウンタは`native/src/python.rs`が担当する。

Cargo feature `python`はdefaultで有効。既存のmaturin wheel生成とPython APIは維持する。
featureを無効化するとPyO3依存なしでRust coreのbuild/testを実行できる。

```sh
cd native
cargo fmt --check
cargo tree --locked --no-default-features --edges normal,build
cargo test --locked --no-default-features
```

CIはPyO3がdependency graphに含まれないこととRust単体テストを確認し、従来のwheel生成・
compilerなしの環境でのinstall・native同値性・Rust backendでのfull regressionも維持する。
Rust公開APIはまだ設けていないため、bindingを外したlibrary buildではunused/dead-code警告が出る。
将来用APIを先に公開して警告を隠すことはしない。

この段階では以下が残る。

- shanten artifact・補助配列の供給はPython側が担当する。
- batched discardの列挙はまだbinding module内にある。
- Policy・入力契約・engine・対局runnerは移植していない。
- Rust指定はopt-in、Pythonがdefaultという現在の選択規則を維持する。
- API_VERSION=3、SOURCE_REVISION、例外・結果順序・カウンタの意味を変更しない。

したがって、Rust単体のbuild/test成功は本番のPython非依存化完了を意味しない。
runtime性能や棋力の改善も本段階では主張しない。

## 関連する既存契約

- [Architecture](architecture.md)
- [Rust backend試作](rust-backend-prototype.md)
- [Rust backend配布](rust-backend-distribution.md)
- [R5 Rust backend](rust-r5-progression-backend.md)
