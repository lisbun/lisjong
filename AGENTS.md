# AGENTS.md

## 適用範囲

このファイルはrepository全体へ適用する恒常的な作業規則である。
Issueまたはユーザーの明示的な指示が本書と異なる場合は、その指示を優先する。

## 開発フロー

- GitHub Issueを作業の目的、スコープ、完了条件の正本とする
- `main`へ直接pushせず、Issue専用branchとPull Requestを使用する
- 1つのPull Requestでは1つの主目的を扱い、無関係な変更を混ぜない
- branch作成、commit、push、Pull Request作成、Ready化、merge、Issue closeは、
  それぞれ実行前にユーザーの明示的な承認を得る
- 破壊的操作、外部公開、課金、認証情報の使用は、対象と影響を示して承認を得る

## 実装規則

- 通常版CPython 3.14を初期基準とし、free-threaded build（3.14t）は互換性を
  個別に検証するまで対象外とする
- RiichiEnvやRiichiLab固有の型・protocolをPolicyへ持ち込まない
- Policyへ渡す情報を当該seatの観測可能範囲に限定する
- 調査前に将来の構造を過剰設計しない
- Rustはprofilingで必要性が確認され、Issueで合意されるまで導入しない
- 外部libraryを追加する場合は、必要性、license、version、保守状況を確認する

## テストと品質確認

変更内容に応じて、Pull Request前に次を実行する。

```text
python -m ruff format --check .
python -m ruff check .
python -m unittest discover -s tests -v
```

- testは正常系だけでなく、情報境界、合法手、異常入力を優先して固定する
- 外部serviceを使うtestでは本物のtokenや個人データを使用しない
- 実行できなかった確認は、理由と影響をPull RequestまたはIssueへ記録する
- code変更により利用方法、設計、制約が変わる場合は関連文書も更新する

## 秘密情報と外部成果物

次をrepositoryへcommitしない。

- `.env`、token、API key、credential
- 外部model weightおよび生成model
- 利用条件を確認していない牌譜・raw data
- 実験artifact、run出力、coverageやcache等の生成物

秘密情報らしき値や大容量binaryを発見した場合は変更を止め、内容を出力せずに
ユーザーへ報告する。外部modelやデータを導入する場合は、提供元、license、version、
取得方法、hash、再配布可否を確認する。
