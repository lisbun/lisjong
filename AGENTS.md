# AGENTS.md

## 適用範囲

このファイルはrepository全体へ適用する恒常的な作業規則である。
Issueまたはユーザーの明示的な指示が本書と異なる場合は、その指示を優先する。

以下の作業分担は現時点のdefault responsibilityであり、恒久的なtool制約や禁止ではない。
利用可能なtool、credit、作業内容、学習目的に応じて担当や作業場所を変更できる。
一方、本書でmandatoryとする安全境界と承認境界は維持する。

## デフォルトの作業分担

- Git変更を伴わない方針・設計相談、Issue整理、実装方針・PR・実測結果のレビュー、
  依頼文やGitHubへ記録する内容の作成は、通常のChatGPT conversationをdefaultとする。
  相談・レビューだけを担当する場合、明示的な依頼なしにbranchを先行作成しない
- source code、test、refactor、実装と不可分な小規模文書、品質確認、Git作業は、
  現時点ではClaude Codeをdefaultの変更担当とする
- `AGENTS.md`、README、設計・調査文書、文書間整合やstale documentationの整理は、
  現時点ではChatGPT WORKをdefaultの変更担当とする
- 学習目的でユーザー自身が操作する作業、RiichiLabへのlive接続、credentialを必要とする
  操作、ローカルOS・GUI・network依存の確認は、ユーザー管理環境をdefaultとする。
  AIはcredentialの受領を前提にせず、原則として実行command、期待結果、確認項目を示す
- 上記は専属担当を定めない。必要に応じてAI間でcode・文書の担当を入れ替え、
  将来のremote/cloud環境を含めて適切な作業場所を選べる

## 開発フロー

### Issue、branch、Pull Request

- GitHub Issueを作業の目的、スコープ、完了条件の正本とする
- `main`へ直接pushせず、実際にGit上のファイル変更を担当する作業主体が、
  対応Issueの主作業branchを作成する。相談・設計・レビューだけの担当は、
  実装担当に先立って将来用のbranchを確保しない
- 原則として1 Issueにつき1つの主作業branchを使い、概ね1つのPull Requestで完結させる。
  同じIssueに理由なく複数の並行した主作業branchを作らない
- Issueが大きい場合、stacked PR、先行refactor、調査と実装の分離など合理的な理由が
  あれば複数branch・PRを使える。その理由、各branch・PRの責務、Issue全体に対する
  担当範囲をIssueまたはPRへ記録する
- 1つのPull Requestでは1つの主目的を扱い、無関係な変更を混ぜない
- Git変更担当AIは、必要なIssue作成、branch作成、ファイル変更、品質確認、commit、push、
  Pull Request作成、Issueとの関連付け、Ready for review化までを追加承認なしで進めてよい
- PRのmergeでIssue全体が完了する場合は`Closes #123`等を使用し、途中PRや一部変更だけを
  扱う場合は`Refs #123`等、Issueを早期closeしない関連付けを使用する

### mergeと完了後cleanup

- Pull Requestのmergeにはユーザーの明示的な承認を必要とする
- ユーザーがmergeを承認した時点で、そのmergeに予定された次の定型cleanupも承認済みと
  みなし、追加承認を必要としない
  - `Closes #...`による対応Issueの自動close
  - merge後、完了条件を満たし追加作業がないIssueの手動close
  - 不要になったremoteのIssue主作業branch・補助branchの自動または手動削除
- merge後は原則として変更担当AIが完了条件とcleanupを確認する。明示的に引き継いだ
  レビュー担当AIも、同じ条件を確認できる場合は実行できる
- PRをmergeせずcloseした場合は、Issueを機械的にcompletedとしてcloseせず、継続、
  not planned、別PRへの引継ぎ等を確認する。remote branchは今後利用しないことが明らかな
  場合だけ削除する
- `main`等の長期branchはcleanup対象にしない

### repository settingsとその他の承認境界

- repository settings変更には個別のユーザー承認を必要とする。ただし、merged PRの不要な
  head branchを自動削除する設定など、visibility、branch protection、Actions・security・
  permission、secret、外部公開、課金へ影響しない安全なcleanup設定に限り追加承認を不要とする
- 上記の承認済みmergeに伴う定型cleanupを除き、破壊的操作、外部公開、課金、認証情報の
  使用は、対象と影響を示して承認を得る

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

- 文書だけの変更では最低限`git diff --check`を実行し、source code・test codeの変更が
  含まれないことを確認する。Markdown lint等が標準化されている場合はそれも実行する
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
