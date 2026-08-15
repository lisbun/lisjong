# lisjong

Personal Japanese riichi mahjong AI for RiichiEnv and RiichiLab.

> [!IMPORTANT]
> lisjong is an independent personal Japanese mahjong AI project developed by
> [lisbun](https://github.com/lisbun). It is not affiliated with any other
> project using the LisJong or lisjong name.

## 概要

lisjongは、日本式立直麻雀AIを自作し、ローカル対局からオンライン対局まで
同じAI Policyを再利用できる形で開発・評価するプロジェクトです。

現在は初期開発段階です。最初の到達目標は、学習済みmodelの強さではなく、
決定的な最小PolicyをRiichiEnvとRiichiLabへ安全に接続し、半荘を完走する
ことです。ロードマップと完了条件は
[親Issue](https://github.com/lisbun/lisjong/issues/1)で管理します。

## 位置づけ

| 対象 | 役割 |
| --- | --- |
| lisjong | 自作麻雀AI、Policy、学習・推論、接続Adapter、評価 |
| [RiichiEnv](https://riichi.dev/docs/local-testing) | ローカル対局・開発・回帰評価環境 |
| [RiichiLab](https://riichi.dev/) | オンライン接続先 |
| Mortal | 比較対象・互換性確認用の外部AI |
| [python-study](https://github.com/lisbun/python-study) | 将来の接続先となる自作麻雀基盤を含む学習repository |

Mortalやpython-studyのコード・modelをlisjongの内部実装として取り込むことは
初期目標に含めません。

責務と依存方向の詳細は[Architecture](docs/architecture.md)を参照してください。

## 開発方針

- 初期実装は通常版CPython 3.14を基準とする
- RiichiEnv、RiichiLabなど外部環境の型・protocolをAI Policyから分離する
- 各プレイヤーから観測可能な情報だけを判断へ使用する
- Policyは合法手からactionを選択し、外部送信前にも合法手を検証する
- 再現可能なseed、version、評価条件を記録する
- AIの強さより先に、接続の正しさ、半荘完走、test可能性を確立する
- Rustは先行導入せず、profilingで必要性が確認された処理に限って検討する

## 開発環境

初期基準は通常版CPython 3.14です。[RiichiLabの公式Local Testing要件](https://riichi.dev/docs/local-testing)
であるPython 3.12以上を満たし、[RiichiEnvの配布package](https://pypi.org/project/riichienv/#files)
に含まれるCPython 3.14向けwheelを利用します。
free-threaded build（3.14t）は、依存libraryを含む互換性を個別に検証するまで
対象外とします。

repositoryを取得後、次のコマンドで開発環境を準備します。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS / Linuxでは、activateコマンドを次のように読み替えます。

```bash
source .venv/bin/activate
```

### 品質確認

ローカルとCIで同じコマンドを使用します。

```text
python -m ruff format --check .
python -m ruff check .
python -m unittest discover -s tests -v
```

testではPolicy契約、RiichiEnv Adapter、共通Policy実行境界、Local game runnerを
単体確認し、RiichiEnv 0.4.8を使う固定seed半荘のintegration testも実行します。

## Local game runner

`LocalGameRunner`は4 seatそれぞれのPolicyをRiichiEnvへ接続し、`env.done()`まで
1半荘を進行します。再現性のためseedは`RiichiEnv`のconstructorへ渡します。

```python
from lisjong.local_game_runner import LocalGameRunner
from lisjong.policies import MinimalPolicy
from lisjong.policy_contract import Seat

policies = {seat: MinimalPolicy() for seat in Seat}
result = LocalGameRunner(
    policies,
    seed=12345,
    game_mode="4p-red-half",
    max_steps=10_000,
).run()

print(result.scores, result.ranks)
```

返却される`LocalGameResult`にはseed、game mode、最終scores / ranks、step数、
Policy判断数が含まれます。`max_steps`はhang防止用の安全上限であり、対局終了前に
到達した場合は正常結果を返さず`StepLimitExceededError`で失敗します。

## RiichiLab validation

`run_validation(policy, token)`は、RiichiLab `/ws/validate`へBearer token付き
WebSocket接続し、1 validation gameを完走して`ValidationResult`(`passed`を含む)
を返します。Policy判断・Observation変換・`possible_actions` validationは
`RiichiLabSeatAdapter`(#38)を再利用し、WebSocket transport lifecycle
(`start_game` / `request_id` / `action_ack` / `end_game` / `validation_result`)
だけをこのpackageが担当します。責務境界と設計判断の詳細は
[RiichiLab WebSocket Client](docs/riichilab-client.md)を参照してください。

実`BOT_TOKEN`を使ったlive validationは、次のコマンドで学習者環境から実行します。

```powershell
$env:BOT_TOKEN = "<実RiichiLab bot token>"
python -m lisjong.riichilab_client.validation
```

`BOT_TOKEN`はrepositoryへcommitせず、環境変数から実行時に注入してください。

## RiichiLab ranked smoke test

`run_ranked_game(policy, token)`は、activeな検証用botでRiichiLab
`/ws/ranked`へ1回だけ接続し、matchmaking queueから1 full hanchanの
`end_game`まで処理して`RankedGameResult`を返します。接続自体がqueue参加であり、
join payloadは送信しません。`end_game`後の自動再queue・次game・reconnectも
行いません。

```powershell
$env:BOT_TOKEN = "<検証用RiichiLab bot token>"
python -m lisjong.riichilab_client.ranked
```

本命bot `lisjong`ではなくdev/smoke用の検証botを使用し、原則1半荘だけ実行します。
順位・score・ratingは成功条件ではありません。tokenはstdout/stderr、結果、test、
docs、Issue / PRへ保存しません。詳しい責務境界とlive確認項目は
[RiichiLab WebSocket Client](docs/riichilab-client.md)を参照してください。

## ロードマップ

1. Python package、test、CI、文書の初期整備
2. RiichiEnvの実API・依存条件の調査
3. 共通Policy境界の設計
4. 学習modelを使わない最小Policyの実装
5. RiichiEnvで最初の局終了・半荘完走
6. RiichiLab validation・ランク戦1半荘
7. 接続MVP完了後に学習・評価を開始

Issue単位の現在地は
[GitHub Issues](https://github.com/lisbun/lisjong/issues)を正本とします。

## データ・model・秘密情報

このrepositoryには、次をcommitまたは再配布しません。

- RiichiLabのBot token、API key、その他の秘密情報
- 利用条件を確認していない牌譜・学習データ
- Mortalなど外部プロジェクトのmodel weight
- 大容量の生成model、raw data、実験artifact

外部データやmodelを利用する場合は、提供元、license、version、取得方法、
hashなどを確認し、repository本体とは分離して管理します。

## 開発状況

Python 3.14、Ruff、GitHub Actions CIを開発基盤とし、共通Policy契約、最小Policy、
RiichiEnv Adapter、共通Policy実行境界、Local game runner、RiichiLab
`request_action` Adapter、RiichiLab `/ws/validate` WebSocket Clientまで
実装し、validationを完走しています。RiichiLab ranked接続(`/ws/ranked`)の
1半荘Clientも実装済みで、検証用botによるlive smoke test待ちです。学習・推論機能は
まだ実装していません。

## License

lisjong自身のsource codeは[MIT License](LICENSE)で公開します。外部library、
model、牌譜、学習データには、それぞれの提供元のlicenseと利用条件が適用されます。
