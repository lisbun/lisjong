# RiichiEnv 調査記録

この文書は、[Issue #3](https://github.com/lisbun/lisjong/issues/3) で行う RiichiEnv の調査について、根拠と実測結果を分離して残すための記録である。

調査用コードは再現性を確保するために保存してよい。ただし、`src/lisjong` の Policy 正式実装には含めず、Policy からも import しない。

## 記録方針

各項目には、次のいずれかの区分を付ける。

| 区分 | 意味 |
| --- | --- |
| 公式情報 | RiichiLab、RiichiEnv の公式文書、公式リポジトリ、PyPI の配布メタデータで確認した情報 |
| 実測 | 記録した環境、バージョン、コマンドまたはコードで実際に確認した情報 |
| 推測・未確認 | 公式情報と実測のどちらでも確認できていない事項、または根拠から導いた仮説 |
| 設計判断 | 調査結果から lisjong の設計へ引き継ぐ判断 |

- 公式サンプルの出力を、lisjong 側の実測結果として扱わない。
- 実測結果には、実施日、OS、Python、RiichiEnv、依存パッケージの正確なバージョンを添える。
- 失敗も結果として残し、例外の型、メッセージ、再現手順を記録する。
- RiichiEnv のバージョンを変更して再調査した場合は、過去の結果を黙って上書きせず、対象バージョンごとに差分を残す。
- ログを残す場合は秘密情報、個人情報、モデル、学習データを含めない。

## 調査対象と情報源

公式情報の初回確認日は 2026-08-13。

| 対象 | 固定した版・情報 | 情報源 |
| --- | --- | --- |
| RiichiLab ローカル実行 | 参照時点の公式文書 | [Local Testing](https://riichi.dev/docs/local-testing) |
| RiichiEnv | `0.4.8` | [PyPI](https://pypi.org/project/riichienv/) |
| RiichiEnv ソース | tag `v0.4.8` | [公式リポジトリ](https://github.com/smly/RiichiEnv/tree/v0.4.8) |
| 配布設定 | tag `v0.4.8` の `pyproject.toml` | [pyproject.toml](https://github.com/smly/RiichiEnv/blob/v0.4.8/pyproject.toml) |
| 公開 API の説明 | tag `v0.4.8` の `README.md` | [README.md](https://github.com/smly/RiichiEnv/blob/v0.4.8/README.md) |

PyPI の provenance は、`v0.4.8` のソースコミットとして `0c1e575a3bf678e30e068149cd0fffa635d001a9` を示している。実測では、インストールされた配布物のバージョンとハッシュを別途確認する。

## 公式文書で確認した情報

### 配布条件

| 項目 | 公式情報 |
| --- | --- |
| パッケージ名 | `riichienv` |
| 対象バージョン | `0.4.8` |
| ライセンス | Apache-2.0 |
| Python 要件 | `>=3.10,<3.15` |
| 直接依存 | `pyyaml>=6.0.3`, `ipython` |
| ビルドバックエンド | Maturin `>=1.9.4,<2.0` |
| ネイティブ実装 | `riichienv._riichienv` を含み、コアは Rust で実装されている |
| ソースビルド | Rust toolchain が必要 |
| lisjong の基準環境との差 | lisjong は通常版 CPython 3.14 系に固定している。RiichiEnv 自体の公式対応範囲とは別の制約である |

PyPI には CPython 3.14 向けのプラットフォーム別 wheel が公開されている。Windows x86-64 用 wheel の公式メタデータは次のとおり。

- ファイル名: `riichienv-0.4.8-cp314-cp314-win_amd64.whl`
- SHA-256: `db49cd21308b6e479cd631bf6f4b63b95e16a1eb3cb6c2b28e64529ca938e1d2`

これは配布物の存在を示す公式情報であり、対象PCでのインストール成功を示す実測ではない。

### RiichiLab との接点

RiichiLab の公式ローカルテスト文書では、Python 3.12 以上と `pip install riichienv` が案内されている。オンライン接続時には別途 `websockets` を使うが、ローカル調査だけでは必須ではない。

エージェントの中心的な境界は、観測を受け取って行動を返す `act(obs: Observation) -> Action` である。`Observation` はそのプレイヤーから見える現在状態を表し、`legal_actions()` で合法手を取得する。

### RiichiEnv の主要 API

以下は `v0.4.8` の公式 README で確認した公開 API である。

| API | 公式文書上の役割 | lisjong で確認する点 |
| --- | --- | --- |
| `RiichiEnv(...)` | 環境を作成する | 初期化引数、既定ルール、例外 |
| `reset()` | ゲームを初期化し、プレイヤーIDから `Observation` への辞書を返す | 初期観測、再実行時の状態、seed指定可否 |
| `step(actions)` | プレイヤーIDから `Action` への辞書を適用し、次の観測辞書を返す | 同時行動、エラー、終了直前・終了後の挙動 |
| `done()` | ゲーム終了を返す | 局終了と対局終了の区別 |
| `scores()` / `ranks()` | 終了時の点数と順位を返す | 戻り値の型、取得可能な時点 |
| `Observation.legal_actions()` | 合法な `Action` の一覧を返す | 順序、同一性、空リストの有無 |
| `Observation.new_events()` | そのプレイヤーに対する新規 MJAI JSON イベントを返す | 可視情報の境界、呼び出しによる消費の有無 |
| `Observation.events` | 観測窓におけるイベント履歴を提供する | 履歴範囲と情報漏えいの有無 |
| `Observation.select_action_from_mjai(...)` | MJAI 応答を合法な `Action` へ対応付ける | 不正・曖昧な入力時の挙動 |
| `apply_event(...)` | MJAI イベントを状態へ適用する | 学習・リプレイ用途との境界 |
| `get_observation(player_id)` | 指定プレイヤーの観測を取得する | 取得可能なタイミング |
| `observe_event(event, player_id)` | イベント適用後、行動可能なら観測を返す | RiichiLab オンライン推論との共通化範囲 |

既定の環境は1局を実行する。東風戦、半荘などのモードでは、指定ルールの終了条件まで継続する。

## 実行環境と正確なバージョン

### 予定する基準環境

| 項目 | 値 | 状態 |
| --- | --- | --- |
| OS | Windows | OSエディション、build番号、architectureは未記録 |
| shell | PowerShell 7.6.3 | 利用予定。実行時に再確認する |
| Python | 通常版 CPython 3.14.6 | 利用予定。free-threaded buildは対象外 |
| RiichiEnv | 0.4.8 | インストール未実施 |
| pip | 未確認 | 実行時に記録する |
| PyYAML | 未確認 | インストール後に記録する |
| IPython | 未確認 | インストール後に記録する |

この文書を作成した作業コンテナは対象PCの実測環境ではないため、RiichiEnv の動作確認結果には含めない。

### 環境記録コマンド

PowerShell で次を実行し、必要な値だけをこの節へ転記する。コマンド出力全体をそのままコミットしない。

```powershell
Get-ComputerInfo |
    Select-Object WindowsProductName, WindowsVersion, OsBuildNumber, OsArchitecture

$PSVersionTable.PSVersion
python --version
python -c "import platform, sys; print(sys.executable); print(platform.python_implementation()); print(platform.python_build()); print(platform.platform())"
python -m pip --version
python -m pip show riichienv PyYAML IPython
python -c "from importlib.metadata import version; print(version('riichienv'))"
```

wheel を明示的に保存して検証する場合は、次のように取得物のハッシュを記録する。

```powershell
python -m pip download --only-binary=:all: --no-deps riichienv==0.4.8 --dest .tmp-riichienv
Get-FileHash .tmp-riichienv\riichienv-0.4.8-cp314-cp314-win_amd64.whl -Algorithm SHA256
```

`.tmp-riichienv` は調査用の一時ディレクトリであり、配布物をリポジトリへコミットしない。

## 実際に動かして確認した情報

2026-08-13 時点では、対象の Windows / 通常版 CPython 3.14.6 環境での実測は未実施である。

| 確認項目 | 状態 | 結果・根拠 |
| --- | --- | --- |
| CPython 3.14 用 wheel の選択 | 未実施 | pip の実出力を記録する |
| import | 未実施 | `from riichienv import RiichiEnv` を確認する |
| `reset()` | 未実施 | 戻り値の型、player ID、初期観測を記録する |
| `legal_actions()` | 未実施 | 型、件数、順序、MJAI表現を記録する |
| 最初の `step()` | 未実施 | 入出力とイベント差分を記録する |
| 1局の完走 | 未実施 | step数、終了判定、点数、順位を記録する |
| 再実行の再現性 | 未実施 | seed指定方法を確認してから比較する |
| プレイヤー別情報境界 | 未実施 | 手牌、ツモ牌、イベント履歴の可視範囲を比較する |
| 不正行動時のエラー | 未実施 | 例外の型とメッセージを記録する |
| 依存パッケージ | 未実施 | 実際に解決された正確なバージョンを記録する |
| CPU、メモリ、実行時間 | 未実施 | 最小再現の参考値だけを記録する |

実測後は、実施日と対象バージョンを付けた小節を追加する。

## 最小再現コード

次のコードは、環境ループと合法手の最小確認を目的とした調査案である。現時点では未実行であり、正式な Policy 実装ではない。

`legal_actions()` が返すリストの先頭を選ぶため、同一環境での挙動確認には使える。一方、リスト順の安定性は未確認なので、再現性の根拠にはしない。

```python
from riichienv import RiichiEnv


class FirstLegalActionAgent:
    def act(self, observation):
        actions = observation.legal_actions()
        if not actions:
            raise RuntimeError("no legal action was returned")
        return actions[0]


def main() -> None:
    env = RiichiEnv()
    agent = FirstLegalActionAgent()
    observations = env.reset()
    step_count = 0

    while not env.done():
        actions = {
            player_id: agent.act(observation)
            for player_id, observation in observations.items()
        }
        observations = env.step(actions)
        step_count += 1

        if step_count > 10_000:
            raise RuntimeError("step limit exceeded")

    print({"steps": step_count, "scores": env.scores(), "ranks": env.ranks()})


if __name__ == "__main__":
    main()
```

実行例:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install riichienv==0.4.8
python .\experiments\riichienv\minimal_run.py
```

調査用コードを残す場合の配置は `experiments/riichienv/` とし、次を守る。

- `src/lisjong` から import しない。
- `experiments` から `src/lisjong` の未確定 API へ依存させない。
- 調査対象バージョン、実行コマンド、期待する確認点を同ディレクトリの README に記録する。
- wheel、仮想環境、生ログ、モデル、学習データ、生成物をコミットしない。
- 正式実装へ移す場合は、別Issueで境界とテストを定義してから書き直す。

## 実測結果の記録テンプレート

```text
実施日:
OS / build / architecture:
PowerShell:
Python implementation / version / build:
pip:
RiichiEnv:
PyYAML:
IPython:
インストール元 wheel:
wheel SHA-256:

実行コマンド:
終了コード:
step数:
scores:
ranks:
例外（発生時）:
観測した API 差分:
再現性の条件:
情報境界の確認結果:
```

## 推測・未確認事項

以下は公式情報または実測で確定していない。

- `legal_actions()` の並び順が、実行間やバージョン間で安定するか。
- `RiichiEnv` が seed を受け取る公開 API と、その適用範囲。
- 同じ seed、ルール、行動列から同じイベント列と最終結果を再現できるか。
- `Observation`、`Action` の比較、hash、シリアライズに依存してよいか。
- `new_events()` を複数回呼んだ場合の結果と、イベント消費の意味。
- 各プレイヤーの `Observation` に、Policy が見てはいけない非公開情報が含まれないか。
- 不正な player ID、不正な行動、終了後の `step()` に対する例外仕様。
- Windows の CPython 3.14 wheel で追加の DLL またはランタイムが必要か。
- 通常版 CPython 3.14.6 での依存解決結果と、インストール後の実行安定性。
- RiichiLab のオンライン要求に含まれる行動候補と、RiichiEnv の `Action` を同一視できるか。
- RiichiEnv のイベント履歴を、Policy 入力としてそのまま採用してよいか。

## lisjong の設計へ引き継ぐ判断

### 確定して引き継ぐ判断

1. Policy は RiichiEnv の `Observation` と `Action` を直接受け取らない。
2. RiichiEnv Adapter が、RiichiEnv の観測を lisjong 内部の Policy 入力へ変換する。
3. RiichiEnv Adapter が、Policy の内部行動を RiichiEnv の合法な `Action` へ対応付ける。
4. Policy は局の進行、`reset()`、`step()`、`done()` を管理しない。
5. RiichiLab 固有の WebSocket、request ID、再接続、送受信形式を Policy に持ち込まない。
6. プレイヤーに見えてよい情報だけを Policy へ渡し、イベント履歴を自動的に全量入力しない。
7. 調査用コードと正式な Policy 実装を、配置、依存、Issue のすべてで分離する。

これらは [architecture.md](architecture.md) の Policy、RiichiEnv Adapter、RiichiLab Client の責務分離を具体化する判断である。

### 実測後に確定する判断

| 判断対象 | 確定に必要な実測 |
| --- | --- |
| Policy 入力の最小スキーマ | プレイヤー別 `Observation` と合法手の内容 |
| 内部行動の識別方法 | `Action` の属性、MJAI変換、比較方法 |
| 乱数・再現性の境界 | seed API と同一行動列での反復結果 |
| エラー変換方針 | RiichiEnv が返す例外の型と発生条件 |
| 局・対局のライフサイクル | game modeごとの `done()`、scores、ranks |
| RiichiLab との共通 Adapter 範囲 | オンライン候補とRiichiEnv合法手の対応 |
| runtime dependencyへの追加 | 通常版CPython 3.14でのインストールと最小完走 |

RiichiEnv を `lisjong` の通常依存へ追加する判断は、対象環境でインストールと最小再現が成功し、必要性と依存範囲を確認した後に別の変更として行う。

## Issue #3 完了に向けた確認項目

- [x] 公式文書、公式リポジトリ、PyPI メタデータの確認先を固定した。
- [x] 公式情報、実測、推測、設計判断を分離する形式を定義した。
- [x] 対象パッケージと調査予定環境を記録した。
- [x] 最小再現コード案を記録した。
- [ ] Windows / 通常版 CPython 3.14.6 で正確な環境情報を採取した。
- [ ] `riichienv==0.4.8` のインストールと import を確認した。
- [ ] `reset()`、合法手選択、`step()`、`done()`、結果取得を実測した。
- [ ] seed と再現性を確認した。
- [ ] プレイヤー別情報境界を確認した。
- [ ] 例外と依存条件を確認した。
- [ ] 実測結果を設計判断へ反映した。
- [ ] 調査用コードを残す場合、Policy正式実装から分離した。

## 変更履歴

| 日付 | 対象 | 内容 |
| --- | --- | --- |
| 2026-08-13 | RiichiEnv 0.4.8 | 公式情報と実測予定を初回記録。対象環境での実測は未実施 |
