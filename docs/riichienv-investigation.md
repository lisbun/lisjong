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
| `RiichiEnv(...)` | 環境を作成する | 初期化引数、既定ルール、例外。constructorの`seed`と再現性は実測済み |
| `reset()` | ゲームを初期化し、プレイヤーIDから `Observation` への辞書を返す | 初期観測、再実行時の状態、seed指定可否。`seed`引数は確認済みだが、期待した再現性は未確認 |
| `step(actions)` | プレイヤーIDから `Action` への辞書を適用し、次の観測辞書を返す | 複数player同時行動、pon / chi競合、一部の異常入力、終了後の挙動は実測済み。ron等の競合は未確認 |
| `done()` | ゲーム終了を返す | 既定の1局終了は実測済み。東風戦・半荘等との区別は未確認 |
| `scores()` / `ranks()` | 終了時の点数と順位を返す | 1局終了時の戻り値は実測済み。取得可能な全タイミングは未確認 |
| `Observation.legal_actions()` | 合法な `Action` の一覧を返す | 順序、同一性、空リストの有無 |
| `Observation.new_events()` | そのプレイヤーに対する新規 MJAI JSON イベントを返す | 可視情報の境界と、同一objectでの連続呼び出しが非消費であることは実測済み。観測更新をまたぐ意味は未確認 |
| `Observation.events` | 観測窓におけるイベント履歴を提供する | 履歴範囲と情報漏えいの有無 |
| `Observation.select_action_from_mjai(...)` | MJAI 応答を合法な `Action` へ対応付ける | 実行経路に出現した打牌、pon、chi、noneのround-tripは実測済み。不正・曖昧な入力と未出現Actionは未確認 |
| `apply_event(...)` | MJAI イベントを状態へ適用する | 学習・リプレイ用途との境界 |
| `get_observation(player_id)` | 指定プレイヤーの観測を取得する | 取得可能なタイミング |
| `observe_event(event, player_id)` | イベント適用後、行動可能なら観測を返す | RiichiLab オンライン推論との共通化範囲 |

既定の環境は1局を実行する。東風戦、半荘などのモードでは、指定ルールの終了条件まで継続する。

### v0.4.8 ソースで確認した実装事実

`RiichiEnv.reset(seed=...)` の再現性を再検証した後、公式tag `v0.4.8` のソースを確認した。`reset(seed=...)` で受け取るseedと、実際のwall shuffleで参照されるseedの扱いには差がある。この記述は対象バージョンの実装事実であり、後述する実測結果とは区別する。

この差が仕様どおりか、意図しない挙動かは確認していないため、本書ではRiichiEnvのバグとは断定しない。

## 実行環境と正確なバージョン

### 基準環境と初回実測環境

| 項目 | 値 | 状態 |
| --- | --- | --- |
| OS | Windows | OSエディション、build番号、architectureは未記録 |
| shell | PowerShell 7.6.3 | 基準環境。初回実測時の再確認は未実施 |
| Python | 通常版 CPython 3.14.6 | `.venv` で実測。free-threaded buildは対象外 |
| Python executable | `C:\Dev\lisjong\.venv\Scripts\python.exe` | 実測済み |
| RiichiEnv | 0.4.8 | `.venv` へのインストール、import、package metadataを実測済み |
| pip | 26.2.1 | 実測済み |
| PyYAML | 未確認 | RiichiEnvの依存として確認済み。正確なバージョンは未記録 |
| IPython | 未確認 | RiichiEnvの依存として確認済み。正確なバージョンは未記録 |

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

### 2026-08-13: Windows / CPython 3.14.6 / RiichiEnv 0.4.8

`C:\Dev\lisjong\.venv` に RiichiEnv 0.4.8 をインストールし、import、初期観測、打牌、ポン応答までを実測した。今回の一時調査用 `tmp_riichienv_probe.py` は正式な調査コードとして残さない。再利用する調査コードが必要になった場合は、既存方針どおり `experiments/riichienv/` に別途整理する。

| 確認項目 | 状態 | 結果・根拠 |
| --- | --- | --- |
| `.venv` へのインストール | 確認済み | CPython 3.14.6、pip 26.2.1 で `riichienv==0.4.8` のインストールに成功した |
| CPython 3.14 用 wheel の選択 | 未確認 | インストールは成功したが、実際に取得されたwheel名とSHA-256は未記録 |
| importとpackage metadata | 確認済み | `import riichienv` に成功し、package metadataから0.4.8を確認した |
| package metadataの配布条件 | 確認済み | `python -m pip show riichienv` でApache-2.0、依存`ipython` / `pyyaml`を確認した |
| 公開型の実体 | 確認済み | `RiichiEnv` は `riichienv._riichienv.RiichiEnv`、`Observation` は `riichienv._riichienv.Observation`、`Action` は `riichienv._riichienv.Action` だった |
| `reset()` | 確認済み | Pythonの`dict`を返し、初回実測では`{0: Observation}`だった |
| `legal_actions()` | 一部確認済み | Pythonの`list`を返し、初期観測の一例では14件の打牌`Action`が含まれた。順序の安定性は未確認 |
| `Action`の属性と変換 | 確認済み | `action_type`、`actor`、`consume_tiles`、`tile`、`to_dict()`、`to_mjai()`を確認した |
| 打牌`Action`のMJAI変換 | 確認済み | 内部表現`{'type': 0, 'tile': 39, 'consume_tiles': [], 'actor': 0}`が`{"actor":0,"pai":"1p","type":"dahai"}`へ変換された |
| 最初の `step()` | 確認済み | `env.step({0: action})` に成功した。通常のツモ・打牌では、次に`Action`を要求されるseatの`Observation`が返る挙動を確認した |
| 鳴き応答と継続行動 | 確認済み | player 1の8m打牌後、player 3に`pon` / `none`が合法手として提示された。`pon`を選ぶと、次の`Observation`もplayer 3へ返り、ポン成立後の打牌`Action`が提示された |
| `Observation.to_dict()`の情報境界 | 一部確認済み | 自席手牌だけが実牌IDで、他家の`hands`は空の`list`だった。複数objectのPython公開面は一部確認済みだが、全状態・全局面と内部可変状態は未確認 |
| MJAI eventの情報境界 | 一部確認済み | `start_kyoku`では自席配牌だけが実牌で、他家配牌は`?`だった。他家ツモ牌は`?`、自席ツモ牌は実牌として通知された |
| `new_events()` | 一部確認済み | seat視点のイベント列を確認した。同一`Observation`で2回連続して呼び出しても同じevent列が返り、呼び出し自体による消費は確認されなかった |
| 1局の完走 | 確認済み | `legal_actions()[0]`を選ぶ方針で84 step後に`done() == True`となり、最終観測、点数、順位を取得した |
| 再実行の再現性 | 一部確認済み | constructorの同一seedではevent列まで再現した。`reset(seed=...)`では期待した再現性を確認できなかった |
| 不正入力と終了後の`step()` | 一部確認済み | 不正Action型、存在しないplayer ID 99、`done()`後の空actionを実測した。他の異常入力は未確認 |
| 依存パッケージ | 一部確認済み | 直接依存名は確認したが、PyYAMLとIPythonの正確なバージョンは未記録 |
| 複数player同時Action要求 | 確認済み | `Phase.WaitResponse`でplayer 0と2が同時に返るケースを確認した |
| 複数`Observation`の独立性 | 一部確認済み | Python公開面では別objectであり、片方への読み取り操作によるserialized stateの相互干渉は確認されなかった |
| Action / MJAI round-trip | 一部確認済み | 90 stepの実行経路に出現した通常・赤牌打牌、pon、chi、noneで往復に成功した |
| CPU、メモリ、実行時間 | 未実施 | 最小再現の参考値だけを記録する |

`reset()` / `step()` の戻り値の`dict`に複数playerが同時に含まれるケースを実測した。したがって、単純な「現在手番」ではなく、「その時点で`Action`選択を要求されているplayerから`Observation`へのmap」として扱う。

#### 1局完走と終了時の値

各`Observation`について`legal_actions()[0]`を選ぶ単純な方針で、既定の1局を完走した。

| 項目 | 実測結果 |
| --- | --- |
| 初期`observations`のkey | `[0]` |
| step数 | 84 |
| 終了判定 | `done() == True` |
| 最終`observations` | `{}` |
| `scores()` | `[25000, 25000, 25000, 25000]` |
| `ranks()` | `[1, 2, 3, 4]` |

この実測では点数移動のない終了となった。別の終了理由、点数移動がある局、東風戦・半荘等の終了条件を一般化する根拠にはしない。

#### seed APIと再現性

`inspect.signature()`で、次のPython公開signatureを確認した。

```text
RiichiEnv(game_mode=None, skip_mjai_logging=False, seed=None, round_wind=None, rule=None)
RiichiEnv.reset(self, /, oya=None, wall=None, round_wind=None, scores=None, honba=None, kyotaku=None, seed=None)
```

constructorと`reset()`の両方が`seed`引数を受け取る。`dir(RiichiEnv)`では、`seed`、`random`、`rng`を名前に含む公開attributeまたはmethodは確認されなかった。

`RiichiEnv(seed=12345)`から開始し、各観測で`legal_actions()[0]`を選ぶ同一方針を2回実行した。

| 比較 | scores | ranks | events |
| --- | --- | --- | --- |
| seed 12345同士 | 一致 | 一致 | 一致 |
| seed 12345と54321 | 一致 | 一致 | 不一致 |

同一constructor seedではevent列まで再現した。異なるseedでも今回の最終scoresとranksは偶然同じだったため、最終結果だけをseed有効性の判定には使わない。

一方、`RiichiEnv().reset(seed=12345)`を単純なprobeで複数回実行し、reset直後の`Observation.hand`、`Observation.new_events()`、`env.wall`の先頭部分を比較したところ、同一seedでもすべて不一致だった。異なるseedとの比較でも、これらは不一致だった。

したがって、RiichiEnv 0.4.8の今回の実測条件では、`reset(seed=...)`に期待した再現性を確認できなかった。前節のv0.4.8ソース確認結果も判断材料になるが、バグとは断定しない。

#### `step()`の異常系と終了後の挙動

| 入力 | 実測結果 |
| --- | --- |
| `env.step({0: "not-an-action"})` | `TypeError("argument 'actions': expected Action or Action3P")` |
| `done() == True`の状態で`env.step({})` | 例外は発生せず、`{}`を返した |
| `env.step({99: valid_action})` | 例外は発生せず、keyが`[0]`の観測mapを返した |

存在しないplayer ID 99のケースは単純なprobeでも再検証した。実行前後で`current_player == 0`、`turn_count == 0`、`last_discard is None`、`wall_len == 83`が一致し、観測した状態は進行せず、現在のAction要求先が再度返った。

これは今回のplayer ID 99と局面で確認した挙動であり、すべての不正player IDが常に無視されるとは一般化しない。

#### `new_events()`の連続呼び出し

同一`Observation` objectに対して`new_events()`を2回連続で呼び出した。初期観測でも次のplayerの観測でも、1回目と2回目は同じevent列だった。今回の実測では、呼び出し自体によるevent列の消費は確認されなかった。

#### 複数player同時Action要求とpon / chi競合

constructorのseed 12345で進行を観察し、step 22で次の状態を確認した。

```text
phase: Phase.WaitResponse
active_players: [0, 2]
observations: [0, 2]
```

player 0にはponとnone、player 2にはchiとnoneが合法手として提示された。

```json
{"actor":0,"consumed":["7p","7p"],"pai":"7p","type":"pon"}
{"actor":0,"type":"none"}
{"actor":2,"consumed":["8p","9p"],"pai":"7p","type":"chi"}
{"actor":2,"type":"none"}
```

player 0のponとplayer 2のchiを同時に`step()`へ渡すと、次は`Phase.WaitAct`、`active_players: [0]`、`observations: [0]`となり、player 0の`new_events()`には次が含まれた。

```json
{"actor":0,"consumed":["7p","7p"],"pai":"7p","target":1,"type":"pon"}
```

今回確認したpon / chi競合ではponが採用され、chiは採用されなかった。ronを含む競合、複数ron、その他のclaim優先順位は未確認である。

#### 複数`Observation`のPython公開面での独立性

同時に返されたplayer 0とplayer 2の`Observation`は別objectだった。片方に対して`new_events()`、`legal_actions()`、`to_dict()`を呼び出しても、両方の`serialize_to_base64()`結果は変化せず、2席のserialized stateは互いに異なっていた。少なくともPython公開面では、片方への読み取り操作によるserialized stateの相互干渉は確認されなかった。

なお、以前のprobeで同一`Observation`の`to_dict()`比較が不一致となったことが一度あったが、単純な再検証では、dict、serialized state、player ID、hands、discards、events、legal actions、meldsのすべてが一致した。以前の不一致は再現せず、設計判断の根拠には使用しない。

#### Action / MJAI round-trip

constructorのseed 12345で局を進め、各`Observation`の全`legal_actions()`について次のround-tripを確認した。

```text
Action
  -> Action.to_mjai()
  -> json.loads()
  -> Observation.select_action_from_mjai()
  -> Action
```

90 stepまで完走し、round-trip失敗による例外は発生しなかった。変換後の`Action`を再度`to_mjai()`した結果も、元のMJAI JSONと一致した。

実行経路上では、通常打牌、赤牌を含む打牌（例: `5pr`、`5sr`）、pon、chi、noneを確認した。ron、tsumo agari、riichi、kan各種など、今回出現しなかったAction種別は未確認である。

## 最小再現コード

次のコードは、環境ループと合法手の最小確認を目的とした調査案であり、正式な Policy 実装ではない。同等の`legal_actions()[0]`選択方針による1局完走は実測済みだが、この掲載コードを恒久的な調査コードとして保存したものではない。

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
- `legal_actions()`が空になる局面の有無と、Actionの比較・同一性に必要なfield。
- OSエディション、build番号、architectureと、初回実測時のPowerShellの正確なバージョン。
- 実際に取得されたwheelのファイル名とSHA-256。
- PyYAMLとIPythonの正確なバージョン。
- CPUのみでの利用条件、Windows / WSL2 / Linuxの差、native runtime要件。
- `RiichiEnv(...)`の既定rule、game modeごとの初期化引数、不正な初期化引数に対する例外。
- 公式文書上の`act(obs: Observation) -> Action`以外にAgent登録APIが存在するか、環境loopをlisjong側で管理することが正式な利用方法か。
- constructor seedの再現性が、今回と異なるrule、game mode、Action選択方針、RiichiEnvバージョンでも保たれるか。
- `reset(seed=...)`で期待した再現性を確認できなかった挙動が、他の環境やRiichiEnvバージョンでも同じか。
- 東風戦、半荘等のgame modeにおける局終了と対局終了の区別、終了時の観測。
- `scores()` / `ranks()`を取得できるすべての時点と、点数移動がある終了時の値。
- Python公開面より下の内部可変状態まで、4席の`Observation`が共有されず独立しているか。
- `Observation`、`Action` の比較やhashに依存してよいか。シリアライズの安定性とバージョン互換性。
- `new_events()`の観測更新をまたぐ差分範囲と、`Observation.events`との関係。
- `Observation.events`の履歴範囲、`apply_event(...)`、`get_observation(player_id)`、`observe_event(...)`の実際の入出力と例外。
- 今回確認していないfieldや局面を含め、各プレイヤーの`Observation`にPolicyが見てはいけない非公開情報が含まれないか。
- player ID 99以外の不正ID、合法だが要求先と異なるplayerのAction、別局面のAction、欠落・余分な応答等に対する`step()`の挙動。
- `done()`後に空でないActionを渡した場合や、`step({})`を繰り返した場合の挙動。
- 今回のWindows環境ではimportとstepに成功したが、追加のDLLまたはruntimeが既存環境に依存していないか、別環境でも同じ条件で動作するか。
- 通常版CPython 3.14.6で、複数局・東風戦・半荘を含む継続的な動作が安定するか。
- ron、複数ron等を含むclaim競合と、pon / chi以外の優先順位。
- ron、tsumo agari、riichi、kan各種等、今回出現しなかったActionのMJAI round-trip。
- 不正または曖昧なMJAI入力を`select_action_from_mjai()`へ渡した場合の挙動。
- RiichiLabの`possible_actions`とRiichiEnvの`Action`を、今回確認したMJAI round-tripを介してどこまで対応付けられるか。
- RiichiEnv のイベント履歴を、Policy 入力としてそのまま採用してよいか。
- 対局ログ、Replay、MJAI event全体の取得方法、保存形式、情報境界。
- timeoutまたはstep上限をRiichiEnv側で設定できるか。CPU、メモリ、実行時間の参考値。

## lisjong の設計へ引き継ぐ判断

### 確定して引き継ぐ判断

1. Policy は RiichiEnv の `Observation` と `Action` を直接受け取らない。
2. RiichiEnv Adapter が、RiichiEnv の観測を lisjong 内部の Policy 入力へ変換する。
3. RiichiEnv Adapter が、Policy の内部行動を RiichiEnv の合法な `Action` へ対応付ける。
4. Policy は局の進行、`reset()`、`step()`、`done()` を管理しない。
5. RiichiLab 固有の WebSocket、request ID、再接続、送受信形式を Policy に持ち込まない。
6. プレイヤーに見えてよい情報だけを Policy へ渡し、イベント履歴を自動的に全量入力しない。
7. 調査用コードと正式な Policy 実装を、配置、依存、Issue のすべてで分離する。
8. Adapterは`reset()` / `step()`の戻り値を、その時点で`Action`選択を要求されているplayerから`Observation`へのmapとして扱い、「現在手番」や単一playerを前提にしない。
9. 再現性が必要な実験では、現時点ではconstructorの`RiichiEnv(seed=...)`を使用する。`reset(seed=...)`を再現性の根拠にはしない。
10. RiichiEnvが不正player IDを必ず例外化することを前提にせず、Adapter側でAction要求playerと入力player IDの整合性を検証する。
11. 複数playerへ同時に応答する場合も、各playerの`Observation`と合法手を混同せず、seatごとに変換・検証する。
12. RiichiEnv ActionとMJAIの接続には`to_mjai()` / `select_action_from_mjai()`を利用できる可能性がある。ただし、Policyの正式なAction契約やRiichiLabとの共通化範囲は、未出現Actionと`possible_actions`の確認後に確定する。

これらは [architecture.md](architecture.md) の Policy、RiichiEnv Adapter、RiichiLab Client の責務分離を具体化する判断である。

### 実測後に確定する判断

| 判断対象 | 今回までの実測 | 確定に必要な残りの実測 |
| --- | --- | --- |
| Policy 入力の最小スキーマ | 自他家の情報境界と、複数seatのPython object・読み取り操作の独立性を一部確認 | 局終了時、未確認field、内部可変状態の独立性 |
| 内部行動の識別方法 | Action属性、MJAI変換、打牌・pon・chi・noneのround-tripを確認 | 未出現Action、不正・曖昧なMJAI入力、比較・hashの要否 |
| 乱数・再現性の境界 | constructor seedでevent列まで再現。`reset(seed=...)`では期待した再現性を確認できず | rule、game mode、バージョンを変えた場合の適用範囲 |
| エラー変換方針 | 不正Action型、player ID 99、`done()`後の空actionを確認 | 要求先不一致、欠落・余分な応答、別局面Action等 |
| 局・対局のライフサイクル | 既定の1局を完走し、`done()`、最終空map、scores、ranksを確認 | 点数移動がある終了、東風戦・半荘等 |
| RiichiLab との共通 Adapter 範囲 | 実行経路上のAction / MJAI round-tripを確認 | オンライン`possible_actions`、未出現Actionとの対応 |
| runtime dependencyへの追加 | 通常版CPython 3.14でインストール、import、1局完走を確認 | wheel hash、依存version、別環境・長時間実行の安定性 |

RiichiEnv を `lisjong` の通常依存へ追加する判断は、対象環境でインストールと最小再現が成功し、必要性と依存範囲を確認した後に別の変更として行う。

## Issue #3 完了に向けた確認項目

- [x] 公式文書、公式リポジトリ、PyPI メタデータの確認先を固定した。
- [x] 公式情報、実測、推測、設計判断を分離する形式を定義した。
- [x] 対象パッケージと調査予定環境を記録した。
- [x] 最小再現コード案を記録した。
- [ ] Windows / 通常版 CPython 3.14.6 で正確な環境情報を採取した（Python、pip、実行ファイルは確認済み。OS build / architecture等は未確認）。
- [x] `riichienv==0.4.8` のインストールと import を確認した。
- [x] `reset()`、合法手選択、最初の`step()`を実測した。
- [x] 既定の1局終了、`done()`、最終空map、`scores()`、`ranks()`を実測した。
- [ ] seedと再現性を確認した（公開API、constructor seedの再現性、`reset(seed=...)`の非再現性は実測済み。適用範囲が残る）。
- [ ] プレイヤー別情報境界を確認した（手牌、ツモ牌、複数objectのPython公開面は確認済み。未確認fieldと内部状態が残る）。
- [ ] 例外と依存条件を確認した（一部の`step()`異常系は確認済み。依存version、wheel、追加異常系が残る）。
- [x] 初回実測結果を設計判断へ反映した。
- [x] 一時調査用`tmp_riichienv_probe.py`を正式な調査コードとして残さない方針を記録した。
- [ ] 正式な調査用コードを残す場合、`experiments/riichienv/`へ整理してPolicy正式実装から分離した。

## 変更履歴

| 日付 | 対象 | 内容 |
| --- | --- | --- |
| 2026-08-13 | RiichiEnv 0.4.8 | 公式情報と実測予定を初回記録。対象環境での実測は未実施 |
| 2026-08-13 | RiichiEnv 0.4.8 | Windows / CPython 3.14.6でインストール、主要型、初期観測、打牌、情報境界、ポン応答を初回実測 |
| 2026-08-13 | RiichiEnv 0.4.8 | 1局完走、seed、異常系、複数player同時要求、pon / chi競合、Observation独立性、Action / MJAI round-tripを追加実測 |
