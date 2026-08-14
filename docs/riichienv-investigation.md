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

RiichiLabの`request_action`には、少なくとも`request_id`、`possible_actions`、base64化された`observation`が含まれる。`observation`はRiichiEnvの`Observation.deserialize_from_base64()`で復元でき、公式の利用例は、復元した観測をAgentへ渡し、返されたRiichiEnv `Action`を`to_mjai()`で変換してオンライン応答へ利用する構造である。

この公式仕様は、RiichiEnvの`Observation` / `Action`を使うAgent境界をローカルとオンラインで再利用できる可能性を示す。一方、WebSocket、`request_id`、`possible_actions`の整合確認、timeout、`action_ack`、再接続はRiichiLab Client固有の責務である。

### RiichiEnv の主要 API

以下は `v0.4.8` の公式 README で確認した公開 API である。

| API | 公式文書上の役割 | lisjong で確認する点 |
| --- | --- | --- |
| `RiichiEnv(...)` | 環境を作成する | 初期化引数、既定ルール、例外。constructorの`seed`と再現性は実測済み |
| `RiichiEnv.game_mode` | 選択されたgame modeを提供する | Python公開型とmodeごとの終了条件 |
| `reset()` | ゲームを初期化し、プレイヤーIDから `Observation` への辞書を返す | 初期観測、再実行時の状態、seed指定可否。`seed`引数は確認済みだが、期待した再現性は未確認 |
| `step(actions)` | プレイヤーIDから `Action` への辞書を適用し、次の観測辞書を返す | 複数player同時行動、pon / chi競合、一部の異常入力、終了後の挙動は実測済み。ron等の競合は未確認 |
| `done()` | ゲーム終了を返す | single / east / halfの終了と延長を実測済み。他rule・score条件は未確認 |
| `scores()` / `ranks()` | 終了時の点数と順位を返す | single / east / half終了時の戻り値は実測済み。取得可能な全タイミングは未確認 |
| `Observation.legal_actions()` | 合法な `Action` の一覧を返す | 順序、同一性、空リストの有無 |
| `Observation.new_events()` | そのプレイヤーに対する新規 MJAI JSON イベントを返す | 可視情報の境界、同一objectでの連続呼び出しが非消費であること、seatごとのObservation更新間のdelta挙動を一部実測済み |
| `Observation.events` | 観測窓におけるイベント履歴を提供する | `new_events()`と同内容になる局面を含むseat別delta挙動を一部実測済み。全局面・全versionの履歴範囲は未確認 |
| `Observation.select_action_from_mjai(...)` | MJAI 応答を合法な `Action` へ対応付ける | 実行経路に出現した打牌、pon、chi、noneのround-tripは実測済み。不正・曖昧な入力と未出現Actionは未確認 |
| `Observation.deserialize_from_base64(...)` | RiichiLabから受け取るserialized observationを復元する | RiichiLab Clientと共通Agentの境界 |
| `RiichiEnv.mjai_log` | 対局のMJAI event列を提供する | Python公開形、完全ログとseat別eventの情報境界 |
| `apply_event(...)` | MJAI イベントを状態へ適用する | 学習・リプレイ用途との境界 |
| `get_observation(player_id)` | 指定プレイヤーの観測を取得する | 取得可能なタイミング |
| `observe_event(event, player_id)` | イベント適用後、行動可能なら観測を返す | RiichiLab オンライン推論との共通化範囲 |

既定の環境は1局を実行する。4人麻雀について、少なくとも`4p-red-single`、`4p-red-east`、`4p-red-half`がある。東風戦、半荘などのmodeではgame-end conditionsや延長条件を含む指定ルールの終了条件まで継続するため、mode名だけから特定の基準局で必ず終了するとは限らない。

### v0.4.8 ソースで確認した実装事実

`RiichiEnv.reset(seed=...)` の再現性を再検証した後、公式tag `v0.4.8` のソースを確認した。`reset(seed=...)` で受け取るseedと、実際のwall shuffleで参照されるseedの扱いには差がある。この記述は対象バージョンの実装事実であり、後述する実測結果とは区別する。

この差が仕様どおりか、意図しない挙動かは確認していないため、本書ではRiichiEnvのバグとは断定しない。

## 実行環境と正確なバージョン

### 基準環境と初回実測環境

| 項目 | 値 | 状態 |
| --- | --- | --- |
| OS | Windows 10 Home | `Get-ComputerInfo`で`WindowsVersion: 2009`、`OsBuildNumber: 26200`、`OsArchitecture: 64 ビット`を実測済み |
| shell | PowerShell 7.6.3 | 対象PCで実測済み |
| Python | 通常版 CPython 3.14.6 | `.venv` で実測。free-threaded buildは対象外 |
| Python executable | `C:\Dev\lisjong\.venv\Scripts\python.exe` | 実測済み |
| RiichiEnv | 0.4.8 | `.venv` へのインストール、import、package metadataを実測済み |
| pip | 26.2.1 | 実測済み |
| PyYAML | 6.0.3 | 対象`.venv`で、RiichiEnvからの依存として実測済み |
| IPython | 9.16.1 | 対象`.venv`で、RiichiEnvからの依存として実測済み |

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
| CPython 3.14 Windows x86-64 wheel | 確認済み | `pip download`で`riichienv-0.4.8-cp314-cp314-win_amd64.whl`（1,394,738 bytes）を明示取得した |
| 明示取得したwheelのSHA-256 | 確認済み | `DB49CD21308B6E479CD631BF6F4B63B95E16A1EB3CB6C2B28E64529CA938E1D2`。大文字・小文字を除いて公式メタデータ記載値と一致した |
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
| 依存パッケージ | 確認済み | 対象`.venv`でPyYAML 6.0.3、IPython 9.16.1をRiichiEnvからの依存として確認した |
| 複数player同時Action要求 | 確認済み | `Phase.WaitResponse`でplayer 0と2が同時に返るケースを確認した |
| 複数`Observation`の独立性 | 一部確認済み | Python公開面では別objectであり、片方への読み取り操作によるserialized stateの相互干渉は確認されなかった |
| Action / MJAI round-trip | 一部確認済み | 90 stepの実行経路に出現した通常・赤牌打牌、pon、chi、noneで往復に成功した |
| `game_mode` | 確認済み | 既定環境ではPythonの`int`値`0`として取得した。single / east / halfを実行し、`done()`までの局遷移を確認した |
| 対局終了 | 確認済み | 今回の3 modeで`end_game`、`done() == True`、最終観測`{}`の対応を確認した |
| `mjai_log` | 確認済み | Pythonの`dict`を要素とする`list`で、全playerの実配牌・実ツモ牌を含む完全対局ログだった |
| `Observation.new_events()`の公開形 | 確認済み | JSON文字列を要素とする`list`で、seat視点に非公開牌が`?`へmaskされていた |
| CPU、メモリ、実行時間 | 未実施 | 最小再現の参考値だけを記録する |

`pip download`で明示取得したwheelのファイル名、サイズ、SHA-256を確認し、公式メタデータ記載hashとの一致を確認した。これは、初回の`pip install`が必ずこの同一ファイルを使用したことの直接証明ではない。確認後に`.tmp-riichienv`を削除し、wheelはリポジトリへ残していない。

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

#### game modeと対局終了

既定の`RiichiEnv(seed=12345)`では、`env.game_mode`はenumではなくPythonの`int`値`0`として取得された。`reset()`直後は`round_wind == 0`、`kyoku_idx == 0`、`oya == 0`、`done() == False`、観測mapのkeyは`[0]`だった。

constructorのseed 12345を使用し、各`Observation`で`legal_actions()[0]`を選ぶ方針で3 modeを実行した。

| mode | 実測した局遷移 | step数 | scores | ranks |
| --- | --- | ---: | --- | --- |
| `4p-red-single` | 東1のみ: `(round_wind=0, kyoku_idx=0, oya=0, honba=0)` | 90 | `[25000, 25000, 25000, 25000]` | `[1, 2, 3, 4]` |
| `4p-red-east` | 東1～東4、南1～南4 | 682 | `[25000, 25000, 25000, 25000]` | `[1, 2, 3, 4]` |
| `4p-red-half` | 東1～東4、南1～南4、西1～西4 | 1042 | `[27000, 23000, 27000, 23000]` | `[1, 3, 2, 4]` |

`4p-red-east`では東4終了後の`end_kyoku`に続き、`bakaze: "S"`、`kyoku: 1`の`start_kyoku`が生成され、南1へ進行した。南4終了後に`end_kyoku`、`end_game`が生成され、`env.done() == True`となった。

実測した`4p-red-east`の`(round_wind, kyoku_idx, oya, honba)`遷移は次のとおり。

```text
(0, 0, 0, 0)
(0, 1, 1, 1)
(0, 2, 2, 2)
(0, 3, 3, 3)
(1, 0, 0, 4)
(1, 1, 1, 5)
(1, 2, 2, 6)
(1, 3, 3, 7)
```

`4p-red-half`では南4終了後に`bakaze: "W"`、`kyoku: 1`の`start_kyoku`が生成され、西1へ進行した。西4終了後に`end_kyoku`、`end_game`が生成され、`env.done() == True`となった。

3 modeとも終了時の観測mapは空だった。今回のseed、Action選択、ruleで観測した結果であり、eastが常に南4、halfが常に西4で終了するとは一般化しない。RiichiEnv利用時の対局終了は`round_wind`や`kyoku_idx`から独自に推測せず、`env.done()`を正本とする。

#### `mjai_log`とseat別eventの情報境界

RiichiEnv 0.4.8の今回のprobeでは、`env.mjai_log`はPythonの`dict`を要素とする`list`だった。reset直後の`start_kyoku`には4player全員の実際の配牌が含まれ、通常進行中の`tsumo`にも他家を含む実際のツモ牌が記録されていた。

一方、`Observation.new_events()`はJSON文字列を要素とする`list`だった。`start_kyoku`では自席の配牌だけが実牌で、他3playerの配牌は`?`へmaskされていた。他家の`tsumo`も`pai: "?"`となり、自席のツモだけが実牌として取得された。

したがって、今回の実測範囲では次の情報境界がある。

| API | Python公開形 | 情報境界 | 想定用途 |
| --- | --- | --- | --- |
| `env.mjai_log` | `list[dict]` | 全playerの非公開情報を含み得る完全対局ログ | Replay、調査、監査等のPolicy外用途 |
| `Observation.new_events()` | JSON文字列の`list` | 当該seatの視点にmaskされたevent | 必要な場合のseat別Policy入力候補 |

`env.mjai_log`をPolicyへ直接渡してはいけない。eventをPolicy入力へ使う場合も、seat別`Observation`の情報境界を維持する。このevent実測だけから、`Observation`の全fieldについて情報漏えいがないとは一般化しない。

#### RiichiLabとローカルRiichiEnvの共通化範囲

RiichiLab公式仕様とローカル実測を合わせると、RiichiEnvの`Observation`を受け取って`Action`を返す外部境界は、ローカルとオンラインで共通化できる。現在のlisjongでは、対局やsessionのオーケストレーションをLocal game runnerまたはRiichiLab Clientが担い、RiichiEnv AdapterがRiichiEnv外部型とlisjong内部型を変換し、Policyは環境非依存の内部型だけを扱う。

| 層 | 責務 |
| --- | --- |
| Policy contract / implementation | 1 seat・1 decision分のlisjong内部型を使用してActionを選択する。RiichiEnv、RiichiLab、mjai、WebSocket固有型や対局・session lifecycleを所有しない |
| Local game runner | `RiichiEnv`の生成・初期化、`reset()`、`step()`、`done()`、ローカル対局loop、複数player要求の進行管理、seatごとのPolicy判断のオーケストレーション、検証済みAction集合の返却を担当する |
| RiichiEnv Adapter | RiichiEnv `Observation` / `Action`とlisjong内部型の変換、seat別の情報境界維持、Policy入力・内部合法Action候補への変換、Policy選択Actionと元のRiichiEnv合法Actionの対応付け・再検証を担当する。対局loopや`reset()` / `step()` / `done()`を管理しない |
| RiichiLab Client | WebSocket等の通信、認証、接続、受信、送信、`request_id`、`possible_actions`、timeout / time budget、`action_ack`、Policy判断のオーケストレーション、送信前のオンライン合法性再検証、session終了処理を担当する |

RiichiLab Clientはserialized observationを`Observation.deserialize_from_base64()`で復元し、RiichiEnv Adapterを通じてPolicy入力と内部合法Action候補へ変換する。Policy判断後は、Adapterが元のRiichiEnv合法Actionへ対応付けて再検証し、Clientが`possible_actions`との整合を送信前に再検証して、`to_mjai()`による応答形式へ変換する。WebSocket protocol固有情報をPolicyへ持ち込まない。

初期スコープでは、オンライン対局中に接続が切断された場合、ゲーム途中からの再接続・復旧を試みず、安全にsessionを終了する。将来の再接続対応を永久に禁止するものではなく、RiichiLabの仕様と必要性を確認し、別Issueで合意した場合に検討する。

現在の責務分離と依存方向は[Architecture](architecture.md)を、Policyの詳細契約は[Policy契約](policy-contract.md)を正本とする。本書は外部仕様、実測結果、およびそれらからlisjongの設計へ引き継いだ根拠を記録する。

今回確認済みなのは、RiichiLabが`possible_actions`を合法手候補として提示する公式仕様、serialized observationを復元する公式境界、`Action.to_mjai()`をオンライン応答に使う公式設計、およびローカルRiichiEnvでの一部ActionのMJAI round-tripである。実際のWebSocket requestに含まれる`possible_actions`と生成Actionの生JSON dictが全fieldで完全一致することは実測していない。照合実装とオンライン実測は後続のRiichiLab Client側で行う。

### 2026-08-14: Issue #11向け追加実測

Windows / CPython 3.14.6 / RiichiEnv 0.4.8の同じ基準環境で、Policy入力と内部Actionの設計根拠を補う追加probeを実施した。以下は今回の実測結果であり、RiichiEnvの全version・全局面へ一般化しない。

#### `Observation`公開属性と`to_dict()`の差

Pythonから直接参照可能な`Observation`属性として、少なくとも次を確認した。

```text
player_id
hand
hands
melds
discards
dora_indicators
scores
riichi_declared
honba
riichi_sticks
round_wind
oya
kyoku_index
waits
is_tenpai
tsumogiri_flags
riichi_sutehais
last_tedashis
last_discard
drawn_tile
events
```

同じ追加probeで`Observation.to_dict()`に含まれたkeyは次のとおりだった。

```text
discards
dora_indicators
events
hands
honba
legal_actions
melds
oya
player_id
riichi_declared
riichi_sticks
round_wind
scores
```

少なくとも`drawn_tile`、`kyoku_index`、`waits`、`is_tenpai`、`tsumogiri_flags`、`riichi_sutehais`、`last_tedashis`、`last_discard`はPython公開属性として存在する一方、今回の`to_dict()`には含まれなかった。初期局面では`drawn_tile == 62`で、物理牌ID `62`が`hand`内にも存在した。これはdrawn tileを現在手牌内のmetadataとして扱える可能性を支持する実測だが、Policy入力での具体表現は未確定である。

したがって、`Observation.to_dict()`だけをObservationの完全表現として扱わない。Policy入力へ採用するfieldは、Issue #11で許可リストとして個別に明示し、境界側で取得・変換する必要がある。

#### `events` / `new_events()`のseat単位delta挙動

同一`Observation`に対する`new_events()`の2回連続呼び出しは、今回確認した全局面で同じ内容を返した。呼び出し自体によるevent消費は確認されなかった。

複数のObservationをseat別に追跡すると、あるseatが連続してObservationを受け取った場合は新たに発生したeventだけが返り、そのseatがしばらくObservationを受け取らなかった場合は、その間に発生した複数player分のseat-visible eventが次のObservationでまとめて返った。

RiichiEnv 0.4.8の今回の実測範囲では、`events` / `new_events()`を「そのseatについて、前回Observationが構築されてから今回Observationが構築されるまでに増えたseat-visible MJAI event群」と捉えることと整合した。ただし、全version・全局面における厳密な契約や、長期間の蓄積挙動までは確認していない。

この結果は、raw event deltaをそのままPolicyへ渡すより、境界側でseat-visible eventを現在状態へ正規化し、Policyへsnapshotを渡す設計の根拠になる。どのcomponentがmaterialized stateを所有するかはArchitecture側で別途確定する。

#### Discard Actionの物理牌identityとMJAI表現

drawn tileと同種牌をもう1枚持つ局面で、次の2件が別々の合法なDiscard Actionとして存在した。

| 物理牌ID | MJAI牌 | `is_drawn` |
| ---: | --- | --- |
| 78 | `"2s"` | `False` |
| 79 | `"2s"` | `True` |

この局面の`drawn_tile`は`79`だった。両Actionの`to_mjai()`は、ともに次の同一表現になった。

```json
{"actor":3,"pai":"2s","type":"dahai"}
```

同じ局面では、drawn tileではない同種牌の複数合法Actionとして、物理牌ID `85`と`86`がいずれもMJAI牌`"4s"`へ変換される例も確認した。RiichiEnvの物理牌identity上は別Actionでも、MJAI表現上は同一になる場合がある。

この実測は、RiichiEnvの物理牌IDをPolicyへそのまま持ち込まず、物理牌差がゲーム上の意味差を生む場合だけ意味fieldへ正規化する判断を支持する。特にdrawn tileと同種牌を捨てる場合の手出し / ツモ切り差は保持する必要がある。確定したDiscard InternalActionのidentityと代表変換例は[Action identity](action-identity.md)を正本とする。

#### 通常進行のwallと鳴かれたdiscard

初期局面では既存実測どおり`len(env.wall) == 83`だった。通常進行では、chi / pon等でdecision数が増えてもwall長は減らず、通常ツモが発生したときに`83 -> 82 -> 81 -> ...`のように1ずつ減少した。

seat 0の打牌をseat 1がchiした局面では、chi前後とも`discards[0] == [9]`で、元の打牌は`Observation.discards`から削除されなかった。seat-visible eventには次が含まれた。

```json
{"actor":1,"consumed":["2m","4m"],"pai":"3m","target":0,"type":"chi"}
```

RiichiEnv 0.4.8の今回の実測範囲では、`Observation.discards`は「現在卓上に残っている牌だけの表示状態」より、「各playerが行った打牌履歴」として扱うことと整合した。他の鳴き種別・全局面には一般化しない。

Policy snapshotに`called_by`等を含める場合は、`dahai`に続く`chi` / `pon` / `daiminkan`をseat-visible eventから対応するdiscard occurrenceへ反映する方式が候補になる。具体schemaと状態所有者は未確定である。

#### リーチ状態遷移

seed=6、step=15、seat=1で、次の遷移を確認した。

| 観測点 | `riichi_declared` | `riichi_sutehais` | 宣言者score | `riichi_sticks` | 主なevent |
| --- | --- | --- | ---: | ---: | --- |
| reach Action直前 | `[False, False, False, False]` | `[None, None, None, None]` | 25000 | 0 | reach関連eventなし |
| reach Action後 / 宣言牌discard前 | `[False, False, False, False]` | `[None, None, None, None]` | 25000 | 0 | `reach` |
| 宣言牌discardを含む`step()`後 | `[False, True, False, False]` | `[None, None, None, None]` | 24000 | 1 | `dahai` → `reach_accepted` |
| `reach_accepted`後の宣言者Observation | `[False, True, False, False]` | `[None, None, None, None]` | 24000 | 1 | 通常進行 |

reach Action実行後、宣言牌discard前の独立Observationが存在した。その時点では`reach` eventは発生済みだが、`riichi_declared == False`、scoreは25000、`riichi_sticks == 0`だった。宣言牌discard後の同じ`step()`内で`reach_accepted`まで進み、次に取得できたObservationでは`riichi_declared == True`、scoreは24000、`riichi_sticks == 1`だった。

`riichi_declared == True`だが`reach_accepted`未発生という独立Observationは今回確認されなかった。`riichi_sutehais`は観測範囲で`None`のままであり、宣言牌位置の確認済み取得元とは扱わない。

単純なboolだけでは「reach Action実行済み・宣言牌discard前」を表現できないため、将来のPolicy入力で`NONE` / `DECLARED` / `ACCEPTED`等の段階を区別できる表現を検討する根拠になる。具体的な型と導出規則は未確定である。

#### daiminkan / ankan / kakanと嶺上ツモ時のwall

次の3ケースを実測した。

| kan kind | 条件 | 槓直前wall長 | 槓処理後wall長 | 嶺上ツモ後wall長 | dora追加タイミング |
| --- | --- | ---: | ---: | ---: | --- |
| daiminkan | seed=4 / step=83 / seat=3 | 17 | 16 | 16 | 槓処理後ではなく、その後の打牌を処理した次Observation |
| ankan | seed=4 / step=76 / seat=2 | 21 | 20 | 20 | 槓処理と同一Observation |
| kakan | seed=2 / step=46 / seat=0 | 45 | 44 | 44 | 槓処理後ではなく、その後の打牌を処理した次Observation |

槓処理と嶺上ツモは同一`env.step()`内で処理され、別Observationとしては取得できなかった。event順序は概ね次のとおりだった。

```text
daiminkan -> tsumo -> 後続Observationで dora -> dahai
ankan     -> dora -> tsumo
kakan     -> tsumo -> 後続Observationで dora -> dahai
```

通常ツモと同様に、3種類すべてで嶺上ツモ時にwall長が1減少した。試験的に「槓直前の`len(env.wall)`から、以降の`new_events()`内の`tsumo` event数を引く」計算を行うと、今回の3ケースでは実際のwall長と一致した。

この結果は、環境非依存な`live_wall_tiles_remaining`概念をPolicy入力候補として検討する根拠を強める。一方、`len(env.wall)`そのものをPolicy fieldの意味とはせず、`tsumo` eventを数えれば常に正確に更新できるとも確定しない。

未確認範囲には、複数ツモ・複数槓を跨ぐ長期間、宣言者以外のseat視点、終局・流局・海底 / 河底、対局全体での累積誤差、RiichiLab実オンライン経路が含まれる。Policy入力への採否、境界での正確なcounter構築、local / online共通化は分けて判断する。

#### materialized state設計への示唆

今回までの実測から、`discards[].tsumogiri`、`discards[].order`、`discards[].called_by`、`players[].riichi`、live-wall関連状態等は、単一Observationを無加工変換するだけでは不足する可能性がある。

seat-visible event deltaを継続処理して現在状態へ正規化し、Policyには不変snapshotを渡す設計を支持する根拠としてIssue #11へ引き継ぐ。ただし、RiichiEnv AdapterまたはRiichiLab Clientが必ずmaterialized stateを所有するとは本書で確定せず、責務配置は`docs/architecture.md`で別途決定する。

### 2026-08-14: Issue #27向け追加実測

[Issue #27](https://github.com/lisbun/lisjong/issues/27)のスコープに沿って、Issue #23「RiichiEnv Adapterを実装する」のproduction実装で未確認事項を推測しないため、未確認だった7 Action variant、target/from_seat解決、representative選択に使えるphysical field、kakan元pon解決、live-wall算出、event重複防止を追加実測した。このIssueは調査Issueであり、production Adapterコードは追加していない。調査用スクリプトは一時領域だけに置き、正式な調査コードとしては残していない。

#### 実測環境（既存基準環境とは別環境）

今回はWindows実機ではなく、コンテナ環境で実測した。既存の基準環境（Windows / CPython 3.14.6）とは異なる環境であるため、区別して記録する。

| 項目 | 値 |
| --- | --- |
| OS | Linux 6.18.5-fc-v20（glibc 2.39） |
| Python | 通常版CPython 3.11.15 |
| RiichiEnv | 0.4.8（`pip show`で確認。当該環境の`.venv`へ新規インストール） |
| PyYAML | 6.0.3（RiichiEnvからの依存として確認） |
| IPython | 9.16.1（RiichiEnvからの依存として確認） |

`riichienv==0.4.8`のLinux / cp311向けwheelが存在し、このコンテナ環境へインストールできることを実測した。既存文書はWindows x86-64 wheelの存在だけを公式情報として確認していたが、Linux wheelの存在と実際のインストール成功は今回はじめて実測した。RiichiEnvのPython API自体の挙動はOS非依存と推測されるが、本節の実測結果は基準環境と別環境として区別し、基準環境表へは統合しない。

#### 未確認だった7 Action variantの実測

多数seedで`RiichiEnv(seed=...)`を実行し、`legal_actions()`に対象`ActionType`が出現した時点でそのActionを選択する探索方針で、次の7 variant全てを実際に出現させた（各1事例）。

| variant | `ActionType` | 実測した公開属性 | `to_mjai()`実測例 |
| --- | --- | --- | --- |
| riichi | `RIICHI` | `tile=None`、`consume_tiles=[]` | `{"actor":2,"type":"reach"}` |
| daiminkan | `DAIMINKAN` | `tile`=召し上げ牌、`consume_tiles`=手牌側3枚 | `{"actor":2,"consumed":["8s","8s","8s"],"pai":"8s","type":"daiminkan"}` |
| ankan | `ANKAN` | `tile`=4枚のうち1枚、`consume_tiles`=4枚全部 | `{"actor":3,"consumed":["W","W","W","W"],"pai":"W","type":"ankan"}` |
| kakan | `KAKAN` | `tile`=追加牌、`consume_tiles`=既存pon側3枚 | `{"actor":0,"consumed":["1p","1p","1p"],"pai":"1p","type":"kakan"}` |
| ron | `RON` | `tile`=和了牌、`consume_tiles=[]` | `{"actor":2,"type":"hora"}` |
| tsumo | `TSUMO` | `tile`=実測範囲内では`drawn_tile`と一致 | `{"actor":1,"type":"hora"}` |
| kyuushu_kyuuhai | `KYUSHU_KYUHAI` | `tile=None`、`consume_tiles=[]` | `{"actor":3,"type":"ryukyoku"}`（reason等の追加fieldは含まない） |

7 variant全てで`Action -> to_mjai() -> json.loads() -> Observation.select_action_from_mjai() -> Action`のround-tripに成功し、再変換後の`to_mjai()`も元の値と一致した。ただし各variant1事例のみの実測であり、全局面・全出現パターンを網羅したものではない。riichiとkan各種の局面遷移は#11向け追加実測で既に一部確認済みだったが、ron・tsumo・kyuushu_kyuuhaiの個別round-tripは今回はじめて実測した。

#### Actionのtarget/from_seatに関する追加実測

`Action`クラスの公開属性は全variant共通で`action_type`、`actor`、`consume_tiles`、`tile`、`to_dict()`、`to_mjai()`だけであり、`target`に相当する属性はどのvariantにも存在しないことを`dir()`で確認した。さらに、Chi / Pon / Daiminkan / Ronの`to_mjai()`出力そのものにも`target`が含まれないことを実測した。

```text
Chi.to_mjai()       -> {"actor":1,"consumed":["2m","4m"],"pai":"3m","type":"chi"}
Pon.to_mjai()       -> {"actor":2,"consumed":["2p","2p"],"pai":"2p","type":"pon"}
Daiminkan.to_mjai() -> {"actor":2,"consumed":["8s","8s","8s"],"pai":"8s","type":"daiminkan"}
Ron.to_mjai()       -> {"actor":2,"type":"hora"}
```

（`target`が現れるのは、Action適用後に`new_events()`へ記録される結果eventの側であり、Action自体にはtargetが乗らない。この区別は既存文書では明示していなかった。）

一方、`Observation.last_discard`が「直近に打牌したseat」を表す整数値を返すことを実測した。値域は実測範囲で常に`{0, 1, 2, 3}`であり、seed=7の対局では87回の比較全てで直近`dahai` eventの`actor`と一致した。ron候補が出現した3事例（seed=19, 170, 200）でも、`last_discard`は実際にron対象となった打牌のseatと一致した。

したがって、`ChiAction.target` / `PonAction.target` / `DaiminkanAction.target` / `RonAction.target`は、RiichiEnvの`Action`側からではなく、そのdecision時点の`Observation.last_discard`から解決できることが実測で裏付けられた。`winning_tile`（ron）や召し上げ牌（chi / pon / daiminkan）は`action.tile`から取得できる。

未確認（初回実測時点）: 槍槓（暗槓に対するchankan）等、直近の打牌以外がron対象になり得るケースでの`last_discard`の挙動は今回検証していない。kakan chankanについては、後段の「2026-08-14: `[AI-REVIEW]`対応の追加実測」節でソース確認と実機再現により解消済みである。ankan chankan（国士無双限定）は既定ルールでは到達しないため、引き続き未実機確認である。

#### representative選択に利用可能なphysical field

同一Observation内で`to_mjai()`が完全に一致する複数の`legal_actions()`が存在するかを広く走査した。

| ActionType | 重複の有無 | 実測例 |
| --- | --- | --- |
| DISCARD | あり | `{"pai":"6p",...}`に対し`tile=57`と`tile=59`の2候補 |
| CHI | あり | `{"consumed":["4s","5s"],"pai":"3s",...}`に対し`consume_tiles`が`[85,89]` / `[86,89]` / `[87,89]`の3候補 |
| PON | あり | `{"consumed":["8s","8s"],"pai":"8s",...}`に対し`consume_tiles`が`[100,101]` / `[100,103]` / `[101,103]`の3候補 |
| ANKAN / KAKAN / DAIMINKAN | 400 seed・延べ219回の合法提示で確認されず | — |

DISCARD / CHI / PONの重複候補は、`consume_tiles`（および`tile`）に含まれるRiichiEnv物理牌ID（整数）を使えば、入力順序に依存しない代表選択が可能である（例: 物理牌IDの昇順で先頭を選ぶ等）。物理牌IDそのものをPolicy契約へ持ち込まない前提は維持できる。具体的な採用規則の確定は本Issueのスコープ外とする。

ANKAN / KAKAN / DAIMINKANで重複候補が一度も出現しなかったことは、各鳴き牌種が最大4枚しか存在しないため「同一牌種内の組み合わせ選択」が構造的に発生しない（ankanは4枚全部、daiminkanは残り3枚全部、kakanは残り最大1枚を使用する）という組み合わせ論的な理由と整合する。ただしこれは全パターンを数学的に証明したものではなく、400 seedでの非出現という実測に基づく推測である。

今回の重複候補には赤牌が絡む牌種の事例は出現しなかった。赤牌を含む物理牌が候補に混在する場合、`to_mjai()`の`consumed`表現が赤牌の有無で変わるため、そもそも別candidateとして扱われ「同一identityの重複」にはならないと考えられるが、これは未確認の推論であり実測による確証ではない。

#### kakan元pon解決とmeld公開状態

kakan実行前後の`Observation.melds`（`list[list[Meld]]`、seat別）を比較した（seed=2, step=46, actor=0）。

```text
kakan前: melds[0][1] = Meld(meld_type=Pon,   tiles=[36,37,38],    called_tile=38, from_who=1, opened=True)
kakan後: melds[0][1] = Meld(meld_type=Kakan, tiles=[36,37,38,39], called_tile=38, from_who=1, opened=True)
```

同一player・同一list indexでPon meldがKakan meldへin-place更新され、`tiles`は元の3枚に追加牌1枚を加えた4枚になり、`called_tile`と`from_who`は元ponの値のまま保持された。`action.consume_tiles`（kakan前のpon側3枚）は更新後meldの先頭3 tilesと完全一致した。

これは`docs/internal-action-model.md`のKakanAction設計（`source_meld_id` / `source_meld_index`を使わず、`from_seat`と`called_tile`で元Ponをちょうど1件へ照合する）と整合する実測結果である。実測した1事例では「`meld_type == Pon`かつ`tiles`が`action.consume_tiles`の3枚と一致するmeld」が該当playerのmeld一覧中にちょうど1件存在し、0件・複数件のfail closedケースは今回発生しなかった（未確認）。

wall長は槓処理と嶺上ツモを含む同一`step()`内で45から44へ1減少し、既存実測（#11向け追加実測の「daiminkan / ankan / kakanと嶺上ツモ時のwall」節）と整合した。

#### `live_wall_tiles_remaining`に関する追加実測

`RiichiEnv`オブジェクトには`turn_count`、`rinshan_draw_count`、`wall`等の進行カウンタが公開されているが、`Observation`側の公開属性一覧にはこれらに相当するfieldが存在しないことを`dir()`で確認した。

つまり、`RiichiEnv`本体を直接保持するLocal game runnerは`env.wall`等から直接壁残数を算出できるが、`Observation`だけを受け取るRiichiEnv Adapter（特にRiichiLabのserialized observationを復元する経路）にはこの手段がない。既存文書が示した「`tsumo` eventを数える」方式が、Observationだけを前提にする場合の実質的に唯一の選択肢であることが、今回の公開属性調査で追加的に裏付けられた。

#### event重複防止に関する追加実測

`chi`、`dahai`、`pon`、`ankan`、`daiminkan`、`kakan`、`dora`、`reach`、`reach_accepted`、`start_game`、`start_kyoku`、`tsumo`の全event typeについてJSON keyを収集したが、一意なevent IDやシーケンス番号に相当するfieldは存在しなかった。

また、`Observation.events`と`Observation.new_events()`は同一Observation instanceに対して常に同じ内容・同じ長さを返し、対局全体を通じて増加し続けるcumulativeな履歴ではないことを確認した（既存実測どおり、instance単位のnon-consuming/delta挙動と整合する）。したがって`events`の長さを重複防止用カウンタとして使うこともできない。

Adapterが複数のObservationにまたがるevent適用の重複を避けるには、RiichiEnv側が提供する識別子に頼ることができない。「同一seatについて新しいObservationを受け取るたびに、その`new_events()`全体を1回だけ未適用分として扱う」という、既存実測済みのnon-consuming / delta契約そのものを運用規則として守る以外の手段が今回のevent key網羅調査でも見つからなかった。

#### 和了(ron / tsumo)の役・点数詳細に関する追加実測（本Issueのスコープ外だが記録）

RON / TSUMOの`to_mjai()`は`{"actor":...,"type":"hora"}`のみであり、役・符・点数移動・裏ドラ等は一切含まれない。それらの詳細は`env.mjai_log`側の`hora` event（`deltas`、`target`、`ura_markers`等を含む）にのみ存在し、seat別Observationからは取得できないことを実測した。対局終了後の`observations`は既存実測どおり空mapになるため、どのseatの`new_events()`からも`hora` / `end_kyoku` / `end_game`は届かなかった。

既存の設計判断14「`env.mjai_log`はPolicyから隔離する」と合わせると、和了の役・点数詳細は現状Policy / RiichiEnv Adapter層では取得できず、必要であればLocal game runner側の責務として別途整理する必要がある。本Issueのスコープ外の発見だが、後続Issueの判断材料として記録する。

なお、`Observation.find_action(action_id: int)`という追加APIも発見した。これは初期観測での`action_space_size`（82）に対応する固定長RL action-space index用のlookupであり、`select_action_from_mjai()`によるMJAI round-tripとは無関係と判断し、これ以上は深追いしていない。

#### 今回追加で未確認のまま残った事項

- 槍槓（kakan chankan）でのtarget解決は、本Issueの`[AI-REVIEW]`対応の追加実測（後述の「2026-08-14: `[AI-REVIEW]`対応の追加実測」節）でソース確認と実機再現の両方により解消した。ankan chankan（国士無双限定、既定ルールでは無効）は未実機確認のまま残る。
- 複数ron（多家和）が同時に競合する場合の採用順序と、各`RonAction`の`target`解決。
- ANKAN / KAKAN / DAIMINKANで重複candidateが構造的に発生しないという結論の数学的な証明（400 seedでの非出現という実測に基づく推測にとどまる）。
- 赤牌を含む牌種でPon / Chiの重複candidateが発生する場合の`to_mjai()`表現とrepresentative選択への影響。
- `Observation.find_action(action_id)`の入出力仕様全体。
- 今回の実測はLinux / CPython 3.11.15コンテナ環境で行った。後述の「2026-08-14: `[AI-REVIEW]`対応の追加実測」節でCPython 3.14.0rc2（python-build-standaloneビルド）による核心結果の再確認を追加したが、既存の基準環境（Windows公式installer由来のCPython 3.14.6）そのものでの再現性は未検証のまま残る。

### 2026-08-14: `[AI-REVIEW]`対応の追加実測（槍槓・live-wall counter・CPython 3.14）

[Issue #27](https://github.com/lisbun/lisjong/issues/27)への`[AI-REVIEW]`コメントで指摘された次の4点に対応し、追加実測した。

1. 通常discard以外がron対象になるケース（槍槓）でのtarget解決
2. `live_wall_tiles_remaining`の具体的なcounter algorithmの確定
3. 実装を左右する核心結果のCPython 3.14環境での再確認
4. 正本文書・Ruff・既存unit testの整理

#### 実測環境（CPython 3.14）

このコンテナには標準のaptリポジトリでCPython 3.14が用意されておらず、`uv python install 3.14`（[python-build-standalone](https://github.com/astral-sh/python-build-standalone)由来のLinux向けビルドを取得）で導入した。

| 項目 | 値 |
| --- | --- |
| Python | CPython 3.14.0rc2（`uv`経由のstandaloneビルド、Clang 20.1.4） |
| OS | Linux 6.18.5-fc-v20（glibc 2.39） |
| RiichiEnv | 0.4.8 |

既存の基準環境（Windows公式installer由来のCPython 3.14.6）とは、正確なpatchバージョン（3.14.0rc2はrelease candidateであり3.14.6ではない）とビルド来源（python-build-standalone対公式installer）の両方が異なる。したがって「CPython 3.14 GA、Windows公式installer」そのものの再現ではなく、production基準に近いメジャーバージョンでの追加確認と位置付ける。

#### 1. 槍槓（chankan）のtarget解決

[RiichiEnv v0.4.8ソース](https://github.com/smly/RiichiEnv/tree/v0.4.8)の`riichienv-core/src/state/mod.rs`（`step()`内の`ActionType::Kakan` / `ActionType::Ankan`分岐）を確認したところ、次の実装事実があった。

- Kakan処理時、actor以外の全playerについて`Conditions{chankan: true, ...}`で和了判定を行い、和了できるplayer（`chankan_ronners`）が1人以上いれば、`self.phase = Phase::WaitResponse`、`self.active_players = chankan_ronners`（kakan行為者自身は含まない）、`self.last_discard = Some((pid, tile))`とする。ソースコード中のコメントは`// Treat Kakan tile as discard for Ron targeting`であり、通常discardの場合と同じ`last_discard`機構をchankanにも転用する設計であることが明記されている。誰も和了できなければ`_resolve_kan()`を呼び、通常どおり嶺上ツモへ進む。
- Ankan処理時も同様の`chankan_ronners`判定があるが、`self.rule.allows_ron_on_ankan_for_kokushi_musou`が`true`の場合に限り、かつ国士無双／国士無双十三面待ち（yaku id 42 / 49）の場合だけ有効になる。`GameRule::default()`は`default_tenhou()`であり、`allows_ron_on_ankan_for_kokushi_musou: false`である。Python側`RiichiEnv(rule=None)`は`rule.unwrap_or_default()`でこの既定値を使うため、既定ルールではankan chankanは発生しない。

実際にkakan chankanをCPython 3.14.0rc2上で再現した（seed=677, step=78）。

```text
kan_actor: 3, kan_type: KAKAN, kan action to_mjai: {"actor":3,"consumed":["9s","9s","9s"],"pai":"9s","type":"kakan"}
obs_map keys after kan: [1]   # kan行為者(3)自身は含まれない
seat 1 observation:
  last_discard: 3             # kan行為者と一致
  legal_actions: RON(tile=107) / none
```

これは、通常のron解決に使う`Observation.last_discard`が槍槓でも同じ意味で使えることを、ソースの実装事実と実際の出力の両方で確認したものである。`env.step()`は、槍槓が成立する場合はkan行為者を含まない`WaitResponse`のobservation mapを返し、成立しない場合は既存実測どおりkan行為者だけの`WaitAct`へ直接進む。

未確認: ankan chankan（国士無双限定）は、既定ルールでは到達しないため実機確認していない。`rule=GameRule(allows_ron_on_ankan_for_kokushi_musou=True, ...)`を明示的に渡し、国士無双聴牌の局面を作る追加実測は今回実施していない。

#### 2. `live_wall_tiles_remaining`のcounter algorithm

同ソースの`riichienv-core/src/state/wall.rs`と`state/mod.rs`を確認したところ、次の実装事実があった。

- 各kyoku開始時、`wall.tiles`は136枚から4人×13枚（52枚）の配牌分を除いた84枚になる（`WallState::shuffle()`）。
- 通常ツモ（`_deal_next()`）は`wall.tiles.pop()`、嶺上ツモ（kan後の`_resolve_kan`内の抽選）は`wall.tiles.remove(0)`で、いずれも`wall.tiles`の長さを1減らす。kan宣言自体は独立した牌消費を伴わない（嶺上ツモの分だけ減る）。
- 通常ツモ・嶺上ツモのいずれも、MJAI `"tsumo"` eventとして記録される（別種別のeventにはならない）。

Python公開面の`len(env.wall)`は、この`wall.tiles`の長さそのものである。したがって、次の関係が成り立つ。

```text
live_wall_tiles_remaining
    = 84 − (そのkyokuのstart_kyoku以降に発生したtsumo event数。dealerの最初の1枚も含む)
```

`start_kyoku`直後の最初のObservationには、dealerの最初のツモが`{"actor":<oya>,"pai":...,"type":"tsumo"}`として`new_events()`に明示的に含まれることを確認した（隠れた・観測不可能なeventではない）。したがって`Observation`だけを持つAdapterでも、`new_events()`中の`"tsumo"` event数を数えるだけでこの式を再現できる。

この式を、真値`len(env.wall)`と比較して次の規模で検証した（CPython 3.14.0rc2、`4p-red-half`モード、東南全kyoku・honba・複数kanを含む）。

| 検証方法 | 対象 | 比較件数 | 不一致 |
| --- | --- | --- | --- |
| `env.mjai_log`から`start_kyoku`以降の`tsumo`件数を数え、式の予測値と`len(env.wall)`を比較 | seed 1〜14（各12 kyoku前後、honba込み） | 10,579ステップ | 0件 |
| 1つのseatに固定し、そのseat自身の`new_events()`だけを累積してカウント（実際のAdapterと同じ視点） | seed 1〜19、seat 2固定 | 3,569ステップ | 0件 |

1 kyoku内で2回kanが発生したケース（複数kan）でも式は成立した（今回の検証範囲内での最大同時kan数は2）。また、`(bakaze, kyoku, honba)`で重複排除した27 kyokuすべてで、`start_kyoku`直後の`len(env.wall)`は常に83（`84 - 1`、dealerの最初のツモ分）だった。honba・東南を通じて変動しなかった。

これにより、「Observationに直接手段がない」という否定的確認だけでなく、Observationの`new_events()`だけから再現できる具体的な更新規則が確定した。

未確認: 3人麻雀（本Issueのスコープ外）、および海底・河底に到達する直前の境界（`drawable_count == 0`時の`_deal_next()`の`ryukyoku`分岐）付近の1テンポずれの有無は個別確認していない。

#### 3. CPython 3.14での核心結果の再確認

上記1・2の槍槓実測とcounter algorithm検証はすべてCPython 3.14.0rc2上で実施した。加えて、未確認7 variantのround-trip代表例、kakan meld のin-place更新、DISCARD / CHI / PONの重複candidate例を同一seedで再実行し、CPython 3.11.15での結果と完全に一致する出力を得た（RiichiEnvのコア実装はRustであり、Python側の挙動差は今回の範囲では確認されなかった）。OS差（Windows対Linux）そのものを検証する目的では実施していない。

#### 4. 正本文書・品質確認

本節を含む今回の追加実測を本文書へ反映した。Ruffと既存unit testの結果は本書末尾の変更履歴および対応するPull Requestの記録を参照する。

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
- CPUのみでの利用条件、Windows / WSL2 / Linuxの差、native runtime要件。Issue #27向け追加実測（2026-08-14）で、Linuxコンテナ環境（CPython 3.11.15）へのインストールとimport、1局分を大きく超える多数対局の実行には成功したが、基準環境との挙動差の網羅的な比較はしていない。
- `RiichiEnv(...)`の既定rule、game modeごとの初期化引数、不正な初期化引数に対する例外。
- 公式文書上の`act(obs: Observation) -> Action`以外にAgent登録APIが存在するか、環境loopをlisjong側で管理することが正式な利用方法か。
- constructor seedの再現性が、今回と異なるrule、game mode、Action選択方針、RiichiEnvバージョンでも保たれるか。
- `reset(seed=...)`で期待した再現性を確認できなかった挙動が、他の環境やRiichiEnvバージョンでも同じか。
- 今回と異なるrule、score、seed、Action選択でのgame-end conditions、延長範囲、終了時の観測。
- `scores()` / `ranks()`を取得できるすべての時点と、点数移動がある終了時の値。
- Python公開面より下の内部可変状態まで、4席の`Observation`が共有されず独立しているか。
- `Observation`、`Action` の比較やhashに依存してよいか。シリアライズの安定性とバージョン互換性。
- `events` / `new_events()`はseatごとのObservation更新間のdeltaと整合したが、全version・全局面、長期間、終局を跨ぐ厳密な範囲と蓄積挙動。
- `apply_event(...)`、`get_observation(player_id)`、`observe_event(...)`の実際の入出力と例外。
- 今回確認していないfieldや局面を含め、各プレイヤーの`Observation`にPolicyが見てはいけない非公開情報が含まれないか。
- player ID 99以外の不正ID、合法だが要求先と異なるplayerのAction、別局面のAction、欠落・余分な応答等に対する`step()`の挙動。
- `done()`後に空でないActionを渡した場合や、`step({})`を繰り返した場合の挙動。
- 今回のWindows環境ではimportとstepに成功したが、追加のDLLまたはruntimeが既存環境に依存していないか、別環境でも同じ条件で動作するか。
- 通常版CPython 3.14.6で、複数対局の反復や長時間実行が安定するか。
- 複数ron（多家和）等を含むclaim競合と、pon / chi以外の優先順位。単独ronおよび槍槓（kakan chankan）の`target`解決（`Observation.last_discard`）はIssue #27向け追加実測（2026-08-14）で確認したが、複数ron競合時の採用順序と各`RonAction`への解決は未確認。
- riichi、daiminkan、ankan、kakan、ron、tsumo agari、kyuushu kyuuhaiは、Issue #27向け追加実測（2026-08-14）でそれぞれ1事例ずつAction/MJAI round-tripを確認した。全局面・全出現パターンを網羅した完全実測ではない。
- 不正または曖昧なMJAI入力を`select_action_from_mjai()`へ渡した場合の挙動。
- 実際のRiichiLab WebSocket requestにおける`possible_actions`と生成Actionの照合方法。生JSON dictの全field完全一致は未実測。
- Python公開属性と`to_dict()`の差は一部実測済みだが、`Observation.tsumogiri_flags`の詳細な更新挙動、furiten、ippatsu、`waits` / `is_tenpai`のPolicy入力への採否を含むseat別情報の最小範囲。
- live-wall counterの更新式は、Issue #27向け追加実測（2026-08-14）で`84 - tsumo event数`として確定し、単一seatの`new_events()`累積だけで再現できることを検証した（4人麻雀・標準ルール範囲）。3人麻雀、および実際のRiichiLabオンライン経路で同値な状態を再構成できるかは未確認のまま残る。
- 鳴かれたdiscard、リーチ段階、wall関連状態を含むmaterialized stateの具体schema、正確な更新規則、所有component。
- RiichiLab実WebSocket経路で、seat-visible eventから同値なstateを再構成できるか。
- RiichiEnvの全機能と全Action種の完全実測。
- `mjai_log`の永続化形式とReplay API。完全ログとしての取得方法と情報境界は確認済み。
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
8. Local game runnerは`reset()` / `step()`の戻り値を、その時点で`Action`選択を要求されているplayerから`Observation`へのmapとして扱い、「現在手番」や単一playerを前提にしない。各seatを独立した変換・Policy判断単位としてRiichiEnv AdapterとPolicy contractへ渡す。
9. 再現性が必要な実験では、現時点ではconstructorの`RiichiEnv(seed=...)`を使用する。`reset(seed=...)`を再現性の根拠にはしない。
10. RiichiEnvが不正player IDを必ず例外化することを前提にせず、Adapter側でAction要求playerと入力player IDの整合性を検証する。
11. 複数playerへ同時に応答する場合も、各playerの`Observation`と合法手を混同せず、seatごとに変換・検証する。
12. ローカルとRiichiLabでは、RiichiEnv AdapterとPolicy contractを共通利用できる。RiichiEnv固有型とlisjong内部型の変換はRiichiEnv Adapterへ閉じ込め、Policyへ直接持ち込まない。
13. RiichiEnv利用時の対局終了は`round_wind`や`kyoku_idx`から推測せず、`env.done()`を正本とする。
14. `env.mjai_log`は非公開情報を含み得る完全対局ログとしてPolicyから隔離し、Replay、調査、監査等のPolicy外用途に限定する。
15. eventをPolicy入力に使う場合は、`env.mjai_log`ではなくseat別`Observation`の情報境界を維持する。WebSocket、`request_id`、`possible_actions`照合、timeout、`action_ack`、session終了処理はRiichiLab Clientへ閉じ込める。初期スコープでは切断後の途中再接続・復旧を試みず、安全に終了する。

これらは[Architecture](architecture.md)のPolicy、Local game runner、RiichiEnv Adapter、RiichiLab Clientの責務分離を具体化する判断である。Policyの詳細契約は[Policy契約](policy-contract.md)を正本とする。

### Issue #11へ引き継いだ判断候補と根拠

次は追加実測からIssue #11へ引き継いだ判断候補である。本書だけでPolicyInput schema、InternalAction identity、materialized stateの責務配置を確定するものではなく、現在の採否と意味契約は各正本文書を参照する。

1. `Observation.to_dict()`をObservationの完全表現とみなさず、Policy入力fieldを許可リストで個別に取得・変換する。
2. RiichiEnvの物理牌IDをPolicyへ直接持ち込まず、物理牌差がゲーム上の意味差を生む場合だけ意味fieldへ正規化する。Discardでは手出し / ツモ切り差を保持する。
3. rawなseat-visible event deltaをPolicyへ直接渡すのではなく、境界側で現在状態へ正規化して不変snapshotを渡す構成を検討する。
4. 鳴かれたdiscardの`called_by`、リーチの宣言・受理段階、live-wall関連状態は、単一Observationだけでなくseat-visible event deltaを使って構成する候補とする。
5. 環境非依存なlive-wall残数概念を検討するが、`len(env.wall)`や未検証のevent計数式をそのまま契約の意味にしない。

PolicyInputの許可fieldとaction identityはIssue #11の各正本文書で確定済みである。
具体的なPython型とpackage / module構成はIssue #20で
`lisjong.policy_contract`として実装済みである。counter更新式等の実装詳細は後続の
Adapter実装Issueで扱う。責務と依存方向は[Architecture](architecture.md)を正本とする。

### Issue #27で追加実測し、Issue #23へ引き継ぐ判断候補

次はIssue #27の追加実測から、Issue #23「RiichiEnv Adapterを実装する」へ引き継ぐ判断候補である。本書だけでAdapterの実装方式を確定するものではなく、採否は#23側で判断する。

1. Chi / Pon / Daiminkan / RonのAction識別だけからは`target`（from_seat）を得られないため、同一decision時点の`Observation.last_discard`から解決する。これは通常discardだけでなく、kakan chankanでも同じ機構で成立することを、RiichiEnv v0.4.8ソースの実装事実（`last_discard`をchankan targeting用に転用するコメント付きの実装）と実機再現の両方で確認した。ankan chankan（国士無双限定）は既定ルールでは到達しないため、この結論の対象外とする。
2. Kakanの元Pon解決は、`meld_type == Pon`かつ`tiles`が`action.consume_tiles`と一致するmeldを対象playerのmeld一覧から1件に絞り込む方式が実測と整合する。0件・複数件はfail closedとする。
3. Ankan / Kakan / Daiminkanでは、同一identityの複数RiichiEnv Action候補が生じる組み合わせ論的余地がない可能性が高い（400 seedで非出現）。一方Discard / Chi / Ponでは、物理牌IDに基づく入力順序非依存の代表選択が必要である。
4. `live_wall_tiles_remaining`は、`Observation`だけを持つAdapter経路（RiichiLabのserialized observation復元を含む）では`tsumo` event計数以外の直接手段がない。`turn_count`等のcounterは`RiichiEnv`本体にのみ公開され、`Observation`側には存在しない。具体的な更新規則は`84 - (そのkyokuのstart_kyoku以降のtsumo event数、dealerの最初の1枚を含む)`として確定し、単一seatの`new_events()`累積だけで再現できることを4人麻雀・標準ルールの範囲で検証した（10,579 + 3,569ステップ、不一致0件）。
5. event適用の重複防止は、RiichiEnv側の識別子（event ID等）に頼れないため、seatごとの新規Observation受信ごとに`new_events()`全体を1回だけ未適用分として扱う運用規則に依拠する。
6. Ron / Tsumoの役・点数・裏ドラ等の詳細はseat別Observationから得られない（`env.mjai_log`側の`hora` eventにのみ存在する）ため、必要であればLocal game runner側の責務として別途設計する。

### 実測から設計・実装へ引き継ぐ判断

| 判断対象 | 今回までの実測 | 確定済み設計と残る確認 |
| --- | --- | --- |
| Policy 入力の最小スキーマ | 自他家の情報境界、seat別eventのmaskとdelta、Python公開属性と`to_dict()`の差、複数seatのPython object・読み取り操作の独立性を一部確認。`mjai_log`は対象外 | 許可field、不変snapshot、canonicalizationはIssue #11で確定済み。残る確認はAdapterでの生成・同期、counter algorithm、将来拡張fieldの実測とtest |
| 内部行動の識別方法 | Action属性、MJAI変換、打牌・pon・chi・noneのround-tripに加え、同種の物理牌Actionが同じMJAI打牌へ潰れる例と手出し / ツモ切り差を確認。riichi / daiminkan / ankan / kakan / ron / tsumo / kyuushu_kyuuhaiの各1事例のround-tripと、Chi / Pon / Daiminkan / Ronの`target`が`Observation.last_discard`で解決できることをIssue #27で追加実測。意味fieldとidentity規則はIssue #11で確定済み | 全出現パターンの完全実測、複数ron競合、不正・曖昧なMJAI入力、Adapter変換実装とtest |
| 乱数・再現性の境界 | constructor seedでevent列まで再現。`reset(seed=...)`では期待した再現性を確認できず | rule、game mode、バージョンを変えた場合の適用範囲 |
| エラー変換方針 | 不正Action型、player ID 99、`done()`後の空actionを確認 | 要求先不一致、欠落・余分な応答、別局面Action等 |
| 局・対局のライフサイクル | single / east / halfを完走し、延長、`end_game`、`done()`、最終空map、scores、ranksを確認 | 他rule・score条件での終了挙動 |
| materialized state | seat別event delta、鳴かれたdiscard、リーチ段階、通常ツモ・槓時のwall変化を一部確認。Issue #27でkakan元pon解決（meldのin-place更新）、槍槓のtarget解決（`last_discard`）、`live_wall_tiles_remaining`の具体的counter algorithm（`84 - tsumo event数`、ソース確認＋10,579+3,569ステップ実測で不一致0件）、event重複防止（IDなし）を追加実測 | schema、更新規則、所有component、長期間・終局・他seat・RiichiLab経路での同値性、3人麻雀への一般化 |
| RiichiLab との共通 Adapter 範囲 | 公式serialized Observation / Action応答境界と、ローカルの一部Action / MJAI round-tripを確認 | 実WebSocket requestの`possible_actions`照合、未出現Actionとの対応、seat-visible eventからの同値なstate再構成 |
| runtime dependencyへの追加 | 通常版CPython 3.14でインストール、import、1局完走、依存version、明示取得したwheelのhashを確認 | 別環境・長時間実行の安定性 |

RiichiEnv を `lisjong` の通常依存へ追加する判断は、対象環境でインストールと最小再現が成功し、必要性と依存範囲を確認した後に別の変更として行う。

## Issue #3 完了に向けた確認項目

- [x] 公式文書、公式リポジトリ、PyPI メタデータの確認先を固定した。
- [x] 公式情報、実測、推測、設計判断を分離する形式を定義した。
- [x] 対象パッケージと調査予定環境を記録した。
- [x] 最小再現コード案を記録した。
- [x] Windows / 通常版 CPython 3.14.6 で正確な環境情報を採取した。
- [x] `riichienv==0.4.8` のインストールと import を確認した。
- [x] `reset()`、合法手選択、最初の`step()`を実測した。
- [x] 既定の1局終了、`done()`、最終空map、`scores()`、`ranks()`を実測した。
- [x] Issue #3で必要なseed指定方法と再現性の扱いを確認した（constructor seedを採用し、`reset(seed=...)`を再現性の根拠にしない）。
- [x] Issue #3で必要なプレイヤー別情報境界を確認した（`mjai_log`は完全ログ、seat別`Observation`はmask済み。全fieldの網羅監査は後続候補）。
- [x] Issue #3で必要な例外と依存条件を確認した（追加異常系、別OS runtime、stress testは後続候補）。
- [x] single / east / halfの終了を実測し、`env.done()`を終了判定の正本とした。
- [x] RiichiEnvの完全ログとseat別eventの情報境界を確認した。
- [x] RiichiLabとローカルRiichiEnvで共通化できるAgent境界と、Client固有責務を整理した。
- [x] 初回実測結果を設計判断へ反映した。
- [x] 一時調査用`tmp_riichienv_probe.py`を正式な調査コードとして残さない方針を記録した。
- [x] 一時probeはcommit対象にせず、恒久的な調査コードを残す場合の`experiments/riichienv/`分離方針を記録した。

## 変更履歴

| 日付 | 対象 | 内容 |
| --- | --- | --- |
| 2026-08-13 | RiichiEnv 0.4.8 | 公式情報と実測予定を初回記録。対象環境での実測は未実施 |
| 2026-08-13 | RiichiEnv 0.4.8 | Windows / CPython 3.14.6でインストール、主要型、初期観測、打牌、情報境界、ポン応答を初回実測 |
| 2026-08-13 | RiichiEnv 0.4.8 | 1局完走、seed、異常系、複数player同時要求、pon / chi競合、Observation独立性、Action / MJAI round-tripを追加実測 |
| 2026-08-13 | RiichiEnv 0.4.8 | OS、PowerShell、PyYAML、IPythonの正確な環境情報と、CPython 3.14 Windows x86-64 wheelのSHA-256が公式メタデータと一致することを追加実測 |
| 2026-08-13 | RiichiEnv 0.4.8 | single / east / halfの終了、完全`mjai_log`とseat別eventの境界、RiichiLabとの共通Agent範囲を追加記録 |
| 2026-08-14 | RiichiEnv 0.4.8 | Observation公開属性と`to_dict()`の差、seat別event delta、Discard物理牌identity、通常・槓時のwall、鳴かれたdiscard、リーチ状態遷移をIssue #11向けに追加実測 |
| 2026-08-14 | RiichiEnv 0.4.8（Linux / CPython 3.11.15、既存基準環境とは別環境） | Issue #27向けに、未確認7 Action variant（riichi / daiminkan / ankan / kakan / ron / tsumo / kyuushu_kyuuhai）のround-trip、Chi / Pon / Daiminkan / Ronの`target`解決（`Observation.last_discard`）、representative選択に使える物理牌ID、kakan元pon解決とmeld公開状態、`live_wall_tiles_remaining`の情報境界、event重複防止の欠如を追加実測 |
| 2026-08-14 | RiichiEnv 0.4.8（Linux / CPython 3.14.0rc2、python-build-standaloneビルド） | Issue #27の`[AI-REVIEW]`対応として、v0.4.8ソース確認（`riichienv-core/src/state/mod.rs`、`wall.rs`）と実機再現により、kakan chankanのtarget解決（`last_discard`）、`live_wall_tiles_remaining`の具体的counter algorithm（`84 - tsumo event数`、不一致0件で検証）を確定し、7 variant round-trip・kakan meld更新・重複candidate例をCPython 3.14系でも再確認した |
