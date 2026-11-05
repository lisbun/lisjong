# L2a runtime互換確認tool（#369）

## 状態

道具（`tools/l2a_runtime_compat.py`）と合成fixtureでの検証だけを実装した。
**実artifact・実sourceでの互換実測は未実施であり、runtime互換はPASSしていない。**
toolの`COMPLETE` / `MATCH`は道具の判定であり、互換PASSの記録は後続の実行Issueが行う
（#368の正式実行開始gate 1）。実測の前に、入力subsetと時間・メモリ上限を別計画へ固定する。

## 目的と比較対象

凍結済みL2a候補（#318）を新しいruntimeで使う前に、同じ固定入力について
旧runtimeと新runtimeのL2a判断が一致するかを、対局なしで照合する。

| 側 | runtime |
| --- | --- |
| 旧 | lisjong `999553b6faf4c515c3a9f862429896308188e504`（Rust backend）＋同revisionでbuildしたnative wheel |
| 新 | 後続計画で固定するexact revision＋同revisionのnative wheel（Rust必須） |

旧側もRustであり、Python対Rustの比較ではない。固定する候補とsourceはtoolの`FROZEN`にある。

| 項目 | 値 |
| --- | --- |
| M1 | `c976ed4a5cb667c406eb57f5cc79c128b1a221fb9b478f0dd017fcf48bde6e03` |
| R1 | `d565713f51a2b9f9cbdd2146bf01cb4bfb13ff859b639de962c74562eaf01efc` |
| 分類 | `lisjong-l2a-decision-classification-v1` |
| source（arena#506） | `ba02b59163e17e128b998c80f59dc386c49796fc3dd35f67b88dc2c756421aaa` |

public-call R1（#326）へは差し替えない。weights・source・reportはrepositoryへcommitしない。

## 設計

### toolと検証対象runtimeの分離

- toolは1 fileの外部scriptで、標準libraryだけでimportできる（torch・lisjongをimportしない）。
  lisjongのcodeを同梱せず、実行したPythonがimportする`lisjong`（検証対象runtime）だけを呼ぶ。
  旧環境では新側のencoder・servingを使わない
- 旧新で公開名が違うのはscorer loader（旧`_load_scorer` / 新`load_l2a_scorer`）だけである。
  captureは両revisionに共通の`load_l2a_policy_factory`と`L2aPolicy.decide`（serving経路）を使い、
  loader名は「既知の2形のどちらか1つ」であることの確認と記録にだけ使う
- tool自身はfileのSHA-256で識別する。runtime revisionとは別に記録し、compareは
  旧新のcaptureが同じtool fileで作られたことを要求する
- lisjong-arenaには依存しない

### 固定入力manifest

`make-manifest`がsourceのmanifest（game summary）だけからwhole gameの入力集合を作る。
payloadは開かない。manifestは次をbindし、内容のSHA-256（`identity`）を持つ。

- source identity・schema・purpose
- M1 / R1 artifact identityと分類identity
- gameごとのsplit・game ordinal・seed・decision数・payloadのbyte長とSHA-256
- 取り出すdecision ordinal（昇順）とrecord総数

TRAIN / SELECT以外のsplit（OFFLINE-EVAL・未知split）、存在しないgame、seed重複、
順序・件数の不整合は拒否する。captureはpayloadを開く前にsourceのgame summaryと全gameを照合し、
manifestにないgame（OFFLINE-EVALを含む）のpayloadを開かない。
manifestは実測前に作り、そのidentityを実行計画へ記録する。結果を見てから入力を変えない。

### 1 recordに記録する値

| field | 内容 |
| --- | --- |
| `key` | split・game ordinal・seed・decision ordinal |
| `input_sha256` | source payloadのその行のbytes（改行なし）のSHA-256。toolが直接読む |
| `kind` | L2a分類（forced / win / riichi / discard / call / pass_response） |
| `single_candidate` | DISCARD / CALLで槓・九種九牌を除いた候補が1つの行 |
| `legal_actions` | 合法actionのsemantic identity（canonical JSON）とその順序 |
| `action` | Policyが返した最終action |
| `scorer` | scorerを呼ばない分岐（guard・単一候補）は`null`（出力なし）。呼んだ分岐は下表 |

| `scorer`のfield | 内容 |
| --- | --- |
| `head` / `encoding` | discard（M1）またはcall（R1）と、encodingのname・identity・次元 |
| `candidates` | scorer候補のsemantic identityと順序 |
| `shared_feature` | #184 shared featureのfloat32 little-endian bytesのSHA-256と次元 |
| `candidate_vectors` | 候補ごとのencoding vectorのfloat32 little-endian bytesのSHA-256と次元 |
| `scores` | scorer出力（文書でlogitsと呼ぶ値）を`float.hex()`で正確に記録 |

captureは各scorer行で同じ入力を再scoreし、process内で値が一致しなければSTOPする。
最大scoreの最初の候補が最終actionであること、候補と最終actionが合法action objectであること、
特徴・vectorの次元と有限性、score数を検査する。

### runtime / artifactのprovenance

| 記録 | 内容 |
| --- | --- |
| lisjong | importした`lisjong`のrevision。packageの`__init__.py`をそのcheckoutが追跡しているclean git checkout（editable install。package内の未追跡file・未commit変更でSTOP）か、VCS install（`direct_url.json`のcommit id。package内の全fileがRECORDにあり、hashが一致すること）だけを受け付ける。ignoreされた`.venv`等を親repositoryのcheckoutとみなさない |
| native | `--native-wheel`で渡したwheelのSHA-256、installed moduleのSHA-256（wheel内の同名memberと一致しなければSTOP）、`SOURCE_REVISION`（lisjong revisionと一致しなければSTOP）、`API_VERSION` |
| shanten backend | `rust`以外はSTOP（旧revisionでは`LISJONG_SHANTEN_BACKEND=rust`が必要） |
| torch | version、`--torch-threads`で固定したthread数、interop thread数、deterministic algorithms（有効化する） |
| Python / platform | version、platform文字列 |
| artifact | M1 / R1のidentity（manifest digest）、artifact directory内の各fileのSHA-256、weights block、encoding block（分類identityを含む）、model、dataset / source identity |

### report

- capture・compare reportはcanonical JSONで、内容のSHA-256を`identity`に持つ。既存fileを上書きしない
  （排他作成。source・artifactのdirectory内にも書かない）
- `decision_digest`はrecord列のSHA-256で、照合はrecordとfieldごとに行う
- 時間・RSS・path・開始時刻は`metadata`に置き、照合・digestに使わない
- captureが失敗した場合も`status: STOP`と理由のreportを残す（recordは書かない）

compareは両captureをstrict readし（canonical JSON・identity・decision digest・summary・
manifestとの対応・record構造・scoreの有限性）、結果を次の3つに分ける。

| status | 条件 |
| --- | --- |
| `INVALID` | captureが読めない・STOP、入力集合や順序がmanifestと違う、artifact provenanceが違う、tool fileが違う、revisionが期待値と違う、`SOURCE_REVISION`不一致、backendがrust以外、torch thread / determinism設定が違う、旧新revisionが同じ |
| `MISMATCH` | 上記を満たし、1 recordでも記録値が違う。recordごとに不一致field（`kind`、`action`、`scorer.candidates`、`scorer.shared_feature`、`scorer.candidate_vectors`、`scorer.scores`等）と旧新の値、field別件数、scoreの最大絶対差を出す |
| `MATCH` | 全recordの全fieldが一致 |

scoreの数値差は自動許容しない（差があれば`MISMATCH`）。Python・torch・nativeのversion差は
`runtime_differences`に記録するが、それだけでは不一致にしない。
`frozen_candidate`はmanifestが上表の候補・sourceと一致するかを示す。

不一致時のfallback、artifact manifestの書換え、再学習、OFFLINE-EVAL ledgerの変更は実装しない。

## 必要なもの

- 各側のPython 3.14 venv（WSL / Linux。RSS計測に`resource`を使う）、`torch==2.13.0`
- 各側のlisjong：exact revisionのclean checkoutをeditable install、またはVCS install
- 各側のnative wheel（同じrevisionでbuildし、`SOURCE_REVISION`が一致するもの）とそのfile
  - 旧側：#507で使った`999553b`の wheel（SHA-256 `9aa0ca4e…`、`SOURCE_REVISION` `999553b`）。
    学習・OFFLINE-EVALで使った`ee02748f…`（`SOURCE_REVISION` `174bf48`）は`SOURCE_REVISION`が
    lisjong revisionと一致しないため、このtoolではSTOPになる
- source（arena#506、#318で使用）とM1 / R1 artifact directory（repository外）
- tool file 1つ（両環境で同じSHA-256のもの。commit済みrevisionから取る）

## 手順（2環境）

実測は後続の実行計画（入力subset・時間・メモリ上限を固定し、承認を受けたもの）に従う。
以下はcommandの形である。

```bash
TOOL=/path/to/lisjong/tools/l2a_runtime_compat.py   # 両側で同じfile

# 1. 固定入力manifest（payloadは開かない）。identityを実行計画へ記録する
NEW/bin/python "$TOOL" make-manifest --source SOURCE --game 0 --game 600 \
  --output manifest.json

# 2. 旧側capture（999553bのvenv）
LISJONG_SHANTEN_BACKEND=rust OLD/bin/python "$TOOL" capture \
  --manifest manifest.json --source SOURCE --m1 M1_DIR --r1 R1_DIR \
  --native-wheel OLD_WHEEL --torch-threads 1 \
  --max-wall-seconds LIMIT --max-rss-mib LIMIT --output old.json

# 3. 新側capture（新revisionのvenv）。引数は旧側と同じ
NEW/bin/python "$TOOL" capture \
  --manifest manifest.json --source SOURCE --m1 M1_DIR --r1 R1_DIR \
  --native-wheel NEW_WHEEL --torch-threads 1 \
  --max-wall-seconds LIMIT --max-rss-mib LIMIT --output new.json

# 4. 照合（どちらのPythonでもよい。torch・lisjongは不要）
python "$TOOL" compare --manifest manifest.json --old old.json --new new.json \
  --new-revision NEW_REVISION --output compare.json
```

- `--game`はsourceのgame ordinal（TRAIN / SELECTだけ）。`--old-revision`の既定値は`999553b…`
- tool fileを旧環境のcheckoutの中に置かない（そのcheckoutが未commit変更でSTOPする）
- 終了code：captureは`COMPLETE`で0、`STOP`で2。compareは`MATCH`で0、それ以外は2
- 新側revisionの`--new-revision`はcaptureの記録と一致しなければ`INVALID`になる

## 合成fixtureでの確認

`tests/test_l2a_runtime_compat.py`（focused test）が次を合成fixtureで固定する。

- 同じruntime挙動の2 captureが`MATCH`になること。候補順・特徴bytes・vector・score（1 ulp）・
  action・分類・合法action・入力行の改変が該当recordとfieldで`MISMATCH`になること
- artifact / runtime / tool provenanceの改変、record欠落・順序入替え、非有限score、
  未sealの改変、STOP captureが`INVALID`になること
- FORCED優先（Passだけ・Ronだけの行）、WIN、RIICHI、PASS_RESPONSE、単一候補、DISCARD、
  CALLのcapture。summaryの件数は合成sourceに実際にある行から数える
- 旧形（`_load_scorer`）・新形（`load_l2a_scorer`）のどちらでも、渡したruntimeの
  factoryとscorerを実際に呼ぶこと。どちらでもない・両方あるruntimeは拒否すること
- 次元違い・非有限の特徴、manifestとsource / artifactの不一致、wall / RSS上限、
  OFFLINE-EVAL・未知split、既存出力の上書きの拒否。OFFLINE-EVALのpayloadを開かないこと
- toolのimportがtorch・lisjongをimportしないこと

2026-11-06に、toolの動作確認として合成source・合成M1 / R1 artifactで、旧環境
（lisjong `999553b`、wheel `9aa0ca4e…`、loader `_load_scorer`）と
lisjong `1d14c09`（loader `load_l2a_scorer`）のcaptureを各20 recordで取り、`MATCH`を確認した。
合成データでの道具の確認であり、凍結M1 / R1・実sourceでの互換確認ではない。
