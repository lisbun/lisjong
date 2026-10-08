# HandBelief精度評価用の手牌正解データ契約

lisbun/lisjong#256（親: #255 手順1）。`HandBelief`の各テーブルの推定精度を測るため、
観測者のすべての打牌判断について、他家3席の手牌の正解を学習専用データとして扱う契約である。
実装は`lisjong.learning.hand_belief_source`。

## 位置付け

#237 S1（`riichi_deal_in_source`、`docs/riichi-deal-in-label.md`）と同じ方式をとる。
Arenaが観測事実（判断時点の`PolicyInput`と他家の手牌）を記録し、lisjongが読み込んで
ラベルを計算する。S1の記録は単独リーチ者の手牌だけを、S1の`SCOPE`の局面だけで持つため、
別の契約にした。lisjongはArenaのmoduleをimportしない。

## ファイル

| ファイル | schema | 内容 | 読む関数 |
|---|---|---|---|
| `manifest.json` | `lisjong-hand-belief-source-manifest-v1` | producer revision、split別seed、各ファイルのbytes / sha256 / 行数 | `read_manifest()` |
| `decisions.jsonl` | `lisjong-hand-belief-decision-record-v1` | player-safe。`DecisionKey`、観測者の判断時点の`PolicyInput`、合法手、選んだ行動 | `read_decisions()`（推論入力の経路） |
| `hand_facts.jsonl` | `lisjong-hand-belief-hand-fact-record-v1` | 学習専用。他家3席それぞれの`seat`・`sequence`・concealed tiles（赤5を区別）・副露 | `read_labelled_source()`（学習専用） |

`SCOPE`は`all-observer-discard-decisions.v1`で、合法手に打牌を1つ以上含む観測者の判断すべてを
対象にする。リーチの有無や選んだ行動の種類では絞らない。

## 検査（すべてfail closed）

- 未知のschema / scope、manifestのdigest・行数の不一致、正準JSONでない行、splitに属さないseed
- 判断記録と手牌記録の`DecisionKey`の欠落・重複・不一致
- 手牌記録が観測者以外の3席を席順に持っていない
- 時点違反: 他家の手牌の`sequence`が判断の`sequence`と違う。手牌は判断の行動を適用する前に
  取ったsnapshot（同じ`sequence`）に限る。観測者の行動は他家の手牌を変えない。前の時点の手牌は
  途中のツモ・打牌で変わっている可能性があり、副露・牌保存則だけでは現在の手牌と保証できない
- 他家の副露が`PolicyInput`の公開副露と一致しない
- 手牌が13枚相当（`len(concealed) + 3 * len(melds) == 13`）でない
- 他家3席のconcealed tilesの合計が、`PolicyInput`から見て未確定の牌
  （`derive_remaining_tile_inventory()`）を牌種別・赤5別・通常5（5の総数−赤5）別に超える

## ラベル

各他家について、手牌から`exact_hand_belief_with_waits()`で正解の`HandBelief`を求める。
`expected_count`・`red_five_probability`はconcealed tilesの実枚数（副露は含めない）、
`wait_probability`と形別7テーブルは構造的な待ち（フリテン・役の有無は含まない）で、
値はすべて0か`SCALE`である。待ち判定は再実装しない。非聴牌の手牌は待ちのテーブルが
すべて0になる。

## 情報境界

`read_decisions()`は`hand_facts.jsonl`を開かない（testで固定）。推定器の推論入力は
判断記録の`PolicyInput`だけから作る。

## Arena側の記録経路（調査結果、2026-10-07）

Arenaの`stage_a0_tenpai_feasibility`で使っている観測の仕組みは流用できる。

- `LocalGameRunner(decision_point_observer=...)`へ`DecisionPointHiddenStateRecorder`を渡すと、
  各stepで`env.step()`の直前（その判断の`PolicyInput`と同じ状態）に、4席のconcealed tiles・
  副露・リーチ宣言を記録する。Policy・`PolicyInput`・`GameTrace`へは渡らない
- 牌は`tile_from_physical_id()`で変換され、赤5を区別する
- `require_same_decision_state()`が、判断の`PolicyInput`とsnapshotが同じ状態であること
  （自分の手牌・全席の副露・リーチ状態）を検査する

流用しないもの: A0のsidecar形式（`cells.jsonl`）はflat-BCの行とTenpai targetに合わせた
相対席の形式なので、この契約の`hand_facts.jsonl`へは使わない。Arena側では、上の
snapshotから観測者以外の3席を取り出し、`sequence`を判断と同じ値にしてこの契約の形で
書けばよい。記録経路の実装はlisjong-arenaの別Issueで扱う。

## 精度の測定（lisbun/lisjong#257、設定化は#274）

このsourceを使う測定は`lisjong.learning.hand_belief_accuracy`で行う。集約・baseline・区間の
詳細はmoduleのdocstringにまとめている。複数のchunk（それぞれ単独で完全なsource）を受け取り、
manifestの分割を合わせたものが登録したseed範囲とちょうど一致することを、ラベルを読む前に
検査する。

設定は2層に分ける。

| 層 | 内容 | 扱い |
|---|---|---|
| protocol preset（`Preset`） | 分割のseed範囲、bootstrap回数・乱数seed、Jeffreys、clip幅、判定保留の閾値、`purpose` | 事前登録して固定する。version付きで、`--preset-sha256`が一致しなければラベルを読む前に拒否する。個別に上書きするoptionはない。変える場合は新しいpresetを事前登録する |
| 差し替え点（`ScopedWaitEstimator`） | 比べる推定器 | 推定器が値を出す行（scope）を宣言する。scope外の行は未提供（`None`）のまま扱い、ゼロ予測にしない。#245は`Riichi245Estimator` |

- #257の条件は`PRESET_257`（`hand-belief-accuracy-257`）。`--reproduce-of result-257.json`を
  付けると、記録済みの結果と測定値（件数・指標・区間・較正）が一致するかを比べ、違うkeyを
  報告して終了コード1を返す。実行環境・pathなどは比べない
- clip幅（`1e-6`）はpresetでは変えられない（#245の学習側と共通のprotocol invariant）。
  seed分割の重複検査、未提供値の扱い、同じ行集合でのpaired比較、episode-macro集約、
  正例0件の明示は、presetによらず共通
- `purpose`は`development-baseline`か`formal-test`。`formal-test`では
  `--allocation-identity` / `--ledger-revision` / `--arena-revision` / `--evaluator-revision`
  が必須で、結果に記録する。live ledgerとの照合はlisjong-arenaの`check-allocation`の責務で、
  lisjong側は検査せず記録だけを行う（arenaへ依存しない）
- 結果（`lisjong-hand-belief-accuracy-result-v2`）は、presetとそのSHA-256、推定器のidentity、
  各chunkの`manifest.json` / `coverage.json`のSHA-256、予約の識別情報、実行環境
  （CPython、wall time、最大RSS）を含む。評価はAWSを必要としない

このsourceを使う形別7channelの改善（#260）は、
[形別待ちテーブルの推定・評価設計](wait-shape-belief.md)に対象・baseline・整合性・
判定条件をまとめている。同文書は実装前設計であり、実測結果ではない。

ロン合法確率（#262）の追加factは[ロン合法確率の設計](ron-legal-belief.md)に定義する。
v1単独には見逃し状態・検証履歴がないため、その正解計算には使用できない。
追加契約は別sourceとしてv1を参照し、本契約のschemaを変更しない。
