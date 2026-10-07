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
- 時点違反: 他家の手牌の`sequence`が判断の`sequence`より後。同じ`sequence`は、判断の行動を
  適用する前に取ったsnapshotを意味する（観測者の行動は他家の手牌を変えない）
- 他家の副露が`PolicyInput`の公開副露と一致しない
- 手牌が13枚相当（`len(concealed) + 3 * len(melds) == 13`）でない
- 他家3席のconcealed tilesの合計が、`PolicyInput`から見て未確定の牌
  （`derive_remaining_tile_inventory()`）を牌種別・赤5別に超える

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
