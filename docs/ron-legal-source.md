# ロン合法source v1（#262 B1）

[意味・採用ルール・後続測定](ron-legal-belief.md)に従う、学習専用の追加契約。
`lisjong.learning.ron_legal_source.read_ron_source(extension, base_directory=base)`が
完全性・履歴・和了機会を検査する。`read_labelled_ron_source()`は同じ検査を行い、
既存`HandBeliefLabelledDecision`の他家truthへbinaryなロン合法テーブルを付ける。
`exact_hand_belief_with_ron()`は検証済みの具体的手牌・河・contextを受ける正解計算器で、
履歴の欠測を補う入口ではない。Cは`project-standard-normal-discard-ron-v1`。

旧v1 sourceは変更しない。以下は別directoryに置く。場所をmanifestから探索しない。
JSONの正準形式は既存Learning sourceと同じ。manifestはsorted/indent=2/LF、
JSONLはsorted/compact/LF。未知field、重複key、欠測、未知enum、bool/intの混同を拒否する。

## manifest.json

schema: `lisjong-ron-legal-source-manifest-v1`。top-levelは次の8fieldだけ。

| field | 値 |
|---|---|
| `schema` | 上記識別子 |
| `base_manifest_sha256` | 明示的引数の元v1 manifestの64桁lowercase SHA-256 |
| `context_protocol` | 固定C identity |
| `rules` | 下記の厳密なsemantic projection |
| `producer` | 元v1と同じ`arena_revision`, `lisjong_revision`, `lisjong_engine_revision`, `policy` |
| `splits` | 元v1と同じ`train`, `valid`, `test`の正確なseed配列 |
| `files` | `ron_facts`, `ron_history`各々の`bytes`, `rows`, `sha256` |
| `coverage` | runnerが独立に数えた、file順の局別counter配列 |

`rules`は次と型も含め完全一致する。これは役・フリテン・ロン解決に必要な値のprojectionで、
engineのRuleSet objectの復元ではない。別RuleSetはこのversionでは受理しない。

```json
{
  "identity": "project-standard-v1",
  "kuitan_enabled": true,
  "red_dora_enabled": true,
  "rounded_mangan_enabled": false,
  "counted_yakuman_enabled": true,
  "multiple_yakuman_enabled": true,
  "double_yakuman_variants": [],
  "double_wind_pair_fu": 4,
  "kokushi_ankan_chankan_enabled": false,
  "ron_resolution_policy": "multiple_ron",
  "triple_ron_abortive_draw": true
}
```

各coverage itemは`seed`, `round_id`（非空string、同じseed内で一意）、
`transitions`, `reactions`, `decisions`, `selectors`だけを持つ。
counterはnon-negative int。transitionsは開始・終端markerを含む外側のjournal行数、
reactionsは和了機会を検証するreaction step数、decisionsは元v1対象判断数、
selectorsは対象外判断も含む全selector呼出し数。
markerを除いた実commit数は`transitions - 2`としてrunner側counterと対応付ける。
全宣言seed・全局・局末までの進行を記録する。対応判断がない局も省かない。
readerはcounter、連続性、key集合を照合するが、runnerとjournalの両方を同じように
誤記した場合に真実性を証明できるわけではない。

## ron_facts.jsonl

各行は`schema`, `key`, `round_id`, `history_boundary`, `opponents`だけ。
schemaは`lisjong-ron-legal-fact-record-v1`、keyは元v1の`seed/sequence/seat`。
opponentsは自席以外の3席をseat順に並べ、各itemは`seat`, `sequence`, `context`だけ。
sequenceはkeyと厳密に同値。contextは次の3fieldだけ。

```json
{"missed_ron_state":"none","riichi_status":"none","is_ippatsu":false}
```

理由は`none/temporary/riichi`、成立statusは`none/riichi/double_riichi`、一発はbool。
未成立でriichi理由または一発は拒否する。捨て牌フリテンは現在の真のWと河から導出する。
事実は非聴牌行を含む元v1の全keyにちょうど1行ずつ必要。

boundary=kは外側journalのprefix `[0,k)`であり、最後のcheckpointは行k-1。
行kの最初のstepがその判断のsequenceと選択actionを消費する必要がある。
進行中のreaction選択収集中、同一commitの途中、未来の状態を参照できない。

## ron_history.jsonl

schemaは`lisjong-ron-legal-history-record-v1`。
各行は`schema`, `seed`, `round_id`, `index`, `steps`だけ。
indexは局ごとの0始まり連続int。開始・終端marker以外は実際のcommitを1行として記録し、
そのcommitで生じたsemantic stepを順に並べる。例えばreaction解決とリーチ成立は
同じ外側行に入り得る。内部checkpointは検証用factであり、selectorのsnapshotには使わない。
selectorを呼ばなかった進行も記録する。1行に複数のselector windowをまとめない。

各stepは`selector_sequences`, `event`, `checkpoint`だけ。
selector_sequencesは半荘全体の0始まり通し番号。重複・欠落・逆順を拒否する。
reactionでは複数席の番号を昇順に記録し、それ以外のselector actionは1番号、
draw・成立・result等の非selector事象は空配列。

checkpointは`views`, `contexts`だけ。viewsは4席の完全な`PolicyInput`をseat順に並べ、
それぞれのself_seatとown_handを保持する**学習専用**factである。
4つのround/playersは一致する必要がある。concealed_tilesはdrawn_tileを既に含む。
contextsは4席順の上記context。public ACCEPTEDと成立status、門前制約、13/14相当の枚数、
表示牌・鳴かれた河を含む全体の物理枚数と赤/通常5を検査する。
元v1とのjoinでは観測者のPolicyInput全体、他家手牌・副露・contextを照合する。

eventは次の閉じたshapeだけを受理する。

| kind | kind以外のfield | semantics |
|---|---|---|
| `round_start` | なし | 最初の行の単独step。4席13枚、河・副露なし、見逃し/成立/一発なし |
| `draw` | `seat`, `draw_kind` (`normal/rinshan`) | 実ツモを前後のhand/drawn_tileで照合。自席TEMPORARYだけ解除 |
| `progress` | `sequence`, `action` | v1と同じtyped actionの明示projection。reaction actionは不可。打牌で自席一発終了、リーチ宣言でpublic DECLAREDへ。槓宣言だけでは一発を消さない |
| `reaction` | `evidence` | 下記の観測機会を検証し、capable − selectedだけに見逃しを記録。成立した鳴きで全席一発終了 |
| `riichi_established` | `seat`, `riichi_status`, `is_ippatsu`, `reaction_id` | 宣言牌のpass/call解決へ参照し成立。鳴かれた宣言牌では一発false |
| `riichi_cancelled` | `seat`, `reaction_id` | 宣言牌のron解決へ参照してpublic DECLAREDをNONEへ戻す。成立役を作らない |
| `kan_confirmed` | `seat` | 記録済み槓宣言の成立。全席一発終了。加槓は同じ席・同じ加槓牌のkakan reactionを1回解決済みであること |
| `round_result` | なし | engine-owned結果・精算のcheckpoint。新しい見逃しを作らない |
| `round_end` | なし | 最後の行の単独step。result等で終局済み、直前と同じcheckpointをflush |

engineの合法手生成・全局遷移・精算をlisjongへ再実装しない。
board checkpointはproducerの観測fact。readerは物理保存・v1との一致・contextを変える
事象とその順序・実ロン機会を照合する。producerの真の内部handを別情報から推測しない。

### reaction evidence

fieldは`reaction_id`, `origin`, `source_seat`, `winning_tile`, `discard_draw_kind`,
`candidates`, `ron_capable`, `ron_selected`, `ron_awarded`, `ron_passed`,
`resolution`, `resolved_action`だけ。

- IDは局内で一意。originは`discard/kakan/ankan`。winning_tileは赤を区別する既存Tile projection。
  discardは最新の未鳴き河牌と一致し、kanは手牌・元Pon/4枚と照合する。
- discard_draw_kindは`normal/rinshan/null`で、履歴に記録した打牌元の実ツモと一致する。
  鳴き後打牌はnull、kan originもnull。
  live wall=0でも嶺上由来の打牌へ河底を付けない。kakanはCHANKANで照合する。
  project-standard-v1は国士の暗槓槍槓を無効にするため、暗槓にはreaction windowがなくankan reactionは拒否する。
  kakan reactionは保留中の加槓宣言と同じ席・同じadded_tileに限り1回だけ受理する。
  槓宣言から成立（または槍槓ロン）までの間にreaction以外の進行・打牌reactionは置けない。
- candidatesは発生元以外の3席をseat順に並べる。各itemは`seat`, `sequence`,
  `legal_actions`, `selected_action`。機会なしは空actions・null sequence/selected、
  機会ありは明示Passを含むactionsとその中の選択を持つ。actionは既存typed projection。
  Ronの有無は当時のstable hand、全河、replayしたcontext、#263の役判定で再検証する。
  non-ron合法手全体の生成はengineの責務。actor/target/牌・消費牌・リーチ制約を検査する。
- 4つのron集合は発生元の次席からのturn順で重複なし。capableは再検証済みRon提示席、
  selectedは実際のRon選択席、passedはcapable − selected。
  resolutionは`ron/call/pass`。固定RuleSetではRon選択は全員awardedで、3人選択も
  engine E2のawarded集合は保持する。三家和の流局は後のE3結果であり、集合を空へ書き換えない。
  選択したが最終的に和了不成立でもpassedへ追加しない。
  head-bump RuleSetはこのschemaのrule projectionでは受理しない。
- resolved_actionはcall解決で実際に成立した選択action、それ以外はnull。
  未リーチのpassedはtemporary、成立済みのpassedはriichi理由になる。
  リーチ後のツモ見逃しを新しい理由として扱わない。

## 検証と残る作業

合成fixtureはsourceの正準形・key・digest・coverage、外側commitと半開snapshot境界、
同巡/リーチ後見逃し、他家ツモ・鳴き・自席/嶺上ツモ、宣言/成立と一発の順序を固定する。
native testは門前/副露の役あり・役なし、捨て牌フリテン、特殊contextの過去の機会、
赤/通常5と反実仮想の物理境界、source joinからの実ラベルを検証する。
native未導入環境で役のtestはskipし、`LISJONG_REQUIRE_NATIVE=1`のCIでは必須にする。
正解生成はnative/APIを事前確認し、非聴牌でも利用不可をラベル0へ変換しない。

推論入力は旧`read_decisions()`のplayer-safe記録だけであり、追加directoryを開かない。
追加source・正解moduleを禁止した別processで公開情報からの既存推定が動くことも確認する。
このschemaの実対局producerと実測は未実装。専用hookによるcommit内の観測fact取得は
Arena/engineで別途具体化する必要があり、現在のplayer-safe APIで取得できるとは主張しない。
