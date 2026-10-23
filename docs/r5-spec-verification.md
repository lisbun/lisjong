# R5の仕様由来fixtureとcaller契約

[#337](https://github.com/lisbun/lisjong/issues/337)は、Python DP廃止前の検証補完。
実装、backend既定値、API_VERSION、既存差分test、稼働pinは変更しない。
追加testは`tests/test_progression_spec_fixtures.py`。
期待値は以下の手計算から定数として記述し、Python evaluator、Rust出力、unpruned referenceで生成しない。

## 分布の導出

terminal distributionは打牌後の向聴数0〜8の9軸。確定3面子に対する4枚字牌手では、
2対子または刻子+単騎が0向聴、対子+2単騎が1向聴、4単騎が2向聴。
字牌には順子がないため、下記は対子・刻子を数えるだけで導出できる。
rootや仮想打牌は残牌へ戻さない。残牌は牌種でなく物理枚数で非復元抽出する。

### 有効牌の多重度と非復元抽出

root `1123z`、残牌 `22456z`。`2z`を1枚でも引けば2対子となり聴牌を維持できる。
`4z/5z/6z`だけでは対子を増やせない。全列数は5、20、60で、非聴牌列数は
3、3×2、3×2×1。よってhorizon 1/2/3の分布は次のとおり。

| horizon | 0向聴 | 1向聴 | 総列数 |
| --- | --- | --- | --- |
| 1 | 2 | 3 | 5 |
| 2 | 14 | 6 | 20 |
| 3 | 54 | 6 | 60 |

残りの軸はすべて0。1向聴のroot値も定数として確認する。

### 直ちに向聴を下げないツモの価値

残牌 `556z`。A=`1123z`、B=`1234z`、C=`1124z`とする。
最初の`5z`は向聴を下げないが、保持して2枚目の`5z`を引けば対子になる。
depth2の物理列6通りは`55`が2通り、`56`と`65`が各2通り。
depth3では全6列に`55`が含まれる。root shantenはA/Cが1、Bが2。

| horizon | A/Cの分布（0,1,2向聴） | Bの分布（0,1,2向聴） |
| --- | --- | --- |
| 1 | (0,3,0) | (0,0,3) |
| 2 | (2,4,0) | (0,2,4) |
| 3 | (6,0,0) | (0,6,0) |

A/Cは異なる手牌で同分布。非対称な順序でA/B/Cと重複Aを一括評価し、rootの順序と重複を保持することも固定する。
別途`1122z`の聴牌rootと残牌`155z`で、horizon 1/2/3の全質量3/6/6が0軸に集まることを確認する。
和了ツモ後も打牌するモデルなのでterminal軸は-1にならない。

## 分布とActionの同点処理は別の契約

caller fixtureは`111m222p333s11234z`、合法候補を`1z/3z/4z`切りに限定。
公開河から導出される残牌を`556z`に固定する（hidden情報や架空のDP戻り値は渡さない）。
完成した3刻子を維持し、残る字牌について上表を適用できる。
`1z`切りは4単騎から対子が1つできるのでterminalは1、`3z/4z`切りは2対子でterminalは0。
残牌には3枚目の`5z`も既存対子の`1z`もないため、3ツモ内のcompletion massは全候補0。

- progression: 向聴massは6/0/0、分布はそれぞれ1軸へ6、0軸へ6、0軸へ6。
  `3z/4z`の同点ではHandValueAwareへ委譲する。西/北の価値・役route・受入は同じで、
  stable orderにより`3z`を選ぶ。合法候補の入力順を反転しても同じActionになる。
- diagnostic: full-legalとbaseline-eligibleは同集合。候補ごとの分布/mass、分母6、
  baseline選択`3z`、両summaryのbest=0・selected=0・regret=0を定数で確認する。

## 既存coverageとの対応

| 対象 | 既存の契約テスト | #337の補完 |
| --- | --- | --- |
| progression | `test_terminal_shanten_progression_mechanism_riichi_defense_policy.py`: exact tie、unique min、strict subset tie。`test_native_progression_backend.py`: oracle同値・実native wiring | 実native分布を使う上記14枚手の固定Actionとcandidate値 |
| targeted honor release | `test_targeted_honor_release_terminal_progression_policy.py::ActivationAndSwitchTest`: 固定massでparent best/tie保持、非字牌best保持、字牌のみbestのswitch、複数字牌bestのHVA委譲、入力順独立。`ThreeCallersEquivalenceTest`と`RustSelectedProductionWiringTest`: native比較/接続 | 選択契約の不足なし。既存固定値テストを再実装しない。DP自体の定数検証は本Issueの共通fixtureで補う |
| diagnostic | opt-in/all-zero gating、full/eligible subset、差分・mass/分母不変条件、native一括呼出 | 候補histogramと両summaryの独立固定値 |
| native境界 | horizon/count/手牌枚数/5枚目、overflow、release異常入力、counter、cache lifetime、GIL | 変更・重複追加なし |

仮想discardのcanonical同mass処理（異なる分布が同massとなる場合）の既存差分testも保持する。
本Issueの同分布fixtureはそれを置換したとは扱わない。Python oracleを実際に削除するPRでは、
削除する各testと代替契約の対応を改めて確認する。

追加native testは既存`_require_native`を使用する。通常のnative未導入環境ではskip、
`LISJONG_REQUIRE_NATIVE=1`では不在をfailにする。既存AL2023 native full suiteの
`discover -s tests`でこのmoduleも実行される。CIを変更してskipを増やすことはしない。
