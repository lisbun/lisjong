# 1向聴・現物なしの対リーチ守備候補

Issue: [#234](https://github.com/lisbun/lisjong/issues/234)

`OneShantenDefensePlacementAwareSpeedCallPolicy` は
`PlacementAwareSpeedCallPolicy` を親とする未昇格の実験候補。

```python
from lisjong.policies import OneShantenDefensePlacementAwareSpeedCallPolicy

policy = OneShantenDefensePlacementAwareSpeedCallPolicy()
action = policy.choose_action(decision_context)
```

## 変更する判断

元の合法打牌集合について、以下がすべて成立するときだけ発動する。

- 他家がリーチ中（既存判定を再利用する）。
- 全リーチ者への共通現物が合法打牌集合にない。
- 打牌後の最小向聴数がちょうど1。

各候補の既存mechanism危険度を公開情報だけから計算し、複数リーチ時は
最も高い相手のscoreを使う。そのscoreが最小の候補をすべて残し、親へ
渡す。赤牌・通常牌・手出し・ツモ切りのaction objectと順序を維持する。

このscoreは校正された放銃確率ではない。安全牌が確定するという意味でもない。
向聴数を戻して降りる場合があるため、放銃損失と聴牌機会のtrade-offを
対戦で評価する必要がある。候補はcross-decision stateを持たない。

発動条件外では入力集合を変更せず親へ委譲する。和了・立直・鳴きの
orchestration、聴牌時の攻撃判断、FH/HVA/R5を直接変更しない。
同じ入力に対する非対象判断は維持されるが、途中の打牌が変われば、
後続の手牌や鳴き機会なども変わる。

## 検証と評価の区別

- focused test: 発動境界、複数リーチ、元action identity、親への委譲、
  オーラストップ時の選択同等性、trace有無の行動一致。
- 開発比較: RiichiEnv 0.4.10、4p-red-half、AABB 4 rotations、
  historical RETIRED seeds 0..19の80半荘。Aが本候補、BがChampion。
  ウマ・オカ込みseed-block平均差と記述的95%区間を用いる。
- 再利用seedによる開発比較は、新規の正式評価や昇格根拠にはしない。
  結果・revision・生記録はIssueと外部artifactへ残す。
- 正式な強さ判定には、独立した未使用populationと事前固定条件による
  Arena評価が必要。既存Champion designationは維持する。

Arenaのcurated aliasや正式protocolは、このlisjong実装PRでは追加しない。
外部比較では上記classを明示した`PolicySpec`のfactoryとして渡す。
