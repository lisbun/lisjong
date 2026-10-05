# 攻撃制限付きFALLBACK回し打ち候補（#249）

Issue: [#249](https://github.com/lisbun/lisjong/issues/249)（親: [#236](https://github.com/lisbun/lisjong/issues/236)）

Championが単独リーチに対して1向聴・現物なしで攻撃評価だけで打牌を選ぶ判断（`FOLD_FALLBACK_ALL_LEGAL`）を、
攻撃面の評価を大きく落とさない範囲で、より安全な打牌へ替える実験候補の仕様である。
どちらも未昇格で、強さは未評価。Championは書き換えていない。

| class | 危険度 | 役割 |
|---|---|---|
| `lisjong.policies.ClassicalAttackLimitedMawashiPolicy` | 古典的危険度score（固定weightの相対値） | 比較版。回し打ち制限そのものの効果を見る |
| `lisjong.learning.riichi_wait_mawashi_policy.RiichiWaitAttackLimitedMawashiPolicy` | #245の推定器が出す構造的待ち確率 | 推定器の寄与を見る |

2つの違いは危険度の生の値だけで、ゲート・攻撃制限・ロンされない牌の扱い・同点処理は同じ関数
（`decide_attack_limited_mawashi()`）を通る。exact parentはどちらも`PlacementAwareSpeedCallPolicy`。

```python
from pathlib import Path

from lisjong.learning.riichi_wait_mawashi_policy import (
    RiichiWaitAttackLimitedMawashiPolicy,
)
from lisjong.policies import ClassicalAttackLimitedMawashiPolicy

classical = ClassicalAttackLimitedMawashiPolicy()
wait = RiichiWaitAttackLimitedMawashiPolicy.from_selection(Path("selection.json"))
```

## 対象の判断（ゲート）

元の合法打牌集合について、次がすべて成り立つ打牌判断だけを対象にする。

1. 自分は非リーチで、他家のちょうど1人がリーチ中（宣言済みを含む）
2. 門前（副露は暗槓だけ）
3. 合法打牌が2牌種以上
4. 点数状況modeが`ALL_LAST_TOP_SPEED`ではない（オーラストップ目の事前守備はChampionの処理を優先する）
5. Championの分類が`FOLD_FALLBACK_ALL_LEGAL`（聴牌を保てる打牌がなく、リーチ者の現物も合法打牌にない）
6. 打牌後の最小向聴がちょうど1

対象外の判断では、Championの`PolicyDecision`をそのまま返す。和了・立直・鳴きの判断は継承し、変えない。
途中の打牌が替われば、その後の手牌や鳴き機会は変わる。

## 攻撃制限（C0比）

C0は、同じ入力でChampionが選ぶ打牌である。各合法打牌 `a` について次の4つを計算する。

| 値 | 定義 |
|---|---|
| `shanten(a)` | 打牌後の向聴数 |
| `ukeire(a)` | 打牌後の受け入れ枚数（Policyから見える残り枚数） |
| `mass(a)` | FiniteHorizon completion mass（Championと同じhorizon） |
| `value(a)` | 手牌価値proxy（打牌後に残るドラ・赤ドラ枚数 + 完成済み役牌の翻相当。`_retained_real_value`） |

次をすべて満たす打牌だけを候補に残す。

```text
shanten(a) == shanten(C0)
4 * ukeire(a) >= 3 * ukeire(C0)
4 * mass(a)   >= 3 * mass(C0)
value(a)      >= value(C0)
```

- 向聴は戻さない。#236段階2の候補は向聴を戻して降り、和了・流局聴牌を失った
- 受け入れとFH massは、C0の3/4以上を保つ。比は整数で比較する
- 手牌価値proxyは下げない（ドラ1枚は1翻で、打点への影響が受け入れ3/4より大きいため、許容幅を設けない）
- C0自身は必ず条件を満たすので、候補集合は空にならない

閾値（3/4、手牌価値の低下0）は対局結果を見る前に固定した値で、結果を見て変えない。変える場合は別の候補として
新しいseedで評価する。

## 危険度と選択

1. ロンされない牌（Policy側の扱い）: 次の牌種は危険度を0とする。公開情報だけで確定するものに限る
   - リーチ者自身の捨て牌にある牌種（ゲートにより、合法打牌には含まれない）
   - リーチ者の最後の打牌より後に他家（自分を含む）が切った牌種
2. それ以外の牌種は、危険度の生の値をそのまま使う
   - 古典score版: `_classical_riichi_danger_score(...).total`
   - 待ち推定版: `LogisticWaitModel.predict()`の構造的待ち確率（clip済み、0にはならない）
3. 候補のうち危険度が最小のものを選ぶ。ただし**C0より厳密に小さい候補がなければC0を維持する**
4. 危険度が同じ候補は、`mass`・`ukeire`・`value`の大きい順、最後にcanonicalな打牌順（`discard_action_sort_key`）で選ぶ

スジ・壁・両面の反対側などの推測で、推定器の生の値を0に上書きしない。観測していないフリテンを確定扱いしない。

構造的待ち確率は「その牌を1枚加えるとリーチ者の手牌が完成形になる確率」の推定であり、放銃確率でも期待放銃点
でもない。古典scoreも相対値である。どちらも、同じ判断の候補牌を並べる順位としてだけ使う。

## 推定器のモデル

- #245の正式評価で選択を固定した`selection.json`の`estimator_logistic`を使う。再学習しない
- `from_selection()` / `load_selected_wait_model()`は、ファイルのSHA-256が
  `14475264d7fe4137a9ac8a23ee1d27b420bff2f4434c6e93ca72ecfcf24ccc38`と一致しない場合、またschema・特徴量セット・
  clip幅がコードと違う場合に`RiichiWaitModelError`で拒否する。別のモデルへ黙って替えない
- `selection.json`はrepositoryに置かない
- 正式testのseed（932000..932099）は、この候補の調整に使わない

## 情報境界

- 入力は`PolicyInput`だけ。リーチ者の手牌・ラベルを読まない
- 推論側の2 moduleは`riichi_deal_in_source`・`riichi_wait_evaluation`・`exact_wait_ground_truth`・`riichi_ron_label`を
  importしない（testで固定）
- Policy instanceは判断をまたぐ状態を持たない

## 観測用の記録

対象の判断では、`PolicyDecision.analysis`に`AttackLimitedMawashiAnalysis`（C0の打牌、選んだ打牌、合法打牌ごとの
4つの値・制限内か・危険度）を返す。対象外の判断では作らない。対象判断数、打牌を替えた割合、攻撃制限で外れた
候補数、C0を維持した数は、この記録から数える。

## 検証と評価の区別

- focused test（`tests/test_attack_limited_mawashi_policy.py`）: ゲートの境界、対象外でChampionと一致すること、
  攻撃制限の境界値、ロンされない牌の扱い、同点時のC0維持、打牌のidentity、情報境界、モデルのhash照合
- 対局での開発比較（Champion・古典score版・待ち推定版）は、seed・規模・相手構成・採否規則を#249へ事前登録してから
  実行する。このPRの時点では未実施で、強さについての主張はない
- 開発比較を通っても、正式な強さの証明やChampion昇格にはしない

## 対象外

聴牌時のPUSH/FOLD、複数リーチ、副露手・ダマの読み、推定器の再学習、Champion昇格、稼働botの変更。
