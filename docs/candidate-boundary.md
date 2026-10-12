# 候補Policyの情報境界チェック

lisbun/lisjong#296。自律強化ループ（lisbun/lisjong-project#97）のS0で、探索やLLMが
作った候補Policyが`DecisionContext`以外の情報を参照していないことを機械的に確かめる。
実装は`src/lisjong/candidate_boundary.py`、testは`tests/test_candidate_boundary.py`。

## 使い方

```python
from lisjong.candidate_boundary import check_candidate_imports, check_candidate_purity

imports = check_candidate_imports("lisjong.policies.placement_aware_speed_call")
purity = check_candidate_purity(
    "lisjong.policies.placement_aware_speed_call:PlacementAwareSpeedCallPolicy",
    decisions,  # DecisionContextの列
)
assert imports.passed and purity.passed
```

どちらも不合格の理由を値で返す（`violations` / `mismatches` / `errors`）。S0では
`passed`がFalseの候補を落とす。

## import制限

候補module（packageなら配下の`.py`すべて）に対して、次を検査する。

| rule | 内容 |
| --- | --- |
| `import-not-allowlisted` | 直接importできるのは、許可した標準library（`dataclasses`・`typing`・`math`・`collections`等）、`lisjong.policy_contract`・`lisjong.policies`・`lisjong.belief`・`lisjong.hand_evaluation`・`lisjong.structural_efficiency`・`lisjong.action_vocabulary`、候補package自身の配下だけ。`sys`・`inspect`・`gc`・`importlib`・`os`等は含めない |
| `forbidden-import` | `lisjong_arena`・`lisjong_engine`・`riichienv`、`lisjong.learning`、training-only truth（`lisjong.belief.exact_wait_ground_truth`・`riichi_ron_label`・`ron_legal_ground_truth`） |
| `forbidden-origin` | `from M import name`や`m.name`で取り込んだobjectの定義元が上の禁止module。`lisjong.belief`からのre-export等を拒否する |
| `star-import` | `from M import *`（取り込む名前を検査できない） |
| `forbidden-builtin` | `getattr`・`eval`・`exec`・`__import__`・`globals`・`vars`・`open`等 |
| `forbidden-attribute` | `_getframe`・`f_back`・`f_globals`等のframe参照と、`__globals__`・`__dict__`・`__subclasses__`等のobject graph走査 |
| `runtime-import` | `lisjong_arena`・`lisjong_engine`・`riichienv`のimportを遮断した別interpreterで候補moduleをimportし、失敗した、またはこれらが読み込まれた |

`lisjong.learning`の推定器を使う候補（例: `TenpaiPushFoldPolicy`）は、現時点では
対象外で不合格になる。必要になった時点で、許可するmoduleを個別に足す。

既存Policyへの適用は要求しない。例えば`TargetedHonorReleaseTerminalProgressionPolicy`は
`getattr`を使うため`forbidden-builtin`で不合格になる。規則は新しい候補を保守的に
落とすためのもので、既存Policyの正しさの判定ではない。

## 純粋性

`PYTHONHASHSEED`の異なる新しいprocess 2つで、同じ`DecisionContext`列に対する行動を比べる。

- process A（`PYTHONHASHSEED=0`）: Policyを1つ作り、正順に2周する
- process B（`PYTHONHASHSEED=1`）: Policyを1つ作り、逆順に1周する

各decisionの行動が全runで一致しなければ不合格。hash順序・呼び出し回数・呼び出し順序への
依存を検出する。返却値は`execute_policy()`で合法候補へ照合し、例外・不正な返却値・
禁止packageのimportは`errors`として不合格にする。Policy契約はPRNG・seedを入力に
持たないため、seedは渡さない。

## 範囲外

- 汎用sandbox。静的検査と実行時検査は機械的な一次選別で、悪意あるcodeの完全な隔離ではない
- `DecisionContext`へのprojection自体の正しさ（Arena側）
- 反実仮想検査（同じ観測で隠れ状態だけが異なる局面での行動一致）。上の2点で不足すると
  分かった場合に別Issueで扱う
