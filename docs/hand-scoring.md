# 役・符・点数計算

Issue: [lisjong #263](https://github.com/lisbun/lisjong/issues/263)。
Python API: `lisjong.hand_evaluation.scoring`。計算core: `native/src/scoring.rs`。

AIの判断と分析が、仮定の完成形（推定した相手の手、反実仮想の和了牌、offlineの正解手牌）の
役・翻・符・点数を求めるための純粋な計算器である。lisjong-engineの点数計算
（合法手判定・局進行・精算の正本）とは別の機能であり、engineへのruntime依存はない。

## 計算と推定の分離

- 入力は具体的な完成手と明示的な和了contextだけである。未完成手の打点期待値・和了確率・
  裏ドラ期待値など推定を含む量は扱わない。必要なcallerが候補和了形を構成してから呼ぶ
- データの取得・推定・正解の読み込みはcallerの責務である。計算器はonlineの推定手牌も
  offlineの正解手牌も受け取れる。onlineの判断経路で未知の他家手牌を入手しないことは
  caller側の責務であり、入力契約では制限しない

## 入力

| 入力 | 内容 |
| --- | --- |
| `WinningHand.concealed_tiles` | 門前牌。和了牌を**含めない**（13 - 3 × 副露数枚） |
| `WinningHand.melds` | `ScoringMeld(kind, tiles)`。`MeldKind`でチー / ポン / 明槓 / 加槓 / 暗槓を区別 |
| `WinningHand.winning_tile` | 和了牌（赤5を区別） |
| `WinContext.method` | ロン / ツモ |
| `WinContext.seat_wind` / `prevailing_wind` | 自風・場風。自風が東の席を親とする |
| `WinContext.riichi` / `is_ippatsu` | リーチ / ダブルリーチ / なし、一発 |
| `WinContext.situation` | 通常・海底・河底・嶺上・槍槓・天和・地和のどれか1つ |
| `WinContext.dora_indicators` | 表ドラ表示牌（通常表示牌の後に槓ドラ表示牌）。0〜5枚 |
| `WinContext.ura_dora` | 下の2モードのどちらか（必須） |
| `ScoringRules` | 喰いタン、赤ドラ、切り上げ満貫、数え役満、複合役満、ダブル役満の役、連風牌雀頭の符 |

`PROJECT_STANDARD_SCORING_RULES`はlisjong-engine `PROJECT_STANDARD_RULES`
（`project-standard-v1`）と同じ条件である（喰いタンあり、赤ありの136枚、切り上げ満貫なし、
数え役満あり、複合役満あり、ダブル役満なし、連風牌雀頭4符）。ここにない外部サービス固有の
ルールは推測で入れない。

和了牌の割当は、複数の面子分解だけでなく和了牌をどの面子・雀頭に割り当てるかまで列挙する。
ロンで完成した刻子は明刻として扱う。

### 裏ドラの2モード

| モード | 入力 | 結果 |
| --- | --- | --- |
| 完全な点数計算 | `UraDoraIndicators(tiles)`。リーチ時はドラ表示牌と同数、リーチなしは空（裏ドラなしの明示） | 裏ドラを含む点数。`dora.ura`は枚数 |
| 裏ドラ除外の基礎点計算 | `URA_DORA_EXCLUDED` | 裏ドラを含まない点数。`ura_dora_excluded=True`、`dora.ura=None` |

未知の裏ドラを黙って0枚として扱わない。役の有無だけを見る用途（#262）は除外モードで足りる。

## 出力

| 結果 | 意味 |
| --- | --- |
| `ValueError` | 牌枚数（同種5枚以上・赤5の重複を含む）・副露・context・ルールの不整合（fail closed） |
| `EvaluationStatus.NOT_COMPLETE` | 和了形でない |
| `EvaluationStatus.NO_YAKU` | 和了形だが役がない。ドラだけでは役ありにしない。正常な評価結果 |
| `EvaluationStatus.SCORED` | `WinScore`: 成立役と役ごとの翻・役満倍率、ドラ / 赤 / 裏、符と内訳、上限区分、支払内訳（親子、ロン / ツモ） |

- 複数の解釈（和了形 × 面子分解 × 和了牌の割当）から**和了者の受取点が最大**の解釈を選ぶ。
  同点時は役満倍率 → 翻 → 符の大きい順、それでも同じなら和了形（通常形 → 七対子 → 国士無双）、
  雀頭、手牌内面子のcanonical順、和了牌の位置（雀頭が先）の小さい順で1つに決める
- 役満は通常役と混ぜず、符を持たず、ドラを加算しない（`dora=None`）。
  数え役満は通常役として翻・符を保持し、上限区分が役満になる
- 本場・供託・責任払い（パオ）は含めない。局の精算はlisjong-engineの責務である

## native未導入時

計算coreはopt-in native拡張`_lisjong_native`にだけあり、Python版の計算器は持たない。
native拡張が未導入でも既存coreと`lisjong.hand_evaluation.scoring`のimport、入力型の構築は成功し、
`evaluate_win()`を呼び出したときだけ`ScoringBackendUnavailableError`になる。
`SCORING_API_VERSION`が一致しない古いwheelも同じエラーでfail closedする。
shanten backendの選択（`LISJONG_SHANTEN_BACKEND`）と`API_VERSION`には影響しない。
既存consumerへのnative必須化はしていない。

## 検証

- Rust単体テスト（`native/src/scoring/tests.rs`）: 手計算で求めた境界fixture。
  七対子25符、平和ツモ20符、門前ロン加符、ロンの和了牌割当（単騎 / 両面、明刻 / 暗刻）、
  役なし＋ドラ、二盃口と七対子、複合役満・ダブル役満、数え役満、切り上げ満貫、
  喰い下がり・喰いタン、赤5、裏ドラの2モード、不正入力
- bindingテスト（`tests/test_hand_evaluation_scoring.py`）: Python APIで同じ境界と、
  native未導入時の挙動を確認する
- engine一致: `tests/fixtures/scoring_engine_fixture.json`（2400件、全役をcover）。
  native CI jobでは`LISJONG_REQUIRE_NATIVE=1`によりskipせず実行する

engine fixtureは固定revisionのlisjong-engineから決定的に生成する。lisjongのtest・runtimeは
engineをimportしない。

```sh
git -C ../lisjong-engine checkout 96b9796c76ef5db8f3968f689a1ca6f3dfc9aa3b
python -m pip install -e ../lisjong-engine
python tools/generate_scoring_engine_fixture.py tests/fixtures/scoring_engine_fixture.json \
    --engine-checkout ../lisjong-engine
```

generatorは`ENGINE_REVISION`と異なるcheckout、local変更のあるcheckout、別の場所からimportした
engineを拒否する。engineとの差はfixtureの`max_candidates`（engineの同点最高得点候補）に
lisjongの選んだ解釈の要約が含まれるかで判定する。
