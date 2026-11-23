# 研究ロードマップの現在地

確認日: 2026-11-23 JST。Issue本文・コメント・merge記録を照合したsnapshotであり、artifactの再実行監査ではない。
長期能力は[project roadmap](https://github.com/lisbun/lisjong-project/blob/main/docs/roadmap.md)、
配分は[project #84](https://github.com/lisbun/lisjong-project/issues/84)、Learning内部は[#74](https://github.com/lisbun/lisjong-project/issues/74)を正本とする。

## 直近の順序

1. **完了**: 凍結L2a M1+R1対Arm Yの正式比較とLearning family昇格（#368、Arena #582、project #127）。
2. **完了**: [Arena #583](https://github.com/lisbun/lisjong-arena/issues/583) / [#588](https://github.com/lisbun/lisjong-arena/issues/588#issuecomment-6886499017)のHeuristic対Learning Overall正式比較。
   400/400 VALID、`HEURISTIC CHAMPION SUPERIOR`（`D=L順位−H順位=+0.4775`、95% [+0.368, +0.587]）。
   ユーザー承認によりOverall Championを`placement-aware-speed-call`（Heuristic / DIRECT）に指定（[project #69](https://github.com/lisbun/lisjong-project/issues/69#issuecomment-6886584183)）。
3. **未決**: 候補RR（リーチ下1向聴の押し引き、#377）の扱い。#379の東1局はSUPPORTED-DEVELOPMENT、
   半荘の[Arena #591](https://github.com/lisbun/lisjong-arena/issues/591)はINCONCLUSIVE-DEVELOPMENT。打点推定版[#384](https://github.com/lisbun/lisjong/issues/384)はNOT-SUPPORTED-DEVELOPMENT。
   判断は[project #84](https://github.com/lisbun/lisjong-project/issues/84)のユーザー判断待ち。
4. **未選定**: 新しい強化仮説。既存の根拠から対象判断・頻度/support・収支経路・失うもの・費用・停止条件を比較し、
   選定した1案を限定Issueへ渡す。
5. **推定研究の優先**: 待ち推定は[#285](https://github.com/lisbun/lisjong/issues/285)を優先する（2026-11-23ユーザー判断）。
   打牌行動尤度による他家待ち推定は研究候補としてDEFERREDとし、Issueは起票しない（[project #84の記録](https://github.com/lisbun/lisjong-project/issues/84#issuecomment-6963636607)）。

Heuristic Championは`placement-aware-speed-call`、Learning Championはexact L2a M1+R1、
Arm Yはpredecessor。**現pairのOverall Championは`placement-aware-speed-call`（Heuristic / DIRECT）**。正確なbindingと適用範囲は[Policy status](policy-status.md)。

## lisjongの残件と直近完了

| Issue | 現在の状態 | 次の条件・作業 |
| --- | --- | --- |
| [#285](https://github.com/lisbun/lisjong/issues/285) リーチ者の構造的待ち推定v2 | open・**推定研究で優先**（2026-11-23）。本文は研究設計、実装・正式評価の完了報告なし | 特徴量・対象・選択規則を事前登録。A1精度改善とA2提供範囲拡大を別判定。棋力改善とは別。formal test用seed予約・source生成は別承認 |
| [#286](https://github.com/lisbun/lisjong/issues/286) 門前非リーチ者の待ち推定 | DEFERRED | 損失集中の証拠または予算配分の判断後、正例supportを得るpopulationを設計。未提供はNone |
| [#297](https://github.com/lisbun/lisjong/issues/297) 決定的な候補生成 | open。探索対象・候補列生成は未完了 | 次仮説・候補identity・適合する評価契約が必要。汎用探索の完成を強化の前提にしない |
| [#368](https://github.com/lisbun/lisjong/issues/368) L2a対Arm Y | 比較・昇格とも完了 | 後続のOverallはArena #583で完了（Heuristic superior）。段階2の外部棋譜追加学習は別Issue・未開始 |
| 打牌行動尤度による他家待ち推定（Issueなし） | 研究候補・DEFERRED（[project #84](https://github.com/lisbun/lisjong-project/issues/84#issuecomment-6963636607)） | 原則#285のA1結果または終了判断の後に再検討。#285のselect用seed `933000..933399`とformal test用seedを使わない。新規モデル・学習・seed予約なし |

## 完了した実験の扱い

| 系列 | 固定した結論 | 継続境界 |
| --- | --- | --- |
| L2a模倣 / CALL特徴追加 | #318: 打牌・Chi NOT ESTABLISHED、Pon PARTIAL。#326: NOT-SUPPORTED-DEVELOPMENT | family昇格で模倣結果を変更しない。OFFLINE-EVAL再利用なし |
| R1価値学習 | Arena #550: NOT-PREDICTIVE | 今回実験終了、系列は再開条件付き保留。具体的新仮説なしに作業を割り当てない |
| P1価値学習 | Arena #558: NOT-PREDICTIVE | 今回実験終了、E4へ進まない。系列全体の棄却ではない |
| リーチ下1向聴の押し引き RR / 打点推定版 | #379: RRのみSUPPORTED-DEVELOPMENT（東1局）。Arena #591: 半荘INCONCLUSIVE-DEVELOPMENT。#384: 打点推定版NOT-SUPPORTED-DEVELOPMENT | RRの扱いは#84で未決。#384は昇格させず救済しない。#377 / #379 / #384 / Arena #591の標本を合算しない |
| paijia v1 / genbutsu v1 / call-value v1 / H5 | Arena #522: NOT-RECONFIRMED、#526/#536: STOP-LOW-ACTIVATION、#542: STOP-NO-SIGNAL | 既定候補へ戻さず、追加seed・条件緩和で救済しない |

旧正式実行#573のseedはRETIRED、#582はCOMMITTED。使用済みTESTや開発seedを独立Overall TESTへ流用しない。
新規対局・学習・seed予約・課金・稼働bot切替・自律loop開始は、この文書整理の範囲外。
