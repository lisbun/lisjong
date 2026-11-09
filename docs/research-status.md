# 研究ロードマップの現在地

確認日: 2026-11-10 JST。Issue本文・コメント・merge記録を照合したsnapshotであり、artifactの再実行監査ではない。
長期能力は[project roadmap](https://github.com/lisbun/lisjong-project/blob/main/docs/roadmap.md)、
配分は[project #84](https://github.com/lisbun/lisjong-project/issues/84)、Learning内部は[#74](https://github.com/lisbun/lisjong-project/issues/74)を正本とする。

## 直近の順序

1. **完了**: 凍結L2a M1+R1対Arm Yの正式比較とLearning family昇格（#368、Arena #582、project #127）。
2. **進行中**: [Arena #583](https://github.com/lisbun/lisjong-arena/issues/583)のHeuristic対Learning Overall比較準備。
   二モデルlock（#585）・局記録（#587）はmerge済み。development smoke/verifyはPASS。
   次は[Arena #588](https://github.com/lisbun/lisjong-arena/issues/588)の実行保護・失敗注入・環境適格性、
   WSL/AWS選択と資源・費用上限の確定。その後、別承認を得て100 seed blocks・4配置＝400半荘を一度だけ実行・検証する。
3. **未選定**: 新しい強化仮説。既存の根拠から対象判断・頻度/support・収支経路・失うもの・費用・停止条件を比較し、
   選定した1案を限定Issueへ渡す。既存L2a比較の完了と、新しい学習実験の未選定を区別する。

Heuristic Championは`placement-aware-speed-call`、Learning Championはexact L2a M1+R1、
Arm Yはpredecessor。**現pairのOverall Championは未確立**。正確なbindingと適用範囲は[Policy status](policy-status.md)。

## lisjongの残件と直近完了

| Issue | 現在の状態 | 次の条件・作業 |
| --- | --- | --- |
| [#285](https://github.com/lisbun/lisjong/issues/285) リーチ者の構造的待ち推定v2 | open。本文は研究設計、実装・正式評価の完了報告なし | 特徴量・対象・選択規則を事前登録。A1精度改善とA2提供範囲拡大を別判定。棋力改善とは別 |
| [#286](https://github.com/lisbun/lisjong/issues/286) 門前非リーチ者の待ち推定 | DEFERRED | 損失集中の証拠または予算配分の判断後、正例supportを得るpopulationを設計。未提供はNone |
| [#297](https://github.com/lisbun/lisjong/issues/297) 決定的な候補生成 | open。探索対象・候補列生成は未完了 | 次仮説・候補identity・適合する評価契約が必要。汎用探索の完成を強化の前提にしない |
| [#368](https://github.com/lisbun/lisjong/issues/368) L2a対Arm Y | 比較・昇格とも完了 | 後続のOverallはArena #583。段階2の外部棋譜追加学習は別Issue・未開始 |

## 完了した実験の扱い

| 系列 | 固定した結論 | 継続境界 |
| --- | --- | --- |
| L2a模倣 / CALL特徴追加 | #318: 打牌・Chi NOT ESTABLISHED、Pon PARTIAL。#326: NOT-SUPPORTED-DEVELOPMENT | family昇格で模倣結果を変更しない。OFFLINE-EVAL再利用なし |
| R1価値学習 | Arena #550: NOT-PREDICTIVE | 今回実験終了、系列は再開条件付き保留。具体的新仮説なしに作業を割り当てない |
| P1価値学習 | Arena #558: NOT-PREDICTIVE | 今回実験終了、E4へ進まない。系列全体の棄却ではない |
| paijia v1 / genbutsu v1 / call-value v1 / H5 | Arena #522: NOT-RECONFIRMED、#526/#536: STOP-LOW-ACTIVATION、#542: STOP-NO-SIGNAL | 既定候補へ戻さず、追加seed・条件緩和で救済しない |

旧正式実行#573のseedはRETIRED、#582はCOMMITTED。使用済みTESTや開発seedを独立Overall TESTへ流用しない。
新規対局・学習・seed予約・課金・稼働bot切替・自律loop開始は、この文書整理の範囲外。
