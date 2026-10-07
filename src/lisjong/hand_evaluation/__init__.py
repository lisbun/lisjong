"""lisjong内部の牌姿評価package。

lisjongの`Tile`から派生的な評価値を計算する、外部環境に依存しない層である。
RiichiEnv、RiichiLab、mjai、WebSocketの型やprotocolへ依存せず、Policy実行、
合法手判定、対局進行も所有しない。

Issue #50時点の公開契約は`calculate_shanten()`だけであり、具体的な計算backend
はprivate moduleに隠す。将来backendをRust / C++ / lookup tableへ交換しても、
利用側の契約は変わらない。

Issue #263で、仮定の完成手の役・符・点数を計算する
`lisjong.hand_evaluation.scoring`を追加した。計算coreはopt-in native拡張にあり、
本packageのimportはnative拡張を必要としない。
"""

from lisjong.hand_evaluation.shanten import calculate_shanten

__all__ = ["calculate_shanten"]
