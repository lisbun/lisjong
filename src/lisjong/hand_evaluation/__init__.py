"""lisjong内部の牌姿評価package。

lisjongの`Tile`から派生的な評価値を計算する、外部環境に依存しない層である。
RiichiEnv、RiichiLab、mjai、WebSocketの型やprotocolへ依存せず、Policy実行、
合法手判定、対局進行も所有しない。

Issue #50時点の公開契約は`calculate_shanten()`だけであり、具体的な計算backend
はprivate moduleに隠す。

Issue #263で、仮定の完成手の役・符・点数を計算する
`lisjong.hand_evaluation.scoring`を追加した。計算coreはnative拡張にあり、
本packageのimportはnative拡張を必要としない。

Issue #346で向聴計算もnative拡張が必須になった。`calculate_shanten`は遅延公開
であり、最初に参照した時点で`shanten`（と`_shanten_backend`）をimportする。
native拡張がなければそこで`ShantenBackendError`になる。package自体、`scoring`、
artifact供給module、offline generatorのimportはnative拡張なしで成功する。
"""

__all__ = ["calculate_shanten"]


def __getattr__(name: str):
    if name == "calculate_shanten":
        from lisjong.hand_evaluation.shanten import calculate_shanten

        return calculate_shanten
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
