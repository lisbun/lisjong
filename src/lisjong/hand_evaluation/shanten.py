"""向聴数計算の公開契約。

Issue #50で確定した、外部環境に依存しない牌姿評価の入口である。入力はlisjong
内部の`Tile`列だけで、RiichiEnv / RiichiLab / mjai等の外部型を受け取らない。

このmoduleは入力のsnapshot、validation、34牌種countへの正規化、確定面子数の
判断までを担当し、実際の探索は`_python_shanten`へ委譲する。

入口は2つあり、責務が異なる。

```text
一般consumer (Tile列)
    ↓
calculate_shanten(Tile...)        # 一般公開API・安全な境界
    ↓  snapshot / type validation / hand-size validation
    ↓  Tile -> 34牌種count正規化 / max 4枚validation
    ↓
_shanten_from_valid_counts()      # 唯一のsemantic core
    ↑
calculate_shanten_from_canonical_counts()
                                  # package-internal hot path
    ↑                             # trusted canonical-count precondition
lisjong内部consumer (すでに34牌種countを保持している側)
```

`calculate_shanten_from_canonical_counts()`はIssue #113で追加した
lisjong内部向けのperformance contractであり、package rootの`__all__`へは
追加しない一般非公開の入口である。新しい向聴semanticを定義するものでも、
private backendを公開するものでもない。standard / 七対子 / 国士無双 /
確定面子数のdispatchは`_shanten_from_valid_counts()`だけが持ち、入口ごとに
複製しない。

34牌種countはprivateな内部表現であり、一般公開APIにはしない。
"""

from collections.abc import Iterable, Sequence

from lisjong.hand_evaluation import _python_shanten
from lisjong.policy_contract.tile import Tile, TileCategory

_CATEGORY_OFFSETS = {
    TileCategory.MANZU: 0,
    TileCategory.PINZU: 9,
    TileCategory.SOUZU: 18,
    TileCategory.HONOR: 27,
}
"""34牌種canonical representationのindex起点。

    0..8    manzu 1..9
    9..17   pinzu 1..9
    18..26  souzu 1..9
    27..33  honor 1..7
"""

_MAX_COPIES_PER_TILE_KIND = 4
"""赤5と通常5を合算した基礎牌種1種あたりの上限枚数。"""

_VALID_CONCEALED_TILE_COUNTS = frozenset({1, 2, 4, 5, 7, 8, 10, 11, 13, 14})
"""確定面子0..4個に対応する、有効な純手牌枚数。"""

_MELDLESS_TILE_COUNTS = frozenset({13, 14})
"""確定面子が0で、七対子・国士無双を候補にできる純手牌枚数。"""


def calculate_shanten(tiles: Iterable[Tile]) -> int:
    """純手牌（concealed tiles）から向聴数を返す。

    返り値は一般的な定義に従う。

        和了形: -1
        聴牌:    0
        一向聴:  1
        以下同様

    `tiles`には、副露・槓で既に確定したmeldの牌を含めない。確定面子数は純手牌
    枚数から判断するため、Chi / Pon / Daiminkan / Ankan / Kakanの具体的な
    identityや`PublicMeld`をこのAPIへ渡す必要はない。

        13 / 14枚 -> 確定面子 0
        10 / 11枚 -> 確定面子 1
         7 /  8枚 -> 確定面子 2
         4 /  5枚 -> 確定面子 3
         1 /  2枚 -> 確定面子 4

    `OwnHandState.drawn_tile`は`concealed_tiles`に含まれる既存契約上のmetadata
    であるため、追加の1枚として渡さない。

    確定面子が0の13 / 14枚の場合だけ、通常形・七対子・国士無双を比較して最小の
    向聴数を返す。それ以外の有効な純手牌枚数（11枚以下）は、既に確定面子が
    ある状態なので通常形だけを評価する。

    赤5と通常5は牌姿構造上同じ牌種として扱う。`Tile`自体のred distinctionは
    変更しない。

    向聴計算と無関係なphysical tile conservation（実牌セット上の赤5枚数等）は
    検証しない。

    Raises:
        TypeError: `tiles`がiterableでない、または`Tile`以外を含む場合。
        ValueError: 純手牌枚数が有効な集合に含まれない場合、または赤5と通常5を
            合算した基礎牌種が5枚以上ある場合。
    """
    snapshot = _snapshot_tiles(tiles)

    if len(snapshot) not in _VALID_CONCEALED_TILE_COUNTS:
        raise ValueError(
            "tiles must contain a concealed hand size of "
            f"{sorted(_VALID_CONCEALED_TILE_COUNTS)}, got {len(snapshot)}"
        )

    counts = _count_tile_kinds(snapshot)
    return _shanten_from_valid_counts(counts, len(snapshot))


def calculate_shanten_from_canonical_counts(counts: Sequence[int]) -> int:
    """canonical 34牌種countから向聴数を返す、lisjong内部のhot path。

    これは`lisjong.hand_evaluation`が所有するpackage-internal performance
    contractであり、**一般公開APIではない**。package rootの`__all__`へは
    追加せず、すでにcanonical 34牌種countを保持しているlisjong内部consumer
    （現状は`FiniteHorizonCompletionPolicy`のDP）だけが利用する。

    向聴数のsemanticsは`calculate_shanten()`とまったく同じである。両者は同じ
    semantic core（`_shanten_from_valid_counts()`）へ委譲し、standard /
    七対子 / 国士無双 / 確定面子数のdispatchを二重実装しない。新しい向聴
    semanticでも、private backendのexposureでもない。

    `calculate_shanten()`と異なり、この入口はTile boundaryのvalidationを
    **再実行しない**。呼び出し側が次を保証すること（precondition）。

        len(counts) == 34
        canonical axis（0..8 manzu / 9..17 pinzu / 18..26 souzu /
        27..33 honor、赤5は通常5と同じindexへ集約済み）
        すべてのcountが0以上4以下
        sum(counts)が有効な純手牌枚数

    FiniteHorizonのDP stateはvalidated root handから構築され、future drawの
    +1とhypothetical discardの-1だけで遷移するため、この前提はcaller側の
    構築で保たれる。1回のdecisionで10万回規模のshanten評価が走るhot pathで
    あり、ここでTile相当のfull validationを繰り返すとcount-native化の意味が
    失われる。preconditionはこのdocstringとtestsで固定し、
    `use_validation=False`のようなruntime flagは持たせない。
    """
    return _shanten_from_valid_counts(counts, sum(counts))


def _shanten_from_valid_counts(counts: Sequence[int], concealed_tile_count: int) -> int:
    """検証済み34牌種countから向聴数を求める、唯一のsemantic coreである。

    Tile入口（`calculate_shanten()`）とcount-native入口
    （`calculate_shanten_from_canonical_counts()`）はどちらもここへ委譲する。
    standard / 七対子 / 国士無双の比較と確定面子数の解釈をこの1箇所だけに
    置き、入口ごとに複製しない。

    `concealed_tile_count`は`sum(counts)`と一致する純手牌枚数である。Tile側は
    すでに数え終えた`len(snapshot)`をそのまま渡せるため、この値を引数で受け
    取って再計算を避ける。
    """
    fixed_meld_count = _fixed_meld_count(concealed_tile_count)

    shanten = _python_shanten.calculate_standard_shanten(counts, fixed_meld_count)
    if concealed_tile_count in _MELDLESS_TILE_COUNTS:
        shanten = min(
            shanten,
            _python_shanten.calculate_seven_pairs_shanten(counts),
            _python_shanten.calculate_thirteen_orphans_shanten(counts),
        )
    return shanten


def _snapshot_tiles(tiles: Iterable[Tile]) -> tuple[Tile, ...]:
    """入力をその場でtupleへ固定し、要素型を検証する。

    後段の判断がiterableの再走査や遅延評価に依存しないよう、最初に1回だけ
    消費してsnapshotを取る。
    """
    try:
        snapshot = tuple(tiles)
    except TypeError:
        raise TypeError("tiles must be an iterable of Tile") from None
    if any(not isinstance(tile, Tile) for tile in snapshot):
        raise TypeError("tiles must contain only Tile instances")
    return snapshot


def _count_tile_kinds(tiles: tuple[Tile, ...]) -> list[int]:
    """`Tile`列を34牌種countへ正規化する。

    赤5と通常5は同じindexへ加算する。入力順序に依存しない。
    """
    counts = [0] * _python_shanten.TILE_KIND_COUNT
    for tile in tiles:
        tile_type = tile.tile_type
        counts[_CATEGORY_OFFSETS[tile_type.category] + tile_type.rank - 1] += 1

    if any(count > _MAX_COPIES_PER_TILE_KIND for count in counts):
        raise ValueError(
            "tiles must not contain more than "
            f"{_MAX_COPIES_PER_TILE_KIND} copies of the same base tile kind"
        )
    return counts


def _fixed_meld_count(concealed_tile_count: int) -> int:
    """純手牌枚数から、副露・槓で確定済みの面子数を求める。"""
    return 4 - (concealed_tile_count - 1) // 3
