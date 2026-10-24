"""向聴数計算の公開契約。

Issue #50で確定した、外部環境に依存しない牌姿評価の入口である。入力はlisjong
内部の`Tile`列だけで、RiichiEnv / RiichiLab / mjai等の外部型を受け取らない。

このmoduleは入力のsnapshot、validation、34牌種countへの正規化、確定面子数の
判断までを担当し、numeric shanten（通常形・七対子・国士無双・確定面子数の
dispatch）と打牌候補の一括評価はnative拡張（`_shanten_backend`）で計算する
（Issue #346でPython計算実装を削除）。structural predicateの七対子・国士無双
判定と共有定数は`_shanten_common`を使う（Issue #338）。

入口は2つあり、責務が異なる。

```text
一般consumer (Tile列)
    ↓
calculate_shanten(Tile...)        # 一般公開API・安全な境界
    ↓  snapshot / type validation / hand-size validation
    ↓  Tile -> 34牌種count正規化 / max 4枚validation
    ↓
_shanten_from_valid_counts()      # numeric shantenの唯一のsemantic core
    ↑                             # （native: StandardShantenTable）
calculate_shanten_from_canonical_counts()
                                  # package-internal hot path
    ↑                             # trusted canonical-count precondition
lisjong内部consumer (すでに34牌種countを保持している側)
```

`calculate_shanten_from_canonical_counts()`はIssue #113で追加した
lisjong内部向けのperformance contractであり、package rootの`__all__`へは
追加しない一般非公開の入口である。新しい向聴semanticを定義するものでも、
private backendを公開するものでもない。numeric shantenのstandard / 七対子 /
国士無双 / 確定面子数のdispatchは`_shanten_from_valid_counts()`だけが持ち、
入口ごとに複製しない。Issue #224の`evaluate_discards_from_canonical_counts()`は
打牌（-1）と仮想ツモ（+1）の列挙を1回の呼び出しにまとめる同じ層の入口で、各向聴数は
同じsemantic coreで求める。package-internalのstructural completion / tenpai
predicateもこのmoduleが所有し、special-hand計算は同じhelperを共有する。

34牌種countはprivateな内部表現であり、一般公開APIにはしない。
"""

from collections.abc import Container, Iterable, Sequence

from lisjong.hand_evaluation import (
    _shanten_backend,
    _shanten_common,
    _structural_predicates,
)
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

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


def evaluate_discards_from_canonical_counts(
    counts: Sequence[int],
    discard_indexes: Iterable[int],
    improving_max_shanten: int | None = None,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...] | None, ...]]:
    """打牌候補の打牌後向聴数と改善牌を一括で求める、lisjong内部の入口（Issue #224）。

    `calculate_shanten_from_canonical_counts()`と同じくpackage-internalで、
    package rootの`__all__`へは追加しない。打牌（-1）と仮想ツモ（+1）の列挙を
    1回の呼び出しにまとめ、各向聴数は同じnumeric core
    （`_shanten_from_valid_counts()`）で求める。打牌戦略（候補の選別規則、
    受入集計、評価順）は持たない。実装はnative（`StandardShantenTable.evaluate_discards`、
    Issue #333でloopをRust coreへ分離）で、`_shanten_backend`だけが結び付ける。
    呼び出し側は`_lisjong_native`を直接importしない。

    入力::

        counts                 canonical 34牌種count（各0..4）。打牌できる純手牌
                               枚数（2 / 5 / 8 / 11 / 14）
        discard_indexes        評価する打牌牌種のcanonical index（int、boolは不可、
                               0..33、重複不可、countsに1枚以上あること）
        improving_max_shanten  Noneなら全候補の改善牌を求める。intなら打牌後向聴数が
                               この値以下の候補だけ改善牌を求める

    返り値は`(shanten_after, improving_after)`で、どちらも`discard_indexes`と
    同じ順序・同じ長さのtupleである（候補が空なら`((), ())`）。

    - `shanten_after[i]`: `discard_indexes[i]`を1枚捨てた後の向聴数
    - `improving_after[i]`: その打牌後に1枚引くと向聴数が下がる牌種の
      canonical index（昇順・重複なし）。打牌後の手中に4枚ある牌種は引けない
      ので除外し、残り枚数（未見枚数0を含む）は参照しない。評価済みの空集合は
      `()`、評価しなかった候補は`None`で、両者を区別する。

    入力は変更しない。不正入力は評価前に例外にする（lenや値域・手牌枚数・
    indexの範囲・重複・保有なし・threshold範囲は`ValueError`、indexやthresholdが
    int以外またはbool、countがint以外は`TypeError`）。
    """
    return _evaluate_discards(counts, discard_indexes, improving_max_shanten)


def calculate_restricted_standard_shanten(
    tiles: Iterable[Tile], usable_tile_types: Container[TileType]
) -> int:
    """`usable_tile_types`以外の牌を面子・塔子・雀頭に使えない牌として扱う通常形向聴数。

    lisjong内部のpackage-internal contractであり、package rootの`__all__`へは
    追加しない。副露後に特定の役route（タンヤオ・混一色等）だけで和了形を
    目指す場合の距離を表す。使えない牌は手牌枠を占めるだけの浮き牌として扱い、
    確定面子数は`calculate_shanten()`と同じく純手牌枚数全体から判断する。
    七対子・国士無双は評価しない。

    全牌種がusableの場合は`calculate_shanten()`の通常形向聴数と一致する。
    入力validationは`calculate_shanten()`と同じである。
    """
    snapshot = _snapshot_tiles(tiles)
    if len(snapshot) not in _VALID_CONCEALED_TILE_COUNTS:
        raise ValueError(
            "tiles must contain a concealed hand size of "
            f"{sorted(_VALID_CONCEALED_TILE_COUNTS)}, got {len(snapshot)}"
        )
    _count_tile_kinds(snapshot)
    usable_counts = _count_tile_kinds(
        tuple(tile for tile in snapshot if tile.tile_type in usable_tile_types)
    )
    return _shanten_backend.calculate_standard_shanten(
        usable_counts, _fixed_meld_count(len(snapshot))
    )


def is_structurally_complete_from_canonical_counts(counts: Sequence[int]) -> bool:
    """Return whether trusted canonical counts have structural shanten ``-1``.

    This is a package-internal, boolean performance contract for validated
    canonical 34-count callers.  Its preconditions are the same as
    ``calculate_shanten_from_canonical_counts()``, with supported concealed
    counts restricted to ``2 / 5 / 8 / 11 / 14``.  It is not a legal-win,
    yaku, furiten, scoring, or remaining-inventory predicate.

    Standard completion comes from the separate lazy compact predicate
    artifact.  Closed 14-tile hands additionally retain the same seven-pairs
    and thirteen-orphans semantics as canonical shanten.  Artifact failures
    propagate without a numeric or recursive fallback.
    """
    concealed_tile_count = sum(counts)
    fixed_meld_count = _fixed_meld_count(concealed_tile_count)
    if _structural_predicates.is_standard_structurally_complete(
        counts, fixed_meld_count
    ):
        return True
    if concealed_tile_count != 14:
        return False
    return _meldless_special_shanten(counts) == -1


def is_structurally_tenpai_from_canonical_counts(counts: Sequence[int]) -> bool:
    """Return whether trusted canonical counts have structural shanten ``0``.

    This package-internal hot-path contract supports concealed counts
    ``1 / 4 / 7 / 10 / 13`` and shares the trusted-input preconditions of the
    count-native shanten contract.  It ignores remaining inventory, so a dead
    wait is still structural tenpai.

    Standard one-added reachability comes from the separate lazy compact
    predicate artifact.  Closed 13-tile hands additionally retain the same
    seven-pairs and thirteen-orphans semantics as canonical shanten.
    """
    concealed_tile_count = sum(counts)
    fixed_meld_count = _fixed_meld_count(concealed_tile_count)
    if _structural_predicates.is_standard_structurally_tenpai(counts, fixed_meld_count):
        return True
    if concealed_tile_count != 13:
        return False
    return _meldless_special_shanten(counts) == 0


_shanten_from_valid_counts = _shanten_backend.native_shanten_from_valid_counts
"""検証済み34牌種countから向聴数を求めるnumeric semantic core（native）。

Tile入口（`calculate_shanten()`）とcount-native入口
（`calculate_shanten_from_canonical_counts()`）はどちらもここへ委譲する。
`(counts, concealed_tile_count)`を受け取り、`concealed_tile_count`は`sum(counts)`と
一致する純手牌枚数である。standard / 七対子 / 国士無双の比較（閉じた13 / 14枚だけ）
と確定面子数の解釈をこの1箇所だけに置き、入口ごとに複製しない。

Issue #115のlookup table、Issue #213のnative化を経て、Issue #346でPython実装を
削除した。期待値の正本は`tests/shanten_spec_fixtures.py`の仕様由来fixtureである。
"""

_evaluate_discards = _shanten_backend.native_evaluate_discards
"""打牌候補の一括構造評価（Issue #224 / #333）。"""


def _meldless_special_shanten(counts: Sequence[int]) -> int:
    """Return the shared closed-hand minimum for seven pairs / orphans."""
    return min(
        _shanten_common.calculate_seven_pairs_shanten(counts),
        _shanten_common.calculate_thirteen_orphans_shanten(counts),
    )


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
    counts = [0] * _shanten_common.TILE_KIND_COUNT
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
