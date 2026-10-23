"""向聴計算の各経路が共有するprivateな定数と特殊形helper（Issue #338）。

`_python_shanten`（旧DFS oracle）、`_lookup_shanten`（Python lookup計算）、
`_shanten_table`（artifact供給）、`_shanten_frontier`（offline generator）、
`shanten.py`（numeric dispatch・structural predicate）が同じ値を参照する。
Python numeric計算本体を将来削除しても、structural predicateとgeneratorが
依存するhelper・定数がここに残るよう、計算本体から分離している。

このmoduleはpackage内の他moduleをimportしない。
"""

from collections.abc import Sequence

TILE_KIND_COUNT = 34
"""向聴計算内部で使う34牌種canonical representationの要素数。"""

SUITED_KIND_COUNT = 27
"""数牌（萬子・筒子・索子）が占めるindexの個数。0..26が数牌、27..33が字牌。"""

MAX_COPIES_PER_TILE_KIND = 4
"""1つの牌種が実在し得る最大枚数。和了形もこの枚数を超えられない。"""

MAX_BLOCK_COUNT = 4
"""通常形で雀頭を除いて必要な面子・塔子候補の総数。"""

STANDARD_SHANTEN_BASE = 8
"""面子・塔子・雀頭を1つも持たない場合の通常形向聴数。"""

SEED_COUNT_CAP = 5
"""探索stateとして保持する種牌数の上限。必要な種牌は最大5個（4面子 + 雀頭）。"""

TERMINAL_OR_HONOR_INDICES = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
"""国士無双の対象となる么九牌13種のindex。"""


def calculate_seven_pairs_shanten(counts: Sequence[int]) -> int:
    """七対子の向聴数を返す。

    同じ牌種の3枚目以降は別の対子として使えないため、対子は牌種ごとに最大1しか
    数えない。牌種数が7未満の場合は、対子にできる牌種を新たに引く必要があるぶん
    だけ補正する。
    """
    pair_count = sum(1 for count in counts if count >= 2)
    kind_count = sum(1 for count in counts if count >= 1)

    shanten = 6 - pair_count
    if kind_count < 7:
        shanten += 7 - kind_count
    return shanten


def calculate_thirteen_orphans_shanten(counts: Sequence[int]) -> int:
    """国士無双の向聴数を返す。

    么九牌の種類数と、么九牌の対子を1つ持つかどうかだけで決まる。
    """
    kind_count = sum(1 for index in TERMINAL_OR_HONOR_INDICES if counts[index] >= 1)
    has_pair = any(counts[index] >= 2 for index in TERMINAL_OR_HONOR_INDICES)
    return 13 - kind_count - (1 if has_pair else 0)
