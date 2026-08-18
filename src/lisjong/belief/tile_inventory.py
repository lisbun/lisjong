"""標準4人麻雀のcanonical physical tile inventory。

Issue #63が固定した、牌保存則検証の基準となるphysical tile inventoryの
唯一の正本である。Issue #59の34基本牌種canonical axisで表現し、`34` /
`4` / `136`、赤5の`1`をconservation計算や#61のrange validationへ
moduleごとに独立したmagic numberとして散在させない。

標準4人麻雀を対象とし、arbitrary rule inventory engineは導入しない。
"""

TILE_TYPE_COUNT = 34
RED_FIVE_AXIS_COUNT = 3

STANDARD_TILE_COUNTS: tuple[int, ...] = (4,) * TILE_TYPE_COUNT
"""34基本牌種それぞれのphysical inventory。標準4人麻雀では各4枚存在する。
physical copy identityは導入しない。"""

STANDARD_RED_FIVE_COUNTS: tuple[int, ...] = (1, 1, 1)
"""赤5m / 赤5p / 赤5sのphysical inventory。対応する5牌4枚のsubsetであり、
追加牌ではない。canonical orderは#59と同じ0=5m, 1=5p, 2=5s。"""

TOTAL_PHYSICAL_TILE_COUNT = sum(STANDARD_TILE_COUNTS)  # 136

BASE_TILE_COUNT_MAX = max(STANDARD_TILE_COUNTS)  # 4
RED_FIVE_COUNT_MAX = max(STANDARD_RED_FIVE_COUNTS)  # 1

__all__ = [
    "BASE_TILE_COUNT_MAX",
    "RED_FIVE_AXIS_COUNT",
    "RED_FIVE_COUNT_MAX",
    "STANDARD_RED_FIVE_COUNTS",
    "STANDARD_TILE_COUNTS",
    "TILE_TYPE_COUNT",
    "TOTAL_PHYSICAL_TILE_COUNT",
]
