"""通常ツモDLVAT型補正のV1・p・単位補正（Issue #391、設計 #390）。

lisjongが所有する「価値と補正のcanonical semantics」を、MLに依存しない純粋関数として
提供する。入力はArenaの`pre_draw`から射影したallowlistの値だけで、lisjongは
Arena・engineをimportしない。κの推定・適用、行のjoin・集約はArena側の責務である。

```text
Arena: pre_draw ──射影──> DrawProbabilityProjection（p用） ──> draw_probabilities()
                     └──> ShantenValueProjection（V用）   ──> shanten_values(a)
                                         │
             unit_draw_corrections(p用, V用, 実現カテゴリ)
               1. supportとp（Fraction）
               2. support全体でV1を計算
               3. そのあとで実現カテゴリをlookup
               4. c_s = V_s(実現) − Σ_a p(a)·V_s(a)   （κ = 1、4席分）
```

詳細な契約・単位・符号・#390の仮定は`docs/draw-correction.md`を正本とする。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from fractions import Fraction

from lisjong.belief.canonical_axes import tile_type_from_index
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

__all__ = [
    "CATEGORY_COUNT",
    "RED_FIVE_CATEGORIES",
    "STANDARD_CATEGORY_STOCK",
    "DrawProbabilityProjection",
    "ShantenValueProjection",
    "draw_probabilities",
    "shanten_values",
    "unit_corrections_from_values",
    "unit_draw_corrections",
    "v1_definition",
]

SEAT_COUNT = 4

CATEGORY_COUNT = 37
"""37カテゴリ: 0..33 = 34牌種（赤でない牌）、34 / 35 / 36 = 赤5m / 赤5p / 赤5s。"""

RED_FIVE_CATEGORIES: Mapping[int, int] = {34: 4, 35: 13, 36: 22}
"""赤五のカテゴリ → 対応する通常の五のカテゴリ（34牌種index）。"""

STANDARD_CATEGORY_STOCK: tuple[int, ...] = tuple(
    1
    if index in RED_FIVE_CATEGORIES
    else 3
    if index in RED_FIVE_CATEGORIES.values()
    else 4
    for index in range(CATEGORY_COUNT)
)
"""37カテゴリの標準在庫。赤五 = 1、対応する通常の五 = 3、それ以外 = 4。"""

_TOTAL_TILES = 136
_DEAD_WALL_TILES = 14
_MAX_REVEALED_INDICATORS = 5
_PRE_DRAW_HAND_SIZES = frozenset({1, 4, 7, 10, 13})
"""ツモ直前の純手牌の枚数（確定面子4..0個）。評価は+1枚の2 / 5 / 8 / 11 / 14枚。"""

_RED_FIVE_SUITS = {
    34: TileCategory.MANZU,
    35: TileCategory.PINZU,
    36: TileCategory.SOUZU,
}

_V1_DEFINITION = (
    ("name", "lisjong.draw_correction"),
    ("version", 1),
    ("categories", "37: tile_type_0_33 + red_five_m_p_s"),
    ("probability", "unseen_category_count_over_unseen_total"),
    ("value", "drawer_shanten_plus_g_others_minus_g_over_3"),
    ("shanten", "lisjong.hand_evaluation.calculate_shanten"),
    ("unit", "kappa_1_rank_direction"),
    ("kappa", "not_applied"),
)


def v1_definition() -> dict[str, object]:
    """V1の特徴定義（identity）をJSON-nativeな新しいdictで返す。

    κの凍結artifactへ記録するためのもの。内容を変えるときはversionを上げる。
    """
    return dict(_V1_DEFINITION)


# ---------------------------------------------------------------------------
# 入力検査（fail closed）
# ---------------------------------------------------------------------------


def _int_value(value: object, context: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an int")
    return value


def _category(value: object, context: str) -> int:
    category = _int_value(value, context)
    if not 0 <= category < CATEGORY_COUNT:
        raise ValueError(f"{context} must be a category 0..{CATEGORY_COUNT - 1}")
    return category


def _category_counts(value: object, context: str) -> tuple[int, ...]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{context} must be a list or tuple of {CATEGORY_COUNT} ints")
    if len(value) != CATEGORY_COUNT:
        raise ValueError(f"{context} must have {CATEGORY_COUNT} categories")
    counts = tuple(
        _int_value(count, f"{context}[{i}]") for i, count in enumerate(value)
    )
    for index, count in enumerate(counts):
        if not 0 <= count <= STANDARD_CATEGORY_STOCK[index]:
            raise ValueError(
                f"{context}[{index}] must be between 0 and "
                f"{STANDARD_CATEGORY_STOCK[index]}"
            )
    return counts


def _seat_category_counts(value: object, context: str) -> tuple[tuple[int, ...], ...]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{context} must be a list or tuple of {SEAT_COUNT} seats")
    if len(value) != SEAT_COUNT:
        raise ValueError(f"{context} must have {SEAT_COUNT} seats")
    return tuple(
        _category_counts(counts, f"{context}[{seat}]")
        for seat, counts in enumerate(value)
    )


def _pre_draw_hand(counts: tuple[int, ...], context: str) -> None:
    if sum(counts) not in _PRE_DRAW_HAND_SIZES:
        raise ValueError(
            f"{context} must hold {sorted(_PRE_DRAW_HAND_SIZES)} tiles before the draw, "
            f"got {sum(counts)}"
        )


def _from_mapping(cls, value: object):
    if type(value) is not dict:
        raise TypeError(f"{cls.__name__} input must be a dict")
    expected = {field.name for field in fields(cls)}
    if set(value) != expected:
        unknown = sorted(str(key) for key in set(value) - expected)
        missing = sorted(expected - set(value))
        raise ValueError(
            f"{cls.__name__} fields are invalid (unknown={unknown}, missing={missing})"
        )
    return cls(**value)


# ---------------------------------------------------------------------------
# allowlistの射影型
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DrawProbabilityProjection:
    """p用の射影。ツモ直前の全席の既知牌を37カテゴリの枚数で持つ。

    各fieldは席順（0..3）の37カテゴリ枚数。`uncalled_discard_counts`は鳴かれて
    いない河の牌だけ（鳴かれた牌は鳴いた席の`meld_counts`に入る）。identity・
    実現牌・post digest・他の行は持たない（fieldにないkeyは`from_mapping`が拒否する）。
    """

    hand_counts: tuple[tuple[int, ...], ...]
    meld_counts: tuple[tuple[int, ...], ...]
    uncalled_discard_counts: tuple[tuple[int, ...], ...]
    revealed_dora_indicator_counts: tuple[int, ...]
    live_tiles_remaining: int

    def __post_init__(self) -> None:
        hands = _seat_category_counts(self.hand_counts, "hand_counts")
        melds = _seat_category_counts(self.meld_counts, "meld_counts")
        discards = _seat_category_counts(
            self.uncalled_discard_counts, "uncalled_discard_counts"
        )
        indicators = _category_counts(
            self.revealed_dora_indicator_counts, "revealed_dora_indicator_counts"
        )
        live = _int_value(self.live_tiles_remaining, "live_tiles_remaining")
        for seat, counts in enumerate(hands):
            _pre_draw_hand(counts, f"hand_counts[{seat}]")
        if live < 1:
            raise ValueError(
                "live_tiles_remaining must be >= 1 before a live-wall draw"
            )
        indicator_count = sum(indicators)
        if not 1 <= indicator_count <= _MAX_REVEALED_INDICATORS:
            raise ValueError(
                "revealed_dora_indicator_counts must hold "
                f"1..{_MAX_REVEALED_INDICATORS} indicators"
            )

        known = [0] * CATEGORY_COUNT
        for group in (*hands, *melds, *discards, indicators):
            for index, count in enumerate(group):
                known[index] += count
        for index, count in enumerate(known):
            if count > STANDARD_CATEGORY_STOCK[index]:
                raise ValueError(
                    f"known tiles of category {index} exceed the standard stock "
                    f"({count} > {STANDARD_CATEGORY_STOCK[index]})"
                )
        unseen_total = _TOTAL_TILES - sum(known)
        expected_total = live + _DEAD_WALL_TILES - indicator_count
        if unseen_total != expected_total:
            raise ValueError(
                "unseen tile total does not match live + 14 - revealed indicators "
                f"({unseen_total} != {expected_total})"
            )

        object.__setattr__(self, "hand_counts", hands)
        object.__setattr__(self, "meld_counts", melds)
        object.__setattr__(self, "uncalled_discard_counts", discards)
        object.__setattr__(self, "revealed_dora_indicator_counts", indicators)

    @classmethod
    def from_mapping(cls, value: object) -> "DrawProbabilityProjection":
        """JSON-nativeなdictから作る。fieldと完全に一致するkey集合だけを受け取る。"""
        return _from_mapping(cls, value)

    def unseen_counts(self) -> tuple[int, ...]:
        """37カテゴリの未見枚数 n_a（標準在庫 − 既知枚数）。"""
        known = [0] * CATEGORY_COUNT
        for group in (
            *self.hand_counts,
            *self.meld_counts,
            *self.uncalled_discard_counts,
            self.revealed_dora_indicator_counts,
        ):
            for index, count in enumerate(group):
                known[index] += count
        return tuple(
            STANDARD_CATEGORY_STOCK[index] - known[index]
            for index in range(CATEGORY_COUNT)
        )


@dataclass(frozen=True, slots=True)
class ShantenValueProjection:
    """V用の射影。drawer席とdrawerのツモ直前の純手牌（37カテゴリ枚数）だけを持つ。"""

    drawer_seat: int
    drawer_hand_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        seat = _int_value(self.drawer_seat, "drawer_seat")
        if not 0 <= seat < SEAT_COUNT:
            raise ValueError("drawer_seat must be a seat index 0..3")
        counts = _category_counts(self.drawer_hand_counts, "drawer_hand_counts")
        _pre_draw_hand(counts, "drawer_hand_counts")
        object.__setattr__(self, "drawer_hand_counts", counts)

    @classmethod
    def from_mapping(cls, value: object) -> "ShantenValueProjection":
        """JSON-nativeなdictから作る。fieldと完全に一致するkey集合だけを受け取る。"""
        return _from_mapping(cls, value)


# ---------------------------------------------------------------------------
# p・V1・単位補正
# ---------------------------------------------------------------------------


def draw_probabilities(projection: DrawProbabilityProjection) -> dict[int, Fraction]:
    """support（n_a > 0のカテゴリ）とp(a) = n_a / Nを、カテゴリ昇順で返す。"""
    if not isinstance(projection, DrawProbabilityProjection):
        raise TypeError("projection must be a DrawProbabilityProjection")
    unseen = projection.unseen_counts()
    total = sum(unseen)
    return {
        category: Fraction(count, total)
        for category, count in enumerate(unseen)
        if count > 0
    }


def _category_tile(category: int) -> Tile:
    if category in _RED_FIVE_SUITS:
        return Tile(TileType(_RED_FIVE_SUITS[category], 5), is_red=True)
    return Tile(tile_type_from_index(category))


def shanten_values(
    projection: ShantenValueProjection, category: int
) -> tuple[Fraction, ...]:
    """候補カテゴリaに対するV1の4席分の値（κ = 1の単位値）を返す。

    g(a) = drawerの純手牌 + aの向聴数（和了形は−1）。drawerは+g、他の3席は−g/3。
    """
    if not isinstance(projection, ShantenValueProjection):
        raise TypeError("projection must be a ShantenValueProjection")
    candidate = _category(category, "category")
    counts = list(projection.drawer_hand_counts)
    counts[candidate] += 1
    if counts[candidate] > STANDARD_CATEGORY_STOCK[candidate]:
        raise ValueError(f"category {candidate} exceeds the standard stock in the hand")
    tiles = [
        _category_tile(index)
        for index, count in enumerate(counts)
        for _ in range(count)
    ]

    from lisjong.hand_evaluation import calculate_shanten

    g = Fraction(calculate_shanten(tiles))
    return tuple(
        g if seat == projection.drawer_seat else -g / 3 for seat in range(SEAT_COUNT)
    )


def _exact(value: object, context: str) -> Fraction:
    if type(value) is int:
        return Fraction(value)
    if type(value) is Fraction:
        return value
    raise TypeError(f"{context} must be an int or Fraction")


def unit_corrections_from_values(
    probabilities: Mapping[int, Fraction],
    values: Mapping[int, Sequence[Fraction]],
    realized_category: int,
) -> tuple[Fraction, ...]:
    """c_s = V_s(実現) − Σ_a p(a)·V_s(a)を4席分、Fractionで返す（κは掛けない）。

    `values`は`probabilities`のsupportと同じkey集合で、support全体を事前に
    計算したものを渡す。実現カテゴリはそのあとでlookupするだけで、supportに
    なければfail closedする。
    """
    if not isinstance(probabilities, Mapping) or not isinstance(values, Mapping):
        raise TypeError("probabilities and values must be mappings")
    if not probabilities:
        raise ValueError("probabilities must have a non-empty support")
    support = {}
    for category, probability in probabilities.items():
        index = _category(category, "probability category")
        exact = _exact(probability, f"p({index})")
        if exact <= 0:
            raise ValueError(f"p({index}) must be positive on the support")
        support[index] = exact
    if sum(support.values()) != 1:
        raise ValueError("probabilities must sum to exactly 1")
    if set(values) != set(support):
        raise ValueError("values must be given for exactly the support categories")
    seat_values = {}
    for category in support:
        row = values[category]
        if type(row) not in (list, tuple) or len(row) != SEAT_COUNT:
            raise ValueError(f"values[{category}] must hold {SEAT_COUNT} seat values")
        seat_values[category] = tuple(
            _exact(value, f"values[{category}][{seat}]")
            for seat, value in enumerate(row)
        )

    realized = _category(realized_category, "realized_category")
    if realized not in support:
        raise ValueError(f"realized category {realized} is outside the support")
    return tuple(
        seat_values[realized][seat]
        - sum(support[a] * seat_values[a][seat] for a in support)
        for seat in range(SEAT_COUNT)
    )


def unit_draw_corrections(
    probability_projection: DrawProbabilityProjection,
    value_projection: ShantenValueProjection,
    realized_category: int,
) -> tuple[Fraction, ...]:
    """V1による1回の通常ツモの単位補正（κ = 1）を4席分返す。

    supportのVを全部計算してから実現カテゴリをlookupする。V用射影の純手牌が
    p用射影のdrawerの手牌と一致しなければfail closedする。
    """
    if not isinstance(probability_projection, DrawProbabilityProjection):
        raise TypeError("probability_projection must be a DrawProbabilityProjection")
    if not isinstance(value_projection, ShantenValueProjection):
        raise TypeError("value_projection must be a ShantenValueProjection")
    drawer = value_projection.drawer_seat
    if (
        value_projection.drawer_hand_counts
        != probability_projection.hand_counts[drawer]
    ):
        raise ValueError("drawer hand differs between the p and V projections")
    probabilities = draw_probabilities(probability_projection)
    values = {
        category: shanten_values(value_projection, category)
        for category in probabilities
    }
    return unit_corrections_from_values(probabilities, values, realized_category)
