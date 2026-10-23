"""向聴numeric coreの仕様由来fixture（Issue #332）。

各fixtureは入力（test記法）、固定期待値、期待値の導出理由を持つtest-onlyデータである。
期待値は向聴数の定義から手計算で導いたもので、Python lookup / 再帰DFS / Rust /
外部libraryの実行結果から生成していない。導出では次の記法を使う。

    通常形: 8 - 2 * (面子数 + 確定面子数) - 塔子数 - 雀頭(0/1)
            面子と塔子の合計は4 - 確定面子数まで。保有4枚の牌種を待ちに
            数える形（5枚目を必要とする形）は採用しない
    七対子: 6 - 対子種類数 + max(0, 7 - 牌種数)。4枚持ちは1対子（13 / 14枚のみ）
    国士:   13 - 么九牌種類数 - 么九牌対子の有無(0/1)（13 / 14枚のみ）

既存`tests/test_shanten.py`のcomment付き手計算fixtureを再利用・整理したものを
含む。理由が導けないものは採用していない。

`tags`はcoverage確認用の分類で、fixtureの値ではなく何を確認する入力かを表す。
"""

from dataclasses import dataclass

STANDARD = "standard"
SEVEN_PAIRS = "seven_pairs"
THIRTEEN_ORPHANS = "thirteen_orphans"
QUAD_IS_ONE_PAIR = "quad_is_one_pair"
FIFTH_COPY_EXCLUDED = "fifth_copy_excluded"
SPECIAL_FORMS_IGNORED = "special_forms_ignored_with_fixed_melds"
RED_FIVE = "red_five"


@dataclass(frozen=True)
class ShantenSpecFixture:
    """入力・固定期待値・導出理由の組。`notation`の`0`は赤5。"""

    name: str
    notation: str
    expected: int
    rationale: str
    tags: frozenset[str]


def _fixture(name, notation, expected, rationale, *tags) -> ShantenSpecFixture:
    return ShantenSpecFixture(name, notation, expected, rationale, frozenset(tags))


SHANTEN_SPEC_FIXTURES: tuple[ShantenSpecFixture, ...] = (
    # --- 確定面子0（14 / 13枚）: 通常形 ---
    _fixture(
        "complete_four_sequences",
        "123456789m123p11s",
        -1,
        "123m 456m 789m 123p の4面子 + 11s雀頭。",
        STANDARD,
    ),
    _fixture(
        "complete_triplets_with_honor_pair",
        "111333m444p777s22z",
        -1,
        "刻子4つ + 22z雀頭。",
        STANDARD,
    ),
    _fixture(
        "complete_overlapping_triplets_or_sequences",
        "111222333m44m567p",
        -1,
        "111m222m333m（または123m×3）+ 567p + 44m雀頭。",
        STANDARD,
    ),
    _fixture(
        "tenpai_single_wait",
        "123456789m123p1s",
        0,
        "4面子 + 1s単騎。8 - 8 = 0。",
        STANDARD,
    ),
    _fixture(
        "tenpai_shanpon_overlapping_sequences",
        "22334455m678p99s",
        0,
        "234m 234m 678p の3面子 + 55m / 99s の対子2つ。8 - 6 - 1 - 1 = 0。",
        STANDARD,
    ),
    _fixture(
        "tenpai_fourteen_tiles_kanchan",
        "123456789m11p24s7z",
        0,
        "3面子 + 11p雀頭 + 24s嵌張、7zは浮き牌。8 - 6 - 1 - 1 = 0。",
        STANDARD,
    ),
    _fixture(
        "one_shanten_five_blocks_without_pair",
        "123456789m12p13s5z",
        1,
        "3面子 + 12p / 13s塔子。ブロック上限4のため塔子1つだけ数え、"
        "雀頭なし。8 - 6 - 1 = 1。",
        STANDARD,
    ),
    _fixture(
        "two_shanten_isolated_tiles",
        "123456789m1p4p7p1s",
        2,
        "3面子 + 間隔3の孤立牌4枚（塔子・対子なし）。8 - 6 = 2。",
        STANDARD,
    ),
    _fixture(
        "dead_honor_spares_cannot_seed_blocks",
        "111122223333z1m",
        3,
        "111z 222z 333z の3面子。残る1z / 2z / 3zは4枚目で、雀頭・塔子に"
        "育てる5枚目がない。生きている1mから面子と雀頭を作るには"
        "2m・3m・単騎用の1枚・和了牌の4枚が要る。",
        STANDARD,
        FIFTH_COPY_EXCLUDED,
    ),
    # --- 確定面子0: 七対子 ---
    _fixture(
        "seven_pairs_complete",
        "11m22m33p44s55z66z77z",
        -1,
        "七対子: 対子7種、牌種7。6 - 7 = -1。",
        SEVEN_PAIRS,
    ),
    _fixture(
        "seven_pairs_tenpai",
        "11m22m33p44s55z66z7z",
        0,
        "七対子: 対子6種、牌種7。6 - 6 = 0。",
        SEVEN_PAIRS,
    ),
    _fixture(
        "seven_pairs_over_standard",
        "1199m1199p1199s1z",
        0,
        "七対子: 対子6種、牌種7で0。通常形は対子ブロック4 + 雀頭で3。",
        SEVEN_PAIRS,
    ),
    _fixture(
        "seven_pairs_kind_count_correction",
        "111m22m33p44s55z66z",
        1,
        "七対子: 対子6種だが牌種6なので6 - 6 + 1 = 1。"
        "通常形は111m + 対子ブロック3 + 雀頭で2。",
        SEVEN_PAIRS,
    ),
    _fixture(
        "seven_pairs_quad_counts_as_one_pair",
        "1111m22m33p44s55z66z",
        1,
        "1111mは1対子。対子6種・牌種6で6 - 6 + 1 = 1（2対子と数えると0）。"
        "通常形は111m + 対子ブロック3 + 雀頭で2、国士は9。",
        SEVEN_PAIRS,
        QUAD_IS_ONE_PAIR,
    ),
    # --- 確定面子0: 国士無双 ---
    _fixture(
        "thirteen_orphans_complete",
        "19m19p19s12345677z",
        -1,
        "国士: 么九13種 + 7z対子。13 - 13 - 1 = -1。",
        THIRTEEN_ORPHANS,
    ),
    _fixture(
        "thirteen_orphans_thirteen_sided_tenpai",
        "19m19p19s1234567z",
        0,
        "国士: 么九13種、対子なし。13 - 13 = 0。",
        THIRTEEN_ORPHANS,
    ),
    _fixture(
        "thirteen_orphans_twelve_kinds_with_pair",
        "19m19p19s1234556z",
        0,
        "国士: 么九12種 + 5z対子。13 - 12 - 1 = 0。",
        THIRTEEN_ORPHANS,
    ),
    _fixture(
        "thirteen_orphans_missing_kind",
        "19m19p19s123456z5m",
        1,
        "国士: 么九12種、対子なし。13 - 12 = 1。七対子は6、通常形はそれ以上。",
        THIRTEEN_ORPHANS,
    ),
    _fixture(
        "far_hand_best_is_seven_pairs",
        "147m258p369s1357z",
        6,
        "通常形: 間隔3の孤立牌だけで塔子・対子なし、8。七対子: 対子0・牌種13で6。"
        "国士: 么九は1m 9s 1z 3z 5z 7zの6種で7。最小は6。",
        STANDARD,
        SEVEN_PAIRS,
        THIRTEEN_ORPHANS,
    ),
    # --- 確定面子1（11 / 10枚） ---
    _fixture(
        "one_meld_complete",
        "123456789m11p",
        -1,
        "確定1 + 3面子 + 11p雀頭。",
        STANDARD,
    ),
    _fixture(
        "one_meld_tenpai",
        "123456789m1p",
        0,
        "確定1 + 3面子 + 1p単騎。8 - 8 = 0。",
        STANDARD,
    ),
    _fixture(
        "one_meld_one_shanten",
        "12356m89p11s4z",
        1,
        "確定1 + 123m + 56m / 89p塔子 + 11s雀頭、4z浮き。8 - 4 - 2 - 1 = 1。",
        STANDARD,
    ),
    _fixture(
        "one_meld_far_without_blocks",
        "147m147p147s1z",
        6,
        "確定1、間隔3の孤立牌だけで塔子・対子なし。8 - 2 = 6。",
        STANDARD,
    ),
    _fixture(
        "one_meld_orphan_shape_uses_standard_only",
        "19m19p19s1234z",
        6,
        "確定1。10枚なので国士・七対子は評価しない。塔子・対子なしで8 - 2 = 6。",
        STANDARD,
        SPECIAL_FORMS_IGNORED,
    ),
    # --- 確定面子2（8 / 7枚） ---
    _fixture(
        "two_melds_complete",
        "123456m11p",
        -1,
        "確定2 + 2面子 + 11p雀頭。",
        STANDARD,
    ),
    _fixture(
        "two_melds_tenpai",
        "123456m1p",
        0,
        "確定2 + 2面子 + 1p単騎。",
        STANDARD,
    ),
    _fixture(
        "two_melds_one_shanten",
        "12m56m11p4z",
        1,
        "確定2 + 12m / 56m塔子 + 11p雀頭、4z浮き。8 - 4 - 2 - 1 = 1。",
        STANDARD,
    ),
    _fixture(
        "two_melds_four_copies_cannot_wait_on_fifth",
        "1111p666z",
        1,
        "確定2 + 666z + 111p。残る1pの単騎は5枚目待ちで採用しない。"
        "11p雀頭 + 1p浮きでは8 - 6 - 1 = 1。",
        STANDARD,
        FIFTH_COPY_EXCLUDED,
    ),
    _fixture(
        "two_melds_quad_split_avoids_fifth_copy",
        "1111m222m",
        0,
        "確定2 + 111m + 12m塔子 + 22m雀頭（3m待ち）。8 - 6 - 1 - 1 = 0。"
        "111m 222m + 1m単騎は5枚目待ちで採用しない。",
        STANDARD,
        FIFTH_COPY_EXCLUDED,
    ),
    # --- 確定面子3（5 / 4枚） ---
    _fixture(
        "three_melds_complete",
        "123m11p",
        -1,
        "確定3 + 123m + 11p雀頭。",
        STANDARD,
    ),
    _fixture(
        "three_melds_tenpai",
        "123m1p",
        0,
        "確定3 + 123m + 1p単騎。",
        STANDARD,
    ),
    _fixture(
        "three_melds_one_shanten",
        "11m59p",
        1,
        "確定3 + 11m雀頭。5p / 9pは間隔4で塔子にならない。8 - 6 - 1 = 1。",
        STANDARD,
    ),
    _fixture(
        "three_melds_four_copies_cannot_be_pair_wait",
        "1111m",
        1,
        "確定3 + 111m + 1m単騎は5枚目待ちで採用しない。"
        "11m雀頭 + 11m対子でも同じ1mを待つため、残りは11m雀頭のみで8 - 6 - 1 = 1。",
        STANDARD,
        FIFTH_COPY_EXCLUDED,
    ),
    _fixture(
        "three_melds_shanpon",
        "1122m",
        0,
        "確定3 + 11m / 22m対子のシャンポン。8 - 6 - 1 - 1 = 0。",
        STANDARD,
    ),
    _fixture(
        "three_melds_red_five_tenpai",
        "340m1p",
        0,
        "確定3 + 34m赤5mの面子 + 1p単騎。赤5は通常5と同じ牌種。",
        STANDARD,
        RED_FIVE,
    ),
    # --- 確定面子4（2 / 1枚） ---
    _fixture(
        "four_melds_complete",
        "11p",
        -1,
        "確定4 + 11p雀頭。",
        STANDARD,
    ),
    _fixture(
        "four_melds_tenpai",
        "1p",
        0,
        "確定4 + 1p単騎。",
        STANDARD,
    ),
    _fixture(
        "four_melds_two_singles",
        "1m9p",
        0,
        "確定4。どちらかを捨てると単騎聴牌。8 - 8 = 0。",
        STANDARD,
    ),
    _fixture(
        "four_melds_red_five_pair",
        "05s",
        -1,
        "確定4 + 赤5sと5sの雀頭。",
        STANDARD,
        RED_FIVE,
    ),
)
"""仕様由来fixture。名前は一意。"""
