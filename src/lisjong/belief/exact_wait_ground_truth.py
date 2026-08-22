"""完全情報手牌からcanonical structural wait ground truthを導出するbuilder。

Issue #84「完全情報手牌からcanonical structural wait ground truthを導出する」を
実装する。Issue #82が`HandBelief`へ追加したLevel 2 wait belief representation
（`wait_probability` + tanki / shanpon / kanchan / penchan / ryanmen low-side /
ryanmen high-side / kokushiのmechanism table群）へ、確率推定ではなく
0 / `SCALE`のみのbinary exact ground truthを機械的に生成する。

対象stateはstable **13-equivalent hand**である。既知のconcealed tilesと、
そのplayer自身の既知のmeld一式（chi / pon / open kan / added kan /
concealed kanを含む既存`PublicMeld`）について、

```text
len(concealed_tiles) + 3 * len(own_melds) == 13
```

を要求する。chi / pon / いずれの槓も、4面子1雀頭を構成するうえではすべて
completed 1 meld = structural 3-equivalentとして数える（Issue #84の
`Structural equivalent count`）。一方、5枚目candidate拒否等のphysical tile
constraintでは実際の物理枚数（chi / pon = 3枚、槓 = 4枚）を数える
（`Physical tile count`）。この2種類のcountを混同しない。

`HandBelief.expected_count` / `red_five_probability`はconcealed handのみの
marginalであり（`self_belief.concealed_hand_marginals()`を再利用する既存
`exact_self_belief()`と同じcontract）、own melds内の牌をここへ加算しない。
wait / wait mechanismだけが、concealed hand + own meldsを条件とした
derived hand-state beliefである。

standard hand completionの全decomposition列挙、chiitoitsu、kokushiの
判定はこのmodule専用のprivate exact-completion searchとして実装する
（Issue #84時点ではshanten計算専用の`_python_shanten._StandardFormSearch`を
mechanism列挙用へ拡張しない。best block scoreを求める探索と、全decomposition
列挙 + どのgroupを完成したかのmulti-label分類とは責務が異なるため）。
"""

from collections.abc import Iterable

from lisjong.belief.canonical_axes import red_five_index, tile_type_index
from lisjong.belief.fixed_point import SCALE
from lisjong.belief.hand_belief import HandBelief
from lisjong.belief.self_belief import concealed_hand_marginals
from lisjong.belief.tile_inventory import BASE_TILE_COUNT_MAX, STANDARD_RED_FIVE_COUNTS
from lisjong.policy_contract.meld import PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.tile import (
    Tile,
    TileCategory,
    TileType,
    _canonicalize_tile_multiset,
)

_TILE_KIND_COUNT = 34
_SUITED_KIND_COUNT = 27
_RANKS_PER_SUIT = 9
_STABLE_STRUCTURAL_EQUIVALENT_COUNT = 13
_STANDARD_MELD_SLOT_COUNT = 4

_SUITED_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)

_TERMINAL_AND_HONOR_INDICES = frozenset(
    tile_type_index(TileType(category, rank))
    for category in _SUITED_CATEGORIES
    for rank in (1, 9)
) | frozenset(
    tile_type_index(TileType(TileCategory.HONOR, rank)) for rank in range(1, 8)
)


def exact_hand_belief_with_waits(
    concealed_tiles: Iterable[Tile],
    own_melds: Iterable[PublicMeld] = (),
) -> HandBelief:
    """完全情報のconcealed tiles + own meldsから、Level 2 exact `HandBelief`を返す。

    `concealed_tiles` + `own_melds`はstable 13-equivalent hand
    （`len(concealed_tiles) + 3 * len(own_melds) == 13`）でなければならない。
    14-equivalentのdrawn stateなど、これを満たさない入力はfail-closedで
    `ValueError`にする（silentにnon-tenpaiへ変換しない）。

    concealed hand + own meldsの物理牌数が、`lisjong.belief.tile_inventory`が
    正本とするcanonical physical tile inventory（基本牌種ごとに4枚、色ごとの
    赤5は1枚）を超える場合もfail-closedで`ValueError`にする。赤5と通常5は
    physical count上同じ基本牌種として扱う。

    返り値は常に`has_wait_belief == True` かつ `has_wait_mechanism_belief ==
    True`のLevel 2 `HandBelief`であり、`None`にはしない（非聴牌はall-zero）。
    wait-related raw値はすべて`0`または`SCALE`のみであり、`wait_probability_raw`
    は常に7つのmechanism tableの論理和（existential OR）と一致する。
    """
    concealed_tiles = _canonicalize_tile_multiset(
        concealed_tiles, None, "concealed_tiles"
    )
    own_melds = _canonicalize_own_melds(own_melds)
    meld_count = len(own_melds)

    if len(concealed_tiles) + 3 * meld_count != _STABLE_STRUCTURAL_EQUIVALENT_COUNT:
        raise ValueError(
            "concealed_tiles and own_melds must form a stable 13-equivalent "
            "hand (len(concealed_tiles) + 3 * len(own_melds) == 13); this "
            "builder does not accept 14-equivalent drawn states"
        )

    concealed_counts = _tile_type_counts(concealed_tiles)
    meld_physical_counts = _tile_type_counts(
        tile for meld in own_melds for tile in meld.tiles
    )
    _validate_physical_inventory(
        concealed_tiles, own_melds, concealed_counts, meld_physical_counts
    )

    mechanisms = _WaitMechanismTables()
    melds_needed = _STANDARD_MELD_SLOT_COUNT - meld_count

    for candidate_index in range(_TILE_KIND_COUNT):
        total_physical_count = (
            concealed_counts[candidate_index] + meld_physical_counts[candidate_index]
        )
        if total_physical_count >= BASE_TILE_COUNT_MAX:
            continue

        candidate_counts = list(concealed_counts)
        candidate_counts[candidate_index] += 1

        for decomposition in _enumerate_standard_decompositions(
            tuple(candidate_counts), melds_needed
        ):
            for block_kind, block_index in decomposition:
                mechanisms.record_standard_block(
                    candidate_index, block_kind, block_index
                )

        if meld_count == 0:
            if _is_chiitoitsu_complete(candidate_counts):
                mechanisms.tanki[candidate_index] = True
            if _is_kokushi_complete(candidate_counts):
                mechanisms.kokushi[candidate_index] = True

    expected_count_raw, red_five_probability_raw = concealed_hand_marginals(
        concealed_tiles
    )

    return HandBelief(
        expected_count_raw=expected_count_raw,
        red_five_probability_raw=red_five_probability_raw,
        **mechanisms.to_raw_fields(),
    )


def exact_hand_belief_with_waits_for_own_hand_state(
    own_hand_state: OwnHandState,
    own_melds: Iterable[PublicMeld] = (),
) -> HandBelief:
    """自席の`OwnHandState`から、stable 13-equivalent handとしてLevel 2 exact
    `HandBelief`を返す。

    `own_hand_state.drawn_tile`が`None`でない場合は打牌前のdraw後stateであり、
    stable 13-equivalent contractと矛盾するため`ValueError`にする。このbuilder
    はdrawn stateを暗黙にdiscard後stateへ変換しない。呼び出し側が明示的に
    打牌後の`concealed_tiles`を用意して`exact_hand_belief_with_waits()`を
    直接呼ぶ必要がある。
    """
    if not isinstance(own_hand_state, OwnHandState):
        raise TypeError("own_hand_state must be an OwnHandState")
    if own_hand_state.drawn_tile is not None:
        raise ValueError(
            "own_hand_state must not have a drawn_tile; this builder requires "
            "a stable post-discard 13-equivalent state and does not implicitly "
            "convert a drawn state"
        )
    return exact_hand_belief_with_waits(own_hand_state.concealed_tiles, own_melds)


def _canonicalize_own_melds(own_melds: object) -> tuple[PublicMeld, ...]:
    try:
        melds = tuple(own_melds)
    except TypeError:
        raise TypeError("own_melds must be an iterable of PublicMeld") from None
    if any(not isinstance(meld, PublicMeld) for meld in melds):
        raise TypeError("own_melds must contain only PublicMeld instances")
    return melds


def _tile_type_counts(tiles: Iterable[Tile]) -> tuple[int, ...]:
    counts = [0] * _TILE_KIND_COUNT
    for tile in tiles:
        counts[tile_type_index(tile.tile_type)] += 1
    return tuple(counts)


def _validate_physical_inventory(
    concealed_tiles: tuple[Tile, ...],
    own_melds: tuple[PublicMeld, ...],
    concealed_counts: tuple[int, ...],
    meld_physical_counts: tuple[int, ...],
) -> None:
    for index in range(_TILE_KIND_COUNT):
        if concealed_counts[index] + meld_physical_counts[index] > BASE_TILE_COUNT_MAX:
            raise ValueError(
                "concealed_tiles and own_melds must not contain more than "
                f"{BASE_TILE_COUNT_MAX} physical copies of the same base tile "
                "kind"
            )

    red_five_counts = [0, 0, 0]
    for tile in concealed_tiles:
        if tile.is_red:
            red_five_counts[red_five_index(tile.tile_type.category)] += 1
    for meld in own_melds:
        for tile in meld.tiles:
            if tile.is_red:
                red_five_counts[red_five_index(tile.tile_type.category)] += 1

    for index, limit in enumerate(STANDARD_RED_FIVE_COUNTS):
        if red_five_counts[index] > limit:
            raise ValueError(
                "concealed_tiles and own_melds must not exceed the canonical "
                "red-five tile inventory"
            )


class _WaitMechanismTables:
    """34牌種canonical axisのmechanism flagを蓄積するmutable集計器。

    `exact_hand_belief_with_waits()`の1回の呼び出しに閉じたスコープでのみ
    使う。`wait`は常にmechanism群の論理和として`to_raw_fields()`が導出する
    ため、独立に設定するfieldを持たない。
    """

    def __init__(self) -> None:
        self.tanki = [False] * _TILE_KIND_COUNT
        self.shanpon = [False] * _TILE_KIND_COUNT
        self.kanchan = [False] * _TILE_KIND_COUNT
        self.penchan = [False] * _TILE_KIND_COUNT
        self.ryanmen_low_side = [False] * _TILE_KIND_COUNT
        self.ryanmen_high_side = [False] * _TILE_KIND_COUNT
        self.kokushi = [False] * _TILE_KIND_COUNT

    def record_standard_block(
        self, candidate_index: int, block_kind: str, block_index: int
    ) -> None:
        if block_kind == "pair":
            if block_index == candidate_index:
                self.tanki[candidate_index] = True
            return
        if block_kind == "triplet":
            if block_index == candidate_index:
                self.shanpon[candidate_index] = True
            return

        classification = _classify_sequence_position(block_index, candidate_index)
        if classification is None:
            return
        getattr(self, classification)[candidate_index] = True

    def to_raw_fields(self) -> dict[str, tuple[int, ...]]:
        mechanism_flags = (
            self.tanki,
            self.shanpon,
            self.kanchan,
            self.penchan,
            self.ryanmen_low_side,
            self.ryanmen_high_side,
            self.kokushi,
        )
        wait_flags = [
            any(flags[index] for flags in mechanism_flags)
            for index in range(_TILE_KIND_COUNT)
        ]
        return {
            "wait_probability_raw": _raw_table(wait_flags),
            "tanki_wait_probability_raw": _raw_table(self.tanki),
            "shanpon_wait_probability_raw": _raw_table(self.shanpon),
            "kanchan_wait_probability_raw": _raw_table(self.kanchan),
            "penchan_wait_probability_raw": _raw_table(self.penchan),
            "ryanmen_low_side_probability_raw": _raw_table(self.ryanmen_low_side),
            "ryanmen_high_side_probability_raw": _raw_table(self.ryanmen_high_side),
            "kokushi_wait_probability_raw": _raw_table(self.kokushi),
        }


def _raw_table(flags: list[bool]) -> tuple[int, ...]:
    return tuple(SCALE if flag else 0 for flag in flags)


def _classify_sequence_position(low_index: int, candidate_index: int) -> str | None:
    """順子blockのどの位置をcandidateが占めるかでmechanism名を返す。

    `low_index`は順子の最も低い牌のcanonical index。penchanは
    `1 2 -> 3`（順子の最高位側、順子の最低位が1）と`8 9 -> 7`（順子の最低位側、
    順子の最高位が9）の2形だけであり、それ以外の端は両面（ryanmen）として
    低い側／高い側を区別する。
    """
    rank_in_suit = low_index % _RANKS_PER_SUIT
    if candidate_index == low_index + 1:
        return "kanchan"
    if candidate_index == low_index:
        return "penchan" if rank_in_suit == _RANKS_PER_SUIT - 3 else "ryanmen_low_side"
    if candidate_index == low_index + 2:
        return "penchan" if rank_in_suit == 0 else "ryanmen_high_side"
    return None


def _enumerate_standard_decompositions(
    counts: tuple[int, ...], melds_needed: int
) -> list[tuple[tuple[str, int], ...]]:
    """通常形（`melds_needed`面子 + 1雀頭）へのすべての有効なdecompositionを
    列挙する。

    `_python_shanten._StandardFormSearch`はshantenのbest block scoreだけを
    求める専用探索であり、全decomposition列挙や、candidateがどのgroupを
    完成させたかのmulti-label分類は行わない。本Issue専用のこの関数は、
    最も若いindexから貪欲にblockを試すbacktrackingで、重複のない
    decomposition集合を列挙する（同じcanonical indexから複数のblockを
    試す順序を固定しているため、同一decompositionを2回記録しない）。

    `melds_needed`が負になることはない（呼び出し側がstable 13-equivalent
    契約を検証済みであるため、`own_melds`の数は常に4以下になる）。
    """
    if melds_needed < 0:
        return []

    results: list[tuple[tuple[str, int], ...]] = []
    working = list(counts)
    blocks: list[tuple[str, int]] = []

    def backtrack(index: int, melds_left: int, pair_used: bool) -> None:
        while index < _TILE_KIND_COUNT and working[index] == 0:
            index += 1
        if index == _TILE_KIND_COUNT:
            if melds_left == 0 and pair_used:
                results.append(tuple(blocks))
            return

        if not pair_used and working[index] >= 2:
            working[index] -= 2
            blocks.append(("pair", index))
            backtrack(index, melds_left, True)
            blocks.pop()
            working[index] += 2

        if melds_left > 0 and working[index] >= 3:
            working[index] -= 3
            blocks.append(("triplet", index))
            backtrack(index, melds_left - 1, pair_used)
            blocks.pop()
            working[index] += 3

        if melds_left > 0 and index < _SUITED_KIND_COUNT:
            rank_in_suit = index % _RANKS_PER_SUIT
            if (
                rank_in_suit <= _RANKS_PER_SUIT - 3
                and working[index + 1] >= 1
                and working[index + 2] >= 1
            ):
                working[index] -= 1
                working[index + 1] -= 1
                working[index + 2] -= 1
                blocks.append(("sequence", index))
                backtrack(index, melds_left - 1, pair_used)
                blocks.pop()
                working[index] += 1
                working[index + 1] += 1
                working[index + 2] += 1

    backtrack(0, melds_needed, False)
    return results


def _is_chiitoitsu_complete(counts: list[int]) -> bool:
    """七対子（7種の対子、同一牌種4枚を2対子と数えない）の完成判定。"""
    distinct_kind_count = sum(1 for count in counts if count > 0)
    if distinct_kind_count != 7:
        return False
    return all(count in (0, 2) for count in counts)


def _is_kokushi_complete(counts: list[int]) -> bool:
    """国士無双（13幺九牌すべて1枚以上、そのうちちょうど1種類が対子）の完成判定。"""
    if any(
        counts[index] != 0
        for index in range(_TILE_KIND_COUNT)
        if index not in _TERMINAL_AND_HONOR_INDICES
    ):
        return False

    pair_kind_count = 0
    for index in _TERMINAL_AND_HONOR_INDICES:
        count = counts[index]
        if count == 0:
            return False
        if count == 2:
            pair_kind_count += 1
        elif count != 1:
            return False
    return pair_kind_count == 1
