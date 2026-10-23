"""34牌種countだけを見るprivateな向聴数計算backend。

このmoduleはlisjongの`Tile`、`OwnHandState`、`PublicMeld`等を一切知らない。
入力は`shanten.py`が正規化した34要素のcount listと、純手牌枚数から判断済みの
確定面子数だけである。

将来profilingで必要性が確認された場合は、この単一moduleをRust / C++ /
lookup table backendへ差し替えられる。公開契約`calculate_shanten()`の意味は
`shanten.py`が保持するため、backendを交換しても利用側の契約は変わらない。

アルゴリズムはMahjongRepository/mahjong、tomohxx/shanten-number、
Cryolite/nyanten、Apricot-S/xiangting等の先行実装から、探索候補の網羅性、
塔子数のcap条件、同じ牌種を5枚必要とする分解を除外する必要性という観点を
学んだうえで、lisjongの命名と設計へ合わせて独立に実装したものである。
source codeの移植は行っていない。

Issue #338で、共有定数と七対子・国士無双helperは`_shanten_common`へ分離した。
本moduleは通常形の再帰探索（旧DFS oracle）だけを持ち、production pathからは
importされない。`_shanten_frontier`もこのmoduleには依存しない。
"""

from collections.abc import Sequence

from lisjong.hand_evaluation import _shanten_common
from lisjong.hand_evaluation._shanten_common import (
    MAX_COPIES_PER_TILE_KIND,
    SUITED_KIND_COUNT,
    TILE_KIND_COUNT,
)

_MAX_BLOCK_COUNT = _shanten_common.MAX_BLOCK_COUNT
_STANDARD_SHANTEN_BASE = _shanten_common.STANDARD_SHANTEN_BASE
_SEED_COUNT_CAP = _shanten_common.SEED_COUNT_CAP
_RANKS_PER_SUIT = 9


def calculate_standard_shanten(counts: Sequence[int], fixed_meld_count: int) -> int:
    """通常形（4面子1雀頭）の向聴数を返す。

    `fixed_meld_count`は副露・槓で既に確定した面子数であり、純手牌枚数から
    判断済みの値を受け取る。確定meldがChi / Pon / Kanのどれであったかは通常形の
    向聴数へ影響しないため、この関数はmeldのidentityを必要としない。

    向聴数は次の式で求める。

        shanten = 8 - 2 * 面子数 - 塔子数 - 雀頭数 + 種牌不足penalty

    面子数は確定面子を含む。雀頭を除く面子・塔子候補の合計は4を超えられない
    ため、探索側で残枠を`4 - fixed_meld_count`へcapする。種牌不足penaltyは
    `_StandardFormSearch.missing_seed_penalty()`を参照する。
    """
    search = _StandardFormSearch(counts)
    block_score = search.best_block_score(_MAX_BLOCK_COUNT - fixed_meld_count)
    return _STANDARD_SHANTEN_BASE - 2 * fixed_meld_count - block_score


class _StandardFormSearch:
    """通常形の分解を探索する、1回の呼び出しに閉じたstate。

    探索中に消費した牌を戻しながら再帰するため可変stateを持つが、instanceは
    `calculate_standard_shanten()`の1回の呼び出しごとに作り直す。memoも同じ
    instanceの寿命に閉じるので、入力が無制限に蓄積するglobal cacheにならない。
    """

    def __init__(self, counts: Sequence[int]) -> None:
        self._original = tuple(counts)
        self._remaining = list(self._original)
        self._memo: dict[tuple[int, ...], int] = {}

    def best_block_score(self, blocks_left: int) -> int:
        """block scoreの最大値を返す。

        block scoreは「2 * 面子数 + 塔子数 + 雀頭数 - 種牌不足penalty」であり、
        大きいほど向聴数が小さい。`blocks_left`は雀頭を除いて使える面子・塔子
        候補の枠数である。
        """
        return self._search(0, blocks_left, 1, 0, 0)

    def _search(
        self,
        index: int,
        blocks_left: int,
        heads_left: int,
        meld_seeds: int,
        head_seeds: int,
    ) -> int:
        """indexの牌を対子系のblockへ使うかどうかを、1回だけ決める。

        同じ牌種を雀頭と対子塔子（シャンポン）の両方へ使うと、和了形がその牌種を
        5枚必要として実在しなくなる。そのため対子系のblockは1牌種につき高々1つ
        しか取らない。面子・塔子とのblock順序は結果へ影響しないので、対子系を先に
        決めてから残りを探索すれば候補は網羅できる。
        """
        remaining = self._remaining
        while index < TILE_KIND_COUNT and remaining[index] == 0:
            index += 1
        if index == TILE_KIND_COUNT:
            return -self.missing_seed_penalty(
                blocks_left, heads_left, meld_seeds, head_seeds
            )

        memo_key = (
            *remaining,
            index,
            blocks_left,
            heads_left,
            meld_seeds,
            head_seeds,
        )
        cached = self._memo.get(memo_key)
        if cached is not None:
            return cached

        best = self._blocks(index, blocks_left, heads_left, meld_seeds, head_seeds)
        if remaining[index] >= 2:
            remaining[index] -= 2
            try:
                if heads_left > 0:
                    reachable = self._blocks(
                        index, blocks_left, heads_left - 1, meld_seeds, head_seeds
                    )
                    best = max(best, 1 + reachable)
                if blocks_left > 0:
                    reachable = self._blocks(
                        index, blocks_left - 1, heads_left, meld_seeds, head_seeds
                    )
                    best = max(best, 1 + reachable)
            finally:
                remaining[index] += 2

        self._memo[memo_key] = best
        return best

    def _blocks(
        self,
        index: int,
        blocks_left: int,
        heads_left: int,
        meld_seeds: int,
        head_seeds: int,
    ) -> int:
        """indexの牌について、対子系以外のblock候補を探索する。

        刻子、順子、両面・辺張相当、嵌張相当を候補とし、どれも取らない場合は
        残り枚数を余り牌としてindexを1つ進める。面子・塔子候補が重なる牌姿で
        greedyに決め打ちしないよう、すべての候補を比較する。
        """
        remaining = self._remaining
        best = self._advance(index, blocks_left, heads_left, meld_seeds, head_seeds)
        if blocks_left == 0 or remaining[index] == 0:
            return best

        candidates: list[tuple[int, tuple[int, ...]]] = []
        if remaining[index] >= 3:
            candidates.append((2, (index, index, index)))
        if self._has_neighbour(index, 1) and self._has_neighbour(index, 2):
            candidates.append((2, (index, index + 1, index + 2)))
        if self._has_neighbour(index, 1):
            candidates.append((1, (index, index + 1)))
        if self._has_neighbour(index, 2):
            candidates.append((1, (index, index + 2)))

        for score, block in candidates:
            for tile_index in block:
                remaining[tile_index] -= 1
            try:
                reachable = self._blocks(
                    index, blocks_left - 1, heads_left, meld_seeds, head_seeds
                )
            finally:
                for tile_index in block:
                    remaining[tile_index] += 1
            best = max(best, score + reachable)
        return best

    def _advance(
        self,
        index: int,
        blocks_left: int,
        heads_left: int,
        meld_seeds: int,
        head_seeds: int,
    ) -> int:
        """indexの残り枚数を余り牌と確定し、次のindexへ進む。

        余り牌は、まだ埋まっていない面子枠や雀頭を育てる種牌になり得る。ただし
        和了形は同じ牌種を4枚までしか含められないため、既にblockへ使った枚数に
        よっては種牌にできない。
        """
        remaining = self._remaining
        leftover = remaining[index]
        if leftover > 0:
            used = self._original[index] - leftover
            if _can_seed_meld(index, used):
                meld_seeds = min(meld_seeds + 1, _SEED_COUNT_CAP)
            if _can_seed_head(used):
                head_seeds = min(head_seeds + 1, _SEED_COUNT_CAP)

        remaining[index] = 0
        try:
            return self._search(
                index + 1, blocks_left, heads_left, meld_seeds, head_seeds
            )
        finally:
            remaining[index] = leftover

    def _has_neighbour(self, index: int, offset: int) -> bool:
        """`index + offset`の牌が、同じ色の数牌として1枚以上あるかを返す。"""
        if index >= SUITED_KIND_COUNT:
            return False
        if index % _RANKS_PER_SUIT + offset >= _RANKS_PER_SUIT:
            return False
        return self._remaining[index + offset] >= 1

    @staticmethod
    def missing_seed_penalty(
        blocks_left: int, heads_left: int, meld_seeds: int, head_seeds: int
    ) -> int:
        """種牌が足りない場合に追加で必要となるツモ数を返す。

        通常形の基本式は、空いている面子枠と雀頭を手牌中の余り牌から育てられる
        ことを前提にしている。育てられる余り牌が無い枠は、まったく新しい牌種を
        引くところから始めるため、1枠につき1回余分にツモが必要になる。
        """
        penalty = 0
        meld_supply = meld_seeds

        if heads_left > 0:
            if head_seeds == 0:
                penalty += 1
            elif head_seeds <= meld_seeds:
                # 雀頭の種牌が面子枠の種牌を兼ねている可能性があるため1つ消費する。
                meld_supply -= 1

        return penalty + max(0, blocks_left - meld_supply)


def _can_seed_meld(index: int, used: int) -> bool:
    """余り牌が、空いている面子枠を育てる種牌になれるかを返す。

    数牌の余り牌は順子へ育てられるので、その牌種は和了形で1枚あればよい。字牌の
    余り牌は刻子にするしかなく、和了形で3枚必要になる。
    """
    needed_in_winning_hand = 1 if index < SUITED_KIND_COUNT else 3
    return used + needed_in_winning_hand <= MAX_COPIES_PER_TILE_KIND


def _can_seed_head(used: int) -> bool:
    """余り牌が雀頭を育てる種牌になれるかを返す。

    雀頭は和了形でその牌種を2枚必要とするため、既に2枚を超えてblockへ使って
    いる牌種は雀頭にできない。4枚使いの余り牌をタンキ待ちにできないのは、この
    制約による。
    """
    return used + 2 <= MAX_COPIES_PER_TILE_KIND
