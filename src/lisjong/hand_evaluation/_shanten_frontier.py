"""通常形shantenのexact local decomposition frontierを導出するprivate generator。

Issue #115で追加した、lookup-table backend（`_lookup_shanten`）用のtableを
**生成する**側のmoduleである。runtimeのshanten計算はこのmoduleを呼ばない。
呼び出すのはoffline generator（`tools/generate_shanten_table.py`）と、
exactnessを固定するtestsだけである。

## なぜlocal frontierで足りるのか

current標準形backend（`_python_shanten._StandardFormSearch`）は、34牌種を
index順に走査しながら次を行う。

- 面子 / 順子 / 塔子 / 対子blockの取り出し（scoreを加算）
- 余り牌からmeld seed / head seedを数える
- 最後に`missing_seed_penalty()`を引く

ここで重要なのは、**blockが牌種group（萬子 / 筒子 / 索子 / 字牌）をまたがない**
ことである。順子は`_has_neighbour()`が同色内に限定し、字牌には順子が存在
しない。刻子・対子も単一牌種で閉じる。したがって牌の消費と4枚制約は各group
内でlocalに完結する。

group間で受け渡されるstateは次の4つだけである。

```text
blocks_left    使用したblock数（groupごとのblocks_used）
heads_left     雀頭を使ったか（groupごとのhead_used、全体で高々1）
meld_seeds     余り牌由来のmeld seed数（cap付き加算）
head_seeds     余り牌由来のhead seed数（cap付き加算）
```

`meld_seeds` / `head_seeds`はsearch中どの分岐条件にも使われず、末端の
`missing_seed_penalty()`でしか読まれない純粋な累積値である。cap付き加算
（`min(value + 1, cap)`）は単調増加なので、最終値は`min(総increment数, cap)`
に等しく、groupごとの部分和からexactに再構成できる。

したがって、各groupについて

```text
(blocks_used, head_used, meld_seed_delta, head_seed_delta) -> 最大score
```

というfrontierを持てば、group間はこの4次元の畳み込みだけでexactに合成できる。
これは近似ではなく、current backendの探索空間をgroup境界で分解したものである。

## 4枚制約とseed-shortage semantics

frontierは`_StandardFormSearch`と同じ手順で列挙するため、次のcurrent
semanticsをそのまま保つ。

- 存在しない5枚目を要求する分解を作らない（`remaining`を実際に減算して
  探索するため、物理的に不可能な組み合わせは列挙されない）
- 余り牌がseedになれるかを`used`枚数から判定する
  （数牌は和了形で1枚、字牌は3枚必要。雀頭は2枚必要）
- 対子系blockは1牌種につき高々1つしか取らない

## 確定面子数との関係

`blocks_used`はgroupごとの使用数で、global budget
`4 - fixed_meld_count`との比較は合成側で行う。生成時はglobal budgetを
仮定せず、取り得る最大の`4`まで列挙する。したがって同じtableが
`fixed_meld_count = 0..4`のすべてで使える。
"""

from collections.abc import Sequence

from lisjong.hand_evaluation import _python_shanten

MAX_BLOCK_COUNT = _python_shanten._MAX_BLOCK_COUNT
"""雀頭を除いて使える面子・塔子候補の最大数。"""

SEED_COUNT_CAP = _python_shanten._SEED_COUNT_CAP
"""meld seed / head seedの上限（これ以上は結果へ影響しない）。"""

MAX_COPIES_PER_TILE_KIND = _python_shanten.MAX_COPIES_PER_TILE_KIND
"""1牌種あたりのphysical上限枚数。"""

SUIT_KIND_COUNT = 9
"""1つの数牌色が持つ牌種数。"""

HONOR_KIND_COUNT = 7
"""字牌の牌種数。"""

MAX_GROUP_TILE_COUNT = 14
"""1 groupへ入り得る最大枚数（純手牌の上限枚数）。"""


def local_frontier(
    counts: Sequence[int], *, suited: bool
) -> dict[tuple[int, int, int, int], int]:
    """1 groupのexact decomposition frontierを返す。

    返り値は`(blocks_used, head_used, meld_seed_delta, head_seed_delta)`から
    そのstateで到達できる**最大local score**への写像である。scoreは
    `_StandardFormSearch`と同じ重み（面子・順子が2、塔子・対子が1）で数える。

    `suited`はこのgroupが数牌色かどうかを表す。順子の有無とseed判定
    （数牌は和了形で1枚、字牌は3枚必要）に影響する。
    """
    size = len(counts)
    original = tuple(counts)
    remaining = list(counts)
    frontier: dict[tuple[int, int, int, int], int] = {}
    meld_seed_requirement = 1 if suited else 3

    def record(
        blocks_used: int, head_used: int, meld_seeds: int, head_seeds: int, score: int
    ) -> None:
        key = (
            blocks_used,
            head_used,
            min(meld_seeds, SEED_COUNT_CAP),
            min(head_seeds, SEED_COUNT_CAP),
        )
        if frontier.get(key, -1) < score:
            frontier[key] = score

    def has_neighbour(index: int, offset: int) -> bool:
        if not suited:
            return False
        if index + offset >= size:
            return False
        return remaining[index + offset] >= 1

    def search(
        index: int, blocks_used: int, head_used: int, ms: int, hs: int, score: int
    ) -> None:
        while index < size and remaining[index] == 0:
            index += 1
        if index == size:
            record(blocks_used, head_used, ms, hs, score)
            return

        blocks(index, blocks_used, head_used, ms, hs, score)
        if remaining[index] >= 2:
            # 対子系blockは1牌種につき高々1つ。同じ牌種を雀頭とシャンポンの
            # 両方へ使うと和了形で5枚必要になり、実在しないためである。
            remaining[index] -= 2
            try:
                if head_used == 0:
                    blocks(index, blocks_used, 1, ms, hs, score + 1)
                if blocks_used < MAX_BLOCK_COUNT:
                    blocks(index, blocks_used + 1, head_used, ms, hs, score + 1)
            finally:
                remaining[index] += 2

    def blocks(
        index: int, blocks_used: int, head_used: int, ms: int, hs: int, score: int
    ) -> None:
        advance(index, blocks_used, head_used, ms, hs, score)
        if blocks_used >= MAX_BLOCK_COUNT or remaining[index] == 0:
            return

        candidates: list[tuple[int, tuple[int, ...]]] = []
        if remaining[index] >= 3:
            candidates.append((2, (index, index, index)))
        if has_neighbour(index, 1) and has_neighbour(index, 2):
            candidates.append((2, (index, index + 1, index + 2)))
        if has_neighbour(index, 1):
            candidates.append((1, (index, index + 1)))
        if has_neighbour(index, 2):
            candidates.append((1, (index, index + 2)))

        for block_score, block in candidates:
            for tile_index in block:
                remaining[tile_index] -= 1
            try:
                blocks(index, blocks_used + 1, head_used, ms, hs, score + block_score)
            finally:
                for tile_index in block:
                    remaining[tile_index] += 1

    def advance(
        index: int, blocks_used: int, head_used: int, ms: int, hs: int, score: int
    ) -> None:
        leftover = remaining[index]
        if leftover > 0:
            used = original[index] - leftover
            if used + meld_seed_requirement <= MAX_COPIES_PER_TILE_KIND:
                ms += 1
            if used + 2 <= MAX_COPIES_PER_TILE_KIND:
                hs += 1
        remaining[index] = 0
        try:
            search(index + 1, blocks_used, head_used, ms, hs, score)
        finally:
            remaining[index] = leftover

    search(0, 0, 0, 0, 0, 0)
    return frontier


def dominant_frontier(
    frontier: dict[tuple[int, int, int, int], int],
) -> dict[tuple[int, int, int, int], int]:
    """exact-safeなdominanceだけでfrontierを縮約する。

    同じ`(blocks_used, head_used)`のstate同士でのみ比較し、

        score >= かつ meld_seed_delta >= かつ head_seed_delta >=

    を満たすstateがもう一方を支配する。`missing_seed_penalty()`は
    `meld_seeds` / `head_seeds`について単調非増加なので、seedが多い側は
    どのblocks_left / heads_left / 他groupとの組み合わせでもpenaltyが
    増えない。scoreも大きい側が常に有利である。

    `blocks_used`と`head_used`は縮約しない。これらはglobal budgetを消費する
    resource座標であり、使用量が少ない方が常に有利とは限らないためである
    （未使用のblock枠は`missing_seed_penalty()`でpenaltyになり得る。雀頭も
    「使えば`heads_left`が減ってpenaltyを避けられる」一方、他groupが雀頭を
    作れる場合は譲った方がよい）。
    """
    grouped: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for (blocks_used, head_used, ms, hs), score in frontier.items():
        grouped.setdefault((blocks_used, head_used), []).append((ms, hs, score))

    reduced: dict[tuple[int, int, int, int], int] = {}
    for (blocks_used, head_used), states in grouped.items():
        for state in states:
            if any(
                other != state
                and other[0] >= state[0]
                and other[1] >= state[1]
                and other[2] >= state[2]
                for other in states
            ):
                continue
            reduced[(blocks_used, head_used, state[0], state[1])] = state[2]
    return reduced


def enumerate_group_keys(kind_count: int) -> list[tuple[int, ...]]:
    """1 groupで物理的に到達し得るcount stateをcanonical順で列挙する。

    各牌種0..4枚、group合計は純手牌上限の14枚以下である。
    """
    states: list[tuple[int, ...]] = [()]
    for _ in range(kind_count):
        extended: list[tuple[int, ...]] = []
        for state in states:
            used = sum(state)
            for count in range(MAX_COPIES_PER_TILE_KIND + 1):
                if used + count <= MAX_GROUP_TILE_COUNT:
                    extended.append((*state, count))
        states = extended
    return states


def group_key(counts: Sequence[int]) -> int:
    """groupのcount stateをbase-5のdeterministicな整数keyへ変換する。

    先頭の牌種が最上位桁になる。dict iteration order、hash randomization、
    object identity、Enum宣言順等には依存しない。
    """
    key = 0
    for count in counts:
        key = key * 5 + count
    return key
