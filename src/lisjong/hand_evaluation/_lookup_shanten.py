"""通常形shantenをexact lookup tableで解くprivate backend。

Issue #115。`_python_shanten._StandardFormSearch`が1 callごとに行っていた
再帰探索を、事前生成したexact local frontier tableの参照と、group間の
小さな畳み込みへ置き換える。

**これはapproximationではない。** 新しいshanten semanticでもない。table
entryは`_shanten_frontier.local_frontier()`がcurrent標準形backendと同じ
手順で列挙したexact frontierであり、group間の合成もexactである。
分解が成立する理由（blockが牌種groupをまたがないこと、seed累積がcap付き
加算で結合的であること）は`_shanten_frontier`のdocstringを正本とする。

## 責務境界

```text
calculate_shanten(Tile...)                  一般公開API
        ↓
shanten._shanten_from_valid_counts()        唯一のsemantic core
        ↓
_lookup_shanten.calculate_standard_shanten()  private backend（本module）
        ↓
_shanten_table.bin                          read-only artifact
```

Policyやその他のconsumerはこのmoduleへ直接依存しない。artifactの生成は
`tools/generate_shanten_table.py`が行い、`_shanten_frontier`がfrontierの
正本である。runtimeでgeneratorやold DFSを呼ばない。

## fail closed

artifactが存在しない、magic / format versionが一致しない、宣言された
dimensionと実サイズが食い違う、参照先entryが範囲外、といった場合は
`ShantenTableError`でfail closedする。old DFSへのsilent fallbackや、
lookup / DFSを切り替えるruntime optionは持たない。

範囲外参照のcheckは2箇所へ分けている。file sizeが宣言dimensionと一致した
ままの内部破損でも、`IndexError`やsilentに短いsliceを返さないためである。

- frontier span（`pool[start : start + count]`）はload時に全件検証する。
  frontierは数千件しかなく、`spawn` workerごとに走らせても無視できる。
- frontier id（`ids[key]`の値）は参照したものだけをO(1)で検証する。dense
  key空間は数百万entryあり、load時に全件走査すると起動コストが約30ms
  増えるため、hot path側の比較1回へ寄せている。

integrity checkはこの範囲に留め、artifact全体のhash検証はoffline
validation（tests）側の責務とする。
"""

import struct
import sys
from array import array
from collections.abc import Sequence
from importlib import resources

MAGIC = b"LISJSHT\x01"
"""artifact先頭のmagic。"""

FORMAT_VERSION = 1
"""artifact format version。formatを変えたら必ず上げる。"""

HEADER_FORMAT = "<8sIIIII"
"""magic, format version, suit frontier数, honor frontier数, suit pool, honor pool。"""

SUIT_KEY_SPACE = 5**9
"""数牌1色のbase-5 key空間（dense index）。"""

HONOR_KEY_SPACE = 5**7
"""字牌のbase-5 key空間（dense index）。"""

TABLE_RESOURCE = "_shanten_table.bin"
"""package resourceとして同梱するartifact名。"""

_MAX_BLOCK_COUNT = 4
_SEED_COUNT_CAP = 5
_SEED_AXIS = _SEED_COUNT_CAP + 1
_HEAD_AXIS = 2
_RESOURCE_STATE_COUNT = (_MAX_BLOCK_COUNT + 1) * _HEAD_AXIS * _SEED_AXIS * _SEED_AXIS
_SCORE_SHIFT = 4
_SCORE_MASK = (1 << _SCORE_SHIFT) - 1
_INVALID_STATE = 0xFFFF
_UNREACHABLE_PENALTY = 127

_SUIT_KIND_COUNT = 9
_HONOR_KIND_COUNT = 7
_STANDARD_SHANTEN_BASE = 8


class ShantenTableError(Exception):
    """lookup table artifactが利用できない場合のfail closed例外。"""


def pack_entry(
    blocks_used: int, head_used: int, meld_seeds: int, head_seeds: int, score: int
) -> int:
    """frontier stateをartifactへ書く16bit値へ詰める。

    上位bitがresource state（`blocks_used` / `head_used` / seed delta）、
    下位4bitがscoreである。generatorとruntimeで同じ関数を使う。
    """
    state = resource_state(blocks_used, head_used, meld_seeds, head_seeds)
    return (state << _SCORE_SHIFT) | score


def resource_state(
    blocks_used: int, head_used: int, meld_seeds: int, head_seeds: int
) -> int:
    """4次元のfrontier座標をcanonicalな1次元indexへ畳む。"""
    return (
        (blocks_used * _HEAD_AXIS + head_used) * _SEED_AXIS + meld_seeds
    ) * _SEED_AXIS + head_seeds


def _decode_state(state: int) -> tuple[int, int, int, int]:
    head_seeds = state % _SEED_AXIS
    state //= _SEED_AXIS
    meld_seeds = state % _SEED_AXIS
    state //= _SEED_AXIS
    head_used = state % _HEAD_AXIS
    return state // _HEAD_AXIS, head_used, meld_seeds, head_seeds


def _missing_seed_penalty(
    blocks_left: int, heads_left: int, meld_seeds: int, head_seeds: int
) -> int:
    """current backendの`missing_seed_penalty()`と同じ定義。

    種牌が足りない枠は、新しい牌種を引くところから始めるため1枠につき1回
    余分にツモが必要になる。
    """
    penalty = 0
    meld_supply = meld_seeds
    if heads_left > 0:
        if head_seeds == 0:
            penalty += 1
        elif head_seeds <= meld_seeds:
            meld_supply -= 1
    return penalty + max(0, blocks_left - meld_supply)


def _build_combine_table() -> array:
    """2つのresource stateを合成した結果を引く表を作る。

    `blocks_used`の合計が4を超える、または`head_used`の合計が2になる
    組み合わせは無効値を返す。seed deltaはcap付きで加算する。
    """
    table = array("H", [_INVALID_STATE]) * (
        _RESOURCE_STATE_COUNT * _RESOURCE_STATE_COUNT
    )
    decoded = [_decode_state(state) for state in range(_RESOURCE_STATE_COUNT)]
    for left in range(_RESOURCE_STATE_COUNT):
        blocks_left_used, head_left_used, ms_left, hs_left = decoded[left]
        row = left * _RESOURCE_STATE_COUNT
        for right in range(_RESOURCE_STATE_COUNT):
            blocks_right, head_right, ms_right, hs_right = decoded[right]
            blocks_used = blocks_left_used + blocks_right
            if blocks_used > _MAX_BLOCK_COUNT:
                continue
            head_used = head_left_used + head_right
            if head_used > 1:
                continue
            meld_seeds = ms_left + ms_right
            head_seeds = hs_left + hs_right
            table[row + right] = resource_state(
                blocks_used,
                head_used,
                meld_seeds if meld_seeds < _SEED_COUNT_CAP else _SEED_COUNT_CAP,
                head_seeds if head_seeds < _SEED_COUNT_CAP else _SEED_COUNT_CAP,
            )
    return table


def _build_penalty_tables() -> list[array]:
    """`fixed_meld_count`ごとに、resource stateから引くpenaltyを作る。

    global budgetを超えて`blocks_used`を使うstateは`_UNREACHABLE_PENALTY`と
    し、合成結果の走査時に弾く。
    """
    tables = []
    for fixed_meld_count in range(_MAX_BLOCK_COUNT + 1):
        budget = _MAX_BLOCK_COUNT - fixed_meld_count
        table = array("b", [_UNREACHABLE_PENALTY]) * _RESOURCE_STATE_COUNT
        for state in range(_RESOURCE_STATE_COUNT):
            blocks_used, head_used, meld_seeds, head_seeds = _decode_state(state)
            if blocks_used > budget:
                continue
            table[state] = _missing_seed_penalty(
                budget - blocks_used, 1 - head_used, meld_seeds, head_seeds
            )
        tables.append(table)
    return tables


def _validate_spans(starts: array, counts: array, pool: array, label: str) -> None:
    """frontier spanがpoolへ収まることをload時に確認する。

    file sizeが宣言dimensionと一致していても、spanが壊れていれば
    `pool[start : start + count]`はsliceなので例外を出さずに短い、あるいは
    空のentry列を返してしまう。これはIndexErrorより悪く、silentに誤った
    shantenを生む。frontierは高々数千件なのでload時に全件見ておく。

    frontier id自体のboundsはここでは見ない。dense key空間は数百万entryあり、
    全件走査するとWindows `spawn` workerごとの起動コストが跳ね上がる
    （実測で+30ms）。idは`calculate_standard_shanten()`が実際に参照した
    ものだけをO(1)で検査する。artifact全体のcryptographic hash検証は
    行わない（それはoffline validationの責務）。
    """
    frontier_count = len(starts)
    if frontier_count == 0 or len(counts) != frontier_count:
        raise ShantenTableError(
            f"{label} frontier index of the shanten table artifact is empty "
            "or inconsistent"
        )
    pool_length = len(pool)
    for frontier_id in range(frontier_count):
        if starts[frontier_id] + counts[frontier_id] > pool_length:
            raise ShantenTableError(
                f"{label} frontier span of the shanten table artifact "
                "reaches past the end of its entry pool"
            )


class _ShantenTable:
    """artifactを読み、base-5 group keyからfrontier entryを引くread-only view。

    key空間はdenseなので`ids[key]`で直接frontier idを引ける。frontier実体は
    重複が多いためdistinctなものだけをpoolへ持ち、idからspanを引く。
    """

    __slots__ = (
        "honor_counts",
        "honor_frontier_count",
        "honor_ids",
        "honor_pool",
        "honor_starts",
        "suit_counts",
        "suit_frontier_count",
        "suit_ids",
        "suit_pool",
        "suit_starts",
    )

    def __init__(self, payload: bytes) -> None:
        header_size = struct.calcsize(HEADER_FORMAT)
        if len(payload) < header_size:
            raise ShantenTableError("shanten table artifact is truncated")
        (
            magic,
            version,
            suit_frontier_count,
            honor_frontier_count,
            suit_pool_entries,
            honor_pool_entries,
        ) = struct.unpack_from(HEADER_FORMAT, payload)

        if magic != MAGIC:
            raise ShantenTableError("shanten table artifact has an unexpected magic")
        if version != FORMAT_VERSION:
            raise ShantenTableError(
                "shanten table artifact format version is "
                f"{version}, expected {FORMAT_VERSION}"
            )

        expected = (
            header_size
            + (SUIT_KEY_SPACE + HONOR_KEY_SPACE) * 2
            + (suit_frontier_count + honor_frontier_count) * 5
            + (suit_pool_entries + honor_pool_entries) * 2
        )
        if len(payload) != expected:
            raise ShantenTableError(
                "shanten table artifact size does not match its declared "
                f"dimensions: {len(payload)} bytes, expected {expected}"
            )

        offset = header_size

        def take(typecode: str, count: int, item_size: int):
            nonlocal offset
            values = array(typecode)
            values.frombytes(payload[offset : offset + count * item_size])
            offset += count * item_size
            if sys.byteorder == "big" and item_size > 1:
                values.byteswap()
            return values

        self.suit_ids = take("H", SUIT_KEY_SPACE, 2)
        self.honor_ids = take("H", HONOR_KEY_SPACE, 2)
        self.suit_starts = take("I", suit_frontier_count, 4)
        self.suit_counts = take("B", suit_frontier_count, 1)
        self.honor_starts = take("I", honor_frontier_count, 4)
        self.honor_counts = take("B", honor_frontier_count, 1)
        self.suit_pool = take("H", suit_pool_entries, 2)
        self.honor_pool = take("H", honor_pool_entries, 2)

        _validate_spans(self.suit_starts, self.suit_counts, self.suit_pool, "suit")
        _validate_spans(self.honor_starts, self.honor_counts, self.honor_pool, "honor")
        # hot pathがO(1)でid boundsを見られるように、frontier数を持たせておく。
        self.suit_frontier_count = len(self.suit_starts)
        self.honor_frontier_count = len(self.honor_starts)


def _load_table() -> _ShantenTable:
    try:
        payload = resources.files(__package__).joinpath(TABLE_RESOURCE).read_bytes()
    except (FileNotFoundError, OSError) as error:
        raise ShantenTableError(
            f"shanten table artifact {TABLE_RESOURCE!r} is missing from the "
            "lisjong.hand_evaluation package"
        ) from error
    return _ShantenTable(payload)


_COMBINE = _build_combine_table()
_PENALTY = _build_penalty_tables()
_TABLE: _ShantenTable | None = None


def _table() -> _ShantenTable:
    """artifactを最初の利用時に1回だけ読み込む。

    import時ではなく初回callで読むことで、artifactを生成するgenerator自身が
    このmoduleをimportできる。読み込みに失敗した場合はfail closedであり、
    old DFSへfallbackしない。
    """
    global _TABLE
    table = _TABLE
    if table is None:
        table = _TABLE = _load_table()
    return table


def calculate_standard_shanten(counts: Sequence[int], fixed_meld_count: int) -> int:
    """通常形（4面子1雀頭）の向聴数を、exact lookup tableから返す。

    `counts`はcanonical 34牌種countで、`_shanten_from_valid_counts()`が
    保証するpreconditionを満たしていることを前提とする。
    """
    table = _table()
    frontier: dict[int, int] | None = None
    suit_frontier_count = table.suit_frontier_count
    honor_frontier_count = table.honor_frontier_count

    for base, kind_count in ((0, 9), (9, 9), (18, 9), (27, 7)):
        key = 0
        for index in range(base, base + kind_count):
            key = key * 5 + counts[index]
        if kind_count == _SUIT_KIND_COUNT:
            frontier_id = table.suit_ids[key]
            if frontier_id >= suit_frontier_count:
                raise ShantenTableError(
                    "suit key index of the shanten table artifact references a "
                    "frontier that does not exist"
                )
            start = table.suit_starts[frontier_id]
            length = table.suit_counts[frontier_id]
            pool = table.suit_pool
        else:
            frontier_id = table.honor_ids[key]
            if frontier_id >= honor_frontier_count:
                raise ShantenTableError(
                    "honor key index of the shanten table artifact references a "
                    "frontier that does not exist"
                )
            start = table.honor_starts[frontier_id]
            length = table.honor_counts[frontier_id]
            pool = table.honor_pool
        entries = pool[start : start + length]

        if frontier is None:
            frontier = {}
            for packed in entries:
                state = packed >> _SCORE_SHIFT
                score = packed & _SCORE_MASK
                if frontier.get(state, -1) < score:
                    frontier[state] = score
            continue

        merged: dict[int, int] = {}
        for left_state, left_score in frontier.items():
            row = left_state * _RESOURCE_STATE_COUNT
            for packed in entries:
                combined = _COMBINE[row + (packed >> _SCORE_SHIFT)]
                if combined == _INVALID_STATE:
                    continue
                score = left_score + (packed & _SCORE_MASK)
                if merged.get(combined, -1) < score:
                    merged[combined] = score
        frontier = merged

    if not frontier:
        raise ShantenTableError(
            "shanten table produced no reachable decomposition for the hand"
        )

    penalties = _PENALTY[fixed_meld_count]
    best = None
    for state, score in frontier.items():
        penalty = penalties[state]
        if penalty == _UNREACHABLE_PENALTY:
            continue
        value = score - penalty
        if best is None or value > best:
            best = value
    if best is None:
        raise ShantenTableError(
            "shanten table produced no decomposition within the meld budget"
        )
    return _STANDARD_SHANTEN_BASE - 2 * fixed_meld_count - best
