"""通常形shantenをexact lookup tableで解くprivate backend（Python計算本体）。

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
_shanten_table（format / reader / combine / penalty）
        ↓
_shanten_table.bin                          read-only artifact
```

Issue #338で、artifactのformat・reader・構造検証・`ShantenTableError`・
combine / penalty表は`_shanten_table`へ分離した。native backendとoffline
generatorはそちらだけに依存し、本moduleはPython lookup計算（artifactの
lazy load、scratch workspace、frontier combine loop）だけを持つ。

Policyやその他のconsumerはこのmoduleへ直接依存しない。artifactの生成は
`tools/generate_shanten_table.py`が行い、`_shanten_frontier`がfrontierの
正本である。runtimeでgeneratorやold DFSを呼ばない。

## fail closed

artifact integrity failureは`_shanten_table.ShantenTableError`で
fail closedし、old DFSへのsilent fallbackや、lookup / DFSを切り替える
runtime optionは持たない。frontier spanはload時（`_shanten_table`）に、
frontier idは本moduleのhot pathが参照したものだけをO(1)で検証する
（理由は`_shanten_table`のdocstring参照）。
"""

import threading
from collections.abc import Sequence

from lisjong.hand_evaluation._shanten_common import STANDARD_SHANTEN_BASE
from lisjong.hand_evaluation._shanten_table import (
    _COMBINE,
    _INVALID_STATE,
    _PENALTY,
    _RESOURCE_STATE_COUNT,
    _SCORE_MASK,
    _SCORE_SHIFT,
    _UNREACHABLE_PENALTY,
    ShantenTableError,
    _ShantenTable,
    read_table_payload,
)


def _load_table() -> _ShantenTable:
    return _ShantenTable(read_table_payload())


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


# --- runtime frontier combineのscratch buffer（Issue #131 / #132） ---
#
# `calculate_standard_shanten()`は1 decisionあたり数万〜十万回呼ばれるhot
# pathで、支配的costはgroup間のfrontier combineだった（PR #118 profile:
# `calculate_standard_shanten` tottime 4.017s / `dict.get` 1,040万call）。
# combine先のresource stateは`_RESOURCE_STATE_COUNT`（360）でdenseに収まる
# ため、callごとに`dict`を作ってhashするのではなく、固定sizeのlistへ
# epoch stampで書き込む。stampが現在のepochと一致しないentryは「この
# callではまだ触れていない」ことを意味し、call間で明示的にlistをclearする
# 必要がない（epochは単調増加するworkspace-local世代番号）。
#
# 2本のbuffer（A / B）をping-pongで使い回す。1回のcallは
# 「初期構築 + 3回のmerge」で計4 phaseを行うため、奇数phaseの出力が
# 偶数phaseの入力になるよう交互に書き込み先を入れ替える。
#
# `calculate_shanten()`は一般公開APIであり、呼び出し側がthreadを使わない
# 保証はない。CPythonはpure Python bytecodeの実行中でもthread切り替えが
# 起き得るため、scratch bufferをmodule-globalに1組だけ持つと、他threadが
# 書き込んだ内容を読んでsilentに誤ったshantenを返し得る（Issue #132
# レビュー）。したがってscratch workspaceは`threading.local()`でthreadごと
# に分離する。globalなlockで全shanten計算をserializeするより、独立した
# workspaceをthreadごとに持たせる方針を優先した。プロセスをまたぐ共有は
# そもそも発生しない（`multiprocessing`はforkでもspawnでも子ごとに別
# interpreter stateを持つ）ので、対象はthread間の共有だけである。
#
# `calculate_standard_shanten()`自身は再帰しない（非reentrant）ため、
# 同一threadの中でこのworkspaceを複数の呼び出しが同時に使うことはない。
# 各threadは1個の`_ScratchWorkspace`を初回callで作り、以後使い回す。
# module-level scratch bufferの再利用はfail-closedな読み取り専用artifact
# の話ではなくpure runtime workspaceなので、artifact整合性boundaryとは
# 無関係である。


class _ScratchWorkspace:
    """1 threadが専有するfrontier combineのscratch領域。"""

    __slots__ = ("epoch", "scores_a", "scores_b", "stamp_a", "stamp_b")

    def __init__(self) -> None:
        self.scores_a = [0] * _RESOURCE_STATE_COUNT
        self.stamp_a = [-1] * _RESOURCE_STATE_COUNT
        self.scores_b = [0] * _RESOURCE_STATE_COUNT
        self.stamp_b = [-1] * _RESOURCE_STATE_COUNT
        self.epoch = 0
        """scratch bufferの単調増加workspace-local世代番号。0は「未使用」。"""


_SCRATCH_LOCAL = threading.local()


def _scratch_workspace() -> _ScratchWorkspace:
    """呼び出しthreadに紐づく`_ScratchWorkspace`を返す（初回callで作る）。"""
    workspace = getattr(_SCRATCH_LOCAL, "workspace", None)
    if workspace is None:
        workspace = _ScratchWorkspace()
        _SCRATCH_LOCAL.workspace = workspace
    return workspace


def calculate_standard_shanten(counts: Sequence[int], fixed_meld_count: int) -> int:
    """通常形（4面子1雀頭）の向聴数を、exact lookup tableから返す。

    `counts`はcanonical 34牌種countで、`_shanten_from_valid_counts()`が
    保証するpreconditionを満たしていることを前提とする。

    frontier combineの実装はIssue #131で最適化した。exact frontier
    definitionとgroup間combineがexactである理由（block/seedがgroupを
    またがないこと）はPR #116の`_shanten_frontier`docstringが正本であり、
    ここで行っているのは同じexact計算を安く行う3つの変更だけである。

    1. **combine順序の並べ替え。** combineは結合的なので、4 groupを
       どの順で畳み込んでもexact resultは変わらない。entry数が少ない
       groupから畳み込むと、中間frontierの膨張を抑えられる（実測で
       支配的fixtureが最大1.8x〜2.6x）。
    2. **dictの代わりにepoch-stamped dense array。** resource stateは
       `_RESOURCE_STATE_COUNT`個に収まるため、`dict`のhashingではなく
       固定sizeのlistへ直接indexする。scratch領域はthreadごとに独立した
       `_ScratchWorkspace`を使う（`_scratch_workspace()`参照）。
    3. **base-5 keyのloop展開。** `range()`ループより約2倍速い。

    どの変更も、参照するfrontier entry・組み合わせるresource state・
    採用する最大scoreを一切変えていない。単に同じ計算を行う順序と
    データ構造を変えただけである。
    """
    table = _table()
    suit_ids = table.suit_ids
    suit_starts = table.suit_starts
    suit_counts = table.suit_counts
    suit_pool = table.suit_pool
    suit_frontier_count = table.suit_frontier_count
    honor_ids = table.honor_ids
    honor_starts = table.honor_starts
    honor_counts = table.honor_counts
    honor_pool = table.honor_pool
    honor_frontier_count = table.honor_frontier_count

    # base-5 group key。4群とも9 / 7牌種のrange loopをやめ、乗加算を展開
    # している（このmoduleのdocstring・profile findings参照）。
    m = counts
    manzu_key = (
        (
            (((((m[0] * 5 + m[1]) * 5 + m[2]) * 5 + m[3]) * 5 + m[4]) * 5 + m[5]) * 5
            + m[6]
        )
        * 5
        + m[7]
    ) * 5 + m[8]
    pinzu_key = (
        (
            (((((m[9] * 5 + m[10]) * 5 + m[11]) * 5 + m[12]) * 5 + m[13]) * 5 + m[14])
            * 5
            + m[15]
        )
        * 5
        + m[16]
    ) * 5 + m[17]
    souzu_key = (
        (
            (((((m[18] * 5 + m[19]) * 5 + m[20]) * 5 + m[21]) * 5 + m[22]) * 5 + m[23])
            * 5
            + m[24]
        )
        * 5
        + m[25]
    ) * 5 + m[26]
    honor_key = (
        ((((m[27] * 5 + m[28]) * 5 + m[29]) * 5 + m[30]) * 5 + m[31]) * 5 + m[32]
    ) * 5 + m[33]

    groups: list[tuple[int, list[int]]] = []
    for key, ids, starts, lengths, pool, frontier_count, label in (
        (
            manzu_key,
            suit_ids,
            suit_starts,
            suit_counts,
            suit_pool,
            suit_frontier_count,
            "suit",
        ),
        (
            pinzu_key,
            suit_ids,
            suit_starts,
            suit_counts,
            suit_pool,
            suit_frontier_count,
            "suit",
        ),
        (
            souzu_key,
            suit_ids,
            suit_starts,
            suit_counts,
            suit_pool,
            suit_frontier_count,
            "suit",
        ),
        (
            honor_key,
            honor_ids,
            honor_starts,
            honor_counts,
            honor_pool,
            honor_frontier_count,
            "honor",
        ),
    ):
        frontier_id = ids[key]
        if frontier_id >= frontier_count:
            raise ShantenTableError(
                f"{label} key index of the shanten table artifact references a "
                "frontier that does not exist"
            )
        start = starts[frontier_id]
        length = lengths[frontier_id]
        groups.append((length, pool[start : start + length].tolist()))

    # combineは結合的なので、entry数が少ないgroupから畳み込んでも
    # exact resultは変わらない（このmoduleのdocstring参照）。
    groups.sort(key=lambda group: group[0])

    workspace = _scratch_workspace()
    combine = _COMBINE
    invalid_state = _INVALID_STATE
    score_shift = _SCORE_SHIFT
    score_mask = _SCORE_MASK
    resource_state_count = _RESOURCE_STATE_COUNT

    cur_scores = workspace.scores_a
    cur_stamp = workspace.stamp_a
    next_scores = workspace.scores_b
    next_stamp = workspace.stamp_b

    touched: list[int] = []
    for step, (_length, entries) in enumerate(groups):
        workspace.epoch += 1
        epoch = workspace.epoch

        if step == 0:
            for packed in entries:
                state = packed >> score_shift
                score = packed & score_mask
                if cur_stamp[state] != epoch:
                    cur_stamp[state] = epoch
                    cur_scores[state] = score
                    touched.append(state)
                elif score > cur_scores[state]:
                    cur_scores[state] = score
            continue

        merged_touched: list[int] = []
        for left_state in touched:
            left_score = cur_scores[left_state]
            row = left_state * resource_state_count
            for packed in entries:
                combined = combine[row + (packed >> score_shift)]
                if combined == invalid_state:
                    continue
                score = left_score + (packed & score_mask)
                if next_stamp[combined] != epoch:
                    next_stamp[combined] = epoch
                    next_scores[combined] = score
                    merged_touched.append(combined)
                elif score > next_scores[combined]:
                    next_scores[combined] = score
        touched = merged_touched
        cur_scores, next_scores = next_scores, cur_scores
        cur_stamp, next_stamp = next_stamp, cur_stamp

    if not touched:
        raise ShantenTableError(
            "shanten table produced no reachable decomposition for the hand"
        )

    penalties = _PENALTY[fixed_meld_count]
    best = None
    for state in touched:
        penalty = penalties[state]
        if penalty == _UNREACHABLE_PENALTY:
            continue
        value = cur_scores[state] - penalty
        if best is None or value > best:
            best = value
    if best is None:
        raise ShantenTableError(
            "shanten table produced no decomposition within the meld budget"
        )
    return STANDARD_SHANTEN_BASE - 2 * fixed_meld_count - best
