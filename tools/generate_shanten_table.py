"""通常形shanten lookup table artifactを生成するdeterministic generator。

Issue #115。`lisjong.hand_evaluation._shanten_frontier`が導出するexact local
frontierを、`_lookup_shanten`がruntimeで読むcompact binary artifactへ書き出す。

    python tools/generate_shanten_table.py \
        src/lisjong/hand_evaluation/_shanten_table.bin

同じsourceからは常にbyte-identicalな出力になる。dict iteration order、hash
randomization、object identityへ依存しない（keyはbase-5整数の昇順、frontier
entryはpacked値の昇順で書き出す）。

runtimeでinternet accessを必要とせず、第三者repositoryの生成物やsource
codeをcopyしていない。artifact formatは`_lookup_shanten`が正本とする。
"""

import argparse
import hashlib
import pathlib
import struct
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lisjong.hand_evaluation import _lookup_shanten  # noqa: E402
from lisjong.hand_evaluation._shanten_frontier import (  # noqa: E402
    HONOR_KIND_COUNT,
    SUIT_KIND_COUNT,
    dominant_frontier,
    enumerate_group_keys,
    group_key,
    local_frontier,
)


def _build_group(kind_count: int, *, suited: bool, progress_label: str):
    """base-5 key空間全体のfrontier id配列と、共有frontier poolを構築する。

    key空間は`5 ** kind_count`のdense配列にする。runtimeがbase-5 keyで直接
    index参照できるようにするためである（到達不能keyはid 0 = empty sentinel）。
    frontier実体は重複が非常に多いので、distinct frontierだけをpoolへ入れ、
    keyごとにはそのidだけを持たせる。
    """
    keys = enumerate_group_keys(kind_count)
    key_space = 5**kind_count
    ids = [0] * key_space
    pool: list[int] = []
    # id 0 は「entryなし」のsentinel。到達不能keyがここへ落ちる。
    id_spans: list[tuple[int, int]] = [(0, 0)]
    shared: dict[tuple[int, ...], int] = {}

    started = time.perf_counter()
    for position, counts in enumerate(keys):
        frontier = dominant_frontier(local_frontier(counts, suited=suited))
        packed = tuple(
            sorted(
                _lookup_shanten.pack_entry(blocks_used, head_used, ms, hs, score)
                for (blocks_used, head_used, ms, hs), score in frontier.items()
            )
        )
        frontier_id = shared.get(packed)
        if frontier_id is None:
            frontier_id = len(id_spans)
            shared[packed] = frontier_id
            id_spans.append((len(pool), len(packed)))
            pool.extend(packed)
        ids[group_key(counts)] = frontier_id

        if position % 20000 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"  {progress_label}: {position:,}/{len(keys):,} "
                f"({elapsed:.0f}s, pool={len(pool):,}, distinct={len(shared):,})",
                flush=True,
            )

    return ids, id_spans, pool, len(keys), time.perf_counter() - started


def build_artifact() -> tuple[bytes, dict[str, int]]:
    suit_ids, suit_spans, suit_pool, suit_reachable, suit_time = _build_group(
        SUIT_KIND_COUNT, suited=True, progress_label="suit"
    )
    honor_ids, honor_spans, honor_pool, honor_reachable, honor_time = _build_group(
        HONOR_KIND_COUNT, suited=False, progress_label="honors"
    )

    header = struct.pack(
        _lookup_shanten.HEADER_FORMAT,
        _lookup_shanten.MAGIC,
        _lookup_shanten.FORMAT_VERSION,
        len(suit_spans),
        len(honor_spans),
        len(suit_pool),
        len(honor_pool),
    )

    chunks = [header]
    for ids in (suit_ids, honor_ids):
        chunks.append(struct.pack(f"<{len(ids)}H", *ids))
    for spans in (suit_spans, honor_spans):
        chunks.append(struct.pack(f"<{len(spans)}I", *(start for start, _c in spans)))
        chunks.append(struct.pack(f"<{len(spans)}B", *(count for _s, count in spans)))
    for pool in (suit_pool, honor_pool):
        chunks.append(struct.pack(f"<{len(pool)}H", *pool))

    payload = b"".join(chunks)
    stats = {
        "suit_key_space": len(suit_ids),
        "honor_key_space": len(honor_ids),
        "suit_reachable_keys": suit_reachable,
        "honor_reachable_keys": honor_reachable,
        "suit_distinct_frontiers": len(suit_spans) - 1,
        "honor_distinct_frontiers": len(honor_spans) - 1,
        "suit_pool_entries": len(suit_pool),
        "honor_pool_entries": len(honor_pool),
        "suit_generation_seconds": round(suit_time, 1),
        "honor_generation_seconds": round(honor_time, 1),
        "bytes": len(payload),
    }
    return payload, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    arguments = parser.parse_args()

    payload, stats = build_artifact()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(payload)

    digest = hashlib.sha256(payload).hexdigest()
    for name, value in stats.items():
        print(f"{name}: {value:,}" if isinstance(value, int) else f"{name}: {value}")
    print(f"sha256: {digest}")
    print(f"written: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
