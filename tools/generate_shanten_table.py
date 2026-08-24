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


def _build_pool(kind_count: int, *, suited: bool, progress_label: str):
    """(key -> (start, count)) と共有poolを、frontier重複を畳んで構築する。"""
    keys = enumerate_group_keys(kind_count)
    pool: list[int] = []
    shared: dict[tuple[int, ...], tuple[int, int]] = {}
    spans: dict[int, tuple[int, int]] = {}

    started = time.perf_counter()
    for position, counts in enumerate(keys):
        frontier = dominant_frontier(local_frontier(counts, suited=suited))
        packed = tuple(
            sorted(
                _lookup_shanten.pack_entry(blocks_used, head_used, ms, hs, score)
                for (blocks_used, head_used, ms, hs), score in frontier.items()
            )
        )
        span = shared.get(packed)
        if span is None:
            span = (len(pool), len(packed))
            shared[packed] = span
            pool.extend(packed)
        spans[group_key(counts)] = span

        if position % 20000 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"  {progress_label}: {position:,}/{len(keys):,} "
                f"({elapsed:.0f}s, pool={len(pool):,}, distinct={len(shared):,})",
                flush=True,
            )

    return keys, spans, pool, shared, time.perf_counter() - started


def build_artifact() -> tuple[bytes, dict[str, int]]:
    suit_keys, suit_spans, suit_pool, suit_shared, suit_time = _build_pool(
        SUIT_KIND_COUNT, suited=True, progress_label="suit"
    )
    honor_keys, honor_spans, honor_pool, honor_shared, honor_time = _build_pool(
        HONOR_KIND_COUNT, suited=False, progress_label="honors"
    )

    header = struct.pack(
        _lookup_shanten.HEADER_FORMAT,
        _lookup_shanten.MAGIC,
        _lookup_shanten.FORMAT_VERSION,
        len(suit_keys),
        len(honor_keys),
        len(suit_pool),
        len(honor_pool),
    )

    chunks = [header]
    for keys, spans in ((suit_keys, suit_spans), (honor_keys, honor_spans)):
        starts = bytearray()
        counts = bytearray()
        for group_counts in keys:
            start, count = spans[group_key(group_counts)]
            starts += struct.pack("<I", start)
            counts += struct.pack("<B", count)
        chunks.append(bytes(starts))
        chunks.append(bytes(counts))
    for pool in (suit_pool, honor_pool):
        chunks.append(struct.pack(f"<{len(pool)}H", *pool))

    payload = b"".join(chunks)
    stats = {
        "suit_keys": len(suit_keys),
        "honor_keys": len(honor_keys),
        "suit_pool_entries": len(suit_pool),
        "honor_pool_entries": len(honor_pool),
        "suit_distinct_frontiers": len(suit_shared),
        "honor_distinct_frontiers": len(honor_shared),
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
