"""Generate the deterministic compact structural-predicate artifact.

Usage::

    python tools/generate_structural_predicate_table.py \
        src/lisjong/hand_evaluation/_structural_predicate_table.bin

The existing ``_shanten_frontier`` semantics remain the source of truth.  This
generator reads the committed ``_shanten_table.bin`` frontier entries produced
from that source instead of repeating the expensive frontier recursion for all
448,480 reachable group keys.  One-added masks are derived from the resulting
completion masks of neighboring keys.  Independent reference validation does
not share this extraction logic.
"""

import argparse
import hashlib
import pathlib
import struct
import sys
import time
from array import array

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lisjong.hand_evaluation import (  # noqa: E402
    _lookup_shanten,
    _structural_predicates,
)
from lisjong.hand_evaluation._shanten_frontier import (  # noqa: E402
    HONOR_KIND_COUNT,
    SUIT_KIND_COUNT,
    enumerate_group_keys,
    group_key,
)

_DEFAULT_SHANTEN_TABLE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "lisjong"
    / "hand_evaluation"
    / _lookup_shanten.TABLE_RESOURCE
)


def _completion_mask(
    table: _lookup_shanten._ShantenTable,
    *,
    key: int,
    tile_count: int,
    suited: bool,
    cache: dict[tuple[int, int], int],
) -> int:
    """Extract exact local forms from an existing generated frontier entry."""
    if suited:
        frontier_id = table.suit_ids[key]
        frontier_count = table.suit_frontier_count
        starts = table.suit_starts
        counts = table.suit_counts
        pool = table.suit_pool
    else:
        frontier_id = table.honor_ids[key]
        frontier_count = table.honor_frontier_count
        starts = table.honor_starts
        counts = table.honor_counts
        pool = table.honor_pool
    if frontier_id >= frontier_count:
        raise _lookup_shanten.ShantenTableError(
            "source shanten table references a frontier that does not exist"
        )

    cache_key = (frontier_id, tile_count)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    mask = 0
    start = starts[frontier_id]
    for position in range(start, start + counts[frontier_id]):
        packed = pool[position]
        blocks, head, _meld_seeds, _head_seeds = _lookup_shanten._decode_state(
            packed >> _lookup_shanten._SCORE_SHIFT
        )
        score = packed & _lookup_shanten._SCORE_MASK
        if tile_count != 3 * blocks + 2 * head:
            continue
        if score != 2 * blocks + head:
            continue
        mask |= 1 << (blocks + 5 * head)
    cache[cache_key] = mask
    return mask


def _build_group(
    table: _lookup_shanten._ShantenTable,
    kind_count: int,
    *,
    suited: bool,
    progress_label: str,
    progress: bool,
) -> tuple[bytes, tuple[tuple[int, int], ...], int, float]:
    keys = enumerate_group_keys(kind_count)
    key_space = 5**kind_count
    completion_masks = array("H", [0]) * key_space
    weights = tuple(5 ** (kind_count - index - 1) for index in range(kind_count))
    started = time.perf_counter()
    completion_cache: dict[tuple[int, int], int] = {}

    for position, counts in enumerate(keys):
        key = group_key(counts)
        completion_masks[key] = _completion_mask(
            table,
            key=key,
            tile_count=sum(counts),
            suited=suited,
            cache=completion_cache,
        )
        if progress and position % 20000 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"  {progress_label} completion: {position:,}/{len(keys):,} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    pairs_by_key: list[tuple[int, int] | None] = [None] * key_space
    pairs_by_key[0] = (0, 0)
    for counts in keys:
        key = group_key(counts)
        one_added = 0
        if sum(counts) < 14:
            for index, count in enumerate(counts):
                if count < 4:
                    one_added |= completion_masks[key + weights[index]]
        pairs_by_key[key] = (completion_masks[key], one_added)

    # Unreachable dense keys use the all-zero sentinel.  Sorting makes pair IDs
    # independent of dict/hash iteration and therefore byte-reproducible.
    pairs = tuple(sorted({pair or (0, 0) for pair in pairs_by_key}))
    if len(pairs) > 256:
        raise ValueError(
            f"{progress_label} has {len(pairs)} mask pairs; one-byte IDs support 256"
        )
    pair_ids = {pair: pair_id for pair_id, pair in enumerate(pairs)}
    ids = bytes(pair_ids[pair or (0, 0)] for pair in pairs_by_key)
    return ids, pairs, len(keys), time.perf_counter() - started


def build_artifact(
    shanten_table_path: pathlib.Path = _DEFAULT_SHANTEN_TABLE,
    *,
    progress: bool = True,
) -> tuple[bytes, dict[str, int | float]]:
    source_table = _lookup_shanten._ShantenTable(shanten_table_path.read_bytes())
    suit_ids, suit_pairs, suit_reachable, suit_time = _build_group(
        source_table,
        SUIT_KIND_COUNT,
        suited=True,
        progress_label="suit",
        progress=progress,
    )
    honor_ids, honor_pairs, honor_reachable, honor_time = _build_group(
        source_table,
        HONOR_KIND_COUNT,
        suited=False,
        progress_label="honor",
        progress=progress,
    )
    header = struct.pack(
        _structural_predicates.HEADER_FORMAT,
        _structural_predicates.MAGIC,
        _structural_predicates.FORMAT_VERSION,
        len(suit_pairs),
        len(honor_pairs),
    )
    chunks = [header, suit_ids, honor_ids]
    for pairs in (suit_pairs, honor_pairs):
        chunks.append(
            struct.pack(
                f"<{len(pairs) * 2}H",
                *(mask for pair in pairs for mask in pair),
            )
        )
    payload = b"".join(chunks)
    return payload, {
        "suit_key_space": len(suit_ids),
        "honor_key_space": len(honor_ids),
        "suit_reachable_keys": suit_reachable,
        "honor_reachable_keys": honor_reachable,
        "suit_mask_pairs": len(suit_pairs),
        "honor_mask_pairs": len(honor_pairs),
        "suit_generation_seconds": round(suit_time, 1),
        "honor_generation_seconds": round(honor_time, 1),
        "bytes": len(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--shanten-table",
        type=pathlib.Path,
        default=_DEFAULT_SHANTEN_TABLE,
        help="source _shanten_table.bin generated from _shanten_frontier",
    )
    arguments = parser.parse_args()
    payload, stats = build_artifact(arguments.shanten_table)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(payload)

    for name, value in stats.items():
        print(f"{name}: {value:,}" if isinstance(value, int) else f"{name}: {value}")
    print(f"sha256: {hashlib.sha256(payload).hexdigest()}")
    print(f"written: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
