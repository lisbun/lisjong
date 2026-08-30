"""Exact structural completion / tenpai predicate artifact backend.

This private module owns the compact, read-only runtime representation used by
the package-internal contracts in :mod:`lisjong.hand_evaluation.shanten`.
Every base-5 suit / honor group key stores a one-byte ID.  The ID selects a
deduplicated pair of 16-bit masks from a small shared pool:

``completion mask``
    Exact local ``meld count / optional head`` decompositions for the group.

``one-added mask``
    Completion masks reachable after adding one structural tile to the group.

The four groups are combined exactly.  Completion uses only completion masks;
tenpai uses one one-added mask and completion masks for the other three groups.
Closed 13 / 14-tile special-hand dispatch remains in ``shanten.py`` so the
Policy never owns or imports this backend.

The separate artifact is loaded only on first predicate use.  Missing,
truncated, version-mismatched, size-inconsistent, or invalid-ID artifacts raise
``StructuralPredicateTableError``.  There is no numeric-shanten, frontier, or
recursive runtime fallback.
"""

import struct
import sys
from array import array
from collections.abc import Sequence
from importlib import resources

MAGIC = b"LISJSPT\x01"
"""Artifact magic, independent of ``_shanten_table.bin``."""

FORMAT_VERSION = 1
"""Artifact format version.  Increment whenever the binary layout changes."""

HEADER_FORMAT = "<8sIBB2x"
"""magic, format version, suit pair count, honor pair count, reserved padding."""

SUIT_KEY_SPACE = 5**9
HONOR_KEY_SPACE = 5**7
TABLE_RESOURCE = "_structural_predicate_table.bin"

_MELD_MASK = 0b11111
_HEAD_SHIFT = 5


class StructuralPredicateTableError(Exception):
    """The structural-predicate artifact is unavailable or internally invalid."""


def _build_meld_combine() -> tuple[int, ...]:
    """Build the 32 x 32 exact set-convolution table for meld-count masks."""
    values: list[int] = []
    for left in range(32):
        for right in range(32):
            combined = 0
            for left_melds in range(5):
                if not left & (1 << left_melds):
                    continue
                for right_melds in range(5 - left_melds):
                    if right & (1 << right_melds):
                        combined |= 1 << (left_melds + right_melds)
            values.append(combined)
    return tuple(values)


_MELD_COMBINE = _build_meld_combine()


def _combine_form_masks(left: int, right: int) -> int:
    """Combine two ``meld count / optional head`` form masks exactly."""
    meld_combine = _MELD_COMBINE
    left_without_head = left & _MELD_MASK
    right_without_head = right & _MELD_MASK
    without_head = meld_combine[(left_without_head << 5) | right_without_head]
    with_head = (
        meld_combine[((left >> _HEAD_SHIFT) << 5) | right_without_head]
        | meld_combine[(left_without_head << 5) | (right >> _HEAD_SHIFT)]
    )
    return without_head | (with_head << _HEAD_SHIFT)


class _StructuralPredicateTable:
    """Compact base-5 key lookup and shared mask-pair pools."""

    __slots__ = (
        "honor_ids",
        "honor_pair_count",
        "honor_pool",
        "suit_ids",
        "suit_pair_count",
        "suit_pool",
    )

    def __init__(self, payload: bytes) -> None:
        header_size = struct.calcsize(HEADER_FORMAT)
        if len(payload) < header_size:
            raise StructuralPredicateTableError(
                "structural predicate table artifact is truncated"
            )
        magic, version, suit_pair_count, honor_pair_count = struct.unpack_from(
            HEADER_FORMAT, payload
        )
        if magic != MAGIC:
            raise StructuralPredicateTableError(
                "structural predicate table artifact has an unexpected magic"
            )
        if version != FORMAT_VERSION:
            raise StructuralPredicateTableError(
                "structural predicate table artifact format version is "
                f"{version}, expected {FORMAT_VERSION}"
            )
        if suit_pair_count == 0 or honor_pair_count == 0:
            raise StructuralPredicateTableError(
                "structural predicate table artifact declares an empty mask-pair pool"
            )

        expected = (
            header_size
            + SUIT_KEY_SPACE
            + HONOR_KEY_SPACE
            + (suit_pair_count + honor_pair_count) * 4
        )
        if len(payload) != expected:
            raise StructuralPredicateTableError(
                "structural predicate table artifact size does not match its declared "
                f"dimensions: {len(payload)} bytes, expected {expected}"
            )

        offset = header_size

        def take(typecode: str, count: int, item_size: int) -> array:
            nonlocal offset
            values = array(typecode)
            values.frombytes(payload[offset : offset + count * item_size])
            offset += count * item_size
            if sys.byteorder == "big" and item_size > 1:
                values.byteswap()
            return values

        self.suit_ids = take("B", SUIT_KEY_SPACE, 1)
        self.honor_ids = take("B", HONOR_KEY_SPACE, 1)
        self.suit_pool = take("H", suit_pair_count * 2, 2)
        self.honor_pool = take("H", honor_pair_count * 2, 2)
        self.suit_pair_count = suit_pair_count
        self.honor_pair_count = honor_pair_count

    @staticmethod
    def _pair(
        key: int, ids: array, pool: array, pair_count: int, label: str
    ) -> tuple[int, int]:
        pair_id = ids[key]
        if pair_id >= pair_count:
            raise StructuralPredicateTableError(
                f"{label} key of the structural predicate table artifact references "
                "a mask pair that does not exist"
            )
        pool_index = pair_id * 2
        return pool[pool_index], pool[pool_index + 1]

    def suit_masks(self, key: int) -> tuple[int, int]:
        return self._pair(
            key, self.suit_ids, self.suit_pool, self.suit_pair_count, "suit"
        )

    def honor_masks(self, key: int) -> tuple[int, int]:
        return self._pair(
            key, self.honor_ids, self.honor_pool, self.honor_pair_count, "honor"
        )


def _load_table() -> _StructuralPredicateTable:
    try:
        payload = resources.files(__package__).joinpath(TABLE_RESOURCE).read_bytes()
    except (FileNotFoundError, OSError) as error:
        raise StructuralPredicateTableError(
            f"structural predicate table artifact {TABLE_RESOURCE!r} is missing from "
            "the lisjong.hand_evaluation package"
        ) from error
    return _StructuralPredicateTable(payload)


_TABLE: _StructuralPredicateTable | None = None


def _table() -> _StructuralPredicateTable:
    """Load the separate artifact once, on first predicate use."""
    global _TABLE
    table = _TABLE
    if table is None:
        table = _TABLE = _load_table()
    return table


def _group_keys(counts: Sequence[int]) -> tuple[int, int, int, int]:
    """Return unrolled base-5 keys for manzu, pinzu, souzu, and honors."""
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
    return manzu_key, pinzu_key, souzu_key, honor_key


def _mask_pairs(counts: Sequence[int]) -> tuple[tuple[int, int], ...]:
    table = _table()
    manzu, pinzu, souzu, honors = _group_keys(counts)
    return (
        table.suit_masks(manzu),
        table.suit_masks(pinzu),
        table.suit_masks(souzu),
        table.honor_masks(honors),
    )


def is_standard_structurally_complete(
    counts: Sequence[int], fixed_meld_count: int
) -> bool:
    """Return whether ``counts`` exactly form the remaining standard hand."""
    form = 1
    for completion_mask, _one_added_mask in _mask_pairs(counts):
        form = _combine_form_masks(form, completion_mask)
        if form == 0:
            return False
    required_concealed_melds = 4 - fixed_meld_count
    return bool(form & (1 << (_HEAD_SHIFT + required_concealed_melds)))


def is_standard_structurally_tenpai(
    counts: Sequence[int], fixed_meld_count: int
) -> bool:
    """Return whether adding one structural tile can complete the standard hand."""
    no_add = 1
    one_add = 0
    for completion_mask, one_added_mask in _mask_pairs(counts):
        one_add = _combine_form_masks(one_add, completion_mask) | _combine_form_masks(
            no_add, one_added_mask
        )
        no_add = _combine_form_masks(no_add, completion_mask)
    required_concealed_melds = 4 - fixed_meld_count
    return bool(one_add & (1 << (_HEAD_SHIFT + required_concealed_melds)))
