"""Independent exhaustive reference validation for Issue #141 metadata."""

import functools
import pathlib
import unittest

from lisjong.hand_evaluation import _shanten_frontier, _structural_predicates


@functools.cache
def _can_form_only_melds(counts: tuple[int, ...], suited: bool) -> bool:
    """Test-only exact decomposition independent of the artifact generator."""
    try:
        index = next(index for index, count in enumerate(counts) if count)
    except StopIteration:
        return True

    if counts[index] >= 3:
        reduced = list(counts)
        reduced[index] -= 3
        if _can_form_only_melds(tuple(reduced), suited):
            return True
    if suited and index + 2 < len(counts) and counts[index + 1] and counts[index + 2]:
        reduced = list(counts)
        reduced[index] -= 1
        reduced[index + 1] -= 1
        reduced[index + 2] -= 1
        if _can_form_only_melds(tuple(reduced), suited):
            return True
    return False


@functools.cache
def _reference_completion_mask(counts: tuple[int, ...], suited: bool) -> int:
    tile_count = sum(counts)
    mask = 0
    if tile_count % 3 == 0 and tile_count // 3 <= 4:
        if _can_form_only_melds(counts, suited):
            mask |= 1 << (tile_count // 3)

    if tile_count >= 2 and (tile_count - 2) % 3 == 0:
        melds = (tile_count - 2) // 3
        if melds <= 4:
            for index, count in enumerate(counts):
                if count < 2:
                    continue
                reduced = list(counts)
                reduced[index] -= 2
                if _can_form_only_melds(tuple(reduced), suited):
                    mask |= 1 << (5 + melds)
                    break
    return mask


class StructuralPredicateGroupExhaustiveTest(unittest.TestCase):
    def _assert_group(self, kind_count: int, *, suited: bool) -> int:
        payload = (
            pathlib.Path(_structural_predicates.__file__).parent
            / _structural_predicates.TABLE_RESOURCE
        ).read_bytes()
        table = _structural_predicates._StructuralPredicateTable(payload)
        keys = _shanten_frontier.enumerate_group_keys(kind_count)

        for counts in keys:
            key = _shanten_frontier.group_key(counts)
            actual_completion, actual_one_added = (
                table.suit_masks(key) if suited else table.honor_masks(key)
            )
            expected_completion = _reference_completion_mask(counts, suited)
            expected_one_added = 0
            if sum(counts) < 14:
                for index, count in enumerate(counts):
                    if count < 4:
                        added = list(counts)
                        added[index] += 1
                        expected_one_added |= _reference_completion_mask(
                            tuple(added), suited
                        )
            self.assertEqual(
                (actual_completion, actual_one_added),
                (expected_completion, expected_one_added),
                msg=f"suited={suited}; key={key}; counts={counts}",
            )
        return len(keys)

    def test_all_reachable_suit_and_honor_keys_match_independent_reference(
        self,
    ) -> None:
        self.assertEqual(
            self._assert_group(_shanten_frontier.SUIT_KIND_COUNT, suited=True),
            405_350,
        )
        self.assertEqual(
            self._assert_group(_shanten_frontier.HONOR_KIND_COUNT, suited=False),
            43_130,
        )


if __name__ == "__main__":
    unittest.main()
