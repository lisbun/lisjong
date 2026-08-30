"""Issue #141 structural completion / tenpai predicate tests."""

import pathlib
import random
import struct
import unittest
from unittest.mock import patch

from lisjong.hand_evaluation import _structural_predicates
from lisjong.hand_evaluation.shanten import (
    calculate_shanten_from_canonical_counts,
    is_structurally_complete_from_canonical_counts,
    is_structurally_tenpai_from_canonical_counts,
)

_PACKAGE_ROOT = pathlib.Path(_structural_predicates.__file__).parent
_OFFSETS = {"m": 0, "p": 9, "s": 18, "z": 27}


def _counts(notation: str) -> tuple[int, ...]:
    counts = [0] * 34
    ranks = ""
    for character in notation:
        if character.isdigit():
            ranks += character
            continue
        offset = _OFFSETS[character]
        for rank in ranks:
            counts[offset + int(rank) - 1] += 1
        ranks = ""
    if ranks:
        raise ValueError(f"notation has trailing ranks: {notation!r}")
    return tuple(counts)


class StructuralPredicateSemanticTest(unittest.TestCase):
    def _assert_completion(self, notation: str) -> None:
        counts = _counts(notation)
        self.assertEqual(calculate_shanten_from_canonical_counts(counts), -1)
        self.assertTrue(is_structurally_complete_from_canonical_counts(counts))

    def _assert_tenpai(self, notation: str) -> None:
        counts = _counts(notation)
        self.assertEqual(calculate_shanten_from_canonical_counts(counts), 0)
        self.assertTrue(is_structurally_tenpai_from_canonical_counts(counts))

    def test_closed_standard_completion_and_tenpai(self) -> None:
        self._assert_completion("123m123p123s11122z")
        self._assert_tenpai("123m123p123s1112z")

    def test_chiitoitsu_completion_and_tenpai(self) -> None:
        self._assert_completion("11m22m33p44p55s66s77z")
        self._assert_tenpai("11m22m33p44p55s66s7z")

    def test_kokushi_completion_and_tenpai(self) -> None:
        self._assert_completion("19m19p19s12345677z")
        self._assert_tenpai("19m19p19s1234567z")

    def test_every_fixed_meld_count_completion_and_tenpai(self) -> None:
        cases = {
            1: ("123m123p123s11z", "123m123p123s1z"),
            2: ("123m123p11z", "123m123p1z"),
            3: ("123m11z", "123m1z"),
            4: ("11z", "1z"),
        }
        for fixed_melds, (completion, tenpai) in cases.items():
            with self.subTest(fixed_melds=fixed_melds):
                self._assert_completion(completion)
                self._assert_tenpai(tenpai)

    def test_seeded_physical_corpus_matches_canonical_shanten(self) -> None:
        generator = random.Random(20260830)
        physical_tiles = [index for index in range(34) for _ in range(4)]
        for size in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14):
            for sample_index in range(100):
                sample = generator.sample(physical_tiles, size)
                counts = tuple(sample.count(index) for index in range(34))
                shanten = calculate_shanten_from_canonical_counts(counts)
                with self.subTest(size=size, sample_index=sample_index):
                    if size in (2, 5, 8, 11, 14):
                        self.assertEqual(
                            is_structurally_complete_from_canonical_counts(counts),
                            shanten == -1,
                        )
                    else:
                        self.assertEqual(
                            is_structurally_tenpai_from_canonical_counts(counts),
                            shanten == 0,
                        )


class StructuralPredicateArtifactTest(unittest.TestCase):
    @staticmethod
    def _payload() -> bytes:
        return (_PACKAGE_ROOT / _structural_predicates.TABLE_RESOURCE).read_bytes()

    def test_artifact_header_and_selected_representation(self) -> None:
        payload = self._payload()
        table = _structural_predicates._StructuralPredicateTable(payload)

        self.assertTrue(payload.startswith(_structural_predicates.MAGIC))
        self.assertEqual(table.suit_pair_count, 22)
        self.assertEqual(table.honor_pair_count, 16)
        self.assertEqual(
            len(payload),
            struct.calcsize(_structural_predicates.HEADER_FORMAT)
            + _structural_predicates.SUIT_KEY_SPACE
            + _structural_predicates.HONOR_KEY_SPACE
            + (table.suit_pair_count + table.honor_pair_count) * 4,
        )

    def test_missing_artifact_fails_closed_with_cause(self) -> None:
        with patch.object(
            _structural_predicates.resources,
            "files",
            side_effect=FileNotFoundError("missing test resource"),
        ):
            with self.assertRaises(
                _structural_predicates.StructuralPredicateTableError
            ) as raised:
                _structural_predicates._load_table()

        self.assertIsInstance(raised.exception.__cause__, FileNotFoundError)

    def test_truncated_magic_version_and_declared_size_fail_closed(self) -> None:
        payload = bytearray(self._payload())
        corruptions = {
            "truncated": bytes(payload[:8]),
            "wrong magic": b"NOTLISJ!" + bytes(payload[8:]),
            "wrong version": (
                bytes(payload[:8])
                + (_structural_predicates.FORMAT_VERSION + 1).to_bytes(4, "little")
                + bytes(payload[12:])
            ),
            "declared size": bytes(
                payload[:12] + bytes([payload[12] + 1]) + payload[13:]
            ),
        }
        for label, corrupted in corruptions.items():
            with self.subTest(label=label):
                with self.assertRaises(
                    _structural_predicates.StructuralPredicateTableError
                ):
                    _structural_predicates._StructuralPredicateTable(corrupted)

    def test_invalid_suit_mask_pair_id_fails_when_referenced(self) -> None:
        payload = bytearray(self._payload())
        header_size = struct.calcsize(_structural_predicates.HEADER_FORMAT)
        payload[header_size] = 255
        table = _structural_predicates._StructuralPredicateTable(bytes(payload))

        with self.assertRaisesRegex(
            _structural_predicates.StructuralPredicateTableError,
            "mask pair that does not exist",
        ):
            table.suit_masks(0)

    def test_invalid_honor_pool_reference_fails_when_referenced(self) -> None:
        payload = bytearray(self._payload())
        header_size = struct.calcsize(_structural_predicates.HEADER_FORMAT)
        honor_zero = header_size + _structural_predicates.SUIT_KEY_SPACE
        payload[honor_zero] = 255
        table = _structural_predicates._StructuralPredicateTable(bytes(payload))

        with self.assertRaisesRegex(
            _structural_predicates.StructuralPredicateTableError,
            "mask pair that does not exist",
        ):
            table.honor_masks(0)

    def test_numeric_shanten_does_not_load_predicate_artifact(self) -> None:
        counts = _counts("123m123p123s11122z")
        with patch.object(_structural_predicates, "_TABLE", None):
            self.assertEqual(calculate_shanten_from_canonical_counts(counts), -1)
            self.assertIsNone(_structural_predicates._TABLE)

    def test_first_predicate_call_lazy_loads_artifact(self) -> None:
        counts = _counts("123m123p123s11122z")
        with patch.object(_structural_predicates, "_TABLE", None):
            self.assertTrue(is_structurally_complete_from_canonical_counts(counts))
            self.assertIsInstance(
                _structural_predicates._TABLE,
                _structural_predicates._StructuralPredicateTable,
            )


if __name__ == "__main__":
    unittest.main()
