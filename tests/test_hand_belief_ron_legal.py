import unittest
from dataclasses import FrozenInstanceError, replace

from lisjong.belief import HandBelief
from lisjong.belief.canonical_axes import tile_type_from_index
from lisjong.belief.exact_wait_ground_truth import exact_hand_belief_with_waits
from lisjong.belief.fixed_point import SCALE, probability_to_raw
from lisjong.belief.self_belief import exact_self_belief
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.tile import Tile

ZERO = (0,) * 34
FULL = (SCALE,) * 34
MECHANISMS = (
    "tanki_wait_probability_raw",
    "shanpon_wait_probability_raw",
    "kanchan_wait_probability_raw",
    "penchan_wait_probability_raw",
    "ryanmen_low_side_probability_raw",
    "ryanmen_high_side_probability_raw",
    "kokushi_wait_probability_raw",
)


def belief(**fields: object) -> HandBelief:
    return HandBelief(ZERO, (0, 0, 0), **fields)


class RonLegalBeliefTest(unittest.TestCase):
    def test_availability_for_every_structural_level(self) -> None:
        for level in (0, 1, 2):
            for ron in (None, ZERO):
                with self.subTest(level=level, ron_provided=ron is not None):
                    fields = {"ron_legal_probability_raw": ron}
                    if level:
                        fields["wait_probability_raw"] = ZERO
                    if level == 2:
                        fields.update({name: ZERO for name in MECHANISMS})
                    if level == 0 and ron is not None:
                        with self.assertRaisesRegex(ValueError, "wait_probability_raw"):
                            belief(**fields)
                        continue
                    result = belief(**fields)
                    self.assertEqual(result.has_wait_belief, level > 0)
                    self.assertEqual(result.has_wait_mechanism_belief, level == 2)
                    self.assertEqual(result.has_ron_legal_belief, ron is not None)
                    for index in range(34):
                        self.assertEqual(
                            result.ron_legal_probability(tile_type_from_index(index)),
                            None if ron is None else 0.0,
                        )

    def test_every_canonical_slot_and_raw_boundary(self) -> None:
        for index in range(34):
            for raw in (0, 1, SCALE // 2, SCALE - 1, SCALE):
                with self.subTest(index=index, raw=raw):
                    values = list(ZERO)
                    values[index] = raw
                    result = belief(
                        wait_probability_raw=tuple(values),
                        ron_legal_probability_raw=tuple(values),
                    )
                    self.assertEqual(
                        result.ron_legal_probability(tile_type_from_index(index)),
                        raw / SCALE,
                    )

    def test_canonical_axis_is_preserved(self) -> None:
        values = tuple(range(34))
        result = belief(wait_probability_raw=FULL, ron_legal_probability_raw=values)
        self.assertEqual(
            tuple(
                result.ron_legal_probability(tile_type_from_index(i)) for i in range(34)
            ),
            tuple(raw / SCALE for raw in values),
        )

    def test_probability_sum_can_exceed_one(self) -> None:
        result = belief(wait_probability_raw=FULL, ron_legal_probability_raw=FULL)
        self.assertEqual(sum(result.ron_legal_probability_raw), 34 * SCALE)

    def test_rejects_non_iterable_and_wrong_length(self) -> None:
        with self.assertRaises(TypeError):
            belief(wait_probability_raw=FULL, ron_legal_probability_raw=0)
        for length in (0, 33, 35):
            with self.subTest(length=length), self.assertRaises(ValueError):
                belief(
                    wait_probability_raw=FULL, ron_legal_probability_raw=(0,) * length
                )

    def test_rejects_non_integer_values(self) -> None:
        for value in (True, False, 0.0, 0.5, "0", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                belief(
                    wait_probability_raw=FULL,
                    ron_legal_probability_raw=(value,) + ZERO[1:],
                )

    def test_rejects_out_of_range_values(self) -> None:
        for value in (-1, SCALE + 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                belief(
                    wait_probability_raw=FULL,
                    ron_legal_probability_raw=(value,) + ZERO[1:],
                )

    def test_rejects_one_raw_unit_excess_in_every_slot(self) -> None:
        for index in range(34):
            for wait in (0, SCALE // 2, SCALE - 1):
                with self.subTest(index=index, wait=wait):
                    waits, rons = list(FULL), list(ZERO)
                    waits[index], rons[index] = wait, wait + 1
                    with self.assertRaisesRegex(ValueError, "must not exceed"):
                        belief(
                            wait_probability_raw=tuple(waits),
                            ron_legal_probability_raw=tuple(rons),
                        )

    def test_validates_quantized_values_without_tolerance(self) -> None:
        # Half-to-even can map different semantic inputs to the same raw value.
        wait = probability_to_raw(100.0 / SCALE)
        same_raw = probability_to_raw(100.5 / SCALE)
        next_raw = probability_to_raw(100.5001 / SCALE)
        result = belief(
            wait_probability_raw=(wait,) + ZERO[1:],
            ron_legal_probability_raw=(same_raw,) + ZERO[1:],
        )
        self.assertEqual(result.ron_legal_probability_raw[0], wait)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            replace(result, ron_legal_probability_raw=(next_raw,) + ZERO[1:])

    def test_replace_revalidates_wait_and_ron(self) -> None:
        result = belief(wait_probability_raw=FULL, ron_legal_probability_raw=FULL)
        for waits in (None, ZERO):
            with self.subTest(waits=waits), self.assertRaises(ValueError):
                replace(result, wait_probability_raw=waits)

    def test_ron_does_not_change_mechanism_contract(self) -> None:
        with self.assertRaises(ValueError):
            belief(
                wait_probability_raw=FULL,
                ron_legal_probability_raw=ZERO,
                tanki_wait_probability_raw=ZERO,
            )
        # The existing mechanism marginals are not bounded by primary wait.
        result = belief(
            wait_probability_raw=ZERO,
            ron_legal_probability_raw=ZERO,
            **{name: FULL if name == MECHANISMS[0] else ZERO for name in MECHANISMS},
        )
        self.assertEqual(result.tanki_wait_probability(tile_type_from_index(0)), 1.0)
        self.assertEqual(result.ron_legal_probability(tile_type_from_index(0)), 0.0)

    def test_normalizes_to_immutable_tuple(self) -> None:
        values = list(ZERO)
        values[0] = SCALE
        result = belief(wait_probability_raw=FULL, ron_legal_probability_raw=values)
        values[0] = 0
        self.assertIsInstance(result.ron_legal_probability_raw, tuple)
        self.assertEqual(result.ron_legal_probability_raw[0], SCALE)
        with self.assertRaises(FrozenInstanceError):
            result.ron_legal_probability_raw = ZERO

    def test_accessor_rejects_invalid_tile_when_provided_or_unavailable(self) -> None:
        for result in (
            belief(),
            belief(wait_probability_raw=ZERO, ron_legal_probability_raw=ZERO),
        ):
            with self.assertRaises(TypeError):
                result.ron_legal_probability("1m")

    def test_existing_positional_constructor_is_compatible(self) -> None:
        for result in (
            HandBelief(ZERO, (0, 0, 0)),
            HandBelief(ZERO, (0, 0, 0), ZERO, *(ZERO for _ in MECHANISMS)),
        ):
            self.assertIsNone(result.ron_legal_probability_raw)
            self.assertFalse(result.has_ron_legal_belief)

    def test_existing_exact_builders_keep_ron_unavailable(self) -> None:
        hand = tuple(
            Tile(tile_type_from_index(i))
            for i in (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6)
        )
        for result in (
            exact_self_belief(OwnHandState(concealed_tiles=hand, drawn_tile=None)),
            exact_hand_belief_with_waits(hand),
        ):
            self.assertFalse(result.has_ron_legal_belief)
            self.assertIsNone(result.ron_legal_probability(tile_type_from_index(6)))


if __name__ == "__main__":
    unittest.main()
