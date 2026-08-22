import unittest

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.fixed_point import PROBABILITY_MAX_RAW, SCALE, probability_to_raw
from lisjong.belief.hand_belief import HandBelief
from lisjong.belief.self_belief import exact_self_belief
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.tile import (
    EAST_WIND,
    GREEN_DRAGON,
    NORTH_WIND,
    RED_DRAGON,
    SOUTH_WIND,
    WEST_WIND,
    WHITE_DRAGON,
    Tile,
    TileCategory,
    TileType,
)

SUITED_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)

HONOR_TILE_TYPES = (
    EAST_WIND,
    SOUTH_WIND,
    WEST_WIND,
    NORTH_WIND,
    WHITE_DRAGON,
    GREEN_DRAGON,
    RED_DRAGON,
)

ALL_TILE_TYPES = (
    tuple(
        TileType(category, rank)
        for category in SUITED_CATEGORIES
        for rank in range(1, 10)
    )
    + HONOR_TILE_TYPES
)

TERMINAL_AND_HONOR_TILE_TYPES = (
    tuple(TileType(category, rank) for category in SUITED_CATEGORIES for rank in (1, 9))
    + HONOR_TILE_TYPES
)

# canonical wait mechanism group（HandBeliefのall-or-none availability単位）。
MECHANISM_FIELD_NAMES = (
    "tanki_wait_probability_raw",
    "shanpon_wait_probability_raw",
    "kanchan_wait_probability_raw",
    "penchan_wait_probability_raw",
    "ryanmen_low_side_probability_raw",
    "ryanmen_high_side_probability_raw",
    "kokushi_wait_probability_raw",
)

# 各mechanism tableがnon-zeroを取り得るtile type（それ以外はcanonical zero）。
MECHANISM_VALID_TILE_TYPES = {
    "tanki_wait_probability_raw": ALL_TILE_TYPES,
    "shanpon_wait_probability_raw": ALL_TILE_TYPES,
    "kanchan_wait_probability_raw": tuple(
        TileType(category, rank)
        for category in SUITED_CATEGORIES
        for rank in range(2, 9)
    ),
    "penchan_wait_probability_raw": tuple(
        TileType(category, rank) for category in SUITED_CATEGORIES for rank in (3, 7)
    ),
    "ryanmen_low_side_probability_raw": tuple(
        TileType(category, rank)
        for category in SUITED_CATEGORIES
        for rank in range(1, 7)
    ),
    "ryanmen_high_side_probability_raw": tuple(
        TileType(category, rank)
        for category in SUITED_CATEGORIES
        for rank in range(4, 10)
    ),
    "kokushi_wait_probability_raw": TERMINAL_AND_HONOR_TILE_TYPES,
}


def _empty_table() -> tuple[int, ...]:
    return tuple(0 for _ in range(34))


def _table(overrides: dict[TileType, int]) -> tuple[int, ...]:
    values = [0] * 34
    for tile_type, raw in overrides.items():
        values[tile_type_index(tile_type)] = raw
    return tuple(values)


def _mechanism_tables(**overrides: object) -> dict[str, object]:
    tables: dict[str, object] = {name: _empty_table() for name in MECHANISM_FIELD_NAMES}
    tables.update(overrides)
    return tables


def _belief(**wait_fields: object) -> HandBelief:
    return HandBelief(
        expected_count_raw=_empty_table(),
        red_five_probability_raw=(0, 0, 0),
        **wait_fields,
    )


def _level_two_belief(
    wait_probability_raw: tuple[int, ...] | None = None, **overrides: object
) -> HandBelief:
    if wait_probability_raw is None:
        wait_probability_raw = _empty_table()
    return _belief(
        wait_probability_raw=wait_probability_raw, **_mechanism_tables(**overrides)
    )


class WaitBeliefAvailabilityTest(unittest.TestCase):
    def test_level_0_omits_wait_and_mechanism_tables(self) -> None:
        belief = _belief()

        self.assertFalse(belief.has_wait_belief)
        self.assertFalse(belief.has_wait_mechanism_belief)
        self.assertIsNone(belief.wait_probability_raw)
        for field_name in MECHANISM_FIELD_NAMES:
            self.assertIsNone(getattr(belief, field_name))

    def test_level_1_provides_wait_only(self) -> None:
        belief = _belief(wait_probability_raw=_empty_table())

        self.assertTrue(belief.has_wait_belief)
        self.assertFalse(belief.has_wait_mechanism_belief)
        self.assertEqual(belief.wait_probability_raw, _empty_table())
        for field_name in MECHANISM_FIELD_NAMES:
            self.assertIsNone(getattr(belief, field_name))

    def test_level_2_provides_wait_and_every_mechanism(self) -> None:
        belief = _level_two_belief()

        self.assertTrue(belief.has_wait_belief)
        self.assertTrue(belief.has_wait_mechanism_belief)
        for field_name in MECHANISM_FIELD_NAMES:
            self.assertEqual(getattr(belief, field_name), _empty_table())

    def test_rejects_partial_mechanism_group(self) -> None:
        for field_name in MECHANISM_FIELD_NAMES:
            with self.subTest(provided=field_name):
                with self.assertRaises(ValueError):
                    _belief(
                        wait_probability_raw=_empty_table(),
                        **{field_name: _empty_table()},
                    )

    def test_rejects_mechanism_group_missing_one_table(self) -> None:
        for omitted in MECHANISM_FIELD_NAMES:
            tables = _mechanism_tables(**{omitted: None})
            with self.subTest(omitted=omitted):
                with self.assertRaises(ValueError):
                    _belief(wait_probability_raw=_empty_table(), **tables)

    def test_rejects_mechanism_group_without_wait_probability(self) -> None:
        with self.assertRaises(ValueError):
            _belief(**_mechanism_tables())


class WaitBeliefTableValidationTest(unittest.TestCase):
    def _wait_field_names(self) -> tuple[str, ...]:
        return ("wait_probability_raw", *MECHANISM_FIELD_NAMES)

    def _level_two_with(self, field_name: str, values: object) -> HandBelief:
        if field_name == "wait_probability_raw":
            return _belief(wait_probability_raw=values, **_mechanism_tables())
        return _belief(
            wait_probability_raw=_empty_table(),
            **_mechanism_tables(**{field_name: values}),
        )

    def test_requires_length_34(self) -> None:
        for field_name in self._wait_field_names():
            for length in (33, 35):
                with self.subTest(field=field_name, length=length):
                    with self.assertRaises(ValueError):
                        self._level_two_with(
                            field_name, tuple(0 for _ in range(length))
                        )

    def test_accepts_raw_range_boundaries(self) -> None:
        manzu_5 = TileType(TileCategory.MANZU, 5)
        for raw in (0, 1, SCALE // 2, PROBABILITY_MAX_RAW):
            with self.subTest(raw=raw):
                belief = _belief(wait_probability_raw=_table({manzu_5: raw}))
                self.assertEqual(belief.wait_probability(manzu_5), raw / SCALE)

    def test_rejects_raw_above_probability_max(self) -> None:
        for field_name in self._wait_field_names():
            # 範囲validationだけを確認するため、その channel でvalidなslotへ置く。
            tile_type = MECHANISM_VALID_TILE_TYPES.get(field_name, ALL_TILE_TYPES)[0]
            with self.subTest(field=field_name):
                with self.assertRaises(ValueError):
                    self._level_two_with(
                        field_name, _table({tile_type: PROBABILITY_MAX_RAW + 1})
                    )

    def test_rejects_negative_raw(self) -> None:
        manzu_5 = TileType(TileCategory.MANZU, 5)
        with self.assertRaises(ValueError):
            _belief(wait_probability_raw=_table({manzu_5: -1}))

    def test_rejects_bool_values(self) -> None:
        values = list(_empty_table())
        values[0] = True
        for field_name in self._wait_field_names():
            with self.subTest(field=field_name):
                with self.assertRaises(TypeError):
                    self._level_two_with(field_name, tuple(values))

    def test_rejects_non_int_values(self) -> None:
        values = list(_empty_table())
        values[0] = 0.5
        with self.assertRaises(TypeError):
            _belief(wait_probability_raw=tuple(values))

    def test_rejects_non_iterable_table(self) -> None:
        with self.assertRaises(TypeError):
            _belief(wait_probability_raw=0)

    def test_all_zero_table_is_valid(self) -> None:
        belief = _level_two_belief()

        for tile_type in ALL_TILE_TYPES:
            self.assertEqual(belief.wait_probability(tile_type), 0.0)
            self.assertEqual(belief.tanki_wait_probability(tile_type), 0.0)
            self.assertEqual(belief.kokushi_wait_probability(tile_type), 0.0)

    def test_table_sum_above_one_is_valid(self) -> None:
        full = tuple(PROBABILITY_MAX_RAW for _ in range(34))
        belief = _belief(wait_probability_raw=full)

        self.assertEqual(sum(belief.wait_probability_raw), 34 * PROBABILITY_MAX_RAW)
        for tile_type in ALL_TILE_TYPES:
            self.assertEqual(belief.wait_probability(tile_type), 1.0)


class WaitBeliefValidSlotTest(unittest.TestCase):
    def _assert_slot_contract(self, field_name: str) -> None:
        valid_tile_types = set(MECHANISM_VALID_TILE_TYPES[field_name])
        for tile_type in ALL_TILE_TYPES:
            tables = _mechanism_tables(
                **{field_name: _table({tile_type: PROBABILITY_MAX_RAW})}
            )
            with self.subTest(field=field_name, tile_type=tile_type):
                if tile_type in valid_tile_types:
                    belief = _belief(wait_probability_raw=_empty_table(), **tables)
                    self.assertEqual(
                        getattr(belief, field_name)[tile_type_index(tile_type)],
                        PROBABILITY_MAX_RAW,
                    )
                else:
                    with self.assertRaises(ValueError):
                        _belief(wait_probability_raw=_empty_table(), **tables)

    def test_wait_accepts_every_tile_type(self) -> None:
        for tile_type in ALL_TILE_TYPES:
            with self.subTest(tile_type=tile_type):
                belief = _belief(
                    wait_probability_raw=_table({tile_type: PROBABILITY_MAX_RAW})
                )
                self.assertEqual(belief.wait_probability(tile_type), 1.0)

    def test_tanki_accepts_every_tile_type(self) -> None:
        self._assert_slot_contract("tanki_wait_probability_raw")

    def test_shanpon_accepts_every_tile_type(self) -> None:
        self._assert_slot_contract("shanpon_wait_probability_raw")

    def test_kanchan_allows_only_ranks_2_to_8(self) -> None:
        self._assert_slot_contract("kanchan_wait_probability_raw")

    def test_penchan_allows_only_ranks_3_and_7(self) -> None:
        self._assert_slot_contract("penchan_wait_probability_raw")

    def test_ryanmen_low_side_allows_only_ranks_1_to_6(self) -> None:
        self._assert_slot_contract("ryanmen_low_side_probability_raw")

    def test_ryanmen_high_side_allows_only_ranks_4_to_9(self) -> None:
        self._assert_slot_contract("ryanmen_high_side_probability_raw")

    def test_kokushi_allows_only_terminals_and_honors(self) -> None:
        self._assert_slot_contract("kokushi_wait_probability_raw")

    def test_invalid_slot_stays_canonical_zero(self) -> None:
        belief = _level_two_belief()
        manzu_1 = TileType(TileCategory.MANZU, 1)
        manzu_5 = TileType(TileCategory.MANZU, 5)

        self.assertEqual(belief.kanchan_wait_probability(manzu_1), 0.0)
        self.assertEqual(belief.penchan_wait_probability(manzu_1), 0.0)
        self.assertEqual(belief.kokushi_wait_probability(manzu_5), 0.0)


class WaitBeliefSemanticsTest(unittest.TestCase):
    def test_ryanmen_sides_are_distinguished(self) -> None:
        # 2m3m -> 1m（low side） / 4m（high side）
        manzu_1 = TileType(TileCategory.MANZU, 1)
        manzu_4 = TileType(TileCategory.MANZU, 4)
        belief = _level_two_belief(
            ryanmen_low_side_probability_raw=_table({manzu_1: PROBABILITY_MAX_RAW}),
            ryanmen_high_side_probability_raw=_table({manzu_4: PROBABILITY_MAX_RAW}),
        )

        self.assertEqual(belief.ryanmen_low_side_probability(manzu_1), 1.0)
        self.assertEqual(belief.ryanmen_high_side_probability(manzu_1), 0.0)
        self.assertEqual(belief.ryanmen_high_side_probability(manzu_4), 1.0)
        self.assertEqual(belief.ryanmen_low_side_probability(manzu_4), 0.0)

    def test_multiple_mechanisms_may_be_non_zero_for_one_tile(self) -> None:
        manzu_4 = TileType(TileCategory.MANZU, 4)
        belief = _level_two_belief(
            tanki_wait_probability_raw=_table({manzu_4: PROBABILITY_MAX_RAW}),
            shanpon_wait_probability_raw=_table({manzu_4: PROBABILITY_MAX_RAW}),
            kanchan_wait_probability_raw=_table({manzu_4: PROBABILITY_MAX_RAW}),
            ryanmen_high_side_probability_raw=_table({manzu_4: PROBABILITY_MAX_RAW}),
        )

        self.assertEqual(belief.tanki_wait_probability(manzu_4), 1.0)
        self.assertEqual(belief.shanpon_wait_probability(manzu_4), 1.0)
        self.assertEqual(belief.kanchan_wait_probability(manzu_4), 1.0)
        self.assertEqual(belief.ryanmen_high_side_probability(manzu_4), 1.0)

    def test_mechanism_marginals_are_not_constrained_by_wait(self) -> None:
        # marginal probability間にsum / max / OR制約を課さない。
        manzu_4 = TileType(TileCategory.MANZU, 4)
        belief = _belief(
            wait_probability_raw=_table({manzu_4: probability_to_raw(0.25)}),
            **_mechanism_tables(
                tanki_wait_probability_raw=_table({manzu_4: PROBABILITY_MAX_RAW}),
                shanpon_wait_probability_raw=_table({manzu_4: PROBABILITY_MAX_RAW}),
            ),
        )

        self.assertEqual(belief.wait_probability(manzu_4), 0.25)
        self.assertEqual(belief.tanki_wait_probability(manzu_4), 1.0)
        self.assertEqual(belief.shanpon_wait_probability(manzu_4), 1.0)

    def test_zero_wait_with_non_zero_mechanism_is_accepted(self) -> None:
        manzu_4 = TileType(TileCategory.MANZU, 4)
        belief = _level_two_belief(
            tanki_wait_probability_raw=_table({manzu_4: PROBABILITY_MAX_RAW})
        )

        self.assertEqual(belief.wait_probability(manzu_4), 0.0)
        self.assertEqual(belief.tanki_wait_probability(manzu_4), 1.0)

    def test_kokushi_thirteen_sided_wait_sets_every_terminal_and_honor(self) -> None:
        table = _table(
            {
                tile_type: PROBABILITY_MAX_RAW
                for tile_type in TERMINAL_AND_HONOR_TILE_TYPES
            }
        )
        belief = _level_two_belief(
            wait_probability_raw=table,
            kokushi_wait_probability_raw=table,
            tanki_wait_probability_raw=_empty_table(),
        )

        for tile_type in ALL_TILE_TYPES:
            expected = 1.0 if tile_type in TERMINAL_AND_HONOR_TILE_TYPES else 0.0
            self.assertEqual(belief.kokushi_wait_probability(tile_type), expected)
            # 国士待ちはtankiへ包含しない。
            self.assertEqual(belief.tanki_wait_probability(tile_type), 0.0)

    def test_seven_pairs_wait_is_expressed_as_tanki(self) -> None:
        # 11m 22m 33p 44p 55s 66s 東 の七対子聴牌はtanki channelで表す。
        belief = _level_two_belief(
            wait_probability_raw=_table({EAST_WIND: PROBABILITY_MAX_RAW}),
            tanki_wait_probability_raw=_table({EAST_WIND: PROBABILITY_MAX_RAW}),
        )

        self.assertEqual(belief.wait_probability(EAST_WIND), 1.0)
        self.assertEqual(belief.tanki_wait_probability(EAST_WIND), 1.0)
        self.assertEqual(belief.kokushi_wait_probability(EAST_WIND), 0.0)


class WaitBeliefSemanticAccessorTest(unittest.TestCase):
    def test_accessors_return_fixed_point_values(self) -> None:
        manzu_3 = TileType(TileCategory.MANZU, 3)
        raw = probability_to_raw(0.5)
        belief = _level_two_belief(
            kanchan_wait_probability_raw=_table({manzu_3: raw}),
            penchan_wait_probability_raw=_table({manzu_3: raw}),
        )

        self.assertEqual(belief.kanchan_wait_probability(manzu_3), 0.5)
        self.assertEqual(belief.penchan_wait_probability(manzu_3), 0.5)

    def test_unavailable_feature_returns_none_not_zero(self) -> None:
        manzu_3 = TileType(TileCategory.MANZU, 3)
        level_0 = _belief()
        level_1 = _belief(wait_probability_raw=_empty_table())

        self.assertIsNone(level_0.wait_probability(manzu_3))
        self.assertIsNone(level_0.tanki_wait_probability(manzu_3))
        self.assertIsNone(level_1.tanki_wait_probability(manzu_3))
        self.assertEqual(level_1.wait_probability(manzu_3), 0.0)
        self.assertIsNotNone(level_1.wait_probability(manzu_3))

    def test_accessors_reject_non_tile_type_even_when_unavailable(self) -> None:
        belief = _belief()

        with self.assertRaises(TypeError):
            belief.wait_probability("1m")
        with self.assertRaises(TypeError):
            belief.kokushi_wait_probability("1m")


class WaitBeliefBackwardCompatibilityTest(unittest.TestCase):
    def test_existing_constructor_call_still_works(self) -> None:
        belief = HandBelief(
            expected_count_raw=_empty_table(),
            red_five_probability_raw=(0, 0, 0),
        )

        self.assertEqual(belief.expected_count(TileType(TileCategory.MANZU, 1)), 0.0)
        self.assertEqual(belief.red_five_probability(TileCategory.MANZU), 0.0)
        self.assertFalse(belief.has_wait_belief)

    def test_positional_construction_still_works(self) -> None:
        belief = HandBelief(_empty_table(), (0, 0, 0))

        self.assertIsNone(belief.wait_probability_raw)

    def test_exact_self_belief_does_not_provide_wait_belief(self) -> None:
        own_hand = OwnHandState(
            concealed_tiles=(Tile(TileType(TileCategory.MANZU, 1)),), drawn_tile=None
        )
        belief = exact_self_belief(own_hand)

        self.assertFalse(belief.has_wait_belief)
        self.assertFalse(belief.has_wait_mechanism_belief)


if __name__ == "__main__":
    unittest.main()
