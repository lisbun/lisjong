"""通常ツモDLVAT型補正のV1・p・単位補正（Issue #391）のtest。"""

import ast
import inspect
import json
import unittest
from fractions import Fraction
from unittest.mock import patch

import lisjong.draw_correction as draw_correction
from lisjong.draw_correction import (
    CATEGORY_COUNT,
    RED_FIVE_CATEGORIES,
    STANDARD_CATEGORY_STOCK,
    DrawProbabilityProjection,
    ShantenValueProjection,
    draw_probabilities,
    shanten_values,
    unit_corrections_from_values,
    unit_draw_corrections,
    v1_definition,
)

# 37カテゴリのindex
M1, M2, M3, M4, M5, M6, M7, M8, M9 = range(9)
P1, P2, P3, P4 = 9, 10, 11, 12
S9 = 26
EAST, HAKU = 27, 31
RED_M5 = 34


def _counts(tiles=()):
    counts = [0] * CATEGORY_COUNT
    for tile in tiles:
        counts[tile] += 1
    return counts


def _state(
    *,
    unseen,
    drawer_hand,
    drawer=0,
    live=None,
    indicators=1,
    drawer_melds=(),
):
    """未見集合と drawer の純手牌を指定して、整合した p用射影のdictを作る。

    残りの既知牌は表示牌 → 他家の手牌（各13枚）→ drawerの河の順に割り当てる。
    """
    pool = [
        STANDARD_CATEGORY_STOCK[i] - unseen[i] - drawer_hand[i]
        for i in range(CATEGORY_COUNT)
    ]
    for tile in drawer_melds:
        pool[tile] -= 1
    assert min(pool) >= 0

    def take(count):
        taken = _counts()
        for index in range(CATEGORY_COUNT):
            while count and pool[index]:
                pool[index] -= 1
                taken[index] += 1
                count -= 1
        assert count == 0
        return taken

    indicator_counts = take(indicators)
    hands = [None] * 4
    melds = [_counts() for _ in range(4)]
    discards = [_counts() for _ in range(4)]
    hands[drawer] = list(drawer_hand)
    melds[drawer] = _counts(drawer_melds)
    for seat in range(4):
        if seat != drawer:
            hands[seat] = take(13)
    discards[drawer] = take(sum(pool))
    unseen_total = sum(unseen)
    return {
        "hand_counts": hands,
        "meld_counts": melds,
        "uncalled_discard_counts": discards,
        "revealed_dora_indicator_counts": indicator_counts,
        "live_tiles_remaining": (
            unseen_total - 14 + indicators if live is None else live
        ),
    }


# 終盤の小さなsupport: 未見14枚 = 1p×4, 9s×4, 東×4, 白×2（live 1、表示牌1）。
# drawerは 123m 456m 789m 23p 白白 の聴牌（1p-4p待ち）。
_SMALL_UNSEEN = _counts([P1] * 4 + [S9] * 4 + [EAST] * 4 + [HAKU] * 2)
_SMALL_HAND = _counts([M1, M2, M3, M4, M5, M6, M7, M8, M9, P2, P3, HAKU, HAKU])

# 序盤に近い大きなsupport（未見76枚）。drawerの手牌に赤5mを含む。
_OPENING_HAND = _counts([M1, M4, M7, P2, P2, P3, 18, 22, 25, EAST, 28, HAKU, RED_M5])


def _opening_unseen():
    # 既知 = drawerの13枚 + 他家39枚 + 表示牌と河。字牌側から2枚ずつ既知にする。
    unseen = [
        STANDARD_CATEGORY_STOCK[i] - _OPENING_HAND[i] for i in range(CATEGORY_COUNT)
    ]
    known = 136 - 76 - 13
    for index in range(33, -1, -1):
        taken = min(2, unseen[index], known)
        unseen[index] -= taken
        known -= taken
    assert known == 0
    return unseen


_OPENING_UNSEEN = _opening_unseen()


def _small(**overrides):
    return _state(unseen=_SMALL_UNSEEN, drawer_hand=_SMALL_HAND, **overrides)


def _projections(state, drawer=0):
    return (
        DrawProbabilityProjection.from_mapping(state),
        ShantenValueProjection(drawer, state["hand_counts"][drawer]),
    )


class CategoryStockTest(unittest.TestCase):
    def test_standard_stock_matches_physical_inventory(self) -> None:
        self.assertEqual(len(STANDARD_CATEGORY_STOCK), 37)
        self.assertEqual(sum(STANDARD_CATEGORY_STOCK), 136)
        for red, base in RED_FIVE_CATEGORIES.items():
            self.assertEqual(STANDARD_CATEGORY_STOCK[red], 1)
            self.assertEqual(STANDARD_CATEGORY_STOCK[base], 3)
        for base in range(34):
            red = [r for r, b in RED_FIVE_CATEGORIES.items() if b == base]
            total = STANDARD_CATEGORY_STOCK[base] + sum(
                STANDARD_CATEGORY_STOCK[r] for r in red
            )
            self.assertEqual(total, 4, msg=f"base kind {base}")

    def test_v1_definition_is_versioned_json_native_and_fresh(self) -> None:
        definition = v1_definition()
        self.assertEqual(definition["name"], "lisjong.draw_correction")
        self.assertEqual(definition["version"], 1)
        self.assertEqual(json.loads(json.dumps(definition)), definition)
        definition["version"] = 99
        self.assertEqual(v1_definition()["version"], 1)


class DrawProbabilityTest(unittest.TestCase):
    def test_small_support_probabilities_are_exact(self) -> None:
        projection = DrawProbabilityProjection.from_mapping(_small())

        self.assertEqual(
            draw_probabilities(projection),
            {
                P1: Fraction(2, 7),
                S9: Fraction(2, 7),
                EAST: Fraction(2, 7),
                HAKU: Fraction(1, 7),
            },
        )

    def test_support_contains_only_unseen_categories_and_sums_to_one(self) -> None:
        projection = DrawProbabilityProjection.from_mapping(
            _state(unseen=_OPENING_UNSEEN, drawer_hand=_OPENING_HAND)
        )
        probabilities = draw_probabilities(projection)

        self.assertEqual(sum(probabilities.values()), 1)
        self.assertEqual(list(probabilities), sorted(probabilities))
        for category in range(CATEGORY_COUNT):
            n_a = projection.unseen_counts()[category]
            self.assertEqual(category in probabilities, n_a > 0)
        # 赤5mは drawer の手牌にあるのでsupport外。
        self.assertNotIn(RED_M5, probabilities)

    def test_unseen_total_follows_live_plus_dead_wall_minus_indicators(self) -> None:
        state = _state(
            unseen=_OPENING_UNSEEN,
            drawer_hand=_OPENING_HAND,
            indicators=2,
            drawer_melds=(),
        )
        projection = DrawProbabilityProjection.from_mapping(state)
        self.assertEqual(
            sum(projection.unseen_counts()), state["live_tiles_remaining"] + 14 - 2
        )


class ShantenValueTest(unittest.TestCase):
    def test_drawer_gets_plus_g_and_others_minus_g_over_three(self) -> None:
        projection = ShantenValueProjection(2, _SMALL_HAND)

        self.assertEqual(
            shanten_values(projection, P1),
            (Fraction(1, 3), Fraction(1, 3), Fraction(-1), Fraction(1, 3)),
        )
        self.assertEqual(shanten_values(projection, S9), (0, 0, 0, 0))

    def test_red_five_is_evaluated_like_the_normal_five(self) -> None:
        hand = _counts([M3, M4, M6, M6, P1, P1, P1, P2, P3, P4, 18, 19, 20])
        projection = ShantenValueProjection(0, hand)
        self.assertEqual(
            shanten_values(projection, RED_M5), shanten_values(projection, M5)
        )
        self.assertEqual(shanten_values(projection, RED_M5)[0], -1)

    def test_every_pre_draw_hand_size_is_evaluated(self) -> None:
        for size in (1, 4, 7, 10, 13):
            with self.subTest(size=size):
                hand = _counts(
                    [M1, M1, M1, M2, M2, M2, M3, M3, M3, M4, M4, M4, M5][:size]
                )
                projection = ShantenValueProjection(1, hand)
                values = shanten_values(projection, M5 if size == 1 else EAST)
                self.assertEqual(sum(values), 0)

    def test_candidate_beyond_stock_in_hand_is_rejected(self) -> None:
        hand = _counts([M1] * 4 + [M2] * 4 + [M3] * 4 + [EAST])
        with self.assertRaises(ValueError):
            shanten_values(ShantenValueProjection(0, hand), M1)

    def test_unknown_category_is_rejected(self) -> None:
        projection = ShantenValueProjection(0, _SMALL_HAND)
        for category in (-1, 37):
            with self.subTest(category=category), self.assertRaises(ValueError):
                shanten_values(projection, category)
        for category in (True, 1.0, "1"):
            with self.subTest(category=category), self.assertRaises(TypeError):
                shanten_values(projection, category)


class UnitCorrectionTest(unittest.TestCase):
    def test_small_support_corrections_match_hand_computation(self) -> None:
        # E[g] = 2/7·(−1) + 2/7·0 + 2/7·0 + 1/7·0 = −2/7
        p_input, v_input = _projections(_small())

        self.assertEqual(
            unit_draw_corrections(p_input, v_input, P1),
            (Fraction(-5, 7), Fraction(5, 21), Fraction(5, 21), Fraction(5, 21)),
        )
        self.assertEqual(
            unit_draw_corrections(p_input, v_input, S9),
            (Fraction(2, 7), Fraction(-2, 21), Fraction(-2, 21), Fraction(-2, 21)),
        )

    def test_expected_correction_is_exactly_zero_and_seats_sum_to_zero(self) -> None:
        cases = {
            "small": (_small(), 0),
            "small_drawer_3": (
                _state(unseen=_SMALL_UNSEEN, drawer_hand=_SMALL_HAND, drawer=3),
                3,
            ),
            "opening": (
                _state(unseen=_OPENING_UNSEEN, drawer_hand=_OPENING_HAND, drawer=1),
                1,
            ),
        }
        for name, (state, drawer) in cases.items():
            with self.subTest(case=name):
                p_input, v_input = _projections(state, drawer)
                probabilities = draw_probabilities(p_input)
                corrections = {
                    a: unit_draw_corrections(p_input, v_input, a) for a in probabilities
                }
                for seat in range(4):
                    self.assertEqual(
                        sum(
                            probabilities[a] * corrections[a][seat]
                            for a in probabilities
                        ),
                        0,
                    )
                for a, correction in corrections.items():
                    self.assertEqual(sum(correction), 0, msg=f"realized {a}")
                    self.assertTrue(all(type(c) is Fraction for c in correction))

    def test_zero_and_constant_value_give_zero_correction(self) -> None:
        probabilities = draw_probabilities(
            DrawProbabilityProjection.from_mapping(_small())
        )
        for row in ((0, 0, 0, 0), (3, Fraction(-1, 3), 0, 7)):
            values = dict.fromkeys(probabilities, row)
            for realized in probabilities:
                with self.subTest(row=row, realized=realized):
                    self.assertEqual(
                        unit_corrections_from_values(probabilities, values, realized),
                        (0, 0, 0, 0),
                    )

    def test_categories_outside_the_support_are_not_evaluated(self) -> None:
        p_input, v_input = _projections(_small())
        with patch.object(
            draw_correction, "shanten_values", wraps=draw_correction.shanten_values
        ) as spy:
            unit_draw_corrections(p_input, v_input, S9)

        evaluated = [call.args[1] for call in spy.call_args_list]
        self.assertEqual(evaluated, [P1, S9, EAST, HAKU])

    def test_value_is_computed_for_the_whole_support_before_lookup(self) -> None:
        p_input, v_input = _projections(_small())
        with patch.object(
            draw_correction, "shanten_values", wraps=draw_correction.shanten_values
        ) as spy:
            for realized in (P1, HAKU):
                spy.reset_mock()
                unit_draw_corrections(p_input, v_input, realized)
                self.assertEqual(spy.call_count, 4)

    def test_realized_category_outside_support_is_rejected(self) -> None:
        p_input, v_input = _projections(_small())
        for realized in (M1, RED_M5):
            with self.subTest(realized=realized), self.assertRaises(ValueError):
                unit_draw_corrections(p_input, v_input, realized)
        with self.assertRaises(ValueError):
            unit_draw_corrections(p_input, v_input, 37)

    def test_drawer_hand_mismatch_between_projections_is_rejected(self) -> None:
        state = _small()
        p_input = DrawProbabilityProjection.from_mapping(state)
        with self.assertRaises(ValueError):
            # drawer 1と名乗りながら、drawer 0の手牌を渡す
            unit_draw_corrections(
                p_input, ShantenValueProjection(1, state["hand_counts"][0]), P1
            )

    def test_inconsistent_probabilities_or_values_are_rejected(self) -> None:
        probabilities = {P1: Fraction(1, 2), S9: Fraction(1, 2)}
        values = {P1: (1, 0, 0, 0), S9: (0, 0, 0, 0)}
        bad_inputs = {
            "sum_not_one": (
                {P1: Fraction(1, 2), S9: Fraction(1, 3)},
                values,
                ValueError,
            ),
            "zero_probability": (
                {P1: Fraction(1), S9: Fraction(0)},
                values,
                ValueError,
            ),
            "float_probability": ({P1: 0.5, S9: 0.5}, values, TypeError),
            "empty": ({}, {}, ValueError),
            "values_missing": (probabilities, {P1: (1, 0, 0, 0)}, ValueError),
            "values_extra": (
                probabilities,
                {**values, EAST: (0, 0, 0, 0)},
                ValueError,
            ),
            "three_seats": (probabilities, {P1: (1, 0, 0), S9: (0, 0, 0)}, ValueError),
            "float_value": (
                probabilities,
                {P1: (1.0, 0, 0, 0), S9: (0, 0, 0, 0)},
                TypeError,
            ),
            "unknown_category": ({37: Fraction(1)}, {37: (0, 0, 0, 0)}, ValueError),
        }
        for name, (p, v, error) in bad_inputs.items():
            with self.subTest(case=name), self.assertRaises(error):
                unit_corrections_from_values(p, v, P1)


class ProjectionValidationTest(unittest.TestCase):
    def _assert_rejected(self, state, error=ValueError) -> None:
        with self.assertRaises(error):
            DrawProbabilityProjection.from_mapping(state)

    def test_projection_normalizes_to_immutable_tuples(self) -> None:
        projection = DrawProbabilityProjection.from_mapping(_small())
        self.assertIs(type(projection.hand_counts), tuple)
        self.assertIs(type(projection.hand_counts[0]), tuple)
        self.assertIs(type(projection.revealed_dora_indicator_counts), tuple)

    def test_live_tiles_mismatch_is_rejected(self) -> None:
        state = _small()
        state["live_tiles_remaining"] += 1
        self._assert_rejected(state)

    def test_no_live_tile_is_rejected(self) -> None:
        state = _small()
        state["live_tiles_remaining"] = 0
        self._assert_rejected(state)

    def test_inventory_exceeding_stock_is_rejected(self) -> None:
        state = _small()
        # 1mはdrawerの手牌に1枚あるので、他家の河に4枚置くと既知が在庫4を超える。
        state["uncalled_discard_counts"][1][M1] = 4
        self._assert_rejected(state)

    def test_red_five_inconsistency_is_rejected(self) -> None:
        state = _small()
        state["hand_counts"][1][RED_M5] = 2
        self._assert_rejected(state)
        state = _small()
        state["hand_counts"][1][M5] = 4
        self._assert_rejected(state)
        # 赤5mは既知のどこかに1枚ある。表示牌にもう1枚置くと合計が在庫1を超える。
        state = _small()
        state["revealed_dora_indicator_counts"][RED_M5] += 1
        self._assert_rejected(state)

    def test_hand_size_violation_is_rejected(self) -> None:
        state = _small()
        state["hand_counts"][0][EAST] = 1  # drawerが14枚
        self._assert_rejected(state)
        tiles = [M1, M2, M3, M4, M5, M6, M7, M8, M9, P1, P2, P3, P4, S9]
        for size in (0, 2, 12, 14):
            with self.subTest(size=size), self.assertRaises(ValueError):
                ShantenValueProjection(0, _counts(tiles[:size]))

    def test_unknown_category_axis_is_rejected(self) -> None:
        state = _small()
        state["hand_counts"][0] = state["hand_counts"][0] + [0]
        self._assert_rejected(state)
        state = _small()
        state["revealed_dora_indicator_counts"] = state[
            "revealed_dora_indicator_counts"
        ][:34]
        self._assert_rejected(state)
        with self.assertRaises(ValueError):
            ShantenValueProjection(4, _SMALL_HAND)

    def test_non_integer_counts_are_rejected(self) -> None:
        for bad in (True, 1.0, "1"):
            with self.subTest(bad=bad):
                state = _small()
                state["meld_counts"][2][M1] = bad
                self._assert_rejected(state, TypeError)
        state = _small()
        state["hand_counts"] = {"0": state["hand_counts"][0]}
        self._assert_rejected(state, TypeError)

    def test_indicator_count_out_of_range_is_rejected(self) -> None:
        state = _small()
        moved = state["revealed_dora_indicator_counts"]
        index = moved.index(1)
        moved[index] = 0
        state["uncalled_discard_counts"][0][index] += 1
        state["live_tiles_remaining"] -= 1
        self._assert_rejected(state)


class AllowlistBoundaryTest(unittest.TestCase):
    _FORBIDDEN_KEYS = (
        "seed",
        "identity",
        "placement",
        "draw_index",
        "realized",
        "tile",
        "post_draw_checkpoint_sha256",
        "drawer_seat",
    )

    def test_probability_projection_rejects_identity_and_realized_fields(self) -> None:
        for key in self._FORBIDDEN_KEYS:
            with self.subTest(key=key):
                state = _small()
                state[key] = 0
                with self.assertRaises(ValueError):
                    DrawProbabilityProjection.from_mapping(state)
                with self.assertRaises(TypeError):
                    DrawProbabilityProjection(**state)

    def test_value_projection_rejects_other_fields(self) -> None:
        base = {"drawer_seat": 0, "drawer_hand_counts": _SMALL_HAND}
        self.assertEqual(
            ShantenValueProjection.from_mapping(base),
            ShantenValueProjection(0, _SMALL_HAND),
        )
        for key in (*self._FORBIDDEN_KEYS[:-1], "hand_counts", "live_tiles_remaining"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    ShantenValueProjection.from_mapping({**base, key: 0})
                with self.assertRaises(TypeError):
                    ShantenValueProjection(**base, **{key: 0})
        with self.assertRaises(ValueError):
            ShantenValueProjection.from_mapping({"drawer_seat": 0})

    def test_projection_types_have_no_extra_attributes(self) -> None:
        projection = ShantenValueProjection(0, _SMALL_HAND)
        with self.assertRaises((AttributeError, TypeError)):
            projection.seed = 1  # type: ignore[attr-defined]

    def test_module_imports_no_arena_engine_or_ml_framework(self) -> None:
        tree = ast.parse(inspect.getsource(draw_correction))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        for name in imported:
            for prefix in (
                "lisjong_arena",
                "lisjong_engine",
                "lisjong.learning",
                "lisjong.policies",
                "torch",
                "numpy",
            ):
                with self.subTest(imported=name, prefix=prefix):
                    self.assertFalse(name == prefix or name.startswith(prefix + "."))


if __name__ == "__main__":
    unittest.main()
