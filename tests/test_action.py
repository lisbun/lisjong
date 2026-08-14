import unittest
from dataclasses import FrozenInstanceError, fields

from lisjong.policy_contract.action import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DiscardAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import EAST_WIND, Tile, TileCategory, TileType

MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))
MANZU_5_RED = Tile(TileType(TileCategory.MANZU, 5), is_red=True)
MANZU_6 = Tile(TileType(TileCategory.MANZU, 6))
MANZU_7 = Tile(TileType(TileCategory.MANZU, 7))
PINZU_5 = Tile(TileType(TileCategory.PINZU, 5))
PINZU_5_RED = Tile(TileType(TileCategory.PINZU, 5), is_red=True)
EAST_TILE = Tile(EAST_WIND)

# 全variantの、正本文書どおりの最小構成での有効なkwargs。
# CommonActionContractTestが構造上の共通契約を横断的に検証するために使う。
_VALID_KWARGS: dict[type, dict[str, object]] = {
    DiscardAction: {"actor": Seat.SEAT_0, "tile": MANZU_5, "tsumogiri": False},
    RiichiAction: {"actor": Seat.SEAT_0},
    ChiAction: {
        "actor": Seat.SEAT_1,
        "target": Seat.SEAT_0,
        "called_tile": MANZU_5,
        "consumed_tiles": (MANZU_4, MANZU_6),
    },
    PonAction: {
        "actor": Seat.SEAT_0,
        "target": Seat.SEAT_1,
        "called_tile": PINZU_5,
        "consumed_tiles": (PINZU_5, PINZU_5_RED),
    },
    DaiminkanAction: {
        "actor": Seat.SEAT_0,
        "target": Seat.SEAT_1,
        "called_tile": PINZU_5,
        "consumed_tiles": (PINZU_5, PINZU_5, PINZU_5_RED),
    },
    AnkanAction: {
        "actor": Seat.SEAT_0,
        "tiles": (PINZU_5, PINZU_5, PINZU_5, PINZU_5_RED),
    },
    KakanAction: {
        "actor": Seat.SEAT_0,
        "added_tile": PINZU_5_RED,
        "from_seat": Seat.SEAT_1,
        "called_tile": PINZU_5,
    },
    RonAction: {"actor": Seat.SEAT_0, "target": Seat.SEAT_1, "winning_tile": MANZU_5},
    TsumoAction: {"actor": Seat.SEAT_0, "winning_tile": MANZU_5},
    PassAction: {"actor": Seat.SEAT_0},
    KyuushuKyuuhaiAction: {"actor": Seat.SEAT_0},
}

_EXPECTED_FIELDS: dict[type, set[str]] = {
    DiscardAction: {"actor", "tile", "tsumogiri"},
    RiichiAction: {"actor"},
    ChiAction: {"actor", "target", "called_tile", "consumed_tiles"},
    PonAction: {"actor", "target", "called_tile", "consumed_tiles"},
    DaiminkanAction: {"actor", "target", "called_tile", "consumed_tiles"},
    AnkanAction: {"actor", "tiles"},
    KakanAction: {"actor", "added_tile", "from_seat", "called_tile"},
    RonAction: {"actor", "target", "winning_tile"},
    TsumoAction: {"actor", "winning_tile"},
    PassAction: {"actor"},
    KyuushuKyuuhaiAction: {"actor"},
}


class CommonActionContractTest(unittest.TestCase):
    """全variant共通の構造契約。docs/internal-action-model.mdを正本とする。"""

    def test_required_fields_match_the_authoritative_docs(self) -> None:
        for action_class, expected_fields in _EXPECTED_FIELDS.items():
            with self.subTest(action_class=action_class):
                field_names = {field.name for field in fields(action_class)}
                self.assertEqual(field_names, expected_fields)
                # source_meld_id/index、physical tile ID、object referenceが
                # 紛れ込んでいないことを、required fieldの完全一致で兼ねて確認する。
                self.assertIn("actor", field_names)

    def test_all_variants_are_immutable(self) -> None:
        for action_class, kwargs in _VALID_KWARGS.items():
            with self.subTest(action_class=action_class):
                action = action_class(**kwargs)
                with self.assertRaises(FrozenInstanceError):
                    action.actor = Seat.SEAT_2

    def test_different_variants_with_same_actor_are_not_equal(self) -> None:
        actor = Seat.SEAT_0
        self.assertNotEqual(PassAction(actor=actor), RiichiAction(actor=actor))
        self.assertNotEqual(PassAction(actor=actor), KyuushuKyuuhaiAction(actor=actor))
        self.assertNotEqual(
            RiichiAction(actor=actor), KyuushuKyuuhaiAction(actor=actor)
        )


class DiscardActionTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        action = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5, tsumogiri=True)
        self.assertEqual(action.actor, Seat.SEAT_0)
        self.assertEqual(action.tile, MANZU_5)
        self.assertTrue(action.tsumogiri)

    def test_rejects_wrong_types(self) -> None:
        with self.assertRaises(TypeError):
            DiscardAction(actor=0, tile=MANZU_5, tsumogiri=False)
        with self.assertRaises(TypeError):
            DiscardAction(actor=Seat.SEAT_0, tile="5m", tsumogiri=False)
        with self.assertRaises(TypeError):
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5, tsumogiri=1)

    def test_tsumogiri_difference_changes_identity(self) -> None:
        tedashi = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5, tsumogiri=False)
        tsumogiri = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5, tsumogiri=True)
        self.assertNotEqual(tedashi, tsumogiri)

    def test_normal_and_red_tile_differ(self) -> None:
        normal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5, tsumogiri=False)
        red = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5_RED, tsumogiri=False)
        self.assertNotEqual(normal, red)

    def test_actor_difference_changes_identity(self) -> None:
        first = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5, tsumogiri=False)
        second = DiscardAction(actor=Seat.SEAT_1, tile=MANZU_5, tsumogiri=False)
        self.assertNotEqual(first, second)


class RiichiActionTest(unittest.TestCase):
    def test_creates_with_actor_only(self) -> None:
        action = RiichiAction(actor=Seat.SEAT_2)
        self.assertEqual(action.actor, Seat.SEAT_2)

    def test_rejects_non_seat_actor(self) -> None:
        with self.assertRaises(TypeError):
            RiichiAction(actor=2)

    def test_actor_difference_changes_identity(self) -> None:
        self.assertNotEqual(
            RiichiAction(actor=Seat.SEAT_0), RiichiAction(actor=Seat.SEAT_1)
        )


class ChiActionTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        action = ChiAction(
            actor=Seat.SEAT_1,
            target=Seat.SEAT_0,
            called_tile=MANZU_5,
            consumed_tiles=(MANZU_4, MANZU_6),
        )
        self.assertEqual(action.consumed_tiles, (MANZU_4, MANZU_6))

    def test_rejects_actor_equal_to_target(self) -> None:
        with self.assertRaises(ValueError):
            ChiAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_0,
                called_tile=MANZU_5,
                consumed_tiles=(MANZU_4, MANZU_6),
            )

    def test_rejects_target_that_is_not_kamicha(self) -> None:
        # actor=SEAT_1の上家はSEAT_0であり、SEAT_2やSEAT_3ではない。
        for non_kamicha_target in (Seat.SEAT_2, Seat.SEAT_3):
            with self.subTest(target=non_kamicha_target), self.assertRaises(ValueError):
                ChiAction(
                    actor=Seat.SEAT_1,
                    target=non_kamicha_target,
                    called_tile=MANZU_5,
                    consumed_tiles=(MANZU_4, MANZU_6),
                )

    def test_rejects_honor_tiles(self) -> None:
        with self.assertRaises(ValueError):
            ChiAction(
                actor=Seat.SEAT_1,
                target=Seat.SEAT_0,
                called_tile=EAST_TILE,
                consumed_tiles=(EAST_TILE, EAST_TILE),
            )

    def test_rejects_non_consecutive_ranks(self) -> None:
        with self.assertRaises(ValueError):
            ChiAction(
                actor=Seat.SEAT_1,
                target=Seat.SEAT_0,
                called_tile=MANZU_5,
                consumed_tiles=(MANZU_4, MANZU_7),
            )

    def test_rejects_mixed_suits(self) -> None:
        pinzu_4 = Tile(TileType(TileCategory.PINZU, 4))
        with self.assertRaises(ValueError):
            ChiAction(
                actor=Seat.SEAT_1,
                target=Seat.SEAT_0,
                called_tile=MANZU_5,
                consumed_tiles=(MANZU_4, pinzu_4),
            )

    def test_rejects_wrong_consumed_count(self) -> None:
        with self.assertRaises(ValueError):
            ChiAction(
                actor=Seat.SEAT_1,
                target=Seat.SEAT_0,
                called_tile=MANZU_5,
                consumed_tiles=(MANZU_4,),
            )

    def test_input_order_does_not_affect_identity(self) -> None:
        first = ChiAction(
            actor=Seat.SEAT_1,
            target=Seat.SEAT_0,
            called_tile=MANZU_5,
            consumed_tiles=(MANZU_4, MANZU_6),
        )
        second = ChiAction(
            actor=Seat.SEAT_1,
            target=Seat.SEAT_0,
            called_tile=MANZU_5,
            consumed_tiles=(MANZU_6, MANZU_4),
        )
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)

    def test_red_construction_changes_identity(self) -> None:
        normal = ChiAction(
            actor=Seat.SEAT_1,
            target=Seat.SEAT_0,
            called_tile=MANZU_4,
            consumed_tiles=(MANZU_5, MANZU_6),
        )
        red = ChiAction(
            actor=Seat.SEAT_1,
            target=Seat.SEAT_0,
            called_tile=MANZU_4,
            consumed_tiles=(MANZU_5_RED, MANZU_6),
        )
        self.assertNotEqual(normal, red)


class PonActionTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        action = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5, PINZU_5_RED),
        )
        self.assertEqual(action.consumed_tiles, (PINZU_5, PINZU_5_RED))

    def test_rejects_actor_equal_to_target(self) -> None:
        with self.assertRaises(ValueError):
            PonAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_0,
                called_tile=PINZU_5,
                consumed_tiles=(PINZU_5, PINZU_5),
            )

    def test_rejects_wrong_consumed_count(self) -> None:
        with self.assertRaises(ValueError):
            PonAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_1,
                called_tile=PINZU_5,
                consumed_tiles=(PINZU_5, PINZU_5, PINZU_5),
            )

    def test_rejects_different_base_tile_kind(self) -> None:
        other_kind = Tile(TileType(TileCategory.PINZU, 6))
        with self.assertRaises(ValueError):
            PonAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_1,
                called_tile=PINZU_5,
                consumed_tiles=(PINZU_5, other_kind),
            )

    def test_input_order_does_not_affect_identity(self) -> None:
        first = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5, PINZU_5_RED),
        )
        second = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5_RED, PINZU_5),
        )
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))

    def test_red_construction_changes_identity(self) -> None:
        with_red = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5, PINZU_5_RED),
        )
        all_normal = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5, PINZU_5),
        )
        self.assertNotEqual(with_red, all_normal)

    def test_allows_duplicate_semantic_tiles(self) -> None:
        # lisjongのTileはphysical copy identityを持たないため、同一semantic
        # Tileの重複は正常である。
        action = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5, PINZU_5),
        )
        self.assertEqual(action.consumed_tiles, (PINZU_5, PINZU_5))


class DaiminkanActionTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        action = DaiminkanAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5, PINZU_5, PINZU_5_RED),
        )
        self.assertEqual(len(action.consumed_tiles), 3)

    def test_rejects_actor_equal_to_target(self) -> None:
        with self.assertRaises(ValueError):
            DaiminkanAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_0,
                called_tile=PINZU_5,
                consumed_tiles=(PINZU_5, PINZU_5, PINZU_5),
            )

    def test_rejects_wrong_consumed_count(self) -> None:
        with self.assertRaises(ValueError):
            DaiminkanAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_1,
                called_tile=PINZU_5,
                consumed_tiles=(PINZU_5, PINZU_5),
            )

    def test_rejects_different_base_tile_kind(self) -> None:
        other_kind = Tile(TileType(TileCategory.PINZU, 6))
        with self.assertRaises(ValueError):
            DaiminkanAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_1,
                called_tile=PINZU_5,
                consumed_tiles=(PINZU_5, PINZU_5, other_kind),
            )

    def test_input_order_does_not_affect_identity(self) -> None:
        first = DaiminkanAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5, PINZU_5, PINZU_5_RED),
        )
        second = DaiminkanAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=PINZU_5,
            consumed_tiles=(PINZU_5_RED, PINZU_5, PINZU_5),
        )
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))


class AnkanActionTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        action = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5_RED),
        )
        self.assertEqual(len(action.tiles), 4)

    def test_rejects_wrong_tile_count(self) -> None:
        with self.assertRaises(ValueError):
            AnkanAction(actor=Seat.SEAT_0, tiles=(PINZU_5, PINZU_5, PINZU_5))

    def test_rejects_different_base_tile_kind(self) -> None:
        other_kind = Tile(TileType(TileCategory.PINZU, 6))
        with self.assertRaises(ValueError):
            AnkanAction(
                actor=Seat.SEAT_0,
                tiles=(PINZU_5, PINZU_5, PINZU_5, other_kind),
            )

    def test_input_order_does_not_affect_identity(self) -> None:
        first = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5_RED),
        )
        second = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(PINZU_5_RED, PINZU_5, PINZU_5, PINZU_5),
        )
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)

    def test_multiplicity_and_red_construction_change_identity(self) -> None:
        # (5p, 5p, 5p, 5pr) と全て通常5pのankanは、赤牌構成が異なるため別。
        with_one_red = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5_RED),
        )
        all_normal = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5),
        )
        self.assertNotEqual(with_one_red, all_normal)

    def test_allows_duplicate_semantic_tiles_without_physical_copy_check(
        self,
    ) -> None:
        action = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5_RED),
        )
        self.assertEqual(action.tiles.count(PINZU_5), 3)


class KakanActionTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        action = KakanAction(
            actor=Seat.SEAT_0,
            added_tile=PINZU_5_RED,
            from_seat=Seat.SEAT_1,
            called_tile=PINZU_5,
        )
        self.assertEqual(action.added_tile, PINZU_5_RED)

    def test_rejects_actor_equal_to_from_seat(self) -> None:
        with self.assertRaises(ValueError):
            KakanAction(
                actor=Seat.SEAT_0,
                added_tile=PINZU_5,
                from_seat=Seat.SEAT_0,
                called_tile=PINZU_5,
            )

    def test_rejects_different_base_tile_kind(self) -> None:
        other_kind = Tile(TileType(TileCategory.PINZU, 6))
        with self.assertRaises(ValueError):
            KakanAction(
                actor=Seat.SEAT_0,
                added_tile=other_kind,
                from_seat=Seat.SEAT_1,
                called_tile=PINZU_5,
            )

    def test_has_no_source_meld_reference_fields(self) -> None:
        field_names = {field.name for field in fields(KakanAction)}
        for forbidden in ("source_meld_id", "source_meld_index", "pon", "meld"):
            self.assertNotIn(forbidden, field_names)


class RonActionTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        action = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_5)
        self.assertEqual(action.winning_tile, MANZU_5)

    def test_rejects_actor_equal_to_target(self) -> None:
        with self.assertRaises(ValueError):
            RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_0, winning_tile=MANZU_5)


class TsumoActionTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        action = TsumoAction(actor=Seat.SEAT_0, winning_tile=MANZU_5)
        self.assertEqual(action.winning_tile, MANZU_5)

    def test_rejects_non_tile_winning_tile(self) -> None:
        with self.assertRaises(TypeError):
            TsumoAction(actor=Seat.SEAT_0, winning_tile="5m")


class PassActionTest(unittest.TestCase):
    def test_value_equality_depends_only_on_actor(self) -> None:
        self.assertEqual(PassAction(actor=Seat.SEAT_0), PassAction(actor=Seat.SEAT_0))
        self.assertNotEqual(
            PassAction(actor=Seat.SEAT_0), PassAction(actor=Seat.SEAT_1)
        )

    def test_not_equal_to_other_actor_only_variants(self) -> None:
        self.assertNotEqual(
            PassAction(actor=Seat.SEAT_0), KyuushuKyuuhaiAction(actor=Seat.SEAT_0)
        )


class KyuushuKyuuhaiActionTest(unittest.TestCase):
    def test_value_equality_depends_only_on_actor(self) -> None:
        self.assertEqual(
            KyuushuKyuuhaiAction(actor=Seat.SEAT_0),
            KyuushuKyuuhaiAction(actor=Seat.SEAT_0),
        )
        self.assertNotEqual(
            KyuushuKyuuhaiAction(actor=Seat.SEAT_0),
            KyuushuKyuuhaiAction(actor=Seat.SEAT_1),
        )


class SemanticIdentityTest(unittest.TestCase):
    """a == b が正本文書のsemantic identityと一致することを重点的に確認する。

    same semantic identity = same action variant + same actor
                              + same variant-specific semantic fields
    """

    def test_value_equality_is_reflexive_for_every_variant(self) -> None:
        for action_class, kwargs in _VALID_KWARGS.items():
            with self.subTest(action_class=action_class):
                action = action_class(**kwargs)
                self.assertEqual(action, action_class(**kwargs))

    def test_deduplicates_via_set_without_relying_on_specific_hash_values(
        self,
    ) -> None:
        # hash値そのものではなく、value equalityによってdeduplicationされる
        # ことを確認する（hash collisionが起きても最終的にはvalue equalityで
        # 判定される設計）。
        duplicates = {
            ChiAction(
                actor=Seat.SEAT_1,
                target=Seat.SEAT_0,
                called_tile=MANZU_5,
                consumed_tiles=(MANZU_4, MANZU_6),
            ),
            ChiAction(
                actor=Seat.SEAT_1,
                target=Seat.SEAT_0,
                called_tile=MANZU_5,
                consumed_tiles=(MANZU_6, MANZU_4),
            ),
            PassAction(actor=Seat.SEAT_0),
            PassAction(actor=Seat.SEAT_0),
        }
        self.assertEqual(len(duplicates), 2)

    def test_variant_specific_field_difference_changes_identity(self) -> None:
        first = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_5)
        second = RonAction(actor=Seat.SEAT_0, target=Seat.SEAT_2, winning_tile=MANZU_5)
        third = RonAction(
            actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=MANZU_5_RED
        )
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)


if __name__ == "__main__":
    unittest.main()
