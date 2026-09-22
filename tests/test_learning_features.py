"""lisjong所有のplayer-safe feature representationのcontract test。"""

import unittest

import learning_fixtures as fixtures

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.learning import (
    FEATURE_DIMENSION,
    FEATURE_IDENTITY,
    FeatureError,
    build_player_safe_feature,
    feature_fingerprint,
    feature_specification,
)
from lisjong.learning._canonical import value_digest
from lisjong.learning.features import (
    FEATURE_BLOCK_OFFSETS,
    FEATURE_LAYOUT,
    TILE_TYPE_AXIS_SIZE,
)
from lisjong.policy_contract import (
    DecisionContext,
    MeldKind,
    OwnHandState,
    PublicMeld,
    RiichiState,
    Seat,
)

# feature contractを変更した場合はidentityとこのpinned fingerprintの双方を
# 更新する。値が変わったことに気付かずartifactを再利用しないための固定点である。
PINNED_FEATURE_FINGERPRINT = (
    "920634f920e094e19a172abb3f66ad591d507657316f275f16eb67b9ec8f74e2"
)

HISTORICAL_ARENA_FEATURE_IDENTITY = "arena-policy-input-feature-v1"
HISTORICAL_ARENA_FEATURE_DIMENSION = 8204


class FeatureIdentityTests(unittest.TestCase):
    def test_identity_is_lisjong_owned_and_distinct_from_arena(self) -> None:
        self.assertEqual(FEATURE_IDENTITY, "lisjong-offense-l0-player-safe-feature-v1")
        self.assertNotEqual(FEATURE_IDENTITY, HISTORICAL_ARENA_FEATURE_IDENTITY)
        self.assertFalse(FEATURE_IDENTITY.startswith("arena"))
        self.assertNotEqual(FEATURE_DIMENSION, HISTORICAL_ARENA_FEATURE_DIMENSION)

    def test_fingerprint_is_stable_and_derived_from_the_specification(self) -> None:
        self.assertEqual(feature_fingerprint(), PINNED_FEATURE_FINGERPRINT)
        self.assertEqual(feature_fingerprint(), value_digest(feature_specification()))
        self.assertEqual(feature_specification()["identity"], FEATURE_IDENTITY)
        self.assertEqual(feature_specification()["dimension"], FEATURE_DIMENSION)

    def test_fingerprint_changes_when_the_layout_changes(self) -> None:
        specification = feature_specification()
        specification["blocks"][0]["size"] += 1
        self.assertNotEqual(value_digest(specification), feature_fingerprint())

    def test_layout_is_contiguous(self) -> None:
        offset = 0
        for block in FEATURE_LAYOUT:
            self.assertEqual(block.offset, offset)
            self.assertGreaterEqual(block.size, 1)
            offset += block.size
        self.assertEqual(offset, FEATURE_DIMENSION)

    def test_layout_exposes_only_player_safe_blocks(self) -> None:
        """blockの集合を固定する。

        `live_wall_tiles_remaining`は公開されている残り枚数であり、wall
        contentsではない。opponentのconcealed手牌、wall / dead wallの実配列、
        future event、oracle / training-only truth、Policy-internal analysis
        （shanten / ukeire / belief等）に対応するblockは存在しない。
        """
        names = tuple(block.name for block in FEATURE_LAYOUT)
        self.assertEqual(
            names,
            (
                "own_concealed_counts",
                "own_concealed_red_fives",
                "own_drawn_tile_type",
                "own_drawn_tile_red_five",
                "own_has_drawn_tile",
                "player[0:self].discard_counts",
                "player[0:self].discard_red_fives",
                "player[0:self].riichi_state",
                "player[0:self].meld_kind_counts",
                "player[0:self].discard_count",
                "player[0:self].score",
                "player[1:shimocha].discard_counts",
                "player[1:shimocha].discard_red_fives",
                "player[1:shimocha].riichi_state",
                "player[1:shimocha].meld_kind_counts",
                "player[1:shimocha].discard_count",
                "player[1:shimocha].score",
                "player[2:toimen].discard_counts",
                "player[2:toimen].discard_red_fives",
                "player[2:toimen].riichi_state",
                "player[2:toimen].meld_kind_counts",
                "player[2:toimen].discard_count",
                "player[2:toimen].score",
                "player[3:kamicha].discard_counts",
                "player[3:kamicha].discard_red_fives",
                "player[3:kamicha].riichi_state",
                "player[3:kamicha].meld_kind_counts",
                "player[3:kamicha].discard_count",
                "player[3:kamicha].score",
                "round_wind",
                "own_seat_wind",
                "own_is_dealer",
                "hand_number",
                "honba",
                "riichi_sticks",
                "live_wall_tiles_remaining",
                "dora_indicator_counts",
                "dora_indicator_red_fives",
                "visible_tile_counts",
                "visible_red_fives",
            ),
        )
        for forbidden in (
            "oracle",
            "future",
            "opponent",
            "shanten",
            "ukeire",
            "belief",
            "reward",
            "teacher",
            "dead_wall",
            "wall_tiles_content",
        ):
            self.assertFalse(
                any(forbidden in name for name in names),
                msg=f"feature layout must not expose {forbidden}",
            )


class FeatureMaterializationTests(unittest.TestCase):
    def test_deterministic_for_equal_inputs(self) -> None:
        first = build_player_safe_feature(fixtures.policy_input())
        second = build_player_safe_feature(fixtures.policy_input())

        self.assertEqual(len(first), FEATURE_DIMENSION)
        self.assertEqual(first, second)
        self.assertTrue(all(isinstance(value, float) for value in first))

    def test_distinct_inputs_produce_distinct_representations(self) -> None:
        base = fixtures.policy_input()
        other = fixtures.policy_input(self_seat=Seat.SEAT_1)

        self.assertNotEqual(
            build_player_safe_feature(base), build_player_safe_feature(other)
        )

    def test_rejects_decision_context_and_other_inputs(self) -> None:
        value = fixtures.policy_input()
        decision = DecisionContext(
            input=value, legal_actions=fixtures.discard_legal_actions(value)
        )

        with self.assertRaises(FeatureError):
            build_player_safe_feature(decision)
        with self.assertRaises(FeatureError):
            build_player_safe_feature(None)

    def test_own_hand_and_drawn_tile_blocks(self) -> None:
        drawn = fixtures.tile(fixtures.MANZU, 1)
        value = fixtures.policy_input(self_seat=Seat.SEAT_0, drawn=drawn)
        values = build_player_safe_feature(value)

        own = FEATURE_BLOCK_OFFSETS["own_concealed_counts"]
        for tile_value in value.own_hand.concealed_tiles:
            index = own + tile_type_index(tile_value.tile_type)
            self.assertGreater(values[index], 0.0)
        drawn_block = FEATURE_BLOCK_OFFSETS["own_drawn_tile_type"]
        self.assertEqual(values[drawn_block + tile_type_index(drawn.tile_type)], 1.0)
        self.assertEqual(values[FEATURE_BLOCK_OFFSETS["own_has_drawn_tile"]], 1.0)

    def test_no_drawn_tile_leaves_the_block_empty(self) -> None:
        value = fixtures.policy_input(drawn=False)
        values = build_player_safe_feature(value)

        block = FEATURE_BLOCK_OFFSETS["own_drawn_tile_type"]
        self.assertEqual(
            values[block : block + TILE_TYPE_AXIS_SIZE], (0.0,) * TILE_TYPE_AXIS_SIZE
        )
        self.assertEqual(values[FEATURE_BLOCK_OFFSETS["own_has_drawn_tile"]], 0.0)

    def test_relative_seat_axis_follows_the_acting_seat(self) -> None:
        marker = fixtures.tile(fixtures.HONOR, 7)
        players = [
            fixtures.player_state(),
            fixtures.player_state(),
            fixtures.player_state(),
            fixtures.player_state(),
        ]
        players[2] = fixtures.player_state(
            discards=(fixtures.discard(marker, 0),), riichi=RiichiState.ACCEPTED
        )
        value = fixtures.policy_input(
            self_seat=Seat.SEAT_1, players=tuple(players), drawn=False
        )
        values = build_player_safe_feature(value)

        # absolute seat 2はseat 1から見て下家（relative index 1）である。
        block = FEATURE_BLOCK_OFFSETS["player[1:shimocha].discard_counts"]
        self.assertEqual(values[block + tile_type_index(marker.tile_type)], 0.25)
        riichi_block = FEATURE_BLOCK_OFFSETS["player[1:shimocha].riichi_state"]
        self.assertEqual(values[riichi_block + 2], 1.0)
        other = FEATURE_BLOCK_OFFSETS["player[0:self].discard_counts"]
        self.assertEqual(values[other + tile_type_index(marker.tile_type)], 0.0)

    def test_visible_counts_exclude_called_discards(self) -> None:
        called = fixtures.tile(fixtures.SOUZU, 9)
        players = [
            fixtures.player_state(
                discards=(fixtures.discard(called, 0, called_by=Seat.SEAT_1),)
            ),
            fixtures.player_state(
                melds=(
                    PublicMeld(
                        kind=MeldKind.PON,
                        tiles=(called, called, called),
                        from_seat=Seat.SEAT_0,
                        called_tile=called,
                    ),
                )
            ),
            fixtures.player_state(),
            fixtures.player_state(),
        ]
        value = fixtures.policy_input(
            self_seat=Seat.SEAT_2,
            concealed=(fixtures.tile(fixtures.MANZU, 1),),
            drawn=False,
            players=tuple(players),
        )
        values = build_player_safe_feature(value)

        visible = FEATURE_BLOCK_OFFSETS["visible_tile_counts"]
        # 副露へ吸収されたdiscardを二重に数えないため、9sの可視枚数は3枚である。
        self.assertEqual(values[visible + tile_type_index(called.tile_type)], 0.75)

    def test_tile_conservation_violation_fails_closed(self) -> None:
        tile_value = fixtures.tile(fixtures.PINZU, 3)
        players = [
            fixtures.player_state(
                discards=tuple(
                    fixtures.discard(tile_value, order) for order in range(4)
                )
            ),
            fixtures.player_state(),
            fixtures.player_state(),
            fixtures.player_state(),
        ]
        value = fixtures.policy_input(
            self_seat=Seat.SEAT_1,
            concealed=(tile_value,),
            drawn=False,
            players=tuple(players),
        )

        with self.assertRaisesRegex(FeatureError, "physical count"):
            build_player_safe_feature(value)

    def test_too_many_concealed_tiles_rejected(self) -> None:
        tiles = (
            *fixtures.hand(0),
            fixtures.tile(fixtures.HONOR, 2),
            fixtures.tile(fixtures.HONOR, 3),
        )
        value = fixtures.policy_input(concealed=tiles, drawn=False)

        with self.assertRaisesRegex(FeatureError, "concealed tiles"):
            build_player_safe_feature(value)

    def test_too_many_melds_rejected(self) -> None:
        meld = PublicMeld(
            kind=MeldKind.ANKAN,
            tiles=(fixtures.tile(fixtures.SOUZU, 1),) * 4,
            from_seat=None,
            called_tile=None,
        )
        players = [
            fixtures.player_state(melds=(meld,) * 5),
            fixtures.player_state(),
            fixtures.player_state(),
            fixtures.player_state(),
        ]
        value = fixtures.policy_input(
            self_seat=Seat.SEAT_1,
            concealed=(fixtures.tile(fixtures.MANZU, 1),),
            drawn=False,
            players=tuple(players),
        )

        with self.assertRaisesRegex(FeatureError, "melds"):
            build_player_safe_feature(value)

    def test_dora_indicator_bound_rejected(self) -> None:
        value = fixtures.policy_input(
            dora_indicators=tuple(
                fixtures.tile(fixtures.PINZU, rank) for rank in range(1, 7)
            )
        )

        with self.assertRaisesRegex(FeatureError, "dora indicators"):
            build_player_safe_feature(value)

    def test_live_wall_bound_rejected(self) -> None:
        value = fixtures.policy_input(live_wall_tiles_remaining=200)

        with self.assertRaisesRegex(FeatureError, "live_wall_tiles_remaining"):
            build_player_safe_feature(value)

    def test_own_hand_only_exposes_the_acting_seat(self) -> None:
        """他家のconcealed手牌はPolicyInput自体に存在せず、featureにも現れない。"""
        self.assertEqual(
            [field.name for field in OwnHandState.__dataclass_fields__.values()],
            ["concealed_tiles", "drawn_tile"],
        )
        first = fixtures.policy_input(self_seat=Seat.SEAT_0)
        # 自席以外の観測可能情報（public discards）を変えた場合だけfeatureが変わる。
        players = list(first.players)
        players[1] = fixtures.player_state(
            discards=(fixtures.discard(fixtures.tile(fixtures.HONOR, 2), 0),)
        )
        second = fixtures.policy_input(self_seat=Seat.SEAT_0, players=tuple(players))
        self.assertNotEqual(
            build_player_safe_feature(first), build_player_safe_feature(second)
        )


if __name__ == "__main__":
    unittest.main()
