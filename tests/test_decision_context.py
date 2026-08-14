import unittest
from dataclasses import FrozenInstanceError, fields

from lisjong.policy_contract.action import (
    ChiAction,
    DiscardAction,
    KyuushuKyuuhaiAction,
    PassAction,
    PonAction,
    RiichiAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

MANZU_3 = Tile(TileType(TileCategory.MANZU, 3))
MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))
MANZU_5 = Tile(TileType(TileCategory.MANZU, 5))
MANZU_5_RED = Tile(TileType(TileCategory.MANZU, 5), is_red=True)
MANZU_6 = Tile(TileType(TileCategory.MANZU, 6))
PINZU_5 = Tile(TileType(TileCategory.PINZU, 5))


def _make_player(score: int = 25000) -> PlayerPublicState:
    return PlayerPublicState(
        score=score, discards=(), melds=(), riichi=RiichiState.NONE
    )


def _make_players() -> tuple[
    PlayerPublicState, PlayerPublicState, PlayerPublicState, PlayerPublicState
]:
    return (_make_player(), _make_player(), _make_player(), _make_player())


def _make_round() -> RoundState:
    return RoundState(
        round_wind=Wind.EAST,
        hand_number=1,
        dealer_seat=Seat.SEAT_0,
        honba=0,
        riichi_sticks=0,
        dora_indicators=(MANZU_3,),
        live_wall_tiles_remaining=70,
    )


def _make_own_hand() -> OwnHandState:
    return OwnHandState(
        concealed_tiles=(MANZU_4, MANZU_5, MANZU_5_RED, MANZU_6, PINZU_5),
        drawn_tile=MANZU_5,
    )


def _make_input(self_seat: Seat = Seat.SEAT_0) -> PolicyInput:
    return PolicyInput(
        self_seat=self_seat,
        round=_make_round(),
        players=_make_players(),
        own_hand=_make_own_hand(),
    )


def _make(**overrides: object) -> DecisionContext:
    kwargs: dict[str, object] = {
        "input": _make_input(),
        "legal_actions": (
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5, tsumogiri=True),
        ),
    }
    kwargs.update(overrides)
    return DecisionContext(**kwargs)


class DecisionContextTest(unittest.TestCase):
    def test_creates_with_valid_values(self) -> None:
        context = _make()
        self.assertIsInstance(context.input, PolicyInput)
        self.assertEqual(len(context.legal_actions), 2)

    def test_is_immutable(self) -> None:
        context = _make()
        with self.assertRaises(FrozenInstanceError):
            context.legal_actions = ()

    def test_equal_values_compare_equal_and_are_hashable(self) -> None:
        first = _make()
        second = _make()
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first, second}), 1)

    def test_normalizes_list_legal_actions_into_tuple(self) -> None:
        context = _make(
            legal_actions=[
                DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
            ]
        )
        self.assertIsInstance(context.legal_actions, tuple)

    def test_preserves_input_sequence_order(self) -> None:
        actions = (
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_6, tsumogiri=False),
            PassAction(actor=Seat.SEAT_0),
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
        )
        context = _make(legal_actions=actions)
        self.assertEqual(context.legal_actions, actions)

    def test_allows_mixed_variants(self) -> None:
        actions = (
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
            RiichiAction(actor=Seat.SEAT_0),
            PassAction(actor=Seat.SEAT_0),
            KyuushuKyuuhaiAction(actor=Seat.SEAT_0),
            ChiAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_3,
                called_tile=MANZU_5,
                consumed_tiles=(MANZU_4, MANZU_6),
            ),
        )
        context = _make(legal_actions=actions)
        self.assertEqual(len(context.legal_actions), 5)

    def test_rejects_empty_legal_actions(self) -> None:
        with self.assertRaises(ValueError):
            _make(legal_actions=())

    def test_rejects_non_policy_input(self) -> None:
        with self.assertRaises(TypeError):
            _make(input="not a policy input")

    def test_rejects_non_iterable_legal_actions(self) -> None:
        with self.assertRaises(TypeError):
            _make(legal_actions=123)

    def test_rejects_non_internal_action_in_legal_actions(self) -> None:
        with self.assertRaises(TypeError):
            _make(
                legal_actions=(
                    DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
                    "not an action",
                )
            )

    def test_rejects_action_actor_mismatching_self_seat(self) -> None:
        with self.assertRaises(ValueError):
            _make(
                input=_make_input(self_seat=Seat.SEAT_1),
                legal_actions=(
                    DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
                ),
            )

    def test_accepts_all_actors_matching_self_seat(self) -> None:
        context = _make(
            input=_make_input(self_seat=Seat.SEAT_2),
            legal_actions=(
                DiscardAction(actor=Seat.SEAT_2, tile=MANZU_4, tsumogiri=False),
                PassAction(actor=Seat.SEAT_2),
            ),
        )
        self.assertEqual(context.input.self_seat, Seat.SEAT_2)

    def test_rejects_duplicate_semantic_discard_action(self) -> None:
        with self.assertRaises(ValueError):
            _make(
                legal_actions=(
                    DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
                    DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
                )
            )

    def test_rejects_duplicate_semantic_pon_action_with_different_input_order(
        self,
    ) -> None:
        # consumed_tilesの入力順だけが異なるPonActionは、constructorで
        # canonicalize済みのため同一semantic actionになる。
        with self.assertRaises(ValueError):
            _make(
                legal_actions=(
                    PonAction(
                        actor=Seat.SEAT_0,
                        target=Seat.SEAT_1,
                        called_tile=PINZU_5,
                        consumed_tiles=(PINZU_5, PINZU_5),
                    ),
                    PonAction(
                        actor=Seat.SEAT_0,
                        target=Seat.SEAT_1,
                        called_tile=PINZU_5,
                        consumed_tiles=(PINZU_5, PINZU_5),
                    ),
                )
            )

    def test_distinguishes_tsumogiri_difference(self) -> None:
        context = _make(
            legal_actions=(
                DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
                DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=True),
            )
        )
        self.assertEqual(len(context.legal_actions), 2)

    def test_distinguishes_normal_and_red_tile(self) -> None:
        context = _make(
            legal_actions=(
                DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5, tsumogiri=True),
                DiscardAction(actor=Seat.SEAT_0, tile=MANZU_5_RED, tsumogiri=True),
            )
        )
        self.assertEqual(len(context.legal_actions), 2)

    def test_distinguishes_different_variants_with_same_actor(self) -> None:
        context = _make(
            legal_actions=(
                PassAction(actor=Seat.SEAT_0),
                RiichiAction(actor=Seat.SEAT_0),
                KyuushuKyuuhaiAction(actor=Seat.SEAT_0),
            )
        )
        self.assertEqual(len(context.legal_actions), 3)

    def test_distinguishes_different_variant_specific_fields(self) -> None:
        context = _make(
            legal_actions=(
                DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
                DiscardAction(actor=Seat.SEAT_0, tile=MANZU_6, tsumogiri=False),
            )
        )
        self.assertEqual(len(context.legal_actions), 2)

    def test_has_exactly_the_documented_fields(self) -> None:
        field_names = {field.name for field in fields(DecisionContext)}
        self.assertEqual(field_names, {"input", "legal_actions"})
        for forbidden in (
            "selected_action",
            "fallback_action",
            "preferred_action",
            "action_map",
            "external_action_ids",
            "decision_id",
            "turn_id",
            "riichienv_observation",
        ):
            self.assertNotIn(forbidden, field_names)


if __name__ == "__main__":
    unittest.main()
