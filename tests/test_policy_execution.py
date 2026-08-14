import inspect
import unittest
from dataclasses import replace

from lisjong.policies import MinimalPolicy
from lisjong.policy_contract import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DecisionContext,
    DiscardAction,
    InternalAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    PassAction,
    PolicyActionValidationError,
    PonAction,
    RiichiAction,
    RonAction,
    TsumoAction,
    execute_policy,
)
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
MANZU_6 = Tile(TileType(TileCategory.MANZU, 6))
PINZU_5 = Tile(TileType(TileCategory.PINZU, 5))


def _make_input() -> PolicyInput:
    player = PlayerPublicState(
        score=25000, discards=(), melds=(), riichi=RiichiState.NONE
    )
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(MANZU_3,),
            live_wall_tiles_remaining=70,
        ),
        players=(player, player, player, player),
        own_hand=OwnHandState(
            concealed_tiles=(MANZU_4, MANZU_5, MANZU_6),
            drawn_tile=MANZU_5,
        ),
    )


def _decision(*actions: InternalAction) -> DecisionContext:
    return DecisionContext(input=_make_input(), legal_actions=actions)


class _ReturningPolicy:
    def __init__(self, selected: object) -> None:
        self.selected = selected
        self.received: DecisionContext | None = None

    def choose_action(self, decision: DecisionContext) -> object:
        self.received = decision
        return self.selected


class _RaisingPolicy:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        raise self.error


class _AmbiguousPassAction(PassAction):
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        return True


class _UncomparablePassAction(PassAction):
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("comparison failed")


class PolicyExecutionTest(unittest.TestCase):
    def test_minimal_policy_can_be_called_through_boundary(self) -> None:
        legal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
        decision = _decision(legal, PassAction(actor=Seat.SEAT_0))

        selected = execute_policy(MinimalPolicy(), decision)

        self.assertIs(selected, legal)

    def test_passes_the_exact_decision_context_object_to_policy(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)
        decision = _decision(legal)
        policy = _ReturningPolicy(legal)

        execute_policy(policy, decision)

        self.assertIs(policy.received, decision)

    def test_returns_canonical_legal_candidate_for_equal_distinct_result(self) -> None:
        legal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
        policy_result = replace(legal)
        self.assertIsNot(policy_result, legal)

        selected = execute_policy(_ReturningPolicy(policy_result), _decision(legal))

        self.assertIs(selected, legal)

    def test_uses_one_common_validation_path_for_all_action_variants(self) -> None:
        actions: tuple[InternalAction, ...] = (
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False),
            RiichiAction(actor=Seat.SEAT_0),
            ChiAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_3,
                called_tile=MANZU_5,
                consumed_tiles=(MANZU_4, MANZU_6),
            ),
            PonAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_1,
                called_tile=PINZU_5,
                consumed_tiles=(PINZU_5, PINZU_5),
            ),
            DaiminkanAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_1,
                called_tile=PINZU_5,
                consumed_tiles=(PINZU_5, PINZU_5, PINZU_5),
            ),
            AnkanAction(
                actor=Seat.SEAT_0,
                tiles=(PINZU_5, PINZU_5, PINZU_5, PINZU_5),
            ),
            KakanAction(
                actor=Seat.SEAT_0,
                added_tile=PINZU_5,
                from_seat=Seat.SEAT_1,
                called_tile=PINZU_5,
            ),
            RonAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_1,
                winning_tile=PINZU_5,
            ),
            TsumoAction(actor=Seat.SEAT_0, winning_tile=PINZU_5),
            PassAction(actor=Seat.SEAT_0),
            KyuushuKyuuhaiAction(actor=Seat.SEAT_0),
        )

        for legal in actions:
            with self.subTest(action_type=type(legal).__name__):
                selected = execute_policy(
                    _ReturningPolicy(replace(legal)), _decision(legal)
                )
                self.assertIs(selected, legal)

    def test_accepts_duck_typed_policy_without_inheritance(self) -> None:
        legal = PassAction(actor=Seat.SEAT_0)

        selected = execute_policy(_ReturningPolicy(legal), _decision(legal))

        self.assertIs(selected, legal)

    def test_rejects_action_outside_legal_actions_with_zero_matches(self) -> None:
        legal = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=False)
        outside = DiscardAction(actor=Seat.SEAT_0, tile=MANZU_6, tsumogiri=False)

        with self.assertRaisesRegex(PolicyActionValidationError, "found 0 matches"):
            execute_policy(_ReturningPolicy(outside), _decision(legal))

    def test_rejects_non_action_result(self) -> None:
        with self.assertRaisesRegex(
            PolicyActionValidationError, "must return an InternalAction"
        ):
            execute_policy(
                _ReturningPolicy(None), _decision(PassAction(actor=Seat.SEAT_0))
            )

    def test_rejects_multiple_semantic_matches(self) -> None:
        decision = _decision(
            PassAction(actor=Seat.SEAT_0), RiichiAction(actor=Seat.SEAT_0)
        )

        with self.assertRaisesRegex(PolicyActionValidationError, "found 2 matches"):
            execute_policy(
                _ReturningPolicy(_AmbiguousPassAction(actor=Seat.SEAT_0)),
                decision,
            )

    def test_rejects_result_that_cannot_be_compared_safely(self) -> None:
        with self.assertRaisesRegex(
            PolicyActionValidationError, "could not be compared safely"
        ) as caught:
            execute_policy(
                _ReturningPolicy(_UncomparablePassAction(actor=Seat.SEAT_0)),
                _decision(PassAction(actor=Seat.SEAT_0)),
            )

        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_policy_exception_propagates_unchanged_without_fallback(self) -> None:
        error = RuntimeError("policy failed")

        with self.assertRaises(RuntimeError) as caught:
            execute_policy(
                _RaisingPolicy(error), _decision(PassAction(actor=Seat.SEAT_0))
            )

        self.assertIs(caught.exception, error)

    def test_public_api_requires_only_policy_and_decision(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(execute_policy).parameters),
            ("policy", "decision"),
        )


if __name__ == "__main__":
    unittest.main()
