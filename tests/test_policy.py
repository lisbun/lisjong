import inspect
import unittest

from lisjong.policy_contract.action import DiscardAction, InternalAction, PassAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy import Policy
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

MANZU_3 = Tile(TileType(TileCategory.MANZU, 3))
MANZU_4 = Tile(TileType(TileCategory.MANZU, 4))


def _make_decision_context() -> DecisionContext:
    player = PlayerPublicState(
        score=25000, discards=(), melds=(), riichi=RiichiState.NONE
    )
    policy_input = PolicyInput(
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
        own_hand=OwnHandState(concealed_tiles=(MANZU_4,), drawn_tile=MANZU_4),
    )
    return DecisionContext(
        input=policy_input,
        legal_actions=(
            DiscardAction(actor=Seat.SEAT_0, tile=MANZU_4, tsumogiri=True),
            PassAction(actor=Seat.SEAT_0),
        ),
    )


class _ExamplePolicy:
    """`Policy`を明示的に継承しない、structurallyに適合するtest double。

    `@runtime_checkable`を付けていないため、isinstance()での適合確認は
    行わず、実際に呼び出せることだけを確認する。
    """

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        return decision.legal_actions[0]


class PolicyProtocolTest(unittest.TestCase):
    def test_is_importable_from_package_root(self) -> None:
        from lisjong.policy_contract import Policy as ReExportedPolicy

        self.assertIs(ReExportedPolicy, Policy)

    def test_declares_choose_action(self) -> None:
        self.assertTrue(hasattr(Policy, "choose_action"))

    def test_choose_action_signature_matches_the_contract(self) -> None:
        signature = inspect.signature(Policy.choose_action)
        self.assertEqual(list(signature.parameters), ["self", "decision"])
        self.assertIs(signature.parameters["decision"].annotation, DecisionContext)
        self.assertEqual(signature.return_annotation, InternalAction)

    def test_has_no_extra_public_methods_or_properties(self) -> None:
        public_members = [name for name in vars(Policy) if not name.startswith("_")]
        self.assertEqual(public_members, ["choose_action"])

    def test_structurally_compatible_object_can_be_used_without_inheritance(
        self,
    ) -> None:
        decision = _make_decision_context()
        policy: Policy = _ExamplePolicy()
        selected = policy.choose_action(decision)
        self.assertIn(selected, decision.legal_actions)


if __name__ == "__main__":
    unittest.main()
