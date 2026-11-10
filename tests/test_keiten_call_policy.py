"""Issue #375 KeitenCallPlacementAwareSpeedCallPolicyのfocused deterministic tests。"""

import unittest

import lisjong.policies.keiten_call_placement_aware_speed_call as keiten
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies import (
    KeitenCallPlacementAwareSpeedCallPolicy,
    PlacementAwareSpeedCallPolicy,
)
from lisjong.policy_contract.action import ChiAction, DiscardAction, PassAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind
from lisjong.structural_efficiency import post_discard_concealed_hand

_CATEGORIES = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}
WINDOW = keiten.KEITEN_MAXIMUM_LIVE_WALL


def _tile(spec: str) -> Tile:
    return Tile(TileType(_CATEGORIES[spec[1]], int(spec[0])), is_red=False)


def _hand(spec: str) -> tuple[Tile, ...]:
    tiles: list[Tile] = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        tiles.extend(_tile(rank + character) for rank in ranks)
        ranks = ""
    return tuple(tiles)


PASS = PassAction(actor=Seat.SEAT_0)


def _chi(called: str, consumed: str) -> ChiAction:
    first, second = _hand(consumed)
    return ChiAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_3,
        called_tile=_tile(called),
        consumed_tiles=(first, second),
    )


def _chi_meld(spec: str) -> PublicMeld:
    tiles = _hand(spec)
    return PublicMeld(
        kind=MeldKind.CHI, tiles=tiles, from_seat=Seat.SEAT_3, called_tile=tiles[0]
    )


def _player(
    *,
    melds: tuple[PublicMeld, ...] = (),
    riichi: RiichiState = RiichiState.NONE,
    discards: str = "",
) -> PlayerPublicState:
    return PlayerPublicState(
        score=25000,
        discards=tuple(
            Discard(tile=tile, tsumogiri=False, order=order, called_by=None)
            for order, tile in enumerate(_hand(discards))
        ),
        melds=melds,
        riichi=riichi,
    )


def _input(
    concealed: str,
    *,
    live_wall: int,
    own_melds: tuple[PublicMeld, ...] = (),
    riichi_discards: str | None = None,
) -> PolicyInput:
    players = [_player(melds=own_melds), _player(), _player(), _player()]
    if riichi_discards is not None:
        players[1] = _player(riichi=RiichiState.ACCEPTED, discards=riichi_discards)
    return PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_1,
            honba=0,
            riichi_sticks=0,
            dora_indicators=_hand("1s"),
            live_wall_tiles_remaining=live_wall,
        ),
        players=tuple(players),  # type: ignore[arg-type]
        own_hand=OwnHandState(concealed_tiles=_hand(concealed), drawn_tile=None),
    )


def _discards(policy_input: PolicyInput) -> tuple[DiscardAction, ...]:
    seen: list[Tile] = []
    for tile in policy_input.own_hand.concealed_tiles:
        if tile not in seen:
            seen.append(tile)
    return tuple(
        DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=False) for tile in seen
    )


def _keeps_tenpai(policy_input: PolicyInput, action: DiscardAction) -> bool:
    return (
        calculate_shanten(
            post_discard_concealed_hand(
                policy_input.own_hand.concealed_tiles, action.tile
            )
        )
        == 0
    )


# 123m 789m 456p 78s + 9p 1z: 門前1向聴。9sチー・打9p / 打1zで役なしの単騎聴牌。
ONE_SHANTEN = "123789m4569p78s1z"
KEITEN_CHI = _chi("9s", "78s")
# 上の手で9sをチーした直後の純手牌（副露789s）。
AFTER_CHI = "123789m4569p1z"
AFTER_CHI_MELDS = (_chi_meld("978s"),)


class KeitenCallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = PlacementAwareSpeedCallPolicy()
        self.policy = KeitenCallPlacementAwareSpeedCallPolicy()

    def _choose(self, concealed: str, actions, **kwargs):
        decision = DecisionContext(
            input=_input(concealed, **kwargs), legal_actions=actions
        )
        self.assertIs(self.parent.choose_action(decision), PASS)
        return self.policy.choose_action(decision)

    def test_fixture_is_closed_one_shanten(self) -> None:
        self.assertEqual(calculate_shanten(_hand(ONE_SHANTEN)), 1)

    def test_call_to_tenpai_is_taken_inside_window(self) -> None:
        self.assertIs(
            self._choose(ONE_SHANTEN, (PASS, KEITEN_CHI), live_wall=WINDOW),
            KEITEN_CHI,
        )

    def test_outside_window_keeps_parent_pass(self) -> None:
        self.assertIs(
            self._choose(ONE_SHANTEN, (PASS, KEITEN_CHI), live_wall=WINDOW + 1),
            PASS,
        )

    def test_call_that_does_not_reach_tenpai_is_rejected(self) -> None:
        # 123m 789m 45p 78s + 9p 1z 3z: 2向聴。9sチーでも1向聴止まり。
        self.assertIs(
            self._choose("123789m459p78s13z", (PASS, KEITEN_CHI), live_wall=WINDOW),
            PASS,
        )

    def test_riichi_requires_genbutsu_mandatory_discard(self) -> None:
        self.assertIs(
            self._choose(
                ONE_SHANTEN,
                (PASS, KEITEN_CHI),
                live_wall=WINDOW,
                riichi_discards="2s5s",
            ),
            PASS,
        )
        self.assertIs(
            self._choose(
                ONE_SHANTEN,
                (PASS, KEITEN_CHI),
                live_wall=WINDOW,
                riichi_discards="2s1z",
            ),
            KEITEN_CHI,
        )

    def test_kuikae_forbidden_discard_is_not_counted(self) -> None:
        # 123m 789m 456p + 6789s: 9sチー(78s)後に聴牌となる打牌は6s / 9sだけで、
        # どちらも喰い替えで打てない。
        policy_input = _input("123789m456p6789s", live_wall=WINDOW)
        self.assertFalse(keiten._call_reaches_tenpai(policy_input, KEITEN_CHI, None))

    def test_selection_does_not_depend_on_legal_action_order(self) -> None:
        other = _chi("6s", "78s")
        hand = "123789m4569p78s1z"
        first = self._choose(hand, (PASS, KEITEN_CHI, other), live_wall=WINDOW)
        second = self._choose(hand, (other, KEITEN_CHI, PASS), live_wall=WINDOW)
        self.assertEqual(first, second)
        self.assertIsInstance(first, ChiAction)


class TenpaiKeepingDiscardTest(unittest.TestCase):
    def test_open_hand_inside_window_keeps_tenpai(self) -> None:
        policy_input = _input(AFTER_CHI, live_wall=WINDOW, own_melds=AFTER_CHI_MELDS)
        actions = _discards(policy_input)
        eligible = keiten._tenpai_keeping_actions(policy_input, actions)
        self.assertEqual({action.tile for action in eligible}, set(_hand("9p1z")))
        selected = KeitenCallPlacementAwareSpeedCallPolicy().choose_action(
            DecisionContext(input=policy_input, legal_actions=actions)
        )
        self.assertTrue(_keeps_tenpai(policy_input, selected))  # type: ignore[arg-type]

    def test_riichi_prefers_genbutsu_among_tenpai_keeping(self) -> None:
        policy_input = _input(
            AFTER_CHI,
            live_wall=WINDOW,
            own_melds=AFTER_CHI_MELDS,
            riichi_discards="1z3m",
        )
        actions = _discards(policy_input)
        eligible = keiten._tenpai_keeping_actions(policy_input, actions)
        self.assertEqual({action.tile for action in eligible}, set(_hand("1z")))
        selected = KeitenCallPlacementAwareSpeedCallPolicy().choose_action(
            DecisionContext(input=policy_input, legal_actions=actions)
        )
        self.assertEqual(selected.tile, _tile("1z"))  # type: ignore[union-attr]

    def test_riichi_without_safe_tenpai_keeping_keeps_all_tenpai_discards(self) -> None:
        policy_input = _input(
            AFTER_CHI,
            live_wall=WINDOW,
            own_melds=AFTER_CHI_MELDS,
            riichi_discards="3m",
        )
        eligible = keiten._tenpai_keeping_actions(policy_input, _discards(policy_input))
        self.assertEqual({action.tile for action in eligible}, set(_hand("9p1z")))

    def test_unchanged_outside_window_closed_hand_or_without_tenpai(self) -> None:
        cases = (
            _input(AFTER_CHI, live_wall=WINDOW + 1, own_melds=AFTER_CHI_MELDS),
            _input("123789m456p78s19p1z", live_wall=WINDOW),
            _input("123789m4589p1z", live_wall=WINDOW, own_melds=AFTER_CHI_MELDS),
        )
        for policy_input in cases:
            actions = _discards(policy_input)
            self.assertIs(
                keiten._tenpai_keeping_actions(policy_input, actions), actions
            )

    def test_outside_window_equals_parent_for_discards(self) -> None:
        policy_input = _input(
            AFTER_CHI, live_wall=WINDOW + 1, own_melds=AFTER_CHI_MELDS
        )
        decision = DecisionContext(
            input=policy_input, legal_actions=_discards(policy_input)
        )
        self.assertEqual(
            KeitenCallPlacementAwareSpeedCallPolicy().choose_action(decision),
            PlacementAwareSpeedCallPolicy().choose_action(decision),
        )


if __name__ == "__main__":
    unittest.main()
