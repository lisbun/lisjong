"""Issue #377 RiichiResponsePlacementAwareSpeedCallPolicyのfocused deterministic tests。"""

import unittest

import lisjong.policies.riichi_response_placement_aware_speed_call as response
from lisjong.policies import (
    PlacementAwareSpeedCallPolicy,
    RiichiResponseParameters,
    RiichiResponsePlacementAwareSpeedCallPolicy,
    riichi_response_search_space,
)
from lisjong.policy_contract.action import DiscardAction
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

_CATEGORIES = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}


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
    drawn: str,
    riichi_discards: str | None,
    dora_indicators: str,
    live_wall: int = 40,
    own_melds: tuple[PublicMeld, ...] = (),
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
            dora_indicators=_hand(dora_indicators),
            live_wall_tiles_remaining=live_wall,
        ),
        players=tuple(players),  # type: ignore[arg-type]
        own_hand=OwnHandState(
            concealed_tiles=_hand(concealed), drawn_tile=_tile(drawn)
        ),
    )


def _decision(policy_input: PolicyInput) -> DecisionContext:
    seen: list[Tile] = []
    for tile in policy_input.own_hand.concealed_tiles:
        if tile not in seen:
            seen.append(tile)
    return DecisionContext(
        input=policy_input,
        legal_actions=tuple(
            DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=False)
            for tile in seen
        ),
    )


def _parameters(push: int | None = None, fold: int | None = None, wall: int = 24):
    return RiichiResponseParameters(
        push_minimum_dora=push, push_minimum_live_wall=wall, fold_maximum_dora=fold
    )


def _choose(parameters: RiichiResponseParameters, policy_input: PolicyInput) -> Tile:
    policy = RiichiResponsePlacementAwareSpeedCallPolicy(parameters)
    return policy.choose_action(_decision(policy_input)).tile  # type: ignore[union-attr]


def _champion(policy_input: PolicyInput) -> Tile:
    return PlacementAwareSpeedCallPolicy().choose_action(_decision(policy_input)).tile  # type: ignore[union-attr]


# 123m 456p 78s 34s 55p + 9m 1z（14枚、1zがツモ）: どの打牌でも聴牌にならず、
# 9m / 1zを切れば1向聴。8sがdora（表示牌7s）でdora 1枚。
ONE_SHANTEN = "1239m45655p3478s1z"
DORA_ONE = "7s"
DORA_ZERO = "1p"
# リーチ者の河に3sだけがある: 共通現物は3s（切ると塔子が崩れる）。
GENBUTSU_3S = "3s"
# リーチ者の河が手牌と重ならない: 共通現物なし。
NO_GENBUTSU = "2z"


class ParametersTest(unittest.TestCase):
    def test_rejects_invalid_values(self) -> None:
        for kwargs in (
            {"push": -1},
            {"fold": -1},
            {"wall": -1},
            {"push": True},
        ):
            with self.assertRaises(ValueError):
                _parameters(**kwargs)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RiichiResponsePlacementAwareSpeedCallPolicy(None)  # type: ignore[arg-type]

    def test_identity_is_derived_from_values(self) -> None:
        self.assertEqual(
            _parameters(push=2, fold=None).identity,
            "placement-aware-speed-call-riichi-response-v1"
            ":push-dora-2:push-wall-24:fold-dora-none",
        )
        self.assertNotEqual(
            _parameters(push=1).identity, _parameters(push=1, wall=30).identity
        )

    def test_search_space_is_deterministic(self) -> None:
        space = riichi_response_search_space()
        self.assertEqual(space, riichi_response_search_space())
        self.assertEqual(
            [(item.push_minimum_dora, item.fold_maximum_dora) for item in space],
            [
                (None, 0),
                (None, 1),
                (1, None),
                (1, 0),
                (1, 1),
                (2, None),
                (2, 0),
                (2, 1),
            ],
        )
        self.assertTrue(all(item.push_minimum_live_wall == 24 for item in space))
        self.assertEqual(len({item.identity for item in space}), len(space))
        self.assertEqual(
            response.RIICHI_RESPONSE_SEARCH_SPACE_VERSION,
            "riichi-response-search-space-v1",
        )


class PushTest(unittest.TestCase):
    def setUp(self) -> None:
        self.riichi = _input(
            ONE_SHANTEN,
            drawn="1z",
            riichi_discards=GENBUTSU_3S,
            dora_indicators=DORA_ONE,
        )
        self.no_riichi = _input(
            ONE_SHANTEN, drawn="1z", riichi_discards=None, dora_indicators=DORA_ONE
        )

    def test_champion_folds_to_genbutsu(self) -> None:
        self.assertEqual(_champion(self.riichi), _tile("3s"))
        self.assertNotEqual(_champion(self.no_riichi), _tile("3s"))

    def test_push_selects_the_no_riichi_discard(self) -> None:
        self.assertEqual(
            _choose(_parameters(push=1), self.riichi), _champion(self.no_riichi)
        )

    def test_push_requires_enough_dora(self) -> None:
        self.assertEqual(_choose(_parameters(push=2), self.riichi), _tile("3s"))

    def test_push_requires_enough_live_wall(self) -> None:
        late = _input(
            ONE_SHANTEN,
            drawn="1z",
            riichi_discards=GENBUTSU_3S,
            dora_indicators=DORA_ONE,
            live_wall=23,
        )
        self.assertEqual(_choose(_parameters(push=1), late), _tile("3s"))

    def test_push_requires_closed_hand(self) -> None:
        pon = PublicMeld(
            kind=MeldKind.PON,
            tiles=(_tile("5z"),) * 3,
            from_seat=Seat.SEAT_2,
            called_tile=_tile("5z"),
        )
        opened = _input(
            "45655p3478s9m1z",
            drawn="1z",
            riichi_discards=GENBUTSU_3S,
            dora_indicators=DORA_ONE,
            own_melds=(pon,),
        )
        self.assertEqual(_choose(_parameters(push=1), opened), _champion(opened))

    def test_fold_rule_does_not_act_when_genbutsu_exists(self) -> None:
        self.assertEqual(_choose(_parameters(fold=1), self.riichi), _tile("3s"))


class FoldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.riichi = _input(
            ONE_SHANTEN,
            drawn="1z",
            riichi_discards=NO_GENBUTSU,
            dora_indicators=DORA_ZERO,
        )

    def test_fold_selects_a_minimum_danger_discard(self) -> None:
        decision = _decision(self.riichi)
        safest = response._minimum_danger_actions(self.riichi, decision.legal_actions)
        self.assertLess(len(safest), len(decision.legal_actions))
        self.assertIn(
            _choose(_parameters(fold=0), self.riichi),
            {action.tile for action in safest},
        )

    def test_fold_requires_low_dora(self) -> None:
        valuable = _input(
            ONE_SHANTEN,
            drawn="1z",
            riichi_discards=NO_GENBUTSU,
            dora_indicators=DORA_ONE,
        )
        self.assertEqual(_choose(_parameters(fold=0), valuable), _champion(valuable))

    def test_push_rule_does_not_act_without_genbutsu(self) -> None:
        self.assertEqual(
            _choose(_parameters(push=0), self.riichi), _champion(self.riichi)
        )


class UnchangedTest(unittest.TestCase):
    BOTH = _parameters(push=0, fold=9)

    def _assert_same(self, policy_input: PolicyInput) -> None:
        self.assertEqual(_choose(self.BOTH, policy_input), _champion(policy_input))

    def test_without_riichi(self) -> None:
        self._assert_same(
            _input(
                ONE_SHANTEN, drawn="1z", riichi_discards=None, dora_indicators=DORA_ONE
            )
        )

    def test_tenpai_reachable_hand(self) -> None:
        # 123m 456p 789s 34s 55p + 1z: 1zを切れば聴牌。
        self._assert_same(
            _input(
                "123m45655p34789s1z",
                drawn="1z",
                riichi_discards=GENBUTSU_3S,
                dora_indicators=DORA_ONE,
            )
        )

    def test_two_shanten_hand(self) -> None:
        # 123m 45p 78s 34s 55p + 9m 1z 2z: 打牌後も2向聴。
        self._assert_same(
            _input(
                "1239m4555p3478s12z",
                drawn="2z",
                riichi_discards=GENBUTSU_3S,
                dora_indicators=DORA_ONE,
            )
        )

    def test_parameters_without_rules_equal_champion(self) -> None:
        for riichi_discards in (GENBUTSU_3S, NO_GENBUTSU):
            policy_input = _input(
                ONE_SHANTEN,
                drawn="1z",
                riichi_discards=riichi_discards,
                dora_indicators=DORA_ONE,
            )
            self.assertEqual(
                _choose(_parameters(), policy_input), _champion(policy_input)
            )


if __name__ == "__main__":
    unittest.main()
