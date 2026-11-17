"""Issue #384 リーチ下1向聴の打点推定候補のfocused deterministic tests。"""

import unittest
from dataclasses import replace
from fractions import Fraction

from test_riichi_response_policy import (
    DORA_ONE,
    DORA_ZERO,
    GENBUTSU_3S,
    NO_GENBUTSU,
    ONE_SHANTEN,
    _decision,
    _input,
    _tile,
)

import lisjong.policies.riichi_response_hand_value as hand_value
import lisjong.policies.riichi_response_placement_aware_speed_call as response
from lisjong.policies import (
    PlacementAwareSpeedCallPolicy,
    RiichiResponseHandValueParameters,
    RiichiResponseHandValuePlacementAwareSpeedCallPolicy,
    riichi_response_hand_value_search_space,
)
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile

# 白ポン + 456p 55p 34s 78s 9m 1z: 9m / 1zを切れば1向聴。役は白（1翻）。
HAKU_OPEN = "45655p3478s9m1z"
HAKU_PON = PublicMeld(
    kind=MeldKind.PON,
    tiles=(_tile("5z"),) * 3,
    from_seat=Seat.SEAT_2,
    called_tile=_tile("5z"),
)
# 123mチー + 同じ純手牌: 1mを含むのでタンヤオがなく、役がない。
NO_YAKU_CHI = PublicMeld(
    kind=MeldKind.CHI,
    tiles=(_tile("1m"), _tile("2m"), _tile("3m")),
    from_seat=Seat.SEAT_3,
    called_tile=_tile("1m"),
)


def _estimate(policy_input: PolicyInput) -> Fraction:
    return hand_value.estimate_one_shanten_hand_value(
        policy_input, _decision(policy_input).legal_actions
    )


def _parameters(
    closed: int | None = None,
    opened: int | None = None,
    fold: int | None = None,
    wall: int = 24,
) -> RiichiResponseHandValueParameters:
    return RiichiResponseHandValueParameters(
        push_closed_minimum_points=closed,
        push_open_minimum_points=opened,
        push_minimum_live_wall=wall,
        fold_maximum_points=fold,
    )


def _choose(
    parameters: RiichiResponseHandValueParameters, policy_input: PolicyInput
) -> Tile:
    policy = RiichiResponseHandValuePlacementAwareSpeedCallPolicy(parameters)
    return policy.choose_action(_decision(policy_input)).tile  # type: ignore[union-attr]


def _champion(policy_input: PolicyInput) -> Tile:
    return PlacementAwareSpeedCallPolicy().choose_action(_decision(policy_input)).tile  # type: ignore[union-attr]


def _without_riichi(policy_input: PolicyInput) -> PolicyInput:
    return response._without_opponent_riichi(policy_input)


class EstimateTest(unittest.TestCase):
    def test_closed_hand_uses_the_riichi_ron_value(self) -> None:
        # リーチ・平和の30符2翻（子のロン2,000点）、doraが1枚あれば3翻3,900点。
        for indicators, expected in ((DORA_ZERO, 2000), (DORA_ONE, 3900)):
            policy_input = _input(
                ONE_SHANTEN,
                drawn="1z",
                riichi_discards=GENBUTSU_3S,
                dora_indicators=indicators,
            )
            self.assertEqual(_estimate(policy_input), Fraction(expected))

    def test_dealer_value_is_converted_to_non_dealer_points(self) -> None:
        policy_input = _input(
            ONE_SHANTEN, drawn="1z", riichi_discards=GENBUTSU_3S, dora_indicators="1p"
        )
        dealer = replace(
            policy_input, round=replace(policy_input.round, dealer_seat=Seat.SEAT_0)
        )
        # 親のリーチ・平和は2,900点。
        self.assertEqual(_estimate(dealer), Fraction(2900) * Fraction(2, 3))

    def test_open_hand_counts_only_its_yaku(self) -> None:
        for indicators, expected in ((DORA_ZERO, 1000), (DORA_ONE, 2000)):
            policy_input = _input(
                HAKU_OPEN,
                drawn="1z",
                riichi_discards=GENBUTSU_3S,
                dora_indicators=indicators,
                own_melds=(HAKU_PON,),
            )
            self.assertEqual(_estimate(policy_input), Fraction(expected))

    def test_open_hand_without_yaku_is_zero_even_with_dora(self) -> None:
        policy_input = _input(
            HAKU_OPEN,
            drawn="1z",
            riichi_discards=GENBUTSU_3S,
            dora_indicators=DORA_ONE,
            own_melds=(NO_YAKU_CHI,),
        )
        self.assertEqual(_estimate(policy_input), Fraction(0))

    def test_estimate_requires_a_one_shanten_discard(self) -> None:
        two_shanten = _input(
            "1239m4555p3478s12z",
            drawn="2z",
            riichi_discards=GENBUTSU_3S,
            dora_indicators=DORA_ONE,
        )
        with self.assertRaises(hand_value.HandValueEstimateError):
            _estimate(two_shanten)

    def test_estimate_does_not_depend_on_opponent_riichi(self) -> None:
        policy_input = _input(
            ONE_SHANTEN, drawn="1z", riichi_discards=GENBUTSU_3S, dora_indicators="7s"
        )
        self.assertEqual(
            _estimate(policy_input), _estimate(_without_riichi(policy_input))
        )


class ParametersTest(unittest.TestCase):
    def test_rejects_invalid_values(self) -> None:
        for kwargs in (
            {"closed": -1},
            {"opened": -1},
            {"fold": -1},
            {"wall": -1},
            {"closed": True},
            {"fold": 1.5},
        ):
            with self.assertRaises(ValueError):
                _parameters(**kwargs)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RiichiResponseHandValuePlacementAwareSpeedCallPolicy(None)  # type: ignore[arg-type]

    def test_identity_is_derived_from_values(self) -> None:
        self.assertEqual(
            _parameters(closed=5200, fold=2000).identity,
            "placement-aware-speed-call-riichi-response-hand-value-v1"
            ":push-closed-5200:push-open-none:push-wall-24:fold-2000",
        )
        self.assertNotEqual(
            _parameters(closed=5200).identity, _parameters(opened=5200).identity
        )

    def test_search_space_is_fixed(self) -> None:
        space = riichi_response_hand_value_search_space()
        self.assertEqual(space, riichi_response_hand_value_search_space())
        self.assertEqual(
            [
                (
                    item.push_closed_minimum_points,
                    item.push_open_minimum_points,
                    item.fold_maximum_points,
                )
                for item in space
            ],
            [
                (5200, None, 2000),
                (5200, None, 3900),
                (5200, 8000, 2000),
                (5200, 8000, 3900),
                (8000, None, 2000),
                (8000, None, 3900),
                (8000, 8000, 2000),
                (8000, 8000, 3900),
            ],
        )
        self.assertTrue(all(item.push_minimum_live_wall == 24 for item in space))
        self.assertEqual(len({item.identity for item in space}), 8)
        self.assertEqual(
            hand_value.RIICHI_RESPONSE_HAND_VALUE_SEARCH_SPACE_VERSION,
            "riichi-response-hand-value-search-space-v2",
        )


class PushTest(unittest.TestCase):
    def setUp(self) -> None:
        # 門前、V = 3,900、共通現物3s。
        self.closed = _input(
            ONE_SHANTEN, drawn="1z", riichi_discards=GENBUTSU_3S, dora_indicators="7s"
        )
        # 白ポン、V = 2,000、共通現物3s。
        self.opened = _input(
            HAKU_OPEN,
            drawn="1z",
            riichi_discards=GENBUTSU_3S,
            dora_indicators=DORA_ONE,
            own_melds=(HAKU_PON,),
        )

    def test_champion_folds_to_genbutsu(self) -> None:
        self.assertEqual(_champion(self.closed), _tile("3s"))
        self.assertNotEqual(_champion(_without_riichi(self.closed)), _tile("3s"))

    def test_closed_push_uses_the_closed_threshold(self) -> None:
        self.assertEqual(
            _choose(_parameters(closed=3900), self.closed),
            _champion(_without_riichi(self.closed)),
        )
        self.assertEqual(_choose(_parameters(closed=3901), self.closed), _tile("3s"))
        self.assertEqual(_choose(_parameters(opened=0), self.closed), _tile("3s"))

    def test_open_push_uses_the_open_threshold(self) -> None:
        self.assertEqual(
            _choose(_parameters(opened=2000), self.opened),
            _champion(_without_riichi(self.opened)),
        )
        self.assertEqual(
            _choose(_parameters(opened=2001), self.opened), _champion(self.opened)
        )
        self.assertEqual(
            _choose(_parameters(closed=0), self.opened), _champion(self.opened)
        )

    def test_push_requires_enough_live_wall(self) -> None:
        late = replace(
            self.closed,
            round=replace(self.closed.round, live_wall_tiles_remaining=23),
        )
        self.assertEqual(_choose(_parameters(closed=0), late), _tile("3s"))

    def test_fold_rule_does_not_act_when_genbutsu_exists(self) -> None:
        self.assertEqual(_choose(_parameters(fold=99999), self.closed), _tile("3s"))


class FoldTest(unittest.TestCase):
    def setUp(self) -> None:
        # 門前、V = 2,000、共通現物なし。
        self.closed = _input(
            ONE_SHANTEN,
            drawn="1z",
            riichi_discards=NO_GENBUTSU,
            dora_indicators=DORA_ZERO,
        )

    def _safest(self, policy_input: PolicyInput) -> set[Tile]:
        decision = _decision(policy_input)
        safest = response._minimum_danger_actions(policy_input, decision.legal_actions)
        self.assertLess(len(safest), len(decision.legal_actions))
        return {action.tile for action in safest}

    def test_fold_selects_a_minimum_danger_discard(self) -> None:
        self.assertIn(
            _choose(_parameters(fold=2000), self.closed), self._safest(self.closed)
        )

    def test_fold_requires_a_low_value(self) -> None:
        self.assertNotIn(_champion(self.closed), self._safest(self.closed))
        self.assertEqual(
            _choose(_parameters(fold=1999), self.closed), _champion(self.closed)
        )

    def test_open_hand_without_yaku_folds_at_any_threshold(self) -> None:
        no_yaku = _input(
            HAKU_OPEN,
            drawn="1z",
            riichi_discards=NO_GENBUTSU,
            dora_indicators=DORA_ONE,
            own_melds=(NO_YAKU_CHI,),
        )
        self.assertIn(_choose(_parameters(fold=0), no_yaku), self._safest(no_yaku))

    def test_push_rules_do_not_act_without_genbutsu(self) -> None:
        self.assertEqual(
            _choose(_parameters(closed=0, opened=0), self.closed),
            _champion(self.closed),
        )


class UnchangedTest(unittest.TestCase):
    ALL = _parameters(closed=0, opened=0, fold=99999)

    def _assert_same(self, policy_input: PolicyInput) -> None:
        self.assertEqual(_choose(self.ALL, policy_input), _champion(policy_input))

    def test_without_riichi(self) -> None:
        self._assert_same(
            _input(
                ONE_SHANTEN, drawn="1z", riichi_discards=None, dora_indicators=DORA_ONE
            )
        )

    def test_tenpai_reachable_hand(self) -> None:
        self._assert_same(
            _input(
                "123m45655p34789s1z",
                drawn="1z",
                riichi_discards=GENBUTSU_3S,
                dora_indicators=DORA_ONE,
            )
        )

    def test_two_shanten_hand(self) -> None:
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
