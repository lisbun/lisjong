"""#277 ロン合法確率の推定器（待ち推定 × 確実な0）のtest。"""

import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from test_learning_hand_belief_source import A_OWN, make_decision, tiles

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.hand_evaluation._shanten_backend import BACKEND_ENVIRONMENT_VARIABLE
from lisjong.learning.riichi_wait_estimator import (
    LogisticWaitModel,
    estimate_riichi_wait_belief,
)
from lisjong.learning.ron_legal_estimator import (
    certain_ron_illegal_tile_types,
    estimate_riichi_ron_legal_belief,
    in_riichi_scope,
    with_certain_zero,
)
from lisjong.policy_contract import Discard, PlayerPublicState, RiichiState, Seat

MODEL = LogisticWaitModel(weights=(("bias", -1.0), ("genbutsu", -2.0)))


def public(
    rivers, riichi=(RiichiState.NONE, RiichiState.ACCEPTED) + (RiichiState.NONE,) * 2
):
    """`rivers`は席ごとの「牌 order」の列（orderは4席全体で一意）。"""
    players = tuple(
        PlayerPublicState(
            score=25000,
            discards=tuple(
                Discard(
                    tile=tiles(tile)[0], tsumogiri=False, order=order, called_by=None
                )
                for tile, order in river
            ),
            melds=(),
            riichi=state,
        )
        for river, state in zip(rivers, riichi)
    )
    return make_decision(0, A_OWN, players).policy_input


def indices(spec):
    return frozenset(tile_type_index(tile.tile_type) for tile in tiles(spec))


# 対象席1の最後の打牌はorder 5。それより後は席2の3s(6)、席3の0m(7)、観測者の7z(8)。
RIVERS = (
    (("1m", 0), ("7z", 8)),
    (("9p", 1), ("0p", 5)),
    (("2s", 2), ("3s", 6)),
    (("4z", 3), ("0m", 7)),
)


class CertainZeroTest(unittest.TestCase):
    def test_genbutsu_and_discards_after_the_targets_last_discard(self):
        zero = certain_ron_illegal_tile_types(public(RIVERS), Seat.SEAT_1)
        # 現物（赤5pは5pと同じ牌種）と、order 5より後の他家打牌（観測者の7zを含む）
        self.assertEqual(zero, indices("95p") | indices("3s5m7z"))
        # 対象席の最後の打牌より前に他家が切った牌は確定しない
        self.assertFalse(zero & indices("1m2s4z"))

    def test_each_target_uses_its_own_last_discard(self):
        zero = certain_ron_illegal_tile_types(public(RIVERS), Seat.SEAT_3)
        self.assertEqual(zero, indices("4z5m") | indices("7z"))

    def test_target_without_a_discard_has_not_changed_its_dealt_hand(self):
        rivers = (RIVERS[0], (), RIVERS[2], RIVERS[3])
        zero = certain_ron_illegal_tile_types(
            public(rivers, (RiichiState.NONE,) * 4), Seat.SEAT_1
        )
        self.assertEqual(zero, indices("1m7z2s3s4z5m"))

    def test_riichi_state_does_not_change_the_rule(self):
        expected = certain_ron_illegal_tile_types(public(RIVERS), Seat.SEAT_1)
        for state in (RiichiState.NONE, RiichiState.DECLARED):
            states = (RiichiState.NONE, state, RiichiState.NONE, RiichiState.NONE)
            self.assertEqual(
                certain_ron_illegal_tile_types(public(RIVERS, states), Seat.SEAT_1),
                expected,
            )

    def test_only_opponents_are_targets(self):
        with self.assertRaises(ValueError):
            certain_ron_illegal_tile_types(public(RIVERS), Seat.SEAT_0)
        with self.assertRaises(TypeError):
            certain_ron_illegal_tile_types(public(RIVERS), 1)


class RiichiEstimatorTest(unittest.TestCase):
    def test_wait_is_kept_and_only_certain_tiles_become_zero(self):
        policy_input = public(RIVERS)
        belief = estimate_riichi_ron_legal_belief(policy_input, Seat.SEAT_1, MODEL)
        wait = estimate_riichi_wait_belief(policy_input, MODEL).wait_probability_raw
        zero = indices("95p3s5m7z")
        self.assertEqual(belief.wait_probability_raw, wait)
        # 待ち推定は現物も0に固定しない。0にするのはronだけ
        self.assertTrue(all(wait))
        for index in range(34):
            self.assertEqual(
                belief.ron_legal_probability_raw[index],
                0 if index in zero else wait[index],
            )

    def test_declared_riichi_is_in_scope(self):
        states = (RiichiState.NONE, RiichiState.DECLARED) + (RiichiState.NONE,) * 2
        belief = estimate_riichi_ron_legal_belief(
            public(RIVERS, states), Seat.SEAT_1, MODEL
        )
        self.assertEqual(
            belief.ron_legal_probability_raw[tile_type_index(tiles("9p")[0].tile_type)],
            0,
        )

    def test_outside_the_wait_estimators_scope_is_unprovided_not_zero(self):
        none, on = RiichiState.NONE, RiichiState.ACCEPTED
        for states, seat in (
            ((none, on, none, none), Seat.SEAT_2),  # 非リーチ者
            ((none, on, on, none), Seat.SEAT_1),  # 2人リーチ
            ((on, on, none, none), Seat.SEAT_1),  # 観測者がリーチ
            ((none, none, none, none), Seat.SEAT_1),
        ):
            policy_input = public(RIVERS, states)
            self.assertFalse(in_riichi_scope(policy_input, seat))
            self.assertIsNone(
                estimate_riichi_ron_legal_belief(policy_input, seat, MODEL)
            )
        with self.assertRaises(ValueError):
            in_riichi_scope(public(RIVERS), Seat.SEAT_0)
        with self.assertRaises(TypeError):
            estimate_riichi_ron_legal_belief(public(RIVERS), Seat.SEAT_1, object())

    def test_an_unprovided_wait_is_not_turned_into_ron(self):
        policy_input = public(RIVERS)
        belief = estimate_riichi_wait_belief(policy_input, MODEL)
        with self.assertRaises(ValueError):
            with_certain_zero(
                replace(belief, wait_probability_raw=None), policy_input, Seat.SEAT_1
            )

    def test_inference_without_truth_source_or_native_imports(self):
        code = """
import importlib.abc,sys
sys.path.insert(0,'tests')
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.startswith(("lisjong.learning.ron_legal_accuracy", "lisjong.learning.ron_legal_source", "lisjong.learning.riichi_wait_evaluation", "lisjong.learning.riichi_deal_in_source", "lisjong.learning.hand_belief_source", "lisjong.belief.ron_legal_ground_truth", "lisjong.belief.exact_wait_ground_truth", "_lisjong_native")):
            raise AssertionError("privileged import: " + name)
from lisjong.learning.riichi_wait_estimator import LogisticWaitModel
from lisjong.learning.ron_legal_estimator import estimate_riichi_ron_legal_belief
from lisjong.policy_contract import Discard,PlayerPublicState,PolicyInput,OwnHandState,RiichiState,RoundState,Seat,Tile,TileCategory,TileType,Wind
t=lambda r,c=TileCategory.MANZU: Tile(TileType(c,r),is_red=False)
p=lambda riichi,*ds: PlayerPublicState(score=25000,discards=tuple(Discard(tile=t(r),tsumogiri=False,order=o,called_by=None) for r,o in ds),melds=(),riichi=riichi)
sys.meta_path.insert(0, Block())
pi=PolicyInput(self_seat=Seat.SEAT_0,round=RoundState(round_wind=Wind.EAST,hand_number=1,dealer_seat=Seat.SEAT_0,honba=0,riichi_sticks=0,dora_indicators=(),live_wall_tiles_remaining=40),players=(p(RiichiState.NONE,(1,0)),p(RiichiState.ACCEPTED,(9,1)),p(RiichiState.NONE,(2,2)),p(RiichiState.NONE)),own_hand=OwnHandState(concealed_tiles=tuple(t(r,TileCategory.PINZU) for r in (1,1,2,2,3,3,4,4,5,5,6,6,7,7)),drawn_tile=None))
b=estimate_riichi_ron_legal_belief(pi,Seat.SEAT_1,LogisticWaitModel(weights=(("bias",-1.0),)))
assert b.ron_legal_probability_raw[8]==0 and b.ron_legal_probability_raw[1]==0
assert b.ron_legal_probability_raw[0]==b.wait_probability_raw[0]>0
assert estimate_riichi_ron_legal_belief(pi,Seat.SEAT_2,LogisticWaitModel(weights=())) is None
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            # The Rust shanten backend legitimately loads the native module;
            # the estimator itself must work on the default Python backend.
            env={
                k: v for k, v in os.environ.items() if k != BACKEND_ENVIRONMENT_VARIABLE
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
