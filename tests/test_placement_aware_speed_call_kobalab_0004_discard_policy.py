"""Issue #226 `PlacementAwareSpeedCallKobalab0004DiscardPolicy`のfocused test。

同じ固定decision入力で、対象外の分岐・非打牌判断が現Champion
`PlacementAwareSpeedCallPolicy`と一致すること、PUSHの攻撃打牌だけが0004の
牌効率（合成版fallback付き）で選ばれることを固定する。0004側の期待値は
`_improving_tile_types()`（Tile入口の向聴数計算によるoracle）と
`derive_remaining_tile_inventory()`の未見枚数から求め、選択関数自体を呼んで
期待値を作らない。
"""

import ast
import dataclasses
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import lisjong.policies.placement_aware_speed_call_kobalab_0004_discard as candidate_module
import lisjong.policies.targeted_honor_release_terminal_progression as targeted_module
from lisjong.belief import derive_remaining_tile_inventory
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies import (
    PlacementAwareSpeedCallKobalab0004DiscardPolicy,
    PlacementAwareSpeedCallPolicy,
)
from lisjong.policies import kobalab_0004_discard as discard_module
from lisjong.policies.finite_horizon_completion import (
    FiniteHorizonCompletionPolicyError,
)
from lisjong.policies.kobalab_0004_discard import (
    Kobalab0004ReferencePolicyError,
    _choose_minimum_shanten_discard,
    _choose_reference_discard,
    _DiscardStructures,
    _evaluation_order,
    _improving_tile_types,
    _PublicCounts,
)
from lisjong.policies.placement_aware_speed_call import _eligible_discard_actions
from lisjong.policies.targeted_honor_release_terminal_progression import (
    TargetedHonorReleaseActivationStage,
    TargetedHonorReleaseAnalysis,
    TargetedHonorReleaseBranch,
    _classify_branch,
)
from lisjong.policy_contract.action import DiscardAction, PassAction, RiichiAction
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.structural_efficiency import post_discard_concealed_hand
from tests.test_placement_aware_speed_call_policy import (
    ALL_LAST_TOP,
    PASS,
    TANYAO_ONE_SHANTEN,
    _chi,
    _chi_meld,
    _discards,
    _hand,
    _input,
    _tile,
)

# 4567m 222p 678p 45s 22z: 4m / 7mどちらを切っても45sの両面（3s6s、未見8枚）で
# 聴牌し受入同数。paijiaも同値で、source順（萬子はrank降順）で7mを先に評価する。
TENPAI_TIE = "4567m222678p45s22z"
# 234mチー済み。3m 6m 99m 234p 89p 55z（白）は1向聴。タンヤオroute制限後の
# 候補（9m 2p 3p 4p 8p 9p 5z）はすべて打牌後2向聴になる。
FALLBACK_CONCEALED = "3m6m9m9m2p3p4p8p9p5z5z"


def _discard(spec: str, *, tsumogiri: bool = False) -> DiscardAction:
    return DiscardAction(actor=Seat.SEAT_0, tile=_tile(spec), tsumogiri=tsumogiri)


def _decision(
    policy_input: PolicyInput, legal_actions: tuple[object, ...] | None = None
) -> DecisionContext:
    return DecisionContext(
        input=policy_input,
        legal_actions=(
            _discards(policy_input) if legal_actions is None else legal_actions
        ),
    )


def _both(decision: DecisionContext):
    return (
        PlacementAwareSpeedCallPolicy().choose_action_with_analysis(decision),
        PlacementAwareSpeedCallKobalab0004DiscardPolicy().choose_action_with_analysis(
            decision
        ),
    )


def _branch(policy_input: PolicyInput) -> TargetedHonorReleaseBranch:
    eligible = _eligible_discard_actions(policy_input, _discards(policy_input))
    return _classify_branch(policy_input, eligible)[0]


def _oracle_ukeire(policy_input: PolicyInput, tile: Tile) -> tuple[int, int]:
    """(打牌後向聴数, 形の改善牌の未見枚数合計)をTile入口の計算だけで求める。"""
    hand = post_discard_concealed_hand(policy_input.own_hand.concealed_tiles, tile)
    remaining = derive_remaining_tile_inventory(policy_input)
    return calculate_shanten(hand), sum(
        remaining.remaining_tile_counts[
            (list(TileCategory).index(tile_type.category)) * 9 + tile_type.rank - 1
        ]
        for tile_type in _improving_tile_types(hand)
    )


def _full_wall() -> list[Tile]:
    tiles: list[Tile] = []
    for category in (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU):
        for rank in range(1, 10):
            tiles.extend(
                Tile(TileType(category, rank), is_red=rank == 5 and copy == 0)
                for copy in range(4)
            )
    for rank in range(1, 8):
        tiles.extend([Tile(TileType(TileCategory.HONOR, rank))] * 4)
    return tiles


def _with_only_unseen(policy_input: PolicyInput, unseen: str) -> PolicyInput:
    """`unseen`以外の全牌を他家3人の捨て牌として公開した有効な入力。"""
    rest = Counter(_full_wall())
    rest -= Counter(policy_input.own_hand.concealed_tiles)
    rest -= Counter(policy_input.round.dora_indicators)
    rest -= Counter(_hand(unseen))
    for player in policy_input.players:
        rest -= Counter(discard.tile for discard in player.discards)
    pool = list(rest.elements())
    players = list(policy_input.players)
    for offset, seat in enumerate((1, 2, 3)):
        player = players[seat]
        players[seat] = dataclasses.replace(
            player,
            discards=player.discards
            + tuple(
                Discard(
                    tile=tile,
                    tsumogiri=False,
                    order=len(player.discards) + index,
                    called_by=None,
                )
                for index, tile in enumerate(pool[offset::3])
            ),
        )
    return dataclasses.replace(policy_input, players=tuple(players))


class NonPushBranchTest(unittest.TestCase):
    """PUSH以外の打牌はChampionと同じAction・analysisになる。"""

    CONCEALED = "34m678p68p234s55s9s7z"  # 1向聴（他家リーチ下で聴牌維持不可）

    def _assert_same_as_champion(
        self, policy_input: PolicyInput, branch: TargetedHonorReleaseBranch
    ) -> None:
        self.assertIs(_branch(policy_input), branch)
        champion, candidate = _both(_decision(policy_input))
        self.assertEqual(candidate, champion)
        self.assertIsInstance(candidate.analysis, TargetedHonorReleaseAnalysis)

    def test_common_genbutsu_fold(self) -> None:
        self._assert_same_as_champion(
            _input(self.CONCEALED, drawn="7z", riichi_discards="9s3z"),
            TargetedHonorReleaseBranch.FOLD_COMMON_GENBUTSU,
        )

    def test_fold_without_common_genbutsu_falls_back_like_champion(self) -> None:
        self._assert_same_as_champion(
            _input(self.CONCEALED, drawn="7z", riichi_discards="3z4z"),
            TargetedHonorReleaseBranch.OTHER_CURRENT_FALLBACK,
        )

    def test_mechanism_defense(self) -> None:
        self._assert_same_as_champion(
            _input("34m68p68p24s59s1z4z7z3z", drawn="3z", riichi_discards="6z2z"),
            TargetedHonorReleaseBranch.MECHANISM_DEFENSE_FILTERED,
        )

    def test_all_last_top_fold_restriction(self) -> None:
        policy_input = _input(
            self.CONCEALED, drawn="7z", riichi_discards="3z4z", **ALL_LAST_TOP
        )
        eligible = _eligible_discard_actions(policy_input, _discards(policy_input))
        self.assertEqual(len(eligible), 1)
        self._assert_same_as_champion(
            policy_input, TargetedHonorReleaseBranch.OTHER_CURRENT_FALLBACK
        )


class PushSelectionTest(unittest.TestCase):
    def test_closed_normal_turn_uses_0004_instead_of_champion(self) -> None:
        policy_input = _input(TENPAI_TIE, drawn="2z")
        self.assertIs(_branch(policy_input), TargetedHonorReleaseBranch.PUSH)
        for spec in ("4m", "7m"):
            self.assertEqual(_oracle_ukeire(policy_input, _tile(spec)), (0, 8))
        champion, candidate = _both(_decision(policy_input))
        self.assertEqual(champion.action, _discard("4m"))
        self.assertEqual(candidate.action, _discard("7m"))
        # 0004へ切り替えた判断にChampionのR5 analysisを流用しない。
        self.assertIsNone(candidate.analysis)

    def test_push_under_opponent_riichi_when_tenpai_is_kept(self) -> None:
        policy_input = _input(TENPAI_TIE, drawn="2z", riichi_discards="1z9m")
        self.assertIs(_branch(policy_input), TargetedHonorReleaseBranch.PUSH)
        self.assertEqual(
            PlacementAwareSpeedCallKobalab0004DiscardPolicy().choose_action(
                _decision(policy_input)
            ),
            _discard("7m"),
        )

    def test_choose_action_matches_traced_action_and_legal_order(self) -> None:
        policy_input = _input(TENPAI_TIE, drawn="2z")
        actions = _discards(policy_input)
        policy = PlacementAwareSpeedCallKobalab0004DiscardPolicy()
        for legal in (actions, tuple(reversed(actions))):
            decision = _decision(policy_input, legal)
            self.assertEqual(
                policy.choose_action(decision),
                policy.choose_action_with_analysis(decision).action,
            )
            self.assertEqual(policy.choose_action(decision), _discard("7m"))

    def test_red_and_normal_five_and_tsumogiri(self) -> None:
        # 123p 456p 789s 1z 4m 0m 6m + ツモ5m: 5を切ると1z単騎（未見3枚）で最大。
        # 赤5 / 通常5・ツモ切り / 手出しは構造を共有し、paijiaの低い通常5の
        # ツモ切りを先に評価して選ぶ（赤5を残す）。
        policy_input = _input("123p456p789s1z4m0m6m5m", drawn="5m")
        self.assertEqual(_oracle_ukeire(policy_input, _tile("5m")), (0, 3))
        legal = (
            _discard("4m"),
            _discard("0m"),
            _discard("6m"),
            _discard("1z"),
            _discard("1p"),
            _discard("5m", tsumogiri=True),
        )
        self.assertEqual(
            PlacementAwareSpeedCallKobalab0004DiscardPolicy().choose_action(
                _decision(policy_input, legal)
            ),
            _discard("5m", tsumogiri=True),
        )

    def test_push_path_does_not_run_champion_offensive_evaluation(self) -> None:
        # 親の重い攻撃評価を全実行してから0004で再選択する二重計算をしない。
        failing = mock.Mock(side_effect=AssertionError("offensive evaluation ran"))
        with (
            mock.patch.object(targeted_module, "_evaluate_completion_masses", failing),
            mock.patch.object(
                targeted_module,
                "_hand_value_aware_evaluate_and_choose_discard",
                failing,
            ),
            mock.patch.object(
                targeted_module,
                "_parent_offensive_evaluate_and_choose_discard",
                failing,
            ),
        ):
            action = PlacementAwareSpeedCallKobalab0004DiscardPolicy().choose_action(
                _decision(_input(TENPAI_TIE, drawn="2z"))
            )
        self.assertEqual(action, _discard("7m"))
        failing.assert_not_called()

    def test_discard_not_in_hand_fails_closed(self) -> None:
        policy_input = _input(TENPAI_TIE, drawn="2z")
        with self.assertRaises(Kobalab0004ReferencePolicyError):
            PlacementAwareSpeedCallKobalab0004DiscardPolicy().choose_action(
                _decision(policy_input, (*_discards(policy_input), _discard("0m")))
            )


class OpenHandTest(unittest.TestCase):
    # 234mチー済み。789pはtanyao routeに使えないため、route制限後の候補は9p / 1z。
    OPEN = "789p234s55s67p1z"

    def test_open_hand_push_selects_within_the_route_restriction(self) -> None:
        policy_input = _input(self.OPEN, own_melds=(_chi_meld("234m"),), drawn="1z")
        eligible = _eligible_discard_actions(policy_input, _discards(policy_input))
        self.assertEqual({action.tile for action in eligible}, set(_hand("9p1z")))
        self.assertIs(_branch(policy_input), TargetedHonorReleaseBranch.PUSH)
        # 1zを切ると67pの両面で聴牌（5p 4枚 + 手中789pの8pを除く3枚）、9pは1向聴。
        self.assertEqual(_oracle_ukeire(policy_input, _tile("1z")), (0, 7))
        self.assertEqual(_oracle_ukeire(policy_input, _tile("9p"))[0], 1)
        champion, candidate = _both(_decision(policy_input))
        self.assertEqual(candidate.action, _discard("1z"))
        self.assertIsNone(candidate.analysis)
        self.assertIn(champion.action, eligible)

    def test_post_call_discard_is_a_push_target(self) -> None:
        # 副露直後の打牌（drawn_tile is None）も同じ判定で0004を使う。
        policy_input = _input(self.OPEN, own_melds=(_chi_meld("234m"),))
        self.assertIs(_branch(policy_input), TargetedHonorReleaseBranch.PUSH)
        champion, candidate = _both(_decision(policy_input))
        self.assertIsInstance(champion.analysis, TargetedHonorReleaseAnalysis)
        self.assertIs(
            champion.analysis.activation_stage,
            TargetedHonorReleaseActivationStage.NOT_NORMAL_TURN,
        )
        self.assertEqual(candidate.action, _discard("1z"))
        self.assertIsNone(candidate.analysis)

    def test_worsening_fallback_uses_minimum_shanten_then_maximum_ukeire(
        self,
    ) -> None:
        policy_input = _input(
            FALLBACK_CONCEALED, own_melds=(_chi_meld("234m"),), drawn="5z"
        )
        actions = _discards(policy_input)
        eligible = _eligible_discard_actions(policy_input, actions)
        self.assertEqual(
            {action.tile for action in eligible}, set(_hand("9m2p3p4p8p9p5z"))
        )
        self.assertIs(_branch(policy_input), TargetedHonorReleaseBranch.PUSH)
        before = calculate_shanten(policy_input.own_hand.concealed_tiles)
        oracle = {
            action.tile: _oracle_ukeire(policy_input, action.tile)
            for action in eligible
        }
        self.assertEqual(before, 1)
        self.assertEqual({shanten for shanten, _ in oracle.values()}, {2})
        self.assertEqual(oracle[_tile("9p")], (2, 46))
        self.assertLess(
            max(ukeire for tile, (_, ukeire) in oracle.items() if tile != _tile("9p")),
            46,
        )
        # 構造的受入はroute外の牌（9m・白等）も改善牌に数える。
        improving = _improving_tile_types(
            post_discard_concealed_hand(
                policy_input.own_hand.concealed_tiles, _tile("9p")
            )
        )
        self.assertIn(_tile("9m").tile_type, improving)
        self.assertEqual(
            PlacementAwareSpeedCallKobalab0004DiscardPolicy().choose_action(
                _decision(policy_input, actions)
            ),
            _discard("9p"),
        )
        # 参照版の規則（評価順の先頭へfallback）なら白を切る。合成版固有の差。
        structures = _DiscardStructures(
            policy_input.own_hand.concealed_tiles,
            tuple(action.tile for action in eligible),
        )
        self.assertEqual(
            _choose_reference_discard(
                _PublicCounts(policy_input), eligible, structures
            ),
            _discard("5z"),
        )


class MinimumShantenSelectionTest(unittest.TestCase):
    def test_fallback_evaluates_improving_sets_in_one_batch(self) -> None:
        policy_input = _input(
            FALLBACK_CONCEALED, own_melds=(_chi_meld("234m"),), drawn="5z"
        )
        eligible = _eligible_discard_actions(policy_input, _discards(policy_input))
        calls = []
        original = discard_module.evaluate_discards_from_canonical_counts

        def spy(counts, indexes, improving_max_shanten=None):
            calls.append((tuple(indexes), improving_max_shanten))
            return original(counts, indexes, improving_max_shanten)

        with mock.patch.object(
            discard_module, "evaluate_discards_from_canonical_counts", spy
        ):
            structures = _DiscardStructures(
                policy_input.own_hand.concealed_tiles,
                tuple(action.tile for action in eligible),
            )
            chosen = _choose_minimum_shanten_discard(
                _PublicCounts(policy_input), eligible, structures
            )
        self.assertEqual(chosen, _discard("9p"))
        # 初回の一括評価（改善牌は打牌前向聴数以下だけ -> なし）と、
        # 最小向聴数の候補の改善牌をまとめた1回。未評価を0で代用しない。
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], 1)
        self.assertEqual(calls[1][1], None)
        self.assertEqual(len(calls[1][0]), len({a.tile.tile_type for a in eligible}))

    def test_non_worsening_candidates_match_the_reference_rule(self) -> None:
        policy_input = _input(TENPAI_TIE, drawn="2z")
        actions = _discards(policy_input)
        counts = _PublicCounts(policy_input)
        for subset in (actions, actions[:5], tuple(reversed(actions))):
            structures = _DiscardStructures(
                policy_input.own_hand.concealed_tiles, tuple(a.tile for a in subset)
            )
            reference = _choose_reference_discard(counts, subset, structures)
            self.assertEqual(
                _choose_minimum_shanten_discard(counts, subset, structures), reference
            )
            self.assertIn(reference, subset)
            self.assertEqual(
                _evaluation_order(counts, subset),
                _evaluation_order(counts, tuple(reversed(subset))),
            )

    def test_empty_candidates_are_rejected(self) -> None:
        policy_input = _input(TENPAI_TIE, drawn="2z")
        with self.assertRaises(ValueError):
            _choose_minimum_shanten_discard(
                _PublicCounts(policy_input),
                (),
                _DiscardStructures(policy_input.own_hand.concealed_tiles),
            )


class RiichiAndCallTest(unittest.TestCase):
    def test_legal_riichi_is_declared_like_champion(self) -> None:
        policy_input = _input(TENPAI_TIE, drawn="2z")
        riichi = RiichiAction(actor=Seat.SEAT_0)
        decision = _decision(policy_input, (*_discards(policy_input), riichi))
        champion, candidate = _both(decision)
        self.assertEqual(candidate, champion)
        self.assertEqual(candidate.action, riichi)

    def test_declaration_discard_uses_0004_within_legal_candidates(self) -> None:
        # 宣言牌decisionでは聴牌を保つ打牌だけが合法。その中を0004で比較する。
        policy_input = _input(TENPAI_TIE, drawn="2z")
        legal = (_discard("4m"), _discard("7m"))
        champion, candidate = _both(_decision(policy_input, legal))
        self.assertEqual(champion.action, _discard("4m"))
        self.assertEqual(candidate.action, _discard("7m"))

    def test_call_decisions_are_unchanged(self) -> None:
        policy_input = _input(TANYAO_ONE_SHANTEN)
        for legal in ((_chi("2m", "34m"), PASS), (PASS,)):
            champion, candidate = _both(_decision(policy_input, legal))
            self.assertEqual(candidate, champion)
        self.assertIsInstance(
            PlacementAwareSpeedCallKobalab0004DiscardPolicy().choose_action(
                _decision(policy_input, (PassAction(actor=Seat.SEAT_0),))
            ),
            PassAction,
        )

    def test_module_does_not_import_0004_non_discard_rules(self) -> None:
        tree = ast.parse(Path(candidate_module.__file__).read_text(encoding="utf-8"))
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("lisjong.policies.kobalab_0004_reference", imported)
        self.assertNotIn("_lisjong_native", {name.split(".")[0] for name in imported})


class HorizonShortageTest(unittest.TestCase):
    """未見枚数がFiniteHorizonの探索幅未満の有効入力（意図した例外差）。"""

    def test_push_selects_where_champion_raises(self) -> None:
        policy_input = _with_only_unseen(_input(TENPAI_TIE, drawn="2z"), "3s6s")
        self.assertEqual(
            sum(derive_remaining_tile_inventory(policy_input).remaining_tile_counts), 2
        )
        decision = _decision(policy_input)
        with self.assertRaises(FiniteHorizonCompletionPolicyError):
            PlacementAwareSpeedCallPolicy().choose_action(decision)
        self.assertEqual(
            PlacementAwareSpeedCallKobalab0004DiscardPolicy().choose_action(decision),
            _discard("7m"),
        )

    def test_non_push_keeps_the_champion_exception(self) -> None:
        policy_input = _with_only_unseen(
            _input(NonPushBranchTest.CONCEALED, drawn="7z", riichi_discards="9s3z"),
            "1z2z",
        )
        decision = _decision(policy_input)
        for policy in (
            PlacementAwareSpeedCallPolicy(),
            PlacementAwareSpeedCallKobalab0004DiscardPolicy(),
        ):
            with self.subTest(policy=type(policy).__name__):
                with self.assertRaises(FiniteHorizonCompletionPolicyError):
                    policy.choose_action(decision)


if __name__ == "__main__":
    unittest.main()
