import inspect
import unittest

from lisjong.action_vocabulary import (
    ACTION_VOCABULARY_BLOCKS,
    ACTION_VOCABULARY_SIZE,
    ACTION_VOCABULARY_VERSION,
    ActionEncodingError,
    ActionIndexCollisionError,
    ActionIndexError,
    ActionVocabularyError,
    IllegalActionIndexError,
    UnsupportedActionVocabularyVersionError,
    build_legal_action_mask,
    decode_action,
    encode_action,
    encode_legal_actions,
    resolve_legal_action,
)
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

UNSUPPORTED_VERSION = "lisjong-action-vocabulary-0"


def _manzu(rank: int, *, red: bool = False) -> Tile:
    return Tile(TileType(TileCategory.MANZU, rank), is_red=red)


def _pinzu(rank: int, *, red: bool = False) -> Tile:
    return Tile(TileType(TileCategory.PINZU, rank), is_red=red)


def _souzu(rank: int, *, red: bool = False) -> Tile:
    return Tile(TileType(TileCategory.SOUZU, rank), is_red=red)


def _honor(rank: int) -> Tile:
    return Tile(TileType(TileCategory.HONOR, rank))


def _make_input(
    self_seat: Seat = Seat.SEAT_0,
    *,
    score: int = 25000,
    dora_indicator: Tile | None = None,
    concealed_tiles: tuple[Tile, ...] = (),
    drawn_tile: Tile | None = None,
) -> PolicyInput:
    player = PlayerPublicState(
        score=score, discards=(), melds=(), riichi=RiichiState.NONE
    )
    return PolicyInput(
        self_seat=self_seat,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(dora_indicator or _manzu(3),),
            live_wall_tiles_remaining=70,
        ),
        players=(player, player, player, player),
        own_hand=OwnHandState(
            concealed_tiles=concealed_tiles or (_manzu(4), _manzu(5), _manzu(6)),
            drawn_tile=drawn_tile,
        ),
    )


def _decision(
    *actions: InternalAction, input: PolicyInput | None = None
) -> DecisionContext:
    return DecisionContext(input=input or _make_input(), legal_actions=actions)


def _sample_actions(actor: Seat = Seat.SEAT_0) -> tuple[InternalAction, ...]:
    """11 variantすべてを1件ずつ含む、赤牌構成を含むsample。"""
    kamicha = Seat((int(actor) + 3) % 4)
    toimen = Seat((int(actor) + 2) % 4)
    shimocha = Seat((int(actor) + 1) % 4)
    return (
        DiscardAction(actor=actor, tile=_pinzu(5, red=True), tsumogiri=True),
        RiichiAction(actor=actor),
        ChiAction(
            actor=actor,
            target=kamicha,
            called_tile=_manzu(5, red=True),
            consumed_tiles=(_manzu(3), _manzu(4)),
        ),
        PonAction(
            actor=actor,
            target=toimen,
            called_tile=_pinzu(5),
            consumed_tiles=(_pinzu(5), _pinzu(5, red=True)),
        ),
        DaiminkanAction(
            actor=actor,
            target=shimocha,
            called_tile=_souzu(5),
            consumed_tiles=(_souzu(5), _souzu(5), _souzu(5, red=True)),
        ),
        AnkanAction(
            actor=actor,
            tiles=(_manzu(5), _manzu(5), _manzu(5), _manzu(5, red=True)),
        ),
        KakanAction(
            actor=actor,
            added_tile=_pinzu(5, red=True),
            from_seat=toimen,
            called_tile=_pinzu(5),
        ),
        RonAction(actor=actor, target=shimocha, winning_tile=_pinzu(3)),
        TsumoAction(actor=actor, winning_tile=_souzu(7)),
        PassAction(actor=actor),
        KyuushuKyuuhaiAction(actor=actor),
    )


class VocabularyIdentityTest(unittest.TestCase):
    def test_version_identity_and_fixed_size(self) -> None:
        self.assertEqual(ACTION_VOCABULARY_VERSION, "lisjong-action-vocabulary-1")
        self.assertEqual(ACTION_VOCABULARY_SIZE, 802)

    def test_documented_index_layout(self) -> None:
        self.assertEqual(
            dict(ACTION_VOCABULARY_BLOCKS),
            {
                DiscardAction: range(0, 74),
                RiichiAction: range(74, 75),
                ChiAction: range(75, 165),
                PonAction: range(165, 312),
                DaiminkanAction: range(312, 477),
                AnkanAction: range(477, 523),
                KakanAction: range(523, 652),
                RonAction: range(652, 763),
                TsumoAction: range(763, 800),
                PassAction: range(800, 801),
                KyuushuKyuuhaiAction: range(801, 802),
            },
        )

    def test_blocks_partition_the_whole_index_space(self) -> None:
        covered: list[int] = []
        for block in ACTION_VOCABULARY_BLOCKS.values():
            covered.extend(block)

        self.assertEqual(len(covered), ACTION_VOCABULARY_SIZE)
        self.assertEqual(sorted(covered), list(range(ACTION_VOCABULARY_SIZE)))

    def test_index_space_is_a_lossless_bijection(self) -> None:
        """定義から到達可能なindexを機械的に検査する。

        range重複、hole、意図しないaliasはいずれもここで検出される。全indexが
        decode可能で、decodeされたActionが相互に異なり、encodeで同じindexへ
        戻り、変換先が自分のvariant blockへ収まることを確認する。
        """
        for actor in Seat:
            with self.subTest(actor=actor):
                decoded: dict[InternalAction, int] = {}
                for index in range(ACTION_VOCABULARY_SIZE):
                    action = decode_action(index, actor)
                    self.assertNotIn(action, decoded)
                    decoded[action] = index
                    self.assertEqual(encode_action(action), index)
                    self.assertIn(index, ACTION_VOCABULARY_BLOCKS[type(action)])
                    self.assertEqual(action.actor, actor)

                self.assertEqual(len(decoded), ACTION_VOCABULARY_SIZE)

    def test_actor_is_restored_from_context_instead_of_the_index(self) -> None:
        """actorはvocabularyへ含めず、decode時のcontextから復元する。"""
        indices = {
            encode_action(DiscardAction(actor=actor, tile=_manzu(5), tsumogiri=False))
            for actor in Seat
        }

        self.assertEqual(len(indices), 1)

    def test_target_and_from_seat_are_actor_relative(self) -> None:
        """target / from_seatはactor相対で表現し、absolute seatを複製しない。"""
        seat_0_ron = RonAction(
            actor=Seat.SEAT_0, target=Seat.SEAT_1, winning_tile=_pinzu(3)
        )
        seat_2_ron = RonAction(
            actor=Seat.SEAT_2, target=Seat.SEAT_3, winning_tile=_pinzu(3)
        )
        other_target_ron = RonAction(
            actor=Seat.SEAT_0, target=Seat.SEAT_2, winning_tile=_pinzu(3)
        )

        self.assertEqual(encode_action(seat_0_ron), encode_action(seat_2_ron))
        self.assertNotEqual(encode_action(seat_0_ron), encode_action(other_target_ron))

    def test_chi_target_is_derived_from_the_actor(self) -> None:
        for actor in Seat:
            with self.subTest(actor=actor):
                chi = ChiAction(
                    actor=actor,
                    target=Seat((int(actor) + 3) % 4),
                    called_tile=_manzu(3),
                    consumed_tiles=(_manzu(4), _manzu(5)),
                )
                self.assertEqual(decode_action(encode_action(chi), actor), chi)


class EncodeDecodeVariantTest(unittest.TestCase):
    def test_all_eleven_variants_round_trip(self) -> None:
        for action in _sample_actions():
            with self.subTest(variant=type(action).__name__):
                index = encode_action(action)
                self.assertEqual(decode_action(index, action.actor), action)

    def test_discard_distinguishes_red_five(self) -> None:
        normal = DiscardAction(actor=Seat.SEAT_0, tile=_pinzu(5), tsumogiri=False)
        red = DiscardAction(
            actor=Seat.SEAT_0, tile=_pinzu(5, red=True), tsumogiri=False
        )

        self.assertNotEqual(encode_action(normal), encode_action(red))
        self.assertEqual(decode_action(encode_action(red), Seat.SEAT_0), red)

    def test_discard_distinguishes_tsumogiri_for_the_same_tile(self) -> None:
        tedashi = DiscardAction(actor=Seat.SEAT_0, tile=_manzu(5), tsumogiri=False)
        tsumogiri = DiscardAction(actor=Seat.SEAT_0, tile=_manzu(5), tsumogiri=True)

        self.assertNotEqual(encode_action(tedashi), encode_action(tsumogiri))
        self.assertEqual(decode_action(encode_action(tedashi), Seat.SEAT_0), tedashi)
        self.assertEqual(
            decode_action(encode_action(tsumogiri), Seat.SEAT_0), tsumogiri
        )

    def test_chi_keeps_distinct_consumed_pairs_for_the_same_called_tile(self) -> None:
        """同一called tileに複数consumed pairがあるChiを衝突なく表現する。"""
        called = _manzu(5)
        pairs = (
            (_manzu(3), _manzu(4)),
            (_manzu(4), _manzu(6)),
            (_manzu(6), _manzu(7)),
        )
        chi_actions = tuple(
            ChiAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_3,
                called_tile=called,
                consumed_tiles=pair,
            )
            for pair in pairs
        )
        indices = [encode_action(action) for action in chi_actions]

        self.assertEqual(len(set(indices)), len(chi_actions))
        for action, index in zip(chi_actions, indices, strict=True):
            self.assertEqual(decode_action(index, Seat.SEAT_0), action)

    def test_chi_keeps_red_composition(self) -> None:
        normal = ChiAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_3,
            called_tile=_souzu(4),
            consumed_tiles=(_souzu(3), _souzu(5)),
        )
        red_consumed = ChiAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_3,
            called_tile=_souzu(4),
            consumed_tiles=(_souzu(3), _souzu(5, red=True)),
        )
        red_called = ChiAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_3,
            called_tile=_souzu(5, red=True),
            consumed_tiles=(_souzu(3), _souzu(4)),
        )
        indices = [
            encode_action(action) for action in (normal, red_consumed, red_called)
        ]

        self.assertEqual(len(set(indices)), 3)
        for action, index in zip((normal, red_consumed, red_called), indices):
            self.assertEqual(decode_action(index, Seat.SEAT_0), action)

    def test_chi_consumed_order_does_not_change_the_index(self) -> None:
        forward = ChiAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_3,
            called_tile=_manzu(3),
            consumed_tiles=(_manzu(4), _manzu(5, red=True)),
        )
        reversed_pair = ChiAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_3,
            called_tile=_manzu(3),
            consumed_tiles=(_manzu(5, red=True), _manzu(4)),
        )

        self.assertEqual(encode_action(forward), encode_action(reversed_pair))

    def test_pon_keeps_red_composition_and_relative_target(self) -> None:
        base = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=_pinzu(5),
            consumed_tiles=(_pinzu(5), _pinzu(5)),
        )
        red_consumed = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=_pinzu(5),
            consumed_tiles=(_pinzu(5), _pinzu(5, red=True)),
        )
        red_called = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_1,
            called_tile=_pinzu(5, red=True),
            consumed_tiles=(_pinzu(5), _pinzu(5)),
        )
        other_target = PonAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_2,
            called_tile=_pinzu(5),
            consumed_tiles=(_pinzu(5), _pinzu(5)),
        )
        actions = (base, red_consumed, red_called, other_target)
        indices = [encode_action(action) for action in actions]

        self.assertEqual(len(set(indices)), len(actions))
        for action, index in zip(actions, indices, strict=True):
            self.assertEqual(decode_action(index, Seat.SEAT_0), action)

    def test_daiminkan_keeps_red_composition(self) -> None:
        normal = DaiminkanAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_2,
            called_tile=_manzu(5),
            consumed_tiles=(_manzu(5), _manzu(5), _manzu(5)),
        )
        with_red = DaiminkanAction(
            actor=Seat.SEAT_0,
            target=Seat.SEAT_2,
            called_tile=_manzu(5),
            consumed_tiles=(_manzu(5), _manzu(5), _manzu(5, red=True)),
        )

        self.assertNotEqual(encode_action(normal), encode_action(with_red))
        self.assertEqual(decode_action(encode_action(with_red), Seat.SEAT_0), with_red)

    def test_ankan_keeps_red_composition(self) -> None:
        normal = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(_souzu(5), _souzu(5), _souzu(5), _souzu(5)),
        )
        with_red = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(_souzu(5), _souzu(5), _souzu(5), _souzu(5, red=True)),
        )
        honors = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(_honor(7), _honor(7), _honor(7), _honor(7)),
        )
        actions = (normal, with_red, honors)
        indices = [encode_action(action) for action in actions]

        self.assertEqual(len(set(indices)), len(actions))
        for action, index in zip(actions, indices, strict=True):
            self.assertEqual(decode_action(index, Seat.SEAT_0), action)

    def test_ankan_tile_order_does_not_change_the_index(self) -> None:
        canonical = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(_souzu(5), _souzu(5), _souzu(5), _souzu(5, red=True)),
        )
        shuffled = AnkanAction(
            actor=Seat.SEAT_0,
            tiles=(_souzu(5, red=True), _souzu(5), _souzu(5), _souzu(5)),
        )

        self.assertEqual(encode_action(canonical), encode_action(shuffled))

    def test_kakan_keeps_added_tile_from_seat_and_called_tile(self) -> None:
        base = KakanAction(
            actor=Seat.SEAT_0,
            added_tile=_pinzu(5),
            from_seat=Seat.SEAT_1,
            called_tile=_pinzu(5),
        )
        red_added = KakanAction(
            actor=Seat.SEAT_0,
            added_tile=_pinzu(5, red=True),
            from_seat=Seat.SEAT_1,
            called_tile=_pinzu(5),
        )
        red_called = KakanAction(
            actor=Seat.SEAT_0,
            added_tile=_pinzu(5),
            from_seat=Seat.SEAT_1,
            called_tile=_pinzu(5, red=True),
        )
        other_from_seat = KakanAction(
            actor=Seat.SEAT_0,
            added_tile=_pinzu(5),
            from_seat=Seat.SEAT_3,
            called_tile=_pinzu(5),
        )
        actions = (base, red_added, red_called, other_from_seat)
        indices = [encode_action(action) for action in actions]

        self.assertEqual(len(set(indices)), len(actions))
        for action, index in zip(actions, indices, strict=True):
            self.assertEqual(decode_action(index, Seat.SEAT_0), action)

    def test_ron_and_tsumo_keep_winning_tile(self) -> None:
        ron = RonAction(
            actor=Seat.SEAT_0, target=Seat.SEAT_2, winning_tile=_manzu(5, red=True)
        )
        other_ron = RonAction(
            actor=Seat.SEAT_0, target=Seat.SEAT_2, winning_tile=_manzu(5)
        )
        tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=_manzu(5, red=True))
        other_tsumo = TsumoAction(actor=Seat.SEAT_0, winning_tile=_honor(1))
        actions = (ron, other_ron, tsumo, other_tsumo)
        indices = [encode_action(action) for action in actions]

        self.assertEqual(len(set(indices)), len(actions))
        for action, index in zip(actions, indices, strict=True):
            self.assertEqual(decode_action(index, Seat.SEAT_0), action)

    def test_riichi_pass_and_kyuushu_kyuuhai_are_single_indices(self) -> None:
        actions = (
            RiichiAction(actor=Seat.SEAT_0),
            PassAction(actor=Seat.SEAT_0),
            KyuushuKyuuhaiAction(actor=Seat.SEAT_0),
        )
        indices = [encode_action(action) for action in actions]

        self.assertEqual(len(set(indices)), len(actions))
        for action, index in zip(actions, indices, strict=True):
            self.assertEqual(decode_action(index, Seat.SEAT_0), action)
            self.assertEqual(len(ACTION_VOCABULARY_BLOCKS[type(action)]), 1)

    def test_encode_rejects_non_internal_action(self) -> None:
        for value in (None, "discard", 0, object()):
            with self.subTest(value=value):
                with self.assertRaises(ActionEncodingError):
                    encode_action(value)

    def test_decode_requires_a_seat_actor(self) -> None:
        with self.assertRaises(TypeError):
            decode_action(0, 0)

    def test_decode_rejects_out_of_range_index(self) -> None:
        for index in (-1, ACTION_VOCABULARY_SIZE, 10**9):
            with self.subTest(index=index):
                with self.assertRaises(ActionIndexError):
                    decode_action(index, Seat.SEAT_0)

    def test_decode_rejects_non_int_index(self) -> None:
        for index in (True, 1.0, "1", None):
            with self.subTest(index=index):
                with self.assertRaises(TypeError):
                    decode_action(index, Seat.SEAT_0)

    def test_codec_does_not_require_a_decision_context(self) -> None:
        """codecはhidden informationも外部engine objectも必要としない。"""
        parameters = inspect.signature(encode_action).parameters
        self.assertEqual(list(parameters), ["action", "version"])
        self.assertEqual(
            list(inspect.signature(decode_action).parameters),
            ["index", "actor", "version"],
        )


class LegalMaskTest(unittest.TestCase):
    def test_mask_is_fixed_size_and_matches_legal_actions_exactly(self) -> None:
        actions = _sample_actions()
        decision = _decision(*actions)
        mask = build_legal_action_mask(decision)
        expected = {encode_action(action) for action in actions}

        self.assertEqual(len(mask), ACTION_VOCABULARY_SIZE)
        self.assertEqual({index for index, legal in enumerate(mask) if legal}, expected)
        self.assertEqual(sum(mask), len(actions))
        self.assertEqual(set(encode_legal_actions(decision)), expected)

    def test_mask_true_indices_resolve_back_to_canonical_legal_actions(self) -> None:
        actions = _sample_actions()
        decision = _decision(*actions)
        mask = build_legal_action_mask(decision)

        resolved = [
            resolve_legal_action(index, decision)
            for index, legal in enumerate(mask)
            if legal
        ]

        self.assertEqual(len(resolved), len(actions))
        for action in resolved:
            self.assertIn(action, decision.legal_actions)

    def test_resolve_returns_the_canonical_legal_action_object(self) -> None:
        """decode後は`legal_actions`側のcanonical objectを返す。"""
        canonical = DiscardAction(
            actor=Seat.SEAT_0, tile=_pinzu(5, red=True), tsumogiri=False
        )
        decision = _decision(canonical, PassAction(actor=Seat.SEAT_0))
        equal_but_distinct = DiscardAction(
            actor=Seat.SEAT_0, tile=_pinzu(5, red=True), tsumogiri=False
        )
        index = encode_action(equal_but_distinct)

        resolved = resolve_legal_action(index, decision)

        self.assertIs(resolved, decision.legal_actions[0])
        self.assertIsNot(resolved, equal_but_distinct)
        self.assertEqual(resolved, equal_but_distinct)

    def test_round_trip_returns_the_canonical_legal_action_for_every_variant(
        self,
    ) -> None:
        for actor in Seat:
            actions = _sample_actions(actor)
            decision = _decision(*actions, input=_make_input(actor))
            for action in actions:
                with self.subTest(actor=actor, variant=type(action).__name__):
                    resolved = resolve_legal_action(encode_action(action), decision)
                    self.assertIs(resolved, action)

    def test_encoded_legal_actions_map_to_canonical_objects(self) -> None:
        actions = _sample_actions()
        decision = _decision(*actions)

        encoded = encode_legal_actions(decision)

        self.assertEqual(len(encoded), len(actions))
        for index, action in encoded.items():
            self.assertIs(action, decision.legal_actions[actions.index(action)])
            self.assertEqual(encode_action(action), index)
        self.assertEqual(list(encoded), sorted(encoded))

    def test_results_do_not_depend_on_legal_action_order(self) -> None:
        actions = _sample_actions()
        decision = _decision(*actions)
        permuted = _decision(*reversed(actions))

        self.assertEqual(
            build_legal_action_mask(decision), build_legal_action_mask(permuted)
        )
        self.assertEqual(
            list(encode_legal_actions(decision).items()),
            list(encode_legal_actions(permuted).items()),
        )
        for action in actions:
            index = encode_action(action)
            self.assertEqual(
                resolve_legal_action(index, decision),
                resolve_legal_action(index, permuted),
            )

    def test_results_do_not_depend_on_policy_input_content(self) -> None:
        """maskとindexはlegal actionsとself_seatだけから決まる。"""
        actions = _sample_actions()
        decision = _decision(*actions)
        other_input = _decision(
            *actions,
            input=_make_input(
                score=12300,
                dora_indicator=_honor(4),
                concealed_tiles=(_souzu(1), _souzu(2), _souzu(3)),
                drawn_tile=_souzu(3),
            ),
        )

        self.assertEqual(
            build_legal_action_mask(decision), build_legal_action_mask(other_input)
        )
        self.assertEqual(
            set(encode_legal_actions(decision)), set(encode_legal_actions(other_input))
        )

    def test_mask_is_identical_for_the_same_relative_decision_on_another_seat(
        self,
    ) -> None:
        seat_0 = _decision(*_sample_actions(Seat.SEAT_0))
        seat_2 = _decision(
            *_sample_actions(Seat.SEAT_2), input=_make_input(Seat.SEAT_2)
        )

        self.assertEqual(
            build_legal_action_mask(seat_0), build_legal_action_mask(seat_2)
        )

    def test_resolve_rejects_an_index_that_is_illegal_in_this_decision(self) -> None:
        legal = DiscardAction(actor=Seat.SEAT_0, tile=_manzu(4), tsumogiri=False)
        decision = _decision(legal)
        illegal_index = encode_action(
            DiscardAction(actor=Seat.SEAT_0, tile=_manzu(4), tsumogiri=True)
        )
        mask = build_legal_action_mask(decision)

        self.assertFalse(mask[illegal_index])
        with self.assertRaises(IllegalActionIndexError):
            resolve_legal_action(illegal_index, decision)

    def test_resolve_rejects_out_of_range_and_non_int_index(self) -> None:
        decision = _decision(PassAction(actor=Seat.SEAT_0))

        for index in (-1, ACTION_VOCABULARY_SIZE):
            with self.subTest(index=index):
                with self.assertRaises(ActionIndexError):
                    resolve_legal_action(index, decision)
        for index in (True, 0.0, "0"):
            with self.subTest(index=index):
                with self.assertRaises(TypeError):
                    resolve_legal_action(index, decision)

    def test_mask_and_resolve_require_a_decision_context(self) -> None:
        for value in (None, (), _make_input()):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(TypeError):
                    build_legal_action_mask(value)
                with self.assertRaises(TypeError):
                    resolve_legal_action(0, value)
                with self.assertRaises(TypeError):
                    encode_legal_actions(value)

    def test_colliding_legal_actions_fail_closed(self) -> None:
        """同一indexへ衝突するlegal actionsは、どちらも採用せず拒否する。

        vocabularyがinjectiveであり、`DecisionContext`がsemantic重複を禁止する
        ため、この状態は正常な構築経路では発生しない。生成後にlegal actionsを
        差し替えたcontextで、defensive guardがfail closedすることを固定する。
        """
        decision = _decision(PassAction(actor=Seat.SEAT_0))
        object.__setattr__(
            decision,
            "legal_actions",
            (PassAction(actor=Seat.SEAT_0), PassAction(actor=Seat.SEAT_0)),
        )

        with self.assertRaises(ActionIndexCollisionError):
            encode_legal_actions(decision)
        with self.assertRaises(ActionIndexCollisionError):
            build_legal_action_mask(decision)
        with self.assertRaises(ActionIndexCollisionError):
            resolve_legal_action(encode_action(PassAction(actor=Seat.SEAT_0)), decision)


class VocabularyVersionTest(unittest.TestCase):
    def test_every_public_api_rejects_an_unsupported_version(self) -> None:
        action = PassAction(actor=Seat.SEAT_0)
        decision = _decision(action)
        index = encode_action(action)
        calls = (
            lambda version: encode_action(action, version=version),
            lambda version: decode_action(index, Seat.SEAT_0, version=version),
            lambda version: encode_legal_actions(decision, version=version),
            lambda version: build_legal_action_mask(decision, version=version),
            lambda version: resolve_legal_action(index, decision, version=version),
        )

        for call in calls:
            for version in (UNSUPPORTED_VERSION, "", None, 1):
                with self.subTest(call=call, version=version):
                    with self.assertRaises(UnsupportedActionVocabularyVersionError):
                        call(version)

    def test_version_is_checked_before_other_validation(self) -> None:
        decision = _decision(PassAction(actor=Seat.SEAT_0))

        with self.assertRaises(UnsupportedActionVocabularyVersionError):
            resolve_legal_action(
                ACTION_VOCABULARY_SIZE, decision, version=UNSUPPORTED_VERSION
            )

    def test_all_errors_share_one_fail_closed_base(self) -> None:
        for error in (
            ActionEncodingError,
            ActionIndexCollisionError,
            ActionIndexError,
            IllegalActionIndexError,
            UnsupportedActionVocabularyVersionError,
        ):
            with self.subTest(error=error.__name__):
                self.assertTrue(issubclass(error, ActionVocabularyError))


class _MaskDrivenPolicy:
    """maskで合法indexを絞り、resolveしたcanonical Actionを返すPolicy。"""

    def __init__(self, selected_index: int | None = None) -> None:
        self.selected_index = selected_index

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        mask = build_legal_action_mask(decision)
        index = self.selected_index
        if index is None:
            index = mask.index(True)
        if not mask[index]:
            raise AssertionError("policy must not select a masked index")
        return resolve_legal_action(index, decision)


class ExecutePolicyNonInterferenceTest(unittest.TestCase):
    def test_execute_policy_signature_is_unchanged(self) -> None:
        self.assertEqual(
            list(inspect.signature(execute_policy).parameters), ["policy", "decision"]
        )

    def test_vocabulary_driven_policy_passes_existing_validation(self) -> None:
        actions = _sample_actions()
        decision = _decision(*actions)
        target = actions[3]
        policy = _MaskDrivenPolicy(encode_action(target))

        selected = execute_policy(policy, decision)

        self.assertIs(selected, target)

    def test_execute_policy_still_returns_the_canonical_legal_action(self) -> None:
        actions = _sample_actions()
        decision = _decision(*actions)

        selected = execute_policy(_MaskDrivenPolicy(), decision)

        self.assertIn(selected, decision.legal_actions)
        self.assertIs(selected, decision.legal_actions[actions.index(selected)])

    def test_decision_context_is_not_modified_by_the_codec(self) -> None:
        actions = _sample_actions()
        decision = _decision(*actions)

        build_legal_action_mask(decision)
        encode_legal_actions(decision)
        resolve_legal_action(encode_action(actions[0]), decision)

        self.assertEqual(decision.legal_actions, actions)
        self.assertEqual(decision, _decision(*actions))


if __name__ == "__main__":
    unittest.main()
