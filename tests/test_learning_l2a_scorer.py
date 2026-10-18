"""L2aの打牌M1・反応R1・dataset・artifact・合成runtime（Issue #316）のtest。"""

import ast
import importlib.util
import tempfile
import unittest
from array import array
from dataclasses import replace
from pathlib import Path

import candidate_fixtures as cf
import learning_fixtures as fixtures

from lisjong.hand_evaluation.shanten import calculate_restricted_standard_shanten
from lisjong.learning import l2a_encoding as encoding_module
from lisjong.learning._canonical import canonical_json_text, seal
from lisjong.learning._typed_values import policy_input_to_value_v2
from lisjong.learning.candidate_encoding import (
    build_scorer_candidates,
    encode_candidates,
    encoding_block,
)
from lisjong.learning.errors import (
    CandidateFeatureError,
    DatasetError,
    LearnedPolicyError,
    ModelArtifactError,
)
from lisjong.learning.features import FEATURE_DIMENSION
from lisjong.learning.l2a_artifact import load_l2a_artifact, write_l2a_artifact
from lisjong.learning.l2a_dataset import publish_l2a_dataset, read_l2a_dataset
from lisjong.learning.l2a_decision import (
    L2aDecisionKind,
    classify_l2a_decision,
    guard_action,
)
from lisjong.learning.l2a_encoding import (
    ENCODING_M0,
    ENCODING_M1,
    ENCODING_R1,
    M1_BLOCK_OFFSETS,
    R1_BLOCK_OFFSETS,
    l2a_encoding,
)
from lisjong.learning.l2a_model import L2aScorerConfig
from lisjong.learning.l2a_policy import L2aPolicy, select_best
from lisjong.learning.source_record import (
    POLICY_SOURCE_RECORD_SCHEMA_V2,
    open_source_record,
    read_source_record,
)
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DecisionContext,
    PassAction,
    PonAction,
    RiichiState,
    Seat,
)
from lisjong.policy_contract.meld import MeldKind, PublicMeld

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_M0 = encoding_module.CANDIDATE_ENCODING_DIMENSION
_L2A_MODULES = (
    "l2a_artifact",
    "l2a_dataset",
    "l2a_encoding",
    "l2a_model",
    "l2a_policy",
    "l2a_training",
)


def _tile(spec):
    return cf.hand(spec)[0]


def _chi(called, consumed, *, seat=Seat.SEAT_0):
    return ChiAction(
        actor=seat,
        target=Seat((int(seat) + 3) % 4),
        called_tile=_tile(called),
        consumed_tiles=tuple(_tile(item) for item in consumed),
    )


def _pon(called, *, seat=Seat.SEAT_0):
    tile = _tile(called)
    return PonAction(
        actor=seat,
        target=Seat((int(seat) + 2) % 4),
        called_tile=tile,
        consumed_tiles=(tile, tile),
    )


def _call(spec, *actions):
    value = cf.policy_input(spec, drawn=False)
    return DecisionContext(
        input=value, legal_actions=(PassAction(actor=value.self_seat), *actions)
    )


def _chi_response():
    return _call(
        "23m456p789s11223z", _chi("1m", ("2m", "3m")), _chi("4m", ("2m", "3m"))
    )


def _single_candidate_discard():
    value = cf.policy_input("1111m456p789s122z3z")
    return DecisionContext(
        input=value,
        legal_actions=(
            *cf.discard_actions(value, tsumogiri_only=True),
            AnkanAction(actor=value.self_seat, tiles=(_tile("1m"),) * 4),
        ),
    )


def _riichi_opponent_decision():
    """下家がリーチし、宣言牌の後に対面が3zを切った打牌判断。"""
    value = cf.policy_input(cf.NEAR_HAND)
    players = list(value.players)
    players[1] = replace(
        players[1],
        discards=(fixtures.discard(_tile("9p"), 0), fixtures.discard(_tile("1z"), 1)),
        riichi=RiichiState.ACCEPTED,
        riichi_declaration_discard_order=1,
    )
    players[2] = replace(players[2], discards=(fixtures.discard(_tile("3z"), 2),))
    value = replace(value, players=tuple(players))
    return DecisionContext(input=value, legal_actions=cf.discard_actions(value))


def _open_discard_decision():
    """南のPonを持つ副露手の打牌判断。"""
    value = cf.policy_input("123m456p78s13z5p")
    south = _tile("2z")
    players = list(value.players)
    players[0] = replace(
        players[0],
        melds=(
            PublicMeld(
                kind=MeldKind.PON,
                tiles=(south, south, south),
                from_seat=Seat.SEAT_2,
                called_tile=south,
            ),
        ),
    )
    value = replace(value, players=tuple(players))
    return DecisionContext(input=value, legal_actions=cf.discard_actions(value))


def _vector(encoding, decision, action):
    encoded = encoding.encode(decision)
    return encoded.vectors[encoded.actions.index(action)]


def _l2a_teacher(decision):
    """L2aの教師規則に従うfixture教師。CALLでは最後の候補（鳴き）を選ぶ。"""
    kind = classify_l2a_decision(decision)
    action = guard_action(decision, kind)
    if action is not None:
        return action
    if kind is L2aDecisionKind.DISCARD:
        return TwoStepUkeirePolicy().choose_action(decision)
    return ENCODING_R1.encode(decision).actions[-1]


def _l2a_decisions():
    return (
        cf.discard_decision(cf.SECOND_NEAR_HAND),
        cf.tsumo_decision(),
        cf.riichi_decision(),
        cf.response_decision(),
        _chi_response(),
        _single_candidate_discard(),
        cf.ron_response_decision(),
        _riichi_opponent_decision(),
    )


def _v2_rows(decisions, *, game_ordinal, seed, split, teacher):
    """宣言牌の位置を持つPolicyInput v2のrow（C0 sourceと同じshape）。"""
    rows = []
    for ordinal, decision in enumerate(decisions):
        value = decision.input
        without_order = replace(
            value,
            players=tuple(
                replace(player, riichi_declaration_discard_order=None)
                for player in value.players
            ),
        )
        row = fixtures.source_row(
            game_ordinal=game_ordinal,
            seed=seed,
            split=split,
            step_ordinal=ordinal,
            decision_ordinal=ordinal,
            policy_input_value=without_order,
            legal_actions=decision.legal_actions,
            selected_action=teacher(decision),
        )
        row["policy_input"] = policy_input_to_value_v2(value, ValueError, "input")
        rows.append(row)
    return rows


def _write_source(root, *, teacher=_l2a_teacher, seed_base=100):
    games = []
    for game_ordinal, split in enumerate(("TRAIN", "SELECT", "OFFLINE-EVAL")):
        seed = seed_base * (game_ordinal + 1)
        rows = _v2_rows(
            _l2a_decisions(),
            game_ordinal=game_ordinal,
            seed=seed,
            split=split,
            teacher=teacher,
        )
        games.append((split, seed, rows))
    return fixtures.write_policy_source_record(
        root, games, schema=POLICY_SOURCE_RECORD_SCHEMA_V2
    )


def _float32_bytes(values):
    return array("f", values).tobytes()


class EncodingIdentityTests(unittest.TestCase):
    def test_m0_is_the_unchanged_l02_encoding(self) -> None:
        self.assertEqual(ENCODING_M0.identity, encoding_block()["identity"])
        self.assertEqual(
            ENCODING_M0.identity, "lisjong-offense-l0.2-discard-candidate-encoding-v1"
        )
        decision = cf.discard_decision(cf.NEAR_HAND)
        candidates = build_scorer_candidates(decision)
        encoded = ENCODING_M0.encode(decision)
        self.assertEqual(encoded.vectors, encode_candidates(candidates))
        self.assertEqual(
            encoded.actions, tuple(candidate.action for candidate in candidates)
        )

    def test_m1_and_r1_have_their_own_identity_and_fingerprint(self) -> None:
        blocks = [item.block() for item in (ENCODING_M0, ENCODING_M1, ENCODING_R1)]
        self.assertEqual(len({block["identity"] for block in blocks}), 3)
        self.assertEqual(len({block["fingerprint"] for block in blocks}), 3)
        self.assertEqual(
            {block["classification"] for block in blocks},
            {"lisjong-l2a-decision-classification-v1"},
        )
        self.assertEqual(ENCODING_M1.dimension, _M0 + 37)
        self.assertEqual(ENCODING_R1.dimension, 95)

    def test_unknown_encoding_fails_closed(self) -> None:
        with self.assertRaises(CandidateFeatureError):
            l2a_encoding("m2")


class M1EncodingTests(unittest.TestCase):
    def test_m1_extends_m0(self) -> None:
        decision = cf.discard_decision(cf.NEAR_HAND)
        m0 = ENCODING_M0.encode(decision)
        m1 = ENCODING_M1.encode(decision)
        self.assertEqual(m1.actions, m0.actions)
        for base, extended in zip(m0.vectors, m1.vectors, strict=True):
            self.assertEqual(extended[:_M0], base)

    def test_riichi_genbutsu_passed_tile_and_suji(self) -> None:
        decision = _riichi_opponent_decision()
        by_tile = {
            action.tile.tile_type: action
            for action in ENCODING_M1.encode(decision).actions
        }

        def field(spec, name, position=0):
            vector = _vector(ENCODING_M1, decision, by_tile[_tile(spec).tile_type])
            return vector[_M0 + M1_BLOCK_OFFSETS[name] + position]

        self.assertEqual(field("1z", "opponent_shimocha"), 1.0)  # リーチ
        self.assertEqual(field("1z", "opponent_shimocha", 1), 1.0)  # 現物
        self.assertEqual(field("3z", "opponent_shimocha", 1), 0.0)
        self.assertEqual(field("3z", "opponent_shimocha", 2), 1.0)  # 通過牌
        self.assertEqual(field("3z", "safe_against_all_riichi"), 1.0)
        self.assertEqual(field("2z", "safe_against_all_riichi"), 0.0)
        self.assertEqual(field("6p", "opponent_shimocha", 3), 1.0)  # 9pの筋
        self.assertEqual(field("1z", "opponent_toimen"), 0.0)

    def test_dora_and_tile_class(self) -> None:
        # fixtureのドラ表示牌は北なので、ドラは東。自席は親の東家。
        decision = cf.discard_decision(cf.NEAR_HAND)
        east = _vector(ENCODING_M1, decision, cf.discard_actions(decision.input)[9])
        self.assertEqual(
            cf.discard_actions(decision.input)[9].tile.tile_type,
            _tile("1z").tile_type,
        )
        offsets = {name: _M0 + offset for name, offset in M1_BLOCK_OFFSETS.items()}
        self.assertEqual(east[offsets["discard_dora"]], 0.25)
        self.assertEqual(east[offsets["retained_dora"]], 1 / 8)
        self.assertEqual(east[offsets["tile_class"] + 1], 1.0)
        self.assertEqual(east[offsets["tile_class"] + 2], 1.0)  # 連風牌
        one_man = _vector(ENCODING_M1, decision, cf.discard_actions(decision.input)[0])
        self.assertEqual(one_man[offsets["tile_class"]], 1.0)
        self.assertEqual(one_man[offsets["retained_dora"]], 2 / 8)

    def test_open_routes_are_masked_by_existing_melds(self) -> None:
        decision = _open_discard_decision()
        offsets = {name: _M0 + offset for name, offset in M1_BLOCK_OFFSETS.items()}
        for vector in ENCODING_M1.encode(decision).vectors:
            self.assertEqual(vector[offsets["open_route_tanyao"]], 0.0)
            self.assertEqual(vector[offsets["open_route_tanyao"] + 1], 0.0)
            self.assertEqual(vector[offsets["open_route_honitsu_manzu"]], 1.0)

        closed = ENCODING_M1.encode(cf.discard_decision(cf.NEAR_HAND))
        for vector in closed.vectors:
            for name in ("tanyao", "honitsu_manzu", "honitsu_pinzu", "honitsu_souzu"):
                self.assertEqual(vector[offsets[f"open_route_{name}"]], 0.0)


class R1EncodingTests(unittest.TestCase):
    def test_pass_first_then_canonical_calls(self) -> None:
        decision = _call(
            "23m456p789s11223z",
            _pon("1z"),
            _chi("4m", ("2m", "3m")),
            _chi("1m", ("2m", "3m")),
        )
        actions = ENCODING_R1.encode(decision).actions
        self.assertIsInstance(actions[0], PassAction)
        self.assertEqual(
            [type(action) for action in actions[1:]], [ChiAction, ChiAction, PonAction]
        )
        self.assertEqual(actions[1].called_tile, _tile("1m"))
        self.assertTrue(
            all(any(a is b for b in decision.legal_actions) for a in actions)
        )

    def test_kind_called_and_consumed_fields(self) -> None:
        decision = _chi_response()
        chi = decision.legal_actions[1]
        vector = _vector(ENCODING_R1, decision, chi)
        offsets = R1_BLOCK_OFFSETS
        self.assertEqual(vector[offsets["kind"] : offsets["kind"] + 3], (0.0, 1.0, 0.0))
        self.assertEqual(vector[offsets["called_tile_type"]], 1.0)  # 1m
        self.assertEqual(vector[offsets["consumed_tile_type"] + 1], 0.5)  # 2m
        passed = _vector(ENCODING_R1, decision, decision.legal_actions[0])
        self.assertEqual(passed[offsets["kind"]], 1.0)
        self.assertEqual(
            sum(passed[offsets["called_tile_type"] : offsets["called_red"]]), 0
        )

    def test_route_mask_for_incompatible_call(self) -> None:
        decision = cf.response_decision()  # 東のPon
        offsets = R1_BLOCK_OFFSETS
        pon = _vector(ENCODING_R1, decision, decision.legal_actions[1])
        passed = _vector(ENCODING_R1, decision, decision.legal_actions[0])
        tanyao = offsets["route_tanyao"]
        self.assertEqual(pon[tanyao : tanyao + 4], (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(passed[tanyao], 1.0)
        self.assertEqual(pon[offsets["yakuhai_triplet"]], 1.0)
        self.assertEqual(passed[offsets["yakuhai_triplet"]], 0.0)

    def test_route_values_are_independent_minimums_over_allowed_discards(self) -> None:
        decision = _chi_response()
        chi = decision.legal_actions[1]  # 1mを23mでChi。1m・4mは切れない
        concealed = list(decision.input.own_hand.concealed_tiles)
        for tile in chi.consumed_tiles:
            concealed.remove(tile)
        hands = [
            concealed[:index] + concealed[index + 1 :]
            for index, tile in enumerate(concealed)
            if tile.tile_type not in {_tile("1m").tile_type, _tile("4m").tile_type}
        ]
        vector = _vector(ENCODING_R1, decision, chi)
        for route in encoding_module.R1_ROUTES:
            base = R1_BLOCK_OFFSETS[f"route_{route.name}"]
            call_tiles = (chi.called_tile, *chi.consumed_tiles)
            if not all(
                tile.tile_type in route.usable_tile_types for tile in call_tiles
            ):
                self.assertEqual(vector[base], 0.0)
                continue
            expected = min(
                calculate_restricted_standard_shanten(hand, route.usable_tile_types)
                for hand in hands
            )
            self.assertEqual(vector[base + 1], expected / 13)

    def test_route_shanten_above_the_standard_bound_is_encodable(self) -> None:
        # 么九牌と字牌だけの手はTanyaoで使える牌が無く、route向聴が8を超える。
        hand = cf.hand("19m19p19s1234567z")
        tanyao = encoding_module._TANYAO_ROUTE.usable_tile_types
        self.assertGreater(encoding_module._route_shanten(hand, tanyao), 8)
        decision = _call("19m19p9s11234567z", _pon("1z"))
        passed = ENCODING_R1.encode(decision).vectors[0]
        base = R1_BLOCK_OFFSETS["route_tanyao"]
        self.assertGreater(passed[base + 1], 8 / 13)

    def test_kuikae_rules(self) -> None:
        forbidden = encoding_module._kuikae_forbidden
        types = lambda *specs: frozenset(_tile(spec).tile_type for spec in specs)  # noqa: E731
        self.assertEqual(forbidden(_chi("1m", ("2m", "3m"))), types("1m", "4m"))
        self.assertEqual(forbidden(_chi("4m", ("2m", "3m"))), types("4m", "1m"))
        self.assertEqual(forbidden(_chi("3m", ("2m", "4m"))), types("3m"))
        self.assertEqual(forbidden(_chi("7m", ("8m", "9m"))), types("7m"))
        self.assertEqual(forbidden(_pon("5p")), types("5p"))

    def test_call_without_allowed_discard_fails_closed(self) -> None:
        decision = _call("123m4m", _chi("1m", ("2m", "3m")))
        with self.assertRaises(CandidateFeatureError):
            ENCODING_R1.encode(decision)


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source_path = _write_source(self.root / "source")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _publish(self, encoding, name=None, **options):
        source = open_source_record(self.source_path)
        return publish_l2a_dataset(
            source, self.root / (name or encoding.name), encoding=encoding, **options
        )

    def test_accounting_and_single_candidate_exclusion(self) -> None:
        manifest = self._publish(ENCODING_M1)
        train = manifest["rows"]["splits"]["TRAIN"]
        self.assertEqual(train["source_decisions"], 8)
        self.assertEqual(train["discard"], 3)
        self.assertEqual(train["single_candidate"], 1)
        self.assertEqual(train["scorer_decisions"], 2)
        self.assertEqual(train["call"], 2)
        self.assertEqual(train["win"], 2)
        reaction = self._publish(ENCODING_R1)["rows"]["splits"]["TRAIN"]
        self.assertEqual(reaction["scorer_decisions"], 2)
        self.assertEqual(reaction["single_candidate"], 0)

    def test_train_serve_agreement(self) -> None:
        source = read_source_record(self.source_path)
        decisions = {
            (item.game_ordinal, item.decision_ordinal): item
            for item in source.decisions()
        }
        for encoding in (ENCODING_M0, ENCODING_M1, ENCODING_R1):
            with self.subTest(encoding=encoding.name):
                self._publish(encoding)
                dataset = read_l2a_dataset(self.root / encoding.name)
                dimension = encoding.dimension
                for index, row in enumerate(dataset.rows):
                    item = decisions[(row.game_ordinal, row.decision_ordinal)]
                    decision = DecisionContext(
                        input=item.policy_input, legal_actions=item.legal_actions
                    )
                    recorder = _RecordingScorer(encoding)
                    served = _policy(recorder).decide(decision)
                    self.assertEqual(served.candidates, row.actions)
                    start = row.candidate_offset * dimension
                    stored = dataset.candidates[
                        start : start + row.candidate_count * dimension
                    ]
                    self.assertEqual(
                        stored.tobytes(),
                        b"".join(_float32_bytes(v) for v in recorder.vectors),
                    )
                    context = dataset.context[
                        index * FEATURE_DIMENSION : (index + 1) * FEATURE_DIMENSION
                    ]
                    self.assertEqual(
                        context.tobytes(), _float32_bytes(recorder.context)
                    )
                    self.assertEqual(
                        row.actions[row.teacher_candidate_index], item.selected_action
                    )

    def test_parallel_output_is_identical(self) -> None:
        serial = self._publish(ENCODING_R1, "serial")
        parallel = self._publish(ENCODING_R1, "parallel", workers=2)
        self.assertEqual(serial["identity"], parallel["identity"])

    def test_unresolvable_teacher_fails_closed(self) -> None:
        def teacher(decision):
            for action in decision.legal_actions:
                if isinstance(action, AnkanAction):
                    return action
            return _l2a_teacher(decision)

        source = open_source_record(_write_source(self.root / "ankan", teacher=teacher))
        with self.assertRaises(DatasetError):
            publish_l2a_dataset(source, self.root / "bad", encoding=ENCODING_M1)
        self.assertFalse((self.root / "bad").exists())

    def test_tampered_dataset_fails_closed(self) -> None:
        self._publish(ENCODING_M1)
        path = self.root / "m1"
        manifest = fixtures.read_manifest_body(path)
        manifest["encoding"] = dict(manifest["encoding"], fingerprint="0" * 64)
        fixtures.rewrite_manifest(path, manifest)
        with self.assertRaises(DatasetError):
            read_l2a_dataset(path)

        self._publish(ENCODING_R1)
        payload = self.root / "r1" / "candidates.f32"
        data = bytearray(payload.read_bytes())
        data[0] ^= 1
        payload.write_bytes(bytes(data))
        with self.assertRaises(DatasetError):
            read_l2a_dataset(self.root / "r1")


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        source = open_source_record(_write_source(self.root / "source"))
        publish_l2a_dataset(source, self.root / "m1", encoding=ENCODING_M1)
        self.dataset = read_l2a_dataset(self.root / "m1")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _write(self, name="artifact", **overrides):
        config = L2aScorerConfig(encoding=ENCODING_M1, hidden_width=64)
        training = fixtures.training_block(
            objective="per-decision-legal-candidate-softmax-cross-entropy"
        )
        return write_l2a_artifact(
            self.root / name,
            dataset=self.dataset,
            model_config=config,
            training=training,
            weights=fixtures.zero_weights(config),
            **overrides,
        )

    def test_artifact_binds_identities(self) -> None:
        artifact = self._write()
        manifest = artifact.manifest
        self.assertEqual(manifest["encoding"], ENCODING_M1.block())
        self.assertEqual(manifest["dataset"]["identity"], self.dataset.identity)
        self.assertEqual(manifest["source"], self.dataset.manifest["source"])
        self.assertEqual(manifest["model"]["encoding"], "m1")
        self.assertEqual(manifest["training"]["selected_epoch"], 1)
        self.assertIn("sha256", manifest["files"]["weights.f32"])

    def test_identity_mismatch_fails_load(self) -> None:
        self._write()
        path = self.root / "artifact"
        mutations = {
            "encoding fingerprint": lambda body: body["encoding"].update(
                fingerprint="0" * 64
            ),
            "classification": lambda body: body["encoding"].update(
                classification="other"
            ),
            "model encoding": lambda body: body["model"].update(encoding="r1"),
            "feature": lambda body: body["feature"].update(fingerprint="0" * 64),
            "weights digest": lambda body: body["files"]["weights.f32"].update(
                sha256="0" * 64
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                body = fixtures.read_manifest_body(path)
                mutate(body)
                broken = self.root / name.replace(" ", "-")
                broken.mkdir()
                (broken / "weights.f32").write_bytes(
                    (path / "weights.f32").read_bytes()
                )
                (broken / "manifest.json").write_text(
                    canonical_json_text(seal(body)), encoding="utf-8", newline="\n"
                )
                with self.assertRaises(ModelArtifactError):
                    load_l2a_artifact(broken)

    def test_model_encoding_must_match_dataset(self) -> None:
        config = L2aScorerConfig(encoding=ENCODING_R1, hidden_width=64)
        with self.assertRaises(ModelArtifactError):
            write_l2a_artifact(
                self.root / "mismatch",
                dataset=self.dataset,
                model_config=config,
                training=fixtures.training_block(),
                weights=fixtures.zero_weights(config),
            )

    def test_held_out_split_is_rejected_by_artifact_write_and_load(self) -> None:
        objective = "per-decision-legal-candidate-softmax-cross-entropy"
        config = L2aScorerConfig(encoding=ENCODING_M1, hidden_width=64)
        for field in ("train_splits", "validation_splits"):
            with self.subTest(field=field):
                training = fixtures.training_block(
                    objective=objective, **{field: ["OFFLINE-EVAL"]}
                )
                with self.assertRaises(ModelArtifactError):
                    write_l2a_artifact(
                        self.root / f"held-out-{field}",
                        dataset=self.dataset,
                        model_config=config,
                        training=training,
                        weights=fixtures.zero_weights(config),
                    )
                self.assertFalse((self.root / f"held-out-{field}").exists())

                self._write(f"valid-{field}")
                path = self.root / f"valid-{field}"
                body = fixtures.read_manifest_body(path)
                body["training"][field] = ["OFFLINE-EVAL"]
                (path / "manifest.json").write_text(
                    canonical_json_text(seal(body)), encoding="utf-8", newline="\n"
                )
                with self.assertRaises(ModelArtifactError):
                    load_l2a_artifact(path)

    def test_unregistered_width_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            L2aScorerConfig(encoding=ENCODING_M1, hidden_width=128)


class TrainingConfigTests(unittest.TestCase):
    def test_train_and_select_splits_are_accepted(self) -> None:
        from lisjong.learning.l2a_training import L2aTrainingConfig

        config = L2aTrainingConfig(train_splits=("TRAIN",), select_splits=("SELECT",))
        self.assertEqual(
            (config.train_splits, config.select_splits), (("TRAIN",), ("SELECT",))
        )

    def test_offline_eval_is_rejected_for_training_and_selection(self) -> None:
        from lisjong.learning.l2a_training import L2aTrainingConfig

        for train, select in (
            (("TRAIN",), ("OFFLINE-EVAL",)),
            (("OFFLINE-EVAL",), ("SELECT",)),
            (("TRAIN", "OFFLINE-EVAL"), ("SELECT",)),
        ):
            with self.subTest(train=train, select=select):
                with self.assertRaises(ValueError):
                    L2aTrainingConfig(train_splits=train, select_splits=select)


class _RecordingScorer:
    def __init__(self, encoding, scorer=None) -> None:
        self.encoding = encoding
        self.scorer = scorer or (lambda vector: 0.0)
        self.calls = 0

    def score(self, context, vectors):
        self.calls += 1
        self.context = context
        self.vectors = vectors
        return tuple(self.scorer(vector) for vector in vectors)


class _FakeRuntime:
    def __init__(self, discard, reaction) -> None:
        self.discard = discard
        self.reaction = reaction

    def scorer_for(self, kind):
        return self.discard if kind is L2aDecisionKind.DISCARD else self.reaction


def _policy(scorer):
    if scorer.encoding.head is L2aDecisionKind.DISCARD:
        return L2aPolicy(_FakeRuntime(scorer, _RecordingScorer(ENCODING_R1)))
    return L2aPolicy(_FakeRuntime(_RecordingScorer(ENCODING_M1), scorer))


class PolicyTests(unittest.TestCase):
    def test_guard_and_single_candidate_do_not_call_scorers(self) -> None:
        discard = _RecordingScorer(ENCODING_M1)
        reaction = _RecordingScorer(ENCODING_R1)
        policy = L2aPolicy(_FakeRuntime(discard, reaction))
        for decision in (
            cf.tsumo_decision(),
            cf.riichi_decision(),
            cf.ron_response_decision(),
            _single_candidate_discard(),
            cf.discard_decision(cf.NEAR_HAND, tsumogiri_only=True),
        ):
            result = policy.decide(decision)
            self.assertTrue(any(result.action is a for a in decision.legal_actions))
            self.assertIsNone(result.scores)
        self.assertEqual((discard.calls, reaction.calls), (0, 0))

    def test_scorer_choice_and_canonical_tie_break(self) -> None:
        decision = _chi_response()
        tied = L2aPolicy(
            _FakeRuntime(_RecordingScorer(ENCODING_M1), _RecordingScorer(ENCODING_R1))
        )
        self.assertIsInstance(tied.choose_action(decision), PassAction)

        kind = R1_BLOCK_OFFSETS["kind"]
        prefer_chi = _RecordingScorer(ENCODING_R1, lambda vector: vector[kind + 1])
        chosen = _policy(prefer_chi).choose_action(decision)
        self.assertIs(chosen, decision.legal_actions[1])  # canonical順で最初のChi

    def test_daiminkan_is_never_selected(self) -> None:
        tile = _tile("1z")
        decision = _call(
            "23m456p789s11123z",
            _pon("1z"),
            DaiminkanAction(
                actor=Seat.SEAT_0,
                target=Seat.SEAT_2,
                called_tile=tile,
                consumed_tiles=(tile, tile, tile),
            ),
        )
        reaction = _RecordingScorer(ENCODING_R1, lambda vector: 1.0)
        action = _policy(reaction).choose_action(decision)
        self.assertNotIsInstance(action, DaiminkanAction)

    def test_non_finite_score_fails_closed(self) -> None:
        reaction = _RecordingScorer(ENCODING_R1, lambda vector: float("nan"))
        with self.assertRaises(LearnedPolicyError):
            _policy(reaction).choose_action(_chi_response())
        with self.assertRaises(LearnedPolicyError):
            select_best(())


class InformationBoundaryTests(unittest.TestCase):
    def test_l2a_modules_do_not_import_policy_modules(self) -> None:
        package = Path(encoding_module.__file__).parent
        for name in _L2A_MODULES:
            tree = ast.parse((package / f"{name}.py").read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                    names = []
                else:
                    continue
                for module in modules:
                    with self.subTest(module=name, imported=module):
                        self.assertFalse(module.startswith("lisjong.policies"))
                        self.assertFalse(module.startswith("lisjong_arena"))
                        self.assertFalse(module.startswith("riichienv"))
                        self.assertNotIn("ground_truth", module)
                        if module.startswith("lisjong.belief"):
                            self.assertFalse(any(n.startswith("_") for n in names))


@unittest.skipUnless(_HAS_TORCH, "requires the optional ML runtime")
class TrainingAndRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source_path = _write_source(self.root / "source")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _train(self, encoding, name=None, source_path=None):
        from lisjong.learning.l2a_training import L2aTrainingConfig, train_l2a_scorer

        name = name or encoding.name
        source = open_source_record(source_path or self.source_path)
        publish_l2a_dataset(source, self.root / f"{name}-dataset", encoding=encoding)
        dataset = read_l2a_dataset(self.root / f"{name}-dataset")
        config = L2aTrainingConfig(
            train_splits=("TRAIN",), select_splits=("SELECT",), epochs=3, batch_size=2
        )
        return train_l2a_scorer(dataset, config, self.root / f"{name}-artifact")

    def test_training_records_epoch_cross_entropy_and_runtime_loads(self) -> None:
        from lisjong.learning.l2a_policy import load_l2a_policy_factory

        discard = self._train(ENCODING_M1)
        reaction = self._train(ENCODING_R1)
        diagnostics = discard.manifest["training"]["diagnostics"]
        for epoch in (1, 2, 3):
            self.assertIn(f"epoch_{epoch:03d}_train_cross_entropy", diagnostics)
            self.assertIn(f"epoch_{epoch:03d}_select_cross_entropy", diagnostics)
        self.assertIn(discard.manifest["training"]["selected_epoch"], (1, 2, 3))

        runtime = load_l2a_policy_factory(
            self.root / "m1-artifact", self.root / "r1-artifact"
        )
        self.assertEqual(runtime.identity, (discard.identity, reaction.identity))
        policy = runtime()
        for item in read_source_record(self.source_path).decisions():
            decision = DecisionContext(
                input=item.policy_input, legal_actions=item.legal_actions
            )
            action = policy.choose_action(decision)
            self.assertTrue(any(action is legal for legal in decision.legal_actions))

    def test_runtime_rejects_wrong_head_and_mixed_sources(self) -> None:
        from lisjong.learning.l2a_policy import load_l2a_policy_factory

        self._train(ENCODING_M1)
        self._train(ENCODING_R1)
        with self.assertRaises(LearnedPolicyError):
            load_l2a_policy_factory(
                self.root / "r1-artifact", self.root / "r1-artifact"
            )

        other = _write_source(self.root / "other-source", seed_base=1000)
        self._train(ENCODING_R1, "other", other)
        with self.assertRaises(LearnedPolicyError):
            load_l2a_policy_factory(
                self.root / "m1-artifact", self.root / "other-artifact"
            )


if __name__ == "__main__":
    unittest.main()
