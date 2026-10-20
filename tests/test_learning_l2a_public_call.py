"""#326: R1-public-call encodingと限定開発比較の集計・境界。実験dataは使わない。"""

import copy
import importlib.util
import json
import random
import tempfile
import unittest
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import candidate_fixtures as cf
import learning_fixtures as fixtures
from test_learning_l2a_scorer import _call, _chi, _l2a_teacher, _pon, _v2_rows

from lisjong.hand_evaluation import calculate_shanten
from lisjong.learning import l2a_encoding as encoding_module
from lisjong.learning.errors import CandidateFeatureError, LearnedPolicyError
from lisjong.learning.l2a_dataset import (
    iter_l2a_dataset_payload,
    publish_l2a_dataset,
    read_l2a_dataset,
)
from lisjong.learning.l2a_encoding import (
    ENCODING_R1,
    ENCODING_R1_PUBLIC_CALL,
    R1_ENCODING_DIMENSION,
    R1_PUBLIC_CALL_BLOCK_OFFSETS,
    PublicSituationMode,
    l2a_encoding,
    public_call_analysis,
    public_situation_mode,
)
from lisjong.learning.l2a_public_call_experiment import (
    BINS,
    CONFIG,
    EXPECTED_A,
    HANDS,
    SEED_GROUPS,
    SEED_KEYS,
    ModelTally,
    collect,
    derived_thresholds,
    expected_a_check,
    judge,
    main,
    paired_bootstrap,
    reconcile_with_324,
    restricted_source,
)
from lisjong.learning.source_record import (
    POLICY_SOURCE_RECORD_SCHEMA_V2,
    open_source_record,
)
from lisjong.policies import placement_aware_speed_call as teacher
from lisjong.policy_contract import (
    ChiAction,
    PassAction,
    PonAction,
    Seat,
)
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.wind import Wind

_HAS_TORCH = importlib.util.find_spec("torch") is not None
FROZEN_R1_FINGERPRINT = (
    "732ca9550252592768c898c71c6222c14b086865ab6f8ee86956e3ceba99917f"
)
"""#318の凍結R1（A）のmanifestがbindするR1 fingerprint。"""


def _tile(spec):
    return cf.hand(spec)[0]


def _with_melds(decision, *melds):
    players = list(decision.input.players)
    players[0] = replace(players[0], melds=melds)
    return replace(decision, input=replace(decision.input, players=tuple(players)))


def _pon_meld(spec, kind=MeldKind.PON):
    tile = _tile(spec)
    if kind is MeldKind.ANKAN:
        return PublicMeld(
            kind=kind, tiles=(tile,) * 4, from_seat=None, called_tile=None
        )
    return PublicMeld(
        kind=kind, tiles=(tile,) * 3, from_seat=Seat.SEAT_2, called_tile=tile
    )


def _with_round(decision, **values):
    value = decision.input
    return replace(decision, input=replace(value, round=replace(value.round, **values)))


def _with_scores(decision, scores):
    players = tuple(
        replace(player, score=score)
        for player, score in zip(decision.input.players, scores, strict=True)
    )
    return replace(decision, input=replace(decision.input, players=players))


def _decisions():
    """門前・赤ドラ・暗槓・副露（字牌・route不成立）・清一色の反応判断。"""
    red = _with_round(
        _call(
            "340m2340p255678s",
            _chi("2m", ("3m", "4m")),
            _chi("5m", ("3m", "4m")),
            _chi("1p", ("2p", "3p")),
            _pon("5s"),
        ),
        dora_indicators=(_tile("4p"), _tile("4s")),
    )
    ankan = _with_melds(
        _call("1122m3344p55s", _pon("1m"), _chi("3m", ("1m", "2m"))),
        _pon_meld("9p", MeldKind.ANKAN),
    )
    honor = _with_melds(
        _call("123m456p78s13z", _chi("6s", ("7s", "8s")), _chi("9s", ("7s", "8s"))),
        _pon_meld("2z"),
    )
    no_route = _with_melds(
        _call("234s567p8m", _chi("4p", ("5p", "6p"))),
        _pon_meld("1m"),
        _pon_meld("9p"),
    )
    return {
        "closed": _call(
            "23m456p789s11223z", _chi("1m", ("2m", "3m")), _chi("4m", ("2m", "3m"))
        ),
        "chiitoitsu": _call("1122m3344p5566s7z", _pon("1m"), _pon("5s")),
        "red_dora": red,
        "ankan": ankan,
        "honor": honor,
        "no_route": no_route,
        "chinitsu": _call(
            "1112345678999m", _chi("2m", ("1m", "3m")), _pon("9m"), _pon("1m")
        ),
    }


def _teacher_values(policy_input, action):
    """教師C0のhelperで同じ量を求める（test側だけの対応確認）。"""
    concealed = policy_input.own_hand.concealed_tiles
    melds = teacher._own_meld_tiles(policy_input)
    dora = policy_input.round.dora_indicators
    if teacher._is_closed_hand(policy_input):
        current = calculate_shanten(concealed)
    else:
        routes = teacher._compatible_routes(melds)
        current = (
            teacher._best_route_evaluation(concealed, melds, routes, dora)[0]
            if routes
            else None
        )
    if isinstance(action, PassAction):
        routes = teacher._compatible_routes(melds)
        joint = (
            teacher._best_route_evaluation(concealed, melds, routes, dora)
            if routes
            else None
        )
    else:
        call_melds = (*melds, *action.consumed_tiles, action.called_tile)
        routes = teacher._compatible_routes(call_melds)
        hands = teacher._post_call_stable_hands(concealed, action)
        joint = (
            min(
                (
                    teacher._best_route_evaluation(hand, call_melds, routes, dora)
                    for hand in hands
                ),
                key=lambda item: (item[0], -item[1]),
            )
            if routes and hands
            else None
        )
    return current, joint


class EncodingTests(unittest.TestCase):
    def test_identity_layout_and_r1_prefix(self):
        self.assertEqual(ENCODING_R1.fingerprint, FROZEN_R1_FINGERPRINT)
        self.assertEqual(ENCODING_R1_PUBLIC_CALL.dimension, R1_ENCODING_DIMENSION + 11)
        self.assertEqual(l2a_encoding("r1-public-call"), ENCODING_R1_PUBLIC_CALL)
        self.assertNotEqual(ENCODING_R1_PUBLIC_CALL.identity, ENCODING_R1.identity)
        self.assertEqual(
            ENCODING_R1_PUBLIC_CALL.specification["base"], ENCODING_R1.block()
        )
        for decision in _decisions().values():
            base = ENCODING_R1.encode(decision)
            extended = ENCODING_R1_PUBLIC_CALL.encode(decision)
            self.assertEqual(base.actions, extended.actions)
            for short, long in zip(base.vectors, extended.vectors, strict=True):
                self.assertEqual(long[:R1_ENCODING_DIMENSION], short)

    def test_values_match_teacher_helpers(self):
        for name, decision in _decisions().items():
            actions = tuple(
                a
                for a in decision.legal_actions
                if isinstance(a, (PassAction, ChiAction, PonAction))
            )
            values = public_call_analysis(decision.input, actions)
            for action, value in zip(actions, values, strict=True):
                current, joint = _teacher_values(decision.input, action)
                with self.subTest(fixture=name, action=action):
                    self.assertEqual(value.current_shanten, current)
                    self.assertEqual(
                        (value.joint_route_shanten, value.joint_route_value),
                        (None, None) if joint is None else joint,
                    )

    def test_closed_uses_special_forms_and_ankan_is_closed(self):
        decisions = _decisions()
        values = public_call_analysis(
            decisions["chiitoitsu"].input, decisions["chiitoitsu"].legal_actions
        )
        self.assertEqual(values[0].current_shanten, 0)  # 七対子聴牌
        self.assertGreater(values[0].joint_route_shanten, 0)
        ankan = decisions["ankan"]
        values = public_call_analysis(ankan.input, ankan.legal_actions)
        self.assertEqual(
            values[0].current_shanten,
            calculate_shanten(ankan.input.own_hand.concealed_tiles),
        )

    def test_route_failure_masks_and_improvement(self):
        decision = _decisions()["no_route"]
        vectors = ENCODING_R1_PUBLIC_CALL.encode(decision).vectors
        offsets = R1_PUBLIC_CALL_BLOCK_OFFSETS
        for vector in vectors:
            extra = vector[R1_ENCODING_DIMENSION:]
            for name in (
                "current_shanten",
                "joint_route_shanten",
                "shanten_improvement",
                "joint_route_value",
            ):
                self.assertEqual(extra[offsets[name] : offsets[name] + 2], (0.0, 0.0))
        decision = _decisions()["closed"]
        encoded = ENCODING_R1_PUBLIC_CALL.encode(decision)
        values = public_call_analysis(decision.input, encoded.actions)
        self.assertEqual(values[0].shanten_improvement, 0)  # Passは差の基準
        for vector, value in zip(encoded.vectors, values, strict=True):
            extra = vector[R1_ENCODING_DIMENSION:]
            base = offsets["shanten_improvement"]
            self.assertEqual(extra[base], 1.0)
            self.assertEqual(extra[base + 1], value.shanten_improvement / 13)
            if value.joint_route_shanten is not None and value is not values[0]:
                self.assertEqual(
                    value.shanten_improvement,
                    value.current_shanten - value.joint_route_shanten,
                )

    def test_kuikae_without_allowed_discard_is_masked(self):
        # 1mを23mでChiすると残る1m・4mはどちらも喰い替えで切れない。
        decision = _call("123m4m", _chi("1m", ("2m", "3m")))
        values = public_call_analysis(decision.input, decision.legal_actions)
        self.assertIsNone(values[1].joint_route_shanten)
        self.assertIsNone(values[1].joint_route_value)
        self.assertIsNone(values[1].shanten_improvement)

    def test_red_dora_and_chinitsu_values(self):
        hand = cf.hand("340m2340p255678s")
        dora = Counter({_tile("5p").tile_type: 1})
        tanyao = encoding_module._TANYAO_ROUTE
        # Tanyao 1 + 赤5m 1 + 赤5p 1 + 5pのドラ 1。
        self.assertEqual(encoding_module._yaku_route_value(tanyao, hand, (), dora), 4)
        manzu = encoding_module.M1_OPEN_ROUTES[1]
        chinitsu = cf.hand("1112345678999m")[:13]
        self.assertEqual(
            encoding_module._yaku_route_value(manzu, chinitsu, (), Counter()), 5
        )
        with_honor = cf.hand("111234567899m1z")
        self.assertEqual(
            encoding_module._yaku_route_value(manzu, with_honor, (), Counter()), 2
        )

    def test_same_minimum_shanten_takes_maximum_value(self):
        plain = cf.hand("234m456p678s2345s")
        red = cf.hand("234m456p678s2340s")
        best = encoding_module._best_yaku_route((plain, red), (), Counter())
        tanyao = encoding_module._TANYAO_ROUTE.usable_tile_types
        self.assertEqual(
            encoding_module._route_shanten(plain, tanyao),
            encoding_module._route_shanten(red, tanyao),
        )
        self.assertEqual(best[1], 2)  # Tanyao 1 + 赤5s

    def test_value_outside_range_fails_closed(self):
        decision = _decisions()["red_dora"]
        with (
            patch.object(encoding_module, "MAX_ROUTE_VALUE", 1),
            self.assertRaises(CandidateFeatureError),
        ):
            ENCODING_R1_PUBLIC_CALL.encode(decision)

    def test_situation_mode_boundaries_match_teacher(self):
        base = _decisions()["closed"]
        cases = [
            (dict(round_wind=Wind.EAST, hand_number=4), (40000, 20000, 20000, 20000)),
            (dict(round_wind=Wind.SOUTH, hand_number=3), (40000, 20000, 20000, 20000)),
            (dict(round_wind=Wind.SOUTH, hand_number=4), (40000, 20000, 20000, 20000)),
            (dict(round_wind=Wind.SOUTH, hand_number=4), (30000, 30000, 20000, 20000)),
            (dict(round_wind=Wind.WEST, hand_number=1), (35000, 25000, 20000, 20000)),
            (dict(round_wind=Wind.SOUTH, hand_number=4), (10000, 30000, 30000, 14100)),
            (dict(round_wind=Wind.SOUTH, hand_number=4), (10000, 30000, 30000, 14000)),
            (dict(round_wind=Wind.SOUTH, hand_number=4), (10000, 30000, 30000, 10000)),
            (
                dict(round_wind=Wind.SOUTH, hand_number=4, dealer_seat=Seat.SEAT_0),
                (10000, 30000, 30000, 20000),
            ),
        ]
        expected = [
            PublicSituationMode.NORMAL,
            PublicSituationMode.NORMAL,
            PublicSituationMode.ALL_LAST_TOP_SPEED,
            PublicSituationMode.NORMAL,
            PublicSituationMode.ALL_LAST_TOP_SPEED,
            PublicSituationMode.ALL_LAST_LAST_VALUE,
            PublicSituationMode.NORMAL,
            PublicSituationMode.NORMAL,
            PublicSituationMode.NORMAL,
        ]
        offsets = R1_PUBLIC_CALL_BLOCK_OFFSETS
        for (round_values, scores), mode in zip(cases, expected, strict=True):
            round_values.setdefault("dealer_seat", Seat.SEAT_1)
            decision = _with_scores(_with_round(base, **round_values), scores)
            with self.subTest(round=round_values, scores=scores):
                self.assertEqual(public_situation_mode(decision.input), mode)
                self.assertEqual(teacher.situation_mode(decision.input).name, mode.name)
                for vector in ENCODING_R1_PUBLIC_CALL.encode(decision).vectors:
                    one_hot = vector[R1_ENCODING_DIMENSION:][
                        offsets["situation_mode"] : offsets["situation_mode"] + 3
                    ]
                    self.assertEqual(
                        one_hot,
                        tuple(float(m is mode) for m in PublicSituationMode),
                    )

    def test_features_ignore_teacher_and_legal_action_order(self):
        decision = _decisions()["red_dora"]
        shuffled = replace(
            decision,
            legal_actions=(decision.legal_actions[0], *decision.legal_actions[:0:-1]),
        )
        self.assertEqual(
            ENCODING_R1_PUBLIC_CALL.encode(decision).vectors,
            ENCODING_R1_PUBLIC_CALL.encode(shuffled).vectors,
        )


# --- experiment ----------------------------------------------------------------


def _scorer(encoding):
    return SimpleNamespace(
        encoding=encoding,
        score=lambda context, vectors: tuple(float(i) for i in range(len(vectors))),
    )


def _write_split_source(root, purpose="SCIENTIFIC"):
    games = []
    decisions = (cf.discard_decision(cf.SECOND_NEAR_HAND), *_decisions().values())
    for ordinal, split in enumerate(("TRAIN", "SELECT", "OFFLINE-EVAL")):
        seed = 100 + ordinal
        rows = _v2_rows(
            decisions,
            game_ordinal=ordinal,
            seed=seed,
            split=split,
            teacher=_l2a_teacher,
        )
        games.append((split, seed, rows))
    fixtures.write_policy_source_record(
        root, games, schema=POLICY_SOURCE_RECORD_SCHEMA_V2, purpose=purpose
    )
    return open_source_record(root)


class RestrictedSourceTests(unittest.TestCase):
    def test_materialization_and_train_serve_without_offline_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_split_source(root / "source")
            forbidden = source.root / "game-002" / "source-record.jsonl"
            forbidden.write_text("malformed and forbidden", encoding="utf-8")
            view = restricted_source(source)
            self.assertEqual([s["split"] for s in view.summaries], ["TRAIN", "SELECT"])
            self.assertEqual(set(view.allocation_bindings), {"TRAIN", "SELECT"})
            manifest = publish_l2a_dataset(
                view, root / "dataset", encoding=ENCODING_R1_PUBLIC_CALL
            )
            self.assertEqual(
                {e["split"] for e in manifest["source"]["population"]},
                {"TRAIN", "SELECT"},
            )
            dataset = read_l2a_dataset(root / "dataset")
            self.assertEqual(set(dataset.split_counts()), {"TRAIN", "SELECT"})

            scorers = {"A": _scorer(ENCODING_R1), "B": _scorer(ENCODING_R1_PUBLIC_CALL)}
            result = collect(
                source,
                scorers,
                view.summaries,
                payload=iter_l2a_dataset_payload(
                    root / "dataset", splits=("TRAIN", "SELECT")
                ),
            )
            self.assertEqual(result["games"], {"TRAIN": 1, "SELECT": 1})
            rows = result["models"]["B"]["SELECT"]["groups"]["all"]["all"]
            self.assertGreater(rows["counts"]["rows"]["rows"], 0)

            tampered = list(
                iter_l2a_dataset_payload(root / "dataset", splits=("TRAIN", "SELECT"))
            )
            tampered[0] = replace(tampered[0], candidates=b"\0" * 4)
            with self.assertRaisesRegex(LearnedPolicyError, "train-serve"):
                collect(source, scorers, view.summaries, payload=tampered)
            with self.assertRaisesRegex(LearnedPolicyError, "not re-encoded"):
                collect(
                    source,
                    scorers,
                    view.summaries[:1],
                    payload=iter_l2a_dataset_payload(
                        root / "dataset", splits=("TRAIN", "SELECT")
                    ),
                )

    def test_offline_eval_game_is_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _write_split_source(Path(tmp) / "source")
            with (
                patch(
                    "lisjong.learning.l2a_public_call_experiment.read_source_game"
                ) as reader,
                self.assertRaises(LearnedPolicyError),
            ):
                collect(source, {"A": _scorer(ENCODING_R1)}, (source.summaries[-1],))
            reader.assert_not_called()


def _fake_report(values):
    report = {}
    for split in ("TRAIN", "SELECT"):
        report[split] = {
            "groups": {
                hand: {b: {"counts": defaultdict(lambda: {"rows": 0})} for b in BINS}
                for hand in HANDS
            }
        }
    for path, number in values.items():
        split, hand, b, key = path.split("/")
        report[split]["groups"][hand][b]["counts"][key] = {"rows": number}
    return report


_A_THRESHOLDS = {
    "SELECT/all/all/pon.fp": 220,
    "SELECT/closed/all/pon.fp": 196,
    "SELECT/open/all/pon.fp": 24,
}
_B_PASSING = {
    "TRAIN/closed/chi_legal/chi.tp": 680,
    "TRAIN/closed/chi_legal/chi.teacher_positive": 699,
    "SELECT/closed/chi_legal/chi.tp": 60,
    "SELECT/closed/chi_legal/chi.teacher_positive": 93,
    "SELECT/closed/chi_legal/chi.fp": 5,
    "SELECT/open/chi_legal/chi.fp": 30,
    "SELECT/open/chi_legal/chi.tp": 120,
    "SELECT/open/chi_legal/chi.teacher_positive": 188,
    "SELECT/all/all/pon.tp": 570,
    "SELECT/all/all/pon.teacher_positive": 727,
    "SELECT/all/all/pon.fp": 210,
    "SELECT/closed/all/pon.fp": 190,
    "SELECT/open/all/pon.fp": 20,
    "SELECT/closed/pon_only/pon.fp": 190,
    "SELECT/all/all/mismatch": 500,
    "SELECT/closed/all/mismatch": 380,
}
_INTERVAL = {"metrics": {"closed_chi_recall": {"status": "COMPLETE", "lower": 0.2}}}
_INTERVAL["metrics"]["closed_chi_recall"]["upper"] = 0.5


class JudgementTests(unittest.TestCase):
    def setUp(self):
        self.a = _fake_report(EXPECTED_A)

    def verdict(self, changes=None, interval=_INTERVAL):
        b = _fake_report({**_B_PASSING, **(changes or {})})
        return judge(self.a, b, interval, _A_THRESHOLDS)

    def test_all_criteria_pass(self):
        result = self.verdict()
        self.assertEqual(result["verdict"], "PROMISING-DEVELOPMENT")
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_each_criterion_can_fail(self):
        # 1は件数を下げると2のgapも悪化するので、区間下限だけを動かして単独で落とす。
        low = {"metrics": {"closed_chi_recall": {"status": "COMPLETE"}}}
        low["metrics"]["closed_chi_recall"].update(lower=-0.01, upper=0.3)
        result = self.verdict(interval=low)
        failed = [c["name"] for c in result["checks"] if not c["passed"]]
        self.assertEqual(failed, ["1.closed_chi_recall_gain"])
        cases = {
            "2.closed_chi_train_select_gap": {"TRAIN/closed/chi_legal/chi.tp": 664},
            "3.false_chi_open_chi_wrong_candidate": {
                "SELECT/open/chi_legal/chi.fp": 37
            },
            "4.pon": {"SELECT/open/all/pon.fp": 25},
            "5.call_mismatch": {"SELECT/closed/all/mismatch": 397},
        }
        for name, change in cases.items():
            result = self.verdict(change)
            with self.subTest(criterion=name):
                self.assertEqual(result["verdict"], "NOT-SUPPORTED-DEVELOPMENT")
                failed = [c["name"] for c in result["checks"] if not c["passed"]]
                self.assertEqual(failed, [name])

    def test_boundaries(self):
        # 35/93は+14件（0.1505）で通過、34/93は不通過。wrong-candidateは1件で不通過。
        self.assertTrue(
            self.verdict({"SELECT/closed/chi_legal/chi.tp": 35})["checks"][0]["passed"]
        )
        self.assertFalse(
            self.verdict({"SELECT/closed/chi_legal/chi.tp": 34})["checks"][0]["passed"]
        )
        result = self.verdict({"SELECT/open/3+/chi.wrong_candidate": 1})
        self.assertFalse(result["checks"][2]["passed"])
        low = {"metrics": {"closed_chi_recall": {"status": "COMPLETE"}}}
        low["metrics"]["closed_chi_recall"].update(lower=0.0, upper=0.3)
        self.assertFalse(self.verdict(interval=low)["checks"][0]["passed"])

    def test_indeterminate_interval_stops(self):
        interval = {"metrics": {"closed_chi_recall": {"status": "INDETERMINATE"}}}
        self.assertEqual(self.verdict(interval=interval)["verdict"], "STOP/INVALID")

    def test_expected_a_and_derived_thresholds(self):
        self.assertEqual(expected_a_check(self.a)["status"], "MATCH")
        changed = _fake_report({**EXPECTED_A, "SELECT/all/all/mismatch": 549})
        self.assertEqual(expected_a_check(changed)["status"], "MISMATCH")
        self.assertEqual(derived_thresholds(_fake_report(_A_THRESHOLDS)), _A_THRESHOLDS)


def _by_seed(rows):
    """{seed: {key: count}}を全SEED_GROUPSへ同じ値で置く。"""
    return {
        f"{hand}/{b}": {
            str(seed): {key: counts.get(key, 0) for key in SEED_KEYS}
            for seed, counts in rows.items()
        }
        for hand, b in SEED_GROUPS
    }


class BootstrapTests(unittest.TestCase):
    def test_paired_resampling_is_pooled_and_deterministic(self):
        a = _by_seed(
            {
                1: {"chi.tp": 1, "chi.teacher_positive": 2},
                2: {"chi.tp": 0, "chi.teacher_positive": 1},
                3: {"chi.tp": 0, "chi.teacher_positive": 3},
            }
        )
        b = _by_seed(
            {
                1: {"chi.tp": 2, "chi.teacher_positive": 2},
                2: {"chi.tp": 1, "chi.teacher_positive": 1},
                3: {"chi.tp": 0, "chi.teacher_positive": 3},
            }
        )
        result = paired_bootstrap(a, b, replicates=5, seed=324)
        self.assertEqual(result, paired_bootstrap(a, b, replicates=5, seed=324))
        rng = random.Random(324)
        positives = {1: 2, 2: 1, 3: 3}
        gains = {1: 1, 2: 1, 3: 0}
        diffs = []
        for _ in range(5):
            sample = rng.choices([1, 2, 3], k=3)
            diffs.append(
                sum(gains[s] for s in sample) / sum(positives[s] for s in sample)
            )
        diffs.sort()
        primary = result["metrics"]["closed_chi_recall"]
        self.assertAlmostEqual(primary["lower"], diffs[0] + (diffs[1] - diffs[0]) * 0.1)
        self.assertAlmostEqual(primary["upper"], diffs[3] + (diffs[4] - diffs[3]) * 0.9)

    def test_identical_models_and_zero_denominator(self):
        same = _by_seed(
            {
                1: {"chi.tp": 1, "chi.teacher_positive": 1},
                2: {"chi.teacher_positive": 1},
            }
        )
        result = paired_bootstrap(same, copy.deepcopy(same), replicates=50, seed=324)
        self.assertEqual(result["metrics"]["all_mismatch"]["lower"], 0.0)
        self.assertEqual(result["metrics"]["closed_chi_recall"]["upper"], 0.0)
        zero = _by_seed({1: {}, 2: {}})
        result = paired_bootstrap(zero, zero, replicates=3, seed=324)
        self.assertEqual(
            result["metrics"]["closed_chi_recall"]["status"], "INDETERMINATE"
        )

    def test_mismatched_teacher_denominators_fail_closed(self):
        a = _by_seed({1: {"chi.teacher_positive": 1}})
        b = _by_seed({1: {"chi.teacher_positive": 2}})
        with self.assertRaises(LearnedPolicyError):
            paired_bootstrap(a, b, replicates=1, seed=324)


class ReconciliationTests(unittest.TestCase):
    def report(self):
        decision = _decisions()["closed"]
        actions = ENCODING_R1.encode(decision).actions
        tally = ModelTally()
        for split, seed in (("TRAIN", 1), ("SELECT", 2)):
            tally.observe(split, "closed", seed, actions, (0.0, 2.0, 1.0), actions[1])
            tally.observe(split, "closed", seed, actions, (2.0, 0.0, 1.0), actions[1])
        return tally.report()

    def test_counts_must_match_and_margin_rounding_is_separate(self):
        ours = self.report()
        reference = {
            "summary": {
                split: {"groups": copy.deepcopy(ours[split]["groups"])}
                for split in ("TRAIN", "SELECT")
            }
        }
        self.assertEqual(reconcile_with_324(ours, reference)["status"], "MATCH")
        margin = reference["summary"]["SELECT"]["groups"]["all"]["all"]["margins"]
        margin["chi_vs_non_chi"]["q25"] += 1e-12
        result = reconcile_with_324(ours, reference)
        self.assertEqual(result["status"], "MATCH")
        self.assertEqual(
            result["margin_rounding_differences"],
            ["SELECT/all/all/chi_vs_non_chi.q25"],
        )
        counts = reference["summary"]["TRAIN"]["groups"]["closed"]["2"]["counts"]
        counts["mismatch"] = {"rows": 2, "hanchan": 1}
        result = reconcile_with_324(ours, reference)
        self.assertEqual(result["status"], "MISMATCH")
        self.assertIn("TRAIN/closed/2/mismatch", result["action_count_mismatches"])

    def test_cross_entropy_and_seed_counts(self):
        report = self.report()["SELECT"]
        ce = report["cross_entropy"]["closed"]
        self.assertEqual(ce["rows"], 2)
        self.assertGreater(ce["mean"], 0)
        seeds = report["by_seed"]["closed/chi_legal"]
        self.assertEqual(seeds["2"]["chi.tp"], 1)
        self.assertEqual(seeds["2"]["mismatch"], 1)


class CliTests(unittest.TestCase):
    def test_failure_writes_stop_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "stop.json"
            argv = [
                "preflight",
                "--source",
                str(Path(tmp) / "missing"),
                "--a",
                str(Path(tmp) / "a"),
                "--output",
                str(output),
            ]
            self.assertEqual(main(argv), 2)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "STOP")
            self.assertEqual(report["config"], CONFIG)
            before = output.read_bytes()
            with self.assertRaises(SystemExit):
                main(argv)
            self.assertEqual(before, output.read_bytes())


@unittest.skipUnless(_HAS_TORCH, "requires the optional ML runtime")
class EpochCallbackTests(unittest.TestCase):
    def test_callback_records_epochs_and_can_stop_before_publication(self):
        from lisjong.learning.l2a_training import L2aTrainingConfig, train_l2a_scorer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_split_source(root / "source", purpose="DEVELOPMENT")
            publish_l2a_dataset(
                restricted_source(source),
                root / "dataset",
                encoding=ENCODING_R1_PUBLIC_CALL,
            )
            dataset = read_l2a_dataset(root / "dataset")
            config = L2aTrainingConfig(
                train_splits=("TRAIN",),
                select_splits=("SELECT",),
                epochs=2,
                batch_size=2,
            )
            seen = []
            artifact = train_l2a_scorer(
                dataset,
                config,
                root / "artifact",
                on_epoch=lambda *values: seen.append(values),
            )
            self.assertEqual([item[0] for item in seen], [1, 2])
            self.assertEqual(artifact.encoding, ENCODING_R1_PUBLIC_CALL)

            def stop(epoch, *_):
                raise LearnedPolicyError(f"budget at {epoch}")

            with self.assertRaisesRegex(LearnedPolicyError, "budget at 1"):
                train_l2a_scorer(dataset, config, root / "stopped", on_epoch=stop)
            self.assertFalse((root / "stopped").exists())


if __name__ == "__main__":
    unittest.main()
