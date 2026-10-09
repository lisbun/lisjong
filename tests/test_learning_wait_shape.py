import ast
import contextlib
import hashlib
import inspect
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from test_learning_hand_belief_source import (
    SEED,
    all_decisions,
    all_facts,
    ankan,
    tt,
    write_source,
)

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.fixed_point import SCALE
from lisjong.belief.hand_belief import HandBelief
from lisjong.learning import wait_shape_estimator as estimator
from lisjong.learning import wait_shape_evaluation as evaluation
from lisjong.learning.hand_belief_accuracy import (
    HandBeliefAccuracyError,
    Riichi245Estimator,
)
from lisjong.learning.hand_belief_source import label_decisions
from lisjong.learning.open_wait_estimator import OpenWaitModel
from lisjong.learning.riichi_wait_estimator import BIAS, LogisticWaitModel
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat

NAMES = estimator.CHANNEL_NAMES
RIICHI_WAIT = LogisticWaitModel(weights=())  # every tile 0.5
OPEN_WAIT = OpenWaitModel(tenpai_weights=(), wait_weights=())  # every tile 0.25
LEVEL1 = evaluation.Level1(
    riichi=Riichi245Estimator(RIICHI_WAIT, "r" * 64),
    open=OPEN_WAIT,
    open_sha256="o" * 64,
)


def index(spec):
    return tile_type_index(tt(spec))


def shape_model(population, bias=0.0, **overrides):
    """全channelが同じ切片だけを持つモデル。``overrides``でchannelごとに差し替える。"""
    return estimator.WaitShapeModel(
        population,
        tuple(
            None
            if population == "open" and name == estimator.KOKUSHI
            else overrides.get(name, ((BIAS, bias),))
            for name in NAMES
        ),
    )


def pi_of(sequence):
    return {d.key.sequence: d for d in all_decisions()}[sequence].policy_input


def tables(belief):
    return {name: getattr(belief, field) for name, field, _ in estimator.CHANNELS}


def labelled():
    return label_decisions(all_decisions(), all_facts())


def with_seed(items, seed):
    return [replace(item, key=replace(item.key, seed=seed)) for item in items]


def rows_of(population):
    target = evaluation.SplitRows.empty()
    evaluation.build_rows(labelled(), LEVEL1, target)
    return target.rows[population]


class BoundaryTest(unittest.TestCase):
    def test_inference_does_not_import_ground_truth(self):
        tree = ast.parse(Path(inspect.getfile(estimator)).read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        forbidden = (
            "hand_belief_source",
            "_evaluation",
            "_accuracy",
            "ground_truth",
            "riichi_ron_label",
            "ron_legal_source",
            "lisjong_arena",
        )
        self.assertFalse(any(word in name for name in imported for word in forbidden))

    def test_inference_takes_only_the_player_safe_input_and_frozen_models(self):
        for function in (
            estimator.estimate_riichi_wait_shape_belief,
            estimator.estimate_open_wait_shape_belief,
        ):
            self.assertEqual(
                list(inspect.signature(function).parameters),
                ["policy_input", "seat", "wait_model", "shape_model"],
            )


class ScopeTest(unittest.TestCase):
    def test_unsupported_seats_are_rejected_not_zero_filled(self):
        a = pi_of(20)  # seat1 riichi, seat2 pon, seat3 ankan only
        riichi, opened = shape_model("riichi"), shape_model("open")
        for seat in (Seat.SEAT_0, Seat.SEAT_2, Seat.SEAT_3):
            with self.assertRaises(ValueError):
                estimator.estimate_riichi_wait_shape_belief(
                    a, seat, RIICHI_WAIT, riichi
                )
        for seat in (Seat.SEAT_0, Seat.SEAT_1, Seat.SEAT_3):
            with self.assertRaises(ValueError):
                estimator.estimate_open_wait_shape_belief(a, seat, OPEN_WAIT, opened)
        # 判断B: リーチ者がいない
        with self.assertRaises(ValueError):
            estimator.estimate_riichi_wait_shape_belief(
                pi_of(30), Seat.SEAT_1, RIICHI_WAIT, riichi
            )

    def test_a_second_riichi_takes_the_riichi_seat_out_of_scope(self):
        a = pi_of(20)
        players = list(a.players)
        players[3] = replace(players[3], riichi=RiichiState.ACCEPTED)
        with self.assertRaises(ValueError):
            estimator.estimate_riichi_wait_shape_belief(
                replace(a, players=tuple(players)),
                Seat.SEAT_1,
                RIICHI_WAIT,
                shape_model("riichi"),
            )

    def test_models_are_bound_to_their_population(self):
        a = pi_of(20)
        with self.assertRaises(ValueError):
            estimator.estimate_riichi_wait_shape_belief(
                a, Seat.SEAT_1, RIICHI_WAIT, shape_model("open")
            )
        with self.assertRaises(ValueError):
            estimator.estimate_open_wait_shape_belief(
                a, Seat.SEAT_2, OPEN_WAIT, shape_model("riichi")
            )
        with self.assertRaises(TypeError):
            estimator.estimate_open_wait_shape_belief(
                a, Seat.SEAT_2, RIICHI_WAIT, shape_model("open")
            )


class ModelTest(unittest.TestCase):
    def test_bad_models_are_rejected(self):
        good = shape_model("riichi").weights
        cases = {
            "unknown population": dict(population="closed", weights=good),
            "missing channel": dict(population="riichi", weights=good[:-1]),
            "missing model": dict(population="riichi", weights=(*good[:-1], None)),
            "open kokushi has weights": dict(population="open", weights=good),
            "non-finite": dict(
                population="riichi", weights=(((BIAS, float("nan")),), *good[1:])
            ),
            "infinite": dict(
                population="riichi", weights=(((BIAS, float("inf")),), *good[1:])
            ),
            "non-float": dict(population="riichi", weights=(((BIAS, 1),), *good[1:])),
            "unknown feature set": dict(
                population="riichi", weights=good, feature_set="wait-shape-features-v0"
            ),
        }
        for name, arguments in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                estimator.WaitShapeModel(**arguments)


class BeliefTest(unittest.TestCase):
    def riichi(self, model, wait=RIICHI_WAIT, policy_input=None):
        return estimator.estimate_riichi_wait_shape_belief(
            policy_input or pi_of(20), Seat.SEAT_1, wait, model
        )

    def test_the_belief_is_level_2_and_keeps_the_level_1_tables(self):
        belief = self.riichi(shape_model("riichi", bias=-1.0))
        self.assertIsInstance(belief, HandBelief)
        self.assertTrue(belief.has_wait_mechanism_belief)
        self.assertTrue(all(table is not None for table in tables(belief).values()))
        self.assertFalse(belief.has_ron_legal_belief)
        from lisjong.learning.riichi_wait_estimator import estimate_riichi_wait_belief

        level1 = estimate_riichi_wait_belief(pi_of(20), RIICHI_WAIT)
        self.assertEqual(belief.wait_probability_raw, level1.wait_probability_raw)
        self.assertEqual(belief.expected_count_raw, level1.expected_count_raw)
        self.assertEqual(
            belief.red_five_probability_raw, level1.red_five_probability_raw
        )

    def test_slots_a_channel_cannot_occupy_are_canonical_zero(self):
        belief = self.riichi(shape_model("riichi", bias=5.0))
        for name, _, static in estimator.CHANNELS:
            table = tables(belief)[name]
            self.assertFalse(any(table[i] for i in range(34) if i not in static), name)
            self.assertTrue(any(table), name)

    def test_every_channel_is_capped_by_the_wait_in_raw(self):
        # q = sigmoid(5) > w = 0.5: 推定するslotはすべて上限に当たる
        belief = self.riichi(shape_model("riichi", bias=5.0))
        wait = belief.wait_probability_raw
        self.assertEqual(set(wait), {SCALE // 2})
        for name, table in tables(belief).items():
            self.assertTrue(set(table) <= {0, SCALE // 2}, name)
        # q < w: 上限に当たらず、qの丸め値のまま
        low = self.riichi(shape_model("riichi", bias=-3.0))
        self.assertEqual(low.tanki_wait_probability_raw[index("1m")], 389)

    def test_a_zero_wait_zeroes_every_channel_and_a_full_wait_does_not_cap(self):
        zero = self.riichi(
            shape_model("riichi", bias=5.0), LogisticWaitModel(((BIAS, -50.0),))
        )
        self.assertEqual(set(zero.wait_probability_raw), {0})
        self.assertTrue(all(set(table) == {0} for table in tables(zero).values()))
        full = self.riichi(
            shape_model("riichi", bias=0.0), LogisticWaitModel(((BIAS, 50.0),))
        )
        self.assertEqual(set(full.wait_probability_raw), {SCALE})
        self.assertEqual(full.tanki_wait_probability_raw[index("1m")], SCALE // 2)

    def test_the_cap_holds_at_the_rounding_boundary(self):
        # 0.5 raw unitは偶数側へ丸める: 2.5 -> 2、3.5 -> 4
        q = [[2.5 / SCALE, 3.5 / SCALE, 3.49 / SCALE, 1.0, 0.0]]
        self.assertEqual(
            estimator.cap_by_wait(q, [2, 3, 3, SCALE, 0]), ((2, 3, 3, SCALE, 0),)
        )
        self.assertEqual(estimator.cap_by_wait(q, [SCALE] * 5), ((2, 4, 3, SCALE, 0),))

    def test_channels_overlap_and_low_and_high_sides_are_separate(self):
        belief = self.riichi(
            shape_model(
                "riichi",
                bias=-1.0,
                ryanmen_low_side=((BIAS, -2.0),),
                ryanmen_high_side=((BIAS, -4.0),),
            )
        )
        slot = index("4m")  # 単騎・双碰・嵌張・両面の両側が同時に非ゼロ
        values = {name: table[slot] for name, table in tables(belief).items()}
        for name in ("tanki", "shanpon", "kanchan"):
            self.assertEqual(values[name], 2203)
        self.assertNotEqual(values["ryanmen_low_side"], values["ryanmen_high_side"])
        self.assertGreater(sum(values.values()), belief.wait_probability_raw[slot])

    def test_genbutsu_and_an_exhausted_wait_tile_are_not_forced_to_zero(self):
        # 判断A: リーチ者の河に9p。9mは暗槓で残り0枚
        belief = self.riichi(shape_model("riichi", bias=-1.0))
        self.assertGreater(belief.tanki_wait_probability_raw[index("9p")], 0)
        self.assertGreater(belief.ryanmen_high_side_probability_raw[index("9p")], 0)
        # 9m自身は残り0枚でも、78mの両面の高い側としては成り立ち得る
        self.assertGreater(belief.ryanmen_high_side_probability_raw[index("9m")], 0)
        # 9mを手牌に必要とする形は成り立たない
        self.assertEqual(belief.tanki_wait_probability_raw[index("9m")], 0)
        self.assertEqual(belief.shanpon_wait_probability_raw[index("9m")], 0)
        self.assertEqual(belief.kanchan_wait_probability_raw[index("8m")], 0)
        self.assertEqual(belief.penchan_wait_probability_raw[index("7m")], 0)
        self.assertGreater(belief.penchan_wait_probability_raw[index("3m")], 0)

    def test_river_features_never_fix_a_shape_at_zero(self):
        heavy = self.riichi(
            shape_model("riichi", tanki=((BIAS, 0.0), ("genbutsu", -8.0)))
        )
        self.assertGreater(heavy.tanki_wait_probability_raw[index("9p")], 0)
        self.assertLess(
            heavy.tanki_wait_probability_raw[index("9p")],
            heavy.tanki_wait_probability_raw[index("1m")],
        )

    def test_kokushi_is_its_own_channel_and_needs_a_hand_without_melds(self):
        model = shape_model("riichi", bias=-9.0, kokushi=((BIAS, -1.0),))
        belief = self.riichi(model)
        kokushi = belief.kokushi_wait_probability_raw
        self.assertEqual(kokushi[index("1m")], 2203)
        self.assertEqual(kokushi[index("9m")], 2203)  # 残り0枚でもmaskしない
        self.assertEqual(kokushi[index("5m")], 0)
        self.assertEqual(belief.tanki_wait_probability_raw[index("1m")], 1)

        a = pi_of(20)
        players = list(a.players)
        players[1] = replace(players[1], melds=(ankan("1111s"),))
        with_ankan = self.riichi(model, policy_input=replace(a, players=tuple(players)))
        self.assertEqual(set(with_ankan.kokushi_wait_probability_raw), {0})

    def test_an_open_seat_has_no_kokushi_and_uses_both_feature_groups(self):
        b = pi_of(30)
        model = shape_model(
            "open", penchan=((BIAS, -1.0), ("tenpai.melds_1", 1.0), ("tile.dora", 9.0))
        )
        belief = estimator.estimate_open_wait_shape_belief(
            b, Seat.SEAT_2, OPEN_WAIT, model
        )
        self.assertTrue(belief.has_wait_mechanism_belief)
        self.assertEqual(set(belief.kokushi_wait_probability_raw), {0})
        self.assertEqual(belief.penchan_wait_probability_raw[index("3m")], 2048)
        self.assertEqual(belief.tanki_wait_probability_raw[index("3m")], 2048)
        from lisjong.learning.open_wait_estimator import open_view

        table, shared = estimator.open_shape_features(open_view(b, 2))
        names = {name for features in table for name, _ in features}
        self.assertEqual(
            {name.split(".")[0] for name in names if name != BIAS}, {"tile"}
        )
        self.assertTrue(all(name.startswith("tenpai.") for name, _ in shared))
        self.assertEqual(sum(name == BIAS for name, _ in table[0]), 1)


class RowTest(unittest.TestCase):
    def test_rows_split_into_the_two_populations_and_count_the_rest(self):
        target = evaluation.SplitRows.empty()
        evaluation.build_rows(labelled(), LEVEL1, target)
        # A: seat1 riichi / seat2 open / seat3 ankan only、B: seat2 open、C: seat1 riichi
        self.assertEqual([(row.episode[1]) for row in target.rows["riichi"]], [1, 1])
        self.assertEqual([(row.episode[1]) for row in target.rows["open"]], [2, 2])
        self.assertEqual(target.coverage, {"riichi": 2, "open": 2, "unprovided": 5})

    def test_labels_are_the_structural_shapes(self):
        a, c = rows_of("riichi")
        positives = {
            name: [i for i, y in enumerate(labels) if y]
            for name, labels in zip(NAMES, a.labels)
        }
        self.assertEqual(positives["ryanmen_low_side"], [index("1m")])
        self.assertEqual(positives["ryanmen_high_side"], [index("4m")])
        self.assertEqual(positives["tanki"], [])
        self.assertEqual(
            [i for i, y in enumerate(c.labels[NAMES.index("tanki")]) if y],
            [index("5z")],
        )
        _, b = rows_of("open")
        self.assertEqual(
            [i for i, y in enumerate(b.labels[NAMES.index("penchan")]) if y],
            [index("3m")],
        )

    def test_a_riichi_decision_outside_s1_is_unprovided(self):
        decisions = all_decisions()
        only = decisions[0].legal_actions[:1]  # 合法打牌の牌種が1つ
        decisions[0] = replace(
            decisions[0], legal_actions=only, selected_action=only[0]
        )
        target = evaluation.SplitRows.empty()
        evaluation.build_rows(label_decisions(decisions, all_facts()), LEVEL1, target)
        self.assertEqual(target.coverage["riichi"], 1)
        self.assertEqual(target.coverage["unprovided"], 6)

    def test_row_predictions_equal_the_inference_belief(self):
        models = {
            "riichi": shape_model("riichi", bias=-1.5, tanki=((BIAS, 2.0),)),
            "open": shape_model("open", bias=-0.5),
        }
        estimate = {
            "riichi": lambda pi, seat: estimator.estimate_riichi_wait_shape_belief(
                pi, seat, RIICHI_WAIT, models["riichi"]
            ),
            "open": lambda pi, seat: estimator.estimate_open_wait_shape_belief(
                pi, seat, OPEN_WAIT, models["open"]
            ),
        }
        sequences = {"riichi": (20, 40), "open": (20, 30)}
        for population in estimator.POPULATIONS:
            for row, sequence in zip(rows_of(population), sequences[population]):
                belief = estimate[population](pi_of(sequence), Seat(row.episode[1]))
                _, raw = evaluation.predict_raw(row, models[population])
                self.assertEqual(row.wait_raw, belief.wait_probability_raw)
                self.assertEqual(raw, tuple(tables(belief).values()))


class FitTest(unittest.TestCase):
    def test_rate_weights_each_episode_once_within_the_population(self):
        rates = evaluation.fit_rate(rows_of("open"))
        penchan = rates[NAMES.index("penchan")]
        # A・Bは同じ局の同じ席（1エピソード2行）で、片方だけが3mの辺張待ち
        self.assertAlmostEqual(penchan[index("3m")], (0.5 + 0.5) / (1 + 1))
        self.assertAlmostEqual(penchan[index("7m")], 0.5 / 2)
        self.assertEqual(penchan[index("4m")], 0.0)  # 辺張が占め得ないslot
        riichi = evaluation.fit_rate(rows_of("riichi"))
        # A・Cは同じ局の同じ席で1エピソード
        self.assertAlmostEqual(
            riichi[NAMES.index("tanki")][index("5z")], (0.5 + 0.5) / 2
        )

    def test_fit_samples_use_the_static_slot_count_and_skip_unsupported_slots(self):
        rows = rows_of("riichi")
        for name in ("tanki", "penchan", "kokushi"):
            channel = NAMES.index(name)
            static = len(estimator.CHANNELS[channel][2])
            _, patterns = evaluation.channel_patterns(rows, channel)
            total = sum(weight for weight, _ in patterns.values())
            supported = sum(len(row.supported[channel]) for row in rows)
            # 各行の重み0.5を静的slot数で割り、推定するslotの分だけを足す
            self.assertAlmostEqual(total, 0.5 * supported / static, msg=name)
        _, patterns = evaluation.channel_patterns(rows, NAMES.index("tanki"))
        self.assertAlmostEqual(
            sum(positive for _, positive in patterns.values()), 0.5 / 34
        )

    def test_l2_also_shrinks_the_intercept(self):
        rows = rows_of("riichi")
        channel = NAMES.index("shanpon")  # 正例なし
        fits = evaluation.fit_channel(rows, channel, (0.01, 100.0))
        weak, strong = (dict(fits[l2])[BIAS] for l2 in (0.01, 100.0))
        self.assertLess(weak, strong)
        self.assertLess(strong, 0.0)
        self.assertGreater(weak, -20.0)  # 正例がなくても発散しない

    def test_the_open_kokushi_is_a_zero_model_without_a_fit(self):
        rows = rows_of("open")
        model, _, selected, grid = evaluation.select_population("open", rows, rows)
        self.assertIsNone(model.weights[NAMES.index("kokushi")])
        self.assertIsNone(selected["kokushi"])
        self.assertEqual(grid["kokushi"], "structural zero model")
        self.assertEqual(len(grid["tanki"]), len(evaluation.L2_GRID))

    def test_ties_select_the_first_l2(self):
        rows = rows_of("open")
        _, _, selected, grid = evaluation.select_population("open", rows, rows)
        for name, candidates in grid.items():
            if name == "kokushi":
                continue
            best = min(candidate["loss"] for candidate in candidates)
            first = next(c["l2"] for c in candidates if c["loss"] == best)
            self.assertEqual(selected[name], first)


class MetricTest(unittest.TestCase):
    def test_metrics_use_the_static_slots_as_the_denominator(self):
        channel = NAMES.index("penchan")
        labels = [False] * 34
        labels[index("3m")] = True
        estimate = [0.0] * 34
        estimate[index("3m")] = 0.5
        metrics = evaluation.channel_metrics(
            channel,
            labels,
            {"estimator": estimate, "rate": [0.0] * 34, "capped_rate": [0.0] * 34},
        )
        from math import log

        zero = -log(1 - 1e-6)
        self.assertAlmostEqual(metrics["positive_rate.static"], 1 / 6)
        self.assertAlmostEqual(
            metrics["estimator.log_loss.static"], (log(2) + 5 * zero) / 6
        )
        self.assertAlmostEqual(
            metrics["estimator.log_loss.all"], (log(2) + 33 * zero) / 34
        )
        self.assertAlmostEqual(
            metrics["rate.log_loss.static"], (-log(1e-6) + 5 * zero) / 6
        )
        self.assertAlmostEqual(
            metrics["delta.log_loss.static"],
            metrics["estimator.log_loss.static"] - metrics["rate.log_loss.static"],
        )

    def test_judgement_excludes_before_it_compares(self):
        def counts(episodes, hanchan):
            return {"positive_episodes": episodes, "positive_hanchan": hanchan}

        judge = evaluation.judgement
        self.assertEqual(judge(0, counts(0, 0), None)["reason"], "no_rows")
        self.assertEqual(judge(9, counts(0, 0), -1.0)["reason"], "no_positive")
        self.assertEqual(
            judge(9, counts(500, 49), -1.0),
            {"verdict": "excluded", "reason": "positive_hanchan_below_minimum"},
        )
        self.assertEqual(
            judge(9, counts(99, 50), -1.0)["reason"], "positive_episodes_below_minimum"
        )
        self.assertEqual(judge(9, counts(100, 50), -1e-9)["verdict"], "improved")
        self.assertEqual(judge(9, counts(100, 50), 0.0)["verdict"], "not_confirmed")
        self.assertEqual(judge(9, counts(100, 50), 0.3)["verdict"], "not_confirmed")

    def test_the_report_counts_slots_and_the_cap(self):
        rows = rows_of("riichi")
        report = evaluation.Report()
        # q = sigmoid(5) > w = 0.5: 推定するslotはすべて上限処理の対象になる
        report.add(rows, shape_model("riichi", bias=5.0), evaluation.fit_rate(rows))
        document = report.document([SEED])
        self.assertEqual(document["counts"], {"rows": 2, "episodes": 1, "hanchan": 1})
        tanki = document["channels"]["tanki"]
        counts = tanki["counts"]
        self.assertEqual(counts["static_slots"], 68)
        self.assertEqual(
            counts["estimated_slots"],
            sum(len(row.supported[0]) for row in rows),
        )
        self.assertEqual(
            counts["over_wait_slots_before_cap"], counts["estimated_slots"]
        )
        self.assertEqual(counts["over_wait_slots_after_cap"], 0)
        self.assertGreater(counts["mean_excess_over_wait_before_cap"], 0.49)
        self.assertEqual((counts["positive_rows"], counts["positive_slots"]), (1, 1))
        self.assertEqual(tanki["verdict"], "excluded")
        self.assertEqual(tanki["reason"], "positive_hanchan_below_minimum")
        self.assertEqual(document["channels"]["shanpon"]["reason"], "no_positive")
        delta = tanki["intervals"]["delta.log_loss.static"]["point"]
        macro = tanki["episode_macro"]
        self.assertAlmostEqual(
            delta,
            macro["estimator.log_loss.static"] - macro["rate.log_loss.static"],
        )

    def test_a_population_without_rows_is_excluded_with_its_reason(self):
        document = evaluation.Report().document([SEED])
        self.assertEqual(document["counts"]["rows"], 0)
        for channel in document["channels"].values():
            self.assertEqual(
                (channel["verdict"], channel["reason"]), ("excluded", "no_rows")
            )


class SelectAndTestTest(unittest.TestCase):
    REVISION = "a" * 40

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.d = Path(directory.name)
        for offset, split in enumerate(("train", "valid", "test")):
            self.write(
                self.d / split,
                SEED + offset,
                {
                    name: [SEED + offset] if name == split else []
                    for name in ("train", "valid", "test")
                },
            )
        self.write(
            self.d / "new", SEED + 3, {"train": [], "valid": [], "test": [SEED + 3]}
        )
        self.sources = [self.d / "train", self.d / "valid", self.d / "test"]
        self.expected = {"train": [SEED], "valid": [SEED + 1], "eval": [SEED + 2]}

    def write(self, root, seed, splits):
        root.mkdir()
        write_source(
            root,
            with_seed(all_decisions(), seed),
            with_seed(all_facts(), seed),
            splits=splits,
        )

    def selection(self):
        chosen = evaluation.select(self.sources, self.expected, LEVEL1, self.REVISION)
        return json.loads(json.dumps(chosen))

    def test_select_then_one_test_on_new_seeds(self):
        selection = self.selection()
        self.assertEqual(selection["feature_set"], "wait-shape-features-v1")
        self.assertEqual(selection["consistency"], "cap-by-wait-v1")
        self.assertEqual(selection["code_revision"], self.REVISION)
        self.assertEqual(selection["level1"], LEVEL1.identity())
        self.assertEqual(
            selection["coverage"]["dev_eval"],
            {"riichi": 2, "open": 2, "unprovided": 5},
        )
        for population in estimator.POPULATIONS:
            value = selection["populations"][population]
            self.assertEqual(list(value["model"]["weights"]), list(NAMES))
            self.assertEqual(set(value["dev_eval"]["channels"]), set(NAMES))
            self.assertEqual(value["dev_eval"]["counts"]["rows"], 2)
        self.assertIsNone(
            selection["populations"]["open"]["model"]["weights"]["kokushi"]
        )

        result = evaluation.run_test(
            [self.d / "new"], [SEED + 3], selection, LEVEL1, self.REVISION
        )
        self.assertEqual(set(result["verdicts"]), set(estimator.POPULATIONS))
        for population in estimator.POPULATIONS:
            self.assertEqual(set(result["verdicts"][population]), set(NAMES))
            self.assertEqual(set(result["verdicts"][population].values()), {"excluded"})
        self.assertEqual(result["coverage"]["unprovided"], 5)

    def test_a_seed_used_by_the_selection_is_rejected(self):
        with self.assertRaises(evaluation.WaitShapeEvaluationError):
            evaluation.run_test(
                [self.d / "test"], [SEED + 2], self.selection(), LEVEL1, self.REVISION
            )

    def test_another_wait_estimator_or_policy_is_rejected(self):
        selection = self.selection()
        other = replace(LEVEL1, open_sha256="x" * 64)
        cases = {
            "level1": (selection, other),
            "consistency": (selection | {"consistency": "none"}, LEVEL1),
            "feature set": (selection | {"feature_set": "v0"}, LEVEL1),
            "schema": (selection | {"schema": "other"}, LEVEL1),
        }
        for name, (chosen, level1) in cases.items():
            with (
                self.subTest(name),
                self.assertRaises(evaluation.WaitShapeEvaluationError),
            ):
                evaluation.run_test(
                    [self.d / "new"], [SEED + 3], chosen, level1, self.REVISION
                )

    def test_a_source_of_another_producer_is_rejected(self):
        selection = self.selection()
        selection["population"]["producer"] = dict(
            selection["population"]["producer"], lisjong_revision="0" * 40
        )
        with self.assertRaises(HandBeliefAccuracyError):
            evaluation.run_test(
                [self.d / "new"], [SEED + 3], selection, LEVEL1, self.REVISION
            )

    def test_the_level_1_selections_are_identified_by_their_digests(self):
        from lisjong.learning import open_wait_evaluation

        path = self.d / "open.json"
        path.write_text(
            json.dumps(open_wait_evaluation.select(self.sources, self.expected)),
            encoding="utf-8",
        )
        with self.assertRaises(evaluation.WaitShapeEvaluationError):
            evaluation.load_level1(path, "0" * 64, path, "0" * 64)

    def arguments(self, command, output, *extra):
        return [
            command,
            *extra,
            "--riichi-wait-selection",
            "unused",
            "--riichi-wait-sha256",
            "r" * 64,
            "--open-wait-selection",
            "unused",
            "--open-wait-sha256",
            "o" * 64,
            "--output",
            str(output),
        ]

    def test_the_command_line_selects_then_tests_once(self):
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        patched = mock.patch.object(evaluation, "load_level1", return_value=LEVEL1)
        patched.start()
        self.addCleanup(patched.stop)
        output = self.d / "selection.json"
        splits = ("--train", SEED, "--valid", SEED + 1, "--dev-eval", SEED + 2)
        select = self.arguments("select", output, *map(str, splits))
        sources = list(map(str, self.sources))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            evaluation.main([*select, "--code-revision", "main", *sources])
        self.assertFalse(output.exists())
        evaluation.main([*select, "--code-revision", self.REVISION, *sources])
        digest = hashlib.sha256(output.read_bytes()).hexdigest()

        def test(result, sha256):
            return [
                *self.arguments(
                    "test",
                    result,
                    "--test",
                    str(SEED + 3),
                    "--selection",
                    str(output),
                    "--selection-sha256",
                    sha256,
                ),
                "--code-revision",
                self.REVISION,
                str(self.d / "new"),
            ]

        result = self.d / "result.json"
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            evaluation.main(test(result, "0" * 64))
        self.assertFalse(result.exists())
        evaluation.main(test(result, digest))
        document = json.loads(result.read_text(encoding="utf-8"))
        self.assertEqual(document["selection_sha256"], digest)
        self.assertEqual(document["test_seeds"], [SEED + 3])
        # 出力は上書きしない
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            evaluation.main(test(result, digest))


if __name__ == "__main__":
    unittest.main()
