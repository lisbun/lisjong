import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from test_learning_hand_belief_source import (
    A_HANDS,
    A_OWN,
    A_PLAYERS,
    SEED,
    all_decisions,
    all_facts,
    make_decision,
    make_facts,
    tiles,
    write_source,
)

from lisjong.learning import hand_belief_accuracy as accuracy
from lisjong.learning._canonical import canonical_json_text
from lisjong.learning.hand_belief_source import label_decisions
from lisjong.learning.riichi_wait_estimator import LogisticWaitModel
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.seat import Seat

MODEL = accuracy.Riichi245Estimator(
    LogisticWaitModel(weights=(("bias", -3.0),)), "0" * 64
)


def preset_for(**changes):
    fields = {
        "preset_id": "test-preset",
        "version": 1,
        "purpose": "development-baseline",
        "train": (str(SEED),),
        "valid": (),
        "eval": (str(SEED + 1),),
        "bootstrap_resamples": 50,
    }
    return accuracy.Preset(**(fields | changes))


class AltEstimator:
    """#245とは別の推定器: 全ての他家の行（リーチ者以外も）に定数を出す。"""

    name = "alt"

    def identity(self):
        return {"name": self.name, "kind": "constant"}

    def scope_seat(self, decision):
        return 2 if len(decision.legal_actions) >= 2 else None

    def predict(self, policy_input):
        return (0.25,) * 34


def labelled():
    return label_decisions(all_decisions(), all_facts())


def with_round(decision, **fields):
    pi = decision.policy_input
    return replace(
        decision, policy_input=replace(pi, round=replace(pi.round, **fields))
    )


def with_seed(items, seed):
    return [replace(item, key=replace(item.key, seed=seed)) for item in items]


class RowsTest(unittest.TestCase):
    def test_strata_follow_the_opponent_state_at_the_decision(self):
        rows = list(accuracy.build_rows(labelled(), MODEL))
        self.assertEqual(len(rows), 9)
        # A: seat1 riichi, seat2 pon, seat3 ankan only (closed)
        self.assertEqual(
            [row.stratum for row in rows[:3]], ["riichi", "open", "closed_non_riichi"]
        )
        # B: seat2 chi
        self.assertEqual(rows[4].stratum, "open")

    def test_245_is_provided_only_for_the_single_riichi_player_in_scope(self):
        rows = list(accuracy.build_rows(labelled(), MODEL))
        provided = [(row.episode[1], row.in_scope) for row in rows]
        self.assertEqual(
            [seat for seat, scope in provided if scope], [1, 1]
        )  # A and C seat 1
        for row in rows:
            self.assertEqual(row.scoped_wait is None, not row.in_scope)

    def test_fewer_than_two_candidate_types_is_out_of_scope(self):
        legal = (DiscardAction(actor=Seat(0), tile=tiles("2p")[0], tsumogiri=False),)
        decision = make_decision(20, A_OWN, A_PLAYERS, legal=legal)
        rows = list(
            accuracy.build_rows(
                label_decisions([decision], [make_facts(20, A_HANDS)]), MODEL
            )
        )
        self.assertFalse(any(row.in_scope for row in rows))


class KyokuInstanceTest(unittest.TestCase):
    def test_renchan_is_a_new_instance(self):
        decisions = all_decisions()
        decisions[1] = with_round(decisions[1], honba=1)
        decisions[2] = with_round(decisions[2], honba=1)
        rows = label_decisions(decisions, all_facts())
        self.assertEqual(accuracy.kyoku_instances(rows), {20: 0, 30: 1, 40: 1})

    def test_a_kyoku_that_reappears_is_rejected(self):
        decisions = all_decisions()
        decisions[1] = with_round(decisions[1], honba=1)
        rows = label_decisions(decisions, all_facts())
        with self.assertRaises(accuracy.HandBeliefAccuracyError):
            accuracy.kyoku_instances(rows)


class AggregationTest(unittest.TestCase):
    def test_strata_are_renormalized_within_each_episode(self):
        groups = accuracy._Groups()
        # episode 1: three rows (value 1) then one row (value 0) in another group
        for _ in range(3):
            groups.add(1, (0, 1), "all", {"m": 1.0})
            groups.add(1, (0, 1), "stratum.riichi", {"m": 1.0})
        groups.add(1, (0, 1), "all", {"m": 0.0})
        groups.add(1, (0, 1), "stratum.open", {"m": 0.0})
        # episode 2: one row (value 0)
        groups.add(1, (0, 2), "all", {"m": 0.0})
        groups.add(1, (0, 2), "stratum.open", {"m": 0.0})
        per = groups.per_seed()
        self.assertAlmostEqual(accuracy.episode_macro(per["all"], [1], "m"), 0.375)
        self.assertEqual(accuracy.episode_macro(per["stratum.riichi"], [1], "m"), 1.0)
        self.assertEqual(accuracy.episode_macro(per["stratum.open"], [1], "m"), 0.0)

    def test_rates_weight_each_episode_once(self):
        rows = list(accuracy.build_rows(labelled(), MODEL))
        weights = accuracy.episode_weights(rows)
        # A, B, C share one kyoku, so each opponent seat is one episode of 3 rows
        self.assertEqual(weights, [1 / 3] * 9)
        fit = accuracy.RateFit()
        fit.add(rows)
        self.assertAlmostEqual(fit.weight, 3.0)

    def test_bootstrap_is_paired_and_deterministic(self):
        per = {"g": {1: {"m": [1.0, 1]}, 2: {"m": [3.0, 1]}}}
        first = accuracy.bootstrap(per, [1, 2], [("g", "m")])
        self.assertEqual(first, accuracy.bootstrap(per, [1, 2], [("g", "m")]))
        self.assertEqual(first["g/m"]["point"], 2.0)
        self.assertEqual(first["g/m"]["low_2.5"], 1.0)
        self.assertEqual(first["g/m"]["high_97.5"], 3.0)


class PopulationTest(unittest.TestCase):
    def write(self, root, seed, splits, producer_policy="MinimalPolicy"):
        root.mkdir()
        write_source(
            root,
            with_seed(all_decisions(), seed),
            with_seed(all_facts(), seed),
            splits=splits,
        )
        if producer_policy != "MinimalPolicy":
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["producer"]["policy"] = producer_policy
            (root / "manifest.json").write_text(canonical_json_text(manifest))

    def test_chunks_must_cover_the_registered_ranges_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory, "a"), Path(directory, "b")
            self.write(a, SEED, {"train": [SEED], "valid": [], "test": []})
            self.write(b, SEED + 1, {"train": [], "valid": [], "test": [SEED + 1]})
            expected = {"train": [SEED], "valid": [], "eval": [SEED + 1]}
            accuracy.check_population([a, b], expected)
            for bad in (
                {"train": [SEED], "valid": [], "eval": [SEED + 1, SEED + 2]},
                {"train": [SEED, SEED + 1], "valid": [], "eval": [SEED + 1]},
            ):
                with self.assertRaises(accuracy.HandBeliefAccuracyError):
                    accuracy.check_population([a, b], bad)
            with self.assertRaises(accuracy.HandBeliefAccuracyError):
                accuracy.check_population([a, a], expected | {"train": [SEED, SEED]})

    def test_end_to_end_reports_counts_and_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory, "a"), Path(directory, "b")
            self.write(a, SEED, {"train": [SEED], "valid": [], "test": []})
            self.write(b, SEED + 1, {"train": [], "valid": [], "test": [SEED + 1]})
            document = accuracy.evaluate_population([a, b], preset_for(), MODEL)
        counts = document["counts"]
        self.assertEqual(counts["eval"]["all"]["rows"], 9)
        self.assertEqual(counts["eval"]["all"]["245_scope_rows"], 2)
        self.assertEqual(counts["eval"]["stratum.riichi"]["rows"], 2)
        # train / valid も同じ件数の形で報告する（validはfit・指標に使わない）
        self.assertEqual(counts["train"]["all"]["rows"], 9)
        # validに行がなくても、全体と3層の全件数を0で明示する
        self.assertEqual(set(counts["valid"]), set(accuracy.COUNT_GROUPS))
        self.assertTrue(
            all(v == 0 for group in counts["valid"].values() for v in group.values())
        )
        self.assertEqual(counts["valid"]["all"], counts["valid"]["stratum.riichi"])
        # 正例0のchannelも件数0で出す（riichi層に国士はない）
        riichi = counts["eval"]["stratum.riichi"]
        for unit in ("rows", "episodes", "hanchan"):
            self.assertEqual(riichi[f"kokushi.positive_{unit}"], 0)
        self.assertEqual(set(document["judgement"]), set(accuracy.COUNT_GROUPS))
        # A: 両面・嵌張、B: 双碰・辺張・国士、C: 単騎の6行が聴牌
        self.assertEqual(counts["eval"]["all"]["wait.positive_rows"], 6)
        self.assertIn("all.tenpai/wait.rate.log_loss.all", document["intervals"])
        self.assertIn("all.not_tenpai", document["episode_macro"])
        # 1半荘しかないので、正例がある表は保留、ない表は推定不能
        self.assertEqual(document["judgement"]["all"]["wait"], "held")
        self.assertEqual(
            document["judgement"]["stratum.riichi"]["kokushi"], "not_estimable"
        )
        interval = document["intervals"]["245_scope/wait.245_minus_rate.log_loss.all"]
        self.assertEqual(interval["low_2.5"], interval["point"])
        self.assertIn("all/channel.kokushi.log_loss.valid_slots", document["intervals"])

    def test_producers_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory, "a"), Path(directory, "b")
            self.write(a, SEED, {"train": [SEED], "valid": [], "test": []})
            self.write(
                b,
                SEED + 1,
                {"train": [], "valid": [], "test": [SEED + 1]},
                producer_policy="Other",
            )
            with self.assertRaises(accuracy.HandBeliefAccuracyError):
                accuracy.check_population(
                    [a, b], {"train": [SEED], "valid": [], "eval": [SEED + 1]}
                )


class PresetTest(unittest.TestCase):
    def test_preset_257_keeps_the_registered_values(self):
        preset = accuracy.PRESET_257
        self.assertEqual(preset.expected()["train"], tuple(range(933000, 933160)))
        self.assertEqual(preset.expected()["valid"], tuple(range(933160, 933240)))
        self.assertEqual(preset.expected()["eval"], tuple(range(933240, 933400)))
        self.assertEqual(preset.bootstrap_resamples, 2000)
        self.assertEqual(preset.bootstrap_seed, 257)
        self.assertEqual(preset.jeffreys, 0.5)
        self.assertEqual(preset.clip_epsilon, 1e-6)
        self.assertEqual((preset.hold_min_hanchan, preset.hold_min_episodes), (50, 100))

    def test_the_hash_follows_every_field(self):
        base = preset_for()
        self.assertEqual(base.sha256(), preset_for().sha256())
        for changes in (
            {"preset_id": "other"},
            {"version": 2},
            {"purpose": "formal-test"},
            {"train": (str(SEED), str(SEED + 5))},
            {"eval": (str(SEED + 2),)},
            {"bootstrap_resamples": 51},
            {"bootstrap_seed": 1},
            {"jeffreys": 1.0},
            {"hold_min_hanchan": 3},
            {"hold_min_episodes": 3},
        ):
            self.assertNotEqual(base.sha256(), preset_for(**changes).sha256(), changes)

    def test_round_trip_and_strict_fields(self):
        preset = preset_for()
        self.assertEqual(accuracy.Preset.from_dict(preset.to_dict()), preset)
        extra = preset.to_dict() | {"unknown": 1}
        missing = {k: v for k, v in preset.to_dict().items() if k != "jeffreys"}
        wrong_schema = preset.to_dict() | {"schema": "other"}
        for bad in (extra, missing, wrong_schema, []):
            with self.assertRaises(accuracy.HandBeliefAccuracyError):
                accuracy.Preset.from_dict(bad)

    def test_invalid_presets_are_rejected(self):
        for changes in (
            {"purpose": "strength-claim"},
            {"train": (str(SEED),), "eval": (str(SEED),)},  # overlap
            {"valid": (f"{SEED}..{SEED + 1}",), "eval": (str(SEED + 1),)},
            {"train": ("abc",)},
            {"eval": ()},
            {"bootstrap_resamples": 0},
            {"jeffreys": 0},
            {"hold_min_episodes": 0},
            {"clip_epsilon": 1e-3},  # protocol invariantはpresetで変えられない
        ):
            with self.assertRaises(accuracy.HandBeliefAccuracyError, msg=changes):
                preset_for(**changes)

    def test_loading_requires_the_registered_hash(self):
        preset = accuracy.PRESET_257
        self.assertIs(accuracy.load_preset(preset.preset_id, preset.sha256()), preset)
        with self.assertRaises(accuracy.HandBeliefAccuracyError):
            accuracy.load_preset(preset.preset_id, "0" * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "preset.json")
            custom = preset_for()
            path.write_text(canonical_json_text(custom.to_dict()))
            self.assertEqual(accuracy.load_preset(str(path), custom.sha256()), custom)
            with self.assertRaises(accuracy.HandBeliefAccuracyError):
                accuracy.load_preset(str(path), preset.sha256())
            with self.assertRaises(accuracy.HandBeliefAccuracyError):
                accuracy.load_preset(str(Path(directory, "missing")), "0" * 64)


class RegistrationTest(unittest.TestCase):
    FIELDS = dict.fromkeys(accuracy.REGISTRATION_FIELDS)

    def test_a_formal_test_must_record_its_reservation(self):
        formal = preset_for(purpose="formal-test")
        with self.assertRaises(accuracy.HandBeliefAccuracyError):
            accuracy.check_registration(formal, self.FIELDS)
        full = {name: "x" for name in accuracy.REGISTRATION_FIELDS}
        self.assertEqual(accuracy.check_registration(formal, full), full)
        for name in accuracy.REGISTRATION_FIELDS:
            with self.assertRaises(accuracy.HandBeliefAccuracyError):
                accuracy.check_registration(formal, full | {name: ""})

    def test_a_development_baseline_may_leave_them_empty(self):
        self.assertEqual(
            accuracy.check_registration(preset_for(), self.FIELDS), self.FIELDS
        )

    def test_unknown_registration_fields_are_rejected(self):
        with self.assertRaises(accuracy.HandBeliefAccuracyError):
            accuracy.check_registration(preset_for(), self.FIELDS | {"extra": "x"})


class ConfiguredEvaluationTest(unittest.TestCase):
    def sources(self, directory):
        a, b = Path(directory, "a"), Path(directory, "b")
        if a.exists():
            return [a, b]
        PopulationTest.write(self, a, SEED, {"train": [SEED], "valid": [], "test": []})
        PopulationTest.write(
            self, b, SEED + 1, {"train": [], "valid": [], "test": [SEED + 1]}
        )
        return [a, b]

    def run_with(self, preset=None, estimator=MODEL, registration=None):
        with tempfile.TemporaryDirectory() as directory:
            return accuracy.evaluate_population(
                self.sources(directory), preset or preset_for(), estimator, registration
            )

    def test_another_estimator_runs_without_changing_the_evaluation_code(self):
        document = self.run_with(estimator=AltEstimator())
        self.assertEqual(document["estimator"], {"name": "alt", "kind": "constant"})
        counts = document["counts"]["eval"]["all"]
        self.assertGreater(counts["alt_scope_rows"], 0)
        self.assertNotIn("245_scope_rows", counts)
        self.assertIn("alt_scope", document["episode_macro"])
        interval = "alt_scope/wait.alt_minus_rate.log_loss.all"
        self.assertIn(interval, document["intervals"])
        self.assertNotIn(
            "245_scope/wait.245_minus_rate.log_loss.all", document["intervals"]
        )

    def test_rows_outside_the_estimators_scope_stay_unprovided(self):
        rows = list(accuracy.build_rows(labelled(), AltEstimator()))
        self.assertTrue(any(row.in_scope for row in rows))
        self.assertTrue(any(not row.in_scope for row in rows))
        for row in rows:
            self.assertEqual(row.scoped_wait is None, not row.in_scope)

    def test_the_result_records_the_preset_estimator_and_registration(self):
        preset = preset_for(purpose="formal-test")
        registration = {name: name.upper() for name in accuracy.REGISTRATION_FIELDS}
        document = self.run_with(preset, registration=registration)
        self.assertEqual(document["schema"], accuracy.RESULT_SCHEMA)
        self.assertEqual(document["preset"]["sha256"], preset.sha256())
        self.assertEqual(document["purpose"], "formal-test")
        self.assertEqual(document["registration"], registration)
        self.assertEqual(document["estimator"], MODEL.identity())
        self.assertEqual(document["estimator"]["selection_sha256"], "0" * 64)
        self.assertEqual(
            set(document["population"]["coverage_sha256"]),
            {str(p) for p in map(Path, document["population"]["manifest_sha256"])},
        )

    def test_a_formal_test_without_registration_runs_nothing(self):
        with self.assertRaises(accuracy.HandBeliefAccuracyError):
            self.run_with(preset_for(purpose="formal-test"))

    def test_preset_values_reach_the_measurement(self):
        base = self.run_with()
        self.assertEqual(base["bootstrap"], {"resamples": 50, "seed": 257})
        other = self.run_with(
            preset_for(bootstrap_resamples=60, bootstrap_seed=7, jeffreys=1.0)
        )
        self.assertEqual(other["bootstrap"], {"resamples": 60, "seed": 7})
        self.assertNotEqual(
            base["rates"]["probabilities"], other["rates"]["probabilities"]
        )
        strict = self.run_with(preset_for(hold_min_hanchan=1, hold_min_episodes=1))
        self.assertEqual(strict["judgement"]["all"]["wait"], "reported")
        self.assertEqual(
            strict["hold_rule"],
            {"min_positive_hanchan": 1, "min_positive_episodes": 1},
        )

    def test_the_invariants_hold_for_every_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = self.sources(directory)
            for bad in (
                preset_for(eval=(str(SEED + 1), str(SEED + 2))),
                preset_for(train=(str(SEED), str(SEED + 1)), eval=(str(SEED + 3),)),
            ):
                with self.assertRaises(accuracy.HandBeliefAccuracyError):
                    accuracy.evaluate_population(sources, bad, MODEL)

    def test_estimator_names_must_be_plain(self):
        class Bad(AltEstimator):
            name = "a.b"

        with self.assertRaises(accuracy.HandBeliefAccuracyError):
            self.run_with(estimator=Bad())


class ReproductionTest(unittest.TestCase):
    def v1(self, document):
        """記録済みの#257 result（v1）の形に直す。"""
        recorded = json.loads(canonical_json_text(document))
        recorded["schema"] = "lisjong-hand-belief-accuracy-result-v1"
        recorded["selection_245_sha256"] = recorded["estimator"]["selection_sha256"]
        for key in ("preset", "purpose", "estimator", "registration"):
            del recorded[key]
        recorded["population"] = {
            "producer": recorded["population"]["producer"],
            "manifest_sha256": {
                f"/elsewhere/{i}": value
                for i, value in enumerate(
                    recorded["population"]["manifest_sha256"].values()
                )
            },
        }
        return recorded

    def document(self):
        return ConfiguredEvaluationTest().run_with()

    def test_the_same_measurement_reproduces_a_v1_record(self):
        document = self.document()
        new = json.loads(canonical_json_text(document))
        self.assertEqual(accuracy.reproduction_differences(self.v1(document), new), [])
        self.assertEqual(accuracy.reproduction_differences(new, new), [])

    def test_a_changed_measurement_is_reported(self):
        document = self.document()
        recorded = self.v1(document)
        recorded["intervals"]["all/expected_count.mse.all"]["point"] += 1e-9
        recorded["counts"]["eval"]["all"]["rows"] += 1
        recorded["selection_245_sha256"] = "1" * 64
        new = json.loads(canonical_json_text(document))
        self.assertEqual(
            accuracy.reproduction_differences(recorded, new),
            ["counts", "intervals", "selection_sha256"],
        )


class CommandLineTest(unittest.TestCase):
    def run_main(self, directory, *extra):
        helper = ConfiguredEvaluationTest()
        sources = helper.sources(directory)
        preset = preset_for()
        path = Path(directory, "preset.json")
        path.write_text(canonical_json_text(preset.to_dict()))
        output = Path(directory, f"result{len(extra)}.json")
        arguments = [
            "--preset", str(path), "--preset-sha256", preset.sha256(),
            "--selection-245", "unused", "--selection-245-sha256", "0" * 64,
            "--output", str(output), *extra, *map(str, sources),
        ]  # fmt: skip
        with (
            mock.patch.object(accuracy, "load_245_estimator", return_value=MODEL),
            mock.patch("sys.stdout"),
            mock.patch("sys.stderr"),
        ):
            code = accuracy.main(arguments)
        return code, json.loads(output.read_text())

    def test_the_result_records_the_execution_and_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            code, first = self.run_main(directory)
            self.assertEqual(code, 0)
            self.assertEqual(first["preset"]["sha256"], preset_for().sha256())
            self.assertGreater(first["execution"]["max_rss_kib"], 0)
            recorded = Path(directory, "recorded.json")
            recorded.write_text(canonical_json_text(first))
            code, again = self.run_main(directory, "--reproduce-of", str(recorded))
            self.assertEqual(code, 0)
            self.assertEqual(again["reproduction"]["differing"], [])
            tampered = json.loads(recorded.read_text())
            tampered["counts"]["eval"]["all"]["rows"] += 1
            recorded.write_text(canonical_json_text(tampered))
            code, third = self.run_main(
                directory, "--reproduce-of", str(recorded), "--evaluator-revision", "r"
            )
            self.assertEqual(code, 1)
            self.assertEqual(third["reproduction"]["differing"], ["counts"])

    def test_a_wrong_preset_hash_stops_before_reading_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "preset.json")
            path.write_text(canonical_json_text(preset_for().to_dict()))
            with (
                mock.patch.object(accuracy, "read_labelled_source") as read,
                self.assertRaises(accuracy.HandBeliefAccuracyError),
            ):
                accuracy.main(
                    [
                        "--preset", str(path), "--preset-sha256", "0" * 64,
                        "--selection-245", "unused", "--selection-245-sha256", "0",
                        "--output", str(Path(directory, "out.json")), directory,
                    ]
                )  # fmt: skip
            read.assert_not_called()


class SupportTest(unittest.TestCase):
    def test_strata_without_rows_are_reported_as_zero(self):
        support = accuracy.Support()
        for row in accuracy.build_rows(labelled(), MODEL):
            if row.stratum == "closed_non_riichi" and not any(
                accuracy.wait_labels(row)
            ):
                support.add(row)
        counts = support.counts()
        self.assertEqual(set(counts), set(accuracy.COUNT_GROUPS))
        self.assertEqual(counts["stratum.riichi"]["rows"], 0)
        self.assertEqual(counts["stratum.riichi"]["wait.positive_hanchan"], 0)
        self.assertGreater(counts["all"]["rows"], 0)
        self.assertEqual(counts["all"]["kokushi.positive_rows"], 0)
        self.assertEqual(accuracy.judgement(counts["all"], "kokushi"), "not_estimable")


class RateTest(unittest.TestCase):
    def test_channels_predict_zero_where_they_cannot_occur(self):
        fit = accuracy.RateFit()
        fit.add(list(accuracy.build_rows(labelled(), MODEL)))
        rates = fit.probabilities()
        for name, _, valid in accuracy.CHANNELS:
            for index, p in enumerate(rates[name]):
                self.assertEqual(p == 0.0, index not in valid, (name, index))
        self.assertTrue(all(p > 0.0 for p in rates["wait"]))


class JudgementTest(unittest.TestCase):
    def test_thresholds(self):
        zero = accuracy.Support().counts()["all"]
        enough = zero | {
            "wait.positive_hanchan": accuracy.HOLD_MIN_HANCHAN,
            "wait.positive_episodes": accuracy.HOLD_MIN_EPISODES,
        }
        self.assertEqual(accuracy.judgement(enough, "wait"), "reported")
        for name in ("wait.positive_hanchan", "wait.positive_episodes"):
            short = enough | {name: enough[name] - 1}
            self.assertEqual(accuracy.judgement(short, "wait"), "held")
        self.assertEqual(accuracy.judgement(zero, "wait"), "not_estimable")


class SelectionTest(unittest.TestCase):
    def test_the_selection_must_match_the_registered_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "selection.json")
            path.write_text("{}")
            with self.assertRaises(accuracy.HandBeliefAccuracyError):
                accuracy.load_245_model(path, hashlib.sha256(b"other").hexdigest())


if __name__ == "__main__":
    unittest.main()
