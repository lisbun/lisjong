import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

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

MODEL = LogisticWaitModel(weights=(("bias", -3.0),))


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
        provided = [(row.episode[1], row.in_245_scope) for row in rows]
        self.assertEqual(
            [seat for seat, scope in provided if scope], [1, 1]
        )  # A and C seat 1
        for row in rows:
            self.assertEqual(row.wait_245 is None, not row.in_245_scope)

    def test_fewer_than_two_candidate_types_is_out_of_scope(self):
        legal = (DiscardAction(actor=Seat(0), tile=tiles("2p")[0], tsumogiri=False),)
        decision = make_decision(20, A_OWN, A_PLAYERS, legal=legal)
        rows = list(
            accuracy.build_rows(
                label_decisions([decision], [make_facts(20, A_HANDS)]), MODEL
            )
        )
        self.assertFalse(any(row.in_245_scope for row in rows))


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
            document = accuracy.evaluate_population(
                [a, b], {"train": [SEED], "valid": [], "eval": [SEED + 1]}, MODEL
            )
        self.assertEqual(document["counts"]["all"]["rows"], 9)
        self.assertEqual(document["counts"]["all"]["245_scope_rows"], 2)
        self.assertEqual(document["counts"]["stratum.riichi"]["rows"], 2)
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


class SelectionTest(unittest.TestCase):
    def test_the_selection_must_match_the_registered_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "selection.json")
            path.write_text("{}")
            with self.assertRaises(accuracy.HandBeliefAccuracyError):
                accuracy.load_245_model(path, hashlib.sha256(b"other").hexdigest())


if __name__ == "__main__":
    unittest.main()
