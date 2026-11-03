"""P1価値モデル（Issue #366）: source reader・10列encoding・ridge・artifact・E2 / E3。

source blockはArena pon-first-pass-source-v1の形をこのtestで合成する（lisjong_arenaはimportしない）。
局面はChampionが実際に門前からPonを選ぶものだけを使う（鳴き解析の再導出がそれを要求する）。
"""

import hashlib
import json
import random
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from test_placement_aware_speed_call_policy import PASS, _input, _tile

from lisjong.learning import p1_value as core
from lisjong.learning import p1_value_evaluation as evaluation
from lisjong.learning._canonical import canonical_json_line, canonical_json_text
from lisjong.learning._typed_values import action_to_value, policy_input_to_value_v2
from lisjong.policies import PlacementAwareSpeedCallPolicy
from lisjong.policy_contract import DecisionContext
from lisjong.policy_contract.action import PonAction
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.seat import Seat

PLAN = "a" * 64
ALLOCATION = "b" * 64
CALLED_ORDER = 20
CASES = (
    ("1235799m112233z", "9m"),  # Honitsuのスピード鳴き
    ("1235799m112233z", "1z"),  # 場風（東）の役牌Pon
    ("123m456p78s5599s77z", "7z"),  # 三元牌の役牌Pon
    ("12m46p78s5599s1577z", "7z"),  # 向聴の遠い役牌Pon
    ("1235799m112255z", "5z"),
)
RIVERS = ("", "9m", "9m1p", "9m1p2s", "9m1p2s8m")
INDICATORS = ("", "8m", "4z", "0p", "6z")
"""ドラ表示牌（9m・1z・5p（赤表示牌）・7z等をドラにする）。"""


def _pon(tile: str) -> PonAction:
    called = _tile(tile)
    return PonAction(
        actor=Seat.SEAT_0,
        target=Seat.SEAT_2,
        called_tile=called,
        consumed_tiles=(called, called),
    )


def _decision(index: int) -> tuple[DecisionContext, PonAction]:
    """手牌・自分の河・ドラ・残り山・点数を行ごとに変える分岐点（自席E、門前）。"""
    hand, tile = CASES[index % len(CASES)]
    context = _input(
        hand,
        dora_indicators=INDICATORS[(index // 2) % len(INDICATORS)],
        scores=(25000 - 1000 * (index % 4), 25000 + 1000 * (index % 3), 25000, 25000),
    )
    players = list(context.players)
    players[0] = replace(
        players[0],
        discards=tuple(
            Discard(tile=item, tsumogiri=False, order=order, called_by=None)
            for order, item in enumerate(_hand_tiles(RIVERS[index % len(RIVERS)]))
        ),
    )
    players[2] = replace(
        players[2],
        discards=(
            Discard(
                tile=_tile(tile), tsumogiri=False, order=CALLED_ORDER, called_by=None
            ),
        ),
    )
    state = replace(context.round, live_wall_tiles_remaining=20 + (index * 7) % 40)
    context = replace(context, players=tuple(players), round=state)
    pon = _pon(tile)
    return DecisionContext(input=context, legal_actions=(PASS, pon)), pon


def _hand_tiles(spec: str):
    tiles, ranks = [], ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        tiles.extend(_tile(rank + character) for rank in ranks)
        ranks = ""
    return tiles


def _delta(index: int) -> int:
    """残り山と点差に依存する決定的なΔ（100点単位）。"""
    return (20 + (index * 7) % 40) // 10 - 3 * (index % 4) + (index % 5) - 1


def _line(ordinal: int, seed: int, index: int, *, resolution="ESTABLISHED") -> dict:
    decision, pon = _decision(index)
    route = core.derived_route(decision, pon)
    g_r = 10 * (index % 3) - 10
    established = resolution == "ESTABLISHED"
    return {
        "branch_id": f"{seed}-{ordinal}-0",
        "branch_ordinal": ordinal,
        "d": {
            "branch_ordinal": ordinal,
            "invalid": [],
            "pass_applied": True,
            "seat": 0,
            "seed": seed,
            "settlement": {"point_deltas": [100 * (g_r + _delta(index)), 0, 0, 0]},
            "status": "branched",
        }
        if established
        else None,
        "d_repeat": None,
        "decision_index": ordinal,
        "decision_input": {
            "call_analysis": {"route": route},
            "call_analysis_error": None,
            "context": {
                "legal_actions": [
                    action_to_value(action, ValueError, "a")
                    for action in decision.legal_actions
                ],
                "policy_input": policy_input_to_value_v2(
                    decision.input, ValueError, "p"
                ),
            },
            "context_sha256": "c" * 64,
            "pon": action_to_value(pon, ValueError, "pon"),
        },
        "kind": "candidate",
        "prefix_sha256": "d" * 64,
        "r_invalid": [],
        "r_post_sha256": "e" * 64,
        "r_settlement": {"point_deltas": [100 * g_r, 0, 0, 0]},
        "round": {},
        "round_ordinal": 0,
        "seat": 0,
        "seed": seed,
        "selection": {"called_discard_order": CALLED_ORDER, "resolution": resolution},
    }


def write_block(
    root: Path,
    block: str,
    *,
    artifact=None,
    rows=12,
    start=0,
    cancelled=(),
    edit=None,
    **changes,
) -> Path:
    split, seeds = core.BLOCKS[block]
    directory = root / block
    directory.mkdir(parents=True)
    header = {
        "allocation_identity": ALLOCATION,
        "artifact_identity": artifact,
        "block": block,
        "call_analysis_version": "call-value-shadow-v1",
        "kind": "header",
        "plan_digest": PLAN,
        "protocol_version": core.SOURCE_PROTOCOL,
        "schema": core.BRANCHES_SCHEMA,
        "split": split,
        "wrapper_version": "pon-first-pass-branch-wrapper-v1",
    }
    header.update(changes.pop("header", {}))
    lines = [header] + [
        _line(
            ordinal,
            seeds[ordinal % len(seeds)],
            start + ordinal,
            resolution="CANCELLED-RON" if ordinal in cancelled else "ESTABLISHED",
        )
        for ordinal in range(rows)
    ]
    if edit is not None:
        edit(lines)
    (directory / "branches.jsonl").write_text(
        "".join(canonical_json_line(line) for line in lines),
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "block": block,
        "decision": "SOURCE-COMPLETE",
        "invalid": {"candidates": [], "count": 0},
        "plan_digest": PLAN,
        "schema": core.SUMMARY_SCHEMA,
        "split": split,
    }
    summary.update(changes.pop("summary", {}))
    (directory / "summary.json").write_text(
        canonical_json_text(summary), encoding="utf-8"
    )
    # readerが内容を読まないfileも、protocolのfile集合として置く
    for name in ("hanchans.json", "timings.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    (directory / "actions.jsonl").write_text("{}\n", encoding="utf-8")
    resum(directory)
    return directory


def resum(directory: Path) -> None:
    lines = []
    for item in sorted(directory.iterdir()):
        if item.name != "SHA256SUMS":
            digest = hashlib.sha256(item.read_bytes()).hexdigest()
            lines.append(f"{digest}  {item.name}\n")
    (directory / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)

    def tearDown(self) -> None:
        self._dir.cleanup()


class FixtureTest(unittest.TestCase):
    def test_the_champion_pons_from_a_closed_hand_in_every_case(self) -> None:
        for index in range(20):
            decision, pon = _decision(index)
            with self.subTest(index=index):
                self.assertEqual(
                    PlacementAwareSpeedCallPolicy().choose_action(decision), pon
                )


class SourceReaderTest(_Tmp):
    def test_reads_rows_deltas_and_the_cancelled_denominator(self) -> None:
        block = core.read_source_block(
            write_block(self.tmp, "TRAIN-1", rows=10, cancelled=(3, 7))
        )
        self.assertEqual(len(block.rows), 8)
        self.assertEqual(block.ron_cancelled, 2)
        self.assertEqual(block.rows[0].delta, _delta(0))
        self.assertEqual(block.identity()["ron_cancelled"], 2)

    def test_selection_and_d_arm_values_do_not_reach_the_features(self) -> None:
        plain = core.read_source_block(write_block(self.tmp / "a", "TRAIN-1", rows=4))

        def edit(lines):
            for line in lines[1:]:
                line["selection"]["called_discard_order"] = 99
                line["d"]["calls_after_branch"] = [{"kind": "chi"}]

        changed = core.read_source_block(
            write_block(self.tmp / "b", "TRAIN-1", rows=4, edit=edit)
        )
        self.assertEqual(
            [core.encode_row(row) for row in plain.rows],
            [core.encode_row(row) for row in changed.rows],
        )

    def test_rejects_tampered_or_mixed_blocks(self) -> None:
        def tamper_mixed(lines):
            lines[1]["decision_input"]["resolution"] = "ESTABLISHED"

        def tamper_selection(lines):
            lines[1]["selection"]["context"] = None

        def tamper_resolution(lines):
            lines[1]["selection"]["resolution"] = "CANCELLED-PRIORITY-CALL"

        def tamper_pass(lines):
            lines[1]["d"]["pass_applied"] = False

        def tamper_cancel_with_d(lines):
            lines[1]["selection"]["resolution"] = "CANCELLED-RON"

        def tamper_route(lines):
            lines[1]["decision_input"]["call_analysis"]["route"] = "other"

        cases = {
            "mixed decision_input": {"edit": tamper_mixed},
            "mixed selection": {"edit": tamper_selection},
            "resolution": {"edit": tamper_resolution},
            "pass": {"edit": tamper_pass},
            "cancel with d": {"edit": tamper_cancel_with_d},
            "header": {"header": {"protocol_version": "riichi-first-dama-source-v1"}},
            "summary": {"summary": {"decision": "STOP-INVALID"}},
            "train artifact": {"artifact": "f" * 64},
        }
        for name, changes in cases.items():
            directory = write_block(self.tmp / name, "TRAIN-1", rows=3, **changes)
            with self.subTest(case=name), self.assertRaises(core.P1ValueError):
                block = core.read_source_block(directory)
                for row in block.rows:
                    core.encode_row(row)
        directory = write_block(self.tmp / "sums", "TRAIN-1", rows=3)
        (directory / "summary.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(core.P1ValueError):
            core.read_source_block(directory)

    def test_the_protocol_file_set_is_required(self) -> None:
        # SHA256SUMSと実directoryの双方から欠けていても、protocolのfile集合で拒否する
        missing = write_block(self.tmp / "missing", "TRAIN-1", rows=3)
        (missing / "actions.jsonl").unlink()
        resum(missing)
        with self.assertRaisesRegex(core.P1ValueError, "holds exactly"):
            core.read_source_block(missing)
        extra = write_block(self.tmp / "extra", "TRAIN-1", rows=3)
        (extra / "features.jsonl").write_text("{}\n", encoding="utf-8")
        resum(extra)
        with self.assertRaisesRegex(core.P1ValueError, "holds exactly"):
            core.read_source_block(extra)
        self.assertEqual(
            core.SOURCE_FILES,
            {
                "SHA256SUMS",
                "actions.jsonl",
                "branches.jsonl",
                "hanchans.json",
                "summary.json",
                "timings.json",
            },
        )

    def test_read_source_needs_one_allocation_per_split(self) -> None:
        for split, names, artifact in (
            (core.TRAIN, ("TRAIN-1", "TRAIN-2", "TRAIN-3"), None),
            (core.TEST, ("TEST-1", "TEST-2", "TEST-3"), "1" * 64),
        ):
            with self.subTest(split=split):
                root = self.tmp / split
                blocks = [
                    write_block(
                        root,
                        name,
                        rows=3,
                        artifact=artifact,
                        header={"allocation_identity": "f" * 64}
                        if name.endswith("-3")
                        else {},
                    )
                    for name in names
                ]
                with self.assertRaisesRegex(core.P1ValueError, "different allocations"):
                    core.read_source(blocks, split)

    def test_read_source_needs_all_three_blocks_and_one_artifact(self) -> None:
        blocks = [
            write_block(self.tmp, name, rows=3) for name in ("TRAIN-1", "TRAIN-2")
        ]
        with self.assertRaises(core.P1ValueError):
            core.read_source(blocks, core.TRAIN)
        blocks.append(write_block(self.tmp, "TRAIN-3", rows=3))
        self.assertEqual(len(core.read_source(blocks, core.TRAIN)), 3)
        tests = [
            write_block(self.tmp, name, rows=3, artifact=digest)
            for name, digest in (
                ("TEST-1", "1" * 64),
                ("TEST-2", "1" * 64),
                ("TEST-3", "2" * 64),
            )
        ]
        with self.assertRaises(core.P1ValueError):
            core.read_source(tests, core.TEST)

    def test_fixed_blocks(self) -> None:
        self.assertEqual(core.SPLIT_SEEDS[core.TRAIN], tuple(range(556_000, 556_120)))
        self.assertEqual(core.SPLIT_SEEDS[core.TEST], tuple(range(556_200, 556_320)))
        self.assertTrue(all(len(seeds) == 40 for _, seeds in core.BLOCKS.values()))


class EncodingTest(unittest.TestCase):
    def test_column_values(self) -> None:
        # index 0: Honitsuの9m Pon（スピード鳴き）、ドラなし、河なし、点差0
        decision, pon = _decision(0)
        route = core.derived_route(decision, pon)
        self.assertEqual(route, "CallDecisionRoute.SPEED_CALL")
        vector = dict(zip(core.COLUMNS, core.encode(decision, pon, route), strict=True))
        self.assertEqual(vector["route_speed_call"], 1.0)
        self.assertEqual(vector["called_tile_is_yakuhai"], 0.0)
        self.assertEqual(vector["shanten_before_pon"], 2.0)
        self.assertEqual(vector["min_shanten_after_pon"], 1.0)
        self.assertEqual(vector["dora_count"], 0.0)
        self.assertEqual(vector["own_discard_count"], 0.0)
        self.assertEqual(vector["live_wall_tiles_remaining"], 20.0)
        self.assertEqual(vector["opponent_riichi"], 0.0)
        self.assertEqual(vector["is_dealer"], 0.0)
        self.assertEqual(vector["gap_to_top_per_1000"], 0.0)
        # index 2: 7zの役牌Pon（親路）、ドラ表示牌8m（9mがドラ）、河2枚、点差（2000 + 2000）
        decision, pon = _decision(2)
        vector = dict(
            zip(
                core.COLUMNS,
                core.encode(decision, pon, core.derived_route(decision, pon)),
                strict=True,
            )
        )
        self.assertEqual(vector["route_speed_call"], 0.0)
        self.assertEqual(vector["called_tile_is_yakuhai"], 1.0)
        self.assertEqual(vector["own_discard_count"], 2.0)
        self.assertEqual(vector["gap_to_top_per_1000"], 4.0)
        self.assertEqual(vector["dora_count"], 0.0)  # ドラ9mは手にない
        # index 4: 表示牌4z -> ドラ1z（風牌の循環）、手に1zが2枚
        decision, pon = _decision(4)
        vector = dict(
            zip(
                core.COLUMNS,
                core.encode(decision, pon, core.derived_route(decision, pon)),
                strict=True,
            )
        )
        self.assertEqual(vector["dora_count"], 2.0)
        self.assertEqual(vector["gap_to_top_per_1000"], 1.0)

    def test_red_five_counts_as_dora(self) -> None:
        decision, pon = _decision(2)  # 123m456p78s5599s77z、ドラ9m（手にない）
        concealed = list(decision.input.own_hand.concealed_tiles)
        index = concealed.index(_tile("5s"))
        concealed[index] = _tile("0s")
        hand = replace(decision.input.own_hand, concealed_tiles=tuple(concealed))
        changed = replace(decision, input=replace(decision.input, own_hand=hand))
        route = core.derived_route(changed, pon)
        vector = dict(zip(core.COLUMNS, core.encode(changed, pon, route), strict=True))
        self.assertEqual(vector["dora_count"], 1.0)

    def test_red_fives_and_riichi_and_dealer(self) -> None:
        decision, pon = _decision(4)
        players = list(decision.input.players)
        players[1] = replace(players[1], riichi=RiichiState.DECLARED)
        state = replace(decision.input.round, dealer_seat=Seat.SEAT_0)
        changed = replace(
            decision,
            input=replace(decision.input, players=tuple(players), round=state),
        )
        vector = dict(
            zip(
                core.COLUMNS,
                core.encode(changed, pon, core.derived_route(decision, pon)),
                strict=True,
            )
        )
        self.assertEqual(vector["opponent_riichi"], 1.0)
        self.assertEqual(vector["is_dealer"], 1.0)

    def test_ankan_tiles_count_as_dora_and_the_hand_shanten(self) -> None:
        # 門前（暗槓4m）: 手牌10枚 123599m1122z、Wの9mをPon（Honitsuのスピード鳴き）。
        # ドラ表示牌3m・8m -> ドラ4m・9m。暗槓4枚 + 手の99m（消費牌は1回だけ）+ 鳴いた9m = 7
        kan = _tile("4m")
        meld = PublicMeld(
            kind=MeldKind.ANKAN, tiles=(kan,) * 4, from_seat=None, called_tile=None
        )
        context = _input("123599m1122z", own_melds=(meld,), dora_indicators="3m8m")
        players = list(context.players)
        players[2] = replace(
            players[2],
            discards=(
                Discard(
                    tile=_tile("9m"),
                    tsumogiri=False,
                    order=CALLED_ORDER,
                    called_by=None,
                ),
            ),
        )
        pon = _pon("9m")
        decision = DecisionContext(
            input=replace(context, players=tuple(players)), legal_actions=(PASS, pon)
        )
        route = core.derived_route(decision, pon)
        self.assertEqual(route, "CallDecisionRoute.SPEED_CALL")
        vector = dict(zip(core.COLUMNS, core.encode(decision, pon, route), strict=True))
        # Pon前: 123m・99m・11z・22z・5m（暗槓1つ）で1向聴。Pon後: 5mを切って123m・11z・22zの聴牌
        self.assertEqual(vector["shanten_before_pon"], 1.0)
        self.assertEqual(vector["min_shanten_after_pon"], 0.0)
        self.assertEqual(vector["dora_count"], 7.0)

    def test_route_must_re_derive(self) -> None:
        decision, pon = _decision(0)
        with self.assertRaises(core.P1ValueError):
            core.encode(decision, pon, "CallDecisionRoute.PARENT_DECISION")

    def test_fingerprint_is_stable(self) -> None:
        self.assertEqual(len(core.COLUMNS), 10)
        self.assertEqual(core.CONTINUOUS, (2, 3, 4, 5, 6, 9))
        self.assertEqual(core.ENCODING_IDENTITY, "lisjong-p1-value-encoding-v1")


class RidgeTest(unittest.TestCase):
    def test_zero_variance_continuous_column_is_rejected(self) -> None:
        matrix = [
            [0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 40.0, 0.0, 0.0, float(i)] for i in range(5)
        ]
        with self.assertRaises(core.P1ValueError):
            core.ridge(matrix, [1, 2, 3, 4, 5])

    def test_intercept_is_the_mean_when_features_are_constant_in_binary(self) -> None:
        rng = random.Random(7)
        matrix = [
            [0.0, 0.0, *(rng.random() for _ in range(4)), 0.0, 0.0, 0.0, rng.random()]
            for _ in range(30)
        ]
        matrix = [[*row[:6], row[6], *row[7:]] for row in matrix]
        for row in matrix:
            row[6] = rng.random()
        targets = [rng.randint(-50, 50) for _ in range(30)]
        _, intercept, coefficients = core.ridge(matrix, targets)
        self.assertAlmostEqual(intercept, sum(targets) / len(targets), places=9)
        self.assertEqual(len(coefficients), 10)


class ArtifactTest(_Tmp):
    def _train(self, rows=15):
        return core.read_source(
            [
                write_block(self.tmp, name, rows=rows, start=offset)
                for name, offset in (("TRAIN-1", 0), ("TRAIN-2", 50), ("TRAIN-3", 100))
            ],
            core.TRAIN,
        )

    def test_fit_save_load_round_trip(self) -> None:
        document = core.build_artifact(self._train())
        path = core.save_artifact(document, self.tmp / "artifact")
        artifact = core.load_artifact(path)
        self.assertEqual(artifact.identity, document["identity"])
        self.assertEqual(document["diagnostics"]["rows"], 45)
        with self.assertRaises(core.P1ValueError):
            core.save_artifact(document, path)

    def test_tampered_artifacts_are_rejected(self) -> None:
        document = core.build_artifact(self._train())
        path = core.save_artifact(document, self.tmp / "artifact")
        manifest = path / core.MANIFEST_FILENAME
        value = json.loads(manifest.read_text(encoding="utf-8"))
        value["fit"]["lambda"] = 1
        manifest.write_text(canonical_json_text(value), encoding="utf-8")
        with self.assertRaises(core.P1ValueError):
            core.load_artifact(path)

    def test_fit_uses_train_rows_only(self) -> None:
        tests = core.read_source(
            [
                write_block(self.tmp, name, rows=5, artifact="1" * 64)
                for name in ("TEST-1", "TEST-2", "TEST-3")
            ],
            core.TEST,
        )
        with self.assertRaises(core.P1ValueError):
            core.build_artifact(tests)


def _reference_bootstrap(seeds, by_seed):
    rng = random.Random(365)
    values = []
    for _ in range(10_000):
        total = count = 0
        for _ in range(len(seeds)):
            value, size = by_seed[seeds[rng.randrange(len(seeds))]]
            total += value
            count += size
        values.append(total / count)
    values.sort()
    return values[499]


class EvaluationTest(_Tmp):
    def setUp(self) -> None:
        super().setUp()
        train = core.read_source(
            [
                write_block(self.tmp, name, rows=15, start=offset)
                for name, offset in (("TRAIN-1", 0), ("TRAIN-2", 50), ("TRAIN-3", 100))
            ],
            core.TRAIN,
        )
        path = core.save_artifact(core.build_artifact(train), self.tmp / "artifact")
        self.artifact_dir = path
        self.artifact = core.load_artifact(path)

    def _tests(self, *, artifact=None, cancelled=(2,)):
        identity = artifact or self.artifact.identity
        return [
            write_block(
                self.tmp / "test",
                name,
                rows=12,
                start=offset,
                artifact=identity,
                cancelled=cancelled,
            )
            for name, offset in (("TEST-1", 200), ("TEST-2", 250), ("TEST-3", 300))
        ]

    def test_estimators_bootstrap_and_decision(self) -> None:
        blocks = core.read_source(self._tests(), core.TEST)
        document = evaluation.evaluate(self.artifact, blocks)
        rows = core.rows_of(blocks)
        values = evaluation.evaluation_values(self.artifact, rows)
        n = len(values)
        self.assertEqual(n, 33)
        self.assertAlmostEqual(
            document["e2"]["estimate"], sum(item["e2"] for item in values) / n
        )
        e3_sum = sum(item["e3"] for item in values)
        self.assertAlmostEqual(document["e3"]["estimate"], e3_sum / n)
        # 全対象判断あたり: ロン取消（各block 1件）を分母に含める
        self.assertEqual(document["population"]["ron_cancelled"], 3)
        self.assertAlmostEqual(document["e3"]["per_target_decision"], e3_sum / 36)
        seeds = core.SPLIT_SEEDS[core.TEST]
        by_seed = {seed: (0.0, 0) for seed in seeds}
        for item in values:
            total, count = by_seed[item["seed"]]
            by_seed[item["seed"]] = (total + item["e2"], count + 1)
        self.assertAlmostEqual(
            document["e2"]["bootstrap"]["P5"], _reference_bootstrap(seeds, by_seed)
        )
        expected = (
            evaluation.NOT_PREDICTIVE
            if not document["e2"]["bootstrap"]["P5"] > 0
            else (
                evaluation.DECISION_VALUE_SIGNAL
                if document["e3"]["bootstrap"]["P5"] > 0
                else evaluation.PREDICTIVE_ONLY
            )
        )
        self.assertEqual(document["decision"], expected)
        self.assertTrue(document["not_a_strength_result"])

    def _counting_reads(self, *, wait=None, fail=False):
        """``read_source``の呼び出し回数を数える（TESTを読んだかの確認）。"""
        calls = []
        real = evaluation.read_source

        def read(directories, split):
            calls.append(split)
            if wait is not None:
                wait.wait(timeout=30)
            if fail:
                raise RuntimeError("synthetic failure while reading TEST")
            return real(directories, split)

        return calls, mock.patch.object(evaluation, "read_source", read)

    def test_test_is_read_once_and_bound_to_the_artifact(self) -> None:
        document = evaluation.run_evaluation(self.artifact_dir, self._tests())
        destination = evaluation.evaluation_dir(
            self.artifact_dir, self.artifact.identity
        )
        self.assertEqual(destination.parent, self.artifact_dir.resolve().parent)
        self.assertEqual(
            {item.name for item in destination.iterdir()},
            {evaluation.START_FILENAME, evaluation.EVALUATION_FILENAME},
        )
        stored = json.loads((destination / evaluation.EVALUATION_FILENAME).read_text())
        self.assertEqual(stored, document)
        start = json.loads((destination / evaluation.START_FILENAME).read_text())
        self.assertEqual(start["schema"], evaluation.START_SCHEMA)
        self.assertEqual(start["artifact_identity"], self.artifact.identity)
        # 別のTEST directory（出力先の指定はない）でも、TESTを読む前に止まる
        calls, patch = self._counting_reads()
        with patch, self.assertRaisesRegex(core.P1ValueError, "already been started"):
            evaluation.run_evaluation(self.artifact_dir, self._tests_again())
        self.assertEqual(calls, [])
        other = self._tests_other()
        with self.assertRaises(core.P1ValueError):
            evaluation.evaluate(self.artifact, core.read_source(other, core.TEST))

    def test_failed_evaluation_is_not_retried(self) -> None:
        tests = self._tests()
        calls, patch = self._counting_reads(fail=True)
        with patch, self.assertRaises(RuntimeError):
            evaluation.run_evaluation(self.artifact_dir, tests)
        self.assertEqual(calls, ["TEST"])
        destination = evaluation.evaluation_dir(
            self.artifact_dir, self.artifact.identity
        )
        self.assertEqual(
            {item.name for item in destination.iterdir()}, {evaluation.START_FILENAME}
        )
        calls, patch = self._counting_reads()
        with patch, self.assertRaisesRegex(core.P1ValueError, "already been started"):
            evaluation.run_evaluation(self.artifact_dir, tests)
        self.assertEqual(calls, [])

    def test_concurrent_evaluation_stops_before_reading_test(self) -> None:
        release = threading.Event()
        calls, patch = self._counting_reads(wait=release)
        results = []
        tests = self._tests()
        with patch:
            first = threading.Thread(
                target=lambda: results.append(
                    evaluation.run_evaluation(self.artifact_dir, tests)
                )
            )
            first.start()
            try:
                for _ in range(300):
                    if calls:
                        break
                    threading.Event().wait(0.01)
                self.assertEqual(calls, ["TEST"])  # 1つ目は開始記録を取ってTESTを読み中
                with self.assertRaisesRegex(core.P1ValueError, "already been started"):
                    evaluation.run_evaluation(self.artifact_dir, self._tests_again())
                self.assertEqual(calls, ["TEST"])  # 2つ目はTESTを読んでいない
            finally:
                release.set()
                first.join(timeout=60)
        self.assertEqual(len(results), 1)

    def _tests_again(self):
        root = self.tmp / "again"
        return [
            write_block(
                root, name, rows=12, start=offset, artifact=self.artifact.identity
            )
            for name, offset in (("TEST-1", 200), ("TEST-2", 250), ("TEST-3", 300))
        ]

    def _tests_other(self):
        root = self.tmp / "other"
        return [
            write_block(root, name, rows=12, start=offset, artifact="9" * 64)
            for name, offset in (("TEST-1", 200), ("TEST-2", 250), ("TEST-3", 300))
        ]


class BoundaryTest(unittest.TestCase):
    def test_no_ml_framework_or_arena_import(self) -> None:
        import sys

        for name in (
            "lisjong.learning.p1_value",
            "lisjong.learning.p1_value_evaluation",
        ):
            module = sys.modules[name]
            self.assertNotIn("torch", vars(module))
        self.assertNotIn("lisjong_arena", sys.modules)


if __name__ == "__main__":
    unittest.main()
