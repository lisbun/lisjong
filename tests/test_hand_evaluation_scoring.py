"""Issue #263: AI側の役・符・点数計算のPython binding・engine一致・境界を固定する。

native拡張の有無に依存しないtest（import境界・利用不可エラー・入力型）は常に
実行する。計算結果のtestは`_lisjong_native`がimportできる環境だけで実行し、
`LISJONG_REQUIRE_NATIVE=1`（native CI job）ではskipせずfailさせる。

engine一致は`tools/generate_scoring_engine_fixture.py`が固定revisionの
lisjong-engineから生成したfixtureで確認し、lisjongのtestはengineをimportしない。
"""

import ast
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import unittest

from lisjong.hand_evaluation import scoring
from lisjong.hand_evaluation.scoring import (
    PROJECT_STANDARD_SCORING_RULES,
    URA_DORA_EXCLUDED,
    EvaluationStatus,
    FuReason,
    RiichiStatus,
    ScoringMeld,
    ScoringRules,
    UraDoraIndicators,
    WaitType,
    WinContext,
    WinMethod,
    WinningHand,
    WinningShape,
    WinSituation,
    Yaku,
    evaluate_win,
)
from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

try:
    import _lisjong_native
except ImportError:
    _lisjong_native = None

_REQUIRE_NATIVE = os.environ.get("LISJONG_REQUIRE_NATIVE") == "1"
_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _REPOSITORY_ROOT / "tests" / "fixtures" / "scoring_engine_fixture.json"
_ENGINE_REVISION = "96b9796c76ef5db8f3968f689a1ca6f3dfc9aa3b"

_SUITS = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}
_WINDS = tuple(Wind)

# fixtureのRuleSet名に対応するlisjongのルール。engineは喰いタン・赤ドラを
# RuleSetで切り替えないため、どの名前でもTrueである。
_FIXTURE_RULES = {
    "project_standard": PROJECT_STANDARD_SCORING_RULES,
    "mahjong_soul": dataclasses.replace(
        PROJECT_STANDARD_SCORING_RULES,
        double_yakuman_variants=frozenset(
            {
                Yaku.SUUANKOU_TANKI,
                Yaku.KOKUSHI_MUSOU_13_WAIT,
                Yaku.DAISUUSHII,
                Yaku.JUNSEI_CHUUREN_POUTOU,
            }
        ),
    ),
    "rounded_mangan": dataclasses.replace(
        PROJECT_STANDARD_SCORING_RULES, rounded_mangan_enabled=True
    ),
    "single_yakuman_no_counted": dataclasses.replace(
        PROJECT_STANDARD_SCORING_RULES,
        counted_yakuman_enabled=False,
        multiple_yakuman_enabled=False,
    ),
    "double_wind_pair_2": dataclasses.replace(
        PROJECT_STANDARD_SCORING_RULES, double_wind_pair_fu=2
    ),
}


def tiles(notation: str) -> tuple[Tile, ...]:
    """`"123m0p7z"`形式（0は赤5）を牌の列へ変換する。"""
    result = []
    pending = []
    for character in notation.replace(" ", ""):
        if character.isdigit():
            pending.append(int(character))
            continue
        category = _SUITS[character]
        for digit in pending:
            result.append(Tile(TileType(category, digit or 5), is_red=digit == 0))
        pending = []
    if pending:
        raise ValueError(f"digits without suit in {notation!r}")
    return tuple(result)


def tile(notation: str) -> Tile:
    (value,) = tiles(notation)
    return value


def hand(concealed: str, winning: str, *melds: tuple[MeldKind, str]) -> WinningHand:
    return WinningHand(
        concealed_tiles=tiles(concealed),
        melds=tuple(ScoringMeld(kind, tiles(notation)) for kind, notation in melds),
        winning_tile=tile(winning),
    )


def context(
    method: WinMethod = WinMethod.RON,
    *,
    seat_wind: Wind = Wind.SOUTH,
    riichi: RiichiStatus = RiichiStatus.NONE,
    situation: WinSituation = WinSituation.NORMAL,
    dora: str = "",
    ura=None,
    ippatsu: bool = False,
) -> WinContext:
    return WinContext(
        method=method,
        seat_wind=seat_wind,
        prevailing_wind=Wind.EAST,
        riichi=riichi,
        is_ippatsu=ippatsu,
        situation=situation,
        dora_indicators=tiles(dora),
        ura_dora=UraDoraIndicators(()) if ura is None else ura,
    )


def _require_native(test: unittest.TestCase) -> None:
    if _lisjong_native is not None:
        return
    if _REQUIRE_NATIVE:
        test.fail("LISJONG_REQUIRE_NATIVE=1 but _lisjong_native is not importable")
    test.skipTest("opt-in native extension _lisjong_native is not installed")


def _run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPOSITORY_ROOT,
        timeout=120,
    )


class NativeBoundaryTest(unittest.TestCase):
    """native拡張がなくてもimportでき、呼び出しだけが明確に失敗する。"""

    def test_missing_native_fails_only_on_call(self) -> None:
        code = (
            "import sys\n"
            "sys.modules['_lisjong_native'] = None\n"
            "import lisjong, lisjong.hand_evaluation\n"
            "from lisjong.hand_evaluation import scoring, calculate_shanten\n"
            "from lisjong.policy_contract.tile import Tile, TileCategory, TileType\n"
            "t = Tile(TileType(TileCategory.MANZU, 1))\n"
            "hand = scoring.WinningHand((t,) * 13, (), t)\n"
            "ctx = scoring.WinContext(scoring.WinMethod.RON, "
            "scoring.Wind.EAST, scoring.Wind.EAST, scoring.RiichiStatus.NONE, "
            "False, scoring.WinSituation.NORMAL, (), scoring.URA_DORA_EXCLUDED)\n"
            "try:\n"
            "    scoring.evaluate_win(hand, ctx)\n"
            "except scoring.ScoringBackendUnavailableError as error:\n"
            "    print('unavailable:', error)\n"
        )
        completed = _run_python(code)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("unavailable:", completed.stdout)
        self.assertIn("_lisjong_native", completed.stdout)
        self.assertIn("no Python fallback", completed.stdout)

    def test_native_without_scoring_entry_fails_closed(self) -> None:
        code = (
            "import sys, types\n"
            "sys.modules['_lisjong_native'] = types.ModuleType('_lisjong_native')\n"
            "from lisjong.hand_evaluation import scoring\n"
            "try:\n"
            "    scoring._load_native()\n"
            "except scoring.ScoringBackendUnavailableError as error:\n"
            "    print('unavailable:', error)\n"
        )
        completed = _run_python(code)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("SCORING_API_VERSION 1, got None", completed.stdout)

    def test_module_has_no_engine_dependency_and_no_python_calculator(self) -> None:
        tree = ast.parse(pathlib.Path(scoring.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertFalse(any(name.startswith("lisjong_engine") for name in imported))
        self.assertIn("_lisjong_native", imported)

    def test_ura_dora_must_be_explicit(self) -> None:
        with self.assertRaises(TypeError):
            context(ura=())
        with self.assertRaises(TypeError):
            WinContext(
                WinMethod.RON,
                Wind.EAST,
                Wind.EAST,
                RiichiStatus.NONE,
                False,
                WinSituation.NORMAL,
                (),
                None,
            )

    def test_rules_validate_types(self) -> None:
        with self.assertRaises(ValueError):
            dataclasses.replace(
                PROJECT_STANDARD_SCORING_RULES,
                double_yakuman_variants=frozenset({Yaku.TENHOU}),
            )
        with self.assertRaises(ValueError):
            dataclasses.replace(PROJECT_STANDARD_SCORING_RULES, double_wind_pair_fu=3)
        with self.assertRaises(TypeError):
            dataclasses.replace(PROJECT_STANDARD_SCORING_RULES, kuitan_enabled=1)

    def test_standard_rules_match_engine_project_standard(self) -> None:
        self.assertEqual(
            PROJECT_STANDARD_SCORING_RULES,
            ScoringRules(
                kuitan_enabled=True,
                red_dora_enabled=True,
                rounded_mangan_enabled=False,
                counted_yakuman_enabled=True,
                multiple_yakuman_enabled=True,
                double_yakuman_variants=frozenset(),
                double_wind_pair_fu=4,
            ),
        )


class BoundaryFixtureTest(unittest.TestCase):
    """手計算で求めた独立の境界fixture（engine一致とは別の根拠）。"""

    def setUp(self) -> None:
        _require_native(self)

    def score(self, winning_hand, win_context, rules=PROJECT_STANDARD_SCORING_RULES):
        evaluation = evaluate_win(winning_hand, win_context, rules)
        self.assertIs(evaluation.status, EvaluationStatus.SCORED)
        return evaluation.score

    def test_seven_pairs_25_fu(self) -> None:
        score = self.score(hand("1133m5577p2266s7z", "7z"), context())
        self.assertIs(score.shape, WinningShape.SEVEN_PAIRS)
        self.assertEqual(score.yaku_set, {Yaku.CHIITOITSU})
        self.assertEqual(score.fu, 25)
        self.assertEqual(score.ron_payment, 1600)

    def test_pinfu_tsumo_20_fu(self) -> None:
        score = self.score(hand("23m567p345s678s22p", "1m"), context(WinMethod.TSUMO))
        self.assertEqual(score.yaku_set, {Yaku.MENZEN_TSUMO, Yaku.PINFU})
        self.assertEqual(score.fu, 20)
        self.assertEqual(score.fu_components, ((FuReason.BASE, 20),))
        self.assertEqual(
            (score.tsumo_dealer_payment, score.tsumo_non_dealer_payment), (700, 400)
        )

    def test_menzen_ron_adds_10_fu(self) -> None:
        score = self.score(
            hand("23m567p345s678s22p", "1m"), context(riichi=RiichiStatus.RIICHI)
        )
        self.assertEqual(
            score.fu_components, ((FuReason.BASE, 20), (FuReason.MENZEN_RON, 10))
        )
        self.assertEqual(score.ron_payment, 2000)

    def test_ron_assignment_tanki_versus_ryanmen(self) -> None:
        winning_hand = hand("1123m456p789p234s", "1m")
        score = self.score(winning_hand, context(riichi=RiichiStatus.RIICHI))
        self.assertIs(score.wait_type, WaitType.RYANMEN)
        self.assertEqual(score.yaku_set, {Yaku.RIICHI, Yaku.PINFU})
        self.assertEqual(score.completed_group, 0)
        self.assertEqual(score.ron_payment, 2000)

    def test_ron_completed_triplet_is_open(self) -> None:
        winning_hand = hand("222m333p444s55s66s", "5s")
        score = self.score(winning_hand, context())
        self.assertEqual(score.yaku_set, {Yaku.TANYAO, Yaku.TOITOI, Yaku.SANANKOU})
        self.assertEqual(score.fu, 50)
        completed = score.groups[score.completed_group]
        self.assertTrue(completed.is_completed_by_ron)
        self.assertFalse(completed.is_concealed_for_scoring)
        self.assertIn((FuReason.OPEN_SIMPLE_TRIPLET, 2), score.fu_components)
        tsumo = self.score(winning_hand, context(WinMethod.TSUMO))
        self.assertEqual(tsumo.yaku_set, {Yaku.SUUANKOU})
        self.assertIsNone(tsumo.dora)
        self.assertIsNone(tsumo.fu)

    def test_complete_without_yaku_is_a_normal_result(self) -> None:
        evaluation = evaluate_win(
            hand("123p456p789s2s", "2s", (MeldKind.PON, "999m")), context(dora="1s")
        )
        self.assertIs(evaluation.status, EvaluationStatus.NO_YAKU)
        self.assertTrue(evaluation.is_complete)
        self.assertFalse(evaluation.has_yaku)
        self.assertIsNone(evaluation.score)

    def test_not_complete(self) -> None:
        evaluation = evaluate_win(hand("1235m567p345s678s", "9m"), context())
        self.assertIs(evaluation.status, EvaluationStatus.NOT_COMPLETE)
        self.assertFalse(evaluation.is_complete)

    def test_ryanpeikou_beats_seven_pairs(self) -> None:
        score = self.score(hand("112233m445566p7s", "7s"), context())
        self.assertIs(score.shape, WinningShape.STANDARD)
        self.assertEqual(score.yaku_set, {Yaku.RYANPEIKOU})
        self.assertEqual(score.ron_payment, 5200)

    def test_compound_and_double_yakuman(self) -> None:
        winning_hand = hand("555z666z777z111z2z", "2z")
        self.assertEqual(self.score(winning_hand, context()).ron_payment, 96000)
        single = dataclasses.replace(
            PROJECT_STANDARD_SCORING_RULES, multiple_yakuman_enabled=False
        )
        self.assertEqual(self.score(winning_hand, context(), single).ron_payment, 32000)
        double = dataclasses.replace(
            PROJECT_STANDARD_SCORING_RULES,
            double_yakuman_variants=frozenset({Yaku.SUUANKOU_TANKI}),
        )
        score = self.score(winning_hand, context(), double)
        self.assertEqual(score.yakuman_units, 4)
        self.assertEqual(score.ron_payment, 128000)

    def test_counted_yakuman_switch(self) -> None:
        winning_context = context(
            WinMethod.TSUMO,
            riichi=RiichiStatus.RIICHI,
            dora="8m",
            ura=UraDoraIndicators(tiles("1z")),
        )
        winning_hand = hand("1122334567899m", "9m")
        score = self.score(winning_hand, winning_context)
        self.assertEqual(score.han, 15)
        self.assertEqual(score.winner_points, 32000)
        self.assertIsNotNone(score.fu)
        rules = dataclasses.replace(
            PROJECT_STANDARD_SCORING_RULES, counted_yakuman_enabled=False
        )
        self.assertEqual(
            self.score(winning_hand, winning_context, rules).winner_points, 24000
        )

    def test_rounded_mangan_and_ura_modes(self) -> None:
        winning_hand = hand("234m567p345s66s78s", "6s")

        def riichi_context(ura):
            return context(riichi=RiichiStatus.RIICHI, dora="1m", ura=ura)

        score = self.score(winning_hand, riichi_context(UraDoraIndicators(tiles("1z"))))
        self.assertEqual((score.han, score.fu, score.ron_payment), (4, 30, 7700))
        self.assertEqual(score.dora.ura, 0)
        self.assertFalse(score.ura_dora_excluded)
        rounded = dataclasses.replace(
            PROJECT_STANDARD_SCORING_RULES, rounded_mangan_enabled=True
        )
        self.assertEqual(
            self.score(
                winning_hand, riichi_context(UraDoraIndicators(tiles("1z"))), rounded
            ).ron_payment,
            8000,
        )
        score = self.score(winning_hand, riichi_context(UraDoraIndicators(tiles("5s"))))
        self.assertEqual(score.dora.ura, 3)
        self.assertEqual(score.ron_payment, 12000)

        excluded = self.score(winning_hand, riichi_context(URA_DORA_EXCLUDED))
        self.assertIsNone(excluded.dora.ura)
        self.assertTrue(excluded.ura_dora_excluded)
        self.assertEqual(excluded.han, 4)

    def test_kuisagari_and_kuitan(self) -> None:
        opened = self.score(
            hand("456m789m234p5s", "5s", (MeldKind.CHI, "123m")), context()
        )
        self.assertEqual(
            (opened.yaku_han, opened.fu, opened.ron_payment), (1, 30, 1000)
        )
        closed = self.score(hand("123456789m234p5s", "5s"), context())
        self.assertEqual(
            (closed.yaku_han, closed.fu, closed.ron_payment), (2, 40, 2600)
        )

        kuitan_hand = hand("234m567m345s6s", "6s", (MeldKind.PON, "222p"))
        self.assertEqual(self.score(kuitan_hand, context()).yaku_set, {Yaku.TANYAO})
        no_kuitan = dataclasses.replace(
            PROJECT_STANDARD_SCORING_RULES, kuitan_enabled=False
        )
        self.assertIs(
            evaluate_win(kuitan_hand, context(), no_kuitan).status,
            EvaluationStatus.NO_YAKU,
        )

    def test_red_five_counts(self) -> None:
        score = self.score(hand("234m0p67p345s678s2s", "2s"), context())
        self.assertEqual(score.dora.red, 1)
        self.assertEqual(score.ron_payment, 2600)

    def test_invalid_inputs_raise_value_error(self) -> None:
        invalid = [
            (hand("23m567p345s678s2p", "1m"), context()),
            (hand("1111m567p345s678s2p", "1m"), context()),
            (
                hand("456m789m234p5s", "5s", (MeldKind.CHI, "123m")),
                context(riichi=RiichiStatus.RIICHI),
            ),
            (hand("23m567p345s678s22p", "1m"), context(ippatsu=True)),
            (
                hand("23m567p345s678s22p", "1m"),
                context(situation=WinSituation.HAITEI),
            ),
            (
                hand("23m567p345s678s22p", "1m"),
                context(ura=UraDoraIndicators(tiles("1z"))),
            ),
            (
                hand("23m567p345s678s22p", "1m"),
                context(riichi=RiichiStatus.RIICHI, dora="1z"),
            ),
            (
                hand("456m789m234p5s", "5s", (MeldKind.CHI, "124m")),
                context(),
            ),
            (
                hand("23m567p345s678s22p", "1m"),
                context(WinMethod.TSUMO, situation=WinSituation.TENHOU),
            ),
        ]
        for winning_hand, win_context in invalid:
            with self.subTest(hand=winning_hand, context=win_context):
                with self.assertRaises(ValueError):
                    evaluate_win(winning_hand, win_context)
        no_red = dataclasses.replace(
            PROJECT_STANDARD_SCORING_RULES, red_dora_enabled=False
        )
        with self.assertRaises(ValueError):
            evaluate_win(hand("234m0p67p345s678s2s", "2s"), context(), no_red)


def _fixture_hand(case: dict) -> WinningHand:
    def parse(notation: str) -> tuple[Tile, ...]:
        return tiles(
            "".join(notation[index : index + 2] for index in range(0, len(notation), 2))
        )

    return WinningHand(
        concealed_tiles=parse(case["concealed"]),
        melds=tuple(
            ScoringMeld(MeldKind(kind), parse(notation))
            for kind, notation in case["melds"]
        ),
        winning_tile=tile(case["winning"]),
    )


def _fixture_context(case: dict) -> WinContext:
    riichi = RiichiStatus(case["riichi"])
    return WinContext(
        method=WinMethod(case["method"]),
        seat_wind=_WINDS[case["seat_wind"]],
        prevailing_wind=_WINDS[case["prevailing_wind"]],
        riichi=riichi,
        is_ippatsu=case["ippatsu"],
        situation=WinSituation(case["situation"]),
        dora_indicators=tiles(case["dora"]),
        ura_dora=UraDoraIndicators(tiles(case["ura"])),
    )


def _summary(score) -> dict:
    return {
        "yaku": {
            value.yaku.value: value.han or value.yakuman_units for value in score.yaku
        },
        "han": 0 if score.yakuman_units else score.han,
        "fu": score.fu,
        "yakuman_units": score.yakuman_units,
        "dora": None
        if score.dora is None
        else [score.dora.dora, score.dora.red, score.dora.ura],
        "limit": score.limit.value,
        "ron": score.ron_payment,
        "tsumo_dealer": score.tsumo_dealer_payment,
        "tsumo_non_dealer": score.tsumo_non_dealer_payment,
        "winner_points": score.winner_points,
    }


class EngineFixtureTest(unittest.TestCase):
    """固定revisionのlisjong-engineが出した結果との一致。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_identity(self) -> None:
        self.assertEqual(self.fixture["format_version"], 1)
        self.assertEqual(self.fixture["engine_revision"], _ENGINE_REVISION)
        self.assertEqual(set(self.fixture["rules"]), set(_FIXTURE_RULES))
        statuses = {case["expected"]["status"] for case in self.fixture["cases"]}
        self.assertEqual(statuses, {"scored", "no_yaku", "not_complete"})
        covered = {
            yaku
            for case in self.fixture["cases"]
            for candidate in case["expected"].get("max_candidates", ())
            for yaku in candidate["yaku"]
        }
        self.assertEqual(covered, {yaku.value for yaku in Yaku})

    def test_matches_engine(self) -> None:
        _require_native(self)
        mismatches = []
        for index, case in enumerate(self.fixture["cases"]):
            evaluation = evaluate_win(
                _fixture_hand(case),
                _fixture_context(case),
                _FIXTURE_RULES[case["rules"]],
            )
            expected = case["expected"]
            if evaluation.status.value != expected["status"]:
                mismatches.append((index, evaluation.status, expected["status"]))
                continue
            if evaluation.score is None:
                continue
            summary = _summary(evaluation.score)
            if summary not in expected["max_candidates"]:
                mismatches.append((index, summary, expected["max_candidates"]))
        self.assertEqual(mismatches[:5], [])


if __name__ == "__main__":
    unittest.main()
