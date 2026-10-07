"""lisjong-engineの得点評価から役・点数のengine一致fixtureを生成する（Issue #263）。

`lisjong.hand_evaluation.scoring`がlisjong-engineの得点評価層と同じ結果を返すことを
確認するためのfixtureを、固定seedで決定的に生成する。lisjongのtest・runtimeは
engineをimportせず、ここで固定したfixtureだけを読む。

    python -m pip install "lisjong-engine @ git+https://github.com/lisbun/lisjong-engine@<ENGINE_REVISION>"
    python tools/generate_scoring_engine_fixture.py \
        tests/fixtures/scoring_engine_fixture.json

engineのrevisionは`ENGINE_REVISION`で固定し、installされたengineのcheckoutが
その値と一致しない場合は生成を中止する（`--engine-checkout`でgit checkoutを指定）。
同じseed・revisionからは常に同一の出力になる。

各caseは仮定の手牌とcontext・RuleSet名・engineの評価結果を持つ。和了形でなければ
`not_complete`、役が1つもなければ`no_yaku`、そうでなければengineの最高得点候補
（同点をすべて保持）を記録する。fixtureが全役をcoverしない場合は生成を失敗させる。
"""

import argparse
import dataclasses
import json
import pathlib
import random
import subprocess
import sys

import lisjong_engine
from lisjong_engine.dora import DoraIndicators
from lisjong_engine.meld import Ankan, Chi, Daiminkan, Kakan, Pon
from lisjong_engine.rule_presets import MAHJONG_SOUL_RULES, PROJECT_STANDARD_RULES
from lisjong_engine.seat import Seat
from lisjong_engine.tile import STANDARD_TILES, TileCategory
from lisjong_engine.win_context import (
    RiichiStatus,
    WinMethod,
    WinningContext,
    WinOrigin,
)
from lisjong_engine.wind import Wind
from lisjong_engine.winning import find_winning_shapes
from lisjong_engine.winning_score import (
    enumerate_winning_score_candidates,
    select_max_score_candidates,
)
from lisjong_engine.yaku import Yaku

ENGINE_REVISION = "96b9796c76ef5db8f3968f689a1ca6f3dfc9aa3b"
SEED = 263
CASE_COUNT = 2400
FORMAT_VERSION = 1

RULES = {
    "project_standard": PROJECT_STANDARD_RULES,
    "mahjong_soul": MAHJONG_SOUL_RULES,
    "rounded_mangan": dataclasses.replace(
        PROJECT_STANDARD_RULES, rounded_mangan_enabled=True
    ),
    "single_yakuman_no_counted": dataclasses.replace(
        PROJECT_STANDARD_RULES,
        counted_yakuman_enabled=False,
        multiple_yakuman_enabled=False,
    ),
    "double_wind_pair_2": dataclasses.replace(
        PROJECT_STANDARD_RULES, double_wind_pair_fu=2
    ),
}
RULE_WEIGHTS = (6, 1, 1, 1, 1)

_SUFFIX = {
    TileCategory.MANZU: "m",
    TileCategory.PINZU: "p",
    TileCategory.SOUZU: "s",
    TileCategory.HONOR: "z",
}
_GREEN = (19, 20, 21, 23, 25, 32)
_TERMINALS = (0, 8, 9, 17, 18, 26)
_HONORS = tuple(range(27, 34))
_KOKUSHI = _TERMINALS + _HONORS


def notation(tile) -> str:
    rank = 0 if tile.is_red else tile.tile_type.rank
    return f"{rank}{_SUFFIX[tile.tile_type.category]}"


def tiles_notation(tiles) -> str:
    return "".join(notation(tile) for tile in tiles)


class Pool:
    """未使用の物理牌から指定牌種の1枚を取り出す。"""

    def __init__(self, generator: random.Random) -> None:
        self._generator = generator
        self._free = {
            kind: list(STANDARD_TILES[kind * 4 : kind * 4 + 4]) for kind in range(34)
        }

    def available(self, kind: int) -> int:
        return len(self._free[kind])

    def take(self, kind: int):
        copies = self._free[kind]
        if not copies:
            raise _Retry
        return copies.pop(self._generator.randrange(len(copies)))

    def take_any(self):
        kinds = [kind for kind in range(34) if self._free[kind]]
        return self.take(self._generator.choice(kinds))


class _Retry(Exception):
    pass


def _kind_choices(generator: random.Random) -> tuple[str, tuple[int, ...]]:
    mode = generator.choices(
        ("any", "suit", "honor", "terminal", "green", "simple"),
        weights=(10, 5, 3, 2, 1, 3),
    )[0]
    if mode == "suit":
        suit = generator.randrange(3)
        kinds = tuple(range(suit * 9, suit * 9 + 9))
        if generator.random() < 0.5:
            kinds += _HONORS
        return mode, kinds
    if mode == "honor":
        return mode, _HONORS + (_TERMINALS if generator.random() < 0.3 else ())
    if mode == "terminal":
        return mode, _TERMINALS + (_HONORS if generator.random() < 0.5 else ())
    if mode == "green":
        return mode, _GREEN
    if mode == "simple":
        return mode, tuple(kind for kind in range(27) if kind % 9 not in (0, 8))
    return mode, tuple(range(34))


def _group(generator: random.Random, kinds, allow_quad: bool):
    """(kind, start)のgroup。kindはsequence / triplet / quad。"""
    sequence_starts = [kind for kind in kinds if kind < 27 and kind % 9 <= 6]
    roll = generator.random()
    if sequence_starts and roll < 0.55:
        return "sequence", generator.choice(sequence_starts)
    if allow_quad and roll > 0.85:
        return "quad", generator.choice(kinds)
    return "triplet", generator.choice(kinds)


def _template(generator: random.Random):
    """複合役を作りやすいgroupの組。"""
    suit = generator.randrange(3) * 9
    rank = generator.randrange(7)
    return generator.choice(
        (
            [("sequence", suit), ("sequence", suit + 3), ("sequence", suit + 6)],
            [("sequence", rank), ("sequence", 9 + rank), ("sequence", 18 + rank)],
            [("triplet", rank), ("triplet", 9 + rank), ("triplet", 18 + rank)],
            [("triplet", 31), ("triplet", 32)],
            [("triplet", kind) for kind in generator.sample(range(27, 31), 3)],
            [("triplet", kind) for kind in range(27, 31)],
            [("sequence", suit + rank), ("sequence", suit + rank)],
        )
    )


def _standard_hand(generator: random.Random, pool: Pool):
    _, kinds = _kind_choices(generator)
    if generator.random() < 0.05:
        meld_count = 4
        groups = [("quad", generator.choice(kinds)) for _ in range(4)]
    else:
        meld_count = generator.choices((0, 1, 2, 3, 4), weights=(9, 4, 3, 2, 1))[0]
        groups = [_group(generator, kinds, index < meld_count) for index in range(4)]
        if generator.random() < 0.25:
            template = _template(generator)
            for position, group in zip(
                generator.sample(range(4), len(template)), template, strict=True
            ):
                groups[position] = group
    pair = generator.choice(kinds)

    melds = []
    concealed = [pool.take(pair), pool.take(pair)]
    for index, (kind, start) in enumerate(groups):
        if kind == "sequence":
            group_tiles = [pool.take(start + offset) for offset in range(3)]
        else:
            group_tiles = [pool.take(start) for _ in range(4 if kind == "quad" else 3)]
        if index >= meld_count:
            concealed.extend(group_tiles)
            continue
        source = generator.choice(tuple(Seat))
        if kind == "sequence":
            called = generator.randrange(3)
            melds.append(
                Chi(
                    group_tiles[called],
                    tuple(tile for i, tile in enumerate(group_tiles) if i != called),
                    source,
                )
            )
        elif kind == "triplet":
            melds.append(Pon(group_tiles[0], tuple(group_tiles[1:]), source))
        else:
            quad_kind = generator.choice(("ankan", "daiminkan", "kakan"))
            if quad_kind == "ankan":
                melds.append(Ankan(tuple(group_tiles)))
            elif quad_kind == "daiminkan":
                melds.append(Daiminkan(group_tiles[0], tuple(group_tiles[1:]), source))
            else:
                pon = Pon(group_tiles[0], tuple(group_tiles[1:3]), source)
                melds.append(Kakan(pon, group_tiles[3]))
    return concealed, melds


def _seven_pairs_hand(generator: random.Random, pool: Pool):
    _, kinds = _kind_choices(generator)
    if len(kinds) < 7:
        kinds = tuple(range(34))
    chosen = generator.sample(kinds, 7)
    return [pool.take(kind) for kind in chosen for _ in range(2)], []


def _kokushi_hand(generator: random.Random, pool: Pool):
    tiles = [pool.take(kind) for kind in _KOKUSHI]
    tiles.append(pool.take(generator.choice(_KOKUSHI)))
    return tiles, []


def _chuuren_hand(generator: random.Random, pool: Pool):
    suit = generator.randrange(3) * 9
    counts = (3, 1, 1, 1, 1, 1, 1, 1, 3)
    tiles = [
        pool.take(suit + rank)
        for rank, count in enumerate(counts)
        for _ in range(count)
    ]
    tiles.append(pool.take(suit + generator.randrange(9)))
    return tiles, []


def _hand(generator: random.Random, pool: Pool):
    kind = generator.choices(
        ("standard", "seven_pairs", "kokushi", "chuuren", "broken"),
        weights=(80, 8, 2, 1, 9),
    )[0]
    if kind == "seven_pairs":
        return _seven_pairs_hand(generator, pool)
    if kind == "kokushi":
        return _kokushi_hand(generator, pool)
    if kind == "chuuren":
        return _chuuren_hand(generator, pool)
    concealed, melds = _standard_hand(generator, pool)
    if kind == "broken":
        index = generator.randrange(len(concealed))
        concealed[index] = pool.take_any()
    return concealed, melds


def _case(generator: random.Random):
    pool = Pool(generator)
    concealed14, melds = _hand(generator, pool)
    winning_tile = generator.choice(concealed14)
    concealed = [tile for tile in concealed14 if tile.id != winning_tile.id]
    menzen = all(isinstance(meld, Ankan) for meld in melds)
    has_kan = any(isinstance(meld, (Ankan, Daiminkan, Kakan)) for meld in melds)

    method = generator.choice(("ron", "tsumo"))
    seat = generator.randrange(4)
    prevailing = generator.choice((0, 0, 1, 1, 2, 3))
    riichi = "none"
    if menzen and generator.random() < 0.45:
        riichi = "double_riichi" if generator.random() < 0.1 else "riichi"
    ippatsu = riichi != "none" and generator.random() < 0.2

    situation = "normal"
    roll = generator.random()
    if roll < 0.08 and method == "tsumo" and not melds and riichi == "none":
        situation = "tenhou" if seat == 0 else "chiihou"
    elif roll < 0.2:
        if method == "tsumo":
            options = ["haitei"] + (["rinshan"] if has_kan and not ippatsu else [])
        else:
            options = ["houtei", "chankan"]
        situation = generator.choice(options)

    indicator_count = 1 + generator.choice((0, 0, 0, 1, 2))
    dora = [pool.take_any() for _ in range(indicator_count)]
    ura = [pool.take_any() for _ in range(indicator_count)]

    origin = {
        ("ron", "chankan"): WinOrigin.KAKAN,
        ("tsumo", "rinshan"): WinOrigin.RINSHAN,
    }.get(
        (method, situation),
        WinOrigin.DISCARD if method == "ron" else WinOrigin.LIVE_WALL,
    )
    context = WinningContext(
        concealed_tiles=tuple(concealed14),
        winning_tile=winning_tile,
        method=WinMethod.RON if method == "ron" else WinMethod.TSUMO,
        origin=origin,
        seat_wind=tuple(Wind)[seat],
        prevailing_wind=tuple(Wind)[prevailing],
        declared_melds=tuple(melds),
        riichi_status={
            "none": RiichiStatus.NONE,
            "riichi": RiichiStatus.RIICHI,
            "double_riichi": RiichiStatus.DOUBLE_RIICHI,
        }[riichi],
        is_ippatsu=ippatsu,
        is_last_tile=situation in ("haitei", "houtei"),
        is_first_uninterrupted_turn=situation in ("tenhou", "chiihou"),
    )
    indicators = DoraIndicators(
        visible=(dora[0],), ura=(ura[0],), kan=tuple(dora[1:]), kan_ura=tuple(ura[1:])
    )
    rule_name = generator.choices(tuple(RULES), weights=RULE_WEIGHTS)[0]
    return {
        "concealed": tiles_notation(sorted(concealed, key=lambda tile: tile.id)),
        "melds": [[_meld_kind(meld), tiles_notation(meld.tiles)] for meld in melds],
        "winning": notation(winning_tile),
        "method": method,
        "seat_wind": seat,
        "prevailing_wind": prevailing,
        "riichi": riichi,
        "ippatsu": ippatsu,
        "situation": situation,
        "dora": tiles_notation(dora),
        "ura": tiles_notation(ura) if riichi != "none" else "",
        "rules": rule_name,
        "expected": _expected(context, indicators, RULES[rule_name]),
    }


def _meld_kind(meld) -> str:
    return {
        Chi: "chi",
        Pon: "pon",
        Daiminkan: "daiminkan",
        Ankan: "ankan",
        Kakan: "kakan",
    }[type(meld)]


def _expected(context, indicators, rules):
    if not find_winning_shapes(context.concealed_tiles, context.declared_melds):
        return {"status": "not_complete"}
    candidates = enumerate_winning_score_candidates(
        context, dora_indicators=indicators, rules=rules
    )
    if not candidates:
        return {"status": "no_yaku"}
    summaries = set()
    for candidate in select_max_score_candidates(candidates):
        hand_value = candidate.hand_value
        score = candidate.score
        dora_count = hand_value.dora_count
        fu = hand_value.fu_calculation
        summaries.add(
            json.dumps(
                {
                    "yaku": {
                        match.yaku.value: match.han or match.yakuman_units
                        for match in sorted(
                            hand_value.yaku_evaluation.matches,
                            key=lambda match: tuple(Yaku).index(match.yaku),
                        )
                    },
                    "han": score.han,
                    "fu": None if fu is None else fu.rounded_fu,
                    "yakuman_units": score.yakuman_units,
                    "dora": None
                    if dora_count is None
                    else [
                        dora_count.visible + dora_count.kan,
                        dora_count.red,
                        dora_count.ura + dora_count.kan_ura,
                    ],
                    "limit": score.limit.value,
                    "ron": score.ron_payment,
                    "tsumo_dealer": score.tsumo_dealer_payment,
                    "tsumo_non_dealer": score.tsumo_non_dealer_payment,
                    "winner_points": score.winner_points,
                },
                sort_keys=True,
            )
        )
    return {
        "status": "scored",
        "max_candidates": [json.loads(summary) for summary in sorted(summaries)],
    }


def _check_engine_revision(checkout: pathlib.Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if revision != ENGINE_REVISION:
        raise SystemExit(
            f"lisjong-engine checkout is at {revision}, expected {ENGINE_REVISION}"
        )
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise SystemExit("lisjong-engine checkout has local changes")
    imported = pathlib.Path(lisjong_engine.__file__).resolve()
    if not imported.is_relative_to(checkout.resolve()):
        raise SystemExit(f"imported lisjong_engine {imported} is not from {checkout}")


def generate() -> dict:
    generator = random.Random(SEED)
    cases = []
    while len(cases) < CASE_COUNT:
        try:
            cases.append(_case(generator))
        except _Retry:
            continue
    covered = {
        yaku
        for case in cases
        for candidate in case["expected"].get("max_candidates", ())
        for yaku in candidate["yaku"]
    }
    missing = sorted(yaku.value for yaku in Yaku if yaku.value not in covered)
    if missing:
        raise SystemExit(f"fixture does not cover every yaku: {missing}")
    return {
        "format_version": FORMAT_VERSION,
        "engine_repository": "https://github.com/lisbun/lisjong-engine",
        "engine_revision": ENGINE_REVISION,
        "generator": "tools/generate_scoring_engine_fixture.py",
        "seed": SEED,
        "rules": sorted(RULES),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--engine-checkout",
        type=pathlib.Path,
        required=True,
        help="git checkout of lisjong-engine at ENGINE_REVISION",
    )
    arguments = parser.parse_args()
    _check_engine_revision(arguments.engine_checkout)
    fixture = generate()
    cases = fixture.pop("cases")
    with arguments.output.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(fixture, sort_keys=True)[:-1])
        stream.write(', "cases": [\n')
        stream.write(
            ",\n".join(
                json.dumps(case, sort_keys=True, separators=(",", ":"))
                for case in cases
            )
        )
        stream.write("\n]}\n")
    statuses = {}
    for case in cases:
        status = case["expected"]["status"]
        statuses[status] = statuses.get(status, 0) + 1
    print(
        f"wrote {len(cases)} cases to {arguments.output}: {statuses}", file=sys.stderr
    )


if __name__ == "__main__":
    main()
