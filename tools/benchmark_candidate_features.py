"""Issue #187のsecond-step materialization contractを決めるdevelopment benchmark。

representativeな通常打牌decisionについて、candidate feature stageごとの実測
時間を出す。

    python tools/benchmark_candidate_features.py
    python tools/benchmark_candidate_features.py --repeat 7

これはdevelopment-only utilityである。CI wall-clock thresholdやcorrectness
testの時間thresholdをここから作らない。productionのcandidate feature
implementationへbenchmark専用のhookを入れないため、stage別の時間は
`lisjong.structural_efficiency`のsupported semanticを直接呼んで測り、
`build_discard_candidate_features()`のtotalだけを別に測る。

測定値は単発最速値ではなくmedianを報告する。
"""

import argparse
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lisjong.learning.candidate_features import (  # noqa: E402
    build_discard_candidate_features,
    legal_discard_candidates,
)
from lisjong.policy_contract import (  # noqa: E402
    DecisionContext,
    Discard,
    DiscardAction,
    OwnHandState,
    PlayerPublicState,
    PolicyInput,
    RiichiState,
    RoundState,
    Seat,
    Tile,
    TileCategory,
    TileType,
    Wind,
)
from lisjong.structural_efficiency import (  # noqa: E402
    StructuralShantenEvaluator,
    evaluate_post_discard_hands,
    known_tile_counts,
    second_step_ukeire_score,
    ukeire_count,
)

_CATEGORIES = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}


def _hand(spec: str) -> tuple[Tile, ...]:
    """`123m456p`形式のhand specをTile tupleへ変換する。`0`は赤5である。"""
    tiles: list[Tile] = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        category = _CATEGORIES[character]
        for rank_character in ranks:
            rank = int(rank_character)
            tiles.append(
                Tile(
                    TileType(category, 5 if rank == 0 else rank),
                    is_red=rank == 0,
                )
            )
        ranks = ""
    if ranks:
        raise ValueError(f"hand spec has trailing ranks: {spec!r}")
    return tuple(tiles)


def _decision(
    concealed_spec: str, opponent_discard_specs: tuple[str, str, str]
) -> DecisionContext:
    """14枚の純手牌と他家3人の捨て牌からrepresentativeな通常打牌decisionを作る。

    `known_tile_counts()`が牌種ごとの可視枚数をfail closedで検証するため、
    tile conservationを破るfixtureはbenchmark実行時点で例外になる。
    """
    concealed = _hand(concealed_spec)
    if len(concealed) != 14:
        raise ValueError(f"concealed hand must have 14 tiles: {concealed_spec!r}")
    players = [
        PlayerPublicState(
            score=25000,
            discards=(),
            melds=(),
            riichi=RiichiState.NONE,
        )
    ]
    for spec in opponent_discard_specs:
        players.append(
            PlayerPublicState(
                score=25000,
                discards=tuple(
                    Discard(tile=tile, tsumogiri=False, order=order, called_by=None)
                    for order, tile in enumerate(_hand(spec))
                ),
                melds=(),
                riichi=RiichiState.NONE,
            )
        )
    policy_input = PolicyInput(
        self_seat=Seat.SEAT_0,
        round=RoundState(
            round_wind=Wind.EAST,
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=(Tile(TileType(TileCategory.HONOR, 6)),),
            live_wall_tiles_remaining=50,
        ),
        players=tuple(players),
        own_hand=OwnHandState(concealed_tiles=concealed, drawn_tile=concealed[-1]),
    )
    legal_actions = tuple(
        DiscardAction(actor=Seat.SEAT_0, tile=tile, tsumogiri=False)
        for tile in dict.fromkeys(concealed)
    )
    return DecisionContext(input=policy_input, legal_actions=legal_actions)


_DECISIONS: tuple[tuple[str, DecisionContext], ...] = (
    (
        "far wide hand",
        _decision("1259m3468p2579s13z", ("78m1p", "12p3s", "34s2z")),
    ),
    (
        "mid honor-pair hand",
        _decision("234m5689p345s1177z", ("78m1p", "12p9s", "67s2z")),
    ),
    (
        "two-step relevant hand",
        _decision("345m56679s333577z", ("78m1p", "12p1s", "2s4z")),
    ),
    (
        "wide suited hand",
        _decision("23455m34567p2345s", ("19m1z", "89m2z", "18p3z")),
    ),
    (
        "tenpai-reachable hand",
        _decision("123456789m111p23p", ("123z", "456z", "77z")),
    ),
)
"""candidate数・向聴数の異なるrepresentativeな通常打牌decision。"""


def _median_ms(function, repeat: int) -> float:
    """`function`をrepeat回実行し、1回あたりのmedian時間をミリ秒で返す。"""
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _stage_times(decision: DecisionContext, repeat: int) -> dict[str, float]:
    """stage別のmedian時間を測る。各stageは前段を含むcumulativeな呼び出しである。"""
    candidates = legal_discard_candidates(decision)
    policy_input = decision.input

    def shanten_stage() -> None:
        evaluator = StructuralShantenEvaluator()
        known_tile_counts(policy_input)
        evaluate_post_discard_hands(policy_input, candidates, evaluator)

    def ukeire_stage() -> None:
        evaluator = StructuralShantenEvaluator()
        known_counts = known_tile_counts(policy_input)
        for evaluation in evaluate_post_discard_hands(
            policy_input, candidates, evaluator
        ):
            ukeire_count(
                evaluation.post_discard_hand,
                known_counts,
                evaluation.post_discard_shanten,
                evaluator,
            )

    def second_step_stage() -> None:
        evaluator = StructuralShantenEvaluator()
        known_counts = known_tile_counts(policy_input)
        for evaluation in evaluate_post_discard_hands(
            policy_input, candidates, evaluator
        ):
            if evaluation.post_discard_shanten < 1:
                continue
            second_step_ukeire_score(
                evaluation.post_discard_hand,
                known_counts,
                evaluation.post_discard_shanten,
                evaluator,
            )

    selective = candidates[: max(1, len(candidates) // 4)]

    return {
        "candidates": float(len(candidates)),
        "shanten_ms": _median_ms(shanten_stage, repeat),
        "all_ukeire_ms": _median_ms(ukeire_stage, repeat),
        "all_second_step_ms": _median_ms(second_step_stage, repeat),
        "build_no_second_step_ms": _median_ms(
            lambda: build_discard_candidate_features(decision), repeat
        ),
        "build_selective_ms": _median_ms(
            lambda: build_discard_candidate_features(
                decision, second_step_actions=selective
            ),
            repeat,
        ),
        "build_all_second_step_ms": _median_ms(
            lambda: build_discard_candidate_features(
                decision, second_step_actions=candidates
            ),
            repeat,
        ),
        "selective_requested": float(len(selective)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat",
        type=int,
        default=7,
        help="1 decision / 1 stageあたりの測定回数（medianを報告する）",
    )
    arguments = parser.parse_args()
    if arguments.repeat < 1:
        parser.error("--repeat must be at least 1")

    header = (
        f"{'decision':<30}{'cand':>5}{'shanten':>10}{'ukeire':>10}"
        f"{'2nd(all)':>10}{'build':>10}{'build+sel':>11}{'build+all':>11}"
    )
    print(f"repeat={arguments.repeat} (median ms per call)")
    print(header)
    print("-" * len(header))
    for label, decision in _DECISIONS:
        times = _stage_times(decision, arguments.repeat)
        print(
            f"{label:<30}{int(times['candidates']):>5}"
            f"{times['shanten_ms']:>10.3f}{times['all_ukeire_ms']:>10.3f}"
            f"{times['all_second_step_ms']:>10.3f}"
            f"{times['build_no_second_step_ms']:>10.3f}"
            f"{times['build_selective_ms']:>11.3f}"
            f"{times['build_all_second_step_ms']:>11.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
