"""既存`OwnHandState`からexact self-beliefを生成するfactory。

Issue #59の「既存`OwnHandState`からexact self-beliefを生成できるfactory」を
実装する。RiichiEnv / RiichiLab固有型へは依存しない。

`concealed_hand_marginals()`は、concealed tilesだけからexact
`expected_count_raw` / `red_five_probability_raw`を導出する共通ロジックで
あり、Issue #84の完全情報wait ground-truth builder
（`lisjong.belief.exact_wait_ground_truth`）もこれを再利用する。tile
marginalsがconcealed handのみから決まるという契約（own meldsを混入しない）は
両者で共有する。
"""

from lisjong.belief.canonical_axes import red_five_index, tile_type_index
from lisjong.belief.fixed_point import SCALE
from lisjong.belief.hand_belief import HandBelief
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.tile import Tile, TileCategory

_SUITED_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)


def concealed_hand_marginals(
    concealed_tiles: tuple[Tile, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """concealed tilesだけから、exact`expected_count_raw` /
    `red_five_probability_raw`のペアを導出する。

    通常5と赤5が混在する場合、34牌種側は両方とも対応する5へ加算し、
    red-five側は赤5が存在する色だけprobability = 1.0とする。
    """
    counts = [0] * 34
    has_red_five = [False, False, False]

    for tile in concealed_tiles:
        counts[tile_type_index(tile.tile_type)] += 1
        if tile.is_red:
            has_red_five[red_five_index(tile.tile_type.category)] = True

    expected_count_raw = tuple(count * SCALE for count in counts)
    red_five_probability_raw = tuple(
        SCALE if has_red_five[red_five_index(category)] else 0
        for category in _SUITED_CATEGORIES
    )
    return expected_count_raw, red_five_probability_raw


def exact_self_belief(own_hand_state: OwnHandState) -> HandBelief:
    """自席の`OwnHandState`から、実際の枚数をexactに表すHandBeliefを作る。

    `own_hand_state.drawn_tile`は`concealed_tiles`内のmetadataであり、
    追加の1枚として数えない。`concealed_tiles`内の各Tileを1回ずつ数える。

    `sum(expected_counts) == len(concealed_tiles)`をraw representation上でも
    exactに満たす。`OwnHandState`自体が13/14枚固定や非空制約を持たないため、
    このfactoryだけで独自にそれらの制約を追加しない。
    """
    if not isinstance(own_hand_state, OwnHandState):
        raise TypeError("own_hand_state must be an OwnHandState")

    expected_count_raw, red_five_probability_raw = concealed_hand_marginals(
        own_hand_state.concealed_tiles
    )

    return HandBelief(
        expected_count_raw=expected_count_raw,
        red_five_probability_raw=red_five_probability_raw,
    )
