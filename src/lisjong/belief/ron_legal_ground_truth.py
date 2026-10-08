"""Offline-only joint ron truth; privileged facts never belong to inference inputs.

The output is a counterfactual normal discard, including physically exhausted
public tile types. Historical opportunities use their actual tile and situation.
Neither function predicts the player's choice or the awarded winner.
"""

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum

from lisjong.belief.canonical_axes import tile_type_from_index
from lisjong.belief.exact_wait_ground_truth import exact_hand_belief_with_waits
from lisjong.belief.hand_belief import HandBelief
from lisjong.hand_evaluation.scoring import (
    PROJECT_STANDARD_SCORING_RULES,
    URA_DORA_EXCLUDED,
    EvaluationStatus,
    RiichiStatus,
    ScoringMeld,
    WinContext,
    WinMethod,
    WinningHand,
    WinSituation,
    _load_native,
    evaluate_win,
)
from lisjong.policy_contract import PublicMeld, Tile, TileCategory, Wind

CONTEXT_PROTOCOL = "project-standard-normal-discard-ron-v1"


class MissedRonState(Enum):
    NONE = "none"
    TEMPORARY = "temporary"
    RIICHI = "riichi"


@dataclass(frozen=True, slots=True)
class RonSeatContext:
    """Confirmed training fact, not an online feature or a default for missing data."""

    missed_ron_state: MissedRonState
    riichi_status: RiichiStatus
    is_ippatsu: bool

    def __post_init__(self):
        if not isinstance(self.missed_ron_state, MissedRonState):
            raise TypeError("missed_ron_state must be a MissedRonState")
        if not isinstance(self.riichi_status, RiichiStatus):
            raise TypeError("riichi_status must be a RiichiStatus")
        if type(self.is_ippatsu) is not bool:
            raise TypeError("is_ippatsu must be a bool")
        if self.riichi_status is RiichiStatus.NONE and (
            self.is_ippatsu or self.missed_ron_state is MissedRonState.RIICHI
        ):
            raise ValueError("ippatsu and riichi missed ron require established riichi")


def require_scoring_backend() -> None:
    """Check native/API availability before generating even all-zero truth."""
    _load_native()


def _waits_and_furiten(concealed, melds, river, context):
    if not isinstance(context, RonSeatContext):
        raise TypeError("context must be a RonSeatContext")
    river = tuple(river)
    if any(not isinstance(tile, Tile) for tile in river):
        raise TypeError("river must contain Tiles (including called discards)")
    truth = exact_hand_belief_with_waits(concealed, melds)
    waits = frozenset(
        tile_type_from_index(index)
        for index, raw in enumerate(truth.wait_probability_raw)
        if raw
    )
    return (
        truth,
        waits,
        (
            context.missed_ron_state is not MissedRonState.NONE
            or bool(waits & {tile.tile_type for tile in river})
        ),
    )


def _has_yaku(concealed, melds, tile, context, seat_wind, prevailing_wind, situation):
    result = evaluate_win(
        WinningHand(
            concealed_tiles=tuple(concealed),
            melds=tuple(ScoringMeld(meld.kind, meld.tiles) for meld in melds),
            winning_tile=tile,
        ),
        WinContext(
            method=WinMethod.RON,
            seat_wind=seat_wind,
            prevailing_wind=prevailing_wind,
            riichi=context.riichi_status,
            is_ippatsu=context.is_ippatsu,
            situation=situation,
            # Deliberate yaku-only projection: dora cannot supply a yaku.
            dora_indicators=(),
            ura_dora=URA_DORA_EXCLUDED,
        ),
        PROJECT_STANDARD_SCORING_RULES,
    )
    if result.status is EvaluationStatus.NOT_COMPLETE:
        raise ValueError("structural wait and scoring completion disagree")
    return result.status is EvaluationStatus.SCORED


def actual_ron_is_legal(
    concealed: tuple[Tile, ...],
    melds: tuple[PublicMeld, ...],
    river: tuple[Tile, ...],
    context: RonSeatContext,
    *,
    winning_tile: Tile,
    seat_wind: Wind,
    prevailing_wind: Wind,
    situation: WinSituation,
) -> bool:
    """Verify a historical discard/kakan opportunity, using its actual context.

    Ankan robbing is disabled in project-standard-v1 and is rejected by the
    history reader before this function. No ValueError becomes a negative label.
    """
    require_scoring_backend()
    if not isinstance(winning_tile, Tile):
        raise TypeError("winning_tile must be a Tile")
    if not isinstance(seat_wind, Wind) or not isinstance(prevailing_wind, Wind):
        raise TypeError("winds must be Wind values")
    if situation not in (
        WinSituation.NORMAL,
        WinSituation.HOUTEI,
        WinSituation.CHANKAN,
    ):
        raise ValueError("unsupported historical ron situation")
    _, waits, furiten = _waits_and_furiten(concealed, melds, river, context)
    # Validate actual physical completion even when a missed-ron mask is set.
    inventory = Counter(concealed)
    inventory.update(tile for meld in melds for tile in meld.tiles)
    if (
        sum(n for t, n in inventory.items() if t.tile_type == winning_tile.tile_type)
        >= 4
    ):
        raise ValueError("actual winning tile would be a fifth tile")
    if (
        winning_tile.tile_type.rank == 5
        and winning_tile.tile_type.category is not TileCategory.HONOR
    ):
        if inventory[winning_tile] >= (1 if winning_tile.is_red else 3):
            raise ValueError("actual winning five exceeds its physical inventory")
    if winning_tile.tile_type not in waits:
        return False
    has_yaku = _has_yaku(
        concealed, melds, winning_tile, context, seat_wind, prevailing_wind, situation
    )
    return has_yaku and not furiten


def exact_hand_belief_with_ron(
    concealed: tuple[Tile, ...],
    melds: tuple[PublicMeld, ...],
    river: tuple[Tile, ...],
    context: RonSeatContext,
    *,
    seat_wind: Wind,
    prevailing_wind: Wind,
) -> HandBelief:
    """Return binary structural and joint ron truth for the fixed protocol C.

    Hidden hand plus own meld inventory chooses plain five first, otherwise
    red five. Public exhaustion never removes a hypothetical structural wait.
    Native availability is mandatory even for non-tenpai/furiten rows.
    """
    require_scoring_backend()
    if not isinstance(seat_wind, Wind) or not isinstance(prevailing_wind, Wind):
        raise TypeError("winds must be Wind values")
    truth, waits, furiten = _waits_and_furiten(concealed, melds, river, context)
    inventory = Counter(concealed)
    inventory.update(tile for meld in melds for tile in meld.tiles)
    raw = [0] * 34
    for index, wait in enumerate(truth.wait_probability_raw):
        if not wait:
            continue
        tile_type = tile_type_from_index(index)
        tile = Tile(tile_type)
        if (
            tile_type.rank == 5
            and tile_type.category is not TileCategory.HONOR
            and inventory[tile] == 3
        ):
            tile = Tile(tile_type, is_red=True)
        if tile_type not in waits:
            raise AssertionError("inconsistent structural truth")
        has_yaku = _has_yaku(
            concealed,
            melds,
            tile,
            context,
            seat_wind,
            prevailing_wind,
            WinSituation.NORMAL,
        )
        if has_yaku and not furiten:
            raw[index] = wait
    return replace(truth, ron_legal_probability_raw=tuple(raw))
