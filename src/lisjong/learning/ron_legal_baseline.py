"""Public-input-only rate baselines for the fixed normal-discard ron context.

Fitting and privileged source reading belong to ron_legal_accuracy. These rates
predict joint ron legality, not a player's choice or the eventual winner.
"""

from dataclasses import dataclass, replace

from lisjong.belief.canonical_axes import tile_type_index, wind_for_seat, wind_index
from lisjong.belief.conditional_uniform_hand_belief import (
    estimate_conditional_uniform_hand_belief,
)
from lisjong.belief.fixed_point import PROBABILITY_MAX_RAW
from lisjong.policy_contract import MeldKind, PolicyInput, RiichiState, Seat

CONTEXT_PROTOCOL = "project-standard-normal-discard-ron-v1"
STRATA = ("riichi", "open", "closed_non_riichi")
BASELINES = ("ron_rate", "wait_genbutsu")


def public_stratum(policy_input: PolicyInput, seat: Seat) -> str:
    if not isinstance(policy_input, PolicyInput) or not isinstance(seat, Seat):
        raise TypeError("a PolicyInput and Seat are required")
    if seat is policy_input.self_seat:
        raise ValueError("ron baseline targets opponents only")
    player = policy_input.players[seat]
    if player.riichi is not RiichiState.NONE:
        return "riichi"
    if any(m.kind is not MeldKind.ANKAN for m in player.melds):
        return "open"
    return "closed_non_riichi"


@dataclass(frozen=True, slots=True)
class StratumRates:
    wait_raw: tuple[int, ...]
    ron_raw: tuple[int, ...]

    def __post_init__(self):
        for field in ("wait_raw", "ron_raw"):
            values = tuple(getattr(self, field))
            if len(values) != 34 or any(
                type(v) is not int or not 0 <= v <= PROBABILITY_MAX_RAW for v in values
            ):
                raise ValueError("rates require 34 fixed-point probabilities")
            object.__setattr__(self, field, values)
        if any(r > w for r, w in zip(self.ron_raw, self.wait_raw)):
            raise ValueError("ron rates must not exceed wait rates")


@dataclass(frozen=True, slots=True)
class RonRateModel:
    """Three immutable tables; a stratum absent from train remains unavailable."""

    tables: tuple[StratumRates | None, ...]
    context_protocol: str = CONTEXT_PROTOCOL

    def __post_init__(self):
        tables = tuple(self.tables)
        if len(tables) != 3 or any(
            t is not None and not isinstance(t, StratumRates) for t in tables
        ):
            raise ValueError("exactly three stratum rate tables are required")
        if self.context_protocol != CONTEXT_PROTOCOL:
            raise ValueError("unknown ron context protocol")
        object.__setattr__(self, "tables", tables)

    def predict(self, policy_input: PolicyInput, seat: Seat, baseline: str):
        """Return a paired wait/ron HandBelief, or None for missing train support.

        wait_genbutsu only zeros tile types in the target's public river. It does
        not know the full wait set, missed-ron state, or concealed yaku.
        """
        if baseline not in BASELINES:
            raise ValueError("unknown ron baseline")
        rates = self.tables[STRATA.index(public_stratum(policy_input, seat))]
        if rates is None:
            return None
        dealer = policy_input.round.dealer_seat
        slots = [0] * 4
        for opponent in Seat:
            if opponent is not policy_input.self_seat:
                slots[wind_index(wind_for_seat(opponent, dealer))] = 13 - 3 * len(
                    policy_input.players[opponent].melds
                )
        marginal = estimate_conditional_uniform_hand_belief(
            policy_input, tuple(slots)
        ).hand(wind_for_seat(seat, dealer))
        river = {
            tile_type_index(d.tile.tile_type)
            for d in policy_input.players[seat].discards
        }
        ron = (
            rates.ron_raw
            if baseline == "ron_rate"
            else tuple(0 if i in river else p for i, p in enumerate(rates.wait_raw))
        )
        return replace(
            marginal, wait_probability_raw=rates.wait_raw, ron_legal_probability_raw=ron
        )
