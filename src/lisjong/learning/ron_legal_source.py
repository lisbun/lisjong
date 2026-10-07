"""Strict offline extension of the unchanged hand-belief v1 source (#262).

The journal contains producer facts, not serialized engine objects. Replay owns
only AI label semantics (missed ron and scoring context); game transitions remain
engine-owned. Full checkpoints bind those facts to v1 snapshots and reactions.
"""

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from lisjong.belief.ron_legal_ground_truth import (
    CONTEXT_PROTOCOL,
    MissedRonState,
    RonSeatContext,
    actual_ron_is_legal,
    exact_hand_belief_with_ron,
    require_scoring_backend,
)
from lisjong.hand_evaluation.scoring import RiichiStatus, WinSituation
from lisjong.learning import hand_belief_source as base
from lisjong.learning._canonical import (
    canonical_json_text,
    expect_bool,
    expect_digest,
    expect_list,
    expect_non_negative_int,
    expect_object,
    expect_str,
    file_digest,
    parse_json_text,
)
from lisjong.learning._typed_values import (
    _parse_enum,
    _parse_seat,
    parse_action,
    parse_policy_input,
    parse_tile,
    policy_input_to_value,
)
from lisjong.policy_contract import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DiscardAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    MeldKind,
    PassAction,
    PolicyInput,
    PonAction,
    PublicMeld,
    RiichiAction,
    RiichiState,
    RonAction,
    Seat,
    TsumoAction,
    Wind,
)

MANIFEST_SCHEMA = "lisjong-ron-legal-source-manifest-v1"
FACT_SCHEMA = "lisjong-ron-legal-fact-record-v1"
HISTORY_SCHEMA = "lisjong-ron-legal-history-record-v1"
FACTS_FILENAME = "ron_facts.jsonl"
HISTORY_FILENAME = "ron_history.jsonl"

# Semantic subset needed by this consumer. Other scoring/settlement rule values
# cannot silently change C; this exact projection and identity are mandatory.
RULES = {
    "identity": "project-standard-v1",
    "kuitan_enabled": True,
    "red_dora_enabled": True,
    "rounded_mangan_enabled": False,
    "counted_yakuman_enabled": True,
    "multiple_yakuman_enabled": True,
    "double_yakuman_variants": [],
    "double_wind_pair_fu": 4,
    "kokushi_ankan_chankan_enabled": False,
    "ron_resolution_policy": "multiple_ron",
    "triple_ron_abortive_draw": True,
}


class RonLegalSourceError(ValueError):
    """Missing, noncanonical or contradictory training observations."""


_E = RonLegalSourceError


@dataclass(frozen=True, slots=True)
class RonSnapshot:
    key: base.DecisionKey
    round_id: str
    history_boundary: int
    opponents: tuple[RonSeatContext, ...]


@dataclass(frozen=True, slots=True)
class RonCheckpoint:
    views: tuple[PolicyInput, ...]
    contexts: tuple[RonSeatContext, ...]


@dataclass(frozen=True, slots=True)
class RonSource:
    """Fully checked offline population; never pass this to a Policy."""

    base_manifest: base.HandBeliefManifest
    decisions: tuple[base.HandBeliefDecision, ...]
    hands: tuple[base.HandBeliefHandFacts, ...]
    snapshots: tuple[RonSnapshot, ...]
    checkpoints: tuple[RonCheckpoint, ...]


def _object(value, fields, context):
    return expect_object(value, frozenset(fields.split()), _E, context)


def _integer(value, context):
    return expect_non_negative_int(value, _E, context)


def _contexts(value, context):
    result = []
    for index, item in enumerate(expect_list(value, _E, context)):
        raw = _object(item, "missed_ron_state riichi_status is_ippatsu", context)
        try:
            result.append(
                RonSeatContext(
                    _parse_enum(MissedRonState, raw["missed_ron_state"], _E, context),
                    _parse_enum(RiichiStatus, raw["riichi_status"], _E, context),
                    expect_bool(
                        raw["is_ippatsu"], _E, f"{context}[{index}].is_ippatsu"
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise _E(f"{context}: {error}") from error
    return tuple(result)


def _hand(view):
    # OwnHandState.concealed_tiles already includes drawn_tile.
    return view.own_hand.concealed_tiles


def _checkpoint(value, context):
    raw = _object(value, "views contexts", context)
    views = tuple(
        parse_policy_input(view, _E, context)
        for view in expect_list(raw["views"], _E, context)
    )
    contexts = _contexts(raw["contexts"], context)
    if len(views) != 4 or len(contexts) != 4:
        raise _E(f"{context}: checkpoint requires four seats")
    for seat, (view, fact) in enumerate(zip(views, contexts)):
        if raw["views"][seat] != policy_input_to_value(view, _E, context):
            raise _E(f"{context}: view does not round-trip canonically")
        if view.self_seat != Seat(seat):
            raise _E(f"{context}: views must be in seat order")
        if view.round != views[0].round or view.players != views[0].players:
            raise _E(f"{context}: public projections disagree")
        public = view.players[seat]
        if (public.riichi is RiichiState.ACCEPTED) != (
            fact.riichi_status is not RiichiStatus.NONE
        ):
            raise _E(f"{context}: established riichi differs from public state")
        if fact.riichi_status is not RiichiStatus.NONE and any(
            meld.kind.value != "ankan" for meld in public.melds
        ):
            raise _E(f"{context}: established riichi with an open meld")
        if len(_hand(view)) + 3 * len(public.melds) not in (13, 14):
            raise _E(f"{context}: hand is neither stable nor drawn")
    # Reuse the v1 physical-pool check, including called rivers and red/plain 5.
    opponents = tuple(
        base.OpponentHand(seat, 0, _hand(views[seat]), views[0].players[seat].melds)
        for seat in range(1, 4)
    )
    base._check_conservation_input(views[0], opponents, context)
    return RonCheckpoint(views, contexts)


def _rows(root, filename, entry):
    base._check_file(root, filename, entry)
    rows = tuple(value for _, value in base._read_lines(root / filename))
    if len(rows) != entry["rows"]:
        raise _E(f"{filename}: row count differs from manifest")
    return rows


def _manifest(root, original):
    path = root / "manifest.json"
    text = path.read_text(encoding="utf-8")
    raw = _object(
        parse_json_text(text, _E, "manifest"),
        "schema base_manifest_sha256 context_protocol rules producer splits files coverage",
        "manifest",
    )
    if raw["schema"] != MANIFEST_SCHEMA or raw["context_protocol"] != CONTEXT_PROTOCOL:
        raise _E("unsupported manifest schema or context protocol")
    expect_digest(raw["base_manifest_sha256"], _E, "base_manifest_sha256")
    if raw["base_manifest_sha256"] != file_digest(original / "manifest.json")["sha256"]:
        raise _E("base manifest digest mismatch")
    if canonical_json_text(raw["rules"]) != canonical_json_text(RULES):
        raise _E("unsupported rule identity or semantic values")
    if text != canonical_json_text(raw):
        raise _E("manifest is not canonical JSON")
    producer = _object(
        raw["producer"],
        "arena_revision lisjong_revision lisjong_engine_revision policy",
        "producer",
    )
    for name, value in producer.items():
        expect_str(value, _E, name)
    files = _object(raw["files"], "ron_facts ron_history", "files")
    for name, value in files.items():
        entry = _object(value, "bytes sha256 rows", name)
        expect_digest(entry["sha256"], _E, name)
        _integer(entry["bytes"], name)
        _integer(entry["rows"], name)
    return raw


def _snapshot(value):
    raw = _object(value, "schema key round_id history_boundary opponents", "ron fact")
    if raw["schema"] != FACT_SCHEMA:
        raise _E("unsupported ron fact schema")
    key = base._parse_key(raw["key"], "ron fact key")
    seats = []
    contexts = []
    for item in expect_list(raw["opponents"], _E, "opponents"):
        opponent = _object(item, "seat sequence context", "opponent")
        seats.append(int(_parse_seat(opponent["seat"], _E, "seat")))
        if _integer(opponent["sequence"], "sequence") != key.sequence:
            raise _E("ron fact is not the decision-point snapshot")
        contexts.extend(_contexts([opponent["context"]], "opponent context"))
    if seats != [seat for seat in range(4) if seat != key.seat]:
        raise _E("ron facts require three opponents in seat order")
    return RonSnapshot(
        key,
        expect_str(raw["round_id"], _E, "round_id"),
        _integer(raw["history_boundary"], "history_boundary"),
        tuple(contexts),
    )


def _seat_set(value, source, context):
    seats = tuple(
        int(_parse_seat(item, _E, context)) for item in expect_list(value, _E, context)
    )
    if (
        seats != tuple(sorted(set(seats), key=lambda seat: (seat - source) % 4))
        or source in seats
    ):
        raise _E(f"{context}: seats must be unique in turn order after source")
    return seats


def _reaction(value, before, selectors, ids, discard_sources):
    raw = _object(
        value,
        "reaction_id origin source_seat winning_tile discard_draw_kind candidates ron_capable ron_selected ron_awarded ron_passed resolution resolved_action",
        "reaction",
    )
    reaction_id = expect_str(raw["reaction_id"], _E, "reaction_id")
    if reaction_id in ids:
        raise _E("duplicate reaction ID")
    ids.add(reaction_id)
    origin = raw["origin"]
    if origin not in ("discard", "kakan", "ankan"):
        raise _E("unsupported reaction origin")
    source = int(_parse_seat(raw["source_seat"], _E, "source_seat"))
    tile = parse_tile(raw["winning_tile"], _E, "winning_tile")
    source_view = before.views[source]
    if origin == "discard":
        river = source_view.players[source].discards
        if not river or river[-1].tile != tile or river[-1].called_by is not None:
            raise _E("reaction tile differs from the latest uncalled source discard")
        if discard_sources.get(source) != (tile, raw["discard_draw_kind"]):
            raise _E("reaction discard draw source differs from the journal")
    elif raw["discard_draw_kind"] is not None:
        raise _E("kan reaction must not claim a discard draw source")
    elif tile not in _hand(source_view):
        raise _E("kan reaction tile is absent from the source hand")
    if (
        origin == "ankan"
        and sum(t.tile_type == tile.tile_type for t in _hand(source_view)) != 4
    ):
        raise _E("ankan reaction source does not hold four tiles")
    if origin == "kakan" and not any(
        m.kind.value == "pon" and m.tiles[0].tile_type == tile.tile_type
        for m in source_view.players[source].melds
    ):
        raise _E("kakan reaction source has no matching pon")
    situation = (
        WinSituation.CHANKAN
        if origin != "discard"
        else WinSituation.HOUTEI
        if source_view.round.live_wall_tiles_remaining == 0
        and raw["discard_draw_kind"] == "normal"
        else WinSituation.NORMAL
    )
    capable, selected, selected_calls = [], [], []
    candidate_sequences = []
    candidates = expect_list(raw["candidates"], _E, "candidates")
    expected_seats = [seat for seat in range(4) if seat != source]
    if len(candidates) != 3:
        raise _E(
            "reaction requires observations for every other seat, including empty candidates"
        )
    for seat, item in zip(expected_seats, candidates):
        candidate = _object(
            item, "seat sequence legal_actions selected_action", "candidate"
        )
        if _parse_seat(candidate["seat"], _E, "candidate seat") != Seat(seat):
            raise _E("candidates must be in seat order")
        actions = tuple(
            parse_action(action, _E, "legal action")
            for action in expect_list(candidate["legal_actions"], _E, "legal_actions")
        )
        if len(set(actions)) != len(actions):
            raise _E("duplicate legal reaction action")
        for action in actions:
            if action.actor != Seat(seat) or not isinstance(
                action, (PassAction, RonAction, ChiAction, PonAction, DaiminkanAction)
            ):
                raise _E("invalid reaction action actor or kind")
            if not isinstance(action, PassAction):
                action_tile = (
                    action.winning_tile
                    if isinstance(action, RonAction)
                    else action.called_tile
                )
                if action.target != Seat(source) or action_tile != tile:
                    raise _E("reaction action has a different target or tile")
                if origin != "discard" and not isinstance(action, RonAction):
                    raise _E("kan reaction cannot offer an open call")
                if isinstance(
                    action, (ChiAction, PonAction, DaiminkanAction)
                ) and Counter(action.consumed_tiles) - Counter(
                    _hand(before.views[seat])
                ):
                    raise _E("call consumes tiles absent from the candidate hand")
                if isinstance(action, ChiAction) and seat != (source + 1) % 4:
                    raise _E("chi candidate is not the next seat")
        if actions:
            if PassAction(Seat(seat)) not in actions:
                raise _E("nonempty reaction candidate must include explicit pass")
            if before.contexts[seat].riichi_status is not RiichiStatus.NONE and any(
                isinstance(a, (ChiAction, PonAction, DaiminkanAction)) for a in actions
            ):
                raise _E("established riichi cannot offer an open call")
            sequence = _integer(candidate["sequence"], "candidate sequence")
            candidate_sequences.append(sequence)
            choice = parse_action(candidate["selected_action"], _E, "selected action")
            if choice not in actions:
                raise _E("selected reaction action is not legal")
        else:
            if (
                candidate["sequence"] is not None
                or candidate["selected_action"] is not None
            ):
                raise _E("empty candidate must not have a selector invocation")
            choice = None
        view = before.views[seat]
        player = view.players[seat]
        expected_ron = (
            False
            if origin == "ankan"
            else actual_ron_is_legal(
                _hand(view),
                player.melds,
                tuple(discard.tile for discard in player.discards),
                before.contexts[seat],
                winning_tile=tile,
                seat_wind=tuple(Wind)[(seat - int(view.round.dealer_seat)) % 4],
                prevailing_wind=view.round.round_wind,
                situation=situation,
            )
        )
        offered = [action for action in actions if isinstance(action, RonAction)]
        if len(offered) != int(expected_ron):
            raise _E("offered ron differs from structural/furiten/yaku legality")
        if offered:
            capable.append(seat)
        if isinstance(choice, RonAction):
            selected.append(seat)
        if isinstance(choice, (ChiAction, PonAction, DaiminkanAction)):
            selected_calls.append(choice)
    if sorted(candidate_sequences) != list(selectors):
        raise _E("reaction selectors differ from candidate sequences")

    def order(seat):
        return (seat - source) % 4

    capable, selected = (
        tuple(sorted(capable, key=order)),
        tuple(sorted(selected, key=order)),
    )
    passed = tuple(seat for seat in capable if seat not in selected)
    for name, expected in (
        ("ron_capable", capable),
        ("ron_selected", selected),
        ("ron_passed", passed),
    ):
        if _seat_set(raw[name], source, name) != expected:
            raise _E(f"{name} differs from the actual candidate choices")
    awarded = _seat_set(raw["ron_awarded"], source, "ron_awarded")
    resolution = raw["resolution"]
    resolved_action = (
        None
        if raw["resolved_action"] is None
        else parse_action(raw["resolved_action"], _E, "resolved_action")
    )
    if selected:
        # Engine E2 retains all awarded seats even for three selected rons.
        # E3 subsequently decides triple-ron abortive draw at round_result.
        expected_awarded = selected
        expected_resolution = "ron"
        if (
            awarded != expected_awarded
            or resolution != expected_resolution
            or resolved_action is not None
        ):
            raise _E("ron award/resolution differs from project-standard rules")
    elif awarded or resolution not in ("call", "pass"):
        raise _E("non-ron reaction has invalid awards/resolution")
    elif resolution == "call":
        if resolved_action not in selected_calls:
            raise _E("resolved call was not selected by a legal candidate")
    elif resolved_action is not None or selected_calls:
        raise _E("pass resolution cannot discard a selected call")
    contexts = list(before.contexts)
    for seat in passed:
        old = contexts[seat]
        reason = (
            MissedRonState.RIICHI
            if old.riichi_status is not RiichiStatus.NONE
            else MissedRonState.TEMPORARY
        )
        contexts[seat] = replace(old, missed_ron_state=reason)
    if resolution == "call":
        contexts = [replace(context, is_ippatsu=False) for context in contexts]
    return tuple(contexts)


def _board_change(before, after, event, kan_action=None):
    """Check recorded hand/meld/river edits, without generating engine actions."""
    actor = None
    expected_hand = None
    expected_melds = None
    expected_rivers = [p.discards for p in before.views[0].players]
    kind = event["kind"]
    if kind == "draw":
        actor = int(event["seat"])
        expected_hand = Counter(_hand(before.views[actor])) + Counter(
            (after.views[actor].own_hand.drawn_tile,)
        )
    elif kind == "progress":
        action = parse_action(event["action"], _E, "board progress")
        if isinstance(action, DiscardAction):
            actor = int(action.actor)
            expected_hand = Counter(_hand(before.views[actor])) - Counter(
                (action.tile,)
            )
            old = expected_rivers[actor]
            new = after.views[0].players[actor].discards
            if (
                len(new) != len(old) + 1
                or new[:-1] != old
                or new[-1].tile != action.tile
                or new[-1].called_by is not None
                or new[-1].tsumogiri != action.tsumogiri
            ):
                raise _E("discard river edit differs from its action")
            expected_rivers[actor] = new
    elif kind == "reaction":
        proof = event["evidence"]
        if proof["resolution"] == "call":
            action = parse_action(proof["resolved_action"], _E, "resolved call")
            actor = int(action.actor)
            expected_hand = Counter(_hand(before.views[actor])) - Counter(
                action.consumed_tiles
            )
            meld_kind = {
                ChiAction: MeldKind.CHI,
                PonAction: MeldKind.PON,
                DaiminkanAction: MeldKind.DAIMINKAN,
            }[type(action)]
            expected_melds = (
                *before.views[0].players[actor].melds,
                PublicMeld(
                    meld_kind,
                    (*action.consumed_tiles, action.called_tile),
                    action.target,
                    action.called_tile,
                ),
            )
            source = int(action.target)
            old = expected_rivers[source]
            expected_rivers[source] = (
                *old[:-1],
                replace(old[-1], called_by=action.actor),
            )
    elif kind == "kan_confirmed":
        actor = int(event["seat"])
        old = before.views[0].players[actor].melds
        if isinstance(kan_action, AnkanAction):
            expected_hand = Counter(_hand(before.views[actor])) - Counter(
                kan_action.tiles
            )
            expected_melds = (
                *old,
                PublicMeld(MeldKind.ANKAN, kan_action.tiles, None, None),
            )
        elif isinstance(kan_action, KakanAction):
            matches = [
                i
                for i, meld in enumerate(old)
                if meld.kind is MeldKind.PON
                and meld.from_seat == kan_action.from_seat
                and meld.called_tile == kan_action.called_tile
                and meld.tiles[0].tile_type == kan_action.added_tile.tile_type
            ]
            if len(matches) != 1:
                raise _E("confirmed kakan does not identify exactly one prior pon")
            index = matches[0]
            expected_hand = Counter(_hand(before.views[actor])) - Counter(
                (kan_action.added_tile,)
            )
            expected_melds = (
                *old[:index],
                PublicMeld(
                    MeldKind.KAKAN,
                    (*old[index].tiles, kan_action.added_tile),
                    kan_action.from_seat,
                    kan_action.called_tile,
                ),
                *old[index + 1 :],
            )
        else:
            raise _E("kan confirmation lacks its typed declaration")
    for seat in range(4):
        expected = (
            expected_hand if seat == actor else Counter(_hand(before.views[seat]))
        )
        if Counter(_hand(after.views[seat])) != expected:
            raise _E("checkpoint hand changed without its recorded action/draw")
        player = after.views[0].players[seat]
        melds = (
            expected_melds
            if seat == actor and expected_melds is not None
            else before.views[0].players[seat].melds
        )
        if player.melds != melds or player.discards != expected_rivers[seat]:
            raise _E("checkpoint meld/river changed without its recorded call/discard")


def _replay_steps(rows, coverage):
    checkpoints = {}
    selector_commits = {}
    previous_sequence = {}
    offset = 0
    round_keys = set()
    coverage_seeds = set()
    for value in expect_list(coverage, _E, "coverage"):
        count = _object(
            value,
            "seed round_id transitions reactions decisions selectors",
            "coverage round",
        )
        seed = _integer(count["seed"], "coverage seed")
        round_id = expect_str(count["round_id"], _E, "coverage round_id")
        key = seed, round_id
        if key in round_keys:
            raise _E("duplicate coverage round")
        round_keys.add(key)
        coverage_seeds.add(seed)
        length = _integer(count["transitions"], "transitions")
        expected_reactions = _integer(count["reactions"], "reactions")
        _integer(count["decisions"], "decisions")
        expected_selectors = _integer(count["selectors"], "selectors")
        if length < 2 or offset + length > len(rows):
            raise _E("round coverage is incomplete")
        states = []
        commit_selectors = []
        before = None
        reactions = 0
        ids = set()
        draw_sources = {}
        discard_sources = {}
        kan_pending = {}
        riichi_resolutions = {}
        rinshan_due = set()
        terminal = False
        for index, value in enumerate(rows[offset : offset + length]):
            raw = _object(
                value,
                "schema seed round_id index selector_sequences event checkpoint",
                "history",
            )
            if (
                raw["schema"] != HISTORY_SCHEMA
                or _integer(raw["seed"], "seed") != seed
                or raw["round_id"] != round_id
                or _integer(raw["index"], "index") != index
            ):
                raise _E("history schema/round/index is not contiguous")
            selectors = tuple(
                _integer(item, "selector sequence")
                for item in expect_list(raw["selector_sequences"], _E, "selectors")
            )
            if selectors != tuple(sorted(set(selectors))):
                raise _E("selector sequences must be unique and increasing")
            if selectors and selectors != tuple(
                range(
                    previous_sequence.get(seed, -1) + 1,
                    previous_sequence.get(seed, -1) + 1 + len(selectors),
                )
            ):
                raise _E("selector sequence was omitted, repeated or went backwards")
            if selectors:
                previous_sequence[seed] = selectors[-1]
            commit_selectors.append(selectors)
            after = _checkpoint(raw["checkpoint"], "history checkpoint")
            event = raw["event"]
            if type(event) is not dict or "kind" not in event:
                raise _E("history event must have a kind")
            kind = event["kind"]
            if terminal and kind not in (
                "riichi_cancelled",
                "round_result",
                "round_end",
            ):
                raise _E("history continues play after a terminal win resolution")
            if index == 0:
                _object(event, "kind", "round_start")
                if (
                    kind != "round_start"
                    or selectors
                    or any(
                        c
                        != RonSeatContext(MissedRonState.NONE, RiichiStatus.NONE, False)
                        for c in after.contexts
                    )
                    or any(
                        p.riichi is not RiichiState.NONE for p in after.views[0].players
                    )
                    or any(p.discards or p.melds for p in after.views[0].players)
                    or any(
                        len(_hand(view)) != 13 or view.own_hand.drawn_tile is not None
                        for view in after.views
                    )
                ):
                    raise _E("round must start with confirmed cleared contexts")
                expected = after.contexts
            elif index == length - 1:
                _object(event, "kind", "round_end")
                if kind != "round_end" or selectors or after != before or not terminal:
                    raise _E("round needs an unchanged, flushed end checkpoint")
                expected = before.contexts
            else:
                if (
                    before.views[0].round.round_wind != after.views[0].round.round_wind
                    or before.views[0].round.dealer_seat
                    != after.views[0].round.dealer_seat
                    or before.views[0].round.hand_number
                    != after.views[0].round.hand_number
                    or before.views[0].round.honba != after.views[0].round.honba
                ):
                    raise _E("round identity changed inside its journal")
                expected = list(before.contexts)
                allowed_riichi = [p.riichi for p in before.views[0].players]
                kan_action = None
                if kind == "draw":
                    _object(event, "kind seat draw_kind", "draw")
                    seat = int(_parse_seat(event["seat"], _E, "draw seat"))
                    if event["draw_kind"] not in ("normal", "rinshan") or selectors:
                        raise _E("invalid draw kind or selector-bearing draw")
                    if (event["draw_kind"] == "rinshan") != (seat in rinshan_due):
                        raise _E("draw source differs from the confirmed kan history")
                    rinshan_due.discard(seat)
                    if (
                        before.views[seat].own_hand.drawn_tile is not None
                        or after.views[seat].own_hand.drawn_tile is None
                        or Counter(after.views[seat].own_hand.concealed_tiles)
                        != Counter(_hand(before.views[seat]))
                        + Counter((after.views[seat].own_hand.drawn_tile,))
                    ):
                        raise _E("draw checkpoint does not record an actual own draw")
                    if expected[seat].missed_ron_state is MissedRonState.TEMPORARY:
                        expected[seat] = replace(
                            expected[seat], missed_ron_state=MissedRonState.NONE
                        )
                    draw_sources[seat] = event["draw_kind"]
                elif kind == "riichi_established":
                    _object(
                        event,
                        "kind seat riichi_status is_ippatsu reaction_id",
                        "riichi_established",
                    )
                    seat = int(_parse_seat(event["seat"], _E, "riichi seat"))
                    status = _parse_enum(
                        RiichiStatus, event["riichi_status"], _E, "riichi_status"
                    )
                    if (
                        selectors
                        or status is RiichiStatus.NONE
                        or before.views[0].players[seat].riichi
                        is not RiichiState.DECLARED
                        or expected[seat].riichi_status is not RiichiStatus.NONE
                    ):
                        raise _E("invalid riichi establishment")
                    ippatsu = expect_bool(event["is_ippatsu"], _E, "is_ippatsu")
                    if riichi_resolutions.get(event["reaction_id"]) != (
                        seat,
                        "pass" if ippatsu else "call",
                    ):
                        raise _E(
                            "riichi establishment differs from declaration reaction resolution"
                        )
                    expected[seat] = replace(
                        expected[seat], riichi_status=status, is_ippatsu=ippatsu
                    )
                    allowed_riichi[seat] = RiichiState.ACCEPTED
                elif kind == "riichi_cancelled":
                    _object(event, "kind seat reaction_id", "riichi_cancelled")
                    seat = int(_parse_seat(event["seat"], _E, "riichi seat"))
                    if (
                        selectors
                        or before.views[0].players[seat].riichi
                        is not RiichiState.DECLARED
                        or riichi_resolutions.get(event["reaction_id"]) != (seat, "ron")
                    ):
                        raise _E(
                            "riichi cancellation lacks a declaration-tile ron resolution"
                        )
                    allowed_riichi[seat] = RiichiState.NONE
                elif kind == "progress":
                    _object(event, "kind sequence action", "progress")
                    action = parse_action(event["action"], _E, "progress action")
                    terminal = isinstance(action, (TsumoAction, KyuushuKyuuhaiAction))
                    if selectors != (
                        _integer(event["sequence"], "progress sequence"),
                    ) or isinstance(
                        action,
                        (PassAction, RonAction, ChiAction, PonAction, DaiminkanAction),
                    ):
                        raise _E(
                            "progress must identify one non-reaction selector action"
                        )
                    if isinstance(action, DiscardAction):
                        seat = int(action.actor)
                        if (
                            Counter(_hand(before.views[seat]))
                            != Counter(_hand(after.views[seat]))
                            + Counter((action.tile,))
                            or not after.views[0].players[seat].discards
                            or after.views[0].players[seat].discards[-1].tile
                            != action.tile
                        ):
                            raise _E(
                                "discard checkpoint does not apply the selected tile"
                            )
                        expected[seat] = replace(expected[seat], is_ippatsu=False)
                        discard_sources[seat] = (
                            action.tile,
                            draw_sources.pop(seat, None),
                        )
                    elif isinstance(action, (AnkanAction, KakanAction)):
                        if Counter(_hand(before.views[int(action.actor)])) != Counter(
                            _hand(after.views[int(action.actor)])
                        ):
                            raise _E(
                                "kan declaration must retain its hand until confirmation"
                            )
                        kan_pending[int(action.actor)] = action
                    elif isinstance(action, RiichiAction):
                        if (
                            before.views[0].players[int(action.actor)].riichi
                            is not RiichiState.NONE
                            or after.views[0].players[int(action.actor)].riichi
                            is not RiichiState.DECLARED
                        ):
                            raise _E("riichi declaration checkpoint mismatch")
                        allowed_riichi[int(action.actor)] = RiichiState.DECLARED
                elif kind == "reaction":
                    _object(event, "kind evidence", "reaction")
                    reactions += 1
                    expected = _reaction(
                        event["evidence"], before, selectors, ids, discard_sources
                    )
                    proof = event["evidence"]
                    terminal = proof["resolution"] in ("ron",)
                    if (
                        proof["origin"] == "discard"
                        and before.views[0].players[proof["source_seat"]].riichi
                        is RiichiState.DECLARED
                    ):
                        riichi_resolutions[proof["reaction_id"]] = (
                            proof["source_seat"],
                            proof["resolution"],
                        )
                    if proof["resolution"] == "call":
                        call = parse_action(
                            proof["resolved_action"], _E, "resolved call"
                        )
                        if isinstance(call, DaiminkanAction):
                            rinshan_due.add(int(call.actor))
                elif kind == "kan_confirmed":
                    _object(event, "kind seat", "kan_confirmed")
                    seat = int(_parse_seat(event["seat"], _E, "kan seat"))
                    if selectors or seat not in kan_pending:
                        raise _E("kan confirmation lacks a declaration")
                    kan_action = kan_pending.pop(seat)
                    rinshan_due.add(seat)
                    expected = [replace(c, is_ippatsu=False) for c in expected]
                elif kind == "round_result":
                    _object(event, "kind", "round_result")
                    if selectors:
                        raise _E("round result must not consume selectors")
                    terminal = True
                else:
                    raise _E("unknown or misplaced history event")
                if [p.riichi for p in after.views[0].players] != allowed_riichi:
                    raise _E(
                        "public riichi changed without its declaration/establishment event"
                    )
                _board_change(before, after, event, kan_action)
                expected = tuple(expected)
            if after.contexts != expected:
                raise _E(
                    "checkpoint context differs from replayed missed/riichi/ippatsu state"
                )
            states.append(after)
            before = after
        if (
            reactions != expected_reactions
            or sum(map(len, commit_selectors)) != expected_selectors
        ):
            raise _E(
                "reaction/selector coverage differs from independent runner counts"
            )
        checkpoints[key] = tuple(states)
        selector_commits[key] = tuple(commit_selectors)
        offset += length
    if offset != len(rows):
        raise _E("history has uncovered rows")
    return checkpoints, selector_commits, coverage_seeds


def _replay(rows, coverage):
    """Group ordered semantic steps into actual committed transaction boundaries.

    A reaction and riichi establishment can occur in one engine commit. The
    intermediate checkpoints are verification evidence, never selector snapshots.
    Producer hooks needed to obtain them belong to the later Arena work.
    """
    expanded, expanded_coverage = [], []
    boundaries, grouped_selectors = {}, {}
    offset = 0
    for item in expect_list(coverage, _E, "coverage"):
        count = _object(
            item, "seed round_id transitions reactions decisions selectors", "coverage"
        )
        seed = _integer(count["seed"], "seed")
        round_id = expect_str(count["round_id"], _E, "round_id")
        length = _integer(count["transitions"], "transitions")
        if length < 2 or offset + length > len(rows):
            raise _E("committed transaction coverage is incomplete")
        limits, sequences = [], []
        index = 0
        for commit, value in enumerate(rows[offset : offset + length]):
            raw = _object(value, "schema seed round_id index steps", "history commit")
            if (
                raw["schema"] != HISTORY_SCHEMA
                or _integer(raw["seed"], "seed") != seed
                or raw["round_id"] != round_id
                or _integer(raw["index"], "index") != commit
            ):
                raise _E("committed history is not contiguous")
            steps = expect_list(raw["steps"], _E, "steps")
            if not steps or (commit in (0, length - 1) and len(steps) != 1):
                raise _E("commit needs steps and start/end must be single checkpoints")
            selectors = []
            for step in steps:
                step = _object(
                    step, "selector_sequences event checkpoint", "history step"
                )
                selectors.extend(
                    expect_list(step["selector_sequences"], _E, "selector_sequences")
                )
                expanded.append(
                    {
                        "schema": HISTORY_SCHEMA,
                        "seed": seed,
                        "round_id": round_id,
                        "index": index,
                        **step,
                    }
                )
                index += 1
            if sum(bool(step["selector_sequences"]) for step in steps) > 1:
                raise _E("a transaction cannot consume multiple selector windows")
            limits.append(index - 1)
            sequences.append(tuple(selectors))
        key = seed, round_id
        if key in boundaries:
            raise _E("duplicate coverage round")
        boundaries[key] = limits
        grouped_selectors[key] = tuple(sequences)
        expanded_coverage.append({**count, "transitions": index})
        offset += length
    if offset != len(rows):
        raise _E("history has transactions outside coverage")
    checkpoints, _, seeds = _replay_steps(expanded, expanded_coverage)
    return (
        {
            key: tuple(checkpoints[key][index] for index in limits)
            for key, limits in boundaries.items()
        },
        grouped_selectors,
        seeds,
    )


def read_ron_source(directory: str | Path, *, base_directory: str | Path) -> RonSource:
    """Check the entire extension, including historical legal opportunities.

    Missingness is checked on every row, even for non-tenpai and furiten hands.
    Reaction verification requires the scoring backend. No sampling/drop path.
    """
    try:
        return _read_ron_source(Path(directory), Path(base_directory))
    except base.HandBeliefSourceError as error:
        raise _E(str(error)) from error


def _read_ron_source(root, original):
    manifest, decisions = base.read_decisions(original)
    hands = tuple(
        base._parse_facts(value, "hand fact")
        for value in _rows(
            original, base.HAND_FACTS_FILENAME, manifest.files["hand_facts"]
        )
    )
    base.label_decisions(decisions, hands)
    extension = _manifest(root, original)
    if extension["producer"] != manifest.producer or extension["splits"] != {
        name: list(values) for name, values in manifest.splits.items()
    }:
        raise _E("producer revisions or exact splits differ from base source")
    snapshots = tuple(
        _snapshot(value)
        for value in _rows(root, FACTS_FILENAME, extension["files"]["ron_facts"])
    )
    by_snapshot = base._unique_by_key(snapshots, "ron fact")
    if set(by_snapshot) != {decision.key for decision in decisions}:
        raise _E("ron fact keys differ from the complete base population")
    rows = _rows(root, HISTORY_FILENAME, extension["files"]["ron_history"])
    states, selectors, seeds = _replay(rows, extension["coverage"])
    if seeds != {seed for values in manifest.splits.values() for seed in values}:
        raise _E("round coverage does not cover the exact declared seeds")
    by_hand = {hand.key: hand for hand in hands}
    joined = []
    per_round_decisions = Counter()
    seen_commits = set()
    for decision in sorted(decisions, key=lambda item: item.key):
        snapshot = by_snapshot[decision.key]
        round_key = decision.key.seed, snapshot.round_id
        boundary = snapshot.history_boundary
        if round_key not in states or not 1 <= boundary < len(states[round_key]) - 1:
            raise _E("snapshot boundary is outside a started, unfinished round")
        checkpoint = states[round_key][boundary - 1]
        next_selectors = selectors[round_key][boundary]
        if next_selectors != (decision.key.sequence,):
            raise _E("snapshot is not immediately before its selector commit")
        commit_key = round_key, boundary
        if commit_key in seen_commits:
            raise _E("more than one discard decision references the same commit")
        seen_commits.add(commit_key)
        # Compare the selected action as well as its sequence to the journal.
        offset = sum(
            len(value)
            for key, value in states.items()
            if list(states).index(key) < list(states).index(round_key)
        )
        event = rows[offset + boundary]["steps"][0]["event"]
        if (
            event["kind"] != "progress"
            or parse_action(event["action"], _E, "snapshot commit action")
            != decision.selected_action
        ):
            raise _E("snapshot selector action differs from v1")
        if checkpoint.views[decision.key.seat] != decision.policy_input:
            raise _E("journal checkpoint differs from the exact player-safe snapshot")
        hand = by_hand[decision.key]
        for position, opponent in enumerate(hand.opponents):
            seat = opponent.seat
            if (
                Counter(_hand(checkpoint.views[seat]))
                != Counter(opponent.concealed_tiles)
                or checkpoint.views[seat].own_hand.drawn_tile is not None
            ):
                raise _E("journal hand differs from the stable v1 opponent hand")
            if checkpoint.contexts[seat] != snapshot.opponents[position]:
                raise _E("ron fact context differs from the half-open replay prefix")
        joined.append(checkpoint)
        per_round_decisions[round_key] += 1
    for count in extension["coverage"]:
        if per_round_decisions[count["seed"], count["round_id"]] != count["decisions"]:
            raise _E("decision coverage differs from the independent runner count")
    ordered = tuple(sorted(decisions, key=lambda item: item.key))
    return RonSource(
        manifest,
        ordered,
        tuple(by_hand[d.key] for d in ordered),
        tuple(by_snapshot[d.key] for d in ordered),
        tuple(joined),
    )


def read_labelled_ron_source(
    directory: str | Path, *, base_directory: str | Path
) -> tuple[base.HandBeliefManifest, tuple[base.HandBeliefLabelledDecision, ...]]:
    """Materialize complete structural + ron truth; never return partial labels."""
    source = read_ron_source(directory, base_directory=base_directory)
    require_scoring_backend()
    labelled = []
    for decision, hand, checkpoint in zip(
        source.decisions, source.hands, source.checkpoints
    ):
        opponents = []
        for opponent in hand.opponents:
            seat = opponent.seat
            view = checkpoint.views[seat]
            truth = exact_hand_belief_with_ron(
                opponent.concealed_tiles,
                opponent.melds,
                tuple(discard.tile for discard in view.players[seat].discards),
                checkpoint.contexts[seat],
                seat_wind=tuple(Wind)[(seat - int(view.round.dealer_seat)) % 4],
                prevailing_wind=view.round.round_wind,
            )
            opponents.append(base.OpponentTruth(seat, truth))
        labelled.append(base.HandBeliefLabelledDecision(decision, tuple(opponents)))
    return source.base_manifest, tuple(labelled)
