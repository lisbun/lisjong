"""Learning L0.2 candidate scorer testで共有するfixture builder。

`learning_fixtures`のsource record writerを再利用し、candidate featureが
materializeできる通常打牌decision（14枚の純手牌）と、O0 guard対象のdecision
（和了 / 即リーチ / response）を持つsource recordを組み立てる。

teacher labelはfixture側で既存`TwoStepUkeirePolicy`から作る。production側の
dataset codeはteacherを再実行しない。
"""

import learning_fixtures as fixtures

from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import (
    AnkanAction,
    DecisionContext,
    Discard,
    DiscardAction,
    OwnHandState,
    PassAction,
    PlayerPublicState,
    PolicyInput,
    PonAction,
    RiichiAction,
    RiichiState,
    RonAction,
    Seat,
    Tile,
    TileCategory,
    TileType,
    TsumoAction,
)

_CATEGORIES = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}

FAR_HAND = "1379m2468p1357s15z"
"""最小向聴が1以上で、second-step finalistが複数残る打牌decision。"""

NEAR_HAND = "123m456p789s1123z5p"
"""最小向聴1で、second-step finalistが2つ残る打牌decision。"""

TENPAI_REACHABLE_HAND = "340m5m456p789s1122z"
"""打牌後聴牌（second-step NOT_APPLICABLE）へ届き、赤5と通常5が両方legalな打牌decision。"""


def hand(spec):
    """`123m456p`形式のhand specをTile tupleへ変換する。`0`は赤5である。"""
    tiles = []
    ranks = ""
    for character in spec:
        if character.isdigit():
            ranks += character
            continue
        category = _CATEGORIES[character]
        for rank_character in ranks:
            rank = int(rank_character)
            tiles.append(
                Tile(TileType(category, 5 if rank == 0 else rank), is_red=rank == 0)
            )
        ranks = ""
    if ranks:
        raise ValueError(f"hand spec has trailing ranks: {spec!r}")
    return tuple(tiles)


def policy_input(spec, *, self_seat=Seat.SEAT_0, drawn=True, riichi=False):
    """seat相対で公開情報の少ないPolicyInputを作る。

    `drawn=True`なら純手牌の最後の牌をツモ牌とする（14枚のown-turn）。
    `drawn=False`なら13枚のresponse局面になる。
    """
    concealed = hand(spec)
    players = tuple(
        PlayerPublicState(
            score=25000,
            discards=(),
            melds=(),
            riichi=RiichiState.ACCEPTED
            if riichi and index == int(self_seat)
            else RiichiState.NONE,
        )
        for index in range(4)
    )
    return PolicyInput(
        self_seat=Seat(self_seat),
        round=fixtures.round_state(),
        players=players,
        own_hand=OwnHandState(
            concealed_tiles=concealed, drawn_tile=concealed[-1] if drawn else None
        ),
    )


def discard_actions(value, *, tsumogiri_only=False):
    """純手牌の牌identityごとの手出しと、ツモ牌のツモ切りを作る。"""
    seat = value.self_seat
    drawn = value.own_hand.drawn_tile
    if tsumogiri_only:
        return (DiscardAction(actor=seat, tile=drawn, tsumogiri=True),)
    tiles = list(value.own_hand.concealed_tiles)
    if drawn is not None:
        tiles.remove(drawn)
    actions = [
        DiscardAction(actor=seat, tile=tile, tsumogiri=False)
        for tile in dict.fromkeys(tiles)
    ]
    if drawn is not None:
        actions.append(DiscardAction(actor=seat, tile=drawn, tsumogiri=True))
    return tuple(actions)


def discard_decision(spec=FAR_HAND, *, extra=(), self_seat=Seat.SEAT_0, **options):
    value = policy_input(spec, self_seat=self_seat)
    return DecisionContext(
        input=value, legal_actions=discard_actions(value, **options) + tuple(extra)
    )


def riichi_decision(spec=NEAR_HAND, *, self_seat=Seat.SEAT_0):
    value = policy_input(spec, self_seat=self_seat)
    return DecisionContext(
        input=value,
        legal_actions=discard_actions(value) + (RiichiAction(actor=value.self_seat),),
    )


def tsumo_decision(spec=NEAR_HAND, *, self_seat=Seat.SEAT_0, riichi=False):
    value = policy_input(spec, self_seat=self_seat)
    return DecisionContext(
        input=value,
        legal_actions=discard_actions(value)
        + (
            RiichiAction(actor=value.self_seat),
            TsumoAction(actor=value.self_seat, winning_tile=value.own_hand.drawn_tile),
        ),
    )


def response_decision(spec="123m456p789s1122z", *, self_seat=Seat.SEAT_0):
    value = policy_input(spec, self_seat=self_seat, drawn=False)
    seat = value.self_seat
    target = Seat((int(seat) + 3) % 4)
    tile = Tile(TileType(TileCategory.HONOR, 1))
    return DecisionContext(
        input=value,
        legal_actions=(
            PassAction(actor=seat),
            PonAction(
                actor=seat, target=target, called_tile=tile, consumed_tiles=(tile, tile)
            ),
        ),
    )


def ron_response_decision(spec="123m456p789s1122z", *, self_seat=Seat.SEAT_0):
    value = policy_input(spec, self_seat=self_seat, drawn=False)
    seat = value.self_seat
    target = Seat((int(seat) + 3) % 4)
    tile = Tile(TileType(TileCategory.HONOR, 2))
    return DecisionContext(
        input=value,
        legal_actions=(
            PassAction(actor=seat),
            RonAction(actor=seat, target=target, winning_tile=tile),
        ),
    )


def ankan_decision(spec="1111m456p789s122z", *, self_seat=Seat.SEAT_0):
    value = policy_input(spec + "3z", self_seat=self_seat)
    tile = Tile(TileType(TileCategory.MANZU, 1))
    return DecisionContext(
        input=value,
        legal_actions=discard_actions(value)
        + (AnkanAction(actor=value.self_seat, tiles=(tile, tile, tile, tile)),),
    )


def teacher_action(decision):
    return TwoStepUkeirePolicy().choose_action(decision)


def source_rows(decisions, *, game_ordinal, seed, split, teacher=teacher_action):
    """decision列を、1 decision = 1 stepのsource rowへ変換する。"""
    return [
        fixtures.source_row(
            game_ordinal=game_ordinal,
            seed=seed,
            split=split,
            step_ordinal=ordinal,
            decision_ordinal=ordinal,
            policy_input_value=decision.input,
            legal_actions=decision.legal_actions,
            selected_action=teacher(decision),
        )
        for ordinal, decision in enumerate(decisions)
    ]


SECOND_NEAR_HAND = "2233m456p789s1359s"
"""別のcandidate数でsecond-step finalistを持つ、軽量な打牌decision。"""


def mixed_decisions():
    """scorer対象とO0 guard対象を混ぜた1 hanchan分のdecision列。

    dataset / diagnostics testの実行時間を抑えるため、finalist評価が重い
    FAR_HANDは含めない。
    """
    return (
        discard_decision(SECOND_NEAR_HAND),
        tsumo_decision(),
        discard_decision(NEAR_HAND),
        riichi_decision(),
        response_decision(),
        discard_decision(TENPAI_REACHABLE_HAND),
        ron_response_decision(),
    )


def write_candidate_source_record(root, *, splits=("TRAIN", "SELECT", "OFFLINE-EVAL")):
    """splitごとに1 hanchanのmixed decision列を持つsource recordを書く。"""
    games = []
    for game_ordinal, split in enumerate(splits):
        seed = 100 * (game_ordinal + 1)
        games.append(
            (
                split,
                seed,
                source_rows(
                    mixed_decisions(), game_ordinal=game_ordinal, seed=seed, split=split
                ),
            )
        )
    return fixtures.write_source_record(root, games)


def discard_with_history(spec=FAR_HAND):
    """公開捨て牌を持つ打牌decision（shared contextが変化することを確認する用）。"""
    value = policy_input(spec)
    players = list(value.players)
    players[1] = PlayerPublicState(
        score=25000,
        discards=(
            Discard(
                tile=Tile(TileType(TileCategory.HONOR, 7)),
                tsumogiri=False,
                order=0,
                called_by=None,
            ),
        ),
        melds=(),
        riichi=RiichiState.NONE,
    )
    changed = PolicyInput(
        self_seat=value.self_seat,
        round=value.round,
        players=tuple(players),
        own_hand=value.own_hand,
    )
    return DecisionContext(input=changed, legal_actions=discard_actions(changed))
