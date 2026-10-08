"""他家のロン合法確率の推定（lisbun/lisjong#277、推論側）。

学習済みの構造的待ち推定器の出力に、公開情報だけで確実にロン不可と言える牌種を0にする
変換を掛ける。入力は`PolicyInput`だけで、正解・見逃し状態・履歴は渡さない。新しい学習と
確率的な補正は行わない。出力は同じ判断・対象席のwaitとronを組にした`HandBelief`で、
ronはwaitと同じrawか0なので、丸め後も`ron <= wait`になる。

0にする牌種（`project-standard-normal-discard-ron-v1`の下で確定するもの）。

1. 対象席の河にある牌種（現物）。その牌が待ちなら捨て牌フリテン
2. 対象席の最後の打牌より後に他家（観測者を含む）が切った牌種。対象席はその間ツモも副露も
   していないので手牌は同じ。待ちでロン合法だったなら見逃しフリテン、役なしならその牌は
   今もロン不可。対象席の打牌がまだない場合は、配牌から手牌が変わっていないので、他家の
   打牌すべてが該当する

赤5と通常5は同じ牌種として扱う。リーチ成立後に通った牌の全体、河にある別の待ち牌による
フリテン、副露者の役なしは`PolicyInput`から確定できないので0にせず、過大評価として残る。

提供範囲は待ち推定器と同じにする。範囲外は`None`（未提供）を返し、ゼロ予測や別の待ち
モデルへ置き換えない。リーチ者は#245（`riichi_wait_estimator`）、副露者は#259範囲1
（`open_wait_estimator`）のモデルを使い、門前非リーチは対象外である。
"""

from dataclasses import replace

from lisjong.belief.canonical_axes import tile_type_index
from lisjong.belief.hand_belief import HandBelief
from lisjong.learning.open_wait_estimator import (
    OpenWaitModel,
    estimate_open_wait_belief,
    is_open_opponent,
)
from lisjong.learning.riichi_wait_estimator import (
    LogisticWaitModel,
    estimate_riichi_wait_belief,
)
from lisjong.policy_contract import PolicyInput, RiichiState, Seat

TRANSFORM = "certain-ron-illegal-zero-v1"
RIICHI_SCOPE = "single-opponent-riichi.self-not-riichi.riichi-seat-has-discard.v1"
OPEN_SCOPE = "open-non-riichi-opponent.v1"


def certain_ron_illegal_tile_types(
    policy_input: PolicyInput, seat: Seat
) -> frozenset[int]:
    """`seat`が確実にロンできない牌種（canonical index）。`PolicyInput`だけから決まる。"""
    if not isinstance(policy_input, PolicyInput) or not isinstance(seat, Seat):
        raise TypeError("a PolicyInput and Seat are required")
    if seat is policy_input.self_seat:
        raise ValueError("ron legality targets opponents only")
    own = policy_input.players[seat].discards
    last_order = max((discard.order for discard in own), default=-1)
    return frozenset(
        tile_type_index(discard.tile.tile_type)
        for other in Seat
        for discard in policy_input.players[other].discards
        if other is seat or discard.order > last_order
    )


def with_certain_zero(
    belief: HandBelief, policy_input: PolicyInput, seat: Seat
) -> HandBelief:
    """waitを変えず、確実にロン不可の牌種だけを0にしたronを組にして返す。"""
    if belief.wait_probability_raw is None:
        raise ValueError("ron legality requires a provided wait")
    zero = certain_ron_illegal_tile_types(policy_input, seat)
    return replace(
        belief,
        ron_legal_probability_raw=tuple(
            0 if index in zero else raw
            for index, raw in enumerate(belief.wait_probability_raw)
        ),
    )


def in_riichi_scope(policy_input: PolicyInput, seat: Seat) -> bool:
    """`riichi_wait_estimator`の提供範囲（観測者は非リーチ、`seat`が唯一のリーチ者）か。"""
    if not isinstance(policy_input, PolicyInput) or not isinstance(seat, Seat):
        raise TypeError("a PolicyInput and Seat are required")
    if seat is policy_input.self_seat:
        raise ValueError("ron legality targets opponents only")
    players = policy_input.players
    return (
        players[policy_input.self_seat].riichi is RiichiState.NONE
        and all(
            (player.riichi is not RiichiState.NONE) is (other is seat)
            for other, player in zip(Seat, players)
            if other is not policy_input.self_seat
        )
        and bool(players[seat].discards)
    )


def estimate_riichi_ron_legal_belief(
    policy_input: PolicyInput, seat: Seat, model: LogisticWaitModel
) -> HandBelief | None:
    """リーチ者`seat`のwaitとronを組にした`HandBelief`。提供範囲外は`None`。"""
    if not isinstance(model, LogisticWaitModel):
        raise TypeError("a LogisticWaitModel is required")
    if not in_riichi_scope(policy_input, seat):
        return None
    return with_certain_zero(
        estimate_riichi_wait_belief(policy_input, model), policy_input, seat
    )


def estimate_open_ron_legal_belief(
    policy_input: PolicyInput, seat: Seat, model: OpenWaitModel
) -> HandBelief | None:
    """副露者`seat`のwaitとronを組にした`HandBelief`。提供範囲外は`None`。

    提供範囲は`open_wait_estimator`と同じ（リーチしておらず暗槓以外の副露がある他家）。
    """
    if not isinstance(model, OpenWaitModel):
        raise TypeError("an OpenWaitModel is required")
    if not isinstance(policy_input, PolicyInput) or not isinstance(seat, Seat):
        raise TypeError("a PolicyInput and Seat are required")
    if seat is policy_input.self_seat:
        raise ValueError("ron legality targets opponents only")
    if not is_open_opponent(policy_input, int(seat)):
        return None
    return with_certain_zero(
        estimate_open_wait_belief(policy_input, int(seat), model), policy_input, seat
    )


__all__ = [
    "OPEN_SCOPE",
    "RIICHI_SCOPE",
    "TRANSFORM",
    "certain_ron_illegal_tile_types",
    "estimate_open_ron_legal_belief",
    "estimate_riichi_ron_legal_belief",
    "in_riichi_scope",
    "with_certain_zero",
]
