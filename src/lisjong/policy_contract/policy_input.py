"""lisjong内部のPolicyInput型。

docs/policy-input-schema.md「PolicyInputの概念schema」の意味契約を実装する。

PolicyInputは、あるseatから観測可能な状態を正規化したsnapshotである。
Policyの合法手候補（legal_actions）は含まない。legal_actionsを含む
1 seat・1 decision分の不変contextは、`DecisionContext`がPolicyInputと
decision-localなlegal_actionsを束ねて表現する。

`players`はindex自体がSeat identityを表す
（`players[index]のindex == Seat`）。`PlayerPublicState`自体はseat fieldを
持たないため、この対応は境界側（RiichiEnv Adapter等）がPolicyInputを構築
する際の責務である。PolicyInputはこの値単体で「players[2]が本当に
Seat.SEAT_2の状態か」を検証できない。`PlayerPublicState`側に照合できる
独立したseat identityが存在しないためである。self_seatをindex 0へ
rotateする、`PlayerPublicState`へseatを追加する、
`dict[Seat, PlayerPublicState]`へ変更するといった代替表現は採用しない。
"""

from dataclasses import dataclass

from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat


def _normalize_players(
    values: object,
) -> tuple[PlayerPublicState, PlayerPublicState, PlayerPublicState, PlayerPublicState]:
    """iterableを4要素固定のtupleへ正規化する。並び順は一切変更しない。

    `players`は「seat 0から3の順」という意味を持つsequenceであり、
    canonical sortやself_seat基準のrotateを行うと、位置自体が表す
    Seat identityを破壊する。
    """
    try:
        items = tuple(values)
    except TypeError:
        raise TypeError("players must be an iterable") from None
    if any(not isinstance(item, PlayerPublicState) for item in items):
        raise TypeError("players must contain only PlayerPublicState instances")
    if len(items) != 4:
        raise ValueError("players must contain exactly 4 PlayerPublicState")

    return items


@dataclass(frozen=True, slots=True)
class PolicyInput:
    """1 seatから観測可能な状態のsnapshot。

    ```text
    PolicyInput
    ├── self_seat
    ├── round
    ├── players
    └── own_hand
    ```

    `own_hand`（自席の非公開手牌）と`players[self_seat]`（自席の公開状態）は
    並存させるだけで、両者の深い整合（牌枚数、tile conservation等）は
    検証しない。正本文書がその具体的な整合規則をまだ定義しておらず、決め打ちで
    検証を実装すると、Adapterがsnapshotを正しく構築する責務との重複、および
    physical tile conservationの検証（Policy契約の責務外）への接近を招く。

    同様に、`round`のround_wind/dealer_seat等から局進行を再検証する、
    discards.orderの4 seat全体での一意性・連番性を検証する、riichi_sticksと
    各playerのscore/riichiを相互整合させる、といった、計算可能ではあるが
    この値の構造そのものが保証すべき条件ではないcross-field検証は行わない。
    """

    self_seat: Seat
    round: RoundState
    players: tuple[
        PlayerPublicState, PlayerPublicState, PlayerPublicState, PlayerPublicState
    ]
    own_hand: OwnHandState

    def __post_init__(self) -> None:
        if not isinstance(self.self_seat, Seat):
            raise TypeError("self_seat must be a Seat")

        if not isinstance(self.round, RoundState):
            raise TypeError("round must be a RoundState")

        players = _normalize_players(self.players)

        if not isinstance(self.own_hand, OwnHandState):
            raise TypeError("own_hand must be an OwnHandState")

        object.__setattr__(self, "players", players)
