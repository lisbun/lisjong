"""lisjong内部の固定4人麻雀seat値。

docs/policy-input-schema.md「Seat」の意味契約を実装する。

Seatは固定のplayer座席位置（0..3）であり、場風・自風（Wind）ではない。
RiichiEnvのplayer_idと数値が一致する場合があっても、RiichiEnv固有identityを
公開する意味ではない。外部seatとの対応付けはRiichiEnv AdapterまたはRiichiLab
Client側の境界が所有する。

WindはSeatと異なりint値と比較できない別型（lisjong.policy_contract.wind.Wind）
として表現し、固定player座席位置と場風・自風を混同しない。
"""

from enum import IntEnum


class Seat(IntEnum):
    """0..3の固定seat位置。

    (seat + 1) mod 4 = 下家 の関係を持つ。自風はdealer_seatとの相対位置から
    別途導出し、Seat自体には保持しない。

    IntEnumを採用しているため、Seat.SEAT_0 == 0 のようにint値と直接比較できる。
    これはdocsが(seat + 1) mod 4という算術でSeatの契約を表現していることに
    合わせた意図的な性質であり、見落としではない。一方、Windはこの性質を
    意図的に持たない別型として表現し、Seatとの混同を防ぐ（wind.pyを参照）。
    """

    SEAT_0 = 0
    SEAT_1 = 1
    SEAT_2 = 2
    SEAT_3 = 3
