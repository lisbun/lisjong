"""聴牌PUSH/FOLD（H2）の比較式`V`と、その表の形（lisbun/lisjong#288、推論側）。

設計は`docs/tenpai-push-fold-design.md`の4節。表（`q`・`R_T`・`R_F`・`L`・`U`）は
`lisjong.learning.tenpai_push_fold_evaluation`が対比較sourceのtrainから作る。このmoduleは
表の形と、表を使った`V`の計算だけを持ち、sourceも正解も読まない。

```text
V(a) = − p(a) · L  +  (1 − p(a)) · C(a)
C(a) = q_ron · G_ron(a) + q_tsumo · G_tsumo(a) + R_T      a が聴牌を保つ
     = R_F                                                a が聴牌を崩す、または(A)の降りる側
```

`V`は表と仮定に依存する比較用の値であり、期待収支ではない。

bucketのキー（境界は`BucketSpec`で与える。この値はまだ事前登録していない）:

- 残りツモ山は3区分（`w0`〜`w2`）
- 役がある待ちの残り枚数は、0枚の`c0`と、1枚以上の3区分（`c1`〜`c3`）
- `q_ron`はロンの枚数、`q_tsumo`はツモの枚数の区分で引く（`c0`は表を引かず`q = 0`）
- `R_T`はツモの枚数の区分で引く（`c0`を含む。役なしの聴牌にも流局時の聴牌料がある）
- `R_F`は残りツモ山の区分だけで引く
"""

from dataclasses import dataclass, field

from lisjong.belief.fixed_point import raw_to_semantic
from lisjong.learning.tenpai_push_fold import GateKind, OwnWaitValue, TenpaiGate
from lisjong.policy_contract.policy_input import PolicyInput

RIICHI_GROUP = "riichi"
"""(A)の表。押す側はリーチした局、降りる側はリーチせずに`a_fold`を切った局で作る。"""
DISCARD_GROUP = "discard"
"""(B)(C)の表。"""
GROUPS = (RIICHI_GROUP, DISCARD_GROUP)

DEALER = "dealer"
NON_DEALER = "non_dealer"

_STICK_POINTS = 1000
_HONBA_POINTS = 300


@dataclass(frozen=True, slots=True)
class BucketSpec:
    """bucketの境界。各組は昇順の上限値で、最後の区分は上限なし。"""

    wall_upper_bounds: tuple[int, int]
    count_upper_bounds: tuple[int, int]

    def __post_init__(self) -> None:
        for name in ("wall_upper_bounds", "count_upper_bounds"):
            bounds = tuple(getattr(self, name))
            if len(bounds) != 2 or any(type(bound) is not int for bound in bounds):
                raise TypeError(f"{name} must be two ints")
            if not bounds[0] < bounds[1]:
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, bounds)
        if self.wall_upper_bounds[0] < 0 or self.count_upper_bounds[0] < 1:
            raise ValueError("bucket bounds are out of range")

    def wall_key(self, live_wall_tiles_remaining: int) -> str:
        return f"w{_bin(live_wall_tiles_remaining, self.wall_upper_bounds)}"

    def count_bin(self, count: int) -> int:
        """0枚は0、1枚以上は1〜3。"""
        return 0 if count <= 0 else 1 + _bin(count, self.count_upper_bounds)

    def tenpai_key(self, live_wall_tiles_remaining: int, count: int) -> str:
        return f"{self.wall_key(live_wall_tiles_remaining)}.c{self.count_bin(count)}"


def _bin(value: int, upper_bounds: tuple[int, int]) -> int:
    return sum(value > bound for bound in upper_bounds)


@dataclass(frozen=True, slots=True)
class TenpaiSideTable:
    """聴牌を保って今の打牌が通った後の表。キーは`BucketSpec.tenpai_key()`。"""

    q_ron: dict[str, float] = field(default_factory=dict)
    q_tsumo: dict[str, float] = field(default_factory=dict)
    r_t: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PushFoldTables:
    """比較式の表。supportが足りずに除外したbucketはキーを持たない。

    `loss`はリーチ者が親か子か（`DEALER` / `NON_DEALER`）、`tenpai`・`r_f`・`uplift`の
    外側のキーは`GROUPS`、`r_f`の内側は`BucketSpec.wall_key()`、`uplift`（(A)の`U`）は
    `RIICHI_GROUP`だけが持ち、内側は`"ron"` / `"tsumo"`である。
    """

    buckets: BucketSpec
    loss: dict[str, float]
    tenpai: dict[str, TenpaiSideTable]
    r_f: dict[str, dict[str, float]]
    uplift: dict[str, float]

    def to_value(self) -> dict[str, object]:
        return {
            "buckets": {
                "count_upper_bounds": list(self.buckets.count_upper_bounds),
                "wall_upper_bounds": list(self.buckets.wall_upper_bounds),
            },
            "loss": dict(self.loss),
            "r_f": {group: dict(table) for group, table in self.r_f.items()},
            "tenpai": {
                group: {
                    "q_ron": dict(table.q_ron),
                    "q_tsumo": dict(table.q_tsumo),
                    "r_t": dict(table.r_t),
                }
                for group, table in self.tenpai.items()
            },
            "uplift": dict(self.uplift),
        }


def gate_group(kind: GateKind) -> str:
    return RIICHI_GROUP if kind is GateKind.RIICHI else DISCARD_GROUP


def comparison_values(
    policy_input: PolicyInput, gate: TenpaiGate, tables: PushFoldTables
) -> tuple[float, float] | None:
    """`(V(押す), V(降りる))`を返す。必要なbucketが表になければ`None`（ゲート条件8）。

    (A)の押す側は`gate.push_wait`（リーチ役を含む裏ドラ除外モード）に上乗せ`U`を足す。
    自分のリーチ棒は`G`に入れない。降りる側は、(B)(C)で`a_fold`が聴牌を保つ場合だけ
    聴牌側の式を使い、それ以外は`R_F`を使う。
    """
    round_state = policy_input.round
    wall = round_state.live_wall_tiles_remaining
    honba = _HONBA_POINTS * round_state.honba
    pot = _STICK_POINTS * round_state.riichi_sticks + honba
    group = gate_group(gate.kind)

    riichi_is_dealer = gate.riichi_seat is round_state.dealer_seat
    loss = tables.loss.get(DEALER if riichi_is_dealer else NON_DEALER)
    push = _continuation(tables, group, wall, gate.push_wait, pot)
    if gate.fold_wait is not None:
        fold = _continuation(tables, DISCARD_GROUP, wall, gate.fold_wait, pot)
    else:
        fold = tables.r_f.get(group, {}).get(tables.buckets.wall_key(wall))
    if loss is None or push is None or fold is None:
        return None

    def value(ron_legal_raw: int, continuation: float) -> float:
        p = raw_to_semantic(ron_legal_raw)
        return -p * (loss + honba) + (1 - p) * continuation

    return (
        value(gate.push_ron_legal_raw, push),
        value(gate.fold_ron_legal_raw, fold),
    )


def _continuation(
    tables: PushFoldTables, group: str, wall: int, wait: OwnWaitValue, pot: int
) -> float | None:
    table = tables.tenpai.get(group)
    if table is None:
        return None
    buckets = tables.buckets
    total = table.r_t.get(buckets.tenpai_key(wall, wait.tsumo_count))
    if total is None:
        return None
    for method, count, points, q_table in (
        ("ron", wait.ron_count, wait.ron_points, table.q_ron),
        ("tsumo", wait.tsumo_count, wait.tsumo_points, table.q_tsumo),
    ):
        if count <= 0:
            continue
        q = q_table.get(buckets.tenpai_key(wall, count))
        uplift = tables.uplift.get(method) if group == RIICHI_GROUP else 0.0
        if q is None or uplift is None:
            return None
        total += q * (points + uplift + pot)
    return total


__all__ = [
    "DEALER",
    "DISCARD_GROUP",
    "GROUPS",
    "NON_DEALER",
    "RIICHI_GROUP",
    "BucketSpec",
    "PushFoldTables",
    "TenpaiSideTable",
    "comparison_values",
    "gate_group",
]
