"""kobalab/majiang-ai legacy 0004の打牌評価（牌効率）の共通処理（Issue #226）。

`Kobalab0004ReferencePolicy`（Issue #211）の打牌選択から、特定Policyに依存しない
部分を切り出したdecision-localの内部処理である。公開仕様の出典・位置付けは
参照版と同じで、公開されている`kobalab/majiang-ai`の`legacy/player-0004.js` /
`legacy/suanpai-0004.js`（MIT License, Copyright (c) Satoshi Kobayashi,
commit e75a9720a12b84c03e6c61c3960c1844b8982eb4）の`select_dapai()` /
`SuanPai.paijia()`を、lisjongのexact shantenへ適用した移植である。upstreamとの
全局面での行動同値性は主張しない（詳細は`docs/kobalab-0004-reference.md`）。

処理は次の3層に分かれる。

- 構造評価 `_DiscardStructures`: 打牌後向聴数と形としての改善牌（未見枚数0も
  含む）。牌姿だけに依存し、未見枚数・ドラ・Beliefには依存しない。
- 枚数集計 `_PublicCounts` / `_PaijiaInput`: 同じdecisionの公開情報から
  再構成した未見枚数と、paijia（ドラ・赤牌・役牌等を含むヒューリスティックな
  牌価）の入力snapshot。仮に捨てる牌を未見牌へ戻さない。
- 候補選択 `_evaluation_order()` / `_choose_reference_discard()`: paijiaと
  source順による決定的な評価順で、受入（未見枚数の合計）が厳密に増えた場合だけ
  選択を更新する。

参照版の選択（`_choose_reference_discard()`）は、打牌前向聴数以下の候補が
なければ評価順の先頭を返すsourceの初期値fallbackを含む。渡された候補集合の外の
Actionを返さない。

このmoduleはPolicyのclassや判断規則（立直・槓・和了・九種九牌・鳴き）を
持たず、特定Policyへ依存しない。instanceはdecision-localで、snapshotを跨ぐ
状態を持たない。向聴数の構造評価は`evaluate_discards_from_canonical_counts()`
（Issue #224）を通じて既存backend境界で行い、`_lisjong_native`を直接importしない。
"""

from collections import Counter
from collections.abc import Sequence

from lisjong.belief import (
    derive_remaining_tile_inventory,
    red_five_index,
    sum_tile_type_values,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
from lisjong.hand_evaluation import calculate_shanten
from lisjong.hand_evaluation.shanten import evaluate_discards_from_canonical_counts
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

_MAX_COPIES_PER_TILE_TYPE = 4

_CATEGORY_ORDER = (
    TileCategory.MANZU,
    TileCategory.PINZU,
    TileCategory.SOUZU,
    TileCategory.HONOR,
)
"""sourceの`['m','p','s','z']`列挙順。"""

_ALL_TILE_TYPES: tuple[TileType, ...] = tuple(
    TileType(category, rank)
    for category in _CATEGORY_ORDER
    for rank in range(1, (7 if category is TileCategory.HONOR else 9) + 1)
)

_WIND_RANKS = 4
_DRAGON_START_RANK = 5
_DRAGON_RANKS = 3
_SUITED_MAXIMUM_RANK = 9


class Kobalab0004ReferencePolicyError(Exception):
    """入力がreference semanticsの前提と整合せず、fail closedする場合。

    0004の打牌評価を使うPolicy（参照版・Belief版）が共有する。
    """


def _dora_tile_type(indicator: TileType) -> TileType:
    """sourceの`Majiang.Shan.zhenbaopai()`に対応する表示牌 -> ドラ牌種。"""
    if indicator.category is TileCategory.HONOR:
        if indicator.rank <= _WIND_RANKS:
            return TileType(TileCategory.HONOR, indicator.rank % _WIND_RANKS + 1)
        offset = indicator.rank - _DRAGON_START_RANK
        return TileType(
            TileCategory.HONOR, (offset + 1) % _DRAGON_RANKS + _DRAGON_START_RANK
        )
    return TileType(indicator.category, indicator.rank % _SUITED_MAXIMUM_RANK + 1)


class _PaijiaInput:
    """sourceの`SuanPai.paijia()`（牌の評価値）の式と、その入力snapshot。

    式は入力の導出から分離している。`tile_counts`（canonical 34牌種index、
    赤5を含む）と`red_five_counts`（萬子・筒子・索子の赤5）は、同じ非負整数の
    尺度で表した未見量である。参照版は実残り枚数（尺度1）を、Belief対応版は
    fixed-point raw（尺度`SCALE`）の残余期待枚数を渡す（Issue #218）。

    式はmin / max / 加算 / 正の整数倍だけで構成され、除算・正規化・整数枚数への
    丸め戻しを行わない。そのため入力全体を共通の正の倍率で拡大するとpaijiaも
    同じ倍率になり、候補間の順位・同点は保たれる。ドラ・場風・自風は入力の
    尺度に依存しない整数倍率として扱う。
    """

    def __init__(
        self,
        tile_counts: Sequence[int],
        red_five_counts: Sequence[int],
        policy_input: PolicyInput,
    ) -> None:
        self._counts = tuple(tile_counts)
        self._red = tuple(red_five_counts)
        weights = [1] * len(_ALL_TILE_TYPES)
        for indicator in policy_input.round.dora_indicators:
            weights[tile_type_index(_dora_tile_type(indicator.tile_type))] *= 2
        self._weights = weights
        self._round_wind = wind_index(policy_input.round.round_wind)
        self._seat_wind = wind_index(
            wind_for_seat(policy_input.self_seat, policy_input.round.dealer_seat)
        )
        self._cache: dict[Tile, int] = {}

    def paijia(self, tile: Tile) -> int:
        cached = self._cache.get(tile)
        if cached is None:
            cached = self._cache[tile] = self._compute_paijia(tile)
        return cached

    def _compute_paijia(self, tile: Tile) -> int:
        tile_type = tile.tile_type
        index = tile_type_index(tile_type)
        n = tile_type.rank
        weights = self._weights

        if tile_type.category is TileCategory.HONOR:
            value = self._counts[index] * weights[index]
            if n == self._round_wind + 1:
                value *= 2
            if n == self._seat_wind + 1:
                value *= 2
            if _DRAGON_START_RANK <= n <= 7:
                value *= 2
        else:
            # rank r（1..9）の値は`num[r - 1]` / `weight[r - 1]`。範囲外rankは0。
            base = index - (n - 1)
            num = self._counts[base : base + _SUITED_MAXIMUM_RANK]
            weight = weights[base : base + _SUITED_MAXIMUM_RANK]
            left = min(num[n - 3], num[n - 2]) if n - 2 >= 1 else 0
            center = min(num[n - 2], num[n]) if n - 1 >= 1 and n + 1 <= 9 else 0
            right = min(num[n], num[n + 1]) if n + 2 <= 9 else 0
            n_pai = (
                left,
                max(left, center),
                num[n - 1],
                max(center, right),
                right,
            )
            value = sum(
                n_pai[offset + 2] * weight[n + offset - 1]
                for offset in range(-2, 3)
                if 1 <= n + offset <= 9
            )
            red = self._red[red_five_index(tile_type.category)]
            if red:
                bonus_index = {7: 0, 6: 1, 5: 2, 4: 3, 3: 4}.get(n)
                if bonus_index is not None:
                    value += min(red, n_pai[bonus_index]) * weight[n + bonus_index - 3]
            if tile.is_red:
                value *= 2
        return value * weights[index]


class _PublicCounts:
    """判断時点の実残り枚数とpaijia入力のsnapshot。

    全discard候補で同じ判断時点の値を共有する。仮に捨てる牌を未知牌へ
    戻さない（sourceの`SuanPai`は自己のツモ牌・打牌を既知として扱う）。
    ukeireは常に実残り枚数を使う。paijiaの入力は参照版では同じ実残り枚数
    （sourceの`_paishu`）である。
    """

    def __init__(self, policy_input: PolicyInput) -> None:
        conservation = derive_remaining_tile_inventory(policy_input)
        self.conservation = conservation
        self._remaining = conservation.remaining_tile_counts
        self.paijia_input = _PaijiaInput(
            conservation.remaining_tile_counts,
            conservation.remaining_red_five_counts,
            policy_input,
        )

    @property
    def remaining_tile_counts(self) -> tuple[int, ...]:
        """canonical 34牌種indexの未見枚数（赤5を含む）。"""
        return self._remaining

    def remaining(self, tile_type: TileType) -> int:
        """sourceの`_paishu[s][n]`（赤5を含む基礎牌種の未見枚数）。"""
        return self._remaining[tile_type_index(tile_type)]

    def paijia(self, tile: Tile) -> int:
        """sourceの`SuanPai.paijia()`（牌の評価値）。"""
        return self.paijia_input.paijia(tile)


def _improving_tile_types(hand: Sequence[Tile]) -> tuple[TileType, ...]:
    """sourceの`Majiang.Util.tingpai()`: 手中4枚でなく向聴数を下げる牌種。

    Tile列を直接評価する単純な定義。判断経路は同値な`_DiscardStructures`を使い、
    この関数はその定義の正本・testのoracleとして残す。
    """
    current = calculate_shanten(hand)
    counts: dict[TileType, int] = {}
    for tile in hand:
        counts[tile.tile_type] = counts.get(tile.tile_type, 0) + 1
    return tuple(
        tile_type
        for tile_type in _ALL_TILE_TYPES
        if counts.get(tile_type, 0) < _MAX_COPIES_PER_TILE_TYPE
        and calculate_shanten([*hand, Tile(tile_type)]) < current
    )


class _DiscardStructures:
    """1 decisionの打牌候補について、打牌後の向聴数と改善牌を牌種ごとに共有する。

    Issue #218の同値最適化。打牌後の牌姿は捨てる牌の基礎牌種だけで決まるため、
    ツモ切り / 手出し、赤5 / 通常5の候補は同じ構造評価を共有する。値は牌姿だけに
    依存し、公開枚数・ドラ・Beliefには依存しない（それらはukeire / paijia側で
    適用する）。instanceはdecision-localで、snapshot間で共有しない。

    打牌前の手牌を`calculate_shanten()`で1度だけ検証し、打牌後の向聴数と改善牌は
    34牌種countから`evaluate_discards_from_canonical_counts()`（Issue #224）で
    求める。最初の問い合わせで、構築時に渡された打牌候補（手中にある牌の基礎牌種）
    をまとめて1回評価し、改善牌は打牌後向聴数が現在の向聴数以下の候補だけ求める
    （参照版の選択が改善牌を使うのはその候補だけ）。それ以外の候補の改善牌
    （和了形からの立直判定等）や、候補に含まれなかった牌種は、問い合わせ時に
    その牌種だけ評価する。未評価（`None`）を空集合として扱わない。

    改善牌は`_improving_tile_types()`と同じ「手中4枚でなく向聴数を下げる牌種」で、
    未見枚数0の牌種も含む（形としての改善牌。立直の和了牌判定にも使う）。
    改善牌はcanonical index昇順・重複なしで、`lisjong.belief.tile_type_set`の
    牌種集合の契約を満たす（Issue #221。集約時に再検証しない）。
    """

    def __init__(
        self, concealed: Sequence[Tile], discard_tiles: Sequence[Tile] = ()
    ) -> None:
        self.shanten = calculate_shanten(concealed)
        self._held = Counter(concealed)
        counts = [0] * len(_ALL_TILE_TYPES)
        for tile in concealed:
            counts[tile_type_index(tile.tile_type)] += 1
        self._counts = counts
        # 手中にない牌の候補は含めず、問い合わせ時に`_discard_index()`が拒否する。
        self._pending = tuple(
            {
                tile_type_index(tile.tile_type): None
                for tile in discard_tiles
                if self._held[tile] > 0
            }
        )
        self._shanten_after: dict[int, int] = {}
        self._improving_after: dict[int, tuple[int, ...]] = {}

    def _discard_index(self, tile: Tile) -> int:
        if self._held[tile] <= 0:
            raise Kobalab0004ReferencePolicyError(
                f"{tile} is not in own_hand.concealed_tiles"
            )
        return tile_type_index(tile.tile_type)

    def _evaluate(
        self, indexes: tuple[int, ...], improving_max_shanten: int | None
    ) -> None:
        shanten_after, improving_after = evaluate_discards_from_canonical_counts(
            self._counts, indexes, improving_max_shanten
        )
        for index, shanten, improving in zip(
            indexes, shanten_after, improving_after, strict=True
        ):
            self._shanten_after[index] = shanten
            if improving is not None:
                self._improving_after[index] = improving

    def _evaluate_pending(self) -> None:
        pending = self._pending
        if pending:
            self._pending = ()
            self._evaluate(pending, self.shanten)

    def shanten_after(self, tile: Tile) -> int:
        """`tile`を捨てた後の向聴数。"""
        index = self._discard_index(tile)
        shanten = self._shanten_after.get(index)
        if shanten is None:
            self._evaluate_pending()
            shanten = self._shanten_after.get(index)
            if shanten is None:
                self._evaluate((index,), self.shanten)
                shanten = self._shanten_after[index]
        return shanten

    def improving_after(self, tile: Tile) -> tuple[int, ...]:
        """`tile`を捨てた後の改善牌（canonical 34牌種indexの牌種集合、昇順）。"""
        index = self._discard_index(tile)
        improving = self._improving_after.get(index)
        if improving is None:
            self._evaluate_pending()
            improving = self._improving_after.get(index)
            if improving is None:
                self._evaluate((index,), None)
                improving = self._improving_after[index]
        return improving


def _source_order_key(action: DiscardAction) -> tuple[int, ...]:
    """`get_dapai().reverse()`の位置を表すkey（小さいほど先）。

    sourceの`get_dapai()`は`m,p,s,z`・rank昇順、5では赤5 -> 通常5の順に
    手出し候補を並べ、最後にツモ切りを加える。reverseによりツモ切りが先頭、
    続いて字牌7..1、索子、筒子、萬子の降順、5では通常5 -> 赤5となる。
    """
    if action.tsumogiri:
        return (0,)
    tile_type = action.tile.tile_type
    return (
        1,
        -_CATEGORY_ORDER.index(tile_type.category),
        -tile_type.rank,
        1 if action.tile.is_red else 0,
    )


def _evaluation_order(
    counts: _PublicCounts, discard_actions: Sequence[DiscardAction]
) -> tuple[DiscardAction, ...]:
    """source `get_dapai().reverse().sort(paijia)`相当の評価順。"""
    return tuple(
        sorted(
            discard_actions,
            key=lambda action: (counts.paijia(action.tile), _source_order_key(action)),
        )
    )


def _maximum_ukeire_discard(
    counts: _PublicCounts,
    ordered_actions: Sequence[DiscardAction],
    structures: _DiscardStructures,
    maximum_shanten_after: int,
) -> DiscardAction | None:
    """打牌後向聴数が上限以下の候補から、評価順で最初に受入最大となる候補。

    受入は改善牌の実残り枚数の合計で、厳密に増えた場合だけ選択を更新する。
    該当候補がなければ`None`を返す。
    """
    remaining = counts.remaining_tile_counts
    chosen: DiscardAction | None = None
    best = -1
    for action in ordered_actions:
        if structures.shanten_after(action.tile) > maximum_shanten_after:
            continue
        ukeire = sum_tile_type_values(
            remaining, structures.improving_after(action.tile)
        )
        if ukeire > best:
            best = ukeire
            chosen = action
    return chosen


def _choose_reference_discard(
    counts: _PublicCounts,
    discard_actions: Sequence[DiscardAction],
    structures: _DiscardStructures,
) -> DiscardAction:
    """sourceの`select_dapai()`: 打牌前向聴数以下で受入最大、なければ評価順の先頭。"""
    ordered = _evaluation_order(counts, discard_actions)
    chosen = _maximum_ukeire_discard(counts, ordered, structures, structures.shanten)
    if chosen is None:
        return ordered[0]
    return chosen
