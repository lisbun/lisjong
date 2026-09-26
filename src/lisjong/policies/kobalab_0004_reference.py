"""kobalab/majiang-ai legacy 0004の選択規則による純牌効率Reference Policy（Issue #211）。

公開されている`kobalab/majiang-ai`の`legacy/player-0004.js` /
`legacy/suanpai-0004.js`（MIT License, Copyright (c) Satoshi Kobayashi）の
公開仕様を、lisjongの`DecisionContext -> InternalAction`契約と既存評価器の上で
独立実装した deterministic reference である。upstreamのJS package、Node.js、
`@kobalab/majiang-core`へのruntime依存は持たない。

位置付けは「0004の選択規則をlisjongのexact shantenへ適用した参照実装」である。
majiang-coreとの向聴定義差（同一牌種5枚目を要する分解）は設計上許容し、
upstreamとの全局面での行動同値性は主張しない。Arena等の結果はこの移植Policyの
成績であり、kobalab氏の原実装やRiichiLab「牌効率くん」の成績とは扱わない。

reference source      kobalab/majiang-ai legacy 0004
                      commit e75a9720a12b84c03e6c61c3960c1844b8982eb4
rule / core           @kobalab/majiang-core 1.4.1 の Majiang.rule() default
RiichiLab「牌効率くん」との完全同一性  not established

これは新しいChampion候補ではなく、lisjongのshanten / ukeire / second-step系
Policyを古典的な純牌効率baselineと比較するための外部documented referenceである。
詳細な意味対応と既存Policyとの差は`docs/kobalab-0004-reference.md`を正本とする。

判断順序（sourceの`action_zimo` / `action_dapai`に対応）:

1. 和了候補があれば和了する。ただしsourceの`select_hule()`は他家の暗槓への
   ロン（国士無双の槍槓）を拒否するため、targetが同じ牌種のANKANを公開して
   いるRonActionは選ばない。
2. 九種九牌が合法で、自摸番14枚の向聴数が4以上なら宣言する。
3. 自摸番の暗槓・加槓候補を`m -> p -> s -> z`、rank昇順（sourceの
   `get_gang_mianzi()`列挙順）で調べ、槓後の向聴数が槓前と**等しい**最初の
   候補を選ぶ（悪化しない`<=`ではない）。
4. 打牌を0004の牌効率で選び、その打牌で立直が成立する（打牌後聴牌かつ
   和了牌が存在する）うえに`RiichiAction`が合法なら、先に`RiichiAction`を返す。
   lisjongの二段階立直では、続く宣言牌decisionで同じ選択規則を再適用する。
5. Chi / Pon / Daiminkan / Ronを行わない応答機会では`PassAction`を選ぶ。

打牌選択（sourceの`select_dapai()`）:

- 打牌前14枚の向聴数`n`を求め、打牌後向聴数が`n`を超える候補を除外する。
- 残った候補について、打牌後13枚の有効牌（向聴数を下げ、かつ手中に4枚ない
  基礎牌種）ごとの**実残り枚数**を合計したukeireを求める。実残り枚数は
  `derive_remaining_tile_inventory()`が`PolicyInput`から再構成する
  player-visibleな未見枚数であり、sourceの`SuanPai._paishu`に対応する。
- 候補はsourceの`get_dapai().reverse().sort(paijia)`と同じ順で評価し、
  ukeireが**厳密に**増えた場合だけ選択を更新する。全候補が除外された場合は
  評価順の最初の候補を返す（sourceの初期値fallback）。
- 評価順は`paijia`昇順、同値ならsource列挙の逆順（ツモ切り -> 字牌7..1 ->
  索子9..1 -> 筒子 -> 萬子、5では通常5 -> 赤5）である。`legal_actions`の
  入力順には依存しない。

hidden opponent hand、wall truth、外部環境privateなstateは参照しない。
"""

from collections.abc import Sequence

from lisjong.belief import (
    derive_remaining_tile_inventory,
    red_five_index,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policy_contract.action import (
    AnkanAction,
    DiscardAction,
    InternalAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    PassAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

KOBALAB_0004_REFERENCE_IDENTITY = "kobalab-0004-tile-efficiency-reference-v1"
"""Arena等から参照するstableなPolicy identity。class名から導出しない。"""

KOBALAB_0004_REFERENCE_SOURCE = (
    "kobalab/majiang-ai legacy 0004 "
    "(commit e75a9720a12b84c03e6c61c3960c1844b8982eb4, MIT License)"
)

_KYUUSHU_MINIMUM_SHANTEN = 4
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
    """入力がreference semanticsの前提と整合せず、fail closedする場合。"""


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


class _PublicCounts:
    """判断時点の実残り枚数とpaijia計算に必要な公開情報のsnapshot。

    全discard候補で同じ判断時点の値を共有する。仮に捨てる牌を未知牌へ
    戻さない（sourceの`SuanPai`は自己のツモ牌・打牌を既知として扱う）。
    """

    def __init__(self, policy_input: PolicyInput) -> None:
        conservation = derive_remaining_tile_inventory(policy_input)
        self._remaining = conservation.remaining_tile_counts
        self._remaining_red = conservation.remaining_red_five_counts
        self._dora_types = tuple(
            _dora_tile_type(indicator.tile_type)
            for indicator in policy_input.round.dora_indicators
        )
        self._round_wind = wind_index(policy_input.round.round_wind)
        self._seat_wind = wind_index(
            wind_for_seat(policy_input.self_seat, policy_input.round.dealer_seat)
        )
        self._paijia_cache: dict[Tile, int] = {}

    def remaining(self, tile_type: TileType) -> int:
        """sourceの`_paishu[s][n]`（赤5を含む基礎牌種の未見枚数）。"""
        return self._remaining[tile_type_index(tile_type)]

    def _num(self, category: TileCategory, rank: int) -> int:
        return self.remaining(TileType(category, rank))

    def _weight(self, category: TileCategory, rank: int) -> int:
        if rank < 1 or rank > 9:
            return 0
        weight = 1
        tile_type = TileType(category, rank)
        for dora_type in self._dora_types:
            if dora_type == tile_type:
                weight *= 2
        return weight

    def paijia(self, tile: Tile) -> int:
        """sourceの`SuanPai.paijia()`（牌の評価値）。"""
        cached = self._paijia_cache.get(tile)
        if cached is None:
            cached = self._paijia_cache[tile] = self._compute_paijia(tile)
        return cached

    def _compute_paijia(self, tile: Tile) -> int:
        category = tile.tile_type.category
        n = tile.tile_type.rank
        weight = self._weight

        if category is TileCategory.HONOR:
            value = self._num(category, n) * weight(category, n)
            if n == self._round_wind + 1:
                value *= 2
            if n == self._seat_wind + 1:
                value *= 2
            if _DRAGON_START_RANK <= n <= 7:
                value *= 2
        else:
            num = self._num
            left = min(num(category, n - 2), num(category, n - 1)) if n - 2 >= 1 else 0
            center = (
                min(num(category, n - 1), num(category, n + 1))
                if n - 1 >= 1 and n + 1 <= 9
                else 0
            )
            right = min(num(category, n + 1), num(category, n + 2)) if n + 2 <= 9 else 0
            n_pai = (
                left,
                max(left, center),
                num(category, n),
                max(center, right),
                right,
            )
            value = sum(
                n_pai[offset + 2] * weight(category, n + offset)
                for offset in range(-2, 3)
            )
            red = self._remaining_red[red_five_index(category)]
            if red:
                bonus_index = {7: 0, 6: 1, 5: 2, 4: 3, 3: 4}.get(n)
                if bonus_index is not None:
                    value += min(red, n_pai[bonus_index]) * weight(
                        category, n + bonus_index - 2
                    )
            if tile.is_red:
                value *= 2
        return value * weight(category, n)


def _remove_exact(tiles: Sequence[Tile], removed: Sequence[Tile]) -> list[Tile]:
    """赤牌区分を含むexact equalityで`removed`を1枚ずつ取り除く。"""
    remaining = list(tiles)
    for tile in removed:
        try:
            remaining.remove(tile)
        except ValueError:
            raise Kobalab0004ReferencePolicyError(
                f"{tile} is not in own_hand.concealed_tiles"
            ) from None
    return remaining


def _improving_tile_types(hand: Sequence[Tile]) -> tuple[TileType, ...]:
    """sourceの`Majiang.Util.tingpai()`: 手中4枚でなく向聴数を下げる牌種。"""
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


def evaluation_order(
    policy_input: PolicyInput, discard_actions: Sequence[DiscardAction]
) -> tuple[DiscardAction, ...]:
    """source `get_dapai().reverse().sort(paijia)`相当の評価順を返す。"""
    counts = _PublicCounts(policy_input)
    return _evaluation_order(counts, discard_actions)


def _evaluation_order(
    counts: _PublicCounts, discard_actions: Sequence[DiscardAction]
) -> tuple[DiscardAction, ...]:
    return tuple(
        sorted(
            discard_actions,
            key=lambda action: (counts.paijia(action.tile), _source_order_key(action)),
        )
    )


def _choose_discard(
    policy_input: PolicyInput, discard_actions: Sequence[DiscardAction]
) -> DiscardAction:
    counts = _PublicCounts(policy_input)
    concealed = policy_input.own_hand.concealed_tiles
    n_xiangting = calculate_shanten(concealed)

    chosen: DiscardAction | None = None
    best = -1
    for action in _evaluation_order(counts, discard_actions):
        if chosen is None:
            chosen = action
        after = _remove_exact(concealed, (action.tile,))
        if calculate_shanten(after) > n_xiangting:
            continue
        ukeire = sum(counts.remaining(t) for t in _improving_tile_types(after))
        if ukeire > best:
            best = ukeire
            chosen = action
    assert chosen is not None
    return chosen


def _allows_riichi_discard(policy_input: PolicyInput, action: DiscardAction) -> bool:
    """source `allow_lizhi(shoupai, p)`の牌姿条件: 打牌後聴牌かつ和了牌あり。"""
    after = _remove_exact(policy_input.own_hand.concealed_tiles, (action.tile,))
    return calculate_shanten(after) == 0 and bool(_improving_tile_types(after))


def _kan_removed_tiles(action: AnkanAction | KakanAction) -> tuple[Tile, ...]:
    if isinstance(action, AnkanAction):
        return action.tiles
    return (action.added_tile,)


def _kan_tile_type(action: AnkanAction | KakanAction) -> TileType:
    if isinstance(action, AnkanAction):
        return action.tiles[0].tile_type
    return action.added_tile.tile_type


def _is_ankan_chankan(policy_input: PolicyInput, action: RonAction) -> bool:
    """暗槓に対するロンか。

    targetが同じ牌種のANKANを公開していれば、その牌種の4枚すべてが暗槓内に
    あるため、この和了牌はtargetの打牌ではあり得ず暗槓へのロンである。
    """
    tile_type = action.winning_tile.tile_type
    return any(
        meld.kind is MeldKind.ANKAN and meld.tiles[0].tile_type == tile_type
        for meld in policy_input.players[int(action.target)].melds
    )


class Kobalab0004ReferencePolicy:
    """kobalab/majiang-ai legacy 0004の選択規則をexact shantenへ適用した参照実装。

    鳴きなし・聴牌即リー・オリなし。upstreamとの全局面での行動同値性は主張しない。RiichiLab「牌効率くん」との完全同一性は
    確立していない。
    """

    identity = KOBALAB_0004_REFERENCE_IDENTITY
    reference_source = KOBALAB_0004_REFERENCE_SOURCE

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        policy_input = decision.input
        legal = decision.legal_actions

        winning = [
            action
            for action in legal
            if isinstance(action, TsumoAction)
            or (
                isinstance(action, RonAction)
                and not _is_ankan_chankan(policy_input, action)
            )
        ]
        if len(winning) > 1:
            raise Kobalab0004ReferencePolicyError(
                "multiple winning candidates in one decision are undefined"
            )
        if winning:
            return winning[0]

        discards = tuple(a for a in legal if isinstance(a, DiscardAction))

        kyuushu = [a for a in legal if isinstance(a, KyuushuKyuuhaiAction)]
        if kyuushu and (
            calculate_shanten(policy_input.own_hand.concealed_tiles)
            >= _KYUUSHU_MINIMUM_SHANTEN
        ):
            return kyuushu[0]

        kans = sorted(
            (a for a in legal if isinstance(a, (AnkanAction, KakanAction))),
            key=lambda action: tile_type_index(_kan_tile_type(action)),
        )
        if kans:
            concealed = policy_input.own_hand.concealed_tiles
            before = calculate_shanten(concealed)
            for action in kans:
                after = _remove_exact(concealed, _kan_removed_tiles(action))
                if calculate_shanten(after) == before:
                    return action

        if discards:
            chosen = _choose_discard(policy_input, discards)
            riichi = [a for a in legal if isinstance(a, RiichiAction)]
            if (
                riichi
                and policy_input.players[int(policy_input.self_seat)].riichi
                is RiichiState.NONE
                and _allows_riichi_discard(policy_input, chosen)
            ):
                return riichi[0]
            return chosen

        passes = [a for a in legal if isinstance(a, PassAction)]
        if passes:
            return passes[0]

        raise Kobalab0004ReferencePolicyError(
            "no winning, kyuushu, kan, discard, or pass action is defined for "
            "this decision"
        )


__all__ = [
    "KOBALAB_0004_REFERENCE_IDENTITY",
    "KOBALAB_0004_REFERENCE_SOURCE",
    "Kobalab0004ReferencePolicy",
    "Kobalab0004ReferencePolicyError",
    "evaluation_order",
]
