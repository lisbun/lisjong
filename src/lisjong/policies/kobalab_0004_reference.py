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

from collections import Counter
from collections.abc import Sequence

from lisjong.belief import (
    ConcealedHandBelief,
    derive_non_player_hidden_belief,
    derive_remaining_tile_inventory,
    estimate_conditional_uniform_hand_belief,
    red_five_index,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
from lisjong.hand_evaluation import calculate_shanten
from lisjong.hand_evaluation.shanten import calculate_shanten_from_canonical_counts
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
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

KOBALAB_0004_REFERENCE_IDENTITY = "kobalab-0004-tile-efficiency-reference-v1"
"""Arena等から参照するstableなPolicy identity。class名から導出しない。"""

KOBALAB_0004_REFERENCE_SOURCE = (
    "kobalab/majiang-ai legacy 0004 "
    "(commit e75a9720a12b84c03e6c61c3960c1844b8982eb4, MIT License)"
)

KOBALAB_0004_BELIEF_PAIJIA_IDENTITY = "kobalab-0004-tile-efficiency-belief-paijia-v1"
"""paijia入力だけをBelief由来にした対応版（Issue #218）のstableなidentity。"""

KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR = (
    "lisjong conditional-uniform-hand-belief "
    "(estimate_conditional_uniform_hand_belief, #65/#68; "
    "opponent slots = 13 - 3 * public melds) "
    "-> derive_non_player_hidden_belief (#67)"
)
"""対応版の再現に必要な推定器identityと設定。推定器を変える場合はidentityも変える。"""

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


def _opponent_concealed_slot_counts_by_wind(
    policy_input: PolicyInput,
) -> tuple[int, int, int, int]:
    """条件付き一様推定器へ渡す他家concealed slot数（canonical Wind順）。

    自分の打牌decisionでは、他家の純手牌は`13 - 3 * 副露・槓の数`枚である
    （暗槓・加槓・大明槓も1面子として数える）。公開された副露だけから導出し、
    Seatは`wind_for_seat()`で自風へ対応付ける。自分のentryは0とする。
    """
    slots = [0, 0, 0, 0]
    for seat in Seat:
        if seat is policy_input.self_seat:
            continue
        count = 13 - 3 * len(policy_input.players[int(seat)].melds)
        if count < 0:
            raise Kobalab0004ReferencePolicyError(
                f"seat {int(seat)} has more public melds than a hand can hold"
            )
        slots[wind_index(wind_for_seat(seat, policy_input.round.dealer_seat))] = count
    return (slots[0], slots[1], slots[2], slots[3])


def _estimate_concealed_hand_belief(policy_input: PolicyInput) -> ConcealedHandBelief:
    """Belief対応版の推定器境界。

    同じ`PolicyInput`だけから他家3人の`HandBelief`を導出する。現行は条件付き
    一様推定器（Issue #65 / #68）である。別の推定器を接続する場合は、同じ
    `PolicyInput`から`ConcealedHandBelief`を返す関数をここへ差し替え、別の
    Policy identityを与える。
    """
    return estimate_conditional_uniform_hand_belief(
        policy_input, _opponent_concealed_slot_counts_by_wind(policy_input)
    )


def _paijia_input_from_belief(
    policy_input: PolicyInput, counts: _PublicCounts
) -> _PaijiaInput:
    """paijia入力を`未見枚数 − 他家3人の手牌内期待枚数`（fixed-point raw）にする。

    ukeireと同じsnapshotの`counts.conservation`から`NonPlayerHiddenBelief`を
    導出する。自手は`conservation`で既に既知として数えているため差し引かず、
    赤5は34牌種の5（赤5を含む）とは別axisで与える。保存則違反は
    `derive_non_player_hidden_belief()`がclampせずに拒否する。
    """
    try:
        belief = _estimate_concealed_hand_belief(policy_input)
        residual = derive_non_player_hidden_belief(
            counts.conservation,
            belief,
            wind_for_seat(policy_input.self_seat, policy_input.round.dealer_seat),
        )
    except ValueError as error:
        raise Kobalab0004ReferencePolicyError(
            f"belief-derived paijia input is inconsistent: {error}"
        ) from error
    return _PaijiaInput(
        residual.expected_count_raw, residual.red_five_probability_raw, policy_input
    )


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

    打牌前の手牌を`calculate_shanten()`で1度だけ検証し、以降は34牌種countの
    -1（打牌）/ +1（仮想ツモ）だけでcount-native hot pathを呼ぶ。改善牌は
    `_improving_tile_types()`と同じ「手中4枚でなく向聴数を下げる牌種」で、
    未見枚数0の牌種も含む（形としての改善牌。立直の和了牌判定にも使う）。
    """

    def __init__(self, concealed: Sequence[Tile]) -> None:
        self.shanten = calculate_shanten(concealed)
        self._held = Counter(concealed)
        counts = [0] * len(_ALL_TILE_TYPES)
        for tile in concealed:
            counts[tile_type_index(tile.tile_type)] += 1
        self._counts = counts
        self._shanten_after: dict[int, int] = {}
        self._improving_after: dict[int, tuple[int, ...]] = {}

    def _discard_index(self, tile: Tile) -> int:
        if self._held[tile] <= 0:
            raise Kobalab0004ReferencePolicyError(
                f"{tile} is not in own_hand.concealed_tiles"
            )
        return tile_type_index(tile.tile_type)

    def shanten_after(self, tile: Tile) -> int:
        """`tile`を捨てた後の向聴数。"""
        index = self._discard_index(tile)
        shanten = self._shanten_after.get(index)
        if shanten is None:
            counts = self._counts
            counts[index] -= 1
            try:
                shanten = calculate_shanten_from_canonical_counts(counts)
            finally:
                counts[index] += 1
            self._shanten_after[index] = shanten
        return shanten

    def improving_after(self, tile: Tile) -> tuple[int, ...]:
        """`tile`を捨てた後の改善牌（canonical 34牌種index、昇順）。"""
        index = self._discard_index(tile)
        improving = self._improving_after.get(index)
        if improving is None:
            current = self.shanten_after(tile)
            counts = self._counts
            counts[index] -= 1
            try:
                found = []
                for drawn in range(len(counts)):
                    if counts[drawn] >= _MAX_COPIES_PER_TILE_TYPE:
                        continue
                    counts[drawn] += 1
                    shanten = calculate_shanten_from_canonical_counts(counts)
                    counts[drawn] -= 1
                    if shanten < current:
                        found.append(drawn)
            finally:
                counts[index] += 1
            improving = self._improving_after[index] = tuple(found)
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
    policy_input: PolicyInput,
    discard_actions: Sequence[DiscardAction],
    structures: _DiscardStructures,
    *,
    belief_paijia: bool = False,
) -> DiscardAction:
    counts = _PublicCounts(policy_input)
    if belief_paijia:
        counts.paijia_input = _paijia_input_from_belief(policy_input, counts)
    remaining = counts.remaining_tile_counts
    n_xiangting = structures.shanten

    chosen: DiscardAction | None = None
    best = -1
    for action in _evaluation_order(counts, discard_actions):
        if chosen is None:
            chosen = action
        if structures.shanten_after(action.tile) > n_xiangting:
            continue
        ukeire = sum(remaining[i] for i in structures.improving_after(action.tile))
        if ukeire > best:
            best = ukeire
            chosen = action
    assert chosen is not None
    return chosen


def _allows_riichi_discard(
    policy_input: PolicyInput,
    action: DiscardAction,
    structures: _DiscardStructures | None = None,
) -> bool:
    """source `allow_lizhi(shoupai, p)`の牌姿条件: 打牌後聴牌かつ和了牌あり。

    和了牌は形としての改善牌で判定し、未見枚数では絞らない。
    """
    if structures is None:
        structures = _DiscardStructures(policy_input.own_hand.concealed_tiles)
    return structures.shanten_after(action.tile) == 0 and bool(
        structures.improving_after(action.tile)
    )


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
    _belief_paijia = False

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
            structures = _DiscardStructures(policy_input.own_hand.concealed_tiles)
            chosen = _choose_discard(
                policy_input,
                discards,
                structures,
                belief_paijia=self._belief_paijia,
            )
            riichi = [a for a in legal if isinstance(a, RiichiAction)]
            if (
                riichi
                and policy_input.players[int(policy_input.self_seat)].riichi
                is RiichiState.NONE
                and _allows_riichi_discard(policy_input, chosen, structures)
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


class Kobalab0004BeliefPaijiaPolicy(Kobalab0004ReferencePolicy):
    """0004参照Policyのpaijia入力だけをBelief由来の残余期待枚数へ替えた版（#218）。

    paijiaの入力を`未見枚数 − 他家3人の手牌内期待枚数`
    （`NonPlayerHiddenBelief`、fixed-point raw）にする。受入枚数・向聴数・
    候補filter・評価順の同点処理・立直・槓・九種九牌・和了の判断規則は参照版と
    同じで、ukeireは実残り枚数のままである。

    現行の条件付き一様推定器では、残余期待枚数は丸め誤差を除いて未見枚数と
    共通比率になる。この接続だけで打牌が改善するとは主張しない。期待枚数を
    代入したpaijiaはヒューリスティックな牌価であり、牌価の期待値・ツモ確率・
    和了確率ではない。
    """

    identity = KOBALAB_0004_BELIEF_PAIJIA_IDENTITY
    belief_estimator = KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR
    _belief_paijia = True


__all__ = [
    "KOBALAB_0004_BELIEF_PAIJIA_ESTIMATOR",
    "KOBALAB_0004_BELIEF_PAIJIA_IDENTITY",
    "KOBALAB_0004_REFERENCE_IDENTITY",
    "KOBALAB_0004_REFERENCE_SOURCE",
    "Kobalab0004BeliefPaijiaPolicy",
    "Kobalab0004ReferencePolicy",
    "Kobalab0004ReferencePolicyError",
    "evaluation_order",
]
