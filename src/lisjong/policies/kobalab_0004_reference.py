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

打牌選択の構造評価・枚数集計・候補選択は、他のPolicyと共有するため
`lisjong.policies.kobalab_0004_discard`に置く（Issue #226）。このmoduleは
参照版・Belief版の判断順序とpaijia入力の選択（Belief対応）を持つ。

hidden opponent hand、wall truth、外部環境privateなstateは参照しない。

`Kobalab0004BeliefPaijiaPolicy`（Issue #218）は、paijiaの入力だけを
`NonPlayerHiddenBelief`由来の残余期待枚数へ替えた別identityの対応版である。
計測・検証・Beliefの供給責務は`docs/kobalab-0004-belief-paijia.md`を正本とする。
"""

from collections.abc import Sequence

from lisjong.belief import (
    ConcealedHandBelief,
    TileConservationResult,
    derive_non_player_hidden_belief,
    tile_type_index,
    wind_for_seat,
    wind_index,
)
from lisjong.belief.conditional_uniform_hand_belief import (
    _estimate_from_conservation,
)
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies.kobalab_0004_discard import (
    Kobalab0004ReferencePolicyError,
    _choose_reference_discard,
    _DiscardStructures,
    _evaluation_order,
    _PaijiaInput,
    _PublicCounts,
)
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
from lisjong.policy_contract.tile import Tile, TileType

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


def _estimate_concealed_hand_belief(
    policy_input: PolicyInput, conservation: TileConservationResult
) -> ConcealedHandBelief:
    """Belief対応版の推定器境界。

    同じ`PolicyInput`だけから他家3人の`HandBelief`を導出する。現行は条件付き
    一様推定器（Issue #65 / #68）である。`conservation`は同じdecisionの
    `_PublicCounts`が同じ`policy_input`から導出した未見枚数で、推定器内部では
    再導出しない（Issue #220。結果は`estimate_conditional_uniform_hand_belief()`と
    同一）。別の推定器を接続する場合は、同じ`PolicyInput`（と同じsnapshotの
    未見枚数）から`ConcealedHandBelief`を返す関数をここへ差し替え、別の
    Policy identityを与える。
    """
    return _estimate_from_conservation(
        policy_input,
        conservation,
        _opponent_concealed_slot_counts_by_wind(policy_input),
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
        belief = _estimate_concealed_hand_belief(policy_input, counts.conservation)
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


def evaluation_order(
    policy_input: PolicyInput, discard_actions: Sequence[DiscardAction]
) -> tuple[DiscardAction, ...]:
    """source `get_dapai().reverse().sort(paijia)`相当の評価順を返す。"""
    counts = _PublicCounts(policy_input)
    return _evaluation_order(counts, discard_actions)


def _choose_discard(
    policy_input: PolicyInput,
    discard_actions: Sequence[DiscardAction],
    structures: _DiscardStructures,
    *,
    belief_paijia: bool = False,
) -> DiscardAction:
    """参照版・Belief版の打牌選択。paijia入力だけが両者で異なる。"""
    counts = _PublicCounts(policy_input)
    if belief_paijia:
        counts.paijia_input = _paijia_input_from_belief(policy_input, counts)
    return _choose_reference_discard(counts, discard_actions, structures)


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
    return (
        structures.shanten_after(action.tile) == 0
        and len(structures.improving_after(action.tile)) > 0
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
            structures = _DiscardStructures(
                policy_input.own_hand.concealed_tiles,
                tuple(action.tile for action in discards),
            )
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
