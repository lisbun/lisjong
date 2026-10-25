"""リーチ判断の特徴APIとD arm用のリーチ抑止wrapper（Issue #352、riichi-dama-branch-v1）。

意味定義の正本はArena #539の「設計確定」§2・§4である。consumerはArena #540。
current Heuristic Champion `PlacementAwareSpeedCallPolicy`と既存Policyの挙動は変えない。

## 1. 解析API `analyze_riichi_decision(decision, declaration_tile)`

入力は`RiichiAction`が合法な判断のpublic `DecisionContext`と、自分の宣言牌（その判断の
手牌に含まれる`Tile`。赤を区別する）だけである。宣言牌を打った後の13枚について
`RiichiDecisionFeatures`を返す。

- 待ち牌の種類W: 打牌後の手で向聴数を下げる（= 和了形にする）基礎牌種。通常形・七対子・
  国士無双をすべて含む（`calculate_shanten`の定義）。自分の手牌に4枚ある牌種は含めない
- 見えない残り枚数: 4 − (自分の手牌 + 河・副露・ドラ表示牌で見えている枚数)。
  `derive_remaining_tile_inventory`で数える。宣言牌は手牌から河へ移るだけなので、
  判断時点の手牌で1回だけ数えた値がそのまま打牌後の値になる（二重に数えない）
- 振聴: 打牌後の自分の河（鳴かれた牌を含む）＋宣言牌の牌種にWのどれかがある
- 待ちごとの点数: ダマのロン / ツモ、リーチのロン / ツモを`evaluate_win`で評価する。
  和了牌は原則として赤でない牌で、ルール上の通常牌の上限（赤ありでは各色の5は通常3枚）を
  評価入力（宣言後の手牌・副露・和了牌・ドラ表示牌）が超える場合だけ赤牌で評価する
  （lisjong #354）。実際に評価した和了牌（`winning_tile`）と赤の加点（`red_dora`）を返す。
  赤でも成立しない場合は解析エラー。一発なし、状況役なし（`WinSituation.NORMAL`）。ダマは裏ドラなしを
  明示し、リーチは`URA_DORA_EXCLUDED`。ダブルリーチになる場合はそのstatusで評価する。
  振聴でもロンの点数は計算する（振聴の判定は`furiten`で行う）
- ダブルリーチ: 自分がまだ打牌しておらず、局内でどの席にも副露（暗槓を含む）がない。
  engineの`was_first_discard and not had_prior_call`と同じ条件
- 巡目: 判断時点の自分の打牌数（宣言牌を含まない。鳴かれた牌を含む）
- 他家のリーチ: 公開されたリーチ状態が`NONE`でない他家の人数（宣言中を含む）
- 親か、オーラス相当か（Championの`_is_all_last`と同じ定義）、現在の持ち点と順位。
  順位は`1 + 自分より持ち点が多い他家の数`で、同点の他家の数を別に返す。起家は
  `PolicyInput`にないため、同点の順位はここで決めない
- 診断: `RiichiAction`を除いた合法手で、新しいChampion instanceが選ぶ行動（ダマでの選択）。
  宣言牌と一致するかの判定はArena側で行う

部分集合の閾値は判定しない。`ScoreLimit`の満貫以上かどうか等は、返した値の比較だけで
判定できる。値はfrozen dataclassで、`riichi_decision_features_to_value()` /
`parse_riichi_decision_features()`でversion付きのJSON objectと往復する。

情報境界: 入力は`DecisionContext`と自分の宣言牌だけである。他家の手牌・山・裏ドラ・
privilegedな観測は受け取らない。入力の不整合は`RiichiDamaAnalysisError`でfail closedする。

## 2. D arm wrapper `RiichiDamaBranchPolicy`

`inner`（Champion）、対象局（`RoundIdentity`: 場風・局・本場）、分岐点で打つ
`DiscardAction`を受け取る。

- 対象局以外の判断では`inner`の選択をそのまま返す
- 対象局で`inner`が初めて`RiichiAction`を選んだ判断（分岐点）では、指定の打牌を返す。
  指定の打牌が合法手に完全一致（赤のidentityとツモ切りflagを含む）で含まれなければ
  `RiichiDamaBranchError`とする
- 同じ局のそれ以降の判断で`inner`が`RiichiAction`を選んだ場合は、`RiichiAction`を除いた
  合法手で`DecisionContext`を作り直して`inner`を再度呼び、その結果を返す。
  差し替えた回数は`replacement_count`で読める
- 分岐点以外の和了・鳴き・打牌は`inner`のまま
- `status`: 分岐前は`PENDING`、分岐後は`BRANCHED`。分岐点が来ないまま対象局を出た
  （対象局の判断の後に別の局の判断が来た）場合は`MISSED`。対象局を出た後に再び対象局の
  判断が来た場合（分岐点がもう一度来うる状態）は`RiichiDamaBranchError`で拒否する

このwrapperは分岐の有無という判断をまたぐ状態を意図的に持つ、対比較実行専用のPolicyであり、
候補Policyではない。1つの対局・1つの席ごとに新しいinstanceを使う。
"""

from dataclasses import dataclass
from enum import Enum

from lisjong.belief.canonical_axes import tile_type_index, wind_for_seat
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.hand_evaluation import calculate_shanten
from lisjong.hand_evaluation.scoring import (
    PROJECT_STANDARD_SCORING_RULES,
    URA_DORA_EXCLUDED,
    EvaluationStatus,
    RiichiStatus,
    ScoreLimit,
    ScoringMeld,
    UraDoraIndicators,
    WinContext,
    WinMethod,
    WinningHand,
    WinSituation,
    evaluate_win,
)
from lisjong.policies.placement_aware_speed_call import (
    PlacementAwareSpeedCallPolicy,
    _is_all_last,
)
from lisjong.policy_contract.action import (
    AnkanAction,
    DiscardAction,
    InternalAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    RiichiAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy import Policy
from lisjong.policy_contract.policy_execution import execute_policy
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind
from lisjong.structural_efficiency import (
    effective_tile_types,
    post_discard_concealed_hand,
)

RIICHI_DAMA_FEATURES_VERSION = "riichi-dama-features-v2"
"""`RiichiDecisionFeatures`のJSON projectionと意味定義（Arena #539 §4）のversion。

v2（lisjong #354）: 通常牌の上限を超える待ちを赤牌で評価し、`WaitFeatures.winning_tile`と
`WaitScore.red_dora`を追加した。v1は読まない。"""

_MAX_COPIES = 4
_PLAIN_FIVE_LIMIT = _MAX_COPIES - 1
"""赤ありのルールで各色の5の通常牌の枚数（赤1枚 + 通常3枚）。"""
_SEAT_COUNT = 4


class RiichiDamaAnalysisError(Exception):
    """解析APIの入力・JSON値が前提と整合しない場合にfail closedする。"""


class RiichiDamaBranchError(Exception):
    """D arm wrapperの前提が崩れた場合（Arena側ではINVALID）。"""


# ---------------------------------------------------------------------------
# 1. 解析API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WaitScore:
    """1つの待ち・和了方法の`evaluate_win`の結果。

    `status`は`SCORED`か`NO_YAKU`。`NO_YAKU`なら他の値はすべて`None`。
    `SCORED`なら`han` / `limit` / `winner_points`は値を持ち、`fu`は役満で`None`になりうる。
    支払点はロンなら`ron_payment`、ツモなら`tsumo_*_payment`（親のツモでは
    `tsumo_dealer_payment`が`None`）で、本場・供託を含まない。
    `red_dora`は翻に含まれる赤ドラの枚数（和了牌が赤ならそれを含む）。役満などドラの内訳が
    ない場合は`None`。
    """

    status: EvaluationStatus
    han: int | None = None
    fu: int | None = None
    limit: ScoreLimit | None = None
    winner_points: int | None = None
    ron_payment: int | None = None
    tsumo_dealer_payment: int | None = None
    tsumo_non_dealer_payment: int | None = None
    red_dora: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.red_dora,
            self.han,
            self.fu,
            self.limit,
            self.winner_points,
            self.ron_payment,
            self.tsumo_dealer_payment,
            self.tsumo_non_dealer_payment,
        )
        if self.status is EvaluationStatus.NO_YAKU:
            if any(value is not None for value in values):
                raise ValueError("a NO_YAKU wait has no score values")
            return
        if self.status is not EvaluationStatus.SCORED:
            raise ValueError("wait score status must be SCORED or NO_YAKU")
        if not isinstance(self.limit, ScoreLimit):
            raise ValueError("a scored wait requires a ScoreLimit")
        for name, value in (("han", self.han), ("winner_points", self.winner_points)):
            if type(value) is not int:
                raise ValueError(f"a scored wait requires an int {name}")
        for name, value in (
            ("fu", self.fu),
            ("ron_payment", self.ron_payment),
            ("tsumo_dealer_payment", self.tsumo_dealer_payment),
            ("tsumo_non_dealer_payment", self.tsumo_non_dealer_payment),
        ):
            if value is not None and type(value) is not int:
                raise ValueError(f"{name} must be an int or None")
        if self.red_dora is not None and (
            type(self.red_dora) is not int or self.red_dora < 0
        ):
            raise ValueError("red_dora must be a non-negative int or None")
        if self.ron_payment is None and self.tsumo_non_dealer_payment is None:
            raise ValueError("a scored wait requires a ron or tsumo payment")

    @property
    def is_scored(self) -> bool:
        return self.status is EvaluationStatus.SCORED


@dataclass(frozen=True, slots=True)
class WaitFeatures:
    """1つの待ち牌種の、見えない残り枚数と4通りの点数。

    `winning_tile`は4通りの点数を評価した和了牌（通常牌、または通常牌の上限を超える場合の
    赤牌）。`unseen_count`は公開情報から数えた値で、`winning_tile`が赤でもその赤が見えて
    いないとは限らない。
    """

    tile_type: TileType
    winning_tile: Tile
    unseen_count: int
    dama_ron: WaitScore
    dama_tsumo: WaitScore
    riichi_ron: WaitScore
    riichi_tsumo: WaitScore

    def __post_init__(self) -> None:
        if not isinstance(self.tile_type, TileType):
            raise ValueError("tile_type must be a TileType")
        if (
            not isinstance(self.winning_tile, Tile)
            or self.winning_tile.tile_type != self.tile_type
        ):
            raise ValueError("winning_tile must be a Tile of tile_type")
        if type(self.unseen_count) is not int or not (
            0 <= self.unseen_count <= _MAX_COPIES
        ):
            raise ValueError("unseen_count must be an int in 0..4")
        scores = (self.dama_ron, self.dama_tsumo, self.riichi_ron, self.riichi_tsumo)
        if any(not isinstance(score, WaitScore) for score in scores):
            raise ValueError("the four wait scores must be WaitScore values")


@dataclass(frozen=True, slots=True)
class RiichiDecisionFeatures:
    """リーチ判断1回の特徴（Arena #539 §4）。`waits`は牌種のcanonical順。"""

    seat: Seat
    declaration_tile: Tile
    waits: tuple[WaitFeatures, ...]
    furiten: bool
    double_riichi: bool
    own_discard_count: int
    opponent_riichi_count: int
    is_dealer: bool
    is_all_last: bool
    score: int
    rank: int
    tied_opponent_count: int
    dama_choice: InternalAction

    def __post_init__(self) -> None:
        if not isinstance(self.seat, Seat):
            raise ValueError("seat must be a Seat")
        if not isinstance(self.declaration_tile, Tile):
            raise ValueError("declaration_tile must be a Tile")
        waits = tuple(self.waits)
        if not waits or any(not isinstance(wait, WaitFeatures) for wait in waits):
            raise ValueError("waits must be a non-empty tuple of WaitFeatures")
        indices = [tile_type_index(wait.tile_type) for wait in waits]
        if indices != sorted(set(indices)):
            raise ValueError("waits must be unique and in canonical tile order")
        object.__setattr__(self, "waits", waits)
        flags = (self.furiten, self.double_riichi, self.is_dealer, self.is_all_last)
        if any(type(flag) is not bool for flag in flags):
            raise ValueError(
                "furiten / double_riichi / is_dealer / is_all_last must be bool"
            )
        counts = (
            self.own_discard_count,
            self.opponent_riichi_count,
            self.score,
            self.rank,
            self.tied_opponent_count,
        )
        if any(type(count) is not int for count in counts):
            raise ValueError("counts, score and rank must be int")
        if self.own_discard_count < 0:
            raise ValueError("own_discard_count must be non-negative")
        if not 0 <= self.opponent_riichi_count < _SEAT_COUNT:
            raise ValueError("opponent_riichi_count must be in 0..3")
        if not 1 <= self.rank <= _SEAT_COUNT:
            raise ValueError("rank must be in 1..4")
        if not 0 <= self.tied_opponent_count <= _SEAT_COUNT - self.rank:
            raise ValueError("tied_opponent_count is inconsistent with rank")
        if self.double_riichi and self.own_discard_count != 0:
            raise ValueError("a double riichi requires no own discard")
        if type(self.dama_choice) not in _ACTION_KINDS:
            raise ValueError("dama_choice must be a non-riichi own-turn action")
        if self.dama_choice.actor != self.seat:
            raise ValueError("dama_choice must be an action of seat")

    @property
    def wait_tile_types(self) -> tuple[TileType, ...]:
        return tuple(wait.tile_type for wait in self.waits)

    @property
    def total_unseen_count(self) -> int:
        """Wの見えない残り枚数の合計。"""
        return sum(wait.unseen_count for wait in self.waits)


def _wait_score(evaluation) -> WaitScore:
    if evaluation.status is EvaluationStatus.NO_YAKU:
        return WaitScore(EvaluationStatus.NO_YAKU)
    if evaluation.status is not EvaluationStatus.SCORED:
        raise RiichiDamaAnalysisError("a structural wait does not complete the hand")
    score = evaluation.score
    return WaitScore(
        status=EvaluationStatus.SCORED,
        han=score.han,
        fu=score.fu,
        limit=score.limit,
        winner_points=score.winner_points,
        ron_payment=score.ron_payment,
        tsumo_dealer_payment=score.tsumo_dealer_payment,
        tsumo_non_dealer_payment=score.tsumo_non_dealer_payment,
        red_dora=None if score.dora is None else score.dora.red,
    )


def winning_tile_for(tile_type: TileType, physical: tuple[Tile, ...]) -> Tile:
    """待ち牌種の和了牌（lisjong #354）。

    原則は通常牌。赤ありのルールで5の牌種のとき、評価入力の物理牌（宣言後の手牌・副露・
    ドラ表示牌）の通常牌と和了牌の合計が通常牌の上限（3枚）を超える場合だけ赤牌を返す。
    赤牌でも成立しない入力は、`evaluate_win`の拒否として呼び出し側で解析エラーにする。
    """
    plain = Tile(tile_type)
    if (
        not PROJECT_STANDARD_SCORING_RULES.red_dora_enabled
        or tile_type.category is TileCategory.HONOR
        or tile_type.rank != 5
    ):
        return plain
    if sum(1 for tile in physical if tile == plain) + 1 <= _PLAIN_FIVE_LIMIT:
        return plain
    return Tile(tile_type, is_red=True)


def _is_double_riichi(decision: DecisionContext) -> bool:
    policy_input = decision.input
    own = policy_input.players[int(policy_input.self_seat)]
    return not own.discards and not any(player.melds for player in policy_input.players)


def analyze_riichi_decision(
    decision: DecisionContext, declaration_tile: Tile
) -> RiichiDecisionFeatures:
    """リーチ判断と宣言牌から、宣言牌を打った後の特徴を返す。"""
    if not isinstance(decision, DecisionContext):
        raise TypeError("decision must be a DecisionContext")
    if not isinstance(declaration_tile, Tile):
        raise TypeError("declaration_tile must be a Tile")
    policy_input = decision.input
    seat = policy_input.self_seat
    if RiichiAction(actor=seat) not in decision.legal_actions:
        raise RiichiDamaAnalysisError("RiichiAction is not legal in this decision")
    own = policy_input.players[int(seat)]
    if own.riichi is not RiichiState.NONE:
        raise RiichiDamaAnalysisError("the own seat has already declared riichi")
    concealed = policy_input.own_hand.concealed_tiles
    if declaration_tile not in concealed:
        raise RiichiDamaAnalysisError("the declaration tile is not in the own hand")

    hand = post_discard_concealed_hand(concealed, declaration_tile)
    if calculate_shanten(hand) != 0:
        raise RiichiDamaAnalysisError("the hand is not tenpai after the declaration")
    wait_types = effective_tile_types(hand, 0)
    if not wait_types:
        raise RiichiDamaAnalysisError("the tenpai hand has no wait tile type")

    remaining = derive_remaining_tile_inventory(policy_input).remaining_tile_counts
    river = {discard.tile.tile_type for discard in own.discards}
    river.add(declaration_tile.tile_type)
    double_riichi = _is_double_riichi(decision)
    riichi_status = RiichiStatus.DOUBLE_RIICHI if double_riichi else RiichiStatus.RIICHI
    melds = tuple(ScoringMeld(meld.kind, meld.tiles) for meld in own.melds)
    round_state = policy_input.round
    seat_wind = wind_for_seat(seat, round_state.dealer_seat)

    physical = (
        *hand,
        *(tile for meld in own.melds for tile in meld.tiles),
        *round_state.dora_indicators,
    )

    def evaluate(winning: Tile, method: WinMethod, riichi: bool) -> WaitScore:
        context = WinContext(
            method=method,
            seat_wind=seat_wind,
            prevailing_wind=round_state.round_wind,
            riichi=riichi_status if riichi else RiichiStatus.NONE,
            is_ippatsu=False,
            situation=WinSituation.NORMAL,
            dora_indicators=round_state.dora_indicators,
            ura_dora=URA_DORA_EXCLUDED if riichi else UraDoraIndicators(()),
        )
        try:
            evaluation = evaluate_win(WinningHand(hand, melds, winning), context)
        except ValueError as error:
            raise RiichiDamaAnalysisError(
                f"the wait {winning.tile_type} cannot complete with a physical tile: {error}"
            ) from None
        return _wait_score(evaluation)

    winning_tiles = {
        tile_type: winning_tile_for(tile_type, physical) for tile_type in wait_types
    }

    waits = tuple(
        WaitFeatures(
            tile_type=tile_type,
            winning_tile=winning_tiles[tile_type],
            unseen_count=remaining[tile_type_index(tile_type)],
            dama_ron=evaluate(winning_tiles[tile_type], WinMethod.RON, False),
            dama_tsumo=evaluate(winning_tiles[tile_type], WinMethod.TSUMO, False),
            riichi_ron=evaluate(winning_tiles[tile_type], WinMethod.RON, True),
            riichi_tsumo=evaluate(winning_tiles[tile_type], WinMethod.TSUMO, True),
        )
        for tile_type in wait_types
    )

    own_score = own.score
    others = [
        player for index, player in enumerate(policy_input.players) if index != seat
    ]
    dama_legal = tuple(
        action
        for action in decision.legal_actions
        if not isinstance(action, RiichiAction)
    )
    dama_choice = execute_policy(
        PlacementAwareSpeedCallPolicy(),
        DecisionContext(input=policy_input, legal_actions=dama_legal),
    )
    if type(dama_choice) not in _ACTION_KINDS:
        raise RiichiDamaAnalysisError("the dama choice is not an own-turn action")

    return RiichiDecisionFeatures(
        seat=seat,
        declaration_tile=declaration_tile,
        waits=waits,
        furiten=any(tile_type in river for tile_type in wait_types),
        double_riichi=double_riichi,
        own_discard_count=len(own.discards),
        opponent_riichi_count=sum(
            player.riichi is not RiichiState.NONE for player in others
        ),
        is_dealer=seat == round_state.dealer_seat,
        is_all_last=_is_all_last(policy_input),
        score=own_score,
        rank=1 + sum(player.score > own_score for player in others),
        tied_opponent_count=sum(player.score == own_score for player in others),
        dama_choice=dama_choice,
    )


# ---------------------------------------------------------------------------
# JSON projection（version付き。未知のversion・余分なfieldは拒否する）
# ---------------------------------------------------------------------------

_ACTION_KINDS: dict[type, str] = {
    DiscardAction: "discard",
    TsumoAction: "tsumo",
    AnkanAction: "ankan",
    KakanAction: "kakan",
    KyuushuKyuuhaiAction: "kyuushu_kyuuhai",
}
"""ダマでの選択になりうる、リーチ判断（自分の手番）の行動。"""

_ACTION_FIELDS = {
    "discard": {"kind", "actor", "tile", "tsumogiri"},
    "tsumo": {"kind", "actor", "winning_tile"},
    "ankan": {"kind", "actor", "tiles"},
    "kakan": {"kind", "actor", "added_tile", "from_seat", "called_tile"},
    "kyuushu_kyuuhai": {"kind", "actor"},
}
_SCORE_FIELDS = {
    "status",
    "han",
    "fu",
    "limit",
    "winner_points",
    "ron_payment",
    "tsumo_dealer_payment",
    "tsumo_non_dealer_payment",
    "red_dora",
}
_WAIT_FIELDS = {
    "tile_type",
    "winning_tile",
    "unseen_count",
    "dama_ron",
    "dama_tsumo",
    "riichi_ron",
    "riichi_tsumo",
}
_FEATURE_FIELDS = {
    "version",
    "seat",
    "declaration_tile",
    "waits",
    "furiten",
    "double_riichi",
    "own_discard_count",
    "opponent_riichi_count",
    "is_dealer",
    "is_all_last",
    "score",
    "rank",
    "tied_opponent_count",
    "dama_choice",
}


def _tile_type_value(tile_type: TileType) -> dict:
    return {"category": tile_type.category.value, "rank": tile_type.rank}


def _tile_value(tile: Tile) -> dict:
    return {**_tile_type_value(tile.tile_type), "is_red": tile.is_red}


def _action_value(action: InternalAction) -> dict:
    kind = _ACTION_KINDS.get(type(action))
    if kind is None:
        raise RiichiDamaAnalysisError("unsupported dama choice action")
    value: dict[str, object] = {"kind": kind, "actor": int(action.actor)}
    if kind == "discard":
        value.update(tile=_tile_value(action.tile), tsumogiri=action.tsumogiri)
    elif kind == "tsumo":
        value["winning_tile"] = _tile_value(action.winning_tile)
    elif kind == "ankan":
        value["tiles"] = [_tile_value(tile) for tile in action.tiles]
    elif kind == "kakan":
        value.update(
            added_tile=_tile_value(action.added_tile),
            from_seat=int(action.from_seat),
            called_tile=_tile_value(action.called_tile),
        )
    return value


def _score_value(score: WaitScore) -> dict:
    return {
        "status": score.status.value,
        "han": score.han,
        "fu": score.fu,
        "limit": None if score.limit is None else score.limit.value,
        "winner_points": score.winner_points,
        "ron_payment": score.ron_payment,
        "tsumo_dealer_payment": score.tsumo_dealer_payment,
        "tsumo_non_dealer_payment": score.tsumo_non_dealer_payment,
        "red_dora": score.red_dora,
    }


def riichi_decision_features_to_value(features: RiichiDecisionFeatures) -> dict:
    """`RiichiDecisionFeatures`をversion付きのJSON objectへ射影する。"""
    if not isinstance(features, RiichiDecisionFeatures):
        raise TypeError("features must be a RiichiDecisionFeatures")
    return {
        "version": RIICHI_DAMA_FEATURES_VERSION,
        "seat": int(features.seat),
        "declaration_tile": _tile_value(features.declaration_tile),
        "waits": [
            {
                "tile_type": _tile_type_value(wait.tile_type),
                "winning_tile": _tile_value(wait.winning_tile),
                "unseen_count": wait.unseen_count,
                "dama_ron": _score_value(wait.dama_ron),
                "dama_tsumo": _score_value(wait.dama_tsumo),
                "riichi_ron": _score_value(wait.riichi_ron),
                "riichi_tsumo": _score_value(wait.riichi_tsumo),
            }
            for wait in features.waits
        ],
        "furiten": features.furiten,
        "double_riichi": features.double_riichi,
        "own_discard_count": features.own_discard_count,
        "opponent_riichi_count": features.opponent_riichi_count,
        "is_dealer": features.is_dealer,
        "is_all_last": features.is_all_last,
        "score": features.score,
        "rank": features.rank,
        "tied_opponent_count": features.tied_opponent_count,
        "dama_choice": _action_value(features.dama_choice),
    }


def _object(value: object, fields: set[str], context: str) -> dict:
    if type(value) is not dict or set(value) != fields:
        raise RiichiDamaAnalysisError(
            f"{context} must be an object with {sorted(fields)}"
        )
    return value


def _int(value: object, context: str) -> int:
    if type(value) is not int:
        raise RiichiDamaAnalysisError(f"{context} must be an int")
    return value


def _optional_int(value: object, context: str) -> int | None:
    return None if value is None else _int(value, context)


def _bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise RiichiDamaAnalysisError(f"{context} must be a bool")
    return value


def _enum(enum_type: type, value: object, context: str):
    if type(value) is not str:
        raise RiichiDamaAnalysisError(f"{context} must be a string")
    try:
        return enum_type(value)
    except ValueError:
        raise RiichiDamaAnalysisError(f"{context} has an unknown value") from None


def _seat(value: object, context: str) -> Seat:
    raw = _int(value, context)
    try:
        return Seat(raw)
    except ValueError:
        raise RiichiDamaAnalysisError(f"{context} is not a seat") from None


def _parse_tile_type(value: object, context: str) -> TileType:
    raw = _object(value, {"category", "rank"}, context)
    try:
        return TileType(
            _enum(TileCategory, raw["category"], f"{context}.category"),
            _int(raw["rank"], f"{context}.rank"),
        )
    except (TypeError, ValueError) as error:
        raise RiichiDamaAnalysisError(f"{context}: {error}") from None


def _parse_tile(value: object, context: str) -> Tile:
    raw = _object(value, {"category", "rank", "is_red"}, context)
    tile_type = _parse_tile_type(
        {"category": raw["category"], "rank": raw["rank"]}, context
    )
    try:
        return Tile(tile_type, is_red=_bool(raw["is_red"], f"{context}.is_red"))
    except (TypeError, ValueError) as error:
        raise RiichiDamaAnalysisError(f"{context}: {error}") from None


def _parse_action(value: object, context: str) -> InternalAction:
    if type(value) is not dict:
        raise RiichiDamaAnalysisError(f"{context} must be an object")
    kind = value.get("kind")
    if kind not in _ACTION_FIELDS:
        raise RiichiDamaAnalysisError(f"{context} has an unsupported kind")
    raw = _object(value, _ACTION_FIELDS[kind], context)
    actor = _seat(raw["actor"], f"{context}.actor")
    try:
        if kind == "discard":
            return DiscardAction(
                actor=actor,
                tile=_parse_tile(raw["tile"], f"{context}.tile"),
                tsumogiri=_bool(raw["tsumogiri"], f"{context}.tsumogiri"),
            )
        if kind == "tsumo":
            return TsumoAction(
                actor=actor,
                winning_tile=_parse_tile(
                    raw["winning_tile"], f"{context}.winning_tile"
                ),
            )
        if kind == "ankan":
            tiles = raw["tiles"]
            if type(tiles) is not list:
                raise RiichiDamaAnalysisError(f"{context}.tiles must be a list")
            return AnkanAction(
                actor=actor,
                tiles=tuple(
                    _parse_tile(tile, f"{context}.tiles[{index}]")
                    for index, tile in enumerate(tiles)
                ),
            )
        if kind == "kakan":
            return KakanAction(
                actor=actor,
                added_tile=_parse_tile(raw["added_tile"], f"{context}.added_tile"),
                from_seat=_seat(raw["from_seat"], f"{context}.from_seat"),
                called_tile=_parse_tile(raw["called_tile"], f"{context}.called_tile"),
            )
        return KyuushuKyuuhaiAction(actor=actor)
    except (TypeError, ValueError) as error:
        raise RiichiDamaAnalysisError(f"{context}: {error}") from None


def _parse_score(value: object, context: str) -> WaitScore:
    raw = _object(value, _SCORE_FIELDS, context)
    try:
        return WaitScore(
            status=_enum(EvaluationStatus, raw["status"], f"{context}.status"),
            han=_optional_int(raw["han"], f"{context}.han"),
            fu=_optional_int(raw["fu"], f"{context}.fu"),
            limit=(
                None
                if raw["limit"] is None
                else _enum(ScoreLimit, raw["limit"], f"{context}.limit")
            ),
            winner_points=_optional_int(
                raw["winner_points"], f"{context}.winner_points"
            ),
            ron_payment=_optional_int(raw["ron_payment"], f"{context}.ron_payment"),
            tsumo_dealer_payment=_optional_int(
                raw["tsumo_dealer_payment"], f"{context}.tsumo_dealer_payment"
            ),
            tsumo_non_dealer_payment=_optional_int(
                raw["tsumo_non_dealer_payment"], f"{context}.tsumo_non_dealer_payment"
            ),
            red_dora=_optional_int(raw["red_dora"], f"{context}.red_dora"),
        )
    except ValueError as error:
        raise RiichiDamaAnalysisError(f"{context}: {error}") from None


def _parse_wait(value: object, context: str) -> WaitFeatures:
    raw = _object(value, _WAIT_FIELDS, context)
    try:
        return WaitFeatures(
            tile_type=_parse_tile_type(raw["tile_type"], f"{context}.tile_type"),
            winning_tile=_parse_tile(raw["winning_tile"], f"{context}.winning_tile"),
            unseen_count=_int(raw["unseen_count"], f"{context}.unseen_count"),
            dama_ron=_parse_score(raw["dama_ron"], f"{context}.dama_ron"),
            dama_tsumo=_parse_score(raw["dama_tsumo"], f"{context}.dama_tsumo"),
            riichi_ron=_parse_score(raw["riichi_ron"], f"{context}.riichi_ron"),
            riichi_tsumo=_parse_score(raw["riichi_tsumo"], f"{context}.riichi_tsumo"),
        )
    except ValueError as error:
        raise RiichiDamaAnalysisError(f"{context}: {error}") from None


def parse_riichi_decision_features(value: object) -> RiichiDecisionFeatures:
    """`riichi_decision_features_to_value()`の値を読み戻す。未知のversionは拒否する。"""
    if type(value) is not dict:
        raise RiichiDamaAnalysisError("features must be a JSON object")
    if value.get("version") != RIICHI_DAMA_FEATURES_VERSION:
        raise RiichiDamaAnalysisError("unsupported riichi-dama features version")
    raw = _object(value, _FEATURE_FIELDS, "features")
    waits = raw["waits"]
    if type(waits) is not list:
        raise RiichiDamaAnalysisError("features.waits must be a list")
    try:
        return RiichiDecisionFeatures(
            seat=_seat(raw["seat"], "features.seat"),
            declaration_tile=_parse_tile(
                raw["declaration_tile"], "features.declaration_tile"
            ),
            waits=tuple(
                _parse_wait(wait, f"features.waits[{index}]")
                for index, wait in enumerate(waits)
            ),
            furiten=_bool(raw["furiten"], "features.furiten"),
            double_riichi=_bool(raw["double_riichi"], "features.double_riichi"),
            own_discard_count=_int(
                raw["own_discard_count"], "features.own_discard_count"
            ),
            opponent_riichi_count=_int(
                raw["opponent_riichi_count"], "features.opponent_riichi_count"
            ),
            is_dealer=_bool(raw["is_dealer"], "features.is_dealer"),
            is_all_last=_bool(raw["is_all_last"], "features.is_all_last"),
            score=_int(raw["score"], "features.score"),
            rank=_int(raw["rank"], "features.rank"),
            tied_opponent_count=_int(
                raw["tied_opponent_count"], "features.tied_opponent_count"
            ),
            dama_choice=_parse_action(raw["dama_choice"], "features.dama_choice"),
        )
    except ValueError as error:
        raise RiichiDamaAnalysisError(f"features: {error}") from None


# ---------------------------------------------------------------------------
# 2. D arm wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoundIdentity:
    """半荘内で局を一意に特定する値（場風・局・本場）。"""

    round_wind: Wind
    hand_number: int
    honba: int

    def __post_init__(self) -> None:
        if not isinstance(self.round_wind, Wind):
            raise TypeError("round_wind must be a Wind")
        if type(self.hand_number) is not int or type(self.honba) is not int:
            raise TypeError("hand_number and honba must be int")
        if not 1 <= self.hand_number <= _SEAT_COUNT:
            raise ValueError("hand_number must be in 1..4")
        if self.honba < 0:
            raise ValueError("honba must be non-negative")

    @classmethod
    def of(cls, round_state: RoundState) -> "RoundIdentity":
        return cls(round_state.round_wind, round_state.hand_number, round_state.honba)


class BranchStatus(Enum):
    """D arm wrapperの分岐の状態。"""

    PENDING = "pending"
    """分岐点はまだ来ていない（対象局の前、または対象局の途中）。"""
    BRANCHED = "branched"
    """分岐点で指定の打牌を返した。"""
    MISSED = "missed"
    """分岐点が来ないまま対象局を出た（Arena側ではINVALID）。"""


class RiichiDamaBranchPolicy:
    """対象局の最初のリーチを指定の打牌へ替え、その局の以降のリーチを抑止するwrapper。"""

    def __init__(
        self,
        inner: Policy,
        target_round: RoundIdentity,
        branch_discard: DiscardAction,
    ) -> None:
        try:
            choose_action = inner.choose_action
        except AttributeError:
            raise TypeError("inner must provide choose_action") from None
        if not callable(choose_action):
            raise TypeError("inner must provide choose_action")
        if not isinstance(target_round, RoundIdentity):
            raise TypeError("target_round must be a RoundIdentity")
        if not isinstance(branch_discard, DiscardAction):
            raise TypeError("branch_discard must be a DiscardAction")
        self._inner = inner
        self._target_round = target_round
        self._branch_discard = branch_discard
        self._seen_target = False
        self._left_target = False
        self._branched = False
        self._replacement_count = 0

    @property
    def target_round(self) -> RoundIdentity:
        return self._target_round

    @property
    def branch_discard(self) -> DiscardAction:
        return self._branch_discard

    @property
    def status(self) -> BranchStatus:
        if self._branched:
            return BranchStatus.BRANCHED
        if self._left_target:
            return BranchStatus.MISSED
        return BranchStatus.PENDING

    @property
    def replacement_count(self) -> int:
        """分岐後に`inner`のリーチを差し替えた回数。"""
        return self._replacement_count

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        if not isinstance(decision, DecisionContext):
            raise TypeError("decision must be a DecisionContext")
        if RoundIdentity.of(decision.input.round) != self._target_round:
            if self._seen_target:
                self._left_target = True
            return self._inner.choose_action(decision)
        if self._left_target:
            raise RiichiDamaBranchError("the target round appeared again")
        self._seen_target = True

        action = self._inner.choose_action(decision)
        if not isinstance(action, RiichiAction):
            return action
        if not self._branched:
            if self._branch_discard not in decision.legal_actions:
                raise RiichiDamaBranchError(
                    "the branch discard is not an exact legal action"
                )
            self._branched = True
            return self._branch_discard

        dama_legal = tuple(
            candidate
            for candidate in decision.legal_actions
            if not isinstance(candidate, RiichiAction)
        )
        if not dama_legal:
            raise RiichiDamaBranchError("no legal action remains without riichi")
        replacement = execute_policy(
            self._inner, DecisionContext(input=decision.input, legal_actions=dama_legal)
        )
        self._replacement_count += 1
        return replacement


__all__ = [
    "RIICHI_DAMA_FEATURES_VERSION",
    "BranchStatus",
    "RiichiDamaAnalysisError",
    "RiichiDamaBranchError",
    "RiichiDamaBranchPolicy",
    "RiichiDecisionFeatures",
    "RoundIdentity",
    "WaitFeatures",
    "WaitScore",
    "analyze_riichi_decision",
    "winning_tile_for",
    "parse_riichi_decision_features",
    "riichi_decision_features_to_value",
]
