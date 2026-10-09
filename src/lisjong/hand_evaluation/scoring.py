"""AI側の役・符・点数計算（Issue #263）。

手牌選択・押し引き・危険度・正解ラベル作成など、AIの判断と分析が仮定の完成形
（推定した相手の手、反実仮想の和了牌、offlineの正解手牌）を評価するための
純粋な計算器である。lisjong-engineの点数計算（合法手判定・局進行・精算の正本）
とは別の機能であり、engineへのruntime依存は持たない。

計算と推定の分離:

- 入力は具体的な完成手＋明示的な和了contextだけである。未完成手の打点期待値・
  和了確率・裏ドラ期待値など不確実性を含む量は扱わない
- 未知の値を暗黙に補わない。裏ドラは`UraDoraIndicators`（具体的な表示牌。
  リーチなしは空を明示）か`URA_DORA_EXCLUDED`（裏ドラを除外した基礎点。結果にも
  除外を表示）のどちらかを必ず指定する。未知の裏ドラを0枚として扱わない
- データの取得・推定・正解の読み込みはcallerの責務である。計算器はonlineの
  推定手牌もofflineの正解手牌も受け取れ、onlineの判断経路で未知の他家手牌を
  入手しないことはcaller側の責務とする

結果は`EvaluationStatus`で区別する。不正入力（牌枚数・副露・contextの不整合）は
`ValueError`、和了形でなければ`NOT_COMPLETE`、和了形だが役がなければ
`NO_YAKU`（ドラだけでは役ありにしない。正常な評価結果）、役があれば`SCORED`で
`WinScore`を返す。複数の解釈（和了形×面子分解×和了牌の割当）から支払点が最大の
解釈を選び、同点時は役満倍率→翻→符の大きい順、それでも同じなら和了形
（通常形→七対子→国士無双）、雀頭、手牌内面子のcanonical順、和了牌の位置
（雀頭が先）の小さい順で1つに決める。

本場・供託・責任払い（パオ）は結果に含めない（局の精算はengineの責務）。

計算coreはopt-in native拡張`_lisjong_native`（`native/`のRust crate）にあり、
本moduleは型検査と変換だけを行う薄いbindingである。native拡張が未導入でも
本moduleのimportと入力型の構築は成功し、`evaluate_win()`を呼び出したときだけ
`ScoringBackendUnavailableError`になる。Python版の計算器は持たない。
"""

from dataclasses import dataclass
from enum import Enum

from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

REQUIRED_NATIVE_SCORING_API_VERSION = 1
"""要求する`_lisjong_native.SCORING_API_VERSION`。"""


class ScoringBackendUnavailableError(RuntimeError):
    """役・点数計算のnative coreを利用できない場合のfail closed例外。"""


class Yaku(Enum):
    """役の識別子。値と順序（結果の並び順）はlisjong-engineの`Yaku`と同じ。"""

    MENZEN_TSUMO = "menzen_tsumo"
    RIICHI = "riichi"
    IPPATSU = "ippatsu"
    DOUBLE_RIICHI = "double_riichi"
    CHANKAN = "chankan"
    RINSHAN_KAIHOU = "rinshan_kaihou"
    HAITEI = "haitei"
    HOUTEI = "houtei"
    TANYAO = "tanyao"
    SEAT_WIND = "seat_wind"
    PREVAILING_WIND = "prevailing_wind"
    WHITE_DRAGON = "white_dragon"
    GREEN_DRAGON = "green_dragon"
    RED_DRAGON = "red_dragon"
    PINFU = "pinfu"
    IIPEIKOU = "iipeikou"
    CHANTA = "chanta"
    ITTSUU = "ittsuu"
    SANSHOKU_DOUJUN = "sanshoku_doujun"
    SANSHOKU_DOUKOU = "sanshoku_doukou"
    SANKANTSU = "sankantsu"
    TOITOI = "toitoi"
    SANANKOU = "sanankou"
    SHOUSANGEN = "shousangen"
    HONROUTOU = "honroutou"
    CHIITOITSU = "chiitoitsu"
    JUNCHAN = "junchan"
    HONITSU = "honitsu"
    RYANPEIKOU = "ryanpeikou"
    CHINITSU = "chinitsu"
    TENHOU = "tenhou"
    CHIIHOU = "chiihou"
    DAISANGEN = "daisangen"
    SUUANKOU = "suuankou"
    SUUANKOU_TANKI = "suuankou_tanki"
    TSUUIISOU = "tsuuiisou"
    RYUUIISOU = "ryuuiisou"
    CHINROUTOU = "chinroutou"
    KOKUSHI_MUSOU = "kokushi_musou"
    KOKUSHI_MUSOU_13_WAIT = "kokushi_musou_13_wait"
    DAISUUSHII = "daisuushii"
    SHOUSUUSHII = "shousuushii"
    SUUKANTSU = "suukantsu"
    CHUUREN_POUTOU = "chuuren_poutou"
    JUNSEI_CHUUREN_POUTOU = "junsei_chuuren_poutou"


DOUBLE_YAKUMAN_CANDIDATES = frozenset(
    {
        Yaku.SUUANKOU_TANKI,
        Yaku.KOKUSHI_MUSOU_13_WAIT,
        Yaku.DAISUUSHII,
        Yaku.JUNSEI_CHUUREN_POUTOU,
    }
)
"""ルールでダブル役満として扱える役（lisjong-engineと同じ候補）。"""


class WinMethod(Enum):
    RON = "ron"
    TSUMO = "tsumo"


class RiichiStatus(Enum):
    NONE = "none"
    RIICHI = "riichi"
    DOUBLE_RIICHI = "double_riichi"


class WinSituation(Enum):
    """和了の状況役。互いに排他なので1つだけを指定する。"""

    NORMAL = "normal"
    HAITEI = "haitei"
    HOUTEI = "houtei"
    RINSHAN = "rinshan"
    CHANKAN = "chankan"
    TENHOU = "tenhou"
    CHIIHOU = "chiihou"


class EvaluationStatus(Enum):
    NOT_COMPLETE = "not_complete"
    NO_YAKU = "no_yaku"
    SCORED = "scored"


class WinningShape(Enum):
    STANDARD = "standard"
    SEVEN_PAIRS = "seven_pairs"
    THIRTEEN_ORPHANS = "thirteen_orphans"


class WaitType(Enum):
    RYANMEN = "ryanmen"
    KANCHAN = "kanchan"
    PENCHAN = "penchan"
    SHANPON = "shanpon"
    TANKI = "tanki"
    KOKUSHI_SINGLE = "kokushi_single"
    KOKUSHI_THIRTEEN_SIDED = "kokushi_thirteen_sided"


class GroupKind(Enum):
    SEQUENCE = "sequence"
    TRIPLET = "triplet"
    QUAD = "quad"


class FuReason(Enum):
    """符要素の理由。値はlisjong-engineの`FuReason`と同じ。"""

    SEVEN_PAIRS = "seven_pairs"
    BASE = "base"
    MENZEN_RON = "menzen_ron"
    TSUMO = "tsumo"
    DRAGON_PAIR = "dragon_pair"
    SEAT_WIND_PAIR = "seat_wind_pair"
    PREVAILING_WIND_PAIR = "prevailing_wind_pair"
    DOUBLE_WIND_PAIR = "double_wind_pair"
    OPEN_SIMPLE_TRIPLET = "open_simple_triplet"
    OPEN_TERMINAL_OR_HONOR_TRIPLET = "open_terminal_or_honor_triplet"
    CLOSED_SIMPLE_TRIPLET = "closed_simple_triplet"
    CLOSED_TERMINAL_OR_HONOR_TRIPLET = "closed_terminal_or_honor_triplet"
    OPEN_SIMPLE_QUAD = "open_simple_quad"
    OPEN_TERMINAL_OR_HONOR_QUAD = "open_terminal_or_honor_quad"
    CLOSED_SIMPLE_QUAD = "closed_simple_quad"
    CLOSED_TERMINAL_OR_HONOR_QUAD = "closed_terminal_or_honor_quad"
    KANCHAN_WAIT = "kanchan_wait"
    PENCHAN_WAIT = "penchan_wait"
    TANKI_WAIT = "tanki_wait"


class ScoreLimit(Enum):
    NONE = "none"
    MANGAN = "mangan"
    HANEMAN = "haneman"
    BAIMAN = "baiman"
    SANBAIMAN = "sanbaiman"
    YAKUMAN = "yakuman"


@dataclass(frozen=True, slots=True)
class ScoringRules:
    """点数計算のルール値。fieldの既定値は持たず、すべて明示する。

    lisjong-engineの`PROJECT_STANDARD_RULES`と同じ条件は
    `PROJECT_STANDARD_SCORING_RULES`である。変更したルールは
    `dataclasses.replace()`で作る。
    """

    kuitan_enabled: bool
    red_dora_enabled: bool
    rounded_mangan_enabled: bool
    counted_yakuman_enabled: bool
    multiple_yakuman_enabled: bool
    double_yakuman_variants: frozenset[Yaku]
    double_wind_pair_fu: int

    def __post_init__(self) -> None:
        for name in (
            "kuitan_enabled",
            "red_dora_enabled",
            "rounded_mangan_enabled",
            "counted_yakuman_enabled",
            "multiple_yakuman_enabled",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        try:
            variants = frozenset(self.double_yakuman_variants)
        except TypeError:
            raise TypeError("double_yakuman_variants must be an iterable") from None
        if any(not isinstance(yaku, Yaku) for yaku in variants):
            raise TypeError("double_yakuman_variants must contain only Yaku")
        if not variants <= DOUBLE_YAKUMAN_CANDIDATES:
            raise ValueError("double_yakuman_variants contains an unsupported yaku")
        if type(self.double_wind_pair_fu) is not int:
            raise TypeError("double_wind_pair_fu must be an int")
        if self.double_wind_pair_fu not in (2, 4):
            raise ValueError("double_wind_pair_fu must be 2 or 4")
        object.__setattr__(self, "double_yakuman_variants", variants)


PROJECT_STANDARD_SCORING_RULES = ScoringRules(
    kuitan_enabled=True,
    red_dora_enabled=True,
    rounded_mangan_enabled=False,
    counted_yakuman_enabled=True,
    multiple_yakuman_enabled=True,
    double_yakuman_variants=frozenset(),
    double_wind_pair_fu=4,
)
"""lisjong-engine `PROJECT_STANDARD_RULES`（`project-standard-v1`）と同じ条件。"""


def _tile_tuple(tiles: object, field_name: str) -> tuple[Tile, ...]:
    try:
        values = tuple(tiles)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable of Tile") from None
    if any(not isinstance(tile, Tile) for tile in values):
        raise TypeError(f"{field_name} must contain only Tile instances")
    return values


@dataclass(frozen=True, slots=True)
class ScoringMeld:
    """副露・槓1件。kindでチー / ポン / 明槓 / 加槓 / 暗槓を区別する。

    牌の整合（枚数・同種・連続）はnative coreが検証する。
    """

    kind: MeldKind
    tiles: tuple[Tile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MeldKind):
            raise TypeError("kind must be a MeldKind")
        object.__setattr__(self, "tiles", _tile_tuple(self.tiles, "tiles"))


@dataclass(frozen=True, slots=True)
class UraDoraIndicators:
    """完全な点数計算用の、具体的な（既知または仮定の）裏ドラ表示牌。

    リーチ時はドラ表示牌と同数、リーチなしは空（裏ドラなしの明示）とする。
    """

    tiles: tuple[Tile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tiles", _tile_tuple(self.tiles, "tiles"))


@dataclass(frozen=True, slots=True)
class UraDoraExcluded:
    """裏ドラを除外した基礎点計算の明示。結果の`ura_dora_excluded`がTrueになる。"""


URA_DORA_EXCLUDED = UraDoraExcluded()


@dataclass(frozen=True, slots=True)
class WinningHand:
    """具体的な完成手。`concealed_tiles`は和了牌を含めない。"""

    concealed_tiles: tuple[Tile, ...]
    melds: tuple[ScoringMeld, ...]
    winning_tile: Tile

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "concealed_tiles",
            _tile_tuple(self.concealed_tiles, "concealed_tiles"),
        )
        try:
            melds = tuple(self.melds)
        except TypeError:
            raise TypeError("melds must be an iterable of ScoringMeld") from None
        if any(not isinstance(meld, ScoringMeld) for meld in melds):
            raise TypeError("melds must contain only ScoringMeld instances")
        object.__setattr__(self, "melds", melds)
        if not isinstance(self.winning_tile, Tile):
            raise TypeError("winning_tile must be a Tile")


@dataclass(frozen=True, slots=True)
class WinContext:
    """明示的な和了context。親は自風が東の席である。

    `dora_indicators`は表ドラ表示牌（通常表示牌の後に槓ドラ表示牌）である。
    contextの整合（リーチは門前、一発はリーチ、海底はツモ等）はnative coreが
    検証する。
    """

    method: WinMethod
    seat_wind: Wind
    prevailing_wind: Wind
    riichi: RiichiStatus
    is_ippatsu: bool
    situation: WinSituation
    dora_indicators: tuple[Tile, ...]
    ura_dora: UraDoraIndicators | UraDoraExcluded

    def __post_init__(self) -> None:
        enum_fields = (
            ("method", WinMethod),
            ("seat_wind", Wind),
            ("prevailing_wind", Wind),
            ("riichi", RiichiStatus),
            ("situation", WinSituation),
        )
        for name, enum_type in enum_fields:
            if not isinstance(getattr(self, name), enum_type):
                raise TypeError(f"{name} must be a {enum_type.__name__}")
        if type(self.is_ippatsu) is not bool:
            raise TypeError("is_ippatsu must be a bool")
        object.__setattr__(
            self,
            "dora_indicators",
            _tile_tuple(self.dora_indicators, "dora_indicators"),
        )
        if not isinstance(self.ura_dora, (UraDoraIndicators, UraDoraExcluded)):
            raise TypeError(
                "ura_dora must be UraDoraIndicators or URA_DORA_EXCLUDED; "
                "unknown ura dora is never treated as zero"
            )


@dataclass(frozen=True, slots=True)
class ScoredGroup:
    """通常形の面子1つ。順子の`tile_type`は先頭牌。"""

    kind: GroupKind
    tile_type: TileType
    is_open: bool
    is_completed_by_ron: bool

    @property
    def is_concealed_for_scoring(self) -> bool:
        """ロンで完成した刻子は手牌内でも暗刻として扱わない。"""
        return not self.is_open and not self.is_completed_by_ron


@dataclass(frozen=True, slots=True)
class YakuValue:
    """成立した役1つ。通常役は`han`、役満は`yakuman_units`が正。"""

    yaku: Yaku
    han: int
    yakuman_units: int


@dataclass(frozen=True, slots=True)
class DoraBreakdown:
    """ドラの内訳。`ura`は裏ドラ除外モードでNone。"""

    dora: int
    red: int
    ura: int | None

    @property
    def total(self) -> int:
        return self.dora + self.red + (self.ura or 0)


@dataclass(frozen=True, slots=True)
class WinScore:
    """選ばれた1解釈の役・翻・符・点数。本場・供託・パオは含めない。

    `groups`は手牌内の面子（canonical順）の後に副露（入力順）を並べる。
    `completed_group`は和了牌で完成した面子のindexで、雀頭（単騎）や特殊形では
    Noneである。役満では`dora`がNone（ドラを加算しない）で`fu`もNoneである。
    数え役満は通常役として`han`と`fu`を保持し、`limit`が`YAKUMAN`になる。
    """

    shape: WinningShape
    wait_type: WaitType
    pair: TileType | None
    groups: tuple[ScoredGroup, ...]
    completed_group: int | None
    yaku: tuple[YakuValue, ...]
    dora: DoraBreakdown | None
    ura_dora_excluded: bool
    yaku_han: int
    han: int
    fu: int | None
    fu_components: tuple[tuple[FuReason, int], ...]
    yakuman_units: int
    method: WinMethod
    is_dealer: bool
    base_points: int
    limit: ScoreLimit
    ron_payment: int | None
    tsumo_dealer_payment: int | None
    tsumo_non_dealer_payment: int | None
    winner_points: int

    @property
    def yaku_set(self) -> frozenset[Yaku]:
        return frozenset(value.yaku for value in self.yaku)


@dataclass(frozen=True, slots=True)
class WinEvaluation:
    """評価結果。`score`は`status`が`SCORED`のときだけ存在する。"""

    status: EvaluationStatus
    score: WinScore | None = None

    @property
    def is_complete(self) -> bool:
        return self.status is not EvaluationStatus.NOT_COMPLETE

    @property
    def has_yaku(self) -> bool:
        return self.status is EvaluationStatus.SCORED


_CATEGORY_OFFSETS = {
    TileCategory.MANZU: 0,
    TileCategory.PINZU: 9,
    TileCategory.SOUZU: 18,
    TileCategory.HONOR: 27,
}
_TILE_TYPES = tuple(
    TileType(category, rank)
    for category in _CATEGORY_OFFSETS
    for rank in range(1, 8 if category is TileCategory.HONOR else 10)
)
_WIND_INDEX = {wind: index for index, wind in enumerate(Wind)}


def _native_tile(tile: Tile) -> tuple[int, bool]:
    tile_type = tile.tile_type
    return _CATEGORY_OFFSETS[tile_type.category] + tile_type.rank - 1, tile.is_red


def _native_tiles(tiles: tuple[Tile, ...]) -> list[tuple[int, bool]]:
    return [_native_tile(tile) for tile in tiles]


def _load_native():
    try:
        import _lisjong_native
    except ImportError as error:
        raise ScoringBackendUnavailableError(
            "lisjong.hand_evaluation.scoring requires the opt-in native extension "
            "'_lisjong_native' (install it with 'python -m pip install ./native'); "
            "there is no Python fallback"
        ) from error
    version = getattr(_lisjong_native, "SCORING_API_VERSION", None)
    if version != REQUIRED_NATIVE_SCORING_API_VERSION:
        raise ScoringBackendUnavailableError(
            "lisjong.hand_evaluation.scoring requires _lisjong_native "
            f"SCORING_API_VERSION {REQUIRED_NATIVE_SCORING_API_VERSION}, got "
            f"{version!r}; rebuild or reinstall the native extension from the "
            "same lisjong revision"
        )
    return _lisjong_native


def require_scoring_backend() -> None:
    """native coreを利用できなければ`ScoringBackendUnavailableError`を送出する。

    `evaluate_win()`を判断中に呼ぶcallerが、構築時に利用可否を確認するために使う。
    """
    _load_native()


def evaluate_win(
    hand: WinningHand,
    context: WinContext,
    rules: ScoringRules = PROJECT_STANDARD_SCORING_RULES,
) -> WinEvaluation:
    """具体的な完成手を評価する。

    不正入力は`ValueError`（型の誤りは`TypeError`）、native拡張が利用できなければ
    `ScoringBackendUnavailableError`を送出する。
    """
    if not isinstance(hand, WinningHand):
        raise TypeError("hand must be a WinningHand")
    if not isinstance(context, WinContext):
        raise TypeError("context must be a WinContext")
    if not isinstance(rules, ScoringRules):
        raise TypeError("rules must be a ScoringRules")
    native = _load_native()

    ura_dora = (
        None
        if isinstance(context.ura_dora, UraDoraExcluded)
        else _native_tiles(context.ura_dora.tiles)
    )
    status, payload = native.evaluate_win(
        _native_tiles(hand.concealed_tiles),
        [(meld.kind.value, _native_tiles(meld.tiles)) for meld in hand.melds],
        _native_tile(hand.winning_tile),
        context.method.value,
        _WIND_INDEX[context.seat_wind],
        _WIND_INDEX[context.prevailing_wind],
        context.riichi.value,
        context.is_ippatsu,
        context.situation.value,
        _native_tiles(context.dora_indicators),
        ura_dora,
        (
            rules.kuitan_enabled,
            rules.red_dora_enabled,
            rules.rounded_mangan_enabled,
            rules.counted_yakuman_enabled,
            rules.multiple_yakuman_enabled,
            sorted(yaku.value for yaku in rules.double_yakuman_variants),
            rules.double_wind_pair_fu,
        ),
    )
    status = EvaluationStatus(status)
    if status is not EvaluationStatus.SCORED:
        return WinEvaluation(status)
    return WinEvaluation(status, _score_from_native(payload))


def _score_from_native(payload: dict) -> WinScore:
    dora = payload["dora"]
    pair = payload["pair"]
    return WinScore(
        shape=WinningShape(payload["shape"]),
        wait_type=WaitType(payload["wait"]),
        pair=None if pair is None else _TILE_TYPES[pair],
        groups=tuple(
            ScoredGroup(
                kind=GroupKind(kind),
                tile_type=_TILE_TYPES[tile],
                is_open=is_open,
                is_completed_by_ron=completed_by_ron,
            )
            for kind, tile, is_open, completed_by_ron in payload["groups"]
        ),
        completed_group=payload["completed_group"],
        yaku=tuple(
            YakuValue(Yaku(name), han, units) for name, han, units in payload["yaku"]
        ),
        dora=None if dora is None else DoraBreakdown(*dora),
        ura_dora_excluded=payload["ura_dora_excluded"],
        yaku_han=payload["yaku_han"],
        han=payload["han"],
        fu=payload["fu"],
        fu_components=tuple(
            (FuReason(reason), fu) for reason, fu in payload["fu_components"]
        ),
        yakuman_units=payload["yakuman_units"],
        method=WinMethod(payload["method"]),
        is_dealer=payload["is_dealer"],
        base_points=payload["base_points"],
        limit=ScoreLimit(payload["limit"]),
        ron_payment=payload["ron_payment"],
        tsumo_dealer_payment=payload["tsumo_dealer_payment"],
        tsumo_non_dealer_payment=payload["tsumo_non_dealer_payment"],
        winner_points=payload["winner_points"],
    )
