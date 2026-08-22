"""1 windの非公開手牌belief。

Issue #59の「風別の非公開手牌beliefを固定小数点canonical representationと
して導入する」を実装する。実際の他家手牌推定algorithm（baseline / uniform
estimator、河・副露・手出しツモ切りを使う推定、neural network、training
dataset等）はこのIssueのscopeではなく、canonical representationだけを
導入する。

`HandBelief`はTile identity（`TileType` + red distinction）が持つ物理的な
copy identityを持たない。34基本牌種それぞれのexpected count（0.0..4.0）と、
数牌の色ごとの赤5 probability（0.0..1.0）だけを、Issue #59が固定した
fixed-point raw表現（`SCALE = 8192`）で保持する。

34牌種側のexpected countは、通常5と赤5を合算した値である。red-five
probabilityを34牌種側へ追加加算しない。

Issue #82で、同じ34牌種canonical axisを共有するwait belief（structural
completion waitのprimary table + wait mechanism table群）をoptional field
として追加した。wait beliefを推定するestimator、完全情報からwait ground
truthを生成するbuilder、Policyからのwait belief利用はIssue #82のscope外で
あり、既存estimatorはwait beliefを未提供（`None`）のままとする。
"""

from dataclasses import dataclass

from lisjong.belief.canonical_axes import red_five_index, tile_type_index
from lisjong.belief.fixed_point import (
    EXPECTED_COUNT_MAX_RAW,
    PROBABILITY_MAX_RAW,
    RED_FIVE_PROBABILITY_MAX_RAW,
    raw_to_semantic,
)
from lisjong.policy_contract.tile import TileCategory, TileType

_SUITED_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)

_ALL_TILE_TYPE_INDICES = frozenset(range(34))


def _suited_rank_indices(ranks: tuple[int, ...]) -> frozenset[int]:
    return frozenset(
        tile_type_index(TileType(category, rank))
        for category in _SUITED_CATEGORIES
        for rank in ranks
    )


_HONOR_INDICES = frozenset(
    tile_type_index(TileType(TileCategory.HONOR, rank)) for rank in range(1, 8)
)

# 各wait mechanismが構造上占め得るcanonical slot。ここに含まれないslotは
# canonical zeroであり、non-zero rawを与えられた場合はfail-closedで拒否する。
_KANCHAN_VALID_INDICES = _suited_rank_indices((2, 3, 4, 5, 6, 7, 8))
_PENCHAN_VALID_INDICES = _suited_rank_indices((3, 7))
_RYANMEN_LOW_SIDE_VALID_INDICES = _suited_rank_indices((1, 2, 3, 4, 5, 6))
_RYANMEN_HIGH_SIDE_VALID_INDICES = _suited_rank_indices((4, 5, 6, 7, 8, 9))
_KOKUSHI_VALID_INDICES = _suited_rank_indices((1, 9)) | _HONOR_INDICES

# wait mechanism group。all-or-noneのavailability contractを持ち、group内の
# partial availabilityは許可しない。
_WAIT_MECHANISM_FIELDS: tuple[tuple[str, frozenset[int]], ...] = (
    ("tanki_wait_probability_raw", _ALL_TILE_TYPE_INDICES),
    ("shanpon_wait_probability_raw", _ALL_TILE_TYPE_INDICES),
    ("kanchan_wait_probability_raw", _KANCHAN_VALID_INDICES),
    ("penchan_wait_probability_raw", _PENCHAN_VALID_INDICES),
    ("ryanmen_low_side_probability_raw", _RYANMEN_LOW_SIDE_VALID_INDICES),
    ("ryanmen_high_side_probability_raw", _RYANMEN_HIGH_SIDE_VALID_INDICES),
    ("kokushi_wait_probability_raw", _KOKUSHI_VALID_INDICES),
)


def _normalize_raw_tuple(
    values: object, expected_length: int, max_raw: int, field_name: str
) -> tuple[int, ...]:
    try:
        raw_values = tuple(values)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable of int") from None
    if len(raw_values) != expected_length:
        raise ValueError(f"{field_name} must contain exactly {expected_length} values")
    for raw in raw_values:
        if type(raw) is not int:
            raise TypeError(f"{field_name} must contain only int values")
        if not 0 <= raw <= max_raw:
            raise ValueError(
                f"{field_name} values must be within their fixed-point range"
            )
    return raw_values


def _normalize_probability_table(
    values: object, valid_indices: frozenset[int], field_name: str
) -> tuple[int, ...]:
    """34牌種canonical axisのprobability tableを検証して正規化する。

    length 34、raw range `0..PROBABILITY_MAX_RAW`、およびinvalid canonical
    slotがcanonical zeroであることをfail-closedで検証する。
    """
    raw_values = _normalize_raw_tuple(values, 34, PROBABILITY_MAX_RAW, field_name)
    for index, raw in enumerate(raw_values):
        if raw != 0 and index not in valid_indices:
            raise ValueError(
                f"{field_name} must be zero at canonical slots this wait "
                "mechanism cannot occupy"
            )
    return raw_values


@dataclass(frozen=True, slots=True)
class HandBelief:
    """1 windの非公開手牌についてのcanonical belief。

    ```text
    HandBelief
    ├── expected_count_raw                (length 34, canonical tile_type_index順)
    ├── red_five_probability_raw          (length 3, canonical red_five_index順)
    │
    ├── wait_probability_raw              (length 34 | None)
    │
    ├── tanki_wait_probability_raw        (length 34 | None)
    ├── shanpon_wait_probability_raw      (length 34 | None)
    ├── kanchan_wait_probability_raw      (length 34 | None)
    ├── penchan_wait_probability_raw      (length 34 | None)
    ├── ryanmen_low_side_probability_raw  (length 34 | None)
    ├── ryanmen_high_side_probability_raw (length 34 | None)
    └── kokushi_wait_probability_raw      (length 34 | None)
    ```

    raw fieldはIssue #59が固定したfixed-point storageそのものであり、通常の
    Policy/domain codeはこのraw値を直接扱わず、`expected_count()` /
    `red_five_probability()`のsemantic accessorを使う。boundaryで
    raw表現が必要な場合だけ`expected_count_raw` / `red_five_probability_raw`
    へ直接アクセスする。

    各色について`red_five_probability <= 対応する5のexpected_count`を、
    raw integerのexact comparisonで検証する。標準的な各色赤5 1枚ルールを
    前提とし、equalは合法とする。1 raw unitでもred-five側が大きい場合は
    拒否する。

    ## wait belief (Issue #82)

    wait tableはすべて既存expected countと同じ34牌種canonical axisであり、
    値は`[0.0, 1.0]`のprobability（raw `0..PROBABILITY_MAX_RAW`）とする。

    `wait_probability`は**structural completion wait**のprimary beliefで
    ある。「現在の手牌構造へ牌種tを1枚加えたとき、通常手・七対子・国士無双
    のいずれかの完成形を構成できる」probabilityを表し、furiten、ron / tsumo
    action legality、yaku、点数、riichi状態、Policy action legality、
    残り枚数（remaining tile availability）とは分離する。場に4枚見えていて
    remaining copiesが0でも、構造上の待ちならwait beliefはnon-zeroになり得る。

    mechanism tableは、その牌種がどのwait mechanismでhand completionを
    成立させるかのauxiliary beliefである。七対子の待ちは`tanki`へ包含し、
    国士の待ちは`tanki`へ包含せず専用の`kokushi`channelで表す。ryanmenは
    待ち牌自身のrankが元taatsuより低い側か高い側かで
    `ryanmen_low_side` / `ryanmen_high_side`へ分ける。

    mechanismはmulti-labelであり、同一牌種について複数mechanismが同時に
    non-zeroでもよい。probabilistic beliefでは各channelがmarginal
    probabilityなので、mechanism間のsum <= 1.0や、
    `wait = sum / max / OR(mechanism)`のような代数制約はconstructorで
    課さない。table sumが1.0を超えることも合法とする（多面待ちでは複数slotが
    同時に1になる）。

    availabilityは3 levelとする。

    ```text
    Level 0: wait_probability_raw = None,     mechanism group = all None
    Level 1: wait_probability_raw = [34],     mechanism group = all None
    Level 2: wait_probability_raw = [34],     mechanism group = all [34]
    ```

    mechanism group内のpartial availabilityと、mechanism groupだけがあって
    `wait_probability_raw`が`None`の状態は拒否する。`None`は「estimatorが
    そのfeatureを提供していない」、all-zeroは「estimatorが全wait
    probabilityを0と推定している（非聴牌等）」を意味し、両者を区別する。
    """

    expected_count_raw: tuple[int, ...]
    red_five_probability_raw: tuple[int, ...]
    wait_probability_raw: tuple[int, ...] | None = None
    tanki_wait_probability_raw: tuple[int, ...] | None = None
    shanpon_wait_probability_raw: tuple[int, ...] | None = None
    kanchan_wait_probability_raw: tuple[int, ...] | None = None
    penchan_wait_probability_raw: tuple[int, ...] | None = None
    ryanmen_low_side_probability_raw: tuple[int, ...] | None = None
    ryanmen_high_side_probability_raw: tuple[int, ...] | None = None
    kokushi_wait_probability_raw: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        expected_count_raw = _normalize_raw_tuple(
            self.expected_count_raw, 34, EXPECTED_COUNT_MAX_RAW, "expected_count_raw"
        )
        red_five_probability_raw = _normalize_raw_tuple(
            self.red_five_probability_raw,
            3,
            RED_FIVE_PROBABILITY_MAX_RAW,
            "red_five_probability_raw",
        )

        for category in _SUITED_CATEGORIES:
            five_raw = expected_count_raw[tile_type_index(TileType(category, 5))]
            red_five_raw = red_five_probability_raw[red_five_index(category)]
            if red_five_raw > five_raw:
                raise ValueError(
                    "red_five_probability must not exceed the corresponding "
                    "five's expected_count"
                )

        object.__setattr__(self, "expected_count_raw", expected_count_raw)
        object.__setattr__(self, "red_five_probability_raw", red_five_probability_raw)

        self._validate_wait_belief()

    def _validate_wait_belief(self) -> None:
        provided_mechanisms = [
            field_name
            for field_name, _valid_indices in _WAIT_MECHANISM_FIELDS
            if getattr(self, field_name) is not None
        ]
        if provided_mechanisms and len(provided_mechanisms) != len(
            _WAIT_MECHANISM_FIELDS
        ):
            raise ValueError(
                "wait mechanism tables must be either all provided or all omitted"
            )
        if provided_mechanisms and self.wait_probability_raw is None:
            raise ValueError(
                "wait_probability_raw must be provided when wait mechanism "
                "tables are provided"
            )

        if self.wait_probability_raw is not None:
            object.__setattr__(
                self,
                "wait_probability_raw",
                _normalize_probability_table(
                    self.wait_probability_raw,
                    _ALL_TILE_TYPE_INDICES,
                    "wait_probability_raw",
                ),
            )

        for field_name, valid_indices in _WAIT_MECHANISM_FIELDS:
            values = getattr(self, field_name)
            if values is None:
                continue
            object.__setattr__(
                self,
                field_name,
                _normalize_probability_table(values, valid_indices, field_name),
            )

    @property
    def has_wait_belief(self) -> bool:
        """primary `wait_probability`が提供されているか（Level 1以上）を返す。"""
        return self.wait_probability_raw is not None

    @property
    def has_wait_mechanism_belief(self) -> bool:
        """wait mechanism group一式が提供されているか（Level 2）を返す。

        mechanism groupはall-or-noneであり、constructorがpartial
        availabilityを拒否するため、group内の任意の1 tableの有無で判定できる。
        """
        return self.tanki_wait_probability_raw is not None

    def expected_count(self, tile_type: TileType) -> float:
        """基本牌種`tile_type`のexpected count（0.0..4.0）を返す。"""
        return raw_to_semantic(self.expected_count_raw[tile_type_index(tile_type)])

    def red_five_probability(self, category: TileCategory) -> float:
        """数牌`category`の赤5 probability（0.0..1.0）を返す。"""
        return raw_to_semantic(self.red_five_probability_raw[red_five_index(category)])

    def wait_probability(self, tile_type: TileType) -> float | None:
        """`tile_type`がstructural completion waitである probabilityを返す。

        wait beliefが未提供（Level 0）の場合は`None`を返す。`None`（未提供）と
        `0.0`（probability 0と推定）は区別する。
        """
        return self._probability(self.wait_probability_raw, tile_type)

    def tanki_wait_probability(self, tile_type: TileType) -> float | None:
        """`tile_type`が単騎待ちを完成させるprobabilityを返す（七対子を含む）。

        mechanism beliefが未提供の場合は`None`を返す。
        """
        return self._probability(self.tanki_wait_probability_raw, tile_type)

    def shanpon_wait_probability(self, tile_type: TileType) -> float | None:
        """`tile_type`が双碰待ちを完成させるprobabilityを返す。

        mechanism beliefが未提供の場合は`None`を返す。
        """
        return self._probability(self.shanpon_wait_probability_raw, tile_type)

    def kanchan_wait_probability(self, tile_type: TileType) -> float | None:
        """`tile_type`が嵌張待ちを完成させるprobabilityを返す。

        mechanism beliefが未提供の場合は`None`を返す。
        """
        return self._probability(self.kanchan_wait_probability_raw, tile_type)

    def penchan_wait_probability(self, tile_type: TileType) -> float | None:
        """`tile_type`が辺張待ちを完成させるprobabilityを返す。

        mechanism beliefが未提供の場合は`None`を返す。
        """
        return self._probability(self.penchan_wait_probability_raw, tile_type)

    def ryanmen_low_side_probability(self, tile_type: TileType) -> float | None:
        """`tile_type`が両面待ちの低い側を完成させるprobabilityを返す。

        `2m3m -> 1m`の`1m`側である。mechanism beliefが未提供の場合は
        `None`を返す。
        """
        return self._probability(self.ryanmen_low_side_probability_raw, tile_type)

    def ryanmen_high_side_probability(self, tile_type: TileType) -> float | None:
        """`tile_type`が両面待ちの高い側を完成させるprobabilityを返す。

        `2m3m -> 4m`の`4m`側である。mechanism beliefが未提供の場合は
        `None`を返す。
        """
        return self._probability(self.ryanmen_high_side_probability_raw, tile_type)

    def kokushi_wait_probability(self, tile_type: TileType) -> float | None:
        """`tile_type`が国士無双を完成させるstructural waitであるprobabilityを返す。

        13幺九牌以外は常に0.0である。mechanism beliefが未提供の場合は
        `None`を返す。
        """
        return self._probability(self.kokushi_wait_probability_raw, tile_type)

    @staticmethod
    def _probability(
        table: tuple[int, ...] | None, tile_type: TileType
    ) -> float | None:
        index = tile_type_index(tile_type)
        if table is None:
            return None
        return raw_to_semantic(table[index])
