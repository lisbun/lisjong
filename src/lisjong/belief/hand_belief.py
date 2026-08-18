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
"""

from dataclasses import dataclass

from lisjong.belief.canonical_axes import red_five_index, tile_type_index
from lisjong.belief.fixed_point import (
    EXPECTED_COUNT_MAX_RAW,
    RED_FIVE_PROBABILITY_MAX_RAW,
    raw_to_semantic,
)
from lisjong.policy_contract.tile import TileCategory, TileType

_SUITED_CATEGORIES = (TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU)


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


@dataclass(frozen=True, slots=True)
class HandBelief:
    """1 windの非公開手牌についてのcanonical belief。

    ```text
    HandBelief
    ├── expected_count_raw       (length 34, canonical tile_type_index順)
    └── red_five_probability_raw (length 3, canonical red_five_index順)
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
    """

    expected_count_raw: tuple[int, ...]
    red_five_probability_raw: tuple[int, ...]

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

    def expected_count(self, tile_type: TileType) -> float:
        """基本牌種`tile_type`のexpected count（0.0..4.0）を返す。"""
        return raw_to_semantic(self.expected_count_raw[tile_type_index(tile_type)])

    def red_five_probability(self, category: TileCategory) -> float:
        """数牌`category`の赤5 probability（0.0..1.0）を返す。"""
        return raw_to_semantic(self.red_five_probability_raw[red_five_index(category)])
