"""複数Policyが共有するstructural discard / 牌効率semanticを所有するcomponent。

Issue #177により、これまで`lisjong.policies.two_step_ukeire`のprivate helperが
所有していた次のsemanticを、concrete Policy implementationから独立した
lisjong-owned componentへ移した。

    canonical `DiscardAction` ordering
    実際のdiscard identityによるpost-discard concealed hand導出
    Policy-visible known tile counting
    decision-local structural shanten evaluation / memoization
    post-discard structural shanten evaluation
    effective tile types
    current ukeire
    second-step ukeire

この移動はbehavior-preserving refactorである。shantenは公開
`calculate_shanten()`だけを正本とし、known tile counting、ukeire、second-step
scoreのsemanticを一切変更しない。

## ownership boundary

```text
policy_contract
    = Policy境界 / visible state / action contract

structural_efficiency
    = reusable AI-domain structural calculation semantics

concrete Policy
    = reusable semanticsを組み合わせてselection behaviorを定義
```

このmoduleはconcrete Policy moduleへ依存しない。逆依存（Policy
implementationへのimport）を作らない。

## supported semantic vs implementation detail

module-level publicな名前が、複数のlisjong内部consumerが依存してよい
supported semanticである。`_`接頭辞を持つ名前は引き続きimplementation detail
であり、consumerから使用しない。

`StructuralShantenEvaluator`はsupported evaluatorだが、内部のdecision-local
dict cacheはimplementation detailである。cache objectやmutable work objectを
supported contractへ露出しない。

このsupported componentは次ではない。

```text
lisjong external top-level APIの永久固定
generic Policy framework
universal CandidateEvaluation
ML feature schema
```

`PostDiscardStructuralEvaluation`は「actual discard identity → post-discard
concealed hand → structural shanten」というlow-levelなstructural factだけを
持つtyped immutable valueであり、Policy固有のstaged selection semanticsを
表さない。TwoStepUkeireの`TwoStepUkeireCandidateEvaluation`のようなstaged
snapshotは、引き続き各concrete Policyが所有する。

## information boundary

このmoduleは`PolicyInput`-visible informationだけを使用する。山の内部状態、
王牌、他家の実手牌、未来のevent、`GameTrace`のprivileged truth、RiichiEnv /
Arena固有の情報へ依存しない。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lisjong.hand_evaluation import calculate_shanten
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import (
    Tile,
    TileCategory,
    TileType,
    tile_sort_key,
)

_MAX_COPIES_PER_TILE_TYPE = 4

_ALL_TILE_TYPES: tuple[TileType, ...] = tuple(
    TileType(category, rank)
    for category, maximum_rank in (
        (TileCategory.MANZU, 9),
        (TileCategory.PINZU, 9),
        (TileCategory.SOUZU, 9),
        (TileCategory.HONOR, 7),
    )
    for rank in range(1, maximum_rank + 1)
)
"""`tile_sort_key()`と同じ明示的な順序を持つ34基礎牌種。"""


class StructuralEfficiencyError(Exception):
    """structural efficiency計算の入力が不整合な場合にfail closedする。

    `lisjong.policies.two_step_ukeire.TwoStepUkeirePolicyError`は、Issue #177の
    ownership移動でcaller behaviorを変えないためのcompatibility aliasとして
    この型を指す。
    """


@dataclass(frozen=True, slots=True)
class PostDiscardStructuralEvaluation:
    """1 legal discardのlow-levelなpost-discard structural fact。

    `action`は元の`DecisionContext.legal_actions`にあるcanonical
    `DiscardAction`をそのまま保持し、赤5 / 通常5やツモ切りのidentityを
    再構築しない。`post_discard_hand`はそのactual discard identityを1枚だけ
    除いた純手牌のimmutable snapshotである。

    この型はPolicy固有のstaged evaluation snapshotではなく、mutable work
    objectでもない。ukeireやsecond-step scoreのようなstage依存値は持たず、
    それらの評価pathは各concrete Policyが所有する。
    """

    action: DiscardAction
    post_discard_hand: tuple[Tile, ...]
    post_discard_shanten: int

    def __post_init__(self) -> None:
        if not isinstance(self.action, DiscardAction):
            raise TypeError("action must be a DiscardAction")
        if type(self.post_discard_shanten) is not int:
            raise TypeError("post_discard_shanten must be an int")
        try:
            hand = tuple(self.post_discard_hand)
        except TypeError:
            raise TypeError("post_discard_hand must be an iterable of Tile") from None
        if any(not isinstance(tile, Tile) for tile in hand):
            raise TypeError("post_discard_hand must contain only Tile values")
        object.__setattr__(self, "post_discard_hand", hand)


def discard_action_sort_key(
    action: DiscardAction,
) -> tuple[tuple[int, int, bool], bool]:
    """canonical `DiscardAction`順を返すstable tie-break key。

    legal actionの入力順序に依存しないdeterministic orderを与える。
    """
    return (tile_sort_key(action.tile), action.tsumogiri)


def post_discard_concealed_hand(
    concealed_tiles: Sequence[Tile], tile: Tile
) -> tuple[Tile, ...]:
    """実際のdiscard identityと一致する牌を純手牌から1枚だけ除く。

    赤5と通常5は別identityとして扱い、一致する最初の1枚だけを除去する。
    一致する牌が無い場合はfail closedする。
    """
    remaining = list(concealed_tiles)
    for index, candidate in enumerate(remaining):
        if candidate == tile:
            del remaining[index]
            return tuple(remaining)
    raise StructuralEfficiencyError(
        "DiscardAction.tile has no matching tile in own_hand.concealed_tiles"
    )


def known_tile_counts(policy_input: PolicyInput) -> dict[TileType, int]:
    """Policy-visibleな既知牌を基礎牌種単位で数える。

    次を既知牌として数える。

        own concealed tiles
        public melds
        uncalled public discards
        dora indicators

    called discardは鳴いたmeld側で数えるため、同一物理牌を二重計上しない。
    1牌種4枚を超える不整合はfail closedする。
    """
    counts: dict[TileType, int] = {}

    def add(tile: Tile) -> None:
        counts[tile.tile_type] = counts.get(tile.tile_type, 0) + 1

    for tile in policy_input.own_hand.concealed_tiles:
        add(tile)

    for player in policy_input.players:
        for meld in player.melds:
            for tile in meld.tiles:
                add(tile)
        for discard in player.discards:
            if discard.called_by is None:
                add(discard.tile)

    for tile in policy_input.round.dora_indicators:
        add(tile)

    for tile_type in _ALL_TILE_TYPES:
        count = counts.get(tile_type, 0)
        if count > _MAX_COPIES_PER_TILE_TYPE:
            raise StructuralEfficiencyError(
                "known tile count is inconsistent with the PolicyInput contract: "
                f"{count} copies of {tile_type} are visible, but at most "
                f"{_MAX_COPIES_PER_TILE_TYPE} exist"
            )
    return counts


def _count_tile_types(tiles: Sequence[Tile]) -> dict[TileType, int]:
    counts: dict[TileType, int] = {}
    for tile in tiles:
        counts[tile.tile_type] = counts.get(tile.tile_type, 0) + 1
    return counts


class StructuralShantenEvaluator:
    """1 decision内の同一structural handだけを再利用するsupported evaluator。

    同じ34基礎牌種countへ落ちる手牌（赤5と通常5、手出しとツモ切り等）を
    1回だけ`calculate_shanten()`へ渡す。shantenのsemanticは公開
    `calculate_shanten()`が正本であり、この局所cacheは結果を変えない。

    cacheは1 decision分のdecision-local optimizationである。cross-decisionで
    共有するglobal cacheとして使わず、内部cache objectをconsumerへ露出しない。
    """

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: dict[tuple[int, ...], int] = {}

    def calculate(self, hand: Sequence[Tile]) -> int:
        counts = _count_tile_types(hand)
        key = tuple(counts.get(tile_type, 0) for tile_type in _ALL_TILE_TYPES)
        if key not in self._cache:
            self._cache[key] = calculate_shanten(hand)
        return self._cache[key]


def _calculate_shanten(
    hand: Sequence[Tile], evaluator: StructuralShantenEvaluator | None
) -> int:
    if evaluator is None:
        return calculate_shanten(hand)
    return evaluator.calculate(hand)


def evaluate_post_discard_hands(
    policy_input: PolicyInput,
    discard_actions: Sequence[DiscardAction],
    evaluator: StructuralShantenEvaluator,
) -> tuple[PostDiscardStructuralEvaluation, ...]:
    """元のlegal discardごとに打牌後純手牌とstructural shantenを評価する。

    入力された`discard_actions`の順序をそのまま保持する。canonical順が必要な
    consumerは`discard_action_sort_key()`で並べ替える。
    """
    concealed_tiles = policy_input.own_hand.concealed_tiles
    return tuple(
        PostDiscardStructuralEvaluation(
            action=action,
            post_discard_hand=remaining_hand,
            post_discard_shanten=evaluator.calculate(remaining_hand),
        )
        for action in discard_actions
        for remaining_hand in (
            post_discard_concealed_hand(concealed_tiles, action.tile),
        )
    )


def effective_tile_types(
    hand: Sequence[Tile],
    current_shanten: int | None = None,
    evaluator: StructuralShantenEvaluator | None = None,
) -> tuple[TileType, ...]:
    """現在向聴数を実際に下げるstructuralな基礎牌種をcanonical順で返す。"""
    shanten = (
        _calculate_shanten(hand, evaluator)
        if current_shanten is None
        else current_shanten
    )
    hand_counts = _count_tile_types(hand)
    return tuple(
        tile_type
        for tile_type in _ALL_TILE_TYPES
        if hand_counts.get(tile_type, 0) < _MAX_COPIES_PER_TILE_TYPE
        and _calculate_shanten([*hand, Tile(tile_type)], evaluator) < shanten
    )


def ukeire_count(
    hand: Sequence[Tile],
    known_counts: Mapping[TileType, int],
    current_shanten: int | None = None,
    evaluator: StructuralShantenEvaluator | None = None,
) -> int:
    """`Σ Policy-visible remaining(t)`を有効牌種`t`について返す。

    有効牌種は実際にshantenを下げる34基礎牌種であり、残り枚数は
    `known_tile_counts()`が数えたPolicy-visibleな既知枚数の補数である。
    """
    return sum(
        _MAX_COPIES_PER_TILE_TYPE - known_counts.get(tile_type, 0)
        for tile_type in effective_tile_types(hand, current_shanten, evaluator)
    )


def _known_counts_after_draw(
    known_counts: Mapping[TileType, int], tile_type: TileType
) -> dict[TileType, int]:
    """仮想ツモを新しい既知牌として1枚追加する。"""
    current = known_counts.get(tile_type, 0)
    if current >= _MAX_COPIES_PER_TILE_TYPE:
        raise StructuralEfficiencyError(
            "cannot draw a tile type with no Policy-visible remaining copy"
        )
    updated = dict(known_counts)
    updated[tile_type] = current + 1
    return updated


def _remove_one_tile_type(tiles: Sequence[Tile], tile_type: TileType) -> list[Tile]:
    """仮想branchで基礎牌種が一致する牌を1枚だけ除く。"""
    remaining = list(tiles)
    for index, candidate in enumerate(remaining):
        if candidate.tile_type == tile_type:
            del remaining[index]
            return remaining
    raise StructuralEfficiencyError(
        "virtual discard tile type has no matching tile in the hypothetical hand"
    )


def _virtual_discard_tile_types(hand: Sequence[Tile]) -> tuple[TileType, ...]:
    """仮想手牌に存在する基礎牌種をcanonical順で重複なく返す。"""
    present = frozenset(tile.tile_type for tile in hand)
    return tuple(tile_type for tile_type in _ALL_TILE_TYPES if tile_type in present)


def _best_next_ukeire(
    post_discard_hand: Sequence[Tile],
    drawn_tile_type: TileType,
    known_counts_after_draw: Mapping[TileType, int],
    evaluator: StructuralShantenEvaluator | None = None,
) -> int:
    """第1有効牌ツモ後の「最小向聴、次いで最大受け入れ」を返す。"""
    hypothetical_hand = [*post_discard_hand, Tile(drawn_tile_type)]
    evaluated = tuple(
        (
            _calculate_shanten(
                _remove_one_tile_type(hypothetical_hand, discard_tile_type),
                evaluator,
            ),
            discard_tile_type,
        )
        for discard_tile_type in _virtual_discard_tile_types(hypothetical_hand)
    )
    minimum_shanten = min(shanten for shanten, _ in evaluated)
    return max(
        ukeire_count(
            _remove_one_tile_type(hypothetical_hand, discard_tile_type),
            known_counts_after_draw,
            shanten,
            evaluator,
        )
        for shanten, discard_tile_type in evaluated
        if shanten == minimum_shanten
    )


def second_step_ukeire_score(
    post_discard_hand: Sequence[Tile],
    known_counts: Mapping[TileType, int],
    current_shanten: int | None = None,
    evaluator: StructuralShantenEvaluator | None = None,
) -> int:
    """`Σ remaining(t) * best_next_ukeire(t)`をexact integerで返す。

    第1有効牌`t`をPolicy-visibleな未見枚数で重み付けし、仮想ツモ後の最善な
    純手牌形が持つ次の受け入れを合計する。probability、expected points、
    final utilityではない。
    """
    shanten = (
        _calculate_shanten(post_discard_hand, evaluator)
        if current_shanten is None
        else current_shanten
    )
    score = 0
    for tile_type in effective_tile_types(post_discard_hand, shanten, evaluator):
        remaining = _MAX_COPIES_PER_TILE_TYPE - known_counts.get(tile_type, 0)
        if remaining <= 0:
            continue
        after_draw = _known_counts_after_draw(known_counts, tile_type)
        score += remaining * _best_next_ukeire(
            post_discard_hand, tile_type, after_draw, evaluator
        )
    return score
