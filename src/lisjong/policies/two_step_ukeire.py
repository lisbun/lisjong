"""同向聴・同受け入れ候補を2段階受け入れで比較するPolicy。

`TwoStepUkeirePolicy`は`UkeirePolicy`を置き換えず、次の独立した戦略世代を
実装する。

    打牌後向聴数 > 現在受け入れ > 2段階受け入れ > stable tie-break

2段目は、最小向聴かつ最大現在受け入れの候補が複数残り、打牌後向聴数が
1以上の場合だけ評価する。各第1有効牌をPolicy-visibleな未見枚数で重み付けし、
仮想ツモ後の最善の純手牌形が持つ次の受け入れを合計する。

山の内部状態、他家の実手牌、belief、未来のlegal actionは使用しない。向聴数は
公開`calculate_shanten()`だけを正本とし、赤5と通常5は仮想branchでは同じ
`TileType`として扱う。実際の最初の`DiscardAction` identityは維持する。
"""

from collections.abc import Mapping, Sequence

from lisjong.hand_evaluation import calculate_shanten
from lisjong.policy_contract.action import (
    DiscardAction,
    InternalAction,
    PassAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import (
    Tile,
    TileCategory,
    TileType,
    tile_sort_key,
)

_WINNING_ACTION_TYPES = (RonAction, TsumoAction)
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


class TwoStepUkeirePolicyError(Exception):
    """入力不整合または未定義の状況をPolicyがfail closedする場合。"""


def _winning_action_sort_key(action: RonAction | TsumoAction) -> tuple[object, ...]:
    if isinstance(action, RonAction):
        return (
            0,
            int(action.actor),
            int(action.target),
            tile_sort_key(action.winning_tile),
        )
    return (1, int(action.actor), tile_sort_key(action.winning_tile))


def _remove_one_matching_tile(
    concealed_tiles: tuple[Tile, ...], tile: Tile
) -> list[Tile]:
    """実際のdiscard identityと一致する牌を純手牌から1枚だけ除く。"""
    remaining = list(concealed_tiles)
    for index, candidate in enumerate(remaining):
        if candidate == tile:
            del remaining[index]
            return remaining
    raise TwoStepUkeirePolicyError(
        "DiscardAction.tile has no matching tile in own_hand.concealed_tiles"
    )


def _remove_one_tile_type(tiles: Sequence[Tile], tile_type: TileType) -> list[Tile]:
    """仮想branchで基礎牌種が一致する牌を1枚だけ除く。"""
    remaining = list(tiles)
    for index, candidate in enumerate(remaining):
        if candidate.tile_type == tile_type:
            del remaining[index]
            return remaining
    raise TwoStepUkeirePolicyError(
        "virtual discard tile type has no matching tile in the hypothetical hand"
    )


def _count_tile_types(tiles: Sequence[Tile]) -> dict[TileType, int]:
    counts: dict[TileType, int] = {}
    for tile in tiles:
        counts[tile.tile_type] = counts.get(tile.tile_type, 0) + 1
    return counts


class _DecisionShantenEvaluator:
    """1 decision内の同一structural handだけを再利用する局所cache。"""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, ...], int] = {}

    def calculate(self, hand: Sequence[Tile]) -> int:
        counts = _count_tile_types(hand)
        key = tuple(counts.get(tile_type, 0) for tile_type in _ALL_TILE_TYPES)
        if key not in self._cache:
            self._cache[key] = calculate_shanten(hand)
        return self._cache[key]


def _calculate_shanten(
    hand: Sequence[Tile], evaluator: _DecisionShantenEvaluator | None
) -> int:
    if evaluator is None:
        return calculate_shanten(hand)
    return evaluator.calculate(hand)


def _known_tile_counts(policy_input: PolicyInput) -> dict[TileType, int]:
    """Policy-visibleな既知牌を基礎牌種単位で数える。"""
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
            raise TwoStepUkeirePolicyError(
                "known tile count is inconsistent with the PolicyInput contract: "
                f"{count} copies of {tile_type} are visible, but at most "
                f"{_MAX_COPIES_PER_TILE_TYPE} exist"
            )
    return counts


def _effective_tile_types(
    hand: Sequence[Tile],
    current_shanten: int | None = None,
    evaluator: _DecisionShantenEvaluator | None = None,
) -> tuple[TileType, ...]:
    """現在向聴数を実際に下げるstructuralな基礎牌種を返す。"""
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


def _ukeire_count(
    hand: Sequence[Tile],
    known_counts: Mapping[TileType, int],
    current_shanten: int | None = None,
    evaluator: _DecisionShantenEvaluator | None = None,
) -> int:
    """Policy-visibleな未見枚数による現在受け入れを返す。"""
    return sum(
        _MAX_COPIES_PER_TILE_TYPE - known_counts.get(tile_type, 0)
        for tile_type in _effective_tile_types(hand, current_shanten, evaluator)
    )


def _known_counts_after_draw(
    known_counts: Mapping[TileType, int], tile_type: TileType
) -> dict[TileType, int]:
    """仮想ツモを新しい既知牌として1枚追加する。"""
    current = known_counts.get(tile_type, 0)
    if current >= _MAX_COPIES_PER_TILE_TYPE:
        raise TwoStepUkeirePolicyError(
            "cannot draw a tile type with no Policy-visible remaining copy"
        )
    updated = dict(known_counts)
    updated[tile_type] = current + 1
    return updated


def _virtual_discard_tile_types(hand: Sequence[Tile]) -> tuple[TileType, ...]:
    """仮想手牌に存在する基礎牌種をcanonical順で重複なく返す。"""
    present = frozenset(tile.tile_type for tile in hand)
    return tuple(tile_type for tile_type in _ALL_TILE_TYPES if tile_type in present)


def _best_next_ukeire(
    post_discard_hand: Sequence[Tile],
    drawn_tile_type: TileType,
    known_counts_after_draw: Mapping[TileType, int],
    evaluator: _DecisionShantenEvaluator | None = None,
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
        _ukeire_count(
            _remove_one_tile_type(hypothetical_hand, discard_tile_type),
            known_counts_after_draw,
            shanten,
            evaluator,
        )
        for shanten, discard_tile_type in evaluated
        if shanten == minimum_shanten
    )


def _second_step_score(
    post_discard_hand: Sequence[Tile],
    known_counts: Mapping[TileType, int],
    current_shanten: int | None = None,
    evaluator: _DecisionShantenEvaluator | None = None,
) -> int:
    """`Σ remaining(t) * best_next_ukeire(t)`を整数で返す。"""
    shanten = (
        _calculate_shanten(post_discard_hand, evaluator)
        if current_shanten is None
        else current_shanten
    )
    score = 0
    for tile_type in _effective_tile_types(post_discard_hand, shanten, evaluator):
        remaining = _MAX_COPIES_PER_TILE_TYPE - known_counts.get(tile_type, 0)
        if remaining <= 0:
            continue
        after_draw = _known_counts_after_draw(known_counts, tile_type)
        score += remaining * _best_next_ukeire(
            post_discard_hand, tile_type, after_draw, evaluator
        )
    return score


def _choose_discard(
    policy_input: PolicyInput, discard_actions: tuple[DiscardAction, ...]
) -> DiscardAction:
    known_counts = _known_tile_counts(policy_input)
    concealed_tiles = policy_input.own_hand.concealed_tiles
    evaluator = _DecisionShantenEvaluator()

    evaluated = tuple(
        (
            evaluator.calculate(remaining_hand),
            action,
            remaining_hand,
        )
        for action in discard_actions
        for remaining_hand in (_remove_one_matching_tile(concealed_tiles, action.tile),)
    )
    minimum_shanten = min(shanten for shanten, _, _ in evaluated)
    minimum_shanten_candidates = tuple(
        (action, hand)
        for shanten, action, hand in evaluated
        if shanten == minimum_shanten
    )

    with_current_ukeire = tuple(
        (
            _ukeire_count(hand, known_counts, minimum_shanten, evaluator),
            action,
            hand,
        )
        for action, hand in minimum_shanten_candidates
    )
    maximum_current_ukeire = max(
        current_ukeire for current_ukeire, _, _ in with_current_ukeire
    )
    finalists = tuple(
        (action, hand)
        for current_ukeire, action, hand in with_current_ukeire
        if current_ukeire == maximum_current_ukeire
    )

    if len(finalists) == 1:
        return finalists[0][0]

    if minimum_shanten == 0:
        return min(
            (action for action, _ in finalists),
            key=lambda action: (tile_sort_key(action.tile), action.tsumogiri),
        )

    with_second_step = tuple(
        (
            _second_step_score(hand, known_counts, minimum_shanten, evaluator),
            action,
        )
        for action, hand in finalists
    )
    maximum_second_step = max(score for score, _ in with_second_step)
    return min(
        (action for score, action in with_second_step if score == maximum_second_step),
        key=lambda action: (tile_sort_key(action.tile), action.tsumogiri),
    )


class TwoStepUkeirePolicy:
    """同向聴・同現在受け入れ候補だけを2段階受け入れで比較するPolicy。"""

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        winning_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, _WINNING_ACTION_TYPES)
        )
        if winning_actions:
            return min(winning_actions, key=_winning_action_sort_key)

        discard_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, DiscardAction)
        )
        if discard_actions:
            return _choose_discard(decision.input, discard_actions)

        pass_actions = tuple(
            action
            for action in decision.legal_actions
            if isinstance(action, PassAction)
        )
        if pass_actions:
            return pass_actions[0]

        if len(decision.legal_actions) == 1:
            return decision.legal_actions[0]

        raise TwoStepUkeirePolicyError(
            "no winning action, discard, or pass is available and multiple "
            "non-discard candidates remain without a defined conservative rule"
        )
