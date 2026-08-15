"""向聴数と受け入れ枚数を順に比較する、2番目の麻雀戦略的Policy。

`ShantenPolicy`(Issue #51)は打牌後向聴数だけを最小化し、同向聴の候補を
`tile_sort_key()`によるtie-breakで選ぶ。実戦では同じ向聴数になる打牌候補が
複数存在することが多く、向聴数だけでは「どちらがより和了へ進みやすいか」を
区別できない。

`UkeirePolicy`はIssue #52として、最小向聴数の候補だけを対象に受け入れ枚数を
比較し、より多くの有効牌を残す打牌を決定的に選ぶ。優先順位は

    打牌後向聴数 > 受け入れ枚数 > semantic fieldだけのstable tie-break

であり、受け入れ枚数が大きいことを理由に、向聴数が悪い候補を最小向聴候補より
優先することはない。

選択規則:

1. `RonAction` / `TsumoAction`が存在すれば、和了候補だけを対象にする。
2. `DiscardAction`が1件以上あれば、各候補の打牌後純手牌を作って
   `calculate_shanten()`で比較し、最小向聴数の候補だけを残す。残った候補に
   ついてのみ受け入れ枚数を求め、最大の候補を選ぶ。受け入れ枚数まで同値なら
   `tile_sort_key(action.tile)`と`action.tsumogiri`でtie-breakする。
3. 打牌候補がなく`PassAction`が合法なら、それを選ぶ。
4. 和了・打牌・Passのいずれもなく、合法候補が1件だけならその強制候補を返す。
5. それ以外は、根拠のないaction type順で選ばずfail closedする。

`MinimalPolicy` / `ShantenPolicy` / `UkeirePolicy`は、後続のPolicy比較で
そのまま並べられるよう、互いに独立した世代として保持する。そのため、この
moduleは`shanten.py`のprivate helperを共有せず、`ShantenPolicy`の意味と挙動を
一切変更しない。打牌simulationやwinning actionのtie-breakは意図的に小さな
重複として持ち、`ShantenPolicy`側の安定した実装をDRY化のためだけに動かす
ことを避ける。

このPolicyはIssue #52のscope内で、打点評価・ドラ価値・鳴きや立直の期待値
評価・他家の待ち推測・守備を一切行わない。それらは後続Issueの責務である。
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
"""赤5と通常5を合算した基礎牌種1種あたりの実在枚数。"""

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
"""有効牌判定の対象となる34基礎牌種。

`tile_sort_key()`と同じcategory順で固定した並びであり、走査順がPython hashの
iteration orderやobject identityへ依存しないようにする。赤5は基礎牌種
`TileType`へ集約されるため、この列に独立した要素として現れない。
"""


class UkeirePolicyError(Exception):
    """`UkeirePolicy`が入力の不整合または未定義の状況をfail closedする場合。"""


def _winning_action_sort_key(action: RonAction | TsumoAction) -> tuple[object, ...]:
    """和了候補だけを対象にした、semantic fieldだけのstable deterministic key。"""
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
    """`concealed_tiles`から`tile`とsemantic equalityで一致する牌を1枚だけ除く。

    赤5と通常5はDiscardAction identity上区別されるため、赤5を切るActionでは
    赤5だけを、通常5を切るActionでは通常5だけを除く。一致する牌が存在しない
    場合、それはDiscardAction / OwnHandState間の不整合なのでfail closedする
    （別の同基礎牌種で代用しない）。
    """
    remaining = list(concealed_tiles)
    for index, candidate in enumerate(remaining):
        if candidate == tile:
            del remaining[index]
            return remaining
    raise UkeirePolicyError(
        "DiscardAction.tile has no matching tile in own_hand.concealed_tiles"
    )


def _count_tile_types(tiles: Sequence[Tile]) -> dict[TileType, int]:
    """`Tile`列を基礎牌種（`TileType`）ごとの枚数へ集約する。

    赤5と通常5は同じ基礎牌種として合算する。入力順序に依存しない。
    """
    counts: dict[TileType, int] = {}
    for tile in tiles:
        counts[tile.tile_type] = counts.get(tile.tile_type, 0) + 1
    return counts


def _known_tile_counts(policy_input: PolicyInput) -> dict[TileType, int]:
    """`PolicyInput`から観測できる既知牌を、基礎牌種ごとに数える。

    山の内部状態は使わない。この seat が実際に見えている物理牌だけを、
    同じ物理牌相当を1回だけ数えるよう次の規則で集計する。

    - 自分の`own_hand.concealed_tiles`（`drawn_tile`は既にこの中に含まれる
      契約なので、別枚として加算しない）
    - 全seatの公開`PublicMeld.tiles`
    - 全seatの捨て牌のうち`called_by is None`のもの。`called_by`を持つ捨て牌は
      鳴きに使われており、同じ物理牌相当が鳴いた側の`PublicMeld.tiles`にも
      現れるため、meld側だけで数える
    - `round.dora_indicators`

    基礎牌種の既知枚数が4を超える等、`PolicyInput`の意味契約と整合しない状態は
    4へ丸めたり推測で修復したりせず、fail closedする。
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
            raise UkeirePolicyError(
                "known tile count is inconsistent with the PolicyInput contract: "
                f"{count} copies of {tile_type} are visible, but at most "
                f"{_MAX_COPIES_PER_TILE_TYPE} exist"
            )
    return counts


def _effective_tile_types(post_discard_hand: Sequence[Tile]) -> tuple[TileType, ...]:
    """打牌後の純手牌について、向聴数を進める基礎牌種を返す。

    34基礎牌種それぞれを1枚加え、`calculate_shanten()`の結果が現在より小さく
    なる牌種を有効牌とする。通常形・七対子・国士無双の区別はIssue #50の
    公開APIへ委ね、Policy側に別の向聴アルゴリズムや役固有の待ち判定を
    持たない。

    打牌後の純手牌だけで既に4枚ある基礎牌種は、5枚目が実在せず
    `calculate_shanten()`のinvalid inputでもあるため、有効牌候補から除外する。
    自手・河・meld・dora indicatorを合わせた既知枚数が4枚に達しているだけの
    牌種は、構造上の有効牌ではあるので除外せず、未見枚数0として数える
    （`_ukeire_count()`を参照）。
    """
    current_shanten = calculate_shanten(post_discard_hand)
    hand_counts = _count_tile_types(post_discard_hand)
    return tuple(
        tile_type
        for tile_type in _ALL_TILE_TYPES
        if hand_counts.get(tile_type, 0) < _MAX_COPIES_PER_TILE_TYPE
        and calculate_shanten([*post_discard_hand, Tile(tile_type)]) < current_shanten
    )


def _ukeire_count(
    post_discard_hand: Sequence[Tile], known_counts: Mapping[TileType, int]
) -> int:
    """打牌後の純手牌について、Policy-visibleな受け入れ枚数を返す。

    hidden wallの実残枚数ではなく、有効牌ごとに「初期4枚 - 既知牌枚数」を
    合計した未見枚数である。`known_counts`は`_known_tile_counts()`が
    `DecisionContext`単位で求めた値を渡す。候補牌を切っても、その牌は自手から
    自分の河へ移動するだけで、基礎牌種ごとのPolicy-visibleな既知枚数総数は
    変わらないため、discard候補ごとに公開状態を仮想生成しない。
    """
    return sum(
        _MAX_COPIES_PER_TILE_TYPE - known_counts.get(tile_type, 0)
        for tile_type in _effective_tile_types(post_discard_hand)
    )


def _choose_discard(
    policy_input: PolicyInput, discard_actions: tuple[DiscardAction, ...]
) -> DiscardAction:
    """打牌後向聴数を第一基準に、受け入れ枚数を第二基準にして1件を選ぶ。

    受け入れ枚数は最小向聴数の候補についてのみ求める。向聴数が悪い候補は、
    受け入れ枚数がどれだけ大きくてもここへ残らない。
    """
    known_counts = _known_tile_counts(policy_input)
    concealed_tiles = policy_input.own_hand.concealed_tiles

    simulated = [
        (action, _remove_one_matching_tile(concealed_tiles, action.tile))
        for action in discard_actions
    ]
    evaluated = [
        (calculate_shanten(remaining_hand), action, remaining_hand)
        for action, remaining_hand in simulated
    ]
    minimum_shanten = min(shanten for shanten, _, _ in evaluated)

    best_action, _ = min(
        (
            (action, remaining_hand)
            for shanten, action, remaining_hand in evaluated
            if shanten == minimum_shanten
        ),
        key=lambda candidate: (
            -_ukeire_count(candidate[1], known_counts),
            tile_sort_key(candidate[0].tile),
            candidate[0].tsumogiri,
        ),
    )
    return best_action


class UkeirePolicy:
    """最小向聴数の打牌候補を、受け入れ枚数で比較するPolicy。"""

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

        raise UkeirePolicyError(
            "no winning action, discard, or pass is available and multiple "
            "non-discard candidates remain without a defined conservative rule"
        )
