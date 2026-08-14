"""送信予定Actionと、server提示`possible_actions`との送信前semantic validation。

Issue #38の中心責務。raw dict完全一致やlist indexへ依存せず、Action typeごとに
serverが合法選択を識別するために意味を持つfieldだけへ正規化して照合する。

- semantic match 0件 -> reject
- semantic match 1件 -> accept
- semantic match複数件 -> reject(ambiguity)

`possible_actions[0]`等のarbitrary fallbackはこのmoduleを含め一切行わない。
"""

from collections.abc import Mapping, Sequence
from typing import assert_never

from lisjong.policy_contract.action import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DiscardAction,
    InternalAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.tile import Tile, tile_sort_key
from lisjong.riichienv_adapter.tile_conversion import tile_from_mjai
from lisjong.riichilab_adapter.errors import PossibleActionsValidationError

# RiichiEnv 0.4.8の`Action.to_mjai()`実測、およびIssue #38本文が例示する
# Action typeに基づくMJAI type文字列。RiichiLab公式protocolの正本は
# `docs/riichilab-adapter.md`を参照。
_CALL_TYPES = {"chi": 2, "pon": 2, "daiminkan": 3}

# 単一のexcept節で複数typeを指定するとparenthesizeが必要になるが、ローカルの
# ruff format実行環境で括弧が意図せず削除される既知の問題があるため(既存
# riichienv_adapter/materialized_state.pyの同種の回避策を参照)、named
# constantへ切り出して単一nameのexceptにしている。
_TILE_FROM_MJAI_ERRORS = (TypeError, ValueError)


def _sorted_tiles(tiles: Sequence[Tile]) -> tuple[Tile, ...]:
    return tuple(sorted(tiles, key=tile_sort_key))


def _selected_semantic_key(selected: InternalAction) -> tuple:
    """resolve済みcanonical `InternalAction`から、送信前validation用semantic keyを作る。

    `InternalAction`はすでにcontext整合が確認済みのvalue型であるため、文字列
    表現へ変換せず型付きfieldを直接使用する。
    """
    if isinstance(selected, DiscardAction):
        return ("dahai", int(selected.actor), selected.tile, selected.tsumogiri)
    if isinstance(selected, RiichiAction):
        return ("reach", int(selected.actor))
    if isinstance(selected, ChiAction):
        return (
            "chi",
            int(selected.actor),
            int(selected.target),
            selected.called_tile,
            _sorted_tiles(selected.consumed_tiles),
        )
    if isinstance(selected, PonAction):
        return (
            "pon",
            int(selected.actor),
            int(selected.target),
            selected.called_tile,
            _sorted_tiles(selected.consumed_tiles),
        )
    if isinstance(selected, DaiminkanAction):
        return (
            "daiminkan",
            int(selected.actor),
            int(selected.target),
            selected.called_tile,
            _sorted_tiles(selected.consumed_tiles),
        )
    if isinstance(selected, AnkanAction):
        return ("ankan", int(selected.actor), _sorted_tiles(selected.tiles))
    if isinstance(selected, KakanAction):
        return ("kakan", int(selected.actor), selected.added_tile)
    if isinstance(selected, RonAction):
        return (
            "hora",
            int(selected.actor),
            int(selected.target),
            selected.winning_tile,
        )
    if isinstance(selected, TsumoAction):
        # mjaiの一般的なhora表現では、tsumoのtargetは自分自身になる。
        return (
            "hora",
            int(selected.actor),
            int(selected.actor),
            selected.winning_tile,
        )
    if isinstance(selected, PassAction):
        return ("none", int(selected.actor))
    if isinstance(selected, KyuushuKyuuhaiAction):
        return ("ryukyoku", int(selected.actor))
    assert_never(selected)


def _tile_from_candidate_field(value: object) -> Tile | None:
    if not isinstance(value, str):
        return None
    try:
        return tile_from_mjai(value)
    except _TILE_FROM_MJAI_ERRORS:
        return None


def _tile_multiset_from_candidate_field(
    value: object, expected_count: int
) -> tuple[Tile, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    items = tuple(value)
    if len(items) != expected_count:
        return None

    tiles = []
    for item in items:
        tile = _tile_from_candidate_field(item)
        if tile is None:
            return None
        tiles.append(tile)
    return _sorted_tiles(tiles)


def _int_field(value: object) -> int | None:
    # boolはintのサブクラスであり、actor/targetとして誤って受理しないよう
    # 明示的に除外する。
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _candidate_semantic_key(candidate: object) -> tuple | None:
    """1件のserver候補を正規化する。malformed / unknownは`None`(非一致)を返す。

    候補列全体をraiseで中断させず、個々の不正候補を「一致しない候補」として
    扱うことで、他の正当な候補への一致判定を継続できるようにする。
    """
    if not isinstance(candidate, Mapping):
        return None

    action_type = candidate.get("type")
    actor = _int_field(candidate.get("actor"))
    if actor is None:
        return None

    if action_type == "dahai":
        pai = _tile_from_candidate_field(candidate.get("pai"))
        tsumogiri = candidate.get("tsumogiri")
        if pai is None or type(tsumogiri) is not bool:
            return None
        return ("dahai", actor, pai, tsumogiri)

    if action_type == "reach":
        return ("reach", actor)

    if action_type in _CALL_TYPES:
        target = _int_field(candidate.get("target"))
        pai = _tile_from_candidate_field(candidate.get("pai"))
        consumed = _tile_multiset_from_candidate_field(
            candidate.get("consumed"), _CALL_TYPES[action_type]
        )
        if target is None or pai is None or consumed is None:
            return None
        return (action_type, actor, target, pai, consumed)

    if action_type == "ankan":
        tiles = _tile_multiset_from_candidate_field(candidate.get("consumed"), 4)
        if tiles is None:
            return None
        return ("ankan", actor, tiles)

    if action_type == "kakan":
        pai = _tile_from_candidate_field(candidate.get("pai"))
        if pai is None:
            return None
        return ("kakan", actor, pai)

    if action_type == "hora":
        target = _int_field(candidate.get("target"))
        pai = _tile_from_candidate_field(candidate.get("pai"))
        if target is None or pai is None:
            return None
        return ("hora", actor, target, pai)

    if action_type == "none":
        return ("none", actor)

    if action_type == "ryukyoku":
        return ("ryukyoku", actor)

    # forward compatibility: 未知のAction typeは、それ単体を理由に全体を失敗
    # させず、単に一致し得ない候補として扱う。
    return None


def validate_against_possible_actions(
    selected: InternalAction, possible_actions: Sequence[object]
) -> None:
    """送信予定`selected`が`possible_actions`へ一意にsemantic matchすることを確認する。

    0件一致、複数件一致(ambiguous)のいずれも
    `PossibleActionsValidationError`でfail closedする。一致したcandidateの
    値そのものは戻り値として使わない(送信payloadはあくまでresolve済みの
    canonical Actionから構築済みのものを使う)。
    """
    target_key = _selected_semantic_key(selected)

    match_count = 0
    for candidate in possible_actions:
        if _candidate_semantic_key(candidate) == target_key:
            match_count += 1

    if match_count == 0:
        raise PossibleActionsValidationError(
            "selected action matches no possible_actions candidate"
        )
    if match_count > 1:
        raise PossibleActionsValidationError(
            "selected action matches multiple possible_actions candidates; "
            f"found {match_count} ambiguous matches"
        )
