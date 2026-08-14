"""送信予定Actionと、server提示`possible_actions`との送信前semantic validation。

Issue #38の中心責務。raw dict完全一致やlist indexへ依存せず、Action typeごとに
serverの`possible_actions` candidateが実際に持つfieldだけへ正規化して照合する。

RiichiLab公式Protocolの`possible_actions` candidate schemaは、Bot-to-Server
response schemaより意図的に小さい最小表現である(Issue #38 review、
`docs/riichilab-adapter.md`参照)。このmoduleはcandidate側のsemantic identity
だけを扱い、`actor` / `target` / `tsumogiri`等のBot response専用fieldを
candidate側へ要求しない。selected側の`InternalAction`も同じ最小identityへ
projectionしてから比較する。

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

# RiichiLab公式`possible_actions`のcandidate schemaにおける、call系(chi/pon/
# daiminkan)候補が持つ`consumed`(手牌から消費する牌)の枚数。RiichiEnv 0.4.8の
# `Action.to_mjai()`実測、およびIssue #38レビューで確認した公式candidate
# schemaに基づく。正本は`docs/riichilab-adapter.md`を参照。
_CALL_TYPES = {"chi": 2, "pon": 2, "daiminkan": 3}

# 単一のexcept節で複数typeを指定するとparenthesizeが必要になるが、ローカルの
# ruff format実行環境で括弧が意図せず削除される既知の問題があるため(既存
# riichienv_adapter/materialized_state.pyの同種の回避策を参照)、named
# constantへ切り出して単一nameのexceptにしている。
_TILE_FROM_MJAI_ERRORS = (TypeError, ValueError)


def _sorted_tiles(tiles: Sequence[Tile]) -> tuple[Tile, ...]:
    return tuple(sorted(tiles, key=tile_sort_key))


def _selected_semantic_key(selected: InternalAction) -> tuple:
    """resolve済みcanonical `InternalAction`を、`possible_actions` candidate側と
    同じ最小semantic identity空間へprojectionする。

    `actor` / `target` / `tsumogiri`は、RiichiLab公式`possible_actions`
    candidateが持たないBot response専用情報であるため、ここでは意図的に
    含めない(1 request_actionのcandidate列はこのAdapterがbindされた1 seat
    分だけであり、actorは常に自明。targetやtsumogiriはBot response
    serialization側(`mjai_response.py`)でのみ使用する)。
    """
    if isinstance(selected, DiscardAction):
        return ("dahai", selected.tile)
    if isinstance(selected, RiichiAction):
        return ("reach",)
    if isinstance(selected, ChiAction):
        return ("chi", selected.called_tile, _sorted_tiles(selected.consumed_tiles))
    if isinstance(selected, PonAction):
        return ("pon", selected.called_tile, _sorted_tiles(selected.consumed_tiles))
    if isinstance(selected, DaiminkanAction):
        return (
            "daiminkan",
            selected.called_tile,
            _sorted_tiles(selected.consumed_tiles),
        )
    if isinstance(selected, AnkanAction):
        return ("ankan", _sorted_tiles(selected.tiles))
    if isinstance(selected, KakanAction):
        return ("kakan", selected.added_tile)
    if isinstance(selected, RonAction):
        return ("hora", selected.winning_tile)
    if isinstance(selected, TsumoAction):
        return ("hora", selected.winning_tile)
    if isinstance(selected, PassAction):
        return ("none",)
    if isinstance(selected, KyuushuKyuuhaiAction):
        return ("ryukyoku",)
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


def _candidate_semantic_key(candidate: object) -> tuple | None:
    """1件のserver `possible_actions` candidateを正規化する。

    RiichiLab公式candidate schemaが実際に持つfieldだけを読み、`actor` /
    `target` / `tsumogiri`等のBot response専用fieldは一切要求しない
    (要求すると、公式candidate `{"type": "dahai", "pai": "1m"}`のような
    最小形が誤ってmalformed判定されfail closedしてしまう)。

    malformed / unknown typeの場合は`None`(非一致)を返す。候補列全体を
    raiseで中断させず、個々の不正候補を「一致しない候補」として扱うことで、
    他の正当な候補への一致判定を継続できるようにする。
    """
    if not isinstance(candidate, Mapping):
        return None

    action_type = candidate.get("type")

    if action_type == "dahai":
        pai = _tile_from_candidate_field(candidate.get("pai"))
        if pai is None:
            return None
        return ("dahai", pai)

    if action_type == "reach":
        return ("reach",)

    if action_type in _CALL_TYPES:
        pai = _tile_from_candidate_field(candidate.get("pai"))
        consumed = _tile_multiset_from_candidate_field(
            candidate.get("consumed"), _CALL_TYPES[action_type]
        )
        if pai is None or consumed is None:
            return None
        return (action_type, pai, consumed)

    if action_type == "ankan":
        tiles = _tile_multiset_from_candidate_field(candidate.get("consumed"), 4)
        if tiles is None:
            return None
        return ("ankan", tiles)

    if action_type == "kakan":
        pai = _tile_from_candidate_field(candidate.get("pai"))
        if pai is None:
            return None
        return ("kakan", pai)

    if action_type == "hora":
        pai = _tile_from_candidate_field(candidate.get("pai"))
        if pai is None:
            return None
        return ("hora", pai)

    if action_type == "none":
        return ("none",)

    if action_type == "ryukyoku":
        return ("ryukyoku",)

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
