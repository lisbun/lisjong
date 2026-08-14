"""送信予定Actionと、server提示`possible_actions`との送信前semantic validation。

Issue #38の中心責務。raw dict完全一致やlist indexへ依存せず、送信予定の
Bot-to-Server responseとserver candidateの両方を、同一の
`possible_actions` candidate semantic identityへprojectionしてから照合する。

```text
send-ready Bot response --projection--> candidate semantic identity
server candidate        --projection--> candidate semantic identity
                                        -> semantic equality
```

RiichiLab公式Protocolの`possible_actions` candidate schemaは、Bot-to-Server
response schemaより小さい最小表現である(Issue #38 review、
`docs/riichilab-adapter.md`参照)。そのためidentityは、公式candidateが
identityとして持つfield(`type` / `pai` / `consumed`)だけで構成し、
`actor` / `target` / `tsumogiri`をcandidateへ要求しない。

ただし、公式Protocolは`possible_actions`の例とAction別field表の間に記述差が
あり、candidateへこれらのfieldが付随し得ることまでは否定できない
(Issue #38 再レビュー)。そのためcandidate側に`actor` / `target`が実際に
存在する場合だけは、送信予定responseと矛盾しないことも確認する
(`_optional_fields_agree`)。

- semantic match 0件 -> reject
- semantic match 1件 -> accept
- semantic match複数件 -> reject(ambiguity)

`possible_actions`内に1件でもmalformed candidate、または未知のAction typeの
candidateが含まれる場合、他のcandidateが一致するかどうかにかかわらず
validation全体をfail closedする(Issue #38 再レビュー: forward compatibility
として許容するのは既知Action typeのunknown追加fieldであり、legal candidate
そのもののunknown Action typeやrequired field欠落ではない)。

`possible_actions[0]`等のarbitrary fallbackはこのmoduleを含め一切行わない。
"""

from collections.abc import Mapping, Sequence

from lisjong.policy_contract.tile import Tile, tile_sort_key
from lisjong.riichienv_adapter.tile_conversion import tile_from_mjai
from lisjong.riichilab_adapter.errors import PossibleActionsValidationError

# `pai`(識別に使う牌1枚)をidentityへ持つAction type。
_PAI_ONLY_TYPES = frozenset({"dahai", "hora"})

# `consumed`(手牌等から消費する牌の組)の枚数。RiichiEnv 0.4.8の
# `Action.to_mjai()`実測とIssue #38レビューで確認した公式candidate schemaに
# 基づく。正本は`docs/riichilab-adapter.md`を参照。
#
# `kakan`は、公式candidate schemaが`pai`(加える牌)に加えて`consumed`
# (元Ponの3枚)を持つ(Issue #38 再レビューのblocking finding)。`pai`だけを
# identityとすると、同じ加槓牌でも元Pon構成が異なるcandidateを誤って同一
# 合法Actionとして受理し得るため、`consumed`もidentityへ含める。
_CONSUMED_COUNTS = {"chi": 2, "pon": 2, "daiminkan": 3, "ankan": 4, "kakan": 3}

# `pai`と`consumed`の両方をidentityへ持つAction type。`ankan`は`pai`を
# identityとして使わず、`consumed`の4枚だけで一意に定まる。
_PAI_AND_CONSUMED_TYPES = frozenset({"chi", "pon", "daiminkan", "kakan"})

# 追加のsemantic fieldを持たず、typeだけでidentityが定まるAction type。
_TYPE_ONLY_TYPES = frozenset({"reach", "none", "ryukyoku"})

# candidate側に存在する場合だけ、送信予定responseと矛盾しないことを確認する
# field。`tsumogiri`は含めない: 公式candidate例は`tsumogiri`を持たず、
# 打牌は`pai`で一意に定まるため、candidate側の`tsumogiri`は仮に付随しても
# identityでも矛盾判定材料でもない(Issue #38 review: candidateへ
# `tsumogiri`を要求しない)。
_OPTIONAL_CONSISTENCY_FIELDS = ("actor", "target")

# 単一のexcept節で複数typeを指定するとparenthesizeが必要になるが、ローカルの
# ruff format実行環境で括弧が意図せず削除される既知の問題があるため(既存
# riichienv_adapter/materialized_state.pyの同種の回避策を参照)、named
# constantへ切り出して単一nameのexceptにしている。
_TILE_FROM_MJAI_ERRORS = (TypeError, ValueError)


class _IdentityProjectionError(Exception):
    """MJAI相当mappingをcandidate semantic identityへprojectionできない場合の内部例外。

    このmodule内部だけで使用し、呼び出し側へは
    `PossibleActionsValidationError`として送出しなおす(projection対象が
    server candidateか送信予定responseかで、報告すべき原因が異なるため)。
    """


def _sorted_tiles(tiles: Sequence[Tile]) -> tuple[Tile, ...]:
    return tuple(sorted(tiles, key=tile_sort_key))


def _tile_field(source: Mapping, field: str) -> Tile:
    value = source.get(field)
    if not isinstance(value, str):
        raise _IdentityProjectionError(f"{field} must be an mjai tile string")
    try:
        return tile_from_mjai(value)
    except _TILE_FROM_MJAI_ERRORS as error:
        raise _IdentityProjectionError(f"{field} is not a valid mjai tile") from error


def _tile_multiset_field(
    source: Mapping, field: str, expected_count: int
) -> tuple[Tile, ...]:
    value = source.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _IdentityProjectionError(f"{field} must be a list of mjai tile strings")
    items = tuple(value)
    if len(items) != expected_count:
        raise _IdentityProjectionError(
            f"{field} must contain exactly {expected_count} tiles, got {len(items)}"
        )

    tiles = []
    for item in items:
        if not isinstance(item, str):
            raise _IdentityProjectionError(f"{field} entries must be mjai tile strings")
        try:
            tiles.append(tile_from_mjai(item))
        except _TILE_FROM_MJAI_ERRORS as error:
            raise _IdentityProjectionError(
                f"{field} contains an invalid mjai tile"
            ) from error
    return _sorted_tiles(tiles)


def _semantic_identity(source: Mapping) -> tuple:
    """MJAI相当のaction mappingを`possible_actions` candidate identityへprojectionする。

    server candidateにも、送信予定のBot responseにも同じ関数を適用する
    ことで、raw dict完全一致に頼らず、かつ両者のschema差(candidate側に
    無い`actor` / `target` / `tsumogiri`等)へ影響されない照合を行う。
    """
    if not isinstance(source, Mapping):
        raise _IdentityProjectionError("action must be a mapping")

    action_type = source.get("type")
    if not isinstance(action_type, str):
        raise _IdentityProjectionError("action is missing a string type")

    if action_type in _TYPE_ONLY_TYPES:
        return (action_type,)

    if action_type in _PAI_ONLY_TYPES:
        return (action_type, _tile_field(source, "pai"))

    if action_type == "ankan":
        return (
            action_type,
            _tile_multiset_field(source, "consumed", _CONSUMED_COUNTS[action_type]),
        )

    if action_type in _PAI_AND_CONSUMED_TYPES:
        return (
            action_type,
            _tile_field(source, "pai"),
            _tile_multiset_field(source, "consumed", _CONSUMED_COUNTS[action_type]),
        )

    # forward compatibilityとして許容するのは既知Action typeのunknown追加
    # fieldまでであり、未知のAction type自体はsilent ignoreせずfail closed
    # する(Issue #38 再レビュー)。
    raise _IdentityProjectionError(f"unknown action type: {action_type!r}")


def _optional_fields_agree(candidate: Mapping, response: Mapping) -> bool:
    """candidateが任意で持つsemantic fieldが、送信予定responseと矛盾しないか確認する。

    公式Protocolは`possible_actions`の例とAction別field表の間に記述差が
    あるため、candidateが`actor` / `target`を持ち得ないとは断言できない
    (Issue #38 再レビュー)。candidate側に存在する場合だけ照合し、
    存在しなければidentityだけで判定する(minimal candidate形を拒否しない)。

    矛盾するcandidateは「別のAction候補」として非一致に倒す。結果として
    一致0件になればfail closedするため、誤受理は起こらない。
    """
    for field in _OPTIONAL_CONSISTENCY_FIELDS:
        if field not in candidate:
            continue
        expected = response.get(field)
        if expected is None:
            continue
        candidate_value = candidate[field]
        if isinstance(candidate_value, bool) or not isinstance(candidate_value, int):
            return False
        if candidate_value != expected:
            return False
    return True


def validate_against_possible_actions(
    response: Mapping, possible_actions: Sequence[object]
) -> None:
    """送信予定`response`が`possible_actions`へ一意にsemantic matchすることを確認する。

    `response`は`build_mjai_response()`が構築した、これからserverへ送ろうと
    しているBot-to-Server response相当のMJAI dictである。canonical
    `InternalAction`ではなく実際の送信内容を照合対象にすることで、
    `KakanAction`のようにInternalAction側が保持しない外部semantic情報
    (元Ponの`consumed`)も落とさずに検証できる。

    次のいずれもsend-ready payloadを返さずfail closedする。

    - `possible_actions`内にmalformed candidate、または未知Action typeの
      candidateが1件でも存在する
    - semantic match 0件
    - semantic match複数件(ambiguous)

    一致したcandidateの値そのものは戻り値として使わない(送信payloadは
    あくまでresolve済みcanonical Actionから構築済みのものを使う)。
    """
    try:
        response_identity = _semantic_identity(response)
    except _IdentityProjectionError as error:
        raise PossibleActionsValidationError(
            f"send-ready response could not be projected onto the "
            f"possible_actions candidate schema: {error}"
        ) from error

    match_count = 0
    for index, candidate in enumerate(possible_actions):
        try:
            candidate_identity = _semantic_identity(candidate)
        except _IdentityProjectionError as error:
            raise PossibleActionsValidationError(
                f"possible_actions[{index}] is not a well-formed candidate: {error}"
            ) from error

        if candidate_identity != response_identity:
            continue
        if not _optional_fields_agree(candidate, response):
            continue
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
