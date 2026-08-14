"""parsed済みのRiichiLab `request_action`相当dataを、安全な境界objectへ変換する。

入力はWebSocket JSON textをすでにparseしたmapping相当object(dict等)を
前提とする。WebSocket受信そのもの、`request_id`のgame内lifecycle管理、
`time`をPolicy入力として扱うことは、この境界の責務ではない
(`docs/riichilab-adapter.md`を参照)。

未知の追加fieldは、それだけを理由に拒否しない。少なくとも次はfail closedする。

- `type != "request_action"`
- `request_id` / `possible_actions` / `observation`の欠落
- `request_id`が現在仕様上安全にechoできない型
- `possible_actions`が期待するcollection形でない
- `observation`が文字列でない、またはdeserialize不能
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from riichienv import Observation

from lisjong.riichilab_adapter.errors import (
    MalformedRequestActionError,
    ObservationDeserializeError,
)


@dataclass(frozen=True, slots=True)
class ParsedRequestAction:
    """安全性を確認した`request_action`の最小限の境界表現。

    `time`は存在すれば保持するだけで、Policy / `DecisionContext`へは
    決して渡さない(transport lifecycle情報であり、Policy入力ではない)。
    """

    request_id: object
    possible_actions: tuple
    observation: Observation
    time: object


def parse_request_action(raw_request_action: object) -> ParsedRequestAction:
    """parsed済みのRiichiLab `request_action`をfail closedで検証・復元する。

    `riichienv.Observation.deserialize_from_base64()`をObservation復元の
    正本として使用し、独自のObservation構築を行わない。
    """
    if not isinstance(raw_request_action, Mapping):
        raise MalformedRequestActionError("request_action must be a mapping")

    request_type = raw_request_action.get("type")
    if request_type != "request_action":
        raise MalformedRequestActionError(
            f"unexpected request_action type: {request_type!r}"
        )

    if "request_id" not in raw_request_action:
        raise MalformedRequestActionError("request_action is missing request_id")
    request_id = raw_request_action["request_id"]
    # bool はint のサブクラスであり、意図せずechoされるとrequest_idの型を
    # 誤って書き換えることになるため、str/int だけを安全にecho可能な型とする。
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
        raise MalformedRequestActionError(
            "request_id must be a str or int that can be echoed safely"
        )

    if "possible_actions" not in raw_request_action:
        raise MalformedRequestActionError("request_action is missing possible_actions")
    possible_actions = raw_request_action["possible_actions"]
    if not isinstance(possible_actions, Sequence) or isinstance(
        possible_actions, (str, bytes)
    ):
        raise MalformedRequestActionError(
            "possible_actions must be a list-like collection of candidates"
        )
    possible_actions = tuple(possible_actions)

    if "observation" not in raw_request_action:
        raise MalformedRequestActionError("request_action is missing observation")
    encoded_observation = raw_request_action["observation"]
    if not isinstance(encoded_observation, str):
        raise MalformedRequestActionError("observation must be a base64-encoded str")

    try:
        observation = Observation.deserialize_from_base64(encoded_observation)
    except Exception as error:
        raise ObservationDeserializeError(
            "observation could not be deserialized as a 4-player Observation"
        ) from error

    if not isinstance(observation, Observation):
        raise ObservationDeserializeError(
            "deserialized observation is not a 4-player riichienv.Observation"
        )

    return ParsedRequestAction(
        request_id=request_id,
        possible_actions=possible_actions,
        observation=observation,
        time=raw_request_action.get("time"),
    )
