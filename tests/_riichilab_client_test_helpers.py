"""Issue #39 test群が共有する、実RiichiEnv Observationからserver-style
`request_action`を組み立てるhelper。

`test_riichilab_adapter.py`の`_server_style_request_action()`と同じ
candidate正規化方針(公式`possible_actions` schemaのidentityでdedupe、
hora/call系へ`pai`を補う)を採用する。このmoduleはIssue #39固有のtest
(pure lifecycle / fake transport / #38 integration)が共有するために
独立させたものであり、Issue #38のtest内部実装への依存を作らない。
"""

import json

from riichienv import ActionType

from lisjong.riichienv_adapter.tile_conversion import (
    tile_from_physical_id,
    tile_to_mjai,
)

_CALL_TYPES = {"chi", "pon", "daiminkan"}


def server_style_request_action(observation, request_id: int) -> dict:
    legal = observation.legal_actions()
    seen = set()
    possible_actions = []
    for action in legal:
        candidate = json.loads(action.to_mjai())

        if candidate["type"] in _CALL_TYPES:
            candidate["pai"] = tile_to_mjai(tile_from_physical_id(action.tile))
        elif candidate["type"] == "hora":
            candidate["pai"] = tile_to_mjai(tile_from_physical_id(action.tile))

        dedupe_key = json.dumps(candidate, sort_keys=True)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        candidate["actor"] = action.actor
        if candidate["type"] == "dahai":
            candidate["tsumogiri"] = action.tile == observation.drawn_tile
        elif candidate["type"] in _CALL_TYPES:
            candidate["target"] = observation.last_discard
        elif candidate["type"] == "hora":
            candidate["target"] = (
                action.actor
                if action.action_type == ActionType.TSUMO
                else observation.last_discard
            )

        possible_actions.append(candidate)

    return {
        "type": "request_action",
        "request_id": request_id,
        "possible_actions": possible_actions,
        "observation": observation.serialize_to_base64(),
    }


def resolve_for_env(observation, action: dict):
    """testのgame進行用に、送信済みresponse dictと同じ選択をRiichiEnv Actionへ戻す。

    `RiichiLabSeatAdapter.process_request_action()`はすでに`resolve()`を
    内部で消費済みのため、production契約の一部ではない。
    """
    response_type = action.get("type")
    response_actor = action.get("actor")
    response_pai = action.get("pai")

    for candidate_action in observation.legal_actions():
        candidate_type = json.loads(candidate_action.to_mjai()).get("type")
        if candidate_type != response_type or candidate_action.actor != response_actor:
            continue
        if response_pai is None:
            return candidate_action
        if tile_to_mjai(tile_from_physical_id(candidate_action.tile)) == response_pai:
            return candidate_action
    raise AssertionError("could not resolve a matching RiichiEnv action for test")
