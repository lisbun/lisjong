"""RiichiLab WebSocket Client専用の例外。

`docs/riichilab-client.md`「責務境界」「fail closed」を実装する。#38
(`lisjong.riichilab_adapter`)、#34 (`lisjong.policy_contract`)、#23
(`lisjong.riichienv_adapter`)が送出する例外はここで変更・再wrapせず、
そのまま`ValidationSession` / `run_validation()`を通じて伝播させる。
"""


class RiichiLabClientError(Exception):
    """RiichiLab WebSocket Client境界のfail closed例外の基底class。"""


class ProtocolError(RiichiLabClientError):
    """server messageがtransport lifecycle契約に違反している場合。

    JSON parse不能、既知lifecycle eventの必須field欠落・型不正、
    `start_game`前の`request_action`、seat不一致、`request_id`
    lifecycle違反(duplicate/old/decreasing/response mismatch)、
    `action_ack`のprotocol不整合(unknown request_id、unknown status、
    `rejected`/`unparseable`)、`validation_result`のmalformed `passed`、
    response serialization失敗を含む。
    """


class TransportError(RiichiLabClientError):
    """WebSocket接続そのものの送受信が失敗した場合。"""


class UnexpectedDisconnectError(TransportError):
    """validation完了(`validation_result`受信)前にconnectionが切断された場合。

    公式protocol上mid-game reconnectはサポートされないため、この例外は
    成功として扱わない。自動的なreconnectやretryは行わない。
    """


__all__ = [
    "ProtocolError",
    "RiichiLabClientError",
    "TransportError",
    "UnexpectedDisconnectError",
]
