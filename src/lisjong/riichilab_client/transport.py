"""RiichiLab validation WebSocket接続そのものを扱う最小限のtransport層。

`ValidationSession`(pure transport lifecycle state、`session.py`)と
WebSocket API自体を分離する。fake/local testは`Transport` protocolへ
準拠するfake objectを実装するだけで、実WebSocket接続・asyncio eventなしに
lifecycleを確認できる。

`websockets` dependencyはこのpackage内だけで使用する
(`policy_contract` / `policies` / `riichienv_adapter`への依存の逆流はしない)。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

import websockets
import websockets.exceptions

from lisjong.riichilab_client.errors import (
    ProtocolError,
    TransportError,
    UnexpectedDisconnectError,
)
from lisjong.riichilab_client.session import ValidationSession

DEFAULT_VALIDATION_URL = "wss://game.riichi.dev/ws/validate"


class TransportClosed(Exception):
    """`Transport.recv()`がconnection close(正常/異常問わず)を検出した場合。

    `drive_validation_session()`側で`UnexpectedDisconnectError`へ変換する
    ための内部signalであり、呼び出し側の公開APIには漏らさない。
    """


class Transport(Protocol):
    """`ValidationSession`を駆動するために必要な最小限のWebSocket操作。

    実装はtext/binary frameの生データだけを扱う。JSON parse、binary
    frame ignore、fail closedの判断は`drive_validation_session()`側の
    責務とする。
    """

    async def recv(self) -> str | bytes: ...

    async def send(self, message: str) -> None: ...

    async def close(self) -> None: ...


class WebSocketTransport:
    """`websockets`library上の実接続を`Transport` protocolへ適合させる薄いwrapper。"""

    __slots__ = ("_connection",)

    def __init__(self, connection: object) -> None:
        self._connection = connection

    async def recv(self) -> str | bytes:
        try:
            return await self._connection.recv()
        except websockets.exceptions.ConnectionClosed as error:
            raise TransportClosed(str(error)) from error

    async def send(self, message: str) -> None:
        try:
            await self._connection.send(message)
        except websockets.exceptions.ConnectionClosed as error:
            raise TransportClosed(str(error)) from error

    async def close(self) -> None:
        await self._connection.close()


@asynccontextmanager
async def connect_validation_transport(
    url: str, token: str
) -> AsyncIterator[Transport]:
    """`url`へBearer tokenでWebSocket接続し、`Transport`として提供する。

    `token`はAuthorization headerを設定する目的だけに使い、戻り値の
    `Transport`・結果側には一切保持しない。mid-game reconnectは行わない
    (`websockets.connect()`を`async with`のreconnectループとしてではなく、
    1回の接続としてだけ使用する)。
    """
    headers = {"Authorization": f"Bearer {token}"}
    try:
        connection = await websockets.connect(url, additional_headers=headers)
    except Exception as error:
        raise TransportError(f"failed to connect to {url}") from error

    transport = WebSocketTransport(connection)
    try:
        yield transport
    finally:
        await connection.close()


def parse_json_event(message: str) -> dict:
    """text frameをJSON top-level objectとしてparseする。fail closed。"""
    try:
        parsed = json.loads(message)
    except (TypeError, ValueError) as error:
        raise ProtocolError("received text frame is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise ProtocolError("received JSON is not a top-level object")
    return parsed


async def drive_validation_session(
    session: ValidationSession, transport: Transport
) -> None:
    """`validation_result`受信までtransportから受信し、`session`を進行させる。

    - binary frameはprotocol failureとしてclient全体を落とさずignoreする
      (Issue #39最新コメント)
    - text frameはJSON objectとしてparseし、parse不能・非objectは
      fail closedする
    - unexpected disconnectは`UnexpectedDisconnectError`として成功扱い
      しない。mid-game reconnectは行わない
    """
    while not session.validation_result_received:
        try:
            message = await transport.recv()
        except TransportClosed as error:
            raise UnexpectedDisconnectError(
                "WebSocket connection closed before validation_result was received"
            ) from error

        if isinstance(message, bytes):
            continue

        event = parse_json_event(message)
        outgoing = session.handle_event(event)
        if outgoing is None:
            continue

        try:
            outgoing_text = json.dumps(outgoing)
        except (TypeError, ValueError) as error:
            raise ProtocolError("failed to serialize outgoing action") from error

        try:
            await transport.send(outgoing_text)
        except TransportClosed as error:
            raise TransportError("failed to send action: connection closed") from error


__all__ = [
    "DEFAULT_VALIDATION_URL",
    "Transport",
    "TransportClosed",
    "WebSocketTransport",
    "connect_validation_transport",
    "drive_validation_session",
    "parse_json_event",
]
