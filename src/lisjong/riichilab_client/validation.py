"""RiichiLab `/ws/validate` validationを1回実行する公開API。

`docs/riichilab-client.md`「公開API」を実装する。呼び出し側は
`run_validation(policy, token)`だけを使えばよい。WebSocket接続、
`request_id` / `action_ack`のtransport lifecycle管理、#38
`RiichiLabSeatAdapter`の呼び出しは、この関数が内部で組み立てる。

`python -m lisjong.riichilab_client.validation`として、環境変数
`BOT_TOKEN`からtokenを読み込むCLI entry pointも提供する
(live validationをユーザー環境から実行するため)。
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass

from lisjong.policies import MinimalPolicy
from lisjong.policy_contract.policy import Policy
from lisjong.riichilab_client.errors import RiichiLabClientError
from lisjong.riichilab_client.session import ValidationSession
from lisjong.riichilab_client.transport import (
    DEFAULT_VALIDATION_URL,
    connect_validation_transport,
    drive_validation_session,
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """1回のvalidation gameの終了結果。

    `passed`(`validation_result.passed`をそのまま採用)が成功の正本。
    tokenやraw Observation、raw request_action全文等のsecretを含み得る
    transport dataは保持しない。
    """

    passed: bool
    validation_result_received: bool
    end_game_received: bool
    failure_reason: str | None
    requests_received: int
    responses_sent: int
    ack_history: Mapping[int, tuple[str, ...]]


async def run_validation(
    policy: Policy,
    token: str,
    *,
    url: str = DEFAULT_VALIDATION_URL,
) -> ValidationResult:
    """`url`のRiichiLab validation endpointへ接続し、1 gameを完走する。

    `policy`はPolicy契約(`DecisionContext` -> `InternalAction`)だけを
    実装すればよく、WebSocket/transport固有の型を意識しない。`token`は
    Authorization headerへ設定するためだけに使う runtime secretであり、
    戻り値の`ValidationResult`には含まれない。

    失敗時(protocol違反、transport failure、unexpected disconnect、
    Adapter/Policyの例外)は`ValidationResult`を返さず、対応する例外を
    送出する。arbitrary fallbackは行わない。
    """
    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")

    session = ValidationSession(policy)
    async with connect_validation_transport(url, token) as transport:
        await drive_validation_session(session, transport)

    status = session.status()
    return ValidationResult(
        passed=bool(status.passed),
        validation_result_received=status.validation_result_received,
        end_game_received=status.end_game_received,
        failure_reason=status.failure_reason,
        requests_received=status.requests_received,
        responses_sent=status.responses_sent,
        ack_history=status.ack_history,
    )


def _run_cli() -> int:
    """`python -m lisjong.riichilab_client.validation`のentry point。

    `BOT_TOKEN`環境変数からtokenを読み込む。secretはstdout/stderrへ
    出力しない。
    """
    token = os.environ.get("BOT_TOKEN")
    if not token:
        print(
            "BOT_TOKEN environment variable is not set. "
            "Set BOT_TOKEN to your RiichiLab bot token and re-run.",
            file=sys.stderr,
        )
        return 2

    policy = MinimalPolicy()
    try:
        result = asyncio.run(run_validation(policy, token))
    except RiichiLabClientError as error:
        print(
            f"RiichiLab validation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    if result.passed:
        print("RiichiLab validation passed")
    else:
        print("RiichiLab validation failed")
        if result.failure_reason:
            print(f"reason: {result.failure_reason}")
    print(f"requests: {result.requests_received}")
    print(f"responses: {result.responses_sent}")
    print(f"end_game: {'yes' if result.end_game_received else 'no'}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(_run_cli())


__all__ = ["ValidationResult", "run_validation"]
