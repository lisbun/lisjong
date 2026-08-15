"""RiichiLab `/ws/ranked`で1半荘だけ実行する公開APIとCLI。"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass

from lisjong.policies import MinimalPolicy
from lisjong.policy_contract.policy import Policy
from lisjong.policy_contract.seat import Seat
from lisjong.riichilab_client.errors import ProtocolError, RiichiLabClientError
from lisjong.riichilab_client.session import RankedSession
from lisjong.riichilab_client.transport import (
    DEFAULT_RANKED_URL,
    connect_ranked_transport,
    drive_ranked_session,
)


@dataclass(frozen=True, slots=True)
class RankedGameResult:
    """1 ranked hanchanのsecret-safeな完走結果。

    公式`end_game` schemaで保証されるfinal scoresだけを保持し、順位やratingは
    推測しない。token、Authorization header、raw Observationは含めない。
    """

    end_game_received: bool
    seat: Seat
    requests_received: int
    responses_sent: int
    ack_history: Mapping[int, tuple[str, ...]]
    scores: tuple[int, int, int, int]


async def run_ranked_game(
    policy: Policy,
    token: str,
    *,
    url: str = DEFAULT_RANKED_URL,
) -> RankedGameResult:
    """ranked endpointへ1回接続し、1 full hanchanの`end_game`で終了する。

    endpointへの接続自体がmatchmaking queue参加であるため、接続直後のjoin
    payloadは送らない。`end_game`後の再queue、次game、自動reconnectも行わない。
    """
    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")

    session = RankedSession(policy)
    async with connect_ranked_transport(url, token) as transport:
        await drive_ranked_session(session, transport)

    status = session.status()
    if status.seat is None or status.scores is None:
        raise ProtocolError("ranked game completed without seat or final scores")

    return RankedGameResult(
        end_game_received=status.end_game_received,
        seat=status.seat,
        requests_received=status.requests_received,
        responses_sent=status.responses_sent,
        ack_history=status.ack_history,
        scores=status.scores,
    )


def _run_cli() -> int:
    """`python -m lisjong.riichilab_client.ranked`のentry point。"""
    token = os.environ.get("BOT_TOKEN")
    if not token:
        print(
            "BOT_TOKEN environment variable is not set. "
            "Set BOT_TOKEN to your RiichiLab bot token and re-run.",
            file=sys.stderr,
        )
        return 2

    try:
        result = asyncio.run(run_ranked_game(MinimalPolicy(), token))
    except RiichiLabClientError as error:
        print(
            f"RiichiLab ranked game failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print("RiichiLab ranked game completed")
    print(f"seat: {int(result.seat)}")
    print(f"requests: {result.requests_received}")
    print(f"responses: {result.responses_sent}")
    print(f"end_game: {'yes' if result.end_game_received else 'no'}")
    print("scores: " + ", ".join(str(score) for score in result.scores))
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())


__all__ = ["RankedGameResult", "run_ranked_game"]
