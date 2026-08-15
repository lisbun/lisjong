"""RiichiLab `/ws/ranked`で1半荘だけ実行する公開APIとCLI。

`run_ranked_game(policy, token, ...)`はPolicyとcredentialを明示的に受け取る
実行境界として維持する。CLI(`_run_cli()`)は`--profile`でIssue #44の
`lisjong-dev` / `lisjong-baseline` / `lisjong` profileを選択し、
`lisjong.riichilab_client.profile` / `cli`が解決したPolicy・credential・
trace pathをこの境界へ明示的に渡すだけのcomposition layerである。
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lisjong.policy_contract.policy import Policy
from lisjong.policy_contract.seat import Seat
from lisjong.riichilab_client.cli import build_arg_parser, resolve_trace_path
from lisjong.riichilab_client.errors import ProtocolError, RiichiLabClientError
from lisjong.riichilab_client.profile import (
    ProfileError,
    build_runtime_summary,
    format_runtime_summary,
    resolve_credential,
    resolve_profile,
)
from lisjong.riichilab_client.session import RankedSession
from lisjong.riichilab_client.trace import JsonlProtocolTraceWriter
from lisjong.riichilab_client.transport import (
    DEFAULT_RANKED_URL,
    connect_ranked_transport,
    drive_ranked_session,
)


@dataclass(frozen=True, slots=True)
class RankedGameResult:
    """1 ranked hanchanのsecret-safeな完走結果。

    実serverの`end_game`にscoresがない場合は`None`を保持する。scoresが通知
    された場合だけ4 seatの値を保持し、順位やratingは推測しない。token、
    Authorization header、raw Observationは含めない。
    """

    end_game_received: bool
    seat: Seat
    requests_received: int
    responses_sent: int
    ack_history: Mapping[int, tuple[str, ...]]
    scores: tuple[int, int, int, int] | None


async def run_ranked_game(
    policy: Policy,
    token: str,
    *,
    url: str = DEFAULT_RANKED_URL,
    trace_path: str | os.PathLike | None = None,
) -> RankedGameResult:
    """ranked endpointへ1回接続し、1 full hanchanの`end_game`で終了する。

    endpointへの接続自体がmatchmaking queue参加であるため、接続直後のjoin
    payloadは送らない。`end_game`後の再queue、次game、自動reconnectも行わない。

    `trace_path`(Issue #45、既定`None`)を渡した場合だけ、送受信した
    protocol eventをsecret-safeなJSONLとして`trace_path`へ追記する。
    `token`とは独立したopt-in設定であり、`trace_path`を渡さない限り
    trace fileは作られない。validationと共通の`drive_session()`を通じて
    同じtrace実装を利用する。
    """
    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")

    session = RankedSession(policy)
    trace_writer = (
        JsonlProtocolTraceWriter(trace_path) if trace_path is not None else None
    )
    try:
        async with connect_ranked_transport(url, token) as transport:
            await drive_ranked_session(session, transport, trace=trace_writer)
    finally:
        if trace_writer is not None:
            trace_writer.close()

    status = session.status()
    if status.seat is None:
        raise ProtocolError("ranked game completed without a bound seat")

    return RankedGameResult(
        end_game_received=status.end_game_received,
        seat=status.seat,
        requests_received=status.requests_received,
        responses_sent=status.responses_sent,
        ack_history=status.ack_history,
        scores=status.scores,
    )


def _run_cli(argv: Sequence[str] | None = None) -> int:
    """`python -m lisjong.riichilab_client.ranked --profile <name>`のentry point。

    Issue #44のprofile層を通じて、bot identity・credential環境変数・Policy・
    runtime namespaceを一方向に解決する。`--profile`未指定・未知profile・
    対応credential未設定は、いずれもfail closed(non-zero exit、secretを
    含まないメッセージ)として扱い、他profileへ暗黙fallbackしない。

    protocol trace(Issue #45)は既定OFFのopt-inのまま維持する。
    `--trace-path`(明示指定) > 既存`RIICHILAB_TRACE_PATH`環境変数
    (後方互換) > `--trace`(profile既定path) > 無効、の優先順位で解決する。
    """
    parser = build_arg_parser(prog="python -m lisjong.riichilab_client.ranked")
    args = parser.parse_args(argv)

    try:
        profile = resolve_profile(args.profile)
        token = resolve_credential(profile)
    except ProfileError as error:
        print(str(error), file=sys.stderr)
        return 2

    trace_path = resolve_trace_path(
        profile, trace_flag=args.trace, trace_path_arg=args.trace_path
    )
    policy = profile.policy_factory()
    summary = build_runtime_summary(
        profile, mode="ranked", trace_path=trace_path, policy=policy
    )
    print(format_runtime_summary(summary))

    try:
        result = asyncio.run(run_ranked_game(policy, token, trace_path=trace_path))
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
    if result.scores is None:
        print("scores: unavailable")
    else:
        print("scores: " + ", ".join(str(score) for score in result.scores))
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())


__all__ = ["RankedGameResult", "run_ranked_game"]
