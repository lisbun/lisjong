"""RiichiLab `/ws/validate` validationを1回実行する公開API。

`docs/riichilab-client.md`「公開API」を実装する。呼び出し側は
`run_validation(policy, token)`だけを使えばよい。WebSocket接続、
`request_id` / `action_ack`のtransport lifecycle管理、#38
`RiichiLabSeatAdapter`の呼び出しは、この関数が内部で組み立てる。

`python -m lisjong.riichilab_client.validation --profile <name>`として、
Issue #44のprofile層(`lisjong.riichilab_client.profile` / `cli`)が解決した
credential・Policy・trace pathを注入するCLI entry pointも提供する。
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lisjong.policy_contract.policy import Policy
from lisjong.riichilab_client.cli import build_arg_parser, resolve_trace_path
from lisjong.riichilab_client.errors import RiichiLabClientError
from lisjong.riichilab_client.profile import (
    ProfileError,
    build_runtime_summary,
    format_runtime_summary,
    resolve_credential,
    resolve_profile,
)
from lisjong.riichilab_client.session import ValidationSession
from lisjong.riichilab_client.trace import JsonlProtocolTraceWriter
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
    trace_path: str | os.PathLike | None = None,
) -> ValidationResult:
    """`url`のRiichiLab validation endpointへ接続し、1 gameを完走する。

    `policy`はPolicy契約(`DecisionContext` -> `InternalAction`)だけを
    実装すればよく、WebSocket/transport固有の型を意識しない。`token`は
    Authorization headerへ設定するためだけに使う runtime secretであり、
    戻り値の`ValidationResult`には含まれない。

    `trace_path`(Issue #45、既定`None`)を渡した場合だけ、送受信した
    protocol eventをsecret-safeなJSONLとして`trace_path`へ追記する。
    `token`とは独立したopt-in設定であり、`trace_path`を渡さない限り
    trace fileは作られず、既存の挙動も変わらない。

    失敗時(protocol違反、transport failure、unexpected disconnect、
    Adapter/Policyの例外、trace書き込み失敗)は`ValidationResult`を
    返さず、対応する例外を送出する。arbitrary fallbackは行わない。
    """
    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")

    session = ValidationSession(policy)
    trace_writer = (
        JsonlProtocolTraceWriter(trace_path) if trace_path is not None else None
    )
    try:
        async with connect_validation_transport(url, token) as transport:
            await drive_validation_session(session, transport, trace=trace_writer)
    finally:
        if trace_writer is not None:
            trace_writer.close()

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


def _run_cli(argv: Sequence[str] | None = None) -> int:
    """`python -m lisjong.riichilab_client.validation --profile <name>`のentry point。

    Issue #44のprofile層を通じて、bot identity・credential環境変数・Policy・
    runtime namespaceを一方向に解決する。`--profile`未指定・未知profile・
    対応credential未設定は、いずれもfail closed(non-zero exit、secretを
    含まないメッセージ)として扱い、他profileへ暗黙fallbackしない。secretは
    stdout/stderrへ出力しない。

    protocol trace(Issue #45)は既定OFFのopt-inのまま維持する。
    `--trace-path`(明示指定) > 既存`RIICHILAB_TRACE_PATH`環境変数
    (後方互換) > `--trace`(profile既定path) > 無効、の優先順位で解決する。
    """
    parser = build_arg_parser(prog="python -m lisjong.riichilab_client.validation")
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
    summary = build_runtime_summary(profile, mode="validation", trace_path=trace_path)
    print(format_runtime_summary(summary))

    try:
        result = asyncio.run(
            run_validation(profile.policy_factory(), token, trace_path=trace_path)
        )
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
