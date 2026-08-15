"""profile CLI共通layer(`lisjong.riichilab_client.cli`)のunit test。

`--profile` / `--trace` / `--trace-path`の引数解析と、trace path解決の
優先順位(明示指定 > `RIICHILAB_TRACE_PATH` > `--trace`既定path > 無効)を、
実CLI subprocessを起動せずに確認する。
"""

import unittest

from lisjong.riichilab_client.cli import (
    TRACE_PATH_ENV_VAR,
    build_arg_parser,
    resolve_trace_path,
)
from lisjong.riichilab_client.profile import resolve_profile


class ArgParserTest(unittest.TestCase):
    def test_profile_is_required(self) -> None:
        parser = build_arg_parser(prog="test")
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_unknown_profile_choice_is_rejected(self) -> None:
        parser = build_arg_parser(prog="test")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--profile", "lisjong-production"])

    def test_known_profile_parses(self) -> None:
        parser = build_arg_parser(prog="test")
        args = parser.parse_args(["--profile", "lisjong-dev"])
        self.assertEqual(args.profile, "lisjong-dev")
        self.assertFalse(args.trace)
        self.assertIsNone(args.trace_path)

    def test_trace_flag_and_trace_path_parse(self) -> None:
        parser = build_arg_parser(prog="test")
        args = parser.parse_args(
            ["--profile", "lisjong", "--trace", "--trace-path", "custom.jsonl"]
        )
        self.assertTrue(args.trace)
        self.assertEqual(args.trace_path, "custom.jsonl")


class ResolveTracePathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = resolve_profile("lisjong-dev")

    def test_defaults_to_disabled(self) -> None:
        path = resolve_trace_path(
            self.profile, trace_flag=False, trace_path_arg=None, env={}
        )
        self.assertIsNone(path)

    def test_explicit_trace_path_arg_wins_over_everything(self) -> None:
        path = resolve_trace_path(
            self.profile,
            trace_flag=True,
            trace_path_arg="explicit.jsonl",
            env={TRACE_PATH_ENV_VAR: "env.jsonl"},
        )
        self.assertEqual(path, "explicit.jsonl")

    def test_env_var_is_used_when_no_explicit_arg(self) -> None:
        path = resolve_trace_path(
            self.profile,
            trace_flag=False,
            trace_path_arg=None,
            env={TRACE_PATH_ENV_VAR: "env.jsonl"},
        )
        self.assertEqual(path, "env.jsonl")

    def test_env_var_wins_over_trace_flag(self) -> None:
        path = resolve_trace_path(
            self.profile,
            trace_flag=True,
            trace_path_arg=None,
            env={TRACE_PATH_ENV_VAR: "env.jsonl"},
        )
        self.assertEqual(path, "env.jsonl")

    def test_trace_flag_uses_profile_default_path_when_nothing_else_set(self) -> None:
        path = resolve_trace_path(
            self.profile, trace_flag=True, trace_path_arg=None, env={}
        )
        self.assertIsNotNone(path)
        self.assertIn("lisjong-dev", path)
        self.assertTrue(path.endswith(".jsonl"))


if __name__ == "__main__":
    unittest.main()
