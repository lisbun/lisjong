import subprocess
import sys
import unittest

_EXPECTED_NAMES = (
    "Seat",
    "Tile",
    "TileCategory",
    "TileType",
    "Wind",
    "MeldKind",
    "PublicMeld",
    "RiichiState",
    "Discard",
    "DiscardAction",
    "InternalAction",
    "PlayerPublicState",
    "RoundState",
    "OwnHandState",
    "PolicyInput",
    "DecisionContext",
    "Policy",
    "PolicyActionValidationError",
    "execute_policy",
)

_PROBE_SCRIPT = (
    "import sys\n"
    "import lisjong.policy_contract as module\n"
    "import lisjong.policies\n"
    "assert 'riichienv' not in sys.modules, sorted(sys.modules)\n"
    "assert 'mjai' not in sys.modules, sorted(sys.modules)\n"
    "assert 'websocket' not in sys.modules, sorted(sys.modules)\n"
    + "\n".join(
        f"assert hasattr(module, {name!r}), {name!r}" for name in _EXPECTED_NAMES
    )
)


class PolicyContractImportTest(unittest.TestCase):
    def test_policy_contract_and_policies_import_without_riichienv(self) -> None:
        # `python -m unittest discover`は全test moduleを同一processへimportする
        # ため、riichienvへ正当に依存するtest(tests/test_riichienv_adapter_*.py)が
        # 同じ実行に含まれると、共有sys.modulesではこのpackage自体の依存を
        # 検証できない。独立したsubprocessでlisjong.policy_contractとpoliciesを
        # importし、riichienvが道連れでimportされないことを確認する。
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
