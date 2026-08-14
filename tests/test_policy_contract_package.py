import importlib
import subprocess
import sys
import unittest


class PolicyContractImportTest(unittest.TestCase):
    def test_imports_without_riichienv(self) -> None:
        # unittest discoverは全testモジュールをcollection段階で一括importする
        # ため、同一process内では他のtestモジュール（lisjong.riichienv_adapter
        # を使うもの等）がすでにriichienvをsys.modulesへ載せている場合がある。
        # この契約（lisjong.policy_contractがriichienvへ依存しないこと）を
        # process横断の副作用と切り離して検証するため、独立したsubprocessで
        # `import lisjong.policy_contract`だけを行い、そのsubprocess自身の
        # sys.modulesを確認する。
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import lisjong.policy_contract; "
                "sys.exit(1 if 'riichienv' in sys.modules else 0)",
            ],
            cwd=None,
        )
        self.assertEqual(
            result.returncode,
            0,
            "importing lisjong.policy_contract must not load riichienv",
        )

    def test_exposes_expected_names(self) -> None:
        module = importlib.import_module("lisjong.policy_contract")
        for name in (
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
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(module, name))


if __name__ == "__main__":
    unittest.main()
