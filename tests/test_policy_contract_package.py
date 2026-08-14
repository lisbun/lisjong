import importlib
import sys
import unittest


class PolicyContractImportTest(unittest.TestCase):
    def test_imports_without_riichienv(self) -> None:
        self.assertNotIn("riichienv", sys.modules)
        module = importlib.import_module("lisjong.policy_contract")
        self.assertNotIn("riichienv", sys.modules)
        for name in (
            "Seat",
            "Tile",
            "TileCategory",
            "TileType",
            "Wind",
            "MeldKind",
            "RiichiState",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(module, name))


if __name__ == "__main__":
    unittest.main()
