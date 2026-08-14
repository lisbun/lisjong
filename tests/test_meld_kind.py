import unittest

from lisjong.policy_contract.meld import MeldKind


class MeldKindTest(unittest.TestCase):
    def test_has_five_kinds(self) -> None:
        self.assertEqual(
            {kind.value for kind in MeldKind},
            {"chi", "pon", "daiminkan", "ankan", "kakan"},
        )


if __name__ == "__main__":
    unittest.main()
