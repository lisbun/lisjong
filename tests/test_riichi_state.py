import unittest

from lisjong.policy_contract.riichi import RiichiState


class RiichiStateTest(unittest.TestCase):
    def test_has_three_states(self) -> None:
        self.assertEqual(
            {state.value for state in RiichiState},
            {"none", "declared", "accepted"},
        )


if __name__ == "__main__":
    unittest.main()
