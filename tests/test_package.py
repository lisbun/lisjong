import unittest

import lisjong


class PackageImportTest(unittest.TestCase):
    def test_package_can_be_imported(self) -> None:
        self.assertEqual(lisjong.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
