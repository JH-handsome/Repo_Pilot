import unittest

from stats_utils import mean


class MeanTest(unittest.TestCase):
    def test_preserves_fractional_result(self):
        self.assertEqual(mean([1, 2]), 1.5)


if __name__ == "__main__":
    unittest.main()
