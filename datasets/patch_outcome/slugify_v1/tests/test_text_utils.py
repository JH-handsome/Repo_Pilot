import unittest

from text_utils import slugify


class SlugifyTest(unittest.TestCase):
    def test_collapses_mixed_whitespace(self):
        self.assertEqual(slugify("  Hello\t  World  "), "hello-world")


if __name__ == "__main__":
    unittest.main()
