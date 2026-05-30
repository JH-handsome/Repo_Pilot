import unittest

from coding_rag.tokenizer import tokenize


class CodeTokenizerTest(unittest.TestCase):
    def test_split_snake_case_identifier(self):
        tokens = tokenize("build_binary_tree")

        self.assertIn("build_binary_tree", tokens)
        self.assertIn("build", tokens)
        self.assertIn("binary", tokens)
        self.assertIn("tree", tokens)

    def test_split_camel_case_identifier(self):
        tokens = tokenize("twoSum")

        self.assertIn("twosum", tokens)
        self.assertIn("two", tokens)
        self.assertIn("sum", tokens)

    def test_chinese_ngrams(self):
        tokens = tokenize("\u4e8c\u53c9\u6811\u8282\u70b9")

        self.assertIn("\u4e8c\u53c9\u6811", tokens)
        self.assertIn("\u8282\u70b9", tokens)

    def test_chinese_code_synonyms(self):
        tokens = tokenize("\u68c0\u7d22\u7ed3\u679c\u91cc\u6709\u6b63\u786e\u6587\u4ef6")

        self.assertIn("retrieval", tokens)
        self.assertIn("result", tokens)
        self.assertIn("relevant", tokens)
        self.assertIn("file", tokens)

    def test_chinese_test_synonyms(self):
        tokens = tokenize("\u6d4b\u8bd5\u548c\u65ad\u8a00")

        self.assertIn("test", tokens)
        self.assertIn("assert", tokens)


if __name__ == "__main__":
    unittest.main()
