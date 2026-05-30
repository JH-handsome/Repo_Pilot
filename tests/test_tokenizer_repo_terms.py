import unittest

from coding_rag.tokenizer import tokenize


class TokenizerRepoTermsTest(unittest.TestCase):
    def test_repo_index_query_terms_expand_to_code_tokens(self):
        tokens = tokenize("函数签名 import graph 调用关系 文件树")

        self.assertIn("function", tokens)
        self.assertIn("signature", tokens)
        self.assertIn("call", tokens)
        self.assertIn("filetree", tokens)

    def test_import_query_expands_to_import_token(self):
        tokens = tokenize("Path 在哪里导入")

        self.assertIn("import", tokens)
        self.assertIn("from", tokens)


if __name__ == "__main__":
    unittest.main()
