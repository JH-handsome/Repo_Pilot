import unittest

from coding_rag.tools.tokenizer import CodeTokenizer, _cached_identifier_tokens, tokenize


class CodeTokenizerTest(unittest.TestCase):
    def test_cached_identifier_tokens_keep_independent_list_semantics(self):
        tokenizer = CodeTokenizer()
        hits_before = _cached_identifier_tokens.cache_info().hits
        first = tokenizer.tokenize_identifier("__BaseHandler.resolve_request__")
        self.assertEqual(first, ["basehandler.resolve_request", "base", "handler.resolve", "request"])
        first.append("mutated")
        self.assertEqual(
            tokenizer.tokenize_identifier("__BaseHandler.resolve_request__"),
            ["basehandler.resolve_request", "base", "handler.resolve", "request"],
        )
        self.assertEqual(
            CodeTokenizer(keep_full_identifier=False).tokenize_identifier("get_resolver"),
            ["get", "resolver"],
        )
        self.assertGreater(_cached_identifier_tokens.cache_info().hits, hits_before)
        self.assertLessEqual(_cached_identifier_tokens.cache_info().currsize, 32768)

    def test_long_identifier_bypasses_cache(self):
        tokenizer = CodeTokenizer()
        identifier = "A" * 129
        before = _cached_identifier_tokens.cache_info()
        self.assertEqual(tokenizer.tokenize_identifier(identifier), [identifier.lower()])
        self.assertEqual(_cached_identifier_tokens.cache_info(), before)

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

    def test_query_only_developer_concepts_do_not_change_document_terms(self):
        tokenizer = CodeTokenizer()
        query = "审批指纹、代码行号与目录"
        document_terms = tokenizer.tokenize(query)
        query_terms = tokenizer.tokenize_query(query)

        self.assertNotIn("fingerprint", document_terms)
        self.assertNotIn("dirnames", document_terms)
        for term in ("approval", "fingerprint", "start_line", "dirnames"):
            self.assertIn(term, query_terms)
            self.assertEqual(query_terms.count(term), 1)


if __name__ == "__main__":
    unittest.main()
