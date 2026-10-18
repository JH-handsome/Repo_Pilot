from pathlib import Path
import unittest
from unittest.mock import patch

from rank_bm25 import BM25Okapi

from coding_rag.tools.bm25 import (
    BM25Retriever,
    PATH_MATCH_STOPWORDS,
    bm25_from_term_frequencies,
    identifier_variants,
    path_identifier_score,
    term_frequencies,
)
from coding_rag.tools.tokenizer import CodeTokenizer
from coding_rag.repository.chunks import CodeChunk


class BM25RetrieverHybridSearchTest(unittest.TestCase):
    def test_explicit_location_finds_definition_instead_of_call_sites(self):
        chunks = [
            CodeChunk(Path("src/impl.py"), 1, 3, "def trace_call_path():\n    return 1\n"),
            CodeChunk(Path("src/usage.py"), 1, 3,
                      "trace_call_path()\ntrace_call_path()\ntrace_call_path()\n"),
        ]
        retriever = BM25Retriever(chunks)
        with patch.object(retriever, "hybrid_scores", return_value=[0.1, 0.9]):
            located, scores = retriever.search_with_scores("trace_call_path 的入口在哪里", top_k=1)
        self.assertEqual(located[0].chunk.file_path, Path("src/impl.py"))
        self.assertGreater(scores[0], scores[1])

        with patch.object(retriever, "hybrid_scores", return_value=[0.1, 0.9]):
            call_sites, scores = retriever.search_with_scores("哪里调用 trace_call_path", top_k=1)
        self.assertEqual(call_sites[0].chunk.file_path, Path("src/usage.py"))
        self.assertEqual(scores, [0.1, 0.9])

    def test_qualified_location_promotes_only_named_method(self):
        chunks = [
            CodeChunk(Path("src/a.py"), 1, 3, "class Alpha:\n    def run_step(self):\n        pass\n"),
            CodeChunk(Path("src/b.py"), 1, 3, "class Beta:\n    def run_step(self):\n        pass\n"),
        ]
        retriever = BM25Retriever(chunks)
        with patch.object(retriever, "hybrid_scores", return_value=[0.1, 0.9]):
            results, scores = retriever.search_with_scores("Alpha.run_step 定义在哪", top_k=2)
        self.assertEqual(results[0].chunk.file_path, Path("src/a.py"))
        self.assertEqual(scores[1], 0.9)

    def test_exact_qualified_method_selects_right_chunk_without_changing_file_order(self):
        path = Path("auth/views.py")
        chunks = [
            CodeChunk(path, 1, 4, "class LoginView:\n    def form_valid(self, form):\n        return form\n"),
            CodeChunk(path, 5, 8, "class OtherView:\n    def form_valid(self, form):\n        return form\n"),
            CodeChunk(Path("auth/forms.py"), 1, 2, "def clean():\n    return True\n"),
        ]
        retriever = BM25Retriever(chunks)
        with patch.object(retriever, "hybrid_scores", return_value=[0.2, 0.9, 0.6]):
            selected, scores = retriever.search_with_scores("解释 LoginView.form_valid()", top_k=2)
        self.assertEqual([item.chunk.file_path for item in selected], [path, Path("auth/forms.py")])
        self.assertEqual(selected[0].chunk.start_line, 1)
        self.assertEqual(selected[0].score, 0.9)
        self.assertEqual(scores, [0.9, 0.9, 0.6])

        with patch.object(retriever, "hybrid_scores", return_value=[0.2, 0.9, 0.6]):
            unqualified, original_scores = retriever.search_with_scores("解释 form_valid()", top_k=2)
        self.assertEqual(unqualified[0].chunk.start_line, 5)
        self.assertEqual(original_scores, [0.2, 0.9, 0.6])

        with patch.object(retriever, "hybrid_scores", return_value=[0.2, 0.9, 0.6]):
            all_chunks, _ = retriever.search_with_scores("解释 LoginView.form_valid()", top_k=3)
        self.assertEqual([item.chunk.start_line for item in all_chunks], [1, 1, 5])

    def test_search_reuses_prepared_filename_terms(self):
        class CountingTokenizer(CodeTokenizer):
            def __init__(self):
                super().__init__()
                object.__setattr__(self, "tokenize_calls", 0)

            def tokenize(self, value):
                object.__setattr__(self, "tokenize_calls", self.tokenize_calls + 1)
                return super().tokenize(value)

        chunks = [
            CodeChunk(Path("src/retrieval.py"), 1, 2, "def search():\n    return 1"),
            CodeChunk(Path("src/retrieval.py"), 3, 4, "def rank():\n    return 2"),
            CodeChunk(Path("src/runtime.py"), 1, 2, "def run():\n    return 3"),
        ]
        tokenizer = CountingTokenizer()
        retriever = BM25Retriever(chunks, tokenizer=tokenizer)
        prepared_call_count = tokenizer.tokenize_calls

        results = retriever.search("where is retrieval search", top_k=2)

        self.assertEqual(tokenizer.tokenize_calls - prepared_call_count, 1)
        self.assertEqual(results[0].chunk.file_path, Path("src/retrieval.py"))
        self.assertEqual(len(retriever.stem_term_sets), len(chunks))

    def test_prepared_path_score_matches_unprepared_score(self):
        tokenizer = CodeTokenizer()
        path = Path("coding_rag/tools/chunks.py")
        path_terms = identifier_variants(
            set(tokenizer.tokenize("coding_rag tools chunks py"))
        ) - PATH_MATCH_STOPWORDS
        stem_terms = frozenset(
            identifier_variants(set(tokenizer.tokenize(path.stem))) - PATH_MATCH_STOPWORDS
        )
        for query in ("chunk", "chunks", "where is caller chunks", "unrelated"):
            query_tokens = set(tokenizer.tokenize(query))
            expected = path_identifier_score(query_tokens, path, tokenizer)
            actual = path_identifier_score(
                query_tokens,
                path,
                tokenizer,
                path_terms,
                prepared_stem_terms=stem_terms,
                prepared_query_terms=identifier_variants(query_tokens) - PATH_MATCH_STOPWORDS,
            )
            self.assertEqual(actual, expected)

    def test_exact_filename_stem_gets_full_path_score(self):
        tokenizer = CodeTokenizer()
        query_tokens = set(tokenizer.tokenize("BM25 检索模块做什么"))

        score = path_identifier_score(
            query_tokens, Path("coding_rag/tools/bm25.py"), tokenizer, {"coding", "rag", "bm25"}
        )

        self.assertEqual(score, 1.0)

    def test_sparse_bm25_state_matches_rank_bm25_scores(self):
        corpus = [
            ["alpha", "alpha", "beta"],
            ["beta", "gamma", "gamma", "gamma"],
            ["delta", "alpha"],
        ]
        expected = BM25Okapi(corpus)
        actual = bm25_from_term_frequencies(
            tuple(term_frequencies(document) for document in corpus)
        )

        for query in (["alpha"], ["beta", "gamma"], ["missing"], ["alpha", "alpha"]):
            self.assertEqual(list(expected.get_scores(query)), list(actual.get_scores(query)))

    def test_search_uses_hybrid_metadata_for_semantic_file_location(self):
        query = "\u5982\u679c Recall@3 \u4e00\u76f4\u662f 0\uff0c\u4f46\u8089\u773c\u770b\u68c0\u7d22\u7ed3\u679c\u91cc\u6709\u6b63\u786e\u6587\u4ef6"
        chunks = [
            CodeChunk(Path("rag/prompt.py"), 1, 10, "Recall context and answer prompt"),
            CodeChunk(Path("scripts/retrieval_eval.py"), 1, 10, "matched_relevant_index recall_at_k"),
        ]

        results = BM25Retriever(chunks).search(query, top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("scripts/retrieval_eval.py"))
        self.assertEqual(results[0].source, "hybrid")

    def test_search_uses_symbol_metadata_without_path_patch(self):
        chunks = [
            CodeChunk(Path("a.py"), 1, 10, "def complete(messages):\n    return send(messages)"),
            CodeChunk(Path("b.py"), 1, 10, "def render_prompt(context):\n    return context"),
        ]

        results = BM25Retriever(chunks).search("where is complete function", top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("a.py"))

    def test_search_deprioritizes_tests_without_test_intent(self):
        chunks = [
            CodeChunk(Path("coding_rag/loader.py"), 1, 10, "def load_data():\n    return []"),
            CodeChunk(Path("tests/test_loader.py"), 1, 10, "def test_load_data():\n    assert load_data() == []"),
        ]

        results = BM25Retriever(chunks).search("load_data 是如何实现的", top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("coding_rag/loader.py"))

    def test_search_allows_tests_when_query_asks_for_tests(self):
        chunks = [
            CodeChunk(Path("coding_rag/loader.py"), 1, 10, "def load_data():\n    return []"),
            CodeChunk(Path("tests/test_loader.py"), 1, 10, "def test_load_data():\n    assert load_data() == []"),
        ]

        results = BM25Retriever(chunks).search("load_data 的测试在哪里", top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("tests/test_loader.py"))

    def test_search_prefers_file_diversity_before_duplicate_chunks(self):
        chunks = [
            CodeChunk(Path("a.py"), 1, 10, "target target target"),
            CodeChunk(Path("a.py"), 11, 20, "target target"),
            CodeChunk(Path("b.py"), 1, 10, "target"),
        ]

        results = BM25Retriever(chunks).search("target", top_k=2)

        self.assertEqual({result.chunk.file_path for result in results}, {Path("a.py"), Path("b.py")})


if __name__ == "__main__":
    unittest.main()
