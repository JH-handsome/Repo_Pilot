import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_rag.evaluation.retrieval import EvalCase, PreparedRetrievalPipeline, RetrievalEvalConfig, evaluate_retrieval
from coding_rag.evaluation.scorecard import build_capability_scorecard


class CapabilityScorecardTest(unittest.TestCase):
    def test_pipeline_builds_once_and_reports_quality_and_efficiency(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "target.py").write_text("def unique_target():\n    return 1\n", encoding="utf-8")
            (root / "other.py").write_text("def unrelated():\n    return 2\n", encoding="utf-8")
            pipeline = PreparedRetrievalPipeline.build(RetrievalEvalConfig(root, top_k=2, recall_window=0))
            result = evaluate_retrieval(pipeline, [
                EvalCase("one", "unique_target", ["target.py"]),
                EvalCase("two", "unrelated", ["other.py"]),
            ])
        self.assertEqual(result["metrics"]["case_count"], 2)
        self.assertEqual(result["metrics"]["hit_at_1"], 1.0)
        self.assertEqual(result["metrics"]["mrr_at_2"], 1.0)
        self.assertEqual(result["metrics"]["recall_at_2"], 1.0)
        self.assertEqual(result["metrics"]["all_gold_at_2"], 1.0)
        self.assertEqual(result["metrics"]["incomplete_case_count"], 0)
        self.assertEqual(result["metrics"]["ndcg_at_2"], 1.0)
        self.assertEqual(result["efficiency"]["file_count"], 2)
        self.assertGreaterEqual(result["efficiency"]["index_build_ms"], 0)
        self.assertGreaterEqual(result["efficiency"]["query_latency_ms_p95"], 0)

    def test_partial_cross_file_hit_is_not_a_complete_call_path(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text("def entry():\n    return helper()\n", encoding="utf-8")
            (root / "helper.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            pipeline = PreparedRetrievalPipeline.build(RetrievalEvalConfig(root, top_k=1, recall_window=0))
            result = evaluate_retrieval(pipeline, [
                EvalCase("call-path", "entry", ["entry.py", "helper.py"], task_type="call_path"),
            ])
        self.assertEqual(result["metrics"]["hit_at_1"], 1.0)
        self.assertEqual(result["metrics"]["all_gold_at_1"], 0.0)
        self.assertEqual(result["metrics"]["bad_case_count"], 0)
        self.assertEqual(result["metrics"]["incomplete_case_ids"], ["call-path"])

    def test_default_excludes_corpus_roots_but_can_be_overridden(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.py").write_text("x = 1\n", encoding="utf-8")
            (root / "data").mkdir()
            (root / "data" / "large.py").write_text("x = 2\n", encoding="utf-8")
            default = PreparedRetrievalPipeline.build(RetrievalEvalConfig(root))
            included = PreparedRetrievalPipeline.build(RetrievalEvalConfig(root, exclude_roots=()))
        self.assertEqual(len(default.files), 1)
        self.assertEqual(len(included.files), 2)

    def test_schema_two_records_both_datasets_and_ignores_fixture_and_template_scores(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "target.py").write_text("def target():\n    return 1\n", encoding="utf-8")
            retrieval_set = root / "retrieval.json"
            agent_set = root / "agent.json"
            retrieval_set.write_text("[]", encoding="utf-8")
            agent_set.write_text("[]", encoding="utf-8")
            config = RetrievalEvalConfig(root, top_k=1, recall_window=0)
            pipeline = PreparedRetrievalPipeline.build(config)
            retrieval = evaluate_retrieval(pipeline, [EvalCase("one", "target", ["target.py"])])
            base_agent = {
                "total": 1, "fixture_diagnostic": {
                    "candidate_k": 5, "candidate_hit_at_5": 0.0,
                    "candidate_recall_at_5": 0.0, "candidate_mrr_at_5": 0.0,
                    "dataset_coverage": 1.0, "discriminative": False,
                }, "trace_complete_count": 1,
                "safe_tool_interface_count": 1,
            }
            low_template = {**base_agent, "avg_expected_tool_coverage": 0.0, "avg_section_coverage": 0.0}
            high_template = {**base_agent, "avg_expected_tool_coverage": 1.0, "avg_section_coverage": 1.0}
            first = build_capability_scorecard(config, retrieval_set, agent_set, pipeline, retrieval, low_template)
            high_template["fixture_diagnostic"] = {
                **base_agent["fixture_diagnostic"],
                "candidate_hit_at_5": 1.0,
                "candidate_recall_at_5": 1.0,
                "candidate_mrr_at_5": 1.0,
            }
            second = build_capability_scorecard(config, retrieval_set, agent_set, pipeline, retrieval, high_template)
        self.assertEqual(first["schema_version"], 2)
        self.assertEqual(first["experiment"]["datasets"]["agent"]["name"], "agent.json")
        self.assertEqual(first["experiment"]["datasets"]["agent"]["case_count"], 1)
        self.assertEqual(first["overall_score"], second["overall_score"])
        self.assertEqual(first["agent"]["evaluation_mode"], "offline_contract")
        self.assertIn("expected_tool_coverage", first["agent"]["offline_contract"])
        self.assertFalse(first["agent"]["fixture_diagnostic"]["discriminative"])


if __name__ == "__main__":
    unittest.main()
