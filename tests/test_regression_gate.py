import copy
import unittest

from coding_rag.evaluation.gate import compare_scorecards


class RegressionGateTest(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "schema_version": 2,
            "experiment": {"datasets": {
                "retrieval": {"sha256": "retrieval", "case_count": 20},
                "agent": {"sha256": "agent", "case_count": 50},
            }, "config": {"top_k": 5}, "retriever": {"name": "hybrid-v2"}},
            "overall_score": 80.0,
            "retrieval": {"hit_at_1": 0.8, "mrr_at_5": 0.9, "recall_at_5": 0.9},
            "agent": {
                "fixture_diagnostic": {"dataset_coverage": 0.9},
                "offline_contract": {"trace_complete_rate": 1.0, "safe_tool_interface_rate": 1.0},
            },
            "efficiency": {"query_latency_ms_p95": 10.0},
        }
        self.policy = {
            "max_drop": {"overall_score": 2.0, "retrieval.mrr_at_5": 0.02},
            "max_ratio": {"efficiency.query_latency_ms_p95": 1.5},
        }

    def test_threshold_boundary_passes(self):
        current = copy.deepcopy(self.baseline)
        current["overall_score"] = 78.0
        current["retrieval"]["mrr_at_5"] = 0.88
        current["efficiency"]["query_latency_ms_p95"] = 15.0
        self.assertTrue(compare_scorecards(current, self.baseline, self.policy).passed)

    def test_regression_reports_specific_metric(self):
        current = copy.deepcopy(self.baseline)
        current["retrieval"]["mrr_at_5"] = 0.87
        result = compare_scorecards(current, self.baseline, self.policy)
        self.assertFalse(result.passed)
        self.assertIn("retrieval.mrr_at_5", result.failures[0])

    def test_dataset_mismatch_is_incompatible(self):
        current = copy.deepcopy(self.baseline)
        current["experiment"]["datasets"]["agent"]["sha256"] = "changed"
        result = compare_scorecards(current, self.baseline, self.policy)
        self.assertFalse(result.compatible)

    def test_schema_mismatch_is_incompatible(self):
        current = copy.deepcopy(self.baseline)
        current["schema_version"] = 3
        result = compare_scorecards(current, self.baseline, self.policy)
        self.assertFalse(result.compatible)

    def test_retrieval_config_mismatch_is_incompatible(self):
        current = copy.deepcopy(self.baseline)
        current["experiment"]["config"]["top_k"] = 10
        result = compare_scorecards(current, self.baseline, self.policy)
        self.assertFalse(result.compatible)
        self.assertIn("experiment.config", result.failures[0])


if __name__ == "__main__":
    unittest.main()
