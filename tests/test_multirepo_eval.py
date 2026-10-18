import copy
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_rag.evaluation.multirepo import (
    MultiRepoValidationError,
    compare_multirepo_scorecards,
    evaluate_suite,
    gate_exit_code,
    load_and_validate_suite,
    normalize_remote,
)


class MultiRepoEvalTest(unittest.TestCase):
    def test_multistep_gate_rejects_losing_a_provider_file(self):
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads((root / "benchmarks/aider-django-multistep-baseline-v1.json").read_text(encoding="utf-8"))
        policy = json.loads((root / "benchmarks/aider-django-multistep-gate-v1.json").read_text(encoding="utf-8"))
        regressed = copy.deepcopy(baseline)
        regressed["metrics"]["all_gold_at_5"] -= 1 / baseline["metrics"]["case_count"]
        result = compare_multirepo_scorecards(regressed, baseline, policy)
        self.assertTrue(result.compatible)
        self.assertEqual(gate_exit_code(result), 1)
        self.assertTrue(any("all_gold_at_5" in failure for failure in result.failures))

    def _repo(self, root: Path, name: str, filename: str, content: str) -> tuple[Path, str, str]:
        repo = root / name
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
        remote = f"https://github.com/example/{name}.git"
        subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", remote], check=True)
        (repo / filename).write_text(content, encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", filename], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        return repo, commit, remote

    def _suite_files(self, root: Path) -> tuple[Path, Path, list[dict]]:
        _, first_commit, first_remote = self._repo(
            root, "first", "alpha.py", "def unique_alpha():\n    return 'alpha'\n"
        )
        _, second_commit, second_remote = self._repo(
            root, "second", "beta.py", "class UniqueBeta:\n    pass\n"
        )
        repositories = [
            {"id": "first", "path": "first", "remote": first_remote, "commit": first_commit,
             "license": "MIT", "license_url": "https://example.com/first-license", "purpose": "test first"},
            {"id": "second", "path": "second", "remote": second_remote, "commit": second_commit,
             "license": "MIT", "license_url": "https://example.com/second-license", "purpose": "test second"},
        ]
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "repositories": repositories}), encoding="utf-8")
        evalset = root / "cases.jsonl"
        evalset.write_text("\n".join((
            json.dumps({"id": "alpha", "repo_id": "first", "query": "unique_alpha", "gold_files": ["alpha.py"], "task_type": "symbol", "difficulty": "easy"}),
            json.dumps({"id": "beta", "repo_id": "second", "query": "UniqueBeta", "gold_files": ["beta.py"], "task_type": "symbol", "difficulty": "easy"}),
        )), encoding="utf-8")
        return manifest, evalset, repositories

    def test_validates_two_pinned_repositories_and_aggregates(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, evalset, _ = self._suite_files(root)
            suite = load_and_validate_suite(root, manifest, evalset)
            scorecard = evaluate_suite(suite, {
                "top_k": 5, "chunk_size": 40, "overlap": 5, "recall_window": 0,
                "max_recall_results": None, "min_final_score": 0.0,
                "graph_weight": 0.0, "path_weight": 0.25, "exclude_roots": (),
            })
        self.assertEqual(scorecard["metrics"]["case_count"], 2)
        self.assertEqual(scorecard["metrics"]["hit_at_1"], 1.0)
        self.assertEqual(scorecard["metrics"]["all_gold_at_5"], 1.0)
        self.assertEqual(scorecard["metrics"]["incomplete_case_ids"], [])
        self.assertEqual(set(scorecard["by_repo"]), {"first", "second"})
        self.assertEqual(scorecard["by_task_type"]["symbol"]["case_count"], 2)
        self.assertNotIn(str(root), json.dumps(scorecard))

    def test_rejects_commit_mismatch_and_unsafe_gold(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, evalset, repositories = self._suite_files(root)
            repositories[0]["commit"] = "0" * 40
            manifest.write_text(json.dumps({"schema_version": 1, "repositories": repositories}), encoding="utf-8")
            evalset.write_text(json.dumps({
                "id": "unsafe", "repo_id": "first", "query": "x", "gold_files": ["../secret.py"],
                "task_type": "path", "difficulty": "hard",
            }), encoding="utf-8")
            with self.assertRaises(MultiRepoValidationError) as raised:
                load_and_validate_suite(root, manifest, evalset)
        message = str(raised.exception)
        self.assertIn("HEAD mismatch", message)
        self.assertIn("unsafe gold path", message)

    def test_remote_normalization_and_gate_exit_semantics(self):
        self.assertEqual(
            normalize_remote("git@github.com:Example/Repo.git"),
            normalize_remote("https://github.com/example/repo.git"),
        )
        baseline = {
            "schema_version": 1, "benchmark": "real_multirepo_source_retrieval",
            "dataset": {"sha256": "data", "case_count": 2},
            "manifest": {"sha256": "manifest", "schema_version": 1},
            "repositories": [{"id": "first", "commit": "abc"}],
            "config": {"top_k": 5}, "retriever": "retriever", "quality_score": 80.0,
        }
        policy = {"max_drop": {"quality_score": 2.0}, "max_ratio": {}}
        self.assertEqual(gate_exit_code(compare_multirepo_scorecards(baseline, baseline, policy)), 0)
        regressed = copy.deepcopy(baseline)
        regressed["quality_score"] = 77.0
        self.assertEqual(gate_exit_code(compare_multirepo_scorecards(regressed, baseline, policy)), 1)
        incompatible = copy.deepcopy(baseline)
        incompatible["repositories"][0]["commit"] = "changed"
        result = compare_multirepo_scorecards(incompatible, baseline, policy)
        self.assertFalse(result.compatible)
        self.assertEqual(gate_exit_code(result), 2)


if __name__ == "__main__":
    unittest.main()
