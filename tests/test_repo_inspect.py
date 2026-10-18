import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from coding_rag.repository.search_index import RepositoryIndexRegistry, build_repository_index
from scripts.repo_inspect import bound_discovery_groups, main, parse_args, serve


class RepoInspectTest(unittest.TestCase):
    def test_class_discovery_is_explicit_and_function_default_remains(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "backend.py").write_text(
                "class ModelBackend:\n"
                "    def authenticate(self, credentials):\n"
                "        return credentials\n"
                "    def get_group_permissions(self, user):\n"
                "        return user.permissions\n",
                encoding="utf-8",
            )
            query = "Find the backend that authenticates credentials and computes group permissions"
            ordinary = self._run(str(root), "discover", query)
            classes = self._run(str(root), "discover", query, "--kind", "class")
            local = self._run(str(root), "locate", "backend.py", query, "--kind", "class")

        self.assertNotIn("ModelBackend", [item["name"] for group in ordinary["output"]["groups"]
                                           for item in group["candidates"]])
        self.assertEqual(classes["output"]["groups"][0]["candidates"][0]["name"],
                         "ModelBackend")
        self.assertEqual(local["output"]["candidates"][0]["kind"], "class")

    def test_discovery_budget_preserves_file_coverage_before_extra_symbols(self):
        groups = [
            {"path": f"package/source_{index}.py", "search_rank": index + 1,
             "candidates": [{"name": f"Source{index}.method_{number}", "score": 1.0}
                            for number in range(3)]}
            for index in range(5)
        ]
        result = {"groups": groups, "truncated": False, "note": "Unverified candidates."}
        one_per_file = {**result, "groups": [
            {**group, "candidates": group["candidates"][:1]} for group in groups
        ]}
        budget = len(json.dumps(one_per_file, ensure_ascii=False)) + 10

        bound_discovery_groups(result, budget)

        self.assertTrue(result["truncated"])
        self.assertEqual([group["path"] for group in result["groups"]],
                         [group["path"] for group in groups])
        self.assertTrue(all(group["candidates"] for group in result["groups"]))
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), budget)

    def test_session_reuses_index_and_rechecks_changed_files(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "entry.py"
            source.write_text("def original():\n    return 1\n", encoding="utf-8")
            builds = []

            def build(key, manifest):
                builds.append(1)
                return build_repository_index(key, manifest)

            registry = RepositoryIndexRegistry(builder=build, disk_cache=False)

            class Requests:
                count = 0

                def readline(self, _limit):
                    self.count += 1
                    if self.count == 1:
                        return '["symbol", "original"]\n'
                    if self.count == 2:
                        return '["symbol", "original"]\n'
                    if self.count == 3:
                        source.write_text("def replacement():\n    return 2\n", encoding="utf-8")
                        return '["symbol", "replacement"]\n'
                    return ""

            responses = StringIO()
            with patch("scripts.repo_inspect.RepositoryIndexRegistry", return_value=registry):
                self.assertEqual(serve(parse_args([str(root), "serve"]), Requests(), responses), 0)
            rows = [json.loads(line) for line in responses.getvalue().splitlines()]
            self.assertEqual([row["ok"] for row in rows], [True, True, True])
            self.assertEqual(rows[0]["result"]["output"]["results"][0]["name"], "original")
            self.assertEqual(rows[2]["result"]["output"]["results"][0]["name"], "replacement")
            self.assertEqual(len(builds), 2)

    def test_session_rejects_invalid_requests_without_exposing_write_tools(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            requests = StringIO(
                'not-json\n'
                '["apply_patch", "entry.py"]\n'
                '["serve"]\n'
                '["read", "../outside.py"]\n'
                '["symbol", "run"]\n'
            )
            responses = StringIO()
            self.assertEqual(serve(parse_args([str(root), "serve"]), requests, responses), 0)
            rows = [json.loads(line) for line in responses.getvalue().splitlines()]
            self.assertEqual([row["ok"] for row in rows], [False, False, False, False, True])
            self.assertEqual(rows[-1]["result"]["output"]["results"][0]["name"], "run")

    def test_json_stdout_round_trips_unicode_on_gbk_windows_stdout(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text("# 中文源码说明\ndef run():\n    return 1\n", encoding="utf-8")
            env = {**os.environ, "PYTHONIOENCODING": "gbk:replace"}
            command = [sys.executable, str(Path(__file__).resolve().parents[1] /
                     "scripts/repo_inspect.py"), str(root), "read", "entry.py",
                     "--start-line", "1", "--end-line", "2"]
            result = subprocess.run(command, capture_output=True, env=env, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            decoded = json.loads(result.stdout.decode("ascii"))
            self.assertIn("中文源码说明", decoded["output"]["text"])

    def test_search_trace_and_read_use_existing_readonly_tools(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "from target import helper\n\ndef run():\n    return helper()\n", encoding="utf-8"
            )
            (root / "target.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            root_arg = str(root)
            search = self._run(root_arg, "search", "Trace run() through helper()")
            self.assertEqual(search["mode"], "offline_readonly")
            self.assertEqual(search["tool"], "search")
            self.assertEqual(search["retriever"], "hybrid-bm25-v7")
            self.assertIn(search["index"]["lifecycle"], {"full_rebuild", "disk_hit"})
            self.assertIn(search["index"]["disk_write_status"], {"written", "failed", "not_attempted"})
            self.assertIn("entry.py", [row["path"] for row in search["output"]["results"]])

            symbol = self._run(root_arg, "symbol", "run")
            self.assertEqual(symbol["output"]["results"][0]["path"], "entry.py")
            self.assertEqual(symbol["output"]["results"][0]["name"], "run")

            outline = self._run(root_arg, "outline", "entry.py")
            self.assertEqual(outline["output"]["path"], "entry.py")
            self.assertEqual(outline["output"]["symbols"][0]["name"], "run")
            self.assertEqual(outline["output"]["total_symbols"], 1)
            located = self._run(root_arg, "locate", "entry.py", "Where does run call helper?")
            self.assertEqual(located["output"]["path"], "entry.py")
            self.assertEqual(located["output"]["candidates"][0]["name"], "run")
            self.assertFalse(located["output"]["truncated"])
            discovered = self._run(root_arg, "discover", "run calls helper", "--top-k", "2")
            self.assertIn("entry.py", [group["path"] for group in discovered["output"]["groups"]])
            entry = next(group for group in discovered["output"]["groups"]
                         if group["path"] == "entry.py")
            self.assertEqual(entry["candidates"][0]["name"], "run")
            bounded = self._run(
                root_arg, "--max-output-chars", "1000", "discover",
                "run calls helper", "--top-k", "2", "--per-file", "20",
            )
            self.assertIsInstance(bounded["output"]["groups"], list)
            self.assertLessEqual(bounded["observation"]["returned_chars"], 1000)

            (root / "nested.py").write_text(
                "def factory():\n    def decorator():\n        def inner():\n            return 1\n        return inner\n    return decorator\n",
                encoding="utf-8",
            )
            nested = self._run(root_arg, "outline", "nested.py")
            self.assertEqual(
                [row["name"] for row in nested["output"]["symbols"]],
                ["factory", "factory.decorator", "factory.decorator.inner"],
            )
            nested_locate = self._run(root_arg, "locate", "nested.py", "inner")
            self.assertEqual(nested_locate["output"]["candidates"][0]["name"],
                             "factory.decorator.inner")
            short = self._run(root_arg, "locate", "nested.py", "decorator inner", "--limit", "1")
            self.assertTrue(short["output"]["truncated"])
            no_match = self._run(root_arg, "locate", "nested.py", "galactic zirconium")
            self.assertEqual(no_match["output"]["candidates"], [])

            trace = self._run(root_arg, "trace", "entry.py", "run")
            self.assertEqual(trace["output"]["edges"][0]["target_path"], "target.py")
            verified = self._run(root_arg, "verify-edge", "entry.py", "run", "target.py", "helper")
            self.assertEqual(verified["output"]["status"], "supported_static_call")
            self.assertEqual([(item["path"], item["line"]) for item in verified["output"]["evidence"]],
                             [("entry.py", 4), ("target.py", 1)])
            self.assertEqual(verified["output"]["direct_edge_count"], 1)
            budgeted = self._run(root_arg, "--max-output-chars", "1000", "verify-edge",
                                 "entry.py", "run", "target.py", "helper")
            self.assertEqual(budgeted["output"]["status"], "supported_static_call")
            self.assertLessEqual(budgeted["observation"]["returned_chars"], 1000)
            absent = self._run(root_arg, "verify-edge", "entry.py", "run", "entry.py", "run")
            self.assertEqual(absent["output"]["status"], "not_established")
            self.assertEqual(absent["output"]["evidence"], [])
            related = self._run(root_arg, "related", "run calls the helper")
            self.assertEqual(related["output"]["candidates"][0]["target_path"], "target.py")
            self.assertEqual(related["output"]["candidates"][0]["source_line"], 4)
            self.assertEqual(related["output"]["candidates"][0]["target_start_line"], 1)
            anchored = self._run(
                root_arg, "related", "run calls the helper", "--source-path", "entry.py",
                "--source-symbol", "run",
            )
            self.assertEqual(anchored["output"]["search_files"], [])
            self.assertEqual(anchored["output"]["source_anchor"], {"path": "entry.py", "symbol": "run"})
            self.assertEqual(anchored["output"]["candidates"][0]["target_path"], "target.py")
            automatic = self._run(root_arg, "related", "run calls the helper", "--auto-anchor")
            self.assertEqual(automatic["output"]["source_anchor"], None)
            self.assertEqual(automatic["output"]["auto_source_candidate"]["source_symbol"], "run")
            self.assertEqual(automatic["output"]["candidates"][0]["target_path"], "target.py")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main([root_arg, "related", "run", "--source-path", "entry.py"]), 2)
                self.assertEqual(main([root_arg, "related", "run", "--auto-anchor",
                                       "--source-path", "entry.py", "--source-symbol", "run"]), 2)
                self.assertEqual(main([root_arg, "related", "run", "--auto-anchor",
                                       "--source-files", "2"]), 2)
            read = self._run(root_arg, "read", "target.py", "--start-line", "1", "--end-line", "2")
            self.assertIn("1: def helper():", read["output"]["text"])

    def test_rejects_oversized_read_and_repository_escape(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            (root / ".env").write_text("secret-value\n", encoding="utf-8")
            for args in (
                ("read", "entry.py", "--start-line", "1", "--end-line", "81"),
                ("read", "../outside.py"),
                ("read", ".env"),
                ("search", "run", "--top-k", "21"),
                ("symbol", "run", "--limit", "21"),
                ("outline", "entry.py", "--limit", "201"),
                ("outline", "../outside.py"),
                ("outline", ".env"),
                ("locate", "../outside.py", "run"),
                ("locate", ".env", "secret"),
                ("locate", "entry.py", "run", "--limit", "21"),
                ("discover", "run", "--top-k", "11"),
                ("discover", "run", "--per-file", "21"),
                ("related", "run", "--limit", "21"),
                ("verify-edge", "entry.py", "run", "../outside.py", "helper"),
                ("verify-edge", "entry.py", "run", ".env", "helper"),
            ):
                with self.subTest(args=args):
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        self.assertEqual(main([str(root), *args]), 2)

    def test_optional_external_cache_reuses_snapshot_across_commands(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            cache = Path(temp_dir) / "cache"
            (root / "entry.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            self._run(str(root), "--cache-dir", str(cache), "search", "run")
            snapshots = list(cache.glob("*.json.gz"))
            self.assertEqual(len(snapshots), 1)
            before = snapshots[0].stat().st_mtime_ns
            self._run(str(root), "--cache-dir", str(cache), "search", "run")
            self.assertEqual(snapshots[0].stat().st_mtime_ns, before)
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main([str(root), "--cache-dir", str(root / "cache"), "search", "run"]), 2)

    def test_same_class_trace_is_explicit(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "class Runner:\n"
                "    def run(self):\n        return self.helper()\n"
                "    def helper(self):\n        return 1\n",
                encoding="utf-8",
            )
            ordinary = self._run(str(root), "trace", "entry.py", "Runner.run")
            expanded = self._run(
                str(root), "trace", "entry.py", "Runner.run", "--same-class",
            )
            missing = self._run(str(root), "discover", "Locate Runner.run_missing")
            self.assertTrue(missing["output"]["abstained"])
            self.assertEqual(missing["output"]["missing_exact_definitions"],
                             ["Runner.run_missing"])
            self.assertEqual(missing["output"]["groups"], [])
            self.assertEqual(ordinary["output"]["edges"], [])
            self.assertEqual(
                expanded["output"]["edges"][0]["kind"], "same_class_method_candidate",
            )
            candidate = self._run(
                str(root), "verify-edge", "entry.py", "Runner.run", "entry.py", "Runner.helper",
            )
            self.assertEqual(candidate["output"]["status"], "not_established")

    def test_assigned_instance_trace_is_explicit_and_not_a_verified_edge(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "provider.py").write_text(
                "class Scraper:\n    def scrape(self):\n        return 1\n", encoding="utf-8",
            )
            (root / "entry.py").write_text(
                "from provider import Scraper\n\n"
                "class Commands:\n"
                "    def web(self):\n"
                "        self.scraper = Scraper()\n"
                "        return self.scraper.scrape()\n",
                encoding="utf-8",
            )
            ordinary = self._run(str(root), "trace", "entry.py", "Commands.web")
            hinted = self._run(
                str(root), "trace", "entry.py", "Commands.web", "--assigned-instance",
            )
        self.assertNotIn("assigned_instance_candidates", ordinary["output"])
        self.assertEqual(hinted["output"]["edges"][0]["target_symbol"], "Scraper")
        self.assertEqual(
            hinted["output"]["assigned_instance_candidates"][0]["target_symbol"],
            "Scraper.scrape",
        )
        self.assertEqual(
            hinted["output"]["assigned_instance_candidates"][0]["kind"],
            "assigned_instance_method_candidate",
        )

    def test_unresolved_trace_is_explicit_and_does_not_infer_target(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "def run(client):\n    return client.linter.lint('file.py')\n",
                encoding="utf-8",
            )
            ordinary = self._run(str(root), "trace", "entry.py", "run")
            hinted = self._run(str(root), "trace", "entry.py", "run", "--unresolved")
        self.assertNotIn("unresolved_calls", ordinary["output"])
        self.assertEqual(hinted["output"]["unresolved_calls"], [
            {"source_line": 2, "call": "client.linter.lint"},
        ])
        self.assertEqual(hinted["output"]["edges"], [])

    def test_unresolved_call_filter_only_limits_visible_call_sites(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "def run(client):\n"
                "    client.io.warn('x')\n"
                "    return client.linter.lint('file.py')\n",
                encoding="utf-8",
            )
            filtered = self._run(
                str(root), "trace", "entry.py", "run", "--unresolved", "--call-contains", "LINT",
            )
            with redirect_stderr(StringIO()):
                self.assertEqual(main([
                    str(root), "trace", "entry.py", "run", "--call-contains", "lint",
                ]), 2)
        self.assertEqual(filtered["output"]["unresolved_calls"], [
            {"source_line": 3, "call": "client.linter.lint"},
        ])
        self.assertEqual(filtered["output"]["unresolved_call_filter"], "LINT")
        self.assertEqual(filtered["output"]["edges"], [])

    def test_verify_edge_requires_direct_caller_not_transitive_local_helper(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "from target import helper\n\ndef run():\n    return local()\n\ndef local():\n    return helper()\n",
                encoding="utf-8",
            )
            (root / "target.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            result = self._run(str(root), "verify-edge", "entry.py", "run", "target.py", "helper")
            self.assertEqual(result["output"]["status"], "not_established")
            self.assertEqual(result["output"]["direct_edge_count"], 1)

    def test_auto_related_omits_edges_from_other_functions_in_same_file(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "from target import helper\n\ndef local_behavior():\n    return 1\n"
                "\ndef unrelated():\n    return helper()\n", encoding="utf-8",
            )
            (root / "target.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
            broad = self._run(str(root), "related", "locate local_behavior")
            focused = self._run(str(root), "related", "locate local_behavior", "--auto-anchor")
            self.assertGreater(len(broad["output"]["candidates"]), 0)
            self.assertEqual(focused["output"]["auto_source_candidate"]["source_symbol"],
                             "local_behavior")
            self.assertEqual(focused["output"]["candidates"], [])

    @staticmethod
    def _run(*args):
        """Capture and parse one successful local JSON command."""
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(StringIO()):
            status = main(list(args))
        if status != 0:
            raise AssertionError(f"read-only command failed: {args}")
        return json.loads(output.getvalue())


if __name__ == "__main__":
    unittest.main()
