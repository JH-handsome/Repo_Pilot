from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from coding_rag.tools.agent_readonly import AgentToolError, ReadOnlyAgentTools
from coding_rag.agent.runtime import UnifiedRunConfig, compact_tool_result, run_unified_query
from coding_rag.evaluation.call_path_runtime import CallPathScriptedPolicy
from coding_rag.repository.search_index import RepositoryIndexKey, RepositoryIndexRegistry


class CallPathToolTest(unittest.TestCase):
    def test_assigned_imported_instance_method_is_opt_in_candidate(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "provider.py").write_text(
                "class Scraper:\n    def scrape(self, url):\n        return url\n",
                encoding="utf-8",
            )
            (root / "entry.py").write_text(
                "from provider import Scraper as WebScraper\n\n"
                "class Commands:\n"
                "    def web(self, url):\n"
                "        if not self.scraper:\n"
                "            self.scraper = WebScraper()\n"
                "        return self.scraper.scrape(url)\n",
                encoding="utf-8",
            )
            tools = ReadOnlyAgentTools(root)
            ordinary = tools.trace_call_path("entry.py", "Commands.web")
            hinted = tools.trace_call_path(
                "entry.py", "Commands.web", include_assigned_instance_candidates=True,
            )

        self.assertNotIn("assigned_instance_candidates", ordinary)
        self.assertEqual(len(hinted["assigned_instance_candidates"]), 1)
        candidate = hinted["assigned_instance_candidates"][0]
        self.assertEqual(candidate["call"], "self.scraper.scrape")
        self.assertEqual(candidate["assignment_line"], 6)
        self.assertEqual(candidate["source_line"], 7)
        self.assertEqual(candidate["target_path"], "provider.py")
        self.assertEqual(candidate["target_symbol"], "Scraper.scrape")
        self.assertEqual(candidate["target_start_line"], 2)
        self.assertEqual(candidate["kind"], "assigned_instance_method_candidate")

    def test_assigned_instance_abstains_on_rebinding_or_shadowed_constructor(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "provider.py").write_text(
                "class Scraper:\n    def scrape(self):\n        return 1\n",
                encoding="utf-8",
            )
            (root / "entry.py").write_text(
                "from provider import Scraper\n\n"
                "class Commands:\n"
                "    def rebound(self):\n"
                "        self.scraper = Scraper()\n"
                "        self.scraper = object()\n"
                "        return self.scraper.scrape()\n\n"
                "    def shadowed(self, Scraper):\n"
                "        self.scraper = Scraper()\n"
                "        return self.scraper.scrape()\n\n"
                "    def nested(self):\n"
                "        def assign():\n"
                "            self.scraper = Scraper()\n"
                "        return self.scraper.scrape()\n",
                encoding="utf-8",
            )
            tools = ReadOnlyAgentTools(root)
            for symbol in ("Commands.rebound", "Commands.shadowed", "Commands.nested"):
                with self.subTest(symbol=symbol):
                    result = tools.trace_call_path(
                        "entry.py", symbol, include_assigned_instance_candidates=True,
                    )
                    self.assertEqual(result["assigned_instance_candidates"], [])

    def test_assigned_instance_candidates_are_removed_whole_for_context_budget(self):
        result = {
            "path": "entry.py", "symbol": "Commands.web", "edges": [],
            "assigned_instance_candidates": [
                {
                    "call": f"self.scraper.scrape_{index}",
                    "target_path": "provider.py",
                    "target_symbol": "Scraper.scrape",
                    "target_start_line": 2,
                }
                for index in range(20)
            ],
        }
        compact, observation = compact_tool_result("trace_call_path", result, max_chars=1000)
        self.assertTrue(observation["output_truncated"])
        self.assertTrue(compact["assigned_instance_candidates_truncated"])
        self.assertLess(len(compact["assigned_instance_candidates"]), 20)
        self.assertEqual(compact["edges"], [])

    def test_graph_trace_reuses_hybrid_search_index_without_second_repository_build(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "from target import helper\n\ndef run():\n    return helper()\n", encoding="utf-8"
            )
            (root / "target.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            registry = RepositoryIndexRegistry(disk_cache=False)
            config = UnifiedRunConfig(repo_path=root, trace_db_path=None, max_steps=4)
            with patch("coding_rag.tools.agent_readonly.load_python_files",
                       side_effect=AssertionError("duplicate source load")):
                with patch("coding_rag.tools.agent_readonly.build_repo_index",
                           side_effect=AssertionError("duplicate symbol index")):
                    run = run_unified_query(
                        "Trace run() through helper().", config,
                        CallPathScriptedPolicy("Trace run() through helper()."),
                        index_registry=registry,
                    )

        self.assertEqual(run.status, "success")
        self.assertEqual(
            [decision.tool for decision in run.decisions if decision.action == "tool"],
            ["search_code", "trace_call_path", "read_file", "read_file"],
        )
        self.assertEqual(registry.entry_count(), 1)

    def test_prepared_index_rejects_another_repository(self):
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            first, second = Path(first_dir), Path(second_dir)
            (first / "entry.py").write_text("def run():\n    pass\n", encoding="utf-8")
            (second / "entry.py").write_text("def run():\n    pass\n", encoding="utf-8")
            snapshot = RepositoryIndexRegistry(disk_cache=False).get(
                RepositoryIndexKey.create(second)
            ).index
            tools = ReadOnlyAgentTools(first, index_provider=lambda: snapshot)
            with self.assertRaisesRegex(AgentToolError, "prepared index does not match"):
                tools.trace_call_path("entry.py", "run")

    def test_local_call_and_package_reexport(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "pkg"
            package.mkdir()
            (package / "__init__.py").write_text("from .worker import execute\n", encoding="utf-8")
            (package / "worker.py").write_text("def execute():\n    return 1\n", encoding="utf-8")
            (package / "entry.py").write_text(
                "from . import execute\n\n"
                "def run():\n    return helper()\n\n"
                "def helper():\n    return execute()\n",
                encoding="utf-8",
            )
            result = ReadOnlyAgentTools(root).trace_call_path("pkg/entry.py", "run")

        self.assertEqual(result["path"], "pkg/entry.py")
        self.assertEqual(result["symbol"], "run")
        self.assertEqual(
            [(edge["kind"], edge["target_path"], edge["target_symbol"])
             for edge in result["edges"]],
            [("local_call", "pkg/entry.py", "helper"),
             ("import_bound_call", "pkg/worker.py", "execute")],
        )
        self.assertEqual(result["edges"][1]["target_start_line"], 1)

    def test_package_instance_follows_imported_class_method(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "pkg"
            package.mkdir()
            (package / "__init__.py").write_text(
                "from .utils import EngineHandler\nengines = EngineHandler()\n", encoding="utf-8"
            )
            (package / "utils.py").write_text(
                "class Other:\n    def all(self):\n        return []\n\n"
                "class EngineHandler:\n    def all(self):\n        return []\n",
                encoding="utf-8",
            )
            (package / "entry.py").write_text(
                "from . import engines\n\ndef get_template():\n    return engines.all()\n",
                encoding="utf-8",
            )
            result = ReadOnlyAgentTools(root).call(
                "trace_call_path", {"path": "pkg/entry.py", "symbol": "get_template"}
            )

        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(result["edges"][0]["target_path"], "pkg/utils.py")
        self.assertEqual(result["edges"][0]["target_symbol"], "all")
        self.assertEqual(result["edges"][0]["target_start_line"], 6)

    def test_dynamic_call_is_not_falsely_resolved(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "class Runner:\n    def run(self):\n        return self.callback()\n",
                encoding="utf-8",
            )
            tools = ReadOnlyAgentTools(root)
            self.assertEqual(tools.trace_call_path("entry.py", "Runner.run")["edges"], [])
            result = tools.trace_call_path(
                "entry.py", "Runner.run", include_same_class_candidates=True,
            )

        self.assertEqual(result["edges"], [])
        self.assertIn("dynamic dispatch", result["limitations"])

    def test_unresolved_calls_are_opt_in_syntax_not_target_claims(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "from target import helper\n\n"
                "class Runner:\n"
                "    def run(self):\n"
                "        helper()\n"
                "        self.coder.linter.lint('file.py')\n"
                "        def later():\n"
                "            self.not_run_here()\n",
                encoding="utf-8",
            )
            (root / "target.py").write_text("def helper():\n    pass\n", encoding="utf-8")
            tools = ReadOnlyAgentTools(root)
            ordinary = tools.trace_call_path("entry.py", "Runner.run")
            hinted = tools.trace_call_path("entry.py", "Runner.run", include_unresolved_calls=True)

        self.assertNotIn("unresolved_calls", ordinary)
        self.assertEqual(hinted["unresolved_calls"], [
            {"source_line": 6, "call": "self.coder.linter.lint"},
        ])
        self.assertEqual(hinted["edges"][0]["target_path"], "target.py")
        self.assertNotIn("target_path", hinted["unresolved_calls"][0])

    def test_unresolved_call_compaction_keeps_complete_sites(self):
        payload = {
            "path": "entry.py", "symbol": "run", "edges": [], "truncated": False,
            "unresolved_calls": [
                {"source_line": index + 1, "call": f"self.dynamic_{index}.method"}
                for index in range(40)
            ],
            "unresolved_calls_truncated": False,
        }
        compact, metadata = compact_tool_result("trace_call_path", payload, max_chars=1000)
        self.assertTrue(metadata["output_truncated"])
        self.assertNotIn("preview", compact)
        self.assertTrue(compact["unresolved_calls_truncated"])
        self.assertLess(len(compact["unresolved_calls"]), 40)

    def test_self_method_is_a_same_file_candidate_not_a_runtime_proof(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "class Runner:\n"
                "    def run(self):\n        return self.helper()\n"
                "    def helper(self):\n        return 1\n",
                encoding="utf-8",
            )
            tools = ReadOnlyAgentTools(root)
            self.assertEqual(tools.trace_call_path("entry.py", "Runner.run")["edges"], [])
            result = tools.trace_call_path(
                "entry.py", "Runner.run", include_same_class_candidates=True,
            )

        self.assertEqual(len(result["edges"]), 1)
        edge = result["edges"][0]
        self.assertEqual(edge["kind"], "same_class_method_candidate")
        self.assertEqual(edge["source_line"], 3)
        self.assertEqual(edge["target_path"], "entry.py")
        self.assertEqual(edge["target_symbol"], "Runner.helper")
        self.assertEqual(edge["target_start_line"], 4)
        policy = CallPathScriptedPolicy("Trace Runner.run() through Runner.helper().")
        self.assertIsNone(policy._target_edge(result["edges"]))

    def test_rebound_self_does_not_suggest_original_class_method(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "class Runner:\n"
                "    def run(self, replacement):\n"
                "        self = replacement\n"
                "        return self.helper()\n"
                "    def helper(self):\n        return 1\n",
                encoding="utf-8",
            )
            result = ReadOnlyAgentTools(root).trace_call_path(
                "entry.py", "Runner.run", include_same_class_candidates=True,
            )

        self.assertEqual(result["edges"], [])

    def test_nested_function_call_is_not_attributed_to_outer_function(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "def run():\n    def later():\n        return helper()\n    return later\n"
                "\ndef helper():\n    return 1\n",
                encoding="utf-8",
            )
            result = ReadOnlyAgentTools(root).trace_call_path("entry.py", "run")

        self.assertEqual(result["edges"], [])

    def test_nested_closure_can_be_traced_explicitly_without_falsely_tracing_factory(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "target.py").write_text("def finish():\n    return 1\n", encoding="utf-8")
            (root / "entry.py").write_text(
                "def factory():\n"
                "    def helper():\n"
                "        from target import finish\n"
                "        return finish()\n"
                "    def wrapper():\n"
                "        return helper()\n"
                "    return wrapper\n",
                encoding="utf-8",
            )
            tools = ReadOnlyAgentTools(root)
            factory = tools.trace_call_path("entry.py", "factory")
            wrapper = tools.trace_call_path("entry.py", "factory.wrapper")

        self.assertEqual(factory["edges"], [])
        self.assertEqual(
            [(edge["kind"], edge["target_path"], edge["target_symbol"])
             for edge in wrapper["edges"]],
            [("local_call", "entry.py", "factory.helper"),
             ("import_bound_call", "target.py", "finish")],
        )

    def test_conditional_duplicate_nested_function_is_not_silently_selected(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text(
                "def factory(flag):\n"
                "    if flag:\n"
                "        def run():\n"
                "            return 1\n"
                "    else:\n"
                "        def run():\n"
                "            return 2\n"
                "    return run\n",
                encoding="utf-8",
            )
            tools = ReadOnlyAgentTools(root)
            self.assertEqual(tools.trace_call_path("entry.py", "factory")["edges"], [])
            with self.assertRaisesRegex(AgentToolError, "multiple conditional definitions"):
                tools.trace_call_path("entry.py", "factory.run")

    def test_local_binding_does_not_resolve_to_shadowed_import_or_function(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "helper.py").write_text("def execute():\n    return 1\n", encoding="utf-8")
            (root / "entry.py").write_text(
                "from helper import execute\n\n"
                "def local_target():\n    return 1\n\n"
                "def by_parameter(execute):\n    return execute()\n\n"
                "def by_assignment():\n    execute = lambda: 1\n    return execute()\n\n"
                "def by_nested_definition():\n    def local_target():\n        return 2\n"
                "    return local_target()\n",
                encoding="utf-8",
            )
            tools = ReadOnlyAgentTools(root)
            for symbol in ("by_parameter", "by_assignment"):
                self.assertEqual(tools.trace_call_path("entry.py", symbol)["edges"], [])
            nested = tools.trace_call_path("entry.py", "by_nested_definition")
            self.assertEqual(
                [(edge["kind"], edge["target_path"], edge["target_symbol"])
                 for edge in nested["edges"]],
                [("local_call", "entry.py", "by_nested_definition.local_target")],
            )

    def test_path_policy_applies_before_tracing(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text("def run():\n    pass\n", encoding="utf-8")
            (root / ".env").write_text("not-a-secret-value\n", encoding="utf-8")
            tools = ReadOnlyAgentTools(root)
            with self.assertRaises(AgentToolError):
                tools.trace_call_path("../outside.py", "run")
            with self.assertRaises(AgentToolError):
                tools.trace_call_path(".env", "run")
            with self.assertRaises(AgentToolError):
                tools.trace_call_path("entry.py", "missing")

    def test_context_compaction_keeps_whole_edge_paths(self):
        payload = {
            "path": "entry.py", "symbol": "run", "truncated": False,
            "edges": [
                {"target_path": f"pkg/very_long_module_name_{index}.py", "target_symbol": "helper"}
                for index in range(20)
            ],
        }

        compact, metadata = compact_tool_result("trace_call_path", payload, max_chars=260)

        self.assertTrue(metadata["output_truncated"])
        self.assertTrue(compact["truncated"])
        self.assertNotIn("preview", compact)
        self.assertTrue(all(edge["target_path"].endswith(".py") for edge in compact["edges"]))
        self.assertEqual(len(payload["edges"]), 20)


if __name__ == "__main__":
    unittest.main()
