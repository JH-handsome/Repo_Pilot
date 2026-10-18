"""Exercise language-independent retrieval through the actual repository pipelines."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coding_rag.agent.executor import AgentExecutor, WriteApprovalRequired
from coding_rag.agent.runtime import UnifiedRunConfig, build_hybrid_search_provider
from coding_rag.agent.workflow import build_recommended_checks
from coding_rag.evaluation.retrieval import PreparedRetrievalPipeline, RetrievalEvalConfig
from coding_rag.learning.workflow import LearningWorkflowConfig, collect_repository_evidence
from coding_rag.rag.prompt import format_results_as_context_with_ranges
from coding_rag.repository.chunks import split_source_files
from coding_rag.repository.files import MAX_SOURCE_BYTES, load_python_files, load_source_files
from coding_rag.repository.index import build_repo_index
from coding_rag.repository.search_index import RepositoryIndexKey, RepositoryIndexRegistry
from coding_rag.tools.agent_readonly import AgentToolError, ReadOnlyAgentTools
from coding_rag.tools.bm25 import SearchResult


class MultilanguageRepositoryTest(unittest.TestCase):
    def test_sources_of_different_languages_are_retrievable_without_python(self):
        fixtures = {
            "app.ts": "export function lunarBeacon() { return 1; }\n",
            "view.tsx": "export const OrbitPanel = () => <div />;\n",
            "ui.vue": "<script>const cometButton = 1;</script>\n",
            "main.go": "package main\nfunc solarRelay() int { return 2 }\n",
            "lib.rs": "pub fn nebulaCache() -> i32 { 3 }\n",
            "Main.java": "class Main { int stellarQueue() { return 4; } }\n",
            "main.cpp": "int asteroidCounter() { return 5; }\n",
            "app.cs": "class App { int Meteor() => 6; }\n",
            "query.sql": "SELECT galaxy FROM stars;\n",
            "deploy.ps1": "Write-Output 'cosmicDeployment'\n",
            "worker.customlang": "define quasarWorker = 7\n",
            "Dockerfile": "FROM scratch\nLABEL pulsarImage=yes\n",
            "package.json": '{"name":"planetary-tools"}\n',
        }
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, text in fixtures.items():
                (root / name).write_text(text, encoding="utf-8")
            files = load_source_files(root)
            self.assertEqual({file.path.name for file in files}, set(fixtures))
            self.assertEqual(load_python_files(root), [])
            tools = ReadOnlyAgentTools(root)
            for name, query in (("app.ts", "lunarBeacon"), ("main.go", "solarRelay"),
                                ("lib.rs", "nebulaCache"), ("worker.customlang", "quasarWorker")):
                with self.subTest(name=name):
                    self.assertEqual(tools.search_code(query, top_k=1)[0]["path"], name)
            self.assertIn("app.ts", tools.call("list_files"))
            provider = build_hybrid_search_provider(UnifiedRunConfig(repo_path=root, recall_window=0))
            self.assertEqual(provider("lunarBeacon", 1)["results"][0]["path"], "app.ts")
            pipeline = PreparedRetrievalPipeline.build(RetrievalEvalConfig(repo_path=root))
            self.assertEqual(pipeline.query("nebulaCache")["seed"][0].chunk.file_path.name, "lib.rs")

    def test_non_python_changes_refresh_incremental_and_disk_indexes(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            cache = Path(temp_dir) / "cache"
            path = root / "app.js"
            path.write_text("export const lunarBeacon = 1;\n", encoding="utf-8")
            key = RepositoryIndexKey.create(root)
            registry = RepositoryIndexRegistry(cache_dir=cache)
            registry.get(key)
            fresh = RepositoryIndexRegistry(cache_dir=cache).get(key)
            self.assertEqual(fresh.index.chunks[0].text, "export const lunarBeacon = 1;")
            path.write_text("export const orbitalReplacement = 200;\n", encoding="utf-8")
            changed = registry.get(key)
            self.assertEqual(changed.status, "stale")
            self.assertIn("orbitalReplacement", changed.index.chunks[0].text)
            (root / "main.go").write_text("package main\n", encoding="utf-8")
            self.assertEqual(registry.get(key).index.file_count, 2)
            path.unlink()
            removed = registry.get(key)
            self.assertEqual(removed.index.file_count, 1)
            self.assertEqual(removed.index.chunks[0].file_path.name, "main.go")

    def test_text_policy_excludes_binary_generated_private_and_oversized_files(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.js").write_text("const safe = 1;\n", encoding="utf-8")
            (root / "binary.custom").write_bytes(b"hello\x00world")
            (root / "invalid.custom").write_bytes(b"\xff\xfe\xff")
            (root / "large.js").write_bytes(b"x" * (MAX_SOURCE_BYTES + 1))
            for name in ("secrets.ts", "credentials.js", ".env.local", "README.md"):
                (root / name).write_text("excluded", encoding="utf-8")
            for name in ("node_modules", "vendor", "dist", "target", ".engramory-memory"):
                (root / name).mkdir()
                (root / name / "private.js").write_text("excluded", encoding="utf-8")
            self.assertEqual([file.path.name for file in load_source_files(root)], ["app.js"])
            with self.assertRaises(AgentToolError):
                ReadOnlyAgentTools(root).read_file("binary.custom")

    def test_bom_encodings_preserve_source_lines_for_read_and_index(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.ps1").write_text("Write-Output '你好'\n# done\n", encoding="utf-16")
            (root / "app.ts").write_text("const greeting = '你好';\n", encoding="utf-8-sig")
            files = load_source_files(root)
            self.assertEqual(len(files), 2)
            self.assertTrue(all("你好" in file.text for file in files))
            result = ReadOnlyAgentTools(root).read_file("app.ps1", 1, 1)
            self.assertEqual(result["text"], "1: Write-Output '你好'")

    def test_non_python_text_is_never_interpreted_as_python_ast(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("app.py", "app.custom"):
                (root / name).write_text("def target():\n    return 1\n", encoding="utf-8")
            chunks = split_source_files(load_source_files(root))
            index = build_repo_index(chunks)
            for chunk in chunks:
                symbols = index.for_chunk(chunk).symbols
                self.assertEqual(bool(symbols), chunk.file_path.suffix == ".py")

    def test_context_fences_follow_language_and_keep_visible_line_ranges(self):
        from coding_rag.repository.chunks import CodeChunk
        for extension, language in (("ts", "typescript"), ("rs", "rust"), ("custom", "text")):
            with self.subTest(extension=extension):
                result = SearchResult(CodeChunk(Path(f"app.{extension}"), 1, 1, "const x = 1;"), 1.0)
                context, ranges = format_results_as_context_with_ranges([result])
                self.assertIn(f"```{language}\n", context)
                self.assertEqual(ranges, [(f"app.{extension}", 1, 1)])

    def test_learning_collects_real_evidence_from_non_python_repository(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.ts").write_text("export function projectEntry() { return 1; }\n", encoding="utf-8")
            evidence, catalog, _ = collect_repository_evidence(
                "projectEntry", LearningWorkflowConfig(repo_path=root), AgentExecutor(root),
            )
            self.assertEqual(evidence[0]["path"], "main.ts")
            self.assertIn("main.ts", catalog)
            self.assertTrue(all("python" not in check for check in build_recommended_checks("feature", ["main.ts"])))

    def test_other_language_commands_still_require_explicit_approval(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.go").write_text("package main\n", encoding="utf-8")
            executor = AgentExecutor(root)
            for command in ("npm test", "cargo check", "go test ./...", "dotnet build", "mvn test"):
                with self.subTest(command=command), self.assertRaises(WriteApprovalRequired):
                    executor.run_command(command, affected_files=["main.go"])
