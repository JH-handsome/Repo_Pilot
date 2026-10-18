from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest

from coding_rag.agent.executor import AgentExecutor, WriteApprovalRequired
from coding_rag.agent.safety import AgentSafetyPolicy, load_agent_safety_policy
from coding_rag.tools.agent_readonly import AgentToolError, ReadOnlyAgentTools, extract_task_identifiers


class AgentToolsTest(unittest.TestCase):
    def test_search_code_returns_ranked_rows(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            tools = ReadOnlyAgentTools(root)

            rows = tools.search_code("load_data read_text", top_k=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rank"], 1)
        self.assertIn("app.py", rows[0]["path"])
        self.assertIn("load_data", rows[0]["text"])

    def test_read_file_returns_line_numbered_range(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            tools = ReadOnlyAgentTools(root)

            result = tools.read_file("app.py", start_line=1, end_line=2)

        self.assertEqual(result["path"], "app.py")
        self.assertIn("1: def load_data", result["text"])
        self.assertIn("2:     return", result["text"])
        self.assertEqual(result["total_lines"], 5)

    def test_read_file_rejects_path_traversal(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            outside = Path(temp_dir).parent / "outside_agent_tool_test.py"
            outside.write_text("SECRET = True\n", encoding="utf-8")
            tools = ReadOnlyAgentTools(root)
            try:
                with self.assertRaises(AgentToolError):
                    tools.read_file(str(outside))
            finally:
                outside.unlink(missing_ok=True)

    def test_read_file_rejects_env_and_ignored_paths(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            (root / ".env").write_text("FAKE_TEST_VALUE\n", encoding="utf-8")
            (root / ".env.local").write_text("FAKE_TEST_VALUE\n", encoding="utf-8")
            git_dir = root / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text("FAKE_TEST_VALUE\n", encoding="utf-8")
            venv_dir = root / ".venv"
            venv_dir.mkdir()
            (venv_dir / "secret.py").write_text("FAKE_TEST_VALUE\n", encoding="utf-8")
            tools = ReadOnlyAgentTools(root)

            for path in (".env", ".env.local", ".git/config", ".venv/secret.py"):
                with self.subTest(path=path), self.assertRaises(AgentToolError):
                    tools.read_file(path)

            self.assertNotIn(".env", tools.list_files("*"))
            self.assertNotIn(".env.local", tools.list_files("*"))

    def test_credential_named_files_are_not_readable_or_searchable(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            paths = ("secrets.py", "credentials.py", "private.pem", "token.json", ".npmrc")
            for path in paths:
                (root / path).write_text("notrealcanary = 1\n", encoding="utf-8")
            tools = ReadOnlyAgentTools(root)

            for path in paths:
                with self.subTest(path=path), self.assertRaises(AgentToolError):
                    tools.read_file(path)
                self.assertNotIn(path, tools.list_files("*"))
            self.assertNotIn("notrealcanary", " ".join(
                row["text"] for row in tools.search_code("notrealcanary", top_k=5)
            ))

    def test_list_files_rejects_outside_file_symlink(self):
        with TemporaryDirectory() as temp_dir, TemporaryDirectory() as outside_dir:
            root = make_repo(temp_dir)
            outside = Path(outside_dir) / "outside_agent_tool_symlink.py"
            outside.write_text("FAKE_TEST_VALUE\n", encoding="utf-8")
            link = root / "linked.py"
            try:
                link.symlink_to(outside)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlink unavailable: {exc}")
            self.assertNotIn("linked.py", ReadOnlyAgentTools(root).list_files("**/*.py"))

    def test_read_file_rejects_symlink_alias_to_env(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            env_file = root / ".env"
            env_file.write_text("FAKE_TEST_VALUE\n", encoding="utf-8")
            try:
                (root / "settings.txt").symlink_to(env_file)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlink unavailable: {exc}")

            with self.assertRaises(AgentToolError):
                ReadOnlyAgentTools(root).read_file("settings.txt")

    def test_list_files_skips_ignored_dirs(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            ignored = root / ".venv" / "ignored.py"
            ignored.parent.mkdir()
            ignored.write_text("x = 1\n", encoding="utf-8")
            tools = ReadOnlyAgentTools(root)

            files = tools.list_files("**/*.py")

        self.assertIn("app.py", files)
        self.assertIn("pkg/service.py", files)
        self.assertNotIn(".venv/ignored.py", files)

    def test_inspect_symbol_returns_context(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            tools = ReadOnlyAgentTools(root)

            matches = tools.inspect_symbol("load_data", context_lines=1)

        self.assertEqual(matches[0]["name"], "load_data")
        self.assertEqual(matches[0]["kind"], "function")
        self.assertIn("def load_data", matches[0]["context"])

    def test_inspect_symbol_exact_qualified_lookup_is_bounded(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "class Alpha:\n    def run(self):\n        return 1\n\n"
                "class Beta:\n    def run(self):\n        return 2\n",
                encoding="utf-8",
            )
            tools = ReadOnlyAgentTools(root)
            matches = tools.inspect_symbol("Alpha.run", exact=True, limit=1)
            self.assertEqual([(row["parent"], row["name"]) for row in matches], [("Alpha", "run")])
            self.assertEqual(tools.inspect_symbol("Alpha.ru", exact=True), [])

    def test_call_rejects_non_readonly_tools(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            tools = ReadOnlyAgentTools(root)

            with self.assertRaises(AgentToolError):
                tools.call("run_checks", {"command": "python -m unittest"})
            with self.assertRaises(AgentToolError):
                tools.call("propose_patch", {"files": ["app.py"]})
            with self.assertRaises(AgentToolError):
                tools.call("unknown", {})

    def test_extract_task_identifiers_filters_common_words(self):
        identifiers = extract_task_identifiers("fix load_data and CacheManager")

        self.assertEqual(identifiers, ["load_data", "CacheManager"])

    def test_executor_exposes_read_and_search_tools(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root)

            file_result = executor.call("read_file", {"path": "app.py", "start_line": 1, "end_line": 1})
            search_result = executor.call("search_code", {"query": "load_data", "top_k": 1})

        self.assertIn("def load_data", file_result["text"])
        self.assertEqual(search_result[0]["rank"], 1)

    def test_executor_applies_patch_and_inspects_diff(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            init_git_repo(root)
            diff = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,5 +1,5 @@
 def load_data(path):
-    return path.read_text(encoding='utf-8')
+    return path.read_text(encoding='utf-8').strip()
 
 class CacheManager:
     pass
"""
            executor, _ = approved_executor(root, "apply_patch", {"diff": diff})

            result = executor.apply_patch(diff)
            inspected = executor.inspect_diff()

        self.assertTrue(result["applied"])
        self.assertEqual(result["affected_files"], ["app.py"])
        self.assertIn(".strip()", result["post_change_diff"]["text"])
        self.assertIn(".strip()", inspected["stdout"])
        self.assertEqual(inspected["returncode"], 0)

    def test_inspect_diff_does_not_return_env_changes(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            init_git_repo(root)
            env_file = root / ".env"
            env_file.write_text("FAKE_OLD_VALUE\n", encoding="utf-8")
            ignored_file = root / ".venv" / "secret.py"
            ignored_file.parent.mkdir()
            ignored_file.write_text("FAKE_OLD_IGNORED\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "-f", ".env", ".venv/secret.py"],
                cwd=root, check=True, capture_output=True, text=True,
            )
            env_file.write_text("FAKE_NEW_VALUE\n", encoding="utf-8")
            ignored_file.write_text("FAKE_NEW_IGNORED\n", encoding="utf-8")
            (root / "app.py").write_text("def changed():\n    return True\n", encoding="utf-8")

            inspected = AgentExecutor(root).inspect_diff()
            subprocess.run(
                ["git", "add", "app.py"], cwd=root, check=True, capture_output=True, text=True
            )
            secret_only = AgentExecutor(root).inspect_diff()

        self.assertEqual(inspected["returncode"], 0)
        self.assertIn("def changed", inspected["stdout"])
        self.assertNotIn(".env", inspected["stdout"])
        self.assertNotIn(".venv", inspected["stdout"])
        self.assertNotIn("FAKE_NEW_VALUE", inspected["stdout"])
        self.assertEqual(secret_only["stdout"], "")

    def test_inspect_diff_skips_deleted_tracked_symlink(self):
        with TemporaryDirectory() as temp_dir, TemporaryDirectory() as outside_dir:
            root = make_repo(temp_dir)
            init_git_repo(root)
            outside = Path(outside_dir) / "outside.py"
            outside.write_text("FAKE_TEST_VALUE\n", encoding="utf-8")
            link = root / "linked.py"
            try:
                link.symlink_to(outside)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlink unavailable: {exc}")
            subprocess.run(
                ["git", "add", "linked.py"], cwd=root, check=True, capture_output=True, text=True
            )
            link.unlink()

            inspected = AgentExecutor(root).inspect_diff()

        self.assertEqual(inspected["returncode"], 0)
        self.assertEqual(inspected["stdout"], "")

    def test_inspect_diff_preserves_regular_file_with_spaces(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            init_git_repo(root)
            target = root / "my module.py"
            target.write_text("before = True\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "my module.py"], cwd=root,
                check=True, capture_output=True, text=True,
            )
            target.write_text("after = True\n", encoding="utf-8")

            inspected = AgentExecutor(root).inspect_diff()

        self.assertEqual(inspected["returncode"], 0)
        self.assertIn("my module.py", inspected["stdout"])
        self.assertIn("+after = True", inspected["stdout"])

    def test_executor_runs_command_in_repo(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            arguments = {"cmd": "python -m compileall app.py", "affected_files": ["app.py"]}
            executor, _ = approved_executor(root, "run_command", arguments)

            result = executor.run_command(**arguments)

        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["executed"])
        self.assertFalse(result["stdout_truncated"])

    def test_executor_rejects_denied_command(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root)

            with self.assertRaises(AgentToolError):
                executor.run_command("python -c \"print('unsafe')\"")

    def test_executor_rejects_shell_composition(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root)

            with self.assertRaises(AgentToolError):
                executor.run_command("python -m compileall app.py && git status")

    def test_executor_rejects_unsafe_allowlisted_options(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root)
            for cmd in (
                "rg --pre python load_data app.py",
                "rg --pre=python load_data app.py",
                "git diff --ext-diff",
                "git show --textconv HEAD",
                "git diff --output=diff.txt",
            ):
                with self.subTest(cmd=cmd), self.assertRaisesRegex(
                    AgentToolError, "command option is not allowed"
                ):
                    executor.preview_write(
                        "run_command", {"cmd": cmd, "affected_files": ["app.py"]}
                    )

    def test_executor_keeps_normal_search_and_git_diff_available(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root)
            for cmd in ("rg load_data app.py", "git diff --no-ext-diff --no-textconv"):
                with self.subTest(cmd=cmd):
                    request = executor.preview_write(
                        "run_command", {"cmd": cmd, "affected_files": ["app.py"]}
                    )
                    self.assertEqual(request["tool"], "run_command")

    def test_executor_rejects_parent_path_in_command_before_approval(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root)

            with self.assertRaisesRegex(AgentToolError, "repository-relative"):
                executor.preview_write(
                    "run_command",
                    {
                        "cmd": "rg load_data ../outside.py",
                        "affected_files": ["app.py"],
                    },
                )

    def test_executor_rejects_absolute_path_in_command_before_approval(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            outside = root.parent / "outside.py"
            executor = AgentExecutor(root)

            with self.assertRaisesRegex(AgentToolError, "repository-relative"):
                executor.preview_write(
                    "run_command",
                    {
                        "cmd": f"rg load_data {outside.as_posix()}",
                        "affected_files": ["app.py"],
                    },
                )

    def test_executor_rejects_outside_path_in_command_option(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root)

            with self.assertRaisesRegex(AgentToolError, "repository-relative"):
                executor.preview_write(
                    "run_command",
                    {
                        "cmd": "python scripts/retrieval_eval.py --trace-out=../outside.json",
                        "affected_files": ["app.py"],
                    },
                )

    def test_executor_requires_exact_approval_before_command(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root)

            with self.assertRaises(WriteApprovalRequired) as caught:
                executor.run_command(
                    "python -m compileall app.py",
                    affected_files=["app.py"],
                )

        self.assertEqual(caught.exception.request["files"], ["app.py"])

    def test_command_approval_is_consumed_after_one_call(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            arguments = {"cmd": "rg load_data app.py", "affected_files": ["app.py"]}
            executor, _ = approved_executor(root, "run_command", arguments)

            first = executor.run_command(**arguments)
            with self.assertRaises(WriteApprovalRequired):
                executor.run_command(**arguments)

        self.assertTrue(first["executed"])

    def test_executor_safe_mode_rejects_command(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root, safe_mode=True)

            with self.assertRaises(AgentToolError):
                executor.run_command(
                    "python -m compileall app.py",
                    affected_files=["app.py"],
                )

    def test_executor_truncates_command_output(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            policy = AgentSafetyPolicy(max_output_chars=8)
            arguments = {"cmd": "rg load_data app.py", "affected_files": ["app.py"]}
            executor, _ = approved_executor(root, "run_command", arguments, safety_policy=policy)

            result = executor.run_command(**arguments)

        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["stdout_truncated"])
        self.assertGreater(result["stdout_chars"], 8)

    def test_executor_command_timeout_is_enforced(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            (root / "test_slow.py").write_text(
                "import time\n"
                "import unittest\n\n"
                "from pathlib import Path\n\n"
                "class SlowTest(unittest.TestCase):\n"
                "    def test_slow(self):\n"
                "        Path('partial.txt').write_text('partial', encoding='utf-8')\n"
                "        time.sleep(2)\n",
                encoding="utf-8",
            )
            arguments = {
                "cmd": "python -m unittest test_slow",
                "affected_files": ["partial.txt"],
            }
            executor, _ = approved_executor(root, "run_command", arguments, command_timeout=1)

            with self.assertRaises(AgentToolError):
                executor.run_command(**arguments)
            self.assertFalse((root / "partial.txt").exists())

    def test_executor_rejects_multi_file_delete_patch(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root)
            diff = """diff --git a/app.py b/app.py
deleted file mode 100644
--- a/app.py
+++ /dev/null
@@ -1 +0,0 @@
-x
diff --git a/pkg/service.py b/pkg/service.py
deleted file mode 100644
--- a/pkg/service.py
+++ /dev/null
@@ -1 +0,0 @@
-x
"""

            with self.assertRaises(AgentToolError):
                executor.apply_patch(diff)

    def test_executor_rejects_mismatched_patch_approval(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            init_git_repo(root)
            diff = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,5 +1,5 @@
 def load_data(path):
-    return path.read_text(encoding='utf-8')
+    return path.read_text(encoding='utf-8').strip()
 
 class CacheManager:
     pass
"""
            preview = AgentExecutor(root).preview_write("apply_patch", {"diff": diff})
            changed_diff = diff.replace(".strip()", ".upper()")
            executor = AgentExecutor(root, write_approval=preview["fingerprint"])

            with self.assertRaises(WriteApprovalRequired):
                executor.apply_patch(changed_diff)
            content = (root / "app.py").read_text(encoding="utf-8")

        self.assertNotIn(".upper()", content)

    def test_executor_rejects_env_patch(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root)
            diff = """diff --git a/.env b/.env
--- a/.env
+++ b/.env
@@ -1 +1 @@
-API_KEY=old
+API_KEY=new
"""

            with self.assertRaises(AgentToolError):
                executor.apply_patch(diff)

    def test_executor_rejects_patch_path_traversal(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            executor = AgentExecutor(root)
            diff = """diff --git a/../outside.py b/../outside.py
--- a/../outside.py
+++ b/../outside.py
@@ -1 +1 @@
-old
+new
"""

            with self.assertRaises(AgentToolError):
                executor.apply_patch(diff)

    def test_patch_without_identifiable_files_cannot_request_approval(self):
        with TemporaryDirectory() as temp_dir:
            root = make_repo(temp_dir)
            diff = """--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-old
+new
"""

            with self.assertRaisesRegex(AgentToolError, "identifiable files"):
                AgentExecutor(root).preview_write("apply_patch", {"diff": diff})

    def test_patch_approval_lists_quoted_file_path(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "file name.py").write_text("value = 1\n", encoding="utf-8")
            diff = """diff --git "a/file name.py" "b/file name.py"
--- "a/file name.py"
+++ "b/file name.py"
@@ -1 +1 @@
-value = 1
+value = 2
"""

            request = AgentExecutor(root).preview_write("apply_patch", {"diff": diff})

        self.assertEqual(request["files"], ["file name.py"])

    def test_policy_file_can_extend_allowlist_but_not_override_denylist(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_dir = root / ".repopilot"
            policy_dir.mkdir()
            (policy_dir / "policy.json").write_text(
                '{"allowlist": ["pytest", "python -c"], "denylist": ["python -c"], "max_output_chars": 12}',
                encoding="utf-8",
            )

            policy = load_agent_safety_policy(root)

        self.assertIn(("pytest",), policy.allowlist)
        self.assertIn(("python", "-c"), policy.allowlist)
        self.assertIn(("python", "-c"), policy.denylist)
        self.assertIn(("git", "reset"), policy.denylist)
        self.assertEqual(policy.max_output_chars, 12)


def make_repo(temp_dir: str) -> Path:
    root = Path(temp_dir)
    (root / "app.py").write_text(
        "def load_data(path):\n"
        "    return path.read_text(encoding='utf-8')\n"
        "\n"
        "class CacheManager:\n"
        "    pass\n",
        encoding="utf-8",
    )
    package = root / "pkg"
    package.mkdir()
    (package / "service.py").write_text(
        "from app import load_data\n\n"
        "def run(path):\n"
        "    return load_data(path)\n",
        encoding="utf-8",
    )
    return root


def init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "app.py", "pkg/service.py"], cwd=root, check=True, capture_output=True, text=True)


def approved_executor(
    root: Path,
    tool: str,
    arguments: dict,
    **executor_options,
) -> tuple[AgentExecutor, dict]:
    """Create an executor carrying one exact approval for a validated write."""
    executor_options.setdefault("snapshot_root", root / ".snapshot-cache")
    previewer = AgentExecutor(root, **executor_options)
    request = previewer.preview_write(tool, arguments)
    return (
        AgentExecutor(root, write_approval=request["fingerprint"], **executor_options),
        request,
    )


if __name__ == "__main__":
    unittest.main()
