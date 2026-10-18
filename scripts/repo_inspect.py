"""Run RepoPilot's production read-only repository tools without an LLM or page."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from io import StringIO
import json
import sys
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.agent.executor import AgentExecutor
from coding_rag.agent.runtime import (
    UnifiedRunConfig,
    build_hybrid_search_provider,
    build_repository_index_provider,
    compact_tool_result,
)
from coding_rag.repository.call_edge_candidates import (
    candidate_call_edges, query_only_call_edges,
)
from coding_rag.repository.call_path import PythonCallPathIndex
from coding_rag.repository.index import normalize_path
from coding_rag.repository.search_index import RETRIEVER_VERSION, RepositoryIndexRegistry
from coding_rag.repository.symbol_locator import (
    exact_definition_lookup_should_abstain,
    locate_file_symbols,
    locate_retrieved_files,
    missing_qualified_definitions,
)
from coding_rag.tools.agent_readonly import AgentToolError


SESSION_MAX_REQUEST_CHARS = 65536
SESSION_TOOLS = frozenset({
    "search", "related", "symbol", "outline", "locate", "discover",
    "trace", "verify-edge", "read",
})


def verify_edge(readonly, source_path: str, source_symbol: str,
                target_path: str, target_symbol: str) -> dict:
    """Check one direct static call, returning bounded evidence or an explicit unknown."""
    target_file = readonly.resolve_repo_path(target_path)
    if (not target_file.is_file() or target_file.suffix.casefold() != ".py"
            or normalize_path(target_file) not in readonly.repo_index.files):
        raise ValueError("target must be an indexed Python file")
    if not target_symbol.strip():
        raise ValueError("target symbol is required")
    target_key = readonly.display_path(target_file)
    trace_result = readonly.trace_call_path(source_path, source_symbol)
    direct_edges = [edge for edge in trace_result["edges"]
                    if edge["source_symbol"] == source_symbol]
    matched = [edge for edge in direct_edges
               if edge["target_path"] == target_key and edge["target_symbol"] == target_symbol]
    verified = next((edge for edge in matched if edge["target_start_line"] is not None), None)
    evidence = []
    if verified is not None:
        for path, line in ((trace_result["path"], verified["source_line"]),
                           (target_key, verified["target_start_line"])):
            snippet = readonly.read_file(path, max(1, line - 2), line + 2)
            evidence.append({"path": path, "line": line, "text": snippet["text"]})
    return {
        "source": {"path": trace_result["path"], "symbol": source_symbol},
        "target": {"path": target_key, "symbol": target_symbol},
        "status": "supported_static_call" if verified else "not_established",
        "edge": verified,
        "evidence": evidence,
        "direct_edge_count": len(direct_edges),
        "trace_truncated": trace_result["truncated"],
        "note": (
            "Static direct call and target definition are both located; read the snippets before asserting runtime behavior."
            if verified else
            "No matching direct call with an indexed target definition was established. "
            "This is not proof that runtime behavior is absent; imports, dispatch, or an incomplete trace may be unresolved."
        ),
    }


def bound_discovery_groups(result: dict, max_chars: int) -> None:
    """Preserve file coverage before extra symbols when the output budget is tight."""
    encoded_size = lambda: len(json.dumps(result, ensure_ascii=False))
    while result["groups"] and encoded_size() > max_chars:
        # The first candidate in each file is more useful than a third candidate
        # in one file when a bounded answer must cover several source files.
        group = next(
            (item for item in reversed(result["groups"])
             if len(item["candidates"]) > 1),
            None,
        )
        if group is not None:
            group["candidates"].pop()
        else:
            result["groups"].pop()
        result["truncated"] = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Accept one repository and one bounded read-only tool request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="repository root")
    parser.add_argument("--cache-dir", type=Path, help="optional persistent index cache outside the repository")
    parser.add_argument("--max-output-chars", type=int, default=12000)
    commands = parser.add_subparsers(dest="tool", required=True)
    search = commands.add_parser("search", help="run the same hybrid search as the unified Agent")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    related = commands.add_parser("related", help="suggest source-grounded cross-file call edges")
    related.add_argument("query")
    related.add_argument("--top-k", type=int, default=5)
    related.add_argument("--source-files", type=int, default=3)
    related.add_argument("--limit", type=int, default=3)
    related.add_argument("--source-path", help="exact repository-relative source file for anchored calls")
    related.add_argument("--source-symbol", help="exact function or Class.method in --source-path")
    related.add_argument("--auto-anchor", action="store_true", help="use one unverified query-only source candidate to reduce unrelated edges")
    symbol = commands.add_parser("symbol", help="locate exact function or Class.method definitions")
    symbol.add_argument("name")
    symbol.add_argument("--limit", type=int, default=10)
    outline = commands.add_parser("outline", help="list indexed symbols in one Python file")
    outline.add_argument("path", help="repository-relative Python file")
    outline.add_argument("--limit", type=int, default=50)
    locate = commands.add_parser("locate", help="rank real function definitions inside one selected file")
    locate.add_argument("path", help="repository-relative Python file found by search")
    locate.add_argument("query", help="natural-language question to rank local functions")
    locate.add_argument("--limit", type=int, default=5)
    locate.add_argument("--kind", choices=("function", "class", "all"), default="function")
    discover = commands.add_parser("discover", help="shortlist functions across retrieved Python files")
    discover.add_argument("query")
    discover.add_argument("--top-k", type=int, default=5)
    discover.add_argument("--per-file", type=int, default=3)
    discover.add_argument("--kind", choices=("function", "class", "all"), default="function")
    trace = commands.add_parser("trace", help="trace static calls from an exact Python symbol")
    trace.add_argument("path", help="repository-relative Python file")
    trace.add_argument("symbol", help="exact function or Class.method")
    trace.add_argument("--same-class", action="store_true", help="include syntactic self.method candidates")
    trace.add_argument("--assigned-instance", action="store_true", help="hint at imported class methods assigned to self in this function")
    trace.add_argument("--unresolved", action="store_true", help="list unresolved dotted call sites without inferring targets")
    trace.add_argument("--call-contains", help="show only unresolved call sites containing this literal text")
    verify = commands.add_parser("verify-edge", help="check one exact direct static call with source evidence")
    verify.add_argument("source_path", help="repository-relative caller file")
    verify.add_argument("source_symbol", help="exact caller function or Class.method")
    verify.add_argument("target_path", help="repository-relative callee file")
    verify.add_argument("target_symbol", help="exact callee function or Class.method")
    read = commands.add_parser("read", help="read at most 80 numbered source lines")
    read.add_argument("path", help="repository-relative file")
    read.add_argument("--start-line", type=int, default=1)
    read.add_argument("--end-line", type=int)
    commands.add_parser("serve", help="serve newline-delimited read-only CLI requests on stdin")
    return parser.parse_args(argv)


def inspect(args: argparse.Namespace, *, index_registry: RepositoryIndexRegistry | None = None) -> dict:
    """Dispatch bounded read-only repository tools without a model."""
    root = args.repo.resolve()
    if not root.is_dir():
        raise ValueError("repository path must be an existing directory")
    if not 1000 <= args.max_output_chars <= 100000:
        raise ValueError("max-output-chars must be between 1000 and 100000")
    if args.tool == "search":
        if not args.query.strip() or not 1 <= args.top_k <= 20:
            raise ValueError("search requires a query and top-k between 1 and 20")
        arguments = {"query": args.query, "top_k": args.top_k}
    elif args.tool == "related":
        if (not args.query.strip() or not 1 <= args.top_k <= 20
                or not 1 <= args.source_files <= 10 or not 1 <= args.limit <= 20):
            raise ValueError("related requires a query, top-k 1..20, source-files 1..10, limit 1..20")
        if bool(args.source_path) != bool(args.source_symbol):
            raise ValueError("related requires --source-path and --source-symbol together")
        if args.auto_anchor and args.source_path:
            raise ValueError("related --auto-anchor cannot be combined with an explicit source")
        if args.auto_anchor and args.source_files != 3:
            raise ValueError("related --auto-anchor selects one source; omit --source-files")
        arguments = {"query": args.query, "top_k": args.top_k}
    elif args.tool == "symbol":
        if not args.name.strip() or not 1 <= args.limit <= 20:
            raise ValueError("symbol requires a name and limit between 1 and 20")
        arguments = {"name": args.name, "limit": args.limit}
    elif args.tool == "outline":
        arguments = {"path": args.path, "limit": args.limit}
    elif args.tool == "locate":
        if not args.query.strip() or not 1 <= args.limit <= 20:
            raise ValueError("locate requires a query and limit between 1 and 20")
        arguments = {"path": args.path, "query": args.query, "limit": args.limit}
    elif args.tool == "discover":
        if not args.query.strip() or not 1 <= args.top_k <= 10 or not 1 <= args.per_file <= 20:
            raise ValueError("discover requires a query, top-k 1..10 and per-file 1..20")
        arguments = {"query": args.query, "top_k": args.top_k}
    elif args.tool == "trace":
        arguments = {"path": args.path, "symbol": args.symbol}
    elif args.tool == "verify-edge":
        arguments = {"path": args.source_path, "symbol": args.source_symbol}
    elif args.tool == "read":
        if args.start_line < 1:
            raise ValueError("start-line must be positive")
        end_line = args.end_line if args.end_line is not None else args.start_line + 79
        if end_line < args.start_line or end_line - args.start_line >= 80:
            raise ValueError("read range must contain 1 to 80 lines")
        arguments = {"path": args.path, "start_line": args.start_line, "end_line": end_line}
    else:
        raise ValueError("unsupported read-only tool")

    config = UnifiedRunConfig(repo_path=root)
    registry = index_registry if index_registry is not None else (
        RepositoryIndexRegistry(cache_dir=args.cache_dir) if args.cache_dir else None
    )
    executor = AgentExecutor(
        root,
        safe_mode=True,
        search_provider=build_hybrid_search_provider(config, index_registry=registry),
        read_index_provider=build_repository_index_provider(config, index_registry=registry),
    )
    if args.tool == "symbol":
        matches = executor.readonly.inspect_symbol(
            arguments["name"], exact=True, limit=arguments["limit"],
        )
        result = {"results": [
            {**{key: value for key, value in match.items() if key != "context"},
             "text": match["context"]}
            for match in matches
        ]}
        compact_tool = "search_code"
    elif args.tool == "related":
        readonly = executor.readonly
        if args.source_path:
            source_path = readonly.resolve_repo_path(args.source_path)
            source_paths = [source_path]
            search_files = []
            anchor_path = readonly.display_path(source_path)
        else:
            search_result = executor.call("search_code", arguments)
            rows = search_result.get("results") or []
            source_paths = [readonly.resolve_repo_path(row["path"]) for row in rows]
            search_files = [row["path"] for row in rows]
            anchor_path = None
        call_index = PythonCallPathIndex(readonly.chunks, readonly.repo_index)
        auto_source_candidate = None
        auto_abstention_reason = None
        if args.auto_anchor:
            auto_source_candidate, candidates, auto_abstention_reason = query_only_call_edges(
                readonly.retriever, call_index, root, args.query, source_paths,
            )
        else:
            candidates = candidate_call_edges(
                readonly.retriever, call_index, root, args.query,
                source_paths,
                source_files=1 if args.source_path else args.source_files,
                source_symbol=args.source_symbol,
            )
        result = {
            "search_files": search_files,
            "search_files_truncated": False,
            "source_anchor": (
                {"path": anchor_path, "symbol": args.source_symbol}
                if args.source_path else None
            ),
            "auto_source_candidate": auto_source_candidate,
            "auto_abstention_reason": auto_abstention_reason,
            "candidates": candidates[:args.limit],
            "candidates_truncated": len(candidates) > args.limit,
            "note": (
                "No traceable non-test source function was selected; this does not rule out a cross-file call."
                if args.auto_anchor and auto_abstention_reason else
                "Edges from one unverified query-only source candidate; read both locations before use. "
                "An empty list does not rule out dynamic or missed calls."
                if args.auto_anchor else
                "No resolvable cross-file call from the selected source; inspect local calls with trace."
                if args.source_path and not candidates else
                "Ranked static call candidates, not a verified answer; read both source locations."
            ),
        }
        while result["candidates"] and len(json.dumps(result, ensure_ascii=False)) > args.max_output_chars:
            result["candidates"].pop()
            result["candidates_truncated"] = True
        while result["search_files"] and len(json.dumps(result, ensure_ascii=False)) > args.max_output_chars:
            result["search_files"].pop()
            result["search_files_truncated"] = True
        compact_tool = "search_code"
    elif args.tool == "trace" and (args.same_class or args.assigned_instance or args.unresolved or args.call_contains is not None):
        if args.call_contains is not None and not args.unresolved:
            raise ValueError("--call-contains requires --unresolved")
        if args.call_contains is not None and not 1 <= len(args.call_contains.strip()) <= 100:
            raise ValueError("--call-contains must have 1 to 100 non-space characters")
        result = executor.readonly.trace_call_path(
            arguments["path"], arguments["symbol"],
            include_same_class_candidates=args.same_class,
            include_unresolved_calls=args.unresolved,
            include_assigned_instance_candidates=args.assigned_instance,
        )
        if args.call_contains is not None:
            needle = args.call_contains.strip().casefold()
            result["unresolved_calls"] = [
                item for item in result["unresolved_calls"]
                if needle in item["call"].casefold()
            ]
            result["unresolved_call_filter"] = args.call_contains.strip()
            result["unresolved_note"] += " Literal filtering does not infer a target or prove absence."
        compact_tool = "trace_call_path"
    elif args.tool == "verify-edge":
        result = verify_edge(executor.readonly, args.source_path, args.source_symbol,
                             args.target_path, args.target_symbol)
        if len(json.dumps(result, ensure_ascii=False)) > args.max_output_chars:
            for item in result["evidence"]:
                item["text"] = next((line[:160] for line in item["text"].splitlines()
                                     if line.startswith(f'{item["line"]}:')), "")
            result["evidence_truncated"] = True
        if len(json.dumps(result, ensure_ascii=False)) > args.max_output_chars:
            for item in result["evidence"]:
                item["text"] = ""
            result["evidence_truncated"] = True
        compact_tool = "search_code"
    elif args.tool == "outline":
        result = executor.readonly.outline_file(arguments["path"], limit=arguments["limit"])
        while result["symbols"] and len(json.dumps(result, ensure_ascii=False)) > args.max_output_chars:
            result["symbols"].pop()
            result["truncated"] = True
        compact_tool = "search_code"
    elif args.tool == "locate":
        readonly = executor.readonly
        path = readonly.resolve_repo_path(arguments["path"])
        candidates = locate_file_symbols(readonly.retriever, path, arguments["query"],
                                         limit=None, kind=args.kind)
        result = {
            "path": readonly.display_path(path), "candidates": candidates[:arguments["limit"]],
            "truncated": len(candidates) > arguments["limit"],
            "note": ("Ranked definitions in this file only; scores are not comparable across files. "
                     "Read candidate source before answering."
                     if candidates else "No matching indexed function in this file; inspect its outline or another file."),
        }
        while result["candidates"] and len(json.dumps(result, ensure_ascii=False)) > args.max_output_chars:
            result["candidates"].pop()
            result["truncated"] = True
        compact_tool = "search_code"
    elif args.tool == "discover":
        readonly = executor.readonly
        missing = missing_qualified_definitions(readonly.retriever, args.query)
        abstained = exact_definition_lookup_should_abstain(args.query, missing)
        if abstained:
            result = {
                "groups": [], "truncated": False, "abstained": True,
                "missing_exact_definitions": missing,
                "note": "No direct indexed definition for the requested exact name. "
                        "It may be inherited or created dynamically; inspect the class before concluding "
                        "the behavior is absent.",
            }
        else:
            search_result = executor.call("search_code", arguments)
            paths = [readonly.resolve_repo_path(row["path"])
                     for row in search_result.get("results") or []]
            groups = locate_retrieved_files(
                readonly.retriever, root, paths, args.query,
                file_limit=args.top_k, symbols_per_file=args.per_file, kind=args.kind,
            )
            result = {
                "groups": groups, "truncated": False, "abstained": False,
                "missing_exact_definitions": missing,
                "note": "Candidate definitions grouped by retrieval rank, not a verified answer. "
                        "Scores only compare symbols within the same file; read source before answering."
                        + (" Some named direct definitions are missing from the index." if missing else ""),
            }
        bound_discovery_groups(result, args.max_output_chars)
        compact_tool = "search_code"
    else:
        compact_tool = {"search": "search_code", "trace": "trace_call_path", "read": "read_file"}[args.tool]
        result = executor.call(compact_tool, arguments)
    compact, observation = compact_tool_result(
        compact_tool,
        result,
        max_chars=args.max_output_chars,
    )
    response = {
        "schema_version": 1,
        "mode": "offline_readonly",
        "retriever": RETRIEVER_VERSION,
        "tool": args.tool,
        "output": compact,
        "observation": observation,
    }
    if args.tool == "search":
        index = result.get("retrieval_trace", {}).get("artifacts", {}).get("index", {})
        response["index"] = {
            key: index[key] for key in (
                "lifecycle", "manifest_scan_ms", "index_build_ms",
                "disk_read_ms", "disk_write_ms", "disk_write_status",
            ) if key in index
        }
    return response


def serve(args: argparse.Namespace, requests: TextIO, responses: TextIO) -> int:
    """Reuse one index registry across bounded JSONL requests without changing tool semantics."""
    root = args.repo.resolve()
    if not root.is_dir():
        raise ValueError("repository path must be an existing directory")
    if not 1000 <= args.max_output_chars <= 100000:
        raise ValueError("max-output-chars must be between 1000 and 100000")
    registry = RepositoryIndexRegistry(cache_dir=args.cache_dir)

    def emit(payload: dict) -> None:
        """Write one ASCII-safe JSON response and make it visible immediately."""
        responses.write(json.dumps(payload, ensure_ascii=True) + "\n")
        responses.flush()

    while True:
        line = requests.readline(SESSION_MAX_REQUEST_CHARS + 2)
        if not line:
            return 0
        if len(line) > SESSION_MAX_REQUEST_CHARS or not line.endswith("\n"):
            while line and not line.endswith("\n"):
                line = requests.readline(SESSION_MAX_REQUEST_CHARS + 2)
            emit({"ok": False, "error": "request exceeds line limit or lacks newline"})
            continue
        try:
            argv = json.loads(line)
            if (not isinstance(argv, list) or not 1 <= len(argv) <= 32
                    or any(not isinstance(item, str) or len(item) > 16384 for item in argv)
                    or argv[0] not in SESSION_TOOLS
                    or any(item in {"-h", "--help"} for item in argv)):
                raise ValueError("request must be an argv array for a read-only tool")
            prefix = [str(root)]
            if args.cache_dir is not None:
                prefix.extend(("--cache-dir", str(args.cache_dir)))
            prefix.extend(("--max-output-chars", str(args.max_output_chars)))
            parser_error = StringIO()
            with redirect_stderr(parser_error):
                try:
                    request_args = parse_args(prefix + argv)
                except SystemExit as exc:
                    raise ValueError(parser_error.getvalue().strip()[:500] or "invalid tool arguments") from exc
            emit({"ok": True, "result": inspect(request_args, index_registry=registry)})
        except (AgentToolError, OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            emit({"ok": False, "error": str(exc)[:500]})


def main(argv: list[str] | None = None) -> int:
    """Print JSON evidence with distinct success and invalid-request exit codes."""
    try:
        args = parse_args(argv)
        if args.tool == "serve":
            return serve(args, sys.stdin, sys.stdout)
        payload = inspect(args)
        # ASCII JSON survives Windows GBK terminals and UTF-8 consumers alike.
        print(json.dumps(payload, ensure_ascii=True))
        return 0
    except (AgentToolError, OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"RepoPilot read-only inspection error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
