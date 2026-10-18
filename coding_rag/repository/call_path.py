"""Bounded, read-only Python call-path navigation over indexed repository source."""

from __future__ import annotations

import ast
import symtable
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path

from coding_rag.repository.chunks import CodeChunk
from coding_rag.repository.index import RepoIndex, normalize_path, rebuild_source


@dataclass(frozen=True)
class ImportedReference:
    """One import-bound module or symbol that can be followed statically."""

    module: str
    symbol: str | None = None


@dataclass
class ModuleCalls:
    """Parsed top-level bindings and directly addressable function scopes."""

    bindings: dict[str, ImportedReference]
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
    shadowed: dict[str, frozenset[str]]
    local_bindings: dict[str, dict[str, ImportedReference]]
    ambiguous_functions: frozenset[str]


def dotted_name(node: ast.AST) -> str | None:
    """Return a syntactic call name; dynamic expressions remain unresolved."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def matching_scope(parent: symtable.SymbolTable, node: ast.AST) -> symtable.SymbolTable | None:
    """Match a direct function/class scope by Python's symbol-table identity."""
    return next(
        (child for child in parent.get_children()
         if child.get_name() == node.name and child.get_lineno() == node.lineno),
        None,
    )


def local_names(scope: symtable.SymbolTable | None) -> frozenset[str]:
    """Names that Python will resolve in a function scope before module imports."""
    if scope is None:
        return frozenset()
    return frozenset(symbol.get_name() for symbol in scope.get_symbols() if symbol.is_local())


def direct_calls(function: ast.FunctionDef | ast.AsyncFunctionDef):
    """Yield calls in a body, excluding nested definitions and deferred lambdas."""
    pending = list(reversed(function.body))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            yield node
        pending.extend(reversed(list(ast.iter_child_nodes(node))))


def direct_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef):
    """Yield nodes in one function without entering deferred or nested scopes."""
    pending = list(reversed(function.body))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        pending.extend(reversed(list(ast.iter_child_nodes(node))))


class PythonCallPathIndex:
    """Resolve import-bound calls without importing or executing repository code."""

    def __init__(self, chunks: list[CodeChunk], repo_index: RepoIndex):
        """Map indexed modules and chunks without executing repository code."""
        self.chunks = chunks
        self.repo_index = repo_index
        self.module_paths: dict[str, str] = {}
        self.chunk_indexes: dict[str, list[int]] = {}
        self._module_cache: OrderedDict[str, ModuleCalls] = OrderedDict()
        for key, file_index in repo_index.files.items():
            self.module_paths[file_index.module] = key
            if file_index.module.endswith(".__init__"):
                self.module_paths[file_index.module.removesuffix(".__init__")] = key
        for index, chunk in enumerate(chunks):
            self.chunk_indexes.setdefault(normalize_path(chunk.file_path), []).append(index)

    @staticmethod
    def _import_base(source_module: str, level: int, module: str | None) -> str:
        """Resolve the syntactic base of an absolute or relative import."""
        if level == 0:
            return module or ""
        package = (
            source_module.removesuffix(".__init__")
            if source_module.endswith(".__init__")
            else source_module.rsplit(".", 1)[0]
        )
        parts = package.split(".")
        if level > 1:
            parts = parts[: -(level - 1)]
        return ".".join([*parts, *([module] if module else [])])

    def _module_calls(self, module: str) -> ModuleCalls:
        """Parse one indexed module on demand and retain a bounded LRU entry."""
        cached = self._module_cache.get(module)
        if cached is not None:
            self._module_cache.move_to_end(module)
            return cached
        key = self.module_paths.get(module)
        if key is None:
            return ModuleCalls({}, {}, {}, {}, frozenset())
        file_index = self.repo_index.files[key]
        source = rebuild_source([self.chunks[index] for index in self.chunk_indexes[key]])
        try:
            tree = ast.parse(source)
            module_scope = symtable.symtable(source, str(file_index.path), "exec")
        except SyntaxError:
            return ModuleCalls({}, {}, {}, {}, frozenset())
        bindings: dict[str, ImportedReference] = {}
        functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        shadowed: dict[str, frozenset[str]] = {}
        local_bindings: dict[str, dict[str, ImportedReference]] = {}
        ambiguous_functions: set[str] = set()

        def import_bindings(nodes: list[ast.stmt]) -> dict[str, ImportedReference]:
            """Collect direct imports from one scope without entering deferred scopes."""
            imported: dict[str, ImportedReference] = {}
            for statement in nodes:
                if isinstance(statement, ast.Import):
                    for alias in statement.names:
                        bound = alias.asname or alias.name.split(".", 1)[0]
                        imported[bound] = ImportedReference(alias.name if alias.asname else bound)
                elif isinstance(statement, ast.ImportFrom):
                    base = self._import_base(file_index.module, statement.level, statement.module)
                    for alias in statement.names:
                        candidate = f"{base}.{alias.name}" if base else alias.name
                        imported[alias.asname or alias.name] = (
                            ImportedReference(candidate)
                            if candidate in self.module_paths
                            else ImportedReference(base, alias.name)
                        )
            return imported

        def register_function(
            node: ast.FunctionDef | ast.AsyncFunctionDef,
            qualified: str,
            parent_scope: symtable.SymbolTable | None,
        ) -> None:
            """Register an addressable lexical function without attributing its calls to parents."""
            scope = matching_scope(parent_scope, node) if parent_scope else None
            if qualified in functions:
                ambiguous_functions.add(qualified)
                return
            functions[qualified] = node
            shadowed[qualified] = local_names(scope)
            local_bindings[qualified] = import_bindings(node.body)
            for statement in node.body:
                for child in nested_definitions(statement):
                    register_function(child, f"{qualified}.{child.name}", scope)

        def nested_definitions(node: ast.AST):
            """Find definitions in this lexical scope, including conditional branches."""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield node
                return
            if isinstance(node, (ast.ClassDef, ast.Lambda)):
                return
            for child in ast.iter_child_nodes(node):
                yield from nested_definitions(child)

        bindings.update(import_bindings(tree.body))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                register_function(node, node.name, module_scope)
            elif isinstance(node, ast.ClassDef):
                class_scope = matching_scope(module_scope, node)
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        register_function(child, f"{node.name}.{child.name}", class_scope)
        # Simple package-level instance: engines = EngineHandler(). This remains
        # a static origin hint, not a claim about runtime dispatch or type flow.
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            constructor = dotted_name(node.value.func)
            if constructor not in bindings:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = bindings[constructor]
        result = ModuleCalls(bindings, functions, shadowed, local_bindings,
                             frozenset(ambiguous_functions))
        self._module_cache[module] = result
        if len(self._module_cache) > 128:
            self._module_cache.popitem(last=False)
        return result

    def _resolve_export(self, reference: ImportedReference) -> ImportedReference:
        """Follow a short chain of package-level reexports, stopping on cycles."""
        seen: set[ImportedReference] = set()
        for _ in range(4):
            if reference in seen or reference.symbol is None:
                break
            seen.add(reference)
            next_reference = self._module_calls(reference.module).bindings.get(reference.symbol)
            if next_reference is None:
                break
            reference = next_reference
        return reference

    def _symbol_range(
        self, path_key: str, symbol: str | None, parent: str | None = None
    ) -> tuple[int | None, int | None]:
        """Find the indexed line range of a target symbol and optional class."""
        if symbol:
            for record in self.repo_index.files[path_key].symbols:
                if record.name == symbol and (parent is None or record.parent == parent):
                    return record.start_line, record.end_line
        return None, None

    @staticmethod
    def _lexical_function(module_calls: ModuleCalls, current_name: str, call_name: str) -> str | None:
        """Resolve a bare nested function in its lexical scope before module functions."""
        if "." not in call_name:
            parent = current_name
            while parent in module_calls.functions:
                candidate = f"{parent}.{call_name}"
                if candidate in module_calls.functions and candidate not in module_calls.ambiguous_functions:
                    return candidate
                parent = parent.rsplit(".", 1)[0] if "." in parent else ""
        return (call_name if call_name in module_calls.functions
                and call_name not in module_calls.ambiguous_functions else None)

    def _assigned_instance_candidates(
        self, file_index, module_calls: ModuleCalls,
        function: ast.FunctionDef | ast.AsyncFunctionDef, symbol: str,
    ) -> list[dict]:
        """Hint at methods of imported instances assigned once to self in this function.

        This is syntactic provenance, not proof of the value at a call site: an
        existing attribute or branch may supply a different runtime instance.
        """
        if (
            symbol.count(".") != 1
            or not (function.args.posonlyargs + function.args.args)
            or (function.args.posonlyargs + function.args.args)[0].arg != "self"
            or any(isinstance(node, ast.Name) and node.id == "self"
                   and isinstance(node.ctx, (ast.Store, ast.Del))
                   for node in direct_nodes(function))
        ):
            return []
        writes: dict[str, list[ast.AST]] = {}
        for node in direct_nodes(function):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name) and node.value.id == "self"
                    and isinstance(node.ctx, (ast.Store, ast.Del))):
                writes.setdefault(node.attr, []).append(node)
        constructors: dict[str, tuple[int, ImportedReference]] = {}
        for node in direct_nodes(function):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if (not isinstance(target, ast.Attribute)
                    or not isinstance(target.value, ast.Name) or target.value.id != "self"
                    or len(writes.get(target.attr, ())) != 1
                    or not isinstance(node.value, ast.Call)
                    or not isinstance(node.value.func, ast.Name)):
                continue
            constructor = node.value.func.id
            binding = module_calls.local_bindings.get(symbol, {}).get(constructor)
            if binding is None and constructor not in module_calls.shadowed.get(symbol, ()):
                binding = module_calls.bindings.get(constructor)
            if binding is None:
                continue
            reference = self._resolve_export(binding)
            target_key = self.module_paths.get(reference.module)
            if target_key is None or target_key == normalize_path(file_index.path):
                continue
            target_file = self.repo_index.files[target_key]
            if not any(record.name == reference.symbol and record.kind == "class"
                       for record in target_file.symbols):
                continue
            constructors[target.attr] = (node.lineno, reference)
        candidates: list[dict] = []
        for node in direct_calls(function):
            if not isinstance(node.func, ast.Attribute):
                continue
            receiver = node.func.value
            if (not isinstance(receiver, ast.Attribute)
                    or not isinstance(receiver.value, ast.Name)
                    or receiver.value.id != "self"
                    or receiver.attr not in constructors):
                continue
            assignment_line, reference = constructors[receiver.attr]
            if node.lineno <= assignment_line or reference.symbol is None:
                continue
            target_key = self.module_paths[reference.module]
            target_module = self._module_calls(reference.module)
            target_symbol = f"{reference.symbol}.{node.func.attr}"
            if (target_symbol not in target_module.functions
                    or target_symbol in target_module.ambiguous_functions):
                continue
            target = target_module.functions[target_symbol]
            candidates.append({
                "source_symbol": symbol,
                "source_line": node.lineno,
                "assignment_line": assignment_line,
                "call": dotted_name(node.func),
                "target_path": str(self.repo_index.files[target_key].path),
                "target_symbol": target_symbol,
                "target_start_line": target.lineno,
                "target_end_line": target.end_lineno,
                "kind": "assigned_instance_method_candidate",
            })
            if len(candidates) >= 20:
                break
        return candidates

    def trace(
        self, path: Path, symbol: str, *, max_local_depth: int = 2, max_edges: int = 30,
        include_same_class_candidates: bool = False,
        include_unresolved_calls: bool = False, max_unresolved_calls: int = 40,
        include_assigned_instance_candidates: bool = False,
    ) -> dict:
        """Return source-grounded static edges from one exact Python function or method."""
        if max_edges <= 0 or max_local_depth < 0 or max_unresolved_calls <= 0:
            raise ValueError("edge/call limits must be positive and local depth non-negative")
        key = normalize_path(path)
        file_index = self.repo_index.files.get(key)
        if file_index is None or path.suffix.casefold() != ".py":
            raise ValueError("path is not an indexed Python file")
        module_calls = self._module_calls(file_index.module)
        if symbol in module_calls.ambiguous_functions:
            raise ValueError("symbol has multiple conditional definitions; specify an unambiguous function")
        function = module_calls.functions.get(symbol)
        if function is None:
            raise ValueError("symbol is not an indexed function or method in this file")
        pending = deque([(symbol, 0)])
        visited: set[str] = set()
        edges: list[dict] = []
        seen_edges: set[tuple[str, int, str, str | None]] = set()
        truncated = False
        while pending and len(edges) < max_edges:
            current_name, depth = pending.popleft()
            if current_name in visited:
                continue
            visited.add(current_name)
            current = module_calls.functions[current_name]
            self_rebound = include_same_class_candidates and any(
                isinstance(child, ast.Name) and child.id == "self" and isinstance(child.ctx, ast.Store)
                for statement in current.body for child in ast.walk(statement)
            )
            for node in direct_calls(current):
                if len(edges) >= max_edges:
                    truncated = True
                    break
                if not isinstance(node, ast.Call):
                    continue
                call_name = dotted_name(node.func)
                if call_name is None:
                    continue
                binding, _, attribute = call_name.partition(".")
                if (
                    include_same_class_candidates and binding == "self"
                    and attribute and "." not in attribute
                    and current_name.count(".") == 1
                    and (current.args.posonlyargs + current.args.args)
                    and (current.args.posonlyargs + current.args.args)[0].arg == "self"
                    and not self_rebound
                ):
                    class_name = current_name.split(".", 1)[0]
                    target_name = f"{class_name}.{attribute}"
                    target = module_calls.functions.get(target_name)
                    if target is not None and target_name not in module_calls.ambiguous_functions:
                        edge_key = (current_name, node.lineno, str(path), target_name)
                        if edge_key not in seen_edges:
                            edges.append({
                                "source_symbol": current_name,
                                "source_line": node.lineno,
                                "call": call_name,
                                "target_path": str(path),
                                "target_symbol": target_name,
                                "target_start_line": target.lineno,
                                "target_end_line": target.end_lineno,
                                "kind": "same_class_method_candidate",
                            })
                            seen_edges.add(edge_key)
                        continue
                local_binding = module_calls.local_bindings.get(current_name, {}).get(binding)
                lexical = None if local_binding is not None else self._lexical_function(
                    module_calls, current_name, call_name
                )
                if binding in module_calls.shadowed.get(current_name, ()) and local_binding is None:
                    # A local variable or parameter cannot resolve to a module import or sibling.
                    if lexical != f"{current_name}.{call_name}":
                        continue
                if lexical is not None:
                    local = module_calls.functions[lexical]
                    edge = {
                        "source_symbol": current_name,
                        "source_line": node.lineno,
                        "call": call_name,
                        "target_path": str(path),
                        "target_symbol": lexical,
                        "target_start_line": local.lineno,
                        "target_end_line": local.end_lineno,
                        "kind": "local_call",
                    }
                    edge_key = (current_name, node.lineno, str(path), lexical)
                    if edge_key not in seen_edges:
                        edges.append(edge)
                        seen_edges.add(edge_key)
                    if depth < max_local_depth:
                        pending.append((lexical, depth + 1))
                    continue
                reference = local_binding or module_calls.bindings.get(binding)
                if reference is None:
                    continue
                reference = self._resolve_export(reference)
                target_key = self.module_paths.get(reference.module)
                if target_key is None or target_key == key:
                    continue
                target_symbol = attribute.rsplit(".", 1)[-1] if attribute else reference.symbol
                parent = reference.symbol if attribute and reference.symbol else None
                start, end = self._symbol_range(target_key, target_symbol, parent)
                target_path = str(self.repo_index.files[target_key].path)
                edge_key = (current_name, node.lineno, target_path, target_symbol)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append({
                    "source_symbol": current_name,
                    "source_line": node.lineno,
                    "call": call_name,
                    "target_path": target_path,
                    "target_symbol": target_symbol,
                    "target_start_line": start,
                    "target_end_line": end,
                    "kind": "import_bound_call",
                })
                if len(edges) >= max_edges:
                    truncated = True
                    break
        result = {
            "path": str(path),
            "symbol": symbol,
            "start_line": function.lineno,
            "end_line": function.end_lineno,
            "edges": edges,
            "truncated": truncated or bool(pending),
            "limitations": (
                "Static Python imports and local calls only; same_class_method_candidate "
                "is syntactic and may have dynamic dispatch elsewhere at runtime. Other dynamic targets are "
                "not inferred. Returned closures must be traced by exact qualified symbol."
            ),
        }
        if include_unresolved_calls:
            resolved = {
                (edge["source_line"], edge["call"])
                for edge in edges if edge["source_symbol"] == symbol
            }
            # These are call-site breadcrumbs only. Attribute access alone cannot
            # establish the callee's identity or even that it is in this repo.
            unresolved = [
                {"source_line": node.lineno, "call": name}
                for node in direct_calls(function)
                if (name := dotted_name(node.func)) is not None
                and "." in name and (node.lineno, name) not in resolved
            ]
            result["unresolved_calls"] = unresolved[:max_unresolved_calls]
            result["unresolved_calls_truncated"] = len(unresolved) > max_unresolved_calls
            result["unresolved_note"] = (
                "Syntactic call sites in the requested function, not target edges. "
                "No target path or runtime dispatch is inferred; inspect the source before following one."
            )
        if include_assigned_instance_candidates:
            result["assigned_instance_candidates"] = self._assigned_instance_candidates(
                file_index, module_calls, function, symbol,
            )
            result["assigned_instance_note"] = (
                "Static same-function assignment and imported class-method hint only. "
                "Conditional paths, pre-existing attributes, and runtime dispatch are not proven. "
                "Read the assignment, call, and target definition before relying on it."
            )
        return result
