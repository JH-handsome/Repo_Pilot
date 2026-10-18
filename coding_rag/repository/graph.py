"""Query-personalized dependency graph scores for repository retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from coding_rag.repository.index import RepoIndex, normalize_path


def build_dependency_graph(repo_index: RepoIndex) -> dict[str, tuple[str, ...]]:
    """Build an undirected file graph from imports resolved to indexed modules."""
    module_paths: dict[str, str] = {}
    for path, file_index in repo_index.files.items():
        module = file_index.module.casefold()
        module_paths[module] = path
        if module.endswith(".__init__"):
            module_paths[module.removesuffix(".__init__")] = path

    neighbors: dict[str, set[str]] = {path: set() for path in repo_index.files}
    for source, file_index in repo_index.files.items():
        for imported in file_index.imports:
            target = module_paths.get(imported.casefold())
            if target is None or target == source:
                continue
            neighbors[source].add(target)
            neighbors[target].add(source)

    return {path: tuple(sorted(linked)) for path, linked in neighbors.items()}


def personalized_dependency_scores(
    repo_index: RepoIndex,
    chunk_paths: Sequence[object],
    seed_scores: Sequence[float],
    *,
    damping: float = 0.75,
    iterations: int = 12,
) -> list[float]:
    """Diffuse lexical relevance across imports and return one score per chunk."""
    if len(chunk_paths) != len(seed_scores):
        raise ValueError("chunk_paths and seed_scores must have equal length")
    if not chunk_paths:
        return []
    if not 0.0 <= damping <= 1.0:
        raise ValueError("damping must be between 0 and 1")

    graph = build_dependency_graph(repo_index)
    file_seeds = {path: 0.0 for path in repo_index.files}
    for path, score in zip(chunk_paths, seed_scores):
        key = normalize_path(path)
        if key in file_seeds:
            file_seeds[key] = max(file_seeds[key], max(float(score), 0.0))

    seed_total = sum(file_seeds.values())
    if seed_total <= 0:
        uniform = 1.0 / max(len(file_seeds), 1)
        personalization = {path: uniform for path in file_seeds}
    else:
        personalization = {path: score / seed_total for path, score in file_seeds.items()}

    rank = dict(personalization)
    for _ in range(max(iterations, 0)):
        updated = {
            path: (1.0 - damping) * personalization[path]
            for path in personalization
        }
        dangling = 0.0
        for source, score in rank.items():
            linked = graph.get(source, ())
            if not linked:
                dangling += damping * score
                continue
            share = damping * score / len(linked)
            for target in linked:
                updated[target] += share
        if dangling:
            for path, weight in personalization.items():
                updated[path] += dangling * weight
        total = sum(updated.values())
        rank = {path: value / total for path, value in updated.items()} if total else updated

    maximum = max(rank.values(), default=0.0)
    if maximum <= 0:
        return [0.0 for _ in chunk_paths]
    return [rank.get(normalize_path(path), 0.0) / maximum for path in chunk_paths]
