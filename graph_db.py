"""DB-backed route-graph derived structures (DB_PRECOMPUTE_PLAN.md T5/T6).

T5 — ``RouteEngine.name_to_stems`` (``UPPER name -> {exporting stems}``) is rebuilt on EVERY
trace by sweeping *every* blueprint via ``_entry_names`` (asm_entry_point_map + global
symbols).  That corpus sweep is the single biggest blueprint-LRU thrash in the route layer.
We persist the map and pre-fill ``RouteEngine._name_to_stems`` at trace start so the sweep is
skipped; the forward-reachability prune then derives cheaply from the (already-loaded) graph
payload + this map.

Behavior preservation: the map's CONTENT is identity-preserving across the gz/JSON round-trip
(sets -> sorted lists -> sets), and every consumer treats values as sets (membership /
iteration into another set), so the forward-reachable file set — and therefore the trace
output — is byte-identical.  ``job_id`` None -> no-op (the sweep runs as before).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from vbt.precompute import db_artifacts as DA

logger = logging.getLogger(__name__)

ROUTE_NAME_TO_STEMS_ARTIFACT = "route_name_to_stems"
ROUTE_GRAPH_VERSION = 1

# T6: per-stem reattributed C++ function call edges (RouteEngine.cpp_call_edges) — the
# blueprint-read + cfg-reattribution that the function-level graph build does per reachable
# stem. Unlike the resolvers, this is normally built ONLY for reachable stems per trace, so
# the all-stems build lives in the explicit precompute step (build_and_store_cpp_call_edges),
# NOT lazily in the trace (that would make a cold first trace O(corpus)). The trace only LOADS.
CPP_CALL_EDGES_ARTIFACT = "cpp_call_edges"
CPP_EDGES_VERSION = 1

# T12: the file_call_graph payload itself, so a trace reads it from index.db (not the JSON file).
FILE_CALL_GRAPH_ARTIFACT = "file_call_graph"


def build_and_store_file_call_graph(job_id, blueprint_dir, graph_file=None) -> int:
    """Persist the file_call_graph payload (the SAME dict the trace loads via _load_graph_payload)
    so RouteEngine can read it from DB instead of the .json file. Returns the edge count."""
    bp = Path(blueprint_dir)
    gf = Path(graph_file) if graph_file else bp / "file_call_graph.json"
    from backward_traversal.runner.chainless_runner import _load_graph_payload
    payload = _load_graph_payload(gf, None)
    if DA.write_blob(job_id, FILE_CALL_GRAPH_ARTIFACT, DA.dumps_gz(payload)):
        DA.write_manifest(job_id, FILE_CALL_GRAPH_ARTIFACT, ROUTE_GRAPH_VERSION,
                          DA.source_manifest_hash([gf], version=ROUTE_GRAPH_VERSION))
    return len((payload or {}).get("edges", []))


def load_file_call_graph(job_id):
    """The file_call_graph payload from DB, or None if absent. O(1)."""
    payload = DA.read_blob(job_id, FILE_CALL_GRAPH_ARTIFACT)
    return DA.loads_gz(payload) if payload is not None else None


# B (cold-startup): the two derived structures that RouteEngine otherwise rebuilds from the raw
# 80MB payload on EVERY cold trace — `reverse_adj`+`node_type` (the walk_callers input; ~O(edges +
# call_sites)) and `forward_file_adj` (the metadata prune; a second full pass). Precomputing them
# lets `_ensure_graph` / `forward_reachable_files` LOAD small targeted blobs and skip the raw
# `json.loads` entirely (the payload is then loaded lazily only if `graph_edges()` is hit).
REVERSE_ADJ_ARTIFACT = "reverse_adj"
FORWARD_FILE_ADJ_ARTIFACT = "forward_file_adj"


def build_and_store_reverse_adj(job_id, blueprint_dir, graph_file=None) -> int:
    """Persist ``build_reverse_adjacency(payload)`` = ``(reverse_adj, node_type)``.

    ORDER-FAITHFUL (the byte-identity crux): ``reverse_adj[key]`` is a list of ``(src, instr)`` whose
    ORDER is load-bearing downstream (``walk_callers`` does ``candidates.setdefault(src, instr)`` —
    the FIRST occurrence wins). We store each list as a JSON array (order preserved) and rebuild it as
    ``list[tuple]`` in the same order on load, so the loaded adjacency is element-for-element identical
    to a fresh build. Returns the number of reverse keys."""
    bp = Path(blueprint_dir)
    gf = Path(graph_file) if graph_file else bp / "file_call_graph.json"
    payload = load_file_call_graph(job_id)
    if payload is None:
        from backward_traversal.runner.chainless_runner import _load_graph_payload
        payload = _load_graph_payload(gf, None)
    from backward_traversal.runner.chainless_caller_walker import build_reverse_adjacency
    reverse, node_type = build_reverse_adjacency(payload)
    # tuples serialize to JSON arrays; the list ORDER (the load-bearing part) is preserved.
    data = {"reverse": {k: [list(p) for p in v] for k, v in reverse.items()},
            "node_type": node_type}
    if DA.write_blob(job_id, REVERSE_ADJ_ARTIFACT, DA.dumps_gz(data)):
        DA.write_manifest(job_id, REVERSE_ADJ_ARTIFACT, ROUTE_GRAPH_VERSION,
                          DA.source_manifest_hash([gf], version=ROUTE_GRAPH_VERSION))
    return len(reverse)


def load_reverse_adj(job_id):
    """``(reverse_adj, node_type)`` from the blob, or ``None`` if absent. Rebuilds each value list as
    ``list[tuple]`` in the STORED order — byte-identical to ``build_reverse_adjacency``."""
    payload = DA.read_blob(job_id, REVERSE_ADJ_ARTIFACT)
    if payload is None:
        return None
    data = DA.loads_gz(payload)
    reverse = {k: [tuple(p) for p in v] for k, v in (data.get("reverse") or {}).items()}
    node_type = dict(data.get("node_type") or {})
    return reverse, node_type


def build_and_store_forward_file_adj(job_id, route) -> int:
    """Persist ``RouteEngine._build_forward_file_adj()`` = ``{src stem -> {dst stems}}``.

    SET-VALUED, so the round-trip (set → sorted list → set) is order-INDEPENDENT — every consumer
    (``forward_reachable_files`` BFS) treats values as sets, so the reachable file set is byte-
    identical. ``route`` must already have its payload + ``name_to_stems`` available (true at
    precompute time after ``preload_route_graph``). Returns the number of source stems."""
    adj = route._build_forward_file_adj()
    data = {k: sorted(v) for k, v in adj.items()}
    if DA.write_blob(job_id, FORWARD_FILE_ADJ_ARTIFACT, DA.dumps_gz(data)):
        DA.write_manifest(job_id, FORWARD_FILE_ADJ_ARTIFACT, ROUTE_GRAPH_VERSION,
                          DA.source_manifest_hash(_route_source_files(Path(route.blueprint_dir)),
                                                  version=ROUTE_GRAPH_VERSION))
    return len(data)


def load_forward_file_adj(job_id):
    """``{src stem -> set(dst stems)}`` from the blob, or ``None`` if absent."""
    payload = DA.read_blob(job_id, FORWARD_FILE_ADJ_ARTIFACT)
    if payload is None:
        return None
    return {k: set(v) for k, v in DA.loads_gz(payload).items()}


def _route_source_files(blueprint_dir: Path):
    """``name_to_stems`` depends on ``file_call_graph.json`` (node list) + every blueprint
    (``_entry_names`` reads ``asm_entry_point_map`` + global symbols).  Hashing all ``*.json``
    in the blueprint dir covers both (the graph file lives there too, freshly regenerated by
    ``ensure_call_graph`` before the RouteEngine is built)."""
    return sorted(Path(blueprint_dir).glob("*.json"))


def preload_route_graph(route, job_id: Optional[str], blueprint_dir, graph_file=None) -> bool:
    """Pre-fill ``route._name_to_stems`` from DB (or build via the sweep + persist).

    Returns True iff it touched ``route._name_to_stems``.  Best-effort: any failure falls back
    to the normal lazy sweep (``route.name_to_stems()`` later)."""
    if not job_id:
        return False

    # T12 trace-time fast path: O(1) blob load; no per-blueprint sweep, no source hashing.
    if DA.is_load_only():
        payload = DA.read_blob(job_id, ROUTE_NAME_TO_STEMS_ARTIFACT)
        if payload is not None:
            try:
                route._name_to_stems = {k: set(v) for k, v in DA.loads_gz(payload).items()}
                return True
            except Exception as exc:
                logger.debug("load-only route name_to_stems failed (sweep fallback): %s", exc)
        return False

    bp = Path(blueprint_dir)
    try:
        shash = DA.source_manifest_hash(_route_source_files(bp), version=ROUTE_GRAPH_VERSION)
    except Exception as exc:
        logger.debug("route-graph source hash failed (lazy sweep will run): %s", exc)
        return False

    if DA.manifest_fresh(job_id, ROUTE_NAME_TO_STEMS_ARTIFACT,
                         version=ROUTE_GRAPH_VERSION, source_hash=shash):
        payload = DA.read_blob(job_id, ROUTE_NAME_TO_STEMS_ARTIFACT)
        if payload is not None:
            try:
                data = DA.loads_gz(payload)
                route._name_to_stems = {k: set(v) for k, v in data.items()}
                return True
            except Exception as exc:
                logger.debug("load route name_to_stems failed, rebuilding: %s", exc)

    # Build via the blueprint sweep (fills route._name_to_stems) + persist for next time.
    m = route.name_to_stems()
    try:
        if DA.write_blob(job_id, ROUTE_NAME_TO_STEMS_ARTIFACT,
                         DA.dumps_gz({k: sorted(v) for k, v in m.items()})):
            DA.write_manifest(job_id, ROUTE_NAME_TO_STEMS_ARTIFACT, ROUTE_GRAPH_VERSION, shash)
    except Exception as exc:
        logger.debug("persist route name_to_stems failed (non-fatal): %s", exc)
    return True


# --------------------------------------------------------------------------- #
# T6 — per-stem C++ function call edges (cpp_call_edges)
# --------------------------------------------------------------------------- #
def _cpp_edges_source_files(blueprint_dir: Path, asm_dir: Path):
    """Inputs to cpp_call_edges: each cpp blueprint's call_graph + the .cpp source (cfg
    reattribution) + the cfg_extract binary (its output drives reattribution)."""
    files = sorted(Path(blueprint_dir).glob("*.cpp.json")) + sorted(Path(asm_dir).glob("*.cpp"))
    try:
        from vbt.cpp_frontend import wrapper as _w
        files.append(Path(_w.DEFAULT_BINARY))
    except Exception:
        pass
    return files


def build_and_store_cpp_call_edges(job_id, blueprint_dir, asm_dir, graph_file):
    """Precompute (one-shot) the reattributed cpp_call_edges for EVERY cpp stem and persist.

    Belongs to the precompute step — NOT the trace path (it is O(corpus): one blueprint read +
    cfg-reattribution per cpp stem). Returns the built {stem: [[caller,callee,line],...]} map."""
    from vbt.reach.route import RouteEngine
    bp, src = Path(blueprint_dir), Path(asm_dir)
    shash = DA.source_manifest_hash(_cpp_edges_source_files(bp, src), version=CPP_EDGES_VERSION)
    route = RouteEngine(bp, Path(graph_file), src)
    node_types = route.node_types()
    data = {}
    for stem in sorted(s for s, t in node_types.items() if t == "cpp"):
        data[stem] = [list(e) for e in route.cpp_call_edges(stem)]
    if DA.write_blob(job_id, CPP_CALL_EDGES_ARTIFACT, DA.dumps_gz(data)):
        DA.write_manifest(job_id, CPP_CALL_EDGES_ARTIFACT, CPP_EDGES_VERSION, shash)
    return data


# --------------------------------------------------------------------------- #
# fn-graph adjacency — the full unified function-level call graph, pre-built.
# --------------------------------------------------------------------------- #
# `CppFnGraph._build()` (the per-trace function-graph build over the chain-tail-reachable file
# set) is the cold trace's dominant ONE-TIME cost at 22k scale (seconds). The adjacency is a pure
# function of the corpus (cfg + cpp_call_edges + the file graph), so it is precomputed ONCE over
# ALL stems here and a load-only trace LOADS it (~0.1s) instead of rebuilding. Byte-identical:
# `enumerate_paths`' DFS from a given `start` only ever visits nodes forward-reachable FROM that
# start, and the distance bound is a shortest-path property — so the paths enumerated over the
# FULL-corpus adjacency are exactly those the per-trace tail-reachable SUBSET build yields (any
# node on a start→target path is reachable from start, hence already in the subset; unreachable
# extra edges are never traversed). A cold/absent blob → live subset build (fallback). Ported from
# optimised_v3, adapted to this branch's CppFnGraph (hybrid DFS + distance bound + fn_facts).
FN_GRAPH_ADJ_ARTIFACT = "fn_graph_adj"
FN_GRAPH_ADJ_META_ARTIFACT = "fn_graph_adj_meta"
# v2: C++ cross-file edges fan out to EVERY owner of a called name (not just
# sorted(owners)[0]) and are emitted even when the name is also defined locally.
# v3: ASM-source edges fan out to all entry-name owners too (not just the file-graph's
# resolved target), and C++ calls to entry/global symbols not seen as cfg functions
# (e.g. DF_OK) resolve via the corpus entry-name map. Closes the residual under-report
# that orphaned dw780000/dw780100. Bump invalidates older artifacts so they rebuild.
FN_GRAPH_ADJ_VERSION = 3


def build_and_store_fn_graph_adj(job_id, route, asm_dir) -> int:
    """Pre-build the full-corpus unified fn-graph adjacency and persist as one blob.

    Belongs to the precompute step (it runs the full O(corpus) build with load_only OFF, so the
    cfg/edge reads happen live). Returns the node count. Stored as
    ``{f"{stem}\\t{fn}": sorted([[cs,cf,ln,cr], ...])}`` (a list, so the gz-JSON is deterministic)."""
    from vbt.reach.cpp_routes import CppFnGraph
    from vbt.precompute.fn_facts_db import set_fn_facts_job
    src = Path(asm_dir)
    node_types = route.node_types()
    all_stems = set(node_types)
    # Reuse the just-built fn_facts artifact when present. Without this, the
    # full-corpus adjacency build re-loads CFG for every C++ stem and repeats
    # cfg_extract failures that cfg-warm/fn_facts already discovered.
    set_fn_facts_job(job_id)
    try:
        graph = CppFnGraph(route, {}, src, all_stems, node_types)
        adj = graph.adjacency()
    finally:
        set_fn_facts_job(None)
    data = {f"{stem}\t{fn}": sorted([cs, cf, ln, cr] for cs, cf, ln, cr in edges)
            for (stem, fn), edges in adj.items()}
    shash = DA.source_manifest_hash(
        _cpp_edges_source_files(Path(route.blueprint_dir), src),
        version=FN_GRAPH_ADJ_VERSION)
    if DA.write_blob(job_id, FN_GRAPH_ADJ_ARTIFACT, DA.dumps_gz(data)):
        DA.write_manifest(job_id, FN_GRAPH_ADJ_ARTIFACT, FN_GRAPH_ADJ_VERSION,
                          shash)
    skipped = dict(getattr(graph, "skipped_cfg_stems", {}) or {})
    empty = sorted(getattr(graph, "empty_cfg_stems", set()) or [])
    meta = {
        "nodes": len(adj),
        "stems": len(all_stems),
        "skipped_cfg_count": len(skipped),
        "skipped_cfg": skipped,
        "empty_cfg_count": len(empty),
        "empty_cfg": empty,
    }
    DA.write_blob(job_id, FN_GRAPH_ADJ_META_ARTIFACT, DA.dumps_gz(meta))
    if skipped:
        examples = ", ".join(f"{k}: {v}" for k, v in sorted(skipped.items())[:5])
        logger.warning("fn_graph_adj: skipped %d C++ stems with unavailable CFG (%s)",
                       len(skipped), examples)
    return len(adj)


def load_fn_graph_adj(job_id):
    """Load the pre-built fn-graph adjacency, or ``None`` if absent (→ live build).

    Returns ``Dict[Tuple[str,str], Set[Tuple[str,str,int,bool]]]`` — exactly the shape
    ``CppFnGraph.adjacency()`` produces, so ``CppFnGraph.from_prebuilt`` is a drop-in."""
    if not job_id:
        return None
    _t0 = time.perf_counter()
    payload = DA.read_blob(job_id, FN_GRAPH_ADJ_ARTIFACT)
    if payload is None:
        return None
    data = DA.loads_gz(payload)
    adj = {}
    for key, edges in data.items():
        stem, fn = key.split("\t", 1)
        adj[(stem, fn)] = {tuple(e) for e in edges}
    logger.debug("load_fn_graph_adj: %d nodes (%.3fs)", len(adj), time.perf_counter() - _t0)
    return adj


def preload_cpp_call_edges(route, job_id: Optional[str], blueprint_dir, asm_dir) -> bool:
    """LOAD-ONLY: if a fresh DB artifact exists, pre-fill ``route._edge_cache`` so the
    function-graph build does no blueprint reads. Returns False (no-op) on a cold/stale DB —
    the trace then builds cpp_call_edges per reachable stem exactly as today (byte-identical).
    Never builds the full corpus inline (that is the precompute step's job)."""
    if not job_id:
        return False
    bp, src = Path(blueprint_dir), Path(asm_dir)
    # T12: at trace time (load-only) skip the source hash — just load the blob by presence.
    if not DA.is_load_only():
        try:
            shash = DA.source_manifest_hash(_cpp_edges_source_files(bp, src), version=CPP_EDGES_VERSION)
        except Exception:
            return False
        if not DA.manifest_fresh(job_id, CPP_CALL_EDGES_ARTIFACT,
                                 version=CPP_EDGES_VERSION, source_hash=shash):
            return False
    payload = DA.read_blob(job_id, CPP_CALL_EDGES_ARTIFACT)
    if payload is None:
        return False
    try:
        data = DA.loads_gz(payload)
    except Exception as exc:
        logger.debug("load cpp_call_edges failed (lazy build will run): %s", exc)
        return False
    for stem, edges in data.items():
        route._edge_cache[stem] = [tuple(e) for e in edges]
    return True
