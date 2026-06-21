"""Forward route finder — is there a call route parent → child, and which ones?

Given a **parent** (a C++ function in a file, or an ASM file) and a **child**
(a C++ function in a file, or an ASM file), this answers:

    1. Is there a call route  parent → … → child ?
    2. If yes, enumerate every route (file hops, each annotated with the calling
       function and the guard condition under which the call fires).

Primary use is **cpp → cpp** at function granularity, but all four directions
(cpp→cpp, cpp→asm, asm→cpp, asm→asm) are covered because the work is delegated
to the four mature cross-language *caller* bridges via ``walk_callers``.

Design — reuse over reinvention
-------------------------------
Rather than build a second (forward) set of cross-language bridges, this walks
the **callers of the child** with the existing ``walk_callers`` (the four
``cross_file_bridge_backward`` bridges + indirection handling + per-call-site
guard conditions), which already resolves each cross-file call at function
granularity (``caller_function`` / ``callee_function`` / ``cpp_function``).  The
caller forest is naturally restricted to files that reach the child, so the
parent appears in it **iff a route exists**.  We then:

* build a **verified** forward adjacency (caller→callee) from those call sites,
* enumerate simple routes parent → child, enforcing **function continuity** at
  every hop via the intra-file call graph (``chainless_reachability``):
  the function entered in a file must transitively reach the function that makes
  the next call.

Soundness mirrors the rest of the chainless engine: only **verified** call
edges form routes; cycle/indirection candidates are surfaced separately
(``unverified``), never woven into a confident route.
"""

from __future__ import annotations

import copy
import logging
import os as _os
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from backward_traversal.runner import chainless_reachability as RCH

# Under the "vbt" logger root so the same --progress switch controls the hang-locating heartbeat.
_HANG_LOG = logging.getLogger("vbt.routefinder")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@dataclass
class Endpoint:
    """A route endpoint: a C++ function in a file, or an ASM file.

    ``function`` is the C++ function name (recommended for cpp; routes are then
    function-accurate).  For ASM leave it ``None`` — an ASM module is reached at
    a single entry, so the file IS the granularity.
    """
    file_stem: str
    file_type: str = "cpp"               # "cpp" | "asm"
    function: Optional[str] = None

    def norm(self) -> "Endpoint":
        return Endpoint(
            file_stem=str(self.file_stem).strip(),
            file_type=(self.file_type or "cpp").lower(),
            function=(self.function or None),
        )


# ---------------------------------------------------------------------------
# Forward adjacency over the verified caller forest
# ---------------------------------------------------------------------------

def _build_forward_edges(
    caller_nodes: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """From the child's caller forest, build verified forward edges.

    Returns ``(fwd, unverified)`` where ``fwd[caller_stem]`` is a list of edge
    dicts ``{callee, caller_fn, callee_fn, line, conditions}`` and ``unverified``
    is the list of caller files the bridge could not confirm (surfaced, not
    woven into routes).
    """
    fwd: Dict[str, List[Dict[str, Any]]] = {}
    unverified: Set[str] = set()
    verified_callees: Set[str] = set()
    for c in caller_nodes:
        if not isinstance(c, dict):
            continue
        caller = str(c.get("stem") or "")
        callee = str(c.get("callee_stem") or "")
        if not caller or not callee:
            continue
        src_type = str(c.get("file_type") or "")
        sites = c.get("call_sites") or []
        if not sites:
            if not c.get("cycle"):
                unverified.add(caller)
            continue
        verified_callees.add(callee)
        for s in sites:
            if not isinstance(s, dict):
                continue
            fwd.setdefault(caller, []).append({
                "callee": callee,
                "caller_fn": RCH.site_caller_fn(s),
                "callee_fn": RCH.site_target_fn(s) or "",
                "line": s.get("line"),
                "conditions": list(s.get("conditions") or []),
                # carried so conditions can be attached lazily AFTER route enumeration, only for
                # the sites on an emitted hop (find_routes' attach_conditions=False path leaves
                # s["conditions"] unset until then). _site/_src_stem/_src_type are what
                # _attach_call_site_conditions needs; internal-only, never reach the output route.
                "_site": s,
                "_src_stem": caller,
                "_src_type": src_type,
            })
    return fwd, sorted(unverified - verified_callees)


# ---------------------------------------------------------------------------
# Route enumeration  (parent → child, function-continuous)
# ---------------------------------------------------------------------------

# Hard visit cap for the route-enumeration DFS (mirrors enumerate_cpp_paths' _DFS_HARD_CAP). A dense
# forward subgraph between (parent, child) has combinatorially many simple paths; when the real route
# count is far below max_routes the DFS would otherwise exhaust that whole space looking for routes that
# don't exist (observed on a hub pair: 270M+ visits, 2 routes, ~hours — effectively a hang). Capping
# GUARANTEES termination: it stops and returns the routes found so far with capped=True. Byte-identical
# when it doesn't fire — every reference trace finishes far under it, so parity stays 3/3; only a
# pathological dense graph reaches it, where the alternative is an unbounded hang. Env-overridable.
_ENUM_HARD_CAP = int(_os.environ.get("VBT_ENUM_VISIT_CAP", "5000000"))


def _enumerate_routes(
    parent: Endpoint, child: Endpoint,
    fwd: Dict[str, List[Dict[str, Any]]],
    blueprint_dir: Path, *, max_routes: int, max_len: int,
    reach_cache: Optional[Dict[str, Any]] = None,
    asm_dir: Optional[Path] = None,
    pfl_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """DFS parent → child over the verified forward edges with function
    continuity.  Returns ``(routes, capped)``."""
    routes: List[Dict[str, Any]] = []
    capped = False
    if reach_cache is None:
        reach_cache = {}

    # Distance bound (admissible ⇒ byte-identical): dist[stem] = MIN file-hops stem→child over `fwd`
    # (backward BFS). A branch whose shortest completion already exceeds max_len yields NO route, so
    # skipping it changes neither the route set nor the sorted output — it just stops the DFS wandering
    # a dense forward graph (the silent per-route hang at 22k, e.g. selectCidDbRecord15) and cuts the
    # per-edge function_reaches calls on dead branches.
    _radj: Dict[str, Set[str]] = {}
    for _src, _edges in fwd.items():
        for _e in _edges:
            _radj.setdefault(_e["callee"], set()).add(_src)
    _dist: Dict[str, int] = {child.file_stem: 0}
    _bq = deque([child.file_stem])
    while _bq:
        _c = _bq.popleft()
        for _p in _radj.get(_c, ()):
            if _p not in _dist:
                _dist[_p] = _dist[_c] + 1
                _bq.append(_p)
    _calls = [0]

    def enters_ok(stem: str, ftype: str, entered_fn: str, exit_fn: str) -> bool:
        # The function entered in `stem` must reach the function that calls out.
        return RCH.function_reaches(stem, ftype, entered_fn, exit_fn, blueprint_dir, reach_cache)

    def dfs(stem: str, ftype: str, entered_fn: str, trail: List[Dict[str, Any]],
            visiting: Set[str]) -> None:
        nonlocal capped
        if capped:
            return
        _calls[0] += 1
        if _calls[0] >= _ENUM_HARD_CAP:          # combinatorial path-space blowout → truncate, don't hang
            if _HANG_LOG.isEnabledFor(logging.INFO):
                _HANG_LOG.info("    _enumerate_routes HARD CAP %d hit (%s->%s, %d routes) — truncating",
                               _ENUM_HARD_CAP, parent.file_stem, child.file_stem, len(routes))
            capped = True
            return
        if _calls[0] % 100000 == 0 and _HANG_LOG.isEnabledFor(logging.INFO):
            _HANG_LOG.info("    _enumerate_routes churning: %d visits (%s->%s, %d routes so far)...",
                           _calls[0], parent.file_stem, child.file_stem, len(routes))
        if stem == child.file_stem:
            # We are in the child file; the entered function must reach the
            # target child function (no-op for ASM / unspecified child fn).
            if enters_ok(stem, ftype, entered_fn, child.function or ""):
                routes.append({"hops": list(trail),
                               "files": [trail[0]["from_file"]] + [h["to_file"] for h in trail]
                               if trail else [stem],
                               "length": len(trail)})
                if max_routes and len(routes) >= max_routes:
                    capped = True
            return
        if len(trail) >= max_len:
            return
        for e in fwd.get(stem, ()):
            callee = e["callee"]
            if callee in visiting:
                continue                          # simple paths only
            _d = _dist.get(callee)
            if _d is None or len(trail) + 1 + _d > max_len:   # can't reach child within max_len → skip
                continue                                       # (admissible: yields no ≤max_len route)
            # In `stem`, the function we entered at must reach the function that
            # makes THIS call (e["caller_fn"]).
            if not enters_ok(stem, ftype, entered_fn, e["caller_fn"]):
                continue
            callee_type = "cpp" if RCH._load_bp(callee, "cpp", blueprint_dir) else "asm"
            hop = {
                "from_file": stem, "from_function": e["caller_fn"],
                "to_file": callee, "to_function": e["callee_fn"],
                "line": e["line"], "conditions": e["conditions"],
                "_edge": e,   # carried so conditions can be attached lazily post-DFS (stripped below)
            }
            visiting.add(callee)
            dfs(callee, callee_type, e["callee_fn"], trail + [hop], visiting)
            visiting.discard(callee)
            if capped:
                return

    dfs(parent.file_stem, parent.file_type, parent.function or "", [], {parent.file_stem})
    routes.sort(key=lambda r: (r["length"], [h["to_file"] for h in r["hops"]]))
    # Lazy condition attach: reconstruct guards ONLY for the call sites that survived onto an emitted
    # route hop (find_routes' attach_conditions=False path left the closure's sites condition-less).
    # `_attach_call_site_conditions` is the SAME function walk_callers would have called eagerly, on the
    # SAME bridge site dict — so each hop's `conditions` are byte-identical (same objects, order, dedup).
    # A site shared by several routes is reconstructed once (guarded by "conditions" not in site); the
    # internal `_edge` key is stripped from every hop so the route shape is unchanged.
    from backward_traversal.runner.chainless_caller_walker import _attach_call_site_conditions
    _pc = pfl_cache if pfl_cache is not None else {}
    for r in routes:
        for h in r["hops"]:
            e = h.pop("_edge", None)
            if e is None:
                continue
            site = e.get("_site")
            if site is not None and "conditions" not in site:
                _attach_call_site_conditions(
                    [site], str(e.get("_src_stem") or ""), str(e.get("_src_type") or ""),
                    blueprint_dir, asm_dir, _pc,
                )
            h["conditions"] = list((site or {}).get("conditions") or [])
    return routes, capped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_routes(
    parent: Endpoint,
    child: Endpoint,
    *,
    blueprint_dir: Path,
    graph_file: Path,
    asm_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
    max_caller_depth: int = 0,
    max_routes: int = 200,
    max_len: int = 16,
    graph_payload: Optional[Dict[str, Any]] = None,
    reverse_adj: Optional[Dict[str, Any]] = None,
    node_type: Optional[Dict[str, str]] = None,
    _pfl_cache: Optional[Dict[str, Any]] = None,
    reach_cache: Optional[Dict[str, Any]] = None,
    forest_cache: Optional[Dict[str, Any]] = None,
    use_bfs: Optional[bool] = None,
    parent_distances: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Find every call route ``parent → child``.

    Returns ``{reachable, route_count, routes, unverified, stats}``.  Each route
    is ``{files: [...], hops: [{from_file, from_function, to_file, to_function,
    line, conditions}], length}``.

    Pass pre-loaded ``graph_payload``, ``reverse_adj`` and ``node_type`` to avoid
    redundant I/O when calling find_routes in a loop (e.g. per-setter).

    Pass ``_pfl_cache`` / ``reach_cache`` to share blueprint caches across calls.
    """
    from backward_traversal.runner.chainless_runner import _load_graph_payload
    from backward_traversal.runner.chainless_caller_walker import (
        build_reverse_adjacency, walk_callers,
    )

    parent = parent.norm()
    child = child.norm()
    blueprint_dir = Path(blueprint_dir)

    # Same-file route: pure intra-file function reachability (no cross-file walk).
    if parent.file_stem == child.file_stem:
        ok = RCH.function_reaches(parent.file_stem, parent.file_type,
                                  parent.function or "", child.function or "",
                                  blueprint_dir, {})
        route = ([{"hops": [], "files": [parent.file_stem], "length": 0,
                   "intra_file": True}] if ok else [])
        return {
            "reachable": ok, "route_count": len(route), "routes": route,
            "unverified": [],
            "stats": {"mode": "intra_file", "parent": asdict(parent), "child": asdict(child)},
        }

    if reverse_adj is None or node_type is None:
        if graph_payload is None:
            graph_payload = _load_graph_payload(Path(graph_file), job_id)
        reverse_adj, node_type = build_reverse_adjacency(graph_payload)

    # Restrict the walk to functions that actually reach the child function.
    anchor = (RCH.reaches_function_set(child.file_stem, child.function, blueprint_dir)
              if (child.file_type == "cpp" and child.function) else None)

    # OPT-IN deeper fix (VBT_BFS_FOREST=1): build the caller forest BREADTH-first so it is depth-CANONICAL
    # and can be bounded to the route-length cap WITHOUT changing the route set (proven boundable). The
    # default DFS walk_callers is NOT byte-identically boundable (a deep cycle-back steals shallow edges →
    # the forest depends on DFS order/depth) and explodes on hub callees (dw780000 22k: depth 194, OOM).
    # BFS fixes both (shallow edges never stolen ⇒ canonical + more complete) but CHANGES output vs the DFS
    # golden — hence opt-in. When on, bound depth to max_len+1 (deeper nodes can't form a <=max_len route).
    if use_bfs is None:
        use_bfs = _os.environ.get("VBT_BFS_FOREST") == "1"
    eff_depth = max_caller_depth
    if use_bfs and not eff_depth:
        eff_depth = (max_len + 1) if max_len else 0

    if parent_distances is not None and max_len:
        child_dist = parent_distances.get(child.file_stem)
        if child_dist is None or child_dist > max_len:
            return {
                "reachable": False,
                "route_count": 0,
                "routes": [],
                "routes_capped": False,
                "unverified": [],
                "stats": {
                    "mode": "cross_file",
                    "parent": asdict(parent),
                    "child": asdict(child),
                    "caller_forest_size": 0,
                    "verified_caller_files": 0,
                    "child_anchor_functions": sorted(anchor) if anchor else None,
                    "pruned_by_parent_distance": True,
                },
            }

    # The child's caller forest walk is the dominant per-query cost. Without a parent-distance prune,
    # it is reusable across parents. With the prune, it also depends on the parent stem because nodes
    # that cannot lie on any <=max_len parent→child route are skipped before bridge verification.
    # Byte-identical for emitted routes: every pruned node is outside the bounded parent cone, so it
    # cannot appear in route enumeration; eviction only recomputes.
    parent_prune_key = parent.file_stem if parent_distances is not None else None
    fkey = (child.file_stem, child.file_type, eff_depth, use_bfs, parent_prune_key,
            frozenset(anchor) if anchor is not None else None)
    cached_forest = forest_cache.get(fkey) if forest_cache is not None else None
    if cached_forest is not None:
        fwd, unverified, forest_size = cached_forest
    else:
        if _HANG_LOG.isEnabledFor(logging.INFO):
            _HANG_LOG.info("    find_routes: building caller forest for %s::%s (uncached, %s)...",
                           child.file_stem, child.function or "", "BFS" if use_bfs else "DFS")
        if use_bfs:
            from backward_traversal.runner.chainless_caller_walker import walk_callers_bfs
            caller_objs = walk_callers_bfs(
                child.file_stem, child.file_type, blueprint_dir, asm_dir, reverse_adj, node_type,
                max_caller_depth=eff_depth, anchor_fns=anchor,
                _pfl_cache=_pfl_cache, attach_conditions=False,
                parent_distances=parent_distances, max_route_len=max_len,
            )
        else:
            caller_objs = walk_callers(
                child.file_stem, child.file_type, blueprint_dir, asm_dir, reverse_adj, node_type,
                max_caller_depth=max_caller_depth, max_nodes=0, visited=set(), anchor_fns=anchor,
                _pfl_cache=_pfl_cache, attach_conditions=False,   # defer guard reconstruction to emitted hops
            )
        if _HANG_LOG.isEnabledFor(logging.INFO):
            _HANG_LOG.info("    find_routes: walk done (%d callers); building forward edges...",
                           len(caller_objs))
        # _build_forward_edges only reads stem/callee_stem/call_sites/cycle. asdict() deep-COPIES
        # every node incl. its call_sites+conditions — a large transient spike (doubled peak RAM)
        # for a widely-shared child whose closure is a big fraction of a 22k corpus. A shallow dict
        # that REFERENCES call_sites is byte-identical (_build_forward_edges only reads, never mutates)
        # and avoids the copy.
        caller_nodes = [{"stem": c.stem, "callee_stem": c.callee_stem,
                         "call_sites": c.call_sites, "cycle": c.cycle,
                         "file_type": c.file_type} for c in caller_objs]
        fwd, unverified = _build_forward_edges(caller_nodes)
        forest_size = len(caller_nodes)
        if _HANG_LOG.isEnabledFor(logging.INFO):
            _HANG_LOG.info("    find_routes: forward edges built (%d source stems)", len(fwd))
        if forest_cache is not None:
            if len(forest_cache) >= 512:        # trace-scoped bound; FIFO-evict oldest forest
                forest_cache.pop(next(iter(forest_cache)))
            forest_cache[fkey] = (fwd, unverified, forest_size)

    if _HANG_LOG.isEnabledFor(logging.INFO):
        _HANG_LOG.info("    find_routes: enumerating routes %s->%s (max_routes=%d, max_len=%d)...",
                       parent.file_stem, child.file_stem, max_routes, max_len)
    routes, capped = _enumerate_routes(
        parent, child, fwd, blueprint_dir, max_routes=max_routes, max_len=max_len,
        reach_cache=reach_cache, asm_dir=asm_dir, pfl_cache=_pfl_cache,
    )

    return {
        "reachable": bool(routes),
        "route_count": len(routes),
        "routes": routes,
        "routes_capped": capped,
        "unverified": unverified,
        "stats": {
            "mode": "cross_file",
            "parent": asdict(parent),
            "child": asdict(child),
            "caller_forest_size": forest_size,
            "verified_caller_files": len(fwd),
            "child_anchor_functions": sorted(anchor) if anchor else None,
        },
    }


def is_reachable_bounded(
    parent: Endpoint,
    child: Endpoint,
    *,
    blueprint_dir: Path,
    asm_dir: Optional[Path] = None,
    max_len: int = 16,
    reverse_adj: Dict[str, List[Tuple[str, str]]],
    node_type: Dict[str, str],
    parent_distances: Optional[Dict[str, int]] = None,
    _pfl_cache: Optional[Dict[str, Any]] = None,
    reach_cache: Optional[Dict[str, Any]] = None,
) -> bool:
    """Fast yes/no reachability for VBT dep-var off-chain filtering.

    This is the bounded, parent-targeted version of ``find_routes(..., max_routes=1)``:
    it walks callers of ``child`` breadth-first, prunes with the parent forward-distance
    cone, and returns as soon as a verified edge reaches ``parent``. It intentionally
    does not build the complete caller forest or route payload.
    """
    from backward_traversal.runner import chainless_caller_walker as CW

    parent = parent.norm()
    child = child.norm()
    blueprint_dir = Path(blueprint_dir)
    if parent.file_stem == child.file_stem:
        return RCH.function_reaches(parent.file_stem, parent.file_type,
                                    parent.function or "", child.function or "",
                                    blueprint_dir, reach_cache or {})
    if parent_distances is not None and max_len:
        child_dist = parent_distances.get(child.file_stem)
        if child_dist is None or child_dist > max_len:
            return False

    anchor = (RCH.reaches_function_set(child.file_stem, child.function, blueprint_dir)
              if (child.file_type == "cpp" and child.function) else None)
    if _pfl_cache is None:
        _pfl_cache = {}
    if reach_cache is None:
        reach_cache = {}

    q = deque([(child.file_stem, child.file_type, anchor, 0)])
    enqueued: Set[str] = {child.file_stem.upper()}
    visited_edges: Set[Tuple[str, str]] = set()
    checked = 0
    try:
        heartbeat = max(1000, int(_os.environ.get("VBT_REACHABLE_BOUNDED_HEARTBEAT", "50000") or "50000"))
    except ValueError:
        heartbeat = 50000

    if _HANG_LOG.isEnabledFor(logging.INFO):
        _HANG_LOG.info("    reachable_bounded: %s::%s -> %s::%s (max_len=%d)...",
                       parent.file_stem, parent.function or "", child.file_stem,
                       child.function or "", max_len)

    while q:
        callee_stem, callee_type, callee_anchor, depth = q.popleft()
        if max_len and depth >= max_len:
            continue

        en_memo = _pfl_cache.setdefault("_entry_names", {})
        en_key = (callee_stem.upper(), callee_type)
        entry_names = en_memo.get(en_key)
        if entry_names is None:
            entry_names = CW._entry_names(callee_stem, callee_type, blueprint_dir)
            en_memo[en_key] = entry_names
        keys = {callee_stem.upper()} | entry_names
        candidates: Dict[str, str] = {}
        for key in keys:
            for src, instr in reverse_adj.get(key, []):
                candidates.setdefault(src, instr)

        for source_stem, instr in sorted(candidates.items()):
            edge_depth = depth + 1
            if max_len and edge_depth > max_len:
                continue
            if parent_distances is not None and max_len:
                parent_depth = parent_distances.get(source_stem)
                if parent_depth is None or parent_depth + edge_depth > max_len:
                    continue
            pair = (source_stem.upper(), callee_stem.upper())
            if pair in visited_edges:
                continue
            visited_edges.add(pair)
            checked += 1
            if checked % heartbeat == 0 and _HANG_LOG.isEnabledFor(logging.INFO):
                _HANG_LOG.info("    reachable_bounded churning: %d verified-edge candidates "
                               "(at %s <- %s, depth %d)...",
                               checked, callee_stem, source_stem, edge_depth)

            source_type = node_type.get(source_stem) or CW._guess_type(source_stem, blueprint_dir)
            bridge = CW._bridge_for(source_type, callee_type)
            if bridge is None:
                continue

            bridge_cache = _pfl_cache.setdefault("_bridge_sites", {})
            bridge_key = (source_stem, callee_stem, source_type, callee_type, False)
            cached_sites = bridge_cache.get(bridge_key)
            if cached_sites is not None:
                sites = copy.deepcopy(cached_sites)
            else:
                try:
                    sites = bridge(source_stem, callee_stem, blueprint_dir, asm_dir) or []
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "bridge %s->%s failed for %s: %s",
                        source_type, callee_type, source_stem, exc)
                    sites = []
                if len(bridge_cache) >= 20000:
                    bridge_cache.pop(next(iter(bridge_cache)))
                bridge_cache[bridge_key] = copy.deepcopy(sites)
            if not sites:
                continue
            if callee_anchor is not None and callee_type == "cpp":
                sites, _ = RCH.filter_sites_by_reach(sites, callee_anchor)
                if not sites:
                    continue

            if source_stem == parent.file_stem:
                if parent.file_type != "cpp" or not parent.function:
                    return True
                for site in sites:
                    caller_fn = RCH.site_caller_fn(site)
                    if RCH.function_reaches(
                        source_stem, source_type, parent.function or "", caller_fn,
                        blueprint_dir, reach_cache,
                    ):
                        return True

            source_key = source_stem.upper()
            if source_key not in enqueued:
                child_anchor = RCH.child_anchor_functions(source_stem, source_type, sites, blueprint_dir)
                enqueued.add(source_key)
                q.append((source_stem, source_type, child_anchor, edge_depth))

    return False
