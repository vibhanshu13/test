"""GAP B — recursive cross-language caller discovery for the chainless tracer.

With no ``selected_chain``, "upstream" is discovered by recursively finding all
callers of the file containing a setter, across all four language edges.  The
``file_call_graph`` gives only *candidate* callers (its edge ``target`` is a
resolved stem only when ``is_internal``); final verification + entry-point→stem
resolution is delegated to the four ``cross_file_bridge_backward`` functions.

Also provides ``find_same_file_callers`` (the verified C++ same-file gap: the
bridges are cross-file only) and surfaces register-indirect / unresolved targets
as ``indirection.unresolved`` rather than dropping them.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from backward_traversal.models.chainless_models import CallerNode, Indirection

logger = logging.getLogger(__name__)
# Hang-locating heartbeat goes under the "vbt" root so the single ``progress`` knob (which sets the
# "vbt" logger's level) controls it too — otherwise a runaway caller-walk would stay silent.
_HANG_LOG = logging.getLogger("vbt.callerwalk")


# ---------------------------------------------------------------------------
# Reverse adjacency over the file call graph
# ---------------------------------------------------------------------------

def build_reverse_adjacency(
    graph_payload: Dict[str, Any],
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, str]]:
    """Return ``(reverse_adj, node_type)``.

    ``reverse_adj`` maps an UPPER target key (the edge ``target`` stem AND each
    ``call_site.target_module``) → list of ``(source_stem, instruction)``.
    ``node_type`` maps stem → "asm"|"cpp" (best-effort).
    """
    node_type: Dict[str, str] = {}
    for n in (graph_payload.get("nodes") or []):
        nid = str(n.get("id") or "").strip()
        nt = str(n.get("type") or "").lower()
        if nid and nt in ("asm", "cpp"):
            node_type[nid] = nt

    reverse: Dict[str, List[Tuple[str, str]]] = {}

    def _add(key: str, src: str, instr: str) -> None:
        k = str(key or "").upper().strip()
        if not k or not src:
            return
        reverse.setdefault(k, []).append((src, instr))

    for e in (graph_payload.get("edges") or []):
        src = str(e.get("source") or "").strip()
        if not src:
            continue
        instrs = e.get("instructions") or []
        instr = str(instrs[0]) if instrs else str(e.get("instruction") or "")
        _add(e.get("target", ""), src, instr)
        for cs in (e.get("call_sites") or []):
            _add(cs.get("target_module", ""), src, str(cs.get("instruction") or instr))
    return reverse, node_type


def _entry_names(callee_stem: str, callee_type: str, blueprint_dir: Path) -> Set[str]:
    """Entry-point / module names a callee may be reached by (for adjacency keys)."""
    names: Set[str] = {callee_stem.upper()}
    try:
        from backward_traversal.utils.blueprint_utils import (
            load_json,
            resolve_asm_blueprint,
            resolve_cpp_blueprint,
        )
    except Exception:  # pragma: no cover
        return names
    bp_path = (
        resolve_cpp_blueprint(callee_stem, blueprint_dir)
        if callee_type == "cpp"
        else resolve_asm_blueprint(callee_stem, blueprint_dir)
    )
    if not bp_path:
        return names
    try:
        bp = load_json(bp_path)
    except Exception:
        return names
    for k in (bp.get("asm_entry_point_map") or {}).keys():
        names.add(str(k).upper())
    for sym in (bp.get("symbols", {}) or {}).get("global_symbols", []) or []:
        if isinstance(sym, dict):
            st = str(sym.get("type") or "").lower()
            if st in ("entry", "csect", "public"):
                nm = str(sym.get("name") or "")
                if nm:
                    names.add(nm.upper())
        elif isinstance(sym, str) and sym:
            # Flat name list (no type metadata) — include as a candidate entry
            # name; the bridge verifies the actual call, so false keys are harmless.
            names.add(sym.upper())
    return names


# ---------------------------------------------------------------------------
# Bridge dispatch
# ---------------------------------------------------------------------------

def _bridge_for(source_type: str, callee_type: str) -> Optional[Callable]:
    from backward_traversal.bridge import cross_file_bridge_backward as B
    key = (source_type, callee_type)
    if key == ("asm", "asm"):
        return B.find_asm_callers
    if key == ("asm", "cpp"):
        return B.find_asm_to_cpp_callers
    if key == ("cpp", "cpp"):
        return B.find_cpp_to_cpp_callers
    if key == ("cpp", "asm"):
        return B.find_cpp_to_asm_callers
    return None


def find_same_file_callers(
    function_or_block: str, stem: str, file_type: str, blueprint_dir: Path
) -> List[Dict[str, Any]]:
    """Intra-file callers of a function/block (closes the C++ same-file gap).

    Scans the blueprint's OWN ``call_graph.edges`` for sources that call
    ``function_or_block`` within the same file.
    """
    try:
        from backward_traversal.utils.blueprint_utils import (
            load_json,
            resolve_asm_blueprint,
            resolve_cpp_blueprint,
        )
    except Exception:  # pragma: no cover
        return []
    bp_path = (
        resolve_cpp_blueprint(stem, blueprint_dir)
        if file_type == "cpp"
        else resolve_asm_blueprint(stem, blueprint_dir)
    )
    if not bp_path:
        return []
    try:
        bp = load_json(bp_path)
    except Exception:
        return []
    target = str(function_or_block or "").upper()
    out: List[Dict[str, Any]] = []
    for e in (bp.get("call_graph", {}) or {}).get("edges", []) or []:
        if str(e.get("target") or "").upper() != target:
            continue
        src = str(e.get("source") or "")
        if src and src.upper() != target:  # ignore pure self-recursion noise
            out.append(
                {
                    "caller_function": src,
                    "callee_function": e.get("target"),
                    "line": e.get("line"),
                    "instruction": e.get("instruction"),
                    "call_type": e.get("call_type"),
                    "same_file": True,
                }
            )
    # Attach the guard condition under which each same-file call fires (ISSUE-2).
    _attach_call_site_conditions(out, stem, file_type, blueprint_dir, None, {})
    return out


def _attach_call_site_conditions(
    sites: List[Dict[str, Any]], source_stem: str, source_type: str,
    blueprint_dir: Path, asm_dir: Optional[Path], pfl_cache: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Attach the guard condition under which each call fires (ISSUE-2).

    ASM: the TRIGGER/BRANCH guards before the call line in its routine.
    C++: the call edge's enclosing ``conditional_context`` (if/switch chain) with
    switch-discriminant reconstruction — the same mechanism ``build_lineage``
    uses, so a caller hop carries *why* it reaches the callee.  Each site gets a
    ``conditions`` list (empty when the call is unconditional).  Fully generic."""
    if not sites:
        return sites
    try:
        if source_type == "asm":
            from backward_traversal.runner.backward_only_runner import (
                _extract_block_conditions_before_line,
            )
            from backward_traversal.utils.blueprint_utils import load_json, resolve_asm_blueprint
            bp_path = resolve_asm_blueprint(source_stem, blueprint_dir)
            bp = None
            if bp_path:
                try:
                    bp = load_json(bp_path)
                except Exception:
                    bp = None
            for s in sites:
                routine = str(s.get("routine") or s.get("caller_function") or "")
                line = int(s.get("line") or 0)
                conds: List[str] = []
                if bp is not None and routine and line:
                    try:
                        raw = _extract_block_conditions_before_line(bp, routine, line, str(bp_path))
                        # _extract_block_conditions_before_line returns dicts, which
                        # are not hashable — dedup by string representation instead.
                        seen_k: set = set()
                        conds = []
                        for c in raw:
                            k = str(c)
                            if k not in seen_k:
                                seen_k.add(k)
                                conds.append(c)
                    except Exception:
                        conds = []
                s.setdefault("conditions", conds)
        else:
            from backward_traversal.runner.chainless_lineage_tree import (
                _conditions_at, _load_pfl, _reconstruct_switch_conditions,
            )
            entry = _load_pfl(source_stem, blueprint_dir, pfl_cache)
            for s in sites:
                scope = str(s.get("cpp_function") or s.get("caller_function") or s.get("routine") or "")
                line = int(s.get("line") or 0)
                conds = []
                if entry and scope:
                    try:
                        conds = _conditions_at(source_stem, scope, line, blueprint_dir, pfl_cache)
                        conds = _reconstruct_switch_conditions(conds, entry, scope, line)
                    except Exception:
                        conds = []
                s.setdefault("conditions", conds)
    except Exception:  # pragma: no cover - never let condition enrichment break the walk
        pass
    return sites


# ---------------------------------------------------------------------------
# Recursive caller walk
# ---------------------------------------------------------------------------

def walk_callers(
    callee_stem: str,
    callee_type: str,
    blueprint_dir: Path,
    asm_dir: Optional[Path],
    reverse_adj: Dict[str, List[Tuple[str, str]]],
    node_type: Dict[str, str],
    *,
    max_caller_depth: int = 0,
    max_nodes: int = 0,
    visited: Optional[Set[Tuple[str, str]]] = None,
    depth: int = 0,
    _count: Optional[Dict[str, int]] = None,
    _pfl_cache: Optional[Dict[str, Any]] = None,
    anchor_fns: Optional[Set[str]] = None,
    attach_conditions: bool = True,
) -> List[CallerNode]:
    """Return verified callers of ``callee_stem``, recursing into callers-of-callers.

    Caps of 0/None = unlimited.  Cycles handled via a ``(caller, callee)``
    visited set (always on).  Unresolvable candidates surface as nodes with
    ``indirection.unresolved=True`` — never silently dropped.

    ``anchor_fns`` (UPPER function names in ``callee_stem`` that reach the setter)
    enables **function-level reachability filtering**: a caller is kept only when
    at least one of its call sites targets a function in the anchor set, i.e. the
    call actually lands on a path that reaches the setter.  ``None`` ⇒ no filter
    (ASM callee, missing call graph, or unresolved enclosing function — see
    ``chainless_reachability``).  This prunes callers that enter ``callee_stem``
    through an unrelated entry point and never execute the setter.
    """
    blueprint_dir = Path(blueprint_dir)
    if visited is None:
        visited = set()
    if _count is None:
        _count = {"n": 0}
    if _pfl_cache is None:
        _pfl_cache = {}
    if max_caller_depth and depth >= max_caller_depth:
        return []
    if max_nodes and _count["n"] >= max_nodes:
        return []

    # Candidate caller stems: by callee stem AND by callee entry-point names. `_entry_names` is a pure
    # function of (callee_stem, callee_type) but was re-scanning the callee blueprint's symbol tables on
    # EVERY recursion frame — the dominant uncached per-node CPU at 22k scale. Memoize trace-scoped in
    # `_pfl_cache` (byte-identical: same set every time; `keys` below is a fresh copy, never mutates it).
    _en_memo = _pfl_cache.setdefault("_entry_names", {})
    _en_key = (callee_stem.upper(), callee_type)
    _en = _en_memo.get(_en_key)
    if _en is None:
        _en = _entry_names(callee_stem, callee_type, blueprint_dir)
        _en_memo[_en_key] = _en
    keys = {callee_stem.upper()} | _en
    candidates: Dict[str, str] = {}  # source_stem -> instruction
    for k in keys:
        for src, instr in reverse_adj.get(k, []):
            candidates.setdefault(src, instr)

    out: List[CallerNode] = []
    for source_stem, instr in sorted(candidates.items()):
        pair = (source_stem.upper(), callee_stem.upper())
        if pair in visited:
            out.append(
                CallerNode(
                    stem=source_stem,
                    file_type=node_type.get(source_stem, "asm"),  # best-effort
                    callee_stem=callee_stem,
                    edge_kind=f"{node_type.get(source_stem,'?')}->{callee_type}",
                    depth=depth + 1,
                    cycle=True,
                )
            )
            continue
        visited.add(pair)
        source_type = node_type.get(source_stem) or _guess_type(source_stem, blueprint_dir)
        bridge = _bridge_for(source_type, callee_type)
        if bridge is None:
            continue
        # SCALE: the bridge call + per-site condition attach (the two dominant per-node costs in
        # the closure walk) are a PURE function of (source, callee, types) + the immutable corpus,
        # but the same (caller→callee) edge recurs across many walks (each setter endpoint's forest
        # re-evaluates shared ancestors). Memoize the attached sites in the trace-scoped _pfl_cache
        # and hand back a private deepcopy each time so downstream mutation can't corrupt the entry.
        # Byte-identical: cached attached-sites == freshly computed; copies keep every caller's list
        # independent (the bridge already returns a fresh list per call today).
        # attach_conditions=False (the find_routes path) DEFERS the per-site guard reconstruction
        # — the dominant per-edge cost — to AFTER route enumeration, where conditions are attached
        # only for the (source→callee@line) sites on an emitted route hop (a tiny fraction of the
        # closure). The cache key includes attach_conditions so a sites-WITH-conditions entry is
        # never handed to a deferred caller and vice-versa; both are byte-identical for their consumer.
        _bc = _pfl_cache.setdefault("_bridge_sites", {})
        _bkey = (source_stem, callee_stem, source_type, callee_type, attach_conditions)
        _cached_sites = _bc.get(_bkey)
        if _cached_sites is not None:
            sites = copy.deepcopy(_cached_sites)
        else:
            try:
                sites = bridge(source_stem, callee_stem, blueprint_dir, asm_dir) or []
            except Exception as exc:
                logger.debug("bridge %s->%s failed for %s: %s", source_type, callee_type, source_stem, exc)
                sites = []
            # Attach the guard condition under which each call fires (why the caller reaches the
            # callee — and therefore why the setter can run). Deferred to post-enumeration when
            # attach_conditions is False (find_routes attaches only the emitted-route sites instead).
            if attach_conditions:
                _attach_call_site_conditions(sites, source_stem, source_type, blueprint_dir, asm_dir, _pfl_cache)
            if len(_bc) >= 20000:                 # trace-scoped bound; FIFO-evict oldest pair
                _bc.pop(next(iter(_bc)))
            _bc[_bkey] = copy.deepcopy(sites)
        if not sites:
            # Candidate that the bridge could not verify — surface as unresolved
            # (e.g. register-indirect / module-name-only edge), never drop.
            out.append(
                CallerNode(
                    stem=source_stem,
                    file_type=source_type,
                    callee_stem=callee_stem,
                    edge_kind=f"{source_type}->{callee_type}",
                    depth=depth + 1,
                    indirection=Indirection(is_indirect=True, unresolved=True, indirect_via=instr or None),
                )
            )
            continue
        # Function-level reachability prune: keep only call sites that target a
        # function which actually reaches the setter (anchor_fns).  If every site
        # provably misses the setter, the whole caller is dropped — it enters the
        # callee through an unrelated entry point.  Only applies for a C++ callee
        # with a known anchor set (see chainless_reachability for soundness).
        if anchor_fns is not None and callee_type == "cpp":
            from backward_traversal.runner.chainless_reachability import filter_sites_by_reach
            sites, n_pruned = filter_sites_by_reach(sites, anchor_fns)
            if n_pruned:
                _count["pruned"] = _count.get("pruned", 0) + n_pruned
            if not sites:
                _count["pruned_callers"] = _count.get("pruned_callers", 0) + 1
                continue
        _count["n"] += 1
        # Hang-locating heartbeat: the caller-walk is unbounded by default (max_nodes=0). A runaway
        # walk emits a line every 5k nodes so a churning walk is visible + attributable instead of a
        # silent multi-hour hang. Fires ONLY past 5k ⇒ normal walks stay silent (no noise, byte-identical).
        if _count["n"] >= _count.get("_hb", 5000) and _HANG_LOG.isEnabledFor(logging.INFO):
            _HANG_LOG.info("    walk_callers churning: %d nodes (callee %s, depth %d, %d pruned)...",
                           _count["n"], callee_stem, depth, _count.get("pruned_callers", 0))
            _count["_hb"] = _count["n"] + 5000
        node = CallerNode(
            stem=source_stem,
            file_type=source_type,
            callee_stem=callee_stem,
            edge_kind=f"{source_type}->{callee_type}",
            depth=depth + 1,
            call_sites=sites,
        )
        # Anchor set for the caller, one hop up: the functions in this caller that
        # reach the call sites we kept.  Threaded into the recursion so the prune
        # is function-accurate at every hop.
        from backward_traversal.runner.chainless_reachability import child_anchor_functions
        child_anchor = child_anchor_functions(source_stem, source_type, sites, blueprint_dir)
        # Recurse into callers-of-callers.
        children = walk_callers(
            source_stem, source_type, blueprint_dir, asm_dir, reverse_adj, node_type,
            max_caller_depth=max_caller_depth, max_nodes=max_nodes,
            visited=visited, depth=depth + 1, _count=_count, _pfl_cache=_pfl_cache,
            anchor_fns=child_anchor, attach_conditions=attach_conditions,
        )
        node.child_keys = [f"{c.stem}->{c.callee_stem}" for c in children]
        out.append(node)
        out.extend(children)
    return out


def walk_callers_bfs(
    callee_stem: str,
    callee_type: str,
    blueprint_dir: Path,
    asm_dir: Optional[Path],
    reverse_adj: Dict[str, List[Tuple[str, str]]],
    node_type: Dict[str, str],
    *,
    max_caller_depth: int = 0,
    anchor_fns: Optional[Set[str]] = None,
    attach_conditions: bool = True,
    _pfl_cache: Optional[Dict[str, Any]] = None,
    parent_distances: Optional[Dict[str, int]] = None,
    max_route_len: int = 0,
) -> List[CallerNode]:
    """Breadth-first caller walk — a depth-CANONICAL alternative to ``walk_callers`` (DFS).

    The DFS walk's forest is not a clean function of the node set: with a single GLOBAL ``visited``
    set, a *deep* cycle-back-edge can mark a ``(caller, callee)`` pair visited BEFORE the shallow
    direct edge is processed, demoting a real shallow edge to a no-site ``cycle`` node — so the forest
    (and the routes derived from it) depend on DFS order and on how deep the walk ran. That makes the
    walk impossible to bound byte-identically (proven: bounding changes the route set) and silently
    drops valid routes for hub callees (e.g. dw780000 at 22k).

    BFS fixes this: every ``(caller, callee)`` edge is claimed at its SHORTEST depth, so a deep cycle
    can never steal a shallow edge. Each callee subtree is expanded exactly once, at its shortest depth,
    with that edge's anchor — the SAME "first-encounter wins" semantics the DFS already has effectively
    (a callee re-recursed under a second parent finds all its caller pairs already visited ⇒ no new
    productive children), just resolved shallow-first instead of deepest-sibling-first. Consequences:
      * the forest is canonical ⇒ truncating at ``max_caller_depth`` is well-defined and byte-identical
        w.r.t. the BFS forest (deeper layers cannot affect any node within the bound), and
      * shallow edges are never stolen ⇒ the route-loss for hub callees is fixed (MORE routes).
    Returns the same ``List[CallerNode]`` shape as ``walk_callers`` (``_build_forward_edges`` consumes
    it unchanged). Per-edge processing (bridge + sites cache + anchor filter + lazy condition attach)
    is byte-identical to the DFS path. This DIFFERS in output from the DFS golden — gate behind a flag.

    ``parent_distances`` is an optional admissible lower-bound map from the route parent to each file
    over the metadata forward graph. When present, candidates whose
    ``parent_dist + reverse_depth > max_route_len`` are skipped before bridge verification because they
    cannot contribute to any emitted bounded route.
    """
    from collections import deque
    blueprint_dir = Path(blueprint_dir)
    if _pfl_cache is None:
        _pfl_cache = {}
    visited: Set[Tuple[str, str]] = set()     # (caller, callee) pairs whose edge is already productive
    enqueued: Set[str] = set()                # callee stems already queued — shortest-depth expansion wins
    out: List[CallerNode] = []
    q: "deque" = deque()
    q.append((callee_stem, callee_type, anchor_fns, 0))
    enqueued.add(callee_stem.upper())
    while q:
        cs, ct, anchor, depth = q.popleft()
        if max_caller_depth and depth >= max_caller_depth:
            continue
        _en_memo = _pfl_cache.setdefault("_entry_names", {})
        _en_key = (cs.upper(), ct)
        _en = _en_memo.get(_en_key)
        if _en is None:
            _en = _entry_names(cs, ct, blueprint_dir)
            _en_memo[_en_key] = _en
        keys = {cs.upper()} | _en
        candidates: Dict[str, str] = {}
        for k in keys:
            for src, instr in reverse_adj.get(k, []):
                candidates.setdefault(src, instr)
        for source_stem, instr in sorted(candidates.items()):
            edge_depth = depth + 1
            if parent_distances is not None and max_route_len:
                parent_depth = parent_distances.get(source_stem)
                if parent_depth is None or parent_depth + edge_depth > max_route_len:
                    continue
            pair = (source_stem.upper(), cs.upper())
            if pair in visited:
                out.append(CallerNode(
                    stem=source_stem, file_type=node_type.get(source_stem, "asm"),
                    callee_stem=cs, edge_kind=f"{node_type.get(source_stem,'?')}->{ct}",
                    depth=edge_depth, cycle=True))
                continue
            visited.add(pair)
            source_type = node_type.get(source_stem) or _guess_type(source_stem, blueprint_dir)
            bridge = _bridge_for(source_type, ct)
            if bridge is None:
                continue
            _bc = _pfl_cache.setdefault("_bridge_sites", {})
            _bkey = (source_stem, cs, source_type, ct, attach_conditions)
            _cached_sites = _bc.get(_bkey)
            if _cached_sites is not None:
                sites = copy.deepcopy(_cached_sites)
            else:
                try:
                    sites = bridge(source_stem, cs, blueprint_dir, asm_dir) or []
                except Exception as exc:
                    logger.debug("bridge %s->%s failed for %s: %s", source_type, ct, source_stem, exc)
                    sites = []
                if attach_conditions:
                    _attach_call_site_conditions(sites, source_stem, source_type, blueprint_dir, asm_dir, _pfl_cache)
                if len(_bc) >= 20000:
                    _bc.pop(next(iter(_bc)))
                _bc[_bkey] = copy.deepcopy(sites)
            if not sites:
                out.append(CallerNode(
                    stem=source_stem, file_type=source_type, callee_stem=cs,
                    edge_kind=f"{source_type}->{ct}", depth=edge_depth,
                    indirection=Indirection(is_indirect=True, unresolved=True, indirect_via=instr or None)))
                continue
            if anchor is not None and ct == "cpp":
                from backward_traversal.runner.chainless_reachability import filter_sites_by_reach
                sites, _ = filter_sites_by_reach(sites, anchor)
                if not sites:
                    continue
            out.append(CallerNode(
                stem=source_stem, file_type=source_type, callee_stem=cs,
                edge_kind=f"{source_type}->{ct}", depth=edge_depth, call_sites=sites))
            if source_stem.upper() not in enqueued:
                from backward_traversal.runner.chainless_reachability import child_anchor_functions
                child_anchor = child_anchor_functions(source_stem, source_type, sites, blueprint_dir)
                enqueued.add(source_stem.upper())
                q.append((source_stem, source_type, child_anchor, edge_depth))
    return out


def _guess_type(stem: str, blueprint_dir: Path) -> str:
    try:
        from backward_traversal.utils.blueprint_utils import resolve_cpp_blueprint
        return "cpp" if resolve_cpp_blueprint(stem, blueprint_dir) else "asm"
    except Exception:  # pragma: no cover
        return "asm"
