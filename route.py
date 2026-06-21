"""Reachability / route enumeration.

Root-variable reachability (which setters are reachable from the chain tail, and
the routes to them, with per-hop guards) reuses ``route_finder.find_routes``
READ-ONLY (verdict: USE-AS-IS for the unbounded case).

The dep-var ``before_line`` bound (SPEC §6 — a setter must be reachable inside a
chain file *before* that file's descend call site) is implemented HERE, in our own
layer (rule 4: we don't modify the existing reachability in place). It reuses the
read-only intra-file forward adjacency from ``chainless_reachability`` and applies
the line cap during the BFS.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from backward_traversal.route_finder import Endpoint, find_routes, is_reachable_bounded
from backward_traversal.runner import chainless_reachability as RCH
from vbt.reach.fn_attr import reattribute_edges, is_globalish, line_to_function


class RouteEngine:
    """Caches the graph payload + blueprint reach caches across many queries."""

    # OOM guards for the trace-scoped memos (fresh per trace, but a 22k-corpus trace touches far
    # more distinct stems/route-pairs than the small corpus). Each memo is a pure function of its
    # key, so FIFO eviction is byte-identical — an evicted entry is simply recomputed on next miss.
    _ROUTES_MEMO_MAX = 8192     # each value = a full routes result (hops+conditions)
    _CFG_CACHE_MAX = 512        # each value = a cfg_extract fn-list (heaviest per-entry)
    _EDGE_CACHE_MAX = 1024      # each value = re-attributed intra-file call edges

    def __init__(self, blueprint_dir: Path, graph_file: Path, asm_dir: Optional[Path] = None,
                 *, max_routes: int = 200, max_len: int = 16, job_id: Optional[str] = None):
        self.blueprint_dir = Path(blueprint_dir)
        self.graph_file = Path(graph_file)
        self.asm_dir = Path(asm_dir) if asm_dir else None
        # T1: DB-backed precompute key. When set, graph/payload loaders MAY consult
        # index.db; when None (CLI/default) the file/compute path runs unchanged. In T1
        # this only reaches the EXISTING index.db fallback in _load_graph_payload, which
        # fires solely when the JSON is absent — so threading it is behavior-neutral.
        self.job_id = job_id
        self.max_routes = max_routes        # cross-file routes cap (route_finder)
        self.max_len = max_len              # cross-file route hop-length cap (route_finder)
        self._graph_payload: Optional[Dict[str, Any]] = None
        self._reverse_adj: Optional[Dict[str, Any]] = None
        self._node_type: Optional[Dict[str, str]] = None
        self._pfl_cache: Dict[str, Any] = {}
        self._reach_cache: Dict[str, Any] = {}
        self._edge_cache: Dict[str, List[Tuple[str, str, int]]] = {}
        self._cfg_cache: Dict[str, List[Dict[str, Any]]] = {}   # cfg_extract per cpp stem
        # metadata-only forward file/file adjacency (built lazily, once) +
        # a cache of forward-reachable file sets keyed by endpoint stem.
        self._fwd_file_adj: Optional[Dict[str, Set[str]]] = None
        self._fwd_reach_cache: Dict[str, Set[str]] = {}
        self._fwd_dist_cache: Dict[Tuple[str, int], Dict[str, int]] = {}
        # Precomputed unified function graph reachability. This is the dep-var hot path:
        # many off-chain setters ask "can one of the chain hops reach this function?".
        # The DB artifact answers that from a per-parent BFS set instead of a per-child
        # caller walk.
        self._fn_graph_adj: Optional[Dict[Tuple[str, str], Set[Tuple[str, str, int, bool]]]] = None
        self._fn_graph_adj_loaded = False
        self._fn_reach_cache: Dict[Tuple[Tuple[str, str], int], Set[Tuple[str, str]]] = {}
        # name (UPPER) -> set of stems that EXPORT it (entry-point / module aliasing),
        # built once alongside the forward file adjacency and reused by the unified
        # route enumerator to resolve a cross-file call TARGET (an entry/module name)
        # back to the file stem(s) that own it.
        self._name_to_stems: Optional[Dict[str, Set[str]]] = None
        # §16 #2 (in-trace): memoize whole route results by (parent, child, max_routes). At
        # depth-3 many setters share a target function, so the cross-file caller-walk (the
        # dominant remaining per-setter cost) runs once per (tail, target), not per setter.
        self._routes_memo: Dict[Any, Any] = {}
        self._reachable_memo: Dict[Any, bool] = {}
        # Per-CHILD caller-forest cache (forward adjacency keyed by child+anchor, parent-independent
        # — see find_routes). The whole-route memo above only hits on identical (parent, child); the
        # dep-var phase checks ONE setter endpoint against many chain-hop parents, so this lets the
        # expensive child closure walk run once per setter instead of once per (hop, setter). Self-
        # bounded (FIFO) inside find_routes; byte-identical (cached forward edges == fresh).
        self._forest_cache: Dict[Any, Any] = {}

    def _stem_fn_nodes(self, stem: str) -> List[Tuple[str, str]]:
        """All fn-graph nodes belonging to ``stem`` (both call-graph sources and targets).

        Used to resolve a whole-file (function-less) C++ endpoint — a chain hop is a FILE,
        so "does it reach X" means "does ANY function in it reach X". Indexed once per
        corpus (keys + edge targets) and cached."""
        adj = self._load_fn_graph_adj()
        if adj is None:
            return []
        idx = getattr(self, "_stem_nodes_idx", None)
        if idx is None:
            idx = {}
            for (s, f), outs in adj.items():
                idx.setdefault(s, set()).add((s, f))
                for cs, cf, _ln, _cr in outs:
                    idx.setdefault(cs, set()).add((cs, cf))
            self._stem_nodes_idx = idx
        return list(idx.get(stem, ()))

    def _endpoint_fn_nodes(self, endpoint: Endpoint) -> List[Tuple[str, str]]:
        endpoint = endpoint.norm()
        if endpoint.file_type == "asm":
            return [(endpoint.file_stem, endpoint.file_stem)]
        fn = endpoint.function or ""
        if not fn:
            # whole-file C++ endpoint (no function) → every node of the stem.
            return self._stem_fn_nodes(endpoint.file_stem)
        out: List[Tuple[str, str]] = []

        def add(name: str) -> None:
            node = (endpoint.file_stem, name)
            if name and node not in out:
                out.append(node)

        add(fn)
        add(fn.split("::")[-1])
        return out

    def _load_fn_graph_adj(self) -> Optional[Dict[Tuple[str, str], Set[Tuple[str, str, int, bool]]]]:
        graph = getattr(self, "_cpp_fn_graph", None)
        if graph is not None:
            try:
                self._fn_graph_adj = graph.adjacency()
                self._fn_graph_adj_loaded = True
                return self._fn_graph_adj
            except Exception:
                pass
        if self._fn_graph_adj_loaded:
            return self._fn_graph_adj
        self._fn_graph_adj_loaded = True
        if not self.job_id:
            return None
        try:
            from vbt.precompute.graph_db import load_fn_graph_adj
            self._fn_graph_adj = load_fn_graph_adj(self.job_id)
        except Exception:
            self._fn_graph_adj = None
        return self._fn_graph_adj

    def _fn_reach_set(
        self, start: Tuple[str, str], *, max_depth: int
    ) -> Optional[Set[Tuple[str, str]]]:
        adj = self._load_fn_graph_adj()
        if adj is None:
            return None
        key = (start, int(max_depth or 0))
        cached = self._fn_reach_cache.get(key)
        if cached is not None:
            return cached
        from collections import deque
        seen: Set[Tuple[str, str]] = {start}
        q = deque([(start, 0)])
        while q:
            cur, depth = q.popleft()
            if max_depth and depth >= max_depth:
                continue
            for callee_stem, callee_fn, _line, _cross in adj.get(cur, ()):
                nxt = (callee_stem, callee_fn)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, depth + 1))
        if len(self._fn_reach_cache) >= 128:
            self._fn_reach_cache.pop(next(iter(self._fn_reach_cache)))
        self._fn_reach_cache[key] = seen
        return seen

    def _fn_graph_reachable(self, parent: Endpoint, child: Endpoint) -> Optional[bool]:
        parent_nodes = self._endpoint_fn_nodes(parent)
        child_nodes = set(self._endpoint_fn_nodes(child))
        if not parent_nodes or not child_nodes:
            return None
        # Force the load before returning false: if the artifact is absent, this fast
        # path is unknown and the verified caller-walk fallback remains authoritative.
        if self._load_fn_graph_adj() is None:
            return None
        for pnode in parent_nodes:
            seen = self._fn_reach_set(pnode, max_depth=self.max_len)
            if seen is not None and child_nodes & seen:
                return True
        return False

    def preload_fn_reachability(self, parents: Optional[List[Endpoint]] = None) -> bool:
        """Load the function graph and optionally warm per-parent reach sets."""
        if self._load_fn_graph_adj() is None:
            return False
        for parent in parents or []:
            for pnode in self._endpoint_fn_nodes(parent):
                self._fn_reach_set(pnode, max_depth=self.max_len)
        return True

    def _cfg_fns(self, stem: str) -> List[Dict[str, Any]]:
        """cfg_extract function list for a cpp stem (cached). Empty when asm_dir is
        unset or the file is absent — re-attribution then no-ops (sources unchanged)."""
        if self.asm_dir is None:
            return []
        c = self._cfg_cache.get(stem)
        if c is None:
            from vbt.cpp_frontend.wrapper import run_cfg_extract
            p = self.asm_dir / f"{stem}.cpp"
            try:
                c = run_cfg_extract(str(p)) if p.exists() else []
            except (FileNotFoundError, RuntimeError):
                c = []
            if len(self._cfg_cache) >= self._CFG_CACHE_MAX:   # 22k OOM guard; FIFO, recompute on miss
                self._cfg_cache.pop(next(iter(self._cfg_cache)))
            self._cfg_cache[stem] = c
        return c

    def cpp_call_edges(self, stem: str) -> List[Tuple[str, str, int]]:
        """Intra-file C++ call-graph edges ``(caller_fn, callee_fn, line)`` from the
        blueprint, cached per stem. READ-ONLY reuse of the parser's call_graph — we
        do NOT recompute it (rule 4). Used to enumerate intra-file inter-function
        call paths (the C6 case), which ``route_finder`` does not surface (it returns
        a same-file route with ``hops:[]``).

        The blueprint attributes every call inside a switch-truncated body to caller
        ``(global)``; we de-poison those sources by nearest-preceding cfg START
        (vbt/reach/fn_attr) so intra_call_paths / _hop_callee don't lose them."""
        cached = self._edge_cache.get(stem)
        if cached is not None:
            return cached
        bp = RCH._load_bp(stem, "cpp", self.blueprint_dir)
        raw = (((bp.get("call_graph") or {}).get("edges") or []) if bp else [])
        edges: List[Tuple[str, str, int]] = []
        for s, t, ln in reattribute_edges(raw, self._cfg_fns(stem)):
            if s and t and ln is not None:
                edges.append((s, t, int(ln)))
        if len(self._edge_cache) >= self._EDGE_CACHE_MAX:   # 22k OOM guard; FIFO, recompute on miss
            self._edge_cache.pop(next(iter(self._edge_cache)))
        self._edge_cache[stem] = edges
        return edges

    def _ensure_graph(self) -> None:
        if self._reverse_adj is not None and self._node_type is not None:
            return
        # B (cold-startup): prefer the precomputed reverse_adj+node_type blob (load-only) so the cold
        # trace neither rebuilds the adjacency (~O(edges+call_sites) every trace) NOR loads the raw
        # 80MB payload here — that is deferred to _payload(), reached only if graph_edges() is hit.
        # Byte-identical: load_reverse_adj rebuilds the SAME value-lists in the SAME order as
        # build_reverse_adjacency (verified by test_graph_db_order).
        if self.job_id:
            try:
                from vbt.precompute import db_artifacts as _DA
                if _DA.is_load_only():
                    from vbt.precompute.graph_db import load_reverse_adj
                    ra = load_reverse_adj(self.job_id)
                    if ra is not None:
                        self._reverse_adj, self._node_type = ra
                        return
            except Exception:
                pass
        # Fallback (CLI / blob absent): build from the raw payload (loaded lazily).
        from backward_traversal.runner.chainless_caller_walker import build_reverse_adjacency
        self._reverse_adj, self._node_type = build_reverse_adjacency(self._payload())

    def _payload(self) -> Dict[str, Any]:
        """The raw file_call_graph payload, loaded LAZILY and cached. In a fully-precomputed
        load-only trace the derived blobs (reverse_adj / forward_file_adj / name_to_stems) cover the
        hot path, so this ~80MB json.loads is skipped entirely unless graph_edges() (cpp_routes) or a
        missing-blob fallback forces it."""
        if self._graph_payload is None:
            payload = None
            if self.job_id:        # T12: prefer the DB blob so the trace reads no .json file
                try:
                    from vbt.precompute import db_artifacts as _DA
                    if _DA.is_load_only():
                        from vbt.precompute.graph_db import load_file_call_graph
                        payload = load_file_call_graph(self.job_id)
                except Exception:
                    payload = None
            if payload is None:
                from backward_traversal.runner.chainless_runner import _load_graph_payload
                payload = _load_graph_payload(self.graph_file, self.job_id)
            self._graph_payload = payload
        return self._graph_payload

    def routes(self, parent: Endpoint, child: Endpoint, *, max_routes: Optional[int] = None) -> Dict[str, Any]:
        """All forward routes parent → child with per-hop guard conditions.

        §16 #2 (in-trace): memoized by (parent, child, max_routes). The result is a deterministic
        function of those + the corpus, and many setters share a target function — so the
        cross-file caller-walk (the dominant remaining per-setter cost) is computed once per
        (tail, target) instead of once per setter. Byte-identical (cached == fresh)."""
        mr = max_routes if max_routes is not None else self.max_routes
        mkey = (parent.file_stem, parent.file_type, parent.function,
                child.file_stem, child.file_type, child.function, mr)
        cached = self._routes_memo.get(mkey)
        if cached is not None:
            return cached
        self._ensure_graph()
        parent_distances = None
        if self.job_id:
            try:
                parent_distances = self.forward_file_distances(parent, max_depth=self.max_len)
            except Exception:
                parent_distances = None
        # DB-backed VBT traces are large-corpus paths; the legacy DFS caller forest can walk
        # hundreds of reverse-call levels for broad ASM endpoints. Bounded BFS respects max_len
        # inside route_finder and keeps the route query finite. Explicit VBT_BFS_FOREST=0 still
        # preserves the old traversal for parity investigations.
        use_bfs = True if self.job_id and os.environ.get("VBT_BFS_FOREST") is None else None
        res = find_routes(
            parent, child,
            blueprint_dir=self.blueprint_dir, graph_file=self.graph_file, asm_dir=self.asm_dir,
            job_id=None, max_routes=mr, max_len=self.max_len,
            graph_payload=self._graph_payload, reverse_adj=self._reverse_adj,
            node_type=self._node_type, _pfl_cache=self._pfl_cache, reach_cache=self._reach_cache,
            forest_cache=self._forest_cache, use_bfs=use_bfs, parent_distances=parent_distances,
        )
        if len(self._routes_memo) >= self._ROUTES_MEMO_MAX:   # 22k OOM guard; FIFO, recompute on miss
            self._routes_memo.pop(next(iter(self._routes_memo)))
        self._routes_memo[mkey] = res
        return res

    def reachable(self, parent: Endpoint, child: Endpoint) -> bool:
        if self.job_id:
            parent = parent.norm()
            child = child.norm()
            rkey = (parent.file_stem, parent.file_type, parent.function,
                    child.file_stem, child.file_type, child.function, self.max_len)
            cached = self._reachable_memo.get(rkey)
            if cached is not None:
                return cached
            # AUTHORITATIVE when the fn-graph can represent BOTH endpoints (returns
            # True/False): the unified function graph now fans cross-file C++ edges out to
            # every owner of a called name (graph_db v2), so its forward closure matches the
            # bridge walk's function-level reachability. Trusting its FALSE is what prunes
            # the off-chain over-keep back to the function-accurate set. It returns None only
            # when an endpoint has no fn-graph node (e.g. a function-less cpp endpoint, or
            # the artifact is absent) — never a wrong negative.
            try:
                fg = self._fn_graph_reachable(parent, child)
            except Exception:
                fg = None
            if fg is not None:
                self._store_reachable(rkey, bool(fg))
                return bool(fg)
            # FALLBACK (fn-graph can't represent the endpoints): forward reachability over
            # the corpus file-call-graph, bounded to max_len and memoized PER SOURCE FILE —
            # a property of the corpus, independent of the API-supplied chain, so it stays
            # O(1) per query however the chain is generated. NO backward bounded walk (its
            # unbounded caller-cone was the dense-graph churn). File granularity is a sound
            # OVER-approximation here: it never drops a reachable setter.
            self._ensure_graph()
            ok = self._file_reachable(parent, child)
            self._store_reachable(rkey, ok)
            return ok
        return bool(self.routes(parent, child, max_routes=1).get("reachable"))

    def _store_reachable(self, rkey: tuple, ok: bool) -> None:
        if len(self._reachable_memo) >= self._ROUTES_MEMO_MAX:
            self._reachable_memo.pop(next(iter(self._reachable_memo)))
        self._reachable_memo[rkey] = ok

    def _file_reachable(self, parent: Endpoint, child: Endpoint) -> bool:
        """Authoritative file-granular reachability (parent/child already ``.norm()``ed).

        Same file ⇒ confirm at FUNCTION granularity (cheap, exact). Cross-file ⇒ child's
        file is within ``max_len`` forward file-hops of the parent's file. Both reads are
        memoized per source file, so this is O(1) after the first BFS per parent file."""
        if parent.file_stem == child.file_stem:
            try:
                return RCH.function_reaches(
                    parent.file_stem, parent.file_type, parent.function or "",
                    child.function or "", self.blueprint_dir, self._reach_cache)
            except Exception:
                return True
        try:
            dist = self.forward_file_distances(parent, max_depth=self.max_len)
        except Exception:
            return bool(self.routes(parent, child, max_routes=1).get("reachable"))
        return child.file_stem in dist

    # ----------------------------------------------------------------------- #
    # Metadata-only forward reachability (the scale prune)
    # ----------------------------------------------------------------------- #
    def _build_forward_file_adj(self) -> Dict[str, "Set[str]"]:
        """Build a forward file→file adjacency from the SAME graph payload that
        ``route_finder`` reaches over — purely from call-graph METADATA, no file
        opens beyond the per-stem entry-name expansion the caller-walk already uses.

        ``route_finder.find_routes`` discovers a route ``parent → … → child`` by
        walking the reverse adjacency of the child (``build_reverse_adjacency``,
        whose keys are every edge ``target`` AND every ``call_site.target_module``)
        and then verifying each cross-file hop with a bridge. The set of files that
        can appear on ANY such route is therefore a SUBSET of the files forward-
        reachable over the *unverified* edge graph derived from those very same
        keys. We build that unverified forward graph here and BFS it, so the result
        is a guaranteed SUPERSET of every file ``routes(tail, setter)`` would accept
        (bridge verification only ever DROPS edges, never adds one). Pruning the
        candidate-setter file set to this superset is a pure speedup.

        An edge target is a NODE NAME (a file stem OR an entry-point / module name).
        We map each name back to the file stem(s) that EXPORT it (same expansion
        ``walk_callers`` uses via ``_entry_names``: the stem itself + its blueprint's
        ``asm_entry_point_map`` keys + entry/csect/public global symbols). A target
        name with no in-corpus owner is an out-of-corpus call (dead end) and is
        simply absent from the forward graph.
        """
        payload = self._payload()
        # name (UPPER) -> exporting stems (the shared entry-name expansion, cached).
        name_to_stems = self.name_to_stems()
        adj: Dict[str, Set[str]] = {}

        def _targets(e: Dict[str, Any]) -> List[str]:
            keys = [e.get("target", "")]
            for cs in (e.get("call_sites") or []):
                keys.append(cs.get("target_module", ""))
            return keys

        for e in (payload.get("edges") or []):
            src = str(e.get("source") or "").strip()
            if not src:
                continue
            for raw_key in _targets(e):
                k = str(raw_key or "").upper().strip()
                if not k:
                    continue
                for dst in name_to_stems.get(k, ()):  # absent name → out-of-corpus, skip
                    if dst and dst != src:
                        adj.setdefault(src, set()).add(dst)
        return adj

    def name_to_stems(self) -> Dict[str, "Set[str]"]:
        """``UPPER name -> {stems that EXPORT it}`` over the whole corpus (cached).

        A cross-file call TARGET is a node NAME — a file stem OR an entry-point /
        module name (``asm_entry_point_map`` key, entry/csect/public global symbol).
        This is the SAME expansion ``walk_callers`` uses (via ``_entry_names``), so a
        target like ``NB81`` resolves to its owning ASM stem ``nb81``, and a C++
        entry like ``GETLWRWRADDRESSES`` resolves to ``av200100``. Used by both the
        forward-file prune and the unified route enumerator (cpp_routes)."""
        if self._name_to_stems is not None:
            return self._name_to_stems
        self._ensure_graph()
        from backward_traversal.runner.chainless_caller_walker import _entry_names
        payload = self._payload()
        node_type = self._node_type or {}
        stems = [str(n.get("id") or "").strip()
                 for n in (payload.get("nodes") or [])
                 if str(n.get("type") or "").lower() in ("asm", "cpp")]
        stems = [s for s in stems if s]
        m: Dict[str, Set[str]] = {}
        for stem in stems:
            ftype = node_type.get(stem, "asm")
            for nm in _entry_names(stem, ftype, self.blueprint_dir):
                m.setdefault(str(nm).upper(), set()).add(stem)
        self._name_to_stems = m
        return m

    def node_types(self) -> Dict[str, str]:
        """``stem -> "asm"|"cpp"`` over the corpus (cached graph payload)."""
        self._ensure_graph()
        return dict(self._node_type or {})

    def graph_edges(self) -> List[Dict[str, Any]]:
        """Raw file_call_graph edges (read-only). Each carries ``source`` (a stem),
        ``target`` (a resolved stem when in-corpus), and ``call_sites`` with
        ``target_module`` (entry name) + ``line``. Used by the unified enumerator to
        derive ASM-source cross edges (an ASM module is single-entry, so its outgoing
        edges come from here, not from any intra-function call graph)."""
        return list(self._payload().get("edges") or [])

    def _load_or_build_fwd_file_adj(self) -> Dict[str, "Set[str]"]:
        """B (cold-startup): prefer the precomputed forward-file-adjacency blob (load-only) so the
        cold trace skips the second full O(edges+call_sites) payload pass AND the payload load. Set-
        valued, so the round-trip (set→sorted list→set) is order-independent ⇒ byte-identical. Else
        build from the (lazily-loaded) payload."""
        if self.job_id:
            try:
                from vbt.precompute import db_artifacts as _DA
                if _DA.is_load_only():
                    from vbt.precompute.graph_db import load_forward_file_adj
                    fa = load_forward_file_adj(self.job_id)
                    if fa is not None:
                        return fa
            except Exception:
                pass
        return self._build_forward_file_adj()

    def forward_reachable_files(self, endpoint: Endpoint) -> "Set[str]":
        """Set of file stems forward-reachable from ``endpoint``'s file over the
        metadata-only forward graph (BFS), INCLUDING the endpoint's own file.

        Guaranteed superset of every file that any ``routes(endpoint, setter)`` could
        traverse (see ``_build_forward_file_adj``). Used to prune the candidate-setter
        file set before opening any file — a pure speedup, never a result change.
        Cached per endpoint stem (function granularity does not change the file set).
        """
        endpoint = endpoint.norm()
        start = str(endpoint.file_stem).strip()
        if start in self._fwd_reach_cache:
            return self._fwd_reach_cache[start]
        if self._fwd_file_adj is None:
            self._fwd_file_adj = self._load_or_build_fwd_file_adj()
        adj = self._fwd_file_adj
        seen: Set[str] = {start}
        stack: List[str] = [start]
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):  # type: ignore[arg-type]
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        self._fwd_reach_cache[start] = seen
        return seen

    def forward_file_distances(self, endpoint: Endpoint, *, max_depth: int = 0) -> Dict[str, int]:
        """Minimum metadata-only file-hop distance from ``endpoint``'s file.

        The metadata graph is a superset of verified route edges, so these distances are
        admissible for pruning reverse caller walks: if ``parent_dist[source] + depth_to_child``
        exceeds ``max_route_len``, no verified route through ``source`` can be emitted.
        """
        endpoint = endpoint.norm()
        start = str(endpoint.file_stem).strip()
        key = (start, int(max_depth or 0))
        cached = self._fwd_dist_cache.get(key)
        if cached is not None:
            return cached
        if self._fwd_file_adj is None:
            self._fwd_file_adj = self._load_or_build_fwd_file_adj()
        from collections import deque
        adj = self._fwd_file_adj
        dist: Dict[str, int] = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            cur_d = dist[cur]
            if max_depth and cur_d >= max_depth:
                continue
            for nxt in adj.get(cur, ()):  # type: ignore[arg-type]
                nd = cur_d + 1
                if max_depth and nd > max_depth:
                    continue
                if nxt not in dist:
                    dist[nxt] = nd
                    q.append(nxt)
        if len(self._fwd_dist_cache) >= 256:
            self._fwd_dist_cache.pop(next(iter(self._fwd_dist_cache)))
        self._fwd_dist_cache[key] = dist
        return dist


# Memo for the reattributed line-aware intra-file adjacency used by the dep-var before_line bound.
# It is a pure function of (stem, blueprint_dir, asm_dir) — NOT of from_fn/to_fn/before_line — but was
# rebuilt + RE-REATTRIBUTED on every call (the per-route hot path: ~0.8s of reattribute_edges at
# trace). Value is the adjacency dict, or None to mark "no call graph" (the conservative branch).
# Byte-identical (deterministic per stem); bounded (FIFO, recompute on eviction).
_FRB_FWD_CACHE: Dict[Any, Any] = {}
_FRB_FWD_CACHE_MAX = 1024
_FRB_MISS = object()


def function_reaches_before(
    stem: str,
    file_type: str,
    from_fn: str,
    to_fn: str,
    blueprint_dir: Path,
    *,
    before_line: Optional[int] = None,
    asm_dir: Optional[Path] = None,
) -> bool:
    """Dep-var bound: does ``from_fn`` reach ``to_fn`` within ``stem`` using only
    intra-file call edges that occur *before* ``before_line``?

    ``before_line=None`` → unbounded (delegates to the read-only ``function_reaches``).
    Otherwise we walk the read-only intra-file forward adjacency ourselves and skip
    call edges at/after the cap. This is the "before the descend call site" rule.

    The blueprint attributes every call inside a switch-truncated body to caller
    ``(global)``; left raw, the BFS from ``from_fn`` never traverses those edges and
    silently reports a real reach as unreachable (dropping legitimate dep-var
    setters). When ``asm_dir`` is given we de-poison the sources by nearest-preceding
    cfg START (vbt/reach/fn_attr) before building the adjacency.
    """
    if not from_fn or not to_fn or from_fn.upper() == to_fn.upper():
        return True
    if file_type != "cpp":
        return True  # ASM module ≈ single entry (intra-file fn granularity n/a)
    if before_line is None:
        return RCH.function_reaches(stem, file_type, from_fn, to_fn, Path(blueprint_dir), {})

    # Memoized per (stem, dirs): build the reattributed line-aware adjacency ONCE, reuse across the
    # many before_line calls for this stem (was re-reattributing on every call).
    _fk = (stem, str(blueprint_dir), str(asm_dir))
    fwd = _FRB_FWD_CACHE.get(_fk, _FRB_MISS)
    if fwd is _FRB_MISS:
        bp = RCH._load_bp(stem, "cpp", Path(blueprint_dir))  # read-only reuse
        edges = ((bp.get("call_graph") or {}).get("edges") or []) if bp else []
        if not edges:
            fwd = None                       # no call graph → conservative (cache the marker too)
        else:
            cfg_fns: List[Dict[str, Any]] = []
            if asm_dir is not None:
                p = Path(asm_dir) / f"{stem}.cpp"
                if p.exists():
                    from vbt.cpp_frontend.wrapper import run_cfg_extract
                    try:
                        cfg_fns = run_cfg_extract(str(p))
                    except (FileNotFoundError, RuntimeError):
                        pass
            # line-aware forward adjacency: caller(UPPER) -> [(callee(UPPER), line)], with
            # (global)/empty sources resolved to their enclosing function by line-range.
            fwd = {}
            for s, t, ln in reattribute_edges(edges, cfg_fns):
                s = s.upper()
                t = t.upper()
                if s and t and s != t:
                    fwd.setdefault(s, []).append((t, ln))
        if len(_FRB_FWD_CACHE) >= _FRB_FWD_CACHE_MAX:   # bounded; recompute on eviction (byte-identical)
            _FRB_FWD_CACHE.pop(next(iter(_FRB_FWD_CACHE)))
        _FRB_FWD_CACHE[_fk] = fwd
    if fwd is None:
        return True  # no call graph to prove non-reachability → conservative

    a, b = from_fn.upper(), to_fn.upper()
    seen: Set[str] = {a}
    stack: List[str] = [a]
    while stack:
        cur = stack.pop()
        for callee, line in fwd.get(cur, []):
            if line is not None and line >= before_line:
                continue  # call after the descend point — not on the path to the setter
            if callee == b:
                return True
            if callee not in seen:
                seen.add(callee)
                stack.append(callee)
    return b in seen


def intra_call_paths(
    edges: List[Tuple[str, str, int]],
    local_fns: Set[str],
    from_fn: str,
    to_fn: str,
    *,
    max_paths: int = 64,
    max_depth: int = 16,
) -> List[List[Tuple[str, int, str]]]:
    """Enumerate ALL distinct intra-file call paths ``from_fn`` → ``to_fn``.

    Each returned path is an ordered list of edges ``(caller_fn, call_line, callee_fn)``.
    Only edges whose callee is a *local* function (defined in this file) are followed,
    so operator/cross-file targets are ignored. Crucially, two call sites between the
    same caller/callee pair are DISTINCT edges (distinct lines) → distinct paths: this
    is what produces R6's two OR-alternative tuples and R3's two-path result.

    ``from_fn == to_fn`` returns ``[[]]`` — one trivial path with no edges (the setter
    sits in the entry function itself; only its local guard + the chain prefix apply).
    Bounded by ``max_paths``/``max_depth`` so a pathological fan-out at 22k scale can't
    explode; simple paths only (a function is not revisited within one path).
    """
    if from_fn == to_fn:
        return [[]], False
    adj: Dict[str, List[Tuple[str, int]]] = {}
    for s, t, ln in edges:
        if t in local_fns:
            adj.setdefault(s, []).append((t, ln))
    results: List[List[Tuple[str, int, str]]] = []
    truncated = False

    def dfs(cur: str, path: List[Tuple[str, int, str]], visited: Set[str]) -> None:
        nonlocal truncated
        if len(results) >= max_paths:
            truncated = True
            return
        if len(path) >= max_depth:
            if adj.get(cur):          # more edges remained to explore past the cap
                truncated = True
            return
        for callee, ln in adj.get(cur, []):
            edge = (cur, ln, callee)
            if callee == to_fn:
                if len(results) >= max_paths:
                    truncated = True
                    return
                results.append(path + [edge])
            elif callee not in visited:
                dfs(callee, path + [edge], visited | {callee})

    dfs(from_fn, [], {from_fn})
    return results, truncated
