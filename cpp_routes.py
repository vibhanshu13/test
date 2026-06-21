"""Robust UNIFIED function-level path enumeration (the route-undercount fix).

WHY THIS EXISTS
===============
The legacy cross-file router (``backward_traversal.route_finder`` + the reverse
caller-walk) under-counts routes to a setter for TWO compounding reasons:

  1. **Caller mis-attribution.** Calls made inside a function whose body contains
     a ``switch`` on an unresolved-type scrutinee get attributed to the caller
     ``(global)`` (the parser/Clang truncates the recovered function body at the
     switch). So a real edge ``processACreditTransaction → processMagSwipeWith4dbc``
     is recorded with caller ``(global)`` — and route gating (which checks the
     entered function reaches the calling function) can never confirm it.
  2. **The caller-walk collapses multi-callee files.** ``build_reverse_adjacency``
     keeps a single callee per caller-file node, so a file that calls BOTH an
     intermediate AND the setter file loses one of those edges and is dropped to
     ``unverified``. This is **header-independent** — it mis-counts even when the
     attribution is perfect.

The net effect: for ``softCardValue`` the trace found only the 2 *direct*
``dw710000 → dw780000`` bridges into ``selectCidDbRecord15`` and silently dropped
every route through ``dw710300`` / ``dw710500`` / ``dw710600`` / ``dw710800`` —
losing the distinct upstream business guards those routes carry.

THE FIX (robust, VBT-owned)
===========================
Never trust the parser's caller label nor the legacy caller-walk. Build a
function-level call graph entirely from:

  * the per-file blueprint call graph (``RouteEngine.cpp_call_edges``) — which
    carries BOTH intra- and cross-file targets, in original case, WITH lines; and
  * an authoritative ``line → enclosing function`` resolver derived from the
    ``cfg_extract`` function START lines (recovered reliably even when the BODY
    truncates — we use the nearest preceding start and ignore the unreliable end).

Any ``(global)``/empty caller is re-attributed by line-range. Cross-file callees
are resolved to their defining file via a global ``fn → file`` map (built from the
same cfg function lists). Then we enumerate **function-node simple paths** (a file
may be re-entered via a DIFFERENT function — that is how the ``dw710600`` round-trip
routes arise) from the chain tail to the setter function.

Each emitted path is a list of edge dicts in the EXACT shape
``_reconstruct_cpp_paths`` produces — ``{caller_stem, caller_fn, line,
callee_stem, callee_fn, cross}`` — so the engine's existing per-edge guard /
``via`` / dep-var machinery consumes it unchanged.

GENERALIZATION TO MIXED HLASM/C++ ROUTES
========================================
The same caller-walk collapse silently drops routes for EVERY language pairing,
not just pure C++. The most common z/TPF case: C++ business logic reaches an ASM
counter / DB module (``dw710700 → nb81``). The legacy ``route_finder`` returns
only the few files on its non-collapsed routes and flags the rest ``unverified``
(reachable=True via a direct route, so the drop is silent). This module therefore
builds a UNIFIED graph whose nodes are BOTH C++ functions ``(stem, fn)`` (via
cfg-START attribution, as before) AND ASM modules as SINGLE-ENTRY nodes
``(asm_stem, asm_stem)``:

  * C++→ASM: a ``cpp_call_edges`` target that is neither a local C++ function nor a
    cross-file C++ function resolves through the corpus entry-name map
    (``RouteEngine.name_to_stems`` — the same ``_entry_names`` expansion the
    caller-walk uses) to its owning ASM stem ⇒ a cross edge to ``(asm_stem,
    asm_stem)``.
  * ASM→ASM / ASM→C++: an ASM module is single-entry, so its outgoing edges are
    read directly from the file_call_graph (``RouteEngine.graph_edges``); each
    ``target`` (resolved stem) / ``call_sites.target_module`` (entry name) maps to a
    destination node (an ASM module node, or a C++ entry function).

DFS from the tail node to the setter node then enumerates the SAME complete simple
path set for an ASM-setter-from-C++-tail (and any mixed route) that pure C++ already
gets. PURE-ASM tail traces stay on the legacy path (ASM modules are single-entry
nodes with direct stem keying — no intra-function paths — and the legacy router is
already complete for them).

This module is read-only over the blueprints + cfg cache + file_call_graph; it adds
NO new parser dependency and is kb_id-free.
"""
from __future__ import annotations

import bisect
import logging
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

# Under the "vbt" logger root so the same --progress/level switch controls these hang-locating logs.
logger = logging.getLogger("vbt.reach.cpp_routes")

# enumerate_paths two-phase budgets (module-level so they're tunable + testable). Phase 1 runs the
# byte-identical LEXICAL DFS up to _DFS_LEX_BUDGET visits; every reference call finishes far under it
# (a few thousand visits), so the gate stays green. Only a dense-graph explosion blows the budget and
# falls back to the best-first phase, which is itself hard-capped at _DFS_HARD_CAP.
_DFS_LEX_BUDGET = 300_000
_DFS_HARD_CAP = 5_000_000
# Phase-2 (best-first) stale-stop: once it has gone this many visits WITHOUT finding a NEW path, stop —
# best-first reaches the real completing paths early, the rest is fruitless churn toward the 5M cap.
# Applies ONLY to phase 2 (the explosion fallback, which has no golden), so it never touches phase 1.
_DFS_STALE_AFTER = 500_000

# A caller label that means "no real enclosing function was recorded".
_GLOBALISH = {"", "(global)", "global", "glob", None}


class CppFnGraph:
    """UNIFIED function-level call graph with robust caller attribution.

    Node = ``(file_stem, function_name)``. A C++ node carries a cfg function name
    (original case); an ASM module is a SINGLE-ENTRY node ``(asm_stem, asm_stem)``.
    Built lazily over a set of candidate file stems (C++ and ASM); cached on the
    RouteEngine so repeated setter queries in one trace reuse it.
    """

    def __init__(self, route, cfg_cache: Dict[str, List[Dict]], asm_dir: Path,
                 candidate_stems: Set[str], node_types: Optional[Dict[str, str]] = None):
        self.route = route
        self.cfg_cache = cfg_cache
        self.asm_dir = Path(asm_dir)
        # node types over the WHOLE corpus (stem -> "asm"|"cpp"); used to classify a
        # resolved cross-file target's owning stem. Falls back to a .cpp-on-disk probe.
        self._node_types: Dict[str, str] = dict(node_types or {})
        # candidate stems split by language. C++ stems get cfg-derived intra-file
        # function structure; ASM stems are single-entry module nodes.
        self.stems = {s for s in candidate_stems if self._type_of(s) == "cpp"}
        self.asm_stems = {s for s in candidate_stems if self._type_of(s) == "asm"}
        # per-stem: sorted [(start_line, fn)] for nearest-preceding-start attribution
        self._starts: Dict[str, List[Tuple[int, str]]] = {}
        # per-stem: set of local function names (cfg original case)
        self._local: Dict[str, Set[str]] = {}
        # fn name -> defining file stem(s)
        self._fn_to_file: Dict[str, Set[str]] = {}
        # adjacency: (stem, fn) -> set of (callee_stem, callee_fn, line, cross)
        self._adj: Optional[Dict[Tuple[str, str], Set[Tuple[str, str, int, bool]]]] = None
        # C++ stems whose CFG could not be loaded. These are treated as empty
        # rather than aborting a full-corpus precompute because one source file
        # is missing or unparseable.
        self.skipped_cfg_stems: Dict[str, str] = {}
        self.empty_cfg_stems: Set[str] = set()

    def _type_of(self, stem: str) -> str:
        """``"asm"`` | ``"cpp"`` for a stem. Prefers the corpus node-type map; falls
        back to a ``.cpp`` on-disk probe (default ``asm`` — a single-entry module)."""
        t = self._node_types.get(stem)
        if t in ("asm", "cpp"):
            return t
        return "cpp" if (self.asm_dir / f"{stem}.cpp").exists() else "asm"

    # ---- cfg-derived per-file facts ---------------------------------------- #
    @classmethod
    def from_prebuilt(cls, adj: Dict[Tuple[str, str], Set[Tuple[str, str, int, bool]]]) -> "CppFnGraph":
        """A graph backed by a PRE-BUILT adjacency (loaded from index.db) — supports only the path
        enumeration API (`adjacency` / `enumerate_paths`). The cfg/edge scan and the `_attribute` /
        `_starts` / `_local` build helpers are skipped: the adjacency already encodes every node and
        edge, and enumeration uses only `_adj` (+ the lazily-derived `_radj` / `_R_cache`). Byte-
        identical — the stored adjacency is exactly what `_build()` produces."""
        inst = object.__new__(cls)
        inst._adj = adj
        inst._radj = None
        inst._R_cache = None
        inst.skipped_cfg_stems = {}
        inst.empty_cfg_stems = set()
        return inst

    def _ensure_file(self, stem: str) -> None:
        if stem in self._local:
            return
        cpp_path = self.asm_dir / f"{stem}.cpp"
        if self._type_of(stem) == "cpp" and not cpp_path.exists():
            reason = f"source file not found: {cpp_path}"
            logger.warning("_ensure_file: skipping stem %s - %s", stem, reason)
            self.skipped_cfg_stems[stem] = reason
            self._starts[stem] = []
            self._local[stem] = set()
            return
        # 3A: in load-only, serve the precomputed (starts, local) instead of loading+parsing the full
        # cfg per stem (the mass cfg load = the cold-DB build cost). Byte-identical (same derivation);
        # a miss (stem absent from the map) falls through to the live _cfg_for below.
        from vbt.precompute.fn_facts_db import get_fn_fact_skip_reason, get_fn_facts
        _facts = get_fn_facts(stem)
        if _facts is not None:
            self._starts[stem], self._local[stem] = _facts
            _reason = get_fn_fact_skip_reason(stem)
            if _reason:
                self.skipped_cfg_stems[stem] = _reason
            elif not self._starts[stem] and not self._local[stem]:
                self.empty_cfg_stems.add(stem)
            for name in self._local[stem]:
                self._fn_to_file.setdefault(name, set()).add(stem)
            return
        from vbt.engine import _cfg_for, _cfg_skip_reason, _fn_lines  # local import: avoid cycle
        try:
            fns = _cfg_for(stem, self.cfg_cache, self.asm_dir)
        except Exception as exc:
            logger.warning("_ensure_file: skipping stem %s - %s", stem, exc)
            self.skipped_cfg_stems[stem] = str(exc)[:500]
            self._starts[stem] = []
            self._local[stem] = set()
            return
        _reason = _cfg_skip_reason(stem)
        if _reason:
            self.skipped_cfg_stems[stem] = _reason
        elif not fns:
            self.empty_cfg_stems.add(stem)
        starts: List[Tuple[int, str]] = []
        local: Set[str] = set()
        for f in fns:
            name = (f.get("function") or "").split("::")[-1]
            if not name:
                continue
            local.add(name)
            lines = _fn_lines(f)
            if lines:
                starts.append((min(lines), name))
        starts.sort()
        self._starts[stem] = starts
        self._local[stem] = local
        for name in local:
            self._fn_to_file.setdefault(name, set()).add(stem)

    def _attribute(self, stem: str, line: Optional[int]) -> Optional[str]:
        """The function whose START line is the greatest ≤ ``line`` (nearest
        preceding definition). Robust to truncated bodies — uses starts only."""
        self._ensure_file(stem)
        starts = self._starts.get(stem) or []
        if not starts or line is None:
            return None
        idx = bisect.bisect_right([s for s, _ in starts], line) - 1
        return starts[idx][1] if idx >= 0 else None

    # ---- cross-file target resolution -------------------------------------- #
    def _resolve_asm_owner(self, tgt: str) -> Optional[str]:
        """If a cross-file target NAME owns an in-candidate ASM module, return its
        stem (deterministic tiebreak). The corpus entry-name map keys are UPPER —
        the same expansion the caller-walk uses (``NB81`` -> ``nb81``)."""
        owners = self.route.name_to_stems().get(str(tgt).upper())
        if not owners:
            return None
        asm_owners = sorted(o for o in owners if self._type_of(o) == "asm")
        for o in asm_owners:
            if o in self.asm_stems:
                return o
        return asm_owners[0] if asm_owners else None

    def _asm_entry_fn(self, dst_stem: str, target_module: str) -> str:
        """The C++ entry function a cross edge into ``dst_stem`` enters, when an ASM
        module calls a C++ file. The file_call_graph ``target_module`` is the UPPER
        entry name; map it back to the cfg function of original case if one matches,
        else use the name as-is (the engine resolves the function block by name)."""
        self._ensure_file(dst_stem)
        up = str(target_module).upper()
        for fn in self._local.get(dst_stem, ()):  # original-case cfg fn match
            if fn.upper() == up:
                return fn
        return target_module

    # ---- graph construction ------------------------------------------------ #
    def _build(self) -> Dict[Tuple[str, str], Set[Tuple[str, str, int, bool]]]:
        # Progress for the one-time graph build (≈6 min at 22k = the silent gap after "START per-route
        # entry building"). INFO-gated ⇒ no-op + byte-identical by default; adj is order-independent
        # (dict-of-sets via setdefault().add()), so enumerate() for the heartbeat changes nothing.
        _ns = len(self.stems)
        for _i, stem in enumerate(self.stems):
            if _i and _i % 1000 == 0 and logger.isEnabledFor(logging.INFO):
                logger.info("    fn-graph build: cfg load %d/%d stems...", _i, _ns)
            self._ensure_file(stem)
        adj: Dict[Tuple[str, str], Set[Tuple[str, str, int, bool]]] = {}
        # ---- C++ source nodes: intra-file + cross-file (C++ AND ASM) edges ----
        for _i, stem in enumerate(self.stems):
            if _i and _i % 1000 == 0 and logger.isEnabledFor(logging.INFO):
                logger.info("    fn-graph build: edges %d/%d stems (%d nodes)...", _i, _ns, len(adj))
            local = self._local.get(stem, set())
            for src, tgt, line in self.route.cpp_call_edges(stem):
                caller = src if src not in _GLOBALISH else None
                if caller is None or caller in _GLOBALISH:
                    caller = self._attribute(stem, line)
                if not caller:
                    continue
                if tgt in local:                       # intra-file call
                    adj.setdefault((stem, caller), set()).add((stem, tgt, line, False))
                    continue
                owners = self._fn_to_file.get(tgt)     # cross-file C++ call
                if owners:
                    # deterministic tiebreak; name collisions across files are rare
                    callee_stem = sorted(owners)[0]
                    if callee_stem != stem:
                        adj.setdefault((stem, caller), set()).add(
                            (callee_stem, tgt, line, True))
                    continue
                asm_stem = self._resolve_asm_owner(tgt)  # cross-file C++→ASM call
                if asm_stem and asm_stem != stem:
                    # ASM module = single-entry node (asm_stem, asm_stem).
                    adj.setdefault((stem, caller), set()).add(
                        (asm_stem, asm_stem, line, True))
        # ---- ASM source nodes: single-entry; outgoing edges from file_call_graph ----
        # An ASM module has no intra-function call graph here, so its outgoing edges
        # are read straight from the file-level graph (resolved targets + entry names).
        if self.asm_stems:
            self._build_asm_edges(adj)
        return adj

    def _build_asm_edges(
        self, adj: Dict[Tuple[str, str], Set[Tuple[str, str, int, bool]]]
    ) -> None:
        """Add outgoing edges for every candidate ASM module from the file_call_graph.

        An ASM file_call_graph edge carries one resolved ``target`` stem and one or
        more ``call_sites`` (each an entry name ``target_module`` + ``line``) into it.
        A C++ destination enters its matching cfg entry function (one node per distinct
        entry name); an ASM destination is its single-entry node. Targets not resolved
        to an in-corpus stem are out-of-corpus dead ends (skipped, like the file-prune).
        """
        n2s = self.route.name_to_stems()

        def _dest_stem(tgt: str, entry: str) -> Optional[str]:
            if tgt and tgt in self._node_types:        # already a resolved in-corpus stem
                return tgt
            owners = sorted(n2s.get(str(entry).upper(), ()) or n2s.get(str(tgt).upper(), ()))
            return owners[0] if owners else None

        for e in self.route.graph_edges():
            src = str(e.get("source") or "").strip()
            if src not in self.asm_stems:
                continue
            tgt = str(e.get("target") or "").strip()
            sites = [(str(cs.get("target_module") or "").strip(), int(cs.get("line") or 0))
                     for cs in (e.get("call_sites") or [])]
            if not sites:
                sites = [(tgt, 0)]
            for entry, ln in sites:
                dst = _dest_stem(tgt, entry)
                if not dst or dst == src:
                    continue
                if self._type_of(dst) == "asm":
                    adj.setdefault((src, src), set()).add((dst, dst, ln, True))
                else:  # ASM → C++: enter the matching cfg entry function
                    self._ensure_file(dst)
                    callee_fn = self._asm_entry_fn(dst, entry)
                    adj.setdefault((src, src), set()).add((dst, callee_fn, ln, True))

    def adjacency(self) -> Dict[Tuple[str, str], Set[Tuple[str, str, int, bool]]]:
        if self._adj is None:
            self._adj = self._build()
        return self._adj

    # ---- path enumeration -------------------------------------------------- #
    def enumerate_paths(
        self, start: Tuple[str, str], target: Tuple[str, str],
        *, max_paths: int = 64, max_depth: int = 16,
    ) -> Tuple[List[List[Dict[str, object]]], bool]:
        """All function-node simple paths ``start → target``.

        Returns ``(paths, truncated)``. Each path is an ordered list of edge dicts
        ``{caller_stem, caller_fn, line, callee_stem, callee_fn, cross}``.
        A node ``(stem, fn)`` is never revisited within one path (so a FILE may
        recur via a different function, but the path always terminates). Bounded
        by ``max_paths`` / ``max_depth`` (truncation is surfaced, never silent)."""
        adj = self.adjacency()
        if start == target:
            return [[]], False

        # 3C + distance bound: dist[n] = MIN hops n→target via backward BFS over the reverse adjacency
        # (cached per target). Two ADMISSIBLE prunes in the DFS, each dropping only branches that yield
        # NO ≤max_depth start→target path, so the set + ORDER of COMPLETE paths stays byte-identical:
        #   (a) dist.get(nxt) is None  ⇒ nxt can't reach target at all          (the original 3C prune);
        #   (b) len(path)+1+dist[nxt] > max_depth ⇒ the SHORTEST completion via nxt already exceeds the
        #       depth cap. THIS is the dense-graph fix: without it the DFS wanders millions of ≤max_depth
        #       dead partials before reaching the real (short) paths, the safety cap then fires, and it
        #       returns a SPURIOUS 0 results (the 22k bug — e.g. callPlastic…→processACreditTransaction,
        #       5 real paths, wrongly capped to 0). With it the DFS heads straight at the target.
        radj = getattr(self, "_radj", None)
        if radj is None:
            radj = {}
            for _n, _outs in adj.items():
                for (_cs, _cf, _l, _c) in _outs:
                    radj.setdefault((_cs, _cf), set()).add(_n)
            self._radj = radj
        _Rc = getattr(self, "_R_cache", None)
        if _Rc is None:
            _Rc = self._R_cache = {}
        dist = _Rc.get(target)
        if dist is None:
            dist = {target: 0}
            _q = deque([target])
            while _q:
                _c = _q.popleft()
                for _p in radj.get(_c, ()):
                    if _p not in dist:
                        dist[_p] = dist[_c] + 1
                        _q.append(_p)
            _Rc[target] = dist

        # SPARSE-COMPLETION FIX (the 22k bug). In a dense graph, a setter whose completing paths are
        # FEW and lexically LATE makes the lexical DFS wander millions of dist-feasible dead partials
        # before reaching them — the safety cap then fires with a spurious 0 results (e.g.
        # callPlastic…→processACreditTransaction: 5 real paths, wrongly capped to 0 at scale).
        #
        # Two phases, byte-identical at reference scale:
        #   (1) LEXICAL DFS under a modest visit budget. Every reference call finishes in a few thousand
        #       visits, far under the budget, so its result is the unchanged lexical order (the gate
        #       stays green). When it finishes within budget we return it verbatim.
        #   (2) ONLY when phase 1 blows the budget (the explosion) do we fall back to a BEST-FIRST DFS
        #       (nearest-to-target neighbour first) — it heads straight at the target and finds the few
        #       completing paths cheaply — then re-sort the paths into the exact order the lexical DFS
        #       would have emitted (lexicographic on each hop's (callee_stem, callee_fn, line, cross)).
        #       For a sparse setter (< max_paths total) that sorted set IS the lexical result.
        _LEX_BUDGET = _DFS_LEX_BUDGET
        _HARD_CAP = _DFS_HARD_CAP

        def _run(order_key, budget, stale_after=None):
            results: List[List[Dict[str, object]]] = []
            st = {"trunc": False, "calls": 0, "exhausted": False, "stale": False, "last_result_at": 0}

            def dfs(node, path, visited):
                st["calls"] += 1
                if st["calls"] > budget:
                    st["exhausted"] = True            # blew the budget → caller falls back to best-first
                    return
                # Phase-2-only stale-stop (stale_after=None in phase 1 ⇒ skipped ⇒ byte-identical):
                # no NEW path in `stale_after` visits ⇒ stop the fruitless best-first churn.
                if stale_after is not None and st["calls"] - st["last_result_at"] > stale_after:
                    st["stale"] = True
                    st["trunc"] = True                # stopped early ⇒ more paths may exist
                    return
                if st["calls"] % 250000 == 0 and logger.isEnabledFor(logging.INFO):
                    logger.info("    enumerate_paths DFS churning: %d visits (target %s::%s)...",
                                st["calls"], target[0], target[1])
                if len(results) >= max_paths:
                    st["trunc"] = True
                    return
                if len(path) >= max_depth:
                    if adj.get(node):
                        st["trunc"] = True
                    return
                nbrs = adj.get(node, ())
                nbrs = sorted(nbrs, key=order_key) if order_key is not None else sorted(nbrs)
                for (cs, cf, line, cross) in nbrs:
                    nxt = (cs, cf)
                    _d = dist.get(nxt)
                    if _d is None:            # 3C: nxt can't reach target → zero-yield branch, skip
                        continue
                    if len(path) + 1 + _d > max_depth:   # distance bound: shortest completion exceeds
                        st["trunc"] = True                # the depth cap ⇒ a longer path exists, skip
                        continue
                    edge = {"caller_stem": node[0], "caller_fn": node[1], "line": line,
                            "callee_stem": cs, "callee_fn": cf, "cross": cross}
                    if nxt == target:
                        if len(results) >= max_paths:
                            st["trunc"] = True
                            return
                        results.append(path + [edge])
                        st["last_result_at"] = st["calls"]   # reset the stale-stop window (phase 2 only)
                    elif nxt not in visited:
                        dfs(nxt, path + [edge], visited | {nxt})
                    if st["exhausted"] or st["stale"] or (st["trunc"] and len(results) >= max_paths):
                        return
            dfs(start, [], {start})
            return results, st["trunc"], st["exhausted"]

        # Phase 1 — lexical (byte-identical when it completes within budget; always true at ref scale).
        results, truncated, exhausted = _run(None, _LEX_BUDGET)
        if not exhausted:
            return results, truncated
        # Phase 2 — explosion: best-first finds the few completing paths (with a stale-stop so it doesn't
        # churn to the hard cap after the last one), then re-sort to lexical order.
        results, truncated, _ = _run(
            lambda e: (dist.get((e[0], e[1]), 1 << 30), e[0], e[1], e[2], e[3]),
            _HARD_CAP, stale_after=_DFS_STALE_AFTER)
        results.sort(key=lambda p: tuple(
            (e["callee_stem"], e["callee_fn"], e["line"], e["cross"]) for e in p))
        return results, truncated


def get_cpp_fn_graph(route, cfg_cache: Dict[str, List[Dict]], asm_dir: Path,
                     tail_ep) -> CppFnGraph:
    """Build (once per RouteEngine) the UNIFIED function graph over BOTH the C++
    and ASM files forward-reachable from the chain tail — the same provable superset
    the file prune uses, so we never miss a reachable route nor scan an unreachable
    file. C++ stems contribute cfg-derived intra-file function nodes; ASM stems are
    single-entry module nodes (so an ASM setter reached from a C++ tail enumerates the
    same complete route set pure C++ already gets)."""
    cached = getattr(route, "_cpp_fn_graph", None)
    if cached is not None:
        return cached
    # Load-only fast path: the full-corpus fn-graph adjacency is precomputed → LOAD it (~0.1s)
    # instead of rebuilding (the cold trace's dominant one-time cost at 22k). Byte-identical:
    # enumerate_paths' DFS from `start` visits only nodes forward-reachable from start, so the
    # full-corpus adjacency yields exactly the paths the tail-reachable-subset build would (any
    # node on a start→target path is reachable from start, hence in the subset; the distance bound
    # is a shortest-path property unchanged by unreachable edges). Cold/absent blob → live build.
    _job = getattr(route, "job_id", None)
    if _job:
        try:
            from vbt.precompute.graph_db import load_fn_graph_adj
            _adj = load_fn_graph_adj(_job)
        except Exception:
            _adj = None
        if _adj is not None:
            graph = CppFnGraph.from_prebuilt(_adj)
            if logger.isEnabledFor(logging.INFO):
                logger.info("  fn-graph loaded from DB: %d nodes (live build skipped)", len(_adj))
            setattr(route, "_cpp_fn_graph", graph)
            return graph
    reachable = route.forward_reachable_files(tail_ep)
    node_types = route.node_types()
    graph = CppFnGraph(route, cfg_cache, asm_dir, reachable, node_types)
    # Hang-locating timing: the one-time graph build is O(reachable edges) — small here, seconds at 22k.
    # Force+time it only when INFO is on (adjacency() is idempotent: identical _adj, just built a moment
    # earlier than the first enumerate_paths would; no behavior change when logging is off).
    if logger.isEnabledFor(logging.INFO):
        _t0 = time.perf_counter()
        _adj = graph.adjacency()
        logger.info("  fn-graph build: %d reachable files, %d nodes, %.2fs",
                    len(reachable), len(_adj), time.perf_counter() - _t0)
    setattr(route, "_cpp_fn_graph", graph)
    return graph


def enumerate_cpp_paths(
    route, cfg_cache: Dict[str, List[Dict]], asm_dir: Path, tail_ep, setter,
    *, max_paths: int = 64, max_depth: int = 16,
) -> Tuple[List[List[Dict[str, object]]], bool]:
    """Robust tail→setter UNIFIED path enumeration (the route-undercount fix).

    Works for a C++ setter (target node ``(stem, fn)``) AND an ASM setter (target is
    the single-entry module node ``(asm_stem, asm_stem)``). Returns ``(paths,
    truncated)`` with each path in the ``_reconstruct_cpp_paths`` edge-dict shape, so
    the engine's per-edge guard/``via`` code consumes it unchanged. Empty list ⇒ no
    verified function path (caller falls back to the legacy reachability verdict)."""
    graph = get_cpp_fn_graph(route, cfg_cache, asm_dir, tail_ep)
    start = (tail_ep.file_stem, tail_ep.function or "")
    setter_lang = getattr(setter, "language", "cpp")
    if setter_lang == "cpp":
        target = (setter.file_stem, setter.function or "")
    else:  # ASM setter = single-entry module node
        target = (setter.file_stem, setter.file_stem)
    return graph.enumerate_paths(start, target, max_paths=max_paths, max_depth=max_depth)
