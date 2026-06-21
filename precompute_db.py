#!/usr/bin/env python3
"""CLI + orchestrator to populate ALL VBT DB artifacts for a job (DB_PRECOMPUTE_PLAN.md T11).

Builds/persists every ``vbt_*`` artifact a trace consumes, so a warm trace reads from index.db
instead of files: the resolvers (name-alias / membership / const / modifier index), the route
``name_to_stems`` map, the per-stem ``cpp_call_edges``, the ASM blueprints, the source text, and
the cfg blobs.  Precompute-only (O(corpus)); the trace only LOADS.  No parser-pipeline change —
everything is computed by VBT's own code from the blueprints + source it already reads.

    env/bin/python3 -m vbt.precompute_db \
        --job-id <id> --blueprint-dir <dir> --asm-dir <dir> [--graph <file>]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional


def build_and_store_cfg_blobs(job_id: str, asm_dir) -> Dict[str, int]:
    """Populate DB-backed cfg blobs from warm disk cache / live extraction.

    This is the final full-precompute step after ``fn_graph_adj``. It is also
    exposed as incremental artifact ``cfg`` so a failed run can continue from
    ``fn_graph_adj`` without rerunning the whole pipeline.
    """
    src = Path(asm_dir)
    from vbt.precompute.cfg_db import install_cfg_db, clear_cfg_db
    from vbt.precompute.fn_facts_db import get_fn_fact_skip_reasons
    from vbt.cpp_frontend import wrapper as _w

    install_cfg_db(job_id)
    n_cfg = 0
    try:
        skip_cfg_stems = set(get_fn_fact_skip_reasons(job_id))
    except Exception:
        skip_cfg_stems = set()
    try:
        # Earlier steps may have populated wrapper._MEMO before the DB PUT hook
        # existed. Clear it so disk-cache hits flow through _CFG_DB_PUT.
        _w._MEMO.clear()
        for p in sorted(src.glob("*.cpp")):
            if p.stem in skip_cfg_stems:
                continue
            try:
                _w.run_cfg_extract(str(p))
                n_cfg += 1
            except Exception:
                pass
    finally:
        clear_cfg_db()

    out = {"cfg": n_cfg}
    if skip_cfg_stems:
        out["cfg_skipped"] = len(skip_cfg_stems)
    return out


def precompute_vbt_db(job_id: str, blueprint_dir, asm_dir, graph_file=None,
                      warm=None) -> Dict[str, object]:
    """Build + persist every vbt_* artifact for ``job_id``. Returns a per-artifact summary.

    ``warm`` (optional, flag-managed): a list of trace specs ``{variable, language, hops, **caps}``
    to EAGER-warm after the artifacts are built — each is run as a real trace with ``job_id`` set,
    so the cross-trace route cache (#2-full) AND the whole-trace result cache (#1) are populated.
    The COLD first trace then preloads warm routes (~17s vs ~44s on the corpus), an exact repeat is
    instant (~0.2s), and a *similar* variable on the same chain reuses the warmed routes. Warming
    runs the real trace ⇒ byte-identical; it never changes any artifact, only fills the caches."""
    bp = Path(blueprint_dir)
    src = Path(asm_dir)
    gf = Path(graph_file) if graph_file else bp / "file_call_graph.json"
    out: Dict[str, object] = {}

    # Precompute always VERIFIES + (re)builds — force load-only OFF (the engine may have left it
    # ON if a trace ran earlier in this process).
    from vbt.precompute import db_artifacts as DA
    DA.set_load_only(False)
    # VBT precompute builds its artifacts from blueprints — it never reads the parser's gvl_edges
    # columns either. set_load_only(False) above reset the engine skip flag, so re-assert it: skip
    # the O(DB-size) gvl_edges backfill COUNT scans (the 20-min Step-5 stall on an 18GB DB). The
    # parser/lineage consumers (which DO use gvl_edges) still run the backfill on their own opens.
    try:
        from api.index_db.engine import set_skip_gvl_backfill
        set_skip_gvl_backfill(True)
    except Exception:
        pass

    # §16 #1: invalidate any cached whole-trace results — they're derived from the artifacts we
    # are about to (re)build, so a rebuild must never serve a stale cached trace.
    try:
        from vbt.precompute.trace_cache import clear as _clear_trace_cache
        out["trace_cache_cleared"] = _clear_trace_cache(job_id)
    except Exception:
        pass

    # §16 #2-full: drop the cross-trace route cache — its routes are derived from the call graph
    # we are about to rebuild, so a stale (parent→child) route must never be served.
    try:
        from vbt.precompute.route_cache_db import ARTIFACT as _RC_ART
        out["route_cache_cleared"] = DA.clear_blobs_prefix(job_id, _RC_ART)
    except Exception:
        pass

    # The call graph must be fresh before the route artifacts derive from it.
    from vbt.precompute.call_graph import ensure_call_graph
    ensure_call_graph(bp, src, out_path=gf)

    # Resolvers (name_alias, membership, const, modifier_index): preload builds+persists when stale.
    from vbt.precompute.resolvers_db import preload_resolvers
    out["resolvers"] = bool(preload_resolvers(job_id, bp, src))

    # file_call_graph payload → DB (so the trace reads it from the blob, not the .json file).
    from vbt.reach.route import RouteEngine
    from vbt.precompute.graph_db import (
        preload_route_graph, build_and_store_cpp_call_edges, build_and_store_file_call_graph,
        build_and_store_reverse_adj, build_and_store_forward_file_adj)
    out["file_call_graph"] = build_and_store_file_call_graph(job_id, bp, gf)

    # Route name_to_stems (build+persist) + per-stem cpp_call_edges (build+store).
    route = RouteEngine(bp, gf, src, job_id=job_id)
    preload_route_graph(route, job_id, bp, gf)
    out["cpp_call_edges"] = len(build_and_store_cpp_call_edges(job_id, bp, src, gf))

    # B (cold-startup): the two derived graph structures the cold trace otherwise rebuilds from the
    # 80MB payload every time — reverse_adj+node_type (order-faithful) + forward_file_adj. With these
    # stored, a load-only trace loads small blobs and skips the raw payload json.loads entirely.
    out["reverse_adj"] = build_and_store_reverse_adj(job_id, bp, gf)
    out["forward_file_adj"] = build_and_store_forward_file_adj(job_id, route)

    # ASM blueprints + raw source text.
    from vbt.precompute.asm_db import build_and_store_asm_blueprints
    out["asm_blueprints"] = build_and_store_asm_blueprints(job_id, bp)
    from vbt.precompute.source_db import build_and_store_source
    out["source"] = build_and_store_source(job_id, src)

    # Phase 1/2: per-file setter maps — precompute the setter SCAN (ASM block scans / C++ tree-sitter
    # parse) so a load-only trace LOOKS IT UP. Built here (load_only OFF ⇒ the finders still scan).
    from vbt.precompute.setter_map_db import build_and_store_asm_setter_map
    out["asm_setter_map"] = build_and_store_asm_setter_map(job_id, bp, src)
    from vbt.precompute.cpp_setter_map_db import build_and_store_cpp_setter_map
    out["cpp_setter_map"] = build_and_store_cpp_setter_map(job_id, bp, src)
    # cpp_file_writes: per-file _file_writes output (the path-family search parse), so a load-only trace
    # LOADS it instead of tree-sitter-parsing each touched .cpp (scales with the corpus — a 22k win).
    from vbt.precompute.cpp_file_writes_db import build_and_store_cpp_file_writes
    out["cpp_file_writes"] = build_and_store_cpp_file_writes(job_id, bp, src)
    from vbt.precompute.fn_facts_db import build_and_store_fn_facts
    out["fn_facts"] = build_and_store_fn_facts(job_id, bp, src)
    # fn_graph_adj: the full-corpus unified function-graph adjacency, so a load-only trace LOADS the
    # graph (~0.1s) instead of rebuilding it per cold trace (the dominant one-time build at 22k).
    from vbt.precompute.graph_db import build_and_store_fn_graph_adj
    out["fn_graph_adj"] = build_and_store_fn_graph_adj(job_id, route, src)

    # cfg blobs: final DB population pass after fn_graph_adj.
    out.update(build_and_store_cfg_blobs(job_id, src))

    # OPTIONAL eager warm (flag-managed): run the given trace(s) NOW so the route cache (#2-full)
    # + the whole-trace result cache (#1) are populated offline. A subsequent COLD trace preloads
    # the warm routes (~17s vs ~44s); an exact repeat is ~0.2s; a similar variable on the same
    # chain reuses the warmed routes. Byte-identical (it runs the real trace). Off unless requested.
    if warm:
        from vbt.engine import trace_root_variable
        DA.set_load_only(False)                 # build/verify mode; the trace flips it on itself
        warmed, errors = [], []
        for w in warm:
            try:
                t0 = time.monotonic()
                trace_root_variable(
                    w["variable"], w["language"], w["hops"],
                    blueprint_dir=bp, asm_dir=src, graph_file=gf, job_id=job_id,
                    max_dep_var_depth=int(w.get("max_dep_var_depth", 1)),
                    max_paths=int(w.get("max_paths", 64)),
                    max_call_depth=int(w.get("max_call_depth", 16)),
                    max_routes=int(w.get("max_routes", 200)),
                    max_route_len=int(w.get("max_route_len", 16)),
                    max_offchain_files=int(w.get("max_offchain_files", 100)),
                    asm_max_levels=int(w.get("asm_max_levels", 16)))
                warmed.append(f"{w['variable']} ({time.monotonic() - t0:.1f}s)")
            except Exception as exc:                # never fail the whole precompute on a warm error
                errors.append(f"{w.get('variable')}: {exc}")
        out["warmed"] = warmed
        if errors:
            out["warm_errors"] = errors
    return out


def build_only(job_id: str, blueprint_dir, asm_dir, graph_file=None, names=None) -> Dict[str, object]:
    """INCREMENTAL build of ONLY the named artifacts against an EXISTING index.db.

    Lets a system that already ran a full precompute (e.g. the Amex run) add NEW artifacts without
    redoing the whole corpus — it reuses the already-stored blobs (file_call_graph, name_to_stems,
    modifier index, source, …) and only (re)builds what's requested. Each artifact has its own builder
    in REGISTRY; new phases register here so `--only <name>` always covers them."""
    bp, src = Path(blueprint_dir), Path(asm_dir)
    gf = Path(graph_file) if graph_file else bp / "file_call_graph.json"

    from vbt.precompute import db_artifacts as DA
    DA.set_load_only(False)                 # build mode: find_asm_setters SCANS (not look up)
    try:
        from api.index_db.engine import set_skip_gvl_backfill
        set_skip_gvl_backfill(True)         # never need the 18GB gvl COUNT scans here
    except Exception:
        pass

    def _reverse_adj():
        from vbt.precompute.graph_db import build_and_store_reverse_adj
        return build_and_store_reverse_adj(job_id, bp, gf)

    def _forward_file_adj():
        from vbt.precompute.graph_db import build_and_store_forward_file_adj, preload_route_graph
        from vbt.reach.route import RouteEngine
        r = RouteEngine(bp, gf, src, job_id=job_id)
        preload_route_graph(r, job_id, bp, gf)
        return build_and_store_forward_file_adj(job_id, r)

    def _asm_setter_map():
        from vbt.precompute.setter_map_db import build_and_store_asm_setter_map
        return build_and_store_asm_setter_map(job_id, bp, src)

    def _cpp_setter_map():
        from vbt.precompute.cpp_setter_map_db import build_and_store_cpp_setter_map
        return build_and_store_cpp_setter_map(job_id, bp, src)

    def _cpp_file_writes():
        from vbt.precompute.cpp_file_writes_db import build_and_store_cpp_file_writes
        return build_and_store_cpp_file_writes(job_id, bp, src)

    def _fn_facts():
        from vbt.precompute.fn_facts_db import build_and_store_fn_facts
        return build_and_store_fn_facts(job_id, bp, src)

    def _fn_graph_adj():
        from vbt.precompute.graph_db import build_and_store_fn_graph_adj, preload_route_graph
        from vbt.reach.route import RouteEngine
        r = RouteEngine(bp, gf, src, job_id=job_id)
        preload_route_graph(r, job_id, bp, gf)
        return build_and_store_fn_graph_adj(job_id, r, src)

    def _cfg():
        return build_and_store_cfg_blobs(job_id, src)

    REGISTRY = {
        "reverse_adj": _reverse_adj,
        "forward_file_adj": _forward_file_adj,
        "asm_setter_map": _asm_setter_map,
        "cpp_setter_map": _cpp_setter_map,
        "cpp_file_writes": _cpp_file_writes,
        "fn_facts": _fn_facts,
        "fn_graph_adj": _fn_graph_adj,
        "cfg": _cfg,
        # future phases register their builder here: value_flow
    }
    out: Dict[str, object] = {}
    for name in (names or []):
        fn = REGISTRY.get(name)
        out[name] = fn() if fn else f"UNKNOWN (known: {sorted(REGISTRY)})"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vbt.precompute_db", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-id", required=True, help="index.db job id (jobs/<id>/index.db)")
    ap.add_argument("--blueprint-dir", required=True, type=Path)
    ap.add_argument("--asm-dir", required=True, type=Path)
    ap.add_argument("--graph", type=Path, default=None,
                    help="file_call_graph.json (default: <blueprint-dir>/file_call_graph.json)")
    ap.add_argument("--only", default=None,
                    help="INCREMENTAL: comma-separated artifacts to (re)build against an EXISTING "
                         "index.db, skipping the full precompute. E.g. "
                         "--only reverse_adj,forward_file_adj,asm_setter_map")
    # ---- OPTIONAL eager-warm flags: run a seed trace at precompute to fill the caches -------
    from vbt.trace import _hop                            # reuse the stem:type[:function] parser
    ap.add_argument("--warm-variable",
                    help="eager-warm: trace this variable now so the COLD first trace preloads "
                         "warm routes (~17s vs ~44s). Pass the SAME --warm-hop/caps you will trace.")
    ap.add_argument("--warm-language", default="cpp")
    ap.add_argument("--warm-hop", action="append", dest="warm_hops", type=_hop,
                    help="chain hop stem:type[:function] (repeatable, tail LAST) for the warm trace")
    ap.add_argument("--warm-max-dep-var-depth", type=int, default=1)
    ap.add_argument("--warm-max-paths", type=int, default=64)
    ap.add_argument("--warm-max-call-depth", type=int, default=16)
    ap.add_argument("--warm-max-routes", type=int, default=200)
    ap.add_argument("--warm-max-route-len", type=int, default=16)
    ap.add_argument("--warm-max-offchain-files", type=int, default=100)
    ap.add_argument("--warm-asm-max-levels", type=int, default=16)
    a = ap.parse_args(argv)
    warm = None
    if a.warm_variable:
        if not a.warm_hops:
            ap.error("--warm-variable requires at least one --warm-hop")
        warm = [{
            "variable": a.warm_variable, "language": a.warm_language, "hops": a.warm_hops,
            "max_dep_var_depth": a.warm_max_dep_var_depth, "max_paths": a.warm_max_paths,
            "max_call_depth": a.warm_max_call_depth, "max_routes": a.warm_max_routes,
            "max_route_len": a.warm_max_route_len, "max_offchain_files": a.warm_max_offchain_files,
            "asm_max_levels": a.warm_asm_max_levels,
        }]
    t0 = time.monotonic()
    if a.only:
        names = [n.strip() for n in a.only.split(",") if n.strip()]
        res = build_only(a.job_id, a.blueprint_dir, a.asm_dir, a.graph, names=names)
        print(f"=== VBT DB incremental build (job={a.job_id}, only={names}, "
              f"{time.monotonic() - t0:.1f}s) ===")
        for k, v in res.items():
            print(f"  {k:16}: {v}")
        return 0
    res = precompute_vbt_db(a.job_id, a.blueprint_dir, a.asm_dir, a.graph, warm=warm)
    print(f"=== VBT DB precompute (job={a.job_id}, {time.monotonic() - t0:.1f}s) ===")
    for k, v in res.items():
        print(f"  {k:16}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
