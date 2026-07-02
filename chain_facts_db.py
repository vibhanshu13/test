"""Standalone CHAINS-precompute step for the DB-only LCA chain trace.

Run AFTER ``vbt.precompute_db`` (NOT wired into it — ``precompute_db.py`` is untouched):

    python -m vbt.precompute.chain_facts_db <job_id> [--blueprint-dir D] [--asm-dir D]

In ONE pass it builds the artifacts that let ``vbt.lca_trace`` run FULLY DB-only (no
trace-time cfg/source/blueprint reads, no live-scan fallback), reusing IMPORTED primitives
from the existing modules so nothing existing is edited:

  (1) ``asm_setters_full:<stem>`` (+ ``asm_setters_full_coverage``) — the COMPLETE per-stem
      ASM setter map for DISCOVERY/LCA (#15). Unlike the legacy ``asm_setter_map`` (which is
      intentionally incomplete + falls back to a live scan), this one is complete *for the
      discovery flow* by construction: its per-stem universe is seeded from the modifier
      index (the SAME source ``lca_trace`` uses to pick candidate stems), so every var ever
      queried for a stem is in the universe ⇒ "var absent from an OK stem ⇒ no setter" is
      SOUND. Coverage status ``ok|empty|failed`` is tracked explicitly (the legacy builder
      collapses empty+failed to ``None``), so a build failure HARD-FAILS the trace instead
      of silently reporting "no setter".

  (2) ``cpp_setter_map_coverage`` (#2) — derived from the existing ``cpp_setter_map:*`` blobs
      (present blob ⇒ ``ok``) + re-running the imported ``_file_writes`` on absent-blob stems
      to split ``empty`` vs ``failed``. The cpp blobs themselves are NOT rebuilt.

  (3) ``chain_facts`` (#14) — distilled per-edge descend-guards + ``descend_line`` and
      per-(stem,scrutinee) constant assignments (both C++ and ASM), computed by INVOKING the
      same functions ``chain_feasibility`` (git c17a8f4) uses (``_prefix_guards`` per 2-hop
      edge — verified edge-independent — + ``find_*_setters_in_file`` + ``collect_*_conditions``),
      so the trace-time prune (``assemble_events_from_chain_facts`` → ``decide_feasibility``)
      equals the live verdict BY CONSTRUCTION. Conservative ⇒ need NOT be complete (a missing
      fact only reduces pruning; never a false drop).

Trace consumes these via the load-only-gated accessors here:
``get_asm_setters_full`` / ``set_asm_setters_full_job`` and
``get_chain_facts`` / ``set_chain_facts_job`` (+ ``load_coverage``).
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from vbt.precompute import db_artifacts as DA
from vbt.interfaces import SetterSite
from backward_traversal.utils.token_utils import normalize_token
from vbt.lca_feasibility import _is_constant, parse_required_guard

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# exception (NEW): a required index.db artifact is absent or stale → hard-fail.
# --------------------------------------------------------------------------- #
class DbArtifactMissing(RuntimeError):
    """A required ``index.db`` artifact is absent or version-stale — the trace must HARD-FAIL
    with an actionable 'rerun precompute' message rather than read source / live-compute."""


class _EdgeTimeout(Exception):
    """Raised by the SIGALRM handler to bound a single ``_prefix_guards`` edge computation."""


def _alarm_handler(signum, frame):
    raise _EdgeTimeout()


# --------------------------------------------------------------------------- #
# artifact names + versions
# --------------------------------------------------------------------------- #
ASM_SETTERS_FULL_ARTIFACT = "asm_setters_full"                  # per-stem blob: f"{ART}:{stem}"
ASM_SETTERS_FULL_COVERAGE_ARTIFACT = "asm_setters_full_coverage"   # single blob {stem: status}
CPP_SETTER_MAP_COVERAGE_ARTIFACT = "cpp_setter_map_coverage"       # single blob {stem: status}
CHAIN_FACTS_ARTIFACT = "chain_facts"                           # single blob

ASM_SETTERS_FULL_VERSION = 1
COVERAGE_VERSION = 1
CHAIN_FACTS_VERSION = 4   # 4: GAP 1 — precomputed parsed value_guards from safe compound guard decomposition (invalidates v3 rows)
                          # 3: GAP 2 — C++ search-loop match conditions add loop_break guards/events (invalidates v2 rows)
                          # 2: _prefix_guards no longer treats unrelated caller calls as descend sites on file-only hops (Issue 2)

# coverage status values
OK, EMPTY, FAILED = "ok", "empty", "failed"


# --------------------------------------------------------------------------- #
# in-process load caches + job hooks (mirror fn_facts_db / setter_map_db)
# --------------------------------------------------------------------------- #
_ASF_LOADED: Dict[tuple, object] = {}      # (job,stem) -> dict | None | sentinel 0
_ASF_LOADED_MAX = 4096
_ASF_JOB: Optional[str] = None

_CF_LOADED: Dict[str, object] = {}         # job -> dict | None | sentinel 0
_CF_JOB: Optional[str] = None

_COV_LOADED: Dict[tuple, object] = {}      # (job,artifact) -> dict | None | sentinel 0


def set_asm_setters_full_job(job_id: Optional[str]) -> None:
    """Arm (or disarm) the COMPLETE ASM setter-map lookups for this trace (#15)."""
    global _ASF_JOB
    _ASF_JOB = job_id


def set_chain_facts_job(job_id: Optional[str]) -> None:
    """Arm (or disarm) chain_facts lookups for this trace's conservative prune (#14)."""
    global _CF_JOB
    _CF_JOB = job_id


# =========================================================================== #
# (1) asm_setters_full — COMPLETE per-stem ASM setter map (#15)
# =========================================================================== #
def _build_one_asm_setters_full(args):
    """ProcessPool worker: build the COMPLETE {normalized_var: [SetterSite-dict]} map for one
    ASM stem + its coverage status. Input tuple: (stem, sorted_seed_vars, bp_str, src_str).

    Universe U = modifier-index seed ∪ FLIPC/implicit-write augment (reused VERBATIM from the
    legacy ``_augment_universe``) ∪ register-indirect resolved dest fields. The seed is the
    modifier-index inverse, i.e. EXACTLY the vars ``lca_trace`` can query for this stem, so U
    ⊇ every queryable var ⇒ a var absent from a built map provably has no setter here.

    Returns (stem, payload_or_None, status) where status ∈ {ok, empty, failed}. ``failed``
    means the stem's blueprint could not be resolved/loaded OR a per-var scan raised — i.e. we
    could NOT certify completeness, so the trace must hard-fail rather than assume 'no setter'."""
    try:
        from pathlib import Path as _Path
        from dataclasses import asdict as _asdict
        from backward_traversal.utils.token_utils import normalize_token as _nt
        from backward_traversal.utils.blueprint_utils import resolve_asm_blueprint, load_json
        from vbt.setters.asm_setters import find_asm_setters_in_file, find_asm_indirect_dest_sites
        from vbt.precompute.setter_map_db import _augment_universe
        from vbt.precompute import db_artifacts as _DA

        stem, seed_vars, bp_str, src_str = args
        bp = _Path(bp_str)
        src = _Path(src_str)

        # Stem-level prerequisite: a resolvable + loadable blueprint. Without it
        # find_asm_setters_in_file silently returns [] (→ would look like 'empty'), which would
        # be UNSOUND in strict DB-only mode — so treat it as FAILED.
        try:
            bpp = resolve_asm_blueprint(stem, bp)
            if not bpp:
                return (stem, None, FAILED)
            load_json(bpp)
        except Exception:
            return (stem, None, FAILED)

        vars_: Set[str] = set(seed_vars)
        vars_ = _augment_universe(stem, vars_, bp, src)            # FLIPC + 4 implicit collectors
        try:                                                       # register-indirect dest fields
            for s in find_asm_indirect_dest_sites(stem, bp, src):
                if s.variable:
                    vars_.add(s.variable)
        except Exception:
            pass

        m: Dict[str, list] = {}
        had_error = False
        for var in sorted(vars_):
            try:
                sites = find_asm_setters_in_file(var, stem, bp, src)
            except Exception:
                had_error = True
                continue
            if sites:
                m[_nt(var).upper()] = [_asdict(s) for s in sites]

        if had_error:
            return (stem, _DA.dumps_gz(m) if m else None, FAILED)
        if not m:
            return (stem, None, EMPTY)
        return (stem, _DA.dumps_gz(m), OK)
    except Exception:
        try:
            return (args[0], None, FAILED)
        except Exception:
            return ("", None, FAILED)


def build_and_store_asm_setters_full(job_id, blueprint_dir, asm_dir) -> Dict[str, int]:
    """Build + store the COMPLETE per-stem ASM setter map + its coverage. Returns counts."""
    from vbt.precompute.modifier_index import get_modifier_index
    bp, src = Path(blueprint_dir), Path(asm_dir)
    midx = get_modifier_index(bp, src)
    # seed = modifier-index inverse: {stem: {vars whose setter dest is this stem}}
    stem_vars: Dict[str, set] = {}
    for var, files in midx.asm.items():
        for stem in files:
            stem_vars.setdefault(stem, set()).add(var)
    # cover EVERY asm source stem too (a candidate stem with no modifier entries ⇒ EMPTY,
    # never ABSENT → never a spurious hard-fail).
    all_stems = sorted(set(stem_vars) | {p.stem for p in src.glob("*.asm")})

    DA.clear_blobs_prefix(job_id, ASM_SETTERS_FULL_ARTIFACT + ":")
    items = [(stem, sorted(stem_vars.get(stem, set())), str(bp), str(src)) for stem in all_stems]
    try:
        from vbt.precompute.parallel import parallel_map
        results = parallel_map(_build_one_asm_setters_full, items)
    except Exception as exc:
        logger.debug("parallel asm_setters_full fallback: %s", exc)
        results = [_build_one_asm_setters_full(it) for it in items]

    coverage: Dict[str, str] = {}
    n_blobs = 0
    for stem, payload, status in results:
        coverage[stem] = status
        if payload is not None:
            try:
                if DA.write_blob(job_id, f"{ASM_SETTERS_FULL_ARTIFACT}:{stem}", payload):
                    n_blobs += 1
            except Exception as exc:
                logger.debug("store asm_setters_full:%s failed: %s", stem, exc)

    DA.write_blob(job_id, ASM_SETTERS_FULL_COVERAGE_ARTIFACT, DA.dumps_gz(coverage))
    DA.write_manifest(job_id, ASM_SETTERS_FULL_COVERAGE_ARTIFACT, COVERAGE_VERSION,
                      DA.source_manifest_hash(sorted(src.glob("*.asm")), version=ASM_SETTERS_FULL_VERSION))
    # §16 #1 (same contract as precompute_vbt_db): asm_setters_full is a TRACE INPUT — traces
    # serve ASM setters from these blobs. Any whole-trace result cached BEFORE this (re)build
    # may no longer match a fresh compute, so it must never be served again.
    try:
        from vbt.precompute.trace_cache import clear as _clear_trace_cache
        n = _clear_trace_cache(job_id)
        if n:
            logger.info("asm_setters_full rebuild invalidated %d cached trace result(s)", n)
    except Exception as exc:
        logger.debug("trace-cache invalidation after asm_setters_full failed: %s", exc)
    return {
        "stems": len(all_stems), "blobs": n_blobs,
        "ok": sum(1 for v in coverage.values() if v == OK),
        "empty": sum(1 for v in coverage.values() if v == EMPTY),
        "failed": sum(1 for v in coverage.values() if v == FAILED),
    }


# =========================================================================== #
# (2) cpp_setter_map_coverage (#2) — derived; the cpp blobs are NOT rebuilt
# =========================================================================== #
def build_and_store_cpp_setter_map_coverage(job_id, blueprint_dir, asm_dir) -> Dict[str, int]:
    """Derive per-stem coverage for the EXISTING cpp_setter_map blobs without editing the
    cpp builder.

    The AUTHORITATIVE source of ``ok`` is the set of blobs that actually exist in the DB — a
    stored ``cpp_setter_map:<stem>`` blob ⇒ ``ok``.  Previously coverage was driven purely by a
    filesystem ``glob('*.cpp')`` over ``asm_dir``: if that dir was wrong/empty/mismatched (a
    different layout than the main precompute used) the coverage came out EMPTY even though the
    real blobs existed, which made EVERY trace hard-fail with "coverage ABSENT".  Deriving ``ok``
    from the blobs makes coverage self-consistent with them regardless of ``asm_dir``.

    The source glob is now used ONLY to classify the blob-LESS stems: a ``.cpp`` with write-sites
    but no stored blob ⇒ drift ⇒ ``failed``; with no write-sites ⇒ ``empty``."""
    from vbt.setters.cpp_setters import _file_writes
    from vbt.precompute.cpp_setter_map_db import CPP_SETTER_MAP_ARTIFACT
    src = Path(asm_dir)
    coverage: Dict[str, str] = {}

    # (a) Every stem with a stored blob is OK — independent of the source dir.
    prefix = f"{CPP_SETTER_MAP_ARTIFACT}:"
    blob_keys = DA.list_blob_keys_prefix(job_id, prefix)
    for key in blob_keys:
        stem = key[len(prefix):]
        if stem:
            coverage[stem] = OK

    # (b) Classify the remaining .cpp source files that have NO stored blob.
    globbed = sorted(src.glob("*.cpp"))
    for cpp in globbed:
        stem = cpp.stem
        if coverage.get(stem) == OK:
            continue
        try:
            writes = _file_writes(cpp)
        except Exception:
            coverage[stem] = FAILED
            continue
        tails = [lc[-1] for (lc, _s) in writes if lc and lc[-1]]
        # A non-empty write set with NO stored blob means the cpp_setter_map is stale vs the
        # source (drift) — fail loudly rather than silently report 'no setter'.
        coverage[stem] = FAILED if tails else EMPTY

    # Surface a likely asm_dir mismatch loudly instead of silently mis-covering: real blobs
    # exist but the source dir yielded no .cpp files to cross-check.
    if blob_keys and not globbed:
        logger.warning(
            "cpp_setter_map_coverage: asm_dir %s has no .cpp files but %d cpp_setter_map blob(s) "
            "exist — coverage derived from blobs only (verify --asm-dir for drift detection).",
            src, len(blob_keys))

    DA.write_blob(job_id, CPP_SETTER_MAP_COVERAGE_ARTIFACT, DA.dumps_gz(coverage))
    DA.write_manifest(job_id, CPP_SETTER_MAP_COVERAGE_ARTIFACT, COVERAGE_VERSION,
                      DA.source_manifest_hash(globbed, version=COVERAGE_VERSION))
    return {
        "stems": len(coverage),
        "ok": sum(1 for v in coverage.values() if v == OK),
        "empty": sum(1 for v in coverage.values() if v == EMPTY),
        "failed": sum(1 for v in coverage.values() if v == FAILED),
    }


# =========================================================================== #
# (3) chain_facts (#14) — per-edge descend-guards + per-(stem,scrut) constants
# =========================================================================== #
def _edge_key(caller_stem: str, caller_fn: str, callee_stem: str, callee_fn: str) -> str:
    return "\t".join((caller_stem, caller_fn, callee_stem, callee_fn))


def _node_facts_key(stem: str, scrut: str) -> str:
    return stem + "\t" + scrut


def _setter_reverse_indices(job_id, src: Path):
    """Build COMPLETE reverse indices tail→{stems} (C++) and key→{stems} (ASM) from the per-stem
    setter-map blobs. This matches EXACTLY the stems where ``find_*_setters_in_file`` returns sites
    (the blobs ARE that output), so phase-B coverage is COMPLETE per scrutinee — every assignment
    (incl. a CLEARING one) is captured, which the soundness contract requires (a missing clear could
    otherwise let a pin persist → a false drop). Falls back to the modifier index per-scrut only if a
    blob is unexpectedly absent."""
    from vbt.precompute.cpp_setter_map_db import CPP_SETTER_MAP_ARTIFACT
    cpp_rev: Dict[str, set] = {}
    asm_rev: Dict[str, set] = {}
    for cpp in src.glob("*.cpp"):
        payload = DA.read_blob(job_id, f"{CPP_SETTER_MAP_ARTIFACT}:{cpp.stem}")
        if payload is None:
            continue
        for tail in DA.loads_gz(payload).keys():
            cpp_rev.setdefault(tail, set()).add(cpp.stem)
    for a in src.glob("*.asm"):
        payload = DA.read_blob(job_id, f"{ASM_SETTERS_FULL_ARTIFACT}:{a.stem}")
        if payload is None:
            continue
        for key in DA.loads_gz(payload).keys():
            asm_rev.setdefault(key, set()).add(a.stem)
    return cpp_rev, asm_rev


# --------------------------------------------------------------------------- #
# ProcessPool workers (module-level → picklable under the 'fork' executor).
# Each worker re-establishes the SAME load-only / cfg-DB / setter-map routing the serial
# parent uses, then runs ONLY O(local) parses (no full-graph load), so its per-edge /
# per-(stem,scrut) facts are BYTE-IDENTICAL to the serial computation. Results are flat
# (key, value) tuple lists; the parent reassembles them in strict canonical order.
# --------------------------------------------------------------------------- #
def _process_stem_edges_worker(args):
    """Phase-A worker: compute the descend-guard facts for EVERY fn-graph edge whose CALLER is
    in a GROUP of stems. Returns ``([(_edge_key, edge_fact), ...], n_timeouts, diag)``.

    One task = ``plan.batch_size`` caller stems (amortises the per-task setup at 22k-file scale;
    caches are keyed by stem so grouping cannot change any per-edge result). ``node_types`` is
    pre-sliced by the parent to just the stems this group's edges touch — shipping the full
    corpus-wide dict in every task pickle is what stalled phase A on large graphs.

    Identical to the serial loop body (``_prefix_guards`` per 2-hop edge, ``skip_fallback=True``,
    SIGALRM per-edge bound). A ``job_id=None`` RouteEngine is used on purpose: with skip_fallback
    the only ``route`` access is ``cpp_call_edges(stem)`` (reads the blueprint directly, job-id
    independent) and the ASM caller bridge — neither needs the 80MB graph, so the child never
    loads it. cfg/setter reads route through the cfg-DB exactly as in the parent."""
    (stem_groups, job_id, blueprint_dir, asm_dir, graph_file,
     node_types, edge_timeout) = args

    from vbt.engine import Hop, _prefix_guards
    from vbt.output.codeblocks import CodeBlockStore
    from vbt.reach.route import RouteEngine
    from vbt.precompute.cfg_db import install_cfg_db
    from vbt.precompute.chain_facts_db import _edge_key, _EdgeTimeout, _alarm_handler
    from vbt.precompute.guard_decompose import decompose_value_guards_diag   # GAP 1
    from vbt.precompute import db_artifacts as DA
    import signal

    # Prevent a Clang live-compile storm: a cfg-DB miss degrades to empty, never a subprocess.
    DA.set_load_only(True)
    install_cfg_db(job_id)

    bp, src = Path(blueprint_dir), Path(asm_dir)
    gf = Path(graph_file)
    route = RouteEngine(bp, gf, src, job_id=None)   # job_id=None ⇒ no full-graph load (see docstring)

    cfg_cache: Dict[str, list] = {}
    asm_bp_cache: Dict[str, Dict] = {}
    store = CodeBlockStore(src)

    def _hop(st: str, fn: str):
        ftype = node_types.get(st, "asm")
        return Hop(st, ftype, fn if ftype == "cpp" else None)

    edge_facts_list = []
    timeouts = 0
    # GAP 1 decomposition diagnostics (per-stem; summed in the parent for the result dict).
    diag = {"compound_guards_seen": 0, "compound_guards_decomposed": 0,
            "value_guards_emitted": 0, "unsafe_compound_guards_kept_unknown": 0,
            "decomposition_cap_hits": 0}

    _signal_active = hasattr(signal, "SIGALRM")
    old_handler = None
    if _signal_active:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        # Group order is sorted-by-stem and each stem's edges are sorted, so this flat
        # iteration matches the serial sorted_edges order within the group.
        for cs, cf, callee_s, callee_f in (e for _s, es in stem_groups for e in es):
            hops = [_hop(cs, cf), _hop(callee_s, callee_f)]
            if _signal_active:
                signal.setitimer(signal.ITIMER_REAL, edge_timeout)
            try:
                guard_conds, bounds = _prefix_guards(hops, route, cfg_cache, src, store,
                                                     bp, asm_bp_cache, skip_fallback=True)
            except _EdgeTimeout:
                timeouts += 1
                continue
            except Exception:
                continue
            finally:
                if _signal_active:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            guards = []
            value_guards = []
            seen_vg: Set[Tuple] = set()
            for cond, _via, _blk in guard_conds:
                text = cond.condition or ""
                lang = "asm" if (cond.raw_test or str(cond.location.file).endswith(".asm")) else "cpp"
                line = int(getattr(cond.location, "start_line", 0) or 0)
                guards.append({"text": text, "line": line, "lang": lang})
                # GAP 1: decompose into SOUND simple value-guard facts (precompute-only — the
                # only place source-derived condition text is parsed for feasibility). A simple
                # guard yields one fact; a safe compound (top-level && conjuncts + homogeneous
                # equality || disjunctions) yields several; anything uncertain yields none.
                facts, capped = decompose_value_guards_diag(text)
                is_compound = ("&&" in text) or ("||" in text)
                if is_compound:
                    diag["compound_guards_seen"] += 1
                    diag["compound_guards_decomposed"] += 1 if facts else 0
                    diag["unsafe_compound_guards_kept_unknown"] += 0 if facts else 1
                    diag["decomposition_cap_hits"] += 1 if capped else 0
                for f in facts:
                    vg = {"scrutinee": f["scrutinee"], "const": f["const"], "op": f["op"],
                          "line": line, "lang": lang, "source_text": f["text"]}
                    dk = (line, lang, vg["scrutinee"], vg["const"], vg["op"], vg["source_text"])
                    if dk in seen_vg:          # dedup per edge (deterministic, first-occurrence)
                        continue
                    seen_vg.add(dk)
                    value_guards.append(vg)
                    diag["value_guards_emitted"] += 1
            edge_facts_list.append((_edge_key(cs, cf, callee_s, callee_f), {
                "guards": guards,
                "value_guards": value_guards,
                "descend_line": bounds.get(cs),
            }))
    finally:
        if _signal_active and old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)

    return edge_facts_list, timeouts, diag


def _process_scrutinee_worker(args):
    """Phase-B worker: the UNFILTERED constant assignments to one scrutinee across its candidate
    stems. Returns ``[(_node_facts_key, assigns), ...]``. Identical to the serial per-scrutinee
    body; ``cand_stems`` is computed once by the parent and passed in (so a worker never rebuilds
    the modifier index / reverse indices)."""
    (scrut, lang, job_id, blueprint_dir, asm_dir, max_scrut_fanout, cand_stems) = args

    from vbt.precompute import db_artifacts as DA
    from vbt.precompute.cfg_db import install_cfg_db
    from vbt.precompute.chain_facts_db import _node_facts_key, _is_constant
    from vbt.setters.cpp_setters import (find_cpp_setters_in_file,
                                         set_cpp_setter_map_job, set_cpp_file_writes_job)
    from vbt.setters.asm_setters import find_asm_setters_in_file, set_asm_setter_map_job
    from vbt.conditions.cpp_conditions import collect_cpp_conditions
    from vbt.conditions.asm_conditions import collect_asm_conditions
    from vbt.cpp_frontend.wrapper import run_cfg_extract
    from backward_traversal.utils.blueprint_utils import resolve_asm_blueprint, load_json

    bp, src = Path(blueprint_dir), Path(asm_dir)
    DA.set_load_only(True)
    install_cfg_db(job_id)
    set_cpp_setter_map_job(job_id); set_cpp_file_writes_job(job_id); set_asm_setter_map_job(job_id)

    cfg_cache: Dict[str, list] = {}
    node_facts_list = []

    if lang == "cpp":
        tail = scrut.split(".")[-1].split("->")[-1]
        fp = [scrut] if ("." in scrut or "->" in scrut) else None
        for stem in cand_stems:
            cpp = src / f"{stem}.cpp"
            if not cpp.exists():
                continue
            cfg = cfg_cache.get(stem)
            if cfg is None:
                try:
                    cfg = run_cfg_extract(str(cpp))
                except Exception:
                    cfg = []
                cfg_cache[stem] = cfg
            assigns = []
            for s in find_cpp_setters_in_file(tail, cpp, full_paths=fp):
                if s.is_declaration and not s.value:
                    continue
                val = (s.value or "").strip()
                conditional = bool(collect_cpp_conditions(cpp, s.line, s.function, functions=cfg))
                assigns.append({"line": int(s.line or 0),
                                "const": (val if _is_constant(val) else None),
                                "conditional": conditional})
            if assigns:
                node_facts_list.append((_node_facts_key(stem, scrut), assigns))
    else:
        for stem in cand_stems:
            try:
                bp_path = str(resolve_asm_blueprint(stem, bp))
                bp_data = load_json(bp_path)
            except Exception:
                continue
            assigns = []
            for s in find_asm_setters_in_file(scrut, stem, bp, src):
                val = (s.value or "").strip()
                conds, _ = collect_asm_conditions(bp_data, bp_path, s.block_id, s.line,
                                                  f"{stem}.asm")
                assigns.append({"line": int(s.line or 0),
                                "const": (val if _is_constant(val) else None),
                                "conditional": bool(conds)})
            if assigns:
                node_facts_list.append((_node_facts_key(stem, scrut), assigns))

    return node_facts_list


def build_and_store_chain_facts(job_id, blueprint_dir, asm_dir, graph_file=None,
                                *, progress: bool = True, max_scrut_fanout: int = 200,
                                edge_timeout: float = 4.0) -> Dict[str, int]:
    """Build + store the per-edge guard/descend-line facts + per-(stem,scrutinee) constant
    assignments, computed with source available by INVOKING the exact functions
    ``chain_feasibility`` (c17a8f4) uses — so the trace-time replay equals the live verdict by
    construction.

    PARALLEL (byte-identical): phase A fans the edges out by CALLER stem across a process pool;
    phase B fans the scrutinees out. Each worker re-establishes the load-only / cfg-DB / setter-map
    routing and runs only O(local) parses, returning flat ``(key, value)`` tuples. The parent
    REASSEMBLES ``edge_facts`` / ``scrutinees`` / ``node_facts`` in the SAME strict sorted order the
    serial loop produced, so the gzipped blob is identical regardless of worker count or completion
    order. Both fan-outs degrade to an in-process serial map if the pool can't start (cold-cliff
    guard), which is itself byte-identical."""
    from vbt.reach.route import RouteEngine
    from vbt.precompute.call_graph import ensure_call_graph
    from vbt.precompute.graph_db import load_fn_graph_adj, load_reverse_adj
    from vbt.precompute.cfg_db import install_cfg_db, clear_cfg_db
    from vbt.precompute.modifier_index import get_modifier_index
    from vbt.precompute.parallel import plan_workers, parallel_map

    def _p(msg: str) -> None:
        if progress:
            print(f"[chain_facts] {msg}", flush=True)

    bp, src = Path(blueprint_dir), Path(asm_dir)
    gf = Path(graph_file) if graph_file else (bp / "file_call_graph.json")
    ensure_call_graph(bp, src, out_path=gf)

    route = RouteEngine(bp, gf, src, job_id=job_id)
    # node_types DB-only via reverse_adj (avoids a RouteEngine graph rebuild).
    rev = load_reverse_adj(job_id)
    node_types = rev[1] if rev else route.node_types()

    adj = load_fn_graph_adj(job_id)
    if adj is None:
        raise DbArtifactMissing(
            f"fn_graph_adj absent for job {job_id} — run `python -m vbt.precompute_db` first")

    install_cfg_db(job_id)
    # Route setter lookups through the DB blobs (built by precompute_db / step 1) so phase B does
    # NOT re-run the tree-sitter / blueprint setter SCAN per (scrut, stem) — a big speedup, and
    # byte-identical (the blobs ARE the find_*_setters output). collect_*_conditions still reads
    # source (the conditionality fact), which is on disk at precompute time.
    from vbt.setters.cpp_setters import set_cpp_setter_map_job, set_cpp_file_writes_job
    from vbt.setters.asm_setters import set_asm_setter_map_job
    _prev_lo = DA.is_load_only()
    DA.set_load_only(True)
    set_cpp_setter_map_job(job_id); set_cpp_file_writes_job(job_id); set_asm_setter_map_job(job_id)
    try:
        from vbt.cpp_frontend import wrapper as _w
        _w._MEMO.clear()
    except Exception:
        pass

    plan = plan_workers()
    _p(f"workers: {plan.worker_count} (batch {plan.batch_size})")

    try:
        # ---- (a) per directed fn-graph edge: descend guards + descend_line --------------- #
        # skip_fallback=True so an ASM edge whose direct caller-bridge lookup misses returns
        # no-guard immediately (never the O(corpus) route.routes caller-walk). The per-edge
        # SIGALRM bound (inside each worker) skips a residual heavy edge (sound, #7: a missing
        # edge ⇒ no-guards/no-bound ⇒ KEEP).
        edges: Set[Tuple[str, str, str, str]] = set()
        for (cs, cf), outs in adj.items():
            for (callee_s, callee_f, _line, _cross) in outs:
                edges.add((cs, cf, callee_s, callee_f))
        sorted_edges = sorted(edges)
        _p(f"phase A: {len(sorted_edges)} distinct fn-graph edges")
        _t_a = time.monotonic()

        # Group edges by CALLER stem so a worker shares one route/cfg/store cache over all of a
        # stem's edges (mirrors the serial shared cache, scoped per stem), then batch
        # ``plan.batch_size`` stems per task. Two scale requirements at 22k files / 500k+ edges:
        # (1) each task's pickle must carry only the node_types entries its own edges touch —
        # the corpus-wide dict × thousands of tasks serialises gigabytes through the pool queue
        # and starves the workers; (2) task count must stay in the hundreds so per-task setup
        # (imports, cfg-DB install, RouteEngine) is amortised. Worker order is irrelevant —
        # the parent reassembles below.
        from collections import defaultdict
        edges_by_stem: Dict[str, list] = defaultdict(list)
        for edge in sorted_edges:
            edges_by_stem[edge[0]].append(edge)
        stem_items = sorted(edges_by_stem.items())
        group_size = max(1, plan.batch_size)
        map_args = []
        for i in range(0, len(stem_items), group_size):
            group = stem_items[i:i + group_size]
            touched = {s for _stem, elist in group for e in elist for s in (e[0], e[2])}
            nt_slice = {s: node_types[s] for s in touched if s in node_types}
            map_args.append((group, job_id, str(bp), str(src), str(gf), nt_slice, edge_timeout))

        _hb = {"t": time.monotonic()}

        def _progress_a(done: int, total: int) -> None:
            now = time.monotonic()
            if now - _hb["t"] >= 15.0 or done == total:
                _hb["t"] = now
                _p(f"phase A progress: {done}/{total} stem-groups")

        try:
            results = parallel_map(_process_stem_edges_worker, map_args, plan=plan,
                                   on_progress=_progress_a)
        except Exception as exc:
            logger.warning("parallel phase A failed; serial fallback: %s", exc)
            results = [_process_stem_edges_worker(arg) for arg in map_args]

        collected_edges: Dict[str, dict] = {}
        _n_timeout = 0
        _diag = {"compound_guards_seen": 0, "compound_guards_decomposed": 0,
                 "value_guards_emitted": 0, "unsafe_compound_guards_kept_unknown": 0,
                 "decomposition_cap_hits": 0}
        for edge_list, timeouts, stem_diag in results:
            _n_timeout += timeouts
            for _dk in _diag:
                _diag[_dk] += stem_diag.get(_dk, 0)
            for k, v in edge_list:
                collected_edges[k] = v

        # Deterministic reassembly: iterate sorted_edges so edge_facts insertion order (and the
        # scrutinee discovery order) matches the serial loop EXACTLY → identical gz bytes.
        edge_facts: Dict[str, dict] = {}
        scrutinees: Dict[str, str] = {}            # scrut -> language ("cpp"|"asm")
        for edge in sorted_edges:
            key = _edge_key(*edge)
            ef = collected_edges.get(key)
            if ef is None:
                continue
            edge_facts[key] = ef
            # GAP 1: discover scrutinees from the PARSED value_guards — a SUPERSET of the old
            # simple-guard scan (now includes scrutinees recovered from safe compound guards,
            # e.g. transactionCode from the CID gate). Deterministic: value_guards are in guard
            # order within an edge and edges are iterated in sorted order.
            for vg in ef.get("value_guards") or []:
                scrutinees.setdefault(vg["scrutinee"], vg.get("lang") or "cpp")

        _p(f"phase A done in {time.monotonic() - _t_a:.1f}s — {len(edge_facts)} edge-facts, "
           f"|S|={len(scrutinees)} scrutinees, {_n_timeout} edge-timeouts")
        _p(f"  GAP1 decompose: {_diag['compound_guards_seen']} compound seen, "
           f"{_diag['compound_guards_decomposed']} decomposed, {_diag['value_guards_emitted']} value_guards, "
           f"{_diag['unsafe_compound_guards_kept_unknown']} kept-unknown, "
           f"{_diag['decomposition_cap_hits']} cap-hits")

        # ---- (b) per (stem, scrutinee): UNFILTERED constant assignments ------------------- #
        # Mirror chain_feasibility's collection EXACTLY; the chain-specific descend_line filter
        # is applied at TRACE (assemble_events_from_chain_facts), not here.
        from backward_traversal.utils.token_utils import normalize_token as _nt
        midx = get_modifier_index(bp, src)
        cpp_rev, asm_rev = _setter_reverse_indices(job_id, src)   # COMPLETE coverage (#7 soundness)

        def _cand_stems(scrut: str, lang: str):
            if lang == "cpp":
                tail = scrut.split(".")[-1].split("->")[-1]
                stems = cpp_rev.get(tail)
                return sorted(stems) if stems is not None else sorted(midx.files_for(scrut, "cpp"))
            stems = asm_rev.get(_nt(scrut).upper())
            return sorted(stems) if stems is not None else sorted(midx.files_for(scrut, "asm"))

        _t_b = time.monotonic()
        # Build the per-scrutinee work list in sorted order; apply the fanout cap HERE (parent) so
        # both the work-list and the reassembly skip the same scrutinees (SOUND, #7: a missing fact
        # ⇒ KEEP). Skip + log so the bound is diagnosable.
        scrut_args = []
        _capped = 0
        for scrut, lang in sorted(scrutinees.items()):
            cand = _cand_stems(scrut, lang)
            if len(cand) > max_scrut_fanout:
                _capped += 1
                if progress:
                    _p(f"  cap: skip scrut {scrut!r} ({lang}) fanout={len(cand)} > {max_scrut_fanout}")
                continue
            scrut_args.append((scrut, lang, job_id, str(bp), str(src), max_scrut_fanout, cand))

        _hb_b = {"t": time.monotonic()}

        def _progress_b(done: int, total: int) -> None:
            now = time.monotonic()
            if now - _hb_b["t"] >= 15.0 or done == total:
                _hb_b["t"] = now
                _p(f"phase B progress: {done}/{total} scrutinees")

        try:
            results_b = parallel_map(_process_scrutinee_worker, scrut_args, plan=plan,
                                     on_progress=_progress_b)
        except Exception as exc:
            logger.warning("parallel phase B failed; serial fallback: %s", exc)
            results_b = [_process_scrutinee_worker(arg) for arg in scrut_args]

        collected_nodes: Dict[str, list] = {}
        for fact_list in results_b:
            for k, v in fact_list:
                collected_nodes[k] = v

        # Deterministic reassembly: same (scrutinee, cand-stem) iteration order as the serial loop.
        node_facts: Dict[str, list] = {}
        for scrut, lang in sorted(scrutinees.items()):
            cand = _cand_stems(scrut, lang)
            if len(cand) > max_scrut_fanout:
                continue
            for stem in cand:
                key = _node_facts_key(stem, scrut)
                v = collected_nodes.get(key)
                if v is not None:
                    node_facts[key] = v

        _p(f"phase B done in {time.monotonic() - _t_b:.1f}s — {len(node_facts)} node-facts, "
           f"capped {_capped} high-fanout scrutinees")
    finally:
        clear_cfg_db()
        DA.set_load_only(_prev_lo)
        set_cpp_setter_map_job(None); set_cpp_file_writes_job(None); set_asm_setter_map_job(None)
        # Drop the AST/CFG condition memo built during this run (bounds long-lived processes).
        try:
            from vbt.conditions.cpp_conditions import clear_cpp_conditions_cache
            clear_cpp_conditions_cache()
        except Exception:
            pass

    facts = {"edges": edge_facts, "scrutinees": scrutinees, "nodes": node_facts}
    _CF_LOADED[job_id] = facts
    if DA.write_blob(job_id, CHAIN_FACTS_ARTIFACT, DA.dumps_gz(facts)):
        DA.write_manifest(job_id, CHAIN_FACTS_ARTIFACT, CHAIN_FACTS_VERSION,
                          DA.source_manifest_hash(
                              sorted(src.glob("*.cpp")) + sorted(src.glob("*.asm")),
                              version=CHAIN_FACTS_VERSION))
    return {"edges": len(edge_facts), "scrutinees": len(scrutinees),
            "node_facts": len(node_facts), "capped_scrutinees": _capped,
            "edge_timeouts": _n_timeout, **_diag}


# =========================================================================== #
# trace-time accessors (load-only-gated reads; mirror fn_facts_db / setter_map_db)
# =========================================================================== #
def get_asm_setters_full(stem: str, variable: str) -> Optional[List[SetterSite]]:
    """COMPLETE-map setter sites for (stem, variable), or None when the stem has no blob OR the
    variable is absent from it. Caller (lca_trace) decides via coverage whether None means
    'genuinely no setter' (OK/EMPTY) or 'incomplete build' (FAILED/absent) — this NEVER live-scans."""
    if not _ASF_JOB:
        return None
    key = (_ASF_JOB, stem)
    m = _ASF_LOADED.get(key, 0)
    if m == 0:
        payload = DA.read_blob(_ASF_JOB, f"{ASM_SETTERS_FULL_ARTIFACT}:{stem}")
        m = DA.loads_gz(payload) if payload is not None else None
        if len(_ASF_LOADED) >= _ASF_LOADED_MAX:
            _ASF_LOADED.pop(next(iter(_ASF_LOADED)))
        _ASF_LOADED[key] = m
    if not m:
        return None
    rows = m.get(normalize_token(variable).upper())
    if rows is None:
        return None
    out: List[SetterSite] = []
    for d in rows:
        s = SetterSite(**d)
        s.variable = variable                  # byte-identity: return the queried form
        out.append(s)
    return out


def load_coverage(job_id: str, artifact: str) -> Optional[Dict[str, str]]:
    """The {stem: status} coverage dict for a coverage artifact, or None if absent."""
    key = (job_id, artifact)
    m = _COV_LOADED.get(key, 0)
    if m == 0:
        payload = DA.read_blob(job_id, artifact)
        m = DA.loads_gz(payload) if payload is not None else None
        _COV_LOADED[key] = m
    return m


def get_chain_facts() -> Optional[dict]:
    """The chain_facts dict for the armed job, or None if absent."""
    if not _CF_JOB:
        return None
    m = _CF_LOADED.get(_CF_JOB, 0)
    if m == 0:
        payload = DA.read_blob(_CF_JOB, CHAIN_FACTS_ARTIFACT)
        m = DA.loads_gz(payload) if payload is not None else None
        _CF_LOADED[_CF_JOB] = m
    return m


# --------------------------------------------------------------------------- #
# trace-time prune: reconstruct the chain_feasibility event list from chain_facts
# --------------------------------------------------------------------------- #
def _parse_node(n: str) -> Tuple[str, str]:
    """A chain node-id → (stem, fn). C++ ``stem::fn`` → (stem, fn); ASM ``stem`` → (stem, stem)
    (matching the fn_graph_adj single-entry node convention used to key edges at build time)."""
    if "::" in n:
        stem, fn = n.split("::", 1)
        return stem, fn
    return n, n


def assemble_events_from_chain_facts(chain: List[str], facts: dict):
    """Reconstruct the ordered ``_Event`` list for a chain from ``chain_facts`` — the EXACT same
    loop body as ``chain_feasibility`` (c17a8f4), reading the precomputed facts instead of files.
    Returns (events, status) where status mirrors ``chain_feasibility``'s early returns:
    ``"feasible"`` for <2 nodes, ``"unknown"`` when no constant value-guards on the chain."""
    from vbt.lca_feasibility import _Event
    nodes = [_parse_node(n) for n in chain]
    if len(nodes) < 2:
        return [], "feasible"

    edge_facts = facts.get("edges") or {}
    node_facts = facts.get("nodes") or {}

    idx_of: Dict[str, int] = {}
    for i, (stem, _fn) in enumerate(nodes):
        idx_of.setdefault(stem, i)             # first occurrence (a file can repeat)

    events: List[_Event] = []
    scrutinees: Dict[str, str] = {}            # scrut -> language
    bounds: Dict[str, Optional[int]] = {}

    # ---- guards per edge (replay in chain order so bounds last-write-wins per caller stem) ----
    for i in range(len(nodes) - 1):
        cs, cf = nodes[i]
        callee_s, callee_f = nodes[i + 1]
        ef = edge_facts.get(_edge_key(cs, cf, callee_s, callee_f))
        if ef is None:
            continue
        bounds[cs] = ef.get("descend_line")    # last write wins for a repeated caller stem
        hop = idx_of.get(cs, 0)
        if "value_guards" in ef:
            # GAP 1 (chain_facts v4+): replay the PRECOMPUTED parsed value-guards directly — no
            # source read, no compound reparse at trace time. This recovers the safe
            # compound-guard scrutinees (e.g. the CID gate) the old simple-text scan dropped.
            for vg in ef["value_guards"]:
                scrut = vg.get("scrutinee")
                const = vg.get("const")
                if not scrut or const is None:
                    continue
                scrutinees.setdefault(scrut, vg.get("lang") or "cpp")
                events.append(_Event(hop=hop, line=int(vg.get("line") or 0), kind="guard",
                                     scrutinee=scrut, const=const, op=vg.get("op") or "eq",
                                     text=vg.get("source_text") or ""))
        else:
            # Legacy (pre-v4) blob: parse the stored guard text (simple equalities only;
            # compounds stay unknown, exactly as before the GAP-1 fix).
            for g in ef.get("guards") or []:
                parsed = parse_required_guard(g.get("text") or "")
                if not parsed:
                    continue
                scrut, const, op = parsed
                scrutinees.setdefault(scrut, g.get("lang") or "cpp")
                events.append(_Event(hop=hop, line=int(g.get("line") or 0), kind="guard",
                                     scrutinee=scrut, const=const, op=op, text=g.get("text") or ""))

    if not events:
        return [], "unknown"

    # ---- constant assignments to those scrutinees, earlier in the chain (bound-filtered) ----
    for scrut in scrutinees:
        for stem in dict.fromkeys(s for s, _ in nodes):   # distinct files, in chain order
            i = idx_of[stem]
            bound = bounds.get(stem)
            for a in node_facts.get(_node_facts_key(stem, scrut)) or []:
                line = int(a.get("line") or 0)
                if bound is not None and line >= bound:
                    continue                   # assignment after this file's descend point
                events.append(_Event(hop=i, line=line, kind="assign", scrutinee=scrut,
                                     const=a.get("const"), conditional=bool(a.get("conditional"))))
    return events, "ok"


def chain_facts_verdict(chain: List[str], facts: dict):
    """The FeasibilityVerdict for a chain from chain_facts (the trace-time prune decision)."""
    from vbt.lca_feasibility import decide_feasibility, FeasibilityVerdict
    events, status = assemble_events_from_chain_facts(chain, facts)
    if status == "feasible":
        return FeasibilityVerdict("feasible")
    if status == "unknown":
        return FeasibilityVerdict("unknown", reason="no constant value-guards on the chain")
    return decide_feasibility(events)


# =========================================================================== #
# orchestrator + CLI
# =========================================================================== #
def build_and_store_all(job_id, blueprint_dir, asm_dir, graph_file=None) -> Dict[str, Any]:
    """Build ALL chains-precompute outputs in one pass (asm_setters_full + both coverages +
    chain_facts). Run AFTER ``vbt.precompute_db``."""
    bp, src = Path(blueprint_dir), Path(asm_dir)
    out: Dict[str, Any] = {}
    t0 = time.monotonic()
    out["asm_setters_full"] = build_and_store_asm_setters_full(job_id, bp, src)
    out["cpp_setter_map_coverage"] = build_and_store_cpp_setter_map_coverage(job_id, bp, src)
    out["chain_facts"] = build_and_store_chain_facts(job_id, bp, src, graph_file)
    out["elapsed"] = round(time.monotonic() - t0, 2)
    return out


def _derive_dirs(job_id: str) -> Tuple[Path, Path]:
    """Best-effort: derive (blueprint_dir, asm_dir) from the job's workspace (mirrors the task)."""
    from api.config import settings
    from api.tasks._task_utils import find_blueprint_dir
    from api.storage.workspace import WorkspaceManager
    bp_out = settings.JOBS_BASE_DIR / job_id / "output"
    bp = find_blueprint_dir(bp_out) or bp_out
    opts = WorkspaceManager(job_id).get_pipeline_options() or {}
    src = Path(opts.get("source_path") or "")
    if src:
        # Descend to the dir that actually holds .cpp/.asm (stored source_path may be the
        # datasource root one level up) — building against the root yields EMPTY guards.
        from api.routes.vbt import _find_source_dir
        src = _find_source_dir(src)
    return Path(bp), src


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vbt.precompute.chain_facts_db", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_id", help="index.db job id (jobs/<id>/index.db); run AFTER vbt.precompute_db")
    ap.add_argument("--blueprint-dir", type=Path, default=None)
    ap.add_argument("--asm-dir", type=Path, default=None)
    ap.add_argument("--graph", type=Path, default=None,
                    help="file_call_graph.json (default: <blueprint-dir>/file_call_graph.json)")
    a = ap.parse_args(argv)

    bp, src = a.blueprint_dir, a.asm_dir
    if bp is None or src is None:
        try:
            dbp, dsrc = _derive_dirs(a.job_id)
        except Exception as exc:
            ap.error(f"could not derive dirs from job ({exc}); pass --blueprint-dir/--asm-dir")
        bp = bp or dbp
        src = src or dsrc
    if not src or not Path(src).exists():
        ap.error(f"source dir does not exist: {src!r}")

    t0 = time.monotonic()
    res = build_and_store_all(a.job_id, bp, src, a.graph)
    print(f"=== chain_facts precompute (job={a.job_id}, {time.monotonic() - t0:.1f}s) ===")
    for k, v in res.items():
        print(f"  {k:24}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
