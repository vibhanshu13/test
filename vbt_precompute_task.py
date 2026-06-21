"""Celery task: run the VBT (Variable Backward Trace) PRECOMPUTE as a managed
background job.

The VBT engine (repo-root ``vbt/``) is a kb_id-free library.  Before any variable
can be traced it needs three idempotent, cached artifacts plus a warmed Clang
cfg cache:

  * Step 2 — ``file_call_graph.json`` regenerated from the blueprints
    (``vbt.precompute.call_graph.ensure_call_graph``).  Written on disk next to
    the blueprints.
  * Step 3 — the modifier / function index (var/fn -> files), persisted under the
    VBT cache dir (``vbt.precompute.modifier_index.get_modifier_index``).
  * Step 4 — the enum / EQU-DC const-value resolver, persisted under the VBT
    cache dir (``vbt.resolve.const_resolver.get_const_resolver``).
  * Step 5 — cfg-warm: a PARALLEL Clang pre-pass over every ``*.cpp`` source so
    each ``{key}.json`` lands in the disk cfg cache up front
    (``vbt.precompute.parallel.parallel_map`` + ``vbt.cpp_frontend.wrapper.
    warm_cfg_cache``).  This is the dominant cold-start win at 22k files.

Running this here (rather than lazily inside the first trace) lets the heavy,
embarrassingly-parallel precompute inherit the production worker resource limits
configured on ``celery_app`` (``worker_max_memory_per_child``,
``worker_max_tasks_per_child``, ``worker_prefetch_multiplier``,
``task_time_limit``) — this module adds NO new worker config.  ``parallel_map``
itself does the fork-context + daemon-flag clearing so its nested ProcessPool
runs correctly under a Celery prefork daemon (and under an eager ``.apply()``).

Cache location — the "stored properly" option
----------------------------------------------
``cache_in_job_dir`` (DEFAULT True) sets ``VBT_CACHE_DIR`` to
``<blueprint_dir>/vbt_cache`` for the scope of this run, so the index / const /
cfg artifacts persist PER-PROJECT, co-located with the blueprints and shareable
across workers (kb_id-free, no index.db).  When False, the global default cache
dir (``vbt/.cache``) is used.

We import ``vbt`` strictly as a read-only library, and call the same
``precompute_all`` building blocks in-process (so the parallel ProcessPool +
cfg-warm run inside the task).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set

from celery.exceptions import SoftTimeLimitExceeded

PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.config import settings
from api.tasks.celery_app import celery_app
from api.tasks._task_utils import redirect_output_to_log, find_blueprint_dir, job_log_handler
from api.storage.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# path resolution (mirrors vbt_trace_task)
# --------------------------------------------------------------------------- #
def _resolve_blueprint_dir(job_id: str, options: Dict[str, Any]) -> Path:
    """Blueprint dir: explicit option wins, else derive from jobs/<id>/output."""
    bp = options.get("blueprint_dir")
    if bp:
        cand = Path(bp)
        if cand.is_dir():
            # tolerate being handed jobs/<id>/output that nests blueprints in a
            # named/zip subfolder — find_blueprint_dir reproduces the pipeline's
            # discovery, but only when it actually finds blueprints.
            return find_blueprint_dir(cand) or cand
        return cand
    out = settings.JOBS_BASE_DIR / job_id / "output"
    return find_blueprint_dir(out) or out


def _resolve_asm_dir(job_id: str, options: Dict[str, Any]) -> Path:
    """Source dir (.cpp/.asm/.mac): explicit option wins, else the pipeline's
    recorded source_path for the job."""
    ad = options.get("asm_dir")
    if ad:
        return Path(ad)
    try:
        opts = WorkspaceManager(job_id).get_pipeline_options() or {}
        sp = opts.get("source_path")
        if sp:
            return Path(sp)
    except Exception:
        pass
    raise ValueError(
        "asm_dir not provided and could not be derived from the job's pipeline "
        "options (source_path). Pass options['asm_dir']."
    )


def _resolve_graph_file(blueprint_dir: Path, options: Dict[str, Any]) -> Path:
    gf = options.get("graph_file")
    if gf:
        return Path(gf)
    return blueprint_dir / "file_call_graph.json"


# --------------------------------------------------------------------------- #
# env scoping — set VBT_CACHE_DIR / VBT_WORKERS only for the duration of the run
# so concurrent runs / the rest of the worker process are not affected.
# --------------------------------------------------------------------------- #
@contextmanager
def _scoped_env(overrides: Dict[str, Optional[str]]) -> Generator[None, None, None]:
    """Temporarily set/clear env vars, restoring the prior values on exit.

    A value of ``None`` means "leave unset for the scope"; otherwise the string
    value is set.  All keys are restored to their original state in the finally
    block, even if the body raises.
    """
    saved: Dict[str, Optional[str]] = {}
    for k, v in overrides.items():
        saved[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


# --------------------------------------------------------------------------- #
# status helpers — defensive so a missing/broken metadata store (e.g. an eager
# run with no job record) never aborts the precompute itself.
# --------------------------------------------------------------------------- #
def _safe_set_status(ws: Optional[WorkspaceManager], status: str, **kw: Any) -> None:
    if ws is None:
        return
    try:
        ws.set_status(status, **kw)
    except Exception as exc:  # pragma: no cover - metadata store unavailable
        logger.warning("[vbt_precompute] could not set status %s: %s", status, exc)


def _make_ws(job_id: str) -> Optional[WorkspaceManager]:
    try:
        return WorkspaceManager(job_id)
    except Exception as exc:  # pragma: no cover - workspace dir not creatable
        logger.warning("[vbt_precompute] WorkspaceManager(%s) unavailable: %s", job_id, exc)
        return None


def _count_cpp_sources(asm_dir: Path) -> List[str]:
    """Sorted list of *.cpp source paths (the cfg-warm workload), as strings."""
    return [str(p) for p in sorted(asm_dir.glob("*.cpp"))]


def _update_kb_status(kb_id: str, status: str, error: Optional[str] = None) -> None:
    try:
        from api.db import SessionLocal
        from api.models.knowledge_base import KnowledgeBase

        db = SessionLocal()
        try:
            kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
            if kb:
                kb.status = status
                kb.error = error
                kb.updated_at = datetime.now(timezone.utc)
                db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.warning("[vbt_precompute] could not update KB status for %s: %s", kb_id, exc)


VBT_DB_REPAIR_ARTIFACTS: Set[str] = {
    "reverse_adj",
    "forward_file_adj",
    "asm_setter_map",
    "cpp_setter_map",
    "cpp_file_writes",
    "fn_facts",
    "fn_graph_adj",
    "cfg",
}


def run_db_repair(
    job_id: str, options: Dict[str, Any], ws: Optional[WorkspaceManager] = None
) -> Dict[str, Any]:
    """Incrementally rebuild selected VBT DB artifacts for an existing KB/job."""
    from vbt.precompute_db import build_only

    t_start = time.monotonic()
    blueprint_dir = _resolve_blueprint_dir(job_id, options)
    asm_dir = _resolve_asm_dir(job_id, options)
    graph_file = _resolve_graph_file(blueprint_dir, options)
    names = [str(n).strip() for n in (options.get("only") or ["fn_graph_adj"]) if str(n).strip()]
    if "fn_graph_adj" in names and "cfg" not in names:
        names.append("cfg")
    unknown = sorted(set(names) - VBT_DB_REPAIR_ARTIFACTS)
    if unknown:
        raise ValueError(f"unknown VBT DB repair artifact(s): {unknown}; known: {sorted(VBT_DB_REPAIR_ARTIFACTS)}")
    if not names:
        raise ValueError("options['only'] must contain at least one artifact")
    if not blueprint_dir.is_dir():
        raise FileNotFoundError(f"blueprint_dir not found: {blueprint_dir}")
    if not asm_dir.is_dir():
        raise FileNotFoundError(f"asm_dir not found: {asm_dir}")

    cache_in_job_dir = bool(options.get("cache_in_job_dir", True))
    cache_dir = blueprint_dir / "vbt_cache" if cache_in_job_dir else PROJECT_ROOT / "vbt" / ".cache"
    env_cache = str(cache_dir) if cache_in_job_dir else None
    if cache_in_job_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    _safe_set_status(ws, "running", progress=f"repairing VBT DB artifacts: {', '.join(names)}")
    print("=== VBT DB repair ===", flush=True)
    print(f"  job_id         : {job_id}", flush=True)
    print(f"  blueprint_dir  : {blueprint_dir}", flush=True)
    print(f"  asm_dir        : {asm_dir}", flush=True)
    print(f"  graph_file     : {graph_file}", flush=True)
    print(f"  only           : {', '.join(names)}", flush=True)

    with _scoped_env({"VBT_CACHE_DIR": env_cache}):
        res = build_only(job_id, blueprint_dir, asm_dir, graph_file, names=names)

    elapsed = time.monotonic() - t_start
    db_complete = False
    try:
        from api.tasks.vbt_trace_task import _has_db_trace_artifacts
        db_complete = _has_db_trace_artifacts(job_id)
    except Exception:
        db_complete = False

    if bool(options.get("mark_kb_success", True)):
        if db_complete:
            _update_kb_status(job_id, "success")
        else:
            _update_kb_status(
                job_id,
                "failed",
                error="VBT DB repair completed, but required trace artifacts are still incomplete.",
            )

    manifest = {
        "job_id": job_id,
        "status": "success",
        "operation": "vbt_db_repair",
        "artifacts": res,
        "db_complete": db_complete,
        "cache_dir": str(cache_dir),
        "graph_path": str(graph_file),
        "elapsed": round(elapsed, 3),
    }
    _safe_set_status(ws, "success", progress="repair complete")
    logger.info("[vbt_precompute] job=%s DB repair DONE — %s elapsed=%.1fs", job_id, res, elapsed)
    return manifest


# --------------------------------------------------------------------------- #
# core worker body — callable directly in tests / from the Celery task
# --------------------------------------------------------------------------- #
def run_precompute(
    job_id: str, options: Dict[str, Any], ws: Optional[WorkspaceManager] = None
) -> Dict[str, Any]:
    """Pure worker body: call graph -> modifier index -> const resolver -> cfg-warm.

    Returns a manifest dict::

        {job_id, status, cache_dir, graph_path,
         steps:[{name, elapsed, detail}, ...], elapsed}

    ``ws`` is optional; when omitted no status / output bookkeeping is done
    (handy for direct calls in tests).

    ``options``::

        blueprint_dir       optional; else jobs/<job_id>/output (find_blueprint_dir)
        asm_dir             optional; else the job's recorded source_path
        graph_file          optional; else <blueprint_dir>/file_call_graph.json
        cache_in_job_dir    bool, DEFAULT True — VBT_CACHE_DIR = <bp>/vbt_cache
                            (per-project, co-located, shareable). False → global.
        rebuild             bool — force a rebuild (clears the chosen cache dir
                            for call graph + indexes).
        cfg_warm            bool, DEFAULT True — run the parallel cfg pre-warm.
        workers             optional int — VBT_WORKERS for this run scope.
        require_db_precompute bool, DEFAULT True — fail instead of silently
                            producing a slow KB when DB artifacts cannot be built.
    """
    # Late, read-only imports of the VBT library so an import-time error in a
    # parallel edit surfaces inside the task body (caught + reported / retried).
    from vbt.precompute.call_graph import ensure_call_graph
    from vbt.precompute.modifier_index import get_modifier_index
    from vbt.resolve.const_resolver import get_const_resolver

    t_start = time.monotonic()

    blueprint_dir = _resolve_blueprint_dir(job_id, options)
    asm_dir = _resolve_asm_dir(job_id, options)
    graph_file = _resolve_graph_file(blueprint_dir, options)

    if not blueprint_dir.is_dir():
        raise FileNotFoundError(f"blueprint_dir not found: {blueprint_dir}")
    if not asm_dir.is_dir():
        raise FileNotFoundError(f"asm_dir not found: {asm_dir}")

    cache_in_job_dir = bool(options.get("cache_in_job_dir", True))
    rebuild = bool(options.get("rebuild", False))
    cfg_warm = bool(options.get("cfg_warm", True))
    require_db_precompute = bool(options.get("require_db_precompute", True))
    workers = options.get("workers")

    # Resolve the cache dir.  Per-project (default): co-located with the
    # blueprints so the index/const/cfg artifacts are shareable across workers
    # and survive worker recycling.  Global: leave VBT_CACHE_DIR unset so the
    # vbt modules fall back to vbt/.cache.
    if cache_in_job_dir:
        cache_dir = blueprint_dir / "vbt_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        env_cache: Optional[str] = str(cache_dir)
    else:
        # vbt/.cache (two levels up from vbt/precompute/) — reported for clarity.
        cache_dir = PROJECT_ROOT / "vbt" / ".cache"
        env_cache = None  # leave unset -> module default

    env_workers: Optional[str] = None
    if workers is not None:
        try:
            env_workers = str(int(workers))
        except (TypeError, ValueError):
            raise ValueError(f"options['workers'] must be an int, got {workers!r}")

    print("=== VBT precompute ===", flush=True)
    print(f"  job_id         : {job_id}", flush=True)
    print(f"  blueprint_dir  : {blueprint_dir}", flush=True)
    print(f"  asm_dir        : {asm_dir}", flush=True)
    print(f"  graph_file     : {graph_file}", flush=True)
    print(f"  cache_dir      : {cache_dir} (cache_in_job_dir={cache_in_job_dir})", flush=True)
    print(f"  rebuild        : {rebuild}   cfg_warm: {cfg_warm}   workers: {workers}", flush=True)

    steps: List[Dict[str, Any]] = []

    def _record(name: str, elapsed: float, detail: str) -> None:
        steps.append({"name": name, "elapsed": round(elapsed, 3), "detail": detail})

    # Scope the env for the whole precompute (cache dir + worker override).  The
    # parallel cfg-warm forks children that inherit os.environ, so the override
    # must be in place around parallel_map too — hence we wrap the entire body.
    with _scoped_env({"VBT_CACHE_DIR": env_cache, "VBT_WORKERS": env_workers}):
        # If rebuilding with a per-project cache, clear the index/const/cfg
        # artifacts under it first (ensure_call_graph(rebuild=True) handles the
        # graph; the index/const/cfg modules key on a manifest hash and have no
        # rebuild flag, so we drop their artifact dirs to force a cold rebuild).
        if rebuild and env_cache is not None:
            import shutil
            for sub in ("index", "const", "cfg"):
                shutil.rmtree(Path(env_cache) / sub, ignore_errors=True)
            print(f"  [rebuild] cleared {env_cache}/(index,const,cfg)", flush=True)

        # ---- Step 2: file_call_graph.json -----------------------------------
        logger.info("[vbt_precompute] job=%s STEP call_graph START", job_id)
        print("--- Step 1/4: call graph (file_call_graph.json) ---", flush=True)
        _t = time.monotonic()
        gf = ensure_call_graph(blueprint_dir, asm_dir, out_path=graph_file, rebuild=rebuild)
        _e = time.monotonic() - _t
        n_nodes = n_edges = -1
        try:
            import json as _json
            g = _json.loads(Path(gf).read_text())
            n_nodes, n_edges = len(g.get("nodes", [])), len(g.get("edges", []))
        except Exception:
            pass
        detail = f"{n_nodes} nodes, {n_edges} edges -> {gf}"
        logger.info("[vbt_precompute] job=%s STEP call_graph DONE — %s elapsed=%.1fs",
                    job_id, detail, _e)
        print(f"  [TIME] call graph: {_e:.1f}s — {detail}", flush=True)
        _record("call_graph", _e, detail)

        # ---- Step 3: modifier + function index ------------------------------
        logger.info("[vbt_precompute] job=%s STEP modifier_index START", job_id)
        print("--- Step 2/4: modifier + function index ---", flush=True)
        _t = time.monotonic()
        midx = get_modifier_index(blueprint_dir, asm_dir)
        _e = time.monotonic() - _t
        detail = (f"asm_vars={len(midx.asm)} cpp_fields={len(midx.cpp)} "
                  f"functions={len(midx.functions)}")
        logger.info("[vbt_precompute] job=%s STEP modifier_index DONE — %s elapsed=%.1fs",
                    job_id, detail, _e)
        print(f"  [TIME] modifier index: {_e:.1f}s — {detail}", flush=True)
        _record("modifier_index", _e, detail)

        # ---- Step 4: enum / EQU-DC const resolver ---------------------------
        logger.info("[vbt_precompute] job=%s STEP const_resolver START", job_id)
        print("--- Step 3/4: enum/const resolver ---", flush=True)
        _t = time.monotonic()
        cr = get_const_resolver(blueprint_dir, asm_dir)
        _e = time.monotonic() - _t
        detail = f"cpp_enum={len(cr.cpp_enum)} asm_const={len(cr.asm_const)}"
        logger.info("[vbt_precompute] job=%s STEP const_resolver DONE — %s elapsed=%.1fs",
                    job_id, detail, _e)
        print(f"  [TIME] const resolver: {_e:.1f}s — {detail}", flush=True)
        _record("const_resolver", _e, detail)

        # ---- Step 5: cfg-warm (parallel Clang pre-pass over cpp sources) ----
        # The parallel pool runs INSIDE this (possibly Celery prefork daemon)
        # task — parallel_map clears the daemon flag + forces the fork context so
        # a nested pool works.  Skipped on cfg_warm=False or when the cfg_extract
        # binary is absent (we never compile it here — it needs LLVM).
        cpp_paths = _count_cpp_sources(asm_dir)
        binary = PROJECT_ROOT / "vbt" / "cpp_frontend" / "cfg_extract"
        if not cfg_warm:
            detail = "skipped (cfg_warm=False)"
            print(f"--- Step 4/4: cfg-warm — {detail} ---", flush=True)
            _record("cfg_warm", 0.0, detail)
        elif not binary.exists():
            detail = f"skipped (cfg_extract binary missing: {binary})"
            logger.warning("[vbt_precompute] job=%s STEP cfg_warm %s", job_id, detail)
            print(f"--- Step 4/4: cfg-warm — {detail} ---", flush=True)
            _record("cfg_warm", 0.0, detail)
        elif not cpp_paths:
            detail = "0/0 files warmed (no cpp sources)"
            print(f"--- Step 4/4: cfg-warm — {detail} ---", flush=True)
            _record("cfg_warm", 0.0, detail)
        else:
            from vbt.precompute.parallel import parallel_map, plan_workers
            from vbt.cpp_frontend.wrapper import warm_cfg_cache

            plan = plan_workers()
            logger.info(
                "[vbt_precompute] job=%s STEP cfg_warm START — %d cpp files, "
                "%d workers (batch %d, mem cap %d MB)",
                job_id, len(cpp_paths), plan.worker_count, plan.batch_size,
                plan.per_proc_mem_cap_mb,
            )
            print(
                f"--- Step 4/4: cfg-warm ({len(cpp_paths)} cpp files, "
                f"{plan.worker_count} workers, batch {plan.batch_size}, "
                f"mem cap {plan.per_proc_mem_cap_mb} MB) ---", flush=True,
            )
            _t = time.monotonic()
            results = parallel_map(warm_cfg_cache, cpp_paths, plan=plan)
            n_warmed = sum(1 for ok in results if ok)
            _e = time.monotonic() - _t
            detail = f"{n_warmed}/{len(cpp_paths)} files warmed"
            logger.info("[vbt_precompute] job=%s STEP cfg_warm DONE — %s elapsed=%.1fs",
                        job_id, detail, _e)
            print(f"  [TIME] cfg-warm: {_e:.1f}s — {detail}", flush=True)
            _record("cfg_warm", _e, detail)

        # ---- Step 5/5: VBT DB artifacts (T11) — resolvers, route name_to_stems,
        # cpp_call_edges, ASM blueprints, source text, cfg blobs → index.db, so warm traces
        # read from DB not files. Required by default for the API's "completed KB" contract:
        # without these artifacts, large traces are correct but can fall back to corpus scans.
        try:
            from vbt.precompute_db import precompute_vbt_db
            _t = time.monotonic()
            db_res = precompute_vbt_db(job_id, blueprint_dir, asm_dir, gf)
            _e = time.monotonic() - _t
            detail = ", ".join(f"{k}={v}" for k, v in db_res.items())
            logger.info("[vbt_precompute] job=%s STEP vbt_db DONE — %s elapsed=%.1fs",
                        job_id, detail, _e)
            print(f"  [TIME] vbt_db: {_e:.1f}s — {detail}", flush=True)
            _record("vbt_db", _e, detail)
        except Exception as exc:
            logger.warning("[vbt_precompute] job=%s STEP vbt_db FAILED: %s", job_id, exc)
            _record("vbt_db", 0.0, f"failed: {exc}")
            if require_db_precompute:
                raise RuntimeError(
                    "VBT DB precompute failed; refusing to mark the KB ready because "
                    "traces would fall back to slow corpus-wide scans. Set "
                    "require_db_precompute=false only for debugging."
                ) from exc

    elapsed = time.monotonic() - t_start
    manifest: Dict[str, Any] = {
        "job_id": job_id,
        "status": "success",
        "cache_dir": str(cache_dir),
        "graph_path": str(gf),
        "steps": steps,
        "elapsed": round(elapsed, 3),
    }

    _safe_set_status(ws, "success")
    logger.info("[vbt_precompute] job=%s COMPLETE — elapsed=%.1fs", job_id, elapsed)
    print(f"=== VBT precompute complete: total elapsed {elapsed:.1f}s ===", flush=True)
    return manifest


# --------------------------------------------------------------------------- #
# Celery task
# --------------------------------------------------------------------------- #
@celery_app.task(
    bind=True,
    name="asm.vbt_precompute",
)
def run_vbt_precompute(self, job_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Run the VBT precompute (call graph + indexes + cfg-warm) as a background job.

    ``options``::

        {
          "blueprint_dir": "jobs/<id>/output",  # optional; else jobs/<id>/output
          "asm_dir": "temp/asm",                # optional; else job source_path
          "graph_file": null,                   # optional; else <bp>/file_call_graph.json
          "cache_in_job_dir": true,             # DEFAULT true — VBT_CACHE_DIR =
                                                #   <blueprint_dir>/vbt_cache
                                                #   (per-project, shareable).
                                                #   false -> global vbt/.cache
          "rebuild": false,                     # force rebuild
          "cfg_warm": true,                     # DEFAULT true; false skips Step 5
          "workers": null,                      # optional int -> VBT_WORKERS
          "require_db_precompute": true          # DEFAULT true; fail if DB artifacts fail
        }

    Returns the manifest from :func:`run_precompute`, or
    ``{"job_id","status":"failed","error":...}`` on failure (matching the
    existing task convention).
    """
    ws = _make_ws(job_id)
    _safe_set_status(ws, "running", operation="vbt_precompute")

    log_file = ws.log_file if ws is not None else (settings.JOBS_BASE_DIR / job_id / "job.log")
    with redirect_output_to_log(log_file):
        with job_log_handler(log_file, job_id=job_id):
            try:
                return run_precompute(job_id, options, ws=ws)
            except SoftTimeLimitExceeded:
                error_msg = "Task exceeded soft time limit and was aborted."
                logger.error("[vbt_precompute] job %s: %s", job_id, error_msg)
                _safe_set_status(ws, "failed", error=error_msg)
                return {"job_id": job_id, "status": "failed", "error": error_msg}
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.exception("[vbt_precompute] job %s failed: %s", job_id, error_msg)
                _safe_set_status(ws, "failed", error=error_msg)
                return {"job_id": job_id, "status": "failed", "error": error_msg}


@celery_app.task(
    bind=True,
    name="asm.vbt_db_repair",
)
def run_vbt_db_repair(self, job_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Incrementally rebuild selected DB-backed VBT artifacts for an existing KB."""
    ws = _make_ws(job_id)
    _safe_set_status(ws, "running", operation="vbt_db_repair")
    if bool(options.get("mark_kb_success", True)):
        _update_kb_status(job_id, "running")

    log_file = ws.log_file if ws is not None else (settings.JOBS_BASE_DIR / job_id / "job.log")
    with redirect_output_to_log(log_file):
        with job_log_handler(log_file, job_id=job_id):
            try:
                return run_db_repair(job_id, options, ws=ws)
            except SoftTimeLimitExceeded:
                error_msg = "Task exceeded soft time limit and was aborted."
                logger.error("[vbt_db_repair] job %s: %s", job_id, error_msg)
                _safe_set_status(ws, "failed", error=error_msg)
                if bool(options.get("mark_kb_success", True)):
                    _update_kb_status(job_id, "failed", error=error_msg)
                return {"job_id": job_id, "status": "failed", "error": error_msg}
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.exception("[vbt_db_repair] job %s failed: %s", job_id, error_msg)
                _safe_set_status(ws, "failed", error=error_msg)
                if bool(options.get("mark_kb_success", True)):
                    _update_kb_status(job_id, "failed", error=error_msg)
                return {"job_id": job_id, "status": "failed", "error": error_msg}
