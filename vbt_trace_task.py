"""Celery task: run a VBT (Variable Backward Trace) lineage trace as a managed
background job.

The VBT engine (repo-root ``vbt/``) is a kb_id-free library that, for a given
(variable, language, chain), finds every setter site (with its set value and the
full guard set along the call path) and the dependent-variable tree.  A deep
trace is heavy (cross-file, cross-language, recursive, with a Clang frontend for
C++), so it runs here as a Celery job that inherits the production resource
limits configured on ``celery_app`` (``worker_max_memory_per_child``,
``worker_max_tasks_per_child``, ``worker_prefetch_multiplier``,
``task_time_limit``) — this module adds NO new worker config.

We import ``vbt`` strictly as a read-only library:

  * ``vbt.precompute_all`` style precompute (call graph + modifier index +
    const resolver) is run once per project; the steps are cached/idempotent.
  * ``vbt.engine.trace_root_variable`` does the trace.
  * ``vbt.output.emit.write_result`` persists the result (optional split / strip
    code / msgpack / zip).

Forward-compat: the engine's ``trace_root_variable`` signature is being extended
by a parallel effort (``disable_dependents`` / ``candidate_functions`` /
``home_hint`` / ``progress`` …).  We pass only the ``trace_options`` keys that
the CURRENT signature accepts (filtered via ``inspect.signature``), so new
params flow through automatically once they land and unknown keys never raise.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from celery.exceptions import SoftTimeLimitExceeded

PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.config import settings
from api.tasks.celery_app import celery_app
from api.tasks._task_utils import redirect_output_to_log, find_blueprint_dir, job_log_handler
from api.storage.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


@contextmanager
def intercept_file_reads(touched_files: set) -> Generator[None, None, None]:
    import builtins
    import io
    import backward_traversal.utils.blueprint_utils as BU
    
    original_open = builtins.open
    original_io_open = io.open
    original_load_json = BU.load_json
    original_source_override = BU.source_override_bytes
    original_read_text = BU._read_text_with_retry

    def _add_stem(path):
        try:
            stem = Path(path).stem
            clean_stem = stem.split(".")[0]
            if clean_stem:
                touched_files.add(clean_stem)
        except Exception:
            pass

    def custom_open(file, *args, **kwargs):
        path_str = str(file)
        if any(path_str.endswith(ext) for ext in (".cpp", ".hpp", ".h", ".mac", ".json")):
            if not any(sub in path_str for sub in ("site-packages", "lib/python", "node_modules", ".pytest_cache", ".git")):
                mode = kwargs.get("mode", "r")
                if "r" in mode:
                    _add_stem(path_str)
                    msg = f"[DISK_READ] {path_str}"
                    print(msg, flush=True)
                    logger.info(msg)
        return original_open(file, *args, **kwargs)

    def custom_load_json(path):
        _add_stem(path)
        return original_load_json(path)

    def custom_source_override(path):
        _add_stem(path)
        return original_source_override(path)

    def custom_read_text(path, *args, **kwargs):
        _add_stem(path)
        return original_read_text(path, *args, **kwargs)

    builtins.open = custom_open
    io.open = custom_open
    BU.load_json = custom_load_json
    BU.source_override_bytes = custom_source_override
    BU._read_text_with_retry = custom_read_text
    try:
        yield
    finally:
        builtins.open = original_open
        io.open = original_io_open
        BU.load_json = original_load_json
        BU.source_override_bytes = original_source_override
        BU._read_text_with_retry = original_read_text




# --------------------------------------------------------------------------- #
# path resolution
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


def _has_db_trace_artifacts(precompute_job_id: str) -> bool:
    """True when the completed KB has enough VBT DB artifacts to skip the
    worker-level live precompute phase.

    The engine itself still validates/falls back artifact-by-artifact. This
    guard only avoids doing corpus-wide file/disk-cache work before the engine
    has entered DB load-only mode.
    """
    try:
        from vbt.precompute import db_artifacts as DA
        from vbt.precompute.asm_db import ASM_BP_MANIFEST
        from vbt.precompute.fn_facts_db import FN_FACTS_ARTIFACT
        from vbt.precompute.graph_db import (
            CPP_CALL_EDGES_ARTIFACT,
            FILE_CALL_GRAPH_ARTIFACT,
            FN_GRAPH_ADJ_ARTIFACT,
            FORWARD_FILE_ADJ_ARTIFACT,
            REVERSE_ADJ_ARTIFACT,
            ROUTE_NAME_TO_STEMS_ARTIFACT,
        )
        from vbt.precompute.resolvers_db import (
            CONST_ARTIFACT,
            MEMBERSHIP_ARTIFACT,
            MODIFIER_ARTIFACT,
            NAME_ALIAS_ARTIFACT,
        )
        from vbt.precompute.source_db import SOURCE_MANIFEST
    except Exception:
        return False

    previous_load_only = DA.is_load_only()
    try:
        # Avoid index-db open side effects such as GVL backfill checks; a VBT
        # trace only needs the vbt_* artifacts.
        DA.set_load_only(True)
        required = (
            FILE_CALL_GRAPH_ARTIFACT,
            ROUTE_NAME_TO_STEMS_ARTIFACT,
            REVERSE_ADJ_ARTIFACT,
            FORWARD_FILE_ADJ_ARTIFACT,
            CPP_CALL_EDGES_ARTIFACT,
            FN_GRAPH_ADJ_ARTIFACT,
            NAME_ALIAS_ARTIFACT,
            MEMBERSHIP_ARTIFACT,
            CONST_ARTIFACT,
            MODIFIER_ARTIFACT,
            ASM_BP_MANIFEST,
            SOURCE_MANIFEST,
            FN_FACTS_ARTIFACT,
        )
        if not all(DA.manifest_present(precompute_job_id, artifact) for artifact in required):
            return False
        # Version guard: a present-but-STALE fn_graph_adj (built by an older edge-resolution)
        # would otherwise be loaded as-is (load_fn_graph_adj does not version-check), silently
        # serving under-reporting reachability. Treat a version mismatch as "not present" so the
        # worker rebuilds it instead of skipping precompute.
        from vbt.precompute.graph_db import FN_GRAPH_ADJ_VERSION
        man = DA.read_manifest(precompute_job_id, FN_GRAPH_ADJ_ARTIFACT)
        if not man or int(man.get("version", 0)) != FN_GRAPH_ADJ_VERSION:
            return False

        # T3 (visibility, not enforcement): warn about partial precompute. The
        # speedup artifacts (per-stem setter maps, cpp_file_writes, blueprints,
        # source) are designed to be byte-identical fallbacks — a trace still
        # returns the correct result if any are missing, just slower on 24k.
        # The fallback is still cheap on a small corpus, so we DO NOT reject
        # the fast path here. The pipeline verifier (vbt_pipeline_task) flags
        # missing artifacts to ops separately; production deployments that
        # require the fast path should set VBT_REQUIRE_FAST_PATH=1 (handled by
        # the verifier, not here).
        return True
    except Exception:
        return False
    finally:
        DA.set_load_only(previous_load_only)


@contextmanager
def _scoped_env(overrides: Dict[str, Optional[str]]) -> Generator[None, None, None]:
    saved: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


# --------------------------------------------------------------------------- #
# status helpers — defensive so a missing/broken metadata store (e.g. an eager
# run with no job record) never aborts the trace itself.
# --------------------------------------------------------------------------- #
def _safe_set_status(ws: Optional[WorkspaceManager], status: str, **kw: Any) -> None:
    if ws is None:
        return
    try:
        ws.set_status(status, **kw)
    except Exception as exc:  # pragma: no cover - metadata store unavailable
        logger.warning("[vbt_trace] could not set status %s: %s", status, exc)


def _make_ws(job_id: str) -> Optional[WorkspaceManager]:
    try:
        return WorkspaceManager(job_id)
    except Exception as exc:  # pragma: no cover - workspace dir not creatable
        logger.warning("[vbt_trace] WorkspaceManager(%s) unavailable: %s", job_id, exc)
        return None


# --------------------------------------------------------------------------- #
# core worker body — callable directly in tests / from the Celery task
# --------------------------------------------------------------------------- #
def run_trace(job_id: str, options: Dict[str, Any], ws: Optional[WorkspaceManager] = None) -> Dict[str, Any]:
    """Pure worker body: precompute → trace → emit → manifest.

    Returns a manifest dict::

        {job_id, variable, language, setters, dependentVariables, outputs:[...],
         elapsed, precomputeElapsed, traceElapsed, emitElapsed, output_files}

    ``ws`` is optional; when omitted no status / output-file bookkeeping is done
    (handy for direct calls in tests).
    """
    # Late, read-only imports of the VBT library so an import-time error in a
    # parallel edit surfaces inside the task body (caught + reported / retried).
    from vbt.engine import Hop, trace_root_variable
    from vbt.precompute.call_graph import ensure_call_graph
    from vbt.precompute.modifier_index import get_modifier_index
    from vbt.resolve.const_resolver import get_const_resolver
    from vbt.output.emit import write_result

    t_start = time.monotonic()

    variable: str = options["variable"]
    language: str = options.get("language", "asm")
    if language not in ("cpp", "asm"):
        raise ValueError(f"language must be 'cpp' or 'asm', got {language!r}")

    blueprint_dir = _resolve_blueprint_dir(job_id, options)
    asm_dir = _resolve_asm_dir(job_id, options)
    graph_file = _resolve_graph_file(blueprint_dir, options)
    precompute_job_id = str(options.get("kb_id") or options.get("precompute_job_id") or job_id)
    allow_live_precompute = bool(options.get("allow_live_precompute", not bool(options.get("kb_id"))))

    if not blueprint_dir.is_dir():
        raise FileNotFoundError(f"blueprint_dir not found: {blueprint_dir}")
    if not asm_dir.is_dir():
        raise FileNotFoundError(f"asm_dir not found: {asm_dir}")
    cache_in_job_dir = bool(options.get("cache_in_job_dir", True))
    env_cache = str(blueprint_dir / "vbt_cache") if cache_in_job_dir else None

    # ---- build the chain (Hop list, TAIL last) ----
    chain_spec: List[Dict[str, Any]] = options.get("chain") or []
    if not chain_spec:
        raise ValueError("options['chain'] must list at least one hop (the tail)")
    chain_prefix: List[Hop] = []
    for h in chain_spec:
        stem = h.get("stem")
        ftype = h.get("type") or h.get("file_type") or "asm"
        if not stem:
            raise ValueError(f"chain hop missing 'stem': {h!r}")
        if ftype not in ("asm", "cpp"):
            raise ValueError(f"chain hop type must be asm|cpp, got {ftype!r}")
        chain_prefix.append(Hop(stem, ftype, h.get("function")))

    print(
        f"=== VBT trace: variable={variable!r} language={language} "
        f"chain={[h.stem for h in chain_prefix]} ===",
        flush=True,
    )
    print(f"  blueprint_dir : {blueprint_dir}", flush=True)
    print(f"  asm_dir       : {asm_dir}", flush=True)
    print(f"  graph_file    : {graph_file}", flush=True)
    print(f"  precompute_id : {precompute_job_id}", flush=True)

    # -----------------------------------------------------------------------
    # Phase 1 — precompute (call graph + modifier index + const resolver).
    # All three are cached/idempotent (call graph on disk next to blueprints;
    # index/resolver under vbt/.cache keyed on a source-manifest hash), so this
    # is near-instant on a warm project.  The cfg_extract C++ frontend binary
    # must already be built (vbt/cpp_frontend/build.sh) — we do NOT compile it
    # here (it needs LLVM and is a one-time per-machine step).
    # -----------------------------------------------------------------------
    with _scoped_env({"VBT_CACHE_DIR": env_cache}):
        db_fast_path = _has_db_trace_artifacts(precompute_job_id)
        if db_fast_path:
            precompute_elapsed = 0.0
            _safe_set_status(ws, "running", progress="precompute skipped: DB artifacts present")
            logger.info(
                "[vbt_trace] job=%s PHASE precompute SKIPPED — DB artifacts present for %s",
                job_id, precompute_job_id,
            )
            print(
                "--- Phase 1/3: precompute skipped "
                f"(DB artifacts present for {precompute_job_id}) ---",
                flush=True,
            )
        else:
            if not allow_live_precompute:
                raise RuntimeError(
                    f"VBT DB artifacts are incomplete for KB {precompute_job_id}. "
                    "Refusing live corpus precompute because it can take hours at this scale. "
                    "Re-run the VBT pipeline/precompute, or set allow_live_precompute=true "
                    "only for a deliberate debug run."
                )
            _safe_set_status(ws, "running", progress="precompute: call graph and local caches")
            logger.info("[vbt_trace] job=%s PHASE precompute START", job_id)
            print("--- Phase 1/3: precompute (call graph + modifier index + const resolver) ---", flush=True)
            _t = time.monotonic()
            ensure_call_graph(blueprint_dir, asm_dir, out_path=graph_file)
            midx = get_modifier_index(blueprint_dir, asm_dir)
            cr = get_const_resolver(blueprint_dir, asm_dir)
            precompute_elapsed = time.monotonic() - _t
            logger.info(
                "[vbt_trace] job=%s PHASE precompute DONE — asm_vars=%d cpp_fields=%d "
                "functions=%d cpp_enum=%d asm_const=%d elapsed=%.1fs",
                job_id, len(midx.asm), len(midx.cpp), len(midx.functions),
                len(cr.cpp_enum), len(cr.asm_const), precompute_elapsed,
            )
            print(f"  [TIME] precompute elapsed: {precompute_elapsed:.1f}s", flush=True)

        # -----------------------------------------------------------------------
        # Phase 2 — trace.
        # FORWARD-COMPAT param passing: keep only the trace_options keys that the
        # CURRENT trace_root_variable signature accepts.  Params being added by the
        # parallel effort flow through automatically once present; unknown keys are
        # dropped (logged) so they never raise a TypeError.
        # -----------------------------------------------------------------------
        trace_options: Dict[str, Any] = dict(options.get("trace_options") or {})
        candidate_stems = trace_options.pop("candidate_stems", None)

        sig_params = inspect.signature(trace_root_variable).parameters
        accepted = {k: v for k, v in trace_options.items() if k in sig_params}
        dropped = sorted(set(trace_options) - set(accepted))
        if dropped:
            logger.info(
                "[vbt_trace] job=%s dropping trace_options not in current engine "
                "signature (forward-compat): %s", job_id, dropped,
            )
            print(f"  [forward-compat] ignoring unsupported trace_options: {dropped}", flush=True)
        # candidate_stems is a first-class kwarg on the engine — pass it explicitly
        # when supplied (and supported), so it isn't lost to the generic filter.
        if candidate_stems is not None and "candidate_stems" in sig_params:
            accepted["candidate_stems"] = candidate_stems
        # Use the completed KB id as the DB-backed precompute key. The trace job
        # has its own output workspace, but VBT artifacts live under the KB job.
        if "job_id" in sig_params:
            accepted["job_id"] = precompute_job_id
        # Default-ON engine progress for managed trace jobs: emits the BUILD tag
        # (self-identifies the live engine code) + per-phase / per-dep STDERR markers,
        # all tee'd into the job log. The caller can still override via trace_options.
        # Set VBT_HANG_DEBUG=1 on the worker to raise these to per-operation DEBUG when
        # chasing a hard hang. Cheap when unused (no-op below INFO).
        if "progress" in sig_params:
            accepted.setdefault("progress", True)

        logger.info(
            "[vbt_trace] job=%s PHASE trace START — precompute_id=%s forwarded params: %s",
            job_id, precompute_job_id, sorted(accepted),
        )
        _safe_set_status(ws, "running", progress="trace: engine traversal")
        print(f"--- Phase 2/3: trace (engine params: {sorted(accepted)}) ---", flush=True)
        _t = time.monotonic()
        out = trace_root_variable(
            variable, language, chain_prefix,
            blueprint_dir=blueprint_dir, asm_dir=asm_dir, graph_file=graph_file,
            **accepted,
        )
        trace_elapsed = time.monotonic() - _t
    setters = (out.get("rootVariable") or {}).get("setters") or []
    dep_vars = out.get("dependentVariables") or []
    logger.info(
        "[vbt_trace] job=%s PHASE trace DONE — setters=%d dependentVariables=%d elapsed=%.1fs",
        job_id, len(setters), len(dep_vars), trace_elapsed,
    )
    print(
        f"  [TIME] trace elapsed: {trace_elapsed:.1f}s "
        f"({len(setters)} (setter,chain) tuples, {len(dep_vars)} dependent variables)",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Phase 3 — emit.
    # Default (no emit options): one indent=2 JSON blob written under the job's
    # output dir.  emit options map 1:1 onto vbt.output.emit.write_result:
    #   split   -> out_dir (also honours emit['out_dir'])
    #   no_code -> include_code=False
    #   format  -> "json" | "msgpack"
    #   zip     -> zip_=True
    # -----------------------------------------------------------------------
    emit_opts: Dict[str, Any] = dict(options.get("emit") or {})
    fmt = emit_opts.get("format", "json")
    no_code = bool(emit_opts.get("no_code", False))
    zip_ = bool(emit_opts.get("zip", False))
    split = bool(emit_opts.get("split", False)) or bool(emit_opts.get("out_dir"))

    base_dir = (ws.output_dir if ws is not None else (settings.JOBS_BASE_DIR / job_id / "output"))
    safe_var = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(variable))[:120] or "var"
    vbt_out_dir = Path(base_dir) / "vbt_trace" / safe_var
    vbt_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[vbt_trace] job=%s PHASE emit START (split=%s no_code=%s fmt=%s zip=%s)",
                job_id, split, no_code, fmt, zip_)
    _safe_set_status(ws, "running", progress="emit: writing result")
    print(f"--- Phase 3/3: emit (split={split} no_code={no_code} fmt={fmt} zip={zip_}) ---", flush=True)
    _t = time.monotonic()
    outputs: List[str] = []
    if split:
        out_dir = Path(emit_opts["out_dir"]) if emit_opts.get("out_dir") else (vbt_out_dir / safe_var)
        summary = write_result(
            out, out_dir=out_dir, fmt=fmt, include_code=not no_code, zip_=zip_, asm_dir=asm_dir,
        )
        # SPLIT writes a directory of parts (or <out_dir>.zip).
        if zip_:
            zpath = out_dir if out_dir.suffix == ".zip" else out_dir.with_suffix(".zip")
            outputs.append(str(zpath))
        else:
            outputs.append(str(out_dir))
    else:
        ext = "json" if fmt == "json" else "msgpack"
        out_path = vbt_out_dir / f"{safe_var}.{ext}"
        summary = write_result(
            out, out_path=out_path, fmt=fmt, include_code=not no_code, zip_=zip_, asm_dir=asm_dir,
        )
        if zip_:
            zpath = out_path if out_path.suffix == ".zip" else out_path.with_suffix(".zip")
            outputs.append(str(zpath))
        else:
            outputs.append(str(out_path))
    emit_elapsed = time.monotonic() - _t
    logger.info("[vbt_trace] job=%s PHASE emit DONE — %s elapsed=%.1fs",
                job_id, summary or outputs, emit_elapsed)
    print(f"  [TIME] emit elapsed: {emit_elapsed:.1f}s — {summary or outputs}", flush=True)

    elapsed = time.monotonic() - t_start
    manifest: Dict[str, Any] = {
        "job_id": job_id,
        "status": "success",
        "variable": variable,
        "language": language,
        "chain": [h.stem for h in chain_prefix],
        "setters": len(setters),
        "dependentVariables": len(dep_vars),
        "outputs": outputs,
        "elapsed": round(elapsed, 3),
        "precomputeElapsed": round(precompute_elapsed, 3),
        "precomputeId": precompute_job_id,
        "precomputeSkipped": bool(db_fast_path),
        "traceElapsed": round(trace_elapsed, 3),
        "emitElapsed": round(emit_elapsed, 3),
    }

    if ws is not None:
        try:
            result_files = ws.collect_output_files()
            manifest["output_files"] = len(result_files)
            _safe_set_status(ws, "success", result_files=result_files, progress="done")
        except Exception as exc:  # pragma: no cover
            logger.warning("[vbt_trace] could not finalize output files: %s", exc)
            _safe_set_status(ws, "success", progress="done")

    logger.info("[vbt_trace] job=%s COMPLETE — elapsed=%.1fs", job_id, elapsed)
    print(f"=== VBT trace complete: total elapsed {elapsed:.1f}s ===", flush=True)
    return manifest


# --------------------------------------------------------------------------- #
# Celery task
# --------------------------------------------------------------------------- #
@celery_app.task(
    bind=True,
    name="asm.vbt_trace",
)
def run_vbt_trace(self, job_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Run a VBT variable-lineage trace as a managed background job.

    ``options``::

        {
          "variable": "LG1OSI",            # required
          "language": "asm",               # "asm" | "cpp" (default "asm")
          "chain": [                        # required; TAIL (deepest hop) LAST
            {"stem": "da78", "type": "asm", "function": null}
          ],
          "blueprint_dir": "jobs/<id>/output",   # optional; else jobs/<id>/output
          "asm_dir": "temp/asm",                 # optional; else job source_path
          "graph_file": null,                    # optional; else <bp>/file_call_graph.json
          "trace_options": {                     # optional engine tunables/scope knobs
            "max_dep_var_depth": 2, "max_routes": 200, "candidate_stems": [...],
            "depvar_workers": 4, "disable_dependents": true, "home_hint": ...
            # forward-compat: dropped if unknown
          },
          "emit": {                              # optional output options
            "split": false, "out_dir": null, "no_code": false,
            "format": "json", "zip": false
          }
        }

    Returns the manifest from :func:`run_trace`, or ``{"job_id","status":"failed",
    "error":...}`` on failure (matching the existing task convention).
    """
    ws = _make_ws(job_id)
    _safe_set_status(ws, "running", operation="vbt_trace")

    log_file = ws.log_file if ws is not None else (settings.JOBS_BASE_DIR / job_id / "job.log")
    with redirect_output_to_log(log_file):
        with job_log_handler(log_file, job_id=job_id):
            touched_files = set()
            with intercept_file_reads(touched_files):
                try:
                    res = run_trace(job_id, options, ws=ws)
                    res["touched_files"] = sorted(touched_files)
                    try:
                        base_dir = ws.output_dir if ws is not None else (settings.JOBS_BASE_DIR / job_id / "output")
                        variable = options["variable"]
                        safe_var = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(variable))[:120] or "var"
                        vbt_out_dir = Path(base_dir) / "vbt_trace" / safe_var
                        vbt_out_dir.mkdir(parents=True, exist_ok=True)
                        import json
                        with open(vbt_out_dir / "touched_files.json", "w") as f:
                            json.dump(sorted(touched_files), f, indent=2)
                    except Exception as exc:
                        logger.warning("[vbt_trace] failed to write touched_files.json: %s", exc)
                    return res
                except SoftTimeLimitExceeded:
                    error_msg = "Task exceeded soft time limit and was aborted."
                    logger.error("[vbt_trace] job %s: %s", job_id, error_msg)
                    _safe_set_status(ws, "failed", error=error_msg, progress="failed")
                    return {"job_id": job_id, "status": "failed", "error": error_msg,
                            "variable": options.get("variable")}
                except Exception as exc:
                    error_msg = f"{type(exc).__name__}: {exc}"
                    logger.exception("[vbt_trace] job %s failed: %s", job_id, error_msg)
                    _safe_set_status(ws, "failed", error=error_msg, progress="failed")
                    return {"job_id": job_id, "status": "failed", "error": error_msg,
                            "variable": options.get("variable")}
