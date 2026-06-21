"""Celery task: run the minimal pipeline steps needed for VBT readiness, then
VBT precompute — all in one dispatch.

This is a slimmed-down variant of ``pipeline_task.run_full_pipeline`` that runs
ONLY the steps the VBT engine depends on:

  * Step 1 — universe build (variable name sets)
  * Step 2 — full analysis (per-file parse → cross-file linking → blueprint
              emission)
  * VBT precompute — call graph + modifier index + const resolver + cfg-warm
              + optional DB artifacts

Steps 2.5–8 (access_identity enrichment, modifier index, identity index, hop
index, process seeds, chain summaries) are NOT run — they serve the index DB /
setter-analysis / chain-summary features and are not required for
``trace_root_variable``.
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from celery.exceptions import SoftTimeLimitExceeded

PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.config import settings
from api.tasks.celery_app import celery_app
from api.tasks._task_utils import redirect_output_to_log, find_blueprint_dir, job_log_handler
from api.tasks.universe_task import _run_universe
from api.tasks.analysis_task import _run_analysis, _trim_heap, _current_rss_mb, _current_cgroup_usage_mb
from api.storage.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


def _safe_set_status(ws: Optional[WorkspaceManager], status: str, **kw: Any) -> None:
    if ws is None:
        return
    try:
        ws.set_status(status, **kw)
    except Exception as exc:
        logger.warning("[vbt_pipeline] could not set status %s: %s", status, exc)


def _update_kb_status(kb_id: str, status: str, error: str | None = None) -> None:
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
        logger.warning("[vbt_pipeline] could not update KB status for %s: %s", kb_id, exc)


def _mark_step(ws: WorkspaceManager, step_name: str, status: str) -> None:
    try:
        ws.set_step_status(step_name, status)
    except Exception as exc:
        logger.warning("Could not mark step %s as %s: %s", step_name, status, exc)


def _log_step_ram(label: str) -> None:
    rss_mb = _current_rss_mb()
    cgroup_mb = _current_cgroup_usage_mb()
    logger.info("[RAM] %s — RSS=%d MB  cgroup=%d MB", label, rss_mb, cgroup_mb)
    print(f"  [RAM] {label}: RSS={rss_mb} MB | cgroup={cgroup_mb} MB", flush=True)


def _run_vbt_pipeline(ws: WorkspaceManager, options: Dict[str, Any]) -> Dict[str, Any]:
    """Universe → analysis → VBT precompute, returning a combined manifest."""

    source_path: str = options["source_path"]
    job_id = ws.job_id
    t_start = time.monotonic()
    _log_step_ram("VBT Pipeline START")

    steps: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------
    # Step 1 — universe build
    # -------------------------------------------------------------------
    universe_opts: Dict[str, Any] = {
        "source_path": source_path,
        "asm_only": options.get("asm_only", False),
        "cpp_only": options.get("cpp_only", False),
        "exclude": options.get("exclude", []),
    }

    logger.info("[vbt_pipeline] job=%s STEP universe START", job_id)
    print("=== Step 1/3: Building variable universe ===", flush=True)
    _mark_step(ws, "step1", "running")
    _log_step_ram("Step 1 START")
    _t = time.monotonic()
    try:
        universe_result = _run_universe(ws, universe_opts, update_status=False)
    except Exception:
        _mark_step(ws, "step1", "failed")
        raise
    _mark_step(ws, "step1", "success")
    step1_elapsed = time.monotonic() - _t
    _log_step_ram("Step 1 END")
    logger.info(
        "[vbt_pipeline] job=%s STEP universe DONE — asm=%d cpp=%d elapsed=%.1fs",
        job_id,
        universe_result.get("asm_variables", 0),
        universe_result.get("cpp_variables", 0),
        step1_elapsed,
    )
    print(f"  [TIME] Step 1 elapsed: {step1_elapsed:.1f}s", flush=True)
    steps.append({"name": "universe", "elapsed": round(step1_elapsed, 3)})

    gc.collect()
    _trim_heap("between universe and analysis (pre-fork trim)")
    _log_step_ram("Post-step1 heap trim")

    # -------------------------------------------------------------------
    # Step 2 — full analysis (blueprint emission)
    # -------------------------------------------------------------------
    analysis_opts: Dict[str, Any] = {
        "source_path": source_path,
        "search_paths": options.get("search_paths", []),
        "macro_libs": options.get("macro_libs", []),
        "no_coverage_review": options.get("no_coverage_review", True),
        "skip_business_summary": options.get("skip_business_summary", True),
        "exclude": options.get("exclude", []),
        "load_universe": True,
        "universe_job_id": ws.job_id,
        "resume_from": "",
    }

    logger.info("[vbt_pipeline] job=%s STEP analysis START", job_id)
    print("=== Step 2/3: Running full analysis (blueprint emission) ===", flush=True)
    _mark_step(ws, "step2", "running")
    _log_step_ram("Step 2 START")
    _t = time.monotonic()
    try:
        analysis_result = _run_analysis(ws, analysis_opts, update_status=False)
    except Exception:
        _mark_step(ws, "step2", "failed")
        raise
    _mark_step(ws, "step2", "success")
    step2_elapsed = time.monotonic() - _t
    _log_step_ram("Step 2 END")
    logger.info(
        "[vbt_pipeline] job=%s STEP analysis DONE — files=%d elapsed=%.1fs",
        job_id,
        analysis_result.get("files_analyzed", 0),
        step2_elapsed,
    )
    print(f"  [TIME] Step 2 elapsed: {step2_elapsed:.1f}s", flush=True)
    steps.append({"name": "analysis", "elapsed": round(step2_elapsed, 3)})

    gc.collect()
    _trim_heap("between analysis and VBT precompute")
    _log_step_ram("Post-step2 heap trim")

    # -------------------------------------------------------------------
    # Step 3 — VBT precompute (call graph + indexes + cfg-warm + DB)
    # -------------------------------------------------------------------
    from api.tasks.vbt_precompute_task import run_precompute
    from api.routes.vbt import _find_source_dir

    blueprint_dir = find_blueprint_dir(ws.output_dir) or ws.output_dir
    asm_dir = _find_source_dir(Path(source_path))

    precompute_options: Dict[str, Any] = {
        "blueprint_dir": str(blueprint_dir),
        "asm_dir": str(asm_dir),
        "cache_in_job_dir": options.get("cache_in_job_dir", True),
        "rebuild": options.get("rebuild", False),
        "cfg_warm": options.get("cfg_warm", True),
        "workers": options.get("workers"),
        "require_db_precompute": options.get("require_db_precompute", True),
    }

    logger.info("[vbt_pipeline] job=%s STEP vbt_precompute START", job_id)
    print("=== Step 3/3: VBT precompute (call graph + indexes + cfg-warm) ===", flush=True)
    _mark_step(ws, "vbt_precompute", "running")
    _log_step_ram("Step 3 START")
    _t = time.monotonic()
    try:
        precompute_manifest = run_precompute(job_id, precompute_options, ws=None)
    except Exception:
        _mark_step(ws, "vbt_precompute", "failed")
        raise
    _mark_step(ws, "vbt_precompute", "success")
    step3_elapsed = time.monotonic() - _t
    _log_step_ram("Step 3 END")
    logger.info(
        "[vbt_pipeline] job=%s STEP vbt_precompute DONE — elapsed=%.1fs",
        job_id, step3_elapsed,
    )
    print(f"  [TIME] Step 3 elapsed: {step3_elapsed:.1f}s", flush=True)
    steps.append({"name": "vbt_precompute", "elapsed": round(step3_elapsed, 3)})

    # -------------------------------------------------------------------
    # Finalize
    # -------------------------------------------------------------------
    result_files = ws.collect_output_files()
    _safe_set_status(ws, "success", result_files=result_files)

    total_elapsed = time.monotonic() - t_start
    _log_step_ram("VBT Pipeline END")

    manifest: Dict[str, Any] = {
        "job_id": job_id,
        "status": "success",
        "steps": steps,
        "files_analyzed": analysis_result.get("files_analyzed", 0),
        "cache_dir": precompute_manifest.get("cache_dir", ""),
        "graph_path": precompute_manifest.get("graph_path", ""),
        "precompute_steps": precompute_manifest.get("steps", []),
        "output_files": len(result_files),
        "elapsed": round(total_elapsed, 3),
    }

    logger.info(
        "[vbt_pipeline] job=%s COMPLETE — elapsed=%.1fs",
        job_id, total_elapsed,
    )
    print(f"=== VBT pipeline complete: total elapsed {total_elapsed:.1f}s ===", flush=True)
    return manifest


@celery_app.task(
    bind=True,
    name="asm.vbt_pipeline",
)
def run_vbt_pipeline(self, job_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Run the minimal pipeline for VBT readiness in a single dispatch.

    Runs universe build → full analysis → VBT precompute.  Skips all index-DB
    ingestion steps, setter-analysis indexes, and chain summaries.

    ``options``::

        {
          "source_path": "/data/codebases/myapp",   # required
          "asm_only": false,                         # universe: ASM files only
          "cpp_only": false,                         # universe: C++ files only
          "exclude": [],                             # glob patterns to skip
          "search_paths": [],                        # extra include dirs
          "macro_libs": [],                          # macro library files
          "no_coverage_review": true,                # skip coverage review (default true)
          "skip_business_summary": true,             # skip LLM summaries (default true)
          "cache_in_job_dir": true,                  # VBT cache co-located with blueprints
          "rebuild": false,                          # force rebuild caches
          "cfg_warm": true,                          # parallel Clang cfg pre-warm
          "workers": null                            # VBT_WORKERS override
        }

    Returns a manifest with step timings and output paths, or
    ``{"job_id", "status": "failed", "error": ...}`` on failure.
    """
    ws = WorkspaceManager(job_id)
    _safe_set_status(ws, "running", operation="vbt_pipeline")
    _update_kb_status(job_id, "running")

    log_file = ws.log_file
    with redirect_output_to_log(log_file):
        with job_log_handler(log_file, job_id=job_id):
            try:
                result = _run_vbt_pipeline(ws, options)
                _update_kb_status(job_id, "success")
                return result
            except SoftTimeLimitExceeded:
                error_msg = "Task exceeded soft time limit and was aborted."
                logger.error("[vbt_pipeline] job %s: %s", job_id, error_msg)
                _safe_set_status(ws, "failed", error=error_msg)
                _update_kb_status(job_id, "failed", error=error_msg)
                return {"job_id": job_id, "status": "failed", "error": error_msg}
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.exception("[vbt_pipeline] job %s failed: %s", job_id, error_msg)
                _safe_set_status(ws, "failed", error=error_msg)
                _update_kb_status(job_id, "failed", error=error_msg)
                return {"job_id": job_id, "status": "failed", "error": error_msg}
