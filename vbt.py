"""Routes: VBT (Variable Backward Trace) pipeline and trace endpoints.

POST /vbt/pipeline          — run the VBT-minimal pipeline (universe + analysis + precompute)
POST /vbt/trace             — run a VBT trace against an existing KB / job
GET  /vbt/{job_id}/status   — poll job status (works for both pipeline and trace jobs)
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from api.config import settings
from api.db import get_db
from api.dependencies import require_admin, get_current_user
from api.models.knowledge_base import KnowledgeBase
from api.models.user import User
from api.schemas.common import JobResponse, JobStatus
from api.storage.workspace import WorkspaceManager

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class VbtPipelineRequest(BaseModel):
    """Request body for POST /vbt/pipeline."""

    kb_name: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable name for the knowledge base created from this run. "
            "Required when starting a new pipeline run."
        ),
    )

    datasource_username: Optional[str] = Field(
        default=None,
        description="Owner of the uploaded datasource (defaults to calling admin).",
    )
    datasource_folder: Optional[str] = Field(
        default=None,
        description="Versioned folder name from POST /v1/codebases.",
    )
    source_path: Optional[str] = Field(
        default=None,
        description="Absolute path to a source directory already on the server.",
    )
    zip_path: Optional[str] = Field(
        default=None,
        description="Absolute path to a .zip file on the server.",
    )

    asm_only: bool = Field(default=False)
    cpp_only: bool = Field(default=False)
    exclude: List[str] = Field(default=[])
    search_paths: List[str] = Field(default=[])
    macro_libs: List[str] = Field(default=[])
    no_coverage_review: bool = Field(default=True)
    skip_business_summary: bool = Field(default=True)

    cache_in_job_dir: bool = Field(
        default=True,
        description="Store VBT cache co-located with blueprints (per-project).",
    )
    rebuild: bool = Field(default=False, description="Force-rebuild all caches.")
    cfg_warm: bool = Field(default=True, description="Run parallel Clang cfg pre-warm.")
    workers: Optional[int] = Field(default=None, description="VBT_WORKERS override.")
    require_db_precompute: bool = Field(
        default=True,
        description=(
            "Fail the VBT pipeline if DB-backed trace artifacts cannot be built. "
            "Disable only for debugging; without these artifacts large traces can "
            "fall back to corpus-wide scans."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "VbtPipelineRequest":
        if not self.kb_name:
            raise ValueError("kb_name is required.")
        if not self.datasource_folder and not self.source_path and not self.zip_path:
            raise ValueError("Provide one of: datasource_folder, source_path, or zip_path.")
        return self


class HopSpec(BaseModel):
    stem: str = Field(..., description="File stem (e.g. 'aa71', 'dw730000').")
    type: str = Field(default="asm", description="'asm' or 'cpp'.")
    function: Optional[str] = Field(default=None, description="C++ function name (for cpp hops).")


class VbtTraceRequest(BaseModel):
    """Request body for POST /vbt/trace."""

    kb_id: str = Field(..., description="Knowledge base / job ID with completed blueprints.")
    variable: str = Field(..., description="Variable name to trace.")
    language: str = Field(default="asm", description="'asm' or 'cpp'.")
    chain: List[HopSpec] = Field(
        ...,
        min_length=1,
        description="Hop chain (TAIL last). At least one hop required.",
    )
    trace_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Engine tunables: max_dep_var_depth, max_routes, candidate_stems, "
            "disable_dependents, home_hint, etc."
        ),
    )
    allow_live_precompute: bool = Field(
        default=False,
        description=(
            "Allow trace workers to fall back to live corpus precompute when DB "
            "artifacts are incomplete. Debug only; on large KBs this can take hours."
        ),
    )
    emit: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output options: split, out_dir, no_code, format ('json'|'msgpack'), zip.",
    )


class VbtDbRepairRequest(BaseModel):
    """Request body for POST /vbt/{kb_id}/repair."""

    only: List[str] = Field(
        default_factory=lambda: ["fn_graph_adj", "cfg"],
        description=(
            "DB-backed VBT artifacts to rebuild incrementally. Common values: "
            "fn_graph_adj, fn_facts, reverse_adj, forward_file_adj, "
            "asm_setter_map, cpp_setter_map, cpp_file_writes, cfg."
        ),
    )
    blueprint_dir: Optional[str] = Field(
        default=None,
        description="Override blueprint directory. Defaults to jobs/<kb_id>/output discovery.",
    )
    asm_dir: Optional[str] = Field(
        default=None,
        description="Override source directory. Defaults to the KB source_path.",
    )
    graph_file: Optional[str] = Field(
        default=None,
        description="Override file_call_graph.json path.",
    )
    cache_in_job_dir: Optional[bool] = Field(
        default=None,
        description="Use the KB's per-project VBT cache. Defaults to the original pipeline option.",
    )
    mark_kb_success: bool = Field(
        default=True,
        description=(
            "After repair, mark the KB success only if all required DB trace artifacts "
            "are present; otherwise leave/mark it failed."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_source_dir(root: Path) -> Path:
    """Find the directory that directly contains .cpp/.asm source files.

    The stored source_path may point to the datasource root while actual source
    files live one level deeper (e.g. a zip extracted with a named subfolder).
    Mirrors the search pattern of ``find_blueprint_dir``.
    """
    _EXTS = ("*.cpp", "*.asm", "*.mac")
    for ext in _EXTS:
        if next(root.glob(ext), None) is not None:
            return root
    try:
        for subdir in sorted(root.iterdir()):
            if subdir.is_dir():
                for ext in _EXTS:
                    if next(subdir.glob(ext), None) is not None:
                        return subdir
    except (OSError, PermissionError):
        pass
    return root


def _resolve_source_path(
    request: VbtPipelineRequest,
    caller_username: str,
    ws_input_dir: Path,
) -> Path:
    if request.zip_path:
        zip_p = Path(request.zip_path)
        if not zip_p.is_absolute():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="zip_path must be absolute.")
        if not zip_p.exists() or not zipfile.is_zipfile(zip_p):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Invalid zip: {zip_p}")
        ws_input_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_p, "r") as zf:
            for member in zf.namelist():
                resolved = (ws_input_dir / member).resolve()
                if not str(resolved).startswith(str(ws_input_dir.resolve())):
                    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="ZIP contains unsafe paths.")
            zf.extractall(ws_input_dir)
        macosx_dir = ws_input_dir / "__MACOSX"
        if macosx_dir.exists():
            shutil.rmtree(macosx_dir)
        return ws_input_dir

    if request.datasource_folder:
        owner = request.datasource_username or caller_username
        p = settings.DATASOURCE_DIR / owner / request.datasource_folder
        if not p.exists() or not p.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Datasource '{request.datasource_folder}' not found for user '{owner}'.",
            )
        return p.resolve()

    p = Path(request.source_path)
    if not p.is_absolute():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="source_path must be absolute.")
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"source_path not found: {p}")
    return p.resolve()


# ---------------------------------------------------------------------------
# POST /vbt/pipeline
# ---------------------------------------------------------------------------

@router.post(
    "/vbt/pipeline",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    summary="Run VBT-minimal pipeline (admin only)",
    description=(
        "Admin-only. Runs only the steps needed for VBT readiness: universe build, "
        "full analysis (blueprint emission), and VBT precompute (call graph + indexes "
        "+ cfg-warm). Skips index-DB ingestion, identity/hop indexes, and chain summaries."
    ),
)
def submit_vbt_pipeline(
    request: VbtPipelineRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> JobResponse:
    from datetime import datetime, timezone

    existing = db.query(KnowledgeBase).filter(KnowledgeBase.name == request.kb_name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A knowledge base named '{request.kb_name}' already exists (id: {existing.id}).",
        )

    job_id = str(uuid4())
    ws = WorkspaceManager(job_id)
    for d in (ws.root, ws.input_dir, ws.output_dir, ws.cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    ws.set_status("pending", operation="vbt_pipeline")

    source_path = _resolve_source_path(request, admin.username, ws.input_dir)

    kb = KnowledgeBase(
        id=job_id,
        name=request.kb_name,
        created_by=admin.username,
        datasource_username=request.datasource_username,
        datasource_folder=request.datasource_folder or (request.zip_path and Path(request.zip_path).stem),
        source_path=str(source_path),
        status="pending",
    )
    db.add(kb)
    db.commit()

    options = request.model_dump(
        exclude={"kb_name", "datasource_username", "datasource_folder", "source_path", "zip_path"}
    )
    options["source_path"] = str(source_path)
    ws.save_pipeline_options(options)

    from api.tasks.vbt_pipeline_task import run_vbt_pipeline
    task = run_vbt_pipeline.apply_async(args=[job_id, options])
    ws.set_status("pending", operation="vbt_pipeline", celery_task_id=task.id)

    logger.info(
        "VBT pipeline dispatched — job=%s kb=%r src=%s user=%s task=%s",
        job_id, request.kb_name, source_path, admin.username, task.id,
    )

    meta = ws.get_job_meta()
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        operation="vbt_pipeline",
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        celery_task_id=task.id,
        step_progress=meta.get("step_progress"),
    )


# ---------------------------------------------------------------------------
# POST /vbt/trace
# ---------------------------------------------------------------------------

@router.post(
    "/vbt/trace",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    summary="Run a VBT trace",
    description=(
        "Dispatch a VBT variable-lineage trace against a knowledge base that has "
        "completed the pipeline (blueprints must exist). The trace runs as a Celery "
        "background job; poll GET /vbt/{job_id}/status for progress."
    ),
)
def submit_vbt_trace(
    request: VbtTraceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == request.kb_id).first()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base '{request.kb_id}' not found.",
        )

    job_id = str(uuid4())
    ws = WorkspaceManager(job_id)
    for d in (ws.root, ws.output_dir):
        d.mkdir(parents=True, exist_ok=True)
    ws.set_status("pending", operation="vbt_trace")

    options: Dict[str, Any] = {
        "kb_id": request.kb_id,
        "variable": request.variable,
        "language": request.language,
        "chain": [h.model_dump() for h in request.chain],
        "allow_live_precompute": request.allow_live_precompute,
    }

    kb_ws = WorkspaceManager(request.kb_id)
    kb_opts = kb_ws.get_pipeline_options() or {}
    bp_out = settings.JOBS_BASE_DIR / request.kb_id / "output"
    from api.tasks._task_utils import find_blueprint_dir
    bp_dir = find_blueprint_dir(bp_out)
    if bp_dir:
        options["blueprint_dir"] = str(bp_dir)
    sp = kb_opts.get("source_path") or (kb.source_path if kb else None)
    if sp:
        options["asm_dir"] = str(_find_source_dir(Path(sp)))
    options["cache_in_job_dir"] = bool(kb_opts.get("cache_in_job_dir", True))

    if request.trace_options:
        options["trace_options"] = request.trace_options
    if request.emit:
        options["emit"] = request.emit

    from api.tasks.vbt_trace_task import run_vbt_trace
    task = run_vbt_trace.apply_async(args=[job_id, options])
    ws.set_status("pending", operation="vbt_trace", celery_task_id=task.id)

    logger.info(
        "VBT trace dispatched — job=%s kb=%s var=%s user=%s task=%s",
        job_id, request.kb_id, request.variable, current_user.username, task.id,
    )

    meta = ws.get_job_meta()
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        operation="vbt_trace",
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        celery_task_id=task.id,
    )


# ---------------------------------------------------------------------------
# POST /vbt/{kb_id}/repair
# ---------------------------------------------------------------------------

@router.post(
    "/vbt/{kb_id}/repair",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    summary="Repair DB-backed VBT artifacts for an existing KB",
    description=(
        "Admin-only. Incrementally rebuilds selected DB-backed VBT artifacts for "
        "an existing KB/job, without rerunning universe build, full analysis, or "
        "cfg-warm. Poll GET /vbt/{kb_id}/status for progress."
    ),
)
def repair_vbt_db_artifacts(
    kb_id: str,
    request: VbtDbRepairRequest = Body(default_factory=VbtDbRepairRequest),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> JobResponse:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base '{kb_id}' not found.",
        )

    ws = WorkspaceManager(kb_id)
    for d in (ws.root, ws.output_dir):
        d.mkdir(parents=True, exist_ok=True)

    options: Dict[str, Any] = {
        "only": request.only,
        "mark_kb_success": request.mark_kb_success,
    }
    if request.blueprint_dir:
        options["blueprint_dir"] = request.blueprint_dir
    else:
        bp_out = settings.JOBS_BASE_DIR / kb_id / "output"
        from api.tasks._task_utils import find_blueprint_dir
        bp_dir = find_blueprint_dir(bp_out)
        if bp_dir:
            options["blueprint_dir"] = str(bp_dir)
    if request.asm_dir:
        options["asm_dir"] = request.asm_dir
    elif kb.source_path:
        options["asm_dir"] = str(_find_source_dir(Path(kb.source_path)))
    if request.graph_file:
        options["graph_file"] = request.graph_file

    kb_opts = ws.get_pipeline_options() or {}
    options["cache_in_job_dir"] = (
        bool(request.cache_in_job_dir)
        if request.cache_in_job_dir is not None
        else bool(kb_opts.get("cache_in_job_dir", True))
    )

    from api.tasks.vbt_precompute_task import run_vbt_db_repair
    task = run_vbt_db_repair.apply_async(args=[kb_id, options])
    ws.set_status(
        "pending",
        operation="vbt_db_repair",
        celery_task_id=task.id,
        progress=f"queued: {', '.join(options['only'])}",
    )

    logger.info(
        "VBT DB repair dispatched — kb=%s only=%s user=%s task=%s",
        kb_id, options["only"], admin.username, task.id,
    )

    meta = ws.get_job_meta()
    return JobResponse(
        job_id=kb_id,
        status=JobStatus.PENDING,
        operation="vbt_db_repair",
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        progress=meta.get("progress"),
        celery_task_id=task.id,
        step_progress=meta.get("step_progress"),
    )


# ---------------------------------------------------------------------------
# GET /vbt/{job_id}/status
# ---------------------------------------------------------------------------

@router.get(
    "/vbt/{job_id}/status",
    response_model=JobResponse,
    summary="Poll VBT job status",
    description="Poll the status of a VBT pipeline or trace job.",
)
def vbt_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    try:
        ws = WorkspaceManager(job_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    meta = ws.get_job_meta()
    if meta is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    return JobResponse(
        job_id=job_id,
        status=JobStatus(meta.get("status", "pending")),
        operation=meta.get("operation", "unknown"),
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        error=meta.get("error"),
        progress=meta.get("progress"),
        celery_task_id=meta.get("celery_task_id"),
        step_progress=meta.get("step_progress"),
    )
