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
from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    """Request body for POST /vbt/pipeline.

    Minimal request is just ``kb_name`` + ``source_path`` — every other field
    has a production-ready default.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kb_name": "my-kb",
                "source_path": "/abs/path/to/source",
            }
        }
    )

    kb_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing knowledge-base / job ID. When provided, the pipeline "
            "reuses this ID as the job ID (no new UUID is generated) and "
            "re-runs against the existing workspace. Mutually exclusive with kb_name."
        ),
    )
    kb_name: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable name for the knowledge base created from this run. "
            "Required when starting a new pipeline run (i.e. when kb_id is not provided)."
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
    require_fast_path: bool = Field(
        default=True,
        description=(
            "Fail the pipeline (instead of WARN) when the health verifier finds missing "
            "speedup artifacts — without them large traces silently fall back to multi-minute "
            "corpus scans. Disable only to accept a degraded KB."
        ),
    )
    include_chain_facts: bool = Field(
        default=True,
        description=(
            "Also build the chain-discovery artifacts (asm_setters_full + coverages + "
            "chain_facts) and the per-KB lineage scans as a pipeline step, so the FIRST "
            "/vbt/lineage or /vbt/trace-all-chains query doesn't pay a multi-minute "
            "on-demand build."
        ),
    )
    warm: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional eager-warm trace specs run after precompute to fill the route + trace "
            "caches. Each: {variable, language, hops: [{stem,type,function}] or "
            "['stem:type[:fn]', ...], plus optional engine caps (max_dep_var_depth, ...)}. "
            "Requires knowing the chain; for variable-only warming use warm_lineage_variables."
        ),
    )
    warm_lineage_variables: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional variable names to fully warm via the lineage path after precompute: "
            "discovers each variable's chains and runs the deep per-chain traces, so the first "
            "production /vbt/lineage query is a cache hit. Failure-isolated (a warm error never "
            "fails the pipeline); outputs land under output/vbt_lineage_warm/<variable>/."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "VbtPipelineRequest":
        if not self.kb_id and not self.kb_name:
            raise ValueError("Either kb_id (existing) or kb_name (new) must be provided.")
        if not self.kb_id and not self.datasource_folder and not self.source_path and not self.zip_path:
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
            "depvar_workers, disable_dependents, home_hint, etc."
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
    inline_result: Optional[bool] = Field(
        default=None,
        description=(
            "Sync route only. True forces the emitted JSON into the HTTP response, "
            "False disables it. Default auto-inlines only small JSON outputs."
        ),
    )
    inline_result_max_bytes: int = Field(
        default=2_000_000,
        ge=0,
        description="Sync route auto-inline limit when inline_result is not set.",
    )


class VbtLcaTraceRequest(BaseModel):
    """Request body for POST /vbt/lca-trace — the DB-only LCA chain trace.

    Takes only a variable (+ language, optional domain/prune); the LCA(s) and root→LCA chains are
    derived automatically. Strictly index.db only — the KB must have completed precompute AND the
    standalone chains-precompute step (``python -m vbt.precompute.chain_facts_db <kb_id>``)."""

    kb_id: str = Field(..., description="Knowledge base / job ID with completed precompute.")
    variable: str = Field(..., description="Variable name to trace.")
    language: str = Field(default="asm", description="'asm' or 'cpp'.")
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain hint; 'plastic_authentication' forces the LCA to "
                    "dw710000::callPlasticAuthenticationComponentInterface (skips LCA computation).")
    prune: bool = Field(
        default=False,
        description="Conservatively drop provably-infeasible chains (needs chain_facts precompute).")
    allow_missing_forced_lca: bool = Field(
        default=False,
        description="If the domain-forced LCA node is absent from the graph, return zero chains + "
                    "a warning instead of hard-failing.")
    allow_unattributed_setters: bool = Field(
        default=False,
        description="Exclude C++ setters that can't be attributed to a (stem,fn) graph node "
                    "(+ warning + excludedSetterCount) instead of hard-failing.")
    trace_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tunables: candidate_stems, home_hint, max_chains_per_lca, max_chain_depth, "
                    "max_feasibility_checks.")
    debug: bool = Field(
        default=False,
        description="Also write lca_trace.debug.json (internal diagnostics; never in the API payload).")


class VbtAllChainsTraceRequest(BaseModel):
    """Request body for POST /vbt/all-chains-trace — the multi-chain coordinator.

    Discovers a variable's setters → call-graph LCA(s) → root→LCA chains (DB-only, same as
    /vbt/lca-trace), then runs a FULL ``trace_root_variable`` for each discovered chain in a single
    process — sharing the hoisted RouteEngine + low-level caches across the loop — and writes one
    consolidated ``vbt_all_chains_trace.json``. The KB must have completed VBT precompute AND the
    standalone chains-precompute step (``python -m vbt.precompute.chain_facts_db <kb_id>``)."""

    kb_id: str = Field(..., description="Knowledge base / job ID with completed precompute.")
    variable: str = Field(..., description="Variable name to trace.")
    language: str = Field(default="asm", description="'asm' or 'cpp'.")
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain hint for LCA computation; 'plastic_authentication' forces the "
                    "LCA to dw710000::callPlasticAuthenticationComponentInterface.")
    prune: bool = Field(
        default=False,
        description="Conservatively drop provably-infeasible chains (needs chain_facts precompute).")
    allow_missing_forced_lca: bool = Field(
        default=False,
        description="If the domain-forced LCA node is absent from the graph, return zero chains.")
    allow_unattributed_setters: bool = Field(
        default=False,
        description="Exclude C++ setters that can't be attributed to a graph node.")
    split_chains: bool = Field(
        default=False,
        description="If true, write ONE fully self-contained JSON per chain (each with its OWN "
                    "codeBlocks) into a vbt_all_chains_trace/ directory plus an index.json manifest, "
                    "instead of the single consolidated vbt_all_chains_trace.json with root codeBlocks.")
    trace_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tunables: candidate_stems, home_hint, max_chains_per_lca, max_chain_depth, "
                    "max_feasibility_checks, max_chains_to_trace (tracing budget, default 50), "
                    "wall_clock_budget_sec, plus engine tunables (max_routes, max_dep_var_depth, …).")
    debug: bool = Field(
        default=False,
        description="Also write debug files (lca discovery diagnostics).")


class VbtTraceAllChainsRequest(BaseModel):
    """Request body for POST /vbt/trace-all-chains — the per-chain coordinator.

    Takes only a ``kb_id`` + ``variable`` (+ language). Discovers ALL of the variable's file chains
    (setters → call-graph LCA(s) → root→LCA chains, DB-only — same core as /vbt/lca-trace), runs a
    full ``trace_root_variable`` for each, and ALWAYS writes one fully self-contained JSON per chain
    (``vbt_all_chains_trace/chain_NNNN.json``) plus an ``index.json`` manifest — never a single
    consolidated file. The KB must have completed VBT precompute AND the standalone chains-precompute
    step (``python -m vbt.precompute.chain_facts_db <kb_id>``)."""

    kb_id: str = Field(..., description="Knowledge base / job ID with completed precompute.")
    variable: str = Field(..., description="Variable name to trace.")
    language: str = Field(default="asm", description="'asm' or 'cpp'.")
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain hint for LCA computation; 'plastic_authentication' forces the "
                    "LCA to dw710000::callPlasticAuthenticationComponentInterface.")
    prune: bool = Field(
        default=False,
        description="Conservatively drop provably-infeasible chains (needs chain_facts precompute).")
    allow_missing_forced_lca: bool = Field(
        default=False,
        description="If the domain-forced LCA node is absent from the graph, return zero chains.")
    allow_unattributed_setters: bool = Field(
        default=False,
        description="Exclude C++ setters that can't be attributed to a graph node.")
    trace_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tunables: candidate_stems, home_hint, max_chains_per_lca, max_chain_depth, "
                    "max_feasibility_checks, max_chains_to_trace (tracing budget, default 50), "
                    "wall_clock_budget_sec, plus engine tunables (max_routes, max_dep_var_depth, …).")
    debug: bool = Field(
        default=False,
        description="Also write debug files (lca discovery diagnostics).")


class VbtLineageRequest(BaseModel):
    """Request body for POST /vbt/lineage — the full variable-lineage coordinator.

    Takes ONLY a ``kb_id`` + ``variable``; language is auto-detected from the precomputed
    modifier index (a name in both universes traces both). Discovers ALL of the variable's
    file chains (DB-only), runs a DEEP trace for each (dependent-variable recursion default
    10 vs the trace endpoints' 1), folds every chain into an explicit lineage graph, and
    classifies every terminal variable (user_input / database / constant / external /
    working_storage / record_field / unresolved) with the evidence that justified it.

    Strict DB mode: the KB's index.db must contain every required VBT artifact; there is NO
    live-precompute or JSON-file fallback — a missing artifact fails the job with the exact
    list. The response payload also carries a ``db_audit`` block reporting missing
    call-graph/fn-graph/entry-point links found in the stored artifacts."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kb_id": "<kb-uuid>",
                "variable": "trxnResult",
            }
        }
    )

    kb_id: str = Field(..., description="Knowledge base / job ID with completed VBT precompute.")
    variable: str = Field(..., description="Variable name to trace (the ONLY required trace input).")
    language: Optional[str] = Field(
        default=None,
        description="'asm' or 'cpp'. Omit to auto-detect from the modifier index; a variable "
                    "present in both universes is traced in both.")
    domain: Optional[str] = Field(
        default=None, description="Optional domain hint for LCA chain discovery.")
    prune: bool = Field(
        default=False,
        description="Conservatively drop provably-infeasible chains (needs chain_facts precompute).")
    allow_missing_forced_lca: bool = Field(default=False)
    allow_unattributed_setters: bool = Field(default=False)
    split_chains: bool = Field(
        default=True,
        description="Default: stream one self-contained JSON per chain "
                    "(vbt_lineage/chain_<lang>_NNNN.json) plus an index.json manifest — bounded "
                    "memory, partial results survive a timeout. False = single consolidated "
                    "lineage.json (can be very large on many-chain variables).")
    trace_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tunables: max_dep_var_depth (default 10), max_chains_to_trace (default 50), "
                    "wall_clock_budget_sec, candidate_stems, home_hint, engine limits "
                    "(max_routes, max_paths, …), plus classifier marker overrides "
                    "(db_read_markers, user_input_markers, input_msg_prefixes, "
                    "working_storage_prefixes).")
    debug: bool = Field(default=False, description="Also write discovery debug diagnostics.")


class VbtSetterOrderRequest(BaseModel):
    """Request body for POST /vbt/setter-order — the DB-only setter-ORDER trace.

    Takes a variable (+ language); derives its setters → call-graph LCA(s) and emits one FLAT
    ordered ``setters`` array per LCA tree (static CFG/source order, NOT a runtime guarantee).
    Strictly index.db only — the KB must have completed VBT precompute."""

    kb_id: str = Field(..., description="Knowledge base / job ID with completed precompute.")
    variable: str = Field(..., description="Variable name to trace.")
    language: str = Field(default="asm", description="'asm' or 'cpp'.")
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain hint; 'plastic_authentication' forces the LCA to "
                    "dw710000::callPlasticAuthenticationComponentInterface (skips LCA computation).")
    allow_missing_forced_lca: bool = Field(
        default=False,
        description="If the domain-forced LCA node is absent from the graph, return zero trees + "
                    "a warning instead of hard-failing.")
    allow_unattributed_setters: bool = Field(
        default=True,
        description="Exclude C++ setters that can't be attributed to a (stem,fn) graph node "
                    "(+ warning + excludedSetterCount) instead of hard-failing. Defaults TRUE for "
                    "setter-order (vs the LCA trace which defaults false).")
    trace_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tunables: candidate_stems, home_hint, max_depth (16), max_nodes (4000), "
                    "asm_cfg_order (true), include_call_path (false, diagnostic), "
                    "include_cpp_condition_hints (false).")
    debug: bool = Field(
        default=False,
        description="Also write setter_order.debug.json (internal diagnostics; never in the payload).")

    @model_validator(mode="after")
    def _validate_language(self) -> "VbtSetterOrderRequest":
        if self.language not in ("asm", "cpp"):
            raise ValueError(f"language must be 'asm' or 'cpp', got {self.language!r}")
        return self


class VbtDbRepairRequest(BaseModel):
    """Request body for POST /vbt/{kb_id}/repair."""

    only: List[str] = Field(
        default_factory=lambda: ["fn_graph_adj", "cfg"],
        description=(
            "DB-backed VBT artifacts to rebuild incrementally. Common values: "
            "fn_graph_adj, fn_facts, reverse_adj, forward_file_adj, "
            "asm_setter_map, cpp_setter_map, cpp_file_writes, depvar_direct, cfg. "
            "Use ['chain_facts', 'lineage_scans'] to (re)run pipeline Step 3.5 — the "
            "chains precompute + /vbt/lineage corpus scans — on an existing KB."
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


class VbtUiExportRequest(BaseModel):
    """Request body for POST /vbt/uiexport."""

    kb_id: str = Field(
        ...,
        description=(
            "Knowledge base ID. Trace output is read from and uiexport output "
            "is written to jobs/{kb_id}/output/."
        ),
    )
    variable: str = Field(
        ...,
        description=(
            "Root variable name for which VBT trace was run. "
            "The trace output must exist at vbt_trace/{variable}/."
        ),
    )
    no_llm: bool = Field(
        default=False,
        description="Skip Gemini prose fill; use deterministic backfill only.",
    )
    force: bool = Field(
        default=False,
        description="Ignore prose cache and regenerate all prose.",
    )
    max_paths: int = Field(
        default=6,
        description="Maximum paths per setter in the structural layer.",
    )
    variables: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated list of dependent variable names to export. "
            "Defaults to root variable + all clickable dependent variables."
        ),
    )
    cache_in_job_dir: bool = Field(
        default=True,
        description="Store prose cache co-located with blueprints (per-project).",
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

    if request.kb_id:
        # ---- Re-run against an existing KB: use kb_id as job_id ----
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == request.kb_id).first()
        if not kb:
            # The job folder may exist on disk (e.g. copied from another system)
            # but the DB record is missing. Auto-create it so the pipeline can run.
            job_dir = settings.JOBS_BASE_DIR / request.kb_id
            if not job_dir.is_dir():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Knowledge base '{request.kb_id}' not found (no DB record or job folder).",
                )
            kb = KnowledgeBase(
                id=request.kb_id,
                name=request.kb_name or request.kb_id,
                created_by=admin.username,
                datasource_username=request.datasource_username,
                datasource_folder=request.datasource_folder,
                source_path="pending",
                status="pending",
            )
            db.add(kb)
            db.commit()
            logger.info("Auto-created KB record for existing job folder %s", request.kb_id)
        else:
            kb.status = "pending"
            db.commit()
        job_id = request.kb_id
    else:
        # ---- Brand-new run: generate a fresh KB ----
        if not request.kb_name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Either kb_id (existing) or kb_name (new) must be provided.",
            )
        existing = db.query(KnowledgeBase).filter(KnowledgeBase.name == request.kb_name).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A knowledge base named '{request.kb_name}' already exists (id: {existing.id}).",
            )
        job_id = str(uuid4())
        kb = KnowledgeBase(
            id=job_id,
            name=request.kb_name,
            created_by=admin.username,
            datasource_username=request.datasource_username,
            datasource_folder=request.datasource_folder or (request.zip_path and Path(request.zip_path).stem),
            source_path="pending",
            status="pending",
        )
        db.add(kb)
        db.commit()

    ws = WorkspaceManager(job_id)
    for d in (ws.root, ws.input_dir, ws.output_dir, ws.cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    ws.set_status("pending", operation="vbt_pipeline")

    if request.source_path or request.zip_path or request.datasource_folder:
        source_path = _resolve_source_path(request, admin.username, ws.input_dir)
    elif request.kb_id and kb.source_path:
        source_path = Path(kb.source_path)
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No source path: provide source_path, zip_path, or datasource_folder.",
        )

    # Update source_path on the KB record now that it's resolved
    kb.source_path = str(source_path)
    db.commit()

    options = request.model_dump(
        exclude={"kb_id", "kb_name", "datasource_username", "datasource_folder", "source_path", "zip_path"}
    )
    options["source_path"] = str(source_path)
    ws.save_pipeline_options(options)

    from api.tasks.vbt_pipeline_task import run_vbt_pipeline
    task = run_vbt_pipeline.apply_async(args=[job_id, options])
    ws.set_status("pending", operation="vbt_pipeline", celery_task_id=task.id)

    logger.info(
        "VBT pipeline dispatched — job=%s kb=%r src=%s user=%s task=%s",
        job_id, request.kb_name or request.kb_id, source_path, admin.username, task.id,
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

    # Reuse the KB's job workspace — all VBT artifacts live under one ID.
    job_id = request.kb_id
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

    kb_opts = ws.get_pipeline_options() or {}
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


@router.post(
    "/vbt/lca-trace",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    summary="Run a DB-only LCA chain trace",
    description=(
        "Derive a variable's setters → call-graph LCA(s) → root→LCA chains, strictly from "
        "index.db (no source/blueprint reads). The KB must have completed VBT precompute and the "
        "standalone chains-precompute step. Runs as a Celery job; poll GET /vbt/{job_id}/status, "
        "then GET /v1/jobs/{job_id}/results → lca_trace.json."
    ),
)
def submit_vbt_lca_trace(
    request: VbtLcaTraceRequest,
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
    ws.set_status("pending", operation="vbt_lca_trace")

    options: Dict[str, Any] = {
        "kb_id": request.kb_id,
        "variable": request.variable,
        "language": request.language,
        "domain": request.domain,
        "prune": request.prune,
        "allow_missing_forced_lca": request.allow_missing_forced_lca,
        "allow_unattributed_setters": request.allow_unattributed_setters,
        "debug": request.debug,
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
    if request.trace_options:
        options["trace_options"] = request.trace_options

    from api.tasks.vbt_lca_trace_task import run_vbt_lca_trace
    task = run_vbt_lca_trace.apply_async(args=[job_id, options])
    ws.set_status("pending", operation="vbt_lca_trace", celery_task_id=task.id)

    logger.info(
        "VBT LCA-trace dispatched — job=%s kb=%s var=%s domain=%s prune=%s user=%s task=%s",
        job_id, request.kb_id, request.variable, request.domain, request.prune,
        current_user.username, task.id,
    )

    meta = ws.get_job_meta()
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        operation="vbt_lca_trace",
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        celery_task_id=task.id,
    )


@router.post(
    "/vbt/setter-order",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    summary="Run a DB-only setter-order trace",
    description=(
        "Derive a variable's setters → call-graph LCA(s) and emit one FLAT ordered setters array "
        "per LCA tree (static CFG/source order — not a runtime guarantee), strictly from index.db "
        "(no source/blueprint reads). The KB must have completed VBT precompute. Runs as a Celery "
        "job; poll GET /vbt/{job_id}/status, then GET /v1/jobs/{job_id}/results → setter_order.json."
    ),
)
def submit_vbt_setter_order(
    request: VbtSetterOrderRequest,
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
    ws.set_status("pending", operation="vbt_setter_order")

    options: Dict[str, Any] = {
        "kb_id": request.kb_id,
        "variable": request.variable,
        "language": request.language,
        "domain": request.domain,
        "allow_missing_forced_lca": request.allow_missing_forced_lca,
        "allow_unattributed_setters": request.allow_unattributed_setters,
        "debug": request.debug,
    }

    # VBT artifacts (blueprints, source) live under the KB job, not this fresh output job.
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
    if request.trace_options:
        options["trace_options"] = request.trace_options

    from api.tasks.vbt_lca_trace_task import run_vbt_setter_order
    task = run_vbt_setter_order.apply_async(args=[job_id, options])
    ws.set_status("pending", operation="vbt_setter_order", celery_task_id=task.id)

    logger.info(
        "VBT setter-order dispatched — job=%s kb=%s var=%s domain=%s user=%s task=%s",
        job_id, request.kb_id, request.variable, request.domain,
        current_user.username, task.id,
    )

    meta = ws.get_job_meta()
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        operation="vbt_setter_order",
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        celery_task_id=task.id,
    )


@router.post(
    "/vbt/all-chains-trace",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    summary="Discover a variable's LCA chains and run a full trace for each",
    description=(
        "Multi-chain coordinator: derive a variable's setters → call-graph LCA(s) → root→LCA "
        "chains (DB-only), then run a detailed VBT trace for EACH discovered chain in one process "
        "(shared caches), writing a single consolidated vbt_all_chains_trace.json. The KB must have "
        "completed VBT precompute and the standalone chains-precompute step. Runs as a Celery job; "
        "poll GET /vbt/{job_id}/status, then GET /v1/jobs/{job_id}/results → vbt_all_chains_trace.json."
    ),
)
def submit_vbt_all_chains_trace(
    request: VbtAllChainsTraceRequest,
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
    ws.set_status("pending", operation="vbt_all_chains_trace")

    options: Dict[str, Any] = {
        "kb_id": request.kb_id,
        "variable": request.variable,
        "language": request.language,
        "domain": request.domain,
        "prune": request.prune,
        "allow_missing_forced_lca": request.allow_missing_forced_lca,
        "allow_unattributed_setters": request.allow_unattributed_setters,
        "split_chains": request.split_chains,
        "debug": request.debug,
    }

    # VBT artifacts (blueprints, source) live under the KB job, not this fresh output job.
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
    if request.trace_options:
        options["trace_options"] = request.trace_options

    from api.tasks.vbt_lca_trace_task import run_vbt_all_chains_trace
    task = run_vbt_all_chains_trace.apply_async(args=[job_id, options])
    ws.set_status("pending", operation="vbt_all_chains_trace", celery_task_id=task.id)

    logger.info(
        "VBT all-chains-trace dispatched — job=%s kb=%s var=%s domain=%s prune=%s user=%s task=%s",
        job_id, request.kb_id, request.variable, request.domain, request.prune,
        current_user.username, task.id,
    )

    meta = ws.get_job_meta()
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        operation="vbt_all_chains_trace",
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        celery_task_id=task.id,
    )


# ---------------------------------------------------------------------------
# POST /vbt/trace-all-chains — discover every file chain, emit one JSON per chain
# ---------------------------------------------------------------------------

@router.post(
    "/vbt/trace-all-chains",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    summary="Discover a variable's file chains and write one JSON per chain",
    description=(
        "Per-chain coordinator: from just a kb_id + variable, derive the variable's setters → "
        "call-graph LCA(s) → root→LCA chains (DB-only), run a detailed VBT trace for EACH discovered "
        "chain in one process (shared caches), and write one fully self-contained JSON per chain "
        "(vbt_all_chains_trace/chain_NNNN.json) plus an index.json manifest — never a single "
        "consolidated file. The KB must have completed VBT precompute and the standalone "
        "chains-precompute step. Runs as a Celery job; poll GET /vbt/{job_id}/status, then "
        "GET /v1/jobs/{job_id}/results for the per-chain files."
    ),
)
def submit_vbt_trace_all_chains(
    request: VbtTraceAllChainsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == request.kb_id).first()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base '{request.kb_id}' not found.",
        )

    # Fresh output job (separate from the KB) so repeated runs don't clobber each other —
    # mirrors /vbt/all-chains-trace.  VBT artifacts (blueprints, source) live under the KB job.
    job_id = str(uuid4())
    ws = WorkspaceManager(job_id)
    for d in (ws.root, ws.output_dir):
        d.mkdir(parents=True, exist_ok=True)
    ws.set_status("pending", operation="vbt_trace_all_chains")

    options: Dict[str, Any] = {
        "kb_id": request.kb_id,
        "variable": request.variable,
        "language": request.language,
        "domain": request.domain,
        "prune": request.prune,
        "allow_missing_forced_lca": request.allow_missing_forced_lca,
        "allow_unattributed_setters": request.allow_unattributed_setters,
        "split_chains": True,  # this endpoint always emits one JSON per chain
        "debug": request.debug,
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
    if request.trace_options:
        options["trace_options"] = request.trace_options

    from api.tasks.vbt_trace_all_chains_task import run_vbt_trace_all_chains
    task = run_vbt_trace_all_chains.apply_async(args=[job_id, options])
    ws.set_status("pending", operation="vbt_trace_all_chains", celery_task_id=task.id)

    logger.info(
        "VBT trace-all-chains dispatched — job=%s kb=%s var=%s domain=%s prune=%s user=%s task=%s",
        job_id, request.kb_id, request.variable, request.domain, request.prune,
        current_user.username, task.id,
    )

    meta = ws.get_job_meta()
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        operation="vbt_trace_all_chains",
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        celery_task_id=task.id,
    )


# ---------------------------------------------------------------------------
# POST /vbt/lineage — full backward lineage from just a variable name
# ---------------------------------------------------------------------------

@router.post(
    "/vbt/lineage",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    summary="Trace a variable's complete backward lineage to classified terminal sources",
    description=(
        "Full-lineage coordinator: from just a kb_id + variable (language auto-detected), "
        "discover ALL of the variable's file chains (setters → call-graph LCA(s) → root→LCA "
        "chains, DB-only, extern C++↔ASM hops included), run a DEEP dependent-variable trace "
        "for each chain, and emit one lineage graph per chain whose terminal variables are "
        "classified as user_input / database / constant / external / working_storage / "
        "record_field / unresolved — each with evidence. Strict DB mode: no live-precompute "
        "and no JSON-file fallback; the KB's index.db must be complete, and the payload "
        "includes a db_audit block reporting any missing graph connections. Runs as a Celery "
        "job; poll GET /vbt/{job_id}/status, then GET /v1/jobs/{job_id}/results for "
        "vbt_lineage/lineage.json."
    ),
)
def submit_vbt_lineage(
    request: VbtLineageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == request.kb_id).first()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base '{request.kb_id}' not found.",
        )
    if request.language is not None and request.language not in ("asm", "cpp"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="language must be 'asm', 'cpp', or omitted for auto-detection.",
        )

    # Fresh output job (separate from the KB) so repeated runs don't clobber each other.
    job_id = str(uuid4())
    ws = WorkspaceManager(job_id)
    for d in (ws.root, ws.output_dir):
        d.mkdir(parents=True, exist_ok=True)
    ws.set_status("pending", operation="vbt_lineage")

    options: Dict[str, Any] = {
        "kb_id": request.kb_id,
        "variable": request.variable,
        "language": request.language,
        "domain": request.domain,
        "prune": request.prune,
        "allow_missing_forced_lca": request.allow_missing_forced_lca,
        "allow_unattributed_setters": request.allow_unattributed_setters,
        "split_chains": request.split_chains,
        "debug": request.debug,
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
    if request.trace_options:
        options["trace_options"] = request.trace_options

    from api.tasks.vbt_lineage_task import run_vbt_lineage
    task = run_vbt_lineage.apply_async(args=[job_id, options])
    ws.set_status("pending", operation="vbt_lineage", celery_task_id=task.id)

    logger.info(
        "VBT lineage dispatched — job=%s kb=%s var=%s lang=%s user=%s task=%s",
        job_id, request.kb_id, request.variable, request.language or "auto",
        current_user.username, task.id,
    )

    meta = ws.get_job_meta()
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        operation="vbt_lineage",
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        celery_task_id=task.id,
    )


# ---------------------------------------------------------------------------
# POST /vbt/trace-sync — synchronous trace (W11 daemon-lite).
#
# Same input shape as POST /vbt/trace, but runs the trace IN-PROCESS in the API
# worker's threadpool and returns the manifest/output path immediately. Skips the Celery dispatch
# + status-polling round-trip (~250-500 ms per trace on warm cache; matters on
# high-frequency query workloads, dominant on cache-hit queries).
#
# For long traces (heavy queries on 24k can take 10-60 s) the HTTP request
# blocks for that duration — clients should use long timeouts. The /vbt/trace
# async endpoint remains for fire-and-forget workloads.
# ---------------------------------------------------------------------------

@router.post(
    "/vbt/trace-sync",
    summary="Run a VBT trace synchronously",
    description=(
        "Synchronous variant of POST /vbt/trace. Skips Celery and returns the trace "
        "manifest plus output file path. Small JSON outputs are inlined automatically; "
        "set inline_result=true to force inline JSON, or false to disable it."
    ),
)
def submit_vbt_trace_sync(
    request: VbtTraceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
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
    ws.set_status("pending", operation="vbt_trace_sync")

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

    logger.info(
        "VBT trace-sync dispatched in-process — job=%s kb=%s var=%s user=%s",
        job_id, request.kb_id, request.variable, current_user.username,
    )

    # Run the trace directly in this API worker's threadpool slot. FastAPI
    # dispatches sync route handlers to a threadpool so we don't block the
    # event loop. The trace itself is CPU-bound Python, so the slot is the
    # bottleneck — keep API_WORKERS >= 2 for any concurrent trace traffic.
    from api.tasks.vbt_trace_task import run_trace
    try:
        manifest = run_trace(job_id, options, ws=ws)
    except Exception as exc:
        logger.exception("[vbt_trace_sync] job=%s failed", job_id)
        try:
            ws.set_status("failed", error=str(exc))
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trace failed: {exc}",
        ) from exc

    # Avoid reparsing large trace JSON just to echo it through the API. The file
    # path is already in the manifest; callers can opt in to inline JSON when the
    # payload is small enough for their latency/memory budget.
    output_files = manifest.get("outputs") or []
    result_blob: Optional[Dict[str, Any]] = None
    result_file: Optional[Dict[str, Any]] = None
    if output_files:
        first = Path(output_files[0])
        if first.is_file():
            size_bytes = first.stat().st_size
            inline_limit = max(0, int(request.inline_result_max_bytes or 0))
            should_inline = (
                bool(request.inline_result)
                if request.inline_result is not None
                else size_bytes <= inline_limit
            )
            result_file = {
                "path": str(first),
                "size_bytes": size_bytes,
                "inline_limit_bytes": inline_limit,
                "inlined": False,
            }
            if first.suffix != ".json":
                result_file["omitted_reason"] = "non_json_output"
            elif not should_inline:
                result_file["omitted_reason"] = (
                    "inline_result_disabled"
                    if request.inline_result is False
                    else "json_larger_than_inline_limit"
                )
            else:
                try:
                    import json as _json
                    result_blob = _json.loads(first.read_text())
                    result_file["inlined"] = True
                except Exception as exc:
                    result_file["omitted_reason"] = "json_read_failed"
                    logger.warning("[vbt_trace_sync] could not read emitted JSON %s: %s", first, exc)

    return {
        "job_id": job_id,
        "kb_id": request.kb_id,
        "variable": request.variable,
        "language": request.language,
        "result": result_blob,
        "result_file": result_file,
        "manifest": manifest,
    }


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
# POST /vbt/uiexport
# ---------------------------------------------------------------------------

@router.post(
    "/vbt/uiexport",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    summary="Run VBT UI export on a completed trace",
    description=(
        "Transform a completed VBT trace into UI-ready per-variable JSON files "
        "(structural facts + LLM prose). Runs as a Celery background job; "
        "poll GET /vbt/{job_id}/status for progress."
    ),
)
def submit_vbt_uiexport(
    request: VbtUiExportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == request.kb_id).first()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base '{request.kb_id}' not found.",
        )

    # Verify trace output exists for the specified variable
    trace_dir = settings.JOBS_BASE_DIR / request.kb_id / "output" / "vbt_trace"
    # Sanitize variable name for filesystem lookup
    safe_var = "".join(
        c if (c.isalnum() or c in "._-") else "_" for c in request.variable
    )[:120] or "var"
    var_trace_dir = trace_dir / safe_var
    var_trace_flat = trace_dir / f"{safe_var}.json"
    if not (var_trace_dir.is_dir() or var_trace_flat.is_file()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No VBT trace output found for variable '{request.variable}' "
                f"in KB '{request.kb_id}'. Run a VBT trace for this variable first."
            ),
        )

    # Reuse the KB's job workspace — trace_job_id = kb_id since trace
    # output is stored under the same KB directory.
    job_id = request.kb_id
    ws = WorkspaceManager(job_id)
    for d in (ws.root, ws.output_dir):
        d.mkdir(parents=True, exist_ok=True)
    ws.set_status("pending", operation="vbt_uiexport")

    options: Dict[str, Any] = {
        "trace_job_id": request.kb_id,
        "variable": request.variable,
        "no_llm": request.no_llm,
        "force": request.force,
        "max_paths": request.max_paths,
        "cache_in_job_dir": request.cache_in_job_dir,
    }
    if request.variables:
        options["variables"] = request.variables

    # Resolve blueprint_dir for prose cache location
    bp_out = settings.JOBS_BASE_DIR / request.kb_id / "output"
    from api.tasks._task_utils import find_blueprint_dir
    bp_dir = find_blueprint_dir(bp_out)
    if bp_dir:
        options["blueprint_dir"] = str(bp_dir)

    from api.tasks.vbt_uiexport_task import run_vbt_uiexport
    task = run_vbt_uiexport.apply_async(args=[job_id, options])
    ws.set_status("pending", operation="vbt_uiexport", celery_task_id=task.id)

    logger.info(
        "VBT uiexport dispatched — kb=%s var=%s user=%s task=%s",
        request.kb_id, request.variable, current_user.username, task.id,
    )

    meta = ws.get_job_meta()
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        operation="vbt_uiexport",
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        celery_task_id=task.id,
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
