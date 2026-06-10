"""Celery task: backward-only lineage against an existing KB output."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery.exceptions import SoftTimeLimitExceeded

PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.config import settings
from api.storage.workspace import WorkspaceManager
from api.tasks._task_utils import redirect_output_to_log, job_log_handler
from api.tasks.celery_app import celery_app
from api.tasks._output_compression import compress_outputs

logger = logging.getLogger(__name__)

import json
import re
import sqlite3
from typing import Tuple


# ---------------------------------------------------------------------------
# Lazy callsite map — avoids preloading 200–400 MB of call-site dicts
# ---------------------------------------------------------------------------

class _LazyCallsiteMap:
    """Dict-like wrapper that loads call-site dicts per (source, target) edge
    on first access via the index DB.

    A typical chain has 2–5 files → 2–8 edge lookups total — trivial DB cost
    compared to preloading the full callsite dict for all 50K+ edges.
    """

    __slots__ = ("_job_id", "_fetch_fn", "_cache")

    def __init__(self, job_id: str, fetch_fn) -> None:
        self._job_id = job_id
        self._fetch_fn = fetch_fn
        self._cache: Dict[Tuple[str, str], list] = {}

    def get(self, key, default=None):
        if key not in self._cache:
            src, tgt = key
            self._cache[key] = self._fetch_fn(self._job_id, src, tgt)
        result = self._cache[key]
        return result if result else default


# ---------------------------------------------------------------------------
# Dep-var enrichment helpers
# ---------------------------------------------------------------------------

_CPP_KEYWORDS: frozenset = frozenset({
    "if", "else", "while", "for", "switch", "return", "case", "default",
    "int", "char", "bool", "void", "const", "static", "auto", "unsigned",
    "long", "short", "new", "delete", "true", "false", "nullptr", "NULL",
    "sizeof", "memcmp", "strcmp", "strncmp", "memset", "memcpy",
    "Pa", "PaTables", "CasTransactionCode", "Dynamic4cscMatchIndicators",
    "Match4dbc3csc", "SwipeTrxnType", "SoftCardValueDetail", "PaMessage",
})


def _extract_dep_vars_from_condition_strings(conditions: List[str]) -> List[str]:
    """Extract plausible C++ variable names from condition expression strings."""
    names: List[str] = []
    seen: set = set()
    for cond in conditions:
        # Match dotted paths (e.g. plasticAuth.process.various.exitIndicator)
        for m in re.finditer(r'\b([a-zA-Z_]\w*(?:\.\w+)*)\b', cond):
            token = m.group(1)
            parts = token.split('.')
            for part in parts:
                # Take the last meaningful component
                if (len(part) >= 4
                        and part not in _CPP_KEYWORDS
                        and not part[0].isupper()  # skip PascalCase enum constants
                        and part not in seen):
                    seen.add(part)
                    names.append(part)
    return names


def _extract_func_from_node_id(node_id: str) -> str:
    """Extract enclosing function from a GVL node ID like 'kind:funcName::field'."""
    colon = node_id.find(":")
    if colon < 0:
        return ""
    rest = node_id[colon + 1:]
    if "::" not in rest:
        return ""
    func = rest.split("::")[0]
    return func if (func and "/" not in func) else ""


def _derive_file_stem(file_value: Any) -> str:
    """Return a normalized lower-case stem from a GVL ``file`` field."""
    raw = str(file_value or "").strip()
    if not raw or raw == "<global>":
        return ""
    tail = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].strip()
    if not tail:
        return ""
    return tail.rsplit(".", 1)[0].strip().lower()


def _classify_dep_var(
    var_name: str,
    scope_stems: set,
    db_path: str,
    job_id: str,
    anchor_function: Optional[str] = None,
    anchor_line: Optional[int] = None,
    modifier_file: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
    prefetched_rows: Optional[List] = None,
    prefetched_arg_bind_by_line: Optional[Dict[int, List]] = None,
    has_file_stem: bool = False,
) -> Dict[str, Any]:
    """Look up setter edges for var_name in the GVL for scope_stems.

    Reaching-definition check: setters in the SAME FUNCTION as anchor_function
    AND after anchor_line in the modifier file are excluded (non-reaching —
    they execute after the variable is already used).

    Returns {"classification": "SETTER"|"TERMINAL", "setter_locations": [...]}.

    PERF: When *prefetched_rows* and *prefetched_arg_bind_by_line* are provided,
    all filtering is done in Python against the pre-loaded data — no per-dep-var
    DB queries (28× faster for 50+ dep_vars).  Falls back to per-call SQL when
    prefetched data is not supplied.
    """
    try:
        var_lower = var_name.lower()

        if prefetched_rows is not None:
            # ── Fast path: filter prefetched rows in Python ──────────
            rows = []
            seen: set = set()
            for r in prefetched_rows:
                if var_lower not in (r["source_lower"] or "") \
                        and var_lower not in (r["target_lower"] or ""):
                    continue
                key = (r["source"], r["relation"], r["file"], r["line"], r["expression"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
                if len(rows) >= 50:
                    break
        else:
            # ── Legacy path: per-call DB query ───────────────────────
            _own_conn = conn is None
            if _own_conn:
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            scope_stems_sorted = sorted(
                str(s or "").strip().lower()
                for s in scope_stems
                if str(s or "").strip()
            )
            if not scope_stems_sorted:
                rows = []
            elif has_file_stem:
                in_clause = ",".join("?" for _ in scope_stems_sorted)
                cur.execute(f"""
                    SELECT DISTINCT source, relation, file, file_stem, line, expression, conditional_context
                    FROM gvl_edges
                    WHERE job_id = ?
                      AND relation IN ('assign', 'define')
                      AND file_stem IN ({in_clause})
                      AND (source_lower LIKE ? OR target_lower LIKE ?)
                    LIMIT 50
                """, [job_id] + scope_stems_sorted + [f"%{var_lower}%", f"%{var_lower}%"])
                rows = cur.fetchall()
            else:
                like_clauses = " OR ".join("file LIKE ?" for _ in scope_stems_sorted)
                like_args = [f"%{s}%" for s in scope_stems_sorted]
                cur.execute(f"""
                    SELECT DISTINCT source, relation, file, line, expression, conditional_context
                    FROM gvl_edges
                    WHERE job_id = ?
                      AND relation IN ('assign', 'define')
                      AND ({like_clauses})
                      AND (source_lower LIKE ? OR target_lower LIKE ?)
                    LIMIT 50
                """, [job_id] + like_args + [f"%{var_lower}%", f"%{var_lower}%"])
                rows = cur.fetchall()
            if not rows:
                if _own_conn:
                    conn.close()
                return {"classification": "TERMINAL", "setter_locations": []}

        if not rows:
            return {"classification": "TERMINAL", "setter_locations": []}

        locations = []
        for r in rows:
            setter_file_stem = str(
                (r["file_stem"] if "file_stem" in r.keys() else None)
                or _derive_file_stem(r["file"])
                or ""
            ).lower()
            setter_file_base = (r["file"] or "").split("/")[-1]
            setter_line = r["line"] or 0

            # Reaching-definition check: a setter in the modifier file at line >
            # anchor_line is non-reaching when:
            #   (a) It has no enclosing function (file-scope node representing the
            #       same write as a function-qualified node — always after anchor), OR
            #   (b) Its enclosing function is the DIRECT CALLEE invoked at anchor_line
            #       (confirmed via arg_bind edge at anchor_line in the modifier file).
            _same_modifier_file = bool(
                modifier_file
                and (
                    (setter_file_stem and setter_file_stem == modifier_file.lower())
                    or modifier_file.lower() in setter_file_base.lower()
                )
            )
            if _same_modifier_file and anchor_line and setter_line > anchor_line:
                setter_func = _extract_func_from_node_id(str(r["source"] or ""))
                if not setter_func or setter_func[0].isupper():
                    # (a) bare file-scope node, or class/namespace mistaken for function
                    # (e.g. PaTables::NotCollected) — non-reaching in same file
                    continue

                # (b) Check if setter_func is the callee invoked at anchor_line
                if prefetched_arg_bind_by_line is not None:
                    called_at_anchor = any(
                        (
                            (modifier_file.lower() == (stem or ""))
                            if stem else modifier_file.lower() in (file_name or "").lower()
                        )
                        and setter_func.lower() in (target_lower or "").lower()
                        for stem, file_name, target_lower in prefetched_arg_bind_by_line.get(anchor_line, [])
                    )
                elif conn is not None:
                    cur2 = conn.cursor()
                    if has_file_stem:
                        cur2.execute("""
                            SELECT COUNT(*) FROM gvl_edges
                            WHERE job_id=? AND relation='arg_bind'
                              AND file_stem=? AND line=?
                              AND target_lower LIKE ?
                        """, [job_id, modifier_file.lower(), anchor_line,
                              f"%{setter_func.lower()}%"])
                    else:
                        cur2.execute("""
                            SELECT COUNT(*) FROM gvl_edges
                            WHERE job_id=? AND relation='arg_bind'
                              AND file LIKE ? AND line=?
                              AND target_lower LIKE ?
                        """, [job_id, f"%{modifier_file}%", anchor_line,
                              f"%{setter_func.lower()}%"])
                    called_at_anchor = cur2.fetchone()[0] > 0
                    cur2.close()
                else:
                    called_at_anchor = False
                if called_at_anchor:
                    continue  # setter is inside the callee invoked at anchor_line

            _raw_cc = r["conditional_context"] if "conditional_context" in r.keys() else None
            _cond_ctx = None
            if _raw_cc:
                try:
                    _cond_ctx = json.loads(_raw_cc)
                except Exception:
                    pass
            locations.append({
                "file": setter_file_base,
                "file_stem": setter_file_stem or None,
                "line": setter_line,
                "expression": str(r["expression"] or "")[:120],
                "relation": r["relation"],
                "routine": _extract_func_from_node_id(str(r["source"] or "")) or None,
                "conditional_context": _cond_ctx,
            })

        if prefetched_rows is None and conn is not None and _own_conn:
            conn.close()
        if not locations:
            return {"classification": "TERMINAL", "setter_locations": []}
        return {"classification": "SETTER", "setter_locations": locations}

    except Exception:
        return {"classification": "UNKNOWN", "setter_locations": []}


def _setter_locations_to_chain_file_results(
    setter_locations: List[Dict[str, Any]],
    chain_stems: set,
) -> Dict[str, Any]:
    """Convert setter_locations from _classify_dep_var into chain_file_results format.

    Groups setter_locations by chain file stem and builds a minimal setter_sites
    list so condition-derived dep vars (match4dbc3csc, token4cscNotMatch, etc.)
    surface their setters in the same structure as runner-traced dep vars.
    """
    results: Dict[str, Any] = {}
    for loc in setter_locations:
        file_stem = str(loc.get("file_stem") or "").lower()
        if file_stem and file_stem in chain_stems:
            matched_stem = file_stem
        else:
            file_base = (loc.get("file") or "").lower()
            matched_stem = next(
                (s for s in chain_stems if s in file_base), None
            )
        if matched_stem is None:
            continue
        if matched_stem not in results:
            results[matched_stem] = {"dep_var": None, "file_type": "cpp", "setter_sites": [],
                                     "call_graph": {}, "warnings": []}
        results[matched_stem]["setter_sites"].append({
            "line": loc.get("line"),
            "routine": loc.get("routine"),
            "file": loc.get("file"),
            "expression": loc.get("expression", ""),
            "conditional_context": loc.get("conditional_context"),
        })
    return results


def _get_direct_callees(job_id: str, selected_chain: List[str]) -> set:
    """Return file stems directly called by chain files (1 hop out)."""
    try:
        from api.index_db.engine import get_engine
        from api.index_db.schema import call_graph_edges as _cg_edges
        from sqlalchemy import select

        engine = get_engine(job_id)
        chain_set = {s.lower() for s in selected_chain}
        # Filter by source stems in the chain so we only fetch relevant edges
        # instead of scanning the entire call_graph_edges table.
        chain_stems_orig = [s.strip() for s in selected_chain if s.strip()]
        callees = set()
        with engine.connect() as conn:
            result = conn.execute(
                select(_cg_edges.c.source, _cg_edges.c.target)
                .where(
                    (_cg_edges.c.job_id == job_id)
                    & (_cg_edges.c.source.in_(chain_stems_orig))
                )
            )
            for row in result:
                src = str(row.source or "").lower()
                tgt = str(row.target or "").lower()
                if src in chain_set and tgt not in chain_set and tgt.isidentifier():
                    callees.add(tgt)
        return callees
    except Exception:
        return set()


def _enrich_dep_var_output(
    output_dir: Path,
    job_id: str,
    kb_id: str,
    selected_chain: List[str],
) -> None:
    """Post-process dep_var output files:
    1. Extract dep_vars from call_chain_conditions and create new files
    2. Classify every dep_var as SETTER or TERMINAL via GVL lookup
    3. Include direct callees (e.g. dw710100) in the setter scope
    """
    root_var_path = output_dir / "root_var.json"
    enrichment_ctx_path = output_dir / "dep_var_enrichment_context.json"
    dep_vars_dir = output_dir / "dep_vars"
    if (not enrichment_ctx_path.exists() and not root_var_path.exists()) or not dep_vars_dir.exists():
        return

    modifier_stem = selected_chain[-1] if selected_chain else ""
    modifier_data: Dict[str, Any] = {}
    if enrichment_ctx_path.exists():
        try:
            with open(enrichment_ctx_path, encoding="utf-8") as f:
                enrichment_ctx = json.load(f) or {}
            modifier_stem = str(enrichment_ctx.get("modifier_file") or modifier_stem)
            modifier_data = enrichment_ctx.get("modifier") or {}
        except Exception as exc:
            logger.debug("Could not read dep-var enrichment context: %s", exc)

    if not modifier_data and root_var_path.exists():
        with open(root_var_path, encoding="utf-8") as f:
            root_var = json.load(f)
        modifier_data = root_var.get("files", {}).get(modifier_stem, {}).get("modifier", {})

    # DB path for GVL lookup
    db_path = str(Path(settings.JOBS_BASE_DIR) / kb_id / "index.db")

    # Ensure additive index.db migrations have run before raw sqlite queries.
    try:
        from api.index_db.engine import get_engine as _ensure_index_engine
        _ensure_index_engine(kb_id)
    except Exception:
        pass

    # Open a single shared connection for all _classify_dep_var calls so the
    # SQLite page cache stays warm across dep_vars (avoids 20-50 cold opens).
    _shared_conn = sqlite3.connect(db_path)
    _shared_conn.row_factory = sqlite3.Row
    _shared_conn.execute("PRAGMA journal_mode=WAL")
    _shared_conn.execute("PRAGMA cache_size=-65536")      # 64 MB page cache
    _shared_conn.execute("PRAGMA temp_store=MEMORY")
    _shared_conn.execute("PRAGMA mmap_size=268435456")    # 256 MB mmap window
    _shared_conn.execute("PRAGMA busy_timeout=5000")
    _gvl_columns = {str(r[1]) for r in _shared_conn.execute("PRAGMA table_info(gvl_edges)").fetchall()}
    _has_file_stem = "file_stem" in _gvl_columns

    try:
        # Build scope: chain files + direct callees (1 hop)
        direct_callees = _get_direct_callees(kb_id, selected_chain)
        all_scope_stems = sorted({
            str(s or "").strip().lower()
            for s in (set(selected_chain) | direct_callees)
            if str(s or "").strip()
        })

        # ── PERF: Prefetch assign/define and arg_bind rows in scope ──────
        # Two queries replace N×71 LIKE scans (28× faster for 50+ dep_vars).
        #
        # Safety guard: if the scope matches too many rows (>100K), the
        # prefetch would consume excessive RAM at scale (18 GB index.db).
        # In that case we skip the prefetch and fall back to per-dep-var
        # SQL queries via the shared connection — slower but bounded.
        try:
            _PREFETCH_ROW_CAP = max(
                0,
                int(os.environ.get("ASM_DEPVAR_ENRICH_PREFETCH_ROW_CAP", "100000")),
            )
        except Exception:
            _PREFETCH_ROW_CAP = 100_000

        # Probe row count before fetching
        if _has_file_stem and all_scope_stems:
            _scope_placeholders = ",".join("?" for _ in all_scope_stems)
            _cnt = _shared_conn.execute(f"""
                SELECT COUNT(*) FROM gvl_edges
                WHERE job_id = ?
                  AND relation IN ('assign', 'define')
                  AND file_stem IN ({_scope_placeholders})
            """, [kb_id] + all_scope_stems).fetchone()[0]
        else:
            _like_clauses = " OR ".join("file LIKE ?" for _ in all_scope_stems)
            _like_args = [f"%{s}%" for s in all_scope_stems]
            _cnt = _shared_conn.execute(f"""
                SELECT COUNT(*) FROM gvl_edges
                WHERE job_id = ?
                  AND relation IN ('assign', 'define')
                  AND ({_like_clauses})
            """, [kb_id] + _like_args).fetchone()[0]

        _prefetched_rows: Optional[List] = None
        _arg_bind_by_line: Optional[Dict[int, List]] = None

        if _PREFETCH_ROW_CAP > 0 and _cnt <= _PREFETCH_ROW_CAP:
            _cur = _shared_conn.cursor()
            if _has_file_stem and all_scope_stems:
                _cur.execute(f"""
                    SELECT source, target, relation, file, file_stem, line, expression,
                           conditional_context, source_lower, target_lower
                    FROM gvl_edges
                    WHERE job_id = ?
                      AND relation IN ('assign', 'define')
                      AND file_stem IN ({_scope_placeholders})
                """, [kb_id] + all_scope_stems)
            else:
                _cur.execute(f"""
                    SELECT source, target, relation, file, line, expression,
                           conditional_context, source_lower, target_lower
                    FROM gvl_edges
                    WHERE job_id = ?
                      AND relation IN ('assign', 'define')
                      AND ({_like_clauses})
                """, [kb_id] + _like_args)
            _prefetched_rows = _cur.fetchall()
            _cur.close()

            _cur2 = _shared_conn.cursor()
            if _has_file_stem and all_scope_stems:
                _cur2.execute(f"""
                    SELECT file_stem, file, line, target_lower
                    FROM gvl_edges
                    WHERE job_id = ?
                      AND relation = 'arg_bind'
                      AND file_stem IN ({_scope_placeholders})
                """, [kb_id] + all_scope_stems)
            else:
                _cur2.execute(f"""
                    SELECT file, line, target_lower
                    FROM gvl_edges
                    WHERE job_id = ?
                      AND relation = 'arg_bind'
                      AND ({_like_clauses})
                """, [kb_id] + _like_args)
            _arg_bind_by_line = {}
            for _ab in _cur2.fetchall():
                _ab_line = _ab["line"] or 0
                if _ab_line not in _arg_bind_by_line:
                    _arg_bind_by_line[_ab_line] = []
                _arg_bind_by_line[_ab_line].append((
                    _ab["file_stem"] if "file_stem" in _ab.keys() else None,
                    _ab["file"] if "file" in _ab.keys() else None,
                    _ab["target_lower"],
                ))
            _cur2.close()

            logger.debug(
                "Prefetched %d assign/define rows + %d arg_bind lines for %d scope stems",
                len(_prefetched_rows), len(_arg_bind_by_line), len(all_scope_stems),
            )
        else:
            logger.info(
                "Prefetch skipped: %d rows exceeds cap %d — falling back to per-var queries "
                "(%d scope stems)",
                _cnt, _PREFETCH_ROW_CAP, len(all_scope_stems),
            )

        # Extract new dep_vars from call_chain_conditions
        call_chain = modifier_data.get("call_chain_conditions", [])

        # Build per-dep_var anchor info for reaching-definition checks.
        # anchor_info[dv_name] = (anchor_function, anchor_line)
        anchor_info: Dict[str, tuple] = {}
        chain_cond_vars: List[str] = []

        for hop in call_chain:
            caller = hop.get("caller", "")
            hop_line = hop.get("line") or 0
            for dv in _extract_dep_vars_from_condition_strings(hop.get("conditions", [])):
                chain_cond_vars.append(dv)
                if dv not in anchor_info:
                    anchor_info[dv] = (caller, hop_line)

        # Also check setter site conditions (anchor = the modifier function / setter line)
        for site in modifier_data.get("setter_sites", []):
            site_func = site.get("routine", "")
            site_line = site.get("line") or 0
            for dv in _extract_dep_vars_from_condition_strings(site.get("conditional_context") or []):
                chain_cond_vars.append(dv)
                if dv not in anchor_info:
                    anchor_info[dv] = (site_func, site_line)

        # Existing dep_var files
        existing_safe_names = {f.stem.upper() for f in dep_vars_dir.glob("*.json")}

        # Track dep_vars classified in pass 1 so pass 2 can skip them
        _already_classified: set = set()
        _chain_stems_lower = set(s.lower() for s in selected_chain)
        _sorted_scope = sorted(all_scope_stems)

        # Create new dep_var files for call_chain vars not already tracked
        for dv_name in chain_cond_vars:
            safe = re.sub(r"[^A-Za-z0-9@$_#\-]", "_", dv_name).upper()
            if safe in existing_safe_names:
                continue
            existing_safe_names.add(safe)
            _af, _al = anchor_info.get(dv_name, (None, None))
            classification = _classify_dep_var(
                dv_name, all_scope_stems, db_path, kb_id,
                anchor_function=_af, anchor_line=_al, modifier_file=modifier_stem,
                conn=_shared_conn,
                prefetched_rows=_prefetched_rows,
                prefetched_arg_bind_by_line=_arg_bind_by_line,
                has_file_stem=_has_file_stem,
            )
            entry = {
                "dep_var": dv_name,
                "depth": 0,
                "truncated": False,
                "source": "call_chain_conditions",
                "classification": classification["classification"],
                "setter_locations": classification["setter_locations"],
                "scope_files_checked": _sorted_scope,
                "chain_file_results": _setter_locations_to_chain_file_results(
                    classification["setter_locations"], _chain_stems_lower,
                ),
                "downstream_file_results": {},
            }
            fname = re.sub(r"[^A-Za-z0-9@$_#\-]", "_", dv_name) + ".json"
            (dep_vars_dir / fname).write_text(json.dumps(entry, indent=2), encoding="utf-8")
            _already_classified.add(fname)
            logger.debug("Created dep_var file: %s (classification=%s)", fname, entry["classification"])

        # Enrich all existing dep_var files with classification
        for dep_var_file in dep_vars_dir.glob("*.json"):
            # Skip dep_vars that were just created and classified in pass 1
            if dep_var_file.name in _already_classified:
                continue
            try:
                with open(dep_var_file, encoding="utf-8") as f:
                    data = json.load(f)
                dv_name = data.get("dep_var", dep_var_file.stem)
                _af, _al = anchor_info.get(dv_name, (None, None))
                classification = _classify_dep_var(
                    dv_name, all_scope_stems, db_path, kb_id,
                    anchor_function=_af, anchor_line=_al, modifier_file=modifier_stem,
                    conn=_shared_conn,
                    prefetched_rows=_prefetched_rows,
                    prefetched_arg_bind_by_line=_arg_bind_by_line,
                    has_file_stem=_has_file_stem,
                )
                data["classification"] = classification["classification"]
                data["setter_locations"] = classification["setter_locations"]
                data["scope_files_checked"] = _sorted_scope
                if not data.get("chain_file_results"):
                    data["chain_file_results"] = _setter_locations_to_chain_file_results(
                        classification["setter_locations"], _chain_stems_lower,
                    )
                dep_var_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.debug("Could not enrich dep_var %s: %s", dep_var_file.name, exc)
    finally:
        _shared_conn.close()


@celery_app.task(bind=True, name="asm.kb_backward_lineage")
def run_kb_backward_lineage(self, job_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run backward-only cross-file lineage using a KB's existing blueprint output.

    Inputs are resolved from jobs/{kb_id}/output and output is written to:
      jobs/{job_id}/output/backward_lineage/{variable}_backward_lineage.json
    """
    ws = WorkspaceManager(job_id)
    ws.set_status("running")

    with redirect_output_to_log(ws.log_file):
        with job_log_handler(ws.log_file, job_id=job_id):
            try:
                return _run(ws, options)
            except SoftTimeLimitExceeded:
                msg = "Task exceeded soft time limit and was aborted."
                logger.error("Job %s: %s", job_id, msg)
                ws.set_status("failed", error=msg)
                return {"job_id": job_id, "status": "failed", "error": msg}
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                logger.exception("Job %s failed: %s", job_id, msg)
                ws.set_status("failed", error=msg)
                return {"job_id": job_id, "status": "failed", "error": msg}


def _resolve_blueprint_dir(kb_output: Path) -> Path:
    from api.tasks._task_utils import find_blueprint_dir
    d = find_blueprint_dir(kb_output)
    if d is None:
        raise FileNotFoundError(f"No .asm.json blueprint files found under {kb_output}")
    return d


def _resolve_asm_dir(blueprint_dir: Path, options: Dict[str, Any]) -> Optional[Path]:
    if options.get("asm_dir"):
        asm_dir = Path(str(options["asm_dir"]))
        if asm_dir.exists():
            return asm_dir
    # Fall back to the source_path recorded in the KB database record — this is
    # the resolved absolute path supplied when the pipeline was originally run.
    kb_id = options.get("kb_id")
    if kb_id:
        try:
            from api.db import SessionLocal
            from api.models.knowledge_base import KnowledgeBase
            db = SessionLocal()
            try:
                kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
                if kb and kb.source_path:
                    p = Path(kb.source_path)
                    if p.exists():
                        return p
            finally:
                db.close()
        except Exception:
            pass
    return None


def _run(ws: WorkspaceManager, options: Dict[str, Any]) -> Dict[str, Any]:
    from backward_traversal.runner.backward_only_runner import run_backward_only

    job_id = ws.job_id
    _task_start = time.monotonic()

    kb_id = str(options["kb_id"])
    variable = str(options["variable_name"])
    chain_files_raw = options.get("chain_files", []) or []
    if isinstance(chain_files_raw, str):
        selected_chain = [x.strip() for x in chain_files_raw.split(",") if x.strip()]
    else:
        selected_chain = [str(x).strip() for x in chain_files_raw if str(x).strip()]

    if not selected_chain:
        raise ValueError("chain_files must include at least one file stem (modifier).")

    # Phase 1/2 — Resolve inputs
    logger.info(
        "[Phase 1/2 START] Resolve inputs — job=%s kb=%s variable=%s chain_len=%d",
        job_id, kb_id, variable, len(selected_chain),
    )
    _t = time.monotonic()
    kb_output = settings.JOBS_BASE_DIR / kb_id / "output"
    if not kb_output.exists():
        raise FileNotFoundError(f"KB output directory not found: {kb_output}")

    blueprint_dir = _resolve_blueprint_dir(kb_output)
    from api.tasks._task_utils import resolve_graph_file
    graph_file = resolve_graph_file(kb_output, kb_id)
    asm_dir = _resolve_asm_dir(blueprint_dir, options)

    output_dir = ws.output_dir / "backward_lineage"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = str(options.get("output_name") or f"{variable}_backward_lineage.json")
    if not output_name.endswith(".json"):
        output_name += ".json"
    output_path = output_dir / output_name
    logger.info(
        "[Phase 1/2 DONE] Inputs resolved — blueprint_dir=%s elapsed=%.1fs",
        blueprint_dir, time.monotonic() - _t,
    )

    # Phase 2/2 — Backward traversal
    logger.info(
        "[Phase 2/2 START] Backward traversal — job=%s variable=%s chain=%s",
        job_id, variable, selected_chain,
    )
    _t = time.monotonic()

    # PERF: build call-graph indexes from DB in one streaming pass instead of
    # loading the monolithic graph_payload dict (400–1200 MB on 22K-file KBs).
    # The _LazyCallsiteMap does per-edge DB lookups on demand (a typical chain
    # has 2–5 files → 2–8 lookups total) rather than preloading 200–400 MB of
    # call-site dicts for all 50K+ edges.
    _prebuilt_kwargs: Dict[str, Any] = {}
    try:
        from api.index_db.readers import (
            build_call_graph_indexes,
            get_call_graph_callsites_for_edge,
        )
        _indexes = build_call_graph_indexes(kb_id, fallback_path=graph_file)
        if _indexes is not None:
            _prebuilt_kwargs["prebuilt_file_type_map"] = _indexes["file_type_map"]
            _prebuilt_kwargs["prebuilt_edge_type_map"] = _indexes["edge_type_map"]
            _prebuilt_kwargs["prebuilt_edge_lines_map"] = _indexes["edge_lines_map"]
            _prebuilt_kwargs["prebuilt_callsite_map"] = _LazyCallsiteMap(
                kb_id, get_call_graph_callsites_for_edge,
            )
            _prebuilt_kwargs["graph_payload"] = {}  # empty sentinel — never read
            logger.info(
                "[Phase 2/2] Using streaming DB indexes (file_type=%d, edge_type=%d)",
                len(_indexes["file_type_map"]), len(_indexes["edge_type_map"]),
            )
    except Exception as _idx_exc:
        logger.debug("[Phase 2/2] Streaming indexes unavailable, using legacy path: %s", _idx_exc)

    try:
        rc = run_backward_only(
            variable=variable,
            selected_chain=selected_chain,
            blueprint_dir=blueprint_dir,
            asm_dir=asm_dir,
            graph_file=graph_file,
            output_path=output_path,
            max_depth=int(options.get("max_depth", 8)),
            max_subroutine_depth=int(options.get("max_subroutine_depth", 2)),
            max_subroutine_nodes=int(options.get("max_subroutine_nodes", 400)),
            max_trace_nodes=int(options.get("max_trace_nodes", 5000)),
            max_dep_vars=int(options.get("max_dep_vars", 20)),
            max_dep_var_depth=int(options.get("max_dep_var_depth", 2)),
            max_downstream_depth=int(options.get("max_downstream_depth", 2)),
            max_downstream_files=int(options.get("max_downstream_files", 30)),
            extend_modifier_sites=bool(options.get("extend_modifier_sites", False)),
            target_setters=options.get("target_setters"),
            **_prebuilt_kwargs,
        )
    finally:
        # Release the cached read-only reach-facts connections opened during
        # the traversal (one per index.db touched).
        try:
            from backward_traversal.utils.reach_facts import close_reach_facts_reader
            close_reach_facts_reader()
        except Exception:
            pass
        # Release seed-lookup connections + module-level GVL projections so a
        # long-lived Celery worker doesn't pin them across jobs.
        try:
            from backward_traversal.utils.cpp_lineage_utils import close_db_seed_state
            close_db_seed_state()
        except Exception:
            pass
        try:
            from backward_traversal.runner.backward_only_runner import (
                clear_gvl_projection_cache,
            )
            clear_gvl_projection_cache()
        except Exception:
            pass
    if rc != 0:
        raise RuntimeError(f"Backward lineage runner returned non-zero exit code: {rc}")

    logger.info(
        "[Phase 2/2 DONE] Backward traversal — rc=%d elapsed=%.1fs total_elapsed=%.1fs",
        rc, time.monotonic() - _t, time.monotonic() - _task_start,
    )

    # Phase 3/3 — Enrich dep_var output with classification + scope expansion
    _t = time.monotonic()
    try:
        # The partitioned output dir is the variable subdirectory (stem of output_path).
        _var_output_dir = output_path.with_suffix("")
        _enrich_dep_var_output(
            output_dir=_var_output_dir,
            job_id=job_id,
            kb_id=kb_id,
            selected_chain=selected_chain,
        )
        logger.info(
            "[Phase 3/3 DONE] Dep-var enrichment — elapsed=%.1fs", time.monotonic() - _t
        )
    except Exception as _enrich_exc:
        logger.warning("[Phase 3/3 WARN] Dep-var enrichment failed (non-fatal): %s", _enrich_exc)

    # Phase 3b — Transform ASM output to unified target schema
    try:
        _t = time.monotonic()
        from backward_traversal.enrichment.asm_schema_transformer import transform_asm_output_dir
        transform_asm_output_dir(
            output_dir=_var_output_dir,
            root_var_name=variable,
        )
        logger.info(
            "[Phase 3b] ASM schema transform — elapsed=%.1fs", time.monotonic() - _t
        )
    except Exception as _xform_exc:
        logger.warning("[Phase 3b WARN] ASM schema transform failed (non-fatal): %s", _xform_exc)

    # Phase 4/4 — Compress output files (opt-in: only when compress_output=True)
    # The flag is set by the scoped backward-lineage endpoint; the plain
    # backward-lineage endpoint leaves it absent so it defaults to False here.
    n_compressed = 0
    if options.get("compress_output", False):
        _t = time.monotonic()
        n_compressed = compress_outputs(output_dir)  # type: ignore[assignment]
        logger.info(
            "[Phase 3/3 DONE] Compression — %d file(s) compressed, elapsed=%.1fs",
            n_compressed, time.monotonic() - _t,
        )
    else:
        logger.debug("[Phase 3/3 SKIP] compress_output not set — skipping compression")

    # Collect only backward_lineage output files (not the entire KB workspace,
    # which may contain thousands of pipeline artefacts from a previous full run).
    # result_files is built after compression so it reflects the final .msgpack.gz
    # names where applicable, and the original .json names for small files.
    result_files = []
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(ws.output_dir))
            result_files.append({"name": p.name, "path": rel, "size_bytes": p.stat().st_size})
    ws.set_status("success", result_files=result_files)

    return {
        "job_id": ws.job_id,
        "status": "success",
        "kb_id": kb_id,
        "variable": variable,
        "selected_chain": selected_chain,
        "output_file": str(output_path.relative_to(ws.output_dir)),
        "output_files": len(result_files),
        "files_compressed": n_compressed,
    }
