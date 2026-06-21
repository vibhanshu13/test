"""3A — precompute the fn-graph build's cfg-derived per-stem facts (the cfg-load cost).

`CppFnGraph._build` calls `_ensure_file(stem)` for every reachable C++ stem (6832 at 22k), and each
call loads+parses that stem's full cfg via `_cfg_for` just to derive two small facts: the function
START lines (for nearest-preceding attribution) and the set of local function names. That mass cfg
load is the 382s cold-DB build (the "6 min before the first per-route log"). This precomputes the two
facts per stem so the trace LOADS them instead of loading+parsing the full cfg.

Byte-identical by construction: the stored `(starts, local)` are produced by the EXACT same derivation
`_ensure_file` runs (`_cfg_for` + `_fn_lines`, same sort, same `::`-tail split). A stem absent from the
map → `get_fn_facts` returns None → `_ensure_file` falls back to the live `_cfg_for` (so an incomplete
precompute can never change output, only forgo the speedup). Lookups are armed only when a job id is
set (the precompute build itself still loads cfgs — it must, to produce the facts).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from vbt.precompute import db_artifacts as DA

logger = logging.getLogger(__name__)

FN_FACTS_ARTIFACT = "fn_facts"        # single blob: {stem: [starts, local]}
FN_FACTS_META_ARTIFACT = "fn_facts_meta"

# job_id -> {stem: [ [ [line, name], ... ], [name, ...] ]} | None (no blob). Loaded once per job.
_LOADED: Dict[str, object] = {}
_META_LOADED: Dict[str, object] = {}
_JOB: Optional[str] = None


def set_fn_facts_job(job_id: Optional[str]) -> None:
    """Arm (or disarm) fn-facts lookups in CppFnGraph._ensure_file for this trace."""
    global _JOB
    _JOB = job_id


def build_and_store_fn_facts(job_id, blueprint_dir, asm_dir) -> int:
    """Build + store per-stem (starts, local) for every C++ source stem. Returns the stem count."""
    from vbt.engine import _cfg_for, _cfg_skip_reason, _fn_lines
    from vbt.precompute.cfg_db import install_cfg_db, clear_cfg_db
    from vbt.cpp_frontend import wrapper as _w
    src = Path(asm_dir)
    install_cfg_db(job_id)
    # Earlier precompute phases may have filled the in-process CFG memo before
    # DB hooks were installed. Clear it so disk-cache hits flow through _CFG_DB_PUT
    # and this pass actually populates vbt_cfg_blob.
    _w._MEMO.clear()
    facts: Dict[str, list] = {}
    skipped: Dict[str, str] = {}
    empty: List[str] = []
    try:
        for cpp in sorted(src.glob("*.cpp")):
            stem = cpp.stem
            try:
                fns = _cfg_for(stem, {}, src)            # same source the trace's _ensure_file reads
            except Exception:
                continue
            reason = _cfg_skip_reason(stem)
            if reason:
                skipped[stem] = reason
            starts: List[list] = []
            local: Set[str] = set()
            for f in fns:                                # EXACT _ensure_file derivation
                name = (f.get("function") or "").split("::")[-1]
                if not name:
                    continue
                local.add(name)
                lines = _fn_lines(f)
                if lines:
                    starts.append([min(lines), name])
            starts.sort()
            facts[stem] = [starts, sorted(local)]
            if not reason and not starts and not local:
                empty.append(stem)
    finally:
        clear_cfg_db()
    meta = {"skipped_cfg": skipped, "empty_cfg": empty}
    _LOADED[job_id] = facts
    _META_LOADED[job_id] = meta
    if DA.write_blob(job_id, FN_FACTS_ARTIFACT, DA.dumps_gz(facts)):
        DA.write_manifest(job_id, FN_FACTS_ARTIFACT, 1,
                          DA.source_manifest_hash(sorted(src.glob("*.cpp")), version=1))
    DA.write_blob(job_id, FN_FACTS_META_ARTIFACT, DA.dumps_gz(meta))
    return len(facts)


def _facts_map(job_id: str):
    m = _LOADED.get(job_id, 0)
    if m == 0:
        payload = DA.read_blob(job_id, FN_FACTS_ARTIFACT)
        m = DA.loads_gz(payload) if payload is not None else None
        _LOADED[job_id] = m
    return m


def _meta_map(job_id: str):
    m = _META_LOADED.get(job_id, 0)
    if m == 0:
        payload = DA.read_blob(job_id, FN_FACTS_META_ARTIFACT)
        m = DA.loads_gz(payload) if payload is not None else {}
        _META_LOADED[job_id] = m
    return m


def get_fn_facts(stem: str) -> Optional[Tuple[List[Tuple[int, str]], Set[str]]]:
    """``(starts, local)`` for ``stem`` from the precomputed map, or None (→ live `_cfg_for` fallback).

    ``starts`` is the sorted ``[(min_line, name)]`` list and ``local`` the set of local fn names —
    byte-identical to what `_ensure_file` derives from the cfg."""
    if not _JOB:
        return None
    m = _facts_map(_JOB)
    if not m:
        return None
    f = m.get(stem)
    if f is None:
        return None
    return [tuple(x) for x in f[0]], set(f[1])


def get_fn_fact_skip_reason(stem: str) -> Optional[str]:
    """Concise CFG failure reason recorded while building fn_facts, if any."""
    if not _JOB:
        return None
    m = _meta_map(_JOB)
    skipped = (m or {}).get("skipped_cfg") or {}
    reason = skipped.get(stem)
    return str(reason) if reason else None


def get_fn_fact_skip_reasons(job_id: str) -> Dict[str, str]:
    """All concise CFG failure reasons recorded for a job's fn_facts build."""
    m = _meta_map(job_id)
    return {str(k): str(v) for k, v in ((m or {}).get("skipped_cfg") or {}).items()}
