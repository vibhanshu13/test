"""Deterministic whole-trace result cache (DB_PRECOMPUTE_PLAN.md §16 #1).

A trace's output is a pure function of its inputs (variable, language, chain, dirs, every cap,
hints). We cache the finished output keyed by a signature of ALL output-affecting inputs; a
later trace with the IDENTICAL inputs returns the stored output instantly — skipping the whole
computation. Byte-identical because the cached value IS a prior run's output and the trace is
deterministic.

Freshness: the result is derived from the precompute artifacts, so ``precompute_db`` CLEARS this
cache on every (re)build — a rebuilt corpus can never serve a stale cached trace. ``job_id`` is
the DB key only (NOT part of the signature — output is identical with or without it).
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from vbt.precompute import db_artifacts as DA

TRACE_CACHE_VERSION = 3   # 3: prefix-guard fix (Issue 2) — file-only C++ hops keep real guards; invalidates v2 rows that may have dropped them
_PREFIX = "trace:"
_LOG = logging.getLogger("vbt.trace_cache")


def signature(variable, language, chain_prefix, blueprint_dir, asm_dir, graph_file, *,
              candidate_stems, candidate_functions, home_hint, disable_dependents,
              max_dep_var_depth, max_paths, max_call_depth, max_routes, max_route_len,
              max_offchain_files, asm_max_levels) -> str:
    """SHA-256 over every input that affects the trace output (paths included, so a different
    blueprint/asm dir form never collides)."""
    payload: List[Any] = [
        TRACE_CACHE_VERSION, language, variable,
        [(h.stem, h.file_type, h.function) for h in chain_prefix],
        str(blueprint_dir), str(asm_dir), str(graph_file),
        sorted(candidate_stems) if candidate_stems else None,
        sorted(candidate_functions) if candidate_functions else None,
        home_hint, bool(disable_dependents),
        max_dep_var_depth, max_paths, max_call_depth, max_routes, max_route_len,
        max_offchain_files, asm_max_levels,
    ]
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]


def load(job_id: Optional[str], sig: str) -> Optional[Dict[str, Any]]:
    if not job_id:
        return None
    payload = DA.read_blob(job_id, _PREFIX + sig)
    return DA.loads_gz(payload) if payload is not None else None


def store(job_id: Optional[str], sig: str, out: Dict[str, Any]) -> bool:
    if not job_id:
        return False
    try:
        payload = DA.dumps_gz(out)
        ok = DA.write_blob(job_id, _PREFIX + sig, payload)
        if not ok:
            _LOG.warning(
                "trace-cache store failed job=%s sig=%s compressed_bytes=%d",
                job_id, sig[:12], len(payload),
            )
        return bool(ok)
    except Exception as exc:
        _LOG.warning("trace-cache store failed job=%s sig=%s: %s", job_id, sig[:12], exc)
        return False


def clear(job_id: Optional[str]) -> int:
    """Delete all cached trace results for the job (called on re-precompute)."""
    return DA.clear_blobs_prefix(job_id, _PREFIX)
