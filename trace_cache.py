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
import os
from typing import Any, Dict, List, Optional

from vbt.precompute import db_artifacts as DA

TRACE_CACHE_VERSION = 12  # 12: resolvedConstants literal/register token filter + ICM memory-form predicate (invalidates v11 rows carrying junk constants / mask-as-comparand conditions)
                          # 11: deterministic ASM guard order — equal-depth dominators tie-break by block id (invalidates v10 rows cached with hash-seed-dependent condition order)
                          # 10: GAP 6 — C++→ASM register-bridge data-source setters (`«SL70»`) under their guard + bare-aggregate descendant scan; emit_indirect_writes in the signature (invalidates v9 rows missing them)
                          # 9: GAP 7 — codeBlocks code/chunks comment-stripped (srcnc); AND GAP 8 — setter chain carries intra-file function hops (invalidates v8 rows built from comment-bearing source / file-collapsed chains)
                          # 8: GAP 9 — "[not set]" outcome entries (¬ of the setters' guards); emit_not_set in the signature (invalidates v7 rows missing them)
                          # 7: GAP 2 — C++ search-loop match conditions emitted as loop_break guards (invalidates v6 rows missing them)
                          # 6: GAP 5 — root output suppresses empty synthetic path-family ancestors (invalidates v5 rows carrying the noisy nodes)
                          # 5: TC-01 — mixed ASM/C++ route chains qualify known C++ hop functions (invalidates v4 rows with bare mixed-route C++ stems)
                          # 4: GAP 6 — setter convenience fields file/line/function/blockId are populated (invalidates v3 rows missing them)
                          # 3: prefix-guard fix (Issue 2) — file-only C++ hops keep real guards; invalidates v2 rows that may have dropped them
_PREFIX = "trace:"


def signature(variable, language, chain_prefix, blueprint_dir, asm_dir, graph_file, *,
              candidate_stems, candidate_functions, home_hint, disable_dependents,
              max_dep_var_depth, max_paths, max_call_depth, max_routes, max_route_len,
              max_offchain_files, asm_max_levels, emit_not_set=True,
              emit_indirect_writes=False, scratch_local_prefixes=()) -> str:
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
        max_offchain_files, asm_max_levels, bool(emit_not_set),
        bool(emit_indirect_writes),
        sorted(str(p).upper() for p in (scratch_local_prefixes or ())),
    ]
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]


def load(job_id: Optional[str], sig: str) -> Optional[Dict[str, Any]]:
    if not job_id:
        return None
    payload = DA.read_blob(job_id, _PREFIX + sig)
    return DA.loads_gz(payload) if payload is not None else None


def store(job_id: Optional[str], sig: str, out: Dict[str, Any]) -> None:
    if not job_id:
        return
    try:
        max_bytes = int(os.environ.get("VBT_TRACE_CACHE_MAX_BYTES", "536870912") or "0")
    except (TypeError, ValueError):
        max_bytes = 536870912
    try:
        payload = DA.dumps_gz(out)
        if max_bytes > 0 and len(payload) > max_bytes:
            logging.getLogger(__name__).info(
                "trace cache store skipped: gzipped payload %d bytes exceeds VBT_TRACE_CACHE_MAX_BYTES=%d",
                len(payload), max_bytes,
            )
            return
        DA.write_blob(job_id, _PREFIX + sig, payload)
    except Exception:
        pass


def clear(job_id: Optional[str]) -> int:
    """Delete all cached trace results for the job (called on re-precompute)."""
    return DA.clear_blobs_prefix(job_id, _PREFIX)
