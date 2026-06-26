"""ASM setter detection — thin, READ-ONLY reuse of the battle-ready engine.

Verdict (SPEC §9): USE-AS-IS. We do not copy or modify the logic; we import the
proven glossary + ``_find_scoped_setter_sites`` (which handles SETTER_INST dest
resolution, ``classify_setter`` edge cases, RMW prior-value, EX/EXRL, implicit
GETCC/DETAC/DBSPA writes, FLIPC aliases) and adapt its output to our ``SetterSite``.

No kb_id. Works against on-disk blueprints + raw ``.asm``/``.mac`` source only.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

from vbt.interfaces import AliasSet, SetterSite

# Read-only reuse of existing, validated implementations.
from backward_traversal.utils.blueprint_utils import (
    resolve_asm_blueprint,
    load_json,
    collect_constant_symbols,
)
from backward_traversal.utils.token_utils import normalize_token
from backward_traversal.runner.backward_only_runner import _find_scoped_setter_sites
from vbt.resolve.register_indirect import resolve_indirect_dest_writes

# Register-indirect DEST sites resolved per (stem, blueprint_dir), once. The resolution
# is a bounded backward register walk; caching keeps repeated per-variable lookups cheap.
_INDIRECT_DEST_CACHE: dict = {}

# Result cache for find_asm_setters_in_file — the ASM mirror of the cpp _SETTER_CACHE (f93bb48).
# The SAME (variable, stem) recurs across many dep vars / routes in a 10k-ASM trace, and each miss
# does ~6 full-file block scans + a constant-symbol source pass. The result is a pure function of
# (stem, dirs, variable) + the in-process-immutable corpus, so the memo is byte-identical. Keyed by
# PATH (not stat) like cpp _SRC_TREE_CACHE — precompute owns freshness; avoids an FS stat per call.
_SETTER_CACHE: dict = {}
_SETTER_CACHE_MAX = 8192

# Phase 1: the job whose precomputed per-file ASM setter map should be consulted (a LOOKUP instead
# of the block scan). Armed by the engine at trace start for a load-only trace; None ⇒ always scan
# (CLI / precompute build / no precompute). The lookup is additionally gated on is_load_only() so the
# precompute build itself (load-only OFF) still scans to produce the map.
_SETTER_MAP_JOB: Optional[str] = None


def set_asm_setter_map_job(job_id: Optional[str]) -> None:
    """Engine hook: route find_asm_setters_in_file through the precomputed setter map for this job."""
    global _SETTER_MAP_JOB
    _SETTER_MAP_JOB = job_id


def _setter_cache_put(key, val):
    if len(_SETTER_CACHE) >= _SETTER_CACHE_MAX:    # FIFO, recompute on miss (byte-identical)
        _SETTER_CACHE.pop(next(iter(_SETTER_CACHE)))
    _SETTER_CACHE[key] = val
    return val


def find_asm_indirect_dest_sites(
    stem: str,
    blueprint_dir: Path,
    asm_dir: Optional[Path] = None,
    route_engine=None,
    bp: Optional[dict] = None,
) -> List[SetterSite]:
    """Every register-indirect-DESTINATION setter site in one ASM/mac file, keyed to the
    resolved field it writes (``variable`` = that field).

    The mirror of the register-indirect SOURCE resolution, at the setter-DETECTION layer:
    a write whose destination is ``disp(Rn)`` (``MVC 0(R1),SRC`` / ``ST R,d(Rn)`` /
    ``OI d(Rn),mask`` …) does NOT name the field, so the reused name-matching finder is
    blind to it. We resolve each such dest to its named field and emit a SetterSite.
    Sound: a dest that doesn't resolve to a named field yields no site. Cached per file."""
    key = (str(stem), str(blueprint_dir))
    cached = _INDIRECT_DEST_CACHE.get(key)
    if cached is not None:
        return cached
    out: List[SetterSite] = []
    if bp is None:
        bp_path = resolve_asm_blueprint(stem, blueprint_dir)
        if not bp_path:
            _INDIRECT_DEST_CACHE[key] = out
            return out
        try:
            bp = load_json(bp_path)
        except Exception:
            _INDIRECT_DEST_CACHE[key] = out
            return out
    else:
        bp_path = resolve_asm_blueprint(stem, blueprint_dir)
    for field, line, bid, inst, src in resolve_indirect_dest_writes(
            bp, str(bp_path), stem, asm_dir=asm_dir, route_engine=route_engine):
        out.append(SetterSite(
            variable=field, file_stem=stem, language="asm", line=line,
            instruction=inst, block_id=bid, value=src, role="indirect_dest"))
    if len(_INDIRECT_DEST_CACHE) >= 8192:    # 22k OOM guard; FIFO, re-resolve on miss (byte-identical)
        _INDIRECT_DEST_CACHE.pop(next(iter(_INDIRECT_DEST_CACHE)))
    _INDIRECT_DEST_CACHE[key] = out
    return out


def find_asm_setters_in_file(
    variable: str,
    stem: str,
    blueprint_dir: Path,
    asm_dir: Optional[Path] = None,
) -> List[SetterSite]:
    """Return every setter site of ``variable`` in one ASM/mac file (full-file scope).

    Two sources: (1) the reused, name-matching ``_find_scoped_setter_sites`` (USE-AS-IS);
    (2) register-indirect DESTINATION writes whose resolved field == ``variable`` — the
    mirror of the source-side register-indirect resolution, which the name-matching finder
    cannot see (the dest names a ``disp(Rn)`` form, not the field)."""
    _ck = (str(stem), str(blueprint_dir), str(asm_dir), normalize_token(variable).upper())
    _hit = _SETTER_CACHE.get(_ck)
    if _hit is not None:
        return _hit
    # Prefer the COMPLETE ASM setter map when this job has the optional chain-facts
    # speedup artifacts. Unlike the legacy asm_setter_map, a covered full-map stem can
    # safely answer "absent" with [] instead of falling back to a live full-file scan.
    if _SETTER_MAP_JOB is not None:
        try:
            from vbt.precompute import db_artifacts as _DA
            if _DA.is_load_only():
                from vbt.precompute.chain_facts_db import (
                    ASM_SETTERS_FULL_COVERAGE_ARTIFACT,
                    EMPTY,
                    OK,
                    get_asm_setters_full,
                    load_coverage,
                    set_asm_setters_full_job,
                )
                _cov = load_coverage(_SETTER_MAP_JOB, ASM_SETTERS_FULL_COVERAGE_ARTIFACT) or {}
                if _cov.get(stem) in (OK, EMPTY):
                    set_asm_setters_full_job(_SETTER_MAP_JOB)
                    _full = get_asm_setters_full(stem, variable)
                    return _setter_cache_put(_ck, _full if _full is not None else [])
        except Exception:
            pass
    # Phase 1: in a load-only trace, return the PRECOMPUTED setter sites (a lookup, not the ~6 block
    # scans below). Disabled during the precompute build (is_load_only() is False there) so the build
    # still scans to produce the map. A miss (no blob for the stem, or var not in it) → live scan.
    if _SETTER_MAP_JOB is not None:
        try:
            from vbt.precompute import db_artifacts as _DA
            if _DA.is_load_only():
                from vbt.precompute.setter_map_db import load_asm_setters
                _pre = load_asm_setters(_SETTER_MAP_JOB, stem, variable)
                if _pre is not None:
                    return _setter_cache_put(_ck, _pre)
        except Exception:
            pass
    bp_path = resolve_asm_blueprint(stem, blueprint_dir)
    if not bp_path:
        return _setter_cache_put(_ck, [])
    try:
        bp = load_json(bp_path)
    except Exception:
        return _setter_cache_put(_ck, [])

    all_blocks: Set[str] = {
        str(b.get("id") or "") for b in (bp.get("blocks") or []) if b.get("id")
    }
    if not all_blocks:
        return _setter_cache_put(_ck, [])

    asm_file = (asm_dir / f"{stem}.asm") if asm_dir else Path(f"{stem}.asm")
    try:
        consts = collect_constant_symbols(bp, asm_file, bp_path=bp_path)
    except Exception:
        consts = set()

    out: List[SetterSite] = []
    # (1) reused name-matching finder (does not raise the whole call on its own failure —
    #     the register-indirect supplement below can still find a dest-only write).
    try:
        sites, _warn = _find_scoped_setter_sites(
            variable, bp, asm_file, all_blocks, asm_dir, constant_symbols=consts
        )
    except Exception:
        sites = []
    for s in sites:
        out.append(
            SetterSite(
                variable=variable,
                file_stem=stem,
                language="asm",
                line=int(s.get("line") or 0),
                instruction=str(s.get("instruction") or ""),
                block_id=str(s.get("routine") or ""),
                value=s.get("setter_expression"),
                setter_code_chunk=s.get("raw_line"),
                constant_source=bool(s.get("constant_source")),
                prior_value_required=bool(s.get("prior_value_required")),
                role=s.get("call_type"),
                flipc_alias_of=s.get("flipc_alias_of"),
                hardware_source=s.get("hardware_source"),
            )
        )

    # (2) register-indirect DESTINATION writes resolving to ``variable`` (the D3 fix).
    vu = normalize_token(variable).upper()
    seen = {(o.line, o.block_id) for o in out}
    for s in find_asm_indirect_dest_sites(stem, blueprint_dir, asm_dir, bp=bp):
        if s.variable.upper() == vu and (s.line, s.block_id) not in seen:
            out.append(SetterSite(
                variable=variable, file_stem=stem, language="asm", line=s.line,
                instruction=s.instruction, block_id=s.block_id, value=s.value,
                role="indirect_dest"))
    return _setter_cache_put(_ck, out)


def find_asm_setters(
    alias_set: AliasSet,
    blueprint_dir: Path,
    asm_dir: Optional[Path] = None,
    candidate_stems: Optional[List[str]] = None,
) -> List[SetterSite]:
    """Find ASM setters for every high-certainty ASM alias across candidate files.

    ``candidate_stems`` should come from the modifier index (cheap, correct).
    If omitted, the caller is expected to pass the relevant stems; this module
    does not glob the whole tree (that belongs to the precompute/orchestrator).
    """
    asm_names = [alias_set.canonical] + alias_set.for_language("asm")
    seen = set()
    results: List[SetterSite] = []
    for stem in candidate_stems or []:
        for name in asm_names:
            for site in find_asm_setters_in_file(name, stem, blueprint_dir, asm_dir):
                key = (site.file_stem, site.line, site.instruction, site.variable)
                if key in seen:
                    continue
                seen.add(key)
                results.append(site)
    return results
