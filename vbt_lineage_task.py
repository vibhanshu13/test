"""Celery task for the full variable-lineage trace — ``asm.vbt_lineage``.

POST /vbt/lineage: from JUST a ``kb_id`` + ``variable`` (language optional — auto-detected from the
precomputed modifier index), produce the variable's COMPLETE backward lineage:

  1. STRICT DB gate — the KB's ``index.db`` must contain every required VBT artifact
     (``verify_db.health_summary``). There is NO live-precompute fallback and NO JSON-file
     fallback: blueprints/source/cfg/setter-maps are all served from ``index.db`` blobs
     (the engine installs the DB overrides when ``job_id`` is passed). Missing artifacts
     ⇒ the task FAILS with the exact list, instead of silently degrading.
  2. Connectivity audit — cross-checks the stored call graph / fn-graph / entry-name map
     against the blueprint+source blobs and reports missing links (``db_audit``).
  3. Chain discovery — ALL of the variable's file chains (setters → call-graph LCA(s) →
     root→LCA chains, DB-only; same core as /vbt/lca-trace), including cross-language
     hops (C++ ``extern`` entry points into .asm and back).
  4. Per-chain DEEP trace — ``trace_root_variable`` with a lineage-grade dependent-variable
     recursion depth (default 10, vs the engine's default of 1) and shared hoisted caches,
     so every dependent variable is followed until it terminates.
  5. Lineage assembly — the engine's per-chain output is folded into an explicit variable
     graph (nodes + parent→child edges via struct/DSECT renames, parameter bindings and
     function outputs), and every TERMINAL variable is classified from DB evidence:
     ``user_input`` / ``database`` / ``constant`` / ``external`` / ``working_storage`` /
     ``unresolved`` — each with the evidence strings that justified it.

Output: ``jobs/<job_id>/output/vbt_lineage/lineage.json`` (plus ``lineage.debug.json``
when ``debug=True``).
"""
from __future__ import annotations

import gzip
import inspect
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from celery.exceptions import SoftTimeLimitExceeded

PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.config import settings
from api.tasks.celery_app import celery_app
from api.tasks._task_utils import redirect_output_to_log, job_log_handler
from api.storage.workspace import WorkspaceManager
from api.tasks.vbt_trace_task import (
    _resolve_blueprint_dir, _resolve_asm_dir, _resolve_graph_file, _safe_set_status, _make_ws)
from api.tasks.vbt_lca_trace_task import _chain_to_hops
from api.tasks.vbt_trace_all_chains_task import _ensure_chain_facts_artifacts

logger = logging.getLogger(__name__)

OPERATION = "vbt_lineage"

# Lineage-grade engine defaults — overridable via trace_options. The stock engine default
# (max_dep_var_depth=1) stops one level down; lineage must reach the true terminals.
LINEAGE_DEFAULT_DEPTH = 10
LINEAGE_DEFAULT_MAX_CHAINS = 50

# trace_options keys consumed by the discovery/budget/lineage layer — never forwarded to the engine.
_DISCOVERY_ONLY_OPT_KEYS = frozenset({
    "candidate_stems", "home_hint", "max_chains_per_lca", "max_chain_depth",
    "max_feasibility_checks", "max_chains_to_trace", "wall_clock_budget_sec",
    "db_read_markers", "user_input_markers", "input_msg_prefixes", "working_storage_prefixes",
})
_RESERVED_ENGINE_KWARGS = frozenset({
    "_shared_route", "_shared_src", "_shared_cfg_cache", "_shared_asm_bp_cache",
    "_shared_dep_node_cache", "bypass_trace_cache", "job_id",
})

# --------------------------------------------------------------------------- #
# Terminal-source markers (TPF conventions observed in the corpus). All are
# overridable per-request via trace_options — they are evidence generators,
# not hard-coded truth; every classification carries its evidence strings.
# --------------------------------------------------------------------------- #
# ASM macros that materialize a DATABASE record into core (TPF find/file family).
DB_READ_MARKERS = ("FINWC", "FIWHC", "FINDC", "FINHC", "FILEC", "FILNC", "FILUC", "GETFC")
# Markers that address the operator/user INPUT MESSAGE block.
USER_INPUT_MARKERS = ("AM0SG", "MI0MI", "IMSG")
# DSECT/field-name prefixes that live in the input-message block.
INPUT_MSG_PREFIXES = ("MI0", "AM0")
# ECB/work-area scratch prefixes — intra-transaction storage, neither user input nor database.
WORKING_STORAGE_PREFIXES = ("EBW", "EBX", "EBC", "CE1", "WK_")
# ASM name prefixes whose dep-var setter search is MODULE-LOCAL (discovery file only).
# Generic module work fields (WRKDBLW-style scratch) are reused by every module for
# unrelated computations — a corpus-wide setter search is noise, not lineage. NOT the
# same set as WORKING_STORAGE_PREFIXES: EBW/EBX/CE1 are ECB storage genuinely shared
# across the programs of one transaction, so those must stay corpus-searchable.
SCRATCH_LOCAL_PREFIXES = ("WRK",)

_MARKER_RE_CACHE: Dict[Tuple[str, ...], "re.Pattern[str]"] = {}


def _marker_re(markers: Tuple[str, ...]) -> "re.Pattern[str]":
    pat = _MARKER_RE_CACHE.get(markers)
    if pat is None:
        pat = re.compile(r"\b(" + "|".join(re.escape(m) for m in markers) + r")\b", re.IGNORECASE)
        _MARKER_RE_CACHE[markers] = pat
    return pat


# --------------------------------------------------------------------------- #
# 0. Per-KB derived-scan cache — the audit, the I/O-marker scan and the linkage-alias
# scan are KB-static, but computing them live means decoding the whole call/fn graph
# and every src: blob per request (fine at 300 files, minutes at 22k). Each is stored
# once as its own gzipped artifact and served O(1) afterwards. The salt embeds the
# corpus size (src/bp blob counts) so a re-precomputed KB invalidates the cache.
# --------------------------------------------------------------------------- #
LINEAGE_IO_SCAN_ARTIFACT = "lineage_io_scan"
LINEAGE_ENTRY_MAPS_ARTIFACT = "lineage_entry_maps"
LINEAGE_AUDIT_ARTIFACT = "lineage_audit"
LINEAGE_SCAN_VERSION = 1


def _corpus_salt(kb_id: str) -> str:
    from vbt.precompute import db_artifacts as DA
    n_src = len(DA.list_blob_keys_prefix(kb_id, "src:"))
    n_bp = len(DA.list_blob_keys_prefix(kb_id, "bp:"))
    return f"src={n_src},bp={n_bp}"


def _cached_scan(kb_id: str, artifact: str, salt: str, builder):
    """Serve ``artifact`` from the DB when fresh (version + salt match), else build and
    persist it. Cache failures degrade to a live build — never to a wrong answer."""
    from vbt.precompute import db_artifacts as DA
    try:
        row = DA.read_manifest(kb_id, artifact)
        if (row and int(row.get("version", -1)) == LINEAGE_SCAN_VERSION
                and row.get("source_hash") == salt):
            payload = DA.read_blob(kb_id, artifact)
            if payload:
                return DA.loads_gz(payload)
    except Exception as exc:
        logger.warning("[vbt_lineage] cache read %s failed (%s) — rebuilding", artifact, exc)
    obj = builder()
    try:
        DA.write_blob(kb_id, artifact, DA.dumps_gz(obj))
        DA.write_manifest(kb_id, artifact, LINEAGE_SCAN_VERSION, salt)
    except Exception as exc:
        logger.warning("[vbt_lineage] cache write %s failed: %s", artifact, exc)
    return obj


def warm_lineage_scans(kb_id: str) -> Dict[str, Any]:
    """Pre-build the three per-KB lineage scan artifacts (connectivity audit, linkage-alias
    map, I/O-marker scan with the DEFAULT markers) so the first /vbt/lineage query serves
    them O(1) instead of decoding the corpus graphs + every src blob. Idempotent — called
    from the pipeline's chains-precompute step. Returns summary counts."""
    salt = _corpus_salt(kb_id)

    def _build_audit() -> Dict[str, Any]:
        report, mctx = _audit_connections(kb_id)
        return {"report": report, "missing_ctx": mctx}

    audit = _cached_scan(kb_id, LINEAGE_AUDIT_ARTIFACT, salt, _build_audit)
    entry_maps = _cached_scan(kb_id, LINEAGE_ENTRY_MAPS_ARTIFACT, salt,
                              lambda: _scan_entry_maps(kb_id))
    io_salt = f"{salt}|db={','.join(DB_READ_MARKERS)}|in={','.join(USER_INPUT_MARKERS)}"
    io = _cached_scan(kb_id, LINEAGE_IO_SCAN_ARTIFACT, io_salt,
                      lambda: _scan_io_stems(kb_id, DB_READ_MARKERS, USER_INPUT_MARKERS))
    return {
        "audit_missing_stems": (audit.get("report") or {}).get("missing_stems_count", 0),
        "entry_aliases": len(entry_maps or {}),
        "io_db_read_stems": len((io or {}).get("db_read") or {}),
        "io_user_input_stems": len((io or {}).get("user_input") or {}),
    }


# --------------------------------------------------------------------------- #
# 1. Strict DB gate + connectivity audit
# --------------------------------------------------------------------------- #
def _strict_db_gate(kb_id: str, blueprint_dir, asm_dir) -> Dict[str, Any]:
    """Hard readiness gate: every artifact a fast DB-only trace needs must be present.

    Raises RuntimeError (actionable message, exact missing list) on FAIL — the lineage
    endpoint has NO fallback tier by design."""
    from vbt.precompute.verify_db import health_summary
    health = health_summary(kb_id, blueprint_dir=blueprint_dir, asm_dir=asm_dir,
                            require_speedup_artifacts=True)
    if health.get("status") != "PASS":
        raise RuntimeError(
            f"KB {kb_id} is not lineage-ready — missing DB artifacts: {health.get('missing')}. "
            f"Re-run VBT precompute (vbt.precompute_db) for this KB; the lineage endpoint has no "
            f"live-precompute or JSON fallback.")
    return health


def _audit_connections(kb_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Cross-check the stored graphs against the blueprint/source blobs and report missing links.

    All reads are index.db blobs — no filesystem access. Never raises (an audit finding is a
    warning on the payload, not a task failure).

    Returns ``(report, missing_ctx)`` where ``missing_ctx`` attributes the gaps for the terminal
    classifier: ``{"stems": {out-of-corpus stems}, "callees_of": {in-corpus stem: [missing stems
    it calls]}}`` — used to tell the user WHICH missing file most likely defines an unresolved
    terminal."""
    from vbt.precompute import db_artifacts as DA
    from vbt.precompute.graph_db import (
        FILE_CALL_GRAPH_ARTIFACT, FN_GRAPH_ADJ_ARTIFACT, ROUTE_NAME_TO_STEMS_ARTIFACT)

    report: Dict[str, Any] = {"checks": [], "missing_links": []}
    missing_ctx: Dict[str, Any] = {"stems": set(), "callees_of": {}}

    bp_stems: Set[str] = set()
    for key in DA.list_blob_keys_prefix(kb_id, "bp:"):
        name = key[3:]                              # "bp:aa71.asm.json"
        bp_stems.add(name.split(".", 1)[0].lower())

    try:
        fcg = DA.loads_gz(DA.read_blob(kb_id, FILE_CALL_GRAPH_ARTIFACT)) or {}
        nodes = {str(n).lower() if not isinstance(n, dict) else str(n.get("id", n.get("name", ""))).lower()
                 for n in fcg.get("nodes", [])}
        edges = fcg.get("edges", [])
        edge_endpoints: Set[str] = set()
        for e in edges:
            if isinstance(e, dict):
                edge_endpoints.add(str(e.get("source", "")).lower())
                edge_endpoints.add(str(e.get("target", "")).lower())
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                edge_endpoints.add(str(e[0]).lower())
                edge_endpoints.add(str(e[1]).lower())
        edge_endpoints.discard("")

        # Restrict to FILE-stem-like nodes: the graph also carries C++ method/function nodes
        # ("foo::operator=") and header names ("a2e.h") that legitimately have no blueprint.
        def _is_stem(n: str) -> bool:
            return bool(n) and "::" not in n and "." not in n and re.fullmatch(r"[a-z0-9_]+", n) is not None

        dangling = sorted(e for e in (edge_endpoints - nodes) if _is_stem(e))
        no_blueprint = sorted(n for n in nodes if _is_stem(n) and n not in bp_stems)

        # Attribution map for the terminal classifier: in-corpus stem → the OUT-of-corpus
        # modules it calls (the likely definition sites of that stem's unresolved reads).
        missing_ctx["stems"] = set(no_blueprint) | set(dangling)
        callees_of: Dict[str, Set[str]] = {}
        for e in edges:
            if isinstance(e, dict):
                src, tgt = str(e.get("source", "")).lower(), str(e.get("target", "")).lower()
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                src, tgt = str(e[0]).lower(), str(e[1]).lower()
            else:
                continue
            if src in bp_stems and tgt in missing_ctx["stems"]:
                callees_of.setdefault(src, set()).add(tgt)
        missing_ctx["callees_of"] = {s: sorted(t) for s, t in callees_of.items()}

        report["checks"].append({
            "name": "file_call_graph", "nodes": len(nodes), "edges": len(edges),
            "dangling_edge_endpoints": len(dangling), "nodes_without_blueprint": len(no_blueprint)})
        if dangling:
            report["missing_links"].append({
                "kind": "call_graph_edge_to_unknown_node",
                "detail": "call-graph edges reference stems that are not graph nodes",
                "count": len(dangling), "samples": dangling[:25]})
        if no_blueprint:
            report["missing_links"].append({
                "kind": "call_graph_node_without_blueprint",
                "detail": "graph stems with no bp:<stem>.* blob — traversal through them is blind",
                "count": len(no_blueprint), "samples": no_blueprint[:25]})
    except Exception as exc:
        report["checks"].append({"name": "file_call_graph", "error": f"{type(exc).__name__}: {exc}"})

    try:
        adj = DA.loads_gz(DA.read_blob(kb_id, FN_GRAPH_ADJ_ARTIFACT)) or {}
        node_keys = set(adj.keys())
        node_stems = {k.split("\t", 1)[0].lower() for k in node_keys}
        unknown_targets = 0
        samples: List[str] = []
        for _src, edge_list in adj.items():
            for e in edge_list or []:
                # edge = [callee_stem, callee_fn, line, cross]
                if not isinstance(e, (list, tuple)) or len(e) < 2:
                    continue
                tgt = f"{e[0]}\t{e[1]}"
                tgt_stem = str(e[0]).lower()
                # A callee with no adjacency key can still be a known LEAF module (no outgoing
                # calls) — only flag it when its stem has no blueprint either.
                if tgt not in node_keys and tgt_stem not in node_stems and tgt_stem not in bp_stems:
                    unknown_targets += 1
                    if len(samples) < 25:
                        samples.append(f"{_src} -> {e[0]}::{e[1]}")
                    src_stem = _src.split("\t", 1)[0].lower()
                    missing_ctx["stems"].add(tgt_stem)
                    missing_ctx["callees_of"].setdefault(src_stem, [])
                    if tgt_stem not in missing_ctx["callees_of"][src_stem]:
                        missing_ctx["callees_of"][src_stem].append(tgt_stem)
        report["checks"].append({"name": "fn_graph_adj", "nodes": len(node_keys),
                                 "edges_to_unknown_nodes": unknown_targets})
        if unknown_targets:
            report["missing_links"].append({
                "kind": "fn_graph_edge_to_unknown_function",
                "detail": "function-graph edges whose callee (stem, fn) has no node — usually an "
                          "extern/indirect call whose body is outside the corpus",
                "count": unknown_targets, "samples": samples})
    except Exception as exc:
        report["checks"].append({"name": "fn_graph_adj", "error": f"{type(exc).__name__}: {exc}"})

    try:
        n2s = DA.loads_gz(DA.read_blob(kb_id, ROUTE_NAME_TO_STEMS_ARTIFACT)) or {}
        orphan_entries = {name: stems for name, stems in n2s.items()
                          if stems and not any(str(s).lower() in bp_stems for s in stems)}
        report["checks"].append({"name": "route_name_to_stems", "entries": len(n2s),
                                 "entries_without_blueprint": len(orphan_entries)})
        if orphan_entries:
            report["missing_links"].append({
                "kind": "entry_name_without_blueprint",
                "detail": "extern entry names mapped to stems that have no blueprint blob",
                "samples": sorted(orphan_entries)[:25]})
    except Exception as exc:
        report["checks"].append({"name": "route_name_to_stems", "error": f"{type(exc).__name__}: {exc}"})

    report["status"] = "clean" if not report["missing_links"] else "missing_links_found"
    report["missing_stems_count"] = len(missing_ctx["stems"])
    # JSON-safe (sorted lists, no sets): the whole tuple is cached as a gzipped artifact.
    missing_ctx["stems"] = sorted(missing_ctx["stems"])
    missing_ctx["bp_stems"] = sorted(bp_stems)
    return report, missing_ctx


# C++→ASM linkage declarations: `#pragma map(fn, "ENTRY")` and `... fn(args) asm("ENTRY")`.
# The call graph records the C++ FUNCTION name as a node, but the missing FILE is the
# entry point's module (e.g. ckXc → TE90 ⇒ te90.asm). Resolving these keeps function
# names out of the missing-FILES report.
_PRAGMA_MAP_RE = re.compile(r"#pragma\s+map\s*\(\s*(\w+)\s*,\s*\"(\w+)\"\s*\)", re.IGNORECASE)
_ASM_ALIAS_RE = re.compile(r"\b(\w+)\s*\([^()]*\)\s*asm\s*\(\s*\"(\w+)\"\s*\)")


def _scan_entry_maps(kb_id: str) -> Dict[str, str]:
    """One pass over the C/C++ ``src:`` blobs → ``{cpp_function_lower: entry_module_lower}``
    for every #pragma map / asm("...") linkage alias whose function name differs from the
    entry name."""
    from vbt.precompute import db_artifacts as DA
    entry_map: Dict[str, str] = {}
    keys = [k for k in DA.list_blob_keys_prefix(kb_id, "src:")
            if not (k.endswith(".asm") or k.endswith(".mac"))]
    for key in keys:
        payload = DA.read_blob(kb_id, key)
        if payload is None:
            continue
        try:
            text = gzip.decompress(payload).decode("utf-8", errors="replace")
        except Exception:
            continue
        for rx in (_PRAGMA_MAP_RE, _ASM_ALIAS_RE):
            for m in rx.finditer(text):
                fn, entry = m.group(1).lower(), m.group(2).lower()
                if fn != entry:
                    entry_map[fn] = entry
    return entry_map


def _resolve_missing_symbols(missing_ctx: Dict[str, Any],
                             entry_map: Dict[str, str]) -> Dict[str, Any]:
    """Rewrite the missing-stem universe through the linkage aliases: a missing name that is
    really a mapped C++ function becomes its entry MODULE stem (dropped entirely when that
    module's blueprint exists). Returns the ``called_as`` map ``{module_stem: [fn names]}``
    for reporting."""
    bp_stems: Set[str] = missing_ctx.get("bp_stems") or set()
    called_as: Dict[str, List[str]] = {}

    def _map(stem: str) -> Optional[str]:
        entry = entry_map.get(stem)
        if entry is None:
            return stem                      # not a linkage alias — unchanged
        if entry in bp_stems:
            return None                      # module IS in the KB — nothing missing
        if stem not in called_as.setdefault(entry, []):
            called_as[entry].append(stem)
        return entry

    new_stems: Set[str] = set()
    for stem in missing_ctx.get("stems") or ():
        mapped = _map(stem)
        if mapped is not None:
            new_stems.add(mapped)
    missing_ctx["stems"] = new_stems

    new_callees: Dict[str, List[str]] = {}
    for src, callees in (missing_ctx.get("callees_of") or {}).items():
        mapped_list: List[str] = []
        for stem in callees:
            mapped = _map(stem)
            if mapped is not None and mapped not in mapped_list:
                mapped_list.append(mapped)
        if mapped_list:
            new_callees[src] = sorted(mapped_list)
    missing_ctx["callees_of"] = new_callees
    missing_ctx["called_as"] = {k: sorted(v) for k, v in called_as.items()}
    return missing_ctx["called_as"]


# --------------------------------------------------------------------------- #
# 2. Language auto-detection (modifier_index blob — DB-only)
# --------------------------------------------------------------------------- #
def _detect_languages(kb_id: str, variable: str, requested: Optional[str]) -> List[str]:
    """Resolve which language(s) to trace ``variable`` in.

    Explicit ``requested`` wins. Otherwise the variable is looked up in the precomputed
    modifier index (asm keys are normalized-upper; cpp keys are tails, matched
    case-insensitively). A name present in both universes traces BOTH."""
    if requested:
        if requested not in ("cpp", "asm"):
            raise ValueError(f"language must be 'cpp' or 'asm', got {requested!r}")
        return [requested]

    from vbt.precompute import db_artifacts as DA
    from vbt.precompute.resolvers_db import MODIFIER_ARTIFACT
    payload = DA.read_blob(kb_id, MODIFIER_ARTIFACT)
    if payload is None:
        raise RuntimeError(f"KB {kb_id}: modifier_index artifact missing — cannot auto-detect "
                           f"language; pass 'language' explicitly or re-run precompute.")
    mi = DA.loads_gz(payload) or {}
    langs: List[str] = []
    vu = variable.upper()
    asm_keys = mi.get("asm") or {}
    if vu in asm_keys or variable in asm_keys:
        langs.append("asm")
    cpp_keys = mi.get("cpp") or {}
    if variable in cpp_keys:
        langs.append("cpp")
    else:
        vl = variable.lower()
        if any(k.lower() == vl for k in cpp_keys):
            langs.append("cpp")
    if not langs:
        raise RuntimeError(
            f"variable {variable!r} not found in KB {kb_id}'s modifier index (neither asm nor cpp "
            f"universe) — check the spelling, or pass 'language' plus trace_options.home_hint if it "
            f"is only ever read (never written) in the corpus.")
    return langs


# --------------------------------------------------------------------------- #
# 3. Terminal-source evidence (src: blobs — DB-only, cached per run)
# --------------------------------------------------------------------------- #
_UPPER_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")


def _scan_io_stems(kb_id: str, db_markers: Tuple[str, ...],
                   input_markers: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    """One bounded pass over the ASM/MAC ``src:`` blobs: which stems read DATABASE records
    (TPF find/file macros) and which address the USER INPUT message. For each marker-hit
    stem the uppercase identifier tokens are kept so a DSECT name can be tied back to the
    stem that materialized its record. Returns
    ``{"db_read": {stem: [markers]}, "user_input": {stem: [markers]},
       "db_read_tokens": {stem: {TOKEN, ...}}}``."""
    from vbt.precompute import db_artifacts as DA
    db_re, in_re = _marker_re(db_markers), _marker_re(input_markers)
    out: Dict[str, Dict[str, Any]] = {"db_read": {}, "user_input": {}, "db_read_tokens": {}}
    keys = [k for k in DA.list_blob_keys_prefix(kb_id, "src:")
            if k.endswith(".asm") or k.endswith(".mac")]
    for key in keys:
        payload = DA.read_blob(kb_id, key)
        if payload is None:
            continue
        try:
            text = gzip.decompress(payload).decode("utf-8", errors="replace")
        except Exception:
            continue
        stem = key[4:].rsplit(".", 1)[0].lower()
        db_hits = sorted({m.group(1).upper() for m in db_re.finditer(text)})
        in_hits = sorted({m.group(1).upper() for m in in_re.finditer(text)})
        if db_hits:
            out["db_read"][stem] = db_hits
            # sorted list, not a set: this structure is cached as a JSON artifact
            out["db_read_tokens"][stem] = sorted(
                {m.group(0) for m in _UPPER_TOKEN_RE.finditer(text)})
        if in_hits:
            out["user_input"][stem] = in_hits
    return out


def _reader_stems(node: Dict[str, Any]) -> Set[str]:
    """File stems where this node was read/found — the attribution sites for evidence."""
    stems: Set[str] = set()
    for dep in node.get("dependencies") or []:
        fa = (dep or {}).get("foundAt") or {}
        f = fa.get("file")
        if f:
            stems.add(Path(str(f)).stem.lower())
    for s in node.get("setters") or []:
        loc = (s or {}).get("location") or {}
        f = loc.get("file")
        if f:
            stems.add(Path(str(f)).stem.lower())
    return stems


def _missing_candidates(node: Dict[str, Any],
                        missing_ctx: Optional[Dict[str, Any]]) -> List[str]:
    """Out-of-corpus modules called from the files where this node was read — the most likely
    definition sites of a variable the trace could not resolve. Uncapped: the whole list
    feeds the blocked-terminals ranking, and truncating it (especially alphabetically)
    silently drops real candidates."""
    if not missing_ctx:
        return []
    callees_of = missing_ctx.get("callees_of") or {}
    cands: Set[str] = set()
    for st in _reader_stems(node):
        cands.update(callees_of.get(st, ()))
    return sorted(cands)


def _classify_terminal(node: Dict[str, Any], *, io_stems: Dict[str, Dict[str, List[str]]],
                       const_names: Dict[str, Set[str]],
                       input_prefixes: Tuple[str, ...],
                       ws_prefixes: Tuple[str, ...],
                       missing_ctx: Optional[Dict[str, Any]] = None,
                       ) -> Tuple[str, List[str], List[str]]:
    """Classify one TERMINAL lineage node → (classification, evidence[], missing_candidates[]).

    ``missing_candidates`` is non-empty only for external/unresolved terminals: the
    out-of-corpus module stems most likely to define the variable (from the call graph of
    its reader files)."""
    name = str(node.get("name") or "")
    upper = name.upper()
    evidence: List[str] = []

    if node.get("externalCall") or node.get("functionBodyUnresolved"):
        cands = _missing_candidates(node, missing_ctx)
        evidence = [f"{name}: value produced by a call whose body is outside the corpus"]
        if cands:
            evidence.append(f"{name}: its reader files call missing module(s) "
                            f"{', '.join(cands)} — likely definition site(s)")
        return "external", evidence, cands

    lang = "asm" if upper == name and not any(c.islower() for c in name) else "cpp"
    if upper in const_names["asm"]:
        return "constant", [f"{name}: resolves in the ASM EQU/DC constant table"], []
    if name in const_names["cpp"]:
        return "constant", [f"{name}: resolves in the C++ enum/const table"], []

    member = node.get("memberOf") or {}
    member_name = str(member.get("name") or member.get("struct") or member.get("dsect") or "").upper()
    counterpart = node.get("counterpart") or {}

    if upper.startswith(tuple(input_prefixes)) or member_name.startswith(tuple(input_prefixes)):
        evidence.append(f"{name}: field of the input-message block "
                        f"({member_name or upper[:3] + '*'} — AM0SG/MI0MI family)")
        return "user_input", evidence, []

    if upper.startswith(tuple(ws_prefixes)):
        return "working_storage", [f"{name}: ECB/work-area prefix ({upper[:3]}*) — "
                                   f"intra-transaction scratch, not an external source"], []

    # DSECT/struct record field: use the stems where the variable is read + its home stem
    # as evidence sites — if any of them materializes records via find/file macros, the
    # field's bytes came from the DATABASE record image.
    stems: Set[str] = _reader_stems(node)
    home = member.get("file") or member.get("stem") or counterpart.get("file")
    if home:
        stems.add(Path(str(home)).stem.lower())

    is_member = bool(member_name) or bool(counterpart)
    db_sites = {st: io_stems["db_read"][st] for st in stems if st in io_stems["db_read"]}
    input_sites = {st: io_stems["user_input"][st] for st in stems if st in io_stems["user_input"]}

    if is_member and db_sites:
        for st, hits in sorted(db_sites.items()):
            evidence.append(f"{name}: record field ({member_name or 'struct member'}) read in "
                            f"{st} which loads records via {'/'.join(hits)}")
        return "database", evidence, []

    # Second-degree database evidence: the field's DSECT (or its ASM counterpart/alias name)
    # is addressed inside SOME stem that materializes records via the find/file macros, even
    # if the field itself was only read on the C++ side of the bridge.
    if is_member:
        dsect_names = {member_name} if member_name else set()
        cp_name = str(counterpart.get("name") or "").upper()
        if cp_name:
            dsect_names.add(cp_name)
        cp_member = counterpart.get("memberOf") or {}
        cpm = str(cp_member.get("name") or "").upper() if isinstance(cp_member, dict) else ""
        if cpm:
            dsect_names.add(cpm)
        for a in node.get("aliases") or []:
            if isinstance(a, dict) and a.get("language") == "asm" and a.get("name"):
                dsect_names.add(str(a["name"]).upper())
        dsect_names.discard("")
        for st, tokens in io_stems.get("db_read_tokens", {}).items():
            hit = sorted(dsect_names & tokens)
            if hit:
                evidence.append(f"{name}: its DSECT/field name(s) {'/'.join(hit)} are addressed "
                                f"in {st} which loads records via "
                                f"{'/'.join(io_stems['db_read'][st])}")
                return "database", evidence, []

    if input_sites and not is_member:
        for st, hits in sorted(input_sites.items()):
            evidence.append(f"{name}: read in {st} which addresses the input message via "
                            f"{'/'.join(hits)}")
        return "user_input", evidence, []
    if is_member:
        return "record_field", [f"{name}: DSECT/struct field ({member_name or 'unknown layout'}) "
                                f"with no find/file evidence on its reader stems"], []

    cands = _missing_candidates(node, missing_ctx)
    evidence = [f"{name}: no setters found and no source evidence matched ({lang})"]
    if cands:
        evidence.append(f"{name}: its reader files call missing module(s) "
                        f"{', '.join(cands)} — likely definition site(s)")
    return "unresolved", evidence, cands


# --------------------------------------------------------------------------- #
# 4. Lineage assembly from one engine chain output
# --------------------------------------------------------------------------- #
def _condition_summary(cond: Dict[str, Any]) -> Dict[str, Any]:
    """One guard, flattened for the lineage output: the expression text, where it sits
    in source, its order in the setter's guard sequence, and the ASM/constant metadata
    when present. blockId is dropped — the lineage output strips codeBlocks, so there
    is nothing for it to reference."""
    loc = cond.get("location") or {}
    # basename only: guard locations sometimes carry the absolute source path while the
    # setters use bare "stem.ext" names — normalize so both address files the same way.
    f = loc.get("file")
    out: Dict[str, Any] = {"condition": cond.get("condition"),
                           "file": Path(str(f)).name if f else None,
                           "line": loc.get("startLine"),
                           "order": cond.get("order")}
    for k in ("via", "resolvedConstants", "asmTest", "asmBranch", "decisionLine"):
        if cond.get(k) is not None:
            out[k] = cond[k]
    return out


def _setter_summary(setter: Dict[str, Any]) -> Dict[str, Any]:
    loc = setter.get("location") or {}
    out = {"file": loc.get("file"), "line": loc.get("startLine"),
           "value": setter.get("value")}
    if setter.get("valueResolved") is not None:
        out["valueResolved"] = setter.get("valueResolved")
    if setter.get("chain"):
        out["chain"] = setter.get("chain")
    conds = setter.get("conditions") or []
    if conds:
        # The full ordered guard list, inline (was a bare count, which forced a
        # second /vbt/trace call to see the expressions).
        out["conditions"] = [_condition_summary(c) for c in conds]
    if setter.get("notSet"):
        out["notSet"] = True
    return out


def _build_lineage(chain_out: Dict[str, Any], variable: str, *,
                   io_stems, const_names, input_prefixes, ws_prefixes,
                   missing_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fold one ``trace_root_variable`` output into an explicit lineage graph with
    classified terminals."""
    root = chain_out.get("rootVariable") or {}
    dep_entries = chain_out.get("dependentVariables") or []

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    seen_edges: Set[Tuple[Any, ...]] = set()
    raw_by_name: Dict[str, Dict[str, Any]] = {}

    root_node = {
        "name": variable, "role": "root", "terminal": False,
        "setters": [_setter_summary(s) for s in (root.get("setters") or [])],
        "memberOf": root.get("memberOf"), "aliases": root.get("aliases"),
    }
    nodes[variable] = root_node

    terminals: List[Dict[str, Any]] = []
    truncated: List[str] = []

    for entry in dep_entries:
        dv = (entry or {}).get("dependentVariable") or {}
        name = dv.get("name")
        if not name:
            continue
        raw_by_name.setdefault(name, dv)
        node = nodes.get(name)
        if node is None:
            node = {"name": name, "role": "dependent",
                    "terminal": bool(dv.get("terminal")),
                    "setters": [_setter_summary(s)
                                for s in (dv.get("setters") or [])]}
            for k in ("memberOf", "counterpart", "aliases", "qualified", "origin",
                      "indirection", "is_function_output", "externalCall",
                      "functionBodyUnresolved", "depthCapped", "scratchLocal"):
                if dv.get(k):
                    node[k] = dv[k]
            nodes[name] = node
        else:
            node["terminal"] = node.get("terminal", True) and bool(dv.get("terminal"))
        for dep in dv.get("dependencies") or []:
            parent = (dep or {}).get("variableName")
            if parent:
                fa = (dep or {}).get("foundAt") or {}
                key = (parent, name, fa.get("file"), fa.get("startLine"))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edge = {"from": parent, "to": name}
                if fa.get("file"):
                    edge["foundAt"] = {"file": fa.get("file"), "line": fa.get("startLine")}
                edges.append(edge)

    for name, node in nodes.items():
        if node.get("depthCapped"):
            truncated.append(name)
        if node.get("role") == "root" or not node.get("terminal"):
            continue
        # give the classifier the raw dependencies for read-site evidence
        raw = raw_by_name.get(name, {})
        cls, evidence, missing_cands = _classify_terminal(
            {**node, "dependencies": raw.get("dependencies") or []},
            io_stems=io_stems, const_names=const_names,
            input_prefixes=input_prefixes, ws_prefixes=ws_prefixes,
            missing_ctx=missing_ctx)
        node["classification"] = cls
        node["evidence"] = evidence
        terminal = {"name": name, "classification": cls, "evidence": evidence}
        if missing_cands:
            node["missingCandidates"] = missing_cands
            terminal["missingCandidates"] = missing_cands
        terminals.append(terminal)

    return {"nodes": list(nodes.values()), "edges": edges,
            "terminals": terminals, "truncatedNodes": sorted(set(truncated))}


# --------------------------------------------------------------------------- #
# 5. Worker body
# --------------------------------------------------------------------------- #
def run_lineage(job_id: str, options: Dict[str, Any],
                ws: Optional[WorkspaceManager] = None) -> Dict[str, Any]:
    from vbt.lca_trace import trace_variable_to_lca_chains
    from vbt.engine import Hop, trace_root_variable

    t_start = time.monotonic()
    variable: str = options["variable"]
    requested_language: Optional[str] = options.get("language") or None

    blueprint_dir = _resolve_blueprint_dir(job_id, options)
    asm_dir = _resolve_asm_dir(job_id, options)
    graph_file = _resolve_graph_file(blueprint_dir, options)
    kb_id = str(options.get("kb_id") or options.get("precompute_job_id") or job_id)
    topts: Dict[str, Any] = dict(options.get("trace_options") or {})
    debug = bool(options.get("debug", False))
    prune = bool(options.get("prune", False))
    max_chains_to_trace = int(topts.get("max_chains_to_trace", LINEAGE_DEFAULT_MAX_CHAINS))
    # split (default): stream one self-contained JSON per chain + an index.json manifest —
    # bounded memory, partial results survive a timeout, clients fetch only what they need.
    # split_chains=False restores the single consolidated lineage.json.
    split_chains = bool(options.get("split_chains", topts.get("split_chains", True)))
    db_markers = tuple(str(m).upper() for m in (topts.get("db_read_markers") or DB_READ_MARKERS))
    input_markers = tuple(str(m).upper() for m in (topts.get("user_input_markers") or USER_INPUT_MARKERS))
    input_prefixes = tuple(str(m).upper() for m in (topts.get("input_msg_prefixes") or INPUT_MSG_PREFIXES))
    ws_prefixes = tuple(str(m).upper() for m in (topts.get("working_storage_prefixes") or WORKING_STORAGE_PREFIXES))

    print(f"=== VBT lineage: variable={variable!r} language={requested_language or 'auto'} "
          f"kb={kb_id} ===", flush=True)

    # ---- 1. strict DB gate + connectivity audit (NO fallback tier) ----
    # LOAD-ONLY for the WHOLE task, switched on BEFORE the first DB open. The strict gate
    # below requires every artifact to be PRESENT in index.db — which is exactly load-only's
    # contract (freshness-by-presence, no source re-hashing). Without it, everything that
    # runs before the first chain trace executes in BUILD mode: the gate's first DB open
    # pays the O(DB-size) gvl-backfill scans (~minutes on a large index.db), and each
    # RouteEngine/route-cache/resolver preload re-hashes its O(corpus) source set — and on
    # ANY mtime/path drift vs the precompute run (redeploy, re-extract, different mount)
    # the freshness check fails and the preload silently falls back to a LIVE corpus sweep
    # (glob + parse of every blueprint/source) instead of reading the DB blobs. At 22k
    # files that live sweep is the "stuck /vbt/lineage" failure. trace_root_variable and
    # trace_variable_to_lca_chains set/restore the same flag internally, so keeping it ON
    # across the whole task is consistent with the per-call behavior. Restored in the main
    # finally below; the except here covers a failure BEFORE the main try is entered.
    from vbt.precompute import db_artifacts as DA
    prev_load_only = DA.is_load_only()
    DA.set_load_only(True)
    try:
        # ---- 1. strict DB gate + connectivity audit (NO fallback tier) ----
        _safe_set_status(ws, "running", progress="gate: DB artifact readiness")
        health = _strict_db_gate(kb_id, blueprint_dir, asm_dir)
        _ensure_chain_facts_artifacts(kb_id, options)     # idempotent; zero-cost when present
        _safe_set_status(ws, "running", progress="audit: graph connectivity")
        salt = _corpus_salt(kb_id)

        def _build_audit() -> Dict[str, Any]:
            report, mctx = _audit_connections(kb_id)
            return {"report": report, "missing_ctx": mctx}

        cached_audit = _cached_scan(kb_id, LINEAGE_AUDIT_ARTIFACT, salt, _build_audit)
        audit = cached_audit["report"]
        missing_ctx = cached_audit["missing_ctx"]
        missing_ctx["stems"] = set(missing_ctx.get("stems") or ())
        missing_ctx["bp_stems"] = set(missing_ctx.get("bp_stems") or ())
        # resolve C++ linkage aliases (#pragma map / asm("ENTRY")) so function names never
        # masquerade as missing FILES — ckXc becomes te90, functions stay symbols.
        entry_map = _cached_scan(kb_id, LINEAGE_ENTRY_MAPS_ARTIFACT, salt,
                                 lambda: _scan_entry_maps(kb_id))
        called_as = _resolve_missing_symbols(missing_ctx, entry_map)
        db_audit = {"artifacts": {"status": health["status"], "counts": health.get("counts", {}),
                                  "warnings": health.get("warnings", [])},
                    "connections": audit}
        if audit["missing_links"]:
            logger.warning("[vbt_lineage] job=%s kb=%s connectivity audit found %d missing-link kinds",
                           job_id, kb_id, len(audit["missing_links"]))

        # ---- 2. language auto-detect (DB-only) ----
        languages = _detect_languages(kb_id, variable, requested_language)
        print(f"  languages     : {languages}", flush=True)

        # ---- 3. terminal-source evidence + const tables (DB-only, cached per KB) ----
        _safe_set_status(ws, "running", progress="scan: terminal-source evidence")
        io_salt = f"{salt}|db={','.join(db_markers)}|in={','.join(input_markers)}"
        io_stems = _cached_scan(kb_id, LINEAGE_IO_SCAN_ARTIFACT, io_salt,
                                lambda: _scan_io_stems(kb_id, db_markers, input_markers))
        # tokens are stored as JSON lists — the classifier needs set intersection
        io_stems["db_read_tokens"] = {k: set(v) for k, v in
                                      (io_stems.get("db_read_tokens") or {}).items()}
        from vbt.precompute.resolvers_db import CONST_ARTIFACT
        const_blob = DA.loads_gz(DA.read_blob(kb_id, CONST_ARTIFACT)) or {}
        const_names = {"asm": {str(k).upper() for k in (const_blob.get("asm_const") or {})},
                       "cpp": set((const_blob.get("cpp_enum") or {}).keys())}
    except BaseException:
        DA.set_load_only(prev_load_only)
        raise

    # ---- engine option forwarding (lineage depth default overrides the engine's 1) ----
    sig_params = inspect.signature(trace_root_variable).parameters
    engine_opts: Dict[str, Any] = {
        k: v for k, v in topts.items()
        if k in sig_params and k not in _DISCOVERY_ONLY_OPT_KEYS and k not in _RESERVED_ENGINE_KWARGS
    }
    engine_opts.setdefault("max_dep_var_depth", LINEAGE_DEFAULT_DEPTH)
    # Generic ASM scratch work fields (WRK*) are searched module-locally by default —
    # a corpus-wide search returned 2,149 setters for one WRKDBLW read (98 modules of
    # noise). Opt out with trace_options.scratch_local_prefixes: [].
    engine_opts.setdefault("scratch_local_prefixes", SCRATCH_LOCAL_PREFIXES)
    if topts.get("candidate_stems") is not None:
        engine_opts["candidate_stems"] = topts["candidate_stems"]
    if topts.get("home_hint") is not None:
        engine_opts["home_hint"] = topts["home_hint"]

    soft_limit = getattr(settings, "TASK_SOFT_TIME_LIMIT", None)
    wall_budget: Optional[float] = None
    if topts.get("wall_clock_budget_sec") is not None:
        wall_budget = float(topts["wall_clock_budget_sec"])
    elif soft_limit:
        wall_budget = float(soft_limit) * 0.9

    traces: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    chain_errors: List[Dict[str, Any]] = []
    totals = {"chains_discovered": 0, "chains_traced": 0}
    terminal_counts: Dict[str, int] = {}
    # missing-file ranking: out-of-corpus stem → how many external/unresolved terminals
    # (across all chains) name it as a likely definition site.
    missing_file_refs: Dict[str, int] = {}

    # output dirs resolved up-front so split mode can stream each per-chain file on completion.
    # output_subdir: internal override (pipeline warm runs) so multiple warmed variables
    # don't overwrite each other's vbt_lineage/ output.
    base_dir = Path(ws.output_dir if ws is not None else (settings.JOBS_BASE_DIR / job_id / "output"))
    out_dir = base_dir / str(options.get("output_subdir") or "vbt_lineage")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ---- 4. hoisted shared caches + DB preloads (once) ----
        # (still under the load-only switch enabled before the gate — every preload takes
        # its O(1) presence-based DB path instead of the O(corpus) source-hash path)
        from vbt.reach.route import RouteEngine
        from vbt.precompute.graph_db import preload_route_graph, preload_cpp_call_edges
        from vbt.precompute.route_cache_db import preload_route_cache, persist_route_cache
        from vbt.precompute.resolvers_db import preload_resolvers
        shared_route = RouteEngine(blueprint_dir, graph_file, asm_dir,
                                   max_routes=int(engine_opts.get("max_routes", 200)),
                                   max_len=int(engine_opts.get("max_route_len", 16)),
                                   job_id=kb_id)
        shared_src: Dict[str, Any] = {}
        cfg_cache: Dict[str, Any] = {}
        asm_bp_cache: Dict[str, Any] = {}
        # Cross-chain dep-var node cache: chains 2..N of the same variable reuse chain 1's
        # dep-var subtree computes wherever the per-node chain-scope revalidation proves the
        # output identical (vbt/depvars/recurse.py DepNodeCache — byte-identical per chain).
        from vbt.depvars.recurse import DepNodeCache
        dep_node_cache = DepNodeCache()
        _safe_set_status(ws, "running", progress="preload: route graph + resolvers")
        preload_route_graph(shared_route, kb_id, blueprint_dir, graph_file)
        preload_cpp_call_edges(shared_route, kb_id, blueprint_dir, asm_dir)
        preload_route_cache(shared_route, kb_id)
        preload_resolvers(kb_id, blueprint_dir, asm_dir)

        loop_start = time.monotonic()
        for language in languages:
            # ---- 5. chain discovery (DB-only; extern hops included by the fn/file graph) ----
            _safe_set_status(ws, "running", progress=f"discovery: chains ({language})")
            disc = trace_variable_to_lca_chains(
                kb_id, variable, language,
                job_id=job_id, domain=options.get("domain"), prune=prune,
                candidate_stems=topts.get("candidate_stems"),
                home_hint=topts.get("home_hint"),
                allow_missing_forced_lca=bool(options.get("allow_missing_forced_lca", False)),
                allow_unattributed_setters=bool(options.get("allow_unattributed_setters", False)),
                max_chains_per_lca=int(topts.get("max_chains_per_lca", 200)),
                max_chain_depth=topts.get("max_chain_depth"),
                max_feasibility_checks=int(topts.get("max_feasibility_checks", 32)),
                blueprint_dir=str(blueprint_dir), asm_dir=str(asm_dir), graph_file=str(graph_file),
                debug=debug,
            )
            chains: List[List[Dict[str, Any]]] = list(disc.get("chains", []) or [])
            totals["chains_discovered"] += len(chains)
            warnings.extend(disc.get("warnings", []) or [])
            lang_trace: Dict[str, Any] = {
                "language": language, "lca_chains_discovered": len(chains),
                "lca_truncated": bool(disc.get("truncated", False)), "chains": []}
            traces.append(lang_trace)
            print(f"--- discovery[{language}]: {len(chains)} chains ---", flush=True)

            if len(chains) > max_chains_to_trace:
                warnings.append({"code": "trace_budget_cap",
                                 "message": f"{language}: tracing first {max_chains_to_trace} of "
                                            f"{len(chains)} discovered chains"})
                chains = chains[:max_chains_to_trace]

            # ---- 6. per-chain DEEP trace + lineage fold ----
            n = len(chains)
            for i, chain in enumerate(chains):
                if wall_budget is not None and (time.monotonic() - loop_start) >= wall_budget:
                    warnings.append({"code": "trace_budget_exceeded",
                                     "message": f"wall-clock budget {wall_budget:.0f}s exceeded; "
                                                f"{n - i} {language} chain(s) not traced"})
                    break
                chain_prefix = _chain_to_hops(chain, Hop)
                stems = [h.stem for h in chain_prefix]
                _t = time.monotonic()
                try:
                    out = trace_root_variable(
                        variable, language, chain_prefix,
                        blueprint_dir=blueprint_dir, asm_dir=asm_dir, graph_file=graph_file,
                        job_id=kb_id,
                        _shared_route=shared_route, _shared_src=shared_src,
                        _shared_cfg_cache=cfg_cache, _shared_asm_bp_cache=asm_bp_cache,
                        _shared_dep_node_cache=dep_node_cache,
                        **engine_opts,
                    )
                except SoftTimeLimitExceeded:
                    raise
                except Exception as exc:
                    chain_errors.append({"language": language, "chain_index": i, "files": stems,
                                         "error": f"{type(exc).__name__}: {exc}"})
                    logger.exception("[vbt_lineage] job=%s [%s %d/%d] chain FAILED", job_id,
                                     language, i + 1, n)
                    continue

                out.pop("codeBlocks", None)   # lineage keeps locations, not source bodies
                lineage = _build_lineage(out, variable, io_stems=io_stems, const_names=const_names,
                                         input_prefixes=input_prefixes, ws_prefixes=ws_prefixes,
                                         missing_ctx=missing_ctx)
                chain_terminal_counts: Dict[str, int] = {}
                for t in lineage["terminals"]:
                    terminal_counts[t["classification"]] = \
                        terminal_counts.get(t["classification"], 0) + 1
                    chain_terminal_counts[t["classification"]] = \
                        chain_terminal_counts.get(t["classification"], 0) + 1
                    for stem in t.get("missingCandidates") or []:
                        missing_file_refs[stem] = missing_file_refs.get(stem, 0) + 1
                root_setters = [_setter_summary(s)
                                for s in ((out.get("rootVariable") or {}).get("setters") or [])]
                if split_chains:
                    # stream ONE self-contained JSON per chain; keep only the manifest entry.
                    fname = f"chain_{language}_{i:04d}.json"
                    doc = {"variable": variable, "language": language,
                           "chain_index": i, "files": stems, "rootSetters": root_setters,
                           "lineage": lineage}
                    (out_dir / fname).write_text(json.dumps(doc, indent=2))
                    lang_trace["chains"].append({
                        "chain_index": i, "files": stems, "file": fname,
                        "nodes": len(lineage["nodes"]), "edges": len(lineage["edges"]),
                        "terminals": chain_terminal_counts,
                    })
                else:
                    lang_trace["chains"].append({
                        "chain_index": i, "files": stems,
                        "lineage": lineage, "rootSetters": root_setters,
                    })
                totals["chains_traced"] += 1
                logger.info("[vbt_lineage] job=%s [%s %d/%d] DONE (%.2fs) nodes=%d terminals=%d "
                            "dep-node-cache: %s",
                            job_id, language, i + 1, n, time.monotonic() - _t,
                            len(lineage["nodes"]), len(lineage["terminals"]),
                            dep_node_cache.stats())
                if (i + 1) % 5 == 0 or (i + 1) == n:
                    _safe_set_status(ws, "running",
                                     progress=f"trace[{language}]: {i + 1}/{n} chains")

        try:
            persist_route_cache(shared_route, kb_id)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("[vbt_lineage] job=%s persist_route_cache failed: %s", job_id, exc)
    finally:
        # global override + per-trace state cleanup (mirrors run_all_chains_trace)
        try:
            from backward_traversal.utils.blueprint_utils import (
                clear_blueprint_override, clear_source_override)
            from vbt.precompute.cfg_db import clear_cfg_db
            from vbt.setters.asm_setters import set_asm_setter_map_job, set_asm_setters_full_job
            from vbt.setters.cpp_setters import set_cpp_setter_map_job, set_cpp_file_writes_job
            from vbt.precompute.fn_facts_db import set_fn_facts_job
            from vbt.precompute.chain_facts_db import set_chain_facts_job
            from vbt.conditions.cpp_conditions import clear_cpp_conditions_cache

            clear_blueprint_override()
            clear_source_override()
            clear_cfg_db()
            set_asm_setter_map_job(None)
            set_asm_setters_full_job(None)
            set_cpp_setter_map_job(None)
            set_cpp_file_writes_job(None)
            set_fn_facts_job(None)
            set_chain_facts_job(None)
            clear_cpp_conditions_cache()
        except Exception:
            pass
        # restore UNCONDITIONALLY — a failed clear above must not leak load-only=True
        # into the next task this worker process runs.
        DA.set_load_only(prev_load_only)

    # ---- 7. emit ----
    # missing_files.json — the actionable answer to "which files are missing": every
    # out-of-corpus name the KB's graphs reference, ranked by how many of THIS variable's
    # unresolved/external terminals point at it (upload the top ones first). Each entry is
    # kind-tagged: a 4-char TPF name is a missing MODULE (.asm/.cpp program); anything else
    # is a SYMBOL — a function declared in an in-corpus header whose implementation file
    # is absent (e.g. castod/caspars ⇒ the casdate implementation), or a builtin.
    _module_re = re.compile(r"^[a-z]{2}[0-9a-z]{2}$")
    _builtins = {"char", "long", "int", "bool", "void", "short", "auto", "enum", "float",
                 "double", "signed", "unsigned", "size_t", "boolean", "abs", "memcpy",
                 "memset", "strlen", "strcpy", "strcmp", "strncpy", "strncmp", "sprintf"}

    def _kind(stem: str) -> str:
        if stem in _builtins:
            return "builtin"
        has_digit = any(c.isdigit() for c in stem)
        if stem in called_as:                 # resolved linkage entry point
            # TPF application program names always mix letters and digits (te90, av5e);
            # an all-letter entry (PACK, UNPACK) is a system/library service routine,
            # not an application source file anyone could upload.
            return "module" if has_digit else "library"
        if _module_re.fullmatch(stem) and has_digit:
            return "module"
        return "library" if _module_re.fullmatch(stem) else "symbol"

    ranked = sorted(missing_file_refs.items(), key=lambda kv: (-kv[1], kv[0]))
    missing_files_payload = {
        "kb_id": kb_id,
        "variable": variable,
        "missing_stems_total": len(missing_ctx.get("stems") or ()),
        "kinds": {"module": "an application program file (.asm/.cpp) absent from the KB — "
                            "upload it",
                  "library": "a TPF system/library service entry (e.g. PACK/UNPACK) — not an "
                             "application source file",
                  "symbol": "a function with no body in the KB — its implementation file is "
                            "absent",
                  "builtin": "a C/C++ builtin type or library function — nothing to upload"},
        "referenced_by_this_lineage": [
            {"stem": stem, "kind": _kind(stem), "blocked_terminals": count,
             **({"called_as": called_as[stem]} if stem in called_as else {})}
            for stem, count in ranked],
        "all_missing_stems": [
            {"stem": s, "kind": _kind(s),
             **({"called_as": called_as[s]} if s in called_as else {})}
            for s in sorted(missing_ctx.get("stems") or ())],
    }
    (out_dir / "missing_files.json").write_text(json.dumps(missing_files_payload, indent=2))

    payload: Dict[str, Any] = {
        "variable": variable,
        "languages": languages,
        "db_audit": db_audit,
        "traces": traces,
        "summary": {**totals, "chain_errors": len(chain_errors),
                    "terminals_by_classification": terminal_counts,
                    "missing_files_top": [
                        {"stem": s, "kind": _kind(s), "blocked_terminals": c}
                        for s, c in ranked[:20]],
                    "missing_files_report": "missing_files.json"},
    }
    if warnings:
        payload["warnings"] = warnings
    if chain_errors:
        payload["chainErrors"] = chain_errors

    if split_chains:
        payload["split"] = True
        payload["chains_dir"] = out_dir.name   # per-chain files streamed during the loop
        out_path = out_dir / "index.json"
    else:
        out_path = out_dir / "lineage.json"
    out_path.write_text(json.dumps(payload, indent=2))

    if ws is not None:
        try:
            result_files = ws.collect_output_files()
            _safe_set_status(ws, "success", result_files=result_files, progress="done")
        except Exception as exc:  # pragma: no cover
            logger.warning("[vbt_lineage] could not finalize output files: %s", exc)
            _safe_set_status(ws, "success", progress="done")

    elapsed = round(time.monotonic() - t_start, 3)
    print(f"=== VBT lineage complete: {totals['chains_traced']}/{totals['chains_discovered']} chains, "
          f"terminals={terminal_counts} ({elapsed}s) → {out_path} ===", flush=True)
    return {
        "job_id": job_id, "status": "success", "variable": variable, "languages": languages,
        **totals, "terminals_by_classification": terminal_counts,
        "db_audit_status": audit["status"], "chain_errors": len(chain_errors),
        "missing_files_total": len(missing_ctx.get("stems") or ()),
        "missing_files_top": [{"stem": s, "kind": _kind(s), "blocked_terminals": c}
                              for s, c in ranked[:10]],
        "warnings": warnings, "split": split_chains, "output_file": str(out_path),
        "output_dir": str(out_dir), "elapsed": elapsed,
    }


@celery_app.task(bind=True, name="asm.vbt_lineage")
def run_vbt_lineage(self, job_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Full-lineage coordinator — see the module docstring for the phase breakdown."""
    ws = _make_ws(job_id)
    _safe_set_status(ws, "running", operation=OPERATION)
    log_file = ws.log_file if ws is not None else (settings.JOBS_BASE_DIR / job_id / "job.log")
    with redirect_output_to_log(log_file):
        with job_log_handler(log_file, job_id=job_id):
            try:
                from vbt.lca_trace import DbArtifactMissing
            except Exception:
                DbArtifactMissing = RuntimeError  # type: ignore
            try:
                return run_lineage(job_id, options, ws=ws)
            except SoftTimeLimitExceeded:
                msg = "Task exceeded soft time limit and was aborted."
                logger.error("[vbt_lineage] job %s: %s", job_id, msg)
                _safe_set_status(ws, "failed", error=msg, progress="failed")
                return {"job_id": job_id, "status": "failed", "error": msg,
                        "variable": options.get("variable")}
            except DbArtifactMissing as exc:
                msg = f"DbArtifactMissing: {exc}"
                logger.error("[vbt_lineage] job %s: %s", job_id, msg)
                _safe_set_status(ws, "failed", error=msg, progress="failed")
                return {"job_id": job_id, "status": "failed", "error": msg,
                        "variable": options.get("variable")}
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                logger.exception("[vbt_lineage] job %s failed: %s", job_id, msg)
                _safe_set_status(ws, "failed", error=msg, progress="failed")
                return {"job_id": job_id, "status": "failed", "error": msg,
                        "variable": options.get("variable")}
