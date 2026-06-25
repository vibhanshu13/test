"""Dependent-variable recursion (SPEC §6) — the same engine, re-rooted per dep var.

For each dep var discovered at (file y, line x) on chain C, find its setters that are
chain-scoped reachable (v1: in the discovery file *before* line x, or in a chain
file), collect their conditions, harvest their own dep vars, and recurse — bounded by
``max_depth`` and memoized so shared sub-walks are computed once (the DAG reuse idea).
"""
from __future__ import annotations

import logging
import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from vbt.interfaces import SetterSite
from vbt.setters.cpp_setters import (
    find_cpp_setters_in_file, find_cpp_writes_under_path, find_cpp_returns_in_function,
)
from vbt.setters.asm_setters import find_asm_setters_in_file
from vbt.conditions.cpp_conditions import collect_cpp_conditions
from vbt.conditions.asm_conditions import collect_asm_conditions
from vbt.cpp_frontend.wrapper import run_cfg_extract as _raw_cfg_extract
from vbt.depvars.extract import extract_dep_vars, DepVarRef, AsmIndirectContext
from vbt.output.codeblocks import CodeBlockStore
from vbt.precompute.modifier_index import get_modifier_index, ModifierIndex
from vbt.reach.route import function_reaches_before
from vbt.reach.fn_attr import line_to_function
from vbt.resolve.name_resolver import resolve as resolve_aliases
from vbt.resolve.membership import get_membership_resolver
from backward_traversal.route_finder import Endpoint
from backward_traversal.utils.blueprint_utils import (
    resolve_asm_blueprint, load_json, collect_constant_symbols,
)


# 22k bound: max distinct off-chain candidate files route-checked per dep var.
_MAX_OFFCHAIN_FILES = 100

# Shares the engine's "vbt" logger root so --progress controls both. STDERR only
# (the JSON result goes to stdout). No-ops below INFO.
_LOG = logging.getLogger("vbt.depvars")


def run_cfg_extract(path: str) -> list:
    """Fault-isolated cfg_extract: returns [] on any extraction failure."""
    try:
        return _raw_cfg_extract(path)
    except (FileNotFoundError, RuntimeError) as exc:
        _LOG.warning("cfg_extract skipped %s — %s", path, exc)
        return []


def _loc(file: str, a: int, b: int) -> Dict[str, Any]:
    return {"file": file, "startLine": a, "endLine": b}


def _fn_lines(f: Dict) -> List[int]:
    return [s.get("line") for b in f.get("cfg_blocks", []) for s in b.get("stmts", []) if s.get("line")]


def _fn_containing_line(cfg_fns: List[Dict], line: int) -> Optional[str]:
    """Function owning ``line`` by nearest-preceding START (vbt/reach/fn_attr) — for
    the discovery-file dep-var rule (which function the read sits in). Was span-
    containment, which returns None for a line past a truncated switch body and so
    silently degraded the call-order-aware chain scoping (D-F7) to the textual test."""
    return line_to_function(cfg_fns, line)


_CONST_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_#$@]*(?:::[A-Za-z_][A-Za-z0-9_]*)?")


def _resolved_consts(text: str, cr) -> Dict[str, str]:
    """Resolved enum/EQU/DC values for constant tokens in a condition/value (D5 —
    mirror the root engine so dep-var conditions carry the same metadata)."""
    if cr is None:
        return {}
    found: Dict[str, str] = {}
    for m in _CONST_TOK.finditer(text or ""):
        tok = m.group(0)
        if tok in found:
            continue
        v = cr.resolve(tok, "cpp")
        if v is None:
            v = cr.resolve(tok, "asm")
        if v is not None:
            found[tok] = v
    return found


def _dep_key(dep: DepVarRef) -> Tuple[str, str]:
    """Identity of a dependent variable for memo + results dedup (D-F5 root-cause fix).

    The canonical ``dep.name`` is the TAIL (C++ ``a.b.c.field`` → ``field``), so two
    genuinely DIFFERENT fields that share a tail (``allCidRecords.cidRecord`` vs
    ``tempCidRecord.cidRecord``, or a ``value``/``mm``/``transactionCode`` field that
    lives in several distinct structs — 187 such tails are written in >1 file in this
    corpus) would COLLIDE on the bare tail: the second is skipped at the memo gate and
    its setters/subtree never computed, and/or it overwrites the first in ``results`` →
    silently-dropped lineage.

    Keying on ``(tail.upper(), qualified)`` distinguishes those distinct fields while
    keeping behavior IDENTICAL for true duplicates (same tail AND same qualified reached
    via two parents → same key → one entry, dependencies merged). A dep whose
    ``qualified`` is empty (full path == tail, no struct prefix seen) keeps tail-only
    keying via ``qualified == ""``. The set of distinct ``(tail, qualified)`` pairs is
    finite, so the recursion still terminates (cycle-safe).

    Function-output deps have an empty ``qualified`` (the canonical form is the bare
    function name), so two DIFFERENT functions sharing a name in different files would
    collide. We disambiguate them by their resolved DEFINING file in the second slot
    (``@fn:<def_file>``) — distinct defs → distinct nodes; same def reached twice still
    merges. Member-path deps keep ``qualified`` (never ``@fn:``-prefixed), so the
    path-family ``_key_for_qual`` 2-tuple keying is unaffected."""
    if dep.qualified:
        return (dep.name.upper(), dep.qualified)
    if dep.is_function_output and dep.def_file:
        return (dep.name.upper(), f"@fn:{dep.def_file}")
    return (dep.name.upper(), "")


def _dep_chunk(store, found_at: Optional[Dict[str, Any]]) -> Optional[str]:
    """The source line(s) where a parent reads this dependency (D5 — populate
    dependencies[].relevantCodeChunk instead of leaving it null)."""
    if store is None or not found_at:
        return None
    try:
        a = found_at.get("startLine") or 0
        b = found_at.get("endLine") or a
        return store.chunk(found_at["file"], a, b) or None
    except Exception:
        return None


def _cpp_fn_block(store, stem: str, fn: Optional[str], cfg: List[Dict]) -> Optional[str]:
    """Register (dedup) + return the codeBlocks id for a C++ function (D5)."""
    if store is None or not fn:
        return None
    lo = hi = 0
    for f in cfg:
        nm = f.get("function") or ""
        if nm == fn or nm.endswith("::" + fn):
            ls = _fn_lines(f)
            if ls:
                lo, hi = min(ls), max(ls)
            break
    return store.cpp_function(stem, fn, lo, hi)


def _function_return_setters(fn_name: str, midx: ModifierIndex, asm_dir: Path,
                             cfg_cache: Dict[str, List[Dict]], *,
                             cr=None, store=None, call_site_stem: Optional[str] = None):
    """Re-root a C++ function-output dep var into the callee's RETURN sites (SPEC §6).

    Finds where the function is DEFINED, then treats each ``return`` as a 'setter' of
    the function's output, with the path-conditions to that return and the returned
    expression's operands as further dep vars.

    C5: when several files locally re-define the same function (header-light corpus),
    prefer the definition in the CALL-SITE file (where this output was discovered) —
    that is the one actually invoked — instead of harvesting from every same-named def.
    (Headerless, returns whose expression has unresolved types are dropped by Clang —
    full capture needs the struct headers present in the 22k codebase.)
    """
    setter_jsons: List[Dict[str, Any]] = []
    child: List[DepVarRef] = []
    by_ref_outputs: List[Dict[str, Any]] = []
    tail = fn_name.split("::")[-1].split(".")[-1].split("->")[-1]
    defs = sorted(midx.files_defining_function(tail))
    if call_site_stem and call_site_stem in defs:
        defs = [call_site_stem]                     # C5: the called definition
    for stem in defs:
        cpp_path = asm_dir / f"{stem}.cpp"
        if not cpp_path.exists():
            continue
        fns = cfg_cache.setdefault(stem, run_cfg_extract(str(cpp_path)))
        f = next((x for x in fns if x.get("function", "").endswith(tail)), None)
        if not f:
            continue
        fblk = _cpp_fn_block(store, stem, f.get("function"), fns)
        # cfg_extract truncates the body at a collapsed switch and drops returns (the
        # Pa::ReturnCode classifier class came back with 0 returns → wrongly terminal).
        # Recover them ROBUSTLY with the tree-sitter parser (find_cpp_returns_in_function
        # — no regex, no span dependency: inline/multi-line/comment-safe), unioned with any
        # CFG returns and deduped by line.
        cfg_rets = [s for b in f["cfg_blocks"] for s in b.get("stmts", []) if s.get("kind") == "return"]
        seen_lines = {s.get("line") for s in cfg_rets}
        ts_rets = [{"line": ln, "text": f"return {expr}"}
                   for (ln, expr) in find_cpp_returns_in_function(cpp_path, f.get("function") or tail)
                   if expr and ln not in seen_lines]
        rets = list(cfg_rets) + ts_rets
        for r in rets:
            line = r.get("line") or 0
            text = str(r.get("text") or "").strip()
            val = re.sub(r"^return\b\s*", "", text).strip() or text
            conds = collect_cpp_conditions(cpp_path, line, f.get("function"), functions=fns)
            fake = SetterSite(variable=tail, file_stem=stem, language="cpp", line=line,
                              instruction="return", block_id=f.get("function") or "", value=val)
            _ret_deps = extract_dep_vars(
                fake, [], cpp_const_names=cr.bare_cpp_const_names() if cr else None)
            child += _ret_deps
            setter_jsons.append({
                "setterId": len(setter_jsons) + 1,
                "value": val,
                "valueResolved": cr.resolve(val, "cpp") if cr else None,
                "setterCodeChunk": text,
                "location": _loc(f"{stem}.cpp", line, line),
                "blockId": fblk,
                "dependentVariables": _setter_dep_vars_json(_ret_deps),
                "conditions": [
                    {"order": c.order, "condition": c.condition, "blockId": fblk,
                     "location": _loc(c.location.file, c.location.start_line, c.location.end_line),
                     **({"resolvedConstants": rc} if (rc := _resolved_consts(c.condition, cr)) else {})}
                    for c in conds
                ],
            })
        # SPEC §6: a function's output is its return value AND every by-reference /
        # pointer parameter it WRITES (cfg_extract flags by_ref_params[*].written).
        # The value written into the param is itself a dependency (D8: *outRef=o->depA
        # → depA), so re-root the write RHS too.
        assigns = f.get("assignments") or []
        for p in (f.get("by_ref_params") or []):
            if not p.get("written"):
                continue
            pname = str(p.get("name") or "")
            if not pname:
                continue
            writes = [a for a in assigns
                      if str(a.get("lhs_path") or "") in (pname, "*" + pname)
                      or str(a.get("lhs_path") or "").startswith(pname + "->")
                      or str(a.get("lhs_path") or "").startswith("*" + pname)]
            for a in writes:
                fw = SetterSite(variable=pname, file_stem=stem, language="cpp",
                                line=a.get("line") or 0, instruction="assign",
                                block_id=f.get("function") or "", value=a.get("rhs_expr"))
                child += extract_dep_vars(
                    fw, [], cpp_const_names=cr.bare_cpp_const_names() if cr else None)
            by_ref_outputs.append({
                "param": pname,
                "writes": [{"value": a.get("rhs_expr"),
                            "location": _loc(f"{stem}.cpp", a.get("line") or 0, a.get("line") or 0)}
                           for a in writes],
            })
    return setter_jsons, child, by_ref_outputs


_ROOT_SUFFIX = re.compile(r"^([A-Za-z_]\w*)(.*)$")
_SIMPLE_LVALUE = re.compile(r"^[A-Za-z_]\w*(?:(?:->|\.)[A-Za-z_]\w*)*$")
_BARE_IDENT = re.compile(r"^[A-Za-z_]\w*$")   # a single identifier ⇒ a PURE rename


def _ref_root_and_suffix(dep: DepVarRef) -> Tuple[str, str]:
    """The ROOT identifier of a C++ dep's reference and the member suffix after it.

    ``countOfCidLrecs`` → ("countOfCidLrecs", ""); ``cidRecord->count`` (qualified)
    → ("cidRecord", "->count"). The root is what a formal parameter is named; the
    suffix is re-attached onto the caller's actual argument so a member read of a
    struct parameter (``param->field``) re-roots to ``actualArg->field``."""
    ref = (dep.qualified or dep.name or "").replace(" ", "")
    m = _ROOT_SUFFIX.match(ref)
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def _enclosing_fn_params(cfg: List[Dict], fn_name: Optional[str]) -> List[Dict]:
    """The ordered formal-parameter records of ``fn_name`` (cfg_extract ``params``)."""
    if not fn_name:
        return []
    for f in cfg:
        nm = f.get("function") or ""
        if nm == fn_name or nm.endswith("::" + fn_name):
            return f.get("params") or []
    return []


_CALLEE_TAILS_MEMO: Dict[int, tuple] = {}        # id(cfg) -> (cfg, frozenset of callee tails)
_CALLEE_TAILS_MEMO_MAX = 512


def _callee_tails(cfg) -> frozenset:
    """Every callee tail name in a cfg, memoized per cfg list (id + held-ref re-check, like the
    bridge edge index). Lets ``_param_binding_setters`` skip — in O(1) — a candidate file that NEVER
    calls the target function F, instead of re-scanning its whole call graph on every one of the
    hundreds of param-binding queries over overlapping candidate sets. Byte-identical: a file with no
    F-call yields no binding sites, so skipping it changes nothing. The tail is split exactly as the
    inner filter (``callee.split('::')[-1]``)."""
    k = id(cfg)
    hit = _CALLEE_TAILS_MEMO.get(k)
    if hit is not None and hit[0] is cfg:
        return hit[1]
    tails = frozenset(str(c.get("callee") or "").split("::")[-1]
                      for g in cfg for c in (g.get("calls") or []))
    if len(_CALLEE_TAILS_MEMO) >= _CALLEE_TAILS_MEMO_MAX:
        _CALLEE_TAILS_MEMO.pop(next(iter(_CALLEE_TAILS_MEMO)))
    _CALLEE_TAILS_MEMO[k] = (cfg, tails)
    return tails


def _param_binding_setters(dep: DepVarRef, disc_stem: str, asm_dir: Path,
                           cfg_cache: Dict[str, List[Dict]], *,
                           route_engine, chain_hops, chain_set: Set[str],
                           reach_union: Optional[Set[str]], store, cr,
                           max_offchain_files: int):
    """IN-direction boundary re-root (the dual of ``_function_return_setters``).

    A formal parameter is **bound, not set**: read-only inside its function, its
    value-defining "setter" is the **actual-argument expression at each call site**.
    The name-keyed setter search finds nothing for such a parameter and the dep is
    wrongly declared *terminal* (e.g. ``countOfCidLrecs`` read in
    ``setSoftCardIndicatorValues`` but bound from the caller's ``numberOfCidRecords``
    at arg-index 2). This re-resolves the dep's identity across the call boundary:

      1. find the function ``F`` the dep was read in, and the index of the formal
         whose name == the dep's ROOT identifier (return empty if the dep's root is
         not a parameter — the normal setter search then applies);
      2. for every reachability-scoped call site of ``F`` (on the live prefix: a
         chain/discovery file, or an off-chain caller reachable from a chain hop,
         bounded by ``max_offchain_files``), read ``args[index]`` — the actual
         argument in the **caller's** scope;
      3. emit it as a binding "setter" (value = the actual, guarded by the call-site
         conditions in the caller), and return the actual's variables as child deps
         so the trace continues upstream in the caller.

    Returns ``(setter_jsons, child_deps)``; ``([], [])`` when the dep is not a formal
    parameter of its enclosing function. Sound: every kept call site is a verified
    binding under some execution (OR alternatives); nothing is fabricated."""
    if dep.language != "cpp" or route_engine is None:
        return [], []
    disc_cpp = asm_dir / f"{disc_stem}.cpp"
    if not disc_cpp.exists():
        return [], []
    disc_cfg = cfg_cache.setdefault(disc_stem, run_cfg_extract(str(disc_cpp)))
    enclosing_fn = _fn_containing_line(disc_cfg, dep.found_at.get("startLine") or 0)
    params = _enclosing_fn_params(disc_cfg, enclosing_fn)
    if not params:
        return [], []
    root, suffix = _ref_root_and_suffix(dep)
    pidx = next((int(p.get("index")) for p in params if p.get("name") == root), None)
    if pidx is None:
        return [], []
    f_tail = (enclosing_fn or "").split("::")[-1]

    # candidate caller files: the live-prefix files (chain + discovery) plus the
    # forward-reachable set (a caller of F sits between a chain hop and F, so it is
    # forward-reachable from that hop → in reach_union). Bounded by the scale prune.
    cand = set(chain_set) | {disc_stem}
    if reach_union is not None:
        cand |= reach_union
    setter_jsons: List[Dict[str, Any]] = []
    children: List[DepVarRef] = []
    seen_sites: Set[Tuple[str, int, str]] = set()
    offchain_opened: Set[str] = set()
    for stem in sorted(cand):
        on_prefix = stem in chain_set or stem == disc_stem
        if not on_prefix:
            if len(offchain_opened) >= max_offchain_files and stem not in offchain_opened:
                continue                       # bounded; off-chain caller cap (surfaced upstream)
        cpp_path = asm_dir / f"{stem}.cpp"
        if not cpp_path.exists():
            continue
        cfg2 = cfg_cache.setdefault(stem, run_cfg_extract(str(cpp_path)))
        if f_tail not in _callee_tails(cfg2):     # F not called in this file → no binding sites; skip scan
            continue
        for g in cfg2:
            caller_fn = g.get("function")
            for c in (g.get("calls") or []):
                if (str(c.get("callee") or "").split("::")[-1] != f_tail
                        or pidx >= len(c.get("args") or [])):
                    continue
                call_line = int(c.get("line") or 0)
                actual = str(c["args"][pidx]).strip()
                if not actual:
                    continue
                # scope to the live prefix: an off-chain caller is kept only if it is
                # reachable from a chain hop (mirrors the off-chain setter rule).
                if not on_prefix:
                    offchain_opened.add(stem)
                    try:
                        ok = any(route_engine.reachable(Endpoint(h[0], h[1], h[2]),
                                                        Endpoint(stem, "cpp", caller_fn))
                                 for h in (chain_hops or []))
                    except Exception:
                        ok = False
                    if not ok:
                        continue
                sig = (stem, call_line, actual)
                if sig in seen_sites:
                    continue
                seen_sites.add(sig)
                # re-attach the member suffix onto a simple-lvalue actual so a struct
                # param's member read (``param->field``) becomes ``actualArg->field``.
                bound_expr = (actual + suffix) if (suffix and _SIMPLE_LVALUE.match(actual)) else actual
                fake = SetterSite(variable=dep.name, file_stem=stem, language="cpp",
                                  line=call_line, instruction="param_binding",
                                  block_id=caller_fn or "", value=bound_expr)
                fchild = extract_dep_vars(
                    fake, [], cpp_const_names=cr.bare_cpp_const_names() if cr else None)
                # A binding is only meaningful when crossing the boundary reveals a
                # name the codebase-wide TAIL-based setter search can't already see.
                # That is exactly a scalar value parameter renamed at the call site
                # (``countOfCidLrecs`` ← ``numberOfCidRecords``): its tail IS the param
                # name and has no global setter. A pointer/reference parameter passed
                # through (``o`` → ``o``, or a member read ``o->depA``) has the SAME
                # tail and is already covered globally — emitting a binding there would
                # only duplicate (and pollute every dep's setter list). So skip unless
                # the actual introduces a tail distinct from the dep's own. (Gated here,
                # BEFORE the expensive call-site condition collection, since pass-through
                # is the common case at scale.)
                if not any(c.name.upper() != dep.name.upper() for c in fchild):
                    continue
                conds = collect_cpp_conditions(cpp_path, call_line, caller_fn, functions=cfg2)
                cblk = _cpp_fn_block(store, stem, caller_fn, cfg2)
                # Classify the binding: a PURE RENAME (the actual is a single identifier —
                # the SAME logical value continuing under a different name across the call
                # boundary, e.g. ``numberOfCidRecords``) vs a DERIVED value (a computed arg
                # like ``a + b`` or ``getX()`` — a genuine data dependency). The recursion
                # enqueues a pure rename at the SAME depth (a rename is not a new derivation,
                # so it must not consume the dependency-depth budget); a derived value costs
                # depth+1. Tagged via ``relationship`` so trace_dependents routes it.
                rel = "param_rename" if _BARE_IDENT.match(bound_expr) else "param_binding"
                for cd in fchild:
                    cd.relationship = rel
                children += fchild
                setter_jsons.append({
                    "setterId": 0,        # renumbered by the caller when merged
                    "value": bound_expr,
                    "valueResolved": cr.resolve(bound_expr, "cpp") if cr else None,
                    "setterCodeChunk": (store.chunk(f"{stem}.cpp", call_line, call_line)
                                        if store else None),
                    "location": _loc(f"{stem}.cpp", call_line, call_line),
                    "blockId": cblk,
                    "dependentVariables": _setter_dep_vars_json(fchild),
                    "conditions": [
                        {"order": c2.order, "condition": c2.condition, "blockId": cblk,
                         "location": _loc(c2.location.file, c2.location.start_line, c2.location.end_line),
                         **({"resolvedConstants": rc} if (rc := _resolved_consts(c2.condition, cr)) else {})}
                        for c2 in conds
                    ],
                    "paramBinding": {"formal": root, "actual": actual, "argIndex": pidx,
                                     "function": enclosing_fn, "callerFunction": caller_fn},
                })
    return setter_jsons, children


def _filter_chain_scoped(setters, dep: DepVarRef, chain_files: Set[str],
                         upstream_bounds: Dict[str, int], *,
                         chain_entry_fn: Optional[Dict[str, Optional[str]]] = None,
                         blueprint_dir: Optional[Path] = None,
                         asm_dir: Optional[Path] = None,
                         cfg_cache: Optional[Dict[str, List[Dict]]] = None):
    """Chain-scoped reaching-def (SPEC §6): a dep-var setter is relevant iff it
    executes on the live prefix that reaches the read.

      - in the discovery file y → in the read's function before the read line, OR in
        a function reachable from it via calls occurring before the read line;
      - in an upstream chain file F → in F's chain-entry function before F's *descend
        call site* (``upstream_bounds[F]``), OR in a function reachable from that
        entry via calls before the descend (D5: ``rxMidHelper`` is called @28 < the
        descend @32, so its body @41 — textually *after* the descend — is still live).

    The textual ``s.line < bound`` test is WRONG for a callee whose body sits after
    the descend line but is *called* before it; we use call-order reachability
    (``function_reaches_before``) when C++ function info + a blueprint are available,
    and fall back to the line test otherwise (so the legacy 4-arg call still works)."""
    disc_file = Path(dep.found_at["file"]).stem
    disc_line = dep.found_at.get("startLine") or 0
    chain_entry_fn = chain_entry_fn or {}

    def _fn_at(stem: str, line: int) -> Optional[str]:
        if cfg_cache is None or asm_dir is None:
            return None
        cfg = cfg_cache.setdefault(stem, run_cfg_extract(str(Path(asm_dir) / f"{stem}.cpp")))
        return _fn_containing_line(cfg, line)

    def _reaches_before(stem: str, entry_fn: Optional[str], target_fn: Optional[str],
                        before: int) -> Optional[bool]:
        if blueprint_dir is None or not entry_fn or not target_fn:
            return None
        return function_reaches_before(stem, "cpp", entry_fn, target_fn,
                                       Path(blueprint_dir), before_line=before,
                                       asm_dir=asm_dir)

    out = []
    for s in setters:
        F = s.file_stem
        if F == disc_file:
            if s.language == "cpp" and s.function and cfg_cache is not None:
                disc_fn = _fn_at(F, disc_line)
                if disc_fn and s.function == disc_fn:
                    if disc_line == 0 or s.line <= disc_line:
                        out.append(s)
                else:
                    rb = _reaches_before(F, disc_fn, s.function, disc_line)
                    if (rb if rb is not None else (disc_line == 0 or s.line <= disc_line)):
                        out.append(s)
            elif disc_line == 0 or s.line <= disc_line:
                out.append(s)
        elif F in upstream_bounds:
            B = upstream_bounds[F]
            entry = chain_entry_fn.get(F)
            if s.language == "cpp" and s.function and entry and cfg_cache is not None:
                if s.function == entry:
                    if s.line < B:
                        out.append(s)
                else:
                    rb = _reaches_before(F, entry, s.function, B)
                    if (rb if rb is not None else (s.line < B)):
                        out.append(s)
            elif s.line < B:
                out.append(s)
        elif F in chain_files:
            # chain file with no recorded descend line (e.g. the entry, or ASM
            # single-entry) → keep (conservative; refined when a bound is known).
            out.append(s)
    return out


def _setter_dep_vars_json(refs: List[DepVarRef]) -> List[Dict[str, Any]]:
    """Serialize a setter's extracted operand dep-vars to the per-setter
    ``dependentVariables`` shape the ROOT setters already emit (engine.py) — so
    non-root dep-var cards also show what each setter's VALUE depends on (e.g. a
    setter ``cidioReturnCode = getCidRecord(amexCmNumber,cidRecord,ll9l9,cidioKey)``
    surfaces those operands instead of an empty list). Deduped by upper-name,
    order-preserving; fields mirror engine.py's root-setter serialization exactly."""
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for d in refs or []:
        if not d.name:
            continue
        ident = d.name.upper()
        if ident in seen:
            continue
        seen.add(ident)
        e: Dict[str, Any] = {"name": d.name, "foundAt": d.found_at}
        if d.qualified:
            e["qualified"] = d.qualified
        if getattr(d, "indirection", None):
            e["indirection"] = d.indirection
        out.append(e)
    return out


# --------------------------------------------------------------------------- #
# Per-node COMPUTE / MERGE split (BFS-level parallelization)
# --------------------------------------------------------------------------- #
# The dep-var BFS is restructured into BATCHES (one batch = one BFS level). Per
# node we separate a PURE, read-only COMPUTE (``_compute_node``: no shared-state
# mutation; uses a LOCAL CodeBlockStore + LOCAL child/family accumulators) from a
# single-threaded MERGE that replays every side effect in BFS pop order. The MERGE
# alone mutates the shared memo/results/edge_seen/node_rescope/family_*/store/q.
#
# Parallelism (VBT_DEPVAR_WORKERS>1): the COMPUTE + MERGE always run SERIALLY; only
# the dominant per-node cost — the cross-file route walks, whose results are pure +
# shared through the route engine's ``_routes_memo`` — is warmed in PARALLEL before a
# level's compute (``_warm_route_memo``: collect the level's route queries in fork
# workers, resolve the unique ones in parallel on the warm engine, merge the answers
# into the parent memo). The serial compute then runs with a warm memo. Because the
# warmed route values are exactly what the serial path would compute on demand, the
# output is BYTE-IDENTICAL for any worker count; serial (the default) skips the pool
# entirely. (Per-node process parallelism of the compute itself was tried and is
# strictly worse here: the route walk dominates and does not parallelize well across
# processes — its shared sub-caches and the route memo cannot be shared mid-batch —
# while shipping each node's heavy code-block output back across the fork boundary
# adds cost. Warming the route memo keeps the win where the time actually is.)


def _path_comps(qual: str) -> List[str]:
    return [c for c in qual.replace("->", ".").split(".") if c]


def _key_for_qual(qual: str) -> Tuple[str, str]:
    comps = _path_comps(qual)
    return ((comps[-1] if comps else qual).upper(), qual)


def _edge_sig(parent: str, dep: DepVarRef) -> Tuple[str, str, int]:
    return (parent, dep.found_at.get("file", ""), dep.found_at.get("startLine") or 0)


@dataclass
class _NodeCtx:
    """Read-only context shared by every per-node COMPUTE call (fork-COW safe).

    Everything here is treated read-only by ``_compute_node``; the per-node
    mutable caches (cfg/const/bp/alias) are recreated per worker process via fork
    (deterministic memoization of pure functions, so a fresh cache yields the
    same result). The merge owns all shared accumulators."""
    blueprint_dir: Path
    asm_dir: Path
    chain_set: Set[str]
    upstream_bounds: Dict[str, int]
    route_engine: Any
    chain_hops: Optional[List[Tuple[str, str, Optional[str]]]]
    max_depth: int
    const_resolver: Any
    max_offchain_files: int
    asm_max_levels: int
    max_descendants: int
    midx: ModifierIndex
    memb: Any
    reach_union: Optional[Set[str]]
    # per-process mutable memoization caches (NOT shared across processes; in the
    # serial path they persist across nodes, which only speeds compute — never
    # changes output, since they memoize pure deterministic functions).
    cfg_cache: Dict[str, List[Dict]] = field(default_factory=dict)
    const_cache: Dict[str, Set[str]] = field(default_factory=dict)
    bp_cache: Dict[str, Tuple[str, Dict]] = field(default_factory=dict)
    alias_cache: Dict[Tuple[str, str], List] = field(default_factory=dict)


@dataclass
class _NodeOut:
    """The pure result of computing ONE node — everything the merge needs to
    apply this node's side effects in pop order. Picklable (returned across the
    fork boundary)."""
    key: Tuple[str, str]
    dep: DepVarRef                       # carries the resolved def_file (function-output)
    result: Dict[str, Any]              # {"dependentVariable": dv}
    edge_seed: Tuple[str, str, int]
    node_rescope: Optional[Dict[str, Any]]   # None for function-output nodes
    child_deps: List[Tuple[str, DepVarRef, int]]   # UNFILTERED (owner, dep, depth)
    family_parent_adds: List[Tuple[Tuple[str, str], str]]   # (depkey, parent_qual)
    family_child_adds: List[Tuple[Tuple[str, str], str]]    # (depkey, child_qual)
    family_capped_adds: List[Tuple[str, str]]               # depkeys
    blocks: List[Tuple[str, Dict]]      # local store blocks in registration order


def _expand_family_compute(dep: DepVarRef, depth: int, ctx: _NodeCtx,
                           child_deps: List[Tuple[str, DepVarRef, int]],
                           fp_add: List, fc_add: List, fcap_add: List) -> None:
    """COMPUTE-side path-family expansion: collect ancestors + descendant deps and
    the family-link adds WITHOUT touching shared state (no memo filter — the merge
    applies it at enqueue). Mirrors the serial ``_expand_family`` exactly; the only
    difference is appending to local lists instead of ``q``/``family_*``/``memo``."""
    if not (dep.expand_family and dep.language == "cpp" and dep.qualified):
        return
    comps = _path_comps(dep.qualified)
    disc_file = dep.found_at.get("file", "")
    # ANCESTORS — consecutive strict prefixes (a.b.c -> a.b -> a).
    child_q = dep.qualified
    for i in range(len(comps) - 1, 0, -1):
        parent_q = ".".join(comps[:i])
        fp_add.append((_key_for_qual(child_q), parent_q))
        fc_add.append((_key_for_qual(parent_q), child_q))
        anc = DepVarRef(name=comps[i - 1], language="cpp", relationship="ancestor",
                        found_at={"file": disc_file, "startLine": 0, "endLine": 0},
                        qualified=parent_q, origin="ancestor",
                        expand_family=False, path_parent=".".join(comps[:i - 1]))
        child_deps.append((dep.name, anc, depth))
        child_q = parent_q
    # DESCENDANTS — writes whose path strictly extends dep.qualified.
    scan = sorted(((ctx.reach_union or set()) | ctx.chain_set | {Path(dep.found_at["file"]).stem}))
    seen_q: Set[str] = set()
    for stem in scan[:ctx.max_offchain_files]:
        if len(seen_q) >= ctx.max_descendants:
            fcap_add.append(_key_for_qual(dep.qualified))
            break
        p = ctx.asm_dir / f"{stem}.cpp"
        if not p.exists():
            continue
        for s in find_cpp_writes_under_path(comps, p):
            qd = getattr(s, "lhs_path", "") or ""
            if not qd or qd in seen_q:
                continue
            if len(seen_q) >= ctx.max_descendants:
                fcap_add.append(_key_for_qual(dep.qualified))
                break
            seen_q.add(qd)
            fc_add.append((_key_for_qual(dep.qualified), qd))
            fp_add.append((_key_for_qual(qd), dep.qualified))
            dcomps = _path_comps(qd)
            dsc = DepVarRef(name=dcomps[-1] if dcomps else qd, language="cpp",
                            relationship="descendant",
                            found_at={"file": f"{s.file_stem}.cpp", "startLine": 0, "endLine": 0},
                            qualified=qd, origin="descendant", expand_family=False,
                            path_parent=dep.qualified)
            child_deps.append((dep.name, dsc, depth))


def _emit_setter_run(setter_jsons, seen, relevant, owner_name, depth, *,
                     ctx: _NodeCtx, store, enqueue) -> bool:
    """Emit each not-yet-seen relevant setter onto ``setter_jsons`` (build its JSON,
    enqueue its child deps via the ``enqueue`` callback), deduped by (file,line,value)
    via ``seen``. Reused for the FIRST build (compute) AND for re-scoping a
    re-encountered node (merge). Returns child_depth_capped.

    ``store`` + ``enqueue`` are injected so the SAME body serves two call sites:
      * COMPUTE   → a LOCAL store, ``enqueue`` appends to a local child list (UNFILTERED);
      * RESCOPE   → the SHARED store, ``enqueue`` applies the memo filter then ``q.append``.
    Output JSON is byte-identical either way (pure function of the setter + caches)."""
    cr = ctx.const_resolver
    capped = False
    for s in relevant:
        ext = "cpp" if s.language == "cpp" else "asm"
        dkey = (f"{s.file_stem}.{ext}", s.line, str(s.value))
        if dkey in seen:        # same write reached via multiple paths / read-sites
            continue
        seen.add(dkey)
        asm_trunc = False   # ASM depth-cap truncation flag (set only in the ASM branch)
        if s.language == "cpp":
            cpp_path = ctx.asm_dir / f"{s.file_stem}.cpp"
            fns = ctx.cfg_cache.setdefault(s.file_stem, run_cfg_extract(str(cpp_path)))
            conds = collect_cpp_conditions(cpp_path, s.line, s.function, functions=fns)
            child = extract_dep_vars(
                s, conds,
                cpp_const_names=cr.bare_cpp_const_names() if cr else None)
            loc = _loc(f"{s.file_stem}.cpp", s.line, s.line)
            dblk = _cpp_fn_block(store, s.file_stem, s.function, fns)   # D5
        else:
            if s.file_stem not in ctx.bp_cache:
                bpp = str(resolve_asm_blueprint(s.file_stem, ctx.blueprint_dir))
                ctx.bp_cache[s.file_stem] = (bpp, load_json(bpp))
            bpp, bp = ctx.bp_cache[s.file_stem]
            consts = ctx.const_cache.setdefault(
                s.file_stem, collect_constant_symbols(bp, ctx.asm_dir / f"{s.file_stem}.asm"))
            conds, asm_trunc = collect_asm_conditions(bp, bpp, s.block_id, s.line,
                                                      f"{s.file_stem}.asm", max_levels=ctx.asm_max_levels)
            # Register-indirect setter-source resolution (SPEC §6 register-drop gap).
            asm_ind_ctx = AsmIndirectContext(bp_data=bp, bp_path=bpp, asm_dir=ctx.asm_dir,
                                             route_engine=ctx.route_engine)
            child = extract_dep_vars(s, conds, constant_symbols=consts,
                                     asm_indirect_ctx=asm_ind_ctx)
            loc = _loc(f"{s.file_stem}.asm", s.line, s.line)
            _bo = next((b for b in (bp.get("blocks") or []) if str(b.get("id")) == s.block_id), {})
            dblk = store.asm_block(s.file_stem, s.block_id,                # D5
                                   int(_bo.get("start_line") or s.line),
                                   int(_bo.get("end_line") or s.line)) if store else None
        if depth < ctx.max_depth:
            for cd in child:
                enqueue(owner_name, cd, depth + 1)
        elif child:
            capped = True   # children exist but max_dep_var_depth hit — surfaced
        # D5: dep-var setter shape mirrors the root setter (blockId, valueResolved,
        # resolvedConstants); ASM also carries verbatim asmTest/asmBranch + decisionLine.
        setter_entry = {
            "setterId": len(setter_jsons) + 1,
            "value": s.value,
            "valueResolved": cr.resolve(s.value or "", s.language) if cr else None,
            "setterCodeChunk": s.setter_code_chunk,
            "location": loc,
            "blockId": dblk,
            # Per-setter dependent variables: the operands THIS setter's value depends
            # on (already extracted as `child` to recurse) — mirrors the root setter
            # shape so non-root dep-var cards show them instead of an empty list.
            "dependentVariables": _setter_dep_vars_json(child),
            "conditions": [
                {"order": c.order, "condition": c.condition, "blockId": dblk,
                 "location": _loc(c.location.file, c.location.start_line, c.location.end_line),
                 **({"asmTest": c.raw_test, "decisionLine": c.location.end_line,
                     **({"asmBranch": c.raw_branch} if c.raw_branch else {})} if c.raw_test else {}),
                 **({"resolvedConstants": rc} if (rc := _resolved_consts(c.condition, cr)) else {})}
                for c in conds
            ],
        }
        # depth-cap truncation: a necessary ASM guard may have been dropped, so this
        # dep-var setter's condition list is partial — never present it as complete.
        if asm_trunc:
            setter_entry["pathsCapped"] = True
        setter_jsons.append(setter_entry)
    return capped


def _eff_line(line):
    # line 0 = a structural (family) read with NO before-line bound → maximally
    # permissive; represent as +inf so it dominates every real read in that file.
    return 10 ** 9 if not line else int(line)


def _compute_node(item: Tuple[str, DepVarRef, int]) -> _NodeOut:
    """PURE per-node computation (no shared-state mutation). Reads the read-only
    context from the module global ``_CTX`` (set before the pool forks so a worker
    inherits it via COW; also set in the serial path). Uses a LOCAL CodeBlockStore +
    LOCAL child/family accumulators; the merge replays the side effects in pop order.
    Reproduces the serial per-node logic EXACTLY. The authoritative call runs SERIALLY
    in the parent; the parallel route-warming pass calls a lean route-only variant
    (``_warm_node_routes``) in workers, not this full builder.

    ``item`` is ``(parent, dep, depth)``. NOTE: function-output ``def_file``
    resolution (the C5 call-site-def rule) is done by the CALLER before keying, so
    ``dep.def_file`` is already set when this runs and ``_dep_key(dep)`` is stable."""
    parent, dep, depth = item
    ctx = _CTX
    # Hang-locating trace: the LAST line before a hang names the dep node whose re-root is stuck.
    if _LOG.isEnabledFor(logging.DEBUG):
        _LOG.debug("    dep node compute [d%d]: %s (%s) <- %s",
                   depth, dep.name, getattr(dep, "language", "?"), parent)
    lstore = CodeBlockStore(ctx.asm_dir)
    key = _dep_key(dep)

    child_deps: List[Tuple[str, DepVarRef, int]] = []
    fp_add: List[Tuple[Tuple[str, str], str]] = []
    fc_add: List[Tuple[Tuple[str, str], str]] = []
    fcap_add: List[Tuple[str, str]] = []

    # path-family ancestors + descendants (collected, not enqueued).
    _expand_family_compute(dep, depth, ctx, child_deps, fp_add, fc_add, fcap_add)

    def _enqueue(owner, cd, d):
        child_deps.append((owner, cd, d))

    # Function-output dep var (C++): re-root into the callee's RETURN sites AND
    # the by-reference/pointer params it writes (SPEC §6).
    if dep.language == "cpp" and dep.is_function_output:
        fset, fchild, by_ref = _function_return_setters(
            dep.name, ctx.midx, ctx.asm_dir, ctx.cfg_cache, cr=ctx.const_resolver, store=lstore,
            call_site_stem=Path(dep.found_at["file"]).stem)   # C5: the called def
        depth_capped = False
        if depth < ctx.max_depth:
            for cd in fchild:
                child_deps.append((dep.name, cd, depth + 1))
        elif fchild:
            depth_capped = True   # children exist but max_dep_var_depth hit — surfaced
        _mf = ctx.memb.resolve(dep.name, dep.language, dep.qualified)
        dvf = {
            "name": dep.name,
            "dependencies": [{"variableName": parent, "foundAt": dep.found_at,
                              "relevantCodeChunk": _dep_chunk(lstore, dep.found_at)}],
            "setters": fset,
            "terminal": len(fset) == 0 and not by_ref,
            "is_function_output": True,
            "memberOf": _mf["memberOf"],        # usually None for a function output
            "counterpart": _mf["counterpart"],
        }
        if by_ref:
            dvf["byRefOutputs"] = by_ref
        if len(fset) == 0 and not by_ref:
            fn_tail = dep.name.split("::")[-1].split(".")[-1].split("->")[-1]
            if ctx.midx.files_defining_function(fn_tail):
                dvf["functionBodyUnresolved"] = True
            else:
                dvf["externalCall"] = True
        if dep.qualified:
            dvf["qualified"] = dep.qualified
        if dep.def_file:
            dvf["definedIn"] = dep.def_file   # resolved defining file (identity)
        if dep.origin != "primary":
            dvf["origin"] = dep.origin   # ancestor | descendant (path-family node)
        if depth_capped:
            dvf["depthCapped"] = True   # dep subtree truncated at max_dep_var_depth — surfaced
        if dep.indirection:
            dvf["indirection"] = dep.indirection   # register-indirect give-up envelope
        return _NodeOut(
            key=key, dep=dep, result={"dependentVariable": dvf},
            edge_seed=_edge_sig(parent, dep), node_rescope=None,
            child_deps=child_deps, family_parent_adds=fp_add, family_child_adds=fc_add,
            family_capped_adds=fcap_add, blocks=list(lstore.blocks.items()))

    # Cross-language COUNTERPART setter search (SPEC §7).
    akey = (dep.name, dep.language)
    if akey not in ctx.alias_cache:
        ctx.alias_cache[akey] = list(resolve_aliases(dep.name, dep.language, ctx.asm_dir).aliases)
    dep_aliases = ctx.alias_cache[akey]
    disc_stem = Path(dep.found_at["file"]).stem

    targets = [(dep.name, dep.language)] + [(a.name, a.language) for a in dep_aliases]
    setters = []
    for tname, tlang in targets:
        writer_files = ctx.midx.files_for(tname, tlang)
        if ctx.reach_union is not None:
            writer_files = {f for f in writer_files if f in ctx.reach_union or f in ctx.chain_set}
        cand = set(ctx.chain_set) | writer_files
        if tlang == dep.language:
            cand.add(disc_stem)
        cand_sorted = sorted(cand)
        if tlang == "cpp":
            tail = tname.split(".")[-1].split("->")[-1]
            fp = [dep.qualified] if (tlang == dep.language and dep.qualified) else None
            for stem in cand_sorted:
                p = ctx.asm_dir / f"{stem}.cpp"
                if p.exists():
                    found = find_cpp_setters_in_file(tail, p, full_paths=fp)
                    _cfg = ctx.cfg_cache.setdefault(stem, run_cfg_extract(str(p)))
                    for _s in found:
                        _fn = line_to_function(_cfg, _s.line)
                        if _fn:
                            _s.function = _fn
                    setters += found
        else:
            for stem in cand_sorted:
                setters += find_asm_setters_in_file(tname, stem, ctx.blueprint_dir, ctx.asm_dir)

    # on-chain reachability (discovery before-line / upstream before-descend).
    relevant = _filter_chain_scoped(
        setters, dep, ctx.chain_set, ctx.upstream_bounds,
        chain_entry_fn={h[0]: h[2] for h in (ctx.chain_hops or [])},
        blueprint_dir=ctx.blueprint_dir, asm_dir=ctx.asm_dir, cfg_cache=ctx.cfg_cache)
    # off-chain reachability (SPEC §6). Bounded for 22k.
    capped_offchain = False
    if ctx.route_engine and ctx.chain_hops:
        kept = {id(s) for s in relevant}
        checked_files: Set[str] = set()
        for s in setters:
            if id(s) in kept or s.file_stem in ctx.chain_set:
                continue
            if s.file_stem not in checked_files:
                if len(checked_files) >= ctx.max_offchain_files:
                    capped_offchain = True
                    continue
                checked_files.add(s.file_stem)
            ep = Endpoint(s.file_stem, s.language, s.function)
            try:
                if any(ctx.route_engine.reachable(Endpoint(h[0], h[1], h[2]), ep) for h in ctx.chain_hops):
                    relevant.append(s)
            except Exception:
                pass

    setter_jsons: List[Dict[str, Any]] = []
    seen_dep_setters: Set[Tuple[str, int, str]] = set()   # D7: (file,line,value) dedup
    child_depth_capped = _emit_setter_run(setter_jsons, seen_dep_setters,
                                          relevant, dep.name, depth,
                                          ctx=ctx, store=lstore, enqueue=_enqueue)

    # IN-direction boundary re-root (SPEC §6, the dual of function-output).
    if dep.language == "cpp" and not dep.is_function_output:
        pb_setters, pb_children = _param_binding_setters(
            dep, disc_stem, ctx.asm_dir, ctx.cfg_cache,
            route_engine=ctx.route_engine, chain_hops=ctx.chain_hops, chain_set=ctx.chain_set,
            reach_union=ctx.reach_union, store=lstore, cr=ctx.const_resolver,
            max_offchain_files=ctx.max_offchain_files)
        for ps in pb_setters:
            ps["setterId"] = len(setter_jsons) + 1
            setter_jsons.append(ps)
        for cd in pb_children:
            if cd.relationship == "param_rename":
                child_deps.append((dep.name, cd, depth))     # PURE RENAME — same depth
            elif depth < ctx.max_depth:
                child_deps.append((dep.name, cd, depth + 1))  # DERIVED — costs depth
            else:
                child_depth_capped = True   # derived binding child capped at depth — surfaced

    dependency = {"variableName": parent, "foundAt": dep.found_at,
                  "relevantCodeChunk": _dep_chunk(lstore, dep.found_at)}
    _m = ctx.memb.resolve(dep.name, dep.language, dep.qualified)
    dv = {
        "name": dep.name,
        "dependencies": [dependency],
        "setters": setter_jsons,
        "terminal": len(setter_jsons) == 0,
        "is_function_output": dep.is_function_output,
        "memberOf": _m["memberOf"],          # struct (cpp) / DSECT-mac (asm) it belongs to
        "counterpart": _m["counterpart"],    # cross-language twin + its struct/mac (or None)
    }
    if dep_aliases:
        dv["aliases"] = [
            {"name": a.name, "language": a.language,
             "certainty": a.certainty, "via": a.via}
            for a in dep_aliases
        ]
    if dep.qualified:
        dv["qualified"] = dep.qualified
    if dep.origin != "primary":
        dv["origin"] = dep.origin   # ancestor | descendant (path-family node)
    if dep.indirection:
        dv["indirection"] = dep.indirection  # register-indirect give-up envelope
    if child_depth_capped:
        dv["depthCapped"] = True   # dep subtree truncated at max_dep_var_depth — surfaced
    if capped_offchain:
        dv["offchainSearchCapped"] = True   # surfaced, not silent (SPEC: no silent caps)
    nrs = {
        "candidates": setters, "depth": depth, "seen": seen_dep_setters,
        "setters": setter_jsons, "dv": dv, "owner": dep.name,
        "readsites": {Path(dep.found_at["file"]).stem: _eff_line(dep.found_at.get("startLine"))},
    }
    return _NodeOut(
        key=key, dep=dep, result={"dependentVariable": dv},
        edge_seed=_edge_sig(parent, dep), node_rescope=nrs,
        child_deps=child_deps, family_parent_adds=fp_add, family_child_adds=fc_add,
        family_capped_adds=fcap_add, blocks=list(lstore.blocks.items()))


def _route_qkey(parent: Endpoint, child: Endpoint, mr: int):
    """Stable memo key for a route query (mirrors RouteEngine.routes' key)."""
    return (parent.file_stem, parent.file_type, parent.function,
            child.file_stem, child.file_type, child.function, mr)


class _RecordingRouteEngine:
    """A route-engine proxy used during the parallel route-QUERY-COLLECTION pass.

    It DELEGATES every read-only graph/metadata method to the real engine but
    RECORDS each ``reachable`` / ``routes`` query (parent, child, max_routes)
    instead of running the (expensive) cross-file route walk, returning a fixed
    stub answer. The collected query SET is independent of route ANSWERS (an
    off-chain answer only decides whether a setter is appended; a param-binding
    answer only decides whether a call site is skipped — neither gates further
    route queries), so the recorded set equals the real serial query set. The
    node OUTPUT produced in this mode is WRONG and is discarded; only the recorded
    queries are kept, deduped, then resolved in parallel and merged into the real
    engine's memo so the authoritative SERIAL compute runs warm + fast."""

    def __init__(self, real, sink: list):
        self._real = real
        self._sink = sink

    def reachable(self, parent: Endpoint, child: Endpoint) -> bool:
        self._sink.append(_route_qkey(parent, child, 1))
        return False   # stub; node output discarded — only the query is kept

    def routes(self, parent: Endpoint, child: Endpoint, *, max_routes=None):
        mr = max_routes if max_routes is not None else getattr(self._real, "max_routes", 1)
        self._sink.append(_route_qkey(parent, child, mr))
        return {"reachable": False, "route_count": 0, "routes": []}

    def __getattr__(self, name):
        # Everything else (forward_reachable_files, name_to_stems, node_types,
        # graph_edges, cpp_call_edges, _ensure_graph, …) delegates read-only.
        return getattr(self._real, name)


# Module globals for the route-collection / route-resolution passes (set on the
# parent before each fork; inherited via COW).
_RESOLVE_ENGINE = None     # the real RouteEngine, used by _resolve_route_query


def _resolve_route_query(qkey: Tuple):
    """Worker: resolve ONE route query on the real (warm, COW-inherited) engine and
    return ``(qkey, routes_dict)``. The result is deterministic; merging it into the
    parent's ``_routes_memo`` is byte-identical to the parent computing it itself."""
    eng = _RESOLVE_ENGINE
    (ps, pt, pf, cs, ct, cf, mr) = qkey
    if mr == 1:
        ok = eng.reachable(Endpoint(ps, pt, pf), Endpoint(cs, ct, cf))
        res = {"reachable": ok, "_reachable_only": True}
    else:
        res = eng.routes(Endpoint(ps, pt, pf), Endpoint(cs, ct, cf), max_routes=mr)
    return (qkey, res)


def _pool_run_chunk(fn_and_chunk):
    """Worker entry: apply ``fn`` to a contiguous chunk and return the results in
    order. Module-level (picklable). The worker's per-process caches accumulate
    across all chunks it ever runs (the pool is persistent, no recycling)."""
    fn, chunk = fn_and_chunk
    return [fn(it) for it in chunk]


def _warm_node_routes(item: Tuple[str, DepVarRef, int]) -> None:
    """ROUTE-ONLY work for one node: reproduce EXACTLY the route queries the full
    ``_compute_node`` issues (the off-chain reachability loop over the setter search,
    and the call-site reachability in param-binding), against the REAL engine so its
    ``_routes_memo`` is populated — but SKIP the family scan, the per-setter JSON build,
    the function-output re-root, and membership (none of those issue route queries).
    Used by the parallel harvest pass to do the dominant route walks in workers while
    keeping the per-node cost lean; the result is discarded, only the route memo is
    harvested. The query SET is identical to the full compute's (route answers don't
    gate further route queries), so warming the parent memo with these results is
    byte-identical."""
    ctx = _CTX
    if ctx.route_engine is None:
        return
    dep = item[1]; depth = item[2]
    if dep.is_function_output and dep.language == "cpp":
        return   # function-output nodes issue no route queries

    akey = (dep.name, dep.language)
    if akey not in ctx.alias_cache:
        ctx.alias_cache[akey] = list(resolve_aliases(dep.name, dep.language, ctx.asm_dir).aliases)
    dep_aliases = ctx.alias_cache[akey]
    disc_stem = Path(dep.found_at["file"]).stem

    targets = [(dep.name, dep.language)] + [(a.name, a.language) for a in dep_aliases]
    setters = []
    for tname, tlang in targets:
        writer_files = ctx.midx.files_for(tname, tlang)
        if ctx.reach_union is not None:
            writer_files = {f for f in writer_files if f in ctx.reach_union or f in ctx.chain_set}
        cand = set(ctx.chain_set) | writer_files
        if tlang == dep.language:
            cand.add(disc_stem)
        cand_sorted = sorted(cand)
        if tlang == "cpp":
            tail = tname.split(".")[-1].split("->")[-1]
            fp = [dep.qualified] if (tlang == dep.language and dep.qualified) else None
            for stem in cand_sorted:
                p = ctx.asm_dir / f"{stem}.cpp"
                if p.exists():
                    found = find_cpp_setters_in_file(tail, p, full_paths=fp)
                    _cfg = ctx.cfg_cache.setdefault(stem, run_cfg_extract(str(p)))
                    for _s in found:
                        _fn = line_to_function(_cfg, _s.line)
                        if _fn:
                            _s.function = _fn
                    setters += found
        else:
            for stem in cand_sorted:
                setters += find_asm_setters_in_file(tname, stem, ctx.blueprint_dir, ctx.asm_dir)

    relevant = _filter_chain_scoped(
        setters, dep, ctx.chain_set, ctx.upstream_bounds,
        chain_entry_fn={h[0]: h[2] for h in (ctx.chain_hops or [])},
        blueprint_dir=ctx.blueprint_dir, asm_dir=ctx.asm_dir, cfg_cache=ctx.cfg_cache)
    # off-chain reachability route queries (same iteration + bounding as the full compute).
    if ctx.route_engine and ctx.chain_hops:
        kept = {id(s) for s in relevant}
        checked_files: Set[str] = set()
        for s in setters:
            if id(s) in kept or s.file_stem in ctx.chain_set:
                continue
            if s.file_stem not in checked_files:
                if len(checked_files) >= ctx.max_offchain_files:
                    continue
                checked_files.add(s.file_stem)
            ep = Endpoint(s.file_stem, s.language, s.function)
            try:
                for h in ctx.chain_hops:          # mirror `any(...)`: stop at first reachable
                    if ctx.route_engine.reachable(Endpoint(h[0], h[1], h[2]), ep):
                        break
            except Exception:
                pass
    # param-binding call-site reachability route queries.
    if dep.language == "cpp" and not dep.is_function_output:
        try:
            _param_binding_setters(
                dep, disc_stem, ctx.asm_dir, ctx.cfg_cache,
                route_engine=ctx.route_engine, chain_hops=ctx.chain_hops, chain_set=ctx.chain_set,
                reach_union=ctx.reach_union, store=None, cr=ctx.const_resolver,
                max_offchain_files=ctx.max_offchain_files)
        except Exception:
            pass


def _collect_routes_chunk(chunk: List[Tuple[str, DepVarRef, int]]) -> List[Tuple]:
    """Worker: LEAN route-QUERY collection for a chunk. Runs the route-only node logic
    (``_warm_node_routes``) against a RECORDING engine (no real route walk → fast,
    parse-bound only) and returns the deduped route query keys the chunk would issue.
    Skips the family scan, per-setter JSON build, function-output re-root, and
    membership — none of which issue route queries."""
    global _CTX
    ctx = _CTX
    sink: List[Tuple] = []
    rec = _RecordingRouteEngine(ctx.route_engine, sink)
    import dataclasses as _dc
    rec_ctx = _dc.replace(ctx, route_engine=rec)
    _saved = _CTX
    _CTX = rec_ctx
    try:
        for it in chunk:
            try:
                _warm_node_routes(it)
            except Exception:
                pass
    finally:
        _CTX = _saved
    seen = set(); out = []
    for q in sink:
        if q not in seen:
            seen.add(q); out.append(q)
    return out


class _DepvarPool:
    """A persistent fork-based process pool for the dep-var route-warming passes.

    Created ONCE per ``trace_dependents`` call and reused across every batch's
    collect + resolve passes, so the fork cost is paid once and each worker's
    per-process route caches (the expensive per-CHILD caller-forest walk) accumulate
    across batches. ``map_chunked`` splits the work into one CONTIGUOUS chunk per
    worker (one task each), preserving input order; the worker init clears the
    parent's cached SQLite engines so each worker opens its own connection."""

    def __init__(self, workers: int):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        self._workers = max(1, workers)
        self._ex = None
        self._was_daemon = False
        try:
            mp_ctx = multiprocessing.get_context("fork")
        except ValueError:
            mp_ctx = None
        proc = multiprocessing.current_process()
        self._was_daemon = getattr(proc, "daemon", False)
        if self._was_daemon:
            proc.daemon = False
        kwargs = {"max_workers": self._workers, "initializer": _compute_worker_init}
        if mp_ctx is not None:
            kwargs["mp_context"] = mp_ctx
        try:
            self._ex = ProcessPoolExecutor(**kwargs)   # persistent: no recycling
        except Exception:
            self._ex = None   # fall back to serial map on any pool-creation failure

    def map_chunked(self, fn, items):
        """Apply ``fn`` across workers, one contiguous chunk each, INPUT-ORDERED."""
        items = list(items)
        n = len(items)
        if self._ex is None or n == 0:
            return [fn(it) for it in items]
        import math
        workers = min(self._workers, n)
        sz = max(1, math.ceil(n / workers))
        bounds = [(lo, min(lo + sz, n)) for lo in range(0, n, sz)]
        results: List = [None] * n
        fut_to_bounds = {
            self._ex.submit(_pool_run_chunk, (fn, items[lo:hi])): (lo, hi)
            for (lo, hi) in bounds
        }
        for fut, (lo, hi) in fut_to_bounds.items():
            chunk_res = fut.result()
            for off, r in enumerate(chunk_res):
                results[lo + off] = r
        return results

    def for_each_chunk(self, chunk_fn, items):
        """Split ``items`` into one contiguous chunk per worker and call
        ``chunk_fn(chunk)`` once per chunk (chunk-granular, not item-granular).
        Returns the list of per-chunk results (order unimportant to the caller)."""
        items = list(items)
        n = len(items)
        if self._ex is None or n == 0:
            return [chunk_fn(items)] if items else []
        import math
        workers = min(self._workers, n)
        sz = max(1, math.ceil(n / workers))
        bounds = [(lo, min(lo + sz, n)) for lo in range(0, n, sz)]
        futs = [self._ex.submit(chunk_fn, items[lo:hi]) for (lo, hi) in bounds]
        return [f.result() for f in futs]

    def shutdown(self):
        import multiprocessing
        try:
            if self._ex is not None:
                self._ex.shutdown(wait=True)
        finally:
            self._ex = None
            if self._was_daemon:
                multiprocessing.current_process().daemon = True


# Module global holding the read-only context for ``_compute_node``. Set on the
# parent immediately before each compute step (and inherited by forked workers via
# COW). The serial path reads it in-process; the parallel path reads each worker's
# fork-inherited copy. Never mutated by the merge.
_CTX: Optional[_NodeCtx] = None


def _set_ctx(ctx: _NodeCtx) -> None:
    global _CTX
    _CTX = ctx


def _compute_worker_init() -> None:
    """ProcessPoolExecutor worker initializer for the dep-var compute pool.

    Clears the parent's cached SQLite engines so each worker opens its OWN
    connection lazily — forking with the parent's open connections is unsafe.
    Also applies the standard niced / SIGINT-ignoring worker setup."""
    try:
        import api.index_db.engine as _eng
        _eng._engines.clear()
    except Exception:
        pass
    # Mirror parallel._worker_init (nice + ignore SIGINT) without importing it as
    # the executor's initializer (we need the engine-clear too).
    try:
        os.nice(5)
    except OSError:
        pass
    try:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass


def trace_dependents(
    seed_deps: List[Tuple[str, DepVarRef]],   # (parent_var, dep) at level 1
    chain_files: List[str],
    *,
    blueprint_dir: Path,
    asm_dir: Path,
    store: CodeBlockStore,
    upstream_bounds: Optional[Dict[str, int]] = None,
    route_engine=None,
    chain_hops: Optional[List[Tuple[str, str, Optional[str]]]] = None,
    max_depth: int = 2,
    const_resolver=None,                 # D5: enum/EQU-DC resolver for dep-var setter metadata
    max_offchain_files: int = _MAX_OFFCHAIN_FILES,
    asm_max_levels: int = 16,
    max_descendants: int = 50,           # path-family: cap distinct sub-field deps per node
    workers: Optional[int] = None,       # None => VBT_DEPVAR_WORKERS/default; >1 warms route memo in parallel
    progress: bool = False,              # per-dep STDERR progress logs (#6/#7)
) -> List[Dict[str, Any]]:
    blueprint_dir = Path(blueprint_dir); asm_dir = Path(asm_dir)
    # Defensive normalization (Issue 1): callers pass file-stem chains, but the trace
    # output's setter "chain" now carries C++ function labels (``stem::fn``). Strip any
    # ``::fn`` suffix so every file-level check here (``F in chain_files``, upstream
    # bounds, off-chain pruning, param-binding scoping) stays stem-based. Idempotent on
    # a bare stem, so a clean stem list is unaffected.
    chain_set = {str(f).split("::", 1)[0] for f in chain_files}
    upstream_bounds = upstream_bounds or {}
    cfg_cache: Dict[str, List[Dict]] = {}
    const_cache: Dict[str, Set[str]] = {}
    bp_cache: Dict[str, Tuple[str, Dict]] = {}
    alias_cache: Dict[Tuple[str, str], List] = {}   # (dep.name, dep.language) -> high-certainty aliases
    # D-F5: both keyed on (tail.upper(), qualified) so genuinely-different same-tail
    # fields no longer collide; true duplicates (same tail+qualified) still merge.
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    memo: Set[Tuple[str, str]] = set()
    # DAG in-edges: the dep-var node is deduped on (tail.upper(), qualified), but it
    # is READ by many parents at many sites. Accumulate the distinct (parent, file,
    # line) read-sites per node so the dependencies list reflects the real DAG, not
    # just the first discoverer. Keyed identically to `results`.
    edge_seen: Dict[Tuple[str, str], Set[Tuple[str, str, int]]] = {}
    # Per-node SOUND multi-read-site setter union. A node's setters are discovery-scoped
    # to the read-site (the before-line rule), but the node is SHARED across read-sites,
    # so the first discoverer's scope must not be the only one applied. We cache each
    # node's (read-site-independent) candidate setters + its emitted-setter dedup set;
    # when the node is re-encountered at a MORE-PERMISSIVE read-site (a later line in the
    # discovery file, or a read in another file — which keeps this file's setters by
    # reachability) we re-filter and union in any newly-live setter. Keyed like `results`.
    node_rescope: Dict[Tuple[str, str], Dict[str, Any]] = {}
    midx = get_modifier_index(blueprint_dir, asm_dir)   # for function-output re-root
    memb = get_membership_resolver(blueprint_dir, asm_dir)  # per-var struct/mac + counterpart

    # Metadata-only forward-reachable file set (union over all chain hops), used to
    # prune off-chain dep-var candidate files BEFORE opening them (the scale prune).
    # Superset of every file a verified route(hop, setter) could traverse, so a
    # setter we'd keep can never be pruned. None ⇒ no route engine ⇒ no prune.
    reach_union: Optional[Set[str]] = None
    if route_engine is not None and chain_hops:
        reach_union = set()
        for h in chain_hops:
            try:
                reach_union |= route_engine.forward_reachable_files(Endpoint(h[0], h[1], h[2]))
            except Exception:
                pass

    # Read-only per-node compute context (shared by serial + parallel; fork-COW
    # for the parallel path). The mutable caches inside are per-process; in the
    # serial path they persist across nodes (speed only, never output).
    ctx = _NodeCtx(
        blueprint_dir=blueprint_dir, asm_dir=asm_dir, chain_set=chain_set,
        upstream_bounds=upstream_bounds, route_engine=route_engine, chain_hops=chain_hops,
        max_depth=max_depth, const_resolver=const_resolver,
        max_offchain_files=max_offchain_files, asm_max_levels=asm_max_levels,
        max_descendants=max_descendants, midx=midx, memb=memb, reach_union=reach_union,
        cfg_cache=cfg_cache, const_cache=const_cache, bp_cache=bp_cache, alias_cache=alias_cache,
    )

    # ---- path-family links (member-path ancestors/descendants) ----------------- #
    # A "primary" member-path dep (a.b.c) gets, as SEPARATE linked nodes, its container
    # ANCESTORS (a.b, a — string prefixes, no struct/type needed) and its written
    # sub-field DESCENDANTS (a.b.c.* — discovered from observed writes). Both are kept
    # because a write to an ancestor sets a.b.c WHOLLY (aggregate/container copy) and a
    # write to a descendant sets part of it. Derived nodes carry expand_family=False so
    # they recurse their own SETTER deps but never re-expand the family (no siblings, no
    # runaway). Links are recorded here and attached to the result nodes at the end.
    family_parents: Dict[Tuple[str, str], Set[str]] = {}   # depkey -> {parent qualified}
    family_children: Dict[Tuple[str, str], Set[str]] = {}  # depkey -> {child qualified}
    family_capped: Set[Tuple[str, str]] = set()

    def _add_parent_edge(key: Tuple[str, str], parent: str, dep: DepVarRef) -> None:
        """Append a deduped (parent, foundAt) read-site edge to an already-built node.
        Replaces the old memo-gate no-op that silently dropped every parent after the
        first — so a dep read in N places now lists all N distinct read-sites."""
        node = results.get(key)
        if node is None:
            return
        sig = _edge_sig(parent, dep)
        seen = edge_seen.setdefault(key, set())
        if sig not in seen:
            seen.add(sig)
            node["dependentVariable"]["dependencies"].append(
                {"variableName": parent, "foundAt": dep.found_at,
                 "relevantCodeChunk": _dep_chunk(store, dep.found_at)})
        # This read-site may make additional candidate setters live that the first
        # discoverer's before-line scope dropped — union them in (sound; deduped).
        _rescope_node(key, dep)

    def _rescope_node(key, dep):
        """Re-scope a re-encountered node for a NEW read-site: union in any candidate
        setter now live that the first discoverer's before-line scope had dropped. Sound
        by monotonicity (a later/same-file read keeps a SUPERSET; a read in another file
        keeps this file's setters by reachability). Bounded — a dominated read-site (same
        file, line <= the most-permissive seen) can add nothing, so it skips.

        Runs in the MERGE (sequential) — it enqueues to the shared ``q`` (memo-filtered)
        and writes to the shared ``store`` directly, exactly as the serial code did."""
        nc = node_rescope.get(key)
        if nc is None:
            return                       # function-output / unbuilt node → nothing to re-scope
        f = Path(dep.found_at["file"]).stem
        l = _eff_line(dep.found_at.get("startLine"))
        rs = nc["readsites"]
        if f in rs and l <= rs[f]:
            return
        rs[f] = max(rs.get(f, 0), l)
        relevant = _filter_chain_scoped(
            nc["candidates"], dep, chain_set, upstream_bounds,
            chain_entry_fn={h[0]: h[2] for h in (chain_hops or [])},
            blueprint_dir=blueprint_dir, asm_dir=asm_dir, cfg_cache=cfg_cache)

        def _merge_enqueue(owner_name, cd, cdepth):
            if _dep_key(cd) not in memo:
                q.append((owner_name, cd, cdepth))

        if _emit_setter_run(nc["setters"], nc["seen"], relevant, nc["owner"], nc["depth"],
                            ctx=ctx, store=store, enqueue=_merge_enqueue):
            nc["dv"]["depthCapped"] = True
        nc["dv"]["terminal"] = len(nc["setters"]) == 0   # may flip terminal → non-terminal

    def _resolve_funcoutput_deffile(dep: DepVarRef) -> None:
        """Function-output identity: resolve the DEFINING file (the C5 call-site-def
        rule) BEFORE keying, so distinct same-named functions across files don't merge
        into one node. Mutates ``dep.def_file`` in place (idempotent)."""
        if dep.is_function_output and dep.language == "cpp" and not dep.qualified and not dep.def_file:
            _ft = dep.name.split("::")[-1].split(".")[-1].split("->")[-1]
            _defs = sorted(midx.files_defining_function(_ft))
            _cs = Path(dep.found_at["file"]).stem if dep.found_at.get("file") else ""
            dep.def_file = _cs if _cs in _defs else (_defs[0] if _defs else "")

    def _merge_node(out: _NodeOut, parent: str) -> None:
        """Apply a computed node's side effects to the shared state, in pop order.
        First occurrence of a key computes + lands here; the family-link adds, the
        result/edge_seen/node_rescope writes, the local store blocks (merged in
        registration order), then the child enqueue under the SAME memo filter the
        serial code used (filter AT MERGE TIME — a child already in memo is NOT
        enqueued, which suppresses a duplicate parent edge)."""
        key = out.key
        memo.add(key)
        for (dk, pq) in out.family_parent_adds:
            family_parents.setdefault(dk, set()).add(pq)
        for (dk, cq) in out.family_child_adds:
            family_children.setdefault(dk, set()).add(cq)
        for dk in out.family_capped_adds:
            family_capped.add(dk)
        results[key] = out.result
        edge_seen[key] = {out.edge_seed}
        if out.node_rescope is not None:
            node_rescope[key] = out.node_rescope
        # Merge the local store blocks in registration order. Ids are content-
        # addressed (cpp:{stem}:{fn} / asm:{stem}:{block_id}); first-writer wins,
        # which preserves the serial registration order.
        for bid, block in out.blocks:
            if bid not in store.blocks:
                store.blocks[bid] = block
        # Enqueue children under the serial memo filter (applied AT MERGE TIME).
        for (owner, cd, cdepth) in out.child_deps:
            if _dep_key(cd) not in memo:
                q.append((owner, cd, cdepth))

    # Worker count: VBT_DEPVAR_WORKERS (default 1 = serial, no pool). >1 → each BFS
    # level's dominant cost (the cross-file route walks) is warmed in parallel via a
    # fork pool before the level's COMPUTE; the COMPUTE + MERGE are ALWAYS serial and
    # in pop order. Serial and parallel share the exact same compute + merge code and
    # differ only in whether the route memo was pre-warmed in parallel — so the output
    # is byte-identical regardless of worker count (the warmed routes are the same
    # deterministic values the serial path would compute on demand).
    try:
        raw_workers = workers if workers is not None else os.environ.get("VBT_DEPVAR_WORKERS", "1")
        _workers = int(raw_workers or "1")
    except (TypeError, ValueError):
        _workers = 1
    _workers = max(1, _workers)

    _dbg = os.environ.get("VBT_DEPVAR_DEBUG_TIMING")

    # ---- persistent worker pool (parallel route warming) ----------------------- #
    # One pool for the WHOLE trace, reused across every batch's collect + resolve
    # passes. A persistent pool (a) pays the fork cost ONCE (not per batch), and (b)
    # lets each worker's per-process route caches (_pfl_cache / _edge_cache /
    # _reach_cache, keyed by the expensive per-CHILD caller-forest walk) ACCUMULATE
    # across batches — later batches reuse a worker's earlier route walks. We pre-warm
    # the engine's read-only graph payload / name→stem map / per-hop forward-reach in
    # the PARENT before forking, so all of that is inherited via copy-on-write.
    _pool = None
    if _workers > 1 and route_engine is not None:
        # Pre-warm the engine's read-only sub-caches AND set the module globals the
        # workers read (``_CTX`` for collect, ``_RESOLVE_ENGINE`` for resolve) BEFORE
        # the pool's workers fork — a fork-based pool inherits the parent's globals AT
        # FORK TIME (it forks lazily on first submit), and globals reassigned in the
        # parent afterward do NOT propagate to already-live workers. So both must be
        # set here, before _DepvarPool is constructed / first used.
        global _RESOLVE_ENGINE
        _RESOLVE_ENGINE = route_engine
        _set_ctx(ctx)
        try:
            route_engine._ensure_graph()
            route_engine.name_to_stems()
            try:
                route_engine.preload_fn_reachability(
                    [Endpoint(h[0], h[1], h[2]) for h in (chain_hops or [])]
                )
            except Exception:
                pass
            for h in (chain_hops or []):
                try:
                    route_engine.forward_reachable_files(Endpoint(h[0], h[1], h[2]))
                except Exception:
                    pass
        except Exception:
            pass
        _pool = _DepvarPool(_workers)

    def _warm_route_memo(items: List[Tuple[str, DepVarRef, int]]) -> None:
        """Parallel route-memo warming for a batch (the byte-identity-preserving
        speedup). The dep-var compute is ~90% cross-file route walks whose results are
        PURE + shared via the engine's ``_routes_memo`` (77% of route calls are memo
        hits in serial), so parallelizing per-NODE scatters that shared work and erases
        the win. Instead we (1) COLLECT every route query the batch will issue
        (parallel, parse-only against a recording engine; the query SET is independent
        of route ANSWERS), (2) RESOLVE the unique ones in parallel on the warm
        COW-inherited engine — sorted by CHILD so each child's caller forest is walked
        once per worker, (3) merge the results into the parent's memo. The authoritative
        compute then runs SERIALLY with a warm memo so every route check is a memo hit.
        Output is byte-identical to pure serial while the route walks ran in parallel."""
        if _pool is None or not items:
            return
        _set_ctx(ctx)
        memo = route_engine._routes_memo
        # (1) LEAN collect of the batch's route queries (parse-only, recording engine).
        if _dbg:
            import time as _t; _tc = _t.perf_counter()
        per_chunk = _pool.for_each_chunk(_collect_routes_chunk, items)
        uniq: List[Tuple] = []
        seen_q: Set[Tuple] = set()
        def _already_warm(qk: Tuple) -> bool:
            if qk[6] == 1 and hasattr(route_engine, "_reachable_memo"):
                rk = (qk[0], qk[1], qk[2], qk[3], qk[4], qk[5],
                      getattr(route_engine, "max_len", 16))
                return rk in route_engine._reachable_memo
            return qk in memo
        for qlist in per_chunk:
            for qk in qlist:
                if qk in seen_q or _already_warm(qk):
                    continue
                seen_q.add(qk)
                uniq.append(qk)
        if _dbg:
            _LOG.warning("[warm collect] n=%d  uniq_q=%d  %.2fs",
                         len(items), len(uniq), _t.perf_counter() - _tc)
        if not uniq:
            return
        # (2) resolve the unique queries in parallel, CHILD-grouped so each child's
        # caller forest is walked once per worker (no cross-worker redundancy).
        uniq.sort(key=lambda qk: (qk[3], qk[4], qk[5] or "", qk[0], qk[2] or "", qk[6]))
        if _dbg:
            _tr = _t.perf_counter()
        resolved = _pool.map_chunked(_resolve_route_query, uniq)
        for (qk, res) in resolved:
            if res.get("_reachable_only") and hasattr(route_engine, "_reachable_memo"):
                rk = (qk[0], qk[1], qk[2], qk[3], qk[4], qk[5],
                      getattr(route_engine, "max_len", 16))
                route_engine._reachable_memo[rk] = bool(res.get("reachable"))
            elif qk not in memo:
                memo[qk] = res
        if _dbg:
            _LOG.warning("[warm resolve] uniq_q=%d  %.2fs", len(uniq),
                         _t.perf_counter() - _tr)

    def _compute_batch(items: List[Tuple[str, DepVarRef, int]]) -> List[_NodeOut]:
        """Compute each unique-new node in ``items``. When VBT_DEPVAR_WORKERS>1 a
        parallel route-memo warming pre-pass runs first (see ``_warm_route_memo``);
        the authoritative compute is ALWAYS serial + in pop order, so output is
        byte-identical to the pure-serial path regardless of worker count."""
        _set_ctx(ctx)
        # Hang-locating marker: a hung dep level shows its depth + node count as the last INFO line.
        if items and _LOG.isEnabledFor(logging.INFO):
            _LOG.info("  dep level [d%d]: computing %d new node(s)...", items[0][2], len(items))
        if _pool is not None and len(items) > 1:
            _warm_route_memo(items)
        if _dbg:
            import time as _t; _t0 = _t.perf_counter()
            r = [_compute_node(it) for it in items]
            _LOG.warning("[batch compute serial] n=%d  %.2fs", len(items),
                         _t.perf_counter() - _t0)
            return r
        return [_compute_node(it) for it in items]

    q: deque = deque((p, d, 1) for p, d in seed_deps)
    _dv_done = 0
    try:
        while q:
            # ---- one BFS LEVEL = one batch -------------------------------------- #
            # Drain the ENTIRE current queue (in order); clear q for this level's
            # children. Then determine the UNIQUE NEW nodes to compute (memo-first-
            # wins): a node is new if its key is not in memo AND not seen earlier in
            # THIS batch. Compute those, then MERGE every batch item in pop order —
            # re-encounters route through _add_parent_edge.
            batch: List[Tuple[str, DepVarRef, int]] = list(q)
            q.clear()
            for (_p, _d, _depth) in batch:
                _resolve_funcoutput_deffile(_d)   # resolve def_file BEFORE keying
            compute_items: List[Tuple[str, DepVarRef, int]] = []
            batch_seen: Set[Tuple[str, str]] = set()
            for (p, d, depth) in batch:
                k = _dep_key(d)
                if k in memo or k in batch_seen:
                    continue
                batch_seen.add(k)
                compute_items.append((p, d, depth))
            outs = _compute_batch(compute_items)
            out_by_key: Dict[Tuple[str, str], _NodeOut] = {o.key: o for o in outs}

            for (parent, dep, depth) in batch:
                key = _dep_key(dep)
                if key in memo:
                    _add_parent_edge(key, parent, dep)   # DAG in-edge from a re-encountered parent
                    continue
                out = out_by_key.get(key)
                if out is None:
                    # Already merged earlier in THIS batch (a within-batch duplicate) →
                    # it's now in memo; route as a re-encounter. (Defensive; the memo
                    # check above normally catches it once merged.)
                    _add_parent_edge(key, parent, dep)
                    continue
                _merge_node(out, parent)
                _dv_done += 1
                if _LOG.isEnabledFor(logging.INFO):
                    _LOG.info("  dep var #%d done [d%d]: %s  (%d queued)",
                              _dv_done, depth, dep.name, len(q))
    finally:
        if _pool is not None:
            _pool.shutdown()


    # ---- attach path-family links (member-path ancestors/descendants) ----------- #
    # All referenced ancestor/descendant qualified paths were enqueued + processed, so
    # they exist as result nodes; wire the parent/child references by qualified path.
    for k, node in results.items():
        dv = node["dependentVariable"]
        if k in family_parents:
            dv["pathParents"] = sorted(family_parents[k])
        if k in family_children:
            dv["pathChildren"] = sorted(family_children[k])
        if k in family_capped:
            dv["descendantsCapped"] = True   # distinct sub-field cap hit — surfaced
    return list(results.values())
