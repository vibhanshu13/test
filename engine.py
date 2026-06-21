"""Root-variable orchestrator (v2).

For a (variable, language, chain_prefix): find setters → keep those reachable from
the chain tail → reconstruct the FULL call path tail→setter (intra-file + cross-file)
→ for each distinct path build a (setter, chain) entry with its set value, every
call-site guard along the path, the setter-local guard, and the upstream chain-prefix
guards → harvest dependent variables from that full condition set → assemble JSON.

Design notes (root-cause fixes from the Round-1 audits):
  * route_finder is reused READ-ONLY for *reachability* + the cross-file file
    sequence, but its per-hop ``conditions`` are an ACCUMULATION of every lineage
    edge in the caller scope (they include sibling guards that close before the call
    line). We therefore NEVER use them for C++: we recompute each call-site guard
    precisely with ``collect_cpp_conditions`` at the exact call line (rule 4: the new
    logic lives in our layer, route_finder is untouched).
  * route_finder does not surface intra-file inter-function calls (a same-file route
    is ``hops:[]``). We enumerate those ourselves (``intra_call_paths``) — this is the
    C6 case (R2) and produces R3's two-path / R6's two-call-site OR-alternative tuples.
  * Dependent variables are extracted from the ENTIRE condition set of a tuple
    (prefix + path + local), so an intra-file gate variable (D9: ``gateVar``) becomes
    a dependent variable.

Pure-ASM / mixed-language setters keep the proven legacy mechanism (route_finder hop
guards resolved through the polarity table) — the precise C++ machinery only applies
when the whole downstream route is C++.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backward_traversal.route_finder import Endpoint

from vbt.reach.route import RouteEngine, intra_call_paths
from vbt.reach.cpp_routes import enumerate_cpp_paths
from vbt.reach.fn_attr import line_to_function
from vbt.setters.cpp_setters import find_cpp_setters_in_file
from vbt.setters.asm_setters import find_asm_setters_in_file
from vbt.conditions.cpp_conditions import collect_cpp_conditions
from vbt.conditions.asm_conditions import collect_asm_conditions, _resolve_predicate, find_trigger_line
from vbt.cpp_frontend.wrapper import run_cfg_extract
from vbt.interfaces import Condition, Location, SetterSite
from vbt.output.codeblocks import CodeBlockStore
from vbt.output.assembler import build_root_output
from vbt.depvars.extract import extract_dep_vars, AsmIndirectContext
from vbt.depvars.recurse import trace_dependents
from vbt.resolve.name_resolver import resolve as resolve_aliases
from vbt.resolve.const_resolver import get_const_resolver
from vbt.resolve.membership import get_membership_resolver
from vbt.precompute.modifier_index import get_modifier_index
from vbt.precompute.call_graph import ensure_call_graph
from backward_traversal.utils.blueprint_utils import (
    resolve_asm_blueprint, load_json, collect_constant_symbols,
)


@dataclass
class Hop:
    stem: str
    file_type: str            # "asm" | "cpp"
    function: Optional[str] = None


# --------------------------------------------------------------------------- #
# per-phase progress logging (#6/#7 — "looks hung"). STDERR ONLY: stdout carries
# the JSON result, so a stray log line there would corrupt it. Gated by the
# ``progress`` param; cheap (no-ops at the default WARNING level).
# --------------------------------------------------------------------------- #
# Attach the stderr handler to the shared "vbt" parent so sibling modules
# (vbt.depvars) emit through it too; ``progress`` raises the level on this root.
_VBT_LOG = logging.getLogger("vbt")
if not _VBT_LOG.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[vbt %(relativeCreated)8.0fms] %(message)s"))
    _VBT_LOG.addHandler(_h)
    _VBT_LOG.propagate = False
_LOG = logging.getLogger("vbt.engine")

# Build tag — logged once at trace start (with --progress) so a run SELF-IDENTIFIES the live code.
# Bumped on any behaviour-affecting change; if this line is ABSENT from a run, the code is pre-fix/stale.
_BUILD_TAG = "walk: bridge edge+flow index + lazy-attach + fn_graph_adj + setter-map skip  [2026-06-21a]"

# Stem -> concise reason for CFG extraction failures seen in this process. The
# function graph treats those stems as empty instead of failing the whole KB, and
# precompute records the reasons for operator visibility.
_CFG_SKIP_REASONS: Dict[str, str] = {}


class _Phase:
    """Context manager that logs START/DONE + elapsed for one phase (no-op when
    logging is below INFO, so the default path pays nothing)."""

    def __init__(self, name: str):
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        if _LOG.isEnabledFor(logging.INFO):
            self.t0 = time.perf_counter()
            _LOG.info("START %s", self.name)
        return self

    def __exit__(self, *exc):
        if _LOG.isEnabledFor(logging.INFO):
            _LOG.info("DONE  %s (%.3fs)", self.name, time.perf_counter() - self.t0)
        return False


def _log(msg: str, *args) -> None:
    if _LOG.isEnabledFor(logging.INFO):
        _LOG.info(msg, *args)


def _dbg(msg: str, *args) -> None:
    """Verbose hang-locating trace: logged BEFORE each expensive operation so the LAST line emitted
    before a hang names the exact stuck call. DEBUG-gated ⇒ no-op (and byte-identical) unless the
    client raises the level to DEBUG when chasing a hang."""
    if _LOG.isEnabledFor(logging.DEBUG):
        _LOG.debug(msg, *args)


def _cfg_skip_reason(stem: str) -> Optional[str]:
    return _CFG_SKIP_REASONS.get(stem)


def _cfg_skip_reason_from_exc(exc: Exception) -> str:
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    for ln in lines:
        if "fatal error:" in ln or "source file not found:" in ln:
            return ln[:500]
    return (lines[0] if lines else type(exc).__name__)[:500]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _loc(file: str, start: int, end: int) -> Dict[str, Any]:
    return {"file": file, "startLine": start, "endLine": end}


def _cfg_for(stem: str, cfg_cache: Dict[str, List[Dict]], asm_dir: Path) -> List[Dict]:
    if stem not in cfg_cache:
        try:
            cfg_cache[stem] = run_cfg_extract(str(asm_dir / f"{stem}.cpp"))
        except (FileNotFoundError, RuntimeError) as exc:
            reason = _cfg_skip_reason_from_exc(exc)
            _CFG_SKIP_REASONS[stem] = reason
            _LOG.warning("_cfg_for: skipping %s - %s", stem, reason)
            cfg_cache[stem] = []
        else:
            _CFG_SKIP_REASONS.pop(stem, None)
    return cfg_cache[stem]


def _fn_lines(f: Dict) -> List[int]:
    return [s.get("line") for b in f.get("cfg_blocks", []) for s in b.get("stmts", []) if s.get("line")]


def _fn_name_matches(name: str, q: str) -> bool:
    """Exact or qualified-boundary match (B4) — a bare ``endswith`` would let
    ``myrxTail`` match ``rxTail`` and pick the wrong function/span."""
    return bool(name) and bool(q) and (name == q or name.endswith("::" + q))


def _cpp_fn_span(functions: List[Dict], fn_name: str) -> Tuple[int, int]:
    for f in functions:
        if _fn_name_matches(f.get("function", ""), fn_name):
            lines = _fn_lines(f)
            if lines:
                return min(lines), max(lines)
    return 0, 0


def _cpp_fn_block(store: CodeBlockStore, stem: str, fn: Optional[str],
                  cfg_cache: Dict[str, List[Dict]], asm_dir: Path) -> Optional[str]:
    """Register (dedup) + return the codeBlocks id for a C++ function so a condition
    located in it can reference its code by ``blockId`` (D5). ASM stems (single-entry
    modules on a mixed route) have no .cpp body → no C++ function block to register."""
    if not fn or not (asm_dir / f"{stem}.cpp").exists():
        return None
    lo, hi = _cpp_fn_span(_cfg_for(stem, cfg_cache, asm_dir), fn)
    return store.cpp_function(stem, fn, lo, hi)


def _local_fn_names(cfg_fns: List[Dict]) -> set:
    return {f.get("function") for f in cfg_fns if f.get("function")}


def _fn_containing_line(cfg_fns: List[Dict], line: int) -> Optional[str]:
    """Function owning ``line`` by nearest-preceding START (vbt/reach/fn_attr).

    Was span-containment (``min(stmt) <= line <= max(stmt)``), which silently
    returns None for any line past a function body truncated at a collapsed switch
    (e.g. a cross-file call at 1093 in a body cfg-recovered only to 1057). The
    START-based resolver is robust to truncation and identical for clean bodies."""
    return line_to_function(cfg_fns, line)


def _mk_cond(text: str, file: str, line: int, *, end_line: Optional[int] = None,
             raw_test: Optional[str] = None, raw_branch: Optional[str] = None) -> Condition:
    return Condition(order=0, condition=text, block_id="",
                     location=Location(file, line, end_line or line),
                     raw_test=raw_test, raw_branch=raw_branch)


def _asm_bp(stem: str, blueprint_dir: Path, cache: Dict[str, Dict]) -> Dict:
    """Load + cache an ASM/MAC blueprint by stem (for compare-line resolution of hop
    guards). Returns ``{}`` on any failure (the caller falls back to the branch line)."""
    bp = cache.get(stem)
    if bp is None:
        try:
            bp = load_json(str(resolve_asm_blueprint(stem, blueprint_dir)))
        except Exception:
            bp = {}
        cache[stem] = bp
    return bp


def _hop_cond_lines(stem: str, raw_test: Optional[str], branch_line: Optional[int],
                    call_line: int, blueprint_dir: Path, cache: Dict[str, Dict]
                    ) -> Tuple[int, int]:
    """``(compare_line, decision_line)`` for an ASM hop guard. route_finder pins a hop
    guard at the conditional-BRANCH line (``cond["line"]``) — or, before this, the engine
    pinned it at the CALL line — but the guard's data (the tested field + comparand) lives
    at the COMPARE instruction. Resolve that compare line from the blueprint
    (``find_trigger_line``), falling back to the branch line, then the call line. Never
    fabricates a line."""
    decision = branch_line or call_line
    compare = None
    if raw_test:
        compare = find_trigger_line(_asm_bp(stem, blueprint_dir, cache), raw_test,
                                    decision or 10 ** 9)
    compare = compare or decision or call_line
    return compare, (decision or compare)


def _asm_block_at(bp: Dict, line: int) -> Optional[Tuple[str, int, int]]:
    """``(block_id, start_line, end_line)`` of the SMALLEST ASM block containing
    ``line`` (so the file-wide container block is never chosen). ``None`` if no block
    covers it."""
    best: Optional[Tuple[str, int, int, int]] = None
    for b in (bp.get("blocks") or []):
        s, e = b.get("start_line"), b.get("end_line")
        if not (s and e):
            continue
        s, e = int(s), int(e)
        if s <= line <= e:
            span = e - s
            if best is None or span < best[3]:
                best = (str(b.get("id")), s, e, span)
    return (best[0], best[1], best[2]) if best else None


def _asm_hop_block(stem: str, line: int, blueprint_dir: Path,
                   cache: Dict[str, Dict], store: CodeBlockStore) -> Optional[str]:
    """Register + return the codeBlocks id for the ASM block containing ``line`` — the
    block where a chain hop's guard AND its descend call (e.g. ``ENTRC DW73``) live. This
    puts the descend call SITE into the trace: its source lands in ``codeBlocks`` and the
    hop guard gets a real ``blockId`` (was null), so a consumer can slice the via's call
    line to show the actual transfer instruction instead of only the guard. ``None`` when
    no block covers the line (the guard then keeps its existing line refs)."""
    if not line:
        return None
    found = _asm_block_at(_asm_bp(stem, blueprint_dir, cache), line)
    if not found:
        return None
    bid, s, e = found
    return store.asm_block(stem, bid, s, e)


_CONST_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_#$@]*(?:::[A-Za-z_][A-Za-z0-9_]*)?")


def _resolve_consts(text: str, cr) -> Dict[str, str]:
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


def _resolve_hop_condition(cond: Any, call_block: Optional[str] = None
                           ) -> Optional[Tuple[str, Optional[str], Optional[str], Optional[int]]]:
    """Normalize a route_finder hop guard into ``(text, raw_test, raw_branch, branch_line)``.

    C++ hops arrive as strings; ASM hops as ``{line, test, branch}`` dicts resolved
    through the polarity table. **Polarity (the GAP-029 fix):** a normal *call-site guard*
    branches AWAY from the call — reaching the call is the FALLTHROUGH, so the predicate is
    negated (``taken=False``). But a *block_gate* recovered from a predecessor branches
    INTO the call's block (route_finder only emits it when its target == the call's block),
    so reaching the block means the branch was TAKEN (``taken=True``). We decide per the
    branch's TARGET vs ``call_block`` — which yields the correct ``taken=False`` for the
    away-branching call-site guard too. (Without ``call_block`` we conservatively assume
    fallthrough, the prior behavior.) ``branch_line`` = the route_finder BRANCH line; the
    caller re-points the condition at the actual COMPARE instruction line."""
    if isinstance(cond, str):
        t = " ".join(cond.split())
        return (t, None, None, None) if t else None
    if isinstance(cond, dict):
        if cond.get("note"):
            return None
        test, branch = cond.get("test"), cond.get("branch")
        if test and branch:
            bline = cond.get("line")
            parts = str(branch).split(None, 1)
            tgt = parts[1].split(",")[0].strip().upper() if len(parts) > 1 else ""
            taken = bool(call_block) and tgt == str(call_block).upper()
            return (_resolve_predicate(str(test), str(branch), taken=taken),
                    str(test), str(branch), int(bline) if bline else None)
        return None
    return None


# --------------------------------------------------------------------------- #
# C++ precise path reconstruction (our layer; route_finder gives only file structure)
# --------------------------------------------------------------------------- #
def _hop_callee(route: RouteEngine, from_stem: str, line: int, next_local: set) -> Optional[str]:
    """The function in the NEXT file entered by the cross-file call at ``line`` in
    ``from_stem`` — the target of that call_graph edge that is defined in the next
    file. Returns None when no such edge exists (B2): an arbitrary fallback target
    poisoned the next segment's entry and silently dropped all its intra-file gates."""
    for s, t, ln in route.cpp_call_edges(from_stem):
        if ln == line and t in next_local:
            return t
    return None


def _reconstruct_cpp_paths(route_one: Dict[str, Any], tail_ep: Endpoint, setter: SetterSite,
                           route: RouteEngine, cfg_cache: Dict[str, List[Dict]],
                           asm_dir: Path, *, max_paths: int = 64, max_call_depth: int = 16
                           ) -> List[List[Dict[str, Any]]]:
    """Reconstruct EVERY concrete call path tail→setter for one route_finder route.

    Returns ``(paths, incomplete)`` — ``paths`` is a list of paths, each an ordered
    list of edge dicts ``{caller_stem, caller_fn, line, callee_stem, callee_fn, cross}``;
    ``incomplete`` is True if any segment's enumeration was truncated (cap hit) or a
    route_finder-reachable segment didn't resolve a concrete intra-file path. Within each file
    we enumerate the intra-file inter-function paths (``intra_call_paths``); files are
    stitched by the route's cross-file hops. The cartesian product over per-file
    segments yields one path per concrete execution route (R3 two paths, R6 two sites).
    """
    files = list(route_one.get("files") or [setter.file_stem])
    if files[-1] != setter.file_stem:
        files.append(setter.file_stem)
    hops = route_one.get("hops") or []
    hop_by_from = {}
    for h in hops:
        hop_by_from.setdefault(h.get("from_file"), h)

    # per-file segment = (stem, entry_fn, exit_fn, hop_out_or_None)
    segments: List[Tuple[str, Optional[str], Optional[str], Optional[Dict[str, Any]]]] = []
    prev_callee: Optional[str] = None
    for i, stem in enumerate(files):
        cfg = _cfg_for(stem, cfg_cache, asm_dir)
        entry = tail_ep.function if i == 0 else prev_callee
        if i < len(files) - 1:
            h = hop_by_from.get(stem)
            line = int(h["line"]) if h and h.get("line") else 0
            caller_fn = _fn_containing_line(cfg, line) or entry
            next_stem = files[i + 1]
            next_local = _local_fn_names(_cfg_for(next_stem, cfg_cache, asm_dir))
            callee_fn = _hop_callee(route, stem, line, next_local)
            segments.append((stem, entry, caller_fn,
                             {"line": line, "caller_fn": caller_fn,
                              "callee_stem": next_stem, "callee_fn": callee_fn}))
            prev_callee = callee_fn
        else:
            segments.append((stem, entry, setter.function, None))

    paths: List[List[Dict[str, Any]]] = [[]]
    incomplete = False
    for (stem, entry, exit_fn, hop_out) in segments:
        cfg = _cfg_for(stem, cfg_cache, asm_dir)
        local = _local_fn_names(cfg)
        ipaths, trunc = intra_call_paths(route.cpp_call_edges(stem), local, entry or "", exit_fn or "",
                                         max_paths=max_paths, max_depth=max_call_depth)
        if trunc:                       # B1/B3: path enumeration hit a cap — surfaced, not silent
            incomplete = True
        if not ipaths:                  # B2: a segment that route_finder said was reachable
            ipaths = [[]]               # didn't resolve a concrete intra-file path — keep the
            incomplete = True           # setter visible but mark its gates as not-fully-known
        new_paths: List[List[Dict[str, Any]]] = []
        for base in paths:
            for ip in ipaths:
                seg_edges = [{"caller_stem": stem, "caller_fn": c, "line": ln,
                              "callee_stem": stem, "callee_fn": cl, "cross": False}
                             for (c, ln, cl) in ip]
                edges = base + seg_edges
                if hop_out and hop_out.get("callee_fn"):
                    edges = edges + [{"caller_stem": stem, "caller_fn": hop_out["caller_fn"],
                                      "line": hop_out["line"], "callee_stem": hop_out["callee_stem"],
                                      "callee_fn": hop_out["callee_fn"], "cross": True}]
                new_paths.append(edges)
        paths = new_paths or [[]]
    return paths, incomplete


def _edge_guard(edge: Dict[str, Any], cfg_cache: Dict[str, List[Dict]], asm_dir: Path) -> List[Condition]:
    """Precise enclosing guard at one C++ call site (NOT route_finder's accumulation).

    Only a C++ caller carries an intra-function guard here: an ASM module is a single-
    entry node (no intra-function call site), so a mixed-route edge whose caller is an
    ASM module contributes no call-site guard (its setter-local guards come from the
    ASM condition path). Skip it cleanly rather than cfg-extract a non-existent .cpp."""
    cpp_path = asm_dir / f"{edge['caller_stem']}.cpp"
    if not cpp_path.exists():
        return []
    fns = _cfg_for(edge["caller_stem"], cfg_cache, asm_dir)
    return collect_cpp_conditions(cpp_path, int(edge["line"]), edge["caller_fn"], functions=fns)


def _edge_via(edge: Dict[str, Any]) -> str:
    if edge["cross"]:
        return f"{edge['caller_stem']}.{edge['caller_fn']}→{edge['callee_stem']}.{edge['callee_fn']}@{edge['line']}"
    return f"{edge['caller_fn']}→{edge['callee_fn']}@{edge['line']}"


# --------------------------------------------------------------------------- #
# prefix (upstream chain) guards — precise for C++ callers, route_finder for ASM
# --------------------------------------------------------------------------- #
def _prefix_guards(chain_prefix: List[Hop], route: RouteEngine,
                   cfg_cache: Dict[str, List[Dict]], asm_dir: Path, store: CodeBlockStore,
                   blueprint_dir: Path, asm_bp_cache: Dict[str, Dict]
                   ) -> Tuple[List[Tuple[Condition, str, Optional[str]]], Dict[str, int]]:
    """Guards to descend the GIVEN chain prefix (rxentry→rxmid→rxtail), plus each
    upstream file's descend bound (``upstream_bounds`` for the dep-var before-descend
    rule). Returns ``(Condition, via, blockId)`` triples (D5 — the caller function's
    code is registered so the guard's source is retrievable by ``blockId``).

    B6: when the caller reaches the next hop via SEVERAL call sites with different
    guards, the descend is an OR of those sites — `(g1) || (g2)` — not the single
    earliest site; if ANY site is unconditional the descend is unconditional (no
    guard). The descend bound is the LATEST site (most permissive — a dep-var setter
    between two sites is live if the path used the later site, so don't drop it)."""
    out: List[Tuple[Condition, str, Optional[str]]] = []
    bounds: Dict[str, int] = {}
    for c_h, e_h in zip(chain_prefix, chain_prefix[1:]):
        via = f"{c_h.stem}→{e_h.stem}"
        if c_h.file_type == "cpp":
            cpp_path = asm_dir / f"{c_h.stem}.cpp"
            fns = _cfg_for(c_h.stem, cfg_cache, asm_dir)
            blk = _cpp_fn_block(store, c_h.stem, c_h.function, cfg_cache, asm_dir)
            sites = sorted({ln for s, t, ln in route.cpp_call_edges(c_h.stem)
                            if (not c_h.function or s == c_h.function)
                            and ((not e_h.function) or _fn_name_matches(t, e_h.function))})
            if not sites:
                continue
            bounds[c_h.stem] = max(sites)
            groups = [(ln, collect_cpp_conditions(cpp_path, ln, c_h.function, functions=fns))
                      for ln in sites]
            if any(len(conds) == 0 for _ln, conds in groups):
                continue                            # an unconditional site → descend is unconditional
            if len(groups) == 1:
                for cc in groups[0][1]:
                    out.append((cc, via, blk))
            else:                                   # B6: OR of (AND of each site's guards)
                parts = []
                for _ln, conds in groups:
                    inner = " && ".join(c.condition for c in conds)
                    parts.append(f"({inner})" if len(conds) > 1 else inner)
                out.append((_mk_cond(" || ".join(parts), f"{c_h.stem}.cpp", groups[0][0]), via, blk))
        else:
            hop_lines: List[int] = []
            # SCALE FIX (b): the descend-guard for an EXPLICIT adjacent chain hop is the LOCAL
            # call-site guard at c_h's transfer to e_h. route.routes(...) computes it by walking
            # e_h's ENTIRE reverse caller closure (O(corpus) — the 10k-ASM caller-walk that hangs);
            # but the loop below already keeps ONLY the from_file==c_h.stem hop, i.e. it only ever
            # uses c_h's own direct call site. So read that directly via the caller bridge (O(1) —
            # c_h's blueprint only). Byte-identical: empirically the bridge's direct c_h→e_h site +
            # conditions == route.routes(max_routes=1)'s single hop for the adjacent case. For a
            # NON-adjacent hop the bridge finds no direct site → fall back to the full route walk.
            try:
                from backward_traversal.runner.chainless_caller_walker import (
                    _bridge_for as _bf, _attach_call_site_conditions as _acc)
                _br = _bf(c_h.file_type, e_h.file_type)
                _sites = (_br(c_h.stem, e_h.stem, blueprint_dir, asm_dir) or []) if _br else []
            except Exception:
                _sites = []
            if _sites:
                # route.routes(max_routes=1) returns ONE route ⇒ ONE c_h hop (the first direct
                # site, in the same bridge order the route DFS uses) — mirror that with _sites[0].
                _acc(_sites, c_h.stem, c_h.file_type, blueprint_dir, asm_dir, {})
                _s0 = _sites[0]
                hr = {"routes": [{"hops": [{"from_file": c_h.stem, "to_file": e_h.stem,
                                            "line": _s0.get("line"),
                                            "conditions": _s0.get("conditions") or []}]}]}
            else:
                hr = route.routes(Endpoint(c_h.stem, c_h.file_type, c_h.function),
                                  Endpoint(e_h.stem, e_h.file_type, e_h.function), max_routes=1)
            for rr in hr.get("routes", []):
                for hop in rr.get("hops", []):
                    if hop.get("from_file") != c_h.stem:
                        continue
                    call_line = int(hop.get("line") or 0)
                    if call_line:
                        hop_lines.append(call_line)
                    # via carries the descend CALL line (the ENTRC into the next file),
                    # mirroring the C++ `caller→callee@line` form so a consumer can locate
                    # the transfer instruction; the block holding it is registered below.
                    hvia = f"{via}@{call_line}" if call_line else via
                    # the call's own block — a block_gate guard whose branch targets it
                    # was TAKEN to reach it (GAP-029 polarity, #4).
                    _cb = _asm_block_at(_asm_bp(c_h.stem, blueprint_dir, asm_bp_cache), call_line) if call_line else None
                    call_block = _cb[0] if _cb else None
                    for cond in (hop.get("conditions") or []):
                        rec = _resolve_hop_condition(cond, call_block=call_block)
                        if rec:
                            text, rt, rb, bline = rec
                            # anchor at the COMPARE instruction (not the branch/call line)
                            cline, dline = _hop_cond_lines(c_h.stem, rt, bline, call_line,
                                                           blueprint_dir, asm_bp_cache)
                            # register the ASM block (guards + the descend ENTRC) so the
                            # call site's source is in the trace and blockId is non-null.
                            blk = _asm_hop_block(c_h.stem, cline, blueprint_dir, asm_bp_cache, store)
                            out.append((_mk_cond(text, f"{c_h.stem}.asm", cline, end_line=dline,
                                                 raw_test=rt, raw_branch=rb), hvia, blk))
            if hop_lines:
                bounds[c_h.stem] = max(hop_lines)   # B6: latest site = permissive bound
    return out, bounds


# --------------------------------------------------------------------------- #
# condition finalisation (dedup + conjunct subsumption + resolved constants)
# --------------------------------------------------------------------------- #
def _finalize_conditions(raw: List[Tuple[Condition, Optional[str], Optional[str]]], cr
                         ) -> Tuple[List[Dict[str, Any]], List[Condition]]:
    """``raw`` = list of (Condition, blockId, via). Returns (output condition dicts,
    kept Condition objects). Exact-duplicate predicates are dropped; a bare conjunct
    already covered by a compound ``A && B`` in the same tuple is dropped; negations
    are never dropped. Constant tokens get a resolved-value map."""
    seen: set = set()
    deduped: List[Tuple[Condition, Optional[str], Optional[str]]] = []
    for c, blk, via in raw:
        key = c.condition
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((c, blk, via))

    def _norm(s: str) -> str:
        return "".join(str(s).split())

    def _conjunct(s: str) -> str:
        """Normalized form of a single conjunct: whitespace-stripped, outer parens
        peeled (so ``(a->b != 0)`` and a bare ``a->b != 0`` compare equal)."""
        t = _norm(s)
        while len(t) >= 2 and t[0] == "(" and t[-1] == ")":
            t = t[1:-1]
        return t

    # C-3: a standalone guard is subsumed only if it equals a WHOLE conjunct of some
    # compound ``A && B`` — token-boundary, not substring. The old `n in comp` substring
    # test wrongly dropped a real guard whose text was a suffix of a qualified path in an
    # unrelated compound (e.g. bare `ccr != 0` ⊂ `addresses->ccr != 0`) — pervasive at
    # 22k scale where short field names are suffixes of member paths.
    compound_conjuncts: set = set()
    for c, _, _ in deduped:
        if "&&" in c.condition:
            for part in c.condition.split("&&"):
                compound_conjuncts.add(_conjunct(part))

    out: List[Dict[str, Any]] = []
    kept: List[Condition] = []
    for c, blk, via in deduped:
        if "&&" not in c.condition and _conjunct(c.condition) in compound_conjuncts:
            continue
        d: Dict[str, Any] = {
            "order": len(out) + 1, "condition": c.condition, "blockId": blk,
            "location": _loc(c.location.file, c.location.start_line, c.location.end_line),
        }
        if via:
            d["via"] = via
        # ASM: `condition` is the engine's NORMALIZED predicate. Surface the verbatim
        # source instructions so a consumer can show real ASM (location.startLine = the
        # compare, endLine = the branch/decision line). C++ conditions slice real source
        # already and carry no raw_test.
        if c.raw_test:
            d["asmTest"] = c.raw_test
            if c.raw_branch:
                d["asmBranch"] = c.raw_branch
            d["decisionLine"] = c.location.end_line
        rc = _resolve_consts(c.condition, cr)
        if rc:
            d["resolvedConstants"] = rc
        out.append(d)
        kept.append(c)
    return out, kept


# --------------------------------------------------------------------------- #
# value-flow feasibility prune (switch-scrutinee constant-clobber)
# --------------------------------------------------------------------------- #
# An equality guard against a SCOPED enum constant, e.g.
#   plasticAuth.process.cidDb.cidDbSelect == PaMessage::ManualExpirationDateOnly
# group(1) = scrutinee member-path (LHS), group(2) = `Scope::Name` (the required const).
_EQ_ENUM_GUARD_RE = re.compile(
    r"^\s*([A-Za-z_][\w.\->:\[\]]*?)\s*==\s*([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)\s*$")


def _enum_tail(name: str) -> str:
    """Trailing component of a (possibly scoped) enum constant — ``A::B::C`` -> ``C``."""
    return name.split("::")[-1].strip()


def _route_value_flow_infeasible(
    conditions: List[Dict[str, Any]],
    route_fns: List[Tuple[str, str]],
    cfg_cache: Dict[str, List[Dict]],
    asm_dir: Path,
    setter_memo: Optional[Dict[Any, Any]] = None,
    cond_memo: Optional[Dict[Any, bool]] = None,
) -> bool:
    """SOUND value-flow prune: is THIS route provably unable to reach the setter?

    A route is infeasible when its own collected guards require a switch scrutinee
    ``V == REQUIRED`` (a scoped enum constant) yet the LAST route function that
    assigns ``V`` does so UNCONDITIONALLY to a *different* scoped enum constant —
    so on this exact route ``V`` is clobbered to that other value before the setter's
    enclosing ``case`` runs, and the ``case`` can never be taken.

    SOUNDNESS (only provably-infeasible routes are pruned):
      * We consider ONLY assignments that are UNCONDITIONAL within their function
        (``collect_cpp_conditions`` at the site returns no enclosing guard) and whose
        RHS is a scoped enum constant. A conditional re-set (``if(..) V = Other;`` —
        the dw710300 ``processManualEntryExpirationDateOnly`` shape) is IGNORED, so a
        feasible route is never pruned.
      * We use the LAST such assignment in route order: a later route function may
        legitimately re-set ``V`` back to the required const, in which case the route
        is feasible and we must NOT prune.
      * If ``V`` is never unconditionally set on the route, or the last unconditional
        set assigns the REQUIRED const, we do NOT prune.

    ``route_fns`` is the ordered ``(callee_stem, callee_fn)`` list of the C++ functions
    ENTERED on this route (setter file/fn last). For a non-pure-C++ route (ASM/mixed/
    unverified) it is empty and this is a definite no-op (returns False).
    """
    if not route_fns:
        return False
    # Gather the route's equality-guards on scoped enum constants: scrutinee -> required.
    required: Dict[str, str] = {}
    for c in conditions:
        # An equality guard inside a negation (``!(V == K)``) does NOT require V==K, so a
        # leading `!(` (polarity False) disqualifies it — only a positive equality forces V.
        if c.get("polarity") is False:
            continue
        m = _EQ_ENUM_GUARD_RE.match(str(c.get("condition") or ""))
        if not m:
            continue
        scrut, const = m.group(1).strip(), m.group(2).strip()
        required.setdefault(scrut, const)        # first positive equality per scrutinee
    if not required:
        return False
    # Trace-scoped memoization (the prune's only real cost): the two parsing calls
    # below — find_cpp_setters_in_file + collect_cpp_conditions — are otherwise re-run
    # per route on the SAME files, so a switch-heavy setter re-parses each file dozens
    # of times. File content is constant within a trace, so caching is sound (identical
    # results, parse once). Caller passes shared dicts; default to local for safety.
    if setter_memo is None:
        setter_memo = {}
    if cond_memo is None:
        cond_memo = {}

    for scrut, req_const in required.items():
        tail = scrut.split(".")[-1].split("->")[-1].split("::")[-1].strip().split("[")[0]
        scrut_full = [scrut]
        # Walk the route functions in order; remember the LAST UNCONDITIONAL scoped-enum
        # assignment to V (its assigned const). Conditional assignments are skipped.
        last_unconditional_const: Optional[str] = None
        for stem, fn in route_fns:
            cpp = asm_dir / f"{stem}.cpp"
            if not cpp.exists():
                continue
            skey = (stem, tail, scrut)
            sites = setter_memo.get(skey)
            if sites is None:
                sites = find_cpp_setters_in_file(tail, cpp, full_paths=scrut_full)
                setter_memo[skey] = sites
            fns = None
            for site in sites:
                if site.function != fn or site.is_declaration:
                    continue
                val = (site.value or "").strip()
                # RHS must be a SCOPED enum constant (``Scope::Name``) — a non-const RHS
                # (another variable, an expression) tells us nothing about feasibility.
                if "::" not in val or val.endswith(")") or " " in val:
                    continue
                # UNCONDITIONAL within its function? (no enclosing guard at the site) —
                # memoized per (file, line, fn); the other parsing hot-path.
                ckey = (stem, site.line, fn)
                is_cond = cond_memo.get(ckey)
                if is_cond is None:
                    if fns is None:
                        fns = _cfg_for(stem, cfg_cache, asm_dir)
                    is_cond = bool(collect_cpp_conditions(cpp, site.line, fn, functions=fns))
                    cond_memo[ckey] = is_cond
                if is_cond:
                    continue                     # conditional set → cannot force V → skip
                last_unconditional_const = val
        # No unconditional set, or it sets the REQUIRED const → route NOT proven infeasible.
        if last_unconditional_const is None:
            continue
        if (last_unconditional_const == req_const
                or _enum_tail(last_unconditional_const) == _enum_tail(req_const)):
            continue
        # The route forces V to a DIFFERENT scoped enum constant than the case requires.
        return True
    return False


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def trace_root_variable(
    variable: str,
    language: str,
    chain_prefix: List[Hop],
    *,
    blueprint_dir: Path,
    asm_dir: Path,
    graph_file: Path,
    job_id: Optional[str] = None,                      # T1: DB-backed precompute key; None = file/compute path (behavior-neutral)
    candidate_stems: Optional[List[str]] = None,
    candidate_functions: Optional[List[str]] = None,  # restrict setters to these functions/blocks (#1)
    home_hint: Optional[str] = None,                   # scope cross-language alias resolution (#1)
    disable_dependents: bool = False,                  # root setters only, no dep-var tree (#8)
    progress: bool = False,                            # per-phase STDERR progress logs (#6/#7)
    # --- tunable limits (defaults = the engine's scale-safety bounds) ------------
    max_dep_var_depth: int = 1,      # dep-var recursion depth (#5: was 2; deep recursion now opt-in)
    max_paths: int = 64,             # distinct intra-file call paths per setter
    max_call_depth: int = 16,        # edges deep when enumerating intra-file paths
    max_routes: int = 200,           # cross-file routes tail→setter (route_finder)
    max_route_len: int = 16,         # cross-file route hop-length (route_finder)
    max_offchain_files: int = 100,   # off-chain candidate files route-checked per dep var
    asm_max_levels: int = 16,        # N-level ASM backward-CFG condition collection
) -> Dict[str, Any]:
    if progress:
        # VBT_HANG_DEBUG=1 raises to DEBUG (the before-each-op trace that names the EXACT stuck call,
        # plus per-dep-node markers) instead of INFO progress. Set it on the client when chasing a hang.
        _VBT_LOG.setLevel(logging.DEBUG if os.environ.get("VBT_HANG_DEBUG") else logging.INFO)
        _LOG.info("VBT BUILD: %s", _BUILD_TAG)   # self-identify the live code (absent ⇒ stale checkout)
    _t_total = time.perf_counter()
    blueprint_dir = Path(blueprint_dir); asm_dir = Path(asm_dir); graph_file = Path(graph_file)
    # T12: a trace with a job_id TRUSTS the DB — O(1) artifact loads, no per-trace file scans.
    # job_id None → load_only OFF → the unchanged file/compute path. Set this FIRST — BEFORE the
    # trace-cache lookup below, which is the very first DB open (`get_engine`) — so load-only is in
    # effect for that first open and the engine can skip the O(DB-size) gvl-backfill COUNT scans VBT
    # never uses (the cold-startup killer on an 18GB DB: ~232s → seconds). Byte-identical: load_only
    # only toggles freshness-by-presence vs source-hashing; the trace output is unaffected.
    try:
        from vbt.precompute import db_artifacts as _DA
        _DA.set_load_only(bool(job_id))
    except Exception:
        pass
    # §16 #1: whole-trace result cache. Return a byte-identical cached output for the SAME inputs
    # (skips ALL computation). job_id-gated; precompute_db clears it on rebuild so a stale result
    # is never served. The trace is deterministic ⇒ a cached result == a fresh compute.
    _trace_sig = None
    if job_id:
        try:
            from vbt.precompute import trace_cache as _TC
            _trace_sig = _TC.signature(
                variable, language, chain_prefix, blueprint_dir, asm_dir, graph_file,
                candidate_stems=candidate_stems, candidate_functions=candidate_functions,
                home_hint=home_hint, disable_dependents=disable_dependents,
                max_dep_var_depth=max_dep_var_depth, max_paths=max_paths,
                max_call_depth=max_call_depth, max_routes=max_routes,
                max_route_len=max_route_len, max_offchain_files=max_offchain_files,
                asm_max_levels=asm_max_levels)
            _cached = _TC.load(job_id, _trace_sig)
            if _cached is not None:
                if progress:
                    _VBT_LOG.info("trace result served from cache (§16 #1) — instant")
                return _cached
        except Exception:
            _trace_sig = None
    # D3: refresh the call graph from the LIBRARY entry point too (not only precompute_all) —
    # cheap mtime no-op when fresh, but at 22k the blueprint-mtime GLOB is O(files). Skip it
    # when this job already has a precomputed file_call_graph in the DB (precompute ensured it;
    # RouteEngine reads the payload from the DB blob).
    _ecg_skip = False
    if job_id:
        try:
            from vbt.precompute import db_artifacts as _DA
            from vbt.precompute.graph_db import FILE_CALL_GRAPH_ARTIFACT
            _ecg_skip = _DA.is_load_only() and _DA.manifest_present(job_id, FILE_CALL_GRAPH_ARTIFACT)
        except Exception:
            _ecg_skip = False
    if not _ecg_skip:
        ensure_call_graph(blueprint_dir, asm_dir, out_path=graph_file)
    store = CodeBlockStore(asm_dir)
    route = RouteEngine(blueprint_dir, graph_file, asm_dir, max_routes=max_routes, max_len=max_route_len, job_id=job_id)
    # T5: DB-backed route name_to_stems preload — fill RouteEngine._name_to_stems from index.db
    # (or build via the blueprint sweep + persist) so the per-trace corpus blueprint sweep is
    # skipped. No-op when job_id is None (sweep runs as before). Best-effort.
    if job_id:
        try:
            from vbt.precompute.graph_db import preload_route_graph, preload_cpp_call_edges
            preload_route_graph(route, job_id, blueprint_dir, graph_file)
            # T6: LOAD-ONLY — fill route._edge_cache from a precomputed cpp_call_edges artifact
            # if present; no-op (per-stem lazy build, unchanged) on a cold/stale DB.
            preload_cpp_call_edges(route, job_id, blueprint_dir, asm_dir)
            # §16 #2-full: pre-fill the route memo from prior traces (cross-trace reuse of the
            # caller-walk). route.routes is unchanged (it serves _routes_memo first) ⇒ byte-
            # identical; a cold/absent blob is a no-op (routes compute live as before).
            from vbt.precompute.route_cache_db import preload_route_cache
            preload_route_cache(route, job_id)
        except Exception:
            pass
    tail = chain_prefix[-1]
    tail_ep = Endpoint(tail.stem, tail.file_type, tail.function)
    prefix_stems = [h.stem for h in chain_prefix]
    chain_stem_set = set(prefix_stems)
    cfg_cache: Dict[str, List[Dict]] = {}
    asm_bp_cache: Dict[str, Dict] = {}     # hop-guard compare-line resolution (read-only)

    # T2: DB-backed resolver preload. When job_id is set, fill the name-alias + membership
    # caches from index.db (or build+persist on a cold/stale DB) so the resolvers below do
    # NO source globbing/parsing. No-op when job_id is None (CLI/parity) -> resolvers build
    # lazily from source, byte-identical. Best-effort; never raises into the trace.
    if job_id:
        try:
            from vbt.precompute.resolvers_db import preload_resolvers
            preload_resolvers(job_id, blueprint_dir, asm_dir)
        except Exception:
            pass

    # T7: ASM-blueprint DB override (serve ASM/MAC blueprints from index.db, not files). Always
    # CLEAR first so a job_id=None trace never inherits a prior trace's override; install only
    # when job_id is set AND a fresh artifact exists (else load_json reads disk, byte-identical).
    try:
        from backward_traversal.utils.blueprint_utils import (
            clear_blueprint_override, clear_source_override)
        from vbt.precompute.cfg_db import clear_cfg_db
        clear_blueprint_override()
        clear_source_override()
        clear_cfg_db()                     # clear any prior trace's hooks
        from vbt.setters.asm_setters import set_asm_setter_map_job
        from vbt.setters.cpp_setters import set_cpp_setter_map_job, set_cpp_file_writes_job
        from vbt.precompute.fn_facts_db import set_fn_facts_job
        set_asm_setter_map_job(None)       # clear any prior trace's setter-map hooks
        set_cpp_setter_map_job(None)
        set_cpp_file_writes_job(None)
        set_fn_facts_job(None)
        if job_id:
            from vbt.precompute.asm_db import install_asm_blueprint_override
            from vbt.precompute.cfg_db import install_cfg_db
            from vbt.precompute.source_db import install_source_override
            install_asm_blueprint_override(job_id, blueprint_dir)
            install_cfg_db(job_id)         # T8: serve/persist cfg via index.db
            install_source_override(job_id, asm_dir)   # T9: serve source text via index.db
            set_asm_setter_map_job(job_id)  # Phase 1: serve ASM setters from the precomputed map
            set_cpp_setter_map_job(job_id)  # Phase 2: serve C++ setters from the precomputed map
            set_cpp_file_writes_job(job_id) # serve _file_writes (path-family search) from the precomputed map
            set_fn_facts_job(job_id)        # 3A: serve fn-graph (starts, local) from the precomputed map
    except Exception:
        pass

    # ---- 1. find setters across the HIGH-CERTAINTY alias-set (SPEC §7) ----
    with _Phase("alias resolution"):
        alias_set = resolve_aliases(variable, language, asm_dir, home_hint=home_hint)
    targets = [(variable, language)] + [(a.name, a.language) for a in alias_set.aliases]
    midx = get_modifier_index(blueprint_dir, asm_dir)
    cr = get_const_resolver(blueprint_dir, asm_dir)

    # ---- chain-first reachability prune (THE explosion fix; RESULT-PRESERVING) ----
    # Before opening ANY candidate file, restrict the writer-file set to those
    # FORWARD-REACHABLE from the chain tail using call-graph METADATA only (no file
    # opens). The reachable set is a guaranteed SUPERSET of every file a verified
    # route(tail, setter) could traverse, so a setter we'd keep can never be pruned;
    # chain/tail files are always kept regardless. At 22k a hot variable's
    # ``files_for`` is huge — opening them all is the explosion this avoids.
    with _Phase("forward-reachable file set (tail prune)"):
        reachable_files = route.forward_reachable_files(tail_ep)
    setters: List[SetterSite] = []
    n_cand_before = 0
    n_cand_after = 0
    with _Phase("root setter search"):
        for name, lang in targets:
            own = (lang == language)
            raw_stems = (candidate_stems if (own and candidate_stems)
                         else sorted(midx.files_for(name, lang)))
            n_cand_before += len(raw_stems)
            # keep only reachable writer files, plus always the chain/tail files.
            pruned = sorted((set(raw_stems) & reachable_files) | (set(raw_stems) & chain_stem_set))
            n_cand_after += len(pruned)
            if lang == "cpp":
                tail_name = name.split(".")[-1].split("->")[-1]
                # Scope to the struct path when the caller gave a qualified variable (mirror
                # the dep-var search). A bare-tail root (e.g. ``softCardValue``) is unscoped —
                # a length-1 target matches any tail by design, so this is a no-op there.
                root_fp = [name] if ("." in name or "->" in name) else None
                for stem in pruned:
                    p = asm_dir / f"{stem}.cpp"
                    if p.exists():
                        setters += find_cpp_setters_in_file(tail_name, p, full_paths=root_fp)
            else:
                for stem in pruned:
                    setters += find_asm_setters_in_file(name, stem, blueprint_dir, asm_dir)
        # #1: restrict to setters whose function/block matches a caller-supplied hint.
        if candidate_functions:
            setters = [s for s in setters
                       if any(_fn_name_matches(s.function or s.block_id or "", q)
                              for q in candidate_functions)]
        _log("root setter search: candidate files %d -> %d after prune, %d setters found",
             n_cand_before, n_cand_after, len(setters))

    # ---- prefix (upstream) guards + descend bounds (computed once) ----
    with _Phase("prefix (chain) guards"):
        prefix_conds, upstream_bounds = _prefix_guards(chain_prefix, route, cfg_cache, asm_dir,
                                                        store, blueprint_dir, asm_bp_cache)

    # ---- 2. reachability filter + 3. per-(setter, path) entries ----
    entries: List[Dict[str, Any]] = []
    seed_deps: List = []                       # (parent_var, DepVarRef)
    seen_tuples: set = set()                   # (setter file:line, chain, sorted cond texts) dedup

    _phase_routes = _Phase("per-route entry building").__enter__()
    # Shared, trace-scoped memo for the value-flow prune (parse each scrutinee/file once
    # for the whole trace, not per route — the prune's hot path).
    _vf_setter_memo: Dict[Any, Any] = {}
    _vf_cond_memo: Dict[Any, bool] = {}
    _n_setters = len(setters)
    for _si, s in enumerate(setters):
        _t_setter = time.perf_counter()
        # Re-attribute a C++ setter's enclosing function via the authoritative cfg-START
        # resolver. tree-sitter's _enclosing_function (header-independent, no preprocessor)
        # mis-parses the z/TPF DB-API idiom ``if MACRO(x){ ... }`` (e.g. ``if DF_OK(filePtr){``)
        # as a *function definition* named after the macro, so a setter inside the block is
        # attributed to ``DF_OK`` rather than the real function — and the reachability check
        # below then drops it as unreachable. The cfg resolver keys off real function starts,
        # so it is immune (returns ``updateDr409x`` here). No-op for cleanly-parsed setters.
        if s.language == "cpp":
            _fn = _fn_containing_line(_cfg_for(s.file_stem, cfg_cache, asm_dir), s.line)
            if _fn:
                s.function = _fn
        child = Endpoint(s.file_stem, s.language, s.function)
        # Hang-locating trace: log BEFORE each expensive sub-call so the LAST line printed before a
        # hang names the exact stuck operation (route.routes' caller-walk vs the path-enumeration DFS).
        _dbg("    setter %d/%d %s::%s  route.routes(%s)...", _si + 1, _n_setters,
             s.file_stem, s.function or "?", s.language)
        _t0 = time.perf_counter()
        r = route.routes(tail_ep, child)
        _t_routes = time.perf_counter() - _t0
        reachable = bool(r.get("reachable"))
        unverified = r.get("unverified", [])
        # ROBUST UNIFIED route enumeration (route-undercount fix, vbt/reach/cpp_routes.py):
        # enumerate function-level paths tail→setter directly over the UNIFIED graph
        # (C++ functions + single-entry ASM module nodes). This recovers the cross-file
        # routes the legacy caller-walk silently drops — multi-callee files + (global)
        # caller mis-attribution — for BOTH a C++ setter (softCardValue's routes through
        # dw710300/dw710500/dw710600/dw710800 into selectCidDbRecord15) AND an ASM setter
        # reached from a C++ tail (e.g. dw710000 → … → nb81: legacy returns 3 files / the
        # unified enumerator the 8 function-reachable files). Gated to a C++ tail: a
        # pure-ASM tail stays on the legacy path (ASM modules are single-entry — the
        # legacy router is already complete for them). When it finds paths it is
        # authoritative (a superset of the legacy reconstruction); when it finds none we
        # fall back to the legacy path below (never losing the legacy coverage).
        cpp_paths: List[List[Dict[str, Any]]] = []
        cpp_paths_trunc = False
        _t_enum = 0.0
        if tail.file_type == "cpp":
            _dbg("    setter %d/%d %s::%s  enumerate_cpp_paths...", _si + 1, _n_setters,
                 s.file_stem, s.function or "?")
            _t0 = time.perf_counter()
            cpp_paths, cpp_paths_trunc = enumerate_cpp_paths(
                route, cfg_cache, asm_dir, tail_ep, s,
                max_paths=max_paths, max_depth=max_call_depth)
            _t_enum = time.perf_counter() - _t0
            if cpp_paths:
                reachable = True          # a verified function path exists → reachable
        use_robust_cpp = bool(cpp_paths)
        # Progress/diagnostic heartbeat (INFO-gated ⇒ no-op + byte-identical by default): the per-route
        # loop is the phase that hung for hours at 22k. Surface steady progress + flag any straggler
        # setter so a slow run is never mistaken for a hang and the offending setter is identifiable.
        if _LOG.isEnabledFor(logging.INFO):
            _dt_setter = time.perf_counter() - _t_setter
            if _dt_setter > 1.0 or (_si + 1) % 100 == 0 or _si + 1 == _n_setters:
                _log("  per-route %d/%d %s::%s  %.2fs (routes %.2fs, enum %.2fs; %d paths%s)",
                     _si + 1, _n_setters, s.file_stem, s.function or "?", _dt_setter,
                     _t_routes, _t_enum, len(cpp_paths), ", TRUNC" if cpp_paths_trunc else "")
        # B7: a setter reachable ONLY through indirect/virtual/cycle (unverified)
        # callers has no verified route — surface it with a flag rather than dropping
        # it silently (SPEC §9). A genuinely unreachable setter (no caller at all,
        # e.g. R4) has no unverified callers either, so it is still dropped.
        via_unverified = (not reachable) and bool(unverified)
        if not reachable and not via_unverified:
            continue
        routes = r.get("routes") or [{"files": [s.file_stem], "hops": []}]
        routes_capped = bool(r.get("routes_capped"))   # B3: route enumeration hit max_routes

        # ---- 3a. setter-local guard + code block (shared by both branches) ----
        if s.language == "cpp":
            fns = _cfg_for(s.file_stem, cfg_cache, asm_dir)
            local = collect_cpp_conditions(asm_dir / f"{s.file_stem}.cpp", s.line, s.function, functions=fns)
            span = _cpp_fn_span(fns, s.function or "")
            blk = store.cpp_function(s.file_stem, s.function or "", span[0], span[1])
            setter_loc = _loc(f"{s.file_stem}.cpp", s.line, s.line)
            consts = None
            asm_ind_ctx = None
        else:
            bp_path = str(resolve_asm_blueprint(s.file_stem, blueprint_dir))
            bp = load_json(bp_path)
            # Register-indirect setter-source resolution context (SPEC §6 register-drop
            # gap): lets extract_dep_vars resolve a ``0(R1)`` source to its named field.
            asm_ind_ctx = AsmIndirectContext(bp_data=bp, bp_path=bp_path, asm_dir=asm_dir,
                                             route_engine=route)
            local = collect_asm_conditions(bp, bp_path, s.block_id, s.line, f"{s.file_stem}.asm",
                                           max_levels=asm_max_levels)
            consts = collect_constant_symbols(bp, asm_dir / f"{s.file_stem}.asm")
            blk_obj = next((b for b in (bp.get("blocks") or []) if str(b.get("id")) == s.block_id), {})
            blk = store.asm_block(s.file_stem, s.block_id,
                                  int(blk_obj.get("start_line") or s.line), int(blk_obj.get("end_line") or s.line))
            setter_loc = _loc(f"{s.file_stem}.asm", s.line, s.line)

        # ---- build the candidate (setter, path) tuples, PER ROUTE ----
        # B5/D1: decide pure-C++ vs legacy PER ROUTE (not once for the whole setter),
        # so a wholly-C++ route still gets precise intra-file gates even when another
        # route to the same setter is mixed. The legacy branch (ASM / mixed route)
        # now also resolves the ROUTE'S OWN discovered hop guards (it previously kept
        # only prefix + setter-local, silently dropping every cross-file call-site
        # guard such as the aa71→nb81 `TM L7DSAF` test).
        # 4th tuple element = ordered (callee_stem, callee_fn) of the C++ functions entered
        # on this route (setter fn last) — threaded so the value-flow prune below can resolve
        # each route function's own assignments to the switch scrutinee. Empty for an
        # ASM/mixed/unverified route (the prune is then a definite no-op).
        built: List[Tuple[List[str], List[Tuple[Condition, Optional[str], Optional[str]]],
                          bool, List[Tuple[str, str]]]] = []
        if via_unverified:
            # No verified downstream path — emit one tuple with prefix + setter-local
            # guards only (the indirect path's call-site guards are unknown).
            raw = ([(c, blk_p, via) for (c, via, blk_p) in prefix_conds]
                   + [(c, blk, None) for c in local])
            built.append(([s.file_stem], raw, False, []))
            routes = []
        for rt in routes:
            rt_files = rt.get("files") or [s.file_stem]
            rt_pure_cpp = (s.language == "cpp" and tail.file_type == "cpp"
                           and all((asm_dir / f"{f}.cpp").exists() for f in rt_files))
            if use_robust_cpp:
                # the robust unified function-level enumerator (below) is authoritative
                # for EVERY route from a C++ tail (pure-C++ routes AND an ASM-setter-from-
                # C++-tail route): it already enumerates the complete simple-path set with
                # per-edge guards. Skip the legacy file-simple reconstruction to avoid
                # duplicating (and under-counting) the same paths. When it finds NOTHING
                # (use_robust_cpp False) the legacy branches below are the fallback.
                continue
            if rt_pure_cpp:
                paths, incomplete = _reconstruct_cpp_paths(rt, tail_ep, s, route, cfg_cache, asm_dir,
                                                           max_paths=max_paths, max_call_depth=max_call_depth)
                for path in paths:
                    raw: List[Tuple[Condition, Optional[str], Optional[str]]] = []
                    raw += [(c, blk_p, via) for (c, via, blk_p) in prefix_conds]
                    for e in path:
                        via = _edge_via(e)
                        # D5: register the caller-function block so the call-site guard
                        # code is retrievable by blockId (not just by location).
                        eblk = _cpp_fn_block(store, e["caller_stem"], e["caller_fn"], cfg_cache, asm_dir)
                        for c in _edge_guard(e, cfg_cache, asm_dir):
                            raw.append((c, eblk, via))
                    raw += [(c, blk, None) for c in local]
                    # files the path traverses, in order (setter file is the last).
                    path_stems = [tail.stem] + [e["callee_stem"] for e in path if e["cross"]]
                    # ordered (callee_stem, callee_fn) of every function entered on the path.
                    route_fns = [(e["callee_stem"], e["callee_fn"]) for e in path]
                    # E-2: if the robust enumerator truncated with ZERO complete paths
                    # (use_robust_cpp False → this legacy fallback ran), its truncation
                    # must still surface here, else pathsCapped is silently lost.
                    built.append((path_stems, raw, incomplete or routes_capped or cpp_paths_trunc,
                                  route_fns))
            else:
                raw = [(c, blk_p, via) for (c, via, blk_p) in prefix_conds]
                for hop in rt.get("hops", []):
                    hln = hop.get("line") or 0
                    hfile = hop.get("from_file") or s.file_stem
                    # via carries the descend CALL line (the ENTRC into the next file),
                    # mirroring the C++ `caller→callee@line` form.
                    hvia = f"{hop.get('from_file')}→{hop.get('to_file')}" + (f"@{hln}" if hln else "")
                    # the call's own block — for the GAP-029 block_gate polarity (#4).
                    _cb = _asm_block_at(_asm_bp(hfile, blueprint_dir, asm_bp_cache), hln) if hln else None
                    call_block = _cb[0] if _cb else None
                    for cond in (hop.get("conditions") or []):
                        rec = _resolve_hop_condition(cond, call_block=call_block)
                        if rec:
                            txt, rtst, rb, bline = rec
                            # anchor at the COMPARE instruction (not the branch/call line)
                            cline, dline = _hop_cond_lines(hfile, rtst, bline, hln,
                                                           blueprint_dir, asm_bp_cache)
                            # register the ASM block (guards + the descend ENTRC) so the
                            # call site's source is in the trace and blockId is non-null.
                            hblk = _asm_hop_block(hfile, cline, blueprint_dir, asm_bp_cache, store)
                            raw.append((_mk_cond(txt, f"{hfile}.asm", cline, end_line=dline,
                                                 raw_test=rtst, raw_branch=rb), hblk, hvia))
                raw += [(c, blk, None) for c in local]
                # ASM/mixed route: no per-function C++ call edges here → empty route_fns
                # (value-flow prune is a no-op, never touches a mixed route).
                built.append((list(rt_files), raw, routes_capped or cpp_paths_trunc, []))

        # ---- robust C++ function-level paths (route-undercount fix) ----
        # Authoritative for pure-C++ routing: each path already carries every
        # intra- and cross-file call edge, so we build its guard set exactly like
        # the legacy pure-C++ branch above. The new routes (through the intermediate
        # files the caller-walk dropped) carry distinct upstream guards → distinct
        # tuples; any that coincide with a legacy/mixed tuple are dropped by the
        # seen_tuples dedup below.
        for path in cpp_paths:
            raw = [(c, blk_p, via) for (c, via, blk_p) in prefix_conds]
            for e in path:
                via = _edge_via(e)
                eblk = _cpp_fn_block(store, e["caller_stem"], e["caller_fn"], cfg_cache, asm_dir)
                for c in _edge_guard(e, cfg_cache, asm_dir):
                    raw.append((c, eblk, via))
            raw += [(c, blk, None) for c in local]
            path_stems = [tail.stem] + [e["callee_stem"] for e in path if e["cross"]]
            route_fns = [(e["callee_stem"], e["callee_fn"]) for e in path]
            built.append((path_stems, raw, cpp_paths_trunc, route_fns))

        # ---- finalise each tuple, extract deps, dedup ----
        for path_stems, raw, incomplete, route_fns in built:
            conditions, kept = _finalize_conditions(raw, cr)
            # SOUND value-flow prune: drop a route that is provably unable to reach the
            # setter because a function ON THIS ROUTE unconditionally clobbers the switch
            # scrutinee to a scoped enum constant incompatible with the setter's enclosing
            # `case` guard (e.g. processManualEntryCardNumberOnly forces cidDbSelect =
            # ManualCardNumberOnly, so `case ManualExpirationDateOnly`@368 can never run on
            # that route). Only provably-infeasible routes are pruned (see helper), so the
            # feasible processManualEntryExpirationDateOnly route — whose set is CONDITIONAL
            # — is kept. No-op for ASM/mixed/unverified routes (route_fns empty).
            if _route_value_flow_infeasible(conditions, route_fns, cfg_cache, asm_dir,
                                            _vf_setter_memo, _vf_cond_memo):
                continue
            dep_refs = extract_dep_vars(s, kept, constant_symbols=consts,
                                        cpp_const_names=cr.bare_cpp_const_names(),
                                        asm_indirect_ctx=asm_ind_ctx)
            chain = list(prefix_stems)
            for f in path_stems:                 # order-preserving union (no dup files)
                if f not in chain:
                    chain.append(f)
            # B8: path identity (ordered via-edges) is part of the key, so two distinct
            # call paths with coincidentally-equal guard text are NOT collapsed.
            via_sig = tuple(via for (_c, _blk, via) in raw if via)
            tkey = (setter_loc["file"], setter_loc["startLine"], tuple(chain),
                    tuple(sorted(c["condition"] for c in conditions)), via_sig)
            if tkey in seen_tuples:
                continue
            seen_tuples.add(tkey)
            entry = {
                "value": s.value,
                "valueResolved": cr.resolve(s.value or "", s.language),
                "setterCodeChunk": s.setter_code_chunk,
                "location": setter_loc,
                "chain": chain,
                # provable overwrite: no guard on the whole path AND the path was
                # fully enumerated (B1 — never claim unconditional under truncation).
                "unconditionalAtSetter": len(conditions) == 0 and not incomplete,
                "conditions": conditions,
                "dependentVariables": [
                    {"name": d.name, "foundAt": d.found_at,
                     **({"qualified": d.qualified} if d.qualified else {}),
                     **({"indirection": d.indirection} if d.indirection else {})}
                    for d in dep_refs
                ],
            }
            if incomplete:
                entry["pathsCapped"] = True      # B1/B3: enumeration was truncated/partial — surfaced
            if via_unverified:
                entry["reachableOnlyViaUnverified"] = True   # B7: no verified route — surfaced
            if unverified:
                entry["unverifiedCallSites"] = unverified
            entries.append(entry)
            seed_deps.extend((variable, d) for d in dep_refs)
    _phase_routes.__exit__()
    _log("per-route entry building: %d (setter,chain) tuples", len(entries))

    # ---- 4. dependent-variable recursion (same engine, re-rooted, memoized) ----
    # #8: disable_dependents short-circuits the whole dep-var tree (root setters only).
    if disable_dependents:
        dep_objs: List[Dict[str, Any]] = []
        _log("dependent expansion: SKIPPED (disable_dependents)")
    else:
        with _Phase("dependent expansion"):
            seen_seed: set = set()
            for _pv, d in seed_deps:
                if d.name not in seen_seed:
                    seen_seed.add(d.name)
                    _log("  dep var: %s", d.name)
            chain_union = sorted({f for e in entries for f in e["chain"]})
            dep_objs = trace_dependents(
                seed_deps, chain_union,
                blueprint_dir=blueprint_dir, asm_dir=asm_dir, store=store,
                upstream_bounds=upstream_bounds, route_engine=route,
                chain_hops=[(h.stem, h.file_type, h.function) for h in chain_prefix],
                max_depth=max_dep_var_depth, const_resolver=cr,
                max_offchain_files=max_offchain_files, asm_max_levels=asm_max_levels,
                progress=progress,
            ) if seed_deps else []
            _log("dependent expansion: %d dependent variables", len(dep_objs))

    out = build_root_output(variable, entries, store, dependent_variables=dep_objs)
    out["rootVariable"]["aliases"] = [
        {"name": a.name, "language": a.language, "certainty": a.certainty, "via": a.via}
        for a in alias_set.aliases
    ]
    # Per-variable membership + cross-language counterpart (+ the counterpart's own
    # struct/mac). Same two keys are attached to every dependent variable (recurse.py),
    # so the whole trace carries "which struct/DSECT each variable belongs to and what
    # its twin in the other language is".
    _memb = get_membership_resolver(blueprint_dir, asm_dir)
    _root = _memb.resolve(variable, language, variable)
    out["rootVariable"]["memberOf"] = _root["memberOf"]
    out["rootVariable"]["counterpart"] = _root["counterpart"]
    _log("TOTAL trace (%.3fs): %d setters, %d dependent variables",
         time.perf_counter() - _t_total, len(entries), len(dep_objs))
    if job_id and _trace_sig:                  # §16 #1: cache the finished result (deterministic)
        try:
            from vbt.precompute import trace_cache as _TC
            _TC.store(job_id, _trace_sig, out)
        except Exception:
            pass
    if job_id:                                 # §16 #2-full: persist the route memo for later traces
        try:
            from vbt.precompute.route_cache_db import persist_route_cache
            persist_route_cache(route, job_id)
        except Exception:
            pass
    return out
