"""ASM intra-file condition collection — IMPROVE-THEN-USE (SPEC §4).

Reuses the proven CFG builder (`_get_file_cfg_maps`) READ-ONLY, then adds the
soundness fixes the survey flagged:

  1. **Per-branch-target polarity.** In a predecessor block, only the conditional
     branch whose *target is the successor we arrived from* was TAKEN; every other
     conditional branch in that block jumped to a different target, so on the path
     to our successor it was NOT taken (fallthrough → NEGATED). Polarity is decided
     per TRIGGER+BRANCH pair from the CFG branch targets, never from a single blanket
     edge type — so a spurious source-order FALLTHROUGH in-edge can never invert a
     guard reached via a real BRANCH edge (audit ⑧-C3).
  2. **Branch recognition independent of the reused set.** A mnemonic is treated as
     a conditional branch when it is present in our polarity tables (which include
     `BL`/`BNL`, audit ⑧-C1) — not solely when it appears in the reused `BRANCH_INST`,
     which omits those base mnemonics and would silently drop the guard.
  3. **Load-and-test rendered against zero.** `LTR R5,R5` / `LT`/`LTGR`/`LTG` and the
     self-form `ICM R,M,R` set the condition code from the sign/zero of the loaded
     value, so the predicate is rendered ``operand <op> 0`` — not a vacuous
     ``R5 != R5`` two-operand compare (audit ⑧-C4).
  4. **N-level predecessor walk** — collect guards across the whole backward CFG path
     to the function/subroutine entry, not just the setter's own block.

Polarity rule: a guard reached on the *fallthrough* path uses the **negation** of the
branch predicate (the branch jumps away from the setter); a guard reached on the
*taken* edge uses the predicate as-is.

Public signature:
    collect_asm_conditions(bp_data, bp_path, block_id, before_line, file, *,
                           max_levels=16) -> Tuple[List[Condition], bool]
The 2nd return is a depth-truncation flag (see the function docstring).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from vbt.interfaces import Condition, Location

# Read-only reuse: CFG construction only. We do our own TRIGGER+BRANCH pairing so
# we (a) can recover BL/BNL that the reused BRANCH_INST omits, and (b) have the
# per-branch target available for per-target polarity.
from backward_traversal.runner.backward_only_runner import _get_file_cfg_maps
from backward_traversal.glossary.instructions import TRIGGER_INST as _TRIGGER_INST_SHARED, BRANCH_INST
from backward_traversal.utils.subroutine_resolution import normalize_subroutine_target
from backward_traversal.utils.token_utils import looks_like_register_token

# VBT-local trigger extension (SPEC #4: reuse the shared glossary read-only, never
# mutate it in place — see vbt.resolve.asm_indirect_glossary for the same pattern).
# Unioned so TP/CHHSI/CLHHSI/CGFI guards are recognized and paired with a branch;
# otherwise they were silently dropped (the CC-setting compare was never a TRIGGER).
from vbt.conditions.asm_condition_glossary import TRIGGER_INST_EXT

TRIGGER_INST = set(_TRIGGER_INST_SHARED) | set(TRIGGER_INST_EXT)

# --------------------------------------------------------------------------- #
# Polarity tables
# --------------------------------------------------------------------------- #
# Operator a compare-branch asserts when TAKEN, after a compare (CLC/CLI/C/CR/CP…).
_CMP_TAKEN_OP: Dict[str, str] = {
    "BE": "==", "BER": "==", "JE": "==", "BZ": "==", "BZR": "==", "JZ": "==",
    "BNE": "!=", "BNER": "!=", "JNE": "!=", "BNZ": "!=", "JNZ": "!=",
    "BH": ">", "BHR": ">", "JH": ">", "BP": ">", "JP": ">",
    "BL": "<", "BLR": "<", "JL": "<", "BM": "<", "JM": "<",
    "BNH": "<=", "JNH": "<=", "BNP": "<=", "JNP": "<=",
    "BNL": ">=", "JNL": ">=", "BNM": ">=", "JNM": ">=",
}
_NEG_OP: Dict[str, str] = {"==": "!=", "!=": "==", ">": "<=", "<=": ">", "<": ">=", ">=": "<"}

# Test-under-mask (TM) branch semantics.
_TM_TAKEN: Dict[str, str] = {
    "BO": "all-ones", "JO": "all-ones", "BNO": "not-all-ones", "JNO": "not-all-ones",
    "BZ": "all-zeros", "JZ": "all-zeros", "BNZ": "not-all-zeros", "JNZ": "not-all-zeros",
    "BM": "mixed", "JM": "mixed", "BNM": "not-mixed", "JNM": "not-mixed",
}

# A mnemonic is a *conditional* branch for guard pairing when it is in our polarity
# tables (covers BL/BNL etc.) OR in the reused BRANCH_INST set — minus the
# unconditional transfers, which are not guards.  Decided independently of the
# reused set so the BL/BNL omission (audit ⑧-C1) cannot drop a guard.
_UNCONDITIONAL = {"B", "J", "BR", "BCR"}
_BRANCH_MNEMONICS = (set(_CMP_TAKEN_OP) | set(_TM_TAKEN) | set(BRANCH_INST)) - _UNCONDITIONAL

# Load-and-test family: the condition code reflects the sign/zero of a single loaded
# value, so the predicate is rendered against 0 (audit ⑧-C4).  For the register/
# register self-forms (LTR R5,R5) the operands are identical; for LT/LTG (load from
# memory) and the ICM self-form the meaningful operand is the loaded register/field.
_LOAD_AND_TEST = {"LTR", "LTGR", "LT", "LTG"}


def _operand_head(arg: str) -> str:
    """The operand token from a (possibly comment-contaminated) HLASM arg.

    HLASM ends the operand field at the first blank that is OUTSIDE a quoted literal;
    everything after is an inline comment. The upstream parser sometimes glues that
    comment onto the LAST operand in the blueprint's columnar ``args`` (e.g.
    ``EBWKIDX+4 BASE THE ADDRESSES AREA`` or ``PAMSG1 '   PRE-AUTH MESSAGE ?``), so the
    comment words otherwise leak into a condition's predicate / ``raw_test`` /
    ``raw_branch`` and then into the dependent-variable set (``THE``/``MESSAGE``/…).

    Cutting at the first UNQUOTED blank is the HLASM operand boundary itself — it can
    only ever remove comment text, never split a real operand. It is the same boundary
    ``modifier_index`` / ``register_indirect`` already rely on (``_parse_raw_operands`` /
    ``_strip_operand_comment``). Quote-aware so ``C'A B'`` / ``B'1111'`` survive intact;
    an (degenerate) unterminated quote keeps the whole arg — the safe direction."""
    s = (arg or "").strip()
    in_q = False
    for i, ch in enumerate(s):
        if ch == "'":
            in_q = not in_q
        elif ch.isspace() and not in_q:
            return s[:i]
    return s


def _split_inst(s: str) -> Tuple[str, List[str]]:
    s = (s or "").strip()
    if not s:
        return "", []
    parts = s.split(None, 1)
    mn = parts[0].upper()
    # Comment-strip each operand at the HLASM boundary (the route-finder hop-guard
    # string path arrives here; the columnar path is cleaned in _extract_pairs).
    args = [_operand_head(a) for a in parts[1].split(",")] if len(parts) > 1 else []
    return mn, args


def _is_conditional_branch(mnemonic: str) -> bool:
    return (mnemonic or "").upper() in _BRANCH_MNEMONICS


def _branch_target(mnemonic: str, args: List[str]) -> str:
    """Resolve a conditional-branch target to a clean block label (uppercased).

    Mirrors the reused CFG target extraction so polarity decisions line up with the
    edges in cfg_incoming: take the rightmost non-register operand.
    """
    for a in reversed(args):
        sym = normalize_subroutine_target(str(a).strip())
        if sym and not looks_like_register_token(sym):
            return sym.upper()
    return ""


def _is_load_and_test_self(t_inst: str, t_args: List[str]) -> bool:
    """True when the trigger is a load-and-test whose CC reflects sign/zero vs 0 — so the
    predicate is ``operand <op> 0``, never the vacuous ``operand <op> operand``.

    LTR/LTGR/LT/LTG always do; ICM does only in the self-form ``ICM R,mask,R``
    (mask all-ones load-and-test idiom) where args[0] == args[-1]; and ``OC X,X`` / ``NC
    X,X`` are the storage zero-test idiom (OR/AND a field with itself preserves it and
    sets CC = zero/non-zero), so ``OC EBWKADD,EBWKADD`` / ``JZ`` is ``EBWKADD == 0`` not
    ``EBWKADD == EBWKADD``. (``XC X,X`` is excluded — it CLEARS the field rather than
    testing it.)
    """
    if t_inst in _LOAD_AND_TEST:
        return True
    if t_inst == "ICM" and len(t_args) >= 3:
        first = (t_args[0] or "").strip().upper()
        last = (t_args[-1] or "").strip().upper()
        return bool(first) and first == last
    if t_inst in ("OC", "NC") and len(t_args) >= 2:
        a0 = (t_args[0] or "").strip().upper()
        a1 = (t_args[1] or "").strip().upper()
        return bool(a0) and a0 == a1
    return False


def _resolve_predicate_parsed(t_inst: str, t_args: List[str], b_mn: str, *, taken: bool) -> str:
    """Render a human predicate from a pre-split TRIGGER + BRANCH pair at a polarity."""
    # Test-under-mask.
    if t_inst in ("TM", "TMY") and b_mn in _TM_TAKEN:
        field = t_args[0] if t_args else "?"
        mask = t_args[1] if len(t_args) > 1 else "?"
        sense = _TM_TAKEN[b_mn]
        if not taken:  # flip the sense
            sense = {"all-ones": "not-all-ones", "not-all-ones": "all-ones",
                     "all-zeros": "not-all-zeros", "not-all-zeros": "all-zeros",
                     "mixed": "not-mixed", "not-mixed": "mixed"}[sense]
        if sense == "all-ones":
            return f"({field} & {mask}) == {mask}"
        if sense == "all-zeros":
            return f"({field} & {mask}) == 0"
        if sense == "not-all-ones":
            return f"({field} & {mask}) != {mask}"
        if sense == "not-all-zeros":
            return f"({field} & {mask}) != 0"
        return f"{sense}({field} & {mask})"

    # Compare-style.
    op = _CMP_TAKEN_OP.get(b_mn)
    if op is None:
        # Unknown branch: keep raw, mark polarity verbally.
        test = (f"{t_inst} {', '.join(t_args)}" if t_args else t_inst).strip()
        branch = b_mn
        return f"{test} {'taken' if taken else 'fallthrough'}:{branch}"
    if not taken:
        op = _NEG_OP[op]
    lhs = t_args[0] if t_args else "?"
    # Load-and-test sets CC from sign/zero of the loaded operand → compare vs 0,
    # never operand-vs-operand (which would render the vacuous "R5 != R5").
    if _is_load_and_test_self(t_inst, t_args):
        rhs = "0"
    elif t_inst == "ICM" and len(t_args) >= 3:
        # Memory form ``ICM R,mask,ADDR``: CC comes from the INSERTED BYTES, not a
        # compare — args[1] is the mask, never a comparand (was rendered "R15 != 15").
        # The meaningful operand is the storage field: predicate is ``ADDR <op> 0``.
        lhs = t_args[-1]
        rhs = "0"
    else:
        rhs = t_args[1] if len(t_args) > 1 else "0"
    return f"{lhs} {op} {rhs}"


def _resolve_predicate(test: str, branch: str, *, taken: bool) -> str:
    """Render a human predicate from TRIGGER + BRANCH instruction strings.

    Stable string-based contract (also used by vbt.engine for raw route-finder hop
    guards).  Splits the operands and delegates to the parsed resolver, so the
    load-and-test (audit ⑧-C4) and TM handling apply uniformly to both call paths.
    """
    t_inst, t_args = _split_inst(test)
    b_mn, _ = _split_inst(branch)
    return _resolve_predicate_parsed(t_inst, t_args, b_mn, taken=taken)


def _iter_flow(block: Dict) -> List[Dict]:
    """Iterate a block's flow entries (handles both old list and columnar formats)."""
    from blueprint_io import iter_flow
    return list(iter_flow(block))


def _extract_pairs(
    block: Dict,
    *,
    before_line: int = 0,
) -> List[Tuple[str, List[str], str, List[str], int, int]]:
    """Pair every TRIGGER with the next conditional BRANCH within a block.

    Returns ``(trigger_inst, trigger_args, branch_inst, branch_args, trigger_line,
    branch_line)`` in source order.  ``trigger_line`` is the line of the COMPARE/TEST
    instruction (where the guarded field + comparand actually live); ``branch_line`` is
    the conditional-branch (decision) line.  Only entries strictly before ``before_line``
    are considered when ``before_line`` is positive (0 = whole block).  An unconditional
    transfer clears any pending trigger so a stale compare is never paired across a jump.
    """
    pairs: List[Tuple[str, List[str], str, List[str], int, int]] = []
    pending: Optional[Tuple[str, List[str], int]] = None
    for flow in sorted(_iter_flow(block), key=lambda e: int(e.get("line") or 0)):
        line_no = int(flow.get("line") or 0)
        if before_line and line_no and line_no >= before_line:
            break
        inst = str(flow.get("inst") or "").upper()
        # Comment-strip each columnar operand at the HLASM boundary (the upstream
        # parser glues inline comments onto the last operand for some instructions).
        # This cleans the rendered predicate AND raw_test/raw_branch in one place, so
        # vbt.depvars.extract never harvests comment words as dependent variables.
        args = [_operand_head(str(a)) for a in (flow.get("args") or [])]
        if inst in TRIGGER_INST:
            pending = (inst, args, line_no)
        elif _is_conditional_branch(inst):
            if pending is not None:
                pairs.append((pending[0], pending[1], inst, args, pending[2], line_no))
            pending = None
        elif inst in _UNCONDITIONAL:
            # An unconditional transfer ends the straight-line run; drop any pending
            # compare so it is not mis-paired with a later branch.
            pending = None
    return pairs


def find_trigger_line(bp_data: Dict, raw_test: str, at_or_before: int) -> Optional[int]:
    """Source line of the TRIGGER (compare/test) instruction ``raw_test`` — the flow
    entry with the GREATEST line ≤ ``at_or_before`` whose mnemonic and first operand
    match. Used to re-point a route_finder hop guard (whose ``line`` is the BRANCH /
    call-site line) at the actual compare instruction (e.g. ``TM L7DSTS,X'40'`` at
    2012 for a guard route_finder pinned to the ``JNO`` at 2014). ``None`` if not found
    (caller falls back to the branch line, never fabricating)."""
    want_mn, want_args = _split_inst(raw_test or "")
    if not want_mn:
        return None
    want_op0 = want_args[0].upper() if want_args else ""
    best: Optional[int] = None
    for block in (bp_data.get("blocks") or []):
        for e in _iter_flow(block):
            ln = int(e.get("line") or 0)
            if not ln or ln > at_or_before:
                continue
            if str(e.get("inst") or "").upper() != want_mn:
                continue
            eargs = [_operand_head(str(a)) for a in (e.get("args") or [])]
            if want_op0 and not (eargs and eargs[0].upper() == want_op0):
                continue
            if best is None or ln > best:
                best = ln
    return best


def _pairs_to_conditions(
    block: Dict,
    block_id: str,
    file: str,
    *,
    taken_target: Optional[str],
    before_line: int,
    order_start: int,
) -> List[Condition]:
    """Convert a block's TRIGGER+BRANCH pairs to polarity-resolved Conditions.

    ``taken_target`` is the successor block we arrived from (uppercased), or ``None``
    for the setter's own block (where every guard before the setter line is on the
    fallthrough path → negated).  Per pair: taken iff the branch target equals
    ``taken_target``; otherwise the branch jumped to a different target and was NOT
    taken on the path to our successor → negated.
    """
    out: List[Condition] = []
    order = order_start
    for t_inst, t_args, b_mn, b_args, trig_line, br_line in _extract_pairs(block, before_line=before_line):
        if taken_target is None:
            taken = False
        else:
            taken = _branch_target(b_mn, b_args) == taken_target
        pred = _resolve_predicate_parsed(t_inst, t_args, b_mn, taken=taken)
        raw_test = (f"{t_inst} {', '.join(t_args)}" if t_args else t_inst).strip()
        raw_branch = (f"{b_mn} {', '.join(b_args)}" if b_args else b_mn).strip()
        # Location spans the COMPARE (start_line) to the BRANCH/decision (end_line): the
        # guard's data (tested field + comparand) lives at the compare instruction, so
        # that — not the branch — is where the condition is anchored. The branch line is
        # kept as end_line so a consumer can show the full compare→decision source span.
        out.append(Condition(
            order=order,
            condition=pred,
            block_id=block_id,
            location=Location(file=file, start_line=trig_line or br_line,
                              end_line=br_line or trig_line),
            polarity=taken,
            raw_test=raw_test,
            raw_branch=raw_branch,
            kind="asm_guard",
        ))
        order += 1
    return out


_ASM_DOM_CACHE: Dict[str, Tuple[Dict[str, Set[str]], Dict[str, List[str]], Dict[str, List[str]]]] = {}


def _cfg_dom_reach(bp_path: str) -> Tuple[Dict[str, Set[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """``(dominators, cfg_out, cfg_in)`` for the file's block CFG, cached per ``bp_path``.

    ``dominators[b]`` = the blocks that DOMINATE ``b`` — every path from a module ENTRY to
    ``b`` passes through them. Computed with a virtual super-source over all entry blocks
    (ASM modules are multi-entry — verified: aa71 has 9 entries), ``$IS$`` + out-of-graph
    edges filtered. This is the basis for collecting only the NECESSARY setter guards: a
    cross-block guard is necessary iff its block dominates the setter AND the branch gates
    the path to it (validated against source — see collect_asm_conditions)."""
    cached = _ASM_DOM_CACHE.get(bp_path)
    if cached is not None:
        return cached
    try:
        bmap, cfg_out_raw, cfg_in_raw = _get_file_cfg_maps(bp_path)
    except Exception:
        bmap, cfg_out_raw, cfg_in_raw = {}, {}, {}
    ids = set(bmap)

    def _clean(d: Dict) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for b in ids:
            lst: List[str] = []
            for x in (d.get(b) or []):
                nid = str(x[0] if isinstance(x, (tuple, list)) else x)
                if nid in ids and nid != "$IS$":
                    lst.append(nid)
            out[b] = lst
        return out

    cfg_out = _clean(cfg_out_raw)
    cfg_in = _clean(cfg_in_raw)
    dom: Dict[str, Set[str]] = {}
    if ids:
        SS = "\x00super"
        sources = [b for b in ids if not cfg_in[b]]      # no in-edges ⇒ entry point
        preds = {b: set(cfg_in[b]) for b in ids}
        for s in sources:
            preds[s].add(SS)
        universe = set(ids) | {SS}
        dom = {b: set(universe) for b in ids}
        dom[SS] = {SS}
        changed = True
        while changed:
            changed = False
            for b in ids:
                nd = set(universe)
                for p in preds[b]:
                    nd &= dom.get(p, {p})
                nd.add(b)
                if nd != dom[b]:
                    dom[b] = nd
                    changed = True
    result = (dom, cfg_out, cfg_in)
    _ASM_DOM_CACHE[bp_path] = result
    return result


def collect_asm_conditions(
    bp_data: Dict,
    bp_path: str,
    block_id: str,
    before_line: int,
    file: str,
    *,
    max_levels: int = 16,
) -> Tuple[List[Condition], bool]:
    """Collect all guard conditions to reach (block_id, before_line), polarity-resolved.

    Same-block guards before ``before_line`` are on the fallthrough path (the branch
    jumps away from the setter) → negated polarity. Then walk ``cfg_incoming`` up to
    ``max_levels``: for each predecessor block, polarity is decided *per branch* by
    whether the branch target is the successor we arrived from (taken) or a different
    block (fallthrough → negated) — never by a single blanket arrival edge type, so a
    spurious source-order FALLTHROUGH in-edge cannot invert a real BRANCH guard.

    Returns ``(conditions, truncated)``. ``truncated`` is True when a dominator of the
    setter block was dropped solely because its CFG depth exceeded ``max_levels`` — i.e.
    a necessary guard may have been lost to the depth cap. Callers propagate this into
    ``pathsCapped`` / ``incomplete`` so a depth-truncated setter is never reported as
    ``unconditionalAtSetter`` (a false positive). The flag is conservative: it fires for
    ANY over-deep dominator, including a non-gating diamond that would have contributed
    no necessary guard, so it can also cap a genuinely-unconditional setter.
    """
    conditions: List[Condition] = []
    truncated = False

    block_map: Dict[str, Dict] = {str(b.get("id") or ""): b for b in (bp_data.get("blocks") or [])}

    # (1) Same-block guards before the setter line → fallthrough polarity (no taken
    #     target: every conditional branch before the setter jumped away).
    setter_block = block_map.get(block_id)
    if setter_block is not None:
        conditions.extend(_pairs_to_conditions(
            setter_block, block_id, file,
            taken_target=None, before_line=before_line, order_start=len(conditions),
        ))

    # (2) NECESSARY cross-block guards (SPEC §4 — necessary-guards semantics).
    #     The OLD behavior BFS-walked EVERY converging backward path and emitted all
    #     their guards as a flat conjunction. But a guard on only one of several
    #     converging paths is NOT necessary — another path reaches the setter without it
    #     — so that OVER-collected (unsound: it presented path-specific guards as though
    #     they all must hold; verified against source — aa71's AA710A5B pulled 164 such,
    #     only a handful necessary). We now keep a cross-block guard ONLY when:
    #       (a) its block DOMINATES the setter block (every path from a module entry to
    #           the setter passes through it), AND
    #       (b) the branch GATES the setter — exactly ONE arm can reach the setter, so
    #           reaching the setter forces that arm's polarity (a diamond whose both arms
    #           reconverge at the setter does not gate it → dropped).
    #     Same-block guards (step 1) are always necessary and unaffected. Validated
    #     against source: LG1OSI's `WK_RRC == ZEROS` is preserved; aa71 164 → necessary
    #     subset (dominance confirmed by a remove-block reachability test).
    dom, cfg_out, cfg_in = _cfg_dom_reach(bp_path)
    if dom and block_id in dom:
        # backward closure: every block that can reach the setter block.
        reach: Set[str] = {block_id}
        stack = [block_id]
        while stack:
            cur = stack.pop()
            for p in cfg_in.get(cur, []):
                if p not in reach:
                    reach.add(p)
                    stack.append(p)
        # dominators of the setter, ordered closest-first (backward-BFS distance),
        # bounded by max_levels.
        depth: Dict[str, int] = {}
        seen: Set[str] = {block_id}
        frontier = [block_id]
        lvl = 0
        while frontier:
            lvl += 1
            nf: List[str] = []
            for cur in frontier:
                for p in cfg_in.get(cur, []):
                    if p not in seen:
                        seen.add(p)
                        depth[p] = lvl
                        nf.append(p)
            frontier = nf
        domset_candidates: List[str] = []
        for p in dom[block_id]:
            if p in block_map and p != block_id:
                if depth.get(p, max_levels + 1) <= max_levels:
                    domset_candidates.append(p)
                else:
                    # a dominator was dropped only because it sits past the depth cap —
                    # a necessary cross-block guard may have been lost here.
                    truncated = True
        # secondary key: block id — dom[block_id] is a SET, so equal-depth dominators
        # otherwise keep hash-seed iteration order and the emitted guard order (the
        # ``order`` field) flips between processes for the same trace.
        domset = sorted(domset_candidates, key=lambda p: (depth.get(p, 1 << 30), str(p)))
        for P in domset:
            pred_block = block_map.get(P)
            if pred_block is None:
                continue
            succs = cfg_out.get(P, [])
            for t_inst, t_args, b_mn, b_args, trig_line, br_line in _extract_pairs(pred_block, before_line=0):
                tgt = (_branch_target(b_mn, b_args) or "").upper()
                taken_reaches = any(s in reach for s in succs if s.upper() == tgt)
                fall_reaches = any(s in reach for s in succs if s.upper() != tgt)
                # GATE: necessary iff exactly one arm can reach the setter.
                if taken_reaches and not fall_reaches:
                    taken = True
                elif fall_reaches and not taken_reaches:
                    taken = False
                else:
                    continue
                pred = _resolve_predicate_parsed(t_inst, t_args, b_mn, taken=taken)
                raw_test = (f"{t_inst} {', '.join(t_args)}" if t_args else t_inst).strip()
                raw_branch = (f"{b_mn} {', '.join(b_args)}" if b_args else b_mn).strip()
                conditions.append(Condition(
                    order=len(conditions) + 1, condition=pred, block_id=P,
                    location=Location(file=file, start_line=trig_line or br_line,
                                      end_line=br_line or trig_line),
                    polarity=taken, raw_test=raw_test, raw_branch=raw_branch, kind="asm_guard"))

    # Re-number order from setter outward (closest guard = order 1).
    for i, c in enumerate(conditions, start=1):
        c.order = i
    return conditions, truncated


# --------------------------------------------------------------------------- #
# GAP 9 — ASM "not-set" outcomes (¬ of the setters' guards), the ASM analog of
# compute_cpp_not_set_conditions. Same formula, over collect_asm_conditions.
# --------------------------------------------------------------------------- #
def _negate_asm_literal(c: Condition) -> str:
    """Logical negation of one ASM guard literal. Preferred: re-render natively at the
    FLIPPED branch polarity (``(f & m) == m`` ↔ ``(f & m) != m``) via ``_resolve_predicate`` —
    a faithful, readable ASM predicate. Falls back to a textual ``!(...)`` wrap when the raw
    instructions / polarity are unavailable."""
    if c.raw_test and c.polarity is not None:
        try:
            neg = _resolve_predicate(c.raw_test, c.raw_branch or "", taken=not bool(c.polarity))
            if neg:
                return neg
        except Exception:
            pass
    t = (c.condition or "").strip()
    if t.startswith("!(") and t.endswith(")"):
        return t[2:-1]
    return f"!({t})"


def compute_asm_not_set_conditions(
    bp_data: Dict,
    bp_path: str,
    file: str,
    setters,
    *,
    asm_max_levels: int = 16,
) -> Tuple[List[Condition], str]:
    """GAP 9 (ASM): the not-set guard for variable ``V`` in one ASM module = the negation of
    the reaching guards of ``V``'s real ASM setters there: ``∧_S (∨_{L∈guard(S)} ¬L)``.

    ``setters`` is the list of real ASM ``SetterSite``s of V in the module (each carries
    ``block_id``/``line``); their guards come from ``collect_asm_conditions`` (the same source
    the real setters use). Returns ``(conditions, status)`` with the same contract as the C++
    helper: ``"ok"`` (emit), ``"unconditional"`` (a setter runs on every path ⇒ V always set),
    ``"incomplete"`` (a contributing guard was depth-cap TRUNCATED ⇒ only necessary, not
    sufficient ⇒ suppress), ``"tautology"`` (two setters reached under a compare and its exact
    opposite ⇒ union is a tautology / guards unreliable ⇒ suppress), ``"no_function"`` (no lines).
    SOUND: never emit a possibly over-broad negation."""
    lines = [s for s in setters if getattr(s, "line", 0)]
    if not lines:
        return [], "no_function"
    anchor = min(lines, key=lambda s: s.line)
    conjuncts: List[Condition] = []
    single_by_test: Dict[str, Set[bool]] = {}   # raw_test -> polarities seen (tautology probe)
    for s in lines:
        conds, truncated = collect_asm_conditions(
            bp_data, bp_path, s.block_id, s.line, file, max_levels=asm_max_levels)
        if truncated:
            return [], "incomplete"
        if not conds:
            return [], "unconditional"
        if len(conds) == 1 and conds[0].raw_test and conds[0].polarity is not None:
            single_by_test.setdefault(conds[0].raw_test.strip(), set()).add(bool(conds[0].polarity))
        neg_terms: List[str] = []
        for c in conds:                       # dedup identical negated literals (collect_asm
            t = _negate_asm_literal(c)         # may emit the same guard via multiple CFG paths)
            if t not in neg_terms:
                neg_terms.append(t)
        text = " || ".join(neg_terms)
        conjuncts.append(Condition(order=0, condition=text, block_id=s.block_id,
                                   location=Location(file, s.line, s.line),
                                   polarity=None, kind="not_set"))
    # Tautology: the same compare reached under both polarities (e.g. set in both arms) ⇒
    # union is a tautology / under-collected ⇒ emit nothing.
    if any(len(v) > 1 for v in single_by_test.values()):
        return [], "tautology"
    seen: Set[str] = set()
    uniq: List[Condition] = []
    for c in conjuncts:
        if c.condition in seen:
            continue
        seen.add(c.condition)
        uniq.append(c)
    for i, c in enumerate(uniq, start=1):
        c.order = i
    return uniq, "ok"
