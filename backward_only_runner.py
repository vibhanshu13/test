"""End-to-end runner for backward-only chain traversal."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from backward_traversal.bridge.cross_file_bridge_backward import (
    find_asm_callers,
    find_asm_to_cpp_callers,
    find_cpp_to_asm_callers,
    find_cpp_to_cpp_callers,
)
from backward_traversal.tracing.asm_backward_tracer import (
    DEFAULT_MAX_SUBROUTINE_DEPTH,
    DEFAULT_MAX_SUBROUTINE_NODES,
    DEFAULT_MAX_TRACE_NODES,
    _get_file_cfg_maps,
    backward_reachable_blocks,
    clear_cfg_cache,
    merge_call_graphs,
    trace_asm_call_site_backward,
)
from backward_traversal.glossary.instructions import (
    ASYNC_CALL_OPCODES, BRANCH_INST, CROSS_FILE_INST, DEST_ARG_INDEX_BY_INST,
    EXECUTE_INST, NON_RETURNING_CALL_OPCODES, SETTER_INST, TRIGGER_INST,
    CREEC_PARAM_KEYWORDS, CROSS_FILE_INST, DB_KEYLIST_INST, DB_MANAGED_FIELD_WRITES,
    DEST_ARG_INDEX_BY_INST, FILE_FIND_INST, IMMEDIATE_OPERAND_SETTER_INST,
    LEVEL_ALLOC_INST, LEVEL_TO_CE1DT,
    LEVEL_TRANSFER_INST, MACRO_FIXED_RESULT_FIELD, PRE_CALL_OPCODES,
    REGISTER_LOAD_FROM_MEM, SETTER_INST,
)
from backward_traversal.glossary.special_cases import classify_setter

# Hardware-sourced stores: the destination receives a CPU-provided value, not a
# named source variable (audit §3c).  Annotated on the emitted setter site.
_HARDWARE_SOURCE_BY_INST = {
    "STCK": "TOD_CLOCK",
    "STCKF": "TOD_CLOCK_FAST",
    "STFLE": "FACILITY_BITS",
}
from backward_traversal.utils.blueprint_utils import (
    collect_constant_symbols,
    discover_asm_dir,
    discover_blueprint_dir,
    extract_single_block_source,
    fetch_raw_line,
    discover_graph_file,
    get_blueprint_cache_info,
    load_json,
    resolve_asm_blueprint,
    resolve_cpp_blueprint,
    resolve_ex_target,
    resolve_file_type,
    resolve_source_file,
)
from backward_traversal.utils.cpp_source_utils import build_func_source_index
from backward_traversal.enrichment.terminal_classifier import (
    classify_terminal_variables,
    extract_terminals_from_output,
    extract_terminals_from_partitioned_dir,
)
from backward_traversal.utils.token_utils import normalize_token, looks_like_register_token, looks_like_equ_constant
from backward_traversal.utils.blueprint_consistency import diagnose_blueprint_consistency
from backward_traversal.utils.cpp_lineage_utils import (
    build_chain_neighbor_set,
    extract_asm_field_aliases,
    extract_tpf_regs_slots,
    find_cpp_files_for_dep_var,
    find_cpp_seed_nodes,
)
from backward_traversal.tracing.cpp_backward_tracer import (
    DEFAULT_CPP_MAX_DEPTH,
    DEFAULT_CPP_MAX_NODES,
    trace_cpp_variable_backward_multi,
)

logger = logging.getLogger(__name__)

# Maximum seed nodes per dep_var in Pass 2 (dep_var BFS).
# Variables with many seeds (e.g. CIDRECORD with 42) cause BFS explosion.
# Seeds are priority-ordered (1=highest from cpp_seed_keys), so truncating
# keeps the most relevant seeds.  Does NOT apply to Pass 1 (initial variable).
_MAX_SEEDS_PER_DEP_VAR: int = int(os.environ.get("ASM_MAX_SEEDS_PER_DEP_VAR", "10"))


def _strip_hlasm_remark_from_val(val: str) -> str:
    """Strip HLASM assembly remark from a blueprint ``flow["val"]`` field.

    The blueprint parser occasionally includes the assembly remark (free-form
    comment) together with the source operand in the ``val`` field, e.g.::

        'WB1LG1CYCL SET CYCLE CUT DATE = DATE FROM MVS'

    In HLASM the remark field begins at the first blank character that is not
    inside a single-quoted string literal.  This function strips everything
    from that blank onward so only the true operand is returned.

    Examples::

        'WB1LG1CYCL SET CYCLE CUT DATE = DATE...' -> 'WB1LG1CYCL'
        "C'D'"                                    -> "C'D'"
        "ZEROS"                                   -> "ZEROS"
        "LG1RXR"                                  -> "LG1RXR"
    """
    in_quote = False
    for i, ch in enumerate(val):
        if ch == "'":
            in_quote = not in_quote
        elif ch == " " and not in_quote:
            return val[:i].strip()
    return val.strip()


from backward_traversal.utils.variable_name_resolver import VariableNameResolver

try:
    from blueprint_io import iter_flow
except ImportError:
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))))
    from blueprint_io import iter_flow

DEFAULT_MAX_DEP_VARS = 50
DEFAULT_MAX_DEP_VAR_DEPTH = 4
DEFAULT_MAX_DOWNSTREAM_DEPTH = 2
DEFAULT_MAX_DOWNSTREAM_FILES = 30

# Maximum line distance between a branch instruction and the call site for the
# condition to be classified as a "call_site_guard" rather than a "block_gate".
CALL_SITE_GUARD_THRESHOLD = 10


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except Exception:
        return max(minimum, default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _norm_file_id(stem: str) -> str:
    return str(stem or "").strip().lower()


def _get_block_index(bp_data: Dict[str, Any]) -> Dict[str, Tuple[int, Dict[str, Any]]]:
    """Return a cached uppercase block-id → (position, block) index."""
    cached = bp_data.get("_block_index_by_upper")
    if isinstance(cached, dict):
        return cached
    index: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for pos, block in enumerate(bp_data.get("blocks") or []):
        bid_upper = str(block.get("id") or "").upper()
        if bid_upper and bid_upper not in index:
            index[bid_upper] = (pos, block)
    bp_data["_block_index_by_upper"] = index
    return index


def _get_block(bp_data: Dict[str, Any], block_id: str) -> Optional[Dict[str, Any]]:
    entry = _get_block_index(bp_data).get(str(block_id or "").upper())
    return entry[1] if entry is not None else None


def _iter_scope_blocks(
    bp_data: Dict[str, Any],
    scope_blocks: Set[str],
):
    """Yield scoped blocks in original blueprint order without scanning all blocks."""
    if not scope_blocks:
        return
    index = _get_block_index(bp_data)
    ordered = [
        entry
        for raw_block_id in scope_blocks
        for entry in [index.get(str(raw_block_id or "").upper())]
        if entry is not None
    ]
    for _, block in sorted(ordered, key=lambda item: item[0]):
        yield block


def _new_full_flow_collector() -> Dict[str, Any]:
    return {"nodes": {}, "edges": {}}


class _DepVarWriter:
    """Accumulates dep_var trace results in memory (single-file mode) or writes each
    dep_var to its own JSON file immediately after it is computed (partitioned mode).

    Single-file mode (dep_var_dir=None):
        Behaves identically to the old ``dep_var_traces_output`` dict.  The entire
        output dict is assembled in memory and written as one JSON file at the end.

    Partitioned mode (dep_var_dir=<Path>):
        Each dep_var is serialised and flushed to ``<dep_var_dir>/<name>.json`` as
        soon as its BFS iteration completes.  The large result dicts (chain_file_results,
        downstream_file_results) are released immediately, keeping peak memory
        proportional to *one* dep_var trace instead of *all* dep_var traces combined.
        A ``manifest.json`` in the output directory references every written file.
    """

    def __init__(self, dep_var_dir: Optional[Path] = None) -> None:
        self._dir = dep_var_dir
        self._data: Dict[str, Any] = {}      # single-file accumulator
        self._manifest: List[str] = []       # partitioned: ordered list of filenames
        if dep_var_dir is not None:
            dep_var_dir.mkdir(parents=True, exist_ok=True)

    @property
    def partitioned(self) -> bool:
        return self._dir is not None

    def write(self, var_name: str, trace: Dict[str, Any]) -> None:
        """Record one dep_var result.  In partitioned mode, flush to disk immediately."""
        if self._dir is None:
            self._data[var_name] = trace
        else:
            safe = re.sub(r"[^A-Za-z0-9@$_#\-]", "_", var_name)
            fname = f"{safe}.json"
            # Embed var_name so readers can identify the variable without parsing filename
            serialised = dict(trace)
            serialised["dep_var"] = var_name
            (self._dir / fname).write_text(
                json.dumps(serialised, indent=2), encoding="utf-8"
            )
            self._manifest.append(fname)
            # Do NOT store a reference — allow the trace dict to be GC'd immediately.

    def to_dict(self) -> Dict[str, Any]:
        """Return the in-memory accumulator (single-file mode)."""
        return self._data

    def manifest(self) -> List[str]:
        """Return ordered list of written filenames (partitioned mode), deduplicated."""
        seen: set = set()
        out: List[str] = []
        for f in self._manifest:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out


def _consolidate_dep_var_setters(
    dep_var_traces: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a consolidated view of setter sites discovered during dep_var BFS.

    Returns a dict keyed by dep_var name, each containing ``chain_setters``
    and ``downstream_setters`` sub-dicts that map file stems to their setter
    site lists.  This provides a single lookup for consumers that need to
    know *all* setter sites associated with a given dependent variable
    without walking the full trace structure.
    """
    summary: Dict[str, Any] = {}
    for dv_name, trace in dep_var_traces.items():
        chain_setters: Dict[str, list] = {}
        downstream_setters: Dict[str, list] = {}

        for stem, fdata in (trace.get("chain_file_results") or {}).items():
            sites = fdata.get("setter_sites") if isinstance(fdata, dict) else None
            if sites:
                chain_setters[stem] = sites

        for stem, fdata in (trace.get("downstream_file_results") or {}).items():
            sites = fdata.get("setter_sites") if isinstance(fdata, dict) else None
            if sites:
                downstream_setters[stem] = sites

        if chain_setters or downstream_setters:
            summary[dv_name] = {
                "chain_setters": chain_setters,
                "downstream_setters": downstream_setters,
            }
    return summary


def _consolidate_dep_var_setters_from_dir(dep_vars_dir: Path) -> Dict[str, Any]:
    """Partitioned-mode variant: reads individual dep_var JSON files from disk."""
    summary: Dict[str, Any] = {}
    if not dep_vars_dir.is_dir():
        return summary
    for fpath in sorted(dep_vars_dir.glob("*.json")):
        try:
            trace = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        dv_name = trace.get("dep_var", fpath.stem)
        chain_setters: Dict[str, list] = {}
        downstream_setters: Dict[str, list] = {}

        for stem, fdata in (trace.get("chain_file_results") or {}).items():
            sites = fdata.get("setter_sites") if isinstance(fdata, dict) else None
            if sites:
                chain_setters[stem] = sites

        for stem, fdata in (trace.get("downstream_file_results") or {}).items():
            sites = fdata.get("setter_sites") if isinstance(fdata, dict) else None
            if sites:
                downstream_setters[stem] = sites

        if chain_setters or downstream_setters:
            summary[dv_name] = {
                "chain_setters": chain_setters,
                "downstream_setters": downstream_setters,
            }
    return summary


# ---------------------------------------------------------------------------
# Deduplicated function/block body store
# ---------------------------------------------------------------------------

class _FunctionBlockCollector:
    """Deduplicated store for function/block source bodies.

    Key format: ``"{stem}::{routine_id}"``.
    """

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def add(self, stem: str, routine_id: str, source_text: str) -> str:
        key = f"{stem}::{routine_id}"
        if key not in self._store:
            self._store[key] = source_text
        return key

    def has(self, stem: str, routine_id: str) -> bool:
        return f"{stem}::{routine_id}" in self._store

    def to_dict(self) -> Dict[str, str]:
        return dict(self._store)


def _annotate_setter_sites(
    setter_sites: List[Dict[str, Any]],
    stem: str,
    file_type: str,
    collector: _FunctionBlockCollector,
    *,
    bp_data: Optional[Dict[str, Any]] = None,
    asm_file: Optional[Path] = None,
    blueprint_dir: Optional[Path] = None,
    asm_dir: Optional[Path] = None,
    _source_cache: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Annotate *setter_sites* with ``block_ref`` keys into *collector*.

    For ASM sites the enclosing block source is extracted via
    ``extract_single_block_source``; for C++ sites the function body comes
    from ``build_func_source_index`` (LRU-cached).
    """
    for site in setter_sites:
        routine = site.get("routine", "")
        if not routine or routine == "(global)":
            continue
        if collector.has(stem, routine):
            site["block_ref"] = f"{stem}::{routine}"
            continue
        source: Optional[str] = None
        if file_type == "asm":
            if bp_data is not None:
                source = extract_single_block_source(
                    bp_data, asm_file, routine, _source_cache
                )
        elif file_type == "cpp":
            if blueprint_dir is not None:
                idx = build_func_source_index(blueprint_dir, asm_dir, stem)
                source = idx.get(routine)
        if source:
            collector.add(stem, routine, source)
            site["block_ref"] = f"{stem}::{routine}"


def _ensure_full_flow_node(
    collector: Dict[str, Any],
    stem: str,
    *,
    blueprint_dir: Path,
    file_type_map: Dict[str, str],
    selected_chain_set: Set[str],
    roles: Optional[List[str]] = None,
    scope_var: Optional[str] = None,
) -> Dict[str, Any]:
    node_id = _norm_file_id(stem)
    if not node_id:
        return {}
    nodes = collector["nodes"]
    if node_id not in nodes:
        resolved_type = resolve_file_type(node_id, blueprint_dir, file_type_map)
        nodes[node_id] = {
            "id": node_id,
            "file_type": resolved_type if resolved_type else "unknown",
            "scope_variables": set(),
            "in_selected_chain": node_id in selected_chain_set,
            "roles": set(),
        }
    node = nodes[node_id]
    if roles:
        for r in roles:
            if r:
                node["roles"].add(str(r))
    if scope_var:
        node["scope_variables"].add(str(scope_var).upper())
    return node


def _add_full_flow_edge(
    collector: Dict[str, Any],
    *,
    source: str,
    target: str,
    blueprint_dir: Path,
    file_type_map: Dict[str, str],
    selected_chain_set: Set[str],
    scope_var: Optional[str] = None,
    instruction: Optional[str] = None,
    discovered_from: Optional[str] = None,
    source_block: Optional[str] = None,
    line_no: Optional[int] = None,
    conditions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    src_id = _norm_file_id(source)
    tgt_id = _norm_file_id(target)
    if not src_id or not tgt_id:
        return

    src_roles = []
    tgt_roles = []
    if discovered_from:
        src_roles.append(f"{discovered_from}_source")
        tgt_roles.append(f"{discovered_from}_target")

    _ensure_full_flow_node(
        collector,
        src_id,
        blueprint_dir=blueprint_dir,
        file_type_map=file_type_map,
        selected_chain_set=selected_chain_set,
        roles=src_roles,
        scope_var=scope_var,
    )
    _ensure_full_flow_node(
        collector,
        tgt_id,
        blueprint_dir=blueprint_dir,
        file_type_map=file_type_map,
        selected_chain_set=selected_chain_set,
        roles=tgt_roles,
        scope_var=scope_var,
    )

    edge_key = (src_id, tgt_id)
    edges = collector["edges"]
    if edge_key not in edges:
        edges[edge_key] = {
            "source": src_id,
            "target": tgt_id,
            "instructions": set(),
            "scope_variables": set(),
            "discovered_from": set(),
            "source_blocks": set(),
            "lines": set(),
            "conditions": [],
        }
    edge = edges[edge_key]
    if instruction:
        edge["instructions"].add(str(instruction).upper())
    if scope_var:
        edge["scope_variables"].add(str(scope_var).upper())
    if discovered_from:
        edge["discovered_from"].add(str(discovered_from))
    if source_block:
        edge["source_blocks"].add(str(source_block))
    if line_no is not None:
        try:
            ln = int(line_no)
            if ln > 0:
                edge["lines"].add(ln)
        except Exception:
            pass
    if conditions:
        # Deduplicate by (line, test, branch) so the same guard from the same
        # call site is not recorded twice when this edge is visited repeatedly,
        # while still preserving distinct occurrences at different line numbers.
        existing_keys = {
            (c.get("line"), c.get("test"), c.get("branch"))
            for c in edge["conditions"]
        }
        for cond in conditions:
            if not cond or not isinstance(cond, dict):
                continue
            key = (cond.get("line"), cond.get("test"), cond.get("branch"))
            if key not in existing_keys:
                existing_keys.add(key)
                edge["conditions"].append(cond)


def _forward_reachable_from(
    bp_path_str: str,
    start_blocks: Set[str],
) -> Optional[Set[str]]:
    """Forward BFS from start_blocks via cfg_outgoing. Returns None when unavailable."""
    if not bp_path_str or not start_blocks:
        return None
    try:
        _, cfg_outgoing, _ = _get_file_cfg_maps(bp_path_str)
    except Exception:
        return None
    valid_starts = {b for b in start_blocks if b in cfg_outgoing}
    if not valid_starts:
        return None
    result: Set[str] = set()
    fwd_q: Deque[str] = deque(valid_starts)
    while fwd_q:
        b = fwd_q.popleft()
        if b in result:
            continue
        result.add(b)
        for succ_b, _ in cfg_outgoing.get(b, []):
            if succ_b not in result:
                fwd_q.append(succ_b)
    return result if result else None


def _extract_block_conditions_before_line(
    bp_data: Dict[str, Any],
    block_id: str,
    call_line: int,
    bp_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return all guard conditions before call_line in block_id as structured dicts.

    Each returned dict has:
      - ``type``:   ``"block_gate"`` or ``"call_site_guard"``
      - ``line``:   line number of the branch instruction (int or None)
      - ``test``:   trigger instruction string, e.g. ``"CLC WK_RRC, ZEROS"``
      - ``branch``: branch instruction string, e.g. ``"BNE DA7_0900"``

    A condition is classified as ``"call_site_guard"`` when the branch falls within
    ``CALL_SITE_GUARD_THRESHOLD`` lines immediately before ``call_line``; everything
    else is ``"block_gate"``.  All occurrences are reported — the same pattern
    appearing multiple times (e.g. three ``CLC WK_RRC,ZEROS / BNE DA7_0900`` checks
    after three consecutive ENTRC calls) produces three separate entries, each with
    its own line number, so the structural information is preserved.

    When block_gate conditions are found but no call_site_guard is present, a
    synthetic ``{"type": "call_site_guard", "line": None,
    "note": "unconditional at call site"}`` entry is appended to clarify that the
    call itself has no immediate guard even though the block is entered conditionally.

    GAP-029: When no in-block conditions are found, falls back to predecessor-block
    scanning via the CFG.  Handles the case where the call is the first instruction
    in block_id and the guarding TRIGGER+BRANCH lives in a predecessor block that
    explicitly branches to block_id.  Predecessor-block conditions are always
    classified as ``"block_gate"``.
    """
    conditions: List[Dict[str, Any]] = []
    pending_trigger: Optional[str] = None

    block = _get_block(bp_data, block_id)
    if block is not None:
        for flow in iter_flow(block):
            line_no = int(flow.get("line") or 0)
            if call_line and line_no and line_no >= call_line:
                break
            inst = str(flow.get("inst") or "").upper()
            args = [str(a) for a in (flow.get("args") or [])]
            if inst in TRIGGER_INST:
                pending_trigger = (f"{inst} {', '.join(args)}" if args else inst).strip()
            elif inst in BRANCH_INST and pending_trigger:
                branch_str = (f"{inst} {', '.join(args)}" if args else inst).strip()
                cond_type = (
                    "call_site_guard"
                    if call_line and line_no and (call_line - line_no) <= CALL_SITE_GUARD_THRESHOLD
                    else "block_gate"
                )
                conditions.append({
                    "type": cond_type,
                    "line": line_no if line_no else None,
                    "test": pending_trigger,
                    "branch": branch_str,
                })
                pending_trigger = None

    # GAP-029: No in-block guard found — scan predecessor blocks for a
    # TRIGGER+BRANCH pair whose BRANCH target is block_id.  Only applies
    # when the call is effectively the first meaningful instruction in the
    # block (condition lives in the block that conditionally jumps here).
    if not conditions and bp_path:
        try:
            _, _, cfg_incoming = _get_file_cfg_maps(bp_path)
            seen_preds: set = set()
            for pred_info in (cfg_incoming.get(block_id) or []):
                pred_id = pred_info[0] if isinstance(pred_info, (tuple, list)) else pred_info
                if pred_id in seen_preds:
                    continue
                seen_preds.add(pred_id)
                pred_block = _get_block(bp_data, str(pred_id))
                if not pred_block:
                    continue
                pred_pending: Optional[str] = None
                for flow in iter_flow(pred_block):
                    inst = str(flow.get("inst") or "").upper()
                    args = [str(a) for a in (flow.get("args") or [])]
                    line_no = int(flow.get("line") or 0)
                    if inst in TRIGGER_INST:
                        pred_pending = (f"{inst} {', '.join(args)}" if args else inst).strip()
                    elif inst in BRANCH_INST:
                        branch_target = str(args[0]).upper() if args else ""
                        if pred_pending and branch_target == block_id.upper():
                            branch_str = (f"{inst} {', '.join(args)}" if args else inst).strip()
                            conditions.append({
                                "type": "block_gate",
                                "line": line_no if line_no else None,
                                "test": pred_pending,
                                "branch": branch_str,
                            })
                        # Any branch clears the pending trigger regardless of target
                        pred_pending = None
        except Exception:
            pass

    # When the block is gated (block_gate conditions exist) but the call site
    # itself has no immediate guard, append a synthetic note so readers know the
    # call is unconditional at its own line even though the block is not.
    has_call_site_guard = any(c.get("type") == "call_site_guard" for c in conditions)
    has_block_gates = any(c.get("type") == "block_gate" for c in conditions)
    if has_block_gates and not has_call_site_guard:
        conditions.append({
            "type": "call_site_guard",
            "line": None,
            "note": "unconditional at call site",
        })

    return conditions


def _get_function_tp_name(bp_data: Dict[str, Any], line_no: int) -> Optional[str]:
    """Return the TP-name operand (args[0]) for a FUNCTION instruction at line_no.

    Scans all blocks in bp_data for a FUNCTION flow entry whose line matches
    line_no and returns the first operand (the 4-char routing key, e.g. ``GNA1``).
    Returns None when no match is found.
    """
    for block in (bp_data.get("blocks") or []):
        for flow in iter_flow(block):
            if int(flow.get("line") or 0) == line_no:
                if str(flow.get("inst") or "").upper() == "FUNCTION":
                    args = [str(a) for a in (flow.get("args") or [])]
                    return args[0].strip() if args else None
    return None


def _extract_condition_field(test_str: str) -> Optional[str]:
    """Extract the first-operand field name from a condition test string.

    Examples::

        'CLI LG1TYP,C\\'D\\''          → 'LG1TYP'
        'CLC WK_RRC,ZEROS'             → 'WK_RRC'
        'TM  WB1SW3,X\\'01\\''          → 'WB1SW3'
        'OC  WK_CNT,WK_CNT'           → 'WK_CNT'
        'CLC WB1ACCTC+12(2),=C\\'00\\'' → 'WB1ACCTC'

    Returns None for register references, literals (``=...``), or bare offsets.
    """
    if not test_str:
        return None
    parts = str(test_str).strip().split()
    if len(parts) < 2:
        return None
    # First operand is everything before the first comma
    raw_operand = parts[1].split(",")[0].strip()
    # Strip displacement / length specifiers: WB1ACCTC+12(2) → WB1ACCTC
    base = re.split(r"[+(]", raw_operand)[0].strip()
    # Skip literals, pure numeric offsets, or empty
    if not base or base.startswith("=") or base[0].isdigit():
        return None
    return base


def _find_preceding_entrc_calls(
    bp_data: Dict[str, Any],
    block_ids_upper: Set[str],
    before_line: int,
) -> Dict[str, List[int]]:
    """Return ENTRC calls in the specified blocks that precede before_line.

    Only ``ENTRC`` (the returning cross-file call) is considered; ``ENTNC``
    (tail-call, no return) cannot contribute data that is available after it.

    Returns ``{target_stem: [line_numbers]}`` for each ENTRC found.
    """
    results: Dict[str, List[int]] = {}
    for block in _iter_scope_blocks(bp_data, block_ids_upper):
        for flow in iter_flow(block):
            flow_line = int(flow.get("line") or 0)
            if flow_line <= 0 or flow_line >= before_line:
                continue
            if str(flow.get("inst") or "").upper() != "ENTRC":
                continue
            args = [str(a) for a in (flow.get("args") or [])]
            if not args:
                continue
            target = _norm_file_id(args[0].strip())
            if not target:
                continue
            if target not in results:
                results[target] = []
            results[target].append(flow_line)
    return results


def _find_preceding_calls_with_bas(
    bp_data: Dict[str, Any],
    block_ids_upper: Set[str],
    before_line: int,
) -> List[Tuple[str, int, str]]:
    """Return ENTRC and BAS calls in block_ids_upper that precede before_line.

    Extends ``_find_preceding_entrc_calls`` to also capture internal ``BAS``
    subroutine calls, which can write output variables that are tested as
    conditions on downstream edges.

    Returns a list of ``(target, line, call_type)`` where:
    - ``target`` is the normalised file stem (ENTRC) or block label (BAS).
    - ``call_type`` is ``"ENTRC"`` or ``"BAS"``.
    """
    results: List[Tuple[str, int, str]] = []
    for block in _iter_scope_blocks(bp_data, block_ids_upper):
        for flow in iter_flow(block):
            flow_line = int(flow.get("line") or 0)
            if flow_line <= 0 or flow_line >= before_line:
                continue
            inst = str(flow.get("inst") or "").upper()
            args = [str(a) for a in (flow.get("args") or [])]
            if inst == "ENTRC" and args:
                target = _norm_file_id(args[0].strip())
                if target:
                    results.append((target, flow_line, "ENTRC"))
            elif inst == "BAS" and len(args) >= 2:
                label = args[1].strip().upper()
                if label:
                    results.append((label, flow_line, "BAS"))
    return results


def _find_call_contributor_for_condition(
    bp_data: Dict[str, Any],
    block_id: str,
    condition_line: int,
    field_upper: str,
) -> Optional[Tuple[str, int, str]]:
    """Find the ENTRC/BAS call that immediately precedes a condition test.

    Scans the named block backward from ``condition_line``.  Returns the first
    ``ENTRC`` or ``BAS`` call encountered before any write to ``field_upper``.
    A write to the field between the call and the test would mean the value was
    set locally, not by the call — in that case returns ``None``.

    This proximity heuristic does not require loading the target's blueprint and
    is the primary attribution mechanism for Type-1 (own-condition) contributions.

    Returns ``(target, call_line, call_inst)`` where ``call_inst`` is
    ``"ENTRC"`` or ``"BAS"``, or ``None`` when no eligible call is found.
    """
    setter_insts: Set[str] = {str(i).upper() for i in SETTER_INST}
    entries: List[Tuple[int, str, List[str]]] = []

    block = _get_block(bp_data, block_id)
    if block is not None:
        for flow in iter_flow(block):
            ln = int(flow.get("line") or 0)
            if ln <= 0 or ln >= condition_line:
                continue
            inst = str(flow.get("inst") or "").upper()
            args = [str(a) for a in (flow.get("args") or [])]
            entries.append((ln, inst, args))

    trigger_insts: Set[str] = {str(i).upper() for i in TRIGGER_INST}

    for ln, inst, args in reversed(entries):
        if inst == "ENTRC":
            target = _norm_file_id(args[0].strip()) if args else ""
            if target:
                return target, ln, "ENTRC"
        elif inst == "BAS":
            label = args[1].strip().upper() if len(args) >= 2 else ""
            if label:
                return label, ln, "BAS"
        elif inst in trigger_insts:
            # Condition test (e.g. OC X,X, TM, CLI) — not a value-changing write; skip
            continue
        elif inst in setter_insts:
            dest_idx = DEST_ARG_INDEX_BY_INST.get(inst, 0)
            dest = args[dest_idx].upper() if dest_idx < len(args) else ""
            dest_base = re.split(r"[+(]", dest)[0].strip()
            if dest_base == field_upper:
                return None  # local write found; call is not the source

    return None


def _blueprint_writes_field(bp_data: Dict[str, Any], field_name: str) -> bool:
    """Return True if bp_data contains any setter instruction targeting field_name.

    Checks all blocks and uses ``DEST_ARG_INDEX_BY_INST`` to identify the
    destination operand for each setter instruction.  Displacement and length
    specifiers are stripped before comparison (``LG1TYP+2`` matches ``LG1TYP``).
    """
    if not bp_data or not field_name:
        return False
    setter_insts = {str(i).upper() for i in SETTER_INST}
    field_upper = field_name.upper()
    for block in (bp_data.get("blocks") or []):
        for flow in iter_flow(block):
            inst = str(flow.get("inst") or "").upper()
            if inst not in setter_insts:
                continue
            args = [str(a) for a in (flow.get("args") or [])]
            if not args:
                continue
            dest_idx = DEST_ARG_INDEX_BY_INST.get(inst, 0)
            if dest_idx >= len(args):
                dest_idx = 0
            dest = args[dest_idx].upper()
            dest_base = re.split(r"[+(]", dest)[0].strip()
            if dest_base == field_upper:
                return True
    return False


def _block_writes_field(
    bp_data: Dict[str, Any],
    block_id: str,
    field_name: str,
) -> bool:
    """Return True if the named block in bp_data contains a setter for field_name.

    Like ``_blueprint_writes_field`` but restricted to a single block — used
    to check whether an intra-file BAS-called subroutine writes a particular
    field without scanning the entire file.
    """
    if not bp_data or not block_id or not field_name:
        return False
    setter_insts: Set[str] = {str(i).upper() for i in SETTER_INST}
    field_upper = field_name.upper()
    block = _get_block(bp_data, block_id)
    if block is not None:
        for flow in iter_flow(block):
            inst = str(flow.get("inst") or "").upper()
            if inst not in setter_insts:
                continue
            args = [str(a) for a in (flow.get("args") or [])]
            if not args:
                continue
            dest_idx = DEST_ARG_INDEX_BY_INST.get(inst, 0)
            if dest_idx >= len(args):
                dest_idx = 0
            dest = args[dest_idx].upper()
            dest_base = re.split(r"[+(]", dest)[0].strip()
            if dest_base == field_upper:
                return True
    return False


def _compute_data_flow_contributions(
    full_flow: Dict[str, Any],
    blueprint_dir: Path,
) -> None:
    """Annotate edges with data_flow_contributions where a preceding call
    builds a field consumed as a condition on this edge or its downstream edges.

    For each edge A→B, the function detects two contribution patterns:

    **Type 1 — own-condition contributor (proximity-based)**: A field tested in
    edge A→B's own conditions (e.g., ``OC WK_CNT,WK_CNT``) is attributed to
    the ``ENTRC`` or ``BAS`` call that immediately precedes the condition test
    in A's source block, with no intervening write to that field.  This does
    not require loading the target's blueprint; proximity in the source block
    is the authoritative indicator.

    **Type 2 — downstream-condition contributor**: A field tested in a
    downstream edge B→C's condition (e.g., ``CLI LG1TYP,C'D'``) is populated
    by an ``ENTRC`` or ``BAS`` call in A's block before the A→B call.  The
    target's blueprint (for ENTRC) or the named internal block in A's blueprint
    (for BAS) is checked to confirm it writes the field.

    Both patterns annotate the A→B edge with::

        {
            "field":                "LG1TYP",
            "built_by":             "ENTRC DB72 (lines 817, 825)",
            "consumed_by_condition":"CLI LG1TYP,C'D' in xh80 (xh80→xh71 gate)"
        }

    Blueprints are loaded at most once per stem (local cache).
    """
    edges_dict = full_flow.get("edges") or {}
    if not edges_dict:
        return

    # Index downstream condition fields: source_node → [(field, test_str, src, tgt)]
    src_cond_fields: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for edge in edges_dict.values():
        src = str(edge.get("source") or "")
        tgt = str(edge.get("target") or "")
        for cond in (edge.get("conditions") or []):
            if not isinstance(cond, dict):
                continue
            test_str = str(cond.get("test") or "")
            field = _extract_condition_field(test_str)
            if not field:
                continue
            src_cond_fields.setdefault(src, []).append((field, test_str, src, tgt))

    bp_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def _load_bp(stem: str) -> Optional[Dict[str, Any]]:
        if stem not in bp_cache:
            bp_path = resolve_asm_blueprint(stem, blueprint_dir)
            try:
                bp_cache[stem] = load_json(bp_path, keys={"blocks"}) if bp_path else None
            except Exception:
                bp_cache[stem] = None
        return bp_cache.get(stem)

    for edge_key, edge in list(edges_dict.items()):
        source_a = str(edge.get("source") or "")
        target_b = str(edge.get("target") or "")

        bp_a = _load_bp(source_a)
        if not bp_a:
            continue

        call_lines = sorted(
            int(l) for l in (edge.get("lines") or set()) if l and int(l) > 0
        )
        if not call_lines:
            continue
        min_call_line = call_lines[0]

        source_blocks_upper: Set[str] = {
            str(b).upper() for b in (edge.get("source_blocks") or set()) if b
        }

        contributions: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[str, str, str]] = set()

        # ── Type 1: per-condition proximity attribution ────────────────────────────────────────────────
        # For each condition on this edge, scan backward in the source block
        # from the condition line to find the ENTRC/BAS immediately before it.
        # No target-blueprint check needed — proximity is the authoritative signal.
        for cond in (edge.get("conditions") or []):
            if not isinstance(cond, dict):
                continue
            cond_line = cond.get("line")
            if not cond_line:
                continue
            test_str = str(cond.get("test") or "")
            field = _extract_condition_field(test_str)
            if not field:
                continue
            field_upper = field.upper()
            result: Optional[Tuple[str, int, str]] = None
            for blk_id in source_blocks_upper:
                result = _find_call_contributor_for_condition(
                    bp_a, blk_id, int(cond_line), field_upper
                )
                if result:
                    break
            if not result:
                continue
            call_target, call_line, call_inst = result
            if call_inst == "ENTRC":
                built_by = f"ENTRC {call_target.upper()} (line {call_line})"
            else:
                built_by = f"BAS {call_target.upper()} (line {call_line})"
            consumed_by = (
                f"{test_str} in {source_a} ({source_a}→{target_b} gate)"
            )
            key = (field, built_by, consumed_by)
            if key not in seen_keys:
                seen_keys.add(key)
                contributions.append({
                    "field": field,
                    "built_by": built_by,
                    "consumed_by_condition": consumed_by,
                })

        # ── Type 2: downstream condition fields (cross-edge) ───────────────────────────────────
        # Fields tested in downstream B→C edges — find which call in A's block
        # built the structure containing those fields.  Covers both ENTRC
        # (checked via the target's blueprint) and BAS (checked via the named
        # internal block within A's blueprint).
        type2_fields: List[Tuple[str, str, str, str]] = list(
            src_cond_fields.get(target_b) or []
        )
        if type2_fields and source_blocks_upper:
            preceding = _find_preceding_calls_with_bas(
                bp_a, source_blocks_upper, min_call_line
            )
            entrc_map: Dict[str, List[int]] = {}
            bas_map: Dict[str, List[int]] = {}
            for t, ln, kind in preceding:
                if kind == "ENTRC":
                    entrc_map.setdefault(t, []).append(ln)
                else:
                    bas_map.setdefault(t, []).append(ln)

            for (field, test_str, cond_src, cond_tgt) in type2_fields:
                field_upper = field.upper()
                consumed_by = (
                    f"{test_str} in {cond_src} ({cond_src}→{cond_tgt} gate)"
                )
                for t_stem, t_lines in entrc_map.items():
                    bp_t = _load_bp(t_stem)
                    if not bp_t or not _blueprint_writes_field(bp_t, field):
                        continue
                    t_lines_sorted = sorted(set(t_lines))
                    line_str = ", ".join(str(ln) for ln in t_lines_sorted)
                    built_by = f"ENTRC {t_stem.upper()} (lines {line_str})"
                    key = (field, built_by, consumed_by)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        contributions.append({
                            "field": field,
                            "built_by": built_by,
                            "consumed_by_condition": consumed_by,
                        })
                for blk_label, b_lines in bas_map.items():
                    if not _block_writes_field(bp_a, blk_label, field):
                        continue
                    b_lines_sorted = sorted(set(b_lines))
                    line_str = ", ".join(str(ln) for ln in b_lines_sorted)
                    built_by = f"BAS {blk_label.upper()} (lines {line_str})"
                    key = (field, built_by, consumed_by)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        contributions.append({
                            "field": field,
                            "built_by": built_by,
                            "consumed_by_condition": consumed_by,
                        })

        if contributions:
            edge["data_flow_contributions"] = contributions
def _enrich_function_dispatch_conditions(
    full_flow: Dict[str, Any],
    blueprint_dir: Path,
) -> None:
    """Inject dispatch_key conditions onto every FUNCTION-type edge in full_flow.

    For each edge carrying a ``FUNCTION`` instruction, loads the source file's
    blueprint and scans for the FUNCTION call at each recorded line number to
    extract the TP-name (``args[0]``).  Appends a structured ``dispatch_key``
    condition entry so callers can see which TP-names route to which targets::

        {
            "type":        "dispatch_key",
            "tp_name":     "GNA1",
            "line":        685,
            "description": "FUNCTION GNA1 → DA76"
        }

    This is a post-processing step that runs after all edge accumulation is
    complete, so it enriches both ``chain_upstream`` edges (which are built
    without blueprint access) and scope-traversal edges uniformly.
    Blueprints are loaded at most once per source file (cached in a local dict).
    """
    bp_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    for edge in (full_flow.get("edges") or {}).values():
        instructions = {str(i).upper() for i in (edge.get("instructions") or set())}
        if "FUNCTION" not in instructions:
            continue
        source_stem = str(edge.get("source") or "")
        target_stem = str(edge.get("target") or "")
        if source_stem not in bp_cache:
            bp_path = resolve_asm_blueprint(source_stem, blueprint_dir)
            try:
                bp_cache[source_stem] = load_json(bp_path, keys={"blocks"}) if bp_path else None
            except Exception:
                bp_cache[source_stem] = None
        bp_data = bp_cache.get(source_stem)
        if not bp_data:
            continue
        existing_keys: Set[Tuple] = {
            (c.get("line"), c.get("test"), c.get("branch"))
            for c in (edge.get("conditions") or [])
        }
        for line_no in sorted(edge.get("lines") or set()):
            tp_name = _get_function_tp_name(bp_data, line_no)
            if not tp_name:
                continue
            dispatch_cond: Dict[str, Any] = {
                "type": "dispatch_key",
                "tp_name": tp_name,
                "line": line_no,
                "description": f"FUNCTION {tp_name} → {target_stem.upper()}",
            }
            key: Tuple = (dispatch_cond.get("line"), dispatch_cond.get("test"), dispatch_cond.get("branch"))
            if key not in existing_keys:
                existing_keys.add(key)
                edge["conditions"].append(dispatch_cond)


def _csect_forward_reachable(
    bp_path_str: str,
    anchor_blocks: Set[str],
    all_file_blocks: Set[str],
) -> Set[str]:
    """Return forward-reachable blocks in the CSECT(s) containing anchor_blocks.

    Backward-walks cfg_incoming from each anchor to find CSECT root blocks
    (no predecessors), then forward-walks cfg_outgoing from those roots to
    collect every block in the same CSECT(s).  Eliminates cross-CSECT leakage
    when a source file contains multiple entry points / CSECTs.
    Falls back to all_file_blocks when the CFG cannot be loaded.
    """
    if not bp_path_str or not anchor_blocks:
        return all_file_blocks
    try:
        _, cfg_outgoing, cfg_incoming = _get_file_cfg_maps(bp_path_str)
    except Exception:
        return all_file_blocks

    # Backward BFS from every anchor to collect all backward-reachable blocks
    # within the same CSECT; blocks with no predecessors are the CSECT roots.
    backward_visited: Set[str] = set()
    bwd_q: Deque[str] = deque(anchor_blocks)
    while bwd_q:
        b = bwd_q.popleft()
        if b in backward_visited:
            continue
        backward_visited.add(b)
        for pred_b, _ in cfg_incoming.get(b, []):
            if pred_b not in backward_visited:
                bwd_q.append(pred_b)

    csect_roots: Set[str] = {b for b in backward_visited if not cfg_incoming.get(b)}
    if not csect_roots:
        csect_roots = set(anchor_blocks)

    fwd = _forward_reachable_from(bp_path_str, csect_roots)
    return fwd if fwd else all_file_blocks

_CROSS_FILE_CALL_CACHE_SIZE = int(os.environ.get("ASM_CROSS_FILE_CALL_CACHE_SIZE", "256"))


def _collect_all_cross_file_calls(
    bp_data: Dict[str, Any],
    asm_file: Optional[Path],
    asm_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cross_inst = {str(i).upper() for i in CROSS_FILE_INST | NON_RETURNING_CALL_OPCODES | ASYNC_CALL_OPCODES}
    from api.utils.indirection import extract_indirection

    def _extract_target_from_source_line(raw_line: str, instruction: str) -> str:
        text = str(raw_line or "").rstrip("\n")
        if not text:
            return ""
        stripped = text.lstrip()
        if not stripped or stripped.startswith("*"):
            return ""

        if len(text) >= 24:
            op_col = text[15:23].strip().upper()
            if op_col == instruction.upper():
                operand_col = text[23:].strip()
                if operand_col:
                    first = re.split(r"[,\s]+", operand_col, maxsplit=1)[0]
                    tok = normalize_token(first)
                    if tok and tok.upper() not in cross_inst:
                        return tok

        parts = stripped.split()
        inst_idx = -1
        for i, p in enumerate(parts):
            if p.rstrip(",").upper() == instruction.upper():
                inst_idx = i
                break
        if inst_idx >= 0 and inst_idx + 1 < len(parts):
            first = parts[inst_idx + 1].split(",")[0]
            tok = normalize_token(first)
            if tok and tok.upper() not in cross_inst:
                return tok
        return ""

    seen: Set[Tuple[str, int, str]] = set()

    def _try_add(source_block: str, instruction: str, target: str, line_no: Any,
                 call_type: str = "", indirection: Optional[Dict[str, Any]] = None) -> None:
        if not target:
            return
        ln = int(line_no) if line_no is not None else 0
        key = (source_block, ln, _norm_file_id(target))
        if key in seen:
            return
        seen.add(key)
        rec: Dict[str, Any] = {
            "source_block": source_block,
            "target_file": _norm_file_id(target),
            "instruction": instruction,
            "call_type": call_type,
            "line": line_no,
        }
        if indirection:
            rec["indirection"] = dict(indirection)
        out.append(rec)

    for edge in (bp_data.get("call_graph", {}) or {}).get("edges", []) or []:
        source_block = str(edge.get("source") or "")
        if not source_block:
            continue
        instruction = str(edge.get("instruction") or "").upper()
        if instruction not in cross_inst:
            continue
        target = normalize_token(str(edge.get("target") or "").strip())
        if not target or target.upper() in cross_inst:
            line_no = edge.get("line")
            file_field = str(edge.get("file") or str(asm_file or "")).strip()
            raw_line = None
            if line_no is not None and file_field:
                try:
                    raw_line = fetch_raw_line(file_field, int(line_no), asm_dir)
                except Exception:
                    raw_line = None
            recovered = _extract_target_from_source_line(str(raw_line or ""), instruction)
            target = recovered or ""
        if not target:
            continue
        raw_ct = str(edge.get("call_type") or "").lower()
        if not raw_ct:
            if instruction in NON_RETURNING_CALL_OPCODES or instruction == "FUNCTION":
                raw_ct = "no_return_call"
            elif instruction in ASYNC_CALL_OPCODES:
                raw_ct = "async_spawn"
        _try_add(source_block, instruction, target, edge.get("line"),
                 call_type=raw_ct, indirection=extract_indirection(edge))

    for block in bp_data.get("blocks") or []:
        block_id = str(block.get("id") or "")
        if not block_id:
            continue
        flow_items = list(iter_flow(block))
        for item in flow_items:
            instruction = str(item.get("inst") or "").upper()
            if instruction not in cross_inst:
                continue
            args = item.get("args") or []
            if instruction == "FUNCTION":
                operand_raw = str(args[1]) if len(args) > 1 else ""
            else:
                operand_raw = str(args[0]) if args else ""
            target = normalize_token(operand_raw.strip())
            if target and looks_like_register_token(target):
                target = ""
            if not target or target.upper() in cross_inst:
                line_no = item.get("line")
                file_field = str(asm_file or "").strip()
                raw_line = None
                if line_no is not None and file_field:
                    try:
                        raw_line = fetch_raw_line(file_field, int(line_no), asm_dir)
                    except Exception:
                        raw_line = None
                recovered = _extract_target_from_source_line(str(raw_line or ""), instruction)
                target = recovered or ""
            if not target:
                continue
            if instruction in NON_RETURNING_CALL_OPCODES or instruction == "FUNCTION":
                inferred_call_type = "no_return_call"
            elif instruction in ASYNC_CALL_OPCODES:
                inferred_call_type = "async_spawn"
            else:
                inferred_call_type = "subroutine_call"
            _try_add(block_id, instruction, target, item.get("line"),
                     call_type=inferred_call_type)

        for _fi, _fitem in enumerate(flow_items):
            _finst = str(_fitem.get("inst") or "").upper()
            if _finst != "BALR":
                continue
            _fargs = [str(a) for a in (_fitem.get("args") or [])]
            if len(_fargs) < 2:
                continue
            _balr_reg = normalize_token(_fargs[1])
            if not looks_like_register_token(_balr_reg):
                continue
            for _pi in range(max(0, _fi - 5), _fi):
                _pitem = flow_items[_pi]
                _pinst = str(_pitem.get("inst") or "").upper()
                if _pinst not in {"L", "LG"}:
                    continue
                _pargs = [str(a) for a in (_pitem.get("args") or [])]
                if len(_pargs) < 2:
                    continue
                _load_dest = normalize_token(_pargs[0])
                if _load_dest != _balr_reg:
                    continue
                _load_src = _pargs[1].strip().upper()
                import re as _re2
                _m = _re2.match(r"^=[AVRYS]\(([A-Z@\$_#][A-Z0-9@\$_#]*)\)$", _load_src)
                if _m:
                    _resolved_stem = _m.group(1)
                    _try_add(block_id, "BALR", _resolved_stem, _fitem.get("line"),
                             call_type="indirect_balr")
                    break
    return out


@lru_cache(maxsize=_CROSS_FILE_CALL_CACHE_SIZE)
def _get_cross_file_call_candidates_cached(
    bp_path_str: str,
    asm_file_str: str,
    asm_dir_str: str,
) -> Tuple[Dict[str, Any], ...]:
    bp_data = load_json(Path(bp_path_str), keys={"blocks", "call_graph"})
    asm_file = Path(asm_file_str) if asm_file_str else None
    asm_dir = Path(asm_dir_str) if asm_dir_str else None
    return tuple(_collect_all_cross_file_calls(bp_data, asm_file, asm_dir))


def _clear_cross_file_call_cache() -> None:
    try:
        _get_cross_file_call_candidates_cached.cache_clear()
    except Exception:
        pass


def _collect_scoped_cross_file_calls(
    bp_data: Dict[str, Any],
    scope_blocks: Set[str],
    asm_file: Optional[Path],
    asm_dir: Optional[Path],
    *,
    bp_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not scope_blocks:
        return out
    scope_blocks_upper: Set[str] = {b.upper() for b in scope_blocks}
    from api.utils.indirection import extract_indirection
    if bp_path is not None:
        candidates = _get_cross_file_call_candidates_cached(
            str(bp_path),
            str(asm_file) if asm_file else "",
            str(asm_dir) if asm_dir else "",
        )
        return [
            dict(rec)
            for rec in candidates
            if str(rec.get("source_block") or "").upper() in scope_blocks_upper
        ]
    cross_inst = {str(i).upper() for i in CROSS_FILE_INST | NON_RETURNING_CALL_OPCODES | ASYNC_CALL_OPCODES}

    def _extract_target_from_source_line(raw_line: str, instruction: str) -> str:
        text = str(raw_line or "").rstrip("\n")
        if not text:
            return ""
        stripped = text.lstrip()
        if not stripped or stripped.startswith("*"):
            return ""

        # Prefer fixed-column parse for assembler lines.
        if len(text) >= 24:
            op_col = text[15:23].strip().upper()
            if op_col == instruction.upper():
                operand_col = text[23:].strip()
                if operand_col:
                    first = re.split(r"[,\s]+", operand_col, maxsplit=1)[0]
                    tok = normalize_token(first)
                    if tok and tok.upper() not in cross_inst:
                        return tok

        parts = stripped.split()
        inst_idx = -1
        for i, p in enumerate(parts):
            if p.rstrip(",").upper() == instruction.upper():
                inst_idx = i
                break
        if inst_idx >= 0 and inst_idx + 1 < len(parts):
            first = parts[inst_idx + 1].split(",")[0]
            tok = normalize_token(first)
            if tok and tok.upper() not in cross_inst:
                return tok
        return ""

    seen: Set[Tuple[str, int, str]] = set()  # (source_block, line, target)

    def _try_add(source_block: str, instruction: str, target: str, line_no: Any,
                 call_type: str = "", indirection: Optional[Dict[str, Any]] = None) -> None:
        if not target:
            return
        ln = int(line_no) if line_no is not None else 0
        key = (source_block, ln, _norm_file_id(target))
        if key in seen:
            return
        seen.add(key)
        rec: Dict[str, Any] = {
            "source_block": source_block,
            "target_file": _norm_file_id(target),
            "instruction": instruction,
            "call_type": call_type,
            "line": line_no,
        }
        if indirection:
            rec["indirection"] = dict(indirection)
        out.append(rec)

    # Source 1: call_graph.edges
    for edge in (bp_data.get("call_graph", {}) or {}).get("edges", []) or []:
        source_block = str(edge.get("source") or "")
        if source_block.upper() not in scope_blocks_upper:  # GAP-010
            continue
        instruction = str(edge.get("instruction") or "").upper()
        if instruction not in cross_inst:
            continue
        target = normalize_token(str(edge.get("target") or "").strip())
        if not target or target.upper() in cross_inst:
            line_no = edge.get("line")
            file_field = str(edge.get("file") or str(asm_file or "")).strip()
            raw_line = None
            if line_no is not None and file_field:
                try:
                    raw_line = fetch_raw_line(file_field, int(line_no), asm_dir)
                except Exception:
                    raw_line = None
            recovered = _extract_target_from_source_line(str(raw_line or ""), instruction)
            target = recovered or ""
        if not target:
            continue
        raw_ct = str(edge.get("call_type") or "").lower()
        # Normalize missing call_type from call_graph.edges to match Source-2 inference.
        if not raw_ct:
            if instruction in NON_RETURNING_CALL_OPCODES or instruction == "FUNCTION":
                raw_ct = "no_return_call"
            elif instruction in ASYNC_CALL_OPCODES:
                raw_ct = "async_spawn"
        _try_add(source_block, instruction, target, edge.get("line"),
                 call_type=raw_ct, indirection=extract_indirection(edge))

    # Source 2: blocks[].flow (catches cross-file calls missing from call_graph)
    for block in bp_data.get("blocks") or []:
        block_id = str(block.get("id") or "")
        if not block_id or block_id.upper() not in scope_blocks_upper:  # GAP-010
            continue
        flow_items = list(iter_flow(block))
        for item in flow_items:
            instruction = str(item.get("inst") or "").upper()
            if instruction not in cross_inst:
                continue
            args = item.get("args") or []
            if instruction == "FUNCTION":
                # Case-2 fix: FUNCTION format is "FUNCTION <name>,<seg>[,options]".
                # args[0] = runtime lookup key (4-char name), args[1] = segment entered.
                operand_raw = str(args[1]) if len(args) > 1 else ""
            else:
                operand_raw = str(args[0]) if args else ""
            target = normalize_token(operand_raw.strip())
            # Detect register-indirect operand (e.g. "(R1)" normalizes to "R1").
            # A register name is never a valid cross-file module identifier — skip.
            if target and looks_like_register_token(target):
                target = ""
            if not target or target.upper() in cross_inst:
                # Try raw-line recovery
                line_no = item.get("line")
                file_field = str(asm_file or "").strip()
                raw_line = None
                if line_no is not None and file_field:
                    try:
                        raw_line = fetch_raw_line(file_field, int(line_no), asm_dir)
                    except Exception:
                        raw_line = None
                recovered = _extract_target_from_source_line(str(raw_line or ""), instruction)
                target = recovered or ""
            if not target:
                continue
            # GAP-004: Source-2 flow entries carry no call_type from the parser.
            # Infer based on instruction semantics.
            if instruction in NON_RETURNING_CALL_OPCODES or instruction == "FUNCTION":
                inferred_call_type = "no_return_call"
            elif instruction in ASYNC_CALL_OPCODES:
                inferred_call_type = "async_spawn"
            else:
                inferred_call_type = "subroutine_call"  # ENTRC, EXSR, SWISC etc.
            _try_add(block_id, instruction, target, item.get("line"),
                     call_type=inferred_call_type)
        # Indirect BALR: detect L Rx,=A(STEM) / BALR R14,Rx pattern.
        # Walk all flow entries for this block once to find BALR with register
        # second operand preceded within 5 instructions by L Rx,=A(STEM).
        for _fi, _fitem in enumerate(flow_items):
            _finst = str(_fitem.get("inst") or "").upper()
            if _finst != "BALR":
                continue
            _fargs = [str(a) for a in (_fitem.get("args") or [])]
            if len(_fargs) < 2:
                continue
            _balr_reg = normalize_token(_fargs[1])
            if not looks_like_register_token(_balr_reg):
                continue  # not register-indirect BALR
            # Look back up to 5 instructions for L Rx,=A(STEM)
            for _pi in range(max(0, _fi - 5), _fi):
                _pitem = flow_items[_pi]
                _pinst = str(_pitem.get("inst") or "").upper()
                if _pinst not in {"L", "LG"}:
                    continue
                _pargs = [str(a) for a in (_pitem.get("args") or [])]
                if len(_pargs) < 2:
                    continue
                _load_dest = normalize_token(_pargs[0])
                if _load_dest != _balr_reg:
                    continue
                _load_src = _pargs[1].strip().upper()
                import re as _re2
                _m = _re2.match(r"^=[AVRYS]\(([A-Z@\$_#][A-Z0-9@\$_#]*)\)$", _load_src)
                if _m:
                    _resolved_stem = _m.group(1)
                    _try_add(block_id, "BALR", _resolved_stem, _fitem.get("line"),
                             call_type="indirect_balr")
                    break
    return out


def _is_returning_cross_file_call(cf: Dict[str, Any]) -> bool:
    """Return True only for cross-file calls that return to caller context.

    For backward variable search expansion, we only traverse ENTRC edges that
    are modeled by the parser as subroutine_call.
    """
    inst = str(cf.get("instruction") or "").upper()
    call_type = str(cf.get("call_type") or "").lower()
    return inst == "ENTRC" and call_type == "subroutine_call"


def _is_followable_cross_file_call(cf: Dict[str, Any]) -> bool:
    """Return True for cross-file calls to follow for backward setter discovery.

    Extends _is_returning_cross_file_call to also include non-returning calls
    (ENTNC/ENTDC) because the target module may contain variable setter logic
    even when it does not return control to the caller (e.g. ECB field assignments
    in z/TPF that are visible to subsequent executions of the same transaction).

    GAP-ENTNC: closes the non-returning call path gap.
    P33-FIX: CALLC is a returning typed C call — semantically equivalent to ENTRC
    for data-flow purposes; it must be followed so that dep_var BFS and root_var
    downstream search reach files connected only via CALLC.
    """
    inst = str(cf.get("instruction") or "").upper()
    call_type = str(cf.get("call_type") or "").lower()
    if inst == "ENTRC" and call_type == "subroutine_call":
        return True
    # P33-FIX: CALLC — IBM's preferred typed cross-language call (CPROC/UPROC).
    # Returns to caller; semantically identical to ENTRC for backward tracing.
    if inst == "CALLC":
        return True
    # GAP-ENTNC: ENTNC/ENTDC — non-returning cross-file transfers that may still
    # contain setter logic for the traced variable.
    if inst in {"ENTNC", "ENTDC"}:
        return True
    return False


def _finalize_full_flow_collector(collector: Dict[str, Any]) -> Dict[str, Any]:
    nodes_out: List[Dict[str, Any]] = []
    for node_id, node in (collector.get("nodes") or {}).items():
        # Issue-3: exclude pure-target noise nodes — files that are only called by
        # traced files but are not themselves part of the chain or making traced calls.
        # A node is "pure-target" when every one of its roles ends in "_target" and
        # it is not flagged as in_selected_chain.
        node_roles = {str(r) for r in (node.get("roles") or set()) if str(r)}
        if not node.get("in_selected_chain") and node_roles:
            if all(r.endswith("_target") for r in node_roles):
                continue
        node_rec: Dict[str, Any] = {
            "id": node_id,
            "file_type": str(node.get("file_type") or "unknown"),
            "scope_variables": sorted({str(v) for v in (node.get("scope_variables") or set()) if str(v)}),
            "in_selected_chain": bool(node.get("in_selected_chain")),
            "roles": sorted(node_roles),
        }
        # Phase-3: include setter_sites on modifier nodes.
        if node.get("setter_sites"):
            node_rec["setter_sites"] = node["setter_sites"]
        nodes_out.append(node_rec)
    nodes_out = sorted(nodes_out, key=lambda n: str(n.get("id") or ""))
    kept_node_ids = {n["id"] for n in nodes_out}

    edges_out: List[Dict[str, Any]] = []
    for edge in sorted((collector.get("edges") or {}).values(), key=lambda e: (str(e.get("source") or ""), str(e.get("target") or ""))):
        # Issue-3: drop edges whose source or target was filtered as pure-target noise.
        if str(edge.get("source") or "") not in kept_node_ids or str(edge.get("target") or "") not in kept_node_ids:
            continue
        rec: Dict[str, Any] = {
            "source": str(edge.get("source") or ""),
            "target": str(edge.get("target") or ""),
            "instructions": sorted({str(i) for i in (edge.get("instructions") or set()) if str(i)}),
            "scope_variables": sorted({str(v) for v in (edge.get("scope_variables") or set()) if str(v)}),
            "discovered_from": sorted({str(d) for d in (edge.get("discovered_from") or set()) if str(d)}),
        }
        source_blocks = sorted({str(b) for b in (edge.get("source_blocks") or set()) if str(b)})
        if source_blocks:
            rec["source_blocks"] = source_blocks
        lines = sorted({int(l) for l in (edge.get("lines") or set()) if isinstance(l, int) and l > 0})
        if lines:
            rec["lines"] = lines
        # Issue-2: serialize guard conditions attached to this edge.
        # Conditions are stored as structured dicts {type, line, test, branch}.
        # Deduplicate by (line, test, branch) while preserving insertion order so
        # multiple occurrences of the same pattern at different lines are all kept.
        conditions_raw = edge.get("conditions") or []
        if conditions_raw:
            seen_cond_keys: set = set()
            conditions_out: List[Dict[str, Any]] = []
            for c in conditions_raw:
                if not isinstance(c, dict):
                    continue
                key = (c.get("line"), c.get("test"), c.get("branch"))
                if key in seen_cond_keys:
                    continue
                seen_cond_keys.add(key)
                conditions_out.append(c)
            if conditions_out:
                rec["conditions"] = conditions_out
        # Phase-4: include data_flow_contributions when present.
        if edge.get("data_flow_contributions"):
            rec["data_flow_contributions"] = edge["data_flow_contributions"]
        edges_out.append(rec)
    edges_out = sorted(edges_out, key=lambda e: (str(e.get("source") or ""), str(e.get("target") or "")))

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "is_partial": True,
        "note": "Only edges traversed during variable tracing are included.",
    }


def _dedupe_sites(sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for site in sorted(
        sites,
        key=lambda s: (
            int(s.get("line") or 0),
            str(s.get("routine") or ""),
            str(s.get("instruction") or ""),
            str(s.get("raw_line") or ""),
        ),
    ):
        key = (
            str(site.get("routine") or ""),
            int(site.get("line") or 0),
            str(site.get("instruction") or "").upper(),
            str(site.get("raw_line") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(site)
    return out


def _filter_setter_sites(
    sites: List[Dict[str, Any]],
    target_setters: List[Dict[str, Any]],
    file_stem: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Keep only setter sites matching *target_setters*.

    Each target must have ``line`` (required).  Optional ``file`` and
    ``routine`` narrow the match.  A site matches a target when all
    present target fields agree (case-insensitive for strings).
    """
    if not target_setters:
        return sites
    target_keys: List[Tuple[int, Optional[str], Optional[str]]] = []
    for ts in target_setters:
        target_keys.append((
            int(ts["line"]),
            Path((ts.get("file") or "").strip()).stem.lower() or None,
            (ts.get("routine") or "").strip().lower() or None,
        ))
    matched: List[Dict[str, Any]] = []
    for site in sites:
        site_line = int(site.get("line") or 0)
        # FIX: read "file" key (emitted by _extract_cpp_setter_sites) with
        # fallback to "source_file" (set by dep_var file-result builders) and
        # then file_stem.  Normalize to stem for apples-to-apples comparison
        # with target_setters which use bare stems.
        _raw_site_file = site.get("file") or site.get("source_file") or file_stem or ""
        site_file = Path(str(_raw_site_file).strip()).stem.lower()
        site_routine = str(site.get("routine") or site.get("function") or "").strip().lower()
        for ts_line, ts_file, ts_routine in target_keys:
            if site_line != ts_line:
                continue
            if ts_file and ts_file != site_file:
                continue
            if ts_routine and ts_routine != site_routine:
                continue
            matched.append(site)
            break
    return matched


def _scope_cpp_dep_vars_to_target_setters(
    cpp_trace: Dict[str, Any],
    matched_setter_sites: List[Dict[str, Any]],
    all_setter_sites: List[Dict[str, Any]],
) -> None:
    """Narrow cpp_trace['dep_vars'] to only dep_vars from matched setter edges.

    When ``target_setters`` reduces the setter-site list, the trace's dep_vars
    still cover ALL setter edges in the file.  This recomputes dep_vars from
    only the edges whose (line, expression) key matches a kept setter site,
    so the BFS queue in Pass 2 is limited to variables that actually feed the
    target setter(s).  Guard-condition dep_vars (_gc_dep_vars) are preserved.

    Mutates *cpp_trace* in place (``dep_vars`` field).
    """
    if not matched_setter_sites or not cpp_trace:
        return
    if len(matched_setter_sites) >= len(all_setter_sites):
        return  # nothing was filtered out

    # Build matched keys from kept setter sites
    matched_keys: Set[Tuple[Any, str]] = set()
    for site in matched_setter_sites:
        matched_keys.add((site.get("line"), str(site.get("expression") or "")))

    node_lookup = {
        str(n.get("id") or ""): n
        for n in (cpp_trace.get("nodes") or [])
        if n.get("id")
    }

    dep_vars: List[str] = []
    seen: Set[str] = set()
    for e in (cpp_trace.get("edges") or []):
        rel = str(e.get("relation") or "").lower()
        if rel not in _CPP_SETTER_RELATIONS:
            continue
        edge_key = (e.get("line"), str(e.get("expression") or ""))
        if edge_key not in matched_keys:
            continue
        src = str(e.get("source") or "").strip()
        if src:
            name = _normalize_cpp_dep_name(node_lookup.get(src), src)
            if name and name not in seen:
                seen.add(name)
                dep_vars.append(name)

    # Preserve guard-condition dep_vars (independent of specific setters)
    for gc_dv in (cpp_trace.get("_gc_dep_vars") or []):
        if gc_dv and gc_dv not in seen:
            seen.add(gc_dv)
            dep_vars.append(gc_dv)

    original_count = len(cpp_trace.get("dep_vars") or [])
    cpp_trace["dep_vars"] = dep_vars
    logger.info(
        "[BACKWARD] target_setters dep_var scoping: %d → %d dep_vars "
        "(matched %d/%d setter sites)",
        original_count, len(dep_vars),
        len(matched_setter_sites), len(all_setter_sites),
    )


def _flatten_subroutine_expansions(expansions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    queue: Deque[Dict[str, Any]] = deque(expansions or [])
    while queue:
        cur = queue.popleft()
        out.append(cur)
        queue.extend(cur.get("subroutine_expansions") or [])
    return out


def _collect_expansion_graphs_and_links(
    expansions: List[Dict[str, Any]],
    graphs: List[Dict[str, Any]],
    link_edges: List[Dict[str, Any]],
) -> None:
    for ex in expansions or []:
        graphs.append(ex.get("call_graph", {}) or {})

        caller_block = str(ex.get("caller_block") or "")
        resolved = str(ex.get("resolved_subroutine") or "")
        line_no = int(ex.get("line") or 0)
        if caller_block and resolved:
            link_edges.append(
                {
                    "source": caller_block,
                    "target": resolved,
                    "type": "SUBROUTINE_CALL",
                    "direction": ["backward"],
                    "scopes": [f"{caller_block}:{line_no}" if line_no else caller_block],
                }
            )

        _collect_expansion_graphs_and_links(
            expansions=list(ex.get("subroutine_expansions") or []),
            graphs=graphs,
            link_edges=link_edges,
        )


def _merged_call_graph_with_subroutines(trace_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    graphs: List[Dict[str, Any]] = []
    subroutine_link_edges: List[Dict[str, Any]] = []

    for traced in trace_results:
        graphs.append(traced.get("call_graph", {}) or {})
        _collect_expansion_graphs_and_links(
            expansions=list(traced.get("subroutine_expansions") or []),
            graphs=graphs,
            link_edges=subroutine_link_edges,
        )

    if subroutine_link_edges:
        graphs.append({"nodes": [], "edges": subroutine_link_edges})
    return merge_call_graphs(graphs)


def _subroutine_summary_from_trace_results(trace_results: List[Dict[str, Any]]) -> Dict[str, int]:
    all_expansions: List[Dict[str, Any]] = []
    for traced in trace_results:
        all_expansions.extend(_flatten_subroutine_expansions(list(traced.get("subroutine_expansions") or [])))
    return {
        "count": len(all_expansions),
        "unresolved": sum(1 for e in all_expansions if str(e.get("resolution_confidence") or "") == "unresolved"),
        "capped": sum(1 for e in all_expansions if bool(e.get("capped"))),
    }


def _aggregate_dep_var_collection_blocks(
    trace_results: List[Dict[str, Any]],
) -> Dict[str, Set[str]]:
    """Walk trace_results + subroutine_expansions; return {dep_var_upper: {block_id, ...}}."""
    out: Dict[str, Set[str]] = {}
    queue: Deque[Dict[str, Any]] = deque(trace_results or [])
    while queue:
        item = queue.popleft()
        for node in (item.get("call_graph", {}) or {}).get("nodes", []) or []:
            block_id = str(node.get("id") or "")
            if not block_id:
                continue
            for dv in (node.get("dependent_variables") or []):
                dv_up = str(dv).upper()
                if dv_up:
                    out.setdefault(dv_up, set()).add(block_id)
        queue.extend(list(item.get("subroutine_expansions") or []))
    return out


def _aggregate_setter_step_blocks(
    trace_results: List[Dict[str, Any]],
) -> Set[str]:
    """Return block IDs that contain at least one SETTER_STEP in relevant_code_lines.

    An upstream Pass 1 backward trace records SETTER_STEPs in relevant_code_lines
    when it finds the root variable being set inside a predecessor block (e.g.
    ``MVC WK_RRC,ZEROS`` at DA7_0050 in da76).  That block is the *actual setter
    scope*, but ``_aggregate_dep_var_collection_blocks`` only reads
    ``dependent_variables`` — it never surfaces the setter block itself into
    ``dep_var_collection_map[root_var]``.

    GAP-ZEROS-SCOPE: by returning the setter-step blocks here, callers can seed
    ``dep_var_collection_map[root_var][stem]`` so that the dep_var BFS scope
    includes initialisation blocks like ``MVC WK_RRC,ZEROS``, which are reached
    before (not after) the comparison/use sites that anchor the collection-based
    scope.
    """
    out: Set[str] = set()
    queue: Deque[Dict[str, Any]] = deque(trace_results or [])
    while queue:
        item = queue.popleft()
        for node in (item.get("call_graph", {}) or {}).get("nodes", []) or []:
            block_id = str(node.get("id") or "")
            if not block_id:
                continue
            for entry in (node.get("relevant_code_lines") or []):
                if str(entry.get("role") or "") == "SETTER_STEP":
                    out.add(block_id)
                    break
        queue.extend(list(item.get("subroutine_expansions") or []))
    return out


# ── P37: FLIPC data-level → CE1CRx alias tables ──────────────────────────────
# FLIPC Dn,Dm swaps ECB data levels n and m.  After the instruction the block
# previously held at CE1CRn is accessible via CE1CRm and vice versa.
# These tables let _collect_flipc_level_aliases() convert FLIPC operands (level
# designators like "D0", "DA") to the canonical CE1CRx field names used in ASM.
_LEVEL_TO_CE1CR: Dict[str, str] = {
    "D0": "CE1CR0", "D1": "CE1CR1", "D2": "CE1CR2", "D3": "CE1CR3",
    "D4": "CE1CR4", "D5": "CE1CR5", "D6": "CE1CR6", "D7": "CE1CR7",
    "D8": "CE1CR8", "D9": "CE1CR9", "DA": "CE1CRA", "DB": "CE1CRB",
    "DC": "CE1CRC", "DD": "CE1CRD", "DE": "CE1CRE", "DF": "CE1CRF",
}


def _collect_flipc_level_aliases(
    bp_data: Dict[str, Any],
    scope_blocks: Set[str],
    scope_line_caps: Optional[Dict[str, int]] = None,  # NEW
) -> Dict[str, Set[str]]:
    """Return CE1CRx bidirectional aliases introduced by FLIPC in scope_blocks.

    FLIPC Dn,Dm swaps data levels n and m, so a write to CE1CRn before the
    FLIPC is effectively a write to CE1CRm after it — and vice versa.

    Returns ``{CE1CRx_upper: {aliased_CE1CRy_upper, ...}}``.  Both sides of
    every FLIPC in scope produce entries so the mapping is bidirectional.

    Conservative approach: any FLIPC anywhere inside scope_blocks creates a
    permanent alias for the full scope.  A future CFG-sensitive implementation
    could restrict the alias to post-FLIPC successor blocks only; the current
    approximation errs on the side of surfacing more setter sites.
    """
    aliases: Dict[str, Set[str]] = {}
    if not scope_blocks:
        return aliases
    for block in _iter_scope_blocks(bp_data, scope_blocks):
        block_id_fl = str(block.get("id") or "")
        _line_cap_fl = (scope_line_caps or {}).get(block_id_fl, 0) or 0
        for flow in iter_flow(block):
            if _line_cap_fl:
                _e_line_fl = int(flow.get("line") or 0)
                if _e_line_fl and _e_line_fl >= _line_cap_fl:
                    continue
            if str(flow.get("inst") or "").upper() != "FLIPC":
                continue
            args = [str(a).strip().upper() for a in (flow.get("args") or [])]
            if len(args) < 2:
                continue
            src_ce1cr = _LEVEL_TO_CE1CR.get(args[0])
            dst_ce1cr = _LEVEL_TO_CE1CR.get(args[1])
            if src_ce1cr and dst_ce1cr:
                aliases.setdefault(src_ce1cr, set()).add(dst_ce1cr)
                aliases.setdefault(dst_ce1cr, set()).add(src_ce1cr)
    return aliases


def _collect_detac_attac_flag_writes(
    bp_data: Dict[str, Any],
    scope_blocks: Set[str],
    asm_file: Optional[Path],
    asm_dir: Optional[Path],
    scope_line_caps: Optional[Dict[str, int]] = None,  # NEW
) -> Dict[str, List[Dict[str, Any]]]:
    """Return implicit CE1DTn setter sites from DETAC/ATTAC instructions in scope_blocks.

    P15: DETAC Dn transfers ownership of data level n away from the ECB, implicitly
    writing CE1DTn (the detach state byte) to mark the level as detached.
    P16: ATTAC Dn reattaches ownership, implicitly updating CE1DTn.

    Returns {CE1DTn_upper: [setter_site_dict, ...]} for any DETAC/ATTAC found in scope.
    The caller appends matching sites to setter_sites when the trace target is CE1DTn.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}
    if not scope_blocks:
        return results
    for block in _iter_scope_blocks(bp_data, scope_blocks):
        block_id = str(block.get("id") or "")
        _line_cap_da = (scope_line_caps or {}).get(block_id, 0) or 0
        for flow in iter_flow(block):
            if _line_cap_da:
                _e_line_da = int(flow.get("line") or 0)
                if _e_line_da and _e_line_da >= _line_cap_da:
                    continue
            inst = str(flow.get("inst") or "").upper()
            if inst not in LEVEL_TRANSFER_INST:
                continue
            args = [str(a).strip().upper() for a in (flow.get("args") or [])]
            if not args:
                continue
            ce1dt = LEVEL_TO_CE1DT.get(args[0])
            if not ce1dt:
                continue
            line_no = int(flow.get("line") or 0)
            raw_inst_args = ", ".join(str(a) for a in (flow.get("args") or []))
            raw_line = fetch_raw_line(str(asm_file), line_no, asm_dir) if asm_file else None
            site: Dict[str, Any] = {
                "routine": block_id,
                "line": line_no,
                "instruction": inst,
                "call_type": "implicit_setter",
                "raw_line": raw_line or f"{inst} {raw_inst_args}".strip(),
                "implicit_target": ce1dt,
            }
            results.setdefault(ce1dt, []).append(site)
    return results


_LEVEL_ALLOC_AND_FIND: frozenset = frozenset(LEVEL_ALLOC_INST | FILE_FIND_INST)
_LEVEL_KW_PREFIX: str = "LEVEL="


def _extract_level_from_args(args: List[str]) -> Optional[str]:
    """Return the level designator (D0–DF) from an instruction's args list.

    Handles two syntax forms:
    - Positional: GETCC D2,L4 → args[0] = "D2"
    - Keyword:    FINHC LEVEL=DE,FILE=X → extract "LEVEL=DE" → "DE"
    """
    for a in args:
        if a.upper().startswith(_LEVEL_KW_PREFIX):
            return a.upper()[len(_LEVEL_KW_PREFIX):]
    return args[0] if args else None


def _collect_level_alloc_setter_sites(
    bp_data: Dict[str, Any],
    scope_blocks: Set[str],
    asm_file: Optional[Path],
    asm_dir: Optional[Path],
    scope_line_caps: Optional[Dict[str, int]] = None,  # NEW
) -> Dict[str, List[Dict[str, Any]]]:
    """Return implicit CE1CRn setter sites from GETCC/GETFC/MALOC/FIWHC/FINHC etc.

    P14: GETCC Dn allocates a core data level and implicitly sets CE1CRn to the
    allocated block pointer (the macro expansion loads R14 and stores it into CE1CRn).
    GETFC similarly handles file-level allocation.  MALOC is a generic level allocator.

    P43: FIWHC/FINHC/FINDC/FINWC (FILE_FIND_INST) simultaneously perform a file read
    AND allocate a core storage block on the specified data level.  The level designator
    is passed as a LEVEL=Dn keyword arg (e.g. FINHC LEVEL=DE,FILE=TDRFILE).
    `_extract_level_from_args` handles both positional (GETCC D2) and keyword
    (FINHC LEVEL=DE) forms so both families map correctly to their CE1CRn slot.

    Returns {CE1CRn_upper: [setter_site_dict, ...]} for allocation instructions found
    in scope.  The caller appends matching sites when the trace target is CE1CRn.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}
    if not scope_blocks:
        return results
    for block in _iter_scope_blocks(bp_data, scope_blocks):
        block_id = str(block.get("id") or "")
        _line_cap_la = (scope_line_caps or {}).get(block_id, 0) or 0
        for flow in iter_flow(block):
            if _line_cap_la:
                _e_line_la = int(flow.get("line") or 0)
                if _e_line_la and _e_line_la >= _line_cap_la:
                    continue
            inst = str(flow.get("inst") or "").upper()
            if inst not in _LEVEL_ALLOC_AND_FIND:
                continue
            args = [str(a).strip().upper() for a in (flow.get("args") or [])]
            level_key = _extract_level_from_args(args)
            if not level_key:
                continue
            ce1cr = _LEVEL_TO_CE1CR.get(level_key)
            if not ce1cr:
                continue
            line_no = int(flow.get("line") or 0)
            raw_inst_args = ", ".join(str(a) for a in (flow.get("args") or []))
            raw_line = fetch_raw_line(str(asm_file), line_no, asm_dir) if asm_file else None
            site: Dict[str, Any] = {
                "routine": block_id,
                "line": line_no,
                "instruction": inst,
                "call_type": "implicit_setter",
                "raw_line": raw_line or f"{inst} {raw_inst_args}".strip(),
                "implicit_target": ce1cr,
            }
            results.setdefault(ce1cr, []).append(site)
    return results


def _collect_db_managed_field_writes(
    bp_data: Dict[str, Any],
    scope_blocks: Set[str],
    asm_file: Optional[Path],
    asm_dir: Optional[Path],
    scope_line_caps: Optional[Dict[str, int]] = None,  # NEW
) -> Dict[str, List[Dict[str, Any]]]:
    """Return implicit setter sites for DB-managed work area fields (P19: DBSPA→SW00WKA).

    P19: DBSPA REF=file allocates an initialized work area scoped to the open DB context
    and implicitly writes its address into SW00WKA (the sw00sr->sw00wka field).  Code then
    accesses work area fields directly via USING SW00SR,R3 without any explicit store to
    SW00WKA.

    Returns {field_upper: [setter_site_dict, ...]} for matching instructions found in scope.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}
    if not scope_blocks or not DB_MANAGED_FIELD_WRITES:
        return results
    db_managed_upper = {str(k).upper(): str(v).upper() for k, v in DB_MANAGED_FIELD_WRITES.items()}
    for block in _iter_scope_blocks(bp_data, scope_blocks):
        block_id = str(block.get("id") or "")
        _line_cap_db = (scope_line_caps or {}).get(block_id, 0) or 0
        for flow in iter_flow(block):
            if _line_cap_db:
                _e_line_db = int(flow.get("line") or 0)
                if _e_line_db and _e_line_db >= _line_cap_db:
                    continue
            inst = str(flow.get("inst") or "").upper()
            target_field = db_managed_upper.get(inst)
            if not target_field:
                continue
            line_no = int(flow.get("line") or 0)
            raw_inst_args = ", ".join(str(a) for a in (flow.get("args") or []))
            raw_line = fetch_raw_line(str(asm_file), line_no, asm_dir) if asm_file else None
            site: Dict[str, Any] = {
                "routine": block_id,
                "line": line_no,
                "instruction": inst,
                "call_type": "implicit_setter",
                "raw_line": raw_line or f"{inst} {raw_inst_args}".strip(),
                "implicit_target": target_field,
            }
            results.setdefault(target_field, []).append(site)
    return results


def _collect_macro_fixed_result_writes(
    bp_data: Dict[str, Any],
    scope_blocks: Set[str],
    asm_file: Optional[Path],
    asm_dir: Optional[Path],
    scope_line_caps: Optional[Dict[str, int]] = None,  # NEW
) -> Dict[str, List[Dict[str, Any]]]:
    """Return implicit setter sites for macros that always write a fixed ECB result field.

    P42: TMSMA (Table Management System Macro) always writes its return code to EBL8URTN
    regardless of the table ID or function type.  When code does:
        TMSMA GM,XH881857,FUNC=FIND,KEY=LGMKEY
    it implicitly writes EBL8URTN (which is then tested by TM EBL8URTN,X'01').

    Returns {field_upper: [setter_site_dict, ...]} for macros found in scope.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}
    if not scope_blocks or not MACRO_FIXED_RESULT_FIELD:
        return results
    macro_map_upper = {str(k).upper(): str(v).upper() for k, v in MACRO_FIXED_RESULT_FIELD.items()}
    for block in _iter_scope_blocks(bp_data, scope_blocks):
        block_id = str(block.get("id") or "")
        _line_cap_mf = (scope_line_caps or {}).get(block_id, 0) or 0
        for flow in iter_flow(block):
            if _line_cap_mf:
                _e_line_mf = int(flow.get("line") or 0)
                if _e_line_mf and _e_line_mf >= _line_cap_mf:
                    continue
            inst = str(flow.get("inst") or "").upper()
            target_field = macro_map_upper.get(inst)
            if not target_field:
                continue
            line_no = int(flow.get("line") or 0)
            raw_inst_args = ", ".join(str(a) for a in (flow.get("args") or []))
            raw_line = fetch_raw_line(str(asm_file), line_no, asm_dir) if asm_file else None
            site: Dict[str, Any] = {
                "routine": block_id,
                "line": line_no,
                "instruction": inst,
                "call_type": "implicit_setter",
                "raw_line": raw_line or f"{inst} {raw_inst_args}".strip(),
                "implicit_target": target_field,
            }
            results.setdefault(target_field, []).append(site)
    return results


_PRE_CALL_LOAD_WINDOW = 15  # maximum instructions before a cross-file call to scan


_CREEC_FAMILY: frozenset = frozenset({"CREEC", "CREMC", "CREDC"})


def _collect_pre_call_dep_vars(
    bp_data: Dict[str, Any],
    scope_blocks: Set[str],
) -> List[str]:
    """Collect named variables loaded into registers immediately before cross-file calls.

    P8/P30: Before ENTRC/CALLC/CREEC, callers pre-load registers with parameter area
    addresses or values:
        LA    R1,WB1XH89RN     POINT TO PARAMETER AREA → R1
        ENTRC XH89             CALL WITH R1 = PARAM ADDRESS

    P9: CREEC/CREMC/CREDC pass parameter block addresses in keyword args on the call
    instruction itself:
        CREEC PROGNM=CINFC,R14=PARM_AREA_A,R15=PARM_AREA_B

    Scans each scope block for cross-file call instructions (PRE_CALL_OPCODES).
    For each call found, looks backward up to _PRE_CALL_LOAD_WINDOW instructions
    in the same block for LA/L/LH/LY/LG/IC Rn,FIELD patterns.
    For CREEC-family calls, also extracts R14=/R15= keyword args directly.
    Returns the list of unique named FIELD variables discovered.
    """
    if not scope_blocks:
        return []
    pre_call_vars: List[str] = []
    seen: Set[str] = set()
    call_opcodes_upper: Set[str] = {str(op).upper() for op in PRE_CALL_OPCODES}

    for block in _iter_scope_blocks(bp_data, scope_blocks):
        flow_entries = sorted(
            [e for e in iter_flow(block) if int(e.get("line") or 0) > 0],
            key=lambda e: int(e.get("line") or 0),
        )
        for i, entry in enumerate(flow_entries):
            call_inst = str(entry.get("inst") or "").upper()
            if call_inst not in call_opcodes_upper:
                continue
            # P9: CREEC/CREMC/CREDC embed parameter addresses as R14=/R15= keyword args
            # on the call instruction itself rather than in preceding register pre-loads.
            if call_inst in _CREEC_FAMILY:
                call_entry_args = [str(a) for a in (entry.get("args") or [])]
                for kw in CREEC_PARAM_KEYWORDS:
                    val = _extract_keyword_arg(call_entry_args, kw)
                    if (val
                            and not looks_like_register_token(val)
                            and not _is_noise_dep_var(val)
                            and val not in seen):
                        seen.add(val)
                        pre_call_vars.append(val)
            # Found a cross-file call; scan backward in the same block
            window_start = max(0, i - _PRE_CALL_LOAD_WINDOW)
            for prev in flow_entries[window_start:i]:
                prev_inst = str(prev.get("inst") or "").upper()
                prev_args = [str(a) for a in (prev.get("args") or [])]
                # LA/LAY load address; REGISTER_LOAD_FROM_MEM loads value
                if prev_inst in {"LA", "LAY"}:
                    src_idx = 1
                elif prev_inst in REGISTER_LOAD_FROM_MEM:
                    src_idx = REGISTER_LOAD_FROM_MEM[prev_inst]
                else:
                    continue
                if len(prev_args) <= src_idx:
                    continue
                raw_src = prev_args[src_idx]
                mem_src = normalize_token(raw_src)
                if (mem_src
                        and not looks_like_register_token(mem_src)
                        and not _is_noise_dep_var(mem_src)
                        and mem_src not in seen):
                    seen.add(mem_src)
                    pre_call_vars.append(mem_src)
    return pre_call_vars


_RANGE_WRITE_PAT = re.compile(
    r"^([A-Z][A-Z_@$#][A-Z0-9_@$#]*)(\d{1,4})\((\d+)\)$",
    re.IGNORECASE,
)


def _range_write_covers_target(raw_dest: str, target: str) -> bool:
    """Return True when raw_dest is a BASE(LENGTH) range-write that covers target.

    z/TPF HLASM instructions often write multi-byte regions using ``BASE(LENGTH)``
    operand syntax.  When variable names encode the byte offset in a fixed-width
    numeric suffix (e.g. ``EBW002`` = byte offset 2 of the EBW working area),
    coverage can be detected without a DSECT layout:

        UNPK EBW000(4),EBW005(3)  →  raw_dest = "EBW000(4)"
        target = "EBW002"
        base_offset=0, tgt_offset=2, length=4  →  0 < 2 < 4  →  True

    Pattern guards (to avoid false positives on block labels and registers):
    - raw_dest must match ``PREFIX + DIGITS(1-4) + (LENGTH)``
    - PREFIX must have ≥ 2 non-digit leading characters
    - target must share the same PREFIX and end with same-width all-digit suffix
    - Offset difference must be strictly positive and less than length

    GAP-RANGE-WRITE: closes the multi-byte range-write coverage gap for
    instructions like UNPK that write interior bytes under an offset-encoded name.
    """
    m = _RANGE_WRITE_PAT.match(raw_dest.strip().upper())
    if not m:
        return False
    base_prefix, base_num_str, length_str = m.group(1), m.group(2), m.group(3)
    tgt = target.upper()
    if not tgt.startswith(base_prefix):
        return False
    tgt_num_str = tgt[len(base_prefix):]
    if not tgt_num_str.isdigit() or len(tgt_num_str) != len(base_num_str):
        return False
    try:
        base_off = int(base_num_str)
        tgt_off = int(tgt_num_str)
        length = int(length_str)
    except ValueError:
        return False
    diff = tgt_off - base_off
    return 0 < diff < length


def _extract_keyword_arg(args: List[str], keyword: str) -> Optional[str]:
    """Extract the value of a keyword argument (e.g. KEYLIST=VALUE) from an args list.

    Scans args for an entry whose upper-case prefix matches ``keyword=`` and returns
    the normalised token after the ``=``.  Returns None if not found or if the value
    looks like a register token.
    """
    kw = str(keyword).upper() + "="
    for arg in args:
        upper_arg = str(arg).upper()
        if upper_arg.startswith(kw):
            raw_val = str(arg)[len(kw):]
            tok = normalize_token(raw_val)
            if tok and not looks_like_register_token(tok):
                return tok
    return None


def _find_scoped_setter_sites(
    variable: str,
    bp_data: Dict[str, Any],
    asm_file: Path,
    scope_blocks: Set[str],
    asm_dir: Optional[Path],
    scope_line_caps: Optional[Dict[str, int]] = None,  # NEW
    constant_symbols: Optional[Set[str]] = None,  # Issue-5
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Find setter sites for variable restricted to scope_blocks."""
    target = str(variable or "").upper()

    # P37: collect FLIPC-derived CE1CRx aliases within scope.
    # FLIPC Dn,Dm swaps data levels, so a write to CE1CRn is visible via CE1CRm
    # after the FLIPC.  Bidirectional aliases mean we find setters on either side.
    flipc_aliases = _collect_flipc_level_aliases(bp_data, scope_blocks, scope_line_caps=scope_line_caps)
    match_names: Set[str] = {target} | flipc_aliases.get(target, set())

    setter_sites: List[Dict[str, Any]] = []
    scoped_warnings: List[str] = []
    for block in _iter_scope_blocks(bp_data, scope_blocks):
        block_id = str(block.get("id") or "")
        _line_cap = (scope_line_caps or {}).get(block_id, 0) or 0
        for flow in iter_flow(block):
            if _line_cap:
                _e_line = int(flow.get("line") or 0)
                if _e_line and _e_line >= _line_cap:
                    continue
            inst = str(flow.get("inst") or "").upper()
            args = [str(a) for a in (flow.get("args") or [])]
            # P20: DBKEY KEYLIST=VAR fills the named key list work area.
            # DBKEY is not in SETTER_INST because it uses keyword operands (not a named
            # destination at a fixed arg index).  Extract the KEYLIST= value explicitly.
            if inst in DB_KEYLIST_INST:
                keylist_var = _extract_keyword_arg(args, "KEYLIST")
                if keylist_var and keylist_var.upper() in match_names:
                    line_no = int(flow.get("line") or 0)
                    raw_line = fetch_raw_line(str(asm_file), line_no, asm_dir)
                    detail = ", ".join(args)
                    setter_sites.append({
                        "routine": block_id,
                        "line": line_no,
                        "instruction": inst,
                        "call_type": "setter_site",
                        "raw_line": raw_line or f"{inst} {detail}".strip(),
                    })
                continue  # DBKEY fully handled above; skip SETTER_INST path
            # EX/EXRL expansion: resolve the target instruction at the label
            # and substitute inst/args so setter detection works on the real op.
            # EX R3,MYMVC → resolves MYMVC MVC DEST,SRC; the EX OR modifies
            # only the length byte, not operand addresses.
            if inst in EXECUTE_INST:
                _ex_label = args[1] if len(args) > 1 else ""
                _ex_target_entry = resolve_ex_target(bp_data, normalize_token(_ex_label))
                if _ex_target_entry:
                    inst = str(_ex_target_entry.get("inst") or "").upper()
                    args = [str(a) for a in (_ex_target_entry.get("args") or [])]
                else:
                    continue  # cannot resolve EX target — skip
            if inst not in SETTER_INST:
                continue
            lhs_from_var = normalize_token(str(flow.get("var") or ""))
            if lhs_from_var:
                lhs = lhs_from_var
                raw_dest = lhs_from_var
            else:
                dest_idx = DEST_ARG_INDEX_BY_INST.get(inst, 0)
                raw_dest = args[dest_idx] if len(args) > dest_idx else ""
                lhs = normalize_token(raw_dest)
            # GAP-RANGE-WRITE: also accept range-write instructions like UNPK EBW000(4)
            # whose normalized LHS is EBW000 but whose written range covers the target
            # (e.g. EBW002 at offset 2 is within [EBW000, EBW000+4)).
            if lhs not in match_names and not _range_write_covers_target(raw_dest, target):
                continue
            # Audit §3a/§3b/§3c/§3d: value-preserving / spec-exception / zero-mask /
            # nonzero-displacement writes are NOT setter sites for the target — skip
            # them (and surface any warning) instead of emitting a false modifier.
            _verdict = classify_setter(inst, args)
            if _verdict.get("skip_setter_site"):
                _w = _verdict.get("warn")
                if _w:
                    scoped_warnings.append(f"{block_id}: {_w}")
                continue

            # Derive setter_expression — the source operand being written to the target.
            # Blueprint stores the parsed source in flow["val"] when identifiable
            # (MVC, MVI, PACK, etc.).  For register-store instructions the blueprint
            # uses only flow["args"] and "val" is absent; derive from args.
            dest_idx = DEST_ARG_INDEX_BY_INST.get(inst, 0)
            _val = flow.get("val")
            if _val is not None:
                _setter_expr: Optional[str] = _strip_hlasm_remark_from_val(str(_val))
            elif dest_idx == 1:  # ST/STH/STC/CVD/STG… — src register is args[0]
                _setter_expr = args[0] if args else None
            elif dest_idx == 2:  # STM/STMG/STCM — register range Rlo-Rhi
                _setter_expr = f"{args[0]}-{args[1]}" if len(args) >= 2 else (args[0] if args else None)
            else:
                # args[1] can contain an embedded remark (e.g. XC FIELD,FIELD CLEAR...)
                _setter_expr = _strip_hlasm_remark_from_val(args[1]) if len(args) > 1 else None

            line_no = int(flow.get("line") or 0)
            raw_line = fetch_raw_line(str(asm_file), line_no, asm_dir)
            detail = ", ".join(args) if args else ""
            site: Dict[str, Any] = {
                "routine": block_id,
                "line": line_no,
                "instruction": inst,
                "call_type": "setter_site",
                "raw_line": raw_line or f"{inst} {detail}".strip(),
                "setter_expression": _setter_expr,
            }
            # Issue-5 / GAP-028: flag sites whose source is a compile-time constant
            # so the backward tracer can skip runtime dep_var BFS for them.
            # Priority:
            #   1. Inline literals (=C'..', =X'..', =F'..') — always constant.
            #   2. XC VAR,VAR — self-clear zeroes the destination (Bug-5B fix):
            #      setter_expression equals the target → no runtime source dep_var.
            #   3. SI/SIY/SIL immediate instructions (OI, NI, XI, MVI, ASI, MVHI…):
            #      the second operand is ALWAYS a compile-time immediate, even when
            #      defined as an EQU in a COPY member absent from the symbol table.
            #   4. Symbol-table lookup — covers DC/EQU labels in local/global syms.
            if _setter_expr and str(_setter_expr).startswith("="):
                site["constant_source"] = True
            elif (inst == "XC" and _setter_expr
                  and normalize_token(str(_setter_expr)).upper() == normalize_token(target).upper()):
                site["constant_source"] = True
            elif inst in IMMEDIATE_OPERAND_SETTER_INST and dest_idx == 0:
                # setter_expr = args[1] for dest_idx==0; that's always an immediate
                site["constant_source"] = True
            elif constant_symbols and _setter_expr:
                _norm_expr = normalize_token(str(_setter_expr))
                if _norm_expr and _norm_expr.upper() in constant_symbols:
                    site["constant_source"] = True
            # EQU constant heuristic: when the setter_expression looks like an
            # EQU constant from a COPY member (e.g. #ACPDBOK, LG#C400) and the
            # instruction uses an immediate operand, flag as constant_source.
            if not site.get("constant_source") and inst in IMMEDIATE_OPERAND_SETTER_INST and _setter_expr:
                if looks_like_equ_constant(normalize_token(str(_setter_expr))):
                    site["constant_source"] = True
            # P37: annotate setter sites discovered via a FLIPC-derived alias so
            # consumers know the link is indirect (level swap).
            if lhs != target and lhs in match_names:
                site["flipc_alias_of"] = target
            # Audit §3a/§3b: prior value of the destination is itself a dependency
            # (read-modify-write, doubling, shift, etc.).
            if _verdict.get("prior_value_required"):
                site["prior_value_required"] = True
            # Audit §3c: hardware-sourced stores (TOD clock, facility bits).
            if inst in _HARDWARE_SOURCE_BY_INST:
                site["hardware_source"] = _HARDWARE_SOURCE_BY_INST[inst]
            setter_sites.append(site)

    # P14: implicit CE1CRn writes from GETCC/GETFC/MALOC (level allocation macros).
    # GETCC Dn allocates a data level and implicitly sets CE1CRn via macro expansion.
    level_alloc_sites = _collect_level_alloc_setter_sites(bp_data, scope_blocks, asm_file, asm_dir, scope_line_caps=scope_line_caps)
    if target in level_alloc_sites:
        setter_sites.extend(level_alloc_sites[target])

    # P15-P16: implicit CE1DTn writes from DETAC/ATTAC (level transfer instructions).
    # DETAC Dn implicitly writes CE1DTn to mark the level as detached.
    detac_sites = _collect_detac_attac_flag_writes(bp_data, scope_blocks, asm_file, asm_dir, scope_line_caps=scope_line_caps)
    if target in detac_sites:
        setter_sites.extend(detac_sites[target])

    # P19: implicit SW00WKA write from DBSPA (DB-managed work area allocation).
    db_spa_sites = _collect_db_managed_field_writes(bp_data, scope_blocks, asm_file, asm_dir, scope_line_caps=scope_line_caps)
    if target in db_spa_sites:
        setter_sites.extend(db_spa_sites[target])

    # P42: implicit EBL8URTN write from TMSMA (table management system macro).
    macro_sites = _collect_macro_fixed_result_writes(bp_data, scope_blocks, asm_file, asm_dir, scope_line_caps=scope_line_caps)
    if target in macro_sites:
        setter_sites.extend(macro_sites[target])

    return _dedupe_sites(setter_sites), scoped_warnings


def _build_dep_var_file_result(
    dep_var: str,
    bp_data: Dict[str, Any],
    bp_path: Path,
    asm_file: Path,
    scope_blocks: Set[str],
    scope_label: str,
    asm_dir: Optional[Path],
    max_depth: int,
    max_subroutine_depth: int,
    max_subroutine_nodes: int,
    max_trace_nodes: int,
    scope_line_caps: Optional[Dict[str, int]] = None,  # NEW
) -> Dict[str, Any]:
    """Trace dep_var setters within scope_blocks and return a file result dict."""
    # Issue-5: pre-compute constant symbols once so _find_scoped_setter_sites can
    # flag setters whose source operand is a compile-time constant (DC/EQU label).
    _const_syms = collect_constant_symbols(bp_data, asm_file=asm_file if asm_file and asm_file.exists() else None, bp_path=bp_path)
    setter_sites, warnings = _find_scoped_setter_sites(
        dep_var, bp_data, asm_file, scope_blocks, asm_dir,
        scope_line_caps=scope_line_caps, constant_symbols=_const_syms,
    )
    trace_results: List[Dict[str, Any]] = []
    for site in setter_sites:
        traced = trace_asm_call_site_backward(
            variable=dep_var,
            blueprint_path=bp_path,
            asm_file=asm_file,
            call_site=site,
            max_depth=max_depth,
            max_subroutine_depth=max_subroutine_depth,
            max_subroutine_nodes=max_subroutine_nodes,
            max_trace_nodes_per_site=max_trace_nodes,
        )
        trace_results.append(traced)
        warnings.extend(traced.get("warnings") or [])
    return {
        "file_type": "asm",
        "scope": scope_label,
        "setter_sites": setter_sites,
        "setter_site_traces": trace_results,
        "call_graph": _merged_call_graph_with_subroutines(trace_results),
        "subroutine_summary": _subroutine_summary_from_trace_results(trace_results),
        "warnings": sorted(set(str(w) for w in warnings if w)),
    }


def _is_noise_dep_var(name: str) -> bool:
    """Return True for C++ tracer infrastructure tokens that are not real variable names.

    These tokens exhaust the max_dep_vars budget with meaningless entries:
      return@getcc, alloc@D7:87, level:D4, memory:HUBGL, etc.
    FIX-new-11: filter them at every insertion point into dep_var_collection_map.

    GAP-038: The original broad "@ in n" rule incorrectly caught valid HLASM names
    with an @ *prefix* (e.g. @CCIBIND, @TPFCV82 which are legitimate z/TPF symbols).
    The rule is narrowed: only reject tokens where @ follows an alphanumeric character
    (C++ style "name@address" separators), or match the explicit infrastructure prefixes.

    Bug-5A fix: also filter bare register numbers (1–15) and register name tokens
    (R0–R15) that leak from BAS/BASR register-clobber analysis into dep_var lists.
    """
    n = name.upper()
    # C++ infrastructure: @ appears as a name separator (e.g. return@func, alloc@D7)
    _has_interior_at = len(n) > 1 and "@" in n[1:]
    # Bare register numbers (1–15) and Rx names (R0–R15) — never valid variable names.
    _is_register = (n.isdigit() and 0 < int(n) <= 15) or bool(re.match(r'^R(1[0-5]|[0-9])$', n))
    return (
        _has_interior_at
        or _is_register
        or n.startswith("LEVEL:")
        or n.startswith("MEMORY:")
        or n.startswith("ALLOC@")
        or n.startswith("RETURN@")
        # GAP-P17: synthetic DB_CONTEXT_* labels emitted by _backtrack_register_dep_vars
        # when R3 is traced back to a DBOPN/DBIFB/DBRED instruction.  These are
        # informational annotations (not real variable names) and must not be traced
        # as dep_vars — they would never match a SETTER_INST write.
        or n.startswith("DB_CONTEXT_")
    )


def _build_child_dep_var_map(
    chain_file_results: Dict[str, Any],
    downstream_file_results: Dict[str, Any],
) -> Dict[str, Dict[str, Set[str]]]:
    """Collect dep_vars from a dep_var's own trace results → feeds next BFS level.

    FIX 20: also reads lineage_trace.dep_vars for C++ file results so that
    multi-depth C++ dep_var traversal works end-to-end.  bridge_back_to_asm:*
    entries are excluded here (GAP B handles register routing separately).
    """
    child_map: Dict[str, Dict[str, Set[str]]] = {}
    for file_stem, file_result in {**chain_file_results, **downstream_file_results}.items():
        # ASM path: dep_vars come from setter_site_traces → dependent_variables fields
        traces = file_result.get("setter_site_traces") or []
        per_file = _aggregate_dep_var_collection_blocks(traces)
        for dv, blocks in per_file.items():
            # GAP-038: apply same noise filter as C++ path — @CCIBIND, @TPFCV82 etc.
            # come from dependent_variables in ASM setter_site_traces and bypass the
            # insertion-point filter at dep_var_collection_map build time.
            if _is_noise_dep_var(dv):
                continue
            child_map.setdefault(dv, {})[file_stem] = blocks

        # FIX 20: C++ path — dep_vars come from lineage_trace.dep_vars
        if file_result.get("file_type") == "cpp":
            cpp_trace = file_result.get("lineage_trace") or {}
            for dv in (cpp_trace.get("dep_vars") or []):
                dv_str = str(dv or "").strip()
                # Exclude bridge-register tokens — handled by GAP B
                if dv_str and not dv_str.startswith("bridge_back_to_asm:"):
                    # FIX-new-1: uppercase for consistency with T2 insertion path
                    # FIX-new-11: drop infrastructure noise tokens
                    dv_up = dv_str.upper()
                    if not _is_noise_dep_var(dv_up):
                        child_map.setdefault(dv_up, {}).setdefault(file_stem, set())

    return child_map


def _build_block_parent_map(
    bp_data: Dict[str, Any],
) -> Dict[str, Tuple[str, Optional[int]]]:
    """Return {child_block_upper → (parent_block_id, call_line)} for BAS and FALLTHROUGH.

    Two relationships are captured:

    * **BAS**: a block containing ``BAS Rn,LABEL`` is the parent of LABEL;
      ``call_line`` is the BAS instruction's line number.
    * **FALLTHROUGH**: when block B immediately follows block A in source order
      and A does not end with an unconditional transfer (B, BR, BCR, ENTNC …),
      B is a FALLTHROUGH child of A; ``call_line`` is ``None``.

    TMSMA return blocks (e.g. ``FIN_03_TMSEND``) fall into the FALLTHROUGH
    category: the blueprint stores TMSMA with only its table arg, not the
    return label, so the return block simply appears as the next sequential
    block after the TMSMA instruction.
    """
    blocks = bp_data.get("blocks") or []

    # Sort blocks by first flow-entry line to establish source order.
    block_order: List[Tuple[int, str]] = []
    for block in blocks:
        blk_id = str(block.get("id") or "")
        first_line = 0
        for fl in iter_flow(block):
            ln = int(fl.get("line") or 0)
            if ln > 0:
                first_line = ln
                break
        if blk_id:
            block_order.append((first_line, blk_id))
    block_order.sort()

    parent_map: Dict[str, Tuple[str, Optional[int]]] = {}

    # 1. Explicit BAS calls.
    for block in blocks:
        blk_id = str(block.get("id") or "")
        for fl in iter_flow(block):
            if str(fl.get("inst") or "").upper() == "BAS":
                args = [str(a) for a in (fl.get("args") or [])]
                if len(args) >= 2:
                    target_upper = args[1].strip().upper()
                    call_ln: Optional[int] = int(fl.get("line") or 0) or None
                    if target_upper not in parent_map:
                        parent_map[target_upper] = (blk_id, call_ln)

    # 2. FALLTHROUGH: preceding block ends without an unconditional transfer.
    _UNCONDITIONAL: Set[str] = {"B", "BR", "BCR", "BC", "ENTNC", "RETURN", "BALR"}
    block_last_inst: Dict[str, str] = {}
    for block in blocks:
        blk_id = str(block.get("id") or "")
        last_inst = ""
        for fl in iter_flow(block):
            inst = str(fl.get("inst") or "").upper()
            if inst:
                last_inst = inst
        if blk_id:
            block_last_inst[blk_id] = last_inst

    for i in range(1, len(block_order)):
        _, curr_id = block_order[i]
        _, prev_id = block_order[i - 1]
        curr_upper = curr_id.upper()
        if curr_upper in parent_map:
            continue  # already has an explicit BAS parent
        if block_last_inst.get(prev_id, "").upper() not in _UNCONDITIONAL:
            parent_map[curr_upper] = (prev_id, None)

    return parent_map


def _collect_site_guards(
    bp_data: Dict[str, Any],
    block_id: str,
    setter_line: int,
) -> List[str]:
    """Return TRIGGER+BRANCH guard strings before setter_line in block_id.

    Also inherits guards from the immediate parent block (via BAS or FALLTHROUGH)
    so that conditions established before entering the current sub-block are
    included.  Example: the basic-card check in FIN_03GLOS is inherited by the
    TMSMA return block FIN_03_TMSEND that falls through from it.

    Returns plain ``"TRIGGER / BRANCH"`` strings in execution order (parent
    guards first, then local guards).
    """
    def _scan(target_id: str, limit: int) -> List[str]:
        result: List[str] = []
        pending: Optional[str] = None
        block = _get_block(bp_data, target_id)
        if block is not None:
            for flow in iter_flow(block):
                ln = int(flow.get("line") or 0)
                if limit and ln and ln >= limit:
                    break
                inst = str(flow.get("inst") or "").upper()
                fa = [str(a) for a in (flow.get("args") or [])]
                if inst in TRIGGER_INST:
                    pending = (f"{inst} {', '.join(fa)}" if fa else inst).strip()
                elif inst in BRANCH_INST and pending:
                    branch_str = (f"{inst} {', '.join(fa)}" if fa else inst).strip()
                    result.append(f"{pending} / {branch_str}")
                    pending = None
        return result

    guards = _scan(block_id, setter_line)

    # One level of parent inheritance (BAS caller or FALLTHROUGH predecessor).
    parent_map = _build_block_parent_map(bp_data)
    parent_entry = parent_map.get(block_id.upper())
    if parent_entry:
        parent_id, call_line = parent_entry
        # call_line = BAS line → collect parent guards before that line.
        # call_line = None (FALLTHROUGH) → collect all parent guards (limit=0).
        inherited = _scan(parent_id, call_line or 0)
        guards = inherited + guards

    return guards


def _find_bas_target_blocks(bp_data: Dict[str, Any]) -> Set[str]:
    """Return uppercased block IDs that are BAS call targets within bp_data.

    Includes not only blocks directly targeted by a ``BAS Rn,<LABEL>``
    instruction but also their transitive FALLTHROUGH children — sub-blocks
    that follow a BAS-entry block without an unconditional transfer out and
    are therefore still inside the same internal subroutine body.

    Example: ``FIN_03GLOS`` is a direct BAS target; ``FIN_03_TMSEND`` falls
    through from it (after a TMSMA call whose return label is recorded as
    the next sequential block).  Both should be tagged ``entered_via_bas``.
    """
    # Collect direct BAS targets.
    direct_targets: Set[str] = set()
    for block in (bp_data.get("blocks") or []):
        for flow in iter_flow(block):
            if str(flow.get("inst") or "").upper() == "BAS":
                args = [str(a) for a in (flow.get("args") or [])]
                if len(args) >= 2:
                    direct_targets.add(args[1].strip().upper())

    # Build forward fallthrough map from _build_block_parent_map:
    # parent_upper → [fallthrough_child_upper]  (call_line is None for fallthrough)
    parent_map = _build_block_parent_map(bp_data)
    fallthrough_fwd: Dict[str, List[str]] = {}
    for child_upper, (parent_id, call_line) in parent_map.items():
        if call_line is None:  # fallthrough, not an explicit BAS
            fallthrough_fwd.setdefault(parent_id.upper(), []).append(child_upper)

    # BFS: expand from direct BAS targets through fallthrough edges.
    all_bas_blocks: Set[str] = set(direct_targets)
    queue: List[str] = list(direct_targets)
    while queue:
        current = queue.pop()
        for child in fallthrough_fwd.get(current, []):
            if child not in all_bas_blocks:
                all_bas_blocks.add(child)
                queue.append(child)

    return all_bas_blocks


def _setter_action_and_bit(
    inst: str, args: List[str]
) -> Tuple[str, Optional[str]]:
    """Classify a setter as set/clear/move/write and extract the named bitmask.

    Returns ``(action, bit)`` where ``bit`` is:

    * For ``OI``/``OIY``: ``args[1]`` directly (e.g. ``"WB#03GLOS"``).
    * For ``NI``/``NIY``: the **named bit** extracted from the complement mask.
      TPF assembler encodes NI clears as ``NI VAR,X'FF'-BIT_SYMBOL``.  The raw
      ``args[1]`` (``"X'FF'-WB#03GLOS"``) is unwrapped to return ``"WB#03GLOS"``.
      Falls back to the full mask string if no symbol can be extracted.
    * For all other setters: ``None``.
    """
    if inst in {"OI", "OIY"}:
        return "set", (args[1].strip() if len(args) > 1 else None)
    if inst in {"NI", "NIY"}:
        raw = args[1].strip() if len(args) > 1 else None
        if raw:
            # Unwrap complement mask: X'FF'-WB#03GLOS → WB#03GLOS
            m = re.search(r"-\s*([A-Za-z#$@_][A-Za-z0-9#$@_]*)\s*$", raw)
            bit: Optional[str] = m.group(1) if m else raw
        else:
            bit = None
        return "clear", bit
    if inst in {"MVI", "MVIY", "MVC", "MVCIN", "MVHI", "MVHHI", "MVGHI"}:
        return "move", None
    return "write", None


def _extract_modifier_setter_sites(
    variable: str,
    asm_stem: str,
    blueprint_dir: Path,
    asm_dir: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    bp = resolve_asm_blueprint(asm_stem, blueprint_dir)
    if bp is None:
        return [], [f"Blueprint not found for modifier {asm_stem}"]

    asm_file = resolve_source_file(asm_stem, "asm", blueprint_dir, asm_dir)
    # GAP-036: do not abort when asm_file is None — setter-site detection is
    # purely blueprint-based.  asm_file is only used for raw-line extraction
    # (fetch_raw_line), which returns None gracefully when the file is absent.
    if asm_file is None:
        warnings.append(f"ASM source not found for modifier {asm_stem}")

    payload = load_json(bp, keys={"blocks", "symbols"})
    target = str(variable or "").upper()
    # Issue-5: pre-compute constant symbols for this modifier file.
    _mod_const_syms = collect_constant_symbols(payload, asm_file=asm_file if asm_file and asm_file.exists() else None, bp_path=bp)

    # P37: collect FLIPC-derived CE1CRx aliases across the full modifier file.
    # Use all block IDs (not a scope subset) because the modifier setter scan
    # is not restricted to a call-site scope.
    all_block_ids: Set[str] = {str(b.get("id") or "") for b in (payload.get("blocks") or [])}
    flipc_aliases = _collect_flipc_level_aliases(payload, all_block_ids)
    match_names: Set[str] = {target} | flipc_aliases.get(target, set())

    # Pre-build BAS target set so each setter site can be tagged entered_via_bas.
    bas_target_blocks: Set[str] = _find_bas_target_blocks(payload)

    setter_sites: List[Dict[str, Any]] = []

    for block in payload.get("blocks", []) or []:
        block_id = str(block.get("id") or "")
        for flow in iter_flow(block):
            inst = str(flow.get("inst") or "").upper()
            args = [str(a) for a in (flow.get("args") or [])]
            # P20: DBKEY KEYLIST=VAR fills the named key list work area.
            if inst in DB_KEYLIST_INST:
                keylist_var = _extract_keyword_arg(args, "KEYLIST")
                if keylist_var and keylist_var.upper() in match_names:
                    line_no = int(flow.get("line") or 0)
                    raw_line = fetch_raw_line(str(asm_file), line_no, asm_dir)
                    detail = ", ".join(args)
                    setter_sites.append({
                        "routine": block_id,
                        "line": line_no,
                        "instruction": inst,
                        "call_type": "setter_site",
                        "raw_line": raw_line or f"{inst} {detail}".strip(),
                        "setter_expression": keylist_var,
                    })
                continue
            # EX/EXRL expansion: resolve the target instruction at the label
            # and substitute inst/args so setter detection works on the real op.
            if inst in EXECUTE_INST:
                _ex_label = args[1] if len(args) > 1 else ""
                _ex_target_entry = resolve_ex_target(payload, normalize_token(_ex_label))
                if _ex_target_entry:
                    inst = str(_ex_target_entry.get("inst") or "").upper()
                    args = [str(a) for a in (_ex_target_entry.get("args") or [])]
                else:
                    continue  # cannot resolve EX target — skip
            if inst not in SETTER_INST:
                continue

            dest_idx = DEST_ARG_INDEX_BY_INST.get(inst, 0)
            lhs_from_var = normalize_token(str(flow.get("var") or ""))
            if lhs_from_var:
                lhs = lhs_from_var
                raw_dest = lhs_from_var
            else:
                raw_dest = args[dest_idx] if len(args) > dest_idx else ""
                lhs = normalize_token(raw_dest)
            # GAP-RANGE-WRITE: also accept range-write instructions like UNPK EBW000(4)
            # whose normalized LHS is EBW000 but whose written range covers the target
            # (e.g. EBW002 at offset 2 is within [EBW000, EBW000+4)).
            if lhs not in match_names and not _range_write_covers_target(raw_dest, target):
                continue
            # Audit §3a/§3b/§3c/§3d: value-preserving / spec-exception / zero-mask /
            # nonzero-displacement writes are NOT modifier sites for the target.
            _verdict = classify_setter(inst, args)
            if _verdict.get("skip_setter_site"):
                _w = _verdict.get("warn")
                if _w:
                    warnings.append(f"{block_id}: {_w}")
                continue

            # Derive setter_expression — the source operand being written to the target.
            # Blueprint stores the parsed source in flow["val"] when it can identify it
            # (MVC, MVI, PACK, etc.).  For register-store instructions (ST, STH, STM…)
            # the blueprint uses only flow["args"] and "val" is absent; derive from args.
            _val = flow.get("val")
            if _val is not None:
                _setter_expr: Optional[str] = _strip_hlasm_remark_from_val(str(_val))
            elif dest_idx == 1:  # ST/STH/STC/CVD/STG… — src register is args[0]
                _setter_expr = args[0] if args else None
            elif dest_idx == 2:  # STM/STMG/STCM — register range Rlo-Rhi
                _setter_expr = f"{args[0]}-{args[1]}" if len(args) >= 2 else (args[0] if args else None)
            else:
                # args[1] can contain an embedded remark (e.g. XC FIELD,FIELD CLEAR...)
                _setter_expr = _strip_hlasm_remark_from_val(args[1]) if len(args) > 1 else None

            line_no = int(flow.get("line") or 0)
            raw_line = fetch_raw_line(str(asm_file), line_no, asm_dir)
            detail = ", ".join(args) if args else ""
            site: Dict[str, Any] = {
                "routine": block_id,
                "line": line_no,
                "instruction": inst,
                "call_type": "setter_site",
                "raw_line": raw_line or f"{inst} {detail}".strip(),
                "setter_expression": _setter_expr,
            }
            # Issue-5 / GAP-028: flag sites whose source is a compile-time constant.
            # Priority:
            #   1. Inline literals (=C'..', =X'..', =F'..') — always constant.
            #   2. XC VAR,VAR — self-clear zeroes the destination (Bug-5B fix):
            #      setter_expression equals the target → no runtime source dep_var.
            #   3. SI/SIY/SIL immediate instructions (OI, NI, XI, MVI, ASI, MVHI…):
            #      the second operand is ALWAYS a compile-time immediate, even when
            #      defined as an EQU in a COPY member absent from the symbol table.
            #   4. Symbol-table lookup — covers DC/EQU labels in local/global syms.
            if _setter_expr and str(_setter_expr).startswith("="):
                site["constant_source"] = True
            elif (inst == "XC" and _setter_expr
                  and normalize_token(str(_setter_expr)).upper() == normalize_token(target).upper()):
                site["constant_source"] = True
            elif inst in IMMEDIATE_OPERAND_SETTER_INST and dest_idx == 0:
                # setter_expr = args[1] for dest_idx==0; that's always an immediate
                site["constant_source"] = True
            elif _setter_expr:
                _norm_expr_mod = normalize_token(str(_setter_expr))
                if _norm_expr_mod and _norm_expr_mod.upper() in _mod_const_syms:
                    site["constant_source"] = True
            # EQU constant heuristic: COPY member EQU symbols (e.g. #ACPDBOK).
            if not site.get("constant_source") and inst in IMMEDIATE_OPERAND_SETTER_INST and _setter_expr:
                if looks_like_equ_constant(normalize_token(str(_setter_expr))):
                    site["constant_source"] = True
            # P37: annotate setters discovered via FLIPC alias so consumers can
            # distinguish direct writes from level-swap-mediated writes.
            if lhs != target and lhs in match_names:
                site["flipc_alias_of"] = target
            if _verdict.get("prior_value_required"):
                site["prior_value_required"] = True
            if inst in _HARDWARE_SOURCE_BY_INST:
                site["hardware_source"] = _HARDWARE_SOURCE_BY_INST[inst]
            # Phase-3: action/bit classification, per-setter guard conditions, BAS flag.
            _action, _bit = _setter_action_and_bit(inst, args)
            site["action"] = _action
            if _bit is not None:
                site["bit"] = _bit
            _guards = _collect_site_guards(payload, block_id, line_no)
            if _guards:
                site["guards"] = _guards
            if block_id.upper() in bas_target_blocks:
                site["entered_via_bas"] = True
            setter_sites.append(site)

    # P14: implicit CE1CRn writes from GETCC/GETFC/MALOC across the full modifier file.
    level_alloc_sites_mod = _collect_level_alloc_setter_sites(payload, all_block_ids, asm_file, asm_dir)
    if target in level_alloc_sites_mod:
        setter_sites.extend(level_alloc_sites_mod[target])

    # P15-P16: implicit CE1DTn writes from DETAC/ATTAC across the full modifier file.
    detac_sites_mod = _collect_detac_attac_flag_writes(payload, all_block_ids, asm_file, asm_dir)
    if target in detac_sites_mod:
        setter_sites.extend(detac_sites_mod[target])

    # P19: implicit SW00WKA write from DBSPA across the full modifier file.
    db_spa_sites_mod = _collect_db_managed_field_writes(payload, all_block_ids, asm_file, asm_dir)
    if target in db_spa_sites_mod:
        setter_sites.extend(db_spa_sites_mod[target])

    # P42: implicit EBL8URTN write from TMSMA across the full modifier file.
    macro_sites_mod = _collect_macro_fixed_result_writes(payload, all_block_ids, asm_file, asm_dir)
    if target in macro_sites_mod:
        setter_sites.extend(macro_sites_mod[target])

    return _dedupe_sites(setter_sites), warnings


def _build_modifier_asm_payload(
    variable: str,
    modifier_stem: str,
    blueprint_dir: Path,
    asm_dir: Optional[Path],
    max_depth: int,
    max_subroutine_depth: int,
    max_subroutine_nodes: int,
    max_trace_nodes: int,
    target_setters: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    bp = resolve_asm_blueprint(modifier_stem, blueprint_dir)
    if bp is None:
        return {
            "role": "modifier",
            "file_type": "asm",
            "setter_sites": [],
            "setter_site_traces": [],
            "call_graph": {"nodes": [], "edges": []},
            "subroutine_summary": {"count": 0, "unresolved": 0, "capped": 0},
            "warnings": [f"Blueprint not found for modifier {modifier_stem}"],
        }

    asm_file = resolve_source_file(modifier_stem, "asm", blueprint_dir, asm_dir)

    warnings: List[str] = []
    # PERF-003 / GAP-036: do NOT abort when asm_file is None.
    # The ASM source is used only for raw-line display and constant-symbol scanning.
    # All setter-site detection and backward tracing operate on the blueprint JSON
    # and work correctly with asm_file=None (build_asm_block_source_index and
    # collect_constant_symbols already handle None gracefully).  Aborting early
    # caused 0 setter sites on any large-codebase job where source files are not
    # co-located with blueprints — silently producing empty traversal results.
    if asm_file is None:
        warnings.append(f"ASM source not found for modifier {modifier_stem}")

    diag = diagnose_blueprint_consistency(load_json(bp, keys={"blocks", "call_graph"}))
    if diag.get("warnings"):
        warnings.append(
            f"Blueprint consistency for {modifier_stem}: " + ", ".join(diag.get("warnings", []))
        )

    setter_sites, setter_warnings = _extract_modifier_setter_sites(
        variable=variable,
        asm_stem=modifier_stem,
        blueprint_dir=blueprint_dir,
        asm_dir=asm_dir,
    )
    warnings.extend(setter_warnings)
    if target_setters:
        setter_sites = _filter_setter_sites(setter_sites, target_setters, file_stem=modifier_stem)
    if not setter_sites:
        return {
            "role": "modifier",
            "file_type": "asm",
            "setter_sites": [],
            "setter_site_traces": [],
            "call_graph": {"nodes": [], "edges": []},
            "subroutine_summary": {"count": 0, "unresolved": 0, "capped": 0},
            "warnings": warnings + [f"No setter sites found for {variable} in modifier {modifier_stem}"],
        }

    trace_results: List[Dict[str, Any]] = []
    for site in setter_sites:
        traced = trace_asm_call_site_backward(
            variable=variable,
            blueprint_path=bp,
            asm_file=asm_file,
            call_site=site,
            max_depth=max_depth,
            max_subroutine_depth=max_subroutine_depth,
            max_subroutine_nodes=max_subroutine_nodes,
            max_trace_nodes_per_site=max_trace_nodes,
        )
        trace_results.append(traced)
        warnings.extend(traced.get("warnings", []))

    return {
        "role": "modifier",
        "file_type": "asm",
        "setter_sites": setter_sites,
        "setter_site_traces": trace_results,
        "call_graph": _merged_call_graph_with_subroutines(trace_results),
        "subroutine_summary": _subroutine_summary_from_trace_results(trace_results),
        "warnings": sorted(set(str(w) for w in warnings if w)),
    }


def _edge_type_map(graph_payload: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    for e in graph_payload.get("edges", []) or []:
        src = str(e.get("source") or "").strip()
        tgt = str(e.get("target") or "").strip()
        if not src or not tgt:
            continue
        inst = str((e.get("instructions") or ["?"])[0])
        out[(src, tgt)] = inst
    return out


def _edge_lines_map(graph_payload: Dict[str, Any]) -> Dict[Tuple[str, str], Set[int]]:
    """Extract line numbers from file_call_graph call_sites per (source, target) pair."""
    out: Dict[Tuple[str, str], Set[int]] = {}
    for e in graph_payload.get("edges", []) or []:
        src = str(e.get("source") or "").strip()
        tgt = str(e.get("target") or "").strip()
        if not src or not tgt:
            continue
        lines: Set[int] = set()
        for cs in e.get("call_sites") or []:
            ln = cs.get("line")
            if isinstance(ln, int) and ln > 0:
                lines.add(ln)
        if lines:
            out[(src, tgt)] = lines
    return out


def _edge_callsite_map(
    graph_payload: Dict[str, Any],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Maps (source, target) → full list of call_site dicts (preserves source_block/target_module)."""
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for e in graph_payload.get("edges", []) or []:
        src = str(e.get("source") or "").strip()
        tgt = str(e.get("target") or "").strip()
        if src and tgt:
            out[(src, tgt)] = list(e.get("call_sites") or [])
    return out


def _filter_callback_call_sites(
    forward_sites: List[Dict[str, Any]],
    reverse_sites: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Exclude call sites that are part of a mutual callback / dispatch pattern.

    A *true* callback requires BOTH conditions to hold simultaneously:

      1. The calling function in A (forward ``source_block``) is the same
         function that B dispatches back to in A (reverse ``target_module``).
         Example: A::F calls B, and B calls A::F back.

      2. The function in B that A is calling (forward ``target_module``) is
         the same function in B that performs the dispatch back to A (reverse
         ``source_block``).
         Example: A calls B::dispatch, and it is B::dispatch that calls A::F.

    Only suppressing when BOTH conditions hold avoids false positives for
    ordinary circular calls where the same function names appear on both
    sides but through different code paths (e.g. A::F→B::G and B::H→A::F
    where G≠H).
    """
    if not reverse_sites:
        return forward_sites

    # Condition 1 operand: functions in A that B dispatches back to
    dispatched_back_in_caller: Set[str] = {
        str(cs.get("target_module") or "").upper()
        for cs in reverse_sites
        if cs.get("target_module")
    }
    # Condition 2 operand: functions in B that perform the dispatch back to A
    callee_dispatcher_fns: Set[str] = {
        str(cs.get("source_block") or "").upper()
        for cs in reverse_sites
        if cs.get("source_block")
    }

    if not dispatched_back_in_caller:
        return forward_sites

    kept = []
    for cs in forward_sites:
        src_block = str(cs.get("source_block") or "").upper()  # fn in A calling B
        tgt_mod   = str(cs.get("target_module") or "").upper()  # fn in B being called

        # Both conditions must hold to classify this as a callback.
        cond1 = bool(src_block and src_block in dispatched_back_in_caller)
        cond2 = bool(tgt_mod   and tgt_mod   in callee_dispatcher_fns)
        if cond1 and cond2:
            continue  # mutual callback — skip
        kept.append(cs)
    return kept


def _scoped_cpp_trace(
    seed_ids: List[str],
    call_sites: List[Dict[str, Any]],
    gvl: Any,
    per_file_lineage: Optional[Dict[str, Any]] = None,
    variable_name: str = "",
) -> Dict[str, Any]:
    """Run C++ GVL backward trace with per-calling-function before_line scoping.

    Groups seeds by their enclosing function (from GVL node ID), maps each
    function to its earliest call-site line, and runs a separate bounded trace
    per function.  Seeds that don't match any known caller are traced without
    restriction.  Results are merged into a single trace dict.
    """
    if not seed_ids or not gvl:
        return {}

    # Earliest call-site line per calling function (use min for same-routine duplicates)
    fn_call_line: Dict[str, int] = {}
    # Issue-9: collect conditional_context from call_sites so prerequisite
    # conditions (e.g. switch-case paths) are propagated to guard_conditions.
    fn_call_conditions: Dict[str, List[Dict[str, Any]]] = {}
    for s in call_sites:
        fn = str(s.get("function") or "")
        ln = int(s.get("line") or 0)
        if fn and ln > 0:
            fn_call_line[fn] = min(fn_call_line.get(fn, ln), ln)
        _conds = s.get("conditional_context") or []
        if fn and _conds:
            fn_call_conditions.setdefault(fn, [])
            for _cc in (_conds if isinstance(_conds, list) else [_conds]):
                fn_call_conditions[fn].append({
                    "file": str(s.get("file") or ""),
                    "line": ln,
                    "scope": fn,
                    "expression": str(_cc),
                    "conditional_context": [str(_cc)],
                    "source": "call_site_prerequisite",
                })

    # Group seeds by their enclosing function
    fn_seeds: Dict[str, List[str]] = {}
    unscoped: List[str] = []
    for sid in seed_ids:
        fn = _enc_fn_from_gvl_id(sid)
        if fn and fn in fn_call_line:
            fn_seeds.setdefault(fn, []).append(sid)
        else:
            unscoped.append(sid)

    if not fn_seeds:
        # No seeds matched a calling function — trace without restriction
        if not unscoped:
            return {}
        _result = trace_cpp_variable_backward_multi(
            seed_ids=unscoped,
            gvl=gvl,
            max_depth=DEFAULT_CPP_MAX_DEPTH,
            max_nodes=DEFAULT_CPP_MAX_NODES,
            per_file_lineage=per_file_lineage,
            variable_name=variable_name,
        )
        # Issue-9: still append call-site prerequisite conditions
        if fn_call_conditions:
            _existing_gc = _result.get("guard_conditions") or []
            _gc_keys: Set[str] = {
                f"{g.get('line')}:{g.get('scope')}:{g.get('expression')}"
                for g in _existing_gc
            }
            for _gc_entries in fn_call_conditions.values():
                for gc in _gc_entries:
                    gk = f"{gc.get('line')}:{gc.get('scope')}:{gc.get('expression')}"
                    if gk not in _gc_keys:
                        _gc_keys.add(gk)
                        _existing_gc.append(gc)
            _result["guard_conditions"] = _existing_gc
        return _result

    # Per-function scoped traces, then merge
    merged_nodes: Dict[str, Any] = {}
    merged_edges: List[Dict[str, Any]] = []
    seen_edge_keys: Set[Tuple[str, str, str]] = set()
    all_dep_vars: List[str] = []
    seen_dep_vars: Set[str] = set()
    all_bridge_regs: List[str] = []
    seen_bridge_regs: Set[str] = set()
    all_warnings: List[str] = []
    all_guard_conditions: List[Dict[str, Any]] = []
    seen_gc_keys: Set[str] = set()
    all_gc_dep_vars: List[str] = []
    seen_gc_dv: Set[str] = set()
    all_per_seed_traces: Dict[str, Dict[str, Any]] = {}
    any_truncated = False

    for fn, fn_seed_list in fn_seeds.items():
        r = trace_cpp_variable_backward_multi(
            seed_ids=fn_seed_list,
            gvl=gvl,
            max_depth=DEFAULT_CPP_MAX_DEPTH,
            max_nodes=DEFAULT_CPP_MAX_NODES,
            before_line=fn_call_line[fn],
            per_file_lineage=per_file_lineage,
            variable_name=variable_name,
        )
        for n in (r.get("nodes") or []):
            nid = str(n.get("id") or "")
            if nid and nid not in merged_nodes:
                merged_nodes[nid] = n
        for e in (r.get("edges") or []):
            ek: Tuple[str, str, str] = (
                str(e.get("source") or ""),
                str(e.get("target") or ""),
                str(e.get("relation") or ""),
            )
            if ek not in seen_edge_keys:
                seen_edge_keys.add(ek)
                merged_edges.append(e)
        for dv in (r.get("dep_vars") or []):
            if dv not in seen_dep_vars:
                seen_dep_vars.add(dv)
                all_dep_vars.append(dv)
        for br in (r.get("bridge_registers") or []):
            if br not in seen_bridge_regs:
                seen_bridge_regs.add(br)
                all_bridge_regs.append(br)
        for gc in (r.get("guard_conditions") or []):
            gc_key = f"{gc.get('line')}:{gc.get('scope')}:{gc.get('expression')}"
            if gc_key not in seen_gc_keys:
                seen_gc_keys.add(gc_key)
                all_guard_conditions.append(gc)
        for gcdv in (r.get("_gc_dep_vars") or []):
            if gcdv not in seen_gc_dv:
                seen_gc_dv.add(gcdv)
                all_gc_dep_vars.append(gcdv)
        all_warnings.extend(r.get("warnings") or [])
        if r.get("truncated"):
            any_truncated = True
        # Collect per-seed traces (Issue #3 fix).
        for _ps_key, _ps_val in (r.get("per_seed_traces") or {}).items():
            all_per_seed_traces[_ps_key] = _ps_val

    # Also trace seeds not in any known calling function, without restriction
    if unscoped:
        r2 = trace_cpp_variable_backward_multi(
            seed_ids=unscoped,
            gvl=gvl,
            max_depth=DEFAULT_CPP_MAX_DEPTH,
            max_nodes=DEFAULT_CPP_MAX_NODES,
            per_file_lineage=per_file_lineage,
            variable_name=variable_name,
        )
        for n in (r2.get("nodes") or []):
            nid = str(n.get("id") or "")
            if nid and nid not in merged_nodes:
                merged_nodes[nid] = n
        for e in (r2.get("edges") or []):
            ek = (str(e.get("source") or ""), str(e.get("target") or ""), str(e.get("relation") or ""))
            if ek not in seen_edge_keys:
                seen_edge_keys.add(ek)
                merged_edges.append(e)
        for dv in (r2.get("dep_vars") or []):
            if dv not in seen_dep_vars:
                seen_dep_vars.add(dv)
                all_dep_vars.append(dv)
        for gc in (r2.get("guard_conditions") or []):
            gc_key = f"{gc.get('line')}:{gc.get('scope')}:{gc.get('expression')}"
            if gc_key not in seen_gc_keys:
                seen_gc_keys.add(gc_key)
                all_guard_conditions.append(gc)
        for gcdv in (r2.get("_gc_dep_vars") or []):
            if gcdv not in seen_gc_dv:
                seen_gc_dv.add(gcdv)
                all_gc_dep_vars.append(gcdv)
        all_warnings.extend(r2.get("warnings") or [])
        for _ps_key, _ps_val in (r2.get("per_seed_traces") or {}).items():
            all_per_seed_traces[_ps_key] = _ps_val

    # Issue-9: Append call-site prerequisite conditions (e.g. switch-case paths
    # that must hold for a calling function to be entered) to guard_conditions.
    for fn, gc_entries in fn_call_conditions.items():
        for gc in gc_entries:
            gc_key = f"{gc.get('line')}:{gc.get('scope')}:{gc.get('expression')}"
            if gc_key not in seen_gc_keys:
                seen_gc_keys.add(gc_key)
                all_guard_conditions.append(gc)

    return {
        "seeds": seed_ids,
        "nodes": list(merged_nodes.values()),
        "edges": merged_edges,
        "dep_vars": all_dep_vars,
        "_gc_dep_vars": all_gc_dep_vars,
        "bridge_registers": all_bridge_regs,
        "guard_conditions": all_guard_conditions,
        "per_seed_traces": all_per_seed_traces,
        "truncated": any_truncated,
        "warnings": all_warnings,
    }


def _build_cpp_upstream_payload(
    caller: str,
    callee: str,
    callee_type: str,
    edge_instruction: str,
    blueprint_dir: Path,
    asm_dir: Optional[Path],
    variable: str = "",
) -> Dict[str, Any]:
    source_index = build_func_source_index(blueprint_dir, asm_dir, caller)

    # Load the caller's cpp blueprint for bridge extraction.
    bp: Dict[str, Any] = {}
    cpp_bp_path = blueprint_dir / f"{caller}.cpp.json"
    if not cpp_bp_path.exists():
        cpp_bp_path = blueprint_dir / f"{caller.lower()}.cpp.json"
    if cpp_bp_path.exists():
        try:
            bp = load_json(cpp_bp_path, keys={"variable_lineage", "field_asm_aliases"})
        except Exception:
            bp = {}

    from backward_traversal.utils.lazy_gvl import make_lazy_gvl
    # Always pass cpp_bp_path (even when the file is absent) so make_lazy_gvl can
    # walk up to index.db via find_index_db() and return a DbLazyGVL fallback when
    # global_variable_lineage.json has been deleted after DB ingestion.
    gvl = make_lazy_gvl(bp, blueprint_path=cpp_bp_path)

    if callee_type == "asm":
        # FIX 2: use entry-point-aware bridge instead of bare stem match
        call_sites = find_cpp_to_asm_callers(caller, callee, blueprint_dir, asm_dir)
        for site in call_sites:
            fn = site.get("function")
            if fn and fn in source_index:
                site["function_code"] = source_index[fn]
        # Enrich call sites with conditional_context from GVL arg_bind edges scoped to caller.
        if gvl:
            _cs_cc: Dict[int, List] = {}
            _caller_lower = caller.lower()
            # Use query_edges(relation="arg_bind") when available (DB-backed GVL) so
            # only arg_bind rows are fetched via the ix_gvl_edges_job_rel index rather
            # than scanning the full edge set.
            _arg_bind_edges = (
                gvl.query_edges(relation="arg_bind")
                if hasattr(gvl, "query_edges")
                else (gvl.get("edges") or [])
            )
            for _e in _arg_bind_edges:
                if _e.get("relation") != "arg_bind" or not _e.get("conditional_context"):
                    continue
                _ef = str(_e.get("file") or "").lower()
                if _caller_lower not in _ef:
                    continue
                _ln = _e.get("line")
                if _ln is not None and _ln not in _cs_cc:
                    _cs_cc[_ln] = _e["conditional_context"]
            for site in call_sites:
                if site.get("conditional_context") is None:
                    site["conditional_context"] = _cs_cc.get(site.get("line"))
        nodes = [
            {
                "id": str(site.get("function") or ""),
                "function_code": source_index.get(str(site.get("function") or ""), ""),
                "line": site.get("line"),
                "instruction": site.get("instruction"),
            }
            for site in call_sites
            if site.get("function")
        ]
        edges = [
            {
                "source": str(site.get("function") or ""),
                "target": callee,
                "type": "CALL_TO_ASM",
                "direction": ["backward"],
            }
            for site in call_sites
            if site.get("function")
        ]

        # ── C++ lineage bridge ──────────────────────────────────────────────
        asm_field_aliases = extract_asm_field_aliases(bp, variable) if variable else {}
        tpf_regs_slots = extract_tpf_regs_slots(gvl, callee)
        seed_ids = find_cpp_seed_nodes(gvl, variable, asm_field_aliases, tpf_regs_slots) if variable else []

        # Per-calling-function scoped GVL trace: each function's seeds are traced
        # with that function's earliest call-site line as before_line so we only
        # follow edges representing assignments that precede the call to the callee.
        per_file_lineage = bp.get("variable_lineage") or {}
        cpp_trace: Dict[str, Any] = _scoped_cpp_trace(
            seed_ids, call_sites, gvl,
            per_file_lineage=per_file_lineage, variable_name=variable,
        ) if (seed_ids and gvl) else {}
        # File-scope to the caller: before_line alone can't separate files (it
        # compares line numbers across files), so drop foreign-file setter edges
        # that the BFS reached via shared canonical nodes.  The callee's own
        # setters are captured when the callee is processed as its own chain hop.
        if cpp_trace:
            cpp_trace = _scope_cpp_trace_to_file(cpp_trace, caller)

        payload: Dict[str, Any] = {
            "role": "upstream",
            "file_type": "cpp",
            "edge_type": edge_instruction,
            "call_sites": call_sites,
            "asm_field_aliases": asm_field_aliases,
            "tpf_regs_slots": tpf_regs_slots,
            "call_graph": {"nodes": nodes, "edges": edges},
        }
        if cpp_trace:
            payload["cpp_lineage_trace"] = cpp_trace
        return payload

    call_pairs = find_cpp_to_cpp_callers(caller, callee, blueprint_dir, asm_dir)
    for pair in call_pairs:
        fn = pair.get("caller_function")
        if fn and fn in source_index:
            pair["function_code"] = source_index[fn]

    node_ids: Set[str] = set()
    nodes = []
    edges = []
    for pair in call_pairs:
        src = str(pair.get("caller_function") or "")
        tgt = str(pair.get("callee_function") or "")
        if src and src not in node_ids:
            node_ids.add(src)
            nodes.append({"id": src, "function_code": source_index.get(src, "")})
        if tgt and tgt not in node_ids:
            node_ids.add(tgt)
            nodes.append({"id": tgt})
        if src and tgt:
            edges.append({"source": src, "target": tgt, "type": "CALL", "direction": ["backward"]})

    # Normalize call_pairs → call_sites format so downstream code that reads
    # upstream["call_sites"] (anchor_blocks extraction at Pass 2 setup, line ~1534)
    # finds a populated list.  call_sites uses key "function" (matching the
    # CPP→ASM branch from find_cpp_to_asm_callers), while call_pairs uses
    # "caller_function"/"callee_function".  Without this normalization,
    # anchor_blocks is always empty for CPP→CPP hops, causing the dep_var BFS
    # to fall back to unscoped (all-edges) traversal instead of restricting to
    # the caller function's scope.
    # Build a line→conditional_context index from GVL arg_bind edges scoped to the caller file.
    _gvl_cc_by_line: Dict[int, List[str]] = {}
    if gvl:
        _caller_stem = caller.lower()
        _arg_bind_edges2 = (
            gvl.query_edges(relation="arg_bind")
            if hasattr(gvl, "query_edges")
            else (gvl.get("edges") or [])
        )
        for _e in _arg_bind_edges2:
            if _e.get("relation") != "arg_bind" or not _e.get("conditional_context"):
                continue
            _ef = str(_e.get("file") or "").lower()
            if _caller_stem not in _ef:
                continue
            _ln = _e.get("line")
            if _ln is not None and _ln not in _gvl_cc_by_line:
                _gvl_cc_by_line[_ln] = _e["conditional_context"]

    call_sites = [
        {
            "function": str(pair.get("caller_function") or ""),
            "callee_function": str(pair.get("callee_function") or ""),
            "line": pair.get("line"),
            "instruction": pair.get("instruction", ""),
            "call_type": pair.get("call_type", ""),
            "raw_line": pair.get("raw_line", ""),
            "function_code": pair.get("function_code", ""),
            "conditional_context": _gvl_cc_by_line.get(pair.get("line")),
        }
        for pair in call_pairs
        if pair.get("caller_function")
    ]

    # FIX-new-6 (T4): run the GVL lineage bridge for C++→C++ edges, mirroring
    # what the callee_type=="asm" branch already does.  Without this the payload
    # carries no cpp_lineage_trace and Pass 2 gets zero dep_vars for the caller.
    asm_field_aliases = extract_asm_field_aliases(bp, variable) if variable else {}
    tpf_regs_slots = extract_tpf_regs_slots(gvl, callee)
    seed_ids = find_cpp_seed_nodes(gvl, variable, asm_field_aliases, tpf_regs_slots) if variable else []

    per_file_lineage = bp.get("variable_lineage") or {}
    cpp_trace: Dict[str, Any] = _scoped_cpp_trace(
        seed_ids, call_sites, gvl,
        per_file_lineage=per_file_lineage, variable_name=variable,
    ) if (seed_ids and gvl) else {}
    # File-scope to the caller (see C++→ASM branch above for rationale).
    if cpp_trace:
        cpp_trace = _scope_cpp_trace_to_file(cpp_trace, caller)

    payload: Dict[str, Any] = {
        "role": "upstream",
        "file_type": "cpp",
        "edge_type": edge_instruction,
        "call_sites": call_sites,
        "asm_field_aliases": asm_field_aliases,
        "tpf_regs_slots": tpf_regs_slots,
        "call_graph": {"nodes": nodes, "edges": edges},
    }
    if cpp_trace:
        payload["cpp_lineage_trace"] = cpp_trace
    return payload


def _build_asm_upstream_payload(
    variable: str,
    caller: str,
    callee: str,
    callee_type: str,
    edge_instruction: str,
    blueprint_dir: Path,
    asm_dir: Optional[Path],
    max_depth: int,
    max_subroutine_depth: int,
    max_subroutine_nodes: int,
    max_trace_nodes: int,
) -> Dict[str, Any]:
    bp = resolve_asm_blueprint(caller, blueprint_dir)
    if bp is None:
        return {
            "role": "upstream",
            "file_type": "asm",
            "edge_type": edge_instruction,
            "call_sites": [],
            "call_site_traces": [],
            "call_graph": {"nodes": [], "edges": []},
            "subroutine_summary": {"count": 0, "unresolved": 0, "capped": 0},
            "warnings": [f"Blueprint not found for {caller}"],
        }

    asm_file = resolve_source_file(caller, "asm", blueprint_dir, asm_dir)
    warnings: List[str] = []
    # GAP-036: do not abort when asm_file is None — call-site detection and
    # backward tracing are blueprint-based. asm_file is only used for raw-line
    # display in trace_asm_call_site_backward; it already handles asm_file=None.
    if asm_file is None:
        warnings.append(f"ASM source not found for {caller}")
    diag = diagnose_blueprint_consistency(load_json(bp, keys={"blocks", "call_graph"}))
    if diag.get("warnings"):
        warnings.append(
            f"Blueprint consistency for {caller}: " + ", ".join(diag.get("warnings", []))
        )

    if callee_type == "cpp":
        call_sites = find_asm_to_cpp_callers(caller, callee, blueprint_dir, asm_dir)
    else:
        call_sites = find_asm_callers(caller, callee, blueprint_dir, asm_dir)

    if not call_sites:
        return {
            "role": "upstream",
            "file_type": "asm",
            "edge_type": edge_instruction,
            "call_sites": [],
            "call_site_traces": [],
            "call_graph": {"nodes": [], "edges": []},
            "subroutine_summary": {"count": 0, "unresolved": 0, "capped": 0},
            "warnings": warnings + [f"No call sites found from {caller} to {callee}"],
        }

    trace_results: List[Dict[str, Any]] = []
    for site in call_sites:
        traced = trace_asm_call_site_backward(
            variable=variable,
            blueprint_path=bp,
            asm_file=asm_file,
            call_site=site,
            max_depth=max_depth,
            max_subroutine_depth=max_subroutine_depth,
            max_subroutine_nodes=max_subroutine_nodes,
            max_trace_nodes_per_site=max_trace_nodes,
        )
        trace_results.append(traced)
        warnings.extend(traced.get("warnings", []))

    merged_graph = _merged_call_graph_with_subroutines(trace_results)

    # Enrich call_sites with conditional_context extracted from trace call_graph nodes.
    # Each trace's call_graph has a node whose id matches the call_site's routine;
    # that node's relevant_code_lines contains TEST_MASK + BRANCH_STEP pairs.
    _node_by_id: Dict[str, Dict] = {}
    for tr in trace_results:
        for node in tr.get("call_graph", {}).get("nodes", []):
            nid = node.get("id", "")
            if nid and nid not in _node_by_id:
                _node_by_id[nid] = node
    for site in call_sites:
        if site.get("conditional_context") is not None:
            continue
        routine = site.get("routine", "")
        node = _node_by_id.get(routine)
        if node:
            rcl = node.get("relevant_code_lines") or []
            conds = _extract_asm_call_conditions(rcl)
            if conds:
                site["conditional_context"] = conds

    payload: Dict[str, Any] = {
        "role": "upstream",
        "file_type": "asm",
        "edge_type": edge_instruction,
        "call_sites": call_sites,
        "call_site_traces": trace_results,
        "call_graph": merged_graph,
        "subroutine_summary": _subroutine_summary_from_trace_results(trace_results),
    }
    if warnings:
        payload["warnings"] = sorted(set(str(w) for w in warnings if w))
    return payload


def _path_to_chain_file(stem: str, selected_chain: List[str]) -> List[str]:
    """Resolver-order chain from modifier (last in selected_chain) to stem (inclusive).

    e.g. selected_chain=[lu82, dx750000, ..., da78], stem=dx730000
    → [da78, dx740200, dx730000]
    Degrades to [stem] when stem is not in selected_chain (safe fallback).
    """
    try:
        idx = selected_chain.index(stem)
    except ValueError:
        return [stem]
    return list(reversed(selected_chain[idx:]))


def _dep_var_origin_chain(
    dep_var: str,
    stem: str,
    selected_chain: List[str],
    dep_var_collection_map: Dict[str, Dict[str, Set[str]]],
) -> List[str]:
    """Return the resolver-order chain prefix that first exposed a dep var.

    This is informational metadata for chunked dep-var tuples. Prefer the
    current stem when it directly collected the dep var; otherwise fall back to
    the nearest selected-chain file recorded for that dep var.
    """
    dep_key = str(dep_var or "").upper()
    collection = dep_var_collection_map.get(dep_key) or {}
    collected_stems = {_norm_file_id(s) for s in collection}
    stem_norm = _norm_file_id(stem)

    if stem_norm in collected_stems:
        return _path_to_chain_file(stem, selected_chain)

    for chain_stem in reversed(selected_chain):
        if _norm_file_id(chain_stem) in collected_stems:
            return _path_to_chain_file(chain_stem, selected_chain)

    return _path_to_chain_file(stem, selected_chain)




_CPP_IDENT_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
# C++ keywords and noise to skip when extracting condition variables
_CPP_COND_SKIP: Set[str] = frozenset({  # type: ignore[assignment]
    "if", "else", "while", "for", "switch", "return", "case",
    "int", "char", "bool", "void", "const", "static", "auto",
    "unsigned", "long", "short", "new", "delete", "true", "false",
    "nullptr", "NULL", "sizeof", "memcmp", "strcmp", "strncmp",
})


def _extract_asm_call_conditions(relevant_code_lines: List[Dict[str, Any]]) -> List[str]:
    """Build condition strings from TM/branch pairs preceding a CALL_SITE entry.

    Each TM (Test under Mask) paired with a conditional branch encodes a guard:
      JNO/BNO → condition is "bits ON" (branch taken if NOT on → fall-through = ON)
      JO/BO   → condition is "bits OFF"
    """
    conditions: List[str] = []
    i = 0
    while i < len(relevant_code_lines):
        entry = relevant_code_lines[i]
        if entry.get("role") == "CALL_SITE":
            break
        if entry.get("role") == "TEST_MASK":
            detail = str(entry.get("detail") or "").strip()
            if i + 1 < len(relevant_code_lines):
                nxt = relevant_code_lines[i + 1]
                if nxt.get("role") == "BRANCH_STEP":
                    branch_inst = str(nxt.get("inst") or "").upper()
                    if branch_inst in ("JNO", "BNO", "JNZ", "BNZ"):
                        conditions.append(f"TM {detail} = ON")
                    elif branch_inst in ("JO", "BO", "JZ", "BZ"):
                        conditions.append(f"TM {detail} = OFF")
                    else:
                        conditions.append(f"TM {detail} ({branch_inst} not taken)")
                    i += 2
                    continue
        i += 1
    return conditions


def _extract_cpp_call_chain_conditions(
    setter_function: str,
    gvl: Any,
    max_depth: int = 6,
) -> List[Dict[str, Any]]:
    """Walk up the C++ call chain from setter_function via GVL arg_bind edges.

    For each hop, finds arg_bind edges whose target is call_arg:{callee}#N,
    extracts the caller function from the source node ID, and collects the
    conditional_context.  Returns a list of dicts ordered from callee→root:
      [{"callee": "processACreditTransaction", "caller": "performPaAuth...",
        "line": 669, "conditions": [...]}, ...]
    """
    chain: List[Dict[str, Any]] = []
    current_callee = setter_function
    visited: set = set()

    for _ in range(max_depth):
        if current_callee in visited:
            break
        visited.add(current_callee)

        call_prefix = f"call_arg:{current_callee}#"
        callers: Dict[str, Dict[str, Any]] = {}

        _ab_edges = (
            gvl.query_edges(relation="arg_bind")
            if hasattr(gvl, "query_edges")
            else (gvl.get("edges") or [])
        )
        for edge in _ab_edges:
            if str(edge.get("relation") or "") != "arg_bind":
                continue
            target = str(edge.get("target") or "")
            if not target.startswith(call_prefix):
                continue
            cc = edge.get("conditional_context")
            if not cc:
                continue
            # Source: "kind:callerFunc::varName" — extract callerFunc
            source = str(edge.get("source") or "")
            colon_idx = source.find(":")
            if colon_idx < 0:
                continue
            rest = source[colon_idx + 1:]
            if "::" not in rest:
                continue
            caller_func = rest.split("::")[0]
            if not caller_func or "/" in caller_func:
                # Skip file-path scoped nodes
                continue
            line = edge.get("line")
            if caller_func not in callers:
                callers[caller_func] = {"line": line, "conditions": list(cc)}
            else:
                for c in cc:
                    if c not in callers[caller_func]["conditions"]:
                        callers[caller_func]["conditions"].append(c)

        if not callers:
            break

        for caller_func, info in callers.items():
            chain.append({
                "callee": current_callee,
                "caller": caller_func,
                "line": info["line"],
                "conditions": info["conditions"],
            })

        current_callee = list(callers.keys())[0]

    return chain


def _extract_cpp_source_conditions(
    source_file: str,
    setter_line: int,
) -> List[Dict[str, Any]]:
    """Return enclosing if/while/for guards for the C++ setter at setter_line.

    Delegates to find_call_conditions.cpp_conditions_for_call which uses
    brace-depth tracking (correct handling of nested scopes) and matches
    if, while, and for headers.  Falls back to empty on any error.
    Each result dict has {line, check, vars_checked}.
    """
    if not source_file or not setter_line:
        return []
    try:
        from find_call_conditions import cpp_conditions_for_call as _cpp_conds
        raw = _cpp_conds(Path(source_file), setter_line)
    except Exception:
        return []

    result: List[Dict[str, Any]] = []
    for cond in raw:
        text = str(cond.get("text") or "").strip()
        line = cond.get("line") or 0
        if not text or not line:
            continue
        # The displayed `check` keeps string/char literals (e.g. lstknpoa == 'Y'),
        # but variable extraction must NOT pick up literal CONTENTS (e.g. a string
        # "ExpressPay" becoming a fake var), so strip quoted literals first.
        text_for_vars = re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', ' ', text)
        # Drop scope-resolved constants (EnumType::Value, Namespace::member) — the
        # type qualifier is not a checked variable.  Then collapse member-access
        # chains (a.b.c, a->b) to their terminal component so struct-path fragments
        # (process, various) don't pollute dep_vars.  Mirrors GVL dep_var normalization.
        text_no_scope = re.sub(r'\b[A-Za-z_]\w*::[A-Za-z_]\w*', ' ', text_for_vars)
        vars_checked: List[str] = []
        seen_vars: Set[str] = set()
        for m in re.finditer(r'[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)*', text_no_scope):
            chain = m.group(0)
            terminal = re.split(r'\.|->', chain)[-1].strip()
            if (not terminal or terminal in _CPP_COND_SKIP or terminal in seen_vars
                    or len(terminal) < 3 or terminal.isupper()):
                continue
            seen_vars.add(terminal)
            vars_checked.append(terminal)
        if vars_checked:
            result.append({
                "line":         line,
                "check":        text,
                "vars_checked": vars_checked,
            })
    return result


_SRC_EXT_RE = re.compile(r"\.(?:cpp|cc|cxx|hpp|hh|hxx|c|h)$", re.IGNORECASE)


def _extract_cpp_setter_sites(cpp_trace: Dict[str, Any], variable: str = "") -> List[Dict[str, Any]]:
    """Derive ASM-equivalent setter_sites from a scoped C++ lineage trace.

    One entry per unique physical statement (line + expression).  When multiple
    GVL edges reference the same source line (alias fan-out), the entry with the
    most informative scope (function-scoped > file-scoped) is kept.

    variable: when supplied, only edges whose expression contains the variable
    name (case-insensitive) are included.  This filters out sibling struct-field
    assignments whose GVL target node is the parent struct object rather than
    the specific field being traced.
    """
    _var_lower = variable.lower() if variable else ""
    # Collect candidates keyed by (line, expression); prefer function-scoped source.
    candidates: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    for e in (cpp_trace.get("edges") or []):
        rel = str(e.get("relation") or "").lower()
        if rel not in ("assign", "define"):
            continue
        # Issue-6: Skip function parameter declarations — a pass-by-reference
        # parameter is an input binding, not an actual setter of the variable.
        # GVL node IDs follow "kind:scope::name"; check the kind prefix.
        _src_id = str(e.get("source") or "")
        if _src_id.startswith("parameter:"):
            continue
        line = e.get("line")
        expr = str(e.get("expression") or "")
        # Drop edges for sibling struct fields: keep only edges whose expression
        # or target node ID contains the target variable name.  The expression
        # may use an alias / encoded name that omits the field name, so also
        # check the GVL target node ID which encodes the full struct path
        # (e.g. "struct_field:fn::parent.child.lwrpoDynamicCidMatchResponseInd").
        _tgt_id = str(e.get("target") or "").lower()
        if _var_lower and _var_lower not in expr.lower() and _var_lower not in _tgt_id:
            continue
        key = (line, expr)
        scope = str(e.get("scope") or "")
        existing = candidates.get(key)
        # Prefer entries that have a real function scope over "(global)" / ""
        if existing is None or (scope and not existing.get("routine")):
            candidates[key] = {
                "line": line,
                "routine": scope or "(global)",
                "file": str(e.get("file") or ""),
                "expression": expr,
                "conditional_context": e.get("conditional_context") or None,
            }
    # Return sorted by line so the list is deterministic
    return sorted(candidates.values(), key=lambda x: (x.get("line") or 0, x.get("expression") or ""))


def _enc_fn_from_gvl_id(node_id: str) -> str:
    """Extract the enclosing function name from a GVL node ID.

    GVL nodes use the format "kind:fn::path.to.field" or "kind:value".
    The function name is the segment between the first ":" and the "::" separator.

    Examples:
      "struct_field:globalLimitsUpdate::regs.r7"          → "globalLimitsUpdate"
      "variable:logUpdates::responseCode"                 → "logUpdates"
      "struct_field:fn::ptr->field"                       → "fn"
      "literal:'0'"                                       → "" (no function)
      'variable:/path/dw710000.cpp::"hidden"'             → "" (file-scope, not a fn)

    File-scope / translation-unit globals carry the SOURCE FILE PATH as their
    scope qualifier instead of a function name; those are rejected (return "")
    so they don't create bogus call_graph nodes whose id is a file path.
    """
    if "::" not in node_id:
        return ""
    after_kind = node_id.split(":", 1)[-1] if ":" in node_id else node_id
    fn = after_kind.split("::")[0]
    # Reject file-path scopes (file-level globals): contain a path separator or
    # end with a C/C++ source extension — not real function names.
    if "/" in fn or "\\" in fn or _SRC_EXT_RE.search(fn):
        return ""
    return fn


# GVL edge relations that map to ASM-equivalent roles for C++ call_graph nodes.
_CPP_RELATION_TO_ROLE: Dict[str, str] = {
    "assign":   "SETTER_STEP",
    "define":   "SETTER_STEP",
    "arg_bind": "CALL_SITE",
}

# Relations whose edge.file tells us where a seed is actually written.
_CPP_SETTER_RELATIONS: Set[str] = {"assign", "define", "arg_bind"}


def _normalize_cpp_dep_name(src_node: Optional[Dict[str, Any]], src_id: str) -> str:
    """Reduce a GVL source node to its terminal dep_var name, or '' to drop it.

    Keeps only variable / struct_field / array_element kinds; strips the
    function-scope prefix (::), pointer chain (->), and (for struct_field) the
    dotted struct path, leaving the terminal identifier.  Returns '' for noise
    tokens (return@func, level:*, etc.) and non-data kinds.
    """
    src_kind = str((src_node or {}).get("kind") or "").lower()
    if src_kind not in ("variable", "struct_field", "array_element"):
        return ""
    raw = str((src_node or {}).get("name") or "") or _enc_fn_from_gvl_id(src_id)
    if "::" in raw:
        raw = raw.split("::")[-1].strip()
    if "->" in raw:
        raw = raw.split("->")[-1].strip()
    if src_kind == "struct_field" and "." in raw:
        raw = raw.split(".")[-1].strip()
    if not raw or _is_noise_dep_var(raw.upper()):
        return ""
    # Drop PascalCase tokens: enum class values (e.g. Dynamic4cscMatchFound,
    # PaMessage::ExpressPay) and class constants are compile-time values, not
    # settable dep_vars.  Consistent with the PascalCase filter applied in
    # _extract_dep_vars_from_condition_strings.
    if raw[0].isupper():
        return ""
    return raw


def _file_stem_from_node_id(node_id: str) -> str:
    """Extract a file stem embedded in a GVL node ID like 'kind:/path/f.cpp::name'.

    Returns the lowercased stem (e.g. 'dw710900') when a file path is embedded,
    '' otherwise.
    """
    if "::" not in node_id:
        return ""
    after_kind = node_id.split(":", 1)[-1] if ":" in node_id else node_id
    segment = after_kind.split("::")[0]
    if "/" in segment or "\\" in segment or _SRC_EXT_RE.search(segment):
        return Path(segment).stem.lower()
    return ""


def _scope_cpp_trace_to_file(cpp_trace: Dict[str, Any], file_stem: str) -> Dict[str, Any]:
    """Return a copy of cpp_trace restricted to setter statements in *file_stem*.

    The GVL is global, so a raw trace mixes setters from every file that writes
    the variable.  This keeps only assign/define/arg_bind edges whose statement
    physically lives in *file_stem*, recomputes dep_vars from those edges' source
    nodes, and drops nodes no longer referenced.

    Improvements over original:
    - Fix 5: Fileless edges — when e["file"] is empty, infer the file from the
      source node ID (e.g. variable:/path/dw710900.cpp::fn → dw710900).  Foreign
      fileless setters are now correctly dropped rather than silently kept.
    - Fix 3: Alias fan-out dedup — after file-scoping, assign/define edges that
      share the same (line, expression) key are collapsed to one representative
      entry (the one with the most explicit function scope).  This eliminates
      duplicate rows produced when multiple GVL alias seeds trace back to the
      same physical statement.
    """
    if not cpp_trace or not file_stem:
        return cpp_trace
    stem_lower = str(file_stem).strip().lower()
    node_lookup = {str(n.get("id") or ""): n for n in (cpp_trace.get("nodes") or []) if n.get("id")}

    kept_edges: List[Dict[str, Any]] = []
    for e in (cpp_trace.get("edges") or []):
        rel = str(e.get("relation") or "").lower()
        if rel in _CPP_SETTER_RELATIONS:
            ef_stem = Path(str(e.get("file") or "")).stem.lower()
            if not ef_stem:
                # Fix 5: infer file stem from source/target node ID
                src_id = str(e.get("source") or "")
                tgt_id = str(e.get("target") or "")
                ef_stem = _file_stem_from_node_id(src_id) or _file_stem_from_node_id(tgt_id)
            if ef_stem and ef_stem != stem_lower:
                continue  # foreign-file setter — drop
        kept_edges.append(e)

    # Fix 3: deduplicate assign/define edges that share the same (line, expression)
    # — alias fan-out produces multiple edges per physical statement.  Keep the
    # entry with the most informative scope (function-scoped > file-scoped > empty).
    deduped_edges: List[Dict[str, Any]] = []
    _stmt_seen: Dict[Tuple[Any, str], int] = {}  # (line, expr) → index in deduped_edges
    for e in kept_edges:
        rel = str(e.get("relation") or "").lower()
        if rel in ("assign", "define"):
            line = e.get("line")
            expr = str(e.get("expression") or "")
            key = (line, expr)
            scope = str(e.get("scope") or "")
            idx = _stmt_seen.get(key)
            if idx is None:
                _stmt_seen[key] = len(deduped_edges)
                deduped_edges.append(e)
            else:
                # Replace existing entry if this one has a better (function) scope
                existing_scope = str(deduped_edges[idx].get("scope") or "")
                if scope and not existing_scope:
                    deduped_edges[idx] = e
        else:
            deduped_edges.append(e)

    # Recompute dep_vars from kept setter edges' source nodes (same normalization
    # the call_graph nodes use, so flat dep_vars and per-node dep_vars agree).
    dep_vars: List[str] = []
    seen_dv: Set[str] = set()
    referenced: Set[str] = set()
    for e in deduped_edges:
        src = str(e.get("source") or "").strip()
        tgt = str(e.get("target") or "").strip()
        if src:
            referenced.add(src)
        if tgt:
            referenced.add(tgt)
        if str(e.get("relation") or "").lower() in _CPP_SETTER_RELATIONS:
            name = _normalize_cpp_dep_name(node_lookup.get(src), src)
            if name and name not in seen_dv:
                seen_dv.add(name)
                dep_vars.append(name)

    kept_nodes = [n for n in (cpp_trace.get("nodes") or []) if str(n.get("id") or "") in referenced]

    # Guard condition dep_vars: the tracer stores them in "_gc_dep_vars" (a separate
    # key from the data-flow dep_vars) so they survive this dep_vars recomputation.
    # Filter them to those in the target file (guard_conditions carry a "file" field).
    for gc_dv in (cpp_trace.get("_gc_dep_vars") or []):
        if gc_dv and gc_dv not in seen_dv:
            seen_dv.add(gc_dv)
            dep_vars.append(gc_dv)

    scoped = dict(cpp_trace)
    scoped["edges"] = deduped_edges
    scoped["nodes"] = kept_nodes
    scoped["dep_vars"] = dep_vars

    # Issue-14: when file-scoping empties the top-level nodes/edges but
    # per_seed_traces still contain data, rebuild the top-level as the union
    # of per-seed data so consumers iterating cpp_lineage_trace.nodes see a
    # consistent view rather than an empty list alongside populated per-seed
    # sub-traces.
    if not kept_nodes and not deduped_edges:
        pst = scoped.get("per_seed_traces") or {}
        if pst:
            _merged_n: Dict[str, Dict[str, Any]] = {}
            _merged_e: List[Dict[str, Any]] = []
            _seen_ekeys: Set[Tuple[str, str, str, Any]] = set()
            for _ps_val in pst.values():
                for n in (_ps_val.get("nodes") or []):
                    nid = str(n.get("id") or "")
                    if nid and nid not in _merged_n:
                        _merged_n[nid] = n
                for e in (_ps_val.get("edges") or []):
                    ek = (
                        str(e.get("source") or ""),
                        str(e.get("target") or ""),
                        str(e.get("relation") or ""),
                        e.get("line"),
                    )
                    if ek not in _seen_ekeys:
                        _seen_ekeys.add(ek)
                        _merged_e.append(e)
            scoped["nodes"] = list(_merged_n.values())
            scoped["edges"] = _merged_e

    return scoped


# ── Per-GVL compact projections (PERF) ────────────────────────────────────────
# find_cpp_seed_nodes and _filter_seeds_to_file stream the ENTIRE GVL per
# distinct variable (DB fetchone + json.loads per row for DB-backed GVLs —
# profiled at ~70% of dep-var BFS wall-clock).  These projections materialize,
# ONCE per GVL instance, exactly the small fields those scans read, so each
# subsequent per-variable scan is an in-memory loop / set lookup:
#   seed_view       — nodes as {id,name,kind,metadata:{asm_alias,kind}} plus
#                     only the cpp_field_to_asm stitch edges (Strategy 3)
#   targets_by_stem — setter-edge target ids grouped by source-file stem
#                     (what _filter_seeds_to_file recomputed per call)
# Keyed by id(gvl) with the instance held in the value so the id cannot be
# recycled.  Bounded FIFO; _GVL_PROJ_CAP is a TOTAL item budget (nodes +
# stitch edges + setter targets) so RAM stays bounded on enormous GVLs —
# beyond it the original streaming behavior is kept (projection disabled,
# cached as None so the over-cap scan is paid at most once per instance).
# At ~300-500 B per projected item the default 1M-item budget keeps one
# projection under ~400 MB; in practice a run touches ONE global GVL.

_GVL_PROJ_CACHE: "OrderedDict[int, Tuple[Any, Optional[Dict[str, Any]]]]" = OrderedDict()
_GVL_PROJ_MAX = 2
_GVL_PROJ_CAP = int(os.environ.get(
    "ASM_GVL_PROJ_CAP",
    os.environ.get("ASM_GVL_PROJ_NODE_CAP", "1000000"),  # legacy name honored
))


def clear_gvl_projection_cache() -> None:
    """Release the module-level GVL projections (and the gvl refs they pin).

    Bounded at _GVL_PROJ_MAX entries (~400 MB each worst case) but
    module-level, so in a long-lived Celery worker they would otherwise
    persist across jobs until worker recycling.  Called at end-of-task next
    to the other cache teardowns (close_reach_facts_reader /
    close_db_seed_state)."""
    _GVL_PROJ_CACHE.clear()


def _gvl_db_info(gvl: Any) -> Optional[Tuple[str, str]]:
    """(db_path, job_id) when *gvl* is DB-backed (IndexDbGVL / DbLazyGVL)."""
    db = getattr(gvl, "_db_path", None)
    job = getattr(gvl, "_job_id", None)
    if db and job:
        return str(db), str(job)
    return None


def _gvl_projection(gvl: Any) -> Optional[Dict[str, Any]]:
    """Build (or fetch) the compact projection for *gvl*; None = too big."""
    key = id(gvl)
    hit = _GVL_PROJ_CACHE.get(key)
    if hit is not None:
        _GVL_PROJ_CACHE.move_to_end(key)
        return hit[1]

    nodes_compact: List[Dict[str, Any]] = []
    proj: Optional[Dict[str, Any]] = None
    items = 0
    over_cap = False
    for n in (gvl.get("nodes") or []):
        items += 1
        if items > _GVL_PROJ_CAP:
            over_cap = True
            break
        meta = n.get("metadata") or {}
        nodes_compact.append({
            "id": n.get("id"),
            "name": n.get("name"),
            "kind": n.get("kind"),
            "metadata": {
                "asm_alias": meta.get("asm_alias"),
                "kind": meta.get("kind"),
            },
        })
    if not over_cap:
        stitch_edges: List[Dict[str, Any]] = []
        targets_by_stem: Dict[str, Set[str]] = {}
        for e in (gvl.get("edges") or []):
            expr = str(e.get("expression") or "")
            if "cpp_field_to_asm" in expr:
                items += 1
                stitch_edges.append({
                    "expression": expr,
                    "source": e.get("source"),
                    "target": e.get("target"),
                })
            if str(e.get("relation") or "").lower() in _CPP_SETTER_RELATIONS:
                f = str(e.get("file") or "")
                tail = f.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                dot = tail.rfind(".")
                stem = (tail[:dot] if dot > 0 else tail).lower()
                if stem:
                    tgt = str(e.get("target") or "").strip()
                    bucket = targets_by_stem.setdefault(stem, set())
                    if tgt and tgt not in bucket:
                        items += 1
                        bucket.add(tgt)
            if items > _GVL_PROJ_CAP:
                over_cap = True
                break
        if not over_cap:
            proj = {
                "seed_view": {"nodes": nodes_compact, "edges": stitch_edges},
                "targets_by_stem": targets_by_stem,
            }

    # over_cap → proj stays None and is cached as None: callers stream as before.
    _GVL_PROJ_CACHE[key] = (gvl, proj)
    while len(_GVL_PROJ_CACHE) > _GVL_PROJ_MAX:
        _GVL_PROJ_CACHE.popitem(last=False)
    return proj


def _filter_seeds_to_file(
    seed_ids: List[str],
    gvl: Any,
    file_stem: str,
) -> List[str]:
    """Restrict seeds to those actually written in *file_stem*.

    The GVL is GLOBAL — it spans every file in the codebase — so
    find_cpp_seed_nodes returns seed nodes for the variable from ALL files that
    write it.  A file-scoped trace (a C++ modifier, or a chain dep_var search
    bound to one stem) must only start from seeds set IN that file; otherwise
    setters from unrelated files (e.g. dw780000's selectCidDbRecord15 leaking
    into a dw710000 trace) pollute the output.  This mirrors the ASM modifier
    path, which scans only the modifier's own blueprint blocks.

    A seed belongs to *file_stem* when it has an assign/define/arg_bind edge
    whose source file basename stem matches.  When *file_stem* never appears in
    any GVL edge (path-format mismatch), the seeds are returned unchanged so we
    never silently drop real setters.
    """
    if not seed_ids or not file_stem or not gvl:
        return seed_ids
    stem_lower = str(file_stem).strip().lower()

    # Tier 1 (scale): DB-backed per-stem setter-target lookup via the
    # (job_id, relation, file_stem) index — no GVL pass at all, RAM ≈ one
    # file's setter targets.  This is the path large corpora take, where the
    # in-RAM projection would exceed its cap.
    _dbi = _gvl_db_info(gvl)
    if _dbi is not None:
        from backward_traversal.utils.cpp_lineage_utils import (
            db_setter_targets_for_stem,
        )
        _db_res = db_setter_targets_for_stem(_dbi[0], _dbi[1], stem_lower)
        if _db_res is not None:
            in_file_db, stem_seen_db = _db_res
            if not stem_seen_db:
                # stem absent from every GVL setter edge — likely path-format
                # mismatch; do not drop seeds (avoid a false-empty trace).
                return seed_ids
            return [s for s in seed_ids if s in in_file_db]

    # Tier 2: per-GVL projection (one edge pass per GVL instead of one per
    # call).  Tier 3: streaming, when the GVL is over the projection cap.
    proj = _gvl_projection(gvl)
    if proj is not None:
        targets_by_stem = proj["targets_by_stem"]
        if stem_lower not in targets_by_stem:
            # file_stem is absent from every GVL setter edge — likely a
            # path-format mismatch; do not drop seeds (avoid a false-empty trace).
            return seed_ids
        in_file = targets_by_stem[stem_lower]
        return [s for s in seed_ids if s in in_file]

    seed_set: Set[str] = set(seed_ids)
    in_file: Set[str] = set()
    stem_seen_anywhere = False
    for e in (gvl.get("edges") or []):
        if str(e.get("relation") or "").lower() not in _CPP_SETTER_RELATIONS:
            continue
        ef_stem = Path(str(e.get("file") or "")).stem.lower()
        if ef_stem == stem_lower:
            stem_seen_anywhere = True
            tgt = str(e.get("target") or "").strip()
            if tgt in seed_set:
                in_file.add(tgt)
    if not stem_seen_anywhere:
        # file_stem is absent from every GVL edge — likely a path-format
        # mismatch; do not drop seeds (avoid a false-empty trace).
        return seed_ids
    return [s for s in seed_ids if s in in_file]


def _build_cpp_cg_nodes(
    cpp_trace: Dict[str, Any],
    source_index: Dict[str, str],
    restrict_to_stem: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build enriched function-level call_graph nodes from a C++ GVL backward trace.

    Equivalent to the ASM call_graph nodes produced by trace_asm_call_site_backward:
      - "id"                  : function name (extracted from GVL node IDs)
      - "function_code"       : source text of the function
      - "relevant_code_lines" : per-line entries with role, inst, detail
                                (sourced from GVL edges: assign/define/arg_bind)
      - "dependent_variables" : variable names that flow INTO the traced field
                                in this function (sourced from edge source nodes)

    The GVL nodes do NOT carry metadata.enclosing_function; the function is
    extracted from the node ID itself ("kind:fn::path" → "fn").

    restrict_to_stem:
        When set, only edges whose statement physically lives in that file
        (edge.file basename stem matches) produce relevant_code_lines and
        function nodes.  The global GVL BFS legitimately crosses files (backward
        data flow), but a file-scoped call_graph (modifier / chain dep_var) must
        not surface another file's independent setters as if they were local.
        Cross-file source variables are still captured as dep_var NAMES because
        they are the source nodes of in-file edges.
    """
    _restrict = str(restrict_to_stem).strip().lower() if restrict_to_stem else ""
    # ── index nodes by ID ────────────────────────────────────────────────────
    node_lookup: Dict[str, Dict[str, Any]] = {
        str(n.get("id") or ""): n
        for n in (cpp_trace.get("nodes") or [])
        if n.get("id")
    }

    # ── discover all functions from trace nodes ───────────────────────────────
    fn_data: Dict[str, Dict[str, Any]] = {}

    def _ensure_fn(fn: str) -> None:
        if fn and fn not in fn_data:
            fn_data[fn] = {"rcl": [], "dep_vars": set(), "seen_lines": set()}

    for n in (cpp_trace.get("nodes") or []):
        # Try metadata first; fall back to ID parsing (most common case)
        enc_fn = str((n.get("metadata") or {}).get("enclosing_function") or "").strip()
        if not enc_fn:
            enc_fn = _enc_fn_from_gvl_id(str(n.get("id") or ""))
        _ensure_fn(enc_fn)

    # ── process edges: relevant_code_lines + dep_vars ─────────────────────────
    _seen_stmt_keys: Set[Tuple[str, int, str]] = set()  # (fn, line, expr[:80])

    for e in (cpp_trace.get("edges") or []):
        relation = str(e.get("relation") or "").lower()
        if relation not in _CPP_RELATION_TO_ROLE:
            continue
        expr  = str(e.get("expression") or "").strip()
        line  = e.get("line") or 0
        if not line:
            continue

        # File scoping: when restricted, drop edges whose statement lives in
        # another file (keeps cross-file dep_var names via in-file edge sources,
        # but not foreign functions' independent setter lines).  Edges with no
        # file info are kept (cannot be confirmed out-of-file).
        if _restrict:
            _ef_stem = Path(str(e.get("file") or "")).stem.lower()
            if _ef_stem and _ef_stem != _restrict:
                continue

        # Determine enclosing function from the TARGET node
        tgt_id   = str(e.get("target") or "")
        tgt_node = node_lookup.get(tgt_id) or {}
        enc_fn   = str((tgt_node.get("metadata") or {}).get("enclosing_function") or "").strip()
        if not enc_fn:
            enc_fn = _enc_fn_from_gvl_id(tgt_id)
        if not enc_fn:
            continue
        _ensure_fn(enc_fn)
        fd = fn_data[enc_fn]

        # relevant_code_lines — deduplicate by (fn, line, expr)
        stmt_key = (enc_fn, line, expr[:80])
        if stmt_key not in _seen_stmt_keys:
            _seen_stmt_keys.add(stmt_key)
            if line not in fd["seen_lines"]:
                fd["seen_lines"].add(line)
                fd["rcl"].append({
                    "line":   line,
                    "inst":   relation.upper(),
                    "role":   _CPP_RELATION_TO_ROLE[relation],
                    "detail": expr.replace("\n", " "),
                })

            # Extract conditions from source that guard this setter line.
            # The GVL edge carries the source file path and setter line number;
            # scan backward in the source for if-conditions (the C++ equivalent
            # of ASM CONDITION_CHECK + BRANCH_STEP pairs).
            src_file = str(e.get("file") or "")
            if src_file:
                for cond in _extract_cpp_source_conditions(src_file, line):
                    cond_line = cond["line"]
                    if cond_line not in fd["seen_lines"]:
                        fd["seen_lines"].add(cond_line)
                        fd["rcl"].append({
                            "line":   cond_line,
                            "inst":   "IF",
                            "role":   "CONDITION_CHECK",
                            "detail": cond["check"],
                        })
                    # Condition variables are dep_vars too (conditional path vars)
                    for cv in cond.get("vars_checked") or []:
                        if cv and not _is_noise_dep_var(cv.upper()):
                            fd["dep_vars"].add(cv)

        # dependent_variables — source node's terminal name (shared normalizer)
        src_id   = str(e.get("source") or "")
        src_name = _normalize_cpp_dep_name(node_lookup.get(src_id), src_id)
        if src_name:
            fd["dep_vars"].add(src_name)

    # ── assemble output nodes ────────────────────────────────────────────────
    cg_nodes: List[Dict[str, Any]] = []
    for fn_name, fd in fn_data.items():
        if not fn_name:
            continue
        rcl = sorted(fd["rcl"], key=lambda r: r.get("line") or 0)
        # When file-scoped, drop functions that contributed no in-file content —
        # they only appeared because the BFS passed through them in another file.
        if _restrict and not rcl and not fd["dep_vars"]:
            continue
        node: Dict[str, Any] = {
            "id":            fn_name,
            "function_code": source_index.get(fn_name, ""),
        }
        if rcl:
            node["relevant_code_lines"] = rcl
        if fd["dep_vars"]:
            node["dependent_variables"] = sorted(fd["dep_vars"])
        cg_nodes.append(node)

    return cg_nodes


def _build_modifier_cpp_payload(
    variable: str,
    modifier_stem: str,
    blueprint_dir: Path,
    asm_dir: Optional[Path],
    target_setters: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the modifier payload when the modifier file is C++.  (T1)

    Traces `variable` backward in the C++ GVL of `modifier_stem`.  There are
    no ASM setter_sites; dep_vars are extracted from cpp_lineage_trace.dep_vars
    during Pass 2 (T2).
    """
    cpp_bp_path = resolve_cpp_blueprint(modifier_stem, blueprint_dir)
    if not cpp_bp_path:
        return {
            "role": "modifier",
            "file_type": "cpp",
            "setter_sites": [],
            "setter_site_traces": [],
            "cpp_lineage_trace": {},
            "call_graph": {"nodes": [], "edges": []},
            "warnings": [f"C++ blueprint not found for modifier {modifier_stem}"],
        }

    try:
        bp = load_json(cpp_bp_path, keys={"variable_lineage", "field_asm_aliases"})
    except Exception as exc:
        return {
            "role": "modifier",
            "file_type": "cpp",
            "setter_sites": [],
            "setter_site_traces": [],
            "cpp_lineage_trace": {},
            "call_graph": {"nodes": [], "edges": []},
            "warnings": [f"Failed to load C++ blueprint for modifier {modifier_stem}: {exc}"],
        }

    from backward_traversal.utils.lazy_gvl import make_lazy_gvl
    # Pass cpp_bp_path so make_lazy_gvl can fall back to DbLazyGVL when the
    # global_variable_lineage.json file has been deleted after DB ingestion.
    gvl = make_lazy_gvl(bp, blueprint_path=cpp_bp_path)
    asm_field_aliases = extract_asm_field_aliases(bp, variable) if variable else {}
    seed_ids = find_cpp_seed_nodes(gvl, variable, asm_field_aliases, {}) if variable else []
    # GVL is global — restrict modifier seeds to setters actually in this file,
    # mirroring the ASM modifier path (which scans only its own blueprint blocks).
    _pre_filter_count = len(seed_ids)
    seed_ids = _filter_seeds_to_file(seed_ids, gvl, modifier_stem)

    if not seed_ids:
        _warn = f"No C++ seed nodes found for '{variable}' in modifier {modifier_stem}"
        if _pre_filter_count:
            _warn += (
                f" ({_pre_filter_count} seed(s) found globally but none had "
                f"assign/define edges in '{modifier_stem}')"
            )
        return {
            "role": "modifier",
            "file_type": "cpp",
            "setter_sites": [],
            "setter_site_traces": [],
            "asm_field_aliases": asm_field_aliases,
            "cpp_lineage_trace": {},
            "call_graph": {"nodes": [], "edges": []},
            "warnings": [_warn],
        }

    per_file_lineage = bp.get("variable_lineage") or {}
    cpp_trace = trace_cpp_variable_backward_multi(
        seed_ids=seed_ids,
        gvl=gvl,
        max_depth=DEFAULT_CPP_MAX_DEPTH,
        max_nodes=DEFAULT_CPP_MAX_NODES,
        per_file_lineage=per_file_lineage,
        variable_name=variable,
    )
    # Scope the global trace to this file so every consumer (call_graph nodes AND
    # the LLM scaffold, which reads cpp_lineage_trace directly) sees only setters
    # that physically occur in the modifier file — matching the ASM modifier path.
    cpp_trace = _scope_cpp_trace_to_file(cpp_trace, modifier_stem)

    # Build enriched function-level call_graph nodes from the scoped GVL trace.
    source_index = build_func_source_index(blueprint_dir, asm_dir, modifier_stem)
    cg_nodes = _build_cpp_cg_nodes(cpp_trace, source_index, restrict_to_stem=modifier_stem)

    setter_sites = _extract_cpp_setter_sites(cpp_trace, variable)
    if target_setters:
        _all_sites_pre_filter = setter_sites
        setter_sites = _filter_setter_sites(setter_sites, target_setters, file_stem=modifier_stem)
        _scope_cpp_dep_vars_to_target_setters(cpp_trace, setter_sites, _all_sites_pre_filter)

    # Build call-chain conditions: for each setter function, walk up the GVL
    # arg_bind edges to find the conditions at each hop in the call chain.
    call_chain_conditions: List[Dict[str, Any]] = []
    seen_setters: set = set()
    for site in setter_sites:
        fn = site.get("routine", "")
        if fn and fn not in seen_setters and fn != "(global)":
            seen_setters.add(fn)
            chain = _extract_cpp_call_chain_conditions(fn, gvl)
            call_chain_conditions.extend(chain)

    # Issue #5 fix: populate setter_site_traces from per-seed traces so
    # consumers have per-seed condition data alongside setter_sites.
    _pst = list((cpp_trace.get("per_seed_traces") or {}).values())
    result: Dict[str, Any] = {
        "role": "modifier",
        "file_type": "cpp",
        "setter_sites": setter_sites,
        "setter_site_traces": _pst,
        "asm_field_aliases": asm_field_aliases,
        "cpp_lineage_trace": cpp_trace,
        "call_graph": {"nodes": cg_nodes, "edges": []},
        "warnings": cpp_trace.get("warnings") or [],
    }
    if call_chain_conditions:
        result["call_chain_conditions"] = call_chain_conditions
    return result


def _build_dep_var_cpp_result(
    dep_var: str,
    gvl: Dict[str, Any],
    resolver: Optional[Any] = None,         # VariableNameResolver instance (FIX 19)
    stem: str = "",                          # target C++ file stem (FIX 19)
    source_stem: Optional[str] = None,       # preceding C++ stem for name translation (FIX 19 / T10)
    blueprint_dir: Optional[Path] = None,   # P0-2: for building function-level call_graph nodes
    asm_dir: Optional[Path] = None,         # P0-2: for build_func_source_index source lookup
    seed_cache: Optional[Dict] = None,      # per-run cache: {(id(gvl), var_lower, stem): [seed_ids]}
    bulk_seed_map: Optional[Dict] = None,  # Fix 3: pre-fetched seeds {(var_lower, stem): [seed_ids]}
    bulk_seed_unfiltered: bool = False,     # True when bulk_seed_map seeds are NOT file-scoped
    _reverse_idx: Any = None,               # pre-built reverse index (memoized across dep_vars)
    _node_lookup: Any = None,               # pre-built node lookup  (memoized across dep_vars)
    source_index_cache: Optional[Dict[str, Dict[str, str]]] = None,  # per-stem cache for build_func_source_index
) -> Dict[str, Any]:
    """Trace dep_var backward in a C++ file's GVL.  Used in Pass 2.  (T3)

    dep_var is already a normalized C++ field name (after FIX 11/21 strip), so
    Strategy 5 in find_cpp_seed_nodes() is the primary finder.  ASM-alias
    lookup is intentionally skipped (dep_vars are C++ names, not ASM names).
    FIX 16: the fallback uses exact suffix match, not substring.
    FIX 19: translate dep_var to its local name in this C++ file before seed
    finding, using resolver.resolve(dep_var, stem, source_stem=source_stem).
    """
    # FIX 19: translate dep_var to its local name in this file.
    # resolver.resolve() returns None when no translation is found — the
    # fallback retains the original name (same as before FIX 19).
    effective_dv = dep_var
    if resolver and stem:
        translated = resolver.resolve(dep_var, stem, source_stem=source_stem)
        if translated:
            effective_dv = translated

    # Opt C: check once whether the DB seed index is queryable.
    # When True, the suffix-scan fallback (FIX 16) is skipped — the derived-
    # column engine covers Strategies 1, 3, 5, 6, 7, 8 exactly, and
    # Strategies 2 & 4 are disabled for the dep_var path (empty
    # asm_field_aliases/tpf_regs_slots).
    # CONVERGENCE: the materialized cpp_seed_keys/cpp_seed_file_keys probes
    # were replaced by the derived-column readiness check.  Our engine
    # always file-scopes when a stem is given (via db_setter_targets_for_stem
    # inside get_cpp_seed_node_ids), so both flags share one probe.
    _has_db_seed_index = False
    _has_file_scoped_seeds = False
    _job_id_c = getattr(gvl, "_job_id", None)
    _db_path_c = getattr(gvl, "_db_path", None)
    if _job_id_c and _db_path_c:
        try:
            from backward_traversal.utils.cpp_lineage_utils import (
                db_seed_index_ready,
            )
            if db_seed_index_ready(str(_db_path_c), str(_job_id_c)):
                _has_db_seed_index = True
                _has_file_scoped_seeds = True
        except Exception:
            pass

    # FIX 17: skip extract_asm_field_aliases — dep_var is a C++ name
    # _db_did_file_filter is True ONLY when the DB returns file-scoped seeds
    # (i.e. cpp_seed_file_keys is populated).  When cpp_seed_file_keys is empty,
    # get_cpp_seed_node_ids falls back to the unfiltered cpp_seed_keys table,
    # so _filter_seeds_to_file must still run.
    _db_did_file_filter = _has_file_scoped_seeds and bool(stem)
    # Fix 3: check bulk_seed_map first (no GVL id dependency).
    _bulk_key = (effective_dv.lower(), stem)
    _from_bulk = False
    if bulk_seed_map is not None and _bulk_key in bulk_seed_map:
        seed_ids = list(bulk_seed_map[_bulk_key])
        _from_bulk = True
        if bulk_seed_unfiltered:
            _db_did_file_filter = False
    elif seed_cache is not None:
        _ck = (id(gvl), effective_dv.lower(), stem)
        if _ck in seed_cache:
            seed_ids = list(seed_cache[_ck])
        else:
            seed_ids = find_cpp_seed_nodes(gvl, effective_dv, {}, {}, file_stem=stem or None, skip_scan_fallback=True)
            seed_cache[_ck] = seed_ids
    else:
        seed_ids = find_cpp_seed_nodes(gvl, effective_dv, {}, {}, file_stem=stem or None, skip_scan_fallback=True)

    if not seed_ids:
        # Hop-based seed acceleration (Change 4): try variable_hops before the
        # expensive suffix scan.  Indexed O(log N) lookup of node_ids where this
        # variable's canonical_key appears with relation='target' in this file.
        _hop_job = getattr(gvl, "_job_id", None)
        if stem and _hop_job:
            try:
                from api.index_db.readers import get_canonical_key, get_hop_seed_node_ids
                _hop_ck = get_canonical_key(_hop_job, effective_dv.upper())
                if _hop_ck:
                    _hop_seeds = get_hop_seed_node_ids(_hop_job, _hop_ck, stem)
                    if _hop_seeds:
                        seed_ids = _hop_seeds
                        _db_did_file_filter = True  # hops are file-scoped
            except Exception:
                pass

    if not seed_ids and not _has_db_seed_index:
        # FIX 16: exact suffix fallback scan (same standard as FIX 9)
        # FIX-new-3: also match the "->" pointer-dereference form so struct fields
        # (e.g. "struct_field:fn::ptr->fieldName") found by Strategy 5 in Pass 1
        # are equally found here in the Pass 2 fallback.
        # Opt C: skipped when the DB seed index is populated — Strategies 1-8
        # are fully indexed so this scan cannot find seeds the DB missed.
        _db_did_file_filter = False  # fallback scan is global, needs file filtering
        dep_lower = effective_dv.lower()
        seen_fallback: Set[str] = set()
        for n in (gvl.get("nodes") or []):
            nid = str(n.get("id") or "").lower()
            nname = str(n.get("name") or "").lower()
            if (nid.endswith(f"::{dep_lower}")
                    # Align with Strategy 5 GAP-012 threshold (>= 6) to prevent
                    # false-seed explosions for short names like "addr", "flag".
                    or (len(dep_lower) >= 6 and nid.endswith(f"->{dep_lower}"))
                    or (len(dep_lower) >= 6 and nid.endswith(f".{dep_lower}"))
                    or nname == dep_lower):
                sid = str(n.get("id") or n.get("name") or "").strip()
                if sid and sid not in seen_fallback:
                    seen_fallback.add(sid)
                    seed_ids.append(sid)

    # GVL is global — restrict this chain stem's dep_var seeds to setters in this
    # file so the same setter isn't over-attributed to every chain stem.  When
    # the dep_var is genuinely set in another file the result is empty and the
    # caller's T8 cross-chain search (when enabled) extends the lookup.
    # Skip when the DB fast path already did file-scoped filtering.
    _pre_filter_count = len(seed_ids)
    if stem and not _db_did_file_filter:
        seed_ids = _filter_seeds_to_file(seed_ids, gvl, stem)

    # Fix 4: Seed cap for dep_var path.  Seeds are already priority-ordered
    # from get_cpp_seed_node_ids (ORDER BY priority).  Truncating prevents
    # BFS explosion for variables with many seeds (e.g. CIDRECORD 42 → 10).
    if _MAX_SEEDS_PER_DEP_VAR > 0 and len(seed_ids) > _MAX_SEEDS_PER_DEP_VAR:
        logger.info(
            "[BACKWARD] seed cap: '%s' in '%s' — %d → %d seeds",
            dep_var, stem, len(seed_ids), _MAX_SEEDS_PER_DEP_VAR,
        )
        seed_ids = seed_ids[:_MAX_SEEDS_PER_DEP_VAR]

    if not seed_ids:
        _warn = f"No C++ seed nodes found for dep_var '{dep_var}' (effective: '{effective_dv}')"
        if _pre_filter_count:
            _warn += (
                f" ({_pre_filter_count} seed(s) found globally but none had "
                f"assign/define edges in '{stem}')"
            )
        return {
            "dep_var": dep_var,
            "file_type": "cpp",
            "lineage_trace": {},
            "setter_sites": [],
            "call_graph": {},
            "warnings": [_warn],
        }

    cpp_trace = trace_cpp_variable_backward_multi(
        seed_ids=seed_ids,
        gvl=gvl,
        max_depth=DEFAULT_CPP_MAX_DEPTH,
        max_nodes=DEFAULT_CPP_MAX_NODES,
        _reverse_idx=_reverse_idx,
        _node_lookup=_node_lookup,
    )
    # Scope the global trace to this stem so per-stem dep_var results aren't
    # over-attributed setters from other files (consistent across cg_nodes and
    # the LLM scaffold which reads lineage_trace directly).
    if stem:
        cpp_trace = _scope_cpp_trace_to_file(cpp_trace, stem)
    # Build enriched function-level call_graph nodes from the scoped GVL trace.
    # Use cached source index when available to avoid re-reading source files.
    if source_index_cache is not None and stem and stem in source_index_cache:
        _source_index = source_index_cache[stem]
    else:
        _source_index = build_func_source_index(blueprint_dir, asm_dir, stem) if (blueprint_dir and stem) else {}
        if source_index_cache is not None and stem:
            source_index_cache[stem] = _source_index
    _cg_nodes = _build_cpp_cg_nodes(cpp_trace, _source_index, restrict_to_stem=stem or None)
    return {
        "dep_var": dep_var,
        "file_type": "cpp",
        "lineage_trace": cpp_trace,
        "setter_sites": _extract_cpp_setter_sites(cpp_trace, effective_dv),
        "setter_site_traces": list((cpp_trace.get("per_seed_traces") or {}).values()),
        "call_graph": {"nodes": _cg_nodes, "edges": []},
        "warnings": cpp_trace.get("warnings") or [],
    }


def _process_one_ext_file(
    stem: str,
    source_stem: str,
    variable: str,
    blueprint_dir: Path,
    asm_dir: Optional[Path],
    file_type_map: Dict[str, str],
    resolver: Any,
    max_depth: int,
    max_subroutine_depth: int,
    max_subroutine_nodes: int,
    max_trace_nodes: int,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Per-file work for extend_modifier_sites — no shared state."""
    stem_type = resolve_file_type(stem, blueprint_dir, file_type_map)
    ext_var = (resolver.resolve(variable, stem, source_stem=source_stem) or variable).upper()

    if stem_type == "asm":
        bp = resolve_asm_blueprint(stem, blueprint_dir)
        if not bp:
            return (stem, None)
        asm_file = resolve_source_file(stem, "asm", blueprint_dir, asm_dir)
        bp_data = load_json(bp, keys={"blocks", "symbols"})
        all_blocks: Set[str] = {str(b["id"]) for b in bp_data.get("blocks", [])}
        # PERF-OPT-8: lightweight scan — find setter sites only, skip the
        # expensive trace_asm_call_site_backward() calls.  Pass 2 does NOT
        # iterate extended files (line 5102 iterates selected_chain only),
        # so full backward traces here are wasted CPU and RAM.  Dep-var
        # seeding uses setter_expression values from the sites themselves.
        _const_syms = collect_constant_symbols(
            bp_data,
            asm_file=asm_file if asm_file and asm_file.exists() else None,
            bp_path=bp,
        )
        setter_sites, site_warnings = _find_scoped_setter_sites(
            ext_var, bp_data, asm_file, all_blocks, asm_dir,
            constant_symbols=_const_syms,
        )
        result: Dict[str, Any] = {
            "scope": "full_file",
            "setter_sites": setter_sites,
            "setter_site_traces": [],
            "call_graph": {"nodes": [], "edges": []},
            "subroutine_summary": {},
            "warnings": sorted(set(str(w) for w in site_warnings if w)),
            "file_type": "asm",
            "source_file": stem,
        }
        for site in setter_sites:
            site["source_file"] = stem
        return (stem, result)

    elif stem_type == "cpp":
        cpp_bp_path = resolve_cpp_blueprint(stem, blueprint_dir)
        if not cpp_bp_path:
            return (stem, None)
        cpp_bp = load_json(cpp_bp_path, keys={"call_graph"})
        from backward_traversal.utils.lazy_gvl import make_lazy_gvl
        gvl = make_lazy_gvl(cpp_bp, blueprint_path=cpp_bp_path)
        result = _build_dep_var_cpp_result(
            ext_var, gvl,
            resolver=resolver, stem=stem, source_stem=source_stem,
            blueprint_dir=blueprint_dir, asm_dir=asm_dir,
        )
        result["file_type"] = "cpp"
        result["source_file"] = stem
        for site in result.get("setter_sites", []):
            site["source_file"] = stem
        return (stem, result)

    return (stem, None)


# PERF-OPT-5/6: thread pool size for parallel per-file scanning in Pass 1a/1b.
# Keep the historical default, but allow low-RAM hosts to dial it down without
# changing code (for example ASM_PARALLEL_WORKERS=1 on 13 GB machines).
_PARALLEL_WORKERS = _env_int(
    "ASM_PARALLEL_WORKERS",
    min(4, os.cpu_count() or 4),
    1,
)


def _get_rss_mb() -> float:
    """Return current process RSS in MB.  Returns 0 on failure."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KB.
        import platform
        if platform.system() == "Darwin":
            return ru / (1024 * 1024)
        return ru / 1024
    except Exception:
        return 0.0


# PERF-OPT-8: abort BFS early if process RSS exceeds this limit (MB).
# Set ASM_MAX_RSS_MB in the environment to override.  If unset, auto-
# detect from Docker cgroup memory limit (container_limit − 1 GB).
# Falls back to 0 (disabled) on bare metal.
def _default_max_rss_mb() -> int:
    explicit = os.environ.get("ASM_MAX_RSS_MB")
    if explicit:
        return int(explicit)
    try:
        # cgroup v2 (Docker ≥ 20.10)
        _cg2 = Path("/sys/fs/cgroup/memory.max")
        if _cg2.exists():
            val = _cg2.read_text().strip()
            if val != "max":
                return max(512, int(val) // (1024 * 1024) - 1024)
        # cgroup v1
        _cg1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if _cg1.exists():
            val = int(_cg1.read_text().strip())
            if val < 2**62:  # sentinel for "no limit"
                return max(512, val // (1024 * 1024) - 1024)
    except Exception:
        pass
    return 0

_MAX_RSS_MB = _default_max_rss_mb()

# Allow operators to skip the KB-wide declaration scan when the priority is the
# lineage trace itself rather than terminal-role enrichment.
_ENABLE_TERMINAL_CLASSIFICATION = _env_bool(
    "ASM_ENABLE_TERMINAL_CLASSIFICATION",
    True,
)


def _extend_modifier_setter_sites(
    modifier_stem: str,
    variable: str,
    blueprint_dir: Path,
    asm_dir: Optional[Path],
    edge_type_map: Dict[Tuple[str, str], str],
    file_type_map: Dict[str, str],
    max_depth: int,
    max_subroutine_depth: int,
    max_subroutine_nodes: int,
    max_trace_nodes: int,
    max_downstream_depth: int,
    max_downstream_files: int,
    resolver: Any,
    selected_chain_set: Set[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """BFS from modifier through outgoing call-graph edges.

    For each reachable file, scan for setter sites of *variable*.
    Returns ``(extended_files, warnings)`` where *extended_files* maps
    ``stem -> result_dict`` (same shape as ``_build_dep_var_file_result``).

    PERF-OPT-5: per-file work is executed in parallel within each BFS
    depth level using ThreadPoolExecutor.
    """
    # Build forward adjacency from the global call-graph edge map.
    forward_adj: Dict[str, Set[str]] = {}
    for (src, tgt) in edge_type_map:
        forward_adj.setdefault(src.strip().lower(), set()).add(tgt.strip().lower())

    visited: Set[str] = set()
    queue: Deque[Tuple[str, int, str]] = deque()  # (stem, depth, source_stem)
    extended_files: Dict[str, Any] = {}
    warnings: List[str] = []
    capped = False

    modifier_lower = _norm_file_id(modifier_stem)
    chain_lower = {_norm_file_id(s) for s in selected_chain_set}
    for callee in sorted(forward_adj.get(modifier_lower, set())):
        if callee not in chain_lower:
            queue.append((callee, 0, modifier_stem))

    while queue and not capped:
        # PERF-OPT-8: abort BFS early if memory pressure detected.
        if _MAX_RSS_MB > 0:
            _rss = _get_rss_mb()
            if _rss > _MAX_RSS_MB:
                warnings.append(
                    f"extend_modifier_sites: aborted — RSS {_rss:.0f} MB "
                    f"exceeds ASM_MAX_RSS_MB={_MAX_RSS_MB} after {len(visited)} files"
                )
                capped = True
                break

        # Drain queue into a batch of files eligible for processing.
        batch: List[Tuple[str, int, str, str]] = []  # (stem, depth, source_stem, norm)
        while queue:
            stem, depth, source_stem = queue.popleft()
            norm = _norm_file_id(stem)
            if norm in visited or norm in chain_lower:
                continue
            if len(visited) + len(batch) >= max_downstream_files:
                warnings.append(
                    f"extend_modifier_sites: max_downstream_files={max_downstream_files} cap reached"
                )
                queue.clear()
                capped = True
                break
            if depth >= max_downstream_depth:
                visited.add(norm)
                continue
            visited.add(norm)
            batch.append((stem, depth, source_stem, norm))

        if not batch:
            break

        # Process batch in parallel.
        if len(batch) == 1:
            # Single file — skip thread pool overhead.
            stem, depth, source_stem, norm = batch[0]
            stem_key, result = _process_one_ext_file(
                stem, source_stem, variable, blueprint_dir, asm_dir,
                file_type_map, resolver, max_depth, max_subroutine_depth,
                max_subroutine_nodes, max_trace_nodes,
            )
            if result:
                extended_files[stem_key] = result
                logger.info("[BACKWARD] Pass 1a — scanned %s (%s) — %d setter sites",
                            stem_key, result.get("file_type", "?"),
                            len(result.get("setter_sites", [])))
            for callee in sorted(forward_adj.get(norm, set())):
                if _norm_file_id(callee) not in visited:
                    queue.append((callee, depth + 1, stem))
        else:
            with ThreadPoolExecutor(max_workers=min(_PARALLEL_WORKERS, len(batch))) as pool:
                futures = {
                    pool.submit(
                        _process_one_ext_file, stem, source_stem, variable,
                        blueprint_dir, asm_dir, file_type_map, resolver,
                        max_depth, max_subroutine_depth, max_subroutine_nodes,
                        max_trace_nodes,
                    ): (stem, depth, norm)
                    for stem, depth, source_stem, norm in batch
                }
                for fut in as_completed(futures):
                    try:
                        stem_key, result = fut.result()
                    except Exception:
                        stem_key = futures[fut][0]
                        logger.debug("[BACKWARD] Pass 1a — error scanning %s", stem_key, exc_info=True)
                        result = None
                    fstm, fdepth, fnorm = futures[fut]
                    if result:
                        extended_files[stem_key] = result
                        logger.info("[BACKWARD] Pass 1a — scanned %s (%s) — %d setter sites",
                                    stem_key, result.get("file_type", "?"),
                                    len(result.get("setter_sites", [])))
                    # Enqueue callees for next depth level.
                    for callee in sorted(forward_adj.get(fnorm, set())):
                        if _norm_file_id(callee) not in visited:
                            queue.append((callee, fdepth + 1, fstm))

        # PERF-OPT-8: evict blueprints loaded for this batch so peak RAM
        # stays proportional to batch size, not total visited files.
        # On a 22K-file codebase each blueprint expands to ~65 MB in RAM;
        # without eviction 1000 files would need 65 GB.
        from backward_traversal.utils.blueprint_utils import (
            clear_blueprint_cache,
            clear_constant_symbols_cache,
        )
        clear_blueprint_cache()
        clear_constant_symbols_cache()
        clear_cfg_cache()
        _clear_cross_file_call_cache()
        import gc
        gc.collect()

    return extended_files, warnings


def run_backward_only(
    variable: str,
    selected_chain: List[str],
    blueprint_dir: Path,
    asm_dir: Optional[Path],
    graph_file: Path,
    output_path: Path,
    max_depth: int = 8,
    max_subroutine_depth: int = DEFAULT_MAX_SUBROUTINE_DEPTH,
    max_subroutine_nodes: int = DEFAULT_MAX_SUBROUTINE_NODES,
    max_trace_nodes: int = DEFAULT_MAX_TRACE_NODES,
    max_dep_vars: int = DEFAULT_MAX_DEP_VARS,
    max_dep_var_depth: int = DEFAULT_MAX_DEP_VAR_DEPTH,
    max_downstream_depth: int = DEFAULT_MAX_DOWNSTREAM_DEPTH,
    max_downstream_files: int = DEFAULT_MAX_DOWNSTREAM_FILES,
    t8_max_scan: int = 0,           # FIX 18: disabled by default; T9 neighbor_set must be provided at scale
    extend_modifier_sites: bool = False,
    target_setters: Optional[List[Dict[str, Any]]] = None,
    partitioned_output: bool = True,
    graph_payload: Optional[Dict[str, Any]] = None,
    prebuilt_file_type_map: Optional[Dict[str, str]] = None,
    prebuilt_edge_type_map: Optional[Dict[Tuple[str, str], str]] = None,
    prebuilt_edge_lines_map: Optional[Dict[Tuple[str, str], Set[int]]] = None,
    prebuilt_callsite_map: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]] = None,
) -> int:
    if not selected_chain:
        print("ERROR: --chain-files must include at least one file (modifier).", file=sys.stderr)
        return 1

    # PERF: skip loading the full graph_payload when all prebuilt index maps
    # are provided — the monolithic dict is never read once the four indexes
    # exist.  For 22K-file codebases this avoids a 400–1200 MB allocation.
    _all_prebuilt = (prebuilt_file_type_map is not None
                     and prebuilt_edge_type_map is not None
                     and prebuilt_edge_lines_map is not None
                     and prebuilt_callsite_map is not None)
    if graph_payload is None and not _all_prebuilt:
        graph_payload = load_json(graph_file)

    # PERF: accept pre-built index maps to avoid each thread rebuilding them
    # from scratch (~220 MB each on 22K-file codebases).  When called from
    # the orchestrator's ThreadPoolExecutor, these are built once and shared.
    if prebuilt_file_type_map is not None:
        file_type_map = prebuilt_file_type_map
    else:
        # GAP-007: build with lowercase duplicates so mixed-case stems from the
        # call-graph (e.g. "Dx740200") are always found by _norm_file_id (which
        # lowercases) without falling through to the filesystem probe.
        file_type_map = {}
        for _n in graph_payload.get("nodes", []):
            _nid = str(_n.get("id") or "")
            _t = str(_n.get("type") or "asm")
            file_type_map[_nid] = _t
            file_type_map[_nid.lower()] = _t

    edge_type_map = prebuilt_edge_type_map if prebuilt_edge_type_map is not None else _edge_type_map(graph_payload)
    edge_lines = prebuilt_edge_lines_map if prebuilt_edge_lines_map is not None else _edge_lines_map(graph_payload)
    callsite_map = prebuilt_callsite_map if prebuilt_callsite_map is not None else _edge_callsite_map(graph_payload)

    modifier_stem = selected_chain[-1]
    root_var = str(variable or "").upper()
    selected_chain_set: Set[str] = {_norm_file_id(s) for s in selected_chain}
    selected_chain_index: Dict[str, int] = {
        _norm_file_id(stem): idx for idx, stem in enumerate(selected_chain)
    }
    prev_chain_stem_map: Dict[str, Optional[str]] = {}
    prev_cpp_chain_stem_map: Dict[str, Optional[str]] = {}
    _last_cpp_chain_stem: Optional[str] = None
    for idx, stem in enumerate(selected_chain):
        stem_norm = _norm_file_id(stem)
        prev_chain_stem_map[stem_norm] = selected_chain[idx - 1] if idx > 0 else None
        prev_cpp_chain_stem_map[stem_norm] = _last_cpp_chain_stem
        if resolve_file_type(stem, blueprint_dir, file_type_map) == "cpp":
            _last_cpp_chain_stem = stem
    full_flow = _new_full_flow_collector()
    files: Dict[str, Any] = {}
    resolver = VariableNameResolver(blueprint_dir)

    # Partitioned output: treat output_path as a directory and write each dep_var
    # trace to its own file immediately after the BFS iteration completes.
    # Single-file output: accumulate everything in memory as before.
    if partitioned_output:
        out_dir = output_path if output_path.suffix == "" else output_path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        _dep_var_dir = out_dir / "dep_vars"
        # Clear stale dep_var files from any previous run so that re-runs
        # (e.g. after constant_source suppression removes all dep_vars) don't
        # leave orphaned files from the earlier BFS result.
        if _dep_var_dir.exists():
            import shutil as _shutil
            _shutil.rmtree(_dep_var_dir)
    else:
        out_dir = None
        _dep_var_dir = None
    _dep_var_writer = _DepVarWriter(_dep_var_dir)
    _block_collector = _FunctionBlockCollector()
    _block_source_cache: Dict[str, List[str]] = {}

    _run_t0 = time.monotonic()
    logger.info("[BACKWARD] Pass 1 — modifier=%s var=%s chain=%s — start",
                modifier_stem, root_var, selected_chain)

    modifier_type = resolve_file_type(modifier_stem, blueprint_dir, file_type_map)
    files[modifier_stem] = {"file_type": modifier_type}
    warnings: List[str] = []
    # GAP-TRUNC: tracks whether any cap was hit during this run; surfaced in
    # root_var.json, manifest.json, and each dep_var output file so consumers
    # can detect incomplete results without parsing warning strings.
    run_truncated: bool = False
    # Per-category truncation detail — which specific cap(s) caused truncation.
    # run_truncated (above) is the summary boolean kept for backward compatibility.
    _trunc_detail: Dict[str, bool] = {
        "rv_downstream_files": False,   # root-var downstream file count cap
        "rv_downstream_depth": False,   # root-var downstream depth cap
        "rv_downstream_memory": False,  # root-var downstream RSS memory cap
        "rv_downstream_not_applicable": False,  # Issue-15: no downstream files exist (not truncated, just empty)
        "dep_var_count": False,         # dep_var BFS count cap
        "dep_var_depth": False,         # dep_var BFS depth cap
        "dep_var_memory": False,        # dep_var BFS RSS memory cap
        "dv_downstream_files": False,   # per-dep_var downstream file count cap
        "dv_downstream_depth": False,   # per-dep_var downstream depth cap
    }

    for stem in selected_chain:
        roles = ["selected_chain"]
        if _norm_file_id(stem) == _norm_file_id(modifier_stem):
            roles.append("modifier")
        _ensure_full_flow_node(
            full_flow,
            stem,
            blueprint_dir=blueprint_dir,
            file_type_map=file_type_map,
            selected_chain_set=selected_chain_set,
            roles=roles,
            scope_var=root_var,
        )

    if modifier_type == "asm":
        modifier_payload = _build_modifier_asm_payload(
            variable=variable,
            modifier_stem=modifier_stem,
            blueprint_dir=blueprint_dir,
            asm_dir=asm_dir,
            max_depth=max_depth,
            max_subroutine_depth=max_subroutine_depth,
            max_subroutine_nodes=max_subroutine_nodes,
            max_trace_nodes=max_trace_nodes,
            target_setters=target_setters,
        )
        files[modifier_stem]["modifier"] = modifier_payload
        if modifier_payload.get("warnings"):
            warnings.extend(modifier_payload.get("warnings", []))
        # Phase-3: attach setter_sites to the modifier's full_flow node so they
        # appear in the file_flow_json output alongside the node's roles.
        _mod_node = full_flow["nodes"].get(_norm_file_id(modifier_stem))
        if _mod_node is not None and modifier_payload.get("setter_sites"):
            _mod_node["setter_sites"] = modifier_payload["setter_sites"]
        # Annotate setter sites with block_ref for function/block body lookup
        if modifier_payload.get("setter_sites"):
            _annotate_setter_sites(
                modifier_payload["setter_sites"],
                stem=modifier_stem, file_type="asm",
                collector=_block_collector,
                bp_data=load_json(resolve_asm_blueprint(modifier_stem, blueprint_dir)),
                asm_file=resolve_source_file(modifier_stem, "asm", blueprint_dir, asm_dir),
                _source_cache=_block_source_cache,
            )
    elif modifier_type == "cpp":
        # T1: C++ modifier — trace backward in GVL; no ASM setter sites
        modifier_payload = _build_modifier_cpp_payload(
            variable=variable,
            modifier_stem=modifier_stem,
            blueprint_dir=blueprint_dir,
            asm_dir=asm_dir,
            target_setters=target_setters,
        )
        files[modifier_stem]["modifier"] = modifier_payload
        if modifier_payload.get("warnings"):
            warnings.extend(modifier_payload.get("warnings", []))
        # Annotate C++ modifier setter sites with block_ref
        if modifier_payload.get("setter_sites"):
            _annotate_setter_sites(
                modifier_payload["setter_sites"],
                stem=modifier_stem, file_type="cpp",
                collector=_block_collector,
                blueprint_dir=blueprint_dir, asm_dir=asm_dir,
            )
    else:
        warnings.append(
            f"Modifier file '{modifier_stem}' is type '{modifier_type}'; only 'asm' and 'cpp' modifiers are supported."
        )

    # --- Pass 1a: extend modifier setter sites ---
    extended_modifier_files: Dict[str, Any] = {}
    if extend_modifier_sites:
        logger.info("[BACKWARD] Pass 1a — extend_modifier_sites BFS from %s", modifier_stem)
        extended_modifier_files, ext_warnings = _extend_modifier_setter_sites(
            modifier_stem=modifier_stem,
            variable=variable,
            blueprint_dir=blueprint_dir,
            asm_dir=asm_dir,
            edge_type_map=edge_type_map,
            file_type_map=file_type_map,
            max_depth=max_depth,
            max_subroutine_depth=max_subroutine_depth,
            max_subroutine_nodes=max_subroutine_nodes,
            max_trace_nodes=max_trace_nodes,
            max_downstream_depth=max_downstream_depth,
            max_downstream_files=max_downstream_files,
            resolver=resolver,
            selected_chain_set=selected_chain_set,
        )
        warnings.extend(ext_warnings)

        if extended_modifier_files:
            for ext_stem, ext_result in extended_modifier_files.items():
                if target_setters:
                    _ext_all_sites = ext_result.get("setter_sites", [])
                    ext_result["setter_sites"] = _filter_setter_sites(
                        _ext_all_sites, target_setters, file_stem=ext_stem,
                    )
                    # Scope dep_vars for C++ extended modifier files
                    _ext_lt = ext_result.get("lineage_trace") or {}
                    if _ext_lt.get("dep_vars"):
                        _scope_cpp_dep_vars_to_target_setters(
                            _ext_lt, ext_result["setter_sites"], _ext_all_sites,
                        )
                files.setdefault(ext_stem, {})["extended_modifier"] = ext_result
                files[ext_stem]["file_type"] = ext_result.get("file_type", "asm")

                _ensure_full_flow_node(
                    full_flow, ext_stem,
                    blueprint_dir=blueprint_dir,
                    file_type_map=file_type_map,
                    selected_chain_set=selected_chain_set,
                    roles=["extended_modifier"],
                    scope_var=root_var,
                )
                _ext_node = full_flow["nodes"].get(_norm_file_id(ext_stem))
                if _ext_node is not None and ext_result.get("setter_sites"):
                    _ext_node["setter_sites"] = ext_result["setter_sites"]
                # Annotate extended modifier setter sites with block_ref
                if ext_result.get("setter_sites"):
                    _ext_ft = ext_result.get("file_type", "asm")
                    _annotate_setter_sites(
                        ext_result["setter_sites"],
                        stem=ext_stem, file_type=_ext_ft,
                        collector=_block_collector,
                        bp_data=load_json(resolve_asm_blueprint(ext_stem, blueprint_dir)) if _ext_ft == "asm" else None,
                        asm_file=resolve_source_file(ext_stem, "asm", blueprint_dir, asm_dir) if _ext_ft == "asm" else None,
                        blueprint_dir=blueprint_dir, asm_dir=asm_dir,
                        _source_cache=_block_source_cache,
                    )

            logger.info("[BACKWARD] Pass 1a done — %d files, %d setter sites — %.1fs",
                        len(extended_modifier_files),
                        sum(len(r.get("setter_sites", []))
                            for r in extended_modifier_files.values()),
                        time.monotonic() - _run_t0)

            # PERF-OPT-8: final sweep — release any remaining cached blueprints
            # and constant symbols from the Pass 1a scan.
            from backward_traversal.utils.blueprint_utils import (
                clear_blueprint_cache,
                clear_constant_symbols_cache,
            )
            _ci = get_blueprint_cache_info()
            if _ci and getattr(_ci, "currsize", 0) > 0:
                logger.info("[BACKWARD] Pass 1a cache cleanup — %d entries, "
                            "hits=%d, misses=%d",
                            _ci.currsize, _ci.hits, _ci.misses)
            clear_blueprint_cache()
            clear_constant_symbols_cache()
            clear_cfg_cache()
            _clear_cross_file_call_cache()
            import gc
            gc.collect()

    upstream_pairs: List[Tuple[str, str]] = [
        (selected_chain[i], selected_chain[i + 1])
        for i in range(len(selected_chain) - 2, -1, -1)
    ]

    for i in range(len(selected_chain) - 1):
        caller = selected_chain[i]
        callee = selected_chain[i + 1]
        edge_instr = edge_type_map.get((caller, callee), "?")

        # Filter out mutual-callback / dispatch patterns: if caller→callee edge
        # is a callback (caller's source_block is dispatched BY callee), skip it.
        forward_sites = callsite_map.get((caller, callee), [])
        reverse_sites = callsite_map.get((callee, caller), [])
        valid_sites = _filter_callback_call_sites(forward_sites, reverse_sites)

        if forward_sites and not valid_sites:
            # Every call site is a callback — this edge is a false upstream
            warnings.append(
                f"chain_upstream edge {caller}→{callee} suppressed: all call sites "
                f"are mutual callbacks (callee dispatches caller's source_block back)"
            )
            print(
                f"[backward-only] suppressed chain_upstream edge {caller}→{callee}: "
                f"mutual callback pattern detected",
                file=sys.stderr,
            )
            continue

        # Derive line numbers from the surviving call sites; fall back to the
        # pre-computed edge_lines set when call_sites are absent (e.g. ASM edges).
        if valid_sites:
            chain_lines: Set[int] = {
                cs["line"] for cs in valid_sites
                if isinstance(cs.get("line"), int) and cs["line"] > 0
            }
        else:
            chain_lines = edge_lines.get((caller, callee), set())

        for cln in chain_lines or {None}:
            _add_full_flow_edge(
                full_flow,
                source=caller,
                target=callee,
                blueprint_dir=blueprint_dir,
                file_type_map=file_type_map,
                selected_chain_set=selected_chain_set,
                scope_var=root_var,
                instruction=edge_instr,
                discovered_from="chain_upstream",
                source_block=None,
                line_no=cln,
            )

    for caller, callee in upstream_pairs:
        # Skip callers whose only call sites are mutual callbacks (same filter as
        # the chain_upstream full_flow edge above — keeps both outputs consistent).
        _fwd = callsite_map.get((caller, callee), [])
        _rev = callsite_map.get((callee, caller), [])
        if _fwd and not _filter_callback_call_sites(_fwd, _rev):
            print(
                f"[backward-only] skipping upstream trace for {caller}: "
                f"mutual callback pattern with {callee}",
                file=sys.stderr,
            )
            continue

        edge_instr = edge_type_map.get((caller, callee), "?")
        caller_type = resolve_file_type(caller, blueprint_dir, file_type_map)
        callee_type = resolve_file_type(callee, blueprint_dir, file_type_map)

        # GAP-015: pass callee as source_stem so Cases C/D (arg_bind, struct
        # canonical) can translate the variable name across ASM file boundaries.
        caller_var = resolver.resolve(variable, caller, source_stem=callee) or root_var

        print(f"[backward-only] tracing upstream {caller} -{edge_instr}-> {callee} ({caller_type} to {callee_type})", file=sys.stderr)

        if caller_type == "cpp":
            upstream_payload = _build_cpp_upstream_payload(
                caller=caller,
                callee=callee,
                callee_type=callee_type,
                edge_instruction=edge_instr,
                blueprint_dir=blueprint_dir,
                asm_dir=asm_dir,
                variable=caller_var,
            )
        elif caller_type == "asm":
            upstream_payload = _build_asm_upstream_payload(
                variable=caller_var,
                caller=caller,
                callee=callee,
                callee_type=callee_type,
                edge_instruction=edge_instr,
                blueprint_dir=blueprint_dir,
                asm_dir=asm_dir,
                max_depth=max_depth,
                max_subroutine_depth=max_subroutine_depth,
                max_subroutine_nodes=max_subroutine_nodes,
                max_trace_nodes=max_trace_nodes,
            )
        else:
            msg = f"Unknown caller file type for {caller}"
            warnings.append(msg)
            upstream_payload = {
                "role": "upstream",
                "file_type": caller_type,
                "edge_type": edge_instr,
                "warnings": [msg],
            }

        rec = files.setdefault(caller, {})
        rec["file_type"] = caller_type
        rec["upstream"] = upstream_payload
        if upstream_payload.get("warnings"):
            warnings.extend(upstream_payload.get("warnings", []))

    output: Dict[str, Any] = {
        "traversal_mode": "backward_only",
        "root_variable": variable,
        "selected_chain": selected_chain,
        "modifier_file": modifier_stem,
        "files": files,
    }
    if extend_modifier_sites and extended_modifier_files:
        output["extend_modifier_sites"] = True
        output["extended_modifier_files"] = sorted(extended_modifier_files.keys())
    if target_setters:
        output["target_setters"] = target_setters
    if warnings:
        output["warnings"] = sorted(set(str(w) for w in warnings if w))

    # --- PASS 2: dep_var tracing ---

    # Build file_meta for each chain file from Pass 1 outputs  (FIX 3 / T2)
    file_meta: Dict[str, Any] = {}
    for stem in selected_chain:
        rec = files.get(stem, {})
        inner = rec.get("modifier") or rec.get("upstream")
        if inner is None:
            continue
        stem_type = inner.get("file_type") or "asm"  # T2: actual type from Pass 1 payload
        # FIX-new-2: C++ call_sites use key "function" (from find_cpp_to_asm_callers),
        # not "routine" (ASM).  Check both so C++ upstream files get non-empty
        # anchor sets and scope-based dep_var tracing works correctly.
        anchor_blocks = {
            str(s.get("routine") or s.get("function") or "")
            for s in (inner.get("setter_sites") or inner.get("call_sites") or [])
            if s.get("routine") or s.get("function")
        }
        trace_results_p1 = inner.get("setter_site_traces") or inner.get("call_site_traces") or []
        bp = resolve_asm_blueprint(stem, blueprint_dir)
        asm_f = resolve_source_file(stem, "asm", blueprint_dir, asm_dir)
        # Use min() so multiple setter/call sites in the same routine use the
        # earliest line as the cap — prevents stale post-call dep_var matches.
        anchor_before_lines: Dict[str, int] = {}
        for _s in (inner.get("setter_sites") or inner.get("call_sites") or []):
            _key = str(_s.get("routine") or _s.get("function") or "")
            _ln  = int(_s.get("line") or 0)
            if _key and _ln > 0:
                anchor_before_lines[_key] = min(anchor_before_lines.get(_key, _ln), _ln)
        meta_entry: Dict[str, Any] = {
            "anchor_blocks": anchor_blocks,
            "anchor_before_lines": anchor_before_lines,
            "trace_results": trace_results_p1,
            "bp": bp,
            "asm_file": asm_f,
            "file_type": stem_type,
        }
        # T2: for C++ stems store the C++ blueprint path and pre-computed lineage trace
        if stem_type == "cpp":
            cpp_bp_p = resolve_cpp_blueprint(stem, blueprint_dir)
            meta_entry["cpp_bp_path"] = str(cpp_bp_p) if cpp_bp_p else None
            meta_entry["cpp_lineage_trace"] = inner.get("cpp_lineage_trace") or {}
        file_meta[stem] = meta_entry

    # Build file_meta for extended modifier files so their dep_vars enter BFS
    if extend_modifier_sites:
        for ext_stem, ext_result in extended_modifier_files.items():
            if ext_stem in file_meta:
                continue
            ext_type = ext_result.get("file_type", "asm")
            ext_anchor_blocks = {
                str(s.get("routine") or s.get("function") or "")
                for s in ext_result.get("setter_sites", [])
                if s.get("routine") or s.get("function")
            }
            ext_trace_results = ext_result.get("setter_site_traces", [])
            ext_bp = resolve_asm_blueprint(ext_stem, blueprint_dir)
            ext_asm_f = resolve_source_file(ext_stem, "asm", blueprint_dir, asm_dir)
            ext_anchor_before: Dict[str, int] = {}
            for _s in ext_result.get("setter_sites", []):
                _key = str(_s.get("routine") or _s.get("function") or "")
                _ln = int(_s.get("line") or 0)
                if _key and _ln > 0:
                    ext_anchor_before[_key] = min(ext_anchor_before.get(_key, _ln), _ln)
            ext_meta: Dict[str, Any] = {
                "anchor_blocks": ext_anchor_blocks,
                "anchor_before_lines": ext_anchor_before,
                "trace_results": ext_trace_results,
                "bp": ext_bp,
                "asm_file": ext_asm_f,
                "file_type": ext_type,
            }
            if ext_type == "cpp":
                ext_cpp_bp = resolve_cpp_blueprint(ext_stem, blueprint_dir)
                ext_meta["cpp_bp_path"] = str(ext_cpp_bp) if ext_cpp_bp else None
                ext_meta["cpp_lineage_trace"] = ext_result.get("lineage_trace", {})
            file_meta[ext_stem] = ext_meta

    # Step 2-A: build dep_var_collection_map from Pass 1 traces  (T2 / GAP B)
    #
    # Issue-5 suppression: if every modifier setter_site is constant_source=True the
    # root variable is always assigned a compile-time constant string (e.g. MVC VAR,WB).
    # The upstream call-graph nodes carry dependent_variables that describe what the
    # CALLER reads on its way to the setter — these are NOT value dependencies of the
    # root variable.  Seeding them into the BFS would generate dozens of spurious
    # dep_var files.  When all_constant_source is True we skip those insertions so the
    # dep_var/ directory is empty, accurately reflecting "no runtime inputs".
    _mod_payload = (files.get(modifier_stem) or {}).get("modifier") or {}
    _mod_setter_sites = _mod_payload.get("setter_sites") or []
    _all_constant_source = bool(_mod_setter_sites) and all(
        ss.get("constant_source") for ss in _mod_setter_sites
    )
    # True when the backward trace found at least one setter site in the modifier
    # file — the root assignment is known regardless of whether downstream or
    # dep_var expansion later hit caps.
    root_setter_found: bool = bool(_mod_setter_sites)

    dep_var_collection_map: Dict[str, Dict[str, Set[str]]] = {}
    for stem, meta in file_meta.items():
        if meta.get("file_type") == "cpp":
            # T2: C++ dep_vars come from cpp_lineage_trace.dep_vars, not setter_site_traces
            cpp_trace = meta.get("cpp_lineage_trace") or {}
            stem_idx = selected_chain_index.get(_norm_file_id(stem), -1)
            for dv in (cpp_trace.get("dep_vars") or []):
                dv_str = str(dv or "").strip()
                if not dv_str:
                    continue
                if dv_str.startswith("bridge_back_to_asm:"):
                    # GAP B: route bridge register to the nearest ASM stem in the chain.
                    # Search forward first (toward downstream); if not found, search
                    # backward (toward modifier/upstream) — covers C++ files that call
                    # ASM upstream rather than downstream.
                    reg = dv_str.split(":", 1)[-1].upper()  # e.g. "R7"
                    next_asm: Optional[str] = None
                    for candidate in (selected_chain[stem_idx + 1:] if stem_idx >= 0 else []):
                        if resolve_file_type(candidate, blueprint_dir, file_type_map) == "asm":
                            next_asm = candidate
                            break
                    if not next_asm:
                        # GAP B fix: also look backward toward the modifier
                        for candidate in reversed(selected_chain[:stem_idx] if stem_idx > 0 else []):
                            if resolve_file_type(candidate, blueprint_dir, file_type_map) == "asm":
                                next_asm = candidate
                                break
                    if next_asm:
                        dep_var_collection_map.setdefault(reg, {}).setdefault(next_asm, set())
                    else:
                        # C2.3 fix: emit warning when no adjacent ASM stem can be found.
                        import logging as _logging
                        _logging.getLogger(__name__).warning(
                            "GAP B: bridge_back_to_asm:%s from C++ stem '%s' — no adjacent "
                            "ASM stem found in chain %s; register dep_var dropped.",
                            reg, stem, selected_chain,
                        )
                else:
                    # Regular C++ dep_var; empty set = T3 searches the full GVL
                    # FIX-new-11: drop infrastructure noise tokens before insertion
                    dv_up = dv_str.upper()
                    if not _is_noise_dep_var(dv_up):
                        dep_var_collection_map.setdefault(dv_up, {}).setdefault(stem, set())
        else:
            # Issue-5: when ALL modifier setters are compile-time constants the root
            # variable's value is fixed at link time — upstream call-graph dep_vars are
            # not genuine value inputs.  Skip seeding them into the BFS map.
            if not _all_constant_source:
                # Partial constant-source filtering (Gap 1): in the MIXED case some
                # setter traces short-circuit inside trace_asm_call_site_backward to
                # {"constant_source": True, "call_graph": {"nodes": []}, ...}.
                # Exclude them explicitly so the intent is clear and we do not rely on
                # the empty-nodes side-effect to keep their dep_vars out of the map.
                _non_const_traces = [
                    tr for tr in (meta.get("trace_results") or [])
                    if not tr.get("constant_source")
                ]
                per_file = _aggregate_dep_var_collection_blocks(_non_const_traces)
                for dv, blocks in per_file.items():
                    # Gap 2: apply noise filter consistent with C++ path (line 3524),
                    # _build_child_dep_var_map (line 1574), and BFS guard (line 4340).
                    # Phase 1 previously had no _is_noise_dep_var gate, allowing tokens
                    # like R5, LEVEL:D4, RETURN@fn to enter dep_var_collection_map.
                    if not _is_noise_dep_var(dv):
                        dep_var_collection_map.setdefault(dv, {})[stem] = blocks
                # GAP-ZEROS-SCOPE: the upstream Pass 1 trace records SETTER_STEPs in
                # relevant_code_lines when it visits a block where the root variable is
                # set (e.g. MVC WK_RRC,ZEROS at DA7_0050 in da76).  Those setter blocks
                # are NOT captured by _aggregate_dep_var_collection_blocks (which only
                # reads dependent_variables), so they never enter dep_var_collection_map
                # and the dep_var BFS scope misses them entirely.  Seeding them here
                # ensures the collection_based scope expands to cover initialisation
                # setters that sit outside the BAS-subroutine-reachable comparison sites.
                _setter_blocks = _aggregate_setter_step_blocks(_non_const_traces)
                if _setter_blocks:
                    dep_var_collection_map.setdefault(root_var, {}).setdefault(stem, set()).update(_setter_blocks)

                # Bug-4 fix: also promote non-constant setter_expression values.
                # CFG dep_var analysis captures variables referenced in the code
                # REGION around a setter but can miss the source operand of the
                # setter instruction itself (e.g. `MVC BR31EFLD,WK_RFD` captures
                # WK_RRC from a surrounding CLC but not WK_RFD from the MVC line).
                # Promote every non-constant setter_expression from this stem's
                # setter_sites as an additional dep_var seed.
                _stem_rec = files.get(stem, {})
                _stem_sites = (
                    (_stem_rec.get("modifier") or {}).get("setter_sites")
                    or (_stem_rec.get("upstream") or {}).get("setter_sites")
                    or (_stem_rec.get("extended_modifier") or {}).get("setter_sites")
                    or []
                )
                for _ss in _stem_sites:
                    if _ss.get("constant_source"):
                        continue
                    _expr = str(_ss.get("setter_expression") or "").strip()
                    # Skip blanks and inline literals (=C'..', =X'..').
                    if not _expr or _expr.startswith("="):
                        continue
                    _dv_up = normalize_token(_expr).upper()
                    if _dv_up and not _is_noise_dep_var(_dv_up):
                        dep_var_collection_map.setdefault(_dv_up, {}).setdefault(stem, set())

    logger.info("[BACKWARD] Pass 1 done — modifier + %d upstream hops — %.1fs",
                len(upstream_pairs), time.monotonic() - _run_t0)

    # Shared exclusion set: never re-enter chain files during downstream BFS.
    exclude_stems: Set[str] = {s.lower() for s in selected_chain}
    if extend_modifier_sites:
        exclude_stems.update(_norm_file_id(s) for s in extended_modifier_files)

    # --- Pass 1b: root_var downstream search ---
    # Mirrors dep_var downstream BFS for the root variable so that files
    # reachable via cross-file calls from within the root var's scope are
    # also searched for root var setters.
    root_var_downstream_traces: Dict[str, Any] = {}
    rv_visited_downstream: Set[str] = set()
    rv_downstream_queue: Deque[Tuple[str, int, str, str, List[str]]] = deque()

    for stem, meta in file_meta.items():
        if meta.get("file_type") == "cpp":
            # FIX-new-P1-5: C++ files have no block-based CFG, but they can call out
            # to other files via call_graph.edges.  Scan those edges, scoped to the
            # anchor functions (if any), and queue callee stems for downstream search.
            cpp_bp_path = resolve_cpp_blueprint(stem, blueprint_dir)
            if not cpp_bp_path:
                continue
            cpp_bp_data = load_json(cpp_bp_path, keys={"call_graph"})
            anchor_fns: Set[str] = meta.get("anchor_blocks") or set()
            # NEW-BUG-1: anchor_fns for an ASM→C++ callee is populated from
            # find_asm_to_cpp_callers() "routine" keys — which are ASM block names,
            # not C++ function names.  Applying the filter would discard every C++
            # edge and produce an empty cpp_cross list.  Only apply the filter when
            # anchor_fns actually overlaps the C++ function names in this blueprint.
            _cpp_fn_names: Set[str] = {
                str(e.get("source") or "")
                for e in (cpp_bp_data.get("call_graph", {}) or {}).get("edges", [])
                if e.get("source")
            }
            _anchor_fns_are_cpp = bool(anchor_fns and (anchor_fns & _cpp_fn_names))
            cpp_cross: List[Dict[str, Any]] = []
            seen_cpp_edges: Set[Tuple[str, Any]] = set()
            for edge in (cpp_bp_data.get("call_graph", {}) or {}).get("edges", []):
                source_fn = str(edge.get("source") or "")
                if _anchor_fns_are_cpp and source_fn not in anchor_fns:
                    continue
                tgt_stem = _norm_file_id(str(edge.get("target") or ""))
                if not tgt_stem:
                    continue
                key_e = (source_fn, edge.get("line"))
                if key_e in seen_cpp_edges:
                    continue
                seen_cpp_edges.add(key_e)
                from api.utils.indirection import merge_indirection
                _cf_rec: Dict[str, Any] = {
                    "source_block": source_fn,
                    "target_file": tgt_stem,
                    "instruction": str(edge.get("instruction") or ""),
                    "call_type": str(edge.get("call_type") or ""),
                    "line": edge.get("line"),
                }
                merge_indirection(_cf_rec, edge)
                cpp_cross.append(_cf_rec)
            for cf in cpp_cross:
                _add_full_flow_edge(
                    full_flow,
                    source=stem,
                    target=str(cf.get("target_file") or ""),
                    blueprint_dir=blueprint_dir,
                    file_type_map=file_type_map,
                    selected_chain_set=selected_chain_set,
                    scope_var=root_var,
                    instruction=str(cf.get("instruction") or ""),
                    discovered_from="root_var_cpp_scope",
                    source_block=str(cf.get("source_block") or ""),
                    line_no=cf.get("line"),
                )
            _stem_path = _path_to_chain_file(stem, selected_chain)
            rv_downstream_queue.extend(
                (cf["target_file"], 0, stem, root_var, _stem_path)
                for cf in cpp_cross
                if cf["target_file"] and cf["target_file"] not in exclude_stems
            )
            continue
        # GAP-036: asm_file may be None when source is absent; blueprint is sufficient.
        if not meta["bp"]:
            continue
        bp_data = load_json(meta["bp"], keys={"blocks", "call_graph"})
        _rv_bp_path = str(meta["bp"]) if meta.get("bp") else None
        if meta["anchor_blocks"]:
            rv_scope: Set[str] = set()
            from backward_traversal.utils.reach_facts import lookup_reach_facts as _lrf
            for b in meta["anchor_blocks"]:
                # Precomputed block_reach_facts when available (Lineage Facts
                # Index Phase 2); identical fallback to the live CFG walk.
                _rv_reach = _lrf(_rv_bp_path, str(b), 0) if _rv_bp_path else None
                if _rv_reach is None:
                    _rv_reach = backward_reachable_blocks(
                        bp_data, b,
                        expand_subroutine_callers=True,
                        bp_path=_rv_bp_path,
                    )
                rv_scope.update(_rv_reach)
        else:
            rv_scope = {str(b["id"]) for b in bp_data.get("blocks", [])}

        rv_cross = _collect_scoped_cross_file_calls(
            bp_data, rv_scope, meta["asm_file"], asm_dir, bp_path=meta["bp"]
        )
        rv_returning_cross = [cf for cf in rv_cross if _is_followable_cross_file_call(cf)]
        for cf in rv_returning_cross:
            _conds = _extract_block_conditions_before_line(
                bp_data, str(cf.get("source_block") or ""), int(cf.get("line") or 0),
                bp_path=_rv_bp_path,
            )
            _add_full_flow_edge(
                full_flow,
                source=stem,
                target=str(cf.get("target_file") or ""),
                blueprint_dir=blueprint_dir,
                file_type_map=file_type_map,
                selected_chain_set=selected_chain_set,
                scope_var=root_var,
                instruction=str(cf.get("instruction") or ""),
                discovered_from="root_var_scope",
                source_block=str(cf.get("source_block") or ""),
                line_no=cf.get("line"),
                conditions=_conds,
            )
        _stem_path = _path_to_chain_file(stem, selected_chain)
        rv_downstream_queue.extend(
            (cf["target_file"], 0, stem, root_var, _stem_path)
            for cf in rv_returning_cross
            if cf["target_file"] and cf["target_file"] not in exclude_stems
        )

    # PERF-OPT-6: per-file downstream work executed in parallel.
    # Helper returns everything the sequential merge needs.
    def _process_one_rv_ds(
        ds_stem: str, ds_depth: int, rv_source_stem: str,
        rv_source_var: str, _path_to_source: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Process one downstream file (ASM or C++).

        Returns a dict with keys: ds_stem, trace, is_cpp, ds_root_var,
        edges (list of edge-info dicts), callees (list of callee stems).
        """
        bp_ds = resolve_asm_blueprint(ds_stem, blueprint_dir)
        asm_ds = resolve_source_file(ds_stem, "asm", blueprint_dir, asm_dir)

        if not bp_ds or not asm_ds:
            # C++ path
            cpp_bp_ds = resolve_cpp_blueprint(ds_stem, blueprint_dir)
            if not cpp_bp_ds:
                return None
            cpp_bp_data_ds = load_json(cpp_bp_ds, keys={"call_graph"})
            from backward_traversal.utils.lazy_gvl import make_lazy_gvl as _mkgvl
            gvl_ds = _mkgvl(cpp_bp_data_ds, blueprint_path=cpp_bp_ds)
            ds_root_var = (resolver.resolve(rv_source_var, ds_stem, source_stem=rv_source_stem) or rv_source_var).upper()
            trace = _build_dep_var_cpp_result(
                ds_root_var, gvl_ds,
                resolver=resolver, stem=ds_stem, source_stem=rv_source_stem,
                blueprint_dir=blueprint_dir, asm_dir=asm_dir,
            )
            # Extract edges and callees from call_graph.
            edges: List[Dict[str, Any]] = []
            callees: List[str] = []
            seen_cpp_ds_edges: Set[Tuple[str, Any]] = set()
            for edge in (cpp_bp_data_ds.get("call_graph", {}) or {}).get("edges", []):
                tgt = _norm_file_id(str(edge.get("target") or ""))
                if not tgt:
                    continue
                key_c: Tuple[str, Any] = (str(edge.get("source") or ""), edge.get("line"))
                if key_c in seen_cpp_ds_edges:
                    continue
                seen_cpp_ds_edges.add(key_c)
                edges.append({
                    "source": ds_stem, "target": tgt,
                    "scope_var": root_var,
                    "instruction": str(edge.get("instruction") or ""),
                    "discovered_from": "root_var_cpp_downstream_scope",
                    "source_block": str(edge.get("source") or ""),
                    "line_no": edge.get("line"),
                    "conditions": [],
                })
                callees.append(tgt)
            return {
                "ds_stem": ds_stem, "trace": trace, "is_cpp": True,
                "ds_root_var": ds_root_var, "ds_depth": ds_depth,
                "edges": edges, "callees": callees,
                "_path_to_source": _path_to_source,
            }

        # ASM path
        bp_data_ds = load_json(bp_ds, keys={"blocks", "call_graph"})
        all_blocks_ds: Set[str] = {str(b["id"]) for b in bp_data_ds.get("blocks", [])}
        ds_rv_cross = _collect_scoped_cross_file_calls(
            bp_data_ds, all_blocks_ds, asm_ds, asm_dir, bp_path=bp_ds
        )
        ds_rv_returning_cross = [cf for cf in ds_rv_cross if _is_followable_cross_file_call(cf)]
        edges = []
        for cf in ds_rv_returning_cross:
            _conds = _extract_block_conditions_before_line(
                bp_data_ds, str(cf.get("source_block") or ""), int(cf.get("line") or 0),
                bp_path=str(bp_ds),
            )
            edges.append({
                "source": ds_stem,
                "target": str(cf.get("target_file") or ""),
                "scope_var": root_var,
                "instruction": str(cf.get("instruction") or ""),
                "discovered_from": "root_var_downstream_scope",
                "source_block": str(cf.get("source_block") or ""),
                "line_no": cf.get("line"),
                "conditions": _conds,
            })
        ds_root_var = (resolver.resolve(rv_source_var, ds_stem, source_stem=rv_source_stem) or rv_source_var).upper()
        trace = _build_dep_var_file_result(
            ds_root_var, bp_data_ds, bp_ds, asm_ds, all_blocks_ds, "full_file",
            asm_dir, max_depth, max_subroutine_depth, max_subroutine_nodes, max_trace_nodes,
        )
        callees = sorted({
            cf["target_file"]
            for cf in ds_rv_returning_cross
            if cf["target_file"]
        })
        return {
            "ds_stem": ds_stem, "trace": trace, "is_cpp": False,
            "ds_root_var": ds_root_var, "ds_depth": ds_depth,
            "edges": edges, "callees": callees,
            "_path_to_source": _path_to_source,
        }

    rv_ds_capped = False
    while rv_downstream_queue and not rv_ds_capped:
        # PERF-OPT-8: abort BFS early if memory pressure detected.
        if _MAX_RSS_MB > 0:
            _rss = _get_rss_mb()
            if _rss > _MAX_RSS_MB:
                output.setdefault("warnings", []).append(
                    f"root_var downstream: aborted — RSS {_rss:.0f} MB "
                    f"exceeds ASM_MAX_RSS_MB={_MAX_RSS_MB} after "
                    f"{len(rv_visited_downstream)} files"
                )
                _trunc_detail["rv_downstream_memory"] = True
                run_truncated = True
                rv_ds_capped = True
                break

        # Drain current depth into a batch, filtering visited/excluded.
        batch_rv: List[Tuple[str, int, str, str, List[str]]] = []
        while rv_downstream_queue:
            ds_stem, ds_depth, rv_source_stem, rv_source_var, _path_to_source = rv_downstream_queue.popleft()
            if ds_stem in rv_visited_downstream or ds_stem in exclude_stems:
                continue
            if len(rv_visited_downstream) + len(batch_rv) >= max_downstream_files:
                output.setdefault("warnings", []).append(
                    f"root_var downstream: max_downstream_files={max_downstream_files} cap reached"
                )
                _trunc_detail["rv_downstream_files"] = True
                run_truncated = True
                rv_downstream_queue.clear()
                rv_ds_capped = True
                break
            if ds_depth >= max_downstream_depth:
                rv_visited_downstream.add(ds_stem)
                output.setdefault("warnings", []).append(
                    f"root_var downstream: depth cap for {ds_stem}"
                )
                _trunc_detail["rv_downstream_depth"] = True
                run_truncated = True
                continue
            rv_visited_downstream.add(ds_stem)
            batch_rv.append((ds_stem, ds_depth, rv_source_stem, rv_source_var, _path_to_source))

        if not batch_rv:
            break

        # Process batch — parallel when >1 file, serial when 1.
        results_rv: List[Optional[Dict[str, Any]]] = []
        if len(batch_rv) == 1:
            ds_stem, ds_depth, rv_source_stem, rv_source_var, _path_to_source = batch_rv[0]
            logger.info("[BACKWARD] Pass 1b — tracing %s (depth=%d)", ds_stem, ds_depth)
            results_rv.append(_process_one_rv_ds(ds_stem, ds_depth, rv_source_stem, rv_source_var, _path_to_source))
        else:
            with ThreadPoolExecutor(max_workers=min(_PARALLEL_WORKERS, len(batch_rv))) as pool:
                futures_rv = {
                    pool.submit(_process_one_rv_ds, ds_stem, ds_depth, rv_src_stem, rv_src_var, ptosrc): ds_stem
                    for ds_stem, ds_depth, rv_src_stem, rv_src_var, ptosrc in batch_rv
                }
                for fut in as_completed(futures_rv):
                    try:
                        results_rv.append(fut.result())
                    except Exception:
                        logger.debug("[BACKWARD] Pass 1b — error tracing %s", futures_rv[fut], exc_info=True)
                        results_rv.append(None)

        # Sequential merge: update full_flow, traces, queue.
        _ds_rv_exclude_snapshot = exclude_stems | rv_visited_downstream
        for res in results_rv:
            if res is None:
                continue
            ds_stem = res["ds_stem"]
            ds_depth = res["ds_depth"]
            ds_root_var = res["ds_root_var"]
            _label = "cpp" if res["is_cpp"] else "asm"
            print(
                f"[backward-only][root_var downstream depth={ds_depth}][{_label}]"
                f" tracing {ds_root_var} in {ds_stem}",
                file=sys.stderr,
            )
            root_var_downstream_traces[ds_stem] = res["trace"]
            # Annotate root_var downstream setter sites with block_ref
            _rv_ds_sites = res["trace"].get("setter_sites", [])
            if _rv_ds_sites:
                _rv_ds_ft = "cpp" if res["is_cpp"] else "asm"
                _annotate_setter_sites(
                    _rv_ds_sites,
                    stem=ds_stem, file_type=_rv_ds_ft,
                    collector=_block_collector,
                    bp_data=load_json(resolve_asm_blueprint(ds_stem, blueprint_dir)) if not res["is_cpp"] else None,
                    asm_file=resolve_source_file(ds_stem, "asm", blueprint_dir, asm_dir) if not res["is_cpp"] else None,
                    blueprint_dir=blueprint_dir, asm_dir=asm_dir,
                    _source_cache=_block_source_cache,
                )
            for ei in res["edges"]:
                _add_full_flow_edge(
                    full_flow,
                    source=ei["source"],
                    target=ei["target"],
                    blueprint_dir=blueprint_dir,
                    file_type_map=file_type_map,
                    selected_chain_set=selected_chain_set,
                    scope_var=ei["scope_var"],
                    instruction=ei["instruction"],
                    discovered_from=ei["discovered_from"],
                    source_block=ei["source_block"],
                    line_no=ei["line_no"],
                    conditions=ei.get("conditions"),
                )
            for callee in res["callees"]:
                if callee not in _ds_rv_exclude_snapshot:
                    rv_downstream_queue.append(
                        (callee, ds_depth + 1, ds_stem, ds_root_var, res["_path_to_source"] + [ds_stem])
                    )

        # PERF-OPT-8: evict blueprints loaded for this batch so peak RAM
        # stays proportional to batch size, not total visited files.
        from backward_traversal.utils.blueprint_utils import (
            clear_blueprint_cache,
            clear_constant_symbols_cache,
        )
        clear_blueprint_cache()
        clear_constant_symbols_cache()
        clear_cfg_cache()
        _clear_cross_file_call_cache()
        import gc
        gc.collect()

    logger.info("[BACKWARD] Pass 1b done — %d root-var downstream files — %.1fs",
                len(rv_visited_downstream), time.monotonic() - _run_t0)

    # GAP-E: extract dep_vars from root_var downstream traces and seed them
    # into dep_var_collection_map so they enter the Pass 2 BFS.  A downstream
    # file may introduce a dep_var not visible in any chain file (e.g. a unique
    # setter whose source operand references a variable not seen elsewhere).
    if not _all_constant_source:
        for _rv_ds_stem, _rv_ds_trace in root_var_downstream_traces.items():
            # ASM path: dep_vars from setter_site_traces → dependent_variables
            _rv_ds_traces_list = _rv_ds_trace.get("setter_site_traces") or []
            _rv_ds_per_file = _aggregate_dep_var_collection_blocks(_rv_ds_traces_list)
            for _rv_dv, _rv_blocks in _rv_ds_per_file.items():
                if not _is_noise_dep_var(_rv_dv):
                    dep_var_collection_map.setdefault(_rv_dv, {}).setdefault(
                        _rv_ds_stem, set()
                    ).update(_rv_blocks)
            # C++ path: dep_vars from lineage_trace.dep_vars
            if _rv_ds_trace.get("file_type") == "cpp":
                _rv_cpp_trace = _rv_ds_trace.get("lineage_trace") or {}
                for _rv_cpp_dv in (_rv_cpp_trace.get("dep_vars") or []):
                    _rv_dv_str = str(_rv_cpp_dv or "").strip()
                    if _rv_dv_str and not _rv_dv_str.startswith("bridge_back_to_asm:"):
                        _rv_dv_up = _rv_dv_str.upper()
                        if not _is_noise_dep_var(_rv_dv_up):
                            dep_var_collection_map.setdefault(_rv_dv_up, {}).setdefault(
                                _rv_ds_stem, set()
                            )

    # Issue-15: When root_var_downstream_traces is empty and no truncation caps
    # were hit, the variable simply has no downstream files to trace.  Flag this
    # explicitly so consumers can distinguish "empty by design" from "truncated".
    if (
        not root_var_downstream_traces
        and not _trunc_detail["rv_downstream_files"]
        and not _trunc_detail["rv_downstream_depth"]
        and not _trunc_detail.get("rv_downstream_memory")
    ):
        _trunc_detail["rv_downstream_not_applicable"] = True

    # Step 2-B: dep_var BFS
    # GAP-ROOT-UPSTREAM: do NOT pre-load root_var into visited_dep_vars.
    # Previously the root variable was excluded from the dep_var BFS entirely,
    # which caused upstream-file setter sites to be missed when the variable has
    # no CLC/CLI/TM comparisons in the modifier file (e.g. WK_RFD: 2 da76
    # setters missed; WB1PTMST: xh80:329 missed).  By letting the BFS process
    # root_var like any other dep_var (with full_file scope — see below), all
    # setter sites in all chain files are discovered.  The cap check (- 1) is
    # unchanged: root_var counts as one entry in visited_dep_vars once processed,
    # so max_dep_vars non-root dep_vars are still allowed.
    visited_dep_vars: Set[str] = set()
    dv_queue: Deque[Tuple[str, int]] = deque((dv, 0) for dv in sorted(dep_var_collection_map))
    # FIX 20: tracks every name ever placed on dv_queue — prevents a dep_var
    # cycle (A→dep_var B→dep_var A) from re-queuing entries at increasing depths.
    # max_dep_vars count-cap alone cannot prevent duplicate queue entries.
    already_queued: Set[str] = {dv for dv, _ in dv_queue}
    # Parent tracking: maps each dep_var to the list of parent dep_vars that
    # discovered it.  Used by the ASM schema transformer to populate the
    # ``dependencies`` field in the target output schema.
    dep_var_parent_map: Dict[str, List[str]] = {}
    # _dep_var_writer flushes each trace to disk immediately in partitioned mode.

    # T9: build graph-distance filtered candidate set once for all dep_vars.
    # Limits T8 candidate scan to ≤2-hop neighbours of selected_chain — typically
    # 5–30 files vs O(N_cpp_files) without filtering.  Built only when T8 is
    # active (t8_max_scan > 0) to avoid unnecessary I/O.
    t8_neighbor_set: Optional[Set[str]] = None
    if t8_max_scan > 0:
        _cg_for_t9 = blueprint_dir / "file_call_graph.json"
        if not _cg_for_t9.exists():
            _cg_for_t9 = graph_file   # fallback to the passed-in graph_file
        t8_neighbor_set = build_chain_neighbor_set(selected_chain, _cg_for_t9)
        print(
            f"[backward-only][T9] neighbor_set built: {len(t8_neighbor_set)} stems "
            f"(max_scan={t8_max_scan})",
            file=sys.stderr,
        )

    # Tracks which dep_vars already triggered a T8 extended scan (at most once each).
    _t8_searched: Set[str] = set()
    extended_dep_var_results: Dict[str, Dict[str, Any]] = {}

    # Per-run cache for find_cpp_seed_nodes results.  Keyed on (id(gvl), var_lower)
    # so the same GVL instance + variable pair is never scanned twice.  Scoped to
    # this dep_var BFS loop — no cross-task leaks.
    _seed_cache: Dict = {}
    # Per-GVL cache for (reverse_idx, node_lookup).  Avoids O(E)+O(N) rebuilds
    # when multiple dep_vars share the same C++ stem/GVL instance.
    # Key: id(gvl) — safe because make_lazy_gvl caches instances by path.
    _gvl_idx_cache: Dict[int, Tuple[Any, Any]] = {}
    # Per-stem cache for build_func_source_index results.  Avoids re-reading
    # C++ source files and re-parsing line_metadata for every dep_var in the
    # same stem.  Key: file stem.
    _source_idx_cache: Dict[str, Dict[str, str]] = {}
    # CONVERGENCE: the per-stem _bp_cache/_gvl_cache wrapper dicts were removed.
    # load_json is served by the path-keyed blueprint frame cache and
    # make_lazy_gvl by lazy_gvl._LAZY_GVL_CACHE — both process-cached, so the
    # wrappers only duplicated state.  GVL object identity per stem (the
    # _seed_cache id(gvl) key) is guaranteed by _LAZY_GVL_CACHE.

    # ── Shared DB reverse index ──
    # When all chain stems share the same index.db (same job), one
    # _DbReverseIndex + _DbNodeLookup pair can serve ALL dep_vars across
    # ALL stems.  This keeps the LRU caches warm (90%+ hit rate) instead
    # of creating cold instances per GVL id().
    _shared_db_rev: Any = None
    _shared_db_nl: Any = None
    _shared_db_key: Optional[Tuple[str, str]] = None  # (db_path, job_id)
    try:
        # Probe the first C++ chain stem to find the DB path.
        for _probe_stem in selected_chain:
            _probe_meta = file_meta.get(_probe_stem)
            if not _probe_meta or _probe_meta.get("file_type") != "cpp":
                continue
            _probe_bp_str = _probe_meta.get("cpp_bp_path")
            if not _probe_bp_str:
                continue
            _probe_bp_p = Path(_probe_bp_str)
            if not _probe_bp_p.exists():
                continue
            _probe_bp = load_json(_probe_bp_p)
            from backward_traversal.utils.lazy_gvl import make_lazy_gvl as _mkgvl_probe
            _probe_gvl = _mkgvl_probe(_probe_bp, blueprint_path=_probe_bp_p)
            _probe_db = getattr(_probe_gvl, "_db_path", None)
            _probe_jid = getattr(_probe_gvl, "_job_id", None)
            if _probe_db and _probe_jid:
                from backward_traversal.tracing.cpp_backward_tracer import (
                    _build_reverse_index as _bri_probe,
                    _build_node_lookup as _bnl_probe,
                )
                _shared_db_rev = _bri_probe(_probe_gvl)
                _shared_db_nl = _bnl_probe(_probe_gvl)
                _shared_db_key = (str(_probe_db), str(_probe_jid))
                logger.info(
                    "[BACKWARD] shared DB reverse index created (db=%s, job=%s)",
                    _probe_db, _probe_jid,
                )
            break
    except Exception:
        pass  # no shared index — fall back to per-GVL caching

    # ── Pre-filter: dep_var → relevant C++ stems (Opt A — computed early) ──
    # Built BEFORE the path planner so the cone can be narrowed to only
    # stems that have seeds for the initial dep_vars.  Typically ~75% fewer
    # stems → edge count drops below 200 K → bulk prefetch succeeds even on
    # 22 K+ file codebases.
    _dv_relevant_stems: Dict[str, Set[str]] = {}
    if _shared_db_key is not None:
        try:
            from api.index_db.readers import get_cpp_seed_candidate_files_bulk as _gcscf_bulk
            _pf_vars = list({dv for dv, _ in dv_queue})
            _pf_bulk = _gcscf_bulk(_shared_db_key[1], _pf_vars)
            if _pf_bulk:
                for _pf_key, _pf_stems in _pf_bulk.items():
                    _dv_relevant_stems[_pf_key] = set(s.lower() for s in _pf_stems)
        except Exception:
            pass
        if _dv_relevant_stems:
            logger.info(
                "[BACKWARD] pre-filter: %d initial dep_vars mapped to relevant stems",
                len(_dv_relevant_stems),
            )

    # ── Upfront path planner (Change 6 + Opt A) ──
    # Bulk-prefetch GVL edges/nodes for the relevant-stems cone in two SQL
    # queries, replacing thousands of per-target DB round-trips with in-memory
    # lookups.  Falls back to _shared_db_rev/_shared_db_nl for targets outside
    # the cone.  Opt A: use the union of _dv_relevant_stems (much smaller than
    # all chain stems) so edge count stays below the prefetch threshold.
    if _shared_db_key is not None:
        try:
            from backward_traversal.runner.backward_path_planner import BackwardPathPlan
            _relevant_cone: Set[str] = set()
            for _rs in _dv_relevant_stems.values():
                _relevant_cone.update(_rs)
            _chain_lower = {s.lower() for s in selected_chain}
            _cone = (_relevant_cone & _chain_lower) if _relevant_cone else _chain_lower
            _plan = BackwardPathPlan(
                job_id=_shared_db_key[1],
                db_path=_shared_db_key[0],
                cone_stems=_cone,
                fallback_rev=_shared_db_rev,
                fallback_nl=_shared_db_nl,
            )
            if _plan.edge_count > 0:
                _shared_db_rev = _plan.make_reverse_index()
                _shared_db_nl = _plan.make_node_lookup()
                logger.info(
                    "[BACKWARD] path planner active: %d edges, %d nodes, %d/%d cone stems "
                    "(relevant/total chain)",
                    _plan.edge_count, _plan.node_count, _plan.cone_stem_count,
                    len(selected_chain),
                )
            elif getattr(_plan, "_skipped_prefetch", False):
                logger.info(
                    "[BACKWARD] path planner: bulk prefetch skipped (cone too large), "
                    "using lazy DB-backed indices for %d stems",
                    _plan.cone_stem_count,
                )
        except Exception as _plan_exc:
            logger.debug("[BACKWARD] path planner unavailable: %s", _plan_exc)

    # ── Pre-compute T8 candidate map from DB ──
    # When cpp_seed_file_keys is populated, batch-query candidate files for
    # ALL known dep_vars upfront.  Replaces per-dep_var blueprint scanning
    # (up to 50 loads each) with indexed DB lookups.
    _t8_precomputed: Dict[str, List[str]] = {}
    if t8_max_scan > 0 and _shared_db_key is not None:
        try:
            from api.index_db.readers import get_cpp_seed_candidate_files_bulk as _gcscf_bulk_t8
            _t8_all_dvs = list({dv for dv, _ in dv_queue})
            _t8_bulk = _gcscf_bulk_t8(_shared_db_key[1], _t8_all_dvs)
            if _t8_bulk:
                for _t8_key, _t8_stems in _t8_bulk.items():
                    if t8_neighbor_set is not None:
                        _t8_stems = [s for s in _t8_stems if s in t8_neighbor_set]
                    if _t8_stems:
                        _t8_precomputed[_t8_key] = _t8_stems
                if _t8_precomputed:
                    logger.info(
                        "[BACKWARD] T8 pre-computed (bulk): %d dep_vars have DB candidate files",
                        len(_t8_precomputed),
                    )
        except Exception:
            pass  # fall back to per-dep_var scanning

    # ── Fix 3: Bulk seed pre-population (DISABLED) ──
    # Bulk SQL prefetch was slower than individual queries in _build_dep_var_cpp_result
    # because the SQLite query planning for IN-clause + _has_rows overhead per stem
    # exceeded the per-dep_var query cost.  Fixes 2 (bp/gvl cache) and 4 (seed cap)
    # make the per-dep_var path fast enough (~8s for the entire BFS loop).
    # The empty dict and False flag keep the call-site signatures valid.
    _bulk_seed_map: Dict[Tuple[str, str], List[str]] = {}  # (var_lower, stem) -> seed_ids
    _bulk_seed_unfiltered: bool = False

    # ── Incremental pre-filter helper (Opt A) ──
    # For dep_vars discovered DURING BFS (not in the initial queue).
    # Initial dep_vars were already populated above (before the path planner).
    def _prefilter_dep_var(dep_var: str) -> None:
        """Incrementally add a dep_var to the pre-filter map."""
        dv_lower = dep_var.lower()
        if dv_lower in _dv_relevant_stems:
            return  # already mapped
        if _shared_db_key is None:
            return
        try:
            from api.index_db.readers import get_cpp_seed_candidate_files as _gcscf
            stems = _gcscf(_shared_db_key[1], dep_var)
            if stems is not None:
                _dv_relevant_stems[dv_lower] = set(s.lower() for s in stems)
        except Exception:
            pass

    # PERF: backward_reachable_blocks is a pure function of (file, block,
    # before_line) for a fixed blueprint, but the dep-var BFS recomputed the
    # CFG walk for the same (stem, block) pair once per dep_var.  Memoize per
    # run — values are small sets of block-id strings (KBs total), and callers
    # only read them via scope_blocks.update(...).  Scoped to this BFS loop.
    #
    # Lookup order: in-run memo → precomputed block_reach_facts in index.db
    # (Lineage Facts Index Phase 2; only materialized for before_line=0) →
    # live CFG walk.  lookup_reach_facts returns None whenever no fact is
    # available, so behavior is identical when the facts table is absent.
    from backward_traversal.utils.reach_facts import lookup_reach_facts

    _reach_cache: Dict[Tuple[str, str, int], Set[str]] = {}

    def _reach_cached(stem_key: str, bp_data: Dict[str, Any], block: Any,
                      before_line: int, bp_path: Optional[str]) -> Set[str]:
        _rk = (stem_key, str(block), int(before_line or 0))
        _rv = _reach_cache.get(_rk)
        if _rv is None:
            if bp_path:
                _rv = lookup_reach_facts(bp_path, str(block), int(before_line or 0))
            if _rv is None:
                _rv = backward_reachable_blocks(
                    bp_data, block,
                    before_line=before_line,
                    expand_subroutine_callers=True,
                    bp_path=bp_path,
                )
            _reach_cache[_rk] = _rv
        return _rv

    while dv_queue:
        Di, dv_depth = dv_queue.popleft()
        if Di in visited_dep_vars:
            continue
        if len(visited_dep_vars) - 1 >= max_dep_vars:
            output.setdefault("warnings", []).append(
                f"max_dep_vars={max_dep_vars} cap reached; remaining dep_vars skipped"
            )
            _trunc_detail["dep_var_count"] = True
            run_truncated = True  # GAP-TRUNC
            break
        if dv_depth >= max_dep_var_depth:
            output.setdefault("warnings", []).append(
                f"dep_var {Di} skipped — max_dep_var_depth={max_dep_var_depth} reached"
            )
            _trunc_detail["dep_var_depth"] = True
            run_truncated = True  # GAP-TRUNC
            continue

        # ── Memory guard: abort dep_var BFS if RSS exceeds cap ──
        # Check every 10 dep_vars to avoid per-iteration syscall overhead.
        if _MAX_RSS_MB > 0 and len(visited_dep_vars) % 10 == 0:
            _rss = _get_rss_mb()
            if _rss > _MAX_RSS_MB:
                output.setdefault("warnings", []).append(
                    f"dep_var BFS aborted: RSS {_rss:.0f} MB exceeds "
                    f"ASM_MAX_RSS_MB={_MAX_RSS_MB}"
                )
                _trunc_detail["dep_var_memory"] = True
                run_truncated = True
                break

        visited_dep_vars.add(Di)
        logger.info("[BACKWARD] dep_var %d — %s (depth=%d, queue=%d) — %.1fs",
                    len(visited_dep_vars), Di, dv_depth, len(dv_queue),
                    time.monotonic() - _run_t0)
        print(f"[backward-only][dep_var depth={dv_depth}] tracing {Di}", file=sys.stderr)

        dv_truncated: bool = False  # GAP-TRUNC: per-dep_var truncation flag
        chain_file_results: Dict[str, Any] = {}
        downstream_targets: Dict[str, Tuple[str, str]] = {}  # target_stem -> (source_stem, source_Di)

        for stem in selected_chain:
            meta = file_meta.get(stem)
            if not meta:
                continue

            # T3 / GAP A: C++ branch — replaces the blanket "not meta['bp']" skip
            if meta.get("file_type") == "cpp":
                cpp_bp_str = meta.get("cpp_bp_path")
                if not cpp_bp_str:
                    continue
                cpp_bp_p = Path(cpp_bp_str)
                if not cpp_bp_p.exists():
                    continue
                # Served by the blueprint frame cache (one entry per path);
                # key-filtered so only the needed frames are decoded.
                cpp_bp = load_json(cpp_bp_p, keys={"call_graph"})

                # Pre-filter (Change 3): skip GVL + BFS + T8 when we know
                # this dep_var has no seeds in this file.  Call_graph scanning
                # (NEW-B block) still runs to populate downstream_targets.
                _pf_relevant = _dv_relevant_stems.get(Di.lower())
                # Guard: when _pf_relevant is an empty set, the DB found no
                # seeds anywhere for this dep_var.  Don't skip the full scan —
                # the DB index may have gaps (short names, stale data).
                if _pf_relevant is not None and _pf_relevant and stem.lower() not in _pf_relevant:
                    chain_file_results[stem] = {
                        "dep_var": Di, "file_type": "cpp",
                        "lineage_trace": {}, "setter_sites": [],
                        "setter_site_traces": [],
                        "call_graph": {}, "warnings": [],
                    }
                    # Jump straight to the NEW-B call_graph downstream block.
                    # This code is duplicated from lines below to avoid a large
                    # refactor; the downstream_targets / full_flow side-effects
                    # must run regardless of seed presence.
                    _anchor_fns_pf: Set[str] = meta.get("anchor_blocks") or set()
                    _cpp_fn_names_pf: Set[str] = {
                        str(e.get("source") or "")
                        for e in (cpp_bp.get("call_graph", {}) or {}).get("edges", [])
                        if e.get("source")
                    }
                    _anchor_pf_are_cpp = bool(_anchor_fns_pf and (_anchor_fns_pf & _cpp_fn_names_pf))
                    _seen_cpp_pf: Set[Tuple[str, Any]] = set()
                    for _edge_pf in (cpp_bp.get("call_graph", {}) or {}).get("edges", []):
                        _src_fn_pf = str(_edge_pf.get("source") or "")
                        if _anchor_pf_are_cpp and _src_fn_pf not in _anchor_fns_pf:
                            continue
                        _tgt_pf = _norm_file_id(str(_edge_pf.get("target") or ""))
                        if not _tgt_pf:
                            continue
                        _key_pf: Tuple[str, Any] = (_src_fn_pf, _edge_pf.get("line"))
                        if _key_pf in _seen_cpp_pf:
                            continue
                        _seen_cpp_pf.add(_key_pf)
                        _add_full_flow_edge(
                            full_flow,
                            source=stem,
                            target=_tgt_pf,
                            blueprint_dir=blueprint_dir,
                            file_type_map=file_type_map,
                            selected_chain_set=selected_chain_set,
                            scope_var=Di,
                            instruction=str(_edge_pf.get("instruction") or ""),
                            discovered_from="dep_var_cpp_scope",
                            source_block=_src_fn_pf,
                            line_no=_edge_pf.get("line"),
                        )
                        if _tgt_pf not in exclude_stems and _tgt_pf not in downstream_targets:
                            downstream_targets[_tgt_pf] = (stem, Di)
                    continue  # skip GVL/BFS/T8 for this stem

                # Served by lazy_gvl._LAZY_GVL_CACHE (stable instance per
                # db/path → id(gvl)-keyed caches hit across dep_vars).
                from backward_traversal.utils.lazy_gvl import make_lazy_gvl as _mkgvl
                cpp_gvl = _mkgvl(cpp_bp, blueprint_path=cpp_bp_p)
                # FIX 19: find the immediately preceding C++ stem in selected_chain
                # so resolver.resolve() can translate dep_var names at C++→C++ boundaries.
                _stem_idx = selected_chain.index(stem) if stem in selected_chain else -1
                _source_stem: Optional[str] = None
                if _stem_idx > 0:
                    for _prev in reversed(selected_chain[:_stem_idx]):
                        # NEW-BUG-3: use resolve_file_type() for case-safe lookup
                        # instead of file_type_map.get(_prev) which may miss mixed-
                        # case stems that differ from the map keys.
                        if resolve_file_type(_prev, blueprint_dir, file_type_map) == "cpp":
                            _source_stem = _prev
                            break
                # Use the shared DB reverse index when this GVL shares the
                # same DB backend — keeps the LRU cache warm across all stems.
                _use_shared = False
                if _shared_db_rev is not None:
                    _gvl_db = getattr(cpp_gvl, "_db_path", None)
                    _gvl_jid = getattr(cpp_gvl, "_job_id", None)
                    if _gvl_db and _gvl_jid and (str(_gvl_db), str(_gvl_jid)) == _shared_db_key:
                        _use_shared = True
                if _use_shared:
                    _cached_rev, _cached_nl = _shared_db_rev, _shared_db_nl
                else:
                    # Memoize reverse_idx + node_lookup per GVL instance
                    _gvl_key = id(cpp_gvl)
                    if _gvl_key not in _gvl_idx_cache:
                        from backward_traversal.tracing.cpp_backward_tracer import (
                            _build_reverse_index, _build_node_lookup,
                        )
                        _gvl_idx_cache[_gvl_key] = (
                            _build_reverse_index(cpp_gvl),
                            _build_node_lookup(cpp_gvl),
                        )
                    _cached_rev, _cached_nl = _gvl_idx_cache[_gvl_key]
                _cpp_result = _build_dep_var_cpp_result(
                    Di,
                    cpp_gvl,
                    resolver=resolver,
                    stem=stem,
                    source_stem=_source_stem,
                    blueprint_dir=blueprint_dir,
                    asm_dir=asm_dir,
                    seed_cache=_seed_cache,
                    bulk_seed_map=_bulk_seed_map,
                    bulk_seed_unfiltered=_bulk_seed_unfiltered,
                    _reverse_idx=_cached_rev,
                    _node_lookup=_cached_nl,
                    source_index_cache=_source_idx_cache,
                )
                chain_file_results[stem] = _cpp_result

                # T8: when chain C++ file has no seeds and T8 is enabled, search
                # nearby C++ files outside the selected chain.  Guard prevents
                # repeat searches for the same dep_var across multiple chain stems.
                if (not _cpp_result.get("lineage_trace")
                        and t8_max_scan > 0
                        and Di not in _t8_searched):
                    _t8_searched.add(Di)
                    # Use precomputed DB candidate map when available (Change 5)
                    _ext_stems = _t8_precomputed.get(Di.lower())
                    if _ext_stems is None:
                        _ext_stems = find_cpp_files_for_dep_var(
                            Di, blueprint_dir, file_type_map,
                            max_scan=t8_max_scan,
                            neighbor_set=t8_neighbor_set,
                        )
                    for _ext_stem in _ext_stems:
                        if _ext_stem in chain_file_results:
                            continue   # already a chain file — skip
                        _ext_bp_path = resolve_cpp_blueprint(_ext_stem, blueprint_dir)
                        if not _ext_bp_path:
                            continue
                        try:
                            _ext_bp = load_json(_ext_bp_path, keys={"call_graph"})
                        except Exception:
                            continue
                        from backward_traversal.utils.lazy_gvl import make_lazy_gvl as _mkgvl
                        _ext_gvl = _mkgvl(_ext_bp, blueprint_path=_ext_bp_path)
                        # Reuse shared DB index for T8 candidates too.
                        _ext_use_shared = False
                        if _shared_db_rev is not None:
                            _ext_db = getattr(_ext_gvl, "_db_path", None)
                            _ext_jid = getattr(_ext_gvl, "_job_id", None)
                            if _ext_db and _ext_jid and (str(_ext_db), str(_ext_jid)) == _shared_db_key:
                                _ext_use_shared = True
                        if _ext_use_shared:
                            _ext_rev, _ext_nl = _shared_db_rev, _shared_db_nl
                        else:
                            _ext_gvl_key = id(_ext_gvl)
                            if _ext_gvl_key not in _gvl_idx_cache:
                                from backward_traversal.tracing.cpp_backward_tracer import (
                                    _build_reverse_index, _build_node_lookup,
                                )
                                _gvl_idx_cache[_ext_gvl_key] = (
                                    _build_reverse_index(_ext_gvl),
                                    _build_node_lookup(_ext_gvl),
                                )
                            _ext_rev, _ext_nl = _gvl_idx_cache[_ext_gvl_key]
                        _ext_result = _build_dep_var_cpp_result(
                            Di, _ext_gvl, resolver=resolver, stem=_ext_stem, source_stem=stem,
                            blueprint_dir=blueprint_dir, asm_dir=asm_dir,
                            seed_cache=_seed_cache,
                            bulk_seed_map=_bulk_seed_map,
                    bulk_seed_unfiltered=_bulk_seed_unfiltered,
                            _reverse_idx=_ext_rev,
                            _node_lookup=_ext_nl,
                            source_index_cache=_source_idx_cache,
                        )
                        if _ext_result.get("lineage_trace"):
                            extended_dep_var_results.setdefault(Di, {})[_ext_stem] = _ext_result

                # NEW-B: scan call_graph.edges for downstream callees reachable from
                # this C++ chain file, scoped to anchor functions (if any).
                # Mirrors the ASM path (~line 1779-1800) which builds downstream_targets
                # after _collect_scoped_cross_file_calls.  Without this, C++ chain files
                # never contribute entries to downstream_targets in the dep_var BFS.
                _anchor_fns_b: Set[str] = meta.get("anchor_blocks") or set()
                # NEW-ISSUE-2: apply the same NEW-BUG-1 fix here — anchor_fns for an
                # ASM→C++ callee contains ASM block names, not C++ function names.
                # Only filter when anchor_fns actually overlaps C++ source node names.
                _cpp_fn_names_b: Set[str] = {
                    str(e.get("source") or "")
                    for e in (cpp_bp.get("call_graph", {}) or {}).get("edges", [])
                    if e.get("source")
                }
                _anchor_fns_b_are_cpp = bool(_anchor_fns_b and (_anchor_fns_b & _cpp_fn_names_b))
                _seen_cpp_b: Set[Tuple[str, Any]] = set()
                for _edge_b in (cpp_bp.get("call_graph", {}) or {}).get("edges", []):
                    _src_fn_b = str(_edge_b.get("source") or "")
                    if _anchor_fns_b_are_cpp and _src_fn_b not in _anchor_fns_b:
                        continue
                    _tgt_b = _norm_file_id(str(_edge_b.get("target") or ""))
                    if not _tgt_b:
                        continue
                    _key_b: Tuple[str, Any] = (_src_fn_b, _edge_b.get("line"))
                    if _key_b in _seen_cpp_b:
                        continue
                    _seen_cpp_b.add(_key_b)
                    _add_full_flow_edge(
                        full_flow,
                        source=stem,
                        target=_tgt_b,
                        blueprint_dir=blueprint_dir,
                        file_type_map=file_type_map,
                        selected_chain_set=selected_chain_set,
                        scope_var=Di,
                        instruction=str(_edge_b.get("instruction") or ""),
                        discovered_from="dep_var_cpp_scope",
                        source_block=_src_fn_b,
                        line_no=_edge_b.get("line"),
                    )
                    if _tgt_b not in exclude_stems and _tgt_b not in downstream_targets:
                        downstream_targets[_tgt_b] = (stem, Di)
                continue  # skip ASM path below

            # Existing ASM path — guard now explicit after C++ was routed above  (GAP A)
            # GAP-036: asm_file may be None when source is absent; blueprint is sufficient.
            if not meta.get("bp"):
                continue

            # GAP-ROOT-UPSTREAM: skip the modifier file when tracing the root variable.
            # The modifier file was already fully and correctly scanned in Pass 1 using
            # the literal variable name (no resolver translation).  Re-scanning here via
            # _build_dep_var_file_result applies resolver.resolve() which can map the
            # root variable to a different name in the modifier context, producing false
            # positives (e.g. WB1PTMST → CR21GLPTDTE in xh72).  Upstream files are
            # unaffected — they are only traced here, never by Pass 1.
            if Di == root_var and stem == modifier_stem:
                continue

            bp_data = load_json(meta["bp"], keys={"blocks", "call_graph"})
            collection_blocks = dep_var_collection_map.get(Di, {}).get(stem, set())

            # GAP-ROOT-UPSTREAM: the root variable is traced with CSECT-scoped
            # forward-reachable blocks to capture error-exit paths (blocks reachable
            # only via forward branches within the CSECT) while preventing cross-CSECT
            # leakage from other entry points in the same source file.  The CSECT root
            # is found by walking cfg_incoming backward from the anchor blocks; the full
            # CSECT scope is then the forward-reachable set from that root.
            # Falls back to full_file when no anchor_blocks are available.
            anchor_before_lines = meta.get("anchor_before_lines") or {}
            scope_line_caps: Dict[str, int] = {}

            if Di == root_var:
                _root_bp_str = str(meta["bp"]) if meta.get("bp") else ""
                _all_file_blocks: Set[str] = {str(b["id"]) for b in bp_data.get("blocks", [])}
                _root_anchors = meta.get("anchor_blocks") or set()
                if _root_bp_str and _root_anchors:
                    scope_blocks = _csect_forward_reachable(
                        _root_bp_str, _root_anchors, _all_file_blocks
                    )
                    scope_label = "csect_scoped"
                else:
                    scope_blocks = _all_file_blocks
                    scope_label = "full_file"
            elif collection_blocks:
                scope_blocks: Set[str] = set()
                # GAP-020: use only collection_blocks as the scope seed.
                # Unioning with meta["anchor_blocks"] (root-var anchors) pollutes
                # the dep_var scope with irrelevant predecessor paths from the
                # root variable's collection sites, producing spurious setter
                # matches that inflate Pass 2 output.  collection_blocks already
                # points at the blocks where the dep_var is set; backward
                # reachability from those blocks is sufficient.
                _p2_bp_path = str(meta["bp"]) if meta.get("bp") else None
                for b in collection_blocks:
                    before_line = anchor_before_lines.get(b, 0) or 0
                    scope_blocks.update(_reach_cached(
                        stem, bp_data, b, before_line, _p2_bp_path,
                    ))
                    if before_line:
                        scope_line_caps[b] = before_line   # NEW
                # FIX-FALLTHROUGH: for the modifier file, also seed backward
                # reachability from the root-var setter blocks (anchor_blocks).
                # collection_blocks only contains blocks that READ the dep_var in
                # the Pass 1 trace.  A block that ONLY WRITES the dep_var (e.g.
                # via a FALLTHROUGH path from a conditional branch) is a CFG
                # predecessor of the root-var setter but never appears as a
                # dependent_variable in the trace nodes, so it is absent from
                # collection_blocks and missed entirely.  For the modifier file
                # the anchor_blocks are setter sites of the root variable (not
                # call sites), so backward reachability from them with the
                # before_line cap is safe and targeted — it does not carry the
                # false-positive risk that motivated GAP-020 for upstream files.
                if stem == modifier_stem and meta.get("anchor_blocks"):
                    for b in meta["anchor_blocks"]:
                        before_line = anchor_before_lines.get(b, 0) or 0
                        scope_blocks.update(_reach_cached(
                            stem, bp_data, b, before_line, _p2_bp_path,
                        ))
                        if before_line:
                            scope_line_caps[b] = min(
                                scope_line_caps.get(b, before_line), before_line
                            )
                scope_label = "collection_based"
            elif meta["anchor_blocks"]:
                scope_blocks = set()
                _p2_bp_path = str(meta["bp"]) if meta.get("bp") else None
                for b in meta["anchor_blocks"]:
                    before_line = anchor_before_lines.get(b, 0) or 0
                    scope_blocks.update(_reach_cached(
                        stem, bp_data, b, before_line, _p2_bp_path,
                    ))
                    if before_line:
                        scope_line_caps[b] = before_line   # NEW
                scope_label = "anchor_based"
            else:
                # No anchor and no collection info — fall back to full file so we
                # don't silently miss setters when Pass 1 found no call/setter sites.
                scope_blocks = {str(b["id"]) for b in bp_data.get("blocks", [])}
                scope_label = "full_file"
                output.setdefault("warnings", []).append(
                    f"dep_var {Di} in {stem}: no anchor_blocks found — using full_file scope"
                )

            scoped_cross_calls = _collect_scoped_cross_file_calls(
                bp_data, scope_blocks, meta["asm_file"], asm_dir, bp_path=meta["bp"]
            )
            returning_scoped_cross_calls = [cf for cf in scoped_cross_calls if _is_followable_cross_file_call(cf)]
            _p2_bp_path_stem = str(meta["bp"]) if meta.get("bp") else None
            for cf in returning_scoped_cross_calls:
                _conds = _extract_block_conditions_before_line(
                    bp_data, str(cf.get("source_block") or ""), int(cf.get("line") or 0),
                    bp_path=_p2_bp_path_stem,
                )
                _add_full_flow_edge(
                    full_flow,
                    source=stem,
                    target=str(cf.get("target_file") or ""),
                    blueprint_dir=blueprint_dir,
                    file_type_map=file_type_map,
                    selected_chain_set=selected_chain_set,
                    scope_var=Di,
                    instruction=str(cf.get("instruction") or ""),
                    discovered_from="dep_var_scope",
                    source_block=str(cf.get("source_block") or ""),
                    line_no=cf.get("line"),
                    conditions=_conds,
                )
            for cf in returning_scoped_cross_calls:
                tgt = cf["target_file"]
                if tgt and tgt not in exclude_stems and tgt not in downstream_targets:
                    downstream_targets[tgt] = (stem, Di)

            # Resolve dep_var name in this chain file when not natively collected here.
            # GAP-015: derive the preceding chain stem so Cases C/D (arg_bind, struct
            # canonical) are attempted at ASM→ASM chain-file boundaries.
            _prev_chain_stem = prev_chain_stem_map.get(_norm_file_id(stem))
            chain_Di = (resolver.resolve(Di, stem, source_stem=_prev_chain_stem) if not collection_blocks else None) or Di

            chain_file_results[stem] = _build_dep_var_file_result(
                chain_Di, bp_data, meta["bp"], meta["asm_file"], scope_blocks, scope_label,
                asm_dir, max_depth, max_subroutine_depth, max_subroutine_nodes, max_trace_nodes,
                scope_line_caps=scope_line_caps,  # NEW
            )

            # P8/P30: collect named variables pre-loaded into registers before cross-file
            # calls (LA R1,PARM_AREA then ENTRC TARGET).  These are implicit dep_vars of
            # the traced variable in this chain file and must be surfaced as new BFS entries
            # so the parameter areas are also traced for setter sites.
            if meta.get("file_type", "asm") == "asm" and meta.get("bp"):
                _pre_call_dvs = _collect_pre_call_dep_vars(bp_data, scope_blocks)
                for _pcdv in _pre_call_dvs:
                    _pcdv_up = _pcdv.upper()
                    if _is_noise_dep_var(_pcdv_up):
                        continue
                    # Add stem → empty collection_blocks (full scope will be used when traced)
                    dep_var_collection_map.setdefault(_pcdv_up, {}).setdefault(stem, set())
                    _pc_parents = dep_var_parent_map.setdefault(_pcdv_up, [])
                    if Di not in _pc_parents:
                        _pc_parents.append(Di)
                    if _pcdv_up not in visited_dep_vars and _pcdv_up not in already_queued:
                        already_queued.add(_pcdv_up)
                        dv_queue.append((_pcdv_up, dv_depth + 1))
                        _prefilter_dep_var(_pcdv_up)  # Change A: incremental pre-filter

        # Downstream file BFS
        downstream_file_results: Dict[str, Any] = {}
        visited_downstream: Set[str] = set()
        ds_queue: Deque[Tuple[str, int, str, str, List[str]]] = deque(
            (s, 0, src_stem, src_var, _path_to_chain_file(src_stem, selected_chain))
            for s, (src_stem, src_var) in sorted(downstream_targets.items())
        )

        while ds_queue:
            ds_stem, ds_depth, ds_source_stem, ds_source_var, _path_to_source = ds_queue.popleft()
            if ds_stem in visited_downstream or ds_stem in exclude_stems:
                continue
            if len(visited_downstream) >= max_downstream_files:
                output.setdefault("warnings", []).append(
                    f"dep_var {Di}: max_downstream_files={max_downstream_files} cap reached"
                )
                dv_truncated = True   # GAP-TRUNC
                _trunc_detail["dv_downstream_files"] = True
                run_truncated = True  # GAP-TRUNC
                break
            if ds_depth >= max_downstream_depth:
                visited_downstream.add(ds_stem)
                output.setdefault("warnings", []).append(
                    f"dep_var {Di}: downstream depth cap for {ds_stem}"
                )
                dv_truncated = True   # GAP-TRUNC
                _trunc_detail["dv_downstream_depth"] = True
                run_truncated = True  # GAP-TRUNC
                continue

            bp_ds = resolve_asm_blueprint(ds_stem, blueprint_dir)
            asm_ds = resolve_source_file(ds_stem, "asm", blueprint_dir, asm_dir)

            if not bp_ds or not asm_ds:
                # NEW-A: C++ callee stems (queued from NEW-B) have no ASM blueprint.
                # Resolve as C++ and trace dep_var through the GVL — mirrors the
                # root_var downstream C++ branch (lines ~1547-1592).
                _cpp_bp_ds = resolve_cpp_blueprint(ds_stem, blueprint_dir)
                if not _cpp_bp_ds:
                    continue
                visited_downstream.add(ds_stem)
                _cpp_bp_data_ds = load_json(_cpp_bp_ds, keys={"call_graph"})
                from backward_traversal.utils.lazy_gvl import make_lazy_gvl as _mkgvl
                _gvl_ds = _mkgvl(_cpp_bp_data_ds, blueprint_path=_cpp_bp_ds)
                # ISSUE-1: pass source_stem so Cases C/D are attempted for dep_var
                # name translation across downstream C++ file boundaries.
                # NEW-ISSUE-3: uppercase the resolved name so it stays consistent with
                # all other dep_var keys in dep_var_collection_map and visited_dep_vars.
                # resolver.resolve() may return mixed-case field names (e.g. "cmSystem");
                # without .upper() the downstream queue entry and the dedup set diverge.
                _ds_Di = (resolver.resolve(ds_source_var, ds_stem, source_stem=ds_source_stem) or ds_source_var).upper()
                print(
                    f"[backward-only][dep_var {Di} downstream depth={ds_depth}][cpp]"
                    f" tracing {_ds_Di} in {ds_stem}",
                    file=sys.stderr,
                )
                downstream_file_results[ds_stem] = _build_dep_var_cpp_result(
                    _ds_Di, _gvl_ds,
                    resolver=resolver, stem=ds_stem, source_stem=ds_source_stem,
                    blueprint_dir=blueprint_dir, asm_dir=asm_dir,
                    seed_cache=_seed_cache,
                    bulk_seed_map=_bulk_seed_map,
                    bulk_seed_unfiltered=_bulk_seed_unfiltered,
                    source_index_cache=_source_idx_cache,
                )
                # Propagate further downstream via call_graph.edges.
                _ds_rv_excl = exclude_stems | visited_downstream
                _seen_ds_cpp: Set[Tuple[str, Any]] = set()
                for _edge_ds in (_cpp_bp_data_ds.get("call_graph", {}) or {}).get("edges", []):
                    _tgt_ds = _norm_file_id(str(_edge_ds.get("target") or ""))
                    if not _tgt_ds or _tgt_ds in _ds_rv_excl:
                        continue
                    _key_ds: Tuple[str, Any] = (str(_edge_ds.get("source") or ""), _edge_ds.get("line"))
                    if _key_ds in _seen_ds_cpp:
                        continue
                    _seen_ds_cpp.add(_key_ds)
                    _add_full_flow_edge(
                        full_flow,
                        source=ds_stem,
                        target=_tgt_ds,
                        blueprint_dir=blueprint_dir,
                        file_type_map=file_type_map,
                        selected_chain_set=selected_chain_set,
                        scope_var=Di,
                        instruction=str(_edge_ds.get("instruction") or ""),
                        discovered_from="dep_var_cpp_downstream_scope",
                        source_block=str(_edge_ds.get("source") or ""),
                        line_no=_edge_ds.get("line"),
                    )
                    ds_queue.append(
                        (_tgt_ds, ds_depth + 1, ds_stem, _ds_Di, _path_to_source + [ds_stem])
                    )
                continue  # skip ASM path below

            visited_downstream.add(ds_stem)
            bp_data_ds = load_json(bp_ds, keys={"blocks", "call_graph"})
            all_blocks: Set[str] = {str(b["id"]) for b in bp_data_ds.get("blocks", [])}

            # GAP-028: when collection_blocks are already known for this dep_var in
            # this downstream file (from a prior BFS pass), use them as scope seeds
            # instead of full-file scope.  Using all_blocks produces scope explosion
            # proportional to file size; collection_blocks anchors the trace to
            # the blocks that actually write or use the dep_var.
            _ds_p2_bp_path = str(bp_ds)
            _ds_coll_blocks = dep_var_collection_map.get(Di, {}).get(ds_stem, set())
            if _ds_coll_blocks:
                _ds_scope_blocks: Set[str] = set()
                for _b in _ds_coll_blocks:
                    # PERF: visited_downstream resets per dep_var, so the same
                    # (file, block) walk recurs across dep_vars — serve it from
                    # the run-scoped _reach_cache.
                    _ds_scope_blocks.update(_reach_cached(
                        ds_stem, bp_data_ds, _b, 0, _ds_p2_bp_path,
                    ))
                _ds_scope_label = "collection_based"
            else:
                # No collection blocks known — restrict to the downstream CSECT's
                # forward-reachable blocks.  The entry block ID is ds_stem.upper()
                # (the CSECT label that was the ENTRC target, normalised to lowercase
                # by _norm_file_id then uppercased back here).  Falls back to
                # all_blocks when the entry block is absent from the blueprint.
                _ds_fwd = _forward_reachable_from(str(bp_ds), {ds_stem.upper()})
                if _ds_fwd is not None:
                    _ds_scope_blocks = _ds_fwd
                    _ds_scope_label = "csect_scoped"
                else:
                    _ds_scope_blocks = all_blocks
                    _ds_scope_label = "full_file"

            # Cross-file call detection always uses full scope so we don't miss
            # downstream call edges that are outside the narrowed scope.
            ds_cross_calls = _collect_scoped_cross_file_calls(
                bp_data_ds, all_blocks, asm_ds, asm_dir, bp_path=bp_ds
            )
            ds_returning_cross_calls = [cf for cf in ds_cross_calls if _is_followable_cross_file_call(cf)]
            for cf in ds_returning_cross_calls:
                _conds = _extract_block_conditions_before_line(
                    bp_data_ds, str(cf.get("source_block") or ""), int(cf.get("line") or 0),
                    bp_path=_ds_p2_bp_path,
                )
                _add_full_flow_edge(
                    full_flow,
                    source=ds_stem,
                    target=str(cf.get("target_file") or ""),
                    blueprint_dir=blueprint_dir,
                    file_type_map=file_type_map,
                    selected_chain_set=selected_chain_set,
                    scope_var=Di,
                    instruction=str(cf.get("instruction") or ""),
                    discovered_from="dep_var_downstream_scope",
                    source_block=str(cf.get("source_block") or ""),
                    line_no=cf.get("line"),
                    conditions=_conds,
                )

            # NEW-P2-1: uppercase for consistent key normalization — completes the
            # pattern from line 1900 (C++ dep_var branch) and lines 1553/1614 (root_var).
            # GAP-015: pass ds_source_stem so Cases C/D fire at ASM→ASM downstream boundaries.
            ds_Di = (resolver.resolve(ds_source_var, ds_stem, source_stem=ds_source_stem) or ds_source_var).upper()

            downstream_file_results[ds_stem] = _build_dep_var_file_result(
                ds_Di, bp_data_ds, bp_ds, asm_ds, _ds_scope_blocks, _ds_scope_label,
                asm_dir, max_depth, max_subroutine_depth, max_subroutine_nodes, max_trace_nodes,
            )
            ds_exclude = exclude_stems | visited_downstream
            ds_queue.extend(
                (s, ds_depth + 1, ds_stem, ds_Di, _path_to_source + [ds_stem])
                for s in sorted({
                    cf["target_file"]
                    for cf in ds_returning_cross_calls
                    if cf["target_file"] and cf["target_file"] not in ds_exclude
                })
            )

        # Annotate setter sites in chain and downstream results with block_ref
        for _cfr_stem, _cfr in chain_file_results.items():
            _cfr_sites = _cfr.get("setter_sites", [])
            if _cfr_sites:
                _cfr_ft = _cfr.get("file_type", "asm")
                _cfr_meta = file_meta.get(_cfr_stem, {})
                _annotate_setter_sites(
                    _cfr_sites, stem=_cfr_stem, file_type=_cfr_ft,
                    collector=_block_collector,
                    bp_data=load_json(_cfr_meta["bp"]) if _cfr_meta.get("bp") else None,
                    asm_file=_cfr_meta.get("asm_file"),
                    blueprint_dir=blueprint_dir, asm_dir=asm_dir,
                    _source_cache=_block_source_cache,
                )
        for _dsr_stem, _dsr in downstream_file_results.items():
            _dsr_sites = _dsr.get("setter_sites", [])
            if _dsr_sites:
                _dsr_ft = _dsr.get("file_type", "asm")
                _annotate_setter_sites(
                    _dsr_sites, stem=_dsr_stem, file_type=_dsr_ft,
                    collector=_block_collector,
                    bp_data=load_json(resolve_asm_blueprint(_dsr_stem, blueprint_dir)) if _dsr_ft == "asm" else None,
                    asm_file=resolve_source_file(_dsr_stem, "asm", blueprint_dir, asm_dir) if _dsr_ft == "asm" else None,
                    blueprint_dir=blueprint_dir, asm_dir=asm_dir,
                    _source_cache=_block_source_cache,
                )

        # Issue-16: skip writing the root variable as its own dep_var (circular).
        # The BFS still processes it (to discover setter sites in non-modifier
        # chain files), but the output file is suppressed.
        if Di.upper() != variable.upper():
            _dep_var_writer.write(Di, {
                "depth": dv_depth,
                "truncated": dv_truncated,  # GAP-TRUNC
                "_parent_vars": dep_var_parent_map.get(Di, []),
                "chain_file_results": chain_file_results,
                "downstream_file_results": downstream_file_results,
                "extended_file_results": extended_dep_var_results.get(Di, {}),  # T8 cross-chain
            })

        # Feed next-level dep_vars into the BFS (must happen before del below)
        # GAP-C: include T8 cross-chain extended results so their dep_vars
        # enter the BFS.  extended_dep_var_results[Di] maps ext_stem →
        # _build_dep_var_cpp_result dict (file_type="cpp", lineage_trace=...).
        _ext_for_child = extended_dep_var_results.get(Di, {})
        child_map = _build_child_dep_var_map(
            chain_file_results, {**downstream_file_results, **_ext_for_child}
        )
        # Record parent→child relationships for the target schema transformer.
        for _child_dv in child_map:
            _parents = dep_var_parent_map.setdefault(_child_dv, [])
            if Di not in _parents:
                _parents.append(Di)
        # In partitioned mode the writer flushed to disk above; release large dicts.
        del chain_file_results, downstream_file_results
        for new_dv, file_blocks in child_map.items():
            # GAP-038: belt-and-suspenders noise filter at BFS insertion — ensures
            # any token that slipped past _build_child_dep_var_map doesn't enter the queue.
            if _is_noise_dep_var(new_dv):
                continue
            existing = dep_var_collection_map.setdefault(new_dv, {})
            # GAP-005: track whether this update adds genuinely new coverage (a new
            # stem or new blocks within an existing stem).  If it does, and the dep_var
            # has already been processed, it must be re-queued so the expanded scope is
            # actually traced.  Without this the union write happens but is never read.
            new_coverage = False
            for stem, blocks in file_blocks.items():
                if stem in existing:
                    old_size = len(existing[stem])
                    existing[stem] |= blocks  # union — never shrink a dep_var's collection
                    if len(existing[stem]) > old_size:
                        new_coverage = True
                else:
                    existing[stem] = set(blocks)
                    new_coverage = True

            _gap005_requeue = False
            if new_dv in visited_dep_vars and new_coverage:
                # GAP-005: new stems/blocks discovered for an already-processed dep_var.
                # Drop it from both guard sets so the standard enqueue check below
                # re-queues it.  After the second trace no further new coverage will
                # be produced, so this fires at most once per dep_var per new stem.
                visited_dep_vars.discard(new_dv)
                already_queued.discard(new_dv)
                _gap005_requeue = True

            if new_dv not in visited_dep_vars and new_dv not in already_queued:  # FIX 20
                already_queued.add(new_dv)
                # GAP-027: GAP-005 re-queues must use depth 0, not dv_depth+1.
                # Using dv_depth+1 for a re-queued already-visited dep_var can push it
                # past max_dep_var_depth and silently discard the expanded-scope trace.
                dv_queue.append((new_dv, 0 if _gap005_requeue else dv_depth + 1))
                _prefilter_dep_var(new_dv)  # Change A: incremental pre-filter

    logger.info("[BACKWARD] Pass 2 done — %d dep_vars traced (pre-filter mapped %d) — %.1fs",
                len(visited_dep_vars), len(_dv_relevant_stems), time.monotonic() - _run_t0)

    # Release large BFS-phase dicts now that all dep_var traces are written.
    del extended_dep_var_results, dep_var_collection_map, dep_var_parent_map

    # ── cache-pressure warning (GAP-011) ────────────────────────────────────
    _ci = get_blueprint_cache_info()
    if _ci is not None:
        _total = (_ci.hits or 0) + (_ci.misses or 0)
        if _total > 0:
            _miss_rate = (_ci.misses or 0) / _total
            _at_cap = _ci.maxsize is not None and (_ci.currsize or 0) >= _ci.maxsize
            if _miss_rate > 0.3 or _at_cap:
                output.setdefault("warnings", []).append(
                    f"Blueprint cache pressure (GAP-011): hits={_ci.hits} "
                    f"misses={_ci.misses} currsize={_ci.currsize}/{_ci.maxsize} "
                    f"miss_rate={_miss_rate:.0%}. "
                    f"Raise ASM_BP_CACHE_SIZE above {_ci.maxsize} to improve performance."
                )

    if "warnings" in output:
        output["warnings"] = sorted(set(str(w) for w in output["warnings"] if w))
    output["truncated"] = run_truncated          # GAP-TRUNC: surfaces in root_var.json
    output["truncation_detail"] = _trunc_detail  # per-category breakdown of which cap(s) fired
    output["root_setter_found"] = root_setter_found
    clear_cfg_cache()

    # ── terminal classification ───────────────────────────────────────────────
    # Only feasible in single-file mode where dep_var_traces is fully in-memory.
    # Partitioned mode is handled downstream by variable_graph_task.py.
    _terminal_classes: List[Dict[str, Any]] = []
    if not _dep_var_writer.partitioned and _ENABLE_TERMINAL_CLASSIFICATION:
        try:
            _tmp_for_terminals = {"dep_var_traces": _dep_var_writer.to_dict()}
            _terminal_vars = extract_terminals_from_output(_tmp_for_terminals)
            if _terminal_vars:
                _terminal_classes = classify_terminal_variables(
                    _terminal_vars,
                    blueprint_dir,
                    asm_dir,
                    graph_file,
                    use_llm=False,
                )
        except Exception as _tc_err:
            import logging as _logging_tc
            _logging_tc.getLogger(__name__).warning(
                "Terminal classification failed (non-fatal): %s", _tc_err
            )
    elif not _ENABLE_TERMINAL_CLASSIFICATION:
        logger.info(
            "[BACKWARD] Terminal classification disabled via ASM_ENABLE_TERMINAL_CLASSIFICATION"
        )

    # ── output assembly ──────────────────────────────────────────────────────
    if not _dep_var_writer.partitioned:
        # ── single-file mode (default) ────────────────────────────────────────
        # Assemble everything into one dict and write atomically, preserving the
        # original output contract expected by downstream consumers.
        output["root_var_downstream_traces"] = root_var_downstream_traces
        output["dep_var_traces"] = _dep_var_writer.to_dict()
        output["dep_var_setter_summary"] = _consolidate_dep_var_setters(
            output["dep_var_traces"]
        )
        _compute_data_flow_contributions(full_flow, blueprint_dir)
        _enrich_function_dispatch_conditions(full_flow, blueprint_dir)
        _ffl = _finalize_full_flow_collector(full_flow)
        del full_flow  # collector consumed; release before serialization
        if _terminal_classes:
            _ffl["terminal_classifications"] = _terminal_classes
            output["terminal_classifications"] = _terminal_classes
        output["full_file_flow"] = _ffl
        _fb = _block_collector.to_dict()
        if _fb:
            output["function_blocks"] = _fb
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        del output  # release all accumulated dicts (files, root_var_downstream_traces, etc.)
        print(f"Wrote backward-only results to: {output_path}", file=sys.stderr)
    else:
        # ── partitioned mode ──────────────────────────────────────────────────
        # dep_var traces were already flushed to dep_vars/<name>.json individually.
        # Write the remaining sections as separate files and a manifest.json index.
        assert out_dir is not None  # guaranteed when partitioned_output=True

        # full_file_flow.json
        _compute_data_flow_contributions(full_flow, blueprint_dir)
        _enrich_function_dispatch_conditions(full_flow, blueprint_dir)
        _ffl_dict = _finalize_full_flow_collector(full_flow)
        del full_flow  # release memory

        # Terminal classification for partitioned mode: scan dep_var files
        if _ENABLE_TERMINAL_CLASSIFICATION:
            try:
                _dep_vars_dir = out_dir / "dep_vars"
                _pt_terminals = extract_terminals_from_partitioned_dir(_dep_vars_dir)
                if _pt_terminals:
                    _pt_classes = classify_terminal_variables(
                        _pt_terminals, blueprint_dir, asm_dir, graph_file, use_llm=False
                    )
                    if _pt_classes:
                        _ffl_dict["terminal_classifications"] = _pt_classes
            except Exception as _tc_err2:
                import logging as _log2
                _log2.getLogger(__name__).warning(
                    "Terminal classification (partitioned) failed (non-fatal): %s", _tc_err2
                )

        _ffl_path = out_dir / "full_file_flow.json"
        _ffl_path.write_text(json.dumps(_ffl_dict, indent=2), encoding="utf-8")

        # Keep dep-var enrichment decoupled from the often much larger
        # root_var.json payload: it only needs the modifier's call-chain and
        # setter-site context, not the downstream trace bodies.
        _modifier_record = ((output.get("files") or {}).get(modifier_stem) or {})
        _enrichment_ctx = {
            "modifier_file": modifier_stem,
            "modifier": _modifier_record.get("modifier") or {},
        }
        _enrichment_ctx_path = out_dir / "dep_var_enrichment_context.json"
        _enrichment_ctx_path.write_text(
            json.dumps(_enrichment_ctx, indent=2),
            encoding="utf-8",
        )
        # dep_var_setter_summary.json — consolidated setter sites across all dep_vars
        _dep_vars_dir = out_dir / "dep_vars"
        _dvss = _consolidate_dep_var_setters_from_dir(_dep_vars_dir)
        if _dvss:
            _dvss_path = out_dir / "dep_var_setter_summary.json"
            _dvss_path.write_text(json.dumps(_dvss, indent=2), encoding="utf-8")

        # root_var.json — Pass 1 chain results + Pass 1b downstream
        output["root_var_downstream_traces"] = root_var_downstream_traces
        _rv_path = out_dir / "root_var.json"
        _rv_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        del root_var_downstream_traces, output  # release memory

        # function_blocks.json — deduplicated function/block source bodies
        _fb = _block_collector.to_dict()
        if _fb:
            (out_dir / "function_blocks.json").write_text(
                json.dumps(_fb, indent=2), encoding="utf-8"
            )

        # manifest.json — index of all output files
        # GAP-031: include variable, selected_chain, and identity_index_path so
        # downstream consumers (API, chunked_session) can reconstruct session
        # context without loading root_var.json first.
        _manifest: Dict[str, Any] = {
            "partitioned": True,
            "variable": variable.upper(),
            "selected_chain": selected_chain,
            "root_var_file": "root_var.json",
            "full_file_flow_file": "full_file_flow.json",
            "dep_var_enrichment_context_file": "dep_var_enrichment_context.json",
            "dep_vars_dir": "dep_vars",
            "dep_var_files": _dep_var_writer.manifest(),
            "identity_index_path": resolver.index_path or "",
            "truncated": run_truncated,               # GAP-TRUNC: surfaces in manifest.json
            "truncation_detail": _trunc_detail,       # per-category breakdown
            "root_setter_found": root_setter_found,
        }
        if _dvss:
            _manifest["dep_var_setter_summary_file"] = "dep_var_setter_summary.json"
        if _fb:
            _manifest["function_blocks_file"] = "function_blocks.json"
        if extend_modifier_sites and extended_modifier_files:
            _manifest["extend_modifier_sites"] = True
            _manifest["extended_modifier_files"] = sorted(extended_modifier_files.keys())
        (out_dir / "manifest.json").write_text(
            json.dumps(_manifest, indent=2), encoding="utf-8"
        )
        print(f"Wrote partitioned backward-only results to: {out_dir}", file=sys.stderr)

    logger.info("[BACKWARD] Complete — total %.1fs", time.monotonic() - _run_t0)
    return 0


def run_backward_only_from_namespace(args) -> int:
    variable = str(args.variable)
    selected_chain = [p.strip() for p in str(args.chain_files or "").split(",") if p.strip()]
    if not selected_chain:
        print("ERROR: --backward-only currently requires --chain-files.", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else Path(f"{variable}_cross_lineage_backward_only.json")

    if args.blueprint_dir:
        blueprint_dir = Path(args.blueprint_dir)
        if not blueprint_dir.exists():
            print(f"ERROR: Blueprint dir not found: {blueprint_dir}", file=sys.stderr)
            return 1
    else:
        blueprint_dir = discover_blueprint_dir()
        if blueprint_dir is None:
            print("ERROR: Could not discover blueprint directory. Pass --blueprint-dir.", file=sys.stderr)
            return 1

    if args.asm_dir:
        asm_dir = Path(args.asm_dir)
    else:
        asm_dir = discover_asm_dir(blueprint_dir)

    if args.graph:
        graph_file = Path(args.graph)
    else:
        graph_file = discover_graph_file(blueprint_dir)

    if graph_file is None or not graph_file.exists():
        print("ERROR: Cannot find file_call_graph.json. Pass --graph.", file=sys.stderr)
        return 1

    print(f"[backward-only blueprints] {blueprint_dir}", file=sys.stderr)
    if asm_dir:
        print(f"[backward-only asm-source] {asm_dir}", file=sys.stderr)
    print(f"[backward-only call graph] {graph_file}", file=sys.stderr)

    return run_backward_only(
        variable=variable,
        selected_chain=selected_chain,
        blueprint_dir=blueprint_dir,
        asm_dir=asm_dir,
        graph_file=graph_file,
        output_path=output_path,
        max_depth=int(args.max_depth) if args.max_depth is not None else 8,
        max_subroutine_depth=int(getattr(args, "max_subroutine_depth", DEFAULT_MAX_SUBROUTINE_DEPTH)),
        max_subroutine_nodes=int(getattr(args, "max_subroutine_nodes", DEFAULT_MAX_SUBROUTINE_NODES)),
        max_trace_nodes=int(getattr(args, "max_trace_nodes", DEFAULT_MAX_TRACE_NODES)),
        max_dep_vars=int(getattr(args, "max_dep_vars", DEFAULT_MAX_DEP_VARS)),
        max_dep_var_depth=int(getattr(args, "max_dep_var_depth", DEFAULT_MAX_DEP_VAR_DEPTH)),
        max_downstream_depth=int(getattr(args, "max_downstream_depth", DEFAULT_MAX_DOWNSTREAM_DEPTH)),
        max_downstream_files=int(getattr(args, "max_downstream_files", DEFAULT_MAX_DOWNSTREAM_FILES)),
        t8_max_scan=int(getattr(args, "t8_max_scan", 0)),   # FIX 18: opt-in; default disabled
        extend_modifier_sites=bool(getattr(args, "extend_modifier_sites", False)),
        # Auto-detect partitioned mode: directory path (no .json suffix) → partitioned.
        partitioned_output=bool(
            getattr(args, "partitioned_output", False)
            or (output_path.suffix != ".json" and not output_path.exists())
            or output_path.is_dir()
        ),
    )
