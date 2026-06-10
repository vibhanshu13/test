"""Variable lineage builder for assembly and C/C++."""

import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict

_log = logging.getLogger(__name__)
_SLOW_FILE_THRESHOLD: float = float(os.environ.get("SLOW_FILE_THRESHOLD_SECONDS", "10"))
from models.analysis_context import AnalysisContext
from tokenizer import ParsedLine
from models.variable_lineage import VariableLineageGraph
from literal_resolver import LiteralResolver
from models.c_family import CTranslationUnit
from models.db_operation import DBOperation, DBOperationCatalog, DBOperationType

# Standard HLASM instructions + assembler directives — anything NOT in this set
# and having a REG=Rn operand is treated as a DSECT registration macro.
_KNOWN_OPCODES = {
    # Data definition / directives
    "DS", "DC", "EQU", "USING", "DROP", "ORG", "COPY", "PRINT", "TITLE",
    "LTORG", "END", "CSECT", "DSECT", "RSECT", "MACRO", "MEND", "MEXIT",
    "PUSH", "POP", "SPACE", "EJECT", "ANOP", "AGO", "AGOB",
    "AIF", "AIFB", "GBLA", "GBLB", "GBLC", "LCLA", "LCLB", "LCLC",
    "SETA", "SETB", "SETC", "ACTR", "CNOP", "ENTRY", "EXTRN", "WXTRN",
    "AMODE", "RMODE", "LOCTR", "ALIAS", "XATTR", "COM", "DXD",
    # Load / store
    "L", "LR", "LTR", "LA", "LH", "LM", "LTM", "IC", "ICM",
    "ST", "STM", "STC", "STCM", "STH", "STCK",
    "LGR", "LGHI", "LLGF", "LGF", "LG", "STG", "STMG", "LMG",
    # Arithmetic
    "A", "AH", "AR", "S", "SH", "SR", "M", "MH", "MR", "D", "DR",
    "AL", "ALR", "SL", "SLR",
    "AP", "SP", "MP", "DP", "ZAP", "CVB", "CVD",
    # Logical
    "N", "NI", "NC", "NR", "O", "OI", "OC", "OR", "X", "XI", "XC", "XR",
    # Compare
    "C", "CH", "CR", "CL", "CLI", "CLC", "CLR", "CLM", "CP",
    # Shift
    "SLL", "SRL", "SLA", "SRA", "SLDL", "SRDL",
    # Move
    "MVC", "MVI", "MVCL", "MVN", "MVZ", "MVO", "MVCP", "MVCS",
    # Branch
    "B", "BE", "BNE", "BZ", "BNZ", "BH", "BNH", "BL", "BNL",
    "BM", "BNM", "BO", "BNO", "BP", "BNP", "BC", "BCR", "BCT", "BCTR",
    "BAS", "BAL", "BASR", "BALR", "BR", "BRAS", "BRASL",
    "J", "JE", "JNE", "JZ", "JNZ", "JH", "JNH", "JL", "JNL",
    "JM", "JNM", "JO", "JNO", "JP", "JNP", "JXHG", "JXLE",
    # Test / translate
    "TM", "TRT", "TR", "PACK", "UNPK",
    # Misc
    "LPR", "LNR", "LCR", "SVC", "EX",
    # TPF-specific
    "ENTRC", "ENTNC", "ENTDC", "BACKC", "EXITC", "CREEC", "CREMC", "SWISC",
    "FINIS", "DETAC", "ATTAC", "GETCC", "RELCC", "CRUSA", "FLIPC",
    "LEVTA", "MALOC", "FREEC", "ACCTA", "CASIO", "FACE", "ALASC",
    "EVNWC", "ISCFA", "GENDB",
    "CALLCPP", "CALLC",
}


class VariableLineageBuilder:
    """Build variable lineage graphs from parsed sources."""

    def __init__(self):
        # Track contributing sources per register for better lineage fidelity
        self.register_sources = {}
        self.literal_resolver = LiteralResolver()
        self.using_map = {}  # {base_register: dsect_name}
        self.active_dsects = {}  # {dsect_name: base_register}
        self.equ_symbols = {}  # {symbol_name: resolved_value}
        self.pointer_aliases: Dict[str, Set[str]] = {}
        self._emitted_memory_aliases: Set[Tuple[str, str]] = set()
        self._legacy_register_label_candidates: Set[str] = set()
        self.context: Optional[AnalysisContext] = None  # Set during process_assembly
        # DSECT field offset map: {dsect_name: {field_name: (offset, length)}}
        self.dsect_field_offsets: Dict[str, Dict[str, tuple]] = {}
        self.dsect_to_ce_slot: Dict[str, str] = {}  # DSECT_NAME → CE1CRx

    def _get_enrichment(self, var_name: str, file: str, line: int) -> Dict[str, any]:
        """Get enrichment data from context cache for a variable at a specific location.

        Args:
            var_name: Variable name
            file: Source file path
            line: Line number

        Returns:
            Dictionary with enrichment fields (operation_type, etc.) or empty dict
        """
        if not self.context or not hasattr(self.context, '_variable_operation_cache'):
            return {}

        cache = self.context._variable_operation_cache
        key = f"{var_name}_{file}_{line}"
        return cache.get(key, {})

    def _normalize_register_token(self, token: str) -> Optional[str]:
        """Normalize a register token or EQU alias to canonical form like R12."""
        if token is None:
            return None
        text = token.strip().upper()
        if not text:
            return None

        # Resolve EQU aliases if present
        if text in self.equ_symbols:
            text = self._resolve_equ_value(text).strip().upper()

        if text.startswith("R") and text[1:].isdigit():
            return text
        if text.isdigit():
            return f"R{int(text)}"
        return None

    def _normalize_register_operand(self, token: str) -> Optional[str]:
        """Normalize an operand only when it is an explicit register or register alias."""
        if token is None:
            return None
        text = token.strip().rstrip(",").upper()
        if not text:
            return None

        if text.startswith("R") and text[1:].isdigit():
            return f"R{int(text[1:])}"

        if text in self.equ_symbols:
            resolved = self._resolve_equ_value(text).strip().upper()
            if resolved.startswith("R") and resolved[1:].isdigit():
                return f"R{int(resolved[1:])}"
            if resolved.isdigit():
                return f"R{int(resolved)}"

        return None

    def _looks_like_label_token(self, token: str) -> bool:
        """Return True when token is a plain symbol label (not literal/expression)."""
        if not token:
            return False
        text = token.strip().rstrip(",")
        if not text:
            return False
        if self._normalize_register_operand(text):
            return False
        if any(ch in text for ch in ("(", ")", "+", "-", "=", "'", '"', "[", "]")):
            return False
        return bool(re.fullmatch(r"[A-Za-z_.$@#][A-Za-z0-9_.$@#]*", text))

    def _is_symbolic_register_name(self, token: str) -> bool:
        """Detect labels that should get legacy register-node compatibility.

        This is context-driven: candidates are discovered from instruction
        usage (e.g., branch targets and LA sources), not hardcoded names.
        """
        if not token:
            return False
        text = token.strip().rstrip(",")
        if not text:
            return False
        if text in self._legacy_register_label_candidates:
            return True
        tail = text.split("::")[-1]
        return tail in self._legacy_register_label_candidates

    def _expand_register_range(self, start: str, end: str) -> List[str]:
        """
        Expand register range handling wraparound (R14->R15->R0->...->R12).

        Args:
            start: Starting register (e.g., "R6")
            end: Ending register (e.g., "R9")

        Returns:
            List of register names in sequence
        """
        # Normalize register tokens (supports EQU aliases like BASEREG EQU 12)
        start_reg = self._normalize_register_token(start)
        end_reg = self._normalize_register_token(end)
        if not start_reg or not end_reg:
            return []

        # Parse register numbers
        start_num = int(start_reg[1:])  # "R6" -> 6
        end_num = int(end_reg[1:])      # "R9" -> 9

        result = []
        current = start_num
        while True:
            result.append(f"R{current}")
            if current == end_num:
                break
            current = (current + 1) % 16
            if len(result) > 16:  # Safety check to prevent infinite loop
                break
        return result

    def _resolve_equ_value(self, value: str) -> str:
        """
        Recursively resolve EQU symbol references.

        Args:
            value: EQU value to resolve

        Returns:
            Resolved value
        """
        value = value.strip()

        # If it's a number, return as-is
        if value.isdigit():
            return value

        # If it's a register like R12, return as-is
        if value.upper().startswith("R") and len(value) > 1 and value[1:].isdigit():
            return value.upper()

        # If it references another EQU symbol, resolve it
        if value.upper() in self.equ_symbols:
            return self.equ_symbols[value.upper()]

        # Otherwise return as-is (might be expression like "*-CSECT")
        return value

    def _extract_symbol_from_address_literal(self, value: str) -> Optional[str]:
        """Extract target symbol from =V(symbol) or =A(symbol) literals."""
        cleaned = value.strip()
        upper = cleaned.upper()
        if upper.startswith("=V(") or upper.startswith("=A("):
            start = cleaned.find("(")
            end = cleaned.rfind(")")
            if start != -1 and end != -1 and end > start + 1:
                return cleaned[start + 1:end].strip()
        return None

    def _resolve_call_target(self, operand: str) -> str:
        """Resolve call target from direct operand or register-held function literal."""
        symbol, kind, _, _ = self._normalize_operand_full(operand)
        if kind == "register":
            for src, src_kind in self.register_sources.get(symbol, set()):
                if src_kind in ("function", "entry"):
                    return src
                extracted = self._extract_symbol_from_address_literal(src)
                if extracted:
                    return extracted
            return symbol
        extracted = self._extract_symbol_from_address_literal(symbol)
        if extracted:
            return extracted
        return symbol

    def _extract_entrc_target(self, operands: List[str]) -> Optional[str]:
        """Extract ENTRC/ENTNC target entry symbol from operands."""
        for operand in operands:
            candidate = operand.strip().rstrip(",")
            if not candidate:
                continue
            if "=" in candidate:
                _, value = candidate.split("=", 1)
                candidate = value.strip().rstrip(",")
            if not candidate or candidate.startswith("&"):
                continue
            return candidate
        return None

    def _extract_creec_target(self, operands: List[str]) -> Optional[str]:
        """Extract CREEC spawn target entry symbol from operands."""
        for operand in operands:
            if not operand:
                continue
            for segment in [part.strip() for part in operand.split(",") if part.strip()]:
                candidate = segment
                if "=" in candidate:
                    key, value = candidate.split("=", 1)
                    key = key.strip().upper().lstrip("&")
                    if key not in {"ENTRY", "SEG", "SEGMENT", "TARGET"}:
                        continue
                    candidate = value.strip()
                candidate = candidate.strip().rstrip(",")
                if not candidate or candidate.startswith("&"):
                    continue
                if re.fullmatch(r"R(?:1[0-5]|[0-9])", candidate, flags=re.IGNORECASE):
                    continue
                if re.fullmatch(r"[+-]?\d+", candidate):
                    continue
                if re.fullmatch(r"D[0-9A-Fa-f]", candidate, flags=re.IGNORECASE):
                    continue
                if candidate.upper() in {"R", "N", "Y", "YES", "NO"}:
                    continue
                return candidate
        return None

    def _extract_attac_target(self, operands: List[str]) -> Optional[str]:
        """Extract ATTAC attach target entry symbol from operands."""
        for operand in operands:
            if not operand:
                continue
            for segment in [part.strip() for part in operand.split(",") if part.strip()]:
                candidate = segment
                if "=" in candidate:
                    key, value = candidate.split("=", 1)
                    key = key.strip().upper().lstrip("&")
                    if key not in {"ENTRY", "SEG", "SEGMENT", "TARGET"}:
                        continue
                    candidate = value.strip()
                candidate = candidate.strip().rstrip(",")
                if not candidate or candidate.startswith("&"):
                    continue
                if re.fullmatch(r"R(?:1[0-5]|[0-9])", candidate, flags=re.IGNORECASE):
                    continue
                if re.fullmatch(r"[+-]?\d+", candidate):
                    continue
                if re.fullmatch(r"D[0-9A-Fa-f]", candidate, flags=re.IGNORECASE):
                    continue
                if candidate.upper() in {"R", "N", "Y", "YES", "NO"}:
                    continue
                return candidate
        return None

    def _extract_callcpp_target(self, operands: List[str]) -> Optional[str]:
        """Extract CALLCPP target C/C++ symbol from operands."""
        for operand in operands:
            if not operand:
                continue
            for segment in [part.strip() for part in operand.split(",") if part.strip()]:
                candidate = segment
                if "=" in candidate:
                    key, value = candidate.split("=", 1)
                    key = key.strip().upper().lstrip("&")
                    if key not in {"ENTRY", "TARGET", "FUNC", "FUNCTION", "NAME", "SEG", "SEGMENT"}:
                        continue
                    candidate = value.strip()
                candidate = candidate.strip().rstrip(",")
                if not candidate or candidate.startswith("&"):
                    continue
                if re.fullmatch(r"R(?:1[0-5]|[0-9])", candidate, flags=re.IGNORECASE):
                    continue
                if re.fullmatch(r"[+-]?\d+", candidate):
                    continue
                if re.fullmatch(r"D[0-9A-Fa-f]", candidate, flags=re.IGNORECASE):
                    continue
                if candidate.upper() in {"R", "N", "Y", "YES", "NO"}:
                    continue
                return candidate
        return None

    def _macro_operand_value(self, operand: str) -> str:
        """Return value side of KEY=VALUE macro operands, else original operand."""
        if not operand:
            return operand
        text = operand.strip()
        if "=" not in text:
            return text
        _, value = text.split("=", 1)
        return value.strip()

    def _split_macro_operand_segments(self, operands: List[str]) -> List[str]:
        """Split macro operand payload into comma-delimited segments."""
        segments: List[str] = []
        for operand in operands or []:
            if not operand:
                continue
            for segment in operand.split(","):
                cleaned = segment.strip().rstrip(",")
                if cleaned:
                    segments.append(cleaned)
        return segments

    def _parse_macro_level_token(self, token: str) -> Optional[int]:
        """Parse D-level/LEVEL tokens (D0..DF, LEVEL0..LEVELF, 0..15)."""
        if not token:
            return None
        cleaned = token.strip().rstrip(",")
        if not cleaned:
            return None
        if cleaned.startswith("&"):
            return None

        while cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1].strip()
            if not cleaned:
                return None

        upper = cleaned.upper()
        dmatch = re.fullmatch(r"D([0-9A-F])", upper)
        if dmatch:
            return int(dmatch.group(1), 16)

        lmatch = re.fullmatch(r"LEVEL_?([0-9A-F]{1,2})", upper)
        if lmatch:
            try:
                level = int(lmatch.group(1), 16)
            except ValueError:
                return None
            return level if 0 <= level <= 15 else None

        if re.fullmatch(r"[0-9]+", upper):
            level = int(upper, 10)
            return level if 0 <= level <= 15 else None

        return None

    def _extract_macro_level_number(self, operands: List[str]) -> Optional[int]:
        """Extract level number from macro operands such as LEVEL=0, DATA=LEVEL0, D4."""
        segments = self._split_macro_operand_segments(operands)
        for segment in segments:
            if "=" in segment:
                key, value = segment.split("=", 1)
                key = key.strip().upper().lstrip("&")
                value = value.strip()
                if key in {"LEVEL", "LVL", "DATA"}:
                    level = self._parse_macro_level_token(value)
                    if level is not None:
                        return level
            else:
                level = self._parse_macro_level_token(segment)
                if level is not None:
                    return level
        return None

    def _macro_level_name(self, level_number: int) -> str:
        """Normalize level-number identity to Dn naming."""
        return f"D{format(level_number, 'X')}"

    def _extract_macro_keyword_value(self, operands: List[str], keywords: Set[str]) -> Optional[str]:
        """Return first KEY=VALUE occurrence for the provided keyword set."""
        wanted = {keyword.upper() for keyword in keywords}
        for segment in self._split_macro_operand_segments(operands):
            if "=" not in segment:
                continue
            key, value = segment.split("=", 1)
            key = key.strip().upper().lstrip("&")
            value = value.strip().rstrip(",")
            if key in wanted and value and not value.startswith("&"):
                return value
        return None

    def _extract_pointer_transfer_operands(self, operands: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Best-effort source/target extraction for PNTRP/PNTRC macros."""
        source_keys = {"FROM", "SRC", "SOURCE", "IN", "INPUT", "OLD", "CUR", "PTRIN"}
        target_keys = {"TO", "DST", "DEST", "TARGET", "OUT", "OUTPUT", "NEW", "PTROUT"}
        skip_keys = {
            "LEVEL", "LVL", "DATA", "SIZE", "TAG", "MODE", "NOTUSED",
            "ENTRY", "SEG", "SEGMENT", "ECB", "R",
        }

        source = self._extract_macro_keyword_value(operands, source_keys)
        target = self._extract_macro_keyword_value(operands, target_keys)

        positional: List[str] = []
        for segment in self._split_macro_operand_segments(operands):
            candidate = segment
            if "=" in segment:
                key, value = segment.split("=", 1)
                key = key.strip().upper().lstrip("&")
                value = value.strip()
                if key in source_keys and not source:
                    source = value
                    continue
                if key in target_keys and not target:
                    target = value
                    continue
                if key in skip_keys:
                    continue
                candidate = value

            candidate = candidate.strip().rstrip(",")
            if not candidate or candidate.startswith("&"):
                continue
            if self._parse_macro_level_token(candidate) is not None:
                continue
            if candidate.upper() in {"R", "N", "Y", "YES", "NO"}:
                continue
            positional.append(candidate)

        if not source and positional:
            source = positional[0]
        if not target and len(positional) >= 2:
            target = positional[1]

        if not source or not target:
            return None, None
        if source.strip() == target.strip():
            return None, None
        return source.strip(), target.strip()

    def _iter_macro_arguments(self, operands: List[str]) -> List[Tuple[Optional[str], str]]:
        """Return normalized macro argument tuples as (KEY|None, VALUE)."""
        items: List[Tuple[Optional[str], str]] = []
        for segment in self._split_macro_operand_segments(operands):
            if "=" in segment:
                key, value = segment.split("=", 1)
                key_name = key.strip().upper().lstrip("&")
                value_text = value.strip()
                if value_text:
                    items.append((key_name, value_text))
                continue
            items.append((None, segment.strip()))
        return items

    def _derive_scope(self, line: ParsedLine, current_scope: Optional[str]) -> Optional[str]:
        """Derive best-effort scope from provenance and section/routine opcodes."""
        def _valid_scope_name(name: Optional[str]) -> bool:
            if not name:
                return False
            # Macro formal labels (e.g., &LABEL) should never become runtime scope IDs.
            if name.startswith("&"):
                return False
            return True

        routine = line.provenance.routine
        section = line.provenance.section
        if _valid_scope_name(routine) or _valid_scope_name(section):
            return routine if _valid_scope_name(routine) else section
        opcode = (line.opcode or "").upper()
        if opcode in ("CSECT", "DSECT", "RSECT") and _valid_scope_name(line.label):
            return line.label
        return current_scope

    def _r1_parameter_symbol(self, displacement: str, base_reg: Optional[str], index_reg: Optional[str]) -> Optional[str]:
        """Resolve conventional R1 parameter-list offsets to semantic caller-parameter names."""
        if base_reg != "R1" or index_reg is not None:
            return None
        try:
            offset = int(displacement, 10)
        except (TypeError, ValueError):
            return None
        if offset < 0:
            return None
        if offset % 4 == 0:
            return f"caller_param[{offset // 4}]"
        return f"caller_param_offset[{offset}]"

    def _parse_storage_length(self, operand: str) -> int:
        """Best-effort byte-size parser for DS/DC operands."""
        token = (operand or "").strip().upper()
        if not token:
            return 0
        quote_idx = token.find("'")
        if quote_idx != -1:
            token = token[:quote_idx]
        token = token.strip()
        if not token:
            return 0

        match = re.match(r"^(\d+)?([A-Z]{1,2})(\d+)?$", token)
        if not match:
            return 0
        count_str, dtype, width_str = match.groups()
        count = int(count_str) if count_str else 1
        width = int(width_str) if width_str else None

        if dtype in ("CL", "XL", "PL") and width is not None:
            return count * width
        if dtype in ("C", "X", "P"):
            return count
        if dtype == "H":
            return count * 2
        if dtype in ("F", "A", "V"):
            return count * 4
        if dtype == "D":
            return count * 8
        return 0

    def _extract_quoted_literal(self, value: str) -> Optional[str]:
        text = (value or "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
            return text
        return None

    def process_assembly(self, lines: List[ParsedLine], context: AnalysisContext) -> None:
        _t0 = time.perf_counter()
        _log.debug("VariableLineageBuilder.process_assembly start: %s", context.file_path)
        # Store context for enrichment data access
        self.context = context

        graph = context.get_metadata("variable_lineage")
        if graph is None:
            graph = VariableLineageGraph()
            context.add_metadata("variable_lineage", graph)
        self._emitted_memory_aliases.clear()
        self._legacy_register_label_candidates.clear()

        # First pass: collect EQU definitions
        for line in lines:
            if line.opcode and line.opcode.upper() == "EQU" and line.label:
                symbol = line.label.upper()
                if line.operands:
                    value = line.operands[0]
                    # Resolve nested EQU references
                    resolved = self._resolve_equ_value(value)
                    self.equ_symbols[symbol] = resolved

        # Second pass: process instructions
        current_scope = None
        current_dsect: Optional[str] = None
        dsect_offsets: Dict[str, int] = {}
        # Per-line register→(dsect, CE-slot) snapshot for DSECT context enrichment.
        # Snapshots are taken AFTER instruction processing so register_sources is current.
        _line_reg_ce_snapshots: Dict[tuple, Dict[str, Optional[str]]] = {}
        _edge_count_at_start = len(graph.edges)
        _prev_file: Optional[str] = None
        _prev_line_no: int = 0

        # Register entry state: track which registers were written before their
        # first USING directive in this file.
        # inherited_regs  — USING base registers NOT written before USING (caller provided)
        # self_loaded_regs — USING base registers written before USING (callee loaded own value)
        _regs_written_so_far: Set[str] = set()
        _using_established_regs: Set[str] = set()
        _inherited_regs_list: List[str] = []
        _self_loaded_regs_list: List[str] = []

        def _take_reg_ce_snapshot(snap_file: str, snap_line: int):
            """Capture current register→(dsect, CE-slot) bindings for a line."""
            if not self.using_map:
                return
            reg_ce = {}
            for reg, dsect_name in self.using_map.items():
                ce = self._resolve_ce_slot_for_register(reg)
                reg_ce[reg] = (dsect_name, ce)
            if reg_ce:
                _line_reg_ce_snapshots[(snap_file, snap_line)] = reg_ce

        for line in lines:
            if not line.opcode:
                continue

            opcode = line.opcode.upper()
            file = line.provenance.original_file
            line_no = line.provenance.original_line
            scope = self._derive_scope(line, current_scope)
            current_scope = scope
            statement_id = line.provenance.statement_id
            macro_parent = line.provenance.macro_expansion_parent
            operands = line.operands

            if opcode == "DSECT" and line.label and not line.label.startswith("&"):
                current_dsect = line.label
                dsect_offsets.setdefault(current_dsect, 0)
                self.dsect_field_offsets.setdefault(current_dsect, {})
            elif opcode in ("CSECT", "RSECT"):
                current_dsect = None

            # Handle ORG directive — resets or adjusts offset within a DSECT
            if opcode == "ORG" and current_dsect:
                if operands:
                    # ORG LABEL — reset to the offset of a previously defined label
                    target_label = operands[0].strip().rstrip(",").upper()
                    dsect_fields = self.dsect_field_offsets.get(current_dsect, {})
                    if target_label in dsect_fields:
                        dsect_offsets[current_dsect] = dsect_fields[target_label][0]
                else:
                    # ORG with no operand — reset to high-water mark (end of DSECT)
                    # Approximate: use the max (offset + length) seen so far
                    dsect_fields = self.dsect_field_offsets.get(current_dsect, {})
                    if dsect_fields:
                        dsect_offsets[current_dsect] = max(
                            off + length for off, length in dsect_fields.values()
                        )

            if opcode in ("DS", "DC"):
                storage_len = sum(self._parse_storage_length(op) for op in operands)
                if line.label and not line.label.startswith("&"):
                    current_offset = dsect_offsets.get(current_dsect, 0) if current_dsect else 0
                    # Get enrichment data for this variable
                    enrichment = self._get_enrichment(line.label, file, line_no)
                    graph.add_definition(line.label, "memory", file, line_no, opcode,
                                         scope, statement_id, macro_parent,
                                         operation_type=enrichment.get("operation_type", "initialize"),
                                         initialization_method=enrichment.get("initialization_method", "direct"),
                                         conditional_context=enrichment.get("conditional_context"),
                                         transformation=enrichment.get("transformation"))
                    if current_dsect:
                        graph.annotate_node(
                            line.label,
                            "memory",
                            byte_offset=current_offset,
                            cpp_struct=current_dsect,
                        )
                        # Record in dsect_field_offsets for cross-file identity
                        self.dsect_field_offsets.setdefault(current_dsect, {})[line.label] = (current_offset, storage_len)
                if opcode == "DC":
                    for raw in operands:
                        literal = self._extract_quoted_literal(raw[raw.find("'"):]) if "'" in raw else self._extract_quoted_literal(raw)
                        if not literal:
                            literal = self._extract_quoted_literal(raw)
                        if not literal:
                            continue
                        enrichment = self._get_enrichment(literal, file, line_no)
                        graph.add_definition(literal, "literal", file, line_no, "DC",
                                             scope, statement_id, macro_parent,
                                             operation_type=enrichment.get("operation_type", "initialize"),
                                             initialization_method=enrichment.get("initialization_method", "direct"),
                                             conditional_context=enrichment.get("conditional_context"),
                                             transformation=enrichment.get("transformation"))
                        if line.label and not line.label.startswith("&"):
                            enrichment = self._get_enrichment(line.label, file, line_no)
                            graph.add_assignment(literal, "literal", line.label, "memory",
                                                 file, line_no, "DC",
                                                 scope, statement_id, macro_parent,
                                                 operation_type=enrichment.get("operation_type", "initialize"),
                                                 initialization_method=enrichment.get("initialization_method", "direct"),
                                                 conditional_context=enrichment.get("conditional_context"),
                                                 transformation=enrichment.get("transformation"))
                if current_dsect and storage_len > 0:
                    dsect_offsets[current_dsect] = dsect_offsets.get(current_dsect, 0) + storage_len
                continue

            # Recognize DSECT registration macros: LG1G1 REG=R1, WB1WB REG=R4, etc.
            # Pattern: unknown opcode (not a standard instruction) + REG=Rn operand
            # Treat as USING <opcode>,Rn — the macro internally generates this.
            if (opcode not in _KNOWN_OPCODES
                    and operands
                    and not opcode.startswith("&")):
                reg_match = None
                for op in operands:
                    stripped = op.strip().rstrip(",").upper()
                    if stripped.startswith("REG="):
                        reg_token = stripped[4:]
                        reg_match = self._normalize_register_token(reg_token)
                        break
                if reg_match:
                    dsect_name = opcode  # opcode IS the DSECT name
                    self.using_map[reg_match] = dsect_name
                    self.active_dsects[dsect_name] = reg_match
                    if reg_match not in _using_established_regs:
                        _using_established_regs.add(reg_match)
                        if reg_match in _regs_written_so_far:
                            _self_loaded_regs_list.append(reg_match)
                        else:
                            _inherited_regs_list.append(reg_match)
                    enrichment = self._get_enrichment(reg_match, file, line_no)
                    graph.add_assignment(dsect_name, "memory", reg_match, "register",
                                         file, line_no, f"{opcode} {' '.join(operands)}",
                                         scope, statement_id, macro_parent,
                                         operation_type=enrichment.get("operation_type"),
                                         initialization_method=enrichment.get("initialization_method"),
                                         conditional_context=enrichment.get("conditional_context"),
                                         transformation=enrichment.get("transformation"))
                    graph.add_definition(reg_match, "register", file, line_no, f"USING({opcode})",
                                         scope, statement_id, macro_parent,
                                         operation_type=enrichment.get("operation_type"),
                                         initialization_method=enrichment.get("initialization_method"),
                                         conditional_context=enrichment.get("conditional_context"),
                                         transformation=enrichment.get("transformation"))
                    _take_reg_ce_snapshot(file, line_no)
                    continue

            # Handle USING directive
            if opcode == "USING" and len(operands) >= 2:
                dsect_or_location = operands[0]
                base_registers = operands[1:]  # Can have multiple base regs

                for base_reg in base_registers:
                    if base_reg.startswith("R") or base_reg.startswith("r"):
                        reg = base_reg.upper()
                        self.using_map[reg] = dsect_or_location
                        self.active_dsects[dsect_or_location] = reg
                        if reg not in _using_established_regs:
                            _using_established_regs.add(reg)
                            if reg in _regs_written_so_far:
                                _self_loaded_regs_list.append(reg)
                            else:
                                _inherited_regs_list.append(reg)
                        # Record DSECT→CE-slot mapping when CE-slot is resolvable
                        _ce = self._resolve_ce_slot_for_register(reg)
                        if _ce and dsect_or_location not in ("*", ""):
                            self.dsect_to_ce_slot.setdefault(dsect_or_location, _ce)
                        source_kind = "memory"
                        source_name = dsect_or_location
                        if dsect_or_location == "*":
                            source_kind = "literal"
                            source_name = "*"
                        enrichment = self._get_enrichment(reg, file, line_no)
                        graph.add_assignment(source_name, source_kind, reg, "register",
                                             file, line_no, " ".join(operands),
                                             scope, statement_id, macro_parent,
                                             operation_type=enrichment.get("operation_type"),
                                             initialization_method=enrichment.get("initialization_method"),
                                             conditional_context=enrichment.get("conditional_context"),
                                             transformation=enrichment.get("transformation"))
                        graph.add_definition(reg, "register", file, line_no, opcode,
                                             scope, statement_id, macro_parent,
                                             operation_type=enrichment.get("operation_type"),
                                             initialization_method=enrichment.get("initialization_method"),
                                             conditional_context=enrichment.get("conditional_context"),
                                             transformation=enrichment.get("transformation"))
                _take_reg_ce_snapshot(file, line_no)
                continue

            # Handle DROP directive
            if opcode == "DROP" and len(operands) >= 1:
                for operand in operands:
                    if operand.startswith("R") or operand.startswith("r"):
                        reg = operand.upper()
                        if reg in self.using_map:
                            dsect = self.using_map[reg]
                            del self.using_map[reg]
                            if dsect in self.active_dsects and self.active_dsects[dsect] == reg:
                                del self.active_dsects[dsect]
                _take_reg_ce_snapshot(file, line_no)
                continue

            # Register definitions (loads / moves)
            if opcode in ("L", "LR", "LTR") and len(operands) >= 2:
                target = self._normalize_register_operand(operands[0]) or operands[0].upper()
                source_raw, source_kind, index_reg, base_reg = self._normalize_operand_full(operands[1])
                source = self._qualify_memory_access(source_raw, source_kind, index_reg, base_reg)
                self._emit_equ_memory_aliases(
                    graph, operands[1], source_raw, source_kind, index_reg, base_reg, source,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )

                # Track index and base register usage
                self._add_index_base_usage(index_reg, base_reg, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=source if source_kind == "memory" else None)

                # When loading from a register, preserve the source register's lineage.
                register_sources = {(source, source_kind)}
                if source_kind == "register":
                    register_sources.update(self.register_sources.get(source, {(source, "register")}))

                external_symbol = self._extract_symbol_from_address_literal(source) if source_kind == "literal" else None
                # Get enrichment for the target register
                enrichment = self._get_enrichment(target, file, line_no)
                # Also check if source is a variable and get its enrichment
                if source_kind == "memory":
                    source_enrichment = self._get_enrichment(source, file, line_no)
                    # Prefer source enrichment if available
                    if source_enrichment:
                        enrichment = source_enrichment

                if external_symbol:
                    register_sources.add((external_symbol, "function"))
                    graph.add_assignment(external_symbol, "function", target, "register", file, line_no,
                                         " ".join(operands), scope, statement_id, macro_parent,
                                         operation_type=enrichment.get("operation_type", "load"),
                                         initialization_method=enrichment.get("initialization_method"),
                                         conditional_context=enrichment.get("conditional_context"),
                                         transformation=enrichment.get("transformation", f"Load {external_symbol} into {target}"))
                self.register_sources[target] = register_sources
                # Track write for register_entry_state (before USING check)
                _regs_written_so_far.add(target)
                for src, src_kind in register_sources:
                    graph.add_assignment(src, src_kind, target, "register", file, line_no, " ".join(operands),
                                         scope, statement_id, macro_parent,
                                         operation_type=enrichment.get("operation_type", "load"),
                                         initialization_method=enrichment.get("initialization_method"),
                                         conditional_context=enrichment.get("conditional_context"),
                                         transformation=enrichment.get("transformation", f"Load {src} into {target}"))
                graph.add_definition(target, "register", file, line_no, opcode, scope, statement_id, macro_parent,
                                     operation_type=enrichment.get("operation_type", "load"),
                                     initialization_method=enrichment.get("initialization_method"),
                                     conditional_context=enrichment.get("conditional_context"),
                                     transformation=enrichment.get("transformation", f"Define {target} via {opcode}"))
                # Snapshot after register load in case it affects a USING base register
                if target in self.using_map:
                    _take_reg_ce_snapshot(file, line_no)

            # Load Address (LA): computes effective address, not memory contents.
            if opcode == "LA" and len(operands) >= 2:
                target = self._normalize_register_operand(operands[0]) or operands[0].upper()
                raw_source, source_kind, index_reg, base_reg = self._normalize_operand_full(operands[1])
                if source_kind == "memory" and self._looks_like_label_token(raw_source):
                    self._legacy_register_label_candidates.add(raw_source.strip().rstrip(","))
                source = self._qualify_memory_access(raw_source, source_kind, index_reg, base_reg)
                self._emit_equ_memory_aliases(
                    graph, operands[1], raw_source, source_kind, index_reg, base_reg, source,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )

                # LA contributes address registers (base/index) and optional displacement/symbol.
                self._add_index_base_usage(index_reg, base_reg, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=None)

                register_sources = set()

                if index_reg or base_reg:
                    for reg in (index_reg, base_reg):
                        if not reg or not reg.startswith("R"):
                            continue
                        register_sources.add((reg, "register"))
                        register_sources.update(self.register_sources.get(reg, {(reg, "register")}))

                    # Non-zero displacement/symbol contributes to effective address.
                    if raw_source and raw_source != "0":
                        disp_kind = source_kind
                        if source_kind == "memory" and re.fullmatch(r"[+-]?\d+", raw_source):
                            disp_kind = "literal"
                        register_sources.add((raw_source, disp_kind))
                else:
                    source_for_la = source
                    source_kind_for_la = source_kind
                    if source_kind_for_la == "memory" and re.fullmatch(r"[+-]?\d+", source_for_la):
                        source_kind_for_la = "literal"

                    register_sources.add((source_for_la, source_kind_for_la))
                    external_symbol = (
                        self._extract_symbol_from_address_literal(source_for_la)
                        if source_kind_for_la == "literal"
                        else None
                    )
                    if external_symbol:
                        register_sources.add((external_symbol, "function"))
                        enrichment = self._get_enrichment(target, file, line_no)
                        graph.add_assignment(external_symbol, "function", target, "register", file, line_no,
                                             " ".join(operands), scope, statement_id, macro_parent,
                                             operation_type=enrichment.get("operation_type", "load"),
                                             initialization_method=enrichment.get("initialization_method"),
                                             conditional_context=enrichment.get("conditional_context"),
                                             transformation=enrichment.get("transformation", f"Load address of {external_symbol} into {target}"))

                if not register_sources:
                    register_sources.add((target, "register"))

                enrichment = self._get_enrichment(target, file, line_no)
                self.register_sources[target] = register_sources
                # Track write for register_entry_state (before USING check)
                _regs_written_so_far.add(target)
                for src, src_kind in register_sources:
                    graph.add_assignment(src, src_kind, target, "register", file, line_no, " ".join(operands),
                                         scope, statement_id, macro_parent,
                                         operation_type=enrichment.get("operation_type", "load"),
                                         initialization_method=enrichment.get("initialization_method"),
                                         conditional_context=enrichment.get("conditional_context"),
                                         transformation=enrichment.get("transformation", f"Load address into {target}"))
                graph.add_definition(target, "register", file, line_no, opcode, scope, statement_id, macro_parent,
                                     operation_type=enrichment.get("operation_type", "load"),
                                     initialization_method=enrichment.get("initialization_method"),
                                     conditional_context=enrichment.get("conditional_context"),
                                     transformation=enrichment.get("transformation", f"Define {target} via LA"))

            # Load Multiple (LM)
            if opcode in ("LM", "LTM") and len(operands) >= 3:
                start_reg = self._normalize_register_token(operands[0])
                end_reg = self._normalize_register_token(operands[1])
                base_addr, _ = self._normalize_operand(operands[2])

                # Calculate register sequence (handles wraparound)
                if not start_reg or not end_reg:
                    continue
                reg_sequence = self._expand_register_range(start_reg, end_reg)

                for offset, reg in enumerate(reg_sequence):
                    memory_loc = f"{base_addr}+{offset*4}"  # 4-byte words
                    self.register_sources[reg] = {(memory_loc, "memory")}
                    # Track write for register_entry_state (before USING check)
                    _regs_written_so_far.add(reg)
                    # Get enrichment for memory location
                    enrichment = self._get_enrichment(memory_loc, file, line_no)
                    graph.add_assignment(memory_loc, "memory", reg, "register",
                                       file, line_no, " ".join(operands),
                                       scope, statement_id, macro_parent,
                                       operation_type=enrichment.get("operation_type", "load"),
                                       initialization_method=enrichment.get("initialization_method"),
                                       conditional_context=enrichment.get("conditional_context"),
                                       transformation=enrichment.get("transformation", f"Load multiple: {memory_loc} to {reg}"))
                    graph.add_definition(reg, "register", file, line_no, opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=enrichment.get("operation_type", "load"),
                                       initialization_method=enrichment.get("initialization_method"),
                                       conditional_context=enrichment.get("conditional_context"),
                                       transformation=enrichment.get("transformation", f"Define {reg} via LM"))
                # Snapshot after LM if any loaded register is a USING base
                if any(r in self.using_map for r in reg_sequence):
                    _take_reg_ce_snapshot(file, line_no)

            # Store Multiple (STM)
            if opcode in ("STM", "STMH") and len(operands) >= 3:
                start_reg = self._normalize_register_token(operands[0])
                end_reg = self._normalize_register_token(operands[1])
                base_addr, _ = self._normalize_operand(operands[2])

                if not start_reg or not end_reg:
                    continue
                reg_sequence = self._expand_register_range(start_reg, end_reg)

                for offset, reg in enumerate(reg_sequence):
                    memory_loc = f"{base_addr}+{offset*4}"
                    # Get enrichment for memory location
                    enrichment = self._get_enrichment(memory_loc, file, line_no)
                    # Store register to memory
                    graph.add_assignment(reg, "register", memory_loc, "memory",
                                       file, line_no, " ".join(operands),
                                       scope, statement_id, macro_parent,
                                       operation_type=enrichment.get("operation_type", "store"),
                                       initialization_method=enrichment.get("initialization_method"),
                                       conditional_context=enrichment.get("conditional_context"),
                                       transformation=enrichment.get("transformation", f"Store multiple: {reg} to {memory_loc}"))
                    # Propagate contributing sources like ST does
                    for src, src_kind in self.register_sources.get(reg, set()):
                        graph.add_assignment(src, src_kind, memory_loc, "memory",
                                           file, line_no, " ".join(operands),
                                           scope, statement_id, macro_parent,
                                           operation_type=enrichment.get("operation_type", "store"),
                                           initialization_method=enrichment.get("initialization_method"),
                                           conditional_context=enrichment.get("conditional_context"),
                                           transformation=enrichment.get("transformation", f"Store {src} to {memory_loc}"))

            # Stores / memory writes
            if opcode in ("ST", "STH") and len(operands) >= 2:
                source_reg = operands[0].upper()
                target_raw, target_kind, index_reg, base_reg = self._normalize_operand_full(operands[1])
                target = self._qualify_memory_access(target_raw, target_kind, index_reg, base_reg)
                self._emit_equ_memory_aliases(
                    graph, operands[1], target_raw, target_kind, index_reg, base_reg, target,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )

                # Track index and base register usage
                self._add_index_base_usage(index_reg, base_reg, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=target)

                # Always record the register write, then propagate contributing sources
                # Get enrichment for the target variable (if it's in universe)
                enrichment = self._get_enrichment(target if target_kind == "memory" else source_reg, file, line_no)
                graph.add_assignment(source_reg, "register", target, "memory", file, line_no, " ".join(operands),
                                     scope, statement_id, macro_parent,
                                     operation_type=enrichment.get("operation_type", "store"),
                                     initialization_method=enrichment.get("initialization_method"),
                                     conditional_context=enrichment.get("conditional_context"),
                                     transformation=enrichment.get("transformation", f"Store {source_reg} to {target}"))
                for src, src_kind in self.register_sources.get(source_reg, set()):
                    graph.add_assignment(src, src_kind, target, "memory", file, line_no, " ".join(operands),
                                         scope, statement_id, macro_parent,
                                         operation_type=enrichment.get("operation_type", "store"),
                                         initialization_method=enrichment.get("initialization_method"),
                                         conditional_context=enrichment.get("conditional_context"),
                                         transformation=enrichment.get("transformation", f"Store {src} to {target}"))

            if opcode == "MVC" and len(operands) >= 2:
                target_raw, target_kind, target_index, target_base = self._normalize_operand_full(operands[0])
                source_raw, source_kind, source_index, source_base = self._normalize_operand_full(operands[1])
                target = self._qualify_memory_access(target_raw, target_kind, target_index, target_base)
                source = self._qualify_memory_access(source_raw, source_kind, source_index, source_base)
                self._emit_equ_memory_aliases(
                    graph, operands[0], target_raw, target_kind, target_index, target_base, target,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )
                self._emit_equ_memory_aliases(
                    graph, operands[1], source_raw, source_kind, source_index, source_base, source,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )
                # Track address contributors for both destination and source operands.
                self._add_index_base_usage(target_index, target_base, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=target if target_kind == "memory" else None)
                self._add_index_base_usage(source_index, source_base, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=source if source_kind == "memory" else None)
                # Get enrichment for MVC target variable
                enrichment = self._get_enrichment(target if target_kind == "memory" else source, file, line_no)
                graph.add_assignment(source, source_kind, target, target_kind, file, line_no, " ".join(operands),
                                     scope, statement_id, macro_parent,
                                     operation_type=enrichment.get("operation_type", "move"),
                                     initialization_method=enrichment.get("initialization_method"),
                                     conditional_context=enrichment.get("conditional_context"),
                                     transformation=enrichment.get("transformation", f"Move {source} to {target}"))

            # SI-format logical immediate operations: OI, NI, XI
            # Format: OI D1(B1),I2 — memory byte is read and modified (OR/AND/XOR with immediate)
            if opcode in ("OI", "NI", "XI") and len(operands) >= 2:
                target_raw, target_kind, target_index, target_base = self._normalize_operand_full(operands[0])
                target = self._qualify_memory_access(target_raw, target_kind, target_index, target_base)
                self._emit_equ_memory_aliases(
                    graph, operands[0], target_raw, target_kind, target_index, target_base, target,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )
                self._add_index_base_usage(target_index, target_base, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=target if target_kind == "memory" else None)
                enrichment = self._get_enrichment(target if target_kind == "memory" else target_raw, file, line_no)
                # Memory is read (use) and modified (assign from immediate literal)
                graph.add_usage(target, target_kind, file, line_no, opcode,
                              scope, statement_id, macro_parent,
                              operation_type=enrichment.get("operation_type", "logical"),
                              initialization_method=enrichment.get("initialization_method"),
                              conditional_context=enrichment.get("conditional_context"),
                              transformation=enrichment.get("transformation"))
                imm_value = operands[1].strip()
                graph.add_assignment(imm_value, "literal", target, target_kind, file, line_no, " ".join(operands),
                                     scope, statement_id, macro_parent,
                                     operation_type=enrichment.get("operation_type", "logical"),
                                     initialization_method=enrichment.get("initialization_method"),
                                     conditional_context=enrichment.get("conditional_context"),
                                     transformation=enrichment.get("transformation", f"{opcode} {imm_value} into {target}"))

            # SS-format logical memory-to-memory operations: OC, NC, XC
            # Format: OC D1(L,B1),D2(B2) — target memory is modified using source memory
            if opcode in ("OC", "NC", "XC") and len(operands) >= 2:
                target_raw, target_kind, target_index, target_base = self._normalize_operand_full(operands[0])
                source_raw, source_kind, source_index, source_base = self._normalize_operand_full(operands[1])
                target = self._qualify_memory_access(target_raw, target_kind, target_index, target_base)
                source = self._qualify_memory_access(source_raw, source_kind, source_index, source_base)
                self._emit_equ_memory_aliases(
                    graph, operands[0], target_raw, target_kind, target_index, target_base, target,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )
                self._emit_equ_memory_aliases(
                    graph, operands[1], source_raw, source_kind, source_index, source_base, source,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )
                self._add_index_base_usage(target_index, target_base, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=target if target_kind == "memory" else None)
                self._add_index_base_usage(source_index, source_base, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=source if source_kind == "memory" else None)
                enrichment = self._get_enrichment(target if target_kind == "memory" else source, file, line_no)
                graph.add_assignment(source, source_kind, target, target_kind, file, line_no, " ".join(operands),
                                     scope, statement_id, macro_parent,
                                     operation_type=enrichment.get("operation_type", "logical"),
                                     initialization_method=enrichment.get("initialization_method"),
                                     conditional_context=enrichment.get("conditional_context"),
                                     transformation=enrichment.get("transformation", f"{opcode} {source} into {target}"))

            # z/TPF data-level movements
            if opcode == "FLIPC" and len(operands) >= 2:
                src_level = f"level:{operands[0].strip().upper()}"
                dst_level = f"level:{operands[1].strip().upper()}"
                # Get enrichment for source or destination
                enrichment = self._get_enrichment(dst_level, file, line_no)
                graph.add_assignment(src_level, "memory", dst_level, "memory",
                                     file, line_no, " ".join(operands),
                                     scope, statement_id, macro_parent,
                                     operation_type=enrichment.get("operation_type", "move"),
                                     initialization_method=enrichment.get("initialization_method"),
                                     conditional_context=enrichment.get("conditional_context"),
                                     transformation=enrichment.get("transformation", f"FLIPC {src_level} to {dst_level}"))
                graph.add_definition(dst_level, "memory", file, line_no, opcode,
                                     scope, statement_id, macro_parent,
                                     operation_type=enrichment.get("operation_type", "move"),
                                     initialization_method=enrichment.get("initialization_method"),
                                     conditional_context=enrichment.get("conditional_context"),
                                     transformation=enrichment.get("transformation", f"Define {dst_level} via FLIPC"))
                graph.add_usage(src_level, "memory", file, line_no, opcode,
                                scope, statement_id, macro_parent,
                                operation_type=enrichment.get("operation_type", "move"),
                                initialization_method=enrichment.get("initialization_method"),
                                conditional_context=enrichment.get("conditional_context"),
                                transformation=enrichment.get("transformation"))
                # Emit level_field_bridge edges so field-level tracers can cross the FLIPC
                # boundary even when the source and destination levels use different DSECT names.
                # Build a reverse map: level_name (e.g. "D9") → [dsect_names] via dsect_to_ce_slot.
                src_level_name = operands[0].strip().upper()
                dst_level_name = operands[1].strip().upper()
                level_to_dsects = {}
                for _dsect, _ce_slot in self.dsect_to_ce_slot.items():
                    _cs = _ce_slot.upper()
                    if _cs.startswith("CE1CR"):
                        try:
                            _lv = f"D{format(int(_cs[5:], 16), 'X')}"
                            level_to_dsects.setdefault(_lv, []).append(_dsect)
                        except ValueError:
                            pass
                # FLIPC is a swap: emit bridges from BOTH levels' DSECTs so the
                # tracer can cross the boundary regardless of which direction it
                # approaches from.
                for _lv_a, _lv_b in [
                    (dst_level_name, src_level_name),  # fields on D9 bridged via D0
                    (src_level_name, dst_level_name),  # fields on D0 bridged via D9
                ]:
                    for _dsect_a in level_to_dsects.get(_lv_a, []):
                        for _dsect_b in level_to_dsects.get(_lv_b, []):
                            if _dsect_a == _dsect_b:
                                continue
                            for _field in self.dsect_field_offsets.get(_dsect_a, {}):
                                graph.add_bridge(_field, "memory", _field, "memory",
                                                 file, line_no,
                                                 relation="level_field_bridge",
                                                 expression=opcode,
                                                 scope=scope,
                                                 statement_id=statement_id,
                                                 macro_expansion_parent=macro_parent)

            # Arithmetic/logical operations
            if opcode in ("AR", "A", "AH", "SR", "S", "SH", "MR", "M", "MH", "DR", "D", "NR", "N", "OR", "O", "XR", "X"):
                if len(operands) >= 2:
                    target = self._normalize_register_operand(operands[0]) or operands[0].upper()
                    source_reg = self._normalize_register_operand(operands[1])
                    # Include prior target sources (since arithmetic updates target)
                    new_sources = set(self.register_sources.get(target, set()))
                    # Always include self-dependency for in-place register updates.
                    new_sources.add((target, "register"))
                    # Include sources contributed by second operand
                    if source_reg:
                        new_sources.add((source_reg, "register"))
                        new_sources.update(self.register_sources.get(source_reg, {(source_reg, "register")}))
                    else:
                        src, src_kind = self._normalize_operand(operands[1])
                        if src_kind == "memory" and re.fullmatch(r"[+-]?\d+", src):
                            src_kind = "literal"
                        new_sources.add((src, src_kind))

                    # Determine operation type
                    if opcode in ("AR", "A", "AH", "SR", "S", "SH", "MR", "M", "MH", "DR", "D"):
                        op_type = "arithmetic"
                    else:
                        op_type = "logical"

                    # Get enrichment for target register
                    enrichment = self._get_enrichment(target, file, line_no)

                    # Record assignments for each contributing source
                    for src, src_kind in new_sources:
                        graph.add_assignment(src, src_kind, target, "register", file, line_no, " ".join(operands),
                                             scope, statement_id, macro_parent,
                                             operation_type=enrichment.get("operation_type", op_type),
                                             initialization_method=enrichment.get("initialization_method"),
                                             conditional_context=enrichment.get("conditional_context"),
                                             transformation=enrichment.get("transformation", f"{opcode}: {src} -> {target}"))
                    self.register_sources[target] = new_sources
                    graph.add_definition(target, "register", file, line_no, opcode, scope, statement_id, macro_parent,
                                       operation_type=enrichment.get("operation_type", op_type),
                                       initialization_method=enrichment.get("initialization_method"),
                                       conditional_context=enrichment.get("conditional_context"),
                                       transformation=enrichment.get("transformation", f"Define {target} via {opcode}"))

            # Shift operations
            SHIFT_OPS = ("SRL", "SLL", "SRA", "SLA", "SRDL", "SLDL", "SRDA", "SLDA")
            if opcode in SHIFT_OPS and len(operands) >= 2:
                target = self._normalize_register_operand(operands[0]) or operands[0].upper()
                shift_amount = operands[1]

                # For double shifts (SRDL, SLDL, SRDA, SLDA), operate on register pair
                if opcode in ("SRDL", "SLDL", "SRDA", "SLDA") and target.startswith("R") and target[1:].isdigit():
                    # Even register only
                    target_num = int(target[1:])
                    if target_num % 2 != 0:
                        # Invalid - skip
                        continue
                    targets = [target, f"R{target_num + 1}"]
                else:
                    targets = [target]

                # Preserve existing sources (shift modifies in-place)
                for tgt in targets:
                    new_sources = set(self.register_sources.get(tgt, {(tgt, "register")}))

                    # If shift amount is register, it contributes to the result
                    if shift_amount.startswith("R") or shift_amount.startswith("r"):
                        shift_reg = shift_amount.upper()
                        new_sources.update(self.register_sources.get(shift_reg, {(shift_reg, "register")}))

                    # Get enrichment for target register
                    enrichment = self._get_enrichment(tgt, file, line_no)

                    # Record assignments for all contributing sources
                    for src, src_kind in new_sources:
                        graph.add_assignment(src, src_kind, tgt, "register",
                                           file, line_no, " ".join(operands),
                                           scope, statement_id, macro_parent,
                                           operation_type=enrichment.get("operation_type", "shift"),
                                           initialization_method=enrichment.get("initialization_method"),
                                           conditional_context=enrichment.get("conditional_context"),
                                           transformation=enrichment.get("transformation", f"{opcode}: shift {tgt} by {shift_amount}"))

                    self.register_sources[tgt] = new_sources
                    graph.add_definition(tgt, "register", file, line_no, opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=enrichment.get("operation_type", "shift"),
                                       initialization_method=enrichment.get("initialization_method"),
                                       conditional_context=enrichment.get("conditional_context"),
                                       transformation=enrichment.get("transformation", f"Define {tgt} via {opcode}"))

            # Branch-on-count instructions decrement counter register in place.
            if opcode in ("BCT", "BCTR", "BCTG", "BCTGR") and len(operands) >= 1:
                counter = self._normalize_register_operand(operands[0]) or operands[0].upper()
                if counter.startswith("R"):
                    new_sources = set(self.register_sources.get(counter, set()))
                    new_sources.add((counter, "register"))
                    new_sources.add(("-1", "literal"))

                    # Get enrichment for counter register
                    enrichment = self._get_enrichment(counter, file, line_no)

                    for src, src_kind in new_sources:
                        graph.add_assignment(src, src_kind, counter, "register",
                                           file, line_no, " ".join(operands),
                                           scope, statement_id, macro_parent,
                                           operation_type=enrichment.get("operation_type", "arithmetic"),
                                           initialization_method=enrichment.get("initialization_method"),
                                           conditional_context=enrichment.get("conditional_context"),
                                           transformation=enrichment.get("transformation", f"{opcode}: decrement {counter}"))

                    self.register_sources[counter] = new_sources
                    graph.add_definition(counter, "register", file, line_no, opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=enrichment.get("operation_type", "arithmetic"),
                                       initialization_method=enrichment.get("initialization_method"),
                                       conditional_context=enrichment.get("conditional_context"),
                                       transformation=enrichment.get("transformation", f"Define {counter} via {opcode}"))

            # Insert Character (IC)
            if opcode == "IC" and len(operands) >= 2:
                target = operands[0].upper()
                source_raw, source_kind, index_reg, base_reg = self._normalize_operand_full(operands[1])
                source = self._qualify_memory_access(source_raw, source_kind, index_reg, base_reg)
                self._emit_equ_memory_aliases(
                    graph, operands[1], source_raw, source_kind, index_reg, base_reg, source,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )

                # Track index and base register usage
                self._add_index_base_usage(index_reg, base_reg, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=source if source_kind == "memory" else None)

                # IC modifies only rightmost byte, preserve existing sources
                existing = self.register_sources.get(target, {(target, "register")})
                new_sources = existing | {(source, source_kind)}

                # Get enrichment for source or target
                enrichment = self._get_enrichment(source if source_kind == "memory" else target, file, line_no)

                for src, src_kind in new_sources:
                    graph.add_assignment(src, src_kind, target, "register",
                                       file, line_no, " ".join(operands),
                                       scope, statement_id, macro_parent,
                                       operation_type=enrichment.get("operation_type", "load"),
                                       initialization_method=enrichment.get("initialization_method"),
                                       conditional_context=enrichment.get("conditional_context"),
                                       transformation=enrichment.get("transformation", f"IC: insert character from {src} into {target}"))
                self.register_sources[target] = new_sources
                graph.add_definition(target, "register", file, line_no, opcode,
                                   scope, statement_id, macro_parent,
                                   operation_type=enrichment.get("operation_type", "load"),
                                   initialization_method=enrichment.get("initialization_method"),
                                   conditional_context=enrichment.get("conditional_context"),
                                   transformation=enrichment.get("transformation", f"Define {target} via IC"))

            # Insert Characters under Mask (ICM)
            if opcode == "ICM" and len(operands) >= 3:
                target = operands[0].upper()
                mask = operands[1]  # Bit mask
                source_raw, source_kind, index_reg, base_reg = self._normalize_operand_full(operands[2])
                source = self._qualify_memory_access(source_raw, source_kind, index_reg, base_reg)
                self._emit_equ_memory_aliases(
                    graph, operands[2], source_raw, source_kind, index_reg, base_reg, source,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )

                # Track index and base register usage
                self._add_index_base_usage(index_reg, base_reg, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=source if source_kind == "memory" else None)

                # Similar to IC but mask indicates which bytes
                existing = self.register_sources.get(target, {(target, "register")})
                new_sources = existing | {(source, source_kind)}

                # Get enrichment for source or target
                enrichment = self._get_enrichment(source if source_kind == "memory" else target, file, line_no)

                for src, src_kind in new_sources:
                    graph.add_assignment(src, src_kind, target, "register",
                                       file, line_no, f"{' '.join(operands)} [mask={mask}]",
                                       scope, statement_id, macro_parent,
                                       operation_type=enrichment.get("operation_type", "load"),
                                       initialization_method=enrichment.get("initialization_method"),
                                       conditional_context=enrichment.get("conditional_context"),
                                       transformation=enrichment.get("transformation", f"ICM: insert characters from {src} into {target} (mask={mask})"))
                self.register_sources[target] = new_sources
                graph.add_definition(target, "register", file, line_no, opcode,
                                   scope, statement_id, macro_parent,
                                   operation_type=enrichment.get("operation_type", "load"),
                                   initialization_method=enrichment.get("initialization_method"),
                                   conditional_context=enrichment.get("conditional_context"),
                                   transformation=enrichment.get("transformation", f"Define {target} via ICM (mask={mask})"))

            # Store Character (STC)
            if opcode == "STC" and len(operands) >= 2:
                source_reg = operands[0].upper()
                target_raw, target_kind, index_reg, base_reg = self._normalize_operand_full(operands[1])
                target = self._qualify_memory_access(target_raw, target_kind, index_reg, base_reg)
                self._emit_equ_memory_aliases(
                    graph, operands[1], target_raw, target_kind, index_reg, base_reg, target,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )

                # Track index and base register usage
                self._add_index_base_usage(index_reg, base_reg, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=target)

                # Get enrichment for target memory location
                enrichment = self._get_enrichment(target, file, line_no)

                graph.add_assignment(source_reg, "register", target, "memory",
                                   file, line_no, " ".join(operands),
                                   scope, statement_id, macro_parent,
                                   operation_type=enrichment.get("operation_type", "store"),
                                   initialization_method=enrichment.get("initialization_method"),
                                   conditional_context=enrichment.get("conditional_context"),
                                   transformation=enrichment.get("transformation", f"STC: store character from {source_reg} to {target}"))
                # Propagate contributing sources
                for src, src_kind in self.register_sources.get(source_reg, set()):
                    graph.add_assignment(src, src_kind, target, "memory",
                                       file, line_no, " ".join(operands),
                                       scope, statement_id, macro_parent,
                                       operation_type=enrichment.get("operation_type", "store"),
                                       initialization_method=enrichment.get("initialization_method"),
                                       conditional_context=enrichment.get("conditional_context"),
                                       transformation=enrichment.get("transformation", f"Store {src} to {target}"))

            # Store Characters under Mask (STCM)
            if opcode == "STCM" and len(operands) >= 3:
                source_reg = operands[0].upper()
                mask = operands[1]
                target_raw, target_kind, index_reg, base_reg = self._normalize_operand_full(operands[2])
                target = self._qualify_memory_access(target_raw, target_kind, index_reg, base_reg)
                self._emit_equ_memory_aliases(
                    graph, operands[2], target_raw, target_kind, index_reg, base_reg, target,
                    file, line_no, " ".join(operands), scope, statement_id, macro_parent
                )

                # Track index and base register usage
                self._add_index_base_usage(index_reg, base_reg, graph, file, line_no,
                                          opcode, scope, statement_id, macro_parent,
                                          memory_symbol=target)

                # Get enrichment for target memory location
                enrichment = self._get_enrichment(target, file, line_no)

                graph.add_assignment(source_reg, "register", target, "memory",
                                   file, line_no, f"{' '.join(operands)} [mask={mask}]",
                                   scope, statement_id, macro_parent,
                                   operation_type=enrichment.get("operation_type", "store"),
                                   initialization_method=enrichment.get("initialization_method"),
                                   conditional_context=enrichment.get("conditional_context"),
                                   transformation=enrichment.get("transformation", f"STCM: store characters from {source_reg} to {target} (mask={mask})"))
                for src, src_kind in self.register_sources.get(source_reg, set()):
                    graph.add_assignment(src, src_kind, target, "memory",
                                       file, line_no, f"{' '.join(operands)} [mask={mask}]",
                                       scope, statement_id, macro_parent,
                                       operation_type=enrichment.get("operation_type", "store"),
                                       initialization_method=enrichment.get("initialization_method"),
                                       conditional_context=enrichment.get("conditional_context"),
                                       transformation=enrichment.get("transformation", f"Store {src} to {target} (mask={mask})"))

            # Comparison operations (read-only, emit use edges)
            COMPARISON_OPS = ("C", "CR", "CH", "CL", "CLR", "CLI", "CLC", "CLCL")
            if opcode in COMPARISON_OPS:
                # Register-to-register/memory comparisons
                if opcode in ("C", "CR", "CH", "CL", "CLR") and len(operands) >= 2:
                    operand1 = operands[0].upper()
                    operand2 = operands[1]

                    # Get enrichment for first operand
                    enrichment = self._get_enrichment(operand1, file, line_no)

                    # First operand is always a register
                    graph.add_usage(operand1, "register", file, line_no, opcode,
                                  scope, statement_id, macro_parent,
                                  operation_type=enrichment.get("operation_type", "compare"),
                                  initialization_method=enrichment.get("initialization_method"),
                                  conditional_context=enrichment.get("conditional_context"),
                                  transformation=enrichment.get("transformation"))
                    # Propagate sources
                    for src, src_kind in self.register_sources.get(operand1, set()):
                        graph.add_usage(src, src_kind, file, line_no, opcode,
                                      scope, statement_id, macro_parent,
                                      operation_type=enrichment.get("operation_type", "compare"),
                                      initialization_method=enrichment.get("initialization_method"),
                                      conditional_context=enrichment.get("conditional_context"),
                                      transformation=enrichment.get("transformation"))

                    # Second operand can be register or memory
                    if operand2.startswith("R") or operand2.startswith("r"):
                        op2_reg = operand2.upper()
                        enrichment2 = self._get_enrichment(op2_reg, file, line_no)
                        graph.add_usage(op2_reg, "register", file, line_no, opcode,
                                      scope, statement_id, macro_parent,
                                      operation_type=enrichment2.get("operation_type", "compare"),
                                      initialization_method=enrichment2.get("initialization_method"),
                                      conditional_context=enrichment2.get("conditional_context"),
                                      transformation=enrichment2.get("transformation"))
                        for src, src_kind in self.register_sources.get(op2_reg, set()):
                            graph.add_usage(src, src_kind, file, line_no, opcode,
                                          scope, statement_id, macro_parent,
                                          operation_type=enrichment2.get("operation_type", "compare"),
                                          initialization_method=enrichment2.get("initialization_method"),
                                          conditional_context=enrichment2.get("conditional_context"),
                                          transformation=enrichment2.get("transformation"))
                    else:
                        op2_mem, _ = self._normalize_operand(operand2)
                        enrichment2 = self._get_enrichment(op2_mem, file, line_no)
                        graph.add_usage(op2_mem, "memory", file, line_no, opcode,
                                      scope, statement_id, macro_parent,
                                      operation_type=enrichment2.get("operation_type", "compare"),
                                      initialization_method=enrichment2.get("initialization_method"),
                                      conditional_context=enrichment2.get("conditional_context"),
                                      transformation=enrichment2.get("transformation"))

                # Memory-to-memory comparisons
                if opcode in ("CLC", "CLCL") and len(operands) >= 2:
                    mem1, _ = self._normalize_operand(operands[0])
                    mem2, _ = self._normalize_operand(operands[1])

                    enrichment1 = self._get_enrichment(mem1, file, line_no)
                    enrichment2 = self._get_enrichment(mem2, file, line_no)

                    graph.add_usage(mem1, "memory", file, line_no, opcode,
                                  scope, statement_id, macro_parent,
                                  operation_type=enrichment1.get("operation_type", "compare"),
                                  initialization_method=enrichment1.get("initialization_method"),
                                  conditional_context=enrichment1.get("conditional_context"),
                                  transformation=enrichment1.get("transformation"))
                    graph.add_usage(mem2, "memory", file, line_no, opcode,
                                  scope, statement_id, macro_parent,
                                  operation_type=enrichment2.get("operation_type", "compare"),
                                  initialization_method=enrichment2.get("initialization_method"),
                                  conditional_context=enrichment2.get("conditional_context"),
                                  transformation=enrichment2.get("transformation"))

                # Immediate comparisons
                if opcode == "CLI" and len(operands) >= 2:
                    mem, _ = self._normalize_operand(operands[0])
                    enrichment = self._get_enrichment(mem, file, line_no)
                    # operands[1] is immediate value (literal), no node needed
                    graph.add_usage(mem, "memory", file, line_no, opcode,
                                  scope, statement_id, macro_parent,
                                  operation_type=enrichment.get("operation_type", "compare"),
                                  initialization_method=enrichment.get("initialization_method"),
                                  conditional_context=enrichment.get("conditional_context"),
                                  transformation=enrichment.get("transformation"))

            # Entry/save boundary macros: preserve incoming context.
            if opcode in ("ENTER", "SAVE", "SAVEX"):
                entry_node = f"entry_boundary@{scope or '(global)'}:{line_no}"
                entry_enrichment = self._get_enrichment(entry_node, file, line_no)
                graph.add_definition(entry_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                   scope, statement_id, macro_parent,
                                   operation_type=entry_enrichment.get("operation_type", "initialize"),
                                   initialization_method=entry_enrichment.get("initialization_method", "macro"),
                                   conditional_context=entry_enrichment.get("conditional_context"),
                                   transformation=entry_enrichment.get("transformation", f"Enter/save boundary via {opcode}"))

                for idx, arg_operand in enumerate(operands, start=1):
                    arg_value = self._macro_operand_value(arg_operand)
                    if not arg_value or arg_value.startswith("&"):
                        continue
                    arg_raw, arg_kind, arg_index, arg_base = self._normalize_operand_full(arg_value)
                    arg_symbol = self._qualify_memory_access(arg_raw, arg_kind, arg_index, arg_base)
                    if not arg_symbol:
                        continue

                    self._emit_equ_memory_aliases(
                        graph, arg_value, arg_raw, arg_kind, arg_index, arg_base, arg_symbol,
                        file, line_no, " ".join(operands) or opcode, scope, statement_id, macro_parent
                    )
                    self._add_index_base_usage(arg_index, arg_base, graph, file, line_no,
                                              opcode, scope, statement_id, macro_parent,
                                              memory_symbol=arg_symbol if arg_kind == "memory" else None)

                    arg_node = f"{entry_node}:ARG{idx}"
                    arg_enrichment = self._get_enrichment(arg_symbol, file, line_no)
                    graph.add_assignment(arg_symbol, arg_kind, arg_node, "parameter",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=arg_enrichment.get("operation_type", "initialize"),
                                       initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                       conditional_context=arg_enrichment.get("conditional_context"),
                                       transformation=arg_enrichment.get("transformation", f"Capture entry argument {arg_symbol}"))
                    if arg_kind == "register":
                        for src, src_kind in self.register_sources.get(arg_symbol, set()):
                            graph.add_assignment(src, src_kind, arg_node, "parameter",
                                               file, line_no, " ".join(operands) or opcode,
                                               scope, statement_id, macro_parent,
                                               operation_type=arg_enrichment.get("operation_type", "initialize"),
                                               initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                               conditional_context=arg_enrichment.get("conditional_context"),
                                               transformation=arg_enrichment.get("transformation", f"Capture entry source {src}"))
                    graph.add_assignment(arg_node, "parameter", entry_node, "parameter",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=arg_enrichment.get("operation_type", "initialize"),
                                       initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                       conditional_context=arg_enrichment.get("conditional_context"),
                                       transformation=arg_enrichment.get("transformation", "Bind entry argument"))

                graph.add_usage(entry_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                scope, statement_id, macro_parent,
                                operation_type=entry_enrichment.get("operation_type", "initialize"),
                                initialization_method=entry_enrichment.get("initialization_method", "macro"),
                                conditional_context=entry_enrichment.get("conditional_context"),
                                transformation=entry_enrichment.get("transformation", "Enter/save context"))
                continue

            # Exit/return boundary macros.
            if opcode in ("EXIT", "EXITC", "RETURN", "RETRN"):
                exit_node = f"exit_boundary@{scope or '(global)'}:{line_no}"
                exit_enrichment = self._get_enrichment(exit_node, file, line_no)
                graph.add_definition(exit_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                   scope, statement_id, macro_parent,
                                   operation_type=exit_enrichment.get("operation_type", "call"),
                                   initialization_method=exit_enrichment.get("initialization_method", "macro"),
                                   conditional_context=exit_enrichment.get("conditional_context"),
                                   transformation=exit_enrichment.get("transformation", f"Exit/return boundary via {opcode}"))

                captured_args = 0
                for idx, arg_operand in enumerate(operands, start=1):
                    arg_value = self._macro_operand_value(arg_operand)
                    if not arg_value or arg_value.startswith("&"):
                        continue
                    arg_raw, arg_kind, arg_index, arg_base = self._normalize_operand_full(arg_value)
                    arg_symbol = self._qualify_memory_access(arg_raw, arg_kind, arg_index, arg_base)
                    if not arg_symbol:
                        continue

                    captured_args += 1
                    self._emit_equ_memory_aliases(
                        graph, arg_value, arg_raw, arg_kind, arg_index, arg_base, arg_symbol,
                        file, line_no, " ".join(operands) or opcode, scope, statement_id, macro_parent
                    )
                    self._add_index_base_usage(arg_index, arg_base, graph, file, line_no,
                                              opcode, scope, statement_id, macro_parent,
                                              memory_symbol=arg_symbol if arg_kind == "memory" else None)

                    arg_node = f"{exit_node}:ARG{idx}"
                    arg_enrichment = self._get_enrichment(arg_symbol, file, line_no)
                    graph.add_assignment(arg_symbol, arg_kind, arg_node, "parameter",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=arg_enrichment.get("operation_type", "call"),
                                       initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                       conditional_context=arg_enrichment.get("conditional_context"),
                                       transformation=arg_enrichment.get("transformation", f"Capture exit argument {arg_symbol}"))
                    if arg_kind == "register":
                        for src, src_kind in self.register_sources.get(arg_symbol, set()):
                            graph.add_assignment(src, src_kind, arg_node, "parameter",
                                               file, line_no, " ".join(operands) or opcode,
                                               scope, statement_id, macro_parent,
                                               operation_type=arg_enrichment.get("operation_type", "call"),
                                               initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                               conditional_context=arg_enrichment.get("conditional_context"),
                                               transformation=arg_enrichment.get("transformation", f"Capture exit source {src}"))
                    graph.add_assignment(arg_node, "parameter", exit_node, "parameter",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=arg_enrichment.get("operation_type", "call"),
                                       initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                       conditional_context=arg_enrichment.get("conditional_context"),
                                       transformation=arg_enrichment.get("transformation", "Bind exit argument"))

                # Most return paths eventually consume R14 even when not explicit.
                if captured_args == 0:
                    r14_enrichment = self._get_enrichment("R14", file, line_no)
                    graph.add_usage("R14", "register", file, line_no, opcode,
                                    scope, statement_id, macro_parent,
                                    operation_type=r14_enrichment.get("operation_type", "call"),
                                    initialization_method=r14_enrichment.get("initialization_method"),
                                    conditional_context=r14_enrichment.get("conditional_context"),
                                    transformation=r14_enrichment.get("transformation", f"Return via {opcode}"))
                    for src, src_kind in self.register_sources.get("R14", set()):
                        graph.add_usage(src, src_kind, file, line_no, opcode,
                                        scope, statement_id, macro_parent,
                                        operation_type=r14_enrichment.get("operation_type", "call"),
                                        initialization_method=r14_enrichment.get("initialization_method"),
                                        conditional_context=r14_enrichment.get("conditional_context"),
                                        transformation=r14_enrichment.get("transformation", f"Return source via {opcode}"))

                graph.add_usage(exit_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                scope, statement_id, macro_parent,
                                operation_type=exit_enrichment.get("operation_type", "call"),
                                initialization_method=exit_enrichment.get("initialization_method", "macro"),
                                conditional_context=exit_enrichment.get("conditional_context"),
                                transformation=exit_enrichment.get("transformation", "Exit/return context"))
                continue

            # Serialization/lock boundaries.
            if opcode in ("ENQ", "DEQ", "SERRC", "RELSC"):
                is_release = opcode in ("DEQ", "RELSC")
                lock_node = f"{'unlock' if is_release else 'lock'}@{scope or '(global)'}:{line_no}"
                lock_enrichment = self._get_enrichment(lock_node, file, line_no)
                graph.add_definition(lock_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                   scope, statement_id, macro_parent,
                                   operation_type=lock_enrichment.get("operation_type", "call"),
                                   initialization_method=lock_enrichment.get("initialization_method", "macro"),
                                   conditional_context=lock_enrichment.get("conditional_context"),
                                   transformation=lock_enrichment.get(
                                       "transformation",
                                       ("Release serialization lock" if is_release else "Acquire serialization lock"),
                                   ))

                lock_keys = {"ECB", "LOCK", "TOKEN", "RESOURCE", "NAME", "ID", "QNAME", "RNAME", "KEY", "ADDR", "PTR", "OBJ"}
                skip_keys = {"MODE", "WAIT", "COND", "SCOPE", "TASK", "OWNER", "R", "N", "Y", "NO", "YES", "LEVEL", "LVL", "DATA"}
                lock_values: List[str] = []
                for key, value in self._iter_macro_arguments(operands):
                    candidate = (value or "").strip()
                    if not candidate or candidate.startswith("&"):
                        continue
                    if key in lock_keys or key is None:
                        lock_values.append(candidate)
                        continue
                    if key not in skip_keys:
                        lock_values.append(candidate)

                for idx, candidate in enumerate(lock_values, start=1):
                    arg_raw, arg_kind, arg_index, arg_base = self._normalize_operand_full(candidate)
                    arg_symbol = self._qualify_memory_access(arg_raw, arg_kind, arg_index, arg_base)
                    if not arg_symbol:
                        continue
                    self._emit_equ_memory_aliases(
                        graph, candidate, arg_raw, arg_kind, arg_index, arg_base, arg_symbol,
                        file, line_no, " ".join(operands) or opcode, scope, statement_id, macro_parent
                    )
                    self._add_index_base_usage(arg_index, arg_base, graph, file, line_no,
                                              opcode, scope, statement_id, macro_parent,
                                              memory_symbol=arg_symbol if arg_kind == "memory" else None)
                    arg_node = f"{lock_node}:ARG{idx}"
                    arg_enrichment = self._get_enrichment(arg_symbol, file, line_no)
                    graph.add_assignment(arg_symbol, arg_kind, arg_node, "parameter",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=arg_enrichment.get("operation_type", "call"),
                                       initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                       conditional_context=arg_enrichment.get("conditional_context"),
                                       transformation=arg_enrichment.get("transformation", f"Pass lock operand {arg_symbol}"))
                    if arg_kind == "register":
                        for src, src_kind in self.register_sources.get(arg_symbol, set()):
                            graph.add_assignment(src, src_kind, arg_node, "parameter",
                                               file, line_no, " ".join(operands) or opcode,
                                               scope, statement_id, macro_parent,
                                               operation_type=arg_enrichment.get("operation_type", "call"),
                                               initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                               conditional_context=arg_enrichment.get("conditional_context"),
                                               transformation=arg_enrichment.get("transformation", f"Pass lock source {src}"))
                    graph.add_assignment(arg_node, "parameter", lock_node, "parameter",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=arg_enrichment.get("operation_type", "call"),
                                       initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                       conditional_context=arg_enrichment.get("conditional_context"),
                                       transformation=arg_enrichment.get("transformation", "Bind lock operand"))

                graph.add_usage(lock_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                scope, statement_id, macro_parent,
                                operation_type=lock_enrichment.get("operation_type", "call"),
                                initialization_method=lock_enrichment.get("initialization_method", "macro"),
                                conditional_context=lock_enrichment.get("conditional_context"),
                                transformation=lock_enrichment.get(
                                    "transformation",
                                    ("Release serialization lock" if is_release else "Acquire serialization lock"),
                                ))
                continue

            # Heap allocation boundaries.
            if opcode in ("MALOC", "RELOC"):
                args = self._iter_macro_arguments(operands)
                target_keys = {"REG", "PTR", "TARGET", "OUT", "TO", "RESULT", "RTN"}
                source_keys = {"OLD", "FROM", "SRC", "SOURCE", "IN", "PTRIN", "CUR"}
                size_keys = {"SIZE", "LEN", "LENGTH", "BYTES"}

                target_reg: Optional[str] = None
                source_ptr: Optional[str] = None
                size_value: Optional[str] = None
                positional_values: List[str] = []

                for key, value in args:
                    candidate = (value or "").strip()
                    if not candidate or candidate.startswith("&"):
                        continue
                    reg = self._normalize_register_operand(candidate)
                    if key in target_keys and reg and not target_reg:
                        target_reg = reg
                    if key in source_keys and not source_ptr:
                        source_ptr = candidate
                    if key in size_keys and not size_value:
                        size_value = candidate
                    if key is None:
                        positional_values.append(candidate)

                if not target_reg:
                    for candidate in positional_values:
                        reg = self._normalize_register_operand(candidate)
                        if reg:
                            target_reg = reg
                            break

                if opcode == "RELOC" and not source_ptr:
                    source_ptr = target_reg or (positional_values[0] if positional_values else None)

                if not size_value:
                    for candidate in positional_values:
                        if self._normalize_register_operand(candidate):
                            continue
                        if self._parse_macro_level_token(candidate) is not None:
                            continue
                        if candidate.upper() in {"R", "N", "Y", "NO", "YES"}:
                            continue
                        size_value = candidate
                        break

                heap_node = f"{'heap_realloc' if opcode == 'RELOC' else 'heap_alloc'}@{scope or '(global)'}:{line_no}"
                heap_enrichment = self._get_enrichment(heap_node, file, line_no)
                graph.add_definition(heap_node, "memory", file, line_no, " ".join(operands) or opcode,
                                   scope, statement_id, macro_parent,
                                   operation_type=heap_enrichment.get("operation_type", "initialize"),
                                   initialization_method=heap_enrichment.get("initialization_method", "macro"),
                                   conditional_context=heap_enrichment.get("conditional_context"),
                                   transformation=heap_enrichment.get(
                                       "transformation",
                                       ("Reallocate heap block" if opcode == "RELOC" else "Allocate heap block"),
                                   ))

                propagated_sources: Set[Tuple[str, str]] = {(heap_node, "memory")}

                if source_ptr:
                    src_raw, src_kind, src_index, src_base = self._normalize_operand_full(source_ptr)
                    src_symbol = self._qualify_memory_access(src_raw, src_kind, src_index, src_base)
                    self._emit_equ_memory_aliases(
                        graph, source_ptr, src_raw, src_kind, src_index, src_base, src_symbol,
                        file, line_no, " ".join(operands) or opcode, scope, statement_id, macro_parent
                    )
                    self._add_index_base_usage(src_index, src_base, graph, file, line_no,
                                              opcode, scope, statement_id, macro_parent,
                                              memory_symbol=src_symbol if src_kind == "memory" else None)
                    graph.add_assignment(src_symbol, src_kind, heap_node, "memory",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=heap_enrichment.get("operation_type", "initialize"),
                                       initialization_method=heap_enrichment.get("initialization_method", "macro"),
                                       conditional_context=heap_enrichment.get("conditional_context"),
                                       transformation=heap_enrichment.get("transformation", f"Use {src_symbol} as realloc source"))
                    if src_kind == "register":
                        reg_sources = self.register_sources.get(src_symbol, {(src_symbol, "register")})
                        propagated_sources.update(reg_sources)
                        for src, src_kind in reg_sources:
                            graph.add_assignment(src, src_kind, heap_node, "memory",
                                               file, line_no, " ".join(operands) or opcode,
                                               scope, statement_id, macro_parent,
                                               operation_type=heap_enrichment.get("operation_type", "initialize"),
                                               initialization_method=heap_enrichment.get("initialization_method", "macro"),
                                               conditional_context=heap_enrichment.get("conditional_context"),
                                               transformation=heap_enrichment.get("transformation", f"Propagate realloc source {src}"))
                    else:
                        propagated_sources.add((src_symbol, src_kind))

                if size_value:
                    size_raw, size_kind, size_index, size_base = self._normalize_operand_full(size_value)
                    size_symbol = self._qualify_memory_access(size_raw, size_kind, size_index, size_base)
                    self._emit_equ_memory_aliases(
                        graph, size_value, size_raw, size_kind, size_index, size_base, size_symbol,
                        file, line_no, " ".join(operands) or opcode, scope, statement_id, macro_parent
                    )
                    self._add_index_base_usage(size_index, size_base, graph, file, line_no,
                                              opcode, scope, statement_id, macro_parent,
                                              memory_symbol=size_symbol if size_kind == "memory" else None)
                    size_node = f"{heap_node}:SIZE"
                    graph.add_assignment(size_symbol, size_kind, size_node, "parameter",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=heap_enrichment.get("operation_type", "initialize"),
                                       initialization_method=heap_enrichment.get("initialization_method", "macro"),
                                       conditional_context=heap_enrichment.get("conditional_context"),
                                       transformation=heap_enrichment.get("transformation", f"Bind allocation size {size_symbol}"))
                    if size_kind == "register":
                        for src, src_kind in self.register_sources.get(size_symbol, set()):
                            graph.add_assignment(src, src_kind, size_node, "parameter",
                                               file, line_no, " ".join(operands) or opcode,
                                               scope, statement_id, macro_parent,
                                               operation_type=heap_enrichment.get("operation_type", "initialize"),
                                               initialization_method=heap_enrichment.get("initialization_method", "macro"),
                                               conditional_context=heap_enrichment.get("conditional_context"),
                                               transformation=heap_enrichment.get("transformation", f"Propagate size source {src}"))

                if target_reg:
                    graph.add_assignment(heap_node, "memory", target_reg, "register",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=heap_enrichment.get("operation_type", "initialize"),
                                       initialization_method=heap_enrichment.get("initialization_method", "macro"),
                                       conditional_context=heap_enrichment.get("conditional_context"),
                                       transformation=heap_enrichment.get("transformation", f"Return heap pointer to {target_reg}"))
                    self.register_sources[target_reg] = propagated_sources
                    for src, src_kind in propagated_sources:
                        graph.add_assignment(src, src_kind, target_reg, "register",
                                           file, line_no, " ".join(operands) or opcode,
                                           scope, statement_id, macro_parent,
                                           operation_type=heap_enrichment.get("operation_type", "initialize"),
                                           initialization_method=heap_enrichment.get("initialization_method", "macro"),
                                           conditional_context=heap_enrichment.get("conditional_context"),
                                           transformation=heap_enrichment.get("transformation", f"Propagate heap source {src} to {target_reg}"))
                    graph.add_definition(target_reg, "register", file, line_no, opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=heap_enrichment.get("operation_type", "initialize"),
                                       initialization_method=heap_enrichment.get("initialization_method", "macro"),
                                       conditional_context=heap_enrichment.get("conditional_context"),
                                       transformation=heap_enrichment.get("transformation", f"Define {target_reg} from {opcode}"))

                graph.add_usage(heap_node, "memory", file, line_no, " ".join(operands) or opcode,
                                scope, statement_id, macro_parent,
                                operation_type=heap_enrichment.get("operation_type", "initialize"),
                                initialization_method=heap_enrichment.get("initialization_method", "macro"),
                                conditional_context=heap_enrichment.get("conditional_context"),
                                transformation=heap_enrichment.get(
                                    "transformation",
                                    ("Reallocate heap block" if opcode == "RELOC" else "Allocate heap block"),
                                ))
                continue

            # CALLC / CALLCPP bridge: ASM macro invokes C/C++ entry and returns.
            # Pattern G fix:
            #   - CALLCPP: extend fixed 3-slot window to also include live registers.
            #   - CALLC: add a dedicated handler (was completely missing).
            if opcode in ("CALLC", "CALLCPP"):
                extracted_target = self._extract_callcpp_target(operands)
                if extracted_target:
                    callee_name = self._resolve_call_target(extracted_target)
                    call_enrichment = self._get_enrichment("R15", file, line_no)

                    _ALWAYS_ARG_REGS = {"R0", "R1", "R15"}
                    _GP_REGS = {f"R{i}" for i in range(16)}
                    live_regs = {r for r, srcs in self.register_sources.items()
                                 if r in _GP_REGS and srcs}
                    arg_regs = sorted(_ALWAYS_ARG_REGS | live_regs)

                    for arg_reg in arg_regs:
                        arg_node = f"arg@{callee_name}:{arg_reg}"
                        arg_enrichment = self._get_enrichment(arg_reg, file, line_no)
                        graph.add_assignment(arg_reg, "register", arg_node, "parameter",
                                           file, line_no, " ".join(operands) or opcode,
                                           scope, statement_id, macro_parent,
                                           operation_type=arg_enrichment.get("operation_type", "call"),
                                           initialization_method=arg_enrichment.get("initialization_method"),
                                           conditional_context=arg_enrichment.get("conditional_context"),
                                           transformation=arg_enrichment.get("transformation", f"Pass {arg_reg} to C++ callee {callee_name}"))
                        for src, src_kind in self.register_sources.get(arg_reg, set()):
                            graph.add_assignment(src, src_kind, arg_node, "parameter",
                                               file, line_no, " ".join(operands) or opcode,
                                               scope, statement_id, macro_parent,
                                               operation_type=arg_enrichment.get("operation_type", "call"),
                                               initialization_method=arg_enrichment.get("initialization_method"),
                                               conditional_context=arg_enrichment.get("conditional_context"),
                                               transformation=arg_enrichment.get("transformation", f"Pass {src} to C++ callee {callee_name}"))

                    graph.add_usage(callee_name, "function", file, line_no, " ".join(operands) or opcode,
                                    scope, statement_id, macro_parent,
                                    operation_type=call_enrichment.get("operation_type", "call"),
                                    initialization_method=call_enrichment.get("initialization_method"),
                                    conditional_context=call_enrichment.get("conditional_context"),
                                    transformation=call_enrichment.get("transformation", f"Invoke C++ callee {callee_name} via CALLCPP"))

                    return_node = f"return@{callee_name}"
                    graph.add_assignment(return_node, "return_value", "R15", "register",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=call_enrichment.get("operation_type", "call"),
                                       initialization_method=call_enrichment.get("initialization_method"),
                                       conditional_context=call_enrichment.get("conditional_context"),
                                       transformation=call_enrichment.get("transformation", f"Return from {callee_name} to R15"))
                    self.register_sources["R15"] = {(return_node, "return_value")}
                    graph.add_definition("R15", "register", file, line_no, f"{opcode} (return)",
                                       scope, statement_id, macro_parent,
                                       operation_type=call_enrichment.get("operation_type", "call"),
                                       initialization_method=call_enrichment.get("initialization_method"),
                                       conditional_context=call_enrichment.get("conditional_context"),
                                       transformation=call_enrichment.get("transformation", "Define R15 from CALLCPP return"))
                    continue

            # ISCFA record exchange boundary.
            if opcode == "ISCFA":
                mode: Optional[str] = None
                for key, value in self._iter_macro_arguments(operands):
                    upper_value = value.strip().upper()
                    if upper_value in {"RECEIVE", "SEND"}:
                        mode = upper_value
                        break
                    if key in {"OP", "MODE", "ACTION", "TYPE"} and upper_value in {"RECEIVE", "SEND"}:
                        mode = upper_value
                        break

                level_number = self._extract_macro_level_number(operands)
                level_name = self._macro_level_name(level_number) if level_number is not None else None
                recptr_value = self._extract_macro_keyword_value(operands, {"RECPTR", "MSGPTR", "BUFFER", "BUF", "PTR"})
                reclen_value = self._extract_macro_keyword_value(operands, {"RECLEN", "LEN", "LENGTH", "SIZE"})

                op_node = f"iscfa_{(mode or 'OP').lower()}@{(level_name or (scope or '(global)'))}:{line_no}"
                op_enrichment = self._get_enrichment(op_node, file, line_no)
                graph.add_definition(op_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                   scope, statement_id, macro_parent,
                                   operation_type=op_enrichment.get("operation_type", "move"),
                                   initialization_method=op_enrichment.get("initialization_method", "macro"),
                                   conditional_context=op_enrichment.get("conditional_context"),
                                   transformation=op_enrichment.get("transformation", f"ISCFA {mode or 'operation'} boundary"))

                if level_name:
                    graph.add_definition(level_name, "memory", file, line_no, opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=op_enrichment.get("operation_type", "move"),
                                       initialization_method=op_enrichment.get("initialization_method", "macro"),
                                       conditional_context=op_enrichment.get("conditional_context"),
                                       transformation=op_enrichment.get("transformation", f"Reference level {level_name} in ISCFA"))
                    graph.add_usage(level_name, "memory", file, line_no, " ".join(operands) or opcode,
                                    scope, statement_id, macro_parent,
                                    operation_type=op_enrichment.get("operation_type", "move"),
                                    initialization_method=op_enrichment.get("initialization_method", "macro"),
                                    conditional_context=op_enrichment.get("conditional_context"),
                                    transformation=op_enrichment.get("transformation", f"Read/write level {level_name} in ISCFA"))

                if recptr_value:
                    ptr_raw, ptr_kind, ptr_index, ptr_base = self._normalize_operand_full(recptr_value)
                    ptr_symbol = self._qualify_memory_access(ptr_raw, ptr_kind, ptr_index, ptr_base)
                    self._emit_equ_memory_aliases(
                        graph, recptr_value, ptr_raw, ptr_kind, ptr_index, ptr_base, ptr_symbol,
                        file, line_no, " ".join(operands) or opcode, scope, statement_id, macro_parent
                    )
                    self._add_index_base_usage(ptr_index, ptr_base, graph, file, line_no,
                                              opcode, scope, statement_id, macro_parent,
                                              memory_symbol=ptr_symbol if ptr_kind == "memory" else None)
                    ptr_enrichment = self._get_enrichment(ptr_symbol, file, line_no)
                    if mode == "RECEIVE":
                        if level_name:
                            graph.add_assignment(level_name, "memory", op_node, "parameter",
                                               file, line_no, " ".join(operands) or opcode,
                                               scope, statement_id, macro_parent,
                                               operation_type=ptr_enrichment.get("operation_type", "move"),
                                               initialization_method=ptr_enrichment.get("initialization_method", "macro"),
                                               conditional_context=ptr_enrichment.get("conditional_context"),
                                               transformation=ptr_enrichment.get("transformation", f"Receive record from {level_name}"))
                        graph.add_assignment(op_node, "parameter", ptr_symbol, ptr_kind,
                                           file, line_no, " ".join(operands) or opcode,
                                           scope, statement_id, macro_parent,
                                           operation_type=ptr_enrichment.get("operation_type", "move"),
                                           initialization_method=ptr_enrichment.get("initialization_method", "macro"),
                                           conditional_context=ptr_enrichment.get("conditional_context"),
                                           transformation=ptr_enrichment.get("transformation", f"Write received record to {ptr_symbol}"))
                    elif mode == "SEND":
                        graph.add_assignment(ptr_symbol, ptr_kind, op_node, "parameter",
                                           file, line_no, " ".join(operands) or opcode,
                                           scope, statement_id, macro_parent,
                                           operation_type=ptr_enrichment.get("operation_type", "move"),
                                           initialization_method=ptr_enrichment.get("initialization_method", "macro"),
                                           conditional_context=ptr_enrichment.get("conditional_context"),
                                           transformation=ptr_enrichment.get("transformation", f"Read send record from {ptr_symbol}"))
                        if ptr_kind == "register":
                            for src, src_kind in self.register_sources.get(ptr_symbol, set()):
                                graph.add_assignment(src, src_kind, op_node, "parameter",
                                                   file, line_no, " ".join(operands) or opcode,
                                                   scope, statement_id, macro_parent,
                                                   operation_type=ptr_enrichment.get("operation_type", "move"),
                                                   initialization_method=ptr_enrichment.get("initialization_method", "macro"),
                                                   conditional_context=ptr_enrichment.get("conditional_context"),
                                                   transformation=ptr_enrichment.get("transformation", f"Propagate send source {src}"))
                        if level_name:
                            graph.add_assignment(op_node, "parameter", level_name, "memory",
                                               file, line_no, " ".join(operands) or opcode,
                                               scope, statement_id, macro_parent,
                                               operation_type=ptr_enrichment.get("operation_type", "move"),
                                               initialization_method=ptr_enrichment.get("initialization_method", "macro"),
                                               conditional_context=ptr_enrichment.get("conditional_context"),
                                               transformation=ptr_enrichment.get("transformation", f"Send record to {level_name}"))
                    else:
                        graph.add_assignment(ptr_symbol, ptr_kind, op_node, "parameter",
                                           file, line_no, " ".join(operands) or opcode,
                                           scope, statement_id, macro_parent,
                                           operation_type=ptr_enrichment.get("operation_type", "move"),
                                           initialization_method=ptr_enrichment.get("initialization_method", "macro"),
                                           conditional_context=ptr_enrichment.get("conditional_context"),
                                           transformation=ptr_enrichment.get("transformation", f"Bind ISCFA pointer {ptr_symbol}"))

                if reclen_value:
                    len_raw, len_kind, len_index, len_base = self._normalize_operand_full(reclen_value)
                    len_symbol = self._qualify_memory_access(len_raw, len_kind, len_index, len_base)
                    self._emit_equ_memory_aliases(
                        graph, reclen_value, len_raw, len_kind, len_index, len_base, len_symbol,
                        file, line_no, " ".join(operands) or opcode, scope, statement_id, macro_parent
                    )
                    self._add_index_base_usage(len_index, len_base, graph, file, line_no,
                                              opcode, scope, statement_id, macro_parent,
                                              memory_symbol=len_symbol if len_kind == "memory" else None)
                    graph.add_assignment(len_symbol, len_kind, f"{op_node}:LEN", "parameter",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=op_enrichment.get("operation_type", "move"),
                                       initialization_method=op_enrichment.get("initialization_method", "macro"),
                                       conditional_context=op_enrichment.get("conditional_context"),
                                       transformation=op_enrichment.get("transformation", f"Bind ISCFA record length {len_symbol}"))

                graph.add_usage(op_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                scope, statement_id, macro_parent,
                                operation_type=op_enrichment.get("operation_type", "move"),
                                initialization_method=op_enrichment.get("initialization_method", "macro"),
                                conditional_context=op_enrichment.get("conditional_context"),
                                transformation=op_enrichment.get("transformation", f"Process ISCFA {mode or 'operation'}"))
                continue

            # CRUSA level clear boundary: reset usage/scratch area for level.
            if opcode == "CRUSA":
                level_number = self._extract_macro_level_number(operands)
                level_name = self._macro_level_name(level_number) if level_number is not None else None
                crusa_enrichment = self._get_enrichment("CRUSA", file, line_no)
                if level_name:
                    graph.add_definition(level_name, "memory", file, line_no, opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=crusa_enrichment.get("operation_type", "initialize"),
                                       initialization_method=crusa_enrichment.get("initialization_method", "macro"),
                                       conditional_context=crusa_enrichment.get("conditional_context"),
                                       transformation=crusa_enrichment.get("transformation", f"Reset level {level_name} usage area"))
                    graph.add_definition("0", "literal", file, line_no, opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=crusa_enrichment.get("operation_type", "initialize"),
                                       initialization_method=crusa_enrichment.get("initialization_method", "macro"),
                                       conditional_context=crusa_enrichment.get("conditional_context"),
                                       transformation=crusa_enrichment.get("transformation", "Initialize clear value"))
                    graph.add_assignment("0", "literal", level_name, "memory",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=crusa_enrichment.get("operation_type", "initialize"),
                                       initialization_method=crusa_enrichment.get("initialization_method", "macro"),
                                       conditional_context=crusa_enrichment.get("conditional_context"),
                                       transformation=crusa_enrichment.get("transformation", f"Clear {level_name}"))
                    graph.add_usage(level_name, "memory", file, line_no, " ".join(operands) or opcode,
                                    scope, statement_id, macro_parent,
                                    operation_type=crusa_enrichment.get("operation_type", "initialize"),
                                    initialization_method=crusa_enrichment.get("initialization_method", "macro"),
                                    conditional_context=crusa_enrichment.get("conditional_context"),
                                    transformation=crusa_enrichment.get("transformation", f"Use {level_name} in CRUSA"))
                continue

            # Message macros: SEND* and RECV* payload boundaries.
            if opcode in ("SENDA", "SENDC", "RECVA", "RECVC"):
                direction = "recv" if opcode.startswith("RECV") else "send"
                payload_value = self._extract_macro_keyword_value(
                    operands,
                    {"DATA", "MSG", "MSGPTR", "RECPTR", "BUFFER", "BUF", "PTR", "TEXT"}
                )
                if not payload_value:
                    for key, value in self._iter_macro_arguments(operands):
                        upper = value.strip().upper()
                        if upper in {"SEND", "RECEIVE", "R", "N", "Y", "YES", "NO"}:
                            continue
                        if self._parse_macro_level_token(value) is not None:
                            continue
                        if key in {"MODE", "TYPE", "CLASS", "FLAG", "OPTIONS"}:
                            continue
                        payload_value = value.strip()
                        break

                msg_node = f"msg_{direction}@{opcode}:{scope or '(global)'}:{line_no}"
                msg_enrichment = self._get_enrichment(msg_node, file, line_no)
                graph.add_definition(msg_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                   scope, statement_id, macro_parent,
                                   operation_type=msg_enrichment.get("operation_type", "move"),
                                   initialization_method=msg_enrichment.get("initialization_method", "macro"),
                                   conditional_context=msg_enrichment.get("conditional_context"),
                                   transformation=msg_enrichment.get(
                                       "transformation",
                                       ("Receive message payload" if direction == "recv" else "Send message payload"),
                                   ))

                if payload_value:
                    payload_raw, payload_kind, payload_index, payload_base = self._normalize_operand_full(payload_value)
                    payload_symbol = self._qualify_memory_access(payload_raw, payload_kind, payload_index, payload_base)
                    self._emit_equ_memory_aliases(
                        graph, payload_value, payload_raw, payload_kind, payload_index, payload_base, payload_symbol,
                        file, line_no, " ".join(operands) or opcode, scope, statement_id, macro_parent
                    )
                    self._add_index_base_usage(payload_index, payload_base, graph, file, line_no,
                                              opcode, scope, statement_id, macro_parent,
                                              memory_symbol=payload_symbol if payload_kind == "memory" else None)

                    if direction == "send":
                        graph.add_assignment(payload_symbol, payload_kind, msg_node, "parameter",
                                           file, line_no, " ".join(operands) or opcode,
                                           scope, statement_id, macro_parent,
                                           operation_type=msg_enrichment.get("operation_type", "move"),
                                           initialization_method=msg_enrichment.get("initialization_method", "macro"),
                                           conditional_context=msg_enrichment.get("conditional_context"),
                                           transformation=msg_enrichment.get("transformation", f"Send payload from {payload_symbol}"))
                        if payload_kind == "register":
                            for src, src_kind in self.register_sources.get(payload_symbol, set()):
                                graph.add_assignment(src, src_kind, msg_node, "parameter",
                                                   file, line_no, " ".join(operands) or opcode,
                                                   scope, statement_id, macro_parent,
                                                   operation_type=msg_enrichment.get("operation_type", "move"),
                                                   initialization_method=msg_enrichment.get("initialization_method", "macro"),
                                                   conditional_context=msg_enrichment.get("conditional_context"),
                                                   transformation=msg_enrichment.get("transformation", f"Propagate message source {src}"))
                    else:
                        graph.add_assignment(msg_node, "parameter", payload_symbol, payload_kind,
                                           file, line_no, " ".join(operands) or opcode,
                                           scope, statement_id, macro_parent,
                                           operation_type=msg_enrichment.get("operation_type", "move"),
                                           initialization_method=msg_enrichment.get("initialization_method", "macro"),
                                           conditional_context=msg_enrichment.get("conditional_context"),
                                           transformation=msg_enrichment.get("transformation", f"Receive payload into {payload_symbol}"))

                graph.add_usage(msg_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                scope, statement_id, macro_parent,
                                operation_type=msg_enrichment.get("operation_type", "move"),
                                initialization_method=msg_enrichment.get("initialization_method", "macro"),
                                conditional_context=msg_enrichment.get("conditional_context"),
                                transformation=msg_enrichment.get(
                                    "transformation",
                                    ("Receive message payload" if direction == "recv" else "Send message payload"),
                                ))
                continue

            # GETCC/GETFC allocation boundary: allocate level block and return pointer linkage.
            if opcode in ("GETCC", "GETFC"):
                level_number = self._extract_macro_level_number(operands)
                level_name = self._macro_level_name(level_number) if level_number is not None else None
                alloc_anchor = level_name or (scope or "(global)")
                alloc_node = f"alloc@{alloc_anchor}:{line_no}"
                alloc_enrichment = self._get_enrichment(alloc_node, file, line_no)

                graph.add_definition(alloc_node, "memory", file, line_no, " ".join(operands) or opcode,
                                   scope, statement_id, macro_parent,
                                   operation_type=alloc_enrichment.get("operation_type", "initialize"),
                                   initialization_method=alloc_enrichment.get("initialization_method", "macro"),
                                   conditional_context=alloc_enrichment.get("conditional_context"),
                                   transformation=alloc_enrichment.get("transformation", f"Allocate block via {opcode}"))

                register_sources = {(alloc_node, "memory")}
                if level_name:
                    graph.add_definition(level_name, "memory", file, line_no, opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=alloc_enrichment.get("operation_type", "initialize"),
                                       initialization_method=alloc_enrichment.get("initialization_method", "macro"),
                                       conditional_context=alloc_enrichment.get("conditional_context"),
                                       transformation=alloc_enrichment.get("transformation", f"Activate ECB level {level_name}"))
                    graph.add_assignment(alloc_node, "memory", level_name, "memory",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=alloc_enrichment.get("operation_type", "initialize"),
                                       initialization_method=alloc_enrichment.get("initialization_method", "macro"),
                                       conditional_context=alloc_enrichment.get("conditional_context"),
                                       transformation=alloc_enrichment.get("transformation", f"Bind allocation to {level_name}"))
                    register_sources.add((level_name, "memory"))

                    ce1cr_slot = f"CE1CR{format(level_number, 'X')}"
                    graph.add_definition(ce1cr_slot, "memory", file, line_no, opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=alloc_enrichment.get("operation_type", "initialize"),
                                       initialization_method=alloc_enrichment.get("initialization_method", "macro"),
                                       conditional_context=alloc_enrichment.get("conditional_context"),
                                       transformation=alloc_enrichment.get("transformation", f"Set pointer slot {ce1cr_slot}"))
                    graph.add_assignment(level_name, "memory", ce1cr_slot, "memory",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=alloc_enrichment.get("operation_type", "initialize"),
                                       initialization_method=alloc_enrichment.get("initialization_method", "macro"),
                                       conditional_context=alloc_enrichment.get("conditional_context"),
                                       transformation=alloc_enrichment.get("transformation", f"Point {ce1cr_slot} to {level_name}"))
                    register_sources.add((ce1cr_slot, "memory"))
                    graph.add_assignment(ce1cr_slot, "memory", "R1", "register",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=alloc_enrichment.get("operation_type", "call"),
                                       initialization_method=alloc_enrichment.get("initialization_method", "macro"),
                                       conditional_context=alloc_enrichment.get("conditional_context"),
                                       transformation=alloc_enrichment.get("transformation", f"Return {ce1cr_slot} pointer in R1"))
                else:
                    graph.add_assignment(alloc_node, "memory", "R1", "register",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=alloc_enrichment.get("operation_type", "call"),
                                       initialization_method=alloc_enrichment.get("initialization_method", "macro"),
                                       conditional_context=alloc_enrichment.get("conditional_context"),
                                       transformation=alloc_enrichment.get("transformation", "Return allocation pointer in R1"))

                self.register_sources["R1"] = register_sources
                graph.add_definition("R1", "register", file, line_no, opcode,
                                   scope, statement_id, macro_parent,
                                   operation_type=alloc_enrichment.get("operation_type", "call"),
                                   initialization_method=alloc_enrichment.get("initialization_method", "macro"),
                                   conditional_context=alloc_enrichment.get("conditional_context"),
                                   transformation=alloc_enrichment.get("transformation", f"Define R1 from {opcode}"))
                continue

            # CALOC heap allocation: emit alloc@heap node and link to R1 result register.
            # Mirrors the GETCC handler but for heap-allocated (non-ECB-level) blocks.
            # CALOC ESIZE=Rx,COUNT=R1 — R1 receives the heap block address; the size
            # register (ESIZE=) is allocation metadata, NOT the returned address.
            if opcode == "CALOC":
                args = list(self._iter_macro_arguments(operands))
                size_sym = None
                for _key, _value in args:
                    if (_key or "").upper() in ("ESIZE", "SIZE") and _value:
                        _candidate = _value.strip()
                        _reg = self._normalize_register_operand(_candidate)
                        size_sym = _reg or _candidate
                        break

                alloc_node = f"alloc@heap:{line_no}"
                alloc_enrichment = self._get_enrichment(alloc_node, file, line_no)

                graph.add_definition(alloc_node, "memory", file, line_no,
                                     " ".join(operands) or opcode,
                                     scope, statement_id, macro_parent,
                                     operation_type=alloc_enrichment.get("operation_type", "initialize"),
                                     initialization_method=alloc_enrichment.get("initialization_method", "macro"),
                                     conditional_context=alloc_enrichment.get("conditional_context"),
                                     transformation=alloc_enrichment.get("transformation",
                                                                          "Allocate heap block via CALOC"))

                if size_sym:
                    size_kind = "register" if self._normalize_register_operand(size_sym) else "memory"
                    graph.add_usage(size_sym, size_kind, file, line_no, opcode,
                                    scope, statement_id, macro_parent,
                                    operation_type="read",
                                    transformation=f"CALOC block size from {size_sym}")

                graph.add_assignment(alloc_node, "memory", "R1", "register",
                                     file, line_no, " ".join(operands) or opcode,
                                     scope, statement_id, macro_parent,
                                     operation_type=alloc_enrichment.get("operation_type", "call"),
                                     initialization_method=alloc_enrichment.get("initialization_method", "macro"),
                                     conditional_context=alloc_enrichment.get("conditional_context"),
                                     transformation=alloc_enrichment.get("transformation",
                                                                          "Return heap address in R1"))

                self.register_sources["R1"] = {(alloc_node, "memory")}
                graph.add_definition("R1", "register", file, line_no, opcode,
                                     scope, statement_id, macro_parent,
                                     operation_type=alloc_enrichment.get("operation_type", "call"),
                                     initialization_method=alloc_enrichment.get("initialization_method", "macro"),
                                     conditional_context=alloc_enrichment.get("conditional_context"),
                                     transformation=alloc_enrichment.get("transformation",
                                                                          "Define R1 from CALOC"))
                continue

            # RELCC/RELFC release boundary: track released pointer lineage.
            if opcode in ("RELCC", "RELFC"):
                release_node = f"free@{scope or '(global)'}:{line_no}"
                release_enrichment = self._get_enrichment(release_node, file, line_no)
                graph.add_definition(release_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                   scope, statement_id, macro_parent,
                                   operation_type=release_enrichment.get("operation_type", "call"),
                                   initialization_method=release_enrichment.get("initialization_method", "macro"),
                                   conditional_context=release_enrichment.get("conditional_context"),
                                   transformation=release_enrichment.get("transformation", f"Create {opcode} release boundary"))

                for idx, arg_operand in enumerate(operands, start=1):
                    arg_value = self._macro_operand_value(arg_operand)
                    if not arg_value or arg_value.startswith("&"):
                        continue
                    arg_raw, arg_kind, arg_index, arg_base = self._normalize_operand_full(arg_value)
                    arg_symbol = self._qualify_memory_access(arg_raw, arg_kind, arg_index, arg_base)
                    if not arg_symbol:
                        continue

                    arg_node = f"{release_node}:ARG{idx}"
                    arg_enrichment = self._get_enrichment(arg_symbol, file, line_no)
                    graph.add_assignment(arg_symbol, arg_kind, arg_node, "parameter",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=arg_enrichment.get("operation_type", "call"),
                                       initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                       conditional_context=arg_enrichment.get("conditional_context"),
                                       transformation=arg_enrichment.get("transformation", f"Pass {arg_symbol} to {opcode}"))

                    if arg_kind == "register":
                        for src, src_kind in self.register_sources.get(arg_symbol, set()):
                            graph.add_assignment(src, src_kind, arg_node, "parameter",
                                               file, line_no, " ".join(operands) or opcode,
                                               scope, statement_id, macro_parent,
                                               operation_type=arg_enrichment.get("operation_type", "call"),
                                               initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                               conditional_context=arg_enrichment.get("conditional_context"),
                                               transformation=arg_enrichment.get("transformation", f"Pass {src} to {opcode}"))

                    freed_node = f"freed@{scope or '(global)'}:{line_no}:{idx}"
                    graph.add_assignment(arg_node, "parameter", freed_node, "clobbered",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=arg_enrichment.get("operation_type", "call"),
                                       initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                       conditional_context=arg_enrichment.get("conditional_context"),
                                       transformation=arg_enrichment.get("transformation", f"Release pointer in {opcode}"))
                    graph.add_assignment(arg_node, "parameter", release_node, "parameter",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=arg_enrichment.get("operation_type", "call"),
                                       initialization_method=arg_enrichment.get("initialization_method", "macro"),
                                       conditional_context=arg_enrichment.get("conditional_context"),
                                       transformation=arg_enrichment.get("transformation", "Bind release argument"))

                graph.add_usage(release_node, "parameter", file, line_no, " ".join(operands) or opcode,
                                scope, statement_id, macro_parent,
                                operation_type=release_enrichment.get("operation_type", "call"),
                                initialization_method=release_enrichment.get("initialization_method", "macro"),
                                conditional_context=release_enrichment.get("conditional_context"),
                                transformation=release_enrichment.get("transformation", f"Release block via {opcode}"))
                continue

            # LEVTA boundary: read level-state and branch condition source.
            if opcode == "LEVTA":
                level_number = self._extract_macro_level_number(operands)
                levta_enrichment = self._get_enrichment("LEVTA", file, line_no)

                if level_number is not None:
                    level_name = self._macro_level_name(level_number)
                    status_node = f"levta@{level_name}:{line_no}"
                    graph.add_definition(level_name, "memory", file, line_no, opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=levta_enrichment.get("operation_type", "compare"),
                                       initialization_method=levta_enrichment.get("initialization_method", "macro"),
                                       conditional_context=levta_enrichment.get("conditional_context"),
                                       transformation=levta_enrichment.get("transformation", f"Reference level {level_name}"))
                    graph.add_usage(level_name, "memory", file, line_no, " ".join(operands) or opcode,
                                    scope, statement_id, macro_parent,
                                    operation_type=levta_enrichment.get("operation_type", "compare"),
                                    initialization_method=levta_enrichment.get("initialization_method", "macro"),
                                    conditional_context=levta_enrichment.get("conditional_context"),
                                    transformation=levta_enrichment.get("transformation", f"Test active state of {level_name}"))
                    graph.add_assignment(level_name, "memory", status_node, "parameter",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=levta_enrichment.get("operation_type", "compare"),
                                       initialization_method=levta_enrichment.get("initialization_method", "macro"),
                                       conditional_context=levta_enrichment.get("conditional_context"),
                                       transformation=levta_enrichment.get("transformation", f"Compute LEVTA status for {level_name}"))
                    graph.add_assignment(status_node, "parameter", "R15", "register",
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=levta_enrichment.get("operation_type", "compare"),
                                       initialization_method=levta_enrichment.get("initialization_method", "macro"),
                                       conditional_context=levta_enrichment.get("conditional_context"),
                                       transformation=levta_enrichment.get("transformation", "Set LEVTA return code in R15"))
                    self.register_sources["R15"] = {(status_node, "parameter"), (level_name, "memory")}
                    graph.add_definition("R15", "register", file, line_no, opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=levta_enrichment.get("operation_type", "compare"),
                                       initialization_method=levta_enrichment.get("initialization_method", "macro"),
                                       conditional_context=levta_enrichment.get("conditional_context"),
                                       transformation=levta_enrichment.get("transformation", "Define R15 from LEVTA"))

                notused_target = self._extract_macro_keyword_value(operands, {"NOTUSED"})
                if notused_target and self._looks_like_label_token(notused_target):
                    branch_enrichment = self._get_enrichment(notused_target, file, line_no)
                    graph.add_usage(notused_target, "memory", file, line_no, " ".join(operands) or opcode,
                                    scope, statement_id, macro_parent,
                                    operation_type=branch_enrichment.get("operation_type", "branch"),
                                    initialization_method=branch_enrichment.get("initialization_method"),
                                    conditional_context=branch_enrichment.get("conditional_context"),
                                    transformation=branch_enrichment.get("transformation", f"Branch target when level inactive: {notused_target}"))
                continue

            # PNTRP/PNTRC pointer-transfer boundary.
            if opcode in ("PNTRP", "PNTRC"):
                src_operand, dst_operand = self._extract_pointer_transfer_operands(operands)
                if src_operand and dst_operand:
                    src_raw, src_kind, src_index, src_base = self._normalize_operand_full(src_operand)
                    dst_raw, dst_kind, dst_index, dst_base = self._normalize_operand_full(dst_operand)
                    src_symbol = self._qualify_memory_access(src_raw, src_kind, src_index, src_base)
                    dst_symbol = self._qualify_memory_access(dst_raw, dst_kind, dst_index, dst_base)

                    self._add_index_base_usage(src_index, src_base, graph, file, line_no,
                                              opcode, scope, statement_id, macro_parent,
                                              memory_symbol=src_symbol if src_kind == "memory" else None)
                    self._add_index_base_usage(dst_index, dst_base, graph, file, line_no,
                                              opcode, scope, statement_id, macro_parent,
                                              memory_symbol=dst_symbol if dst_kind == "memory" else None)

                    transfer_enrichment = self._get_enrichment(dst_symbol, file, line_no)
                    graph.add_alias(src_symbol, src_kind, dst_symbol, dst_kind,
                                    file, line_no, " ".join(operands) or opcode,
                                    scope, statement_id, macro_parent)
                    graph.add_assignment(src_symbol, src_kind, dst_symbol, dst_kind,
                                       file, line_no, " ".join(operands) or opcode,
                                       scope, statement_id, macro_parent,
                                       operation_type=transfer_enrichment.get("operation_type", "move"),
                                       initialization_method=transfer_enrichment.get("initialization_method", "macro"),
                                       conditional_context=transfer_enrichment.get("conditional_context"),
                                       transformation=transfer_enrichment.get("transformation", f"Transfer pointer {src_symbol} -> {dst_symbol} via {opcode}"))

                    transfer_sources = {(src_symbol, src_kind)}
                    if src_kind == "register":
                        transfer_sources = set(self.register_sources.get(src_symbol, {(src_symbol, "register")}))

                    for source_name, source_kind in transfer_sources:
                        graph.add_assignment(source_name, source_kind, dst_symbol, dst_kind,
                                           file, line_no, " ".join(operands) or opcode,
                                           scope, statement_id, macro_parent,
                                           operation_type=transfer_enrichment.get("operation_type", "move"),
                                           initialization_method=transfer_enrichment.get("initialization_method", "macro"),
                                           conditional_context=transfer_enrichment.get("conditional_context"),
                                           transformation=transfer_enrichment.get("transformation", f"Propagate pointer source {source_name} to {dst_symbol}"))

                    if dst_kind == "register":
                        self.register_sources[dst_symbol] = transfer_sources
                        graph.add_definition(dst_symbol, "register", file, line_no, opcode,
                                           scope, statement_id, macro_parent,
                                           operation_type=transfer_enrichment.get("operation_type", "move"),
                                           initialization_method=transfer_enrichment.get("initialization_method", "macro"),
                                           conditional_context=transfer_enrichment.get("conditional_context"),
                                           transformation=transfer_enrichment.get("transformation", f"Define {dst_symbol} via {opcode}"))
                    else:
                        graph.add_definition(dst_symbol, dst_kind, file, line_no, opcode,
                                           scope, statement_id, macro_parent,
                                           operation_type=transfer_enrichment.get("operation_type", "move"),
                                           initialization_method=transfer_enrichment.get("initialization_method", "macro"),
                                           conditional_context=transfer_enrichment.get("conditional_context"),
                                           transformation=transfer_enrichment.get("transformation", f"Update {dst_symbol} via {opcode}"))

                    graph.add_usage(src_symbol, src_kind, file, line_no, " ".join(operands) or opcode,
                                    scope, statement_id, macro_parent,
                                    operation_type=transfer_enrichment.get("operation_type", "move"),
                                    initialization_method=transfer_enrichment.get("initialization_method", "macro"),
                                    conditional_context=transfer_enrichment.get("conditional_context"),
                                    transformation=transfer_enrichment.get("transformation", f"Read source pointer {src_symbol}"))
                    continue

            # DETAC boundary: model explicit detach handoff as parameter lineage.
            if opcode == "DETAC":
                detach_node = f"detach@{scope or '(global)'}:{line_no}"
                detach_enrichment = self._get_enrichment(detach_node, file, line_no)
                graph.add_definition(detach_node, "parameter", file, line_no, " ".join(operands) or "DETAC",
                                   scope, statement_id, macro_parent,
                                   operation_type=detach_enrichment.get("operation_type", "call"),
                                   initialization_method=detach_enrichment.get("initialization_method"),
                                   conditional_context=detach_enrichment.get("conditional_context"),
                                   transformation=detach_enrichment.get("transformation", "Create DETAC boundary"))

                for idx, arg_operand in enumerate(operands, start=1):
                    arg_value = self._macro_operand_value(arg_operand)
                    if not arg_value or arg_value.startswith("&"):
                        continue

                    arg_symbol, arg_kind, _, _ = self._normalize_operand_full(arg_value)
                    if not arg_symbol:
                        continue

                    arg_node = f"{detach_node}:ARG{idx}"
                    arg_enrichment = self._get_enrichment(arg_symbol, file, line_no)
                    graph.add_assignment(arg_symbol, arg_kind, arg_node, "parameter",
                                       file, line_no, " ".join(operands) or "DETAC",
                                       scope, statement_id, macro_parent,
                                       operation_type=arg_enrichment.get("operation_type", "call"),
                                       initialization_method=arg_enrichment.get("initialization_method"),
                                       conditional_context=arg_enrichment.get("conditional_context"),
                                       transformation=arg_enrichment.get("transformation", f"Pass {arg_symbol} to DETAC"))

                    if arg_kind == "register":
                        for src, src_kind in self.register_sources.get(arg_symbol, set()):
                            graph.add_assignment(src, src_kind, arg_node, "parameter",
                                               file, line_no, " ".join(operands) or "DETAC",
                                               scope, statement_id, macro_parent,
                                               operation_type=arg_enrichment.get("operation_type", "call"),
                                               initialization_method=arg_enrichment.get("initialization_method"),
                                               conditional_context=arg_enrichment.get("conditional_context"),
                                               transformation=arg_enrichment.get("transformation", f"Pass {src} to DETAC"))

                    graph.add_assignment(arg_node, "parameter", detach_node, "parameter",
                                       file, line_no, " ".join(operands) or "DETAC",
                                       scope, statement_id, macro_parent,
                                       operation_type=arg_enrichment.get("operation_type", "call"),
                                       initialization_method=arg_enrichment.get("initialization_method"),
                                       conditional_context=arg_enrichment.get("conditional_context"),
                                       transformation=arg_enrichment.get("transformation", "Bind DETAC argument"))

                graph.add_usage(detach_node, "parameter", file, line_no, " ".join(operands) or "DETAC",
                                scope, statement_id, macro_parent,
                                operation_type=detach_enrichment.get("operation_type", "call"),
                                initialization_method=detach_enrichment.get("initialization_method"),
                                conditional_context=detach_enrichment.get("conditional_context"),
                                transformation=detach_enrichment.get("transformation", "Detach attached ECB/task"))
                continue

            # Indirect register dispatch: BR Rx / BCR mask,Rx where the branch target is
            # held in a register at runtime (R14/R10 returns are excluded — handled separately).
            # Emit an indirect_call@ annotation node so the lineage graph records that the
            # register value flows into an unknown call target at this point.
            _indirect_reg = None
            if opcode.upper() == "BR" and operands:
                _br_reg = operands[0].strip().upper()
                if _br_reg not in ("R14", "14", "R10", "10"):
                    _indirect_reg = _br_reg
            elif opcode.upper() == "BCR" and len(operands) >= 2:
                _br_mask = operands[0].strip()
                _br_reg = operands[1].strip().upper()
                if _br_mask != "0" and _br_reg not in ("R14", "14", "R10", "10"):
                    _indirect_reg = _br_reg
            if _indirect_reg:
                ic_node = f"indirect_call@{_indirect_reg}:{line_no}"
                ic_enrichment = self._get_enrichment(ic_node, file, line_no)
                graph.add_definition(ic_node, "call_target", file, line_no,
                                     " ".join(operands) or opcode,
                                     scope, statement_id, macro_parent,
                                     operation_type="indirect_call",
                                     transformation=f"Indirect dispatch via {_indirect_reg}")
                norm_reg = self._normalize_register_operand(_indirect_reg) or _indirect_reg
                graph.add_usage(norm_reg, "register", file, line_no, opcode,
                                scope, statement_id, macro_parent,
                                operation_type="indirect_call",
                                transformation=f"Use {norm_reg} as indirect call target")
                for src, src_kind in self.register_sources.get(norm_reg, set()):
                    graph.add_assignment(src, src_kind, ic_node, "call_target",
                                         file, line_no, " ".join(operands),
                                         scope, statement_id, macro_parent,
                                         operation_type="indirect_call",
                                         transformation=f"Indirect dispatch: {src} \u2192 {ic_node}")
                continue

            # Call instructions with return value tracking
            CALL_INSTRUCTIONS = ("BAL", "BALR", "BAS", "BASR", "BRAS", "BRASL", "ENTRC", "ENTNC", "CREEC", "ATTAC", "ATTC")
            if opcode in CALL_INSTRUCTIONS and len(operands) >= 1:
                return_reg = "R14"
                target = operands[-1]
                if opcode in ("ENTRC", "ENTNC"):
                    extracted_target = self._extract_entrc_target(operands)
                    if not extracted_target:
                        continue
                    target = extracted_target
                elif opcode == "CREEC":
                    extracted_target = self._extract_creec_target(operands)
                    if not extracted_target:
                        continue
                    target = extracted_target
                elif opcode in ("ATTAC", "ATTC"):
                    extracted_target = self._extract_attac_target(operands)
                    if not extracted_target:
                        continue
                    target = extracted_target
                elif len(operands) >= 2:
                    return_reg = operands[0].upper()  # R14 typically
                    target = operands[1]

                # Resolve target symbol
                callee_name = self._resolve_call_target(target)

                # Get enrichment for return register
                enrichment = self._get_enrichment(return_reg, file, line_no)

                # Model z/TPF ENTRC/ENTNC calling convention boundary:
                # R0/R1 are common argument registers and R15 carries the call result/address context.
                if opcode in ("ENTRC", "ENTNC"):
                    _ALWAYS_ARG_REGS = {"R0", "R1", "R15"}
                    _GP_REGS = {f"R{i}" for i in range(16)}
                    live_regs = {r for r, srcs in self.register_sources.items()
                                 if r in _GP_REGS and srcs}
                    for arg_reg in sorted(_ALWAYS_ARG_REGS | live_regs):
                        arg_node = f"arg@{callee_name}:{arg_reg}"
                        arg_enrichment = self._get_enrichment(arg_reg, file, line_no)
                        graph.add_assignment(arg_reg, "register", arg_node, "parameter",
                                           file, line_no, " ".join(operands),
                                           scope, statement_id, macro_parent,
                                           operation_type=arg_enrichment.get("operation_type", "call"),
                                           initialization_method=arg_enrichment.get("initialization_method"),
                                           conditional_context=arg_enrichment.get("conditional_context"),
                                           transformation=arg_enrichment.get("transformation", f"Pass {arg_reg} to {callee_name}"))
                        for src, src_kind in self.register_sources.get(arg_reg, set()):
                            graph.add_assignment(src, src_kind, arg_node, "parameter",
                                               file, line_no, " ".join(operands),
                                               scope, statement_id, macro_parent,
                                               operation_type=arg_enrichment.get("operation_type", "call"),
                                               initialization_method=arg_enrichment.get("initialization_method"),
                                               conditional_context=arg_enrichment.get("conditional_context"),
                                               transformation=arg_enrichment.get("transformation", f"Pass {src} to {callee_name}"))

                # No-return transfer patterns: ENTNC tail-transfer and async spawns.
                if opcode in ("ENTNC", "CREEC", "ATTAC", "ATTC"):
                    if opcode in ("CREEC", "ATTAC", "ATTC"):
                        # Model explicit spawn/attach parameter handoff operands.
                        for idx, arg_operand in enumerate(operands[1:], start=1):
                            arg_value = self._macro_operand_value(arg_operand)
                            if not arg_value or arg_value.startswith("&"):
                                continue
                            arg_symbol, arg_kind, _, _ = self._normalize_operand_full(arg_value)
                            if not arg_symbol:
                                continue
                            arg_node = (
                                f"arg@{callee_name}:SPAWN{idx}"
                                if opcode == "CREEC"
                                else f"arg@{callee_name}:ATTACH{idx}"
                            )
                            arg_enrichment = self._get_enrichment(arg_symbol, file, line_no)
                            graph.add_assignment(arg_symbol, arg_kind, arg_node, "parameter",
                                               file, line_no, " ".join(operands),
                                               scope, statement_id, macro_parent,
                                               operation_type=arg_enrichment.get("operation_type", "call"),
                                               initialization_method=arg_enrichment.get("initialization_method"),
                                               conditional_context=arg_enrichment.get("conditional_context"),
                                               transformation=arg_enrichment.get(
                                                   "transformation",
                                                   (
                                                       f"Pass {arg_symbol} to spawned ECB {callee_name}"
                                                       if opcode == "CREEC"
                                                       else f"Pass {arg_symbol} to attached ECB/task {callee_name}"
                                                   ),
                                               ))
                            if arg_kind == "register":
                                for src, src_kind in self.register_sources.get(arg_symbol, set()):
                                    graph.add_assignment(src, src_kind, arg_node, "parameter",
                                                       file, line_no, " ".join(operands),
                                                       scope, statement_id, macro_parent,
                                                       operation_type=arg_enrichment.get("operation_type", "call"),
                                                       initialization_method=arg_enrichment.get("initialization_method"),
                                                       conditional_context=arg_enrichment.get("conditional_context"),
                                                       transformation=arg_enrichment.get(
                                                           "transformation",
                                                           (
                                                               f"Pass {src} to spawned ECB {callee_name}"
                                                               if opcode == "CREEC"
                                                               else f"Pass {src} to attached ECB/task {callee_name}"
                                                           ),
                                                       ))
                    graph.add_usage(callee_name, "function", file, line_no, " ".join(operands),
                                    scope, statement_id, macro_parent,
                                    operation_type=enrichment.get("operation_type", "call"),
                                    initialization_method=enrichment.get("initialization_method"),
                                    conditional_context=enrichment.get("conditional_context"),
                                    transformation=enrichment.get(
                                        "transformation",
                                        (
                                            f"Spawn {callee_name} in new ECB (no return)"
                                            if opcode == "CREEC"
                                            else (
                                                f"Attach and dispatch {callee_name} (no return)"
                                                if opcode in ("ATTAC", "ATTC")
                                                else f"Call {callee_name} (no return)"
                                            )
                                        ),
                                    ))
                    continue

                # Create a virtual return node for the callee
                return_node = f"return@{callee_name}"
                # Link return value to the return register
                graph.add_assignment(return_node, "return_value", return_reg, "register",
                                   file, line_no, " ".join(operands),
                                   scope, statement_id, macro_parent,
                                   operation_type=enrichment.get("operation_type", "call"),
                                   initialization_method=enrichment.get("initialization_method"),
                                   conditional_context=enrichment.get("conditional_context"),
                                   transformation=enrichment.get("transformation", f"Return from {callee_name} to {return_reg}"))

                # Update register sources
                self.register_sources[return_reg] = {(return_node, "return_value")}

                # Record definition of return register
                graph.add_definition(return_reg, "register", file, line_no,
                                   f"{opcode} (return)", scope, statement_id, macro_parent,
                                   operation_type=enrichment.get("operation_type", "call"),
                                   initialization_method=enrichment.get("initialization_method"),
                                   conditional_context=enrichment.get("conditional_context"),
                                   transformation=enrichment.get("transformation", f"Define {return_reg} from call return"))

                # Mark registers R15, R0, R1 as potentially defined (common return value registers)
                for ret_val_reg in ["R15", "R0", "R1"]:
                    ret_node = f"return@{callee_name}:{ret_val_reg}"
                    ret_enrichment = self._get_enrichment(ret_val_reg, file, line_no)
                    graph.add_assignment(ret_node, "return_value", ret_val_reg, "register",
                                       file, line_no, " ".join(operands),
                                       scope, statement_id, macro_parent,
                                       operation_type=ret_enrichment.get("operation_type", "call"),
                                       initialization_method=ret_enrichment.get("initialization_method"),
                                       conditional_context=ret_enrichment.get("conditional_context"),
                                       transformation=ret_enrichment.get("transformation", f"Return from {callee_name} to {ret_val_reg}"))
                    self.register_sources[ret_val_reg] = {(ret_node, "return_value")}
                    graph.add_definition(ret_val_reg, "register", file, line_no,
                                       f"{opcode} (return)", scope, statement_id, macro_parent,
                                       operation_type=ret_enrichment.get("operation_type", "call"),
                                       initialization_method=ret_enrichment.get("initialization_method"),
                                       conditional_context=ret_enrichment.get("conditional_context"),
                                       transformation=ret_enrichment.get("transformation", f"Define {ret_val_reg} from call return"))

                # Clobber work registers R2 (not already marked as return value)
                for reg in ["R2"]:
                    if reg != return_reg:
                        clobber_node = f"clobbered@{scope}:{line_no}:{reg}"
                        clobber_enrichment = self._get_enrichment(reg, file, line_no)
                        graph.add_assignment(clobber_node, "clobbered", reg, "register",
                                           file, line_no, f"{opcode} (clobber)",
                                           scope, statement_id, macro_parent,
                                           operation_type=clobber_enrichment.get("operation_type", "call"),
                                           initialization_method=clobber_enrichment.get("initialization_method"),
                                           conditional_context=clobber_enrichment.get("conditional_context"),
                                           transformation=clobber_enrichment.get("transformation", f"Clobber {reg} in call"))
                        # Preserve return value sources if already set, otherwise mark clobbered
                        if reg not in self.register_sources or not any("return" in s[1] for s in self.register_sources[reg]):
                            self.register_sources[reg] = {(clobber_node, "clobbered")}
                        graph.add_definition(reg, "register", file, line_no,
                                           f"{opcode} (clobber)", scope, statement_id, macro_parent,
                                           operation_type=clobber_enrichment.get("operation_type", "call"),
                                           initialization_method=clobber_enrichment.get("initialization_method"),
                                           conditional_context=clobber_enrichment.get("conditional_context"),
                                           transformation=clobber_enrichment.get("transformation", f"Define {reg} as clobbered"))

            # Generic use tracking
            for operand in operands:
                reg = self._normalize_register_operand(operand)
                if reg:
                    enrichment = self._get_enrichment(reg, file, line_no)
                    graph.add_usage(reg, "register", file, line_no, opcode, scope, statement_id, macro_parent,
                                  operation_type=enrichment.get("operation_type"),
                                  initialization_method=enrichment.get("initialization_method"),
                                  conditional_context=enrichment.get("conditional_context"),
                                  transformation=enrichment.get("transformation"))
                    if reg in self.register_sources:
                        for src, src_kind in self.register_sources[reg]:
                            graph.add_usage(src, src_kind, file, line_no, opcode,
                                            scope, statement_id, macro_parent,
                                            operation_type=enrichment.get("operation_type"),
                                            initialization_method=enrichment.get("initialization_method"),
                                            conditional_context=enrichment.get("conditional_context"),
                                            transformation=enrichment.get("transformation"))

            # Track branch target labels as usages so symbolic control-flow
            # targets remain visible in lineage (legacy compatibility + debugging).
            if opcode.startswith("B") and opcode not in CALL_INSTRUCTIONS and operands:
                branch_target = operands[-1].strip().rstrip(",")
                if branch_target and not self._normalize_register_operand(branch_target):
                    if branch_target not in ("*",) and not branch_target.startswith("="):
                        if self._looks_like_label_token(branch_target):
                            self._legacy_register_label_candidates.add(branch_target)
                        enrichment = self._get_enrichment(branch_target, file, line_no)
                        graph.add_usage(branch_target, "memory", file, line_no, opcode,
                                        scope, statement_id, macro_parent,
                                        operation_type=enrichment.get("operation_type", "branch"),
                                        initialization_method=enrichment.get("initialization_method"),
                                        conditional_context=enrichment.get("conditional_context"),
                                        transformation=enrichment.get("transformation", f"Branch to {branch_target}"))

        # Post-pass: enrich memory-access edges with DSECT context (dsect, ce_slot, field_offset, access_identity)
        self._enrich_edges_with_dsect_context(graph, _edge_count_at_start, _line_reg_ce_snapshots)

        # Store DSECT field offsets in context for blueprint emission
        if self.dsect_field_offsets:
            existing = context.get_metadata("dsect_field_offsets") or {}
            existing.update(self.dsect_field_offsets)
            context.add_metadata("dsect_field_offsets", existing)

        # Store per-line USING + register CE-slot snapshots for line_metadata emission
        if _line_reg_ce_snapshots:
            using_snapshots = {}
            for (snap_file, snap_line), reg_bindings in _line_reg_ce_snapshots.items():
                active_usings = {}
                register_ce_slots = {}
                for reg, (dsect_name, ce) in reg_bindings.items():
                    active_usings[reg] = dsect_name
                    if ce:
                        register_ce_slots[reg] = ce
                using_snapshots[str(snap_line)] = {
                    "active_usings": active_usings,
                    "register_ce_slots": register_ce_slots,
                }
            existing_snaps = context.get_metadata("line_using_snapshots") or {}
            existing_snaps.update(using_snapshots)
            context.add_metadata("line_using_snapshots", existing_snaps)

        # Emit register entry state: inherited_regs vs self_loaded_regs
        # Used by cross-file register liveness check to validate no_match_carried.
        if _inherited_regs_list or _self_loaded_regs_list:
            context.add_metadata("register_entry_state", {
                "inherited_regs": sorted(set(_inherited_regs_list)),
                "self_loaded_regs": sorted(set(_self_loaded_regs_list)),
            })

        # Backward-compatibility pass:
        # mirror key memory/register node spellings that older outputs used
        # (e.g., 4(R13) -> 4, caller_param[0](R1) -> caller_param[0], R2 -> R2)).
        self._emit_legacy_operand_compat_edges(graph)

        # ASM Phase 9b: emit db_key_dep edges from DBOPN catalog entries.
        # Mirrors the C++ dfred handler (Phase 9) so key→record dependency is
        # visible in the lineage graph for assembler DB operations.
        #
        # Two passes:
        #  1. Build a file_record → [key_operands] map from DBOPN (CONTROL) entries.
        #  2. Emit key→file_record edges for every operation that has a known key
        #     (either its own, for DBOPN, or the DBOPN key for READ/WRITE/UPDATE/DELETE).
        _asm_db_catalog = context.get_metadata("db_operation_catalog")
        if _asm_db_catalog:
            # Pass 1: collect keys from DBOPN entries
            _dbopn_key_map: dict = {}
            for _op in _asm_db_catalog.operations:
                if _op.operation_type != DBOperationType.CONTROL:
                    continue
                if _op.file_record_identifier and _op.key_operands:
                    _dbopn_key_map.setdefault(_op.file_record_identifier, []).extend(
                        _k for _k in _op.key_operands if _k not in _dbopn_key_map.get(_op.file_record_identifier, [])
                    )

            # Pass 2: emit key→file_record db_key_dep edges
            for _op in _asm_db_catalog.operations:
                if not _op.file_record_identifier:
                    continue
                _file_rec = _op.file_record_identifier
                if _op.operation_type == DBOperationType.CONTROL:
                    _keys = _op.key_operands  # DBOPN carries its own key
                else:
                    # READ/WRITE/UPDATE/DELETE inherit the DBOPN key for the same file
                    _keys = _dbopn_key_map.get(_file_rec, [])
                for _key in _keys:
                    _sid = f"ASM_DBOPN_{_op.source_line}_KEY"
                    graph.add_assignment(
                        _key, "memory",
                        _file_rec, "memory",
                        _op.source_file, _op.source_line,
                        f"DBOPN key→record: {_key}→{_file_rec}",
                        _op.calling_routine, _sid, None,
                        operation_type="db_key_dep",
                        transformation=f"Key {_key} selected record {_file_rec}",
                    )

            # Pass 3: span-based field linkage.
            # For each DBOPN/DBCLS span, emit key→field db_key_dep edges for every
            # memory node whose name starts with the record's DSECT prefix and whose
            # defining/assign edge falls within the open→close line range.
            # This fills the gap between key→file_record (Pass 2) and the
            # field-level dependency that downstream tracer consumers need.
            _dbopn_spans: list = []
            for _op in _asm_db_catalog.operations:
                if (_op.operation_type == DBOperationType.CONTROL
                        and _op.macro_name.upper() == "DBOPN"
                        and _op.file_record_identifier
                        and _op.key_operands):
                    _dbopn_spans.append({
                        "file_rec": _op.file_record_identifier,
                        "keys": list(_op.key_operands),
                        "open_line": _op.source_line,
                        "close_line": None,
                        "src_file": _op.source_file,
                        "routine": _op.calling_routine,
                    })

            # Match each DBOPN to its first subsequent DBCLS for the same file_rec.
            _sorted_ctrl = sorted(
                (o for o in _asm_db_catalog.operations if o.operation_type == DBOperationType.CONTROL),
                key=lambda o: o.source_line,
            )
            for _op in _sorted_ctrl:
                if _op.macro_name.upper() != "DBCLS" or not _op.file_record_identifier:
                    continue
                for _span in _dbopn_spans:
                    if (_span["file_rec"] == _op.file_record_identifier
                            and _span["close_line"] is None
                            and _op.source_line > _span["open_line"]):
                        _span["close_line"] = _op.source_line
                        break

            if _dbopn_spans:
                _span_field_seen: set = set()
                for _span in _dbopn_spans:
                    _open = _span["open_line"]
                    # Use 300-line window as fallback when no matching DBCLS found.
                    _close = _span["close_line"] or (_open + 300)
                    # z/TPF convention: DSECT fields share a prefix with the file name.
                    # e.g., CR21AEFM → prefix "CR21"; fields are CR21GLRTTOD1, CR21KEY, …
                    _rec_prefix = _span["file_rec"][:4].upper()
                    _rec_upper = _span["file_rec"].upper()
                    _keys = _span["keys"]

                    # Build a set of known field names from any DSECT whose name
                    # matches the file record (exact) or shares its 4-char prefix.
                    # This is the authoritative source: if a field was declared in
                    # the matching DSECT block, we accept it regardless of naming.
                    # Covers implicit-base-register routines where edge.dsect is
                    # not set but the DSECT layout was parsed from this file.
                    _dsect_known_fields: set = set()
                    for _dsect_nm, _dsect_flds in self.dsect_field_offsets.items():
                        if (_dsect_nm.upper() == _rec_upper
                                or _dsect_nm.upper()[:4] == _rec_prefix):
                            _dsect_known_fields.update(
                                fn.upper() for fn in _dsect_flds
                            )

                    # Match condition 4 (CE1CR-slot correlation):
                    # DSECTs that map to the same CE1CR level-slot as a
                    # prefix-matched DSECT are semantically equivalent DB
                    # record overlays — e.g. a working-storage copy of
                    # CR21AEFM fields that shares CE1CR5 with the real DSECT.
                    # Including their fields closes the gap for implicit-base-
                    # register routines that access the record through an
                    # alternate DSECT name without an explicit USING statement.
                    _rec_ce_slots: set = set()
                    for _dsect_nm in list(self.dsect_field_offsets.keys()):
                        if (_dsect_nm.upper() == _rec_upper
                                or _dsect_nm.upper()[:4] == _rec_prefix):
                            _slot = self.dsect_to_ce_slot.get(_dsect_nm)
                            if _slot:
                                _rec_ce_slots.add(_slot.upper())
                    if _rec_ce_slots:
                        for _dsect_nm, _dsect_flds in self.dsect_field_offsets.items():
                            _slot = self.dsect_to_ce_slot.get(_dsect_nm)
                            if _slot and _slot.upper() in _rec_ce_slots:
                                _dsect_known_fields.update(
                                    fn.upper() for fn in _dsect_flds
                                )

                    # Match condition 5 (USING-map correlation for implicit base registers):
                    # A routine can establish a DSECT base register at its entry point via
                    # an explicit USING statement (or DSECT registration macro such as
                    # "LG1G1 REG=R1") without repeating the USING at every field access.
                    # When the active using_map contains a DSECT that wasn't matched by
                    # prefix (conditions 1–4), include its declared fields so that accesses
                    # in implicit-base-register routines are correctly linked to the DB span.
                    # This is only applied when conditions 1–4 left _dsect_known_fields
                    # empty to avoid broadening an already well-specified match set.
                    if not _dsect_known_fields:
                        for _using_dsect in self.using_map.values():
                            _using_flds = self.dsect_field_offsets.get(_using_dsect)
                            if _using_flds:
                                _dsect_known_fields.update(
                                    fn.upper() for fn in _using_flds
                                )

                    for _edge in graph.edges:
                        if _edge.relation not in ("assign", "define"):
                            continue
                        if not (_open < _edge.line <= _close):
                            continue
                        if not _edge.target.startswith("memory:"):
                            continue
                        _field_name = _edge.target[7:]  # strip "memory:"
                        # Five match conditions (any one is sufficient):
                        #  1. 4-char prefix heuristic.
                        #  2. edge.dsect explicitly names the record (or its prefix).
                        #  3. field is a declared member of the matching DSECT block.
                        #  4. CE1CR-slot correlation (alternate DSECT overlay).
                        #  5. USING-map field (implicit-base-register routines, above).
                        _edge_dsect = getattr(_edge, "dsect", None)
                        _matches_prefix = _field_name.upper().startswith(_rec_prefix)
                        _matches_dsect = bool(_edge_dsect) and (
                            _edge_dsect.upper() == _rec_upper
                            or _edge_dsect.upper()[:4] == _rec_prefix
                        )
                        _matches_known = _field_name.upper() in _dsect_known_fields
                        if not _matches_prefix and not _matches_dsect and not _matches_known:
                            continue
                        for _key in _keys:
                            _kf = (_key, _field_name)
                            if _kf in _span_field_seen:
                                continue
                            _span_field_seen.add(_kf)
                            _sid = f"ASM_DBSPAN_{_open}_{_edge.line}_FIELD"
                            graph.add_assignment(
                                _key, "memory",
                                _field_name, "memory",
                                _span["src_file"], _edge.line,
                                f"DBOPN span: {_key}→{_field_name}",
                                _span["routine"], _sid, None,
                                operation_type="db_key_dep",
                                transformation=f"Key {_key} selected field {_field_name}",
                            )

        _elapsed = time.perf_counter() - _t0
        if _elapsed >= _SLOW_FILE_THRESHOLD:
            _log.warning(
                "VariableLineageBuilder.process_assembly slow file (%.1fs): %s",
                _elapsed, context.file_path,
            )
        else:
            _log.debug(
                "VariableLineageBuilder.process_assembly done in %.3fs: %s",
                _elapsed, context.file_path,
            )

    def _normalize_operand_full(self, operand: str) -> tuple[str, str, Optional[str], Optional[str]]:
        """
        Parse operand and return (symbol, kind, index_reg, base_reg).

        Args:
            operand: Assembly operand string

        Returns:
            symbol: The memory symbol or displacement
            kind: "register", "memory", or "literal"
            index_reg: Index register if present (e.g., "R7") or None
            base_reg: Base register if present (e.g., "R13") or None
        """
        if self.literal_resolver.is_literal(operand):
            return operand.strip(), "literal", None, None

        cleaned = operand.strip()
        if cleaned.startswith("=V(") or cleaned.startswith("=A("):
            return cleaned, "literal", None, None

        index_reg = None
        base_reg = None
        displacement = None

        if "(" in cleaned:
            parts = cleaned.split("(", 1)
            displacement_token = parts[0].strip() if parts[0] else "0"
            displacement = displacement_token
            displacement_equ_symbol = None

            # Resolve EQU if applicable
            if displacement_token.upper() in self.equ_symbols:
                displacement_equ_symbol = displacement_token
                displacement = self.equ_symbols[displacement_token.upper()]

            # Parse (index,base) or (base)
            addr_part = parts[1].rstrip(")")
            if "," in addr_part:
                index_token, base_token = addr_part.split(",", 1)
                index_reg = self._normalize_register_token(index_token)
                base_reg = self._normalize_register_token(base_token)
            else:
                base_reg = self._normalize_register_token(addr_part)

            # Qualify with USING if base register has mapping
            if base_reg and base_reg in self.using_map:
                dsect = self.using_map[base_reg]
                # Qualify the displacement with DSECT
                symbol_disp = displacement_equ_symbol if displacement_equ_symbol else displacement
                symbol = f"{dsect}:{symbol_disp}" if symbol_disp and symbol_disp != "0" else dsect
                return symbol, "memory", index_reg, base_reg

            r1_param_symbol = self._r1_parameter_symbol(displacement, base_reg, index_reg)
            if r1_param_symbol:
                return r1_param_symbol, "memory", index_reg, base_reg

            if displacement_equ_symbol:
                return displacement_equ_symbol, "memory", index_reg, base_reg
            return displacement if displacement else "0", "memory", index_reg, base_reg

        register_operand = self._normalize_register_operand(cleaned)
        if register_operand:
            return register_operand, "register", None, None

        return cleaned, "memory", None, None

    def _resolve_dsect_context(self, var_name: str, base_reg: Optional[str] = None) -> Dict[str, any]:
        """Resolve DSECT base register context for a memory variable.

        Returns a dict with keys: dsect, base_reg, ce_slot, field_offset, access_identity.
        All values are None if the context cannot be resolved.
        """
        result: Dict[str, any] = {
            "dsect": None, "base_reg": None, "ce_slot": None,
            "field_offset": None, "access_identity": None,
        }

        # Determine the base register if not provided
        if base_reg is None:
            # Try to find a USING mapping for this variable by checking active DSECTs
            for reg, dsect in self.using_map.items():
                dsect_fields = self.dsect_field_offsets.get(dsect, {})
                # Check if var_name is a field in this DSECT
                field_key = var_name.upper()
                if field_key in dsect_fields:
                    base_reg = reg
                    break

        if base_reg is None:
            return result

        # Resolve DSECT from base register
        dsect = self.using_map.get(base_reg)
        if not dsect:
            return result
        result["dsect"] = dsect
        result["base_reg"] = base_reg

        # Resolve CE-slot from register sources (with transitive resolution)
        ce_slot = self._resolve_ce_slot_for_register(base_reg)
        if not ce_slot and dsect:
            ce_slot = self.dsect_to_ce_slot.get(dsect)
        if ce_slot:
            result["ce_slot"] = ce_slot

        # Resolve field offset from DSECT field offsets
        field_key = var_name.upper()
        dsect_fields = self.dsect_field_offsets.get(dsect, {})
        if field_key in dsect_fields:
            offset, length = dsect_fields[field_key]
            result["field_offset"] = offset
        else:
            # Try stripping DSECT qualifier prefix (e.g., "LG1G1:LG1OSI" → "LG1OSI")
            if ":" in field_key:
                bare_name = field_key.split(":", 1)[1]
                if bare_name in dsect_fields:
                    offset, length = dsect_fields[bare_name]
                    result["field_offset"] = offset

        # Build access_identity
        if result["ce_slot"] and result["field_offset"] is not None:
            result["access_identity"] = f"{result['ce_slot']}:{result['field_offset']}"

        return result

    def _resolve_ce_slot_for_register(self, reg: str, max_depth: int = 5) -> Optional[str]:
        """Resolve a register back to its CE1CR slot source, following transitive loads.

        Handles chains like: R5 ← LR R5,R3 ← L R3,CE1CR5 → returns "CE1CR5".

        DETERMINISM: ``register_sources`` values are sets of (src, kind)
        tuples; set iteration order is hash-randomized per process, so
        "first match wins" made the result flip between runs whenever a
        register had both a CE1CR source and a register-chain source
        (observed as run-to-run register_ce_slots diffs in line_metadata).
        Resolution is now order-independent: any direct CE1CR source wins
        over register-chasing, and ties break by sorted order.
        """
        visited = set()
        current = reg
        for _ in range(max_depth):
            if current in visited:
                break
            visited.add(current)
            sources = self.register_sources.get(current, set())
            ce_hits = sorted(
                src.upper() for src, _kind in sources
                if src.upper().startswith("CE1CR")
            )
            if ce_hits:
                return ce_hits[0]
            # Follow register-to-register chains (sorted for determinism)
            next_regs = sorted(
                src.upper() for src, kind in sources
                if kind == "register" and src.upper() != current
            )
            if not next_regs:
                break
            current = next_regs[0]
        return None

    def _enrich_edge_with_dsect_context(self, edge, var_name: str, base_reg: Optional[str] = None):
        """Attach DSECT context (dsect, base_reg, ce_slot, field_offset, access_identity) to an edge."""
        ctx = self._resolve_dsect_context(var_name, base_reg)
        if ctx["dsect"]:
            edge.dsect = ctx["dsect"]
        if ctx["base_reg"]:
            edge.base_reg = ctx["base_reg"]
        if ctx["ce_slot"]:
            edge.ce_slot = ctx["ce_slot"]
        if ctx["field_offset"] is not None:
            edge.field_offset = ctx["field_offset"]
        if ctx["access_identity"]:
            edge.access_identity = ctx["access_identity"]

    def _enrich_edges_with_dsect_context(
        self,
        graph,
        start_index: int,
        line_reg_ce_snapshots: Dict[tuple, Dict[str, Optional[str]]],
    ):
        """Post-pass: enrich newly added edges with DSECT context from node metadata and snapshots.

        For each edge added during process_assembly (from start_index onward):
        - Look up the source/target node's byte_offset and cpp_struct metadata
        - Look up the CE-slot from the most recent register→CE-slot snapshot at or before the edge's line
        - Compute access_identity = "CE-slot:offset"
        """
        # Build sorted snapshot index per file for efficient "most recent at or before" lookup
        file_snap_lines: Dict[str, List[int]] = {}
        for (snap_file, snap_line) in line_reg_ce_snapshots:
            file_snap_lines.setdefault(snap_file, []).append(snap_line)
        for lines_list in file_snap_lines.values():
            lines_list.sort()

        import bisect

        def _find_nearest_snapshot(edge_file: str, edge_line: int):
            """Find the most recent snapshot at or before edge_line in the same file."""
            snap_lines = file_snap_lines.get(edge_file)
            if not snap_lines:
                return None
            idx = bisect.bisect_right(snap_lines, edge_line) - 1
            if idx < 0:
                return None
            nearest_line = snap_lines[idx]
            return line_reg_ce_snapshots.get((edge_file, nearest_line))

        for edge in graph.edges[start_index:]:
            # Only enrich memory-related edges (skip register-only, literal-only)
            for node_id in (edge.source, edge.target):
                node = graph.nodes.get(node_id)
                if not node or node.kind not in ("memory", "variable"):
                    continue
                meta = node.metadata or {}

                # Resolve DSECT name from:
                # 1. Node metadata cpp_struct (set during DSECT DS/DC parsing)
                # 2. USING-qualified node name pattern "DSECT:FIELD"
                dsect = meta.get("cpp_struct")
                byte_offset = meta.get("byte_offset")
                field_name = None

                if dsect is None and ":" in node.name:
                    # USING-qualified name like "LG1G1:LG1OSI"
                    parts = node.name.split(":", 1)
                    dsect = parts[0]
                    field_name = parts[1]
                    # Look up byte offset from dsect_field_offsets
                    # Try exact match first, then with &A suffix (macro parameter convention)
                    if byte_offset is None and field_name:
                        for dsect_key in (dsect, f"{dsect}&A", f"{dsect}&B"):
                            fields = self.dsect_field_offsets.get(dsect_key, {})
                            entry = fields.get(field_name) or fields.get(f"{field_name}&A")
                            if entry:
                                byte_offset = entry[0]
                                break

                # If field_name is a pure numeric displacement, use it directly as byte_offset
                if byte_offset is None and field_name is not None:
                    try:
                        byte_offset = int(field_name)
                    except ValueError:
                        pass

                if dsect is None and byte_offset is None:
                    continue

                # Set dsect and field_offset from resolved values
                if dsect is not None and edge.dsect is None:
                    edge.dsect = dsect
                if byte_offset is not None and edge.field_offset is None:
                    edge.field_offset = byte_offset

                # Set symbolic_identity (L2: DSECT:FIELD_NAME) when field name is symbolic
                if edge.symbolic_identity is None and dsect and field_name:
                    try:
                        int(field_name)  # numeric displacement → L3 only, not L2
                    except ValueError:
                        edge.symbolic_identity = f"{dsect}:{field_name}"

                # Resolve CE-slot from the nearest snapshot at or before this edge's line
                if edge.ce_slot is None and dsect:
                    snap = _find_nearest_snapshot(edge.file, edge.line)
                    if snap:
                        for reg, (snap_dsect, ce) in snap.items():
                            if snap_dsect == dsect and ce:
                                edge.base_reg = reg
                                edge.ce_slot = ce
                                break

                # Fallback: resolve CE-slot from cross-file dsect_to_ce_slot mapping
                if edge.ce_slot is None and edge.dsect:
                    _ce_fallback = self.dsect_to_ce_slot.get(edge.dsect)
                    if _ce_fallback:
                        edge.ce_slot = _ce_fallback

                # Compute access_identity
                if edge.ce_slot and edge.field_offset is not None and edge.access_identity is None:
                    edge.access_identity = f"{edge.ce_slot}:{edge.field_offset}"

                # Only enrich from the first matching node (prefer target for assignments)
                if edge.dsect:
                    break

    def _legacy_variants_for_memory_name(self, name: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Return legacy-compatible memory/register spellings for qualified memory names."""
        if not name:
            return [], []
        match = re.fullmatch(r"(.+)\(([^()]+)\)", name)
        if not match:
            return [], []

        displacement, regs_text = match.groups()
        displacement = displacement.strip()
        regs = [token.strip().upper() for token in regs_text.split(",") if token.strip()]

        memory_variants: List[str] = []
        register_aliases: List[Tuple[str, str]] = []

        if displacement:
            memory_variants.append(displacement)

        # De-duplicate while preserving order.
        dedup_memory: List[str] = []
        seen = set()
        for variant in memory_variants:
            if variant == name or variant in seen:
                continue
            seen.add(variant)
            dedup_memory.append(variant)
        return dedup_memory, register_aliases

    def _emit_legacy_operand_compat_edges(self, graph: VariableLineageGraph) -> None:
        """Emit additive compatibility edges for legacy operand node spellings."""
        # Deduplicate by core edge identity + provenance tuple.
        seen: Set[Tuple[str, str, str, str, int, Optional[str], Optional[str], Optional[str], Optional[str]]] = {
            (
                edge.source,
                edge.target,
                edge.relation,
                edge.file,
                edge.line,
                edge.scope,
                edge.statement_id,
                edge.expression,
                edge.macro_expansion_parent,
            )
            for edge in graph.edges
        }

        def add_definition(name: str, edge) -> None:
            node_id = f"memory:{name}"
            key = (
                node_id, node_id, "define",
                edge.file, edge.line, edge.scope, edge.statement_id, edge.expression, edge.macro_expansion_parent
            )
            if key in seen:
                return
            graph.add_definition(
                name, "memory",
                edge.file, edge.line,
                edge.expression, edge.scope, edge.statement_id, edge.macro_expansion_parent
            )
            seen.add(key)

        def add_assignment(src_name: str, src_kind: str, dst_name: str, dst_kind: str, edge) -> None:
            src_id = f"{src_kind}:{src_name}"
            dst_id = f"{dst_kind}:{dst_name}"
            key = (
                src_id, dst_id, "assign",
                edge.file, edge.line, edge.scope, edge.statement_id, edge.expression, edge.macro_expansion_parent
            )
            if key in seen:
                return
            graph.add_assignment(
                src_name, src_kind, dst_name, dst_kind,
                edge.file, edge.line, edge.expression, edge.scope, edge.statement_id, edge.macro_expansion_parent
            )
            seen.add(key)

        def add_usage(name: str, kind: str, edge) -> None:
            src_id = f"{kind}:{name}"
            use_id = f"use:use@{edge.file}:{edge.line}"
            key = (
                src_id, use_id, "use",
                edge.file, edge.line, edge.scope, edge.statement_id, edge.expression, edge.macro_expansion_parent
            )
            if key in seen:
                return
            graph.add_usage(
                name, kind,
                edge.file, edge.line, edge.expression, edge.scope, edge.statement_id, edge.macro_expansion_parent
            )
            seen.add(key)

        def add_alias(src_name: str, src_kind: str, dst_name: str, dst_kind: str, edge) -> None:
            src_id = f"{src_kind}:{src_name}"
            dst_id = f"{dst_kind}:{dst_name}"
            key = (
                src_id, dst_id, "alias",
                edge.file, edge.line, edge.scope, edge.statement_id, edge.expression, edge.macro_expansion_parent
            )
            if key in seen:
                return
            graph.add_alias(
                src_name, src_kind, dst_name, dst_kind,
                edge.file, edge.line, edge.expression, edge.scope, edge.statement_id, edge.macro_expansion_parent
            )
            seen.add(key)

        edges_snapshot = list(graph.edges)
        for edge in edges_snapshot:
            if ":" not in edge.source or ":" not in edge.target:
                continue
            src_kind, src_name = edge.source.split(":", 1)
            dst_kind, dst_name = edge.target.split(":", 1)

            src_memory_variants: List[str] = []
            dst_memory_variants: List[str] = []
            src_register_aliases: List[Tuple[str, str]] = []
            dst_register_aliases: List[Tuple[str, str]] = []

            if src_kind == "memory":
                src_memory_variants, src_register_aliases = self._legacy_variants_for_memory_name(src_name)
            if dst_kind == "memory":
                dst_memory_variants, dst_register_aliases = self._legacy_variants_for_memory_name(dst_name)

            if edge.relation == "define":
                for variant in src_memory_variants:
                    add_definition(variant, edge)
                if src_kind == "memory" and self._is_symbolic_register_name(src_name):
                    graph.add_definition(src_name, "register", edge.file, edge.line, edge.expression,
                                         edge.scope, edge.statement_id, edge.macro_expansion_parent)
            elif edge.relation == "assign":
                # Legacy compatibility: older outputs represented numeric address
                # constants as memory nodes in some LA-style assignments.
                if src_kind == "literal" and re.fullmatch(r"[+-]?\d+", src_name):
                    add_assignment(src_name, "memory", dst_name, dst_kind, edge)
                # Older outputs also represented =F'/=C'/=X' immediates as
                # memory:=... sources on assignment edges.
                if src_kind == "literal" and src_name.startswith("="):
                    add_assignment(src_name, "memory", dst_name, dst_kind, edge)
                for variant in src_memory_variants:
                    add_assignment(variant, "memory", dst_name, dst_kind, edge)
                for variant in dst_memory_variants:
                    add_assignment(src_name, src_kind, variant, "memory", edge)
                    if src_kind == "literal" and src_name.startswith("="):
                        add_assignment(src_name, "memory", variant, "memory", edge)
                if src_kind == "memory" and self._is_symbolic_register_name(src_name):
                    add_assignment(src_name, "register", dst_name, dst_kind, edge)
                    for variant in dst_memory_variants:
                        add_assignment(src_name, "register", variant, "memory", edge)
                if dst_kind == "memory" and self._is_symbolic_register_name(dst_name):
                    add_assignment(src_name, src_kind, dst_name, "register", edge)
            elif edge.relation == "use":
                for variant in src_memory_variants:
                    add_usage(variant, "memory", edge)
                if src_kind == "memory" and self._is_symbolic_register_name(src_name):
                    add_usage(src_name, "register", edge)
            elif edge.relation == "alias":
                for variant in src_memory_variants:
                    add_alias(variant, "memory", dst_name, dst_kind, edge)
                for variant in dst_memory_variants:
                    add_alias(src_name, src_kind, variant, "memory", edge)
                for src_variant in src_memory_variants:
                    for dst_variant in dst_memory_variants:
                        add_alias(src_variant, "memory", dst_variant, "memory", edge)
                if src_kind == "memory" and self._is_symbolic_register_name(src_name):
                    add_alias(src_name, "register", dst_name, dst_kind, edge)
                if dst_kind == "memory" and self._is_symbolic_register_name(dst_name):
                    add_alias(src_name, src_kind, dst_name, "register", edge)

            for src_reg, legacy_reg in src_register_aliases + dst_register_aliases:
                add_alias(src_reg, "register", legacy_reg, "register", edge)
                add_alias(legacy_reg, "register", src_reg, "register", edge)
            if src_kind == "memory" and self._is_symbolic_register_name(src_name):
                add_alias(src_name, "memory", src_name, "register", edge)
                add_alias(src_name, "register", src_name, "memory", edge)
            if dst_kind == "memory" and self._is_symbolic_register_name(dst_name):
                add_alias(dst_name, "memory", dst_name, "register", edge)
                add_alias(dst_name, "register", dst_name, "memory", edge)

    def _normalize_operand(self, operand: str) -> tuple[str, str]:
        """Backward compatible version that returns (symbol, kind)."""
        symbol, kind, _, _ = self._normalize_operand_full(operand)
        return symbol, kind

    def _qualify_memory_access(self, symbol: str, kind: str,
                               index_reg: Optional[str], base_reg: Optional[str]) -> str:
        """Return a stable memory-access node name preserving index/base addressing."""
        if kind != "memory":
            return symbol
        if not index_reg and not base_reg:
            return symbol

        regs = []
        if index_reg:
            regs.append(index_reg)
        if base_reg:
            regs.append(base_reg)
        return f"{symbol}({','.join(regs)})"

    def _replace_standalone_token(self, text: str, old: str, new: str) -> str:
        """Replace a standalone symbol token without touching larger identifiers."""
        if not text:
            return text
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])")
        return pattern.sub(new, text)

    def _emit_equ_memory_aliases(self, graph, operand_text: str,
                                 normalized_symbol: str, kind: str,
                                 index_reg: Optional[str], base_reg: Optional[str],
                                 qualified_symbol: str,
                                 file: str, line_no: int, expression: str,
                                 scope: Optional[str], statement_id: Optional[str],
                                 macro_parent: Optional[str]) -> None:
        """Emit symbolic<->numeric aliases for EQU displacement-based memory operands."""
        if kind != "memory":
            return
        operand = (operand_text or "").strip()
        if "(" not in operand:
            return

        displacement_token = operand.split("(", 1)[0].strip()
        if not displacement_token:
            return

        symbol_token = displacement_token.upper()
        if symbol_token not in self.equ_symbols:
            return

        resolved = self._resolve_equ_value(symbol_token).strip().upper()
        if not re.fullmatch(r"[+-]?\d+", resolved):
            return
        numeric_token = str(int(resolved, 10))

        symbolic_access = self._qualify_memory_access(symbol_token, "memory", index_reg, base_reg)
        numeric_access = self._qualify_memory_access(numeric_token, "memory", index_reg, base_reg)
        normalized_numeric = self._replace_standalone_token(normalized_symbol, symbol_token, numeric_token)
        normalized_numeric_access = self._qualify_memory_access(
            normalized_numeric, "memory", index_reg, base_reg
        )

        alias_pairs = (
            (symbol_token, numeric_token),
            (symbolic_access, numeric_access),
            (symbolic_access, qualified_symbol),
            (qualified_symbol, normalized_numeric_access),
        )
        for source_name, target_name in alias_pairs:
            if not source_name or not target_name or source_name == target_name:
                continue
            alias_key = (source_name, target_name)
            if alias_key in self._emitted_memory_aliases:
                continue
            graph.add_alias(source_name, "memory", target_name, "memory",
                            file, line_no, expression,
                            scope, statement_id, macro_parent)
            self._emitted_memory_aliases.add(alias_key)

    def _add_index_base_usage(self, index_reg: Optional[str], base_reg: Optional[str], graph,
                             file, line_no, opcode, scope, statement_id, macro_parent,
                             memory_symbol: Optional[str] = None):
        """Add usage edges for index/base registers and address-derivation lineage."""
        def _record_register_usage(reg: str, role: str) -> None:
            graph.add_usage(reg, "register", file, line_no,
                            f"{opcode} ({role})", scope, statement_id, macro_parent)
            for src, src_kind in self.register_sources.get(reg, set()):
                graph.add_usage(src, src_kind, file, line_no,
                                f"{opcode} ({role})", scope, statement_id, macro_parent)

            # Link address contributors into the effective-memory-access node.
            if memory_symbol:
                graph.add_assignment(reg, "register", memory_symbol, "memory",
                                     file, line_no, f"{opcode} (address)",
                                     scope, statement_id, macro_parent)
                for src, src_kind in self.register_sources.get(reg, set()):
                    if src_kind == "memory" and src == memory_symbol:
                        continue
                    graph.add_assignment(src, src_kind, memory_symbol, "memory",
                                         file, line_no, f"{opcode} (address)",
                                         scope, statement_id, macro_parent)

        if index_reg and index_reg.startswith("R"):
            _record_register_usage(index_reg, "index")

        if base_reg and base_reg.startswith("R"):
            _record_register_usage(base_reg, "base")

    def process_c_family(self, unit: CTranslationUnit, context: AnalysisContext) -> None:
        _t0 = time.perf_counter()
        _log.debug("VariableLineageBuilder.process_c_family start: %s", context.file_path)
        graph = context.get_metadata("variable_lineage")
        if graph is None:
            graph = VariableLineageGraph()
            context.add_metadata("variable_lineage", graph)

        file = str(unit.file_path)
        alias_map: Dict[str, Set[str]] = {}
        _glob_ptr_map: Dict[str, str] = {}  # qualified ptr var → uppercase global name (Pattern E)
        return_sources_by_func: Dict[str, Set[str]] = {}
        symbol_aliases = getattr(unit, "symbol_aliases", {}) or {}
        function_aliases = getattr(unit, "function_aliases", {}) or {}
        numeric_literal_re = re.compile(
            r"^[+-]?(?:"
            r"0[xX][0-9A-Fa-f]+"
            r"|0[bB][01]+"
            r"|0[0-7]+"
            r"|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
            r")(?:[uUlLfF]+)?$"
        )

        def qualify(name: str, scope: Optional[str], function: Optional[str], storage: Optional[str]) -> str:
            if not name:
                return name
            stars = ""
            raw_name = name
            while raw_name.startswith("*"):
                stars += "*"
                raw_name = raw_name[1:]
            if scope in ("global", "extern"):
                return f"{stars}{file}::{raw_name}"
            if scope == "static":
                return f"{stars}{file}::static::{raw_name}"
            if scope == "local_static":
                func = function or "(global)"
                return f"{stars}{func}::static::{raw_name}"
            if function:
                if raw_name.isidentifier() or any(tok in raw_name for tok in ("->", ".", "[")):
                    return f"{stars}{function}::{raw_name}"
            return f"{stars}{raw_name}"

        def normalize_callable_name(name: Optional[str]) -> Optional[str]:
            if not name:
                return name
            normalized = name.strip()
            return normalized

        def asm_alias_for_callable(name: Optional[str]) -> Optional[str]:
            if not name:
                return None
            if name in symbol_aliases:
                return symbol_aliases[name]
            tail = name.split("::")[-1]
            return symbol_aliases.get(tail)

        def source_kind(name: str, aliases: Dict[str, Set[str]]) -> str:
            if not name:
                return "variable"
            if len(name) >= 2 and name[0] == name[-1] and name[0] in ("'", '"'):
                return "literal"
            if bool(numeric_literal_re.fullmatch(name.strip())):
                return "literal"
            if "[" in name and "]" in name:
                return "array_element"
            if "->" in name:
                return "struct_field"
            tail = name.split("::")[-1]
            if "." in tail:
                return "struct_field"
            if name.startswith("*") or name in aliases:
                return "memory"
            return "variable"

        def resolve_aliases(name: str, aliases: Dict[str, Set[str]], visited: Optional[Set[str]] = None) -> Set[str]:
            if not name:
                return set()
            if visited is None:
                visited = set()
            if name in visited:
                return {name}
            visited.add(name)
            roots = aliases.get(name)
            if not roots:
                return {name}
            resolved = {name}
            for root in roots:
                resolved.update(resolve_aliases(root, aliases, visited))
            return resolved

        def _discover_regs_slots(qarg: str) -> Set[int]:
            """Inspect already-built struct_field nodes to find which r<n>
            members of *qarg* the code actually touches. Returns the set of
            slot numbers (0..15). Catches both ``regs.rN`` and ``regs->rN``."""
            slots: Set[int] = set()
            if not qarg:
                return slots
            bases = {qarg, qarg.split("::")[-1]}
            for node in graph.nodes.values():
                if node.kind != "struct_field":
                    continue
                name = node.name
                for base in bases:
                    suffix = None
                    if name.startswith(f"{base}.r"):
                        suffix = name[len(base) + 2:]
                    elif name.startswith(f"{base}->r"):
                        suffix = name[len(base) + 3:]
                    if suffix and suffix.isdigit():
                        n = int(suffix)
                        if 0 <= n <= 15:
                            slots.add(n)
            return slots

        def model_tpf_regs_passthrough(callee: str, scope: str, line_no: int, expression: str,
                                       statement_id: str, qarg: str, include_input: bool = True,
                                       input_slots: Optional[Set[int]] = None,
                                       return_slots: Optional[Set[int]] = None) -> None:
            """Model TPF_regs convention across any register slot the caller
            actually wrote (input) or reads (return).

            Legacy behaviour (R1 input, R0/R2 return) is preserved when
            ``input_slots``/``return_slots`` aren't provided AND the graph
            has no ``regs.rN`` struct_field evidence — so existing traces
            are unchanged for pre-TPF_regs-aware corpora."""
            if include_input:
                slots = input_slots
                if slots is None:
                    slots = _discover_regs_slots(qarg) or {1}
                for slot in sorted(slots):
                    for form in (f"{qarg}.r{slot}", f"{qarg}->r{slot}"):
                        graph.add_assignment(form, "struct_field",
                                             f"arg@{callee}:R{slot}", "register",
                                             file, line_no, expression, scope, statement_id, None)
                        graph.add_usage(form, "struct_field", file, line_no,
                                        expression, scope, statement_id, None)

            ret_slots = return_slots
            if ret_slots is None:
                # Default return convention: R0 (return code), R2 (secondary
                # return), R15 (classic TPF status). Only emit return-slot
                # nodes for slots actually read on the regs object.
                discovered = _discover_regs_slots(qarg)
                ret_slots = {0, 2, 15} & discovered or {0, 2}
            for reg_num in sorted(ret_slots):
                ret_reg = f"return@{callee}:R{reg_num}"
                for form in (f"{qarg}.r{reg_num}", f"{qarg}->r{reg_num}"):
                    graph.add_assignment(ret_reg, "return_value", form, "struct_field",
                                         file, line_no, expression, scope, statement_id, None)

        def has_register_struct_fields(qarg: str) -> bool:
            """Detect register-shaped member fields (e.g. obj.r1 / obj->r1) for a handoff object."""
            if not qarg:
                return False
            candidates = {qarg, qarg.split("::")[-1]}
            for node in graph.nodes.values():
                if node.kind != "struct_field":
                    continue
                name = node.name
                for base in candidates:
                    reg_suffix = None
                    if name.startswith(f"{base}.r"):
                        reg_suffix = name[len(base) + 2:]
                    elif name.startswith(f"{base}->r"):
                        reg_suffix = name[len(base) + 3:]
                    if reg_suffix and reg_suffix.isdigit():
                        return True
            return False

        field_asm_aliases: Dict[str, str] = getattr(unit, "field_asm_aliases", {}) or {}
        enum_values: Dict[str, Dict[str, str]] = getattr(unit, "enum_values", {}) or {}

        def _strip_cv_ptr(type_text: str) -> str:
            if not type_text:
                return ""
            t = type_text.strip()
            for qual in ("const ", "volatile ", "mutable "):
                while t.startswith(qual):
                    t = t[len(qual):].lstrip()
            while t.endswith(("*", "&")):
                t = t[:-1].rstrip()
            for qual in (" const", " volatile"):
                while t.endswith(qual):
                    t = t[: -len(qual)].rstrip()
            return re.sub(r"\s+", " ", t).strip()

        type_by_qualified_var: Dict[str, str] = {}
        for decl in unit.declarations:
            if decl.kind not in ("variable", "parameter", "member"):
                continue
            if not decl.data_type:
                continue
            qname = qualify(decl.name, decl.scope, decl.function, decl.storage_class)
            if not qname:
                continue
            stripped = _strip_cv_ptr(decl.data_type)
            if stripped:
                type_by_qualified_var.setdefault(qname, stripped)

        # Create declaration entry points (including declarations from headers when analyzed).
        for decl in unit.declarations:
            qualified_name = qualify(decl.name, decl.scope, decl.function, decl.storage_class)
            if not qualified_name:
                continue
            decl_scope = decl.function or "(global)"
            statement_id = f"C_{decl_scope}_{decl.line}_DECL"
            graph.add_definition(qualified_name, decl.kind, file, decl.line, decl.expression or "declaration",
                                 decl_scope, statement_id, None)
            node_metadata: Dict[str, object] = {}
            if decl.declaration_type:
                node_metadata["declaration_type"] = decl.declaration_type
            if decl.data_type:
                node_metadata["data_type"] = decl.data_type
            if decl.storage_class:
                node_metadata["storage_class"] = decl.storage_class
            if decl.initial_value:
                node_metadata["initial_value"] = decl.initial_value
            if decl.related_symbols:
                node_metadata["related_symbols"] = list(dict.fromkeys(decl.related_symbols))
            if decl.aliased_as:
                node_metadata["aliased_as"] = list(dict.fromkeys(decl.aliased_as))
            if decl.scope:
                node_metadata["scope"] = decl.scope
            if decl.function:
                node_metadata["enclosing_function"] = decl.function
            if getattr(decl, "metadata", None):
                for key, value in decl.metadata.items():
                    if value is not None:
                        node_metadata[key] = value
            # T12: emit canonical_name for file-scope variable nodes so that
            # VariableNameResolver Case D can resolve pure global variable
            # renames across translation units via canonical identity matching.
            if (decl.kind == "variable"
                    and not decl.function
                    and decl.scope in ("global", "extern", "static")):
                node_metadata["canonical_name"] = f"{Path(file).stem}::{decl.name}"
                node_metadata["is_file_scope"] = True
            if node_metadata:
                graph.annotate_node(qualified_name, decl.kind, **node_metadata)
            for src in decl.initializer_sources:
                src_candidates = resolve_aliases(src, alias_map)
                for candidate in src_candidates:
                    qsrc = qualify(candidate, "local" if decl.function else "global", decl.function, None)
                    graph.add_assignment(qsrc, source_kind(candidate, alias_map), qualified_name, decl.kind,
                                         file, decl.line, decl.expression, decl_scope, statement_id, None)

        # Bridge ASM ENTRC-style argument registers to C/C++ parameter declarations.
        params_by_function: Dict[str, List[str]] = {}
        for decl in unit.declarations:
            if decl.kind != "parameter" or not decl.function:
                continue
            qparam = qualify(decl.name, decl.scope, decl.function, decl.storage_class)
            if not qparam:
                continue
            items = params_by_function.setdefault(decl.function, [])
            if qparam not in items:
                items.append(qparam)

        for func_name, params in params_by_function.items():
            # z/TPF ABI passes up to 12 arguments in R0–R11.  The previous
            # hard cap of R0–R3 left parameters 4+ without a callee-side
            # receiver node, breaking cross-file lineage for any function
            # with more than 4 arguments (Pattern A fix).
            for idx, qparam in enumerate(params):
                if idx >= 12:
                    continue  # beyond R11; no register slot to bridge
                reg = f"R{idx}"
                arg_node = f"arg@{func_name}:{reg}"
                graph.add_assignment(arg_node, "parameter", qparam, "parameter",
                                     file, 0, "c_family_call_boundary",
                                     func_name, f"C_{func_name}_ARG_BRIDGE", None)

        for ret in unit.returns:
            func = ret.function or "(global)"
            return_key = f"return@{func}"
            for source in ret.sources:
                if source:
                    is_deref = source.startswith("*")
                    base_source = source.lstrip("*") if is_deref else source
                    qsource = qualify(base_source, "local" if ret.function else "global", ret.function, None)
                    resolved = resolve_aliases(qsource, alias_map)
                    for candidate in resolved:
                        edge_source = f"{'*' * (len(source) - len(base_source))}{candidate}" if is_deref else candidate
                        skind = "memory" if is_deref else source_kind(candidate, alias_map)
                        graph.add_assignment(edge_source, skind, return_key, "return_value",
                                             file, ret.line, ret.expression, func,
                                             f"C_{func}_{ret.line}", None)
            graph.add_definition(return_key, "return_value", file, ret.line, ret.expression or "return",
                                 func, f"C_{func}_{ret.line}", None)
            return_sources_by_func.setdefault(func, set()).update([s for s in ret.sources if s])
            # Mirror variable node for compatibility with older traces.
            graph.add_definition(return_key, "variable", file, ret.line, ret.expression or "return",
                                 func, f"C_{func}_{ret.line}", None)
            alias = function_aliases.get(func)
            if alias and alias != func:
                alias_return = f"return@{alias}"
                graph.add_definition(alias_return, "return_value", file, ret.line, ret.expression or "return",
                                     func, f"C_{func}_{ret.line}", None)
                graph.add_definition(alias_return, "variable", file, ret.line, ret.expression or "return",
                                     func, f"C_{func}_{ret.line}", None)
                graph.add_assignment(return_key, "return_value", alias_return, "return_value",
                                     file, ret.line, ret.expression or "return", func,
                                     f"C_{func}_{ret.line}", None)
                graph.add_assignment(alias_return, "return_value", return_key, "return_value",
                                     file, ret.line, ret.expression or "return", func,
                                     f"C_{func}_{ret.line}", None)

        literal_bindings: Dict[str, str] = {}
        emitted_arg_bind_keys: Set[Tuple[str, str, str, int, str, Optional[str]]] = set()

        def emit_call_arg_bindings(callee: str, callee_asm: Optional[str], args: List[str],
                                   scope: str, function: Optional[str], line: int,
                                   expression: Optional[str], statement_id: str,
                                   address_of_args: Optional[List[str]] = None,
                                   conditional_context: Optional[List[str]] = None) -> None:
            """Emit call argument bindings for C/C++ callsites in a generic way."""
            def is_asm_like_callable(name: Optional[str]) -> bool:
                if not name:
                    return False
                tail = name.split("::")[-1]
                if tail in symbol_aliases or name in symbol_aliases:
                    return True
                if tail.lower().startswith("asm_"):
                    return True
                if tail.isupper() and ("_" in tail or any(ch.isdigit() for ch in tail)):
                    return True
                return False

            bridge_callee = callee_asm or (callee if is_asm_like_callable(callee) else None)
            qualified_address_of: Set[str] = set()
            if address_of_args:
                for addr_arg in address_of_args:
                    if not addr_arg:
                        continue
                    q_addr = qualify(addr_arg, "local" if function else "global", function, None)
                    qualified_address_of.add(q_addr)
                    qualified_address_of.add(q_addr.split("::")[-1])

            def bind_to_call_arg(source_name: str, source_kind_name: str, call_arg_name: str) -> None:
                bind_key = (source_name, call_arg_name, scope, line, "arg_bind", None)
                if bind_key in emitted_arg_bind_keys:
                    return
                graph.add_arg_bind(
                    source_name, source_kind_name,
                    call_arg_name, "call_arg",
                    file, line, expression, scope,
                    statement_id, None,
                    conditional_context=conditional_context,
                )
                emitted_arg_bind_keys.add(bind_key)

            for idx, arg in enumerate(args):
                if not arg:
                    continue
                raw_arg = arg.strip()
                arg_deref = 0
                arg_address = 0
                while raw_arg and raw_arg[0] in ("*", "&"):
                    if raw_arg[0] == "*":
                        arg_deref += 1
                    else:
                        arg_address += 1
                    raw_arg = raw_arg[1:].strip()
                arg_base = raw_arg
                if not arg_base:
                    continue
                qarg = qualify(arg_base, "local" if function else "global", function, None)
                if arg_deref > 0:
                    edge_source = f"{'*' * arg_deref}{qarg}"
                    skind = "memory"
                else:
                    edge_source = qarg
                    skind = source_kind(qarg, alias_map)
                call_arg_name = f"{callee}#{idx}"
                graph.annotate_node(call_arg_name, "call_arg", arg_index=idx)
                bind_to_call_arg(edge_source, skind, call_arg_name)

                # When an object is handed off by address (e.g. &obj), bind
                # its known member/element nodes to the same call arg so
                # lineage can traverse object fields across call boundaries.
                qarg_tail = qarg.split("::")[-1]
                is_address_handoff = (
                    arg_address > 0
                    or qarg in qualified_address_of
                    or qarg_tail in qualified_address_of
                    or arg_base in qualified_address_of
                )
                if is_address_handoff:
                    candidate_bases = {qarg, qarg_tail}
                    register_slots: Set[int] = set()
                    for node in list(graph.nodes.values()):
                        if node.kind not in ("struct_field", "array_element"):
                            continue
                        node_name = node.name
                        matched = False
                        for base in candidate_bases:
                            if (
                                node_name.startswith(f"{base}.")
                                or node_name.startswith(f"{base}->")
                                or node_name.startswith(f"{base}[")
                            ):
                                matched = True
                                reg_suffix = None
                                if node_name.startswith(f"{base}.r"):
                                    reg_suffix = node_name[len(base) + 2:]
                                elif node_name.startswith(f"{base}->r"):
                                    reg_suffix = node_name[len(base) + 3:]
                                if reg_suffix and reg_suffix.isdigit():
                                    register_slots.add(int(reg_suffix))
                        if matched:
                            bind_to_call_arg(node_name, node.kind, call_arg_name)

                    # Bridge aggregate register-style handoff objects (e.g. TPF_regs*)
                    # to per-slot caller_param nodes so cross-language tracing can reach
                    # ASM loads like REGS_R4(R1) -> caller_param[4](R1).
                    if bridge_callee and register_slots:
                        for slot in sorted(register_slots):
                            for caller_param_name in (f"caller_param[{slot}]", f"caller_param[{slot}](R1)"):
                                bridge_key = (call_arg_name, caller_param_name, scope, line, "arg_bind", bridge_callee)
                                if bridge_key in emitted_arg_bind_keys:
                                    continue
                                graph.add_arg_bind(
                                    call_arg_name, "call_arg",
                                    caller_param_name, "memory",
                                    file, line, expression, scope,
                                    statement_id, None,
                                    callee=bridge_callee,
                                    conditional_context=conditional_context,
                                )
                                emitted_arg_bind_keys.add(bridge_key)

                if bridge_callee:
                    bridge_key = (call_arg_name, f"caller_param[{idx}]", scope, line, "arg_bind", bridge_callee)
                    if bridge_key not in emitted_arg_bind_keys:
                        graph.add_arg_bind(
                            call_arg_name, "call_arg",
                            f"caller_param[{idx}]", "memory",
                            file, line, expression, scope,
                            statement_id, None,
                            callee=bridge_callee,
                            conditional_context=conditional_context,
                        )
                        emitted_arg_bind_keys.add(bridge_key)

        for assignment in unit.assignments:
            target = qualify(assignment.target, assignment.scope, assignment.function, assignment.storage_class)
            target_kind = assignment.target_kind or "variable"
            scope = assignment.function or "(global)"
            if assignment.expression and "&" in assignment.expression and target_kind in ("variable", "memory", "struct_field", "array_element"):
                for source in assignment.sources:
                    qsource = qualify(source.lstrip("*"), assignment.scope, assignment.function, None)
                    alias_map.setdefault(target, set()).add(qsource)
                    graph.add_alias(qsource, source_kind(source, alias_map), target, target_kind,
                                    file, assignment.line, assignment.expression, scope,
                                    f"C_{scope}_{assignment.line}", None,
                                    conditional_context=assignment.conditional_context)

            # Track multi-level alias chains, e.g. p2 = p1.
            if target_kind == "variable" and assignment.sources:
                for source in assignment.sources:
                    qsource = qualify(source.lstrip("*"), assignment.scope, assignment.function, None)
                    if qsource in alias_map:
                        alias_map.setdefault(target, set()).update(alias_map[qsource])
                    if source and not source.startswith("*"):
                        graph.add_alias(qsource, source_kind(source, alias_map), target, target_kind,
                                        file, assignment.line, assignment.expression, scope,
                                        f"C_{scope}_{assignment.line}", None,
                                        conditional_context=assignment.conditional_context)

            for source in assignment.sources:
                if source:
                    quoted = self._extract_quoted_literal(source)
                    if quoted and target_kind in ("variable", "memory"):
                        literal_bindings[target] = quoted
                    deref_level = 0
                    while deref_level < len(source) and source[deref_level] == "*":
                        deref_level += 1
                    base_source = source[deref_level:]
                    qsource = qualify(base_source, assignment.scope, assignment.function, None)
                    resolved_sources = resolve_aliases(qsource, alias_map)
                    # Phase 5 — look up bare source identifier in the enum index.
                    # For scope-qualified names like "EnumClass::Value", also try
                    # the bare tail ("Value") since _extract_enum_values keys are
                    # stored without the qualifier prefix.
                    if enum_values:
                        enum_rec = enum_values.get(base_source)
                        if enum_rec is None and "::" in base_source:
                            enum_rec = enum_values.get(base_source.split("::")[-1])
                    else:
                        enum_rec = None
                    for resolved in resolved_sources:
                        if deref_level > 0:
                            edge_source = f"{'*' * deref_level}{resolved}"
                            skind = "memory"
                            # ASM-CPP Pattern D fix: deref-read from a glob()-aliased pointer.
                            # When the source is a dereference of a variable in _glob_ptr_map,
                            # the actual value is read from global:<NAME>; emit that alias edge
                            # so the cross-language read is visible in the lineage graph.
                            _gname_read = _glob_ptr_map.get(resolved)
                            if _gname_read and target_kind in ("variable", "memory", "struct_field"):
                                graph.add_assignment(
                                    _gname_read, "global",
                                    target, target_kind,
                                    file, assignment.line, assignment.expression,
                                    scope, f"C_{scope}_{assignment.line}", None,
                                    conditional_context=assignment.conditional_context,
                                )
                        else:
                            edge_source = resolved
                            skind = source_kind(resolved, alias_map)
                        graph.add_assignment(edge_source, skind, target, target_kind,
                                             file, assignment.line, assignment.expression, scope,
                                             f"C_{scope}_{assignment.line}", None,
                                             conditional_context=assignment.conditional_context)
                        # Pattern E fix part 2: propagate the write-value into global:NAME
                        # when target is a dereference of a glob()-aliased pointer variable.
                        if target_kind == "memory":
                            _base = target.lstrip("*")
                            _global_name = _glob_ptr_map.get(_base)
                            if _global_name:
                                graph.add_assignment(
                                    edge_source, skind,
                                    _global_name, "global",
                                    file, assignment.line, assignment.expression,
                                    scope, f"C_{scope}_{assignment.line}", None,
                                    conditional_context=assignment.conditional_context,
                                )
                        if enum_rec:
                            # Attach enum metadata to the edge we just emitted.
                            if graph.edges[-1].metadata is None:
                                graph.edges[-1].metadata = {}
                            graph.edges[-1].metadata["resolved_value"] = enum_rec["value"]
                            graph.edges[-1].metadata["enum_type"] = enum_rec["enum_type"]
            graph.add_definition(target, target_kind, file, assignment.line, assignment.expression, scope,
                                f"C_{scope}_{assignment.line}", None,
                                conditional_context=assignment.conditional_context)
            # Ternary condition annotation: create use edges for condition
            # variables in ternary expressions (cond ? a : b).  BFS treats
            # use edges as terminal — correct: condition is a guard, not a
            # data source to trace further.
            if assignment.conditional_context:
                for _cc_entry in assignment.conditional_context:
                    if _cc_entry.startswith("ternary(") and _cc_entry.endswith(")"):
                        _tern_expr = _cc_entry[8:-1]  # strip ternary(...)
                        # Extract identifiers from the ternary condition text
                        _tern_ids = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', _tern_expr)
                        for _tid in _tern_ids:
                            if _tid and _tid not in ("true", "false", "nullptr",
                                                      "NULL", "const", "volatile"):
                                _qtid = qualify(_tid, assignment.scope,
                                               assignment.function, None)
                                graph.add_usage(
                                    _qtid, source_kind(_qtid, alias_map),
                                    file, assignment.line,
                                    _tern_expr, scope,
                                    f"C_{scope}_{assignment.line}", None,
                                    conditional_context=assignment.conditional_context,
                                )
            if assignment.target and target != assignment.target and target_kind in ("struct_field", "array_element"):
                # Backward-compatibility alias so existing traces can still resolve unqualified member/index nodes.
                graph.add_definition(assignment.target, target_kind, file, assignment.line, assignment.expression, scope,
                                     f"C_{scope}_{assignment.line}", None,
                                     conditional_context=assignment.conditional_context)
                graph.add_assignment(target, target_kind, assignment.target, target_kind,
                                     file, assignment.line, assignment.expression, scope,
                                     f"C_{scope}_{assignment.line}", None,
                                     conditional_context=assignment.conditional_context)

            # Keep struct field hierarchy explicit.
            if target_kind == "struct_field" and assignment.target_base and assignment.target_field:
                base_q = qualify(assignment.target_base, assignment.scope, assignment.function, assignment.storage_class)
                graph.add_definition(base_q, "variable", file, assignment.line, assignment.expression, scope,
                                     f"C_{scope}_{assignment.line}", None,
                                     conditional_context=assignment.conditional_context)
                graph.add_assignment(base_q, "variable", target, target_kind,
                                     file, assignment.line, assignment.expression, scope,
                                     f"C_{scope}_{assignment.line}", None,
                                     conditional_context=assignment.conditional_context)

            if (
                target_kind == "struct_field"
                and assignment.target_base
                and assignment.target_field
                and field_asm_aliases
            ):
                base_q = qualify(assignment.target_base, assignment.scope,
                                 assignment.function, assignment.storage_class)
                type_name = type_by_qualified_var.get(base_q)
                if type_name:
                    asm_alias = field_asm_aliases.get(f"{type_name}.{assignment.target_field}")
                    if asm_alias:
                        graph.annotate_node(target, "struct_field",
                                            asm_alias=asm_alias,
                                            enclosing_type=type_name)
                        if assignment.target and assignment.target != target:
                            graph.annotate_node(assignment.target, "struct_field",
                                                asm_alias=asm_alias,
                                                enclosing_type=type_name)

            # Keep array element hierarchy explicit.
            if target_kind == "array_element" and assignment.target_base:
                base_q = qualify(assignment.target_base, assignment.scope, assignment.function, assignment.storage_class)
                graph.add_definition(base_q, "variable", file, assignment.line, assignment.expression, scope,
                                     f"C_{scope}_{assignment.line}", None,
                                     conditional_context=assignment.conditional_context)
                graph.add_assignment(base_q, "variable", target, target_kind,
                                     file, assignment.line, assignment.expression, scope,
                                     f"C_{scope}_{assignment.line}", None,
                                     conditional_context=assignment.conditional_context)

            # If assignment comes from a call, attach callee return lineage
            if assignment.call_function:
                callee = normalize_callable_name(assignment.call_function)
                callee_asm = assignment.call_asm_target or asm_alias_for_callable(callee)
                statement_id = f"C_{scope}_{assignment.line}"
                emit_call_arg_bindings(
                    callee=callee,
                    callee_asm=callee_asm,
                    args=assignment.call_args,
                    scope=scope,
                    function=assignment.function,
                    line=assignment.line,
                    expression=assignment.expression,
                    statement_id=statement_id,
                    address_of_args=assignment.call_address_of_args,
                    conditional_context=assignment.conditional_context,
                )

                return_key = f"return@{callee_asm or callee}"
                graph.add_assignment(return_key, "return_value", target, target_kind,
                                     file, assignment.line, assignment.expression, scope,
                                     statement_id, None,
                                     conditional_context=assignment.conditional_context)
                graph.add_assignment(return_key, "variable", target, target_kind,
                                     file, assignment.line, assignment.expression, scope,
                                     statement_id, None,
                                     conditional_context=assignment.conditional_context)
                for source in return_sources_by_func.get(callee, set()):
                    qsource = qualify(source, "local", callee, None)
                    graph.add_assignment(qsource, source_kind(source, alias_map), target, target_kind,
                                         file, assignment.line, assignment.expression, scope,
                                         statement_id, None,
                                         conditional_context=assignment.conditional_context)
                if callee_asm and callee_asm != callee:
                    graph.add_assignment(f"return@{callee}", "return_value", f"return@{callee_asm}", "return_value",
                                         file, assignment.line, assignment.expression, scope, statement_id, None,
                                         conditional_context=assignment.conditional_context)
                    graph.add_assignment(f"return@{callee_asm}", "return_value", f"return@{callee}", "return_value",
                                         file, assignment.line, assignment.expression, scope, statement_id, None,
                                         conditional_context=assignment.conditional_context)

                # Pattern D fix: glob(_NAME) → alias the assignment target
                # (local pointer variable) to the canonical global:NAME node so
                # cross-file lineage connects all readers of the same control byte.
                if callee and callee.lower() == "glob" and assignment.call_args:
                    _raw_gname = assignment.call_args[0].strip().lstrip("_")
                    _gname = _raw_gname.upper()
                    if _gname:
                        graph.add_alias(
                            target, target_kind,
                            _gname, "global",
                            file, assignment.line,
                            f"glob(_{_gname.lower()})",
                            scope, statement_id, None,
                            conditional_context=assignment.conditional_context,
                        )
                        # Pattern E fix part 1: record the pointer variable so the
                        # deref-write check below can resolve it to global:NAME.
                        if target_kind == "variable":
                            _glob_ptr_map[target] = _gname
                            # GAP-D: encode glob alias into node metadata so
                            # the stitcher can propagate it across TU boundaries.
                            graph.annotate_node(target, "variable", glob_alias=_gname)

                # For pointer/register-passthrough conventions (e.g., TPF_regs*), model
                # common return registers as potential side effects on address-of arguments.
                if assignment.call_address_of_args:
                    for arg in assignment.call_address_of_args:
                        qarg = qualify(arg, assignment.scope, assignment.function, None)
                        graph.add_assignment(return_key, "return_value", f"*{qarg}", "memory",
                                             file, assignment.line, assignment.expression, scope,
                                             statement_id, None,
                                             conditional_context=assignment.conditional_context)
                        if has_register_struct_fields(qarg):
                            model_tpf_regs_passthrough(
                                callee=callee_asm or callee,
                                scope=scope,
                                line_no=assignment.line,
                                expression=assignment.expression,
                                statement_id=statement_id,
                                qarg=qarg,
                                include_input=True,
                            )
                            # Calls returning into assignment target typically correspond to R0.
                            graph.add_assignment(f"return@{callee_asm or callee}:R0", "return_value", target, target_kind,
                                                 file, assignment.line, assignment.expression, scope,
                                                 statement_id, None,
                                                 conditional_context=assignment.conditional_context)

        # Model named-memory APIs that hand off data by runtime name.
        for call in unit.calls:
            callee = normalize_callable_name(call.function)
            if not callee or not call.args:
                continue
            lower = callee.lower()
            if lower not in ("heap_locate", "decb_create"):
                continue
            first_arg = call.args[0].strip()
            literal_arg = self._extract_quoted_literal(first_arg)
            if not literal_arg:
                q_first = qualify(first_arg, "local" if call.enclosing_function else "global", call.enclosing_function, None)
                literal_arg = literal_bindings.get(q_first)
            if not literal_arg:
                literal_arg = literal_bindings.get(first_arg)
            if not literal_arg and "::" in first_arg:
                literal_arg = literal_bindings.get(first_arg.split("::")[-1])
            if not literal_arg:
                suffix = f"::{first_arg}"
                for bound_name, bound_literal in literal_bindings.items():
                    if bound_name.endswith(suffix):
                        literal_arg = bound_literal
                        break
            if literal_arg:
                name_key = literal_arg[1:-1]
                named_node = f"named:{name_key}"
                call_scope = call.enclosing_function or "(global)"
                statement_id = f"C_{call_scope}_{call.line}_CALL"
                graph.add_definition(named_node, "memory", file, call.line, f"call {callee}",
                                     call_scope, statement_id, None,
                                     conditional_context=call.conditional_context)
                graph.add_definition(literal_arg, "literal", file, call.line, f"call {callee}",
                                     call_scope, statement_id, None,
                                     conditional_context=call.conditional_context)
                graph.add_assignment(literal_arg, "literal", named_node, "memory",
                                     file, call.line, f"call {callee}", call_scope, statement_id, None,
                                     conditional_context=call.conditional_context)
                graph.add_assignment(named_node, "memory", f"return@{callee}", "return_value",
                                     file, call.line, f"call {callee}", call_scope, statement_id, None,
                                     conditional_context=call.conditional_context)

        # Side effects for standalone calls with address-of arguments.
        for call in unit.calls:
            if not call.address_of_args:
                continue
            callee = normalize_callable_name(call.function)
            if not callee:
                continue
            callee_asm = call.asm_target or asm_alias_for_callable(callee)
            call_scope = call.enclosing_function or "(global)"
            return_key = f"return@{callee_asm or callee}"
            statement_id = f"C_{call_scope}_{call.line}_CALL"
            for arg in call.address_of_args:
                qarg = qualify(arg, "local" if call.enclosing_function else "global", call.enclosing_function, None)
                graph.add_assignment(return_key, "return_value", f"*{qarg}", "memory",
                                     file, call.line, f"call {callee}", call_scope, statement_id, None,
                                     conditional_context=call.conditional_context)
                if has_register_struct_fields(qarg):
                    model_tpf_regs_passthrough(
                        callee=callee_asm or callee,
                        scope=call_scope,
                        line_no=call.line,
                        expression=f"call {callee}",
                        statement_id=statement_id,
                        qarg=qarg,
                        include_input=True,
                    )

        # Emit argument bindings for call expressions that are not represented
        # through assignment.call_function metadata (e.g., standalone/return calls).
        for call in unit.calls:
            callee = normalize_callable_name(call.function)
            if not callee or not call.args:
                continue
            call_scope = call.enclosing_function or "(global)"
            statement_id = f"C_{call_scope}_{call.line}_CALL"
            callee_asm = call.asm_target or asm_alias_for_callable(callee)
            emit_call_arg_bindings(
                callee=callee,
                callee_asm=callee_asm,
                args=call.args,
                scope=call_scope,
                function=call.enclosing_function,
                line=call.line,
                expression=f"call {callee}",
                statement_id=statement_id,
                address_of_args=call.address_of_args,
                conditional_context=call.conditional_context,
            )

            # Pattern D fix (standalone context): glob(_NAME) call not captured
            # by an assignment → emit return@glob alias to global:NAME so the
            # cross-file global node is reachable even without a local variable.
            if callee.lower() == "glob" and call.args:
                _raw_gname = call.args[0].strip().lstrip("_")
                _gname = _raw_gname.upper()
                if _gname:
                    graph.add_alias(
                        f"return@glob", "return_value",
                        _gname, "global",
                        file, call.line,
                        f"glob(_{_gname.lower()})",
                        call_scope, statement_id, None,
                        conditional_context=call.conditional_context,
                    )

        # Track variable usage when no assignment edges were added
        for assignment in unit.assignments:
            usage_scope = assignment.function or "(global)"
            for source in assignment.sources:
                if source:
                    qsource = qualify(source.lstrip("*"), assignment.scope, assignment.function, None)
                    skind = "memory" if source.startswith("*") else source_kind(source, alias_map)
                    use_source = f"{'*' * (len(source) - len(source.lstrip('*')))}{qsource}" if source.startswith("*") else qsource
                    graph.add_usage(use_source, skind, file, assignment.line, assignment.expression, usage_scope,
                                    f"C_{usage_scope}_{assignment.line}", None,
                                    conditional_context=assignment.conditional_context)

        # Phase 5b — emit use-edges for each comparison-context operand so
        # enum identifiers appearing in `if (task == AddLimit)` end up in the
        # lineage graph with conditional_context populated and enum metadata
        # stamped when applicable.
        for cond in getattr(unit, "conditions", []) or []:
            cond_scope = cond.scope or cond.function or "(global)"
            statement_id = f"C_{cond_scope}_{cond.line}_cond"
            seen_operands: Set[str] = set()
            for operand in cond.operands or []:
                if not operand or operand in seen_operands:
                    continue
                seen_operands.add(operand)
                qoperand = qualify(operand, cond.scope, cond.function, None)
                graph.add_usage(
                    qoperand,
                    source_kind(operand, alias_map),
                    file,
                    cond.line,
                    cond.expression,
                    cond_scope,
                    statement_id,
                    None,
                    conditional_context=[cond.expression],
                )
                enum_rec = enum_values.get(operand) if enum_values else None
                if enum_rec and graph.edges:
                    if graph.edges[-1].metadata is None:
                        graph.edges[-1].metadata = {}
                    graph.edges[-1].metadata["resolved_value"] = enum_rec["value"]
                    graph.edges[-1].metadata["enum_type"] = enum_rec["enum_type"]

        # Phase 7 — materialize every glob(_name) reference as a kind="global"
        # lineage node so downstream tracers can "see" the system-wide global
        # exactly like a DSECT/memory variable. Per-call alias edges are now
        # emitted in the assignment / calls loops above (Pattern D fix).
        # The stitcher still creates the cross-language alias to the ASM side.
        for gname in getattr(unit, "globals_used", []) or []:
            if not gname:
                continue
            graph.add_definition(
                gname, "global", file, 0,
                f"glob(_{gname.lower()})",
                "system", f"GLOBAL_{gname}", None,
            )

        # Phase 8 — C++ getcc() level-identity nodes (Pattern F fix).
        #
        # ASM GETCC emits alloc@Dn:line → Dn → CE1CRn.  C++ getcc(D0, ...) is
        # treated as a generic call by the parser, losing the level/slot metadata
        # that downstream cross-language bridges depend on.  Reconstruct the same
        # three-node chain here so the existing stitch_ce1cr_level_bindings Phase-3
        # step can connect the C++ pointer to the ASM CE1CR node.
        _GETCC_LEVEL_RE = re.compile(r"^D([0-9A-Fa-f]+)$")
        for _asn in unit.assignments:
            if not _asn.call_function:
                continue
            if _asn.call_function.lower() not in ("getcc", "getfc"):
                continue
            if not _asn.call_args:
                continue
            _m = _GETCC_LEVEL_RE.fullmatch(_asn.call_args[0].strip())
            if not _m:
                continue
            _level_n = int(_m.group(1), 16)
            _level_hex = format(_level_n, "X")
            _alloc_node = f"alloc@D{_level_hex}:{_asn.line}"
            _level_name = f"D{_level_hex}"
            _ce1cr_slot = f"CE1CR{_level_hex}"
            _asn_scope = _asn.function or "(global)"
            _asn_sid = f"C_{_asn_scope}_{_asn.line}_GETCC"
            _expr = _asn.expression or f"getcc(D{_level_hex},...)"

            graph.add_definition(_alloc_node, "memory", file, _asn.line,
                                 _expr, _asn_scope, _asn_sid, None,
                                 operation_type="initialize",
                                 initialization_method="getcc",
                                 transformation=f"C++ getcc allocates level D{_level_hex}",
                                 conditional_context=_asn.conditional_context)
            graph.add_definition(_level_name, "memory", file, _asn.line,
                                 _expr, _asn_scope, _asn_sid, None,
                                 operation_type="initialize",
                                 initialization_method="getcc",
                                 transformation=f"Activate ECB level {_level_name}",
                                 conditional_context=_asn.conditional_context)
            graph.add_definition(_ce1cr_slot, "memory", file, _asn.line,
                                 _expr, _asn_scope, _asn_sid, None,
                                 operation_type="initialize",
                                 initialization_method="getcc",
                                 transformation=f"Set pointer slot {_ce1cr_slot}",
                                 conditional_context=_asn.conditional_context)
            graph.add_assignment(_alloc_node, "memory", _level_name, "memory",
                                  file, _asn.line, _expr, _asn_scope, _asn_sid, None,
                                  transformation=f"Bind allocation to {_level_name}",
                                  conditional_context=_asn.conditional_context)
            graph.add_assignment(_level_name, "memory", _ce1cr_slot, "memory",
                                  file, _asn.line, _expr, _asn_scope, _asn_sid, None,
                                  transformation=f"Link {_level_name} to {_ce1cr_slot}",
                                  conditional_context=_asn.conditional_context)
            # Alias: C++ return target ↔ CE1CR slot so stitcher can bridge.
            _tgt = qualify(_asn.target, _asn.scope, _asn.function, _asn.storage_class)
            _tgt_kind = _asn.target_kind or "variable"
            graph.add_alias(_ce1cr_slot, "memory", _tgt, _tgt_kind,
                            file, _asn.line, _expr, _asn_scope, _asn_sid, None,
                            conditional_context=_asn.conditional_context)
            graph.add_assignment("return@getcc", "return_value",
                                  _alloc_node, "memory",
                                  file, _asn.line, _expr, _asn_scope, _asn_sid, None,
                                  conditional_context=_asn.conditional_context)

        # Phase 9 — C++ z/TPFDF database key-dependency edges (Pattern E fix).
        #
        # The ASM db_operation_extractor recognises DBOPN/DBRED macros; the C++
        # equivalents (dfopn_acc / dfred) were treated as generic calls leaving
        # db_operations empty and key→field dependency invisible.
        #
        # Approach: (a) scan dfopn_acc assignments to build a
        # file_handle→(key_var, key_kind) map; (b) for each dfred assignment that
        # uses a known file_handle, emit a db_key_dep assignment edge from the key
        # variable to the LREC pointer; (c) populate a CPP DBOperationCatalog so
        # downstream consumers can list C++ database operations alongside ASM ones.
        _CPP_DFOPN = {"dfopn_acc", "dfopn"}
        _CPP_DFRED = {"dfred"}
        _CPP_DFMOD = {"dfmod"}
        _CPP_DFCLS = {"dfcls"}

        # Map: qualified file_handle_var → (key_var_name, key_kind, calling_func, line)
        _dfopn_map: Dict[str, tuple] = {}
        _cpp_db_catalog = DBOperationCatalog()

        for _asn in unit.assignments:
            if not _asn.call_function:
                continue
            _clow = _asn.call_function.lower()

            if _clow in _CPP_DFOPN:
                # dfopn_acc(file_name, mode, alg, hold, (dft_alg*)&packedKey)
                # Key argument is the last positional arg (index 4 for dfopn_acc).
                _args = _asn.call_args
                _key_raw = _args[-1].strip() if _args else ""
                # Strip address-of & cast wrappers: (dft_alg*)&packedKey → packedKey
                _key_raw = re.sub(r"^\(.*?\)\s*", "", _key_raw)  # strip cast
                _key_raw = _key_raw.lstrip("&")
                _key_var = qualify(_key_raw, _asn.scope, _asn.function, None) if _key_raw else ""
                _key_kind = source_kind(_key_var, alias_map) if _key_var else "variable"
                _file_rec = _args[0].strip().strip('"') if _args else ""
                _fh_var = qualify(_asn.target, _asn.scope, _asn.function, _asn.storage_class)
                if _fh_var and _key_var:
                    _dfopn_map[_fh_var] = (_key_var, _key_kind, _asn.function or "(global)", _asn.line)
                # Record in catalog
                _cpp_db_catalog.add_operation(DBOperation(
                    operation_type=DBOperationType.CONTROL,
                    macro_name="dfopn_acc",
                    calling_routine=_asn.function or "(global)",
                    file_record_identifier=_file_rec,
                    key_operands=[_key_raw] if _key_raw else [],
                    access_mode="keyed",
                    source_file=file,
                    source_line=_asn.line,
                    source_section=_asn.function,
                    confidence="high" if _file_rec else "low",
                ))

            elif _clow in _CPP_DFRED:
                # dfred(file_handle, mode) → lrec_ptr
                _args = _asn.call_args
                _fh_raw = _args[0].strip() if _args else ""
                _fh_qvar = qualify(_fh_raw, _asn.scope, _asn.function, None)
                _entry = _dfopn_map.get(_fh_qvar)
                if not _entry:
                    # Try unqualified fallback
                    _entry = _dfopn_map.get(_fh_raw)
                if _entry:
                    _key_var, _key_kind, _calling_func, _opn_line = _entry
                    _lrec_var = qualify(_asn.target, _asn.scope, _asn.function, _asn.storage_class)
                    _lrec_kind = _asn.target_kind or "variable"
                    _dfrd_scope = _asn.function or "(global)"
                    _dfrd_sid = f"C_{_dfrd_scope}_{_asn.line}_DFRED_KEY"
                    graph.add_assignment(
                        _key_var, _key_kind,
                        _lrec_var, _lrec_kind,
                        file, _asn.line,
                        _asn.expression or f"dfred(key_dep)",
                        _dfrd_scope, _dfrd_sid, None,
                        operation_type="db_key_dep",
                        transformation=f"Key {_key_var} selected record {_lrec_var}",
                        conditional_context=_asn.conditional_context,
                    )
                # Record in catalog
                _cpp_db_catalog.add_operation(DBOperation(
                    operation_type=DBOperationType.READ,
                    macro_name="dfred",
                    calling_routine=_asn.function or "(global)",
                    file_record_identifier=None,
                    key_operands=[],
                    access_mode="keyed",
                    source_file=file,
                    source_line=_asn.line,
                    source_section=_asn.function,
                    confidence="low",
                ))

            elif _clow in _CPP_DFMOD:
                _cpp_db_catalog.add_operation(DBOperation(
                    operation_type=DBOperationType.UPDATE,
                    macro_name="dfmod",
                    calling_routine=_asn.function or "(global)",
                    file_record_identifier=None,
                    key_operands=[],
                    access_mode="keyed",
                    source_file=file,
                    source_line=_asn.line,
                    source_section=_asn.function,
                    confidence="low",
                ))

            elif _clow in _CPP_DFCLS:
                _cpp_db_catalog.add_operation(DBOperation(
                    operation_type=DBOperationType.CONTROL,
                    macro_name="dfcls",
                    calling_routine=_asn.function or "(global)",
                    file_record_identifier=None,
                    key_operands=[],
                    access_mode="keyed",
                    source_file=file,
                    source_line=_asn.line,
                    source_section=_asn.function,
                    confidence="low",
                ))

        if _cpp_db_catalog.operations:
            context.add_metadata("db_operation_catalog_cpp", _cpp_db_catalog)

        _elapsed = time.perf_counter() - _t0
        if _elapsed >= _SLOW_FILE_THRESHOLD:
            _log.warning(
                "VariableLineageBuilder.process_c_family slow file (%.1fs): %s",
                _elapsed, context.file_path,
            )
        else:
            _log.debug(
                "VariableLineageBuilder.process_c_family done in %.3fs: %s",
                _elapsed, context.file_path,
            )