"""Enum / constant value resolution (SPEC §6).

Resolve symbolic constants to their values, for BOTH the set value and guard
conditions (which compare against constants):

  C++ enum constants  → from blueprint ``enum_values`` ({const: {enum_type, value}}).
                        Present wherever the defining header is parsed — i.e. fully in
                        the 22k codebase; in temp/asm only the locally-defined enums.
  ASM EQU/DC symbols  → parsed from the .mac/.asm source definition lines (blueprint
                        symbol ``value`` is null), e.g. ``LG#C400&A EQU C'D'`` → ``C'D'``.

Symbolic form is always kept (schema-accepted); this adds a resolved value when known.
Indices are built once and cached; memory-light (name→value strings only).

Scaling (22k files)
-------------------
A naive build re-globs ALL ``*.cpp.json``/``*.hpp.json`` blueprints and ALL
``*.hpp``/``*.cpp``/``*.mac``/``*.asm`` sources and re-parses every enum/EQU/DC on
*every fresh process* — multi-minute cold start at corpus scale. To fix this
without changing semantics we DISK-BACK the resolver, mirroring the exact
mechanism in ``precompute/modifier_index.py``:

* **Disk persistence.** The built ``cpp_enum`` / ``asm_const`` dicts are serialized
  to a compact JSON artifact under the VBT cache dir, keyed on a *manifest hash*
  of the contributing files (path + size + mtime) plus an artifact-version tag.
  On a hit we reload the artifact instead of re-globbing + re-parsing; any
  add/remove/edit of a contributing file (or a version bump) rebuilds.
* **In-process fast path.** ``_CACHE`` still short-circuits repeat calls in the
  same process (no disk touch at all).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from blueprint_io import load_blueprint

_CACHE: Dict[Tuple[str, str], "ConstResolver"] = {}

# Disk artifact location — mirrors modifier_index.py. VBT_CACHE_DIR overrides the
# base; default is vbt/.cache/const . const_resolver.py lives in vbt/resolve/, so
# vbt/ is two levels up.
_VBT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CACHE_DIR = os.path.join(_VBT_DIR, ".cache", "const")

# Bump if the artifact layout or parse semantics change (forces a rebuild).
# v2: brace-aware scope scan — enums nested in a struct/class now also emit the
#     ENCLOSING-scope-qualified key (`SoftCardValueDetail::Const`), so the index content
#     changes; the bump invalidates stale v1 caches.
_ARTIFACT_VERSION = 2


def _cache_dir() -> str:
    base = os.environ.get("VBT_CACHE_DIR")
    if base:
        return os.path.join(base, "const")
    return _DEFAULT_CACHE_DIR

# HLASM: LABEL[&suffix]  EQU|DC|DS  operand  comment
_ASM_DEF = re.compile(
    r"^([A-Za-z@#$_][A-Za-z0-9@#$_]*)(?:&\w+)?\s+(EQU|DC|DS)\s+(\S+)", re.M)

# One enumerator inside a body:  NAME [ = VALUE ] [ , ]  (comments already stripped).
_CPP_ENUMERATOR_RE = re.compile(
    r"([A-Za-z_]\w*)\s*(?:=\s*([^,}]+?))?\s*(?:,|$)", re.S)
# Brace-aware scope scanner tokens: a keyword that opens a named scope or an enum, and
# the optional name following it (handling `enum class X` / `enum struct X`).
_SCOPE_KW_RE = re.compile(r"\b(enum|struct|class|namespace)\b")
_SCOPE_NAME_RE = re.compile(r"\s*(?:(?:class|struct)\s+)?([A-Za-z_]\w*)?")


# Identifier-shaped tokens in a condition/value text (mirrors the historical
# engine/recurse tokenizer; `::`-qualified C++ enum spellings kept whole).
_RESOLVE_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_#$@]*(?:::[A-Za-z_][A-Za-z0-9_]*)?")
_REGISTER_TOK = re.compile(r"R(?:[0-9]|1[0-5])", re.I)


def iter_resolvable_tokens(text: str):
    """Identifier tokens of ``text`` that are candidates for constant resolution.

    Skips tokens that are lexical pieces of a LITERAL rather than symbols — any token
    adjacent to a quote: the HLASM literal/attribute prefix in ``C'D'`` / ``X'78'`` /
    ``=C'CV'`` / ``L'FLD``, and quoted contents (``'N'``, ``"CA"``). The corpus defines
    single-letter EQUs, so resolving those pieces produced junk like ``C → '*'`` /
    ``X → 23`` / ``N → 13`` on nearly half of all emitted resolvedConstants. Registers
    R0–R15 are skipped too (they only ever resolve to their own numbers)."""
    for m in _RESOLVE_TOK.finditer(text or ""):
        i, j = m.start(), m.end()
        if i > 0 and text[i - 1] in "'\"":
            continue
        if j < len(text) and text[j] in "'\"":
            continue
        if _REGISTER_TOK.fullmatch(m.group(0)):
            continue
        yield m.group(0)


class ConstResolver:
    def __init__(self) -> None:
        self.cpp_enum: Dict[str, str] = {}   # "Const" and "Type::Const" -> value
        self.asm_const: Dict[str, str] = {}  # SYMBOL_UPPER -> operand text
        self._bare_cpp: Optional[frozenset] = None

    def bare_cpp_const_names(self) -> frozenset:
        """The set of BARE (unqualified) C++ enum-constant names — ``cpp_enum`` keys with
        no ``::``. A bare identifier appearing in a guard/operand that is one of these is a
        CONSTANT being compared against, not a traceable variable: e.g. ``task != DacAutoRes``
        where ``DacAutoRes = 0x64``. The dep-var extractor consults this to drop such names
        (the ``EnumType::Const`` form is already dropped by the ``::`` rule; the ALL-CAPS rule
        catches ``#define``-style macros — this closes the remaining bare TitleCase/camelCase
        enumerator case). Memoized; lazy so it also covers a disk-reloaded resolver."""
        if self._bare_cpp is None:
            self._bare_cpp = frozenset(k for k in self.cpp_enum if "::" not in k)
        return self._bare_cpp

    def resolve(self, name: str, language: str) -> Optional[str]:
        if not name:
            return None
        if language == "cpp":
            n = name.strip()
            if n in self.cpp_enum:
                return self.cpp_enum[n]
            return self.cpp_enum.get(n.split("::")[-1])
        return self.asm_const.get(name.strip().upper())


# --------------------------------------------------------------------------- #
# Per-file partial builders.
#
# Each returns the (ordered) (key, value) pairs ONE file contributes, as a list
# (NOT a dict) so the parent can apply ``setdefault`` in a deterministic order:
# blueprint-enum → cpp-source → asm, with files sorted within each group. That
# fixed first-writer-wins order is what makes blueprint values beat source
# values and the artifact identical regardless of worker count. Workers return
# only the small list of pairs — no parsed blueprint crosses IPC.
# --------------------------------------------------------------------------- #
def _cpp_pairs_from_blueprint(bp_path_str: str) -> List[Tuple[str, str]]:
    """C++ enum (key, value) pairs from one blueprint's ``enum_values``."""
    pairs: List[Tuple[str, str]] = []
    try:
        ev = load_blueprint(bp_path_str, keys={"enum_values"}).get("enum_values") or {}
    except Exception:
        return pairs
    if not isinstance(ev, dict):
        return pairs
    for const, meta in ev.items():
        val = meta.get("value") if isinstance(meta, dict) else meta
        etype = meta.get("enum_type") if isinstance(meta, dict) else None
        if val is None:
            continue
        pairs.append((str(const), str(val)))
        if etype:
            pairs.append((f"{etype}::{const}", str(val)))
    return pairs


def _strip_cpp_comments(text: str) -> str:
    """Drop // line comments and /* ... */ block comments so enum bodies parse
    cleanly (corpus enum lines carry trailing `// R001`-style revision tags)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _int_or_none(tok: str) -> Optional[int]:
    """Parse a decimal / hex / octal integer literal, else None (e.g. char lits,
    expressions). Trailing integer suffixes (U/L) are tolerated."""
    t = tok.strip().rstrip("uUlL")
    try:
        return int(t, 0)        # base 0 -> honours 0x.. / 0.. prefixes
    except (ValueError, TypeError):
        return None


def _matching_brace(text: str, open_idx: int) -> int:
    """Index of the ``}`` matching the ``{`` at ``open_idx`` (end of text if unbalanced)."""
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def _emit_enumerators(pairs: List[Tuple[str, str]], body: str,
                      enum_name: Optional[str], prefix: str) -> None:
    """Append (key, value) for every enumerator in an enum ``body``.

    ``prefix`` is the enclosing struct/class/namespace qualifier chain (``"A::B::"`` or
    ``""``). We emit the BARE name, the enum-qualified name, AND — the C9b fix — the
    ENCLOSING-SCOPE-qualified name: an *unscoped* ``enum`` nested in ``struct X`` leaks
    its enumerators into X's scope, so the codebase spells them ``X::Const`` (the
    ``SoftCardValueDetail::ExpressPayTransaction`` idiom), NOT ``X::EnumName::Const``.
    Emitting every valid spelling lets the resolver hit the exact key the code uses
    instead of relying on the lossy bare-tail fallback (which collides at 22k scale)."""
    running: Optional[int] = -1     # next implicit = running + 1
    for enr in _CPP_ENUMERATOR_RE.finditer(body):
        const = enr.group(1)
        if not const:
            continue
        raw = (enr.group(2) or "").strip()
        if raw:
            # Keep the SOURCE literal verbatim (`0x80`, `'A'`, an expression) — matches
            # the blueprint convention + stays comparable to the guard text. `ival` only
            # drives auto-increment of any following implicit enumerators.
            running = _int_or_none(raw)     # may be None for char/expr literals
            value = raw
        else:
            if running is None:
                continue                    # can't infer past a non-int literal
            running += 1
            value = str(running)
        pairs.append((const, value))                                   # bare Const
        if enum_name:
            pairs.append((f"{enum_name}::{const}", value))             # EnumName::Const
        if prefix:
            pairs.append((f"{prefix}{const}", value))                  # Struct::Const  (C9b)
            if enum_name:
                pairs.append((f"{prefix}{enum_name}::{const}", value))  # Struct::EnumName::Const


def _scan_scopes(text: str, prefix: str, pairs: List[Tuple[str, str]]) -> None:
    """Brace-aware recursive walk: emit enum (key, value) pairs qualified by the
    enclosing struct/class/namespace chain in ``prefix``. A struct/class/namespace body
    is recursed with its name appended; an enum body's enumerators are emitted (scoped by
    both the enum and the enclosing scope). Forward decls / variable decls (no ``{``) are
    skipped. Generic — no type names are hard-coded."""
    i, n = 0, len(text)
    while True:
        m = _SCOPE_KW_RE.search(text, i)
        if not m:
            return
        kw = m.group(1)
        nm = _SCOPE_NAME_RE.match(text, m.end())
        name = nm.group(1) if nm else None
        p = nm.end() if nm else m.end()
        while p < n and text[p] not in "{;":   # skip a base-class spec / underlying type
            p += 1
        if p >= n or text[p] == ";":            # forward declaration / variable — no body
            i = p + 1
            continue
        close = _matching_brace(text, p)
        if kw == "enum":
            _emit_enumerators(pairs, text[p + 1:close], name, prefix)
        else:                                   # struct / class / namespace → recurse
            _scan_scopes(text[p + 1:close], (prefix + name + "::") if name else prefix, pairs)
        i = close + 1


def _cpp_pairs_from_source(src_path_str: str) -> List[Tuple[str, str]]:
    """C9 — C++ enum DEFINITIONS parsed straight from one source file (.hpp/.cpp),
    mirroring the ASM EQU/DC source parser. Closes the gap where blueprint
    ``enum_values`` is absent for an enum whose defining header was not parsed.

    Brace-aware (C9b): handles the ``struct X { enum Type { ... } }`` namespace idiom by
    qualifying enumerators with the ENCLOSING struct/class as well as the enum — so
    ``SoftCardValueDetail::ExpressPayTransaction`` resolves on its exact key, not via the
    lossy bare-tail fallback. Char literals keep their ``'x'`` form; implicit enumerators
    auto-increment per C++ rules. Returns (key, value) pairs in source order; the parent
    applies setdefault so an explicit blueprint value (merged first) always wins and
    source only FILLS gaps."""
    pairs: List[Tuple[str, str]] = []
    try:
        text = _strip_cpp_comments(Path(src_path_str).read_text(errors="replace"))
    except OSError:
        return pairs
    _scan_scopes(text, "", pairs)
    return pairs


def _asm_pairs_from_source(src_path_str: str) -> List[Tuple[str, str]]:
    """ASM EQU/DC (LABEL_UPPER, operand) pairs from one .mac/.asm source file."""
    pairs: List[Tuple[str, str]] = []
    try:
        text = Path(src_path_str).read_text(errors="replace")
    except OSError:
        return pairs
    for m in _ASM_DEF.finditer(text):
        label, op, operand = m.group(1).upper(), m.group(2), m.group(3)
        if op == "DS":               # storage decl, not a value
            continue
        pairs.append((label, operand))
    return pairs


# --------------------------------------------------------------------------- #
# Disk persistence — same manifest-hash + atomic-write mechanism as
# precompute/modifier_index.py.
# --------------------------------------------------------------------------- #
def _source_files(blueprint_dir: Path, asm_dir: Path) -> List[Path]:
    """Every file that contributes to the resolver, in a stable order.

    Must match what the three builders read: blueprint enum sources
    (``*.cpp.json`` / ``*.hpp.json``) plus the source-side enum / EQU-DC files
    (``*.hpp`` / ``*.cpp`` / ``*.mac`` / ``*.asm``).
    """
    files: List[Path] = []
    for pat in ("*.cpp.json", "*.hpp.json"):
        files.extend(blueprint_dir.glob(pat))
    for pat in ("*.hpp", "*.cpp", "*.mac", "*.asm"):
        files.extend(asm_dir.glob(pat))
    return sorted(files, key=lambda p: str(p))


def _manifest_hash(files: List[Path]) -> str:
    """Fingerprint the source set: per-file (path, size, mtime). Cheap (one stat)."""
    h = hashlib.sha256()
    h.update(f"v{_ARTIFACT_VERSION}\n".encode("utf-8"))
    for p in files:
        try:
            st = p.stat()
            meta = f"{p}\x00{st.st_size}\x00{st.st_mtime_ns}"
        except OSError:
            meta = f"{p}\x00?\x00?"
        h.update(meta.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _artifact_path(blueprint_dir: Path, asm_dir: Path, manifest: str) -> str:
    dir_key = hashlib.sha256(
        f"{blueprint_dir}\x00{asm_dir}".encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(_cache_dir(), f"const-{dir_key}-{manifest[:32]}.json")


def _artifact_load(path: str) -> Optional[ConstResolver]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("version") != _ARTIFACT_VERSION:
        return None
    cpp_enum, asm_const = data.get("cpp_enum"), data.get("asm_const")
    if not isinstance(cpp_enum, dict) or not isinstance(asm_const, dict):
        return None
    idx = ConstResolver()
    idx.cpp_enum = {str(k): str(v) for k, v in cpp_enum.items()}
    idx.asm_const = {str(k): str(v) for k, v in asm_const.items()}
    return idx


def _artifact_store(path: str, idx: ConstResolver) -> None:
    """Atomically persist the resolver dicts. Best-effort."""
    payload = {
        "version": _ARTIFACT_VERSION,
        "cpp_enum": idx.cpp_enum,
        "asm_const": idx.asm_const,
    }
    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def _build_resolver(blueprint_dir: Path, asm_dir: Path) -> ConstResolver:
    """Build the resolver, scanning files in parallel and MERGING deterministically.

    Each file's contribution is computed by a module-level worker (small list of
    (key, value) pairs — no parsed blueprint crosses IPC). The parent applies
    ``setdefault`` in a FIXED order so semantics are unchanged and the artifact
    is identical regardless of worker count:

      1. blueprint enum_values  → explicit values WIN (merged first)
      2. C9 cpp-source enums     → FILL gaps the blueprint missed
      3. ASM EQU/DC source       → asm_const

    Within each group files are processed in SORTED path order; pairs within a
    file keep source order. First-writer-wins is therefore deterministic.
    """
    from vbt.precompute.parallel import parallel_map

    idx = ConstResolver()

    # --- C++ enums: blueprints first (sorted), then source (sorted) ---------
    cpp_bp_files = sorted(
        str(p) for pat in ("*.cpp.json", "*.hpp.json")
        for p in blueprint_dir.glob(pat)
    )
    cpp_src_files = sorted(
        str(p) for pat in ("*.hpp", "*.cpp") for p in asm_dir.glob(pat)
    )
    for pairs in parallel_map(_cpp_pairs_from_blueprint, cpp_bp_files):
        for k, v in pairs:
            idx.cpp_enum.setdefault(k, v)
    for pairs in parallel_map(_cpp_pairs_from_source, cpp_src_files):
        for k, v in pairs:
            idx.cpp_enum.setdefault(k, v)

    # --- ASM EQU/DC: sorted source order ------------------------------------
    asm_src_files = sorted(
        str(p) for pat in ("*.mac", "*.asm") for p in asm_dir.glob(pat)
    )
    for pairs in parallel_map(_asm_pairs_from_source, asm_src_files):
        for k, v in pairs:
            idx.asm_const.setdefault(k, v)

    return idx


def get_const_resolver(blueprint_dir: Path, asm_dir: Path) -> ConstResolver:
    blueprint_dir, asm_dir = Path(blueprint_dir), Path(asm_dir)
    key = (str(blueprint_dir), str(asm_dir))
    cached = _CACHE.get(key)
    if cached is not None:           # process-local fast path (no disk touch)
        return cached

    files = _source_files(blueprint_dir, asm_dir)
    manifest = _manifest_hash(files)
    artifact = _artifact_path(blueprint_dir, asm_dir, manifest)

    idx = _artifact_load(artifact)   # disk fast path: skip re-glob + re-parse
    if idx is None:
        idx = _build_resolver(blueprint_dir, asm_dir)
        _artifact_store(artifact, idx)

    _CACHE[key] = idx
    return idx
