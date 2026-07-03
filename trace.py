"""Command-line entry point for the VBT variable-lineage traversal.

Run a trace:

    env/bin/python3 -m vbt.trace \
        --variable softCardValue --language cpp \
        --hop aa71:asm \
        --hop dw730000:cpp:DW73 \
        --hop dw710000:cpp:callPlasticAuthenticationComponentInterface \
        --blueprint-dir jobs/baseline-temp-asm/output --asm-dir temp/asm

See every tunable limit + its default:

    env/bin/python3 -m vbt.trace --help

A ``--hop`` is ``stem:type[:function]`` (``type`` = ``asm`` | ``cpp``); pass them in
chain order with the TAIL (deepest hop) LAST. ``--graph-file`` defaults to
``<blueprint-dir>/file_call_graph.json``. Output is the rootVariable / dependentVariables
JSON (SPEC §8) to stdout, or to ``--out FILE``.

Optional, all default-off emit modes (see the ``--help`` group "output emit" and
RUNNING.md §5): ``--out-dir DIR`` splits the one dict into per-part files,
``--no-code`` drops embedded source text (path+line refs kept), ``--format
msgpack`` packs as msgpack, ``--zip`` produces a single archive. With NO new
flags the behavior is unchanged: one JSON blob to stdout or ``--out FILE``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _hop(spec: str):
    """Parse a ``stem:type[:function]`` hop spec into a vbt.engine.Hop."""
    from vbt.engine import Hop
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"--hop must be stem:type[:function], got {spec!r}")
    stem, ftype = parts[0], parts[1]
    fn = ":".join(parts[2:]) or None
    if ftype not in ("asm", "cpp"):
        raise argparse.ArgumentTypeError(f"hop type must be asm|cpp, got {ftype!r}")
    return Hop(stem, ftype, fn)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vbt.trace",
        description="VBT variable backward-lineage traversal: one entry per (setter, chain) "
                    "plus the dependent-variable tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Fixed internal bounds (edit source to change, not CLI flags): C++ early-exit "
               "local-walk max_blocks=6 (vbt/conditions/cpp_conditions.py); resolver "
               "#include-follow _INCLUDE_MAX_DEPTH=8 (vbt/resolve/_find_cpp_chain.py). When any "
               "path/route cap is hit the affected setter carries pathsCapped=true and is never "
               "reported unconditionalAtSetter; an over-budget dep var carries "
               "offchainSearchCapped=true — truncation is always surfaced, never silent.",
    )

    req = p.add_argument_group("required")
    req.add_argument("--variable", required=True,
                     help="variable to trace (C++ field tail, or ASM symbol)")
    req.add_argument("--language", required=True, choices=["cpp", "asm"],
                     help="language of --variable")
    req.add_argument("--hop", action="append", dest="hops", required=True, type=_hop,
                     metavar="STEM:TYPE[:FN]",
                     help="chain entry hop (repeatable); list in chain order, TAIL (deepest) LAST")
    req.add_argument("--blueprint-dir", required=True, type=Path,
                     help="dir of .asm.json/.cpp.json blueprints (from the FULL analysis pipeline)")
    req.add_argument("--asm-dir", required=True, type=Path,
                     help="dir of the .cpp/.asm/.mac source files")

    io = p.add_argument_group("optional i/o")
    io.add_argument("--graph-file", type=Path, default=None,
                    help="file_call_graph.json (default: <blueprint-dir>/file_call_graph.json)")
    io.add_argument("--job-id", default=None,
                    help="index.db job id (jobs/<id>/index.db). When set, serve precomputed "
                         "artifacts from the DB (run `python -m vbt.precompute_db` first). "
                         "Omit for the pure file/compute path (byte-identical).")
    io.add_argument("--out", type=Path, default=None, help="write JSON here (default: stdout)")
    io.add_argument("--compact", action="store_true",
                    help="write compact JSON (no indent): ~6x faster serialize + ~38% smaller file "
                         "for a multi-MB result; default is human-readable indent=2")
    io.add_argument("--candidate-stems", default=None,
                    help="comma-separated file stems to restrict the own-language setter search")
    io.add_argument("--candidate-functions", default=None,
                    help="comma-separated function/block names to restrict the setter search "
                         "(kept only when the setter's function/block matches one)")
    io.add_argument("--home-hint", default=None,
                    help="file stem that scopes cross-language alias resolution")

    sc = p.add_argument_group("scale / output scope")
    sc.add_argument("--disable-dependents", action="store_true",
                    help="root setters only — skip the dependent-variable tree (dependentVariables: [])")
    sc.add_argument("--progress", action="store_true",
                    help="emit per-phase progress logs to STDERR (does not touch the JSON on stdout)")
    sc.add_argument("--no-not-set", dest="emit_not_set", action="store_false",
                    help="GAP 9: do NOT emit \"[not set]\" outcome entries (default: emit them; "
                         "--no-not-set gives output byte-identical to pre-GAP-9)")
    sc.set_defaults(emit_not_set=True)

    emit = p.add_argument_group("output emit (all default-off; default = one JSON to "
                                "stdout/--out, byte-identical to before)")
    emit.add_argument("--out-dir", type=Path, default=None,
                      help="SPLIT mode: write rootVariable + per-dep-var + shared codeBlocks "
                           "+ index manifest into DIR (mutually exclusive with --out)")
    emit.add_argument("--format", choices=["json", "msgpack"], default="json",
                      help="serialization format for emitted files")
    emit.add_argument("--no-code", action="store_true",
                      help="omit embedded source TEXT (drops code/setterCodeChunk/"
                           "relevantCodeChunk); every file+startLine+endLine ref is kept")
    emit.add_argument("--zip", dest="zip_", action="store_true",
                      help="produce ONE zip: with --out-dir a <out-dir>.zip of the parts; "
                           "with --out a zipped single entry. Most compact: "
                           "--no-code --format msgpack --out-dir X --zip")

    lim = p.add_argument_group("tunable limits (defaults = the engine's scale-safety bounds)")
    lim.add_argument("--max-dep-var-depth", type=int, default=1,
                     help="dependent-variable recursion depth (deep recursion is opt-in at 22k scale)")
    lim.add_argument("--max-paths", type=int, default=64,
                     help="distinct intra-file call paths enumerated per setter")
    lim.add_argument("--max-call-depth", type=int, default=16,
                     help="edges deep when enumerating intra-file call paths")
    lim.add_argument("--max-routes", type=int, default=200,
                     help="cross-file routes tail->setter (route_finder)")
    lim.add_argument("--max-route-len", type=int, default=16,
                     help="cross-file route hop-length (route_finder)")
    lim.add_argument("--max-offchain-files", type=int, default=100,
                     help="off-chain candidate files route-checked per dependent variable")
    lim.add_argument("--asm-max-levels", type=int, default=16,
                     help="N-level ASM backward-CFG condition collection")
    lim.add_argument("--depvar-workers", type=int, default=None,
                     help="workers for dependent-variable route memo warming "
                          "(default: VBT_DEPVAR_WORKERS or 1; try 2 on 8 GB RAM)")
    return p


def _auto_job_id(blueprint_dir: Path):
    """Derive the index.db job id from a ``jobs/<id>/...`` blueprint dir, when that job's
    ``index.db`` exists — so a precomputed DB is used automatically. Returns None otherwise."""
    try:
        from api.index_db.engine import job_id_from_path
        from api.config import settings
        jid = job_id_from_path(Path(blueprint_dir).resolve())
        if jid and (Path(settings.JOBS_BASE_DIR) / jid / "index.db").is_file():
            return jid
    except Exception:
        pass
    return None


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out is not None and args.out_dir is not None:
        parser.error("--out and --out-dir are mutually exclusive")
    from vbt.engine import trace_root_variable

    graph = args.graph_file or (args.blueprint_dir / "file_call_graph.json")
    # job-id wiring: if not given, auto-derive from a jobs/<id>/ blueprint dir so a precomputed
    # index.db is used by default (file/compute path otherwise). `--job-id ""` forces compute.
    job_id = args.job_id
    if job_id is None:
        job_id = _auto_job_id(args.blueprint_dir)
        if job_id:
            print(f"[vbt.trace] DB-backed: using precomputed index.db for job '{job_id}' "
                  f"(pass --job-id '' to force the file/compute path)", file=sys.stderr)
    cand = [s.strip() for s in args.candidate_stems.split(",")] if args.candidate_stems else None
    cand_fns = ([s.strip() for s in args.candidate_functions.split(",") if s.strip()]
                if args.candidate_functions else None)
    out = trace_root_variable(
        args.variable, args.language, args.hops,
        blueprint_dir=args.blueprint_dir, asm_dir=args.asm_dir, graph_file=graph,
        job_id=job_id,
        candidate_stems=cand,
        candidate_functions=cand_fns,
        home_hint=args.home_hint,
        disable_dependents=args.disable_dependents,
        emit_not_set=args.emit_not_set,
        progress=args.progress,
        max_dep_var_depth=args.max_dep_var_depth,
        max_paths=args.max_paths,
        max_call_depth=args.max_call_depth,
        max_routes=args.max_routes,
        max_route_len=args.max_route_len,
        max_offchain_files=args.max_offchain_files,
        asm_max_levels=args.asm_max_levels,
        depvar_workers=args.depvar_workers,
    )
    # DEFAULT path (no new emit flags): behave EXACTLY as before — one indent=2
    # JSON blob to --out or stdout, with the original stderr summary on --out.
    new_emit = (args.out_dir is not None or args.format != "json"
                or args.no_code or args.zip_)
    if not new_emit:
        text = json.dumps(out) if args.compact else json.dumps(out, indent=2)
        if args.out:
            args.out.write_text(text)
            print(f"wrote {args.out}  "
                  f"({len(out['rootVariable']['setters'])} (setter,chain) tuples, "
                  f"{len(out['dependentVariables'])} dependent variables)", file=sys.stderr)
        else:
            print(text)
        return 0

    from vbt.output.emit import write_result
    summary = write_result(
        out,
        out_path=args.out,
        out_dir=args.out_dir,
        fmt=args.format,
        include_code=not args.no_code,
        zip_=args.zip_,
        asm_dir=args.asm_dir,
    )
    if summary:
        print(summary, file=sys.stderr)
    return 0


if __name__ == "__main__":
    _rc = main()
    # One-shot CLI: the output is already written + flushed, and every DB write (trace/route cache)
    # committed inside its own `with engine.begin()` block DURING the trace — nothing is pending. Skip
    # Python's exit-time teardown (GC of the multi-MB result dict + tree-sitter/cfg caches ≈ 1s on a deep
    # trace); the OS reclaims the memory instantly. main() itself returns normally, so in-process callers
    # (tests, the API) never hit this os._exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc if isinstance(_rc, int) else 0)
