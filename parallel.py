"""Env-driven parallelism + RAM-awareness for the VBT precompute layer.

The precompute steps (modifier index, const resolver, cfg-warm) are
embarrassingly parallel over *files*: each file is scanned/parsed independently
and contributes a small, picklable partial result that the parent merges
deterministically. This module provides:

* :func:`plan_workers` — decide worker count / batch size / per-proc memory cap
  from environment overrides, falling back to a RAM- and CPU-aware auto plan
  that REUSES the battle-tested probes in ``api.config`` (read-only). If
  ``api.config`` is unavailable the plan still works via a conservative CPU
  fallback, so VBT stays self-contained.
* :func:`parallel_map` — a ``ProcessPoolExecutor`` map mirroring the proven
  pool setup in ``multi_file/orchestrator.py`` (fork context, daemon-flag
  clearing, niced + SIGINT-ignoring workers), returning results in **input
  order** for determinism, with a sequential fallback for trivial workloads.

Determinism contract
---------------------
``parallel_map`` returns results keyed to their INPUT index (never
as-completed order). Callers pass file PATHS (picklable) and merge partials in
a fixed order, so the precompute artifacts are byte/content-identical
regardless of worker count.
"""
from __future__ import annotations

import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParallelPlan:
    """The resolved parallelism plan for one precompute run."""
    worker_count: int           # number of worker processes (>=1)
    batch_size: int             # files per chunk hint
    per_proc_mem_cap_mb: int    # advisory per-worker memory cap (MB)


def _env_int(name: str, default: int) -> int:
    """Read an int env var, tolerating blanks / garbage (→ default)."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def plan_workers() -> ParallelPlan:
    """Resolve the parallelism plan from env, with a RAM/CPU-aware auto path.

    Environment overrides
    ----------------------
    ``VBT_WORKERS``               worker count; ``0`` (default) → auto.
    ``VBT_BATCH_SIZE``            files per chunk hint (default ``20``).
    ``VBT_MEM_BUDGET_FRACTION``   fraction of total RAM to budget (default ``0.60``).
    ``VBT_FILE_TIMEOUT_SECONDS``  per-file subprocess timeout (default ``120``;
                                  consumed by cfg-warm callers, surfaced here so
                                  it lives with the other knobs).

    Auto path reuses ``api.config`` probes read-only:
    ``_detect_total_ram_gb`` / ``_detect_cgroup_memory_limit_gb`` /
    ``_detect_available_ram_gb`` (conservative RAM = min of these),
    ``_detect_cgroup_cpu_count`` / ``_detect_cpu_load_factor``, then
    ``_auto_analysis_workers`` + ``_auto_worker_max_memory_mb``. If that import
    fails we fall back to ``max(1, cpu_count - 2)`` workers.
    """
    batch_size = max(1, _env_int("VBT_BATCH_SIZE", 20))
    mem_fraction = _env_float("VBT_MEM_BUDGET_FRACTION", 0.60)
    if not (0.0 < mem_fraction <= 1.0):
        mem_fraction = 0.60

    explicit = _env_int("VBT_WORKERS", 0)

    worker_count: int
    per_proc_mem_cap_mb: int
    if explicit > 0:
        # Explicit override: honour the count; still derive a sane memory cap.
        worker_count = explicit
        per_proc_mem_cap_mb = _auto_mem_cap(worker_count, mem_fraction)
    else:
        worker_count, per_proc_mem_cap_mb = _auto_plan(batch_size, mem_fraction)

    return ParallelPlan(
        worker_count=max(1, worker_count),
        batch_size=batch_size,
        per_proc_mem_cap_mb=max(256, per_proc_mem_cap_mb),
    )


def _auto_plan(batch_size: int, mem_fraction: float) -> tuple[int, int]:
    """RAM/CPU-aware worker count + per-worker memory cap (MB).

    Reuses api.config probes read-only; falls back to a CPU heuristic if that
    package can't be imported (keeps VBT standalone).
    """
    try:
        from api.config import (
            _auto_analysis_workers,
            _auto_worker_max_memory_mb,
            _detect_available_ram_gb,
            _detect_cgroup_cpu_count,
            _detect_cgroup_memory_limit_gb,
            _detect_cpu_load_factor,
            _detect_total_ram_gb,
        )
    except Exception:
        # api.config unavailable — conservative CPU-only fallback.
        workers = max(1, (os.cpu_count() or 2) - 2)
        return workers, _auto_mem_cap(workers, mem_fraction)

    total_ram = _detect_total_ram_gb()
    cgroup_ram = _detect_cgroup_memory_limit_gb()      # inf when unlimited
    avail_ram = _detect_available_ram_gb()
    # Be conservative: never plan against more RAM than is actually usable
    # right now, nor more than a cgroup cap allows.
    conservative_ram = min(total_ram, cgroup_ram, avail_ram)
    if conservative_ram <= 0 or conservative_ram != conservative_ram:  # NaN guard
        conservative_ram = total_ram

    cpu = _detect_cgroup_cpu_count()
    load = _detect_cpu_load_factor()

    workers = _auto_analysis_workers(
        total_ram_gb=conservative_ram,
        memory_budget_fraction=mem_fraction,
        service_reserve_gb=3.0,
        batch_size=batch_size,
        celery_concurrency=1,
        effective_cpu=cpu,
        cpu_load_factor=load,
    )
    workers = max(1, int(workers))
    mem_cap = _auto_worker_max_memory_mb(conservative_ram, workers, 3.0)
    return workers, int(mem_cap)


def _auto_mem_cap(workers: int, mem_fraction: float) -> int:
    """Per-worker memory cap (MB) using api.config when available, else a split
    of a fraction of total RAM (used for the explicit-VBT_WORKERS path and the
    no-api.config fallback)."""
    try:
        from api.config import _auto_worker_max_memory_mb, _detect_total_ram_gb
        return int(_auto_worker_max_memory_mb(_detect_total_ram_gb(), max(1, workers), 3.0))
    except Exception:
        # 8 GB conservative default RAM if we can't probe at all.
        budget_mb = 8.0 * 1024 * mem_fraction
        return max(256, int(budget_mb / max(1, workers)))


# --------------------------------------------------------------------------- #
# Pool
# --------------------------------------------------------------------------- #
def _worker_init() -> None:
    """ProcessPoolExecutor worker initializer (mirrors orchestrator._worker_init).

    - ``os.nice(5)`` so precompute workers yield CPU under contention.
    - Ignore SIGINT so Ctrl-C in the parent doesn't cascade; the executor's
      context-manager shutdown handles clean exit.
    - DISPOSE inherited SQLAlchemy engines on fork. Under the ``fork`` start method a
      child inherits the parent's ``api.index_db.engine._engines`` cache, including any
      pooled SQLite connection — sharing one sqlite3 file handle across two processes
      corrupts reads. We drop the inherited pool with ``dispose(close=False)`` (the
      documented fork-safe form: it does NOT close the fds the parent still uses) and
      clear ``_engines`` so each worker lazily reopens its OWN handles on first access.
      Workers are read-only on index.db (blob reads), so no write contention. Wrapped in
      try/except so VBT stays importable without the api package.
    """
    try:
        os.nice(5)
    except OSError:
        pass  # not supported on Windows / some sandboxes
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
    try:
        import api.index_db.engine as _eng
        with _eng._lock:
            for engine in list(_eng._engines.values()):
                try:
                    engine.dispose(close=False)
                except Exception:
                    pass
            _eng._engines.clear()
    except Exception:
        pass


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    plan: Optional[ParallelPlan] = None,
    ordered: bool = True,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[R]:
    """Apply ``fn`` to each item across worker processes, INPUT-ORDERED.

    Mirrors the pool setup in ``multi_file/orchestrator.py`` (fork context,
    daemon-flag clearing, niced/SIGINT-ignoring workers). ``fn`` and every item
    must be picklable — pass a MODULE-LEVEL function and FILE PATHS, not objects.

    Determinism: results are keyed to their input index and returned in input
    order (NOT as-completed order) when ``ordered`` is True (the default and the
    only mode the precompute callers use). A sequential fallback runs when the
    plan resolves to ``worker_count <= 1`` or there is ``<= 1`` item.

    ``on_progress`` (optional) is called in the PARENT as ``on_progress(done, total)``
    each time an item finishes, in completion order — purely observational, results
    are unaffected. Exceptions from the callback are swallowed.
    """
    items = list(items)
    if plan is None:
        plan = plan_workers()

    def _notify(done: int, total: int) -> None:
        if on_progress is not None:
            try:
                on_progress(done, total)
            except Exception:
                pass

    n = len(items)
    if plan.worker_count <= 1 or n <= 1:
        # Sequential fallback — identical results, no process overhead.
        out: List[R] = []
        for i, it in enumerate(items):
            out.append(fn(it))
            _notify(i + 1, n)
        return out

    workers = min(plan.worker_count, n)

    # Force 'fork' on macOS/Linux: avoids pickling AuthenticationString when
    # spawning from inside a daemonic worker, and skips re-importing the world.
    # Fall back to the default context on spawn-only platforms (Windows).
    try:
        mp_ctx = multiprocessing.get_context("fork")
    except ValueError:
        mp_ctx = None

    # A Celery ForkPoolWorker is a daemon; daemons can't fork children. Clear
    # the flag for the duration of the pool, then restore it.
    proc = multiprocessing.current_process()
    was_daemon = getattr(proc, "daemon", False)
    if was_daemon:
        proc.daemon = False

    executor_kwargs = {"max_workers": workers, "initializer": _worker_init}
    if mp_ctx is not None:
        executor_kwargs["mp_context"] = mp_ctx
    # Recycle workers periodically to bound RAM growth. Only supported on
    # Python 3.11+; omit gracefully on 3.10 (TypeError on unknown kwarg).
    max_tasks = max(50, plan.batch_size * 4)

    results: List[Optional[R]] = [None] * n
    try:
        try:
            executor = ProcessPoolExecutor(
                max_tasks_per_child=max_tasks, **executor_kwargs
            )
        except (TypeError, ValueError):
            # TypeError: Python 3.10 (no max_tasks_per_child kwarg).
            # ValueError: fork start method incompatible with max_tasks_per_child (e.g. inside a
            #             Celery prefork worker on Python 3.12+) — would otherwise CRASH the precompute,
            #             leaving no artifacts so every client trace runs fully uncached (cold cliff).
            executor = ProcessPoolExecutor(**executor_kwargs)
        with executor:
            # Key each future to its INPUT index → deterministic input-ordered
            # output regardless of completion order.
            future_to_idx = {
                executor.submit(fn, item): i for i, item in enumerate(items)
            }
            done = 0
            for fut in as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()
                done += 1
                _notify(done, n)
    finally:
        if was_daemon:
            proc.daemon = True

    if ordered:
        return results  # type: ignore[return-value]
    # Unordered requested: collapse to a plain list (still complete, any order).
    return [r for r in results]  # type: ignore[return-value]
