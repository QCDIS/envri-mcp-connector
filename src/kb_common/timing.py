"""Per-request stage timing: `stage()` measures a block's wall-clock time
and, if a `request()` scope is open
"""
import time
from contextlib import contextmanager
from contextvars import ContextVar

_current: ContextVar[dict | None] = ContextVar("kb_timing_current", default=None)


def _record(name: str, elapsed: float) -> None:
    stages = _current.get()
    if stages is not None:
        stages[name] = stages.get(name, 0.0) + elapsed


@contextmanager
def stage(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        _record(name, time.perf_counter() - start)


def record(name: str, elapsed_seconds: float) -> None:
    """Record a duration measured elsewhere (e.g. Elasticsearch's own
    `took`, in ms) instead of timing a block directly."""
    _record(name, elapsed_seconds)


@contextmanager
def request():
    """Opens a per-request stage dict that nested `stage()`/`record()` calls
    (anywhere on the call stack, same async context) accumulate into."""
    token = _current.set({})
    try:
        yield _current.get()
    finally:
        _current.reset(token)
