"""Minimal in-memory sliding-window rate limiter.

Sufficient for a single-process deployment (both kb_mcp and kb_api run as
one uvicorn worker each here) - state is per-process, not shared across
multiple workers/replicas.
"""
import time
from collections import deque
from threading import Lock


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        """True and records a hit if `key` is under its limit, else False."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True
