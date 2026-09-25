"""Single logging setup shared by every entrypoint (kb_api, kb_mcp, the
fetch/pipeline CLIs). Library modules only do `logging.getLogger(__name__)`
- they never configure logging themselves.

Uvicorn should be started with `log_config=None` so its `uvicorn.*` loggers
propagate to the root handler installed here instead of getting their own.
"""
import logging
from collections.abc import Iterable

from kb_common import config

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# Chatty at INFO (one line per HTTP request to Elasticsearch / kb_api).
_QUIET_LOGGERS = ("elastic_transport", "urllib3", "httpx")

_configured = False


class EndpointFilter(logging.Filter):
    """Drops uvicorn access-log records for the given request paths.

    Uvicorn logs access lines as `'%s - "%s %s HTTP/%s" %d'` with args
    `(client, method, path_with_query, http_version, status)`.
    """

    def __init__(self, excluded_paths: Iterable[str]):
        super().__init__()
        self.excluded_paths = frozenset(excluded_paths)

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3 or not isinstance(args[2], str):
            return True
        return args[2].split("?", 1)[0] not in self.excluded_paths


def setup_logging(level: str | None = None, excluded_paths: Iterable[str] | None = None) -> None:
    """Configures the root logger and the uvicorn access-log filter. Safe to
    call more than once - only the first call has any effect."""
    global _configured
    if _configured:
        return
    _configured = True

    logging.basicConfig(level=level or config.LOG_LEVEL, format=LOG_FORMAT, force=True)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    paths = config.LOG_EXCLUDE_PATHS if excluded_paths is None else excluded_paths
    if paths:
        logging.getLogger("uvicorn.access").addFilter(EndpointFilter(paths))
