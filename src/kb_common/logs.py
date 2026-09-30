"""Single logging setup shared by every entrypoint (kb_api, kb_mcp, the
fetch/pipeline CLIs).
"""
import logging
from collections.abc import Iterable

from kb_common import config

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

_QUIET_LOGGERS = ("elastic_transport", "urllib3", "httpx")

_configured = False


class EndpointFilter(logging.Filter):
    """Drops uvicorn access-log records for the given request paths."""

    def __init__(self, excluded_paths: Iterable[str]):
        super().__init__()
        self.excluded_paths = frozenset(excluded_paths)

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3 or not isinstance(args[2], str):
            return True
        return args[2].split("?", 1)[0] not in self.excluded_paths


def setup_logging(level: str | None = None, excluded_paths: Iterable[str] | None = None) -> None:
    """Configures the root logger and the uvicorn access-log filter."""

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
