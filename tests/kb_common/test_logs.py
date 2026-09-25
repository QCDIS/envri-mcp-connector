import logging

import pytest

from kb_common import logs


def _access_record(path: str) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 0,
        '%s - "%s %s HTTP/%s" %d', ("127.0.0.1:5000", "GET", path, "1.1", 200), None,
    )


@pytest.fixture
def fresh_setup(monkeypatch):
    """Lets setup_logging() run again without touching pytest's own root
    handlers, and removes the filters it adds afterwards."""
    access_logger = logging.getLogger("uvicorn.access")
    before = list(access_logger.filters)
    monkeypatch.setattr(logs, "_configured", False)
    monkeypatch.setattr(logs.logging, "basicConfig", lambda **kwargs: None)
    yield access_logger
    access_logger.filters[:] = before


@pytest.mark.parametrize("path", ["/health", "/health?verbose=1"])
def test_endpoint_filter_drops_excluded_paths(path):
    assert logs.EndpointFilter({"/health"}).filter(_access_record(path)) is False


@pytest.mark.parametrize("path", ["/search?query=x", "/healthz", "/"])
def test_endpoint_filter_keeps_other_paths(path):
    assert logs.EndpointFilter({"/health"}).filter(_access_record(path)) is True


def test_endpoint_filter_passes_non_access_records():
    no_args = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 0, "/health", None, None)
    short_args = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 0, "%s", ("/health",), None)
    assert logs.EndpointFilter({"/health"}).filter(no_args) is True
    assert logs.EndpointFilter({"/health"}).filter(short_args) is True


def test_setup_logging_adds_filter_once(fresh_setup):
    before = len(fresh_setup.filters)
    logs.setup_logging(excluded_paths={"/health"})
    logs.setup_logging(excluded_paths={"/health"})
    assert len(fresh_setup.filters) == before + 1


def test_setup_logging_empty_exclusions_adds_no_filter(fresh_setup):
    before = len(fresh_setup.filters)
    logs.setup_logging(excluded_paths=())
    assert len(fresh_setup.filters) == before


def test_setup_logging_suppresses_health_access_lines(fresh_setup, caplog):
    logs.setup_logging(excluded_paths={"/health"})
    with caplog.at_level(logging.INFO, logger="uvicorn.access"):
        fresh_setup.handle(_access_record("/health"))
        fresh_setup.handle(_access_record("/search?query=x"))
    assert [r.args[2] for r in caplog.records] == ["/search?query=x"]
