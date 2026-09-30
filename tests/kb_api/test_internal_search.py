import pytest
from pydantic import ValidationError

from kb_api import main


def test_internal_request_filters_default_to_none():
    req = main.InternalSearchRequest(query="x")
    assert req.sea_area is req.ocean_basin is req.sensor is req.status is None
    assert req.date_min is None and req.date_max is None


def test_internal_request_strips_filter_whitespace_and_rejects_blank():
    assert main.InternalSearchRequest(query="x", sea_area="  Black Sea ").sea_area == "Black Sea"
    with pytest.raises(ValidationError):
        main.InternalSearchRequest(query="x", sea_area="   ")


def test_internal_request_rejects_inverted_date_range():
    with pytest.raises(ValidationError):
        main.InternalSearchRequest(query="x", date_min="2024-06-01", date_max="2024-01-01")


def test_internal_search_passes_filters_to_hybrid_search(monkeypatch):
    captured = {}
    monkeypatch.setattr(main.hybrid_search, "search", lambda index, query, **kw: captured.update(kw) or [])

    req = main.InternalSearchRequest(
        query="x", sea_area="Sea of Japan", ocean_basin="Pacific Ocean", sensor="DOXY",
        status="O", date_min="2024-01-01", date_max="2024-12-31",
    )
    main.internal_search(req, _caller="test")

    assert captured["sea_area"] == "Sea of Japan"
    assert captured["ocean_basin"] == "Pacific Ocean"
    assert captured["sensor"] == "DOXY"
    assert captured["status"] == "O"
    assert captured["date_min"].isoformat() == "2024-01-01"
    assert captured["date_max"].isoformat() == "2024-12-31"
