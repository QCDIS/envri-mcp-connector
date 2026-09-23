import json

from kb_argo import transform


def test_join_filters_falsy_and_joins():
    assert transform._join(["a", None, "b", ""]) == "a, b"


def test_join_empty_list_returns_none():
    assert transform._join([]) is None
    assert transform._join(None) is None


def test_parse_date_valid_iso_string():
    dt = transform._parse_date("2020-01-15T00:00:00Z".replace("Z", "+00:00"))
    assert dt.year == 2020 and dt.month == 1 and dt.day == 15


def test_parse_date_invalid_returns_none():
    assert transform._parse_date("not-a-date") is None


def test_parse_date_empty_returns_none():
    assert transform._parse_date(None) is None
    assert transform._parse_date("") is None


def test_geopoint_valid_coordinates():
    assert transform._geopoint(45.0, -10.5) == {"lat": 45.0, "lon": -10.5}


def test_geopoint_none_coordinates():
    assert transform._geopoint(None, -10.5) is None
    assert transform._geopoint(45.0, None) is None


def test_geopoint_rejects_out_of_range_sentinel_values():
    # Argo uses sentinels like -99.999 for "no position".
    assert transform._geopoint(-99.999, -99.999) is None
    assert transform._geopoint(200.0, 0.0) is None


def test_resolve_sensors_uses_nerc_label_and_falls_back_to_code(monkeypatch):
    monkeypatch.setattr(
        transform.nerc_vocab,
        "get_label",
        lambda uri: "Temperature" if "TEMP" in uri else None,
    )
    sensors = [{"id": "TEMP"}, {"id": "UNKNOWN_CODE"}, {}]

    assert transform.resolve_sensors(sensors) == ["Temperature", "UNKNOWN_CODE"]


def test_resolve_sensors_deduplicates():
    sensors = [{"id": "TEMP"}, {"id": "TEMP"}]
    # No monkeypatch: nerc_vocab.get_label returns None (empty cache) -> falls back to raw id.
    assert transform.resolve_sensors(sensors) == ["TEMP"]


def _sample_raw_record(**overrides):
    raw = {
        "wmo": "1900001",
        "platform": {"type": "PROVOR", "name": "Provor"},
        "maker": "NKE",
        "model": "Provor Mk2",
        "owner": "Ifremer",
        "projectName": "Euro-Argo",
        "networks": ["Core"],
        "variables": ["TEMP", "PSAL"],
        "sensors": [{"id": "TEMP"}],
        "deployment": {
            "principalInvestigatorName": "Jane Doe",
            "launchDate": "2020-01-01",
            "platform": "R/V Thalassa",
            "lat": 42.0,
            "lon": 5.0,
        },
        "lastCycleBasicInfo": {
            "numCycle": 12,
            "date": "2020-06-01",
            "lat": 42.5,
            "lon": 5.5,
        },
        "dataCenter": {"name": "Coriolis", "code": "CO"},
        "institution": {"name": "Ifremer"},
        "battery": {"status": "OK"},
        "statusCode": "O",
        "cycleIds": list(range(12)),
    }
    raw.update(overrides)
    return raw


def test_build_summary_text_includes_key_facts():
    raw = _sample_raw_record()
    derived = {
        "sensor_names": ["Temperature"],
        "num_cycles": 12,
        "mission_duration_days": 152,
        "sea_area": "Mediterranean Sea",
        "ocean_region": "Atlantic Ocean",
    }
    summary = transform.build_summary_text(raw, derived)

    assert "Argo float 1900001" in summary
    assert "operated by Ifremer" in summary
    assert "principal investigator Jane Doe" in summary
    assert "Euro-Argo project" in summary
    assert "Temperature" in summary
    assert "deployed on 2020-01-01" in summary
    assert "Coriolis data center" in summary
    assert "Mediterranean Sea" in summary
    assert "Platform status: operational." in summary


def test_build_record_uses_mocked_seas_and_link_oso(monkeypatch):
    monkeypatch.setattr(transform.seas, "classify", lambda lat, lon: ("Mediterranean Sea", "Atlantic Ocean"))
    monkeypatch.setattr(transform.link_oso, "match_organization", lambda *candidates: "oso:ifremer")
    monkeypatch.setattr(transform.nerc_vocab, "get_label", lambda uri: None)

    record = transform.build_record(_sample_raw_record())

    assert record["_id"] == "euro_argo:1900001"
    assert record["source"] == "euro_argo"
    assert record["wmo"] == "1900001"
    assert record["sea_area"] == "Mediterranean Sea"
    assert record["ocean_region"] == "Atlantic Ocean"
    assert record["oso_organization_id"] == "oso:ifremer"
    assert record["num_cycles"] == 12
    assert record["mission_duration_days"] == 152
    assert record["last_cycle_geopoint"] == {"lat": 42.5, "lon": 5.5}
    assert record["has_quality_flags"] is False
    assert "summary_text" in record and record["summary_text"]


def test_build_record_flags_quality_issues(monkeypatch):
    monkeypatch.setattr(transform.seas, "classify", lambda lat, lon: (None, None))
    monkeypatch.setattr(transform.link_oso, "match_organization", lambda *candidates: None)
    monkeypatch.setattr(transform.nerc_vocab, "get_label", lambda uri: None)

    raw = _sample_raw_record(greyListParameters=["PSAL"])
    record = transform.build_record(raw)

    assert record["has_quality_flags"] is True


def test_iter_raw_records_reads_json_files_from_dir(tmp_path):
    (tmp_path / "1900001.json").write_text(json.dumps({"wmo": "1900001"}))
    (tmp_path / "1900002.json").write_text(json.dumps({"wmo": "1900002"}))

    records = sorted(transform.iter_raw_records(tmp_path), key=lambda r: r["wmo"])

    assert records == [{"wmo": "1900001"}, {"wmo": "1900002"}]
