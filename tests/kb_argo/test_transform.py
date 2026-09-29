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

    assert transform.resolve_sensors(sensors) == ["Temperature", "UNKNOWN CODE"]


def test_resolve_sensors_deduplicates(monkeypatch):
    monkeypatch.setattr(transform.nerc_vocab, "get_label", lambda uri: None)
    sensors = [{"id": "CTD_TEMP"}, {"id": "CTD_TEMP"}]

    assert transform.resolve_sensors(sensors) == ["CTD TEMP"]


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


def test_count_cycles_prefers_cycle_ids_then_last_cycle_number():
    assert transform.count_cycles({"cycleIds": [1, 2, 3]}) == 3
    assert transform.count_cycles({"lastCycleBasicInfo": {"numCycle": 7}}) == 7
    assert transform.count_cycles({}) is None


def test_is_placeholder_true_when_no_cycles():
    assert transform.is_placeholder({"wmo": "9999995", "cycleIds": []}) is True
    assert transform.is_placeholder({"wmo": "9999996", "lastCycleBasicInfo": {"numCycle": 0}}) is True
    assert transform.is_placeholder({"wmo": "9999997"}) is True


def test_is_placeholder_false_for_real_float():
    assert transform.is_placeholder(_sample_raw_record()) is False


def _stub_enrichment(monkeypatch):
    monkeypatch.setattr(transform.seas, "classify", lambda lat, lon: (None, None))
    monkeypatch.setattr(transform.link_oso, "match_organization", lambda *candidates: None)
    monkeypatch.setattr(transform.nerc_vocab, "get_label", lambda uri: None)


def test_iter_records_skips_placeholders(tmp_path, monkeypatch):
    _stub_enrichment(monkeypatch)
    (tmp_path / "1900001.json").write_text(json.dumps(_sample_raw_record()))
    (tmp_path / "9999995.json").write_text(
        json.dumps(_sample_raw_record(wmo="9999995", cycleIds=[], lastCycleBasicInfo={"numCycle": 0}))
    )

    wmos = [r["wmo"] for r in transform.iter_records(tmp_path)]

    assert wmos == ["1900001"]


def _stub_vocab(monkeypatch, labels=None):
    labels = labels or {}
    monkeypatch.setattr(transform.nerc_vocab, "get_label", lambda uri: labels.get(uri))


_R03 = "http://vocab.nerc.ac.uk/collection/R03/current/{}/"
_R03_LABELS = {
    _R03.format("PRES"): "Sea water pressure, equals 0 at sea-level",
    _R03.format("TEMP"): "Sea temperature in-situ ITS-90 scale",
    _R03.format("PSAL"): "Practical salinity",
    _R03.format("DOXY"): "Dissolved oxygen",
}


def test_resolve_parameter_uses_r03_label(monkeypatch):
    _stub_vocab(monkeypatch, _R03_LABELS)

    assert transform.resolve_parameter("DOXY") == "Dissolved oxygen"
    assert transform.resolve_parameter(" psal ") == "Practical salinity"


def test_resolve_parameter_adjusted_uses_base_name(monkeypatch):
    _stub_vocab(monkeypatch, _R03_LABELS)

    assert transform.resolve_parameter("PSAL_ADJUSTED") == "Practical salinity"


def test_resolve_parameter_unknown_code_is_spaced_not_dropped(monkeypatch):
    _stub_vocab(monkeypatch)

    assert transform.resolve_parameter("NEW_PARAM") == "NEW PARAM"


def test_resolve_variables_spells_out_parameter_codes(monkeypatch):
    _stub_vocab(monkeypatch, _R03_LABELS)

    assert transform.resolve_variables(["PRES", "TEMP", "PSAL", "DOXY"]) == [
        "Sea water pressure, equals 0 at sea-level",
        "Sea temperature in-situ ITS-90 scale",
        "Practical salinity",
        "Dissolved oxygen",
    ]


def test_resolve_variables_deduplicates_and_skips_blank(monkeypatch):
    _stub_vocab(monkeypatch, _R03_LABELS)

    assert transform.resolve_variables(["TEMP", "TEMP", "", None, "TEMP_ADJUSTED"]) == [
        "Sea temperature in-situ ITS-90 scale"
    ]


def test_join_natural_uses_semicolons_when_items_contain_commas():
    assert transform._join_natural(["a", "b", "c"]) == "a, b and c"
    assert transform._join_natural(["a, x", "b"]) == "a, x and b"
    assert transform._join_natural(["a, x", "b", "c"]) == "a, x; b and c"
    assert transform._join_natural([]) is None


def test_resolve_ship_looks_up_c17_uri(monkeypatch):
    uri = "http://vocab.nerc.ac.uk/collection/C17/current/35TH/"
    _stub_vocab(monkeypatch, {uri: "Thalassa"})

    assert transform.resolve_ship(uri) == "Thalassa"
    # https / missing trailing slash variants resolve to the same concept
    assert transform.resolve_ship("https://vocab.nerc.ac.uk/collection/C17/current/35TH") == "Thalassa"


def test_resolve_ship_unresolvable_uri_is_dropped_not_leaked(monkeypatch):
    _stub_vocab(monkeypatch)

    assert transform.resolve_ship("http://vocab.nerc.ac.uk/collection/C17/current/ZZZZ/") is None
    assert transform.resolve_ship("https://example.org/ships/42") is None


def test_resolve_ship_plain_name_and_blank(monkeypatch):
    _stub_vocab(monkeypatch)

    assert transform.resolve_ship("  R/V Thalassa ") == "R/V Thalassa"
    assert transform.resolve_ship("   ") is None
    assert transform.resolve_ship(None) is None


def _summary(monkeypatch, raw, labels=None):
    _stub_vocab(monkeypatch, labels)
    derived = {"sensor_names": [], "num_cycles": None, "mission_duration_days": None, "sea_area": None, "ocean_region": None}
    return transform.build_summary_text(raw, derived)


def test_summary_uses_readable_variables_and_ship(monkeypatch):
    uri = "http://vocab.nerc.ac.uk/collection/C17/current/35TH/"
    raw = _sample_raw_record(variables=["PRES", "TEMP", "PSAL", "DOXY"])
    raw["deployment"]["platform"] = uri

    summary = _summary(monkeypatch, raw, {uri: "Thalassa", **_R03_LABELS})

    assert "Practical salinity" in summary and "Dissolved oxygen" in summary
    assert "from Thalassa" in summary
    for raw_token in ("PRES", "PSAL", "DOXY", "http"):
        assert raw_token not in summary


def test_summary_omits_unresolvable_ship_uri(monkeypatch):
    raw = _sample_raw_record()
    raw["deployment"]["platform"] = "http://vocab.nerc.ac.uk/collection/C17/current/ZZZZ/"

    summary = _summary(monkeypatch, raw)

    assert "deployed on 2020-01-01" in summary
    assert "http" not in summary and " from " not in summary


def test_summary_without_owner_but_with_pi_and_project(monkeypatch):
    summary = _summary(monkeypatch, _sample_raw_record(owner=None))

    assert "Its principal investigator is Jane Doe, within the Euro-Argo project." in summary
    assert "operated" not in summary


def test_summary_without_owner_or_pi(monkeypatch):
    raw = _sample_raw_record(owner="  ")
    raw["deployment"]["principalInvestigatorName"] = None

    summary = _summary(monkeypatch, raw)

    assert "It is part of the Euro-Argo project." in summary
    assert "operated" not in summary and ", ." not in summary


def test_summary_without_owner_pi_or_project(monkeypatch):
    raw = _sample_raw_record(owner=None, projectName=None)
    raw["deployment"]["principalInvestigatorName"] = None

    summary = _summary(monkeypatch, raw)

    assert "operated" not in summary and "project" not in summary


def test_summary_variables_without_networks_has_no_dangling_phrase(monkeypatch):
    summary = _summary(monkeypatch, _sample_raw_record(networks=[], variables=["DOXY"]), _R03_LABELS)

    assert "It measures Dissolved oxygen." in summary
    assert "belongs to the" not in summary


def test_summary_networks_singular_and_plural(monkeypatch):
    assert "belongs to the Core network." in _summary(monkeypatch, _sample_raw_record(networks=["Core"]))
    assert "belongs to the Core and BGC networks." in _summary(
        monkeypatch, _sample_raw_record(networks=["Core", "BGC"])
    )
