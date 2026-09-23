from kb_api.format import _local_id, format_hit


def test_local_id_strips_source_prefix():
    assert _local_id("euro_argo:1900001") == "1900001"


def test_local_id_without_colon_returns_unchanged():
    assert _local_id("1900001") == "1900001"


def test_format_hit_euro_argo():
    hit = {"_id": "euro_argo:1900001", "score": 1.5, "source": "euro_argo", "summary_text": "A float."}
    result = format_hit(hit)

    assert result["url"] == "https://fleetmonitoring.euro-argo.eu/float/1900001"
    assert result["header"] == "Argo float 1900001"
    assert result["score"] == 1.5
    assert result["summary"] == "A float."


def test_format_hit_oso_uses_pref_label_as_header():
    hit = {"_id": "oso:some-org", "score": 2.0, "source": "oso", "pref_label": "Ifremer"}
    result = format_hit(hit)

    assert result["url"] == "https://w3id.org/earthsemantics/OSO#some-org"
    assert result["header"] == "Ifremer"


def test_format_hit_oso_falls_back_to_local_id_without_pref_label():
    hit = {"_id": "oso:some-org", "score": 2.0, "source": "oso"}
    result = format_hit(hit)

    assert result["header"] == "some-org"


def test_format_hit_unknown_source_has_no_url():
    hit = {"_id": "weird:123", "score": 1.0, "source": "weird"}
    result = format_hit(hit)

    assert result["url"] is None
    assert result["header"] == "123"


def test_format_hit_defaults_for_missing_optional_fields():
    hit = {"_id": "euro_argo:1900001", "score": 1.0, "source": "euro_argo"}
    result = format_hit(hit)

    assert result["vector"] is None
    assert result["summary"] is None
    assert result["highlight"] == []
    assert result["last_modified"] is None
