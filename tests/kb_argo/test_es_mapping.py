from kb_argo.es_mapping import EXTRA_PROPERTIES


def test_every_property_has_a_type():
    for field, mapping in EXTRA_PROPERTIES.items():
        assert "type" in mapping, f"{field} mapping is missing 'type'"


def test_text_and_keyword_fields_have_keyword_subfield():
    text_and_keyword_fields = ["owner", "principal_investigator", "project_name", "data_center_name", "sea_area", "ocean_region"]
    for field in text_and_keyword_fields:
        mapping = EXTRA_PROPERTIES[field]
        assert mapping["type"] == "text"
        assert mapping["fields"]["keyword"]["type"] == "keyword"


def test_geopoint_field():
    assert EXTRA_PROPERTIES["last_cycle_geopoint"] == {"type": "geo_point"}
