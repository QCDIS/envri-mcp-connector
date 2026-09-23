from kb_oso.es_mapping import EXTRA_PROPERTIES


def test_every_property_has_a_type():
    for field, mapping in EXTRA_PROPERTIES.items():
        assert "type" in mapping, f"{field} mapping is missing 'type'"


def test_entity_types_is_text_and_keyword():
    mapping = EXTRA_PROPERTIES["entity_types"]
    assert mapping["type"] == "text"
    assert mapping["fields"]["keyword"]["type"] == "keyword"


def test_external_ids_is_flattened():
    assert EXTRA_PROPERTIES["external_ids"] == {"type": "flattened"}
