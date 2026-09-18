"""OSO-specific fields layered on top of kb_common.es_index.BASE_PROPERTIES."""

def _text_and_keyword():
    return {"type": "text", "fields": {"keyword": {"type": "keyword"}}}


EXTRA_PROPERTIES = {
    "oso_id": {"type": "keyword"},
    # text+keyword: kb_mcp aggregates/filters on entity_types.keyword.
    "entity_types": _text_and_keyword(),
    "pref_label": {"type": "text"},
    "pref_label_en": {"type": "keyword"},
    "pref_label_fr": {"type": "keyword"},
    "alt_labels": {"type": "text"},
    "definition_en": {"type": "text"},
    "definition_fr": {"type": "text"},
    "external_ids": {"type": "flattened"},
}
