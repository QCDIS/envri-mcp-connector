"""OSO-specific fields layered on top of kb_common.es_index.BASE_PROPERTIES."""

from kb_common import es_index

_text_and_keyword = es_index.text_and_lowercase_keyword


EXTRA_PROPERTIES = {
    "oso_id": {"type": "keyword"},
    # text+keyword: kb_mcp aggregates/filters on entity_types.keyword
    # (case-insensitive via the lowercase normalizer).
    "entity_types": _text_and_keyword(),
    "pref_label": {"type": "text"},
    "pref_label_en": {"type": "keyword"},
    "pref_label_fr": {"type": "keyword"},
    "alt_labels": {"type": "text"},
    "definition_en": {"type": "text"},
    "definition_fr": {"type": "text"},
    "external_ids": {"type": "flattened"},
}
