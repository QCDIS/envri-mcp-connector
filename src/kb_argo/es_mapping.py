"""Argo-specific fields layered on top of kb_common.es_index.BASE_PROPERTIES."""

# text + keyword multi-field: `text` (analyzed, case-insensitive) lets a
# multi_match query score directly on "this float's owner/PI/project IS X"
# instead of relying on X appearing once inside the much longer summary_text
# (where BM25's length normalization buries it); `.keyword` is kept for exact
# filtering/aggregations.
def _text_and_keyword():
    return {"type": "text", "fields": {"keyword": {"type": "keyword"}}}


EXTRA_PROPERTIES = {
    "wmo": {"type": "keyword"},
    "platform_type": {"type": "keyword"},
    "platform_name": {"type": "text"},
    "maker": {"type": "keyword"},
    "model": {"type": "keyword"},
    "owner": _text_and_keyword(),
    "principal_investigator": _text_and_keyword(),
    "project_name": _text_and_keyword(),
    "projects": {"type": "keyword"},
    "networks": {"type": "keyword"},
    "variables": {"type": "keyword"},
    "data_center_code": {"type": "keyword"},
    "data_center_name": _text_and_keyword(),
    "status_code": {"type": "keyword"},
    "country_code": {"type": "keyword"},
    "deployment_date": {"type": "date"},
    "deployment_lat": {"type": "float"},
    "deployment_lon": {"type": "float"},
    "deployment_ship": {"type": "keyword"},
    "last_cycle_number": {"type": "integer"},
    "last_cycle_date": {"type": "date"},
    "last_cycle_lat": {"type": "float"},
    "last_cycle_lon": {"type": "float"},
    "last_cycle_geopoint": {"type": "geo_point"},
    "sensor_codes": {"type": "keyword"},
    "num_cycles": {"type": "integer"},
    "mission_duration_days": {"type": "integer"},
    "sea_area": _text_and_keyword(),
    "ocean_region": _text_and_keyword(),
    "has_quality_flags": {"type": "boolean"},
    "oso_organization_id": {"type": "keyword"},
}
