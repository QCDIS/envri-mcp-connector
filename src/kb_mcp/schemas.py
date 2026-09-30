"""Typed result models for the MCP tools.

The MCP SDK publishes each tool's `outputSchema` from its return annotation and
validates every result against it before it goes on the wire. The models are
deliberately tolerant - every field is optional and unknown fields are kept
(`extra="allow"`) - so an index document with an unexpected value can't turn a
successful lookup into a validation error, while the fields clients rely on
still have a declared name and type.

Documents come back with an `_id` key; pydantic disallows leading-underscore
field names, so it is declared as `id` with the alias `_id` (the SDK
serializes by alias, so the wire format is unchanged).
"""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


DocId = Annotated[str, Field(alias="_id", description='Document id, "<source>:<local id>" (e.g. "euro_argo:6902919").')]
Url = Annotated[str | None, Field(description="Dereferenceable link to the original data source.")]


class KbHit(_Model):
    """A search / list hit: identity, source link and the embedded summary."""

    id: DocId
    score: Annotated[float | None, Field(description="Relevance score (absent/null for exact-filter listings).")] = None
    url: Url = None
    source: Annotated[str | None, Field(description='"euro_argo" or "oso".')] = None
    summary_text: Annotated[str | None, Field(description="Natural-language summary of the record.")] = None


class SearchHit(KbHit):
    """A search_knowledge_base hit: a KbHit plus the matched summary fragments."""

    highlights: Annotated[
        list[str], Field(description="Matched fragments of summary_text, wrapped in <em>.")
    ] = []


class NearbyFloat(_Model):
    """A float returned by find_argo_floats_near, with its distance from the query point."""

    id: DocId
    distance_km: Annotated[float, Field(description="Distance from the query point, in km.")]
    url: Url = None
    source: str | None = None
    summary_text: str | None = None


class FloatListing(_Model):
    """A float returned by list_argo_floats_by_status_and_deployment."""

    id: DocId
    url: Url = None
    source: str | None = None
    summary_text: str | None = None
    status_code: Annotated[str | None, Field(description="Platform status as stored upstream.")] = None
    deployment_date: Annotated[str | None, Field(description="ISO deployment date/time.")] = None


class FacetValue(_Model):
    """One distinct field value and how many documents carry it."""

    value: Annotated[str | int | float | bool, Field(description="The value, in its original casing where known.")]
    count: Annotated[int, Field(ge=0)]


class IndexStats(_Model):
    total: Annotated[int, Field(ge=0, description="Total documents in the knowledge base.")]
    by_source: Annotated[dict[str, int], Field(description='Document count per source ("euro_argo", "oso").')]


class GeoPoint(_Model):
    lat: float
    lon: float


class ArgoFloatRecord(_Model):
    """Full Euro-Argo float document."""

    id: DocId
    url: Url = None
    source: str | None = None
    wmo: str | int | None = None
    platform_type: str | None = None
    platform_name: str | None = None
    maker: str | None = None
    model: str | None = None
    owner: str | None = None
    principal_investigator: str | None = None
    project_name: str | None = None
    projects: list[str] | None = None
    networks: list[str] | None = None
    variables: list[str] | None = None
    sensor_codes: list[str] | None = None
    data_center_code: str | None = None
    data_center_name: str | None = None
    status_code: str | None = None
    country_code: str | None = None
    deployment_date: str | None = None
    deployment_lat: float | None = None
    deployment_lon: float | None = None
    deployment_ship: str | None = None
    last_cycle_number: int | None = None
    last_cycle_date: str | None = None
    last_cycle_lat: float | None = None
    last_cycle_lon: float | None = None
    last_cycle_geopoint: GeoPoint | None = None
    num_cycles: int | None = None
    mission_duration_days: int | None = None
    sea_area: str | None = None
    ocean_region: str | None = None
    has_quality_flags: bool | None = None
    oso_organization_id: str | None = None
    summary_text: str | None = None
    indexed_at: str | None = None


class OsoEntityRecord(_Model):
    """Full OSO ontology entity document."""

    id: DocId
    url: Url = None
    source: str | None = None
    oso_id: str | None = None
    entity_types: list[str] | None = None
    pref_label: str | None = None
    pref_label_en: str | None = None
    pref_label_fr: str | None = None
    alt_labels: list[str] | None = None
    definition_en: str | None = None
    definition_fr: str | None = None
    external_ids: dict[str, str | list[str]] | None = None
    summary_text: str | None = None
    indexed_at: str | None = None
