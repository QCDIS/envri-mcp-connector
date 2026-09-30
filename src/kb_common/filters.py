"""Elasticsearch `filter` clauses for the structured (non-free-text) criteria
shared by kb_api's hybrid search and kb_mcp's list_* tools.

Deliberately dependency-free (no embedding model, no ES client): the kb-mcp
image doesn't ship the ML stack that kb_common.hybrid_search pulls in.
"""
import calendar
import re
from datetime import date

# Keyword filters -> the indexed field each is enforced on. These fields carry
# the lowercase normalizer (see kb_common.es_index), so term queries take a
# lowercased value.
FILTER_FIELDS = {
    "sea_area": "sea_area.keyword",
    "ocean_basin": "ocean_region.keyword",
    "sensor": "sensor_codes",
}
STATUS_FIELD = "status_code"
LAST_CYCLE_DATE_FIELD = "last_cycle_date"
DEPLOYMENT_DATE_FIELD = "deployment_date"

# `status_code` is a plain (un-normalized) keyword holding the platform status
# as reported upstream. Friendly words map to the codes/labels that mean the
# same thing; anything not listed is matched as given. If the index uses other
# spellings, extend this table - list_argo_statuses shows what is really stored.
STATUS_SYNONYMS = {
    "active": ["A", "ACTIVE", "O", "OPERATIONAL"],
    "operational": ["O", "OPERATIONAL", "A", "ACTIVE"],
    "inactive": ["I", "INACTIVE"],
    "closed": ["C", "CLOSED"],
}

# YYYY, YYYY-MM or YYYY-MM-DD
_PARTIAL_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")
DATE_BOUND_PATTERN = r"^\d{4}(-\d{2}(-\d{2})?)?$"


def normalize(value: str) -> str:
    return value.strip().lower()


def status_values(status: str) -> list[str]:
    """Every stored spelling that `status` should match, in a stable order."""
    status = status.strip()
    candidates = STATUS_SYNONYMS.get(status.lower(), [status])
    values = []
    for candidate in candidates:
        values += [candidate, candidate.upper(), candidate.lower()]
    values += [status, status.upper(), status.lower()]
    return list(dict.fromkeys(values))


def _bound_text(value: date | str) -> str:
    text = value.isoformat() if isinstance(value, date) else str(value).strip()
    if not _PARTIAL_DATE_RE.match(text):
        raise ValueError(f"invalid date {text!r}: expected YYYY, YYYY-MM or YYYY-MM-DD")
    return text


def _period(text: str) -> tuple[date, date]:
    """First and last day covered by a (possibly partial) ISO date."""
    year, month, day = (int(g) if g else None for g in _PARTIAL_DATE_RE.match(text).groups())
    if month is None:
        return date(year, 1, 1), date(year, 12, 31)
    if day is None:
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    return date(year, month, day), date(year, month, day)


def date_range_clause(field: str, gte: date | str | None = None, lte: date | str | None = None) -> dict:
    """Inclusive range on `field`. Partial dates cover the whole period:
    gte="2020" starts 2020-01-01 and lte="2020" ends 2020-12-31 (Elasticsearch
    rounds a partial `lte` up), so a bare year selects that year."""
    bounds = {"format": "strict_date_optional_time"}
    low = _bound_text(gte) if gte else None
    high = _bound_text(lte) if lte else None
    if low and high and _period(low)[0] > _period(high)[1]:
        raise ValueError(f"start date {low} is after end date {high}")
    if low:
        bounds["gte"] = low
    if high:
        bounds["lte"] = high
    return {"range": {field: bounds}}


def build_filters(
    source: str | None = None,
    sea_area: str | None = None,
    ocean_basin: str | None = None,
    sensor: str | None = None,
    status: str | None = None,
    date_min: date | str | None = None,
    date_max: date | str | None = None,
    deployed_after: date | str | None = None,
    deployed_before: date | str | None = None,
) -> list[dict]:
    """Filter clauses (non-scoring, strictly enforced); unset arguments add
    nothing. `date_min`/`date_max` bound the last reported cycle date and
    `deployed_after`/`deployed_before` the deployment date; all bounds are
    inclusive."""
    filters = []
    if source:
        filters.append({"term": {"source": source}})
    for arg, value in (("sea_area", sea_area), ("ocean_basin", ocean_basin), ("sensor", sensor)):
        if value:
            filters.append({"term": {FILTER_FIELDS[arg]: normalize(value)}})
    if status:
        filters.append({"terms": {STATUS_FIELD: status_values(status)}})
    if date_min or date_max:
        filters.append(date_range_clause(LAST_CYCLE_DATE_FIELD, date_min, date_max))
    if deployed_after or deployed_before:
        filters.append(date_range_clause(DEPLOYMENT_DATE_FIELD, deployed_after, deployed_before))
    return filters
