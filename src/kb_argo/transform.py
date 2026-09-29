"""Turn a raw Euro-Argo float record into a KB document: structured metadata
plus a natural-language summary that gets embedded for semantic search.
"""
import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

from kb_argo import config, link_oso
from kb_common import nerc_vocab, seas

log = logging.getLogger(__name__)

SOURCE = "euro_argo"

_R03_BASE = "http://vocab.nerc.ac.uk/collection/R03/current/{}/"
_ADJUSTED_SUFFIX = "_ADJUSTED"
_R25_BASE = "http://vocab.nerc.ac.uk/collection/R25/current/{}/"
_C17_BASE = "http://vocab.nerc.ac.uk/collection/C17/current/{}/"
_C17_RE = re.compile(r"collection/C17/current/([^/\s]+)", re.IGNORECASE)
_URI_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def _clean(value):
    """Stripped string, or None for None / blank / non-string values."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _join(items, sep=", "):
    return sep.join(str(i) for i in items if i) if items else None


def _join_natural(items):
    """"a, b and c" - None for an empty list."""
    items = [str(i) for i in items or [] if i]
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    sep = "; " if any("," in i for i in items) else ", "
    return f"{sep.join(items[:-1])} and {items[-1]}"


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None

def _geopoint(lat, lon):
    """None unless lat/lon are valid - Argo uses sentinel values like -99.999
    for "no position", which a geo_point mapping rejects outright."""
    if lat is None or lon is None:
        return None
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None
    return {"lat": lat, "lon": lon}


def count_cycles(raw: dict):
    """Number of cycles the float has completed: length of `cycleIds`, else the
    last reported cycle number. 0 (or None if the record carries neither) means
    no cycle was ever recorded."""
    last_cycle = raw.get("lastCycleBasicInfo") or {}
    return len(raw.get("cycleIds") or []) or last_cycle.get("numCycle")


def is_placeholder(raw: dict) -> bool:
    """True for uninitialized/placeholder floats (e.g. WMO 9999995-9999998) that
    have never completed an ocean cycle. These records carry owner/data-center
    metadata (e.g. "Ifremer") but no real data, so they must not compete with
    real floats in search. Detected by cycle count == 0 rather than by WMO
    pattern, so it also catches placeholders outside the 99999xx range."""
    return not count_cycles(raw)

def resolve_parameter(code: str) -> str:
    """Readable name for an Argo parameter code from the NERC R03 vocabulary."""
    code = code.strip()
    base = code.upper()
    if base.endswith(_ADJUSTED_SUFFIX):
        base = base[:-len(_ADJUSTED_SUFFIX)]
    return nerc_vocab.get_label(_R03_BASE.format(base)) or code.replace("_", " ")

def resolve_sensors(sensors: list) -> list:
    """Return distinct readable sensor names: the NERC R25 label, else the R03
    parameter name, else the raw code with underscores spaced out."""
    resolved = []
    for sensor in sensors or []:
        sensor_id = _clean(sensor.get("id"))
        if not sensor_id:
            continue
        label = nerc_vocab.get_label(_R25_BASE.format(sensor_id)) or resolve_parameter(sensor_id)
        resolved.append(label)
    return list(dict.fromkeys(resolved))


def resolve_variables(variables: list) -> list:
    """Distinct readable names for raw parameter codes."""
    names = [resolve_parameter(v) for v in variables or [] if _clean(v)]
    return list(dict.fromkeys(names))


def resolve_ship(value) -> str | None:
    """Readable vessel name for `deployment.platform`.

    The API gives a NERC C17 (ICES platform codes) URI; those are looked up in
    the vocab cache. A URI we can't resolve yields None rather than leaking the
    raw URI into the summary. A plain name is returned as-is."""
    value = _clean(value)
    if not value:
        return None
    match = _C17_RE.search(value)
    if match:
        return nerc_vocab.get_label(_C17_BASE.format(match.group(1)))
    if _URI_RE.match(value):
        return nerc_vocab.get_label(value)
    return value


def build_summary_text(raw: dict, derived: dict) -> str:
    wmo = raw.get("wmo")
    platform = raw.get("platform") or {}
    deployment = raw.get("deployment") or {}
    last_cycle = raw.get("lastCycleBasicInfo") or {}
    data_center = raw.get("dataCenter") or {}
    battery = raw.get("battery") or {}

    parts = []

    ptype = platform.get("type") or platform.get("name")
    maker = raw.get("maker")
    model = raw.get("model")
    header = f"Argo float {wmo}"
    if ptype:
        header += f" is a {ptype} profiling float"
    if maker or model:
        header += f" ({_join([maker, model])})"
    parts.append(header + ".")

    owner = _clean(raw.get("owner"))
    pi = _clean(deployment.get("principalInvestigatorName"))
    project = _clean(raw.get("projectName"))
    if owner:
        line = f"It is operated by {owner}"
        if pi and pi != owner:
            line += f" (principal investigator {pi})"
        if project:
            line += f" under the {project} project"
        parts.append(line + ".")
    elif pi:
        line = f"Its principal investigator is {pi}"
        if project:
            line += f", within the {project} project"
        parts.append(line + ".")
    elif project:
        parts.append(f"It is part of the {project} project.")

    networks = [n for n in raw.get("networks") or [] if n]
    if networks:
        noun = "network" if len(networks) == 1 else "networks"
        parts.append(f"It belongs to the {_join_natural(networks)} {noun}.")

    variable_names = resolve_variables(raw.get("variables"))
    if variable_names:
        parts.append(f"It measures {_join_natural(variable_names)}.")

    sensor_names = derived["sensor_names"]
    if sensor_names:
        parts.append(f"It is equipped with sensors for: {_join(sensor_names)}.")

    launch_date = deployment.get("launchDate")
    ship = resolve_ship(deployment.get("platform"))
    dep_lat, dep_lon = deployment.get("lat"), deployment.get("lon")
    if launch_date:
        line = f"It was deployed on {launch_date}"
        if ship:
            line += f" from {ship}"
        if dep_lat is not None and dep_lon is not None:
            line += f" at latitude {dep_lat}, longitude {dep_lon}"
        parts.append(line + ".")

    dc_name = data_center.get("name") or data_center.get("code")
    if dc_name:
        parts.append(f"Its data is processed by the {dc_name} data center.")

    num_cycle = last_cycle.get("numCycle")
    lc_lat, lc_lon = last_cycle.get("lat"), last_cycle.get("lon")
    lc_date = last_cycle.get("date")
    if num_cycle is not None:
        line = f"Its last reported cycle is number {num_cycle}"
        if lc_date:
            line += f" on {lc_date}"
        if lc_lat is not None and lc_lon is not None:
            line += f", located at latitude {lc_lat}, longitude {lc_lon}"
            if derived["sea_area"]:
                line += f" in the {derived['sea_area']}"
                ocean = derived["ocean_region"]
                if ocean and ocean not in derived["sea_area"]:
                    line += f" ({ocean})"
        parts.append(line + ".")

    if derived["num_cycles"] and derived["mission_duration_days"]:
        parts.append(
            f"It has completed {derived['num_cycles']} cycles over roughly "
            f"{derived['mission_duration_days']} days of mission life."
        )

    bottom = last_cycle.get("bottomMeasure") or {}
    if bottom.get("temp") is not None or bottom.get("psal") is not None:
        line = "At its deepest measured point"
        if bottom.get("pres") is not None:
            line += f" ({bottom['pres']} dbar)"
        line += ", it recorded"
        if bottom.get("temp") is not None:
            line += f" temperature {bottom['temp']} degC"
        if bottom.get("psal") is not None:
            line += f" and salinity {bottom['psal']} psu"
        parts.append(line + ".")

    status = raw.get("statusCode")
    if status:
        status_map = {"O": "operational", "I": "inactive", "R": "regular", "A": "active"}
        parts.append(f"Platform status: {status_map.get(status, status)}.")

    quality_notes = []
    grey_list = raw.get("greyListParameters")
    if grey_list:
        quality_notes.append(f"parameters on the grey list: {_join(grey_list)}")
    battery_status = battery.get("status")
    if battery_status and battery_status.lower() != "ok":
        quality_notes.append(f"battery status: {battery_status}")
    if quality_notes:
        parts.append(f"Data quality note: {'; '.join(quality_notes)}.")

    return " ".join(parts)


def build_record(raw: dict) -> dict:
    wmo = raw.get("wmo")
    platform = raw.get("platform") or {}
    deployment = raw.get("deployment") or {}
    last_cycle = raw.get("lastCycleBasicInfo") or {}
    data_center = raw.get("dataCenter") or {}
    institution = raw.get("institution") or {}
    battery = raw.get("battery") or {}

    sensor_names = resolve_sensors(raw.get("sensors"))

    num_cycles = count_cycles(raw)
    launch_dt = _parse_date(deployment.get("launchDate"))
    last_dt = _parse_date(last_cycle.get("date"))
    mission_duration_days = (last_dt - launch_dt).days if launch_dt and last_dt else None

    sea_area, ocean_region = seas.classify(last_cycle.get("lat"), last_cycle.get("lon"))

    has_quality_flags = bool(raw.get("greyListParameters")) or (
        bool(battery.get("status")) and battery.get("status", "").lower() != "ok"
    )

    oso_organization_id = link_oso.match_organization(
        raw.get("owner"), data_center.get("name"), institution.get("name")
    )

    derived = {
        "sensor_names": sensor_names,
        "num_cycles": num_cycles,
        "mission_duration_days": mission_duration_days,
        "sea_area": sea_area,
        "ocean_region": ocean_region,
    }

    lc_lat, lc_lon = last_cycle.get("lat"), last_cycle.get("lon")
    last_cycle_geopoint = _geopoint(lc_lat, lc_lon)

    return {
        "_id": f"{SOURCE}:{wmo}",
        "source": SOURCE,
        "wmo": wmo,
        "platform_type": platform.get("type"),
        "platform_name": platform.get("name"),
        "maker": raw.get("maker"),
        "model": raw.get("model"),
        "owner": raw.get("owner"),
        "principal_investigator": deployment.get("principalInvestigatorName"),
        "project_name": raw.get("projectName"),
        "projects": raw.get("projects") or [],
        "networks": raw.get("networks") or [],
        "variables": raw.get("variables") or [],
        "sensor_codes": [s.get("id") for s in (raw.get("sensors") or []) if s.get("id")],
        "data_center_code": data_center.get("code"),
        "data_center_name": data_center.get("name"),
        "status_code": raw.get("statusCode"),
        "country_code": raw.get("countryCode"),
        "deployment_date": deployment.get("launchDate"),
        "deployment_lat": deployment.get("lat"),
        "deployment_lon": deployment.get("lon"),
        "deployment_ship": deployment.get("platform"),
        "last_cycle_number": last_cycle.get("numCycle"),
        "last_cycle_date": last_cycle.get("date"),
        "last_cycle_lat": last_cycle.get("lat"),
        "last_cycle_lon": last_cycle.get("lon"),
        "last_cycle_geopoint": last_cycle_geopoint,
        "num_cycles": num_cycles,
        "mission_duration_days": mission_duration_days,
        "sea_area": sea_area,
        "ocean_region": ocean_region,
        "has_quality_flags": has_quality_flags,
        "oso_organization_id": oso_organization_id,
        "summary_text": build_summary_text(raw, derived),
    }


def iter_raw_records(raw_dir=None) -> Iterator[dict]:
    raw_dir = raw_dir or config.ARGO_RAW_DIR
    for path in Path(raw_dir).glob("*.json"):
        yield json.loads(path.read_text())


def iter_records(raw_dir=None) -> Iterator[dict]:
    """Yield KB documents, skipping placeholder floats (cycle count 0)."""
    skipped = 0
    for raw in iter_raw_records(raw_dir):
        if is_placeholder(raw):
            skipped += 1
            log.debug("skipping placeholder float wmo=%s (0 cycles)", raw.get("wmo"))
            continue
        yield build_record(raw)
    if skipped:
        log.info("skipped %d placeholder floats (0 cycles)", skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview generated summaries")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    for i, rec in enumerate(iter_records()):
        if i >= args.limit:
            break
        print(json.dumps(rec, indent=2, ensure_ascii=False))
