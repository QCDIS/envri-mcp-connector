"""Turn a raw Euro-Argo float record into a KB document: structured metadata
plus a natural-language summary that gets embedded for semantic search.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from kb_argo import config, link_oso
from kb_common import nerc_vocab, seas

SOURCE = "euro_argo"

_R25_BASE = "http://vocab.nerc.ac.uk/collection/R25/current/{}/"


def _join(items, sep=", "):
    return sep.join(str(i) for i in items if i) if items else None


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


def resolve_sensors(sensors: list) -> list:
    """Return distinct resolved sensor names (falls back to the raw code)."""
    resolved = []
    for sensor in sensors or []:
        sensor_id = sensor.get("id")
        if not sensor_id:
            continue
        label = nerc_vocab.get_label(_R25_BASE.format(sensor_id)) or sensor_id
        resolved.append(label)
    return list(dict.fromkeys(resolved))


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

    owner = raw.get("owner")
    pi = deployment.get("principalInvestigatorName")
    project = raw.get("projectName")
    if owner or pi or project:
        line = "It is operated"
        if owner:
            line += f" by {owner}"
        if pi and pi != owner:
            line += f", principal investigator {pi},"
        if project:
            line += f" under the {project} project"
        parts.append(line.strip().rstrip(",") + ".")

    networks = _join(raw.get("networks"))
    variables = _join(raw.get("variables"))
    if networks or variables:
        line = "It belongs to the"
        if networks:
            line += f" {networks} network(s)"
        if variables:
            line += f" and measures {variables.lower()}"
        parts.append(line + ".")

    sensor_names = derived["sensor_names"]
    if sensor_names:
        parts.append(f"It is equipped with sensors for: {_join(sensor_names)}.")

    launch_date = deployment.get("launchDate")
    ship = deployment.get("platform")
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

    num_cycles = len(raw.get("cycleIds") or []) or last_cycle.get("numCycle")
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
    for raw in iter_raw_records(raw_dir):
        yield build_record(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview generated summaries")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    for i, rec in enumerate(iter_records()):
        if i >= args.limit:
            break
        print(json.dumps(rec, indent=2, ensure_ascii=False))
