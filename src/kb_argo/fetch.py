"""Fetch Argo float records from the Euro-Argo Fleet Monitoring API.

API confirmed live at https://fleetmonitoring.euro-argo.eu:
  GET /platformCodes      -> list of all WMO codes (~21k)
  GET /floats/{wmo}       -> full float metadata (matches example/argo.json shape)
"""
import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from kb_argo import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

session = requests.Session()


def get_platform_codes() -> list[str]:
    resp = session.get(f"{config.ARGO_API_BASE}/platformCodes", timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_float(wmo: str) -> dict:
    resp = session.get(f"{config.ARGO_API_BASE}/floats/{wmo}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_one(wmo: str, raw_dir, force: bool) -> str | None:
    dest = raw_dir / f"{wmo}.json"
    if dest.exists() and not force:
        return None
    try:
        data = get_float(wmo)
    except requests.RequestException as exc:
        log.warning("failed to fetch wmo=%s: %s", wmo, exc)
        return wmo
    dest.write_text(json.dumps(data))
    return None


def fetch_all(raw_dir=None, workers=None, limit: int | None = None, force: bool = False):
    raw_dir = raw_dir or config.ARGO_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or config.ARGO_FETCH_WORKERS

    codes = get_platform_codes()
    if limit:
        codes = codes[:limit]
    log.info("fetching %d floats with %d workers", len(codes), workers)

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, wmo, raw_dir, force): wmo for wmo in codes}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            wmo = fut.result()
            if wmo:
                failed.append(wmo)

    if failed:
        log.warning("%d floats failed to fetch: %s", len(failed), failed[:20])
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Argo float records from Euro-Argo API")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N floats (testing)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached on disk")
    args = parser.parse_args()
    fetch_all(limit=args.limit, force=args.force)
