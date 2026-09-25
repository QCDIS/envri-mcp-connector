"""Fetch the OSO ontology OWL file from its GitHub release.

Usage:
    python -m kb_oso.fetch
    python -m kb_oso.fetch --force
"""
import argparse
import logging

import requests

from kb_oso import config
from kb_common.logs import setup_logging

log = logging.getLogger(__name__)


def fetch_all(url: str = None, dest=None, force: bool = False) -> None:
    url = url or config.OSO_OWL_URL
    dest = dest or config.OSO_OWL_PATH

    if dest.exists() and not force:
        log.info("%s already exists, skipping (use --force to re-fetch)", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    log.info("fetched %s -> %s", url, dest)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Fetch the OSO ontology OWL file")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if already cached on disk")
    args = parser.parse_args()
    fetch_all(force=args.force)
