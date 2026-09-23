"""Resolve NERC NVS (vocab.nerc.ac.uk) concept URIs to human-readable labels.

Usage (run locally, where the NVS turtles zip lives):
    PYTHONPATH=src python -m kb_common.nerc_vocab --zip /path/to/turtles.zip
"""
import argparse
import json
import logging
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import rdflib
from rdflib.namespace import SKOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_PATH = Path("data/cache/nerc_vocab_cache.json")

# Argo sensor `id` codes (e.g. CTD_TEMP) are SeaDataNet R25 device-catalogue
# concepts, needed regardless of what any particular OSO revision references.
BASE_COLLECTIONS = {"R25"}

_COLLECTION_RE = re.compile(r"vocab\.nerc\.ac\.uk/collection/([A-Za-z0-9]+)/")


def discover_needed_collections(owl_path: Path) -> set:
    collections = set(BASE_COLLECTIONS)
    if owl_path.exists():
        text = owl_path.read_text(errors="ignore")
        collections |= set(_COLLECTION_RE.findall(text))
    return collections


def _parse_collection(ttl_bytes: bytes) -> dict:
    g = rdflib.Graph()
    g.parse(data=ttl_bytes, format="turtle")
    entries = {}
    for concept in g.subjects(rdflib.RDF.type, SKOS.Concept):
        pref_label = None
        for label in g.objects(concept, SKOS.prefLabel):
            if getattr(label, "language", None) == "en" or pref_label is None:
                pref_label = str(label)
            if getattr(label, "language", None) == "en":
                break
        definition = None
        for d in g.objects(concept, SKOS.definition):
            if getattr(d, "language", None) == "en" or definition is None:
                definition = str(d)
            if getattr(d, "language", None) == "en":
                break
        if pref_label:
            entries[str(concept)] = {"prefLabel": pref_label, "definition": definition}
    return entries


def build_cache(zip_path: Path, owl_path: Path = None, out_path: Path = CACHE_PATH) -> dict:
    owl_path = owl_path or Path("data/cache/oso/oso.owl")
    needed = discover_needed_collections(owl_path)
    log.info("resolving NERC collections: %s", sorted(needed))

    entries = {}
    with zipfile.ZipFile(zip_path) as zf:
        available = set(zf.namelist())
        for code in sorted(needed):
            member = f"{code}.ttl"
            if member not in available:
                log.warning("collection %s not found in zip, skipping", code)
                continue
            data = zf.read(member)
            collection_entries = _parse_collection(data)
            log.info("%s: %d concepts", code, len(collection_entries))
            entries.update(collection_entries)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, ensure_ascii=False))
    log.info("wrote %d total concepts to %s", len(entries), out_path)
    return entries


@lru_cache(maxsize=1)
def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text())


def get_label(uri: str) -> str | None:
    entry = _load_cache().get(uri)
    return entry["prefLabel"] if entry else None


def get_definition(uri: str) -> str | None:
    entry = _load_cache().get(uri)
    return entry.get("definition") if entry else None


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=os.environ.get("NERC_TURTLES_ZIP"), help="Path to the NVS turtles.zip (default: $NERC_TURTLES_ZIP)")
    parser.add_argument("--owl", default="data/cache/oso/oso.owl", help="OSO owl file to scan for referenced collections")
    parser.add_argument("--out", default=str(CACHE_PATH))
    args = parser.parse_args()
    if not args.zip:
        parser.error("--zip is required (or set NERC_TURTLES_ZIP)")
    build_cache(Path(args.zip), Path(args.owl), Path(args.out))
