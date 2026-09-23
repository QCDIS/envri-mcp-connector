"""Match Argo institution/owner/data-center names against OSO Organization
individuals
"""
import argparse
import re
from collections import defaultdict
from functools import lru_cache

from kb_oso import transform as oso_transform

_MIN_MATCH_LEN = 4
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def normalize(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = _PUNCT_RE.sub(" ", name)
    return " ".join(name.split())


@lru_cache(maxsize=1)
def build_oso_org_index() -> dict:
    """normalized label -> oso_id, for every Organization individual's pref/alt labels."""
    index = {}
    for rec in oso_transform.iter_records():
        if "Organization" not in rec["entity_types"]:
            continue
        labels = [rec["pref_label"]] + rec["alt_labels"]
        for label in labels:
            norm = normalize(label)
            if len(norm) >= _MIN_MATCH_LEN:
                index.setdefault(norm, rec["oso_id"])
    return index


def _substring_match(norm_name: str, index: dict) -> str | None:
    for norm_label, oso_id in index.items():
        if len(norm_label) < _MIN_MATCH_LEN:
            continue
        if norm_label in norm_name or norm_name in norm_label:
            return oso_id
    return None


def match_organization(*candidates, index: dict = None) -> str | None:
    index = index if index is not None else build_oso_org_index()
    for candidate in candidates:
        norm = normalize(candidate)
        if len(norm) < _MIN_MATCH_LEN:
            continue
        if norm in index:
            return index[norm]
        hit = _substring_match(norm, index)
        if hit:
            return hit
    return None


def _report():
    from kb_argo import transform as argo_transform  # deferred: avoids a transform<->link_oso import cycle

    index = build_oso_org_index()
    print(f"OSO organization index: {len(index)} normalized labels\n")

    matches = defaultdict(set)
    total = 0
    for raw in argo_transform.iter_raw_records():
        total += 1
        data_center = raw.get("dataCenter") or {}
        institution = raw.get("institution") or {}
        candidates = [raw.get("owner"), data_center.get("name"), institution.get("name")]
        oso_id = match_organization(*candidates, index=index)
        if oso_id:
            for c in candidates:
                if c:
                    matches[(c, oso_id)].add(raw.get("wmo"))

    print(f"Checked {total} cached Argo records.\n")
    print(f"{len(matches)} distinct (Argo name -> OSO organization) pairs matched:\n")
    for (name, oso_id), wmos in sorted(matches.items(), key=lambda kv: -len(kv[1])):
        sample = ", ".join(list(wmos)[:5])
        print(f"  {name!r} -> oso:{oso_id}  ({len(wmos)} floats, e.g. {sample})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report Argo<->OSO organization matches for manual review")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    _report()
